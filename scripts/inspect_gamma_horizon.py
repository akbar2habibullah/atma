#!/usr/bin/env python3
"""Inspect the learned Titans/GDN retention horizons in an ATMA checkpoint.

This script is intentionally runnable as a single Colab command after cloning the
ATMA repository.  It downloads the public 10B-token ATMA-Raven-Titans checkpoint by
default, runs a small streaming text sample through the model, captures

    gamma_t = sigmoid(w_gamma(x_t) + mem_gamma_bias),

and converts the observed retention into decay-only half-lives and 1/e horizons.
It never materializes vocabulary logits, so the default run fits comfortably on a
Colab T4.  FLA is optional; without it the repository's PyTorch fallbacks are used.

Examples
--------
    python scripts/inspect_gamma_horizon.py
    python scripts/inspect_gamma_horizon.py --num-sequences 4 --seq-len 256
    python scripts/inspect_gamma_horizon.py --checkpoint /content/checkpoint
    python scripts/inspect_gamma_horizon.py --text-file /content/probe.txt

Colab setup
-----------
    !git clone https://github.com/akbar2habibullah/atma.git
    %cd atma
    !pip -q install huggingface_hub transformers datasets matplotlib
    !python scripts/inspect_gamma_horizon.py

Interpretation
--------------
For a varying gate, the typical retention after L tokens is

    exp(L * mean(log(gamma_t))).

The reported half-life solves this expression for 0.5, and the e-fold horizon
solves it for 1/e.  These describe explicit gamma decay only.  The delta update
also overwrites the matrix along incoming key directions, so actual associative
recall can disappear sooner (or behave differently under structured inputs).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_REPO = (
    "ChavyvAkvar/"
    "atma-10b-L40S-mbs16-atma-raven-titans__reg-baseline__distr-0__mem-1__win-0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure activation-conditioned gamma horizons in an ATMA checkpoint."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="Local checkpoint directory or weights.pt path. Overrides --repo-id.",
    )
    source.add_argument(
        "--repo-id",
        default=DEFAULT_REPO,
        help=f"Hugging Face model repository (default: {DEFAULT_REPO}).",
    )
    parser.add_argument(
        "--revision",
        default="8427bfefc77ed199e912d31c80fd6ff0ea179876",
        help="Checkpoint revision/commit to download (default: archived benchmark revision).",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-key", default="text")
    parser.add_argument(
        "--text-file",
        type=Path,
        help="Use this UTF-8 text instead of streaming --dataset.",
    )
    parser.add_argument("--tokenizer", default=None, help="Override the checkpoint tokenizer.")
    parser.add_argument("--num-sequences", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="auto uses float16 on CUDA (including T4) and float32 on CPU.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("gamma_horizon_report"))
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.num_sequences < 1 or args.seq_len < 2:
        parser.error("--num-sequences must be >=1 and --seq-len must be >=2")
    return args


def resolve_checkpoint(args: argparse.Namespace) -> tuple[Path, Path, str]:
    """Return weights path, config path, and a human-readable source."""
    if args.checkpoint is not None:
        candidate = args.checkpoint.expanduser().resolve()
        if candidate.is_dir():
            weights = candidate / "weights.pt"
            config = candidate / "config.json"
        else:
            weights = candidate
            config = candidate.parent / "config.json"
        missing = [str(p) for p in (weights, config) if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing checkpoint file(s): {missing}")
        return weights, config, str(candidate)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install checkpoint support with: pip install huggingface_hub") from exc

    print(f"[download] {args.repo_id}@{args.revision}", flush=True)
    snapshot = Path(
        snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            allow_patterns=["weights.pt", "config.json", "tokenizer.json"],
        )
    )
    return snapshot / "weights.pt", snapshot / "config.json", f"{args.repo_id}@{args.revision}"


def choose_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        # T4 has no native BF16 tensor cores; FP16 is both supported and much faster.
        return torch.float16 if device.type == "cuda" else torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_model(weights_path: Path, config_path: Path, device: torch.device, dtype: torch.dtype):
    from raven_baseline.model import create_model

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("arch_type") != "atma_raven_titans":
        raise ValueError(
            "This probe expects arch_type='atma_raven_titans'; got "
            f"{config.get('arch_type')!r}."
        )
    model = create_model(config)
    print(f"[load] reading {weights_path} on CPU", flush=True)
    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = payload.get("model", payload)
    state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch:\n"
            f"  missing={result.missing_keys}\n  unexpected={result.unexpected_keys}"
        )
    del payload, state
    model.to(device=device, dtype=dtype).eval()
    return model, config


def tokenizer_name(config_path: Path, override: str | None) -> str:
    if override:
        return override
    token_meta = config_path.parent / "tokenizer.json"
    if token_meta.is_file():
        try:
            return json.loads(token_meta.read_text(encoding="utf-8"))["tokenizer_name"]
        except (KeyError, TypeError, json.JSONDecodeError):
            pass
    return "gpt2"


def packed_sequences(args: argparse.Namespace, tokenizer) -> Iterator[torch.Tensor]:
    """Pack text examples into fixed-length token sequences without padding."""
    eos = tokenizer.eos_token_id
    eos = [] if eos is None else [int(eos)]
    buffer: list[int] = []

    if args.text_file is not None:
        texts = [args.text_file.expanduser().read_text(encoding="utf-8")]
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit("Install corpus support with: pip install datasets") from exc
        print(
            f"[data] streaming {args.dataset}/{args.dataset_config} [{args.dataset_split}]",
            flush=True,
        )
        dataset = load_dataset(
            args.dataset,
            args.dataset_config or None,
            split=args.dataset_split,
            streaming=True,
        )
        dataset = dataset.shuffle(seed=args.seed, buffer_size=1_000)
        texts = (row.get(args.text_key, "") for row in dataset)

    produced = 0
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        buffer.extend(eos)
        while len(buffer) >= args.seq_len:
            yield torch.tensor(buffer[: args.seq_len], dtype=torch.long)
            del buffer[: args.seq_len]
            produced += 1
            if produced >= args.num_sequences:
                return
    raise RuntimeError(
        f"The text source produced only {produced}/{args.num_sequences} full sequences "
        f"of {args.seq_len} tokens. Use more text or smaller probe settings."
    )


class GammaCapture:
    """Capture the exact input-conditioned scalar gamma for every memory head."""

    def __init__(self, model):
        self.by_layer: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.static_by_layer: dict[int, torch.Tensor] = {}
        self._current: dict[int, torch.Tensor] = {}
        self._handles = []
        for layer, block in enumerate(model.blocks):
            memory = getattr(block.attn, "mem", None)
            if memory is None:
                continue
            bias = float(memory.gamma_bias)
            if memory.w_gamma.bias is None:
                static_logit = torch.full((memory.H,), bias)
            else:
                static_logit = memory.w_gamma.bias.detach().float().cpu() + bias
            self.static_by_layer[layer] = torch.sigmoid(static_logit)

            def hook(_module, _inputs, output, *, layer=layer, bias=bias):
                # output is (B,T,H), before the fixed gamma_bias is added.
                self._current[layer] = torch.sigmoid(output.detach().float() + bias).cpu()

            self._handles.append(memory.w_gamma.register_forward_hook(hook))
        if not self._handles:
            raise RuntimeError("No TitansMemory.w_gamma modules were found in this checkpoint.")

    def finish_sequence(self):
        expected = set(self.static_by_layer)
        if set(self._current) != expected:
            raise RuntimeError(
                f"Gamma hooks did not all fire: expected layers {sorted(expected)}, "
                f"observed {sorted(self._current)}"
            )
        for layer, gamma in self._current.items():
            self.by_layer[layer].append(gamma.squeeze(0))  # (T,H)
        self._current.clear()

    def close(self):
        for handle in self._handles:
            handle.remove()


@torch.inference_mode()
def collect_gamma(model, sequences: Iterator[torch.Tensor], capture: GammaCapture,
                  num_sequences: int, device: torch.device):
    for index, tokens in enumerate(sequences, 1):
        inputs = tokens.unsqueeze(0).to(device, non_blocking=True)
        x = model.embed(inputs)
        for block in model.blocks:
            x, _, _ = block(x)
        capture.finish_sequence()
        print(f"[probe] sequence {index}/{num_sequences}", flush=True)
        del inputs, x


def safe_horizon(log_gamma: float, target_log_retention: float) -> float:
    if log_gamma >= 0.0:
        return math.inf
    return target_log_retention / log_gamma


def percentile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float(), q).item())


def summarize(capture: GammaCapture, seq_len: int) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for layer in sorted(capture.by_layer):
        values = torch.cat(capture.by_layer[layer], dim=0).clamp_(1e-12, 1.0)  # (tokens,H)
        static = capture.static_by_layer[layer]
        for head in range(values.shape[1]):
            g = values[:, head]
            mean_log = float(g.log().mean().item())
            mean_gamma = float(g.mean().item())
            static_gamma = float(static[head].item())
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "samples": int(g.numel()),
                    "static_gamma": static_gamma,
                    "static_half_life": safe_horizon(math.log(max(static_gamma, 1e-12)), math.log(0.5)),
                    "mean_gamma": mean_gamma,
                    "geometric_mean_gamma": math.exp(mean_log),
                    "gamma_p01": percentile(g, 0.01),
                    "gamma_p10": percentile(g, 0.10),
                    "gamma_p50": percentile(g, 0.50),
                    "gamma_p90": percentile(g, 0.90),
                    "gamma_p99": percentile(g, 0.99),
                    "half_life_tokens": safe_horizon(mean_log, math.log(0.5)),
                    "efold_tokens": safe_horizon(mean_log, -1.0),
                    "typical_retention_at_32": math.exp(32 * mean_log),
                    "typical_retention_at_128": math.exp(128 * mean_log),
                    "typical_retention_at_512": math.exp(512 * mean_log),
                    "typical_retention_at_2048": math.exp(2048 * mean_log),
                }
            )

    half_lives = torch.tensor([row["half_life_tokens"] for row in rows], dtype=torch.float64)
    finite = half_lives[torch.isfinite(half_lives)]
    buckets = {
        "lt_32": int((half_lives < 32).sum()),
        "32_to_128": int(((half_lives >= 32) & (half_lives < 128)).sum()),
        "128_to_512": int(((half_lives >= 128) & (half_lives < 512)).sum()),
        "512_to_2048": int(((half_lives >= 512) & (half_lives < 2048)).sum()),
        "ge_2048": int((half_lives >= 2048).sum()),
    }
    overall = {
        "num_layer_heads": len(rows),
        "tokens_per_layer": sum(x.shape[0] for x in next(iter(capture.by_layer.values()))),
        "probe_sequence_length": seq_len,
        "half_life_min": float(finite.min()) if finite.numel() else math.inf,
        "half_life_median": float(finite.median()) if finite.numel() else math.inf,
        "half_life_max": float(finite.max()) if finite.numel() else math.inf,
        "half_life_buckets": buckets,
    }
    return rows, overall


def write_outputs(output_dir: Path, rows: list[dict], metadata: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gamma_horizons.csv"
    json_path = output_dir / "gamma_horizons.json"
    plot_path = output_dir / "gamma_half_lives.png"

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps({"metadata": metadata, "layer_heads": rows}, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    try:
        import matplotlib.pyplot as plt

        layers = sorted({row["layer"] for row in rows})
        fig, ax = plt.subplots(figsize=(10, 5))
        for layer in layers:
            selected = [row for row in rows if row["layer"] == layer]
            ax.plot(
                [row["head"] for row in selected],
                [row["half_life_tokens"] for row in selected],
                marker="o",
                label=f"layer {layer}",
            )
        for y, label in ((32, "32"), (128, "128"), (512, "512"), (2048, "training length")):
            ax.axhline(y, color="gray", linewidth=0.7, linestyle="--")
            ax.text(-0.45, y, label, va="bottom", fontsize=8, color="gray")
        ax.set_yscale("log")
        ax.set_xlabel("memory/query head")
        ax.set_ylabel("decay-only half-life (tokens, log scale)")
        ax.set_title("Activation-conditioned Titans gamma horizons")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(ncol=max(1, min(4, len(layers))))
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
    except ImportError:
        plot_path = None

    return csv_path, json_path, plot_path


def print_report(rows: list[dict], overall: dict):
    print("\nActivation-conditioned decay horizons")
    print("=" * 78)
    print(" layer head  mean gamma  gamma p10..p90       half-life     1/e horizon")
    for row in rows:
        print(
            f" {row['layer']:>5} {row['head']:>4}  {row['mean_gamma']:.6f}    "
            f"{row['gamma_p10']:.6f}..{row['gamma_p90']:.6f}  "
            f"{row['half_life_tokens']:>10.1f}  {row['efold_tokens']:>12.1f}"
        )
    print("-" * 78)
    print(
        "Across layer-heads: half-life "
        f"min={overall['half_life_min']:.1f}, median={overall['half_life_median']:.1f}, "
        f"max={overall['half_life_max']:.1f} tokens"
    )
    print("Half-life buckets:", overall["half_life_buckets"])
    print(
        "\nCaution: this is the explicit gamma-decay timescale. Delta-rule key "
        "replacement adds content-dependent forgetting, so it is not a recall guarantee."
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    weights_path, config_path, checkpoint_source = resolve_checkpoint(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    dtype = choose_dtype(args.dtype, device)
    if device.type == "cuda":
        print(f"[device] {torch.cuda.get_device_name(device)} | dtype={dtype}", flush=True)
    else:
        print(f"[device] CPU | dtype={dtype} (use --device cuda on Colab)", flush=True)

    model, config = load_model(weights_path, config_path, device, dtype)
    try:
        from raven_baseline.layers import _HAS_FLA_GSA
        from model.blocks import _HAS_FLA

        print(
            f"[kernels] Raven FLA={_HAS_FLA_GSA}, gated-delta FLA={_HAS_FLA}; "
            "PyTorch fallback is correct but slower.",
            flush=True,
        )
    except ImportError:
        pass

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install tokenization support with: pip install transformers") from exc
    tok_name = tokenizer_name(config_path, args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    tokenizer.model_max_length = 10**30

    capture = GammaCapture(model)
    try:
        sequences = packed_sequences(args, tokenizer)
        collect_gamma(model, sequences, capture, args.num_sequences, device)
    finally:
        capture.close()

    rows, overall = summarize(capture, args.seq_len)
    metadata = {
        "checkpoint": checkpoint_source,
        "architecture": config.get("arch_type"),
        "training_sequence_length": config.get("seq_len"),
        "probe_dataset": str(args.text_file) if args.text_file else args.dataset,
        "probe_dataset_config": None if args.text_file else args.dataset_config,
        "num_sequences": args.num_sequences,
        "sequence_length": args.seq_len,
        "tokenizer": tok_name,
        "device": str(device),
        "dtype": str(dtype),
        "summary": overall,
    }
    csv_path, json_path, plot_path = write_outputs(args.output_dir, rows, metadata)
    print_report(rows, overall)
    print(f"\nWrote {csv_path.resolve()}")
    print(f"Wrote {json_path.resolve()}")
    if plot_path:
        print(f"Wrote {plot_path.resolve()}")


if __name__ == "__main__":
    main()
