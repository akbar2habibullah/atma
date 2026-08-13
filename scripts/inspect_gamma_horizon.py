#!/usr/bin/env python3
"""Read learned Titans/GDN gamma parameters from ATMA 10B-token checkpoints.

No model forward, tokenizer, dataset, or GPU is used. The script downloads each
checkpoint, finds every ``*.mem.w_gamma.{weight,bias}``, and reports the learned
zero-input gate operating point

    gamma_0 = sigmoid(w_gamma.bias + mem_gamma_bias)

and its constant-decay half-life/e-folding length. It also reports the norm of
each learned weight row because gamma is input-conditioned in the actual model:

    gamma_t = sigmoid(w_gamma.weight @ x_t + w_gamma.bias + mem_gamma_bias).

Consequently, gamma_0 is the checkpoint parameter-only answer, not the gate's
runtime distribution on text. Measuring the latter necessarily requires inputs.

Colab
------
    !git clone https://github.com/akbar2habibullah/atma.git
    %cd atma
    !pip -q install huggingface_hub
    !python scripts/inspect_gamma_horizon.py

Custom checkpoints (repeat either option as needed):

    python scripts/inspect_gamma_horizon.py --checkpoint /content/run/weights.pt
    python scripts/inspect_gamma_horizon.py --repo-id owner/model-a --repo-id owner/model-b
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


# The known base 10B-token checkpoints that contain the shared Titans branch.
# This includes the additional NoPE hardware/microbatch controls from the stress
# sweep. Raven-native is deliberately absent because it used mem_enabled=false.
DEFAULT_REPOS = [
    "ChavyvAkvar/atma-10b-L4-mbs4-nope__reg-baseline__distr-0__mem-1__win-0",
    "ChavyvAkvar/atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0",
    "ChavyvAkvar/atma-10b-L40S-mbs4-nope__reg-baseline__distr-0__mem-1__win-0",
    "ChavyvAkvar/atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0",
    "ChavyvAkvar/atma-10b-L40S-mbs16-rope__reg-baseline__distr-0__mem-1__win-0",
    "ChavyvAkvar/atma-10b-L40S-mbs16-atma-raven-titans__reg-baseline__distr-0__mem-1__win-0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read gamma biases/horizons from one or more ATMA checkpoints."
    )
    parser.add_argument(
        "--repo-id",
        action="append",
        default=None,
        help="Hugging Face repo to inspect. Repeatable. Defaults to all known base 10B memory runs.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        default=None,
        help="Local checkpoint directory or weights.pt. Repeatable; may be combined with --repo-id.",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("gamma_parameter_report"))
    return parser.parse_args()


def resolve_local(candidate: Path) -> tuple[str, Path, Path]:
    candidate = candidate.expanduser().resolve()
    if candidate.is_dir():
        weights_path = candidate / "weights.pt"
        config_path = candidate / "config.json"
        label = candidate.name
    else:
        weights_path = candidate
        config_path = candidate.parent / "config.json"
        label = candidate.parent.name or candidate.stem
    missing = [str(path) for path in (weights_path, config_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint file(s): {missing}")
    return label, weights_path, config_path


def resolve_repo(repo_id: str, revision: str, cache_dir: Path | None) -> tuple[str, Path, Path]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install download support with: pip install huggingface_hub") from exc

    print(f"[download] {repo_id}@{revision}", flush=True)
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            allow_patterns=["weights.pt", "config.json"],
        )
    )
    return repo_id.rsplit("/", 1)[-1], snapshot / "weights.pt", snapshot / "config.json"


def unwrap_state(payload) -> dict[str, torch.Tensor]:
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload).__name__}")
    return {key.removeprefix("_orig_mod."): value for key, value in state.items()}


def horizon(gamma: float, target: float) -> float:
    """L such that gamma**L == target."""
    if gamma >= 1.0:
        return math.inf
    if gamma <= 0.0:
        return 0.0
    return math.log(target) / math.log(gamma)


def inspect_checkpoint(label: str, weights_path: Path, config_path: Path) -> list[dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fixed_bias = float(config.get("mem_gamma_bias", 3.9))
    # mmap avoids copying the full ~1.4 GB checkpoint into Colab RAM.
    try:
        payload = torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # older PyTorch without mmap=
        payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = unwrap_state(payload)

    weight_keys = sorted(key for key in state if key.endswith(".mem.w_gamma.weight"))
    bias_keys = sorted(key for key in state if key.endswith(".mem.w_gamma.bias"))
    if not weight_keys:
        print(f"[skip] {label}: no Titans w_gamma parameters (mem_enabled=false?)")
        return []

    weights = {key[: -len(".weight")]: state[key].detach().float() for key in weight_keys}
    biases = {key[: -len(".bias")]: state[key].detach().float() for key in bias_keys}
    rows: list[dict] = []
    for module_name, weight in weights.items():
        if weight.ndim != 2:
            raise ValueError(f"Expected matrix at {module_name}.weight, got {tuple(weight.shape)}")
        learned_bias = biases.get(module_name)
        if learned_bias is None:
            learned_bias = torch.zeros(weight.shape[0], dtype=torch.float32)
        if learned_bias.shape != (weight.shape[0],):
            raise ValueError(
                f"Bias/weight mismatch for {module_name}: {tuple(learned_bias.shape)} vs {tuple(weight.shape)}"
            )

        gamma0 = torch.sigmoid(learned_bias + fixed_bias)
        row_norm = weight.norm(dim=1)
        row_rms = weight.square().mean(dim=1).sqrt()
        layer = next(
            (int(part) for previous, part in zip(module_name.split("."), module_name.split(".")[1:])
             if previous == "blocks" and part.isdigit()),
            -1,
        )
        for head in range(weight.shape[0]):
            gamma = float(gamma0[head])
            rows.append(
                {
                    "checkpoint": label,
                    "architecture": config.get("arch_type", config.get("attn_type", "unknown")),
                    "layer": layer,
                    "head": head,
                    "learned_bias": float(learned_bias[head]),
                    "fixed_config_bias": fixed_bias,
                    "total_zero_input_logit": float(learned_bias[head] + fixed_bias),
                    "gamma_zero_input": gamma,
                    "half_life_tokens": horizon(gamma, 0.5),
                    "efold_tokens": horizon(gamma, math.exp(-1.0)),
                    "weight_l2": float(row_norm[head]),
                    "weight_rms": float(row_rms[head]),
                    "input_dim": int(weight.shape[1]),
                    "parameter": module_name,
                }
            )

    del state, payload
    print(f"[read] {label}: {len(weight_keys)} memory layers, {len(rows)} layer-heads")
    return rows


def print_report(rows: list[dict]):
    print("\nLearned parameter-only gamma operating points")
    print("=" * 99)
    print(" checkpoint                    layer head learned_b  gamma_0   half-life  1/e length  W-row L2")
    print("-" * 99)
    for row in rows:
        short = row["checkpoint"].replace("atma-10b-L40S-mbs16-", "").split("__", 1)[0]
        print(
            f" {short[:29]:<29} {row['layer']:>5} {row['head']:>4} "
            f"{row['learned_bias']:>9.4f}  {row['gamma_zero_input']:.6f} "
            f"{row['half_life_tokens']:>10.1f} {row['efold_tokens']:>11.1f} "
            f"{row['weight_l2']:>9.4f}"
        )

    print("\nCheckpoint summaries")
    print("-" * 99)
    for checkpoint in dict.fromkeys(row["checkpoint"] for row in rows):
        selected = [row for row in rows if row["checkpoint"] == checkpoint]
        half_lives = torch.tensor([row["half_life_tokens"] for row in selected])
        gammas = torch.tensor([row["gamma_zero_input"] for row in selected])
        short = checkpoint.replace("atma-10b-L40S-mbs16-", "").split("__", 1)[0]
        print(
            f" {short:<29} gamma_0 {gammas.min():.6f}..{gammas.max():.6f} | "
            f"half-life min/median/max "
            f"{half_lives.min():.1f}/{half_lives.median():.1f}/{half_lives.max():.1f} tokens"
        )
    print(
        "\nMeaning: gamma_0 uses only learned checkpoint parameters (x=0). The nonzero W-row L2 "
        "shows that runtime gamma is token/activation-dependent; it cannot be inferred from parameters alone."
    )


def write_outputs(output_dir: Path, rows: list[dict]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gamma_parameters.csv"
    json_path = output_dir / "gamma_parameters.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path


def main():
    args = parse_args()
    sources: list[tuple[str, Path, Path]] = []
    repos = args.repo_id if args.repo_id is not None else ([] if args.checkpoint else DEFAULT_REPOS)
    for repo_id in repos:
        sources.append(resolve_repo(repo_id, args.revision, args.cache_dir))
    for checkpoint in args.checkpoint or []:
        sources.append(resolve_local(checkpoint))
    if not sources:
        raise SystemExit("No checkpoints selected.")

    rows: list[dict] = []
    for source in sources:
        rows.extend(inspect_checkpoint(*source))
    if not rows:
        raise SystemExit("None of the selected checkpoints contains Titans w_gamma parameters.")

    print_report(rows)
    csv_path, json_path = write_outputs(args.output_dir, rows)
    print(f"\nWrote {csv_path.resolve()}")
    print(f"Wrote {json_path.resolve()}")


if __name__ == "__main__":
    main()
