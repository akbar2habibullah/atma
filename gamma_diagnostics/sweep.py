#!/usr/bin/env python3
"""Causally test whether selective runtime gamma caps improve extrapolation.

Each checkpoint is loaded once. The script evaluates the untouched baseline and
then reversible caps on the highest zero-input-horizon layer-head(s). ``p90``
and ``p99`` are checkpoint-local parameter quantiles; ``hl:256`` is an absolute
runtime half-life ceiling. Runtime activations can still make gamma smaller.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("FLA_CUSTOM_OP", "1")

import torch

from eval import load_from_checkpoint
from gamma_diagnostics.clamp import FORMAT_VERSION, apply_gamma_clamp, half_life_to_logit
from gamma_diagnostics.selection import recommend
from scaled_ablation.evaluate import _set_full_context
from scaled_ablation.eval_hf_checkpoints import (
    _atomic_json_dump,
    _download_checkpoint,
    _load_json_if_present,
    _load_or_create_docs,
    _runtime_metadata,
    _sdpa_context,
    eval_clean,
    eval_needle,
)


DEFAULT_MODELS = [
    "ChavyvAkvar/atma-10b-L4-mbs4-nope__reg-baseline__distr-0__mem-1__win-0",
    "ChavyvAkvar/atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0",
    "ChavyvAkvar/atma-10b-L40S-mbs4-nope__reg-baseline__distr-0__mem-1__win-0",
    "ChavyvAkvar/atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0",
    "ChavyvAkvar/atma-10b-L40S-mbs16-rope__reg-baseline__distr-0__mem-1__win-0",
]
_BLOCK_RE = re.compile(r"(?:^|\.)blocks\.(\d+)(?:\.|$)")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Hugging Face repo IDs or local checkpoint directories")
    parser.add_argument("--caps", nargs="+", default=["p90", "p99", "hl:256", "hl:512"])
    parser.add_argument("--num-target-heads", type=int, default=1,
                        help="highest zero-input-horizon layer-heads to intervene on")
    parser.add_argument("--lengths", nargs="+", type=int, default=[2048, 16384, 65536])
    parser.add_argument("--metrics", nargs="+", choices=("clean", "needle"),
                        default=["clean", "needle"])
    parser.add_argument("--num-eval-docs", type=int, default=4)
    parser.add_argument("--num-needle-trials", type=int, default=8)
    parser.add_argument("--needle-value-len", type=int, default=5)
    parser.add_argument("--loss-chunk", type=int, default=8192)
    parser.add_argument("--short-loss-tolerance", type=float, default=0.05,
                        help="maximum allowed shortest-length clean-loss regression in nats")
    parser.add_argument("--min-needle-improvement", type=float, default=0.05,
                        help="minimum longest-distance needle CE reduction in nats")
    parser.add_argument("--sdpa-backend", choices=("auto", "flash", "math", "efficient"),
                        default="flash")
    parser.add_argument("--clean-dataset", default="codelion/finepdfs-1B")
    parser.add_argument("--clean-text-key", default="text")
    parser.add_argument("--clean-split", default="train")
    parser.add_argument("--doc-manifest", default=None,
                        help="shared document cache (default derives from max length and doc count)")
    parser.add_argument("--rebuild-doc-manifest", action="store_true")
    parser.add_argument("--hf-cache", default=None)
    parser.add_argument("--output", type=Path,
                        default=Path("gamma_diagnostics/results/clamp_sweep.json"))
    return parser.parse_args()


def _checkpoint_path(model_ref: str, cache_dir: str | None) -> Path:
    path = Path(model_ref).expanduser()
    if path.exists():
        root = path if path.is_dir() else path.parent
        if not (root / "weights.pt").is_file() or not (root / "config.json").is_file():
            raise FileNotFoundError(f"{root} must contain weights.pt and config.json")
        return root.resolve()
    return _download_checkpoint(model_ref, cache_dir)


def parameter_operating_points(model) -> list[dict]:
    rows = []
    for name, module in model.named_modules():
        if not (hasattr(module, "w_gamma") and hasattr(module, "gamma_bias")):
            continue
        match = _BLOCK_RE.search(name)
        if match is None:
            continue
        bias = module.w_gamma.bias
        if bias is None:
            bias = torch.zeros(module.w_gamma.out_features, device=module.w_gamma.weight.device)
        for head, learned_bias in enumerate(bias.detach().float().cpu()):
            rows.append({
                "layer": int(match.group(1)),
                "head": head,
                "learned_bias": float(learned_bias),
                "fixed_bias": float(module.gamma_bias),
                "zero_input_logit": float(learned_bias) + float(module.gamma_bias),
            })
    return sorted(rows, key=lambda row: row["zero_input_logit"], reverse=True)


def _resolve_cap(text: str, points: list[dict]) -> tuple[str, float, dict]:
    normalized = text.lower().strip()
    logits = torch.tensor([row["zero_input_logit"] for row in points], dtype=torch.float64)
    if normalized.startswith("p"):
        percentile = float(normalized[1:])
        if not 0 < percentile < 100:
            raise ValueError(f"invalid checkpoint percentile cap {text!r}")
        cap = float(torch.quantile(logits, percentile / 100.0))
        return normalized, cap, {"kind": "checkpoint_parameter_percentile", "percentile": percentile}
    if normalized.startswith("hl:"):
        half_life = float(normalized.split(":", 1)[1])
        return f"hl-{half_life:g}", half_life_to_logit(half_life), {
            "kind": "absolute_half_life", "half_life_tokens": half_life
        }
    raise ValueError(f"unknown cap {text!r}; use p90, p99, or hl:256")


def _spec(targets: list[dict], label: str, cap_logit: float, cap_source: dict) -> dict:
    return {
        "format": FORMAT_VERSION,
        "label": label,
        "cap_source": cap_source,
        "targets": [
            {"layer": row["layer"], "heads": [row["head"]], "max_logit": cap_logit}
            for row in targets
        ],
    }


def _evaluate(model, args, docs, lengths) -> dict:
    metrics = {}
    if "clean" in args.metrics:
        metrics["clean"] = eval_clean(model, docs, lengths, torch.device("cuda"), args.loss_chunk)
    if "needle" in args.metrics:
        metrics["needle"] = eval_needle(
            model, docs, lengths, args.num_needle_trials, args.needle_value_len,
            torch.device("cuda"),
        )
    return metrics


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.num_target_heads <= 0 or args.num_eval_docs <= 0 or args.num_needle_trials <= 0:
        raise SystemExit("target-head and evaluation counts must be positive")
    if args.short_loss_tolerance < 0 or args.min_needle_improvement <= 0:
        raise SystemExit("selection tolerances must be nonnegative/positive")
    lengths = sorted(set(args.lengths))
    if not lengths or min(lengths) <= 0:
        raise SystemExit("--lengths must contain positive integers")
    if args.doc_manifest is None:
        args.doc_manifest = str(
            Path("gamma_diagnostics/manifests")
            / f"finepdfs_{max(lengths)}_{args.num_eval_docs}.pt"
        )

    doc_args = SimpleNamespace(**vars(args))
    need = max(lengths) + (64 if "needle" in args.metrics else 0)
    docs = _load_or_create_docs(doc_args, need)
    output = {
        "runtime": _runtime_metadata(args.sdpa_backend),
        "settings": {
            "models": args.models, "caps": args.caps, "num_target_heads": args.num_target_heads,
            "lengths": lengths, "metrics": args.metrics, "num_eval_docs": args.num_eval_docs,
            "num_needle_trials": args.num_needle_trials,
            "short_loss_tolerance": args.short_loss_tolerance,
            "min_needle_improvement": args.min_needle_improvement,
        },
        "checkpoints": {},
    }
    _atomic_json_dump(args.output, output)

    for model_ref in args.models:
        print(f"\n{'=' * 100}\nGamma clamp sweep: {model_ref}\n{'=' * 100}", flush=True)
        checkpoint = _checkpoint_path(model_ref, args.hf_cache)
        model, _ = load_from_checkpoint(str(checkpoint), torch.device("cuda"), compile_model=False)
        model = getattr(model, "_orig_mod", model)
        model.eval()
        _set_full_context(model)
        points = parameter_operating_points(model)
        if not points:
            raise RuntimeError(f"{model_ref} contains no Titans gamma modules")
        targets = points[:args.num_target_heads]
        entry = {
            "checkpoint_dir": str(checkpoint),
            "model_config": _load_json_if_present(checkpoint / "config.json"),
            "selected_targets": targets,
            "parameter_operating_points": points,
            "conditions": {},
            "recommendation": None,
        }
        output["checkpoints"][model_ref] = entry

        with _sdpa_context(args.sdpa_backend):
            print("\n--- baseline (no clamp) ---", flush=True)
            entry["conditions"]["baseline"] = {
                "spec": None, "resolved_targets": [],
                "metrics": _evaluate(model, args, docs, lengths),
            }
            _atomic_json_dump(args.output, output)
            for cap_text in args.caps:
                label, cap_logit, source = _resolve_cap(cap_text, points)
                spec = _spec(targets, label, cap_logit, source)
                print(f"\n--- {label}: final-logit cap {cap_logit:.6f} ---", flush=True)
                handle = apply_gamma_clamp(model, spec)
                try:
                    metrics = _evaluate(model, args, docs, lengths)
                finally:
                    handle.remove()
                entry["conditions"][label] = {
                    "spec": spec,
                    "resolved_targets": handle.resolved_targets,
                    "metrics": metrics,
                }
                _atomic_json_dump(args.output, output)

        entry["recommendation"] = recommend(
            entry["conditions"], lengths, args.short_loss_tolerance,
            args.min_needle_improvement,
        )
        _atomic_json_dump(args.output, output)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nResults written to {args.output.resolve()}")
    for model_ref, entry in output["checkpoints"].items():
        recommendation = entry["recommendation"]
        print(f"  {model_ref}: {recommendation['condition'] if recommendation else 'no qualifying recovery'}")


if __name__ == "__main__":
    main()
