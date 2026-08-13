#!/usr/bin/env python3
"""Inspect learned Titans gamma parameters without running the model.

This reads every ``*.mem.w_gamma.{weight,bias}`` from every selected checkpoint.
No tokenizer, sequences, model forward, or GPU are involved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


DEFAULT_REPOS = [
    "ChavyvAkvar/atma-10b-L4-mbs4-nope__reg-baseline__distr-0__mem-1__win-0",
    "ChavyvAkvar/atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0",
    "ChavyvAkvar/atma-10b-L40S-mbs4-nope__reg-baseline__distr-0__mem-1__win-0",
    "ChavyvAkvar/atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0",
    "ChavyvAkvar/atma-10b-L40S-mbs16-rope__reg-baseline__distr-0__mem-1__win-0",
    "ChavyvAkvar/atma-10b-L40S-mbs16-atma-raven-titans__reg-baseline__distr-0__mem-1__win-0",
]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", action="append", default=None)
    parser.add_argument("--checkpoint", action="append", type=Path, default=None)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("gamma_diagnostics/results/parameters"))
    return parser.parse_args()


def _resolve_local(candidate: Path):
    candidate = candidate.expanduser().resolve()
    weights = candidate / "weights.pt" if candidate.is_dir() else candidate
    config = weights.parent / "config.json"
    missing = [str(path) for path in (weights, config) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoint file(s): {missing}")
    return weights.parent.name or weights.stem, weights, config


def _validate_source(label: str, weights: Path, config: Path):
    missing = [str(path) for path in (weights, config) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label} is missing checkpoint file(s): {missing}")
    return label, weights, config


def _resolve_repo(repo_id: str, revision: str, cache_dir: Path | None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install download support with: pip install huggingface_hub") from exc
    print(f"[download] {repo_id}@{revision}", flush=True)
    root = Path(snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        allow_patterns=["weights.pt", "config.json"],
    ))
    return _validate_source(
        repo_id.rsplit("/", 1)[-1], root / "weights.pt", root / "config.json"
    )


def _unwrap(payload):
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"unsupported checkpoint payload: {type(payload).__name__}")
    return {key.removeprefix("_orig_mod."): value for key, value in state.items()}


def log_sigmoid(logit: float) -> float:
    if logit >= 0:
        return -math.log1p(math.exp(-logit))
    return logit - math.log1p(math.exp(logit))


def horizon(log_gamma: float, target: float) -> float:
    return math.log(target) / log_gamma if log_gamma < 0 else math.inf


def inspect_checkpoint(label: str, weights_path: Path, config_path: Path) -> list[dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fixed_bias = float(config.get("mem_gamma_bias", 3.9))
    try:
        payload = torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = _unwrap(payload)
    weight_keys = sorted(key for key in state if key.endswith(".mem.w_gamma.weight"))
    rows = []
    for weight_key in weight_keys:
        base = weight_key.removesuffix(".weight")
        weight = state[weight_key].detach().float()
        bias = state.get(base + ".bias", torch.zeros(weight.shape[0])).detach().float()
        layer_parts = base.split(".")
        layer = next((int(part) for prev, part in zip(layer_parts, layer_parts[1:])
                      if prev == "blocks" and part.isdigit()), -1)
        for head, (learned_bias, weight_row) in enumerate(zip(bias, weight)):
            total_logit = float(learned_bias) + fixed_bias
            log_gamma = log_sigmoid(total_logit)
            rows.append({
                "checkpoint": label,
                "architecture": config.get("arch_type", config.get("attn_type", "unknown")),
                "layer": layer,
                "head": head,
                "learned_bias": float(learned_bias),
                "fixed_config_bias": fixed_bias,
                "total_zero_input_logit": total_logit,
                "gamma_zero_input": math.exp(log_gamma),
                "one_minus_gamma": -math.expm1(log_gamma),
                "half_life_tokens": horizon(log_gamma, 0.5),
                "efold_tokens": horizon(log_gamma, math.exp(-1)),
                "weight_l2": float(weight_row.norm()),
                "weight_rms": float(weight_row.square().mean().sqrt()),
                "input_dim": int(weight.shape[1]),
                "parameter": base,
            })
    print(f"[read] {label}: {len(weight_keys)} memory layers, {len(rows)} layer-heads")
    return rows


def _short_name(name: str) -> str:
    return name.replace("atma-10b-L40S-mbs16-", "").split("__", 1)[0]


def print_report(rows: list[dict]):
    print("\nLearned parameter-only gamma operating points")
    print("=" * 99)
    print(" checkpoint                    layer head learned_b  gamma_0   half-life  1/e length  W-row L2")
    print("-" * 99)
    for row in rows:
        print(
            f" {_short_name(row['checkpoint'])[:29]:<29} {row['layer']:>5} {row['head']:>4} "
            f"{row['learned_bias']:>9.4f}  {row['gamma_zero_input']:.9f} "
            f"{row['half_life_tokens']:>10.1f} {row['efold_tokens']:>11.1f} {row['weight_l2']:>9.4f}"
        )
    print("\nCheckpoint summaries")
    print("-" * 99)
    for checkpoint in dict.fromkeys(row["checkpoint"] for row in rows):
        selected = [row for row in rows if row["checkpoint"] == checkpoint]
        half_lives = sorted(row["half_life_tokens"] for row in selected)
        gammas = [row["gamma_zero_input"] for row in selected]
        median = half_lives[(len(half_lives) - 1) // 2]
        print(
            f" {_short_name(checkpoint):<29} gamma_0 {min(gammas):.9f}..{max(gammas):.9f} | "
            f"half-life min/median/max {min(half_lives):.1f}/{median:.1f}/{max(half_lives):.1f} tokens"
        )
    print("\nParameter-only gamma uses x=0. Nonzero W-row norms mean runtime gamma remains token-dependent.")


def _write(output_dir: Path, rows: list[dict]):
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
    args = _parse_args()
    repos = args.repo_id if args.repo_id is not None else ([] if args.checkpoint else DEFAULT_REPOS)
    sources = [_resolve_repo(repo, args.revision, args.cache_dir) for repo in repos]
    sources.extend(_resolve_local(path) for path in args.checkpoint or [])
    if not sources:
        raise SystemExit("No checkpoints selected.")
    rows = [row for source in sources for row in inspect_checkpoint(*source)]
    if not rows:
        raise SystemExit("None of the selected checkpoints contains Titans w_gamma parameters.")
    print_report(rows)
    csv_path, json_path = _write(args.output_dir, rows)
    print(f"\nWrote {csv_path.resolve()}\nWrote {json_path.resolve()}")


if __name__ == "__main__":
    main()
