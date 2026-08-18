#!/usr/bin/env python3
"""Prepare or execute paired baseline/clamped benchmark re-evaluation.

The input is a completed ``clamp_sweep.json``. A benchmark plan is generated
only when the sweep's predeclared recovery rule selected a clamp. By default
this is a dry run; add ``--execute`` to launch the correctness-first benchmark
harness with identical baseline/clamped settings.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path,
                        default=Path("gamma_diagnostics/results/clamp_sweep.json"))
    parser.add_argument("--model", default=None,
                        help="checkpoint key in the sweep (optional when it contains one checkpoint)")
    parser.add_argument("--checkpoint", default=None,
                        help="override the checkpoint path recorded in the sweep")
    parser.add_argument("--benchmarks", nargs="+", choices=("retrieval", "longdoc", "base", "babilong"),
                        default=["retrieval", "longdoc"])
    parser.add_argument("--lengths", nargs="+", default=["2k", "16k", "64k"])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--base-tasks", nargs="+", default=["hellaswag", "piqa"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("gamma_diagnostics/results/re_evaluation"))
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value.rsplit("/", 1)[-1]).strip("-")


def _select_entry(payload: dict, model_key: str | None):
    checkpoints = payload.get("checkpoints", {})
    if model_key is None:
        if len(checkpoints) != 1:
            raise SystemExit("--model is required when the sweep contains multiple checkpoints")
        model_key = next(iter(checkpoints))
    if model_key not in checkpoints:
        raise SystemExit(f"model {model_key!r} is not present in the sweep")
    entry = checkpoints[model_key]
    recommendation = entry.get("recommendation")
    if not recommendation:
        raise SystemExit(
            "The sweep found no qualifying recovery. No benchmark re-evaluation was prepared."
        )
    return model_key, entry, recommendation


def _benchmark_args(name: str, args) -> list[str]:
    if name == "retrieval":
        return [
            "--tasks", "passkey", "niah", "--lengths", *args.lengths,
            "--depths", "0.0", "0.25", "0.5", "0.75", "1.0",
            "--samples", str(args.samples), "--seed", str(args.seed),
        ]
    if name == "longdoc":
        return [
            "--datasets", "pg19", "proof_pile", "finepdfs", "--lengths", *args.lengths,
            "--num_docs", str(max(1, min(args.samples, 8))),
        ]
    if name == "base":
        return ["--tasks", *args.base_tasks, "--limit", str(args.samples)]
    return [
        "--tasks", "qa1", "--lengths", *args.lengths, "--samples", str(args.samples),
        "--babilong_backend", "direct",
    ]


def main():
    args = _parse_args()
    payload = json.loads(args.sweep.read_text(encoding="utf-8"))
    model_key, entry, recommendation = _select_entry(payload, args.model)
    checkpoint = args.checkpoint or entry["checkpoint_dir"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = _slug(model_key)
    spec_path = args.output_dir / f"{stem}.gamma-clamp.json"
    spec_path.write_text(json.dumps(recommendation["spec"], indent=2), encoding="utf-8")

    jobs = []
    for benchmark in args.benchmarks:
        common = [
            sys.executable, "-m", "benchmarks.run", "--benchmark", benchmark,
            "--model", str(checkpoint), *_benchmark_args(benchmark, args),
        ]
        for condition, clamp_args in (
            ("baseline", []),
            ("clamped", ["--gamma-clamp", str(spec_path.resolve())]),
        ):
            output_path = args.output_dir / f"{stem}.{benchmark}.{condition}.log"
            jobs.append({
                "benchmark": benchmark,
                "condition": condition,
                "command": [*common, *clamp_args, "--out", str(output_path.resolve())],
                "output": str(output_path.resolve()),
            })

    plan = {
        "source_sweep": str(args.sweep.resolve()),
        "model": model_key,
        "checkpoint": str(checkpoint),
        "recommendation": recommendation,
        "gamma_clamp_spec": str(spec_path.resolve()),
        "paired_seed": args.seed,
        "jobs": jobs,
    }
    plan_path = args.output_dir / f"{stem}.benchmark-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Wrote clamp spec: {spec_path.resolve()}")
    print(f"Wrote paired benchmark plan: {plan_path.resolve()}")
    for job in jobs:
        print("\n" + subprocess.list2cmdline(job["command"]))
        if args.execute:
            subprocess.run(job["command"], check=True)


if __name__ == "__main__":
    main()
