#!/usr/bin/env python3
"""Run the paired BPB and retrieval evaluation for the four 10B replications.

This is a deliberately narrow wrapper around ``gamma_diagnostics.rebenchmark_all``.
It schedules only synthetic retrieval, real-text retrieval, and fixed-target
long-document BPB.  Each checkpoint is evaluated untouched and with its single
largest parameter-only gamma head capped to a 256-token half-life.  It never
schedules downstream/base tasks or BABILong.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gamma_diagnostics.rebenchmark_all import main as rebenchmark_main


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "supplementary" / "robustness" / "replication_benchmark_manifest.json"
DEFAULT_OUTPUT_DIR = (
    ROOT / "supplementary" / "robustness" / "work" / "evaluation" / "replication"
)
MODELS = (
    "repl_seed1_nope",
    "repl_seed1_polar",
    "repl_seed2_nope",
    "repl_seed2_polar",
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--hf-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--execute", action="store_true",
        help="run the jobs; otherwise download/inspect checkpoints and print the plan",
    )
    return parser.parse_args(argv)


def _rebenchmark_args(args) -> list[str]:
    command = [
        "--models", *args.models,
        "--benchmarks", "retrieval", "longdoc",
        "--base-manifest", str(MANIFEST),
        "--output-dir", str(args.output_dir),
        "--max-half-life", "256",
        "--paired",
        "--gpu", str(args.gpu),
    ]
    if args.hf_cache is not None:
        command.extend(("--hf-cache", str(args.hf_cache)))
    if args.offline:
        command.append("--offline")
    if args.execute:
        command.append("--execute")
    return command


def main(argv=None):
    args = _parse_args(argv)
    return rebenchmark_main(_rebenchmark_args(args))


if __name__ == "__main__":
    main()
