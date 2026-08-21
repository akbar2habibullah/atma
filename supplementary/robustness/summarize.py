"""Collect all supplementary structured logs into one immutable result view."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scaled_ablation.parse_logs import parse_log


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=ROOT / "work" / "logs")
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "results" / "all_results.json")
    args = parser.parse_args()
    records = [parse_log(path) for path in sorted(args.logs.glob("*/*.log"))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    statuses = {}
    for record in records:
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
    print(f"wrote {len(records)} records -> {args.out}; status={statuses}")


if __name__ == "__main__":
    main()

