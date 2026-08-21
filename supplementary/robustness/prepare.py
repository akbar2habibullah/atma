"""Copy immutable source configs into a mutable GPU work directory."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from supplementary.robustness.validate_plan import ROOT, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "configs")
    parser.add_argument("--work", type=Path, default=ROOT / "work" / "configs")
    parser.add_argument("--force", action="store_true", help="replace an existing work config tree")
    args = parser.parse_args()
    allowed = (ROOT / "work").resolve()
    try:
        args.work.resolve().relative_to(allowed)
    except ValueError as exc:
        raise SystemExit(f"--work must stay under {allowed}") from exc
    errors = validate(args.source)
    if errors:
        raise SystemExit("source plan is invalid:\n" + "\n".join(f"- {e}" for e in errors))
    if args.work.exists():
        if not args.force:
            raise SystemExit(f"{args.work} already exists; use --force only before runs have started")
        shutil.rmtree(args.work)
    shutil.copytree(args.source, args.work)
    print(f"prepared mutable configs -> {args.work}")


if __name__ == "__main__":
    main()
