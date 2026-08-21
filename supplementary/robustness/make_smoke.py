"""Create disposable three-step configs covering every new code path."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_root", type=Path, default=ROOT / "work" / "configs")
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "smoke" / "configs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    allowed = (ROOT / "work").resolve()
    try:
        args.out.resolve().relative_to(allowed)
    except ValueError as exc:
        raise SystemExit(f"--out must stay under {allowed}") from exc
    if args.out.exists():
        if not args.force:
            raise SystemExit(f"{args.out} exists; pass --force to recreate disposable smoke configs")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    paths = []
    paths.extend((args.work_root / "replication").glob("repl_seed1_*.json"))
    paths.extend((args.work_root / "polar_components").glob("*.json"))
    paths.extend((args.work_root / "baseline_pilots").glob("*.json"))
    for path in sorted(paths):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        cfg["run_id"] = "smoke_" + cfg["run_id"]
        cfg["max_steps"] = 3
        cfg["num_chunks"] = 1
        cfg["val_tokens"] = 524288
        cfg["eval_lengths"] = [2048, 4096]
        cfg["needle_distances"] = [2048, 4096]
        cfg["num_eval_docs"] = 2
        cfg["num_needle_trials"] = 2
        cfg["enabled"] = True
        (args.out / f"{cfg['run_id']}.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(paths)} smoke configs -> {args.out}")


if __name__ == "__main__":
    main()
