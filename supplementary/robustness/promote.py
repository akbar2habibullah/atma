"""Record pilot decisions and enable only prespecified scaled baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from scaled_ablation.parse_logs import parse_log


ROOT = Path(__file__).resolve().parent


def _record(log_dir: Path, arch: str):
    path = log_dir / f"pilot_{arch}.log"
    if not path.exists():
        return None
    record = parse_log(path)
    return {
        "status": record["status"],
        "metrics": record["metrics"],
        "error": record["error"],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_root", type=Path, default=ROOT / "work" / "configs")
    parser.add_argument("--pilot_logs", type=Path, default=ROOT / "work" / "logs" / "baseline_pilots")
    parser.add_argument("--tda", choices=("promote", "omit"), required=True)
    parser.add_argument("--linear", choices=("mamba3_native", "gdn2_native"), required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--force", action="store_true", help="record decision without completed pilot logs")
    args = parser.parse_args()

    evidence = {arch: _record(args.pilot_logs, arch) for arch in ("tda_hybrid", "mamba3_native", "gdn2_native")}
    missing = [arch for arch in ("mamba3_native", "gdn2_native")
               if not evidence[arch] or evidence[arch]["status"] != "done"]
    if args.tda == "promote" and (not evidence["tda_hybrid"] or evidence["tda_hybrid"]["status"] != "done"):
        missing.append("tda_hybrid")
    if args.tda == "omit" and (
        not evidence["tda_hybrid"] or evidence["tda_hybrid"]["status"] not in {"done", "error"}
    ):
        missing.append("tda_hybrid_terminal_log")
    if missing and not args.force:
        raise SystemExit(f"cannot promote without completed pilot logs: {missing}")

    scaled = args.work_root / "baseline_scaled"
    for path in scaled.glob("*.json"):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        arch = cfg["arch_type"]
        cfg["enabled"] = (arch == args.linear) or (arch == "tda_hybrid" and args.tda == "promote")
        path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tda": args.tda,
        "linear": args.linear,
        "reason": args.reason,
        "pilot_logs": str(args.pilot_logs),
        "pilot_evidence": evidence,
        "forced_without_complete_logs": bool(missing),
    }
    out = args.work_root.parent / "promotion_decision.json"
    out.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(f"recorded promotion -> {out}")


if __name__ == "__main__":
    main()
