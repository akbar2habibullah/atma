from __future__ import annotations

import argparse
import json
import os

from raven_baseline.config_schema import ARCH_TYPES, expand_bridge, expand_scaled


def main():
    ap = argparse.ArgumentParser(description="Generate Raven bridge/scaled config files.")
    ap.add_argument("--out", default="raven_baseline/configs")
    ap.add_argument("--scaled", action="store_true", help="generate scaled-promotion configs instead of 1B bridge")
    ap.add_argument("--arch_types", nargs="+", default=None, choices=ARCH_TYPES)
    ap.add_argument("--num_chunks", type=int, default=None)
    ap.add_argument("--val_tokens", type=int, default=None)
    ap.add_argument("--mbs", type=int, default=None)
    ap.add_argument("--num_eval_docs", type=int, default=None)
    ap.add_argument("--num_needle_trials", type=int, default=None)
    ap.add_argument("--optimizer", choices=["adamw_raven", "atma_muon"], default=None)
    ap.add_argument("--adamw_lr", type=float, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()

    overrides = {}
    for name in ("num_chunks", "val_tokens", "mbs", "num_eval_docs", "num_needle_trials"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    if args.optimizer is not None:
        overrides["optimizer"] = args.optimizer
    if args.adamw_lr is not None:
        overrides["adamw_lr"] = args.adamw_lr

    arch_types = args.arch_types or ARCH_TYPES
    configs = expand_scaled(arch_types, **overrides) if args.scaled else [
        c for c in expand_bridge(**overrides) if c.arch_type in arch_types
    ]
    os.makedirs(args.out, exist_ok=True)
    for c in configs:
        d = c.to_dict()
        if args.max_steps is not None:
            d["max_steps"] = args.max_steps
        with open(os.path.join(args.out, f"{c.run_id}.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    kind = "scaled" if args.scaled else "1B bridge"
    print(f"Wrote {len(configs)} Raven {kind} config(s) to {args.out}/")
    for c in configs:
        print(f"  {c.run_id} ({c.mixer_ratio})")


if __name__ == "__main__":
    main()

