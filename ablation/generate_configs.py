"""Write the 120 ablation config JSONs to a directory.

    python -m ablation.generate_configs                       # -> ablation/configs/*.json (120)
    python -m ablation.generate_configs --out ablation/smoke --num_chunks 1 --val_tokens 524288 \
        --max_steps 3                                          # tiny smoke configs

Each file is <run_id>.json holding the full resolved RunConfig. Wall can still be generated
explicitly for incompatibility diagnostics, but it is not part of the fair grid.
"""
import argparse
import json
import os

from ablation.config_schema import ALL_ATTN_TYPES, ATTN_TYPES, REG_MODES, expand_grid, shard_configs


def main():
    ap = argparse.ArgumentParser(description="Generate the ablation grid config files.")
    ap.add_argument("--out", default="ablation/configs", help="output directory for *.json")
    ap.add_argument("--attn_types", nargs="+", default=None,
                    help="restrict to these attn_type(s); `wall` is diagnostic-only/incompatible")
    ap.add_argument("--shards", type=int, default=1,
                    help="split into N balanced subdirs out/shard{0..N-1}/ (one per GPU/host)")
    ap.add_argument("--num_chunks", type=int, default=None, help="override token budget (chunks)")
    ap.add_argument("--val_tokens", type=int, default=None, help="override validation token count")
    ap.add_argument("--mbs", type=int, default=None, help="override microbatch size")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="cap training steps (smoke tests); stored as an extra key, honored by ablation.train")
    args = ap.parse_args()

    overrides = {}
    if args.num_chunks is not None:
        overrides["num_chunks"] = args.num_chunks
    if args.val_tokens is not None:
        overrides["val_tokens"] = args.val_tokens
    if args.mbs is not None:
        overrides["mbs"] = args.mbs

    if args.attn_types:
        unknown = sorted(set(args.attn_types) - set(ALL_ATTN_TYPES))
        assert not unknown, f"unknown attn_type(s): {unknown}"
    configs = expand_grid(attn_types=args.attn_types, **overrides)
    if args.attn_types is None:
        assert len(configs) == len(ATTN_TYPES) * len(REG_MODES) * 2 * 2 * 2, f"grid size {len(configs)}"
    ids = [c.run_id for c in configs]
    assert len(set(ids)) == len(ids), "duplicate run_id in grid"
    print(f"{len(configs)} configs (attn_types={args.attn_types or ATTN_TYPES})")
    if args.attn_types and "wall" in args.attn_types:
        print("  note: wall is diagnostic-only; exclude from fair comparison dashboards/tables")

    def _write(cfgs, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for c in cfgs:
            d = c.to_dict()
            if args.max_steps is not None:
                d["max_steps"] = args.max_steps      # consumed by ablation.train for smoke runs
            with open(os.path.join(out_dir, f"{c.run_id}.json"), "w") as f:
                json.dump(d, f, indent=2)

    if args.shards > 1:
        shards = shard_configs(configs, args.shards)
        for i, sc in enumerate(shards):
            sub = os.path.join(args.out, f"shard{i}")
            _write(sc, sub)
            mem_on = sum(1 for c in sc if c.memory)
            print(f"  shard{i}: {len(sc)} configs ({mem_on} mem-on) -> {sub}/")
        print(f"\nWrote {len(configs)} configs into {args.shards} shards under {args.out}/")
        print("Launch one shard per GPU/host, e.g.:")
        for i in range(args.shards):
            print(f"  FLA_CUSTOM_OP=1 python -m ablation.run_worker "
                  f"--config_dir {args.out}/shard{i} --log_dir ablation/logs --gpu 0")
    else:
        _write(configs, args.out)
        print(f"Wrote {len(configs)} configs to {args.out}/")
        for a in (args.attn_types or ATTN_TYPES):
            n = sum(1 for c in configs if c.attn_type == a)
            print(f"  attn_type={a:<6} {n} configs")
        print(f"  reg_modes={REG_MODES}")
        print(f"  distractor/memory/window each in {{off,on}}")


if __name__ == "__main__":
    main()
