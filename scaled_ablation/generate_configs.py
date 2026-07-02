"""Write the final scaled ablation config JSONs to a directory.

    python -m scaled_ablation.generate_configs                # -> scaled_ablation/configs/*.json (4)
    python -m scaled_ablation.generate_configs --out scaled_ablation/smoke --num_chunks 1 --val_tokens 524288 \
        --max_steps 3                                          # tiny smoke configs

Each file is <run_id>.json holding the full resolved RunConfig.
"""
import argparse
import json
import os

from scaled_ablation.config_schema import expand_grid, shard_configs, ATTN_TYPES, REG_MODES


def main():
    ap = argparse.ArgumentParser(description="Generate the scaled-ablation config files.")
    ap.add_argument("--out", default="scaled_ablation/configs", help="output directory for *.json")
    ap.add_argument("--attn_types", nargs="+", default=None,
                    help="restrict to these attn_type(s), e.g. `--attn_types wall` for the 40 wall cells")
    ap.add_argument("--shards", type=int, default=1,
                    help="split into N balanced subdirs out/shard{0..N-1}/ (one per GPU/host)")
    ap.add_argument("--num_chunks", type=int, default=None, help="override token budget (chunks)")
    ap.add_argument("--val_tokens", type=int, default=None, help="override validation token count")
    ap.add_argument("--mbs", type=int, default=None, help="override microbatch size")
    ap.add_argument("--num_eval_docs", type=int, default=None, help="override clean/junk eval docs")
    ap.add_argument("--num_needle_trials", type=int, default=None, help="override needle trials")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="cap training steps (smoke tests); stored as an extra key, honored by scaled_ablation.train")
    args = ap.parse_args()

    overrides = {}
    if args.num_chunks is not None:
        overrides["num_chunks"] = args.num_chunks
    if args.val_tokens is not None:
        overrides["val_tokens"] = args.val_tokens
    if args.mbs is not None:
        overrides["mbs"] = args.mbs
    if args.num_eval_docs is not None:
        overrides["num_eval_docs"] = args.num_eval_docs
    if args.num_needle_trials is not None:
        overrides["num_needle_trials"] = args.num_needle_trials

    configs = expand_grid(**overrides)
    assert len(configs) == len(ATTN_TYPES) * len(REG_MODES), f"grid size {len(configs)}"
    if args.attn_types:
        keep = set(args.attn_types)
        configs = [c for c in configs if c.attn_type in keep]
        assert configs, f"no configs for attn_types={args.attn_types}"
    ids = [c.run_id for c in configs]
    assert len(set(ids)) == len(ids), "duplicate run_id in grid"
    print(f"{len(configs)} configs (attn_types={args.attn_types or ATTN_TYPES})")

    def _write(cfgs, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for c in cfgs:
            d = c.to_dict()
            if args.max_steps is not None:
                d["max_steps"] = args.max_steps      # consumed by scaled_ablation.train for smoke runs
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
            print(f"  FLA_CUSTOM_OP=1 python -m scaled_ablation.run_worker "
                  f"--config_dir {args.out}/shard{i} --log_dir scaled_ablation/logs --gpu 0")
    else:
        _write(configs, args.out)
        print(f"Wrote {len(configs)} configs to {args.out}/")
        for a in ATTN_TYPES:
            n = sum(1 for c in configs if c.attn_type == a)
            print(f"  attn_type={a:<6} {n} configs")
        print(f"  reg_modes={REG_MODES}")
        print("  recipe=baseline reg, no distractor, memory on, no window")


if __name__ == "__main__":
    main()
