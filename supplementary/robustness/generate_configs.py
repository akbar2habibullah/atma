"""Generate resolved configs for the 68B-token robustness experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ablation.config_schema import RunConfig as PilotRunConfig
from scaled_ablation.config_schema import RunConfig as ScaledRunConfig


ROOT = Path(__file__).resolve().parent
EVAL = json.loads((ROOT / "eval_manifest.json").read_text(encoding="utf-8"))
DEPENDENCIES = json.loads((ROOT / "dependencies.json").read_text(encoding="utf-8"))


def _seed_fields(seed: int) -> dict:
    return {
        "seed": seed,
        "init_seed": seed,
        "data_seed": seed,
        "eval_seed": int(EVAL["eval_seed"]),
        "deterministic_algorithms": False,
    }


def _write(group: str, cfg: dict, out: Path):
    target = out / group
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{cfg['run_id']}.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def _atma_cfg(*, scaled: bool, attn_type: str, run_id: str, seed: int, group: str) -> dict:
    cls = ScaledRunConfig if scaled else PilotRunConfig
    cfg = cls(attn_type=attn_type, reg_mode="baseline", distractor=False, memory=True, window=False).to_dict()
    cfg.update(
        run_id=run_id,
        runner="scaled_ablation.train" if scaled else "ablation.train",
        experiment_group=group,
        comparison_group=group,
        declared_tokens=10_000_000_000 if scaled else 1_000_000_000,
        enabled=True,
        **_seed_fields(seed),
    )
    return cfg


def _external_cfg(arch: str, *, scaled: bool, run_id: str, seed: int, enabled: bool) -> dict:
    spec = EVAL["scaled" if scaled else "pilot"]
    tda = arch == "tda_hybrid"
    hidden = 1024 if tda else 960  # provisional; GPU calibration must approve the exact count
    head_dim = 128 if tda else 64
    cfg = {
        "run_id": run_id,
        "runner": "external_baselines.train",
        "baseline_family": "external",
        "arch_type": arch,
        "attn_type": arch,
        "comparison_group": "external_scaled" if scaled else "external_pilot_1b",
        "experiment_group": "baseline_scaled" if scaled else "baseline_pilots",
        "reg_mode": "baseline",
        "distractor": False,
        "memory": tda,
        "window": False,
        "vocab_size": 50304,
        "hidden_size": hidden,
        "num_hidden_layers": 16,
        "head_dim": head_dim,
        "conv_kernel_size": 3,
        "seq_len": 2048,
        "num_chunks": 99 if scaled else 10,
        "batch_size": 524288,
        "val_tokens": 2097152,
        "mbs": 4,
        "cooldown_frac": 0.7,
        "sketch_dim": 64,
        "sigr_alpha": 0.0,
        "dist_align_loss_weight": 0.0,
        "optimizer": "adamw_external",
        "adamw_lr": 0.0003,
        "adamw_lr_min_frac": 0.1,
        "adamw_warmup_frac": 0.05,
        "adamw_beta1": 0.9,
        "adamw_beta2": 0.95,
        "adamw_eps": 1e-15,
        "adamw_weight_decay": 0.1,
        "skip_nan_inf": True,
        "compile_model": False,
        "mem_enabled": tda,
        "mem_chunk": 128,
        "mem_gamma_bias": 3.9,
        "mem_beta_bias": 0.0,
        "mem_kernel": "auto",
        "clean_dataset": spec["clean_dataset"],
        "val_data": "finewebedu10B/finewebedu_val_*.bin",
        "eval_lengths": spec["lengths"],
        "needle_distances": spec["lengths"],
        "num_eval_docs": spec["num_eval_docs"],
        "num_needle_trials": spec["num_needle_trials"],
        "needle_val_len": spec["needle_value_tokens"],
        "mamba3_state_size": 128,
        "mamba3_expand": 2,
        "mamba3_head_dim": 64,
        "mamba3_n_groups": 1,
        "mamba3_rope_fraction": 0.5,
        "mamba3_mimo": False,
        "mamba3_chunk_size": 64,
        "gdn2_expand_v": 1.0,
        "gdn2_short_conv": True,
        "gdn2_allow_neg_eigval": False,
        "gdn2_conv_size": 4,
        "tda_beta": 1.0,
        "tda_lambda_init": 0.5,
        "tda_relu_power": 2.0,
        "tda_source_dir": "third_party/TDA",
        "parameter_count_target": 378_200_000,
        "parameter_tolerance_frac": 0.05,
        "parameter_count_approved": False,
        "resolved_num_params": None,
        "dependency_commits": {name: dep["commit"] for name, dep in DEPENDENCIES.items()},
        "declared_tokens": 10_000_000_000 if scaled else 1_000_000_000,
        "enabled": enabled,
        **_seed_fields(seed),
    }
    return cfg


def generate(out: Path):
    for pair, seed in (("seed1", 202701), ("seed2", 202702)):
        for attn in ("polar", "nope"):
            _write(
                "replication",
                _atma_cfg(
                    scaled=True, attn_type=attn,
                    run_id=f"repl_{pair}_{attn}", seed=seed, group="replication",
                ),
                out,
            )

    variants = ["full", "direction_only", "constant_magnitude", "fixed_null", "fixed_temperature"]
    for variant in variants:
        cfg = _atma_cfg(
            scaled=False, attn_type="polar", run_id=f"polar_component_{variant}",
            seed=202703, group="polar_components",
        )
        cfg["polar_variant"] = variant
        _write("polar_components", cfg, out)

    for arch in ("tda_hybrid", "mamba3_native", "gdn2_native"):
        _write(
            "baseline_pilots",
            _external_cfg(arch, scaled=False, run_id=f"pilot_{arch}", seed=202704, enabled=True),
            out,
        )
        _write(
            "baseline_scaled",
            _external_cfg(arch, scaled=True, run_id=f"scaled_{arch}", seed=202701, enabled=False),
            out,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "configs")
    parser.add_argument("--clean", action="store_true", help="remove stale JSON configs under --out first")
    args = parser.parse_args()
    if args.clean and args.out.exists():
        for path in args.out.glob("*/*.json"):
            path.unlink()
    generate(args.out)
    print(f"generated robustness configs under {args.out}")


if __name__ == "__main__":
    main()
