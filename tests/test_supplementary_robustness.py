import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from supplementary.robustness.evaluate_replications import (
    MANIFEST,
    MODELS,
    _rebenchmark_args,
)
from supplementary.robustness.generate_configs import generate
from supplementary.robustness.validate_plan import validate
from raven_baseline.train import _fill_runtime_defaults


def test_generated_robustness_plan_is_complete(tmp_path):
    generate(tmp_path)
    assert validate(tmp_path) == []
    configs = [json.loads(path.read_text()) for path in tmp_path.glob("*/*.json")]
    assert len(configs) == 15
    assert sum(c["declared_tokens"] for c in configs if c["enabled"]) == 48_000_000_000


def test_replication_pairs_share_seed_and_budget(tmp_path):
    generate(tmp_path)
    replication = [json.loads(path.read_text()) for path in (tmp_path / "replication").glob("*.json")]
    by_seed = {}
    for cfg in replication:
        by_seed.setdefault(cfg["init_seed"], []).append(cfg)
    assert len(by_seed) == 2
    for pair in by_seed.values():
        assert {c["attn_type"] for c in pair} == {"polar", "nope"}
        assert {c["data_seed"] for c in pair} == {pair[0]["init_seed"]}
        assert {c["declared_tokens"] for c in pair} == {10_000_000_000}


def test_scaled_external_candidates_start_disabled(tmp_path):
    generate(tmp_path)
    configs = [json.loads(path.read_text()) for path in (tmp_path / "baseline_scaled").glob("*.json")]
    assert {c["arch_type"] for c in configs} == {"tda_hybrid", "mamba3_native", "gdn2_native"}
    assert not any(c["enabled"] for c in configs)
    assert not any(c["parameter_count_approved"] for c in configs)


def test_external_configs_do_not_require_raven_head_fields(tmp_path):
    generate(tmp_path)
    for path in (tmp_path / "baseline_pilots").glob("*.json"):
        cfg = json.loads(path.read_text())
        resolved = _fill_runtime_defaults(cfg)
        assert resolved["optimizer"] == "adamw_external"
        assert "num_heads" not in resolved
        assert "num_kv_heads" not in resolved


def test_external_compile_and_dependency_matrix(tmp_path):
    generate(tmp_path)
    configs = [
        json.loads(path.read_text())
        for group in ("baseline_pilots", "baseline_scaled")
        for path in (tmp_path / group).glob("*.json")
    ]
    expected_dependencies = {
        "tda_hybrid": {"flash_linear_attention", "tda"},
        "mamba3_native": {"flash_linear_attention", "mamba"},
        "gdn2_native": {"flash_linear_attention"},
    }
    for cfg in configs:
        optimized = cfg["arch_type"] == "mamba3_native"
        assert cfg["compile_model"] is optimized
        assert cfg["external_custom_op"] is optimized
        assert cfg["gdn2_cuda_graph"] is (cfg["arch_type"] == "gdn2_native")
        assert cfg["tda_cuda_graph"] is (cfg["arch_type"] == "tda_hybrid")
        assert cfg["tda_tuned_kernel"] is (cfg["arch_type"] == "tda_hybrid")
        assert set(cfg["dependency_commits"]) == expected_dependencies[cfg["arch_type"]]


def test_worker_rejects_enabled_unapproved_external_config(tmp_path):
    config_root = tmp_path / "configs"
    generate(config_root)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "supplementary.robustness.run_worker",
            "--config_dir",
            str(config_root / "baseline_pilots"),
            "--log_dir",
            str(tmp_path / "logs"),
            "--state_dir",
            str(tmp_path / "state"),
            "--once",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "not parameter-count approved" in result.stderr


def test_replication_evaluation_scope_is_paired_bpb_and_retrieval_only(tmp_path):
    args = Namespace(
        models=MODELS,
        gpu="1",
        hf_cache=Path("/tmp/hf-cache"),
        output_dir=tmp_path,
        offline=True,
        execute=True,
    )
    command = _rebenchmark_args(args)
    assert command[command.index("--benchmarks") + 1:command.index("--base-manifest")] == [
        "retrieval", "longdoc",
    ]
    assert "base" not in command
    assert "babilong" not in command
    assert "--paired" in command
    assert command[command.index("--max-half-life") + 1] == "256"
    assert command[command.index("--base-manifest") + 1] == str(MANIFEST)
    assert "--execute" in command


def test_replication_benchmark_manifest_matches_training_configs():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    config_dir = MANIFEST.parent / "configs" / "replication"
    configs = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in config_dir.glob("*.json")
    }
    assert set(manifest["models"]) == set(configs) == set(MODELS)
    for run_id, record in manifest["models"].items():
        assert record["architecture"] == configs[run_id]["attn_type"]
        assert record["seed"] == configs[run_id]["seed"]
        assert len(record["resolved_revision"]) == 40
    assert manifest["protocol"]["benchmarks"] == ["retrieval", "longdoc"]
    assert manifest["protocol"]["conditions"] == [
        "baseline", "gamma_half_life_256",
    ]
    assert manifest["protocol"]["gamma_cap"]["max_half_life_tokens"] == 256
