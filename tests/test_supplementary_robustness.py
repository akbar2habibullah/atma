import json
import subprocess
import sys

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
