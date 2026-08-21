import json

from supplementary.robustness.generate_configs import generate
from supplementary.robustness.validate_plan import validate


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

