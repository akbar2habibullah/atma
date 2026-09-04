"""CPU-safe validation for manifests and resolved experiment configs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_configs(root: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*/*.json"))]


def validate(config_root: Path) -> list[str]:
    errors: list[str] = []
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    configs = load_configs(config_root)
    ids = [c.get("run_id") for c in configs]
    if len(ids) != len(set(ids)):
        errors.append("duplicate run_id in generated configs")
    expected = {"replication": 4, "polar_components": 5, "baseline_pilots": 3, "baseline_scaled": 3}
    for group, count in expected.items():
        actual = sum(c.get("experiment_group") == group for c in configs)
        if actual != count:
            errors.append(f"{group}: expected {count} configs, found {actual}")

    pairs: dict[tuple[str, int], set[str]] = {}
    for c in configs:
        if c.get("experiment_group") == "replication":
            pairs.setdefault((c["comparison_group"], c["init_seed"]), set()).add(c["attn_type"])
    if sorted(pairs.values(), key=sorted) != [{"nope", "polar"}, {"nope", "polar"}]:
        errors.append(f"replication pairing invalid: {pairs}")

    fixed = sum(c["declared_tokens"] for c in configs if c.get("enabled"))
    # Scaled promotion adds TDA and exactly one of Mamba-3/GDN-2.
    ceiling = fixed + 20_000_000_000
    if ceiling != manifest["token_ceiling"]:
        errors.append(f"token ceiling mismatch: configs imply {ceiling}, manifest says {manifest['token_ceiling']}")

    for c in configs:
        if c.get("baseline_family") == "external" and c.get("parameter_count_approved"):
            target = c["parameter_count_target"]
            tol = c["parameter_tolerance_frac"]
            actual = c.get("resolved_num_params")
            if actual is None or abs(actual - target) / target > tol:
                errors.append(f"{c['run_id']}: invalid parameter-count approval")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, default=ROOT / "configs")
    args = parser.parse_args()
    errors = validate(args.configs)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"plan valid: {len(load_configs(args.configs))} configs, 68B-token ceiling")


if __name__ == "__main__":
    main()

