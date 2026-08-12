"""Build the self-contained Stage I ablation dashboard.

The maintainable source lives in ``pages/dashboard_template.html`` and ``pages/assets``.
This command embeds the parsed experiment payload, CSS, and JavaScript so the generated
dashboard still opens offline and can be deployed as a plain static file.

Examples::

    python -m ablation.build_dashboard \
        --results ablation/results.json \
        --out pages/dashboard.html

    python -m ablation.build_dashboard \
        --log-dir ablation/logs raven_baseline/logs \
        --out pages/dashboard.html
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from ablation.config_schema import ATTN_TYPES, EVAL_LENGTHS, REG_MODES, expand_grid
from ablation.parse_logs import parse_log


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "pages" / "dashboard_template.html"
DEFAULT_OUTPUT = ROOT / "pages" / "dashboard.html"
SITE_CSS = ROOT / "pages" / "assets" / "site.css"
DASHBOARD_CSS = ROOT / "pages" / "assets" / "dashboard.css"
DASHBOARD_JS = ROOT / "pages" / "assets" / "dashboard.js"


def metric_catalog() -> list[dict[str, str]]:
    """Return dashboard metrics in a paper-oriented default order."""
    catalog = [
        {"name": "needle_acc_65536", "label": "teacher-forced needle accuracy @64K (%)", "dir": "higher"},
        {"name": "clean_ppl_65536", "label": "clean perplexity @64K (nats)", "dir": "lower"},
        {"name": "final_val_loss", "label": "final validation loss", "dir": "lower"},
        {"name": "needle_acc_wavg", "label": "needle accuracy · length weighted (%)", "dir": "higher"},
        {"name": "clean_ppl_wavg", "label": "clean perplexity · length weighted", "dir": "lower"},
        {"name": "junk_ppl_wavg", "label": "junk perplexity · length weighted", "dir": "lower"},
        {"name": "perf_index", "label": "exploratory performance index", "dir": "higher"},
        {"name": "eff_index", "label": "exploratory performance + MFU index", "dir": "higher"},
    ]
    for length in EVAL_LENGTHS:
        name = f"needle_acc_{length}"
        if name != "needle_acc_65536":
            catalog.append({"name": name, "label": f"teacher-forced needle accuracy @{length:,} (%)", "dir": "higher"})
    for prefix, label in (("clean_ppl", "clean perplexity"), ("junk_ppl", "junk perplexity")):
        for length in EVAL_LENGTHS:
            name = f"{prefix}_{length}"
            if name != "clean_ppl_65536":
                catalog.append({"name": name, "label": f"{label} @{length:,} (nats)", "dir": "lower"})
    catalog.extend(
        [
            {"name": "mfu_final", "label": "training MFU (%)", "dir": "higher"},
            {"name": "train_elapsed_s", "label": "training wall time (seconds)", "dir": "lower"},
        ]
    )
    return catalog


def load_records(results: Path | None, log_dirs: list[Path]) -> list[dict]:
    if results is not None:
        return json.loads(results.read_text(encoding="utf-8"))

    paths: list[str] = []
    for log_dir in log_dirs:
        paths.extend(glob.glob(str(log_dir / "*.log")))
    return [parse_log(path) for path in sorted(paths)]


def axis_values(records: list[dict], axis: str, defaults: tuple | list) -> list:
    values = list(defaults)
    seen = set(values)
    for record in records:
        value = record.get(axis)
        if isinstance(value, bool):
            value = int(value)
        if value is not None and value not in seen:
            values.append(value)
            seen.add(value)
    return values


def build_payload(records: list[dict]) -> dict:
    expected_grid = [config.run_id for config in expand_grid()]
    expected = sorted(set(expected_grid) | {record["run_id"] for record in records})
    return {
        "records": records,
        "catalog": metric_catalog(),
        "axes": ["attn_type", "reg_mode", "distractor", "memory", "window"],
        "axis_values": {
            "attn_type": axis_values(records, "attn_type", ATTN_TYPES),
            "reg_mode": axis_values(records, "reg_mode", REG_MODES),
            "distractor": axis_values(records, "distractor", [0, 1]),
            "memory": axis_values(records, "memory", [0, 1]),
            "window": axis_values(records, "window", [0, 1]),
        },
        "expected": expected,
        "eval_lengths": list(EVAL_LENGTHS),
        "base_len": min(EVAL_LENGTHS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static ATMA ablation dashboard.")
    parser.add_argument("--log-dir", "--log_dir", dest="log_dirs", nargs="+", default=[ROOT / "ablation" / "logs"])
    parser.add_argument("--results", type=Path, help="Use an existing results.json instead of parsing logs.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = load_records(args.results, [Path(path) for path in args.log_dirs])
    payload = build_payload(records)
    template = args.template.read_text(encoding="utf-8")
    if "/*DATA*/" not in template:
        raise ValueError(f"dashboard template has no /*DATA*/ placeholder: {args.template}")

    # Prevent data strings from terminating the JSON script element.
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    html = template.replace("/*DATA*/", encoded)
    html = html.replace(
        '<link rel="stylesheet" href="assets/site.css">',
        f"<style>\n{SITE_CSS.read_text(encoding='utf-8')}\n</style>",
    )
    html = html.replace(
        '<link rel="stylesheet" href="assets/dashboard.css">',
        f"<style>\n{DASHBOARD_CSS.read_text(encoding='utf-8')}\n</style>",
    )
    html = html.replace(
        '<script src="assets/dashboard.js"></script>',
        f"<script>\n{DASHBOARD_JS.read_text(encoding='utf-8')}\n</script>",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")

    done = sum(record.get("status") == "done" for record in records)
    print(f"wrote {args.out} ({len(records)} records, {done} complete, grid={len(payload['expected'])})")


if __name__ == "__main__":
    main()
