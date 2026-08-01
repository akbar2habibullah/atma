"""Aggregate all structured benchmark logs into a tidy JSON/CSV matrix."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path


MARKERS = {
    "retrieval": "===RETRIEVAL_RESULTS_JSON===",
    "babilong": "===BABILONG_RESULTS_JSON===",
    "base": "===BASE_RESULTS_JSON===",
    "longdoc": "===LONGDOC_RESULTS_JSON===",
    "serving": "===SERVING_RESULTS_JSON===",
}


def _extract(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    found = []
    for benchmark, marker in MARKERS.items():
        if marker not in text or "===END===" not in text.rsplit(marker, 1)[1]:
            continue
        block = text.rsplit(marker, 1)[1].split("===END===", 1)[0].strip()
        try:
            result = json.loads(block)
        except json.JSONDecodeError:
            continue
        found.append((benchmark, result))
    return found


def _model_name(result, path):
    cfg = result.get("model_config") or {}
    name = cfg.get("arch_type") or cfg.get("attn_type")
    if name:
        return name
    for candidate in ("atma_raven_titans", "raven_native", "polar", "nope", "rope"):
        if candidate in path.stem:
            return candidate
    return "unknown"


def _row(model, benchmark, source, **values):
    return {"model": model, "benchmark": benchmark, "source_log": str(source), **values}


def _flatten(benchmark, result, source):
    model = _model_name(result, source)
    rows = []
    if benchmark == "retrieval":
        suite = "synthetic" if result.get("haystack") == "synthetic-filler" else "real"
        for task, lengths in result.get("results", {}).items():
            for length, depths in lengths.items():
                for depth, value in depths.items():
                    rows.append(_row(
                        model, benchmark, source, suite=suite, task=task, dataset=result.get("haystack"),
                        length=length, depth=depth, metric="exact_match", value=value,
                        samples=result.get("num_samples"),
                    ))
        for cell in result.get("oom_cells", []):
            rows.append(_row(
                model, benchmark, source, suite=suite,
                task=cell.get("kind") or cell.get("task"),
                dataset=result.get("haystack"), length=cell.get("length"),
                depth=cell.get("depth"), metric="oom", value=True,
                samples=result.get("num_samples"),
            ))
    elif benchmark == "base":
        for task, metrics in result.get("results", {}).items():
            for metric in ("accuracy", "accuracy_norm", "target_nll", "target_perplexity"):
                if metrics.get(metric) is not None:
                    rows.append(_row(
                        model, benchmark, source, suite="zero_shot", task=task, dataset=None,
                        length=None, depth=None, metric=metric, value=metrics[metric],
                        samples=metrics.get("samples"),
                    ))
    elif benchmark == "longdoc":
        for dataset, dataset_result in result.get("results", {}).items():
            for length, metrics in dataset_result.get("lengths", {}).items():
                for metric in ("nll_nats_per_token", "perplexity", "bits_per_byte"):
                    if metrics.get(metric) is not None:
                        rows.append(_row(
                            model, benchmark, source, suite="fixed_target", task=None,
                            dataset=dataset_result.get("dataset_id", dataset),
                            length=length, depth=None, metric=metric,
                            value=metrics[metric], samples=metrics.get("documents"),
                        ))
                if metrics.get("oom"):
                    rows.append(_row(
                        model, benchmark, source, suite="fixed_target", task=None,
                        dataset=dataset_result.get("dataset_id", dataset),
                        length=length, depth=None, metric="oom", value=True,
                        samples=metrics.get("documents"),
                    ))
    elif benchmark == "serving":
        for length, metrics in result.get("results", {}).items():
            for source_key, metric in (
                ("prefill_tokens_per_s", "prefill_tokens_per_s"),
                ("decode_tokens_per_s", "decode_tokens_per_s"),
                ("peak_allocated_bytes", "peak_allocated_bytes"),
                ("peak_reserved_bytes", "peak_reserved_bytes"),
                ("oom", "oom"),
            ):
                if source_key in metrics:
                    rows.append(_row(
                        model, benchmark, source, suite="generation", task=None, dataset=None,
                        length=length, depth=None, metric=metric, value=metrics[source_key],
                        samples=metrics.get("samples"),
                    ))
        if result.get("max_successful_context_tokens") is not None:
            rows.append(_row(
                model, benchmark, source, suite="generation", task=None, dataset=None,
                length=None, depth=None, metric="max_successful_context_tokens",
                value=result["max_successful_context_tokens"], samples=None,
            ))
    elif benchmark == "babilong":
        for task, lengths in result.get("results", {}).items():
            for length, value in lengths.items():
                rows.append(_row(
                    model, benchmark, source, suite="adapted_reasoning", task=task,
                    dataset=result.get("dataset_id"), length=length, depth=None,
                    metric="accuracy", value=value,
                    samples=result.get("counts", {}).get(task, {}).get(length),
                ))
    return rows


def aggregate(log_dir: Path):
    rows = []
    sources = []
    latest = {}
    for path in sorted(log_dir.rglob("*.log")):
        if path.name.endswith(".console.log"):
            continue
        extracted = _extract(path)
        if not extracted:
            continue
        logical_stem = re.sub(r"\.attempt-\d+$", "", path.stem)
        logical_path = path.with_name(logical_stem + path.suffix)
        previous = latest.get(logical_path)
        if previous is None or path.stat().st_mtime > previous[0].stat().st_mtime:
            latest[logical_path] = (path, extracted)

    for path, extracted in sorted(
        latest.values(), key=lambda item: str(item[0])
    ):
        for benchmark, result in extracted:
            sources.append({"path": str(path), "benchmark": benchmark})
            rows.extend(_flatten(benchmark, result, path))
    rows.sort(key=lambda row: (
        row["model"], row["benchmark"], str(row.get("suite")), str(row.get("task")),
        str(row.get("dataset")), str(row.get("length")), str(row.get("depth")), row["metric"],
    ))
    return {
        "schema_version": 1,
        "generated_at_unix": int(time.time()),
        "log_dir": str(log_dir),
        "sources": sources,
        "rows": rows,
    }


def write_outputs(result, json_path: Path, csv_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    fields = [
        "model", "benchmark", "suite", "task", "dataset", "length", "depth",
        "metric", "value", "samples", "source_log",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["rows"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Aggregate ATMA structured benchmark logs.")
    ap.add_argument("--log_dir", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--out_csv", type=Path, default=None)
    args = ap.parse_args(argv)
    log_dir = args.log_dir.resolve()
    result = aggregate(log_dir)
    json_path = args.out_json or log_dir / "benchmark_matrix.json"
    csv_path = args.out_csv or log_dir / "benchmark_matrix.csv"
    write_outputs(result, json_path, csv_path)
    print(f"[aggregate] {len(result['rows'])} rows -> {json_path}, {csv_path}")


if __name__ == "__main__":
    main()
