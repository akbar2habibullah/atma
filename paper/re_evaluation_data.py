"""Shared loaders for untouched and retention-capped evaluation artifacts.

All paper and web figures use this module so that plotted values are derived
from the archived benchmark matrices rather than duplicated by hand.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_MATRIX = ROOT / "benchmarks" / "logs" / "atma_10b" / "benchmark_matrix.json"
RE_EVALUATION = ROOT / "gamma_diagnostics" / "results" / "re_evaluation" / "run-summary.json"
BABILONG_ROOT = ROOT / "benchmarks" / "logs" / "babilong_2k_ft" / "hub"

MODELS = ("nope", "polar", "rope")
REFERENCE_MODELS = ("raven_native", "atma_raven_titans")
ALL_MODELS = MODELS + REFERENCE_MODELS
LENGTHS = ("2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k")
BABI_LENGTHS = ("0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k")
DEPTHS = ("0.1", "0.5", "0.9")
DATASETS = ("finepdfs", "pg19", "proof_pile")
DATASET_IDS = {
    "finepdfs": "codelion/finepdfs-1B",
    "pg19": "emozilla/pg19",
    "proof_pile": "hoskinson-center/proof-pile",
}
TASK_METRICS = {
    "lambada": "accuracy",
    "hellaswag": "accuracy_norm",
    "piqa": "accuracy_norm",
    "winogrande": "accuracy",
    "arc_easy": "accuracy_norm",
    "arc_challenge": "accuracy_norm",
    "openbookqa": "accuracy_norm",
    "boolq": "accuracy",
}


def _mean(values):
    values = list(values)
    return sum(values) / len(values)


def baseline_rows():
    return json.loads(BASELINE_MATRIX.read_text(encoding="utf-8"))["rows"]


def baseline_haystack_retrieval(metric="token_accuracy", models=ALL_MODELS):
    """Full-run task/depth means; smoke measurements never enter paper tables."""
    grouped = defaultdict(dict)
    for row in baseline_rows():
        if row.get("benchmark") != "retrieval" or row.get("metric") != metric:
            continue
        if "/smoke_" in row.get("source_log", "").replace("\\", "/"):
            continue
        if row.get("model") not in models or row.get("suite") not in ("synthetic", "real"):
            continue
        key = (row["model"], row["suite"], row["length"])
        cell = (row["task"], str(row["depth"]))
        if cell in grouped[key]:
            raise ValueError(f"duplicate full retrieval cell: {key}, {cell}")
        grouped[key][cell] = float(row["value"])
    expected = {(task, depth) for task in ("niah", "passkey") for depth in DEPTHS}
    result = {}
    for model in models:
        result[model] = {}
        for suite in ("synthetic", "real"):
            result[model][suite] = {}
            for length in LENGTHS:
                cells = grouped[(model, suite, length)]
                if set(cells) != expected:
                    raise ValueError(f"incomplete retrieval cells: {model}, {suite}, {length}")
                result[model][suite][length] = _mean(cells.values())
    return result


def capped_jobs():
    jobs = json.loads(RE_EVALUATION.read_text(encoding="utf-8"))["results"]
    return {(job["family"], job["model"], job["suite"]): job["result"] for job in jobs}


def baseline_retrieval(metric="token_accuracy", *, by_depth=False, models=MODELS):
    grouped = defaultdict(list)
    for row in baseline_rows():
        if row.get("benchmark") != "retrieval" or row.get("metric") != metric:
            continue
        if "/smoke_" in row.get("source_log", "") or row.get("model") not in models:
            continue
        grouped[(row["model"], row["length"], str(row["depth"]))].append(float(row["value"]))
    result = {model: {} for model in models}
    for model in models:
        for length in LENGTHS:
            depths = {depth: _mean(grouped[(model, length, depth)]) for depth in DEPTHS}
            result[model][length] = depths if by_depth else _mean(depths.values())
    return result


def capped_retrieval(metric="token_accuracy", *, by_depth=False):
    key = "results" if metric == "token_accuracy" else "exact_results"
    jobs = capped_jobs()
    result = {model: {} for model in MODELS}
    for model in MODELS:
        for length in LENGTHS:
            depths = {}
            for depth in DEPTHS:
                values = []
                for suite in ("synthetic", "real"):
                    payload = jobs[("retrieval", model, suite)][key]
                    for task in ("niah", "passkey"):
                        values.append(float(payload[task][length][depth]))
                depths[depth] = _mean(values)
            result[model][length] = depths if by_depth else _mean(depths.values())
    return result


def baseline_babilong(models=MODELS):
    result = {}
    for model in models:
        payload = json.loads((BABILONG_ROOT / model / "babilong_full_eval_result.json").read_text(encoding="utf-8"))
        result[model] = {length: float(payload["macro_average"][length]) for length in BABI_LENGTHS}
    return result


def capped_babilong():
    jobs = capped_jobs()
    return {
        model: {length: float(jobs[("babilong", model, "heldout")]["macro_average"][length]) for length in BABI_LENGTHS}
        for model in MODELS
    }


def baseline_longdoc(models=MODELS):
    grouped = defaultdict(list)
    id_to_name = {value: key for key, value in DATASET_IDS.items()}
    for row in baseline_rows():
        if row.get("benchmark") != "longdoc" or row.get("metric") != "bits_per_byte":
            continue
        if row.get("model") not in models:
            continue
        grouped[(row["model"], id_to_name[row["dataset"]], row["length"])].append(float(row["value"]))
    return {
        model: {
            dataset: {length: _mean(grouped[(model, dataset, length)]) for length in LENGTHS}
            for dataset in DATASETS
        }
        for model in models
    }


def capped_longdoc():
    jobs = capped_jobs()
    return {
        model: {
            dataset: {
                length: float(jobs[("longdoc", model, "fixed-target")]["results"][dataset]["lengths"][length]["bits_per_byte"])
                for length in LENGTHS
            }
            for dataset in DATASETS
        }
        for model in MODELS
    }


def mean_longdoc(values, models=None):
    models = tuple(values) if models is None else models
    return {
        model: {length: _mean(values[model][dataset][length] for dataset in DATASETS) for length in LENGTHS}
        for model in models
    }


def baseline_downstream(models=MODELS):
    indexed = defaultdict(list)
    for row in baseline_rows():
        if row.get("benchmark") != "base" or row.get("model") not in models:
            continue
        indexed[(row["model"], row["task"], row["metric"])].append(float(row["value"]))
    return {
        model: {
            task: 100.0 * _mean(indexed[(model, task, metric)])
            for task, metric in TASK_METRICS.items()
        }
        for model in models
    }


def capped_downstream():
    jobs = capped_jobs()
    return {
        model: {
            task: 100.0 * float(jobs[("base", model, "zero-shot")]["results"][task][metric])
            for task, metric in TASK_METRICS.items()
        }
        for model in MODELS
    }
