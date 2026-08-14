#!/usr/bin/env python3
"""Re-benchmark the matched attention variants with a runtime gamma ceiling.

The promoted base checkpoints are used for zero-shot downstream, retrieval, and
fixed-target long-document BPB tasks.  The separately fine-tuned BABILong
checkpoints are used for BABILong.  Every final checkpoint is inspected
independently, the largest parameter-only gamma layer-head is selected, and a
reversible runtime ceiling is installed by ``benchmarks.run``.  Checkpoint
tensors are never rewritten.

By default only the clamped condition is run because the repository already
contains the corresponding baseline benchmark logs.  Pass ``--paired`` to
rerun baseline and clamped conditions in the same environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.babilong import DEFAULT_DATASET_ID, EVAL_LENGTHS, TASKS
from benchmarks.base_tasks import BASE_TASK_SPECS
from benchmarks.longdoc import LONGDOC_SPECS
from benchmarks.run_pipeline import _extract_result, _is_complete
from gamma_diagnostics.clamp import FORMAT_VERSION


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MANIFEST = ROOT / "benchmarks" / "logs" / "atma_10b" / "checkpoint_manifest.json"
DEFAULT_BABILONG_MANIFEST = (
    ROOT / "benchmarks" / "logs" / "babilong_2k_ft" / "hub_sources.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "gamma_diagnostics" / "results" / "re_evaluation"
DEFAULT_MODELS = ("nope", "polar", "rope")
DEFAULT_LENGTHS = ("2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k")
WEIGHT_NAMES = ("weights.pt", "model.pt", "pytorch_model.bin")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument(
        "--benchmarks", nargs="+", choices=("base", "retrieval", "longdoc", "babilong"),
        default=("base", "retrieval", "longdoc", "babilong"),
    )
    parser.add_argument("--max-half-life", type=float, default=256.0)
    parser.add_argument("--num-target-heads", type=int, default=1)
    parser.add_argument(
        "--paired", action="store_true",
        help="rerun the untouched baseline before each clamped benchmark",
    )
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--babilong-manifest", type=Path, default=DEFAULT_BABILONG_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hf-cache", type=Path, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--base-tasks", nargs="+", choices=tuple(BASE_TASK_SPECS),
                        default=tuple(BASE_TASK_SPECS))
    parser.add_argument("--base-limit", type=int, default=None)
    parser.add_argument("--base-batch-size", type=int, default=8)
    parser.add_argument("--retrieval-suites", nargs="+", choices=("synthetic", "real"),
                        default=("synthetic", "real"))
    parser.add_argument("--retrieval-tasks", nargs="+", choices=("passkey", "niah"),
                        default=("passkey", "niah"))
    parser.add_argument("--retrieval-lengths", nargs="+", default=DEFAULT_LENGTHS)
    parser.add_argument("--retrieval-depths", nargs="+", type=float, default=(0.1, 0.5, 0.9))
    parser.add_argument("--retrieval-samples", type=int, default=50)
    parser.add_argument("--retrieval-value-tokens", type=int, default=5)
    parser.add_argument("--haystack", default="codelion/finepdfs-1B")
    parser.add_argument("--longdoc-datasets", nargs="+", choices=tuple(LONGDOC_SPECS),
                        default=tuple(LONGDOC_SPECS))
    parser.add_argument("--longdoc-lengths", nargs="+", default=DEFAULT_LENGTHS)
    parser.add_argument("--target-tokens", type=int, default=256)
    parser.add_argument("--num-docs", type=int, default=8)
    parser.add_argument("--max-scan", type=int, default=100000)
    parser.add_argument("--babilong-tasks", nargs="+", choices=TASKS[:10], default=TASKS[:10])
    parser.add_argument("--babilong-lengths", nargs="+", choices=EVAL_LENGTHS,
                        default=EVAL_LENGTHS)
    parser.add_argument("--babilong-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--execute", action="store_true",
        help="execute jobs; otherwise resolve checkpoints and write/print the plan only",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required manifest not found: {path}") from exc


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _source_record(manifest: dict, model: str, family: str) -> dict:
    try:
        record = manifest["models"][model]
    except KeyError as exc:
        raise KeyError(f"{family} manifest has no model entry {model!r}") from exc
    repo_id = record.get("repo_id")
    revision = record.get("resolved_revision") or record.get("revision")
    if not repo_id or not revision:
        raise ValueError(f"{family}/{model} manifest entry lacks repo_id or pinned revision")
    return {"repo_id": repo_id, "revision": revision}


def _resolve_checkpoint(source: dict, cache_dir: Path | None, offline: bool) -> tuple[Path, Path]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install checkpoint support with: pip install huggingface_hub") from exc

    print(f"[gamma-rebench] resolve {source['repo_id']}@{source['revision']}", flush=True)
    root = Path(snapshot_download(
        repo_id=source["repo_id"],
        revision=source["revision"],
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=offline,
        allow_patterns=[
            "config.json", "run_config.json", "tokenizer.json",
            "finetune_manifest.json", *WEIGHT_NAMES,
        ],
    )).resolve()
    weights = next((root / name for name in WEIGHT_NAMES if (root / name).is_file()), None)
    if weights is None or not (root / "config.json").is_file():
        raise FileNotFoundError(f"checkpoint at {root} lacks weights or config.json")
    return root, weights


def _select_targets(rows: list[dict], count: int) -> list[dict]:
    if count <= 0:
        raise ValueError("--num-target-heads must be positive")
    if len(rows) < count:
        raise ValueError(f"requested {count} target heads but checkpoint contains {len(rows)}")
    ranked = sorted(
        rows,
        key=lambda row: (row["total_zero_input_logit"], -row["layer"], -row["head"]),
        reverse=True,
    )
    return ranked[:count]


def _clamp_spec(source: dict, targets: list[dict], max_half_life: float) -> dict:
    return {
        "format": FORMAT_VERSION,
        "label": f"hl-{max_half_life:g}",
        "cap_source": {
            "kind": "absolute_half_life",
            "half_life_tokens": float(max_half_life),
        },
        "selection": {
            "kind": "largest_parameter_only_zero_input_logit",
            "num_target_heads": len(targets),
            "checkpoint_repo_id": source["repo_id"],
            "checkpoint_revision": source["revision"],
        },
        "targets": [
            {
                "layer": int(row["layer"]),
                "heads": [int(row["head"])],
                "max_half_life_tokens": float(max_half_life),
            }
            for row in targets
        ],
    }


def _job_id(job: dict) -> str:
    stable = {key: value for key, value in job.items() if key not in ("output", "console")}
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def _dataset_revision(manifest: dict, dataset_id: str) -> str:
    try:
        record = manifest["datasets"][dataset_id]
    except KeyError as exc:
        raise KeyError(f"base manifest has no pinned dataset {dataset_id!r}") from exc
    revision = record.get("resolved_revision") if isinstance(record, dict) else record
    if not revision:
        raise ValueError(f"base manifest has no pinned revision for {dataset_id!r}")
    return str(revision)


def _benchmark_variants(args, benchmark: str, base_manifest: dict,
                        babilong_manifest: dict) -> list[tuple[str, list[str]]]:
    if benchmark == "base":
        command = [
            "--benchmark", "base",
            "--tasks", *args.base_tasks,
            "--batch_size", str(args.base_batch_size),
            "--scoring_max_length", "2048",
            "--dataset_revisions", str(args.base_manifest.resolve()),
        ]
        if args.base_limit is not None:
            command.extend(("--limit", str(args.base_limit)))
        return [("zero-shot", command)]
    if benchmark == "retrieval":
        variants = []
        common = [
            "--benchmark", "retrieval",
            "--tasks", *args.retrieval_tasks,
            "--lengths", *args.retrieval_lengths,
            "--depths", *(str(depth) for depth in args.retrieval_depths),
            "--samples", str(args.retrieval_samples),
            "--seed", str(args.seed),
            "--retrieval_value_tokens", str(args.retrieval_value_tokens),
        ]
        for suite in args.retrieval_suites:
            extra = []
            if suite == "real":
                extra = [
                    "--haystack", args.haystack,
                    "--haystack_revision", _dataset_revision(base_manifest, args.haystack),
                ]
            variants.append((suite, [*common, *extra]))
        return variants
    if benchmark == "longdoc":
        return [("fixed-target", [
            "--benchmark", "longdoc",
            "--datasets", *args.longdoc_datasets,
            "--lengths", *args.longdoc_lengths,
            "--target_tokens", str(args.target_tokens),
            "--num_docs", str(args.num_docs),
            "--max_scan", str(args.max_scan),
            "--dataset_revisions", str(args.base_manifest.resolve()),
        ])]
    return [("heldout", [
            "--benchmark", "babilong",
            "--dataset", DEFAULT_DATASET_ID,
            "--dataset_revision", str(babilong_manifest["dataset_revision"]),
            "--tasks", *args.babilong_tasks,
            "--lengths", *args.babilong_lengths,
            "--row_start", "90", "--row_end", "100",
            "--samples", str(args.babilong_samples),
            "--seed", str(args.seed),
            "--babilong_backend", "direct",
            "--max_tokens", "16",
    ])]


def _run_job(job: dict, env: dict) -> dict:
    output = Path(job["output"])
    if _is_complete(output):
        print(f"[gamma-rebench] skip complete: {output}", flush=True)
        return {"status": "skipped_complete", "result": _extract_result(output)}

    output.parent.mkdir(parents=True, exist_ok=True)
    console = Path(job["console"])
    print(f"[gamma-rebench] run: {subprocess.list2cmdline(job['command'])}", flush=True)
    started = time.time()
    with console.open("w", encoding="utf-8", buffering=1) as stream:
        process = subprocess.Popen(
            job["command"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        returncode = process.wait()
    complete = _is_complete(output)
    return {
        "status": "complete" if returncode == 0 and complete else "failed",
        "returncode": returncode,
        "elapsed_s": round(time.time() - started, 1),
        "result": _extract_result(output) if complete else None,
    }


def main():
    args = _parse_args()
    if args.max_half_life <= 0:
        raise SystemExit("--max-half-life must be positive")
    if args.babilong_samples <= 0:
        raise SystemExit("--babilong-samples must be positive")
    if args.retrieval_samples <= 0 or args.retrieval_value_tokens <= 0:
        raise SystemExit("retrieval sample and value-token counts must be positive")
    if args.target_tokens <= 0 or args.num_docs <= 0 or args.max_scan <= 0:
        raise SystemExit("long-document evaluation counts must be positive")

    # Keep argument parsing and --help lightweight; Torch is only required once
    # checkpoint parameter inspection actually begins.
    from gamma_diagnostics.inspect_parameters import inspect_checkpoint

    base_manifest = _read_json(args.base_manifest)
    babilong_manifest = _read_json(args.babilong_manifest)
    manifests = {"base": base_manifest, "babilong": babilong_manifest}
    conditions = ("baseline", "clamped") if args.paired else ("clamped",)
    output_dir = args.output_dir.resolve()
    plan = {
        "schema_version": 1,
        "created_at_unix": int(time.time()),
        "max_half_life_tokens": args.max_half_life,
        "num_target_heads": args.num_target_heads,
        "models": list(args.models),
        "benchmarks": list(args.benchmarks),
        "conditions": list(conditions),
        "jobs": [],
    }

    checkpoint_cache = {}
    for benchmark in args.benchmarks:
        checkpoint_family = "babilong" if benchmark == "babilong" else "base"
        for model in args.models:
            cache_key = (checkpoint_family, model)
            if cache_key not in checkpoint_cache:
                source = _source_record(
                    manifests[checkpoint_family], model, checkpoint_family
                )
                checkpoint, weights = _resolve_checkpoint(
                    source, args.hf_cache, args.offline
                )
                rows = inspect_checkpoint(
                    f"{checkpoint_family}/{model}", weights, checkpoint / "config.json"
                )
                targets = _select_targets(rows, args.num_target_heads)
                checkpoint_dir = output_dir / "checkpoints" / checkpoint_family / model
                spec_path = checkpoint_dir / f"hl-{args.max_half_life:g}.gamma-clamp.json"
                parameters_path = checkpoint_dir / "gamma-parameters.json"
                _write_json(spec_path, _clamp_spec(source, targets, args.max_half_life))
                _write_json(parameters_path, rows)
                checkpoint_cache[cache_key] = (
                    source, checkpoint, rows, targets, spec_path
                )
            source, checkpoint, rows, targets, spec_path = checkpoint_cache[cache_key]
            model_dir = output_dir / benchmark / model

            print(
                f"[gamma-rebench] {benchmark}/{model}: target(s) "
                + ", ".join(
                    f"block {row['layer']} head {row['head']} "
                    f"(zero-input half-life={row['half_life_tokens']:.1f})"
                    for row in targets
                ),
                flush=True,
            )
            variants = _benchmark_variants(
                args, benchmark, base_manifest, babilong_manifest
            )
            for suite, benchmark_args in variants:
                for condition in conditions:
                    command = [
                        sys.executable, "-m", "benchmarks.run",
                        "--model", str(checkpoint), *benchmark_args,
                    ]
                    if condition == "clamped":
                        command.extend(("--gamma-clamp", str(spec_path)))
                    proto = {
                        "family": benchmark,
                        "suite": suite,
                        "model": model,
                        "condition": condition,
                        "source": source,
                        "command": command,
                    }
                    fingerprint = _job_id(proto)
                    output = model_dir / f"{benchmark}.{suite}.{condition}.{fingerprint}.log"
                    console = model_dir / (
                        f"{benchmark}.{suite}.{condition}.{fingerprint}.console.log"
                    )
                    command.extend(("--out", str(output)))
                    plan["jobs"].append({
                        **proto,
                        "command": command,
                        "output": str(output),
                        "console": str(console),
                        "clamp_spec": str(spec_path) if condition == "clamped" else None,
                    })

    plan_path = output_dir / "benchmark-plan.json"
    _write_json(plan_path, plan)
    print(f"[gamma-rebench] wrote plan: {plan_path}", flush=True)
    for job in plan["jobs"]:
        print(subprocess.list2cmdline(job["command"]), flush=True)

    if not args.execute:
        print("[gamma-rebench] dry run only; append --execute to launch all jobs", flush=True)
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("FLA_CUSTOM_OP", "1")
    state = {**plan, "started_at_unix": int(time.time()), "results": []}
    state_path = output_dir / "run-summary.json"
    _write_json(state_path, state)
    for job in plan["jobs"]:
        result = _run_job(job, env)
        state["results"].append({
            "family": job["family"], "model": job["model"],
            "suite": job["suite"], "condition": job["condition"],
            "output": job["output"], **result,
        })
        _write_json(state_path, state)
        if result["status"] == "failed":
            raise SystemExit(
                f"benchmark failed for {job['family']}/{job['model']}/{job['condition']}; "
                f"resume with the same command after inspecting {job['console']}"
            )
    state["finished_at_unix"] = int(time.time())
    _write_json(state_path, state)
    print(f"[gamma-rebench] all jobs complete: {state_path}", flush=True)


if __name__ == "__main__":
    main()
