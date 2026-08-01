"""Download and benchmark the five promoted 10B-token ATMA checkpoints.

The pipeline deliberately launches every benchmark job in a fresh subprocess. This releases
GPU allocations and compiler state between architectures, and ensures a failed/OOM cell cannot
contaminate the next model. Hugging Face ``main`` revisions are resolved once and pinned in
``checkpoint_manifest.json`` for reproducibility.

Examples:

    # Cheap correctness/scheduling check for every model.
    python -m benchmarks.run_pipeline --stage smoke --gpu 0

    # Full literal-retrieval pipeline (synthetic filler + real-text distractors).
    python -m benchmarks.run_pipeline --stage all --gpu 0

    # Resume only the matched attention ablation; completed fingerprints are skipped.
    python -m benchmarks.run_pipeline --stage full --models nope polar rope --gpu 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "benchmarks" / "logs" / "atma_10b"
WEIGHT_NAMES = ("weights.pt", "model.pt", "pytorch_model.bin")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    architecture: str
    comparison_group: str


MODEL_SPECS = {
    "nope": ModelSpec(
        "nope",
        "ChavyvAkvar/atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0",
        "nope",
        "matched_atma_muon",
    ),
    "polar": ModelSpec(
        "polar",
        "ChavyvAkvar/atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0",
        "polar",
        "matched_atma_muon",
    ),
    "rope": ModelSpec(
        "rope",
        "ChavyvAkvar/atma-10b-L40S-mbs16-rope__reg-baseline__distr-0__mem-1__win-0",
        "rope",
        "matched_atma_muon",
    ),
    "atma_raven_titans": ModelSpec(
        "atma_raven_titans",
        "ChavyvAkvar/atma-10b-L40S-mbs16-atma-raven-titans__reg-baseline__distr-0__mem-1__win-0",
        "atma_raven_titans",
        "external_raven_adamw",
    ),
    "raven_native": ModelSpec(
        "raven_native",
        "ChavyvAkvar/atma-10b-L40S-mbs16-raven-native__reg-baseline__distr-0__mem-0__win-0",
        "raven_native",
        "external_raven_adamw",
    ),
}


@dataclass(frozen=True)
class BenchmarkJob:
    stage: str
    benchmark: str
    suite: str
    model: str
    checkpoint_revision: str
    dataset_revisions: tuple[tuple[str, str], ...]
    command_args: tuple[str, ...]


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _resolved_revision(snapshot: Path, requested: str) -> str:
    # snapshot_download normally returns .../snapshots/<commit sha>.
    if snapshot.parent.name == "snapshots" and len(snapshot.name) >= 12:
        return snapshot.name
    return requested


def _validate_checkpoint(snapshot: Path, spec: ModelSpec) -> dict:
    cfg_path = snapshot / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"{spec.repo_id}: missing config.json in {snapshot}")
    weights = next((snapshot / name for name in WEIGHT_NAMES if (snapshot / name).is_file()), None)
    if weights is None:
        raise FileNotFoundError(
            f"{spec.repo_id}: expected one of {', '.join(WEIGHT_NAMES)} in {snapshot}"
        )

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    architecture = cfg.get("arch_type") or cfg.get("attn_type", "polar")
    if architecture != spec.architecture:
        raise ValueError(
            f"{spec.repo_id}: expected architecture={spec.architecture!r}, "
            f"found {architecture!r}"
        )

    warnings = []
    run_cfg_path = snapshot / "run_config.json"
    run_cfg = _read_json(run_cfg_path, {})
    if run_cfg.get("mbs") not in (None, 16):
        raise ValueError(f"{spec.repo_id}: expected mbs=16, found {run_cfg['mbs']!r}")
    if run_cfg.get("num_chunks") not in (None, 99):
        warnings.append(f"expected num_chunks=99, found {run_cfg['num_chunks']!r}")

    tokenizer_cfg = _read_json(snapshot / "tokenizer.json", {})
    tokenizer_name = tokenizer_cfg.get("tokenizer_name")
    if tokenizer_name not in (None, "gpt2"):
        warnings.append(f"expected GPT-2 tokenizer metadata, found {tokenizer_name!r}")

    return {
        "snapshot_path": str(snapshot),
        "weights_path": str(weights),
        "weights_bytes": weights.stat().st_size,
        "architecture": architecture,
        "mbs": run_cfg.get("mbs"),
        "num_chunks": run_cfg.get("num_chunks"),
        "tokenizer": tokenizer_name,
        "warnings": warnings,
    }


def _download_checkpoint(
    spec: ModelSpec,
    record: dict | None,
    *,
    requested_revision: str,
    cache_dir: str | None,
    offline: bool,
    refresh_revision: bool,
) -> dict:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required; install it in the benchmark environment"
        ) from exc

    revision = requested_revision
    if (
        record
        and not refresh_revision
        and record.get("repo_id") == spec.repo_id
        and record.get("requested_revision") == requested_revision
        and record.get("resolved_revision")
    ):
        revision = record["resolved_revision"]

    print(f"[pipeline] resolve {spec.key}: {spec.repo_id}@{revision}", flush=True)
    snapshot = Path(
        snapshot_download(
            repo_id=spec.repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=offline,
            allow_patterns=[
                "config.json",
                "run_config.json",
                "tokenizer.json",
                *WEIGHT_NAMES,
            ],
        )
    ).resolve()
    validation = _validate_checkpoint(snapshot, spec)
    return {
        **asdict(spec),
        "requested_revision": requested_revision,
        "resolved_revision": _resolved_revision(snapshot, revision),
        "resolved_at_unix": int(time.time()),
        **validation,
    }


def _resolve_dataset_revision(
    dataset_id: str,
    requested_revision: str,
    record: dict | None,
    *,
    offline: bool,
    refresh_revision: bool,
) -> dict:
    if (
        record
        and not refresh_revision
        and record.get("dataset_id") == dataset_id
        and record.get("requested_revision") == requested_revision
        and record.get("resolved_revision")
    ):
        return record
    if offline:
        raise RuntimeError(
            f"no pinned revision for dataset {dataset_id!r}; run once without --offline"
        )
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required; install it in the benchmark environment"
        ) from exc
    info = HfApi().dataset_info(repo_id=dataset_id, revision=requested_revision)
    return {
        "dataset_id": dataset_id,
        "requested_revision": requested_revision,
        "resolved_revision": info.sha,
        "resolved_at_unix": int(time.time()),
    }


def _job_fingerprint(job: BenchmarkJob) -> str:
    payload = json.dumps(asdict(job), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def _job_output(log_dir: Path, job: BenchmarkJob) -> Path:
    return log_dir / (
        f"{job.stage}_{job.benchmark}_{job.suite}_{job.model}_{_job_fingerprint(job)}.log"
    )


RESULT_MARKERS = (
    "===RETRIEVAL_RESULTS_JSON===",
    "===BASE_RESULTS_JSON===",
    "===LONGDOC_RESULTS_JSON===",
    "===SERVING_RESULTS_JSON===",
    "===BABILONG_RESULTS_JSON===",
)


def _is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return any(marker in text for marker in RESULT_MARKERS) and text.rstrip().endswith("===END===")


def _latest_complete(path: Path) -> Path | None:
    candidates = [path, *path.parent.glob(f"{path.stem}.attempt-*{path.suffix}")]
    completed = [candidate for candidate in candidates if _is_complete(candidate)]
    return max(completed, key=lambda candidate: candidate.stat().st_mtime) if completed else None


def _attempt_path(path: Path) -> Path:
    if not path.exists():
        return path
    attempt = 2
    while True:
        candidate = path.with_name(f"{path.stem}.attempt-{attempt}{path.suffix}")
        if not candidate.exists():
            return candidate
        attempt += 1


def _extract_result(path: Path) -> dict | None:
    if not _is_complete(path):
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    positions = [(text.rfind(marker), marker) for marker in RESULT_MARKERS if marker in text]
    _, marker = max(positions)
    block = text.rsplit(marker, 1)[1].split("===END===", 1)[0].strip()
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        return None


def _result_has_oom(result: dict | None) -> bool:
    if not result:
        return False
    if result.get("oom_cells"):
        return True
    if result.get("benchmark") == "serving":
        return any(cell.get("oom") for cell in result.get("results", {}).values())
    if result.get("benchmark") == "longdoc":
        return any(
            cell.get("oom")
            for dataset in result.get("results", {}).values()
            for cell in dataset.get("lengths", {}).values()
        )
    return False


def _jobs(args) -> list[BenchmarkJob]:
    jobs = []
    retrieval_stages = []
    if args.stage in ("smoke", "all"):
        retrieval_stages.append(("smoke", ("synthetic",), args.smoke_lengths, args.smoke_samples))
    if args.stage == "pilot":
        retrieval_stages.append(("pilot", tuple(args.suites), args.lengths, args.pilot_samples))
    if args.stage in ("retrieval", "full", "all"):
        retrieval_stages.append(("retrieval", tuple(args.suites), args.lengths, args.samples))

    for stage, suites, lengths, samples in retrieval_stages:
        for model in args.models:
            for suite in suites:
                dataset_revisions = ()
                extra = []
                if suite == "real":
                    revision = args.dataset_revisions[args.haystack]
                    dataset_revisions = ((args.haystack, revision),)
                    extra = ["--haystack", args.haystack, "--haystack_revision", revision]
                jobs.append(
                    BenchmarkJob(
                        stage=stage,
                        benchmark="retrieval",
                        suite=suite,
                        model=model,
                        checkpoint_revision=args.checkpoint_revisions[model],
                        dataset_revisions=dataset_revisions,
                        command_args=tuple([
                            "--tasks", *args.tasks,
                            "--lengths", *lengths,
                            "--depths", *(str(depth) for depth in args.depths),
                            "--samples", str(samples),
                            "--seed", str(args.seed),
                            "--max_tokens", str(args.max_tokens),
                            "--max_num_seqs", str(args.max_num_seqs),
                            *extra,
                        ]),
                    )
                )

    if args.stage in ("base", "full", "all"):
        from benchmarks.base_tasks import BASE_TASK_SPECS

        revisions = tuple(sorted({
            spec.dataset_id: args.dataset_revisions[spec.dataset_id]
            for task, spec in BASE_TASK_SPECS.items() if task in args.base_tasks
        }.items()))
        for model in args.models:
            command_args = [
                "--tasks", *args.base_tasks,
                "--batch_size", str(args.base_batch_size),
                "--scoring_max_length", str(args.base_max_length),
                "--dataset_revisions", str(args.manifest_path),
            ]
            if args.base_limit:
                command_args.extend(("--limit", str(args.base_limit)))
            jobs.append(BenchmarkJob(
                stage="base", benchmark="base", suite="zero_shot", model=model,
                checkpoint_revision=args.checkpoint_revisions[model],
                dataset_revisions=revisions, command_args=tuple(command_args),
            ))

    if args.stage in ("longdoc", "full", "all"):
        from benchmarks.longdoc import LONGDOC_SPECS

        revisions = tuple(sorted({
            LONGDOC_SPECS[name].dataset_id: args.dataset_revisions[LONGDOC_SPECS[name].dataset_id]
            for name in args.longdoc_datasets
        }.items()))
        for model in args.models:
            jobs.append(BenchmarkJob(
                stage="longdoc", benchmark="longdoc", suite="fixed_target", model=model,
                checkpoint_revision=args.checkpoint_revisions[model],
                dataset_revisions=revisions,
                command_args=tuple([
                    "--datasets", *args.longdoc_datasets,
                    "--lengths", *args.longdoc_lengths,
                    "--target_tokens", str(args.target_tokens),
                    "--num_docs", str(args.num_docs),
                    "--max_scan", str(args.max_scan),
                    "--dataset_revisions", str(args.manifest_path),
                ]),
            ))

    if args.stage in ("serving", "full", "all"):
        for model in args.models:
            command_args = [
                "--lengths", *args.serving_lengths,
                "--decode_tokens", str(args.decode_tokens),
                "--serving_samples", str(args.serving_samples),
                "--max_num_seqs", str(args.max_num_seqs),
            ]
            if args.max_num_batched_tokens:
                command_args.extend((
                    "--max_num_batched_tokens", str(args.max_num_batched_tokens)
                ))
            jobs.append(BenchmarkJob(
                stage="serving", benchmark="serving", suite="generation", model=model,
                checkpoint_revision=args.checkpoint_revisions[model], dataset_revisions=(),
                command_args=tuple(command_args),
            ))
    return jobs


def _command(args, job: BenchmarkJob, checkpoint: Path, output: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.run",
        "--benchmark",
        job.benchmark,
        "--model",
        str(checkpoint),
        *job.command_args,
        "--out",
        str(output),
        "--strict",
    ]
    if job.benchmark == "retrieval" and args.max_model_len:
        cmd.extend(("--max_model_len", str(args.max_model_len)))
    if job.benchmark == "retrieval" and args.max_num_batched_tokens:
        cmd.extend(("--max_num_batched_tokens", str(args.max_num_batched_tokens)))
    return cmd


def _run_with_console_log(cmd: list[str], console_path: Path, env: dict) -> int:
    console_path.parent.mkdir(parents=True, exist_ok=True)
    print("[pipeline] " + " ".join(cmd), flush=True)
    with console_path.open("w", encoding="utf-8", buffering=1) as console:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                console.write(line)
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return process.wait()


def _parser() -> argparse.ArgumentParser:
    from benchmarks.base_tasks import BASE_TASK_SPECS
    from benchmarks.longdoc import LONGDOC_SPECS

    ap = argparse.ArgumentParser(
        description="Download and run the complete promoted ATMA 10B-token benchmark matrix."
    )
    ap.add_argument(
        "--stage",
        choices=(
            "download", "smoke", "pilot", "retrieval", "base", "longdoc",
            "serving", "full", "all",
        ),
        default="smoke",
        help="full = retrieval+base+longdoc+serving; all = smoke then full",
    )
    ap.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS),
                    default=list(MODEL_SPECS))
    ap.add_argument("--suites", nargs="+", choices=("synthetic", "real"),
                    default=("synthetic", "real"))
    ap.add_argument("--tasks", nargs="+", choices=("passkey", "niah"),
                    default=("passkey", "niah"))
    ap.add_argument("--base_tasks", nargs="+", choices=tuple(BASE_TASK_SPECS),
                    default=tuple(BASE_TASK_SPECS))
    ap.add_argument("--longdoc_datasets", nargs="+", choices=tuple(LONGDOC_SPECS),
                    default=tuple(LONGDOC_SPECS))
    ap.add_argument("--smoke_lengths", nargs="+", default=("2k", "8k"))
    ap.add_argument(
        "--lengths",
        nargs="+",
        default=("2k", "8k", "32k", "64k", "128k", "256k"),
    )
    ap.add_argument("--depths", nargs="+", type=float, default=(0.1, 0.5, 0.9))
    ap.add_argument("--longdoc_lengths", nargs="+",
                    default=("2k", "8k", "32k", "64k", "128k", "256k"))
    ap.add_argument("--serving_lengths", nargs="+",
                    default=("2k", "8k", "32k", "64k", "128k", "256k"))
    ap.add_argument("--smoke_samples", type=int, default=2)
    ap.add_argument("--pilot_samples", type=int, default=10)
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_tokens", type=int, default=16)
    ap.add_argument("--base_limit", type=int, default=None,
                    help="examples per base task; default evaluates each complete split")
    ap.add_argument("--base_batch_size", type=int, default=8)
    ap.add_argument("--base_max_length", type=int, default=2048)
    ap.add_argument("--target_tokens", type=int, default=256)
    ap.add_argument("--num_docs", type=int, default=8)
    ap.add_argument("--max_scan", type=int, default=100000)
    ap.add_argument("--decode_tokens", type=int, default=32)
    ap.add_argument("--serving_samples", type=int, default=1)
    ap.add_argument("--max_num_seqs", type=int, default=1,
                    help="one is safest and makes memory comparable at 256K")
    ap.add_argument("--max_model_len", type=int, default=None)
    ap.add_argument("--max_num_batched_tokens", type=int, default=None)
    ap.add_argument("--haystack", default="codelion/finepdfs-1B")
    ap.add_argument("--haystack_revision", default="main")
    ap.add_argument("--dataset_revision", default="main",
                    help="requested revision for downstream datasets; resolved commits are pinned")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    ap.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--cache_dir", default=None, help="optional Hugging Face cache directory")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--refresh_revisions", action="store_true",
                    help="resolve the requested revision again instead of reusing pinned commits")
    ap.add_argument("--offline", action="store_true",
                    help="use only snapshots already present in the Hugging Face cache")
    ap.add_argument("--rerun", action="store_true",
                    help="run a new attempt even when the exact fingerprint completed")
    ap.add_argument("--dry_run", action="store_true",
                    help="download/validate checkpoints and print jobs without launching them")
    ap.add_argument("--fail_fast", action="store_true")
    return ap


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    for name in (
        "smoke_samples", "pilot_samples", "samples", "max_tokens", "max_num_seqs",
        "base_batch_size", "base_max_length", "target_tokens", "num_docs", "max_scan",
        "decode_tokens", "serving_samples",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if any(depth < 0.0 or depth > 1.0 for depth in args.depths):
        parser.error("--depths values must be between 0 and 1")
    args.log_dir = args.log_dir.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.log_dir / "checkpoint_manifest.json"
    manifest = _read_json(manifest_path, {"schema_version": 1, "models": {}})

    checkpoints = {}
    for key in args.models:
        spec = MODEL_SPECS[key]
        record = manifest.get("models", {}).get(key)
        resolved = _download_checkpoint(
            spec,
            record,
            requested_revision=args.revision,
            cache_dir=args.cache_dir,
            offline=args.offline,
            refresh_revision=args.refresh_revisions,
        )
        manifest.setdefault("models", {})[key] = resolved
        manifest["updated_at_unix"] = int(time.time())
        _write_json(manifest_path, manifest)
        checkpoints[key] = Path(resolved["snapshot_path"])
        for warning in resolved.get("warnings", []):
            print(f"[pipeline] warning {key}: {warning}", flush=True)

    args.checkpoint_revisions = {
        key: manifest["models"][key]["resolved_revision"] for key in args.models
    }
    args.manifest_path = manifest_path
    requested_datasets = {}
    if args.stage in ("pilot", "retrieval", "full", "all") and "real" in args.suites:
        requested_datasets[args.haystack] = args.haystack_revision
    if args.stage in ("base", "full", "all"):
        from benchmarks.base_tasks import BASE_TASK_SPECS

        for task in args.base_tasks:
            requested_datasets.setdefault(
                BASE_TASK_SPECS[task].dataset_id, args.dataset_revision
            )
    if args.stage in ("longdoc", "full", "all"):
        from benchmarks.longdoc import LONGDOC_SPECS

        for name in args.longdoc_datasets:
            requested_datasets.setdefault(
                LONGDOC_SPECS[name].dataset_id, args.dataset_revision
            )

    args.dataset_revisions = {}
    for dataset_id, requested_revision in requested_datasets.items():
        dataset_record = manifest.get("datasets", {}).get(dataset_id)
        resolved_dataset = _resolve_dataset_revision(
            dataset_id,
            requested_revision,
            dataset_record,
            offline=args.offline,
            refresh_revision=args.refresh_revisions,
        )
        manifest.setdefault("datasets", {})[dataset_id] = resolved_dataset
        manifest["updated_at_unix"] = int(time.time())
        _write_json(manifest_path, manifest)
        args.dataset_revisions[dataset_id] = resolved_dataset["resolved_revision"]
        print(
            f"[pipeline] pinned dataset: {dataset_id}@{resolved_dataset['resolved_revision']}",
            flush=True,
        )
    jobs = _jobs(args)
    if args.stage == "download":
        print(f"[pipeline] checkpoints ready; manifest: {manifest_path}")
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("FLA_CUSTOM_OP", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")

    records = []
    failures = 0
    failed_smoke_models = set()
    for job in jobs:
        if job.stage != "smoke" and job.model in failed_smoke_models:
            print(
                f"[pipeline] skip {job.model}/{job.benchmark}/{job.suite}: its smoke gate failed",
                flush=True,
            )
            records.append(
                {"job": asdict(job), "status": "skipped_failed_smoke_gate", "log": None}
            )
            continue

        canonical = _job_output(args.log_dir, job)
        completed = _latest_complete(canonical)
        if completed is not None and not args.rerun:
            print(f"[pipeline] skip completed: {completed.name}", flush=True)
            completed_result = _extract_result(completed)
            completed_has_oom = _result_has_oom(completed_result)
            if job.stage == "smoke" and completed_has_oom:
                failed_smoke_models.add(job.model)
                failures += 1
            records.append(
                {
                    "job": asdict(job),
                    "status": (
                        "skipped_complete_with_oom"
                        if completed_has_oom
                        else "skipped_complete"
                    ),
                    "log": str(completed),
                    "result": completed_result,
                }
            )
            continue

        output = _attempt_path(canonical) if canonical.exists() else canonical
        cmd = _command(args, job, checkpoints[job.model], output)
        if args.dry_run:
            print("[pipeline] dry-run: " + " ".join(cmd), flush=True)
            records.append({"job": asdict(job), "status": "dry_run", "log": str(output)})
            continue

        started = time.time()
        console = output.with_suffix(".console.log")
        returncode = _run_with_console_log(cmd, console, env)
        complete = _is_complete(output)
        result = _extract_result(output)
        if returncode == 0 and complete:
            status = "complete_with_oom" if _result_has_oom(result) else "complete"
        else:
            status = "failed"
        record = {
            "job": asdict(job),
            "status": status,
            "returncode": returncode,
            "elapsed_s": round(time.time() - started, 1),
            "log": str(output),
            "console_log": str(console),
            "result": result,
        }
        records.append(record)
        _write_json(args.log_dir / "pipeline_summary.json", {
            "schema_version": 1,
            "checkpoint_manifest": str(manifest_path),
            "records": records,
        })
        if status == "complete_with_oom" and job.stage == "smoke":
            failures += 1
            failed_smoke_models.add(job.model)
            print(
                f"[pipeline] smoke gate OOM for {job.model}; its full jobs will be skipped",
                flush=True,
            )
        if status == "failed":
            failures += 1
            if job.stage == "smoke":
                failed_smoke_models.add(job.model)
            print(
                f"[pipeline] FAILED {job.model}/{job.stage}/{job.suite}; "
                f"returncode={returncode}, console={console}",
                flush=True,
            )
            if args.fail_fast:
                break

    summary_path = args.log_dir / "pipeline_summary.json"
    _write_json(summary_path, {
        "schema_version": 1,
        "checkpoint_manifest": str(manifest_path),
        "records": records,
    })
    print(f"[pipeline] summary: {summary_path}", flush=True)
    if not args.dry_run:
        from benchmarks.aggregate import aggregate, write_outputs

        matrix = aggregate(args.log_dir)
        matrix_json = args.log_dir / "benchmark_matrix.json"
        matrix_csv = args.log_dir / "benchmark_matrix.csv"
        write_outputs(matrix, matrix_json, matrix_csv)
        print(
            f"[pipeline] aggregate: {len(matrix['rows'])} rows -> "
            f"{matrix_json}, {matrix_csv}",
            flush=True,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
