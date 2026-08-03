"""Fine-tune, evaluate, and optionally upload every promoted BABILong checkpoint.

Each GPU stage runs in a fresh subprocess. Completed checkpoints and structured evaluation
logs are reused on restart, so an interrupted multi-model run can be resumed with the same
command. Uploads require an explicit --upload flag and a Hugging Face login or HF_TOKEN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from benchmarks.babilong import (
    DEFAULT_DATASET_ID,
    EVAL_LENGTHS,
    TASKS,
    TRAIN_LENGTHS,
)
from benchmarks.run_pipeline import (
    MODEL_SPECS,
    ROOT,
    _attempt_path,
    _download_checkpoint,
    _extract_result,
    _is_complete,
    _latest_complete,
    _read_json,
    _resolve_dataset_revision,
    _result_has_oom,
    _run_with_console_log,
    _write_json,
)


DEFAULT_OUTPUT_ROOT = ROOT / "checkpoints" / "babilong_2k_ft"
DEFAULT_LOG_DIR = ROOT / "benchmarks" / "logs" / "babilong_2k_ft"
DEFAULT_NAMESPACE = next(iter(MODEL_SPECS.values())).repo_id.split("/", 1)[0]
DEFAULT_REPO_PREFIX = "atma-10b-babilong-2k-ft"


def _repo_id(namespace: str, prefix: str, model: str) -> str:
    namespace = namespace.strip().strip("/")
    prefix = prefix.strip().strip("/")
    if not namespace or "/" in namespace:
        raise ValueError("--hf_namespace must be one Hugging Face user or organization")
    if not prefix or "/" in prefix:
        raise ValueError("--hf_repo_prefix must be a repository-name prefix, not a repo ID")
    return f"{namespace}/{prefix}-{model.replace('_', '-')}"


def _finetune_command(args, source: Path, output: Path, source_repo: str, source_revision: str):
    return [
        sys.executable,
        "-m",
        "benchmarks.finetune_babilong",
        "--model",
        str(source),
        "--output_dir",
        str(output),
        "--dataset",
        args.dataset,
        "--dataset_revision",
        args.pinned_dataset_revision,
        "--source_repo_id",
        source_repo,
        "--source_revision",
        source_revision,
        "--tasks",
        *args.tasks,
        "--train_lengths",
        *args.train_lengths,
        "--seq_len",
        str(args.seq_len),
        "--train_start",
        str(args.train_start),
        "--train_end",
        str(args.train_end),
        "--val_start",
        str(args.val_start),
        "--val_end",
        str(args.val_end),
        "--epochs",
        str(args.epochs),
        "--micro_batch_size",
        str(args.micro_batch_size),
        "--grad_accum_steps",
        str(args.grad_accum_steps),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
    ]


def _eval_command(args, checkpoint: Path, output: Path, *, pilot: bool):
    tasks = [args.pilot_task] if pilot else list(args.tasks)
    lengths = ["256k"] if pilot else list(args.eval_lengths)
    samples = args.pilot_samples if pilot else args.eval_samples
    return [
        sys.executable,
        "-m",
        "benchmarks.run",
        "--benchmark",
        "babilong",
        "--model",
        str(checkpoint),
        "--dataset",
        args.dataset,
        "--dataset_revision",
        args.pinned_dataset_revision,
        "--tasks",
        *tasks,
        "--lengths",
        *lengths,
        "--row_start",
        str(args.test_start),
        "--row_end",
        str(args.test_end),
        "--samples",
        str(samples),
        "--babilong_backend",
        "direct",
        "--max_tokens",
        str(args.max_tokens),
        "--out",
        str(output),
    ]


def _checkpoint_matches(
    output: Path,
    *,
    source_repo: str,
    source_revision: str,
    dataset: str,
    dataset_revision: str,
    tasks,
    train_lengths,
    seq_len: int,
    train_rows,
    validation_rows,
    test_rows,
) -> bool:
    weights = output / "weights.pt"
    if not weights.exists():
        return False
    required = (
        output / "config.json",
        output / "finetune_manifest.json",
        output / "training_summary.json",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"incomplete existing checkpoint at {output}: missing {missing}; "
            "use a new --output_root rather than overwriting it"
        )
    manifest = _read_json(output / "finetune_manifest.json", {})
    expected = {
        "protocol": "heldout-short-finetune-v1",
        "source_repo_id": source_repo,
        "source_revision": source_revision,
        "dataset_id": dataset,
        "dataset_revision": dataset_revision,
        "tasks": list(tasks),
        "train_lengths": list(train_lengths),
        "seq_len": seq_len,
        "train_rows": list(train_rows),
        "validation_rows": list(validation_rows),
        "reserved_test_rows": list(test_rows),
    }
    mismatches = {
        key: {"found": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"existing checkpoint protocol mismatch at {output}: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return True


def _run_stage(cmd, console: Path, env: dict) -> tuple[int, float]:
    started = time.time()
    returncode = _run_with_console_log(cmd, console, env)
    return returncode, round(time.time() - started, 1)


def _run_eval(args, checkpoint: Path, canonical: Path, env: dict, *, pilot: bool):
    completed = _latest_complete(canonical)
    if completed is not None:
        result = _extract_result(completed)
        has_oom = _result_has_oom(result)
        return {
            "status": "skipped_complete_with_oom" if has_oom else "skipped_complete",
            "returncode": 0,
            "log": str(completed),
            "console_log": None,
            "elapsed_s": 0.0,
            "result": result,
        }

    output = _attempt_path(canonical) if canonical.exists() else canonical
    output.parent.mkdir(parents=True, exist_ok=True)
    command = _eval_command(args, checkpoint, output, pilot=pilot)
    console = output.with_suffix(".console.log")
    returncode, elapsed = _run_stage(command, console, env)
    complete = _is_complete(output)
    result = _extract_result(output)
    has_oom = _result_has_oom(result)
    status = (
        "complete_with_oom"
        if returncode == 0 and complete and has_oom
        else "complete"
        if returncode == 0 and complete
        else "failed"
    )
    return {
        "status": status,
        "returncode": returncode,
        "log": str(output),
        "console_log": str(console),
        "elapsed_s": elapsed,
        "result": result,
    }


def _model_card(spec, target_repo: str, finetune_manifest: dict, stage_record: dict) -> str:
    pilot = stage_record.get("pilot", {}).get("status", "not_run")
    full = stage_record.get("full_eval", {}).get("status", "not_run")
    return f"""---
library_name: pytorch
base_model: {spec.repo_id}
datasets:
- {finetune_manifest.get("dataset_id", DEFAULT_DATASET_ID)}
tags:
- babilong
- long-context
- pytorch
---

# {target_repo.split("/", 1)[1]}

ATMA checkpoint adapted on BABILong qa1 through qa10 using only the 0K, 1K, and
2K configurations. This is an internal controlled adaptation checkpoint, not an
official BABILong leaderboard submission.

## Protocol

- Base model: {spec.repo_id}
- Base revision: {finetune_manifest.get("source_revision")}
- Dataset revision: {finetune_manifest.get("dataset_revision")}
- Prompt protocol: {finetune_manifest.get("prompt_protocol")}
- Training lengths: {", ".join(finetune_manifest.get("train_lengths", []))}
- Sequence length: {finetune_manifest.get("seq_len")}
- Train rows: {finetune_manifest.get("train_rows")}
- Validation rows: {finetune_manifest.get("validation_rows")}
- Reserved test rows: {finetune_manifest.get("reserved_test_rows")}
- 256K pilot status: {pilot}
- Full 0K through 256K evaluation status: {full}

Load this repository with the checkpoint-exact ATMA benchmark code. It is not a
Transformers AutoModel checkpoint.
"""


def _write_upload_artifacts(
    output: Path,
    *,
    spec,
    source_record: dict,
    target_repo: str,
    dataset_record: dict,
    stages: dict,
):
    manifest = _read_json(output / "finetune_manifest.json", {})
    pipeline_manifest = {
        "schema_version": 1,
        "model": spec.key,
        "source_repo_id": spec.repo_id,
        "source_revision": source_record["resolved_revision"],
        "target_repo_id": target_repo,
        "dataset_id": dataset_record["dataset_id"],
        "dataset_revision": dataset_record["resolved_revision"],
        "stages": {
            name: {
                key: value
                for key, value in record.items()
                if key != "result"
            }
            for name, record in stages.items()
            if name != "upload"
        },
    }
    _write_json(output / "babilong_pipeline_manifest.json", pipeline_manifest)
    for name, filename in (
        ("pilot", "babilong_256k_pilot_result.json"),
        ("full_eval", "babilong_full_eval_result.json"),
    ):
        result = stages.get(name, {}).get("result")
        if result is not None:
            _write_json(output / filename, result)
    (output / "README.md").write_text(
        _model_card(spec, target_repo, manifest, stages),
        encoding="utf-8",
    )


def _upload_fingerprint(output: Path) -> str:
    digest = hashlib.sha256()
    weights = output / "weights.pt"
    stat = weights.stat()
    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    for filename in (
        "finetune_manifest.json",
        "training_summary.json",
        "babilong_pipeline_manifest.json",
        "babilong_256k_pilot_result.json",
        "babilong_full_eval_result.json",
        "README.md",
    ):
        path = output / filename
        if path.is_file():
            digest.update(filename.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _hub_api_with_identity():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for --upload") from exc
    api = HfApi()
    try:
        identity = api.whoami()
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face authentication is required before training starts. "
            "Run 'hf auth login' or export HF_TOKEN, then retry the same command."
        ) from exc
    return api, identity


def _upload_checkpoint(
    api,
    output: Path,
    *,
    repo_id: str,
    private: bool,
    commit_message: str,
):
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(output),
        commit_message=commit_message,
        ignore_patterns=["*.tmp"],
    )
    return {
        "status": "complete",
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "commit": getattr(info, "oid", None),
        "uploaded_at_unix": int(time.time()),
    }


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune all promoted ATMA checkpoints on BABILong <=2K, evaluate through "
            "262,144 tokens, and optionally upload the adapted checkpoints."
        )
    )
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS),
                        default=list(MODEL_SPECS))
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--revision", default="main",
                        help="requested source checkpoint revision")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset_revision", default="main")
    parser.add_argument("--refresh_revisions", action="store_true")
    parser.add_argument("--offline", action="store_true")

    parser.add_argument("--tasks", nargs="+", choices=TASKS[:10], default=TASKS[:10])
    parser.add_argument("--train_lengths", nargs="+", choices=TRAIN_LENGTHS,
                        default=TRAIN_LENGTHS)
    parser.add_argument("--eval_lengths", nargs="+", choices=EVAL_LENGTHS,
                        default=EVAL_LENGTHS)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_end", type=int, default=80)
    parser.add_argument("--val_start", type=int, default=80)
    parser.add_argument("--val_end", type=int, default=90)
    parser.add_argument("--test_start", type=int, default=90)
    parser.add_argument("--test_end", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--pilot_task", choices=TASKS[:10], default="qa1")
    parser.add_argument("--pilot_samples", type=int, default=1)
    parser.add_argument("--eval_samples", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=16)

    parser.add_argument("--upload", action="store_true",
                        help="upload each local fine-tuned checkpoint after its evaluation")
    parser.add_argument("--hf_namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--hf_repo_prefix", default=DEFAULT_REPO_PREFIX)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--force_upload", action="store_true")
    parser.add_argument(
        "--commit_message",
        default="Upload controlled BABILong 2K fine-tune and held-out evaluation",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    return parser


def _validate_args(parser, args):
    for name in (
        "seq_len",
        "epochs",
        "micro_batch_size",
        "grad_accum_steps",
        "pilot_samples",
        "eval_samples",
        "max_tokens",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.seq_len != 2048:
        parser.error("--seq_len must be exactly 2048 for the controlled all-model protocol")
    if tuple(args.tasks) != tuple(TASKS[:10]):
        parser.error("--tasks must be exactly qa1 through qa10 for the controlled protocol")
    if tuple(args.train_lengths) != tuple(TRAIN_LENGTHS):
        parser.error("--train_lengths must be exactly 0k 1k 2k")
    if tuple(args.eval_lengths) != tuple(EVAL_LENGTHS):
        parser.error(
            "--eval_lengths must be exactly 0k 1k 2k 4k 8k 16k 32k 64k 128k 256k"
        )
    row_protocol = (
        args.train_start,
        args.train_end,
        args.val_start,
        args.val_end,
        args.test_start,
        args.test_end,
    )
    if row_protocol != (0, 80, 80, 90, 90, 100):
        parser.error("controlled row ranges must be train [0,80), val [80,90), test [90,100)")
    if args.upload and args.offline:
        parser.error("--upload and --offline cannot be used together")
    _repo_id(args.hf_namespace, args.hf_repo_prefix, args.models[0])


def _dry_run(args):
    print("[babilong-pipeline] dry-run plan")
    for model in args.models:
        spec = MODEL_SPECS[model]
        output = args.output_root / model
        target = _repo_id(args.hf_namespace, args.hf_repo_prefix, model)
        print(f"  {model}: {spec.repo_id} -> {output}")
        print(f"    eval: 0K through 256K; upload: {target if args.upload else 'disabled'}")
    return 0


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    args.output_root = args.output_root.resolve()
    args.log_dir = args.log_dir.resolve()
    if args.dry_run:
        return _dry_run(args)

    hub_api = None
    if args.upload:
        try:
            hub_api, identity = _hub_api_with_identity()
        except RuntimeError as exc:
            parser.error(str(exc))
        print(
            f"[babilong-pipeline] Hugging Face identity: {identity.get('name', 'unknown')}",
            flush=True,
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.log_dir / "pipeline_summary.json"
    state = _read_json(state_path, {"schema_version": 1, "models": {}})

    dataset_record = _resolve_dataset_revision(
        args.dataset,
        args.dataset_revision,
        state.get("dataset"),
        offline=args.offline,
        refresh_revision=args.refresh_revisions,
    )
    state["dataset"] = dataset_record
    args.pinned_dataset_revision = dataset_record["resolved_revision"]
    _write_json(state_path, state)
    print(
        f"[babilong-pipeline] pinned dataset: "
        f"{args.dataset}@{args.pinned_dataset_revision}",
        flush=True,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("FLA_CUSTOM_OP", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")

    failures = 0
    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        model_state = state.setdefault("models", {}).setdefault(model_key, {})
        stages = model_state.setdefault("stages", {})
        target_repo = _repo_id(args.hf_namespace, args.hf_repo_prefix, model_key)
        output = args.output_root / model_key
        print(f"[babilong-pipeline] ===== {model_key} =====", flush=True)
        try:
            source_record = _download_checkpoint(
                spec,
                model_state.get("source"),
                requested_revision=args.revision,
                cache_dir=args.cache_dir,
                offline=args.offline,
                refresh_revision=args.refresh_revisions,
            )
            model_state.update({
                "source": source_record,
                "output_dir": str(output),
                "target_repo_id": target_repo,
            })
            source = Path(source_record["snapshot_path"])
            _write_json(state_path, state)

            ready = _checkpoint_matches(
                output,
                source_repo=spec.repo_id,
                source_revision=source_record["resolved_revision"],
                dataset=args.dataset,
                dataset_revision=args.pinned_dataset_revision,
                tasks=args.tasks,
                train_lengths=args.train_lengths,
                seq_len=args.seq_len,
                train_rows=(args.train_start, args.train_end),
                validation_rows=(args.val_start, args.val_end),
                test_rows=(args.test_start, args.test_end),
            )
            if ready:
                print(f"[babilong-pipeline] skip completed fine-tune: {output}", flush=True)
                stages["finetune"] = {
                    "status": "skipped_complete",
                    "checkpoint": str(output),
                }
            else:
                train_console = args.log_dir / "train" / f"{model_key}.console.log"
                command = _finetune_command(
                    args,
                    source,
                    output,
                    spec.repo_id,
                    source_record["resolved_revision"],
                )
                returncode, elapsed = _run_stage(command, train_console, env)
                stages["finetune"] = {
                    "status": "complete" if returncode == 0 else "failed",
                    "returncode": returncode,
                    "elapsed_s": elapsed,
                    "checkpoint": str(output),
                    "console_log": str(train_console),
                }
                _write_json(state_path, state)
                if returncode != 0:
                    raise RuntimeError(f"fine-tune failed; see {train_console}")
                _checkpoint_matches(
                    output,
                    source_repo=spec.repo_id,
                    source_revision=source_record["resolved_revision"],
                    dataset=args.dataset,
                    dataset_revision=args.pinned_dataset_revision,
                    tasks=args.tasks,
                    train_lengths=args.train_lengths,
                    seq_len=args.seq_len,
                    train_rows=(args.train_start, args.train_end),
                    validation_rows=(args.val_start, args.val_end),
                    test_rows=(args.test_start, args.test_end),
                )

            pilot_log = args.log_dir / "pilot" / f"babilong_256k_{model_key}.log"
            stages["pilot"] = _run_eval(args, output, pilot_log, env, pilot=True)
            _write_json(state_path, state)
            pilot_ok = stages["pilot"]["status"] in ("complete", "skipped_complete")
            if pilot_ok:
                full_log = args.log_dir / "full" / f"babilong_{model_key}.log"
                stages["full_eval"] = _run_eval(
                    args, output, full_log, env, pilot=False
                )
            else:
                stages["full_eval"] = {
                    "status": "skipped_failed_256k_pilot",
                    "result": None,
                }
            _write_json(state_path, state)

            _write_upload_artifacts(
                output,
                spec=spec,
                source_record=source_record,
                target_repo=target_repo,
                dataset_record=dataset_record,
                stages=stages,
            )
            if args.upload:
                fingerprint = _upload_fingerprint(output)
                previous = stages.get("upload", {})
                if (
                    not args.force_upload
                    and previous.get("status") in ("complete", "skipped_complete")
                    and previous.get("fingerprint") == fingerprint
                    and previous.get("repo_id") == target_repo
                ):
                    print(f"[babilong-pipeline] skip uploaded: {target_repo}", flush=True)
                    stages["upload"] = {
                        **previous,
                        "status": "skipped_complete",
                    }
                else:
                    print(f"[babilong-pipeline] upload -> {target_repo}", flush=True)
                    upload_record = _upload_checkpoint(
                        hub_api,
                        output,
                        repo_id=target_repo,
                        private=args.private,
                        commit_message=args.commit_message,
                    )
                    stages["upload"] = {**upload_record, "fingerprint": fingerprint}
                _write_json(state_path, state)

            full_status = stages["full_eval"]["status"]
            if full_status not in ("complete", "skipped_complete"):
                failures += 1
                print(
                    f"[babilong-pipeline] warning: full evaluation status={full_status}",
                    flush=True,
                )
                if args.fail_fast:
                    break
            else:
                model_state.pop("error", None)
                model_state.pop("failed_at_unix", None)
        except Exception as exc:
            failures += 1
            model_state["error"] = str(exc)
            model_state["failed_at_unix"] = int(time.time())
            _write_json(state_path, state)
            print(f"[babilong-pipeline] FAILED {model_key}: {exc}", flush=True)
            if args.fail_fast:
                break

    full_dir = args.log_dir / "full"
    if full_dir.is_dir():
        from benchmarks.aggregate import aggregate, write_outputs

        matrix = aggregate(full_dir)
        write_outputs(
            matrix,
            args.log_dir / "benchmark_matrix.json",
            args.log_dir / "benchmark_matrix.csv",
        )
        print(
            f"[babilong-pipeline] aggregate: {len(matrix['rows'])} rows -> "
            f"{args.log_dir / 'benchmark_matrix.json'}",
            flush=True,
        )
    print(f"[babilong-pipeline] summary: {state_path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
