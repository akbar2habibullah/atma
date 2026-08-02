"""Mirror the published BABILong result artifacts from immutable Hub commits.

The fine-tuned model repositories contain the only surviving copy of the raw per-model
evaluation bundle. This importer deliberately downloads JSON metadata/results only, never the
1.4 GB checkpoint weights, and reconstructs the local structured logs and aggregate matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "benchmarks" / "logs" / "babilong_2k_ft"

MODEL_SOURCES = {
    "nope": (
        "ChavyvAkvar/atma-10b-babilong-2k-ft-nope",
        "d8cf012152bfaeed2439e83313223bd982b2bdbb",
    ),
    "polar": (
        "ChavyvAkvar/atma-10b-babilong-2k-ft-polar",
        "c1625542a0576a547d888523da8d3067b633d9d2",
    ),
    "rope": (
        "ChavyvAkvar/atma-10b-babilong-2k-ft-rope",
        "a2262bb0d50f825f12e47a4974a1c56d9496df34",
    ),
    "atma_raven_titans": (
        "ChavyvAkvar/atma-10b-babilong-2k-ft-atma-raven-titans",
        "bbb6c7d4ba2f197c4ca980b2a97e27994031d65a",
    ),
    "raven_native": (
        "ChavyvAkvar/atma-10b-babilong-2k-ft-raven-native",
        "f97dd4bf59c2bfb083672696dbc0cbe51991e22a",
    ),
}

ARTIFACTS = (
    "babilong_full_eval_result.json",
    "babilong_256k_pilot_result.json",
    "babilong_pipeline_manifest.json",
    "finetune_manifest.json",
    "training_summary.json",
)

EXPECTED_TASKS = tuple(f"qa{index}" for index in range(1, 11))
EXPECTED_LENGTHS = (
    "0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k"
)


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "atma-babilong-result-import/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _load_json_bytes(payload: bytes, source: str) -> dict:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON downloaded from {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object from {source}")
    return value


def _validate_full_result(model: str, result: dict) -> None:
    if result.get("benchmark") != "babilong":
        raise RuntimeError(f"{model}: full result is not a BABILong result")
    if result.get("protocol") != "heldout-short-finetune-v1":
        raise RuntimeError(f"{model}: unexpected protocol {result.get('protocol')!r}")
    if tuple(result.get("tasks", ())) != EXPECTED_TASKS:
        raise RuntimeError(f"{model}: expected qa1 through qa10")
    if tuple(result.get("lengths", ())) != EXPECTED_LENGTHS:
        raise RuntimeError(f"{model}: unexpected evaluation length grid")
    if tuple(result.get("row_range", ())) != (90, 100):
        raise RuntimeError(f"{model}: result does not use reserved rows [90, 100)")
    missing = [task for task in EXPECTED_TASKS if task not in result.get("results", {})]
    if missing:
        raise RuntimeError(f"{model}: missing task results: {missing}")


def _write_structured_log(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "[import] Published Hugging Face BABILong full-evaluation result.\n"
        "\n===BABILONG_RESULTS_JSON===\n"
        + json.dumps(result, sort_keys=True)
        + "\n===END===\n"
    )
    path.write_text(text, encoding="utf-8")


def _portable_path(value: str | Path) -> str:
    path = Path(value).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def import_results(output_dir: Path, *, offline: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_root = output_dir / "hub"
    full_log_root = output_dir / "full"
    source_records = {}
    full_results = {}

    for model, (repo_id, revision) in MODEL_SOURCES.items():
        model_dir = raw_root / model
        model_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        parsed = {}
        for filename in ARTIFACTS:
            destination = model_dir / filename
            url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
            if offline:
                if not destination.is_file():
                    raise FileNotFoundError(
                        f"offline import is missing {destination}; run once without --offline"
                    )
                payload = destination.read_bytes()
            else:
                payload = _download(url)
                destination.write_bytes(payload)
            parsed[filename] = _load_json_bytes(payload, url)
            files[filename] = {
                "url": url,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }

        full_result = parsed["babilong_full_eval_result.json"]
        _validate_full_result(model, full_result)
        manifest = parsed["babilong_pipeline_manifest.json"]
        if manifest.get("model") != model:
            raise RuntimeError(
                f"{model}: pipeline manifest identifies model={manifest.get('model')!r}"
            )
        if manifest.get("target_repo_id") != repo_id:
            raise RuntimeError(f"{model}: pipeline manifest target repository mismatch")

        full_results[model] = full_result
        source_records[model] = {
            "repo_id": repo_id,
            "revision": revision,
            "tree_url": f"https://huggingface.co/{repo_id}/tree/{revision}",
            "files": files,
        }
        _write_structured_log(full_log_root / f"babilong_{model}.log", full_result)
        print(f"[import-babilong] {model}: {repo_id}@{revision[:12]}")

    dataset_revisions = {result.get("dataset_revision") for result in full_results.values()}
    prompt_protocols = {result.get("prompt_protocol") for result in full_results.values()}
    if len(dataset_revisions) != 1 or None in dataset_revisions:
        raise RuntimeError(f"cross-model dataset revision mismatch: {dataset_revisions}")
    if len(prompt_protocols) != 1 or None in prompt_protocols:
        raise RuntimeError(f"cross-model prompt protocol mismatch: {prompt_protocols}")

    provenance = {
        "schema_version": 1,
        "imported_at_unix": int(time.time()),
        "dataset_revision": next(iter(dataset_revisions)),
        "prompt_protocol": next(iter(prompt_protocols)),
        "models": source_records,
    }
    (output_dir / "hub_sources.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    from benchmarks.aggregate import aggregate, write_outputs

    matrix = aggregate(full_log_root)
    expected_rows = len(MODEL_SOURCES) * (
        len(EXPECTED_TASKS) * len(EXPECTED_LENGTHS) + len(EXPECTED_LENGTHS)
    )
    if len(matrix["rows"]) != expected_rows:
        raise RuntimeError(
            f"aggregate produced {len(matrix['rows'])} rows; expected {expected_rows}"
        )
    matrix["log_dir"] = _portable_path(full_log_root)
    for source in matrix["sources"]:
        source["path"] = _portable_path(source["path"])
    for row in matrix["rows"]:
        row["source_log"] = _portable_path(row["source_log"])
    write_outputs(
        matrix,
        output_dir / "benchmark_matrix.json",
        output_dir / "benchmark_matrix.csv",
    )
    print(f"[import-babilong] aggregate: {len(matrix['rows'])} rows -> {output_dir}")
    return provenance


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Import pinned BABILong raw results from the five published checkpoints."
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="validate existing raw files and rebuild derived logs/matrices without network",
    )
    args = parser.parse_args(argv)
    import_results(args.output_dir.resolve(), offline=args.offline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
