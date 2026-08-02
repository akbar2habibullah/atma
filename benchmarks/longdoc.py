"""Fixed-target long-document likelihood controls.

For each document, every context length predicts the same target token span. This removes the
position/content confound in prefix-average loss, where longer contexts otherwise score a
different and usually easier region of the document.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from dataclasses import dataclass

from benchmarks.retrieval import _clear_after_oom, _is_cuda_oom


@dataclass(frozen=True)
class LongDocSpec:
    dataset_id: str
    config: str | None
    split: str
    text_field: str = "text"
    loader: str | None = None
    data_file: str | None = None
    streaming: bool = True


LONGDOC_SPECS = {
    "pg19": LongDocSpec("emozilla/pg19", None, "test"),
    "proof_pile": LongDocSpec(
        "hoskinson-center/proof-pile", None, "test", loader="json",
        data_file="test/proofpile_test.jsonl.gz",
    ),
    "finepdfs": LongDocSpec(
        "codelion/finepdfs-1B", None, "train", loader="parquet",
        data_file="data/*.parquet", streaming=False,
    ),
}


def _parse_length(value):
    text = str(value).strip().lower()
    if text.endswith("k"):
        return int(float(text[:-1]) * 1024)
    if text.endswith("m"):
        return int(float(text[:-1]) * 1024 * 1024)
    return int(text)


def _load_stream(spec: LongDocSpec, revision: str | None):
    from datasets import load_dataset

    if spec.loader and spec.data_file:
        pinned = revision or "main"
        uri = f"hf://datasets/{spec.dataset_id}@{pinned}/{spec.data_file}"
        return load_dataset(
            spec.loader, data_files={spec.split: uri}, split=spec.split,
            streaming=spec.streaming,
        )

    args = [spec.dataset_id]
    if spec.config:
        args.append(spec.config)
    kwargs = {"split": spec.split, "streaming": True}
    if revision:
        kwargs["revision"] = revision
    return load_dataset(*args, **kwargs)


def _token_hash(ids):
    digest = hashlib.sha256()
    for token in ids:
        digest.update(struct.pack("<I", int(token)))
    return digest.hexdigest()


def select_documents(
    tokenizer,
    spec: LongDocSpec,
    *,
    revision: str | None,
    required_tokens: int,
    num_docs: int,
    max_scan: int,
    log_fn,
):
    dataset = _load_stream(spec, revision)
    iterator = iter(dataset)
    documents = []
    scanned = 0
    for row in iterator:
        scanned += 1
        text = row.get(spec.text_field)
        if not text or len(text) < required_tokens:
            if scanned >= max_scan:
                break
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= required_tokens:
            documents.append(ids[:required_tokens])
            log_fn(
                f"[longdoc] selected {len(documents)}/{num_docs} from {spec.dataset_id} "
                f"after scanning {scanned} rows"
            )
            if len(documents) >= num_docs:
                break
        if scanned >= max_scan:
            break
    close = getattr(iterator, "close", None)
    if close is not None:
        close()
    return documents, scanned


def run_longdoc(
    scorer,
    datasets,
    lengths,
    *,
    target_tokens: int = 256,
    num_docs: int = 8,
    max_scan: int = 100_000,
    dataset_revisions: dict | None = None,
    log_fn=print,
):
    dataset_revisions = dataset_revisions or {}
    parsed = [(str(label), _parse_length(label)) for label in lengths]
    max_context = max(value for _, value in parsed)
    required = max_context + target_tokens
    results = {}
    t0 = time.perf_counter()

    for name in datasets:
        spec = LONGDOC_SPECS[name]
        revision = dataset_revisions.get(spec.dataset_id)
        docs, scanned = select_documents(
            scorer.tokenizer,
            spec,
            revision=revision,
            required_tokens=required,
            num_docs=num_docs,
            max_scan=max_scan,
            log_fn=log_fn,
        )
        document_hashes = [_token_hash(doc) for doc in docs]
        dataset_result = {
            "dataset_id": spec.dataset_id,
            "dataset_revision": revision,
            "documents": len(docs),
            "scanned_rows": scanned,
            "target_tokens_per_document": target_tokens,
            "document_token_hashes": document_hashes,
            "lengths": {},
        }
        if not docs:
            dataset_result["error"] = (
                f"no rows with at least {required} GPT-2 tokens in first {max_scan} rows"
            )
            results[name] = dataset_result
            continue

        target_start = max_context
        target_spans = [doc[target_start:target_start + target_tokens] for doc in docs]
        target_bytes = [
            len(scorer.tokenizer.decode(tokens).encode("utf-8")) for tokens in target_spans
        ]

        for label, context_length in parsed:
            total_nll = 0.0
            total_tokens = 0
            total_bytes = 0
            completed = 0
            error = None
            for doc, target, byte_count in zip(docs, target_spans, target_bytes):
                context = doc[target_start - context_length:target_start]
                try:
                    score = scorer.score_token_ids(context, target)
                except Exception as exc:
                    if not _is_cuda_oom(exc):
                        raise
                    error = str(exc)[:500]
                    _clear_after_oom()
                    break
                finally:
                    clear = getattr(scorer, "clear_cache", None)
                    if clear is not None:
                        clear()
                total_nll -= score["loglikelihood"]
                total_tokens += score["tokens"]
                total_bytes += byte_count
                completed += 1

            if completed:
                nll = total_nll / total_tokens
                try:
                    perplexity = math.exp(nll)
                except OverflowError:
                    perplexity = float("inf")
                cell = {
                    "documents": completed,
                    "nll_nats_per_token": nll,
                    "perplexity": perplexity,
                    "bits_per_byte": total_nll / math.log(2.0) / total_bytes,
                    "target_tokens": total_tokens,
                    "target_bytes": total_bytes,
                    "oom": error is not None,
                    "error": error,
                }
            else:
                cell = {"documents": 0, "oom": error is not None, "error": error}
            dataset_result["lengths"][label] = cell
            if completed:
                log_fn(
                    f"[longdoc] {name:>10} context={label:>5} docs={completed} "
                    f"nll={cell['nll_nats_per_token']:.4f} bpb={cell['bits_per_byte']:.4f}"
                    + (" OOM" if error else "")
                )
            else:
                log_fn(f"[longdoc] {name:>10} context={label:>5}: unavailable ({error})")
        results[name] = dataset_result

    return {
        "benchmark": "longdoc",
        "protocol": "fixed-target-v1",
        "datasets": list(datasets),
        "lengths": list(lengths),
        "target_tokens": target_tokens,
        "requested_documents": num_docs,
        "dataset_revisions": dataset_revisions,
        "results": results,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "model_config": scorer.cfg,
    }


def emit_log(fh, result):
    fh.write("\n===LONGDOC_RESULTS_JSON===\n")
    fh.write(json.dumps(result))
    fh.write("\n===END===\n")
