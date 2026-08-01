"""Controlled prefill/decode throughput, memory, and maximum-context sweep."""

from __future__ import annotations

import gc
import json
import time

from benchmarks.longdoc import _parse_length
from benchmarks.retrieval import _is_cuda_oom


def _prompt_ids(tokenizer, target_tokens):
    unit = tokenizer.encode(
        "The grass is green. The sky is blue. The sun is yellow. ",
        add_special_tokens=False,
    )
    repetitions = target_tokens // len(unit) + 1
    return (unit * repetitions)[:target_tokens]


def _clear_cuda():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_serving(
    model_path,
    lengths,
    *,
    decode_tokens=32,
    samples=1,
    max_num_seqs=1,
    max_num_batched_tokens=None,
    strict=True,
    log_fn=print,
):
    import torch
    from transformers import AutoTokenizer

    from benchmarks.model import EvalModel
    from benchmarks.model import read_checkpoint_config

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    results = {}
    max_success = None
    t0 = time.perf_counter()
    for label in lengths:
        context_tokens = _parse_length(label)
        engine = None
        _clear_cuda()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            budget = context_tokens + decode_tokens + 64
            engine = EvalModel(
                model_path,
                max_tokens=decode_tokens,
                strict=strict,
                quiet=True,
                max_model_len=budget,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max(max_num_batched_tokens or 0, budget, 16384),
            )
            prompt = _prompt_ids(tokenizer, context_tokens)
            outputs = engine.generate(
                [prompt for _ in range(samples)], max_tokens=decode_tokens, use_tqdm=False
            )
            metrics = engine.last_call_metrics or {}
            peak_allocated = (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            )
            peak_reserved = (
                torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
            )
            cell = {
                "context_tokens": context_tokens,
                "samples": samples,
                "requested_decode_tokens": decode_tokens,
                "generated_texts": len(outputs),
                "prefill_tokens": metrics.get("prefill_tokens", 0),
                "decode_tokens": metrics.get("decode_tokens", 0),
                "prefill_time_s": metrics.get("prefill_time", 0.0),
                "decode_time_s": metrics.get("decode_time", 0.0),
                "prefill_tokens_per_s": metrics.get("prefill_throughput", 0.0),
                "decode_tokens_per_s": metrics.get("decode_throughput", 0.0),
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "wall_time_s": round(time.perf_counter() - started, 3),
                "oom": False,
            }
            max_success = context_tokens
            log_fn(
                f"[serving] context={label:>5} prefill={cell['prefill_tokens_per_s']:.1f} "
                f"tok/s decode={cell['decode_tokens_per_s']:.1f} tok/s "
                f"peak={peak_reserved / 2**30:.2f} GiB"
            )
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            cell = {
                "context_tokens": context_tokens,
                "samples": samples,
                "requested_decode_tokens": decode_tokens,
                "oom": True,
                "error": str(exc)[:500],
                "wall_time_s": round(time.perf_counter() - started, 3),
            }
            log_fn(f"[serving] context={label:>5}: OOM")
        finally:
            if engine is not None:
                engine.close()
            _clear_cuda()
        results[str(label)] = cell

    return {
        "benchmark": "serving",
        "protocol": "exact-token-prefill-v1",
        "lengths": list(lengths),
        "decode_tokens": decode_tokens,
        "samples": samples,
        "max_num_seqs": max_num_seqs,
        "max_successful_context_tokens": max_success,
        "results": results,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "model_config": read_checkpoint_config(model_path),
    }


def emit_log(fh, result):
    fh.write("\n===SERVING_RESULTS_JSON===\n")
    fh.write(json.dumps(result))
    fh.write("\n===END===\n")
