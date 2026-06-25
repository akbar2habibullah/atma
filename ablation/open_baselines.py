"""Evaluate pretrained Hugging Face causal LMs on the ablation metrics.

This is the no-training companion to ablation.train:

    python -m ablation.open_baselines --log_dir ablation/open_logs

For each model it emits the same structured blocks understood by
ablation.parse_logs:

  ABLATION_CONFIG_JSON
  ABLATION_CURVE_JSON   (empty: pretrained baseline, no train curve)
  ABLATION_EVAL_JSON
  ABLATION_ERROR_JSON   (on failure)

Metrics mirror the ablation eval at full context:
  clean_ppl[L]  : coherent single-document prefixes.
  junk_ppl[L]   : retokenized GPT-2 FineWeb-Edu validation stream.
  needle[d]     : induction needle CE + greedy value-token accuracy.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import socket
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from ablation.config_schema import EVAL_LENGTHS


OPEN_BASELINE_MODELS = [
    "google/gemma-3-270m",
    "LiquidAI/LFM2.5-230M-Base",
    "LiquidAI/LFM2.5-350M-Base",
    "Qwen/Qwen3-0.6B-Base",
    "HuggingFaceTB/SmolLM2-360M",
    "ibm-granite/granite-4.0-350m-base",
    "tiiuae/Falcon-H1-0.5B-Base",
    "Qwen/Qwen3.5-0.8B-Base",
]


def _slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_id).strip("_")


def _emit_block(fh, name: str, obj):
    fh.write(f"\n==={name}===\n{json.dumps(obj)}\n===END===\n")


def _log_open(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return open(path, "w", buffering=1, encoding="utf-8")


def _empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _eos_id(tokenizer):
    for name in ("eos_token_id", "bos_token_id", "pad_token_id"):
        val = getattr(tokenizer, name, None)
        if val is not None:
            return int(val)
    return None


def _model_context_limit(model, tokenizer) -> int | None:
    cfg = getattr(model, "config", None)
    candidates = []
    for obj in (cfg, getattr(cfg, "text_config", None), tokenizer):
        if obj is None:
            continue
        for name in (
            "max_position_embeddings",
            "max_sequence_length",
            "seq_length",
            "n_positions",
            "model_max_length",
        ):
            val = getattr(obj, name, None)
            if isinstance(val, int) and 0 < val < 10**9:
                candidates.append(val)
    return max(candidates) if candidates else None


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def select_long_docs(
    dataset_id: str,
    tokenizer,
    text_key: str,
    split: str,
    min_tokens: int,
    num_docs: int,
    dataset_config: str | None = None,
    streaming: bool = True,
):
    """Select model-tokenized long documents as CPU int64 tensors."""
    from datasets import load_dataset

    kwargs = {"split": split, "streaming": streaming}
    if dataset_config:
        kwargs["name"] = dataset_config
    ds = load_dataset(dataset_id, **kwargs)

    need = min_tokens + 1
    char_min = max(1024, need)
    docs, scanned = [], 0
    eos = _eos_id(tokenizer)

    for row in ds:
        scanned += 1
        text = row.get(text_key)
        if not text or len(text) < char_min:
            continue
        ids = _encode(tokenizer, text)
        if eos is not None:
            ids = [eos] + ids
        if len(ids) >= need:
            docs.append(torch.tensor(ids[:need], dtype=torch.int64))
            if len(docs) >= num_docs:
                break
    print(f"[open-baseline] selected {len(docs)}/{num_docs} clean docs "
          f"from {dataset_id} after scanning {scanned:,} rows")
    return docs


class RetokenizedGpt2Stream:
    """Decode GPT-2-tokenized .bin validation shards, then retokenize for a target model.

    This keeps the "junk" condition close to the original ablation's FineWeb-Edu
    concatenated stream while respecting each open model's native tokenizer.
    """

    def __init__(self, pattern: str, target_tokenizer, source_tokenizer_name: str = "gpt2"):
        from transformers import AutoTokenizer
        from train.data import _load_data_shard, get_data

        if not glob.glob(pattern) and pattern == "finewebedu10B/finewebedu_val_*.bin":
            get_data("finewebedu_val_000000.bin")

        files = sorted(Path.cwd().glob(pattern))
        if not files:
            raise FileNotFoundError(f"no validation shards match {pattern!r}")
        self.files = files
        self._load_data_shard = _load_data_shard
        self.source_tokenizer = AutoTokenizer.from_pretrained(source_tokenizer_name)
        self.target_tokenizer = target_tokenizer
        self.file_idx = 0
        self.tokens = self._load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def _take_source_tokens(self, n: int) -> list[int]:
        out = []
        while len(out) < n:
            if self.pos >= len(self.tokens):
                self.file_idx = (self.file_idx + 1) % len(self.files)
                self.tokens = self._load_data_shard(self.files[self.file_idx])
                self.pos = 0
            take = min(n - len(out), len(self.tokens) - self.pos)
            out.extend(int(x) for x in self.tokens[self.pos:self.pos + take].tolist())
            self.pos += take
        return out

    def next_ids(self, length: int) -> torch.Tensor:
        need = length + 1
        source_take = max(need + 128, int(need * 1.25))
        text_parts = []
        target_ids: list[int] = []
        while len(target_ids) < need:
            src = self._take_source_tokens(source_take)
            text_parts.append(self.source_tokenizer.decode(src, skip_special_tokens=True))
            target_ids = _encode(self.target_tokenizer, "".join(text_parts))
            source_take = max(256, source_take // 2)
        return torch.tensor(target_ids[:need], dtype=torch.int64)


@torch.inference_mode()
def sequence_loss(model, ids: torch.Tensor, device, chunk_size: int) -> float:
    """Full-context CE by streaming input chunks through the KV cache."""
    input_ids = ids[:-1].to(device=device, dtype=torch.long).view(1, -1)
    targets = ids[1:].to(device=device, dtype=torch.long).view(-1)
    past = None
    total, count = 0.0, 0
    for start in range(0, input_ids.shape[1], chunk_size):
        end = min(start + chunk_size, input_ids.shape[1])
        out = model(input_ids=input_ids[:, start:end], past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits.float().view(-1, out.logits.shape[-1])
        tgt = targets[start:end]
        loss = F.cross_entropy(logits, tgt, reduction="sum")
        total += loss.item()
        count += int(tgt.numel())
        del out, logits, tgt, loss
    return total / max(count, 1)


@torch.inference_mode()
def tail_value_logits(model, ids: torch.Tensor, n_last: int, device, chunk_size: int):
    """Return logits for the last n_last targets of ids[1:]."""
    input_ids = ids[:-1].to(device=device, dtype=torch.long).view(1, -1)
    first_keep = input_ids.shape[1] - n_last
    past = None
    kept = []
    for start in range(0, input_ids.shape[1], chunk_size):
        end = min(start + chunk_size, input_ids.shape[1])
        out = model(input_ids=input_ids[:, start:end], past_key_values=past, use_cache=True)
        past = out.past_key_values
        if end > first_keep:
            offset = max(first_keep - start, 0)
            kept.append(out.logits[:, offset:].float().cpu())
        del out
    logits = torch.cat(kept, dim=1)[0]
    return logits[-n_last:]


def _safe_eval_lengths(lengths, context_limit: int | None, respect_model_max: bool):
    if not respect_model_max or context_limit is None:
        return list(lengths)
    return [L for L in lengths if L <= context_limit]


@torch.inference_mode()
def clean_perplexity(model, docs, lengths, device, chunk_size: int):
    out = {}
    for L in lengths:
        total, n = 0.0, 0
        try:
            for doc in docs:
                if doc.numel() < L + 1:
                    continue
                total += sequence_loss(model, doc[:L + 1], device, chunk_size) * L
                n += L
                _empty_cuda_cache()
            out[L] = total / n if n else None
        except torch.cuda.OutOfMemoryError:
            _empty_cuda_cache()
            out[L] = None
    return out


@torch.inference_mode()
def junk_perplexity(model, junk_stream, lengths, num_seqs: int, device, chunk_size: int):
    out = {}
    for L in lengths:
        total, n = 0.0, 0
        try:
            for _ in range(num_seqs):
                ids = junk_stream.next_ids(L)
                total += sequence_loss(model, ids, device, chunk_size) * L
                n += L
                _empty_cuda_cache()
            out[L] = total / n if n else None
        except torch.cuda.OutOfMemoryError:
            _empty_cuda_cache()
            out[L] = None
    return out


@torch.inference_mode()
def needle_retrieval(model, tokenizer, haystack, distances, num_trials, vlen, device, chunk_size: int):
    random.seed(1234)
    eos = _eos_id(tokenizer)
    prefix = [eos] if eos is not None else []

    ce = {d: 0.0 for d in distances}
    acc = {d: 0.0 for d in distances}
    done = {d: 0 for d in distances}
    base_ce, base_done = 0.0, 0
    d0 = min(distances)

    def make_needle():
        key = random.randint(10**6, 10**7 - 1)
        digits = [random.randint(0, 9) for _ in range(vlen)]
        cue_ids = _encode(tokenizer, f" The access code for record {key} is")
        val_ids = _encode(tokenizer, "".join(f" {x}" for x in digits))
        return cue_ids, val_ids

    for t in range(num_trials):
        hay = haystack[t % len(haystack)].tolist()
        cue, val = make_needle()
        if not val:
            continue
        needle = cue + val
        val_t = torch.tensor(val, dtype=torch.long)

        try:
            g = hay[:d0 + len(needle)]
            seq = torch.tensor(prefix + g + cue + val, dtype=torch.int64)
            lg = tail_value_logits(model, seq, len(val), device, chunk_size)
            base_ce += F.cross_entropy(lg, val_t, reduction="mean").item()
            base_done += 1

            for d in distances:
                gap = hay[:d]
                seq = torch.tensor(prefix + needle + gap + cue + val, dtype=torch.int64)
                lg = tail_value_logits(model, seq, len(val), device, chunk_size)
                ce[d] += F.cross_entropy(lg, val_t, reduction="mean").item()
                acc[d] += (lg.argmax(-1) == val_t).float().mean().item()
                done[d] += 1
                _empty_cuda_cache()
        except torch.cuda.OutOfMemoryError:
            _empty_cuda_cache()
            continue

    needle = {}
    for d in distances:
        n = max(done[d], 1)
        needle[d] = {"ce": ce[d] / n, "acc": 100.0 * acc[d] / n, "trials": done[d]}
    return needle, base_ce / max(base_done, 1)


def _load_model(model_id: str, args, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(args.dtype)
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32

    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer


def run_one(model_id: str, args, device):
    started = time.perf_counter()
    model, tokenizer = _load_model(model_id, args, device)
    num_params = sum(p.numel() for p in model.parameters())
    context_limit = _model_context_limit(model, tokenizer)
    lengths = _safe_eval_lengths(args.lengths, context_limit, args.respect_model_max)
    distances = _safe_eval_lengths(args.needle_distances, context_limit, args.respect_model_max)
    if not lengths or not distances:
        raise ValueError(f"{model_id} has no eval lengths after applying context limit {context_limit}")

    need = max(max(lengths), max(distances) + 128)
    docs = select_long_docs(
        args.clean_dataset,
        tokenizer,
        args.clean_text_key,
        args.clean_split,
        need,
        args.num_eval_docs,
        dataset_config=args.clean_dataset_config,
        streaming=not args.no_streaming,
    )
    result = {
        "clean_ppl": {},
        "junk_ppl": {},
        "needle": {},
        "needle_baseline": None,
        "num_clean_docs": len(docs),
    }

    if docs:
        result["clean_ppl"] = clean_perplexity(model, docs, lengths, device, args.chunk_size)
        needle, base = needle_retrieval(
            model,
            tokenizer,
            docs,
            distances,
            args.num_needle_trials,
            args.needle_val_len,
            device,
            args.chunk_size,
        )
        result["needle"] = needle
        result["needle_baseline"] = base
    else:
        print("[open-baseline] WARNING: no clean docs found; clean_ppl/needle skipped")

    junk_stream = RetokenizedGpt2Stream(args.junk_bin, tokenizer, args.junk_source_tokenizer)
    result["junk_ppl"] = junk_perplexity(
        model,
        junk_stream,
        lengths,
        args.num_eval_docs,
        device,
        args.chunk_size,
    )
    elapsed = time.perf_counter() - started
    result.update(
        {
            "num_params": num_params,
            "train_elapsed_s": 0.0,
            "eval_elapsed_s": round(elapsed, 3),
            "mfu_final": None,
            "model_context_limit": context_limit,
            "chunk_size": args.chunk_size,
        }
    )
    del model
    _empty_cuda_cache()
    return result, num_params, context_limit


def make_config(model_id: str, args, num_params=None, context_limit=None):
    run_id = "open__" + _slug(model_id)
    return {
        "run_id": run_id,
        "attn_type": "open",
        "reg_mode": "pretrained",
        "distractor": False,
        "memory": False,
        "window": False,
        "model_id": model_id,
        "source": "huggingface",
        "eval_lengths": list(args.lengths),
        "needle_distances": list(args.needle_distances),
        "clean_dataset": args.clean_dataset,
        "clean_dataset_config": args.clean_dataset_config,
        "clean_split": args.clean_split,
        "junk_bin": args.junk_bin,
        "num_eval_docs": args.num_eval_docs,
        "num_needle_trials": args.num_needle_trials,
        "needle_val_len": args.needle_val_len,
        "chunk_size": args.chunk_size,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": args.trust_remote_code,
        "respect_model_max": args.respect_model_max,
        "model_context_limit": context_limit,
        "num_params": num_params,
        "host": socket.gethostname(),
        "device": str(args.device),
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Run open-weight pretrained baselines.")
    ap.add_argument("--models", nargs="+", default=OPEN_BASELINE_MODELS,
                    help="HF model ids to evaluate")
    ap.add_argument("--log_dir", default="ablation/open_logs",
                    help="directory for one structured .log per model")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    ap.add_argument("--attn_implementation", default="sdpa",
                    help="Transformers attention backend, e.g. sdpa/eager/flash_attention_2; empty disables")
    ap.add_argument("--trust_remote_code", action="store_true",
                    help="allow model repos to provide custom modeling code")
    ap.add_argument("--lengths", type=int, nargs="+", default=list(EVAL_LENGTHS))
    ap.add_argument("--needle_distances", type=int, nargs="+", default=list(EVAL_LENGTHS))
    ap.add_argument("--respect_model_max", action="store_true",
                    help="skip lengths above the model config/tokenizer max instead of extrapolating")
    ap.add_argument("--chunk_size", type=int, default=512,
                    help="tokens per cached forward chunk; lower if logits/KV memory OOMs")
    ap.add_argument("--clean_dataset", default="codelion/finepdfs-100M")
    ap.add_argument("--clean_dataset_config", default=None)
    ap.add_argument("--clean_text_key", default="text")
    ap.add_argument("--clean_split", default="train")
    ap.add_argument("--no_streaming", action="store_true",
                    help="load clean dataset normally instead of HF streaming")
    ap.add_argument("--junk_bin", default="finewebedu10B/finewebedu_val_*.bin",
                    help="GPT-2-tokenized FineWeb-Edu .bin stream to decode and retokenize")
    ap.add_argument("--junk_source_tokenizer", default="gpt2")
    ap.add_argument("--num_eval_docs", type=int, default=16)
    ap.add_argument("--num_needle_trials", type=int, default=16)
    ap.add_argument("--needle_val_len", type=int, default=5)
    ap.add_argument("--stop_on_error", action="store_true",
                    help="abort the run after the first model failure")
    return ap.parse_args()


def main():
    args = parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)
    os.makedirs(args.log_dir, exist_ok=True)

    for model_id in args.models:
        cfg = make_config(model_id, args)
        log_path = os.path.join(args.log_dir, f"{cfg['run_id']}.log")
        fh = _log_open(log_path)

        def p0(s: str):
            print(s)
            fh.write(s + "\n")

        p0("=" * 100)
        p0(f"[open-baseline] model={model_id} host={socket.gethostname()} device={device} "
           f"torch={torch.__version__}")
        p0("=" * 100)
        try:
            eval_res, num_params, context_limit = run_one(model_id, args, device)
            cfg = make_config(model_id, args, num_params=num_params, context_limit=context_limit)
            _emit_block(fh, "ABLATION_CONFIG_JSON", cfg)
            _emit_block(fh, "ABLATION_CURVE_JSON", [])
            _emit_block(fh, "ABLATION_EVAL_JSON", eval_res)
            p0("[open-baseline] eval:")
            p0(f"  clean_ppl(nats): {eval_res.get('clean_ppl')}")
            p0(f"  junk_ppl(nats):  {eval_res.get('junk_ppl')}")
            p0(f"  needle:          "
               f"{ {d: round(v['acc'], 1) for d, v in (eval_res.get('needle') or {}).items()} }")
            p0(f"  needle_baseline CE: {eval_res.get('needle_baseline')}")
            p0("[open-baseline] DONE")
        except Exception:
            tb = traceback.format_exc()
            _emit_block(fh, "ABLATION_CONFIG_JSON", cfg)
            _emit_block(fh, "ABLATION_CURVE_JSON", [])
            _emit_block(fh, "ABLATION_ERROR_JSON", {"error": tb})
            p0("[open-baseline] FAILED:\n" + tb)
            fh.close()
            if args.stop_on_error:
                raise
            continue
        fh.close()


if __name__ == "__main__":
    main()
