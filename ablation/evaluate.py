"""Structured post-training evaluation for the ablation grid.

Reuses eval.py's memory-safe building blocks (_blocks_forward + time-chunked head, so the
T x vocab logit spike never materializes) and select_long_docs. Returns plain dicts (no
printing-only), at FULL context for every length/distance:

  clean_ppl[L]  : nats/token on coherent long docs (clean_dataset), nested prefixes.
  junk_ppl[L]   : nats/token on the concatenated .bin val stream.
  needle[d]     : {ce, acc} induction needle at gap d, plus a needle-absent baseline.

All evaluation is at full context: any training-time sliding window is removed first.
"""
import random

import torch
import torch.nn.functional as F

from eval import _blocks_forward, _chunked_loss, select_long_docs
from train.data import data_generator

LOGIT_SOFTCAP = 15.0   # mirrors Model.forward: 15 * z * rsqrt(z^2 + 15^2)


def _unwrap(model):
    return getattr(model, "_orig_mod", model)


def _set_full_context(model):
    """Drop any training-time sliding window on every attention layer -> eval at full context."""
    for block in _unwrap(model).blocks:
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "window"):
            attn.window = None


def _value_logits(model, inp, n_last):
    """Soft-capped logits for the last n_last positions (mirrors Model.forward)."""
    x = _blocks_forward(model, inp)
    z = model.proj(model.norm(x[:, -n_last:])).float()
    return (LOGIT_SOFTCAP * z * (z.square() + LOGIT_SOFTCAP ** 2).rsqrt())[0]   # (n_last, vocab)


@torch.no_grad()
def clean_perplexity(model, docs, lengths, device, loss_chunk=8192):
    """nats/token on nested prefixes of the SAME coherent docs (fair across lengths)."""
    out = {}
    for L in lengths:
        tot, n = 0.0, 0
        try:
            for d in docs:
                if d.numel() < L + 1:
                    continue
                buf = d[:L + 1]
                x = buf[:-1].view(1, -1).to(device, torch.int32)
                tgt = buf[1:].view(1, -1).to(device, torch.int64)
                xb = _blocks_forward(model, x)
                ls, c = _chunked_loss(model, xb, tgt, chunk=loss_chunk)
                tot += ls; n += c
                torch.cuda.empty_cache()
            out[L] = (tot / n) if n else None
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); out[L] = None
    return out


@torch.no_grad()
def junk_perplexity(model, val_data, lengths, num_seqs, device, loss_chunk=8192):
    """nats/token on the concatenated .bin val stream (incoherent across doc boundaries)."""
    out = {}
    for L in lengths:
        tot, n = 0.0, 0
        try:
            gen = data_generator(val_data, L, seq_len=L)
            for _ in range(num_seqs):
                x, tgt = next(gen)
                xb = _blocks_forward(model, x)
                ls, c = _chunked_loss(model, xb, tgt, chunk=loss_chunk)
                tot += ls; n += c
                torch.cuda.empty_cache()
            out[L] = (tot / n) if n else None
        except (torch.cuda.OutOfMemoryError, StopIteration):
            torch.cuda.empty_cache(); out[L] = None
    return out


@torch.no_grad()
def needle_retrieval(model, haystack, distances, num_trials, vlen, device):
    """Induction needle-in-haystack: plant a unique-key sentence + spaced-digit value at the
    start, re-present the cue at the end with a gap of `distance` real tokens, score CE +
    greedy per-digit accuracy on the value. `haystack` = list of CPU int64 token tensors
    (each >= max(distances)+overhead). Returns {dist: {ce, acc}} and a needle-absent baseline."""
    from transformers import AutoTokenizer
    random.seed(1234)
    tok = AutoTokenizer.from_pretrained("gpt2")
    eot = tok.eos_token_id

    ce = {d: 0.0 for d in distances}
    acc = {d: 0.0 for d in distances}
    cnt = {d: 0 for d in distances}
    base_ce = 0.0
    base_cnt = 0
    d0 = min(distances)

    for t in range(num_trials):
        hay = haystack[t % len(haystack)]
        key = random.randint(10 ** 6, 10 ** 7 - 1)
        digits = [random.randint(0, 9) for _ in range(vlen)]
        cue = tok.encode(f" The access code for record {key} is")
        val = tok.encode("".join(f" {x}" for x in digits))
        needle = cue + val
        val_t = torch.tensor(val, device=device)

        # needle-absent baseline: cue appears only at the query (the model's prior on the value)
        try:
            g = hay[:d0 + len(needle)].tolist()
            seq = [eot] + g + cue + val
            inp = torch.tensor(seq[:-1], dtype=torch.int32, device=device).view(1, -1)
            base_ce += F.cross_entropy(_value_logits(model, inp, len(val)), val_t, reduction="mean").item()
            base_cnt += 1
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

        for d in distances:
            try:
                gap = hay[:d].tolist()
                seq = [eot] + needle + gap + cue + val          # needle at start, query at end, gap = d
                inp = torch.tensor(seq[:-1], dtype=torch.int32, device=device).view(1, -1)
                lg = _value_logits(model, inp, len(val))
                ce[d] += F.cross_entropy(lg, val_t, reduction="mean").item()
                acc[d] += (lg.argmax(-1) == val_t).float().mean().item()
                cnt[d] += 1
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

    return ({d: ({"ce": ce[d] / cnt[d], "acc": 100.0 * acc[d] / cnt[d]} if cnt[d] else None)
             for d in distances},
            (base_ce / base_cnt) if base_cnt else None)


@torch.no_grad()
def run_eval(model, cfg: dict, device):
    """Full structured eval. cfg is the resolved RunConfig dict. Returns a JSON-able dict."""
    model = _unwrap(model)
    model.eval()
    _set_full_context(model)

    lengths = cfg["eval_lengths"]
    distances = cfg["needle_distances"]
    # one dataset scan sized for the largest need (clean-ppl prefix OR needle gap+scaffold)
    overhead = 64
    need = max(max(lengths), max(distances) + overhead)
    docs = select_long_docs(cfg["clean_dataset"], "text", "train", need, cfg["num_eval_docs"])

    result = {"clean_ppl": {}, "junk_ppl": {}, "needle": {}, "needle_baseline": None,
              "num_clean_docs": len(docs)}

    if docs:
        result["clean_ppl"] = clean_perplexity(model, docs, lengths, device)
        needle, base = needle_retrieval(model, docs, distances, cfg["num_needle_trials"],
                                        cfg["needle_val_len"], device)
        result["needle"] = needle
        result["needle_baseline"] = base
    else:
        print("[evaluate] WARNING: no clean docs found -> clean_ppl/needle skipped.")

    result["junk_ppl"] = junk_perplexity(model, cfg["val_data"], lengths,
                                         cfg["num_eval_docs"], device)
    model.train()
    return result
