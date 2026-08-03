"""Training-aligned synthetic and real-text needle retrieval.

These checkpoints are base language models, not instruction-tuned generators. Retrieval is
therefore measured the same way as scaled_ablation.eval_hf_checkpoints: run the training
model's full-context forward path, teacher-force a short digit value, and report token accuracy,
exact-value accuracy, and cross entropy. Serving generation is benchmarked separately.
"""
import gc
import json
import random
import re
import time

# classic passkey filler (Mohtashami & Jaggi); repeated to pad the haystack.
_FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
           "Here we go. There and back again. ")

_TOK = None
_INTEGER_RE = re.compile(r"(?<!\d)\d+(?!\d)")


def compare_retrieval_answer(output, target):
    """Exact numeric retrieval match with digit boundaries.

    The previous BABILong substring scorer accepted a target such as ``1234567`` inside
    ``12345678``. Retrieval answers are generated integer keys, so extracting complete integer
    spans keeps harmless prose ("the key is ...") while rejecting partial/extended keys.
    """
    target = str(target).strip()
    return target in _INTEGER_RE.findall(str(output))


def _is_cuda_oom(exc):
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "gpu" in message)


def _clear_after_oom():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("gpt2")
        _TOK.model_max_length = 10**30
    return _TOK


def _parse_len(s):
    s = str(s).lower().strip()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1024 * 1024)
    return int(s)


def make_sample(kind, target_tokens, depth, rng, haystack_ids=None, value_tokens=5):
    """Build an exact-length (context_ids, value_ids) teacher-forced needle."""
    tok = _tokenizer()
    if kind == "passkey":
        record = rng.randint(10 ** 6, 10 ** 7 - 1)
        cue = f" The access code for record {record} is"
    else:  # niah
        word = rng.choice(["ocean", "garden", "mountain", "river", "forest", "desert"])
        cue = f" The special magic {word} number is"

    digits = [rng.randint(0, 9) for _ in range(value_tokens)]
    value_ids = tok.encode("".join(f" {digit}" for digit in digits))
    cue_ids = tok.encode(cue)
    needle_ids = cue_ids + value_ids
    scaffold = 1 + len(needle_ids) + len(cue_ids)
    if target_tokens < scaffold:
        raise ValueError(
            f"target length {target_tokens} is shorter than needle scaffold {scaffold}"
        )
    budget = target_tokens - scaffold

    if haystack_ids is not None:
        body = list(haystack_ids[:budget])
    else:
        unit = tok.encode(_FILLER)
        reps = budget // max(len(unit), 1) + 1
        body = (unit * reps)[:budget]

    insert = int(depth * len(body))
    context = [tok.eos_token_id] + body[:insert] + needle_ids + body[insert:] + cue_ids
    assert len(context) == target_tokens
    return context, value_ids


def run_retrieval(scorer, kinds, lengths, depths, num_samples=10, value_tokens=5,
                  seed=1234, haystack=None, haystack_revision=None, log_fn=print):
    """Evaluate retrieval accuracy on a (kind, length, depth) grid. Returns a results dict.
    Scoring goes through the checkpoint-exact training model forward path. Every sample keeps
    the same cue and value across lengths and depths so cells are paired."""
    length_toks = [(_l, _parse_len(_l)) for _l in lengths]
    hay_ids = None
    if haystack:
        from datasets import load_dataset
        dataset_kwargs = {"split": "train", "streaming": True}
        if haystack_revision:
            dataset_kwargs["revision"] = haystack_revision
        ds = load_dataset(haystack, **dataset_kwargs)
        tok = _tokenizer()
        big = []
        for row in ds:                                   # accumulate enough real text
            big += tok.encode(row.get("text", "") + "\n")
            if len(big) >= max(t for _, t in length_toks) + 64:
                break
        required_haystack = max(t for _, t in length_toks)
        if len(big) < required_haystack:
            raise RuntimeError(
                f"{haystack} yielded only {len(big)} GPT-2 tokens; "
                f"{required_haystack} are required for the requested grid"
            )
        hay_ids = big

    results = {k: {} for k in kinds}
    exact_results = {k: {} for k in kinds}
    nll_results = {k: {} for k in kinds}
    oom_cells = []
    t0 = time.perf_counter()
    for kind in kinds:
        for lname, ltok in length_toks:
            for depth in depths:
                correct_tokens = total_tokens = exact = completed = 0
                total_nll = 0.0
                error = None
                for sample_index in range(num_samples):
                    sample_rng = random.Random(f"{seed}:{kind}:{sample_index}")
                    context_ids, target_ids = make_sample(
                        kind, ltok, depth, sample_rng, hay_ids, value_tokens=value_tokens
                    )
                    try:
                        score = scorer.score_token_ids(context_ids, target_ids)
                        correct_tokens += score["correct_tokens"]
                        total_tokens += score["tokens"]
                        exact += int(score["greedy_exact"])
                        total_nll -= score["loglikelihood"]
                        completed += 1
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

                depth_key = str(depth)
                if error is not None or completed != num_samples:
                    oom_cells.append({
                        "kind": kind,
                        "length": lname,
                        "depth": depth,
                        "error": error,
                    })
                    results[kind].setdefault(lname, {})[depth_key] = None
                    exact_results[kind].setdefault(lname, {})[depth_key] = None
                    nll_results[kind].setdefault(lname, {})[depth_key] = None
                    log_fn(f"[retrieval] {kind:>7} len={lname:>4} depth={depth:>4}: OOM")
                    continue

                token_acc = 100.0 * correct_tokens / total_tokens
                exact_acc = 100.0 * exact / completed
                nll = total_nll / total_tokens
                results[kind].setdefault(lname, {})[depth_key] = token_acc
                exact_results[kind].setdefault(lname, {})[depth_key] = exact_acc
                nll_results[kind].setdefault(lname, {})[depth_key] = nll
                log_fn(
                    f"[retrieval] {kind:>7} len={lname:>4} depth={depth:>4}: "
                    f"token_acc={token_acc:.1f}% exact={exact_acc:.1f}% nll={nll:.4f}"
                )

    return {
        "benchmark": "retrieval",
        "protocol": "teacher-forced-needle-v2",
        "kinds": kinds,
        "lengths": lengths,
        "depths": depths,
        "num_samples": num_samples,
        "value_tokens": value_tokens,
        "seed": seed,
        "paired_across_lengths_and_depths": True,
        "haystack": haystack or "synthetic-filler",
        "haystack_revision": haystack_revision,
        "results": results,
        "exact_results": exact_results,
        "nll_results": nll_results,
        "oom_cells": oom_cells,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "unsupported_checkpoint": False,
    }


def emit_log(fh, scorer, res):
    fh.write("\n[retrieval] teacher-forced token accuracy %, per kind (rows=length, cols=depth):\n")
    depths = [str(d) for d in res["depths"]]
    for kind in res["kinds"]:
        fh.write(f"  {kind}:\n")
        fh.write("    len   " + "  ".join(f"{d:>5}" for d in depths) + "\n")
        for lname in res["lengths"]:
            row = res["results"].get(kind, {}).get(lname, {})
            fh.write(f"   {lname:>5}  " + "  ".join(
                (f"{row[d]:5.1f}" if row.get(d) is not None else f"{'—':>5}") for d in depths) + "\n")
    if res["unsupported_checkpoint"]:
        fh.write("\n  *** INVALID: checkpoint is unsupported by the benchmark inference path ***\n")
    if res.get("oom_cells"):
        fh.write(f"\n  OOM cells: {len(res['oom_cells'])} (recorded as null)\n")
    fh.write("\n===RETRIEVAL_RESULTS_JSON===\n")
    fh.write(json.dumps({**res, "model_config": getattr(scorer, "cfg", {}),
                         "model_unsupported": [], "serving_metrics": None}))
    fh.write("\n===END===\n")
