"""Synthetic long-context retrieval — passkey & needle-in-a-haystack (NIAH).

Generate-and-match, so it fits the autoregressive inference interface (like BABILong). No
external dataset: the haystack is built deterministically with the GPT-2 tokenizer (the model's
tokenizer) so every (length, depth) cell is an EXACT token length and the needle sits at a
controlled depth. Covers the realistic-at-370M subset of RULER (niah_single); multi-key and the
full RULER 13-task pipeline are deferred (heavy data-gen, ~0 signal at 370M base).

This complements two existing things:
  - BABILong tests retrieval+*reasoning*; this tests pure retrieval at length.
  - eval.py's induction-needle scores the value by loglikelihood via direct forward; this scores
    it by *generation* through the served engine — the deployed-model view of the same capability.
"""
import json
import random
import time

from benchmarks.babilong import compare_answers

# classic passkey filler (Mohtashami & Jaggi); repeated to pad the haystack.
_FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
           "Here we go. There and back again. ")

_TOK = None


def _tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("gpt2")
    return _TOK


def _parse_len(s):
    s = str(s).lower().strip()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1024 * 1024)
    return int(s)


def make_sample(kind, target_tokens, depth, rng, haystack_ids=None):
    """Build one (prompt_ids, answer) at an exact token length with the needle at `depth`.

    kind="passkey": numeric key in repeated filler.
    kind="niah":    a 'magic number' bound to a random word, in filler (or real text if
                    haystack_ids is given).
    """
    tok = _tokenizer()
    if kind == "passkey":
        key = str(rng.randint(10 ** 6, 10 ** 7 - 1))
        needle = f" The pass key is {key}. Remember it. {key} is the pass key. "
        question = f" What is the pass key? The pass key is"
        answer = key
    else:  # niah
        key = str(rng.randint(10 ** 6, 10 ** 7 - 1))
        word = rng.choice(["ocean", "garden", "mountain", "river", "forest", "desert"])
        needle = f" The special magic {word} number is {key}. "
        question = f" What is the special magic {word} number? The special magic {word} number is"
        answer = key

    needle_ids = tok.encode(needle)
    q_ids = tok.encode(question)
    budget = max(0, target_tokens - len(needle_ids) - len(q_ids))

    if haystack_ids is not None:
        body = list(haystack_ids[:budget])
    else:
        unit = tok.encode(_FILLER)
        reps = budget // max(len(unit), 1) + 1
        body = (unit * reps)[:budget]

    insert = int(depth * len(body))
    full = body[:insert] + needle_ids + body[insert:] + q_ids
    return full, answer


def run_retrieval(model, kinds, lengths, depths, num_samples=10, max_tokens=16,
                  seed=1234, haystack=None, log_fn=print):
    """Evaluate retrieval accuracy on a (kind, length, depth) grid. Returns a results dict.
    Generation goes through the production inference adapter."""
    rng = random.Random(seed)
    length_toks = [(_l, _parse_len(_l)) for _l in lengths]
    hay_ids = None
    if haystack:
        from datasets import load_dataset
        ds = load_dataset(haystack, split="train", streaming=True)
        tok = _tokenizer()
        big = []
        for row in ds:                                   # accumulate enough real text
            big += tok.encode(row.get("text", "") + "\n")
            if len(big) >= max(t for _, t in length_toks) + 64:
                break
        hay_ids = big

    results = {k: {} for k in kinds}
    t0 = time.perf_counter()
    for kind in kinds:
        for lname, ltok in length_toks:
            for depth in depths:
                prompts, answers = [], []
                for _ in range(num_samples):
                    ids, ans = make_sample(kind, ltok, depth, rng, hay_ids)
                    prompts.append(ids)                  # list[int] prompt (engine accepts token ids)
                    answers.append(ans)
                gens = model.generate(prompts, max_tokens=max_tokens)
                acc = 100.0 * sum(compare_answers(g, a) for g, a in zip(gens, answers)) / num_samples
                results[kind].setdefault(lname, {})[str(depth)] = acc
                log_fn(f"[retrieval] {kind:>7} len={lname:>4} depth={depth:>4}: acc={acc:.1f}%")

    return {
        "benchmark": "retrieval",
        "kinds": kinds,
        "lengths": lengths,
        "depths": depths,
        "num_samples": num_samples,
        "haystack": haystack or "synthetic-filler",
        "results": results,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "unsupported_checkpoint": bool(getattr(model, "wip", [])),
    }


def emit_log(fh, model, res):
    fh.write("\n[retrieval] accuracy %, per kind (rows=length, cols=depth):\n")
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
    fh.write("\n===RETRIEVAL_RESULTS_JSON===\n")
    fh.write(json.dumps({**res, "model_config": getattr(model, "cfg", {}),
                         "model_unsupported": getattr(model, "wip", []),
                         "serving_metrics": getattr(model, "last_metrics", None)}))
    fh.write("\n===END===\n")
