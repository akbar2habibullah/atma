"""BABILong harness — long-context reasoning-in-a-haystack (bAbI tasks embedded in PG-19).

Paper: arXiv 2406.10149 (NeurIPS 2024 D&B). Dataset: RMT-team/babilong[-1k-samples].
This is a GENERATIVE QA benchmark, a natural fit for the autoregressive generate() interface.

Two important notes:
  1. BABILong is a *fine-tuned* capability probe. A 370M BASE model scores ~0 zero-shot (it
     won't follow the QA format). The decisive protocol (RMT/Mamba 130-137M reached 90%+ on
     qa1 to 1M+ tokens) is: fine-tune a candidate on bAbI qa1 (then qa1-5) with a length
     curriculum, THEN run this harness across the length configs. The memory-on vs memory-off
     contrast here is the single most decisive test of the Titans branch.

For leaderboard-parity prompts, `pip install babilong` and this harness uses its official
DEFAULT_PROMPTS automatically; otherwise it falls back to the built-in minimal templates below.
"""
import json
import re
import time

# BABILong length configs (HF dataset subset names) and the standard bAbI task splits.
LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1M"]
TASKS = [f"qa{i}" for i in range(1, 21)]

# --- Built-in fallback prompts (minimal; official babilong.prompts is preferred) -------------
# instruction + (context) + question + post_prompt. Marked DEFAULT — for exact leaderboard
# numbers install the `babilong` package so the official per-task prompts/few-shot are used.
_DEFAULT_PROMPTS = {
    "qa1": dict(
        instruction="I will give you context with facts about people's locations hidden in "
                    "irrelevant text, and a question. Answer using only the facts; if a person "
                    "was in several places, use the most recent one.",
        post_prompt="Your answer must be a single word (the location). Do not explain."),
    "qa2": dict(
        instruction="I will give you context with facts about people and the objects they carry "
                    "hidden in irrelevant text, and a question. Answer using only the facts.",
        post_prompt="Your answer must be a single word (the location). Do not explain."),
    "qa3": dict(
        instruction="I will give you context with facts about people, objects, and locations "
                    "over time hidden in irrelevant text, and a question. Answer using only the "
                    "facts and the order of events.",
        post_prompt="Your answer must be a single word (the location). Do not explain."),
    "qa4": dict(
        instruction="I will give you context with facts about the relative positions of objects "
                    "hidden in irrelevant text, and a question. Answer using only the facts.",
        post_prompt="Your answer must be a single word (a direction or object). Do not explain."),
    "qa5": dict(
        instruction="I will give you context with facts about people giving and receiving objects "
                    "hidden in irrelevant text, and a question. Answer using only the facts.",
        post_prompt="Your answer must be a single word. Do not explain."),
}
_GENERIC_PROMPT = dict(
    instruction="I will give you a long context with a few relevant facts hidden in irrelevant "
                "text, followed by a question. Answer using only the relevant facts.",
    post_prompt="Answer with as few words as possible. Do not explain.")


def _load_official_prompts():
    """Return (DEFAULT_PROMPTS, get_formatted_input) from the official babilong package, or None."""
    try:
        from babilong.prompts import DEFAULT_PROMPTS, get_formatted_input  # type: ignore
        return DEFAULT_PROMPTS, get_formatted_input
    except Exception:
        return None


def format_prompt(task, context, question, official=None):
    if official is not None:
        DEFAULT_PROMPTS, get_formatted_input = official
        p = DEFAULT_PROMPTS.get(task, {})
        return get_formatted_input(context, question, p.get("examples", ""),
                                   p.get("instruction", ""), p.get("post_prompt", ""),
                                   template=p.get("template"))
    p = _DEFAULT_PROMPTS.get(task, _GENERIC_PROMPT)
    return (f"{p['instruction']}\n\n<context>\n{context}\n</context>\n\n"
            f"Question: {question}\n{p['post_prompt']}\nAnswer:")


_norm_re = re.compile(r"[^a-z0-9 ,]+")


def _normalize(s):
    return _norm_re.sub(" ", str(s).lower()).strip()


def compare_answers(output, target):
    """BABILong-style match: every comma-separated target token must appear in the output
    (case/punctuation-insensitive). Single-word answers reduce to substring containment."""
    out = _normalize(output)
    parts = [t.strip() for t in _normalize(target).split(",") if t.strip()]
    if not parts:
        return False
    out_tokens = set(out.split())
    return all((t in out_tokens) or (t in out) for t in parts)


def _columns(row):
    ctx = row.get("input") if "input" in row else row.get("context", "")
    q = row.get("question", "")
    tgt = row.get("target") if "target" in row else row.get("answer", "")
    return ctx, q, tgt


def run_babilong(model, tasks, lengths, num_samples=100,
                 dataset_id="RMT-team/babilong-1k-samples", max_tokens=16, log_fn=print):
    """Evaluate `model` (benchmarks.EvalModel) on BABILong. Returns a results dict
    {task: {length: accuracy}} plus metadata."""
    from datasets import load_dataset

    official = _load_official_prompts()
    log_fn(f"[babilong] prompts = {'official babilong.prompts' if official else 'built-in defaults'}")
    results = {t: {} for t in tasks}
    counts = {t: {} for t in tasks}
    t_start = time.perf_counter()

    for length in lengths:
        for task in tasks:
            try:
                ds = load_dataset(dataset_id, length, split=task)
            except Exception as e:
                log_fn(f"[babilong] skip {task}@{length}: {e}")
                continue
            n = min(num_samples, len(ds))
            rows = ds.select(range(n))
            prompts, targets = [], []
            for r in rows:
                ctx, q, tgt = _columns(r)
                prompts.append(format_prompt(task, ctx, q, official))
                targets.append(tgt)
            gens = model.generate(prompts, max_tokens=max_tokens)
            correct = sum(compare_answers(g, t) for g, t in zip(gens, targets))
            acc = 100.0 * correct / n if n else None
            results[task][length] = acc
            counts[task][length] = n
            log_fn(f"[babilong] {task:>4} @ {length:>4}: acc={acc:.1f}% (n={n})")

    return {
        "benchmark": "babilong",
        "dataset_id": dataset_id,
        "tasks": tasks,
        "lengths": lengths,
        "num_samples": num_samples,
        "results": results,
        "counts": counts,
        "elapsed_s": round(time.perf_counter() - t_start, 1),
        "unsupported_checkpoint": bool(getattr(model, "wip", [])),
    }


def emit_log(fh, model, res):
    """Write a self-describing log: human table + a dashboard-parseable JSON block."""
    fh.write("\n[babilong] results (accuracy %, rows=task, cols=length):\n")
    cols = res["lengths"]
    fh.write("  task  " + "  ".join(f"{c:>6}" for c in cols) + "\n")
    for t in res["tasks"]:
        row = res["results"].get(t, {})
        fh.write(f"  {t:>4}  " + "  ".join(
            (f"{row[c]:6.1f}" if row.get(c) is not None else f"{'—':>6}") for c in cols) + "\n")
    if res["unsupported_checkpoint"]:
        fh.write("\n  *** INVALID: checkpoint is unsupported by the benchmark inference path ***\n")
    fh.write("\n===BABILONG_RESULTS_JSON===\n")
    fh.write(json.dumps({**res, "model_config": getattr(model, "cfg", {}),
                         "model_unsupported": getattr(model, "wip", [])}))
    fh.write("\n===END===\n")
