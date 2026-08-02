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
from pathlib import Path

# Controlled adaptation uses only <=2K data and evaluates length extrapolation through exactly
# 2**18 = 262,144 tokens. The official 100-row dataset has every config in this range.
DEFAULT_DATASET_ID = "RMT-team/babilong"
TRAIN_LENGTHS = ("0k", "1k", "2k")
EVAL_LENGTHS = ("0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k")
LENGTHS = list(EVAL_LENGTHS)
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


def select_row_ids(dataset, row_start=90, row_end=100, num_samples=None):
    """Select a deterministic half-open row range and return (rows, row_ids)."""
    if row_start < 0 or row_end <= row_start:
        raise ValueError("row range must satisfy 0 <= row_start < row_end")
    if row_end > len(dataset):
        raise ValueError(
            f"requested rows [{row_start}, {row_end}) but split has only {len(dataset)} rows"
        )
    row_ids = list(range(row_start, row_end))
    if num_samples is not None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        row_ids = row_ids[:num_samples]
    return dataset.select(row_ids), row_ids


def _is_oom_error(exc):
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "outofmemory" in name or "out of memory" in text


def _finetune_manifest(model):
    """Load the protocol manifest beside a fine-tuned checkpoint, when present."""
    checkpoint_dir = (
        getattr(model, "checkpoint_dir", None)
        or getattr(model, "ckpt_dir", None)
    )
    if checkpoint_dir is None:
        return None
    path = Path(checkpoint_dir) / "finetune_manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid BABILong fine-tune manifest: {path}") from exc


def run_babilong(model, tasks, lengths, num_samples=100,
                 dataset_id=DEFAULT_DATASET_ID, dataset_revision=None,
                 max_tokens=16, row_start=90, row_end=100, log_fn=print):
    """Evaluate `model` on BABILong. Returns a results dict
    {task: {length: accuracy}} plus metadata."""
    from datasets import load_dataset

    unknown_tasks = sorted(set(tasks) - set(TASKS))
    unknown_lengths = sorted(set(lengths) - set(EVAL_LENGTHS))
    if unknown_tasks:
        raise ValueError(f"unknown BABILong tasks: {unknown_tasks}")
    if unknown_lengths:
        raise ValueError(
            f"unsupported BABILong lengths: {unknown_lengths}; controlled eval stops at "
            "256k (262,144 tokens)"
        )

    official = _load_official_prompts()
    prompt_protocol = "official-babilong" if official else "builtin-v1"
    log_fn(f"[babilong] prompts = {prompt_protocol}")
    manifest = _finetune_manifest(model)
    if manifest is not None:
        manifest_dataset = manifest.get("dataset_id")
        if manifest_dataset != dataset_id:
            raise ValueError(
                "BABILong train/eval dataset mismatch: fine-tuned on "
                f"{manifest_dataset!r}, evaluating on {dataset_id!r}"
            )
        manifest_revision = manifest.get("dataset_revision")
        if dataset_revision is None:
            dataset_revision = manifest_revision
        elif manifest_revision != dataset_revision:
            raise ValueError(
                "BABILong train/eval dataset revision mismatch: fine-tuned on "
                f"{manifest_revision!r}, evaluating on {dataset_revision!r}"
            )
        reserved_rows = manifest.get("reserved_test_rows")
        if reserved_rows != [row_start, row_end]:
            raise ValueError(
                "BABILong held-out row mismatch: fine-tune manifest reserves "
                f"{reserved_rows}, evaluation requested {[row_start, row_end]}"
            )

    if manifest is not None and manifest.get("prompt_protocol") != prompt_protocol:
        raise ValueError(
            "BABILong train/eval prompt mismatch: fine-tuned with "
            f"{manifest.get('prompt_protocol')!r}, evaluating with {prompt_protocol!r}"
        )

    log_fn(
        f"[babilong] held-out rows = [{row_start}, {row_end}); "
        f"dataset = {dataset_id}@{dataset_revision or 'main'}"
    )
    results = {t: {} for t in tasks}
    counts = {t: {} for t in tasks}
    selected_rows = {t: {} for t in tasks}
    oom_cells = []
    t_start = time.perf_counter()

    for length in lengths:
        for task in tasks:
            try:
                kwargs = {"split": task}
                if dataset_revision:
                    kwargs["revision"] = dataset_revision
                ds = load_dataset(dataset_id, length, **kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"failed to load required BABILong cell {task}@{length} from "
                    f"{dataset_id}@{dataset_revision or 'main'}"
                ) from e
            rows, row_ids = select_row_ids(
                ds, row_start=row_start, row_end=row_end, num_samples=num_samples
            )
            n = len(rows)
            prompts, targets = [], []
            for r in rows:
                ctx, q, tgt = _columns(r)
                prompts.append(format_prompt(task, ctx, q, official))
                targets.append(tgt)
            try:
                gens = model.generate(prompts, max_tokens=max_tokens)
            except Exception as exc:
                if not _is_oom_error(exc):
                    raise
                oom_cells.append({"task": task, "length": length, "error": str(exc)})
                results[task][length] = None
                counts[task][length] = n
                selected_rows[task][length] = row_ids
                log_fn(f"[babilong] OOM {task}@{length}: {exc}")
                if hasattr(model, "clear_cache"):
                    model.clear_cache()
                continue
            correct = sum(compare_answers(g, t) for g, t in zip(gens, targets))
            acc = 100.0 * correct / n if n else None
            results[task][length] = acc
            counts[task][length] = n
            selected_rows[task][length] = row_ids
            rendered = f"{acc:.1f}%" if acc is not None else "n/a"
            log_fn(f"[babilong] {task:>4} @ {length:>4}: acc={rendered} (n={n})")
            if hasattr(model, "clear_cache"):
                model.clear_cache()

    macro_average = {}
    macro_task_counts = {}
    for length in lengths:
        values = [
            results[task].get(length)
            for task in tasks
            if results[task].get(length) is not None
        ]
        macro_task_counts[length] = len(values)
        macro_average[length] = (
            sum(values) / len(values) if len(values) == len(tasks) else None
        )

    return {
        "benchmark": "babilong",
        "protocol": "heldout-short-finetune-v1",
        "prompt_protocol": prompt_protocol,
        "generation_backend": type(model).__name__,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "tasks": tasks,
        "lengths": lengths,
        "num_samples": num_samples,
        "results": results,
        "counts": counts,
        "macro_average": macro_average,
        "macro_task_counts": macro_task_counts,
        "row_range": [row_start, row_end],
        "row_ids": selected_rows,
        "oom_cells": oom_cells,
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
    macro = res.get("macro_average", {})
    fh.write("  mean  " + "  ".join(
        (f"{macro[c]:6.1f}" if macro.get(c) is not None else f"{'—':>6}") for c in cols
    ) + "\n")
    if res["unsupported_checkpoint"]:
        fh.write("\n  *** INVALID: checkpoint is unsupported by the benchmark inference path ***\n")
    fh.write("\n===BABILONG_RESULTS_JSON===\n")
    fh.write(json.dumps({**res, "model_config": getattr(model, "cfg", {}),
                         "model_unsupported": getattr(model, "wip", []),
                         "serving_metrics": getattr(model, "last_metrics", None)}))
    fh.write("\n===END===\n")
