"""Zero-shot base-model quality controls using conditional loglikelihood only."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    config: str | None
    split: str


BASE_TASK_SPECS = {
    "lambada": DatasetSpec("EleutherAI/lambada_openai", None, "test"),
    "hellaswag": DatasetSpec("Rowan/hellaswag", None, "validation"),
    "piqa": DatasetSpec("ybisk/piqa", None, "validation"),
    "winogrande": DatasetSpec("allenai/winogrande", "winogrande_xl", "validation"),
    "arc_easy": DatasetSpec("allenai/ai2_arc", "ARC-Easy", "validation"),
    "arc_challenge": DatasetSpec("allenai/ai2_arc", "ARC-Challenge", "validation"),
    "openbookqa": DatasetSpec("allenai/openbookqa", "main", "validation"),
    "boolq": DatasetSpec("google/boolq", None, "validation"),
}

PRIMARY_METRIC = {
    "lambada": "accuracy",
    "hellaswag": "accuracy_norm",
    "piqa": "accuracy_norm",
    "winogrande": "accuracy",
    "arc_easy": "accuracy_norm",
    "arc_challenge": "accuracy_norm",
    "openbookqa": "accuracy_norm",
    "boolq": "accuracy",
}


def _hellaswag_preprocess(text):
    text = str(text).strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text)


def _choice_example(task: str, row: dict):
    """Return `(context, choices, gold_index)` for a supported MCQ row."""
    if task == "hellaswag":
        context = (
            str(row.get("activity_label", ""))
            + ": "
            + str(row.get("ctx_a", ""))
            + " "
            + str(row.get("ctx_b", "")).capitalize()
        )
        choices = [_hellaswag_preprocess(value) for value in row["endings"]]
        return _hellaswag_preprocess(context), choices, int(row["label"])

    if task == "piqa":
        return str(row["goal"]), [str(row["sol1"]), str(row["sol2"])], int(row["label"])

    if task == "winogrande":
        before, after = str(row["sentence"]).split("_", 1)
        context = before.rstrip()
        choices = [f"{option}{after}" for option in (row["option1"], row["option2"])]
        return context, choices, int(row["answer"]) - 1

    if task in {"arc_easy", "arc_challenge", "openbookqa"}:
        question = row.get("question") or row.get("question_stem")
        choice_block = row["choices"]
        choices = [str(value) for value in choice_block["text"]]
        labels = [str(value) for value in choice_block["label"]]
        answer = str(row["answerKey"])
        if answer not in labels and answer.isdigit():
            index = int(answer) - 1
        else:
            index = labels.index(answer)
        return str(question), choices, index

    if task == "boolq":
        question = str(row["question"]).strip()
        if not question.endswith("?"):
            question += "?"
        context = f"{row['passage']}\nQuestion: {question}\nAnswer:"
        return context, ["no", "yes"], 1 if bool(row["answer"]) else 0

    raise KeyError(task)


def _continuation(choice: str) -> str:
    return choice if choice.startswith((" ", "\n", "\t")) else " " + choice


def _mean_and_se(correct: int, count: int):
    value = correct / count if count else None
    se = math.sqrt(value * (1.0 - value) / count) if count and value is not None else None
    return value, se


def _load_dataset(spec: DatasetSpec, revision: str | None):
    from datasets import load_dataset

    args = [spec.dataset_id]
    if spec.config:
        args.append(spec.config)
    kwargs = {"split": spec.split}
    if revision:
        kwargs["revision"] = revision
    return load_dataset(*args, **kwargs)


def _run_lambada(scorer, rows, log_fn, batch_size):
    pattern = re.compile(r"^(.*?)(\s+\S+)\s*$", re.DOTALL)
    pairs = []
    for row in rows:
        match = pattern.match(str(row["text"]))
        if match:
            pairs.append((match.group(1), match.group(2)))

    scores = scorer.score_pairs(pairs, batch_size=batch_size)
    correct = sum(score["greedy_exact"] for score in scores)
    tokens = sum(score["tokens"] for score in scores)
    total_nll = -sum(score["loglikelihood"] for score in scores)
    accuracy, se = _mean_and_se(correct, len(scores))
    result = {
        "samples": len(scores),
        "accuracy": accuracy,
        "accuracy_se": se,
        "target_tokens": tokens,
        "target_nll": total_nll / tokens if tokens else None,
        "target_perplexity": math.exp(total_nll / tokens) if tokens else None,
        "primary_metric": "accuracy",
    }
    log_fn(
        f"[base] lambada samples={len(scores)} acc={100 * accuracy:.2f}% "
        f"target_ppl={result['target_perplexity']:.3f}"
    )
    return result


def _run_multiple_choice(task, scorer, rows, log_fn, batch_size):
    pairs = []
    questions = []
    for row in rows:
        context, choices, gold = _choice_example(task, row)
        questions.append((len(pairs), len(choices), gold))
        pairs.extend((context, _continuation(choice)) for choice in choices)

    scores = scorer.score_pairs(pairs, batch_size=batch_size)
    raw_correct = norm_correct = valid = 0
    for index, (start, count, gold) in enumerate(questions):
        choice_scores = scores[start:start + count]
        raw = max(
            range(len(choice_scores)),
            key=lambda i: choice_scores[i]["loglikelihood"],
        )
        normalized = max(
            range(len(choice_scores)),
            key=lambda i: choice_scores[i]["mean_loglikelihood"],
        )
        raw_correct += int(raw == gold)
        norm_correct += int(normalized == gold)
        valid += 1
        if (index + 1) % 250 == 0:
            log_fn(f"[base] {task}: {index + 1}/{len(rows)}")

    raw_acc, raw_se = _mean_and_se(raw_correct, valid)
    norm_acc, norm_se = _mean_and_se(norm_correct, valid)
    primary = PRIMARY_METRIC[task]
    result = {
        "samples": valid,
        "accuracy": raw_acc,
        "accuracy_se": raw_se,
        "accuracy_norm": norm_acc,
        "accuracy_norm_se": norm_se,
        "primary_metric": primary,
        "primary_value": raw_acc if primary == "accuracy" else norm_acc,
    }
    log_fn(
        f"[base] {task} samples={valid} acc={100 * raw_acc:.2f}% "
        f"acc_norm={100 * norm_acc:.2f}% primary={primary}"
    )
    return result


def run_base_tasks(
    scorer,
    tasks,
    *,
    limit: int | None = None,
    dataset_revisions: dict | None = None,
    batch_size: int = 8,
    log_fn=print,
):
    dataset_revisions = dataset_revisions or {}
    results = {}
    t0 = time.perf_counter()
    for task in tasks:
        spec = BASE_TASK_SPECS[task]
        revision = dataset_revisions.get(spec.dataset_id)
        dataset = _load_dataset(spec, revision)
        count = min(limit, len(dataset)) if limit else len(dataset)
        rows = [dataset[i] for i in range(count)]
        log_fn(
            f"[base] task={task} dataset={spec.dataset_id}@{revision or 'main'} "
            f"samples={count}"
        )
        if task == "lambada":
            results[task] = _run_lambada(scorer, rows, log_fn, batch_size)
        else:
            results[task] = _run_multiple_choice(
                task, scorer, rows, log_fn, batch_size
            )

    return {
        "benchmark": "base",
        "protocol": "atma-base-loglikelihood-v1",
        "tasks": list(tasks),
        "limit": limit,
        "dataset_revisions": dataset_revisions,
        "results": results,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "model_config": scorer.cfg,
        "scoring_max_length": scorer.max_length,
    }


def emit_log(fh, result):
    fh.write("\n===BASE_RESULTS_JSON===\n")
    fh.write(json.dumps(result))
    fh.write("\n===END===\n")
