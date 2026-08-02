"""Controlled BABILong supervised fine-tuning on 0K/1K/2K contexts only."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.babilong import (
    DEFAULT_DATASET_ID,
    TASKS,
    TRAIN_LENGTHS,
    _columns,
    _load_official_prompts,
    format_prompt,
)
from benchmarks.scoring import DirectScorer, encode_pair


@dataclass(frozen=True)
class EncodedExample:
    task: str
    length: str
    row_id: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    target_tokens: int
    unpadded_tokens: int


def validate_protocol(
    train_lengths,
    *,
    seq_len: int,
    train_start: int,
    train_end: int,
    val_start: int,
    val_end: int,
):
    unknown = sorted(set(train_lengths) - set(TRAIN_LENGTHS))
    if unknown:
        raise ValueError(
            f"fine-tuning lengths must be a subset of {list(TRAIN_LENGTHS)}; got {unknown}"
        )
    if not train_lengths:
        raise ValueError("at least one fine-tuning length is required")
    if seq_len <= 0 or seq_len > 2048:
        raise ValueError("BABILong fine-tuning seq_len must be in [1, 2048]")
    if not (0 <= train_start < train_end <= val_start < val_end):
        raise ValueError(
            "row ranges must be ordered and disjoint: "
            "0 <= train_start < train_end <= val_start < val_end"
        )

    if val_end > 90:
        raise ValueError("validation rows must end by 90; rows 90..99 are reserved for test")


def resolve_dataset_revision(dataset_id: str, requested_revision: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id=dataset_id, revision=requested_revision)
    if not info.sha:
        raise RuntimeError(
            f"could not resolve immutable revision for {dataset_id}@{requested_revision}"
        )
    return info.sha


def encode_example(
    tokenizer,
    *,
    task: str,
    length: str,
    row_id: int,
    row: dict,
    seq_len: int,
    official_prompts=None,
) -> EncodedExample:
    context, question, target = _columns(row)
    prompt = format_prompt(task, context, question, official_prompts)
    request = encode_pair(tokenizer, prompt, " " + str(target).strip())
    eos = tokenizer.eos_token_id
    context_ids = list(request.context_ids)
    continuation_ids = list(request.continuation_ids)
    full_ids = context_ids + continuation_ids + [eos]
    input_ids = full_ids[:-1]
    labels = (
        [-100] * (len(context_ids) - 1)
        + continuation_ids
        + [eos]
    )
    if len(input_ids) != len(labels):
        raise AssertionError("BABILong input/label alignment failed")
    if len(input_ids) > seq_len:
        raise ValueError(
            f"{task}@{length} row {row_id} needs {len(input_ids)} tokens, "
            f"exceeding fine-tune seq_len={seq_len}; do not truncate task evidence"
        )
    target_tokens = sum(label != -100 for label in labels)
    pad = seq_len - len(input_ids)
    return EncodedExample(
        task=task,
        length=length,
        row_id=row_id,
        input_ids=tuple(input_ids + [eos] * pad),
        labels=tuple(labels + [-100] * pad),
        target_tokens=target_tokens,
        unpadded_tokens=len(input_ids),
    )


def load_examples(
    tokenizer,
    *,
    dataset_id: str,
    dataset_revision: str,
    tasks,
    lengths,
    row_start: int,
    row_end: int,
    seq_len: int,
    official_prompts,
    log_fn=print,
):
    from datasets import load_dataset

    output = []
    for length in lengths:
        for task in tasks:
            dataset = load_dataset(
                dataset_id,
                length,
                split=task,
                revision=dataset_revision,
            )
            if row_end > len(dataset):
                raise ValueError(
                    f"{task}@{length} has {len(dataset)} rows, cannot select "
                    f"[{row_start}, {row_end})"
                )
            for row_id in range(row_start, row_end):
                output.append(
                    encode_example(
                        tokenizer,
                        task=task,
                        length=length,
                        row_id=row_id,
                        row=dataset[row_id],
                        seq_len=seq_len,
                        official_prompts=official_prompts,
                    )
                )
            log_fn(
                f"[babilong-ft] loaded {task}@{length} rows "
                f"[{row_start}, {row_end})"
            )
    return output


def _batch(examples, device):
    import torch

    inputs = torch.tensor(
        [example.input_ids for example in examples],
        dtype=torch.int32,
        device=device,
    )
    labels = torch.tensor(
        [example.labels for example in examples],
        dtype=torch.long,
        device=device,
    )
    return inputs, labels


def evaluate_nll(model, examples, *, micro_batch_size: int, device) -> float:
    import torch

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, len(examples), micro_batch_size):
            batch = examples[start:start + micro_batch_size]
            inputs, labels = _batch(batch, device)
            loss, _, _ = model(inputs, labels)
            total_loss += float(loss.item())
            total_tokens += sum(example.target_tokens for example in batch)
            del inputs, labels, loss
    model.train()
    return total_loss / max(total_tokens, 1)


def _checkpoint_model(model):
    return getattr(model, "_orig_mod", model)


def save_checkpoint(model, scorer, output_dir: Path, metadata: dict):
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_model = _checkpoint_model(model)
    state = {
        key.removeprefix("_orig_mod."): value.detach().cpu()
        for key, value in raw_model.state_dict().items()
    }
    temporary = output_dir / "weights.pt.tmp"
    torch.save({"model": state, "finetune": metadata}, temporary)
    temporary.replace(output_dir / "weights.pt")
    (output_dir / "config.json").write_text(
        json.dumps(scorer.cfg, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    scorer.tokenizer.save_pretrained(output_dir)
    (output_dir / "finetune_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _lr_factor(step: int, total_steps: int, warmup_ratio: float, min_lr_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
        1.0 + math.cos(math.pi * progress)
    )


def run_finetune(args):
    import torch

    validate_protocol(
        args.train_lengths,
        seq_len=args.seq_len,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
    )
    unknown_tasks = sorted(set(args.tasks) - set(TASKS[:10]))
    if unknown_tasks:
        raise ValueError(f"controlled protocol supports qa1..qa10; got {unknown_tasks}")
    if args.epochs <= 0 or args.micro_batch_size <= 0 or args.grad_accum_steps <= 0:
        raise ValueError("epochs, micro_batch_size, and grad_accum_steps must be positive")

    if args.lr <= 0 or args.grad_clip <= 0:
        raise ValueError("lr and grad_clip must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if not 0.0 < args.min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in (0, 1]")
    existing_checkpoint = args.output_dir / "weights.pt"
    if existing_checkpoint.exists():
        raise FileExistsError(
            f"refusing to overwrite existing fine-tuned checkpoint: {existing_checkpoint}"
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    revision = resolve_dataset_revision(args.dataset, args.dataset_revision)
    official_prompts = _load_official_prompts()
    prompt_protocol = "official-babilong" if official_prompts else "builtin-v1"
    print(
        f"[babilong-ft] dataset={args.dataset}@{revision} "
        f"prompts={prompt_protocol}"
    )

    scorer = DirectScorer(
        args.model,
        device=args.device,
        max_length=args.seq_len,
        batch_size=args.micro_batch_size,
    )
    model = scorer.model
    device = scorer.device
    try:
        train_examples = load_examples(
            scorer.tokenizer,
            dataset_id=args.dataset,
            dataset_revision=revision,
            tasks=args.tasks,
            lengths=args.train_lengths,
            row_start=args.train_start,
            row_end=args.train_end,
            seq_len=args.seq_len,
            official_prompts=official_prompts,
        )
        val_examples = load_examples(
            scorer.tokenizer,
            dataset_id=args.dataset,
            dataset_revision=revision,
            tasks=args.tasks,
            lengths=args.train_lengths,
            row_start=args.val_start,
            row_end=args.val_end,
            seq_len=args.seq_len,
            official_prompts=official_prompts,
        )
        maximum = max(example.unpadded_tokens for example in train_examples + val_examples)
        print(
            f"[babilong-ft] train={len(train_examples)} val={len(val_examples)} "
            f"seq_len={args.seq_len} max_unpadded={maximum}"
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            fused=device.type == "cuda",
        )
        examples_per_update = args.micro_batch_size * args.grad_accum_steps
        updates_per_epoch = math.ceil(len(train_examples) / examples_per_update)
        total_steps = updates_per_epoch * args.epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: _lr_factor(
                step, total_steps, args.warmup_ratio, args.min_lr_ratio
            ),
        )

        curve = []
        best_val_nll = float("inf")
        best_epoch = None
        global_step = 0
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            indices = list(range(len(train_examples)))
            random.Random(args.seed + epoch).shuffle(indices)
            model.train()
            epoch_loss = 0.0
            epoch_tokens = 0
            for group_start in range(0, len(indices), examples_per_update):
                group_ids = indices[group_start:group_start + examples_per_update]
                group = [train_examples[index] for index in group_ids]
                group_tokens = sum(example.target_tokens for example in group)
                optimizer.zero_grad(set_to_none=True)
                for start in range(0, len(group), args.micro_batch_size):
                    micro = group[start:start + args.micro_batch_size]
                    inputs, labels = _batch(micro, device)
                    loss, _, _ = model(inputs, labels)
                    (loss / group_tokens).backward()
                    epoch_loss += float(loss.item())
                    epoch_tokens += sum(example.target_tokens for example in micro)
                    del inputs, labels, loss
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(
                        f"non-finite gradient norm at update {global_step}: {grad_norm}"
                    )
                optimizer.step()
                scheduler.step()
                global_step += 1

            val_nll = evaluate_nll(
                model,
                val_examples,
                micro_batch_size=args.micro_batch_size,
                device=device,
            )
            record = {
                "epoch": epoch,
                "step": global_step,
                "train_nll": epoch_loss / max(epoch_tokens, 1),
                "val_nll": val_nll,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": round(time.perf_counter() - started, 1),
            }
            curve.append(record)
            print("[babilong-ft] " + json.dumps(record, sort_keys=True))
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_epoch = epoch
                metadata = {
                    "protocol": "heldout-short-finetune-v1",
                    "source_checkpoint": str(Path(args.model).resolve()),
                    "source_repo_id": args.source_repo_id,
                    "source_revision": args.source_revision,
                    "dataset_id": args.dataset,
                    "dataset_revision": revision,
                    "prompt_protocol": prompt_protocol,
                    "tasks": list(args.tasks),
                    "train_lengths": list(args.train_lengths),
                    "seq_len": args.seq_len,
                    "train_rows": [args.train_start, args.train_end],
                    "validation_rows": [args.val_start, args.val_end],
                    "reserved_test_rows": [90, 100],
                    "optimizer": "AdamW",
                    "lr": args.lr,
                    "betas": [args.beta1, args.beta2],
                    "eps": args.eps,
                    "weight_decay": args.weight_decay,
                    "epochs": args.epochs,
                    "micro_batch_size": args.micro_batch_size,
                    "grad_accum_steps": args.grad_accum_steps,
                    "examples_per_update": examples_per_update,
                    "seed": args.seed,
                    "best_epoch": best_epoch,
                    "best_val_nll": best_val_nll,
                    "curve": curve,
                }
                save_checkpoint(model, scorer, args.output_dir, metadata)

        summary = {
            "best_epoch": best_epoch,
            "best_val_nll": best_val_nll,
            "total_steps": global_step,
            "elapsed_s": round(time.perf_counter() - started, 1),
            "curve": curve,
            "checkpoint": str(args.output_dir),
        }
        (args.output_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print("[babilong-ft] complete " + json.dumps(summary, sort_keys=True))
        return summary
    finally:
        scorer.close()


def _parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune an ATMA checkpoint on held-in BABILong contexts <=2K."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source_repo_id", default=None)
    parser.add_argument("--source_revision", default=None)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset_revision", default="main")
    parser.add_argument("--tasks", nargs="+", choices=TASKS[:10], default=TASKS[:10])
    parser.add_argument("--train_lengths", nargs="+", choices=TRAIN_LENGTHS,
                        default=TRAIN_LENGTHS)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_end", type=int, default=80)
    parser.add_argument("--val_start", type=int, default=80)
    parser.add_argument("--val_end", type=int, default=90)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default=None)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    run_finetune(args)


if __name__ == "__main__":
    main()
