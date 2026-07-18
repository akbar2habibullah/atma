"""Cross-evaluate Atma Hugging Face checkpoints in one controlled environment.

Designed for the L4/PyTorch-version investigation. The default checkpoint set is:

  * ChavyvAkvar/atma-nope__reg-baseline__distr-0__mem-1__win-0_L4
  * ChavyvAkvar/atma-10b-nope__reg-baseline__distr-0__mem-1__win-0
  * ChavyvAkvar/atma-10b-polar__reg-baseline__distr-0__mem-1__win-0

All checkpoints see the same cached coherent documents and the same FineWeb-Edu validation
prefixes. Results include successful-example and CUDA-OOM counts so hardware-dependent skips
cannot silently change the denominator.

Quick discriminating run (junk loss only):

    FLA_CUSTOM_OP=1 python -m scaled_ablation.eval_hf_checkpoints --metrics junk

Full reproduction through 131K:

    FLA_CUSTOM_OP=1 python -m scaled_ablation.eval_hf_checkpoints \
        --metrics junk clean needle \
        --output scaled_ablation/cross_eval_l4_torch212.json

Run from the repository root. CUDA is required because train.data.data_generator places its
batches directly on CUDA.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

# model.blocks reads this at import time. Set it before importing eval/train.model below.
os.environ.setdefault("FLA_CUSTOM_OP", "1")

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from eval import _blocks_forward, _chunked_loss, load_from_checkpoint, select_long_docs
from scaled_ablation.evaluate import LOGIT_SOFTCAP, _set_full_context
from train.data import data_generator, get_data


DEFAULT_MODELS = [
    "ChavyvAkvar/atma-nope__reg-baseline__distr-0__mem-1__win-0_L4",
    "ChavyvAkvar/atma-10b-nope__reg-baseline__distr-0__mem-1__win-0",
    "ChavyvAkvar/atma-10b-polar__reg-baseline__distr-0__mem-1__win-0",
]
DEFAULT_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]


def _empty_cache():
    torch.cuda.empty_cache()


def _atomic_json_dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _runtime_metadata(sdpa_backend: str) -> dict:
    props = torch.cuda.get_device_properties(0)
    cuda_backends = {}
    for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"):
        fn = getattr(torch.backends.cuda, name, None)
        cuda_backends[name] = fn() if fn is not None else None
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": props.name,
        "gpu_total_memory_bytes": props.total_memory,
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "initial_seed": torch.initial_seed(),
        "requested_sdpa_backend": sdpa_backend,
        "available_sdpa_backends": cuda_backends,
        "fla_custom_op": os.environ.get("FLA_CUSTOM_OP"),
    }


def _sdpa_context(name: str):
    if name == "auto":
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backend = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "math": SDPBackend.MATH,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
    }[name]
    return sdpa_kernel(backends=[backend])


def _download_checkpoint(repo_id: str, cache_dir: str | None) -> Path:
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        cache_dir=cache_dir,
        allow_patterns=["weights.pt", "config.json", "run_config.json", "tokenizer.json"],
    )
    checkpoint = Path(path)
    missing = [name for name in ("weights.pt", "config.json") if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{repo_id} is missing required checkpoint files: {missing}")
    return checkpoint


def _load_json_if_present(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _load_or_create_docs(args, need: int) -> list[torch.Tensor]:
    manifest = Path(args.doc_manifest)
    metadata_path = manifest.with_suffix(manifest.suffix + ".json")
    expected = {
        "dataset": args.clean_dataset,
        "text_key": args.clean_text_key,
        "split": args.clean_split,
        "tokenizer": "gpt2",
        "min_tokens": need,
        "num_docs": args.num_eval_docs,
    }

    if manifest.is_file() and not args.rebuild_doc_manifest:
        metadata = _load_json_if_present(metadata_path)
        if metadata != expected:
            raise RuntimeError(
                f"Document manifest metadata mismatch at {metadata_path}. "
                "Use a different --doc-manifest or pass --rebuild-doc-manifest."
            )
        docs = torch.load(manifest, map_location="cpu", weights_only=True)
        if len(docs) != args.num_eval_docs or any(d.numel() < need + 1 for d in docs):
            raise RuntimeError(f"Invalid or incomplete document manifest: {manifest}")
        print(f"Loaded {len(docs)} shared evaluation documents from {manifest}")
        return docs

    docs = select_long_docs(
        args.clean_dataset,
        args.clean_text_key,
        args.clean_split,
        need,
        args.num_eval_docs,
        tokenizer_name="gpt2",
    )
    if len(docs) != args.num_eval_docs:
        raise RuntimeError(
            f"Only found {len(docs)}/{args.num_eval_docs} documents with at least {need + 1} tokens"
        )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(docs, manifest)
    _atomic_json_dump(metadata_path, expected)
    print(f"Saved shared evaluation documents to {manifest}")
    return docs


def _value_logits(model, inp: torch.Tensor, n_last: int) -> torch.Tensor:
    x = _blocks_forward(model, inp)
    z = model.proj(model.norm(x[:, -n_last:])).float()
    return (LOGIT_SOFTCAP * z * (z.square() + LOGIT_SOFTCAP**2).rsqrt())[0]


@torch.no_grad()
def eval_clean(model, docs, lengths, device, loss_chunk: int) -> dict:
    results = {}
    for length in lengths:
        total, tokens, completed, ooms = 0.0, 0, 0, 0
        for doc in docs:
            inputs = targets = hidden = None
            try:
                buf = doc[: length + 1]
                inputs = buf[:-1].view(1, -1).to(device, torch.int32)
                targets = buf[1:].view(1, -1).to(device, torch.int64)
                hidden = _blocks_forward(model, inputs)
                loss_sum, count = _chunked_loss(model, hidden, targets, chunk=loss_chunk)
                total += loss_sum
                tokens += count
                completed += 1
            except torch.cuda.OutOfMemoryError:
                ooms += 1
            finally:
                del inputs, targets, hidden
                _empty_cache()
        results[str(length)] = {
            "loss_nats": total / tokens if tokens else None,
            "tokens": tokens,
            "completed": completed,
            "oom": ooms,
        }
        print(f"  clean {length:>6,}: {results[str(length)]}", flush=True)
    return results


@torch.no_grad()
def eval_junk(model, val_data: str, lengths, num_seqs: int, loss_chunk: int) -> dict:
    results = {}
    for length in lengths:
        total, tokens, completed, ooms = 0.0, 0, 0, 0
        generator = data_generator(val_data, length, seq_len=length)
        for _ in range(num_seqs):
            inputs = targets = hidden = None
            try:
                inputs, targets = next(generator)
                hidden = _blocks_forward(model, inputs)
                loss_sum, count = _chunked_loss(model, hidden, targets, chunk=loss_chunk)
                total += loss_sum
                tokens += count
                completed += 1
            except torch.cuda.OutOfMemoryError:
                ooms += 1
            finally:
                del inputs, targets, hidden
                _empty_cache()
        results[str(length)] = {
            "loss_nats": total / tokens if tokens else None,
            "tokens": tokens,
            "completed": completed,
            "oom": ooms,
        }
        print(f"  junk  {length:>6,}: {results[str(length)]}", flush=True)
    return results


@torch.no_grad()
def eval_needle(model, docs, distances, num_trials: int, value_len: int, device) -> dict:
    from transformers import AutoTokenizer

    rng = random.Random(1234)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eot = tokenizer.eos_token_id
    accumulators = {
        distance: {"ce_sum": 0.0, "acc_sum": 0.0, "completed": 0, "oom": 0}
        for distance in distances
    }
    baseline = {"ce_sum": 0.0, "completed": 0, "oom": 0}
    shortest = min(distances)

    # Match scaled_ablation.evaluate: one key/value pair is shared across all distances in a trial.
    for trial in range(num_trials):
        doc = docs[trial % len(docs)]
        key = rng.randint(10**6, 10**7 - 1)
        digits = [rng.randint(0, 9) for _ in range(value_len)]
        cue = tokenizer.encode(f" The access code for record {key} is")
        value = tokenizer.encode("".join(f" {digit}" for digit in digits))
        needle = cue + value

        inputs = targets = logits = None
        try:
            gap = doc[: shortest + len(needle)].tolist()
            sequence = [eot] + gap + cue + value
            inputs = torch.tensor(sequence[:-1], dtype=torch.int32, device=device).view(1, -1)
            targets = torch.tensor(value, dtype=torch.int64, device=device)
            logits = _value_logits(model, inputs, len(value))
            baseline["ce_sum"] += F.cross_entropy(logits, targets, reduction="mean").item()
            baseline["completed"] += 1
        except torch.cuda.OutOfMemoryError:
            baseline["oom"] += 1
        finally:
            del inputs, targets, logits
            _empty_cache()

        for distance in distances:
            stats = accumulators[distance]
            inputs = targets = logits = None
            try:
                gap = doc[:distance].tolist()
                sequence = [eot] + needle + gap + cue + value
                inputs = torch.tensor(sequence[:-1], dtype=torch.int32, device=device).view(1, -1)
                targets = torch.tensor(value, dtype=torch.int64, device=device)
                logits = _value_logits(model, inputs, len(value))
                stats["ce_sum"] += F.cross_entropy(logits, targets, reduction="mean").item()
                stats["acc_sum"] += (logits.argmax(-1) == targets).float().mean().item()
                stats["completed"] += 1
            except torch.cuda.OutOfMemoryError:
                stats["oom"] += 1
            finally:
                del inputs, targets, logits
                _empty_cache()

    result = {}
    for distance in distances:
        stats = accumulators[distance]
        completed = stats["completed"]
        result[str(distance)] = {
            "ce_nats": stats["ce_sum"] / completed if completed else None,
            "accuracy_pct": 100.0 * stats["acc_sum"] / completed if completed else None,
            "completed": completed,
            "oom": stats["oom"],
        }
        print(f"  needle {distance:>6,}: {result[str(distance)]}", flush=True)
    baseline_completed = baseline["completed"]
    return {
        "by_distance": result,
        "absent_baseline": {
            "ce_nats": baseline["ce_sum"] / baseline_completed if baseline_completed else None,
            "completed": baseline_completed,
            "oom": baseline["oom"],
        },
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Hugging Face model repo IDs")
    parser.add_argument("--lengths", nargs="+", type=int, default=DEFAULT_LENGTHS)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=("junk", "clean", "needle"),
        default=("junk", "clean", "needle"),
    )
    parser.add_argument("--num-eval-docs", type=int, default=64)
    parser.add_argument("--num-needle-trials", type=int, default=64)
    parser.add_argument("--needle-value-len", type=int, default=5)
    parser.add_argument("--loss-chunk", type=int, default=8192)
    parser.add_argument("--sdpa-backend", choices=("auto", "flash", "math", "efficient"), default="flash")
    parser.add_argument("--val-data", default="finewebedu10B/finewebedu_val_*.bin")
    parser.add_argument("--clean-dataset", default="codelion/finepdfs-1B")
    parser.add_argument("--clean-text-key", default="text")
    parser.add_argument("--clean-split", default="train")
    parser.add_argument("--doc-manifest", default="scaled_ablation/eval_manifests/finepdfs_131k_64.pt")
    parser.add_argument("--rebuild-doc-manifest", action="store_true")
    parser.add_argument("--hf-cache", default=None)
    parser.add_argument("--output", default="scaled_ablation/cross_eval_l4_torch212.json")
    return parser.parse_args()


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not args.lengths or any(length <= 0 for length in args.lengths):
        raise SystemExit("--lengths must contain positive integers")
    if args.sdpa_backend == "math" and max(args.lengths) > 16384:
        print("WARNING: math SDPA is quadratic and will probably OOM at the requested lengths.")

    device = torch.device("cuda")
    lengths = sorted(set(args.lengths))
    output_path = Path(args.output)
    output = {
        "runtime": _runtime_metadata(args.sdpa_backend),
        "settings": {
            "models": args.models,
            "lengths": lengths,
            "metrics": args.metrics,
            "num_eval_docs": args.num_eval_docs,
            "num_needle_trials": args.num_needle_trials,
            "needle_value_len": args.needle_value_len,
            "loss_chunk": args.loss_chunk,
            "val_data": args.val_data,
            "clean_dataset": args.clean_dataset,
            "doc_manifest": args.doc_manifest,
        },
        "checkpoints": {},
    }
    _atomic_json_dump(output_path, output)
    print(json.dumps(output["runtime"], indent=2), flush=True)

    docs = None
    if "clean" in args.metrics or "needle" in args.metrics:
        # Needle construction adds a small scaffold beyond the requested gap.
        need = max(lengths) + (64 if "needle" in args.metrics else 0)
        docs = _load_or_create_docs(args, need)
    if "junk" in args.metrics and args.val_data == "finewebedu10B/finewebedu_val_*.bin":
        get_data("finewebedu_val_000000.bin")

    for repo_id in args.models:
        print(f"\n{'=' * 100}\nEvaluating {repo_id}\n{'=' * 100}", flush=True)
        checkpoint_dir = _download_checkpoint(repo_id, args.hf_cache)
        model, _config = load_from_checkpoint(
            str(checkpoint_dir), device, compile_model=False, force_probe_path=False
        )
        model = getattr(model, "_orig_mod", model)
        model.eval()
        _set_full_context(model)

        entry = {
            "checkpoint_dir": str(checkpoint_dir),
            "model_config": _load_json_if_present(checkpoint_dir / "config.json"),
            "run_config": _load_json_if_present(checkpoint_dir / "run_config.json"),
            "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "metrics": {},
        }
        output["checkpoints"][repo_id] = entry
        _atomic_json_dump(output_path, output)

        with _sdpa_context(args.sdpa_backend):
            if "junk" in args.metrics:
                entry["metrics"]["junk"] = eval_junk(
                    model, args.val_data, lengths, args.num_eval_docs, args.loss_chunk
                )
                _atomic_json_dump(output_path, output)
            if "clean" in args.metrics:
                entry["metrics"]["clean"] = eval_clean(
                    model, docs, lengths, device, args.loss_chunk
                )
                _atomic_json_dump(output_path, output)
            if "needle" in args.metrics:
                entry["metrics"]["needle"] = eval_needle(
                    model, docs, lengths, args.num_needle_trials, args.needle_value_len, device
                )
                _atomic_json_dump(output_path, output)

        del model
        gc.collect()
        _empty_cache()

    print(f"\nResults written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
