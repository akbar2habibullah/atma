from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Titans' opaque forward/backward wrappers must be selected before importing
# model.blocks. This is the compile-clean path used by the source pretraining.
os.environ.setdefault("FLA_CUSTOM_OP", "1")

import torch

from .attention import HAS_FLEX_ATTENTION, HAS_POLAR_TRITON
from .checkpoint import load_foveal_weights, load_pretrained
from .config import FovealConfig
from .data import TokenShardLoader
from .model import FovealCPTModel
from .prepare_data import ensure_training_data
from .runtime import (
    cpt_optimizers,
    format_stats,
    load_run,
    save_run,
    seed_everything,
    set_lr,
    validate_resume_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 32K Foveal continual pretraining")
    parser.add_argument("--config", required=True)
    parser.add_argument("--index-checkpoint", help="completed calibration .pt file")
    parser.add_argument("--resume", help="restartable CPT checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--output-dir", help="override config output_dir (used for isolated smoke runs)")
    parser.add_argument("--allow-uncalibrated-index", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FovealConfig.load(args.config)
    ensure_training_data(config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA CPT requested but CUDA is unavailable")
    if (
        config.requires_calibration
        and not args.index_checkpoint
        and not args.resume
        and not args.allow_uncalibrated_index
    ):
        raise ValueError(
            f"{config.adaptation_mode} requires --index-checkpoint or --resume; "
            "use --allow-uncalibrated-index only for an intentional control"
        )
    seed_everything(config.seed)

    base, atma_config, checkpoint_dir = load_pretrained(config, device)
    has_cuda_sparse = HAS_FLEX_ATTENTION or (
        atma_config.attn_type == "polar" and HAS_POLAR_TRITON
    )
    if config.sequence_length > 4096 and (device.type != "cuda" or not has_cuda_sparse):
        raise RuntimeError(
            f"32K {atma_config.attn_type} CPT requires its CUDA sparse-attention kernel"
        )
    model = FovealCPTModel(base, config)
    if args.index_checkpoint and not args.resume:
        payload = load_foveal_weights(model.base, args.index_checkpoint)
        if payload.get("stage") != "calibration":
            raise ValueError("--index-checkpoint must be produced by foveal_cpt.calibrate")
    model.configure_adaptation()
    model.set_mode("sparse")
    model.train()
    optimizers = cpt_optimizers(model, config)
    compiled_forward = torch.compile(model) if config.compile_model else model
    loader = TokenShardLoader(config.train_glob, config.batch_tokens, config.sequence_length, device)

    start_step = 0
    tokens_seen = 0
    if args.resume:
        payload = load_run(args.resume, model=model, optimizers=optimizers)
        validate_resume_config(payload, config)
        if payload.get("stage") != "cpt":
            raise ValueError("resume checkpoint is not a CPT run")
        loader.load_state_dict(payload["loader"])
        start_step = int(payload["step"])
        tokens_seen = int(payload["tokens_seen"])

    total_steps = config.train_steps
    if args.smoke_steps is not None:
        total_steps = min(total_steps, start_step + args.smoke_steps)
    print(
        f"[cpt] mode={config.adaptation_mode} source={checkpoint_dir} "
        f"steps={total_steps} seq={config.sequence_length} "
        f"global_sequences={config.global_sequences} accumulation={config.accumulation_steps}"
    )

    for step in range(start_step, total_steps):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.set_step(step)
        top_p, kmin, kmax = config.route_at(tokens_seen)
        model.set_route(top_p, kmin, kmax)
        inputs, targets = loader.next()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        lm_value = index_value = 0.0
        micro = config.microbatch_sequences
        for start in range(0, config.global_sequences, micro):
            x = inputs[start : start + micro]
            y = targets[start : start + micro]
            lm_loss, _, index_loss = compiled_forward(x, y)
            loss = lm_loss / config.batch_tokens
            loss = loss + config.index_loss_weight * index_loss / config.accumulation_steps
            loss.backward()
            lm_value += float(lm_loss.detach()) / config.batch_tokens
            index_value += float(index_loss.detach()) / config.accumulation_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        set_lr(optimizers, step, config.train_steps, config.cooldown_fraction)
        for optimizer in optimizers:
            optimizer.step()
        tokens_seen += config.batch_tokens

        if step % config.log_every == 0 or step + 1 == total_steps:
            elapsed = time.perf_counter() - started
            stats = model.route_stats()
            peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
            print(
                f"[cpt] step={step + 1}/{total_steps} tokens={tokens_seen} lm={lm_value:.5f} "
                f"index={index_value:.5f} p={top_p:.4f} k={kmin}:{kmax} "
                f"tok/s={config.batch_tokens / elapsed:.1f} peak_gib={peak:.2f} {format_stats(stats)}"
            )
        if (step + 1) % config.save_every == 0 or step + 1 == total_steps:
            path = save_run(
                Path(args.output_dir or config.output_dir) / "cpt",
                model=model,
                config=config,
                step=step + 1,
                tokens_seen=tokens_seen,
                loader_state=loader.state_dict(),
                optimizers=optimizers,
                stage="cpt",
            )
            print(f"[cpt] saved {path}")


if __name__ == "__main__":
    main()
