from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from .checkpoint import load_pretrained
from .config import FovealConfig
from .data import TokenShardLoader
from .model import FovealCPTModel
from .runtime import (
    format_stats,
    index_optimizer,
    load_run,
    save_run,
    seed_everything,
    validate_resume_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the Foveal MQA index against full attention")
    parser.add_argument("--config", default="foveal_cpt/pilot.json")
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-steps", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FovealConfig.load(args.config)
    if config.calibration_sequence_length % config.page_size:
        raise ValueError("calibration_sequence_length must be divisible by page_size")
    if config.calibration_batch_tokens % config.calibration_sequence_length:
        raise ValueError("calibration_batch_tokens must be divisible by calibration_sequence_length")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA calibration requested but CUDA is unavailable")
    seed_everything(config.seed)

    base, _, checkpoint_dir = load_pretrained(config, device)
    model = FovealCPTModel(base, config)
    model.set_mode("dense_teacher")
    model.freeze_except_index()
    model.train()
    optimizer = index_optimizer(model, config)
    optimizers = [optimizer]
    loader = TokenShardLoader(
        config.train_glob,
        config.calibration_batch_tokens,
        config.calibration_sequence_length,
        device,
    )

    start_step = 0
    tokens_seen = 0
    if args.resume:
        payload = load_run(args.resume, model=model, optimizers=optimizers)
        validate_resume_config(payload, config)
        if payload.get("stage") != "calibration":
            raise ValueError("resume checkpoint is not a calibration run")
        loader.load_state_dict(payload["loader"])
        start_step = int(payload["step"])
        tokens_seen = int(payload["tokens_seen"])

    total_steps = math.ceil(config.calibration_tokens / config.calibration_batch_tokens)
    if args.smoke_steps is not None:
        total_steps = min(total_steps, start_step + args.smoke_steps)
    sequences = config.calibration_batch_tokens // config.calibration_sequence_length
    micro = config.microbatch_sequences
    if sequences % micro:
        raise ValueError("calibration sequences must divide evenly into microbatches")
    accumulation = sequences // micro
    print(
        f"[calibrate] source={checkpoint_dir} steps={total_steps} seq={config.calibration_sequence_length} "
        f"batch_tokens={config.calibration_batch_tokens} accumulation={accumulation}"
    )

    for step in range(start_step, total_steps):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.set_step(step)
        inputs, _ = loader.next()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for start in range(0, sequences, micro):
            loss = model.calibration_loss(inputs[start : start + micro]) / accumulation
            loss.backward()
            loss_value += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.index_parameters(), config.grad_clip)
        optimizer.step()
        tokens_seen += config.calibration_batch_tokens

        if step % config.log_every == 0 or step + 1 == total_steps:
            elapsed = time.perf_counter() - started
            stats = model.route_stats()
            peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
            print(
                f"[calibrate] step={step + 1}/{total_steps} tokens={tokens_seen} "
                f"index_loss={loss_value:.5f} tok/s={config.calibration_batch_tokens / elapsed:.1f} "
                f"peak_gib={peak:.2f} {format_stats(stats)}"
            )
        if (step + 1) % config.save_every == 0 or step + 1 == total_steps:
            path = save_run(
                Path(config.output_dir) / "calibration",
                model=model,
                config=config,
                step=step + 1,
                tokens_seen=tokens_seen,
                loader_state=loader.state_dict(),
                optimizers=optimizers,
                stage="calibration",
            )
            print(f"[calibrate] saved {path}")


if __name__ == "__main__":
    main()
