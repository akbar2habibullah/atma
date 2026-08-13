from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from train.optimizer import Muon

from .config import FovealConfig
from .model import FovealCPTModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def index_optimizer(model: FovealCPTModel, config: FovealConfig) -> AdamW:
    params = [param for param in model.index_parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("no trainable index parameters")
    return AdamW(params, lr=config.calibration_lr, betas=(0.9, 0.95), weight_decay=0.0)


def cpt_optimizers(model: FovealCPTModel, config: FovealConfig) -> list[torch.optim.Optimizer]:
    index = {id(param) for param in model.index_parameters()}
    embed = model.base.embed.weight
    head = model.base.proj.weight
    index_params = []
    matrix_params = []
    vector_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in index:
            index_params.append(param)
        elif param is embed or param is head:
            continue
        elif param.ndim >= 2:
            matrix_params.append(param)
        else:
            vector_params.append(param)

    adam = AdamW(
        [
            {"params": [embed], "lr": 0.3 * config.base_lr_scale, "name": "embedding"},
            {"params": [head], "lr": (1.0 / 320.0) * config.base_lr_scale, "name": "head"},
            {"params": vector_params, "lr": 0.01 * config.base_lr_scale, "name": "vectors"},
            {"params": index_params, "lr": config.index_lr, "name": "index"},
        ],
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.0,
        fused=embed.is_cuda,
    )
    optimizers: list[torch.optim.Optimizer] = [adam]
    if matrix_params:
        optimizers.append(
            Muon(
                matrix_params,
                lr=0.02 * config.base_lr_scale,
                weight_decay=config.weight_decay,
            )
        )

    assigned = {id(param) for optimizer in optimizers for group in optimizer.param_groups for param in group["params"]}
    expected = {id(param) for param in model.parameters() if param.requires_grad}
    if assigned != expected:
        raise RuntimeError(
            f"optimizer partition mismatch: missing={len(expected - assigned)}, extra={len(assigned - expected)}"
        )
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers


def set_lr(optimizers: list[torch.optim.Optimizer], step: int, total_steps: int, cooldown: float) -> None:
    progress = step / max(total_steps, 1)
    if cooldown == 0.0 or progress < 1.0 - cooldown:
        factor = 1.0
    else:
        factor = max(0.0, (1.0 - progress) / cooldown)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * factor


def _rng_state() -> dict:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_run(
    output_dir: str | Path,
    *,
    model: FovealCPTModel,
    config: FovealConfig,
    step: int,
    tokens_seen: int,
    loader_state: dict,
    optimizers: list[torch.optim.Optimizer],
    stage: str,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config.save(output / "config.json")
    payload = {
        "model": model.base.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "loader": loader_state,
        "rng": _rng_state(),
        "stage": stage,
        "config": config.to_dict(),
    }
    numbered = output / f"{stage}-step-{step:06d}.pt"
    temporary = output / f".{numbered.name}.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, numbered)
    latest = output / "latest.json"
    latest.write_text(json.dumps({"checkpoint": numbered.name, "step": step, "tokens_seen": tokens_seen}, indent=2) + "\n")
    return numbered


def load_run(
    path: str | Path,
    *,
    model: FovealCPTModel,
    optimizers: list[torch.optim.Optimizer] | None = None,
) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model.base.load_state_dict(payload["model"], strict=True)
    if optimizers is not None:
        saved = payload.get("optimizers", [])
        if len(saved) != len(optimizers):
            raise RuntimeError(f"resume has {len(saved)} optimizers, expected {len(optimizers)}")
        for optimizer, state in zip(optimizers, saved):
            optimizer.load_state_dict(state)
    restore_rng(payload.get("rng"))
    return payload


def validate_resume_config(payload: dict, config: FovealConfig) -> None:
    saved = payload.get("config") or {}
    structural = (
        "checkpoint",
        "adaptation_mode",
        "sequence_length",
        "batch_tokens",
        "microbatch_sequences",
        "index_dim",
        "page_size",
        "query_block_size",
        "local_window",
        "remote_capacity",
        "calibration_sequence_length",
        "calibration_batch_tokens",
    )
    mismatches = {
        key: (saved.get(key), getattr(config, key))
        for key in structural
        if saved.get(key) != getattr(config, key)
    }
    if mismatches:
        raise ValueError(f"resume configuration mismatch: {mismatches}")


def format_stats(values: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in sorted(values.items()))
