from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch

from model.config import AtmaConfig
from train.model import Model

from .attention import FovealAttention
from .config import FovealConfig


def resolve_checkpoint(source: str, cache_dir: str | None = None) -> Path:
    path = Path(source)
    if path.exists():
        path = path if path.is_dir() else path.parent
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - training environment dependency.
            raise RuntimeError(
                "huggingface_hub is required when checkpoint is a repository id"
            ) from exc
        path = Path(
            snapshot_download(
                source,
                cache_dir=cache_dir,
                allow_patterns=["weights.pt", "config.json", "run_config.json", "tokenizer.json"],
            )
        )
    missing = [name for name in ("weights.pt", "config.json") if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint {path} is missing {missing}")
    return path


def _atma_config(path: Path) -> AtmaConfig:
    raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
    fields = {field.name for field in dataclasses.fields(AtmaConfig)}
    data = {key: value for key, value in raw.items() if key in fields}
    dtype = data.get("dtype")
    if isinstance(dtype, str):
        name = dtype.removeprefix("torch.")
        if not hasattr(torch, name):
            raise ValueError(f"unsupported checkpoint dtype {dtype!r}")
        data["dtype"] = getattr(torch, name)
    return AtmaConfig(**data)


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only.
        payload = torch.load(path / "weights.pt", map_location="cpu")
    state = payload.get("model", payload)
    return {key.removeprefix("_orig_mod."): value for key, value in state.items()}


def wrap_foveal(model: Model, config: FovealConfig) -> Model:
    wrapped = 0
    for block in model.blocks:
        if not hasattr(block.attn, "num_heads"):
            continue
        block.attn = FovealAttention(
            block.attn,
            hidden_size=model.embed.embedding_dim,
            index_dim=config.index_dim,
            page_size=config.page_size,
            local_window=config.local_window,
            remote_capacity=config.remote_capacity,
            top_p=config.initial_top_p,
            min_remote_pages=config.initial_min_remote_pages,
            max_remote_pages=config.initial_max_remote_pages,
            teacher_query_blocks=config.teacher_query_blocks,
            teacher_interval=config.teacher_interval,
            teacher_mean_weight=config.teacher_mean_weight,
            adaptation_mode=config.adaptation_mode,
            compile_flex=config.flex_compile,
            flex_kernel_options=config.flex_kernel_options,
        )
        wrapped += 1
    if wrapped != model.num_attn_layers:
        raise RuntimeError(f"wrapped {wrapped} attention layers, expected {model.num_attn_layers}")
    return model


def load_pretrained(config: FovealConfig, device: torch.device | str = "cpu") -> tuple[Model, AtmaConfig, Path]:
    path = resolve_checkpoint(config.checkpoint, config.hf_cache)
    atma_config = _atma_config(path)
    if atma_config.attn_type not in {"nope", "rope", "polar"}:
        raise ValueError(f"Foveal CPT does not support attention core {atma_config.attn_type!r}")
    atma_config.max_position_embeddings = config.sequence_length
    atma_config.num_random_keys = 0
    atma_config.attn_window = None
    model = Model(atma_config, reg_mode="baseline")
    result = model.load_state_dict(_state_dict(path), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"base checkpoint mismatch: {result}")
    wrap_foveal(model, config)
    return model.to(device), atma_config, path


def load_foveal_weights(model: Model, path: str | Path) -> dict:
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Foveal checkpoint mismatch: {result}")
    return payload if isinstance(payload, dict) else {"model": state}
