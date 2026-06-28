from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from edge.config import EdgeConfig, resolve_device, resolve_dtype
from edge.model import EdgeAtma
from model.config import AtmaConfig


_DEFAULT_CKPT_DIRS = ("../checkpoints", "checkpoints")
_WEIGHT_NAMES = ("weights.pt", "model.pt", "pytorch_model.bin")


def _unwrap_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "model_state_dict", "weights", "ema"):
            if key in obj and isinstance(obj[key], dict):
                return _unwrap_state_dict(obj[key])
    return obj


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        out[key] = value
    return out


def _tokenizer_name(directory: Path) -> str | None:
    path = directory / "tokenizer.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text()).get("tokenizer_name")
    except Exception:
        return None


def find_checkpoint(explicit: str | None = None) -> tuple[Path | None, Path | None, str | None, list[str]]:
    searched: list[str] = []
    repo_root = Path(__file__).resolve().parent.parent

    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            cfg = path.parent / "config.json"
            return path, cfg if cfg.is_file() else None, _tokenizer_name(path.parent), [str(path)]
        dirs = [path]
    else:
        dirs = []

    dirs.extend((repo_root / item).resolve() for item in _DEFAULT_CKPT_DIRS)
    for directory in dirs:
        for name in _WEIGHT_NAMES:
            weight = directory / name
            searched.append(str(weight))
            if weight.is_file():
                cfg = directory / "config.json"
                return weight, cfg if cfg.is_file() else None, _tokenizer_name(directory), searched
    return None, None, None, searched


def config_from_json(data: dict[str, Any]) -> AtmaConfig:
    config = AtmaConfig()
    for key in AtmaConfig.__dataclass_fields__:
        if key in data:
            setattr(config, key, data[key])
    return config


def infer_config(state_dict: dict[str, torch.Tensor]) -> AtmaConfig:
    vocab_size, hidden = state_dict["embed.weight"].shape
    layer_ids = sorted(
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("blocks.") and len(key.split(".")) > 2 and key.split(".")[1].isdigit()
    )
    num_layers = max(layer_ids) + 1 if layer_ids else AtmaConfig().num_hidden_layers
    attn_layers = sorted({i for i in layer_ids if f"blocks.{i}.attn.v_null" in state_dict})
    head_dim = state_dict[f"blocks.{attn_layers[0]}.attn.v_null"].shape[1] if attn_layers else AtmaConfig().head_dim
    attn_heads = hidden // head_dim
    config = AtmaConfig(vocab_size=vocab_size, hidden_size=hidden, num_hidden_layers=num_layers, head_dim=head_dim)
    if attn_layers:
        first = attn_layers[0]
        config.attn_kernel_size = state_dict[f"blocks.{first}.attn.canon_q.weight"].shape[2]
        config.attn_window = None
        config.mem_enabled = any(f"blocks.{first}.attn.mem." in key for key in state_dict)
    conv_layer = next((i for i in layer_ids if f"blocks.{i}.attn.conv.weight" in state_dict), None)
    if conv_layer is not None:
        config.conv_kernel_size = state_dict[f"blocks.{conv_layer}.attn.conv.weight"].shape[2]
    if attn_heads < 4:
        raise ValueError("AtmaConfig.num_key_value_heads derives a 1:4 GQA ratio and requires at least 4 heads")
    return config


def load_edge_model(config: EdgeConfig | None = None) -> tuple[EdgeAtma, dict[str, Any]]:
    config = config or EdgeConfig()
    if config.backend != "tinygrad":
        raise NotImplementedError("edge currently implements the tinygrad backend")

    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    weight_path, config_path, tokenizer_name, searched = find_checkpoint(config.model)

    if weight_path is None:
        model_config = AtmaConfig()
        loaded = False
        state_dict = None
    else:
        state_dict = _clean_state_dict(_unwrap_state_dict(torch.load(weight_path, map_location="cpu", weights_only=False)))
        if config.use_checkpoint_config and config_path is not None:
            model_config = config_from_json(json.loads(config_path.read_text()))
        else:
            model_config = infer_config(state_dict)
        loaded = True

    if config.max_context is not None:
        model_config.max_position_embeddings = config.max_context
    model = EdgeAtma(model_config).to(device=device, dtype=dtype)
    load_info: dict[str, Any] = {
        "loaded": loaded,
        "path": str(weight_path) if weight_path is not None else None,
        "config_path": str(config_path) if config_path is not None else None,
        "searched": searched,
        "tokenizer": config.tokenizer or tokenizer_name or "gpt2",
        "device": str(device),
        "dtype": str(dtype).replace("dtypes.", ""),
    }
    if state_dict is not None:
        incompatible = model.load_state_dict(state_dict, strict=False)
        load_info["missing"] = list(incompatible.missing_keys)
        load_info["unexpected"] = list(incompatible.unexpected_keys)

    return model, load_info
