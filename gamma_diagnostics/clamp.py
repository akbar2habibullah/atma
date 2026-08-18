"""Opt-in, reversible caps for Titans retention logits.

The intervention is implemented as a forward hook on ``mem.w_gamma``. It does
not edit checkpoint tensors, and removing the returned handle restores the
unmodified model. The cap applies to the final logit (including
``mem.gamma_bias``), even though the hook itself sees the learned linear output.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


FORMAT_VERSION = "atma-gamma-clamp-v1"
_BLOCK_RE = re.compile(r"(?:^|\.)blocks\.(\d+)(?:\.|$)")


def half_life_to_gamma(tokens: float) -> float:
    """Return the constant retention whose half-life is ``tokens`` steps."""
    tokens = float(tokens)
    if not math.isfinite(tokens) or tokens <= 0:
        raise ValueError("max_half_life_tokens must be finite and positive")
    return math.exp(math.log(0.5) / tokens)


def half_life_to_logit(tokens: float) -> float:
    """Stable logit of :func:`half_life_to_gamma`, including long horizons."""
    log_gamma = math.log(0.5) / float(tokens)
    if not math.isfinite(log_gamma) or log_gamma >= 0:
        raise ValueError("max_half_life_tokens must be finite and positive")
    return log_gamma - math.log(-math.expm1(log_gamma))


def _log_sigmoid(logit: float) -> float:
    if logit >= 0:
        return -math.log1p(math.exp(-logit))
    return logit - math.log1p(math.exp(logit))


def _target_logit(target: Mapping[str, Any]) -> tuple[float, str, float]:
    choices = [
        key for key in ("max_half_life_tokens", "max_gamma", "max_logit")
        if key in target
    ]
    if len(choices) != 1:
        raise ValueError(
            "each target must define exactly one of max_half_life_tokens, max_gamma, max_logit"
        )
    key = choices[0]
    value = float(target[key])
    if key == "max_half_life_tokens":
        return half_life_to_logit(value), key, value
    if key == "max_gamma":
        if not 0.0 < value < 1.0:
            raise ValueError("max_gamma must be strictly between 0 and 1")
        return math.log(value) - math.log1p(-value), key, value
    if not math.isfinite(value):
        raise ValueError("max_logit must be finite")
    return value, key, value


def load_clamp_spec(spec: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load and minimally validate a clamp specification."""
    if isinstance(spec, Mapping):
        payload = dict(spec)
    else:
        path = Path(spec).expanduser()
        payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("format", FORMAT_VERSION)
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported gamma clamp format {version!r}")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("gamma clamp spec must contain a non-empty targets list")
    return {**payload, "format": FORMAT_VERSION}


def make_clamp_spec(layer: int, heads: list[int], half_life: float, **metadata) -> dict:
    """Build the canonical JSON-serializable v1 spec."""
    return {
        "format": FORMAT_VERSION,
        **metadata,
        "targets": [{
            "layer": int(layer),
            "heads": [int(head) for head in heads],
            "max_half_life_tokens": float(half_life),
        }],
    }


@dataclass
class GammaClampHandle:
    """Owns installed hooks; ``remove`` restores the original forward path."""

    handles: list[Any]
    resolved_targets: list[dict[str, Any]]

    def remove(self) -> None:
        while self.handles:
            self.handles.pop().remove()

    close = remove

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.remove()


def _memory_modules(model):
    for name, module in model.named_modules():
        if not (hasattr(module, "w_gamma") and hasattr(module, "gamma_bias")):
            continue
        match = _BLOCK_RE.search(name)
        if match is not None:
            yield int(match.group(1)), name, module


def apply_gamma_clamp(
    model,
    spec: str | Path | Mapping[str, Any],
    *,
    require_all: bool = True,
) -> GammaClampHandle:
    """Install a selective gamma cap and return a reversible handle.

    Targets use transformer block indices, not "memory-layer ordinal" indices.
    Heads are query/Titans heads. Omitting ``heads`` selects every gamma output
    in the target layer.
    """
    import torch

    payload = load_clamp_spec(spec)
    memories = {layer: (name, module) for layer, name, module in _memory_modules(model)}
    caps_by_layer: dict[int, dict[int, float]] = {}
    requested: list[tuple[int, int]] = []

    for target in payload["targets"]:
        if not isinstance(target, Mapping) or "layer" not in target:
            raise ValueError("each gamma clamp target must contain a layer")
        layer = int(target["layer"])
        if layer not in memories:
            if require_all:
                raise ValueError(f"model has no Titans memory module at block {layer}")
            continue
        _, memory = memories[layer]
        width = int(memory.w_gamma.out_features)
        heads_value = target.get("heads")
        heads = list(range(width)) if heads_value is None else [int(head) for head in heads_value]
        if not heads:
            raise ValueError(f"target for block {layer} selects no heads")
        final_logit_cap, _, _ = _target_logit(target)
        for head in heads:
            if not 0 <= head < width:
                raise ValueError(f"head {head} is outside [0, {width}) for block {layer}")
            requested.append((layer, head))
            previous = caps_by_layer.setdefault(layer, {}).get(head, math.inf)
            caps_by_layer[layer][head] = min(previous, final_logit_cap)

    handles = []
    resolved = []
    for layer, head_caps in caps_by_layer.items():
        module_name, memory = memories[layer]
        width = int(memory.w_gamma.out_features)
        learned_caps = torch.full((width,), math.inf, dtype=torch.float64)
        gamma_bias = float(memory.gamma_bias)
        for head, final_cap in head_caps.items():
            learned_caps[head] = final_cap - gamma_bias
            log_gamma = _log_sigmoid(final_cap)
            resolved.append({
                "layer": layer,
                "head": head,
                "module": module_name,
                "max_final_logit": final_cap,
                "max_gamma": math.exp(log_gamma),
                "max_half_life_tokens": math.log(0.5) / log_gamma,
            })

        def cap_output(_linear, _inputs, output, caps=learned_caps):
            # Titans immediately promotes this result to fp32. Promote here as
            # well so long-horizon caps are not rounded away by bf16/fp16.
            output_fp32 = output.float()
            return torch.minimum(output_fp32, caps.to(device=output.device, dtype=torch.float32))

        handles.append(memory.w_gamma.register_forward_hook(cap_output))

    if require_all and len(set(requested)) != len(resolved):
        raise RuntimeError("not every requested gamma clamp target was installed")
    return GammaClampHandle(handles, sorted(resolved, key=lambda row: (row["layer"], row["head"])))
