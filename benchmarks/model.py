"""EvalModel: adapter over the production inference engine (`inference.LLM`).

The paged engine implements the polar + Canon + Titans memory path. Non-polar
ablation checkpoints are rejected because the engine-side model is the polar
serving model.
"""
from __future__ import annotations

import json
import os

_BANNER = "=" * 78
_WEIGHT_NAMES = ("weights.pt", "model.pt", "pytorch_model.bin")


def resolve_checkpoint(model_path: str) -> tuple[str, str]:
    """Return `(weights_path, checkpoint_dir)` for a checkpoint file or directory."""
    if os.path.isfile(model_path):
        return model_path, os.path.dirname(os.path.abspath(model_path))
    if os.path.isdir(model_path):
        for name in _WEIGHT_NAMES:
            p = os.path.join(model_path, name)
            if os.path.isfile(p):
                return p, os.path.abspath(model_path)
    return model_path, os.path.dirname(os.path.abspath(model_path))


def read_checkpoint_config(model_path: str) -> dict:
    """Load the AtmaConfig JSON saved alongside a checkpoint."""
    _, ckpt_dir = resolve_checkpoint(model_path)
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            return json.load(open(cfg_path, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def unsupported_features(cfg: dict) -> list[str]:
    """Features present in a checkpoint that this benchmark adapter should not run."""
    if not cfg:
        return ["missing config.json (cannot verify architecture)"]
    attn_type = cfg.get("attn_type", "polar")
    if attn_type != "polar":
        return [f"attn_type={attn_type!r} (benchmark inference currently serves polar checkpoints)"]
    return []


def _dtype_from_config(value):
    import torch

    if value in ("float32", "torch.float32", "fp32"):
        return torch.float32
    if value in ("float16", "torch.float16", "fp16"):
        return torch.float16
    return torch.bfloat16


def atma_config_from_dict(cfg: dict):
    from dataclasses import fields

    from model.config import AtmaConfig

    names = {f.name for f in fields(AtmaConfig)}
    kwargs = {k: v for k, v in cfg.items() if k in names and k != "dtype"}
    kwargs["dtype"] = _dtype_from_config(cfg.get("dtype"))
    return AtmaConfig(**kwargs)


class EvalModel:
    """Autoregressive generation adapter over `inference.LLM`.

    Usage:
        m = EvalModel("checkpoints/<run_id>", max_tokens=16)
        texts = m.generate(["...prompt..."])

    Pass `strict=True` to hard-fail if the checkpoint is not supported by the
    polar inference path.
    """

    def __init__(
        self,
        model_path: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 16,
        strict: bool = False,
        quiet: bool = False,
        **llm_kwargs,
    ):
        self.model_path = model_path
        self.weights_path, self.ckpt_dir = resolve_checkpoint(model_path)
        self.cfg = read_checkpoint_config(model_path)
        self.hf_config = atma_config_from_dict(self.cfg) if self.cfg else None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None
        self._llm_kwargs = llm_kwargs
        if self.hf_config is not None:
            self._llm_kwargs.setdefault("hf_config", self.hf_config)
        self.wip = unsupported_features(self.cfg)
        if not quiet:
            self._announce(strict)
        if strict and self.wip:
            raise NotImplementedError(
                "Benchmark inference cannot run this checkpoint: " + "; ".join(self.wip)
            )

    def _announce(self, strict):
        print(_BANNER)
        print("benchmarks.EvalModel")
        if self.wip:
            print("This checkpoint is not supported by the benchmark inference path:")
            for feature in self.wip:
                print(f"  - {feature}")
            print(
                "=> "
                + (
                    "Aborting (strict)."
                    if strict
                    else "Proceeding only because --strict was not set. Treat scores as invalid."
                )
            )
        else:
            print(f"Using polar inference checkpoint: {self.weights_path}")
        print(_BANNER)

    def load(self):
        """Construct the underlying inference.LLM lazily."""
        if self._llm is not None:
            return self
        from inference import LLM

        if LLM is None:
            raise RuntimeError("inference.LLM is unavailable (transformers not importable).")
        self._llm = LLM(self.weights_path, **self._llm_kwargs)
        return self

    def generate(self, prompts, max_tokens=None, temperature=None, use_tqdm=False):
        """Generate continuations for a list of string or token-id prompts."""
        self.load()
        from inference import SamplingParams

        sp = SamplingParams(
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        outs = self._llm.generate(list(prompts), sp, use_tqdm=use_tqdm)
        return [o["text"] for o in outs]
