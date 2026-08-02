"""Checkpoint-exact direct loglikelihood scoring for every promoted architecture."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Match scaled_ablation.eval_hf_checkpoints: model.blocks reads this during import.
os.environ.setdefault("FLA_CUSTOM_OP", "1")

from benchmarks.model import (
    atma_config_from_dict,
    read_checkpoint_config,
    resolve_checkpoint,
)


LOGIT_SOFTCAP = 15.0


@dataclass(frozen=True)
class TokenRequest:
    context_ids: tuple[int, ...]
    continuation_ids: tuple[int, ...]


def encode_pair(tokenizer, context: str, continuation: str) -> TokenRequest:
    """Tokenize a conditional-likelihood pair without losing boundary whitespace."""
    context = str(context)
    continuation = str(continuation)
    trailing = len(context) - len(context.rstrip(" "))
    if trailing:
        continuation = context[-trailing:] + continuation
        context = context[:-trailing]
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    joined_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
    if joined_ids[: len(context_ids)] != context_ids:
        raise ValueError(
            "context/continuation token boundary is unstable; prefix the continuation with "
            "whitespace or supply token IDs directly"
        )
    continuation_ids = joined_ids[len(context_ids):]
    if not continuation_ids:
        raise ValueError("continuation must contain at least one token")
    return TokenRequest(tuple(context_ids), tuple(continuation_ids))


class DirectScorer:
    """Load a training checkpoint and score conditional token sequences.

    Inputs are right-padded only after their final scored position. Causal forward values before
    that padding are therefore unchanged, allowing choices from one question to share a batch
    without requiring an attention-mask path that the training models do not expose.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str | None = None,
        max_length: int | None = 2048,
        batch_size: int = 8,
    ):
        import torch
        from transformers import AutoTokenizer

        self.model_path = model_path
        self.weights_path, self.checkpoint_dir = resolve_checkpoint(model_path)
        self.cfg = read_checkpoint_config(model_path)
        if not self.cfg:
            raise FileNotFoundError(f"missing or invalid config.json beside {self.weights_path}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = self._load_tokenizer(AutoTokenizer)
        self.model = self._load_model(torch)

    def _load_tokenizer(self, auto_tokenizer):
        try:
            tokenizer = auto_tokenizer.from_pretrained(self.checkpoint_dir, use_fast=True)
        except Exception:
            tokenizer = auto_tokenizer.from_pretrained("gpt2", use_fast=True)
        # These architectures do not use a learned absolute-position table. Avoid the
        # GPT-2 metadata warning when selecting intentionally long evaluation documents.
        tokenizer.model_max_length = self.max_length or 10**30
        return tokenizer

    def _load_model(self, torch):
        architecture = self.cfg.get("arch_type") or self.cfg.get("attn_type", "polar")
        if "arch_type" in self.cfg:
            from raven_baseline.model import create_model
            model = create_model(self.cfg)
        else:
            from train.model import Model
            cfg = dict(self.cfg)
            cfg["num_random_keys"] = 0
            model = Model(atma_config_from_dict(cfg))

        payload = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        state = payload.get("model", payload)
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"checkpoint layout mismatch for {architecture}: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        model.to(self.device)
        model.eval()
        # Length extrapolation is evaluated at full context, matching
        # scaled_ablation.eval_hf_checkpoints rather than any training-only window.
        for block in model.blocks:
            attention = getattr(block, "attn", None)
            if attention is not None and hasattr(attention, "window"):
                attention.window = None
        return model

    @property
    def architecture(self) -> str:
        return self.cfg.get("arch_type") or self.cfg.get("attn_type", "polar")

    def close(self):
        import gc
        import torch

        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_cache(self):
        """Release completed-example tensors between long-context forwards."""
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _truncate(self, request: TokenRequest) -> TokenRequest:
        context = list(request.context_ids)
        continuation = list(request.continuation_ids)
        eos = self.tokenizer.eos_token_id
        if not context:
            context = [eos]
        if self.max_length is not None:
            if len(continuation) >= self.max_length:
                raise ValueError(
                    f"continuation has {len(continuation)} tokens, exceeding max_length="
                    f"{self.max_length}"
                )
            keep = self.max_length - len(continuation)
            context = context[-max(1, keep):]
        return TokenRequest(tuple(context), tuple(continuation))

    def _forward_hidden(self, input_ids):
        x = self.model.embed(input_ids)
        for block in self.model.blocks:
            out = block(x)
            x = out[0] if isinstance(out, tuple) else out
        return x

    def _score_batch(self, requests: list[TokenRequest]) -> list[dict]:
        import torch
        import torch.nn.functional as F

        prepared = [self._truncate(request) for request in requests]
        full = [list(req.context_ids + req.continuation_ids) for req in prepared]
        valid_input_lengths = [len(ids) - 1 for ids in full]
        width = max(valid_input_lengths)
        batch = torch.full(
            (len(full), width),
            self.tokenizer.eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        for row, ids in enumerate(full):
            batch[row, : len(ids) - 1] = torch.tensor(
                ids[:-1], dtype=torch.int32, device=self.device
            )

        results = []
        with torch.inference_mode():
            hidden = self._forward_hidden(batch)
            for row, req in enumerate(prepared):
                start = len(req.context_ids) - 1
                count = len(req.continuation_ids)
                scored_hidden = hidden[row, start:start + count]
                logits = self.model.proj(self.model.norm(scored_hidden)).float()
                logits = LOGIT_SOFTCAP * logits * (
                    logits.square() + LOGIT_SOFTCAP ** 2
                ).rsqrt()
                targets = torch.tensor(
                    req.continuation_ids, dtype=torch.long, device=self.device
                )
                token_nll = F.cross_entropy(logits, targets, reduction="none")
                nll = token_nll.sum().item()
                correct_tokens = int((logits.argmax(-1) == targets).sum().item())
                results.append(
                    {
                        "loglikelihood": -nll,
                        "mean_loglikelihood": -nll / count,
                        "tokens": count,
                        "correct_tokens": correct_tokens,
                        "token_accuracy": correct_tokens / count,
                        "greedy_exact": correct_tokens == count,
                    }
                )
        return results

    def score_requests(self, requests: list[TokenRequest], batch_size: int | None = None):
        size = batch_size or self.batch_size
        output = []
        for start in range(0, len(requests), size):
            output.extend(self._score_batch(requests[start:start + size]))
        return output

    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int | None = None):
        requests = [encode_pair(self.tokenizer, context, continuation)
                    for context, continuation in pairs]
        return self.score_requests(requests, batch_size=batch_size)

    def score_token_ids(self, context_ids, continuation_ids):
        request = TokenRequest(tuple(context_ids), tuple(continuation_ids))
        return self._score_batch([request])[0]
