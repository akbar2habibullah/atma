from __future__ import annotations

import numpy as np
from tinygrad import Tensor, dtypes

from edge.config import EdgeConfig, EdgeSamplingParams, resolve_device
from edge.loader import load_edge_model


def _get_tokenizer(name: str):
    try:
        import tiktoken

        return tiktoken.get_encoding(name or "gpt2")
    except Exception:
        pass
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name or "gpt2")
    except Exception:
        return None


class EdgeLLM:
    """Minimal llama.cpp-style session runner for Atma checkpoints."""

    def __init__(self, config: EdgeConfig | str | None = None, **kwargs):
        if isinstance(config, str):
            config = EdgeConfig(model=config, **kwargs)
        elif config is None:
            config = EdgeConfig(**kwargs)
        elif kwargs:
            raise ValueError("pass either EdgeConfig or keyword options, not both")

        self.config = config
        self.device = resolve_device(config.device)
        self.model, self.info = load_edge_model(config)
        self.tokenizer = _get_tokenizer(self.info.get("tokenizer", "gpt2"))

    @property
    def loaded(self) -> bool:
        return bool(self.info.get("loaded"))

    def encode(self, prompt: str) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("no tokenizer available; pass token ids instead of text")
        return list(self.tokenizer.encode(prompt))

    def decode(self, token_ids: list[int]) -> str:
        if self.tokenizer is None:
            return ""
        return self.tokenizer.decode(token_ids)

    def generate_ids(self, input_ids: list[int], sampling: EdgeSamplingParams | None = None) -> list[int]:
        sampling = sampling or EdgeSamplingParams(eos_token_id=self.config.eos_token_id)
        eos = self.config.eos_token_id if sampling.eos_token_id is None else sampling.eos_token_id
        state = self.model.new_state()
        ids = list(input_ids)
        if not ids:
            ids = [0]

        prompt = Tensor([ids], device=self.device, dtype=dtypes.int32)
        logits = self.model(prompt, state)[:, -1, :]
        new_ids: list[int] = []

        for _ in range(sampling.max_tokens):
            scores_np = logits.numpy()[0].astype(np.float64)
            if sampling.temperature <= 0:
                next_id = int(np.argmax(scores_np))
            else:
                scores = scores_np / sampling.temperature
                if sampling.top_k is not None and sampling.top_k > 0:
                    kth = np.partition(scores, -min(sampling.top_k, scores.shape[-1]))[-min(sampling.top_k, scores.shape[-1])]
                    scores = np.where(scores < kth, -np.inf, scores)
                scores = scores - np.max(scores)
                probs = np.exp(scores)
                probs = probs / probs.sum()
                next_id = int(np.random.choice(len(probs), p=probs))

            ids.append(next_id)
            new_ids.append(next_id)
            if eos is not None and not sampling.ignore_eos and next_id == eos:
                break
            logits = self.model(Tensor([[next_id]], device=self.device, dtype=dtypes.int32), state)[:, -1, :]

        return ids

    def generate(
        self,
        prompts: str | list[str] | list[int] | list[list[int]],
        sampling: EdgeSamplingParams | None = None,
    ) -> list[dict]:
        if isinstance(prompts, str):
            prompt_items: list[str | list[int]] = [prompts]
        elif prompts and isinstance(prompts[0], int):
            prompt_items = [prompts]  # type: ignore[list-item]
        else:
            prompt_items = prompts  # type: ignore[assignment]

        outputs = []
        for prompt in prompt_items:
            input_ids = self.encode(prompt) if isinstance(prompt, str) else list(prompt)
            token_ids = self.generate_ids(input_ids, sampling)
            outputs.append({"text": self.decode(token_ids), "token_ids": token_ids})
        return outputs
