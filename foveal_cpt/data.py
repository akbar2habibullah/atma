from __future__ import annotations

import glob
from pathlib import Path

import torch

from train.data import _load_data_shard


class TokenShardLoader:
    """Small resumable counterpart to ``train.data.data_generator``."""

    def __init__(self, pattern: str, batch_tokens: int, sequence_length: int, device: torch.device):
        self.files = [Path(path) for path in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"no token shards match {pattern!r}")
        if batch_tokens % sequence_length:
            raise ValueError("batch_tokens must be divisible by sequence_length")
        self.batch_tokens = int(batch_tokens)
        self.sequence_length = int(sequence_length)
        self.device = device
        self.shard_index = 0
        self.position = 0
        self._tokens = _load_data_shard(self.files[0])

    def _advance(self) -> None:
        self.shard_index += 1
        if self.shard_index >= len(self.files):
            raise StopIteration("token shards exhausted")
        self.position = 0
        self._tokens = _load_data_shard(self.files[self.shard_index])

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        while self.position + self.batch_tokens + 1 >= len(self._tokens):
            self._advance()
        buf = self._tokens[self.position : self.position + self.batch_tokens + 1]
        self.position += self.batch_tokens
        inputs = buf[:-1].to(self.device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(self.device, dtype=torch.int64, non_blocking=True)
        return (
            inputs.view(-1, self.sequence_length),
            targets.view(-1, self.sequence_length),
        )

    def state_dict(self) -> dict:
        return {"shard_index": self.shard_index, "position": self.position}

    def load_state_dict(self, state: dict) -> None:
        shard_index = int(state["shard_index"])
        position = int(state["position"])
        if not 0 <= shard_index < len(self.files):
            raise ValueError(f"invalid shard index {shard_index}")
        self.shard_index = shard_index
        self.position = position
        self._tokens = _load_data_shard(self.files[self.shard_index])
        if not 0 <= self.position < len(self._tokens):
            raise ValueError(f"invalid shard position {self.position}")
