from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FovealConfig:
    """Resolved modeling and training settings for the CPT pilot.

    Query and KV block sizes are intentionally equal in the first implementation.
    This lets one learned route map directly to one FlexAttention or sparse
    Polar Triton block row.
    """

    checkpoint: str = (
        "ChavyvAkvar/atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0"
    )
    train_glob: str = "finewebedu10B/finewebedu_train_*.bin"
    output_dir: str = "foveal_cpt/output/polar"
    hf_cache: str | None = None
    adaptation_mode: str = "lm_output_kl"

    sequence_length: int = 32768
    batch_tokens: int = 524288
    microbatch_sequences: int = 1
    train_tokens: int = 1_000_000_000

    index_dim: int = 16
    page_size: int = 64
    query_block_size: int = 64
    local_window: int = 512
    remote_capacity: int = 64
    top_p: float = 0.95
    min_remote_pages: int = 0
    max_remote_pages: int = 32

    # Conservative route handoff for the 1B-token run.
    handoff_start_tokens: int = 50_000_000
    handoff_end_tokens: int = 200_000_000
    initial_top_p: float = 0.98
    initial_min_remote_pages: int = 8
    initial_max_remote_pages: int = 64

    teacher_query_blocks: int = 4
    teacher_interval: int = 1
    index_loss_weight: float = 0.1
    teacher_mean_weight: float = 0.5

    activation_checkpointing: bool = True
    flex_compile: bool = True
    flex_kernel_options: dict | None = None
    xent_impl: str = "auto"
    xent_token_chunk: int = 512
    xent_vocab_chunk: int = 2048

    base_lr_scale: float = 0.1
    index_lr: float = 1e-3
    weight_decay: float = 0.01
    cooldown_fraction: float = 0.7
    grad_clip: float = 1.0

    log_every: int = 10
    save_every: int = 250
    seed: int = 1234

    # Frozen-backbone index calibration defaults.
    calibration_sequence_length: int = 4096
    calibration_batch_tokens: int = 65536
    calibration_tokens: int = 20_000_000
    calibration_lr: float = 1e-3

    def validate(self) -> None:
        modes = {"local", "lm_output", "kl", "lm_output_kl"}
        if self.adaptation_mode not in modes:
            raise ValueError(
                f"adaptation_mode must be one of {sorted(modes)}, got {self.adaptation_mode!r}"
            )
        positive = {
            "sequence_length": self.sequence_length,
            "batch_tokens": self.batch_tokens,
            "microbatch_sequences": self.microbatch_sequences,
            "train_tokens": self.train_tokens,
            "index_dim": self.index_dim,
            "page_size": self.page_size,
            "query_block_size": self.query_block_size,
            "local_window": self.local_window,
            "remote_capacity": self.remote_capacity,
            "teacher_interval": self.teacher_interval,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.page_size != self.query_block_size:
            raise ValueError("the pilot currently requires page_size == query_block_size")
        if self.sequence_length % self.page_size:
            raise ValueError("sequence_length must be divisible by page_size")
        if self.local_window % self.page_size:
            raise ValueError("local_window must be divisible by page_size")
        if self.batch_tokens % self.sequence_length:
            raise ValueError("batch_tokens must be divisible by sequence_length")
        sequences = self.batch_tokens // self.sequence_length
        if sequences % self.microbatch_sequences:
            raise ValueError("global sequences must be divisible by microbatch_sequences")
        if not 0.0 < self.top_p <= 1.0 or not 0.0 < self.initial_top_p <= 1.0:
            raise ValueError("top_p values must lie in (0, 1]")
        if not 0.0 <= self.teacher_mean_weight <= 1.0:
            raise ValueError("teacher_mean_weight must lie in [0, 1]")
        for prefix in ("", "initial_"):
            kmin = getattr(self, f"{prefix}min_remote_pages")
            kmax = getattr(self, f"{prefix}max_remote_pages")
            if not 0 <= kmin <= kmax <= self.remote_capacity:
                raise ValueError(
                    f"require 0 <= {prefix}min_remote_pages <= "
                    f"{prefix}max_remote_pages <= remote_capacity"
                )
        if not 0 <= self.handoff_start_tokens <= self.handoff_end_tokens:
            raise ValueError("handoff token boundaries are invalid")
        if not 0.0 <= self.cooldown_fraction <= 1.0:
            raise ValueError("cooldown_fraction must lie in [0, 1]")

    @property
    def uses_index(self) -> bool:
        return self.adaptation_mode != "local"

    @property
    def uses_lm_output(self) -> bool:
        return self.adaptation_mode in {"lm_output", "lm_output_kl"}

    @property
    def uses_kl(self) -> bool:
        return self.adaptation_mode in {"kl", "lm_output_kl"}

    @property
    def requires_calibration(self) -> bool:
        return self.uses_kl

    @property
    def global_sequences(self) -> int:
        return self.batch_tokens // self.sequence_length

    @property
    def accumulation_steps(self) -> int:
        return self.global_sequences // self.microbatch_sequences

    @property
    def train_steps(self) -> int:
        return (self.train_tokens + self.batch_tokens - 1) // self.batch_tokens

    def route_at(self, tokens_seen: int) -> tuple[float, int, int]:
        if tokens_seen <= self.handoff_start_tokens:
            return (
                self.initial_top_p,
                self.initial_min_remote_pages,
                self.initial_max_remote_pages,
            )
        if tokens_seen >= self.handoff_end_tokens:
            return self.top_p, self.min_remote_pages, self.max_remote_pages
        span = max(1, self.handoff_end_tokens - self.handoff_start_tokens)
        alpha = (tokens_seen - self.handoff_start_tokens) / span
        p = self.initial_top_p + alpha * (self.top_p - self.initial_top_p)
        kmin = round(
            self.initial_min_remote_pages
            + alpha * (self.min_remote_pages - self.initial_min_remote_pages)
        )
        kmax = round(
            self.initial_max_remote_pages
            + alpha * (self.max_remote_pages - self.initial_max_remote_pages)
        )
        return float(p), int(kmin), int(kmax)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FovealConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown Foveal config fields: {unknown}")
        config = cls(**raw)
        config.validate()
        return config
