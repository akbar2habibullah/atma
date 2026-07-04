"""Config schema for Raven bridge and scaled-promotion runs.

These runs are architecture baselines, not additional cells in the Atma attention-core
factorial. They still emit the same ABLATION_* log blocks and axis fields so the existing
parsers and dashboards can ingest them.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


ARCH_TYPES = ["raven_native", "atma_raven", "atma_raven_titans"]
EVAL_LENGTHS_1B = [2048, 4096, 8192, 16384, 32768, 65536]
EVAL_LENGTHS_SCALED = [2048, 4096, 8192, 16384, 32768, 65536, 131072]


@dataclass
class RavenRunConfig:
    arch_type: str

    # Dashboard compatibility axes. `attn_type` intentionally names the whole Raven variant.
    attn_type: str = field(init=False, default="")
    reg_mode: str = "baseline"
    distractor: bool = False
    memory: bool = field(init=False, default=False)
    window: bool = False

    # Model shape.
    vocab_size: int = 50304
    hidden_size: int = 1024
    num_hidden_layers: int = field(init=False, default=0)
    num_heads: int = 4
    atma_head_match: bool = True
    num_slots: int = 256
    topk: int = 32
    feature_map: str = "swish"
    router_score: str = "sigmoid"
    router_type: str = "lin"
    decay_type: str = "Mamba2"
    add_gumbel_noise: bool = True
    bias_rmm: bool = False
    gate_logit_normalizer: int = 8
    conv_kernel_size: int = 3

    # Data / optimization.
    seq_len: int = 2048
    num_chunks: int = 10
    batch_size: int = 8 * 64 * 1024
    val_tokens: int = 4 * 524288
    mbs: int = 4
    cooldown_frac: float = 0.7
    sketch_dim: int = 64
    sigr_alpha_on: float = 0.01
    dist_align_weight_on: float = 0.01
    optimizer: str = "adamw_raven"      # "adamw_raven" | "atma_muon"
    adamw_lr: float = 3e-4
    adamw_lr_min_frac: float = 0.1
    adamw_warmup_frac: float = 0.05
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    adamw_eps: float = 1e-15
    adamw_weight_decay: float = 0.1
    skip_nan_inf: bool = True
    compile_model: bool = True

    # Titans branch, only meaningful for atma_raven_titans.
    mem_chunk: int = 128
    mem_gamma_bias: float = 3.9
    mem_beta_bias: float = 0.0
    mem_kernel: str = "auto"
    fla_custom_op: bool = True

    # Eval.
    eval_lengths: list[int] = field(default_factory=lambda: list(EVAL_LENGTHS_1B))
    needle_distances: list[int] = field(default_factory=lambda: list(EVAL_LENGTHS_1B))
    clean_dataset: str = "codelion/finepdfs-100M"
    val_data: str = "finewebedu10B/finewebedu_val_*.bin"
    num_eval_docs: int = 16
    num_needle_trials: int = 16
    needle_val_len: int = 5

    # Derived.
    run_id: str = field(init=False, default="")
    comparison_group: str = field(init=False, default="")
    mixer_ratio: str = field(init=False, default="")
    num_random_keys: int = field(init=False, default=0)
    dist_align_loss_weight: float = field(init=False, default=0.0)
    sigr_alpha: float = field(init=False, default=0.0)
    attn_window: int | None = field(init=False, default=None)
    mem_enabled: bool = field(init=False, default=False)

    def __post_init__(self):
        assert self.arch_type in ARCH_TYPES, self.arch_type
        self.attn_type = self.arch_type
        self.memory = self.arch_type == "atma_raven_titans"
        self.mem_enabled = self.memory
        self.num_hidden_layers = 16
        if self.arch_type != "raven_native" and self.atma_head_match and self.num_heads == 4:
            self.num_heads = 8
        self.mixer_ratio = "16_raven" if self.arch_type == "raven_native" else "12_lfm2_4_raven"
        self.comparison_group = "raven_bridge_1b"
        self.num_random_keys = 0
        self.dist_align_loss_weight = 0.0
        self.sigr_alpha = self.sigr_alpha_on if self.reg_mode != "baseline" else 0.0
        self.attn_window = None
        self.run_id = f"{self.arch_type}__reg-baseline__distr-0__mem-{int(self.memory)}__win-0"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "RavenRunConfig":
        init_names = {f.name for f in dataclasses.fields(RavenRunConfig) if f.init}
        return RavenRunConfig(**{k: v for k, v in d.items() if k in init_names})


def expand_bridge(**base_overrides) -> list[RavenRunConfig]:
    return [RavenRunConfig(arch_type=a, **base_overrides) for a in ARCH_TYPES]


def expand_scaled(arch_types: list[str], **base_overrides) -> list[RavenRunConfig]:
    overrides = {
        "num_chunks": 99,
        "eval_lengths": list(EVAL_LENGTHS_SCALED),
        "needle_distances": list(EVAL_LENGTHS_SCALED),
        "clean_dataset": "codelion/finepdfs-1B",
        "num_eval_docs": 64,
        "num_needle_trials": 64,
    }
    overrides.update(base_overrides)
    cells = [RavenRunConfig(arch_type=a, **overrides) for a in arch_types]
    for c in cells:
        c.comparison_group = "raven_scaled_promotion"
    return cells

