"""RunConfig schema + grid expansion for the final scaled ablation.

Grid axes (1 x 1 x 1 x 1 x 3 = 3):
  reg_mode  : baseline
  distractor: off
  memory    : on
  window    : off
  attn_type : rope | nope | polar

Everything else is fixed base config (see RunConfig defaults). The JSON written per run is
the full *resolved* config (axes + base + derived), so training and the dashboard need no
re-derivation. `run_id` is deterministic and unique per cell.
"""
from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass, field

REG_MODES = ["baseline"]
ATTN_TYPES = ["rope", "nope", "polar"]
INCOMPATIBLE_ATTN_TYPES = ["wall"]  # diagnostic only: unstable under this Atma/Muon protocol
ALL_ATTN_TYPES = ATTN_TYPES + INCOMPATIBLE_ATTN_TYPES
EVAL_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]


@dataclass
class RunConfig:
    # --- grid axes (categorical; the dashboard groups/filters on these) ---
    attn_type: str
    reg_mode: str
    distractor: bool
    memory: bool
    window: bool

    # --- base model (fixed across the grid) ---
    vocab_size: int = 50304
    num_hidden_layers: int = 16
    hidden_size: int = 1024
    head_dim: int = 128
    window_size: int = 1024            # the SWA width used when window=on

    # --- data / optimization (fixed) ---
    seq_len: int = 2048
    num_chunks: int = 99               # all kjj0/finewebedu10B-gpt2 train chunks (~10B tokens)
    batch_size: int = 8 * 64 * 1024    # tokens per optimizer step (524288)
    val_tokens: int = 4 * 524288       # tokens per validation pass (wall-clock curve)
    mbs: int = 4                       # microbatch (sequences); 4 for seq_len=2048 (=8 at seq_len=1024)
    cooldown_frac: float = 0.7
    sketch_dim: int = 64               # sigreg sketch dim
    sigr_alpha_on: float = 0.01        # reg loss weight when reg_mode != baseline
    dist_align_weight_on: float = 0.01 # distractor loss weight when distractor on

    # --- memory branch knobs (fixed) ---
    mem_chunk: int = 128
    mem_gamma_bias: float = 3.9
    mem_beta_bias: float = 0.0
    mem_kernel: str = "auto"
    fla_custom_op: bool = True         # set FLA_CUSTOM_OP=1 (compile-clean FLA path)
    wall_gate_bias: float | None = None # Wall diagnostic only; incompatible with fair grid protocol

    # --- eval (fixed) ---
    eval_lengths: list = field(default_factory=lambda: list(EVAL_LENGTHS))
    needle_distances: list = field(default_factory=lambda: list(EVAL_LENGTHS))
    clean_dataset: str = "codelion/finepdfs-1B"
    val_data: str = "finewebedu10B/finewebedu_val_*.bin"
    num_eval_docs: int = 64            # docs for clean perplexity (and needle haystack)
    num_needle_trials: int = 64
    needle_val_len: int = 5

    # --- derived (filled in __post_init__; included in the JSON) ---
    run_id: str = field(init=False, default="")
    num_random_keys: int = field(init=False, default=0)
    dist_align_loss_weight: float = field(init=False, default=0.0)
    sigr_alpha: float = field(init=False, default=0.0)
    attn_window: "int | None" = field(init=False, default=None)
    mem_enabled: bool = field(init=False, default=False)

    def __post_init__(self):
        assert self.attn_type in ALL_ATTN_TYPES, self.attn_type
        assert self.reg_mode in REG_MODES, self.reg_mode
        self.num_random_keys = self.seq_len if self.distractor else 0
        self.dist_align_loss_weight = self.dist_align_weight_on if self.distractor else 0.0
        self.sigr_alpha = self.sigr_alpha_on if self.reg_mode != "baseline" else 0.0
        self.attn_window = self.window_size if self.window else None
        self.mem_enabled = bool(self.memory)
        self.run_id = (f"{self.attn_type}__reg-{self.reg_mode}"
                       f"__distr-{int(self.distractor)}"
                       f"__mem-{int(self.memory)}"
                       f"__win-{int(self.window)}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "RunConfig":
        init_names = {f.name for f in dataclasses.fields(RunConfig) if f.init}
        return RunConfig(**{k: v for k, v in d.items() if k in init_names})


def expand_grid(attn_types=None, **base_overrides) -> list:
    """The final scaled ablation cells.

    `attn_types=["wall"]` remains available for reproducing the incompatible diagnostic run,
    but Wall is excluded from the default expected grid and main dashboards.
    """
    attn_types = list(ATTN_TYPES if attn_types is None else attn_types)
    unknown = sorted(set(attn_types) - set(ALL_ATTN_TYPES))
    assert not unknown, f"unknown attn_type(s): {unknown}"
    cells = itertools.product(attn_types, REG_MODES, (False,), (True,), (False,))
    return [RunConfig(attn_type=a, reg_mode=r, distractor=d, memory=m, window=w, **base_overrides)
            for (a, r, d, m, w) in cells]


def shard_configs(configs, n, seed=0):
    """Split configs into n balanced shards (round-robin over a deterministic shuffle, so each
    shard gets a similar mix of heavy/light cells -> GPUs finish in comparable wall-time)."""
    import random
    idx = list(range(len(configs)))
    random.Random(seed).shuffle(idx)
    shards = [[] for _ in range(n)]
    for rank, j in enumerate(idx):
        shards[rank % n].append(configs[j])
    return shards
