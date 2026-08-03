"""Minimal ModelRunner forks for benchmark-only attention baselines."""

from collections import deque
import torch

from inference.engine.model_runner import ModelRunner
from inference.engine.sequence import Sequence
from inference.layers.sampler import Sampler
from inference.utils.loader import load_model
from baseline_inference.softmax_model import SoftmaxLM
from baseline_inference.raven_model import RavenLM


class SoftmaxModelRunner(ModelRunner):
    def __init__(self, config, rank=0, event=None):
        self.config = config
        hf = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf.dtype)
        if self.device.type == "cuda":
            torch.set_default_device("cuda")
        self.model = SoftmaxLM(hf)
        try:
            load_model(self.model, config.model, strict=config.strict_weights)
        except Exception as e:
            if config.strict_weights:
                raise RuntimeError(f"strict baseline checkpoint loading failed: {e}") from e
            print(f"Baseline weight loading failed ({e}); using initialized weights")
        self.sampler = Sampler()
        self.model.eval()
        self.allocate_conv_state_tables()
        self.allocate_kv_cache()
        self.warmup_model()
        if not self.enforce_eager and self.device.type == "cuda":
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(old_dtype)

    def allocate_conv_state_tables(self):
        hf = self.config.hf_config
        n = self.config.max_num_seqs
        hidden = hf.hidden_size
        self.conv_state_tables = {}
        for i in range(hf.num_hidden_layers):
            if i % 4 == 2:
                if hf.attn_type == "nope":
                    for suffix, dim in (
                        ("q", hidden),
                        ("k", hf.num_key_value_heads * hf.head_dim),
                        ("v", hf.num_key_value_heads * hf.head_dim),
                    ):
                        self.conv_state_tables[f"attn_{i}_{suffix}"] = torch.zeros(
                            n, dim, hf.attn_kernel_size - 1, dtype=hf.dtype
                        )
                if hf.mem_enabled:
                    self.conv_state_tables[f"mem_{i}"] = torch.zeros(
                        n,
                        hf.num_attention_heads,
                        hf.head_dim,
                        hf.head_dim,
                        dtype=torch.float32,
                    )
            else:
                self.conv_state_tables[f"conv_{i}_gated"] = torch.zeros(
                    n, hidden, hf.conv_kernel_size - 1, dtype=hf.dtype
                )
        self._free_slots = deque(range(n))


class RavenModelRunner(ModelRunner):
    def __init__(self, config, rank=0, event=None):
        self.config = config
        self.raven_cfg = config.hf_config.raven_cfg
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = 1
        self.rank = rank
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        old = torch.get_default_dtype()
        torch.set_default_dtype(config.hf_config.dtype)
        if self.device.type == "cuda":
            torch.set_default_device("cuda")
        self.model = RavenLM(self.raven_cfg)
        try:
            load_model(self.model, config.model, strict=config.strict_weights)
        except Exception as e:
            if config.strict_weights:
                raise RuntimeError(f"strict Raven checkpoint loading failed: {e}") from e
            print(f"Raven weight loading failed ({e}); using initialized weights")
        self.sampler = Sampler()
        self.model.eval()
        self.allocate_conv_state_tables()
        self.allocate_kv_cache()
        self.warmup_model()
        if not self.enforce_eager and self.device.type == "cuda":
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(old)

    def allocate_kv_cache(self):
        # Raven is recurrent; blocks are scheduler bookkeeping only.
        self.config.num_kvcache_blocks = max(
            16,
            self.config.max_num_seqs
            * ((self.config.max_model_len + self.block_size - 1) // self.block_size),
        )

    def allocate_conv_state_tables(self):
        c = self.raven_cfg
        n = self.config.max_num_seqs
        D = c["hidden_size"]
        H = c["num_heads"]
        hd = D // H
        M = c["num_slots"]
        native = c["arch_type"] == "raven_native"
        self.conv_state_tables = {}
        for i in range(c["num_hidden_layers"]):
            conv = not native and i % 4 != 2
            if conv:
                self.conv_state_tables[f"conv_{i}_gated"] = torch.zeros(
                    n, D, c["conv_kernel_size"] - 1, dtype=self.config.hf_config.dtype
                )
            else:
                # FLA GSA final-state layout: [H,D,M] then [H,M,D].
                self.conv_state_tables[f"raven_k_{i}"] = torch.zeros(
                    n, H, hd, M, dtype=torch.float32
                )
                self.conv_state_tables[f"raven_v_{i}"] = torch.zeros(
                    n, H, M, hd, dtype=torch.float32
                )
                if c["arch_type"] == "atma_raven_titans":
                    self.conv_state_tables[f"mem_{i}"] = torch.zeros(
                        n, H, hd, hd, dtype=torch.float32
                    )
        self._free_slots = deque(range(n))


def make_runner(config):
    typ = config.hf_config.attn_type
    if typ in ("nope", "rope"):
        return SoftmaxModelRunner(config)
    if typ == "raven":
        return RavenModelRunner(config)
    raise ValueError(f"unsupported baseline architecture {typ!r}")
