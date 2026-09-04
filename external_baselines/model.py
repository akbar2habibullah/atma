"""Atma-evaluation-compatible wrappers for TDA, Mamba-3, and GDN-2.

The wrappers deliberately expose the same lightweight contract as ``raven_baseline``:
``embed``, ``blocks``, ``norm``, and ``proj``.  This lets the existing long-context
evaluation code compare checkpoints without architecture-specific metric paths.

Mamba-3 and GDN-2 delegate their token mixer to the pinned upstream FLA layers.  TDA
delegates the attention reduction to the authors' pinned Triton implementation.  These
dependencies are not vendored because their kernels must be installed and verified on
the target GPU instance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from model.layers import MLP, RMSNorm
from model.blocks import TitansMemory
from raven_baseline.layers import Linear
from train.model import LFM2Conv
from train.reg import sigreg


class TDAAttention(nn.Module):
    """Projection surround for the official Threshold Differential Attention kernel."""

    def __init__(self, cfg: dict):
        super().__init__()
        hidden = int(cfg["hidden_size"])
        head_dim = int(cfg.get("head_dim", 128))
        if hidden % head_dim:
            raise ValueError("TDA hidden_size must be divisible by head_dim")
        self.hidden_size = hidden
        self.head_dim = head_dim
        self.num_heads = hidden // head_dim
        self.q_proj = Linear(hidden, 2 * hidden, bias=False)
        self.k_proj = Linear(hidden, 2 * hidden, bias=False)
        self.v_proj = Linear(hidden, hidden, bias=False)
        self.o_proj = Linear(hidden, hidden, bias=False)
        self.lambda_param = nn.Parameter(torch.tensor(float(cfg.get("tda_lambda_init", 0.5))))
        self.register_buffer("beta", torch.tensor(float(cfg.get("tda_beta", 1.0))), persistent=True)
        self.relu_power = float(cfg.get("tda_relu_power", 2.0))
        self.source_dir = str(cfg.get("tda_source_dir", "third_party/TDA"))
        self.mem = (
            TitansMemory(
                hidden, self.num_heads, self.head_dim, Linear,
                chunk=int(cfg.get("mem_chunk", 128)),
                gamma_bias=float(cfg.get("mem_gamma_bias", 3.9)),
                beta_bias=float(cfg.get("mem_beta_bias", 0.0)),
                kernel=str(cfg.get("mem_kernel", "auto")),
            )
            if bool(cfg.get("mem_enabled", False)) else None
        )

    def _kernel(self):
        source = Path(self.source_dir).resolve()
        if source.is_dir() and str(source) not in sys.path:
            sys.path.insert(0, str(source))
        try:
            from triton_threshold_attention import differential_threshold_rela_triton
        except ImportError as exc:
            raise RuntimeError(
                "TDA kernel unavailable. Clone the pinned snap-research/TDA commit into "
                f"{source} and run the GPU preflight before training."
            ) from exc
        return differential_threshold_rela_triton

    def forward(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        q1, q2 = self.q_proj(x).chunk(2, dim=-1)
        k1, k2 = self.k_proj(x).chunk(2, dim=-1)
        q1, q2, k1, k2 = (z.view(shape).transpose(1, 2).contiguous() for z in (q1, q2, k1, k2))
        v = self.v_proj(x).view(shape).transpose(1, 2).contiguous()
        y = self._kernel()(
            q1, q2, k1, k2, v, self.beta, self.lambda_param,
            relu_power=self.relu_power, normalize=True,
        )
        # The paper's output definition includes Norm after the signed reduction.
        y = F.rms_norm(y, (self.head_dim,))
        out = self.o_proj(y.transpose(1, 2).reshape(batch, length, self.hidden_size))
        if self.mem is not None:
            # Use the excitatory view for the matched Titans side channel. This is
            # recorded explicitly because TDA has no unique Q/K view to share.
            out = out + self.mem(x, q1, k1, v)
        return out


def _make_external_mixer(cfg: dict, layer_idx: int) -> nn.Module:
    arch = cfg["arch_type"]
    hidden = int(cfg["hidden_size"])
    head_dim = int(cfg.get("head_dim", 128))
    if bool(cfg.get("external_custom_op", False)):
        from external_baselines.custom_ops import install_external_custom_op

        install_external_custom_op(arch)
    if arch == "mamba3_native":
        try:
            from fla.layers import Mamba3
        except ImportError as exc:
            raise RuntimeError("Mamba-3 requires the pinned flash-linear-attention source checkout") from exc
        return Mamba3(
            hidden_size=hidden,
            state_size=int(cfg.get("mamba3_state_size", 128)),
            expand=int(cfg.get("mamba3_expand", 2)),
            head_dim=int(cfg.get("mamba3_head_dim", 64)),
            n_groups=int(cfg.get("mamba3_n_groups", 1)),
            rope_fraction=float(cfg.get("mamba3_rope_fraction", 0.5)),
            is_outproj_norm=bool(cfg.get("mamba3_outproj_norm", False)),
            is_mimo=bool(cfg.get("mamba3_mimo", False)),
            mimo_rank=int(cfg.get("mamba3_mimo_rank", 4)),
            chunk_size=int(cfg.get("mamba3_chunk_size", 64)),
            layer_idx=layer_idx,
        )
    if arch == "gdn2_native":
        try:
            from fla.layers import GatedDeltaNet2
        except ImportError as exc:
            raise RuntimeError("GDN-2 requires the pinned flash-linear-attention source checkout") from exc
        if hidden % head_dim:
            raise ValueError("GDN-2 hidden_size must be divisible by head_dim")
        heads = hidden // head_dim
        return GatedDeltaNet2(
            hidden_size=hidden,
            expand_v=float(cfg.get("gdn2_expand_v", 1.0)),
            head_dim=head_dim,
            num_heads=heads,
            num_v_heads=int(cfg.get("gdn2_num_v_heads", heads)),
            mode="chunk",
            use_short_conv=bool(cfg.get("gdn2_short_conv", True)),
            allow_neg_eigval=bool(cfg.get("gdn2_allow_neg_eigval", False)),
            conv_size=int(cfg.get("gdn2_conv_size", 4)),
            layer_idx=layer_idx,
        )
    if arch == "tda_hybrid":
        return TDAAttention(cfg)
    raise ValueError(f"unsupported external arch_type={arch!r}")


class ExternalBlock(nn.Module):
    def __init__(self, cfg: dict, layer_idx: int, local: bool):
        super().__init__()
        hidden = int(cfg["hidden_size"])
        self.attn = (
            LFM2Conv(hidden, kernel_size=int(cfg.get("conv_kernel_size", 3)))
            if local else _make_external_mixer(cfg, layer_idx)
        )
        self.norm1 = RMSNorm(hidden)
        self.norm2 = RMSNorm(hidden)
        self.mlp = MLP(hidden, linear_cls=Linear)
        self.reg_mode = cfg.get("reg_mode", "baseline")
        self.sketch_dim = int(cfg.get("sketch_dim", 64))

    def forward(self, x: Tensor):
        mixed = self.attn(self.norm1(x))
        if isinstance(mixed, tuple):
            mixed = mixed[0]
        x = x + mixed
        x = x + self.mlp(self.norm2(x))
        return x, sigreg(x, self.reg_mode, self.sketch_dim), x.new_zeros(())


class ExternalLM(nn.Module):
    """Native-mixer or Atma-hybrid causal LM with the standard evaluation contract."""

    def __init__(self, cfg: dict):
        super().__init__()
        hidden = int(cfg["hidden_size"])
        layers = int(cfg.get("num_hidden_layers", 16))
        self.cfg = dict(cfg)
        self.embed = nn.Embedding(int(cfg["vocab_size"]), hidden)
        hybrid = cfg["arch_type"] == "tda_hybrid"
        self.blocks = nn.ModuleList([
            ExternalBlock(cfg, i, local=(hybrid and i % 4 != 2))
            for i in range(layers)
        ])
        self.norm = RMSNorm(hidden)
        self.proj = Linear(hidden, int(cfg["vocab_size"]), bias=False)
        # FLA kernels are intended for bf16/fp16 training; the Atma data path emits
        # bf16 embeddings and does not wrap the forward pass in autocast.
        self.bfloat16()

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.embed(inputs)
        reg = x.new_zeros(())
        align = x.new_zeros(())
        for block in self.blocks:
            x, block_reg, block_align = block(x)
            reg = reg + block_reg
            align = align + block_align
        logits = self.proj(self.norm(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        loss = F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")
        return loss, reg / len(self.blocks), align / len(self.blocks)


def create_model(cfg: dict) -> ExternalLM:
    if cfg.get("baseline_family") != "external":
        raise ValueError("external baseline config must set baseline_family='external'")
    return ExternalLM(cfg)
