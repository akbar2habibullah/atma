"""
model/reference.py — canonical reference implementation.

Pure PyTorch, no custom kernels, no KV cache, no TP. This is the ground truth
that both train and inference forward passes must match numerically.
"""

import torch
from torch import nn
import torch.nn.functional as F

from model.config import AtmaConfig
from model.layers import RMSNorm, MLP
from model.blocks import AtmaConvBase, AtmaAttnBase, polar_reduce

# Polar structural-prior inits (softplus^-1 of validated targets: g~0.3, slope~1, beta~0.2)
_LEN_GAIN_INIT = -1.0
_NULL_SLOPE_INIT = 0.5
_NULL_BASE_INIT = 2.0
_MAG_BETA_INIT = -1.5


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x) if self.bias is not None else None)


class LFM2Conv(AtmaConvBase):
    """Reference LFM2 gated conv: pure PyTorch depthwise causal conv1d, batch-first (B, T, H)."""

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__(dim, linear_cls=Linear, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H = x.shape
        projected = self.in_proj(x)
        B_gate, C, x_proj = projected.chunk(3, dim=-1)
        x_gated = B_gate * x_proj

        x_in = x_gated.transpose(1, 2)  # (B, H, T)
        K = self.conv.weight.size(2)
        x_padded = F.pad(x_in, (K - 1, 0))
        x_conv = F.conv1d(x_padded, self.conv.weight, stride=1, padding=0, groups=H)
        x_conv = x_conv.transpose(1, 2)  # (B, T, H)

        return self.out_proj(C * x_conv)


class CausalSelfAttention(AtmaAttnBase):
    """Reference Canon-B attention: pure SDPA, batch-first (B, T, H)."""

    def __init__(self, dim: int, head_dim: int = 128, num_kv_heads: int = None, kernel_size: int = 4):
        super().__init__(dim, linear_cls=Linear, head_dim=head_dim, num_kv_heads=num_kv_heads, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q_gate = self.q(x).view(B, T, self.num_heads, self.head_dim * 2)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        k = self.k(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_kv_heads, self.head_dim)

        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))

        q_in = q.reshape(B, T, -1).transpose(1, 2)  # (B, hdim, T)
        k_in = k.reshape(B, T, -1).transpose(1, 2)  # (B, kv_hdim, T)
        v_in = v.reshape(B, T, -1).transpose(1, 2)

        def _causal_conv1d(x_in: torch.Tensor, conv_mod: nn.Conv1d) -> torch.Tensor:
            K = conv_mod.weight.size(2)
            return F.conv1d(F.pad(x_in, (K - 1, 0)), conv_mod.weight, stride=1, padding=0, groups=x_in.size(1))

        q_out = q_in + _causal_conv1d(q_in, self.canon_q)
        k_out = k_in + _causal_conv1d(k_in, self.canon_k)
        v_out = v_in + _causal_conv1d(v_in, self.canon_v)

        q_attn = q_out.transpose(1, 2).reshape(B, T, self.num_heads, self.head_dim)
        k_attn = k_out.transpose(1, 2).reshape(B, T, self.num_kv_heads, self.head_dim)
        v_attn = v_out.transpose(1, 2).reshape(B, T, self.num_kv_heads, self.head_dim)

        # Expand KV heads to match query heads for SDPA (GQA)
        groups = self.num_heads // self.num_kv_heads
        k_attn = k_attn.repeat_interleave(groups, dim=2)
        v_attn = v_attn.repeat_interleave(groups, dim=2)

        y = F.scaled_dot_product_attention(
            q_attn.transpose(1, 2),
            k_attn.transpose(1, 2),
            v_attn.transpose(1, 2),
            is_causal=True,
        ).transpose(1, 2)  # (B, T, num_heads, head_dim)

        y = y.reshape(B, T, -1) * torch.sigmoid(gate.reshape(B, T, -1))
        return self.proj(y)


class PolarAttention(AtmaAttnBase):
    """Reference Polar attention: direction + bounded count channels, pure PyTorch.

    Same projections / Canon conv / GQA as CausalSelfAttention; the SDPA core is
    replaced by the validated polar_reduce, and a per-head count channel is injected.
    """

    def __init__(self, dim: int, head_dim: int = 128, num_kv_heads: int = None, kernel_size: int = 4):
        super().__init__(dim, linear_cls=Linear, head_dim=head_dim, num_kv_heads=num_kv_heads, kernel_size=kernel_size)
        H, dk = self.num_heads, self.head_dim
        self.mu_proj = Linear(H, dim)                                   # count channel -> residual
        self.v_null = nn.Parameter(torch.zeros(H, dk))                 # default direction (null sink)
        self.null_base = nn.Parameter(torch.full((H,), _NULL_BASE_INIT))
        self.null_slope_raw = nn.Parameter(torch.full((H,), _NULL_SLOPE_INIT))
        self.len_gain_raw = nn.Parameter(torch.full((H,), _LEN_GAIN_INIT))
        self.mag_beta_raw = nn.Parameter(torch.full((H,), _MAG_BETA_INIT))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, dk = self.num_heads, self.head_dim

        q_gate = self.q(x).view(B, T, H, dk * 2)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        k = self.k(x).view(B, T, self.num_kv_heads, dk)
        v = self.v(x).view(B, T, self.num_kv_heads, dk)

        q = F.rms_norm(q, (dk,))
        k = F.rms_norm(k, (dk,))

        q_in = q.reshape(B, T, -1).transpose(1, 2)
        k_in = k.reshape(B, T, -1).transpose(1, 2)
        v_in = v.reshape(B, T, -1).transpose(1, 2)

        def _causal_conv1d(x_in: torch.Tensor, conv_mod: nn.Conv1d) -> torch.Tensor:
            K = conv_mod.weight.size(2)
            return F.conv1d(F.pad(x_in, (K - 1, 0)), conv_mod.weight, stride=1, padding=0, groups=x_in.size(1))

        q_out = q_in + _causal_conv1d(q_in, self.canon_q)
        k_out = k_in + _causal_conv1d(k_in, self.canon_k)
        v_out = v_in + _causal_conv1d(v_in, self.canon_v)

        q_attn = q_out.transpose(1, 2).reshape(B, T, H, dk)
        k_attn = k_out.transpose(1, 2).reshape(B, T, self.num_kv_heads, dk)
        v_attn = v_out.transpose(1, 2).reshape(B, T, self.num_kv_heads, dk)

        groups = H // self.num_kv_heads
        k_attn = k_attn.repeat_interleave(groups, dim=2)
        v_attn = v_attn.repeat_interleave(groups, dim=2)

        q_t = q_attn.transpose(1, 2)                       # (B, H, T, dk)
        k_t = k_attn.transpose(1, 2)
        v_t = v_attn.transpose(1, 2)

        sigma = torch.matmul(q_t, k_t.transpose(-2, -1)) / (dk ** 0.5)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device, dtype=sigma.dtype), diagonal=1)
        sigma = sigma + mask
        n_keys = torch.arange(1, T + 1, device=x.device, dtype=torch.float32)

        c, mag = polar_reduce(
            sigma, v_t, n_keys,
            v_null=self.v_null, null_base=self.null_base, null_slope_raw=self.null_slope_raw,
            len_gain_raw=self.len_gain_raw, mag_beta_raw=self.mag_beta_raw,
        )

        c_flat = c.transpose(1, 2).reshape(B, T, H * dk)
        content = self.proj(c_flat * torch.sigmoid(gate.reshape(B, T, -1)))
        count = self.mu_proj(mag.transpose(1, 2))          # (B, T, H) -> (B, T, dim)
        return content + count


class Block(nn.Module):
    """Reference decoder block: no regularization, returns tensor only."""

    def __init__(
        self,
        dim: int,
        attention: bool = True,
        head_dim: int = 128,
        num_kv_heads: int = None,
        attn_kernel_size: int = 4,
        conv_kernel_size: int = 3,
    ):
        super().__init__()
        self.attn = (
            PolarAttention(dim, head_dim=head_dim, num_kv_heads=num_kv_heads, kernel_size=attn_kernel_size)
            if attention
            else LFM2Conv(dim, kernel_size=conv_kernel_size)
        )
        self.mlp = MLP(dim, linear_cls=Linear)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ReferenceModel(nn.Module):
    """Full reference model: takes token ids, returns logits for all positions."""

    def __init__(self, config: AtmaConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([
            Block(
                config.hidden_size,
                attention=(i % 4 == 2),
                head_dim=config.head_dim,
                num_kv_heads=config.num_key_value_heads,
                attn_kernel_size=config.attn_kernel_size,
                conv_kernel_size=config.conv_kernel_size,
            )
            for i in range(config.num_hidden_layers)
        ])
        self.proj = Linear(config.hidden_size, config.vocab_size)
        self.norm = RMSNorm(config.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm(x)).float()
        logits = 15.0 * logits * (logits.square() + 225.0).rsqrt()
        return logits
