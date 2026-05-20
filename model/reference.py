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
from model.blocks import AtmaConvBase, AtmaAttnBase


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

    def __init__(self, dim: int, head_dim: int = 128, kernel_size: int = 4):
        super().__init__(dim, linear_cls=Linear, head_dim=head_dim, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q_gate = self.q(x).view(B, T, self.num_heads, self.head_dim * 2)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)

        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))

        q_in = q.reshape(B, T, -1).transpose(1, 2)  # (B, hdim, T)
        k_in = k.reshape(B, T, -1).transpose(1, 2)
        v_in = v.reshape(B, T, -1).transpose(1, 2)

        def _causal_conv1d(x_in: torch.Tensor, conv_mod: nn.Conv1d) -> torch.Tensor:
            K = conv_mod.weight.size(2)
            return F.conv1d(F.pad(x_in, (K - 1, 0)), conv_mod.weight, stride=1, padding=0, groups=x_in.size(1))

        q_out = q_in + _causal_conv1d(q_in, self.canon_q)
        k_out = k_in + _causal_conv1d(k_in, self.canon_k)
        v_out = v_in + _causal_conv1d(v_in, self.canon_v)

        q_attn = q_out.transpose(1, 2).reshape(B, T, self.num_heads, self.head_dim)
        k_attn = k_out.transpose(1, 2).reshape(B, T, self.num_heads, self.head_dim)
        v_attn = v_out.transpose(1, 2).reshape(B, T, self.num_heads, self.head_dim)

        y = F.scaled_dot_product_attention(
            q_attn.transpose(1, 2),
            k_attn.transpose(1, 2),
            v_attn.transpose(1, 2),
            is_causal=True,
        ).transpose(1, 2)  # (B, T, num_heads, head_dim)

        y = y.reshape(B, T, -1) * torch.sigmoid(gate.reshape(B, T, -1))
        return self.proj(y)


class Block(nn.Module):
    """Reference decoder block: no regularization, returns tensor only."""

    def __init__(self, dim: int, attention: bool = True):
        super().__init__()
        self.attn = CausalSelfAttention(dim) if attention else LFM2Conv(dim)
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
            Block(config.hidden_size, attention=True) if i % 4 == 2
            else Block(config.hidden_size, attention=False)
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
