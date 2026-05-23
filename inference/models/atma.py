import torch
from torch import nn
import torch.nn.functional as F

from model.config import AtmaConfig
from model.layers import RMSNorm, MLP
from model.blocks import AtmaConvBase, AtmaAttnBase
from inference.layers.linear import ReplicatedLinear
from inference.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from inference.layers.attention import (
    Attention, store_kvcache,
    HAS_FLASH_ATTN2, flash_attn_varlen_func, flash_attn_with_kvcache,
    _fa3,
)
from inference.utils.context import get_context


try:
    from kernels import get_kernel as _get_kernel
    _conv_mod = _get_kernel("kernels-community/causal-conv1d")
    causal_conv1d_fn = _conv_mod.causal_conv1d_fn
    causal_conv1d_update = _conv_mod.causal_conv1d_update
except Exception:
    causal_conv1d_fn = None
    causal_conv1d_update = None


def _infer_linear(in_f, out_f):
    return ReplicatedLinear(in_f, out_f, bias=True)


# ---------------------------------------------------------------------------
# Causal conv helpers
# ---------------------------------------------------------------------------

def prefill_causal_conv1d(
    layer_id: str,
    seq,
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    bias,
    conv_state_tables: dict,
) -> torch.Tensor:
    """Run causal conv over a full prefill sequence; save final state to GPU conv state table."""
    seqlen, hdim = x_seq.shape
    kernel_size = weight.shape[1]
    x_input = x_seq.transpose(0, 1).unsqueeze(0)  # (1, hdim, seqlen)

    if causal_conv1d_fn is not None and x_input.is_cuda:
        out, final_state = causal_conv1d_fn(x_input, weight, bias, return_final_states=True)
        conv_state_tables[layer_id][seq.seq_slot] = final_state.squeeze(0)  # (hdim, ks-1)
        out = out.squeeze(0).transpose(0, 1)  # (seqlen, hdim)
    else:
        x_padded = F.pad(x_input, (kernel_size - 1, 0))
        w = weight.unsqueeze(1)
        out = F.conv1d(x_padded, w, bias, stride=1, padding=0, groups=hdim)
        out = out.squeeze(0).transpose(0, 1)
        if seqlen < kernel_size - 1:
            pad_len = (kernel_size - 1) - seqlen
            state = F.pad(x_seq.transpose(0, 1), (pad_len, 0))
        else:
            state = x_seq[-(kernel_size - 1):].transpose(0, 1).contiguous()
        conv_state_tables[layer_id][seq.seq_slot] = state

    return out


def _gpu_conv_step(
    layer_id: str,
    seq_slots: torch.Tensor,
    conv_state_tables: dict,
    new_vals: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
) -> torch.Tensor:
    """CUDA-graph-compatible batched causal conv step via GPU-indexed gather/scatter.

    seq_slots         : (bs,) int64 GPU tensor — indices into conv_state_tables rows
    conv_state_tables : dict[str, Tensor(max_seqs, hdim, ks-1)]
    new_vals          : (bs, hdim) — new input token features
    weight            : (hdim, kernel_size) — depthwise conv weights
    Returns           : (bs, hdim) conv output (residual; caller adds to input)
    """
    states = conv_state_tables[layer_id][seq_slots]          # (bs, hdim, ks-1) gather
    out = (states * weight[:, :-1].unsqueeze(0)).sum(2)      # (bs, hdim)
    out = out + new_vals * weight[:, -1].unsqueeze(0)
    if bias is not None:
        out = out + bias.unsqueeze(0)
    new_state = torch.cat([states[:, :, 1:], new_vals.unsqueeze(2)], dim=2)
    conv_state_tables[layer_id][seq_slots] = new_state        # (bs, hdim, ks-1) scatter
    return out


# ---------------------------------------------------------------------------
# Attention layer
# ---------------------------------------------------------------------------

class AtmaAttention(AtmaAttnBase):

    def __init__(self, layer_idx: int, dim: int, head_dim: int = 128, num_kv_heads: int = None, kernel_size: int = 4):
        super().__init__(dim, linear_cls=_infer_linear, head_dim=head_dim, num_kv_heads=num_kv_heads, kernel_size=kernel_size)
        self.layer_idx = layer_idx
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim ** -0.5, self.num_kv_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_context()
        w_q = self.canon_q.weight.squeeze(1)  # (hdim, kernel_size)
        w_k = self.canon_k.weight.squeeze(1)
        w_v = self.canon_v.weight.squeeze(1)

        if context.is_prefill:
            total = x.shape[0]
            q_gate = self.q(x).view(total, self.num_heads, self.head_dim * 2)
            q_all, gate_all = torch.chunk(q_gate, 2, dim=-1)
            k_all = self.k(x).view(total, self.num_kv_heads, self.head_dim)
            v_all = self.v(x).view(total, self.num_kv_heads, self.head_dim)

            q_all = F.rms_norm(q_all, (self.head_dim,))
            k_all = F.rms_norm(k_all, (self.head_dim,))

            conv_state_tables = context.conv_state_tables
            q_parts, k_parts, v_parts = [], [], []
            start = 0
            for i, seqlen in enumerate(context.seqlens_q):
                seq = context.seqs[i]
                qi = q_all[start:start + seqlen].reshape(seqlen, -1)
                ki = k_all[start:start + seqlen].reshape(seqlen, -1)
                vi = v_all[start:start + seqlen].reshape(seqlen, -1)
                qi_conv = qi + prefill_causal_conv1d(f"attn_{self.layer_idx}_q", seq, qi, w_q, None, conv_state_tables)
                ki_conv = ki + prefill_causal_conv1d(f"attn_{self.layer_idx}_k", seq, ki, w_k, None, conv_state_tables)
                vi_conv = vi + prefill_causal_conv1d(f"attn_{self.layer_idx}_v", seq, vi, w_v, None, conv_state_tables)
                q_parts.append(qi_conv.view(seqlen, self.num_heads, self.head_dim))
                k_parts.append(ki_conv.view(seqlen, self.num_kv_heads, self.head_dim))
                v_parts.append(vi_conv.view(seqlen, self.num_kv_heads, self.head_dim))
                start += seqlen

            q_packed = torch.cat(q_parts, dim=0)
            k_packed = torch.cat(k_parts, dim=0)
            v_packed = torch.cat(v_parts, dim=0)

            # Write K/V to paged cache
            if self.attn.k_cache.numel() > 0 and context.slot_mapping is not None and context.slot_mapping.numel() > 0:
                store_kvcache(k_packed, v_packed, self.attn.k_cache, self.attn.v_cache, context.slot_mapping)

            if _fa3 is not None and context.block_tables is None and q_packed.is_cuda:
                # FA3 varlen has no block_table argument; when prefix cache is active,
                # cu_seqlens_k > cu_seqlens_q but k_packed holds only new tokens —
                # FA3 reads past k_packed and triggers an illegal memory access.
                y = _fa3.flash_attn_varlen_func(
                    q_packed, k_packed, v_packed,
                    context.cu_seqlens_q, context.cu_seqlens_k,
                    context.max_seqlen_q, context.max_seqlen_k,
                    softmax_scale=self.attn.scale, causal=True,
                )
            elif HAS_FLASH_ATTN2 and q_packed.is_cuda and (context.block_tables is None or self.attn.k_cache.shape[1] % 256 == 0):
                # With prefix cache, K/V come from paged cache via block_table.
                # Without prefix cache, pass packed K/V directly (cache format mismatch).
                k_src = self.attn.k_cache if context.block_tables is not None else k_packed
                v_src = self.attn.v_cache if context.block_tables is not None else v_packed
                y = flash_attn_varlen_func(
                    q_packed, k_src, v_src,
                    cu_seqlens_q=context.cu_seqlens_q,
                    cu_seqlens_k=context.cu_seqlens_k,
                    max_seqlen_q=context.max_seqlen_q,
                    max_seqlen_k=context.max_seqlen_k,
                    softmax_scale=self.attn.scale,
                    causal=True,
                    block_table=context.block_tables,
                )
            else:
                # SDPA fallback — expand KV heads for GQA
                groups = self.num_heads // self.num_kv_heads
                q_b = q_packed.unsqueeze(0).transpose(1, 2)
                k_b = k_packed.unsqueeze(0).transpose(1, 2).repeat_interleave(groups, dim=1)
                v_b = v_packed.unsqueeze(0).transpose(1, 2).repeat_interleave(groups, dim=1)
                y = F.scaled_dot_product_attention(q_b, k_b, v_b, scale=self.attn.scale, is_causal=True)
                y = y.transpose(1, 2).squeeze(0)

            y = y.reshape(total, self.num_heads * self.head_dim)
            y = y * torch.sigmoid(gate_all.reshape(total, -1))
            return self.proj(y)

        else:
            # Decode — all GPU-indexed ops, CUDA-graph-compatible
            batch_size = x.shape[0]
            conv_state_tables = context.conv_state_tables
            seq_slots = context.seq_slots  # (bs,) int64 GPU tensor

            q_gate = self.q(x).view(batch_size, self.num_heads, self.head_dim * 2)
            q_all, gate_all = torch.chunk(q_gate, 2, dim=-1)
            k_all = self.k(x).view(batch_size, self.num_kv_heads, self.head_dim)
            v_all = self.v(x).view(batch_size, self.num_kv_heads, self.head_dim)

            q_all = F.rms_norm(q_all, (self.head_dim,))
            k_all = F.rms_norm(k_all, (self.head_dim,))

            q_flat = q_all.reshape(batch_size, -1)
            k_flat = k_all.reshape(batch_size, -1)
            v_flat = v_all.reshape(batch_size, -1)

            q_conv = q_flat + _gpu_conv_step(f"attn_{self.layer_idx}_q", seq_slots, conv_state_tables, q_flat, w_q)
            k_conv = k_flat + _gpu_conv_step(f"attn_{self.layer_idx}_k", seq_slots, conv_state_tables, k_flat, w_k)
            v_conv = v_flat + _gpu_conv_step(f"attn_{self.layer_idx}_v", seq_slots, conv_state_tables, v_flat, w_v)

            k_attn = k_conv.view(batch_size, self.num_kv_heads, self.head_dim)
            v_attn = v_conv.view(batch_size, self.num_kv_heads, self.head_dim)
            q_attn = q_conv.view(batch_size, self.num_heads, self.head_dim)

            store_kvcache(k_attn, v_attn, self.attn.k_cache, self.attn.v_cache, context.slot_mapping)

            if _fa3 is not None and q_attn.is_cuda:
                # Token already written via store_kvcache; pass full context_lens, use page_table.
                # num_splits=1 disables FA3's auto-split: with num_splits=0 the split decision is
                # made in Python before the kernel launch, so capture (context_lens=0 → no split)
                # and replay (context_lens>0 → split) choose different kernel paths, causing a
                # workspace-pointer mismatch and an illegal memory access inside CUDA graphs.
                y = _fa3.flash_attn_with_kvcache(
                    q_attn.unsqueeze(1), self.attn.k_cache, self.attn.v_cache,
                    cache_seqlens=context.context_lens,
                    page_table=context.block_tables,
                    softmax_scale=self.attn.scale,
                    causal=False,
                    num_splits=1,
                ).squeeze(1).reshape(batch_size, -1)
            elif HAS_FLASH_ATTN2 and q_attn.is_cuda and self.attn.k_cache.shape[1] % 256 == 0:
                y = flash_attn_with_kvcache(
                    q_attn.unsqueeze(1),
                    self.attn.k_cache, self.attn.v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.attn.scale,
                    causal=False,
                ).squeeze(1).reshape(batch_size, -1)
            else:
                # SDPA fallback — batched gather from paged cache, no Python loop
                max_seqlen = int(context.context_lens.max().item())
                block_size = self.attn.k_cache.shape[1]
                n_blocks = context.block_tables.shape[1]
                k_full = self.attn.k_cache[context.block_tables.clamp(min=0)].reshape(
                    batch_size, n_blocks * block_size, self.num_kv_heads, self.head_dim
                )[:, :max_seqlen]
                v_full = self.attn.v_cache[context.block_tables.clamp(min=0)].reshape(
                    batch_size, n_blocks * block_size, self.num_kv_heads, self.head_dim
                )[:, :max_seqlen]
                # Expand KV heads for GQA
                groups = self.num_heads // self.num_kv_heads
                k_full = k_full.repeat_interleave(groups, dim=2)
                v_full = v_full.repeat_interleave(groups, dim=2)
                q_b = q_attn.unsqueeze(1).transpose(1, 2)
                k_b = k_full.transpose(1, 2)
                v_b = v_full.transpose(1, 2)
                y = F.scaled_dot_product_attention(q_b, k_b, v_b, scale=self.attn.scale, is_causal=False)
                y = y.transpose(1, 2).reshape(batch_size, -1)

            return self.proj(y * torch.sigmoid(gate_all.reshape(batch_size, -1)))


# ---------------------------------------------------------------------------
# Conv layer
# ---------------------------------------------------------------------------

class AtmaLFM2Conv(AtmaConvBase):

    def __init__(self, layer_idx: int, dim: int, kernel_size: int = 3):
        super().__init__(dim, linear_cls=_infer_linear, kernel_size=kernel_size)
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_context()
        projected = self.in_proj(x)
        B_all, C_all, x_proj_all = projected.chunk(3, dim=-1)
        x_gated_all = B_all * x_proj_all
        w_conv = self.conv.weight.squeeze(1)  # (hdim, kernel_size)

        if context.is_prefill:
            conv_state_tables = context.conv_state_tables
            y_parts = []
            start = 0
            for i, seqlen in enumerate(context.seqlens_q):
                seq = context.seqs[i]
                x_gated_i = x_gated_all[start:start + seqlen]
                C_i = C_all[start:start + seqlen]
                x_conv_i = prefill_causal_conv1d(
                    f"conv_{self.layer_idx}_gated", seq, x_gated_i, w_conv, None, conv_state_tables
                )
                y_parts.append(C_i * x_conv_i)
                start += seqlen
            return self.out_proj(torch.cat(y_parts, dim=0))

        else:
            x_conv = _gpu_conv_step(
                f"conv_{self.layer_idx}_gated",
                context.seq_slots, context.conv_state_tables,
                x_gated_all, w_conv,
            )
            return self.out_proj(C_all * x_conv)


# ---------------------------------------------------------------------------
# Decoder block and full model
# ---------------------------------------------------------------------------

class AtmaDecoderBlock(nn.Module):

    def __init__(
        self,
        layer_idx: int,
        dim: int,
        attention: bool = True,
        head_dim: int = 128,
        num_kv_heads: int = None,
        attn_kernel_size: int = 4,
        conv_kernel_size: int = 3,
    ):
        super().__init__()
        self.attn = (
            AtmaAttention(layer_idx, dim, head_dim=head_dim, num_kv_heads=num_kv_heads, kernel_size=attn_kernel_size)
            if attention
            else AtmaLFM2Conv(layer_idx, dim, kernel_size=conv_kernel_size)
        )
        self.mlp = MLP(dim, linear_cls=_infer_linear)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Atma(nn.Module):

    def __init__(self, config: AtmaConfig):
        super().__init__()
        self.embed = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([
            AtmaDecoderBlock(
                i, config.hidden_size,
                attention=(i % 4 == 2),
                head_dim=config.head_dim,
                num_kv_heads=config.num_key_value_heads,
                attn_kernel_size=config.attn_kernel_size,
                conv_kernel_size=config.conv_kernel_size,
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)
        self.proj = ParallelLMHead(config.vocab_size, config.hidden_size, bias=True)

        print(f"Total parameters: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        """Returns hidden states. Call compute_logits() separately (required for CUDA graph)."""
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.proj(hidden_states)
        return 15.0 * logits * (logits.square() + 225.0).rsqrt()
