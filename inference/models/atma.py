import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from model.config import AtmaConfig
from model.layers import RMSNorm, MLP
from model.blocks import AtmaConvBase, AtmaAttnBase
from inference.layers.linear import ReplicatedLinear
from inference.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from inference.layers.attention import Attention, store_kvcache, gather_kv_from_cache, _fa3
from inference.utils.context import get_context
from inference.engine.sequence import Sequence


def causal_conv1d_fallback(x: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    """
    Pure PyTorch causal 1D depthwise convolution fallback.
    x: (B, hdim, T)
    weight: (hdim, kernel_size)
    bias: (hdim,) or None
    """
    kernel_size = weight.shape[1]
    x_padded = F.pad(x, (kernel_size - 1, 0))
    w = weight.unsqueeze(1)  # (hdim, 1, kernel_size)
    out = F.conv1d(x_padded, w, bias, stride=1, padding=0, groups=x.shape[1])
    return out


try:
    from kernels import get_kernel as _get_kernel
    _conv_mod = _get_kernel("kernels-community/causal-conv1d")
    causal_conv1d_fn = _conv_mod.causal_conv1d_fn
    causal_conv1d_update = _conv_mod.causal_conv1d_update
except Exception:
    print("causal-conv1d kernel not available, using PyTorch fallback")
    causal_conv1d_fn = None
    causal_conv1d_update = None


@torch.compiler.disable
def prefill_causal_conv1d(
    layer_id: str,
    seq: Sequence,
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
) -> torch.Tensor:
    """
    Runs causal convolution over a full prefill sequence, then seeds the sequence's conv_state cache.
    x_seq: (seqlen, hdim)
    weight: (hdim, kernel_size)
    """
    hdim, kernel_size = weight.shape
    seqlen = x_seq.shape[0]
    x_input = x_seq.transpose(0, 1).unsqueeze(0)  # (1, hdim, seqlen)

    if causal_conv1d_fn is not None and x_input.is_cuda:
        out, final_state = causal_conv1d_fn(x_input, weight, bias, return_final_states=True)
        seq.conv_states[layer_id] = final_state.squeeze(0)  # (hdim, kernel_size-1)
        out = out.squeeze(0).transpose(0, 1)  # (seqlen, hdim)
    else:
        out = causal_conv1d_fallback(x_input, weight, bias).squeeze(0).transpose(0, 1)
        if seqlen < kernel_size - 1:
            pad_len = (kernel_size - 1) - seqlen
            state = F.pad(x_seq.transpose(0, 1), (pad_len, 0))
        else:
            state = x_seq[-(kernel_size - 1):].transpose(0, 1)
        seq.conv_states[layer_id] = state.clone()

    return out


def step_causal_conv1d(
    layer_id: str,
    seq: Sequence,
    new_val: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
) -> torch.Tensor:
    """
    Performs constant-time O(1) causal convolution step for a single token during decode phase.
    new_val: (hdim,)
    weight: (hdim, kernel_size)
    """
    hdim, kernel_size = weight.shape

    if layer_id not in seq.conv_states:
        seq.conv_states[layer_id] = torch.zeros(hdim, kernel_size - 1, dtype=new_val.dtype, device=new_val.device)

    state = seq.conv_states[layer_id]  # (hdim, kernel_size-1)

    if causal_conv1d_update is not None and new_val.is_cuda:
        state_b = state.unsqueeze(0)  # (1, hdim, kernel_size-1)
        out = causal_conv1d_update(new_val.unsqueeze(0), state_b, weight, bias)  # (1, hdim)
        seq.conv_states[layer_id] = state_b.squeeze(0)  # updated in-place
        return out.squeeze(0)

    full_window = torch.cat([state, new_val.unsqueeze(1)], dim=1)  # (hdim, kernel_size)
    out = (full_window * weight).sum(dim=1)
    if bias is not None:
        out = out + bias
    seq.conv_states[layer_id] = full_window[:, 1:].clone()
    return out


def batch_step_causal_conv1d(
    layer_id: str,
    seqs: list,
    new_vals: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    Batched O(1) causal convolution step across all decode sequences in parallel.
    new_vals: (batch, hdim)
    weight: (hdim, kernel_size)
    Returns: (batch, hdim)
    """
    batch, hdim = new_vals.shape
    kernel_size = weight.shape[1]

    states = torch.stack([
        seq.conv_states.get(layer_id, torch.zeros(hdim, kernel_size - 1, dtype=new_vals.dtype, device=new_vals.device))
        for seq in seqs
    ])  # (batch, hdim, kernel_size-1)

    if causal_conv1d_update is not None and new_vals.is_cuda:
        out = causal_conv1d_update(new_vals, states, weight)  # (batch, hdim), states updated in-place
        for i, seq in enumerate(seqs):
            seq.conv_states[layer_id] = states[i]
    else:
        full_window = torch.cat([states, new_vals.unsqueeze(2)], dim=2)  # (batch, hdim, kernel_size)
        out = (full_window * weight.unsqueeze(0)).sum(dim=2)  # (batch, hdim)
        new_state = full_window[:, :, 1:]
        for i, seq in enumerate(seqs):
            seq.conv_states[layer_id] = new_state[i].clone()

    return out


def _infer_linear(in_f, out_f):
    return ReplicatedLinear(in_f, out_f, bias=True)


class AtmaAttention(AtmaAttnBase):

    def __init__(self, layer_idx: int, dim: int, head_dim: int = 128, kernel_size: int = 4):
        super().__init__(dim, linear_cls=_infer_linear, head_dim=head_dim, kernel_size=kernel_size)
        self.layer_idx = layer_idx
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim ** -0.5, self.num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_context()

        if context.is_prefill:
            total = x.shape[0]
            q_gate = self.q(x).view(total, self.num_heads, self.head_dim * 2)
            q_all, gate_all = torch.chunk(q_gate, 2, dim=-1)
            k_all = self.k(x).view(total, self.num_heads, self.head_dim)
            v_all = self.v(x).view(total, self.num_heads, self.head_dim)

            q_all = F.rms_norm(q_all, (self.head_dim,))
            k_all = F.rms_norm(k_all, (self.head_dim,))

            w_q = self.canon_q.weight.squeeze(1)
            w_k = self.canon_k.weight.squeeze(1)
            w_v = self.canon_v.weight.squeeze(1)

            if _fa3 is not None:
                # FA3 path: per-seq conv + kv store/gather, then single flash_attn_varlen_func call
                q_parts, k_parts, v_parts = [], [], []
                start = 0
                for i, seqlen in enumerate(context.seqlens_q):
                    seq = context.seqs[i]

                    qi = q_all[start : start + seqlen].reshape(seqlen, -1)
                    ki = k_all[start : start + seqlen].reshape(seqlen, -1)
                    vi = v_all[start : start + seqlen].reshape(seqlen, -1)

                    qi_conv = qi + prefill_causal_conv1d(f"attn_{self.layer_idx}_q", seq, qi, w_q)
                    ki_conv = ki + prefill_causal_conv1d(f"attn_{self.layer_idx}_k", seq, ki, w_k)
                    vi_conv = vi + prefill_causal_conv1d(f"attn_{self.layer_idx}_v", seq, vi, w_v)

                    qi_attn = qi_conv.view(seqlen, self.num_heads, self.head_dim)
                    ki_attn = ki_conv.view(seqlen, self.num_heads, self.head_dim)
                    vi_attn = vi_conv.view(seqlen, self.num_heads, self.head_dim)

                    slot_i = context.slot_mapping[start : start + seqlen] if (
                        context.slot_mapping is not None and context.slot_mapping.numel() > 0
                    ) else None
                    block_table_i = context.block_tables[i] if context.block_tables is not None else None

                    if slot_i is not None:
                        store_kvcache(ki_attn, vi_attn, self.attn.k_cache, self.attn.v_cache, slot_i)

                    full_seq_len = seq.num_cached_tokens + seqlen
                    if block_table_i is not None and full_seq_len > 0:
                        ki_full, vi_full = gather_kv_from_cache(
                            self.attn.k_cache, self.attn.v_cache, block_table_i, full_seq_len
                        )
                    else:
                        ki_full, vi_full = ki_attn, vi_attn

                    q_parts.append(qi_attn)
                    k_parts.append(ki_full)
                    v_parts.append(vi_full)
                    start += seqlen

                q_packed = torch.cat(q_parts, dim=0)
                k_packed = torch.cat(k_parts, dim=0)
                v_packed = torch.cat(v_parts, dim=0)

                y = _fa3.flash_attn_varlen_func(
                    q_packed, k_packed, v_packed,
                    context.cu_seqlens_q, context.cu_seqlens_k,
                    context.max_seqlen_q, context.max_seqlen_k,
                    softmax_scale=self.attn.scale,
                    causal=True,
                )  # (total_q, num_heads, head_dim)
                y = y.reshape(total, self.num_heads * self.head_dim)
                y = y * torch.sigmoid(gate_all.reshape(total, -1))
                return self.proj(y)

            else:
                # Fallback: per-sequence SDPA via self.attn
                y_parts = []
                start = 0
                for i, seqlen in enumerate(context.seqlens_q):
                    seq = context.seqs[i]

                    qi = q_all[start : start + seqlen].reshape(seqlen, -1)
                    ki = k_all[start : start + seqlen].reshape(seqlen, -1)
                    vi = v_all[start : start + seqlen].reshape(seqlen, -1)
                    gatei = gate_all[start : start + seqlen].reshape(seqlen, -1)

                    qi_conv = qi + prefill_causal_conv1d(f"attn_{self.layer_idx}_q", seq, qi, w_q)
                    ki_conv = ki + prefill_causal_conv1d(f"attn_{self.layer_idx}_k", seq, ki, w_k)
                    vi_conv = vi + prefill_causal_conv1d(f"attn_{self.layer_idx}_v", seq, vi, w_v)

                    qi_attn = qi_conv.view(seqlen, self.num_heads, self.head_dim)
                    ki_attn = ki_conv.view(seqlen, self.num_heads, self.head_dim)
                    vi_attn = vi_conv.view(seqlen, self.num_heads, self.head_dim)

                    slot_i = context.slot_mapping[start : start + seqlen] if (
                        context.slot_mapping is not None and context.slot_mapping.numel() > 0
                    ) else None
                    block_table_i = context.block_tables[i] if context.block_tables is not None else None
                    yi = self.attn(
                        qi_attn, ki_attn, vi_attn,
                        slot_mapping=slot_i,
                        block_table=block_table_i,
                        seq_len=seq.num_cached_tokens + seqlen,
                    )
                    yi = yi.reshape(seqlen, -1) * torch.sigmoid(gatei)
                    y_parts.append(yi)
                    start += seqlen

                return self.proj(torch.cat(y_parts, dim=0))

        else:
            batch_size = x.shape[0]

            q_gate = self.q(x).view(batch_size, self.num_heads, self.head_dim * 2)
            q_all, gate_all = torch.chunk(q_gate, 2, dim=-1)
            k_all = self.k(x).view(batch_size, self.num_heads, self.head_dim)
            v_all = self.v(x).view(batch_size, self.num_heads, self.head_dim)

            q_all = F.rms_norm(q_all, (self.head_dim,))
            k_all = F.rms_norm(k_all, (self.head_dim,))

            q_flat = q_all.reshape(batch_size, -1)
            k_flat = k_all.reshape(batch_size, -1)
            v_flat = v_all.reshape(batch_size, -1)

            w_q = self.canon_q.weight.squeeze(1)
            w_k = self.canon_k.weight.squeeze(1)
            w_v = self.canon_v.weight.squeeze(1)

            q_conv = q_flat + batch_step_causal_conv1d(f"attn_{self.layer_idx}_q", context.seqs, q_flat, w_q)
            k_conv = k_flat + batch_step_causal_conv1d(f"attn_{self.layer_idx}_k", context.seqs, k_flat, w_k)
            v_conv = v_flat + batch_step_causal_conv1d(f"attn_{self.layer_idx}_v", context.seqs, v_flat, w_v)

            if _fa3 is not None and self.attn.k_cache.numel() > 0:
                # FA3 path: flash_attn_with_kvcache handles kv write + paged attention in one call
                q_b = q_conv.view(batch_size, 1, self.num_heads, self.head_dim)
                k_b = k_conv.view(batch_size, 1, self.num_heads, self.head_dim)
                v_b = v_conv.view(batch_size, 1, self.num_heads, self.head_dim)
                # cache_seqlens = tokens already in cache before the new decode token
                cache_seqlens = context.context_lens - 1

                y = _fa3.flash_attn_with_kvcache(
                    q_b,
                    self.attn.k_cache,
                    self.attn.v_cache,
                    k=k_b,
                    v=v_b,
                    cache_seqlens=cache_seqlens,
                    block_table=context.block_tables,
                    softmax_scale=self.attn.scale,
                    causal=False,
                )  # (batch, 1, num_heads, head_dim)
                y = y.squeeze(1).reshape(batch_size, -1)  # (batch, num_heads * head_dim)
                return self.proj(y * torch.sigmoid(gate_all.reshape(batch_size, -1)))

            else:
                # Fallback: per-sequence SDPA via self.attn
                q_attn = q_conv.view(batch_size, self.num_heads, self.head_dim)
                k_attn = k_conv.view(batch_size, self.num_heads, self.head_dim)
                v_attn = v_conv.view(batch_size, self.num_heads, self.head_dim)

                y_parts = []
                for i in range(batch_size):
                    seq = context.seqs[i]
                    slot_i = context.slot_mapping[i : i + 1] if context.slot_mapping is not None else None
                    block_table_i = context.block_tables[i] if context.block_tables is not None else None
                    seq_len_i = context.context_lens_list[i] if context.context_lens_list else len(seq)

                    yi = self.attn(
                        q_attn[i], k_attn[i], v_attn[i],
                        slot_mapping=slot_i,
                        block_table=block_table_i,
                        seq_len=seq_len_i,
                    )
                    yi = yi.reshape(-1) * torch.sigmoid(gate_all[i].reshape(-1))
                    y_parts.append(yi)

                return self.proj(torch.stack(y_parts, dim=0))


class AtmaLFM2Conv(AtmaConvBase):

    def __init__(self, layer_idx: int, dim: int, kernel_size: int = 3):
        super().__init__(dim, linear_cls=_infer_linear, kernel_size=kernel_size)
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_context()

        if context.is_prefill:
            projected = self.in_proj(x)  # (total_tokens, 3*hidden_size)
            B_all, C_all, x_proj_all = projected.chunk(3, dim=-1)
            x_gated_all = B_all * x_proj_all

            w_conv = self.conv.weight.squeeze(1)

            # Per-sequence causal conv (variable lengths and state saving require seq loop)
            y_parts = []
            start = 0
            for i, seqlen in enumerate(context.seqlens_q):
                seq = context.seqs[i]

                x_gated_i = x_gated_all[start : start + seqlen]
                C_i = C_all[start : start + seqlen]

                x_conv_i = prefill_causal_conv1d(f"conv_{self.layer_idx}_gated", seq, x_gated_i, w_conv)
                y_parts.append(C_i * x_conv_i)
                start += seqlen

            return self.out_proj(torch.cat(y_parts, dim=0))

        else:
            projected = self.in_proj(x)  # (batch, 3*hidden_size)
            B_all, C_all, x_proj_all = projected.chunk(3, dim=-1)
            x_gated = B_all * x_proj_all

            w_conv = self.conv.weight.squeeze(1)
            x_conv = batch_step_causal_conv1d(
                f"conv_{self.layer_idx}_gated", context.seqs, x_gated, w_conv
            )

            return self.out_proj(C_all * x_conv)


class AtmaDecoderBlock(nn.Module):

    def __init__(
        self,
        layer_idx: int,
        dim: int,
        attention: bool = True,
        head_dim: int = 128,
        attn_kernel_size: int = 4,
        conv_kernel_size: int = 3,
    ):
        super().__init__()
        self.attn = (
            AtmaAttention(layer_idx, dim, head_dim=head_dim, kernel_size=attn_kernel_size)
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
                attn_kernel_size=config.attn_kernel_size,
                conv_kernel_size=config.conv_kernel_size,
            )
            for i in range(config.num_hidden_layers)
        ])
        self.proj = ParallelLMHead(config.vocab_size, config.hidden_size, bias=True)
        self.norm = RMSNorm(config.hidden_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x_normed = self.norm(x)
        logits = self.proj(x_normed)
        logits = 15.0 * logits * (logits.square() + 225.0).rsqrt()
        return logits
