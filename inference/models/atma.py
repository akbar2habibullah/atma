import torch
from torch import nn
import torch.nn.functional as F

from inference.layers.layernorm import RMSNorm
from inference.layers.linear import ReplicatedLinear, RowParallelLinear
from inference.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from inference.layers.attention import Attention
from inference.utils.context import get_context
from inference.engine.sequence import Sequence


def causal_conv1d_fallback(x: torch.Tensor, weight: torch.Tensor, bias = None) -> torch.Tensor:
    """
    Pure PyTorch causal 1D depthwise convolution fallback.
    x: (B, hdim, T)
    weight: (hdim, kernel_size)
    bias: (hdim,) or None
    """
    kernel_size = weight.shape[1]
    # Pad left with kernel_size - 1 zeros to maintain causality
    x_padded = F.pad(x, (kernel_size - 1, 0))
    # Depthwise Conv1d (groups = hdim)
    w = weight.unsqueeze(1) # (hdim, 1, kernel_size)
    out = F.conv1d(x_padded, w, bias, stride=1, padding=0, groups=x.shape[1])
    return out


def prefill_causal_conv1d(
    layer_id: str,
    seq: Sequence,
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    bias = None,
) -> torch.Tensor:
    """
    Runs causal convolution over a full prefill sequence, then seeds the sequence's conv_state cache.
    x_seq: (seqlen, hdim)
    weight: (hdim, kernel_size)
    """
    hdim, kernel_size = weight.shape
    seqlen = x_seq.shape[0]

    # Input: (1, hdim, seqlen)
    x_input = x_seq.transpose(0, 1).unsqueeze(0)
    out = causal_conv1d_fallback(x_input, weight, bias)
    out = out.squeeze(0).transpose(0, 1) # (seqlen, hdim)

    # Save the final (kernel_size - 1) states as seed for decode steps
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
    bias = None,
) -> torch.Tensor:
    """
    Performs constant-time O(1) causal convolution step for a single token during decode phase.
    new_val: (hdim,)
    weight: (hdim, kernel_size)
    """
    hdim, kernel_size = weight.shape

    if layer_id not in seq.conv_states:
        seq.conv_states[layer_id] = torch.zeros(hdim, kernel_size - 1, dtype=new_val.dtype, device=new_val.device)

    state = seq.conv_states[layer_id]

    # Concat past state with current token to get the full convolution window
    full_window = torch.cat([state, new_val.unsqueeze(1)], dim=1) # (hdim, kernel_size)

    # Depthwise convolution dot product
    out = (full_window * weight).sum(dim=1)
    if bias is not None:
        out = out + bias

    # Shift history window by 1 step
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

    full_window = torch.cat([states, new_vals.unsqueeze(2)], dim=2)  # (batch, hdim, kernel_size)
    out = (full_window * weight.unsqueeze(0)).sum(dim=2)  # (batch, hdim)

    new_state = full_window[:, :, 1:]
    for i, seq in enumerate(seqs):
        seq.conv_states[layer_id] = new_state[i].clone()

    return out


class AtmaAttention(nn.Module):

    def __init__(self, layer_idx: int, dim: int, head_dim: int = 128, kernel_size: int = 4):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.hdim = hdim
        self.kernel_size = kernel_size

        self.q = ReplicatedLinear(dim, hdim * 2, bias=True)
        self.k = ReplicatedLinear(dim, hdim, bias=True)
        self.v = ReplicatedLinear(dim, hdim, bias=True)

        self.canon_q = nn.Conv1d(hdim, hdim, kernel_size=kernel_size, padding=kernel_size - 1, groups=hdim, bias=False)
        self.canon_k = nn.Conv1d(hdim, hdim, kernel_size=kernel_size, padding=kernel_size - 1, groups=hdim, bias=False)
        self.canon_v = nn.Conv1d(hdim, hdim, kernel_size=kernel_size, padding=kernel_size - 1, groups=hdim, bias=False)

        self.proj = ReplicatedLinear(hdim, dim, bias=True)
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim ** -0.5, self.num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_context()

        if context.is_prefill:
            # Linear projections and QK-norm over ALL packed tokens at once
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

            # Per-sequence causal conv + attention (variable lengths and paged KV require seq loop)
            y_parts = []
            start = 0
            for i in range(len(context.cu_seqlens_q) - 1):
                seqlen = context.cu_seqlens_q[i + 1].item() - context.cu_seqlens_q[i].item()
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

                slot_i = context.slot_mapping[start : start + seqlen] if (context.slot_mapping is not None and context.slot_mapping.numel() > 0) else None
                block_table_i = context.block_tables[i] if context.block_tables is not None else None
                yi = self.attn(
                    qi_attn,
                    ki_attn,
                    vi_attn,
                    slot_mapping=slot_i,
                    block_table=block_table_i,
                    seq_len=seq.num_cached_tokens + seqlen,
                )
                yi = yi.reshape(seqlen, -1) * torch.sigmoid(gatei)
                y_parts.append(yi)
                start += seqlen

            # Output projection batched over all sequences at once
            return self.proj(torch.cat(y_parts, dim=0))

        else:
            # Decode: linear projections, QK-norm, and causal conv math all batched across sequences
            batch_size = x.shape[0]

            q_gate = self.q(x).view(batch_size, self.num_heads, self.head_dim * 2)
            q_all, gate_all = torch.chunk(q_gate, 2, dim=-1)
            k_all = self.k(x).view(batch_size, self.num_heads, self.head_dim)
            v_all = self.v(x).view(batch_size, self.num_heads, self.head_dim)

            q_all = F.rms_norm(q_all, (self.head_dim,))
            k_all = F.rms_norm(k_all, (self.head_dim,))

            q_flat = q_all.reshape(batch_size, -1)  # (batch, hdim)
            k_flat = k_all.reshape(batch_size, -1)
            v_flat = v_all.reshape(batch_size, -1)

            w_q = self.canon_q.weight.squeeze(1)
            w_k = self.canon_k.weight.squeeze(1)
            w_v = self.canon_v.weight.squeeze(1)

            q_conv = q_flat + batch_step_causal_conv1d(f"attn_{self.layer_idx}_q", context.seqs, q_flat, w_q)
            k_conv = k_flat + batch_step_causal_conv1d(f"attn_{self.layer_idx}_k", context.seqs, k_flat, w_k)
            v_conv = v_flat + batch_step_causal_conv1d(f"attn_{self.layer_idx}_v", context.seqs, v_flat, w_v)

            q_attn = q_conv.view(batch_size, self.num_heads, self.head_dim)
            k_attn = k_conv.view(batch_size, self.num_heads, self.head_dim)
            v_attn = v_conv.view(batch_size, self.num_heads, self.head_dim)

            # Per-sequence attention loop: paged KV cache needs per-seq block tables
            y_parts = []
            for i in range(batch_size):
                seq = context.seqs[i]
                slot_i = context.slot_mapping[i : i + 1] if context.slot_mapping is not None else None
                block_table_i = context.block_tables[i] if context.block_tables is not None else None
                seq_len_i = context.context_lens[i].item() if context.context_lens is not None else len(seq)

                yi = self.attn(
                    q_attn[i],
                    k_attn[i],
                    v_attn[i],
                    slot_mapping=slot_i,
                    block_table=block_table_i,
                    seq_len=seq_len_i,
                )
                yi = yi.reshape(-1) * torch.sigmoid(gate_all[i].reshape(-1))
                y_parts.append(yi)

            return self.proj(torch.stack(y_parts, dim=0))


class AtmaLFM2Conv(nn.Module):

    def __init__(self, layer_idx: int, dim: int, kernel_size: int = 3):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = dim
        self.kernel_size = kernel_size

        self.in_proj = ReplicatedLinear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.conv = nn.Conv1d(
            self.hidden_size,
            self.hidden_size,
            kernel_size=self.kernel_size,
            padding=self.kernel_size - 1,
            groups=self.hidden_size,
            bias=False,
        )
        self.out_proj = ReplicatedLinear(self.hidden_size, self.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_context()

        if context.is_prefill:
            # Input projection and first gating over ALL packed tokens at once
            projected = self.in_proj(x)  # (total_tokens, 3*hidden_size)
            B_all, C_all, x_proj_all = projected.chunk(3, dim=-1)
            x_gated_all = B_all * x_proj_all

            w_conv = self.conv.weight.squeeze(1)

            # Per-sequence causal conv (variable lengths and state saving require seq loop)
            y_parts = []
            start = 0
            for i in range(len(context.cu_seqlens_q) - 1):
                seqlen = context.cu_seqlens_q[i + 1].item() - context.cu_seqlens_q[i].item()
                seq = context.seqs[i]

                x_gated_i = x_gated_all[start : start + seqlen]
                C_i = C_all[start : start + seqlen]

                x_conv_i = prefill_causal_conv1d(f"conv_{self.layer_idx}_gated", seq, x_gated_i, w_conv)
                y_parts.append(C_i * x_conv_i)
                start += seqlen

            # Output projection batched over all sequences at once
            return self.out_proj(torch.cat(y_parts, dim=0))

        else:
            # Decode: all ops batched across sequences
            projected = self.in_proj(x)  # (batch, 3*hidden_size)
            B_all, C_all, x_proj_all = projected.chunk(3, dim=-1)
            x_gated = B_all * x_proj_all  # (batch, hidden_size)

            w_conv = self.conv.weight.squeeze(1)
            x_conv = batch_step_causal_conv1d(
                f"conv_{self.layer_idx}_gated", context.seqs, x_gated, w_conv
            )

            return self.out_proj(C_all * x_conv)


class AtmaMLP(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = ReplicatedLinear(dim, hdim * 2, bias=True)
        self.proj = ReplicatedLinear(hdim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gate = self.fc(x)
        x, gate = torch.chunk(x_gate, 2, dim=-1)
        # Activation function identical to training loop: gate * relu(x)^2
        x = gate * x.relu().square()
        return self.proj(x)


class AtmaDecoderBlock(nn.Module):

    def __init__(self, layer_idx: int, dim: int, attention: bool = True):
        super().__init__()
        self.attn = AtmaAttention(layer_idx, dim) if attention else AtmaLFM2Conv(layer_idx, dim)
        self.mlp = AtmaMLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # attention/convolution branch
        normed_x = self.norm1(x)
        x = x + self.attn(normed_x)

        # MLP branch
        normed_x2 = self.norm2(x)
        x = x + self.mlp(normed_x2)
        return x


class Atma(nn.Module):

    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        # Every 4th layer (specifically when i % 4 == 2) has self-attention, other layers have LFM2Conv
        self.blocks = nn.ModuleList([
            AtmaDecoderBlock(i, model_dim, attention=True) if i % 4 == 2 
            else AtmaDecoderBlock(i, model_dim, attention=False) 
            for i in range(num_layers)
        ])
        self.proj = ParallelLMHead(vocab_size, model_dim, bias=True)
        self.norm = RMSNorm(model_dim)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        # input_ids: (total_tokens,) flat tensor in prefill, or (batch_size,) in decode
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
            
        x_normed = self.norm(x)
        logits = self.proj(x_normed)
        
        # Apply the exact training-loop logit clamping and scaling
        # logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        logits = 15.0 * logits * (logits.square() + 225.0).rsqrt()
        return logits
