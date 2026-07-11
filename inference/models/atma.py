import torch
from torch import nn
import torch.nn.functional as F

from model.config import AtmaConfig
from model.layers import RMSNorm, MLP
from model.blocks import AtmaConvBase, AtmaAttnBase, TitansMemory, gated_delta_chunked, polar_reduce
from inference.layers.linear import ReplicatedLinear
from inference.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from inference.layers.attention import Attention, store_kvcache
from inference.utils.context import get_context


try:
    from kernels import get_kernel as _get_kernel
    _conv_mod = _get_kernel("kernels-community/causal-conv1d")
    causal_conv1d_fn = _conv_mod.causal_conv1d_fn
    causal_conv1d_update = _conv_mod.causal_conv1d_update
except Exception:
    causal_conv1d_fn = None
    causal_conv1d_update = None


# FlashAttention-style polar-attention kernels, forward-only (no autograd), for inference.
# Prefill runs polar_attention_fwd per sequence (is_causal=True for a fresh prefill;
# is_causal=False with offset n_keys + gathered prefix K/V for a chunked-prefill
# continuation). Decode runs polar_attention_decode, which reads K/V DIRECTLY from the
# paged cache via block_tables/context_lens (no gather, fixed launch shape -> CUDA-graph
# capturable). Both fall back to the materialized polar_reduce on CPU. The sliding window
# (attn_window) and the Titans MAG memory branch (mem_enabled) match model/reference.py.
try:
    from kernel.polar_triton import polar_attention_fwd, polar_attention_decode, HAS_TRITON  # noqa: F401
except Exception:
    polar_attention_fwd = None
    polar_attention_decode = None
    HAS_TRITON = False

# Titans memory kernels:
#  - prefill: flash-linear-attention's fused chunk_gated_delta_rule (CUDA/Triton), with
#    initial_state / output_final_state carrying the per-sequence state across chunks.
#  - decode: our fused single-step kernel (kernel/gated_delta_triton.py), which reads and
#    writes the slot-indexed state table IN PLACE — one state read + one write per step,
#    vs 3x traffic for a gather -> kernel -> scatter sequence. Exact fp32, graph-safe.
# Falls back to the pure-PyTorch gated_delta_chunked / explicit step on CPU or when the
# kernels are unavailable.
# NOTE on layout: FLA states are [N, H, K, V]; the torch gated_delta_chunked state is
# (B, H, dv, dk) — the transpose. The mem state tables store the FLA layout (so the GPU
# hot paths run transpose-free); the torch fallback transposes at its boundary. Validate
# the bridge on GPU with verify_fla.py.
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    _HAS_FLA = True
except Exception:
    chunk_gated_delta_rule = None
    _HAS_FLA = False

try:
    from kernel.gated_delta_triton import gated_delta_decode_step
    from kernel.gated_delta_triton import HAS_TRITON as _HAS_STEP_KERNEL
except Exception:
    gated_delta_decode_step = None
    _HAS_STEP_KERNEL = False

try:
    from kernel.causal_conv1d_triton import causal_conv1d_decode_step
    from kernel.causal_conv1d_triton import HAS_TRITON as _HAS_CONV_STEP_KERNEL
except Exception:
    causal_conv1d_decode_step = None
    _HAS_CONV_STEP_KERNEL = False

try:
    from kernel.inference_ops_triton import squared_relu_gate, softcap_logits
    from kernel.inference_ops_triton import HAS_TRITON as _HAS_INFERENCE_OPS
except Exception:
    squared_relu_gate = None
    softcap_logits = None
    _HAS_INFERENCE_OPS = False


def _infer_linear(in_f, out_f):
    return ReplicatedLinear(in_f, out_f, bias=True)


class InferenceMLP(MLP):
    """MLP with a forward-only fused activation for CUDA inference."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gate = self.fc(x)
        x, gate = torch.chunk(x_gate, 2, dim=-1)
        if squared_relu_gate is not None and _HAS_INFERENCE_OPS and x.is_cuda:
            x = squared_relu_gate(x, gate)
        else:
            x = gate * x.relu().square()
        return self.proj(x)


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
    """Run causal conv over a prefill chunk; save final state to the GPU conv state table.

    A fresh prefill (seq.num_cached_tokens == 0) left-pads with zeros. A chunked-prefill
    continuation left-pads with the conv state saved by the previous chunk, so the first
    ks-1 outputs see the true left context (the fused CUDA kernel is only used for the
    fresh case; the continuation runs the plain depthwise conv, which is rare and cheap)."""
    seqlen, hdim = x_seq.shape
    kernel_size = weight.shape[1]
    cached = getattr(seq, "num_cached_tokens", 0)
    x_input = x_seq.transpose(0, 1).unsqueeze(0)  # (1, hdim, seqlen)

    if causal_conv1d_fn is not None and x_input.is_cuda and cached == 0:
        out, final_state = causal_conv1d_fn(x_input, weight, bias, return_final_states=True)
        conv_state_tables[layer_id][seq.seq_slot] = final_state.squeeze(0)  # (hdim, ks-1)
        out = out.squeeze(0).transpose(0, 1)  # (seqlen, hdim)
    else:
        if cached > 0:
            state_in = conv_state_tables[layer_id][seq.seq_slot].to(x_input.dtype)  # (hdim, ks-1)
            x_padded = torch.cat([state_in.unsqueeze(0), x_input], dim=2)
        else:
            x_padded = F.pad(x_input, (kernel_size - 1, 0))
        w = weight.unsqueeze(1)
        out = F.conv1d(x_padded, w, bias, stride=1, padding=0, groups=hdim)
        out = out.squeeze(0).transpose(0, 1)
        # final state = last ks-1 columns of the padded stream (covers seqlen < ks-1 too)
        conv_state_tables[layer_id][seq.seq_slot] = x_padded[0, :, -(kernel_size - 1):].contiguous()

    return out


def prefill_causal_conv1d_dense(
    layer_id: str,
    seq_slots: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias,
    conv_state_tables: dict,
) -> torch.Tensor:
    """Batched fresh-prefill depthwise convolution and final-state scatter."""
    _, _, channels = x.shape
    kernel_size = weight.shape[1]
    x_input = x.transpose(1, 2).contiguous()
    x_padded = F.pad(x_input, (kernel_size - 1, 0))
    out = F.conv1d(x_padded, weight.unsqueeze(1), bias, groups=channels)
    conv_state_tables[layer_id][seq_slots] = x_padded[:, :, -(kernel_size - 1):]
    return out.transpose(1, 2)


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
    if (bias is None and causal_conv1d_decode_step is not None
            and _HAS_CONV_STEP_KERNEL and new_vals.is_cuda):
        return causal_conv1d_decode_step(
            new_vals, weight, seq_slots, conv_state_tables[layer_id]
        )
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

    def __init__(self, layer_idx: int, dim: int, head_dim: int = 128, num_kv_heads: int = None, kernel_size: int = 4,
                 window: int = None, mem_enabled: bool = False, mem_chunk: int = 64,
                 mem_gamma_bias: float = 3.9, mem_beta_bias: float = 0.0, mem_kernel: str = "auto"):
        super().__init__(dim, linear_cls=_infer_linear, head_dim=head_dim, num_kv_heads=num_kv_heads, kernel_size=kernel_size)
        self.layer_idx = layer_idx
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim ** -0.5, self.num_kv_heads)
        # Polar-attention parameters (per head). Replaces softmax SDPA with the
        # length-invariant direction + bounded-count reduction (kernel/polar_triton.py).
        H, dk = self.num_heads, self.head_dim
        self.window = window                                       # causal sliding window
        self.mu_proj = _infer_linear(H, dim)                       # count channel -> residual
        self.v_null = nn.Parameter(torch.zeros(H, dk))
        self.null_base = nn.Parameter(torch.full((H,), 2.0))
        self.null_slope_raw = nn.Parameter(torch.full((H,), 0.5))
        self.len_gain_raw = nn.Parameter(torch.full((H,), -1.0))
        self.mag_beta_raw = nn.Parameter(torch.full((H,), -1.5))
        # Titans MAG memory branch (additive 3rd channel, matches model/reference.py).
        # Inference never calls mem.forward(): the per-seq recurrent state S lives in the
        # mem state table (context.conv_state_tables["mem_{layer_idx}"], FLA [K,V] layout)
        # and is advanced by _mem_prefill (FLA chunked kernel / torch chunked scan) and
        # _mem_decode (FLA fused recurrent kernel / torch step), both state-carrying and
        # gather/scattered by seq slot. mem_kernel: "auto" | "fla" | "torch".
        self.mem = (TitansMemory(dim, H, dk, _infer_linear, chunk=mem_chunk,
                                 gamma_bias=mem_gamma_bias, beta_bias=mem_beta_bias, kernel=mem_kernel)
                    if mem_enabled else None)

    def _polar_params(self):
        return dict(v_null=self.v_null, null_base=self.null_base, null_slope_raw=self.null_slope_raw,
                    len_gain_raw=self.len_gain_raw, mag_beta_raw=self.mag_beta_raw)

    def _polar(self, q_t, k_t, v_t, n_keys, is_causal):
        """Polar reduction. q_t,k_t,v_t: (B, H, T, dk) with KV heads expanded to H.
        n_keys: (Tq,) valid-key count per query (absolute, so a chunked-prefill
        continuation passes start+1..start+Tq with is_causal=False).
        Returns c (B,H,Tq,dk) and mag (B,H,Tq)."""
        if polar_attention_fwd is not None and q_t.is_cuda:
            return polar_attention_fwd(q_t, k_t, v_t, n_keys, is_causal=is_causal,
                                       window=self.window, **self._polar_params())
        Tk = k_t.shape[2]
        sigma = torch.matmul(q_t.float(), k_t.float().transpose(-2, -1)) / (self.head_dim ** 0.5)
        ki = torch.arange(Tk, device=q_t.device)
        invalid = ki[None, None, None, :] >= n_keys.view(1, 1, -1, 1)               # future
        n_temp = n_keys
        if self.window is not None:
            invalid = invalid | (ki[None, None, None, :] < (n_keys.view(1, 1, -1, 1) - self.window))
            n_temp = torch.minimum(n_keys, n_keys.new_tensor(float(self.window)))
        sigma = sigma.masked_fill(invalid, float("-inf"))
        return polar_reduce(sigma, v_t, n_temp, **self._polar_params())

    def _combine(self, c, mag, gate, n_tokens):
        """content = W_o(reshape(c) * sigmoid(gate)) + W_mu(mag).  c:(B,H,T,dk) mag:(B,H,T)."""
        H, dk = self.num_heads, self.head_dim
        c_flat = c.transpose(1, 2).reshape(n_tokens, H * dk)
        content = self.proj(c_flat * torch.sigmoid(gate.reshape(n_tokens, -1)))
        count = self.mu_proj(mag.transpose(1, 2).reshape(n_tokens, H))
        return content + count

    # ------------------------------------------------------------------
    # Titans MAG memory branch (state-carrying inference forms of
    # model.blocks.TitansMemory.forward; weights are shared via self.mem)
    # ------------------------------------------------------------------

    def _mem_use_fla(self, t: torch.Tensor) -> bool:
        return _HAS_FLA and t.is_cuda and self.mem.kernel in ("auto", "fla")

    def _mem_prefill(self, seq, x_seq, q_t, k_t, v_t, mem_state_table) -> torch.Tensor:
        """x_seq: (T, D); q_t,k_t,v_t: (1, H, T, dk) fresh-chunk tensors (KV heads
        expanded). Reads the running state S from the per-seq table (zeros for a fresh
        sequence, carried over for a chunked-prefill continuation), runs the chunkwise
        gated-delta scan, writes the final state back. Returns (T, D).

        GPU: FLA's fused chunk_gated_delta_rule (initial_state/output_final_state carry
        the per-seq state; in-kernel L2-norm; scale=1.0 washes out post-RMSNorm).
        CPU/fallback: the validated torch gated_delta_chunked, whose (B,H,dv,dk) state is
        the transpose of the table's FLA [K,V] layout."""
        mem = self.mem
        T = x_seq.shape[0]
        H, dk = self.num_heads, self.head_dim
        g_logit = mem.w_gamma(x_seq).float() + mem.gamma_bias            # (T, H)
        b_logit = mem.w_beta(x_seq).float() + mem.beta_bias
        S0 = mem_state_table[seq.seq_slot].unsqueeze(0)                  # (1, H, dk, dv) fp32, FLA layout

        if self._mem_use_fla(x_seq):
            g = F.logsigmoid(g_logit).view(1, T, H)                      # log-decay (<=0)
            beta = torch.sigmoid(b_logit).view(1, T, H)
            q = q_t.transpose(1, 2).contiguous()                         # (1, T, H, dk)
            k = k_t.transpose(1, 2).contiguous()
            v = v_t.transpose(1, 2).contiguous()
            r, S = chunk_gated_delta_rule(q=q, k=k, v=v, g=g.contiguous(), beta=beta.contiguous(),
                                          scale=1.0, initial_state=S0,
                                          output_final_state=True, use_qk_l2norm_in_kernel=True)
            mem_state_table[seq.seq_slot] = S.squeeze(0)
            r = F.rms_norm(r, (dk,))                                     # (1, T, H, dk)
        else:
            gamma = torch.sigmoid(g_logit).transpose(0, 1).unsqueeze(0)  # (1, H, T)
            beta = torch.sigmoid(b_logit).transpose(0, 1).unsqueeze(0)
            qn = F.normalize(q_t.float(), dim=-1)                        # unit keys/queries
            kn = F.normalize(k_t.float(), dim=-1)
            r, S = gated_delta_chunked(qn, kn, v_t.float(), gamma, beta, chunk=mem.chunk,
                                       S0=S0.transpose(-1, -2))          # -> torch (1,H,dv,dk)
            mem_state_table[seq.seq_slot] = S.squeeze(0).transpose(-1, -2)
            r = F.rms_norm(r.transpose(1, 2), (dk,))                     # (1, T, H, dk)

        r_flat = r.reshape(T, H * dk).to(x_seq.dtype)
        return mem.proj(r_flat * torch.sigmoid(mem.gate(x_seq)))

    def _mem_prefill_dense(self, x, q_t, k_t, v_t, seq_slots, mem_state_table) -> torch.Tensor:
        """Batched equivalent of _mem_prefill for fresh equal-length prompts."""
        mem = self.mem
        B, T, _ = x.shape
        H, dk = self.num_heads, self.head_dim
        g_logit = mem.w_gamma(x).float() + mem.gamma_bias
        b_logit = mem.w_beta(x).float() + mem.beta_bias
        S0 = mem_state_table[seq_slots]

        if self._mem_use_fla(x):
            g = F.logsigmoid(g_logit).contiguous()
            beta = torch.sigmoid(b_logit).contiguous()
            q = q_t.transpose(1, 2).contiguous()
            k = k_t.transpose(1, 2).contiguous()
            v = v_t.transpose(1, 2).contiguous()
            r, S = chunk_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta, scale=1.0,
                initial_state=S0, output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            mem_state_table[seq_slots] = S
            r = F.rms_norm(r, (dk,))
        else:
            gamma = torch.sigmoid(g_logit).transpose(1, 2)
            beta = torch.sigmoid(b_logit).transpose(1, 2)
            qn = F.normalize(q_t.float(), dim=-1)
            kn = F.normalize(k_t.float(), dim=-1)
            r, S = gated_delta_chunked(
                qn, kn, v_t.float(), gamma, beta, chunk=mem.chunk,
                S0=S0.transpose(-1, -2),
            )
            mem_state_table[seq_slots] = S.transpose(-1, -2)
            r = F.rms_norm(r.transpose(1, 2), (dk,))

        r_flat = r.reshape(B * T, H * dk).to(x.dtype)
        x_flat = x.reshape(B * T, -1)
        return mem.proj(r_flat * torch.sigmoid(mem.gate(x_flat)))

    def _mem_decode(self, x, q_t, k_t, v_t, seq_slots, mem_state_table) -> torch.Tensor:
        """x: (B, D); q_t,k_t,v_t: (B, H, dk) current-token tensors (KV heads expanded).
        Single gated-delta step, batched over sequences (CUDA-graph compatible).

        GPU: the fused step kernel (kernel/gated_delta_triton.py) — reads and writes the
        slot-indexed state table IN PLACE in its native [K,V] layout, so the (large) fp32
        state moves once per step instead of gather + kernel + scatter.
        CPU/fallback: the explicit N=1 step of gated_delta_chunked (decay-first, undecayed
        write, self-inclusive readout M_t q_t), transposing at the layout boundary."""
        mem = self.mem
        B = x.shape[0]
        H, dk = self.num_heads, self.head_dim
        g_logit = mem.w_gamma(x).float() + mem.gamma_bias                # (B, H)
        b_logit = mem.w_beta(x).float() + mem.beta_bias
        gamma = torch.sigmoid(g_logit)
        beta = torch.sigmoid(b_logit)

        if gated_delta_decode_step is not None and _HAS_STEP_KERNEL and x.is_cuda:
            r = gated_delta_decode_step(q_t, k_t, v_t, gamma, beta,
                                        mem_state_table, seq_slots)      # (B, H, dk) fp32
        else:
            S = mem_state_table[seq_slots]                               # (B, H, dk, dv) gather
            qn = F.normalize(q_t.float(), dim=-1)                        # (B, H, dk)
            kn = F.normalize(k_t.float(), dim=-1)
            St = S.transpose(-1, -2)                                     # -> torch (B, H, dv, dk)
            Sd = gamma[..., None, None] * St                             # decay first
            pred = torch.einsum("bhvk,bhk->bhv", Sd, kn)
            u = beta[..., None] * (v_t.float() - pred)                   # undecayed write
            S_new = Sd + u.unsqueeze(-1) * kn.unsqueeze(-2)              # (B, H, dv, dk)
            mem_state_table[seq_slots] = S_new.transpose(-1, -2)         # scatter (FLA layout)
            r = torch.einsum("bhvk,bhk->bhv", S_new, qn)                 # readout M_t q_t

        r = F.rms_norm(r, (dk,))
        r_flat = r.reshape(B, H * dk).to(x.dtype)
        return mem.proj(r_flat * torch.sigmoid(mem.gate(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_context()
        w_q = self.canon_q.weight.squeeze(1)  # (hdim, kernel_size)
        w_k = self.canon_k.weight.squeeze(1)
        w_v = self.canon_v.weight.squeeze(1)
        H, dk = self.num_heads, self.head_dim
        groups = H // self.num_kv_heads
        mem_table = (context.conv_state_tables[f"mem_{self.layer_idx}"]
                     if self.mem is not None else None)

        if context.is_prefill:
            total = x.shape[0]
            q_gate = self.q(x).view(total, self.num_heads, self.head_dim * 2)
            q_all, gate_all = torch.chunk(q_gate, 2, dim=-1)
            k_all = self.k(x).view(total, self.num_kv_heads, self.head_dim)
            v_all = self.v(x).view(total, self.num_kv_heads, self.head_dim)

            q_all = F.rms_norm(q_all, (self.head_dim,))
            k_all = F.rms_norm(k_all, (self.head_dim,))

            if context.dense_prefill:
                B, T = context.dense_batch_size, context.dense_seq_len
                slots = context.seq_slots
                q_flat = q_all.reshape(B, T, -1)
                k_flat = k_all.reshape(B, T, -1)
                v_flat = v_all.reshape(B, T, -1)
                q_conv = q_flat + prefill_causal_conv1d_dense(
                    f"attn_{self.layer_idx}_q", slots, q_flat, w_q, None, context.conv_state_tables)
                k_conv = k_flat + prefill_causal_conv1d_dense(
                    f"attn_{self.layer_idx}_k", slots, k_flat, w_k, None, context.conv_state_tables)
                v_conv = v_flat + prefill_causal_conv1d_dense(
                    f"attn_{self.layer_idx}_v", slots, v_flat, w_v, None, context.conv_state_tables)

                k_packed = k_conv.reshape(total, self.num_kv_heads, dk)
                v_packed = v_conv.reshape(total, self.num_kv_heads, dk)
                if self.attn.k_cache.numel() > 0 and context.slot_mapping.numel() > 0:
                    store_kvcache(k_packed, v_packed, self.attn.k_cache,
                                  self.attn.v_cache, context.slot_mapping)

                qh = q_conv.view(B, T, H, dk).transpose(1, 2).contiguous()
                kh = (k_conv.view(B, T, self.num_kv_heads, dk)
                      .repeat_interleave(groups, dim=2).transpose(1, 2).contiguous())
                vh = (v_conv.view(B, T, self.num_kv_heads, dk)
                      .repeat_interleave(groups, dim=2).transpose(1, 2).contiguous())
                n_keys = torch.arange(1, T + 1, device=x.device, dtype=torch.float32)
                c, mag = self._polar(qh, kh, vh, n_keys, is_causal=True)
                out = self._combine(c, mag, gate_all, total)
                if self.mem is not None:
                    out = out + self._mem_prefill_dense(
                        x.view(B, T, -1), qh, kh, vh, slots, mem_table)
                return out

            conv_state_tables = context.conv_state_tables
            q_parts, k_parts, v_parts, tok_starts = [], [], [], []
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
                tok_starts.append(start)
                start += seqlen

            k_packed = torch.cat(k_parts, dim=0)
            v_packed = torch.cat(v_parts, dim=0)

            # Write K/V to paged cache
            if self.attn.k_cache.numel() > 0 and context.slot_mapping is not None and context.slot_mapping.numel() > 0:
                store_kvcache(k_packed, v_packed, self.attn.k_cache, self.attn.v_cache, context.slot_mapping)

            # Polar attention per sequence. A fresh prefill (num_cached_tokens == 0) is
            # standard causal (n_keys = 1..seqlen). A chunked-prefill continuation
            # gathers its cached prefix K/V from the paged cache and runs is_causal=False
            # with absolute n_keys = start+1..start+seqlen, so queries attend to the
            # whole context. NOTE: a cross-request prefix-cache hit shares K/V blocks
            # correctly here, but the per-seq conv/memory state tables start from zeros
            # for the new sequence, so its early outputs can drift — see docs/inference.md.
            c_parts, mag_parts, mem_parts = [], [], []
            for i, (qi, ki, vi) in enumerate(zip(q_parts, k_parts, v_parts)):
                seq = context.seqs[i]
                seqlen = qi.shape[0]
                cached = seq.num_cached_tokens
                qh = qi.transpose(0, 1).unsqueeze(0).contiguous()                    # (1,H,T,dk)
                if cached > 0:
                    bt = torch.as_tensor(seq.block_table, device=x.device, dtype=torch.long)
                    block_size = self.attn.k_cache.shape[1]
                    n_blk = (cached + block_size - 1) // block_size
                    k_pref = self.attn.k_cache[bt[:n_blk]].reshape(-1, self.num_kv_heads, dk)[:cached]
                    v_pref = self.attn.v_cache[bt[:n_blk]].reshape(-1, self.num_kv_heads, dk)[:cached]
                    k_seq = torch.cat([k_pref, ki], dim=0)
                    v_seq = torch.cat([v_pref, vi], dim=0)
                    n_keys = torch.arange(cached + 1, cached + seqlen + 1, device=x.device, dtype=torch.float32)
                    is_causal = False
                else:
                    k_seq, v_seq = ki, vi
                    n_keys = torch.arange(1, seqlen + 1, device=x.device, dtype=torch.float32)
                    is_causal = True
                kh = k_seq.repeat_interleave(groups, dim=1).transpose(0, 1).unsqueeze(0).contiguous()
                vh = v_seq.repeat_interleave(groups, dim=1).transpose(0, 1).unsqueeze(0).contiguous()
                c, mag = self._polar(qh, kh, vh, n_keys, is_causal=is_causal)
                c_parts.append(c)
                mag_parts.append(mag)
                if self.mem is not None:
                    # memory consumes only the fresh chunk (the table state covers the
                    # prefix); kh/vh time-sliced past `cached` are the fresh expanded k/v.
                    x_seq = x[tok_starts[i]:tok_starts[i] + seqlen]
                    mem_parts.append(self._mem_prefill(seq, x_seq, qh,
                                                       kh[:, :, cached:], vh[:, :, cached:], mem_table))
            c = torch.cat(c_parts, dim=2)        # (1, H, total, dk)
            mag = torch.cat(mag_parts, dim=2)    # (1, H, total)
            out = self._combine(c, mag, gate_all, total)
            if self.mem is not None:
                out = out + torch.cat(mem_parts, dim=0)
            return out

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

            # Polar decode: the single new query attends to its cached context. On CUDA
            # the paged Triton kernel reads K/V directly from the cache via block_tables +
            # context_lens (no gather, no host sync, fixed shapes -> CUDA-graph capturable,
            # GQA done in-kernel). The CPU fallback gathers per sequence and runs the
            # materialized polar_reduce.
            if polar_attention_decode is not None and HAS_TRITON and x.is_cuda:
                c, mag = polar_attention_decode(
                    q_attn, self.attn.k_cache, self.attn.v_cache,
                    context.block_tables, context.context_lens,
                    window=self.window, **self._polar_params())
                c = c.unsqueeze(2)         # (B, H, 1, dk)
                mag = mag.unsqueeze(-1)    # (B, H, 1)
            else:
                max_seqlen = int(context.context_lens.max().item())
                block_size = self.attn.k_cache.shape[1]
                n_blocks = context.block_tables.shape[1]
                k_full = self.attn.k_cache[context.block_tables.clamp(min=0)].reshape(
                    batch_size, n_blocks * block_size, self.num_kv_heads, dk)[:, :max_seqlen]
                v_full = self.attn.v_cache[context.block_tables.clamp(min=0)].reshape(
                    batch_size, n_blocks * block_size, self.num_kv_heads, dk)[:, :max_seqlen]
                k_full = k_full.repeat_interleave(groups, dim=2)        # (B, S, H, dk)
                v_full = v_full.repeat_interleave(groups, dim=2)
                ctx = context.context_lens
                W = self.window
                c_list, mag_list = [], []
                for b in range(batch_size):
                    n = int(ctx[b])
                    # window: slice to the last min(n, W) keys — equivalent to the band
                    # mask, and _polar's temp/null then see the capped count directly.
                    lo = 0 if W is None else max(0, n - W)
                    qh = q_attn[b].unsqueeze(0).unsqueeze(2).contiguous()                # (H,dk)->(1,H,1,dk)
                    kh = k_full[b, lo:n].transpose(0, 1).unsqueeze(0).contiguous()       # (1, H, n-lo, dk)
                    vh = v_full[b, lo:n].transpose(0, 1).unsqueeze(0).contiguous()
                    n_keys = torch.tensor([float(n - lo)], device=x.device)
                    cc, mm = self._polar(qh, kh, vh, n_keys, is_causal=False)
                    c_list.append(cc)          # (1, H, 1, dk)
                    mag_list.append(mm)        # (1, H, 1)
                c = torch.cat(c_list, dim=0)       # (B, H, 1, dk)
                mag = torch.cat(mag_list, dim=0)   # (B, H, 1)
            out = self._combine(c, mag, gate_all, batch_size)
            if self.mem is not None:
                k_mem = k_attn.repeat_interleave(groups, dim=1)         # (B, H, dk)
                v_mem = v_attn.repeat_interleave(groups, dim=1)
                out = out + self._mem_decode(x, q_attn, k_mem, v_mem, seq_slots, mem_table)
            return out


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
            if context.dense_prefill:
                B, T = context.dense_batch_size, context.dense_seq_len
                x_gated = x_gated_all.view(B, T, -1)
                x_conv = prefill_causal_conv1d_dense(
                    f"conv_{self.layer_idx}_gated", context.seq_slots, x_gated,
                    w_conv, None, conv_state_tables,
                )
                return self.out_proj((C_all.view(B, T, -1) * x_conv).reshape(B * T, -1))
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
        attn_window: int = None,
        mem_enabled: bool = False,
        mem_chunk: int = 64,
        mem_gamma_bias: float = 3.9,
        mem_beta_bias: float = 0.0,
        mem_kernel: str = "auto",
    ):
        super().__init__()
        self.attn = (
            AtmaAttention(layer_idx, dim, head_dim=head_dim, num_kv_heads=num_kv_heads, kernel_size=attn_kernel_size,
                          window=attn_window, mem_enabled=mem_enabled, mem_chunk=mem_chunk,
                          mem_gamma_bias=mem_gamma_bias, mem_beta_bias=mem_beta_bias, mem_kernel=mem_kernel)
            if attention
            else AtmaLFM2Conv(layer_idx, dim, kernel_size=conv_kernel_size)
        )
        self.mlp = InferenceMLP(dim, linear_cls=_infer_linear)
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
                attn_window=config.attn_window,
                mem_enabled=config.mem_enabled,
                mem_chunk=config.mem_chunk,
                mem_gamma_bias=config.mem_gamma_bias,
                mem_beta_bias=config.mem_beta_bias,
                mem_kernel=config.mem_kernel,
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
        if softcap_logits is not None and _HAS_INFERENCE_OPS and logits.is_cuda:
            return softcap_logits(logits)
        return 15.0 * logits * (logits.square() + 225.0).rsqrt()
