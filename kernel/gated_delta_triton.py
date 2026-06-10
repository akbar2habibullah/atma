"""Fused gated-delta decode step (Titans MAG memory) with seq-slot indirection.

One batched single-token step of the gated delta rule (FLA convention: decay-first,
undecayed write, self-inclusive readout M_t q_t), reading and writing the per-sequence
state DIRECTLY in the engine's slot-indexed state table:

    Sd      = gamma * S[slot]                  (decay first)
    pred_v  = sum_k kn_k Sd[k, v]
    u_v     = beta * (v_v - pred_v)            (undecayed write)
    S'[k,v] = Sd[k, v] + kn_k u_v              -> stored back to S[slot] IN PLACE
    r_v     = sum_k qn_k S'[k, v]              (readout M_t q_t)

with qn, kn the L2-normalized q, k (in-kernel, matching F.normalize). The state uses
FLA's [K, V] layout, fp32. Replaces the gather -> fla.fused_recurrent -> scatter
sequence in the decode path, which moved the (large) state three times per step; this
kernel moves it once (read + write in place). No host sync, fixed launch shape ->
CUDA-graph capturable. Numerically equals the explicit torch step in
inference/models/atma.py::_mem_decode (fp32; validate with verify.py --cuda and the
verify_fla.py inference-bridge section).
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = torch.cuda.is_available()
except Exception:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _gated_delta_step_kernel(
        Q, K, V, GAMMA, BETA, SLOTS, STATE, R,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kd,
        stride_vb, stride_vh, stride_vd,
        stride_gb, stride_gh,                    # GAMMA and BETA (both (B, H) contiguous)
        stride_ss, stride_sh, stride_sk, stride_sv,
        stride_rb, stride_rh, stride_rd,
        eps,
        DK: tl.constexpr, BLOCK_V: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        pv = tl.program_id(2)
        eps = eps.to(tl.float32)                 # python-float args are fp64 on some Tritons

        offs_k = tl.arange(0, DK)
        offs_v = pv * BLOCK_V + tl.arange(0, BLOCK_V)

        q = tl.load(Q + b * stride_qb + h * stride_qh + offs_k * stride_qd).to(tl.float32)
        k = tl.load(K + b * stride_kb + h * stride_kh + offs_k * stride_kd).to(tl.float32)
        v = tl.load(V + b * stride_vb + h * stride_vh + offs_v * stride_vd).to(tl.float32)
        gamma = tl.load(GAMMA + b * stride_gb + h * stride_gh).to(tl.float32)
        beta = tl.load(BETA + b * stride_gb + h * stride_gh).to(tl.float32)
        # unit-norm keys/queries (delta-rule stability), matching F.normalize's eps clamp
        qn = q / tl.maximum(tl.sqrt(tl.sum(q * q)), eps)
        kn = k / tl.maximum(tl.sqrt(tl.sum(k * k)), eps)

        slot = tl.load(SLOTS + b).to(tl.int64)
        s_ptr = (STATE + slot * stride_ss + h * stride_sh
                 + offs_k[:, None] * stride_sk + offs_v[None, :] * stride_sv)
        S = tl.load(s_ptr)                                   # (DK, BLOCK_V) fp32
        Sd = gamma * S                                       # decay first
        pred = tl.sum(kn[:, None] * Sd, axis=0)              # (BLOCK_V,)
        u = beta * (v - pred)                                # undecayed write
        S_new = Sd + kn[:, None] * u[None, :]
        tl.store(s_ptr, S_new)                               # in-place state update
        r = tl.sum(qn[:, None] * S_new, axis=0)              # readout M_t q_t
        tl.store(R + b * stride_rb + h * stride_rh + offs_v * stride_rd, r)


@torch.no_grad()
def gated_delta_decode_step(q, k, v, gamma, beta, state_table, slots,
                            eps=1e-12, block_v=32):
    """One batched gated-delta step, in place on the slot-indexed state table.

    q, k, v     : (B, H, dk) current-token tensors (KV heads expanded to H)
    gamma, beta : (B, H) retention / write strength in (0, 1)
    state_table : (max_seqs, H, dk, dv) fp32, FLA [K, V] layout — updated IN PLACE
    slots       : (B,) int64 row indices into state_table

    Returns r (B, H, dv) fp32. CUDA-graph capturable.
    """
    B, H, dk = q.shape
    dv = state_table.shape[3]
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    gamma = gamma.float().contiguous()
    beta = beta.float().contiguous()
    r = torch.empty((B, H, dv), device=q.device, dtype=torch.float32)

    grid = (B, H, triton.cdiv(dv, block_v))
    _gated_delta_step_kernel[grid](
        q, k, v, gamma, beta, slots, state_table, r,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        gamma.stride(0), gamma.stride(1),
        state_table.stride(0), state_table.stride(1), state_table.stride(2), state_table.stride(3),
        r.stride(0), r.stride(1), r.stride(2),
        eps,
        DK=dk, BLOCK_V=block_v,
        num_warps=4, num_stages=2,
    )
    return r
