"""Paged GQA softmax decode for the isolated NoPE/RoPE benchmark engine."""

import math
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
    def _paged_softmax_kernel(
        Q,
        KC,
        VC,
        BT,
        LENS,
        OUT,
        sqb,
        sqh,
        sqd,
        skb,
        skp,
        skh,
        skd,
        svb,
        svp,
        svh,
        svd,
        sbtb,
        sbtn,
        sob,
        soh,
        sod,
        scale,
        BLOCK_SIZE: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
        BN: tl.constexpr,
        DK: tl.constexpr,
        G: tl.constexpr,
        GP: tl.constexpr,
        WINDOW: tl.constexpr,
        DOT_DTYPE: tl.constexpr,
    ):
        b, kvh = tl.program_id(0), tl.program_id(1)
        od = tl.arange(0, DK)
        og = tl.arange(0, GP)
        gmask = og < G
        heads = tl.where(gmask, kvh * G + og, 0)
        q = tl.load(
            Q + b * sqb + heads[:, None] * sqh + od[None, :] * sqd,
            mask=gmask[:, None],
            other=0.0,
        ).to(DOT_DTYPE)
        n = tl.load(LENS + b)
        hi = tl.minimum(tl.cdiv(n, BN) * BN, MAX_BLOCKS * BLOCK_SIZE)
        lo = (tl.maximum(n - WINDOW, 0) // BN) * BN if WINDOW > 0 else 0
        m = tl.full([GP], -1e38, tl.float32)
        z = tl.zeros([GP], tl.float32)
        acc = tl.zeros([GP, DK], tl.float32)
        scale = scale.to(tl.float32)
        for start in range(lo, hi, BN):
            on = start + tl.arange(0, BN)
            valid = on < n
            if WINDOW > 0:
                valid &= on >= n - WINDOW
            logical, within = on // BLOCK_SIZE, on % BLOCK_SIZE
            phys = tl.load(BT + b * sbtb + logical * sbtn, mask=valid, other=0)
            kp = (
                KC
                + phys[:, None] * skb
                + within[:, None] * skp
                + kvh * skh
                + od[None, :] * skd
            )
            vp = (
                VC
                + phys[:, None] * svb
                + within[:, None] * svp
                + kvh * svh
                + od[None, :] * svd
            )
            k = tl.load(kp, mask=valid[:, None], other=0.0).to(DOT_DTYPE)
            v = tl.load(vp, mask=valid[:, None], other=0.0).to(DOT_DTYPE)
            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            scores = tl.where(valid[None, :] & gmask[:, None], scores, -1e38).to(
                tl.float32
            )
            m_new = tl.maximum(m, tl.max(scores, 1))
            alpha = tl.exp(m - m_new)
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(valid[None, :], p, 0.0)
            z = z * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(
                p.to(DOT_DTYPE), v, input_precision="ieee"
            )
            m = m_new
        out = acc / tl.maximum(z[:, None], 1e-9)
        tl.store(
            OUT + b * sob + heads[:, None] * soh + od[None, :] * sod,
            out,
            mask=gmask[:, None],
        )


@torch.no_grad()
def paged_softmax_decode(
    q, k_cache, v_cache, block_tables, context_lens, *, scale, window=None
):
    B, H, D = q.shape
    kvh = k_cache.shape[2]
    groups = H // kvh
    gp = max(16, triton.next_power_of_2(groups))
    out = torch.empty_like(q)
    dot_dtype = (
        tl.float16
        if q.dtype == torch.float16
        else (tl.float32 if q.dtype == torch.float32 else tl.bfloat16)
    )
    _paged_softmax_kernel[(B, kvh)](
        q.contiguous(),
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        out,
        *q.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *block_tables.stride(),
        *out.stride(),
        float(scale),
        BLOCK_SIZE=k_cache.shape[1],
        MAX_BLOCKS=block_tables.shape[1],
        BN=64,
        DK=D,
        G=groups,
        GP=gp,
        WINDOW=0 if window is None else int(window),
        DOT_DTYPE=dot_dtype,
        num_warps=4,
        num_stages=2,
    )
    return out
