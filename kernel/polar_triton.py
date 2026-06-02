"""FlashAttention-style Triton kernels for Polar Attention.

Polar attention factors each query's result into a length-invariant *direction*
channel ``c`` (unit vector) and a bounded *magnitude* channel ``mag`` derived from
a single temperature-sharpened softmax with an EV-corrected null sink.  See
``atma/POLAR_ATTENTION.md`` for the full derivation and ``model/blocks.py`` for the
pure-PyTorch oracle (``polar_reduce`` materialized / ``polar_attention_online``
streamed) that these kernels reproduce.

Math (per query ``i`` over causal keys ``j < n_i``):

    sigma_ij = (q_i . k_j) / sqrt(dk)
    temp_i   = 1 + softplus(len_gain_raw) * log(n_i)
    null_i   = null_base + softplus(null_slope_raw) * sqrt(log(n_i + 1))
    a_ij     = temp_i * sigma_ij           (real keys; masked j>=n_i -> -inf)
    a_iN     = temp_i * null_i             (null sink)
    w        = softmax([a_i*, a_iN])
    s_i      = sum_j w_ij v_j + w_iN v_null ;  c_i = s_i / ||s_i||
    n_eff    = L^2 / Q2          (L = sum p, Q2 = sum p^2, participation ratio)
    m_eff    = n_eff * (L / Z)   (Z = L + p_null) = L^3 / (Q2 * Z)
    mag_i    = tanh(softplus(mag_beta_raw) * log1p(m_eff))

The forward kernel streams keys in blocks maintaining (M, L, Q2, S); the backward
splits the cheap per-query preamble (PyTorch, O(B*H*T*dk)) from the two O(T^2)
matmul loops (Triton: dq, and dk/dv).
"""

import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = torch.cuda.is_available()
except Exception:  # pragma: no cover - triton always present in this env
    triton = None
    tl = None
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _polar_fwd_kernel(
        Q, K, V, NKEYS, VNULL, SPG, NULLBASE, SPS, BETA,
        C, MAG, M_OUT, L_OUT, Q2_OUT, S_OUT,
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_cb, stride_ch, stride_ct, stride_cd,
        stride_sb, stride_sh, stride_st, stride_sd,
        stride_mb, stride_mh, stride_mt,            # for MAG / M_OUT / L_OUT / Q2_OUT (B,H,T)
        B, H, Tq, Tk,
        scale, eps,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DK: tl.constexpr,
        IS_CAUSAL: tl.constexpr, INPUT_PRECISION: tl.constexpr, DOT_DTYPE: tl.constexpr,
        WINDOW: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        # Force fp32: un-annotated python-float kernel args are fp64 on some Triton
        # versions, which would silently promote the fp32 loop-carried accumulators.
        scale = scale.to(tl.float32)
        eps = eps.to(tl.float32)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, DK)
        m_valid = offs_m < Tq

        # Keep q/k/v in native dtype for tl.dot (bf16 -> tensor cores, fp32 accum);
        # only the softmax/elementwise math runs in fp32.
        q_ptrs = (Q + b * stride_qb + h * stride_qh
                  + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd)
        q = tl.load(q_ptrs, mask=m_valid[:, None], other=0.0).to(DOT_DTYPE)

        # per-query length quantities. n_i is the RAW valid-key count used for the
        # causal mask (matches the oracle's `key_idx >= n_keys`); the clamp to 1 is
        # applied ONLY to temp/null (matches polar_temp_null), so a query with
        # n_keys=0 correctly drains entirely to the null sink.
        n_i = tl.load(NKEYS + offs_m, mask=m_valid, other=0.0).to(tl.float32)
        # WINDOW>0: causal sliding window — each query sees only its last WINDOW keys,
        # so temp/null use the capped count min(n_i, WINDOW) and the score loop masks
        # the band (key_pos >= n_i - WINDOW). WINDOW==0 disables it (full causal).
        if WINDOW > 0:
            n_clamp = tl.maximum(tl.minimum(n_i, float(WINDOW)), 1.0)
        else:
            n_clamp = tl.maximum(n_i, 1.0)
        spg = tl.load(SPG + h).to(tl.float32)
        sps = tl.load(SPS + h).to(tl.float32)
        beta = tl.load(BETA + h).to(tl.float32)
        nb = tl.load(NULLBASE + h).to(tl.float32)
        logn = tl.log(n_clamp)
        temp = 1.0 + spg * logn
        nullv = nb + sps * tl.sqrt(tl.log(n_clamp + 1.0))

        m_i = tl.full([BLOCK_M], -1e38, tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        q2_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, DK], tl.float32)

        if IS_CAUSAL:
            hi = tl.minimum((pid_m + 1) * BLOCK_M, Tk)
        else:
            hi = Tk

        for start_n in range(0, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_valid = offs_n < Tk
            k_ptrs = (K + b * stride_kb + h * stride_kh
                      + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd)
            v_ptrs = (V + b * stride_vb + h * stride_vh
                      + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd)
            k = tl.load(k_ptrs, mask=n_valid[:, None], other=0.0).to(DOT_DTYPE)
            v = tl.load(v_ptrs, mask=n_valid[:, None], other=0.0).to(DOT_DTYPE)

            sig = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * scale
            a = sig * temp[:, None]
            valid = offs_n[None, :] < n_i[:, None]                          # future
            if WINDOW > 0:
                valid = valid & (offs_n[None, :] >= (n_i[:, None] - WINDOW))  # older than window
            a = tl.where(valid, a, -1e38).to(tl.float32)   # stay fp32 (sentinel literal guard)

            m_new = tl.maximum(m_i, tl.max(a, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(a - m_new[:, None])
            p = tl.where(valid, p, 0.0)

            l_i = l_i * alpha + tl.sum(p, 1)
            q2_i = q2_i * alpha * alpha + tl.sum(p * p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(DOT_DTYPE), v, input_precision=INPUT_PRECISION)
            m_i = m_new

        # fold null sink
        a_n = temp * nullv
        m_new = tl.maximum(m_i, a_n)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha
        q2_i = q2_i * alpha * alpha
        acc = acc * alpha[:, None]
        m_i = m_new
        p_n = tl.exp(a_n - m_i)
        Z = l_i + p_n

        vnull = tl.load(VNULL + h * DK + offs_d).to(tl.float32)
        s = acc + p_n[:, None] * vnull[None, :]
        snorm = tl.sqrt(tl.sum(s * s, 1))
        snorm = tl.maximum(snorm, eps)
        c = s / snorm[:, None]

        n_eff = l_i * l_i / tl.maximum(q2_i, eps)
        m_eff = n_eff * (l_i / tl.maximum(Z, eps))
        # tanh(x) = 2*sigmoid(2x) - 1  (tl.math.tanh absent in triton 3.7).
        # log(1+m_eff): triton 3.7 has no log1p; for m_eff < ~1e-7 this rounds mag to 0
        # instead of an ~1e-N value, which is semantically "no matches" (negligible, well
        # under fp32 eps). The backward recomputes mag with exact torch.log1p, so gradients
        # are unaffected.
        mag = 2.0 * tl.sigmoid(2.0 * (beta * tl.log(1.0 + m_eff))) - 1.0

        c_ptrs = (C + b * stride_cb + h * stride_ch
                  + offs_m[:, None] * stride_ct + offs_d[None, :] * stride_cd)
        tl.store(c_ptrs, c, mask=m_valid[:, None])
        s_ptrs = (S_OUT + b * stride_sb + h * stride_sh
                  + offs_m[:, None] * stride_st + offs_d[None, :] * stride_sd)
        tl.store(s_ptrs, s, mask=m_valid[:, None])

        mbase = b * stride_mb + h * stride_mh + offs_m * stride_mt
        tl.store(MAG + mbase, mag, mask=m_valid)
        tl.store(M_OUT + mbase, m_i, mask=m_valid)
        tl.store(L_OUT + mbase, l_i, mask=m_valid)
        tl.store(Q2_OUT + mbase, q2_i, mask=m_valid)


def _softplus(x):
    return F.softplus(x)


_DOT = {torch.float32: tl.float32, torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16} if HAS_TRITON else {}


def _dtype_meta(dtype):
    """(triton dot dtype, input_precision, is_fp32). bf16/fp16 -> tensor-core dots."""
    if dtype == torch.float32:
        return tl.float32, "ieee", True
    return _DOT.get(dtype, tl.bfloat16), "ieee", False


def _fwd_config(dk, is_fp32):
    """Block sizes / pipelining fitting L4 shared memory (~99 KB). 16-bit tiles are
    half the size of fp32 tiles, so they can run wider / deeper."""
    if is_fp32:
        if dk >= 128:
            return dict(block_m=64, block_n=32, num_warps=4, num_stages=2)
        return dict(block_m=64, block_n=64, num_warps=4, num_stages=2)
    # bf16 / fp16
    if dk >= 128:
        return dict(block_m=128, block_n=64, num_warps=8, num_stages=2)
    return dict(block_m=128, block_n=64, num_warps=4, num_stages=3)


def _bwd_config(dk, is_fp32):
    """Backward keeps an extra resident gs tile (and dk/dv accumulators for the
    dk/dv kernel), so it needs smaller tiles than forward. Separate dq / dk-dv
    launch params."""
    if is_fp32:
        if dk >= 128:
            return dict(
                dq=dict(block_m=64, block_n=32, num_warps=4, num_stages=1),
                kv=dict(block_m=32, block_n=32, num_warps=4, num_stages=1),
            )
        return dict(
            dq=dict(block_m=64, block_n=64, num_warps=4, num_stages=2),
            kv=dict(block_m=64, block_n=64, num_warps=4, num_stages=2),
        )
    # bf16 / fp16 (gs passed in 16-bit, dots on tensor cores)
    if dk >= 128:
        return dict(
            dq=dict(block_m=64, block_n=64, num_warps=4, num_stages=2),
            kv=dict(block_m=64, block_n=64, num_warps=4, num_stages=2),
        )
    return dict(
        dq=dict(block_m=64, block_n=64, num_warps=4, num_stages=2),
        kv=dict(block_m=64, block_n=64, num_warps=4, num_stages=2),
    )


def _polar_forward(q, k, v, n_keys, v_null, null_base, null_slope_raw,
                   len_gain_raw, mag_beta_raw, eps, is_causal, input_precision,
                   block_m=None, block_n=None, num_warps=None, num_stages=None, window=None):
    """Launch the forward kernel. Returns c, mag, and the saved stats (M,L,Q2,s).

    window (int, optional): eval-only causal sliding window (each query attends to its
    last `window` keys). Forward-only — the streaming backward does not model the band."""
    B, H, Tq, dk = q.shape
    Tk = k.shape[2]
    out_dtype = q.dtype
    dev = q.device

    # is_causal uses the triangular loop bound hi=(block+1)*BLOCK_M, which is only
    # valid for standard self-attention (n_keys = arange(1,T+1), Tq == Tk). Any other
    # layout (decode, offset prefill) must pass is_causal=False, which scans all keys
    # and relies solely on the n_keys mask. Guard against the silent-wrong-result trap.
    if is_causal and Tq != Tk:
        raise ValueError(
            f"is_causal=True requires Tq == Tk (standard causal self-attention); got "
            f"Tq={Tq}, Tk={Tk}. Use is_causal=False with an explicit n_keys for "
            f"decode / offset-prefill layouts.")
    # DK indexes a tl.arange / tl.dot tile -> must be a power of two (model uses 128).
    if dk & (dk - 1) != 0:
        raise ValueError(f"head_dim (dk={dk}) must be a power of two for the Triton kernel.")

    dot_dtype, ip, is_fp32 = _dtype_meta(out_dtype)
    if is_fp32:
        ip = input_precision           # caller may request "tf32" for fp32 speed
    cfg = _fwd_config(dk, is_fp32)
    block_m = block_m or cfg["block_m"]
    block_n = block_n or cfg["block_n"]
    num_warps = num_warps or cfg["num_warps"]
    num_stages = num_stages or cfg["num_stages"]

    spg = _softplus(len_gain_raw.float()).contiguous()
    sps = _softplus(null_slope_raw.float()).contiguous()
    beta = _softplus(mag_beta_raw.float()).contiguous()
    nb = null_base.float().contiguous()
    vnull = v_null.float().contiguous()
    n_keys = n_keys.to(dev).float().contiguous()

    c = torch.empty((B, H, Tq, dk), device=dev, dtype=torch.float32)
    s = torch.empty((B, H, Tq, dk), device=dev, dtype=torch.float32)
    mag = torch.empty((B, H, Tq), device=dev, dtype=torch.float32)
    M = torch.empty((B, H, Tq), device=dev, dtype=torch.float32)
    L = torch.empty((B, H, Tq), device=dev, dtype=torch.float32)
    Q2 = torch.empty((B, H, Tq), device=dev, dtype=torch.float32)

    scale = 1.0 / math.sqrt(dk)
    grid = (triton.cdiv(Tq, block_m), B * H)
    _polar_fwd_kernel[grid](
        q, k, v, n_keys, vnull, spg, nb, sps, beta,
        c, mag, M, L, Q2, s,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        c.stride(0), c.stride(1), c.stride(2), c.stride(3),
        s.stride(0), s.stride(1), s.stride(2), s.stride(3),
        mag.stride(0), mag.stride(1), mag.stride(2),
        B, H, Tq, Tk,
        scale, eps,
        BLOCK_M=block_m, BLOCK_N=block_n, DK=dk,
        IS_CAUSAL=is_causal, INPUT_PRECISION=ip, DOT_DTYPE=dot_dtype,
        WINDOW=(0 if window is None else int(window)),
        num_warps=num_warps, num_stages=num_stages,
    )

    # Diagnostic probe (off by default → skipped entirely). The kernel doesn't emit
    # n_eff / w_null, so recover them from the saved stats (same algebra as the
    # backward preamble) and feed the shared sink in model.blocks.
    from model import blocks as _blocks
    if _blocks._PROBE is not None:
        # temp/null use the windowed count when a window is active (matches the kernel).
        nf = (n_keys if window is None
              else torch.minimum(n_keys, n_keys.new_tensor(float(window)))).clamp(min=1.0)  # (Tq,)
        logn = torch.log(nf).view(1, Tq)
        t = 1.0 + spg.view(H, 1) * logn                              # (H, Tq)
        nu = nb.view(H, 1) + sps.view(H, 1) * torch.sqrt(torch.log(nf + 1.0)).view(1, Tq)
        p_n = torch.exp((t * nu).view(1, H, Tq) - M)                 # (B, H, Tq)
        Z = L + p_n
        n_eff = L * L / Q2.clamp_min(eps)
        _blocks._probe_emit(n_eff, mag, p_n / Z.clamp_min(eps))

    return c.to(out_dtype), mag.to(out_dtype), M, L, Q2, s


@torch.no_grad()
def polar_attention_fwd(q, k, v, n_keys, *, v_null, null_base, null_slope_raw,
                        len_gain_raw, mag_beta_raw, eps=1e-6, is_causal=True,
                        input_precision="ieee", window=None):
    """Forward-only polar attention (no autograd graph). q,k,v: (B,H,T,dk).

    Returns (c, mag) cast to q's dtype, matching ``polar_reduce`` numerically.
    Intended for inference / no-grad use.

    window (int, optional): eval-only causal sliding window — each query attends to
    only its last `window` keys. Matches polar_attention_online(window=...).
    """
    c, mag, *_ = _polar_forward(
        q.contiguous(), k.contiguous(), v.contiguous(), n_keys,
        v_null, null_base, null_slope_raw, len_gain_raw, mag_beta_raw,
        eps, is_causal, input_precision, window=window)
    return c, mag


# ---------------------------------------------------------------------------
# Paged decode kernel: one query per sequence attends to its whole cached
# context, read DIRECTLY from the paged KV cache via block_tables + context_lens
# (like flash_attn_with_kvcache, but with the polar reduction). No gather, no
# host sync, fixed launch shape -> CUDA-graph capturable, scales to large batch.
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _polar_decode_kernel(
        Q, K_CACHE, V_CACHE, BLOCK_TABLES, CONTEXT_LENS,
        VNULL, SPG, NULLBASE, SPS, BETA,
        C_OUT, MAG_OUT,
        stride_qb, stride_qh, stride_qd,
        stride_kc_blk, stride_kc_pos, stride_kc_h, stride_kc_d,
        stride_vc_blk, stride_vc_pos, stride_vc_h, stride_vc_d,
        stride_btb, stride_btn,
        stride_cb, stride_ch, stride_cd,
        stride_mb, stride_mh,
        H, NUM_KV_HEADS, scale, eps,
        BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
        BLOCK_N: tl.constexpr, DK: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        scale = scale.to(tl.float32)     # python-float args are fp64 on some Triton versions
        eps = eps.to(tl.float32)
        groups = H // NUM_KV_HEADS
        kv_h = h // groups
        offs_d = tl.arange(0, DK)

        q = tl.load(Q + b * stride_qb + h * stride_qh + offs_d * stride_qd).to(tl.float32)  # (DK,)

        n_i = tl.load(CONTEXT_LENS + b)                  # int32 valid-key count
        n_f = tl.maximum(n_i.to(tl.float32), 1.0)
        spg = tl.load(SPG + h).to(tl.float32)
        sps = tl.load(SPS + h).to(tl.float32)
        beta = tl.load(BETA + h).to(tl.float32)
        nb = tl.load(NULLBASE + h).to(tl.float32)
        temp = 1.0 + spg * tl.log(n_f)
        nullv = nb + sps * tl.sqrt(tl.log(n_f + 1.0))

        m_i = -1e38
        l_i = 0.0
        q2_i = 0.0
        acc = tl.zeros([DK], tl.float32)

        total = MAX_BLOCKS * BLOCK_SIZE
        for start in range(0, total, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)        # global key positions
            valid = offs_n < n_i
            blk = offs_n // BLOCK_SIZE                     # logical block index
            within = offs_n % BLOCK_SIZE
            phys = tl.load(BLOCK_TABLES + b * stride_btb + blk * stride_btn,
                           mask=valid, other=0).to(tl.int32)
            k_ptr = (K_CACHE + phys[:, None] * stride_kc_blk + within[:, None] * stride_kc_pos
                     + kv_h * stride_kc_h + offs_d[None, :] * stride_kc_d)
            v_ptr = (V_CACHE + phys[:, None] * stride_vc_blk + within[:, None] * stride_vc_pos
                     + kv_h * stride_vc_h + offs_d[None, :] * stride_vc_d)
            k = tl.load(k_ptr, mask=valid[:, None], other=0.0).to(tl.float32)
            v = tl.load(v_ptr, mask=valid[:, None], other=0.0).to(tl.float32)

            sig = tl.sum(q[None, :] * k, axis=1) * scale  # (BLOCK_N,)
            a = sig * temp
            a = tl.where(valid, a, -1e38)
            m_new = tl.maximum(m_i, tl.max(a, 0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(a - m_new)
            p = tl.where(valid, p, 0.0)
            l_i = l_i * alpha + tl.sum(p, 0)
            q2_i = q2_i * alpha * alpha + tl.sum(p * p, 0)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

        # fold null sink
        a_n = temp * nullv
        m_new = tl.maximum(m_i, a_n)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha
        q2_i = q2_i * alpha * alpha
        acc = acc * alpha
        m_i = m_new
        p_n = tl.exp(a_n - m_i)
        Z = l_i + p_n

        vnull = tl.load(VNULL + h * DK + offs_d).to(tl.float32)
        s = acc + p_n * vnull
        snorm = tl.maximum(tl.sqrt(tl.sum(s * s)), eps)
        c = s / snorm
        n_eff = l_i * l_i / tl.maximum(q2_i, eps)
        m_eff = n_eff * (l_i / tl.maximum(Z, eps))
        mag = 2.0 * tl.sigmoid(2.0 * (beta * tl.log(1.0 + m_eff))) - 1.0

        tl.store(C_OUT + b * stride_cb + h * stride_ch + offs_d * stride_cd, c)
        tl.store(MAG_OUT + b * stride_mb + h * stride_mh, mag)


@torch.no_grad()
def polar_attention_decode(q, k_cache, v_cache, block_tables, context_lens, *,
                           v_null, null_base, null_slope_raw, len_gain_raw, mag_beta_raw,
                           eps=1e-6, block_n=64):
    """Paged polar decode. One query per sequence over its cached context.

    q            : (B, H, dk)  current-token query (KV heads NOT expanded; GQA done in-kernel)
    k_cache      : (num_blocks, block_size, num_kv_heads, dk)  paged cache (contiguous)
    v_cache      : same shape as k_cache
    block_tables : (B, max_blocks) int32  logical->physical block map
    context_lens : (B,) int32  valid key count per sequence (incl. current token)

    Returns c (B, H, dk), mag (B, H). CUDA-graph capturable (no host sync, fixed shapes).
    Matches model.blocks.polar_reduce for the equivalent dense computation to ~1e-3 (bf16).
    """
    B, H, dk = q.shape
    num_kv_heads = k_cache.shape[2]
    max_blocks = block_tables.shape[1]
    block_size = k_cache.shape[1]
    out_dtype = q.dtype
    dev = q.device

    spg = F.softplus(len_gain_raw.float()).contiguous()
    sps = F.softplus(null_slope_raw.float()).contiguous()
    beta = F.softplus(mag_beta_raw.float()).contiguous()
    nb = null_base.float().contiguous()
    vnull = v_null.float().contiguous()
    q = q.contiguous()

    c = torch.empty((B, H, dk), device=dev, dtype=torch.float32)
    mag = torch.empty((B, H), device=dev, dtype=torch.float32)
    scale = 1.0 / math.sqrt(dk)

    grid = (B, H)
    _polar_decode_kernel[grid](
        q, k_cache, v_cache, block_tables, context_lens,
        vnull, spg, nb, sps, beta,
        c, mag,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        mag.stride(0), mag.stride(1),
        H, num_kv_heads, scale, eps,
        BLOCK_SIZE=block_size, MAX_BLOCKS=max_blocks, BLOCK_N=block_n, DK=dk,
        num_warps=4, num_stages=2,
    )
    return c.to(out_dtype), mag.to(out_dtype)


# ---------------------------------------------------------------------------
# Backward kernels: the O(T^2) matmul loops only.  dq (query-parallel) and
# dk/dv (key-parallel) consume per-query (gs, gL, gQ2) produced by the cheap
# PyTorch preamble below.
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _polar_bwd_dq_kernel(
        Q, K, V, GS, GL, GQ2, TEMP, M, NKEYS,
        DQ, DT,
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_gsb, stride_gsh, stride_gst, stride_gsd,
        stride_mb, stride_mh, stride_mt,            # GL, GQ2, M, DT (B,H,T)
        stride_th,                                   # TEMP (H,T): row stride
        B, H, Tq, Tk, scale,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DK: tl.constexpr,
        IS_CAUSAL: tl.constexpr, INPUT_PRECISION: tl.constexpr, DOT_DTYPE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        scale = scale.to(tl.float32)   # guard: python-float args are fp64 on some Triton versions

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, DK)
        m_valid = offs_m < Tq

        q = tl.load(Q + b * stride_qb + h * stride_qh
                    + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                    mask=m_valid[:, None], other=0.0).to(DOT_DTYPE)
        gs = tl.load(GS + b * stride_gsb + h * stride_gsh
                     + offs_m[:, None] * stride_gst + offs_d[None, :] * stride_gsd,
                     mask=m_valid[:, None], other=0.0).to(DOT_DTYPE)
        mbase = b * stride_mb + h * stride_mh + offs_m * stride_mt
        gL = tl.load(GL + mbase, mask=m_valid, other=0.0).to(tl.float32)
        gQ2 = tl.load(GQ2 + mbase, mask=m_valid, other=0.0).to(tl.float32)
        M_i = tl.load(M + mbase, mask=m_valid, other=0.0).to(tl.float32)
        temp = tl.load(TEMP + h * stride_th + offs_m, mask=m_valid, other=1.0).to(tl.float32)
        n_i = tl.load(NKEYS + offs_m, mask=m_valid, other=0.0).to(tl.float32)  # raw, for mask

        dq = tl.zeros([BLOCK_M, DK], tl.float32)
        dt = tl.zeros([BLOCK_M], tl.float32)

        hi = tl.minimum((pid_m + 1) * BLOCK_M, Tk) if IS_CAUSAL else Tk
        for start_n in range(0, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_valid = offs_n < Tk
            k = tl.load(K + b * stride_kb + h * stride_kh
                        + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
                        mask=n_valid[:, None], other=0.0).to(DOT_DTYPE)
            v = tl.load(V + b * stride_vb + h * stride_vh
                        + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
                        mask=n_valid[:, None], other=0.0).to(DOT_DTYPE)

            sig = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * scale
            valid = offs_n[None, :] < n_i[:, None]
            a = sig * temp[:, None]
            a = tl.where(valid, a, -1e38).to(tl.float32)   # stay fp32 (sentinel literal guard)
            p = tl.exp(a - M_i[:, None])
            p = tl.where(valid, p, 0.0)

            gs_v = tl.dot(gs, tl.trans(v), input_precision=INPUT_PRECISION)
            dLdp = gs_v + gL[:, None] + gQ2[:, None] * (2.0 * p)
            da = dLdp * p
            da = tl.where(valid, da, 0.0)
            dt += tl.sum(da * sig, 1)
            gsig = da * temp[:, None]
            dq += tl.dot(gsig.to(DOT_DTYPE), k, input_precision=INPUT_PRECISION) * scale

        tl.store(DQ + b * stride_qb + h * stride_qh
                 + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                 dq, mask=m_valid[:, None])
        tl.store(DT + mbase, dt, mask=m_valid)


    @triton.jit
    def _polar_bwd_dkdv_kernel(
        Q, K, V, GS, GL, GQ2, TEMP, M, NKEYS,
        DK_OUT, DV_OUT,
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_gsb, stride_gsh, stride_gst, stride_gsd,
        stride_mb, stride_mh, stride_mt,
        stride_th,
        B, H, Tq, Tk, scale,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DK: tl.constexpr,
        IS_CAUSAL: tl.constexpr, INPUT_PRECISION: tl.constexpr, DOT_DTYPE: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        scale = scale.to(tl.float32)   # guard: python-float args are fp64 on some Triton versions

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, DK)
        n_valid = offs_n < Tk

        k = tl.load(K + b * stride_kb + h * stride_kh
                    + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
                    mask=n_valid[:, None], other=0.0).to(DOT_DTYPE)
        v = tl.load(V + b * stride_vb + h * stride_vh
                    + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
                    mask=n_valid[:, None], other=0.0).to(DOT_DTYPE)

        dk = tl.zeros([BLOCK_N, DK], tl.float32)
        dv = tl.zeros([BLOCK_N, DK], tl.float32)

        lo = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M if IS_CAUSAL else 0
        for start_m in range(lo, Tq, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            offs_m = start_m + tl.arange(0, BLOCK_M)
            m_valid = offs_m < Tq
            q = tl.load(Q + b * stride_qb + h * stride_qh
                        + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                        mask=m_valid[:, None], other=0.0).to(DOT_DTYPE)
            gs = tl.load(GS + b * stride_gsb + h * stride_gsh
                         + offs_m[:, None] * stride_gst + offs_d[None, :] * stride_gsd,
                         mask=m_valid[:, None], other=0.0).to(DOT_DTYPE)
            mbase = b * stride_mb + h * stride_mh + offs_m * stride_mt
            gL = tl.load(GL + mbase, mask=m_valid, other=0.0).to(tl.float32)
            gQ2 = tl.load(GQ2 + mbase, mask=m_valid, other=0.0).to(tl.float32)
            M_i = tl.load(M + mbase, mask=m_valid, other=0.0).to(tl.float32)
            temp = tl.load(TEMP + h * stride_th + offs_m, mask=m_valid, other=1.0).to(tl.float32)
            n_i = tl.load(NKEYS + offs_m, mask=m_valid, other=0.0).to(tl.float32)

            sig = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * scale
            valid = offs_n[None, :] < n_i[:, None]
            a = sig * temp[:, None]
            a = tl.where(valid, a, -1e38).to(tl.float32)   # stay fp32 (sentinel literal guard)
            p = tl.exp(a - M_i[:, None])
            p = tl.where(valid, p, 0.0)

            gs_v = tl.dot(gs, tl.trans(v), input_precision=INPUT_PRECISION)
            dLdp = gs_v + gL[:, None] + gQ2[:, None] * (2.0 * p)
            da = dLdp * p
            da = tl.where(valid, da, 0.0)
            gsig = (da * temp[:, None]).to(DOT_DTYPE)

            dv += tl.dot(tl.trans(p.to(DOT_DTYPE)), gs, input_precision=INPUT_PRECISION)
            dk += tl.dot(tl.trans(gsig), q, input_precision=INPUT_PRECISION) * scale

        tl.store(DK_OUT + b * stride_kb + h * stride_kh
                 + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
                 dk, mask=n_valid[:, None])
        tl.store(DV_OUT + b * stride_vb + h * stride_vh
                 + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
                 dv, mask=n_valid[:, None])


def _polar_backward(gc, gm, q, k, v, n_keys, v_null, null_base, null_slope_raw,
                    len_gain_raw, mag_beta_raw, M, L, Q2, s, eps, is_causal,
                    input_precision):
    """Backward: cheap per-query preamble (PyTorch) + dq / dk,dv loops (Triton)."""
    B, H, Tq, dk = q.shape
    Tk = k.shape[2]
    dev = q.device
    fdt = torch.float32

    gc = gc.to(fdt)
    gm = gm.to(fdt)
    n_raw = n_keys.to(dev).to(fdt)                                  # raw, for the causal mask
    n = n_raw.clamp(min=1.0)                                        # clamped, for temp/null/param-grads
    spg = _softplus(len_gain_raw.to(fdt)).view(1, H, 1)
    sps = _softplus(null_slope_raw.to(fdt)).view(1, H, 1)
    beta = _softplus(mag_beta_raw.to(fdt)).view(1, H, 1)
    logn = torch.log(n).view(1, 1, Tq)
    sqrt_lognp1 = torch.sqrt(torch.log(n + 1.0)).view(1, 1, Tq)
    t = 1.0 + spg * logn                                           # (1,H,Tq)
    nu = null_base.to(fdt).view(1, H, 1) + sps * sqrt_lognp1       # (1,H,Tq)

    a_n = (t * nu).expand(B, H, Tq)
    p_n = torch.exp(a_n - M)
    Z = L + p_n
    Zc, Q2c = Z.clamp_min(eps), Q2.clamp_min(eps)
    m_eff = (L ** 3) / (Q2c * Zc)
    log1p_m = torch.log1p(m_eff)
    mag = torch.tanh(beta * log1p_m)

    # magnitude path -> (gL, gQ2, gZ via p_n)
    gme = gm * beta * (1.0 - mag * mag) / (1.0 + m_eff)
    dm_dL = 3.0 * L * L / (Q2c * Zc)
    dm_dQ2 = -(L ** 3) / (Q2c * Q2c * Zc)
    dm_dZ = -(L ** 3) / (Q2c * Zc * Zc)
    gL = gme * (dm_dL + dm_dZ)
    gQ2 = gme * dm_dQ2
    gZ_pn = gme * dm_dZ

    # direction path -> gs, plus grads onto p_n / v_null
    s = s.to(fdt)
    snorm = s.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    c = s / snorm
    gs = (gc - (gc * c).sum(-1, keepdim=True) * c) / snorm          # (B,H,Tq,dk)
    gpn = (gs * v_null.to(fdt).view(1, H, 1, dk)).sum(-1) + gZ_pn
    grad_v_null = (p_n.unsqueeze(-1) * gs).sum(dim=(0, 2))          # (H,dk)

    da_n = gpn * p_n
    grad_t_null = da_n * nu                                         # (B,H,Tq)
    grad_nu = da_n * t

    # --- Triton matmul loops ---
    out_dtype = q.dtype
    dot_dtype, ip, is_fp32 = _dtype_meta(out_dtype)
    if is_fp32:
        ip = input_precision
    # gs feeds the matmul loops in the compute dtype (16-bit -> tensor cores);
    # the fp32 gs above already produced the param grads, so precision is preserved.
    gs = gs.to(out_dtype).contiguous()
    gL = gL.contiguous(); gQ2 = gQ2.contiguous(); M = M.contiguous()
    temp_ht = t.view(H, Tq).contiguous()                            # (H,Tq), b-independent
    n_keys_f = n_raw.contiguous()                                   # RAW counts for the kernel mask
    scale = 1.0 / math.sqrt(dk)

    dq = torch.zeros((B, H, Tq, dk), device=dev, dtype=fdt)
    dt_real = torch.zeros((B, H, Tq), device=dev, dtype=fdt)
    dk_out = torch.zeros((B, H, Tk, dk), device=dev, dtype=fdt)
    dv_out = torch.zeros((B, H, Tk, dk), device=dev, dtype=fdt)

    cfg = _bwd_config(dk, is_fp32)
    cq, ckv = cfg["dq"], cfg["kv"]

    common = (
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        gs.stride(0), gs.stride(1), gs.stride(2), gs.stride(3),
        gL.stride(0), gL.stride(1), gL.stride(2),
        temp_ht.stride(0),
        B, H, Tq, Tk, scale,
    )
    grid_q = (triton.cdiv(Tq, cq["block_m"]), B * H)
    _polar_bwd_dq_kernel[grid_q](
        q, k, v, gs, gL, gQ2, temp_ht, M, n_keys_f, dq, dt_real, *common,
        BLOCK_M=cq["block_m"], BLOCK_N=cq["block_n"], DK=dk, IS_CAUSAL=is_causal,
        INPUT_PRECISION=ip, DOT_DTYPE=dot_dtype,
        num_warps=cq["num_warps"], num_stages=cq["num_stages"],
    )
    grid_kv = (triton.cdiv(Tk, ckv["block_n"]), B * H)
    _polar_bwd_dkdv_kernel[grid_kv](
        q, k, v, gs, gL, gQ2, temp_ht, M, n_keys_f, dk_out, dv_out, *common,
        BLOCK_M=ckv["block_m"], BLOCK_N=ckv["block_n"], DK=dk, IS_CAUSAL=is_causal,
        INPUT_PRECISION=ip, DOT_DTYPE=dot_dtype,
        num_warps=ckv["num_warps"], num_stages=ckv["num_stages"],
    )

    # assemble parameter grads
    grad_t = dt_real + grad_t_null
    sig_lg = torch.sigmoid(len_gain_raw.to(fdt))
    sig_ns = torch.sigmoid(null_slope_raw.to(fdt))
    sig_mb = torch.sigmoid(mag_beta_raw.to(fdt))
    grad_len_gain = (grad_t * logn).sum(dim=(0, 2)) * sig_lg
    grad_null_base = grad_nu.sum(dim=(0, 2))
    grad_null_slope = (grad_nu * sqrt_lognp1).sum(dim=(0, 2)) * sig_ns
    grad_mag_beta = (gm * (1.0 - mag * mag) * log1p_m).sum(dim=(0, 2)) * sig_mb

    return (dq, dk_out, dv_out, grad_v_null, grad_null_base, grad_null_slope,
            grad_len_gain, grad_mag_beta)


class PolarAttentionTriton(torch.autograd.Function):
    """FlashAttention-style polar attention with a hand-written streaming backward.

    Numerically matches ``model.blocks.polar_attention_online`` (the gradchecked
    oracle) up to floating-point summation order.
    """

    @staticmethod
    def forward(ctx, q, k, v, n_keys, v_null, null_base, null_slope_raw,
                len_gain_raw, mag_beta_raw, eps, is_causal, input_precision):
        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        c, mag, M, L, Q2, s = _polar_forward(
            q, k, v, n_keys, v_null, null_base, null_slope_raw,
            len_gain_raw, mag_beta_raw, eps, is_causal, input_precision)
        ctx.save_for_backward(q, k, v, n_keys, v_null, null_base, null_slope_raw,
                              len_gain_raw, mag_beta_raw, M, L, Q2, s)
        ctx.eps = eps
        ctx.is_causal = is_causal
        ctx.input_precision = input_precision
        return c, mag

    @staticmethod
    def backward(ctx, gc, gm):
        (q, k, v, n_keys, v_null, null_base, null_slope_raw,
         len_gain_raw, mag_beta_raw, M, L, Q2, s) = ctx.saved_tensors
        (dq, dk_out, dv_out, grad_v_null, grad_null_base, grad_null_slope,
         grad_len_gain, grad_mag_beta) = _polar_backward(
            gc, gm, q, k, v, n_keys, v_null, null_base, null_slope_raw,
            len_gain_raw, mag_beta_raw, M, L, Q2, s,
            ctx.eps, ctx.is_causal, ctx.input_precision)

        def cast(x, ref):
            return x.to(ref.dtype)

        return (cast(dq, q), cast(dk_out, k), cast(dv_out, v), None,
                cast(grad_v_null, v_null), cast(grad_null_base, null_base),
                cast(grad_null_slope, null_slope_raw), cast(grad_len_gain, len_gain_raw),
                cast(grad_mag_beta, mag_beta_raw), None, None, None)


def polar_attention(q, k, v, n_keys, *, v_null, null_base, null_slope_raw,
                    len_gain_raw, mag_beta_raw, eps=1e-6, is_causal=True,
                    input_precision="ieee"):
    """Autograd-aware FlashAttention-style polar attention. q,k,v: (B,H,T,dk)
    with GQA KV-heads already expanded to H. Drop-in for
    ``model.blocks.polar_attention_online`` (the streaming PyTorch oracle).

    Returns (c, mag): direction unit vectors (B,H,T,dk) and bounded magnitude (B,H,T).
    """
    return PolarAttentionTriton.apply(
        q, k, v, n_keys, v_null, null_base, null_slope_raw,
        len_gain_raw, mag_beta_raw, eps, is_causal, input_precision)
