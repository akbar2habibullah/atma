"""FlashAttention-style Triton kernels for Polar Attention.

Polar attention factors each query's result into a length-invariant *direction*
channel ``c`` (unit vector) and a bounded *magnitude* channel ``mag`` derived from
a single temperature-sharpened softmax with an EV-corrected null sink.  See
``atma/docs/POLAR_ATTENTION.md`` for the full derivation and ``model/blocks.py`` for the
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
import os
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
        WINDOW: tl.constexpr, PRESERVE_LENGTH: tl.constexpr,
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
        if WINDOW > 0 and not PRESERVE_LENGTH:
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
        if WINDOW > 0:
            # Skip key tiles that are wholly older than every query in this
            # block. The previous implementation only masked them after doing
            # tl.dot, making a sliding window quadratically expensive.
            lo = tl.maximum(pid_m * BLOCK_M - WINDOW, 0)
            lo = (lo // BLOCK_N) * BLOCK_N
        else:
            lo = 0

        for start_n in range(lo, hi, BLOCK_N):
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

        # Q2 is expressed in the running-max scale and can legitimately be far
        # below the direction-normalization epsilon when the null logit wins.
        # Clamping it at `eps` breaks the scale-invariant L^2/Q2 ratio.
        n_eff = l_i * l_i / tl.maximum(q2_i, 1.0e-30)
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


    @triton.jit
    def _polar_packed_fwd_kernel(
        Q, K, V, TILE_SEQ_START, TILE_Q_START, TILE_SEQ_LEN,
        VNULL, SPG, NULLBASE, SPS, BETA, C, MAG,
        stride_qt, stride_qh, stride_qd,
        stride_kt, stride_kh, stride_kd,
        stride_vt, stride_vh, stride_vd,
        stride_ct, stride_ch, stride_cd,
        stride_mt, stride_mh,
        H, scale, eps,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DK: tl.constexpr,
        INPUT_PRECISION: tl.constexpr, DOT_DTYPE: tl.constexpr,
        WINDOW: tl.constexpr,
    ):
        """Fresh causal Polar attention over a packed, ragged request batch.

        TILE_* maps each program to an independent query tile. Unlike padding all
        requests to max_seqlen, every scheduled tile contains useful query rows and
        tiles from short and long requests can execute concurrently across SMs.
        """
        tile = tl.program_id(0)
        h = tl.program_id(1)
        seq_start = tl.load(TILE_SEQ_START + tile)
        q_start = tl.load(TILE_Q_START + tile)
        seq_len = tl.load(TILE_SEQ_LEN + tile)

        scale = scale.to(tl.float32)
        eps = eps.to(tl.float32)
        offs_m = q_start + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, DK)
        m_valid = offs_m < seq_len
        q_idx = seq_start + offs_m
        q = tl.load(
            Q + q_idx[:, None] * stride_qt + h * stride_qh
            + offs_d[None, :] * stride_qd,
            mask=m_valid[:, None], other=0.0,
        ).to(DOT_DTYPE)

        # Fresh causal prompts have exactly local_query_position + 1 live keys.
        n_i = (offs_m + 1).to(tl.float32)
        if WINDOW > 0:
            n_clamp = tl.maximum(tl.minimum(n_i, float(WINDOW)), 1.0)
        else:
            n_clamp = tl.maximum(n_i, 1.0)
        spg = tl.load(SPG + h).to(tl.float32)
        sps = tl.load(SPS + h).to(tl.float32)
        beta = tl.load(BETA + h).to(tl.float32)
        nb = tl.load(NULLBASE + h).to(tl.float32)
        temp = 1.0 + spg * tl.log(n_clamp)
        nullv = nb + sps * tl.sqrt(tl.log(n_clamp + 1.0))

        m_i = tl.full([BLOCK_M], -1e38, tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        q2_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, DK], tl.float32)

        # A tile only needs keys through its last query. The runtime bound is
        # different for every tile/request and avoids max-length padding work.
        hi = tl.minimum(q_start + BLOCK_M, seq_len)
        if WINDOW > 0:
            lo = tl.maximum(q_start - WINDOW, 0)
            lo = (lo // BLOCK_N) * BLOCK_N
        else:
            lo = 0
        for start_n in range(lo, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_valid = offs_n < seq_len
            k_idx = seq_start + offs_n
            k = tl.load(
                K + k_idx[:, None] * stride_kt + h * stride_kh
                + offs_d[None, :] * stride_kd,
                mask=n_valid[:, None], other=0.0,
            ).to(DOT_DTYPE)
            v = tl.load(
                V + k_idx[:, None] * stride_vt + h * stride_vh
                + offs_d[None, :] * stride_vd,
                mask=n_valid[:, None], other=0.0,
            ).to(DOT_DTYPE)
            sig = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * scale
            a = sig * temp[:, None]
            valid = offs_n[None, :] < n_i[:, None]
            if WINDOW > 0:
                valid = valid & (offs_n[None, :] >= (n_i[:, None] - WINDOW))
            a = tl.where(valid, a, -1e38).to(tl.float32)

            m_new = tl.maximum(m_i, tl.max(a, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.where(valid, tl.exp(a - m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, 1)
            q2_i = q2_i * alpha * alpha + tl.sum(p * p, 1)
            acc = acc * alpha[:, None] + tl.dot(
                p.to(DOT_DTYPE), v, input_precision=INPUT_PRECISION)
            m_i = m_new

        a_n = temp * nullv
        m_new = tl.maximum(m_i, a_n)
        alpha = tl.exp(m_i - m_new)
        l_i *= alpha
        q2_i *= alpha * alpha
        acc *= alpha[:, None]
        m_i = m_new
        p_n = tl.exp(a_n - m_i)
        z = l_i + p_n
        vn = tl.load(VNULL + h * DK + offs_d).to(tl.float32)
        s = acc + p_n[:, None] * vn[None, :]
        c = s / tl.maximum(tl.sqrt(tl.sum(s * s, 1)), eps)[:, None]
        n_eff = l_i * l_i / tl.maximum(q2_i, 1.0e-30)
        m_eff = n_eff * (l_i / tl.maximum(z, eps))
        mag = 2.0 * tl.sigmoid(2.0 * beta * tl.log(1.0 + m_eff)) - 1.0

        tl.store(
            C + q_idx[:, None] * stride_ct + h * stride_ch
            + offs_d[None, :] * stride_cd,
            c, mask=m_valid[:, None],
        )
        tl.store(MAG + q_idx * stride_mt + h * stride_mh, mag, mask=m_valid)


def _softplus(x):
    return F.softplus(x)


_DOT = {torch.float32: tl.float32, torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16} if HAS_TRITON else {}


def _dtype_meta(dtype):
    """(triton dot dtype, input_precision, is_fp32). bf16/fp16 -> tensor-core dots."""
    if dtype == torch.float32:
        return tl.float32, "ieee", True
    return _DOT.get(dtype, tl.bfloat16), "ieee", False


_PROFILE_ALIASES = {
    "auto": "auto",
    "default": "auto",
    "l4": "l4",
    "cuda": "l4",
    "small": "small",
    "conservative": "small",
    "low_smem": "small",
    "hip": "small",
    "rocm": "small",
    "amd": "small",
    "large": "large",
    "high_smem": "large",
    "a100": "large",
    "h100": "large",
    "hopper": "large",
}


def _smem_optin(props):
    return int(getattr(props, "shared_memory_per_block_optin",
                       getattr(props, "shared_memory_per_block", 0)) or 0)


def _polar_tuning_profile(device=None):
    """Return the launch-config profile for this device.

    Override with ATMA_POLAR_TRITON_PROFILE={l4,small,large,auto} or the shorter
    ATMA_POLAR_TRITON_CFG alias. The L4 profile preserves the original measured
    defaults; `small` is for lower-smem or ROCm devices, and `large` uses wider
    tiles on A100/H100-class parts.
    """
    requested = os.environ.get("ATMA_POLAR_TRITON_PROFILE") or os.environ.get("ATMA_POLAR_TRITON_CFG")
    if requested:
        key = requested.strip().lower().replace("-", "_")
        if key not in _PROFILE_ALIASES:
            valid = ", ".join(sorted(k for k in _PROFILE_ALIASES if k != "auto"))
            raise ValueError(f"unknown polar Triton profile {requested!r}; valid: auto, {valid}")
        mapped = _PROFILE_ALIASES[key]
        if mapped != "auto":
            return mapped

    if not HAS_TRITON or not torch.cuda.is_available():
        return "l4"
    if getattr(torch.version, "hip", None):
        return "small"

    dev = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    idx = torch.cuda.current_device() if dev.index is None else dev.index
    props = torch.cuda.get_device_properties(idx)
    name = props.name.lower()
    smem = _smem_optin(props)
    major = getattr(props, "major", 0)

    if " l4" in f" {name}" or name.endswith("l4"):
        return "l4"
    if major >= 9 or smem >= 160 * 1024:
        return "large"
    if smem and smem < 96 * 1024:
        return "small"
    return "l4"


def _fwd_config(dk, is_fp32, device=None):
    """Block sizes / pipelining selected by GPU profile.

    The L4 profile is the original measured default (~99 KB opt-in shared memory).
    16-bit tiles are half the size of fp32 tiles, so they can run wider / deeper.
    """
    profile = _polar_tuning_profile(device)
    if profile == "small":
        if is_fp32:
            if dk >= 128:
                return dict(block_m=32, block_n=32, num_warps=4, num_stages=1)
            return dict(block_m=64, block_n=32, num_warps=4, num_stages=2)
        if dk >= 128:
            return dict(block_m=64, block_n=64, num_warps=4, num_stages=2)
        return dict(block_m=64, block_n=64, num_warps=4, num_stages=2)

    if profile == "large":
        if is_fp32:
            if dk >= 128:
                return dict(block_m=64, block_n=64, num_warps=4, num_stages=2)
            return dict(block_m=128, block_n=64, num_warps=4, num_stages=2)
        if dk >= 128:
            return dict(block_m=128, block_n=128, num_warps=8, num_stages=2)
        return dict(block_m=128, block_n=128, num_warps=4, num_stages=3)

    if is_fp32:
        if dk >= 128:
            return dict(block_m=64, block_n=32, num_warps=4, num_stages=2)
        return dict(block_m=64, block_n=64, num_warps=4, num_stages=2)
    # bf16 / fp16
    if dk >= 128:
        return dict(block_m=128, block_n=64, num_warps=8, num_stages=2)
    return dict(block_m=128, block_n=64, num_warps=4, num_stages=3)


def _bwd_config(dk, is_fp32, device=None):
    """Backward keeps an extra resident gs tile (and dk/dv accumulators for the
    dk/dv kernel), so it needs smaller tiles than forward. Separate dq / dk-dv
    launch params."""
    profile = _polar_tuning_profile(device)
    if profile == "small":
        if is_fp32:
            if dk >= 128:
                return dict(
                    dq=dict(block_m=32, block_n=32, num_warps=4, num_stages=1),
                    kv=dict(block_m=32, block_n=32, num_warps=4, num_stages=1),
                )
            return dict(
                dq=dict(block_m=64, block_n=32, num_warps=4, num_stages=1),
                kv=dict(block_m=32, block_n=32, num_warps=4, num_stages=1),
            )
        if dk >= 128:
            return dict(
                dq=dict(block_m=64, block_n=32, num_warps=4, num_stages=2),
                kv=dict(block_m=64, block_n=32, num_warps=4, num_stages=2),
            )
        return dict(
            dq=dict(block_m=64, block_n=64, num_warps=4, num_stages=2),
            kv=dict(block_m=64, block_n=32, num_warps=4, num_stages=2),
        )

    if profile == "large":
        if is_fp32:
            if dk >= 128:
                return dict(
                    dq=dict(block_m=64, block_n=64, num_warps=4, num_stages=2),
                    kv=dict(block_m=64, block_n=32, num_warps=4, num_stages=2),
                )
            return dict(
                dq=dict(block_m=128, block_n=64, num_warps=4, num_stages=2),
                kv=dict(block_m=64, block_n=64, num_warps=4, num_stages=2),
            )
        if dk >= 128:
            return dict(
                dq=dict(block_m=64, block_n=128, num_warps=4, num_stages=2),
                kv=dict(block_m=64, block_n=64, num_warps=4, num_stages=2),
            )
        return dict(
            dq=dict(block_m=128, block_n=64, num_warps=4, num_stages=2),
            kv=dict(block_m=64, block_n=64, num_warps=4, num_stages=2),
        )

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


def _decode_config(dk, is_fp32, device=None):
    profile = _polar_tuning_profile(device)
    if profile == "small":
        return dict(block_n=32 if dk >= 128 else 64, num_warps=4, num_stages=1)
    if profile == "large":
        if is_fp32:
            return dict(block_n=64, num_warps=4, num_stages=2)
        return dict(block_n=128 if dk >= 128 else 64, num_warps=4, num_stages=2)
    return dict(block_n=64, num_warps=4, num_stages=2)


def _polar_forward(q, k, v, n_keys, v_null, null_base, null_slope_raw,
                   len_gain_raw, mag_beta_raw, eps, is_causal, input_precision,
                   block_m=None, block_n=None, num_warps=None, num_stages=None, window=None,
                   preserve_length=False):
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
    cfg = _fwd_config(dk, is_fp32, dev)
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
        PRESERVE_LENGTH=bool(preserve_length),
        num_warps=num_warps, num_stages=num_stages,
    )

    # Diagnostic probe (off by default → skipped entirely). The kernel doesn't emit
    # n_eff / w_null, so recover them from the saved stats (same algebra as the
    # backward preamble) and feed the shared sink in model.blocks.
    from model import blocks as _blocks
    if _blocks._PROBE is not None:
        # temp/null use the windowed count when a window is active (matches the kernel).
        nf = (n_keys if window is None or preserve_length
              else torch.minimum(n_keys, n_keys.new_tensor(float(window)))).clamp(min=1.0)  # (Tq,)
        logn = torch.log(nf).view(1, Tq)
        t = 1.0 + spg.view(H, 1) * logn                              # (H, Tq)
        nu = nb.view(H, 1) + sps.view(H, 1) * torch.sqrt(torch.log(nf + 1.0)).view(1, Tq)
        p_n = torch.exp((t * nu).view(1, H, Tq) - M)                 # (B, H, Tq)
        Z = L + p_n
        n_eff = L * L / Q2.clamp_min(1.0e-30)
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


@torch.no_grad()
def polar_attention_packed_fwd(
    q, k, v, tile_seq_starts, tile_q_starts, tile_seq_lens, *,
    v_null, null_base, null_slope_raw, len_gain_raw, mag_beta_raw,
    eps=1e-6, input_precision="ieee", window=None,
):
    """Grouped fresh-prefill Polar attention for packed variable-length requests.

    q/k/v use ``[total_tokens, heads, head_dim]`` layout. ``tile_*`` contains one
    entry per BLOCK_M query tile and is prepared once with the serving context.
    The route is intentionally fresh-causal only; chunked/prefix requests retain
    the established per-sequence implementation.
    """
    if not HAS_TRITON or not q.is_cuda:
        raise RuntimeError("packed Polar attention requires CUDA and Triton")
    if q.ndim != 3 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, and v must have identical [tokens, heads, dim] shapes")
    total, heads, dk = q.shape
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("packed Polar route is enabled only for 16-bit inference")
    if dk & (dk - 1):
        raise ValueError(f"head_dim (dk={dk}) must be a power of two")
    if not (tile_seq_starts.numel() == tile_q_starts.numel() == tile_seq_lens.numel()):
        raise ValueError("packed Polar tile-map tensors must have equal lengths")

    dot_dtype, ip, is_fp32 = _dtype_meta(q.dtype)
    if is_fp32:
        ip = input_precision
    cfg = _fwd_config(dk, is_fp32, q.device)
    # The serving context builds its reusable map at the L40S-tuned 128 rows.
    # Keep this explicit so an environment profile override cannot silently make
    # the map and launch disagree.
    block_m, block_n = 128, cfg["block_n"]
    spg = _softplus(len_gain_raw.float()).contiguous()
    sps = _softplus(null_slope_raw.float()).contiguous()
    beta = _softplus(mag_beta_raw.float()).contiguous()
    nb = null_base.float().contiguous()
    vn = v_null.float().contiguous()
    tile_seq_starts = tile_seq_starts.to(device=q.device, dtype=torch.int32).contiguous()
    tile_q_starts = tile_q_starts.to(device=q.device, dtype=torch.int32).contiguous()
    tile_seq_lens = tile_seq_lens.to(device=q.device, dtype=torch.int32).contiguous()
    c = torch.empty_like(q)
    mag = torch.empty((total, heads), device=q.device, dtype=q.dtype)
    _polar_packed_fwd_kernel[(tile_seq_starts.numel(), heads)](
        q, k, v, tile_seq_starts, tile_q_starts, tile_seq_lens,
        vn, spg, nb, sps, beta, c, mag,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        mag.stride(0), mag.stride(1),
        heads, 1.0 / math.sqrt(dk), eps,
        BLOCK_M=block_m, BLOCK_N=block_n, DK=dk,
        INPUT_PRECISION=ip, DOT_DTYPE=dot_dtype,
        WINDOW=0 if window is None else int(window),
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
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
        scale, eps,
        BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
        BLOCK_N: tl.constexpr, DK: tl.constexpr,
        G: tl.constexpr, GP: tl.constexpr, DOT_DTYPE: tl.constexpr,
        WINDOW: tl.constexpr,
    ):
        # One program per (sequence, KV head): the G query heads of the GQA group are
        # the rows of a (GP, ...) tile (GP = G padded to >=16 for tl.dot), so each
        # cached K/V byte is read ONCE per group instead of once per query head
        # (a `groups`-fold cut of decode's dominant memory traffic).
        b = tl.program_id(0)
        kv_h = tl.program_id(1)
        scale = scale.to(tl.float32)     # python-float args are fp64 on some Triton versions
        eps = eps.to(tl.float32)
        offs_d = tl.arange(0, DK)
        offs_g = tl.arange(0, GP)                          # query-head rows (padded)
        g_valid = offs_g < G
        heads = tl.where(g_valid, kv_h * G + offs_g, 0)    # global query-head ids (0-clamped pad)

        q = tl.load(Q + b * stride_qb + heads[:, None] * stride_qh + offs_d[None, :] * stride_qd,
                    mask=g_valid[:, None], other=0.0).to(DOT_DTYPE)        # (GP, DK)

        n_i = tl.load(CONTEXT_LENS + b)                  # int32 valid-key count
        # WINDOW>0: causal sliding window — the query sees only its last WINDOW keys,
        # so temp/null use the capped count min(n_i, WINDOW) and the key loop masks
        # positions older than n_i - WINDOW. WINDOW==0 disables it (full context).
        if WINDOW > 0:
            n_f = tl.maximum(tl.minimum(n_i.to(tl.float32), float(WINDOW)), 1.0)
        else:
            n_f = tl.maximum(n_i.to(tl.float32), 1.0)
        spg = tl.load(SPG + heads, mask=g_valid, other=0.0).to(tl.float32)   # (GP,)
        sps = tl.load(SPS + heads, mask=g_valid, other=0.0).to(tl.float32)
        beta = tl.load(BETA + heads, mask=g_valid, other=0.0).to(tl.float32)
        nb = tl.load(NULLBASE + heads, mask=g_valid, other=0.0).to(tl.float32)
        temp = 1.0 + spg * tl.log(n_f)
        nullv = nb + sps * tl.sqrt(tl.log(n_f + 1.0))

        m_i = tl.full([GP], -1e38, tl.float32)
        l_i = tl.zeros([GP], tl.float32)
        q2_i = tl.zeros([GP], tl.float32)
        acc = tl.zeros([GP, DK], tl.float32)

        # dynamic loop bounds: only scan the live context (and, with a window, skip
        # keys older than the band) — in CUDA-graph mode MAX_BLOCKS*BLOCK_SIZE is the
        # padded max_model_len, which would otherwise be scanned in full every step.
        hi = tl.minimum(tl.cdiv(n_i, BLOCK_N) * BLOCK_N, MAX_BLOCKS * BLOCK_SIZE)
        if WINDOW > 0:
            lo = (tl.maximum(n_i - WINDOW, 0) // BLOCK_N) * BLOCK_N
        else:
            lo = 0
        for start in range(lo, hi, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)        # global key positions
            valid = offs_n < n_i
            if WINDOW > 0:
                valid = valid & (offs_n >= (n_i - WINDOW))  # older than window
            blk = offs_n // BLOCK_SIZE                     # logical block index
            within = offs_n % BLOCK_SIZE
            phys = tl.load(BLOCK_TABLES + b * stride_btb + blk * stride_btn,
                           mask=valid, other=0).to(tl.int32)
            k_ptr = (K_CACHE + phys[:, None] * stride_kc_blk + within[:, None] * stride_kc_pos
                     + kv_h * stride_kc_h + offs_d[None, :] * stride_kc_d)
            v_ptr = (V_CACHE + phys[:, None] * stride_vc_blk + within[:, None] * stride_vc_pos
                     + kv_h * stride_vc_h + offs_d[None, :] * stride_vc_d)
            k = tl.load(k_ptr, mask=valid[:, None], other=0.0).to(DOT_DTYPE)   # (BLOCK_N, DK)
            v = tl.load(v_ptr, mask=valid[:, None], other=0.0).to(DOT_DTYPE)

            sig = tl.dot(q, tl.trans(k), input_precision="ieee") * scale       # (GP, BLOCK_N)
            a = sig * temp[:, None]
            a = tl.where(valid[None, :] & g_valid[:, None], a, -1e38).to(tl.float32)
            m_new = tl.maximum(m_i, tl.max(a, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(a - m_new[:, None])
            p = tl.where(valid[None, :], p, 0.0)
            l_i = l_i * alpha + tl.sum(p, 1)
            q2_i = q2_i * alpha * alpha + tl.sum(p * p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(DOT_DTYPE), v, input_precision="ieee")
            m_i = m_new

        # fold null sink
        a_n = temp * nullv                                 # (GP,)
        m_new = tl.maximum(m_i, a_n)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha
        q2_i = q2_i * alpha * alpha
        acc = acc * alpha[:, None]
        m_i = m_new
        p_n = tl.exp(a_n - m_i)
        Z = l_i + p_n

        vnull = tl.load(VNULL + heads[:, None] * DK + offs_d[None, :],
                        mask=g_valid[:, None], other=0.0).to(tl.float32)        # (GP, DK)
        s = acc + p_n[:, None] * vnull
        snorm = tl.maximum(tl.sqrt(tl.sum(s * s, 1)), eps)
        c = s / snorm[:, None]
        n_eff = l_i * l_i / tl.maximum(q2_i, 1.0e-30)
        m_eff = n_eff * (l_i / tl.maximum(Z, eps))
        mag = 2.0 * tl.sigmoid(2.0 * (beta * tl.log(1.0 + m_eff))) - 1.0

        tl.store(C_OUT + b * stride_cb + heads[:, None] * stride_ch + offs_d[None, :] * stride_cd,
                 c, mask=g_valid[:, None])
        tl.store(MAG_OUT + b * stride_mb + heads * stride_mh, mag, mask=g_valid)


@torch.no_grad()
def polar_attention_decode(q, k_cache, v_cache, block_tables, context_lens, *,
                           v_null, null_base, null_slope_raw, len_gain_raw, mag_beta_raw,
                           eps=1e-6, block_n=None, num_warps=None, num_stages=None, window=None):
    """Paged polar decode. One query per sequence over its cached context.

    q            : (B, H, dk)  current-token query (KV heads NOT expanded; GQA done in-kernel)
    k_cache      : (num_blocks, block_size, num_kv_heads, dk)  paged cache (contiguous)
    v_cache      : same shape as k_cache
    block_tables : (B, max_blocks) int32  logical->physical block map
    context_lens : (B,) int32  valid key count per sequence (incl. current token)
    window       : (int, optional) causal sliding window — the query attends to only its
                   last `window` cached keys (temp/null use min(context_len, window)),
                   matching polar_attention_fwd(window=...).

    Returns c (B, H, dk), mag (B, H). CUDA-graph capturable (no host sync, fixed shapes).
    Matches model.blocks.polar_reduce for the equivalent dense computation to ~1e-3 (bf16).

    GQA-aware: one program per (sequence, KV head) computes all `H // num_kv_heads`
    query heads of the group, so the paged K/V is read once per group (not per head).
    """
    B, H, dk = q.shape
    num_kv_heads = k_cache.shape[2]
    groups = H // num_kv_heads
    gp = max(16, triton.next_power_of_2(groups))   # tl.dot needs M >= 16; pad + mask
    max_blocks = block_tables.shape[1]
    block_size = k_cache.shape[1]
    out_dtype = q.dtype
    dev = q.device
    dot_dtype, _, is_fp32 = _dtype_meta(out_dtype)
    cfg = _decode_config(dk, is_fp32, dev)
    block_n = block_n or cfg["block_n"]
    num_warps = num_warps or cfg["num_warps"]
    num_stages = num_stages or cfg["num_stages"]

    spg = F.softplus(len_gain_raw.float()).contiguous()
    sps = F.softplus(null_slope_raw.float()).contiguous()
    beta = F.softplus(mag_beta_raw.float()).contiguous()
    nb = null_base.float().contiguous()
    vnull = v_null.float().contiguous()
    q = q.contiguous()

    c = torch.empty((B, H, dk), device=dev, dtype=torch.float32)
    mag = torch.empty((B, H), device=dev, dtype=torch.float32)
    scale = 1.0 / math.sqrt(dk)

    grid = (B, num_kv_heads)
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
        scale, eps,
        BLOCK_SIZE=block_size, MAX_BLOCKS=max_blocks, BLOCK_N=block_n, DK=dk,
        G=groups, GP=gp, DOT_DTYPE=dot_dtype,
        WINDOW=(0 if window is None else int(window)),
        num_warps=num_warps, num_stages=num_stages,
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
        WINDOW: tl.constexpr,
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
        if WINDOW > 0:
            lo = tl.maximum(pid_m * BLOCK_M - WINDOW, 0)
            lo = (lo // BLOCK_N) * BLOCK_N
        else:
            lo = 0
        for start_n in range(lo, hi, BLOCK_N):
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
            valid = offs_n[None, :] < n_i[:, None]                            # future
            if WINDOW > 0:
                valid = valid & (offs_n[None, :] >= (n_i[:, None] - WINDOW))  # older than window
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
        WINDOW: tl.constexpr,
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
        if WINDOW > 0:
            hi = tl.minimum((pid_n + 1) * BLOCK_N + WINDOW, Tq)
        else:
            hi = Tq
        for start_m in range(lo, hi, BLOCK_M):
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
            valid = offs_n[None, :] < n_i[:, None]                            # future
            if WINDOW > 0:
                valid = valid & (offs_n[None, :] >= (n_i[:, None] - WINDOW))  # older than window
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
                    input_precision, window=None, preserve_length=False):
    """Backward: cheap per-query preamble (PyTorch) + dq / dk,dv loops (Triton).

    window (int, optional): causal sliding window — temp/null use the windowed count
    min(n_keys, window) and the dq / dk,dv loops mask keys older than n_keys - window,
    mirroring the forward kernel's WINDOW path so the band backward is exact."""
    B, H, Tq, dk = q.shape
    Tk = k.shape[2]
    dev = q.device
    fdt = torch.float32

    gc = gc.to(fdt)
    gm = gm.to(fdt)
    n_raw = n_keys.to(dev).to(fdt)                                  # raw, for the causal mask
    n_cnt = (
        n_raw
        if window is None or preserve_length
        else torch.minimum(n_raw, n_raw.new_tensor(float(window)))
    )
    n = n_cnt.clamp(min=1.0)                                        # windowed clamp, for temp/null/param-grads
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
    # Z contains the global maximum term and is therefore >= 1.  Q2 can be
    # tiny solely because the null term set that maximum; it must not use the
    # vector-normalization epsilon or the participation ratio ceases to be
    # invariant to the online softmax shift.
    stats_tiny = 1.0e-30
    Zc, Q2c = Z.clamp_min(stats_tiny), Q2.clamp_min(stats_tiny)
    n_eff = (L * L) / Q2c
    m_eff = n_eff * (L / Zc)
    log1p_m = torch.log1p(m_eff)
    mag = torch.tanh(beta * log1p_m)

    # magnitude path -> (gL, gQ2, gZ via p_n)
    gme = gm * beta * (1.0 - mag * mag) / (1.0 + m_eff)
    dm_dL = torch.where(L > 0, 3.0 * m_eff / L, torch.zeros_like(L))
    dm_dQ2 = -(m_eff / Q2c) * (Q2 > stats_tiny)
    dm_dZ = -(m_eff / Zc) * (Z > stats_tiny)
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

    cfg = _bwd_config(dk, is_fp32, dev)
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
    win = 0 if window is None else int(window)
    grid_q = (triton.cdiv(Tq, cq["block_m"]), B * H)
    _polar_bwd_dq_kernel[grid_q](
        q, k, v, gs, gL, gQ2, temp_ht, M, n_keys_f, dq, dt_real, *common,
        BLOCK_M=cq["block_m"], BLOCK_N=cq["block_n"], DK=dk, IS_CAUSAL=is_causal,
        INPUT_PRECISION=ip, DOT_DTYPE=dot_dtype, WINDOW=win,
        num_warps=cq["num_warps"], num_stages=cq["num_stages"],
    )
    grid_kv = (triton.cdiv(Tk, ckv["block_n"]), B * H)
    _polar_bwd_dkdv_kernel[grid_kv](
        q, k, v, gs, gL, gQ2, temp_ht, M, n_keys_f, dk_out, dv_out, *common,
        BLOCK_M=ckv["block_m"], BLOCK_N=ckv["block_n"], DK=dk, IS_CAUSAL=is_causal,
        INPUT_PRECISION=ip, DOT_DTYPE=dot_dtype, WINDOW=win,
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
                len_gain_raw, mag_beta_raw, eps, is_causal, input_precision, window,
                preserve_length):
        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        c, mag, M, L, Q2, s = _polar_forward(
            q, k, v, n_keys, v_null, null_base, null_slope_raw,
            len_gain_raw, mag_beta_raw, eps, is_causal, input_precision, window=window,
            preserve_length=preserve_length)
        ctx.save_for_backward(q, k, v, n_keys, v_null, null_base, null_slope_raw,
                              len_gain_raw, mag_beta_raw, M, L, Q2, s)
        ctx.eps = eps
        ctx.is_causal = is_causal
        ctx.input_precision = input_precision
        ctx.window = window
        ctx.preserve_length = preserve_length
        return c, mag

    @staticmethod
    def backward(ctx, gc, gm):
        (q, k, v, n_keys, v_null, null_base, null_slope_raw,
         len_gain_raw, mag_beta_raw, M, L, Q2, s) = ctx.saved_tensors
        (dq, dk_out, dv_out, grad_v_null, grad_null_base, grad_null_slope,
         grad_len_gain, grad_mag_beta) = _polar_backward(
            gc, gm, q, k, v, n_keys, v_null, null_base, null_slope_raw,
            len_gain_raw, mag_beta_raw, M, L, Q2, s,
            ctx.eps, ctx.is_causal, ctx.input_precision, window=ctx.window,
            preserve_length=ctx.preserve_length)

        def cast(x, ref):
            return x.to(ref.dtype)

        return (cast(dq, q), cast(dk_out, k), cast(dv_out, v), None,
                cast(grad_v_null, v_null), cast(grad_null_base, null_base),
                cast(grad_null_slope, null_slope_raw), cast(grad_len_gain, len_gain_raw),
                cast(grad_mag_beta, mag_beta_raw), None, None, None, None, None)


def polar_attention(q, k, v, n_keys, *, v_null, null_base, null_slope_raw,
                    len_gain_raw, mag_beta_raw, eps=1e-6, is_causal=True,
                    input_precision="ieee", window=None, preserve_length=False):
    """Autograd-aware FlashAttention-style polar attention. q,k,v: (B,H,T,dk)
    with GQA KV-heads already expanded to H. Drop-in for
    ``model.blocks.polar_attention_online`` (the streaming PyTorch oracle).

    window (int, optional): trainable causal sliding window — each query attends to only
    its last `window` keys (temp/null use the windowed count). The backward models the
    band, so unlike ``polar_attention_fwd(window=...)`` this is usable in training.

    Returns (c, mag): direction unit vectors (B,H,T,dk) and bounded magnitude (B,H,T).
    """
    return PolarAttentionTriton.apply(
        q, k, v, n_keys, v_null, null_base, null_slope_raw,
        len_gain_raw, mag_beta_raw, eps, is_causal, input_precision, window,
        preserve_length)


# ---------------------------------------------------------------------------
# Foveal sparse training kernel
# ---------------------------------------------------------------------------
# Each query page attends to (a) its exact token-level causal sliding window and
# (b) a query-dependent list of completed remote KV pages.  Unlike the paged
# decode kernel above, this path skips unselected pages and implements backward.

if HAS_TRITON:

    @triton.jit
    def _polar_sparse_fwd_kernel(
        Q, K, V, PAGE_INDICES, PAGE_COUNTS, VNULL, SPG, NULLBASE, SPS, BETA,
        C, MAG, M_OUT, L_OUT, Q2_OUT, S_OUT,
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_pib, stride_piq, stride_pis,
        stride_pcb, stride_pcq,
        stride_cb, stride_ch, stride_ct, stride_cd,
        stride_sb, stride_sh, stride_st, stride_sd,
        stride_mb, stride_mh, stride_mt,
        B, H, T, scale, eps,
        PAGE_SIZE: tl.constexpr, LOCAL_WINDOW: tl.constexpr,
        LOCAL_BLOCKS: tl.constexpr, REMOTE_CAPACITY: tl.constexpr,
        DK: tl.constexpr, INPUT_PRECISION: tl.constexpr, DOT_DTYPE: tl.constexpr,
    ):
        query_page = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        scale = scale.to(tl.float32)
        eps = eps.to(tl.float32)

        offs_m = query_page * PAGE_SIZE + tl.arange(0, PAGE_SIZE)
        offs_d = tl.arange(0, DK)
        m_valid = offs_m < T
        q = tl.load(
            Q + b * stride_qb + h * stride_qh
            + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
            mask=m_valid[:, None], other=0.0,
        ).to(DOT_DTYPE)

        route_count = tl.load(
            PAGE_COUNTS + b * stride_pcb + query_page * stride_pcq
        ).to(tl.int32)
        route_count = tl.minimum(tl.maximum(route_count, 0), REMOTE_CAPACITY)
        # Preserve the pretrained Polar calibration: temperature and the null
        # floor describe the full visible prefix even though only selected keys
        # are reduced.  Changing this to sparse-support cardinality would shift
        # both channels immediately at the dense-to-sparse handoff.
        n_i = (offs_m + 1).to(tl.float32)
        n_clamp = tl.maximum(n_i, 1.0)
        spg = tl.load(SPG + h).to(tl.float32)
        sps = tl.load(SPS + h).to(tl.float32)
        beta = tl.load(BETA + h).to(tl.float32)
        nb = tl.load(NULLBASE + h).to(tl.float32)
        logn = tl.log(n_clamp)
        temp = 1.0 + spg * logn
        nullv = nb + sps * tl.sqrt(tl.log(n_clamp + 1.0))

        m_i = tl.full([PAGE_SIZE], -1e38, tl.float32)
        l_i = tl.zeros([PAGE_SIZE], tl.float32)
        q2_i = tl.zeros([PAGE_SIZE], tl.float32)
        acc = tl.zeros([PAGE_SIZE, DK], tl.float32)

        # Token-exact local band.  LOCAL_BLOCKS includes the potentially partial
        # oldest page and the current query page.
        for local_slot in range(0, LOCAL_BLOCKS):
            key_page = query_page - (LOCAL_BLOCKS - 1) + local_slot
            offs_n = key_page * PAGE_SIZE + tl.arange(0, PAGE_SIZE)
            n_valid = (key_page >= 0) & (offs_n >= 0) & (offs_n < T)
            k = tl.load(
                K + b * stride_kb + h * stride_kh
                + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
                mask=n_valid[:, None], other=0.0,
            ).to(DOT_DTYPE)
            v = tl.load(
                V + b * stride_vb + h * stride_vh
                + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
                mask=n_valid[:, None], other=0.0,
            ).to(DOT_DTYPE)
            sig = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * scale
            valid = (n_valid[None, :] & (offs_n[None, :] <= offs_m[:, None])
                     & (offs_n[None, :] > offs_m[:, None] - LOCAL_WINDOW))
            a = tl.where(valid, sig * temp[:, None], -1e38).to(tl.float32)
            m_new = tl.maximum(m_i, tl.max(a, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.where(valid, tl.exp(a - m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, 1)
            q2_i = q2_i * alpha * alpha + tl.sum(p * p, 1)
            acc = acc * alpha[:, None] + tl.dot(
                p.to(DOT_DTYPE), v, input_precision=INPUT_PRECISION
            )
            m_i = m_new

        # Runtime loop: do not unroll the maximum capacity into one enormous
        # program or execute masked tl.dot instructions for unused route slots.
        for remote_slot in range(0, route_count):
            selected = True
            key_page = tl.load(
                PAGE_INDICES + b * stride_pib + query_page * stride_piq
                + remote_slot * stride_pis,
                mask=selected, other=0,
            ).to(tl.int32)
            offs_n = key_page * PAGE_SIZE + tl.arange(0, PAGE_SIZE)
            n_valid = selected & (key_page >= 0) & (offs_n >= 0) & (offs_n < T)
            k = tl.load(
                K + b * stride_kb + h * stride_kh
                + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
                mask=n_valid[:, None], other=0.0,
            ).to(DOT_DTYPE)
            v = tl.load(
                V + b * stride_vb + h * stride_vh
                + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
                mask=n_valid[:, None], other=0.0,
            ).to(DOT_DTYPE)
            sig = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * scale
            # Enforce completed, non-local pages even if a malformed route is
            # supplied.  Valid routes produced by Foveal already satisfy this.
            valid = (n_valid[None, :] & (offs_n[None, :] <= offs_m[:, None])
                     & (offs_n[None, :] <= offs_m[:, None] - LOCAL_WINDOW))
            a = tl.where(valid, sig * temp[:, None], -1e38).to(tl.float32)
            m_new = tl.maximum(m_i, tl.max(a, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.where(valid, tl.exp(a - m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, 1)
            q2_i = q2_i * alpha * alpha + tl.sum(p * p, 1)
            acc = acc * alpha[:, None] + tl.dot(
                p.to(DOT_DTYPE), v, input_precision=INPUT_PRECISION
            )
            m_i = m_new

        a_n = temp * nullv
        m_new = tl.maximum(m_i, a_n)
        alpha = tl.exp(m_i - m_new)
        l_i *= alpha
        q2_i *= alpha * alpha
        acc *= alpha[:, None]
        m_i = m_new
        p_n = tl.exp(a_n - m_i)
        z = l_i + p_n
        vnull = tl.load(VNULL + h * DK + offs_d).to(tl.float32)
        s = acc + p_n[:, None] * vnull[None, :]
        snorm = tl.maximum(tl.sqrt(tl.sum(s * s, 1)), eps)
        c = s / snorm[:, None]
        # Do not reuse the direction epsilon here: Q2 is max-shifted and can
        # legitimately be tiny when the null sink is the largest logit.
        n_eff = l_i * l_i / tl.maximum(q2_i, 1.0e-30)
        m_eff = n_eff * (l_i / tl.maximum(z, eps))
        mag = 2.0 * tl.sigmoid(2.0 * beta * tl.log(1.0 + m_eff)) - 1.0

        c_ptrs = (C + b * stride_cb + h * stride_ch
                  + offs_m[:, None] * stride_ct + offs_d[None, :] * stride_cd)
        s_ptrs = (S_OUT + b * stride_sb + h * stride_sh
                  + offs_m[:, None] * stride_st + offs_d[None, :] * stride_sd)
        tl.store(c_ptrs, c, mask=m_valid[:, None])
        tl.store(s_ptrs, s, mask=m_valid[:, None])
        mbase = b * stride_mb + h * stride_mh + offs_m * stride_mt
        tl.store(MAG + mbase, mag, mask=m_valid)
        tl.store(M_OUT + mbase, m_i, mask=m_valid)
        tl.store(L_OUT + mbase, l_i, mask=m_valid)
        tl.store(Q2_OUT + mbase, q2_i, mask=m_valid)


    @triton.jit
    def _polar_sparse_bwd_kernel(
        Q, K, V, PAGE_INDICES, PAGE_COUNTS, GS, GL, GQ2, TEMP, M,
        DQ, DK_OUT, DV_OUT, DT,
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_pib, stride_piq, stride_pis,
        stride_pcb, stride_pcq,
        stride_gsb, stride_gsh, stride_gst, stride_gsd,
        stride_mb, stride_mh, stride_mt,
        stride_tb, stride_th, stride_tt,
        B, H, T, scale,
        PAGE_SIZE: tl.constexpr, LOCAL_WINDOW: tl.constexpr,
        LOCAL_BLOCKS: tl.constexpr, REMOTE_CAPACITY: tl.constexpr,
        DK: tl.constexpr, INPUT_PRECISION: tl.constexpr, DOT_DTYPE: tl.constexpr,
        DQ_ONLY: tl.constexpr,
    ):
        query_page = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        scale = scale.to(tl.float32)
        offs_m = query_page * PAGE_SIZE + tl.arange(0, PAGE_SIZE)
        offs_d = tl.arange(0, DK)
        m_valid = offs_m < T
        q = tl.load(
            Q + b * stride_qb + h * stride_qh
            + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
            mask=m_valid[:, None], other=0.0,
        ).to(DOT_DTYPE)
        gs = tl.load(
            GS + b * stride_gsb + h * stride_gsh
            + offs_m[:, None] * stride_gst + offs_d[None, :] * stride_gsd,
            mask=m_valid[:, None], other=0.0,
        ).to(DOT_DTYPE)
        mbase = b * stride_mb + h * stride_mh + offs_m * stride_mt
        g_l = tl.load(GL + mbase, mask=m_valid, other=0.0).to(tl.float32)
        g_q2 = tl.load(GQ2 + mbase, mask=m_valid, other=0.0).to(tl.float32)
        m_i = tl.load(M + mbase, mask=m_valid, other=0.0).to(tl.float32)
        temp = tl.load(
            TEMP + b * stride_tb + h * stride_th + offs_m * stride_tt,
            mask=m_valid, other=1.0,
        ).to(tl.float32)
        route_count = tl.load(
            PAGE_COUNTS + b * stride_pcb + query_page * stride_pcq
        ).to(tl.int32)
        route_count = tl.minimum(tl.maximum(route_count, 0), REMOTE_CAPACITY)
        dq = tl.zeros([PAGE_SIZE, DK], tl.float32)
        dt = tl.zeros([PAGE_SIZE], tl.float32)

        for local_slot in range(0, LOCAL_BLOCKS):
            key_page = query_page - (LOCAL_BLOCKS - 1) + local_slot
            offs_n = key_page * PAGE_SIZE + tl.arange(0, PAGE_SIZE)
            n_valid = (key_page >= 0) & (offs_n >= 0) & (offs_n < T)
            k_ptrs = (K + b * stride_kb + h * stride_kh
                      + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd)
            v_ptrs = (V + b * stride_vb + h * stride_vh
                      + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd)
            k = tl.load(k_ptrs, mask=n_valid[:, None], other=0.0).to(DOT_DTYPE)
            v = tl.load(v_ptrs, mask=n_valid[:, None], other=0.0).to(DOT_DTYPE)
            sig = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * scale
            valid = (n_valid[None, :] & (offs_n[None, :] <= offs_m[:, None])
                     & (offs_n[None, :] > offs_m[:, None] - LOCAL_WINDOW))
            a = tl.where(valid, sig * temp[:, None], -1e38).to(tl.float32)
            p = tl.where(valid, tl.exp(a - m_i[:, None]), 0.0)
            gs_v = tl.dot(gs, tl.trans(v), input_precision=INPUT_PRECISION)
            dldp = gs_v + g_l[:, None] + g_q2[:, None] * (2.0 * p)
            da = tl.where(valid, dldp * p, 0.0)
            dt += tl.sum(da * sig, 1)
            gsig = (da * temp[:, None]).to(DOT_DTYPE)
            dq += tl.dot(gsig, k, input_precision=INPUT_PRECISION) * scale
            if not DQ_ONLY:
                dk = tl.dot(tl.trans(gsig), q, input_precision=INPUT_PRECISION) * scale
                dv = tl.dot(tl.trans(p.to(DOT_DTYPE)), gs, input_precision=INPUT_PRECISION)
                tl.atomic_add(
                    DK_OUT + b * stride_kb + h * stride_kh
                    + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
                    dk, mask=n_valid[:, None],
                )
                tl.atomic_add(
                    DV_OUT + b * stride_vb + h * stride_vh
                    + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
                    dv, mask=n_valid[:, None],
                )

        for remote_slot in range(0, route_count):
            selected = True
            key_page = tl.load(
                PAGE_INDICES + b * stride_pib + query_page * stride_piq
                + remote_slot * stride_pis,
                mask=selected, other=0,
            ).to(tl.int32)
            offs_n = key_page * PAGE_SIZE + tl.arange(0, PAGE_SIZE)
            n_valid = selected & (key_page >= 0) & (offs_n >= 0) & (offs_n < T)
            k_ptrs = (K + b * stride_kb + h * stride_kh
                      + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd)
            v_ptrs = (V + b * stride_vb + h * stride_vh
                      + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd)
            k = tl.load(k_ptrs, mask=n_valid[:, None], other=0.0).to(DOT_DTYPE)
            v = tl.load(v_ptrs, mask=n_valid[:, None], other=0.0).to(DOT_DTYPE)
            sig = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * scale
            valid = (n_valid[None, :] & (offs_n[None, :] <= offs_m[:, None])
                     & (offs_n[None, :] <= offs_m[:, None] - LOCAL_WINDOW))
            a = tl.where(valid, sig * temp[:, None], -1e38).to(tl.float32)
            p = tl.where(valid, tl.exp(a - m_i[:, None]), 0.0)
            gs_v = tl.dot(gs, tl.trans(v), input_precision=INPUT_PRECISION)
            dldp = gs_v + g_l[:, None] + g_q2[:, None] * (2.0 * p)
            da = tl.where(valid, dldp * p, 0.0)
            dt += tl.sum(da * sig, 1)
            gsig = (da * temp[:, None]).to(DOT_DTYPE)
            dq += tl.dot(gsig, k, input_precision=INPUT_PRECISION) * scale
            if not DQ_ONLY:
                dk = tl.dot(tl.trans(gsig), q, input_precision=INPUT_PRECISION) * scale
                dv = tl.dot(tl.trans(p.to(DOT_DTYPE)), gs, input_precision=INPUT_PRECISION)
                tl.atomic_add(
                    DK_OUT + b * stride_kb + h * stride_kh
                    + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
                    dk, mask=n_valid[:, None],
                )
                tl.atomic_add(
                    DV_OUT + b * stride_vb + h * stride_vh
                    + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
                    dv, mask=n_valid[:, None],
                )

        tl.store(
            DQ + b * stride_qb + h * stride_qh
            + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
            dq, mask=m_valid[:, None],
        )
        tl.store(DT + mbase, dt, mask=m_valid)


    @triton.jit
    def _polar_sparse_bwd_kv_kernel(
        Q, K, V, REVERSE_QUERIES, REVERSE_OFFSETS, REVERSE_COUNTS,
        GS, GL, GQ2, TEMP, M, DK_OUT, DV_OUT,
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_rqb, stride_rqs,
        stride_rob, stride_rop,
        stride_rcb, stride_rcp,
        stride_gsb, stride_gsh, stride_gst, stride_gsd,
        stride_mb, stride_mh, stride_mt,
        stride_tb, stride_th, stride_tt,
        B, H, T, scale,
        PAGE_SIZE: tl.constexpr, LOCAL_WINDOW: tl.constexpr,
        LOCAL_BLOCKS: tl.constexpr, REVERSE_CAPACITY: tl.constexpr,
        DK: tl.constexpr, INPUT_PRECISION: tl.constexpr, DOT_DTYPE: tl.constexpr,
    ):
        """Key-page-parallel sparse dK/dV without atomic accumulation."""
        key_page = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        scale = scale.to(tl.float32)
        offs_n = key_page * PAGE_SIZE + tl.arange(0, PAGE_SIZE)
        offs_d = tl.arange(0, DK)
        n_valid = offs_n < T
        k = tl.load(
            K + b * stride_kb + h * stride_kh
            + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
            mask=n_valid[:, None], other=0.0,
        ).to(DOT_DTYPE)
        v = tl.load(
            V + b * stride_vb + h * stride_vh
            + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
            mask=n_valid[:, None], other=0.0,
        ).to(DOT_DTYPE)
        dk = tl.zeros([PAGE_SIZE, DK], tl.float32)
        dv = tl.zeros([PAGE_SIZE, DK], tl.float32)

        # Exact local band: key page p can be visible only from query pages
        # p..p+LOCAL_BLOCKS-1. Token masks handle causal/window boundaries.
        for local_delta in range(0, LOCAL_BLOCKS):
            query_page = key_page + local_delta
            offs_m = query_page * PAGE_SIZE + tl.arange(0, PAGE_SIZE)
            m_valid = (query_page < (T // PAGE_SIZE)) & (offs_m < T)
            q = tl.load(
                Q + b * stride_qb + h * stride_qh
                + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                mask=m_valid[:, None], other=0.0,
            ).to(DOT_DTYPE)
            gs = tl.load(
                GS + b * stride_gsb + h * stride_gsh
                + offs_m[:, None] * stride_gst + offs_d[None, :] * stride_gsd,
                mask=m_valid[:, None], other=0.0,
            ).to(DOT_DTYPE)
            mbase = b * stride_mb + h * stride_mh + offs_m * stride_mt
            g_l = tl.load(GL + mbase, mask=m_valid, other=0.0).to(tl.float32)
            g_q2 = tl.load(GQ2 + mbase, mask=m_valid, other=0.0).to(tl.float32)
            m_i = tl.load(M + mbase, mask=m_valid, other=0.0).to(tl.float32)
            temp = tl.load(
                TEMP + b * stride_tb + h * stride_th + offs_m * stride_tt,
                mask=m_valid, other=1.0,
            ).to(tl.float32)
            sig = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * scale
            valid = (m_valid[:, None] & n_valid[None, :]
                     & (offs_n[None, :] <= offs_m[:, None])
                     & (offs_n[None, :] > offs_m[:, None] - LOCAL_WINDOW))
            a = tl.where(valid, sig * temp[:, None], -1e38).to(tl.float32)
            p = tl.where(valid, tl.exp(a - m_i[:, None]), 0.0)
            gs_v = tl.dot(gs, tl.trans(v), input_precision=INPUT_PRECISION)
            dldp = gs_v + g_l[:, None] + g_q2[:, None] * (2.0 * p)
            da = tl.where(valid, dldp * p, 0.0)
            gsig = (da * temp[:, None]).to(DOT_DTYPE)
            dk += tl.dot(tl.trans(gsig), q, input_precision=INPUT_PRECISION) * scale
            dv += tl.dot(tl.trans(p.to(DOT_DTYPE)), gs, input_precision=INPUT_PRECISION)

        reverse_count = tl.load(
            REVERSE_COUNTS + b * stride_rcb + key_page * stride_rcp
        ).to(tl.int32)
        reverse_offset = tl.load(
            REVERSE_OFFSETS + b * stride_rob + key_page * stride_rop
        ).to(tl.int32)
        reverse_count = tl.minimum(tl.maximum(reverse_count, 0), REVERSE_CAPACITY)
        for remote_slot in range(0, reverse_count):
            query_page = tl.load(
                REVERSE_QUERIES + b * stride_rqb
                + (reverse_offset + remote_slot) * stride_rqs
            ).to(tl.int32)
            offs_m = query_page * PAGE_SIZE + tl.arange(0, PAGE_SIZE)
            m_valid = (query_page >= 0) & (query_page < (T // PAGE_SIZE)) & (offs_m < T)
            q = tl.load(
                Q + b * stride_qb + h * stride_qh
                + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                mask=m_valid[:, None], other=0.0,
            ).to(DOT_DTYPE)
            gs = tl.load(
                GS + b * stride_gsb + h * stride_gsh
                + offs_m[:, None] * stride_gst + offs_d[None, :] * stride_gsd,
                mask=m_valid[:, None], other=0.0,
            ).to(DOT_DTYPE)
            mbase = b * stride_mb + h * stride_mh + offs_m * stride_mt
            g_l = tl.load(GL + mbase, mask=m_valid, other=0.0).to(tl.float32)
            g_q2 = tl.load(GQ2 + mbase, mask=m_valid, other=0.0).to(tl.float32)
            m_i = tl.load(M + mbase, mask=m_valid, other=0.0).to(tl.float32)
            temp = tl.load(
                TEMP + b * stride_tb + h * stride_th + offs_m * stride_tt,
                mask=m_valid, other=1.0,
            ).to(tl.float32)
            sig = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * scale
            valid = (m_valid[:, None] & n_valid[None, :]
                     & (offs_n[None, :] <= offs_m[:, None] - LOCAL_WINDOW))
            a = tl.where(valid, sig * temp[:, None], -1e38).to(tl.float32)
            p = tl.where(valid, tl.exp(a - m_i[:, None]), 0.0)
            gs_v = tl.dot(gs, tl.trans(v), input_precision=INPUT_PRECISION)
            dldp = gs_v + g_l[:, None] + g_q2[:, None] * (2.0 * p)
            da = tl.where(valid, dldp * p, 0.0)
            gsig = (da * temp[:, None]).to(DOT_DTYPE)
            dk += tl.dot(tl.trans(gsig), q, input_precision=INPUT_PRECISION) * scale
            dv += tl.dot(tl.trans(p.to(DOT_DTYPE)), gs, input_precision=INPUT_PRECISION)

        tl.store(
            DK_OUT + b * stride_kb + h * stride_kh
            + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
            dk, mask=n_valid[:, None],
        )
        tl.store(
            DV_OUT + b * stride_vb + h * stride_vh
            + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
            dv, mask=n_valid[:, None],
        )


def _polar_sparse_forward(
    q, k, v, page_indices, page_counts, v_null, null_base, null_slope_raw,
    len_gain_raw, mag_beta_raw, page_size, local_window, eps, input_precision,
):
    B, H, T, dk = q.shape
    dev, out_dtype = q.device, q.dtype
    dot_dtype, ip, is_fp32 = _dtype_meta(out_dtype)
    if is_fp32:
        ip = input_precision
    spg = _softplus(len_gain_raw.float()).contiguous()
    sps = _softplus(null_slope_raw.float()).contiguous()
    beta = _softplus(mag_beta_raw.float()).contiguous()
    nb = null_base.float().contiguous()
    vn = v_null.float().contiguous()
    page_indices = page_indices.to(device=dev, dtype=torch.int32).contiguous()
    page_counts = page_counts.to(device=dev, dtype=torch.int32).contiguous()
    remote_capacity = page_indices.shape[-1]
    local_blocks = local_window // page_size + 1

    c = torch.empty_like(q, dtype=torch.float32)
    s = torch.empty_like(q, dtype=torch.float32)
    mag = torch.empty((B, H, T), device=dev, dtype=torch.float32)
    M = torch.empty_like(mag)
    L = torch.empty_like(mag)
    Q2 = torch.empty_like(mag)
    grid = (T // page_size, B * H)
    cfg = _fwd_config(dk, is_fp32, dev)
    _polar_sparse_fwd_kernel[grid](
        q, k, v, page_indices, page_counts, vn, spg, nb, sps, beta,
        c, mag, M, L, Q2, s,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        page_indices.stride(0), page_indices.stride(1), page_indices.stride(2),
        page_counts.stride(0), page_counts.stride(1),
        c.stride(0), c.stride(1), c.stride(2), c.stride(3),
        s.stride(0), s.stride(1), s.stride(2), s.stride(3),
        mag.stride(0), mag.stride(1), mag.stride(2),
        B, H, T, 1.0 / math.sqrt(dk), eps,
        PAGE_SIZE=page_size, LOCAL_WINDOW=local_window, LOCAL_BLOCKS=local_blocks,
        REMOTE_CAPACITY=remote_capacity, DK=dk, INPUT_PRECISION=ip, DOT_DTYPE=dot_dtype,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return c.to(out_dtype), mag.to(out_dtype), M, L, Q2, s, page_indices, page_counts


def _polar_sparse_backward(
    gc, gm, q, k, v, page_indices, page_counts, v_null, null_base,
    null_slope_raw, len_gain_raw, mag_beta_raw, M, L, Q2, s,
    page_size, local_window, eps, input_precision,
):
    B, H, T, dk = q.shape
    dev, fdt = q.device, torch.float32
    gc, gm = gc.to(fdt), gm.to(fdt)
    # Match the checkpoint's dense Polar length calibration.  Routing changes
    # the reduced support, not the meaning of n_keys.
    n = torch.arange(1, T + 1, device=dev, dtype=fdt).view(1, T).expand(B, -1)
    spg = _softplus(len_gain_raw.to(fdt)).view(1, H, 1)
    sps = _softplus(null_slope_raw.to(fdt)).view(1, H, 1)
    beta = _softplus(mag_beta_raw.to(fdt)).view(1, H, 1)
    logn = torch.log(n).unsqueeze(1)
    sqrt_lognp1 = torch.sqrt(torch.log(n + 1.0)).unsqueeze(1)
    temp = 1.0 + spg * logn
    nu = null_base.to(fdt).view(1, H, 1) + sps * sqrt_lognp1
    a_n = temp * nu
    p_n = torch.exp(a_n - M)
    z = L + p_n
    stats_tiny = 1.0e-30
    zc, q2c = z.clamp_min(stats_tiny), Q2.clamp_min(stats_tiny)
    n_eff = (L * L) / q2c
    m_eff = n_eff * (L / zc)
    log1p_m = torch.log1p(m_eff)
    mag = torch.tanh(beta * log1p_m)
    gme = gm * beta * (1.0 - mag * mag) / (1.0 + m_eff)
    dm_dl = torch.where(L > 0, 3.0 * m_eff / L, torch.zeros_like(L))
    dm_dq2 = -(m_eff / q2c) * (Q2 > stats_tiny)
    dm_dz = -(m_eff / zc) * (z > stats_tiny)
    g_l = gme * (dm_dl + dm_dz)
    g_q2 = gme * dm_dq2
    gpn = gme * dm_dz

    s = s.to(fdt)
    snorm = s.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    c = s / snorm
    gs = (gc - (gc * c).sum(-1, keepdim=True) * c) / snorm
    gpn = gpn + (gs * v_null.to(fdt).view(1, H, 1, dk)).sum(-1)
    grad_v_null = (p_n.unsqueeze(-1) * gs).sum(dim=(0, 2))
    da_n = gpn * p_n
    grad_t_null = da_n * nu
    grad_nu = da_n * temp

    out_dtype = q.dtype
    dot_dtype, ip, is_fp32 = _dtype_meta(out_dtype)
    if is_fp32:
        ip = input_precision
    gs_kernel = gs.to(out_dtype).contiguous()
    g_l = g_l.contiguous()
    g_q2 = g_q2.contiguous()
    M = M.contiguous()
    temp = temp.contiguous()
    dq = torch.zeros_like(q, dtype=fdt)
    dk_out = torch.zeros_like(k, dtype=fdt)
    dv_out = torch.zeros_like(v, dtype=fdt)
    dt_real = torch.zeros((B, H, T), device=dev, dtype=fdt)
    local_blocks = local_window // page_size + 1
    remote_capacity = page_indices.shape[-1]
    query_pages = T // page_size

    # Invert the arbitrary query->remote-page lists into contiguous per-key
    # query lists. This fixed-shape GPU sort is tiny relative to attention and
    # enables a key-page-parallel dK/dV kernel with no atomic accumulation.
    slots = torch.arange(remote_capacity, device=dev).view(1, 1, -1)
    route_valid = slots < page_counts[..., None]
    route_keys = torch.where(
        route_valid,
        page_indices.to(torch.int64),
        page_indices.new_full((), query_pages, dtype=torch.int64),
    ).reshape(B, -1)
    _, permutation = route_keys.sort(dim=1)
    query_ids = torch.arange(query_pages, device=dev, dtype=torch.int64)
    query_ids = query_ids.view(1, query_pages, 1).expand(B, -1, remote_capacity).reshape(B, -1)
    reverse_queries = query_ids.gather(1, permutation).to(torch.int32).contiguous()
    reverse_counts_all = torch.zeros(
        (B, query_pages + 1), device=dev, dtype=torch.int64
    ).scatter_add(1, route_keys, route_valid.reshape(B, -1).to(torch.int64))
    reverse_counts = reverse_counts_all[:, :query_pages].to(torch.int32).contiguous()
    reverse_offsets = (
        reverse_counts.to(torch.int64).cumsum(dim=1) - reverse_counts
    ).to(torch.int32).contiguous()

    configs = _bwd_config(dk, is_fp32, dev)
    cfg = configs["dq"]
    _polar_sparse_bwd_kernel[(T // page_size, B * H)](
        q, k, v, page_indices, page_counts, gs_kernel, g_l, g_q2, temp, M,
        dq, dk_out, dv_out, dt_real,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        page_indices.stride(0), page_indices.stride(1), page_indices.stride(2),
        page_counts.stride(0), page_counts.stride(1),
        gs_kernel.stride(0), gs_kernel.stride(1), gs_kernel.stride(2), gs_kernel.stride(3),
        g_l.stride(0), g_l.stride(1), g_l.stride(2),
        temp.stride(0), temp.stride(1), temp.stride(2),
        B, H, T, 1.0 / math.sqrt(dk),
        PAGE_SIZE=page_size, LOCAL_WINDOW=local_window, LOCAL_BLOCKS=local_blocks,
        REMOTE_CAPACITY=remote_capacity, DK=dk, INPUT_PRECISION=ip, DOT_DTYPE=dot_dtype,
        DQ_ONLY=True,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    cfg_kv = configs["kv"]
    _polar_sparse_bwd_kv_kernel[(query_pages, B * H)](
        q, k, v, reverse_queries, reverse_offsets, reverse_counts,
        gs_kernel, g_l, g_q2, temp, M, dk_out, dv_out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        reverse_queries.stride(0), reverse_queries.stride(1),
        reverse_offsets.stride(0), reverse_offsets.stride(1),
        reverse_counts.stride(0), reverse_counts.stride(1),
        gs_kernel.stride(0), gs_kernel.stride(1), gs_kernel.stride(2), gs_kernel.stride(3),
        g_l.stride(0), g_l.stride(1), g_l.stride(2),
        temp.stride(0), temp.stride(1), temp.stride(2),
        B, H, T, 1.0 / math.sqrt(dk),
        PAGE_SIZE=page_size, LOCAL_WINDOW=local_window, LOCAL_BLOCKS=local_blocks,
        REVERSE_CAPACITY=query_pages, DK=dk, INPUT_PRECISION=ip, DOT_DTYPE=dot_dtype,
        num_warps=cfg_kv["num_warps"], num_stages=cfg_kv["num_stages"],
    )

    grad_t = dt_real + grad_t_null
    grad_len_gain = (grad_t * logn).sum(dim=(0, 2)) * torch.sigmoid(len_gain_raw.to(fdt))
    grad_null_base = grad_nu.sum(dim=(0, 2))
    grad_null_slope = ((grad_nu * sqrt_lognp1).sum(dim=(0, 2))
                       * torch.sigmoid(null_slope_raw.to(fdt)))
    grad_mag_beta = ((gm * (1.0 - mag * mag) * log1p_m).sum(dim=(0, 2))
                     * torch.sigmoid(mag_beta_raw.to(fdt)))
    return (dq, dk_out, dv_out, grad_v_null, grad_null_base, grad_null_slope,
            grad_len_gain, grad_mag_beta)


class PolarSparseAttentionTriton(torch.autograd.Function):
    """Block-sparse Polar attention for Foveal local-plus-remote routes."""

    @staticmethod
    def forward(
        ctx, q, k, v, page_indices, page_counts, v_null, null_base,
        null_slope_raw, len_gain_raw, mag_beta_raw, page_size, local_window,
        eps, input_precision,
    ):
        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        c, mag, M, L, Q2, s, page_indices, page_counts = _polar_sparse_forward(
            q, k, v, page_indices, page_counts, v_null, null_base,
            null_slope_raw, len_gain_raw, mag_beta_raw, page_size,
            local_window, eps, input_precision,
        )
        ctx.save_for_backward(
            q, k, v, page_indices, page_counts, v_null, null_base,
            null_slope_raw, len_gain_raw, mag_beta_raw, M, L, Q2, s,
        )
        ctx.page_size = page_size
        ctx.local_window = local_window
        ctx.eps = eps
        ctx.input_precision = input_precision
        return c, mag

    @staticmethod
    def backward(ctx, gc, gm):
        (q, k, v, page_indices, page_counts, v_null, null_base,
         null_slope_raw, len_gain_raw, mag_beta_raw, M, L, Q2, s) = ctx.saved_tensors
        grads = _polar_sparse_backward(
            gc, gm, q, k, v, page_indices, page_counts, v_null, null_base,
            null_slope_raw, len_gain_raw, mag_beta_raw, M, L, Q2, s,
            ctx.page_size, ctx.local_window, ctx.eps, ctx.input_precision,
        )
        refs = (q, k, v, v_null, null_base, null_slope_raw, len_gain_raw, mag_beta_raw)
        grads = tuple(grad.to(ref.dtype) for grad, ref in zip(grads, refs))
        return (grads[0], grads[1], grads[2], None, None, grads[3], grads[4],
                grads[5], grads[6], grads[7], None, None, None, None)


def polar_attention_sparse(
    q, k, v, page_indices, page_counts, *, page_size, local_window,
    v_null, null_base, null_slope_raw, len_gain_raw, mag_beta_raw,
    eps=1e-6, input_precision="ieee",
):
    """Trainable block-sparse Polar attention for a Foveal route.

    ``q``, ``k`` and ``v`` are ``(B,H,T,D)`` with GQA heads already expanded.
    ``page_indices`` is ``(B,T/page_size,capacity)`` and ``page_counts`` gives
    the active prefix of each route row.  The support is the union of the exact
    causal ``local_window`` and those completed remote pages. Polar temperature
    and null-floor calibration still use the full causal prefix length ``t+1``.
    Page selection is discrete: gradients are returned for attention tensors and
    Polar parameters, but never for the integer route.
    """
    if not HAS_TRITON or not q.is_cuda:
        raise RuntimeError("sparse Polar attention requires CUDA and Triton")
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, and v must have identical (B,H,T,D) shapes")
    B, H, T, dk = q.shape
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported sparse Polar dtype: {q.dtype}")
    if dk < 16 or dk & (dk - 1):
        raise ValueError(f"head_dim (dk={dk}) must be a power of two of at least 16")
    if page_size < 16 or page_size > 128 or page_size & (page_size - 1):
        raise ValueError("page_size must be a power of two in [16, 128]")
    if T % page_size:
        raise ValueError("token length must be divisible by page_size")
    if local_window <= 0 or local_window % page_size:
        raise ValueError("local_window must be positive and divisible by page_size")
    query_pages = T // page_size
    if page_indices.ndim != 3 or page_indices.shape[:2] != (B, query_pages):
        raise ValueError("page_indices must have shape (B,T/page_size,capacity)")
    if page_counts.shape != (B, query_pages):
        raise ValueError("page_counts must have shape (B,T/page_size)")
    if not 0 < page_indices.shape[-1] <= 128:
        raise ValueError("the sparse route capacity must lie in [1, 128]")
    expected_heads = (H, dk)
    if v_null.shape != expected_heads:
        raise ValueError(f"v_null must have shape {expected_heads}")
    for name, param in (("null_base", null_base), ("null_slope_raw", null_slope_raw),
                        ("len_gain_raw", len_gain_raw), ("mag_beta_raw", mag_beta_raw)):
        if param.shape != (H,):
            raise ValueError(f"{name} must have shape ({H},)")
    return PolarSparseAttentionTriton.apply(
        q, k, v, page_indices, page_counts, v_null, null_base,
        null_slope_raw, len_gain_raw, mag_beta_raw, int(page_size),
        int(local_window), float(eps), input_precision,
    )
