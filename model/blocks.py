import os
import torch
from torch import nn
import torch.nn.functional as F

# FlashAttention-style Triton kernel (CUDA only). Re-exported so the `model`,
# `train`, and `inference` packages can all reach it via model.blocks. Falls back
# to None when Triton / CUDA is unavailable; callers must guard on HAS_TRITON.
try:
    from kernel.polar_triton import (
        polar_attention as polar_attention_triton,
        polar_attention_fwd as polar_attention_triton_fwd,
        HAS_TRITON,
    )
except Exception:
    polar_attention_triton = None
    polar_attention_triton_fwd = None
    HAS_TRITON = False

# flash-linear-attention's fused chunked gated-delta kernel (CUDA/Triton). When present
# it replaces the pure-PyTorch gated_delta_chunked for the memory branch -- far faster and,
# being a custom autograd.Function, opaque to torch.compile (no inductor blow-up). Optional.
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    _HAS_FLA = True
    _FLA_IMPORT_ERROR = None
except Exception as error:
    chunk_gated_delta_rule = None
    _HAS_FLA = False
    _FLA_IMPORT_ERROR = repr(error)

# FLA's kernel is a Triton autograd.Function -> torch.compile graph-breaks at every memory
# layer (lost fusion + lost cross-break memory planning = the ~2x time/RAM regression). The
# DEFAULT path is the plain call (graph break, but known-correct and trains fine).
#
# Opt-in FLA_CUSTOM_OP=1 removes the break by wrapping FLA as opaque custom ops. BOTH the
# forward AND the backward must be opaque: a single custom op with a python backward fails
# because AOTAutograd traces the backward into the joint graph and re-enters the Triton
# kernel (autotuner chokes: "unhashable type: non-nested SymInt"). So fwd and bwd are each a
# custom_op with register_fake; the bwd recomputes FLA's autograd eagerly (correct grads +
# fwd activations not stored). Experimental on a given torch/FLA combo -> verify_fla.py first.
_fla_gated_delta = None
if _HAS_FLA:
    def _fla_raw(q, k, v, g, beta):                     # default: plain call (graph-breaks, works)
        # beta keyword-bound on purpose: FLA's fused_recurrent variant has gk/gv params
        # between g and beta, and chunk may grow them too — positional beta would bind
        # to gk and be indexed as [B, T, H, K] (out-of-bounds reads).
        return chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta,
                                      scale=1.0, use_qk_l2norm_in_kernel=True)[0]
    _fla_gated_delta = _fla_raw

    if os.environ.get("FLA_CUSTOM_OP", "0") == "1":
        try:
            @torch.library.custom_op("atma::fla_gd_fwd", mutates_args=())
            def _fla_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         g: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
                return _fla_raw(q, k, v, g, beta)

            @_fla_fwd.register_fake
            def _(q, k, v, g, beta):
                return torch.empty_like(v)

            @torch.library.custom_op("atma::fla_gd_bwd", mutates_args=())
            def _fla_bwd(grad_o: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                         v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> list[torch.Tensor]:
                with torch.enable_grad():                # recompute FLA fwd -> its autograd graph
                    ins = [t.detach().requires_grad_(True) for t in (q, k, v, g, beta)]
                    o = _fla_raw(*ins)
                    return list(torch.autograd.grad(o, ins, grad_o))

            @_fla_bwd.register_fake
            def _(grad_o, q, k, v, g, beta):
                return [torch.empty_like(t) for t in (q, k, v, g, beta)]

            def _setup(ctx, inputs, output):
                ctx.save_for_backward(*inputs)

            def _bwd(ctx, grad_o):
                return tuple(_fla_bwd(grad_o, *ctx.saved_tensors))   # opaque bwd op (not traced)

            _fla_fwd.register_autograd(_bwd, setup_context=_setup)
            _fla_gated_delta = _fla_fwd
        except Exception:
            _fla_gated_delta = _fla_raw                  # fall back to the safe path on any error


# --- Diagnostic probe ---------------------------------------------------------
# Off by default (None): zero numeric impact, no overhead. When eval.py --diagnose
# sets this to a list, the polar reductions append per-call internals
# (n_eff, mag, w_null), one dict per attention layer per forward, so the activation
# probe can see whether the count channel saturates / the null sink drains as N grows.
_PROBE = None


def _probe_emit(n_eff, mag, w_null):
    if _PROBE is not None:
        _PROBE.append(dict(
            n_eff=n_eff.detach().float(),     # (B, H, T) effective match count
            mag=mag.detach().float(),         # (B, H, T) bounded count channel in [0,1)
            w_null=w_null.detach().float(),   # (B, H, T) mass drained to the null sink
        ))


def polar_temp_null(n_keys, len_gain_raw, null_base, null_slope_raw):
    """Per-head length temperature and EV-corrected null floor.

    n_keys: (Tq,) valid key count per query (causal -> 1..T).
    len_gain_raw, null_base, null_slope_raw: (H,) per-head params.
    Returns temp, null each shaped (1, H, Tq, 1) for broadcasting over keys.
    """
    H = len_gain_raw.shape[0]
    n = n_keys.clamp(min=1.0)
    temp = 1.0 + F.softplus(len_gain_raw).view(1, H, 1, 1) * torch.log(n).view(1, 1, -1, 1)
    null = (null_base.view(1, H, 1, 1)
            + F.softplus(null_slope_raw).view(1, H, 1, 1)
            * torch.sqrt(torch.log(n + 1.0)).view(1, 1, -1, 1))
    return temp, null


def polar_reduce(sigma, v, n_keys, *, v_null, null_base, null_slope_raw,
                 len_gain_raw, mag_beta_raw, eps=1e-6):
    """Polar attention reduction (the validated materialized oracle). Computed in fp32.

    Splits attention into two length-invariant channels from ONE temp-sharpened
    softmax with an EV-corrected null sink:
      - direction c : unit vector, count-blind ("what"), via convex value mix.
      - magnitude mag: bounded "how much", = participation-ratio multiplicity gated
                       by null-sink confidence, squashed through tanh(beta*log1p(.)).

    sigma : (B, H, Tq, Tk) raw scores; masked (future) entries must be -inf.
    v     : (B, H, Tk, dk) values (KV heads already expanded to H).
    Returns c (B, H, Tq, dk), mag (B, H, Tq), both cast back to v's dtype.
    """
    out_dtype = v.dtype
    cd = _compute_dtype(out_dtype)            # bf16/fp16 -> fp32; fp32/fp64 preserved
    B, H, Tq, Tk = sigma.shape
    dk = v.shape[-1]
    sigma = sigma.to(cd)
    v = v.to(cd)

    temp, null = polar_temp_null(n_keys.to(cd), len_gain_raw.to(cd), null_base.to(cd), null_slope_raw.to(cd))
    # Scale BEFORE masking: multiplying -inf scores by temp makes grad_temp = 0*(-inf) = NaN.
    # Neutralize masked entries to a finite value for the product, then re-apply -inf.
    masked = torch.isneginf(sigma)
    sigma_safe = torch.where(masked, torch.zeros_like(sigma), sigma)
    real = (sigma_safe * temp).masked_fill(masked, float("-inf"))
    logits = torch.cat([real, null.expand(B, H, Tq, 1) * temp], dim=-1)
    w = torch.softmax(logits, dim=-1)
    w_null = w[..., -1:]          # (B, H, Tq, 1)  mass drained to the null sink
    w_r = w[..., :-1]             # (B, H, Tq, Tk) weights over real keys

    # direction channel
    s = torch.matmul(w_r, v) + w_null * v_null.to(cd).view(1, H, 1, dk)
    c = F.normalize(s, p=2, dim=-1, eps=eps)

    # count/magnitude channel — participation ratio gated by confidence, bounded
    denom = w_r.sum(-1, keepdim=True).clamp_min(eps)
    w_hat = w_r / denom
    n_eff = 1.0 / w_hat.square().sum(-1).clamp_min(eps)        # (B, H, Tq)
    m_eff = n_eff * (1.0 - w_null.squeeze(-1))
    mag = torch.tanh(F.softplus(mag_beta_raw.to(cd)).view(1, H, 1) * torch.log1p(m_eff))

    _probe_emit(n_eff, mag, w_null.squeeze(-1))
    return c.to(out_dtype), mag.to(out_dtype)


def _compute_dtype(dtype):
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


class _PolarOnline(torch.autograd.Function):
    """FlashAttention-style online polar reduction: O(T * k_block) memory in BOTH
    passes. Outputs are exactly polar_reduce(...) up to fp summation order.

    All quantities are invariant to the running-max shift M, so M is treated as a
    detached constant in the backward (the standard flash trick). The count channel
    needs one extra streamed accumulator beyond vanilla attention: Q2 = sum p_j^2
    (rescales by alpha^2), giving the participation ratio n_eff = L^2 / Q2.
    """

    @staticmethod
    def forward(ctx, q, k, v, n_keys, v_null, null_base, null_slope_raw,
                len_gain_raw, mag_beta_raw, k_block, eps, window=None):
        out_dtype = q.dtype
        cd = _compute_dtype(out_dtype)
        B, H, T, dk = q.shape
        Tk = k.shape[2]
        qd, kd, vd = q.to(cd), k.to(cd), v.to(cd)
        scale = dk ** -0.5

        # window (eval-only sliding window): each query sees only its last `window`
        # keys, so temp/null use the capped count min(n_keys, window) and the block
        # loop also masks keys older than the window. backward does NOT model the
        # band — guarded against grad use in polar_attention_online.
        n_count = n_keys if window is None else torch.minimum(n_keys, n_keys.new_tensor(float(window)))
        n = n_count.to(cd).clamp(min=1.0)
        t = (1.0 + F.softplus(len_gain_raw.to(cd)).view(1, H, 1) * torch.log(n).view(1, 1, T))   # (1,H,T)
        nu = (null_base.to(cd).view(1, H, 1)
              + F.softplus(null_slope_raw.to(cd)).view(1, H, 1) * torch.sqrt(torch.log(n + 1.0)).view(1, 1, T))

        NEG = torch.finfo(cd).min
        M = torch.full((B, H, T), NEG, dtype=cd, device=q.device)
        L = torch.zeros((B, H, T), dtype=cd, device=q.device)
        Q2 = torch.zeros((B, H, T), dtype=cd, device=q.device)
        S = torch.zeros((B, H, T, dk), dtype=cd, device=q.device)
        key_idx = torch.arange(Tk, device=q.device)

        for ks in range(0, Tk, k_block):
            ke = min(ks + k_block, Tk)
            kb, vb = kd[:, :, ks:ke], vd[:, :, ks:ke]                       # (B,H,Kb,dk)
            sig = torch.matmul(qd, kb.transpose(-2, -1)) * scale            # (B,H,T,Kb)
            a = sig * t.unsqueeze(-1)
            kidx = key_idx[ks:ke].view(1, 1, 1, -1)
            invalid = kidx >= n_keys.view(1, 1, T, 1)                       # future
            if window is not None:
                invalid = invalid | (kidx < (n_keys.view(1, 1, T, 1) - window))  # older than window
            a = a.masked_fill(invalid, NEG)
            blk_max = a.amax(dim=-1)                                        # (B,H,T)
            M_new = torch.maximum(M, blk_max)
            alpha = torch.exp(M - M_new)
            p = torch.exp(a - M_new.unsqueeze(-1))                          # invalid -> 0
            L = L * alpha + p.sum(-1)
            Q2 = Q2 * alpha * alpha + p.square().sum(-1)
            S = S * alpha.unsqueeze(-1) + torch.matmul(p, vb)
            M = M_new

        # fold null sink (single logit per query)
        a_n = t * nu                                                       # (1,H,T)*(1,H,T) -> (1,H,T)
        a_n = a_n.expand(B, H, T)
        M_new = torch.maximum(M, a_n)
        alpha = torch.exp(M - M_new)
        L = L * alpha
        Q2 = Q2 * alpha * alpha
        S = S * alpha.unsqueeze(-1)
        M = M_new
        p_n = torch.exp(a_n - M)                                           # (B,H,T)
        Z = L + p_n

        s = S + p_n.unsqueeze(-1) * v_null.to(cd).view(1, H, 1, dk)
        c = F.normalize(s, p=2, dim=-1, eps=eps)
        n_eff = L * L / Q2.clamp_min(eps)
        m_eff = n_eff * (L / Z.clamp_min(eps))
        beta = F.softplus(mag_beta_raw.to(cd)).view(1, H, 1)
        mag = torch.tanh(beta * torch.log1p(m_eff))

        _probe_emit(n_eff, mag, p_n / Z.clamp_min(eps))
        ctx.save_for_backward(q, k, v, n_keys, v_null, null_base, null_slope_raw,
                              len_gain_raw, mag_beta_raw, M, L, Q2, s)
        ctx.k_block, ctx.eps, ctx.window = k_block, eps, window
        return c.to(out_dtype), mag.to(out_dtype)

    @staticmethod
    def backward(ctx, gc, gm):
        (q, k, v, n_keys, v_null, null_base, null_slope_raw,
         len_gain_raw, mag_beta_raw, M, L, Q2, s) = ctx.saved_tensors
        k_block, eps, window = ctx.k_block, ctx.eps, ctx.window
        out_dtype = q.dtype
        cd = _compute_dtype(out_dtype)
        B, H, T, dk = q.shape
        Tk = k.shape[2]
        qd, kd, vd = q.to(cd), k.to(cd), v.to(cd)
        gc = gc.to(cd)
        gm = gm.to(cd)
        scale = dk ** -0.5

        # windowed count: temp/null use min(n_keys, window) so backward matches the
        # forward's band (each query attends only to its last `window` keys).
        n_count = n_keys if window is None else torch.minimum(n_keys, n_keys.new_tensor(float(window)))
        n = n_count.to(cd).clamp(min=1.0)
        spg = F.softplus(len_gain_raw.to(cd)).view(1, H, 1)
        sps = F.softplus(null_slope_raw.to(cd)).view(1, H, 1)
        logn = torch.log(n).view(1, 1, T)
        sqrt_lognp1 = torch.sqrt(torch.log(n + 1.0)).view(1, 1, T)
        t = 1.0 + spg * logn                                               # (1,H,T)
        nu = null_base.to(cd).view(1, H, 1) + sps * sqrt_lognp1
        beta = F.softplus(mag_beta_raw.to(cd)).view(1, H, 1)

        a_n = (t * nu).expand(B, H, T)
        p_n = torch.exp(a_n - M)
        Z = L + p_n
        Zc, Q2c = Z.clamp_min(eps), Q2.clamp_min(eps)
        n_eff = L * L / Q2c
        m_eff = n_eff * (L / Zc)
        log1p_m = torch.log1p(m_eff)
        mag = torch.tanh(beta * log1p_m)

        # --- magnitude path -> m_eff ---
        gme = gm * beta * (1.0 - mag * mag) / (1.0 + m_eff)                # (B,H,T)
        # m_eff = L^3 / (Q2 * Z)
        dm_dL = 3.0 * L * L / (Q2c * Zc)
        dm_dQ2 = -(L ** 3) / (Q2c * Q2c * Zc)
        dm_dZ = -(L ** 3) / (Q2c * Zc * Zc)
        gL = gme * (dm_dL + dm_dZ)          # Z = L + p_n  -> dZ/dL = 1
        gQ2 = gme * dm_dQ2
        gZ_pn = gme * dm_dZ                  # part of dLoss/dp_n via Z

        # --- direction path -> s ---
        s_cd = s.to(cd)
        snorm = s_cd.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        c = s_cd / snorm
        gs = (gc - (gc * c).sum(-1, keepdim=True) * c) / snorm             # (B,H,T,dk)

        # grads onto p_n, v_null
        gpn = (gs * v_null.to(cd).view(1, H, 1, dk)).sum(-1) + gZ_pn       # (B,H,T)
        grad_v_null = (p_n.unsqueeze(-1) * gs).sum(dim=(0, 2))             # (H,dk)

        # null-logit grad -> contributes to t and nu
        da_n = gpn * p_n                                                   # (B,H,T)
        grad_t = da_n * nu                                                 # accumulate (B,H,T)
        grad_nu = da_n * t

        grad_q = torch.zeros_like(qd)
        grad_k = torch.zeros_like(kd)
        grad_v = torch.zeros_like(vd)
        key_idx = torch.arange(Tk, device=q.device)

        for ks in range(0, Tk, k_block):
            ke = min(ks + k_block, Tk)
            kb, vb = kd[:, :, ks:ke], vd[:, :, ks:ke]
            sig = torch.matmul(qd, kb.transpose(-2, -1)) * scale
            a = sig * t.unsqueeze(-1)
            kidx = key_idx[ks:ke].view(1, 1, 1, -1)
            invalid = kidx >= n_keys.view(1, 1, T, 1)                      # future
            if window is not None:
                invalid = invalid | (kidx < (n_keys.view(1, 1, T, 1) - window))  # older than window
            a = a.masked_fill(invalid, torch.finfo(cd).min)
            p = torch.exp(a - M.unsqueeze(-1))                             # (B,H,T,Kb); invalid->0

            gs_v = torch.matmul(gs, vb.transpose(-2, -1))                  # (B,H,T,Kb) = gs . v_j
            dLdp = gs_v + gL.unsqueeze(-1) + gQ2.unsqueeze(-1) * (2.0 * p)
            da = dLdp * p                                                  # dLoss/da_j
            da = da.masked_fill(invalid, 0.0)

            grad_v = grad_v.index_add(2, key_idx[ks:ke], torch.matmul(p.transpose(-2, -1), gs))
            grad_t = grad_t + (da * sig).sum(-1)
            gsig = da * t.unsqueeze(-1)                                    # dLoss/dsig_j
            grad_q = grad_q + torch.matmul(gsig, kb) * scale
            grad_k = grad_k.index_add(2, key_idx[ks:ke], torch.matmul(gsig.transpose(-2, -1), qd) * scale)

        # --- params from t and nu ---
        grad_len_gain = (grad_t * logn).sum(dim=(0, 2)) * torch.sigmoid(len_gain_raw.to(cd))
        grad_null_base = grad_nu.sum(dim=(0, 2))
        grad_null_slope = (grad_nu * sqrt_lognp1).sum(dim=(0, 2)) * torch.sigmoid(null_slope_raw.to(cd))
        grad_mag_beta = (gm * (1.0 - mag * mag) * log1p_m).sum(dim=(0, 2)) * torch.sigmoid(mag_beta_raw.to(cd))

        def cast(x, ref):
            return x.to(ref.dtype)
        return (cast(grad_q, q), cast(grad_k, k), cast(grad_v, v), None,
                cast(grad_v_null, v_null), cast(grad_null_base, null_base),
                cast(grad_null_slope, null_slope_raw), cast(grad_len_gain, len_gain_raw),
                cast(grad_mag_beta, mag_beta_raw), None, None, None)


def polar_attention_online(q, k, v, n_keys, *, v_null, null_base, null_slope_raw,
                           len_gain_raw, mag_beta_raw, k_block=512, eps=1e-6, window=None):
    """Memory-efficient polar attention. q,k,v: (B,H,T,dk) with KV heads expanded.
    Streams keys in blocks of k_block -> O(B*H*T*k_block) peak (fwd and bwd).
    Numerically equals materialized scores + polar_reduce.

    window (int, optional): causal sliding window — each query attends to only its last
    `window` keys. The streaming backward now models the band (temp/null use the windowed
    count and the key loop masks keys older than n_keys - window), so this is trainable."""
    return _PolarOnline.apply(q, k, v, n_keys, v_null, null_base, null_slope_raw,
                              len_gain_raw, mag_beta_raw, k_block, eps, window)


# --- Titans-style linear compression memory (MAG branch) -------------------------
# Validated standalone in titans_proto.py / verify_titans.py (chunked == sequential
# scan, fp64). The Titans neural memory with momentum eta=0 reparametrizes exactly to
# the Gated DeltaNet recurrence  M_t = gamma_t (M_{t-1} + w_t k_t^T),  w_t = beta_t
# (v_t - M_{t-1} k_t),  with gamma = 1 - alpha (decay), beta = theta / (1 - alpha).

@torch.compiler.disable   # the unrolled chunk loop + linalg solve blows up inductor compile;
                          # run it eager (FLA's kernel is the compiled fast path instead).
def gated_delta_chunked(q, k, v, gamma, beta, chunk=64, S0=None):
    """Chunkwise-parallel gated delta rule (flash-linear-attention convention). q,k:
    (B,H,N,dk); v: (B,H,N,dv); gamma,beta: (B,H,N) in (0,1). Returns (R (B,H,N,dv),
    S_final (B,H,dv,dk)). Matches gated_delta_sequential (decay-first, write undecayed,
    self-inclusive readout M_t q_t) and fla.ops.gated_delta_rule.

    Within a chunk the running-state coupling is removed by solving the unit lower-
    triangular UT system  (I + diag(beta) D) U = diag(beta)(V - C_carry),  where the
    increment u_t = beta_t(v_t - gamma_t M_{t-1} k_t), D[p,s] = (g_p/g_s)(k_s . k_p) for
    s<p, g_p = prod_{l<=p} gamma_l (INCLUSIVE -> decay applied before the write). Readout
    r_p = g_p(M_0 q_p) + sum_{s<=p} (g_p/g_s)(k_s . q_p) u_s  (s<=p is self-inclusive).
    Sequential only across chunks; intra-chunk is batched matmuls + one triangular solve.
    NOTE: keys/queries must be unit-norm for delta-rule stability (see TitansMemory)."""
    B, H, N, dk = q.shape
    dv = v.shape[-1]
    dtype, device = q.dtype, q.device
    S = torch.zeros(B, H, dv, dk, dtype=dtype, device=device) if S0 is None else S0
    Rs = []
    for cs in range(0, N, chunk):
        ce = min(cs + chunk, N)
        C = ce - cs
        qc, kc, vc = q[:, :, cs:ce], k[:, :, cs:ce], v[:, :, cs:ce]
        gc_, bc = gamma[:, :, cs:ce], beta[:, :, cs:ce]

        clg = torch.cumsum(torch.log(gc_), dim=-1)                       # clg[p]=log g_p (inclusive)
        idx = torch.arange(C, device=device)
        diff = clg[..., :, None] - clg[..., None, :]                     # log(g_p/g_s)  (B,H,C,C)
        strict = idx[:, None] > idx[None, :]                             # s < p
        incl = idx[:, None] >= idx[None, :]                              # s <= p (self-inclusive)
        dec_strict = torch.exp(diff.masked_fill(~strict, float("-inf")))  # g_p/g_s, 0 where s>=p
        dec_incl = torch.exp(diff.masked_fill(~incl, float("-inf")))      # g_p/g_s, 0 where s>p

        D = dec_strict * torch.einsum("bhpd,bhsd->bhps", kc, kc)         # (g_p/g_s)(k_s.k_p), s<p
        Rq = dec_incl * torch.einsum("bhpd,bhsd->bhps", qc, kc)          # (g_p/g_s)(k_s.q_p), s<=p
        carry = torch.exp(clg)                                           # g_p (decay incl. step p)
        c_mat = carry[..., None] * torch.einsum("bhvd,bhcd->bhcv", S, kc)   # g_p M_0 k_p
        cprime = carry[..., None] * torch.einsum("bhvd,bhcd->bhcv", S, qc)  # g_p M_0 q_p

        # (I + diag(beta) D) is unit lower-triangular (D strictly lower) -> exact triangular
        # solve. unitriangular=True treats the diagonal as 1, so pass the strictly-lower part.
        L = bc[..., :, None] * D
        U = torch.linalg.solve_triangular(L, bc[..., :, None] * (vc - c_mat),
                                          upper=False, unitriangular=True)
        Rs.append(cprime + torch.einsum("bhij,bhjv->bhiv", Rq, U))

        out_ratio = torch.exp(clg[..., -1:] - clg)                      # g_C/g_s
        gC = torch.exp(clg[..., -1])                                    # g_C
        S = gC[..., None, None] * S + torch.einsum("bhcv,bhcd->bhvd", out_ratio[..., None] * U, kc)
    return torch.cat(Rs, dim=2), S


class TitansMemory(nn.Module):
    """Titans-style linear long-term memory (MAG branch) for PolarAttention.

    Per-head matrix memory M (dv x dk) updated by the gated delta rule with
    data-dependent per-head retention gamma_t = sigmoid(W_gamma x + b_gamma) (init high
    ~0.98) and write strength beta_t = sigmoid(W_beta x + b_beta). Keys/queries are
    L2-normalized (UNIT norm) for delta-rule stability -- distinct from polar's RMS-norm,
    which would make ||k||^2 = dk and the rule expansive. The readout M q is RMS-normed,
    output-gated, and projected to the residual as an additive third channel
    (out = content + count + mem). proj is zero-init so the branch starts as a no-op.

    Two backends (`kernel`):
      "fla"   -> flash-linear-attention's fused chunk_gated_delta_rule (CUDA, fast,
                 torch.compile-safe). Used when available on a CUDA tensor.
      "torch" -> the pure-PyTorch gated_delta_chunked (eager, compile-disabled).
      "auto"  -> fla if installed + CUDA, else torch.
    Both backends use FLA's native convention (decay-first, undecayed write, self-inclusive
    readout M_t q_t), so the FLA path passes q,k,v straight through -- NO value pre-scaling.
    g = logsigmoid(gamma_logit) = log(gamma) (log-decay), beta = sigmoid(beta_logit),
    use_qk_l2norm_in_kernel=True handles the unit-key requirement, and scale=1.0 washes out
    because the readout is RMS-normed afterwards. They agree up to bf16-vs-fp32 numerics
    (validate with verify_fla.py)."""

    def __init__(self, dim, num_heads, head_dim, linear_cls, chunk=64,
                 gamma_bias=3.9, beta_bias=0.0, kernel="auto"):
        super().__init__()
        self.H, self.dk, self.chunk, self.kernel = num_heads, head_dim, chunk, kernel
        self.gamma_bias, self.beta_bias = gamma_bias, beta_bias
        H, dk = num_heads, head_dim
        self.w_gamma = linear_cls(dim, H)         # data-dependent retention logits
        self.w_beta = linear_cls(dim, H)          # data-dependent write logits
        self.gate = linear_cls(dim, H * dk)       # output gate (per head-channel)
        self.proj = linear_cls(H * dk, dim)       # readout -> residual (zero-init)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x, q_t, k_t, v_t):
        """x: (B,T,D); q_t,k_t,v_t: (B,H,T,dk) (KV heads already expanded). -> (B,T,D)."""
        B, H, T, dk = q_t.shape
        g_logit = self.w_gamma(x).float() + self.gamma_bias            # (B,T,H)
        b_logit = self.w_beta(x).float() + self.beta_bias
        use_fla = (self.kernel in ("auto", "fla") and _HAS_FLA and q_t.is_cuda)

        if use_fla:
            # gated_delta_chunked / gated_delta_sequential now use FLA's native convention,
            # so pass q,k,v straight through (no value pre-scaling). g = log(gamma) decay,
            # in-kernel L2-norm handles the unit-key requirement; scale washes out post-RMSNorm.
            g = F.logsigmoid(g_logit)                                  # log-decay (<=0), (B,T,H)
            beta = torch.sigmoid(b_logit)
            q = q_t.transpose(1, 2).contiguous()                       # (B,T,H,dk)
            k = k_t.transpose(1, 2).contiguous()
            v = v_t.transpose(1, 2).contiguous()
            r = _fla_gated_delta(q, k, v, g.contiguous(), beta.contiguous())  # (B,T,H,dk), compile-opaque
        else:
            q = F.normalize(q_t.float(), dim=-1)                       # unit keys/queries
            k = F.normalize(k_t.float(), dim=-1)
            gamma = torch.sigmoid(g_logit).transpose(1, 2)             # (B,H,T)
            beta = torch.sigmoid(b_logit).transpose(1, 2)
            r, _ = gated_delta_chunked(q, k, v_t.float(), gamma, beta, chunk=self.chunk)  # (B,H,T,dk)
            r = r.transpose(1, 2)                                      # -> (B,T,H,dk)

        r = F.rms_norm(r, (dk,))                                       # bf16 from FLA (no fp32 upcast copy)
        r_flat = r.reshape(B, T, H * dk).to(x.dtype)
        return self.proj(r_flat * torch.sigmoid(self.gate(x)))


class AtmaConvBase(nn.Module):
    """Shared __init__ for LFM2 gated conv block. Subclass must implement forward()."""

    def __init__(self, dim: int, linear_cls, kernel_size: int = 3):
        super().__init__()
        self.hidden_size = dim
        self.kernel_size = kernel_size
        self.in_proj = linear_cls(dim, 3 * dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size - 1, groups=dim, bias=False)
        self.out_proj = linear_cls(dim, dim)

    def forward(self, x):
        raise NotImplementedError


class AtmaAttnBase(nn.Module):
    """Shared __init__ for Canon-B attention block. Subclass must implement forward()."""

    def __init__(self, dim: int, linear_cls, head_dim: int = 128, num_kv_heads: int = None, kernel_size: int = 4,
                 num_heads: int = None):
        super().__init__()
        # num_heads defaults to dim // head_dim, but can be set explicitly to match a
        # trained checkpoint whose head count is not dim // head_dim (hdim != dim).
        self.num_heads = num_heads if num_heads is not None else dim // head_dim
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else self.num_heads // 4
        self.head_dim = head_dim
        self.hdim = self.num_heads * self.head_dim
        self.kv_hdim = self.num_kv_heads * self.head_dim
        self.kernel_size = kernel_size

        self.q = linear_cls(dim, self.hdim * 2)
        self.k = linear_cls(dim, self.kv_hdim)
        self.v = linear_cls(dim, self.kv_hdim)
        self.canon_q = nn.Conv1d(self.hdim,    self.hdim,    kernel_size=kernel_size, padding=kernel_size - 1, groups=self.hdim,    bias=False)
        self.canon_k = nn.Conv1d(self.kv_hdim, self.kv_hdim, kernel_size=kernel_size, padding=kernel_size - 1, groups=self.kv_hdim, bias=False)
        self.canon_v = nn.Conv1d(self.kv_hdim, self.kv_hdim, kernel_size=kernel_size, padding=kernel_size - 1, groups=self.kv_hdim, bias=False)
        self.proj = linear_cls(self.hdim, dim)

    def forward(self, x):
        raise NotImplementedError
