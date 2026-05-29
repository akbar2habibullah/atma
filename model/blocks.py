import torch
from torch import nn
import torch.nn.functional as F


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
    """Polar attention reduction (validated in polar_proto.py). Computed in fp32.

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
                len_gain_raw, mag_beta_raw, k_block, eps):
        out_dtype = q.dtype
        cd = _compute_dtype(out_dtype)
        B, H, T, dk = q.shape
        Tk = k.shape[2]
        qd, kd, vd = q.to(cd), k.to(cd), v.to(cd)
        scale = dk ** -0.5

        n = n_keys.to(cd).clamp(min=1.0)
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
            invalid = key_idx[ks:ke].view(1, 1, 1, -1) >= n_keys.view(1, 1, T, 1)
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

        ctx.save_for_backward(q, k, v, n_keys, v_null, null_base, null_slope_raw,
                              len_gain_raw, mag_beta_raw, M, L, Q2, s)
        ctx.k_block, ctx.eps = k_block, eps
        return c.to(out_dtype), mag.to(out_dtype)

    @staticmethod
    def backward(ctx, gc, gm):
        (q, k, v, n_keys, v_null, null_base, null_slope_raw,
         len_gain_raw, mag_beta_raw, M, L, Q2, s) = ctx.saved_tensors
        k_block, eps = ctx.k_block, ctx.eps
        out_dtype = q.dtype
        cd = _compute_dtype(out_dtype)
        B, H, T, dk = q.shape
        Tk = k.shape[2]
        qd, kd, vd = q.to(cd), k.to(cd), v.to(cd)
        gc = gc.to(cd)
        gm = gm.to(cd)
        scale = dk ** -0.5

        n = n_keys.to(cd).clamp(min=1.0)
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
            invalid = key_idx[ks:ke].view(1, 1, 1, -1) >= n_keys.view(1, 1, T, 1)
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
                cast(grad_mag_beta, mag_beta_raw), None, None)


def polar_attention_online(q, k, v, n_keys, *, v_null, null_base, null_slope_raw,
                           len_gain_raw, mag_beta_raw, k_block=512, eps=1e-6):
    """Memory-efficient polar attention. q,k,v: (B,H,T,dk) with KV heads expanded.
    Streams keys in blocks of k_block -> O(B*H*T*k_block) peak (fwd and bwd).
    Numerically equals materialized scores + polar_reduce."""
    return _PolarOnline.apply(q, k, v, n_keys, v_null, null_base, null_slope_raw,
                              len_gain_raw, mag_beta_raw, k_block, eps)


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

    def __init__(self, dim: int, linear_cls, head_dim: int = 128, num_kv_heads: int = None, kernel_size: int = 4):
        super().__init__()
        self.num_heads = dim // head_dim
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
