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
    B, H, Tq, Tk = sigma.shape
    dk = v.shape[-1]
    sigma = sigma.float()
    v = v.float()

    temp, null = polar_temp_null(n_keys, len_gain_raw, null_base, null_slope_raw)
    logits = torch.cat([sigma, null.expand(B, H, Tq, 1)], dim=-1) * temp
    w = torch.softmax(logits, dim=-1)
    w_null = w[..., -1:]          # (B, H, Tq, 1)  mass drained to the null sink
    w_r = w[..., :-1]             # (B, H, Tq, Tk) weights over real keys

    # direction channel
    s = torch.matmul(w_r, v) + w_null * v_null.float().view(1, H, 1, dk)
    c = F.normalize(s, p=2, dim=-1, eps=eps)

    # count/magnitude channel — participation ratio gated by confidence, bounded
    denom = w_r.sum(-1, keepdim=True).clamp_min(eps)
    w_hat = w_r / denom
    n_eff = 1.0 / w_hat.square().sum(-1).clamp_min(eps)        # (B, H, Tq)
    m_eff = n_eff * (1.0 - w_null.squeeze(-1))
    mag = torch.tanh(F.softplus(mag_beta_raw).view(1, H, 1) * torch.log1p(m_eff))

    return c.to(out_dtype), mag.to(out_dtype)


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
