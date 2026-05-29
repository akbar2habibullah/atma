"""Correctness tests for the Triton polar-attention kernel.

Oracle = model.blocks.polar_attention_online (gradchecked to ~1e-15 in
verify_polar_online.py) and polar_reduce (materialized). Run from repo root:

    python -m kernel.test_polar_kernel
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import torch.nn.functional as F

from model.blocks import polar_reduce, polar_attention_online
from kernel.polar_triton import polar_attention, polar_attention_fwd

torch.manual_seed(0)
dev = "cuda"
PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  {extra}" if extra else ""))
    PASS += bool(cond); FAIL += (not cond)


def materialized(q, k, v, n_keys, params, eps=1e-6):
    B, H, T, dk = q.shape
    sigma = torch.matmul(q, k.transpose(-2, -1)) * (dk ** -0.5)
    mask = torch.triu(torch.full((T, T), float("-inf"), dtype=sigma.dtype, device=q.device), 1)
    return polar_reduce(sigma + mask, v, n_keys, eps=eps, **params)


def make_params(H, dk, dtype):
    return {
        "v_null": (torch.randn(H, dk, device=dev, dtype=dtype) * 0.1),
        "null_base": torch.full((H,), 2.0, device=dev, dtype=dtype),
        "null_slope_raw": torch.full((H,), 0.5, device=dev, dtype=dtype),
        "len_gain_raw": torch.full((H,), -1.0, device=dev, dtype=dtype),
        "mag_beta_raw": torch.full((H,), -1.5, device=dev, dtype=dtype),
    }


# ── 1. forward parity (fp32) ──────────────────────────────────────────────
# The kernel reproduces the *online* (streaming) formulation m_eff = L^3/(Q2*Z),
# which is its exact target. It is algebraically identical to the materialized
# 1/sum(w_hat^2) but, like polar_attention_online itself, diverges from it by the
# fp32 summation order on rare ill-conditioned queries at long T -- so the
# materialized comparison uses a looser bound, gated by the online-vs-materialized
# gap (the kernel must not be worse than the existing online path).
print("\n── forward parity vs online + materialized polar_reduce (fp32) ──")
for (B, H, T, dk) in [(1, 1, 16, 64), (2, 4, 128, 64), (2, 8, 200, 128),
                      (1, 2, 512, 128), (3, 4, 65, 128), (2, 3, 1000, 128)]:
    q = torch.randn(B, H, T, dk, device=dev)
    k = torch.randn(B, H, T, dk, device=dev)
    v = torch.randn(B, H, T, dk, device=dev)
    p = make_params(H, dk, torch.float32)
    n_keys = torch.arange(1, T + 1, device=dev, dtype=torch.float32)
    c_mat, m_mat = materialized(q, k, v, n_keys, p)
    c_on, m_on = polar_attention_online(q, k, v, n_keys, k_block=64, **p)
    c_t, m_t = polar_attention_fwd(q, k, v, n_keys, **p)
    # tight: kernel vs the streaming oracle it targets
    dc_on = (c_on - c_t).abs().max().item()
    dm_on = (m_on - m_t).abs().max().item()
    # loose: vs materialized, must not exceed online-vs-materialized
    dm_mat = (m_mat - m_t).abs().max().item()
    dm_on_mat = (m_mat - m_on).abs().max().item()
    cn = c_t.norm(dim=-1)
    ok = (dc_on < 1e-5 and dm_on < 1e-5
          and dm_mat <= max(dm_on_mat * 1.5, 1e-4)
          and cn.max() <= 1.0001 and m_t.min() >= 0 and m_t.max() < 1.0)
    check(f"T={T:<4} dk={dk}", ok,
          f"vs_online: dc={dc_on:.1e} dmag={dm_on:.1e} | vs_mat dmag={dm_mat:.1e} (online_gap={dm_on_mat:.1e})")


# ── 2. backward parity vs online oracle (fp32) ────────────────────────────
print("\n── backward parity vs polar_attention_online (fp32) ──")


def bwd_case(B, H, T, dk, dtype, ftol, gtol):
    base = {kk: vv.clone() for kk, vv in make_params(H, dk, dtype).items()}
    qd = torch.randn(B, H, T, dk, device=dev, dtype=dtype)
    kd = torch.randn(B, H, T, dk, device=dev, dtype=dtype)
    vd = torch.randn(B, H, T, dk, device=dev, dtype=dtype)
    n_keys = torch.arange(1, T + 1, device=dev, dtype=torch.float32)
    gc = torch.randn(B, H, T, dk, device=dev, dtype=dtype)
    gm = torch.randn(B, H, T, device=dev, dtype=dtype)

    def leafs():
        q = qd.clone().requires_grad_(True)
        k = kd.clone().requires_grad_(True)
        v = vd.clone().requires_grad_(True)
        p = {kk: vv.clone().requires_grad_(True) for kk, vv in base.items()}
        return q, k, v, p

    q1, k1, v1, p1 = leafs()
    c1, m1 = polar_attention_online(q1, k1, v1, n_keys, k_block=64, **p1)
    (c1.float() * gc.float() + m1.float().unsqueeze(-1) * gm.float().unsqueeze(-1)).sum().backward()

    q2, k2, v2, p2 = leafs()
    c2, m2 = polar_attention(q2, k2, v2, n_keys, **p2)
    (c2.float() * gc.float() + m2.float().unsqueeze(-1) * gm.float().unsqueeze(-1)).sum().backward()

    tag = f"[{dtype} T={T} dk={dk}]"
    worst = 0.0
    for nm, a, b in [("dq", q1.grad, q2.grad), ("dk", k1.grad, k2.grad), ("dv", v1.grad, v2.grad)] + \
                    [(kk, p1[kk].grad, p2[kk].grad) for kk in p1]:
        d = (a.float() - b.float()).abs().max().item()
        rel = d / (a.float().abs().max().item() + 1e-9)
        worst = max(worst, rel)
        check(f"{tag} {nm}", rel < gtol, f"abs={d:.1e} rel={rel:.1e}")
    return worst


for (B, H, T, dk) in [(1, 1, 16, 64), (2, 4, 130, 64), (2, 4, 200, 128), (1, 2, 300, 128)]:
    bwd_case(B, H, T, dk, torch.float32, 1e-4, 1e-3)

# ── 3. backward parity (bf16 / fp16, tensor-core dots, loose) ─────────────
print("\n── backward parity (bf16/fp16, loose tol) ──")
for (B, H, T, dk) in [(2, 4, 128, 128), (2, 8, 256, 128)]:
    bwd_case(B, H, T, dk, torch.bfloat16, 5e-2, 5e-2)
for (B, H, T, dk) in [(2, 8, 256, 128), (2, 4, 128, 64)]:
    bwd_case(B, H, T, dk, torch.float16, 5e-2, 5e-2)

# ── 3b. edge shapes: T smaller than BLOCK_M, T=1 (decode-like), odd T ──────
print("\n── edge shapes (T<BLOCK_M, T=1, odd T) ──")
for (B, H, T, dk) in [(1, 4, 8, 128), (2, 2, 1, 128), (1, 4, 17, 64), (3, 2, 33, 128)]:
    bwd_case(B, H, T, dk, torch.bfloat16, 5e-2, 5e-2)

# ── 4. non-contiguous inputs (transposed, like the real call site) ────────
print("\n── non-contiguous inputs through autograd ──")
B, H, T, dk = 2, 4, 96, 128
x = torch.randn(B, T, H, dk, device=dev, requires_grad=True)
q = x.transpose(1, 2)                       # (B,H,T,dk), non-contiguous
k = torch.randn(B, H, T, dk, device=dev)
v = torch.randn(B, H, T, dk, device=dev)
p = {kk: vv.requires_grad_(True) for kk, vv in make_params(H, dk, torch.float32).items()}
n_keys = torch.arange(1, T + 1, device=dev, dtype=torch.float32)
c, m = polar_attention(q, k, v, n_keys, **p)
(c.sum() + m.sum()).backward()
check("non-contiguous q forward+backward finite",
      torch.isfinite(c).all() and torch.isfinite(m).all() and torch.isfinite(x.grad).all())

# ── 5. n_keys=0 padding query: must drain entirely to the null sink (mask uses
#       RAW n_keys; clamp-to-1 only feeds temp/null, exactly like the oracle) ──
print("\n── n_keys=0 padding query matches oracle ──")
B, H, Tk, dk = 2, 3, 16, 128
q = torch.randn(B, H, 4, dk, device=dev)
k = torch.randn(B, H, Tk, dk, device=dev)
v = torch.randn(B, H, Tk, dk, device=dev)
p = make_params(H, dk, torch.float32)
nk = torch.tensor([5., 0., 12., 0.], device=dev)            # queries 1,3 are padding (0 valid keys)
co, mo = polar_attention_online(q, k, v, nk, k_block=8, **p)
ct, mt = polar_attention_fwd(q, k, v, nk, is_causal=False, **p)
dcz = (co - ct).abs().max().item(); dmz = (mo - mt).abs().max().item()
check("n_keys=0 forward matches oracle", dcz < 1e-4 and dmz < 1e-4,
      f"dc={dcz:.1e} dmag={dmz:.1e}; padding mag={mt[0,0,1].item():.4f},{mt[0,0,3].item():.4f} (oracle "
      f"{mo[0,0,1].item():.4f},{mo[0,0,3].item():.4f})")

print(f"\n{'='*52}\nResults: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
