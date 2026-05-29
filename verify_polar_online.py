"""
verify_polar_online.py — correctness of the online (memory-efficient) polar reduction.

  1. gradcheck (float64) on the custom autograd.Function, multi-block keys.
  2. forward parity vs materialized polar_reduce (the oracle).
  3. backward parity: grads to q,k,v and all params match the oracle's autograd.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F
from model.blocks import polar_reduce, polar_attention_online

torch.manual_seed(0)
PASS = FAIL = 0
def check(name, cond, extra=""):
    global PASS, FAIL
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  {extra}" if extra else ""))
    PASS += bool(cond); FAIL += (not cond)


def materialized(q, k, v, n_keys, params, eps=1e-6):
    """Oracle: build full scores + causal mask, call polar_reduce."""
    B, H, T, dk = q.shape
    sigma = torch.matmul(q, k.transpose(-2, -1)) * (dk ** -0.5)
    mask = torch.triu(torch.full((T, T), float("-inf"), dtype=sigma.dtype), diagonal=1)
    sigma = sigma + mask
    return polar_reduce(sigma, v, n_keys, eps=eps, **params)


# ── 1. gradcheck (float64, 3 key-blocks) ─────────────────────────────────────
print("\n── gradcheck (float64) ──")
B, H, T, dk = 1, 2, 5, 4
g = lambda *s: torch.randn(*s, dtype=torch.float64, requires_grad=True)
q, k, v = g(B, H, T, dk), g(B, H, T, dk), g(B, H, T, dk)
v_null = g(H, dk)
null_base = torch.zeros(H, dtype=torch.float64, requires_grad=True)
null_slope_raw = torch.zeros(H, dtype=torch.float64, requires_grad=True)
len_gain_raw = torch.zeros(H, dtype=torch.float64, requires_grad=True)
mag_beta_raw = torch.zeros(H, dtype=torch.float64, requires_grad=True)
n_keys = torch.arange(1, T + 1, dtype=torch.float64)

def fn(q, k, v, v_null, null_base, null_slope_raw, len_gain_raw, mag_beta_raw):
    return polar_attention_online(q, k, v, n_keys, v_null=v_null, null_base=null_base,
                                  null_slope_raw=null_slope_raw, len_gain_raw=len_gain_raw,
                                  mag_beta_raw=mag_beta_raw, k_block=2)

ok = torch.autograd.gradcheck(fn, (q, k, v, v_null, null_base, null_slope_raw, len_gain_raw, mag_beta_raw),
                              eps=1e-6, atol=1e-5, rtol=1e-3)
check("gradcheck passes (multi-block, causal)", ok)


# ── 2 & 3. forward + backward parity vs oracle ───────────────────────────────
print("\n── forward + backward parity vs materialized polar_reduce ──")
B, H, T, dk = 2, 3, 17, 8
params_keys = ["v_null", "null_base", "null_slope_raw", "len_gain_raw", "mag_beta_raw"]

def make_inputs():
    q = torch.randn(B, H, T, dk, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, H, T, dk, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, H, T, dk, dtype=torch.float64, requires_grad=True)
    p = {
        "v_null": torch.randn(H, dk, dtype=torch.float64, requires_grad=True),
        "null_base": torch.full((H,), 2.0, dtype=torch.float64, requires_grad=True),
        "null_slope_raw": torch.full((H,), 0.5, dtype=torch.float64, requires_grad=True),
        "len_gain_raw": torch.full((H,), -1.0, dtype=torch.float64, requires_grad=True),
        "mag_beta_raw": torch.full((H,), -1.5, dtype=torch.float64, requires_grad=True),
    }
    return q, k, v, p

n_keys = torch.arange(1, T + 1, dtype=torch.float64)

# shared random upstream grads so the two backward passes are comparable
gc = torch.randn(B, H, T, dk, dtype=torch.float64)
gm = torch.randn(B, H, T, dtype=torch.float64)

# oracle
q1, k1, v1, p1 = make_inputs()
c1, mag1 = materialized(q1, k1, v1, n_keys, p1)
(c1 * gc + mag1.unsqueeze(-1) * gm.unsqueeze(-1)).sum().backward()

# online (k_block forces multiple blocks)
q2, k2, v2, p2 = make_inputs()
# copy oracle's input values so grads are comparable
with torch.no_grad():
    q2.copy_(q1); k2.copy_(k1); v2.copy_(v1)
    for kk in params_keys: p2[kk].copy_(p1[kk])
c2, mag2 = polar_attention_online(q2, k2, v2, n_keys, k_block=4, **p2)
(c2 * gc + mag2.unsqueeze(-1) * gm.unsqueeze(-1)).sum().backward()

dc = (c1 - c2).abs().max().item()
dmag = (mag1 - mag2).abs().max().item()
check("forward c matches oracle", dc < 1e-9, f"max_diff={dc:.2e}")
check("forward mag matches oracle", dmag < 1e-9, f"max_diff={dmag:.2e}")

for name, a, b in [("grad_q", q1.grad, q2.grad), ("grad_k", k1.grad, k2.grad), ("grad_v", v1.grad, v2.grad)]:
    d = (a - b).abs().max().item()
    check(f"{name} matches oracle", d < 1e-8, f"max_diff={d:.2e}")
for kk in params_keys:
    d = (p1[kk].grad - p2[kk].grad).abs().max().item()
    check(f"grad {kk} matches oracle", d < 1e-8, f"max_diff={d:.2e}")

print(f"\n{'='*50}\nResults: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
