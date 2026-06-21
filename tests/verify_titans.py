"""Verify the Titans linear-memory prototype (Step 1 of plans/linked-forging-sparrow.md).

Bar (same as the old verify_polar_online): fp64 autograd correctness + chunked == sequential
forward AND backward parity. Exits non-zero on any failure.

  1. forward parity   : gated_delta_chunked == gated_delta_sequential across chunk sizes
  2. backward parity  : grads of a scalar loss wrt (q,k,v,gamma,beta) agree
  3. gradcheck        : torch.autograd.gradcheck on the chunked fn (analytic vs numeric jacobian)
"""

import sys
import torch

from scripts.titans_proto import gated_delta_sequential, gated_delta_chunked, make_inputs

FAILS = []


def _check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def test_forward_parity():
    print("=== 1. forward parity (chunked == sequential, fp64) ===")
    q, k, v, g, b = make_inputs(2, 3, 257, 16, 24, seed=11)   # odd N exercises ragged last chunk
    Rseq, Sseq = gated_delta_sequential(q, k, v, g, b)
    for C in (1, 8, 32, 64, 128, 257):
        Rchk, Schk = gated_delta_chunked(q, k, v, g, b, chunk=C)
        dR = (Rseq - Rchk).abs().max().item()
        dS = (Sseq - Schk).abs().max().item()
        _check(f"chunk={C} forward", dR < 1e-9 and dS < 1e-9, f"max|dR|={dR:.2e} max|dS|={dS:.2e}")


def test_backward_parity():
    print("=== 2. backward parity (grads agree, fp64) ===")
    q, k, v, g, b = make_inputs(2, 2, 96, 12, 12, seed=23)
    # ONE fixed random readout-weighting, shared by both paths so the loss is identical
    wts = torch.randn(2, 2, 96, 12, generator=torch.Generator().manual_seed(5), dtype=q.dtype)

    def loss_of(C=None):
        qr, kr, vr, gr, br = (t.clone().requires_grad_(True) for t in (q, k, v, g, b))
        R, S = (gated_delta_sequential(qr, kr, vr, gr, br) if C is None
                else gated_delta_chunked(qr, kr, vr, gr, br, chunk=C))
        L = (R * wts).sum() + S.pow(2).sum()      # exercise both readouts and final state
        L.backward()
        return L.item(), [t.grad for t in (qr, kr, vr, gr, br)]

    L0, g0 = loss_of(None)
    for C in (8, 32, 64):
        LC, gC = loss_of(C)
        gmax = max((a - b).abs().max().item() for a, b in zip(g0, gC))
        _check(f"chunk={C} backward", abs(L0 - LC) < 1e-9 and gmax < 1e-8,
               f"|dL|={abs(L0 - LC):.2e} max|dgrad|={gmax:.2e}")


def test_gradcheck():
    print("=== 3. gradcheck on chunked fn (fp64) ===")
    B, H, N, dk, dv, C = 1, 2, 8, 3, 3, 4
    q, k, v, g, b = make_inputs(B, H, N, dk, dv, seed=31)

    def fn(q, k, v, g, b):
        R, S = gated_delta_chunked(q, k, v, g, b, chunk=C)
        return R, S

    inputs = tuple(t.clone().requires_grad_(True) for t in (q, k, v, g, b))
    try:
        ok = torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-6, rtol=1e-4)
    except Exception as e:  # gradcheck raises on mismatch
        ok = False
        print(f"      gradcheck raised: {e}")
    _check("gradcheck chunked", ok)


if __name__ == "__main__":
    torch.manual_seed(0)
    test_forward_parity()
    test_backward_parity()
    test_gradcheck()
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} -> {FAILS}")
        sys.exit(1)
    print("ALL PASS")
