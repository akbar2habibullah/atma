"""GPU profiling harness for the Titans memory branch (gated_delta_chunked).

Why: under torch.compile (train.py wraps the model), torch.linalg.solve_triangular is a
non-fused extern call (a sync point per chunk), and fp32 matmuls skip tensor cores. This
script measures fwd+bwd time across:
  - inverse method : "tri" (linalg.solve_triangular)  vs  "neumann" (matmul-only doubling,
                     exact for the unit lower-triangular nilpotent system -> stays in the
                     inductor graph, hits tensor cores in bf16)
  - chunk size     : 32 / 64 / 128 / 256
  - dtype          : fp32 vs bf16 inputs
  - eager vs torch.compile
and reports the polar attention cost at the same shape for context.

Run on the GPU box:  python bench_mem.py
The fastest (method, chunk, dtype, compiled) combo tells us what to put in the model.
"""

import math
import time
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, H, N, dk = 4, 8, 2048, 128


def _inv_neumann(L, steps):
    """(I + L)^{-1} for strictly-lower (nilpotent) L via doubling: only matmuls, no linalg.
    S_{m+1} = (I + L^{2^m}) S_m ... here with X=-L: sum_{i>=0} (-L)^i (terminates)."""
    C = L.shape[-1]
    eye = torch.eye(C, dtype=L.dtype, device=L.device)
    inv = eye.expand_as(L)
    P = -L
    for _ in range(steps):
        inv = inv + torch.matmul(P, inv)
        P = torch.matmul(P, P)
    return inv


def chunked(q, k, v, gamma, beta, chunk, method):
    """gated_delta_chunked with a switchable inverse (mirrors model/blocks.py)."""
    B, H, N, dk = q.shape
    dv = v.shape[-1]
    dtype, device = q.dtype, q.device
    steps = max(1, math.ceil(math.log2(chunk)))
    S = torch.zeros(B, H, dv, dk, dtype=dtype, device=device)
    Rs = []
    for cs in range(0, N, chunk):
        ce = min(cs + chunk, N)
        C = ce - cs
        qc, kc, vc = q[:, :, cs:ce], k[:, :, cs:ce], v[:, :, cs:ce]
        gc_, bc = gamma[:, :, cs:ce], beta[:, :, cs:ce]
        gb = gc_ * bc
        clg = torch.cumsum(torch.log(gc_), dim=-1)
        Lgfull = torch.cat([torch.zeros_like(clg[..., :1]), clg], dim=-1)
        Lp, Ls1 = Lgfull[..., :C], Lgfull[..., 1:C + 1]
        idx = torch.arange(C, device=device)
        strict = idx[:, None] > idx[None, :]
        ratio = torch.exp((Lp[..., :, None] - Ls1[..., None, :]).masked_fill(~strict, float("-inf")))
        D = ratio * torch.einsum("bhpd,bhsd->bhps", kc, kc)
        Rq = ratio * torch.einsum("bhpd,bhsd->bhps", qc, kc)
        carry = torch.exp(Lp)
        c_mat = carry[..., None] * torch.einsum("bhvd,bhcd->bhcv", S, kc)
        cprime = carry[..., None] * torch.einsum("bhvd,bhcd->bhcv", S, qc)
        L = gb[..., :, None] * D
        rhs = gb[..., :, None] * (vc - c_mat)
        if method == "tri":
            A = torch.linalg.solve_triangular(L, rhs, upper=False, unitriangular=True)
        else:
            A = torch.matmul(_inv_neumann(L, steps), rhs)
        Rs.append(cprime + torch.einsum("bhij,bhjv->bhiv", Rq, A))
        out_ratio = torch.exp(Lgfull[..., C:C + 1] - Ls1)
        gC = torch.exp(Lgfull[..., C])
        S = gC[..., None, None] * S + torch.einsum("bhcv,bhcd->bhvd", out_ratio[..., None] * A, kc)
    return torch.cat(Rs, dim=2)


def _mk(dtype):
    q = F.normalize(torch.randn(B, H, N, dk, device=DEV), dim=-1).to(dtype).requires_grad_(True)
    k = F.normalize(torch.randn(B, H, N, dk, device=DEV), dim=-1).to(dtype).requires_grad_(True)
    v = torch.randn(B, H, N, dk, device=DEV, dtype=dtype, requires_grad=True)
    g = torch.sigmoid(3.9 + 0.3 * torch.randn(B, H, N, device=DEV)).to(dtype)
    b = torch.sigmoid(0.3 * torch.randn(B, H, N, device=DEV)).to(dtype)
    return q, k, v, g, b


def _time(fn, args, iters=10):
    r = fn(*args); r.sum().backward()                # warmup (also triggers compile)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for t in args[:3]:
            t.grad = None
        r = fn(*args)
        r.sum().backward()
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    print(f"device={DEV}  shape B={B} H={H} N={N} dk={dk}\n")
    print(f"{'method':>8} {'chunk':>6} {'dtype':>6} {'eager(ms)':>11} {'compiled(ms)':>13}")
    print("-" * 50)
    for dtype in (torch.float32, torch.bfloat16):
        for method in ("tri", "neumann"):
            for chunk in (32, 64, 128, 256):
                args = _mk(dtype)
                eager = _time(lambda *a: chunked(*a, chunk=chunk, method=method), args)
                try:
                    cfn = torch.compile(lambda *a: chunked(*a, chunk=chunk, method=method))
                    comp = _time(cfn, _mk(dtype))
                except Exception as e:
                    comp = float("nan")
                    print(f"   compile failed ({method},{chunk},{dtype}): {e}")
                print(f"{method:>8} {chunk:>6} {str(dtype).split('.')[-1]:>6} {eager:>11.1f} {comp:>13.1f}")
    print("\nPick the fastest (method, chunk, dtype, compiled) -> promote into model/blocks.py")


if __name__ == "__main__":
    main()
