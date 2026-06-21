"""Verify the trainable windowed polar backward (Step 2 of plans/linked-forging-sparrow.md).

The materialized `polar_reduce` is the oracle: a sliding window of W is equivalent to
band-masking sigma (key j invalid for query i when j < n_keys[i] - W) AND feeding the
windowed count n_count = min(n_keys, W) to polar_temp_null. That is exactly what the
online streaming path now does, so the two must agree in forward AND backward (fp64).

This validates the PyTorch `_PolarOnline` band backward. The Triton kernel band backward
mirrors the same masking but needs a CUDA box to run (see kernel/polar_triton.py).
"""

import sys
import torch

from model.blocks import polar_reduce, polar_attention_online

FAILS = []


def _check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def _params(H, dk, dtype, seed=0):
    """Params centered on the model's structural-prior inits (train/model.py) with small
    noise. NOTE: keep len_gain moderate -- a huge temp makes the softmax a near-delta with
    w_null~1, where n_eff (real-key participation ratio) is ill-defined and the materialized
    1/sum(w_hat^2).clamp(1e-6) path and the online L^3/(Q2 Z) path disagree (both give
    negligible mag). That degenerate regime is a pre-existing materialized-vs-online wart,
    not a property of the band backward; realistic operating points never reach it."""
    g = torch.Generator().manual_seed(seed)

    def around(center, n):
        return center + 0.3 * torch.randn(n, generator=g, dtype=dtype)

    return dict(
        v_null=torch.randn(H, dk, generator=g, dtype=dtype),
        null_base=around(2.0, H),       # _NULL_BASE_INIT
        null_slope_raw=around(0.5, H),  # _NULL_SLOPE_INIT
        len_gain_raw=around(-1.0, H),   # _LEN_GAIN_INIT
        mag_beta_raw=around(-1.5, H),   # _MAG_BETA_INIT
    )


def _materialized_window(q, k, v, n_keys, window, params):
    """Oracle: band-masked materialized polar_reduce with windowed count."""
    B, H, T, dk = q.shape
    sigma = torch.matmul(q, k.transpose(-2, -1)) / (dk ** 0.5)
    kidx = torch.arange(T, device=q.device).view(1, 1, 1, T)
    nk = n_keys.view(1, 1, T, 1)
    invalid = kidx >= nk                                  # future
    if window is not None:
        invalid = invalid | (kidx < (nk - window))        # older than window
    sigma = sigma.masked_fill(invalid, float("-inf"))
    n_count = n_keys if window is None else torch.minimum(n_keys, n_keys.new_tensor(float(window)))
    return polar_reduce(sigma, v, n_count, **params)


def _rand_inputs(B, H, T, dk, dtype, seed=1):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, dk, generator=g, dtype=dtype)
    k = torch.randn(B, H, T, dk, generator=g, dtype=dtype)
    v = torch.randn(B, H, T, dk, generator=g, dtype=dtype)
    n_keys = torch.arange(1, T + 1, dtype=dtype)
    return q, k, v, n_keys


def test_parity():
    print("=== forward+backward parity: online window vs materialized oracle (fp64) ===")
    B, H, T, dk = 2, 3, 16, 8
    dtype = torch.float64
    params = _params(H, dk, dtype)
    q0, k0, v0, n_keys = _rand_inputs(B, H, T, dk, dtype)
    wts_c = torch.randn(B, H, T, dk, generator=torch.Generator().manual_seed(9), dtype=dtype)
    wts_m = torch.randn(B, H, T, generator=torch.Generator().manual_seed(10), dtype=dtype)

    def run(online, window, k_block):
        ins = {n: t.clone().requires_grad_(True) for n, t in params.items()}
        q, k, v = (t.clone().requires_grad_(True) for t in (q0, k0, v0))
        if online:
            c, mag = polar_attention_online(q, k, v, n_keys, k_block=k_block, window=window, **ins)
        else:
            c, mag = _materialized_window(q, k, v, n_keys, window, ins)
        L = (c * wts_c).sum() + (mag * wts_m).sum()
        L.backward()
        grads = {"q": q.grad, "k": k.grad, "v": v.grad, **{n: ins[n].grad for n in ins}}
        return c.detach(), mag.detach(), L.item(), grads

    for window in (None, 4, 7, 32):           # 32 > T exercises the "window disables" path
        for k_block in (2, 5, 16):            # block boundaries crossing the band
            c_o, m_o, L_o, g_o = run(True, window, k_block)
            c_m, m_m, L_m, g_m = run(False, window, k_block)
            dc = (c_o - c_m).abs().max().item()
            dmag = (m_o - m_m).abs().max().item()
            dgrad = max((g_o[n] - g_m[n]).abs().max().item() for n in g_o)
            ok = dc < 1e-10 and dmag < 1e-10 and dgrad < 1e-9
            _check(f"window={window} k_block={k_block}", ok,
                   f"dc={dc:.1e} dmag={dmag:.1e} dgrad={dgrad:.1e}")


def test_gradcheck():
    print("=== gradcheck on online windowed path (fp64) ===")
    B, H, T, dk, W = 1, 2, 8, 4, 3
    dtype = torch.float64
    params = _params(H, dk, dtype, seed=3)
    q0, k0, v0, n_keys = _rand_inputs(B, H, T, dk, dtype, seed=2)

    pnames = list(params.keys())
    q = q0.clone().requires_grad_(True)
    k = k0.clone().requires_grad_(True)
    v = v0.clone().requires_grad_(True)
    pt = [params[n].clone().requires_grad_(True) for n in pnames]

    def fn(q, k, v, *ps):
        kw = dict(zip(pnames, ps))
        return polar_attention_online(q, k, v, n_keys, k_block=2, window=W, **kw)

    try:
        ok = torch.autograd.gradcheck(fn, (q, k, v, *pt), eps=1e-6, atol=1e-6, rtol=1e-4)
    except Exception as e:
        ok = False
        print(f"      gradcheck raised: {e}")
    _check("gradcheck online window", ok)


if __name__ == "__main__":
    torch.manual_seed(0)
    test_parity()
    test_gradcheck()
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} -> {FAILS}")
        sys.exit(1)
    print("ALL PASS")
