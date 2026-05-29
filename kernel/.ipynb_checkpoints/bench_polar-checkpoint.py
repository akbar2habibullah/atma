"""Benchmark: Triton polar attention vs the PyTorch online & materialized paths.

    python -m kernel.bench_polar
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch

from model.blocks import polar_reduce, polar_attention_online
from kernel.polar_triton import polar_attention

dev = "cuda"
torch.manual_seed(0)


def materialized(q, k, v, n_keys, p):
    B, H, T, dk = q.shape
    sigma = torch.matmul(q, k.transpose(-2, -1)) * (dk ** -0.5)
    mask = torch.triu(torch.full((T, T), float("-inf"), device=dev, dtype=sigma.dtype), 1)
    return polar_reduce(sigma + mask, v, n_keys, **p)


def timed(fn, bwd, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def peak_mem(fn):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


def bench(B, H, T, dk, dtype=torch.bfloat16):
    q = torch.randn(B, H, T, dk, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, dk, device=dev, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, dk, device=dev, dtype=dtype, requires_grad=True)
    p = dict(v_null=(torch.randn(H, dk, device=dev, dtype=dtype) * 0.1).requires_grad_(True),
             null_base=torch.full((H,), 2.0, device=dev, dtype=dtype, requires_grad=True),
             null_slope_raw=torch.full((H,), 0.5, device=dev, dtype=dtype, requires_grad=True),
             len_gain_raw=torch.full((H,), -1.0, device=dev, dtype=dtype, requires_grad=True),
             mag_beta_raw=torch.full((H,), -1.5, device=dev, dtype=dtype, requires_grad=True))
    n_keys = torch.arange(1, T + 1, device=dev, dtype=torch.float32)
    gc = torch.randn(B, H, T, dk, device=dev, dtype=dtype)
    gm = torch.randn(B, H, T, device=dev, dtype=dtype)

    def fwd_bwd(fn):
        c, m = fn()
        (c.float() * gc.float() + m.float().unsqueeze(-1) * gm.float().unsqueeze(-1)).sum().backward()

    impls = {
        "triton": lambda: polar_attention(q, k, v, n_keys, **p),
        "online": lambda: polar_attention_online(q, k, v, n_keys, k_block=512, **p),
    }
    if T <= 2048:
        impls["material"] = lambda: materialized(q, k, v, n_keys, p)

    print(f"\n[B={B} H={H} T={T} dk={dk} {dtype}]")
    print(f"  {'impl':<10} {'fwd ms':>9} {'fwd+bwd ms':>12} {'peak MB':>9}")
    base_fb = None
    for name, fn in impls.items():
        try:
            tf = timed(lambda: fn()[0].sum(), None)
            tfb = timed(lambda: fwd_bwd(fn), True)
            mem = peak_mem(lambda: fwd_bwd(fn))
            if name == "triton":
                base_fb = tfb
            sp = "" if base_fb is None or name == "triton" else f"  ({tfb/base_fb:.2f}x)"
            print(f"  {name:<10} {tf:>9.3f} {tfb:>12.3f} {mem:>9.1f}{sp}")
        except RuntimeError as e:
            print(f"  {name:<10}  OOM/err: {str(e)[:40]}")


for (B, H, T, dk) in [(4, 16, 512, 128), (2, 16, 1024, 128), (1, 16, 2048, 128),
                      (1, 16, 4096, 128), (1, 16, 8192, 128)]:
    bench(B, H, T, dk)
