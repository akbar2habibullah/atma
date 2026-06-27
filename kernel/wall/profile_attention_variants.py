"""Isolated attention/core memory profiler for the Wall investigation.

This script is intentionally small and synthetic. It compares the attention core
memory shape, not the full model. Use it to test whether Wall's fused op itself
is the cause of a full-model memory spike before making claims about the model.

Example:
    CPATH=/path/to/cuda/include python -m kernel.wall.profile_attention_variants
"""

from __future__ import annotations

import argparse
import gc
import inspect
import sys
import time
import types

import torch
import torch.nn.functional as F


def _install_einops_reduce_shim() -> None:
    if "einops" in sys.modules:
        return

    def reduce(x, pattern, *, g, reduction):
        if reduction != "sum":
            raise NotImplementedError(reduction)
        if pattern == "b t (h g) k -> b t h k":
            b, t, hg, k = x.shape
            return x.reshape(b, t, hg // g, g, k).sum(dim=3)
        if pattern == "b t (h g) v -> b t h v":
            b, t, hg, v = x.shape
            return x.reshape(b, t, hg // g, g, v).sum(dim=3)
        raise NotImplementedError(pattern)

    einops = types.ModuleType("einops")
    einops.reduce = reduce
    sys.modules["einops"] = einops


def _load_wall_impls():
    from kernel.wall import wall_attn as local_wall

    _install_einops_reduce_shim()
    sys.path.insert(0, "/home/sagemaker-user/wall-attention-release")
    try:
        from wall_attn import wall_attn as upstream_wall
    except Exception:
        upstream_wall = None
    return local_wall, upstream_wall


def _reset():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()


def _finish(label, status, elapsed_ms=None, error=None):
    torch.cuda.synchronize()
    fields = {
        "label": label,
        "status": status,
        "peak_alloc_gb": f"{torch.cuda.max_memory_allocated() / 1024**3:.3f}",
        "peak_reserved_gb": f"{torch.cuda.max_memory_reserved() / 1024**3:.3f}",
    }
    if elapsed_ms is not None:
        fields["elapsed_ms"] = f"{elapsed_ms:.2f}"
    if error is not None:
        fields["error"] = str(error).splitlines()[0]
    print(",".join(f"{k}={v}" for k, v in fields.items()), flush=True)


def run_wall(label, wall_fn, args, window):
    _reset()
    try:
        q = torch.randn(args.B, args.T, args.HQ, args.K, device="cuda", dtype=args.dtype, requires_grad=True)
        k = torch.randn(args.B, args.T, args.H, args.K, device="cuda", dtype=args.dtype, requires_grad=True)
        v = torch.randn(args.B, args.T, args.H, args.V, device="cuda", dtype=args.dtype, requires_grad=True)
        g = -(torch.rand(args.B, args.T, args.HQ, args.K, device="cuda", dtype=args.dtype) * 0.02 + 0.01).requires_grad_(True)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        o = wall_fn(q, k, v, g, scale=args.K ** -0.5, window_size=window)
        o.float().square().mean().backward()
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1000
        _finish(label, "ok", elapsed_ms=dt)
    except RuntimeError as e:
        _finish(label, "oom" if "out of memory" in str(e).lower() else "error", error=e)


def run_sdpa(label, args, window, rope_scale=False):
    _reset()
    try:
        q = torch.randn(args.B, args.T, args.HQ, args.K, device="cuda", dtype=args.dtype, requires_grad=True)
        k = torch.randn(args.B, args.T, args.H, args.K, device="cuda", dtype=args.dtype, requires_grad=True)
        v = torch.randn(args.B, args.T, args.H, args.V, device="cuda", dtype=args.dtype, requires_grad=True)
        k = k.repeat_interleave(args.HQ // args.H, dim=2)
        v = v.repeat_interleave(args.HQ // args.H, dim=2)
        q_t = q.transpose(1, 2).contiguous()
        k_t = k.transpose(1, 2).contiguous()
        v_t = v.transpose(1, 2).contiguous()
        attn_mask, is_causal = None, True
        if window is not None:
            qi = torch.arange(args.T, device="cuda").view(args.T, 1)
            ki = torch.arange(args.T, device="cuda").view(1, args.T)
            band = (ki <= qi) & (ki > qi - window)
            attn_mask = torch.zeros(args.T, args.T, device="cuda", dtype=args.dtype).masked_fill(~band, float("-inf"))
            is_causal = False
        scale = 0.12 if rope_scale else None
        torch.cuda.synchronize(); t0 = time.perf_counter()
        o = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask, is_causal=is_causal, scale=scale)
        o.float().square().mean().backward()
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1000
        _finish(label, "ok", elapsed_ms=dt)
    except RuntimeError as e:
        _finish(label, "oom" if "out of memory" in str(e).lower() else "error", error=e)


def run_polar(label, args, window):
    _reset()
    try:
        from kernel.polar_triton import polar_attention
        H = args.HQ
        q = torch.randn(args.B, H, args.T, args.K, device="cuda", dtype=args.dtype, requires_grad=True)
        k = torch.randn(args.B, H, args.T, args.K, device="cuda", dtype=args.dtype, requires_grad=True)
        v = torch.randn(args.B, H, args.T, args.V, device="cuda", dtype=args.dtype, requires_grad=True)
        n_keys = torch.arange(1, args.T + 1, device="cuda", dtype=torch.float32)
        params = dict(
            v_null=(torch.randn(H, args.V, device="cuda", dtype=args.dtype) * 0.1).requires_grad_(True),
            null_base=torch.full((H,), 2.0, device="cuda", dtype=args.dtype, requires_grad=True),
            null_slope_raw=torch.full((H,), 0.5, device="cuda", dtype=args.dtype, requires_grad=True),
            len_gain_raw=torch.full((H,), -1.0, device="cuda", dtype=args.dtype, requires_grad=True),
            mag_beta_raw=torch.full((H,), -1.5, device="cuda", dtype=args.dtype, requires_grad=True),
        )
        torch.cuda.synchronize(); t0 = time.perf_counter()
        c, mag = polar_attention(q, k, v, n_keys, window=window, **params)
        (c.float().square().mean() + mag.float().square().mean()).backward()
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1000
        _finish(label, "ok", elapsed_ms=dt)
    except RuntimeError as e:
        _finish(label, "oom" if "out of memory" in str(e).lower() else "error", error=e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--HQ", type=int, default=8)
    p.add_argument("--H", type=int, default=2)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--V", type=int, default=128)
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    args.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    local_wall, upstream_wall = _load_wall_impls()
    print(f"local_wall_file={inspect.getfile(local_wall)}")
    if upstream_wall is not None:
        print(f"upstream_wall_file={inspect.getfile(upstream_wall)}")
    print(f"shape=B{args.B},T{args.T},HQ{args.HQ},H{args.H},K{args.K},V{args.V},dtype={args.dtype},window={args.window}")

    for window in (None, args.window):
        suffix = "full" if window is None else f"win{window}"
        run_sdpa(f"sdpa_nope_{suffix}", args, window, rope_scale=False)
        run_sdpa(f"sdpa_rope_{suffix}", args, window, rope_scale=True)
        run_polar(f"polar_triton_{suffix}", args, window)
        run_wall(f"wall_local_{suffix}", local_wall, args, window)
        if upstream_wall is not None:
            run_wall(f"wall_upstream_{suffix}", upstream_wall, args, window)


if __name__ == "__main__":
    main()
