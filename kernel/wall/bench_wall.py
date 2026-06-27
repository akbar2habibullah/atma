"""Memory/time benchmark for Wall Attention training backward."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=("local", "upstream", "naive"), default="local")
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--HQ", type=int, default=8)
    p.add_argument("--H", type=int, default=2)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--V", type=int, default=128)
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--window", type=int, default=0, help="0 means full causal attention")
    p.add_argument("--R", type=int, default=0, help="packed random-prefix length")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    return p.parse_args()


def load_kernel(impl: str):
    if impl == "local":
        from kernel.wall import wall_attn
    elif impl == "naive":
        from kernel.wall.reference import wall_attn_reference as wall_attn
    else:
        from wall_attn import wall_attn
    return wall_attn


def make_inputs(args: argparse.Namespace, dtype: torch.dtype):
    device = "cuda"
    T_total = args.T + args.R
    q = torch.randn(args.B, T_total, args.HQ, args.K, device=device, dtype=dtype)
    k = torch.randn(args.B, T_total, args.H, args.K, device=device, dtype=dtype)
    v = torch.randn(args.B, T_total, args.H, args.V, device=device, dtype=dtype)
    g = -(torch.rand(args.B, T_total, args.HQ, args.K, device=device, dtype=dtype) * 0.02 + 0.01)
    if args.R:
        q[:, :args.R].zero_()
        g[:, :args.R].zero_()
    return (
        q.detach().requires_grad_(True),
        k.detach().requires_grad_(True),
        v.detach().requires_grad_(True),
        g.detach().requires_grad_(True),
    )


def run_once(wall_attn, args: argparse.Namespace, dtype: torch.dtype):
    q, k, v, g = make_inputs(args, dtype)
    window = None if args.window == 0 else args.window
    out = wall_attn(q, k, v, g, scale=args.K ** -0.5, window_size=window)
    if args.R:
        out = out[:, args.R:]
    loss = out.float().square().mean()
    loss.backward()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    wall_attn = load_kernel(args.impl)

    try:
        for _ in range(args.warmup):
            run_once(wall_attn, args, dtype)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        times = []
        peak_alloc = 0
        peak_reserved = 0
        for _ in range(args.iters):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_once(wall_attn, args, dtype)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
            peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated())
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        print(
            "impl={impl},B={B},T={T},R={R},HQ={HQ},H={H},K={K},V={V},dtype={dtype},"
            "window={window},status=oom,peak_alloc_gb={alloc:.3f},peak_reserved_gb={reserved:.3f},"
            "error={error}".format(
                impl=args.impl,
                B=args.B,
                T=args.T,
                R=args.R,
                HQ=args.HQ,
                H=args.H,
                K=args.K,
                V=args.V,
                dtype=args.dtype,
                window=None if args.window == 0 else args.window,
                alloc=torch.cuda.max_memory_allocated() / 1024**3,
                reserved=torch.cuda.max_memory_reserved() / 1024**3,
                error=str(e).split("\n")[0],
            )
        )
        return

    print(
        "impl={impl},B={B},T={T},R={R},HQ={HQ},H={H},K={K},V={V},dtype={dtype},"
        "window={window},status=ok,peak_alloc_gb={alloc:.3f},peak_reserved_gb={reserved:.3f},"
        "elapsed_ms={elapsed:.2f}".format(
            impl=args.impl,
            B=args.B,
            T=args.T,
            R=args.R,
            HQ=args.HQ,
            H=args.H,
            K=args.K,
            V=args.V,
            dtype=args.dtype,
            window=None if args.window == 0 else args.window,
            alloc=peak_alloc / 1024**3,
            reserved=peak_reserved / 1024**3,
            elapsed=sum(times) / len(times),
        )
    )


if __name__ == "__main__":
    main()
