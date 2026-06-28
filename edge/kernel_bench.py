from __future__ import annotations

import argparse
import math
import time

import numpy as np
from tinygrad import Tensor, TinyJit, dtypes

from edge.config import resolve_device, resolve_dtype
from edge.kernels import gdn_prefill, gdn_prefill_chunked, polar_prefill
from edge.model import _gated_delta_sequential, _polar_reduce


def make_inputs(args, device: str, dtype):
    rng = np.random.default_rng(args.seed)
    shape = (args.heads, args.tokens, args.head_dim)
    q = Tensor(rng.standard_normal(shape, dtype=np.float32), device=device).cast(dtype).realize()
    k = Tensor(rng.standard_normal(shape, dtype=np.float32), device=device).cast(dtype).realize()
    v = Tensor(rng.standard_normal(shape, dtype=np.float32), device=device).cast(dtype).realize()
    v_null = Tensor(rng.standard_normal((args.heads, args.head_dim), dtype=np.float32), device=device).cast(dtype).realize()
    null_base = Tensor(rng.standard_normal((args.heads,), dtype=np.float32), device=device).cast(dtype).realize()
    null_slope_raw = Tensor(rng.standard_normal((args.heads,), dtype=np.float32), device=device).cast(dtype).realize()
    len_gain_raw = Tensor(rng.standard_normal((args.heads,), dtype=np.float32), device=device).cast(dtype).realize()
    mag_beta_raw = Tensor(rng.standard_normal((args.heads,), dtype=np.float32), device=device).cast(dtype).realize()
    gamma = Tensor(rng.random((args.heads, args.tokens), dtype=np.float32), device=device).realize()
    beta = Tensor(rng.random((args.heads, args.tokens), dtype=np.float32), device=device).realize()
    state = Tensor(rng.standard_normal((args.heads, args.head_dim, args.head_dim), dtype=np.float32), device=device).realize()
    return {
        "q": q,
        "k": k,
        "v": v,
        "v_null": v_null,
        "null_base": null_base,
        "null_slope_raw": null_slope_raw,
        "len_gain_raw": len_gain_raw,
        "mag_beta_raw": mag_beta_raw,
        "gamma": gamma,
        "beta": beta,
        "state": state,
    }


def flash_polar(args, inputs):
    return polar_prefill(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        window_size=args.window,
        v_null=inputs["v_null"],
        null_base=inputs["null_base"],
        null_slope_raw=inputs["null_slope_raw"],
        len_gain_raw=inputs["len_gain_raw"],
        mag_beta_raw=inputs["mag_beta_raw"],
    )


def eager_polar(args, inputs):
    q = inputs["q"].reshape(1, args.heads, args.tokens, args.head_dim)
    k = inputs["k"].reshape(1, args.heads, args.tokens, args.head_dim)
    v = inputs["v"].reshape(1, args.heads, args.tokens, args.head_dim)
    key_idx = Tensor.arange(args.tokens, device=inputs["q"].device).reshape(1, -1)
    query_next = Tensor.arange(1, args.tokens + 1, device=inputs["q"].device, dtype=dtypes.float32)
    invalid = key_idx >= query_next.reshape(-1, 1)
    if args.window < args.tokens:
        invalid = invalid | (key_idx < (query_next.reshape(-1, 1) - args.window))
    sigma = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(args.head_dim)
    sigma = sigma.masked_fill(invalid.reshape(1, 1, args.tokens, args.tokens), float("-inf"))
    return _polar_reduce(
        sigma,
        v,
        query_next.minimum(float(args.window)),
        v_null=inputs["v_null"],
        null_base=inputs["null_base"],
        null_slope_raw=inputs["null_slope_raw"],
        len_gain_raw=inputs["len_gain_raw"],
        mag_beta_raw=inputs["mag_beta_raw"],
    )


def flash_gdn(args, inputs):
    if args.gdn_variant == "chunked":
        return gdn_prefill_chunked(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["gamma"],
            inputs["beta"],
            inputs["state"],
            chunk_size=args.chunk_size,
        )
    return gdn_prefill(inputs["q"], inputs["k"], inputs["v"], inputs["gamma"], inputs["beta"], inputs["state"])


def eager_gdn(args, inputs):
    return _gated_delta_sequential(
        inputs["q"].reshape(1, args.heads, args.tokens, args.head_dim).float(),
        inputs["k"].reshape(1, args.heads, args.tokens, args.head_dim).float(),
        inputs["v"].reshape(1, args.heads, args.tokens, args.head_dim).float(),
        inputs["gamma"].reshape(1, args.heads, args.tokens).float(),
        inputs["beta"].reshape(1, args.heads, args.tokens).float(),
        inputs["state"].reshape(1, args.heads, args.head_dim, args.head_dim).float(),
    )


def checksum(outputs) -> float:
    total = 0.0
    for out in outputs:
        total += float(out.numpy().sum())
    return total


def run_eager(label: str, fn, args, inputs) -> dict:
    t0 = time.perf_counter()
    outputs = fn(args, inputs)
    Tensor.realize(*outputs)
    value = checksum(outputs)
    t1 = time.perf_counter()
    return {"label": label, "seconds": t1 - t0, "checksum": value}


def run_jit(label: str, fn, args, inputs) -> dict:
    def wrapped():
        outs = fn(args, inputs)
        Tensor.realize(*outs)
        return outs

    jit_fn = TinyJit(wrapped)
    t_first0 = time.perf_counter()
    outs = jit_fn()
    checksum(outs)
    t_first1 = time.perf_counter()
    t_capture0 = time.perf_counter()
    outs = jit_fn()
    checksum(outs)
    t_capture1 = time.perf_counter()

    t0 = time.perf_counter()
    last = outs
    for _ in range(args.iters):
        last = jit_fn()
    value = checksum(last)
    t1 = time.perf_counter()
    return {
        "label": label,
        "first_call_s": t_first1 - t_first0,
        "capture_s": t_capture1 - t_capture0,
        "replay_s": t1 - t0,
        "runs_s": args.iters / (t1 - t0),
        "tokens_s": (args.iters * args.tokens) / (t1 - t0),
        "checksum": value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone flash polar/GDN kernel profiler")
    parser.add_argument("--device", default="cl")
    parser.add_argument("--dtype", default="fp16", choices=["auto", "fp16", "float16", "bf16", "bfloat16", "fp32", "float32"])
    parser.add_argument("--kernel", default="both", choices=["polar", "gdn", "both"])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--gdn-variant", default="naive", choices=["naive", "chunked"])
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--skip-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    args.window = min(args.window, args.tokens)
    inputs = make_inputs(args, device, dtype)
    print(
        f"[edge kernel bench] device={device} dtype={str(dtype).replace('dtypes.', '')} "
        f"kernel={args.kernel} heads={args.heads} tokens={args.tokens} head_dim={args.head_dim} "
        f"window={args.window} iters={args.iters} gdn_variant={args.gdn_variant} chunk_size={args.chunk_size}"
    )

    jobs = []
    if args.kernel in ("polar", "both"):
        jobs.append(("polar", flash_polar, eager_polar))
    if args.kernel in ("gdn", "both"):
        label = "gdn_chunked" if args.gdn_variant == "chunked" else "gdn"
        jobs.append((label, flash_gdn, eager_gdn))

    for label, flash_fn, eager_fn in jobs:
        if not args.skip_eager:
            eager = run_eager(f"eager_{label}", eager_fn, args, inputs)
            print(f"[edge kernel bench] {eager['label']} seconds={eager['seconds']:.4f} checksum={eager['checksum']:.6f}")
        jitted = run_jit(f"flash_{label}", flash_fn, args, inputs)
        print(
            f"[edge kernel bench] {jitted['label']} runs_s={jitted['runs_s']:.2f} "
            f"tokens_s={jitted['tokens_s']:.2f} replay_s={jitted['replay_s']:.4f} "
            f"first_call_s={jitted['first_call_s']:.4f} capture_s={jitted['capture_s']:.4f} "
            f"checksum={jitted['checksum']:.6f}"
        )


if __name__ == "__main__":
    main()
