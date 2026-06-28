from __future__ import annotations

import argparse
import time
import numpy as np
from tinygrad import Tensor, TinyJit, Variable, dtypes

from edge.config import resolve_device, resolve_dtype
from edge.kernels import causal_conv1d_decode_step
from edge.model import EdgeAtma
from model.config import AtmaConfig


def _make_tensors(channels: int, kernel_size: int, device: str, dtype):
    x = Tensor(np.random.randn(channels).astype(np.float32), device=device).cast(dtype).realize()
    prev = Tensor(np.random.randn(channels, kernel_size - 1).astype(np.float32), device=device).cast(dtype).realize()
    weight = Tensor(np.random.randn(channels, 1, kernel_size).astype(np.float32), device=device).cast(dtype).realize()
    return x, prev, weight


def conv_step(x: Tensor, prev: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    y, state = causal_conv1d_decode_step(x, prev, weight)
    return y.realize(), state.realize()


def bench(fn, x: Tensor, prev: Tensor, weight: Tensor, iters: int) -> dict:
    state = prev
    y = None
    t0 = time.perf_counter()
    for _ in range(iters):
        y, state = fn(x, state, weight)
    # Synchronize and make sure the chain is not optimized away.
    checksum = float(y.numpy().sum() + state.numpy().sum())
    t1 = time.perf_counter()
    return {
        "seconds": t1 - t0,
        "steps_s": iters / (t1 - t0),
        "checksum": checksum,
    }


def model_config(args) -> AtmaConfig:
    return AtmaConfig(
        vocab_size=args.vocab_size,
        num_hidden_layers=args.layers,
        hidden_size=args.hidden_size,
        head_dim=args.head_dim,
        attn_kernel_size=args.attn_kernel_size,
        conv_kernel_size=args.conv_kernel_size,
        attn_window=args.attn_window,
        mem_enabled=args.memory,
    )


def bench_static_model_decode(args, device: str, dtype) -> dict:
    max_context = args.model_iters + 2
    model = EdgeAtma(model_config(args)).to(device=device, dtype=dtype)
    state = model.new_static_state(max_context=max_context)
    pos = Variable("jit_model_pos", 0, max_context - 1)
    token = Tensor([[1]], device=device, dtype=dtypes.int32).realize()

    def decode_step(tok: Tensor, bound_pos):
        return model.decode_static(tok, bound_pos, state).realize()

    jit_step = TinyJit(decode_step)
    # First call executes normally; second call captures/compiles the replay graph.
    t_first0 = time.perf_counter()
    jit_step(token, pos.bind(0))
    t_first1 = time.perf_counter()
    t_capture0 = time.perf_counter()
    jit_step(token, pos.bind(1))
    t_capture1 = time.perf_counter()

    t0 = time.perf_counter()
    out = None
    for i in range(2, args.model_iters + 2):
        out = jit_step(token, pos.bind(i))
    checksum = float(out.numpy().sum())
    t1 = time.perf_counter()
    return {
        "first_call_s": t_first1 - t_first0,
        "capture_s": t_capture1 - t_capture0,
        "seconds": t1 - t0,
        "steps_s": args.model_iters / (t1 - t0),
        "checksum": checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyJit microbenchmarks for edge custom kernels")
    parser.add_argument("--device", default="cl", help="cpu, cl/opencl, webgpu, ...")
    parser.add_argument("--dtype", default="fp16", choices=["auto", "fp16", "float16", "bf16", "bfloat16", "fp32", "float32"])
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--model-iters", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--attn-kernel-size", type=int, default=3)
    parser.add_argument("--conv-kernel-size", type=int, default=3)
    parser.add_argument("--attn-window", type=int, default=16)
    parser.add_argument("--memory", action="store_true", help="enable Titans memory in the static decode benchmark")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    x, prev, weight = _make_tensors(args.channels, args.kernel_size, device, dtype)

    for _ in range(args.warmup):
        conv_step(x, prev, weight)
    eager = bench(conv_step, x, prev, weight, args.iters)

    jit_step = TinyJit(conv_step)
    for _ in range(args.warmup):
        jit_step(x, prev, weight)
    jitted = bench(jit_step, x, prev, weight, args.iters)

    print(
        f"[edge jit bench] device={device} dtype={str(dtype).replace('dtypes.', '')} "
        f"channels={args.channels} kernel={args.kernel_size} iters={args.iters}"
    )
    print(f"[edge jit bench] eager_custom_kernel steps_s={eager['steps_s']:.2f} seconds={eager['seconds']:.4f}")
    print(f"[edge jit bench] tinyjit_custom_kernel steps_s={jitted['steps_s']:.2f} seconds={jitted['seconds']:.4f}")
    print(f"[edge jit bench] speedup={jitted['steps_s']/eager['steps_s']:.2f}x checksum={jitted['checksum']:.6f}")

    if not args.skip_model:
        model_jit = bench_static_model_decode(args, device, dtype)
        print(
            f"[edge jit bench] tinyjit_static_model_decode steps_s={model_jit['steps_s']:.2f} "
            f"seconds={model_jit['seconds']:.4f} layers={args.layers} hidden={args.hidden_size} "
            f"max_context={args.model_iters + 2} memory={args.memory} checksum={model_jit['checksum']:.6f}"
        )
        print(
            f"[edge jit bench] tinyjit_static_model_compile first_call_s={model_jit['first_call_s']:.4f} "
            f"capture_s={model_jit['capture_s']:.4f}"
        )


if __name__ == "__main__":
    main()
