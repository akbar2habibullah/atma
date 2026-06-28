from __future__ import annotations

import argparse
import time
import numpy as np

from edge.config import resolve_device, resolve_dtype
from edge.model import EdgeAtma
from model.config import AtmaConfig


def bench_config(args) -> AtmaConfig:
    return AtmaConfig(
        vocab_size=args.vocab_size,
        num_hidden_layers=args.layers,
        hidden_size=args.hidden_size,
        head_dim=args.head_dim,
        attn_kernel_size=args.attn_kernel_size,
        conv_kernel_size=args.conv_kernel_size,
        attn_window=args.attn_window,
        mem_enabled=not args.no_memory,
        mem_chunk=args.mem_chunk,
    )


def run_once(model: EdgeAtma, prompt_len: int, decode_tokens: int) -> dict:
    prompt = [i % model.config.vocab_size for i in range(prompt_len)]
    state = model.new_state()

    t0 = time.perf_counter()
    logits = model(prompt, state).numpy()
    t1 = time.perf_counter()

    next_id = int(np.argmax(logits[0, -1]))
    t2 = time.perf_counter()
    for _ in range(decode_tokens):
        logits = model([next_id], state).numpy()
        next_id = int(np.argmax(logits[0, -1]))
    t3 = time.perf_counter()

    prefill_s = t1 - t0
    decode_s = t3 - t2
    return {
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "prefill_tok_s": prompt_len / prefill_s if prefill_s > 0 else float("inf"),
        "decode_tok_s": decode_tokens / decode_s if decode_s > 0 else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tinygrad edge inference on a tiny configurable model")
    parser.add_argument("--device", default="cl", help="cpu, cl/opencl, webgpu, ...")
    parser.add_argument("--dtype", default="fp16", choices=["auto", "fp16", "float16", "bf16", "bfloat16", "fp32", "float32"])
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--attn-kernel-size", type=int, default=3)
    parser.add_argument("--conv-kernel-size", type=int, default=3)
    parser.add_argument("--attn-window", type=int, default=16)
    parser.add_argument("--mem-chunk", type=int, default=8)
    parser.add_argument("--no-memory", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model = EdgeAtma(bench_config(args)).to(device=device, dtype=dtype)

    for _ in range(args.warmup):
        run_once(model, args.prompt_len, args.decode_tokens)

    results = [run_once(model, args.prompt_len, args.decode_tokens) for _ in range(args.runs)]
    prefill = np.array([r["prefill_tok_s"] for r in results], dtype=np.float64)
    decode = np.array([r["decode_tok_s"] for r in results], dtype=np.float64)
    print(
        f"[edge bench] device={device} dtype={str(dtype).replace('dtypes.', '')} "
        f"layers={args.layers} hidden={args.hidden_size} prompt={args.prompt_len} decode={args.decode_tokens} "
        f"memory={not args.no_memory}"
    )
    print(f"[edge bench] prefill_tok_s mean={prefill.mean():.2f} min={prefill.min():.2f} max={prefill.max():.2f}")
    print(f"[edge bench] decode_tok_s  mean={decode.mean():.2f} min={decode.min():.2f} max={decode.max():.2f}")


if __name__ == "__main__":
    main()
