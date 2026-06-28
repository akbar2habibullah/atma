from __future__ import annotations

import argparse
import numpy as np
import sys
from tinygrad import dtypes

from edge.config import resolve_device, resolve_dtype
from edge.model import EdgeAtma
from model.config import AtmaConfig


def tiny_config(mem_enabled: bool = True) -> AtmaConfig:
    return AtmaConfig(
        vocab_size=64,
        num_hidden_layers=4,
        hidden_size=32,
        head_dim=8,
        attn_kernel_size=3,
        conv_kernel_size=3,
        attn_window=4,
        mem_enabled=mem_enabled,
        mem_chunk=4,
    )


def run_probe(device: str, dtype: str, mem_enabled: bool = True) -> dict:
    tg_device = resolve_device(device)
    tg_dtype = resolve_dtype(dtype, tg_device)
    model = EdgeAtma(tiny_config(mem_enabled=mem_enabled)).to(device=tg_device, dtype=tg_dtype)
    logits = model([3, 1, 4, 1, 5], model.new_state()).numpy()
    return {
        "device": tg_device,
        "dtype": str(tg_dtype).replace("dtypes.", ""),
        "shape": tuple(logits.shape),
        "finite": bool(np.isfinite(logits).all()),
        "min": float(logits.min()),
        "max": float(logits.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the tinygrad edge backend with a tiny random model")
    parser.add_argument("--device", default="auto", help="auto, cpu, cl/opencl, webgpu, cuda, ...")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp16", "float16", "bf16", "bfloat16", "fp32", "float32"])
    parser.add_argument("--no-memory", action="store_true", help="disable the Titans memory branch")
    args = parser.parse_args()

    try:
        info = run_probe(args.device, args.dtype, mem_enabled=not args.no_memory)
    except Exception as exc:
        print(f"[edge probe] failed device={args.device} dtype={args.dtype}: {type(exc).__name__}: {exc}")
        sys.exit(1)
    print(
        f"[edge probe] device={info['device']} dtype={info['dtype']} shape={info['shape']} "
        f"finite={info['finite']} min={info['min']:.6f} max={info['max']:.6f}"
    )


if __name__ == "__main__":
    main()
