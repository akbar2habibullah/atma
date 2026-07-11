"""Roofline model for Atma prefill and decode inference.

The default attainable ceilings are measured on the repository's NVIDIA L40S:
representative BF16 GEMMs and a 1 GiB device-to-device copy. Pass --measure to
re-run those calibration microbenchmarks on the current CUDA device.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    layers: int = 16
    dim: int = 1024
    heads: int = 8
    head_dim: int = 128
    kv_heads: int = 2
    attn_layers: int = 4
    vocab: int = 50304
    dtype_bytes: int = 2

    @property
    def body_matrix_weights(self) -> int:
        # 12 conv blocks * (projections 4D² + MLP 12D²)
        # 4 attention blocks * (QKVO 3.5D² + memory 2D² + MLP 12D²)
        # plus count and memory scalar projections.
        return 262 * self.dim * self.dim + self.attn_layers * 3 * self.heads * self.dim

    @property
    def memory_state_bytes(self) -> int:
        return self.heads * self.head_dim * self.head_dim * 4


@dataclass(frozen=True)
class Hardware:
    bf16_tflops: float = 362.05
    hbm_gbps: float = 864.0
    attainable_tflops: float = 211.2
    attainable_gbps: float = 653.2


def prefill_cost(m: Model, batch: int, prompt: int) -> tuple[float, float]:
    linear = 2 * m.body_matrix_weights
    polar = 2 * m.attn_layers * m.heads * m.head_dim * (prompt + 1)
    memory = 7 * m.attn_layers * m.heads * m.head_dim * m.head_dim
    elementwise = 40 * m.layers * m.dim
    lm_head = 2 * m.vocab * m.dim / prompt
    flops = linear + polar + memory + elementwise + lm_head

    # Compulsory-HBM lower bound. Attention tiles may be reread from L2/shared
    # memory, so algorithmic K/V traffic is intentionally not charged as HBM.
    weight_bytes = ((m.body_matrix_weights + m.vocab * m.dim) * m.dtype_bytes
                    / (batch * prompt))
    activation_bytes = 4 * m.layers * m.dim * m.dtype_bytes
    kv_write_bytes = m.attn_layers * 2 * m.kv_heads * m.head_dim * m.dtype_bytes
    state_bytes = 2 * m.attn_layers * m.memory_state_bytes / prompt
    embedding_bytes = m.dim * m.dtype_bytes
    return flops, weight_bytes + activation_bytes + kv_write_bytes + state_bytes + embedding_bytes


def decode_cost(m: Model, batch: int, context: int) -> tuple[float, float]:
    linear = 2 * m.body_matrix_weights + 2 * m.vocab * m.dim
    polar = 4 * m.attn_layers * m.heads * m.head_dim * context
    memory = 7 * m.attn_layers * m.heads * m.head_dim * m.head_dim
    elementwise = 40 * m.layers * m.dim
    flops = linear + polar + memory + elementwise

    weights = ((m.body_matrix_weights + m.vocab * m.dim) * m.dtype_bytes / batch)
    states = 2 * m.attn_layers * m.memory_state_bytes
    kv = (m.attn_layers * 2 * context * m.kv_heads * m.head_dim * m.dtype_bytes)
    activations = 4 * m.layers * m.dim * m.dtype_bytes
    return flops, weights + states + kv + activations


def measure_l40s() -> tuple[float, float]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("--measure requires CUDA")

    def mm(m, n, k, iters=20):
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
        for _ in range(5):
            torch.mm(a, b)
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            torch.mm(a, b)
        end.record()
        end.synchronize()
        seconds = start.elapsed_time(end) * 1e-3 / iters
        return 2 * m * n * k / seconds / 1e12

    shapes = ((4096, 8192, 1024), (8192, 8192, 1024), (8192, 4096, 1024))
    achieved = sum(mm(*shape) for shape in shapes) / len(shapes)
    torch.cuda.empty_cache()

    n = 1 << 30
    src = torch.empty(n, device="cuda", dtype=torch.uint8)
    dst = torch.empty_like(src)
    src.fill_(1)
    for _ in range(5):
        dst.copy_(src)
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(20):
        dst.copy_(src)
    end.record()
    end.synchronize()
    seconds = start.elapsed_time(end) * 1e-3 / 20
    bandwidth = 2 * n / seconds / 1e9
    return achieved, bandwidth


def show(name, flops, byte_count, hw, observed):
    intensity = flops / byte_count
    theoretical = min(hw.bf16_tflops * 1e12 / flops, hw.hbm_gbps * 1e9 / byte_count)
    attainable = min(hw.attainable_tflops * 1e12 / flops,
                     hw.attainable_gbps * 1e9 / byte_count)
    print(f"{name}:")
    print(f"  cost              {flops / 1e6:9.2f} MFLOP/token, {byte_count / 1e6:7.3f} MB/token")
    print(f"  intensity         {intensity:9.1f} FLOP/byte")
    print(f"  theoretical roof  {theoretical:9,.0f} token/s")
    print(f"  attainable roof   {attainable:9,.0f} token/s")
    if observed:
        useful_tflops = observed * flops / 1e12
        useful_gbps = observed * byte_count / 1e9
        print(f"  observed          {observed:9,.0f} token/s")
        print(f"  efficiency        {100*useful_tflops/hw.bf16_tflops:8.1f}% peak MFU, "
              f"{100*useful_tflops/hw.attainable_tflops:5.1f}% attainable MFU")
        print(f"                    {100*useful_gbps/hw.hbm_gbps:8.1f}% peak MBU, "
              f"{100*useful_gbps/hw.attainable_gbps:5.1f}% attainable MBU")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--decode-batch", type=int, default=512)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--prefill-tok-s", type=float, default=0)
    p.add_argument("--decode-tok-s", type=float, default=0)
    p.add_argument("--measure", action="store_true")
    args = p.parse_args()

    hw = Hardware()
    if args.measure:
        tf, gb = measure_l40s()
        hw = Hardware(attainable_tflops=tf, attainable_gbps=gb)
    print(f"L40S ceilings: {hw.bf16_tflops:.2f} TFLOP/s BF16, {hw.hbm_gbps:.1f} GB/s HBM")
    print(f"Calibrated:     {hw.attainable_tflops:.1f} TFLOP/s "
          f"({100*hw.attainable_tflops/hw.bf16_tflops:.1f}% MFU), "
          f"{hw.attainable_gbps:.1f} GB/s ({100*hw.attainable_gbps/hw.hbm_gbps:.1f}% MBU)")
    model = Model()
    show(f"Prefill B={args.batch_size}, T={args.prompt_len}",
         *prefill_cost(model, args.batch_size, args.prompt_len), hw, args.prefill_tok_s)
    show(f"Decode B={args.decode_batch}, S={args.context_len}",
         *decode_cost(model, args.decode_batch, args.context_len), hw, args.decode_tok_s)


if __name__ == "__main__":
    main()
