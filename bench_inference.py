"""Throughput benchmark for the Atma polar-attention model.

Measures the polar-attention forward (prefill) throughput across batch sizes and
sequence lengths — this is the path that exercises the FlashAttention-style polar
Triton kernel (kernel/polar_triton.py). Also times short autoregressive generation.

Loads the trained checkpoint if found under ../checkpoints, otherwise uses random
weights (fine for a throughput benchmark). See inference/generate.py.

Run:  python bench_inference.py
"""
import time
import torch

from inference.generate import load_model, generate, HAS_TRITON

_WARMUP = 3   # passes to let the Triton kernels compile before timing


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def time_forward(model, B, T, vocab, device, iters=10):
    ids = torch.randint(0, vocab, (B, T), device=device)
    for _ in range(_WARMUP):
        model(ids)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(ids)
    _sync(device)
    return (time.perf_counter() - t0) / iters


def main():
    print("Initializing the Atma polar-attention model for benchmarking...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model, info = load_model(device=device, dtype=dtype)
    cfg = info["config"]
    vocab = cfg["vocab_size"]

    print(f"\nConfiguration:")
    print(f"- checkpoint : {info['path'] if info['loaded'] else 'random init'}")
    print(f"- model      : {cfg['num_hidden_layers']} layers, dim {cfg['hidden_size']}, "
          f"{cfg['num_heads']} heads / {cfg['num_kv_heads']} kv-heads (GQA), head_dim {cfg['head_dim']}")
    print(f"- attention  : polar @ layers {cfg['attn_layers']} ({'Triton kernel' if (HAS_TRITON and device=='cuda') else 'polar_reduce'}), LFM2 conv elsewhere")
    print(f"- device     : {device} ({dtype})")

    print(f"\nForward (prefill) throughput — exercises the polar kernel:")
    print(f"  {'batch':>6} {'seq_len':>8} {'fwd ms':>9} {'tok/s':>12}")
    configs = [(1, 512), (1, 1024), (1, 2048), (1, 4096),
               (4, 512), (8, 1024), (16, 1024)]
    for B, T in configs:
        try:
            dt = time_forward(model, B, T, vocab, device)
            print(f"  {B:>6} {T:>8} {dt*1e3:>9.2f} {B * T / dt:>12.0f}")
        except RuntimeError as e:
            print(f"  {B:>6} {T:>8}  OOM/err: {str(e)[:40]}")

    print(f"\nAutoregressive generation (full-recompute) — end-to-end gen speed:")
    print(f"  {'prompt':>7} {'new':>5} {'total s':>9} {'tok/s':>8}")
    for prompt_len, new in [(64, 64), (256, 64)]:
        ids = torch.randint(0, vocab, (prompt_len,), device=device).tolist()
        generate(model, ids, max_new_tokens=4, device=device)  # warmup
        _sync(device)
        t0 = time.perf_counter()
        generate(model, ids, max_new_tokens=new, temperature=0.0, device=device)
        _sync(device)
        dt = time.perf_counter() - t0
        print(f"  {prompt_len:>7} {new:>5} {dt:>9.2f} {new / dt:>8.1f}")


if __name__ == "__main__":
    main()
