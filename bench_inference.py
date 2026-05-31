"""Batch-throughput scaling benchmark for the Atma polar-attention model.

Sweeps batch size 1 -> 512 to find the peak inference throughput, exercising the
FlashAttention-style polar Triton kernel (kernel/polar_triton.py).

Two paths:
  * If the full vLLM-style engine is importable (needs `transformers`), `--engine`
    runs the real end-to-end serving benchmark (prefill + paged-KV decode) across
    batch sizes, reporting prefill / decode / overall tok/s.
  * By default it runs a direct polar-model forward (prefill) throughput sweep
    (no `transformers` needed): embed -> polar/conv blocks -> norm. The LM head is
    excluded because materializing logits for all positions at bs=512 would OOM
    (B*T*vocab); in real decode the head is applied only to the last token.

Loads the trained checkpoint if found under ../checkpoints, otherwise random weights
(fine for a throughput benchmark). See inference/generate.py.

Run:  python bench_inference.py            # direct polar throughput sweep
      python bench_inference.py --engine   # full serving engine (needs transformers)
"""
import sys
import time
import torch

from inference.generate import load_model, HAS_TRITON

BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256, 512]
SEQ_LEN = 512          # prompt length per sequence for the direct sweep
_WARMUP = 3            # passes to let the Triton kernels compile before timing


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def _forward_body(model, ids):
    """embed -> blocks (polar attn / LFM2 conv) -> norm. Skips the LM head."""
    x = model.embed(ids)
    for block in model.blocks:
        x = block(x)
    return model.norm(x)


def bench_direct():
    print("Initializing the Atma polar-attention model (direct throughput sweep)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model, info = load_model(device=device, dtype=dtype)
    cfg = info["config"]
    vocab = cfg["vocab_size"]

    print(f"\nConfiguration:")
    print(f"- checkpoint : {info['path'] if info['loaded'] else 'random init'}")
    print(f"- model      : {cfg['num_hidden_layers']} layers, dim {cfg['hidden_size']}, "
          f"{cfg['num_heads']} heads / {cfg['num_kv_heads']} kv-heads (GQA), head_dim {cfg['head_dim']}")
    print(f"- attention  : polar @ layers {cfg['attn_layers']} "
          f"({'Triton kernel' if (HAS_TRITON and device == 'cuda') else 'polar_reduce'}), LFM2 conv elsewhere")
    print(f"- device     : {device} ({dtype})")

    print(f"\nForward (prefill) throughput scaling — seq_len={SEQ_LEN}, body only (no LM head):")
    print(f"  {'batch':>6} {'tokens':>9} {'fwd ms':>10} {'tok/s':>12}")
    peak = 0.0
    peak_bs = 0
    for bs in BATCH_SIZES:
        try:
            ids = torch.randint(0, vocab, (bs, SEQ_LEN), device=device)
            for _ in range(_WARMUP):
                _forward_body(model, ids)
            _sync(device)
            iters = 10 if bs <= 64 else 4
            t0 = time.perf_counter()
            for _ in range(iters):
                _forward_body(model, ids)
            _sync(device)
            dt = (time.perf_counter() - t0) / iters
            tok_s = bs * SEQ_LEN / dt
            if tok_s > peak:
                peak, peak_bs = tok_s, bs
            print(f"  {bs:>6} {bs * SEQ_LEN:>9} {dt * 1e3:>10.2f} {tok_s:>12,.0f}")
        except torch.cuda.OutOfMemoryError:
            print(f"  {bs:>6}   OOM — stopping sweep")
            torch.cuda.empty_cache()
            break

    print(f"\n  peak throughput: {peak:,.0f} tok/s at batch size {peak_bs}")


def bench_engine():
    """Original-style end-to-end serving benchmark via the vLLM-style engine (now
    polar). Requires `transformers`. Sweeps batch size 1 -> 512."""
    from inference import LLM, SamplingParams
    if LLM is None:
        print("Engine (LLM) unavailable — `transformers` not installed. "
              "Run without --engine for the direct polar throughput sweep.")
        return
    print("Initializing the polar-attention inference engine...")
    llm = LLM(model="gpt2", kvcache_block_size=256)
    use_cuda = torch.cuda.is_available()

    prompt = ("Lorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex "
              "sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis "
              "convallis. Tempus leo eu aenean sed diam urna tempor. ") * 4
    max_tokens = 256

    print(f"\nEnd-to-end serving throughput scaling (prompt ~{len(prompt.split())} words, "
          f"{max_tokens} new tokens):")
    print(f"  {'batch':>6} {'gen tok':>9} {'time s':>8} {'overall':>10} {'prefill':>10} {'decode':>10}")
    peak = 0.0
    peak_bs = 0
    for bs in BATCH_SIZES:
        prompts = [prompt] * bs
        sp = SamplingParams(temperature=1.0, max_tokens=max_tokens, ignore_eos=True)
        for _ in range(2):                      # warmup (let torch.compile / graphs settle)
            llm.generate(prompts, sp, use_tqdm=False)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        if use_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        total = bs * max_tokens
        thr = total / dt
        m = getattr(llm, "last_metrics", {}) or {}
        if thr > peak:
            peak, peak_bs = thr, bs
        print(f"  {bs:>6} {total:>9} {dt:>8.2f} {thr:>10.0f} "
              f"{m.get('prefill_throughput', 0):>10.0f} {m.get('decode_throughput', 0):>10.0f}")

    print(f"\n  peak overall throughput: {peak:,.0f} tok/s at batch size {peak_bs}")


def main():
    if "--engine" in sys.argv:
        bench_engine()
    else:
        bench_direct()


if __name__ == "__main__":
    main()
