"""Batch-throughput scaling benchmark for the Atma polar-attention model.

Sweeps batch size 1 -> 512 to find the peak inference throughput, exercising the
FlashAttention-style polar Triton kernels (kernel/polar_triton.py).

Two paths:
  * Default: the full vLLM-style engine (needs `transformers`) — the real
    end-to-end serving benchmark (prefill + paged-KV polar decode + Titans memory)
    across batch sizes, reporting prefill / decode / overall tok/s.
  * `--direct`: a direct polar-model forward (prefill) throughput sweep (no
    `transformers` needed): embed -> polar/conv blocks -> norm. The LM head is
    excluded because materializing logits for all positions at bs=512 would OOM
    (B*T*vocab); in real decode the head is applied only to the last token.

Both paths load the trained checkpoint if found under ../checkpoints (same search
as inference/generate.py); otherwise they run with random weights — fine for a
throughput benchmark.

Run:  python bench_inference.py             # full serving engine (default)
      python bench_inference.py --direct    # direct polar forward sweep
"""
import sys
import time
import torch

BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256, 512]
SEQ_LEN = 512          # prompt length per sequence for the direct sweep
_WARMUP = 3            # passes to let the Triton kernels compile before timing


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _find_weights():
    """Locate a checkpoint with inference/generate.py's search (explicit dirs,
    ../checkpoints, ./checkpoints). Returns (weights_path | None, AtmaConfig built
    from the checkpoint's config.json when present, else defaults)."""
    import json
    from dataclasses import fields as dc_fields
    from inference.generate import find_checkpoint
    from model.config import AtmaConfig

    wpath, cfgpath, _tok, searched = find_checkpoint()
    hf = AtmaConfig()
    if cfgpath:
        names = {f.name for f in dc_fields(AtmaConfig)} - {"dtype"}
        d = {k: v for k, v in json.load(open(cfgpath)).items() if k in names}
        hf = AtmaConfig(**d)
    return wpath, hf, searched


@torch.no_grad()
def _forward_body(model, ids):
    """embed -> blocks (polar attn / LFM2 conv) -> norm. Skips the LM head."""
    x = model.embed(ids)
    for block in model.blocks:
        x = block(x)
    return model.norm(x)


def bench_direct():
    from inference.generate import load_model, HAS_TRITON
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
    """End-to-end serving benchmark via the vLLM-style engine (polar attention +
    Titans memory, paged decode kernel). Requires `transformers` (tokenizer only).
    Sweeps batch size 1 -> 512."""
    from inference import LLM, SamplingParams
    if LLM is None:
        print("Engine (LLM) unavailable — `transformers` not installed. "
              "Falling back to the direct polar throughput sweep.\n")
        bench_direct()
        return

    wpath, hf, searched = _find_weights()
    if wpath:
        print(f"[checkpoint] {wpath}")
    else:
        print("[checkpoint] none found -> random weights (searched: " + ", ".join(searched) + ")")
    print("Initializing the polar-attention inference engine...")
    # model=wpath loads the weights (the engine falls back to random weights when the
    # path doesn't resolve) and the tokenizer falls back to gpt2 either way.
    llm = LLM(model=wpath or "gpt2", kvcache_block_size=256, hf_config=hf)
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
    if "--direct" in sys.argv:
        bench_direct()
    else:
        bench_engine()       # default (--engine kept as an accepted no-op)


if __name__ == "__main__":
    main()
