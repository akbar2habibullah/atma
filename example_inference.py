"""Polar-Attention inference example.

Loads the trained checkpoint if found under ../checkpoints (config.json + weights.pt);
otherwise initializes the polar-attention model with random weights. Attention runs
through the FlashAttention-style polar Triton kernel on CUDA (falls back to the
materialized polar_reduce on CPU). See inference/generate.py for the implementation.
"""
import torch
from inference.generate import load_model, generate, _get_tokenizer, HAS_TRITON


def main():
    print("Initializing the Atma polar-attention model for inference...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model, info = load_model(device=device, dtype=dtype)
    cfg = info["config"]
    if info["loaded"]:
        print(f"[checkpoint] loaded {info['path']}")
    else:
        print("[checkpoint] none found -> random init (demonstrates execution only)")
    print(f"[config] {cfg['num_hidden_layers']} layers, dim {cfg['hidden_size']}, "
          f"{cfg['num_heads']} heads / {cfg['num_kv_heads']} kv-heads, "
          f"polar attention at layers {cfg['attn_layers']} + LFM2 gated conv elsewhere")
    print(f"[attn] {'Triton polar kernel' if (HAS_TRITON and device == 'cuda') else 'PyTorch polar_reduce'}")

    tok = _get_tokenizer(info.get("tokenizer"))
    text_prompts = [
        "In the heart of artificial intelligence research, a new model called Atma was born. It used",
        "Liquid Foundation Models (LFM2) leverage gated causal convolutions to",
        "The main difference between standard attention and polar attention is that polar attention",
    ]
    # With a tokenizer we use the text prompts; otherwise fall back to raw token-id prompts.
    if tok is not None:
        prompts = [(p, tok.encode(p)) for p in text_prompts]
    else:
        print("[tokenizer] none (pip install tiktoken for text) -> using raw token-id prompts")
        prompts = [(f"ids={ids}", ids) for ids in ([464, 995, 318], [40, 716, 257], [464, 3797, 318, 257])]

    print("\nStarting autoregressive text generation...")
    print("=" * 80)
    for label, ids in prompts:
        out = generate(model, ids, max_new_tokens=50, temperature=0.7, top_k=50, device=device)
        completion = tok.decode(out[len(ids):]) if tok is not None else str(out[len(ids):])
        print(f"\n[PROMPT]    : {label}")
        print(f"[GENERATED] : {completion}")
        print("-" * 80)


if __name__ == "__main__":
    main()
