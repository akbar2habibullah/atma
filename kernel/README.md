# `kernel/` — FlashAttention-style Triton kernel for Polar Attention

An efficient, fused Triton implementation of [Polar Attention](../docs/POLAR_ATTENTION.md).
It reproduces the validated PyTorch reductions in `model/blocks.py`
(`polar_reduce` materialized / `polar_attention_online` streamed) to floating-point
tolerance, but runs **7–27× faster** and uses **~5× less memory** than those paths
(measured below on an NVIDIA L4).

```python
from kernel.polar_triton import polar_attention, polar_attention_fwd
```

## What it computes

Polar attention factors each query's result into two length-invariant channels from
**one** temperature-sharpened softmax with an EV-corrected null sink:

- **direction** `c` — unit vector, "what was attended to" (count-blind), `(B,H,T,dk)`
- **magnitude** `mag` — bounded "how much", participation ratio gated by null
  confidence, `tanh`-squashed into `[0,1)`, `(B,H,T)`

See [`POLAR_ATTENTION.md`](../docs/POLAR_ATTENTION.md) §2 for the full math. The streaming
formulation keeps four running accumulators — `M` (max), `L` (Σp), `S` (Σp·v), and the
**extra** `Q2` (Σp²) needed for the participation ratio `n_eff = L²/Q2` — rescaling
`Q2` by `α²` on each max update.

## Public API

```python
# Autograd-aware. Drop-in for model.blocks.polar_attention_online (minus k_block).
c, mag = polar_attention(
    q, k, v, n_keys,                       # q,k,v: (B,H,T,dk), KV heads expanded to H
    v_null=..., null_base=..., null_slope_raw=...,
    len_gain_raw=..., mag_beta_raw=...,
    eps=1e-6, is_causal=True, input_precision="ieee",
)

# Forward-only (no autograd), for inference.
c, mag = polar_attention_fwd(q, k, v, n_keys, is_causal=..., **polar_params)
```

- `n_keys`: `(Tq,)` valid-key count per query (`torch.arange(1,T+1)` for self-attention;
  `context_len` for decode).
- `is_causal=True`: triangular self-attention (`Tq == Tk`), with an optimized causal
  loop bound. `is_causal=False`: any layout — scans all keys and masks by `n_keys`
  (correct for decode `Tq=1`, offset prefill `Tq ≠ Tk`, etc.).
- `input_precision`: only affects **fp32** inputs (`"ieee"` exact vs `"tf32"` faster).
  bf16/fp16 always run on tensor cores with fp32 accumulation.

## Design

- **Forward** (`_polar_fwd_kernel`): one program per (query-block, batch·head). Streams
  key blocks maintaining `(M, L, Q2, S)`, folds the null sink, writes `c, mag` and saves
  `(M, L, Q2, s)` for the backward. O(T·block) memory.
- **Backward** is split for robustness:
  - the cheap per-query *preamble* (`gs`, `gL`, `gQ2`, and the `v_null` / `null_base` /
    `null_slope_raw` / `mag_beta_raw` grads) runs in PyTorch — all O(B·H·T·dk), no T²
    materialization;
  - the two O(T²) matmul loops are Triton kernels: `_polar_bwd_dq_kernel`
    (query-parallel → `dq`, plus the real-key part of `len_gain_raw`'s grad) and
    `_polar_bwd_dkdv_kernel` (key-parallel → `dk`, `dv`).
- All softmax/elementwise math is fp32; only `tl.dot` operands use the input dtype
  (bf16/fp16 → tensor cores). GQA is handled outside the kernel (KV heads are
  `repeat_interleave`-expanded to H by the caller, exactly as the PyTorch path does).
- Block sizes / pipelining are chosen per (dtype, `dk`) to fit the L4's ~99 KB of
  shared memory (see `_fwd_config` / `_bwd_config`).

## Numerical parity

Verified against the gradchecked oracle (`polar_attention_online`, itself float64
`gradcheck`ed to ~1e-15 when it was first written):

| | fp32 | bf16 / fp16 |
|---|---|---|
| forward `c`, `mag` | ~3e-7 abs | within dtype tolerance |
| grads (q,k,v + all params) | ~1e-6 rel | <5e-2 rel |

At very long context the kernel matches the *streaming* oracle to ~3e-7; its gap vs the
*materialized* oracle equals the existing online-vs-materialized fp32 summation-order
gap (the two `m_eff` forms are algebraically identical) — i.e. the kernel is no less
accurate than the existing online path.

## Benchmark (NVIDIA L4, bf16, fwd+bwd, dk=128)

| shape | Triton | online (PyTorch) | speedup | peak mem (Triton vs online) |
|---|---|---|---|---|
| B4 H16 T512  | 2.9 ms  | 21.7 ms  | 7.5×  | 196 vs 767 MB |
| B2 H16 T1024 | 3.2 ms  | 40.0 ms  | 12.3× | 213 vs 936 MB |
| B1 H16 T2048 | 4.1 ms  | 76.0 ms  | 18.5× | 213 vs 936 MB |
| B1 H16 T4096 | 13.9 ms | 306.7 ms | 22.1× | 395 vs 1855 MB |
| B1 H16 T8192 | 44.3 ms | 1213.6 ms| 27.4× | 772 vs 3694 MB |

(`python -m kernel.bench_polar` to reproduce.)

## Use from the three packages

- **train** — set `AtmaConfig(attn_kernel="triton")` (or pass `attn_kernel="triton"` to
  `train.model.PolarAttention`). Falls back to the torch path on CPU / when Triton is
  unavailable. Default is `"torch"` to preserve the bit-exact parity tests.
- **model** — re-exported as `model.blocks.polar_attention_triton` /
  `polar_attention_triton_fwd` (guarded by `model.blocks.HAS_TRITON`).
- **inference** — `inference/models/atma.py` uses `polar_attention_fwd` for prefill
  (`is_causal=True` for a fresh sequence; `is_causal=False` + explicit `n_keys` for a
  chunked-prefill continuation) and `polar_attention_decode` for paged decode (reads K/V
  directly from the paged cache via `block_tables`/`context_lens`; supports `window=`;
  CUDA-graph capturable).

## Tests

```bash
python -m kernel.test_polar_kernel    # 103 parity / edge-case checks vs the oracle
python -m kernel.test_integration     # train.model PolarAttention + full Model wiring
python -m kernel.bench_polar          # performance vs online / materialized
```
