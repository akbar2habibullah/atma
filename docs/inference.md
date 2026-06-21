# Inference

> **Polar Attention + Titans MAG memory are wired into the paged engine** (`inference/llm.py` → `inference/models/atma.py`), matching [model/reference.py](../model/reference.py) and [train/model.py](../train/model.py): prefill runs the FlashAttention-style Triton polar kernel per sequence (`polar_attention_fwd`), decode runs the **paged polar decode kernel** (`polar_attention_decode` in [kernel/polar_triton.py](../kernel/polar_triton.py)) reading K/V directly from the paged cache via block tables — no gather, fixed launch shape, **CUDA-graph capturable**. The sliding window (`attn_window`) is honored in both phases, and the Titans memory state (per-head `M`, fp32, FLA `[K, V]` layout) lives in per-sequence state tables next to the conv states. On CUDA the memory branch runs **FLA's fused `chunk_gated_delta_rule`** for prefill (`initial_state`/`output_final_state` carry the per-seq state across chunks; `mem_kernel: auto|fla|torch` selects the prefill backend) and a **custom fused step kernel** ([kernel/gated_delta_triton.py](../kernel/gated_delta_triton.py)) for decode, which updates the slot-indexed state table **in place** — one fp32 state read+write per step instead of gather → kernel → scatter (3× the traffic; the state is the dominant decode cost at large batch: `H·dk² ·4B` ≈ 512 KB/seq/layer). The paged polar decode kernel is **GQA-grouped**: one program per (sequence, KV head) serves all its query heads, so cached K/V is read once per group, not once per head, and the key loop is bounded by the live context length rather than the graph-padded `max_model_len`. **All three implementations — training, reference, inference — pass numerical verification** (2026-06-10, NVIDIA L4): `verify.py` 30/30 on CPU **and** 30/30 with `--cuda`, which routes the same per-layer + end-to-end checks through the Triton kernels (polar prefill, paged polar decode incl. the `WINDOW` band, chunked prefill with state carry, fused causal conv); `verify_fla.py`'s *inference bridge* section confirms the FLA state layout, chunked-prefill state carry, chunk→recurrent decode continuity, and the fused step kernel (all rel_err ≤ 0.005). The self-contained full-recompute generator [inference/generate.py](../inference/generate.py) remains as a simple cross-check (no window/memory).
>
> **Known limitation — cross-request prefix-cache hits.** A hash-matched prefix from *another* request reuses its K/V blocks correctly, but the new sequence's conv/memory state tables start from zeros (those states were never computed for this sequence), so the first tokens after such a hit can drift. Same-request chunked prefill is exact (state carried in the tables, prefix K/V gathered from the cache).

Production-grade inference engine featuring:

- **Paged KV cache** with hash-based prefix caching (xxhash) for memory sharing
- **Chunked prefill** and **preemption** scheduling
- **CUDA graph capture** for decode at multiple batch sizes
- **Centralized per-seq state tables** (conv states + Titans memory) enabling graph-captured decode
- **Tensor parallelism** support (ColumnParallel, RowParallel, QKVParallel, VocabParallel)
- **Triton polar attention kernels** (prefill + paged decode) with a pure-PyTorch CPU fallback
- **Gumbel-max sampling**

## Usage

```python
from inference import LLM, SamplingParams

llm = LLM(model="path/to/weights.pt")
params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Hello, world!"], params)
print(outputs[0]["text"])
```

## Decoding performance — Polar Attention + Titans memory (current architecture)

Model size: 378.22M params (16 num_layers + 1024 hidden_dim, `mem_enabled=True`,
`attn_window=1024`). Measured 2026-06-10 on NVIDIA L4 via `python -m scripts.bench_inference`
(prompt ~128 words, 256 generated tokens per sequence, kvcache_block_size=256,
CUDA graphs on):

| Batch Size | Decode (tok/s) | Overall (tok/s) |
|------------|----------------|-----------------|
| 1          | 212            | 205             |
| 4          | 778            | 721             |
| 8          | 1,549          | 1,367           |
| 16         | 2,812          | 2,135           |
| 32         | 4,748          | 3,360           |
| 64         | 8,694          | 5,086           |
| 128        | 13,377         | 6,748           |
| 256        | 17,109         | 7,552           |
| 512        | **19,270**     | **7,926**       |

Decode is ~1.4× slower than the legacy softmax baseline below at bs=512 — the expected
architectural price, dominated by the Titans matrix state (fp32 `H·dk²` ≈ 512 KB per
sequence per memory layer, read + written every step) plus the polar core's fp32
reduction; both costs scale with batch size, which is why the gap only opens at large
batch (bs=1 is within ~10%). Prefill (~1,550 tok/s) runs per-sequence polar kernels and
the FLA chunked memory scan; batching the per-sequence loop is the remaining headroom.

### Legacy softmax baseline (previous architecture)

`CausalSelfAttention` (FlashAttention) **without** the Titans memory — kept for
reference; this configuration is no longer wired into the model. 369.72M params,
prompt ~320 words, 256 generated tokens:

| Batch Size | NVIDIA L4  (tok/s) | NVIDIA H100 (tok/s) | NVIDIA T4 (tok/s)  |
|------------|--------------------|---------------------|--------------------|
| 1          | 235                | 344                 | 55                 |
| 4          | 906                | 1,432               | 136                |
| 8          | 1,775              | 2,809               | 240                |
| 16         | 3,454              | 5,269               | 386                |
| 32         | 6,180              | 10,112              | 558                |
| 64         | 10,859             | 18,548              | 723                |
| 128        | 17,531             | 33,638              | OOM                |
| 256        | 24,023             | 51,966              | OOM                |
| 512        | 27,806             | 75,749              | OOM                |
| 1024       | 27,912             | 92,941              | OOM                |
| 2048       | 28,268             | **96,972**          | OOM                |

- NVIDIA L4: FA3 enabled and kvcache_block_size=64.
- NVIDIA H100: FA2 enabled and kvcache_block_size=256.
- NVIDIA T4: SDPA only, max_num_batched_tokens=4096, and kvcache_block_size=32.

## Comparison with vLLM (H100)

Using identical model [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) with command:

```
> VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 vllm bench throughput --model LiquidAI/LFM2.5-350M --random-input-len 512 --random-output-len 256 --num-prompts 512

Throughput: 70.77 requests/s, 54350.30 total tokens/s, 18116.77 output tokens/s
Total num prompt tokens:  262144
Total num output tokens:  131072
```