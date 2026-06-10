# Inference

> **Polar Attention + Titans MAG memory are wired into the paged engine** (`inference/llm.py` → `inference/models/atma.py`), matching [model/reference.py](../model/reference.py) and [train/model.py](../train/model.py): prefill runs the FlashAttention-style Triton polar kernel per sequence (`polar_attention_fwd`), decode runs the **paged polar decode kernel** (`polar_attention_decode` in [kernel/polar_triton.py](../kernel/polar_triton.py)) reading K/V directly from the paged cache via block tables — no gather, fixed launch shape, **CUDA-graph capturable**. The sliding window (`attn_window`) is honored in both phases, and the Titans memory state (per-head `M`, fp32, FLA `[K, V]` layout) lives in per-sequence state tables next to the conv states. On CUDA the memory branch runs **FLA's fused gated-delta kernels** — `chunk_gated_delta_rule` for prefill and `fused_recurrent_gated_delta_rule` for the batched decode step — with `initial_state`/`output_final_state` carrying the per-seq state across prefill chunks and decode steps (`mem_kernel: auto|fla|torch`; the pure-PyTorch path remains the CPU/fallback and the graph-safe escape hatch if FLA misbehaves under CUDA-graph capture). `verify.py` checks train == reference == inference parity per layer and end-to-end (30/30 on CPU); **on the target GPU run `verify.py --cuda`** (routes the same checks through the Triton kernels: polar prefill, paged polar decode incl. the `WINDOW` band, fused causal conv — plain `verify.py` only exercises the CPU fallbacks) **plus `verify_fla.py`**, whose *inference bridge* section validates the FLA state layout, chunked-prefill state carry, and chunk→recurrent decode continuity. The self-contained full-recompute generator [inference/generate.py](../inference/generate.py) remains as a simple cross-check (no window/memory).
>
> **Known limitation — cross-request prefix-cache hits.** A hash-matched prefix from *another* request reuses its K/V blocks correctly, but the new sequence's conv/memory state tables start from zeros (those states were never computed for this sequence), so the first tokens after such a hit can drift. Same-request chunked prefill is exact (state carried in the tables, prefix K/V gathered from the cache). Decoding throughput numbers below predate the Polar port (measured with the softmax path) — re-benchmark.

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

## Decoding performance

Model Size: 369.72M Params (16 num_layers + 1024 hidden_dim)

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

Measured at 256 generated tokens per sequence, prompt length ~320 words.

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

## Further throughput benchmark

We also try larger model size 8937.41M parameters (32 num_layers + 4096 hidden_dim) on H100 to test the performance ceiling of the Atma inference engine.

| Batch Size | Decoding (tok/s) |
|------------|------------------|
| 1          | 91               |
| 4          | 347              |
| 8          | 688              |
| 16         | 1,320            |
| 32         | 2,557            |
| 64         | 4,785            |
| 128        | 8,329            |
| 256        | 13,043           |
| 512        | 16,981           |
| 1024       | OOM              |
| 2048       | OOM              |

> Measured at 256 generated tokens per sequence, prompt length ~320 words. FA2 enabled and kvcache_block_size=256.
