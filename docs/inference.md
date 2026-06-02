# Inference

> **Polar Attention is not yet wired into the paged inference engine.** The production engine (`inference/llm.py` → `inference/models/atma.py` → `inference/layers/attention.py`) still runs the **legacy softmax causal attention** (FlashAttention / SDPA paged path), so its outputs match a trained checkpoint only for softmax models. For **Polar checkpoints**, use the self-contained generator [inference/generate.py](../inference/generate.py), which runs the FlashAttention-style Triton polar kernel (full-recompute). A paged polar decode kernel already exists (`_polar_decode_kernel` in [kernel/polar_triton.py](../kernel/polar_triton.py)); porting it into the paged engine — so the engine, training, and reference paths are once again numerically equivalent — is the main tracked **future task**. `verify.py`'s attention/block/model *inference* checks are expected to fail until then.

Production-grade inference engine featuring:

- **Paged KV cache** with hash-based prefix caching (xxhash) for memory sharing
- **Chunked prefill** and **preemption** scheduling
- **CUDA graph capture** for decode at multiple batch sizes
- **Centralized conv state tables** enabling graph-captured decode
- **Tensor parallelism** support (ColumnParallel, RowParallel, QKVParallel, VocabParallel)
- **Flash Attention 3/2** with Triton KV cache kernel and SDPA fallback
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
