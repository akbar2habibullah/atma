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
architectural price, dominated by the Titans matrix state (fp32 H·dk² ≈ 512 KB per
sequence per memory layer, read + written every step) plus the polar core's fp32
reduction; both costs scale with batch size, which is why the gap only opens at large
batch (bs=1 is within ~10%).

## L40S roofline, targets, and inference results

Measured 2026-07-11 on the local NVIDIA L40S (BF16, random weights, full engine
prefill including the last-token LM head and sampler). The reproducible calculator is
[scripts/roofline_inference.py](../scripts/roofline_inference.py):

    python -m scripts.roofline_inference --measure \
      --prefill-tok-s 174571 --decode-tok-s 64583

The hardware ceilings used are 362.05 TFLOP/s dense BF16 tensor compute and 864 GB/s
HBM. Representative model-shaped GEMMs measured 208.9–217.4 TFLOP/s and a 1 GiB
device copy measured 653.2 GB/s. The calibrated ceilings are therefore 211.2 TFLOP/s
(58.3% peak MFU) and 653.2 GB/s (75.6% peak MBU).

The model accounts for body matrix FLOPs, causal polar QK/value reductions, the
gated-delta state update/read, the amortized last-token LM head, compulsory weights,
recurrent-state traffic, raw GQA KV traffic, and a conservative activation lower bound.
It intentionally counts attention tile reuse as cache/shared-memory traffic rather than
repeated HBM traffic.

| Phase | Shape | Cost/token | Arithmetic intensity | Peak roof | Calibrated roof | Measured | Limiter |
|---|---:|---:|---:|---:|---:|---:|---|
| Prefill | B=8, T=512 | 558.38 MFLOP, 0.305 MB | 1,832 FLOP/B | 648k tok/s | 378k tok/s | **174.6k tok/s** | compute/dispatch |
| Decode (CUDA graph) | B=512, S=512 | 665.39 MFLOP, 7.697 MB | 86 FLOP/B | 112k tok/s | 84.9k tok/s | **64.6k tok/s** | HBM/state |

The realistic targets were set below the calibrated roofs:

- Dense prefill: at least 150k tok/s at B=8, T=512 (39.7% of attainable MFU) and
  170k tok/s at B=16. Achieved **174.6k** and **179.8k tok/s**, respectively.
- Decode: 35–45k tok/s at B=512, S=512 was the initial target (41–53% of attainable
  MBU). The tuned CUDA-graph path achieves **64.6k tok/s**, or 76% of the calibrated
  bandwidth roof. The eager reference was 28.4k tok/s.

### Dense prefill benchmark

| Batch | Packed tok/s | Dense tok/s | Speedup | Dense time |
|---:|---:|---:|---:|---:|
| 2 | 28,313 | 40,441 | 1.43× | 25.32 ms |
| 4 | 33,743 | 89,854 | 2.66× | 22.79 ms |
| 8 | 35,966 | **174,571** | **4.85×** | 23.46 ms |
| 16 | 37,630 | **179,820** | **4.78×** | 45.56 ms |

Operator profiling at B=8, T=512 explains the gain. The packed path launched 192
depthwise convolutions, 32 polar kernels, and 32 FLA scans and spent about 136 ms in
CPU dispatch. The dense path launched 24, 4, and 4, respectively, reducing CPU dispatch
to about 23 ms. GEMMs are now the dominant CUDA work instead of per-sequence launch gaps.

Numerical comparison on CUDA was exact for dense versus packed logits, every paged K/V
entry, and all 28 convolution/Titans state tables. Regression tests cover the batched
convolution state and the conservative routing predicate.

### Decode profiling and tuning

At B=512, S=512 the original eager step took about 18 ms. Its 511 launches left the
GPU idle between 9.61 ms of actual CUDA work. The four Titans state kernels consumed
3.43 ms, four paged-polar kernels 1.67 ms, and GEMMs 2.29 ms. Capturing the model body
reduced the step to 9.69 ms (**52.9k tok/s**) before kernel changes.

The tuned path makes four measured changes:

- Select a 64-column Titans state tile at B>=256. This raises large-state HBM
  throughput while retaining the 32-column tile's occupancy at small batches.
- Replace gather → arithmetic → concatenate → scatter causal-convolution updates with
  one slot-indexed Triton kernel. All 24 decode convolution updates now take ~0.15 ms.
- Fuse each MLP's squared-ReLU gate into one forward-only kernel with exact BF16
  intermediate rounding.
- Fuse the last-token output soft cap, avoiding four extra passes over the
  B×50,304 logit tensor.

Together these reduce B=512 graph time from 9.69 ms to **7.92 ms**, a further 1.22×
speedup after graph capture and 2.27× over eager. Useful modeled traffic is ~497 GB/s,
76% of the measured 653 GB/s attainable HBM bandwidth. The state and polar kernels are
individually already close to that bandwidth ceiling, so further gains require reducing
state/KV bytes or overlapping independent work rather than another tile sweep.

| Batch | Context | Decode time | Decode tok/s |
|---:|---:|---:|---:|
| 1 | 512 | 1.60 ms | 624 |
| 8 | 512 | 1.90 ms | 4,210 |
| 64 | 512 | 2.55 ms | 25,098 |
| 256 | 512 | 4.85 ms | 52,785 |
| 512 | 128 | 6.78 ms | 75,575 |
| 512 | 512 | **7.93 ms** | **64,583** |
| 512 | 1024 | 9.37 ms | 54,666 |

Graph capture now respects `max_num_seqs`; small deployments no longer allocate and
capture unused buckets through batch 2048. Eager/graph output parity is exact in the
engine runner, and decode kernel regression tests cover the wide Titans tile, fused
convolution state update, MLP activation, and output soft cap.

## Dense batched prefill (implemented)

The paged engine now has a dense prefill fast path. This is a general inference-engine
optimization, not benchmark-only plumbing: serving and evaluation workloads often have batches
of fresh prompts where prefill dominates and decode is short.

The existing paged/chunked path should stay as the fallback. Route to dense prefill only when
the scheduled prefill batch is safe:

```text
all seq.num_cached_tokens == 0
all seq.num_scheduled_tokens == seq.num_tokens
no cross-request prefix-cache reuse
same prompt length for v1
len(seqs) > 1
sum prompt tokens fits max_num_batched_tokens
```

Initial v1 scope:

1. Fresh same-length request batches only; no padding and no prefix-cache hits.
2. Run one dense `[B, T]` prefill forward instead of a Python loop over sequences.
3. Compute and store final conv states and Titans memory states for each sequence.
4. Scatter K/V into the existing paged cache using the existing slot mapping.
5. Enter the current CUDA-graph decode path unchanged.

Later extensions:

- Length buckets with exact pad masks.
- `torch.compile` per `(B, T)` bucket after correctness is stable.
- Optional prefix-cache support only after conv/memory state reuse is made exact.

Correctness requirements:

- Dense prefill logits and final conv/memory states must match the current per-sequence prefill
  path for same-length fresh prompts.
- Padding tokens, once supported, must not contribute labels, attention context, conv state, or
  Titans memory updates.
- Cross-request prefix-cache hits remain disabled in dense mode until the known state-table drift
  limitation is fixed.

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
