# Inference kernels

This document is the source of truth for ATMA's inference kernel routes, L40S measurements, and
optimization decisions. Model equations remain in [POLAR_ATTENTION.md](POLAR_ATTENTION.md) and
[TITANS_MEMORY.md](TITANS_MEMORY.md); engine behavior and usage are in
[inference.md](inference.md).

## Current routes

The canonical model has 16 layers, hidden size 1024, 8 query heads, 2 KV heads, head dimension
128, four Polar Attention layers, a default 1024-token window, and Titans MAG memory.

| Phase and workload | Route | Fallback boundary |
|---|---|---|
| Equal-length fresh prefill | Dense batched convolution, Polar, and Titans | Requires complete prompts and `B > 1` |
| Heterogeneous fresh prefill | Packed Canon/LFM convolution and tile-mapped Polar | Requires complete CUDA BF16/FP16 prompts and `B > 1` |
| Chunked or prefix prefill | Per-sequence convolution, Polar, and Titans | Oracle route; supports paged prefixes and state carry |
| Decode | CUDA graph, paged Polar, in-place state kernels | Eager/PyTorch implementations remain available |

Routing is conservative. Equal inputs continue to use the dense route; grouped kernels never
handle cached prefixes or continuation chunks. The paged cache and recurrent state-table layouts
are shared by every route.

## Grouped heterogeneous prefill

Packed token storage removes padding but does not by itself balance work: the old path still
launched Canon, Polar, and Titans separately for every request. Fresh heterogeneous batches now
prepare two small maps once in `ModelRunner`:

- query tiles map to packed sequence starts, local query starts, and sequence lengths;
- tokens map to sequence starts, ends, and state-table slots.

### Packed Polar

`polar_attention_packed_fwd` schedules one program per `(query tile, head)`. Tiles from all
requests share a launch, so short requests do not leave SMs idle while a long request finishes.
Each tile scans only through its final causal query and applies the existing window band.

The kernel preserves Polar's exact online statistics: direction accumulator, normalization,
participation ratio, null sink, magnitude compression, causality, windowing, and per-head
parameters. It stores results in packed token order. GQA-expanded K/V is prepared once for the
launch.

### Packed convolution

`packed_causal_conv1d` covers both Canon's Q/K/V filters and LFM convolution. It reads the token
boundary map before every causal tap, so values never cross request boundaries. The program that
contains a sequence's final token also writes its last `kernel_size - 1` inputs to the existing
slot-indexed state table; short sequences are correctly left-padded with zeros.

This reduces each convolution site from one launch per sequence to one launch per packed batch.
Titans prefill is still per sequence because its FLA call must emit an independent final matrix
state. No Polar/convolution/Titans megakernel is used.

## Other inference fusions

The decode path uses dedicated Triton kernels for slot-indexed causal-convolution updates and
Titans gated-delta steps. Small forward-only kernels fuse the MLP squared-ReLU gate and final
logit softcap while preserving BF16 intermediate rounding. Paged Polar decode groups query heads
that share a KV head so each cached K/V value is loaded once per GQA group.

A projection-plus-per-head-RMSNorm Triton prototype is retained in
`kernel/inference_ops_triton.py` for controlled benchmarking, but is not called by the model.
It lost to cuBLAS at the important M=4096 and M=8192 prefill shapes. Lazy `norm2 -> MLP.fc`
likewise lost across M=1..8192 and introduced extra BF16 folding error. Broader CODA-style GEMM
rewriting is therefore not justified by the measured profile.

## L40S measurements

Environment captured 2026-07-12 UTC:

| Field | Value |
|---|---|
| Base commit | `06b2d8599674620e86d584f6248980110d2e1e82` |
| GPU | NVIDIA L40S, compute capability 8.9, 46,068 MiB |
| Driver / CUDA | 595.71.05 / driver 13.2 / PyTorch build 13.0 |
| Python / PyTorch / Triton | 3.12.13 / 2.13.0+cu130 / 3.7.1 |
| Model benchmark | canonical shape, deterministic zero weights, BF16, TP=1, eager prefill |

The model-shaped ceilings measured 208.6 TFLOP/s BF16 and 653.2 GB/s. Dense prefill is
compute/dispatch limited; large-batch decode is state/HBM limited.

| Phase | Shape | Calibrated roof | Measured | Efficiency and limiter |
|---|---:|---:|---:|---|
| Dense prefill | B=8, T=512 | 373.6k tok/s | 174.6k tok/s | 46.7% attainable compute |
| Decode | B=512, S=512 | 84.9k tok/s | 64.6k tok/s | 76.1% attainable HBM |

### Grouped Polar component

Measured with 20 warmups and 100 iterations:

| Distribution | Lengths | Oracle p50 | Grouped p50 | Speedup | Max error |
|---|---|---:|---:|---:|---:|
| Short-heavy | 32, 48, 64, 64, 96, 128, 128, 256 | 2.1443 ms | 0.1300 ms | 16.49x | 0 |
| Mixed | 64, 96, 128, 256, 512, 768, 1024, 1536 | 2.1801 ms | 0.1475 ms | 14.78x | 0 |
| Long-tail | 64, 64, 128, 256, 512, 1024, 2048, 4096 | 2.1258 ms | 0.4751 ms | 4.47x | 0 |

Polar launches fall from eight to one. A complete attention layer including Canon and Titans
improved 1.61x, 1.61x, and 1.58x on these workloads.

### Full 16-layer heterogeneous prefill

Each distribution ran in a fresh process with two warmups and ten measurements so FLA shape
caches did not contaminate peak-memory comparisons.

| Distribution | Oracle p50/p95 | Grouped p50/p95 | Throughput change | Speedup | Peak memory change |
|---|---:|---:|---:|---:|---:|
| Short-heavy | 113.05 / 114.95 ms | 67.95 / 70.94 ms | 7,218 -> 12,009 tok/s | 1.66x | 776.5 -> 776.5 MiB |
| Mixed | 111.47 / 112.71 ms | 67.97 / 69.44 ms | 39,328 -> 64,496 tok/s | 1.64x | 901.2 -> 901.3 MiB |
| Long-tail | 111.13 / 113.63 ms | 65.57 / 67.71 ms | 73,716 -> 124,941 tok/s | 1.69x | 1,039.8 -> 1,039.9 MiB |

Mixed and long-tail exceed the 20% end-to-end shipping gate. Temporary memory growth is below
0.1%. Homogeneous production prefill is unchanged and remains on its faster dense route.

## Reproducing benchmarks

```bash
# A1 prototype and grouped Polar microbenchmarks
python -m scripts.bench_kernel_efficiency --warmup 20 --iterations 100

# Full model; run one distribution per process
python -m scripts.bench_kernel_efficiency \
  --full-model short-heavy --warmup 2 --iterations 10
python -m scripts.bench_kernel_efficiency \
  --full-model mixed --warmup 2 --iterations 10
python -m scripts.bench_kernel_efficiency \
  --full-model long-tail --warmup 2 --iterations 10

# Roofline and serving sweeps
python -m scripts.roofline_inference --measure \
  --prefill-tok-s 174571 --decode-tok-s 64583
python -m scripts.bench_inference
```

The benchmark reports checkpoint/weight status, dtype, TP size, warmups, iterations, p50/p95,
throughput, and peak allocation. Compile and autotune time is excluded from steady state.

## Verification

The completed L40S run passed:

- 20 targeted baseline, dense/grouped-prefill, and decode-kernel tests;
- 30/30 reference checks on CPU and 30/30 on CUDA;
- FLA layout, chunked state carry, chunk-to-decode continuity, and fused step checks;
- exact packed Polar comparisons for randomized non-power-of-two lengths and window variants;
- packed-convolution boundary sentinels and final-state comparisons;
- grouped hidden output, paged K/V, every Canon state, and every Titans state comparison.

Commands:

```bash
python -m pytest tests/test_baseline_inference.py tests/test_dense_prefill.py \
  tests/test_decode_kernels.py -q
python -m tests.verify
python -m tests.verify --cuda
python -m tests.verify_fla
```

`tests/test_edge_kernels.py` additionally requires the optional `tinygrad` dependency, which was
not installed during the L40S run.

## 9.2B L40S stress test

The serving model was also scaled to `hidden_size=4096`, `num_hidden_layers=32`, and
`head_dim=128`. This produces 32 query heads, 8 KV heads, 8 Polar/Titans layers, and exactly
9,209,125,504 parameters. BF16 weights occupy 17.15 GiB.

The stress harness is [scripts/stress_inference.py](../scripts/stress_inference.py). It runs one
workload per process, uses deterministic zero weights, allocates only the pages and sequence-state
rows required by that workload, and includes the last-token LM head. Prefill uses the production
dense or grouped route. Decode captures the model body in one exact-batch CUDA graph and runs the
LM head outside the graph, matching `ModelRunner`; scheduler and sampler time is excluded.

Zero values do not change GEMM/kernel shapes or memory traffic, but these results are systems
stress measurements rather than checkpoint-quality measurements. Peak memory includes weights,
live K/V, recurrent states, workspaces, outputs, and the single graph pool. It does not reserve
unused serving cache capacity or retain all production graph buckets simultaneously, so the OOM
boundary is an isolated-run ceiling rather than a recommended deployment limit.

### Prefill

All measurements used one warmup and five iterations.

| Requests | Shape | Route | p50 | p95 | Throughput | Peak allocated |
|---|---|---|---:|---:|---:|---:|
| Homogeneous | 4 x 512 | Dense | 170.90 ms | 171.19 ms | **11,984 tok/s** | 17.68 GiB |
| Homogeneous | 8 x 512 | Dense | 342.80 ms | 343.09 ms | 11,949 tok/s | 18.19 GiB |
| Homogeneous | 16 x 512 | Dense | 711.85 ms | 712.10 ms | 11,508 tok/s | 19.21 GiB |
| Heterogeneous short-heavy | `32,48,64,64,96,128,128,256` | Grouped | 138.03 ms | 140.58 ms | 5,912 tok/s | 17.43 GiB |
| Heterogeneous mixed | `64,96,128,256,512,768,1024,1536` | Grouped | 342.95 ms | 343.10 ms | **12,783 tok/s** | 17.97 GiB |
| Heterogeneous long-tail | `64,64,128,256,512,1024,2048,4096` | Grouped | 668.64 ms | 668.73 ms | 12,252 tok/s | 18.57 GiB |

Peak observed prefill throughput was 11,984 tok/s for homogeneous prompts and 12,783 tok/s for
heterogeneous prompts. The mixed batch is faster per token because it supplies enough work to
saturate the large model without the long-tail batch's additional windowed attention tiles.

### Decode

Decode used context 512 for homogeneous requests. Heterogeneous requests repeat the mixed pattern
`64,96,128,256,512,768,1024,1536`, whose mean context is 548 and maximum is 1536. Each measurement
used two warmups and ten iterations.

| Requests | Batch | Context | p50 | p95 | Throughput | Peak allocated | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Homogeneous | 128 | 512 | 43.41 ms | 43.48 ms | 2,948 tok/s | 21.28 GiB | pass |
| Homogeneous | 256 | 512 | 56.68 ms | 56.78 ms | 4,517 tok/s | 25.39 GiB | pass |
| Homogeneous | 512 | 512 | 85.74 ms | 85.82 ms | 5,971 tok/s | 33.62 GiB | pass |
| Homogeneous | 768 | 512 | 126.54 ms | 126.64 ms | **6,069 tok/s** | 41.81 GiB | pass |
| Homogeneous | 800 | 512 | 134.89 ms | 134.99 ms | 5,931 tok/s | 42.84 GiB | pass |
| Homogeneous | 832 | 512 | - | - | - | 43.80 GiB before failure | OOM |
| Heterogeneous | 704 | 64-1536 | 120.01 ms | 120.11 ms | 5,866 tok/s | 41.85 GiB | pass |
| Heterogeneous | 736 | 64-1536 | 122.76 ms | 122.80 ms | 5,996 tok/s | 42.96 GiB | pass |
| Heterogeneous | 752 | 64-1536 | 124.18 ms | 124.27 ms | **6,056 tok/s** | 43.53 GiB | pass |
| Heterogeneous | 768 | 64-1536 | - | - | - | 43.01 GiB before graph allocation | OOM |

The homogeneous throughput peak is batch 768; batch 800 still fits but is slower. The isolated
capacity boundary lies between batch 800 and 832. Heterogeneous decode peaks and reaches its
capacity boundary earlier because its average context is larger and page rounding reserves more
K/V blocks.

At context 512, each active sequence requires approximately 16 MiB across the eight FP32 Titans
states and another 16 MiB across the eight BF16 K/V caches, before convolution state, graph, and
output workspace. This explains the nearly linear rise from 21.28 GiB at batch 128 to 41.81 GiB
at batch 768.

Reproduction examples:

```bash
python -m scripts.stress_inference \
  --mode prefill --batch 8 --prompt-length 512 --warmup 1 --iterations 5
python -m scripts.stress_inference \
  --mode prefill --lengths 64,96,128,256,512,768,1024,1536 \
  --warmup 1 --iterations 5
python -m scripts.stress_inference \
  --mode decode --batch 768 --context-length 512 --warmup 2 --iterations 10
python -m scripts.stress_inference \
  --mode decode --context-lengths 64,96,128,256,512,768,1024,1536 \
  --repeat 94 --warmup 2 --iterations 10
```

## Illustrative Qwen3.5-9B / vLLM comparison

[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) is a useful similar-scale reference:
its language model also has hidden size 4096, 32 layers, about 9B parameters, and a repeating
3:1 hybrid layout. It is not an architecture-equivalent baseline.

| Property | ATMA stress model | Qwen3.5-9B text model |
|---|---|---|
| Hybrid layout | 24 LFM convolutions + 8 Polar/Titans layers | 24 Gated DeltaNet + 8 gated-attention layers |
| Heads | 32 Q / 8 KV, dimension 128 | GDN: 32 V / 16 QK, dimension 128; attention: 16 Q / 4 KV, dimension 256 |
| FFN intermediate | 16,384 | 12,288 |
| Vocabulary / LM head | 50,304 | 248,320 |
| Parameters used | 9,209,125,504 | Official 9B language checkpoint |
| Weights | Deterministic BF16 zeros | Official BF16 checkpoint |

The much wider Qwen LM head and different allocation of parameters between embeddings, FFNs,
attention, and recurrent state make this a systems-scale comparison, not evidence that one model
architecture is intrinsically faster.

### Environment and protocol

Measured on the same L40S with vLLM 0.25.0, PyTorch 2.11.0+cu130, text-only mode, TP=1,
BF16, prefix caching disabled, chunked prefill enabled, no quantization, and no speculative/MTP
decoding. `gpu_memory_utilization=0.90` and `max_num_batched_tokens=8192` were used. vLLM's
optional FlashInfer sampler was disabled because this machine's system CUDA toolkit did not match
the isolated CUDA 13 environment; both systems therefore used greedy non-FlashInfer sampling.

ATMA results use `scripts/stress_inference.py --include-sampler`: one exact-batch CUDA graph,
last-token LM head, and sampler, without scheduler overhead. Qwen results use the offline vLLM
engine with real scheduling, LM head, and sampler. Prefill is total prompt tokens divided by the
median wall time for one generated token. ATMA decode is direct steady-state step timing; Qwen
decode is estimated by subtracting a separate one-token prefill batch from the wall time of a
multi-token generation batch. Compile, checkpoint load, and graph capture are excluded.

### Prefill

| Workload | Tokens | ATMA | Qwen/vLLM | ATMA / Qwen |
|---|---:|---:|---:|---:|
| Homogeneous B=8, T=512 | 4,096 | 11,947 tok/s | **12,772 tok/s** | 0.94x |
| Short-heavy heterogeneous | 816 | 5,752 tok/s | **8,002 tok/s** | 0.72x |
| Mixed heterogeneous | 4,384 | **12,779 tok/s** | 11,996 tok/s | 1.07x |
| Long-tail heterogeneous | 8,192 | **12,251 tok/s** | 11,721 tok/s | 1.05x |

Qwen/vLLM leads small and homogeneous batches, while ATMA's grouped path is slightly ahead once
the ragged batch contains enough tokens to saturate the L40S. The short-heavy ATMA p95 was noisy
(203 ms versus 142 ms p50), so that row should not be over-interpreted.

### Decode at context 512

| Batch | ATMA steady step | Qwen/vLLM estimated steady generation | ATMA / Qwen |
|---:|---:|---:|---:|
| 128 | 2,939 tok/s | 2,247 tok/s | 1.31x |
| 256 | 4,501 tok/s | 2,921 tok/s | 1.54x |
| 512 | 5,919 tok/s | 2,558 tok/s | 2.31x |

For a heterogeneous batch of 128 requests using the mixed context pattern (64-1536, mean 548),
ATMA measured 2,950 tok/s and Qwen/vLLM measured 2,403 tok/s (1.23x). Qwen's 248k-wide LM head,
24 GDN recurrent layers, and vLLM scheduler are included in its number; ATMA has a 50k-wide head,
eight Titans states, and no scheduler in the timed step. These results explain system behavior
but should not be presented as a controlled kernel or model-quality comparison.

### Reproduction

The comparison runner is [scripts/bench_vllm_qwen.py](../scripts/bench_vllm_qwen.py). The official
checkpoint revision tested was `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (17.98 GiB loaded by
vLLM).

```bash
uv venv ~/vllm-qwen --python 3.12
uv pip install --python ~/vllm-qwen/bin/python 'vllm==0.25.0' --torch-backend=auto
HF_HOME=~/.cache/huggingface ~/vllm-qwen/bin/hf download Qwen/Qwen3.5-9B

# This host also needed FFmpeg shared libraries for TorchCodec.
micromamba create -p ~/ffmpeg-libs -c conda-forge 'ffmpeg>=7,<8' -y
source ~/vllm-qwen/bin/activate
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/ffmpeg-libs/lib:$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0
python -m scripts.bench_vllm_qwen \
  --batch 8 --prompt-length 512 --output-tokens 1 --warmup 1 --iterations 3
python -m scripts.bench_vllm_qwen \
  --batch 256 --prompt-length 512 --output-tokens 32 --warmup 1 --iterations 1
```

## Remaining work

- Group Titans variable-length prefill only if FLA can emit each final state directly into its
  sequence slot and an end-to-end profile shows a useful remaining dispatch gap.
- Keep chunked/prefix grouped execution disabled until convolution, Titans, paged-prefix reads,
  and cross-request prefix-state semantics are proven together.
- Seek further decode gains by reducing or overlapping Titans/Polar bytes; standalone launch
  fusion is unlikely to move a path already at 76% of measured attainable HBM bandwidth.
