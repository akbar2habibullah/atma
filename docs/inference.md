# Inference

ATMA includes a paged inference engine for the production Polar Attention + Titans MAG model.
The engine supports chunked prefill, paged KV storage, per-sequence recurrent state, tensor
parallel linears, and CUDA-graph decode. Kernel selection, L40S measurements, and tuning details
live in [kernel.md](kernel.md).

## Usage

```python
from inference import LLM, SamplingParams

llm = LLM(model="path/to/weights.pt")
params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Hello, world!"], params)
print(outputs[0]["text"])
```

The engine loads the canonical 16-layer, hidden-size-1024 model by default. It uses BF16 model
weights and activations, with FP32 Titans recurrent states. `inference/generate.py` is a smaller
legacy full-recompute utility; it does not exercise the paged serving path.

## Execution model

### Prefill

The scheduler packs scheduled tokens and selects one of three routes:

| Workload | Route |
|---|---|
| Fresh, complete, equal-length batch | Dense batched prefill |
| Fresh, complete, heterogeneous batch | Grouped packed prefill |
| Single request, chunked continuation, or cached prefix | Per-sequence fallback |

Dense prefill operates on `[B, T]` storage and batches convolution, Polar Attention, and Titans
memory. Grouped prefill retains packed token storage and uses a tile-to-sequence map for one
ragged Polar launch plus boundary-aware packed Canon/LFM convolutions. Titans prefill remains
per sequence. Both optimized routes write into the same paged K/V cache and slot-indexed state
tables as the fallback.

Chunked continuation is exact: Canon/LFM and Titans states carry through their tables, while
Polar gathers the cached prefix from paged K/V and applies absolute causal/window bounds. It is
deliberately not routed through the fresh grouped kernels.

### Decode

Decode is fixed-shape and CUDA-graph captured at supported batch sizes. Its hot path includes:

- direct paged Polar reads through block tables, without gathering K/V into a second buffer;
- one GQA-grouped Polar program per `(sequence, KV head)`;
- in-place slot-indexed Titans state updates;
- fused causal-convolution state updates;
- fused squared-ReLU gating and output softcap.

The Titans state is FP32 in FLA `[K, V]` layout. At the canonical shape it occupies roughly
512 KiB per sequence per memory layer and is the dominant large-batch decode cost. The graph
path therefore avoids changes that add state or cache traffic merely to reduce launches.

### Tensor parallelism

The engine supplies replicated, column-parallel, row-parallel, merged-column, QKV-parallel,
vocabulary embedding, and LM-head layers. Model weights are sharded during loading according to
the layer's tensor-parallel dimension. Kernel fast paths preserve the existing shard layouts.

## Cache and recurrent state

K/V is stored in fixed-size pages managed by `inference/engine/block_manager.py`. The scheduler
may reuse hash-matched prefix pages and preempt sequences when necessary. Canon, LFM, and Titans
state is held separately in centralized tables indexed by a stable sequence slot, allowing CUDA
graphs to perform all decode state access with GPU indices.

Known limitation: a prefix page reused from a different request supplies correct K/V, but the new
request does not inherit the source request's convolution and Titans states. Its first outputs
after the shared prefix can therefore drift. Same-request chunked continuation is exact. Optimized
fresh-prefill routes exclude prefix hits.

## Verification

The inference implementation is checked against `model/reference.py` and the training model:

```bash
python -m pytest tests/test_baseline_inference.py tests/test_dense_prefill.py \
  tests/test_decode_kernels.py -q
python -m tests.verify
python -m tests.verify --cuda
python -m tests.verify_fla
```

Coverage includes dense and grouped routing, non-power-of-two ragged lengths, sliding windows,
sequence-boundary sentinels, paged K/V writes, every convolution and Titans state, chunked state
carry, eager/graph decode behavior, and fused kernel parity. Current L40S results and exact test
status are recorded in [kernel.md](kernel.md#verification).

## Benchmarks

Use the serving benchmark for end-to-end prefill and decode:

```bash
python -m scripts.bench_inference
```

Use the kernel-efficiency harness for deterministic L40S route comparisons:

```bash
python -m scripts.bench_kernel_efficiency --only b
python -m scripts.bench_kernel_efficiency --full-model mixed --warmup 2 --iterations 10
python -m scripts.roofline_inference --measure \
  --prefill-tok-s 174571 --decode-tok-s 64583
```

For isolated large-model capacity and throughput stress tests:

```bash
python -m scripts.stress_inference --mode prefill --batch 8 --prompt-length 512
python -m scripts.stress_inference --mode decode --batch 768 --context-length 512
```

The 9.2B L40S sweep and its memory-model caveats are documented in
[kernel.md](kernel.md#92b-l40s-stress-test).

Benchmark compile/autotune separately from steady state and report the checkpoint status, dtype,
tensor-parallel size, route, warmups, iterations, p50/p95 latency, throughput, and peak memory.
