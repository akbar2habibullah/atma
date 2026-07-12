# L40S kernel-efficiency results

Execution record for [`KERNEL_EFFICIENCY_PLAN.md`](KERNEL_EFFICIENCY_PLAN.md), captured
2026-07-12 UTC.

## Environment

| Field | Value |
|---|---|
| Base commit | `06b2d8599674620e86d584f6248980110d2e1e82` (clean before work; result tree dirty) |
| GPU | NVIDIA L40S, compute capability 8.9, 46,068 MiB |
| Driver / advertised CUDA | 595.71.05 / 13.2 |
| Python / PyTorch / build CUDA | 3.12.13 / 2.13.0+cu130 / 13.0 |
| Triton | 3.7.1 |
| Model | canonical 16-layer, hidden 1024, H=8, KVH=2, D=128, window=1024, Titans enabled |
| Benchmark weights | deterministic zero weights (shape/dispatch benchmark; no checkpoint found) |
| Inference | BF16, TP=1, eager fresh prefill; warmup=2, measured=10 |

The calibrated roofline remained 208.6 TFLOP/s BF16 and 653.2 GB/s. The recorded dense-prefill
baseline is compute/dispatch limited (46.7% of its attainable compute roof), while decode is at
76.1% of attainable HBM bandwidth. Decode was therefore left unchanged.

## Workstream A

### A1 projection plus per-head RMSNorm

A narrow Triton prototype computes the K projection plus head RMSNorm, or the interleaved Q/gate
projection while normalizing only Q. FP32 reductions use epsilon `1e-6`; the maximum difference
from cuBLAS plus PyTorch RMSNorm was 0.03125 in BF16.

The candidate was inconsistent across the required row counts. It won some launch-latency shapes
but lost important prefill shapes: at M=4096/8192, K speedups were 0.92x/0.78x and Q speedups were
0.57x/0.71x. It is retained only as a benchmark prototype and is not routed by the model.

### A2 Lazy Pre-Norm

The derived-weight algebra was evaluated for `norm2 -> MLP.fc` at output width 8192. An unfused
derived-weight implementation reached only 0.48-0.73x of the existing route over M=1..8192 and
introduced additional BF16 rounding when gamma was folded into a BF16 derived weight. A custom
mainloop was not justified after A1 and this result. Training and broader CODA rewriting were
stopped, as required by the gate.

## Workstreams B1-B3

The shipped fresh heterogeneous route adds:

- a reusable query-tile-to-sequence map built once per scheduled batch;
- one packed ragged Polar launch per layer, with exact causal/window masking and Polar statistics;
- one boundary-aware packed depthwise-convolution launch per Canon/LFM convolution, including
  exact final-state writes to existing sequence slots.

Equal complete prompts retain the dense route. Single prompts, chunked continuation, cached
prefixes, CPU execution, and decode retain their prior routes. No cache representation or state
table layout changed.

### Polar component

`python -m scripts.bench_kernel_efficiency --only b` (20 warmups, 100 iterations):

| Distribution | Tokens | Oracle p50 | Grouped p50 | Speedup | Max error |
|---|---:|---:|---:|---:|---:|
| short-heavy | 816 | 2.1443 ms | 0.1300 ms | 16.49x | 0 |
| mixed | 4,384 | 2.1801 ms | 0.1475 ms | 14.78x | 0 |
| long-tail | 8,192 | 2.1258 ms | 0.4751 ms | 4.47x | 0 |

Polar launches fall from eight to one. A full attention layer including Canon and Titans improved
1.61x, 1.61x, and 1.58x on short-heavy, mixed, and long-tail respectively.

### Full 16-layer prefill

Each row was run in a fresh process to keep FLA shape caches out of the memory comparison:

| Distribution | Route | p50 | p95 | Mean | Throughput | Peak temporary allocation |
|---|---|---:|---:|---:|---:|---:|
| short-heavy | oracle | 113.0476 ms | 114.9492 ms | 112.8396 ms | 7,218 tok/s | 776.5 MiB |
|  | grouped | 67.9475 ms | 70.9427 ms | 68.4168 ms | 12,009 tok/s | 776.5 MiB |
| mixed | oracle | 111.4716 ms | 112.7148 ms | 111.3880 ms | 39,328 tok/s | 901.2 MiB |
|  | grouped | 67.9731 ms | 69.4395 ms | 67.9602 ms | 64,496 tok/s | 901.3 MiB |
| long-tail | oracle | 111.1296 ms | 113.6251 ms | 110.5155 ms | 73,716 tok/s | 1,039.8 MiB |
|  | grouped | 65.5667 ms | 67.7120 ms | 65.9255 ms | 124,941 tok/s | 1,039.9 MiB |

The end-to-end speedups are 1.66x, 1.64x, and 1.69x. Mixed and long-tail exceed the 20% gate,
and measured temporary memory growth is below 0.1%. The homogeneous production route is not
modified; routing tests assert that it remains dense.

## Correctness and limitations

Passing checks:

- `tests/test_baseline_inference.py`, `tests/test_dense_prefill.py`, and
  `tests/test_decode_kernels.py`: 20 passed;
- `python -m tests.verify`: 30 passed on CPU;
- `python -m tests.verify --cuda`: 30 passed on CUDA;
- `python -m tests.verify_fla`: FLA mapping/state/decode bridge passed at established tolerances;
- randomized non-power-of-two ragged Polar tests, window variants, convolution boundary
  sentinels, final convolution states, and K/V cache comparisons pass.

`tests/test_edge_kernels.py` could not be collected because `tinygrad` is not installed on this
machine. Chunked continuation deliberately remains on the oracle route; grouped continuation and
prefix-cache state reconstruction are not claimed. Nsight Systems was not available, so launch
reductions are reported from route structure rather than an Nsight trace.

## Decision

Enable grouped Polar and grouped Canon/LFM convolution for complete fresh heterogeneous CUDA
prefills. Retain every established fallback. Do not enable the A1 prototype, proceed with A2, or
start broader CODA work. The remaining heterogeneous launches are the per-sequence Titans FLA
calls; grouping them is a possible later independent workstream, not required for this gate.
