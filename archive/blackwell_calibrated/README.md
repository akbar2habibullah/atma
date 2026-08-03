# Calibrated NVIDIA B300 results

Run captured 2026-07-13 on one NVIDIA B300 SXM6 AC. Raw logs and structured measurements are in
[`logs/`](logs/) and [`logs/summary.json`](logs/summary.json).

## Result status

| Area | Status |
|---|---|
| CUDA environment and 13 targeted tests | Pass |
| B300 tensor/HBM calibration | Pass |
| 9.2B prefill saturation sweep | Pass |
| 9.2B decode saturation sweep | Pass |
| Polar route sweep and canonical grouped comparison | Pass |
| 9.2B training MFU | **Not measured** |
| Nsight Systems / Compute | Not installed |

Training stopped before model construction because the environment could discover an `fla` module
but `from fla.ops.gated_delta_rule import chunk_gated_delta_rule` was unavailable. The convolution
fallback was accepted. The training log contains no latency, throughput, memory, or MFU result.

## B300 calibration

### Physical ceilings

| Calibration | Measured | B300 SXM nominal | Efficiency |
|---|---:|---:|---:|
| Peak BF16 GEMM, maximum | **2,239.4 TFLOP/s** | 2,250 TFLOP/s | **99.5%** |
| Peak BF16 GEMM, geometric mean | 2,237.8 TFLOP/s | 2,250 TFLOP/s | 99.5% |
| 4 GiB device copy | **6,597.1 GB/s** | 8,000 GB/s | **82.5%** |

The three peak GEMMs are extremely consistent: 2,236.0-2,239.4 TFLOP/s. The old K=1024
calibration substantially understated this GPU's attainable compute ceiling.

Memory copy scaling was 6,163.6 GB/s at 256 MiB, 6,437.3 GB/s at 1 GiB, and 6,597.1 GB/s at
4 GiB. The measured peak balance point is approximately 339 FLOP/byte; the nominal balance point
is 281 FLOP/byte.

### Model-shape GEMMs

| Shape family | Shapes | Minimum | Geometric mean | Maximum | Geomean / measured peak |
|---|---:|---:|---:|---:|---:|
| Atma 9.2B prefill | 5 | 1,908.7 | **2,094.7** | 2,224.5 | **93.5%** |
| Atma 9.2B decode | 12 | 1,651.3 | **2,103.4** | 2,252.2 | **93.9%** |
| Atma 378M training fwd/dX/dW | 24 | 1,737.4 | **2,013.8** | 2,262.7 | **89.9%** |
| Atma 9.2B training fwd/dX/dW | 24 | 1,872.7 | **2,084.4** | 2,248.9 | **93.1%** |

The 9.2B tensor shapes are large enough to saturate B300 Tensor Cores. Small decode batch 512 is
the weak edge (1,651 TFLOP/s); decode batch 1024 already reaches roughly 2,080-2,090 TFLOP/s for
the measured MLP/head shapes, and batches 4096+ are near physical peak.

These are isolated GEMM efficiencies, not end-to-end MFU. Polar, Titans, convolution, norms,
state traffic, loss, and optimizer execution are not represented by a GEMM-only number.

## 9.2B prefill saturation

All rows use BF16 deterministic zero weights and include the sampler.

| Batch | Prompt tokens | p50 | Throughput | Peak allocated | Relative to maximum |
|---:|---:|---:|---:|---:|---:|
| 8 | 4,096 | 64.95 ms | 63,060 tok/s | 18.21 GiB | 91.8% |
| 16 | 8,192 | 125.32 ms | 65,370 tok/s | 19.24 GiB | 95.2% |
| 32 | 16,384 | 240.19 ms | 68,214 tok/s | 21.28 GiB | 99.3% |
| 64 | 32,768 | 477.21 ms | **68,666 tok/s** | 25.39 GiB | 100% |

The automated 95% knee is batch 16. For a stricter practical plateau, batch 32 is preferable: it
is within 0.7% of maximum, while batch 64 doubles latency for only 0.66% more throughput. The sweep
demonstrates saturation; a larger batch is unnecessary.

## 9.2B decode saturation at context 512

| Batch | p50 | Throughput | Peak allocated | Relative to maximum |
|---:|---:|---:|---:|---:|
| 512 | 14.67 ms | 34,897 tok/s | 33.81 GiB | 80.4% |
| 1,024 | 25.35 ms | 40,394 tok/s | 50.36 GiB | 93.1% |
| 2,048 | 48.45 ms | 42,269 tok/s | 83.51 GiB | 97.4% |
| 4,096 | 94.89 ms | 43,164 tok/s | 149.79 GiB | 99.5% |
| 6,144 | 141.65 ms | **43,375 tok/s** | 216.08 GiB | 100% |
| 7,168 | 166.21 ms | 43,127 tok/s | 249.23 GiB | 99.4% |

Decode is conclusively saturated. Batch 6144 is the throughput maximum; batch 7168 consumes an
additional 33.1 GiB and is slower. The automated 95% knee is batch 2048. Batch 1024 remains the
better latency-oriented operating point, while batch 4096 is within 0.5% of maximum throughput.

## Heterogeneous prefill and Polar routing

The 9.2B mixed workload (lengths 64-1536, 4,384 tokens) reached 23,735 tok/s at 184.71 ms p50 and
17.99 GiB peak allocation. This is materially below dense saturation, so heterogeneous routing and
Titans execution remain the highest-value end-to-end optimization target.

The canonical 378M mixed model improved from 23,801 tok/s on the oracle route to 39,177 tok/s on
the grouped route: **1.65x**, with unchanged peak memory.

### Polar profile choice

The automatic selector reported `l4` because it summed four cases, including homogeneous grouped
prefill. Production homogeneous prefill uses the dense route, so that case should not determine
the grouped heterogeneous profile.

| Profile | Sum p50, three heterogeneous cases | Relative | Numerical result |
|---|---:|---:|---|
| `large` | **0.5390 ms** | 1.000x | Exact in this sweep |
| `l4` | 0.5951 ms | 1.104x slower | Exact |
| `small` | 0.7170 ms | 1.330x slower | Max error 0.00195 |

For the actual grouped route, retain `large`. The selector should exclude homogeneous or weight
the production distribution explicitly.

## Training status and next action

The isolated training GEMMs prove that both 378M and 9.2B training tensor shapes can drive B300 at
approximately 90-93% of its measured GEMM ceiling. They do **not** provide training MFU.

Current Flash Linear Attention releases support Blackwell, so the failed import is an environment
version/dependency problem rather than evidence that B300 is unsupported. Capture the installed
`flash-linear-attention` version and full import traceback, then use v0.5.1 or newer. If it cannot
be repaired in the rental image, the benchmark can run with the explicitly labeled eager PyTorch
Titans fallback, but that result must not be presented as fused-FLA MFU.

## Conclusions

1. The calibrated B300 reaches essentially full advertised BF16 GEMM throughput.
2. Atma 9.2B GEMM shapes are sufficiently large; increasing model dimensions is not required for
   Tensor Core saturation.
3. Dense prefill saturates by batch 32, and decode saturates by batch 4096-6144.
4. Heterogeneous prefill, not dense GEMM sizing, is the primary inference optimization gap.
5. `large` is the correct grouped Polar profile for the measured heterogeneous workloads.
6. Training MFU is still outstanding because no training step executed.
