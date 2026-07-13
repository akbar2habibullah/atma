# NVIDIA B300 profiling results

Run captured 2026-07-13 on one NVIDIA B300 SXM6 AC at commit
`dbfc0659594cb52dc721790eef5ebaed7aa65b25`.

## Validity

- PyTorch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1, compute capability 10.3.
- 287,428,771,840 bytes visible device memory (275,040 MiB in `nvidia-smi`).
- 13 targeted CUDA tests passed.
- BF16 CUDA matmul passed; no ECC errors or active thermal/power throttle were reported.
- Nsight Systems and Nsight Compute were not installed, so no timeline/counter reports were made.
- Training MFU was **not measured**: the first training process stopped because the optimized
  causal-conv1d package was unavailable. No training result should be inferred from this run.

## Hardware calibration

| Calibration | Measured | Nominal B300 SXM | Efficiency |
|---|---:|---:|---:|
| Representative BF16 GEMM | 1,449.3 TFLOP/s | 2,250 TFLOP/s | 64.4% |
| Device copy bandwidth | 6,504.5 GB/s | 8,000 GB/s | 81.3% |

The original `roofline_calibration.log` printed 7,100 GB/s and 91.6% because the harness
misclassified decimal 288 GB as less than 270 GiB. The GPU product is explicitly SXM6; 8,000 GB/s
is the correct nominal denominator. The measured 6,504.5 GB/s value itself is unaffected.

Relative to the repository's L40S calibration, this B300 delivered 6.95x the measured BF16 GEMM
throughput and 9.96x the device-copy bandwidth.

## Polar launch profile

All grouped kernels passed their correctness and speedup gates. `large` had the lowest summed p50:

| Profile | Sum of grouped p50 | Relative to `large` | Max observed error |
|---|---:|---:|---:|
| `large` | 0.7215 ms | 1.000x | 0 |
| `l4` | 0.7323 ms | 1.015x slower | 0 |
| `small` | 0.7932 ms | 1.099x slower | 0.00195 |

Keep Blackwell mapped to `large`. The win over `l4` is small, but `large` is also exact against the
oracle in this sweep.

The grouped Polar component reached 31.1M token/s on the mixed distribution and 26.9M token/s on
the long-tail distribution. At the complete canonical 378M-parameter model level, grouped mixed
prefill improved 23,574 to 30,795 token/s (1.31x) with unchanged peak memory.

## 9.2B inference

All rows include the sampler and use deterministic BF16 zero weights. They are systems-shape
measurements, not checkpoint-quality measurements.

### Prefill

| Workload | Tokens | p50 / p95 | Throughput | Peak allocated |
|---|---:|---:|---:|---:|
| Dense 8 x 512 | 4,096 | 64.97 / 68.35 ms | **63,046 tok/s** | 18.21 GiB |
| Grouped mixed 64-1536 | 4,384 | 270.89 / 281.50 ms | **16,184 tok/s** | 17.99 GiB |

Using the corrected 32-layer analytical cost, dense prefill delivered about 1,113 useful TFLOP/s:
76.8% of the measured GEMM ceiling and 49.5% of nominal BF16 peak. The approximate mixed workload
delivered only 286 useful TFLOP/s, or 19.7% of the measured ceiling.

Dense prefill is 5.28x the documented same-protocol L40S result (11,947 tok/s), while mixed prefill
is only 1.27x the L40S result (12,779 tok/s). The grouped Polar microkernel is fast, so the weak
mixed scaling is elsewhere in the heterogeneous full-model route. The leading candidate is the
per-sequence Titans/FLA state path and its dispatch behavior; this run has no Nsight trace to prove
the attribution.

### Decode at context 512

| Batch | p50 / p95 | Throughput | Peak allocated | Useful TFLOP/s* | Useful GB/s* |
|---:|---:|---:|---:|---:|---:|
| 512 | 14.69 / 14.78 ms | 34,861 tok/s | 33.81 GiB | 631 | 3,017 |
| 1,024 | 25.38 / 25.43 ms | 40,340 tok/s | 50.36 GiB | 730 | 2,782 |
| 2,048 | 48.49 / 48.54 ms | 42,232 tok/s | 83.51 GiB | 765 | 2,541 |
| 4,096 | 94.98 / 95.02 ms | **43,126 tok/s** | 149.79 GiB | 781 | 2,405 |

\*Derived from the corrected 32-layer roofline cost model; useful traffic is a lower bound, not a
hardware-counter measurement.

Batch 1,024 is the practical knee. Doubling to 2,048 adds only 4.7% throughput while increasing
latency 91%; doubling again to 4,096 adds 2.1% throughput while increasing latency 96%. Batch 4,096
still leaves substantial capacity, but further capacity points are unlikely to improve throughput
materially without reducing the recurrent-state and decode-body bottlenecks.

At batch 512 the B300 is 5.89x the documented L40S sampler-inclusive result. p95 is within 0.3% of
p50 across the large decode points, so steady-state decode is highly stable.

## Conclusions

1. B300 dense prefill and decode scale strongly and reproducibly; use batch 1,024 as the default
   throughput/latency operating point for the 9.2B model.
2. Retain the `large` Polar profile on Blackwell.
3. The highest-value optimization target is heterogeneous prefill outside the grouped Polar
   kernel. Profile or isolate per-sequence Titans/FLA execution before changing Polar tiles.
4. Training MFU remains open. A follow-up run must either cache a Blackwell-compatible
   causal-conv1d kernel or explicitly benchmark the compiled PyTorch fallback and label it.
5. Install Nsight tools in the image before another paid session; their absence prevented direct
   kernel attribution in this run.

Raw data is in [`logs/`](logs/); the structured aggregation is [`logs/summary.json`](logs/summary.json).
