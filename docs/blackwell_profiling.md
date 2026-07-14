# B200/B300 profiling runbook

This run is designed for a single rented GPU with less than one hour of paid time. The default
session spends at most 45 minutes on work and preserves the final 5 minutes for copying artifacts.
It profiles deterministic BF16 Atma inference, not model quality: the 9.2B stress model uses zero
weights but executes the production GEMM and kernel shapes.

## Before starting the rental

Build an image or persistent volume with the repository and its Python environment already
installed. At minimum, verify imports for `torch`, `triton`, `fla` (flash-linear-attention), and
`pytest`. Install Nsight Systems
if a timeline is required. Do not plan to install packages, compile a custom PyTorch build, or
download a checkpoint during the paid session.

Validate the command plan on any machine (CUDA is not needed):

```bash
python -m scripts.profile_blackwell --dry-run
```

Copy the repository or persistent Triton cache to the rental host before the clock starts when the
provider permits it. A cache compiled for another GPU architecture is not a substitute for the
Blackwell warmups; the runner deliberately compiles before timing.

## Paid session

From the repository root:

```bash
python -m scripts.profile_blackwell --phase preflight --output-dir /mnt/results/preflight
python -m scripts.profile_blackwell --phase all --budget-minutes 50 \
  --reserve-minutes 5 --output-dir /mnt/results/blackwell
```

The preflight captures the exact GPU SKU, clocks/power information, driver, PyTorch CUDA build,
compute capability, optional shared memory, installed-package discovery, git commit and dirty
state, and Nsight versions. Inspect `metadata.json` immediately. CUDA availability plus a BF16
matmul is the only hard preflight gate; heavyweight FLA/Triton kernel imports and execution are
gated immediately before the relevant benchmark. Stop the rental if CUDA is unavailable, the GPU
is not the paid SKU, or the driver/runtime combination cannot launch the BF16 tensor operation.

The correctness tests and two small-model smoke workloads are hard gates. The remaining order is:

1. Blackwell peak-seeking, Atma 9.2B inference, and 378M/9.2B training GEMM calibration;
2. Polar `l4`, `small`, and `large` launch-profile sweep;
3. canonical 16-layer heterogeneous prefill;
4. 9.2B dense prefill at batches 8, 16, 32, and 64, plus mixed prefill;
5. 9.2B decode at batches 512, 1024, 2048, 4096, 6144, and 7168;
6. compact Nsight Systems traces and one bounded grouped-Polar Nsight Compute report, when the
   respective tools are installed.

Each workload runs in a new process so graph pools, Triton/FLA shape state, and peak-memory
statistics do not leak between measurements. Every command gets its own log and `manifest.json`
records status and elapsed time. `summary.json` gathers the structured stress measurements and
selected tuning profile for quick copying. When the time reserve is reached, no new workload starts.

For a shorter or lower-risk first rental:

```bash
python -m scripts.profile_blackwell --phase all --budget-minutes 35 \
  --reserve-minutes 5 --decode-batches 512,1024,2048
```

Run only the B300 tensor and end-to-end saturation calibration with:

```bash
python -m scripts.profile_blackwell --phase calibrate --budget-minutes 25 \
  --reserve-minutes 3 --prefill-batches 8,16,32,64 \
  --decode-batches 512,1024,2048,4096,6144,7168 \
  --output-dir /mnt/results/b300-calibration
```

`blackwell_tensor_calibration.log` separates physical peak-seeking GEMMs from exact matrix
families for 9.2B prefill/decode, canonical 378M training, and 9.2B training. Use the peak group for
physical attainable MFU and the matching model group for shape efficiency.

For B300 capacity exploration, add larger fresh-process points only after the standard run:

```bash
python -m scripts.profile_blackwell --phase benchmark --budget-minutes 15 \
  --reserve-minutes 2 --decode-batches 4096,6144,8192 \
  --output-dir /mnt/results/b300-capacity
```

## Training MFU

Training MFU uses synthetic token IDs so it performs no dataset download, validation, logging, or
checkpoint I/O. It executes the real compiled training model, summed cross-entropy, backward,
gradient clipping, fused AdamW, and Muon. Warmup/compilation is reported separately and excluded
from steady-state latency.

Run a B300 microbatch sweep in fresh processes:

```bash
python -m scripts.profile_blackwell --phase training --budget-minutes 30 \
  --reserve-minutes 3 --train-microbatches 8,16,32,64 \
  --train-global-sequences 512 \
  --train-require-fla \
  --train-measured-peak-tflops 2239.4 \
  --output-dir /mnt/results/b300-training
```

This is the primary fair training comparison: it keeps the canonical 378M/D1024 model and the
original 524,288-token global batch fixed. Microbatches 8/16/32/64 therefore use gradient
accumulation 64/32/16/8. Changing to D4096/L32 is a separate capacity/saturation experiment, not a
replacement for the canonical MFU measurement.

For a 9.2B/D4096 training saturation sweep, the default microbatches automatically become 1,2,4,8:

```bash
python -m scripts.profile_blackwell --phase training --budget-minutes 35 \
  --reserve-minutes 3 --train-hidden-size 4096 --train-layers 32 \
  --output-dir /mnt/results/b300-training-9b
```

Or run one shape directly:

```bash
python -m scripts.bench_training_mfu --microbatch 16 --seq-length 1024 \
  --grad-accum 1 --warmup 2 --iterations 5 --measure-peak
```

The compiled PyTorch causal-convolution fallback and validated eager PyTorch Titans fallback are
allowed and explicitly recorded because the optional kernels may be unavailable for B300. Use
`--require-optimized-conv` and/or `--require-fla` only in a future environment expected to provide
those compatible optimized kernels. MFU from a fallback run describes that deployed backend and
must not be presented as fused-kernel performance.

FLA 0.5.x imports its optional model package before `fla.ops`. A binary-incompatible optional
`torchaudio` installation can therefore make the kernel import fail inside Transformers. Atma does
not use audio; remove that package (`python -m pip uninstall -y torchaudio`) or install the exact
TorchAudio build matching the installed PyTorch build before using `--train-require-fla`.

The primary `mfu_hybrid_*` fields use `6*N + 12*attention_layers*hidden*sequence` FLOPs per token.
The `mfu_legacy_*` fields reproduce `train.py`'s historical convention, which charges quadratic
attention to all layers even though Atma has four attention layers out of sixteen. Both formulas
time Polar, Titans, causal convolution, loss, clipping, and optimizer work but do not fully credit
those operations in the numerator. Treat MFU as a documented useful-model-FLOP proxy, not a count
of every executed instruction. Nominal MFU uses the published dense BF16 peak; measured MFU uses
the same-session representative BF16 GEMM calibration.

An OOM is a capacity observation, not a deployment recommendation. The stress harness allocates
only live state and one exact-batch CUDA graph; it does not retain every production graph bucket or
reserve unused serving cache.

## Decisions to make from the artifacts

- The runner selects the profile with the lowest summed grouped-kernel p50 and applies it to later
  workloads; inspect `tuning.json` and require the canonical full-model result to confirm the gain.
  `auto` currently maps Blackwell to `large` when no explicit profile is selected.
- Compare prefill against measured BF16 throughput and decode against measured HBM throughput;
  nominal peak percentages alone are sensitive to provider power and clock limits.
- Use the Nsight timeline to decide whether the next experiment targets dispatch gaps, GEMMs,
  Polar/Titans kernels, or memory traffic. The automated Compute report collects only one grouped
  Polar launch. Do not run broad Nsight Compute section sets over the full model under a one-hour
  budget; select any additional hot kernel from the Systems trace for a follow-up command.
- Report SKU and memory size explicitly. B300 SXM (288 GB) and the 252 GB DGX Station B300 have
  different published memory bandwidth, and provider power limits can change attainable results.

The roofline helper auto-labels B200/B300 and uses published dense BF16 peaks. It distinguishes the
252 GB B300 memory SKU from 288 GB B300 for the nominal bandwidth; override uncertain provider
SKUs explicitly with `--bf16-tflops` and `--hbm-gbps`.
