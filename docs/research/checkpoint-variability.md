# Pretrained checkpoint stress sweep to 256K

This report interprets the completed long-context stress sweep in
[`scaled_ablation/logs_stress/checkpoint_stress.json`](../../scaled_ablation/logs_stress/checkpoint_stress.json).
It is a post-training comparison of five 10B-token checkpoints trained at a
2,048-token context length. The sweep extends them to 262,144 tokens (128x the
training length).

## Experiment

```bash
FLA_CUSTOM_OP=1 python -m scaled_ablation.eval_hf_checkpoints \
  --metrics stress \
  --lengths 2048 4096 8192 16384 32768 65536 131072 262144 \
  --stress-num-docs 64 \
  --stress-modal-lengths 2048 4096 8192 16384 32768 65536 131072 262144 \
  --stress-modal-docs 64 \
  --stress-modal-samples 64 \
  --output scaled_ablation/checkpoint_stress.json
```

The recorded evaluation runtime was an NVIDIA L4 with PyTorch `2.13.0+cu130`,
CUDA 13.0, cuDNN 9.2, BF16 model weights, the Triton attention kernel, and
`FLA_CUSTOM_OP=1`. Every checkpoint completed all 64 passive documents and all
64 modal documents at every length, with zero OOMs and no recorded non-finite
components.

| Short name | Training device | Microbatch | Attention |
|---|---|---:|---|
| NoPE L4 mbs4 | L4 | 4 | NoPE |
| NoPE L40S mbs16 | L40S | 16 | NoPE |
| Polar L40S mbs16 | L40S | 16 | Polar |
| RoPE L40S mbs16 | L40S | 16 | RoPE |
| NoPE L40S mbs4 | L40S | 4 | NoPE |

The model has 16 blocks. Blocks 2, 6, 10, and 14 are attention blocks; the
other blocks use the local convolutional mixer. All attention variants include
the same Titans memory branch.

## How to read the probes

The stress runner reports three complementary signals:

- **Clean-document loss** is the functional result. Lower is better.
- **Passive RMS drift** measures how an activation's RMS changes relative to
  its 2K baseline. The generated summary calls the first departure outside
  `[0.8x, 1.25x]` a yield point. This is an operating-envelope alarm, not proof
  of failure by itself.
- **Randomized block secant gain** perturbs a block input by 2% RMS and measures
  the output response. The reported `random_secant_gain_max` is the average,
  over 64 documents, of each document's maximum across 64 random directions.
  It is a sampled lower bound on worst-direction gain, not an exact singular
  value.

A sampled gain above one is not sufficient to diagnose instability. Stable
models can contain locally expansive directions. The strong evidence is a
gain that grows with length, agrees with passive activation drift, and tracks a
loss regression.

## Functional result

Loss is in nats per native tokenizer token.

| Checkpoint | 2K | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NoPE L4 mbs4 | 1.9011 | 1.8300 | 2.2247 | 2.8940 | 3.3461 | 3.7031 | 3.9512 | 4.0668 |
| NoPE L40S mbs16 | 1.8900 | 1.7979 | 2.1145 | 3.4222 | 5.7994 | 7.9240 | 9.5894 | 10.7659 |
| **Polar L40S mbs16** | 1.9260 | 1.7847 | 1.6981 | 1.5583 | 1.4083 | 1.3411 | **1.2748** | **1.2918** |
| RoPE L40S mbs16 | 1.9498 | 1.9186 | 2.0192 | 2.1523 | 2.3164 | 2.5455 | 2.6980 | 2.8464 |
| NoPE L40S mbs4 | 1.8804 | 2.1572 | 3.6959 | 5.3346 | 7.0981 | 8.4679 | 9.3765 | 9.9048 |

The ordering is unambiguous:

1. **Polar is stable through the full sweep.** Its loss improves through 128K
   and changes only slightly at 256K.
2. **RoPE degrades slowly without a runaway activation chain.** It remains the
   second-best L40S checkpoint at 256K.
3. **NoPE L4 degrades moderately.** It is substantially more robust than both
   L40S NoPE checkpoints despite having the same attention type and memory.
4. **Both L40S NoPE checkpoints fail functionally.** Changing their microbatch
   changes the onset and shape of the failure but does not reproduce the L4
   checkpoint's stability.

## Sampled modal gain

Each entry is the largest block gain at that length, formatted as `gain@block`.

| Checkpoint | 2K | 8K | 32K | 128K | 256K |
|---|---:|---:|---:|---:|---:|
| NoPE L4 mbs4 | 1.364@14 | 1.320@14 | 1.247@14 | 1.220@10 | 1.233@10 |
| NoPE L40S mbs16 | 1.390@14 | 1.355@14 | 1.299@14 | 1.314@7 | 1.376@7 |
| Polar L40S mbs16 | 1.427@14 | 1.445@14 | 1.412@14 | 1.377@14 | 1.369@14 |
| RoPE L40S mbs16 | 1.325@14 | 1.291@14 | 1.268@14 | 1.254@14 | 1.249@14 |
| NoPE L40S mbs4 | 1.276@14 | 1.211@14 | 1.211@7 | 1.363@7 | 1.401@7 |

This table shows why absolute gain alone is misleading. Polar has the largest
2K gain, yet it remains functionally stable because that mode does not grow
with length and its passive outputs remain controlled. In contrast, block 7
in both L40S NoPE checkpoints becomes progressively more expansive as loss
worsens.

## Mechanistic interpretation by checkpoint

### Polar L40S mbs16: controlled length scaling

Polar is the only checkpoint whose language-model loss improves across almost
the entire range. Its block-14 maximum sampled gain declines from `1.427` at 2K
to `1.369` at 256K. Block 6 grows only mildly, from `1.024` to `1.109`.

The attention projections at blocks 2, 6, 10, and 14 finish at `1.001x`,
`1.076x`, `1.032x`, and `0.999x` their 2K RMS values. The final residual RMS
shrinks from `36.40` to `21.98` (`0.604x`) rather than exploding.

Polar's participation ratio `n_eff` does grow strongly with length. For
example, block 14 rises from `44.25` at 2K to `14,435.64` at 256K. The generic
25% envelope therefore flags `polar_n_eff` at 4K. This is expected state
scaling as more keys participate, not a failure: the block-14 attention
projection stays essentially flat (`1.122` to `1.121` RMS), while loss improves.
The null weight and memory gates adjust smoothly rather than producing a hard
transition.

**Interpretation:** Polar exposes a high but bounded local mode and converts a
length-growing internal count into a stable output. No unstable chain is
observed through 256K.

### RoPE L40S mbs16: gradual attenuation

RoPE loss increases from `1.9498` to `2.8464`, but its maximum block gain falls
from `1.325` to `1.249`. The block-15 residual finishes at `0.813x` its 2K RMS.
Several of its earliest passive yields are shrinking MLP outputs, including
block 8 (`0.568x` at 256K), blocks 11-12 (about `0.61x`), and block 7
(`0.767x`). Attention projections remain close to baseline.

**Interpretation:** RoPE's failure mode is representation attenuation or
position-distribution mismatch, not an amplifying residual chain.

### NoPE L4 mbs4: downstream redistribution, not an early-layer defect

The full sweep does **not** reproduce the block-0 spike seen in the pilot run.
Block 0 stays contractive and nearly constant: its sampled maximum gain ranges
only from `0.911` to `0.918` across all lengths. The earlier 16K observation was
therefore a low-sample, document/direction-sensitive outlier rather than
evidence that damage begins before the first attention layer.

The robust weak chain starts later:

- block-10 attention projection reaches `3.190x` its 2K RMS;
- block-10 sampled gain rises gradually from `1.128` to `1.233`;
- block-13 and block-9 MLP RMS reach `3.032x` and `2.750x` baseline;
- the final residual RMS contracts to `0.629x` baseline.

**Interpretation:** this checkpoint redistributes and partially suppresses the
residual stream while a block-10/downstream subspace becomes more sensitive.
Its loss degradation is real but does not resemble the L40S NoPE runaway.

### NoPE L40S mbs16: block 6 -> block 7 -> block 10 amplification

This checkpoint shows the clearest unstable chain:

1. Block-6 attention projection grows to `2.482x` its 2K RMS.
2. The following block-7 MLP grows to `19.768x` baseline.
3. Block-7 sampled gain rises monotonically from `1.018` at 2K to `1.376` at
   256K, crossing `1.20` at 64K.
4. Block-10 attention projection grows to `10.564x`; block-10 residual output
   reaches `3.951x` baseline.
5. Downstream block-13 and block-14 MLP outputs reach `6.703x` and `3.466x`.

Loss begins its sharp regression at 16K and reaches `10.7659` at 256K. The
coincidence of increasing local gain, large passive drift, and worsening loss
makes block 7 the strongest candidate for the checkpoint's unstable mode.

### NoPE L40S mbs4: the same family, earlier onset

The microbatch-4 L40S checkpoint develops a similar block-7 chain, but earlier:

- passive components first leave the 25% envelope at 4K;
- block-7 sampled gain rises from `1.018` to `1.401` and exceeds `1.20` at 32K;
- block-7 MLP RMS reaches `15.655x` baseline;
- block-10 attention projection and residual output reach `5.287x` and
  `3.166x` baseline.

Its loss is already `3.6959` at 8K and reaches `9.9048` at 256K.

**Interpretation:** reducing the L40S microbatch from 16 to 4 does not recover
the L4 checkpoint. It changes where and how quickly the NoPE instability
emerges, but the block-6/7/10 chain remains.

## What the sweep says about RMSNorm, microbatch, and hardware

The results do not support an evaluation-time RMSNorm sensitivity explanation:

- all five checkpoints were evaluated in the same process, device type,
  PyTorch build, dtype, kernels, documents, lengths, and perturbation protocol;
- no checkpoint produces NaN/Inf values or OOMs;
- RMSNorm keeps normalized inputs finite, but it cannot guarantee that a
  residual branch is contractive in every direction;
- the two unstable NoPE checkpoints show a directional block-7 mode and large
  branch-output drift, which can coexist with finite RMS-normalized inputs.

The L40S mbs4 control also shows that microbatch size alone is not sufficient to
explain the L4/L40S gap. However, this sweep is observational: the checkpoints
are separate training runs, and the recorded run configuration does not provide
a paired, controlled training seed. It therefore cannot distinguish GPU
reduction order, nondeterministic kernels, optimizer trajectory, data ordering,
or random initialization as the original cause. It establishes that the
difference is stored in the trained checkpoints, not introduced by this stress
evaluation.

## Conclusions and next causal tests

- **Polar is stable to 128x its training length** under this clean-document
  stress test. Its growing `n_eff` is internally absorbed rather than exported
  as activation or loss growth.
- **RoPE degrades gradually through attenuation**, without a growing modal
  hotspot.
- **NoPE has checkpoint-dependent stability.** The L4 checkpoint is moderately
  robust, while both L40S checkpoints develop a block-7-centered amplification
  chain.
- **The pilot block-0 defect is rejected by the full sample.** Block 0 is stable
  in the L4 checkpoint.
- **The observed failures are functional, not hard numerical failures.** Every
  run remains finite and completes at 256K.

The next experiments should be causal rather than another larger observational
sweep:

1. Inject the same normalized perturbation immediately before and after block 6
   and measure its transfer through block 7 and block 10.
2. Split the block-7 secant probe into convolutional-mixer and MLP gains.
3. Clamp or replace block-7 MLP output with the 2K-scale distribution and check
   whether long-context loss recovers.
4. Compare intermediate training checkpoints to identify when the block-7 mode
   appears.
5. Repeat paired L4/L40S training with identical recorded seeds, data order,
   optimizer state, and deterministic settings before assigning causality to
   hardware or microbatch.

The implementation and metric definitions are in
[`scaled_ablation/stress.py`](../../scaled_ablation/stress.py).
