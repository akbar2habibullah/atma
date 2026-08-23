# Robustness and modern-baseline supplement

This directory is the source of truth for the third supplementary experiment in the
paper revision. Commands below are run from the repository root. Training workers read
the single operational tree at `supplementary/robustness/configs`; do not create a second
config tree under `work/`.

## Current status (2026-08-23)

| Experiment group | Runs | Status |
|---|---:|---|
| Polar component attribution, 1B | 5 | **Complete**; logs are committed |
| TDA, Mamba-3, GDN-2 screening, 1B | 3 | **Complete**; logs are committed |
| Paired Polar/NoPE seeds, 10B | 4 | **Not run** |
| Promoted TDA plus one of Mamba-3/GDN-2, 10B | 2 | **Not run**; promotion decision is not yet recorded |

The six final 10B jobs are the four Polar/NoPE replications, promoted TDA, and exactly
one selected linear-recurrent baseline (Mamba-3 or GDN-2). They are independent jobs and
are intended to run on six separate one-GPU machines.

The 68B-token ceiling is 40B for paired replications, 5B for the completed component
study, 3B for external pilots, 10B for TDA, and 10B for one selected Mamba-3/GDN-2 model.

## Completed 1B results and interpretation

All eight runs completed 1,900 training steps without a logged error. The component
runs use the same seed and 378.2M parameters. The external models are within the
prespecified 5% parameter-matching band at 366.3M--388.6M parameters. The tables report
validation loss, coherent FinePDFs and concatenated FineWeb-Edu losses at the 2K and 65K
endpoints, and teacher-forced five-token needle accuracy at every evaluated context
length. Lower is better for loss and higher is better for needle accuracy. The log fields
call the two evaluation losses `clean_ppl` and `junk_ppl`, so the tables retain that
label, but the implementation records mean cross-entropy in nats/token rather than
exponentiated perplexity. Full-precision records are in [`work/logs`](work/logs/).

### Polar component attribution

| Variant | Final val. loss | Clean PPL, 2K / 65K | Junk PPL, 2K / 65K | Needle accuracy (%), 2K / 4K / 8K / 16K / 32K / 65K |
|---|---:|---:|---:|---:|
| Full Polar | 3.1688 | 2.810 / 2.047 | 3.110 / 3.129 | 91.25 / 93.75 / 87.50 / 88.75 / 86.25 / 77.50 |
| Direction only (remove count channel) | **3.1629** | **2.709** / 1.982 | **3.103 / 3.124** | 91.25 / 90.00 / 86.25 / 85.00 / 81.25 / 71.25 |
| Constant magnitude | 3.1717 | 2.734 / **1.924** | 3.113 / 3.134 | 92.50 / 81.25 / 61.25 / 55.00 / 36.25 / 16.25 |
| Fixed null slope | 3.1741 | 2.725 / 1.938 | 3.117 / 3.142 | **96.25 / 98.75 / 97.50 / 96.25 / 97.50 / 91.25** |
| Fixed temperature gain | 3.1695 | 2.707 / 1.967 | 3.111 / 3.155 | 93.75 / 97.50 / 72.50 / 53.75 / 33.75 / 16.25 |

The validation losses differ by only 0.0112 nats, and every component variant preserves
or improves clean perplexity relative to Full Polar at matched lengths. The distinguishing
signal is therefore long-range retrieval rather than ordinary language-model loss.

- The direction channel carries most of Polar's retrieval ability: removing the count
  channel retains 71.25% at 65K, only 6.25 percentage points below Full Polar, while
  slightly improving validation loss and perplexity.
- An input-dependent magnitude is important if the count channel is present. Replacing
  it with a constant leaves short-context accuracy intact but falls from 92.50% at 2K
  to 16.25% at 65K. The direction-only result shows that this is specifically harm from
  an uninformative constant count signal, not evidence that a count channel is always
  required.
- Length-dependent temperature is the clearest necessary component in this sweep.
  Removing its learned length gain produces the same long-context collapse as constant
  magnitude: 16.25% at 65K, versus 77.50% for Full Polar.
- The learned length-dependent null slope is not supported by this pilot. Fixing that
  slope near zero gives the best retrieval result at every length, including 91.25% at
  65K, although its final validation loss is slightly worse. This points to null-slope
  simplification or retuning; it does not establish the effect across seeds.

### TDA, Mamba-3, and GDN-2 screening

| Pilot | Params | Final val. loss | Clean PPL, 2K / 65K | Junk PPL, 2K / 65K | Needle accuracy (%), 2K / 4K / 8K / 16K / 32K / 65K | Final MFU |
|---|---:|---:|---:|---:|---:|---:|
| TDA hybrid | 388.6M | **3.7749** | **3.350 / 2.645** | **3.707 / 3.794** | **56.25 / 28.75 / 11.25 / 5.00 / 3.75 / 2.50** | **38.49%** |
| Mamba-3 native | 368.1M | 3.8383 | 3.810 / 3.516 | 3.747 / 3.805 | 1.25 / 2.50 / 1.25 / 1.25 / 1.25 / **3.75** | 31.69% |
| GDN-2 native | 366.3M | 3.9381 | 3.825 / 3.342 | 3.844 / 3.913 | 2.50 / 2.50 / 1.25 / 3.75 / 1.25 / 2.50 | 30.12% |

TDA is the strongest screening result: it has the best final validation loss, clean and
junk perplexity at both endpoints, short-range needle accuracy, and measured throughput.
Its retrieval nevertheless decays sharply with length, reaching 11.25% at 8K and 2.50%
at 65K. The completed, stable pilot and its lead on the prespecified metrics support
promoting TDA to the 10B run, but the pilot alone is not evidence of robust long-context
retrieval.

Neither linear-recurrent pilot demonstrates useful needle retrieval at this scale; both
remain near zero token accuracy. Their secondary metrics give a mixed selection signal.
Mamba-3 has better final validation loss, junk perplexity, and MFU. GDN-2 has better
needle cross-entropy at all six lengths (despite similarly low token accuracy) and better
clean perplexity at 4K, 32K, and 65K. Because the protocol does not encode a scalar
tie-break among validation, clean/junk perplexity, and retrieval, these results do not
justify presenting either linear model as an unambiguous winner. Any promotion record
should state which prespecified metric priority resolves that tradeoff.

These are single-seed, 1B-token screening and attribution runs evaluated on 16 clean
documents and 16 needle trials. They support model selection and mechanism diagnosis,
not uncertainty estimates or final comparisons with the planned 9.816B-token runs.

## Configuration safety

Validate the checked-in configs without regenerating them:

```bash
python -m supplementary.robustness.validate_plan
```

Do **not** run `generate_configs --clean` now. It resets parameter approvals and scaled
promotion flags. Claims, logs, and state markers live under
`supplementary/robustness/work/`; checkpoints live under
`checkpoints/supplementary_robustness/`.

## Pinned GPU setup

Every GPU machine must start from the same repository commit and the validated runtime:
PyTorch `2.13.0+cu130`, CUDA `13.0`, and Triton `3.7.1`. The setup command verifies and
preserves that build, checks out
only the pinned sources required by the machine role, installs them editable, and fails
on a dirty or mismatched checkout.

```bash
python -m supplementary.robustness.setup_gpu_machine --role ROLE
```

`ROLE` is one of `replication`, `tda`, `mamba3`, or `gdn2`. The pins are recorded in
`dependencies.json`. TDA remains subject to its upstream non-commercial research-only
license.

Mamba-3 is configured with `compile_model=true` and `external_custom_op=true`. Its fused
SISO recurrence and RMSNorm are opaque to `torch.compile`, using the same
forward/recomputed-backward pattern as `model/blocks.py`. Exact rotary-angle gradients
come from the pinned Mamba backward directly. The pinned upstream tree is not patched.

The recomputed-backward custom-op approach was rejected for GDN-2 after measuring only
0.81× eager block speed. The final GDN-2 path instead retains the pinned FLA autograd
kernels, compiles the projection/MLP/loss regions around their explicit boundaries, and
replays the fixed training microstep through a CUDA graph. A complete L40S global step at
`mbs=4`, `seq_len=2048` measured 12.318 seconds, 30.28% MFU, and 12.09 GiB peak reserved
memory, versus 22.73 seconds, 16.4% MFU, and 24.72 GiB for the old eager path. TDA is
also split-compiled and CUDA-graphed without modifying the pinned checkout. Its fixed
threshold is passed as a host scalar, eliminating eight device/host synchronizations per
microbatch; tensor-only projection, normalization, memory, MLP, and loss regions compile
around the pinned TDA and FLA kernels. The same pinned FP32 TDA kernels use launch tiles
tuned for the approved `head_dim=64` shape; checked bf16 outputs and gradients are exactly
equal to the upstream 64x64 launch at sequence lengths 128 and 2048.

On the validation L40S, three complete 524,288-token TDA steps after warmup measured a
9.853-second median, **40.20% MFU**, and **10.77 GiB peak reserved memory**. Each step
included 64 microbatches, gradient clipping, and a fused AdamW update. The earlier eager
path with the same pinned dependencies measured 19.933 seconds, 19.87% MFU, and 22.68
GiB reserved. Ordinary validation,
evaluation, and checkpoint execution remain unchanged.

## Reproducing the completed 1B pilots

First install all pilot dependencies and run the GPU preflight. This is synthetic only;
it does not download training data or launch a pilot. It checks the exact commits,
approved parameter counts, finite gradients, TDA kernel/materialized parity, strict
full-graph compiled forward/backward for Mamba-3, and eager/optimized forward parity plus
CUDA-graph forward/backward for TDA and GDN-2.

```bash
python -m supplementary.robustness.setup_gpu_machine --role mamba3
python -m supplementary.robustness.setup_gpu_machine --role tda
python -m supplementary.robustness.gpu_preflight --approve
```

Then launch the three pilots individually:

```bash
python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/baseline_pilots \
  --include pilot_tda_hybrid.json \
  --log_dir supplementary/robustness/work/logs/baseline_pilots \
  --state_dir supplementary/robustness/work/state/baseline_pilots \
  --ckpt_dir checkpoints/supplementary_robustness --gpu 0 --once

python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/baseline_pilots \
  --include pilot_mamba3_native.json \
  --log_dir supplementary/robustness/work/logs/baseline_pilots \
  --state_dir supplementary/robustness/work/state/baseline_pilots \
  --ckpt_dir checkpoints/supplementary_robustness --gpu 0 --once

python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/baseline_pilots \
  --include pilot_gdn2_native.json \
  --log_dir supplementary/robustness/work/logs/baseline_pilots \
  --state_dir supplementary/robustness/work/state/baseline_pilots \
  --ckpt_dir checkpoints/supplementary_robustness --gpu 0 --once
```

After every pilot, verify its saved checkpoint:

```bash
python -m external_baselines.verify_checkpoint checkpoints/supplementary_robustness/<run_id>
```

## Record the 10B baseline selection once

After all three pilot logs are complete, run exactly one promotion command on the
coordinator machine. Examples:

```bash
# Promote TDA and select Mamba-3
python -m supplementary.robustness.promote \
  --tda promote --linear mamba3_native \
  --reason "Stable pilots; selected by the prespecified validation/retrieval rule."

# Or promote TDA and select GDN-2
python -m supplementary.robustness.promote \
  --tda promote --linear gdn2_native \
  --reason "Stable pilots; selected by the prespecified validation/retrieval rule."
```

Use `--tda omit` only if the TDA pilot fails its gate. Promotion writes
`configs/promotion_decision.json` and enables only the selected scaled configs. Commit
and push the resulting decision plus `configs/baseline_scaled/*.json`; all 10B machines
must check out that exact dispatch commit. Do not rerun promotion independently on the
worker machines.

## Six-machine 10B dispatch

Each machine needs roughly 25 GB free for its 99 training shards plus checkpoint and log
headroom. Run only its assigned setup, preflight (where shown), and worker command.
The first compiled call can take several minutes; that one-time compile is not a stalled
training step.

### GPU 1 — seed 1 Polar

```bash
python -m supplementary.robustness.setup_gpu_machine --role replication
python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/replication \
  --include repl_seed1_polar.json \
  --log_dir supplementary/robustness/work/logs/replication \
  --state_dir supplementary/robustness/work/state/replication \
  --ckpt_dir checkpoints/supplementary_robustness --gpu 0 --once
```

### GPU 2 — seed 1 NoPE

```bash
python -m supplementary.robustness.setup_gpu_machine --role replication
python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/replication \
  --include repl_seed1_nope.json \
  --log_dir supplementary/robustness/work/logs/replication \
  --state_dir supplementary/robustness/work/state/replication \
  --ckpt_dir checkpoints/supplementary_robustness --gpu 0 --once
```

### GPU 3 — seed 2 Polar

```bash
python -m supplementary.robustness.setup_gpu_machine --role replication
python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/replication \
  --include repl_seed2_polar.json \
  --log_dir supplementary/robustness/work/logs/replication \
  --state_dir supplementary/robustness/work/state/replication \
  --ckpt_dir checkpoints/supplementary_robustness --gpu 0 --once
```

### GPU 4 — seed 2 NoPE

```bash
python -m supplementary.robustness.setup_gpu_machine --role replication
python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/replication \
  --include repl_seed2_nope.json \
  --log_dir supplementary/robustness/work/logs/replication \
  --state_dir supplementary/robustness/work/state/replication \
  --ckpt_dir checkpoints/supplementary_robustness --gpu 0 --once
```

### GPU 5 — TDA, only when promoted

```bash
python -m supplementary.robustness.setup_gpu_machine --role tda
python -m supplementary.robustness.gpu_preflight \
  --configs supplementary/robustness/configs/baseline_scaled \
  --include scaled_tda_hybrid.json
python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/baseline_scaled \
  --include scaled_tda_hybrid.json \
  --log_dir supplementary/robustness/work/logs/baseline_scaled \
  --state_dir supplementary/robustness/work/state/baseline_scaled \
  --ckpt_dir checkpoints/supplementary_robustness --gpu 0 --once
```

### GPU 6 — selected Mamba-3 or GDN-2

Mamba-3 command:

```bash
python -m supplementary.robustness.setup_gpu_machine --role mamba3
python -m supplementary.robustness.gpu_preflight \
  --configs supplementary/robustness/configs/baseline_scaled \
  --include scaled_mamba3_native.json
python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/baseline_scaled \
  --include scaled_mamba3_native.json \
  --log_dir supplementary/robustness/work/logs/baseline_scaled \
  --state_dir supplementary/robustness/work/state/baseline_scaled \
  --ckpt_dir checkpoints/supplementary_robustness --gpu 0 --once
```

GDN-2 command:

```bash
python -m supplementary.robustness.setup_gpu_machine --role gdn2
python -m supplementary.robustness.gpu_preflight \
  --configs supplementary/robustness/configs/baseline_scaled \
  --include scaled_gdn2_native.json
python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/baseline_scaled \
  --include scaled_gdn2_native.json \
  --log_dir supplementary/robustness/work/logs/baseline_scaled \
  --state_dir supplementary/robustness/work/state/baseline_scaled \
  --ckpt_dir checkpoints/supplementary_robustness --gpu 0 --once
```

Run only the selected GPU-6 block. A disabled scaled config is skipped; if that happens,
the machine is not on the coordinator's dispatch commit.

## Return and summarize artifacts

From each machine, return its `.log`, `.done` marker, exact source config, checkpoint
SHA-256, repository commit, and preflight output. Strictly reload external checkpoints:

```bash
python -m external_baselines.verify_checkpoint checkpoints/supplementary_robustness/<run_id>
```

After all artifacts are copied into the common paths, run:

```bash
python -m supplementary.robustness.summarize
```

If a process dies, inspect its `.running` JSON and verify that its host/PID no longer
exists before using `run_worker --reset_running`. Preserve failed logs before
`--reset_failed`.
