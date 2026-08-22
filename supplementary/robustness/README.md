# Robustness and modern-baseline supplement

This directory owns the third supplementary experiment for the ICLR 2027 revision. It
contains the plan and resolved configs, while reusing the existing Stage-I,
Stage-II, evaluation, and structured-log implementations.

The 68B-token ceiling consists of:

| Group | Runs | Ceiling |
|---|---:|---:|
| Paired Polar/NoPE replications | 4 × 10B | 40B |
| Polar full control + four component variants | 5 × 1B | 5B |
| TDA, Mamba-3, and GDN-2 pilots | 3 × 1B | 3B |
| Promoted TDA | 1 × 10B | 10B |
| One promoted Mamba-3/GDN-2 model | 1 × 10B | 10B |

`manifest.json` and `eval_manifest.json` are the protocol source of truth. All workers
read configs directly from `configs/`. GPU preflight records parameter approvals there,
and promotion enables selected scaled configs there. Claims, logs, smoke outputs, and the
promotion decision remain under ignored paths in `work/`.

## 1. CPU preparation

Run these commands from the repository root:

```bash
python -m supplementary.robustness.generate_configs --clean
python -m supplementary.robustness.validate_plan
```

Do not use `generate_configs --clean` after approval or a run has started. It resets
parameter approvals and scaled promotion flags in the operational config tree.

## 2. Install pinned GPU dependencies

On the GPU instance, create the source checkouts exactly as pinned in
`dependencies.json`:

```bash
mkdir -p third_party
git clone https://github.com/fla-org/flash-linear-attention.git third_party/flash-linear-attention
git -C third_party/flash-linear-attention checkout e47d5d20aeb5989b58a3738b872e7c288a9fb75f

git clone https://github.com/state-spaces/mamba.git third_party/mamba
git -C third_party/mamba checkout e9594ce1c732d97440f0332fdc43170a2294dbfa

git clone https://github.com/snap-research/TDA.git third_party/TDA
git -C third_party/TDA checkout cd8ddc9d5b43a1dcf86f9cfda302edb5cc108da2
```

Install FLA and the Mamba checkout in the instance's CUDA/PyTorch environment. Follow
their pinned installation instructions rather than installing an unpinned PyPI latest.
The TDA checkout is imported directly and remains subject to its non-commercial,
research-only license.

## 3. Mandatory GPU-instance Codex/debug pass

This repository has no local CUDA runtime, so the external adapters are intentionally
blocked until a GPU preflight succeeds:

```bash
python -m supplementary.robustness.gpu_preflight --approve
```

The preflight verifies dependency commits, constructs each model, checks the parameter
count against the 378.2M target, and runs a finite forward/backward pass. Approval is
copied to pilot and scaled configs only when the count is within 5%.

The external shapes are provisional and may need adjustment on the GPU instance. If a
model misses the tolerance, change its pilot config and rerun the preflight. On
approval, the tool propagates shape fields to the matching scaled config. Backport
the resolved shape into `generate_configs.py` before publishing results. Never bypass the
approval by manually setting the boolean.

The GPU-instance Codex pass must additionally check:

1. **TDA:** compare the official Triton forward and gradients with a small materialized
   causal PyTorch reference for batch sizes 1 and 2, bf16 and fp32, and lengths crossing
   a 64-token tile boundary. Confirm `lambda_param` receives a finite gradient. The
   upstream kernel treats beta as a fixed hyperparameter, so beta is a buffer here.
2. **Mamba-3:** confirm `mamba3_siso_combined` is available, run at 2K without fallback,
   and compare full-sequence output with recurrent decoding on a short sequence. State
   must reset between independent documents.
3. **GDN-2:** compare chunk and fused-recurrent outputs/final states on a short sequence,
   then run backward through the chunk training path. State must reset between documents.
4. **All three:** record peak memory and step time at `mbs=4`; reduce microbatch size only
   through a documented config revision. Confirm checkpoint save/reload before any 1B run.
5. **Reproducibility:** inspect the first log block and confirm `init_seed`, `data_seed`,
   `eval_seed`, dependency commits, parameter count, and exact config are present.

The automated preflight includes the TDA materialized forward/backward parity case. The
remaining recurrent-state checks require inspection on the actual FLA/Mamba build.

Keep `compile_model=false` for the first pilots. TDA's upstream wrapper reads scalar beta
on the host, and the new FLA paths need eager correctness confirmation before any compile
optimization is attempted.

## 4. Three-step smoke matrix

After approval:

```bash
python -m supplementary.robustness.make_smoke
python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/work/smoke/configs \
  --log_dir supplementary/robustness/work/smoke/logs \
  --state_dir supplementary/robustness/work/smoke/state \
  --ckpt_dir checkpoints/supplementary_robustness_smoke --gpu 0
```

This covers Polar/NoPE replication, every Polar component switch, and all three new
baseline paths. Do not begin pilots until every smoke config completes. Verify each saved
external checkpoint with:

```bash
python -m external_baselines.verify_checkpoint checkpoints/supplementary_robustness_smoke/<run_id>
```

## 5. Wave 1: 1B attribution and screening

Run the component cells and baseline pilots as separate worker pools:

```bash
python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/polar_components \
  --log_dir supplementary/robustness/work/logs/polar_components \
  --state_dir supplementary/robustness/work/state/polar_components --gpu 0

python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/baseline_pilots \
  --log_dir supplementary/robustness/work/logs/baseline_pilots \
  --state_dir supplementary/robustness/work/state/baseline_pilots \
  --ckpt_dir checkpoints/supplementary_robustness --gpu 0
```

The five Polar cells use the same initialization and data stream. They are one full
control plus four one-factor interventions: direction only, constant magnitude, fixed
null, and fixed temperature.

The FineWeb-Edu loader consumes sorted shards as one contiguous stream and does not
shuffle, so `data_seed` is a pairing/provenance label rather than a shuffle control.
Fused kernels run with `deterministic_algorithms=false` for the published throughput
path; these are statistical replications, not promises of bitwise replay.

## 6. Record promotion before scaled baselines

Inspect the fixed pilot metrics, then record the decision:

```bash
python -m supplementary.robustness.promote \
  --tda promote \
  --linear mamba3_native \
  --reason "Stable pilot; selected by the prespecified validation/retrieval rule."
```

Use `--tda omit` if TDA fails the gate. `--force` exists only for repairing imported
historical logs; it must not be used to promote an incomplete fresh pilot.

## 7. Wave 2: paired replications

On the Wave 2 machine, schedule one model at a time and finish both models from a seed
pair before starting the next seed. Run the following four commands from the repository
root, in order.

Seed 1 Polar:

```bash
python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/replication \
  --include repl_seed1_polar.json \
  --log_dir supplementary/robustness/work/logs/replication \
  --state_dir supplementary/robustness/work/state/replication \
  --ckpt_dir checkpoints/supplementary_robustness \
  --gpu 0 --once
```

Seed 1 NoPE:

```bash
python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/replication \
  --include repl_seed1_nope.json \
  --log_dir supplementary/robustness/work/logs/replication \
  --state_dir supplementary/robustness/work/state/replication \
  --ckpt_dir checkpoints/supplementary_robustness \
  --gpu 0 --once
```

Seed 2 Polar:

```bash
python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/replication \
  --include repl_seed2_polar.json \
  --log_dir supplementary/robustness/work/logs/replication \
  --state_dir supplementary/robustness/work/state/replication \
  --ckpt_dir checkpoints/supplementary_robustness \
  --gpu 0 --once
```

Seed 2 NoPE:

```bash
python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/replication \
  --include repl_seed2_nope.json \
  --log_dir supplementary/robustness/work/logs/replication \
  --state_dir supplementary/robustness/work/state/replication \
  --ckpt_dir checkpoints/supplementary_robustness \
  --gpu 0 --once
```

All four commands intentionally share the same log and state directories. If Wave 2 is
run on another host, copy its replication logs, state markers, source configs, and
checkpoint hashes back to the primary experiment archive before summarizing.

## 8. Wave 3: promoted 10B baselines

Only the models enabled by `promotion_decision.json` are runnable:

```bash
python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/baseline_scaled \
  --log_dir supplementary/robustness/work/logs/baseline_scaled \
  --state_dir supplementary/robustness/work/state/baseline_scaled \
  --ckpt_dir checkpoints/supplementary_robustness --gpu 0
```

Finally collect the structured logs:

```bash
python -m supplementary.robustness.summarize
```

The aggregate is a view over the logs, not a replacement for them. Archive the resolved
configs, promotion decision, logs, dependency commit output, and checkpoint hashes
together when the experiment is frozen.

If a worker process dies, inspect the JSON in its `.running` marker and confirm that the
recorded host/PID is no longer alive before using `run_worker --reset_running`. Failed
markers require `--reset_failed`; preserve the error log before retrying.
