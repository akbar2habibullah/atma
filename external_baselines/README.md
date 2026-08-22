# External baseline adapters

This package supplies Atma-evaluation-compatible wrappers for the supplementary TDA,
Mamba-3, and GDN-2 runs. It does not vendor third-party kernels. The experiment pins
their source commits in [`supplementary/robustness/dependencies.json`](../supplementary/robustness/dependencies.json).

- `tda_hybrid` keeps Atma's 12 local LFM2 layers and replaces the four global layers
  with TDA. This is the Stage-II mechanism baseline.
- `mamba3_native` and `gdn2_native` use the upstream mixer in all 16 blocks, matching
  the role of `raven_native` as an external model-family baseline.

The wrappers deliberately share Raven's AdamW training harness and the normal
`ABLATION_*_JSON` log contract. They require CUDA and must pass the GPU preflight in the
supplementary runbook before any 1B-token pilot is launched.

## Preparation status on the pilot machine

The pilot machine is preparation-only until an operator explicitly launches the 1B
worker. No 10B external baseline should be run on this machine.

Preparation record from 2026-08-22 on one NVIDIA L40S with PyTorch 2.13.0+cu130:

| Model | Resolved parameters | Difference from 378.2M target |
|---|---:|---:|
| GDN-2 | 366,322,864 | 3.14% |
| Mamba-3 | 368,108,416 | 2.67% |
| TDA hybrid | 388,641,924 | 2.76% |

The atomic GPU preflight passed for all three models. The complete ten-cell, three-step
smoke matrix also passed, including structured 2K/4K evaluation. All three external
smoke checkpoints passed strict reload and a finite forward pass. No 1B external pilot
and no 10B run was launched as part of this preparation record.

The pinned dependency commits are:

| Dependency | Commit |
|---|---|
| Flash Linear Attention | `e47d5d20aeb5989b58a3738b872e7c288a9fb75f` |
| Mamba | `e9594ce1c732d97440f0332fdc43170a2294dbfa` |
| TDA | `cd8ddc9d5b43a1dcf86f9cfda302edb5cc108da2` |

Run `python -m supplementary.robustness.gpu_preflight --approve` after generating and
validating `supplementary/robustness/configs`. Approval is written directly to the pilot
and matching scaled configs in this single operational config tree.

The experiment uses Mamba-3 SISO (`mamba3_mimo=false`). On Python 3.12, optional
TileLang 0.1.8 may fail while importing its bundled TVM even though MIMO is disabled. If
that occurs, uninstall `tilelang` after installing the pinned Mamba checkout and verify:

```bash
python -m pip uninstall -y tilelang
PYTHONPATH="$PWD/third_party/flash-linear-attention:$PWD/third_party/mamba${PYTHONPATH:+:$PYTHONPATH}" \
python - <<'PY'
from fla.layers.mamba3 import is_fast_path_available, mamba3_mimo_combined
assert is_fast_path_available, "Mamba-3 SISO kernel is unavailable"
assert mamba3_mimo_combined is None, "this protocol does not use the optional MIMO path"
print("Mamba-3 SISO import OK")
PY
```

Do not manually set `parameter_count_approved`, and do not regenerate configs with
`generate_configs --clean` after approval or training has started.

## Handoff to the 10B machine

Wave 1 produces three separate 1B screening runs. After their structured evaluations
finish, copy all three pilot logs to the machine where the promotion decision will be
recorded:

```text
supplementary/robustness/work/logs/baseline_pilots/
  pilot_tda_hybrid.log
  pilot_mamba3_native.log
  pilot_gdn2_native.log
```

Archive and transfer the following together:

- the exact repository commit;
- the three pilot logs and their SHA-256 hashes;
- `supplementary/robustness/dependencies.json`;
- the approved pilot and scaled configs;
- `promotion_decision.json` after promotion;
- dependency commit output and GPU/preflight output.

On the 10B machine, install the same pinned checkouts, generate its config tree, and run
the GPU preflight on that machine. Copy the pilot logs into the path above, then
record one prespecified decision. TDA may be promoted or omitted; exactly one of Mamba-3
and GDN-2 must be selected.

TDA plus Mamba-3:

```bash
python -m supplementary.robustness.promote \
  --tda promote \
  --linear mamba3_native \
  --reason "Stable pilots; selected by the prespecified validation/retrieval rule."
```

TDA plus GDN-2:

```bash
python -m supplementary.robustness.promote \
  --tda promote \
  --linear gdn2_native \
  --reason "Stable pilots; selected by the prespecified validation/retrieval rule."
```

Use `--tda omit` instead when the TDA pilot fails its gate. Promotion updates only
`supplementary/robustness/configs/baseline_scaled`.

## Individual 10B commands

Run only the configs enabled by the recorded promotion decision. These commands are
intentionally one model per worker invocation.

TDA 10B, when promoted:

```bash
python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/baseline_scaled \
  --include scaled_tda_hybrid.json \
  --log_dir supplementary/robustness/work/logs/baseline_scaled \
  --state_dir supplementary/robustness/work/state/baseline_scaled \
  --ckpt_dir checkpoints/supplementary_robustness \
  --gpu 0 --once
```

Mamba-3 10B, when selected:

```bash
python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/baseline_scaled \
  --include scaled_mamba3_native.json \
  --log_dir supplementary/robustness/work/logs/baseline_scaled \
  --state_dir supplementary/robustness/work/state/baseline_scaled \
  --ckpt_dir checkpoints/supplementary_robustness \
  --gpu 0 --once
```

GDN-2 10B, when selected:

```bash
python -m supplementary.robustness.run_worker \
  --config_dir supplementary/robustness/configs/baseline_scaled \
  --include scaled_gdn2_native.json \
  --log_dir supplementary/robustness/work/logs/baseline_scaled \
  --state_dir supplementary/robustness/work/state/baseline_scaled \
  --ckpt_dir checkpoints/supplementary_robustness \
  --gpu 0 --once
```

After each run, strictly reload its checkpoint:

```bash
python -m external_baselines.verify_checkpoint \
  checkpoints/supplementary_robustness/<run_id>
```

Finally return the 10B logs, state markers, resolved configs, checkpoint hashes,
and dependency/preflight records to the primary experiment archive before running
`python -m supplementary.robustness.summarize`.
