# Foveal sparse-attention CPT sweep

This directory is an isolated 12-run adaptation experiment over the three existing ATMA 10B
checkpoints. Every cell trains for **1B tokens** at 32K context and a 524,288-token global batch
(1,908 optimizer steps). There is no 100M-token screening stage.

| attention core | local SWA-512 | LM index output | index KL | LM output + KL |
| --- | --- | --- | --- | --- |
| Polar | `polar-local.json` | `polar-lm_output.json` | `polar-kl.json` | `polar-lm_output_kl.json` |
| RoPE | `rope-local.json` | `rope-lm_output.json` | `rope-kl.json` | `rope-lm_output_kl.json` |
| NoPE | `nope-local.json` | `nope-lm_output.json` | `nope-kl.json` | `nope-lm_output_kl.json` |

All files live in `foveal_cpt/configs/` and use isolated output directories.

## Dataset preparation

The sweep uses [`kjj0/finewebedu10B-gpt2`](https://huggingface.co/datasets/kjj0/finewebedu10B-gpt2).
Before allocating a model, the launcher downloads and verifies enough sequential shards under
`finewebedu10B/` to satisfy all 1,908 steps, plus the validation shard. With 100M-token source
shards and the 524,288-token batch, this requires train shards `000001` through `000011`: ten
shards are slightly short because batches never cross shard boundaries. All 12 cells reuse these
same local files.

To prepare data without starting training:

```bash
python -m foveal_cpt.prepare_data \
  --config foveal_cpt/configs/polar-local.json
```

Existing files are reused. Each shard's header, token width, declared token count, and exact file
size are checked before training. `--dry-run` never downloads data.

## What the four cells test

- `local`: exact causal SWA-512 with no remote pages and no trained index parameters.
- `lm_output`: the 16D MQA q/k stream selects sparse pages. A matching 16D value stream performs
  a differentiable causal page read and is projected into the residual stream, so ordinary LM
  loss trains index q/k/v/output parameters.
- `kl`: the 16D q/k stream selects pages and receives an auxiliary page-mass KL loss. A short
  frozen-backbone calibration initializes it. During 32K CPT, KL only gathers the selected local
  and remote support; it does not run full dense attention.
- `lm_output_kl`: combines the two indexer gradient paths.

The index projections consume `x.detach()`: index learning does not reshape the checkpoint hidden
states through the auxiliary branch. The backbone still receives ordinary LM gradients through
the sparse attention path. Hard page indices are detached because discrete top-P selection is not
differentiable; LM-output and KL provide the continuous indexer gradients.

CUDA NoPE/RoPE execution uses PyTorch FlexAttention. Polar calls the selected-page Triton path in
`kernel/polar_triton.py`, including its custom backward. No eager Polar reduction is used for the
32K CUDA path. CPU execution is only a correctness reference and is capped at 4,096 tokens.

## Run the complete sweep

Use the ATMA CUDA training environment, then run:

```bash
python -m foveal_cpt.tests
python -m foveal_cpt.sweep --dry-run
python -m foveal_cpt.sweep --smoke-steps 2
python -m foveal_cpt.sweep
```

The default invocation covers all 12 cells. The two KL cells for a given attention core share one
20M-token calibration produced from that core's checkpoint. Local and LM-output-only cells start
directly from the source checkpoint. Calibration is not counted in any cell's 1B CPT budget.

The runner reads each output's `latest.json`, resumes incomplete work, and skips completed work.
Smoke artifacts are isolated under each cell's `smoke/` subdirectory and are never resumed into
the full scientific run.
To distribute the matrix across three GPUs or machines, launch one core per process, for example:

```bash
python -m foveal_cpt.sweep --cores polar --device cuda:0
python -m foveal_cpt.sweep --cores rope  --device cuda:1
python -m foveal_cpt.sweep --cores nope  --device cuda:2
```

For one cell, invoke the trainer directly:

```bash
python -m foveal_cpt.train \
  --config foveal_cpt/configs/polar-lm_output.json \
  --device cuda
```

KL cells require the matching calibration checkpoint when they are not launched by the sweep
runner. Checkpoints contain model, optimizer, RNG, data cursor, step, token count, stage, and the
resolved experiment config.

## Run gates

The first two-step smoke is mandatory for validating CUDA compilation, memory, backward, and
actual step time. At the existing dense baseline's 9 seconds/step, 12 x 1,908 steps is 57.2 L40S
GPU-hours; sparse routing, KL, and kernel compile overhead must be measured rather than assumed.
After CPT, evaluate loss and routing curves with `python -m foveal_cpt.evaluate`, then run the
repository's coherent-document, needle/RULER, and Polar diagnostics against the untouched source
checkpoint.
