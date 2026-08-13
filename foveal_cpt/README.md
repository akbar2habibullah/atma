# Foveal checkpoint-CPT pilot

This directory is an isolated experiment. It does not change ATMA's shipping training or
inference paths. It adds a learned 16D MQA page index to an existing ATMA checkpoint, calibrates
the index against the checkpoint's full attention, and then runs local-plus-remote sparse
continual pretraining.

The default configuration targets the Polar L40S checkpoint:

- 32K sequences and a 512-token exact sliding window;
- 64-token query/KV blocks;
- capped top-P routing, annealed from `p=0.98, K=8:64` to `p=0.95, K=0:32`;
- four full-rank teacher anchor queries per sequence and attention layer;
- 1B CPT tokens at the original 524,288-token global batch.

CUDA NoPE/RoPE execution uses PyTorch FlexAttention `BlockMask`. Polar uses the trainable selected-
page path in `kernel/polar_triton.py`, which fuses its direction and participation-ratio statistics
in one sparse pass. Remote pages are full blocks and local pages use an exact causal/window mask;
no `32768 x 32768` mask is created. CPU execution is a correctness reference limited to 4,096
tokens. The pilot pins 64x64 tiles to stay aligned with the sparse layout and reduce shared-memory
pressure on the L40S; treat the first CUDA smoke as a mandatory kernel-compatibility gate.

## Requirements

Use the same environment as ATMA training, with CUDA, Triton, FLA, causal-conv1d, and a recent
PyTorch build that provides `torch.nn.attention.flex_attention.BlockMask.from_kv_blocks` and GQA.
The checkpoint downloader also needs `huggingface_hub`.

Run the portable correctness suite before using a GPU budget:

```bash
python -m foveal_cpt.tests
```

The suite checks causal page selection, full-support parity for NoPE/RoPE/Polar, and indexer
gradients. On CUDA it also compares sparse Polar outputs and all trainable gradients against the
materialized oracle. Run the smoke stages below as well; CUDA kernel compilation cannot be
validated by the portable reference alone.

## 1. Prepare data

`pilot.json` defaults to the existing binary shards:

```text
finewebedu10B/finewebedu_train_*.bin
```

For the scientific run, use document-coherent 32K packs or add explicit document reset masks.
The legacy flat shards are sufficient for kernel and optimizer smoke tests, but arbitrary
cross-document slices can make the index learn recency/boundary artifacts.

## 2. Calibrate the new index

Calibration freezes every checkpoint parameter and trains only the new query/key index
projections. The original dense attention supplies page-mass targets at 4K by default.

```bash
python -m foveal_cpt.calibrate --config foveal_cpt/pilot.json --smoke-steps 2
python -m foveal_cpt.calibrate --config foveal_cpt/pilot.json
```

Outputs are written under `foveal_cpt/output/polar/calibration/`. A saved calibration is fully
restartable:

```bash
python -m foveal_cpt.calibrate \
  --config foveal_cpt/pilot.json \
  --resume foveal_cpt/output/polar/calibration/calibration-step-000250.pt
```

## 3. Smoke-test and launch CPT

Pass the completed calibration checkpoint. The trainer refuses an uncalibrated index unless
`--allow-uncalibrated-index` is explicit.

```bash
python -m foveal_cpt.train \
  --config foveal_cpt/pilot.json \
  --index-checkpoint foveal_cpt/output/polar/calibration/calibration-step-000306.pt \
  --smoke-steps 2
```

Inspect peak memory, index loss, teacher top-K recall/mass, cap rate, mean remote pages, and
tokens/s. Then remove `--smoke-steps`:

```bash
python -m foveal_cpt.train \
  --config foveal_cpt/pilot.json \
  --index-checkpoint foveal_cpt/output/polar/calibration/calibration-step-000306.pt
```

Resume CPT with:

```bash
python -m foveal_cpt.train \
  --config foveal_cpt/pilot.json \
  --resume foveal_cpt/output/polar/cpt/cpt-step-000250.pt
```

Checkpoints contain model, optimizer, RNG, token-shard cursor, step, and token count. The Hugging
Face source artifacts do not contain optimizer state, so the first CPT run starts fresh optimizer
states by design.

## 4. Evaluate the routing curve

The lightweight evaluator reports validation loss and routing statistics for several top-P and
fixed-K budgets:

```bash
python -m foveal_cpt.evaluate \
  --config foveal_cpt/pilot.json \
  --checkpoint foveal_cpt/output/polar/cpt/cpt-step-001908.pt \
  --lengths 2048 8192 32768 \
  --top-p 0.90 0.95 0.98 \
  --fixed-k 8 16 32 64 \
  --output foveal_cpt/output/polar/routing-eval.json
```

This is a run gate, not the complete research evaluation. Promote a checkpoint only after running
the repository's coherent-document loss, needle/RULER, and Polar diagnostic suites against the
untouched full-attention checkpoint and the matched `SWA=512 + Titans` CPT control.

## Other attention cores

Copy `pilot.json` and change `checkpoint`/`output_dir` to the RoPE or NoPE repository. Calibrate
each index independently. Start those conditions with a 100M-token screen by setting
`train_tokens` to `100000000` and shorten the handoff boundaries proportionally. NoPE is a
diagnostic because its source checkpoint already has a known long-context instability.
