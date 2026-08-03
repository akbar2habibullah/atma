# Atma Scaled Ablation

Final 10B-token ablation for the selected Atma recipe.

| Axis | Value |
|---|---|
| `reg_mode` | `baseline` |
| `distractor` | off |
| `memory` | on |
| `window` | off |
| `attn_type` | `rope`, `nope`, `polar` |

Wall Attention is excluded from the fair scaled grid because it was incompatible with the
standardized Atma hybrid + native Muon protocol. Raven is tracked separately through
[raven_baseline](../raven_baseline/) as the stronger outsider baseline, with the caveat that it uses
a different architecture/protocol and defaults to Raven's AdamW recipe.

Defaults use all 99 `kjj0/finewebedu10B-gpt2` train chunks and evaluate clean PPL, junk PPL, and
needle retrieval at:

`2048, 4096, 8192, 16384, 32768, 65536, 131072`

Evaluation catches CUDA OOM per document/trial and records `None` for lengths that could not
complete.

## Generate Configs

```bash
python -m scaled_ablation.generate_configs
# -> scaled_ablation/configs/*.json (3 configs)
```

Smoke test:

```bash
python -m scaled_ablation.generate_configs \
  --out scaled_ablation/smoke --num_chunks 1 --val_tokens 524288 --max_steps 3 \
  --num_eval_docs 2 --num_needle_trials 2

FLA_CUSTOM_OP=1 python -m scaled_ablation.run_worker \
  --config_dir scaled_ablation/smoke --log_dir scaled_ablation/smoke_logs --gpu 0 --once
```

Wall can still be generated explicitly for reproducing the failed/incompatible diagnostic:

```bash
python -m scaled_ablation.generate_configs --attn_types wall --out scaled_ablation/wall_diagnostic
```

## Run

Single shared config directory, one worker per GPU:

```bash
FLA_CUSTOM_OP=1 python -m scaled_ablation.run_worker \
  --config_dir scaled_ablation/configs --log_dir scaled_ablation/logs --ckpt_dir checkpoints --gpu 0
```

The scaled trainer saves checkpoints by default:

```text
checkpoints/<run_id>/
  weights.pt
  config.json
  run_config.json
  tokenizer.json
```

Use `--no_save_ckpt` only for smoke/debug runs where the checkpoint is disposable.

## Raven Outsider Baseline

After choosing the Raven bridge variant to promote, generate and run only that variant:

```bash
python -m raven_baseline.generate_configs --scaled \
  --arch_types atma_raven_titans \
  --out raven_baseline/scaled_configs

FLA_CUSTOM_OP=1 python -m raven_baseline.run_worker \
  --config_dir raven_baseline/scaled_configs \
  --log_dir raven_baseline/scaled_logs \
  --ckpt_dir checkpoints \
  --gpu 0
```

Build a combined scaled dashboard:

```bash
python -m scaled_ablation.build_dashboard \
  --log_dir scaled_ablation/logs raven_baseline/scaled_logs \
  --out pages/scaled_dashboard.html
```

## Hugging Face Upload

Upload after each successful worker run:

```bash
python -m scaled_ablation.run_worker \
  --config_dir scaled_ablation/configs --log_dir scaled_ablation/logs --ckpt_dir checkpoints \
  --gpu 0 --push_to_hub --hf_repo_prefix your-org/atma-10b --hf_private
```

This creates repos named like:

```text
your-org/atma-10b-polar__reg-baseline__distr-0__mem-1__win-0
```

`HF_TOKEN` must be available in the environment for upload.

## Cross-evaluate Hugging Face checkpoints

Use the checkpoint evaluator to compare saved Atma models under one fixed GPU/PyTorch runtime.
It downloads each repository, persists one shared FinePDF document manifest, and records successful
and OOM evaluation counts.

The quickest test of the L4/PyTorch 2.12 anomaly uses the shared FineWeb-Edu stream only:

```bash
FLA_CUSTOM_OP=1 python -m scaled_ablation.eval_hf_checkpoints \
  --metrics junk \
  --sdpa-backend flash \
  --output scaled_ablation/cross_eval_l4_torch212_junk.json
```

Run all original metrics through 131K with:

```bash
FLA_CUSTOM_OP=1 python -m scaled_ablation.eval_hf_checkpoints \
  --metrics junk clean needle \
  --sdpa-backend flash \
  --output scaled_ablation/cross_eval_l4_torch212.json
```

The default model list contains the L4 NoPE, L40S NoPE, and L40S Polar checkpoints. Override it
with `--models <repo-id> ...`. Use the same `--doc-manifest` when repeating the comparison on a
different runtime.

### Checkpoint stress analysis

The `stress` metric performs a post-training load-to-failure sweep on the same nested document
prefixes. It records streaming moments for every block's residual input/output, mixer, MLP,
projected attention, Polar count channel, and Titans memory contribution. Polar checkpoints also
report per-head `n_eff`, magnitude, and null-sink mass; memory checkpoints report per-head
retention (`gamma`), write strength (`beta`), and attention/memory output-gate saturation. All
activation summaries include non-finite rates. The JSON summary ranks the first length where a
component leaves the configured operating envelope relative to the shortest evaluated length.

```bash
FLA_CUSTOM_OP=1 python -m scaled_ablation.eval_hf_checkpoints \
  --metrics stress \
  --lengths 2048 4096 8192 16384 32768 65536 131072 \
  --stress-num-docs 8 \
  --output scaled_ablation/checkpoint_stress.json
```

Randomized finite-difference modal analysis is opt-in because it repeats every block for each
perturbation direction. Its local secant gain is a sampled lower bound, not an exact maximum
Jacobian singular value:

```bash
FLA_CUSTOM_OP=1 python -m scaled_ablation.eval_hf_checkpoints \
  --metrics stress \
  --lengths 2048 4096 8192 16384 32768 65536 131072 \
  --stress-modal-lengths 2048 8192 \
  --stress-modal-docs 1 \
  --stress-modal-samples 2 \
  --output scaled_ablation/checkpoint_stress_modal.json
```

## Parse And Dashboard

```bash
python -m scaled_ablation.parse_logs \
  --log_dir scaled_ablation/logs --out scaled_ablation/results.json

python -m scaled_ablation.build_dashboard \
  --log_dir scaled_ablation/logs raven_baseline/scaled_logs \
  --out pages/scaled_dashboard.html
```

## Downstream Benchmarks

The benchmark harness currently serves polar checkpoints through `inference.LLM`.

```bash
python -m benchmarks.run --benchmark retrieval --model checkpoints/polar__reg-baseline__distr-0__mem-1__win-0 \
  --tasks passkey niah --lengths 1k 2k 4k 8k 16k 32k 64k 128k \
  --depths 0 0.25 0.5 0.75 1 --samples 50 --max_num_seqs 16 --strict
```

For BABILong, use a fine-tuned checkpoint rather than the base scaled-ablation checkpoint.
