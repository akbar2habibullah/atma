# Atma Scaled Ablation

Final 10B-token ablation for the selected recipe:

| Axis | Value |
|---|---|
| `reg_mode` | `baseline` |
| `distractor` | off |
| `memory` | on |
| `window` | off |
| `attn_type` | `rope`, `nope`, `polar`, `wall` |

Defaults use all 99 `kjj0/finewebedu10B-gpt2` train chunks and evaluate clean PPL,
junk PPL, and needle retrieval at:

`2048, 4096, 8192, 16384, 32768, 65536, 131072`

Evaluation catches CUDA OOM per document/trial and records `None` for lengths that could
not complete.

## Generate Configs

```bash
python -m scaled_ablation.generate_configs
# -> scaled_ablation/configs/*.json (4 configs)
```

Smoke test:

```bash
python -m scaled_ablation.generate_configs \
  --out scaled_ablation/smoke --num_chunks 1 --val_tokens 524288 --max_steps 3 \
  --num_eval_docs 2 --num_needle_trials 2

FLA_CUSTOM_OP=1 ATMA_WALL_CUSTOM_OP=1 python -m scaled_ablation.run_worker \
  --config_dir scaled_ablation/smoke --log_dir scaled_ablation/smoke_logs --gpu 0 --once
```

## Run

Single shared config directory, one worker per GPU:

```bash
FLA_CUSTOM_OP=1 ATMA_WALL_CUSTOM_OP=1 python -m scaled_ablation.run_worker \
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

## Parse And Dashboard

```bash
python -m scaled_ablation.parse_logs \
  --log_dir scaled_ablation/logs --out scaled_ablation/results.json

python -m scaled_ablation.build_dashboard \
  --log_dir scaled_ablation/logs --out pages/scaled_dashboard.html
```

## Downstream Benchmarks

The benchmark harness currently serves polar checkpoints through `inference.LLM`.

```bash
python -m benchmarks.run --benchmark retrieval --model checkpoints/polar__reg-baseline__distr-0__mem-1__win-0 \
  --tasks passkey niah --lengths 1k 2k 4k 8k 16k 32k 64k 128k \
  --depths 0 0.25 0.5 0.75 1 --samples 50 --max_num_seqs 16 --strict
```

For BABILong, use a fine-tuned checkpoint rather than the base scaled-ablation checkpoint.
