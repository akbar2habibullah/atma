# Raven Baseline Runs

This package keeps Raven-specific code outside the Atma `attn_type` implementation path.
The runs are architecture baselines that emit the normal `ABLATION_*_JSON` log blocks, so
`ablation.parse_logs` and `scaled_ablation.parse_logs` can consume them.

## 1B Bridge

```bash
python -m raven_baseline.generate_configs --out raven_baseline/configs
FLA_CUSTOM_OP=1 python -m raven_baseline.run_worker \
  --config_dir raven_baseline/configs --log_dir raven_baseline/logs --gpu 0
```

This writes three configs:

- `raven_native`: 16 Raven routed-memory layers.
- `atma_raven`: 16 Atma-style layers with 12 LFM2 conv layers and 4 Raven mixers.
- `atma_raven_titans`: the same 3:1 transplant with Titans memory added to Raven mixer layers.

## Scaled Promotion

After choosing the best bridge run, generate only that promoted variant:

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

