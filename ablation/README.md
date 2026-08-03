# Atma Full-Grid Ablation

A complete 120-cell factorial over the architecture recipe space, plus a multi-GPU runner and an
interactive static-HTML dashboard. See [TITANS_MEMORY.md](../docs/TITANS_MEMORY.md) and
[POLAR_ATTENTION.md](../docs/POLAR_ATTENTION.md) for the architecture being ablated.

## Grid

5 x 2 x 2 x 2 x 3 = 120 fair-comparison cells.

| Axis | Values |
|---|---|
| `reg_mode` | baseline, weak, strong, discrete, zipfian |
| `distractor` | off / on; `num_random_keys in {0, seq_len}` |
| `memory` | off / on; Titans gated-delta memory branch |
| `window` | off / on; training sliding window = 1024 |
| `attn_type` | `rope`, `nope`, `polar` |

All cells share: 16 layers (12 conv + 4 attention), hidden 1024, head_dim 128, GQA 1:4,
`seq_len=2048`, and about 1B training tokens. Distractor, memory, and window are wired into all
three fair attention cores, so every cell is distinct. Window is a training knob; every eval runs at
full context.

## Workflow

```bash
# 1) Split the 120 cells into 4 balanced shards of 30.
python -m ablation.generate_configs --out ablation/shards --shards 4

# 2) Run one shard per GPU/host. The worker is resumable.
FLA_CUSTOM_OP=1 python -m ablation.run_worker \
  --config_dir <shardN> --log_dir ablation/logs --gpu 0

# 3) Collect logs and build a dashboard. Raven logs can be included as outsider baselines.
python -m ablation.parse_logs --log_dir ablation/logs --out ablation/results.json
python -m ablation.build_dashboard \
  --log_dir ablation/logs raven_baseline/logs \
  --out pages/dashboard.html
```

## Attempted Contender: Wall Attention

`attn_type="wall"` (Tilde Research [Wall Attention](https://github.com/tilde-research/wall-attention-release))
is implemented and can be generated explicitly, but it is not part of the fair comparison grid.
Under this codebase's standardized hybrid architecture and native Atma Muon optimizer, the Wall run
improved early and then regressed badly. We therefore treat Wall as incompatible with this protocol
rather than as a scored contender.

This is not a general negative claim about Wall. The official Wall recipe uses per-head
Muon/MuonSplit or Aurora-style optimization rather than the native Atma Muon split used here.

```bash
python -m ablation.generate_configs \
  --attn_types wall --mbs 2 --out ablation/wall_diagnostic

FLA_CUSTOM_OP=1 ATMA_WALL_CUSTOM_OP=1 python -m ablation.run_worker \
  --config_dir ablation/wall_diagnostic --log_dir ablation/logs --gpu 0
```

Wall keeps canon, so it was the matched comparison to `nope` and isolates per-channel gating. On
CUDA, training requires the Wall Triton kernel and defaults to `ATMA_WALL_CUSTOM_OP=1`. Keep Wall
results in diagnostic/appendix material only. See [FUTURE.md section 4](../docs/FUTURE.md).

## Outsider Baseline: Raven

Raven replaces Wall as the stronger outsider baseline. It intentionally lives in
[raven_baseline](../raven_baseline/) because it uses a different architecture/protocol and defaults
to Raven's AdamW recipe, not the Atma Muon sweep protocol.

```bash
python -m raven_baseline.generate_configs --out raven_baseline/configs
FLA_CUSTOM_OP=1 python -m raven_baseline.run_worker \
  --config_dir raven_baseline/configs --log_dir raven_baseline/logs --gpu 0
```

To compare Raven against the Atma grid in one dashboard:

```bash
python -m ablation.build_dashboard \
  --log_dir ablation/logs raven_baseline/logs \
  --out pages/dashboard.html
```

## Single-Cell And Smoke Runs

Train a single cell directly:

```bash
FLA_CUSTOM_OP=1 python -m ablation.train \
  --config ablation/configs/polar__reg-zipfian__distr-1__mem-1__win-1.json \
  --log ablation/logs/polar__reg-zipfian__distr-1__mem-1__win-1.log
```

Smoke test:

```bash
python -m ablation.generate_configs \
  --out ablation/smoke --num_chunks 1 --val_tokens 524288 --max_steps 3

FLA_CUSTOM_OP=1 python -m ablation.run_worker \
  --config_dir ablation/smoke --log_dir ablation/smoke_logs --gpu 0
```

## Open-Weight Pretrained Baselines

Run the same metrics against small Hugging Face causal LMs without training:

```bash
python -m ablation.open_baselines --log_dir ablation/open_logs
```

Default model list:

- `google/gemma-3-270m`
- `LiquidAI/LFM2.5-230M-Base`
- `LiquidAI/LFM2.5-350M-Base`
- `Qwen/Qwen2.5-0.5B`
- `Qwen/Qwen3-0.6B-Base`
- `Qwen/Qwen3.5-0.8B-Base`
- `HuggingFaceTB/SmolLM2-360M`
- `ibm-granite/granite-4.0-350m-base`
- `ibm-granite/granite-4.0-h-350m-base`
- `tiiuae/Falcon-H1-0.5B-Base`

`clean_ppl` and `needle` use each model's native tokenizer on coherent
`codelion/finepdfs-100M` documents. `junk_ppl` decodes the GPT-2 FineWeb-Edu validation `.bin`
stream and retokenizes it for the target model. Use `clean_bits_per_gpt2_token` /
`junk_bits_per_gpt2_token` for comparison to Atma's GPT-2-token eval, or `clean_bpb` /
`junk_bpb` for tokenizer-independent bits per byte.

Merge pretrained and Raven baselines into the normal dashboard:

```bash
python -m ablation.build_dashboard \
  --log_dir ablation/logs ablation/open_logs raven_baseline/logs \
  --out pages/dashboard.html
```

## Evaluation

- `clean_ppl[L]`: nats/token on coherent docs (`codelion/finepdfs-100M`), nested prefixes.
- `junk_ppl[L]`: nats/token on the concatenated `.bin` val stream.
- `needle[d]`: induction needle CE plus greedy per-digit accuracy at gap `d`.
- `L, d in {2048, 4096, 8192, 16384, 32768, 65536}`; plus MFU, wall-clock, and the val-loss-vs-wall-clock curve.

## Log Format

Each `<run_id>.log` contains human-readable training/eval lines plus delimited JSON blocks:
`===ABLATION_CONFIG_JSON===`, `===ABLATION_CURVE_JSON===`, `===ABLATION_EVAL_JSON===`, and
`===ABLATION_ERROR_JSON===` on failure. Each block is closed by `===END===`.

## Dashboard

`dashboard.html` is a single self-contained file with per-axis filters, a per-metric leaderboard,
run details, and a val-loss-vs-wall-clock plot for selected runs. It shows done / running / error /
missing counts against the 120-cell Atma grid, plus any extra outsider logs such as Raven.

## Files

| File | Role |
|---|---|
| `config_schema.py` | `RunConfig` dataclass, fair-grid expansion, and deterministic `run_id` |
| `generate_configs.py` | writes `configs/<run_id>.json` for the fair grid or explicit diagnostics |
| `train.py` | config-driven training plus in-process eval and structured logs |
| `evaluate.py` | structured clean/junk perplexity and needle eval |
| `open_baselines.py` | pretrained Hugging Face baseline runner with the same structured eval log format |
| `run_worker.py` | multi-GPU atomic file-claim runner |
| `parse_logs.py` | logs to `results.json` |
| `build_dashboard.py` | logs or `results.json` to self-contained dashboard HTML |
