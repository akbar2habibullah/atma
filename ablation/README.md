# Atma full-grid ablation

A complete **120-cell factorial** over the architecture recipe space, plus a multi-GPU
runner and an interactive static-HTML dashboard. See [TITANS_MEMORY.md](../docs/TITANS_MEMORY.md)
/ [POLAR_ATTENTION.md](../docs/POLAR_ATTENTION.md) for the architecture being ablated.

## The grid (5 × 2 × 2 × 2 × 3 = 120)

| Axis | Values |
|---|---|
| `reg_mode` | baseline, weak, strong, discrete, zipfian |
| `distractor` | off / on — `num_random_keys ∈ {0, seq_len}` |
| `memory` | off / on — Titans gated-delta memory branch |
| `window` | off / on — training sliding window = 1024 (training-only) |
| `attn_type` | `rope` (rotary, no canon) · `nope` (canon, no position) · `polar` · `wall` (canon + Wall Attention, 2nd batch) |

All cells share: 16 layers (12 conv + 4 attn), hidden 1024, head_dim 128, GQA 1:4,
`seq_len=2048`, ~1B tokens (1 epoch). Distractor/memory/window are wired into **all three**
attention cores, so every cell is distinct (incl. softmax + Titans). Window is a *training*
knob; every eval runs at **full context**.

## Workflow (4 separate GPU hosts, 30 configs each)

```bash
# 1) split the 120 cells into 4 balanced shards of 30 (one per GPU; round-robin over a
#    deterministic shuffle so each shard has a comparable heavy/light mix).
python -m ablation.generate_configs --out ablation/shards --shards 4
#    -> ablation/shards/shard{0,1,2,3}/  (30 *.json each)

# 2) copy one shard dir to each GPU host and run a worker on it (these hosts need NOT share
#    a filesystem). The worker trains the 30 cells sequentially; it is resumable (restart it
#    and it skips done cells). FLA_CUSTOM_OP=1 and ATMA_WALL_CUSTOM_OP=1 are set automatically.
FLA_CUSTOM_OP=1 ATMA_WALL_CUSTOM_OP=1 python -m ablation.run_worker --config_dir <shardN> --log_dir ablation/logs --gpu 0
#    (--once does a single cell; --reset un-claims crashed/failed cells back to pending)

# 3) collect every <run_id>.log from the 4 hosts into ONE folder, then build the dashboard
python -m ablation.parse_logs      --log_dir ablation/logs --out ablation/results.json
python -m ablation.build_dashboard --log_dir ablation/logs --out pages/dashboard.html
#    open pages/dashboard.html in any browser (offline, no install); it is also the
#    ablation page linked from the GitHub Pages site in pages/index.html
```

### Second batch — Wall attention (`shard5`)

`attn_type="wall"` (Tilde Research [Wall Attention](https://github.com/tilde-research/wall-attention-release))
is a 4th core, added after the first 120-cell batch — making the full grid **160**. Its 40 cells
(5×2×2×2) are generated separately into `ablation/shards/shard5`:

```bash
python -m ablation.generate_configs --attn_types wall --mbs 2 --out ablation/shards/shard5   # 40 *.json
# run like any other shard (one or more GPUs):
FLA_CUSTOM_OP=1 ATMA_WALL_CUSTOM_OP=1 python -m ablation.run_worker --config_dir ablation/shards/shard5 --log_dir ablation/logs --gpu 0
```

Wall keeps canon (so it's the matched comparison to `nope` - isolates the per-channel gating).
On CUDA, training requires the Wall Triton kernel and defaults to `ATMA_WALL_CUSTOM_OP=1`, which
wraps the Wall forward/backward as opaque `torch.library` custom ops to reduce `torch.compile`
graph liveness. Set `ATMA_WALL_CUSTOM_OP=0` for the raw kernel path. Faithful long-context eval
(>~4k) needs the Wall kernel installed on the host and validated before trusting 65k
needle/perplexity numbers. See [FUTURE.md section 4](../docs/FUTURE.md).

> The full flat set is also available (`python -m ablation.generate_configs --out ablation/configs`
> → 120 `*.json`). On a single host with multiple GPUs that *do* share a filesystem, point one
> worker per GPU at the same `--config_dir`: they coordinate by atomic `os.rename` file-claim so
> no cell runs twice.

Train a single cell directly (what the worker calls):

```bash
FLA_CUSTOM_OP=1 python -m ablation.train \
    --config ablation/configs/polar__reg-zipfian__distr-1__mem-1__win-1.json \
    --log    ablation/logs/polar__reg-zipfian__distr-1__mem-1__win-1.log
```

Smoke (3 steps, 1 chunk) to validate the pipeline before a full sweep:

```bash
python -m ablation.generate_configs --out ablation/smoke --num_chunks 1 --val_tokens 524288 --max_steps 3
FLA_CUSTOM_OP=1 python -m ablation.run_worker --config_dir ablation/smoke --log_dir ablation/smoke_logs --gpu 0
```

## Open-weight pretrained baselines

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

The runner writes one `<run_id>.log` per model using the same `ABLATION_*_JSON`
blocks as trained cells. `clean_ppl` and `needle` use each model's native tokenizer
on coherent `codelion/finepdfs-100M` documents. `junk_ppl` decodes the GPT-2
FineWeb-Edu validation `.bin` stream and retokenizes it for the target model, so
the baseline keeps the original junk-stream content while respecting each tokenizer.
`clean_ppl`/`junk_ppl` remain nats per native model token for backwards compatibility;
use `clean_bits_per_gpt2_token` / `junk_bits_per_gpt2_token` for comparison to Atma's
GPT-2-token eval, or `clean_bpb` / `junk_bpb` for tokenizer-independent bits per byte.
Gemma-family checkpoints default to fp32 in this runner because fp16 forwards on
T4/SDPA can emit non-finite logits; the other models keep the CUDA fp16 default
when `--dtype auto` is used.

Colab T4-friendly knobs:

```bash
# quick smoke: one model, 2 docs/trials, short lengths
python -m ablation.open_baselines \
  --models HuggingFaceTB/SmolLM2-360M \
  --lengths 2048 4096 \
  --needle_distances 2048 4096 \
  --num_eval_docs 2 \
  --num_needle_trials 2 \
  --chunk_size 256 \
  --log_dir ablation/open_smoke_logs

# full default set, but reduce chunk size if a model OOMs at 64K
python -m ablation.open_baselines --chunk_size 256 --log_dir ablation/open_logs
```

If a repo needs custom Transformers code, add `--trust_remote_code`. If you want
strict advertised-context evaluation instead of extrapolating to 64K, add
`--respect_model_max`.

Merge pretrained baselines into the normal dashboard:

```bash
python -m ablation.parse_logs --log_dir ablation/open_logs --out ablation/open_results.json
python -m ablation.build_dashboard --log_dir ablation/open_logs --out pages/open_baselines.html

# or copy/open logs into ablation/logs, then rebuild the combined dashboard
python -m ablation.build_dashboard --log_dir ablation/logs --out pages/dashboard.html
```

## Evaluation (each run, at full context)

- **clean_ppl[L]** — nats/token on coherent docs (`codelion/finepdfs-100M`), nested prefixes.
- **junk_ppl[L]** — nats/token on the concatenated `.bin` val stream.
- **needle[d]** — induction needle CE + greedy per-digit accuracy at gap `d` (+ needle-absent baseline).
- `L, d ∈ {2048, 4096, 16384, 32768, 65536}`; plus MFU, wall-clock, and the val-loss-vs-wall-clock curve.

## Log format (`<run_id>.log`, self-describing)

Human-readable training/eval lines **plus** three delimited JSON blocks the parser keys on:
`===ABLATION_CONFIG_JSON===` · `===ABLATION_CURVE_JSON===` · `===ABLATION_EVAL_JSON===`
(and `===ABLATION_ERROR_JSON===` on failure), each closed by `===END===`.

## Dashboard

`dashboard.html` is a single self-contained file: per-axis filters, a per-metric leaderboard
(each row drills down to the full config + all metrics), and a val-loss-vs-wall-clock canvas
plot for selected runs. It also shows done / running / error / missing counts against the
120-cell grid, so it is legible while the sweep is still in flight.

## Files

| File | Role |
|---|---|
| `config_schema.py` | `RunConfig` dataclass + `expand_grid()` (the 120 cells) + deterministic `run_id` |
| `generate_configs.py` | writes `configs/<run_id>.json` ×120 |
| `train.py` | config-driven training (mirrors [train.py](../train.py)) + in-process eval + structured log |
| `evaluate.py` | structured clean/junk perplexity + needle (reuses [eval.py](../eval.py) helpers) |
| `open_baselines.py` | pretrained Hugging Face baseline runner with the same structured eval log format |
| `run_worker.py` | multi-GPU atomic file-claim runner (resumable; `--reset`) |
| `parse_logs.py` | logs → `results.json` |
| `build_dashboard.py` | `results.json` → self-contained `dashboard.html` |

> Compute note: 120 × ~14–18 h ≈ 1.7–2.2k GPU-hours. Run in waves; the dashboard updates as
> logs arrive. Checkpoints are **not** persisted by default (eval is in-process); pass
> `--save_ckpt` to `ablation.train` if you want them.
