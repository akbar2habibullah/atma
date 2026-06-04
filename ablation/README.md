# Atma full-grid ablation

A complete **120-cell factorial** over the architecture recipe space, plus a multi-GPU
runner and an interactive static-HTML dashboard. See [TITANS_MEMORY.md](../TITANS_MEMORY.md)
/ [POLAR_ATTENTION.md](../POLAR_ATTENTION.md) for the architecture being ablated.

## The grid (5 × 2 × 2 × 2 × 3 = 120)

| Axis | Values |
|---|---|
| `reg_mode` | baseline, weak, strong, discrete, zipfian |
| `distractor` | off / on — `num_random_keys ∈ {0, seq_len}` |
| `memory` | off / on — Titans gated-delta memory branch |
| `window` | off / on — training sliding window = 1024 (training-only) |
| `attn_type` | `rope` (rotary, no canon) · `nope` (canon, no position) · `polar` |

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
#    and it skips done cells). FLA_CUSTOM_OP=1 is set automatically.
FLA_CUSTOM_OP=1 python -m ablation.run_worker --config_dir <shardN> --log_dir ablation/logs --gpu 0
#    (--once does a single cell; --reset un-claims crashed/failed cells back to pending)

# 3) collect every <run_id>.log from the 4 hosts into ONE folder, then build the dashboard
python -m ablation.parse_logs      --log_dir ablation/logs --out ablation/results.json
python -m ablation.build_dashboard --log_dir ablation/logs --out ablation/dashboard.html
#    open ablation/dashboard.html in any browser (offline, no install)
```

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
| `run_worker.py` | multi-GPU atomic file-claim runner (resumable; `--reset`) |
| `parse_logs.py` | logs → `results.json` |
| `build_dashboard.py` | `results.json` → self-contained `dashboard.html` |

> Compute note: 120 × ~14–18 h ≈ 1.7–2.2k GPU-hours. Run in waves; the dashboard updates as
> logs arrive. Checkpoints are **not** persisted by default (eval is in-process); pass
> `--save_ckpt` to `ablation.train` if you want them.
