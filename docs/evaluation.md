# Evaluation and current results

ATMA is evaluated as a multi-objective long-context system. The current result set covers recipe selection, matched long-context comparisons, retrieval, fixed-target likelihood, short-context controls, adapted BABILong, and serving. The ICLR 2027 draft in [`paper/iclr2027/`](../paper/iclr2027/) is the source of truth for paper claims.

## Experimental stages

### Stage I: recipe selection

The full factorial trains 120 approximately 370–378M-parameter models for 1,900 optimizer steps, or about 1B FineWeb-Edu tokens, at sequence length 2,048 on NVIDIA L4 GPUs.

The grid crosses:

- attention core: Polar, NoPE, or RoPE;
- representation regularizer: five settings;
- distractor alignment: on or off;
- gated-delta memory: on or off;
- 1,024-token training window: on or off.

At 64K, adding memory improves Polar in all 20 matched nuisance-factor settings by **47.8 percentage points on average**. Among memory-enabled cells, Polar is never worse than NoPE and exceeds RoPE in all 20 comparisons. A training window reduces Polar+memory by **25.5 points on average**; distractor alignment has a smaller, nonuniform **−6.0-point** mean effect. The promoted recipe is therefore full-context Polar + memory without distractor alignment.

The selected baseline cell reaches **92.5% teacher-forced five-token accuracy at 64K**. This Stage I value comes from 16 trials in the original sweep and is a recipe-selection statistic, not the paper's final headline evaluation.

Browse all cells in the [interactive dashboard](../pages/dashboard.html) or rebuild it from [`ablation/results.json`](../ablation/results.json):

```bash
python -m ablation.build_dashboard \
  --results ablation/results.json \
  --out pages/dashboard.html
```

### Stage II: matched 9.816B-token comparison

NoPE, RoPE, and Polar models are trained for 18,722 optimizer steps—**9.816B tokens**—at length 2K in the same L40S software and hardware environment. Their sizes range from 378.16M to 378.22M parameters and all use the selected memory-enabled recipe.

Atma-Raven-Titans (382.37M) and Raven Native (388.54M) use the same data, tokenizer, token budget, and device class, but use a different model family and AdamW optimizer. They are useful operating points, not optimizer-matched architectural ablations.

## Long-context endpoints

| Group | Model | Retrieval token 2K | Retrieval token 256K | Exact 256K | BABILong 256K | Mean BPB 2K | Mean BPB 256K |
|---|---|---:|---:|---:|---:|---:|---:|
| Matched | NoPE | 99.2% | 0.6% | 0.0% | 0% | **1.432** | 8.097 |
| Matched | Polar | 98.9% | **34.4%** | **9.0%** | **28%** | 1.451 | **1.825** |
| Matched | RoPE | 88.2% | 0.0% | 0.0% | 14% | 1.439 | 2.512 |
| Raven / AdamW | Atma-Raven-Titans | 43.1% | 10.0% | 0.0% | 38% | **1.525** | **1.551** |
| Raven / AdamW | Raven Native | 41.1% | **20.3%** | 0.0% | **46%** | 1.581 | 1.597 |

Retrieval values average passkey and NIAH tasks, synthetic and FinePDFs haystacks, three needle depths, and paired trials. BABILong is macro exact match after task adaptation. BPB averages FinePDFs, PG-19, and Proof-Pile fixed-target evaluations.

## Retrieval boundary

The retrieval scorer appends the complete five-token target and evaluates each target position conditioned on preceding ground-truth target tokens.

- **Token accuracy** measures per-position argmax accuracy under teacher forcing.
- **Exact accuracy** requires all five target tokens to be correct under teacher forcing.
- Neither metric is autoregressive free-generation success.

Polar's result depends strongly on haystack type:

| Haystack | Metric | 2K | 64K | 256K |
|---|---|---:|---:|---:|
| Synthetic | Token accuracy | 99.6% | 92.3% | 68.4% |
| Synthetic | Exact five-token accuracy | 96.3% | 65.7% | 18.0% |
| FinePDFs | Token accuracy | 98.6% | 16.3% | 0.3% |
| FinePDFs | Exact five-token accuracy | 93.0% | 0.0% | 0.0% |

The defensible claim is retained target signal and exact synthetic retrieval. The current checkpoints do **not** demonstrate exact real-text retrieval at 64K or 256K.

## Short-context and systems trade-offs

Mean zero-shot accuracy across eight 2K controls is 45.15% for NoPE, 44.60% for RoPE, and 43.29% for Polar. Polar's 1.86-point gap to NoPE prevents a blanket quality claim.

On one L40S at 128K, the attention variants decode at about 16.3–16.5 ms/token and allocate about 39.4 GiB. The recurrent variants keep fixed-size state: Atma-Raven-Titans records 2.27 ms/token and 8.62 GiB, while Raven Native records 3.27 ms/token and 6.46 GiB. These are single-sequence, single-sample descriptive measurements rather than confidence intervals.

## Archived result payloads

| Artifact | Contents |
|---|---|
| [`ablation/results.json`](../ablation/results.json) | Complete Stage I grid |
| [`benchmarks/logs/atma_10b/benchmark_matrix.json`](../benchmarks/logs/atma_10b/benchmark_matrix.json) | Stage II retrieval, base-task, long-document, and serving rows |
| [`benchmarks/logs/babilong_2k_ft/benchmark_matrix.json`](../benchmarks/logs/babilong_2k_ft/benchmark_matrix.json) | Adapted BABILong rows |
| [`scaled_ablation/logs_stress/checkpoint_stress.json`](../scaled_ablation/logs_stress/checkpoint_stress.json) | Checkpoint stress diagnostics |

For protocol details and commands, see [`benchmarks/README.md`](../benchmarks/README.md). For the post-hoc checkpoint audit, see [checkpoint variability](research/checkpoint-variability.md).

## `eval.py` reference

The lightweight evaluation entry point supports loss extrapolation and induction-needle probes against local checkpoints:

```bash
python eval.py --checkpoint checkpoints
python eval.py --checkpoint checkpoints --needle
python eval.py --checkpoint checkpoints --no_mem
```

For paper-scale reproduction, use the checkpoint-exact harness in [`benchmarks/`](../benchmarks/) rather than treating this convenience script as the complete published protocol.
