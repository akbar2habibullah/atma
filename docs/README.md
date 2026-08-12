# Documentation

This directory separates stable user documentation from experiment-specific research notes. The current ICLR 2027 manuscript and archived benchmark payloads remain authoritative for reported numbers.

## Start here

| Topic | Document |
|---|---|
| Architecture and Polar equations | [Polar Attention](POLAR_ATTENTION.md) |
| Gated-delta memory | [Titans compression memory](TITANS_MEMORY.md) |
| Training a model | [Training guide](training.md) |
| Current results and evaluation protocol | [Evaluation guide](evaluation.md) |
| Serving checkpoints | [Inference guide](inference.md) |
| Kernel routing and performance | [Kernel guide](kernel.md) |

## Experiment documentation

- [120-cell recipe selection](../ablation/README.md)
- [9.816B-token matched runs](../scaled_ablation/README.md)
- [Benchmark matrix and protocols](../benchmarks/README.md)
- [Raven-family baselines](../raven_baseline/README.md)
- [Edge runtime](../edge/README.md)

## Systems and runbooks

- [Checkpoint variability and 256K stress audit](research/checkpoint-variability.md)
- [B200/B300 profiling runbook](runbooks/blackwell-profiling.md)
- [Triton kernel package](../kernel/README.md)

## Research notes

- [Future directions](research/future-directions.md) contains hypotheses and proposed experiments. It is not a statement of implemented behavior or current headline results.
- [`archive/`](../archive/) contains retired prototypes and historical implementations.

## Source-of-truth policy

When results disagree, use this order:

1. Archived benchmark payloads and experiment logs.
2. The current manuscript in [`paper/iclr2027/`](../paper/iclr2027/).
3. This documentation and the Pages site.
4. Historical drafts and research notes.

Every result-facing documentation change should name the evaluation protocol, distinguish Stage I from Stage II, and avoid treating teacher-forced token accuracy as autoregressive generation success.
