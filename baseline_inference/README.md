# Baseline inference forks

This package serves ablation checkpoints without changing the production `inference/`
model. It deliberately reuses the production scheduler, sampling contract, state helpers,
and cache storage by import, while keeping baseline model definitions and kernels here.

Supported checkpoint configurations:

- `attn_type=nope`: Canon convolution + paged GQA softmax + optional Titans memory.
- `attn_type=rope`: half-truncate RoPE + paged GQA softmax + optional Titans memory.
- `arch_type=raven_native|atma_raven|atma_raven_titans`: FLA GSA recurrent state tables;
  hybrid variants retain the 12 LFM2 layers and the Titans variant carries both states.

The `benchmarks.EvalModel` adapter selects this package automatically from `config.json`.
Cross-request prefix reuse is disabled because recurrent Canon/Titans/Raven states are not
stored in the prefix cache. Dense same-length fresh prefill, chunk state carry, CUDA-graph
decode, and the existing `last_metrics` interface remain available.

L40S random-weight serving smoke benchmark (B=16, prompt/context=512, full 16-layer shape):

| architecture | prefill tok/s | decode tok/s |
|---|---:|---:|
| NoPE + Titans | 202,106 | 6,561 |
| RoPE + Titans | 207,051 | 6,266 |
| Atma-Raven + Titans | 156,169 | 4,911 |

These are systems numbers, not quality results. Use identical checkpoint scale, hardware,
batch/context settings, tokenizer, generation policy, and evaluation budget for reports.
