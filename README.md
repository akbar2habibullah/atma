# ATMA

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![Numerical verification](https://img.shields.io/badge/numerical_verification-30%2F30-success.svg)](tests/verify.py)

ATMA is a 378M-parameter hybrid language model for studying long-context extrapolation. Its global layers combine **Polar Attention**—a bounded direction/magnitude attention rule—with **gated-delta recurrent memory**. The repository contains numerically cross-checked reference, training, and paged-inference implementations, plus the experiment and paper artifacts used in the ICLR 2027 draft.

> Current manuscript: **[ATMA: Long-Context Language Modeling via Polar Attention and Gated-Delta Compression Memory](paper/iclr2027/iclr2027_conference.pdf)**. The paper draft and its archived experiment artifacts are the source of truth for reported results.

## Current result

The study has two stages:

- A complete **120-cell, ~1B-token factorial** selects full-context Polar + memory, without a training window or distractor alignment. Adding memory improves Polar's teacher-forced five-token accuracy at 64K in **20/20** matched cells (mean **+47.8 points**).
- Matched NoPE, RoPE, and Polar models are then trained for **9.816B tokens at length 2K** and evaluated through **256K**. Polar retains **34.4% teacher-forced target-token accuracy** and **9.0% exact five-token accuracy** overall at 256K. The exact successes are synthetic: FinePDFs exact retrieval is **0.0%** at 64K and 256K.

This is a multi-objective result, not a blanket architecture win. Polar has the strongest extreme-length retrieval and fixed-target likelihood among the matched attention variants, but trails NoPE by **1.86 points** on the mean of eight 2K controls. Separately optimized Raven baselines lead BABILong and provide length-independent recurrent decode state.

See the [research article](pages/index.html), [interactive ablation dashboard](pages/dashboard.html), and [evaluation guide](docs/evaluation.md) for the scoped claims and full protocol.

## Architecture

The canonical model uses 16 pre-norm decoder blocks with a 3:1 convolution-to-attention ratio:

- 12 LFM2-style gated depthwise causal convolution layers.
- 4 global mixing layers using Polar Attention and an additive gated-delta memory channel.
- Exact order: **Local → Local → Global → Local**, repeated four times (global blocks 3, 7, 11, and 15 in one-based numbering).
- Squared-ReLU gated MLPs with 4× hidden expansion.

| Configuration | Value | Configuration | Value |
|---|---:|---|---:|
| Parameters | 378.16–378.22M | Layers | 16 (12 conv + 4 global) |
| Hidden size | 1024 | Attention heads | 8 |
| Head dimension | 128 | KV heads | 2 |
| Vocabulary | 50,304 | Training length | 2,048 |

The optimized paths are checked against the pure-PyTorch oracle at `atol=1e-4`:

```text
training implementation ─┐
                         ├─ reference model ── numerical parity ── paged inference
Triton / FLA kernels ─────┘
```

## Quick start

```bash
python train.py
python -m tests.verify
```

```python
from inference import LLM, SamplingParams

llm = LLM(model="checkpoints/weights.pt")
outputs = llm.generate(
    ["Hello, world!"],
    SamplingParams(temperature=0.7, max_tokens=256),
)
print(outputs[0]["text"])
```

GPU verification and fused paths:

```bash
FLA_CUSTOM_OP=1 python -m tests.verify --cuda
```

## Repository map

| Path | Purpose |
|---|---|
| [`model/`](model/) | Shared configuration, layers, blocks, and PyTorch oracle |
| [`train/`](train/) | Optimized training model, data loading, and optimizers |
| [`inference/`](inference/) | Paged production inference engine |
| [`kernel/`](kernel/) | Triton Polar, convolution, decode, and cross-entropy kernels |
| [`tests/`](tests/) | Cross-pipeline parity and integration checks |
| [`ablation/`](ablation/) | Stage I 120-cell factorial and dashboard builder |
| [`scaled_ablation/`](scaled_ablation/) | Stage II 9.816B-token matched attention runs |
| [`benchmarks/`](benchmarks/) | Retrieval, BABILong, long-document, base-task, and serving evaluation |
| [`raven_baseline/`](raven_baseline/) | Separately optimized recurrent-family baselines |
| [`supplementary/robustness/`](supplementary/robustness/) | Paired replications, Polar component study, and gated modern-baseline pilots |
| [`baseline_inference/`](baseline_inference/) | Checkpoint-exact inference forks used by the benchmark harness |
| [`edge/`](edge/) | Experimental tinygrad edge runtime |
| [`pages/`](pages/) | Static research article and generated dashboard |
| [`paper/iclr2027/`](paper/iclr2027/) | Current manuscript source and compiled PDF |
| [`docs/`](docs/) | Documentation hub, guides, systems notes, and research notes |
| [`archive/`](archive/) | Historical prototypes retained for provenance, not active code |

The top-level experiment packages intentionally remain importable (`python -m ablation...`, `python -m benchmarks...`). Historical or non-operational material belongs in `archive/`; new user-facing documentation should be linked from [`docs/README.md`](docs/README.md).

## Documentation

Start at the **[documentation hub](docs/README.md)**. The most-used references are:

- [Polar Attention](docs/POLAR_ATTENTION.md)
- [Gated-delta compression memory](docs/TITANS_MEMORY.md)
- [Training](docs/training.md)
- [Evaluation and current results](docs/evaluation.md)
- [Inference](docs/inference.md)
- [Kernel routes and measurements](docs/kernel.md)
- [Benchmark protocol and result artifacts](benchmarks/README.md)

## Reproducing published artifacts

```bash
# Rebuild the self-contained 120-cell dashboard.
python -m ablation.build_dashboard \
  --results ablation/results.json \
  --out pages/dashboard.html

# Run the benchmark test suite without requiring model checkpoints.
python -m pytest tests/test_benchmark_pipeline.py tests/test_pipeline_completion.py
```

The Stage II benchmark matrix is committed at [`benchmarks/logs/atma_10b/benchmark_matrix.json`](benchmarks/logs/atma_10b/benchmark_matrix.json). Treat generated figures, the Pages article, and manuscript tables as views over the archived logs—not independent sources of truth.

## References

- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)
- [LFM2: Liquid Foundation Models 2](https://arxiv.org/abs/2511.23404)
- [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt)
- [Physics of Language Models: Architecture Design and the Magic of Canon Layers](https://arxiv.org/abs/2512.17351)
