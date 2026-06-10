```text
  █████╗ ████████╗███╗   ███╗ █████╗ 
 ██╔══██╗╚══██╔══╝████╗ ████║██╔══██╗
 ███████║   ██║   ██╔████╔██║███████║
 ██╔══██║   ██║   ██║╚██╔╝██║██╔══██║
 ██║  ██║   ██║   ██║ ╚═╝ ██║██║  ██║
 ╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚═╝╚═╝  ╚═╝
```

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![Verify Status](https://img.shields.io/badge/numerical_verification-passed-success.svg)](verify.py)

**Atma** is a hybrid transformer-convolutional language model that integrates three parallel forward implementations—**reference**, **training**, and **inference**—into a single repository. Because every layer is numerically cross-verified, implementing and testing new architectural variants is straightforward.

## Architecture

Atma uses a **3:1 conv-to-attention ratio** across 16 decoder layers:

- **LFM2 Gated Convolution** (12 layers): inspired by Liquid Foundation Models 2. Gated depthwise causal conv1d provides linear-complexity sequence mixing.
- **Polar Attention** (4 layers, **default**): a length-invariant replacement for softmax SDPA. It keeps the Canon-B surround (GQA, horizontal residual convs on Q/K/V, QK-norm, `output * sigmoid(gate)`) but replaces the softmax core with two channels — a count-blind **direction** unit vector and a bounded **magnitude** (participation ratio through an extreme-value-corrected null sink). This bounds the attention output at any length, where softmax dilutes and blows up. Full derivation: **[POLAR_ATTENTION.md](POLAR_ATTENTION.md)**. (The legacy softmax `CausalSelfAttention` remains in the tree but is no longer wired into the model.)
- **Titans compression memory (MAG, default-on)**: each polar layer carries a length-invariant linear long-term memory (a per-head gated-delta fast-weight store), added as an **additive third channel** alongside a sliding-window short-term branch — `out = content + count + memory`. It resolves the window-vs-retrieval tradeoff (window wins perplexity, full wins recall, neither both): the memory supplies the diffuse long-context perplexity gain while full polar + the distractor supply exact retrieval. Full derivation, the FLA fused-kernel integration, and the first end-to-end results: **[TITANS_MEMORY.md](TITANS_MEMORY.md)**.

Each decoder block is pre-norm: `x = x + sublayer(norm(x))` then `x = x + MLP(norm(x))`. The MLP uses squared-ReLU gating with 4× hidden expansion.

| Config | Value | | Config | Value |
|---|---|---|---|---|
| Parameters | 369.72M | | Layers | 16 (12 conv + 4 attn) |
| Hidden dim | 1024 | | Heads | 8 (head_dim=128) |
| Vocab | 50304 | | KV Heads | 2 (1:4 GQA) |
| Sequence length | 1024 | | | |

The repo maintains strict numerical equivalence across its optimized pipelines (`verify.py`, `atol=1e-4`):

```text
             ┌───────────────────┐     ┌─────────────────────┐
             │ TRAINING PIPELINE │ ──> │  REFERENCE MODEL    │
             │ (FP8/FP16, Muon)  │     │ (PyTorch reference) │
             └───────────────────┘     └─────────────────────┘
                       │                         │
                       └───────────┬─────────────┘
                                   │ (verify.py atol=1e-4)
                                   ▼
             ┌─────────────────────────────────────────────┐
             │              INFERENCE ENGINE               │
             │ (Paged KV Cache, Centralized Conv State)    │
             └─────────────────────────────────────────────┘
```

> Equivalence holds across **training ↔ reference ↔ inference** (Polar attention + Titans memory; `verify.py` 30/30 on CPU). The paged engine's Triton decode path still needs validation on the target GPU — see [docs/inference.md](docs/inference.md).

## Quick start

```bash
python train.py          # train (downloads FineWebEdu-10B, GPT-2 tokenized)
python verify.py         # cross-check training vs reference at atol=1e-4
```

```python
from inference import LLM, SamplingParams

llm = LLM(model="checkpoints/weights.pt")
outputs = llm.generate(["Hello, world!"], SamplingParams(temperature=0.7, max_tokens=256))
print(outputs[0]["text"])
```

## Highlights — train short, infer long

Polar Attention keeps the attention output bounded at any length, where softmax dilutes and explodes:

| validation loss @ 512× context | Softmax | **Polar** |
|---|---|---|
| (trained at seq_len 4096) | 13.55 | **6.48** |

And the optional distractor loss (`num_random_keys > 0`) extends long-range **retrieval** dramatically — an induction needle planted **32× beyond** the training length is still recalled (6.3% vs 0% / chance without it). Full evidence, the perplexity-vs-window analysis, and the `eval.py` guide: **[docs/evaluation.md](docs/evaluation.md)**.

Adding the **Titans memory** (MAG) closes the rest of the gap: full-attention perplexity becomes *best and monotonic* (`1.93 @ 64×`, a reversal of polar-only where full was worst), the needle is recalled at `42%` out to `65536` tokens (64×), and convergence/perplexity now ranks **Polar+Titans > Causal > Polar-only** — at ~6–9% MFU overhead with the FLA fused kernel (`FLA_CUSTOM_OP=1`). Details: **[TITANS_MEMORY.md](TITANS_MEMORY.md)**.

## Documentation

| Doc | Contents |
|---|---|
| [POLAR_ATTENTION.md](POLAR_ATTENTION.md) | Polar Attention: derivation, design rationale, verification |
| [TITANS_MEMORY.md](TITANS_MEMORY.md) | Titans compression memory (MAG): gated-delta math, FLA fused kernel, results |
| [docs/training.md](docs/training.md) | Training pipeline, performance, checkpoints, tokenizing custom data |
| [docs/evaluation.md](docs/evaluation.md) | Length extrapolation, long-range retrieval, `eval.py` reference |
| [docs/inference.md](docs/inference.md) | Inference engine, usage, throughput, the Polar-port status |
| [kernel/README.md](kernel/README.md) | FlashAttention-style Triton polar kernels |

## Project structure

```
atma/
├── train.py / eval.py / verify.py     # train · evaluate (extrapolation, window, needle) · cross-verify
├── POLAR_ATTENTION.md                 # Polar Attention derivation
├── model/      # config, layers, blocks (Polar reductions), reference.py (pure-PyTorch oracle)
├── kernel/     # FlashAttention-style Triton polar kernels (fwd/bwd, decode, sliding window) + tests
├── train/      # TrainModel (Polar + FP16/FP8 matmuls), data loader, Muon optimizer, SigReg
└── inference/  # paged engine (Polar + Titans, paged decode kernel) + generate.py (self-contained Polar)
```

## References

- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)
- [LFM2: Liquid Foundation Models 2](https://arxiv.org/abs/2511.23404)
- [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt)
- [Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers](https://arxiv.org/abs/2512.17351)
