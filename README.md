# Atma

A hybrid transformer-convolutional language model with three parallel forward implementations (reference, training, inference) — all numerically verified against each other.

## Architecture

Atma uses a **3:1 conv-to-attention ratio** across 16 decoder layers:

- **LFM2 Gated Convolution** (12 layers): Inspired by Liquid Foundation Models 2. Gated depthwise causal conv1d provides linear-complexity sequence mixing.
- **Canon-B Attention** (4 layers): Multi-head attention with horizontal residual convolutions on Q/K/V, QK-norm, and learned adversarial gating (`output * sigmoid(gate)`).

Each decoder block follows a pre-norm pattern: `x = x + sublayer(norm(x))` followed by `x = x + MLP(norm(x))`. The MLP uses squared ReLU gating with 4x hidden expansion.

| Config | Value |
|---|---|
| Parameters | 376.04M |
| Hidden dim | 1024 |
| Heads | 8 (head_dim=128) |
| Layers | 16 (12 conv + 4 attn) |
| Vocab | 50304 |
| Sequence length | 1024 |

## Training

Training pipeline (based on NanoGPT speedrun methodology) with dual optimizers:
- **Muon** for 2D+ weight matrices (lr=0.02, wd=0.01)
- **AdamW** for embeddings, projections, and 1D params
- Optional **SigReg** regularization (covariance whitening, kernel matching, discrete, or Zipfian modes)

Supports FP16 (safe scaled matmul) and FP8 (E4M3/E5M2) custom ops.

### Performance on NVIDIA L4 (100M tokens, 190 steps)

| Metric | Value |
|---|---|
| Model FLOPs Utilization (MFU) | **36.2%** |
| Step time (steady state) | ~28.3s |
| Loss at step 190 | 4.19 |
| Batch size | 524,288 tokens (8 × 64K microbatches) |

### Training on more data

```
python train.py
```

Downloads FineWebEdu-10B (GPT-2 tokenized), memory-maps shards, and trains with gradient accumulation.

## Inference

Production-grade inference engine featuring:

- **Paged KV cache** with hash-based prefix caching (xxhash) for memory sharing
- **Chunked prefill** and **preemption** scheduling
- **CUDA graph capture** for decode at multiple batch sizes
- **Centralized conv state tables** enabling graph-captured decode
- **Tensor parallelism** support (ColumnParallel, RowParallel, QKVParallel, VocabParallel)
- **Flash Attention 3/2** with Triton KV cache kernel and SDPA fallback
- **Gumbel-max sampling**

### Performance on NVIDIA L4

| Batch Size | Throughput (tok/s) |
|------------|--------------------|
| 1          | 197                |
| 4          | 757                |
| 8          | 1,447              |
| 16         | 2,754              |
| 32         | 4,906              |
| 64         | 8,224              |
| 128        | 12,534             |
| 256        | 15,718             |
| 512        | **17,142**         |

Measured at 256 generated tokens per sequence, prompt length ~320 words.

### Usage

```python
from inference import LLM, SamplingParams

llm = LLM(model="path/to/weights.pt")
params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Hello, world!"], params)
print(outputs[0]["text"])
```

## Verification

All three implementations produce identical outputs at `atol=1e-4`:

```
python verify.py
```

Tests each layer type (RMSNorm, MLP, LFM2Conv, Attention, DecoderBlock) and end-to-end prefill + decode.

## Project Structure

```
atma/
├── train.py                     # Main training script
├── bench_inference.py           # Multi-batch-size throughput benchmark
├── example_inference.py         # End-to-end generation demo
├── verify.py                    # Numerical cross-implementation verification
├── model/
│   ├── config.py                # AtmaConfig dataclass
│   ├── layers.py                # RMSNorm, MLP (shared bases)
│   ├── blocks.py                # AtmaConvBase, AtmaAttnBase (shared bases)
│   └── reference.py             # ReferenceModel (ground-truth pure PyTorch)
├── train/
│   ├── model.py                 # TrainModel (custom FP16/FP8 matmuls)
│   ├── data.py                  # FineWebEdu binary shard loader
│   ├── optimizer.py             # Muon optimizer with Newton-Schulz
│   └── reg.py                   # SigReg regularization modes
└── inference/
    ├── llm.py                   # User-facing LLM class
    ├── config.py                # Inference Config
    ├── sampling_params.py       # SamplingParams
    ├── models/atma.py           # Inference model forward pass
    ├── layers/                  # attention, sampler, linear, embed_head, layernorm
    ├── engine/                  # llm_engine, scheduler, block_manager, sequence, model_runner
    └── utils/                   # Weight loader, thread-local context
```

## References

- [LFM2: Liquid Foundation Models 2](https://arxiv.org/abs/2511.23404)
- [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt)
- [Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers](https://arxiv.org/abs/2512.17351)
