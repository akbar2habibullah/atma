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

### One-Codebase Workflow
The repository maintains strict numerical equivalence across different optimized pipelines:

```text
             ┌───────────────────┐     ┌───────────────────┐
             │ TRAINING PIPELINE │ ──> │  REFERENCE MODEL  │
             │ (FP8/FP16, Muon)  │     │ (Pure PyTorch SDPA│
             └───────────────────┘     └───────────────────┘
                       │                         │
                       └───────────┬─────────────┘
                                   │ (verify.py atol=1e-4)
                                   ▼
             ┌─────────────────────────────────────────────┐
             │              INFERENCE ENGINE               │
             │ (Paged KV Cache, Centralized Conv State)    │
             └─────────────────────────────────────────────┘
```

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

### Loading a checkpoint for inference

After training completes, `train.py` writes three files to `checkpoints/`:

| File | Contents |
|---|---|
| `weights.pt` | `{"model": state_dict}` with all `_orig_mod.` compile prefixes stripped |
| `config.json` | `AtmaConfig` fields (dtype stored as a plain string, e.g. `"bfloat16"`) |
| `tokenizer.json` | `{"tokenizer_name": "<hf-repo-id>"}` for `AutoTokenizer.from_pretrained` |

To load the checkpoint into the inference engine:

```python
import json, torch
from model.config import AtmaConfig
from inference import LLM, SamplingParams
from transformers import AutoTokenizer

with open("checkpoints/config.json") as f:
    cfg = json.load(f)
with open("checkpoints/tokenizer.json") as f:
    tok = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(tok["tokenizer_name"])

atma_config = AtmaConfig(
    vocab_size=cfg["vocab_size"],
    num_hidden_layers=cfg["num_hidden_layers"],
    hidden_size=cfg["hidden_size"],
    head_dim=cfg["head_dim"],
    attn_kernel_size=cfg["attn_kernel_size"],
    conv_kernel_size=cfg["conv_kernel_size"],
    max_position_embeddings=cfg["max_position_embeddings"],
    rms_norm_eps=cfg["rms_norm_eps"],
    dtype=getattr(torch, cfg["dtype"]),
)

llm = LLM(model="checkpoints/weights.pt", hf_config=atma_config)
params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Hello, world!"], params)
print(outputs[0]["text"])
```

The `CHECKPOINT_DIR` and `TOKENIZER_NAME` constants at the top of `train.py` control where the checkpoint lands and which tokenizer is recorded.

### Tokenizing a custom dataset

`train/data.py` provides `tokenize_to_bin` to preprocess any HuggingFace dataset into the binary shard format expected by the training loop.

```python
from train.data import tokenize_to_bin

tokenize_to_bin(
    dataset_name="HuggingFaceFW/fineweb",
    tokenizer_name="gpt2",                 # any HF tokenizer repo id
    output_dir="./my_dataset_bins",
    file_prefix="fineweb",
    dataset_config="sample-10BT",          # optional dataset config name
    shard_size=10**8,                      # tokens per shard (default 100M)
    text_field="text",                     # dataset column containing document text
    split="train",
)
```

Each shard is written as a `.bin` file with a 256 × int32 header followed by packed token ids. The first shard is named `val`, the rest `train`:

```
my_dataset_bins/
├── fineweb_val_000000.bin
├── fineweb_train_000001.bin
├── fineweb_train_000002.bin
└── ...
```

**Token storage width** is chosen automatically based on vocabulary size:

| Vocab size | Storage | Example tokenizers |
|---|---|---|
| ≤ 65 536 | uint16 (2 bytes/token) | GPT-2, GPT-NeoX |
| > 65 536 | uint32 (4 bytes/token) | Llama-3, Mistral, Gemma |

The width is stored in `header[3]` so `_load_data_shard` detects it automatically. Legacy files produced before this change have `header[3] == 0` and are read as uint16.

To point `train.py` at the new shards, update the `filename_pattern` argument passed to `data_generator`:

```python
from train.data import data_generator

train_loader = data_generator("my_dataset_bins/fineweb_train_*.bin", batch_size, seq_len=1024)
val_loader   = data_generator("my_dataset_bins/fineweb_val_*.bin",   batch_size, seq_len=1024)
```

## Inference

Production-grade inference engine featuring:

- **Paged KV cache** with hash-based prefix caching (xxhash) for memory sharing
- **Chunked prefill** and **preemption** scheduling
- **CUDA graph capture** for decode at multiple batch sizes
- **Centralized conv state tables** enabling graph-captured decode
- **Tensor parallelism** support (ColumnParallel, RowParallel, QKVParallel, VocabParallel)
- **Flash Attention 3/2** with Triton KV cache kernel and SDPA fallback
- **Gumbel-max sampling**

### Usage

```python
from inference import LLM, SamplingParams

llm = LLM(model="path/to/weights.pt")
params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Hello, world!"], params)
print(outputs[0]["text"])
```

### Inference Decoding Performance

Model Size: 376M Params (16 num_layers + 1024 hidden_dim)

| Batch Size | NVIDIA L4  (tok/s) | NVIDIA H100 (tok/s) | NVIDIA T4 (tok/s)  |
|------------|--------------------|---------------------|--------------------|
| 1          | 197                | 438                 | 48                 |
| 4          | 757                | 1,686               | 136                |
| 8          | 1,447              | 3,211               | 233                |
| 16         | 2,754              | 6,128               | 383                |
| 32         | 4,906              | 11,785              | 561                |
| 64         | 8,224              | 21,326              | 743                |
| 128        | 12,534             | 36,120              | 799                |
| 256        | 15,718             | 54,140              | OOM                |
| 512        | 17,142             | **73,018**          | OOM                |

Measured at 256 generated tokens per sequence, prompt length ~320 words. 

- NVIDIA L4: FA3 enabled and kvcache_block_size=64.
- NVIDIA H100: FA2 enabled and kvcache_block_size=256.
- NVIDIA T4: SDPA only, max_num_batched_tokens=4096, and kvcache_block_size=64.

### Comparison with vLLM (H100)

Using identical model [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) with command:

```
> VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 vllm bench throughput --model LiquidAI/LFM2.5-350M --random-input-len 512 --random-output-len 256 --num-prompts 512

Throughput: 70.77 requests/s, 54350.30 total tokens/s, 18116.77 output tokens/s
Total num prompt tokens:  262144
Total num output tokens:  131072
``` 

### Further Throughput Benchmark

We also try larger model size 2249.97M (30 num_layers + 2048 hidden_dim) on H100 to test performance ceiling of Atma inference engine.

| Batch Size | Decoding   (tok/s) |
|------------|--------------------|
| 1          | 177                |
| 4          | 667                |
| 8          | 1,296              |
| 16         | 2,481              |
| 32         | 4,591              |
| 64         | 8,450              |
| 128        | 13,864             |
| 256        | 20,548             |
| 512        | 25,826             |
| 1024       | 26,714             |
| 2048       | 28,830             |

> Measured at 256 generated tokens per sequence, prompt length ~320 words. FA2 enabled and kvcache_block_size=256.

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
