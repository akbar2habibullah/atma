# Training

Training pipeline (based on [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt) methodology) with dual optimizers:

- **Muon** for 2D+ weight matrices (lr=0.02, wd=0.01)
- **AdamW** for embeddings, projections, and 1D params
- Optional **SigReg** regularization (covariance whitening, kernel matching, discrete, or Zipfian modes)

Supports FP16 (safe scaled matmul) and FP8 (E4M3/E5M2) custom ops.

```bash
python train.py
```

Downloads FineWebEdu-10B (GPT-2 tokenized), memory-maps shards, and trains with gradient accumulation. The `CHECKPOINT_DIR` and `TOKENIZER_NAME` constants at the top of `train.py` control where the checkpoint lands and which tokenizer is recorded.

## Training performance

- Total steps: 190
- Total training tokens: ~100M
- Batch size: 524,288 tokens (8 × 64K microbatches)

| GPU         | Avg Step Time | Train Time | Model FLOPs Utilization (MFU) |
|-------------|---------------|------------|-------------------------------|
| NVIDIA L4   | 28.64s        | 5485.82s   | 36.6%                         |
| NVIDIA H100 | 2.84s         | 605.11s    | 45.1%                         |

> For length-extrapolation training (`num_random_keys > 0`, the distractor calibration loss), see [evaluation.md](evaluation.md) — it is decisive for long-range retrieval and costs ~16% MFU per step.

## Loading a checkpoint for inference

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

> **Polar checkpoints:** the paged engine above still runs legacy softmax attention. For a Polar checkpoint, generate with the self-contained [inference/generate.py](../inference/generate.py) until the paged port lands — see [inference.md](inference.md).

## Tokenizing a custom dataset

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
