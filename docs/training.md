# Training

Training pipeline (based on [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt) methodology) with dual optimizers:

- **Muon** for 2D+ weight matrices (`lr=0.02`, `weight_decay=0.01`).
- **AdamW** for the rest, in three groups: embeddings (`lr=0.3`), the LM-head projection (`lr=1/320`), and all 1D params (`lr=0.01`); `betas=(0.8, 0.95)`, `weight_decay=0`.
- A linear-cooldown LR schedule (`cooldown_frac=0.7` — constant for the first 30% of steps, then decayed to 0).
- Optional **SigReg** regularization (covariance whitening, kernel matching, discrete, or Zipfian modes; `train.py` defaults to `baseline, `SIGR_ALPHA=0.0`).

Supports FP16 (safe scaled matmul) and FP8 (E4M3/E5M2) custom ops.

```bash
python train.py
```

For short synthetic training throughput/MFU profiling without downloading data:

```bash
python -m scripts.bench_training_mfu --microbatch 16 --seq-length 1024 \
  --warmup 2 --iterations 5 --measure-peak
```

The benchmark permits and labels the compiled PyTorch causal-convolution fallback. Pass
`--require-optimized-conv` only when a compatible optional kernel is expected.

See [blackwell_profiling.md](blackwell_profiling.md#training-mfu) for the B200/B300 sweep protocol
and the MFU counting conventions.

Downloads FineWebEdu-10B (GPT-2 tokenized), memory-maps the shards, and trains with microbatch gradient accumulation (`mbs` microbatches per `batch_size` step). `train_steps` is derived from the tokens actually present — one 100M shard → ~190 steps, the full 1B-token set → 1900 steps. The `CHECKPOINT_DIR` and `TOKENIZER_NAME` constants at the top of `train.py` control where the checkpoint lands and which tokenizer is recorded.

## Training performance

Measured on **NVIDIA L4** for the current architecture — Polar attention + Titans memory (`378.22M` params with the memory branch; `369.72M` without), `seq_len=2048`, 1B tokens → **1900 steps**:

| Run | Avg step time | MFU | Wall-clock | Final val loss |
|---|---|---|---|---|
| NoPE | 27.33 s | 41.6% | ~14.42 h | 3.224 |
| NoPE + memory | 31.48 s | 36.8% | ~16.60 h | 3.140 |
| NoPE + memory + distractor (`num_random_keys=2048`) | 38.53 s | 30.0% | ~20.31 h | 3.148 |
| Polar | 28.33 s | 40.1% | ~14.93 h | 3.323 |
| Polar + memory | 32.49 s | 35.6% | ~17.14 h | 3.169 |
| Polar + memory + distractor (`num_random_keys=2048`) | 36.39 s | 31.8% | ~19.18 h | 3.178 |

> Per the [120-cell ablation](evaluation.md), it is **no longer recommended**: once the Titans memory is present, the distractor (and the sliding window) *hurt* long-range retrieval. The winning recipe is full polar + memory with neither — see [evaluation.md](evaluation.md).

## Loading a checkpoint for inference

After training completes, `train.py` writes three files to `checkpoints/`:

| File | Contents |
|---|---|
| `weights.pt` | `{"model": state_dict}` with all `_orig_mod.` compile prefixes stripped |
| `config.json` | `AtmaConfig` fields (dtype stored as a plain string, e.g. `"bfloat16"`) |
| `tokenizer.json` | `{"tokenizer_name": "<hf-repo-id>"}` for `AutoTokenizer.from_pretrained` |

Load the checkpoint into the paged inference engine — which now runs **Polar attention + Titans memory end to end** (ported and GPU-verified; see [inference.md](inference.md)):

```python
from inference import LLM, SamplingParams

llm = LLM(model="checkpoints/weights.pt")
outputs = llm.generate(["Hello, world!"], SamplingParams(temperature=0.7, max_tokens=256))
print(outputs[0]["text"])
```

The tokenizer repo id recorded in `tokenizer.json` is what you pass to `AutoTokenizer.from_pretrained(...)` to decode the returned `token_ids`. For decode throughput, the paged-state layout, and the verification status, see [inference.md](inference.md).

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

To point `train.py` at the new shards, update the `filename_pattern` argument passed to `data_generator` (`seq_len` is configurable — the current runs train at 2048):

```python
from train.data import data_generator

train_loader = data_generator("my_dataset_bins/fineweb_train_*.bin", batch_size, seq_len=2048)
val_loader   = data_generator("my_dataset_bins/fineweb_val_*.bin",   batch_size, seq_len=2048)
```
