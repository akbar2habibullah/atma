# Length Extrapolation & Long-Range Retrieval

Polar Attention is designed to **train short, infer long**. All evidence below is produced by [eval.py](../eval.py) and runs on a single 24 GB L4. See [POLAR_ATTENTION.md](../POLAR_ATTENTION.md) for the mechanism.

## 1. Polar bounds extrapolation; softmax does not

Matched runs (both trained at seq_len 4096, no distractor), validation loss vs context multiple:

| Context | 1× | 8× | 64× | 512× |
|---|---|---|---|---|
| Softmax `CausalSelfAttention` | 3.20 | 3.38 | 10.05 | **13.55** |
| **Polar Attention** | 3.34 | 3.71 | 6.02 | **6.48** |

Softmax dilutes as keys accumulate and the loss explodes; Polar stays bounded at every length, for a small in-distribution cost.

The residual degradation is the unbounded attention *pool* growing out-of-distribution (the participation ratio `n_eff` blows up past the training length). Capping the pool with a sliding window ≈ the training length is **near-optimal for perplexity** — on coherent long documents a window of ~1024–2048 beats full attention at every position, and full attention is the *worst* option at long range.

## 2. The distractor loss heavy-lifts long-range retrieval

Setting `num_random_keys > 0` adds a calibration loss: random keys projected through `K` must lose to the extreme-value-corrected null sink (it learns to reject noise and keeps attention sharp at length). It barely moves perplexity, but it is **decisive for retrieval**.

Induction needle-in-haystack: plant a natural sentence with a unique key + 5-digit value, re-present it at distance, score greedy per-digit accuracy on the value (trained at seq_len 2048, chance ≈ 1–4 %, full attention):

| Needle distance | `num_random_keys=0` | `num_random_keys=2048` |
|---|---|---|
| 2,048 (1× train) | 97.5 % | 93.8 % |
| 4,096 (2×) | 60.0 % | 70.0 % |
| 8,192 (4×) | 2.5 % (chance) | **48.8 %** |
| 16,384 (8×) | 0 % | **32.5 %** |
| 32,768 (16×) | 0 % | **10.0 %** |
| 65,536 (32×) | 0 % | **6.3 %** |

Without the distractor, retrieval falls off a cliff just past the training length; with it, the model still recalls a fact planted **32× beyond** its training context.

## Choosing an attention mode (workload-dependent)

A sliding window is **retrieval-blind** past its width, while full Polar attention retrieves far but pays an out-of-distribution perplexity tax. So:

- **Plain language modeling / locally-coherent generation** → sliding window ≈ training length (near-optimal perplexity, clean extrapolation, no extra cost).
- **Tasks needing recall of specific distant content** → full Polar attention with `num_random_keys > 0` (the only setting that retrieves far). A bounded-pool recurrent memory trained in-loop is the path to *both* in-distribution perplexity and distant recall.

## `eval.py` reference

```bash
# loss-only extrapolation sweep
python eval.py --multipliers 1 2 4 8 16 32 64

# perplexity vs sliding window, on single coherent long docs, with per-position L(t)
python eval.py --hf_dataset codelion/finepdfs-100M --windows 128 512 2048 full --per_position

# induction needle-in-haystack retrieval vs distance
python eval.py --needle --hf_dataset codelion/finepdfs-100M \
  --needle_distances 2048 4096 8192 16384 32768 65536
```

| Flag | What it does |
|---|---|
| *(none)* | loss-only extrapolation sweep at context multipliers |
| `--diagnose` | per-layer activation distribution + polar internals (`n_eff`, `w_null`, `mag`) vs N |
| `--window W` | eval-only causal sliding window of width `W` |
| `--windows … --per_position` | multi-window + full comparison in one run, with `L(t)` curves |
| `--hf_dataset ID` | single coherent long documents (nested prefixes) instead of the concatenated stream |
| `--needle` | induction needle-in-haystack: retrieval accuracy vs needle→query distance |

All probe modes run `embed → blocks` eager with a time-chunked LM head (fits 24 GB at 64×) and force the streaming/Triton polar path (the materialized `O(T²)` path OOMs past ~16×).
