# Length Extrapolation & Long-Range Retrieval

Polar Attention is designed to **train short, infer long**. All numbers below come from the **120-cell ablation** — identical 370M models, `seq_len=2048`, ~1B tokens, evaluated at full context from 2K to 64K (32× train length) on a single 24 GB L4. The full grid is browsable in [pages/dashboard.html](../pages/dashboard.html); the probes themselves are produced by [eval.py](../eval.py) (reference at the bottom). See [POLAR_ATTENTION.md](POLAR_ATTENTION.md) for the mechanism.

## 1. Attention alone does not extrapolate — with or without polar

Memoryless cores (baseline regularizer, no distractor, no window). Both collapse on retrieval and degrade on perplexity within a few × of the training length:

| Length (× train) | softmax clean ppl | polar clean ppl | softmax needle | polar needle |
|---|---|---|---|---|
| 2,048 (1×) | 2.76 | 2.83 | 98 % | 96 % |
| 8,192 (4×) | 2.67 | 3.32 | 28 % | 15 % |
| 32,768 (16×) | 3.38 | 3.60 | 3 % | 3 % |
| 65,536 (32×) | 3.71 | 3.60 | 1 % | 0 % |

Softmax dilutes as keys accumulate; polar's attention *pool* grows out-of-distribution (the participation ratio `n_eff` blows up past the training length). Neither core retrieves far on its own. Polar degrades a little more gracefully on the raw stream at long range (junk-stream ppl `5.68` vs softmax `6.68` @ 32×), but both are far from usable — the fix is not in the attention core alone.

> The distractor loss (`num_random_keys > 0`) calibrates the null floor against random keys. On the **memoryless** polar core in this sweep it does **not** rescue retrieval — it collapses the needle (`85 % → 0 %`). The lever that works is the memory.

## 2. The Titans memory is the unlock

Adding the [Titans compression memory](TITANS_MEMORY.md) to **full polar, with no window and no distractor** (`mem_enabled=True`, `attn_window=None`, `num_random_keys=0`) — the ablation winner — holds **both** properties at once:

- **Perplexity reverses and improves monotonically.** Clean-document perplexity (coherent finepdfs documents) *falls* with length, `2.70 → 1.96` across the 2K→64K sweep, where the memoryless polar core was the **worst** option (blowing up past `3.6`). The gain is the memory, not a window: the polar core trains at the same `N ≤ 1024` operating point either way.
- **Retrieval stays flat above 90 % to 32× train length.** The induction needle holds `91–98 %` across the whole sweep (length-weighted `94 %`).

| Needle distance | polar, no memory (§1) | softmax + memory | **polar + memory** |
|---|---|---|---|
| 2,048 (1×) | 96 % | 98 % | 91 % |
| 4,096 (2×) | 40 % | 98 % | 95 % |
| 8,192 (4×) | 15 % | 94 % | 93 % |
| 16,384 (8×) | 8 % | 85 % | **98 %** |
| 32,768 (16×) | 3 % | 48 % | **96 %** |
| 65,536 (32×) | 0 % | 16 % | **93 %** |

**Polar earns its keep at extreme length.** Softmax + the *same* memory holds early but collapses past ~16× (the `n_eff` blow-up reappears in its readout); only **polar + memory** stays flat to 32× — and its perplexity is best and monotonic (`1.96` vs softmax-memory `2.34 @ 64×`). Convergence and quality rank **Polar+Titans > Softmax+Titans > polar-only**.

## 3. With the memory, the distractor and window only hurt

Starting from the winner and adding back the two levers that "helped" the *memoryless* core makes retrieval **worse**, not better (needle accuracy):

| Needle distance | polar + memory (winner) | + distractor | + window (1024) |
|---|---|---|---|
| 2,048 (1×) | 91 % | 74 % | 51 % |
| 16,384 (8×) | **98 %** | 76 % | 53 % |
| 65,536 (32×) | **93 %** | 59 % | 30 % |

The window is retrieval-blind past its width (it never trains attention on distance-`1024`+ keys); the distractor over-sharpens the null floor and fights the memory's diffuse readout. Clean-document perplexity is essentially unchanged across the three (`1.96 – 2.01 @ 64×`), so the cleanest recipe is strictly best — it wins on retrieval at no perplexity cost.

**Ablation — `--no_mem` confirms the memory is load-bearing.** Stripping the memory from a trained checkpoint breaks it globally (loss `2.8 → 5.7` at 1×, needle `0 %` everywhere): the model uses the memory at *all* lengths, not just long context. See [TITANS_MEMORY.md §7](TITANS_MEMORY.md#7-empirical-results) for the gated-delta math and the ~6–9 % MFU overhead.

## Choosing an attention mode

The ablation collapses the earlier per-workload guidance into one default:

- **Just use full polar + the Titans memory — no window, no distractor** (`mem_enabled=True`, `attn_window=None`, `num_random_keys=0`). It is the best recipe for in-distribution perplexity *and* distant retrieval at once, at ~6–9 % MFU overhead.
- A sliding window (`attn_window ≈ train length`) remains an option only for pure **local** language modeling, where distant recall is irrelevant and you want the lowest cost — but it is retrieval-blind past its width.

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
| `--no_mem` | strip the Titans memory branch from the checkpoint (sets `attn.mem = None`) to isolate its contribution |

All probe modes run `embed → blocks` eager with a time-chunked LM head (fits 24 GB at 64×) and force the streaming/Triton polar path (the materialized `O(T²)` path OOMs past ~16×).
