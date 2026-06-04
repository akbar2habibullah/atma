# Benchmarks (scaled-up evals for final candidates)

Harness for the larger benchmark suite that the few final ablation winners will run at the
10B-token scale. It is built against the **production inference interface**
(`inference.LLM.generate`, see [docs/inference.md](../docs/inference.md)) so it stays fast and
matches how the model is actually served.

> ## ⚠️ NOT FUNCTIONAL YET — placeholder numbers
> The paged inference engine still runs **legacy softmax attention**. It does **not** implement
> Polar attention, the Titans memory branch, the Canon convs, or the training sliding window
> (porting them is the tracked task in [docs/inference.md](../docs/inference.md)). So for any
> checkpoint from the polar/titans line, the engine's generations are **numerically invalid**,
> and **every score this harness produces is a placeholder** until the inference port is done.
> The harness is written now against the intended stable autoregressive interface so it runs
> unchanged the moment inference lands. `benchmarks.EvalModel` prints this disclaimer on
> construction and lists exactly which features the engine is missing for your checkpoint;
> pass `--strict` to hard-fail instead of producing placeholders.

## First benchmark: BABILong

[BABILong](https://arxiv.org/abs/2406.10149) (NeurIPS 2024) embeds the 20 bAbI reasoning tasks
(qa1–qa20: fact-chaining, multi-hop, counting, lists, …) inside PG-19 background text —
*reasoning-in-a-haystack*. It's the most architecture-relevant long-context benchmark for this
model: it was **designed and validated at 130–137M params with recurrent memory** (RMT/ARMT/
Mamba reached 90%+ on qa1 out to 1M–50M tokens), which is exactly the polar + Titans regime.

**It is a *fine-tuned* capability probe, not a zero-shot quality eval.** A 370M *base* model
scores ~0 zero-shot (it won't follow the QA format). The decisive protocol:

1. **Fine-tune** a final candidate on bAbI qa1 (then qa1–5) with a **length curriculum**
   (RMT used 1→2→4→…→32 segments).
2. **Evaluate** across the length configs (0k → 64k realistic; 128k–1M aspirational) with this
   harness.
3. **The headline experiment:** `mem_enabled=True` vs `False`, both fine-tuned — if polar+titans
   sustains accuracy to N tokens while memory-off collapses at the training length, that is the
   single most decisive evidence for the Titans branch (the direct analog of "RMT recurrence
   enables 1M tokens"). Peer baselines: RMT-GPT2-137M, ARMT, Mamba-130M.

```bash
# once inference is ported (and on a fine-tuned checkpoint):
python -m benchmarks.run --benchmark babilong --model checkpoints/<run_id> \
    --tasks qa1 qa2 --lengths 0k 1k 2k 4k 8k 16k 32k 64k --samples 100 \
    --out benchmarks/logs/babilong_<run_id>.log

# leaderboard-parity prompts: `pip install babilong` and the harness auto-uses the official
# babilong.prompts; otherwise it falls back to built-in minimal templates (clearly logged).
```

Log format mirrors the ablation: a human table plus a `===BABILONG_RESULTS_JSON===` block
(accuracy matrix task×length, model config, and a `wip_placeholder` flag) for downstream parsing.

## Second benchmark: synthetic retrieval (passkey + NIAH)

Generate-and-match retrieval over a **length × depth** grid, built deterministically with the
GPT-2 tokenizer so each cell is an *exact* token length with the needle at a controlled depth
(no external dataset; optional real-text haystack via `--haystack`). This is the realistic-at-
370M subset of RULER (`niah_single`); it complements BABILong (retrieval+*reasoning*) with
*pure* retrieval, and complements [eval.py](../eval.py)'s loglikelihood induction-needle with the
*generation* (served-model) view of the same capability.

```bash
python -m benchmarks.run --benchmark retrieval --model checkpoints/<run_id> \
    --tasks passkey niah --lengths 1k 2k 4k 8k 16k 32k 64k --depths 0 0.25 0.5 0.75 1 \
    --samples 50 --out benchmarks/logs/retrieval_<run_id>.log
# --haystack codelion/finepdfs-100M  -> real-text NIAH instead of synthetic filler
```

## Files

| File | Role |
|---|---|
| `model.py` | `EvalModel` — adapter over `inference.LLM.generate()` + the WIP disclaimer/guard (reads the checkpoint `config.json` and lists unsupported features) |
| `babilong.py` | BABILong harness: prompt formatting (official `babilong.prompts` if installed, else built-in), generation, `compare_answers` scoring across tasks×lengths |
| `retrieval.py` | synthetic passkey + NIAH over a length×depth grid (exact GPT-2-token lengths), generation-scored |
| `run.py` | CLI dispatcher (`python -m benchmarks.run --benchmark babilong\|retrieval`) + structured log |

## Roadmap — where the *other* benchmarks live

The split is driven by the inference interface (generate-only today):

| Benchmark | Fits generate()? | Status / home |
|---|---|---|
| BABILong (reasoning-in-haystack) | yes | **`benchmarks/babilong.py`** ✓ |
| Passkey / NIAH (retrieval) | yes | **`benchmarks/retrieval.py`** ✓ |
| RULER full 13-task suite | yes, but heavy data-gen + ~0 signal at 370M base | deferred; the useful `niah_single` subset is covered by `retrieval.py` |
| MCQ quality suite (HellaSwag, PIQA, WinoGrande, ARC-e/c, OBQA, SIQA, LAMBADA…) | **no** — needs per-continuation loglikelihood | lm-eval-harness, once the engine exposes prompt/sample logprobs (small future interface add). Baselines: **Pythia-410M @ matched ~10B-token checkpoint** + your own RoPE/NoPE ablation cells |
| Long-doc perplexity (PG-19, Proof-pile, finepdfs; bits-per-byte) | **no** — needs per-token logprobs | already feasible via direct-forward [eval.py](../eval.py) |

> Why BABILong/retrieval over RULER for this model: RULER targets ≥6B instruct models and most of
> its 13 subtasks read ~0 for a 370M base; BABILong was built at 130–137M with memory — your exact
> regime — and passkey/NIAH are feasible at this scale.
>
> **Unlocking the MCQ suite** needs one small engine change: have `generate()`/`SamplingParams`
> optionally return prompt token logprobs (vLLM-style `prompt_logprobs`). Then a thin
> lm-eval-harness `LM` adapter over `EvalModel` covers all the loglikelihood tasks. That's the next
> interface step after the polar/titans inference port itself.
