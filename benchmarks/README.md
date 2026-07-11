# Benchmarks (downstream evals for final candidates)

This directory is for downstream evals run after `scaled_ablation/` has produced checkpoints.
Benchmarks stay outside the ablation worker so training, intrinsic eval, downstream generation,
and any task fine-tuning remain separable.

The current generation adapter supports checkpoints whose `config.json` has `attn_type="polar"`.
It passes the saved `AtmaConfig` into `inference.LLM` and resolves checkpoint directories to
`weights.pt`. Non-polar checkpoints should be run with direct-forward evals until the serving
engine has matching non-polar paths.

## Confound Rules

Use these rules for every reported number:

| Rule | Reason |
|---|---|
| Compare checkpoints with matched training tokens, data, tokenizer, parameter scale, and eval budget | Avoid treating scale/data differences as architecture effects |
| Use loglikelihood scoring for base-model knowledge/commonsense tasks | Avoid instruction-following and decoding-policy confounds |
| Use generation only for tasks where the answer is short and exact-matchable | Avoid subjective judging and verbosity confounds |
| Treat memory as fixed in the scaled run | The first ablation sweep already selected the recipe; the 10B run compares attention types under that recipe |
| Report OOM/skipped cells instead of silently shortening context | Avoid hidden context-length confounds |
| Do not compare base Atma checkpoints to instruction-tuned models on instruction benchmarks | Avoid alignment/instruction-tuning confounds |

## Benchmark Matrix

| Group | Benchmarks | Status | Purpose | Confound control |
|---|---|---|---|---|
| Intrinsic long-context LM | FinePDFs clean PPL, FineWeb-Edu junk PPL, induction needle loglikelihood | Covered by `scaled_ablation.evaluate` | Direct architecture signal during the scaled sweep | Direct forward, same tokenizer/data, OOM recorded per length |
| Exact retrieval generation | Passkey, synthetic NIAH, real-text NIAH | Implemented in `retrieval.py` | Served-model retrieval check | GPT-2 token exact lengths, greedy decode, exact-match short answers |
| Long reasoning after adaptation | BABILong qa1-10 | Implemented harness, requires task fine-tune | Measures each 10B attention type's adapted long-context potential | Same fixed recipe and fine-tune protocol for each attention type, with per-task reporting |
| Base LM quality controls | LAMBADA, HellaSwag, PIQA, WinoGrande, ARC-E, ARC-C, OpenBookQA, BoolQ | Needs loglikelihood adapter | Check that long-context architecture did not destroy normal LM capability | Multiple-choice/loglikelihood only, no chat prompts |
| Long-doc LM controls | PG-19, Proof-Pile, FinePDFs bits-per-byte | Direct-forward eval path, not generation | Domain/generalization check at long lengths | Per-token/byte likelihood, no decoding |
| Serving performance | Prefill tok/s, decode tok/s, max context before OOM | Partly exposed by `inference.LLM.last_metrics` | Practical deployment tradeoff | Same hardware, same batch/context settings |

## Open-Source Baseline Policy

Do not compare a task-fine-tuned Atma checkpoint against zero-shot open-source baselines.
That would measure fine-tuning as much as architecture.

Use two separate tables:

| Table | Models | Tasks | Interpretation |
|---|---|---|---|
| No-finetune base-model comparison | Atma scaled checkpoints and open-source pretrained checkpoints, all used as-is | PPL/BPB, LAMBADA, MCQ loglikelihood, exact retrieval if the prompt is model-agnostic | General base-model quality and retrieval behavior |
| Fine-tuned adaptation probe | Scaled Atma checkpoints fine-tuned with the same task recipe | BABILong qa1-10 | Adapted long-context potential per attention type under the fixed recipe |

Open-source baselines that we do not fine-tune should stay in the no-finetune table. For
BABILong, they can be reported only as zero-shot sanity checks, not as the main comparison,
because base pretrained models at this scale usually fail the output format and reasoning
protocol rather than just the long-context part.

## Excluded For Now

| Benchmark | Why excluded from the no-confound set |
|---|---|
| Full RULER 13-task suite | Many subtasks are instruction-following heavy and near-zero for 370M base models; use the NIAH/passkey subset first |
| LongBench / ZeroSCROLLS / Qasper / NarrativeQA generation | Mostly instruction/QA formatting and judge/prompt sensitive for base models |
| MMLU / GSM8K / BBH | Primarily scale/instruction-tuning limited at this model size and token budget |
| Summarization benchmarks | Require instruction tuning or subjective generation metrics |

## Retrieval

Synthetic passkey + NIAH over a length x depth grid. Prompts are built with the GPT-2 tokenizer
so each cell has an exact token length and controlled needle depth.

```bash
python -m benchmarks.run --benchmark retrieval --model checkpoints/<run_id> \
    --tasks passkey niah --lengths 1k 2k 4k 8k 16k 32k 64k 128k \
    --depths 0 0.25 0.5 0.75 1 --samples 50 \
    --max_num_seqs 16 --out benchmarks/logs/retrieval_<run_id>.log --strict
```

Add `--haystack codelion/finepdfs-100M` for real-text NIAH instead of synthetic filler.
Use `--max_model_len`, `--max_num_batched_tokens`, and `--max_num_seqs` to control memory.

## BABILong

[BABILong](https://arxiv.org/abs/2406.10149) embeds the 20 bAbI reasoning tasks inside PG-19
background text. Treat it as a fine-tuned adaptation probe for each 10B attention type, not a
zero-shot base-model quality benchmark.

In the scaled run, `memory=true` is part of the fixed recipe inherited from the first ablation
sweep. BABILong should therefore compare `rope`, `nope`, `polar`, and `wall` under the same
memory-on recipe and the same fine-tune protocol. It is not a memory-on/off experiment.

Keep the fine-tune protocol simple: fine-tune on the 0K, 1K, and 2K subsets only, then
evaluate at longer lengths. This tests task-format learning at short/normal context lengths
plus length generalization from the pretrained attention type.

The public BABILong length configs are small. The 2K config is 1000 rows total across QA
types, not 1000 rows per QA type. In the Hugging Face layout used by this harness, each
task split has about 100 rows for a given length. The row split must therefore be per QA
type. Do not report results on rows used for fine-tuning.

This is an internal controlled probe, not leaderboard replication. Because it uses a custom
train/validation/test row split and a 2K-only fine-tuning protocol, the BABILong results are
comparable across our attention-type variants but not directly comparable to the BABILong paper
or external submissions. Leaderboard comparison would require matching the official data
generation, prompts, fine-tuning curriculum, task set, and evaluation settings.

Protocol:

1. Use BABILong `qa1..qa10` on `0k`, `1k`, and `2k` so the fine-tune has about 2400 train
   examples total while preserving per-task reporting.
2. Create a deterministic row split per QA type, for example rows `0..79` train, `80..89` val,
   and `90..99` test, applied independently for each train length.
3. Fine-tune each selected scaled checkpoint on only the `0k`/`1k`/`2k` train rows.
4. Pad shorter sequences to the fine-tune sequence length, but mask padding out of both loss
   and attention. Padding tokens must not contribute labels or usable context.
5. Select checkpoints and stopping rules using only the `0k`/`1k`/`2k` val rows.
6. Report results only on the held-out test row IDs. Use the same test row IDs at
   0k, 1k, 2k, 4k, 8k, 16k, 32k, 64k, and 128k.
7. Use the same task set, optimizer, token budget, batch-token budget, seed, and stopping rule
   for every attention type.
8. Report both macro-average across tasks and the per-task matrix.
9. Keep non-fine-tuned open-source models out of the main BABILong comparison table.

```bash
python -m benchmarks.run --benchmark babilong --model checkpoints/<fine_tuned_run_id> \
    --tasks qa1 qa2 qa3 qa4 qa5 qa6 qa7 qa8 qa9 qa10 \
    --lengths 0k 1k 2k 4k 8k 16k 32k 64k 128k --samples 10 \
    --max_num_seqs 16 --out benchmarks/logs/babilong_<run_id>.log --strict
```

For leaderboard-parity prompts, install `babilong`; otherwise the harness uses built-in
minimal prompts and logs that choice.

## Next Implementation Step

To complete the quality-control group without confounds, add a direct-forward loglikelihood
adapter for Atma checkpoints. That unlocks LAMBADA and multiple-choice tasks without using
generation or instruction prompts. The generation adapter should remain limited to retrieval
and BABILong-style exact-answer tasks.

## Files

| File | Role |
|---|---|
| `model.py` | `EvalModel` adapter over `inference.LLM.generate()`; loads `config.json` and checkpoint weights |
| `babilong.py` | BABILong prompt formatting, generation, and answer scoring |
| `retrieval.py` | Synthetic passkey + NIAH over length x depth grids |
| `run.py` | CLI dispatcher and structured logs |

The log format mirrors the ablation style with `===BABILONG_RESULTS_JSON===` or
`===RETRIEVAL_RESULTS_JSON===` blocks for downstream parsing.
