# ATMA benchmark suite

This directory contains the post-training benchmark pipeline for the five promoted 10B-token
checkpoints. Training, intrinsic evaluation, downstream likelihood scoring, generation, and
task adaptation remain separate so a benchmark cannot silently change the training comparison.

The automatic pipeline covers every no-finetune group below. BABILong is also implemented, but
remains opt-in because a meaningful comparison requires the same task fine-tuning recipe for
every architecture.

The generation adapter supports Polar, NoPE, RoPE, and Raven checkpoints. Polar uses the
production `inference.LLM`; the other architectures are selected from `config.json` and use
the isolated minimal forks in `baseline_inference/`. Checkpoint directories resolve to
`weights.pt` without conversion.

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
| Report NoPE/Polar/RoPE as the matched Atma+Muon comparison and Raven models separately | Raven also changes model family and optimizer, so it is not an attention-only control |

## Benchmark Matrix

| Group | Benchmarks | Status | Purpose | Confound control |
|---|---|---|---|---|
| Intrinsic long-context LM | FinePDFs clean PPL, FineWeb-Edu junk PPL, induction needle loglikelihood | Covered by `scaled_ablation.evaluate` | Direct architecture signal during the scaled sweep | Direct forward, same tokenizer/data, OOM recorded per length |
| Exact retrieval generation | Passkey, synthetic NIAH, real-text NIAH | Implemented in `retrieval.py` | Served-model retrieval check | GPT-2 token exact lengths, greedy decode, exact-match short answers |
| Long reasoning after adaptation | BABILong qa1-10 | Implemented harness, requires task fine-tune | Measures each 10B attention type's adapted long-context potential | Same fixed recipe and fine-tune protocol for each attention type, with per-task reporting |
| Base LM quality controls | LAMBADA, HellaSwag, PIQA, WinoGrande, ARC-E, ARC-C, OpenBookQA, BoolQ | Implemented in `base_tasks.py` + `scoring.py` | Check that long-context architecture did not destroy normal LM capability | Multiple-choice/loglikelihood only, no chat prompts |
| Long-doc LM controls | PG-19, Proof-Pile, FinePDFs bits-per-byte | Implemented in `longdoc.py` | Domain/generalization check at long lengths | Same fixed target at every context length, no decoding |
| Serving performance | Prefill tok/s, decode tok/s, peak memory, max context before OOM | Implemented in `serving.py` | Practical deployment tradeoff | Same hardware, batch, exact token prompt, and context settings |

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

Add `--haystack codelion/finepdfs-1B` for real-text NIAH instead of synthetic filler.
Use `--max_model_len`, `--max_num_batched_tokens`, and `--max_num_seqs` to control memory.

## BABILong

[BABILong](https://arxiv.org/abs/2406.10149) embeds the 20 bAbI reasoning tasks inside PG-19
background text. Treat it as a fine-tuned adaptation probe for each 10B attention type, not a
zero-shot base-model quality benchmark.

In the scaled run, `memory=true` is part of the fixed recipe inherited from the first ablation
sweep. Keep it fixed within the matched NoPE/Polar/RoPE comparison and use the same fine-tune
protocol for every promoted checkpoint. Raven results must remain a separate model-family/AdamW
comparison rather than being presented as a pure attention ablation.

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

## Base LM quality controls

`scoring.py` loads the training checkpoint directly and computes the checkpoint's exact
soft-capped logits. It does not use generation, an instruction prompt, or a serving-engine
approximation. LAMBADA uses greedy exact final-word prediction. HellaSwag, PIQA, WinoGrande,
ARC-Easy, ARC-Challenge, OpenBookQA, and BoolQ use conditional choice likelihood; raw and
token-length-normalized accuracies are both recorded where applicable.

```bash
python -m benchmarks.run --benchmark base --model checkpoints/<run_id> \
    --tasks lambada hellaswag piqa winogrande arc_easy arc_challenge openbookqa boolq \
    --batch_size 8 --scoring_max_length 2048 \
    --out benchmarks/logs/base_<run_id>.log --strict
```

`--limit 100` is useful for plumbing checks. Reported results should use complete splits and the
dataset commits pinned by the pipeline.

## Long-document likelihood

For every document, the evaluator fixes one target span immediately after the maximum context
position. Every requested context length predicts that same target. This avoids the
content/position confound created when each length scores a different part of a document.

```bash
python -m benchmarks.run --benchmark longdoc --model checkpoints/<run_id> \
    --datasets pg19 proof_pile finepdfs \
    --lengths 2k 8k 32k 64k 128k 256k \
    --target_tokens 256 --num_docs 8 \
    --out benchmarks/logs/longdoc_<run_id>.log --strict
```

The result records NLL, perplexity, bits/byte, dataset commits, and a SHA-256 hash for each token
fixture. If no qualifying document is found within `--max_scan`, that condition is recorded; the
runner does not concatenate unrelated documents.

## Serving performance

The serving sweep reinitializes the engine at every context length, uses an exact-length token
prompt, resets peak CUDA memory statistics, and records OOM cells.

```bash
python -m benchmarks.run --benchmark serving --model checkpoints/<run_id> \
    --lengths 2k 8k 32k 64k 128k 256k \
    --decode_tokens 32 --serving_samples 1 --max_num_seqs 1 \
    --out benchmarks/logs/serving_<run_id>.log --strict
```

Run serving comparisons on the same physical GPU with no competing workload. Throughput from a
different GPU or software stack is not an architecture-only comparison.

## Files

| File | Role |
|---|---|
| `model.py` | `EvalModel` architecture router; loads `config.json` and checkpoint weights |
| `babilong.py` | BABILong prompt formatting, generation, and answer scoring |
| `retrieval.py` | Synthetic passkey + NIAH over length x depth grids |
| `scoring.py` | Checkpoint-exact conditional loglikelihood scorer |
| `base_tasks.py` | LAMBADA and zero-shot multiple-choice scorers |
| `longdoc.py` | Fixed-target long-document NLL/PPL/BPB |
| `serving.py` | Throughput, memory, and maximum-context sweep |
| `aggregate.py` | Structured logs to tidy JSON/CSV conversion |
| `run_pipeline.py` | Checkpoint/dataset pinning, job matrix, resume, and smoke gating |
| `run.py` | CLI dispatcher and structured logs |

Each evaluator writes a benchmark-specific `===..._RESULTS_JSON===` block for aggregation.

## Promoted 10B-token checkpoint pipeline

`run_pipeline.py` contains the five supplied Hugging Face repositories. It resolves checkpoint
and dataset revisions to immutable commits, validates architecture and checkpoint files, and
launches every model/benchmark job in a fresh process so GPU or compiler state cannot leak
between architectures.

Run the cheap retrieval gate first:

```bash
python -m benchmarks.run_pipeline --stage smoke --gpu 0
```

Run the complete automatic matrix—retrieval, base LM tasks, long-document likelihood, and
serving performance—without the smoke gate:

```bash
python -m benchmarks.run_pipeline --stage full --gpu 0
```

Or run the gate and complete matrix together:

```bash
python -m benchmarks.run_pipeline --stage all --gpu 0
```

With `--stage all`, a model whose smoke job fails or OOMs is not advanced; other models continue.
Individual groups can be run or resumed with:

```bash
python -m benchmarks.run_pipeline --stage retrieval --gpu 0
python -m benchmarks.run_pipeline --stage base --gpu 0
python -m benchmarks.run_pipeline --stage longdoc --gpu 0
python -m benchmarks.run_pipeline --stage serving --gpu 0
```

Useful cost controls include `--models polar nope rope`, `--base_limit 100`, `--samples 10`,
`--num_docs 2`, and the group-specific length arguments. `--stage pilot` runs retrieval with 10
samples per cell. Defaults are lengths `2K, 8K, 32K, 64K, 128K, 256K`, retrieval depths 10%,
50%, and 90%, and one concurrent generation sequence.

Every exact job configuration has a fingerprint. Completed fingerprints are skipped on resume;
`--rerun` creates a new attempt without overwriting the prior log. Failed/OOM jobs are recorded
and the pipeline continues unless `--fail_fast` is set. Set `HF_TOKEN` if authentication is
required.

The default output directory, `benchmarks/logs/atma_10b/`, contains:

| Output | Contents |
|---|---|
| `checkpoint_manifest.json` | Immutable checkpoint/dataset commits and validation metadata |
| `pipeline_summary.json` | Job status, elapsed time, paths, and parsed result blocks |
| `benchmark_matrix.json` | Aggregated tidy records from all structured logs |
| `benchmark_matrix.csv` | The same matrix for analysis and plotting |
| `*.log` / `*.console.log` | Structured result and complete console output per job |

Aggregation can also be rerun independently:

```bash
python -m benchmarks.aggregate --log_dir benchmarks/logs/atma_10b
```

These checkpoints are base models. BABILong is intentionally excluded from `--stage full`; run
it only after applying the same task-fine-tuning protocol to every comparison model.
