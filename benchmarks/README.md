# Benchmarks (scaled-up evals for final candidates)

Harness for downstream generation benchmarks at the 10B-token scale. It uses the
production inference interface (`inference.LLM.generate`, see [docs/inference.md](../docs/inference.md)).

The current benchmark adapter supports checkpoints whose `config.json` has
`attn_type="polar"`. It passes the saved `AtmaConfig` into the paged engine and resolves
checkpoint directories to `weights.pt`. Non-polar ablation checkpoints are rejected with
`--strict` because the serving model is the polar + Canon + Titans path.

## BABILong

[BABILong](https://arxiv.org/abs/2406.10149) embeds the 20 bAbI reasoning tasks inside
PG-19 background text: reasoning-in-a-haystack.

This is a fine-tuned capability probe, not a zero-shot quality eval. The decisive protocol:

1. Fine-tune a final candidate on bAbI qa1, then qa1-5, with a length curriculum.
2. Evaluate across 0k to 64k realistic lengths, with 128k+ as aspirational.
3. Compare `mem_enabled=True` vs `False` after the same fine-tune recipe.

```bash
python -m benchmarks.run --benchmark babilong --model checkpoints/<run_id> \
    --tasks qa1 qa2 --lengths 0k 1k 2k 4k 8k 16k 32k 64k --samples 100 \
    --max_num_seqs 16 --out benchmarks/logs/babilong_<run_id>.log --strict
```

For leaderboard-parity prompts, install `babilong`; otherwise the harness uses built-in
minimal prompts and logs that choice.

## Retrieval

Synthetic passkey + NIAH over a length x depth grid. Prompts are built with the GPT-2
tokenizer so each cell has an exact token length and controlled needle depth.

```bash
python -m benchmarks.run --benchmark retrieval --model checkpoints/<run_id> \
    --tasks passkey niah --lengths 1k 2k 4k 8k 16k 32k 64k 128k \
    --depths 0 0.25 0.5 0.75 1 --samples 50 \
    --max_num_seqs 16 --out benchmarks/logs/retrieval_<run_id>.log --strict
```

Add `--haystack codelion/finepdfs-100M` for real-text NIAH instead of synthetic filler.
Use `--max_model_len`, `--max_num_batched_tokens`, and `--max_num_seqs` to control memory.

## Files

| File | Role |
|---|---|
| `model.py` | `EvalModel` adapter over `inference.LLM.generate()`; loads `config.json` and checkpoint weights |
| `babilong.py` | BABILong prompt formatting, generation, and answer scoring |
| `retrieval.py` | Synthetic passkey + NIAH over length x depth grids |
| `run.py` | CLI dispatcher and structured logs |

The log format mirrors the ablation style with `===BABILONG_RESULTS_JSON===` or
`===RETRIEVAL_RESULTS_JSON===` blocks for downstream parsing.
