# Gamma extrapolation diagnostic

This directory separates three questions that should not be conflated:

1. Do checkpoint parameters contain extreme zero-input gamma operating points?
2. Does selectively capping the **runtime** gamma of those heads causally improve
   long-context behavior without damaging short-context behavior?
3. If the pilot recovers, does that recovery reproduce on the repository's
   independent retrieval and long-document benchmarks?

Nothing is clamped by default. Clamp specs are opt-in, affect only listed
transformer block/head pairs, do not rewrite checkpoint tensors, and are removed
after each sweep condition.

## 1. Parameter-only scan

From the repository root:

```bash
python -m gamma_diagnostics.inspect_parameters
```

This scans all known 10B-token memory checkpoints. It does not accept a sequence
count because it never runs the model. Reports go to
`gamma_diagnostics/results/parameters/`.

For local or custom checkpoints:

```bash
python -m gamma_diagnostics.inspect_parameters \
  --checkpoint /content/checkpoint-a \
  --checkpoint /content/checkpoint-b
```

The old `scripts/inspect_gamma_horizon.py` path remains as a compatibility shim.

## 2. Causal clamp sweep

A small Colab/T4 pilot for Polar is:

```bash
python -m gamma_diagnostics.sweep \
  --models ChavyvAkvar/atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0 \
  --caps p90 p99 hl:256 hl:512 \
  --lengths 2048 16384 65536 \
  --num-eval-docs 4 \
  --num-needle-trials 8 \
  --output gamma_diagnostics/results/polar_clamp_sweep.json
```

The untouched baseline and every cap use the same model instance, documents,
needle construction, and SDPA backend. By default only the single layer-head
with the largest parameter-only gamma logit is capped. Use
`--num-target-heads 2` only as a separate ablation.

When `--models` is omitted, the sweep also includes L4 NoPE as a negative
control alongside both L40S NoPE runs, Polar, and RoPE. A convincing hardware-
divergence result should be selective: saturated runs improve while the L4
control does not gain from a cap.

- `p90`/`p99` cap the selected runtime logits at the checkpoint-local percentile
  of all learned zero-input layer-head logits.
- `hl:256`/`hl:512` impose an absolute constant-gamma half-life ceiling.
- The cap is an upper bound. Token-conditioned gamma may remain lower.

The output keeps the complete baseline, all interventions, selected targets,
resolved caps, OOM/completion counts, and a recommendation. A recommendation is
made only if longest-distance needle CE improves and shortest-length clean loss
does not regress by more than `--short-loss-tolerance` (default 0.05 nats).
The CE improvement must be at least `--min-needle-improvement` (also 0.05 nats
by default), preventing a tiny noisy change from triggering promotion.

For a less noisy confirmation before benchmark promotion, increase to at least
`--num-eval-docs 16 --num-needle-trials 32` and include 64K/128K if the GPU fits.

## 3. Paired benchmark re-evaluation

To re-benchmark all three matched attention variants on the complete downstream
and BABILong suites with a fixed 256-token runtime half-life ceiling, run:

```bash
python -m gamma_diagnostics.rebenchmark_all \
  --models nope polar rope \
  --benchmarks base babilong \
  --max-half-life 256 \
  --execute
```

The command resolves the pinned base and BABILong-fine-tuned checkpoints from
the existing benchmark manifests. Each final checkpoint is scanned separately,
its largest parameter-only gamma layer-head is selected, and the runtime cap is
applied without modifying checkpoint tensors. Completed jobs are resumable with
the same command. Results, clamp specs, parameter scans, and the run manifest are
written under `gamma_diagnostics/results/re_evaluation/`.

The repository already contains the corresponding untouched benchmark logs, so
the command above runs only the clamped condition. Add `--paired` to rerun both
baseline and clamped conditions in the same environment. Use
`--babilong-lengths` or `--base-limit` for a cheaper smoke test.

The BABILong jobs use the already fine-tuned checkpoints and apply the cap only
during held-out evaluation; they do not fine-tune again. All jobs use the
checkpoint-exact direct scorer because paged serving does not support the
diagnostic hook.

### Re-evaluate a clamp selected by a pilot sweep

First dry-run the benchmark plan generated from a qualifying sweep:

```bash
python -m gamma_diagnostics.re_evaluate \
  --sweep gamma_diagnostics/results/polar_clamp_sweep.json \
  --benchmarks retrieval longdoc \
  --lengths 2k 16k 64k \
  --samples 20
```

When a sweep contains multiple checkpoints, also pass the exact `--model` key.
The command refuses to prepare a plan if the predeclared recovery criterion was
not met. It writes the chosen clamp JSON and paired baseline/clamped commands to
`gamma_diagnostics/results/re_evaluation/`.

After reviewing the plan, append `--execute`. Both conditions use the same seed
and benchmark settings. This route deliberately uses `DirectScorer`, the
checkpoint-exact correctness path. Paged serving is excluded because changing
inference implementation while testing a behavioral hypothesis would introduce
another variable.

You can also apply a reviewed clamp spec directly:

```bash
python -m benchmarks.run \
  --benchmark retrieval \
  --model /content/polar-checkpoint \
  --gamma-clamp gamma_diagnostics/results/re_evaluation/polar.gamma-clamp.json \
  --tasks passkey niah --lengths 2k 16k 64k --samples 20
```

## Interpreting a positive result

A clamp recovery would support the claim that the saturated retention head is
causally involved in degradation. It would not by itself show that it is the
only cause, or that the same cap should be used during training. Require paired
benchmark replication, inspect short-context regressions, and report the exact
target and cap rather than describing the whole model as "fixed."
