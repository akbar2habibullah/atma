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

## Results and interpretation

The completed `hl:256` re-evaluation caps only the largest parameter-only gamma
layer-head in each final checkpoint. All 15 requested jobs completed without an
OOM. The table compares the clamped runs in
`results/re_evaluation/run-summary.json` with the pinned untouched results in
`benchmarks/logs/atma_10b/benchmark_matrix.json` and
`benchmarks/logs/babilong_2k_ft/benchmark_matrix.json`.

| Model | Downstream mean (%) | Mean BPB at 256K | Retrieval at 256K, token / exact (%) | BABILong at 256K (%) |
|---|---:|---:|---:|---:|
| NoPE | 45.15 -> 45.10 | **8.097 -> 1.595** | 0.60 / 0.00 -> 0.93 / 0.00 | **0 -> 39** |
| Polar | 43.29 -> 43.25 | **1.826 -> 1.501** | **34.37 / 9.00 -> 47.07 / 16.33** | **28 -> 42** |
| RoPE | 44.60 -> 44.66 | 2.512 -> 2.460 | 0.00 / 0.00 -> 0.00 / 0.00 | 14 -> 11 |

Retrieval entries average the synthetic and FinePDFs suites, passkey and NIAH,
and all three tested depths. BPB averages FinePDFs, PG-19, and Proof-Pile.
Downstream is the mean primary metric across the eight 2K base-model tasks.

The selected zero-input operating point survives BABILong fine-tuning almost
unchanged:

| Model | Base half-life (tokens) | BABILong-fine-tuned half-life (tokens) |
|---|---:|---:|
| NoPE | 21.01M | 20.86M |
| Polar | 3.07M | 3.05M |
| RoPE | 682 | 680 |

The results support the following interpretation:

- **Excessive learned retention is a major causal mediator of NoPE and Polar
  degradation.** An inference-only intervention on one recurrent head changes
  neither checkpoint weights nor attention, yet largely removes NoPE's
  long-document collapse and improves Polar's likelihood, retrieval, and
  adapted reasoning.
- **NoPE's state stability and exact retrieval are distinct problems.** Its
  mean 256K BPB falls from 8.097 to 1.595 and BABILong rises from 0% to 39%, but
  exact retrieval remains 0%. The gamma outlier is therefore a major cause, not
  a complete explanation of NoPE's extrapolation limits.
- **Polar's promoted checkpoint is materially weakened by its outlier.** At
  256K, target-token retrieval rises by 12.70 points, exact retrieval by 7.33
  points, and BABILong by 14 points. Its mean BPB degradation relative to 2K
  falls from about 1.26x to 1.03x.
- **RoPE is a useful negative control.** Its much smaller 682-token operating
  point does not resemble the million-token outliers. Capping it at 256 does
  not restore retrieval and slightly reduces 256K BABILong, so the intervention
  is not a universal benchmark booster.
- **Ordinary downstream quality is effectively unchanged**, but the cap is not
  free: 2K teacher-forced retrieval declines for some model/suite combinations,
  especially RoPE. A fixed 256-token ceiling should not be presented as a
  generally optimal deployment setting.

This experiment identifies the checkpoint mechanism, not its origin. It does
not establish that hardware, reduction order, seed, or any particular optimizer
event caused the outlier, nor does it validate training with a bounded-gamma
parameterization. The broad re-evaluation ran only the clamped condition and
uses the repository's pinned archived baselines; the earlier clamp sweep is the
paired same-process intervention. Rerun with `--paired` before making a strict
paired benchmark claim.

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

To re-benchmark all three matched attention variants on downstream tasks,
synthetic and real-text retrieval, fixed-target long-document BPB, and BABILong
with a fixed 256-token runtime half-life ceiling, run:

```bash
python -m gamma_diagnostics.rebenchmark_all \
  --models nope polar rope \
  --benchmarks base retrieval longdoc babilong \
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

## Reporting guidance

Report the exact targeted block/head and runtime ceiling. Describe the result as
a selective inference-time intervention, not as a retrained or generally fixed
model. The experiment does not show that the outlier is the only cause, that the
same cap should be used during training, or that every checkpoint benefits from
clipping.
