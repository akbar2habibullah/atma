# Future Directions

Tracking doc for potential development beyond the current architecture
([Polar Attention](POLAR_ATTENTION.md) + [Titans MAG memory](TITANS_MEMORY.md)). These are
**not committed work** — they are ideas, framings, and falsifiable experiments parked here so
they survive until there's bandwidth to pursue them.

> Most of the experimental items are **blocked on the 120-way ablation sweep** (5×2×2×2×3,
> see [ablation/README.md](../ablation/README.md)) currently running (~2–3 weeks). Do not start
> new diagnostic interventions until that grid lands — they are *post-sweep* candidates.

---

## Research sequencing: ATMA paper vs. hierarchical-memory sequel

Keep the scope split deliberately. The current ATMA paper should finish the Polar + Titans story
before attempting the extreme-context hierarchy.

**Paper 1 - ATMA.**

- Architecture: Polar Attention + Titans gated-delta memory.
- Baselines: RoPE, NoPE, Raven/linear-memory-style baselines, open-weight reference models, and
  the industrial ablation sweep for the pre-scaled recipe.
- Scale: 370M-class model, 10B-token training target, `seq_len=2048`.
- Evaluation: extrapolation up to `64x` training length where feasible, realistic downstream tasks
  appropriate for the 370M/10B scale, and enough theoretical formalization to make the
  length-invariance claim precise.

**Paper 2 - hierarchical memory for extreme context.**

- Same base scale unless evidence says otherwise: 370M parameters, 10B training tokens,
  `seq_len=2048`.
- Add YOCO-style KV-bank sharing and test `N in {1, 2, 4}`. Treat this ablation as an intrinsic
  performance check up to the normal 64x extrapolation regime; it is not the main scientific grid.
- Pick the best non-harmful shared-bank setting, likely `N=2` or `N=4`, for the extreme-context
  hierarchy experiments.
- Add HOLA-style sparse KV filtering following the ablation ladder in section 6.
- Add hierarchical memory alignment up to `512x` training length (`2048 * 512 = 1,048,576` tokens).
- Initial sparse budgets: capped top-P bank at `4096` or `8192` slots; per-layer top-K read around
  `256` or `512` slots.
- Use chunked long-sequence simulation during pretraining so the model sees extreme memory age
  without explicit long-context mid-training or post-training.
- Open-source model comparisons are illustrative anchors, not the main claim.

The sequel's intended claim is proof-of-concept: an end-to-end pretrained memory hierarchy can be
trained to support extreme extrapolation out of the box, without relying on a separate explicit
long-context adaptation phase.

**Cost discipline.** This sequel should be cheaper than the first ATMA paper if it avoids another
Cartesian ablation grid. Paper 1 is roughly `120+` 1B-token ablations plus `4+` 10B-token scaled
runs, i.e. `160B+` trained-token equivalent. The sequel target should be closer to:

- YOCO pilot curve: `5` short 1B-token runs for `N in {1, 2, 3, 4, 6}`.
- Sparse hierarchy probes: a few 1B-token filtering/alignment variants.
- Full proof runs: `2-4` selected 10B-token runs, likely around the chosen `N=2` or `N=4` setting.

Even with `20-50%` wall-clock overhead from chunked alignment rehearsal, this should stay around
`40B-80B` effective-token equivalent if scoped tightly, roughly `1/2` to `1/3` of the first paper's
training cost. The important rule is to pick winners sequentially:

```text
YOCO curve -> choose N
sparse-cache ladder -> choose top-P/top-K
alignment schedule -> choose interval/length
full 10B proof runs -> evaluate extreme extrapolation
```

Do not run the full cross product `N x top-P x top-K x alignment-loss x interval x chunk-size x
eval-length`; that would recreate the first paper's cost.

**Shape caveat.** The canonical 16-layer ATMA has only 4 attention layers, so YOCO `N=4` collapses
to one shared bank for all attention layers and is not a very informative sharing curve. For the
sequel's YOCO ablation, prefer an illustrative 48-layer shape with `hidden_size=512`,
`head_dim=64`, and 12 attention layers. This is about `257.9M` parameters with Polar+Titans and
allows `N in {1, 2, 3, 4, 6}`. Keep the canonical ATMA result as an architectural reference row,
not a strict same-shape ablation. A 96-layer / 24-attention-layer variant would be about `464M`
parameters with Polar+Titans, but should be deferred as a scaling follow-up because it changes
depth, optimization, wall-clock cost, and extrapolation behavior at the same time as YOCO sharing.

---

## Systems status

Dense equal-length and grouped heterogeneous fresh-prefill routes are implemented in the paged
engine. Both retain the paged K/V cache and recurrent state tables; chunked/prefix prefill remains
on the exact per-sequence fallback, and decode remains CUDA-graph captured. Implementation,
measurements, rejected fusion experiments, and remaining systems work are maintained in
[kernel.md](kernel.md).

---

## 1. The polar read flaw, stated as a regression problem

A framing that unifies the diagnosis and points at the open work.

- **Recurrence = optimization** (Miras / ATLAS line, [arXiv 2504.13173](https://arxiv.org/abs/2504.13173),
  [arXiv 2505.23735](https://arxiv.org/abs/2505.23735)): the memory state `M` is a *parameter*
  fit by online SGD on `‖v − Mk‖²` under retention. Finite-capacity **parametric** regression.
  Our gated-delta `TitansMemory` is exactly one cell of this taxonomy (ℓ2 bias + γ retention +
  η=0 GD).
- **Softmax attention = non-parametric regression** (Nadaraya–Watson kernel smoother): output
  `Σ wᵢvᵢ = Ê[v|q]`, a soft-argmin over keys with **fixed bandwidth** (temperature). The
  "batch" is the context, and it **grows with position** — query `t` regresses over `t` keys.
- **Polar = the inferential / calibration layer of that same smoother.** It discards the scale
  of the estimate (keeps direction) and reports two functionals of the weight distribution:
  - `n_eff = 1/Σŵᵢ² = exp(H₂(w))` — the **effective sample size** / perplexity of the read =
    `exp` of the order-2 Rényi (collision) entropy. The conditioning of the soft-argmin.
  - `1 − w_null` — a **significance test** of the best match against an EV noise floor.

**The flaw** ([polar extrapolation diagnosis](POLAR_ATTENTION.md)): a fixed-bandwidth smoother
over a growing batch **over-smooths** — `n_eff` is non-length-invariant (12–37× ramp). Polar's
`1 + softplus(g)·log N` temperature is an **open-loop** bandwidth schedule that *measures* the
right invariant (`n_eff`) but *controls* it with the wrong lever — it does not actually pin
`n_eff`, which is why the ramp persists. The drift hides at the output (`tanh`) while the
operating point leaks downstream as a DC mean-shift → OOD activations.

---

## 2. Closed-loop entropy-targeting read (the principled bandwidth fix)

**Idea.** Replace the open-loop `log N` temperature with a **closed-loop regulator** that pins
the read's effective sample size to a setpoint — a variable-bandwidth / balloon-estimator form
of kernel regression.

- For a target `n_eff*`, solve per-query for the temperature `T(q)` such that
  `n_eff(T) = exp H₂(softmax(scores/T)) = n_eff*`. `n_eff(T)` is monotonic in `T` and bounded
  by the current batch size → a well-posed 1-D root find per query (differentiable, or
  solve-then-straight-through).
- This makes the read **length-invariant by construction** rather than by meta-learned
  approximation. Folds into the existing `polar_temp_null` in [model/blocks.py](../model/blocks.py).

**Falsifiable prediction.** The [eval.py](../eval.py) `--diagnose` `n_eff`-vs-N curve goes **flat**
where it currently shows a 12–37× ramp. That single curve settles whether the closed loop works.

**Caveat.** This fixes the *non-parametric* read's calibration; it does not make it
*bounded-compute* (still O(N)). Compression remains the memory's job — the two are complementary
branches, not substitutes.

---

## 3. Memory as compensation vs. repair (resolve which)

First MAG results showed the Titans memory **dampens the perplexity distribution past train
length even with full (un-windowed) polar attention** (`full` reversed from worst ~3.21 to best
& monotonic ~1.93 @64×). Open question: does the memory **repair** the read flaw or merely
**mask** its symptom?

- **Pure additive compensation** (hypothesis A): the memory is a length-stable gist channel
  (delta state-norm is flat in N; RMSNorm'd readout) that dilutes the mean-shift and gives
  downstream layers an in-distribution signal to lean on. The read's `n_eff` ramp is **unchanged**;
  it is only masked. (Supported by: the effect persists with *full* attention, where windowing
  would have *prevented* the ramp — so the memory compensates, it does not prevent.)
- **Training-time reshaping** (hypothesis B): training *with* memory lets the attention channel
  offload diffuse averaging and specialize to a *lower, flatter* `n_eff` — a partial real repair.

**Test (deferred — needs the memory-trained checkpoints + GPU).** Run [eval.py](../eval.py)
`--diagnose --no_mem` on a memory-trained checkpoint and compare the `n_eff`-vs-N curve to the
memory-on curve and to a no-mem-trained model:
- Ramp **identical** with/without mem at eval, output OOD only when mem ablated → **compensation**
  (flaw intact, masked).
- Ramp **flatter** even with mem off (vs a no-mem-trained model) → **training-time repair**.

Prior: mostly compensation (the additive channel exerts no direct gradient pressure on the
softmax weights), with possibly a little reshaping from the offload effect.

**Division-of-labor reminder:** the linear gated-delta memory is a lossy GIST (cap ~`d_k`) — it
fixes *diffuse perplexity* at length but cannot exact-recall a needle. Pinpoint retrieval stays
on full-polar + the distractor. So compensation is *sufficient for perplexity workloads* and
*structurally insufficient for distant-retrieval workloads* — consistent with the
workload-dependent verdict.

---

## 4. Wall Attention (Tilde Research) - incompatible attempted contender

**Protocol verdict (2026-07-07):** Wall Attention remains implemented for diagnostics, but it is
excluded from the fair Atma grid. Under the standardized hybrid architecture plus native Atma Muon
split, the baseline Wall cell improved early and then regressed badly; the comparable Raven native
Muon control shows the same qualitative instability. Treat this as a protocol incompatibility, not
as a general negative result on Wall, because the official Wall recipe uses per-head Muon/MuonSplit
and Aurora-style training details. Raven now replaces Wall as the stronger outsider baseline through
[raven_baseline](../raven_baseline/), with the protocol difference stated explicitly.

[Blog](https://blog.tilderesearch.com/blog/wall-attn) ·
[Code](https://github.com/tilde-research/wall-attention-release). Data-dependent **diagonal
forget gates** lifted into softmax attention via the *induced action* `Ã φ(x) := φ(Ax)`,
yielding `w_ij = exp(qᵢᵀ diag(∏ g_r) kⱼ)`. Factorizes to a per-channel rescale
`q̃ = exp(P)⊙q, k̃ = exp(−P)⊙k` (with `P` the log-gate prefix sum) feeding standard attention.
Generalizes RoPE/FoX/PaTH; 4k→160k+ zero-shot; FA-compatible Triton kernels (WallDecode ≈ FA3).

**Where it sits relative to our stack (orthogonal axes, composable):**

| Job | Owner | Wall? |
|---|---|---|
| Scores / position (into softmax) | RoPE/NoPE/polar-scores | **Wall replaces this** |
| Normalization / cardinality (`n_eff`, null sink) | **Polar** | untouched |
| Compression / bounded state | **Titans memory** | untouched (Wall is still O(N)) |

**Why it's interesting for us:**

1. **Alternative resolution to our window-vs-retrieval dilemma.** Wall's *Bimodal Channels*
   finding — channels split into **static memory** (retention=1.0, full-attention, distant
   recall) and **dynamic forgetting** (snap shut at semantic boundaries, recency) — solves
   *inside attention, per-channel* the exact tradeoff that motivated the Titans memory. It is
   also **independent confirmation of Titans [Finding 2](TITANS_MEMORY.md)** (per-head γ as a
   learned retention spectrum), at finer (per-channel, in-read) granularity.
2. **A concrete adaptive-bandwidth kernel.** In the regression lens (§1), Wall is a
   **non-stationary kernel** whose bandwidth shrinks along the time axis at a learned,
   content-dependent, per-channel rate — i.e. one realization of the variable-bandwidth fix from
   §2, but on the *score* side. It is *adaptive* but does **not regulate `n_eff` to a setpoint**
   (static channels still ramp). Complementary to entropy-targeting (§2), not a substitute.
3. **FlashAttention compatibility.** Wall is a pre-softmax q/k rescale → keeps the easy
   inference path, where polar's null column + weight-level reductions sacrifice it (root of the
   deferred [inference port](POLAR_ATTENTION.md) pain). If the port stays painful, "Wall-NoPE
   alone" is an attractive FA-compatible baseline.

**Historical candidate cells, now superseded by the incompatibility verdict:**

- `attn_type += wall` — drop-in NoPE positional, FA-compatible baseline (the ablation harness
  already has rope/nope/polar as an axis).
- **Wall-scores → polar-normalization** — does Wall's adaptive kernel reduce polar's `n_eff`
  ramp on the `--diagnose` probe? (Note: Wall+polar still inherits polar's non-FA inference.)
- **Wall vs Titans on the window-vs-retrieval cell** — does per-channel gating make the gist
  memory partially redundant *for quality*, while the memory still wins on compute/KV-cache?

**Diagnostic status.** `attn_type="wall"` is implemented in
[train/model.py](../train/model.py) (`CausalSelfAttention`, `pos="wall"`): **keeps canon** (so it's
the matched comparison to `nope` - isolates the gating; all params used -> no Muon issue), adds a
bias-free per-channel gate-logit projection `l = W_g*x + b` initialized with `b=6`, maps it through
`logsigmoid` and Tilde's soft clamp into a bounded log-decay `g in [-0.87, 0]`, then applies Wall's
score `q_i*k_j*exp(P_i-P_j)` per channel via the stable tiled rescale into attention. The active CUDA
training path uses the selected Wall Triton kernel (`ATMA_WALL_IMPL=local|upstream|auto`), and a
CPU/missing-kernel fallback remains for development. Explicit Wall configs can still be generated
with `--attn_types wall`, but they are diagnostic-only and should not be counted in the fair grid.

**Training-instability note (2026-06-30).** After fixing the invalid raw-gate parameterization,
an upstream-kernel run no longer immediately NaNs, but was reported to hover between about `5.8`
and `7.0` loss after `>500` training steps, including regression from `5.8` back toward `7.0`.
Treat Wall in this codebase as numerically unvalidated for pretraining stability until a controlled
run pins the mechanism. This is not evidence that the exact Wall score is inherently explosive:
in exact arithmetic `prod(g) <= 1` only attenuates per-channel dot-product contributions. The risk
is the training implementation and optimization path: tiled `exp(P_i-R)` / `exp(R-P_j)` rescaling,
suffix-cumsum gate gradients, bf16 dot paths, and Muon updates on a gate matrix whose gradients are
long-horizon credit assignments.

**Mechanistic questions to settle before trusting Wall results:**

- Does the gate distribution stay near-open, or does Muon drive a rapid bimodal split into shut and
  static channels? Track retention quantiles `exp(g)`, prefix range `max(P)-min(P)`, and gate-logit
  weight norms per attention layer.
- Is the loss jump preceded by a spike in `w_wall.grad`, reverse-cumsum `dg`, or total grad norm
  before clipping? If yes, test a separate lower LR / AdamW-only group for `w_wall`.
- Does upstream behave differently from the local kernel with the same bounded log-decay inputs?
  If yes, isolate forward parity, `dq/dk/dv`, and `dg` finite differences in the bounded-gate
  regime, not the old signed-random-gate tests.
- Does disabling distractors, MAG memory, or windowing remove the oscillation? If one switch fixes
  it, the instability is an interaction with the surrounding training objective rather than Wall
  attention alone.

**Caveat.** All Wall numbers are single-source (Tilde blog, 1B scale, their benchmarks). The
mechanism is sound and the bimodal result is credible *because* it echoes our own retention
spectrum — but treat "beats RoPE+FoX / SOTA" as promising, not settled, until run in our harness.

---

## 5. YOCO-style shared KV banks for 1M-token serving

**Goal.** Keep a 30B-class ATMA model's KV cache below a sub-15 GB budget at
`1024*1024` context without jumping straight to MLA/TPA/CCA-style latent attention. The simple
variant is YOCO-inspired: multiple attention layers consume the same paged K/V bank while keeping
their own query, polar read parameters, output projection, count projection, and Titans memory
branch.

**Why ATMA may tolerate this better than a dense Transformer:**

- Attention is already sparse in depth (`attention=(i % 4 == 2)`), with LFM2 blocks doing most
  local token mixing between global reads.
- The Titans MAG branch remains layer-local and adds a per-layer contextualization path even when
  K/V is shared.
- We only need modest sharing to hit the serving target; this is not aggressive "one cache for
  every layer" unless the width/depth shape forces it.

**Small-scale ablation shape.** The canonical 16-layer ATMA only has 4 attention layers, so use a
YOCO-informative proof-of-concept shape for the second paper:

```text
hidden_size = 512
head_dim = 64
num_hidden_layers = 48
attention layers = 12
num_attention_heads = 8
num_key_value_heads = 2
params ~= 257.9M with Polar+Titans
```

This enables a real sharing curve: `N in {1, 2, 3, 4, 6}`. Do not start with the 96-layer
`~464M` variant; reserve it for a later scaling run after the 48-layer hierarchy works.

**Sizing target.** For the proposed 30B-ish shape:

```text
hidden_size = 6144
num_hidden_layers = 48
num_attention_heads = 48
num_key_value_heads = 8
head_dim = 128
attention layers = 12
```

At bf16, one distinct 1M-token KV bank costs:

```text
2 * num_kv_heads * head_dim * 2 bytes * 1,048,576 tokens
= 2 * 8 * 128 * 2 * 1,048,576
= 4 GiB ~= 4.295 GB
```

Therefore:

| Distinct KV banks | Sharing group over 12 attn layers | 1M-token KV footprint |
|---:|---:|---:|
| 12 | `N=1` | 48 GiB ~= 51.54 GB |
| 6 | `N=2` | 24 GiB ~= 25.77 GB |
| 4 | `N=3` | 16 GiB ~= 17.18 GB |
| 3 | `N=4` | 12 GiB ~= 12.88 GB |
| 2 | `N=6` | 8 GiB ~= 8.59 GB |
| 1 | `N=12` | 4 GiB ~= 4.29 GB |

**Initial research cell.** Train with `kv_share_group_size=4` on the 12 attention layers, yielding
3 distinct K/V banks and a 1M-token KV footprint of about `12.88 GB` decimal. This is the first
interesting sub-15 GB point and is far less restrictive than full YOCO sharing. If quality is flat,
`N=6` gives the under-10 GB version.

**Implementation sketch.**

- Define sharing over attention-layer order, not raw block index. For the 48-layer target:
  `[2, 6, 10, 14]`, `[18, 22, 26, 30]`, `[34, 38, 42, 46]`.
- Producer layer computes and stores K/V into the shared paged cache bank.
- Consumer layers compute their own Q/gate and run their own polar reduction over the producer
  bank. Keep each consumer's `v_null`, null schedule, length gain, count projection, output
  projection, and Titans memory weights separate.
- Keep Titans memory state tables per attention layer. Sharing the non-parametric K/V cache is the
  experiment; sharing the parametric memory would confound the result and remove the layer-local
  residual channel that makes this plausible.

**Important caveat.** KV sharing primarily reduces cache capacity. It does not automatically reduce
attention read bandwidth if every consuming layer still performs a separate read over the same
bank. Treat the first milestone as longer context / higher concurrency; benchmark decode throughput
separately.

**Ablations to run.**

- `N in {1, 2, 4, 6}` at fixed `hidden=6144`, `heads=48`, `kv_heads=8`, `layers=48`.
- Producer-only K/V vs. cheap learned per-consumer affine on shared K/V before the read.
- Quality: clean-document perplexity, long-context perplexity drift, needle retrieval, and
  `eval.py --diagnose` activation/read statistics.
- Systems: allocated KV blocks, max model length at fixed memory, decode tok/s, and prefill cost.

---

## 6. HOLA-style sparse Polar episodic cache

**Goal.** Turn Polar attention into a bounded sparse long-context path without first designing a
query-time block-sparse indexer. HOLA ([arXiv 2607.02303](https://arxiv.org/abs/2607.02303))
suggests a clean write-side alternative: keep a small exact KV cache of high-surprise tokens and
read it as an episodic memory, while the recurrent state handles compressible gist.

**Sequel target.** For the hierarchical-memory paper, keep the same 370M / 10B-token /
`seq_len=2048` base model and test the sparse hierarchy at up to `512x` training length
(`1,048,576` tokens). Start with capped top-P banks of `4096` or `8192` slots and per-layer
top-K reads of `256` or `512` slots. The point is not a giant ablation grid; it is a proof of
concept that the hierarchy can be pretrained end-to-end for extreme extrapolation.

**L4 feasibility note for 256K/512K alignment.** For the 370M shape (`num_key_value_heads=2`,
`head_dim=128`), bf16 KV storage costs:

```text
2 * kv_heads * head_dim * 2 bytes = 1024 bytes/token/bank ~= 1 KB
```

With YOCO sharing, persistent full-KV storage is therefore not the limiting factor:

| Effective length | `N=2` shared banks | `N=4` shared banks |
|---:|---:|---:|
| 256K | ~512 MB | ~256 MB |
| 512K | ~1.0 GB | ~512 MB |

The L4 constraint is activation memory and full-read compute, not KV storage. The long-horizon pass
must be a streaming rehearsal, not a dense 256K/512K training sequence:

```text
for chunk in 2K/4K/8K chunks:
    update persistent banks/state
    sample layers and anchor queries
    compute full/high-budget teacher reads only for anchors
    compute sparse/top-P/top-K/Titans student reads
    backprop local alignment modules or state only
    detach banks/state and discard chunk activations
```

Initial L4-safe schedule:

- `N=4`, effective length `256K`, top-P cap `4096`, top-K read `256`.
- Run alignment every `20` normal training steps.
- Sample `1` attention layer and `128-256` anchor queries per pass.

Scale only after profiling:

- Try `512K`, `N in {2, 4}`, top-P cap `8192`, top-K read `512`.
- Keep alignment every `10-20` steps only if wall-clock overhead stays acceptable.
- Use a curriculum: `64K/128K` early, `256K` later, `512K` only occasionally.

Expected bottleneck is wall-clock time, not L4 memory. A full dense 512K training pass is out of
scope; a chunked, anchor-local alignment rehearsal is the feasible target.

For ATMA, the conservative version is not to replace the full Polar read. Train both:

```text
full_polar   = Polar(Q, K_all,   V_all)     # teacher / diagnostic full-context read
sparse_polar = Polar(Q, K_cache, V_cache)   # bounded episodic read over retained tokens

out = local_or_lfm2 + titans_memory + gate_full * full_polar + gate_sparse * sparse_polar
```

Then distill the sparse read from the full read. At inference, the deployable long-context mode can
drop or window the full read and keep `TitansMemory + sparse_polar`.

**Why this is cleaner than direct sparse attention.**

- Sparsity is decided on the write path: every query attends densely over the same bounded retained
  cache, instead of carrying a dynamic query-specific sparse index.
- The full Polar path supplies the target behavior during training, so the sparse cache learns what
  parts of full-context Polar actually matter.
- The cache is an exact non-parametric episodic path, complementary to the lossy gated-delta memory.

**Candidate write scores.**

- HOLA residual: `importance_t = beta_t * ||v_t - memory_read(k_t)||`.
- LM surprise: token loss or residual norm at the layer.
- Polar significance: high `1 - w_null`, high sharpness, or large mismatch between full Polar and
  memory read.
- Matched control: same cache size with pure recency eviction.

**Filtering variants.**

- `top-K bank`: keep the highest-importance `K` tokens. This is the simplest fixed-budget baseline.
- `capped top-P bank`: sort by importance mass and keep the smallest set whose cumulative mass
  reaches `P`, with `min_slots <= |cache| <= max_slots`. This lets easy contexts use fewer slots
  and high-entropy contexts retain a broader tail, which is especially useful when one shared KV
  bank feeds multiple consumer layers.
- `capped top-P bank + layer-local top-K read`: store an adaptive candidate pool globally, then let
  each attention layer select its own top-`K_read` slots from that pool using its local Polar
  pre-scores. This separates "what is worth preserving?" from "what does this layer/query need?"
  and may preserve layer diversity under KV-bank sharing.

Treat the layer-local top-K read as a second-stage ablation, not the first implementation. The
first sparse-cache milestone should still be `top-K` or capped `top-P` storage with a dense Polar
read over the retained bank; only add local read filtering if the retained bank is too large or too
noisy.

**Alignment losses.** Align behavior, not raw K/V:

```text
L_dir  = 1 - cosine(direction_sparse, stopgrad(direction_full))
L_mag  = huber(magnitude_sparse - stopgrad(magnitude_full))
L_out  = mse(projected_sparse, stopgrad(projected_full))
L_null = huber(null_conf_sparse - stopgrad(null_conf_full))
```

Weight the loss by full-read confidence, e.g. `stopgrad(1 - w_null_full)`, so the sparse cache does
not waste capacity imitating diffuse low-signal reads.

**Ablation ladder.**

- `ATMA full Polar + Titans memory` (current winner).
- `+ sparse Polar cache` with no alignment loss.
- `+ sparse Polar cache + full-to-sparse alignment`.
- `top-K bank` vs. `capped top-P bank` at fixed maximum cache budget.
- `capped top-P bank + layer-local top-K read` for the shared-KV-bank setting.
- `window/local Polar + Titans memory + sparse Polar cache`.
- `Titans memory + sparse Polar cache only`.

**Success criteria.** The sparse-cache variants should preserve clean perplexity drift, improve
multi-needle and far-needle recall, and keep bounded cache memory at 64K/128K while matching the
full Polar teacher's `--diagnose` direction/count/null statistics on high-confidence reads.

**Sleep-style consolidation framing.** "Language Models Need Sleep"
([arXiv 2606.03979](https://arxiv.org/abs/2606.03979)) frames sleep as consolidating fragile
short-term memory into more stable long-term knowledge. The ATMA version should be softer than
parameter self-modification: consolidate behavior across memory tiers while leaving dense LM
training to the normal token objective.

```text
full Polar KV        = complete wake memory / highest-capacity teacher
capped top-P bank    = consolidated episodic memory
per-layer top-K read = query-local sparse recall
Titans memory        = compressed semantic gist
```

Normal 32K/64K LM training keeps next-token quality and trains the full model. Periodic
long-horizon consolidation can stream an effective 1M-16M sequence in chunks, detach state across
chunks, and apply only local representation-alignment losses:

```text
for chunk in stream:
    run full/high-capacity teacher reads for sampled layers/queries
    run top-P/top-K/Titans student reads from persistent memory state
    align internal read behavior, not output-token distributions
    consolidate cache, optionally update memory state, detach, discard chunk activations
```

This avoids full token-distillation cost. The supervision is inside each layer on small read
tensors (`direction`, `magnitude`, `null/confidence`, projected read), so the consolidation step can
sample layers, heads, and anchor queries rather than computing vocab KL over every token. The
full-vs-sparse asymmetry supplies the anti-collapse signal: the teacher has strictly more context
than the bottlenecked student.

**Deployment-time closed loop.** At inference, alignment can be used without weight updates:

- **Controller-only mode:** compute full-vs-sparse alignment on sampled anchors inside a segment,
  then adapt `P`, `max_slots`, `K_read`, or eviction thresholds until the error is below a budget.
- **State-consolidation mode:** freeze weights but optimize mutable memory state, especially
  Titans `M`, against a gist-only teacher:

```text
M_sleep = argmin_M L_align(memory_read(M), stopgrad(full_or_topP_gist))
                  + lambda_anchor * ||M - M_online||^2
                  + lambda_norm * norm_penalty(M)
```

This turns Titans into a two-timescale memory: the usual gated-delta recurrence writes online, while
a rare segment-boundary optimization step acts like a sparse nonlinear RNN update over the
matrix-valued state. Keep this target gist-only; sharp needles should remain the job of the sparse
KV hierarchy.

**Caveat.** This does not make the full Polar computation cheaper when both paths are enabled.
The systems win only arrives if the full path is a training teacher / quality mode and the sparse
path becomes the long-context deployment path.

---

## 7. SDM-style sparse recurrent memory capacity ladder

**Motivation.** Sparse Delta Memory (SDM, arXiv 2607.07386) is a direct capacity upgrade for the
recurrent side of the stack, not a replacement for the sparse Polar episodic cache. It sparsifies
the Gated DeltaNet/Titans-style fast-weight table into a large explicit slot memory with sparse
PKM-style top-`W` writes and top-`R` reads. That targets the exact weakness noted above: Titans is
a length-stable but lossy gist channel whose capacity is roughly the dense gated-delta state size.
Sparse Polar still owns exact episodic recall; SDM is the candidate for a much larger compressed
gist tier.

For the current 370M-class ATMA shape (`hidden_size=1024`, `head_dim=128`, `8` attention heads,
`4` attention/memory layers), the dense Titans/GDN state is:

```text
state/layer = 8 * 128 * 128 = 131,072 values
state/all 4 memory layers = 524,288 values
```

The full paper-style `H_sdm=1` setting would give:

```text
N_slots = (1024 / 4)^2 = 65,536
value_dim = 1024
state/layer = 67,108,864 values
increase = 512x
```

That is feasible as a systems target but too aggressive as the first research cell. At bf16 it is
about `512 MiB` across the four memory layers before optimizer state and sparse backward working
memory; with a learned `M0`, optimizer treatment can push the persistent training overhead into
the multi-GiB range. The paper also reports SDM training around `1.49x` slower than GDN at 8B
because sparse gathers/scatters are HBM-bound even when FLOPs are matched.

**Budgeted ATMA ladder.** Make `sdm_num_slots` explicit instead of inheriting the paper's full
capacity formula, and sweep state size directly:

| Target | `N_slots` | `value_dim` | State / layer | All 4 layers | bf16 table |
|---:|---:|---:|---:|---:|---:|
| `32x` | `4096` | `1024` | `4.19M` | `16.78M` | `32 MiB` |
| `64x` | `8192` | `1024` | `8.39M` | `33.55M` | `64 MiB` |
| `128x` | `16384` | `1024` | `16.78M` | `67.11M` | `128 MiB` |
| `512x` | `65536` | `1024` | `67.11M` | `268.44M` | `512 MiB` |

Start at `32x` or `64x`, not `512x`. A `32x` state is already a large capacity jump over Titans
while keeping the persistent table small enough that the experiment primarily measures sparse
kernel quality and modeling benefit, not raw memory pressure.

**Initial ablation ladder.**

- `ATMA full Polar + Titans memory` (current winner).
- `ATMA full Polar + SDM-32x`, null `M0`.
- `ATMA full Polar + SDM-32x`, learned `M0`.
- `ATMA full Polar + SDM-64x`, learned `M0`.
- `ATMA full Polar + SDM-128x`, learned `M0`, only if `64x` is stable and useful.
- Defer `SDM-512x` until the sparse memory kernel is proven and the 32x/64x/128x curve shows
  capacity-limited gains.

**Expected overhead envelope.** Treat these as planning numbers until profiled in the ATMA
training loop:

| Target | Likely training-memory overhead | Likely speed overhead with decent kernels |
|---:|---:|---:|
| `32x` | `~0.5-2 GiB` | `10-30%` |
| `64x` | `~1-3 GiB` | `20-40%` |
| `128x` | `~2-5 GiB` | `30-60%` |

Naive PyTorch scatter/gather is likely too slow for meaningful training. Do not read negative
results from an unoptimized kernel as a modeling verdict.

**Evaluation split.**

- Quality: clean-document perplexity drift, long-code/document perplexity by position, and the
  normal short-context validation loss.
- Recall: induction needle, multi-needle/RULER-style retrieval, and failure cases where Titans
  behaves like lossy gist.
- Diagnostics: memory slot utilization, read/write entropy, collision rate, state norm vs length,
  and whether Polar's `n_eff` ramp changes or is merely compensated.
- Systems: persistent table memory, backward working memory, step time, MFU, decode state
  footprint, and prefill/decode throughput.

**Combination with sparse Polar.** If SDM improves diffuse long-context modeling but still misses
sharp needles, combine it with section 6's sparse Polar cache:

```text
out = local_or_lfm2 + sdm_gist + gate_full * full_polar + gate_sparse * sparse_polar
```

This preserves the division of labor: SDM is the high-capacity recurrent gist tier; sparse Polar is
the exact episodic tier; full Polar remains the teacher / diagnostic path until the sparse hierarchy
is strong enough to deploy without it.

---

## 8. Two unifications our project already spans

Context for why these directions fit together:

- **Miras / ATLAS** unify attention ↔ recurrence on the **optimization** axis (memory
  architecture × attentional bias × retention × optimizer). ATLAS's levers against the
  capacity-`d` ceiling — the **Omega rule** (memorize a context window, not one token),
  **feature maps**, **Muon** — are the menu if we ever want the *memory itself* to do pinpoint
  retrieval. The Omega rule is the principled version of the deferred **Step 4** (distractor on
  the memory write path).
- **Wall's induced action** unifies them on the **operator-lifting** axis (a state transition
  `A` acting as `φ(Ax)`).

Our stack already lives in both: a parametric optimizer-defined memory (Miras cell) + a
calibrated non-parametric read (polar) that adds a *fifth* axis those frameworks omit — the
**geometry/cardinality of the read** (`n_eff` + noise-floor calibration).
