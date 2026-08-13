# Future directions (research notes)

> Status: exploratory planning notes preserved for research provenance. Some sequencing language
> and pilot numbers predate the completed ICLR 2027 experiments; use the
> [evaluation guide](../evaluation.md) for current results.

Tracking doc for potential development beyond the current architecture
([Polar Attention](../POLAR_ATTENTION.md) + [Titans MAG memory](../TITANS_MEMORY.md)). These are
**not committed work** — they are ideas, framings, and falsifiable experiments parked here so
they survive until there's bandwidth to pursue them.

> Most of the experimental items are **blocked on the 120-way ablation sweep** (5×2×2×2×3,
> see [ablation/README.md](../../ablation/README.md)) is complete. Treat the sequencing below as
> historical planning rather than current run status. Do not start
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
- Mechanistic checkpoint stress is analysis-only until the primary experiment matrix and
  checkpoint collection are frozen. Do not add a new length regularizer retroactively to the
  Paper-1 grid; use the completed models to sharpen the hypothesis first.

**Paper 2 - hierarchical memory for extreme context.**

- Same base scale unless evidence says otherwise: 370M parameters, 10B training tokens,
  `seq_len=2048`.
- Add YOCO-style KV-bank sharing and test `N in {1, 2, 4}`. Treat this ablation as an intrinsic
  performance check up to the normal 64x extrapolation regime; it is not the main scientific grid.
- Pick the best non-harmful shared-bank setting, likely `N=2` or `N=4`, for the extreme-context
  hierarchy experiments.
- Add HOLA-style sparse KV filtering following the ablation ladder in section 7.
- After the write-side sparse cache is understood, test the query-side Foveal Polar retriever in
  section 9. Treat it as an alternative deployment path first, not another axis in the initial
  HOLA grid.
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
[kernel.md](../kernel.md).

---

## 1. The polar read question, stated as a regression problem

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

**Candidate failure mode** ([polar extrapolation diagnosis](../POLAR_ATTENTION.md)): a
fixed-bandwidth smoother can over-smooth when a growing context contributes mostly irrelevant
keys. Polar's `1 + softplus(g)·log N` temperature is an **open-loop** bandwidth schedule, so its
operating point can still move with length and leak downstream through branch-energy,
direction/covariance, count-channel, or DC shifts.

The checkpoint-stress pilot in section 4 weakens the stronger claim that a rising pooled `n_eff`
is itself the flaw. On coherent prefixes, a larger context can contain more genuinely useful
matches, and the pooled statistic also changes its position/content mixture. Polar can therefore
show large `n_eff` growth while its projected attention energy and late-shell NLL remain stable.
Treat raw `n_eff` as a descriptive cardinality statistic — possibly useful capacity utilization —
not a sufficient stand-alone failure criterion. The stronger object is anchor-conditioned: does an
answer-preserving context extension erode the task margin or push the attention, memory, or
normalized pre-MLP state outside its short-context operating tube?

---

## 2. Closed-loop entropy-targeting read (a conditional bandwidth controller)

**Idea.** If paired analysis shows that irrelevant key growth raises anchor-local `n_eff` and
damages the read, replace the open-loop `log N` temperature with a **closed-loop regulator** that
tracks a teacher-relative or task-conditioned setpoint — a variable-bandwidth / balloon-estimator
form of kernel regression.

- For a target `n_eff*`, solve per-query for the temperature `T(q)` such that
  `n_eff(T) = exp H₂(softmax(scores/T)) = n_eff*`. `n_eff(T)` is monotonic in `T` and bounded
  by the current batch size → a well-posed 1-D root find per query (differentiable, or
  solve-then-straight-through).
- This can make the chosen anchor-local statistic **length-controlled by construction** rather
  than by meta-learned approximation. It folds into the existing `polar_temp_null` in
  [model/blocks.py](../../model/blocks.py).

**Falsifiable prediction.** Under a paired extension containing known-irrelevant keys, the
controller should preserve the aligned query's useful read, final margin, and downstream branch
operating point. A flat global `n_eff` curve proves only that the controller moves its stated
statistic; it does **not** by itself prove length robustness. A fixed global setpoint may even
discard useful repeated evidence in a coherent document.

**Caveat.** The target must be feasible for the current key count and score-tie structure; a fixed
global setpoint is not always attainable or desirable. Even if validated, this addresses the
*non-parametric* read's calibration; it does not make it *bounded-compute* (still O(N)). Compression
remains the memory's job — the two are complementary branches, not substitutes.

---

## 3. Memory as compensation vs. repair (resolve which)

First MAG results showed the Titans memory **dampens the perplexity distribution past train
length even with full (un-windowed) polar attention** (`full` reversed from worst ~3.21 to best
& monotonic ~1.93 @64×). Open question: does the memory **repair** the read flaw or merely
**mask** its symptom?

- **Pure additive compensation** (hypothesis A): the memory is a length-stable gist channel
  (delta state-norm is flat in N; RMSNorm'd readout) that keeps the final margin usable while the
  projected attention branch, normalized pre-MLP query, or count contribution still drifts.
- **Training-time reshaping** (hypothesis B): training *with* memory changes the upstream query/read
  geometry so the attention branch itself remains inside its paired short-context operating tube.
  This repair need not make pooled `n_eff` flat.

**Test (deferred — needs the memory-trained checkpoints + GPU).** Run [eval.py](../../eval.py)
`--diagnose --no_mem`, but compare paired projected-attention drift, attention/memory energy,
normalized pre-MLP drift, and raw pre-cap margin in addition to `n_eff`:

- Attention drift unchanged while memory removal destroys the final margin → **compensation**.
- Attention/pre-MLP drift is already smaller than in a no-memory-trained model, even with memory
  disabled at evaluation → **training-time repair**.
- Raw `n_eff` changes without corresponding branch or margin changes → cardinality diagnostic,
  not evidence for either mechanism.

The prior remains some mixture of compensation and training-time reshaping, but the full
checkpoint atlas and causal tests in section 4 should decide the balance.

**Division-of-labor reminder:** the linear gated-delta memory is a capacity-limited GIST (cap
~`d_k`) with no guarantee of exact episodic recall. Pinpoint retrieval should remain the job of a
full or sparse non-parametric Polar path. Whether a distractor objective helps that path is an
empirical ablation, not part of the architectural claim.

---

## 4. Mechanistic length-failure program (analysis first, intervention later)

**Scope gate.** This program begins after the complete Paper-1 checkpoint experiment lands. During
the current paper, use already-trained checkpoints to collect evidence and sharpen the causal
hypothesis only. Any new regularizer, paired-view loss, or source-layer constraint belongs to later
paper/model iterations and must not enter the primary grid retroactively.

This section extends section 1's read-calibration question and section 3's
compensation-versus-repair test. The target is not merely a marginal activation that changes with
length; it is a reproducible chain from an answer-preserving extension to a source-layer change,
downstream amplification, and task-margin erosion.

**Pilot observation (2026-07-19; provisional).** The
[checkpoint stress driver](../../scaled_ablation/eval_hf_checkpoints.py) and
[streaming probe](../../scaled_ablation/stress.py) evaluated three 378M-class checkpoints on the same
eight coherent documents from 2K through 131K. Every passive run completed without OOM or
non-finite activations. The checkpoints were almost tied at 2K but separated sharply at length.
Because the lengths double and the completed document sets are identical, disjoint late-shell
statistics can be recovered exactly from the cumulative JSON:

```text
L_shell(N) = 2 L_cumulative(N) - L_cumulative(N/2)
R_shell(N) = sqrt(2 R_cumulative(N)^2 - R_cumulative(N/2)^2)
```

For a general length grid, use count-weighted moment subtraction rather than the simplified dyadic
form.

| Checkpoint | Shell NLL, 2K → 131K | Block-7 MLP RMS | Block-10 attention RMS |
|---|---:|---:|---:|
| NoPE, 10B-token run | 2.54 → **13.66** | 3.80 → **107.86** | 6.02 → **83.57** |
| NoPE, L4 run | 2.53 → 5.04 | 3.99 → 3.80 | 7.19 → 29.62 |
| Polar, 10B-token run | 2.55 → **1.63** | 2.14 → 3.03 | 1.12 → 1.22 |

Treat these values as hypothesis-generating, not causal evidence. The run has only three
checkpoints, the modal pass has one document and two random directions, and later shells contain
different document content as well as greater context age. Pooled scalar moments also hide
per-channel shifts, rotations, and covariance changes.

**Revised working hypotheses.** Candidate block numbers are provisional until the full atlas and
held-out confirmation agree.

1. **Length failure may be a branch-amplification cascade, not generic numerical instability.** In the
   failing NoPE checkpoint, projected attention grows around blocks 6 and 10 while the Titans
   branch stays finite and generally shrinks. At block 10, late-shell attention exceeds memory by
   about `18.8x`; Polar's ratio is about `0.48x`. These are branch-RMS associations, not proof that
   one vector contribution dominates the sum; paired branch cosine and cross terms must test that.
2. **The dangerous MLP change may be directional/covariant.** Block 7's MLP output grows about
   `28x` despite receiving an RMS-normalized query. A modest marginal input-scale change can hide
   motion into a high-gain feature sector — the concrete margin-tube hypothesis. RMSNorm rules out
   scalar input norm as the sole explanation, but MLP output RMS is not itself a gain estimate.
3. **Raw effective count is not sufficient as a stand-alone failure criterion in this pilot.**
   Polar keeps projected attention `O(1)` and late-shell NLL stable while pooled `n_eff` grows
   strongly. The useful quantities may instead be anchor-local significance, projected
   content/count energy, branch balance, and final margin.
4. **Final-layer smoothness is insufficient for localization.** A final SIGReg, raw residual-norm,
   or STP-style angular improvement can coexist with an upstream attention/MLP failure that later
   layers compensate. A task-aware final margin remains the behavioral guardrail.

**Protocol repair before the full checkpoint atlas.** Freeze the protocol once Paper 1 completes,
then make the analysis comparable and auditable:

- Emit disjoint shell loss and moments alongside cumulative values; retain per-document records and
  bootstrap over documents rather than treating pooled tokens as independent samples.
- Add last-512/last-2K and log-spaced position bins. For stronger causal isolation, evaluate the
  same suffix targets under full and truncated/extended context.
- Rename `first_yield_length` to `first_envelope_exit`. The `1.25x` threshold marks distribution
  departure, not behavioral failure. Exclude raw `n_eff` from this generic rule; report per-query
  `n_eff/n_keys`, its log-log scaling exponent, `mag`, `w_null`, projected count RMS, and
  attention/memory energy instead.
- Record native-window and explicitly forced-full-context results separately, plus the effective
  runtime configuration, repository/checkpoint revisions, weight hash, actual attention/memory
  kernels, library versions, and training provenance where available.
- Audit the needle scaffold's extra EOT and make within-document versus cross-document retrieval
  explicit. Use a distance-matched absent baseline and report exact-value as well as per-token
  accuracy.
- Keep the passive streaming probe cheap across all checkpoints. Treat the current isotropic modal
  statistic as a **random secant gain**, not a singular value or worst-case Lipschitz constant.
  Reserve more documents/directions and structured perturbations for selected representatives.
- Hold out a fresh document manifest, needle seed, lengths, and preferably checkpoints from the
  exploratory atlas for confirmation.

**Analysis ladder.** Do not select a training attachment point from the pilot table alone.

1. **Passive checkpoint atlas.** Run every completed checkpoint under the repaired protocol. Plot
   shell NLL against branch-to-residual ratios, normalized Polar internals, gate saturation, and
   per-layer MLP output. Use checkpoint family/seed replication to nominate candidate sources and
   downstream amplifiers.
2. **Paired aligned anchors.** On representative robust, graceful-degradation, and catastrophic
   checkpoints, compare the same suffix/query under a short sufficient context and under
   log-spaced answer-preserving extensions. Use both EOT-reset pairs for language-model stability
   and cue/value-with-intervening-distractors pairs for retrieval. At aligned anchors record:
   - projected attention content and Titans output at candidate attention layers;
   - normalized pre-MLP queries and MLP outputs at candidate amplifier layers;
   - full, DC, longitudinal, and transverse short-to-long drift;
   - raw pre-softcap decoding margin/radius and the drift-to-margin ratio;
   - the observed length-direction secant gain
     `||B_l(h_long) - B_l(h_short)|| / (||h_long - h_short|| + eps)`.
3. **Frozen causal interventions.** Patch long-context activations with the aligned short-context
   values at nominated sites. Compare full-vector patching, norm-only clamping, direction-only
   patching, attention-content patching, normalized pre-MLP patching, memory on/off, negative-control
   layers, and shuffled patches. Apply the intervention only at frozen anchor positions so a global
   late-state replacement cannot trivially solve the task.

| Observation | Interpretation |
|---|---|
| Norm clamp rescues behavior | Primarily magnitude/branch-energy failure |
| Vector or direction patch rescues, norm clamp does not | Direction/covariance tube escape |
| Attention-content patch repairs downstream MLP and margin | Attention branch is a causal source |
| Pre-MLP patch helps but upstream attention patch does not | Intervening mixer/MLP interface is the source or amplifier |
| Memory removal preserves upstream drift but destroys margin | Titans is compensating rather than repairing |
| Candidate patches fail on held-out documents/checkpoints | Reject or relocate the proposed causal chain |

**Modified PLMT objective (deferred).** Only after paired drift predicts held-out margin loss and
causal-patching effects replicate should a later paper iteration train the mechanism. PLMT means a
**paired length-conditioned margin tube**, not generic hidden-state matching. Construct a short
sufficient view and an answer-preserving extended view:

```text
# Language-model stability: preserve the same reset boundary and suffix.
x_short_lm = (EOT, suffix)
x_long_lm  = (irrelevant_prefix_N, EOT, suffix)

# Retrieval: preserve the cue/value and query, inserting only known distractors.
x_short_ret = (cue, value, query)
x_long_ret  = (cue, value, distractors_N, query)

choose (x_short, x_long) from one of the paired constructions above
target(x_short) = target(x_long) = y

h_short[a] = stopgrad(f_theta(x_short)[a])
h_long[a]  =          f_theta(x_long )[a]
W_bar      = stopgrad(W_theta)
b_bar      = stopgrad(b_theta)
delta_N[a] = h_long[a] - h_short[a]
```

Here `h[a]` is the final normalized hidden state immediately before the vocabulary projection at
the aligned answer anchor. Use the same current parameters in both views for the primary
experiment so `delta_N` isolates the representation change caused by the extension. An EMA short
teacher is a stabilization ablation, not part of the exact interpretation: it also introduces
teacher/student parameter lag. If tested, take the short state and detached head from the same EMA
snapshot and retain a same-model comparison. Current Atma is dropout-free; any future stochastic
variant must disable stochastic layers or share their RNG masks between views for this
extension-only interpretation.

The extension may be an EOT-reset prefix for language-model stability or intervening distractors
between a cue/value and its query for retrieval. Do not apply PLMT when the extension can
legitimately change the target.

**1. One-sided decision-margin tube - exact derivation.** For the fixed raw pre-softcap head, write

```text
z_i(h)     = W_bar[i] dot h + b_bar[i]
a_yj       = W_bar[y] - W_bar[j]
s_yj       = norm(a_yj)
u_yj       = a_yj / s_yj
rho_yj(h)  = (a_yj dot h + b_bar[y] - b_bar[j]) / s_yj

rho_long[y,j]
    = rho_short[y,j] + u_yj dot delta_N                 # exact
    >= rho_short[y,j] - norm(delta_N).                  # Cauchy-Schwarz
```

`rho_yj` is the signed Euclidean distance to the pairwise `y`/`j` decision hyperplane in final
hidden space. Exclude pairs with `s_yj <= eps` from the Euclidean-distance calculation and handle
their bias-only comparison directly: they are always safe when `b_bar[y] > b_bar[j]`, and invalid
or tied otherwise. Do not call an epsilon-clamped value an exact distance. Therefore, when the
short view predicts `y`,

```text
norm(delta_N) < min_{j != y} rho_short[y,j]
```

is a sufficient condition for preserving the token decision. The minimum over the full vocabulary
is the exact ambient-space inradius of the raw linear-head decision cell; a top-K subset is only an
approximation. Because the final state lies on the image of RMSNorm, this ambient ball can be more
conservative than the set of feasible hidden perturbations. Atma's elementwise logit softcap is
strictly monotone and therefore preserves token ranking and equality boundaries, although it
changes probability calibration and does not preserve normalized-radius ordering or Euclidean
distance. The raw pre-cap geometry is the cleaner metric.

This identity motivates preserving a safe radius, not forcing `h_long == h_short`. Let

```text
tau[a,j] = max(rho_min, kappa * stopgrad(rho_short[a,j]))
ell_margin[a,j] = [tau[a,j] - rho_long[a,j]]_+^2

L_margin
    = sum_a M_a * max_{j in C_a} ell_margin[a,j]
      / (sum_a M_a + eps)
```

Use a temperature-smoothed maximum if the hard worst-boundary switch is unstable. Logit top-K is
not sufficient: the closest boundary minimizes `(z_y - z_j) / norm(W_y - W_j)`, so a lower-logit
competitor can still have the smallest radius. Mine `C_a` as the union of the short/long
**K-smallest normalized radii** (streaming detached vocabulary blocks), ordinary logit-top-K, and
known retrieval alternatives; use the full vocabulary when affordable. Report how often a held-out
full-vocabulary audit finds an omitted tighter boundary.

`M_a` is one only when the short view is correct and its minimum included radius already exceeds
`rho_min`, with `0 < kappa <= 1`. Otherwise ordinary NTP supplies the learning signal. Normalize by
the number of active anchors and report gate coverage separately so the effective weight does not
silently shrink when fewer teachers qualify. The one-sided hinge preserves an absolute floor and a
fraction of the clean radius, but gives no reward for confidence inflation. For a multi-token
value, aggregate teacher-forced token margins or use a sequence-level candidate margin when
explicit alternative values are available. Use `kappa < 1` when calibrating a positive worst-case
source radius; `kappa = 1` leaves zero norm-wise margin budget and is meaningful only as a stricter
margin-only ablation.

**2. Causally validated source tube - theorem-inspired derivation.** In Garcia et al.'s Hebbian
kernel-memory analysis, query-Lipschitzness and isotropic values give the noisy-margin bound
(Theorem B.62 and Corollary B.64 in
[*MLPs are Hebbians*](https://arxiv.org/abs/2607.10034v1))

```text
L_prob = log(C_0 * F^2 / delta_prob)

gamma_noisy
    >= gamma_clean
       - L_k * epsilon * (2 + C_1 * sqrt(F * L_prob / d))

A_general = L_k * (2 + C_1 * sqrt(F * L_prob / d))
A_common  = C_2 * L_k * sqrt(F * L_prob / d)   # only when F * L_prob >= d

epsilon < (gamma_clean - gamma_required) / A
```

for `gamma_clean > gamma_required > 0` and the applicable `A`. The paper then applies this
principle to its constructed bilinear MLP in a single-block synthetic recall setting: safe query
radius is clean margin budget divided by downstream sensitivity. Neither those assumptions nor the
constants transfer to a deep trained Atma model, whose internal stored key, fact identity, geometry,
and global Lipschitz constant are unavailable.

PLMT therefore defines the source statement only counterfactually, not from passive correlation.
For candidate state `z_l` at anchor `a`, let `E_long[l,a]` denote the non-intervened long-run inputs,
caches, and upstream state fixed by the patch protocol, let `G_l(z; E_long)` be the downstream
recomputation after patching `z`, and define

```text
R_laj(z; E_long) = rho_yj(G_l(z; E_long)).
```

If `R_laj` is `A_laj`-Lipschitz along an interpolation between the patched short and long values,
using the **same normalized metric `d_l` used by the loss**, then

```text
R_laj(z_2; E_long)
    >= R_laj(z_1; E_long) - A_laj * d_l(z_2, z_1)

rho_short_patch[l,a,j] = R_laj(z_short; E_long)
r_safe_raw[l,a,j]
    = [rho_short_patch[l,a,j] - tau[a,j]]_+ / (A_hat[l,a,j] + eps)

r_safe[l]
    = low_quantile_a (min_{j in C_a} r_safe_raw[l,a,j])
```

Estimate `A_hat` with held-out patch interpolation/JVP probes in that metric, not with an
uncontrolled short/long secant. Require the short patch to restore the long-run margin; otherwise
the proposed source is incomplete and should be rejected. Even a path maximum or high-quantile
`A_hat` and low-quantile `r_safe` define an empirical risk target, not a certified upper bound.

Let `r_natural[l]` be a high quantile of harmless answer-preserving same-length nuisance drift in
the same metric. A defensible source tolerance must admit

```text
r_natural[l] < r_source[l] < r_safe[l].
```

If no interval exists, reject the site, metric, or invariance assumption rather than forcing exact
matching. The soft hinge does not guarantee `d_l <= r_source`; report tube-violation and
margin-violation rates as well as mean loss. Natural-drift calibration alone does not establish
margin safety. Use only source sites that survive the checkpoint atlas and causal patching. For the
current pilot candidates:

```text
c_v[l,a] = projected attention-content vector before Polar count and Titans, l in S_content
q_v[l,a] = normalized state entering the MLP,                         l in S_query

s_content[l,a] = max(stopgrad(norm(c_short[l,a])), content_floor[l])
d_content[l,a]
    = norm(c_long[l,a] - stopgrad(c_short[l,a])) / s_content[l,a]

d_query[l,a]
    = norm(q_long[l,a] - stopgrad(q_short[l,a])) / sqrt(hidden_dim)

L_source
    = mean_{l in S_content,a} [d_content[l,a] - r_content[l]]_+^2
      + beta_q * mean_{l in S_query,a} [d_query[l,a] - r_query[l]]_+^2
```

Set `content_floor[l]` to the larger of a fixed nonzero minimum and a stopped running/EMA low
quantile of short-view content norms, so relative error is not singular when the clean attention
branch is nearly zero. This running scale is unrelated to the optional EMA teacher. The query
normalization follows directly from RMS-scaled geometry. If
`r_v = norm(q_v) / sqrt(hidden_dim)` and `theta` is the short/long angle, then

```text
d_query^2
    = (r_long - r_short)^2
      + 2 * r_long * r_short * (1 - cos(theta)).
```

Under ideal unit-RMS normalization this becomes `2 * (1 - cos(theta))`. Learned RMSNorm gains can
make the radii differ, so the full distance deliberately retains both gain-weighted radial and
angular drift.

The provisional sets are `S_content = {6, 10}` and `S_query = {7, 13}`; they are not universal
architectural constants. Set `r_content` and `r_query` using both the natural-drift and safe-radius
tests above. Full vector drift catches amplitude, longitudinal, transverse, and DC changes. If
causal tests isolate a directional failure, a shrinkage/EMA Mahalanobis metric may replace
Euclidean `d_query`; if norm-only clamping rescues the model, retain an explicit magnitude term.
A single anchor activation is still a mechanistic proxy unless patching shows it contains the
relevant causal state.

**3. Optional excess bending.** For nonzero ordered increments `p = h_r - h_s` and
`v = h_t - h_r`, the ideal STP loss obeys

```text
STP = 1 - cos(p, v)
    = 0.5 * norm(normalize(p) - normalize(v))^2

sin(theta)^2 = 2 * STP - STP^2
```

so at small angle it measures normalized transverse bending. This is the useful overlap with
length-noise robustness, but it is narrower than PLMT: STP is invariant to a common translation of
`(h_s, h_r, h_t)` and to positive rescaling of either increment, and it does not measure large
collinear drift or distance to a decision boundary. Its semantic-geodesic, anchored-endpoint, and
Brownian inference-cone assumptions have not been established for Atma's length failure.

Only if paired measurements show a predominantly transverse damaging component, compute STP on
the same aligned local trajectory window around the answer anchor and at the same causally
nominated representation in both views. Keep the paper's final-layer attachment as a reference
ablation, and use

```text
M_speed = 1{min(norm(p_short), norm(v_short), norm(p_long), norm(v_long)) >= speed_min}

L_bend
    = sum M_speed * [STP_long - stopgrad(STP_short) - r_bend]_+^2
      / (sum M_speed + eps)
```

Use epsilon-stabilized cosine in code and report speed-mask coverage; without the mask, nearly
stationary increments have arbitrary angles and unstable gradients. Do not span the inserted
distractor interval itself. This preserves legitimate short-view curvature and penalizes only
length-induced excess bending; it cannot replace full source drift or the task-aware decision
margin.

The complete deferred objective is therefore:

```text
L_PLMT_mean = L_NTP_mean
            + lambda_margin * L_margin
            + lambda_source * L_source
            + lambda_bend * L_bend
```

Retain ordinary short-context NTP batches so consistency is never the sole semantic anchor. In the
current trainer, cross-entropy is summed while these auxiliaries are means, so preserve the
intended scale as:

```text
N_CE = exact number of non-ignored tokens represented by L_CE_sum

L_code = L_CE_sum
       + N_CE
         * (lambda_margin * L_margin
            + lambda_source * L_source
            + lambda_bend * L_bend)
```

If CE supervises both paired views, `N_CE` counts non-ignored targets from both; if only the long
view receives paired CE, it counts only that view. Give PLMT its own weights rather than reusing
`SIGR_ALPHA`. Start the later ablation with `lambda_bend = 0`: margin-only, then
margin-plus-causally-validated-source, and add excess bending only if the transverse hypothesis
survives the analysis ladder. The no-grad short pass still adds roughly one forward pass plus
activation capture, so record step time, peak memory, and the effect of applying paired batches
only at a sampled interval.

**Design lineage and deliberate changes.** These are design precedents, not claims that their
proofs directly certify Atma or PLMT.

| Source | Principle retained | What PLMT deliberately changes |
|---|---|---|
| Garcia et al., [*MLPs are Hebbians: Constructing Efficient Fact-Storing MLPs for Transformers*](https://arxiv.org/abs/2607.10034v1) | Robust factual use depends on decoding margin, not merely correct clean decoding; noisy attention queries consume that margin. Embedding geometry and whitening affect cross-talk. | Their guarantees concern constructed bilinear MLPs, explicit keys, and a one-layer synthetic recall setting. PLMT replaces unavailable internal margins and noise ceilings with paired drift, empirical tolerances, and the fixed vocabulary-boundary radius. No theorem is transferred. |
| Huang, LeCun, and Balestriero, [*Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA*](https://arxiv.org/abs/2602.22617v1) | Ordered hidden-state bending exposes angular/transverse deviation under a local-linearity hypothesis. | PLMT penalizes only short-to-long **excess** bending on aligned local windows, retains a curvature tolerance, and keeps full-vector drift and output margin. Unlike STP's view-free default, PLMT deliberately uses controlled two-view pairs because length itself is the intervention. |
| Tarvainen and Valpola, [*Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results*](https://arxiv.org/abs/1703.01780) | An EMA-weight teacher can provide a slower detached consistency target. | Same-model stop-gradient is the PLMT default so the contrast isolates length. EMA is an ablation for stability, applied only to the sufficient short view and gated by correctness; ordinary NTP remains the target anchor. |
| Zhang and Sennrich, [*Root Mean Square Layer Normalization*](https://arxiv.org/abs/1910.07467) | RMS scaling motivates a dimensionless `norm(delta_q) / sqrt(d)` diagnostic. | PLMT does not discard learned-gain radial drift; the exact radius-angle decomposition above determines when the metric is genuinely angular. |
| Current four non-baseline [SIGReg modes](../../train/reg.py) | `weak` and `discrete` expose covariance/whitening geometry; `strong` matches projected characteristic functions to a Gaussian; `zipfian` separates angular decorrelation from norm-rank structure. | SIGReg pools the unpaired `B x T` marginal and has no event alignment, length contrast, target identity, or decision boundary. PLMT is paired, anchor-aligned, one-sided, and margin-aware; it is complementary, not a fifth SIGReg mode. |
| [Polar Attention](../POLAR_ATTENTION.md) | Its content/direction, effective-count/magnitude, and null decomposition supplies branch-resolved diagnostic sites. | PLMT constrains only a causally validated source, provisionally projected content before count and memory. It does not clamp `n_eff`, count, or null merely because they change with length. |
| [Titans MAG memory](../TITANS_MEMORY.md) | Its additive branch makes upstream repair versus downstream compensation testable. | PLMT places the source tube before Titans and uses memory on/off patching as a causal test. It does not regularize memory by default, and Atma's gated-delta memory must not be conflated with Garcia et al.'s constructed Hebbian MLP. |

Keep the epistemic labels explicit: the fixed linear-head boundary relation and the RMS/STP
algebra are exact; the internal noisy-query transfer is theorem-inspired; the causally validated
source tube and its tolerances are empirical surrogates; and excess STP bending is an optional
geometric heuristic. STP itself notes that latent semantic signal and nuisance noise are not
directly observable, so neither STP nor PLMT should be described as a certified SNR improvement
until paired geometry, margin, behavior, and causal interventions agree.

**Evidence and cost gate.** Promote no pilot hotspot or intervention to a paper claim unless it
replicates across checkpoint families/seeds, predicts held-out length-onset, and survives negative
controls. If it does, run a sequential 1B-token ladder — baseline, margin-only, then
margin-plus-source — with at least three seeds and clean NLL, shell NLL, distant retrieval,
calibration, and geometry probes. Run a selected 10B-token proof only if the pilot improves length
outcomes without material short-context regression. Do not construct a Cartesian grid over layers,
losses, weights, and length schedules.

---

## 5. Wall Attention (Tilde Research) - incompatible attempted contender

**Protocol verdict (2026-07-07):** Wall Attention remains implemented for diagnostics, but it is
excluded from the fair Atma grid. Under the standardized hybrid architecture plus native Atma Muon
split, the baseline Wall cell improved early and then regressed badly; the comparable Raven native
Muon control shows the same qualitative instability. Treat this as a protocol incompatibility, not
as a general negative result on Wall, because the official Wall recipe uses per-head Muon/MuonSplit
and Aurora-style training details. Raven now replaces Wall as the stronger outsider baseline through
[raven_baseline](../../raven_baseline/), with the protocol difference stated explicitly.

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
   also **independent confirmation of Titans [Finding 2](../TITANS_MEMORY.md)** (per-head γ as a
   learned retention spectrum), at finer (per-channel, in-read) granularity.
2. **A concrete adaptive-bandwidth kernel.** In the regression lens (§1), Wall is a
   **non-stationary kernel** whose bandwidth shrinks along the time axis at a learned,
   content-dependent, per-channel rate — i.e. one realization of the variable-bandwidth fix from
   §2, but on the *score* side. It is *adaptive* but does **not regulate `n_eff` to a setpoint**
   (static channels still ramp). Complementary to entropy-targeting (§2), not a substitute.
3. **FlashAttention compatibility.** Wall is a pre-softmax q/k rescale → keeps the easy
   inference path, where polar's null column + weight-level reductions sacrifice it (root of the
   deferred [inference port](../POLAR_ATTENTION.md) pain). If the port stays painful, "Wall-NoPE
   alone" is an attractive FA-compatible baseline.

**Historical candidate cells, now superseded by the incompatibility verdict:**

- `attn_type += wall` — drop-in NoPE positional, FA-compatible baseline (the ablation harness
  already has rope/nope/polar as an axis).
- **Wall-scores → polar-normalization** — does Wall's adaptive kernel reduce polar's `n_eff`
  ramp on the `--diagnose` probe? (Note: Wall+polar still inherits polar's non-FA inference.)
- **Wall vs Titans on the window-vs-retrieval cell** — does per-channel gating make the gist
  memory partially redundant *for quality*, while the memory still wins on compute/KV-cache?

**Diagnostic status.** `attn_type="wall"` is implemented in
[train/model.py](../../train/model.py) (`CausalSelfAttention`, `pos="wall"`): **keeps canon** (so it's
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

## 6. YOCO-style shared KV banks for 1M-token serving

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

## 7. HOLA-style sparse Polar episodic cache

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

## 8. SDM-style sparse recurrent memory capacity ladder

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
sharp needles, combine it with section 7's sparse Polar cache:

```text
out = local_or_lfm2 + sdm_gist + gate_full * full_polar + gate_sparse * sparse_polar
```

This preserves the division of labor: SDM is the high-capacity recurrent gist tier; sparse Polar is
the exact episodic tier; full Polar remains the teacher / diagnostic path until the sparse hierarchy
is strong enough to deploy without it.

---

## 9. Foveal ATMA: query-sparse Polar, resolution-tiered MoE, and quantization

**Status and scope.** This track adapts the Foveal Transformer proposal into ATMA-sized,
falsifiable experiments. It does **not** commit ATMA to the proposal's 64-layer, 367B-parameter
target. The useful contribution here is the experimental decomposition:

1. a cheap full-context signal chooses which full-resolution KV pages deserve a read;
2. expert width and routing frequency form a regular multi-resolution compute hierarchy;
3. weight precision follows expert width and activation frequency.

These are three independent hypotheses. Prove each against a simpler matched baseline before
composing them. Sparse Polar belongs to the hierarchical-memory sequel; MoE and microscaling are a
separate model-scaling/systems track and must not block that paper.

### 9.1 Query-aware sparse Polar over a full host KV store

**Goal.** Bound long-context Polar read bandwidth while retaining a recoverable copy of every KV
page. Keep a low-dimensional learned index resident on GPU, exact recent GQA KV in a local window,
and old full-rank GQA KV in pinned host memory behind a GPU page cache.

| Path | Scope | Representation | Initial residency |
|---|---|---|---|
| Peripheral index | all causal pages | learned 16D MQA key; optional value | GPU |
| Local Polar | most recent `W` tokens | normal GQA KV | GPU |
| Remote Polar | selected old pages | normal GQA KV | pinned CPU RAM + GPU cache |

This is complementary to section 7 rather than a renamed HOLA cache:

| Question | HOLA-style sparse cache | Foveal Polar retriever |
|---|---|---|
| Where is sparsity decided? | write/eviction time | query/read time |
| Is all old full-rank KV recoverable? | no | yes, from host memory |
| Per-query support | shared bounded bank | dynamic selected pages |
| Dominant risk | irreversible eviction | index misses and PCIe latency |

The first combined design, only after both work independently, can use the HOLA bank as the GPU
episodic cache and the Foveal index as a host-store fallback. Do not begin with that composition.

**Polar-specific selection target.** Polar uses temperature-sharpened softmax weights plus a null
sink, so a predicted-mass controller is still meaningful, but the target must preserve both the
direction and participation-ratio channels. For query `t`, train the index against full-Polar
teacher statistics and select the smallest remote page set that reaches target real-key mass `p`,
subject to `K_min <= pages <= K_max`. If the predicted null confidence plus resident-local mass
already satisfies the controller, allow a no-fetch decision.

Selection loss should report behavior, not only page labels:

- recall of pages covering 90%, 95%, and 99% of teacher real-key mass;
- omitted mass and the cosine error of the unprojected direction accumulator;
- error in `n_eff`, `mag`, and `w_null`;
- selected pages, late fetches, and speculative overfetch bytes per query;
- per-GQA-group recall when one shared MQA index chooses the support.

Start with index-only retrieval. Add the optional low-dimensional value/residual path only as an
ablation; otherwise it can hide retrieval failure by becoming a second approximate attention
mechanism.

**Exact Polar merge on selected support.** Reuse the online Polar reduction rather than applying a
softmax-attention merge unchanged. Each local or remote block returns:

```text
M  = max scaled score
L  = sum exp(score - M)
S  = sum exp(score - M) * value
Q2 = sum exp(2 * (score - M))
```

When combining blocks at `M_new = max(M_a, M_b)`, rescale `L` and `S` by
`exp(M_old - M_new)` and `Q2` by its square. Fold in the null sink once after all selected blocks
arrive, then recover `direction`, `n_eff`, `mag`, and `w_null`. This is exact on the selected
support; omission of unselected remote pages is the only read approximation.

**Decode pipeline.** At each attention layer and autoregressive step:

1. form the 16D index query and scan page scores;
2. run exact local-window Polar immediately;
3. speculatively copy high-scoring remote pages on a second CUDA stream;
4. finalize capped top-P after the complete index scan;
5. run each arriving remote tile and merge its `M/L/S/Q2` statistics;
6. append new full KV to the local window and host store, and append its compact index state.

Top-P is a quality target, not a latency guarantee: exact top-P still needs the complete index
score distribution. `K_max` is the service-level guardrail. Report index scan time, unhidden DMA,
remote reduction time, cache hit rate, and wasted prefetch bytes separately.

The full index scan is still `O(N)` per decode step and its straightforward prefill is still
quadratic in sequence length. Treat this first as a decode-bandwidth and memory-capacity proposal;
hierarchical page indexes or approximate retrieval are later work, not implied wins.

**Ablation ladder.** Keep the modeling and systems questions separate:

1. On-GPU teacher experiment: dense/full Polar vs. local window vs. oracle page support vs. learned
   fixed top-K vs. learned capped top-P. No CPU offload yet.
2. Index structure: 16D shared MQA vs. 32D shared MQA vs. one index per KV group.
3. Retrieval controls: recency pages, random pages, score threshold, entropy-conditioned `p`, and
   HOLA's query-independent retained bank at the same read budget.
4. Runtime experiment: pinned-memory page store, GPU LRU cache, and overlapped copies; compare
   end-to-end decode against full resident Polar and windowed Polar.
5. Only if query-time retrieval wins, test HOLA-as-L1 plus host fallback.

**Checkpoint-CPT pilot (preferred first experiment).** Reuse the completed 9.816B-token L40S
checkpoints rather than relearning the base model. The isolated runnable implementation and launch
instructions live in [`foveal_cpt/`](../../foveal_cpt/README.md):

| Core | Starting checkpoint | Pilot role |
|---|---|---|
| Polar | [`atma-10b-L40S-mbs16-polar`](https://huggingface.co/ChavyvAkvar/atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0) | primary Foveal run |
| RoPE | [`atma-10b-L40S-mbs16-rope`](https://huggingface.co/ChavyvAkvar/atma-10b-L40S-mbs16-rope__reg-baseline__distr-0__mem-1__win-0) | softmax transfer control |
| NoPE | [`atma-10b-L40S-mbs16-nope`](https://huggingface.co/ChavyvAkvar/atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0) | instability diagnostic |

All three checkpoints have the same 16-layer, `hidden_size=1024`, `head_dim=128`, four-attention-
layer shape, GQA 1:4, enabled Titans memory, and no training window. Only the attention core
differs. This makes weight loading straightforward: load the existing state dict non-strictly,
or, equivalently, load the unmodified base model strictly before wrapping its attention layers with
the new indexer. The implementation uses the latter so any base-checkpoint mismatch fails early.

The NoPE checkpoint is not an equal-confidence candidate. Existing checkpoint stress shows its
clean-document loss rising sharply by 32K, while Polar remains stable and RoPE degrades gradually.
Use NoPE to learn whether sparse CPT can repair or worsens that stored failure mode; do not treat a
NoPE failure alone as falsification of Foveal retrieval.

**Initial 32K operating point.** Use one learned index per attention layer:

```text
sequence length             = 32768
local exact window          = 512 tokens
KV page / query block       = 64 / 64 tokens
index                       = 16D MQA key-only
train top-P                 = 0.95 after warm-up
remote K_min / K_max        = 0 / 32 pages
maximum remote support      = 2048 tokens
maximum exact support       = 512 local + 2048 remote = 2560 tokens
global batch                = 524288 tokens = 16 sequences
microbatch / accumulation   = 1 sequence / 16 microbatches
CPT budget                  = 1908 steps = 1,000,341,504 tokens
```

A 32K retrieval pilot needs document-coherent training examples. Do not simply slice a flat token
stream across unrelated document boundaries: either build packs from sufficiently long documents
or carry an explicit end-of-document reset mask through attention and Titans memory. Otherwise the
index can look successful by learning recency and document-boundary artifacts rather than remote
semantic retrieval.

A direct `hidden_size -> 16` query/key pair adds only about 32.8K weights per attention layer, or
131K across ATMA's four attention layers before biases. At 32K, a BF16 key-only index is 1 MiB per
layer and sequence. The index is therefore not the capacity risk; sparse gather/reduction kernels,
activation memory, and the Titans backward are the items to profile.

Share one remote page set across each 64-token query block during prefill. Per-token selection is
appropriate for decode but makes 32K training irregular and difficult to fuse. Use the block's
first query token as its causal routing representative, aggregate full-rank teacher mass across
heads conservatively, and report head-wise recall so mean aggregation cannot silently erase a
specialized head.

For Polar, preserve the original full causal `n_keys=t+1` when computing temperature and the null
floor. Do not replace it with the number of selected tokens. The sparse kernel reduces over the
selected local/remote support, but the length calibration must still describe the full visible
context.

**Indexer warm start and loss.** Hard page selection does not pass a useful LM gradient into the
router. Before 32K CPT, freeze the checkpoint and calibrate only `W_Q_index/W_K_index` for roughly
10M-25M tokens at 2K-8K, where the original full attention is affordable. Distill the full-rank
teacher's page distribution:

```text
L_index = KL(stopgrad(full_rank_page_mass) || softmax(index_page_logits))
```

For the shared MQA index, aggregate teacher mass across query heads with a mean-plus-maximum or
log-sum-exp target and retain head-wise recall as a diagnostic. Initialize each checkpoint's index
independently. RoPE should give the index its own trained positional treatment; do not assume that
post-RoPE full-rank keys can be projected to 16D without changing their ranking.

During 32K CPT, keep the auxiliary teacher alive on a small sample of query blocks. For example,
sample four blocks per sequence and attention layer and use the first query in each block as a
causal routing anchor. Stream those four queries against all 32K keys and update the index with
`L_index` while the LM path remains sparse for all tokens. Using future queries in the block to
choose support for its first query would leak information through the sparsity pattern. This
first-query rule avoids that leak and avoids constructing a full `32768 x 32768` teacher matrix.

Use a conservative sparsity handoff to avoid changing full attention to `SWA=512` in one step:

1. first 50M tokens: `p=0.98`, `K_min=8`, `K_max=64` remote pages;
2. next 150M tokens: anneal to `p=0.95`, `K_min=0`, `K_max=32`;
3. final 800M tokens: hold `p=0.95`, `K_max=32`, adjusting only if the cap-hit or miss rate trips
   a predeclared gate.

At evaluation, sweep `p in {0.90, 0.95, 0.98}` and fixed remote `K in {8, 16, 32, 64}` from the
same trained checkpoint. This estimates the quality/bandwidth curve without training a Cartesian
grid. If more than roughly 25% of query blocks hit `K_max=32`, report that top-P is cap-dominated
and evaluate `K_max=48/64`; do not describe the run as genuinely adaptive.

The Hugging Face artifacts contain model weights but not optimizer state, so this is CPT from a
weight initialization, not an exact optimizer resume. Give the calibrated indexer a separate
learning-rate group. Start existing parameters at about one tenth of the original peak learning
rates, keep the indexer's rate higher, and select the exact pair with a short loss/gradient-norm
smoke sweep. Profile microbatch 1 before committing the run; add per-block activation checkpointing
if the 32K backward does not fit comfortably on the L40S.

This requires a dedicated CPT entry point. The current scaled-ablation trainer initializes a fresh
model and has no non-strict checkpoint-load path, while the NoPE/RoPE training window constructs a
dense `T x T` mask. Changing only `seq_len` and `attn_window` to 32K is therefore neither a resume
nor a memory-safe sparse experiment. Add explicit checkpoint loading, index-only missing-key
validation, the block-sparse local-plus-remote kernel, sampled-teacher loss, and restartable CPT
state before launching the budgeted runs.

**Minimal run matrix.** Preserve one matched control without turning the pilot into another grid:

1. Polar `SWA=512 + Titans`, 1B-token 32K CPT control with no remote read.
2. Polar Foveal, 1B-token 32K CPT: primary result.
3. RoPE Foveal, 100M-token screen; promote to 1B only if retrieval recall and clean loss pass.
4. NoPE Foveal, 100M-token diagnostic; promote only if the known 32K activation/loss failure does
   not reappear.
5. Add matched 1B SWA controls for RoPE/NoPE only when their Foveal screens are promoted.

The control is scientifically necessary: without it, improvements after 32K CPT could come from
long-sequence adaptation or Titans rather than remote sparse retrieval. Evaluate the untouched
full-attention checkpoint, the SWA-CPT control, and the Foveal-CPT model on the same 2K-256K
clean-loss, needle/RULER, and `--diagnose` suite.

Before spending the full 1B tokens, require a short end-to-end gate: sparse forward/backward parity
on forced supports; index top-K recall materially above recency/random; finite gradients; no
direction/count/null discontinuity for Polar; and projected wall-clock at an acceptable fraction
of the original 10B-token training run.

**Go/no-go gate.** Continue beyond the on-GPU phase only if the learned index closes most of the
gap between fixed-budget selection and oracle support on both retrieval and clean perplexity.
Continue beyond the runtime phase only if it improves the quality-latency-memory Pareto frontier;
lower KV capacity alone is already covered more simply by YOCO and HOLA.

### 9.2 Resolution-tiered MoE as an independent ATMA scaling track

**Goal.** Replace the current dense squared-ReLU gated MLP with several regular expert bands whose
active parameter budgets are equal even though their widths and routing frequencies differ. For
expert width `r`, expert count `E`, and selected count `k_r`, use the planning invariant

```text
E * r_dense = k_1 * r_1 = k_2 * r_2 = k_3 * r_3 = B
```

where `B` is the active bottleneck budget per band. A proposal-style two-matrix expert costs
approximately `2 * hidden_size * B` active weights per band and token. Preserving ATMA's gated
squared-ReLU expert requires two input matrices and one output matrix, changing that factor to
approximately `3 * hidden_size * B`. Choose the expert form explicitly and derive `B` from the
current MLP's measured active MAC budget; if the pilot is deliberately cheaper, include a
compute-matched narrower dense MLP rather than attributing a compute reduction to MoE quality.

The proposed organization is:

| Resolution | Ownership | Routing | Purpose |
|---|---|---|---|
| narrowest | global/local | all `E` experts | dense training path; no starvation |
| narrow | global/local | moderate top-K | broad reusable specialization |
| wide | latent-head-local | small top-K per head | detailed specialization |
| widest | latent-head-local | very small top-K per head | rare high-resolution compute |

For ATMA's canonical `hidden_size=1024` and `head_dim=128`, the natural latent decomposition is
eight 128D heads. The wide expert banks can be owned by latent head, so a multi-GPU run exchanges
complete heads once before both wide bands and once after them. Routing and expert execution then
stay local. Start with a fixed reshape or fixed orthogonal rotation; learned full
`hidden_size x hidden_size` input/output mixing adds substantial always-active compute and should
be introduced only if head factorization measurably hurts quality.

**Implementation order.** Do not start with distributed Head Parallel:

1. Implement a single-device BF16 reference with band-major routing and explicit capacity-free
   top-K semantics. Verify forward/backward against a slow expert loop.
2. Compare independent bands against a conventional fixed-width top-2 MoE and parameter-/compute-
   matched dense MLP. Track load balance, expert starvation, router entropy, overflow, and kernel
   occupancy.
3. Add grouped/fused expert kernels. Reject the dense narrowest band if launch or weight-read cost
   makes it slower than an equivalent dense projection.
4. Only for a multi-GPU scaling run, shard the two wide bands by latent head and compare Head
   Parallel with conventional Expert Parallel at equal total parameters and active FLOPs.

The canonical eight-head shape permits Head Parallel group sizes `P in {2, 4, 8}`. Communication
should be deterministic and approximately one hidden vector per token per direction, independent
of routed top-K, but latency at batch size one remains a measured risk.

**MoE falsifiers.** Stop or simplify if dense-band fusion loses to the dense MLP, latent-head
factorization causes a quality gap not recovered by cheap mixing, routing collapses despite the
dense path, or deterministic head exchange loses to ordinary Expert Parallel on the target
hardware.

### 9.3 Precision by expert width and activation frequency

**Initial policy.** Establish the BF16 MoE result first, then lower precision one band at a time:

| Band | First execution format | Promotion fallback |
|---|---|---|
| dense narrowest | BF16 | remain BF16 |
| narrow sparse | MXFP8 | BF16 |
| wide sparse | MXFP4 | MXFP8 |
| widest sparse | MXFP4 | MXFP8 |

ATMA's existing custom FP8 path uses E4M3/E5M2 operands with scalar scales and is a useful
low-precision baseline, but it is not OCP MXFP8. MXFP8/MXFP4 require block-scaled storage (one E8M0
scale per 32 values in the base specification), layout-aware kernels, and separate policies for
activations, gradients, and optimizer state.

For training, keep sharded BF16 master weights and higher-precision optimizer state. Use quantized
copies for forward and activation-gradient GEMMs, accumulate weight gradients at tested higher
precision, update the owner copy, and requantize changed blocks asynchronously. Weight-gradient
MXFP4 is out of scope for the first experiment.

**Quantization ladder.** Each step must retain the previous step as its matched baseline:

1. BF16 master and execution weights for every band.
2. Existing ATMA FP8 path or MXFP8 on the narrow sparse band only.
3. MXFP8 on all sparse bands.
4. Promote one wide band at a time to MXFP4, beginning with inference-only weight quantization.
5. Quantized-weight training with BF16/FP32 gradient accumulation and optimizer state.

Track loss delta, output divergence, saturation and zero rates, block-scale distributions,
gradient cosine similarity, update error, convergence tokens, and run-to-run variance. Add BF16
shadow checks and automatically promote sensitive experts or layers from MXFP4 to MXFP8. Report
refresh traffic, quantization time, optimizer time, and weight staleness; host-resident masters
solve capacity, not necessarily throughput.

### 9.4 Sequencing and composition gate

Use a winner-picking sequence, not a Cartesian product:

```text
oracle sparse Polar -> learned on-GPU index -> paged/offloaded runtime
BF16 multi-band MoE -> fused kernels -> optional Head Parallel
BF16 MoE winner -> MXFP8 -> one-band-at-a-time MXFP4
independent winners -> pairwise interactions -> complete composition
```

Before the final composition, run the smallest useful factorial check for interaction effects:
full vs. sparse Polar, dense MLP vs. BF16 MoE, and BF16 vs. quantized MoE. The combined proposal is
rejected in its strongest form if approximation errors compound or if it fails to improve the
quality-latency-memory Pareto frontier over a simpler query-sparse Polar plus conventional
quantized top-2 MoE baseline.

**Shared evaluation envelope.** Report quality and systems results together:

- validation loss/perplexity, long-context perplexity drift, RULER/needle recall, and generation
  divergence;
- direction/count/null fidelity for Polar and utilization/specialization for experts;
- prefill and decode separately, at batch 1 and throughput-oriented continuous batching;
- HBM, host RAM, PCIe/NVLink bytes, overlap fraction, page-cache hit rate, grouped-GEMM efficiency,
  and energy/token where available;
- equal-quality, equal-active-FLOP, equal-memory, and equal-wall-clock comparisons.

Relevant starting points are [Quest](https://arxiv.org/abs/2406.10774),
[InfiniGen](https://arxiv.org/abs/2406.19707),
[RetrievalAttention](https://arxiv.org/abs/2409.10516),
[Native Sparse Attention](https://arxiv.org/abs/2502.11089),
[MoBA](https://arxiv.org/abs/2502.13189), the
[OCP MX specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf),
and [Multi-Head LatentMoE / Head Parallel](https://arxiv.org/abs/2602.04870).

---

## 10. Two unifications our project already spans

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
