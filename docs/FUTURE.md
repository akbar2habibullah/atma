# Future Directions

Tracking doc for potential development beyond the current architecture
([Polar Attention](POLAR_ATTENTION.md) + [Titans MAG memory](TITANS_MEMORY.md)). These are
**not committed work** — they are ideas, framings, and falsifiable experiments parked here so
they survive until there's bandwidth to pursue them.

> Most of the experimental items are **blocked on the 120-way ablation sweep** (5×2×2×2×3,
> see [ablation/README.md](../ablation/README.md)) currently running (~2–3 weeks). Do not start
> new diagnostic interventions until that grid lands — they are *post-sweep* candidates.

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

## 4. Wall Attention (Tilde Research) — post-sweep axis

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

**Candidate cells for the *next* sweep (after the current 120-way):**

- `attn_type += wall` — drop-in NoPE positional, FA-compatible baseline (the ablation harness
  already has rope/nope/polar as an axis).
- **Wall-scores → polar-normalization** — does Wall's adaptive kernel reduce polar's `n_eff`
  ramp on the `--diagnose` probe? (Note: Wall+polar still inherits polar's non-FA inference.)
- **Wall vs Titans on the window-vs-retrieval cell** — does per-channel gating make the gist
  memory partially redundant *for quality*, while the memory still wins on compute/KV-cache?

**Status — wired as a 2nd-batch axis (2026-06-20).** `attn_type="wall"` is implemented in
[train/model.py](../train/model.py) (`CausalSelfAttention`, `pos="wall"`): **keeps canon** (so it's
the matched comparison to `nope` — isolates the gating; all params used → no Muon issue), adds a
per-channel log-forget gate `g = -softplus(W_g·x + wall_gate_bias)` (slow-forget init), and applies
Wall's score `q_i·k_j·exp(P_i−P_j)` per channel via the stable rescale `q̃=exp(P)q, k̃=exp(−P)k`
into standard attention. Two backends: a **pure-PyTorch fallback** (recentered prefix sum →
exact at the training length, compile-friendly, CPU-testable; used for the compiled training pass)
and **Tilde's `wall_attn` Triton kernel** (per-chunk anchors → faithful at long context; used at
eval on CUDA when installed). The 40 wall cells (5×2×2×2) are generated at
[ablation/shards/shard5](../ablation/shards/shard5) (grid is now 160). **Caveat:** the torch fallback
recenters+clamps the prefix sum, which is exact only while the centred range is small (≈ train
length); for faithful long-context eval (>~4k) the host must `pip install` the `wall_attn` kernel —
validate it (à la `verify_fla.py`) before trusting wall's 65k needle/perplexity numbers.

**Caveat.** All Wall numbers are single-source (Tilde blog, 1B scale, their benchmarks). The
mechanism is sound and the bimodal result is credible *because* it echoes our own retention
spectrum — but treat "beats RoPE+FoX / SOTA" as promising, not settled, until run in our harness.

---

## 5. Two unifications our project already spans

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
