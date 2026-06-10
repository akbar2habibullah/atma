# Polar Attention

A length-invariant replacement for softmax scaled-dot-product attention (SDPA), used
in place of `CausalSelfAttention` in the atma model. The goal: **train at short
sequence length, infer at any length.**

> Status: integrated and parity-verified in **training** ([train/model.py](train/model.py)),
> the **reference** ([model/reference.py](model/reference.py)), and the **paged inference
> engine** ([inference/models/atma.py](inference/models/atma.py); prefill kernel + paged
> decode kernel, CPU-verified — GPU validation pending). See
> [Limitations](#limitations--deferred-work).

---

## 1. Motivation

Softmax attention is numerically stable (the output is a convex combination of values,
bounded regardless of sequence length), but it is **not length-invariant**:

- As the number of keys grows, softmax mass spreads thinner ("dilution"), so a model
  trained at length `T` behaves differently at `32·T` — it never learned to scale its
  computation with length.
- The magnitude information ("*how many* things matched") is entangled with the
  direction information ("*what* matched") in a single normalized output.

The inspiration is the **size-invariance of convolution**: a conv kernel extracts local
features identically on a 100×100 grid or a 10⁶×10⁶ grid. We want that property for the
global mixing step, plus a principled, magnitude-bounded way to represent *how much* was
found, with **infinite dynamic range** (1,000 vs 1,000,000 matches should remain
distinguishable but bounded, e.g. `0.88` vs `0.99`, never `1000×`).

Polar attention factors each query's result into two channels:

| Channel | Meaning | Property |
|---|---|---|
| **direction** `c` | *what* was attended to (unit vector) | size-invariant, count-blind |
| **magnitude** `mag` | *how much* / how many effective matches | bounded in `[0,1)`, length-invariant, ordered |

Both channels are derived from **one** temperature-sharpened softmax with a learned null
sink, then assembled back into the residual stream.

---

## 2. Mathematical definition

Let a single query `i` attend over keys `j` (causal: `j < n_i`, where `n_i` = number of
visible keys = `i+1`). Scores are `σ_ij = (q_i · k_j) / √d_k` (with QK-norm applied to
`q`, `k`). Per-head learned scalars drive two length-aware quantities:

**Length temperature** (sharpens the softmax as context grows — "Scalable-Softmax" style):

```
temp_i = 1 + softplus(len_gain_raw) · log(n_i)
```

**Extreme-value-corrected null floor** (the logit of an off-by-one "null sink" key):

```
null_i = null_base + softplus(null_slope_raw) · sqrt(log(n_i + 1))
```

The `√(log n)` growth is essential: the *maximum* of `n` noisy scores grows like
`√(2 ln n)`, so any **fixed** threshold is eventually overtaken by noise. The floor must
track it.

### Direction channel ("what")

Append the null logit to the real-key logits, softmax, and form a convex combination of
values (plus a learned default direction `v_null` for the null sink), then project to the
unit sphere:

```
logits_ij = temp_i · σ_ij        (real keys)
logits_iN = temp_i · null_i      (null sink)
w_i        = softmax([logits_i•, logits_iN])          # over real keys + null
s_i        = Σ_j w_ij · v_j  +  w_iN · v_null
c_i        = s_i / ‖s_i‖                               # unit vector
```

`c_i` is invariant to how many keys matched and to total length — it is the "what".

### Magnitude channel ("how much")

Reuse the **same** softmax weights. The count is the **participation ratio** (effective
number of attended real keys), gated by confidence `(1 − w_null)`, then bounded:

```
ŵ_ij   = w_ij / Σ_k w_ik              # renormalize over real keys only
n_eff_i = 1 / Σ_j ŵ_ij²              # participation ratio (inverse Simpson index)
m_eff_i = n_eff_i · (1 − w_iN)        # gate by "is there real signal?"
mag_i   = tanh( softplus(mag_beta_raw) · log1p(m_eff_i) )    ∈ [0, 1)
```

- `n_eff` counts **effective** matches: 1,000 strong keys → `n_eff ≈ 1000`; pure noise →
  a small bounded number (the temperature keeps it from growing with length).
- `tanh(β·log1p(·))` is the **bounded, saturating** map: monotone, `0` at zero count,
  `→1` as count `→∞`. This keeps the count-projection's input in-distribution at any
  length (raw `log(m)` would grow without bound and push the projection out of
  distribution).

### Assembly

The two channels are recombined into the residual stream. The direction is gated by a
sigmoid gate (carried in the `q` projection) and projected; the magnitude is injected
additively through a tiny per-head projection:

```
content = W_o( reshape(c) · sigmoid(gate) )     # W_o = self.proj  (hdim → dim)
count   = W_mu( mag )                            # W_mu = self.mu_proj  (num_heads → dim)
out     = content + count
```

---

## 3. Design rationale (what was validated, and what failed)

The formula was derived by **falsification** in a standalone sandbox before integration
(that prototype script has since been removed). The non-obvious decisions:

| Decision | Rejected alternative | Why |
|---|---|---|
| count = participation ratio `1/Σŵ²` | additive soft-count `Σ sigmoid(σ−θ)` | the additive sum leaks `O(N)`: a tiny per-key leak × huge `N` accumulates (89× drift over a 1000× length sweep). PR is flat (1.0×). |
| — | relative floor `mean + κ·std` | also `O(N)` — the noise tail above any fixed number of std's is a constant *fraction* of `N`. |
| bounded `tanh(β·log1p(m))` | raw `log(m)` | raw log is unbounded → the count projection sees out-of-distribution inputs at long context. Bounded map keeps it in range and also damps residual drift. |
| EV null floor `+ slope·√(log n)` | fixed null logit | a fixed floor is overtaken by the noise maximum `√(2 ln n)`; pure noise then reads as a strong match at long context. |
| distractor + task counterweight | distractor alone | minimizing only "reject noise" drives the floor `→∞` and collapses the count channel (kills real signal too). The task loss is the counterweight. |

**Length-invariance condition.** The participation-ratio count is length-invariant when
`len_gain · Δ ≥ 1`, where `Δ` is the QK separation margin between signal and noise. The
temperature sharpening then suppresses each noise weight faster than `N` grows. End-to-end,
the same key population at 1× and 64× length produces identical `c` (cosine 1.0) and
identical `mag` (e.g. `0.27` for 3 matches, `0.66` for 50).

---

## 4. Auxiliary objective: distractor loss

The null floor and `Q` geometry are calibrated by a **distractor loss** (returned as the
block's `align_loss`, summed into the training objective). Random keys are projected
through the same `k` projection and must lose to the null sink:

```
align_loss = mean over (queries, heads) of  softmax([ temp·σ_rand , temp·null ]) restricted to the random keys
```

This is `O(T·R)` (R = `num_random_keys`), memory-friendly, and trains `null_base`,
`null_slope_raw`, `len_gain_raw`, and `Q` to push random keys below the floor. Real keys
staying *above* the floor is the task loss's responsibility (the counterweight that
prevents collapse). Disabled when `num_random_keys = 0` (the default).

> The polar-Zipf orthogonality regularizer from the original proposal was **deferred** —
> it was not load-bearing in validation. The existing residual-stream `sigreg`
> ([train/reg.py](train/reg.py)) still runs.

---

## 5. Memory: online (FlashAttention-style) softmax

The materialized path builds the full `(B, H, T, T)` score matrix — `O(T²)` memory. The
**online** path ([`polar_attention_online`](model/blocks.py)) streams keys in blocks of
`k_block`, maintaining running accumulators, for **`O(B·H·T·k_block)` memory in both the
forward and backward passes**.

Beyond a vanilla flash kernel, the count channel needs **one extra streamed accumulator**:

```
M  = running max of (temp·σ)            # numerical stability
L  = Σ exp(temp·σ − M)                   # softmax denominator (real keys)
S  = Σ exp(temp·σ − M) · v               # direction numerator
Q2 = Σ exp(temp·σ − M)²                  # for the participation ratio  ← extra
```

On each max update by `α = exp(M_old − M_new)`: `L *= α`, `S *= α`, **`Q2 *= α²`** (the
squared accumulator rescales by the square of the correction). After folding the null sink
(`Z = L + p_null`), the channels are recovered exactly:

```
c     = normalize(S + p_null · v_null)
n_eff = L² / Q2
m_eff = n_eff · (L / Z)            # = L³ / (Q2 · Z)
```

It is implemented as a custom `torch.autograd.Function` with a hand-written streaming
backward (autograd alone would retain the full `O(T²)` graph and defeat the purpose). All
quantities are invariant to the max shift `M`, so `M` is treated as a detached constant in
the backward — the standard flash trick.

**Correctness** is established by float64 `gradcheck` and by matching the materialized
oracle in forward and backward to ~1e-15 (see [§9 Verification](#9-verification)).

> **Scope:** this reduces the key dimension to `O(k_block)` but is still `O(T)` in the
> query dimension (each query's row reduction is computed for all queries at once). Adding
> query-blocking would make it fully `O(k_block)`; deferred until query-dim memory is the
> bottleneck.

### FlashAttention-style Triton kernel ([kernel/polar_triton.py](kernel/polar_triton.py))

A fused Triton kernel implements the same streaming reduction with **query-blocking too**
(fully `O(block)` memory) and runs on tensor cores. It reproduces the online path to
floating-point tolerance (~3e-7 fp32; gradchecked-oracle parity on all inputs and params)
and is **7–27× faster** with **~5× less memory** than the PyTorch online/materialized
paths on an L4 (see [kernel/README.md](kernel/README.md)). The backward keeps the cheap
per-query preamble in PyTorch and runs only the two `O(T²)` matmul loops (`dq`, `dk/dv`)
as Triton kernels. Opt in with `AtmaConfig(attn_kernel="triton")`; forward-only
`polar_attention_fwd` serves inference (causal prefill, or `is_causal=False` + explicit
`n_keys` for decode / offset prefill).

---

## 6. Architecture & integration

`PolarAttention` subclasses `AtmaAttnBase` ([model/blocks.py](model/blocks.py)) and keeps
the surrounding architecture identical to `CausalSelfAttention`:

- **Projections.** `q` (outputs `2·hdim`: query + sigmoid gate), `k`, `v`
  (`kv_hdim`, GQA), and `proj` (= `W_o`). QK-norm (`F.rms_norm` per head) on `q`, `k`.
- **Canon conv.** Depthwise causal `conv1d` horizontal residual on `q`/`k`/`v`
  (`canon_q/k/v`), exactly as in the existing attention.
- **GQA.** `num_kv_heads = num_heads // 4` (1:4); KV heads are `repeat_interleave`-expanded
  to `num_heads` before scoring.
- **Polar-specific parameters** (per head, added by `PolarAttention`):

  | Param | Shape | Init (raw) | Effective at init |
  |---|---|---|---|
  | `mu_proj` (`W_mu`) | `Linear(num_heads, dim)` | — | count → residual |
  | `v_null` | `(H, d_k)` | `0` | null-sink default direction |
  | `null_base` | `(H,)` | `2.0` | floor offset |
  | `null_slope_raw` | `(H,)` | `0.5` | `softplus ≈ 0.97` (√log n gain) |
  | `len_gain_raw` | `(H,)` | `-1.0` | `softplus ≈ 0.31` (temperature gain) |
  | `mag_beta_raw` | `(H,)` | `-1.5` | `softplus ≈ 0.20` (magnitude slope) |

The shared reduction lives in [model/blocks.py](model/blocks.py) (`polar_temp_null`,
`polar_reduce`, `polar_attention_online`) so the **training and reference forward passes
call identical math** — train == reference is **bit-exact** in the default (materialized)
configuration.

---

## 7. Configuration

In [model/config.py](model/config.py):

| Field | Default | Meaning |
|---|---|---|
| `num_random_keys` | `0` | distractor count `R`; `>0` enables the null-floor calibration loss |
| `attn_online` | `False` | use the streaming `O(T·k_block)` path instead of materialized `O(T²)` |
| `attn_k_block` | `512` | key block size for the online path |

> **Recommendation.** For a real length-extrapolation run, set `num_random_keys > 0` — the
> distractor loss is what buys the length budget (it widens the QK margin `Δ`), not
> optional. Enable `attn_online` for long sequences. Defaults keep the legacy
> `O(T²)`/no-distractor behavior and exact parity.

---

## 8. Implementation notes

- **dtype.** The reduction computes in fp32 for bf16/fp16 activations (preserving fp32/fp64
  inputs), then casts the outputs back. The model's bf16 path matches the fp32 reference at
  reference tolerance.
- **`-inf` × `temp` NaN (fixed).** Scaling must happen **before** masking. Multiplying
  causal `-inf` scores by `temp` yields a backward term `grad_temp = Σ grad_logit · σ =
  0·(-inf) = NaN`, which poisons `len_gain_raw`'s gradient on the first causal backward.
  `polar_reduce` neutralizes masked entries to a finite value for the temp product and
  re-applies `-inf` after; the online path masks with a finite sentinel. Guarded by a
  finite-backward regression test.
- **Online vs materialized** differ only by fp summation order (~1e-7 in fp32); they are
  the same function.

---

## 9. Verification

| Script | Checks |
|---|---|
| [verify.py](verify.py) | per-layer train == reference == inference (prefill + decode) parity for RMSNorm, MLP, LFM2 conv, **Polar attention** (incl. window + Titans mem + chunked prefill), full blocks, and the full model logits. 30/30. |
| [kernel/test_polar_kernel.py](kernel/test_polar_kernel.py) | Triton kernel vs the gradchecked oracle: forward + all gradients (fp32/bf16/fp16), edge shapes (T=1, T<block, odd T), non-contiguous inputs, `n_keys=0` padding. 104/104. |
| [kernel/test_integration.py](kernel/test_integration.py) | `train.model` PolarAttention + full Model with `attn_kernel="triton"` vs the torch path. 21/21. |

Run: `python verify.py`, `python -m kernel.test_polar_kernel`, `python -m kernel.test_integration`.

---

## 10. Limitations & deferred work

- **Inference ported** (2026-06-10). [inference/models/atma.py](inference/models/atma.py) runs
  polar in the paged engine: `polar_attention_fwd` per sequence in prefill (with prefix K/V
  gather for chunked prefill), the paged `polar_attention_decode` kernel in decode (reads the
  paged KV cache via block tables; CUDA-graph capturable), window + Titans memory included.
  `verify.py` passes 30/30 on CPU; the Triton decode path still needs GPU validation.
- **Query-dim memory.** The online path is `O(T)` in the query dimension (see §5).
- **Polar-Zipf regularizer** deferred (see §4).
- **`CausalSelfAttention`** classes remain in the codebase but are unused (the blocks now
  build `PolarAttention`).

---

## 11. File map

| File | Role |
|---|---|
| [kernel/polar_triton.py](kernel/polar_triton.py) | FlashAttention-style Triton kernel: fused fwd + hand-written bwd (`polar_attention`, `polar_attention_fwd`); 7–27× faster than the PyTorch paths |
| [model/blocks.py](model/blocks.py) | shared `polar_temp_null`, `polar_reduce` (materialized), `polar_attention_online` (streaming custom autograd); re-exports the Triton kernel |
| [train/model.py](train/model.py) | training `PolarAttention` (+ distractor → `align_loss`, online flag) |
| [model/reference.py](model/reference.py) | reference `PolarAttention` (materialized oracle) |
| [model/config.py](model/config.py) | `num_random_keys`, `attn_online`, `attn_k_block`, `attn_kernel` |
| [inference/generate.py](inference/generate.py) | standalone polar inference (checkpoint-seek + random fallback, Triton kernel) |
| [verify.py](verify.py), [kernel/test_polar_kernel.py](kernel/test_polar_kernel.py), [kernel/test_integration.py](kernel/test_integration.py) | parity, gradcheck-oracle, integration tests |
