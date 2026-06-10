# Titans Compression Memory (MAG)

A Titans-style ([arXiv 2501.00663](https://arxiv.org/html/2501.00663v1)) linear compression
memory added to [Polar Attention](POLAR_ATTENTION.md) as an **additive third channel**. The
goal: give the attention layer a *length-invariant long-term memory* so the model gets **both**
low perplexity at length **and** distant retrieval — the two properties the polar diagnosis
showed a single attention core cannot deliver at once.

> Status: integrated and parity-verified in **training** ([train/model.py](train/model.py)) and
> the **reference** ([model/reference.py](model/reference.py)); the standalone recurrence is
> float64-gradchecked ([verify_titans.py](verify_titans.py)). First end-to-end training runs
> (2026-06-04) confirm the memory earns its cost. The inference path is **ported to the paged
> engine** (2026-06-10; CPU-verified, GPU validation pending), and the *softmax-SWA-vs-polar
> ablation* (does the polar core still pay for itself once memory is present?) is **deferred**
> — see [Open & deferred work](#9-open--deferred-work).

---

## 1. Motivation — the window-vs-retrieval tradeoff

The polar [extrapolation diagnosis](POLAR_ATTENTION.md) reached a sharp verdict that a single
attention core cannot resolve:

- **Sliding window** (`W ≈ train length`) keeps the participation ratio `n_eff` in-distribution
  → **wins perplexity, but is retrieval-blind** past `W`.
- **Full polar** preserves distant recall (an induction needle is recalled far beyond train
  length *with the distractor on*) but **leaks perplexity** — `n_eff` explodes, the count
  channel shifts the residual DC level, and downstream layers go out-of-distribution.

> *"Window wins perplexity, full wins retrieval; neither gives both. Compression/recurrent
> memory is the only path to BOTH, and it must be trained in-loop."*

Titans **Memory-as-Gate (MAG)** is structurally that resolution: a precise short-term branch
(the windowed polar core) combined with a compressed long-term memory branch. We attach the
memory as an **additive third channel** rather than replacing polar's sigmoid gate, so the
memory can *inject* recalled content into the residual (a multiplicative gate on a unit
direction vector cannot):

```
out = content + count + memory
      └── polar direction ─┘   └ Titans memory readout
          + magnitude
```

`content` and `count` are unchanged from [`PolarAttention`](POLAR_ATTENTION.md#assembly). The
new `memory` term is the readout of a per-head linear associative memory updated by a **gated
delta rule**.

---

## 2. Memory math

Each head carries a matrix memory `M ∈ ℝ^{d_v × d_k}` (a fast-weight key→value store). It reuses
the layer's `q, k, v` (the same projections the polar core uses, KV-expanded to `H` heads), plus
two small data-dependent scalar gates per head from linear heads on `x`:

- **retention** `γ_t = σ(W_γ · x_t + b_γ) ∈ (0,1)` — per-head forget gate (`b_γ` init `3.9`
  → `σ ≈ 0.98`, a long horizon).
- **write strength** `β_t = σ(W_β · x_t + b_β) ∈ (0,1)` (`b_β` init `0` → `0.5`).

### Recurrence (Gated DeltaNet / flash-linear-attention convention)

The Titans neural memory with momentum `η = 0` reparametrizes **exactly** to the Gated DeltaNet
recurrence. We adopt FLA's convention as canonical (decay-first, undecayed write,
self-inclusive readout):

```
M_t = γ_t · M_{t-1} · (I − β_t · k_t k_tᵀ)  +  β_t · v_t k_tᵀ
r_t = M_t · q_t                              # readout AFTER the write (self-inclusive)
```

Per step, read decay → predict on the decayed state → write the undecayed correction → read out:

```
M   ← γ_t · M                       # decay first
pred = M · k_t                       # predict on the decayed state
M   ← M + β_t · (v_t − pred) k_tᵀ    # delta write (undecayed)
r_t  = M · q_t                       # self-inclusive readout
```

### Readout assembly

The readout is pulled back into polar's discipline before it enters the residual:

```
mem = proj( RMSNorm(r) · σ(gate(x)) )      # proj zero-init → branch starts as a no-op
```

The RMSNorm re-imposes a fixed output scale (the same reason polar projects `c` to the unit
sphere); the sigmoid gate lets the layer route the memory in per head-channel; `proj` is
zero-initialized so enabling the branch on a trained polar checkpoint is a safe no-op at step 0.

Implementation: [`TitansMemory`](model/blocks.py) (class), wired in
[`PolarAttention.forward`](train/model.py) as `out = out + self.mem(x, q_t, k_t, v_t)`.

---

## 3. Two load-bearing findings (validated in the prototype)

Both were discovered in [titans_proto.py](titans_proto.py) / [verify_titans.py](verify_titans.py)
before touching the model, and both **corrected the original plan's premise**.

### Finding 1 — memory keys must be **L2-normalized (unit norm)**, not RMS-normed

The delta rule's per-step eigenvalue in the key direction is `γ·(1 − β‖k‖²)`. Polar's
`F.rms_norm` gives `‖k‖² = d_k`, so the eigenvalue is `≈ −7` → **expansive** → the memory state
diverges (the prototype oracle itself blew up to `~1e57`). Unit keys give `γ·(1 − β) ∈ (0,1)`,
stable. **The memory branch therefore L2-normalizes its own `q, k`** (via `F.normalize`, or
`use_qk_l2norm_in_kernel=True` in the FLA path) — distinct from the RMS-norm polar uses on the
same projections.

### Finding 2 — the delta rule **self-stabilizes**; `γ` is a horizon knob, not a safety gate

The plan assumed the forget gate was required to prevent an `n_eff`-style state-norm blow-up.
The invariance sweep (`N` from 256 → 16384) showed that is true only for a **Hebbian**/linear-
attention memory (state-norm grows `~√N`: 45.9 → 374 over 64×). The **delta** memory's
state-norm is **flat in N** at both `γ = 0.98` (~11) and even `γ = 1` (~18.6) — the
`(I − β k kᵀ)` key-replacement term makes it converge toward the least-squares solution
`M ≈ V K⁺` (capacity ~`d_k`), self-stabilizing in norm.

So `γ` is **not** a stability requirement here — it is a **temporal-horizon / recall-vs-
perplexity dial**: `γ < 1` forgets by recency (tightest length-invariance, but *decays* a
distant planted needle); `γ = 1` retains until overwritten (capacity-limited). Hence `γ` is
learned **per-head and data-dependent** (some heads can hold `γ ≈ 1` for distant recall, others
`γ < 1` for recency) — not pinned to a constant.

---

## 4. Chunked-parallel form & the FLA fused kernel

The recurrence is sequential, but it has a closed **chunkwise-parallel** form so it trains
in-loop cheaply. There are two interchangeable backends, selected by `mem_kernel`:

### `torch` — `gated_delta_chunked` ([model/blocks.py](model/blocks.py))

A pure-PyTorch UT-transform: within a chunk the running-state coupling is removed by an exact
unit-lower-triangular solve
`(I + diag(β)·D)·U = diag(β)·(V − C_carry)`, with `D[p,s] = (g_p/g_s)(k_s·k_p)` for `s<p` and
inclusive decay `g_p = ∏_{l≤p} γ_l`; the readout mask is self-inclusive (`s ≤ p`). It is
sequential only **across** chunks; intra-chunk is batched matmuls + one `solve_triangular`.
Decorated `@torch.compiler.disable` — the unrolled chunk loop + `linalg` solve blows up the
inductor compile (it pinned CPU for minutes and crashed a pod), so it runs eager.

### `fla` — flash-linear-attention's `chunk_gated_delta_rule`

The fused CUDA/Triton fast path. Mapping: `g = logsigmoid(γ_logit)` (log-decay),
`beta = σ(β_logit)`, `scale = 1.0` (washes out under the post-readout RMSNorm),
`use_qk_l2norm_in_kernel=True` (Finding 1). `mem_kernel="auto"` picks FLA when it is installed
and the tensors are on CUDA, else the torch path.

> **Convention note.** The FLA mapping was wrong on the first GPU check (`rel_err 0.25`). FLA's
> naive reference revealed its recurrence differs from a naïve delta rule in **two** ways:
> *(a)* decay-first (`M ← γM`, predict on the decayed state, write undecayed →
> `M_t = γ M_{t-1}(I − β k kᵀ) + β v kᵀ`), and *(b)* the readout `M_t q_t` is **self-inclusive**
> (post-write), not `M_{t-1} q_t`. **Resolution: FLA's convention is now canonical** — both the
> prototype oracle and `gated_delta_chunked` were rewritten to it, so all three agree.

### The torch.compile / Triton interop (perf saga)

FLA's kernel is a custom `autograd.Function`. Under `torch.compile` it **graph-breaks at every
memory layer**, losing fusion and cross-break memory planning — that was the **~2× time / ~2×
peak-RAM** regression (the memory's raw FLOPs are only ~5% of the model, so the cost was
overhead, not compute). The fix:

- **Default** (`_fla_gated_delta = _fla_raw`): a plain call. Graph-breaks, but known-correct and
  trains fine.
- **Opt-in `FLA_CUSTOM_OP=1`**: wraps FLA as **two opaque custom ops** — `atma::fla_gd_fwd` and
  `atma::fla_gd_bwd`, each with a `register_fake` (so dynamo gets the output shape *without*
  running the Triton kernel → no graph break). The forward's `register_autograd` calls the
  opaque backward op, which **recomputes** FLA's autograd eagerly (correct grads, and the fwd
  activations are recomputed rather than stored → also cuts peak RAM). Both fwd and bwd must be
  opaque: a single custom op with a Python backward fails because AOTAutograd traces the
  backward into the joint graph and re-enters the Triton autotuner
  (`unhashable type: non-nested SymInt`).

This opt-in path is what removed the regression: see [§7](#7-empirical-results) for the
measured overhead (~6–9% relative MFU, not 2×).

---

## 5. Configuration

In [model/config.py](model/config.py):

| Field | Default | Meaning |
|---|---|---|
| `mem_enabled` | `True` | add the Titans memory branch (`out += mem`) |
| `attn_window` | `1024` | train-time causal sliding window for the polar core (`None` = full); the MAG short branch. Acts only at extrapolation when `≥ train length`. |
| `mem_chunk` | `128` | chunk size for the gated-delta parallel scan (GPU-tuned: 128 ≈ 2× faster than 64) |
| `mem_gamma_bias` | `3.9` | retention logit init: `σ(3.9) ≈ 0.98` (long horizon) |
| `mem_beta_bias` | `0.0` | write-strength logit init: `σ(0) = 0.5` |
| `mem_kernel` | `"auto"` | `"auto"｜"fla"｜"torch"` gated-delta backend |

> Defaults with `mem_enabled=False` / `attn_window=None` leave the model **byte-identical** to
> plain polar attention (so [verify.py](verify.py) is unchanged). Enable `mem_enabled` and set a
> finite `attn_window` together for the MAG configuration. The `FLA_CUSTOM_OP=1` env var
> (separate from the config) is what makes the FLA path compile-clean — set it for real runs.

---

## 6. Verification (CPU, bit-exact)

The recurrence and its integration are validated to floating-point tolerance against
token-by-token oracles, mirroring the polar workflow:

| Script | Checks |
|---|---|
| [verify_titans.py](verify_titans.py) | `gated_delta_chunked` == sequential scan, forward (~1e-15) + backward (~1e-14) + fp64 `gradcheck`. |
| [verify_polar_window.py](verify_polar_window.py) | the trainable windowed polar core (band-aware backward) == a masked materialized oracle (fp64 gradcheck), windows `{None, 4, 7, 32}`. |
| [verify_mag.py](verify_mag.py) | train == reference **bit-exact** with the memory branch active + window; zero-init `proj` verified to be an exact no-op. |
| [verify_fla.py](verify_fla.py) | (GPU) FLA path vs the torch reference; expect `rel_err < 0.05` (bf16-vs-fp32 only). Also smoke-checks `FLA_CUSTOM_OP=1` grads. |

The invariance sweep in [titans_proto.py](titans_proto.py) is the Finding-2 gate: memory
state-norm and readout RMS must be flat across `N` (256 → 16384) before any perplexity claim.

---

## 7. Empirical results

First end-to-end MAG training runs (2026-06-04; two models, `seq_len=2048`,
`attn_window=1024`, `mem_enabled=True`, ± distractor; evaluated on finepdfs). **The memory earns
its cost.**

### Quality

Convergence rate and perplexity rank:

```
Polar + Titans   >   Causal Self-Attention   >   Polar only
```

The memory is the **quality driver**: it flips polar from *below* a vanilla transformer to
*above* it. Notably, **polar-only is dominated by plain causal attention** (worse perplexity
*and* slower per step) — the polar machinery pays for itself only through length extrapolation;
the memory is what makes the combined stack beat a standard transformer.

### Length extrapolation (perplexity)

On coherent documents, full-attention perplexity is now **best and monotonic** — `1.93 @ 64×` —
a **reversal** of the original diagnosis where full attention was *worst* (`~3.21 @ 64×`). The
improvement (~1.3 nats @ 64×) is attributable to the **memory**, not windowing: with
`n_count = min(n_keys, 1024)` the polar core trains at the same `N ≤ 1024` operating point
whether full-@1024 or windowed-@2048, so eval-at-full is equally OOD for both — the only new
variable vs the prior no-memory runs is the memory branch.

### Retrieval (induction needle)

| Config | 2048 → 65536 tokens |
|---|---|
| full polar **+ distractor** | 71% → 42% |
| full polar, no distractor | 27% → 12% |

Pinpoint retrieval needs the **distractor** (`num_random_keys > 0`) on the full-polar core. The
linear gated-delta memory is a **lossy gist** (capacity ~`d_k`): it helps diffuse perplexity at
length but *cannot* exact-recall a random planted needle. So memory-only eval columns reading at
chance is **expected** — the division of labor is clean:

```
memory          = lossy gist  → perplexity at length
full polar + distractor = exact pinpoint retrieval (the needle)
```

### Ablation — memory is essential

[eval.py](eval.py) `--no_mem` (sets `attn.mem = None` on the checkpoint) breaks the model
**globally**: loss `2.8 → 5.7` at 1×, needle `0%` everywhere, baseline CE `5.98 → 8.43`. The
trained model relies on the memory at *all* lengths — it is a general capacity/convergence
contribution, not just a long-context add-on. (The ablation is too blunt to isolate the
*length-specific* role; it confirms the branch is load-bearing.)

### Overhead

The `FLA_CUSTOM_OP=1` opaque-custom-op path brings the memory overhead down from the ~2×
graph-break regression to **~6–9% relative MFU** (`seq_len=2048`):

| Config | MFU without memory | MFU with memory | relative cost |
|---|---|---|---|
| + distractor (`num_random_keys=2048`) | 30.1% | 28.4% | ~6% |
| no distractor | 36.1% | 32.9% | ~9% |

Per-step **speed** ranks the reverse of quality:
`Causal > Polar only > Polar + Titans`. The combined stack is the slowest but the best.

---

## 8. File map

| File | Role |
|---|---|
| [model/blocks.py](model/blocks.py) | `TitansMemory` module, `gated_delta_chunked` (torch backend), `_fla_gated_delta` (FLA wrapper + `FLA_CUSTOM_OP` opaque custom ops), band-aware `_PolarOnline` backward |
| [train/model.py](train/model.py) | `PolarAttention` memory branch (`out += self.mem(...)`) + trainable window |
| [model/reference.py](model/reference.py) | reference mirror (train == reference parity) |
| [model/config.py](model/config.py) | `mem_enabled`, `attn_window`, `mem_chunk`, `mem_gamma_bias`, `mem_beta_bias`, `mem_kernel` |
| [kernel/polar_triton.py](kernel/polar_triton.py) | windowed (`WINDOW`) band mask in the polar fwd/bwd kernels (GPU band-backward **unvalidated** — see below) |
| [titans_proto.py](titans_proto.py) | standalone oracle (sequential scan) + chunked form + invariance sweep |
| [verify_titans.py](verify_titans.py), [verify_polar_window.py](verify_polar_window.py), [verify_mag.py](verify_mag.py), [verify_fla.py](verify_fla.py) | parity / gradcheck / integration tests |
| [bench_mem.py](bench_mem.py) | GPU profiling harness (chunk size, tri-solve vs Neumann, dtype, eager vs compiled) |

---

## 9. Open & deferred work

- **Softmax-SWA-vs-polar ablation (future task).** The quality ranking shows *Titans* is the
  driver and *polar-only* is dominated by plain causal attention. The open question is whether
  the polar core still pays for itself once the memory is present — i.e. does **softmax sliding-
  window + Titans** (vanilla Titans-MAG) match **polar windowed + Titans**, especially on the
  eval-at-full needle where softmax `n_eff` blows up? This decides whether the polar half is
  carrying its weight. **Not yet run.**
- **Step 4 — write-path distractor.** Symmetry `null-sink : attention :: write-gate : memory`:
  extend the distractor so random keys also incur a *write* penalty (memory must not memorize
  noise). Reuse the `align_loss` plumbing. **Not yet implemented.**
- **Triton band-backward (GPU).** The windowed band mask in
  [kernel/polar_triton.py](kernel/polar_triton.py) mirrors the validated forward + torch
  backward but is **unvalidated on CUDA** (no GPU in the dev loop) — gradcheck it on a box.
- **`mem_layers` perf lever.** Restricting the memory to the last N attention layers is a
  compile-agnostic overhead cut, offered but not built (overhead is already ~6–9%, so low
  priority).
- **Inference port — done** (2026-06-10). The paged engine carries the per-head `M` in fp32
  per-sequence state tables, FLA `[K, V]` layout ([inference/models/atma.py](inference/models/atma.py)):
  `_mem_prefill` runs FLA's `chunk_gated_delta_rule`, `_mem_decode` FLA's
  `fused_recurrent_gated_delta_rule` (batched T=1 step), both with
  `initial_state`/`output_final_state` for the state carry; the pure-torch paths remain as
  CPU fallback. CPU-verified via `verify.py`; validate the FLA bridge on GPU with
  `verify_fla.py` (inference-bridge section).
- **Momentum / deep memory.** Titans' momentum term (`η_t`) and the deep-MLP memory are deferred
  — revisit only if the linear gated-delta version caps out.
