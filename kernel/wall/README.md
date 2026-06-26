# Wall Attention Kernel Strategy

This directory is reserved for an in-tree Wall Attention kernel fork. Keep it separate from
`kernel/polar_triton.py` so the Wall-specific math, validation, and performance work do not blur
the boundary with the existing Polar kernel.

## Context

The training model currently imports Tilde Research's package-level kernel:

```python
from wall_attn import wall_attn as _wall_attn_kernel
```

and calls it from `train/model.py` for `attn_type="wall"`. The upstream reference is at:

- https://github.com/tilde-research/wall-attention-release
- https://github.com/tilde-research/wall-attention-release/blob/main/wall_attn/reference.py
- https://github.com/tilde-research/wall-attention-release/blob/main/wall_attn/training.py

The reference implementation is only a correctness oracle. It materializes pairwise `T x T`
objects and should not be used to reason about training memory. The relevant implementation is
`wall_attn/training.py`.

## Problem

On an NVIDIA L4 24 GB, other attention variants train around 11 to 12 GB at `mbs=4`, while Wall
Attention still uses roughly 15 GB at `mbs=2`. That suggests the remaining overhead is inside the
Wall autograd function, not only in our Python wrapper.

The likely large allocations in upstream `training.py` are:

```python
g_cumsum = chunk_global_cumsum(g, ...)
ctx.save_for_backward(q, k, v, o, g_cumsum, g_scalar_cumsum, lse, ...)
```

and, in backward, GQA-shaped temporary gradients:

```python
dq = torch.empty(B, T, HQ, K, ...)
dk = torch.empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float)
dv = torch.empty(B, T, HQ, V, dtype=v.dtype if H == HQ else torch.float)
dg_cumsum = torch.empty(B, T, HQ, K, dtype=torch.float)
dg_cumsum_k = torch.empty_like(dg_cumsum)

dk = reduce(dk, "b t (h g) k -> b t h k", g=G, reduction="sum")
dv = reduce(dv, "b t (h g) v -> b t h v", g=G, reduction="sum")
```

For our default config, `HQ=8`, `H=2`, `G=4`, `K=V=128`. Allocating `dk` and `dv` as
`(B,T,HQ,K/V)` in fp32 and reducing later is expensive. `dg_cumsum` and `dg_cumsum_k` are also full
`(B,T,HQ,K)` fp32 tensors.

## Goal

Build a low-memory Wall training kernel path that keeps exact forward and backward behavior within
normal floating point tolerance, while reducing peak memory on GQA models.

Initial target:

- Same API shape as upstream `wall_attn(q, k, v, g, scale, window_size, ...)`.
- Preserve GQA: `q/g` have `HQ` heads, `k/v` have `H` KV heads, `HQ = H * G`.
- Avoid any `T x T` materialization.
- Prefer memory reduction over MFU for the L4 ablation sweep.

## Strategy

### 1. Vendor the upstream training kernel first

Copy upstream `wall_attn/training.py` into this directory as the baseline implementation before
making changes. Keep the original names initially so parity tests are easy.

Suggested files:

```text
kernel/wall/training.py
kernel/wall/reference.py
kernel/wall/__init__.py
kernel/wall/test_wall_kernel.py
kernel/wall/bench_wall.py
```

Do not mix this into `kernel/polar_triton.py`.

### 2. Add memory instrumentation before changing math

Create a small benchmark that measures both peak allocated and peak reserved memory:

```python
torch.cuda.reset_peak_memory_stats()
out = wall_attn(q, k, v, g, scale=scale, window_size=window)
loss = out.float().square().mean()
loss.backward()
peak_alloc = torch.cuda.max_memory_allocated()
peak_reserved = torch.cuda.max_memory_reserved()
```

Run at the ablation shape:

```text
B=2, T=2048, HQ=8, H=2, K=128, V=128
dtype=bf16
window_size in {None, 1024}
distractor packed length: T2 = 4096 when R=T
```

Also test `B=1` for eval-like long contexts:

```text
T in {2048, 4096, 8192, 16384, 32768, 65536}
```

### 3. Remove full-HQ `dk` and `dv` temporaries

Current upstream backward computes per-query-head `dk/dv` and reduces afterward. For GQA, this is
wasteful.

Preferred low-memory approach:

- Change the DKV backward kernel so each program accumulates directly into KV-head-shaped outputs:
  `dk_out: (B,T,H,K)` and `dv_out: (B,T,H,V)`.
- Inside the kernel, loop over or tile across the `G` query heads that share the same KV head.
- Accumulate the contribution from all query heads for that KV head in registers or block-local
  accumulators.
- Store one reduced result directly to `(B,T,H,K/V)`.

This avoids allocating:

```text
dk_tmp: (B,T,HQ,K) fp32
dv_tmp: (B,T,HQ,V) fp32
```

and avoids the separate `einops.reduce` allocation.

If direct reduction in one kernel is too awkward, use a two-stage low-memory path:

1. compute one KV head at a time into `(B,T,H,K/V)`;
2. accumulate with atomics or split by KV head so no full `HQ` temporary exists.

Prefer deterministic non-atomic accumulation first if practical.

### 4. Reduce `dg_cumsum` peak

The backward appears to create both query-side and key-side prefix gradients:

```text
dg_cumsum
dg_cumsum_k
```

Both are `(B,T,HQ,K)` fp32. Options:

- Compute query-side and key-side contributions sequentially into the same buffer if their
  lifetimes do not overlap.
- Accumulate key-side contribution into `dg_cumsum` in place.
- If the final gradient only needs reverse-cumsum of the sum, avoid storing two full tensors.

Be careful: this is the easiest place to introduce a subtle sign or prefix-direction bug. Keep a
small fp64/PyTorch oracle test for `g.grad`.

### 5. Consider recomputing `g_cumsum` in backward

Forward saves `g_cumsum` because the backward needs prefix positions. Saving it costs one full
`(B,T,HQ,K)` tensor.

Low-memory option:

- Save `g` instead of `g_cumsum`, or rely on `g` already being available from autograd inputs.
- Recompute `g_cumsum = chunk_global_cumsum(g)` in backward.

This trades compute for memory. It may be worthwhile for L4. If recomputing inside the autograd
function is difficult because of FLA helper APIs, keep it as a second optimization after `dk/dv`.

### 6. Preserve the packed distractor formulation

For wall distractor loss, `train/model.py` can represent random distractors as a random prefix:

```text
[random prefix length R] + [real sequence length T]
```

with zero gates for the random prefix. Causal Wall Attention over the packed sequence gives each
real query access to all random keys plus causal real keys. This avoids a custom `T x (R+T)` mask.

The kernel should support this path through ordinary causal attention with `window_size=None`.

## Validation

### Correctness

Compare against upstream `wall_attn_reference` for small shapes:

```text
B in {1, 2}
T in {8, 17, 64}
HQ/H in {(4, 4), (8, 2)}
K/V in {(32, 32), (64, 64)}
window_size in {None, 4, 16}
dtype in {float32, bfloat16}
```

Check:

- forward max error and relative error
- gradients for `q`, `k`, `v`, and `g`
- GQA reduction correctness for `k.grad` and `v.grad`
- packed distractor path with `R=T`

Use a tiny float64 or float32 PyTorch oracle where possible. The upstream reference should be fine
for small `T`.

### Memory

Record peak memory for:

```text
upstream wall_attn
local baseline copy
local low-memory backward
```

At minimum, report:

```text
B,T,HQ,H,K,V,dtype,window,R,peak_alloc,peak_reserved,elapsed_ms
```

The first win condition is that local low-memory backward is lower than upstream at:

```text
B=2, T=2048, HQ=8, H=2, K=128, V=128, dtype=bf16
```

The second win condition is that eval-like forward can complete as far as possible on L4:

```text
B=1, T up to 65536
```

## Performance Report

Measured on the local NVIDIA L4 24 GB box on 2026-06-26 with the isolated kernel benchmark in
`kernel/wall/bench_wall.py`. Each reported row uses one warmup iteration and one measured training
iteration:

```text
out = wall_attn(q, k, v, g, scale=K**-0.5, window_size=window)
loss = out.float().square().mean()
loss.backward()
```

The benchmark uses monotone nonpositive Wall gates, matching the model path in `train/model.py`
(`g = -softplus(...)`). The numbers below are isolated kernel allocation peaks, not full-model
training process memory.

### L4 GQA training backward target

Shape:

```text
B=2, T=2048, HQ=8, H=2, G=4, K=128, V=128, dtype=bf16, R=0
```

| impl | window | peak allocated | peak reserved | elapsed / status |
| --- | ---: | ---: | ---: | ---: |
| naive PyTorch reference | `None` | OOM | OOM | requested 32.00 GiB allocation |
| upstream `wall_attn` | `None` | 0.305 GB | 0.307 GB | 7.04 ms |
| local low-memory | `None` | 0.121 GB | 0.135 GB | 16.55 ms |
| naive PyTorch reference | 1024 | OOM | OOM | requested 32.00 GiB allocation |
| upstream `wall_attn` | 1024 | 0.305 GB | 0.307 GB | 6.16 ms |
| local low-memory | 1024 | 0.121 GB | 0.135 GB | 14.44 ms |

Result:

- The naive PyTorch reference cannot run this target shape on L4. It materializes pairwise
  `T x T` objects and failed before backward with `CUDA out of memory. Tried to allocate 32.00 GiB`.
- Against upstream `wall_attn`, peak allocated memory drops from 0.305 GB to 0.121 GB, about a 60%
  reduction.
- Against upstream `wall_attn`, peak reserved memory drops from 0.307 GB to 0.135 GB, about a 56%
  reduction.
- Runtime is slower in this isolated benchmark because the local DKV backward uses a fixed launch
  and atomic GQA accumulation to avoid full-HQ temporaries. This is an intentional memory-over-MFU
  tradeoff for the L4 ablation sweep.

### Numerical parity

The local low-memory fork keeps the upstream forward kernel and query-gradient kernel unchanged. The
only math-path changes are in DKV backward storage/reduction:

- upstream writes per-query-head `dk/dv` as `(B,T,HQ,K/V)` and reduces with `einops.reduce`;
- local writes directly into KV-head-shaped `(B,T,H,K/V)` outputs with atomic accumulation;
- upstream writes key-side `dP` to `dg_cumsum_k` then adds it to `dg_cumsum`;
- local accumulates key-side `dP` into `dg_cumsum` in place.

This is intended to be numerically equivalent to upstream within normal floating-point tolerance,
but not bitwise identical: GQA `dk/dv` use atomics, so accumulation order can differ from upstream's
post-kernel reduction. The validation suite checks the local fork against the eager Wall oracle on
small shapes and passed on the L4 before this report was written:

```text
pytest -q kernel/wall/test_wall_kernel.py
14 passed
```

Important tolerances in that suite:

| check | tolerance | notes |
| --- | ---: | --- |
| forward vs eager reference | `rtol=2e-2, atol=2e-2` | covers MHA, GQA, varlen, sink bias, scalar gate, sliding window |
| `q/k/v` gradients vs eager reference | `rtol=8e-2, atol=8e-2` | covers GQA reduction for `k.grad` and `v.grad` |
| `g` gradient finite differences | `rtol=0.22, atol=0.13` | central finite differences on a tiny fp32 problem |
| `g_scalar` gradient finite differences | `rtol=0.22, atol=0.13` | central finite differences on a tiny fp32 problem |

A direct fresh-process upstream-vs-local run was not recorded in this report because the current
environment is missing `cuda.h`, and Triton fails when rebuilding its CUDA helper module from an
empty cache. Earlier CUDA validation was run before that cache miss.

### Current implementation notes

- `dk` and `dv` are allocated as KV-head-shaped tensors, `(B,T,H,K)` and `(B,T,H,V)`, for GQA.
- The DKV backward kernel atomically accumulates per-query-head contributions into those KV-head
  outputs instead of writing `(B,T,HQ,K/V)` temporaries and reducing afterward.
- The key-side prefix gradient is accumulated into the existing `dg_cumsum` buffer, eliminating the
  separate `dg_cumsum_k` allocation.
- DKV backward bypasses Triton autotune because autotune candidates would repeatedly accumulate
  into the same output buffers.

### Integration

After the local kernel is validated, update `train/model.py` import order:

```python
try:
    from kernel.wall import wall_attn as _wall_attn_kernel
except Exception:
    from wall_attn import wall_attn as _wall_attn_kernel
```

Keep the external package as fallback until the in-tree kernel is trusted.

## Open Questions

- Does recomputing `g_cumsum` in backward save enough memory after removing full-HQ `dk/dv`, or is
  it unnecessary?
- Can the DKV backward directly reduce over GQA groups without atomics while preserving reasonable
  occupancy?
- Does `window_size=1024` change the best block sizes on L4?
- Are upstream autotune configs choosing a high-memory/low-speed shape on L4? Test fixed configs
  before changing algorithmic storage.
