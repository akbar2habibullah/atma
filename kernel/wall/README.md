# Wall Attention Kernel Strategy

This directory is reserved for an in-tree Wall Attention kernel fork. Keep it separate from
`kernel/polar_triton.py` so the Wall-specific math, validation, and performance work do not blur
the boundary with the existing Polar kernel.

## Context

The training model now prefers the in-tree kernel and keeps Tilde Research's package-level kernel
as fallback:

```python
try:
    from kernel.wall import wall_attn as _wall_attn_kernel
except Exception:
    from wall_attn import wall_attn as _wall_attn_kernel
```

`train/model.py` calls this path for `attn_type="wall"`. The upstream reference is at:

- https://github.com/tilde-research/wall-attention-release
- https://github.com/tilde-research/wall-attention-release/blob/main/wall_attn/reference.py
- https://github.com/tilde-research/wall-attention-release/blob/main/wall_attn/training.py

The reference implementation is only a correctness oracle. It materializes pairwise `T x T`
objects and should not be used to reason about training memory. The relevant implementation is
`wall_attn/training.py`.

## Investigation Status

The initial concern was that Wall Attention might be intrinsically exhausting much more memory than
the other attention variants. The current measurements do not support making that allegation yet.

The corrected interpretation is:

- The `14-15 GB` Wall result at `mbs=2` was measured in an eager synthetic full-model path. In the
  ablation-like `torch.compile` path, local Wall at the same high-overhead setting measured
  `11.530 GB` peak allocated and `11.953 GB` peak reserved.
- The `21.064 GB` `mbs=4` OOM row was reproduced for both upstream Wall and the local fork in the
  eager full-model path. It is not evidence that the profiler accidentally selected upstream only.
- The isolated local Wall backward intervention does reduce Wall kernel memory, but that saving is
  hidden by larger full-model allocations at the heavy ablation setting.
- In synthetic full-model breakdowns, the memory branch is the main source of the Wall peak increase;
  distractor alignment mainly increases runtime in the tested configuration.
- Before claiming a Wall algorithm problem, compare the compiled training path, the exact enabled
  features, and the attention kernel in isolation.

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

Measured on the local NVIDIA L4 24 GB box on 2026-06-26 and 2026-06-27. The report intentionally
separates three different questions:

- isolated attention/core memory: does Wall's fused op itself allocate much more memory?
- eager full-model memory: what happens without the `torch.compile` path used by ablation training?
- compiled full-model memory: what happens in the ablation-like path?

All GPU memory runs below were performed sequentially on an idle L4. Runs used:

```text
CPATH=/opt/conda/lib/python3.12/site-packages/tensorflow/include/external/cuda_cudart/include
FLA_CUSTOM_OP=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The synthetic full-model profiler uses random token batches and the same `train.model.Model` path.
It avoids dataset IO and does not prove dataloader or optimizer memory behavior. In this
environment, `causal-conv1d` was unavailable, so the repository's PyTorch fallback was used for the
memory branch.

### Isolated attention/core comparison

Command:

```text
python -m kernel.wall.profile_attention_variants --B 2 --T 2048 --HQ 8 --H 2 --K 128 --V 128 --dtype bf16 --window 1024
```

Shape:

```text
B=2, T=2048, HQ=8, H=2, G=4, K=128, V=128, dtype=bf16
```

Measured source files:

```text
local_wall_file=/home/sagemaker-user/atma/kernel/wall/training.py
upstream_wall_file=/home/sagemaker-user/wall-attention-release/wall_attn/training.py
```

| impl | window | peak allocated | peak reserved | elapsed |
| --- | ---: | ---: | ---: | ---: |
| SDPA nope | full causal | 0.137 GB | 0.139 GB | 104.18 ms |
| SDPA rope-scale | full causal | 0.137 GB | 0.139 GB | 5.39 ms |
| Polar Triton | full causal | 0.151 GB | 0.160 GB | 3679.46 ms |
| Wall local | full causal | 0.348 GB | 0.354 GB | 3124.16 ms |
| Wall upstream | full causal | 0.387 GB | 0.393 GB | 5567.32 ms |
| SDPA nope | 1024 | 0.149 GB | 0.158 GB | 38.98 ms |
| SDPA rope-scale | 1024 | 0.149 GB | 0.158 GB | 5.64 ms |
| Polar Triton | 1024 | 0.151 GB | 0.160 GB | 2069.15 ms |
| Wall local | 1024 | 0.129 GB | 0.139 GB | 13.21 ms |
| Wall upstream | 1024 | 0.141 GB | 0.158 GB | 12.96 ms |

Interpretation:

- Full-causal isolated Wall is heavier than SDPA and Polar at this shape, although the local fork is
  lower than upstream.
- Training-window Wall at `window=1024` is not the isolated memory outlier. Local Wall is slightly
  lower than upstream and comparable to or lower than the SDPA windowed rows.
- This table is attention/core-only. It does not explain the full-model `14-15 GB` or `21.064 GB`
  results by itself.

### Naive PyTorch reference

The small correctness reference is not a viable training baseline at the target shape because it
materializes pairwise `T x T` tensors.

Shape:

```text
B=2, T=2048, HQ=8, H=2, G=4, K=128, V=128, dtype=bf16, R=0
```

| impl | window | status |
| --- | ---: | --- |
| naive PyTorch reference | full causal | OOM; requested 32.00 GiB allocation |
| naive PyTorch reference | 1024 | OOM; requested 32.00 GiB allocation |

This only shows that the naive reference is unsuitable for this shape. It should not be used to
claim anything about the optimized upstream Wall kernel.

### Local Wall kernel intervention

Measured with the isolated Wall benchmark:

```text
out = wall_attn(q, k, v, g, scale=K**-0.5, window_size=window)
loss = out.float().square().mean()
loss.backward()
```

Shape:

```text
B=2, T=2048, HQ=8, H=2, G=4, K=128, V=128, dtype=bf16, R=0
```

| impl | window | peak allocated | peak reserved | elapsed / status |
| --- | ---: | ---: | ---: | ---: |
| upstream `wall_attn` | `None` | 0.305 GB | 0.307 GB | 7.04 ms |
| local low-memory | `None` | 0.121 GB | 0.135 GB | 16.55 ms |
| upstream `wall_attn` | 1024 | 0.305 GB | 0.307 GB | 6.16 ms |
| local low-memory | 1024 | 0.121 GB | 0.135 GB | 14.44 ms |

Result for the Wall kernel alone:

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

Direct upstream/local source comparisons were recorded in fresh processes after setting `CPATH` to
the CUDA headers above.

### Full-model memory: eager path

These checks use the heavy Wall ablation settings from
`wall__reg-strong__distr-1__mem-1__win-1`: 16 layers, hidden size 1024, head dim 128,
`seq_len=2048`, `num_random_keys=2048`, strong regularization, distractor alignment, training
window 1024, and memory enabled.

Command family:

```text
python -m kernel.wall.profile_full_model_memory --attn_type <variant> --mbs <N> \
  --seq_len 2048 --layers 16 --reg_mode strong --distractor --memory --window
```

Eager synthetic full-model rows:

| attention | mbs | status | peak allocated | peak reserved | notes |
| --- | ---: | --- | ---: | ---: | --- |
| nope | 2 | OK | 14.008 GB | 14.486 GB | full attention |
| rope | 2 | OK | 13.973 GB | 14.439 GB | full attention |
| polar | 2 | OK | 15.354 GB | 15.836 GB | slowest non-Wall row here |
| wall local | 2 | OK | 13.726 GB | 14.193 GB | `/home/sagemaker-user/atma/kernel/wall/training.py` |
| wall upstream | 2 | OK | 13.726 GB | 14.193 GB | `/home/sagemaker-user/wall-attention-release/wall_attn/training.py` |
| nope | 4 | OOM | 21.399 GB | 21.574 GB | failed on a 1.54 GiB allocation request |
| rope | 4 | OOM | 21.317 GB | 21.469 GB | failed on a 1.54 GiB allocation request |
| polar | 4 | OOM | 21.114 GB | 21.285 GB | failed on a 1.54 GiB allocation request |
| wall local | 4 | OOM | 20.913 GB | 21.064 GB | failed on a 1.54 GiB allocation request |
| wall upstream | 4 | OOM | 20.913 GB | 21.064 GB | same as local in this path |

Interpretation:

- The `21.064 GB` `mbs=4` Wall number is an eager full-model OOM number and was reproduced for both
  local and upstream Wall. It is not a measurement of only upstream Wall.
- In this eager synthetic path, the high-overhead `mbs=4` configuration OOMs for all tested
  attention variants, not only Wall.
- In this eager synthetic path, Wall at `mbs=2` is not above nope/rope and is below Polar. This
  contradicts the initial premise that Wall alone explains a `14-15 GB` `mbs=2` footprint.

### Full-model memory: compiled path

The ablation training script uses `torch.compile`, so the compiled profiler is the closer survival
test:

```text
python -m kernel.wall.profile_full_model_memory --compile --attn_type <variant> --mbs <N> \
  --seq_len 2048 --layers 16 --reg_mode strong --distractor --memory --window
```

Compiled synthetic full-model rows:

| attention | mbs | status | peak allocated | peak reserved | notes |
| --- | ---: | --- | ---: | ---: | --- |
| nope | 2 | OK | 11.906 GB | 12.324 GB | finite losses |
| rope | 2 | OK | 11.828 GB | 12.246 GB | finite losses |
| polar | 2 | OK | 13.221 GB | 13.637 GB | allocation row only; losses were `nan` in this synthetic run |
| wall local | 2 | OK | 11.530 GB | 11.953 GB | finite losses; local source confirmed |
| wall local | 4 | OOM | 19.577 GB | 20.398 GB | failed on a 1.54 GiB allocation request |

Attempted compiled upstream Wall at `mbs=2` did not finish after several minutes and was terminated
to keep later measurements uncontaminated. Do not use the eager upstream row as a substitute for a
compiled upstream result.

Interpretation:

- The compiled local Wall `mbs=2` result is in the same 11-12 GB reserved band as nope and rope.
- The earlier `14-15 GB` `mbs=2` concern is therefore best treated as an eager/profiling-path
  artifact until a real ablation training run proves otherwise.
- The compiled local Wall `mbs=4` row still OOMs under the highest-overhead synthetic setting, but
  it OOMs below the old eager `21.064 GB` reserved number.

### Full-model Wall feature breakdown

These rows keep `attn_type=wall`, `wall_impl=local`, `mbs=2`, `layers=16`, `seq_len=2048`,
`reg_mode=strong`, and `window=1024`, then toggle distractor alignment and memory.

| distractor | memory | peak allocated | peak reserved | forward | backward | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| off | off | 12.156 GB | 12.602 GB | 2470.63 ms | 1720.94 ms | base Wall model path |
| on | off | 12.156 GB | 12.602 GB | 4621.84 ms | 3798.94 ms | runtime increase, no peak increase observed |
| off | on | 13.726 GB | 14.193 GB | 2765.10 ms | 1855.57 ms | memory branch adds about 1.57 GB allocated |
| on | on | 13.726 GB | 14.193 GB | 4883.66 ms | 3950.24 ms | memory branch dominates peak; distractor dominates extra time |

This is the clearest current breakdown of where the full-model peak comes from. At this setting,
Wall attention's isolated kernel saving is smaller than the model-level memory branch and activation
footprint.

### Ablation survival sanity

Training survival on NVIDIA L4, with the Wall kernel source forced explicitly:

```text
local:    /home/sagemaker-user/atma/kernel/wall/training.py
upstream: /home/sagemaker-user/wall-attention-release/wall_attn/training.py
```

| impl | mbs | status | peak allocated | peak reserved | notes |
| --- | ---: | --- | ---: | ---: | --- |
| local low-memory, compiled | 2 | OK | 11.530 GB | 11.953 GB | closest synthetic match to ablation training |
| local low-memory, compiled | 4 | OOM | 19.577 GB | 20.398 GB | highest-overhead synthetic setting still exceeds L4 |
| local low-memory, eager | 4 | OOM | 20.913 GB | 21.064 GB | old-style number, not compiled |
| upstream `wall_attn`, eager | 4 | OOM | 20.913 GB | 21.064 GB | same eager failure profile as local |

The `mbs=4` eager OOM is therefore not an accidental upstream-kernel measurement. It also is not
fixed by the local DKV-backward storage change because this full-model peak is dominated by tensors
outside that isolated allocation. In the local eager `mbs=2` split measurement, peak memory was
already 12.198 GB after forward and rose to 13.726 GB after backward.

The `mbs=2` run completed one full forward/backward optimizer-step equivalent over the heavy loss
composition:

```text
loss = (1 - 0.01) * lm_loss + 0.01 * strong_reg_loss + 0.01 * distractor_align_loss
```

The first attempt exposed a checkpointing integration bug: PyTorch's non-reentrant checkpoint path
raises a private `_StopRecomputationError` as internal control flow, and the Wall wrapper's broad
exception handler was converting it into `RuntimeError("wall_attn Triton kernel failed during
training")`. `train/model.py` now re-raises that control-flow exception unchanged.

### Long-context 64K forward sanity

Full-context 64K Wall forward sanity compared the explicit upstream package and local fork sources:

```text
local:    /home/sagemaker-user/atma/kernel/wall/training.py
upstream: /home/sagemaker-user/wall-attention-release/wall_attn/training.py
```

Shape: `B=1, T=65536, HQ=8, H=2, K=V=128, dtype=bf16, window=None`.

| impl | status | peak allocated | peak reserved | elapsed |
| --- | --- | ---: | ---: | ---: |
| local low-memory | OK | 1.062 GB | 1.191 GB | 91.880 s |
| upstream `wall_attn` | OK | 1.062 GB | 1.191 GB | 167.802 s |

Forward memory is identical because the local fork intentionally leaves the upstream forward kernel
unchanged; the low-memory changes are in backward. Fresh-process elapsed times include compile and
autotune effects and should not be read as a stable throughput comparison.

### Current conclusion

Do not claim yet that Wall Attention's approach inherently more than doubles memory versus the
other attention variants. The current evidence says:

- local Wall's isolated backward intervention is real and source-confirmed;
- eager full-model `mbs=4` OOM at `21.064 GB` is shared by upstream and local Wall, and all tested
  eager high-overhead variants OOM near the L4 limit;
- compiled local Wall at `mbs=2` survives in the 11-12 GB band;
- the full-model Wall peak at `mbs=2` is mostly explained by model/pipeline allocations, especially
  the memory branch, not by the isolated Wall kernel alone;
- a compiled upstream Wall full-model row is still missing because that run did not terminate within
  a useful profiling window on this machine.

Evidence still needed before making a stronger statement:

- a real ablation training step with optimizer enabled and the same data path;
- compiled full-model upstream Wall if it can be made to finish reproducibly;
- a memory snapshot or allocator trace that assigns live bytes to attention, memory branch,
  vocabulary loss, regularization, and optimizer state;
- repeated compiled `mbs=4` rows for nope/rope/polar if the claim depends on those variants
  surviving the exact highest-overhead config.

### Current implementation notes

- `dk` and `dv` are allocated as KV-head-shaped tensors, `(B,T,H,K)` and `(B,T,H,V)`, for GQA.
- The DKV backward kernel atomically accumulates per-query-head contributions into those KV-head
  outputs instead of writing `(B,T,HQ,K/V)` temporaries and reducing afterward.
- The key-side prefix gradient is accumulated into the existing `dg_cumsum` buffer, eliminating the
  separate `dg_cumsum_k` allocation.
- DKV backward bypasses Triton autotune because autotune candidates would repeatedly accumulate
  into the same output buffers.

### Integration

The training model prefers the in-tree kernel and keeps the external package as fallback:

```python
try:
    from kernel.wall import wall_attn as _wall_attn_kernel
except Exception:
    from wall_attn import wall_attn as _wall_attn_kernel
```

## Open Questions

- Does recomputing `g_cumsum` in backward save enough memory after removing full-HQ `dk/dv`, or is
  it unnecessary?
- Can the DKV backward directly reduce over GQA groups without atomics while preserving reasonable
  occupancy?
- Does `window_size=1024` change the best block sizes on L4?
- Are upstream autotune configs choosing a high-memory/low-speed shape on L4? Test fixed configs
  before changing algorithmic storage.
