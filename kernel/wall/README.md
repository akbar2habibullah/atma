# Wall Attention Kernel Report

This directory contains the in-tree Wall Attention training kernel fork used by Atma when
`attn_type="wall"`. The goal is to keep the upstream Wall Attention math while reducing GQA
training memory on the NVIDIA L4 ablation setup.

## Current Conclusion

The real ablation worker path shows a Wall-specific memory problem at the highest-overhead
configuration, but the problem is not explained by the isolated Wall kernel allocation.

There is now also a separate training-stability concern. After the model integration was changed
from raw signed gates to Tilde-style bounded log-decays (`logsigmoid(W_g x + b)` plus soft clamp,
default `b=6`), an upstream-kernel run was reported on 2026-06-30 to survive past the previous
early-NaN point but hover between about `5.8` and `7.0` loss after `>500` optimizer steps, including
a regression from `5.8` back toward `7.0`. This report is not yet a controlled ablation result, but
it is strong enough that Wall pretraining in this codebase should be treated as unstable until the
mechanism is isolated.

Responsible numbers from the actual worker path:

| attention | mbs | status | peak GPU process memory | notes |
| --- | ---: | --- | ---: | --- |
| `nope` | 4 | train step completed | about `11304 MiB` training plateau | long eval was interrupted |
| `rope` | 4 | completed | `11346-11350 MiB` | repeated with memory phase marks |
| `polar` | 4 | completed | about `14380 MiB` training plateau | full smoke peak `15732 MiB` with eval |
| Wall local fork | 4 | completed | `21770-21776 MiB` | warm-cache step `84.9 s`; cold first step `557.6 s` |
| Wall upstream | 4 | completed | `22480 MiB` | cold first step `956.9 s` |

Interpretation:

- `nope`, `rope`, and `polar` do not hit OOM at `mbs=4` under strong regularization, distractor,
  memory, window, and compile-on worker settings. They stay in the expected sub-15 GB training band.
- Wall `mbs=4` also survives on the L4, but it peaks in the `21-22 GB` class.
- The local fork saves about `704-710 MiB` versus upstream in the full worker graph, but that is far
  smaller than the `~10.4 GiB` process-memory gap between Wall local and `rope`.
- The high Wall memory is a compiled forward/backward graph-liveness issue around the Wall
  integration, not optimizer state, evaluation, or a single giant Wall kernel tensor.

## Active Kernel Selection

`train/model.py` defaults to the in-tree kernel and falls back to the installed upstream package:

```python
try:
    from kernel.wall import wall_attn as _wall_attn_kernel
except Exception:
    from wall_attn import wall_attn as _wall_attn_kernel
```

For profiling, force the source explicitly:

```text
ATMA_WALL_IMPL=auto      # default: local first, upstream fallback
ATMA_WALL_IMPL=local     # require in-tree kernel.wall
ATMA_WALL_IMPL=upstream  # require installed wall_attn package
```

The upstream package used for comparisons is editable from:

```text
/home/sagemaker-user/wall-attention-release
```

The model-level Wall input is a bounded natural-log decay, not a raw gate:

```text
logits = W_g x + wall_gate_bias
g_hat  = logsigmoid(logits)
g      = -g_max * (1 - exp(g_hat / g_max))   # g in [-g_max, 0], g_max=0.87
```

This matters because the kernels assume `g <= 0` when bounding the per-tile rescaling factors.
Passing raw signed `W_g x` can invalidate those assumptions even though the exact Wall score is
bounded when the retention gates are in `(0, 1]`.

## Environment

Worker measurements in this report were taken on 2026-06-27 on an NVIDIA L4 with:

```text
torch==2.12.1+cu130
flash-linear-attention==0.5.1
fla-core==0.5.1
transformers==4.57.6
kernels==0.11.7
wall-attn==0.1.0 editable from /home/sagemaker-user/wall-attention-release
FLA_CUSTOM_OP=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`causal-conv1d` could not be installed on this host because its source build detected CUDA 12.9
while PyTorch was compiled for CUDA 13.0. The worker loaded the Hugging Face `kernels`
causal-conv path instead. Earlier local `atma/fla` shadowing was removed; real FLA from
site-packages is used.

## Valid Evidence

All current full-model memory conclusions come from:

```text
FLA_CUSTOM_OP=1 python -m ablation.run_worker --config_dir <shard> --log_dir <logs> --gpu 0 --once
```

The smoke configs preserve the model and high-overhead settings:

```text
seq_len=2048
mbs=4
num_hidden_layers=16
hidden_size=1024
head_dim=128
reg_mode=strong
distractor=true
memory=true
window=true
attn_window=1024
num_random_keys=2048
batch_size=524288
```

Only runtime-bounding fields were reduced:

```text
max_steps=1
num_chunks=1
val_tokens=8192
eval_lengths=[2048]
needle_distances=[2048]
num_eval_docs=1
num_needle_trials=1
```

Memory was sampled externally with `nvidia-smi` once per second. For attribution runs,
`torch.cuda` allocator marks were also logged inside the worker.

## Invalidated Evidence

Do not use `kernel.wall.profile_full_model_memory` for `mbs=4` conclusions.

That synthetic profiler reported 20+ GB or OOM for non-Wall variants that the actual worker runs at
about 11-14.4 GB. It was also used before the dependency environment was corrected. Its rows are
useful only as evidence that the profiler path was mismatched; they are not evidence about real
ablation memory.

Specifically, stale claims such as the synthetic `21.064 GB` `mbs=4` Wall OOM row, or synthetic
compiled `mbs=4` OOM rows for `nope`/`polar`, are superseded by worker results above.

## Allocation Attribution

The local Wall `mbs=4` worker was rerun after the first compile with the same inductor cache:

```text
TORCHINDUCTOR_CACHE_DIR=/tmp/atma_inductor_worker_wall_mbs4
ATMA_WALL_IMPL=local
```

It still peaked at `21770 MiB`, while step time dropped from `557.6 s` to `84.9 s`. The high memory
is therefore not a cold-compile artifact.

An opt-in worker profiler was added:

```text
ATMA_MEM_PROFILE=1
ATMA_MEM_TRACE=/tmp/atma_wall_investigation/wall_mbs4_profile_local_snapshot.pkl
ATMA_MEM_TRACE_MAX_ENTRIES=300000
```

With `batch_size=524288`, `seq_len=2048`, and `mbs=4`, one train step has 64 microbatches.

Matched phase marks for `rope` and warm-cache local Wall:

| phase / metric | rope mbs=4 | Wall local mbs=4 | Wall - rope |
| --- | ---: | ---: | ---: |
| external `nvidia-smi` process peak | `11346 MiB` | `21770 MiB` | `+10424 MiB` |
| `torch.cuda` max allocated | `10938.9 MiB` | `20457.6 MiB` | `+9518.7 MiB` |
| `torch.cuda` max reserved | `11016.0 MiB` | `21270.0 MiB` | `+10254.0 MiB` |
| steady live before micro forward | `2711.0 MiB` | `2743.2 MiB` | `+32.2 MiB` |
| live after micro forward | `10922.9 MiB` | `15741.6 MiB` | `+4818.7 MiB` |
| backward transient over post-forward live | `16.0 MiB` | `4716.0 MiB` | `+4700.0 MiB` |
| live after micro backward | `2711.0 MiB` | `2743.2 MiB` | `+32.2 MiB` |
| live after optimizer step | `4351.3 MiB` | `4399.6 MiB` | `+48.3 MiB` |

This accounts for the 20+ GB result:

- Wall adds about `9.52 GiB` allocator peak over `rope`.
- About `4.82 GiB` is live immediately after forward.
- About `4.70 GiB` more is a backward transient.
- Optimizer state is not the source.
- Post-backward live state is not the source.

Allocator trace summary for the local Wall attribution run:

```text
snapshot=/tmp/atma_wall_investigation/wall_mbs4_profile_local_snapshot.pkl
events=300000  # hit configured cap; reconstructed peak is a lower bound
peak_live=17714.4 MiB

live breakdown at sampled peak:
  train_model              9790.4 MiB
  unattributed_empty       7860.0 MiB
  wall_local_kernel          64.0 MiB
```

The trace is capped, so `17714.4 MiB` is lower than the actual `20457.6 MiB` allocator peak. It is
still useful: direct live allocations from `kernel/wall/training.py` are small at the sampled peak;
the large live buckets are the compiled `train/model.py` graph and autograd temporaries.

## Why The Isolated Kernel Does Not Explain 20+ GB

At the worker shape:

```text
B=4, T=2048, HQ=8, H=2, K=V=128
```

Raw tensor sizes are small:

| tensor | size |
| --- | ---: |
| full query-head bf16 `(B,T,HQ,K)` | `16 MiB` |
| full query-head fp32 `(B,T,HQ,K)` | `32 MiB` |
| KV-head bf16 `(B,T,H,K)` | `4 MiB` |
| KV-head fp32 `(B,T,H,K)` | `8 MiB` |
| upstream GQA `dk` or `dv` temporary `(B,T,HQ,K)` fp32 | `32 MiB` each |
| local GQA `dk` or `dv` output `(B,T,H,K)` fp32 | `8 MiB` each |
| distractor full query-head bf16 at `R+T=4096` | `32 MiB` |

The isolated local Wall kernel at the original benchmark shape is also small:

| shape | peak allocated | peak reserved |
| --- | ---: | ---: |
| `B=2,T=2048,HQ=8,H=2,K=V=128,window=1024` | `0.121 GB` | `0.135 GB` |

Even the distractor-length isolated local Wall kernel is too small to explain the model-level jump:

```text
python -m kernel.wall.bench_wall --impl local --B 4 --T 4096 --HQ 8 --H 2 --K 128 --V 128 --dtype bf16 --window 0 --warmup 1 --iters 1
impl=local,B=4,T=4096,...,window=None,status=ok,peak_alloc_gb=0.485,peak_reserved_gb=0.490,elapsed_ms=71.94
```

The remaining suspect is graph liveness around the Wall integration:

- non-reentrant checkpointed Wall forward;
- the second distractor Wall call over `[random prefix R] + [real T]`;
- MSE alignment between `y_dist` and `y`;
- backward recomputation of the Wall path;
- compile/autograd liveness keeping more of the graph live than the non-Wall SDPA/Flash path.

## Local Kernel Intervention

The local fork keeps upstream forward and query-gradient paths, but changes DKV backward storage:

- upstream writes per-query-head `dk/dv` as `(B,T,HQ,K/V)` and reduces afterward;
- local writes directly into KV-head-shaped `(B,T,H,K/V)` outputs with atomic accumulation;
- upstream writes key-side `dP` to a separate `dg_cumsum_k`;
- local accumulates key-side `dP` into the existing `dg_cumsum`.

Isolated Wall benchmark at:

```text
B=2, T=2048, HQ=8, H=2, G=4, K=V=128, dtype=bf16, R=0
```

| impl | window | peak allocated | peak reserved | elapsed |
| --- | ---: | ---: | ---: | ---: |
| upstream `wall_attn` | `None` | `0.305 GB` | `0.307 GB` | `7.04 ms` |
| local low-memory | `None` | `0.121 GB` | `0.135 GB` | `16.55 ms` |
| upstream `wall_attn` | `1024` | `0.305 GB` | `0.307 GB` | `6.16 ms` |
| local low-memory | `1024` | `0.121 GB` | `0.135 GB` | `14.44 ms` |

This confirms the local backward memory intervention works in isolation. It does not solve the
full-model compiled graph liveness problem.

## Numerical Parity

The local fork is not bitwise identical to upstream because GQA `dk/dv` use atomics, so accumulation
order can differ. It is tested against the eager Wall oracle within floating-point tolerance.

Latest verification:

```text
pytest -q kernel/wall/test_wall_kernel.py
14 passed, 14 warnings
```

Coverage includes:

| check | tolerance | notes |
| --- | ---: | --- |
| forward vs eager reference | `rtol=2e-2, atol=2e-2` | MHA, GQA, varlen, sink bias, scalar gate, sliding window |
| `q/k/v` gradients vs eager reference | `rtol=8e-2, atol=8e-2` | includes GQA `k.grad` and `v.grad` |
| `g` gradient finite differences | `rtol=0.22, atol=0.13` | tiny fp32 problem |
| `g_scalar` gradient finite differences | `rtol=0.22, atol=0.13` | tiny fp32 problem |

Later note: after the bounded-gate integration fix, a rerun of the direct local-kernel test suite
on 2026-06-30 failed several random signed-gate parity cases before being interrupted. Those tests
feed `g ~ N(0, 0.05)`, which is outside the production log-decay contract because it includes
positive values. The relevant next parity target is bounded log-decay input (`g in [-0.87, 0]`,
with realistic open-gate bias), for both local and upstream kernels.

## Training Stability Investigation

The exact Wall score

```text
s_ij = sum_n q_i,n k_j,n prod_{r=j+1..i} retention_r,n
```

does not make logits larger than vanilla attention when each retention is in `(0, 1]`; it only
attenuates channel contributions. The instability risk is in the factorized/tiled implementation
and optimizer dynamics:

- `P_t = cumsum(log retention_t)` drifts negative with position. The kernel avoids naive
  `exp(P) q` / `exp(-P) k` overflow by anchoring tiles, but backward still contains large local
  rescale factors and explicit clamps.
- The gate gradient is a reverse cumsum over prefix sensitivities, so `W_g` receives long-horizon
  credit assignment with potentially high variance and step-size sensitivity.
- Muon orthogonalizes matrix updates. That may be too aggressive for `w_wall`, whose initialized
  gate derivative is small at bias 6 but whose prefix effect is global once logits move.
- Surrounding objectives can amplify the issue: distractor MSE adds a second Wall call, the MAG
  memory branch changes residual statistics, and sliding-window/full-window differences alter the
  prefix ranges seen by the kernel.

Minimum diagnostic run before more Wall ablations:

```text
ATMA_WALL_IMPL=upstream ATMA_WALL_CUSTOM_OP=1 FLA_CUSTOM_OP=1 ...
```

For each validation interval or every 25-50 train steps, record:

| signal | why |
| --- | --- |
| train loss and val loss | distinguish noisy minibatches from true divergence |
| pre-clip total grad norm and `w_wall.grad` norms | detects gate-gradient spikes hidden by clipping |
| `w_wall.weight` norm per Wall layer | detects Muon-driven gate drift |
| retention quantiles `exp(g)` | shows whether channels are near-open, shut, or bimodal |
| prefix range `max(P)-min(P)` per layer/head | measures numerical stress on tiled rescaling |
| output RMS before/after Wall and after projection | detects residual-scale drift |

Falsification matrix:

| Variant | Interpretation if stable |
| --- | --- |
| `w_wall` in AdamW at 5-10x lower LR, excluded from Muon | optimizer/update geometry is the culprit |
| `wall_gate_bias=8` | open-gate operating point was too easy to leave |
| no distractor, memory/window unchanged | second Wall call or MSE alignment destabilizes gradients |
| no memory, distractor/window unchanged | residual interaction with MAG destabilizes training |
| local vs upstream with bounded-gate parity checked first | kernel backward difference matters |

## Long-Context Forward Sanity

Forward-only 64K sanity uses:

```text
B=1, T=65536, HQ=8, H=2, K=V=128, dtype=bf16, window=None
```

| impl | status | peak allocated | peak reserved | elapsed |
| --- | --- | ---: | ---: | ---: |
| local low-memory | OK | `1.062 GB` | `1.191 GB` | `91.880 s` |
| upstream `wall_attn` | OK | `1.062 GB` | `1.191 GB` | `167.802 s` |

Forward memory is identical because the local fork intentionally leaves upstream forward unchanged.

## Reproduction Commands

Run a worker smoke config:

```text
CPATH=/opt/conda/lib/python3.12/site-packages/tensorflow/include/external/cuda_cudart/include \
FLA_CUSTOM_OP=1 \
ATMA_WALL_CUSTOM_OP=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCHINDUCTOR_CACHE_DIR=/tmp/atma_inductor_worker_wall_mbs4 \
ATMA_WALL_IMPL=local \
python -m ablation.run_worker --config_dir /tmp/atma_wall_investigation/shard_wall_mbs4_profile_local --log_dir /tmp/atma_wall_investigation/logs_wall_mbs4_profile_local --gpu 0 --once
```

Enable phase marks and allocator trace:

```text
ATMA_MEM_PROFILE=1
ATMA_MEM_TRACE=/tmp/atma_wall_investigation/wall_mbs4_profile_local_snapshot.pkl
ATMA_MEM_TRACE_MAX_ENTRIES=300000
```

Summarize a trace:

```text
python -m kernel.wall.summarize_memory_trace /tmp/atma_wall_investigation/wall_mbs4_profile_local_snapshot.pkl --top 40
```

Run isolated Wall benchmark:

```text
python -m kernel.wall.bench_wall --impl local --B 4 --T 4096 --HQ 8 --H 2 --K 128 --V 128 --dtype bf16 --window 0 --warmup 1 --iters 1
```

Run parity tests:

```text
pytest -q kernel/wall/test_wall_kernel.py
```

## Next Work

The next rigorous step is a full worker-level permutation sweep, not another synthetic profiler:

```text
attention type: nope, rope, polar, wall-local, wall-upstream
memory: off/on
window: off/on
distractor: off/on
regularization: baseline/strong
mbs: 4
seq_len: 2048
```

The highest-priority Wall-specific isolation tests are:

- Wall local with distractor disabled, memory/window/strong unchanged;
- Wall local with `ATMA_WALL_CUSTOM_OP=1` vs `0` (opaque compile boundary vs raw Wall autograd path);
- Wall local with checkpoint disabled for the Wall attention call only;
- Wall local with the second distractor Wall call disabled but `align_loss=0` retained;
- Wall local with `torch.compile` disabled only as a diagnostic, not as a final ablation number;
- allocator trace with a higher `ATMA_MEM_TRACE_MAX_ENTRIES` if the current cap truncates too much.

Do not make a stronger claim about Tilde Research's algorithm until the full worker sweep assigns
the excess to the Wall algorithm itself rather than Atma's integration, checkpointing, or compile
graph liveness.
