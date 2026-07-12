# L40S Kernel-Efficiency Implementation Plan

Execution handoff for evaluating three related optimization directions on the repository's
NVIDIA L40S target:

- [PackInfer](https://arxiv.org/abs/2602.06072): cross-request attention-work packing for
  heterogeneous serving batches.
- [CODA](https://arxiv.org/abs/2605.19269): Transformer computation expressed as
  GEMM-plus-epilogue programs.
- [PyTorch normalization fusion](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/):
  Lazy Pre-Norm, Multi-CTA normalization, and attention/norm fusion.

This is an implementation and measurement plan, not a commitment to port any system wholesale.
For ATMA, execute in this order:

1. fuse small, high-confidence normalization/GEMM patterns;
2. add PackInfer-inspired grouped heterogeneous prefill;
3. consider broader CODA-style rewriting only if profiling still shows material non-attention
   HBM or dispatch overhead.

---

## 1. Relevant ATMA context

### Model and normalization sites

The canonical model is a 16-layer, hidden-size-1024 hybrid with 8 query heads, 2 KV heads,
head dimension 128, and a 3:1 convolution-to-attention ratio. Only four layers run Polar
Attention. The attention window defaults to 1024 tokens, and Titans MAG is enabled.

Each decoder block is pre-norm:

```text
x = x + sublayer(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

Relevant code:

- RMSNorm and MLP definitions: [`model/layers.py`](../model/layers.py)
- Inference blocks and attention: [`inference/models/atma.py`](../inference/models/atma.py)
- Training blocks and custom FP8 linears: [`train/model.py`](../train/model.py)
- Polar prefill/decode kernels: [`kernel/polar_triton.py`](../kernel/polar_triton.py)
- Existing inference fusions: [`kernel/inference_ops_triton.py`](../kernel/inference_ops_triton.py)

Normalization sites are:

- two learned full-width RMSNorms in every block;
- unweighted per-head Q/K RMSNorm after projection and before Canon convolution;
- unweighted per-head Titans readout RMSNorm before its gate and projection;
- final learned RMSNorm before the LM head.

### Existing serving optimizations

The inference engine already has:

- paged KV cache with direct paged reads in Polar decode;
- packed token preparation through cumulative sequence lengths;
- a dense fast path for fresh, complete, equal-length prompts;
- chunked prefill and centralized per-sequence convolution/Titans state tables;
- fixed-shape, CUDA-graph-captured decode;
- tensor-parallel linears;
- fused decode convolution updates, squared-ReLU activation, and output softcap.

The variable-length prefill path packs tokens but still iterates over sequences inside
`AtmaAttention.forward` for Canon convolution, Polar Attention, and Titans prefill. This is the
main PackInfer-relevant opportunity. Token packing removes padding from representation; work
packing assigns tiles from different requests to balanced kernel launches.

### Recorded L40S baseline

The latest baseline in [`docs/inference.md`](inference.md) is:

| Phase | Representative shape | Measured | Diagnosed limiter |
|---|---:|---:|---|
| Dense prefill | `B=8, T=512` | 174.6k tok/s | compute/dispatch |
| Decode | `B=512, S=512` | 64.6k tok/s | HBM/state |

Dense equal-length prefill is 4.85x faster than the packed fallback at `B=8, T=512`, mainly
because it reduces 192 convolution launches, 32 Polar launches, and 32 FLA scans to 24, 4,
and 4. Decode reaches about 76% of measured attainable L40S HBM bandwidth; Titans state and
paged-Polar kernels dominate more than standalone norms.

Therefore:

- grouped heterogeneous prefill can have a large workload-dependent benefit;
- norm/GEMM fusion should provide a smaller, broadly useful benefit;
- decode work must not add KV/state traffic merely to save launches;
- an isolated microkernel win is not enough without an end-to-end improvement.

---

## 2. Scope and non-goals

In scope:

- inference-first BF16 kernel prototypes at actual ATMA shapes;
- grouping variable-length prefill without changing model semantics;
- conservative shape dispatch and an eager/cuBLAS fallback;
- training work only after profiling demonstrates a material target.

Not in the first pass:

- porting PackInfer's complete CUDA/C++ system;
- compacting paged KV into a second contiguous buffer;
- changing cache allocation or block management;
- optimizing shared-prefix I/O before convolution and Titans prefix states are correct;
- porting CODA's CuTeDSL abstraction or reparameterized backward wholesale;
- implementing B200-style Multi-CTA normalization with distributed shared memory;
- architectural or numerical changes to Polar, Titans, QK norm, or RMSNorm.

---

## 3. Capture the baseline first

Before editing a kernel, record:

```text
git SHA and dirty status
GPU, compute capability, driver, CUDA, PyTorch, and Triton versions
checkpoint or random-weight status
dtype, tensor-parallel size, and model configuration
CUDA graph setting
warmup and measured iteration counts
```

Minimum commands:

```bash
nvidia-smi
python -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__); print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability())"
python -m scripts.roofline_inference --measure --prefill-tok-s 174571 --decode-tok-s 64583
python -m scripts.bench_inference
```

Capture Nsight Systems or `torch.profiler` traces for:

1. dense prefill at `B=8, T=512`;
2. heterogeneous fresh prefill with lengths such as
   `[64, 96, 128, 256, 512, 768, 1024, 1536]`;
3. decode at `B in {1, 32, 512}`, context 512;
4. one training step in BF16 and, if production-relevant, FP8.

Add a deterministic heterogeneous-prefill benchmark before evaluating PackInfer ideas. Report
p50/p95 time, tok/s, launch count, GPU-active time, important kernel times, and peak memory.
Keep prefill and decode separate.

---

## 4. Workstream A: normalization and GEMM fusion

### A1. Q/K projection plus per-head RMSNorm

Current attention pattern:

```text
q_gate = linear_q(x) -> split into q and gate
k      = linear_k(x)
q      = RMSNorm(q, normalized_shape=128, affine=None)
k      = RMSNorm(k, normalized_shape=128, affine=None)
```

Start with K because its output is smaller and has no gate split. The fused epilogue must:

1. preserve current GEMM/bias behavior;
2. reduce each 128-element head independently in FP32;
3. apply the current RMSNorm epsilon exactly;
4. store BF16 in the existing layout;
5. preserve tensor-parallel sharding.

For Q, normalize only Q and store the gate half unchanged. Do not change Q/gate layout unless all
consumers change together and parity is demonstrated.

Prefer a narrowly shaped Triton prototype; retain cuBLAS plus standalone norm as fallback.
Replacing cuBLAS can regress the mainloop, so measure separately:

```text
current GEMM only
current GEMM + standalone RMSNorm
fused GEMM + RMSNorm
full attention layer
```

Test token-row counts `M in {1, 32, 256, 4096, 8192}`. Use shape dispatch where the fused route
wins; protect latency-sensitive small-batch decode.

### A2. Lazy Pre-Norm for `norm2 -> MLP.fc`

For affine-free RMSNorm:

```text
RMSNorm(A) @ W = (A @ W) * rstd(A)
```

ATMA block norms have learned `gamma`. At inference, static gamma can be folded into a derived
projection weight with the correct repository weight orientation:

```text
(A * rstd(A) * gamma) @ W = (A @ (diag(gamma) @ W)) * rstd(A)
```

Do not mutate checkpoint tensors in place. Build derived inference weights after loading and per
TP shard, or fuse gamma explicitly.

Start with `norm2 -> MLP.fc` because there is one consumer. Accumulate `sum(A^2)` alongside the
GEMM K loop, compute row `rstd`, scale the accumulator, then optionally absorb the existing
split/squared-ReLU gate. Evaluate independently:

- norm + MLP input GEMM;
- norm + MLP input GEMM + split/squared-ReLU;
- output projection/residual only as a later experiment.

Do not initially carry this into training. Learned-gamma gradients, custom FP8 autograd,
checkpointing, and backward equivalence make training a separate task.

### Deferred fusion targets

- Titans readout norm: the following per-feature gate prevents its per-head scale from simply
  commuting through the cross-head projection.
- Final norm + LM head: tensor parallelism and the 50,304-wide output make it a later target.
- Full FlashNormAttention: Canon convolution, Polar statistics, output gate, and Titans branch
  differ substantially from the published pattern.

### Workstream A gate

Ship a fused route only if it:

- passes numerical checks at established tolerances;
- improves its fused operator sequence by at least 10% at an important shape;
- improves full prefill or decode by at least 3%, or is required by a measured later fusion;
- does not regress `B=1` decode by over 2% (otherwise use shape dispatch);
- remains CUDA-graph capturable and retains fallback.

If neither A1 nor A2 improves end-to-end performance, stop. Do not proceed to a broad CODA port
based only on paper results.

---

## 5. Workstream B: grouped heterogeneous prefill

### B1. Benchmark and cost model

Add deterministic fresh-prompt distributions:

| Name | Lengths |
|---|---|
| Short-heavy | `32, 48, 64, 64, 96, 128, 128, 256` |
| Mixed | `64, 96, 128, 256, 512, 768, 1024, 1536` |
| Long-tail | `64, 64, 128, 256, 512, 1024, 2048, 4096` |
| Homogeneous control | eight sequences of 512 |

Estimate effective Polar query/key tile work, not only tokens. Account for causal versus chunked
geometry, the 1024-token window, GQA grouping, Polar direction/magnitude reductions, fresh chunk
length, and cached prefix length.

### B2. Remove per-sequence Polar launches

Implement grouped variable-length Polar prefill over packed Q/K/V with cumulative lengths or a
tile-to-sequence map. Schedule independent query/key tiles across requests so long requests do not
strand SMs after short requests finish.

PackInfer's merge is for online softmax. ATMA must use Polar's own exact sufficient statistics and
merge equations. Match null sink, direction normalization, magnitude/participation statistic,
causality, windowing, and GQA.

Milestones:

1. fresh prompts, no prefix and default window;
2. mixed fresh lengths in one launch;
3. window variants;
4. chunked continuation with paged-prefix reads;
5. prefix-cache hits only after convolution/Titans prefix-state correctness is fixed.

Retain the dense route for equal fresh prompts and the per-sequence route as oracle/fallback.

### B3. Group Canon and Titans separately

After grouped Polar is correct and profiled:

- extend dense causal convolution to packed variable lengths or add boundary-aware grouped work;
- investigate an FLA variable-length/chunk call that emits each final Titans state into its
  existing sequence slot;
- never cross sequence boundaries or alter state-table layout;
- do not build a Polar/convolution/Titans megakernel initially.

### B4. Scheduler and cache constraints

Begin inside the already scheduled batch. Do not redesign scheduling until profiling demonstrates
a remaining imbalance. If necessary, greedily group by L40S-profiled effective tile work and
window-capped key length.

Do not consolidate the paged KV cache in this workstream. Direct page-table reads already avoid a
gather; copying adds traffic to a bandwidth-limited phase; the window bounds the scan; Titans state
is a larger decode cost; and a second cache representation complicates prefix sharing and graphs.

### Workstream B gate

Ship only if the grouped path:

- matches hidden outputs, K/V writes, all convolution states, and Titans states;
- improves mixed and long-tail prefill throughput by at least 20%;
- visibly reduces launches and GPU idle gaps;
- does not regress homogeneous dense prefill over 2% (otherwise preserve routing);
- grows peak memory by less than 5% unless explicitly justified;
- proves exact chunked continuation before enabling it by default.

Ship individual winning components; do not require all grouping stages to win.

---

## 6. Workstream C: selective CODA-style epilogues

CODA is an algebra/kernel-structure reference, not a direct backend fit. Its experiments use one
H100, standard Transformer++ blocks, hidden sizes 2048-8192, and CuTeDSL. ATMA is hidden size
1024 with hybrid convolution, Polar/Titans, TP inference, and custom FP8 training.

After Workstream A, consider only measured candidates:

1. MLP input GEMM + split + squared-ReLU;
2. MLP output GEMM + residual;
3. attention/conv output projection + residual;
4. output projection + partial statistics for following RMSNorm;
5. LM head + softcap/cross-entropy only if existing kernels leave material traffic.

Do not begin CODA-style backward rewriting without a training trace showing a worthwhile target.
If pursued, verify gradients against unfused BF16 before FP8 and design TP/distributed behavior
explicitly; CODA itself lists distributed execution as future work.

---

## 7. Correctness requirements

Minimum existing checks:

```bash
python -m pytest tests/test_dense_prefill.py -q
python -m pytest tests/test_decode_kernels.py -q
python -m pytest tests/test_edge_kernels.py -q
python -m tests.verify
python -m tests.verify --cuda
python -m tests.verify_fla
```

Add tests for each route covering randomized/non-power-of-two shapes, FP32 reductions, learned
affine and epsilon where applicable, TP or an explicit guard, CUDA graph replay, sequence-boundary
sentinels, and gradients for training kernels.

Grouped prefill must compare:

```text
hidden output
paged K and V
every Q/K/V Canon state
every LFM2 convolution state
every Titans state
last-token logits
```

Use the existing per-sequence path as oracle until the entire matrix passes.

---

## 8. Reporting template

Every result must include:

| Field | Required value |
|---|---|
| Commit | exact SHA and dirty status |
| Hardware | L40S, driver, CUDA, controlled clocks/power if used |
| Software | Python, PyTorch, Triton, kernel-library versions |
| Model | checkpoint/random, dtype, TP, architecture config |
| Workload | batch, individual/cached lengths, generated tokens |
| Route | dense, oracle, grouped, eager, CUDA graph |
| Timing | warmups, iterations, p50, p95, mean |
| Throughput | prefill, decode, overall tok/s |
| GPU evidence | launches, active time, important kernel times |
| Memory | peak allocation and workspace |
| Correctness | tests, tolerances, maximum observed error |

Report absolute time and speedup. Separate compile/autotune time from steady state. A percentage
win on a microsecond kernel is not a project-level result.

---

## 9. Execution sequence

```text
Phase 0: capture clean L40S baselines and heterogeneous-prefill traces
Phase 1: A1 K-projection + per-head RMSNorm; then Q if viable
Phase 2: A2 norm2 + MLP input GEMM; optionally absorb squared-ReLU
Phase 3: B1 benchmark harness and tile-cost model
Phase 4: B2 grouped fresh variable-length Polar prefill
Phase 5: B3 grouped Canon and Titans, one component at a time
Phase 6: decide from the remaining profile whether selective CODA work is justified
```

Commit or checkpoint each independently useful milestone. Do not combine fusion, scheduler, and
cache changes in one benchmark diff.

The likely useful endpoint is one or two shape-dispatched fused kernels plus grouped
variable-length Polar prefill, while retaining dense equal-length prefill, the paged cache,
state tables, and CUDA-graph decode. Large further decode gains will likely require reducing or
overlapping Titans/Polar bytes rather than more standalone elementwise fusion.
