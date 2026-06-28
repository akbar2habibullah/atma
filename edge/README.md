# Edge Inference

`edge/` is a new, isolated inference runtime for small-device and single-session
use.  It deliberately does not import or modify the production `inference/`
scheduler.

The first backend is tinygrad:

- Atma checkpoint-compatible module names
- FP16 on tinygrad accelerator backends when available, FP32 CPU fallback with `--dtype auto`
- Tinygrad device selection through `--device cpu`, `--device cl`, or `--device webgpu`
- Stateful causal conv, polar K/V cache, and Titans memory recurrence
- Sequential prompt generation with raw token IDs or a best-effort tokenizer

Run:

```bash
python -m edge --model checkpoints/weights.pt --prompt "The future of edge inference is" --device cl --dtype fp16
python -m edge --ids 1 2 3 --max-new-tokens 16 --device cpu --dtype fp32
```

Backend smoke:

```bash
python -m edge.probe --device cpu --dtype fp32
python -m edge.probe --device cl --dtype fp16
```

Benchmark:

```bash
python -m edge.bench --device cl --dtype fp16 --prompt-len 16 --decode-tokens 8
python -m edge.jit_bench --device cl --dtype fp16
python -m edge.kernel_bench --device cl --dtype fp16 --heads 4 --tokens 32 --head-dim 16 --window 32 --skip-eager
python -m edge.kernel_sweep --profile rx6700xt --device cl --dtype fp16 --resume
```

## Verification

Recorded on 2026-06-24 in this workspace:

| Check | Result |
|---|---|
| `python -m pytest tests/test_edge.py -q` | `23 passed` FP32 CPU parity against `model.reference.ReferenceModel` |
| OpenCL FP32, tiny 4-layer no-memory model | max abs error `5.66e-7` vs reference |
| OpenCL FP32, tiny 4-layer memory + window model | max abs error `3.58e-7` vs reference |
| `python -m edge.probe --device cpu --dtype fp32` | finite logits |
| `python -m edge.probe --device cl --dtype fp16` | finite logits |
| `python -m edge.probe --device webgpu --dtype fp16` | blocked: tinygrad cannot load `webgpu`; set `WEBGPU_PATH` |

## OpenCL Profiling

Recorded on 2026-06-24 with the tiny random benchmark model
(`layers=4`, `hidden=64`, `prompt_len=16`, `decode_tokens=8`):

| Command | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| `python -m edge.bench --device cl --dtype fp16 --prompt-len 16 --decode-tokens 8 --warmup 1 --runs 3` before batched state realization | 35.10 | 3.51 |
| same command after batched state realization | 49.29 | 5.79 |
| same command with `--no-memory` | 96.30 | 5.81 |
| same command after custom UOp decode conv-step | 57.11 | 5.96 |
| same command after custom UOp decode conv-step with `--no-memory` | 108.27 | 7.23 |
| `python -m edge.bench --device cpu --dtype fp32 --prompt-len 16 --decode-tokens 8 --warmup 1 --runs 2` | 45.03 | 6.14 |

JIT microbenchmark for the custom decode conv-step kernel:

| Command | Eager custom kernel | TinyJit custom kernel |
|---|---:|---:|
| `python -m edge.jit_bench --device cl --dtype fp16 --channels 64 --kernel-size 3 --warmup 5 --iters 200 --model-iters 64 --layers 4 --hidden-size 64 --head-dim 16 --vocab-size 256 --attn-window 16` | 312.28 steps/s | 1445.10 steps/s |

The same command also JITs a fixed-state, no-memory, one-token model decode:

| Path | Decode steps/s |
|---|---:|
| dynamic eager no-memory decode (`edge.bench --no-memory`) | 7.23 |
| fixed-state `TinyJit` no-memory decode (`edge.jit_bench`) | 89.46 |
| fixed-state `TinyJit` memory decode (`edge.jit_bench --memory`) | 75.50 |

The JIT benchmark now reports one-time compile/capture timing separately from
replay throughput.  A current memory-enabled tiny run reports
`78.00 steps/s` replay, `first_call_s=11.3574`, and `capture_s=1.2553` for:

```bash
python -m edge.jit_bench --device cl --dtype fp16 --channels 64 --kernel-size 3 --warmup 5 --iters 200 --model-iters 64 --layers 4 --hidden-size 64 --head-dim 16 --vocab-size 256 --attn-window 16 --memory
```

The current optimization batches lazy state updates and realizes logits plus all
updated conv/K/V/memory tensors once per model call.  That removed many
per-layer synchronization points and improved OpenCL FP16 decode by about
`1.65x` on this tiny benchmark.  A second, narrower optimization uses
`Tensor.custom_kernel` and tinygrad UOps for the decode causal-conv step, fusing
conv output plus state rolling into one semi-manual OpenCL kernel.  The UOp DSL
code is kept in `edge/kernels.py`; model code calls the tensor-level
`causal_conv1d_decode_step(...)` helper so future custom kernels can be added
without spreading tinygrad compiler internals through the runtime.  This helps
the no-memory decode path more visibly than the memory path, which suggests the
remaining memory-enabled decode cost is dominated by the broader per-token
attention/cache/memory graph rather than only the conv state update.

The dynamic end-to-end decode path is not TinyJit-safe: `EdgeState` stores
Python dictionaries whose tensors are replaced during forward, and direct
`TinyJit(model_decode_step)` diverges after capture because Python state mutation
does not run on replay.  The optimized path adds a separate `EdgeStaticState`
with fixed-shape conv, K/V, and Titans memory buffers plus in-place
`assign(...)` updates.  The memory branch uses a custom `edge_gdn_step` UOp
kernel for one-token gated-delta read/update, enabling correct TinyJit decode
for `mem_enabled=True`.  Static polar attention now uses a custom
`edge_polar_decode` UOp kernel for the one-token softmax/null/content/magnitude
reduction.  On this tiny benchmark it improves the memory-enabled path slightly
but does not yet beat the previous no-memory static decode result, so the next
polar work should focus on the kernel shape and grouped K/V cache layout rather
than more API plumbing.

Tinygrad's upstream LLM reference updates K/V cache with
`cache.uop.after(cache_slice.uop.store(...))` inside a compiled `@function`.
Trying to port that directly into the current `EdgeStaticState` layout exposed a
callify shape-rewrite failure because this runtime still has per-layer
`realize()/assign()` boundaries.  For now the static cache stays on the proven
`assign(...)` update path; the cleaner reference-style cache update should be
revisited together with a deeper whole-block `@function` capture.

### Larger OpenCL Profile

Recorded on 2026-06-24 with a larger synthetic model
(`layers=6`, `hidden=96`, `head_dim=16`, `vocab=512`, `attn_window=32`,
`prompt_len=32`, `decode_tokens=16`):

| Command | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| `python -m edge.bench --device cl --dtype fp16 --layers 6 --hidden-size 96 --head-dim 16 --vocab-size 512 --attn-window 32 --prompt-len 32 --decode-tokens 16 --warmup 1 --runs 1` | 52.78 | 5.26 |
| same command with `--no-memory --runs 2` | 154.22 | 5.50 |

Static TinyJit decode on the same model shape, with a shorter 32-step decode
window (`max_context=34`):

| Command | Static decode steps/s |
|---|---:|
| `python -m edge.jit_bench --device cl --dtype fp16 --channels 96 --kernel-size 3 --warmup 5 --iters 300 --model-iters 32 --layers 6 --hidden-size 96 --head-dim 16 --vocab-size 512 --attn-window 32` | 64.95 |
| same command with `--memory` | 57.63 |

An attempted larger static profile
(`layers=8`, `hidden=128`, `vocab=1024`, `attn_window=64`,
`model_iters=64/128`) was interrupted after more than 90 seconds of first-capture
compilation without producing replay throughput.  That points at the current
`edge_polar_decode` implementation scaling OpenCL source size poorly because the
attention loop is unrolled over `max_context`.  The next flash-polar iteration
should use a less compile-heavy kernel shape before pushing context length.

### Standalone Flash Kernels

`edge.kernel_bench` profiles standalone flash prefill kernels before integrating
them into the full model graph.  This keeps compile/replay behavior attributable
to the kernel itself:

- `edge_polar_prefill`: causal polar prefill without materializing the
  `tokens x tokens` score matrix.
- `edge_gdn_prefill`: causal Titans gated-delta prefill scan, returning all reads
  plus final memory state.

Recorded on 2026-06-24 on OpenCL FP16:

| Command | Kernel | Replay runs/s | Replay tok/s | First call | Capture |
|---|---|---:|---:|---:|---:|
| `python -m edge.kernel_bench --device cl --dtype fp16 --heads 4 --tokens 16 --head-dim 16 --window 16 --iters 20 --skip-eager` | polar | 2303.09 | 36849.38 | 5.9337s | 0.0342s |
| same command | GDN | 3297.55 | 52760.88 | 4.7053s | 0.0364s |
| `python -m edge.kernel_bench --device cl --dtype fp16 --heads 4 --tokens 32 --head-dim 16 --window 32 --iters 20 --skip-eager` | polar | 1734.21 | 55494.86 | 11.1479s | 0.0530s |
| same command | GDN | 2679.89 | 85756.40 | 11.5207s | 0.0662s |
| `python -m edge.kernel_bench --device cl --dtype fp16 --kernel polar --heads 4 --tokens 48 --head-dim 16 --window 48 --iters 10 --skip-eager` | polar | 1611.14 | 77334.54 | 21.7240s | 0.0723s |

The current standalone kernels are intentionally simple and unroll over
`tokens`/`head_dim`.  That gives strong replay numbers at small-to-medium
prefill lengths, but compile time grows quickly: standalone GDN at 48 tokens and
combined polar+GDN at 64 tokens were both interrupted after more than 90 seconds
of compile time.  The next iteration should keep the standalone profiler but
change kernel shape, likely splitting work across token blocks or rows instead
of fully unrolling the prefill scan into one large kernel.

`edge.kernel_sweep` runs each shape in a fresh Python process and appends results
to `edge/results/kernel_sweep_rx6700xt.csv` plus JSONL.  The default RX 6700 XT
profile sweeps:

- `heads`: 4, 8
- `head_dim`: 16, 32
- `tokens`: 16, 32, 48, 64, 96, 128
- `kernel`: polar, GDN

Each case records `ok`, `compile_timeout`, `compile_error`, or `parse_error`,
along with first-call, capture, and replay timings.  The default per-case
timeout is 1800 seconds; use `--case-timeout-s 0` for an overnight/no-timeout
compile run, and `--resume` to skip completed `ok` cases.  Different GPUs should
start by changing the sweep grid, especially `tokens` and `head_dim`, because
the current kernels trade compile time for replay speed and are sensitive to
compiler/backend limits.

Starter RX 6700 XT sweep command:

```bash
python -m edge.kernel_sweep --profile rx6700xt --device cl --dtype fp16 --kernel polar gdn --heads 4 --tokens 16 32 48 --head-dim 16 --iters 10 --case-timeout-s 180 --out edge/results/kernel_sweep_rx6700xt_starter.csv --jsonl edge/results/kernel_sweep_rx6700xt_starter.jsonl
```

Results:

| Kernel | Heads | Tokens | Head dim | Status | Wall s | Replay tok/s | First call | Capture |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| polar | 4 | 16 | 16 | ok | 5.65 | 33573.95 | 2.4564s | 0.0331s |
| polar | 4 | 32 | 16 | ok | 10.09 | 48635.18 | 5.5685s | 0.0756s |
| polar | 4 | 48 | 16 | ok | 12.93 | 102548.76 | 8.5298s | 0.0766s |
| GDN | 4 | 16 | 16 | ok | 7.25 | 41569.24 | 3.1305s | 0.0355s |
| GDN | 4 | 32 | 16 | ok | 11.39 | 82847.90 | 7.1436s | 0.0629s |
| GDN | 4 | 48 | 16 | compile_timeout | 181.04 | | | |

Suggested overnight RX 6700 XT run:

```bash
python -m edge.kernel_sweep --profile rx6700xt --device cl --dtype fp16 --case-timeout-s 0 --resume
```

Full overnight RX 6700 XT sweep output lives in
`edge/results/kernel_sweep_rx6700xt.csv` and completed all 48 default cases with
status `ok`.  Best replay results:

| Kernel | Heads | Tokens | Head dim | Replay tok/s | First call |
|---|---:|---:|---:|---:|---:|
| polar | 4 | 96 | 16 | 174348.92 | 78.9964s |
| polar | 4 | 64 | 32 | 121919.28 | 79.2241s |
| GDN | 4 | 32 | 16 | 123302.19 | 6.8496s |
| GDN | 4 | 16 | 32 | 72031.19 | 12.7738s |

Key observations from the full sweep:

- Polar `head_dim=16` peaks around 96 tokens; 128 tokens still compiles but
  replay drops slightly.
- Polar `head_dim=32` peaks earlier, around 64 tokens, and compile time jumps
  above 180 seconds by 96 tokens.
- GDN `head_dim=16` replays well from 32 to 128 tokens, but first-call compile
  jumps from ~7 seconds at 32 tokens to ~192 seconds at 48 tokens, then ~18.5
  minutes at 128 tokens.
- GDN `head_dim=32` is much less attractive in the current unrolled kernel:
  replay is lower and compile time is already ~5 minutes at 32 tokens.

This makes the first real kernel-shape target pretty clear: implement a chunked
or tiled GDN prefill variant first, especially for `head_dim=16`, because replay
is promising but compile time explodes at the token loop.  For polar, a tiled
variant should aim to keep the `head_dim=16`, 64-96 token replay region while
reducing first-call source size; `head_dim=32` needs tiling sooner.

First GDN token-chunk tiling result:

```bash
python -m edge.kernel_sweep --profile rx6700xt --device cl --dtype fp16 --kernel gdn --gdn-variant chunked --chunk-size 16 24 32 --heads 4 --tokens 48 64 96 128 --head-dim 16 --iters 8 --case-timeout-s 240 --out edge/results/kernel_sweep_rx6700xt_gdn_chunked.csv --jsonl edge/results/kernel_sweep_rx6700xt_gdn_chunked.jsonl
```

| Tokens | Best chunk | Replay tok/s | First call | Wall s |
|---:|---:|---:|---:|---:|
| 48 | 32 | 53103.22 | 10.1868s | 13.96 |
| 64 | 32 | 71970.76 | 7.0611s | 10.13 |
| 96 | 32 | 79915.92 | 8.9482s | 12.15 |
| 128 | 32 | 75764.86 | 7.9561s | 11.20 |

Compared with the fully unrolled GDN kernel, token chunking cuts first-call
compile from minutes to roughly 7-10 seconds for 64-128 tokens.  Replay is lower
than the unrolled kernel, so this is the compile-friendly baseline rather than
the final fast path.  The next GDN tile should preserve this bounded source size
while recovering replay, likely by grouping multiple rows per work item or using
local memory for the per-token `k` vector shared across value rows.

Python:

```python
from edge import EdgeLLM, EdgeSamplingParams

llm = EdgeLLM("checkpoints/weights.pt", dtype="fp16")
out = llm.generate("Hello", EdgeSamplingParams(max_tokens=32, temperature=0.7))[0]
print(out["text"])
```

This is intentionally closer to tinygrad/llama.cpp in shape than to vLLM: one
runtime object, one mutable state cache, no paged scheduler, no CUDA graphs, and
no tensor parallelism.  Tinygrad owns the backend dispatch, so OpenCL/WebGPU
work can happen inside this runtime without touching the production engine.
