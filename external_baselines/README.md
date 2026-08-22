# External baseline adapters

This package supplies checkpoint- and evaluation-compatible wrappers for the
supplementary TDA, Mamba-3, and GDN-2 experiments. The authoritative status, installation,
promotion, and six-machine commands are in
[`supplementary/robustness/README.md`](../supplementary/robustness/README.md).

## Architecture contract

- `tda_hybrid` retains Atma's 12 local LFM2 layers and replaces the four global
  layers with the official TDA kernel. Its matched Titans side channel remains enabled.
- `mamba3_native` uses the pinned upstream Mamba-3 SISO mixer in all 16 blocks.
- `gdn2_native` uses the pinned upstream GDN-2 mixer in all 16 blocks.
- All three share the same AdamW training harness and `ABLATION_*_JSON` log/evaluation
  contract.

Approved parameter counts are 388,641,924 for TDA, 368,108,416 for Mamba-3, and
366,322,864 for GDN-2, each within 5% of the 378.2M target.

## Compile integration

The pinned Mamba kernel graph-breaks under `torch.compile`. `custom_ops.py` wraps its
parameter-free SISO recurrence and fused RMSNorm with local `torch.library.custom_op`
forward/backward operators, following the established pattern in `model/blocks.py`.
The backward invokes the pinned implementation directly so rotary-angle gradients are
preserved. Projections, residuals, and MLPs remain visible to the compiler. Checkpoint
keys and parameter counts are unchanged, and no pinned third-party file is patched.

An opaque custom-op wrapper was also tested for GDN-2, but it slowed a resolved-width
full block to 0.81× eager speed because its backward repeated too much forward work. That
wrapper was removed. `gdn2_training.py` instead keeps the pinned FLA autograd kernels and
makes their three compiler-disabled boundaries explicit: short convolution, GDN-2
recurrence, and gated normalization. Projection/gate preparation, the output/MLP tail,
and the loss head compile separately; the fixed `mbs=4`, `seq_len=2048`
forward/backward microstep is replayed through a CUDA graph.

On the validation L40S, a complete 524,288-token step (64 microbatches, gradient
clipping, and one fused AdamW update) measured 12.318 seconds, **30.28% MFU**, and
12.09 GiB peak reserved memory. The prior eager path measured 22.73 seconds, 16.4% MFU,
and 24.72 GiB reserved. Checkpoint keys and the ordinary validation/evaluation path are
unchanged because the compiled helpers only reference parameters owned by the original
model.

`tda_training.py` removes the pinned wrapper's repeated `beta.item()` synchronization by
passing the fixed configured threshold as a Python scalar, split-compiles the tensor-only
regions around TDA and the matched Titans recurrence, and CUDA-graphs the microstep. It
uses the unchanged pinned FP32 Triton kernels with launch geometry tuned for the approved
64-dimensional heads. The tuned and upstream launches produced bit-identical bf16 output
and gradients at sequence lengths 128 and 2048. Three complete post-warmup L40S steps
measured a 9.853-second median, **40.20% MFU**, and **10.77 GiB peak reserved memory**,
compared with 19.933 seconds, 19.87% MFU, and 22.68 GiB for the old eager path using the
same pinned dependencies.

Synthetic CUDA checks on 2026-08-22 passed strict full-graph finite forward/backward for
Mamba-3 and forward parity plus split-compiled CUDA-graph backward for TDA and GDN-2. The
required per-machine `gpu_preflight` repeats the applicable check with the resolved full
model.

## Experiment status

The five Polar-component 1B runs are complete. None of the three external 1B pilots and
none of the external 10B runs is complete. A stale, interrupted GDN-2 marker/log from an
attempt that never reached step 1 is not an experiment result and must not be promoted.

After the three 1B pilot logs complete, `supplementary.robustness.promote` records the
prespecified choice. The two external 10B jobs are then TDA when promoted and exactly one
of Mamba-3/GDN-2. They must run on separate machines from the exact same dispatch commit.

Do not manually edit `parameter_count_approved`, do not regenerate configs with
`generate_configs --clean`, and do not patch `third_party/` checkouts. Use the setup,
preflight, and individual worker commands in the main robustness runbook.
