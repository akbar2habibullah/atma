# External baseline adapters

This package supplies Atma-evaluation-compatible wrappers for the supplementary TDA,
Mamba-3, and GDN-2 runs. It does not vendor third-party kernels. The experiment pins
their source commits in [`supplementary/robustness/dependencies.json`](../supplementary/robustness/dependencies.json).

- `tda_hybrid` keeps Atma's 12 local LFM2 layers and replaces the four global layers
  with TDA. This is the Stage-II mechanism baseline.
- `mamba3_native` and `gdn2_native` use the upstream mixer in all 16 blocks, matching
  the role of `raven_native` as an external model-family baseline.

The wrappers deliberately share Raven's AdamW training harness and the normal
`ABLATION_*_JSON` log contract. They require CUDA and must pass the GPU preflight in the
supplementary runbook before any 1B-token pilot is launched.

