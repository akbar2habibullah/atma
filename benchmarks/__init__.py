"""Atma benchmark harness (scaled-up evals for final candidates).

Built against the production inference interface (inference.LLM.generate; see
docs/inference.md). FIRST benchmark: BABILong (long-context reasoning-in-a-haystack).

*** NOT FUNCTIONAL YET ***
The paged inference engine still runs LEGACY softmax attention and does not implement Polar
attention, the Titans memory branch, the Canon convs, or the training sliding window. So for
any checkpoint from the polar/titans research line, the engine's outputs are INVALID and every
number this harness produces is a PLACEHOLDER until the inference port (the tracked task in
docs/inference.md) is complete. The harness is written now against the intended stable
autoregressive interface so it 'just works' the moment inference is finished.
"""
