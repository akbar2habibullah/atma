"""Triton kernels for the atma model.

Public entry points
-------------------
- ``polar_attention``      : autograd-aware FlashAttention-style polar attention
                             (drop-in for ``model.blocks.polar_attention_online``).
- ``polar_attention_fwd``  : forward-only (inference / no-grad) variant.

All kernels expect q, k, v shaped ``(B, H, T, dk)`` with GQA KV-heads already
expanded to H (exactly like the existing ``polar_reduce`` / ``polar_attention_online``
inputs). Everything is computed in fp32 internally; outputs are cast back to the
input dtype.
"""

from .polar_triton import polar_attention_fwd, HAS_TRITON

try:
    from .polar_triton import polar_attention, PolarAttentionTriton  # noqa: F401
except ImportError:  # backward not wired yet
    polar_attention = None
    PolarAttentionTriton = None

__all__ = [
    "polar_attention",
    "polar_attention_fwd",
    "PolarAttentionTriton",
    "HAS_TRITON",
]
