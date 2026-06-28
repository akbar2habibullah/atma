"""Small-footprint Atma inference runtime.

The edge package is intentionally separate from ``inference/``.  It starts with
a single-process FP16-capable tinygrad backend and a simple state cache, leaving
the production paged scheduler untouched.
"""

from edge.config import EdgeConfig, EdgeSamplingParams
from edge.engine import EdgeLLM
from edge.loader import load_edge_model
from edge.model import EdgeAtma, EdgeState

__all__ = [
    "EdgeAtma",
    "EdgeConfig",
    "EdgeLLM",
    "EdgeSamplingParams",
    "EdgeState",
    "load_edge_model",
]
