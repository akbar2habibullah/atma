"""Memory-efficient cross entropy kernels for the ATMA training head."""

from .softcap import HAS_TRITON, softcap_cross_entropy_reference, softcap_linear_cross_entropy

__all__ = [
    "HAS_TRITON",
    "softcap_cross_entropy_reference",
    "softcap_linear_cross_entropy",
]
