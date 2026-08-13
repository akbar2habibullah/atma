"""Diagnostics and causal interventions for ATMA's Titans retention gate."""

from .clamp import (
    FORMAT_VERSION,
    apply_gamma_clamp,
    half_life_to_gamma,
    half_life_to_logit,
    load_clamp_spec,
)

__all__ = [
    "FORMAT_VERSION",
    "apply_gamma_clamp",
    "half_life_to_gamma",
    "half_life_to_logit",
    "load_clamp_spec",
]
