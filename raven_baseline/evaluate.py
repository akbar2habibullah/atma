"""Raven baseline evaluation wrapper.

The Raven models implement Atma's lightweight eval contract:
`embed`, iterable `blocks`, `norm`, `proj`, and block outputs `(x, reg_loss, align_loss)`.
"""

from ablation.evaluate import run_eval

__all__ = ["run_eval"]

