"""Shared RNG setup for reproducible training runs.

The training stream is intentionally not shuffled, so ``data_seed`` is recorded for
provenance but does not currently change shard order.  ``init_seed`` controls Python,
NumPy, PyTorch, and CUDA RNGs.  Evaluation code receives ``eval_seed`` separately.
"""
from __future__ import annotations

import random
import subprocess
import sys
from typing import Any


def seed_run(cfg: dict[str, Any], torch_module=None) -> dict[str, Any]:
    """Seed all RNGs used by a training process and return resolved metadata."""
    seed = int(cfg.get("init_seed", cfg.get("seed", 0)))
    data_seed = int(cfg.get("data_seed", seed))
    eval_seed = int(cfg.get("eval_seed", 1234))

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except ImportError:
        pass

    torch = torch_module
    if torch is None:
        import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms are opt-in: several fused training kernels either do
    # not provide a deterministic implementation or pay a large throughput penalty.
    deterministic = bool(cfg.get("deterministic_algorithms", False))
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic

    return {
        "init_seed": seed,
        "data_seed": data_seed,
        "eval_seed": eval_seed,
        "deterministic_algorithms": deterministic,
        "data_order": "sorted_finewebedu_shards_contiguous_stream",
    }


def runtime_metadata(torch_module=None) -> dict[str, Any]:
    """Collect lightweight provenance without making training depend on Git."""
    torch = torch_module
    if torch is None:
        import torch
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    cuda = torch.cuda.is_available()
    return {
        "git_commit": commit,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "gpu": torch.cuda.get_device_name() if cuda else None,
    }
