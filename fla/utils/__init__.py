"""Minimal FLA utility surface used by the in-tree Wall kernel."""

from functools import wraps

import torch


def contiguous(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args = tuple(x.contiguous() if isinstance(x, torch.Tensor) else x for x in args)
        kwargs = {k: (v.contiguous() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
        return fn(*args, **kwargs)
    return wrapper


def autocast_custom_fwd(fn):
    return fn


def autocast_custom_bwd(fn):
    return fn


def check_shared_mem(arch: str | None = None, device_index: int | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    if device_index is None:
        device_index = torch.cuda.current_device()
    major, _minor = torch.cuda.get_device_capability(device_index)
    if arch == "hopper":
        return major >= 9
    if arch == "ampere":
        return major >= 8
    return major >= 8
