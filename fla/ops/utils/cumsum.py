"""Minimal global cumsum helper compatible with FLA's Wall usage."""

import torch


def _time_dim(x: torch.Tensor, head_first: bool) -> int:
    if x.dim() == 3:
        return 2 if head_first else 1
    if x.dim() == 4:
        return 2 if head_first else 1
    raise ValueError(f"Unsupported input shape {tuple(x.shape)}")


def _cumsum(x: torch.Tensor, dim: int, reverse: bool) -> torch.Tensor:
    if reverse:
        return torch.flip(torch.cumsum(torch.flip(x, dims=(dim,)), dim=dim), dims=(dim,))
    return torch.cumsum(x, dim=dim)


def chunk_global_cumsum(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if cu_seqlens is not None and s.shape[0] != 1:
        raise AssertionError("Only batch size 1 is supported when cu_seqlens are provided")

    dim = _time_dim(s, head_first)
    out_dtype = output_dtype or s.dtype
    src = s.to(out_dtype)

    if cu_seqlens is None:
        out = _cumsum(src, dim=dim, reverse=reverse)
    else:
        out = torch.empty_like(src)
        cu_cpu = cu_seqlens.detach().cpu().tolist()
        index = [slice(None)] * s.dim()
        for bos, eos in zip(cu_cpu[:-1], cu_cpu[1:], strict=False):
            index[dim] = slice(bos, eos)
            seg = tuple(index)
            out[seg] = _cumsum(src[seg], dim=dim, reverse=reverse)

    if scale is not None:
        out = out * scale
    return out
