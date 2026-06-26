"""Small varlen chunk-index helper compatible with FLA's API."""

import torch


def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    pairs = []
    cu_cpu = cu_seqlens.detach().cpu().tolist()
    for i, (bos, eos) in enumerate(zip(cu_cpu[:-1], cu_cpu[1:], strict=False)):
        length = eos - bos
        for chunk in range((length + chunk_size - 1) // chunk_size):
            pairs.append((i, chunk))
    return torch.tensor(pairs, dtype=torch.long, device=cu_seqlens.device)
