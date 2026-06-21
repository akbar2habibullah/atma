import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offsets = tl.arange(0, D)
    key = tl.load(key_ptr + idx * key_stride + offsets)
    value = tl.load(value_ptr + idx * value_stride + offsets)
    cache_offset = slot.to(tl.int64) * D + offsets
    tl.store(k_cache_ptr + cache_offset, key)
    tl.store(v_cache_ptr + cache_offset, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Write (N, num_heads, head_dim) key/value into paged KV cache slots."""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert slot_mapping.numel() == N
    if not key.is_cuda:
        # CPU fallback: Python scatter into flattened cache
        valid = slot_mapping >= 0
        slots = slot_mapping[valid]
        k_flat = k_cache.reshape(-1, num_heads, head_dim)
        v_flat = v_cache.reshape(-1, num_heads, head_dim)
        k_flat[slots] = key[valid]
        v_flat[slots] = value[valid]
        return
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    _store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0),
        k_cache, v_cache, slot_mapping, D,
    )


class Attention(nn.Module):
    """Holds per-layer KV cache references and attention scale.

    Actual attention computation is handled by AtmaAttention, which calls
    store_kvcache + the polar Triton kernels (kernel/polar_triton.py) directly.
    """

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])
