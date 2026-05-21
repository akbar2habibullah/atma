import torch
from torch import nn
import triton
import triton.language as tl


try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    HAS_FLASH_ATTN2 = True
except ImportError:
    HAS_FLASH_ATTN2 = False
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None


def _load_fa3():
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('kernels-community/flash-attn3').flash_attn_interface
    except Exception:
        return None


_fa3 = _load_fa3()

if not HAS_FLASH_ATTN2 and _fa3 is None:
    print("Warning: neither flash-attn2 nor FA3 found; attention will fall back to SDPA (slow path).")


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
    tl.store(k_cache_ptr + slot * D + offsets, key)
    tl.store(v_cache_ptr + slot * D + offsets, value)


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

    Actual attention computation is handled by AtmaAttention which calls
    store_kvcache + flash_attn_with_kvcache / flash_attn_varlen_func directly.
    """

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])
