import os
import torch
from torch import nn
import torch.nn.functional as F

from inference.utils.context import get_context


def _load_fa3():
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major != 9:
            return None
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('kernels-community/flash-attn3').flash_attn_interface
    except Exception:
        print("FA3 not available, falling back to PyTorch's scaled_dot_product_attention")
        return None


_fa3 = _load_fa3()


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """
    Store new keys and values into their allocated slots in the paged KV cache.
    key, value: (seqlen, num_heads, head_dim) or (num_heads, head_dim)
    slot_mapping: (seqlen,) or (1,) containing slot indices
    """
    # Normalize shapes to include batch/token dimension if squeezed
    if key.ndim == 2:
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        
    N = key.shape[0]
    block_size = k_cache.shape[1]
    num_blocks = k_cache.shape[0]
    num_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]

    # Reshape cache to flat list of slots to easily write using index assignment
    k_cache_flat = k_cache.view(num_blocks * block_size, num_heads, head_dim)
    v_cache_flat = v_cache.view(num_blocks * block_size, num_heads, head_dim)

    # Convert slot_mapping to 1D if scalar
    if slot_mapping.ndim == 0:
        slot_mapping = slot_mapping.unsqueeze(0)

    valid_mask = slot_mapping != -1
    valid_slots = slot_mapping[valid_mask].long()
    if valid_slots.numel() > 0:
        k_cache_flat[valid_slots] = key[valid_mask].to(k_cache.dtype)
        v_cache_flat[valid_slots] = value[valid_mask].to(v_cache.dtype)


def gather_kv_from_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gathers the past keys and values from the paged KV cache blocks for a specific sequence.
    Returns:
        k_out, v_out: (seq_len, num_heads, head_dim)
    """
    block_size = k_cache.shape[1]
    num_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]

    # Filter out the placeholder block IDs (-1)
    if block_table is not None:
        if block_table.ndim == 0:
            block_table = block_table.unsqueeze(0)
        blocks = [b.item() for b in block_table if b != -1]
    else:
        blocks = []

    # Pre-allocate output tensors
    k_out = torch.zeros(seq_len, num_heads, head_dim, dtype=k_cache.dtype, device=k_cache.device)
    v_out = torch.zeros(seq_len, num_heads, head_dim, dtype=v_cache.dtype, device=v_cache.device)

    curr_len = 0
    for block_id in blocks:
        chunk_len = min(block_size, seq_len - curr_len)
        if chunk_len <= 0:
            break
        k_out[curr_len : curr_len + chunk_len] = k_cache[block_id, :chunk_len]
        v_out[curr_len : curr_len + chunk_len] = v_cache[block_id, :chunk_len]
        curr_len += chunk_len

    return k_out, v_out


class Attention(nn.Module):

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        slot_mapping: torch.Tensor = None,
        block_table: torch.Tensor = None,
        seq_len: int = 0,
    ) -> torch.Tensor:
        """
        Runs attention for a single sequence.
        q, k, v shape: (seqlen, num_heads, head_dim) or (num_heads, head_dim)
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 1. Store newly projected keys/values in the physical block slots
        if k_cache.numel() > 0 and v_cache.numel() > 0 and slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, slot_mapping)

        # 2. Gather keys and values from physical blocks if we have cache allocated
        if block_table is not None and seq_len > 0:
            ki, vi = gather_kv_from_cache(k_cache, v_cache, block_table, seq_len)
        else:
            ki, vi = k, v

        # 3. Perform attention
        if context.is_prefill:
            # Prefill: q, ki, vi shapes: (seqlen, num_heads, head_dim)
            qi = q.unsqueeze(0).transpose(1, 2)
            ki = ki.unsqueeze(0).transpose(1, 2)
            vi = vi.unsqueeze(0).transpose(1, 2)
            
            # Use PyTorch SDPA with causal masking
            oi = F.scaled_dot_product_attention(qi, ki, vi, scale=self.scale, is_causal=True)
            return oi.transpose(1, 2).squeeze(0)
            
        else:
            # Decode: single token query. q shape: (num_heads, head_dim)
            qi = q.unsqueeze(0).unsqueeze(0).transpose(1, 2)  # (1, num_heads, 1, head_dim)
            ki = ki.unsqueeze(0).transpose(1, 2)  # (1, num_heads, seq_len, head_dim)
            vi = vi.unsqueeze(0).transpose(1, 2)  # (1, num_heads, seq_len, head_dim)
            
            # Non-causal scaled dot product attention for a single query token
            oi = F.scaled_dot_product_attention(qi, ki, vi, scale=self.scale, is_causal=False)
            return oi.transpose(1, 2).squeeze(0).squeeze(0)  # (num_heads, head_dim)
