"""
verify.py — Per-layer numerical verification.

Checks that train and inference forward passes produce the same outputs as the
canonical reference implementation in model/reference.py, using the same random
weights and random input tensors.

For each layer type:
  train      — batch-first (B, T, H), same math as reference
  infer prefill — flat packed tokens (T, H), should match reference on all T positions
  infer decode  — single token (1, H) after prefill state seeding, should match
                  reference output at the last position

Usage:
    python verify.py
"""

import sys
import torch
import torch.nn.functional as F

from model.config import AtmaConfig
import model.reference as ref_mod
from model.layers import RMSNorm, MLP

from inference.models.atma import (
    AtmaAttention as InferAttn,
    AtmaLFM2Conv as InferConv,
    AtmaDecoderBlock as InferBlock,
    Atma as InferModel,
    _infer_linear,
)
from inference.engine.sequence import Sequence
from inference.utils.context import set_context, get_context, reset_context

# ─── test config ────────────────────────────────────────────────────────────

# Small dimensions for fast verification; head_dim must divide hidden_size
TEST_CONFIG = AtmaConfig(
    vocab_size=256,
    num_hidden_layers=4,   # layers 0,1,3 → conv, layer 2 → attn
    hidden_size=128,
    head_dim=64,           # 2 heads
    attn_kernel_size=4,
    conv_kernel_size=3,
)
SEQ_LEN = 8   # full sequence length; prefill = SEQ_LEN-1, decode = last token
ATOL = 1e-4   # absolute tolerance for float32 comparisons

# ─── utilities ───────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0


def _result(label: str, y_ref: torch.Tensor, y_test: torch.Tensor):
    global PASS, FAIL
    y_ref = y_ref.detach().float().reshape(-1)
    y_test = y_test.detach().float().reshape(-1)
    assert y_ref.shape == y_test.shape, f"{label}: shape mismatch {y_ref.shape} vs {y_test.shape}"
    diff = (y_ref - y_test).abs().max().item()
    if diff <= ATOL:
        print(f"  PASS  {label}  (max_diff={diff:.2e})")
        PASS += 1
    else:
        print(f"  FAIL  {label}  (max_diff={diff:.2e}, atol={ATOL:.2e})")
        FAIL += 1


def _copy(src: torch.nn.Module, dst: torch.nn.Module):
    """Copy weights from src into dst. Fails if dst has any param not covered by src."""
    missing, _ = dst.load_state_dict(src.state_dict(), strict=False)
    if missing:
        raise RuntimeError(f"Weight copy failed — dst has uncovered params: {missing}")


# ─── inference context helpers ───────────────────────────────────────────────

def _ctx_prefill(seq: Sequence, seqlen: int, conv_state_tables: dict = None):
    cu = torch.tensor([0, seqlen], dtype=torch.int32)
    slots = torch.arange(seqlen, dtype=torch.int32)
    set_context(
        is_prefill=True,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=seqlen, max_seqlen_k=seqlen,
        slot_mapping=slots,
        block_tables=None,
        seqlens_q=[seqlen],
        conv_state_tables=conv_state_tables,
    )
    get_context().seqs = [seq]


def _ctx_decode(seq: Sequence, decode_slot: int, context_len: int, block_tables: torch.Tensor,
                conv_state_tables: dict = None, seq_slots: torch.Tensor = None):
    set_context(
        is_prefill=False,
        slot_mapping=torch.tensor([decode_slot], dtype=torch.int32),
        context_lens=torch.tensor([context_len], dtype=torch.int32),
        block_tables=block_tables,
        context_lens_list=[context_len],
        conv_state_tables=conv_state_tables,
        seq_slots=seq_slots,
    )
    get_context().seqs = [seq]


def _alloc_kv(attn_layer: InferAttn, config: AtmaConfig, capacity: int = 32):
    """Allocate a fresh KV cache (capacity slots) wired into attn_layer.attn."""
    block_size = 16
    num_blocks = (capacity + block_size - 1) // block_size
    nh, hd = config.num_attention_heads, config.head_dim
    attn_layer.attn.k_cache = torch.zeros(num_blocks, block_size, nh, hd)
    attn_layer.attn.v_cache = torch.zeros(num_blocks, block_size, nh, hd)


def _fresh_seq(n_tokens: int) -> Sequence:
    """Sequence with n_tokens filled, 1 block allocated, num_cached_tokens=0."""
    seq = Sequence([0] * n_tokens)
    seq.num_scheduled_tokens = n_tokens
    seq.block_table = [0]
    seq.seq_slot = 0
    return seq


def _alloc_conv_state_tables(config: AtmaConfig, layers: list) -> dict:
    """Allocate conv state tensors (1 seq slot).

    layers: list of ints (uses i%4==2 to detect attn) or (layer_idx, is_attn) tuples.
    """
    tables = {}
    for entry in layers:
        if isinstance(entry, tuple):
            i, is_attn = entry
        else:
            i, is_attn = entry, (entry % 4 == 2)
        if is_attn:
            for suffix in ("q", "k", "v"):
                tables[f"attn_{i}_{suffix}"] = torch.zeros(
                    1, config.hidden_size, config.attn_kernel_size - 1
                )
        else:
            tables[f"conv_{i}_gated"] = torch.zeros(
                1, config.hidden_size, config.conv_kernel_size - 1
            )
    return tables


# ─── per-layer verifiers ─────────────────────────────────────────────────────

def verify_rmsnorm(config: AtmaConfig):
    print("\n── RMSNorm ──")
    dim = config.hidden_size
    ref = RMSNorm(dim)
    infer = RMSNorm(dim)
    _copy(ref, infer)

    x = torch.randn(1, SEQ_LEN, dim)
    with torch.no_grad():
        _result("infer == ref", ref(x), infer(x))

    # Also verify the optional residual path
    r = torch.randn_like(x)
    with torch.no_grad():
        y_ref = ref(x + r)
        y_infer_res = infer(x, residual=r)
    _result("infer residual == ref", y_ref, y_infer_res)


def verify_mlp(config: AtmaConfig):
    print("\n── MLP ──")
    dim = config.hidden_size

    ref_mlp = MLP(dim, linear_cls=ref_mod.Linear)
    infer_mlp = MLP(dim, linear_cls=_infer_linear)
    _copy(ref_mlp, infer_mlp)

    x_batch = torch.randn(1, SEQ_LEN, dim)   # (B, T, H)
    x_flat = x_batch.squeeze(0)               # (T, H)  — inference prefill shape
    x_tok = x_batch[0, :1, :]                 # (1, H)  — inference decode shape

    with torch.no_grad():
        y_ref = ref_mlp(x_batch).squeeze(0)
        y_infer_prefill = infer_mlp(x_flat)
        y_infer_decode = infer_mlp(x_tok)

    _result("infer prefill == ref", y_ref, y_infer_prefill)
    _result("infer decode == ref (single token)", ref_mlp(x_tok.unsqueeze(0)).squeeze(), y_infer_decode.squeeze())

    # Train MLP uses same shared class — only import if kernels available
    try:
        from train.model import Linear as TrainLinear
        train_mlp = MLP(dim, linear_cls=TrainLinear)
        _copy(ref_mlp, train_mlp)
        with torch.no_grad():
            y_train = train_mlp(x_batch).squeeze(0)
        _result("train == ref", y_ref, y_train)
    except Exception:
        print("  SKIP  train (kernel unavailable)")


def verify_conv(config: AtmaConfig):
    print("\n── LFM2Conv ──")
    dim = config.hidden_size
    k = config.conv_kernel_size
    prefill_len = SEQ_LEN - 1  # tokens seen before decode step

    ref_conv = ref_mod.LFM2Conv(dim, kernel_size=k)
    infer_conv = InferConv(layer_idx=0, dim=dim, kernel_size=k)
    _copy(ref_conv, infer_conv)

    x = torch.randn(1, SEQ_LEN, dim)
    x_ref = x                           # (1, T, H)
    x_prefill = x[0, :prefill_len, :]   # (T-1, H)
    x_decode = x[0, prefill_len:, :]    # (1, H)  = last token

    with torch.no_grad():
        y_ref_all = ref_conv(x_ref).squeeze(0)   # (T, H)

    cst = _alloc_conv_state_tables(config, [0])
    seq_slots = torch.tensor([0])

    # ── prefill ──
    seq_p = _fresh_seq(prefill_len)
    _ctx_prefill(seq_p, prefill_len, conv_state_tables=cst)
    with torch.no_grad():
        y_infer_prefill = infer_conv(x_prefill)  # (T-1, H)
    reset_context()

    _result("infer prefill == ref (positions 0..T-2)", y_ref_all[:prefill_len], y_infer_prefill)

    # ── decode ── seed conv state via prefill, then step one token
    seq_d = _fresh_seq(prefill_len)
    _ctx_prefill(seq_d, prefill_len, conv_state_tables=cst)
    with torch.no_grad():
        _ = infer_conv(x_prefill)   # populate conv_states
    reset_context()

    seq_d.num_cached_tokens = prefill_len
    seq_d.append_token(0)   # num_tokens = SEQ_LEN now

    # slot for new token: block_table[-1]*block_size + last_block_num_tokens - 1
    decode_slot = seq_d.block_table[-1] * Sequence.block_size + seq_d.last_block_num_tokens - 1
    _ctx_decode(seq_d, decode_slot, SEQ_LEN, block_tables=torch.tensor([[0]], dtype=torch.int32),
                conv_state_tables=cst, seq_slots=seq_slots)
    with torch.no_grad():
        y_infer_decode = infer_conv(x_decode)   # (1, H)
    reset_context()

    _result("infer decode == ref (last position)", y_ref_all[prefill_len], y_infer_decode.squeeze(0))

    # ── train ──
    try:
        from train.model import LFM2Conv as TrainConv
        train_conv = TrainConv(dim, kernel_size=k)
        _copy(ref_conv, train_conv)
        with torch.no_grad():
            y_train = train_conv(x_ref).squeeze(0)
        _result("train == ref", y_ref_all, y_train)
    except Exception:
        print("  SKIP  train (kernel unavailable)")


def verify_attention(config: AtmaConfig):
    print("\n── CausalSelfAttention ──")
    dim = config.hidden_size
    hd = config.head_dim
    k = config.attn_kernel_size
    prefill_len = SEQ_LEN - 1

    ref_attn = ref_mod.CausalSelfAttention(dim, head_dim=hd, kernel_size=k)
    infer_attn = InferAttn(layer_idx=0, dim=dim, head_dim=hd, kernel_size=k)
    _copy(ref_attn, infer_attn)
    _alloc_kv(infer_attn, config, capacity=SEQ_LEN + 4)

    x = torch.randn(1, SEQ_LEN, dim)
    x_prefill = x[0, :prefill_len, :]   # (T-1, H)
    x_decode = x[0, prefill_len:, :]    # (1, H)

    with torch.no_grad():
        y_ref_all = ref_attn(x).squeeze(0)   # (T, H)

    cst = _alloc_conv_state_tables(config, [(0, True)])
    seq_slots = torch.tensor([0])

    # ── prefill ──
    seq_p = _fresh_seq(prefill_len)
    _ctx_prefill(seq_p, prefill_len, conv_state_tables=cst)
    with torch.no_grad():
        y_infer_prefill = infer_attn(x_prefill)   # (T-1, H)
    reset_context()

    _result("infer prefill == ref (positions 0..T-2)", y_ref_all[:prefill_len], y_infer_prefill)

    # ── decode ── re-seed via fresh prefill, then one decode step
    _alloc_kv(infer_attn, config, capacity=SEQ_LEN + 4)   # fresh cache
    seq_d = _fresh_seq(prefill_len)
    _ctx_prefill(seq_d, prefill_len, conv_state_tables=cst)
    with torch.no_grad():
        _ = infer_attn(x_prefill)   # fill KV cache slots 0..T-2 + conv states
    reset_context()

    seq_d.num_cached_tokens = prefill_len
    seq_d.append_token(0)
    decode_slot = seq_d.block_table[-1] * Sequence.block_size + seq_d.last_block_num_tokens - 1
    _ctx_decode(seq_d, decode_slot, SEQ_LEN, block_tables=torch.tensor([[0]], dtype=torch.int32),
                conv_state_tables=cst, seq_slots=seq_slots)
    with torch.no_grad():
        y_infer_decode = infer_attn(x_decode)   # (1, H)
    reset_context()

    _result("infer decode == ref (last position)", y_ref_all[prefill_len], y_infer_decode.squeeze(0))

    # ── train ──
    try:
        from train.model import CausalSelfAttention as TrainAttn
        train_attn = TrainAttn(dim, head_dim=hd, kernel_size=k)
        _copy(ref_attn, train_attn)
        with torch.no_grad():
            y_train = train_attn(x).squeeze(0)
        _result("train == ref", y_ref_all, y_train)
    except Exception:
        print("  SKIP  train (kernel unavailable)")


def verify_block(config: AtmaConfig):
    """Verify a full decoder block (pre-norm + attn/conv + MLP with residuals)."""
    for layer_idx in range(config.num_hidden_layers):
        is_attn = (layer_idx % 4 == 2)
        kind = "attn" if is_attn else "conv"
        print(f"\n── Block {layer_idx} ({kind}) ──")
        _verify_one_block(config, layer_idx, is_attn)


def _verify_one_block(config: AtmaConfig, layer_idx: int, is_attn: bool):
    dim = config.hidden_size
    prefill_len = SEQ_LEN - 1

    ref_block = ref_mod.Block(
        dim, attention=is_attn,
        head_dim=config.head_dim,
        attn_kernel_size=config.attn_kernel_size,
        conv_kernel_size=config.conv_kernel_size,
    )
    infer_block = InferBlock(
        layer_idx, dim, attention=is_attn,
        head_dim=config.head_dim,
        attn_kernel_size=config.attn_kernel_size,
        conv_kernel_size=config.conv_kernel_size,
    )
    _copy(ref_block, infer_block)

    # Wire KV cache into the attention sub-layer if this is an attention block
    if is_attn:
        _alloc_kv(infer_block.attn, config, capacity=SEQ_LEN + 4)

    x = torch.randn(1, SEQ_LEN, dim)
    x_prefill = x[0, :prefill_len, :]
    x_decode = x[0, prefill_len:, :]

    with torch.no_grad():
        y_ref_all = ref_block(x).squeeze(0)   # (T, H)

    cst = _alloc_conv_state_tables(config, [(layer_idx, is_attn)])
    seq_slots = torch.tensor([0])

    # ── prefill ──
    seq_p = _fresh_seq(prefill_len)
    _ctx_prefill(seq_p, prefill_len, conv_state_tables=cst)
    with torch.no_grad():
        y_infer_prefill = infer_block(x_prefill)
    reset_context()

    _result(f"block {layer_idx} infer prefill == ref", y_ref_all[:prefill_len], y_infer_prefill)

    # ── decode ── re-seed state
    if is_attn:
        _alloc_kv(infer_block.attn, config, capacity=SEQ_LEN + 4)

    seq_d = _fresh_seq(prefill_len)
    _ctx_prefill(seq_d, prefill_len, conv_state_tables=cst)
    with torch.no_grad():
        _ = infer_block(x_prefill)
    reset_context()

    seq_d.num_cached_tokens = prefill_len
    seq_d.append_token(0)
    decode_slot = seq_d.block_table[-1] * Sequence.block_size + seq_d.last_block_num_tokens - 1
    _ctx_decode(seq_d, decode_slot, SEQ_LEN, block_tables=torch.tensor([[0]], dtype=torch.int32),
                conv_state_tables=cst, seq_slots=seq_slots)
    with torch.no_grad():
        y_infer_decode = infer_block(x_decode)
    reset_context()

    _result(f"block {layer_idx} infer decode == ref", y_ref_all[prefill_len], y_infer_decode.squeeze(0))

    # ── train ──
    try:
        from train.model import Block as TrainBlock
        train_block = TrainBlock(
            dim, attention=is_attn,
            head_dim=config.head_dim,
            attn_kernel_size=config.attn_kernel_size,
            conv_kernel_size=config.conv_kernel_size,
        )
        _copy(ref_block, train_block)
        with torch.no_grad():
            y_train, _ = train_block(x)   # Block returns (x, reg_loss)
        _result(f"block {layer_idx} train == ref", y_ref_all, y_train.squeeze(0))
    except Exception:
        print(f"  SKIP  block {layer_idx} train (kernel unavailable)")


def verify_model(config: AtmaConfig):
    """End-to-end model: reference prefill logits == inference prefill logits."""
    print("\n── Full Model (prefill) ──")
    prefill_len = SEQ_LEN - 1
    block_size = Sequence.block_size

    ref_model = ref_mod.ReferenceModel(config)
    infer_model = InferModel(config)
    _copy(ref_model, infer_model)

    # Allocate KV caches for all attention layers in the inference model
    attn_modules = [m for m in infer_model.modules() if isinstance(m, InferAttn)]
    capacity = SEQ_LEN + 4
    for attn_mod in attn_modules:
        _alloc_kv(attn_mod, config, capacity=capacity)

    all_layer_idxs = list(range(config.num_hidden_layers))
    cst = _alloc_conv_state_tables(config, all_layer_idxs)
    seq_slots = torch.tensor([0])

    input_ids = torch.randint(0, config.vocab_size, (SEQ_LEN,))

    # Reference: logits for all positions, take last
    with torch.no_grad():
        y_ref = ref_model(input_ids.unsqueeze(0))[0, -1, :]   # (vocab_size,)

    # ── infer prefill (returns only last-token logits) ──
    seq = _fresh_seq(SEQ_LEN)
    _ctx_prefill(seq, SEQ_LEN, conv_state_tables=cst)
    with torch.no_grad():
        y_infer_prefill = infer_model.compute_logits(infer_model(input_ids))[-1:]   # (1, vocab_size)
    reset_context()

    _result("infer prefill logits == ref", y_ref, y_infer_prefill.squeeze(0))

    # ── infer decode: prefill on [0..T-2], decode on [T-1] ──
    print("\n── Full Model (decode) ──")
    # Reset KV caches and conv state tables
    for attn_mod in attn_modules:
        _alloc_kv(attn_mod, config, capacity=capacity)
    cst = _alloc_conv_state_tables(config, all_layer_idxs)

    seq_d = _fresh_seq(prefill_len)
    _ctx_prefill(seq_d, prefill_len, conv_state_tables=cst)
    with torch.no_grad():
        _ = infer_model(input_ids[:prefill_len])
    reset_context()

    seq_d.num_cached_tokens = prefill_len
    seq_d.append_token(input_ids[prefill_len].item())
    decode_slot = seq_d.block_table[-1] * block_size + seq_d.last_block_num_tokens - 1
    num_blocks_padded = (SEQ_LEN + block_size - 1) // block_size
    block_tables = torch.tensor(
        [seq_d.block_table + [-1] * (num_blocks_padded - len(seq_d.block_table))],
        dtype=torch.int32,
    )
    _ctx_decode(seq_d, decode_slot, SEQ_LEN, block_tables,
                conv_state_tables=cst, seq_slots=seq_slots)
    with torch.no_grad():
        y_infer_decode = infer_model.compute_logits(infer_model(input_ids[prefill_len:]))   # (1, vocab_size)
    reset_context()

    _result("infer decode logits == ref", y_ref, y_infer_decode.squeeze(0))


# ─── entry point ─────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    config = TEST_CONFIG

    print(f"Verifying model: hidden_size={config.hidden_size}, "
          f"num_layers={config.num_hidden_layers}, head_dim={config.head_dim}, "
          f"seq_len={SEQ_LEN}")

    verify_rmsnorm(config)
    verify_mlp(config)
    verify_conv(config)
    verify_attention(config)
    verify_block(config)
    verify_model(config)

    print(f"\n{'='*50}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
