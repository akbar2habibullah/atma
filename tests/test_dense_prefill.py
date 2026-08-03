from types import SimpleNamespace

import torch

from inference.engine.model_runner import ModelRunner
from inference.engine.sequence import Sequence
from inference.models.atma import AtmaAttention, prefill_causal_conv1d, prefill_causal_conv1d_dense
from inference.utils.context import get_context, reset_context


def test_dense_conv_matches_sequence_prefill_and_final_states():
    torch.manual_seed(0)
    batch, seqlen, channels, kernel = 3, 7, 8, 4
    x = torch.randn(batch, seqlen, channels)
    weight = torch.randn(channels, kernel)
    slots = torch.arange(batch)

    scalar_table = {"x": torch.zeros(batch, channels, kernel - 1)}
    dense_table = {"x": torch.zeros_like(scalar_table["x"])}
    seqs = [SimpleNamespace(num_cached_tokens=0, seq_slot=i) for i in range(batch)]
    expected = torch.stack([
        prefill_causal_conv1d("x", seqs[i], x[i], weight, None, scalar_table)
        for i in range(batch)
    ])
    actual = prefill_causal_conv1d_dense("x", slots, x, weight, None, dense_table)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(dense_table["x"], scalar_table["x"])


def _runner(max_tokens=128):
    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = 16
    runner.device = torch.device("cpu")
    runner.config = SimpleNamespace(max_num_batched_tokens=max_tokens)
    runner.conv_state_tables = {}
    return runner


def _seq(length, slot, scheduled=None, cached=0):
    seq = Sequence(list(range(length)))
    seq.seq_slot = slot
    seq.num_cached_tokens = cached
    seq.num_scheduled_tokens = length - cached if scheduled is None else scheduled
    seq.block_table = [slot]
    return seq


def test_dense_prefill_router_accepts_only_fresh_complete_equal_batches():
    runner = _runner()
    seqs = [_seq(8, 0), _seq(8, 1)]
    runner.prepare_prefill(seqs)
    ctx = get_context()
    assert ctx.dense_prefill
    assert (ctx.dense_batch_size, ctx.dense_seq_len) == (2, 8)
    torch.testing.assert_close(ctx.seq_slots, torch.tensor([0, 1]))
    reset_context()

    unsafe = [
        [_seq(8, 0)],                              # batch of one
        [_seq(8, 0), _seq(7, 1)],                 # unequal lengths
        [_seq(8, 0, cached=1), _seq(8, 1)],       # cached prefix
        [_seq(8, 0, scheduled=4), _seq(8, 1)],    # chunked/incomplete
    ]
    for seq_batch in unsafe:
        runner.prepare_prefill(seq_batch)
        assert not get_context().dense_prefill
        reset_context()


def test_grouped_polar_router_and_tile_map_on_cuda():
    if not torch.cuda.is_available():
        return
    runner = _runner(max_tokens=512)
    runner.device = torch.device("cuda")
    seqs = [_seq(33, 0), _seq(65, 1), _seq(129, 2)]
    next_block = 0
    for seq in seqs:
        count = (seq.num_tokens + runner.block_size - 1) // runner.block_size
        seq.block_table = list(range(next_block, next_block + count))
        next_block += count
    runner.prepare_prefill(seqs)
    ctx = get_context()
    assert ctx.grouped_polar_prefill
    assert not ctx.dense_prefill
    torch.testing.assert_close(
        ctx.polar_tile_seq_starts.cpu(), torch.tensor([0, 33, 98, 98], dtype=torch.int32))
    torch.testing.assert_close(
        ctx.polar_tile_q_starts.cpu(), torch.tensor([0, 0, 0, 128], dtype=torch.int32))
    torch.testing.assert_close(
        ctx.polar_tile_seq_lens.cpu(), torch.tensor([33, 65, 129, 129], dtype=torch.int32))
    assert ctx.token_seq_starts.numel() == sum(seq.num_tokens for seq in seqs)
    assert ctx.token_seq_ends[-1].item() == sum(seq.num_tokens for seq in seqs)
    assert ctx.token_seq_slots[-1].item() == seqs[-1].seq_slot
    reset_context()


def test_dense_attention_matches_packed_outputs_kv_and_memory_state():
    torch.manual_seed(1)
    batch, seqlen, dim, head_dim = 2, 5, 32, 16
    layer = AtmaAttention(
        2, dim, head_dim=head_dim, num_kv_heads=1, kernel_size=4,
        mem_enabled=True, mem_chunk=4, mem_kernel="torch",
    )
    for parameter in layer.parameters():
        parameter.data.normal_(0, 0.02)
    layer.attn.k_cache = torch.zeros(batch, 8, 1, head_dim)
    layer.attn.v_cache = torch.zeros_like(layer.attn.k_cache)
    x = torch.randn(batch * seqlen, dim)
    seqs = [SimpleNamespace(num_cached_tokens=0, seq_slot=i, block_table=[i])
            for i in range(batch)]
    slots = torch.arange(batch)
    slot_mapping = torch.tensor(list(range(seqlen)) + list(range(8, 8 + seqlen)),
                                dtype=torch.int32)

    def run(dense):
        tables = {
            "attn_2_q": torch.zeros(batch, dim, 3),
            "attn_2_k": torch.zeros(batch, head_dim, 3),
            "attn_2_v": torch.zeros(batch, head_dim, 3),
            "mem_2": torch.zeros(batch, 2, head_dim, head_dim),
        }
        layer.attn.k_cache.zero_()
        layer.attn.v_cache.zero_()
        from inference.utils.context import set_context
        set_context(
            True, seqlens_q=[seqlen] * batch, slot_mapping=slot_mapping,
            conv_state_tables=tables, seq_slots=slots,
            dense_prefill=dense, dense_batch_size=batch if dense else 0,
            dense_seq_len=seqlen if dense else 0,
        )
        get_context().seqs = seqs
        with torch.inference_mode():
            output = layer(x)
        cache = (layer.attn.k_cache.clone(), layer.attn.v_cache.clone())
        reset_context()
        return output, tables, cache

    expected, expected_states, expected_cache = run(False)
    actual, actual_states, actual_cache = run(True)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    for key in expected_states:
        torch.testing.assert_close(actual_states[key], expected_states[key], atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(actual_cache[0], expected_cache[0], atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(actual_cache[1], expected_cache[1], atol=2e-5, rtol=2e-4)


def test_grouped_heterogeneous_attention_matches_oracle_and_all_states_cuda():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(9)
    lengths, dim, head_dim = [3, 5], 256, 128
    batch, total = len(lengths), sum(lengths)
    layer = AtmaAttention(
        2, dim, head_dim=head_dim, num_kv_heads=1, kernel_size=4,
        window=4, mem_enabled=True, mem_chunk=4, mem_kernel="torch",
    ).cuda().bfloat16()
    for parameter in layer.parameters():
        parameter.data.normal_(0, 0.02)
    layer.attn.k_cache = torch.zeros(batch, 8, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    layer.attn.v_cache = torch.zeros_like(layer.attn.k_cache)
    x = torch.randn(total, dim, device="cuda", dtype=torch.bfloat16)
    seqs = [SimpleNamespace(num_cached_tokens=0, seq_slot=i, block_table=[i])
            for i in range(batch)]
    slot_mapping = torch.tensor(list(range(3)) + list(range(8, 13)),
                                device="cuda", dtype=torch.int32)

    def run(grouped):
        tables = {
            "attn_2_q": torch.zeros(batch, dim, 3, device="cuda", dtype=torch.bfloat16),
            "attn_2_k": torch.zeros(batch, head_dim, 3, device="cuda", dtype=torch.bfloat16),
            "attn_2_v": torch.zeros(batch, head_dim, 3, device="cuda", dtype=torch.bfloat16),
            "mem_2": torch.zeros(batch, 2, head_dim, head_dim, device="cuda"),
        }
        layer.attn.k_cache.zero_()
        layer.attn.v_cache.zero_()
        kwargs = {}
        if grouped:
            values = [torch.tensor(v, device="cuda", dtype=torch.int32) for v in (
                [0, 3], [0, 0], lengths,
                [0] * 3 + [3] * 5, [3] * 3 + [8] * 5, [0] * 3 + [1] * 5,
            )]
            kwargs = dict(
                grouped_polar_prefill=True,
                polar_tile_seq_starts=values[0], polar_tile_q_starts=values[1],
                polar_tile_seq_lens=values[2], token_seq_starts=values[3],
                token_seq_ends=values[4], token_seq_slots=values[5])
        from inference.utils.context import set_context
        set_context(True, seqlens_q=lengths, slot_mapping=slot_mapping,
                    conv_state_tables=tables, **kwargs)
        get_context().seqs = seqs
        with torch.inference_mode():
            output = layer(x)
        cache = (layer.attn.k_cache.clone(), layer.attn.v_cache.clone())
        reset_context()
        return output, tables, cache

    expected, expected_states, expected_cache = run(False)
    actual, actual_states, actual_cache = run(True)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    for key in expected_states:
        torch.testing.assert_close(actual_states[key], expected_states[key], atol=0.02, rtol=0.02)
    torch.testing.assert_close(actual_cache[0], expected_cache[0])
    torch.testing.assert_close(actual_cache[1], expected_cache[1])
