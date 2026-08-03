import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_fused_causal_conv_decode_matches_torch():
    from kernel.causal_conv1d_triton import causal_conv1d_decode_step

    torch.manual_seed(3)
    batch, channels, kernel = 7, 256, 4
    slots = torch.randperm(batch + 3, device="cuda")[:batch].long()
    x = torch.randn(batch, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, kernel, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(batch + 3, channels, kernel - 1, device="cuda", dtype=torch.bfloat16)
    expected_state = state.clone()
    old = expected_state[slots].clone()
    expected = (old * weight[:, :-1]).sum(2) + x * weight[:, -1]
    expected_state[slots] = torch.cat((old[:, :, 1:], x.unsqueeze(2)), dim=2)

    actual = causal_conv1d_decode_step(x, weight, slots, state)
    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.02)
    torch.testing.assert_close(state, expected_state)


def test_wide_gated_delta_decode_tile_matches_torch():
    from kernel.gated_delta_triton import gated_delta_decode_step

    torch.manual_seed(4)
    batch, heads, dim = 2, 2, 64
    q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    gamma = torch.rand(batch, heads, device="cuda")
    beta = torch.rand_like(gamma)
    slots = torch.arange(batch, device="cuda")
    state = torch.randn(batch, heads, dim, dim, device="cuda")
    expected_state = state.clone()

    qn = torch.nn.functional.normalize(q.float(), dim=-1)
    kn = torch.nn.functional.normalize(k.float(), dim=-1)
    decayed = gamma[..., None, None] * expected_state.transpose(-1, -2)
    pred = torch.einsum("bhvk,bhk->bhv", decayed, kn)
    update = beta[..., None] * (v.float() - pred)
    new = decayed + update.unsqueeze(-1) * kn.unsqueeze(-2)
    expected = torch.einsum("bhvk,bhk->bhv", new, qn)

    actual = gated_delta_decode_step(
        q, k, v, gamma, beta, state, slots, block_v=64
    )
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(state, new.transpose(-1, -2), atol=2e-5, rtol=2e-5)


def test_inference_elementwise_fusions_are_exact():
    from kernel.inference_ops_triton import softcap_logits, squared_relu_gate

    torch.manual_seed(5)
    packed = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16)
    x, gate = packed.chunk(2, dim=-1)
    torch.testing.assert_close(squared_relu_gate(x, gate), gate * x.relu().square())

    logits = torch.randn(32, 50304, device="cuda", dtype=torch.bfloat16)
    expected = 15.0 * logits * (logits.square() + 225.0).rsqrt()
    torch.testing.assert_close(softcap_logits(logits), expected)


@pytest.mark.parametrize("window", [None, 64])
def test_grouped_packed_polar_matches_per_sequence(window):
    from kernel.polar_triton import polar_attention_fwd, polar_attention_packed_fwd

    torch.manual_seed(6)
    lengths, heads, dim = [33, 65, 129], 2, 128
    total = sum(lengths)
    q = torch.randn(total, heads, dim, device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    params = dict(
        v_null=torch.randn(heads, dim, device="cuda"),
        null_base=torch.randn(heads, device="cuda"),
        null_slope_raw=torch.randn(heads, device="cuda"),
        len_gain_raw=torch.randn(heads, device="cuda"),
        mag_beta_raw=torch.randn(heads, device="cuda"),
    )
    seq_starts, query_starts, seq_lens = [], [], []
    packed_start = 0
    for length in lengths:
        for query_start in range(0, length, 128):
            seq_starts.append(packed_start)
            query_starts.append(query_start)
            seq_lens.append(length)
        packed_start += length
    tile_map = [torch.tensor(x, device="cuda", dtype=torch.int32)
                for x in (seq_starts, query_starts, seq_lens)]
    actual_c, actual_mag = polar_attention_packed_fwd(
        q, k, v, *tile_map, window=window, **params)

    expected_c, expected_mag, packed_start = [], [], 0
    for length in lengths:
        sl = slice(packed_start, packed_start + length)
        n_keys = torch.arange(1, length + 1, device="cuda", dtype=torch.float32)
        c, mag = polar_attention_fwd(
            q[sl].transpose(0, 1)[None].contiguous(),
            k[sl].transpose(0, 1)[None].contiguous(),
            v[sl].transpose(0, 1)[None].contiguous(),
            n_keys, is_causal=True, window=window, **params)
        expected_c.append(c[0].transpose(0, 1))
        expected_mag.append(mag[0].transpose(0, 1))
        packed_start += length
    torch.testing.assert_close(actual_c, torch.cat(expected_c))
    torch.testing.assert_close(actual_mag, torch.cat(expected_mag))


@pytest.mark.parametrize("gated,heads", [(False, 2), (True, 8)])
def test_fused_projection_head_rms_matches_cublas(gated, heads):
    from kernel.inference_ops_triton import linear_head_rms

    torch.manual_seed(7)
    rows, hidden, head_dim = 33, 1024, 128
    width = head_dim * (2 if gated else 1)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(heads * width, hidden, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(heads * width, device="cuda", dtype=torch.bfloat16)
    projected = torch.nn.functional.linear(x, weight, bias).view(rows, heads, width)
    if gated:
        q, gate = projected.split(head_dim, dim=-1)
        expected = torch.cat((torch.nn.functional.rms_norm(q, (head_dim,)), gate), dim=-1)
    else:
        expected = torch.nn.functional.rms_norm(projected, (head_dim,))
    actual = linear_head_rms(
        x, weight, bias, num_heads=heads, head_dim=head_dim, gated=gated)
    torch.testing.assert_close(actual.view_as(expected), expected, atol=0.02, rtol=0.02)


def test_packed_causal_conv_respects_boundaries_and_writes_states():
    from kernel.inference_ops_triton import packed_causal_conv1d
    from inference.models.atma import prefill_causal_conv1d
    from types import SimpleNamespace

    torch.manual_seed(8)
    lengths, channels, kernel = [1, 3, 7, 17], 257, 4
    total = sum(lengths)
    x = torch.randn(total, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, kernel, device="cuda", dtype=torch.bfloat16)
    starts, ends, slots, packed_start = [], [], [], 0
    for slot, length in enumerate(lengths):
        starts.extend([packed_start] * length)
        ends.extend([packed_start + length] * length)
        slots.extend([slot] * length)
        packed_start += length
    maps = [torch.tensor(values, device="cuda", dtype=torch.int32)
            for values in (starts, ends, slots)]
    actual_state = torch.zeros(len(lengths), channels, kernel - 1,
                               device="cuda", dtype=torch.bfloat16)
    actual = packed_causal_conv1d(x, weight, actual_state, *maps)

    expected_state = {"x": torch.zeros_like(actual_state)}
    expected, packed_start = [], 0
    for slot, length in enumerate(lengths):
        seq = SimpleNamespace(num_cached_tokens=0, seq_slot=slot)
        expected.append(prefill_causal_conv1d(
            "x", seq, x[packed_start:packed_start + length], weight, None, expected_state))
        packed_start += length
    torch.testing.assert_close(actual, torch.cat(expected), atol=0.04, rtol=0.02)
    torch.testing.assert_close(actual_state, expected_state["x"])
