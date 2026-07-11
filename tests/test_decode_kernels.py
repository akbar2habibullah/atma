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
