import torch
import torch.nn.functional as F

from train.model import (
    _WALL_GATE_BIAS_INIT,
    _WALL_GATE_LOG_MAX,
    CausalSelfAttention,
    _wall_log_decay,
)


def test_wall_log_decay_is_bounded_and_non_positive():
    logits = torch.tensor([-100.0, -6.0, 0.0, 6.0, 100.0])
    g = _wall_log_decay(logits)

    assert torch.isfinite(g).all()
    assert torch.all(g <= 0)
    assert torch.all(g >= -_WALL_GATE_LOG_MAX)


def test_wall_default_bias_starts_near_vanilla_attention():
    logits = torch.full((8,), _WALL_GATE_BIAS_INIT)
    g = _wall_log_decay(logits)
    retention = torch.exp(g)

    expected = torch.sigmoid(logits)
    torch.testing.assert_close(retention, expected, rtol=3e-3, atol=3e-3)
    assert retention.min() > 0.99


def test_wall_zero_bias_matches_blog_operating_point():
    g_hat = F.logsigmoid(torch.tensor(0.0))
    expected = -_WALL_GATE_LOG_MAX * (1.0 - torch.exp(g_hat / _WALL_GATE_LOG_MAX))

    g = _wall_log_decay(torch.tensor(0.0))

    torch.testing.assert_close(g, expected)
    torch.testing.assert_close(torch.exp(g), torch.tensor(0.620), rtol=2e-3, atol=2e-3)


def test_wall_module_zero_weight_emits_open_log_decay():
    attn = CausalSelfAttention(dim=32, head_dim=8, num_kv_heads=1, pos="wall")
    x = torch.randn(2, 4, 32)

    logits = attn.w_wall(x).view(2, 4, attn.num_heads, attn.head_dim) + attn.wall_gate_bias
    g = _wall_log_decay(logits)

    assert attn.wall_gate_bias == _WALL_GATE_BIAS_INIT
    assert torch.all(g <= 0)
    assert torch.all(g >= -_WALL_GATE_LOG_MAX)
    assert torch.exp(g).min() > 0.99
