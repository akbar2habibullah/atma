import json
import pytest
import torch

from benchmarks.model import EvalModel
from model.config import AtmaConfig


@pytest.mark.parametrize("attn_type", ["nope", "rope"])
def test_softmax_fork_has_checkpoint_exact_layout(attn_type):
    from train.model import Model
    from baseline_inference.softmax_model import SoftmaxLM

    cfg = AtmaConfig(
        attn_type=attn_type, num_hidden_layers=4, hidden_size=256, head_dim=64
    )
    result = SoftmaxLM(cfg).load_state_dict(Model(cfg).state_dict(), strict=False)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


@pytest.mark.parametrize("arch", ["raven_native", "atma_raven", "atma_raven_titans"])
def test_raven_fork_has_checkpoint_exact_layout(arch):
    from raven_baseline.config_schema import RavenRunConfig
    from raven_baseline.model import create_model
    from baseline_inference.raven_model import RavenLM

    cfg = RavenRunConfig(
        arch_type=arch, hidden_size=256, num_heads=4, num_slots=32, topk=8
    ).to_dict()
    result = RavenLM(cfg).load_state_dict(create_model(cfg).state_dict(), strict=False)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_eval_adapter_routes_supported_baselines(tmp_path):
    from raven_baseline.config_schema import RavenRunConfig

    for name, cfg in (
        ("polar", {"attn_type": "polar"}),
        ("nope", {"attn_type": "nope"}),
        ("rope", {"attn_type": "rope"}),
        ("raven", RavenRunConfig(arch_type="atma_raven").to_dict()),
    ):
        path = tmp_path / name
        path.mkdir()
        (path / "config.json").write_text(json.dumps(cfg))
        (path / "weights.pt").touch()
        model = EvalModel(str(path), strict=True, quiet=True)
        assert model.wip == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_paged_softmax_decode_matches_sdpa():
    from baseline_inference.softmax_triton import paged_softmax_decode

    B, S, H, KV, D, BS = 2, 65, 8, 2, 128, 32
    NB = (S + BS - 1) // BS
    torch.manual_seed(17)
    k = torch.randn(B * NB, BS, KV, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
    bt = torch.arange(B * NB, device="cuda", dtype=torch.int32).view(B, NB)
    lens = torch.full((B,), S, device="cuda", dtype=torch.int32)
    actual = paged_softmax_decode(q, k, v, bt, lens, scale=D**-0.5)
    expected = []
    for b in range(B):
        kb = (
            k[bt[b].long()]
            .reshape(-1, KV, D)[:S]
            .repeat_interleave(H // KV, 1)
            .transpose(0, 1)
        )
        vb = (
            v[bt[b].long()]
            .reshape(-1, KV, D)[:S]
            .repeat_interleave(H // KV, 1)
            .transpose(0, 1)
        )
        expected.append(
            torch.nn.functional.scaled_dot_product_attention(
                q[b].unsqueeze(1), kb, vb, scale=D**-0.5
            ).squeeze(1)
        )
    torch.testing.assert_close(actual, torch.stack(expected), atol=0.01, rtol=0.02)
