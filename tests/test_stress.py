import pytest
import torch
from torch import nn

from scaled_ablation.stress import (
    PerHeadMoments,
    StressProbe,
    TensorMoments,
    build_stress_summary,
    randomized_block_gains,
)


class _ZeroMLP(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class _Memory(nn.Module):
    def __init__(self, dim=4, heads=2):
        super().__init__()
        self.w_gamma = nn.Linear(dim, heads)
        self.w_beta = nn.Linear(dim, heads)
        self.gate = nn.Linear(dim, dim)
        self.H = heads
        self.dk = dim // heads
        self.gamma_bias = 1.0
        self.beta_bias = 0.0

    def forward(self, x, *_):
        gamma = torch.sigmoid(self.w_gamma(x) + self.gamma_bias)
        beta = torch.sigmoid(self.w_beta(x) + self.beta_bias)
        gate = torch.sigmoid(self.gate(x))
        # Keep all three projections live, as in the real memory implementation.
        return 0.25 * x + 0.0 * (gamma.mean(-1, keepdim=True) + beta.mean(-1, keepdim=True) + gate)


class _Attention(nn.Module):
    def __init__(self, dim=4, polar=True, memory=True):
        super().__init__()
        self.proj = nn.Identity()
        if polar:
            self.mu_proj = nn.Linear(2, dim, bias=False)
            self.v_null = nn.Parameter(torch.zeros(2, 2))
        self.mem = _Memory(dim) if memory else None

    def forward(self, x):
        content = self.proj(x)
        count = self.mu_proj(torch.ones(*x.shape[:-1], 2)) if hasattr(self, "mu_proj") else 0
        memory = self.mem(x, None, None, None) if self.mem is not None else 0
        return content + count + memory, x.new_tensor(0.0)


class _Block(nn.Module):
    def __init__(self, dim=4, polar=True, memory=True):
        super().__init__()
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.attn = _Attention(dim, polar, memory)
        self.mlp = _ZeroMLP()

    def forward(self, x):
        attention, align = self.attn(self.norm1(x))
        y = x + attention
        y = y + self.mlp(self.norm2(y))
        return y, x.new_tensor(0.0), align


class _Model(nn.Module):
    def __init__(self, blocks=1, dim=4):
        super().__init__()
        self.embed = nn.Embedding(16, dim)
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(blocks)])


def test_tensor_and_per_head_moments_merge():
    left, right = TensorMoments(), TensorMoments()
    left.update(torch.tensor([1.0, 2.0]))
    right.update(torch.tensor([3.0]))
    left.merge(right)
    snap = left.snapshot()
    assert snap["mean"] == pytest.approx(2.0)
    assert snap["rms"] == pytest.approx((14.0 / 3.0) ** 0.5)
    assert snap["absmax"] == 3.0
    assert snap["nonfinite_pct"] == 0.0

    finite = TensorMoments()
    finite.update(torch.tensor([1.0, float("nan"), float("inf")]))
    assert finite.snapshot()["mean"] == pytest.approx(1.0)
    assert finite.snapshot()["nonfinite_pct"] == pytest.approx(200.0 / 3.0)

    heads = PerHeadMoments()
    heads.update(torch.tensor([[[0.5, 1.0], [0.7, 0.9]]]), head_dim=-1)
    hs = heads.snapshot()
    assert hs["per_head_mean"] == pytest.approx([0.6, 0.95])
    assert hs["per_head_near_one_pct"] == pytest.approx([0.0, 50.0])


def test_probe_discards_partial_samples_and_reports_channels():
    model = _Model()
    probe = StressProbe(model)
    x = torch.ones(1, 3, 4)

    probe.begin_sample()
    model.blocks[0](x)
    probe.discard_sample()
    assert probe.snapshot()["0"] == {}

    probe.begin_sample()
    model.blocks[0](x)
    probe.consume_polar([{
        "n_eff": torch.full((1, 2, 3), 2.0),
        "mag": torch.full((1, 2, 3), 0.5),
        "w_null": torch.full((1, 2, 3), 0.25),
    }])
    probe.commit_sample()
    snap = probe.snapshot()["0"]
    probe.close()

    assert snap["residual_input"]["rms"] == pytest.approx(1.0)
    assert snap["attention_projected"]["rms"] == pytest.approx(1.0)
    assert snap["memory"]["rms"] == pytest.approx(0.25)
    assert snap["polar_n_eff"]["mean"] == pytest.approx(2.0)
    assert len(snap["memory_gamma"]["per_head_mean"]) == 2


def test_summary_finds_first_yield_and_safety_factor():
    def row(rms, nonfinite=0.0):
        return {"completed": 1, "blocks": {"0": {
            "residual_output": {"rms": rms, "nonfinite_pct": nonfinite},
        }}}

    summary = build_stress_summary(
        {"2048": row(2.0), "4096": row(2.2), "8192": row(3.0, nonfinite=0.1)},
        train_length=2048,
        yield_ratio=1.25,
    )
    component = next(row for row in summary["components"] if row["path"].endswith(".rms"))
    assert component["first_yield_length"] == 8192
    assert component["safety_factor"] == pytest.approx(4.0)
    hard_failure = next(row for row in summary["components"] if row["path"].endswith("nonfinite_pct"))
    assert hard_failure["first_yield_length"] == 8192


def test_randomized_gain_recovers_identity_residual_gain():
    model = _Model()
    # Remove count and memory: attention(x)=x, so the complete block map is 2*x.
    model.blocks[0].attn = _Attention(polar=False, memory=False)
    inputs = torch.tensor([[1, 2, 3, 4]])
    result = randomized_block_gains(model, inputs, samples=2, perturbation=0.05)
    assert result["0"]["random_secant_gain_mean"] == pytest.approx(2.0, rel=1e-4)
