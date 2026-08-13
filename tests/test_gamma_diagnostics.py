import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from gamma_diagnostics.clamp import (
    FORMAT_VERSION,
    apply_gamma_clamp,
    half_life_to_gamma,
    half_life_to_logit,
    load_clamp_spec,
)
from gamma_diagnostics.inspect_parameters import inspect_checkpoint
from gamma_diagnostics.selection import recommend


class Memory(nn.Module):
    def __init__(self, width=4, dim=3, gamma_bias=3.9):
        super().__init__()
        self.gamma_bias = gamma_bias
        self.w_gamma = nn.Linear(dim, width)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.mem = Memory()


class Block(nn.Module):
    def __init__(self, memory=False):
        super().__init__()
        if memory:
            self.attn = Attention()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block(), Block(memory=True)])


class GammaDiagnosticsTest(unittest.TestCase):
    def test_half_life_conversion_is_stable_and_exact_for_long_horizons(self):
        for half_life in (1, 256, 2_000_000):
            gamma = half_life_to_gamma(half_life)
            logit = half_life_to_logit(half_life)
            recovered = math.exp(-math.log1p(math.exp(-logit)))
            self.assertAlmostEqual(recovered, gamma, delta=gamma * 1e-10)
            self.assertAlmostEqual(gamma ** half_life, 0.5, delta=1e-9)

    def test_selective_runtime_cap_and_remove_restore_original_output(self):
        torch.manual_seed(7)
        model = TinyModel()
        memory = model.blocks[2].attn.mem
        with torch.no_grad():
            memory.w_gamma.weight.zero_()
            memory.w_gamma.bias.copy_(torch.tensor([12.0, 2.0, 12.0, -2.0]))
        x = torch.randn(2, 5, 3)
        original = memory.w_gamma(x)
        spec = {
            "format": FORMAT_VERSION,
            "targets": [{"layer": 2, "heads": [0], "max_half_life_tokens": 256}],
        }

        handle = apply_gamma_clamp(model, spec)
        capped_learned = memory.w_gamma(x)
        final_logit = capped_learned + memory.gamma_bias
        cap = half_life_to_logit(256)
        self.assertAlmostEqual(final_logit[..., 0].max().item(), cap, delta=1e-6)
        self.assertTrue(torch.equal(capped_learned[..., 1:], original[..., 1:].float()))
        self.assertEqual(handle.resolved_targets[0]["layer"], 2)
        self.assertEqual(handle.resolved_targets[0]["head"], 0)

        handle.remove()
        self.assertTrue(torch.equal(memory.w_gamma(x), original))

    def test_overlapping_targets_take_the_stricter_cap(self):
        model = TinyModel()
        spec = {
            "targets": [
                {"layer": 2, "heads": [1], "max_half_life_tokens": 512},
                {"layer": 2, "heads": [1], "max_half_life_tokens": 64},
            ]
        }
        handle = apply_gamma_clamp(model, spec)
        output = model.blocks[2].attn.mem.w_gamma(torch.zeros(1, 3))
        final = output[0, 1].item() + model.blocks[2].attn.mem.gamma_bias
        self.assertLessEqual(final, half_life_to_logit(64) + 1e-6)
        handle.remove()

    def test_spec_validation_and_json_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clamp.json"
            path.write_text(json.dumps({
                "format": FORMAT_VERSION,
                "targets": [{"layer": 2, "heads": [0], "max_gamma": 0.99}],
            }), encoding="utf-8")
            self.assertEqual(load_clamp_spec(path)["targets"][0]["max_gamma"], 0.99)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            apply_gamma_clamp(TinyModel(), {
                "targets": [{
                    "layer": 2, "heads": [0],
                    "max_gamma": 0.99, "max_half_life_tokens": 10,
                }]
            })
        with self.assertRaisesRegex(ValueError, "no Titans memory"):
            apply_gamma_clamp(TinyModel(), {
                "targets": [{"layer": 1, "heads": [0], "max_gamma": 0.99}]
            })

    def test_parameter_inspector_preserves_large_logit_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({
                "model": {
                    "blocks.2.attn.mem.w_gamma.weight": torch.zeros(2, 3),
                    "blocks.2.attn.mem.w_gamma.bias": torch.tensor([13.327, 0.0]),
                }
            }, root / "weights.pt")
            (root / "config.json").write_text(
                json.dumps({"attn_type": "nope", "mem_gamma_bias": 3.9}),
                encoding="utf-8",
            )
            rows = inspect_checkpoint("synthetic", root / "weights.pt", root / "config.json")
        outlier = rows[0]
        self.assertEqual((outlier["layer"], outlier["head"]), (2, 0))
        self.assertGreater(outlier["half_life_tokens"], 20_000_000)
        self.assertGreater(outlier["gamma_zero_input"], 0.9999999)
        self.assertLess(outlier["gamma_zero_input"], 1.0)

    def test_recommendation_requires_both_recovery_and_short_context_guardrail(self):
        baseline = {
            "clean": {"2048": {"loss_nats": 2.0}},
            "needle": {"by_distance": {"65536": {"ce_nats": 1.0}}},
        }
        good = {
            "clean": {"2048": {"loss_nats": 2.02}},
            "needle": {"by_distance": {"65536": {"ce_nats": 0.8}}},
        }
        bad_short = {
            "clean": {"2048": {"loss_nats": 2.10}},
            "needle": {"by_distance": {"65536": {"ce_nats": 0.7}}},
        }
        conditions = {
            "baseline": {"metrics": baseline},
            "good": {"metrics": good, "spec": {"label": "good"}},
            "bad-short": {"metrics": bad_short, "spec": {"label": "bad-short"}},
        }
        recommendation = recommend(conditions, [2048, 65536], 0.05)
        self.assertEqual(recommendation["condition"], "good")
        self.assertIsNone(recommend({
            "baseline": {"metrics": baseline},
            "tiny": {"metrics": {
                "clean": {"2048": {"loss_nats": 2.0}},
                "needle": {"by_distance": {"65536": {"ce_nats": 0.98}}},
            }, "spec": {}},
        }, [2048, 65536], 0.05))
        self.assertIsNone(recommend({
            "baseline": {"metrics": {"needle": baseline["needle"]}},
            "good": {"metrics": {"needle": good["needle"]}, "spec": {}},
        }, [2048, 65536], 0.05))


if __name__ == "__main__":
    unittest.main()
