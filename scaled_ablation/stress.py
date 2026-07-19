"""Post-training load-to-failure diagnostics for Atma checkpoints.

The passive probe is deliberately streaming: hooks reduce activations to scalar or
per-head moments on device and retain no token-sized tensors after a forward.  The
optional modal pass estimates local block gains with randomized finite differences;
it is a secant-gain diagnostic, not an exact singular-value computation.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch


_REDUCE_CHUNK_ELEMENTS = 8 * 1024 * 1024  # at most ~32 MiB temporary fp32 storage


def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class TensorMoments:
    """Mergeable scalar moments reduced over every element of each tensor."""

    def __init__(self):
        self.count = 0
        self.finite_count = None
        self.sum = None
        self.sumsq = None
        self.absmax = None

    def update(self, value):
        tensor = _first_tensor(value)
        if tensor is None or tensor.numel() == 0:
            return
        flat = tensor.detach().reshape(-1)
        total = sumsq = absmax = finite_count = None
        for start in range(0, flat.numel(), _REDUCE_CHUNK_ELEMENTS):
            x = flat[start:start + _REDUCE_CHUNK_ELEMENTS].float()
            finite = torch.isfinite(x)
            safe = torch.where(finite, x, torch.zeros_like(x))
            chunk_sum = safe.sum()
            chunk_sumsq = safe.square().sum()
            chunk_absmax = safe.abs().amax()
            chunk_finite = finite.sum()
            if total is None:
                total, sumsq = chunk_sum, chunk_sumsq
                absmax, finite_count = chunk_absmax, chunk_finite
            else:
                total = total + chunk_sum
                sumsq = sumsq + chunk_sumsq
                absmax = torch.maximum(absmax, chunk_absmax)
                finite_count = finite_count + chunk_finite
        if self.sum is None:
            self.sum, self.sumsq, self.absmax = total, sumsq, absmax
            self.finite_count = finite_count
        else:
            self.sum = self.sum + total
            self.sumsq = self.sumsq + sumsq
            self.absmax = torch.maximum(self.absmax, absmax)
            self.finite_count = self.finite_count + finite_count
        self.count += flat.numel()

    def merge(self, other: "TensorMoments"):
        if not other.count:
            return
        if self.sum is None:
            self.sum = other.sum.clone()
            self.sumsq = other.sumsq.clone()
            self.absmax = other.absmax.clone()
            self.finite_count = other.finite_count.clone()
        else:
            self.sum = self.sum + other.sum
            self.sumsq = self.sumsq + other.sumsq
            self.absmax = torch.maximum(self.absmax, other.absmax)
            self.finite_count = self.finite_count + other.finite_count
        self.count += other.count

    def snapshot(self):
        if not self.count:
            return None
        finite_count = int(self.finite_count.item())
        if not finite_count:
            return {
                "count": self.count,
                "finite_count": 0,
                "nonfinite_pct": 100.0,
                "mean": None,
                "std": None,
                "rms": None,
                "absmax": None,
            }
        mean = self.sum / finite_count
        second = self.sumsq / finite_count
        std = (second - mean.square()).clamp_min(0).sqrt()
        return {
            "count": self.count,
            "finite_count": finite_count,
            "nonfinite_pct": 100.0 * (self.count - finite_count) / self.count,
            "mean": mean.item(),
            "std": std.item(),
            "rms": second.sqrt().item(),
            "absmax": self.absmax.item(),
        }


class PerHeadMoments:
    """Mergeable per-head moments with an optional near-one saturation count."""

    def __init__(self):
        self.count = 0
        self.finite_count = None
        self.sum = None
        self.sumsq = None
        self.near_one = None

    def update(self, value: torch.Tensor, head_dim: int):
        x = value.detach().float().movedim(head_dim, -1)
        flat = x.reshape(-1, x.shape[-1])
        finite = torch.isfinite(flat)
        safe = torch.where(finite, flat, torch.zeros_like(flat))
        finite_count = finite.sum(0)
        total = safe.sum(0)
        sumsq = safe.square().sum(0)
        near_one = (flat > 0.99).sum(0)
        if self.sum is None:
            self.sum, self.sumsq, self.near_one = total, sumsq, near_one
            self.finite_count = finite_count
        else:
            self.sum = self.sum + total
            self.sumsq = self.sumsq + sumsq
            self.near_one = self.near_one + near_one
            self.finite_count = self.finite_count + finite_count
        self.count += flat.shape[0]

    def merge(self, other: "PerHeadMoments"):
        if not other.count:
            return
        if self.sum is None:
            self.sum = other.sum.clone()
            self.sumsq = other.sumsq.clone()
            self.near_one = other.near_one.clone()
            self.finite_count = other.finite_count.clone()
        else:
            self.sum = self.sum + other.sum
            self.sumsq = self.sumsq + other.sumsq
            self.near_one = self.near_one + other.near_one
            self.finite_count = self.finite_count + other.finite_count
        self.count += other.count

    def snapshot(self):
        if not self.count:
            return None
        denom = self.finite_count.clamp_min(1)
        mean = self.sum / denom
        second = self.sumsq / denom
        std = (second - mean.square()).clamp_min(0).sqrt()
        near_one = 100.0 * self.near_one.float() / self.count
        nonfinite = 100.0 * (self.count - self.finite_count).float() / self.count
        return {
            "count_per_head": self.count,
            "finite_count_per_head": self.finite_count.cpu().tolist(),
            "nonfinite_pct": nonfinite.float().mean().item(),
            "per_head_nonfinite_pct": nonfinite.cpu().tolist(),
            "mean": mean.mean().item(),
            "rms": second.mean().sqrt().item(),
            "per_head_mean": mean.cpu().tolist(),
            "per_head_std": std.cpu().tolist(),
            "per_head_near_one_pct": near_one.cpu().tolist(),
            "near_one_pct": near_one.mean().item(),
        }


class StressProbe:
    """Forward-hook instrumentation for one model and one evaluation length.

    Updates are transactional per document.  If a document OOMs after a partial
    forward, ``discard_sample`` prevents its partial hook data from contaminating the
    reported population.
    """

    def __init__(self, model):
        self.model = model
        self.handles = []
        self._sample_scalar = None
        self._sample_head = None
        self.scalar = defaultdict(TensorMoments)
        self.head = defaultdict(PerHeadMoments)
        self.polar_layers = []
        self._register()

    def _scalar_hook(self, path, *, pre=False):
        def hook(_module, inputs, output=None):
            if self._sample_scalar is None:
                return
            value = inputs[0] if pre else output
            self._sample_scalar[path].update(value)
        return hook

    def _memory_scalar_gate_hook(self, block_index, name, bias):
        def hook(_module, _inputs, output):
            if self._sample_head is None:
                return
            value = torch.sigmoid(output.float() + bias)
            self._sample_head[(block_index, name)].update(value, head_dim=-1)
        return hook

    def _channel_gate_hook(self, block_index, name, heads, channels, *, split=False):
        def hook(_module, _inputs, output):
            if self._sample_head is None:
                return
            value = output.view(*output.shape[:-1], heads, -1)
            if split:
                value = value[..., channels:]
            value = torch.sigmoid(value)
            self._sample_head[(block_index, name)].update(value, head_dim=-2)
        return hook

    def _register(self):
        for index, block in enumerate(self.model.blocks):
            self.handles.append(block.register_forward_pre_hook(
                self._scalar_hook((index, "residual_input"), pre=True)))
            self.handles.append(block.register_forward_hook(
                self._scalar_hook((index, "residual_output"))))
            self.handles.append(block.attn.register_forward_hook(
                self._scalar_hook((index, "mixer_total"))))
            self.handles.append(block.mlp.register_forward_hook(
                self._scalar_hook((index, "mlp"))))

            attn = block.attn
            if hasattr(attn, "proj"):
                self.handles.append(attn.proj.register_forward_hook(
                    self._scalar_hook((index, "attention_projected"))))
            if hasattr(attn, "q") and hasattr(attn, "num_heads") and hasattr(attn, "head_dim"):
                self.handles.append(attn.q.register_forward_hook(self._channel_gate_hook(
                    index, "attention_output_gate", attn.num_heads, attn.head_dim, split=True)))
            if hasattr(attn, "mu_proj"):
                self.polar_layers.append(index)
                self.handles.append(attn.mu_proj.register_forward_hook(
                    self._scalar_hook((index, "polar_count"))))
            memory = getattr(attn, "mem", None)
            if memory is not None:
                self.handles.append(memory.w_gamma.register_forward_hook(
                    self._memory_scalar_gate_hook(index, "memory_gamma", memory.gamma_bias)))
                self.handles.append(memory.w_beta.register_forward_hook(
                    self._memory_scalar_gate_hook(index, "memory_beta", memory.beta_bias)))
                self.handles.append(memory.gate.register_forward_hook(self._channel_gate_hook(
                    index, "memory_output_gate", memory.H, memory.dk)))
                self.handles.append(memory.register_forward_hook(
                    self._scalar_hook((index, "memory"))))

    def begin_sample(self):
        self._sample_scalar = defaultdict(TensorMoments)
        self._sample_head = defaultdict(PerHeadMoments)

    def consume_polar(self, records):
        if self._sample_head is None:
            return
        if len(records) != len(self.polar_layers):
            raise RuntimeError(
                f"polar probe emitted {len(records)} records for "
                f"{len(self.polar_layers)} PolarAttention layers"
            )
        for index, record in zip(self.polar_layers, records):
            self._sample_head[(index, "polar_n_eff")].update(record["n_eff"], head_dim=1)
            self._sample_head[(index, "polar_mag")].update(record["mag"], head_dim=1)
            self._sample_head[(index, "polar_w_null")].update(record["w_null"], head_dim=1)

    def commit_sample(self):
        if self._sample_scalar is None:
            return
        for key, moments in self._sample_scalar.items():
            self.scalar[key].merge(moments)
        for key, moments in self._sample_head.items():
            self.head[key].merge(moments)
        self._sample_scalar = self._sample_head = None

    def discard_sample(self):
        self._sample_scalar = self._sample_head = None

    def snapshot(self):
        result = {str(i): {} for i in range(len(self.model.blocks))}
        for (index, name), moments in self.scalar.items():
            result[str(index)][name] = moments.snapshot()
        for (index, name), moments in self.head.items():
            result[str(index)][name] = moments.snapshot()

        for block in result.values():
            before = block.get("residual_input")
            after = block.get("residual_output")
            if before and after and before["rms"]:
                block["residual_rms_gain"] = after["rms"] / before["rms"]
            for name in ("mixer_total", "mlp", "attention_projected", "polar_count", "memory"):
                contribution = block.get(name)
                if before and contribution and before["rms"]:
                    contribution["rms_over_residual_input"] = contribution["rms"] / before["rms"]
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _iter_rms_signals(length_result):
    for block_index, block in length_result.get("blocks", {}).items():
        for name, stats in block.items():
            if isinstance(stats, dict) and stats.get("rms") is not None:
                yield f"block.{block_index}.{name}.rms", stats["rms"]
        n_eff = block.get("polar_n_eff")
        if n_eff and n_eff.get("mean") is not None:
            yield f"block.{block_index}.polar_n_eff.mean", n_eff["mean"]


def _iter_nonfinite_signals(length_result):
    for block_index, block in length_result.get("blocks", {}).items():
        for name, stats in block.items():
            if isinstance(stats, dict) and stats.get("nonfinite_pct") is not None:
                yield f"block.{block_index}.{name}.nonfinite_pct", stats["nonfinite_pct"]


def build_stress_summary(by_length: dict, train_length: int, yield_ratio: float = 1.25):
    """Rank the first operating-envelope departure relative to the shortest run."""
    available = sorted(int(length) for length, row in by_length.items() if row.get("completed", 0))
    if not available:
        return {"baseline_length": None, "train_length": train_length, "components": []}
    baseline_length = available[0]
    baseline = dict(_iter_rms_signals(by_length[str(baseline_length)]))
    components = []
    for path, base in baseline.items():
        if not math.isfinite(base) or abs(base) < 1e-12:
            continue
        curve = []
        first_yield = None
        max_log2_drift = 0.0
        for length in available:
            signals = dict(_iter_rms_signals(by_length[str(length)]))
            value = signals.get(path)
            if value is None or not math.isfinite(value):
                continue
            ratio = value / base
            log2_drift = abs(math.log2(max(ratio, 1e-12)))
            max_log2_drift = max(max_log2_drift, log2_drift)
            curve.append({"length": length, "value": value, "ratio_to_baseline": ratio})
            if length != baseline_length and first_yield is None and (
                    ratio > yield_ratio or ratio < 1.0 / yield_ratio):
                first_yield = length
        components.append({
            "path": path,
            "baseline": base,
            "curve": curve,
            "first_yield_length": first_yield,
            "safety_factor": first_yield / train_length if first_yield is not None else None,
            "max_abs_log2_drift": max_log2_drift,
        })

    # Any NaN/Inf is a hard yield even when the finite subset's RMS remains stable.
    nonfinite_paths = set()
    for length in available:
        nonfinite_paths.update(dict(_iter_nonfinite_signals(by_length[str(length)])))
    for path in sorted(nonfinite_paths):
        curve, first_yield, maximum = [], None, 0.0
        for length in available:
            value = dict(_iter_nonfinite_signals(by_length[str(length)])).get(path, 0.0)
            curve.append({"length": length, "value": value, "ratio_to_baseline": None})
            maximum = max(maximum, value)
            if value > 0.0 and first_yield is None:
                first_yield = length
        if first_yield is not None:
            components.append({
                "path": path,
                "baseline": curve[0]["value"],
                "curve": curve,
                "first_yield_length": first_yield,
                "safety_factor": first_yield / train_length,
                "max_abs_log2_drift": 0.0,
                "max_nonfinite_pct": maximum,
            })
    components.sort(key=lambda row: (
        row["first_yield_length"] is None,
        row["first_yield_length"] or math.inf,
        0 if row.get("max_nonfinite_pct", 0.0) > 0.0 else 1,
        -row["max_abs_log2_drift"],
    ))
    return {
        "baseline_length": baseline_length,
        "train_length": train_length,
        "yield_ratio": yield_ratio,
        "components": components,
    }


def _block_map(block, x):
    attention, _ = block.attn(block.norm1(x))
    y = x + attention
    return y + block.mlp(block.norm2(y))


@torch.no_grad()
def randomized_block_gains(model, inputs, *, samples: int = 2,
                           perturbation: float = 0.02, seed: int = 1234):
    """Estimate local secant gains for every block with isotropic perturbations.

    The reported maximum is a lower bound on the worst local gain over the sampled
    directions.  ``actual_perturbation_rms`` accounts for bf16/fp16 rounding.
    """
    x = model.embed(inputs)
    result = {}
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    for index, block in enumerate(model.blocks):
        baseline = _block_map(block, x)
        x_rms = x.float().square().mean().sqrt().clamp_min(1e-12)
        gains = []
        relative_effects = []
        actual_scales = []
        for _ in range(samples):
            direction = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
            direction_rms = direction.float().square().mean().sqrt().clamp_min(1e-12)
            candidate = x + direction * ((perturbation * x_rms) / direction_rms).to(x.dtype)
            delta = candidate - x
            delta_rms = delta.float().square().mean().sqrt().clamp_min(1e-12)
            perturbed = _block_map(block, candidate)
            response_rms = (perturbed - baseline).float().square().mean().sqrt()
            gains.append((response_rms / delta_rms).item())
            relative_effects.append((response_rms / baseline.float().square().mean().sqrt().clamp_min(1e-12)).item())
            actual_scales.append((delta_rms / x_rms).item())
            del direction, candidate, delta, perturbed
        result[str(index)] = {
            "random_secant_gain_mean": sum(gains) / len(gains),
            "random_secant_gain_max": max(gains),
            "relative_output_effect_mean": sum(relative_effects) / len(relative_effects),
            "actual_perturbation_rms_mean": sum(actual_scales) / len(actual_scales),
            "samples": samples,
        }
        x = baseline
    return result


@torch.no_grad()
def eval_stress(model, docs, lengths, device, loss_chunk: int, *, train_length: int,
                num_docs: int = 8, yield_ratio: float = 1.25,
                modal_lengths=(), modal_docs: int = 1, modal_samples: int = 2,
                perturbation: float = 0.02):
    """Run passive and optional randomized-modal diagnostics on coherent documents."""
    from eval import _blocks_forward, _chunked_loss
    from model import blocks as blocks_module

    by_length = {}
    selected_docs = docs[:num_docs]
    for length in lengths:
        probe = StressProbe(model)
        total, tokens, completed, ooms = 0.0, 0, 0, 0
        for doc in selected_docs:
            inputs = targets = hidden = None
            probe.begin_sample()
            try:
                buf = doc[: length + 1]
                inputs = buf[:-1].view(1, -1).to(device, torch.int32)
                targets = buf[1:].view(1, -1).to(device, torch.int64)
                blocks_module._PROBE = []
                hidden = _blocks_forward(model, inputs)
                probe.consume_polar(blocks_module._PROBE)
                loss_sum, count = _chunked_loss(model, hidden, targets, chunk=loss_chunk)
                total += loss_sum
                tokens += count
                completed += 1
                probe.commit_sample()
            except torch.cuda.OutOfMemoryError:
                ooms += 1
                probe.discard_sample()
            finally:
                blocks_module._PROBE = None
                del inputs, targets, hidden
                torch.cuda.empty_cache()
        by_length[str(length)] = {
            "loss_nats": total / tokens if tokens else None,
            "tokens": tokens,
            "completed": completed,
            "oom": ooms,
            "blocks": probe.snapshot(),
        }
        probe.close()
        print(f"  stress {length:>6,}: loss={by_length[str(length)]['loss_nats']} "
              f"completed={completed} oom={ooms}", flush=True)

    modal = {}
    modal_set = set(modal_lengths)
    for length in lengths:
        if length not in modal_set:
            continue
        runs, ooms = [], 0
        for doc_index, doc in enumerate(docs[:modal_docs]):
            inputs = None
            try:
                inputs = doc[:length].view(1, -1).to(device, torch.int32)
                runs.append(randomized_block_gains(
                    model, inputs, samples=modal_samples, perturbation=perturbation,
                    seed=1234 + doc_index,
                ))
            except torch.cuda.OutOfMemoryError:
                ooms += 1
            finally:
                del inputs
                torch.cuda.empty_cache()
        aggregate = {}
        for index in range(len(model.blocks)):
            rows = [run[str(index)] for run in runs]
            if not rows:
                continue
            aggregate[str(index)] = {
                key: sum(row[key] for row in rows) / len(rows)
                for key in ("random_secant_gain_mean", "random_secant_gain_max",
                            "relative_output_effect_mean", "actual_perturbation_rms_mean")
            }
            aggregate[str(index)]["documents"] = len(rows)
            aggregate[str(index)]["samples_per_document"] = modal_samples
        ranking = sorted(
            ({"block": int(index), "random_secant_gain_max": row["random_secant_gain_max"]}
             for index, row in aggregate.items()),
            key=lambda row: -row["random_secant_gain_max"],
        )
        modal[str(length)] = {
            "blocks": aggregate,
            "ranking": ranking,
            "completed": len(runs),
            "oom": ooms,
        }
        print(f"  modal  {length:>6,}: completed={len(runs)} oom={ooms}", flush=True)

    summary = build_stress_summary(by_length, train_length, yield_ratio)
    yielded = [row for row in summary["components"] if row["first_yield_length"] is not None]
    if yielded:
        print("  first-yield components:", flush=True)
        for row in yielded[:10]:
            print(f"    {row['first_yield_length']:>7,}  SF={row['safety_factor']:.2f}  "
                  f"{row['path']}", flush=True)
    else:
        print(f"  no component crossed the {yield_ratio:.3g}x operating envelope", flush=True)

    return {
        "by_length": by_length,
        "summary": summary,
        "modal": modal,
        "settings": {
            "num_docs": len(selected_docs),
            "yield_ratio": yield_ratio,
            "modal_lengths": sorted(modal_set),
            "modal_docs": modal_docs,
            "modal_samples": modal_samples,
            "perturbation": perturbation,
            "modal_method": "randomized_finite_difference_secant_gain",
        },
    }
