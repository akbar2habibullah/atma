"""Short controlled diagnostic for PyTorch-version and microbatch training drift.

The experiment preserves the production sequence length, hidden size, head dimension, kernels,
gradient accumulation, clipping, AdamW, and Muon semantics, but uses four layers (one attention
layer), a smaller vocabulary, and deterministic repeated-prefix synthetic data so mbs=16 fits an
L4. A fixture created once supplies byte-identical initial weights and batches to every run.

Workflow (run from the repository root):

  # Create once, preferably in the PyTorch 2.12 environment.
  python -m scripts.diagnose_training_artifact prepare

  # Run these two commands in each PyTorch environment.
  python -m scripts.diagnose_training_artifact run --microbatch 4  --output tmp/training_artifact_diag/torch212_mbs4.json
  python -m scripts.diagnose_training_artifact run --microbatch 16 --output tmp/training_artifact_diag/torch212_mbs16.json

  # After producing the analogous torch213 files:
  python -m scripts.diagnose_training_artifact compare --results tmp/training_artifact_diag/*.json

Each run tests both NoPE and Polar. It records loss trajectories, pre-clip gradient norms, grouped
gradient statistics, critical FLA/Titans and Polar gradient vectors, post-step weight vectors,
package/kernel metadata, time, and peak memory. The comparator reports all pairwise relative error
and cosine similarity results.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import itertools
import json
import math
import os
import platform
import time
from pathlib import Path

# model.blocks reads this during import. Keep the production compile-clean FLA path enabled.
os.environ.setdefault("FLA_CUSTOM_OP", "1")

import torch


DEFAULT_FIXTURE = "tmp/training_artifact_diag/fixture.pt"
DEFAULT_STEPS = 8
DEFAULT_TOTAL_SEQUENCES = 16
DEFAULT_SEQUENCE_LENGTH = 2048
DEFAULT_LAYERS = 4
DEFAULT_HIDDEN_SIZE = 1024
DEFAULT_HEAD_DIM = 128
DEFAULT_VOCAB_SIZE = 4096
CRITICAL_FULL_LIMIT = 32768
CRITICAL_SAMPLE_SIZE = 256


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def _config_dict(args, attn_type: str) -> dict:
    return {
        "vocab_size": args.vocab_size,
        "num_hidden_layers": args.layers,
        "hidden_size": args.hidden_size,
        "head_dim": args.head_dim,
        "max_position_embeddings": args.sequence_length,
        "attn_type": attn_type,
        "attn_window": None,
        "mem_enabled": True,
        "mem_chunk": 128,
        "mem_gamma_bias": 3.9,
        "mem_beta_bias": 0.0,
        "mem_kernel": "auto",
        "num_random_keys": 0,
    }


def _make_model(config_dict: dict):
    from model.config import AtmaConfig
    from train.model import Model

    model = Model(AtmaConfig(**config_dict), reg_mode="baseline", sketch_dim=64)
    for name, parameter in model.named_parameters():
        if "proj" in name:
            parameter.data.zero_()
    return model


def _synthetic_tokens(steps: int, total_sequences: int, sequence_length: int,
                      vocab_size: int, seed: int) -> torch.Tensor:
    """Repeated-prefix streams exercise induction and long-range memory without external data."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    prefix_length = (sequence_length + 2) // 2
    prefixes = torch.randint(
        1,
        vocab_size,
        (steps, total_sequences, prefix_length),
        generator=generator,
        dtype=torch.int32,
    )
    repeated = torch.cat((prefixes, prefixes), dim=-1)[..., : sequence_length + 1]
    # A step/row-specific sentinel prevents every example from being an identical phase shift.
    sentinels = (
        torch.arange(steps, dtype=torch.int32)[:, None] * total_sequences
        + torch.arange(total_sequences, dtype=torch.int32)[None, :]
    ) % (vocab_size - 1) + 1
    repeated[..., 0] = sentinels
    return repeated.contiguous()


def prepare(args) -> None:
    path = Path(args.fixture)
    if path.exists() and not args.force:
        raise SystemExit(f"fixture already exists: {path} (pass --force to replace it)")
    if args.layers < 3:
        raise SystemExit("--layers must be at least 3 so the hybrid stack contains an attention layer")
    if args.vocab_size < 32:
        raise SystemExit("--vocab-size must be at least 32")

    states = {}
    for index, attn_type in enumerate(args.attn_types):
        torch.manual_seed(args.seed + index)
        config = _config_dict(args, attn_type)
        model = _make_model(config)
        states[attn_type] = {
            "config": config,
            "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        }
        del model

    fixture = {
        "format_version": 1,
        "seed": args.seed,
        "steps": args.steps,
        "total_sequences": args.total_sequences,
        "sequence_length": args.sequence_length,
        "vocab_size": args.vocab_size,
        "attn_types": list(args.attn_types),
        "tokens": _synthetic_tokens(
            args.steps, args.total_sequences, args.sequence_length, args.vocab_size, args.seed + 1000
        ),
        "models": states,
        # torch.__version__ is a TorchVersion object in some releases. Persist a primitive so
        # fixtures remain loadable by weights_only=True across PyTorch versions.
        "created_with_torch": str(torch.__version__),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, path)
    print(f"wrote fixture: {path.resolve()} ({path.stat().st_size / 2**20:.1f} MiB)")


def _make_optimizers(model):
    from train.optimizer import Muon

    adamw = torch.optim.AdamW(
        [
            dict(params=[model.embed.weight], lr=0.3),
            dict(params=[model.proj.weight], lr=1 / 320),
            dict(params=[parameter for parameter in model.parameters() if parameter.ndim < 2], lr=0.01),
        ],
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0,
        fused=True,
    )
    muon = Muon(
        [parameter for parameter in model.blocks.parameters() if parameter.ndim >= 2],
        lr=0.02,
        weight_decay=0.01,
    )
    optimizers = (adamw, muon)
    claimed = {
        parameter
        for optimizer in optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert claimed == set(model.parameters())
    return optimizers


def _parameter_group(name: str) -> str:
    if ".attn.mem." in name:
        return "titans_memory"
    if any(token in name for token in ("len_gain_raw", "null_base", "null_slope_raw", "mag_beta_raw", "v_null", "mu_proj")):
        return "polar_structural"
    if ".attn." in name:
        return "attention_core"
    return "other"


def _is_critical(name: str) -> bool:
    return (
        ".attn.mem." in name
        or any(token in name for token in (
            "len_gain_raw", "null_base", "null_slope_raw", "mag_beta_raw", "v_null", "mu_proj",
            ".attn.q.", ".attn.k.", ".attn.v.", ".attn.canon_", ".attn.proj.",
        ))
    )


def _vector_payload(tensor: torch.Tensor) -> dict:
    flat = tensor.detach().float().reshape(-1).cpu()
    if flat.numel() <= CRITICAL_FULL_LIMIT:
        values = flat.tolist()
        mode = "full"
    else:
        count = min(CRITICAL_SAMPLE_SIZE, flat.numel())
        indices = torch.linspace(0, flat.numel() - 1, steps=count, dtype=torch.float64).round().long()
        values = flat[indices].tolist()
        mode = "sample"
    return {
        "mode": mode,
        "numel": flat.numel(),
        "values": values,
        "norm": torch.linalg.vector_norm(flat).item(),
        "max_abs": flat.abs().max().item() if flat.numel() else 0.0,
        "sum": flat.sum().item(),
        "finite": bool(torch.isfinite(flat).all()),
    }


def _tensor_signatures(model, gradients: bool) -> dict:
    groups = {}
    critical = {}
    for name, parameter in model.named_parameters():
        tensor = parameter.grad if gradients else parameter.data
        if tensor is None:
            continue
        detached = tensor.detach().float()
        group_name = _parameter_group(name)
        accumulator = groups.setdefault(
            group_name, {"sumsq": 0.0, "sum": 0.0, "max_abs": 0.0, "numel": 0, "finite": True}
        )
        accumulator["sumsq"] += detached.square().sum().item()
        accumulator["sum"] += detached.sum().item()
        accumulator["max_abs"] = max(accumulator["max_abs"], detached.abs().max().item())
        accumulator["numel"] += detached.numel()
        accumulator["finite"] = accumulator["finite"] and bool(torch.isfinite(detached).all())
        if _is_critical(name):
            critical[name] = _vector_payload(detached)
    for accumulator in groups.values():
        accumulator["norm"] = math.sqrt(accumulator.pop("sumsq"))
    return {"groups": groups, "critical": critical}


def _runtime_metadata(args) -> dict:
    from model import blocks as model_blocks
    import train.model as train_model

    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "triton": _package_version("triton"),
        "flash_linear_attention": _package_version("flash-linear-attention", "fla-core", "fla"),
        "causal_conv1d": _package_version("causal-conv1d"),
        "causal_conv_backend": (
            f"{train_model.causal_conv1d_fn.__module__}.{train_model.causal_conv1d_fn.__name__}"
        ),
        "gpu": properties.name,
        "gpu_capability": list(torch.cuda.get_device_capability()),
        "gpu_total_memory_bytes": properties.total_memory,
        "fla_available": model_blocks._HAS_FLA,
        "fla_import_error": getattr(model_blocks, "_FLA_IMPORT_ERROR", None),
        "fla_custom_op": os.environ.get("FLA_CUSTOM_OP"),
        "polar_triton": train_model.HAS_TRITON,
        "flash_sdpa_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "math_sdpa_enabled": torch.backends.cuda.math_sdp_enabled(),
        "mem_efficient_sdpa_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "compiled": not args.no_compile,
    }


def run(args) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for run mode")
    fixture_path = Path(args.fixture)
    # Fixtures created before created_with_torch was normalized to str contain the benign
    # torch.torch_version.TorchVersion metadata class. Explicitly allow just that class while
    # retaining weights_only=True; do not fall back to unrestricted pickle loading.
    torch_version_class = torch.torch_version.TorchVersion
    with torch.serialization.safe_globals([torch_version_class]):
        fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    if fixture.get("format_version") != 1:
        raise SystemExit(f"unsupported fixture format: {fixture.get('format_version')}")
    total_sequences = int(fixture["total_sequences"])
    if total_sequences % args.microbatch:
        raise SystemExit(
            f"fixture total_sequences={total_sequences} is not divisible by microbatch={args.microbatch}"
        )
    selected_types = args.attn_types or fixture["attn_types"]
    missing = sorted(set(selected_types) - set(fixture["models"]))
    if missing:
        raise SystemExit(f"fixture does not contain attention types: {missing}")

    output_path = Path(args.output)
    result = {
        "format_version": 1,
        "fixture": str(fixture_path.resolve()),
        "fixture_created_with_torch": fixture.get("created_with_torch"),
        "microbatch": args.microbatch,
        "total_sequences": total_sequences,
        "gradient_accumulation": total_sequences // args.microbatch,
        "runtime": _runtime_metadata(args),
        "models": {},
    }
    _atomic_json(output_path, result)
    torch.manual_seed(int(fixture["seed"]))
    torch.cuda.manual_seed_all(int(fixture["seed"]))
    tokens = fixture["tokens"]

    for attn_type in selected_types:
        print(f"\n[{attn_type}] torch={torch.__version__} mbs={args.microbatch}", flush=True)
        model_entry = fixture["models"][attn_type]
        base_model = _make_model(model_entry["config"])
        base_model.load_state_dict(model_entry["state_dict"], strict=True)
        base_model = base_model.cuda().train()
        optimizers = _make_optimizers(base_model)
        executable = base_model if args.no_compile else torch.compile(base_model)

        trajectory = []
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for step in range(int(fixture["steps"])):
            step_started = time.perf_counter()
            loss_sum = 0.0
            for start in range(0, total_sequences, args.microbatch):
                stop = start + args.microbatch
                stream = tokens[step, start:stop]
                inputs = stream[:, :-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
                targets = stream[:, 1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
                loss, _, _ = executable(inputs, targets)
                loss_sum += loss.detach().item()
                loss.backward()

            torch.cuda.synchronize()
            gradient_signature = _tensor_signatures(base_model, gradients=True)
            preclip_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0).item()
            for optimizer in optimizers:
                optimizer.step()
            base_model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            weight_signature = _tensor_signatures(base_model, gradients=False)
            elapsed = time.perf_counter() - step_started
            entry = {
                "step": step,
                "loss_sum": loss_sum,
                "loss_per_token": loss_sum / (total_sequences * int(fixture["sequence_length"])),
                "preclip_grad_norm": preclip_norm,
                "gradient": gradient_signature,
                "weights": weight_signature,
                "wall_seconds": elapsed,
            }
            trajectory.append(entry)
            print(
                f"  step={step} loss/token={entry['loss_per_token']:.8f} "
                f"grad_norm={preclip_norm:.6g} wall={elapsed:.1f}s",
                flush=True,
            )

        result["models"][attn_type] = {
            "config": model_entry["config"],
            "parameters": sum(parameter.numel() for parameter in base_model.parameters()),
            "trajectory": trajectory,
            "total_wall_seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        _atomic_json(output_path, result)
        del executable, base_model, optimizers
        gc.collect()
        torch.cuda.empty_cache()
        reset = getattr(torch, "_dynamo", None)
        if reset is not None:
            torch._dynamo.reset()

    print(f"\nwrote result: {output_path.resolve()}")


def _flatten_critical(signature: dict, group: str | None = None) -> tuple[list[str], list[float]]:
    names = []
    values = []
    for name in sorted(signature["critical"]):
        if group is not None and _parameter_group(name) != group:
            continue
        payload = signature["critical"][name]
        for index, value in enumerate(payload["values"]):
            names.append(f"{name}:{payload['mode']}:{index}")
            values.append(float(value))
    return names, values


def _vector_comparison(left_signature: dict, right_signature: dict,
                       group: str | None = None) -> dict:
    left_names, left_values = _flatten_critical(left_signature, group)
    right_names, right_values = _flatten_critical(right_signature, group)
    if left_names != right_names:
        return {"compatible": False, "reason": "critical parameter/sample names differ"}
    dot = sum(left * right for left, right in zip(left_values, right_values))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    diff_norm = math.sqrt(sum((left - right) ** 2 for left, right in zip(left_values, right_values)))
    if left_norm < 1e-30 and right_norm < 1e-30:
        cosine = 1.0
    elif left_norm < 1e-30 or right_norm < 1e-30:
        cosine = 0.0
    else:
        cosine = dot / (left_norm * right_norm)
    return {
        "compatible": True,
        "sample_count": len(left_values),
        "relative_l2_vs_left": diff_norm / (left_norm + 1e-30),
        "cosine": cosine,
        "left_norm": left_norm,
        "right_norm": right_norm,
    }


def _label(result: dict, path: Path) -> str:
    return f"torch={result['runtime']['torch']} mbs={result['microbatch']} ({path.name})"


def compare(args) -> None:
    loaded = []
    for name in args.results:
        path = Path(name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format_version") != 1 or not payload.get("models"):
            continue  # permits a broad shell glob containing the fixture metadata or unrelated JSON
        loaded.append((path, payload))
    if len(loaded) < 2:
        raise SystemExit("compare mode needs at least two completed result JSON files")

    summary = {"comparisons": []}
    print("\nPairwise comparison of identical fixtures (critical gradient/weight vectors):\n")
    for (left_path, left), (right_path, right) in itertools.combinations(loaded, 2):
        for attn_type in sorted(set(left["models"]) & set(right["models"])):
            left_steps = left["models"][attn_type]["trajectory"]
            right_steps = right["models"][attn_type]["trajectory"]
            count = min(len(left_steps), len(right_steps))
            step_results = []
            for index in range(count):
                left_step, right_step = left_steps[index], right_steps[index]
                gradient_groups = {
                    group: _vector_comparison(left_step["gradient"], right_step["gradient"], group)
                    for group in ("titans_memory", "attention_core", "polar_structural")
                }
                weight_groups = {
                    group: _vector_comparison(left_step["weights"], right_step["weights"], group)
                    for group in ("titans_memory", "attention_core", "polar_structural")
                }
                step_results.append({
                    "step": index,
                    "loss_abs_delta": abs(left_step["loss_per_token"] - right_step["loss_per_token"]),
                    "grad_norm_relative_delta": abs(
                        left_step["preclip_grad_norm"] - right_step["preclip_grad_norm"]
                    ) / (abs(left_step["preclip_grad_norm"]) + 1e-30),
                    "critical_gradient": _vector_comparison(
                        left_step["gradient"], right_step["gradient"]
                    ),
                    "critical_gradient_by_group": gradient_groups,
                    "critical_weights": _vector_comparison(
                        left_step["weights"], right_step["weights"]
                    ),
                    "critical_weights_by_group": weight_groups,
                })
            compatible_steps = [
                step for step in step_results if step["critical_gradient"].get("compatible")
            ]
            row = {
                "attention": attn_type,
                "left": _label(left, left_path),
                "right": _label(right, right_path),
                "steps": step_results,
                "max_loss_abs_delta": max(step["loss_abs_delta"] for step in step_results),
                "max_grad_norm_relative_delta": max(step["grad_norm_relative_delta"] for step in step_results),
                "max_critical_grad_relative_l2": max(
                    step["critical_gradient"]["relative_l2_vs_left"] for step in compatible_steps
                ) if compatible_steps else None,
                "min_critical_grad_cosine": min(
                    step["critical_gradient"]["cosine"] for step in compatible_steps
                ) if compatible_steps else None,
                "final_critical_weight_relative_l2": step_results[-1]["critical_weights"].get(
                    "relative_l2_vs_left"
                ),
                "groups": {},
            }
            for group in ("titans_memory", "attention_core", "polar_structural"):
                group_steps = [
                    step["critical_gradient_by_group"][group]
                    for step in step_results
                    if step["critical_gradient_by_group"][group].get("compatible")
                    and step["critical_gradient_by_group"][group].get("sample_count", 0) > 0
                ]
                row["groups"][group] = {
                    "max_gradient_relative_l2": max(
                        item["relative_l2_vs_left"] for item in group_steps
                    ) if group_steps else None,
                    "min_gradient_cosine": min(item["cosine"] for item in group_steps) if group_steps else None,
                    "final_weight_relative_l2": step_results[-1]["critical_weights_by_group"][group].get(
                        "relative_l2_vs_left"
                    ),
                }
            summary["comparisons"].append(row)
            memory = row["groups"]["titans_memory"]
            attention = row["groups"]["attention_core"]
            print(
                f"{attn_type:>5} | {row['left']}  VS  {row['right']}\n"
                f"      max Δloss={row['max_loss_abs_delta']:.3e}  "
                f"max Δgradnorm={row['max_grad_norm_relative_delta']:.3e}  "
                f"max grad relL2={row['max_critical_grad_relative_l2']:.3e}  "
                f"min grad cos={row['min_critical_grad_cosine']:.8f}  "
                f"final weight relL2={row['final_critical_weight_relative_l2']:.3e}\n"
                f"      Titans: relL2={memory['max_gradient_relative_l2']:.3e} "
                f"cos={memory['min_gradient_cosine']:.8f} | "
                f"attention: relL2={attention['max_gradient_relative_l2']:.3e} "
                f"cos={attention['min_gradient_cosine']:.8f}"
            )

    if args.output:
        _atomic_json(Path(args.output), summary)
        print(f"\nwrote comparison: {Path(args.output).resolve()}")
    print(
        "\nInterpretation: mbs-only divergence isolates accumulation/kernel batch-shape effects; "
        "version-only divergence isolates the PyTorch/Triton/FLA stack. Critical-gradient cosine "
        "well below 0.999 or relative L2 above a few percent is large enough to investigate."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="create shared weights and synthetic batches")
    prepare_parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.add_argument("--seed", type=int, default=1234)
    prepare_parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    prepare_parser.add_argument("--total-sequences", type=int, default=DEFAULT_TOTAL_SEQUENCES)
    prepare_parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    prepare_parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    prepare_parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    prepare_parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    prepare_parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    prepare_parser.add_argument("--attn-types", nargs="+", choices=("nope", "polar"), default=("nope", "polar"))
    prepare_parser.set_defaults(handler=prepare)

    run_parser = subparsers.add_parser("run", help="run one version/microbatch condition")
    run_parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    run_parser.add_argument("--microbatch", type=int, choices=(4, 16), required=True)
    run_parser.add_argument("--attn-types", nargs="+", choices=("nope", "polar"), default=None)
    run_parser.add_argument("--no-compile", action="store_true")
    run_parser.add_argument("--output", required=True)
    run_parser.set_defaults(handler=run)

    compare_parser = subparsers.add_parser("compare", help="compare completed condition JSON files")
    compare_parser.add_argument("--results", nargs="+", required=True)
    compare_parser.add_argument("--output", default="tmp/training_artifact_diag/comparison.json")
    compare_parser.set_defaults(handler=compare)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
