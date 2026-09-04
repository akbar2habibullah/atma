"""GPU-only dependency, parameter-count, forward, and backward preflight."""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _check_sources(repo_root: Path, required: set[str]):
    deps = json.loads((ROOT / "dependencies.json").read_text(encoding="utf-8"))
    for name in sorted(required):
        dep = deps[name]
        path = repo_root / dep["checkout"]
        if not path.is_dir():
            raise RuntimeError(f"missing {name} checkout: {path}")
        actual = _git_head(path)
        if actual != dep["commit"]:
            raise RuntimeError(f"{name} commit mismatch: expected {dep['commit']}, found {actual}")
        tracked_dirty = subprocess.run(["git", "-C", str(path), "diff", "--quiet"]).returncode != 0
        staged_dirty = subprocess.run(["git", "-C", str(path), "diff", "--cached", "--quiet"]).returncode != 0
        if tracked_dirty or staged_dirty:
            raise RuntimeError(f"{name} checkout has tracked changes; update the pin instead of patching silently")
        if name in {"flash_linear_attention", "mamba"} and str(path) not in sys.path:
            sys.path.insert(0, str(path))
        if name == "tda" and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_SHAPE_KEYS = {
    "vocab_size", "hidden_size", "num_hidden_layers", "head_dim", "conv_kernel_size",
    "mamba3_state_size", "mamba3_expand", "mamba3_head_dim", "mamba3_n_groups",
    "mamba3_rope_fraction", "mamba3_mimo", "mamba3_mimo_rank", "mamba3_chunk_size",
    "gdn2_expand_v", "gdn2_num_v_heads", "gdn2_short_conv", "gdn2_allow_neg_eigval",
    "gdn2_conv_size", "tda_beta", "tda_lambda_init", "tda_relu_power", "tda_tuned_kernel",
    "mem_enabled", "mem_chunk", "mem_gamma_bias", "mem_beta_bias", "mem_kernel",
}


def _approve_same_arch(config_root: Path, source: dict, count: int):
    arch = source["arch_type"]
    for path in config_root.glob("*/*.json"):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if cfg.get("arch_type") != arch:
            continue
        for key in _SHAPE_KEYS:
            if key in source:
                cfg[key] = source[key]
        cfg["resolved_num_params"] = count
        cfg["parameter_count_approved"] = True
        path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, default=ROOT / "configs" / "baseline_pilots")
    parser.add_argument("--include", default="*.json", help="config filename glob")
    parser.add_argument(
        "--config_root", "--work_root", dest="config_root", type=Path,
        default=ROOT / "configs", help="config tree updated on approval",
    )
    parser.add_argument("--repo_root", type=Path, default=Path.cwd())
    parser.add_argument("--approve", action="store_true", help="approve counts within tolerance in the config tree")
    parser.add_argument("--sequence_length", type=int, default=128)
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this preflight")
    config_paths = sorted(args.configs.glob(args.include))
    if not config_paths:
        raise SystemExit(f"no configs match {args.include!r} under {args.configs}")
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in config_paths]
    required = {
        name
        for cfg in configs
        for name in (cfg.get("dependency_commits") or {})
    }
    _check_sources(args.repo_root.resolve(), required)
    from external_baselines.model import create_model
    if "tda" in required:
        from external_baselines.gpu_checks import check_tda, check_tda_tuned
        check_tda((args.repo_root / "third_party" / "TDA").resolve())
        if any(cfg.get("tda_tuned_kernel", False) for cfg in configs):
            check_tda_tuned(args.sequence_length)

    failures = []
    approvals = []
    for path, cfg in zip(config_paths, configs):
        model = create_model(cfg).cuda().train()
        count = sum(p.numel() for p in model.parameters())
        target = int(cfg["parameter_count_target"])
        delta = abs(count - target) / target
        print(f"{cfg['run_id']}: {count:,} params ({delta:.2%} from target)")
        try:
            x = torch.randint(0, cfg["vocab_size"], (1, args.sequence_length), device="cuda", dtype=torch.int32)
            y = torch.randint(0, cfg["vocab_size"], (1, args.sequence_length), device="cuda", dtype=torch.int64)
            checked_model = model
            if cfg.get("gdn2_cuda_graph", False) or cfg.get("tda_cuda_graph", False):
                checkpoint_keys = tuple(model.state_dict())
                with torch.no_grad():
                    eager_loss, _, _ = model(x, y)
                if cfg.get("gdn2_cuda_graph", False):
                    from external_baselines.gdn2_training import GDN2CUDAGraphTrainer

                    trainer = GDN2CUDAGraphTrainer(model)
                else:
                    from external_baselines.tda_training import TDACUDAGraphTrainer

                    trainer = TDACUDAGraphTrainer(model)
                if tuple(model.state_dict()) != checkpoint_keys:
                    raise RuntimeError(f"optimized {cfg['arch_type']} runner changed checkpoint keys")
                optimized_loss = trainer.forward_loss(x, y)
                torch.testing.assert_close(optimized_loss, eager_loss, rtol=2e-3, atol=2e-3)
                trainer.backward(x, y)
                loss = optimized_loss
                mode = "split-compiled CUDA-graph"
            elif cfg.get("external_custom_op", False):
                from external_baselines.custom_ops import custom_op_is_installed
                if not custom_op_is_installed(cfg["arch_type"]):
                    raise RuntimeError("required external custom op was not installed")
                checked_model = torch.compile(model, fullgraph=True)
                loss, _, _ = checked_model(x, y)
                (loss / args.sequence_length).backward()
                mode = "fullgraph compiled"
            else:
                loss, _, _ = checked_model(x, y)
                (loss / args.sequence_length).backward()
                mode = "eager"
            finite = torch.isfinite(loss) and all(
                p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
            )
            if not finite:
                raise RuntimeError("non-finite loss or gradients")
            print(f"  {mode} forward/backward OK; loss/token={loss.item() / args.sequence_length:.5f}")
        except Exception as exc:
            failures.append(f"{cfg['run_id']}: {exc}")
        else:
            if delta > float(cfg["parameter_tolerance_frac"]):
                failures.append(f"{cfg['run_id']}: parameter count outside tolerance")
            elif args.approve:
                approvals.append((cfg, count))
        finally:
            if "checked_model" in locals():
                del checked_model
            if "trainer" in locals():
                del trainer
            if "optimized_loss" in locals():
                del optimized_loss
            if "eager_loss" in locals():
                del eager_loss
            if "loss" in locals():
                del loss
            if "checkpoint_keys" in locals():
                del checkpoint_keys
            if "x" in locals():
                del x
            if "y" in locals():
                del y
            del model
            gc.collect()
            torch.cuda.empty_cache()

    if failures:
        raise SystemExit("GPU preflight failed:\n" + "\n".join(f"- {x}" for x in failures))
    for cfg, count in approvals:
        _approve_same_arch(args.config_root, cfg, count)
    print("GPU preflight passed" + (" and parameter counts approved" if args.approve else ""))


if __name__ == "__main__":
    main()
