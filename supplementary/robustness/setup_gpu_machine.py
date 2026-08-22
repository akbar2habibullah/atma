"""Install and verify the pinned kernel sources needed by one 10B worker role."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROBUSTNESS = Path(__file__).resolve().parent
ROLE_DEPENDENCIES = {
    "replication": ("flash_linear_attention",),
    "tda": ("flash_linear_attention", "tda"),
    "mamba3": ("flash_linear_attention", "mamba"),
    "gdn2": ("flash_linear_attention",),
}


def _run(*args: str):
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def _checkout(name: str, dep: dict):
    path = ROOT / dep["checkout"]
    if path.exists() and not (path / ".git").is_dir():
        raise SystemExit(f"refusing to replace non-git path: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _run("git", "clone", "--filter=blob:none", dep["repository"], str(path))
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise SystemExit(f"refusing to overwrite tracked changes in {path}:\n{dirty}")
    _run("git", "-C", str(path), "fetch", "--depth=1", "origin", dep["commit"])
    _run("git", "-C", str(path), "checkout", "--detach", dep["commit"])
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != dep["commit"]:
        raise SystemExit(f"{name} checkout mismatch: expected {dep['commit']}, found {actual}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=tuple(ROLE_DEPENDENCIES), required=True)
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA-enabled PyTorch and a visible GPU are required")

    dependencies = json.loads((ROBUSTNESS / "dependencies.json").read_text(encoding="utf-8"))
    import triton
    runtime = dependencies["_runtime"]
    actual_runtime = {
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "triton": str(triton.__version__),
    }
    if actual_runtime != {key: runtime[key] for key in actual_runtime}:
        raise SystemExit(
            f"unvalidated GPU runtime: expected {runtime}, found {actual_runtime}; "
            "use the same image/environment as the prepared pilot machine"
        )
    required = ROLE_DEPENDENCIES[args.role]
    for name in required:
        _checkout(name, dependencies[name])

    # Preserve the machine's CUDA-enabled torch/triton build. The pinned source trees
    # are installed editable without dependency resolution so pip cannot replace them.
    _run(sys.executable, "-m", "pip", "install", "ninja", "einops", "transformers", "packaging")
    _run(
        sys.executable, "-m", "pip", "install", "--no-deps", "-e",
        str(ROOT / dependencies["flash_linear_attention"]["checkout"]),
    )
    if "mamba" in required:
        _run(sys.executable, "-m", "pip", "install", "quack-kernels==0.6.4")
        _run(
            sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e",
            str(ROOT / dependencies["mamba"]["checkout"]),
        )

    # Import from the pinned worktrees even if the machine has another global version.
    sys.path.insert(0, str(ROOT / dependencies["flash_linear_attention"]["checkout"]))
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401

    if args.role == "mamba3":
        sys.path.insert(0, str(ROOT / dependencies["mamba"]["checkout"]))
        from fla.layers.mamba3 import is_fast_path_available
        if not is_fast_path_available:
            raise SystemExit("Mamba-3 SISO fused kernel is unavailable after installation")
    elif args.role == "gdn2":
        from fla.layers import GatedDeltaNet2  # noqa: F401
    elif args.role == "tda":
        sys.path.insert(0, str(ROOT / dependencies["tda"]["checkout"]))
        from triton_threshold_attention import differential_threshold_rela_triton  # noqa: F401

    print(f"GPU machine setup OK for role={args.role}; torch={torch.__version__}; gpu={torch.cuda.get_device_name()}")


if __name__ == "__main__":
    main()
