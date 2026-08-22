"""GPU-side strict reload and finite-forward check for an external checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_pinned_sources():
    """Mirror the worker's import path so reload uses the approved checkouts."""
    repo_root = Path(__file__).resolve().parents[1]
    for relative in ("third_party/mamba", "third_party/flash-linear-attention"):
        source = repo_root / relative
        if source.is_dir() and str(source) not in sys.path:
            sys.path.insert(0, str(source))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sequence_length", type=int, default=128)
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    _add_pinned_sources()
    from external_baselines.model import create_model

    cfg = json.loads((args.checkpoint / "run_config.json").read_text(encoding="utf-8"))
    model = create_model(cfg).cuda().eval()
    state = torch.load(args.checkpoint / "weights.pt", map_location="cuda", weights_only=True)["model"]
    model.load_state_dict(state, strict=True)
    actual = sum(p.numel() for p in model.parameters())
    if actual != cfg.get("resolved_num_params"):
        raise RuntimeError(f"parameter count mismatch after reload: {actual} != {cfg.get('resolved_num_params')}")
    x = torch.randint(0, cfg["vocab_size"], (1, args.sequence_length), device="cuda", dtype=torch.int32)
    y = torch.randint(0, cfg["vocab_size"], (1, args.sequence_length), device="cuda", dtype=torch.int64)
    with torch.no_grad():
        loss, _, _ = model(x, y)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite loss after checkpoint reload")
    print(f"strict reload OK: {cfg['run_id']}, params={actual:,}, loss/token={loss.item()/args.sequence_length:.5f}")


if __name__ == "__main__":
    main()
