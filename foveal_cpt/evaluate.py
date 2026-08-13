from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .attention import FovealAttention
from .checkpoint import load_foveal_weights, load_pretrained
from .config import FovealConfig
from .data import TokenShardLoader
from .model import FovealCPTModel, foveal_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Foveal loss and routing budgets")
    parser.add_argument("--config", default="foveal_cpt/pilot.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-glob", default="finewebedu10B/finewebedu_val_*.bin")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 32768])
    parser.add_argument("--top-p", type=float, nargs="+", default=[0.90, 0.95, 0.98])
    parser.add_argument("--fixed-k", type=int, nargs="*", default=[8, 16, 32, 64])
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    return parser.parse_args()


@torch.no_grad()
def evaluate_route(
    model: FovealCPTModel,
    config: FovealConfig,
    *,
    val_glob: str,
    length: int,
    sequences: int,
    top_p: float,
    kmin: int,
    kmax: int,
    device: torch.device,
) -> dict:
    model.set_route(top_p, kmin, kmax)
    loader = TokenShardLoader(val_glob, length, length, device)
    loss_sum = 0.0
    token_count = 0
    stats: dict[str, list[float]] = {}
    for _ in range(sequences):
        inputs, targets = loader.next()
        loss, _, _ = model(inputs, targets)
        loss_sum += float(loss)
        token_count += targets.numel()
        for key, value in model.route_stats().items():
            stats.setdefault(key, []).append(value)
    return {
        "loss_nats": loss_sum / token_count,
        "tokens": token_count,
        "route": {key: sum(values) / len(values) for key, values in stats.items()},
    }


def main() -> None:
    args = parse_args()
    config = FovealConfig.load(args.config)
    device = torch.device(args.device)
    base, _, _ = load_pretrained(config, device)
    model = FovealCPTModel(base, config)
    load_foveal_weights(model.base, args.checkpoint)
    model.set_mode("sparse")
    model.config.activation_checkpointing = False
    for layer in foveal_layers(model.base):
        layer.teacher_query_blocks = 0
    model.eval()

    result = {"checkpoint": args.checkpoint, "top_p": {}, "fixed_k": {}}
    for length in args.lengths:
        if length % config.page_size:
            raise ValueError(f"length {length} must be divisible by page size")
        for p in args.top_p:
            key = f"length={length},p={p}"
            result["top_p"][key] = evaluate_route(
                model,
                config,
                val_glob=args.val_glob,
                length=length,
                sequences=args.sequences,
                top_p=p,
                kmin=0,
                kmax=config.max_remote_pages,
                device=device,
            )
            print(key, result["top_p"][key])
        for fixed_k in args.fixed_k:
            if fixed_k > config.remote_capacity:
                continue
            key = f"length={length},k={fixed_k}"
            result["fixed_k"][key] = evaluate_route(
                model,
                config,
                val_glob=args.val_glob,
                length=length,
                sequences=args.sequences,
                top_p=1.0,
                kmin=fixed_k,
                kmax=fixed_k,
                device=device,
            )
            print(key, result["fixed_k"][key])

    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
