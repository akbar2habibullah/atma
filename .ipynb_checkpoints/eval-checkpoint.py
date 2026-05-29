"""
eval.py

Evaluate length extrapolation from a saved checkpoint.
Measures cross-entropy loss at multiples of the training sequence length
to quantify how well the model generalises beyond its training context window.

Usage:
    python eval.py
    python eval.py --checkpoint checkpoints --num_seqs 16
    python eval.py --multipliers 1 2 4 8 16 32 64
"""

import os
import json
import argparse

import torch

from model.config import AtmaConfig
from train.model import Model
from train.data import data_generator


def load_from_checkpoint(checkpoint_dir: str, device: torch.device):
    config_path = os.path.join(checkpoint_dir, "config.json")
    weights_path = os.path.join(checkpoint_dir, "weights.pt")

    with open(config_path) as f:
        cfg = json.load(f)

    cfg["dtype"] = getattr(torch, cfg["dtype"])
    cfg["num_random_keys"] = 0  # distractor path is training-only; no weights attached

    config = AtmaConfig(**cfg)
    model = Model(config).to(device)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)["model"]
    model.load_state_dict(state_dict)
    model = torch.compile(model)
    model.eval()

    return model, config


def eval_length(model: torch.nn.Module, val_data: str, seq_len: int, num_seqs: int):
    gen = data_generator(val_data, seq_len, seq_len=seq_len)  # yields (1, seq_len) each call
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_seqs):
            inputs, targets = next(gen)  # one sequence at a time — no pre-loaded batch
            loss, _, _ = model(inputs, targets)
            total_loss += loss.item()
    return total_loss / (num_seqs * seq_len)


def main():
    parser = argparse.ArgumentParser(description="Length extrapolation evaluation")
    parser.add_argument("--checkpoint", default="checkpoints",
                        help="Checkpoint directory written by train.py (default: checkpoints)")
    parser.add_argument("--val_data", default="finewebedu10B/finewebedu_val_*.bin",
                        help="Glob pattern for validation shards")
    parser.add_argument("--base_len", type=int, default=1024,
                        help="Training sequence length (default: 1024)")
    parser.add_argument("--num_seqs", type=int, default=16,
                        help="Sequences evaluated per length — more gives a stabler estimate (default: 16)")
    parser.add_argument("--multipliers", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64],
                        help="Context length multipliers to evaluate (default: 1 2 4 8 16 32 64)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required — data_generator writes directly to GPU memory.")

    device = torch.device("cuda")

    print(f"Loading checkpoint from '{args.checkpoint}'...")
    model, config = load_from_checkpoint(args.checkpoint, device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {num_params:.2f}M parameters  |  hidden={config.hidden_size}  "
          f"layers={config.num_hidden_layers}  vocab={config.vocab_size}")

    print(f"\nEvaluating {args.num_seqs} sequences per length "
          f"(base_len={args.base_len}, multipliers={args.multipliers}):\n")

    header = f"{'Multiplier':>12}  {'Seq length':>12}  {'Loss':>10}  {'Δ vs 1×':>10}"
    print(header)
    print("-" * len(header))

    results = {}
    for mult in args.multipliers:
        seq_len = args.base_len * mult

        torch.cuda.empty_cache()

        loss = eval_length(model, args.val_data, seq_len, args.num_seqs)
        results[mult] = loss

        baseline = results.get(1, loss)
        delta = loss - baseline
        delta_str = f"{delta:+.5f}" if mult != 1 else "—"
        print(f"{f'{mult}x':>12}  {seq_len:>12,}  {loss:>10.5f}  {delta_str:>10}")

    if len(results) > 1 and 1 in results:
        best_extrap = min((m for m in results if m > 1), key=lambda m: results[m])
        worst_extrap = max((m for m in results if m > 1), key=lambda m: results[m])
        print(f"\nBest extrapolation:  {best_extrap}×  loss={results[best_extrap]:.5f}")
        print(f"Worst extrapolation: {worst_extrap}×  loss={results[worst_extrap]:.5f}")


if __name__ == "__main__":
    main()
