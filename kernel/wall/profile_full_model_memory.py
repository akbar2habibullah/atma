"""Synthetic full-model memory profiler for the Wall investigation.

This uses random token batches and the ablation model path. It avoids data downloads
and is meant to isolate model/pipeline allocation behavior from dataset IO.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import sys
import time
import types

import torch


def install_environment_shims() -> None:
    # train.model treats HuggingFace kernels as optional. In minimal environments,
    # provide a module that makes those imports fall back cleanly.
    if "kernels" not in sys.modules:
        kernels = types.ModuleType("kernels")
        def get_kernel(*args, **kwargs):
            raise RuntimeError("kernels package unavailable in profiler environment")
        kernels.get_kernel = get_kernel
        sys.modules["kernels"] = kernels

    # Upstream wall-attention-release imports only einops.reduce for two GQA reductions.
    if "einops" not in sys.modules:
        def reduce(x, pattern, *, g, reduction):
            if reduction != "sum":
                raise NotImplementedError(reduction)
            if pattern == "b t (h g) k -> b t h k":
                b, t, hg, k = x.shape
                return x.reshape(b, t, hg // g, g, k).sum(dim=3)
            if pattern == "b t (h g) v -> b t h v":
                b, t, hg, v = x.shape
                return x.reshape(b, t, hg // g, g, v).sum(dim=3)
            raise NotImplementedError(pattern)
        einops = types.ModuleType("einops")
        einops.reduce = reduce
        sys.modules["einops"] = einops


def load_wall_kernel(name: str):
    if name == "local":
        from kernel.wall import wall_attn
        return wall_attn
    if name == "upstream":
        sys.path.insert(0, "/home/sagemaker-user/wall-attention-release")
        from wall_attn import wall_attn
        return wall_attn
    return None


def make_config(args):
    from model.config import AtmaConfig
    return AtmaConfig(
        vocab_size=50304,
        num_hidden_layers=args.layers,
        hidden_size=1024,
        head_dim=128,
        max_position_embeddings=args.seq_len,
        num_random_keys=args.seq_len if args.distractor else 0,
        attn_type=args.attn_type,
        attn_window=args.window_size if args.window else None,
        mem_enabled=args.memory,
        mem_chunk=128,
        mem_gamma_bias=3.9,
        mem_beta_bias=0.0,
        mem_kernel="auto",
        wall_gate_bias=-4.0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attn_type", choices=("rope", "nope", "polar", "wall"), default="wall")
    p.add_argument("--wall_impl", choices=("local", "upstream", "default"), default="default")
    p.add_argument("--mbs", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--reg_mode", default="strong")
    p.add_argument("--distractor", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--window", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--window_size", type=int, default=1024)
    p.add_argument("--sigr_alpha", type=float, default=0.01)
    p.add_argument("--dist_align_weight", type=float, default=0.01)
    p.add_argument("--compile", action="store_true", help="wrap model with torch.compile, matching ablation.train")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    install_environment_shims()

    import train.model as tm
    if args.attn_type == "wall" and args.wall_impl != "default":
        wall = load_wall_kernel(args.wall_impl)
        tm._wall_attn_kernel = wall
        tm._HAS_WALL = True
        print(f"wall_impl={args.wall_impl},module={wall.__module__},file={inspect.getfile(wall)}")
    elif args.attn_type == "wall":
        print(f"wall_impl=default,module={tm._wall_attn_kernel.__module__},file={inspect.getfile(tm._wall_attn_kernel)}")

    from train.model import Model
    ac = make_config(args)

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); gc.collect()
    try:
        model = Model(ac, reg_mode=args.reg_mode, sketch_dim=64).cuda().train()
        for name, param in model.named_parameters():
            if "proj" in name:
                param.data.zero_()
        if args.compile:
            model = torch.compile(model)
        x = torch.randint(0, ac.vocab_size, (args.mbs, args.seq_len), device="cuda", dtype=torch.int32)
        y = torch.randint(0, ac.vocab_size, (args.mbs, args.seq_len), device="cuda", dtype=torch.int64)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        lm_loss, reg_loss, align_loss = model(x, y)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        peak_fwd_alloc = torch.cuda.max_memory_allocated() / 1024**3
        peak_fwd_reserved = torch.cuda.max_memory_reserved() / 1024**3

        loss = (1 - args.sigr_alpha) * lm_loss + args.sigr_alpha * reg_loss + args.dist_align_weight * align_loss
        loss.backward()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        print(
            f"status=ok,attn_type={args.attn_type},wall_impl={args.wall_impl},mbs={args.mbs},"
            f"layers={args.layers},compile={args.compile},distractor={args.distractor},memory={args.memory},window={args.window},"
            f"forward_ms={(t1-t0)*1000:.2f},backward_ms={(t2-t1)*1000:.2f},"
            f"forward_peak_alloc_gb={peak_fwd_alloc:.3f},forward_peak_reserved_gb={peak_fwd_reserved:.3f},"
            f"total_peak_alloc_gb={torch.cuda.max_memory_allocated()/1024**3:.3f},"
            f"total_peak_reserved_gb={torch.cuda.max_memory_reserved()/1024**3:.3f},"
            f"lm_loss={float(lm_loss.detach()):.6f},reg_loss={float(reg_loss.detach()):.6f},"
            f"align_loss={float(align_loss.detach()):.6f}",
            flush=True,
        )
    except RuntimeError as e:
        status = "oom" if "out of memory" in str(e).lower() else "error"
        print(
            f"status={status},attn_type={args.attn_type},wall_impl={args.wall_impl},mbs={args.mbs},"
            f"layers={args.layers},compile={args.compile},distractor={args.distractor},memory={args.memory},window={args.window},"
            f"peak_alloc_gb={torch.cuda.max_memory_allocated()/1024**3:.3f},"
            f"peak_reserved_gb={torch.cuda.max_memory_reserved()/1024**3:.3f},"
            f"error={str(e).splitlines()[0]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
