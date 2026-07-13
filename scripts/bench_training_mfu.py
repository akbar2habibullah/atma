"""Synthetic end-to-end training throughput and MFU benchmark.

This intentionally avoids datasets, validation, logging, and checkpoints. It executes the real
training model, loss, backward, gradient clipping, and the repository's AdamW+Muon optimizer mix.
Compile/warmup time is reported separately and excluded from steady-state measurements.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def nominal_bf16_tflops(name: str) -> float | None:
    lowered = name.lower()
    if "b200" in lowered or "b300" in lowered:
        return 2250.0
    if "h100" in lowered or "h200" in lowered:
        return 989.0
    if "l40s" in lowered:
        return 362.05
    if "l4" in lowered:
        return 121.0
    if "a100" in lowered:
        return 312.0
    return None


def measure_bf16_peak() -> float:
    shapes = ((4096, 8192, 1024), (8192, 8192, 1024), (8192, 4096, 1024))
    values = []
    for m, n, k in shapes:
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
        for _ in range(5):
            torch.mm(a, b)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            torch.mm(a, b)
        end.record()
        end.synchronize()
        seconds = start.elapsed_time(end) * 1e-3 / 20
        values.append(2 * m * n * k / seconds / 1e12)
    torch.cuda.empty_cache()
    return statistics.fmean(values)


def make_optimizers(model):
    from train.optimizer import Muon

    adamw = torch.optim.AdamW(
        [
            dict(params=[model.embed.weight], lr=0.3),
            dict(params=[model.proj.weight], lr=1 / 320),
            dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01),
        ],
        betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True,
    )
    muon = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
                lr=0.02, weight_decay=0.01)
    claimed = {p for optimizer in (adamw, muon) for group in optimizer.param_groups
               for p in group["params"]}
    assert claimed == set(model.parameters())
    return adamw, muon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--microbatch", type=int, default=16,
                        help="sequences per forward/backward microbatch")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--allow-conv-fallback", action="store_true",
                        help="permit the slower PyTorch causal-convolution fallback")
    parser.add_argument("--measure-peak", action="store_true")
    parser.add_argument("--peak-tflops", type=float,
                        help="override nominal dense BF16 Tensor Core peak")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if min(args.microbatch, args.grad_accum, args.seq_length, args.iterations) < 1:
        raise SystemExit("microbatch, grad accumulation, sequence length, and iterations must be positive")

    from model.config import AtmaConfig
    import train.model as train_model
    from model import blocks as model_blocks

    causal_conv_backend = (f"{train_model.causal_conv1d_fn.__module__}."
                           f"{train_model.causal_conv1d_fn.__name__}")
    using_conv_fallback = train_model.causal_conv1d_fn is train_model._causal_conv1d_fallback
    if using_conv_fallback and not args.allow_conv_fallback:
        raise SystemExit(
            "optimized causal-conv1d kernel unavailable; cache/install it before profiling "
            "or pass --allow-conv-fallback to measure the fallback explicitly")
    if not model_blocks._HAS_FLA:
        raise SystemExit("flash-linear-attention fused gated-delta kernel is required")

    torch.manual_seed(0)
    device_name = torch.cuda.get_device_name()
    config = AtmaConfig(
        vocab_size=args.vocab_size,
        num_hidden_layers=args.layers,
        hidden_size=args.hidden_size,
        head_dim=args.head_dim,
        max_position_embeddings=args.seq_length,
    )
    model = train_model.Model(config, reg_mode="baseline", sketch_dim=64).cuda().train()
    for name, parameter in model.named_parameters():
        if "proj" in name:
            parameter.data.zero_()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    attention_layers = model.num_attn_layers
    optimizers = make_optimizers(model)
    if not args.no_compile:
        model = torch.compile(model)

    inputs = torch.randint(
        0, args.vocab_size, (args.microbatch, args.seq_length),
        device="cuda", dtype=torch.int64,
    )
    targets = torch.randint(
        0, args.vocab_size, (args.microbatch, args.seq_length),
        device="cuda", dtype=torch.int64,
    )

    def step() -> torch.Tensor:
        total_loss = 0.0
        for _ in range(args.grad_accum):
            loss, _, _ = model(inputs, targets)
            total_loss += loss.detach()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for optimizer in optimizers:
            optimizer.step()
        model.zero_grad(set_to_none=True)
        return total_loss

    compile_started = time.perf_counter()
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    compile_warmup_seconds = time.perf_counter() - compile_started

    torch.cuda.reset_peak_memory_stats()
    wall_ms, gpu_ms = [], []
    losses = []
    for _ in range(args.iterations):
        start_event, end_event = (torch.cuda.Event(enable_timing=True),
                                  torch.cuda.Event(enable_timing=True))
        torch.cuda.synchronize()
        wall_started = time.perf_counter()
        start_event.record()
        loss_value = step()
        end_event.record()
        torch.cuda.synchronize()
        losses.append(float(loss_value))
        wall_ms.append((time.perf_counter() - wall_started) * 1000)
        gpu_ms.append(start_event.elapsed_time(end_event))

    tokens_per_step = args.microbatch * args.seq_length * args.grad_accum
    wall_p50_s = percentile(wall_ms, 0.50) / 1000
    # PaLM-style training FLOPs. The legacy repository formula charges quadratic attention to
    # every layer; the hybrid formula charges only the actual attention layers. Neither credits
    # extra Polar/Titans/conv/optimizer operations, so these are useful-model FLOP proxies.
    dense_parameter_flops = 6 * parameter_count
    hybrid_attention_flops = 12 * attention_layers * args.hidden_size * args.seq_length
    legacy_attention_flops = 12 * args.layers * args.hidden_size * args.seq_length
    hybrid_flops_per_token = dense_parameter_flops + hybrid_attention_flops
    legacy_flops_per_token = dense_parameter_flops + legacy_attention_flops
    achieved_hybrid_tflops = hybrid_flops_per_token * tokens_per_step / wall_p50_s / 1e12
    achieved_legacy_tflops = legacy_flops_per_token * tokens_per_step / wall_p50_s / 1e12
    nominal_peak = args.peak_tflops or nominal_bf16_tflops(device_name)
    measured_peak = measure_bf16_peak() if args.measure_peak else None

    result = {
        "status": "ok",
        "gpu": device_name,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "causal_conv_backend": causal_conv_backend,
        "causal_conv_fallback": using_conv_fallback,
        "fla_fused_gated_delta": model_blocks._HAS_FLA,
        "polar_triton": train_model.HAS_TRITON,
        "compiled": not args.no_compile,
        "model": {
            "parameters": parameter_count,
            "layers": args.layers,
            "attention_layers": attention_layers,
            "hidden_size": args.hidden_size,
            "head_dim": args.head_dim,
            "vocab_size": args.vocab_size,
            "attn_type": config.attn_type,
            "attn_window": config.attn_window,
            "memory_enabled": config.mem_enabled,
        },
        "workload": {
            "microbatch_sequences": args.microbatch,
            "gradient_accumulation": args.grad_accum,
            "sequence_length": args.seq_length,
            "tokens_per_step": tokens_per_step,
            "warmup_steps": args.warmup,
            "measured_steps": args.iterations,
            "optimizer": "fused AdamW + Muon",
            "includes_optimizer": True,
            "includes_gradient_clip": True,
        },
        "compile_warmup_seconds": compile_warmup_seconds,
        "step_wall_ms": {
            "p50": percentile(wall_ms, 0.50),
            "p95": percentile(wall_ms, 0.95),
            "mean": statistics.fmean(wall_ms),
            "samples": wall_ms,
        },
        "step_gpu_ms": {
            "p50": percentile(gpu_ms, 0.50),
            "p95": percentile(gpu_ms, 0.95),
            "mean": statistics.fmean(gpu_ms),
        },
        "throughput_tokens_s": tokens_per_step / wall_p50_s,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "loss_last_sum": losses[-1],
        "flop_convention": {
            "hybrid_primary": "6*N + 12*attention_layers*hidden_size*sequence_length",
            "legacy_train_py": "6*N + 12*all_layers*hidden_size*sequence_length",
            "caveat": "Polar, Titans, convolution, loss, clipping, and optimizer FLOPs are timed but not fully credited",
        },
        "hybrid_flops_per_token": hybrid_flops_per_token,
        "legacy_flops_per_token": legacy_flops_per_token,
        "achieved_hybrid_tflops": achieved_hybrid_tflops,
        "achieved_legacy_tflops": achieved_legacy_tflops,
        "nominal_peak_bf16_tflops": nominal_peak,
        "measured_peak_bf16_tflops": measured_peak,
        "mfu_hybrid_nominal_pct": (100 * achieved_hybrid_tflops / nominal_peak
                                    if nominal_peak else None),
        "mfu_legacy_nominal_pct": (100 * achieved_legacy_tflops / nominal_peak
                                    if nominal_peak else None),
        "mfu_hybrid_measured_pct": (100 * achieved_hybrid_tflops / measured_peak
                                     if measured_peak else None),
        "mfu_legacy_measured_pct": (100 * achieved_legacy_tflops / measured_peak
                                     if measured_peak else None),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
