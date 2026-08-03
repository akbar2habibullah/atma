"""B200/B300 tensor-shape and memory-bandwidth calibration.

Unlike the legacy L4/L40S calibration, this separates peak-seeking GEMMs from
the exact matrix families exercised by Atma 9.2B inference and training.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass, asdict

import torch


@dataclass(frozen=True)
class GemmShape:
    group: str
    name: str
    m: int
    n: int
    k: int


PEAK_SHAPES = (
    GemmShape("peak", "square_16k", 16384, 16384, 16384),
    GemmShape("peak", "wide_32k", 16384, 32768, 16384),
    GemmShape("peak", "tall_32k", 32768, 16384, 16384),
)

ATMA_9B_PREFILL = (
    GemmShape("atma9b_prefill", "q_gate", 4096, 8192, 4096),
    GemmShape("atma9b_prefill", "conv_in", 4096, 12288, 4096),
    GemmShape("atma9b_prefill", "mlp_expand", 4096, 32768, 4096),
    GemmShape("atma9b_prefill", "mlp_project", 4096, 4096, 16384),
    GemmShape("atma9b_prefill", "lm_head", 4096, 50304, 4096),
)


def decode_shapes() -> tuple[GemmShape, ...]:
    shapes = []
    for batch in (512, 1024, 2048, 4096, 6144, 7168):
        shapes.extend((
            GemmShape("atma9b_decode", f"mlp_expand_b{batch}", batch, 32768, 4096),
            GemmShape("atma9b_decode", f"lm_head_b{batch}", batch, 50304, 4096),
        ))
    return tuple(shapes)


def training_shapes() -> tuple[GemmShape, ...]:
    shapes = []

    def add_linear(group: str, name: str, rows: int, in_features: int, out_features: int):
        # Forward, activation-gradient, and weight-gradient GEMMs for Y = X @ W.
        shapes.extend((
            GemmShape(group, f"{name}_fwd_m{rows}", rows, out_features, in_features),
            GemmShape(group, f"{name}_dx_m{rows}", rows, in_features, out_features),
            GemmShape(group, f"{name}_dw_m{rows}", in_features, out_features, rows),
        ))

    # Canonical D=1024 training: microbatch 8/16/32/64 at sequence length 1024.
    for rows in (8192, 16384, 32768, 65536):
        add_linear("train_378m", "mlp_expand", rows, 1024, 8192)
        add_linear("train_378m", "lm_head", rows, 1024, 50304)
    # 9.2B D=4096 training: microbatch 1/2/4/8 at sequence length 1024.
    for rows in (1024, 2048, 4096, 8192):
        add_linear("train_9b", "mlp_expand", rows, 4096, 32768)
        add_linear("train_9b", "lm_head", rows, 4096, 50304)
    return tuple(shapes)


def iterations_for(shape: GemmShape, target_flops: float, minimum: int, maximum: int) -> int:
    per_launch = 2 * shape.m * shape.n * shape.k
    return max(minimum, min(maximum, math.ceil(target_flops / per_launch)))


def measure_gemm(shape: GemmShape, warmup: int, target_flops: float) -> dict:
    a = torch.empty((shape.m, shape.k), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((shape.k, shape.n), device="cuda", dtype=torch.bfloat16)
    for _ in range(warmup):
        torch.mm(a, b)
    torch.cuda.synchronize()
    iterations = iterations_for(shape, target_flops, minimum=5, maximum=50)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        torch.mm(a, b)
        end.record()
    torch.cuda.synchronize()
    times_ms = sorted(start.elapsed_time(end) for start, end in zip(starts, ends))
    p50_ms = times_ms[len(times_ms) // 2]
    tflops = 2 * shape.m * shape.n * shape.k / (p50_ms * 1e-3) / 1e12
    result = asdict(shape)
    result.update(iterations=iterations, p50_ms=p50_ms,
                  p95_ms=times_ms[min(len(times_ms) - 1, int(0.95 * len(times_ms)))],
                  tflops=tflops)
    del a, b
    torch.cuda.empty_cache()
    return result


def measure_bandwidth(size_bytes: int, warmup: int) -> dict:
    src = torch.empty(size_bytes, device="cuda", dtype=torch.uint8)
    dst = torch.empty_like(src)
    src.fill_(1)
    for _ in range(warmup):
        dst.copy_(src)
    torch.cuda.synchronize()
    iterations = max(5, min(64, math.ceil((16 * 2**30) / size_bytes)))
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        dst.copy_(src)
        end.record()
    torch.cuda.synchronize()
    times_ms = sorted(start.elapsed_time(end) for start, end in zip(starts, ends))
    p50_ms = times_ms[len(times_ms) // 2]
    # Copy traffic reads and writes size_bytes.
    gbps = 2 * size_bytes / (p50_ms * 1e-3) / 1e9
    del src, dst
    torch.cuda.empty_cache()
    return {"size_bytes": size_bytes, "iterations": iterations,
            "p50_ms": p50_ms, "gbps": gbps}


def group_summary(results: list[dict]) -> dict:
    grouped = {}
    for result in results:
        grouped.setdefault(result["group"], []).append(result["tflops"])
    return {
        name: {
            "max_tflops": max(values),
            "geomean_tflops": statistics.geometric_mean(values),
            "min_tflops": min(values),
            "shapes": len(values),
        }
        for name, values in grouped.items()
    }


def nominal_peaks(name: str, memory_bytes: int) -> tuple[float | None, float | None]:
    lowered = name.lower()
    if "b200" in lowered:
        return 2250.0, 8000.0
    if "b300" in lowered:
        bandwidth = 8000.0 if "sxm" in lowered or memory_bytes >= 250 * 2**30 else 7100.0
        return 2250.0, bandwidth
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default="peak,atma9b_prefill,atma9b_decode,train_378m,train_9b")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--target-tflop", type=float, default=100.0,
                        help="approximately this many teraFLOPs per measured shape")
    parser.add_argument("--skip-bandwidth", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    requested = {value.strip() for value in args.groups.split(",") if value.strip()}
    all_shapes = PEAK_SHAPES + ATMA_9B_PREFILL + decode_shapes() + training_shapes()
    shapes = [shape for shape in all_shapes if shape.group in requested]
    if not shapes:
        raise SystemExit("no calibration groups selected")

    props = torch.cuda.get_device_properties(0)
    nominal_tf, nominal_gb = nominal_peaks(props.name, props.total_memory)
    gemms = []
    for shape in shapes:
        result = measure_gemm(shape, args.warmup, args.target_tflop * 1e12)
        gemms.append(result)
        print(f"{shape.group:16} {shape.name:24} "
              f"[{shape.m},{shape.n},{shape.k}] {result['tflops']:8.1f} TFLOP/s")
    bandwidth = [] if args.skip_bandwidth else [
        measure_bandwidth(size, args.warmup) for size in (256 * 2**20, 1 * 2**30, 4 * 2**30)
    ]
    result = {
        "status": "ok",
        "calibration_kind": "blackwell_shape_sweep",
        "gpu": props.name,
        "capability": [props.major, props.minor],
        "memory_bytes": props.total_memory,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nominal_bf16_tflops": nominal_tf,
        "nominal_hbm_gbps": nominal_gb,
        "gemms": gemms,
        "groups": group_summary(gemms),
        "bandwidth": bandwidth,
        "max_measured_bandwidth_gbps": max((x["gbps"] for x in bandwidth), default=None),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
