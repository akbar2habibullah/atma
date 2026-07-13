"""Budget-aware, one-command NVIDIA B200/B300 profiling session.

The runner uses a fresh process for each workload, preserves every command and
stdout/stderr in an artifact directory, and stops before the paid-time budget.
Use --dry-run on a CPU machine to validate the exact command plan.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
MIXED = "64,96,128,256,512,768,1024,1536"


def command_plan(phase: str, decode_batches: list[int],
                 train_microbatches: list[int] | None = None,
                 prefill_batches: list[int] | None = None,
                 train_hidden_size: int = 1024,
                 train_layers: int = 16) -> list[dict]:
    py = PYTHON
    train_microbatches = train_microbatches or ([1, 2, 4, 8] if train_hidden_size >= 4096
                                                 else [8, 16, 32, 64])
    prefill_batches = prefill_batches or [8, 16, 32, 64]
    smoke = [
        dict(name="cuda_tests", argv=[py, "-m", "pytest", "tests/test_decode_kernels.py",
                                      "tests/test_dense_prefill.py", "-q"]),
        dict(name="smoke_prefill", argv=[py, "-m", "scripts.stress_inference", "--mode",
             "prefill", "--hidden-size", "1024", "--layers", "16", "--batch", "2",
             "--prompt-length", "128", "--warmup", "1", "--iterations", "2"]),
        dict(name="smoke_decode", argv=[py, "-m", "scripts.stress_inference", "--mode",
             "decode", "--hidden-size", "1024", "--layers", "16", "--batch", "32",
             "--context-length", "128", "--warmup", "1", "--iterations", "2"]),
    ]
    calibration = [
        dict(name="blackwell_tensor_calibration", argv=[
            py, "-m", "scripts.calibrate_blackwell"]),
    ]
    calibration += [
        dict(name=f"stress_prefill_dense_b{batch}", argv=[
            py, "-m", "scripts.stress_inference", "--mode", "prefill",
            "--batch", str(batch), "--prompt-length", "512", "--warmup", "1",
            "--iterations", "5", "--include-sampler"])
        for batch in prefill_batches
    ]
    calibration += [
        dict(name=f"stress_decode_b{batch}", argv=[py, "-m", "scripts.stress_inference",
             "--mode", "decode", "--batch", str(batch), "--context-length", "512",
             "--warmup", "2", "--iterations", "20", "--include-sampler"])
        for batch in decode_batches
    ]
    benchmark = calibration + [
        dict(name=f"polar_{profile}", env={"ATMA_POLAR_TRITON_PROFILE": profile},
             argv=[py, "-m", "scripts.bench_kernel_efficiency", "--only", "b",
                   "--warmup", "10", "--iterations", "30"])
        for profile in ("l4", "small", "large")
    ]
    benchmark += [
        dict(name="canonical_mixed", argv=[py, "-m", "scripts.bench_kernel_efficiency",
             "--full-model", "mixed", "--warmup", "2", "--iterations", "10"]),
        dict(name="stress_prefill_mixed", argv=[py, "-m", "scripts.stress_inference", "--mode",
             "prefill", "--lengths", MIXED, "--warmup", "1", "--iterations", "10",
             "--include-sampler"]),
    ]
    trace = [
        dict(name="nsys_prefill_mixed", optional_tool="nsys", argv=[
            "nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi", "--stop-on-range-end=true",
            "--force-overwrite=true", "--output={output}", py, "-m",
            "scripts.stress_inference", "--mode", "prefill", "--lengths", MIXED,
            "--warmup", "1", "--iterations", "2", "--cuda-profiler-range"]),
        dict(name="nsys_decode_b512", optional_tool="nsys", argv=[
            "nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi", "--stop-on-range-end=true",
            "--force-overwrite=true", "--output={output}", py, "-m",
            "scripts.stress_inference", "--mode", "decode", "--batch", "512",
            "--context-length", "512", "--warmup", "1", "--iterations", "2",
            "--cuda-profiler-range"]),
        dict(name="ncu_polar_mixed", optional_tool="ncu", argv=[
            "ncu", "--profile-from-start", "off", "--target-processes", "all",
            "--section", "SpeedOfLight", "--section", "MemoryWorkloadAnalysis",
            "--section", "Occupancy", "--launch-count", "1", "--force-overwrite",
            "--export", "{output}", py, "-m", "scripts.bench_kernel_efficiency",
            "--only", "b", "--distribution", "mixed", "--cuda-profiler-range"]),
    ]
    training = [
        dict(name=f"training_mfu_d{train_hidden_size}_l{train_layers}_mb{microbatch}", argv=[
            py, "-m", "scripts.bench_training_mfu", "--microbatch", str(microbatch),
            "--hidden-size", str(train_hidden_size), "--layers", str(train_layers),
            "--seq-length", "1024", "--grad-accum", "1", "--warmup", "2",
            "--iterations", "5", "--measure-peak", "--allow-conv-fallback"])
        for microbatch in train_microbatches
    ]
    phases = {"smoke": smoke, "calibrate": calibration, "benchmark": benchmark,
              "trace": trace, "training": training}
    if phase == "all":
        return smoke + benchmark + training + trace
    return phases.get(phase, [])


def capture(cmd: list[str], timeout: float = 30) -> str:
    try:
        return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=timeout).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"


def metadata() -> dict:
    cuda_probe = (
        "import json,torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
        "p=torch.cuda.get_device_properties(0); x=torch.ones((128,128),device='cuda',dtype=torch.bfloat16); "
        "y=x@x; torch.cuda.synchronize(); "
        "print(json.dumps(dict(gpu=p.name, capability=[p.major,p.minor], memory_bytes=p.total_memory, "
        "smem_optin=getattr(p,'shared_memory_per_block_optin',None), torch=torch.__version__, "
        "torch_cuda=torch.version.cuda, bf16_matmul_ok=bool(torch.isfinite(y).all()))))"
    )
    dependency_probe = (
        "import importlib.metadata as m,importlib.util as u,json; "
        "v=lambda n: m.version(n) if u.find_spec(n.replace('-','_')) else None; "
        "print(json.dumps(dict(triton=v('triton'),pytest=v('pytest'),"
        "fla_module=bool(u.find_spec('fla')),kernels_module=bool(u.find_spec('kernels')))))"
    )
    return {
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "git_head": capture(["git", "rev-parse", "HEAD"]),
        "git_status": capture(["git", "status", "--short"]),
        "nvidia_smi": capture(["nvidia-smi", "-q"]),
        "cuda_probe": capture([PYTHON, "-c", cuda_probe], timeout=60),
        "dependency_probe": capture([PYTHON, "-c", dependency_probe]),
        "nsys_version": capture(["nsys", "--version"]),
        "ncu_version": capture(["ncu", "--version"]),
    }


def run_one(item: dict, output_dir: Path, deadline: float, reserve_seconds: float) -> dict:
    name = item["name"]
    tool = item.get("optional_tool")
    if tool and shutil.which(tool) is None:
        return {"name": name, "status": "skipped", "reason": f"{tool} not installed"}
    remaining = deadline - time.monotonic() - reserve_seconds
    if remaining <= 0:
        return {"name": name, "status": "skipped", "reason": "budget reserve reached"}
    argv = [arg.format(output=str(output_dir / name)) for arg in item["argv"]]
    env = os.environ.copy()
    env.update(item.get("env", {}))
    env.setdefault("PYTHONUNBUFFERED", "1")
    log_path = output_dir / f"{name}.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(argv)}\n")
        log.flush()
        try:
            completed = subprocess.run(argv, cwd=ROOT, env=env, text=True, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=remaining)
            status = "ok" if completed.returncode == 0 else "failed"
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            status, returncode = "timeout", None
    return {"name": name, "status": status, "returncode": returncode,
            "elapsed_seconds": round(time.monotonic() - started, 3), "log": str(log_path)}


def select_polar_profile(output_dir: Path) -> dict:
    scores = {}
    for profile in ("l4", "small", "large"):
        path = output_dir / f"polar_{profile}.log"
        if not path.exists():
            continue
        values = [float(value) for value in re.findall(
            r"^\s*grouped\s+p50=([0-9.]+)", path.read_text(encoding="utf-8"), re.MULTILINE)]
        if values:
            scores[profile] = {"sum_grouped_p50_ms": sum(values), "samples": values}
    selected = min(scores, key=lambda name: scores[name]["sum_grouped_p50_ms"]) if scores else "large"
    return {"selected": selected, "scores": scores,
            "fallback": not bool(scores), "criterion": "minimum sum of grouped p50 latency"}


def build_summary(output_dir: Path, metadata_record: dict, results: list[dict]) -> dict:
    measurements = {}
    for result in results:
        path_value = result.get("log")
        if not path_value:
            continue
        path = Path(path_value)
        if not path.exists():
            continue
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and ("mode" in value or "mfu_hybrid_nominal_pct" in value
                                            or "calibration_kind" in value):
                measurements[result["name"]] = value
                break
    tuning_path = output_dir / "tuning.json"
    saturation = {}
    for label, prefix in (("prefill", "stress_prefill_dense_b"),
                          ("decode", "stress_decode_b")):
        points = []
        for name, value in measurements.items():
            if name.startswith(prefix) and value.get("status") == "ok":
                points.append({"batch": int(name.removeprefix(prefix)),
                               "throughput_tok_s": value["throughput_tok_s"],
                               "p50_ms": value["p50_ms"],
                               "peak_allocated_gib": value["peak_allocated_gib"]})
        if points:
            points.sort(key=lambda point: point["batch"])
            maximum = max(point["throughput_tok_s"] for point in points)
            knee = next(point for point in points
                        if point["throughput_tok_s"] >= 0.95 * maximum)
            saturation[label] = {"maximum_throughput_tok_s": maximum,
                                 "knee_95pct": knee, "points": points}
    return {
        "cuda_probe": metadata_record["cuda_probe"],
        "tuning": json.loads(tuning_path.read_text(encoding="utf-8"))
                  if tuning_path.exists() else None,
        "measurements": measurements,
        "saturation": saturation,
        "runs": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "smoke", "calibrate", "benchmark",
                                            "training", "trace", "all"),
                        default="all")
    parser.add_argument("--budget-minutes", type=float, default=50.0)
    parser.add_argument("--reserve-minutes", type=float, default=5.0)
    parser.add_argument("--decode-batches", default="512,1024,2048,4096,6144,7168")
    parser.add_argument("--prefill-batches", default="8,16,32,64")
    parser.add_argument("--train-microbatches",
                        help="default: 8,16,32,64 for D=1024; 1,2,4,8 for D>=4096")
    parser.add_argument("--train-hidden-size", type=int, default=1024)
    parser.add_argument("--train-layers", type=int, default=16)
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    batches = [int(value) for value in args.decode_batches.split(",") if value]
    train_microbatches = ([int(value) for value in args.train_microbatches.split(",") if value]
                          if args.train_microbatches else None)
    prefill_batches = [int(value) for value in args.prefill_batches.split(",") if value]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir or ROOT / "profile_results" / stamp).resolve()
    plan = command_plan(args.phase, batches, train_microbatches, prefill_batches,
                        args.train_hidden_size, args.train_layers)
    if args.dry_run:
        print(json.dumps({"output_dir": str(output_dir), "plan": plan}, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = metadata()
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if "unavailable:" in meta["cuda_probe"] or "Traceback" in meta["cuda_probe"]:
        raise SystemExit(f"CUDA preflight failed; see {output_dir / 'metadata.json'}")
    if args.phase == "preflight":
        print(output_dir)
        return
    deadline = time.monotonic() + args.budget_minutes * 60
    results = []
    selected_profile = None
    for item in plan:
        if selected_profile and not item["name"].startswith("polar_"):
            item = dict(item)
            item["env"] = dict(item.get("env", {}), ATMA_POLAR_TRITON_PROFILE=selected_profile)
        result = run_one(item, output_dir, deadline, args.reserve_minutes * 60)
        results.append(result)
        print(json.dumps(result), flush=True)
        if result["status"] in ("timeout",) or result.get("reason") == "budget reserve reached":
            break
        if item["name"] in {"cuda_tests", "smoke_prefill", "smoke_decode"} \
                and result["status"] != "ok":
            print("Correctness gate failed; stopping before expensive benchmarks.", file=sys.stderr)
            break
        if item["name"].startswith("training_mfu_") and result["status"] != "ok":
            print("Training MFU gate failed; stopping the microbatch sweep.", file=sys.stderr)
            break
        if item["name"] == "polar_large":
            tuning = select_polar_profile(output_dir)
            selected_profile = tuning["selected"]
            (output_dir / "tuning.json").write_text(
                json.dumps(tuning, indent=2), encoding="utf-8")
            print(json.dumps({"polar_tuning": tuning}), flush=True)
    (output_dir / "manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(build_summary(output_dir, meta, results), indent=2), encoding="utf-8")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
