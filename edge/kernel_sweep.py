from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


BENCH_RE = re.compile(
    r"\[edge kernel bench\] flash_(?P<kernel>\w+) runs_s=(?P<runs_s>[-+0-9.eE]+) "
    r"tokens_s=(?P<tokens_s>[-+0-9.eE]+) replay_s=(?P<replay_s>[-+0-9.eE]+) "
    r"first_call_s=(?P<first_call_s>[-+0-9.eE]+) capture_s=(?P<capture_s>[-+0-9.eE]+) "
    r"checksum=(?P<checksum>[-+0-9.eE]+)"
)


@dataclass
class SweepCase:
    kernel: str
    heads: int
    tokens: int
    head_dim: int
    window: int
    iters: int
    gdn_variant: str = "naive"
    chunk_size: int = 0


def ints(values: list[str]) -> list[int]:
    out: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def profile_defaults(profile: str) -> dict[str, list[int]]:
    if profile != "rx6700xt":
        raise ValueError(f"unknown profile {profile!r}")
    return {
        "heads": [4, 8],
        "tokens": [16, 32, 48, 64, 96, 128],
        "head_dim": [16, 32],
    }


def make_cases(args) -> list[SweepCase]:
    defaults = profile_defaults(args.profile)
    heads = ints(args.heads) if args.heads else defaults["heads"]
    tokens = ints(args.tokens) if args.tokens else defaults["tokens"]
    head_dims = ints(args.head_dim) if args.head_dim else defaults["head_dim"]
    kernels = args.kernel if args.kernel != ["both"] else ["polar", "gdn"]
    gdn_variants = args.gdn_variant
    chunk_sizes = ints(args.chunk_size) if args.chunk_size else [16, 24, 32]
    cases: list[SweepCase] = []
    for kernel in kernels:
        for h in heads:
            for d in head_dims:
                for t in tokens:
                    window = min(args.window or t, t)
                    iters = max(1, args.iters if args.iters is not None else max(5, min(50, 1024 // t)))
                    if kernel == "gdn":
                        for variant in gdn_variants:
                            if variant == "chunked":
                                for chunk_size in chunk_sizes:
                                    cases.append(SweepCase(kernel=kernel, heads=h, tokens=t, head_dim=d, window=window, iters=iters, gdn_variant=variant, chunk_size=min(chunk_size, t)))
                            else:
                                cases.append(SweepCase(kernel=kernel, heads=h, tokens=t, head_dim=d, window=window, iters=iters, gdn_variant=variant, chunk_size=0))
                    else:
                        cases.append(SweepCase(kernel=kernel, heads=h, tokens=t, head_dim=d, window=window, iters=iters))
    return cases


def read_done(path: Path) -> set[tuple[str, int, int, int, int, str, int]]:
    done: set[tuple[str, int, int, int, int, str, int]] = set()
    if not path.exists() or path.stat().st_size == 0:
        return done
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                done.add((
                    row["kernel"],
                    int(row["heads"]),
                    int(row["tokens"]),
                    int(row["head_dim"]),
                    int(row["window"]),
                    row.get("gdn_variant") or "naive",
                    int(row.get("chunk_size") or 0),
                ))
    return done


def append_csv(path: Path, row: dict) -> None:
    fields = [
        "timestamp",
        "profile",
        "device",
        "dtype",
        "kernel",
        "heads",
        "tokens",
        "head_dim",
        "window",
        "iters",
        "gdn_variant",
        "chunk_size",
        "status",
        "wall_s",
        "runs_s",
        "tokens_s",
        "replay_s",
        "first_call_s",
        "capture_s",
        "checksum",
        "returncode",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def run_case(args, case: SweepCase) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "edge.kernel_bench",
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--kernel",
        case.kernel,
        "--heads",
        str(case.heads),
        "--tokens",
        str(case.tokens),
        "--head-dim",
        str(case.head_dim),
        "--window",
        str(case.window),
        "--iters",
        str(case.iters),
        "--skip-eager",
    ]
    if case.kernel == "gdn":
        cmd.extend(["--gdn-variant", case.gdn_variant])
        if case.gdn_variant == "chunked":
            cmd.extend(["--chunk-size", str(case.chunk_size)])
    start = time.perf_counter()
    base = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "profile": args.profile,
        "device": args.device,
        "dtype": args.dtype,
        **asdict(case),
    }
    try:
        timeout = None if args.case_timeout_s <= 0 else args.case_timeout_s
        proc = subprocess.run(cmd, cwd=args.cwd, text=True, capture_output=True, timeout=timeout)
        wall = time.perf_counter() - start
    except subprocess.TimeoutExpired as exc:
        return {
            **base,
            "status": "compile_timeout",
            "wall_s": time.perf_counter() - start,
            "returncode": "",
            "error": f"timeout after {args.case_timeout_s}s",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }

    match = BENCH_RE.search(proc.stdout)
    if proc.returncode == 0 and match:
        return {
            **base,
            "status": "ok",
            "wall_s": wall,
            "returncode": proc.returncode,
            **{key: float(value) for key, value in match.groupdict().items() if key != "kernel"},
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    return {
        **base,
        "status": "compile_error" if proc.returncode != 0 else "parse_error",
        "wall_s": wall,
        "returncode": proc.returncode,
        "error": (proc.stderr or proc.stdout)[-2000:],
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep standalone edge flash kernel shapes")
    parser.add_argument("--profile", default="rx6700xt", choices=["rx6700xt"])
    parser.add_argument("--device", default="cl")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--kernel", nargs="+", default=["both"], choices=["polar", "gdn", "both"])
    parser.add_argument("--heads", nargs="*", help="space or comma separated list; default profile values")
    parser.add_argument("--tokens", nargs="*", help="space or comma separated list; default profile values")
    parser.add_argument("--head-dim", nargs="*", help="space or comma separated list; default profile values")
    parser.add_argument("--window", type=int, default=None, help="default: equal to tokens for each case")
    parser.add_argument("--iters", type=int, default=None, help="default scales down with token count")
    parser.add_argument("--gdn-variant", nargs="+", default=["naive"], choices=["naive", "chunked"])
    parser.add_argument("--chunk-size", nargs="*", help="chunk sizes for --gdn-variant chunked; default 16 24 32")
    parser.add_argument("--case-timeout-s", type=int, default=1800, help="0 means no per-case timeout")
    parser.add_argument("--out", default="edge/results/kernel_sweep_rx6700xt.csv")
    parser.add_argument("--jsonl", default="edge/results/kernel_sweep_rx6700xt.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cwd", default=".")
    args = parser.parse_args()

    out_path = Path(args.out)
    jsonl_path = Path(args.jsonl)
    done = read_done(out_path) if args.resume else set()
    cases = make_cases(args)
    if args.limit > 0:
        cases = cases[:args.limit]

    print(
        f"[edge kernel sweep] profile={args.profile} device={args.device} dtype={args.dtype} "
        f"cases={len(cases)} timeout_s={args.case_timeout_s} out={out_path}"
    )
    for idx, case in enumerate(cases, 1):
        key = (case.kernel, case.heads, case.tokens, case.head_dim, case.window, case.gdn_variant, case.chunk_size)
        if key in done:
            print(f"[edge kernel sweep] skip {idx}/{len(cases)} {case}")
            continue
        print(f"[edge kernel sweep] run {idx}/{len(cases)} {case}", flush=True)
        row = run_case(args, case)
        append_csv(out_path, row)
        append_jsonl(jsonl_path, row)
        print(
            f"[edge kernel sweep] {row['status']} kernel={case.kernel} heads={case.heads} "
            f"tokens={case.tokens} head_dim={case.head_dim} variant={case.gdn_variant} chunk={case.chunk_size} "
            f"wall_s={float(row.get('wall_s') or 0):.2f} "
            f"tokens_s={row.get('tokens_s', '')}"
        )


if __name__ == "__main__":
    main()
