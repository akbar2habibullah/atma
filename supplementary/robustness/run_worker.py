"""Atomic, non-mutating worker for one supplementary config group."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _verify_external_dependencies(cfg: dict):
    expected = cfg.get("dependency_commits") or {}
    deps = json.loads((ROOT / "dependencies.json").read_text(encoding="utf-8"))
    for name, commit in expected.items():
        path = Path(deps[name]["checkout"])
        if not path.is_dir():
            raise RuntimeError(f"missing pinned dependency checkout: {path}")
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != commit:
            raise RuntimeError(f"{name} commit changed after approval: expected {commit}, found {actual}")
        tracked_dirty = subprocess.run(["git", "-C", str(path), "diff", "--quiet"]).returncode != 0
        staged_dirty = subprocess.run(["git", "-C", str(path), "diff", "--cached", "--quiet"]).returncode != 0
        if tracked_dirty or staged_dirty:
            raise RuntimeError(f"{name} has unrecorded tracked changes after approval")


def _claim(path: Path, metadata: dict) -> bool:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=Path, required=True)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--state_dir", type=Path, required=True)
    parser.add_argument("--ckpt_dir", type=Path, default=Path("checkpoints/supplementary_robustness"))
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--include", default="*.json", help="config filename glob, e.g. repl_seed1_*.json")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--reset_failed", action="store_true")
    parser.add_argument("--reset_running", action="store_true", help="clear stale running markers after verifying no worker is alive")
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    if args.reset_failed:
        for path in args.state_dir.glob("*.failed"):
            path.unlink()
        print("cleared failed markers")
        return
    if args.reset_running:
        paths = list(args.state_dir.glob("*.running"))
        for path in paths:
            path.unlink()
        print(f"cleared {len(paths)} running marker(s); only safe after checking the recorded host/pid")
        return

    env = dict(os.environ)
    env.setdefault("FLA_CUSTOM_OP", "1")
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    fla_source = Path("third_party/flash-linear-attention").resolve()
    if fla_source.is_dir():
        env["PYTHONPATH"] = str(fla_source) + os.pathsep + env.get("PYTHONPATH", "")

    host, pid = socket.gethostname(), os.getpid()
    while True:
        selected = None
        for config_path in sorted(args.config_dir.glob(args.include)):
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            if not cfg.get("enabled", True):
                continue
            if cfg.get("baseline_family") == "external" and not cfg.get("parameter_count_approved", False):
                continue
            run_id = cfg["run_id"]
            if any((args.state_dir / f"{run_id}.{suffix}").exists() for suffix in ("running", "done", "failed")):
                continue
            running = args.state_dir / f"{run_id}.running"
            if _claim(running, {"host": host, "pid": pid, "gpu": args.gpu, "started": time.time()}):
                selected = config_path, cfg, running
                break
        if selected is None:
            print("no runnable configs; disabled, unapproved, claimed, and completed configs are skipped")
            return

        config_path, cfg, running = selected
        run_id = cfg["run_id"]
        if cfg.get("baseline_family") == "external":
            try:
                _verify_external_dependencies(cfg)
            except Exception:
                running.unlink(missing_ok=True)
                raise
        runner = cfg["runner"]
        log_path = args.log_dir / f"{run_id}.log"
        env["PYTHONHASHSEED"] = str(cfg.get("init_seed", cfg.get("seed", 0)))
        cmd = [sys.executable, "-m", runner, "--config", str(config_path), "--log", str(log_path)]
        if runner in {"scaled_ablation.train", "external_baselines.train"}:
            cmd += ["--ckpt_dir", str(args.ckpt_dir)]
        started = time.perf_counter()
        rc = subprocess.run(cmd, env=env).returncode
        elapsed = time.perf_counter() - started
        suffix = "done" if rc == 0 else "failed"
        target = args.state_dir / f"{run_id}.{suffix}"
        os.replace(running, target)
        with target.open("w", encoding="utf-8") as handle:
            json.dump({"host": host, "pid": pid, "gpu": args.gpu, "rc": rc, "elapsed_s": elapsed}, handle, indent=2)
        print(f"{run_id}: {suffix} after {elapsed / 3600:.2f}h")
        if args.once:
            return


if __name__ == "__main__":
    main()
