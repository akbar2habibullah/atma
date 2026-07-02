"""Worker for Raven bridge/scaled configs."""
from __future__ import annotations

import argparse
import glob
import os
import socket
import subprocess
import sys
import time


def _pending(config_dir):
    return sorted(p for p in glob.glob(os.path.join(config_dir, "*.json")))


def _run_id(json_path):
    return os.path.basename(json_path)[:-len(".json")]


def reset_claims(config_dir):
    n = 0
    for suffix in ("*.json.claimed.*", "*.json.failed"):
        for p in glob.glob(os.path.join(config_dir, suffix)):
            base = p.split(".json")[0] + ".json"
            if not os.path.exists(base):
                os.rename(p, base)
                n += 1
            else:
                os.remove(p)
    print(f"reset {n} claim(s) back to pending in {config_dir}")


def main():
    ap = argparse.ArgumentParser(description="Raven worker: claim + train configs.")
    ap.add_argument("--config_dir", default="raven_baseline/configs")
    ap.add_argument("--log_dir", default="raven_baseline/logs")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--no_save_ckpt", action="store_true")
    args = ap.parse_args()

    if args.reset:
        reset_claims(args.config_dir)
        return

    os.makedirs(args.log_dir, exist_ok=True)
    host, pid = socket.gethostname(), os.getpid()
    env = dict(os.environ)
    env.setdefault("FLA_CUSTOM_OP", "1")
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    tag = f"[raven-worker {host}:{pid} gpu={args.gpu}]"

    while True:
        claimed = None
        for cand in _pending(args.config_dir):
            target = f"{cand}.claimed.{host}.{pid}"
            try:
                os.rename(cand, target)
                claimed = target
                break
            except OSError:
                continue
        if claimed is None:
            print(f"{tag} no pending configs - exiting.")
            return

        run_id = _run_id(claimed.split(".json")[0] + ".json")
        log_path = os.path.join(args.log_dir, f"{run_id}.log")
        print(f"{tag} >>> {run_id}", flush=True)
        t0 = time.perf_counter()
        cmd = [
            sys.executable, "-m", "raven_baseline.train",
            "--config", claimed, "--log", log_path, "--ckpt_dir", args.ckpt_dir,
        ]
        if args.no_save_ckpt:
            cmd.append("--no_save_ckpt")
        rc = subprocess.run(cmd, env=env).returncode
        dt = time.perf_counter() - t0
        if rc == 0:
            os.rename(claimed, f"{args.config_dir}/{run_id}.json.done")
            print(f"{tag} <<< {run_id} OK ({dt/3600:.2f}h)", flush=True)
        else:
            os.rename(claimed, f"{args.config_dir}/{run_id}.json.failed")
            with open(os.path.join(args.log_dir, f"{run_id}.error"), "a", encoding="utf-8") as ef:
                ef.write(f"{run_id} exited rc={rc} after {dt:.1f}s\n")
            print(f"{tag} <<< {run_id} FAILED rc={rc} ({dt:.1f}s)", flush=True)
        if args.once:
            return


if __name__ == "__main__":
    main()

