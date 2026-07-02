"""Multi-GPU scaled-ablation worker: claim configs from a shared directory and train them.

    # one worker per GPU (run several, one per device); they coordinate via atomic file claims
    python -m scaled_ablation.run_worker --config_dir scaled_ablation/configs --log_dir scaled_ablation/logs --gpu 0
    python -m scaled_ablation.run_worker --config_dir scaled_ablation/configs --log_dir scaled_ablation/logs --gpu 1
    ...

    python -m scaled_ablation.run_worker --reset      # un-claim crashed/failed configs back to pending

Claiming is by os.rename (atomic on a shared FS), so no two workers take the same cell:
    <run_id>.json                      pending
    <run_id>.json.claimed.<host>.<pid> in progress
    <run_id>.json.done                 completed (a <run_id>.log was written)
    <run_id>.json.failed               crashed (a <run_id>.error was written)
Glob `*.json` matches only pending cells, so a re-run resumes automatically. Across machines,
share the dir (NFS) or split the 120 files into per-machine folders.
"""
import argparse
import glob
import os
import socket
import subprocess
import sys
import time


def _pending(config_dir):
    # only pristine <id>.json (claimed/done/failed have extra suffixes and won't match)
    return sorted(p for p in glob.glob(os.path.join(config_dir, "*.json")))


def _run_id(json_path):
    return os.path.basename(json_path)[:-len(".json")]


def reset_claims(config_dir):
    n = 0
    for suffix in ("*.json.claimed.*", "*.json.failed"):
        for p in glob.glob(os.path.join(config_dir, suffix)):
            base = p.split(".json")[0] + ".json"
            if not os.path.exists(base):
                os.rename(p, base); n += 1
            else:
                os.remove(p)
    print(f"reset {n} claim(s) back to pending in {config_dir}")


def main():
    ap = argparse.ArgumentParser(description="Ablation worker: claim + train configs.")
    ap.add_argument("--config_dir", default="scaled_ablation/configs")
    ap.add_argument("--log_dir", default="scaled_ablation/logs")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--gpu", default=None, help="CUDA device index for this worker (sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--once", action="store_true", help="train a single config then exit")
    ap.add_argument("--reset", action="store_true", help="un-claim crashed/failed configs and exit")
    ap.add_argument("--no_save_ckpt", action="store_true", help="disable checkpoint persistence")
    ap.add_argument("--push_to_hub", action="store_true", help="upload each saved checkpoint folder to Hugging Face")
    ap.add_argument("--hf_repo_prefix", default=None,
                    help="repo prefix for uploads, e.g. org/atma-10b; run id is appended")
    ap.add_argument("--hf_private", action="store_true", help="create/use private Hugging Face repos")
    args = ap.parse_args()

    if args.reset:
        reset_claims(args.config_dir)
        return

    os.makedirs(args.log_dir, exist_ok=True)
    host, pid = socket.gethostname(), os.getpid()
    env = dict(os.environ)
    env.setdefault("FLA_CUSTOM_OP", "1")          # compile-clean FLA path
    env.setdefault("ATMA_WALL_CUSTOM_OP", "1")    # compile-opaque Wall path; train/model.py falls back if unsupported
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    tag = f"[worker {host}:{pid} gpu={args.gpu}]"

    while True:
        claimed = None
        for cand in _pending(args.config_dir):
            target = f"{cand}.claimed.{host}.{pid}"
            try:
                os.rename(cand, target)               # atomic claim; loser raises and we try the next
                claimed = target
                break
            except OSError:
                continue
        if claimed is None:
            print(f"{tag} no pending configs — exiting.")
            return

        run_id = _run_id(claimed.split(".json")[0] + ".json")
        log_path = os.path.join(args.log_dir, f"{run_id}.log")
        print(f"{tag} >>> {run_id}", flush=True)
        t0 = time.perf_counter()
        cmd = [sys.executable, "-m", "scaled_ablation.train",
               "--config", claimed, "--log", log_path, "--ckpt_dir", args.ckpt_dir]
        if args.no_save_ckpt:
            cmd.append("--no_save_ckpt")
        if args.push_to_hub:
            if not args.hf_repo_prefix:
                raise SystemExit("--hf_repo_prefix is required with --push_to_hub")
            cmd += ["--push_to_hub", "--hf_repo_id", f"{args.hf_repo_prefix}-{run_id}"]
            if args.hf_private:
                cmd.append("--hf_private")
        rc = subprocess.run(cmd, env=env).returncode
        dt = time.perf_counter() - t0

        if rc == 0:
            os.rename(claimed, f"{args.config_dir}/{run_id}.json.done")
            print(f"{tag} <<< {run_id} OK ({dt/3600:.2f}h)", flush=True)
        else:
            os.rename(claimed, f"{args.config_dir}/{run_id}.json.failed")
            with open(os.path.join(args.log_dir, f"{run_id}.error"), "a") as ef:
                ef.write(f"{run_id} exited rc={rc} after {dt:.1f}s\n")
            print(f"{tag} <<< {run_id} FAILED rc={rc} ({dt:.1f}s)", flush=True)

        if args.once:
            return


if __name__ == "__main__":
    main()
