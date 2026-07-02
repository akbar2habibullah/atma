"""Run a benchmark through the production inference adapter.

    python -m benchmarks.run --benchmark babilong --model checkpoints/<run_id> \
        --tasks qa1 qa2 --lengths 0k 1k 2k 4k 8k 16k 32k --samples 100 \
        --out benchmarks/logs/babilong_<run_id>.log

BABILong needs a *fine-tuned* checkpoint to be meaningful (see benchmarks/babilong.py).
"""
import argparse
import os
import time


def _parse_len(s):
    s = str(s).lower().strip()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1024 * 1024)
    return int(s)


def _infer_max_model_len(lengths, max_tokens):
    vals = []
    for length in lengths:
        try:
            vals.append(_parse_len(length))
        except ValueError:
            pass
    return (max(vals) if vals else 65536) + max_tokens + 64


def main():
    ap = argparse.ArgumentParser(description="Atma benchmark harness (inference-interface).")
    ap.add_argument("--benchmark", default="babilong", choices=["babilong", "retrieval"])
    ap.add_argument("--model", required=True, help="checkpoint dir or weights.pt path")
    ap.add_argument("--tasks", nargs="+", default=["qa1"],
                    help="babilong: bAbI tasks (qa1..qa20). retrieval: kinds (passkey niah)")
    ap.add_argument("--lengths", nargs="+",
                    default=["0k", "1k", "2k", "4k", "8k", "16k", "32k"],
                    help="length configs (babilong: dataset subset names; retrieval: token targets)")
    ap.add_argument("--depths", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="retrieval only: needle depth fractions")
    ap.add_argument("--samples", type=int, default=100, help="samples per cell")
    ap.add_argument("--dataset", default="RMT-team/babilong-1k-samples")
    ap.add_argument("--haystack", default=None,
                    help="retrieval only: HF text dataset for real-text NIAH (default: synthetic filler)")
    ap.add_argument("--max_tokens", type=int, default=16)
    ap.add_argument("--max_model_len", type=int, default=None,
                    help="inference context budget; default inferred from --lengths")
    ap.add_argument("--max_num_seqs", type=int, default=16,
                    help="max concurrent sequences; lower values reduce Titans memory-state allocation")
    ap.add_argument("--max_num_batched_tokens", type=int, default=None,
                    help="prefill token budget; default is at least max_model_len")
    ap.add_argument("--out", default=None)
    ap.add_argument("--strict", action="store_true",
                    help="hard-fail if the benchmark inference path can't run this checkpoint")
    args = ap.parse_args()

    out = args.out or f"benchmarks/logs/{args.benchmark}_{int(time.time())}.log"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fh = open(out, "a", buffering=1)

    def log(s):
        print(s)
        fh.write(str(s) + "\n")

    from benchmarks.model import EvalModel
    log(f"[run] benchmark={args.benchmark} model={args.model} tasks={args.tasks} "
        f"lengths={args.lengths} samples={args.samples}")

    max_model_len = args.max_model_len or _infer_max_model_len(args.lengths, args.max_tokens)
    max_num_batched_tokens = args.max_num_batched_tokens or max(max_model_len, 16384)
    model = EvalModel(
        args.model,
        max_tokens=args.max_tokens,
        strict=args.strict,
        max_model_len=max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    if args.benchmark == "babilong":
        from benchmarks.babilong import run_babilong, emit_log
        res = run_babilong(model, args.tasks, args.lengths, num_samples=args.samples,
                           dataset_id=args.dataset, max_tokens=args.max_tokens, log_fn=log)
        emit_log(fh, model, res)
    elif args.benchmark == "retrieval":
        from benchmarks.retrieval import run_retrieval, emit_log
        kinds = [k for k in args.tasks if k in ("passkey", "niah")] or ["passkey"]
        res = run_retrieval(model, kinds, args.lengths, args.depths, num_samples=args.samples,
                            max_tokens=args.max_tokens, haystack=args.haystack, log_fn=log)
        emit_log(fh, model, res)
    log(f"[run] done in {res['elapsed_s']}s -> {out}")
    fh.close()


if __name__ == "__main__":
    main()
