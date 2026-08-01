"""Run generation, likelihood, long-document, or serving benchmarks.

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


def _dataset_revisions(path):
    if not path:
        return {}
    import json

    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    records = payload.get("datasets", payload)
    return {
        dataset_id: (record.get("resolved_revision") if isinstance(record, dict) else record)
        for dataset_id, record in records.items()
    }


def main():
    ap = argparse.ArgumentParser(description="ATMA benchmark harness.")
    ap.add_argument(
        "--benchmark",
        default="retrieval",
        choices=["babilong", "retrieval", "base", "longdoc", "serving"],
    )
    ap.add_argument("--model", required=True, help="checkpoint dir or weights.pt path")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="BABILong qa tasks, retrieval kinds, or base benchmark tasks")
    ap.add_argument("--lengths", nargs="+", default=None,
                    help="length configs (babilong: dataset subset names; retrieval: token targets)")
    ap.add_argument("--depths", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="retrieval only: needle depth fractions")
    ap.add_argument("--samples", type=int, default=100, help="samples per cell")
    ap.add_argument("--serving_samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234,
                    help="retrieval sample seed (same seed yields paired items across models)")
    ap.add_argument("--dataset", default="RMT-team/babilong-1k-samples")
    ap.add_argument("--dataset_revision", default=None)
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="longdoc dataset aliases: pg19 proof_pile finepdfs")
    ap.add_argument("--dataset_revisions", default=None,
                    help="JSON checkpoint manifest or dataset-id to revision mapping")
    ap.add_argument("--haystack", default=None,
                    help="retrieval only: HF text dataset for real-text NIAH (default: synthetic filler)")
    ap.add_argument("--haystack_revision", default=None,
                    help="retrieval only: immutable Hugging Face dataset revision")
    ap.add_argument("--max_tokens", type=int, default=16)
    ap.add_argument("--decode_tokens", type=int, default=32, help="serving decode length")
    ap.add_argument("--limit", type=int, default=None, help="base-task examples per task")
    ap.add_argument("--batch_size", type=int, default=8, help="direct-scoring batch size")
    ap.add_argument("--scoring_max_length", type=int, default=None)
    ap.add_argument("--target_tokens", type=int, default=256)
    ap.add_argument("--num_docs", type=int, default=8)
    ap.add_argument("--max_scan", type=int, default=100000)
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

    if args.tasks is None:
        if args.benchmark == "babilong":
            args.tasks = ["qa1"]
        elif args.benchmark == "retrieval":
            args.tasks = ["passkey", "niah"]
        elif args.benchmark == "base":
            from benchmarks.base_tasks import BASE_TASK_SPECS
            args.tasks = list(BASE_TASK_SPECS)
        else:
            args.tasks = []
    if args.lengths is None:
        args.lengths = (
            ["0k", "1k", "2k", "4k", "8k", "16k", "32k"]
            if args.benchmark == "babilong"
            else ["2k", "8k", "32k", "64k", "128k", "256k"]
        )
    if args.datasets is None:
        args.datasets = ["pg19", "proof_pile", "finepdfs"]

    out = args.out or f"benchmarks/logs/{args.benchmark}_{int(time.time())}.log"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fh = open(out, "a", buffering=1)

    def log(s):
        print(s)
        fh.write(str(s) + "\n")

    log(f"[run] benchmark={args.benchmark} model={args.model} tasks={args.tasks} "
        f"lengths={args.lengths} samples={args.samples}")

    model = None
    scorer = None
    try:
        if args.benchmark in {"babilong", "retrieval"}:
            from benchmarks.model import EvalModel

            max_model_len = args.max_model_len or _infer_max_model_len(
                args.lengths, args.max_tokens
            )
            max_num_batched_tokens = args.max_num_batched_tokens or max(
                max_model_len, 16384
            )
            model = EvalModel(
                args.model,
                max_tokens=args.max_tokens,
                strict=args.strict,
                max_model_len=max_model_len,
                max_num_seqs=args.max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
            )

        if args.benchmark == "babilong":
            from benchmarks.babilong import emit_log, run_babilong

            res = run_babilong(
                model, args.tasks, args.lengths, num_samples=args.samples,
                dataset_id=args.dataset, dataset_revision=args.dataset_revision,
                max_tokens=args.max_tokens, log_fn=log,
            )
            emit_log(fh, model, res)
        elif args.benchmark == "retrieval":
            from benchmarks.retrieval import emit_log, run_retrieval

            kinds = [kind for kind in args.tasks if kind in ("passkey", "niah")] or ["passkey"]
            res = run_retrieval(
                model, kinds, args.lengths, args.depths, num_samples=args.samples,
                max_tokens=args.max_tokens, seed=args.seed, haystack=args.haystack,
                haystack_revision=args.haystack_revision, log_fn=log,
            )
            emit_log(fh, model, res)
        elif args.benchmark == "base":
            from benchmarks.base_tasks import emit_log, run_base_tasks
            from benchmarks.scoring import DirectScorer

            scorer = DirectScorer(
                args.model, max_length=args.scoring_max_length or 2048,
                batch_size=args.batch_size
            )
            res = run_base_tasks(
                scorer, args.tasks, limit=args.limit,
                dataset_revisions=_dataset_revisions(args.dataset_revisions),
                batch_size=args.batch_size, log_fn=log,
            )
            emit_log(fh, res)
        elif args.benchmark == "longdoc":
            from benchmarks.longdoc import emit_log, run_longdoc
            from benchmarks.scoring import DirectScorer

            scorer = DirectScorer(
                args.model,
                max_length=args.scoring_max_length or (
                    max(_parse_len(length) for length in args.lengths) + args.target_tokens
                ),
                batch_size=1,
            )
            res = run_longdoc(
                scorer, args.datasets, args.lengths, target_tokens=args.target_tokens,
                num_docs=args.num_docs, max_scan=args.max_scan,
                dataset_revisions=_dataset_revisions(args.dataset_revisions), log_fn=log,
            )
            emit_log(fh, res)
        elif args.benchmark == "serving":
            from benchmarks.serving import emit_log, run_serving

            res = run_serving(
                args.model, args.lengths, decode_tokens=args.decode_tokens,
                samples=args.serving_samples, max_num_seqs=args.max_num_seqs,
                max_num_batched_tokens=args.max_num_batched_tokens,
                strict=args.strict, log_fn=log,
            )
            emit_log(fh, res)
        log(f"[run] done in {res['elapsed_s']}s -> {out}")
    finally:
        if model is not None:
            model.close()
        if scorer is not None:
            scorer.close()
        fh.close()


if __name__ == "__main__":
    main()
