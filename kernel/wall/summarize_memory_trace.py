"""Summarize a PyTorch CUDA allocator snapshot by live allocation source.

The input is a pickle produced by ``torch.cuda.memory._snapshot()`` while memory
history recording is enabled. The summary reconstructs live bytes through the
trace and reports the category breakdown at the global peak.
"""

from __future__ import annotations

import argparse
import collections
import pickle
from pathlib import Path


def _frame_text(frames):
    return "\n".join(f"{f.get('filename', '')}:{f.get('line', 0)}:{f.get('name', '')}" for f in frames or [])


def _category(frames):
    text = _frame_text(frames)
    if "/kernel/wall/" in text:
        return "wall_local_kernel"
    if "/wall-attention-release/wall_attn/" in text or "/site-packages/wall_attn/" in text:
        return "wall_upstream_kernel"
    if "/train/model.py" in text:
        if "_wall_attention" in text or "wall_fn" in text:
            return "wall_integration"
        return "train_model"
    if "/model/blocks.py" in text:
        return "memory_or_polar_blocks"
    if "/torch/_inductor/" in text or "/triton/" in text or "triton_" in text:
        return "inductor_triton"
    if "optimizer" in text or "adam" in text.lower() or "muon" in text.lower():
        return "optimizer"
    if "cross_entropy" in text or "nll_loss" in text:
        return "loss"
    if "empty_strided" in text or "empty" in text:
        return "unattributed_empty"
    return "other"


def _best_frame(frames):
    for frame in reversed(frames or []):
        filename = frame.get("filename", "")
        if filename.startswith("/home/sagemaker-user/atma") or "wall-attention-release" in filename:
            return f"{filename}:{frame.get('line', 0)}:{frame.get('name', '')}"
    for frame in reversed(frames or []):
        filename = frame.get("filename", "")
        if filename and filename != "??":
            return f"{filename}:{frame.get('line', 0)}:{frame.get('name', '')}"
    return "<no python frame>"


def _fmt_mib(n):
    return f"{n / 1024**2:.1f} MiB"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    snap = pickle.loads(args.snapshot.read_bytes())
    traces = snap.get("device_traces", [])
    events = [event for trace in traces for event in trace]
    events.sort(key=lambda e: e.get("time_us", 0))

    live = {}
    live_by_category = collections.Counter()
    peak_total = 0
    peak_by_category = collections.Counter()
    peak_time_us = 0
    alloc_totals = collections.Counter()
    alloc_counts = collections.Counter()
    alloc_by_site = collections.Counter()

    for event in events:
        action = event.get("action")
        addr = event.get("addr")
        size = int(event.get("size") or 0)
        if action == "alloc" and addr is not None:
            frames = event.get("frames") or []
            cat = _category(frames)
            site = _best_frame(frames)
            live[addr] = (size, cat, site)
            live_by_category[cat] += size
            alloc_totals[cat] += size
            alloc_counts[cat] += 1
            alloc_by_site[(cat, site)] += size
            total = sum(live_by_category.values())
            if total > peak_total:
                peak_total = total
                peak_by_category = live_by_category.copy()
                peak_time_us = int(event.get("time_us") or 0)
        elif action in {"free_requested", "free_completed"} and addr in live:
            size0, cat, _ = live.pop(addr)
            live_by_category[cat] -= size0
            if live_by_category[cat] <= 0:
                live_by_category.pop(cat, None)

    print(f"snapshot={args.snapshot}")
    print(f"events={len(events)}")
    print(f"peak_live={_fmt_mib(peak_total)} time_us={peak_time_us}")
    print()
    print("live breakdown at peak:")
    for cat, size in peak_by_category.most_common():
        print(f"  {cat:24s} {_fmt_mib(size)}")
    print()
    print("allocated bytes by category over whole trace:")
    for cat, size in alloc_totals.most_common():
        print(f"  {cat:24s} {_fmt_mib(size)} count={alloc_counts[cat]}")
    print()
    print(f"top {args.top} allocation sites by cumulative allocated bytes:")
    for (cat, site), size in alloc_by_site.most_common(args.top):
        print(f"  {_fmt_mib(size):>12s} {cat:24s} {site}")


if __name__ == "__main__":
    main()
