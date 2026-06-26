"""Parse ablation .log files into one results.json (one record per run).

    python -m ablation.parse_logs --log_dir ablation/logs --out ablation/results.json

Each .log carries three delimited JSON blocks (ABLATION_CONFIG_JSON / _CURVE_ / _EVAL_),
optionally _ERROR_. We extract them robustly (ignoring the surrounding human-readable text)
and flatten the eval into a `metrics` dict keyed for the dashboard leaderboards.
"""
import argparse
import glob
import json
import os
import re

AXES = ["attn_type", "reg_mode", "distractor", "memory", "window"]
_BLOCK = lambda name: re.compile(rf"==={name}===\s*\n(.*?)\n===END===", re.DOTALL)
_BLOCKS = {k: _BLOCK(v) for k, v in {
    "config": "ABLATION_CONFIG_JSON", "curve": "ABLATION_CURVE_JSON",
    "eval": "ABLATION_EVAL_JSON", "error": "ABLATION_ERROR_JSON"}.items()}


def _extract(text, key):
    m = _BLOCKS[key].search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _flatten_metrics(curve, ev):
    """Flat name -> value for leaderboards. Lower/higher direction lives in metric_catalog()."""
    m = {}
    if curve:
        m["final_val_loss"] = curve[-1].get("val_loss")
    if ev:
        for L, v in (ev.get("clean_ppl") or {}).items():
            if v is not None:
                m[f"clean_ppl_{L}"] = v
        for L, v in (ev.get("junk_ppl") or {}).items():
            if v is not None:
                m[f"junk_ppl_{L}"] = v
        for L, v in (ev.get("clean_bpb") or {}).items():
            if v is not None:
                m[f"clean_bpb_{L}"] = v
        for L, v in (ev.get("junk_bpb") or {}).items():
            if v is not None:
                m[f"junk_bpb_{L}"] = v
        for L, v in (ev.get("clean_bits_per_gpt2_token") or {}).items():
            if v is not None:
                m[f"clean_bits_gpt2tok_{L}"] = v
        for L, v in (ev.get("junk_bits_per_gpt2_token") or {}).items():
            if v is not None:
                m[f"junk_bits_gpt2tok_{L}"] = v
        for d, v in (ev.get("needle") or {}).items():
            if v is not None:
                m[f"needle_acc_{d}"] = v.get("acc")
                m[f"needle_ce_{d}"] = v.get("ce")
        for k in ("mfu_final", "train_elapsed_s", "num_params"):
            if ev.get(k) is not None:
                m[k] = ev[k]
    return m


def parse_log(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    cfg = _extract(text, "config")
    curve = _extract(text, "curve") or []
    ev = _extract(text, "eval")
    err = _extract(text, "error")
    run_id = (cfg or {}).get("run_id") or os.path.basename(path)[:-len(".log")]
    rec = {"run_id": run_id, "config": cfg, "curve": curve, "eval": ev,
           "error": err, "metrics": _flatten_metrics(curve, ev)}
    for ax in AXES:
        rec[ax] = (cfg or {}).get(ax)
    rec["status"] = "done" if ev else ("error" if err else ("running" if curve else "unknown"))
    return rec


def main():
    ap = argparse.ArgumentParser(description="Parse ablation logs -> results.json")
    ap.add_argument("--log_dir", default="ablation/logs")
    ap.add_argument("--out", default="ablation/results.json")
    args = ap.parse_args()

    logs = sorted(glob.glob(os.path.join(args.log_dir, "*.log")))
    records = [parse_log(p) for p in logs]
    json.dump(records, open(args.out, "w"), indent=1)

    by_status = {}
    for r in records:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print(f"parsed {len(records)} logs -> {args.out}")
    for s, n in sorted(by_status.items()):
        print(f"  {s}: {n}")


if __name__ == "__main__":
    main()
