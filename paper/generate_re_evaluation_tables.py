"""Generate complete raw diagnostic and re-evaluation tables for ICLR 2027."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARAMS_FILE = ROOT / "gamma_diagnostics" / "results" / "parameters" / "gamma_parameters.json"
SWEEP_FILE = ROOT / "gamma_diagnostics" / "results" / "all_clamp_sweep_262k.json"
RE_EVAL_FILE = ROOT / "gamma_diagnostics" / "results" / "re_evaluation" / "run-summary.json"
BASELINE_MATRIX = ROOT / "benchmarks" / "logs" / "atma_10b" / "benchmark_matrix.json"
OUT_FILE = ROOT / "paper" / "iclr2027" / "gamma_re_evaluation_tables.tex"

from re_evaluation_data import (
    BABI_LENGTHS,
    DATASETS,
    LENGTHS,
    MODELS,
    TASK_METRICS,
    baseline_babilong,
    baseline_downstream,
    baseline_longdoc,
    baseline_retrieval,
    capped_babilong,
    capped_downstream,
    capped_longdoc,
    capped_retrieval,
    mean_longdoc,
)

LABELS = {"nope": "NoPE", "polar": "Polar", "rope": "RoPE"}
TASK_LABELS = {
    "lambada": "LAMBADA", "hellaswag": "HellaSwag", "piqa": "PIQA",
    "winogrande": "WinoGrande", "arc_easy": "ARC-E", "arc_challenge": "ARC-C",
    "openbookqa": "OBQA", "boolq": "BoolQ",
}

def arrow(a, b, digits=1):
    return f"{a:.{digits}f} $\\rightarrow$ {b:.{digits}f}"

def table_env(body, caption, label, columns, size=r"\scriptsize", tabcolsep="2.4pt"):
    return "\n".join([
        r"\begin{table}[!ht]",
        r"\centering",
        size,
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        body,
        r"\bottomrule",
        r"\end{tabular}}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
        "",
    ])

def generate_parameter_scan_table():
    params = json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
    checkpoints = [
        ("atma-10b-L4-mbs4-nope__reg-baseline__distr-0__mem-1__win-0", "NoPE (L4, mbs4)", "Earlier L4 baseline"),
        ("atma-10b-L40S-mbs4-nope__reg-baseline__distr-0__mem-1__win-0", "NoPE (L40S, mbs4)", "Microbatch ablation"),
        ("atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0", "NoPE (L40S, mbs16)", "Promoted matched"),
        ("atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0", "Polar (L40S, mbs16)", "Promoted matched"),
        ("atma-10b-L40S-mbs16-rope__reg-baseline__distr-0__mem-1__win-0", "RoPE (L40S, mbs16)", "Matched negative control"),
        ("atma-10b-L40S-mbs16-atma-raven-titans__reg-baseline__distr-0__mem-1__win-0", "Atma-Raven-Titans", "Recurrent reference"),
    ]
    
    header = (
        r"\textbf{Checkpoint} & \textbf{Role} & \textbf{Outlier Block/Head} & "
        r"\textbf{Learned $b_\gamma$} & \textbf{Logit $z_0$} & \textbf{$\gamma_0$} & "
        r"\textbf{Max $H_{1/2}$ (tok)} & \textbf{Median $H_{1/2}$} & \textbf{Min $H_{1/2}$} & \textbf{$\|W_\gamma\|_2$} \\ \midrule"
    )
    rows = []
    for cp_key, name, role in checkpoints:
        cp_rows = [p for p in params if p["checkpoint"] == cp_key]
        cp_rows.sort(key=lambda x: x["total_zero_input_logit"], reverse=True)
        max_r = cp_rows[0]
        half_lives = sorted([r["half_life_tokens"] for r in cp_rows])
        med_hl = half_lives[len(half_lives) // 2]
        min_hl = half_lives[0]
        
        if max_r["half_life_tokens"] > 1e6:
            hl_str = f"\\textbf{{{max_r['half_life_tokens']/1e6:.2f}M}}"
        elif max_r["half_life_tokens"] > 1e3:
            hl_str = f"{max_r['half_life_tokens']/1e3:.1f}k"
        else:
            hl_str = f"{max_r['half_life_tokens']:.1f}"
            
        rows.append(
            f"{name} & {role} & Block {max_r['layer']}, H{max_r['head']} & "
            f"{max_r['learned_bias']:+.3f} & {max_r['total_zero_input_logit']:.3f} & "
            f"{max_r['gamma_zero_input']:.7f} & {hl_str} & {med_hl:.1f} & {min_hl:.1f} & {max_r['weight_l2']:.2f} \\\\"
        )
    return table_env(
        header + "\n" + "\n".join(rows),
        "Learned zero-input retention operating points across base checkpoints. Outlier half-lives reach astronomical values ($>3$M--$21$M tokens) on isolated Block 2 heads in L40S NoPE and Polar, whereas RoPE (max 682 tok) and Atma-Raven-Titans (max 41 tok) remain bounded without pathological drift.",
        "table:gamma_parameters_full",
        "llcrrrrrrr",
        size=r"\small",
        tabcolsep="3.5pt",
    )

def generate_clamp_sweep_table():
    sweep = json.loads(SWEEP_FILE.read_text(encoding="utf-8"))
    checkpoints = [
        ("ChavyvAkvar/atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0", "NoPE (L40S, mbs16)"),
        ("ChavyvAkvar/atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0", "Polar (L40S, mbs16)"),
        ("ChavyvAkvar/atma-10b-L40S-mbs16-rope__reg-baseline__distr-0__mem-1__win-0", "RoPE (L40S, mbs16)"),
        ("ChavyvAkvar/atma-10b-L40S-mbs4-nope__reg-baseline__distr-0__mem-1__win-0", "NoPE (L40S, mbs4)"),
        ("ChavyvAkvar/atma-10b-L4-mbs4-nope__reg-baseline__distr-0__mem-1__win-0", "NoPE (L4, mbs4)"),
    ]
    cond_order = ["baseline", "p90", "p99", "hl-256", "hl-512"]
    cond_labels = {
        "baseline": "Baseline (none)",
        "p90": "p90 quantile",
        "p99": "p99 quantile",
        "hl-256": "hl:256 (cap)",
        "hl-512": "hl:512 (cap)",
    }
    
    header = (
        r"\textbf{Model} & \textbf{Condition} & \multicolumn{4}{c}{\textbf{Clean NLL (nats)}} & \multicolumn{4}{c}{\textbf{Needle Cross-Entropy (nats)}} & \multicolumn{2}{c}{\textbf{Needle Acc. (\%)}} \\ "
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-12}"
        r"& & \textbf{2K} & \textbf{16K} & \textbf{64K} & \textbf{256K} & \textbf{2K} & \textbf{16K} & \textbf{64K} & \textbf{256K} & \textbf{2K} & \textbf{256K} \\ \midrule"
    )
    rows = []
    for cp_key, name in checkpoints:
        cp_data = sweep["checkpoints"][cp_key]
        for idx, cond in enumerate(cond_order):
            cdata = cp_data["conditions"][cond]
            clean = cdata["metrics"]["clean"]
            needle = cdata["metrics"]["needle"]["by_distance"]
            
            c2k = clean.get("2048", {}).get("loss_nats", 0.0)
            c16k = clean.get("16384", {}).get("loss_nats", 0.0)
            c64k = clean.get("65536", {}).get("loss_nats", 0.0)
            c256k = clean.get("262144", {}).get("loss_nats", 0.0)
            
            nce2k = needle.get("2048", {}).get("ce_nats", 0.0)
            nce16k = needle.get("16384", {}).get("ce_nats", 0.0)
            nce64k = needle.get("65536", {}).get("ce_nats", 0.0)
            nce256k = needle.get("262144", {}).get("ce_nats", 0.0)
            
            nacc2k = needle.get("2048", {}).get("accuracy_pct", 0.0)
            nacc256k = needle.get("262144", {}).get("accuracy_pct", 0.0)
            
            prefix = rf"\multirow{{5}}{{*}}{{{name}}}" if idx == 0 else ""
            c_label = cond_labels[cond]
            if cond == "hl-256" and cp_key in [
                "ChavyvAkvar/atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0",
                "ChavyvAkvar/atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0",
            ]:
                c_label = r"\textbf{" + c_label + r" (sel.)}"
                
            row_str = (
                f"{prefix} & {c_label} & {c2k:.3f} & {c16k:.3f} & {c64k:.3f} & {c256k:.3f} & "
                f"{nce2k:.3f} & {nce16k:.3f} & {nce64k:.3f} & {nce256k:.3f} & {nacc2k:.1f} & {nacc256k:.1f} \\\\"
            )
            rows.append(row_str)
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()
        
    return table_env(
        header + "\n" + "\n".join(rows),
        "Complete paired causal clamp sweep across quantile and half-life caps from 2K to 256K. Capping the single outlier head at hl:256 eliminates NoPE and Polar extrapolation collapse while satisfying the 2K short-context guardrail. In contrast, p99 fails because the 99th percentile remains corrupted by the outlier, and RoPE suffers short-range needle accuracy collapse (82.5\\% $\\rightarrow$ 50.0\\%) without long-range retrieval gain.",
        "table:gamma_sweep_full",
        "llrrrrrrrrrr",
        size=r"\scriptsize",
        tabcolsep="2.2pt",
    )

def generate_longdoc_dataset_breakdown_table():
    jobs = json.loads(RE_EVAL_FILE.read_text(encoding="utf-8"))["results"]
    capped_data = {
        job["model"]: job["result"]["results"]
        for job in jobs if job["family"] == "longdoc" and job["suite"] == "fixed-target"
    }
    base_longdoc = baseline_longdoc()
    
    header = (
        r"\textbf{Model} & \textbf{Dataset} & \textbf{2K} & \textbf{4K} & \textbf{8K} & \textbf{16K} & \textbf{32K} & \textbf{64K} & \textbf{128K} & \textbf{256K} \\ \midrule"
    )
    rows = []
    ds_labels = {"finepdfs": "FinePDFs", "pg19": "PG-19", "proof_pile": "Proof-Pile"}
    for model in MODELS:
        for idx, ds in enumerate(DATASETS):
            prefix = rf"\multirow{{3}}{{*}}{{{LABELS[model]}}}" if idx == 0 else ""
            cells = []
            for length in LENGTHS:
                b_val = base_longdoc[model][ds][length]
                c_val = float(capped_data[model][ds]["lengths"][length]["bits_per_byte"])
                cells.append(arrow(b_val, c_val, 3))
            rows.append(f"{prefix} & {ds_labels[ds]} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()
        
    return table_env(
        header + "\n" + "\n".join(rows),
        "Dataset-level fixed-target long-document likelihood curve, untouched $\\rightarrow$ capped (bits per byte; lower is better). NoPE catastrophic likelihood divergence on FinePDFs (11.666 $\\rightarrow$ 1.180), PG-19 (4.902 $\\rightarrow$ 1.224), and Proof-Pile (7.723 $\\rightarrow$ 2.382) is completely prevented across all lengths.",
        "table:gamma_longdoc_dataset_breakdown",
        "llrrrrrrrr",
        size=r"\scriptsize",
        tabcolsep="2.2pt",
    )

def generate_haystack_retrieval_breakdown_table():
    jobs = json.loads(RE_EVAL_FILE.read_text(encoding="utf-8"))["results"]
    capped_jobs = {(job["family"], job["model"], job["suite"]): job["result"] for job in jobs}
    
    # baseline by haystack
    base_rows = json.loads(BASELINE_MATRIX.read_text(encoding="utf-8"))["rows"]
    from collections import defaultdict
    base_grouped = defaultdict(list)
    for r in base_rows:
        if r.get("benchmark") != "retrieval" or r.get("metric") != "token_accuracy":
            continue
        if "/smoke_" in r.get("source_log", "") or r.get("model") not in MODELS:
            continue
        suite = r.get("suite")
        if suite in ("synthetic", "real"):
            base_grouped[(r["model"], suite, r["length"])].append(float(r["value"]))
    
    header = (
        r"\textbf{Model} & \textbf{Haystack} & \textbf{2K} & \textbf{4K} & \textbf{8K} & \textbf{16K} & \textbf{32K} & \textbf{64K} & \textbf{128K} & \textbf{256K} \\ \midrule"
    )
    rows = []
    for model in MODELS:
        for idx, suite in enumerate(["synthetic", "real"]):
            suite_name = "Synthetic" if suite == "synthetic" else "FinePDFs (Real)"
            prefix = rf"\multirow{{2}}{{*}}{{{LABELS[model]}}}" if idx == 0 else ""
            cells = []
            payload = capped_jobs[("retrieval", model, suite)]["results"]
            for length in LENGTHS:
                b_vals = base_grouped[(model, suite, length)]
                b_mean = sum(b_vals) / len(b_vals) if b_vals else 0.0
                c_vals = []
                for task in ("niah", "passkey"):
                    for depth in ("0.1", "0.5", "0.9"):
                        c_vals.append(float(payload[task][length][depth]))
                c_mean = sum(c_vals) / len(c_vals)
                cells.append(arrow(b_mean, c_mean, 1))
            rows.append(f"{prefix} & {suite_name} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()
        
    return table_env(
        header + "\n" + "\n".join(rows),
        "Teacher-forced retrieval token accuracy (\\%) by haystack type, untouched $\\rightarrow$ capped. Polar gains +9.1\\% synthetic and +16.3\\% FinePDFs retrieval at 256K, while NoPE FinePDFs retrieval remains 0.0\\% despite synthetic signal gains.",
        "table:gamma_haystack_retrieval_full",
        "llrrrrrrrr",
        size=r"\scriptsize",
        tabcolsep="2.2pt",
    )

def main():
    sections = ["% Generated by paper/generate_re_evaluation_tables.py; do not edit by hand.\n"]
    
    # 1. Raw Parameter Inspection Table
    sections.append(generate_parameter_scan_table())
    
    # 2. Raw Multi-Clamp Sweep Grid Table
    sections.append(generate_clamp_sweep_table())

    # 3. Downstream LM controls
    base_d, cap_d = baseline_downstream(), capped_downstream()
    header = r"\textbf{Model} & " + " & ".join(rf"\textbf{{{TASK_LABELS[t]}}}" for t in TASK_METRICS) + r" & \textbf{Mean} \\ \midrule"
    rows = []
    for model in MODELS:
        values = [arrow(base_d[model][task], cap_d[model][task], 2) for task in TASK_METRICS]
        values.append(arrow(sum(base_d[model].values()) / 8, sum(cap_d[model].values()) / 8, 2))
        rows.append(LABELS[model] + " & " + " & ".join(values) + r" \\")
    sections.append(table_env(header + "\n" + "\n".join(rows),
        "Complete short-context control, untouched $\\rightarrow$ capped (accuracy, \\%). The intervention changes no task mean materially ($<0.06$ points delta).",
        "table:gamma_downstream_full", "l" + "r" * 9))

    # 4. Retrieval curves (token and exact)
    bt, ct = baseline_retrieval("token_accuracy"), capped_retrieval("token_accuracy")
    be, ce = baseline_retrieval("exact_match"), capped_retrieval("exact_match")
    header = r"\textbf{Model} & " + " & ".join(rf"\textbf{{{length.upper()}}}" for length in LENGTHS) + r" \\ \midrule"
    for metric_name, before, after, label in (
        ("teacher-forced target-token accuracy", bt, ct, "table:gamma_retrieval_token_full"),
        ("exact five-token accuracy", be, ce, "table:gamma_retrieval_exact_full"),
    ):
        rows = []
        for model in MODELS:
            cells = [arrow(before[model][length], after[model][length], 1) for length in LENGTHS]
            rows.append(LABELS[model] + " & " + " & ".join(cells) + r" \\")
        sections.append(table_env(header + "\n" + "\n".join(rows),
            f"Complete retrieval curve, untouched $\\rightarrow$ capped, reported as {metric_name} (\\%). Each entry averages passkey and NIAH, synthetic and FinePDFs haystacks, and three depths.",
            label, "l" + "r" * 8))

    # 5. Haystack retrieval breakdown
    sections.append(generate_haystack_retrieval_breakdown_table())

    # 6. BABILong reasoning curve
    bb, cb = baseline_babilong(), capped_babilong()
    header = r"\textbf{Model} & " + " & ".join(rf"\textbf{{{length.upper()}}}" for length in BABI_LENGTHS) + r" \\ \midrule"
    rows = []
    for model in MODELS:
        rows.append(LABELS[model] + " & " + " & ".join(arrow(bb[model][length], cb[model][length], 0) for length in BABI_LENGTHS) + r" \\")
    sections.append(table_env(header + "\n" + "\n".join(rows),
        "Complete BABILong curve after adaptation, untouched $\\rightarrow$ capped (macro exact match, \\%). Clamping restores NoPE adapted reasoning to 35--57\\% across 8K--256K contexts (vs 0\\% untouched) and boosts Polar at 256K from 28\\% to 42\\%.",
        "table:gamma_babilong_full", "l" + "r" * 10))

    # 7. Mean BPB curve
    bl, cl = mean_longdoc(baseline_longdoc()), mean_longdoc(capped_longdoc())
    header = r"\textbf{Model} & " + " & ".join(rf"\textbf{{{length.upper()}}}" for length in LENGTHS) + r" \\ \midrule"
    rows = []
    for model in MODELS:
        rows.append(LABELS[model] + " & " + " & ".join(arrow(bl[model][length], cl[model][length], 3) for length in LENGTHS) + r" \\")
    sections.append(table_env(header + "\n" + "\n".join(rows),
        "Complete fixed-target likelihood curve, untouched $\\rightarrow$ capped (mean bits per byte; lower is better), averaged over FinePDFs, PG-19, and Proof-Pile.",
        "table:gamma_bpb_full", "l" + "r" * 8))

    # 8. Longdoc dataset-level breakdown
    sections.append(generate_longdoc_dataset_breakdown_table())

    OUT_FILE.write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {OUT_FILE}")

if __name__ == "__main__":
    main()
