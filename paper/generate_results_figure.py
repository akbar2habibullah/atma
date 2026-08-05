"""Generate the main 10B-token retrieval and BABILong result figure.

The figure is derived only from the archived benchmark artifacts committed in
``benchmarks/logs``.  Run from the repository root with::

    python paper/generate_results_figure.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ATMA_FIGURE_OUT", Path(__file__).with_name("fig_main_results.pdf")))

LENGTHS = ["2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k"]
BABI_LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k"]
MODELS = ["polar", "nope", "rope", "raven_native", "atma_raven_titans"]
LABELS = {
    "polar": "Polar",
    "nope": "NoPE",
    "rope": "RoPE",
    "raven_native": "Raven",
    "atma_raven_titans": "Atma-Raven+Titans",
}
COLORS = {
    "polar": "#0072B2",
    "nope": "#D55E00",
    "rope": "#CC79A7",
    "raven_native": "#009E73",
    "atma_raven_titans": "#E69F00",
}
MARKERS = {
    "polar": "circle",
    "nope": "square",
    "rope": "diamond",
    "raven_native": "triangle",
    "atma_raven_titans": "down_triangle",
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def load_retrieval() -> dict[str, dict[str, list[float]]]:
    path = ROOT / "benchmarks" / "logs" / "atma_10b" / "benchmark_matrix.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("benchmark") != "retrieval" or row.get("metric") != "token_accuracy":
            continue
        if "/smoke_" in row.get("source_log", ""):
            continue
        grouped[(row["model"], row["length"], str(row["depth"]))].append(float(row["value"]))

    result: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for model in MODELS:
        for length in LENGTHS:
            depth_means = [mean(grouped[(model, length, depth)]) for depth in ("0.1", "0.5", "0.9")]
            result[model][length] = depth_means
    return result


def load_babilong() -> dict[str, list[float]]:
    root = ROOT / "benchmarks" / "logs" / "babilong_2k_ft" / "hub"
    result: dict[str, list[float]] = {}
    for model in MODELS:
        path = root / model / "babilong_full_eval_result.json"
        macro = json.loads(path.read_text(encoding="utf-8"))["macro_average"]
        result[model] = [float(macro[length]) for length in BABI_LENGTHS]
    return result


def draw_marker(c: canvas.Canvas, kind: str, x: float, y: float, size: float = 1.65) -> None:
    if kind == "circle":
        c.circle(x, y, size, stroke=1, fill=1)
    elif kind == "square":
        c.rect(x - size, y - size, 2 * size, 2 * size, stroke=1, fill=1)
    else:
        p = c.beginPath()
        if kind == "diamond":
            points = [(x, y + 1.25 * size), (x + size, y), (x, y - 1.25 * size), (x - size, y)]
        elif kind == "triangle":
            points = [(x, y + 1.25 * size), (x + size, y - size), (x - size, y - size)]
        else:
            points = [(x, y - 1.25 * size), (x + size, y + size), (x - size, y + size)]
        p.moveTo(*points[0])
        for point in points[1:]:
            p.lineTo(*point)
        p.close()
        c.drawPath(p, stroke=1, fill=1)


def draw_axes(c, x0, y0, width, height, labels, ymax, ylabel, title, rotate=False):
    c.setStrokeColor(HexColor("#333333"))
    c.setLineWidth(0.6)
    c.line(x0, y0, x0, y0 + height)
    c.line(x0, y0, x0 + width, y0)
    c.setFont("Helvetica", 6.5)
    for value in range(0, int(ymax) + 1, 20):
        y = y0 + height * value / ymax
        c.setStrokeColor(HexColor("#D0D0D0"))
        c.setLineWidth(0.35)
        c.line(x0, y, x0 + width, y)
        c.setFillColor(HexColor("#444444"))
        c.drawRightString(x0 - 4, y - 2, str(value))
    xs = [x0 + width * i / (len(labels) - 1) for i in range(len(labels))]
    for x, label in zip(xs, labels):
        c.setFillColor(HexColor("#444444"))
        if rotate:
            c.saveState()
            c.translate(x + 2, y0 - 10)
            c.rotate(-32)
            c.drawRightString(0, 0, label.upper())
            c.restoreState()
        else:
            c.drawCentredString(x, y0 - 12, label.upper())
    c.setFillColor(HexColor("#111111"))
    c.setFont("Helvetica-Bold", 8.2)
    c.drawCentredString(x0 + width / 2, y0 + height + 9, title)
    c.setFont("Helvetica", 7.0)
    c.saveState()
    c.translate(x0 - 26, y0 + height / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, ylabel)
    c.restoreState()
    c.drawCentredString(x0 + width / 2, y0 - (27 if rotate else 23), "Context length")
    return xs, lambda value: y0 + height * value / ymax


def draw_series(c, xs, values, model, project_y):
    color = HexColor(COLORS[model])
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(1.25)
    c.setDash(4, 2) if model in {"raven_native", "atma_raven_titans"} else c.setDash()
    points = [(x, project_y(value)) for x, value in zip(xs, values)]
    p = c.beginPath()
    p.moveTo(*points[0])
    for point in points[1:]:
        p.lineTo(*point)
    c.drawPath(p, stroke=1, fill=0)
    c.setDash()
    for x, y in points:
        draw_marker(c, MARKERS[model], x, y)


def main() -> None:
    retrieval = load_retrieval()
    babilong = load_babilong()
    page_width, page_height = 7.15 * 72, 3.1 * 72
    c = canvas.Canvas(str(OUT), pagesize=(page_width, page_height))

    legend_widths = [52, 48, 48, 55, 112]
    legend_x = (page_width - sum(legend_widths)) / 2
    legend_y = page_height - 10
    c.setFont("Helvetica", 6.8)
    for model, item_width in zip(MODELS, legend_widths):
        color = HexColor(COLORS[model])
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(1.25)
        c.setDash(4, 2) if model in {"raven_native", "atma_raven_titans"} else c.setDash()
        c.line(legend_x, legend_y, legend_x + 13, legend_y)
        c.setDash()
        draw_marker(c, MARKERS[model], legend_x + 6.5, legend_y, 1.6)
        c.setFillColor(HexColor("#222222"))
        c.drawString(legend_x + 17, legend_y - 2.2, LABELS[model])
        legend_x += item_width

    plot_y, plot_w, plot_h = 42, 205, 132
    rxs, rpy = draw_axes(c, 38, plot_y, plot_w, plot_h, LENGTHS, 100,
                         "Teacher-forced token accuracy (%)", "(a) Retrieval depth sweep")
    for model in MODELS:
        by_depth = [retrieval[model][length] for length in LENGTHS]
        lower = [min(values) for values in by_depth]
        upper = [max(values) for values in by_depth]
        c.saveState()
        c.setFillColor(HexColor(COLORS[model]))
        c.setFillAlpha(0.10)
        p = c.beginPath()
        p.moveTo(rxs[0], rpy(lower[0]))
        for x, value in zip(rxs[1:], lower[1:]):
            p.lineTo(x, rpy(value))
        for x, value in reversed(list(zip(rxs, upper))):
            p.lineTo(x, rpy(value))
        p.close()
        c.drawPath(p, stroke=0, fill=1)
        c.restoreState()
        draw_series(c, rxs, [mean(values) for values in by_depth], model, rpy)
    c.setFont("Helvetica", 5.9)
    c.setFillColor(HexColor("#555555"))
    c.drawString(43, plot_y + plot_h - 10.0, "Line: mean; band: 10/50/90% depth range")

    bxs, bpy = draw_axes(c, 296, plot_y, plot_w, plot_h, BABI_LENGTHS, 70,
                         "Macro exact-match accuracy (%)", "(b) BABILong after matched 2K adaptation", True)
    c.saveState()
    c.setFillColor(HexColor("#777777"))
    c.setFillAlpha(0.07)
    half_step = (bxs[1] - bxs[0]) / 2
    c.rect(bxs[0] - half_step, plot_y, bxs[2] - bxs[0] + half_step, plot_h, stroke=0, fill=1)
    c.restoreState()
    for model in MODELS:
        draw_series(c, bxs, babilong[model], model, bpy)
    c.setFont("Helvetica", 5.9)
    c.setFillColor(HexColor("#555555"))
    c.drawString(301, plot_y + plot_h - 10.0, "Shaded: adaptation lengths")

    c.showPage()
    c.save()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
