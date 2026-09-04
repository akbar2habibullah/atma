"""Generate separate primary-model and retention-intervention length curves."""

from __future__ import annotations

import math
import os
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

from re_evaluation_data import (
    ALL_MODELS,
    BABI_LENGTHS,
    LENGTHS,
    REFERENCE_MODELS,
    baseline_babilong,
    baseline_retrieval,
    baseline_longdoc,
    capped_babilong,
    capped_retrieval,
    capped_longdoc,
    mean_longdoc,
)


OUT = Path(os.environ.get("ATMA_FIGURE_OUT", Path(__file__).with_name("fig_main_results.pdf")))
MODELS = ("nope", "polar", "rope")
LABELS = {
    "nope": "NoPE", "polar": "Polar", "rope": "RoPE",
    "raven_native": "Raven Native", "atma_raven_titans": "Atma-Raven-Titans",
}
COLORS = {
    "nope": "#D55E00", "polar": "#0072B2", "rope": "#CC79A7",
    "raven_native": "#009E73", "atma_raven_titans": "#E69F00",
}
MARKERS = {
    "nope": "square", "polar": "circle", "rope": "diamond",
    "raven_native": "triangle", "atma_raven_titans": "down_triangle",
}


def draw_marker(c, kind, x, y, color, *, filled, size=1.7):
    c.setStrokeColor(color)
    c.setFillColor(color if filled else HexColor("#FFFFFF"))
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


def draw_axes(c, x0, y0, width, height, labels, ticks, project_y, title, ylabel, *, rotate=False):
    c.setStrokeColor(HexColor("#333333"))
    c.setLineWidth(0.55)
    c.line(x0, y0, x0, y0 + height)
    c.line(x0, y0, x0 + width, y0)
    c.setFont("Helvetica", 7.0)
    for value, label in ticks:
        y = project_y(value)
        c.setStrokeColor(HexColor("#D8DEE5"))
        c.setLineWidth(0.3)
        c.line(x0, y, x0 + width, y)
        c.setFillColor(HexColor("#444444"))
        c.drawRightString(x0 - 3, y - 1.8, label)
    xs = [x0 + width * i / (len(labels) - 1) for i in range(len(labels))]
    shown = {0, 2, 4, 6, len(labels) - 1} if len(labels) >= 8 else set(range(len(labels)))
    for i, (x, label) in enumerate(zip(xs, labels)):
        if i not in shown:
            continue
        c.setFillColor(HexColor("#444444"))
        if rotate:
            c.saveState()
            c.translate(x + 2, y0 - 8)
            c.rotate(-35)
            c.drawRightString(0, 0, label.upper())
            c.restoreState()
        else:
            c.drawCentredString(x, y0 - 9, label.upper())
    c.setFillColor(HexColor("#111111"))
    c.setFont("Helvetica-Bold", 7.5)
    c.drawCentredString(x0 + width / 2, y0 + height + 8, title)
    c.setFont("Helvetica", 7.0)
    c.saveState()
    c.translate(x0 - 21, y0 + height / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, ylabel)
    c.restoreState()
    c.drawCentredString(x0 + width / 2, y0 - (23 if rotate else 18), "Context length")
    return xs


def draw_series(c, xs, values, project_y, model, *, capped=False, reference=False):
    color = HexColor(COLORS[model])
    c.setStrokeColor(color)
    c.setLineWidth(1.35 if capped else (1.05 if reference else 0.9))
    if capped:
        c.setDash()
    elif reference:
        c.setDash(1.2, 2.0)
    else:
        c.setDash(3, 2)
    points = [(x, project_y(v)) for x, v in zip(xs, values)]
    p = c.beginPath()
    p.moveTo(*points[0])
    for point in points[1:]:
        p.lineTo(*point)
    c.drawPath(p, stroke=1, fill=0)
    c.setDash()
    for x, y in points:
        draw_marker(c, MARKERS[model], x, y, color, filled=capped or reference)


def generate_pdf(out_path, *, include_caps=False):
    baseline_r = baseline_retrieval(models=ALL_MODELS)
    retrieval = (baseline_r, capped_retrieval())
    baseline_b = baseline_babilong(models=ALL_MODELS)
    babilong = (baseline_b, capped_babilong())
    baseline_l = mean_longdoc(baseline_longdoc(models=ALL_MODELS))
    longdoc = (baseline_l, mean_longdoc(capped_longdoc()))

    page_w, page_h = 7.15 * 72, 3.1 * 72
    c = canvas.Canvas(str(out_path), pagesize=(page_w, page_h))

    legend_y = page_h - 9
    c.setFont("Helvetica", 6.2)
    legend_x = 42
    legend_widths = (48, 48, 46, 82, 121)
    for model, item_width in zip(ALL_MODELS, legend_widths):
        color = HexColor(COLORS[model])
        draw_marker(c, MARKERS[model], legend_x, legend_y, color, filled=True, size=1.8)
        c.setFillColor(HexColor("#222222"))
        c.drawString(legend_x + 5, legend_y - 2.1, LABELS[model])
        legend_x += item_width
    legend_y -= 13
    legend_x = 126
    for label, capped, reference in ((("Matched untouched", False, False), ("Matched one-head cap", True, False), ("Raven reference", False, True)) if include_caps else (("Matched attention", True, False), ("Raven reference", False, True))):
        c.setStrokeColor(HexColor("#444444"))
        c.setLineWidth(1.25 if capped else 0.9)
        if capped:
            c.setDash()
        elif reference:
            c.setDash(1.2, 2.0)
        else:
            c.setDash(3, 2)
        c.line(legend_x, legend_y, legend_x + 15, legend_y)
        c.setDash()
        draw_marker(c, "circle", legend_x + 7.5, legend_y, HexColor("#444444"), filled=capped or reference, size=1.5)
        c.setFillColor(HexColor("#222222"))
        c.drawString(legend_x + 19, legend_y - 2.1, label)
        legend_x += 105 if not capped else 123

    y0, height, width = 39, 124, 133
    x_positions = (34, 204, 374)

    py = lambda value: y0 + height * value / 100.0
    xs = draw_axes(c, x_positions[0], y0, width, height, LENGTHS,
                   [(0, "0"), (25, "25"), (50, "50"), (75, "75"), (100, "100")], py,
                   "(a) Retrieval", "Token accuracy (%)")
    for model in MODELS:
        for condition, capped in (zip(retrieval, (False, True)) if include_caps else [(baseline_r, True)]):
            draw_series(c, xs, [condition[model][length] for length in LENGTHS], py, model, capped=capped)
    for model in REFERENCE_MODELS:
        draw_series(c, xs, [baseline_r[model][length] for length in LENGTHS], py, model, reference=True)

    bpy = lambda value: y0 + height * value / 70.0
    bxs = draw_axes(c, x_positions[1], y0, width, height, BABI_LENGTHS,
                    [(0, "0"), (20, "20"), (40, "40"), (60, "60")], bpy,
                    "(b) BABILong", "Macro exact match (%)", rotate=True)
    c.saveState()
    c.setFillColor(HexColor("#718096"))
    c.setFillAlpha(0.08)
    c.rect(bxs[0] - 4, y0, bxs[2] - bxs[0] + 8, height, stroke=0, fill=1)
    c.restoreState()
    for model in MODELS:
        for condition, capped in (zip(babilong, (False, True)) if include_caps else [(baseline_b, True)]):
            draw_series(c, bxs, [condition[model][length] for length in BABI_LENGTHS], bpy, model, capped=capped)
    for model in REFERENCE_MODELS:
        draw_series(c, bxs, [baseline_b[model][length] for length in BABI_LENGTHS], bpy, model, reference=True)
    c.setFont("Helvetica", 6.5)
    c.setFillColor(HexColor("#555555"))
    c.drawString(x_positions[1] + 4, y0 + height - 8, "Shaded: adaptation lengths")

    log_min, log_max = math.log2(0.75), math.log2(10.0)
    lpy = lambda value: y0 + height * (math.log2(value) - log_min) / (log_max - log_min)
    lxs = draw_axes(c, x_positions[2], y0, width, height, LENGTHS,
                    [(1, "1"), (2, "2"), (4, "4"), (8, "8")], lpy,
                    "(c) Long-document", "Mean bits per byte (log)")
    for model in MODELS:
        for condition, capped in (zip(longdoc, (False, True)) if include_caps else [(baseline_l, True)]):
            draw_series(c, lxs, [condition[model][length] for length in LENGTHS], lpy, model, capped=capped)
    for model in REFERENCE_MODELS:
        draw_series(c, lxs, [baseline_l[model][length] for length in LENGTHS], lpy, model, reference=True)
    c.setFont("Helvetica", 6.5)
    c.setFillColor(HexColor("#555555"))
    c.drawRightString(x_positions[2] + width, y0 + height - 8, "Lower is better")

    c.showPage()
    c.save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    generate_pdf(OUT)
    generate_pdf(OUT.with_name("fig_retention_results.pdf"), include_caps=True)
