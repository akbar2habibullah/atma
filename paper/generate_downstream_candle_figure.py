"""Generate untouched task-level downstream controls; cap values remain in the appendix."""

from __future__ import annotations

import os
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

from re_evaluation_data import ALL_MODELS, TASK_METRICS, baseline_downstream


OUT = Path(os.environ.get("ATMA_DOWNSTREAM_CANDLE_OUT", Path(__file__).with_name("fig_downstream_candle.pdf")))
LABELS = {
    "nope": "NoPE", "polar": "Polar", "rope": "RoPE",
    "raven_native": "Raven Native", "atma_raven_titans": "Atma-Raven-Titans",
}
COLORS = {
    "nope": "#D55E00", "polar": "#0072B2", "rope": "#CC79A7",
    "raven_native": "#009E73", "atma_raven_titans": "#E69F00",
}
TASKS = tuple(TASK_METRICS)
TASK_LABELS = ("LAMBADA", "HellaSwag", "PIQA", "WinoGrande", "ARC-E", "ARC-C", "OBQA", "BoolQ", "MEAN")


def generate_pdf(out_path):
    baseline = baseline_downstream(models=ALL_MODELS)
    for model in ALL_MODELS:
        baseline[model]["mean"] = sum(baseline[model].values()) / len(TASKS)

    page_w, page_h = 504.0, 165.0
    c = canvas.Canvas(str(out_path), pagesize=(page_w, page_h))
    c.setFont("Helvetica-Bold", 9.5)
    c.setFillColor(HexColor("#1A2530"))
    c.drawString(12, page_h - 14, "Short-context controls: untouched checkpoints")

    legend_x, legend_y = 66, page_h - 26
    c.setFont("Helvetica", 7.2)
    legend_widths = (47, 47, 45, 74, 111)
    for model, item_width in zip(ALL_MODELS, legend_widths):
        c.setFillColor(HexColor(COLORS[model]))
        c.rect(legend_x, legend_y - 2, 7, 7, stroke=0, fill=1)
        c.setFillColor(HexColor("#222222"))
        c.drawString(legend_x + 10, legend_y - 1, LABELS[model])
        legend_x += item_width
    x0, y0 = 34.0, 22.0
    width, height = page_w - 46.0, page_h - 69.0
    project_y = lambda value: y0 + height * value / 75.0
    c.setFont("Helvetica", 7.2)
    for value in range(0, 76, 15):
        y = project_y(value)
        c.setStrokeColor(HexColor("#E2E8F0"))
        c.setLineWidth(0.4)
        c.line(x0, y, x0 + width, y)
        c.setFillColor(HexColor("#444444"))
        c.drawRightString(x0 - 4, y - 2, f"{value}%")
    c.saveState()
    c.setFont("Helvetica-Bold", 6.2)
    c.setFillColor(HexColor("#2D3748"))
    c.translate(x0 - 22, y0 + height / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Zero-shot accuracy (%)")
    c.restoreState()

    fields = TASKS + ("mean",)
    group_w = width / len(fields)
    slots = len(ALL_MODELS)
    bar_w = (group_w - 7.0) / slots
    for task_i, (task, label) in enumerate(zip(fields, TASK_LABELS)):
        gx = x0 + task_i * group_w + 3.5
        if task == "mean":
            c.setStrokeColor(HexColor("#CBD5E0"))
            c.setDash(2, 2)
            c.line(gx - 3.5, y0, gx - 3.5, y0 + height)
            c.setDash()
        for slot_i, model in enumerate(ALL_MODELS):
            value = baseline[model][task]
            x = gx + slot_i * bar_w
            c.setFillColor(HexColor(COLORS[model]))
            c.rect(x, y0, bar_w - 0.35, project_y(value) - y0, stroke=0, fill=1)
        c.setFont("Helvetica-Bold" if task == "mean" else "Helvetica", 7.2)
        c.setFillColor(HexColor("#0F2942") if task == "mean" else HexColor("#2D3748"))
        c.drawCentredString(gx + (group_w - 7.0) / 2, y0 - 10, label)
    c.save()
    print(f"wrote {out_path.resolve()}")


if __name__ == "__main__":
    generate_pdf(OUT)
