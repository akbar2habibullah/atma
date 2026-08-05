"""Generate the grouped vertical bar chart for downstream zero-shot benchmark comparison.

Renders task-by-task grouped vertical bars across the 8 control benchmarks + Mean score
for NoPE, RoPE, Polar, Atma-Raven-Titans, and Raven Native.
"""

from __future__ import annotations

import os
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ATMA_DOWNSTREAM_CANDLE_OUT", Path(__file__).with_name("fig_downstream_candle.pdf")))

MODELS = ["nope", "rope", "polar", "atma_raven_titans", "raven_native"]
LABELS = {
    "nope": "NoPE",
    "rope": "RoPE",
    "polar": "Polar (ATMA)",
    "atma_raven_titans": "Atma-Raven+Titans",
    "raven_native": "Raven Native",
}
COLORS = {
    "nope": "#D55E00",
    "rope": "#CC79A7",
    "polar": "#0072B2",
    "atma_raven_titans": "#E69F00",
    "raven_native": "#009E73",
}

TASKS = ["lambada", "hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge", "openbookqa", "boolq", "mean"]
TASK_LABELS = ["LAMBADA", "HellaSwag", "PIQA", "WinoGrande", "ARC-E", "ARC-C", "OBQA", "BoolQ", "MEAN"]

DATA = {
    "nope": [30.16, 37.88, 66.27, 51.93, 50.53, 32.44, 31.60, 60.37, 45.15],
    "rope": [30.25, 37.19, 66.76, 50.67, 49.12, 29.43, 32.40, 60.95, 44.60],
    "polar": [28.47, 36.19, 66.87, 51.62, 49.12, 26.09, 31.80, 56.21, 43.29],
    "atma_raven_titans": [26.00, 34.12, 65.56, 52.49, 48.07, 27.76, 29.00, 61.41, 43.05],
    "raven_native": [27.48, 33.56, 64.69, 51.70, 49.12, 27.09, 28.80, 61.93, 43.05],
}


def generate_pdf(out_path: Path):
    page_w = 504.0
    page_h = 165.0

    c = canvas.Canvas(str(out_path), pagesize=(page_w, page_h))

    # Title
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(HexColor("#1A2530"))
    c.drawString(12, page_h - 14, "Zero-Shot Downstream Language Modeling Benchmarks Comparison")

    # Legend at Top Right / Center
    legend_widths = [48, 45, 65, 82, 70]
    legend_x = (page_w - sum(legend_widths)) / 2.0
    legend_y = page_h - 26.0
    c.setFont("Helvetica-Bold", 6.5)

    for m_key, item_w in zip(MODELS, legend_widths):
        color = HexColor(COLORS[m_key])
        c.setFillColor(color)
        c.setStrokeColor(color)
        c.rect(legend_x, legend_y - 2.5, 8, 8, stroke=0, fill=1)
        c.setFillColor(HexColor("#222222"))
        c.drawString(legend_x + 11, legend_y - 1.5, LABELS[m_key])
        legend_x += item_w

    x0 = 34.0
    y0 = 22.0
    width = page_w - 46.0
    height = page_h - 58.0

    ymin = 0.0
    ymax = 75.0

    def proj_y(val: float) -> float:
        return y0 + height * (val - ymin) / (ymax - ymin)

    # Y-axis Grid
    c.setFont("Helvetica", 6.0)
    for yval in range(0, 80, 15):
        yp = proj_y(float(yval))
        c.setStrokeColor(HexColor("#E2E8F0"))
        c.setLineWidth(0.4)
        c.line(x0, yp, x0 + width, yp)
        c.setFillColor(HexColor("#444444"))
        c.drawRightString(x0 - 4, yp - 2, f"{yval}%")

    # Y-axis Label
    c.saveState()
    c.setFont("Helvetica-Bold", 6.5)
    c.setFillColor(HexColor("#2D3748"))
    c.translate(x0 - 22, y0 + height / 2.0)
    c.rotate(90)
    c.drawCentredString(0, 0, "Zero-Shot Accuracy (%)")
    c.restoreState()

    # Draw Grouped Vertical Bars
    num_tasks = len(TASKS)
    num_models = len(MODELS)

    group_w = width / num_tasks
    bar_w = (group_w - 8.0) / num_models

    for t_idx, t_lbl in enumerate(TASK_LABELS):
        gx = x0 + t_idx * group_w + 4.0

        # Separator line before MEAN
        if t_lbl == "MEAN":
            c.setStrokeColor(HexColor("#CBD5E0"))
            c.setLineWidth(0.8)
            c.setDash(2, 2)
            c.line(gx - 4.0, y0, gx - 4.0, y0 + height)
            c.setDash()

        for m_idx, m_key in enumerate(MODELS):
            val = DATA[m_key][t_idx]
            bx = gx + m_idx * bar_w
            by = y0
            bh = proj_y(val) - y0

            color = HexColor(COLORS[m_key])
            c.setFillColor(color)
            c.setStrokeColor(color)
            c.rect(bx, by, bar_w - 0.5, bh, stroke=0, fill=1)

            # Draw value on top of MEAN bars
            if t_lbl == "MEAN":
                c.setFont("Helvetica-Bold", 4.5)
                c.setFillColor(color)
                c.drawCentredString(bx + bar_w / 2.0, by + bh + 2.5, f"{val:.1f}")

        # X-axis Task Label
        c.setFont("Helvetica-Bold" if t_lbl == "MEAN" else "Helvetica", 6.2)
        c.setFillColor(HexColor("#0F2942") if t_lbl == "MEAN" else HexColor("#2D3748"))
        c.drawCentredString(gx + (group_w - 8.0) / 2.0, y0 - 10, t_lbl)

    c.save()
    print(f"wrote {out_path.resolve()}")


if __name__ == "__main__":
    generate_pdf(OUT)
