"""Generate the 2-panel serving latency resource scaling figure (fig_serving_scaling.pdf).

Panel (a): Prefill Latency (seconds) vs Context Length (2K -> 128K).
Panel (b): Decode Latency (ms/token) vs Context Length (2K -> 128K).
Highlights constant O(1) decode latency for recurrent Raven models vs linear O(L) KV cache decode latency for attention models.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ATMA_SERVING_SCALING_OUT", Path(__file__).with_name("fig_serving_scaling.pdf")))

LENGTHS = ["2k", "4k", "8k", "16k", "32k", "64k", "128k"]
LENGTH_LABELS = ["2K", "4K", "8K", "16K", "32K", "64K", "128K"]

MODELS = ["polar", "nope", "rope", "atma_raven_titans", "raven_native"]
LABELS = {
    "polar": "Polar",
    "nope": "NoPE",
    "rope": "RoPE",
    "atma_raven_titans": "Atma-Raven-Titans",
    "raven_native": "Raven Native",
}
COLORS = {
    "polar": "#0072B2",
    "nope": "#D55E00",
    "rope": "#CC79A7",
    "atma_raven_titans": "#E69F00",
    "raven_native": "#009E73",
}

# Empirical serving latency data from benchmarks/logs/atma_10b/serving_*.log
PREFILL_LATENCY_S = {
    "polar": [0.131, 0.031, 0.051, 0.101, 0.248, 0.601, 1.619],
    "nope": [0.157, 0.028, 0.049, 0.097, 0.230, 0.576, 1.473],
    "rope": [0.160, 0.028, 0.049, 0.096, 0.232, 0.566, 1.472],
    "atma_raven_titans": [0.198, 0.041, 0.059, 0.111, 0.247, 0.499, 0.999],
    "raven_native": [0.214, 0.054, 0.070, 0.139, 0.320, 0.672, 1.357],
}

DECODE_LATENCY_MS = {
    "polar": [2.27, 2.49, 2.88, 3.70, 5.66, 9.22, 16.30],
    "nope": [2.22, 2.45, 2.83, 3.68, 5.62, 9.22, 16.40],
    "rope": [2.34, 2.53, 2.95, 3.79, 5.73, 9.33, 16.51],
    "atma_raven_titans": [2.24, 2.25, 2.24, 2.24, 2.26, 2.25, 2.27],
    "raven_native": [3.26, 3.25, 3.24, 3.24, 3.25, 3.25, 3.27],
}


def draw_panel(c: canvas.Canvas, x0: float, y0: float, width: float, height: float,
               data_dict: dict[str, list[float]], ymax: float, ylabel: str, title: str):
    # Axes
    c.setStrokeColor(HexColor("#333333"))
    c.setLineWidth(0.6)
    c.line(x0, y0, x0, y0 + height)
    c.line(x0, y0, x0 + width, y0)

    # Y-axis Grid
    c.setFont("Helvetica", 6.0)
    step = ymax / 4.0
    for i in range(5):
        yval = i * step
        yp = y0 + height * (yval / ymax)
        c.setStrokeColor(HexColor("#E2E8F0"))
        c.setLineWidth(0.35)
        c.line(x0, yp, x0 + width, yp)
        c.setFillColor(HexColor("#444444"))
        c.drawRightString(x0 - 3, yp - 2, f"{yval:.1f}" if ymax < 5 else f"{int(round(yval))}")

    # X-axis Labels
    xs = [x0 + width * i / (len(LENGTHS) - 1) for i in range(len(LENGTHS))]
    for x, lbl in zip(xs, LENGTH_LABELS):
        c.setFillColor(HexColor("#444444"))
        c.drawCentredString(x, y0 - 10, lbl)

    # Title & Axis Labels
    c.setFillColor(HexColor("#1A2530"))
    c.setFont("Helvetica-Bold", 8.0)
    c.drawCentredString(x0 + width / 2.0, y0 + height + 8, title)

    c.saveState()
    c.setFont("Helvetica-Bold", 6.5)
    c.setFillColor(HexColor("#2D3748"))
    c.translate(x0 - 22, y0 + height / 2.0)
    c.rotate(90)
    c.drawCentredString(0, 0, ylabel)
    c.restoreState()

    c.setFont("Helvetica", 6.0)
    c.setFillColor(HexColor("#4A5568"))
    c.drawCentredString(x0 + width / 2.0, y0 - 18, "Context length")

    # Series lines
    for m_key in MODELS:
        color = HexColor(COLORS[m_key])
        vals = data_dict[m_key]
        points = [(x, y0 + height * (v / ymax)) for x, v in zip(xs, vals)]

        c.setStrokeColor(color)
        c.setLineWidth(1.2)
        if m_key in ("raven_native", "atma_raven_titans"):
            c.setDash(3, 2)
        else:
            c.setDash()

        p = c.beginPath()
        p.moveTo(*points[0])
        for pt in points[1:]:
            p.lineTo(*pt)
        c.drawPath(p, stroke=1, fill=0)
        c.setDash()

        # Dots
        c.setFillColor(color)
        for px_pt, py_pt in points:
            c.circle(px_pt, py_pt, 1.5, stroke=1, fill=1)


def generate_pdf(out_path: Path):
    page_w = 504.0
    page_h = 172.0

    c = canvas.Canvas(str(out_path), pagesize=(page_w, page_h))

    # Legend at Top
    legend_widths = [48, 45, 45, 80, 70]
    legend_x = (page_w - sum(legend_widths)) / 2.0
    legend_y = page_h - 10.0
    c.setFont("Helvetica-Bold", 6.5)

    for m_key, item_w in zip(MODELS, legend_widths):
        color = HexColor(COLORS[m_key])
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(1.2)
        if m_key in ("raven_native", "atma_raven_titans"):
            c.setDash(3, 2)
        else:
            c.setDash()

        c.line(legend_x, legend_y, legend_x + 12, legend_y)
        c.setDash()
        c.circle(legend_x + 6, legend_y, 1.5, stroke=1, fill=1)
        c.setFillColor(HexColor("#222222"))
        c.drawString(legend_x + 15, legend_y - 2.2, LABELS[m_key])
        legend_x += item_w

    plot_y = 26.0
    plot_w = 205.0
    plot_h = 108.0

    # Panel (a): Prefill Latency
    draw_panel(c, 34.0, plot_y, plot_w, plot_h, PREFILL_LATENCY_S, 1.8,
               "Prefill Time (seconds)", "(a) Prefill Latency Scaling")

    # Panel (b): Decode Latency
    draw_panel(c, 280.0, plot_y, plot_w, plot_h, DECODE_LATENCY_MS, 18.0,
               "Decode Latency (ms/token)", "(b) Decode Latency Scaling")

    # Highlight Raven flat decode on Panel (b)
    c.setFont("Helvetica-Bold", 6.2)
    c.setFillColor(HexColor("#009E73"))
    c.drawString(288.0, plot_y + 55.0, "Raven: Flat 2.27ms/tok (7.2x faster @128K)")

    c.save()
    print(f"wrote {out_path.resolve()}")


if __name__ == "__main__":
    generate_pdf(OUT)
