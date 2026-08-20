"""Generate paired retrieval depth heatmaps before and after retention clamping."""

from __future__ import annotations

import os
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

from re_evaluation_data import ALL_MODELS, DEPTHS, LENGTHS, MODELS, REFERENCE_MODELS, baseline_retrieval, capped_retrieval


OUT = Path(os.environ.get("ATMA_NIAH_HEATMAP_OUT", Path(__file__).with_name("fig_niah_heatmap.pdf")))
LABELS = {
    "nope": "NoPE", "polar": "Polar", "rope": "RoPE",
    "raven_native": "Raven Native", "atma_raven_titans": "Atma-Raven-Titans",
}
MODEL_COLORS = {
    "nope": "#D55E00", "polar": "#0072B2", "rope": "#CC79A7",
    "raven_native": "#009E73", "atma_raven_titans": "#E69F00",
}


def accuracy_color(value):
    if value >= 90:
        return HexColor("#2ECC71")
    if value >= 70:
        return HexColor("#A3E4D7")
    if value >= 50:
        return HexColor("#F9E79F")
    if value >= 25:
        return HexColor("#FAD7A0")
    if value >= 10:
        return HexColor("#EDBB99")
    if value > 1:
        return HexColor("#E5E7E9")
    return HexColor("#F2F4F4")


def text_color(value):
    if value >= 80:
        return HexColor("#004D20")
    if value >= 50:
        return HexColor("#4A3B00")
    if value >= 10:
        return HexColor("#5C2C00")
    return HexColor("#7F8C8D")


def generate_pdf(out_path):
    baseline = baseline_retrieval(by_depth=True, models=ALL_MODELS)
    capped = capped_retrieval(by_depth=True)
    page_w, page_h = 504.0, 330.0
    c = canvas.Canvas(str(out_path), pagesize=(page_w, page_h))
    c.setFont("Helvetica-Bold", 9.5)
    c.setFillColor(HexColor("#1A2530"))
    c.drawString(12, page_h - 14, "Retrieval token accuracy by context length and needle depth")

    panel_w, panel_h = 154.0, 88.0
    margin_x, gap_x, gap_y = 12.0, 12.0, 8.0
    cell_w = (panel_w - 22.0) / len(LENGTHS)
    cell_h = (panel_h - 22.0) / len(DEPTHS)

    panels = []
    for row_i, (condition, data) in enumerate((("Untouched", baseline), ("One-head cap", capped))):
        for col_i, model in enumerate(MODELS):
            panels.append((condition, data, model, margin_x + col_i * (panel_w + gap_x),
                           page_h - 26.0 - panel_h - row_i * (panel_h + gap_y)))
    ref_y = page_h - 26.0 - panel_h - 2 * (panel_h + gap_y)
    ref_xs = (58.0, 292.0)
    for model, px in zip(REFERENCE_MODELS, ref_xs):
        panels.append(("Untouched reference", baseline, model, px, ref_y))

    for condition, data, model, px, py in panels:
        top_y = py + panel_h
        endpoint = sum(data[model]["256k"].values()) / len(DEPTHS)
        is_reference = condition == "Untouched reference"
        c.setFont("Helvetica-Bold", 6.6 if is_reference else 7.4)
        c.setFillColor(HexColor(MODEL_COLORS[model]))
        condition_label = "Reference" if is_reference else condition
        c.drawString(px, top_y - 8, f"{LABELS[model]} - {condition_label} ({endpoint:.1f}% @256K)")
        c.setFont("Helvetica", 5.4)
        c.setFillColor(HexColor("#4A5568"))
        for i, label in enumerate(LENGTHS):
            c.drawCentredString(px + 20 + i * cell_w + cell_w / 2, top_y - 18, label.upper())
        grid_top = top_y - 21
        for depth_i, depth in enumerate(DEPTHS):
            cy = grid_top - (depth_i + 1) * cell_h
            c.setFont("Helvetica-Bold", 5.4)
            c.setFillColor(HexColor("#2D3748"))
            c.drawString(px, cy + (cell_h - 5) / 2, f"{int(float(depth) * 100)}%")
            for length_i, length in enumerate(LENGTHS):
                value = data[model][length][depth]
                cx = px + 20 + length_i * cell_w
                c.setFillColor(accuracy_color(value))
                c.setStrokeColor(HexColor("#FFFFFF"))
                c.setLineWidth(0.3)
                c.rect(cx, cy, cell_w, cell_h, stroke=1, fill=1)
                c.setFont("Helvetica", 4.8)
                c.setFillColor(text_color(value))
                c.drawCentredString(cx + cell_w / 2, cy + (cell_h - 4.5) / 2, f"{value:.0f}")
    c.save()
    print(f"wrote {out_path.resolve()}")


if __name__ == "__main__":
    generate_pdf(OUT)
