"""Generate the multi-panel NIAH depth-sweep retrieval heatmap figure (fig_niah_heatmap.pdf).

Displays the exact 8 lengths x 3 depths retrieval accuracy matrix for all 5 models:
Polar, NoPE, RoPE, Raven Native, and Atma-Raven-Titans.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ATMA_NIAH_HEATMAP_OUT", Path(__file__).with_name("fig_niah_heatmap.pdf")))
DATA_PATH = ROOT / "benchmarks" / "logs" / "atma_10b" / "benchmark_matrix.json"

LENGTHS = ["2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k"]
LENGTH_LABELS = ["2K", "4K", "8K", "16K", "32K", "64K", "128K", "256K"]
DEPTHS = ["0.1", "0.5", "0.9"]
DEPTH_LABELS = ["10%", "50%", "90%"]

MODELS = ["polar", "nope", "rope", "raven_native", "atma_raven_titans"]
MODEL_TITLES = {
    "polar": "Polar: Bounded (34.4% @256K)",
    "nope": "NoPE: Dilution (0.6% @256K)",
    "rope": "RoPE: Collapse (0.0% @256K)",
    "raven_native": "Raven Native: (20.3% @256K)",
    "atma_raven_titans": "Atma-Raven-Titans: (10.0% @256K)",
}


def load_retrieval_matrices() -> dict[str, dict[str, list[float]]]:
    rows = json.loads(DATA_PATH.read_text(encoding="utf-8"))["rows"]
    grouped = defaultdict(list)
    for row in rows:
        if row.get("benchmark") == "retrieval" and row.get("metric") == "token_accuracy":
            if "/smoke_" not in row.get("source_log", ""):
                grouped[(row["model"], row["length"], str(row["depth"]))].append(float(row["value"]))

    res = {}
    for m in MODELS:
        m_dict = {}
        for d in DEPTHS:
            row = []
            for L in LENGTHS:
                vals = grouped[(m, L, d)]
                avg = sum(vals) / len(vals) if vals else 0.0
                row.append(float(avg))
            m_dict[d] = row
        res[m] = m_dict
    return res


def accuracy_color(v: float) -> HexColor:
    """Color scale for retrieval accuracy percentage (0 to 100%)."""
    if v >= 90.0:
        return HexColor("#2ECC71")  # Bright vivid green
    elif v >= 70.0:
        return HexColor("#A3E4D7")  # Soft green / mint
    elif v >= 50.0:
        return HexColor("#F9E79F")  # Warm yellow
    elif v >= 25.0:
        return HexColor("#FAD7A0")  # Soft orange
    elif v >= 10.0:
        return HexColor("#EDBB99")  # Muted coral
    elif v > 1.0:
        return HexColor("#E5E7E9")  # Light neutral grey
    else:
        return HexColor("#F2F4F4")  # Very pale grey / zero collapse


def text_color(v: float) -> HexColor:
    if v >= 80.0:
        return HexColor("#004D20")
    elif v >= 50.0:
        return HexColor("#4A3B00")
    elif v >= 10.0:
        return HexColor("#5C2C00")
    else:
        return HexColor("#7F8C8D")


def generate_pdf(out_path: Path):
    matrices = load_retrieval_matrices()

    # Landscape PDF (width = 504pt = 7in, height = 240pt = 3.33in)
    page_w = 504.0
    page_h = 240.0

    c = canvas.Canvas(str(out_path), pagesize=(page_w, page_h))

    # Title
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(HexColor("#1A2530"))
    c.drawString(12, page_h - 14, "NIAH Retrieval Depth-Sweep Heatmaps Across Lengths (2K -> 256K) and Depths (10%, 50%, 90%)")

    # Layout 5 panels (3 in top row, 2 in bottom row)
    panel_w = 154.0
    panel_h = 95.0
    margin_x = 12.0
    gap_x = 12.0
    gap_y = 12.0

    cols = len(LENGTHS)
    rows = len(DEPTHS)

    cell_w = (panel_w - 22.0) / cols  # 22pt for depth labels
    cell_h = (panel_h - 22.0) / rows  # 22pt for title & col labels

    positions = [
        (margin_x, page_h - 26.0 - panel_h),                                # Top left: Polar
        (margin_x + panel_w + gap_x, page_h - 26.0 - panel_h),             # Top mid: NoPE
        (margin_x + 2 * (panel_w + gap_x), page_h - 26.0 - panel_h),        # Top right: RoPE
        (margin_x, page_h - 26.0 - 2 * panel_h - gap_y),                   # Bottom left: Raven Native
        (margin_x + panel_w + gap_x, page_h - 26.0 - 2 * panel_h - gap_y), # Bottom mid: Atma-Raven-Titans
    ]

    for p_idx, m_key in enumerate(MODELS):
        px, py = positions[p_idx]
        top_y = py + panel_h

        m_data = matrices[m_key]

        # Panel Header
        c.setFont("Helvetica-Bold", 7.5)
        c.setFillColor(HexColor("#0F2942") if m_key == "polar" else HexColor("#C0392B" if m_key in ("nope", "rope") else "#16A085"))
        c.drawString(px, top_y - 8, MODEL_TITLES[m_key])

        # Column labels (lengths)
        c.setFont("Helvetica", 5.5)
        c.setFillColor(HexColor("#4A5568"))
        for c_idx, l_lbl in enumerate(LENGTH_LABELS):
            cx = px + 20.0 + c_idx * cell_w + cell_w / 2.0
            cy = top_y - 18.0
            c.drawCentredString(cx, cy, l_lbl)

        # Draw cells
        grid_top = top_y - 21.0
        for r_idx, d_key in enumerate(DEPTHS):
            d_lbl = DEPTH_LABELS[r_idx]
            cy = grid_top - (r_idx + 1) * cell_h

            # Row label (Depth percentage)
            c.setFont("Helvetica-Bold", 5.5)
            c.setFillColor(HexColor("#2D3748"))
            c.drawString(px, cy + (cell_h - 5.0) / 2.0, d_lbl)

            row_vals = m_data[d_key]
            for c_idx in range(cols):
                val = row_vals[c_idx]
                cx = px + 20.0 + c_idx * cell_w

                # Fill box
                c.setFillColor(accuracy_color(val))
                c.setStrokeColor(HexColor("#FFFFFF"))
                c.setLineWidth(0.3)

                c.rect(cx, cy, cell_w, cell_h, stroke=1, fill=1)

                # Cell text value
                c.setFont("Helvetica", 4.8)
                c.setFillColor(text_color(val))
                c.drawCentredString(cx + cell_w / 2.0, cy + (cell_h - 4.5) / 2.0, f"{int(round(val))}")

    c.save()
    print(f"wrote {out_path.resolve()}")


if __name__ == "__main__":
    generate_pdf(OUT)
