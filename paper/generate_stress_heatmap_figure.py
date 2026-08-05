"""Generate the 3-panel FEA secant gain stress heatmap figure for the paper.

The figure visualizes the 16 blocks x 8 context lengths local secant gain matrix
for Polar, RoPE, and NoPE, clearly illustrating the Block 7 yield locus in NoPE.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ATMA_STRESS_HEATMAP_OUT", Path(__file__).with_name("fig_stress_heatmap.pdf")))
DATA_PATH = ROOT / "scaled_ablation" / "logs_stress" / "checkpoint_stress.json"

LENGTHS = ["2048", "4096", "8192", "16384", "32768", "65536", "131072", "262144"]
LENGTH_LABELS = ["2K", "4K", "8K", "16K", "32K", "64K", "128K", "256K"]

MODELS = ["polar", "rope", "nope"]
MODEL_TITLES = {
    "polar": "Polar (ATMA): Flat Gains Across Length",
    "rope": "RoPE: Representational Attenuation",
    "nope": "NoPE (L40S mbs16): Block 7 Yield Locus",
}


def load_secant_matrices() -> dict[str, list[list[float]]]:
    raw = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    result = {}
    for repo_id, info in raw["checkpoints"].items():
        if "mbs16-polar" in repo_id:
            key = "polar"
        elif "mbs16-nope" in repo_id:
            key = "nope"
        elif "mbs16-rope" in repo_id:
            key = "rope"
        else:
            continue

        modal = info["metrics"]["stress"].get("modal", {})
        matrix = []
        for b in range(16):
            row = []
            for L in LENGTHS:
                bgain = modal.get(L, {}).get("blocks", {}).get(str(b), {}).get("random_secant_gain_max", 0.0)
                row.append(float(bgain))
            matrix.append(row)
        result[key] = matrix
    return result


def gain_color(v: float) -> HexColor:
    """Color scale for local secant gain G."""
    if v < 0.90:
        return HexColor("#E6F2FF")  # Cool light blue
    elif v < 1.00:
        return HexColor("#D0E1FD")  # Light blue
    elif v < 1.10:
        return HexColor("#FFF2CC")  # Neutral yellow
    elif v < 1.20:
        return HexColor("#FFE699")  # Soft gold
    elif v < 1.30:
        return HexColor("#FCE4D6")  # Light orange
    elif v < 1.35:
        return HexColor("#F8CBAD")  # Medium orange
    else:
        return HexColor("#F4B084")  # Deep coral / red yield alert


def text_color(v: float) -> HexColor:
    if v >= 1.30:
        return HexColor("#700000")
    elif v >= 1.10:
        return HexColor("#503000")
    else:
        return HexColor("#102040")


def generate_pdf(out_path: Path):
    matrices = load_secant_matrices()

    # Single-page landscape PDF (width = 504pt = 7in, height = 240pt = 3.33in)
    page_w = 504.0
    page_h = 245.0

    c = canvas.Canvas(str(out_path), pagesize=(page_w, page_h))

    # Title
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(HexColor("#1A2530"))
    c.drawString(12, page_h - 14, "FEA Structural Stress Probing: Local Block Secant Gains G Across Length (2K -> 256K)")

    panel_w = 152.0
    panel_h = 210.0
    margin_x = 12.0
    gap_x = 12.0
    top_y = page_h - 24.0

    cols = len(LENGTHS)
    rows = 16

    cell_w = (panel_w - 22.0) / cols  # 22pt for block labels
    cell_h = (panel_h - 26.0) / rows  # 26pt for title & col labels

    for p_idx, m_key in enumerate(MODELS):
        px = margin_x + p_idx * (panel_w + gap_x)
        py = top_y - panel_h

        mat = matrices[m_key]

        # Panel Header
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(HexColor("#0F2942") if m_key == "polar" else HexColor("#C0392B" if m_key == "nope" else "#7F8C8D"))
        c.drawString(px, top_y - 10, MODEL_TITLES[m_key])

        # Column labels (lengths)
        c.setFont("Helvetica", 5.5)
        c.setFillColor(HexColor("#4A5568"))
        for c_idx, l_lbl in enumerate(LENGTH_LABELS):
            cx = px + 20.0 + c_idx * cell_w + cell_w / 2.0
            cy = top_y - 20.0
            c.drawCentredString(cx, cy, l_lbl)

        # Draw cells
        grid_top = top_y - 23.0
        for r_idx in range(rows):
            # Row 0 at top, Row 15 at bottom
            b_num = r_idx
            cy = grid_top - (r_idx + 1) * cell_h

            # Row label (Block number)
            c.setFont("Helvetica-Bold", 5.5)
            c.setFillColor(HexColor("#2D3748"))
            c.drawString(px, cy + (cell_h - 5.0) / 2.0, f"B{b_num:<2d}")

            for c_idx in range(cols):
                val = mat[b_num][c_idx]
                cx = px + 20.0 + c_idx * cell_w

                # Fill box
                c.setFillColor(gain_color(val))
                
                # Highlight NoPE Block 7 yield explosion
                if m_key == "nope" and b_num == 7 and c_idx >= 5:
                    c.setStrokeColor(HexColor("#C0392B"))
                    c.setLineWidth(0.8)
                else:
                    c.setStrokeColor(HexColor("#FFFFFF"))
                    c.setLineWidth(0.3)

                c.rect(cx, cy, cell_w, cell_h, stroke=1, fill=1)

                # Cell text value
                c.setFont("Helvetica", 4.8)
                c.setFillColor(text_color(val))
                c.drawCentredString(cx + cell_w / 2.0, cy + (cell_h - 4.5) / 2.0, f"{val:.2f}")

    c.save()
    print(f"wrote {out_path.resolve()}")


if __name__ == "__main__":
    generate_pdf(OUT)
