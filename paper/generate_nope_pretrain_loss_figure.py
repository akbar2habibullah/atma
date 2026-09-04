"""Generate the NoPE pretraining loss trajectory comparison figure (fig_nope_pretrain_loss.pdf).

Panel (a): Full 10B-token pretraining validation loss curves (Steps 0 -> 18,722) for
           NoPE L40S mbs16, NoPE L40S mbs4, and NoPE L4 mbs4.
Panel (b): Zoomed-in resolution view (Steps 15,000 -> 18,722), showing that short-context losses remain close (2.8126 vs 2.8128 vs 2.8160 nats).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ATMA_NOPE_LOSS_OUT", Path(__file__).with_name("fig_nope_pretrain_loss.pdf")))

PATHS = {
    "mbs16_L40S": ROOT / "scaled_ablation" / "logs" / "nope__reg-baseline__distr-0__mem-1__win-0.log",
    "mbs4_L40S": ROOT / "scaled_ablation" / "logs_mbs4" / "nope__reg-baseline__distr-0__mem-1__win-0.log",
    "mbs4_L4": ROOT / "archive" / "logs" / "nope__reg-baseline__distr-0__mem-1__win-0.log",
}

LABELS = {
    "mbs16_L40S": "NoPE (L40S mbs16 Promoted)",
    "mbs4_L40S": "NoPE (L40S mbs4 Control)",
    "mbs4_L4": "NoPE (L4 mbs4 Control)",
}

COLORS = {
    "mbs16_L40S": "#D55E00",  # Vermillion red
    "mbs4_L40S": "#E69F00",   # Amber / Orange
    "mbs4_L4": "#0072B2",     # Deep blue
}


def load_curves() -> dict[str, list[tuple[int, float]]]:
    curves = {}
    for key, p in PATHS.items():
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"===ABLATION_CURVE_JSON===\s*\n({.*?}|\[.*?\])\n===END===", text, re.DOTALL)
        if not m:
            continue
        raw_list = json.loads(m.group(1))
        pts = [(int(item["step"]), float(item["val_loss"])) for item in raw_list]
        curves[key] = pts
    return curves


def draw_panel(c: canvas.Canvas, x0: float, y0: float, width: float, height: float,
               curves: dict[str, list[tuple[int, float]]], min_step: int, max_step: int,
               min_loss: float, max_loss: float, title: str, ylabel: str):
    # Axes
    c.setStrokeColor(HexColor("#333333"))
    c.setLineWidth(0.6)
    c.line(x0, y0, x0, y0 + height)
    c.line(x0, y0, x0 + width, y0)

    def proj_x(step: int) -> float:
        return x0 + width * (step - min_step) / (max_step - min_step)

    def proj_y(loss: float) -> float:
        return y0 + height * (loss - min_loss) / (max_loss - min_loss)

    # Y-axis Grid
    c.setFont("Helvetica", 6.0)
    num_ticks = 5
    step_y = (max_loss - min_loss) / (num_ticks - 1)
    for i in range(num_ticks):
        yval = min_loss + i * step_y
        yp = proj_y(yval)
        c.setStrokeColor(HexColor("#E2E8F0"))
        c.setLineWidth(0.35)
        c.line(x0, yp, x0 + width, yp)
        c.setFillColor(HexColor("#444444"))
        c.drawRightString(x0 - 3, yp - 2, f"{yval:.3f}" if max_loss < 4.0 else f"{yval:.1f}")

    # X-axis Ticks
    num_xticks = 5
    step_x = (max_step - min_step) / (num_xticks - 1)
    for i in range(num_xticks):
        xstep = int(min_step + i * step_x)
        xp = proj_x(xstep)
        c.setFillColor(HexColor("#444444"))
        c.drawCentredString(xp, y0 - 10, f"{xstep:,}")

    # Title & Axis Labels
    c.setFillColor(HexColor("#1A2530"))
    c.setFont("Helvetica-Bold", 8.0)
    c.drawCentredString(x0 + width / 2.0, y0 + height + 8, title)

    c.saveState()
    c.setFont("Helvetica-Bold", 6.5)
    c.setFillColor(HexColor("#2D3748"))
    c.translate(x0 - 24, y0 + height / 2.0)
    c.rotate(90)
    c.drawCentredString(0, 0, ylabel)
    c.restoreState()

    c.setFont("Helvetica", 6.0)
    c.setFillColor(HexColor("#4A5568"))
    c.drawCentredString(x0 + width / 2.0, y0 - 18, "Training Steps (10B Tokens)")

    # Series lines (with canvas clipping to prevent overflow past max_loss)
    for key in ["mbs4_L4", "mbs4_L40S", "mbs16_L40S"]:
        if key not in curves:
            continue
        pts = [(s, v) for s, v in curves[key] if min_step <= s <= max_step]
        if not pts:
            continue

        color = HexColor(COLORS[key])
        c.saveState()
        clip_p = c.beginPath()
        clip_p.rect(x0, y0, width, height)
        c.clipPath(clip_p, stroke=0, fill=0)

        c.setStrokeColor(color)
        c.setLineWidth(1.2 if key == "mbs16_L40S" else 0.9)
        if key == "mbs4_L40S":
            c.setDash(3, 1.5)
        elif key == "mbs4_L4":
            c.setDash(1.5, 1.5)
        else:
            c.setDash()

        p = c.beginPath()
        first_v = min(pts[0][1], max_loss)
        p.moveTo(proj_x(pts[0][0]), proj_y(first_v))
        for s, v in pts[1:]:
            p.lineTo(proj_x(s), proj_y(min(v, max_loss)))
        c.drawPath(p, stroke=1, fill=0)
        c.restoreState()


def generate_pdf(out_path: Path):
    curves = load_curves()

    page_w = 504.0
    page_h = 172.0

    c = canvas.Canvas(str(out_path), pagesize=(page_w, page_h))

    # Legend at Top
    legend_widths = [118, 102, 98]
    legend_x = (page_w - sum(legend_widths)) / 2.0
    legend_y = page_h - 10.0
    c.setFont("Helvetica-Bold", 6.5)

    for key, item_w in zip(["mbs16_L40S", "mbs4_L40S", "mbs4_L4"], legend_widths):
        color = HexColor(COLORS[key])
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(1.2 if key == "mbs16_L40S" else 0.9)
        if key == "mbs4_L40S":
            c.setDash(3, 1.5)
        elif key == "mbs4_L4":
            c.setDash(1.5, 1.5)
        else:
            c.setDash()

        c.line(legend_x, legend_y, legend_x + 12, legend_y)
        c.setDash()
        c.setFillColor(HexColor("#222222"))
        c.drawString(legend_x + 15, legend_y - 2.2, LABELS[key])
        legend_x += item_w

    plot_y = 26.0
    plot_w = 202.0
    plot_h = 108.0

    # Panel (a): Full Pretraining Trajectory (Steps 0 -> 18,722)
    draw_panel(c, 36.0, plot_y, plot_w, plot_h, curves, 0, 18722, 2.5, 11.0,
               "(a) Full 10B Pretraining Validation Loss", "Validation Loss (nats)")

    # Panel (b): Zoomed-In View (Steps 15,000 -> 18,722, Loss 2.80 -> 2.86)
    draw_panel(c, 282.0, plot_y, plot_w, plot_h, curves, 15000, 18722, 2.80, 2.86,
               "(b) Zoomed-In Resolution (Steps 15K-18.7K)", "Validation Loss (nats)")

    # Annotation on Panel (b) - Bottom left empty region
    c.setFont("Helvetica-Bold", 5.8)
    c.setFillColor(HexColor("#D55E00"))
    c.drawString(288.0, plot_y + 18.0, "Final validation: 2.8126 / 2.8128 / 2.8160 nats")
    c.setFillColor(HexColor("#C0392B"))
    c.drawString(288.0, plot_y + 10.0, "Diverges to 10.77 vs 4.07 NLL at 256K extrapolation")

    c.save()
    print(f"wrote {out_path.resolve()}")


if __name__ == "__main__":
    generate_pdf(OUT)
