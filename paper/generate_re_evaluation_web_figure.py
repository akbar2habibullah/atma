"""Generate the article/dashboard SVG for the full retention-cap re-evaluation."""

from __future__ import annotations

import html
import math
from pathlib import Path

from re_evaluation_data import (
    ALL_MODELS, BABI_LENGTHS, LENGTHS, MODELS, REFERENCE_MODELS,
    baseline_babilong, baseline_longdoc, baseline_retrieval,
    capped_babilong, capped_longdoc, capped_retrieval, mean_longdoc,
)


OUT = Path(__file__).resolve().parents[1] / "pages" / "assets" / "figures" / "retention-recovery.svg"
LABELS = {
    "nope": "NoPE", "polar": "Polar", "rope": "RoPE",
    "raven_native": "Raven Native", "atma_raven_titans": "Atma-Raven-Titans",
}
COLORS = {
    "nope": "#D55E00", "polar": "#5275ff", "rope": "#C2536A",
    "raven_native": "#009E73", "atma_raven_titans": "#E69F00",
}


def polyline(points, model, capped=False, reference=False):
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash = "" if capped else (' stroke-dasharray="3 5"' if reference else ' stroke-dasharray="8 6"')
    opacity = "1" if capped else (".9" if reference else ".65")
    filled = capped or reference
    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.1" fill="{COLORS[model] if filled else "#f7f7f5"}" stroke="{COLORS[model]}" stroke-width="1.8"/>'
        for x, y in points
    )
    return f'<polyline points="{coords}" fill="none" stroke="{COLORS[model]}" stroke-width="{3 if capped else 2}" opacity="{opacity}"{dash}/>{circles}'


def panel(x0, title, ylabel, lengths, base, cap, y_ticks, project):
    y0, width, height = 505, 265, 300
    xs = [x0 + width * i / (len(lengths) - 1) for i in range(len(lengths))]
    parts = [f'<text x="{x0 + width/2}" y="166" text-anchor="middle" class="sans panel-title">{html.escape(title)}</text>']
    for value, label in y_ticks:
        y = project(value)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+width}" y2="{y:.1f}" class="grid"/><text x="{x0-10}" y="{y+4:.1f}" text-anchor="end" class="mono tick">{label}</text>')
    parts.append(f'<line x1="{x0}" y1="205" x2="{x0}" y2="{y0}" class="axis"/><line x1="{x0}" y1="{y0}" x2="{x0+width}" y2="{y0}" class="axis"/>')
    shown = {0, 2, 4, 6, len(lengths)-1}
    for i, (x, length) in enumerate(zip(xs, lengths)):
        if i in shown:
            parts.append(f'<text x="{x:.1f}" y="527" text-anchor="middle" class="mono tick">{length.upper()}</text>')
    parts.append(f'<text x="{x0-43}" y="355" text-anchor="middle" transform="rotate(-90 {x0-43} 355)" class="sans axis-label">{html.escape(ylabel)}</text>')
    for model in MODELS:
        parts.append(polyline([(x, project(base[model][length])) for x, length in zip(xs, lengths)], model, False))
        parts.append(polyline([(x, project(cap[model][length])) for x, length in zip(xs, lengths)], model, True))
    for model in REFERENCE_MODELS:
        parts.append(polyline([(x, project(base[model][length])) for x, length in zip(xs, lengths)], model, reference=True))
    return "".join(parts)


def main():
    retrieval = (baseline_retrieval(models=ALL_MODELS), capped_retrieval())
    babilong = (baseline_babilong(models=ALL_MODELS), capped_babilong())
    longdoc = (mean_longdoc(baseline_longdoc(models=ALL_MODELS)), mean_longdoc(capped_longdoc()))
    linear100 = lambda value: 505 - 300 * value / 100
    linear70 = lambda value: 505 - 300 * value / 70
    lo, hi = math.log2(.75), math.log2(10)
    log_bpb = lambda value: 505 - 300 * (math.log2(value) - lo) / (hi - lo)
    body = [panel(85, "Retrieval", "token accuracy (%)", LENGTHS, *retrieval,
                  [(0, "0"), (25, "25"), (50, "50"), (75, "75"), (100, "100")], linear100),
            panel(400, "BABILong", "macro exact match (%)", BABI_LENGTHS, *babilong,
                  [(0, "0"), (20, "20"), (40, "40"), (60, "60")], linear70),
            panel(715, "Long-document likelihood", "mean BPB (log; lower is better)", LENGTHS, *longdoc,
                  [(1, "1"), (2, "2"), (4, "4"), (8, "8")], log_bpb)]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1050 650" role="img" aria-labelledby="title desc">
<title id="title">Full length-wise retention-cap re-evaluation</title>
<desc id="desc">Untouched and one-head capped NoPE, Polar, and RoPE checkpoints across retrieval, BABILong, and long-document likelihood from 2K through 256K.</desc>
<defs><style>.sans{{font-family:Inter,Arial,sans-serif}}.serif{{font-family:Georgia,'Times New Roman',serif}}.mono{{font-family:'SFMono-Regular',Consolas,monospace}}.grid{{stroke:#dde1e7;stroke-width:1}}.axis{{stroke:#111318;stroke-width:1.5}}.tick{{font-size:11px;fill:#737985}}.axis-label{{font-size:12px;fill:#555c68}}.panel-title{{font-size:17px;font-weight:700;fill:#111318}}</style></defs>
<rect width="1050" height="650" fill="#f7f7f5"/>
<text x="52" y="42" class="sans" font-size="11" font-weight="700" letter-spacing="1.8" fill="#5275ff">FULL RE-EVALUATION · INFERENCE-ONLY DIAGNOSTIC</text>
<text x="52" y="79" class="serif" font-size="28" fill="#111318">Recovery appears across lengths, not only at 256K</text>
<text x="52" y="106" class="sans" font-size="13" fill="#6e7480">Dashed/open = matched untouched · solid/filled = one-head cap · dotted/filled = untouched Raven reference</text>
<g transform="translate(390 124)" class="sans" font-size="11">
<circle cx="0" cy="0" r="5" fill="{COLORS['nope']}"/><text x="10" y="4">NoPE</text>
<circle cx="70" cy="0" r="5" fill="{COLORS['polar']}"/><text x="80" y="4">Polar</text>
<circle cx="140" cy="0" r="5" fill="{COLORS['rope']}"/><text x="150" y="4">RoPE</text>
<circle cx="210" cy="0" r="5" fill="{COLORS['raven_native']}"/><text x="220" y="4">Raven Native</text>
<circle cx="330" cy="0" r="5" fill="{COLORS['atma_raven_titans']}"/><text x="340" y="4">Atma-Raven-Titans</text>
</g>{''.join(body)}
<text x="525" y="585" text-anchor="middle" class="sans" font-size="12" fill="#555c68">Context length</text>
<text x="52" y="625" class="sans" font-size="11" fill="#737985">Raven Native and Atma-Raven-Titans remain untouched references. The cap is a checkpoint probe applied only to the three matched attention variants.</text>
</svg>'''
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
