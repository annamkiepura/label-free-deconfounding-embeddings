"""Matplotlib style for figures drawn at the manuscript's printed size.

Figures are created at the exact TACL \\columnwidth (3.03 in) or \\textwidth
(6.30 in) and included at that width (no scaling), so the font sizes below are
the printed sizes: 8 pt labels and 7 pt ticks against the 10-11 pt body text.
STIX is a Times-compatible serif bundled with Matplotlib, matching `times.sty`.
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", str(__import__("pathlib").Path(
    os.environ.get("TMPDIR", "/tmp")) / "paper_assets_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

COLUMN = config.COLUMN_WIDTH_IN
TEXT = config.TEXT_WIDTH_IN
MIN_FONT_PT = 6.5           # nothing printed smaller than this

RC = {
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.title_fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "grid.color": "#E2E2E2",
    "grid.linewidth": 0.5, "lines.linewidth": 1.1, "lines.markersize": 3.5,
    "errorbar.capsize": 1.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "figure.dpi": 150, "axes.titlepad": 4,
}


def apply() -> None:
    plt.rcParams.update(RC)


def figure(width: str = "text", height: float = 2.4, **kw):
    """A figure exactly one column or the full text width wide."""
    apply()
    w = {"column": COLUMN, "text": TEXT}[width]
    return plt.subplots(figsize=(w, height), **kw)


def smallest_font(fig) -> float:
    """Smallest font size of any visible text artist (printed points)."""
    sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip()]
    return min(sizes) if sizes else float("inf")


def plain(label: str) -> str:
    """A LaTeX-source label for Matplotlib text: ``--`` becomes an en dash."""
    return label.replace("---", "\u2014").replace("--", "\u2013")


def _undrawn_ticks(fig) -> set:
    """Tick labels Matplotlib keeps but does not draw (location outside the view)."""
    skip = set()
    for ax in fig.axes:
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())), (ax.yaxis, sorted(ax.get_ylim()))):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo - 1e-9 <= tick.get_loc() <= hi + 1e-9):
                    skip.update({id(tick.label1), id(tick.label2)})
    return skip


def clipped_texts(fig) -> list[str]:
    """Drawn text artists that extend beyond the figure canvas."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    skip = _undrawn_ticks(fig)
    out = []
    for t in fig.findobj(matplotlib.text.Text):
        if not (t.get_visible() and t.get_text().strip()) or id(t) in skip:
            continue
        bb = t.get_window_extent(r)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
            out.append(t.get_text()[:40])
    return out


def save(fig, name: str, ex) -> list:
    """Vector PDF + PNG preview at the declared size (no bbox cropping, so the
    printed width is exactly the manuscript width). Records legibility checks."""
    from .provenance import check
    check(smallest_font(fig) >= MIN_FONT_PT, "figure_font_size",
          f"{name}: smallest font {smallest_font(fig):.1f} pt >= {MIN_FONT_PT} pt")
    clipped = clipped_texts(fig)
    check(not clipped, "figure_no_clipping", f"{name}: no text outside the canvas", clipped=clipped)
    paths = []
    for ext in ("pdf", "png"):
        p = config.OUT_DIRS["figures"] / f"{name}.{ext}"
        fig.savefig(p, metadata={"CreationDate": None} if ext == "pdf" else None)
        ex.output(p)
        paths.append(p)
    w, h = fig.get_size_inches()
    ex.note(f"figure size {w:.3f} x {h:.3f} in (declared width = manuscript width)")
    plt.close(fig)
    return paths
