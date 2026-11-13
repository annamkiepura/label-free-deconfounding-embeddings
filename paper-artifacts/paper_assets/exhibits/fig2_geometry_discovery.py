"""Main Figure 2: three independent panels, then a composite of the finished panels.

  fig2a_pca_accessibility   confounder subspace captured by the leading C-1 PCs (capture[C-1])
  fig2b_cluster_alignment   ARI of the silhouette-K k-means groups with the confounder classes
  fig2c_residual_leakage    AUC(method) - AUC(Metadata-LEACE) for PC-1, k-means LEACE, SAE-LEACE

Each panel is drawn by one function at fixed physical sizes (inches), so the six setting
rows fall at identical heights in every panel. Standalone versions carry the setting
labels; for the composite, B and C are re-rendered by the same function without the labels
and the three finished vector PDFs are placed side by side with LaTeX (no redrawing in a
shared axes system). Values, definitions, cohorts and checks come from the shared
collector (`fig2_geometry_discovery_scatter._collect`); nothing is fitted.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from ..lib import config, latex, style
from ..lib.io import run_id, sha256
from ..lib.loaders import Cohort
from ..lib.methods import METHODS
from ..lib.provenance import check, issue
from .fig2_geometry_discovery_scatter import GAPS, _collect

NAME, KIND, TITLE = "fig2_geometry_discovery", "figure", "Geometry and label-free erasure performance"
LABEL = "fig:geometry-discovery"
FONT = {"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"}
LABELS = ["E5 · SwiLTra DE–FR", "E5 · SwiLTra DE–FR–IT", "E5 · SCOTUS Oyez–Lexis",
          "MiniLM · SCOTUS Oyez–Lexis", "E5 · SCOTUS 3 sources", "MiniLM · SCOTUS 3 sources"]
PANELS = {"a": "fig2a_pca_accessibility", "b": "fig2b_cluster_alignment", "c": "fig2c_residual_leakage"}
TITLES = {"a": "A  PCA\naccessibility", "b": "B  Unsupervised\ndiscovery", "c": "C  Residual leakage"}
#: What each x axis measures, set as a grey subtitle under the title (so the bottom band holds only
#: the tick labels). Every header starts at its plot's left edge; A's and B's titles need two lines
#: in their 1.1-1.2 in, and A's subtitle two more (the header band is sized for A's four lines).
SUBTITLES = {"a": "Confounder subspace\nin leading $C-1$ PCs", "b": "Cluster–confounder ARI",
             "c": "Linear AUC above\nMetadata-LEACE (lower is better)"}
C_LEGEND = {"pca_top1": "PCA-1", "kmeans_auto": "k-means LEACE", "sae_leace": "SAE-LEACE (N = 32)"}
C_OFFSET = {"pca_top1": -0.17, "kmeans_auto": 0.0, "sae_leace": 0.17}
#: Rows shaded to guide the eye across panels (second, fourth, sixth setting), and the shade.
BANDED_ROWS = (1, 3, 5)
BAND = "#EEF0F3"
# --- shared physical geometry (inches): identical rows in every panel ------------------ #
H = 2.74                  # figure height (rows unchanged at 0.30 in; header and bottom bands compacted)
BOTTOM = 0.25             # x tick labels only (the axis description is the subtitle)
TOP = 0.67                # A's four header lines (B and C: three) | the method key over the label column
LABEL_W = 1.78            # y-label column (standalone panels and composite A)
PLOT_W = {"a": 1.05, "b": 1.13, "c": 1.72}
PAD_L, PAD_R = 0.16, 0.1  # left pad of label-less panels (holds the separator); right pad of every panel
#: Header text start relative to the plot's left edge (in), the same in every panel (just right of
#: B's and C's separator rule), so each title lines up with its plot.
HEAD_DX = {"a": -0.04, "b": -0.04, "c": -0.04}
LABEL_PAD = 6             # (pt) gap between the setting labels and panel A (matplotlib default 3.5)
HEAD_TOP, TITLE_LINE, SUB_LINE = 0.05, 0.155, 0.125   # (in) top margin, title and subtitle line pitch
FS = {"label": 8, "tick": 8, "annot": 8, "title": 9, "subtitle": 7.5, "legend": 8}
MIN_PT = 7.5
_STATE: dict = {}


def _range_text(vals: list) -> str:
    """'2' if every seed agrees, else 'min–max' (never a single collapsed value)."""
    lo, hi = min(vals), max(vals)
    return str(lo) if lo == hi else f"{lo}–{hi}"


def _key(fig, W: float) -> None:
    """The method key (PC-1 square, k-means cross, SAE-LEACE circle), top left over the label column.
    It also explains A's and B's markers, which are PC-1's and k-means LEACE's."""
    handles = [Line2D([], [], marker=METHODS[m].marker, color=METHODS[m].color, linestyle="none",
                      markersize=7.5 if m != "kmeans_auto" else 8.5) for m in GAPS]
    fig.legend(handles, [C_LEGEND[m] for m in GAPS], loc="upper left",
               bbox_to_anchor=(0.02 / W, 1 - HEAD_TOP / H), ncol=1, frameon=False, fontsize=FS["legend"],
               handletextpad=0.4, handlelength=1.2, labelspacing=0.25, borderaxespad=0, borderpad=0)


def _panel(which: str, cells: list, *, labels: bool, key: bool = False) -> plt.Figure:
    left = LABEL_W if labels else PAD_L
    W = left + PLOT_W[which] + PAD_R
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, BOTTOM / H, PLOT_W[which] / W, (H - BOTTOM - TOP) / H])
    n = len(cells)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n), LABELS if labels else [""] * n, fontsize=FS["label"])
    ax.tick_params(axis="y", length=0, pad=LABEL_PAD)
    ax.tick_params(axis="x", labelsize=FS["tick"], length=3)
    ax.minorticks_off()
    ax.spines["left"].set_visible(False)
    # SwiLTra | SCOTUS separator: across the label column too when labels are shown.
    y_sep = fig.transFigure.inverted().transform(ax.transData.transform((0, 1.5)))[1]
    fig.add_artist(Line2D([0.02 if labels else ax.get_position().x0, ax.get_position().x1],
                          [y_sep, y_sep], transform=fig.transFigure, color="0.7", linewidth=0.7,
                          zorder=0))
    ax.patch.set_visible(False)
    # Alternating row bands, edge to edge in figure coordinates. Every panel has the same
    # row geometry, so in the composite the bands join into continuous stripes that lead the
    # eye from a setting label through A, B and C.
    for yi in BANDED_ROWS:
        y_lo, y_hi = (fig.transFigure.inverted().transform(ax.transData.transform((0, yi + d)))[1]
                      for d in (0.5, -0.5))
        fig.add_artist(Rectangle((0, y_lo), 1, y_hi - y_lo, transform=fig.transFigure,
                                 facecolor=BAND, edgecolor="none", zorder=-1))
    x_t = ax.get_position().x0
    if not labels:
        # Composite parts B and C: a vertical rule at the left edge separates them from the
        # panel to their left (A|B and B|C). It sits above the row bands, full panel height.
        x_v = 0.5 * PAD_L / W
        fig.add_artist(Line2D([x_v, x_v], [0.02, 0.98], transform=fig.transFigure, color="0.55",
                              linewidth=0.8, zorder=2))
    # Header: bold title, then the grey subtitle saying what the x axis measures (no x label below).
    x_t = x_t + HEAD_DX[which] / W
    fig.text(x_t, 1 - HEAD_TOP / H, TITLES[which], ha="left", va="top", fontsize=FS["title"],
             fontweight="bold", linespacing=1.2)
    y_sub = HEAD_TOP + TITLE_LINE * (TITLES[which].count("\n") + 1) + 0.015
    fig.text(x_t, 1 - y_sub / H, SUBTITLES[which], ha="left", va="top", fontsize=FS["subtitle"],
             color="#555555", linespacing=1.2)
    if key:
        _key(fig, W)
    if which in "ab":
        vals = [c["capc"]["mean"] if which == "a" else c["ari"]["mean"] for c in cells]
        mt = METHODS["pca_top1" if which == "a" else "kmeans_auto"]
        lo = 0.0 if min(vals) >= 0 else min(vals) - 0.05
        ax.set_xlim(lo - 0.05, 1.08)          # room right of 1.00 for markers and labels
        ax.set_xticks([lo, 0.5, 1.0] if lo < 0 else [0, 0.5, 1.0])
        ax.set_xticklabels([f"{t:.1f}" for t in ax.get_xticks()])
        for yi, (v, c) in enumerate(zip(vals, cells)):
            ax.plot(v, yi, marker=mt.marker, color=mt.color, markersize=8.5, linestyle="none", zorder=3)
            txt = f"{v:.2f}" if which == "a" else f"{v:.2f} · K={_range_text(c['k'])}"
            right = v < (0.7 if which == "a" else 0.5)
            ax.annotate(txt, (v, yi), xytext=(8 if right else -8, 0), textcoords="offset points",
                        ha="left" if right else "right", va="center", fontsize=FS["annot"],
                        color="#222222", zorder=4,
                        bbox=dict(boxstyle="square,pad=0.08", ec="none",
                                  fc=BAND if yi in BANDED_ROWS else "white"))
    else:
        allv = [c["gaps"][m].mean for c in cells for m in GAPS]
        lo = 0.0 if min(allv) >= 0 else min(allv)
        hi = max(allv)
        ax.set_xlim(lo - 0.02, hi + 0.04)
        ax.set_xticks(np.round(np.arange(0 if lo >= 0 else np.floor(lo * 10) / 10, hi + 0.04, 0.1), 1))
        ax.set_xticklabels([f"{t:.1f}" for t in ax.get_xticks()])
        ax.axvline(0, color="#000000", linewidth=1.0, zorder=1)
        for m in GAPS:
            mt = METHODS[m]
            ax.plot([c["gaps"][m].mean for c in cells], np.arange(n) + C_OFFSET[m], marker=mt.marker,
                    color=mt.color, markersize=7.5 if m != "kmeans_auto" else 8.5, linestyle="none",
                    zorder=3, label=C_LEGEND[m])
    return fig


def _inspect(fig, tag: str) -> None:
    """Readability checks on the rendered panel: sizes, clipping, overlapping annotations."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(plt.Text) if t.get_visible() and t.get_text().strip()]
    check(min(t.get_fontsize() for t in texts) >= MIN_PT, "f2p_font",
          f"{tag}: smallest text {min(t.get_fontsize() for t in texts):.1f} pt >= {MIN_PT} pt")
    clipped = style.clipped_texts(fig)
    check(not clipped, "f2p_clipping", f"{tag}: no text outside the canvas", clipped=clipped)
    ax = fig.axes[0]
    ann = [t for t in ax.texts if t.get_text().strip()]
    boxes = [t.get_window_extent(r) for t in ann]
    overlap = [(ann[i].get_text(), ann[j].get_text()) for i in range(len(boxes))
               for j in range(i + 1, len(boxes)) if boxes[i].overlaps(boxes[j])]
    check(not overlap, "f2p_annot_overlap", f"{tag}: no two value annotations overlap", overlap=overlap)
    marker_hits = []
    for t, bb in zip(ann, boxes):
        px, py = ax.transData.transform(t.xy)
        if bb.x0 - 2 <= px <= bb.x1 + 2 and bb.y0 <= py <= bb.y1:
            marker_hits.append(t.get_text())
    check(not marker_hits, "f2p_annot_marker", f"{tag}: no annotation covers its marker", hits=marker_hits)
    # The key (if any) must not touch any header text, and every header must end above the plot.
    head = [t for t in fig.texts if t.get_text().strip()]
    head_bottom = min(t.get_window_extent(r).y0 for t in head)
    for leg in fig.legends:
        lb = leg.get_window_extent(r)
        hit = [t.get_text() for t in head if t.get_window_extent(r).overlaps(lb)]
        check(not hit, "f2p_key_clear", f"{tag}: the method key overlaps no title or subtitle", hits=hit)
        head_bottom = min(head_bottom, lb.y0)
    plot_top = ax.get_window_extent(r).y1
    check(head_bottom >= plot_top, "f2p_header_above_plot",
          f"{tag}: header ends {(head_bottom - plot_top) / fig.dpi:.3f} in above the plot")
    labels = [t for t in ax.get_yticklabels() if t.get_text()]
    if labels:
        lb = [t.get_window_extent(r) for t in labels]
        check(all(b.x0 >= -0.5 for b in lb), "f2p_labels_inside", f"{tag}: setting labels fully on the canvas")


def _save(fig, path_base: Path, ex, png: bool = True) -> None:
    for ext in ("pdf", "png") if png else ("pdf",):
        p = path_base.with_suffix(f".{ext}")
        fig.savefig(p, dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
        ex.output(p)
    plt.close(fig)


def _compose(parts: list[Path], out: Path, ex) -> bool:
    """Place the finished panel PDFs side by side (bottoms aligned; equal heights => rows align)."""
    if not shutil.which("pdflatex"):
        issue("error", "no_latex", "pdflatex not available: composite not built")
        return False
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, p in enumerate(parts):
            shutil.copy(p, tmp / f"p{i}.pdf")
        body = "".join(f"\\includegraphics{{p{i}.pdf}}" for i in range(len(parts)))
        (tmp / "c.tex").write_text("\\documentclass[border=0pt]{standalone}\n\\usepackage{graphicx}\n"
                                   "\\begin{document}\\setlength{\\lineskip}{0pt}%\n" + body +
                                   "\\end{document}\n")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "c.tex"], cwd=tmp, capture_output=True,
                           text=True)
        if r.returncode != 0:
            issue("error", "latex_failed", "composite failed to compile", log=r.stdout[-1500:])
            return False
        shutil.copy(tmp / "c.pdf", out.with_suffix(".pdf"))
    subprocess.run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", "-r300",
                    f"-sOutputFile={out.with_suffix('.png')}", str(out.with_suffix(".pdf"))], capture_output=True)
    ex.output(out.with_suffix(".pdf"))
    ex.output(out.with_suffix(".png"))
    return True


def build(ex) -> None:
    style.apply()
    with plt.rc_context(FONT):
        _build(ex)


def _build(ex) -> None:
    cells, rows_csv, notes = _collect()
    figs = config.OUT_DIRS["figures"]
    parts_dir = figs / "fig2_parts"
    parts_dir.mkdir(exist_ok=True)
    # 1. Three standalone panels (each with the setting labels), inspected.
    for w in "abc":
        f = _panel(w, cells, labels=True, key=(w == "c"))     # only C needs the key on its own
        _inspect(f, f"standalone {w.upper()}")
        _save(f, figs / PANELS[w], ex)
    # 2. Composite parts: A with labels (identical to its standalone), B and C without.
    parts = []
    for w in "abc":
        f = _panel(w, cells, labels=(w == "a"), key=(w == "a"))   # composite key: top left, over the labels
        _inspect(f, f"composite part {w.upper()}")
        p = parts_dir / f"{PANELS[w]}_part"
        _save(f, p, ex, png=False)
        parts.append(p.with_suffix(".pdf"))
    widths = {w: (LABEL_W if w == "a" else PAD_L) + PLOT_W[w] + PAD_R for w in "abc"}
    total = sum(widths.values())
    check(abs(total - style.TEXT) < 0.02, "f2p_composite_width",
          f"composite width {total:.2f} in = manuscript text width {style.TEXT:.2f} in")
    plot_share = {w: PLOT_W[w] / sum(PLOT_W.values()) for w in "abc"}
    ex.note("composite plotting-area shares: " + ", ".join(f"{w.upper()} {100 * s:.0f}%" for w, s in plot_share.items())
            + f"; label column {LABEL_W:.2f} in inside A")
    ok = _compose(parts, figs / NAME, ex)
    check(ok, "f2p_composite", "composite built from the three finished panel PDFs")
    # 3. Data files.
    _write_data(cells, rows_csv, ex)
    caption = (r"Geometry and label-free erasure. Rows are six settings (encoder $\cdot$ corpus; the "
               r"line separates SwiLTra from SCOTUS). \textbf{A}~PCA accessibility: share of the "
               r"supervised confounder subspace (the rank-$(C-1)$ span of the label cross-covariance, which "
               r"Metadata-LEACE removes) captured by the leading $C-1$ principal components of the training "
               r"embeddings. \textbf{B}~Unsupervised discovery: adjusted Rand index between the "
               r"silhouette-selected $k$-means groups and the confounder classes (post hoc), with the "
               r"selected $K$ (a range where seeds differ). \textbf{C}~Residual leakage: held-out linear "
               r"AUC of each label-free eraser minus that of Metadata-LEACE, formed per seed (0 = the "
               r"supervised reference). Points are seed means (SDs in the data files). A and B use seeds "
               r"0--2 of the geometry run; C uses each setting's performance cohort (SCOTUS/E5-small: "
               r"coverage seeds 0--2; SwiLTra and MiniLM: seeds 0--4), so the two do not always share a "
               r"seed population. Leading PCs needed for 90\% capture, by row: "
               + "; ".join(f"{i + 1}: {_range_text(c['r90'])}" for i, c in enumerate(cells)) + ".")
    latex.write_figure_include(NAME, caption, LABEL, ex, wide=True)
    for w in "abc":
        inc = figs / f"{PANELS[w]}.include.tex"
        inc.write_text(f"% Standalone panel {w.upper()} (python -m paper_assets build {NAME})\n"
                       f"\\includegraphics{{paper_assets/figures/{PANELS[w]}.pdf}}\n")
        ex.output(inc)
    _STATE.update(cells=cells, notes=notes, widths=widths, plot_share=plot_share)


def _write_data(cells, rows_csv, ex) -> None:
    combined = []
    for i, c in enumerate(cells):
        coh = Cohort.assigned(c["key"], c["enc"])
        g = next(r for r in rows_csv if r["setting_number"] == c["n"])
        auc = {m: coh.values(m, "linear_auc", register=False) for m in ("metadata_leace",) + GAPS}
        rec = {"row": c["n"], "label": LABELS[i], "corpus": c["key"],
               "dataset_version": config.setting(c["key"])["long"], "encoder": c["enc"], "C": c["C"],
               "selected_K_per_geometry_seed": c["k"], "selected_K": _range_text(c["k"]),
               "capture_leading_Cminus1_PCs_mean": c["capc"]["mean"],
               "capture_leading_Cminus1_PCs_sd": c["capc"]["sd"],
               "pcs_for_90pct_per_geometry_seed": c["r90"], "pcs_for_90pct": _range_text(c["r90"]),
               "ari_mean": c["ari"]["mean"], "ari_sd": c["ari"]["sd"],
               "geometry_seeds": g["geometry_seeds"], "geometry_run_id": g["geometry_run_id"],
               "metadata_leace_auc_mean": auc["metadata_leace"].mean,
               "performance_cohort": g["performance_cohort"], "performance_seeds": g["performance_seeds"]}
        for m in GAPS:
            d = c["gaps"][m]
            rec[f"{m}_auc_mean"] = auc[m].mean
            rec[f"{m}_gap_mean"] = d.mean
            rec[f"{m}_gap_sd"] = d.sd
            rec[f"{m}_seeds"] = d.seeds
            rec[f"{m}_run_ids"] = sorted({s["run_id"] for s in d.sources})
        rec["same_seed_population"] = g["same_seed_population"]
        rec["sd_ddof"] = 1
        combined.append(rec)
    keep = {"a": ["capture_leading_Cminus1_PCs_mean", "capture_leading_Cminus1_PCs_sd", "pcs_for_90pct",
                  "pcs_for_90pct_per_geometry_seed", "geometry_seeds", "geometry_run_id"],
            "b": ["ari_mean", "ari_sd", "selected_K", "selected_K_per_geometry_seed", "geometry_seeds",
                  "geometry_run_id"],
            "c": ["metadata_leace_auc_mean"] + [f"{m}_{k}" for m in GAPS for k in
                                                  ("auc_mean", "gap_mean", "gap_sd", "seeds", "run_ids")]
                 + ["performance_cohort", "performance_seeds"]}
    base = ["row", "label", "corpus", "dataset_version", "encoder", "C"]
    for w in "abc":
        latex.write_csv(PANELS[w], [{k: r[k] for k in base + keep[w]} for r in combined], ex)
    latex.write_csv("fig2_geometry_discovery_combined", combined, ex)
    _STATE["combined"] = combined


DESIGN = """Figure 2 - design report (three independent panels + composite)
================================================================

Built by `python -m paper_assets build fig2_geometry_discovery` (no model was fitted or rerun).

How it was built
- One drawing function renders each panel at fixed physical sizes: figure height {H} in,
  bottom band {BOTTOM} in, top band {TOP} in, so the six rows sit at identical heights in
  every panel (row pitch {pitch:.2f} in). The fonts are {fs}.
- The standalone panels A, B and C each carry the six setting labels. Every rendered panel is
  checked for: smallest text >= {min_pt} pt, no clipping, no overlapping value
  annotations, and no annotation covering its marker.
- Composite: A (with labels) and label-less re-renders of B and C from the same function,
  placed side by side as finished vector PDFs with LaTeX (standalone + includegraphics).
  Nothing is redrawn in a shared axes system, and each panel keeps its own x scale.

Deviations from the specification
1. Width shares. The composite is the full text width ({total:.2f} in). The label column
   ({label_w} in) lives inside A, as asked, so shares are given for the plotting areas:
   A {sa:.0f}%, B {sb:.0f}%, C {sc:.0f}%. As fractions of the whole figure (labels included in
   A) they are A {fa:.0f}%, B {fb:.0f}%, C {fc:.0f}%. Holding A to 27% of the total would leave
   under 0.1 in to plot once the longest label ("MiniLM · SCOTUS Oyez–Lexis", 8 pt) is in.
2. Panel A: the optional "[90%: n PCs]" annotation is omitted. On the low-value rows it runs
   past the panel ("0.27  [90%: 17–19 PCs]"), so those values are in the caption and the data files.
3. Panel B: K is annotated as the silhouette K of the geometry run's seeds (0-2), the seeds
   the ARI is computed on. Where they differ it is shown as a range ("K=4–5" for MiniLM ·
   SCOTUS 3 sources: seeds 5/4/4). The performance cohort's seeds 3-4 of that row chose
   K = 3 and 2. That is recorded in the data files, not in the panel.
4. Panel C: numeric annotations are omitted (18 values would crowd the panel). The three
   markers per row are offset vertically by ±0.22 of a row so near-equal gaps (PC-1 ≈ k-means
   on MiniLM · SCOTUS Oyez–Lexis) stay distinguishable. The method key (one entry per line) sits at
   the top left of the composite, over the setting labels, where the header band is otherwise empty;
   it also explains A's and B's markers (PC-1's square, k-means LEACE's cross). The standalone C
   panel carries the same key over its own label column. The k-means entry reads "k-means LEACE";
   the caption states that K is the silhouette K.
7. Compact headers (2026-09-30, to save vertical space): a bold title and a grey subtitle stating
   what the x axis measures, so there is no x label below the tick labels; C's subtitle also carries
   "lower is better". The earlier grey reading cues ("higher → ...") are dropped: they repeated the
   axis descriptions. Every header starts at its plot's left edge. A and B have two-line titles (their
   titles cannot fit 1.1-1.2 in on one line); A's subtitle also takes two lines, so the header band is
   sized for A's four lines (B and C use three). The setting labels sit {label_pad} pt from panel A.
   Checks: the key overlaps no header text and every header ends above its plot.
5. No error bars: the figure shows seed means; SDs are in every CSV.
6. The separator after row 2 is drawn per panel; in the composite it also crosses A's label column.

8. A and B use plain points (no lollipop stems). With the value printed beside each point
   and the axes starting at 0, the stems added nothing, and near 1.0 they showed as fragments
   on both sides of the label.

9. Row tracing. Settings 2, 4 and 6 sit on a light grey band drawn edge to edge in every
   panel. Because the panels share one row geometry, the bands join into continuous stripes in
   the composite, carrying each setting label across to its points in B and C. Panel C's
   markers sit within ±0.17 of the row centre, clearly inside their row. The near-equal PC-1
   and k-means gaps of MiniLM · SCOTUS Oyez–Lexis (0.355 vs 0.356) overlap by design: they
   are equal to two decimals.

Unavailable quantities: none. Every requested field is present in
data/fig2_geometry_discovery_combined.csv.
"""


def finalize(ex) -> None:
    rep = config.OUT_DIRS["reports"]
    w, sh = _STATE["widths"], _STATE["plot_share"]
    total = sum(w.values())
    (rep / "fig2_geometry_discovery_design_report.txt").write_text(DESIGN.format(
        H=H, BOTTOM=BOTTOM, TOP=TOP, pitch=(H - BOTTOM - TOP) / 6,
        fs=", ".join(f"{k} {v} pt" for k, v in FS.items()), min_pt=MIN_PT, label_pad=LABEL_PAD, total=total, label_w=LABEL_W,
        sa=100 * sh["a"], sb=100 * sh["b"], sc=100 * sh["c"],
        fa=100 * w["a"] / total, fb=100 * w["b"] / total, fc=100 * w["c"] / total))
    ex.output(rep / "fig2_geometry_discovery_design_report.txt")
    L = ["# Figure 2 (three panels + composite): provenance and validation", "",
         f"Build: `python -m paper_assets build {NAME}`. Values from the shared collector (identical to the "
         "earlier Figure 2 versions). Descriptive only.", "",
         "| # | Setting | C | K | capture(C−1) | PCs 90% | ARI | Meta AUC | PC-1 gap | k-means gap | SAE gap | geometry seeds | performance |",
         "|---:|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|---|"]
    for r in _STATE["combined"]:
        L.append(f"| {r['row']} | {r['label']} | {r['C']} | {r['selected_K']} | "
                 f"{r['capture_leading_Cminus1_PCs_mean']:.3f} | {r['pcs_for_90pct']} | {r['ari_mean']:.3f} | "
                 f"{r['metadata_leace_auc_mean']:.3f} | {r['pca_top1_gap_mean']:.3f} | "
                 f"{r['kmeans_auto_gap_mean']:.3f} | {r['sae_leace_gap_mean']:.3f} | {r['geometry_seeds']} | "
                 f"{r['performance_cohort']} {r['performance_seeds']} |")
    L += ["", "## Cohorts", "", "Geometry (A, B) uses seeds 0–2 in every row; C uses each setting's "
          "locked cohort. The populations differ in:"] + [f"- {n_}" for n_ in _STATE["notes"]]
    L += ["", f"## Checks ({len(ex.checks)} passed)", ""] + [f"- {c['message']}" for c in ex.checks]
    L += ["", "## Issues", ""] + ([f"- **{i['level']}** `{i['code']}`: {i['message']}" for i in ex.issues] or ["None."])
    p = rep / f"{NAME}_validation.md"
    p.write_text("\n".join(L) + "\n")
    ex.output(p)
