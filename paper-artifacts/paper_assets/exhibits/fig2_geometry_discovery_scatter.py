"""Main Figure 2: why label-free erasers work in some geometries and fail in others.

Six settings (numbered 1-6 in the key), three panels sharing one y-axis: the
held-out linear-AUC gap to Metadata-LEACE (method - Metadata, per matched seed;
0 = the supervised reference, higher = more residual leakage).

  (a) PC-1's gap against the share of the supervised nuisance subspace that lies
      in PC1 (`capture["1"]` of alternatives.run_geometry: mean squared projection of
      an orthonormal basis of span(X_c^T (Z - Z_bar)), rank C-1, onto the first PC).
      PC1 is the direction PC-1 removes; with C = 3 one direction can hold at most
      half of the rank-2 subspace, which is the coverage limit of Figure 1.
  (b) k-means LEACE's gap against the ARI of the silhouette-K grouping with the
      nuisance classes (`ari.auto`), a post-hoc diagnostic.
  (c) the gaps of all three label-free erasers per setting.

Diagnostics use the geometry run's seeds (0-2); gaps use each setting's locked
cohort. The two populations are recorded separately. Descriptive only: no
correlation, fitted model or test.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from ..lib import config, latex, style
from ..lib.aggregate import paired_difference, stat
from ..lib.io import read
from ..lib.loaders import Cohort, geometry, validate_file
from ..lib.methods import ENCODERS, METHODS
from ..lib.provenance import check

NAME, KIND, TITLE = "fig2_geometry_discovery_scatter", "figure", "Geometry and label-free erasure performance"
LABEL = "fig:geometry-discovery-scatter"
ROWS = [("swiltra", "e5-small"), ("swiltra_de_fr_it", "e5-small"), ("scotus_oyez_lexis", "e5-small"),
        ("scotus_oyez_lexis", "mini"), ("scotus_oyez_lexis_wiki", "e5-small"),
        ("scotus_oyez_lexis_wiki", "mini")]
ROW_LABEL = {"swiltra": "SwiLTra DE–FR", "swiltra_de_fr_it": "SwiLTra DE–FR–IT",
             "scotus_oyez_lexis": "SCOTUS Oyez–Lexis", "scotus_oyez_lexis_wiki": "SCOTUS 3 sources"}
GAPS = ("pca_top1", "kmeans_auto", "sae_leace")
GAP_LABEL = {"pca_top1": "PC-1", "kmeans_auto": "$k$-means LEACE (silhouette $K$)",
             "sae_leace": "SAE-LEACE ($N$ = 32)"}
#: Font of Figure 1 (and of the revised-results figures), for a consistent pair of main figures.
FONT = {"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"}
#: Index of the first SCOTUS setting (a separator in panel c).
CORPUS_BREAK = 2
_STATE: dict = {}


def _by_seed(vals: list) -> str:
    """One value when constant across seeds, else the per-seed values joined by '/'."""
    return str(vals[0]) if len(set(vals)) == 1 else "/".join(map(str, vals))


def _perf_k(coh: Cohort) -> dict:
    """Silhouette K per performance seed, as recorded by the k-means run of the cohort."""
    role = "kcount" if coh.name == "coverage" else "kmeans"
    d = validate_file(coh.paths[role])
    return {p["seed"]: [int(dg["k_auto"]) for dg in p["diagnostics"]] for p in d["per_seed"]}


def _collect() -> tuple[list, list, list]:
    """Per-setting diagnostics and gaps, with every check; returns (cells, csv rows, notes)."""
    cells, rows, notes = [], [], []
    for yi, (key, enc) in enumerate(ROWS):
        s = config.setting(key)
        C = len(s["classes"])
        g = geometry(key, enc)
        check(g is not None and g["n_classes"] == C and g["class_rank"] == C - 1, "f2_geometry_version",
              f"{key}/{enc}: geometry run has C={C}, nuisance subspace rank {C - 1}")
        gd = read(g["file"])
        main = read(Cohort.assigned(key, enc).paths["main"])
        check(gd["config"]["split"] == main["config"]["split"]
              and gd["corpus_stats"]["n_docs"] == main["corpus_stats"]["n_docs"],
              "f2_geometry_matches_runs", f"{key}/{enc}: geometry uses the same corpus version and split policy")
        check("1" in g["capture"] and str(C - 1) in g["capture"], "f2_capture_rank",
              f"{key}/{enc}: capture recorded at rank 1 and at C-1={C - 1}")
        pc1 = stat(g["capture"]["1"])
        capc = stat(g["capture"][str(C - 1)])
        ari = stat(g["ari"]["auto"])
        pc1_var = stat([np.mean([f["pc1_var_share"] for f in p["folds"]]) for p in gd["per_seed"]])
        r90 = [int(f["rank_90"]) for p in gd["per_seed"] for f in p["folds"]]
        k_geo = [k for ks in g["k_auto"] for k in ks]
        coh = Cohort.assigned(key, enc)
        meta = coh.values("metadata_leace", "linear_auc")
        gaps = {m: paired_difference(coh.values(m, "linear_auc"), meta, method=f"{m}-metadata_leace")
                .register(kind="difference") for m in GAPS}
        perf_seeds = meta.seeds
        pk = _perf_k(coh)
        common = sorted(set(g["seeds"]) & set(pk))
        gk = dict(zip(g["seeds"], g["k_auto"]))
        check(all(gk[s_] == pk[s_] for s_ in common), "f2_k_consistent",
              f"{key}/{enc}: silhouette K identical in geometry and performance runs on seeds {common}")
        k_perf = [k for s_ in perf_seeds for k in pk[s_]]
        same_seeds = list(g["seeds"]) == list(perf_seeds)
        if not same_seeds:
            notes.append(f"{ROW_LABEL[key]}, {ENCODERS[enc]}: geometry seeds {g['seeds']}, performance seeds "
                         f"{perf_seeds} ({coh.name} cohort)")
        cells.append({"n": yi + 1, "key": key, "enc": enc, "C": C, "pc1": pc1, "capc": capc, "ari": ari,
                      "pc1_var": pc1_var, "r90": r90, "k": k_geo, "gaps": gaps})
        base = {"setting_number": yi + 1, "setting": key, "encoder": enc, "dataset_version": s["long"],
                "C": C, "nuisance_subspace_rank": C - 1, "geometry_file": g["file"],
                "geometry_run_id": g["run_id"], "geometry_seeds": g["seeds"],
                "geometry_n_seeds": len(g["seeds"]), "performance_cohort": coh.name,
                "performance_seeds": perf_seeds, "performance_n_seeds": len(perf_seeds),
                "same_seed_population": same_seeds}
        rows.append({**base, "panel": "a:x", "quantity": "share of nuisance subspace in PC1 (capture[1])",
                     "mean": pc1["mean"], "sd": pc1["sd"], "n": pc1["n"], "values": g["capture"]["1"]})
        rows.append({**base, "panel": "table", "quantity": f"share of nuisance subspace in top C-1={C - 1} PCs",
                     "mean": capc["mean"], "sd": capc["sd"], "n": capc["n"],
                     "values": g["capture"][str(C - 1)], "pcs_for_90pct_per_fit": r90})
        rows.append({**base, "panel": "caption", "quantity": "PC1 share of total variance",
                     "mean": pc1_var["mean"], "sd": pc1_var["sd"], "n": pc1_var["n"]})
        rows.append({**base, "panel": "b:x", "quantity": "ARI(silhouette k-means, nuisance classes)",
                     "mean": ari["mean"], "sd": ari["sd"], "n": ari["n"], "values": g["ari"]["auto"],
                     "silhouette_K_geometry_fits": k_geo, "silhouette_K_performance_fits": k_perf})
        for m in GAPS:
            d = gaps[m]
            rows.append({**base, "panel": "y (a: pca_top1, b: kmeans_auto, c: all)",
                         "quantity": f"linear AUC gap: {m} - metadata_leace", "mean": d.mean, "sd": d.sd,
                         "n": d.n, "values": d.values,
                         "performance_sources": [x["file"] for x in d.sources],
                         "performance_run_ids": [x["run_id"] for x in d.sources],
                         "negative_gap": d.mean < 0})
    return cells, rows, notes


def _numbered(ax, pts, color, merged: list, panel: str) -> None:
    """Numbered markers; points closer than a marker's width share one marker ("1,2")."""
    fig = ax.figure
    fig.canvas.draw()
    disp = {n: ax.transData.transform((x, y)) for n, x, y, _, _ in pts}
    groups: list[list] = []
    for p in sorted(pts, key=lambda t: t[0]):
        for gr in groups:
            if any(np.hypot(*(disp[p[0]] - disp[q[0]])) < 14 for q in gr):   # ~ marker width (px @150 dpi)
                gr.append(p)
                break
        else:
            groups.append([p])
    for gr in groups:
        for n, x, y, xs, ys in gr:
            ax.errorbar(x, y, xerr=xs, yerr=ys, fmt="none", ecolor=color, elinewidth=0.7, alpha=0.6,
                        capsize=0, zorder=2)
        x = float(np.mean([p[1] for p in gr]))
        y = float(np.mean([p[2] for p in gr]))
        label = ",".join(str(p[0]) for p in sorted(gr))
        ax.text(x, y, label, ha="center", va="center", fontsize=6.5, fontweight="bold", color="white",
                zorder=4, bbox=dict(boxstyle="circle,pad=0.25" if len(gr) == 1 else "round,pad=0.25",
                                    fc=color, ec="white", lw=0.6))
        if len(gr) > 1:
            merged.append(f"panel {panel}: settings {label} share one marker (their points coincide "
                          f"within a marker width: " + "; ".join(f"{p[0]}: ({p[1]:.3f}, {p[2]:.3f})" for p in gr) + ")")


def build(ex) -> None:
    style.apply()
    with plt.rc_context(FONT):
        _draw(ex)


def _draw(ex) -> None:
    cells, rows, notes = _collect()
    fig = plt.figure(figsize=(style.TEXT, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], left=0.095, right=0.985, top=0.84,
                          bottom=0.365, wspace=0.12)
    axA, axB, axC = (fig.add_subplot(gs[0, c]) for c in range(3))
    for ax in (axB, axC):
        ax.sharey(axA)
    merged: list[str] = []
    # Shared y-range: every gap mean +- SD, with 0 (the supervised reference) included.
    ys = [c["gaps"][m].mean + sg * c["gaps"][m].sd for c in cells for m in GAPS for sg in (-1, 1)]
    y0, y1 = min(-0.03, min(ys) - 0.02), max(ys) + 0.03
    axA.set_ylim(y0, y1)
    ref = dict(color="#000000", linewidth=0.8, linestyle="--", zorder=1)
    for ax in (axA, axB, axC):
        ax.axhline(0, **ref)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=7)
    for ax in (axB, axC):
        ax.tick_params(labelleft=False)
    axA.set_ylabel("AUC gap to Metadata-LEACE ↓", fontsize=7.5)
    # (a) PC-1 vs the nuisance share in PC1.
    axA.set_xlim(-0.05, 1.05)
    axA.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axA.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    _numbered(axA, [(c["n"], c["pc1"]["mean"], c["gaps"]["pca_top1"].mean, c["pc1"]["sd"],
                     c["gaps"]["pca_top1"].sd) for c in cells], METHODS["pca_top1"].color, merged, "a")
    axA.set_xlabel("Nuisance share in PC1", fontsize=7.5)
    axA.set_title("(a) PC-1\nIs the nuisance in PC1?", fontsize=7.5, fontweight="bold", loc="left",
                  linespacing=1.3)
    # (b) k-means LEACE vs cluster-class agreement.
    axB.set_xlim(-0.05, 1.05)
    axB.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axB.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    _numbered(axB, [(c["n"], c["ari"]["mean"], c["gaps"]["kmeans_auto"].mean, c["ari"]["sd"],
                     c["gaps"]["kmeans_auto"].sd) for c in cells], METHODS["kmeans_auto"].color, merged, "b")
    axB.set_xlabel("Cluster–class agreement (ARI)", fontsize=7.5)
    axB.set_title("(b) $k$-means LEACE\nDo the clusters find it?", fontsize=7.5, fontweight="bold",
                  loc="left", linespacing=1.3)
    # (c) all three erasers per setting.
    off = dict(zip(GAPS, (-0.24, 0.0, 0.24)))
    for c in cells:
        for m in GAPS:
            d, mt = c["gaps"][m], METHODS[m]
            axC.errorbar(c["n"] + off[m], d.mean, yerr=d.sd, fmt=mt.marker, color=mt.color, markersize=4.5,
                         capsize=1.5, elinewidth=0.8, zorder=3, label=GAP_LABEL[m] if c["n"] == 1 else None)
    axC.axvline(CORPUS_BREAK + 0.5, color="0.6", linewidth=0.7, zorder=0)
    axC.set_xlim(0.45, len(cells) + 0.55)
    axC.set_xticks([c["n"] for c in cells], [str(c["n"]) for c in cells])
    axC.tick_params(axis="x", length=0)
    axC.text(1.5, 0.985, "SwiLTra", transform=axC.get_xaxis_transform(), ha="center", va="top",
             fontsize=7, color="0.35")
    axC.text(4.5, 0.985, "SCOTUS", transform=axC.get_xaxis_transform(), ha="center", va="top",
             fontsize=7, color="0.35")
    axC.set_xlabel("Setting (see key)", fontsize=7.5)
    axC.set_title("(c) All label-free erasers\nGap per setting", fontsize=7.5, fontweight="bold",
                  loc="left", linespacing=1.3)
    # Method legend, then the setting key (with C and the silhouette K).
    h, l = axC.get_legend_handles_labels()
    x0, x1 = axA.get_position().x0, axC.get_position().x1
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=((x0 + x1) / 2, 0.175),
               fontsize=7.5, columnspacing=2.0, handletextpad=0.5, handlelength=1.8)
    # Setting key: two columns (SwiLTra and SCOTUS E5-small | SCOTUS MiniLM and 3 sources).
    key = [f"{c['n']}   {ROW_LABEL[c['key']]} · {ENCODERS[c['enc']]}   (C = {c['C']}, K̂ = {_by_seed(c['k'])})"
           for c in cells]
    colx = [x0 + 0.01, x0 + (x1 - x0) / 2 + 0.01]
    for i, txt in enumerate(key):
        fig.text(colx[i // 3], 0.125 - (i % 3) * 0.044, txt, fontsize=7, ha="left", va="center")
    for ax in (axA, axB, axC):
        lo, hi = ax.get_ylim()
        check(lo <= 0 <= hi, "f2_zero_visible", f"{ax.get_title()[:3]}: the zero reference is on the axis")
    check(all(y0 <= v <= y1 for v in ys), "f2_no_truncation",
          f"every gap mean +- SD lies inside the shared y-axis [{y0:.3f}, {y1:.3f}] (negative gaps kept)")
    ex.note("merged markers: " + ("; ".join(merged) if merged else "none"))
    var = [c["pc1_var"]["mean"] for c in cells]
    caption = (r"Why label-free erasers succeed or fail. All panels share the $y$-axis: held-out linear "
               r"AUC minus Metadata-LEACE's, formed per matched seed (dashed 0 = the supervised "
               r"reference; higher = more residual leakage). Numbers identify the six settings in the "
               r"key. (a)~PC-1's gap against the share of the supervised nuisance subspace (the rank-"
               r"$(C-1)$ span of the label cross-covariance) that lies in the first principal component, "
               r"the direction PC-1 removes. With $C=3$ one direction can hold at most half of that "
               r"subspace (settings 2 and 5), the coverage limit of Figure~\ref{fig:rank-composition}. "
               r"What matters is this alignment, not how much total variance PC1 explains, which is "
               + f"{min(var):.2f}--{max(var):.2f}" + r" in every setting. (b)~$k$-means LEACE's gap against "
               r"the adjusted Rand index between the training-silhouette $k$-means groups and the "
               r"nuisance classes (post hoc; $\hat K$ in the key). Coinciding points share one marker. "
               r"(c)~The gaps of PC-1, $k$-means LEACE and SAE-LEACE ($N=32$) per setting. Diagnostics "
               r"use the labels retrospectively, on seeds 0--2; gaps use each setting's performance "
               r"cohort (SCOTUS/E5-small: coverage seeds 0--2; SwiLTra and MiniLM: seeds 0--4), so the "
               r"two do not always share a seed population. Bars are $\pm$1 sample SD. Descriptive only: "
               r"no correlation or test is computed.")
    style.save(fig, NAME, ex)
    latex.write_figure_include(NAME, caption, LABEL, ex, wide=True)
    latex.write_csv(NAME, rows, ex)
    _STATE.update(cells=cells, rows=rows, notes=notes, merged=merged, caption=caption)


def finalize(ex) -> None:
    cells = _STATE["cells"]
    L = ["# Figure 2 (why label-free erasers work or fail): provenance and validation", "",
         f"Build: `python -m paper_assets build {NAME}`. Descriptive only: no correlations, p-values or "
         "fitted models.", "",
         "## Definitions (from `code/alternatives.py::run_geometry`, unchanged)", "",
         "- Panel (a) x: `capture[\"1\"]` = mean over an orthonormal basis B of span(X_cᵀ(Z − Z̄)) (rank C−1, "
         "the subspace Metadata-LEACE removes) of the squared norm of its projection on the first right "
         "singular vector (PC1) of the centred training embeddings. PC1 is what PC-1 removes. With C = 3 "
         "it is at most 0.5. This is not the PC1 variance share (`pc1_var_share`), which is reported in the "
         "caption only.",
         "- Capture by the top C−1 PCs (`capture[C-1]`) and `rank_90` (PCs needed for 90%) are in the CSV. "
         "They are not plotted: PC-1 removes one direction, so pairing it with C−1 capture would misread "
         "the coverage limit as an alignment failure.",
         "- Panel (b) x: `ari.auto` = ARI between the classes and k-means with the training-silhouette K.",
         "- y (all panels): per-seed linear AUC(method) − linear AUC(Metadata-LEACE), mean and sample SD "
         "(ddof = 1).", "",
         "## Values", "",
         "| # | Setting | Encoder | C | PC1 share of nuisance | top C−1 capture | PCs to 90% | PC1 var. share | ARI | K̂ | PC-1 gap | k-means gap | SAE gap |",
         "|---:|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|---:|"]
    for c in cells:
        g = c["gaps"]
        L.append(f"| {c['n']} | {c['key']} | {c['enc']} | {c['C']} | {c['pc1']['mean']:.3f} | "
                 f"{c['capc']['mean']:.3f} | {_by_seed(c['r90'])} | {c['pc1_var']['mean']:.3f} | "
                 f"{c['ari']['mean']:.3f} | {_by_seed(c['k'])} | {g['pca_top1'].mean:.3f} | "
                 f"{g['kmeans_auto'].mean:.3f} | {g['sae_leace'].mean:.3f} |")
    L += ["", "## Notes", "",
          "- Merged markers: " + ("; ".join(_STATE["merged"]) if _STATE["merged"] else "none"),
          "- Seed populations differ in these rows:"] + [f"  - {n}" for n in _STATE["notes"]]
    L += ["", f"## Checks ({len(ex.checks)} passed)", ""] + [f"- {c['message']}" for c in ex.checks]
    L += ["", "## Issues", ""] + ([f"- **{i['level']}** `{i['code']}`: {i['message']}" for i in ex.issues] or ["None."])
    p = config.OUT_DIRS["reports"] / f"{NAME}_validation.md"
    p.write_text("\n".join(L) + "\n")
    ex.output(p)
