"""Figures, drawn from the raw result JSONs and nothing else.

Every function here takes either a live result dict or the path to a saved one,
so restyling a figure never means re-running an SAE. `regenerate_figures` walks a
results tree and redraws each figure into ``figures/<experiment>/`` mirroring the
``raw/<experiment>/`` it came from.

Colour
------
`PALETTE` is the reference categorical palette in its **documented slot order**
(blue, orange, aqua, yellow, magenta, green, violet, red). The order matters: it
is what passes the CVD and normal-vision separation gates on the adjacent
pairlist. Slots are assigned in fixed order and never cycled.

Forms that put every series against every other -- the frontier scatter -- carry a
three-slot cap, because past three the full palette cannot clear the all-pairs
floors. Aqua, yellow and magenta sit below 3:1 contrast on a light surface, so
every chart here ships visible value labels or an axis a reader can read the
number off (the relief rule).
"""

from __future__ import annotations

import os

import numpy as np

from artifacts import load_result, savefig

#: Reference categorical palette, documented slot order. Never re-order: the
#: separation gates are order-dependent on the adjacent pairlist.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
#: Forms where every series is compared with every other (scatter/frontier).
ALL_PAIRS_CAP = 3
INK, MUTED, GRID = "#3d3d3a", "#6b6b63", "#e4e4e0"

#: Stable colour per condition -- colour follows the entity, never its rank, so a
#: figure that drops a baseline does not repaint the survivors.
CONDITION_SLOT = {
    "raw": 0, "metadata_leace": 1, "sae_leace": 2,
    "kmeans_leace_k2": 3, "kmeans_leace_k4": 4, "kmeans_leace_k8": 5,
    "kmeans_leace_auto": 3,
    "pca_top1": 6, "pca_top2": 6, "pca_top4": 6, "pca_top8": 6, "pca_matched": 6,
    "random_latents": 7,
}
CONDITION_LABEL = {
    "raw": "Raw", "metadata_leace": "Metadata oracle", "sae_leace": "SAE (ours)",
    "random_latents": "Random latents", "pca_matched": "PCA top-|S| (size-matched)",
    "kmeans_leace_auto": "k-means LEACE (K auto)",
}
METRIC_LABEL = {
    "leakage_auc": "Leakage AUC",
    "leakage_auc_flip": "Leakage (flip-adj.)",
    "crossfit_leakage_auc": "Leakage (cross-fitted)",
    "nonlinear_leakage_auc_mlp_flip": "Nonlinear leakage (MLP)",
    "nonlinear_leakage_auc_mlp": "Nonlinear leakage (MLP)",
    "nonlinear_leakage_auc_gb_flip": "Nonlinear leakage (GB)",
    "nonlinear_leakage_auc_gb": "Nonlinear leakage (GB)",
    "cluster_purity": "Cluster purity",
    "recall@1": "Recall@1", "recall@10": "Recall@10",
    "gold_sim_spearman": "Gold-similarity rho",
    "neighbor_retention@10": "Neighbourhood retention@10",
}
#: Panel order for `plot_headline`. Explicit rather than "every key in
#: METRIC_LABEL": that dict aliases several spellings of the same measure onto
#: one label, which drew the GB panel twice.
HEADLINE_PLOT_KEYS = (
    "leakage_auc", "leakage_auc_flip", "crossfit_leakage_auc",
    "nonlinear_leakage_auc_mlp_flip", "nonlinear_leakage_auc_gb_flip",
    "cluster_purity", "recall@1", "recall@10",
    "neighbor_retention@10", "gold_sim_spearman",
)
LOWER_IS_BETTER = {"leakage_auc", "leakage_auc_flip", "crossfit_leakage_auc",
                   "nonlinear_leakage_auc_mlp_flip", "nonlinear_leakage_auc_gb_flip",
                   "nonlinear_leakage_auc_gb", "cluster_purity"}


def condition_color(name: str) -> str:
    """Stable colour per condition. Any PCA arm shares the PCA slot, including
    the size-matched one, so a figure that adds it does not repaint the rest."""
    if name.startswith("pca"):
        return PALETTE[CONDITION_SLOT["pca_top1"] % len(PALETTE)]
    return PALETTE[CONDITION_SLOT.get(name, 7) % len(PALETTE)]


def label_of(name: str) -> str:
    if name.endswith("_matched") and name.startswith("pca_top"):
        return f"PCA top-{name[len('pca_top'):-len('_matched')]} (matched to |S|)"
    return CONDITION_LABEL.get(name, name.replace("_", " "))


def _axes(ax, ylabel=None, xlabel=None, title=None):
    """Recessive grid and axes; text in ink, never in a series colour."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, pad=8)
    return ax


def _seeds(n: int) -> str:
    return f"{n} seed" + ("s" if n != 1 else "")


def _chance_line(ax, axis: str = "y", value: float = 0.5, label: bool = False):
    """Draw the chance reference. Returns True if it was in range.

    `label=False` by default: in a small-multiples grid there is no in-panel
    position that stays clear of the bars across every corpus, so the caller
    captions the line once for the whole figure instead.
    """
    lo, hi = (ax.get_ylim() if axis == "y" else ax.get_xlim())
    if not (lo <= value <= hi):
        return False
    (ax.axhline if axis == "y" else ax.axvline)(value, color=INK, ls="--",
                                                lw=1.0, zorder=2)
    if label:
        xy = (0.995, value) if axis == "y" else (value, 0.995)
        coords = ("axes fraction", "data") if axis == "y" else ("data", "axes fraction")
        ax.annotate("chance", xy, xycoords=coords, fontsize=7.5, color=MUTED,
                    ha="right", va="bottom" if axis == "y" else "top",
                    bbox={"fc": "white", "ec": "none", "pad": 1.2})
    return True


def _bar_labels(ax, bars, values, errs=None, fmt="{:.3f}"):
    """Direct value labels -- the relief the low-contrast slots require.

    Anchored above the error-bar cap, not the bar top, so the number never sits
    on the whisker.
    """
    errs = errs if errs is not None else [0.0] * len(bars)
    for b, v, e in zip(bars, values, errs):
        if not np.isfinite(v):
            continue
        top = b.get_height() + (e if np.isfinite(e) else 0.0)
        # White backing and a high zorder: a bar close to a reference line
        # otherwise has its value struck through by that line.
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, top),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=7.5, color=MUTED, zorder=6,
                    bbox={"fc": "white", "ec": "none", "pad": 1.0})


# --------------------------------------------------------------------------- #
# 1. Headline: small multiples, one panel per metric
# --------------------------------------------------------------------------- #
def plot_headline(result, keys=None, conditions=None, out=None, show=False):
    """One panel per metric, conditions along x. Small multiples rather than one
    42-bar chart: each metric has its own scale and its own better-direction."""
    import matplotlib.pyplot as plt

    res = load_result(result)
    agg = res["agg"]
    conds = list(conditions or res["conditions"])
    keys = [kk for kk in (keys or HEADLINE_PLOT_KEYS)
            if any(kk in agg.get(c, {}) for c in conds)]
    ncol = min(4, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.1 * nrow), squeeze=False)
    drew_chance = drew_floor = False

    for i, kk in enumerate(keys):
        ax = _axes(axes[i // ncol][i % ncol], title=METRIC_LABEL.get(kk, kk))
        means = [agg.get(c, {}).get(kk, {}).get("mean", np.nan) for c in conds]
        stds = [agg.get(c, {}).get(kk, {}).get("std", 0.0) for c in conds]
        bars = ax.bar(np.arange(len(conds)), means, 0.68,
                      color=[condition_color(c) for c in conds],
                      yerr=stds, capsize=2.5, ecolor=MUTED,
                      error_kw={"lw": 1.0}, zorder=3)
        _bar_labels(ax, bars, means, stds)
        ax.set_xticks(np.arange(len(conds)))
        ax.set_xticklabels([label_of(c) for c in conds], rotation=35,
                           ha="right", fontsize=8)
        arrow = "lower is better" if kk in LOWER_IS_BETTER else "higher is better"
        ax.set_xlabel(arrow, color=MUTED, fontsize=8)
        # The oracle is the ACHIEVABLE floor at this training size; chance is
        # only the asymptotic one. Drawing the floor stops a reader measuring the
        # label-free method against a bar it could not reach either.
        floor = agg.get("metadata_leace", {}).get(kk, {}).get("mean")
        if floor is not None and "metadata_leace" in conds:
            ax.axhline(floor, color=condition_color("metadata_leace"), ls="-.",
                       lw=1.3, zorder=2)
            drew_floor = True
        if "leakage" in kk:
            drew_chance = _chance_line(ax, "y") or drew_chance
    for j in range(len(keys), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    sp = res["per_seed"][0]["split"]
    # Say how the corpus was split, not just how many seeds: under K-fold the
    # per-fit held-out count is a fifth of the corpus but every document is
    # scored out of sample, and the two read very differently.
    if sp.get("n_folds") and sp.get("foldwise"):
        scope = (f"{sp['n_folds']}-fold x {_seeds(len(res['seeds']))}, "
                 f"{sp['n_train_docs']} train / {sp['n_test_docs_per_fold']} held out per fold")
    elif sp.get("n_folds"):
        scope = (f"{sp['n_folds']}-fold x {_seeds(len(res['seeds']))}, "
                 f"{sp['n_train_docs']} train / {sp['n_test_docs']} scored per fold")
    else:
        scope = (f"{res.get('protocol', '')} holdout, {_seeds(len(res['seeds']))}, "
                 f"{sp['n_test_docs']} held-out documents")
    fig.suptitle(f"{res.get('label', res.get('corpus'))} - "
                 f"{res.get('encoder', '')} ({scope})",
                 color=INK, fontsize=12, y=1.0)
    notes = []
    if drew_floor:
        notes.append("dash-dot = oracle, the floor attainable at this training size")
    if drew_chance:
        notes.append("dashed = chance (0.5)")
    if notes:
        fig.text(0.5, -0.005, "   |   ".join(notes), ha="center",
                 fontsize=8.5, color=MUTED)
    fig.tight_layout()
    return savefig(fig, out or f"headline_{res.get('corpus')}_{res.get('encoder','')}", show)


# --------------------------------------------------------------------------- #
# 2. Protocol comparison
# --------------------------------------------------------------------------- #
def plot_protocol_comparison(result, key="leakage_auc", out=None,
                             exclude=("random_latents",), note_size=13,
                             show=False):
    """Pair holdout vs document holdout, per condition.

    Two series (the protocols), so two categorical slots and a legend. The gap
    between the two bars for the metadata oracle is the whole point of the
    figure: it is how much of the transductive protocol's leakage number came
    from having fit the eraser on the evaluation documents.
    """
    import matplotlib.pyplot as plt

    res = load_result(result)
    protos = res["variant_order"]
    conds = [c for c in res["variants"][protos[0]]["conditions"]
             if c not in exclude
             and all(c in res["variants"][p]["agg"] for p in protos)]
    def _has(kk):
        return any(np.isfinite(res["variants"][p]["agg"].get(c, {}).get(kk, {})
                               .get("mean", np.nan))
                   for p in protos for c in conds)

    # Same reason as in `plot_pseudo_groups`: an absent metric must not leave an
    # empty panel behind its title.
    metrics = [kk for kk in (key, "recall@10") if _has(kk)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.0 * len(metrics), 4.4),
                             squeeze=False)
    axes = axes[0]

    for ax, kk in zip(axes, metrics):
        _axes(ax, ylabel=METRIC_LABEL.get(kk, kk), title=METRIC_LABEL.get(kk, kk))
        x = np.arange(len(conds))
        w = 0.8 / len(protos)
        drew = False
        for i, p in enumerate(protos):
            agg = res["variants"][p]["agg"]
            means = [agg.get(c, {}).get(kk, {}).get("mean", np.nan) for c in conds]
            if not np.any(np.isfinite(means)):
                continue
            stds = [agg.get(c, {}).get(kk, {}).get("std", 0.0) for c in conds]
            bars = ax.bar(x + (i - (len(protos) - 1) / 2) * w, means, w * 0.92,
                          label=f"{p} holdout", color=PALETTE[i], yerr=stds,
                          capsize=2.5, ecolor=MUTED, error_kw={"lw": 1.0}, zorder=3)
            _bar_labels(ax, bars, means, stds)
            drew = True
        if not drew:                       # unreachable after the filter above
            ax.set_title("")
            ax.axis("off")
            continue
        ax.set_xticks(x)
        ax.set_xticklabels([label_of(c) for c in conds], rotation=25, ha="right", fontsize=8)
        if "leakage" in kk:
            _chance_line(ax, "y")
        # Above the axes: with full-height bars there is no in-plot space that
        # stays empty across corpora.
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=2,
                  loc="lower center", bbox_to_anchor=(0.5, 1.06))
        ax.set_title("")
        ax.set_ylabel(METRIC_LABEL.get(kk, kk), color=INK, fontsize=10)

    fig.suptitle(f"Unseen documents - {res.get('label')} ({_seeds(len(res['seeds']))})",
                 color=INK, fontsize=12)
    fig.text(0.5, -0.02, PROTOCOL_NOTE, ha="center", fontsize=note_size,
             color=MUTED)
    fig.tight_layout()
    return savefig(fig, out or f"protocol_{res.get('corpus')}", show)


# --------------------------------------------------------------------------- #
# 3. Sweeps with a variant axis (selection / erasure / lambda / K / corpus size)
# --------------------------------------------------------------------------- #
def plot_variants(result, keys=("leakage_auc", "recall@10", "cluster_purity",
                                "neighbor_retention@10"),
                  condition="sae_leace", out=None, show=False, xlabel=None):
    """One panel per metric; the sweep's variants along x, for one condition.

    Two measures never share an axis -- each metric gets its own panel, which is
    also why a lambda sweep does not become a dual-axis chart.
    """
    import matplotlib.pyplot as plt

    res = load_result(result)
    order = res.get("variant_order") or list(res["variants"])
    keys = [kk for kk in keys
            if any(kk in res["variants"][v]["agg"].get(condition, {}) for v in order)]
    if not keys:
        print(f"[plots] nothing to draw for {res.get('experiment')}")
        return None
    ncol = min(4, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow), squeeze=False)

    ref_handles: dict = {}
    for i, kk in enumerate(keys):
        ax = _axes(axes[i // ncol][i % ncol], title=METRIC_LABEL.get(kk, kk),
                   xlabel=xlabel)
        means = [res["variants"][v]["agg"].get(condition, {}).get(kk, {}).get("mean", np.nan)
                 for v in order]
        stds = [res["variants"][v]["agg"].get(condition, {}).get(kk, {}).get("std", 0.0)
                for v in order]
        bars = ax.bar(np.arange(len(order)), means, 0.66, color=PALETTE[2],
                      yerr=stds, capsize=2.5, ecolor=MUTED,
                      error_kw={"lw": 1.0}, zorder=3)
        _bar_labels(ax, bars, means, stds)
        # Reference lines: what raw and the oracle score under the same protocol.
        for ref, slot, ls in (("raw", 0, ":"), ("metadata_leace", 1, "--")):
            vals = [res["variants"][v]["agg"].get(ref, {}).get(kk, {}).get("mean")
                    for v in order]
            vals = [v for v in vals if v is not None]
            if vals:
                ln = ax.axhline(float(np.mean(vals)), color=PALETTE[slot], ls=ls,
                                lw=1.4, zorder=2)
                ref_handles.setdefault(label_of(ref), ln)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    for j in range(len(keys), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    # Figure-level, between title and panels: an in-axes legend sat on the bars
    # of whichever panel drew it, and there is no in-panel gap that stays empty
    # across every corpus.
    if ref_handles:
        fig.legend(list(ref_handles.values()), list(ref_handles.keys()),
                   loc="upper center", bbox_to_anchor=(0.5, 0.945),
                   ncol=len(ref_handles), frameon=False, fontsize=9,
                   labelcolor=INK)

    fig.suptitle(f"{res.get('experiment', '').replace('_', ' ')} - "
                 f"{res.get('label')} ({res.get('protocol', '')} holdout, "
                 f"{_seeds(len(res['seeds']))})", color=INK, fontsize=12)
    fig.tight_layout()
    stem = out or f"{res.get('experiment')}_{res.get('corpus')}"
    return savefig(fig, stem, show)


def plot_pseudo_groups(result, out=None, show=False):
    """The K sweep, with the label-free silhouette pick marked."""
    import matplotlib.pyplot as plt

    res = load_result(result)
    ks = res["ks"]

    def _series(cond, kk):
        """(means, stds) across the K grid, for one condition and metric."""
        agg = [res["variants"][f"K={K}"]["agg"].get(cond, {}).get(kk, {}) for K in ks]
        return ([a.get("mean", np.nan) for a in agg], [a.get("std", 0.0) for a in agg])

    # Panel only the metrics this corpus actually has. Bills--Tweets is unpaired,
    # so recall@10 does not exist for it; building the panel regardless left its
    # title floating above an empty frame, because `axis("off")` hides the axes
    # but not the title attached to them.
    metrics = [kk for kk in ("leakage_auc", "recall@10")
               if any(np.any(np.isfinite(_series(c, kk)[0]))
                      for c in ("sae_leace", "metadata_leace"))]
    ncol = 1 + len(metrics)
    fig, axes = plt.subplots(1, ncol, figsize=(4.35 * ncol, 3.8), squeeze=False)
    axes = axes[0]
    sil = [res["silhouette"][str(K)] if str(K) in res["silhouette"]
           else res["silhouette"][K] for K in ks]
    ax = _axes(axes[0], xlabel="pseudo-groups K", title="Silhouette (train docs, label-free)")
    ax.plot(ks, sil, "-o", color=PALETTE[0], lw=2, ms=8, zorder=3)
    ax.axvline(res["silhouette_k"], color=PALETTE[1], ls="--", lw=1.4,
               label=f"silhouette pick K={res['silhouette_k']}")
    ref_k = res.get("n_classes", res.get("recipe_k"))
    if ref_k is not None:
        ax.axvline(ref_k, color=PALETTE[2], ls=":", lw=1.4,
                   label=f"number of classes = {ref_k} (reference only)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    for ax, kk in zip(axes[1:], metrics):
        _axes(ax, xlabel="pseudo-groups K", title=METRIC_LABEL.get(kk, kk))
        for cond, slot in (("sae_leace", 2), ("metadata_leace", 1)):
            m, sd = _series(cond, kk)
            if not np.any(np.isfinite(m)):
                continue
            ax.errorbar(ks, m, yerr=sd, fmt="-o", color=PALETTE[slot], lw=2, ms=7,
                        capsize=3, label=label_of(cond), zorder=3)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    fig.suptitle(f"Pseudo-group count - {res.get('label')} "
                 f"({res.get('protocol')} holdout)", color=INK, fontsize=12)
    fig.tight_layout()
    return savefig(fig, out or f"pseudo_groups_{res.get('corpus')}", show)


def _size_axis(ax, sizes, fontsize: int = 8):
    """Log x-axis ticked at the actual corpus sizes and nowhere else.

    Matplotlib keeps its own minor ticks on a log scale, which labelled
    6x10^2 / 2x10^3 / 3x10^3 across the corpus sizes underneath them.
    """
    import matplotlib.ticker as mticker

    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], fontsize=fontsize)


def plot_corpus_size_grid(results: dict, out=None, title=None, show=False):
    """Both corpus-size sweeps in one figure: corpora down, metrics across.

    Rows keep their own x axis: the two corpora are sized in different units
    (pairs and documents) over different grids, so the axis label is repeated
    once per row rather than shared along the bottom.
    """
    import matplotlib.pyplot as plt

    rows = _ordered(results)
    if not rows:
        print("[plots] nothing to draw"); return None
    keys = ("leakage_auc", "nonlinear_leakage_auc_mlp", "nonlinear_leakage_auc_gb")
    cols = [kk for kk in keys
            if any(kk in r["variants"][v]["agg"].get("sae_leace", {})
                   for _, r in rows for v in r["variant_order"])]
    nrow, ncol = len(rows), len(cols)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.7 * nrow),
                             squeeze=False)
    handles: dict = {}
    for i, (_, res) in enumerate(rows):
        order = res["variant_order"]
        sizes = [res["variants"][v]["size"] for v in order]
        for j, kk in enumerate(cols):
            ax = _axes(axes[i][j],
                       title=METRIC_LABEL.get(kk, kk) if i == 0 else None,
                       ylabel="leakage AUC" if j == 0 else None)
            for cond in ("raw", "metadata_leace", "sae_leace"):
                m = [res["variants"][v]["agg"].get(cond, {}).get(kk, {}).get("mean", np.nan)
                     for v in order]
                sd = [res["variants"][v]["agg"].get(cond, {}).get(kk, {}).get("std", 0.0)
                      for v in order]
                ln = ax.errorbar(sizes, m, yerr=sd, fmt="-o", lw=2, ms=7, capsize=3,
                                 color=condition_color(cond), zorder=3)
                handles.setdefault(label_of(cond), ln)
            _chance_line(ax, "y")
            _size_axis(ax, sizes)
            if j == ncol // 2:      # units differ by row, so label once per row
                ax.set_xlabel(f"corpus size ({res['unit']})", color=INK, fontsize=10)
        axes[i][0].set_ylabel(f"{res.get('label', '')}\nleakage AUC", color=INK,
                              fontsize=10)
    if handles:
        fig.legend(list(handles.values()), list(handles.keys()),
                   loc="upper center", bbox_to_anchor=(0.5, 0.955),
                   ncol=len(handles), frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(title or f"Corpus size - both sweeps "
                 f"({rows[0][1].get('protocol')} holdout, "
                 f"{_seeds(len(rows[0][1]['seeds']))})", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return savefig(fig, out or "corpus_size_all", show)


def plot_corpus_size(result, out=None, show=False):
    """Leakage vs corpus size, linear and nonlinear, oracle and SAE."""
    import matplotlib.pyplot as plt

    res = load_result(result)
    order = res["variant_order"]
    sizes = [res["variants"][v]["size"] for v in order]
    keys = ("leakage_auc", "nonlinear_leakage_auc_mlp", "nonlinear_leakage_auc_gb")
    keys = [kk for kk in keys
            if any(kk in res["variants"][v]["agg"].get("sae_leace", {}) for v in order)]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4.0), squeeze=False)

    for ax, kk in zip(axes[0], keys):
        _axes(ax, xlabel=f"corpus size ({res['unit']})",
              ylabel="leakage AUC", title=METRIC_LABEL.get(kk, kk))
        for cond in ("raw", "metadata_leace", "sae_leace"):
            m = [res["variants"][v]["agg"].get(cond, {}).get(kk, {}).get("mean", np.nan)
                 for v in order]
            s = [res["variants"][v]["agg"].get(cond, {}).get(kk, {}).get("std", 0.0)
                 for v in order]
            ax.errorbar(sizes, m, yerr=s, fmt="-o", lw=2, ms=7, capsize=3,
                        color=condition_color(cond), label=label_of(cond), zorder=3)
        _chance_line(ax, "y")
        _size_axis(ax, sizes)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    fig.suptitle(f"Corpus size - {res.get('label')} ({res.get('protocol')} holdout, "
                 f"{_seeds(len(res['seeds']))})", color=INK, fontsize=12)
    fig.tight_layout()
    return savefig(fig, out or f"corpus_size_{res.get('corpus')}", show)


# --------------------------------------------------------------------------- #
# 4. Frontier
# --------------------------------------------------------------------------- #
def _draw_frontier(ax, res, budget_labels: bool = True, legend: bool = True,
                   title: str | None = None):
    """Draw one corpus's frontier into `ax`.

    Shared by `plot_frontier` (one corpus, its own figure) and
    `plot_frontier_grid` (every corpus, one figure), so the two cannot drift.
    """
    curves = {kk: v for kk, v in res["curves"].items() if v}
    if len(curves) > ALL_PAIRS_CAP:
        raise ValueError(f"{len(curves)} curves exceeds the all-pairs cap of "
                         f"{ALL_PAIRS_CAP}; facet instead of adding hues")
    ylab = METRIC_LABEL.get(res["y_metric"], res["y_metric"])
    _axes(ax, xlabel="leakage AUC (lower is better)",
          ylabel=f"{ylab} (higher is better)")
    ax.grid(axis="x", color=GRID, lw=0.8)

    xs_all = [p["leakage"] for pts in curves.values() for p in pts]
    ys_all = [p["y"] for pts in curves.values() for p in pts]
    for r in res["references"].values():
        xs_all.append(r["leakage"])
        ys_all.append(r["y"])
    padx = 0.14 * (max(xs_all) - min(xs_all) or 0.1)
    pady = 0.12 * (max(ys_all) - min(ys_all) or 0.1)
    ax.set_xlim(min(xs_all) - 2.2 * padx, max(xs_all) + padx)
    ax.set_ylim(min(ys_all) - pady, max(ys_all) + 1.5 * pady)

    for i, (sel, pts) in enumerate(curves.items()):
        pts = sorted(pts, key=lambda p: p["n_selected"])
        xs = [p["leakage"] for p in pts]
        ys = [p["y"] for p in pts]
        ax.plot(xs, ys, "-o", color=PALETTE[i], lw=2, ms=8, label=sel, zorder=3,
                markeredgecolor="white", markeredgewidth=1.2)
        # Direct labels only when there is something to tell apart; with one
        # curve the label just collided with the |S| annotation beside it.
        if len(curves) > 1:
            lead = min(pts, key=lambda p: p["leakage"])
            ax.annotate(sel, (lead["leakage"], lead["y"]),
                        textcoords="offset points",
                        xytext=(-12, 9 if i % 2 == 0 else -11), fontsize=9,
                        color=INK, va="center", ha="right",
                        bbox={"fc": "white", "ec": "none", "pad": 1.5})
        if budget_labels and i == 0:   # the budget grid is shared, so label it once
            for p, dy in ((pts[0], -15), (pts[-1], 13)):
                ax.annotate(f"|S|={p['n_selected']}", (p["leakage"], p["y"]),
                            textcoords="offset points", xytext=(0, dy),
                            fontsize=7.5, color=MUTED, ha="center",
                            bbox={"fc": "white", "ec": "none", "pad": 1.2})

    for cond, marker in (("raw", "s"), ("metadata_leace", "D")):
        r = res["references"].get(cond)
        if not r:
            continue
        ax.scatter([r["leakage"]], [r["y"]], marker=marker, s=110, zorder=4,
                   facecolor="white", edgecolor=INK, linewidth=1.6)
        ax.annotate(label_of(cond), (r["leakage"], r["y"]),
                    textcoords="offset points", xytext=(-13, 0), fontsize=9,
                    color=INK, va="center", ha="right",
                    bbox={"fc": "white", "ec": "none", "pad": 1.5})

    _chance_line(ax, "x")
    if legend:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=len(curves),
                  loc="lower center", bbox_to_anchor=(0.5, 1.02))
    if title:
        ax.set_title(title, color=INK, fontsize=12, pad=26 if legend else 8)
    return ax


def plot_frontier(result, out=None, show=False):
    """Leakage against downstream quality as the erasure budget grows.

    Every curve is compared with every other, so this form takes the three-slot
    all-pairs cap; the two reference points are drawn in outline ink rather than
    taking categorical slots. Curves carry both a legend and a direct label, the
    label anchored at the low-leakage end and pushed left so the two curves'
    labels cannot land on each other. Axis limits are taken from the data and
    padded, so the chance line is drawn only when it is actually in range --
    otherwise it would stretch the axis and flatten the frontier into a corner.
    """
    import matplotlib.pyplot as plt

    res = load_result(result)
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    _draw_frontier(ax, res, budget_labels=True, legend=True,
                   title=f"Erasure-budget frontier - {res.get('label')} "
                         f"({res.get('protocol')} holdout, "
                         f"{_seeds(len(res['seeds']))})")
    fig.tight_layout()
    return savefig(fig, out or f"frontier_{res.get('corpus')}", show)


# --------------------------------------------------------------------------- #
# 5. Encoder generalisation
# --------------------------------------------------------------------------- #
def plot_encoder_grid(result, key="leakage_auc", out=None, title_size=12,
                      show=False):
    """Encoders down the y-axis, one small multiple per corpus.

    Ten encoders are never ten hues: the encoder is the *row*, and the three
    conditions take the three all-pairs-safe categorical slots. Each row is a
    dot plot -- raw, oracle and SAE on one line -- so the reader sees how far the
    label-free method travels from raw toward the oracle, per encoder.
    """
    import matplotlib.pyplot as plt

    res = load_result(result)
    cells = {kk: v for kk, v in res.items() if not kk.startswith("_") and "agg" in v}
    corpora_keys = []
    for v in cells.values():
        if v["corpus"] not in corpora_keys:
            corpora_keys.append(v["corpus"])
    if not corpora_keys:
        print("[plots] encoder grid has no finished cells")
        return None

    # One encoder order shared by every panel, so a reader can scan across them.
    # Sorting each panel independently would make the rows mean different things
    # in different columns.
    by_enc: dict = {}
    for v in cells.values():
        by_enc.setdefault(v["encoder"], []).append(v["agg"]["sae_leace"][key]["mean"])
    enc_order = sorted(by_enc, key=lambda e: float(np.mean(by_enc[e])))

    # Height tracks the number of encoders: a fixed height spreads two rows
    # across a page and crushes twelve.
    height = max(2.8, 0.42 * len(enc_order) + 1.9)
    fig, axes = plt.subplots(1, len(corpora_keys),
                             figsize=(4.6 * len(corpora_keys), height), squeeze=False)
    for ax, ck in zip(axes[0], corpora_keys):
        per_enc = {v["encoder"]: v for v in cells.values() if v["corpus"] == ck}
        rows = [(e, per_enc[e]) for e in enc_order if e in per_enc]
        encs = [e for e, _ in rows]
        _axes(ax, xlabel=METRIC_LABEL.get(key, key),
              title=rows[0][1]["label"] if rows else ck)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.grid(axis="y", visible=False)
        for yi, (_, v) in enumerate(rows):
            xs = [v["agg"][c][key]["mean"] for c in ("raw", "metadata_leace", "sae_leace")]
            ax.plot([min(xs), max(xs)], [yi, yi], color=GRID, lw=2, zorder=1)
            for c in ("raw", "metadata_leace", "sae_leace"):
                ax.scatter([v["agg"][c][key]["mean"]], [yi], s=80, zorder=3,
                           color=condition_color(c),
                           label=label_of(c) if yi == 0 else None,
                           edgecolor="white", linewidth=1.2)
        ax.set_yticks(range(len(encs)))
        ax.set_yticklabels(encs, fontsize=9)
        ax.axvline(0.5, color=INK, ls="--", lw=1.0, zorder=2)
        ax.margins(y=0.12)

    # Above the axes: every row is data, so no in-plot corner is reliably free.
    handles, labels = axes[0][0].get_legend_handles_labels()
    # Legend, chance note and title share one size: they are all figure-level
    # annotation, and a smaller legend read as a footnote rather than a key.
    fig.legend(handles, labels, frameon=False, fontsize=title_size, labelcolor=INK,
               ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.955))

    meta = res.get("_meta", {})
    fig.suptitle(f"Encoder generalisation ({meta.get('protocol', '')} holdout, "
                 f"{_seeds(len(meta.get('seeds', [])))})", color=INK,
                 fontsize=title_size)
    fig.text(0.5, 0.005, "dashed line = chance (0.5)", ha="center",
             fontsize=title_size, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    return savefig(fig, out or "encoder_grid", show)


#: Footer both PCA figures carry, so the two cannot end up disagreeing.
PCA_NOTE = "one fit on the whole corpus, no holdout - descriptive only"


def plot_pca_projection_grid(results: dict, out=None, title=None, legend_size=13,
                             note_size=13, show=False):
    """Every corpus's before/after PCA in one figure: corpora down, conditions across.

    Each row carries its own legend, titled with the corpus: the classes differ
    from corpus to corpus (de/en/es/zh, de/fr, bill/tweet, lexis/oyez), so a
    single shared key would be wrong. Legends sit to the right of each row so
    they never land on the scatter.
    """
    import matplotlib.pyplot as plt

    rows = _ordered(results)
    if not rows:
        print("[plots] nothing to draw"); return None
    conds = ["raw", "metadata_leace", "sae_leace"]
    markers = ["o", "^", "s", "D", "v", "P"]
    nrow, ncol = len(rows), len(conds)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.8 * nrow),
                             squeeze=False)

    for i, (_, res) in enumerate(rows):
        classes = res["classes"]
        labels = np.asarray(res["labels"])
        for j, name in enumerate(conds):
            ax = axes[i][j]
            if name not in res["panels"]:
                ax.set_title(""); ax.axis("off"); continue
            pan = res["panels"][name]
            coords = np.asarray(pan["coords"], float)
            ev = pan.get("explained_variance_ratio", [float("nan")] * 2)
            _axes(ax, xlabel=f"PC1 ({ev[0]:.0%} var)",
                  ylabel=f"PC2 ({ev[1]:.0%} var)",
                  title=label_of(name) if i == 0 else None)
            ax.grid(axis="x", color=GRID, lw=0.8)
            for ci, c in enumerate(classes):
                m = labels == c
                ax.scatter(coords[m, 0], coords[m, 1], s=10, alpha=0.65,
                           color=PALETTE[ci % len(PALETTE)],
                           marker=markers[ci % len(markers)],
                           label=c if j == 0 else None, linewidths=0)
            ax.set_aspect("equal", adjustable="datalim")
        # One key per row, to the right of it and titled by corpus, because the
        # classes are corpus-specific.
        h, lab = axes[i][0].get_legend_handles_labels()
        if h:
            leg = axes[i][ncol - 1].legend(
                h, lab, frameon=False, fontsize=legend_size, labelcolor=INK,
                loc="center left", bbox_to_anchor=(1.02, 0.5), markerscale=2.2,
                title=res.get("label", ""))
            leg.get_title().set_color(INK)
            leg.get_title().set_fontsize(legend_size)

    fig.suptitle(title or "Geometry before and after erasure - all corpora",
                 color=INK, fontsize=15)
    fig.tight_layout(rect=(0, 0.025, 0.90, 0.965))
    fig.text(0.45, 0.006, PCA_NOTE, ha="center", fontsize=note_size, color=MUTED)
    return savefig(fig, out or "pca_projection_all", show)


def plot_pca_projection(result, out=None, legend_size=13, note_size=13,
                        show=False):
    """PCA(2) of each condition, coloured by the true class.

    The classes are the categorical series here, so the palette is assigned by
    class in fixed order and every panel keeps the same assignment -- a reader
    compares panels, and a class that changed colour between them would be
    unreadable. Scatter compares every series with every other, so this form
    takes the all-pairs cap: past three classes the marker shape carries
    identity alongside the hue.
    """
    import matplotlib.pyplot as plt

    res = load_result(result)
    classes = res["classes"]
    labels = np.asarray(res["labels"])
    panels = [p for p in ("raw", "metadata_leace", "sae_leace") if p in res["panels"]]
    markers = ["o", "^", "s", "D", "v", "P"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.4),
                             squeeze=False)

    for ax, name in zip(axes[0], panels):
        pan = res["panels"][name]
        coords = np.asarray(pan["coords"], float)
        ev = pan.get("explained_variance_ratio", [float("nan")] * 2)
        _axes(ax, xlabel=f"PC1 ({ev[0]:.0%} var)", ylabel=f"PC2 ({ev[1]:.0%} var)",
              title=label_of(name))
        ax.grid(axis="x", color=GRID, lw=0.8)
        for i, c in enumerate(classes):
            m = labels == c
            ax.scatter(coords[m, 0], coords[m, 1], s=12, alpha=0.65,
                       color=PALETTE[i % len(PALETTE)],
                       marker=markers[i % len(markers)],
                       label=c if ax is axes[0][0] else None,
                       linewidths=0)
        ax.set_aspect("equal", adjustable="datalim")

    handles, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lab, frameon=False, fontsize=legend_size, labelcolor=INK,
               ncol=min(len(classes), 6), loc="upper center",
               bbox_to_anchor=(0.5, 0.96), markerscale=1.8)
    fig.suptitle(f"Geometry before and after erasure - {res.get('label')}",
                 color=INK, fontsize=12)
    fig.text(0.5, 0.005, PCA_NOTE, ha="center", fontsize=note_size, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    return savefig(fig, out or f"pca_projection_{res.get('corpus')}", show)


def _canon_candidate(name: str) -> str:
    low = name.lower()
    if low.startswith("pca"):
        return "pca"
    if "random" in low:
        return "random"
    if "raw atoms" in low:
        return "atoms"
    return "selected"


#: Candidate -> (palette slot, marker, short label) for the geometry figures.
CANDIDATE_STYLE = {
    "selected": (2, "o", "SAE selected ($\\Sigma$-weighted)"),
    "atoms":    (1, "D", "SAE selected (raw atoms)"),
    "random":   (5, "^", "SAE random ($\\Sigma$-weighted)"),
    "pca":      (0, "s", "PCA top-$|S|$"),
}


def plot_subspace_angles_grid(results: dict, exclude=("random",), out=None,
                              title=None, legend_size=13, show=False):
    """How far each candidate removal subspace sits from the confounder subspace.

    One panel, every corpus, in **degrees** rather than capture. Two reasons the
    earlier two-panel form did not work: rank(L) is 1 on three of the four
    corpora, so the principal-angle panel was a line chart with a single point;
    and at rank 1 capture is exactly cos^2(theta), so the capture bars and the
    angle panel encoded the same number twice. Degrees also read directly -- a
    subspace 2 degrees from L contains it, one 25 degrees away does not -- where
    a bar of length 0.99 against one of 0.75 spends most of its axis on a range
    no candidate occupies.

    Corpora with rank(L) > 1 contribute one marker per principal angle, so their
    spread is visible rather than averaged away.
    """
    import matplotlib.pyplot as plt

    rows = _ordered(results)
    if not rows:
        print("[plots] nothing to draw"); return None
    order = [c for c in ("selected", "atoms", "random", "pca") if c not in exclude]
    fig, ax = plt.subplots(figsize=(9.6, 0.62 * len(rows) + 2.3))
    _axes(ax, xlabel="principal angle to the confounder subspace $L$ (degrees)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)

    seen, ymax = {}, 0.0
    for i, (_, res) in enumerate(rows):
        y0 = len(rows) - 1 - i                       # first corpus at the top
        cands = {_canon_candidate(n): c for n, c in res["candidates"].items()}
        angs = [a for c in order if c in cands for a in cands[c]["angles_deg"]]
        if angs:
            ax.plot([min(angs), max(angs)], [y0, y0], color=GRID, lw=2, zorder=1)
            ymax = max(ymax, max(angs))
        for k, c in enumerate(order):
            if c not in cands:
                continue
            slot, marker, lab = CANDIDATE_STYLE[c]
            dy = (k - (len(order) - 1) / 2) * 0.19
            xs = cands[c]["angles_deg"]
            h = ax.scatter(xs, [y0 + dy] * len(xs), s=95, marker=marker,
                           color=PALETTE[slot], zorder=3, edgecolor="white",
                           linewidth=1.1)
            seen.setdefault(lab, h)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r.get("label", k) for k, r in rows][::-1], fontsize=11)
    ax.set_xlim(-1.5, ymax * 1.12)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.axvline(0.0, color=INK, ls="--", lw=1.1, zorder=2)
    ax.annotate("0$^\\circ$ = $L$ fully contained", (0.0, len(rows) - 0.52),
                textcoords="offset points", xytext=(6, 0), fontsize=9,
                color=MUTED, va="center")
    fig.legend(list(seen.values()), list(seen.keys()), frameon=False,
               fontsize=legend_size, labelcolor=INK, ncol=min(len(seen), 3),
               loc="upper center", bbox_to_anchor=(0.5, 0.955), markerscale=1.2)
    ranks = ", ".join(f"{r.get('label')} {r.get('subspace_rank', r['language_rank'])}"
                      for _, r in rows)
    fig.suptitle(title or "What is being removed - distance to the confounder "
                 "subspace", color=INK, fontsize=14)
    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
    fig.text(0.5, 0.005, f"rank($L$): {ranks}.  A corpus of rank $r$ contributes "
             f"$r$ angles per candidate.", ha="center", fontsize=9, color=MUTED)
    return savefig(fig, out or "subspace_geometry_all", show)


def plot_subspace_capture(result, out=None, show=False):
    """How much of the language subspace each candidate removal subspace contains.

    Left: normalised capture per candidate, the headline number. Right: the
    principal-angle profile behind it -- a candidate can post a middling capture
    either by covering part of the subspace well or by covering all of it
    poorly, and only the angles distinguish the two.
    """
    import matplotlib.pyplot as plt

    res = load_result(result)
    names = list(res["candidates"])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    conf = res.get("confounder", "confounder")
    ax = _axes(axes[0], xlabel="capture (mean squared cosine)",
               title=f"Capture of the {conf} subspace")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    vals = [res["candidates"][n]["capture"] for n in names]
    ax.barh(np.arange(len(names)), vals, 0.62,
            color=[PALETTE[i % len(PALETTE)] for i in range(len(names))], zorder=3)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (v, i), textcoords="offset points", xytext=(6, 0),
                    va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, min(1.0, max(vals) * 1.25 + 0.06))

    ax = _axes(axes[1], xlabel="principal angle index", ylabel="angle (degrees)",
               title=f"Principal angles to the {conf} subspace")
    for i, n in enumerate(names):
        a = res["candidates"][n]["angles_deg"]
        ax.plot(np.arange(1, len(a) + 1), a, "-o", lw=2, ms=7,
                color=PALETTE[i % len(PALETTE)], label=n, zorder=3,
                markeredgecolor="white", markeredgewidth=1.0)
    ax.axhline(90, color=INK, ls="--", lw=1.0, zorder=2)
    ax.annotate("orthogonal (90 deg)", (0.99, 90), xycoords=("axes fraction", "data"),
                fontsize=7.5, color=MUTED, ha="right", va="bottom",
                bbox={"fc": "white", "ec": "none", "pad": 1.2})
    ax.set_xticks(np.arange(1, res.get("subspace_rank", res["language_rank"]) + 1))
    ax.set_ylim(0, 100)

    # One legend above both panels: the left panel already names each candidate
    # on its y-axis, and inside the right panel every corner is on a curve.
    handles, lab = ax.get_legend_handles_labels()
    fig.legend(handles, lab, frameon=False, fontsize=8.5, labelcolor=INK,
               ncol=min(len(names), 4), loc="upper center",
               bbox_to_anchor=(0.5, 0.935))

    rank = res.get("subspace_rank", res["language_rank"])
    fig.suptitle(f"What is being removed - {res.get('label', res.get('corpus'))} "
                 f"({conf} subspace rank {rank}, |S| = {res['n_selected']})",
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return savefig(fig, out or f"subspace_geometry_{res.get('corpus')}", show)


def plot_gain_recovery(result, out=None, pretty=True, title_size=11,
                       tick_size=9, clip_at=1.8, show=False):
    """How much of the oracle's gain the label-free method recovers, per metric.

    Drawn in the house style of `plot_headline_compact`: vertical seagreen bars
    (the SAE's colour there), +-1 std error bars with caps, a dashed grid over
    the bars, metric names along x. The dashed reference at 1.0 is where the
    label-free method matches the label-supervised oracle; a bar above it beat
    the oracle on that metric.

    Vertical rather than horizontal on purpose: with bars running left to right
    the value label, the error bar and the 1.0 reference all landed on the same
    few pixels, and the percentages were struck through by the reference line.
    """
    import matplotlib.pyplot as plt

    res = load_result(result)
    rows = [(kk, m) for kk, m in res["metrics"].items()
            if m["recovery"]["mean"] is not None]
    if not rows:
        print("[plots] no metric had a usable oracle gain")
        return None
    vals = [m["recovery"]["mean"] for _, m in rows]
    errs = [m["recovery"]["std"] for _, m in rows]
    # The ratio has the oracle's gain in its denominator, so a metric the
    # oracle barely moved yields a huge recovery (Bills--Tweets GB: the oracle
    # gained 0.028 and the SAE 0.116, hence 413%). Unclipped, one such bar
    # flattens every other against the axis, so the axis is capped at
    # `clip_at`; a bar past it is marked with an arrow and still labelled with
    # its true value.
    tops = [v + e for v, e in zip(vals, errs)]
    ylim_top = min(max(1.15, max(tops) * 1.20), clip_at)

    fig, ax = plt.subplots(figsize=(max(6.0, 1.25 * len(rows)), 5))
    ax.bar(np.arange(len(rows)), vals, 0.68, yerr=errs, capsize=3,
           color=COMPACT_COLOR["sae_leace"], zorder=2)
    # Above the error-bar cap, so a value never sits on its bar or on the
    # reference line -- the two collisions that made the old figure hard to read.
    # A white backing box: a bar whose recovery is near 1.0 puts its label
    # right on the reference line, which otherwise strikes the digits out.
    label_bg = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
    for i, (v, e) in enumerate(zip(vals, errs)):
        if v + e > ylim_top:
            ax.annotate(f"{v:.0%} \u2191", (i, ylim_top), fontweight="bold",
                        textcoords="offset points", xytext=(0, -7), ha="center",
                        va="top", fontsize=tick_size, color=INK, bbox=label_bg,
                        zorder=6)
        else:
            ax.annotate(f"{v:.0%}", (i, v + e), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=tick_size,
                        color=INK, bbox=label_bg, zorder=6)
    ax.axhline(1.0, color=COMPACT_COLOR["metadata_leace"], ls="--", lw=1.4,
               zorder=3, label="matches the oracle")
    ax.legend(loc="best", fontsize=tick_size, framealpha=0.9)

    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([METRIC_LABEL.get(kk, kk) if pretty else kk
                        for kk, _ in rows],
                       rotation=60, ha="right", fontsize=tick_size)
    ax.set_ylabel("fraction of the oracle's gain recovered",
                  fontsize=tick_size + 1)
    ax.set_ylim(0, ylim_top)
    ax.grid(True, axis="y", ls="--", lw=0.5)
    ax.set_title(f"Oracle-gain recovery \u2014 {res.get('label', res.get('corpus'))} "
                 f"({_seeds(res['n_seeds'])})", fontsize=title_size + 1)
    fig.tight_layout()
    return savefig(fig, out or f"gain_recovery_{res.get('corpus')}",
                   show=show, dpi=150)


# --------------------------------------------------------------------------- #
# Merged figures: one file per experiment, every corpus in it
# --------------------------------------------------------------------------- #
#: Corpus reading order for the merged grids, matching `corpora.HEADLINE` and
#: the tables. Anything not listed keeps its dict order and follows.
GRID_ORDER = ("semeval", "swiltra", "bills_tweets", "scotus_oyez_lexis")


def _ordered(results: dict) -> list:
    """``(key, result)`` pairs in the paper's corpus reading order."""
    loaded = {kk: load_result(v) for kk, v in results.items()}
    keys = ([kk for kk in GRID_ORDER if kk in loaded]
            + [kk for kk in loaded if kk not in GRID_ORDER])
    return [(kk, loaded[kk]) for kk in keys]


#: Footer both protocol figures carry. One string, so the single-corpus and
#: merged forms cannot end up saying different things.
PROTOCOL_NOTE = ("Both arms use one 50/50 split so only the protocol varies - a "
                 "controlled contrast, not the headline number for a K-fold corpus.")


def plot_protocol_grid(results: dict, key="leakage_auc", conditions=None,
                       exclude=("random_latents",), out=None, title=None,
                       note_size=13, show=False):
    """Every corpus's pair-vs-document contrast in one figure.

    Corpora down, metrics across. `conditions` restricts and orders the bars;
    by default every condition the corpora share is drawn.
    """
    import matplotlib.pyplot as plt

    rows = _ordered(results)
    if not rows:
        print("[plots] nothing to draw"); return None
    metrics = [kk for kk in (key, "recall@10")
               if any(np.isfinite(r["variants"][p]["agg"].get(c, {}).get(kk, {})
                                  .get("mean", np.nan))
                      for _, r in rows for p in r["variant_order"]
                      for c in r["variants"][p]["agg"])]
    nrow, ncol = len(rows), len(metrics)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 3.6 * nrow),
                             squeeze=False)
    handles: dict = {}
    for i, (_, res) in enumerate(rows):
        protos = res["variant_order"]
        conds = [c for c in (conditions or res["variants"][protos[0]]["conditions"])
                 if c not in exclude
                 and all(c in res["variants"][p]["agg"] for p in protos)]
        x, w = np.arange(len(conds)), 0.8 / len(protos)
        for j, kk in enumerate(metrics):
            ax = _axes(axes[i][j],
                       title=METRIC_LABEL.get(kk, kk) if i == 0 else None)
            drew = False
            for pi, p in enumerate(protos):
                agg = res["variants"][p]["agg"]
                means = [agg.get(c, {}).get(kk, {}).get("mean", np.nan) for c in conds]
                if not np.any(np.isfinite(means)):
                    continue
                stds = [agg.get(c, {}).get(kk, {}).get("std", 0.0) for c in conds]
                bars = ax.bar(x + (pi - (len(protos) - 1) / 2) * w, means, w * 0.92,
                              color=PALETTE[pi], yerr=stds, capsize=2.5,
                              ecolor=MUTED, error_kw={"lw": 1.0}, zorder=3)
                _bar_labels(ax, bars, means, stds)
                handles.setdefault(f"{p} holdout", bars)
                drew = True
            if not drew:                    # metric absent for this corpus
                ax.set_title(""); ax.axis("off"); continue
            if "leakage" in kk:
                _chance_line(ax, "y")
            ax.set_xticks(x)
            if i == nrow - 1:
                ax.set_xticklabels([label_of(c) for c in conds], rotation=25,
                                   ha="right", fontsize=8)
            else:
                ax.set_xticklabels([])
        axes[i][0].set_ylabel(res.get("label", ""), color=INK, fontsize=10)

    if handles:
        fig.legend(list(handles.values()), list(handles.keys()),
                   loc="upper center", bbox_to_anchor=(0.5, 0.955),
                   ncol=len(handles), frameon=False, fontsize=11, labelcolor=INK)
    fig.suptitle(title or f"Unseen documents - all corpora "
                 f"({_seeds(len(rows[0][1]['seeds']))})", color=INK, fontsize=14)
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.text(0.5, 0.008, PROTOCOL_NOTE, ha="center", fontsize=note_size,
             color=MUTED)
    return savefig(fig, out or "protocol_all", show)


def plot_variants_grid(results: dict,
                       keys=("leakage_auc", "recall@10", "cluster_purity",
                             "neighbour_retention@10"),
                       condition="sae_leace", out=None, title=None, show=False):
    """One figure for a variant sweep on every corpus: corpora down, metrics across.

    The per-corpus twin of `plot_variants`. Corpus labels run down the left, so
    each row is one corpus and each column one metric on a shared scale; the raw
    and oracle reference lines are per corpus (they must be) but appear in the
    figure legend once.
    """
    import matplotlib.pyplot as plt

    keys = tuple("neighbor_retention@10" if k == "neighbour_retention@10" else k
                 for k in keys)
    rows = _ordered(results)
    cols = [kk for kk in keys
            if any(kk in r["variants"][v]["agg"].get(condition, {})
                   for _, r in rows for v in r["variant_order"])]
    if not rows or not cols:
        print("[plots] nothing to draw"); return None
    nrow, ncol = len(rows), len(cols)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.85 * nrow),
                             squeeze=False)
    ref_handles: dict = {}
    same_order = len({tuple(r["variant_order"]) for _, r in rows}) == 1

    for i, (_, res) in enumerate(rows):
        order = res["variant_order"]
        for j, kk in enumerate(cols):
            ax = _axes(axes[i][j],
                       title=METRIC_LABEL.get(kk, kk) if i == 0 else None)
            agg = {v: res["variants"][v]["agg"] for v in order}
            means = [agg[v].get(condition, {}).get(kk, {}).get("mean", np.nan)
                     for v in order]
            if not np.any(np.isfinite(means)):
                # Metric absent for this corpus (recall on an unpaired one).
                # Clear the title too: `axis("off")` does not remove it.
                ax.set_title(""); ax.axis("off"); continue
            stds = [agg[v].get(condition, {}).get(kk, {}).get("std", 0.0)
                    for v in order]
            bars = ax.bar(np.arange(len(order)), means, 0.66, color=PALETTE[2],
                          yerr=stds, capsize=2.5, ecolor=MUTED,
                          error_kw={"lw": 1.0}, zorder=3)
            _bar_labels(ax, bars, means, stds)
            for ref, slot, ls in (("raw", 0, ":"), ("metadata_leace", 1, "--")):
                vals = [agg[v].get(ref, {}).get(kk, {}).get("mean") for v in order]
                vals = [v for v in vals if v is not None]
                if vals:
                    ln = ax.axhline(float(np.mean(vals)), color=PALETTE[slot],
                                    ls=ls, lw=1.4, zorder=2)
                    ref_handles.setdefault(label_of(ref), ln)
            ax.set_xticks(np.arange(len(order)))
            if i == nrow - 1 or not same_order:
                ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
            else:
                ax.set_xticklabels([])
        axes[i][0].set_ylabel(res.get("label", ""), color=INK, fontsize=10)

    if ref_handles:
        fig.legend(list(ref_handles.values()), list(ref_handles.keys()),
                   loc="upper center", bbox_to_anchor=(0.5, 0.955),
                   ncol=len(ref_handles), frameon=False, fontsize=9,
                   labelcolor=INK)
    exp = rows[0][1].get("experiment", "").replace("_", " ")
    fig.suptitle(title or f"{exp} - all corpora "
                 f"({rows[0][1].get('protocol', '')} holdout, "
                 f"{_seeds(len(rows[0][1]['seeds']))})", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return savefig(fig, out or f"{rows[0][1].get('experiment', 'variants')}_all",
                   show)


def plot_frontier_grid(results: dict, out=None, title=None, ncol: int = 2,
                       show=False):
    """Every corpus's erasure-budget frontier in one figure, as a 2x2 grid.

    Each panel keeps its own axes: the corpora sit at very different leakage and
    quality levels, and a shared scale would flatten three of the four curves.
    """
    import matplotlib.pyplot as plt

    rows = _ordered(results)
    if not rows:
        print("[plots] nothing to draw"); return None
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 4.6 * nrow),
                             squeeze=False)
    for i, (_, res) in enumerate(rows):
        _draw_frontier(axes[i // ncol][i % ncol], res, budget_labels=True,
                       legend=False, title=res.get('label'))
    for j in range(len(rows), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title or "Erasure-budget frontier - all corpora "
                 f"({rows[0][1].get('protocol')} holdout, "
                 f"{_seeds(len(rows[0][1]['seeds']))})", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return savefig(fig, out or "frontier_all", show)


def plot_pseudo_groups_grid(results: dict, out=None, title=None, show=False):
    """Every corpus's K sweep in one figure: corpora down, the three panels across."""
    import matplotlib.pyplot as plt

    rows = _ordered(results)
    if not rows:
        print("[plots] nothing to draw"); return None
    metrics = ["leakage_auc", "recall@10"]
    nrow, ncol = len(rows), 1 + len(metrics)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow),
                             squeeze=False)
    cond_handles: dict = {}

    for i, (_, res) in enumerate(rows):
        ks = res["ks"]
        sil = [res["silhouette"][str(K)] if str(K) in res["silhouette"]
               else res["silhouette"][K] for K in ks]
        ax = _axes(axes[i][0], xlabel="pseudo-groups K" if i == nrow - 1 else None,
                   title="Silhouette (train docs, label-free)" if i == 0 else None)
        ax.plot(ks, sil, "-o", color=PALETTE[0], lw=2, ms=7, zorder=3)
        ax.axvline(res["silhouette_k"], color=PALETTE[1], ls="--", lw=1.4)
        ref_k = res.get("n_classes")
        if ref_k is not None:
            ax.axvline(ref_k, color=PALETTE[2], ls=":", lw=1.4)
        ax.set_ylabel(res.get("label", ""), color=INK, fontsize=10)
        ax.annotate(f"pick K={res['silhouette_k']}"
                    + (f"; classes={ref_k}" if ref_k is not None else ""),
                    (0.97, 0.93), xycoords="axes fraction", ha="right",
                    va="top", fontsize=8, color=MUTED)

        for j, kk in enumerate(metrics, start=1):
            ax = _axes(axes[i][j],
                       xlabel="pseudo-groups K" if i == nrow - 1 else None,
                       title=METRIC_LABEL.get(kk, kk) if i == 0 else None)
            drew = False
            for cond, slot in (("sae_leace", 2), ("metadata_leace", 1)):
                agg = [res["variants"][f"K={K}"]["agg"].get(cond, {}).get(kk, {})
                       for K in ks]
                m = [a.get("mean", np.nan) for a in agg]
                if not np.any(np.isfinite(m)):
                    continue
                sd = [a.get("std", 0.0) for a in agg]
                ln = ax.errorbar(ks, m, yerr=sd, fmt="-o", color=PALETTE[slot],
                                 lw=2, ms=6, capsize=3, zorder=3)
                cond_handles.setdefault(label_of(cond), ln)
                drew = True
            if not drew:
                ax.set_title(""); ax.axis("off")

    if cond_handles:
        fig.legend(list(cond_handles.values()), list(cond_handles.keys()),
                   loc="upper center", bbox_to_anchor=(0.5, 0.955),
                   ncol=len(cond_handles), frameon=False, fontsize=9,
                   labelcolor=INK)
    fig.suptitle(title or "Pseudo-group count K - all corpora "
                 f"({rows[0][1].get('protocol')} holdout, "
                 f"{_seeds(len(rows[0][1]['seeds']))})", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return savefig(fig, out or "pseudo_groups_all", show)


# --------------------------------------------------------------------------- #
# Redraw everything from raw
# --------------------------------------------------------------------------- #
#: experiment tag -> plotting function.
#: Colours of the three-way headline figure. Kept apart from `PALETTE`: this
#: figure is a fixed three-condition comparison, not a scan over conditions.
COMPACT_COLOR = {"raw": "midnightblue", "metadata_leace": "darkorange",
                 "sae_leace": "seagreen"}
CORE_PLOT_CONDITIONS = ("raw", "metadata_leace", "sae_leace")


def plot_headline_compact(result, keys=None, conditions=None, out=None,
                          pretty=False, title_size=11, tick_size=9, show=False):
    """The three-way headline figure as one row of panels, one per metric.

    The layout of `baselines.plot_baselines`, which produced the reference
    figures in `example_format_results/`: a single row, three bars per panel
    (raw / oracle / SAE) in midnightblue / darkorange / seagreen, +-1 std error
    bars with caps, and a dashed y-grid drawn *over* the bars.

    Defaults reproduce those figures literally, down to the raw metric keys and
    condition names as labels. `pretty=True` swaps in the prose labels of
    `METRIC_LABEL` / `label_of`; `conditions` widens the bar set beyond the
    three core conditions. `title_size` / `tick_size` set the panel-title and
    condition-label point sizes (the reference figures used 9 and 7, which are
    small at print size).
    """
    import matplotlib.pyplot as plt

    res = load_result(result)
    agg = res["agg"]
    conds = [c for c in (conditions or CORE_PLOT_CONDITIONS) if c in agg]
    keys = [kk for kk in (keys or HEADLINE_PLOT_KEYS)
            if any(kk in agg.get(c, {}) for c in conds)]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 5),
                             squeeze=False)
    for ax, kk in zip(axes[0], keys):
        names = [c for c in conds if kk in agg[c]]
        ax.bar(range(len(names)), [agg[c][kk]["mean"] for c in names],
               yerr=[agg[c][kk]["std"] for c in names], capsize=3,
               color=[COMPACT_COLOR.get(c, "grey") for c in names])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([label_of(c) if pretty else c for c in names],
                           rotation=60, ha="right", fontsize=tick_size)
        ax.set_title(METRIC_LABEL.get(kk, kk) if pretty else kk,
                     fontsize=title_size)
        # Left on top of the bars, as in the reference figures (no axisbelow).
        ax.grid(True, axis="y", ls="--", lw=0.5)
    name = res.get("label", res["corpus"]) if pretty else res["corpus"]
    fig.suptitle(f"Metadata-free deconfounding \u2014 {name}")
    fig.tight_layout()
    return savefig(fig, out or f"headline_{res['corpus']}_{res.get('encoder', '')}",
                   show=show, dpi=150)


DISPATCH = {
    "headline": plot_headline_compact,
    "protocol_comparison": plot_protocol_comparison,
    "selection_comparison": plot_variants,
    "erasure_comparison": plot_variants,
    "score_function_sweep": plot_variants,
    "lambda_sweep": plot_variants,
    "pseudo_group_sweep": plot_pseudo_groups,
    "corpus_size_sweep": plot_corpus_size,
    "budget_frontier": plot_frontier,
    "encoder_generalization": plot_encoder_grid,
    "gain_recovery": plot_gain_recovery,
    "pca_projection": plot_pca_projection,
    "subspace_geometry": plot_subspace_capture,
}


def regenerate_figures(section_dir: str | None = None, show: bool = False) -> list:
    """Redraw every figure from the JSONs under ``<section>/raw/``.

    Each figure lands in ``figures/<same sub-path>/``, so a result and its figure
    always sit at mirrored paths -- which is what makes restyling a per-folder
    operation that never touches the pipeline.
    """
    import artifacts

    section_dir = section_dir or artifacts.get_results_dir()
    raw_root = os.path.join(section_dir, "raw")
    if not os.path.isdir(raw_root):
        print(f"[plots] no raw/ under {section_dir}")
        return []

    made = []
    for root, _dirs, files in sorted(os.walk(raw_root)):
        rel = os.path.relpath(root, raw_root)
        rel = "" if rel == "." else rel
        for fn in sorted(f for f in files if f.endswith(".json")):
            path = os.path.join(root, fn)
            try:
                res = load_result(path)
            except Exception as exc:                     # noqa: BLE001
                print(f"[skip] {path}: unreadable ({exc})")
                continue
            if not isinstance(res, dict):
                continue
            tag = res.get("experiment") or res.get("_meta", {}).get("experiment")
            fn_plot = DISPATCH.get(tag)
            if fn_plot is None:
                continue
            prev = artifacts.EXPERIMENT
            try:
                artifacts.set_experiment(rel or None)
                out = fn_plot(res, out=os.path.splitext(fn)[0], show=show)
                if out:
                    made.append(out)
            except Exception as exc:                     # noqa: BLE001
                print(f"[fail] {path}: {type(exc).__name__}: {exc}")
            finally:
                artifacts.set_experiment(prev)
    print(f"\n[plots] redrew {len(made)} figure(s) from {raw_root}")
    return made


# --------------------------------------------------------------------------- #
# Frozen cross-corpus transfer (Figure 1, Figure C1), from `ood.summarize`
# --------------------------------------------------------------------------- #
OOD_TRANSFORMS = ("pca_top1", "metadata_leace", "sae_leace")
OOD_LABEL = {"pca_top1": "PCA-1", "metadata_leace": "Metadata-LEACE",
             "sae_leace": "SAE-LEACE"}
#: A marker per transformation as well as a colour, so the series stay apart
#: without colour vision. Transformations registered at run time (`ood`) take
#: the extra markers in order.
OOD_MARKER = {"pca_top1": "s", "metadata_leace": "D", "sae_leace": "o"}
_EXTRA_MARKERS = ("v", "^", "P", "X")
ENCODER_LABEL = {"e5-small": "E5-small", "mini": "MiniLM", "e5-base-v2": "E5-base-v2"}


def _ood_transforms(s: dict) -> tuple:
    """The transformations a summary holds, in its order."""
    return tuple(s.get("transforms") or OOD_TRANSFORMS)


def _ood_marker(c: str, transforms) -> str:
    if c in OOD_MARKER:
        return OOD_MARKER[c]
    extras = [t for t in transforms if t not in OOD_MARKER]
    return _EXTRA_MARKERS[extras.index(c) % len(_EXTRA_MARKERS)]


def _ood_label(s: dict, c: str) -> str:
    """A condition's display name: the one the run recorded, else the default."""
    return (s.get("labels") or {}).get(c) or OOD_LABEL.get(c, c)


def _ood_offsets(transforms) -> dict:
    """Vertical offsets of a row's points, spread evenly within +-0.3."""
    n = len(transforms)
    return {c: (0.3 - 0.6 * i / (n - 1) if n > 1 else 0.0)
            for i, c in enumerate(transforms)}


def _ood_setting_label(v: dict) -> str:
    return f"{ENCODER_LABEL.get(v['encoder'], v['encoder'])} / {v['donor_label']}"


def _ood_limits(pairs) -> tuple:
    """Padded x-limits spanning every (delta, [lo, hi]) and zero."""
    vals = [0.0] + [x for d, (lo, hi) in pairs for x in (d, lo, hi)]
    lo, hi = min(vals), max(vals)
    pad = 0.06 * (hi - lo or 1.0)
    return lo - pad, hi + pad


def _ood_point(ax, c, delta, ci, y, xlim=None, marker="o"):
    """One transformation's change with its interval. Outside a zoomed axis the
    point is marked at the edge with its value, never dropped silently."""
    lo, hi = ci
    kw = dict(color=condition_color(c), marker=marker, ms=6, mec=INK,
              mew=0.6, zorder=4)
    if xlim is not None and not (xlim[0] <= delta <= xlim[1]):
        edge = xlim[0] if delta < xlim[0] else xlim[1]
        ax.plot([edge], [y], ls="none", **{**kw, "marker": "<" if delta < edge else ">"})
        ax.annotate(f"{delta:+.3f}", (edge, y), textcoords="offset points",
                    xytext=(6 if delta < edge else -6, 0), va="center",
                    ha="left" if delta < edge else "right", fontsize=7, color=MUTED,
                    bbox={"fc": "white", "ec": "none", "pad": 0.8}, zorder=6)
        return
    ax.errorbar([delta], [y], xerr=[[delta - lo], [hi - delta]], ls="none",
                ecolor=condition_color(c), elinewidth=1.4, capsize=2.5, **kw)


def _ood_handles(s: dict, transforms, raw: bool = True) -> list:
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="none", marker=_ood_marker(c, transforms),
                      color=condition_color(c), mec=INK, mew=0.6, ms=7,
                      label=_ood_label(s, c)) for c in transforms]
    if raw:
        handles.append(Line2D([], [], color=INK, lw=1.0, label="Raw (no change)"))
    return handles


def plot_ood_retrieval_changes(summary, xlim=None, out=None, show=False):
    """Figure 1: task-level $\\Delta$nDCG@10 of each frozen transformation.

    Encoders down, donors across. Every panel lists the same five legal tasks
    top to bottom, one point per transformation with its paired 95% bootstrap
    interval; the vertical line at zero is Raw. All panels share one x-axis
    spanning every estimate and interval, so a large decline is never clipped.
    Pass `xlim` for a clearly titled zoom of the near-zero region instead;
    points beyond it are marked at the edge with their value.
    """
    import matplotlib.pyplot as plt

    s = load_result(summary)
    transforms = _ood_transforms(s)
    settings = [s["settings"][k] for k in s["order"]]
    encoders = list(dict.fromkeys(v["encoder"] for v in settings))
    donors = list(dict.fromkeys(v["donor"] for v in settings))
    grid = {(v["encoder"], v["donor"]): v for v in settings}
    tasks = settings[0]["legal"]
    labels = [settings[0]["tasks"][t]["label"] for t in tasks]
    lim = xlim or _ood_limits([(v["changes"][c][t]["delta"], v["changes"][c][t]["ci"])
                               for v in settings for c in transforms for t in tasks])
    off = _ood_offsets(transforms)
    height = 3.3 * len(encoders) + 1.4
    fig, axes = plt.subplots(len(encoders), len(donors), squeeze=False,
                             figsize=(5.8 * len(donors) + 1.6, height))
    ys = np.arange(len(tasks))[::-1]
    for i, enc in enumerate(encoders):
        for j, don in enumerate(donors):
            ax, v = axes[i][j], grid.get((enc, don))
            if v is None:
                ax.axis("off")
                continue
            _axes(ax, title=_ood_setting_label(v))
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color=GRID, lw=0.8)
            ax.axvline(0.0, color=INK, lw=1.0, zorder=2)
            for c in transforms:
                for t, y in zip(tasks, ys):
                    ch = v["changes"][c][t]
                    _ood_point(ax, c, ch["delta"], ch["ci"], y + off[c],
                               xlim=lim if xlim else None,
                               marker=_ood_marker(c, transforms))
            ax.set_xlim(*lim)
            ax.set_ylim(-0.6, len(tasks) - 0.4)
            ax.set_yticks(ys)
            ax.set_yticklabels(labels if j == 0 else [], fontsize=9, color=INK)
            if i == len(encoders) - 1:
                ax.set_xlabel(r"$\Delta$nDCG@10 (transformed $-$ Raw)", color=INK,
                              fontsize=10)
    handles = _ood_handles(s, transforms)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.55 / height),
               ncol=len(handles), frameon=False, fontsize=10, labelcolor=INK)
    title = "Frozen transfer to legal retrieval"
    if xlim:
        title += (f" - zoom to [{xlim[0]:+.2f}, {xlim[1]:+.2f}]; "
                  "values beyond it marked at the edge")
    fig.suptitle(title, color=INK, fontsize=14, y=1 - 0.12 / height)
    n = len(settings[0]["seeds"])
    fig.text(0.5, 0.2 / height, f"Point: change in the score averaged over {n} donor "
             f"fit{'s' if n != 1 else ''}, per task. Bar: paired 95% bootstrap interval, "
             "queries resampled, conditional on the fits.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.45 / height, 1, 1 - 0.95 / height))
    return savefig(fig, out or ("ood_retrieval_changes" + ("_zoom" if xlim else "")), show)


def plot_ood_sts_changes(summary, out=None, show=False):
    """Figure C1: STSBenchmark $\\Delta$Spearman, one row per donor/encoder.

    One point per transformation with its paired 95% bootstrap interval; zero is
    Raw. Its own Spearman axis, independent of the retrieval figure, spanning
    every estimate and interval.
    """
    import matplotlib.pyplot as plt

    s = load_result(summary)
    transforms = _ood_transforms(s)
    settings = [s["settings"][k] for k in s["order"]]
    task = settings[0]["sts"][0]
    lim = _ood_limits([(v["changes"][c][task]["delta"], v["changes"][c][task]["ci"])
                       for v in settings for c in transforms])
    off = _ood_offsets(transforms)
    height = 0.8 * len(settings) + 1.9
    fig, ax = plt.subplots(figsize=(7.8, height))
    _axes(ax, xlabel=r"$\Delta$Spearman (transformed $-$ Raw)",
          title=f"Frozen transfer to {settings[0]['tasks'][task]['label']}")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.axvline(0.0, color=INK, lw=1.0, zorder=2)
    ys = np.arange(len(settings))[::-1]
    for v, y in zip(settings, ys):
        for c in transforms:
            ch = v["changes"][c][task]
            _ood_point(ax, c, ch["delta"], ch["ci"], y + off[c],
                       marker=_ood_marker(c, transforms))
    ax.set_xlim(*lim)
    ax.set_ylim(-0.6, len(settings) - 0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([_ood_setting_label(v) for v in settings], fontsize=9, color=INK)
    # A strip of fixed height under the axes, so the legend never meets the
    # x-label whatever the number of rows.
    fig.tight_layout(rect=(0, 0.45 / height, 1, 1))
    fig.legend(handles=_ood_handles(s, transforms, raw=False), loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=len(transforms), frameon=False,
               fontsize=9, labelcolor=INK)
    return savefig(fig, out or "ood_sts_changes", show)


def plot_ood(summary, out=None, show=False):
    """Both OOD figures from ``ood_summary.json``, under their own names -- the
    `regenerate_figures` entry point, which passes the JSON's name as `out`."""
    plot_ood_sts_changes(summary, show=show)
    return plot_ood_retrieval_changes(summary, show=show)


DISPATCH["ood_summary"] = plot_ood
