"""Main Figure 1: erasure rank across binary and three-class compositions (E5-small).

2 x 4 panels grouped by corpus: SwiLTra DE-FR and DE-FR-IT, SCOTUS Oyez-Lexis and
Oyez-Lexis-Wikipedia; linear AUC (top), R@10 (bottom). Curves: PCA removal at rank
r, k-means LEACE with fixed K at x = K - 1, SAE-LEACE with MI-selected and with
random latents at x = N. Every point and reference comes from the coverage cohort
(seeds 0-2): the alternatives run and the k-means group-count run on the same
splits, and -- for the random-latent curve -- the band rerun on the same splits and
SAEs (`random_band/`, code/focused_additions.py): five draws per seed from the
latents MI scores (training firing rate in [0.02, 0.98]), averaged within seed.
It replaces the alternatives run's single draw per budget. No five-seed headline
value, no random-direction (non-SAE) curve.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ..lib import config, latex, style
from ..lib.aggregate import SeedValues
from ..lib.loaders import (RANDOM_BAND_DRAWS, RANDOM_BAND_FILE, RANDOM_BAND_POOL, Cohort, random_band,
                           random_band_values)
from ..lib.methods import METHODS
from ..lib.provenance import check, issue

NAME, KIND, TITLE = "fig1_rank_composition", "figure", "Erasure rank across binary and three-class compositions"
LABEL = "fig:rank-composition"
ENCODER = "e5-small"
COLUMNS = ["swiltra", "swiltra_de_fr_it", "scotus_oyez_lexis", "scotus_oyez_lexis_wiki"]
TITLES = {"swiltra": ("SwiLTra", "DE–FR"), "swiltra_de_fr_it": ("SwiLTra", "DE–FR–IT"),
          "scotus_oyez_lexis": ("SCOTUS", "Oyez–Lexis"),
          "scotus_oyez_lexis_wiki": ("SCOTUS", "Oyez–Lexis–Wikipedia")}
XLABEL = "Erasure rank budget (log₂ scale)"
#: Font of the revised-results figures (code/revised_exhibits.py::_plt), used for this figure only.
FONT = {"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"}
METRICS = [("linear_auc", r"Linear-probe AUC $\downarrow$"), ("recall@10", r"R@10 $\uparrow$")]
CURVES = [("pca", r"pca_top(\d+)", "pca_top1", "main", lambda v: v, "PCA removal, rank $r$"),
          ("kmeans", r"kmeans_k(\d+)", "kmeans_auto", "kcount", lambda v: v - 1,
           "$k$-means LEACE, fixed $K$ (at $K-1$)"),
          ("sae", r"sae_mi_n(\d+)", "sae_leace", "main", lambda v: v, "SAE-LEACE, MI top-$N$"),
          ("sae_random", r"sae_random_band_n(\d+)", "sae_random", "band", lambda v: v,
           "SAE-LEACE, $N$ random latents (mean of 5 draws)")]
#: The random-latent curve's source and design (`lib.loaders.random_band`).
BAND_FILE, BAND_DRAWS, BAND_POOL = RANDOM_BAND_FILE, RANDOM_BAND_DRAWS, RANDOM_BAND_POOL
#: Legend labels (shorter than the curve labels; the caption defines each x position).
LEGEND = {"pca": "PCA removal (rank $r$)", "kmeans": "$k$-means LEACE (fixed $K$)",
          "sae": "SAE-LEACE (MI top-$N$)", "sae_random": "SAE-LEACE (random $N$)",
          "metadata_leace": "Metadata-LEACE (supervised)"}
SEEDS = [0, 1, 2]
_STATE: dict = {}
#: Every panel gets exactly this many y ticks.
N_YTICKS = 5
#: Tick steps whose multiples are exact at two decimals, so every row prints "0.00".
STEPS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25)
DECIMALS = 2
#: Column pairs that share a y-axis in each row (one corpus: binary, then three-class).
PAIRS = {"SwiLTra": ("swiltra", "swiltra_de_fr_it"),
         "SCOTUS": ("scotus_oyez_lexis", "scotus_oyez_lexis_wiki")}
#: Metrics whose y-axis is shared within each corpus pair; the others get one axis per panel.
SHARED_Y = {"linear_auc"}


def _ticks(lo: float, hi: float, n: int = N_YTICKS, bounds=(0.0, 1.0)):
    """Axis limits covering [lo, hi] and exactly n evenly spaced ticks inside them.

    Takes the smallest step of `STEPS` whose multiples inside [lo, hi] number n; if a
    step gives fewer, the range is widened to the nearest further multiples (above if
    that stays within `bounds`, else below). No tick leaves [0, 1]."""
    b0, b1 = bounds
    req = (lo, hi)                        # the axis may extend past a bound as headroom ...
    lo, hi = max(lo, b0), min(hi, b1)     # ... but ticks never do
    for step in STEPS:
        first, last = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
        if last - first + 1 > n:
            continue
        while last - first + 1 < n:
            if (last + 1) * step <= b1 + 1e-9:
                last += 1
            elif (first - 1) * step >= b0 - 1e-9:
                first -= 1
            else:
                break
        if last - first + 1 != n:
            continue
        ticks = [round(i * step, 10) for i in range(first, last + 1)]
        margin = 0.015 * (ticks[-1] - ticks[0])
        return ticks, (min(req[0], ticks[0] - margin), max(req[1], ticks[-1] + margin))
    raise ValueError((lo, hi))


def _grid(coh: Cohort, role: str, pattern: str, d: dict | None = None) -> list[tuple[int, str]]:
    """Conditions actually evaluated in every seed (no interpolation, no assumed grid)."""
    d = d if d is not None else coh.file(role)
    sets = [{c for c in p["metrics"] if re.fullmatch(pattern, c)} for p in d["per_seed"]]
    common = set.intersection(*sets)
    check(all(s == common for s in sets), "f1_grid_same_all_seeds",
          f"{coh.setting}: {pattern} grid evaluated in every seed ({sorted(common)})")
    return sorted((int(re.fullmatch(pattern, c)[1]), c) for c in common)


def build(ex) -> None:
    per_seed, summary, val = [], [], []
    style.apply()
    with plt.rc_context(FONT):
        _draw(ex, per_seed, summary, val)


def _draw(ex, per_seed, summary, val) -> None:
    fig = plt.figure(figsize=(style.TEXT, 5.0))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 0.14, 1, 1], left=0.078, right=0.968,
                          top=0.915, bottom=0.2, wspace=0.36, hspace=0.13)
    cols = {k: c for k, c in zip(COLUMNS, (0, 1, 3, 4))}
    axes = np.array([[fig.add_subplot(gs[i, cols[k]]) for k in COLUMNS] for i in range(2)])
    series: dict = defaultdict(dict)

    def record(sv: SeedValues, curve, x, param):
        summary.append(latex.sv_row(sv, curve=curve, x=x, param=param))
        for s, v in zip(sv.seeds, sv.values):
            src = sv.sources[0]
            per_seed.append({"setting": sv.setting, "encoder": sv.encoder, "cohort": sv.cohort,
                             "metric": sv.metric, "curve": curve, "x": x, "param": param,
                             "seed": s, "value": v, "source_file": src["file"],
                             "run_id": src["run_id"], "saved_condition": src["condition"]})

    for j, key in enumerate(COLUMNS):
        coh = Cohort("coverage", key, ENCODER)
        C = len(config.setting(key)["classes"])
        main, kc = coh.file("main"), coh.file("kcount")
        check(main["n_classes"] == C and main["seeds"] == SEEDS and kc["seeds"] == SEEDS,
              "f1_cohort", f"{key}: coverage cohort, seeds {SEEDS}, C={C}")
        coh.compatibility()        # same splits, audit, recipe, corpus, shared values
        band_rel, band = random_band(coh)
        # The rerun's raw file records, per fit, how many latents of the single-draw pool (firing rate
        # >= 0.02) lay above MI's band; 0 everywhere means the two pools were the same set.
        src = config.REPO / band.get("source", "")
        if src.is_file():
            above = [dg["legacy_pool_above_band"] for p in json.loads(src.read_text())["per_seed"]
                     for dg in p["diagnostics"]]
            _STATE.setdefault("pool_above", {})[key] = above
            check(all(a == 0 for a in above), "f1_random_pools_coincide",
                  f"{key}: no latent of the single-draw pool fired above {BAND_POOL[1]} in any fit, so the "
                  f"single-draw and 5-draw curves sample the same latents ({above})")
        else:
            issue("warning", "f1_random_band_raw_missing",
                  f"{key}: {src} not found, so the pool comparison is not checked")
        grids = {name: _grid(coh, role, pat, band if role == "band" else None)
                 for name, pat, _, role, _, _ in CURVES}
        single = [n for n, _ in _grid(coh, "main", r"sae_random_n(\d+)")]
        check([n for n, _ in grids["sae_random"]] == single, "f1_random_band_grid",
              f"{key}: the 5-draw random curve covers the single-draw curve's budgets {single}")
        k_auto = {p["seed"]: [int(dg["k_auto"]) for dg in p["diagnostics"]] for p in kc["per_seed"]}
        for i, (metric, ylab) in enumerate(METRICS):
            ax = axes[i, j]
            refs = {}
            for ref in ("raw", "metadata_leace"):
                sv = coh.values(ref, metric)
                check(sv.seeds == SEEDS, "f1_ref_seeds", f"{key} {ref} {metric}: reference from seeds {SEEDS}")
                refs[ref] = sv
                record(sv, ref, "", "")
            for name, pat, mkey, role, xf, label in CURVES:
                mt = METHODS[mkey]
                pts = []
                for v, cond in grids[name]:
                    sv = (random_band_values(coh, band_rel, band, cond, metric) if role == "band"
                          else coh.values(cond, metric))
                    check(sv.seeds == SEEDS, "f1_point_seeds", f"{key} {cond} {metric}: seeds {SEEDS}")
                    record(sv, name, xf(v), f"{'r' if name == 'pca' else 'K' if name == 'kmeans' else 'N'}={v}")
                    pts.append((xf(v), sv))
                xs = np.array([x for x, _ in pts], float)
                mu = np.array([s.mean for _, s in pts])
                sd = np.array([s.sd for _, s in pts])
                series[(key, metric)][name] = (xs, mu, sd)
                ax.plot(xs, mu, color=mt.color, marker=mt.marker, linestyle=mt.linestyle,
                        markersize=2.7, linewidth=1.0, label=label, zorder=3)
            # Not plotted, but checked: the default SAE-LEACE arm is the N = 32 point, and the
            # silhouette k-means arm is the fixed-K point at each seed's chosen K.
            sae = coh.values("sae_leace", metric, register=False)
            n32 = coh.values("sae_mi_n32", metric, register=False)
            check(sae.values == n32.values, "f1_sae_primary",
                  f"{key} {metric}: SAE-LEACE default (sae_mi) equals the N=32 budget point")
            auto = coh.values("kmeans_auto", metric, register=False)
            for s_, v in zip(auto.seeds, auto.values):
                ks = k_auto[s_]
                fixed = coh.values(f"kmeans_k{ks[0]}", metric, register=False)
                check(len(set(ks)) == 1 and abs(dict(zip(fixed.seeds, fixed.values))[s_] - v) < 1e-12,
                      "f1_kauto_is_fixed_k", f"{key} seed {s_} {metric}: silhouette arm equals fixed K={ks[0]}")
            if i == 0:
                val.append(f"- {key}: silhouette K per seed {k_auto} (not plotted)")
                ax.set_title(f"{TITLES[key][0]} (C = {C})\n{TITLES[key][1]}", fontsize=7.5, pad=4,
                             linespacing=1.25, fontweight="bold")
            _STATE.setdefault("refs", {})[(key, metric)] = refs
            if j == 0:
                ax.set_ylabel(ylab)

    # y-axes. AUC: shared within each corpus pair (binary and three-class side by side).
    # R@10: one axis per panel, since its ranges differ too much between settings to share.
    # A range spans the panel's plotted means and Metadata-LEACE line, plus headroom on
    # the side where Raw lies. Raw is never drawn: its value is printed in every panel,
    # in that headroom. No SD bands are drawn (per-seed values and SDs are in the CSVs).
    raw_sides = {}
    for i, (metric, _) in enumerate(METRICS):
        groups = (PAIRS if metric in SHARED_Y else {TITLES[k][0] + " " + TITLES[k][1]: (k,) for k in COLUMNS})
        for group, pair in groups.items():
            means = []
            for k in pair:
                refs = _STATE["refs"][(k, metric)]
                means.append(refs["metadata_leace"].mean)
                for _, mu, _ in series[(k, metric)].values():
                    means += list(mu)
            lo, hi = max(min(means), 0.0), min(max(means), 1.0)
            raws = {k: _STATE["refs"][(k, metric)]["raw"].mean for k in pair}
            # Side of Raw relative to the curves of ITS OWN panel (the arrow's meaning).
            own = {k: [_STATE["refs"][(k, metric)]["metadata_leace"].mean]
                   + [v for _, mu, _ in series[(k, metric)].values() for v in mu] for k in pair}
            sides = {k: "above" if r >= max(own[k]) else "below" if r <= min(own[k]) else "inside"
                     for k, r in raws.items()}
            pad = {"below": 0.06 * (hi - lo), "above": 0.06 * (hi - lo)}
            for side in set(sides.values()) - {"inside"}:
                pad[side] = 0.2 * (hi - lo)
            ticks, (y0, y1) = _ticks(lo - pad["below"], hi + pad["above"])
            labels = [f"{t:.{DECIMALS}f}" for t in ticks]
            check(all(abs(float(lb) - t) < 1e-12 for lb, t in zip(labels, ticks)) and len(ticks) == N_YTICKS
                  and ticks[0] >= 0 and ticks[-1] <= 1 + 1e-9, "f1_yticks",
                  f"{group} {metric}: {N_YTICKS} ticks, exact at {DECIMALS} decimals, within [0, 1]: {labels}")
            check(all(y0 <= v <= y1 for v in means), "f1_panel_range",
                  f"{group} {metric}: every plotted mean and Metadata-LEACE line lies inside [{y0:.3f}, {y1:.3f}]")
            for jj, k in enumerate(pair):
                ax = axes[i, COLUMNS.index(k)]
                ax.set_ylim(y0, y1)
                ax.set_yticks(ticks, labels if jj == 0 else [""] * len(ticks))
                meta = METHODS["metadata_leace"]
                ax.axhline(_STATE["refs"][(k, metric)]["metadata_leace"].mean, color=meta.color,
                           linestyle=meta.linestyle, linewidth=1.1, zorder=1)
                side, r = sides[k], raws[k]
                arrow = {"above": " ↑", "below": " ↓", "inside": ""}[side]
                top = side != "below"
                ax.text(0.96 if top else 0.04, 0.96 if top else 0.04, f"Raw: {r:.3f}{arrow}",
                        transform=ax.transAxes, ha="right" if top else "left",
                        va="top" if top else "bottom", fontsize=7, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
                check(side != "inside", "f1_raw_outside_curves",
                      f"{k} {metric}: Raw ({r:.3f}) lies {side} every plotted mean of its panel, "
                      f"printed at that edge")
                raw_sides[(k, metric)] = side
    _STATE["raw_sides"] = raw_sides
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.8, 80)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
    for ax in axes[0]:
        ax.tick_params(labelbottom=False)
    # Legend: three columns, one per family (filled column by column): the two cluster/PCA
    # erasers, the two SAE-LEACE variants, and the supervised reference.
    handles = [Line2D([], [], color=METHODS[m].color, marker=METHODS[m].marker, markersize=5,
                      linestyle=METHODS[m].linestyle, linewidth=1.5, label=LEGEND[name])
               for name, _, m, _, _, _ in CURVES]
    handles += [Line2D([], [], color=METHODS["metadata_leace"].color,
                       linestyle=METHODS["metadata_leace"].linestyle, linewidth=1.5,
                       label=LEGEND["metadata_leace"])]
    x0, x1 = axes[1, 0].get_position().x0, axes[1, -1].get_position().x1
    fig.text((x0 + x1) / 2, 0.142, XLABEL, ha="center", va="center", fontsize=8)
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=((x0 + x1) / 2, 0.012),
               frameon=False, fontsize=8, handlelength=2.4, handletextpad=0.5,
               columnspacing=1.6, labelspacing=0.55)
    caption = (r"Erasure rank across binary and three-class compositions (E5-small). Top: held-out "
               r"linear-probe nuisance AUC; bottom: R@10. The $x$-axis is the erasure rank budget on a "
               r"$\log_2$ scale: $r$ principal directions for PCA removal, $K-1$ for $k$-means LEACE "
               r"with $K$ fixed groups, and $N$ latent directions for SAE-LEACE, selected by MI or "
               r"uniformly at random from the same SAE's latents that MI scores (training firing rate in "
               r"$[0.02, 0.98]$; same operator). Each random point averages five draws per seed, so the "
               r"random curve removes the selection signal. It is a nominal (maximum) rank, not a measured effective rank, and equal "
               r"budgets do not imply equal removed variance. Points are means over seeds 0--2 of the "
               r"coverage experiment (the random curve: a rerun on its splits and SAEs); seed SDs are not drawn and are given, with every per-seed "
               r"value, in the accompanying data (they are largest for SAE-LEACE at small $N$, up to "
               r"0.19 in R@10). Metadata-LEACE (dashed) and Raw come from the same seeds. Raw is "
               r"not drawn; its value is printed in each panel, the arrow showing that it lies above "
               r"(AUC) or below (R@10) every curve of that panel. The two AUC panels of each corpus share "
               r"a $y$-axis; each R@10 panel has its own. Only evaluated configurations are plotted, "
               r"and lines only connect them.")
    style.save(fig, NAME, ex)
    latex.write_figure_include(NAME, caption, LABEL, ex, wide=True)
    latex.write_csv(f"{NAME}_per_seed", per_seed, ex)
    latex.write_csv(f"{NAME}_summary", summary, ex)
    _STATE.update(val=val, caption=caption)


def finalize(ex) -> None:
    L = ["# Figure 1 (erasure rank across compositions): provenance and validation", "",
         f"Build: `python -m paper_assets build {NAME}`. Cohort: `coverage` (manifest), E5-small, seeds "
         f"{SEEDS}, for every curve and reference.", "",
         "## Sources", ""]
    L += [f"- `{m['run_id']}`: {m.get('experiment')} {m.get('corpus')} seeds {m.get('seeds')}"
          + (" (documented Colab merge note)" if "merged_note" in m else "")
          for rel, m in sorted(ex.sources.items())]
    L += ["", "## Evaluated grids and silhouette choices", "",
          "Every seed evaluated PCA r in {1,2,3,4,8,16,32,64}, SAE-LEACE N (MI and random) in "
          "{1,2,4,8,12,16,24,32,48,64} and fixed K in {2,3,4,5,6,8} (checked per seed). The x positions "
          "are the saved nominal values; nothing is interpolated.", "",
          "## Random-latent curve", "",
          f"The random SAE-LEACE curve is read from `{BAND_FILE}` (checked against the coverage run: "
          "`random_band_compatible`, `random_band_design`): per seed the mean "
          f"of {BAND_DRAWS} draws from the latents MI scores (training firing rate in {BAND_POOL}), nested "
          "across N, on the published SAEs (the rerun reproduced every fit's MI selection, and its MI top-32 "
          "scores equal the published ones) and the coverage splits and audit. It replaces the alternatives "
          "run's `sae_random_n<N>` (one draw per budget from firing rate >= 0.02). In every fit of these "
          "settings no latent fired on more than 98% of the training documents (check "
          "`f1_random_pools_coincide`; per setting the count per fit: "
          + "; ".join(f"{k} {v}" for k, v in _STATE.get("pool_above", {}).items()) + "), so the two pools "
          "coincide and only the number of draws (and nesting) differs. Per-draw values are kept in the "
          "band files (`sae_random_band_n<N>_draw<j>`).", ""]
    L += _STATE["val"]
    L += ["", "## Notes", "",
          "- The saved runs record no effective rank for these arms (the `matched_rank` diagnostic is "
          "the variance-matched PCA rank, not an effective rank), so the axis is a rank budget "
          "(\"Erasure rank budget\"), described in the caption as nominal.",
          "- The two AUC panels of each corpus share a y-axis; each R@10 panel has its own. Every "
          "panel has five ticks at two decimals, and its range spans the plotted means and the "
          "Metadata-LEACE line. "
          "Raw is never drawn; its value is printed in every panel, on the side where it lies "
          "beyond every plotted mean of that panel (arrow):",
          *[f"  - {k} {m}: Raw {side}" for (k, m), side in _STATE.get("raw_sides", {}).items()],
          "- No SD bands are drawn for any curve; per-seed values and SDs are in the CSVs.",
          "- The coverage and group-count runs share splits, audit and recipe, and hold identical "
          "values for the conditions they share. Checked by `Cohort.compatibility`.",
          "- Five-seed run_id estimates exist for SwiLTra but are not used here. For seeds 0-2 they "
          "equal these values (see reports/validation_report.md, cohort agreement).",
          "", f"## Checks ({len(ex.checks)} passed)", ""]
    L += [f"- {c['message']}" for c in ex.checks]
    L += ["", "## Issues", ""] + ([f"- **{i['level']}** `{i['code']}`: {i['message']}" for i in ex.issues] or ["None."])
    p = config.OUT_DIRS["reports"] / f"{NAME}_validation.md"
    p.write_text("\n".join(L) + "\n")
    ex.output(p)
