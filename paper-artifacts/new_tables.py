"""Tables 3, erasure extent (4), SAE construction and B2 (the ten-encoder study, relative to Raw).

Sources: `results/extensions/raw/` (notebooks/04_extensions.ipynb), `results/main/raw/` (02_main_experiments),
`results/fedlex/raw/` (03_fedlex) and `results/selectors/raw/` (05_latent_selectors). Nothing is fitted or rerun: every cell is read seed by seed from a
saved file and summarised as the mean and sample SD over seeds, or as a per-seed paired difference
with its t-interval. Each table writes its LaTeX, its per-seed and summary CSVs, its provenance and
a compiled preview on the TACL page.
"""
from __future__ import annotations

import numpy as np

import common as C
from paper_assets.lib import config, latex
from paper_assets.lib.provenance import check

T3X = C.FOCUSED + "/table3x/table3x_{setting}_{encoder}{tag}.json"
SEL = C.SELECTORS + "/selectors_{setting}_{encoder}.json"
MAIN2 = [("swiltra_de_fr_it", "SwiLTra (DE/FR/IT)"), ("scotus_oyez_lexis_wiki", "SCOTUS (3 sources)")]
LIN, MLP, R10 = r"Lin.\ AUC$\downarrow$", r"MLP AUC$\downarrow$", r"R@10$\uparrow$"
UTIL = {"cap": ("topic_purity", r"Purity$\uparrow$")}
#: Table 3's rows: (block, [(row id, label, condition)]). A row id is its block's letter and its place in the block, so
#: that the ids run in order down the table. In the first draft's numbering these were rows 1, 2, 3, (3m), 4, 5,
#: (5c), (5s) and 6; the bracketed ones were added in the revision (notebooks/04_extensions.ipynb, Section 3).
T3_BLOCKS = [
    (r"A. Direction selection (Directions-LEACE; $N{=}32$ except A4)", [
        ("A1", "MI-selected SAE directions", "sae_mi"),
        ("A2", "Random SAE directions", "sae_random_band"),
        ("A3", "Random non-SAE directions", "gauss"),
        ("A4", "Non-SAE directions, displ.-matched", "gauss_matched")]),
    (r"B. Operator (same MI-selected latent set $S$ as row A1)", [
        ("B1", "SAE activations + Activation-LEACE", "activations"),
        ("B2", "SAE directions + orthogonal removal", "directions_orth"),
        ("B3", "Orthogonal removal, mean kept", "directions_orth_centered"),
        ("B4", r"Orth.\ removal of $\mathrm{span}(\Sigma D_S^{\top})$", "sigma_orth_centered")]),
    ("C. Erasure extent", [("C1", "PCA, displacement-matched", "pca_matched_var")]),
    ("References", [("", "Raw", "raw"), ("", "Metadata-LEACE (supervised)", "metadata_leace")]),
]


def _rng(v) -> str:
    v = sorted(set(int(x) for x in v))
    return str(v[0]) if len(v) == 1 else f"{v[0]}--{v[-1]}"


def _ci(d, digits: int = 3) -> str:
    """A paired difference: signed mean with its 95% t-interval over seeds in small type."""
    if not d.ok:
        return latex.NA_TEX
    ci = C.t_interval(d)
    out = latex.num(d.mean, digits, signed=True)
    if ci is None:
        return out
    return out + rf"{{\scriptsize\,[{latex.num(ci[0], digits, signed=True)}, {latex.num(ci[1], digits, signed=True)}]}}"


def _p(p: float, digits: int = 3) -> str:
    """A p-value; one below the last printed digit is shown as a bound, never as zero."""
    floor = 10.0 ** -digits
    return rf"$<${latex.num(floor, digits)}" if p < floor else latex.num(p, digits)


def _rows_csv(per_seed: list, summary: list, sv, **extra) -> None:
    if not sv.ok:
        summary.append({**extra, "setting": sv.setting, "encoder": sv.encoder, "method": sv.method, "metric": sv.metric,
                        "status": sv.status, "reason": sv.reason})
        return
    summary.append({**latex.sv_row(sv, **extra)})
    for s, v in zip(sv.seeds, sv.values):
        per_seed.append({**extra, "setting": sv.setting, "encoder": sv.encoder, "method": sv.method, "metric": sv.metric,
                         "seed": s, "value": v, "source_file": sv.sources[0]["file"], "run_id": sv.sources[0]["run_id"],
                         "saved_condition": sv.sources[0]["condition"]})


def _t3x_checks(key: str, enc: str, d: dict, rel: str, arms: str = "full") -> dict:
    """What every table built from a Table 3 run states about it: seeds, design, one SAE per split, and how
    the refit relates to the published fit of the same seeds."""
    tag = f"{key}/{enc}"
    check(d["corpus"] == key and d["encoder"] == enc and d["seeds"] == [int(p["seed"]) for p in d["per_seed"]],
          "t3x_identity", f"{rel}: the Table 3 run of {tag}, seeds {d['seeds']}")
    check(d["design"]["arms"] == arms and d["design"]["n"] == 32 and d["design"]["draws"] == 5,
          "t3x_design", f"{rel}: arms '{arms}', N = 32, five draws per random arm")
    dg = [x for p in d["per_seed"] for x in p["diagnostics"]]
    if arms == "full":
        check(all(x["operators_same_set"] for x in dg), "t3x_same_set",
              f"{tag}: rows A1 and B1-B4 use one MI-selected set of 32 latents in every fit")
    folds = [{int(round(m["audit_folds"])) for m in p["metrics"].values() if "audit_folds" in m} for p in d["per_seed"]]
    check(all(len(f) == 1 for f in folds), "t3x_audit_folds",
          f"{tag}: within each seed every row is audited with the same number of probe folds "
          f"({sorted(set.union(*folds))} over seeds)")
    rep = [p["reproduction"] for p in d["per_seed"] if p["reproduction"].get("reference")]
    free = [abs(v["linear_auc"]) for r in rep for c, v in r["diff"].items()
            if c in ("raw", "pca_top1", "kmeans_leace", "gauss0") and "linear_auc" in v]
    check(bool(free) and max(free) < 5e-4, "t3x_sae_free_rows",
          f"{tag}: the rows that use no SAE (Raw, PCA-1, k-means LEACE, the Gaussian draw) equal the published values "
          f"on the {len(rep)} seeds that have them (largest linear-AUC difference {max(free):.1e}), so splits and audit "
          "are the published ones")
    return {"file": rel, "seeds": d["seeds"], "published_seeds": len(rep),
            "mi_reproduced": sum(bool(r.get("mi_selection_reproduced")) for r in rep),
            "k_reproduced": sum(bool(r.get("k_reproduced")) for r in rep),
            "sae_mi_diff": max([abs(r["diff"]["sae_mi"]["linear_auc"]) for r in rep if "sae_mi" in r["diff"]] or [0.0]),
            "ranks": [x["matched_rank"] for x in dg], "shares": [x["sae_displacement_share"] for x in dg],
            "gauss_n": [n for x in dg for n in x.get("gauss_matched_n", [])],
            "band": [x["mi_scores"]["band_size"] for x in dg], "environment": d["environment"]}


def _refit_sentence(info: dict) -> str:
    n = sum(v["mi_reproduced"] for v in info.values())
    tot = sum(v["published_seeds"] for v in info.values())
    return (r"Every row of a setting is fitted on one SAE per split, refitted for this table: its MI selection equals "
            rf"the published fit's in {n} of the {tot} seeds that were published, so the SAE rows are a new estimate, "
            r"while the rows that use no SAE reproduce the published values.")


# --------------------------------------------------------------------------- #
# Table 3 at five seeds, with the rows added in the second round
# --------------------------------------------------------------------------- #
#: The selection rules of the latent-selector study, as rows of Table 3's first block: (number, label, arm).
SELECTOR_ROWS = [("A5", "Entropy-selected SAE directions", "entropy"), ("A6", "Dip-selected SAE directions", "dip"),
                 ("A7", "Co-firing-selected SAE directions", "cofire"),
                 ("A8", "Co-firing-selected, exclusive pairs", "cofire_neg"), ("A9", "Oracle (MI with the true classes)", "oracle")]
DAGGER = r"$^{\dagger}$"
#: Row A1 on the fits of the dagger rows: their matched baseline, printed first among them.
MATCHED_MI = ("A1", "MI-selected, fits of A5--A9", "mi")
#: The runs behind Table 2's cells, by (setting, encoder): repository-relative file and cohort.
PRIMARY_RUNS = {
    ("swiltra_de_fr_it", "e5-small"): (f"{C.MAIN}/id/id_swiltra_de_fr_it_e5-small.json", "id"),
    ("scotus_oyez_lexis_wiki", "e5-small"): (f"{C.FIVE}/alternatives/alternatives_scotus_oyez_lexis_wiki_e5-small.json", "coverage"),
    ("semeval", "e5-small"): (f"{C.FIVE}/id/id_semeval_e5-small.json", "id5"),
    ("cap", "e5-small"): (f"{C.FIVE}/id/id_cap_e5-small.json", "id5"),
    ("fedlex_domestic", "e5-small"): (f"{C.FEDLEX}/id/id_fedlex_domestic_e5-small.json", "fedlex"),
    ("scotus_oyez_lexis_wiki", "mini"): (f"{C.MAIN}/id/id_scotus_oyez_lexis_wiki_mini.json", "id"),
}
#: Largest per-seed difference allowed between a reference row audited in a Table 3 run and in Table 2's run. Set
#: after seeing the values: Raw is identical; Metadata-LEACE is refitted (float32 rounding in the map), which moves
#: the seeded MLP probe by up to 0.0046 and the linear probe by up to 0.0008, and leaves R@10 unchanged.
REFERENCE_TOL = {("raw", "linear_auc"): 5e-5, ("raw", "mlp_auc"): 5e-5, ("raw", "recall@10"): 5e-5,
                 ("metadata_leace", "linear_auc"): 1e-3, ("metadata_leace", "mlp_auc"): 5e-3,
                 ("metadata_leace", "recall@10"): 5e-5}


def _primary_sae(key: str, enc: str, metric: str):
    """SAE-LEACE (N = 32) of the run behind Table 2, whatever name that run saved it under."""
    rel, cohort = PRIMARY_RUNS[(key, enc)]
    for cond in ("sae_leace", "sae_mi"):
        sv = C.seed_values(rel, key, enc, cond, metric, cohort=cohort, method="sae_leace")
        if sv.ok:
            return sv
    raise ValueError(f"{rel}: no SAE-LEACE condition")


def _selector_file(key: str, enc: str) -> tuple:
    """The latent-selector run of one cell (notebooks/05_latent_selectors.ipynb)."""
    rel = SEL.format(setting=key, encoder=enc)
    d = C.read(rel)
    check(d["corpus"] == key and d["encoder"] == enc and d["seeds"] == C.SEEDS5 and d["design"]["draws"] == 5
          and d["environment"]["sae_device"] == "cpu", "sel_identity",
          f"{rel}: the selector study of {key}/{enc}, seeds 0-4, five random draws, SAEs fitted on a CPU")
    return rel, d


def table3(ex, name: str = "table3_mechanism_controls") -> None:
    """Table 3 on seeds 0-4 with the selection rules that avoid k-means in its first block.

    Rows A1-A4, B1-B4, C1 and the references come from the Table 3 runs, as in the 2026-10-06 table. The
    selector rows come from the latent-selector study, whose SAEs are a separate fit of the same recipe, seeds
    and splits. They are merged as they are (decision 5 of 2026-10-07): the caption gives row A1 on those fits, so that the
    offset is on the page."""
    metrics = [("linear_auc", LIN), ("mlp_auc", MLP), ("recall@10", R10)]
    first = T3_BLOCKS[0][0]
    cells, info, own, reaudit, per_seed, summary = {}, {}, {}, {}, [], []
    for key, _ in MAIN2:
        rel = T3X.format(setting=key, encoder="e5-small", tag="")
        info[key] = _t3x_checks(key, "e5-small", C.read(rel), rel)
        for block, rows in T3_BLOCKS:
            for num, label, cond in rows:
                for metric, _ in metrics:
                    sv = C.seed_values(rel, key, "e5-small", cond, metric, cohort="table3x")
                    cells[(key, cond, metric)] = sv
                    _rows_csv(per_seed, summary, sv, block=block, row=num, condition=label, sae_fits="Table 3 runs")
        srel, d = _selector_file(key, "e5-small")
        check(32 in d["design"]["nonlinear_budgets"] and 32 in d["budgets"], "t3_selector_budget",
              f"{srel}: every selection rule was run and probed with the MLP at N = 32")
        for num, label, arm in [MATCHED_MI] + SELECTOR_ROWS:
            for metric, _ in metrics:
                sv = C.seed_values(srel, key, "e5-small", f"{arm}_n32", metric, cohort="latent_selectors", method=arm)
                cells[(key, arm, metric)] = sv
                _rows_csv(per_seed, summary, sv, block=first, row=num + " (dagger)", condition=label,
                          sae_fits="latent-selector study")
        own[key] = cells[(key, MATCHED_MI[2], "linear_auc")]
        prel, pcohort = PRIMARY_RUNS[(key, "e5-small")]         # the reference rows are Table 2's runs, shown once
        for cond in ("raw", "metadata_leace"):
            for metric, _ in metrics:
                here, ref = cells[(key, cond, metric)], C.seed_values(prel, key, "e5-small", cond, metric, cohort=pcohort)
                gap = max(abs(a - b) for a, b in zip(here.values, ref.values))
                check(ref.ok and ref.seeds == here.seeds and gap <= REFERENCE_TOL[(cond, metric)], "t3_reference_shared",
                      f"{key}: {cond} {metric} of Table 2's run, printed here, and of the Table 3 run differ by at most "
                      f"{gap:.4f} in a seed (means {ref.mean:.4f} and {here.mean:.4f})")
                _rows_csv(per_seed, summary, here, block="not printed", row="", sae_fits="Table 3 runs",
                          condition=f"{cond} as audited in the Table 3 run (the table prints Table 2's run)")
                cells[(key, cond, metric)] = ref
                reaudit[(key, cond, metric)] = gap
        for cond in ("raw", "metadata_leace"):       # rows without an SAE must not depend on which run they come from
            for metric in ("linear_auc", "recall@10"):
                other = C.seed_values(srel, key, "e5-small", cond, metric, cohort="latent_selectors")
                check(abs(cells[(key, cond, metric)].mean - other.mean) <= 5e-4, "t3_sae_free_agree",
                      f"{key}: {cond} {metric} agrees between the Table 3 runs and the selector study "
                      f"({cells[(key, cond, metric)].mean:.4f} and {other.mean:.4f})")
    check(all(sv.ok and sv.seeds == C.SEEDS5 for sv in list(cells.values()) + list(own.values())), "t3x_seeds",
          "every cell is a mean over seeds 0-4, as the caption states")
    nm, ncols = len(metrics), 2 + len(MAIN2) * len(metrics)
    head1 = ["", ""] + [rf"\multicolumn{{{nm}}}{{c}}{{{lab}}}" for _, lab in MAIN2]
    head2 = ["", "Condition"] + [lab for _ in MAIN2 for _, lab in metrics]
    cmid = "".join(rf"\cmidrule(lr){{{3 + i * nm}-{2 + (i + 1) * nm}}}" for i in range(len(MAIN2)))
    row = lambda num, label, cond: [num, label] + [latex.cell(cells[(key, cond, m)]) for key, _ in MAIN2 for m, _ in metrics]  # noqa: E731
    body = []
    for bi, (block, rows) in enumerate(T3_BLOCKS):
        if bi:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{{ncols}}}{{l}}{{\textbf{{{block}}}}} \\[1pt]")
        body += [row(num, label, cond) for num, label, cond in rows]
        if bi == 0:
            body.append(r"\addlinespace[2pt]")
            body += [row(num + DAGGER, label, arm) for num, label, arm in [MATCHED_MI] + SELECTOR_ROWS]
        if bi < len(T3_BLOCKS) - 1:
            body.append(r"\addlinespace[2pt]")
    lines = latex.tabular("ll" + "r" * (ncols - 2), [head1, head2], body, cmidrules=[cmid])
    short = lambda lab: lab.split(" ")[0]                                                       # noqa: E731
    gn = "; ".join(f"{short(lab)} {_rng(info[k]['gauss_n'])}" for k, lab in MAIN2)
    ranks = "; ".join(f"{short(lab)} {_rng(info[k]['ranks'])}" for k, lab in MAIN2)
    caption = (r"\textbf{Candidate directions, intervention operators, and erasure extent.}" "\n"
               r"E5-small results on three-class SwiLTra and SCOTUS (mean $\pm$ sample SD, seeds 0--4). AUCs are held-out macro "
               r"one-vs-rest probe scores; R@10 measures paired retrieval. Random SAE selection (row A2) draws uniformly from "
               r"the same latents that MI scores (training firing rate in $[0.02, 0.98]$; Appendix~\ref{sec:appendix_A3}). "
               r"Row A4 uses the smallest number of random non-SAE directions whose Directions-LEACE map reaches or exceeds "
               r"row A1's normalized squared training displacement "
               r"($N$: " + gn + r"). The random rows average five draws per seed. Rows A5--A8 rank the same eligible latents "
               r"without $k$-means: by the entropy of a latent's firing indicator, by Hartigan's dip statistic of the training "
               r"coordinates along its decoder direction, and by its mean firing mutual information with the other eligible "
               r"latents (all pairs; negatively correlated pairs only). The oracle (row A9) ranks by mutual information with the true "
               r"classes: a post hoc reference, not a label-free method. Row B3 is row B2 with the training mean restored, and "
               r"row B4 removes orthogonally the subspace that Directions-LEACE displaces, $\mathrm{span}(\Sigma D_S^{\top})$, "
               r"also with the mean restored. Displacement-matched PCA (row C1) removes the fewest principal directions whose "
               r"training displacement reaches that of row A1 (Appendix~\ref{sec:appendix_A4}; ranks: "
               + ranks + r"); this matches displacement, not rank or total-variance reduction, and "
               r"Table~\ref{tab:erasure-extent} repeats the comparison on six settings. The SAEs of this table were fitted "
               r"separately from those of Table~\ref{tab:primary-id}, and those of the rows marked $\dagger$ separately again "
               r"(same recipe, seeds and splits throughout); row A1$^{\dagger}$ is row A1 on the latter fits, the baseline for "
               r"rows A5--A9. The reference rows are the runs of Table~\ref{tab:primary-id}.")
    C.write_table(name, lines, caption, "tab:mechanism-controls", ex, wide=True, size=r"\footnotesize", tabcolsep="3pt")
    latex.write_csv(f"{name}_per_seed", per_seed, ex)
    latex.write_csv(f"{name}_summary", summary, ex)
    ex.note("Decision 5 of 2026-10-07: the selector rows are merged as they are. Row A1 on the selector study's fits "
            "is printed as row A1 (dagger): "
            + "; ".join(f"{k} {own[k].mean:.4f} (row A1: {cells[(k, 'sae_mi', 'linear_auc')].mean:.4f})" for k, _ in MAIN2))
    ex.note("Reference rows: Table 2's runs are printed. Audited again in the Table 3 runs they differ per seed by at "
            "most " + "; ".join(f"{k} {c} {m}: {g:.4f}" for (k, c, m), g in reaudit.items() if g > 5e-5)
            + " (Raw: identical). The Metadata-LEACE map is refitted there, and the seeded MLP probe is sensitive to "
            "its float32 rounding.")


# --------------------------------------------------------------------------- #
# Erasure extent: SAE-LEACE against PCA matched in displacement, on six cells
# --------------------------------------------------------------------------- #
EXTENT_CELLS = [("swiltra_de_fr_it", "e5-small", "full"), ("scotus_oyez_lexis_wiki", "e5-small", "full"),
                ("semeval", "e5-small", "full"), ("cap", "e5-small", "full"), ("fedlex_domestic", "e5-small", "full"),
                ("scotus_oyez_lexis_wiki", "mini", "extent")]


def table_erasure_extent(ex, name: str = "table_erasure_extent") -> None:
    per_seed, summary, body, vs_table2 = [], [], [], {}
    for key, enc, arms in EXTENT_CELLS:
        rel = T3X.format(setting=key, encoder=enc, tag="")
        info = _t3x_checks(key, enc, C.read(rel), rel, arms)
        util = UTIL.get(key, ("recall@10", R10))[0]
        row = [C.setting_label(key) + (", MiniLM" if enc == "mini" else ""),
               latex.num(float(np.mean(info["shares"])), 2), _rng(info["ranks"])]
        for metric in ("linear_auc", util):
            a = C.seed_values(rel, key, enc, "sae_mi", metric, cohort="table3x")
            b = C.seed_values(rel, key, enc, "pca_matched_var", metric, cohort="table3x")
            d = C.paired(b, a, method="pca_matched_var-sae_mi")
            for sv, lab in ((a, "SAE-LEACE"), (b, "PCA, displacement-matched"), (d, "matched PCA - SAE-LEACE")):
                _rows_csv(per_seed, summary, sv, condition=lab)
            summary[-1]["t_interval_95"] = C.t_interval(d)
            row += [latex.num(a.mean), latex.num(b.mean), _ci(d)]
        body.append(row)
        here = C.seed_values(rel, key, enc, "sae_mi", "linear_auc", cohort="table3x")
        vs_table2[(key, enc)] = here.mean - _primary_sae(key, enc, "linear_auc").mean
        if (key, enc) == ("fedlex_domestic", "e5-small"):
            body.append(r"\addlinespace[2pt]")
    worst = max(vs_table2, key=lambda k: abs(vs_table2[k]))
    rest = max(abs(v) for k, v in vs_table2.items() if k != worst)
    check(worst == ("scotus_oyez_lexis_wiki", "mini"), "extent_refit_gap",
          f"the SAE column is furthest from Table 2 on MiniLM SCOTUS ({vs_table2[worst]:+.4f}; elsewhere at most {rest:.4f}), "
          "as the caption states")
    head1 = ["", "", ""] + [r"\multicolumn{3}{c}{Linear AUC$\downarrow$}", r"\multicolumn{3}{c}{Utility$\uparrow$}"]
    head2 = ["Setting", "Displ.", "$r^{*}$"] + ["SAE", "PCA", r"PCA $-$ SAE"] * 2
    lines = latex.tabular("lcc" + "rrl" * 2, [head1, head2], body, cmidrules=[r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}"])
    caption = (r"\textbf{Erasure extent: SAE-LEACE against PCA matched in displacement.}" "\n"
               r"Per setting (E5-small unless marked; seeds 0--4), SAE-LEACE at $N{=}32$ and PCA removal at the "
               r"smallest rank $r^{*}$ whose normalized squared training displacement reaches SAE-LEACE's (Displ.: that "
               r"displacement, mean over fits). PCA $-$ SAE is the per-seed paired difference with its 95\% "
               r"$t$-interval over seeds. Utility is R@10, and topic purity for CAP. Both maps of a seed are fitted on "
               r"the same split and audited on the same held-out documents. The first two rows are rows A1 and C1 of "
               r"Table~\ref{tab:mechanism-controls}. As there, the SAEs are fitted separately from those of "
               r"Table~\ref{tab:primary-id} (same recipe, seeds and splits), so the SAE column differs from that table's "
               rf"SAE-LEACE by {latex.num(abs(vs_table2[worst]))} on {C.setting_label(worst[0])} with MiniLM and by at most "
               rf"{latex.num(rest)} elsewhere.")
    ex.note("SAE column minus Table 2's SAE-LEACE linear AUC: "
            + "; ".join(f"{k}/{e} {v:+.4f}" for (k, e), v in vs_table2.items()))
    C.write_table(name, lines, caption, "tab:erasure-extent", ex, wide=True, size=r"\footnotesize", tabcolsep="4pt")
    latex.write_csv(f"{name}_per_seed", per_seed, ex)
    latex.write_csv(f"{name}_summary", summary, ex)


# --------------------------------------------------------------------------- #
# SAE construction: other dictionary sizes and sparsities
# --------------------------------------------------------------------------- #
SAE_CONFIGS = [(8, 32, ""), (16, 64, "_sae_e16_k64"), (8, 64, "_sae_e8_k64"), (4, 16, "_sae_e4_k16")]


def table_sae_construction(ex, name: str = "table_SAE_construction_block") -> None:
    per_seed, summary, body = [], [], []
    skipped = C.read(f"{C.FOCUSED}/table3x/sae_configs_skipped.json")["skipped"]
    for si, (key, lab) in enumerate(MAIN2):
        if si:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{7}}{{l}}{{\textbf{{{lab}}}}} \\[1pt]")
        for ef, k, tag in SAE_CONFIGS:
            rel = T3X.format(setting=key, encoder="e5-small", tag=tag)
            sae = rf"${ef}d$ latents, $k{{=}}{k}$" + (" (the paper's)" if not tag else "")
            skip = next((z for z in skipped if z["setting"] == key and z["expansion_factor"] == ef and z["k"] == k), None)
            if skip is not None:
                body.append([sae, str(skip["band_size"])] + [r"\multicolumn{5}{l}{\textit{not comparable at $N{=}32$: "
                            rf"too few eligible latents (seed {skip['seed']})}}}}"])
                continue
            d = C.read(rel)
            _t3x_checks(key, "e5-small", d, rel, "full" if not tag else "core")
            seeds = [0, 1, 2]
            dg = [x for p in d["per_seed"] if int(p["seed"]) in seeds for x in p["diagnostics"]]
            mi = C.seed_values(rel, key, "e5-small", "sae_mi", "linear_auc", cohort="table3x", seeds=seeds)
            rnd = C.seed_values(rel, key, "e5-small", "sae_random_band", "linear_auc", cohort="table3x", seeds=seeds)
            pca = C.seed_values(rel, key, "e5-small", "pca_matched_var", "linear_auc", cohort="table3x", seeds=seeds)
            d1, d2 = C.paired(rnd, mi, "sae_random_band-sae_mi"), C.paired(pca, mi, "pca_matched_var-sae_mi")
            check(mi.seeds == seeds, "sae_cfg_seeds", f"{key} {ef}d k={k}: seeds 0-2, as the caption states")
            for sv, cond in ((mi, "MI"), (rnd, "random"), (pca, "matched PCA"), (d1, "random - MI"), (d2, "PCA - MI")):
                _rows_csv(per_seed, summary, sv, sae=f"{ef}d, k={k}", condition=cond)
            body.append([sae, _rng([x["mi_scores"]["band_size"] for x in dg]),
                         latex.num(float(np.mean([x["sae_displacement_share"] for x in dg])), 2),
                         _rng([x["matched_rank"] for x in dg]), latex.cell(mi), _ci(d1), _ci(d2)])
    head1 = ["", "", "", ""] + [r"\multicolumn{3}{c}{Linear AUC$\downarrow$}"]
    head2 = ["TopK SAE", "Eligible", "Displ.", "$r^{*}$", "MI-selected", r"Random $-$ MI", r"Matched PCA $-$ MI"]
    lines = latex.tabular("lccc" + "rll", [head1, head2], body, cmidrules=[r"\cmidrule(lr){5-7}"])
    caption = (r"\textbf{SAE construction.}" "\n"
               r"Table~\ref{tab:mechanism-controls}'s core comparisons with other TopK dictionaries (latents as a "
               r"multiple of the embedding dimension $d$, and $k$ active latents; E5-small, seeds 0--2, $N{=}32$). "
               r"Eligible: latents per fit with a training firing rate in $[0.02, 0.98]$, the set MI scores. Displ.: "
               r"normalized squared training displacement of MI-selected SAE-LEACE, and $r^{*}$ the rank of the PCA "
               r"matched to it. Linear AUC of MI-selected SAE-LEACE (mean $\pm$ sample SD), and the per-seed paired "
               r"differences to it of random SAE directions (five draws per seed) and of the matched PCA, with 95\% "
               r"$t$-intervals over seeds. A construction with fewer than 32 eligible latents in some fit cannot be "
               r"compared at $N{=}32$.")
    C.write_table(name, lines, caption, "tab:sae-construction", ex, wide=True, size=r"\footnotesize", tabcolsep="5pt")
    latex.write_csv(f"{name}_per_seed", per_seed, ex)
    latex.write_csv(f"{name}_summary", summary, ex)


# --------------------------------------------------------------------------- #
# The ten-encoder study on the settings run after the original study
# --------------------------------------------------------------------------- #
ENC10 = ("mini", "gist-small", "gist-base", "gist-large", "e5-small", "e5-base", "e5-large", "mpnet", "nomic-v2", "mxb")
#: (setting, column header, encoders run, result tree). The three-class cells of all ten encoders are in
#: results/extensions (seven run in the first round, the other six cells later; `focused_extensions.run_encoders_3class`).
LATER_SETTINGS = [("swiltra_de_fr_it", "SwiLTra (3 lang.)", ENC10, C.FOCUSED),
                  ("scotus_oyez_lexis_wiki", "SCOTUS (3 src.)", ENC10, C.FOCUSED),
                  ("fedlex_domestic", "Fedlex", ENC10, C.FEDLEX)]


def _later_cells(order, methods, ref: str) -> tuple:
    """The later settings in the cell layout of the first draft's ten-encoder table: per encoder and method, the
    per-seed paired difference to Metadata-LEACE in linear AUC and in R@10. An encoder that a setting was not
    run with gets a not-applicable cell. Also, per encoder, the k-means diagnostics of Panel C (ARI with the
    classes, share of the confounder subspace in the leading C - 1 principal components), and the per-cell values
    that the table does not print."""
    from paper_assets.lib.aggregate import SeedValues
    cells, ucells, diag, extra = {}, {}, {}, []
    for key, _, encoders, root in LATER_SETTINGS:
        for enc in order:
            if enc not in encoders:
                for m in methods:
                    for store, metric in ((cells, "linear_auc"), (ucells, "recall@10")):
                        store[(enc, key, m)] = SeedValues(key, enc, f"{m}-{ref}", metric, cohort="encoder_study",
                                                          status="not_applicable", reason="the setting was not run with this encoder")
                continue
            rel = f"{root}/encoders/encoder_{key}_{enc}.json"
            d = C.read(rel)
            check(d["seeds"] == [0, 1, 2], "enc_seeds", f"{rel}: seeds 0-2, as the caption states")
            if key == "fedlex_domestic":
                km_rel, folds = rel, [q["geometry"] for q in d["per_seed"]]
            else:
                km_rel = f"{root}/alternatives/generous_k_{key}_{enc}.json"
                gk, geo = C.read(km_rel), C.read(f"{root}/alternatives/geometry_{key}_{enc}.json")
                check(gk["seeds"] == d["seeds"] == geo["seeds"]
                      and all(abs(x["metrics"][ref]["linear_auc"] - y["metrics"][ref]["linear_auc"]) < 1e-12
                              for x, y in zip(d["per_seed"], gk["per_seed"])),
                      "enc_same_fits", f"{key}/{enc}: the encoder cell, its k-means run and its geometry run share seeds "
                      "and Metadata-LEACE values (the same splits)")
                folds = [f for q in geo["per_seed"] for f in q["folds"]]
            for store, metric in ((cells, "linear_auc"), (ucells, "recall@10")):
                for m in methods:
                    src = km_rel if m == "kmeans_auto" else rel
                    v = C.seed_values(src, key, enc, m, metric, cohort="encoder_study")
                    r = C.seed_values(src, key, enc, ref, metric, cohort="encoder_study", method=f"{ref} ({m} run)")
                    store[(enc, key, m)] = C.paired(v, r, f"{m}-{ref}")
            n_cls = len(d["corpus_stats"]["classes"])
            diag[(enc, key)] = (float(np.mean([f["ari"]["auto"] for f in folds])),
                                float(np.mean([f["capture"][str(n_cls - 1)] for f in folds])))
            raw, meta = (C.seed_values(rel, key, enc, c, "linear_auc", cohort="encoder_study") for c in ("raw", ref))
            extra.append({"setting": key, "encoder": enc, "dim": d["dim"], "raw_linear_auc": raw.mean,
                          "metadata_leace_linear_auc": meta.mean,
                          **{f"gap_{m}_mean": cells[(enc, key, m)].mean for m in methods},
                          **{f"gap_{m}_sd": cells[(enc, key, m)].sd for m in methods},
                          **{f"r10_gap_{m}_mean": ucells[(enc, key, m)].mean for m in methods},
                          **{f"r10_gap_{m}_sd": ucells[(enc, key, m)].sd for m in methods},
                          "kmeans_ari": diag[(enc, key)][0], "pca_capture": diag[(enc, key)][1], "source_file": rel})
    return cells, ucells, diag, extra


#: The manuscript's single-page version of the ten-encoder table, in the layout typeset there (Overleaf,
#: tableB2_ten_encoder_robustness_singlepage.tex): a non-floating block with \captionof, each panel a stack of
#: tabulars with one row per encoder in family groups; Panel A with Mean and Range rows, Panel B with a Mean row.
#: Here the three panels have the same five settings in the same order: the original corpora without Bills-Tweets
#: (dropped at the user's request), and Fedlex, with one version each of SwiLTra and SCOTUS (`SP_VARIANTS`),
#: `SP_PER_TABULAR` triple columns per tabular. Five triple columns need two tabulars per panel, and the block (826 pt
#: at \footnotesize) no longer fits the 703 pt text block, so it is cut into two pages at a fixed point: Panels A-B
#: (607 pt), then Panel C and the caption (219 pt).
SP_VARIANTS = {"SwiLTra": [("swiltra", "SwiLTra DE--FR"), ("swiltra_de_fr_it", "SwiLTra DE--FR--IT")],
               "SCOTUS": [("scotus_oyez_lexis", "SCOTUS O--L"), ("scotus_oyez_lexis_wiki", "SCOTUS O--L--W")]}
#: The version shown is the one in which the three label-free erasers lower linear AUC most relative to Raw (mean
#: over the ten encoders and the three erasers), among the versions run with all ten encoders. The other pairings
#: (SwiLTra DE-IT and FR-IT, SCOTUS O-W and W-L) were run with one or two encoders only.
SP_PER_TABULAR, SP_C_PER_TABULAR = 3, 5
#: Utility is checked, not used, for the choice: the version shown must not lose more utility than this to the other.
SP_UTILITY_TIE = 0.005
#: Wording that names the reference: Panel A and B titles and the caption's second sentence.
SP_TEXT = {"metadata_leace": ("Linear AUC above Metadata-LEACE", "Semantic utility relative to Metadata-LEACE",
                              "the linear-AUC gap to Metadata-LEACE", "$0$ matches the supervised reference"),
           "raw": ("Linear AUC relative to Raw", "Semantic utility relative to Raw",
                   "the change in linear AUC from Raw", "$0$ means no change from the raw embeddings")}


def _original_diag(order, corpora) -> dict:
    """k-means ARI with the classes and PCA capture (leading C - 1 components) of the original study's settings, from
    its geometry runs, averaged over seeds 0-2 and folds."""
    diag = {}
    for enc in order:
        for key, _ in corpora:
            rel = f"{C.MAIN}/alternatives/geometry_{key}_{enc}.json"
            geo = C.read(rel)
            check(geo["seeds"] == [0, 1, 2] and geo["corpus"] == key and geo["encoder"] == enc, "enc_geometry",
                  f"{rel}: the geometry run of {key}/{enc}, seeds 0-2")
            folds = [f for q in geo["per_seed"] for f in q["folds"]]
            diag[(enc, key)] = (float(np.mean([f["ari"]["auto"] for f in folds])),
                                float(np.mean([f["capture"][str(int(geo["n_classes"]) - 1)] for f in folds])))
    return diag


def _sp_cells(ref: str) -> tuple:
    """``(b2, order, cells, ucells, diag)``: per encoder, setting and eraser, the per-seed differences to `ref` in linear
    AUC and in the setting's utility metric, for every setting run with the ten encoders, and the k-means diagnostics.
    The original settings' cells come from the first draft's `_collect` with its reference set to `ref` for the call."""
    import importlib
    b2 = importlib.import_module("paper_assets.exhibits.tableB2_ten_encoder_robustness")
    saved = b2.REF
    b2.REF = ref
    try:
        order, _, cells, _, _, ucells, _ = b2._collect()
    finally:
        b2.REF = saved
    later, ulater, diag, _ = _later_cells(order, b2.GAPS, ref)
    diag.update(_original_diag(order, b2.CORPORA))
    return b2, order, {**cells, **later}, {**ucells, **ulater}, diag


def _sp_choose(b2, order, cells, ucells) -> tuple:
    """``(columns, record)``: the five settings with the version of SwiLTra and SCOTUS chosen by the rule above (from
    differences to Raw), and per candidate version its mean change from Raw in linear AUC and in utility."""
    record, pick = [], {}
    for corpus, versions in SP_VARIANTS.items():
        score = {}
        for key, lab in versions:
            check(all(cells[(e, key, m)].ok and ucells[(e, key, m)].ok for e in order for m in b2.GAPS),
                  "sp_variant_complete", f"{key}: run with all ten encoders, so it can be a column")
            score[key] = (float(np.mean([cells[(e, key, m)].mean for e in order for m in b2.GAPS])),
                          float(np.mean([ucells[(e, key, m)].mean for e in order for m in b2.GAPS])))
            record.append({"corpus": corpus, "setting": key, "label": lab, "mean_linear_auc_minus_raw": score[key][0],
                           "mean_utility_minus_raw": score[key][1]})
        best = min(score, key=lambda k: score[k][0])
        pick[corpus] = next(v for v in versions if v[0] == best)
        for r in record:
            if r["corpus"] == corpus:
                r["shown"] = r["setting"] == best
                r["utility_agrees"] = score[best][1] >= max(u for _, u in score.values()) - SP_UTILITY_TIE
    cols = [("semeval", "SemEval"), pick["SwiLTra"], pick["SCOTUS"], ("cap", "CAP"),
            ("fedlex_domestic", "Fedlex")]
    return cols, record


def _singlepage(ref: str, cols, data, note: str = "") -> str:
    """The manuscript's non-floating ten-encoder table with every difference taken to `ref`, on the settings `cols` in
    all three panels, with a page break before Panel C; `note` is appended to the caption. A setting without a utility
    measure (Bills-Tweets) shows ---."""
    b2, order, cells, ucells, diag = data

    def triple(svs, signed: bool) -> str:
        if not any(v is not None and v.ok for v in svs):
            return "---"
        return r"\,/\,".join(latex.num(v.mean, 3, signed=signed) for v in svs)

    def tabular(part, row, summary) -> list:
        out = [r"\begin{tabular}{l" + "c" * len(part) + "}", r"\toprule",
               "Encoder & " + " & ".join(rf"\multicolumn{{1}}{{c}}{{{lab}}}" for _, lab in part) + r" \\", r"\midrule"]
        for enc in order:
            if enc in b2.FAMILY_BREAK_BEFORE:
                out.append(r"\addlinespace[1pt]")
            out.append(b2.SHORT[enc] + " & " + " & ".join(row(enc, k) for k, _ in part) + r" \\")
        if summary:
            out.append(r"\midrule")
            out += [label + " & " + " & ".join(f(k) for k, _ in part) + r" \\" for label, f in summary]
        return out + [r"\bottomrule", r"\end{tabular}"]

    def summary(store, signed: bool, ranges: bool) -> list:
        def over(k, f):
            if (order[0], k, b2.GAPS[0]) not in store:
                return "---"
            return r"\,/\,".join(f([store[(e, k, m)].mean for e in order]) for m in b2.GAPS)
        rows = [("Mean", lambda k: over(k, lambda v: latex.num(float(np.mean(v)), 3, signed=signed)))]
        if ranges:
            rows.append((r"Range (max$-$min)", lambda k: over(k, lambda v: latex.num(float(np.max(v) - np.min(v)), 3))))
        return rows

    def stack(per, row, summ) -> list:
        out = []
        for i in range(0, len(cols), per):
            out += (["", r"\vspace{1pt}"] if i else []) + tabular(cols[i:i + per], row, summ)
        return out

    a_title, b_title, cap_gap, cap_zero = SP_TEXT[ref]
    a_rows = lambda enc, k: triple([cells.get((enc, k, m)) for m in b2.GAPS], False)          # noqa: E731
    b_rows = lambda enc, k: triple([ucells.get((enc, k, m)) for m in b2.GAPS], True)          # noqa: E731
    c_rows = lambda enc, k: latex.num(diag[(enc, k)][0], 2) + r"\,/\," + latex.num(diag[(enc, k)][1], 2)  # noqa: E731
    lines = [r"% Two-page, non-floating version of Table~\ref{tab:ten-encoder-robustness}",
             r"% (Panels A--B on the first page, Panel C and the caption on the second), used to enforce exact ordering + pagination.",
             r"\begingroup", r"\centering", r"\footnotesize",
             r"% Make the table occupy more of the page (while Panels A--B still fit on one page)",
             r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{0.90}", "", "",
             rf"\par{{\textbf{{Panel A.}} {a_title} (PCA-1/$k$-means LEACE/SAE-LEACE; lower is better)}}\par\vspace{{1pt}}"]
    lines += stack(SP_PER_TABULAR, a_rows, summary(cells, False, True))
    lines += ["", r"\vspace{1pt}",
              rf"\par{{\textbf{{Panel B.}} {b_title} (PCA-1/$k$-means LEACE/SAE-LEACE; R@10, for CAP topic purity; "
              r"higher is better)}\par\vspace{1pt}"]
    lines += stack(SP_PER_TABULAR, b_rows, summary(ucells, True, False))
    lines += ["", r"\newpage", r"\par{\textbf{Panel C.} $k$-means ARI\,/\,PCA capture (post hoc)}\par\vspace{1pt}"]
    lines += stack(SP_C_PER_TABULAR, c_rows, [])
    lines += ["", r"\vspace{2pt}",
              r"\captionof{table}{\textbf{Robustness across embedding models.} All erasers are refit in each model's "
              rf"embedding space (not frozen transfer). Each triple reports {cap_gap} for PCA-1/$k$-means LEACE "
              rf"(silhouette-selected $K$)/SAE-LEACE ($N{{=}}32$); {cap_zero}. Values average matched differences over "
              r"seeds 0--2; means and ranges summarize available models." + (" " + note if note else "") + "}",
              r"\label{tab:ten-encoder-robustness}", "", r"\endgroup", ""]
    return "\n".join(lines)


def table_encoders_vs_raw(ex, name: str = "tableB2_ten_encoder_robustness_singlepage") -> None:
    """The manuscript's single-page ten-encoder table with leakage and utility relative to Raw instead of
    Metadata-LEACE, the same five settings in every panel. Same label as the manuscript's table."""
    data = _sp_cells("raw")
    b2, order, cells, ucells, diag = data
    cols, record = _sp_choose(b2, order, cells, ucells)
    shown = {r["corpus"]: r for r in record if r["shown"]}
    other = {r["corpus"]: r for r in record if not r["shown"]}
    for r in record:
        if r["shown"]:
            check(r["utility_agrees"], "sp_variant_utility", f"{r['setting']}: the version shown for {r['corpus']} gains "
                  f"at least as much utility relative to Raw, up to {SP_UTILITY_TIE} ({r['mean_utility_minus_raw']:+.4f} vs "
                  f"{other[r['corpus']]['mean_utility_minus_raw']:+.4f})")
    full = {"swiltra": "DE--FR", "swiltra_de_fr_it": "DE--FR--IT", "scotus_oyez_lexis": "Oyez--Lexis",
            "scotus_oyez_lexis_wiki": "Oyez--Lexis--Wikipedia"}
    auc = lambda r: latex.num(r["mean_linear_auc_minus_raw"], 3)                               # noqa: E731
    note = (r"SwiLTra and SCOTUS are shown in the version, among those run with all ten models, where the label-free "
            rf"erasers lower linear AUC most: {full[shown['SwiLTra']['setting']]} and {full[shown['SCOTUS']['setting']]} "
            rf"(mean over models and erasers {auc(shown['SwiLTra'])} and {auc(shown['SCOTUS'])}; "
            rf"{full[other['SwiLTra']['setting']]} and {full[other['SCOTUS']['setting']]}: {auc(other['SwiLTra'])} and "
            rf"{auc(other['SCOTUS'])}).")
    tex = _singlepage("raw", cols, data, note)
    # The Metadata-LEACE version of the same columns differs by Metadata-LEACE - Raw, the same for every eraser of a
    # cell (shared splits).
    data_m = _sp_cells("metadata_leace")
    _, _, cells_m, ucells_m, diag_m = data_m
    keys = {k for k, _ in cols}
    shift = {}
    for store, store_m, tag in ((cells, cells_m, "linear_auc"), (ucells, ucells_m, "utility")):
        for (enc, key, m), v in store.items():
            if key in keys and (enc, key, m) in store_m and v.ok and store_m[(enc, key, m)].ok:
                shift.setdefault((tag, enc, key), []).append(v.mean - store_m[(enc, key, m)].mean)
    spread = max(max(s) - min(s) for s in shift.values())
    check(spread < 1e-9, "sp_shift_constant", f"{name}: (method - Raw) - (method - Metadata-LEACE) is the same for "
          f"PCA-1, k-means LEACE and SAE-LEACE in every cell (largest spread {spread:.1e})")
    check(all(diag[(e, k)] == diag_m[(e, k)] for e in order for k in keys), "sp_panel_c",
          f"{name}: Panel C does not depend on the reference")
    check(all((e, k) in diag for e in order for k in keys)
          and all(cells[(e, k, m)].ok for e in order for k in keys for m in b2.GAPS)
          and all(ucells[(e, k, m)].ok for e in order for k in keys for m in b2.GAPS),
          "sp_complete", f"{name}: every cell of Panels A-C is available for the ten encoders "
          f"(settings {[k for k, _ in cols]})")
    path = C.HERE / "tables" / f"{name}.tex"
    path.write_text(tex)
    ex.output(path)
    inc = C.HERE / "tables" / f"{name}.include.tex"
    inc.write_text(f"% Include in the manuscript (requires booktabs and caption, non-floating):\n"
                   f"\\input{{{C.REL}/tables/{name}}}\n"
                   f"% Cross-reference: Table~\\ref{{tab:ten-encoder-robustness}}\n")
    ex.output(inc)
    ex.caption = tex[tex.index(r"\captionof{table}{") + len(r"\captionof{table}{"):tex.index(r"\label{") - 2]
    ex.note("Version choice (differences to Raw, mean over ten encoders and three label-free erasers): "
            + "; ".join(f"{r['setting']} AUC {r['mean_linear_auc_minus_raw']:+.3f}, utility {r['mean_utility_minus_raw']:+.3f}"
                        + (" (shown)" if r["shown"] else "") for r in record))
    rows = []
    for panel_id, store in (("A", cells), ("B", ucells)):
        for (enc, key, m), d in sorted(store.items()):
            if key not in keys or not d.ok:
                continue
            metric = "linear_auc" if panel_id == "A" else ("topic_purity" if key == "cap" else "recall@10")
            rows.append({"panel": panel_id, "encoder": enc, "setting": key, "method": m,
                         "quantity": f"{metric}_minus_raw", "seeds": d.seeds, "per_seed": d.values,
                         "mean": d.mean, "sd": d.sd, "sd_ddof": 1, "source_files": [x["file"] for x in d.sources]})
    rows += [{"panel": "C", "encoder": enc, "setting": key, "method": "diagnostics", "quantity": "kmeans_ari|pca_capture",
              "mean": f"{a}|{c}"} for (enc, key), (a, c) in sorted(diag.items()) if key in keys]
    latex.write_csv(name, rows, ex)
    latex.write_csv(f"{name}_version_choice", record, ex)


# --------------------------------------------------------------------------- #
# Frozen transfer at five seeds, with PCA matched in displacement
# --------------------------------------------------------------------------- #
TRANSFER = f"{C.FOCUSED}/transfer_matched/transfer_matched_five.json"
DONORS = [("e5-small", "cap", "E5-small, donor CAP"), ("e5-small", "scotus_oyez_lexis_wiki", "E5-small, donor SCOTUS (3 sources)"),
          ("e5-small", "swiltra_de_fr_it", "E5-small, donor SwiLTra (3 languages)"), ("mini", "cap", "MiniLM, donor CAP"),
          ("mini", "scotus_oyez_lexis_wiki", "MiniLM, donor SCOTUS (3 sources)")]
MAPS = [("pca_top1", "PC-1"), ("kmeans_leace", r"$k$-means LEACE"), ("metadata_leace", "Metadata-LEACE"),
        ("sae_leace", "SAE-LEACE"), ("pca_matched_var", "PCA, displacement-matched")]


def transfer_rows() -> list:
    """The saved five-seed transfer changes of the five maps, per donor and target: the change, its paired query
    bootstrap interval, and how many of the five fitted maps raise or lower the score. Figure B3 shows the
    changes and intervals (`regenerated.figB3` checks them against its own) and saves these rows as data."""
    d = C.read(TRANSFER)
    check(d["five_seeds"] and all(v == C.SEEDS5 for v in d["seeds"].values()), "transfer_seeds",
          f"{TRANSFER}: every donor's maps are those of seeds 0-4")
    rows = {(r["encoder"], r["donor"], r["transform"], r["target"]): r for r in d["variability"]}
    out = []
    for enc, donor, _ in DONORS:
        for tr, _ in MAPS:
            for target, short in (("legal macro nDCG@10", "legal_macro"), ("STSBenchmark Spearman", "STSBenchmark")):
                r = rows[(enc, donor, tr, target)]
                saved = d["changes"][f"{enc}/{donor}"][tr][short]
                check(abs(saved["delta"] - r["boot_delta"]) < 1e-12 and abs(np.mean(r["per_seed"]) - r["boot_delta"]) < 1e-9,
                      "transfer_delta", f"{enc}/{donor} {tr} {short}: the change is the mean of the five fitted maps' changes")
                out.append({"encoder": enc, "donor": donor, "map": tr, "target": target, "delta": r["boot_delta"],
                            "ci_low": r["boot_ci"][0], "ci_high": r["boot_ci"][1],
                            "interval_excludes_zero": bool(r["boot_ci"][0] > 0 or r["boot_ci"][1] < 0),
                            "per_seed": r["per_seed"], "seed_sd": r["sd"], "seeds_positive": r["n_pos"],
                            "seeds_negative": r["n_neg"], "reading": r["verdict"], "source_file": TRANSFER})
    return out


TABLES = {"table3_mechanism_controls": (table3, "Table 3: mechanism controls on seeds 0-4, with the latent-selector rows"),
          "table_erasure_extent": (table_erasure_extent, "Table 4: SAE-LEACE against displacement-matched PCA"),
          "table_SAE_construction_block": (table_sae_construction, "SAE construction: other TopK dictionaries"),
          "tableB2_ten_encoder_robustness_singlepage": (table_encoders_vs_raw,
                                                        "Table B2: the ten-encoder study relative to Raw")}
