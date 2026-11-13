"""Appendix Table B1: extended in-distribution results (one page).

Main block: the eight E5-small composition settings (SwiLTra DE-FR, DE-IT, FR-IT,
DE-FR-IT; SCOTUS Oyez-Lexis, Oyez-Wikipedia, Lexis-Wikipedia, all three sources) and
Fedlex DE-FR-IT x the five primary methods x linear AUC, MLP AUC, R@1, R@10.
Supplementary block: SemEval graded-similarity Spearman and CAP topic ARI for the same
methods.

Every cell is `Cohort.values` on the manifest's locked cohort: SwiLTra run_id seeds
0-4 (all four settings), Fedlex its notebook's run (seeds 0-4), SCOTUS/E5-small the
three-seed coverage cohort (seeds 0-2), SemEval and CAP run_id seeds 0-2. The
coverage files record all four metrics, so no
SCOTUS metric is borrowed from another run; a metric a file lacks would be reported
missing, never estimated. Each setting is validated with Table 2's column checks
(dataset version, seeds, split policy, audit and retrieval definitions, SAE and
k-means configuration). The metric definitions are written as a paragraph for the
appendix text (tables/<NAME>_metrics.tex), which the caption references, so that the
table fits one page. `finalize` compares every overlapping cell with the main
exhibits' CSVs and checks the compiled preview fits one TACL page.
"""
from __future__ import annotations

import csv
import subprocess

from ..lib import config, latex
from ..lib.io import run_id, sha256
from ..lib.loaders import Cohort
from ..lib.provenance import check, issue
from .table2_primary_id import ROW_LABEL, ROWS, _seed_record, _validate_column

NAME, KIND, TITLE = "tableB1_extended_id", "table", "Extended in-distribution results"
LABEL = "tab:extended-id"
#: The metric definitions live in a paragraph for the appendix text (tables/<NAME>_metrics.tex), which
#: the caption references; previews show the label as declared here.
METRICS_LABEL = "app:extended-id-metrics"
TEXT_LABELS = {METRICS_LABEL: "B"}
ENCODER = "e5-small"
BLOCKS = [("SwiLTra", "language", [("swiltra", "DE--FR"), ("swiltra_de_it", "DE--IT"),
                                   ("swiltra_fr_it", "FR--IT"), ("swiltra_de_fr_it", "DE--FR--IT")]),
          ("SCOTUS", "source", [("scotus_oyez_lexis", "Oyez--Lexis"), ("scotus_oyez_wiki", "Oyez--Wikipedia"),
                                ("scotus_wiki_lexis", "Lexis--Wikipedia"),
                                ("scotus_oyez_lexis_wiki", "All three sources")])]
#: Fedlex DE/FR/IT, shown between SwiLTra and SCOTUS. Kept out of BLOCKS, which code/focused_additions.py
#: iterates over registry settings (as Table 2 keeps Fedlex out of its COLUMNS).
FEDLEX_BLOCK = ("Fedlex", "language", [("fedlex_domestic", "DE--FR--IT")])
TABLE_BLOCKS = BLOCKS[:1] + [FEDLEX_BLOCK] + BLOCKS[1:]
#: The block heading's cohort note.
COHORT_NOTE = {"coverage": "coverage cohort", "fedlex": "split by whole law"}
METRICS =[("linear_auc", r"Linear AUC $\downarrow$"), ("mlp_auc", r"MLP AUC $\downarrow$"),
           ("recall@1", r"R@1 $\uparrow$"), ("recall@10", r"R@10 $\uparrow$")]
SUPP = [("semeval", "gold_sim_spearman", r"Graded sim.\ $\rho$ $\uparrow$"),
        ("cap", "topic_ari", r"Topic ARI $\uparrow$")]
HEAD = {"raw": ("Raw", ""), "pca_top1": ("PC-1", ""), "kmeans_auto": (r"$k$-means", "LEACE"),
        "metadata_leace": ("Metadata-", "LEACE"), "sae_leace": ("SAE-LEACE", r"($N{=}32$)")}
# Main-exhibit CSVs whose cells overlap this table (compared in finalize).
MAIN_CSVS = {"table2_primary_id": "table2_primary_id_summary.csv",
             "table3_mechanism_controls": "table3_mechanism_controls_summary.csv",
             "fig1_rank_composition": "fig1_rank_composition_summary.csv",
             "fig3_pair_specific_retrieval": "fig3_pair_specific_retrieval_summary.csv",
             "tab_composition": "tab_composition.csv"}
# Block separation: a heavy rule with space on both sides between SwiLTra, SCOTUS and
# the supplementary rows; each block heading is underlined across the table.
BLOCK_BREAK = [r"\addlinespace[4pt]", r"\midrule[\heavyrulewidth]", r"\addlinespace[4pt]"]
HEAD_RULE = r"\cmidrule(lr){1-7}"
_STATE: dict = {}


def _seeds_txt(seeds) -> str:
    return f"{len(seeds)} seeds ({seeds[0]}--{seeds[-1]})"


def _rows(key: str, cells: dict, metrics, coh: Cohort, per_seed: list, summary: list, block: str) -> None:
    for metric, _ in metrics:
        for m in ROWS:
            sv = cells[(m, metric)]
            summary.append(latex.sv_row(sv, block=block))
            if not sv.ok:
                continue
            for s, v in zip(sv.seeds, sv.values):
                rec = _seed_record(coh, m, s)
                sp = rec["split"] or {}
                per_seed.append({"block": block, "setting": key, "encoder": ENCODER, "cohort": coh.name,
                                 "method": m, "metric": metric, "seed": s, "value": v,
                                 "split_scheme": sp.get("scheme"), "n_folds": sp.get("n_folds"),
                                 "n_test_docs": sp.get("n_test_docs"), "n_test_pairs": sp.get("n_test_pairs"),
                                 "source_file": rec["file"], "run_id": run_id(rec["file"], sha256(rec["file"])),
                                 "saved_condition": rec["condition"],
                                 "path": f"per_seed[seed={s}].metrics.{rec['condition']}.{metric}"})


def build(ex) -> None:
    per_seed, summary, body, seeds_of = [], [], [], {}
    ncols = 2 + len(ROWS)
    for bi, (corpus, confounder, settings) in enumerate(TABLE_BLOCKS):
        block_rows, block_seeds = [], set()
        for si, (key, name) in enumerate(settings):
            coh = Cohort.assigned(key, ENCODER)
            cells = {(m, mt): coh.values(m, mt) for mt, _ in METRICS for m in ROWS}
            _validate_column(key, ENCODER, coh, {m: cells[(m, "linear_auc")] for m in ROWS})
            seeds = {tuple(v.seeds) for v in cells.values() if v.ok}
            check(len(seeds) == 1, "b1_same_seeds_all_metrics",
                  f"{key}: all 5 methods x 4 metrics use seeds {sorted(seeds)}")
            files = {src["file"] for v in cells.values() if v.ok for src in v.sources}
            check(all(f.startswith("alternatives/") for f in files) if coh.name == "coverage" else True,
                  "b1_coverage_only", f"{key}: every SCOTUS cell comes from the coverage run ({sorted(files)})")
            folds = {v for m in ROWS for v in coh.values(m, "audit_folds", register=False).values}
            check(folds == {5}, "b1_audit_folds", f"{key}: every probe AUC used 5 audit folds (metric definitions)",
                  audit_folds=sorted(folds))
            seeds_of[key] = next(iter(seeds))
            block_seeds |= seeds
            _rows(key, cells, METRICS, coh, per_seed, summary, corpus)
            if si:
                block_rows.append(r"\addlinespace[3pt]")
            C = len(config.setting(key)["classes"])
            for mi, (metric, mlab) in enumerate(METRICS):
                first = (rf"\textbf{{{name}}}" if mi == 0 else
                         rf"{{\footnotesize $C{{=}}{C}$}}" if mi == 1 else "")
                block_rows.append([first, mlab] + [latex.cell(cells[(m, metric)]) for m in ROWS])
        check(len(block_seeds) == 1, "b1_block_cohort", f"{corpus}: one seed set across its {len(settings)} "
              f"setting{'s' if len(settings) > 1 else ''} ({sorted(block_seeds)})")
        coh_name = Cohort.assigned(settings[0][0], ENCODER).name
        if bi:
            body += BLOCK_BREAK
        body += [rf"\multicolumn{{{ncols}}}{{l}}{{\textbf{{{corpus}}} \quad {{\footnotesize confounder: "
                    rf"{confounder}; {_seeds_txt(next(iter(block_seeds)))}, "
                    rf"{COHORT_NOTE.get(coh_name, 'primary cohort')}}}}} \\", HEAD_RULE]
        body += block_rows
    # Supplementary panel: SemEval graded similarity and CAP topic ARI.
    supp_rows, supp_seeds = [], set()
    for key, metric, mlab in SUPP:
        coh = Cohort.assigned(key, ENCODER)
        coh.compatibility()
        cells = {(m, metric): coh.values(m, metric) for m in ROWS}
        seeds = {tuple(v.seeds) for v in cells.values() if v.ok}
        check(len(seeds) == 1, "b1_same_seeds_supp", f"{key}: all methods use seeds {sorted(seeds)}")
        seeds_of[key] = next(iter(seeds))
        _rows(key, cells, [(metric, mlab)], coh, per_seed, summary, "Supplementary")
        lab = config.setting(key)["label"]
        supp_seeds |= seeds
        supp_rows.append([rf"\textbf{{{lab}}} {{\footnotesize $C{{=}}{len(config.setting(key)['classes'])}$}}", mlab]
                         + [latex.cell(cells[(m, metric)]) for m in ROWS])
    supp_txt = (_seeds_txt(next(iter(supp_seeds))) if len(supp_seeds) == 1
                else "; ".join(f"{k} {_seeds_txt(seeds_of[k])}" for k, *_ in SUPP))
    body += BLOCK_BREAK + [rf"\multicolumn{{{ncols}}}{{l}}{{\textbf{{Supplementary: other primary corpora}} "
                          rf"\quad {{\footnotesize {supp_txt}, primary cohort}}}} \\", HEAD_RULE] + supp_rows
    head = [["Setting", "Metric"] + [HEAD[m][0] for m in ROWS],
            ["", ""] + [HEAD[m][1] for m in ROWS]]
    lines = latex.tabular("ll" + "c" * len(ROWS), head, body)
    sw, sc = seeds_of["swiltra"], seeds_of["scotus_oyez_lexis"]
    caption = (
        r"Extended in-distribution results for the eight E5-small composition settings and Fedlex. Cells are "
        r"the mean $\pm$ sample SD ($\mathrm{ddof}{=}1$) across fitting seeds; each seed is scored on its own "
        r"held-out split and nothing is pooled across seeds. Cohorts: SwiLTra, the primary cohort, "
        + _seeds_txt(sw) + r", in all four settings; Fedlex, as in Table~\ref{tab:primary-id} (outer splits by "
        r"whole law), " + _seeds_txt(seeds_of["fedlex_domestic"]) + r"; SCOTUS, the three-seed coverage cohort "
        r"of Table~\ref{tab:primary-id}, " + _seeds_txt(sc) + r". Within a setting all methods and metrics "
        r"come from the same saved run and splits. Metrics are defined in Appendix~\ref{" + METRICS_LABEL + r"}. "
        r"$k$-means LEACE uses the training-silhouette $K$; only Metadata-LEACE uses confounder labels.")
    # The metric definitions, moved out of the caption so that the table fits one page with Fedlex.
    metrics_par = (
        r"\paragraph{Metrics of Table~\ref{" + LABEL + r"}.}\label{" + METRICS_LABEL + "}\n"
        r"Linear AUC is an L2 logistic-regression probe for the confounder on the held-out documents (5 audit "
        r"folds, $C$ chosen on an inner split; one-vs-rest macro ROC-AUC when $C>2$; unflipped). MLP AUC is a "
        r"standardised MLP (256, 128) probe on the same folds. For R@1 and R@10, each held-out pair queries the "
        r"whole transformed corpus in both directions, excluding the query and, in three-class settings, its "
        r"other linked counterpart; recall is averaged over all held-out pairs. In the supplementary rows, "
        r"SemEval graded similarity is Spearman's $\rho$ between held-out pair cosines and the negated Overall "
        r"rating (higher = more similar; the ratings are almost all 1 or 2), one value per seed over the "
        r"held-out pairs of all five outer folds; CAP topic ARI is the adjusted Rand index between a 21-cluster "
        r"$k$-means of the held-out documents and the gold topics; both use " + _seeds_txt(seeds_of["semeval"]) + ".")
    latex.write_table(NAME, lines, caption, LABEL, ex, wide=True, size=r"\small", tabcolsep="5pt")
    p = config.OUT_DIRS["tables"] / f"{NAME}_metrics.tex"
    p.write_text(f"% Generated by paper_assets with Table B1; do not edit by hand. Rebuild:\n"
                 f"%   python -m paper_assets build {NAME}\n"
                 f"% Paste into the appendix text: Table B1's caption points here (\\ref{{{METRICS_LABEL}}}).\n"
                 + metrics_par + "\n")
    ex.output(p)
    latex.write_csv(f"{NAME}_per_seed", per_seed, ex)
    latex.write_csv(f"{NAME}_summary", summary, ex)
    _STATE.update(summary=summary, seeds_of=seeds_of, n_rows=sum(1 for r in body if isinstance(r, list)),
                  lines=lines, caption=caption)


# --------------------------------------------------------------------------- #
def _vals(s) -> list[float]:
    return [float(x) for x in str(s).split("|")] if str(s) not in ("", "nan") else []


def _agreement() -> dict:
    """Every cell of this table that a main exhibit also reports, compared value by value."""
    mine = {(r["setting"], r["encoder"], r["method"], r["metric"]): r for r in _STATE["summary"]
            if r["status"] == "ok"}
    out = {"compared": 0, "agree": 0, "other_cohort": [], "mismatch": [], "absent_csv": []}
    for exhibit, fn in MAIN_CSVS.items():
        p = config.OUT_DIRS["data"] / fn
        if not p.exists():
            out["absent_csv"].append(exhibit)
            continue
        for r in csv.DictReader(p.open()):
            k = (r.get("setting"), r.get("encoder"), r.get("method"), r.get("metric"))
            if k not in mine or r.get("status") != "ok" or r.get("kind", "value") != "value":
                continue
            me = mine[k]
            theirs_seeds = [int(x) for x in r["seeds"].split("|")]
            if r["cohort"] != me["cohort"] or theirs_seeds != list(me["seeds"]):
                out["other_cohort"].append((exhibit, k, r["cohort"], theirs_seeds))
                continue
            out["compared"] += 1
            a, b = _vals(r["values"]), me["values"]
            if len(a) == len(b) and all(abs(x - y) < 1e-12 for x, y in zip(a, b)):
                out["agree"] += 1
            else:
                out["mismatch"].append((exhibit, k, a, b))
    return out


def _layout(pdf) -> dict:
    """Page count of the compiled preview (Ghostscript) and the float's dimensions
    measured by TeX on the same TACL page: tabular box at \\small and the caption
    set at \\textwidth in the float's \\small, as the class sets it ("Table N: ..."), plus the caption skip."""
    pages = subprocess.run(["gs", "-q", "-dNODISPLAY", "-dNOSAFER", "-c",
                            f"({pdf}) (r) file runpdfbegin pdfpagecount = quit"],
                           capture_output=True, text=True).stdout.strip()
    tab = "\n".join(_STATE["lines"])
    body = (r"\newsavebox\tabbox\sbox\tabbox{\small\setlength{\tabcolsep}{5pt}" + tab + "}\n"
            r"\setbox0\vbox{\small\hsize\textwidth\noindent Table B1: " + _STATE["caption"].replace(
                r"Table~\ref{tab:primary-id}", "Table 2").replace(r"\ref{" + METRICS_LABEL + "}", "B.1") + "}\n"
            r"\typeout{MEASURE tabwd=\the\wd\tabbox}\typeout{MEASURE tabht=\the\dimexpr\ht\tabbox+\dp\tabbox\relax}"
            r"\typeout{MEASURE capht=\the\dimexpr\ht0+\dp0\relax}\typeout{MEASURE capskip=\the\abovecaptionskip}"
            r"\typeout{MEASURE textheight=\the\textheight}\typeout{MEASURE textwidth=\the\textwidth}"
            r"\typeout{MEASURE floatsep=\the\textfloatsep}\mbox{}")
    m = {k: float(v.rstrip("pt")) for k, v in latex.measure(body).items()}
    if not m:
        return {"pages": int(pages) if pages.isdigit() else None}
    height = m["tabht"] + m["capskip"] + m["capht"]
    return {"pages": int(pages) if pages.isdigit() else None, **m, "float_ht": height,
            "fits_height": height <= m["textheight"], "fits_width": m["tabwd"] <= m["textwidth"]}


def finalize(ex) -> None:
    agr = _agreement()
    check(not agr["mismatch"], "b1_agrees_with_main_exhibits",
          f"{agr['agree']}/{agr['compared']} overlapping cells (same setting, method, metric, cohort and "
          f"seeds) equal the main exhibits' values seed by seed", mismatches=agr["mismatch"])
    if agr["absent_csv"]:
        issue("warning", "b1_main_csv_absent", f"not compared (CSV not built): {agr['absent_csv']}")
    pdf = config.OUT_DIRS["reports"] / "previews" / f"{NAME}.pdf"
    lay = _layout(pdf) if pdf.exists() else None
    if lay:
        check(lay["pages"] == 1, "b1_one_page", f"compiled preview has {lay['pages']} page(s)")
        if "float_ht" in lay:
            check(lay["fits_height"], "b1_fits_height", f"table + caption {lay['float_ht']:.0f}pt of the "
                  f"{lay['textheight']:.0f}pt text height")
            check(lay["fits_width"], "b1_fits_width", f"tabular {lay['tabwd']:.0f}pt wide, text width "
                  f"{lay['textwidth']:.0f}pt")
        else:
            issue("warning", "b1_no_measure", "TeX measurement failed; height not verified beyond page count")
    else:
        issue("warning", "b1_no_preview", "preview not compiled; one-page fit not verified")
    L = [f"# Table B1 ({NAME}): provenance and layout validation", "",
         f"Build: `python -m paper_assets build {NAME}`. Reporting only: every value is a saved per-seed "
         "metric read through `lib/loaders.Cohort` (no refits, no estimates).", "", "## Cohorts", ""]
    for key, s in _STATE["seeds_of"].items():
        coh = Cohort.assigned(key, ENCODER)
        L.append(f"- {key}: cohort `{coh.name}`, seeds {list(s)}; files "
                 + ", ".join(f"`{p}`" for r, p in coh.paths.items() if coh.available(r) and r != "mech"))
    L += ["", "SCOTUS/E5-small: the coverage files (`alternatives/alternatives_*`) record linear AUC, MLP AUC, "
          "R@1 and R@10 for all five methods, so no metric was taken from another run. The k-means arm is "
          "checked equal to the silhouette-K arm of `alternatives/kmeans_count_*`.", "",
          "## Missing or not recorded", ""]
    miss = [r for r in _STATE["summary"] if r["status"] != "ok"]
    L += [f"- {r['setting']} {r['method']} {r['metric']}: {r['status']} ({r['reason']})" for r in miss] or ["None."]
    L += ["", "## Agreement with main exhibits", "",
          f"- Compared {agr['compared']} overlapping cells (same setting, encoder, method, metric, cohort, seeds): "
          f"{agr['agree']} identical seed by seed (tolerance 1e-12), {len(agr['mismatch'])} mismatches."]
    by_ex = {}
    for e, k, coh_, s in agr["other_cohort"]:
        by_ex.setdefault(e, set()).add((k[0], coh_, tuple(s)))
    for e, items in sorted(by_ex.items()):
        L.append(f"- Not compared, different cohort by design ({e}): "
                 + "; ".join(f"{k} uses `{c}` seeds {list(s)}" for k, c, s in sorted(items)))
    L += [f"- Mismatch: {e} {k}: theirs {a}, ours {b}" for e, k, a, b in agr["mismatch"]]
    L += ["", "## Layout", ""]
    if lay:
        L += [f"- Preview: `reports/previews/{NAME}.pdf`, {lay['pages']} page(s); body \\small (the manuscript's "
              f"table size, no further reduction), SD in \\scriptsize; {_STATE['n_rows']} data rows; table* at "
              f"\\textwidth."]
        if "float_ht" in lay:
            L += [f"- Tabular: {lay['tabwd']:.1f}pt wide of {lay['textwidth']:.1f}pt text width; "
                  f"{lay['tabht']:.1f}pt tall.",
                  f"- Caption: {lay['capht']:.1f}pt (+ {lay['capskip']:.1f}pt skip). Table + caption: "
                  f"{lay['float_ht']:.1f}pt of the {lay['textheight']:.1f}pt text height "
                  f"({100 * lay['float_ht'] / lay['textheight']:.0f}%), leaving {lay['textheight'] - lay['float_ht']:.1f}pt.",
                  "- No overfull boxes and no 'Float too large' warning in the preview log (checked)."]
    L += ["", f"## Checks ({len(ex.checks)} passed)", ""] + [f"- {c['message']}" for c in ex.checks]
    L += ["", "## Issues", ""] + ([f"- **{i['level']}** `{i['code']}`: {i['message']}" for i in ex.issues]
                                 or ["None."])
    p = config.OUT_DIRS["reports"] / f"{NAME}_validation.md"
    p.write_text("\n".join(L) + "\n")
    ex.output(p)
