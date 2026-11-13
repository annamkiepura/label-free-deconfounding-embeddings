"""Main Table 2: primary in-distribution leakage (Panel A) and semantic utility (Panel B).

Columns: E5-small on the five primary corpora and on Fedlex DE/FR/IT (after SwiLTra;
its run comes from notebooks/03_fedlex.ipynb), plus a separated
MiniLM column for three-source SCOTUS (the only primary corpus with a MiniLM
paired-utility run in this comparison). Rows: Raw, PC-1, k-means LEACE (silhouette K),
Metadata-LEACE, SAE-LEACE (N = 32). Cells: mean and sample SD across fitting
seeds of the manifest's locked cohort. Each seed's value is the saved seed-level
metric, in which folds (SemEval) are already combined within the seed. Nothing
is pooled across seeds.
"""
from __future__ import annotations

import numpy as np

from ..lib import config, latex
from ..lib.datasets import from_manifest, processed
from ..lib.io import read, run_id, sha256
from ..lib.loaders import Cohort, utility_metric, validate_file
from ..lib.methods import ENCODERS, METHODS, metric_label
from ..lib.provenance import check

NAME, KIND, TITLE = "table2_primary_id", "table", "Primary in-distribution leakage and semantic utility"
LABEL = "tab:primary-id"
COLUMNS = [(k, "e5-small") for k in ("semeval", "swiltra_de_fr_it", "bills_tweets",
                                     "scotus_oyez_lexis_wiki", "cap")] + [("scotus_oyez_lexis_wiki", "mini")]
#: Fedlex DE/FR/IT. Its run comes from its own notebook (manifest cohort `fedlex`), not from the corpus
#: registry, so it is not in COLUMNS, which code/focused_additions.py and the Metadata-LEACE sanity check
#: iterate. Table 2 (and Table B4) show it after SwiLTra, the other language-confounded corpus.
FEDLEX = ("fedlex_domestic", "e5-small")
TABLE_COLUMNS = COLUMNS[:2] + [FEDLEX] + COLUMNS[2:]
ROWS =("raw", "pca_top1", "kmeans_auto", "metadata_leace", "sae_leace")
ROW_LABEL = {"kmeans_auto": r"$k$-means LEACE", "sae_leace": r"SAE-LEACE ($N{=}32$)"}
SAE_N = 32


def _seed_record(coh: Cohort, method: str, seed: int) -> dict:
    role, cond = coh.resolve(method)
    d = read(coh.paths[role])
    p = next(x for x in d["per_seed"] if x["seed"] == seed)
    return {"file": coh.paths[role], "condition": cond, "split": p.get("split"),
            "per_fold_linear_auc": (p.get("per_fold") or {}).get(cond, {}).get("linear_auc")}


def _validate_column(key: str, enc: str, coh: Cohort, cells: dict) -> None:
    tag = f"{key}/{enc} ({coh.name})"
    files = {r: validate_file(p) for r, p in coh.paths.items()
             if r in {coh.resolve(m)[0] for m in ROWS}}
    # Dataset version: every source file's corpus statistics equal the processed definition (for a
    # corpus outside the registry, its cohort manifest, whose fingerprint the run must also carry).
    from_cohort_manifest = "cohort_manifest" in config.setting(key)
    p = from_manifest(key) if from_cohort_manifest else processed(key)
    for r, d in files.items():
        cs = d["corpus_stats"]
        check(cs["n_docs"] == p["n_docs"] and cs["n_pairs"] == p["n_retrieval_pairs"]
              and dict(zip(cs["classes"], cs["class_counts"])) == dict(zip(p["classes"], p["class_counts"])),
              "t2_dataset_version", f"{tag} {coh.paths[r]}: corpus statistics equal the processed "
              f"{config.setting(key)['long']} definition")
        if from_cohort_manifest:
            check(d["dataset"]["manifest"] == p["manifest"] and d["dataset"]["cohort_sha256"] == p["cohort_sha256"]
                  and p["n_unique_ids"] == p["n_tuples"],
                  "t2_cohort_manifest", f"{tag} {coh.paths[r]}: computed on exactly the cohort of {p['manifest']} "
                  f"({p['n_tuples']:,} tuples; fingerprint {p['cohort_sha256'][:12]})")
    # Same seeds for every method of the column.
    seeds = {m: tuple(v.seeds) for m, v in cells.items() if v.ok}
    check(len(set(seeds.values())) == 1, "t2_same_seeds", f"{tag}: every method uses seeds "
          f"{next(iter(seeds.values()))}")
    # Evaluation definition and split policy identical across the column's files (and the audit is
    # the one every column uses).
    ref_audit = read(Cohort.assigned(*COLUMNS[0]).paths["main"])["config"]["audit"]
    for r, d in files.items():
        c = d["config"]
        check(c["audit"] == ref_audit, "t2_same_audit",
              f"{tag} {coh.paths[r]}: audit protocol identical to every other column's")
        check(c["recipe"]["recall_k"] == 10 and c["audit"]["topic_k"] == 21, "t2_eval_def",
              f"{tag} {coh.paths[r]}: R@10 and 21-topic purity definitions")
        check(c["split"] == files["main"]["config"]["split"], "t2_same_split_policy",
              f"{tag} {coh.paths[r]}: split policy {c['split']}")
    # Method configuration.
    main = files["main"]
    check(main["config"]["recipe"]["top_features"] == SAE_N and main["config"]["selection"] == "topn",
          "t2_sae_config", f"{tag}: SAE-LEACE recipe is MI top-{SAE_N}")
    if coh.name == "coverage":
        check(main.get("sae_budget") == SAE_N and all(len(dg["selected_mi"]) == SAE_N
                                                      for s in main["per_seed"] for dg in s["diagnostics"]),
              "t2_sae_selected", f"{tag}: every fit erased exactly {SAE_N} selected latents")
        # The coverage run's k-means arm must be the silhouette-K arm of the group-count run.
        kc = validate_file(coh.paths["kcount"])
        same = all(abs(a["metrics"]["kmeans_leace"][mt] - b["metrics"]["kmeans_auto"][mt]) < 1e-12
                   for a, b in zip(main["per_seed"], kc["per_seed"])
                   for mt in ("linear_auc", utility_metric(key) or "linear_auc"))
        check(same, "t2_kmeans_is_silhouette", f"{tag}: k-means arm equals the silhouette-K arm of "
              f"{coh.paths['kcount']}")
        km_diag = [dg for s in kc["per_seed"] for dg in s["diagnostics"]]
    else:
        check(all(n == SAE_N for s in main["per_seed"] for n in s["n_selected"]), "t2_sae_selected",
              f"{tag}: every fit erased exactly {SAE_N} selected latents")
        km_diag = ([dg for s in files["kmeans"]["per_seed"] for dg in s["diagnostics"]] if "kmeans" in files
                   else [s["kmeans"] for s in main["per_seed"]])     # one-run cohort (Fedlex): k-means per seed
    check(all(dg["k_auto"] == int(max(dg["silhouette"], key=dg["silhouette"].get)) for dg in km_diag),
          "t2_kmeans_silhouette", f"{tag}: each fit's K is the training-silhouette maximiser "
          f"(K chosen: {sorted({dg['k_auto'] for dg in km_diag})})")
    # Fold-within-seed aggregation: each seed's AUC equals the mean of its own folds.
    lin = cells["sae_leace"]
    for s, v in zip(lin.seeds, lin.values):
        folds = _seed_record(coh, "sae_leace", s)["per_fold_linear_auc"]
        if folds and len(folds) > 1:
            check(abs(np.mean(folds) - v) < 1e-12, "t2_fold_within_seed",
                  f"{tag} seed {s}: AUC is the mean of its {len(folds)} fold AUCs, not pooled")


def build(ex) -> None:
    per_seed, summary = [], []
    panels = {"A": {}, "B": {}}
    for key, enc in TABLE_COLUMNS:
        coh = Cohort.assigned(key, enc)
        u = utility_metric(key)
        a = {m: coh.values(m, "linear_auc") for m in ROWS}
        b = {m: (coh.values(m, u) if u else None) for m in ROWS}
        _validate_column(key, enc, coh, a)
        panels["A"][(key, enc)], panels["B"][(key, enc)] = a, b
        for panel, cells, metric in (("A", a, "linear_auc"), ("B", b, u or "none")):
            for m in ROWS:
                sv = cells[m]
                if sv is None:
                    summary.append({"panel": panel, "setting": key, "encoder": enc, "method": m,
                                    "metric": "none", "status": "not_applicable", "reason":
                                    "no semantic-utility metric (unpaired, no topic labels)"})
                    continue
                summary.append(latex.sv_row(sv, panel=panel))
                for s, v in zip(sv.seeds, sv.values):
                    rec = _seed_record(coh, m, s)
                    per_seed.append({"panel": panel, "setting": key, "encoder": enc, "cohort": coh.name,
                                     "method": m, "metric": metric, "seed": s, "value": v,
                                     "n_folds": (rec["split"] or {}).get("n_folds"),
                                     "fold_values": rec["per_fold_linear_auc"] if metric == "linear_auc" else "",
                                     "n_test_docs": (rec["split"] or {}).get("n_test_docs"),
                                     "n_test_pairs": (rec["split"] or {}).get("n_test_pairs"),
                                     "source_file": rec["file"], "run_id": run_id(rec["file"], sha256(rec["file"])),
                                     "saved_condition": rec["condition"],
                                     "path": f"per_seed[seed={s}].metrics.{rec['condition']}.{metric}"})
    # --- LaTeX: two stacked panels, methods as rows, corpora as columns ------------- #
    ncol = len(TABLE_COLUMNS)
    head1 = ([""] + [config.setting(k)["label"] for k, e in TABLE_COLUMNS[:-1]]
             + [config.setting(TABLE_COLUMNS[-1][0])["label"]])
    head2 = [""] + [f"C={len(config.setting(k)['classes'])}" for k, _ in TABLE_COLUMNS]
    head0 = ["", rf"\multicolumn{{{ncol - 1}}}{{c}}{{E5-small}}", "MiniLM"]
    seeds_of = {c: panels["A"][c]["raw"].seeds for c in TABLE_COLUMNS}
    body = []
    for panel, title in (("A", r"Panel A: linear-probe nuisance ROC-AUC $\downarrow$ "
                                r"(one-vs-rest macro average for $C>2$)"),
                         ("B", r"Panel B: semantic utility $\uparrow$ (R@10; CAP: topic purity)")):
        body.append(rf"\multicolumn{{{ncol + 1}}}{{l}}{{\textit{{{title}}}}} \\")
        for m in ROWS:
            line = [ROW_LABEL.get(m, METHODS[m].label)]
            for c in TABLE_COLUMNS:
                sv = panels[panel][c][m]
                line.append(r"\multicolumn{1}{c}{---}" if sv is None else latex.cell(sv))
            body.append(line)
        if panel == "A":
            body.append(r"\midrule")
    colspec = "l" + "r" * (ncol - 1) + "@{\\hskip 14pt}r"
    lines = latex.tabular(colspec, [head0, head1, head2], body,
                          cmidrules=[rf"\cmidrule(lr){{2-{ncol}}}\cmidrule(l{{6pt}}){{{ncol + 1}-{ncol + 1}}}"])
    seed_txt = "; ".join(
        f"{config.setting(k)['label']}{' MiniLM' if e == 'mini' else ''} "
        f"{len(seeds_of[(k, e)])} seeds ({seeds_of[(k, e)][0]}--{seeds_of[(k, e)][-1]})"
        for k, e in TABLE_COLUMNS)
    fx = from_manifest(FEDLEX[0])
    fedlex_txt = (r"Fedlex is a seeded sample of " + f"{fx['n_tuples']:,}".replace(",", "{,}")
                  + r" German/French/Italian article triples of Swiss domestic federal law"
                  + (r" (SwiLTra's size)" if fx["n_docs"] == processed("swiltra_de_fr_it")["n_docs"] else "")
                  + r", with outer splits by whole law.")
    caption = (r"Primary in-distribution leakage and semantic utility. Cells are the mean "
               r"$\pm$ sample SD ($\mathrm{ddof}{=}1$) across fitting seeds. Each seed's score is "
               r"computed within that seed; SemEval's five outer folds are combined within each seed "
               r"(fold AUCs averaged, held-out retrieval hits pooled), and nothing is pooled across "
               r"seeds. Seeds: " + seed_txt + r". SCOTUS with E5-small uses the final three-seed "
               r"coverage cohort. " + fedlex_txt + r" MiniLM results are shown only for three-source "
               r"SCOTUS, not for the other corpora. AUCs are unflipped. $k$-means LEACE uses the training-silhouette "
               r"$K$; only Metadata-LEACE uses nuisance labels. Bills--Tweets has no semantic-utility "
               r"measure (---). Equal rounded values are not evidence of equivalence.")
    latex.write_table(NAME, lines, caption, LABEL, ex, wide=True, size=r"\footnotesize", tabcolsep="4pt")
    latex.write_csv(f"{NAME}_per_seed", per_seed, ex)
    latex.write_csv(f"{NAME}_summary", summary, ex)
