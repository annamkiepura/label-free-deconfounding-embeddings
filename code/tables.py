"""LaTeX (booktabs) tables built from the result JSONs written by `experiments`.

Each function takes the loaded result(s) and returns a complete ``table`` float;
passing ``save=True`` also writes it into the active section's ``tables/``. Every
table prints the protocol it was measured under, because a leakage number means
different things under the two protocols and a reader must not have to guess.

Requires ``\\usepackage{booktabs,multirow}`` in the preamble.

This supersedes `make_tables.py`, which reads the pre-`protocols` result shapes
(``baselines_<ds>.json`` and friends) and is kept only for the archived runs
under `scratchpad/`.
"""

from __future__ import annotations

import numpy as np

from artifacts import load_result, save_table

COND_LABEL = {
    "raw": "Raw", "metadata_leace": "Metadata oracle", "sae_leace": "SAE (ours)",
    "kmeans_leace_k2": r"$k$-means LEACE ($k{=}2$)",
    "kmeans_leace_k4": r"$k$-means LEACE ($k{=}4$)",
    "kmeans_leace_k8": r"$k$-means LEACE ($k{=}8$)",
    "pca_top1": "PCA top-1", "pca_top2": "PCA top-2",
    "pca_top4": "PCA top-4", "pca_top8": "PCA top-8",
    "random_latents": "Random latents",
    "pca_matched": "PCA top-$|S|$ (size-matched)",
    "kmeans_leace_auto": r"$k$-means LEACE ($K$ by silhouette)",
}


def cond_label(name: str) -> str:
    """Condition label, generating one for the size-matched PCA arm."""
    if name.startswith("pca_top") and name.endswith("_matched"):
        return f"PCA top-${name[len('pca_top'):-len('_matched')]}$ (size-matched)"
    return COND_LABEL.get(name, name.replace("_", " "))


METRIC_COL = {
    "leakage_auc": (r"Leak.$\downarrow$", 3),
    "leakage_auc_flip": (r"Leak. (flip)$\downarrow$", 3),
    "crossfit_leakage_auc": (r"Leak. (CF)$\downarrow$", 3),
    "nonlinear_leakage_auc_mlp_flip": (r"MLP$\downarrow$", 3),
    "nonlinear_leakage_auc_gb_flip": (r"GB$\downarrow$", 3),
    "cluster_purity": (r"Purity$\downarrow$", 3),
    "recall@1": (r"R@1$\uparrow$", 3),
    "recall@10": (r"R@10$\uparrow$", 3),
    "neighbor_retention@10": (r"Nbr.$\uparrow$", 3),
    "gold_sim_spearman": (r"Gold-sim$\uparrow$", 3),
}
HEADLINE_METRICS = ("leakage_auc", "nonlinear_leakage_auc_gb_flip", "cluster_purity",
                    "recall@1", "recall@10", "neighbor_retention@10")


def _cell(agg, cond, key, dec=3, bold=False):
    if cond not in agg or key not in agg[cond]:
        return "--"
    v = agg[cond][key]
    body = f"{v['mean']:.{dec}f}\\,{{\\scriptsize$\\pm${v.get('std', 0.0):.{dec}f}}}"
    return f"\\textbf{{{body}}}" if bold else body


def _table(caption, label, colspec, header, rows, note=None) -> str:
    lines = [r"\begin{table}[t]", r"  \centering", f"  \\caption{{{caption}}}",
             f"  \\label{{{label}}}", f"  \\begin{{tabular}}{{{colspec}}}",
             r"    \toprule", "    " + header + r" \\", r"    \midrule"]
    lines += [f"    {r}" if "&" not in r else f"    {r} \\\\" for r in rows]
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    if note:
        lines.append(f"  \\par\\smallskip\\footnotesize {note}")
    lines += [r"\end{table}", ""]
    return "\n".join(lines)


def _emit(body: str, name: str, save: bool) -> str:
    if save:
        save_table(body, name)
    return body


# --------------------------------------------------------------------------- #
def table_datasets(stats: dict, save: bool = True) -> str:
    """Table 1, from `corpora.stats` rather than hard-coded counts."""
    rows, seen_family = [], None
    for v in stats.values():
        family = v["confounder"]
        if seen_family and family != seen_family:
            rows.append(r"\midrule")
        seen_family = family
        pairs = f"{v['n_pairs']:,}".replace(",", "{,}") if v["n_pairs"] else "--"
        unp = v.get("n_unpaired", v["n_docs"] - 2 * v["n_pairs"])
        unp_s = f"{unp:,}".replace(",", "{,}") if unp else "--"
        rows.append(f"{v['label']} & {family.capitalize()} & "
                    f"{v['n_docs']:,}".replace(",", "{,}") +
                    f" & {pairs} & {unp_s} & {', '.join(v['classes'])}")
    body = _table(
        "Datasets, as loaded (exact duplicate texts removed). \\#Pairs is the "
        "number of matched cross-confounder pairs (retrieval targets); "
        "\\#Unpaired is the number of documents pooled with them as distractors "
        "and eraser training data but not used as retrieval targets, following "
        "the parent paper's construction.",
        "tab:datasets", "llrrrl",
        r"Dataset & Confounder & \#Docs & \#Pairs & \#Unpaired & Classes", rows,
        note=_selector_note(stats))
    return _emit(body, "table_datasets.tex", save)


def _selector_note(stats: dict) -> str:
    """The dataset-table note, worded for the selectors the corpora actually use."""
    sels = {v["label"]: v.get("selection", "?") for v in stats.values()}
    if set(sels.values()) == {"topn"}:
        lead = "All corpora use MI top-$N$ selection. "
    else:
        lead = ("Selection: " + ", ".join(f"{lab} {sel.replace('topn', 'top-$N$')}"
                                         for lab, sel in sels.items()) + ". ")
    return (lead + "Paired corpora report recall; the unpaired corpus reports "
            "leakage, cluster purity and within-source neighbourhood retention, "
            "since recall is undefined without matched pairs.")


def table_headline(results: dict, metrics=HEADLINE_METRICS, save: bool = True) -> str:
    """Raw / oracle / SAE across corpora. `results` maps corpus key -> result."""
    results = {kk: load_result(v) for kk, v in results.items()}
    metrics = [m for m in metrics
               if any(m in r["agg"].get("sae_leace", {}) for r in results.values())]
    header = "Dataset & Method & " + " & ".join(METRIC_COL[m][0] for m in metrics)
    rows, keys = [], list(results)
    for n, (key, r) in enumerate(results.items()):
        for i, cond in enumerate(("raw", "metadata_leace", "sae_leace")):
            cells = [_cell(r["agg"], cond, m, METRIC_COL[m][1], bold=(cond == "sae_leace"))
                     for m in metrics]
            lead = f"\\multirow{{3}}{{*}}{{{r['label']}}}" if i == 0 else ""
            rows.append(f"{lead} & {COND_LABEL[cond]} & " + " & ".join(cells))
        if n < len(keys) - 1:
            rows.append(r"\midrule")
    proto = next(iter(results.values())).get("protocol", "document")
    sels = sorted({r.get("config", {}).get("selection", "?") for r in results.values()})
    sel_note = {"topn": "MI top-$N$ selection", "greedy": "greedy selection"}
    sel_text = (" / ".join(sel_note.get(x, x) for x in sels)
                if len(sels) == 1 else
                "greedy selection (paired corpora) or MI top-$N$ (unpaired)")
    body = _table(
        "Metadata-free deconfounding, mean$\\pm$std over seeds, measured on "
        f"held-out documents under the {proto}-level holdout. Leakage and purity "
        "lower is better; recall and neighbourhood retention higher.",
        "tab:headline", "ll" + "c" * len(metrics), header, rows,
        note=f"SAE = {sel_text} with "
             "decoder-direction LEACE. Nbr.\\ is within-source neighbourhood "
             "retention@10, which raw attains at 1.000 by construction. Read the "
             "SAE row against the \\emph{oracle} row, not against chance: the "
             "oracle is label-supervised, so its held-out value is the floor "
             "attainable at this training size (see Table~\\ref{tab:floor}).")
    return _emit(body, "table_headline.tex", save)


def table_baselines(result, metrics=HEADLINE_METRICS, save: bool = True) -> str:
    """Every condition for one corpus, including the Section 4.5 baselines."""
    r = load_result(result)
    metrics = [m for m in metrics if any(m in r["agg"].get(c, {}) for c in r["conditions"])]
    header = "Method & " + " & ".join(METRIC_COL[m][0] for m in metrics)
    rows = []
    for cond in r["conditions"]:
        cells = [_cell(r["agg"], cond, m, METRIC_COL[m][1], bold=(cond == "sae_leace"))
                 for m in metrics]
        rows.append(f"{cond_label(cond)} & " + " & ".join(cells))
        if cond == "metadata_leace":
            rows.append(r"\midrule")
    body = _table(
        f"All conditions on {r['label']} ({r.get('protocol')}-level holdout, "
        f"{len(r['seeds'])} seeds).",
        f"tab:baselines-{r['corpus']}", "l" + "c" * len(metrics), header, rows)
    return _emit(body, f"table_baselines_{r['corpus']}.tex", save)


def table_baselines_all(results: dict, metrics=HEADLINE_METRICS,
                        exclude=("random_latents",), save: bool = True) -> str:
    """Every condition on every corpus, in one table.

    The four per-corpus `table_baselines` floats stacked into one, so the full
    condition set can be read across corpora instead of four tables apart.
    `results` maps corpus key -> result, exactly as `table_headline` takes it.
    `exclude` drops conditions from every block -- by default the size-matched
    random-latent control, which stays in the per-corpus tables. Pass ``()`` to
    keep every condition.
    """
    results = {kk: load_result(v) for kk, v in results.items()}
    metrics = [m for m in metrics
               if any(m in r["agg"].get("sae_leace", {}) for r in results.values())]
    ncols = 2 + len(metrics)
    header = "Dataset & Method & " + " & ".join(METRIC_COL[m][0] for m in metrics)
    rows, keys = [], list(results)
    for n, (key, r) in enumerate(results.items()):
        conds = [c for c in r["conditions"] if c not in exclude]
        for i, cond in enumerate(conds):
            cells = [_cell(r["agg"], cond, m, METRIC_COL[m][1],
                           bold=(cond == "sae_leace")) for m in metrics]
            lead = f"\\multirow{{{len(conds)}}}{{*}}{{{r['label']}}}" if i == 0 else ""
            rows.append(f"{lead} & {cond_label(cond)} & " + " & ".join(cells))
            if cond == "metadata_leace":
                # Rule the data columns only, so the corpus label is not cut in two.
                rows.append(f"\\cmidrule(l){{2-{ncols}}}")
        if n < len(keys) - 1:
            rows.append(r"\midrule")
    proto = next(iter(results.values())).get("protocol", "document")
    by_seeds: dict = {}
    for r in results.values():
        by_seeds.setdefault(len(r["seeds"]), []).append(r["label"])
    seed_text = "; ".join(f"{', '.join(v)} {k}" for k, v in sorted(by_seeds.items()))
    no_recall = [r["label"] for r in results.values()
                 if "recall@10" not in r["agg"].get("raw", {})]
    recall_note = (f" Recall is undefined for {', '.join(no_recall)} "
                   "(no matched pairs)." if no_recall else "")
    dropped = [cond_label(c) for c in exclude
               if any(c in r["conditions"] for r in results.values())]
    drop_note = (f" {' and '.join(dropped)} omitted here; see the per-corpus "
                 "baseline tables." if dropped else "")
    body = _table(
        "Raw, the metadata oracle, SAE-LEACE and the label-free baselines on all "
        "four corpora, mean$\\pm$std over seeds, measured "
        f"on held-out documents under the {proto}-level holdout. Leakage and "
        "purity lower is better; recall and neighbourhood retention higher.",
        "tab:baselines-all", "ll" + "c" * len(metrics), header, rows,
        note="Raw and the label-supervised metadata oracle are the two reference "
             "conditions, ruled off from the label-free method and the baselines "
             "it is compared against." + drop_note + recall_note +
             f" Seeds: {seed_text}.")
    return _emit(body, "table_baselines_all.tex", save)


def table_protocol(results: dict, key: str = "leakage_auc", save: bool = True) -> str:
    """Pair holdout vs document holdout, per corpus and condition."""
    results = {kk: load_result(v) for kk, v in results.items()}
    header = (r"Dataset & Method & Pair holdout & Document holdout & $\Delta$")
    rows, keys = [], list(results)
    for n, (ck, r) in enumerate(results.items()):
        for i, cond in enumerate(("raw", "metadata_leace", "sae_leace")):
            a = r["variants"]["pair"]["agg"].get(cond, {}).get(key)
            b = r["variants"]["document"]["agg"].get(cond, {}).get(key)
            if a is None or b is None:
                continue
            lead = f"\\multirow{{3}}{{*}}{{{r['label']}}}" if i == 0 else ""
            rows.append(f"{lead} & {COND_LABEL[cond]} & {a['mean']:.3f} & "
                        f"{b['mean']:.3f} & {b['mean'] - a['mean']:+.3f}")
        if n < len(keys) - 1:
            rows.append(r"\midrule")
    body = _table(
        "Unseen documents. The same corpora, seeds and recipe under both split "
        "protocols; the column is the linear leakage AUC. The pair holdout "
        "withholds retrieval links but shows every document to the SAE, the "
        "$k$-means proxy, the selector and the erasers, which drives the oracle's "
        "cross-validated AUC below chance; the document holdout withholds the "
        "documents themselves. Both arms are pinned to a single 50/50 split so "
        "that only the protocol varies, so the right-hand column is a controlled "
        "contrast and \\emph{not} the headline number: corpora whose own policy "
        "is $K$-fold (SemEval) are reported in Table~\\ref{tab:headline} at a "
        "larger training size and therefore a lower floor.",
        "tab:protocol", "llccc", header, rows,
        note="A large positive $\\Delta$ means the transductive protocol was "
             "flattering the condition. Leakage and recall are both comparable "
             "across the two columns: under either protocol the held-out pairs "
             "are the queries and the retrieval index is the eraser applied to "
             "the whole corpus, so the candidate pool is the same size.")
    return _emit(body, "table_protocol.tex", save)


def table_selection(results: dict, save: bool = True) -> str:
    """Top-N vs greedy, with the size-matched arm."""
    results = {kk: load_result(v) for kk, v in results.items()}
    metrics = ("leakage_auc", "recall@10", "neighbor_retention@10")
    header = (r"Dataset & Selector & $|S|$ & "
              + " & ".join(METRIC_COL[m][0] for m in metrics))
    labels = {"topn": "Top-$N$ ($N{=}32$)", "topn_matched": "Top-$N$ (size-matched)",
              "greedy": "Greedy"}
    rows, keys = [], list(results)
    for n, (ck, r) in enumerate(results.items()):
        for i, v in enumerate(r["variant_order"]):
            e = r["variants"][v]
            sel = np.mean(e.get("n_selected") or
                          [s["n_selected"] for s in e["per_seed"]])
            cells = [_cell(e["agg"], "sae_leace", m, 3, bold=(v == "greedy"))
                     for m in metrics]
            lead = f"\\multirow{{3}}{{*}}{{{r['label']}}}" if i == 0 else ""
            rows.append(f"{lead} & {labels.get(v, v)} & {sel:.0f} & " + " & ".join(cells))
        if n < len(keys) - 1:
            rows.append(r"\midrule")
    body = _table(
        "Latent selection at an equal erasure budget. Greedy chooses its own "
        "$|S|$; the size-matched row truncates the mutual-information ranking to "
        "that same $|S|$ per seed, which is what separates \\emph{which} latents a "
        "selector picks from \\emph{how many}.",
        "tab:selection", "llc" + "c" * len(metrics), header, rows,
        note="Document-level holdout; the SAE condition only.")
    return _emit(body, "table_selection.tex", save)


def table_score_function(results: dict, save: bool = True) -> str:
    """MI vs normalized MI vs correlation ratio, at an equal erasure budget."""
    results = {kk: load_result(v) for kk, v in results.items()}
    metrics = ("leakage_auc", "nonlinear_leakage_auc_gb_flip", "cluster_purity",
               "recall@10", "neighbor_retention@10")
    metrics = [m for m in metrics
               if any(m in r["variants"][v]["agg"].get("sae_leace", {})
                      for r in results.values() for v in r["variant_order"])]
    header = (r"Dataset & Score & $|S|$ & "
              + " & ".join(METRIC_COL[m][0] for m in metrics))
    labels = {"mi": "MI (ours)", "nmi": "Normalized MI",
              "corr_ratio": r"Correlation ratio ($\eta^2$)"}
    rows, keys = [], list(results)
    for n, (ck, r) in enumerate(results.items()):
        order = r["variant_order"]
        for i, v in enumerate(order):
            e = r["variants"][v]
            sel = np.mean([s["n_selected"] for s in e["per_seed"]])
            cells = [_cell(e["agg"], "sae_leace", m, 3, bold=(v == "mi"))
                     for m in metrics]
            lead = (f"\\multirow{{{len(order)}}}{{*}}{{{r['label']}}}"
                    if i == 0 else "")
            rows.append(f"{lead} & {labels.get(v, v)} & {sel:.0f} & "
                        + " & ".join(cells))
        if n < len(keys) - 1:
            rows.append(r"\midrule")
    ov = []
    for r in results.values():
        j = r.get("selection_jaccard", {})
        if j:
            ov.append(f"{r['label']} " + "/".join(f"{x:.2f}" for x in j.values()))
    body = _table(
        "Does the scoring function matter? Latents ranked by mutual information "
        "(the shipped score), normalized MI, and the magnitude-aware correlation "
        "ratio, all against the same label-free $k$-means pseudo-grouping of the "
        "training documents. One SAE per split is shared by the three rankings "
        "and selection is top-$N$ at a fixed budget, so $|S|$ is identical and "
        "only \\emph{which} latents are chosen varies.",
        "tab:score-function", "llc" + "c" * len(metrics), header, rows,
        note="Document-level holdout, SAE condition only, mean$\\pm$std over "
             "seeds. Jaccard overlap of the selected sets "
             "(mi$|$nmi / mi$|$corr / nmi$|$corr): " + "; ".join(ov) + ".")
    return _emit(body, "table_score_function.tex", save)


def table_encoders(grid: dict, key: str = "leakage_auc", save: bool = True) -> str:
    """The encoder x corpus grid: raw -> SAE -> oracle leakage per cell."""
    grid = load_result(grid)
    cells = {kk: v for kk, v in grid.items() if not kk.startswith("_") and "agg" in v}
    if not cells:
        print("[tables] encoder grid has no finished cells")
        return ""
    corpora_keys, encoders = [], []
    for v in cells.values():
        if v["corpus"] not in corpora_keys:
            corpora_keys.append(v["corpus"])
        if v["encoder"] not in encoders:
            encoders.append(v["encoder"])
    labels = {v["corpus"]: v["label"] for v in cells.values()}
    header = "Encoder & " + " & ".join(labels[c] for c in corpora_keys)
    rows = []
    for enc in encoders:
        out = [enc.replace("_", "-")]
        for ck in corpora_keys:
            v = next((x for x in cells.values()
                      if x["encoder"] == enc and x["corpus"] == ck), None)
            if v is None:
                out.append("--")
                continue
            raw, orc, sae = (v["agg"][c][key]["mean"] for c in
                             ("raw", "metadata_leace", "sae_leace"))
            out.append(f"{raw:.2f} $\\to$ \\textbf{{{sae:.2f}}} ({orc:.2f})")
        rows.append(" & ".join(out))
    body = _table(
        "Encoder generalisation: linear leakage AUC on held-out documents, as "
        "raw $\\to$ \\textbf{SAE} (oracle), under the document-level holdout.",
        "tab:encoders", "l" + "c" * len(corpora_keys), header, rows,
        note="No encoder is fine-tuned; each is used with its pretrained weights "
             "and its own recommended prompt, applied identically across conditions.")
    return _emit(body, "table_encoders.tex", save)


def table_corpus_size(result, save: bool = True) -> str:
    """Linear and nonlinear leakage against corpus size."""
    r = load_result(result)
    header = (r"Size (" + r["unit"] + r") & \#Docs & "
              r"Oracle lin. & Oracle GB & SAE lin. & SAE GB")
    rows = []
    for v in r["variant_order"]:
        e = r["variants"][v]
        a = e["agg"]
        def g(cond, kk):
            m = a.get(cond, {}).get(kk)
            return f"{m['mean']:.3f}" if m else "--"
        rows.append(f"{v} & {e['n_docs']:,}".replace(",", "{,}") +
                    f" & {g('metadata_leace','leakage_auc')}"
                    f" & {g('metadata_leace','nonlinear_leakage_auc_gb')}"
                    f" & {g('sae_leace','leakage_auc')}"
                    f" & {g('sae_leace','nonlinear_leakage_auc_gb')}")
    body = _table(
        f"Corpus size on {r['label']}. The subsample at each size is fixed by the "
        "size rather than by the pipeline seed, so the seeds vary the pipeline and "
        "not the sample.",
        f"tab:corpus-size-{r['corpus']}", "lrcccc", header, rows,
        note="GB = gradient-boosted nonlinear probe. If nonlinear leakage climbs "
             "with $n$, near-chance nonlinear leakage at small $n$ was probe "
             "under-training and the claim must be scoped to corpus size.")
    return _emit(body, f"table_corpus_size_{r['corpus']}.tex", save)


def table_corpus_size_all(results: dict, save: bool = True) -> str:
    """Both corpus-size sweeps in one table, one block per corpus.

    `results` maps corpus key -> result. The size column carries a different
    unit per corpus (pairs on a paired corpus, documents on an unpaired one),
    so the units are named in the note rather than in the column head.
    """
    results = {kk: load_result(v) for kk, v in results.items()}
    header = (r"Dataset & Size & \#Docs & "
              r"Oracle lin. & Oracle GB & SAE lin. & SAE GB")
    rows, keys = [], list(results)
    for n, (ck, r) in enumerate(results.items()):
        order = r["variant_order"]
        for i, v in enumerate(order):
            e = r["variants"][v]
            a = e["agg"]

            def g(cond, kk, _a=a):
                m = _a.get(cond, {}).get(kk)
                return f"{m['mean']:.3f}" if m else "--"
            lead = (f"\\multirow{{{len(order)}}}{{*}}{{{r['label']}}}"
                    if i == 0 else "")
            rows.append(f"{lead} & {v} & " + f"{e['n_docs']:,}".replace(",", "{,}")
                        + f" & {g('metadata_leace','leakage_auc')}"
                        f" & {g('metadata_leace','nonlinear_leakage_auc_gb')}"
                        f" & {g('sae_leace','leakage_auc')}"
                        f" & {g('sae_leace','nonlinear_leakage_auc_gb')}")
        if n < len(keys) - 1:
            rows.append(r"\midrule")
    units = "; ".join(f"{r['label']} in {r['unit']}" for r in results.values())
    body = _table(
        "Leakage against corpus size, on the two corpora large enough to sweep. "
        "The subsample at each size is fixed by the size rather than by the "
        "pipeline seed, so the seeds vary the pipeline and not the sample.",
        "tab:corpus-size", "llrcccc", header, rows,
        note=f"Sizes are counted per corpus: {units}. GB = gradient-boosted "
             "nonlinear probe. If nonlinear leakage climbs with $n$, "
             "near-chance nonlinear leakage at small $n$ was probe "
             "under-training and the claim must be scoped to corpus size.")
    return _emit(body, "table_corpus_size_all.tex", save)


def table_hyperparameters(section_dirs=("results/A", "results/B", "results/C"),
                          save: bool = True) -> str:
    """Table 10: the configuration each experiment actually ran with.

    Walks the ``raw/`` trees and reads the ``config`` block every result carries,
    so the hyperparameter table is a report of the runs rather than a hand-kept
    list that can drift from them. One row per distinct (experiment, config).
    """
    import json
    import os

    fields = ("expansion_factor", "k", "n_epochs", "pseudo_groups", "erase",
              "selection", "top_features", "greedy_lambda", "greedy_budget",
              "greedy_pool", "protocol", "scheme", "holdout")
    seen: dict = {}
    for sec in section_dirs:
        raw = os.path.join(sec, "raw")
        if not os.path.isdir(raw):
            continue
        for root, _dirs, files in sorted(os.walk(raw)):
            for fn in sorted(f for f in files if f.endswith(".json")):
                try:
                    with open(os.path.join(root, fn)) as fh:
                        r = json.load(fh)
                except (OSError, ValueError):
                    continue
                if not isinstance(r, dict):
                    continue
                exp = r.get("experiment")
                cfg = r.get("config") or {}
                if not exp or not cfg:
                    continue
                row = tuple(cfg.get(f, "--") for f in fields)
                seen.setdefault((exp, row), []).append(r.get("corpus", "?"))

    if not seen:
        print("[tables] no configs found; run the experiments first")
        return ""
    show = ("expansion_factor", "k", "n_epochs", "pseudo_groups", "erase",
            "selection", "greedy_lambda", "protocol", "scheme")
    idx = [fields.index(f) for f in show]
    head_map = {"expansion_factor": r"$\gamma$", "k": "$k$", "n_epochs": "epochs",
                "pseudo_groups": "$K$", "erase": "erasure", "selection": "selector",
                "greedy_lambda": r"$\lambda$", "protocol": "holdout",
                "scheme": "split"}
    header = "Experiment & Corpora & " + " & ".join(head_map[f] for f in show)
    rows = []
    for (exp, row), corps in sorted(seen.items()):
        uniq = sorted(set(c for c in corps if c))
        corp = ", ".join(uniq) if len(uniq) <= 2 else f"{len(uniq)} corpora"
        cells = [str(row[i]).replace("_", r"\_") for i in idx]
        rows.append(f"{exp.replace('_', ' ')} & {corp.replace('_', chr(92) + '_')} & "
                    + " & ".join(cells))
    body = _table(
        "Configuration per experiment, read back from the saved runs rather than "
        "kept by hand. $\\gamma$ is the SAE expansion factor and $k$ its TopK "
        "sparsity; $K$ is the number of $k$-means pseudo-groups (``auto'' = chosen "
        "by silhouette on the training documents, without labels); $\\lambda$ "
        "weights the leakage term in the greedy objective.",
        "tab:hyperparameters", "ll" + "c" * len(show), header, rows,
        note="``document\'\' holdout = every component fit on the training "
             "documents and evaluated on held-out ones.")
    return _emit(body, "table_hyperparameters.tex", save)


def table_latents(results: dict, top_m: int = 6, save: bool = True) -> str:
    """tab:latents -- the most strongly aligned selected latents, per corpus.

    The first three numeric columns are the label-free quantities available to
    selection; purity, lift and dense AUC are the post-hoc alignment check.
    Requires `analysis.latent_alignment` results that carry ``purity``: older
    JSONs scored the TopK-gated activation by AUC, which is capped by the firing
    rate and reads as chance for any sparse latent. Rerun the descriptive cell
    to regenerate them.
    """
    results = {kk: load_result(v) for kk, v in results.items()}
    for kk, r in results.items():
        if not r["latents"] or "purity" not in r["latents"][0]:
            raise ValueError(
                f"{kk}: this latent_alignment result predates the purity metric "
                "(it carries only the gated-activation AUC, which is capped by the "
                "firing rate). Rerun analysis.fit_descriptive + latent_alignment.")
    header = (r"Dataset & Latent & MI & Pair asym. & Firing & "
              r"Purity & Lift & Dense AUC & Class")
    rows, keys = [], list(results)
    for n, (ck, r) in enumerate(results.items()):
        top = r["latents"][:top_m]
        for i, lat in enumerate(top):
            asym = ("--" if lat["pair_asymmetry"] is None
                    else f"{lat['pair_asymmetry']:.2f}")
            lead = (f"\\multirow{{{len(top)}}}{{*}}{{{r['label']}}}" if i == 0 else "")
            rows.append(f"{lead} & {lat['latent']} & {lat['mi_score']:.4f} & {asym} & "
                        f"{lat['firing_freq']:.3f} & "
                        f"\\textbf{{{lat['purity']:.2f}}} & {lat['purity_lift']:.1f} & "
                        f"{lat['dense_auc']:.3f} & "
                        f"\\texttt{{{lat['aligned_class']}}}")
        if n < len(keys) - 1:
            rows.append(r"\midrule")
    body = _table(
        "The most strongly aligned selected latents, grouped by \\emph{dataset} "
        "(one block per corpus) and ranked within a block by purity. "
        "\\emph{Latent} is the SAE dictionary index. The next three columns are "
        "label-free and are all that selection sees: \\emph{MI} is the mutual "
        "information (bits) between ``this latent fires'' and a $k$-means "
        "pseudo-grouping of the raw embeddings, and is the top-$N$ ranking score; "
        "\\emph{pair asym.} is the mean absolute within-pair activation difference "
        "normalised by the latent's activation scale, so a latent firing on one "
        "side of a translation pair only scores high (it needs a paired corpus --- "
        "``--'' where there is none --- and only the greedy selector ranks by it); "
        "\\emph{firing} is the fraction of documents on which the latent is active. "
        "The remaining columns use labels, are computed \\emph{after the fact} and "
        "never enter scoring or selection: \\emph{purity} is the share of the "
        "documents a latent fires on that belong to its aligned class, \\emph{lift} "
        "is that purity over the class's prior (so $2.0$ is the ceiling on a "
        "balanced binary corpus), \\emph{dense AUC} is the one-vs-rest AUC of the "
        "pre-activation against the aligned class, and \\emph{class} is the class "
        "attaining maximum purity.",
        "tab:latents", "llrrrrrrl", header, rows,
        note="One SAE per corpus (seed 0), fit on the full corpus with no "
             "holdout. Descriptive: no performance number is quoted from these "
             "runs. The AUC of the TopK-gated activation is not shown: it is "
             "capped at $0.5 + \\min(1, f/p)/2$ for firing rate $f$ and class "
             "prior $p$, so a perfectly class-pure latent firing on 3\\% of "
             "documents cannot exceed 0.53 against a 50\\% class.")
    return _emit(body, "table_latents.tex", save)


def table_subspace(results: dict, exclude=("random",), save: bool = True) -> str:
    """Capture of the confounder subspace by each candidate removal subspace.

    Candidate names carry their own dimension (``"PCA top-21"``), which differs
    per corpus because it is $|S|$; the columns are therefore matched on a
    canonical prefix, not on the full name, so a corpus with a different $|S|$
    lands in the right column instead of falling through to "--".

    `exclude` drops candidates by canonical name -- by default the random
    size-matched control. Pass ``()`` to show every candidate.
    """
    results = {kk: load_result(v) for kk, v in results.items()}

    def canon(name: str) -> str:
        low = name.lower()
        if low.startswith("pca"):
            return "pca"
        if "random" in low:
            return "random"
        if "raw atoms" in low:
            return "atoms"
        return "selected"

    order = [c for c in ("selected", "atoms", "random", "pca")
             if c not in exclude]
    head_label = {
        "selected": r"SAE selected ($\Sigma$-wt.)",
        "atoms": "SAE selected (atoms)",
        "random": r"SAE random ($\Sigma$-wt.)",
        "pca": r"PCA top-$|S|$",
    }
    header = "Dataset & rank $L$ & $|S|$ & " + " & ".join(head_label[c] for c in order)
    rows = []
    for r in results.values():
        rank = r.get("subspace_rank", r["language_rank"])
        by_key = {canon(n): c["capture"] for n, c in r["candidates"].items()
                  if canon(n) not in exclude}
        best = max(by_key, key=lambda kk: by_key[kk]) if by_key else None
        cells = []
        for c in order:
            v = by_key.get(c)
            cells.append("--" if v is None else
                         (f"\\textbf{{{v:.3f}}}" if c == best else f"{v:.3f}"))
        rows.append(f"{r['label']} & {rank} & {r['n_selected']} & " + " & ".join(cells))
    body = _table(
        "Normalised capture of the confounder subspace $L=\\operatorname{col}"
        "(X_c^\\top Y_c)$ by each size-matched candidate removal subspace: the "
        "mean squared cosine of the principal angles from $L$, where 1 means the "
        "candidate contains $L$ entirely and 0 means orthogonality. Bold marks "
        "the largest capture in each row.",
        "tab:subspace", "lrr" + "c" * len(order), header, rows,
        note="All candidates have dimension $|S|$. The covariance-weighted "
             "selected directions are what \\texttt{directions\\_leace} removes; "
             "the unweighted atoms are what \\texttt{directions\\_orth} removes. "
             "Capture is necessary but not sufficient: it says how much of $L$ a "
             "subspace contains, not what else it removes.")
    return _emit(body, "table_subspace.tex", save)


def table_floor(results: dict, key: str = "leakage_auc", save: bool = True) -> str:
    """tab:floor -- the label-free method against the attainable floor.

    Chance (0.5) is the asymptotic floor. The oracle, being label-supervised on
    the same held-out documents, is the floor actually reachable at this corpus
    size and encoder dimension -- and on a small corpus the two are far apart.
    The $n/d$ column is why: it is the training rows per embedding dimension, and
    below roughly 2 the LEACE whitening is barely determined.
    """
    results = {kk: load_result(v) for kk, v in results.items()}
    header = (r"Dataset & Split & $n_{\text{train}}$ & $n/d$ & Raw & "
              r"Floor (oracle) & SAE & Recovered")
    rows = []
    for r in results.values():
        m = r["metrics"].get(key)
        if m is None:
            continue
        d = r["diagnostics"]
        rec = m["recovery"]
        scheme = r.get("scheme") or "holdout"
        nd = d.get("n_over_d")
        nd_s = "--" if nd is None else (f"\\textbf{{{nd:.2f}}}" if nd < 2 else f"{nd:.2f}")
        rows.append(
            f"{r['label']} & {scheme} & {d['n_train_docs']:,}".replace(",", "{,}")
            + f" & {nd_s} & {m['raw']:.3f} & {m['floor']:.3f} & {m['sae']:.3f} & "
            + ("--" if rec is None else f"{rec:.0%}").replace("%", r"\%"))
    body = _table(
        "The label-free method against the \\emph{attainable} floor. The floor is "
        "the metadata oracle measured on the same held-out documents: being "
        "label-supervised, whatever it fails to remove out of sample is "
        "estimation error rather than a limit of erasure. $n/d$ is training rows "
        "per embedding dimension; bold marks $n/d<2$, where the whitening is "
        "barely determined and the floor is sample-size limited.",
        "tab:floor", "llrrcccc", header, rows,
        note="Recovered $=(\\text{raw}-\\text{SAE})/(\\text{raw}-\\text{floor})$: "
             "the share of the removable damage the label-free method removes.")
    return _emit(body, "table_floor.tex", save)


# --------------------------------------------------------------------------- #
# Frozen cross-corpus transfer (Appendix C), from `ood.summarize`
# --------------------------------------------------------------------------- #
OOD_TRANSFORMS = ("pca_top1", "metadata_leace", "sae_leace")
OOD_LABEL = {"raw": "Raw", "pca_top1": "PCA-1", "metadata_leace": "Metadata-LEACE",
             "sae_leace": "SAE-LEACE"}
ENCODER_LABEL = {"e5-small": "E5-small", "mini": "MiniLM", "e5-base-v2": "E5-base-v2"}


def _ood_transforms(summary: dict) -> tuple:
    """The transformations the summary holds, in its order."""
    return tuple(summary.get("transforms") or OOD_TRANSFORMS)


def _ood_label(summary: dict, name: str) -> str:
    """A condition's display name: the one the run recorded, else the default."""
    return (summary.get("labels") or {}).get(name) or OOD_LABEL.get(name, name)


def _num(n) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def _signed(v: float, dec: int = 3) -> str:
    return f"${v:+.{dec}f}$"


def _panels_table(caption, label, panels, note=None) -> str:
    """One float holding several tabulars, one per panel ``(title, colspec,
    header, rows)``, for tables whose panels have different columns."""
    lines = [r"\begin{table}[t]", r"  \centering", f"  \\caption{{{caption}}}",
             f"  \\label{{{label}}}"]
    for n, (title, colspec, header, rows) in enumerate(panels):
        if n:
            lines.append(r"  \medskip")
        lines += [f"  \\par\\textit{{{title}}}\\par\\smallskip",
                  f"  \\begin{{tabular}}{{{colspec}}}", r"    \toprule",
                  "    " + header + r" \\", r"    \midrule"]
        lines += [f"    {r}" if "&" not in r else f"    {r} \\\\" for r in rows]
        lines += [r"    \bottomrule", r"  \end{tabular}"]
    if note:
        lines.append(f"  \\par\\smallskip\\footnotesize {note}")
    lines += [r"\end{table}", ""]
    return "\n".join(lines)


def _ood_settings(summary: dict) -> list:
    return [summary["settings"][k] for k in summary["order"]]


def _ood_fits(summary: dict) -> str:
    """"the three donor fits", worded from the seeds the summary actually holds."""
    n = len(_ood_settings(summary)[0]["seeds"])
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    return f"the {words.get(n, n)} donor fit{'s' if n != 1 else ''}"


def table_ood_inventory(summary, save: bool = True) -> str:
    """Table C1: the six targets (Panel A) and the five donor fits (Panel B)."""
    s = load_result(summary)
    settings = _ood_settings(s)
    tasks = settings[0]["tasks"]
    rows_a = []
    for t, v in tasks.items():
        size = v["size"]
        size_s = (f"{_num(size['queries'])} / {_num(size['documents'])} / "
                  f"{_num(size['judgements'])}" if v["kind"] == "retrieval"
                  else f"{_num(size['pairs'])} pairs")
        metric = "nDCG@10" if v["kind"] == "retrieval" else "Cosine Spearman"
        rows_a.append(f"{t} & \\texttt{{{v['revision'][:7]}}} & test, English & "
                      f"{size_s} & {metric}")
    rows_b = []
    for v in settings:
        n_tr = sorted(set(v["n_train_docs"]))
        seeds = ", ".join(str(x) for x in v["seeds"])
        how = "all documents" if v.get("fit_on") == "full" else "50/50 holdout"
        rows_b.append(f"{ENCODER_LABEL.get(v['encoder'], v['encoder'])} & "
                      f"{v['donor_label']} & "
                      f"{' / '.join(_num(n) for n in n_tr)} ({how}) & "
                      f"{seeds} & all six")
    body = _panels_table(
        "Frozen-transfer evaluation inventory. Each eraser is fit on the donor's "
        "ID training split and applied unchanged to every target, encoded by the "
        "same model.",
        "tab:ood_inventory",
        [("A. Targets (mteb task definitions; dataset revisions as pinned there)",
          "lllrl", r"Task & Revision & Split, language & Size & Score", rows_a),
         ("B. Donor fits", "llrll",
          r"Encoder & Donor & Training documents & Fit seeds & Targets", rows_b)],
        note="Retrieval sizes are \\#queries / \\#corpus documents / \\#relevance "
             "judgements; STS sizes are rated sentence pairs. A 50/50 holdout fit "
             "uses the ID training split of each seed; the fingerprint of the "
             "documents each fit saw is in \\texttt{ood\\_summary.json}.")
    return _emit(body, "table_ood_inventory.tex", save)


def table_ood_scores(summary, save: bool = True) -> str:
    """Table C2: absolute target scores, one panel per donor/encoder.

    Transformed cells are mean$\\pm$SD over the fitting seeds; Raw has no fit
    and is one value, repeated across the donor panels of an encoder. The legal
    macro-average is taken within each seed, then summarised across seeds.
    """
    s = load_result(summary)
    transforms = _ood_transforms(s)
    ncol = 3 + len(transforms)
    header = "Target task & Metric & " + " & ".join(_ood_label(s, c) for c in
                                                      ("raw",) + transforms)
    rows = []
    for n, v in enumerate(_ood_settings(s)):
        if n:
            rows.append(r"\midrule")
        rows.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{"
                    f"{ENCODER_LABEL.get(v['encoder'], v['encoder'])} / "
                    f"{v['donor_label']}}}}} \\\\")
        keys = v["legal"] + ["legal_macro"] + v["sts"]
        for kk in keys:
            if kk == "legal_macro":
                name, metric = "Legal macro-average", "Mean nDCG@10"
            else:
                name = v["tasks"][kk]["label"]
                metric = v["tasks"][kk]["metric"]
            cells = [f"{v['scores']['raw'][kk]['mean']:.3f}"]
            for c in transforms:
                x = v["scores"][c][kk]
                cells.append(f"{x['mean']:.3f}\\,{{\\scriptsize$\\pm${x['std']:.3f}}}")
            if kk in v["sts"]:
                rows.append(f"\\cmidrule(l){{1-{ncol}}}")
            rows.append(f"{name} & {metric} & " + " & ".join(cells))
    body = _table(
        "All frozen-transfer target scores. Transformed conditions are mean$\\pm$SD "
        f"over {_ood_fits(s)}; Raw is deterministic given the cached "
        "embeddings and is shown once.",
        "tab:ood_scores", "ll" + "c" * (ncol - 2), header, rows,
        note="The legal macro-average is the equal-weight mean of the five legal "
             "tasks within each fit; STSBenchmark is not part of it.")
    return _emit(body, "table_ood_scores.tex", save)


def table_ood_changes(summary, save: bool = True) -> str:
    """Table C3: changes from Raw with paired bootstrap intervals.

    Panel A: mean legal $\\Delta$nDCG@10 and the task with the smallest change
    (descriptive, not a test). Panel B: STSBenchmark $\\Delta$Spearman.
    """
    s = load_result(summary)
    lvl = int(round(100 * s["level"]))
    rows_a, rows_b = [], []
    for n, v in enumerate(_ood_settings(s)):
        if n:
            rows_a.append(r"\midrule")
            rows_b.append(r"\midrule")
        enc = ENCODER_LABEL.get(v["encoder"], v["encoder"])
        for c in _ood_transforms(s):
            ch = v["changes"][c]
            lo, hi = ch["legal_macro"]["ci"]
            worst = ch["worst_task"]
            rows_a.append(f"{enc} & {v['donor_label']} & {_ood_label(s, c)} & "
                          f"{_signed(ch['legal_macro']['delta'])} & "
                          f"[{_signed(lo)}, {_signed(hi)}] & "
                          f"{v['tasks'][worst['task']]['label']} & "
                          f"{_signed(worst['delta'])}")
            for t in v["sts"]:
                lo, hi = ch[t]["ci"]
                rows_b.append(f"{enc} & {v['donor_label']} & {_ood_label(s, c)} & "
                              f"{_signed(ch[t]['delta'])} & [{_signed(lo)}, {_signed(hi)}]")
    body = _panels_table(
        "Changes from Raw under frozen transfer, in the score averaged over "
        f"{_ood_fits(s)}, with paired {lvl}\\% bootstrap intervals conditional "
        "on those fits.",
        "tab:ood_changes",
        [("A. Legal retrieval ($\\Delta$nDCG@10)", "lllcccc",
          r"Encoder & Donor & Transform & Mean legal $\Delta$ & "
          f"{lvl}\\% CI & Smallest-change task & Task $\\Delta$", rows_a),
         ("B. STSBenchmark ($\\Delta$Spearman)", "lllcc",
          r"Encoder & Donor & Transform & $\Delta$Spearman & "
          f"{lvl}\\% CI", rows_b)],
        note=f"{s['n_boot']:,}".replace(",", "{,}") + " bootstrap resamples. "
             "Queries are resampled within each task, with the two AILA tasks "
             "(one shared query set) resampled jointly, and the task deltas "
             "averaged; STS Spearman is recomputed on resampled pairs. The "
             "smallest-change task is the lowest seed-averaged task delta; it is "
             "descriptive, not a prespecified test.")
    return _emit(body, "table_ood_changes.tex", save)


def table_ood_parent(summary, save: bool = True) -> str:
    """Our Raw and metadata-LEACE scores beside the parent paper's (Tables 7, 9).

    One row per encoder / donor / task the parent ran: a check that the data,
    encoders, donor fits and evaluation reproduce the parent's pipeline.
    """
    s = load_result(summary)
    rows, prev = [], None
    for r in s.get("parent_comparison", []):
        head = (r["encoder"], r["role"])
        if prev and head != prev:
            rows.append(r"\midrule")
        prev = head
        o, p = r["ours"], r["parent"]
        rows.append(f"{ENCODER_LABEL.get(r['encoder'], r['encoder'])} & "
                    f"{'CAP' if r['role'] == 'cap' else 'Legal'} & {r['label']} & "
                    f"{o['raw']:.3f} & {p['raw']:.3f} & {o['metadata_leace']:.3f} & "
                    f"{p['metadata_leace']:.3f}")
    body = _table(
        "Replication check against the parent paper. Ours are the mean over "
        f"{_ood_fits(s)}; the parent's are its Tables 7 (legal nDCG@10) and 9 "
        "(STSBenchmark Spearman).",
        "tab:ood_parent", "lllcccc",
        r"Encoder & Donor & Task & \multicolumn{2}{c}{Raw} & "
        r"\multicolumn{2}{c}{Metadata-LEACE} \\"
        "\n    & & & ours & parent & ours & parent", rows)
    return _emit(body, "table_ood_parent.tex", save)
