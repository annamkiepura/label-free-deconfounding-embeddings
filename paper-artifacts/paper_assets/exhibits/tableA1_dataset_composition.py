"""Appendix Table A1: dataset composition and linked-group construction.

The four SwiLTra and four SCOTUS settings exactly as the project's loaders build them
(`corpora.load`: `run_sae_experiment.load_swiltra`, `load_case_summaries`,
`load_case_summaries_union`). Linked groups are the connected components of the
registered pair links (`protocols.link_groups`), the unit the split engine keeps on one
side; retrieval pairs are every two members of a group. Nothing is merged, relinked or
re-deduplicated here: the construction diagnostics re-run the loaders' own steps and the
result is checked to equal `corpora.load` exactly.
"""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np

from ..lib import config, latex
from ..lib.datasets import processed
from ..lib.io import file_sha256
from ..lib.provenance import check, issue

NAME, KIND, TITLE = "tableA1_dataset_composition", "table", "Dataset composition and linked-group construction"
LABEL = "tab:dataset-composition"
BLOCKS = [("SwiLTra", "language", [("swiltra", "DE--FR"), ("swiltra_de_it", "DE--IT"),
                                   ("swiltra_fr_it", "FR--IT"), ("swiltra_de_fr_it", "DE--FR--IT")]),
          ("SCOTUS", "source", [("scotus_oyez_lexis", "Oyez--Lexis"), ("scotus_oyez_wiki", "Oyez--Wikipedia"),
                                ("scotus_wiki_lexis", "Lexis--Wikipedia"),
                                ("scotus_oyez_lexis_wiki", "All three")])]
CLS = {"de": "DE", "fr": "FR", "it": "IT", "oyez": "Oyez", "lexis": "Lexis", "wiki": "Wikipedia"}
#: Compact labels for the single-column table: the class-column heads of each block and the
#: row labels (SCOTUS sources abbreviated O/L/W, defined in the caption).
HEAD = {**CLS, "wiki": "Wiki"}
SHORT = {"swiltra": "DE--FR", "swiltra_de_it": "DE--IT", "swiltra_fr_it": "FR--IT",
         "swiltra_de_fr_it": "DE--FR--IT", "scotus_oyez_lexis": "O--L", "scotus_oyez_wiki": "O--W",
         "scotus_wiki_lexis": "L--W", "scotus_oyez_lexis_wiki": "O--L--W"}
ABBR = {"oyez": "O", "lexis": "L", "wiki": "W"}
ORDER = {"de": 0, "fr": 1, "it": 2, "oyez": 0, "lexis": 1, "wiki": 2}
_STATE: dict = {}


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def _combo(a: str, b: str) -> str:
    return "|".join(sorted((a, b), key=ORDER.get))


# --------------------------------------------------------------------------- #
def _compose(key: str, d: dict, P) -> dict:
    """Composition of one loaded setting, with the structural checks."""
    y = np.asarray(d["source"])
    n = int(d["n_docs"])
    classes = sorted(np.unique(y).tolist(), key=ORDER.get)
    counts = {c: int((y == c).sum()) for c in classes}
    pairs = [tuple(map(int, p)) for p in (d["pairs"] or [])]
    of_pair, groups = P.link_groups(pairs)
    sizes = Counter(len(g) for g in groups)
    in_group = Counter(i for g in groups for i in g)
    linked = set(in_group)
    tag = f"{key}"
    check(sum(counts.values()) == n == len(d["texts"]), "a1_class_totals",
          f"{tag}: class counts {counts} sum to the {n} documents")
    check(max(in_group.values(), default=1) == 1, "a1_group_membership",
          f"{tag}: every linked document belongs to exactly one linked group ({len(linked)} documents)")
    check(all(len(set(y[g])) == len(g) for g in groups), "a1_group_class_unique",
          f"{tag}: no linked group holds two documents of one {d['config']['confounder']} class")
    check(set(sizes) <= {2, 3}, "a1_group_sizes", f"{tag}: group sizes {dict(sizes)} (pairs and triples only)")
    implied = sum(len(g) * (len(g) - 1) // 2 for g in groups)
    per = Counter(int(g) for g in of_pair)
    complete = all(per[gi] == len(g) * (len(g) - 1) // 2 for gi, g in enumerate(groups))
    check(implied == len(pairs) and complete and len(set(pairs)) == len(pairs), "a1_pairs_implied",
          f"{tag}: {len(pairs)} retrieval pairs = sum over groups of C(size, 2) = {implied}; every group is "
          f"fully linked, no pair repeated")
    unpaired = n - len(linked)
    check(unpaired == int(d["n_unpaired"]) and len(pairs) == int(d["n_pairs"]), "a1_unpaired",
          f"{tag}: {unpaired} unpaired documents (no link) = the loader's n_unpaired; pairs = loader n_pairs")
    by_combo = Counter(_combo(y[a], y[b]) for a, b in pairs)
    linked_by_class = {c: int(sum(1 for i in linked if y[i] == c)) for c in classes}
    return {"setting": key, "classes": classes, "class_counts": counts, "n_docs": n,
            "n_linked_pairs": sizes.get(2, 0), "n_linked_triples": sizes.get(3, 0), "n_groups": len(groups),
            "n_retrieval_pairs": len(pairs), "pairs_by_combination": dict(sorted(by_combo.items(),
                                                        key=lambda kv: [ORDER[c] for c in kv[0].split("|")])),
            "n_linked_docs": len(linked), "n_unpaired": unpaired, "linked_docs_by_class": linked_by_class,
            "unpaired_by_class": {c: counts[c] - linked_by_class[c] for c in classes},
            "split_unit": ("linked group (pair or triple)" if any(len(g) > 2 for g in groups)
                           else "linked pair"),
            "split_path": ("protocols._make_group_split" if any(len(g) > 2 for g in groups)
                           else "protocols.make_split (disjoint pairs)")}


def _construction(key: str, d: dict, R, pd) -> dict:
    """What the loader did to the raw file(s): counts only, from the loaders' own steps."""
    load = d["config"]["load"]
    if load["dataset"] == "swiltra":
        langs = tuple(load["langs"])
        path = "data/swiltra_bench_decision_summaries.csv"
        raw = pd.read_csv(config.REPO / path)
        rows = len(raw)
        kept_rows = len(raw.dropna(subset=[f"{l}_bgeText" for l in langs]))
        texts = [t for l in langs for t in raw.dropna(subset=[f"{l}_bgeText" for l in langs])[f"{l}_bgeText"].astype(str)]
        return {"files": {path: file_sha256(config.REPO / path)}, "rule": "row-aligned decisions; every two language "
                "versions of a row are linked", "raw_rows": rows, "rows_with_all_languages": kept_rows,
                "texts_before_dedup": len(texts), "exact_duplicates_dropped": len(texts) - int(d["n_docs"]),
                "max_pairs": load.get("max_pairs")}
    if load["dataset"] == "case_summaries":
        path, n_pairs = load["csv"], load["n_pairs"]
        raw = pd.read_csv(config.REPO / path)
        blocks = list(dict.fromkeys(raw["source"].astype(str)))
        block_rows = {b: int((raw["source"] == b).sum()) for b in blocks}
        designated = int(min(n_pairs, *block_rows.values()))
        return {"files": {path: file_sha256(config.REPO / path)},
                "rule": f"row i of each source block pairs with row i of the other for i < {n_pairs}; later rows "
                        f"pooled unpaired", "block_order": blocks, "raw_rows_by_source": block_rows,
                "n_pairs_parameter": n_pairs, "designated_pairs": designated,
                "exact_duplicates_dropped": int(len(raw) - d["n_docs"]),
                "designated_pairs_lost_to_dedup": designated - int(d["n_pairs"])}
    # Three-source union: re-run `load_case_summaries_union` step by step and require equality.
    files = R.SCOTUS_FILES
    docs, src, index, links, per_file = [], [], {}, [], {}
    for path, n_pairs in files:
        texts, source, prs, _ = R.load_case_summaries(str(config.REPO / path), n_pairs=n_pairs)
        ids, new = [], 0
        for t, s in zip(texts, source):
            ids.append(index.setdefault((str(s), t), len(docs)))
            if ids[-1] == len(docs):
                docs.append(t)
                src.append(str(s))
                new += 1
        per_file[path] = {"documents": len(texts), "pairs": len(prs or []), "new_documents": new,
                          "already_seen": len(texts) - new}
        links += [(ids[a], ids[b]) for a, b in (prs or [])]
    src = np.asarray(src)
    appear = Counter()
    for path, n_pairs in files:
        texts, source, _, _ = R.load_case_summaries(str(config.REPO / path), n_pairs=n_pairs)
        for t, s in set(zip(texts, map(str, source))):
            appear[(s, t)] += 1
    shared = Counter(s for (s, t), k in appear.items() if k > 1)
    parent = list(range(len(docs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in links:
        parent[find(b)] = find(a)
    comps: dict = {}
    for i in range(len(docs)):
        comps.setdefault(find(i), []).append(i)
    multi = [m for m in comps.values() if len(m) > 1]
    ambiguous = [m for m in multi if len(set(src[m])) != len(m)]
    amb_docs = {i for m in ambiguous for i in m}
    amb_links = [l for l in links if l[0] in amb_docs]
    kept = [m for m in multi if len(set(src[m])) == len(m)]
    direct = {tuple(sorted(l)) for l in links}
    implied = {tuple(sorted(p)) for m in kept for p in itertools.combinations(m, 2)}
    return {"files": {p: file_sha256(config.REPO / p) for p, _ in files},
            "rule": "document = (source, text); the three files' pair links joined transitively into connected "
                    "components; a component holding two summaries of one source is left unpaired",
            "per_file": per_file, "union_documents": len(docs),
            "summaries_in_two_files_by_source": dict(sorted(shared.items(), key=lambda kv: ORDER[kv[0]])),
            "file_links": len(links), "distinct_links": len(direct),
            "components_with_links": len(multi), "ambiguous_components": len(ambiguous),
            "ambiguous_component_sizes": sorted(len(m) for m in ambiguous),
            "documents_in_ambiguous_components": len(amb_docs), "file_links_in_ambiguous_components": len(amb_links),
            "retrieval_pairs_from_links": len(implied & direct),
            "retrieval_pairs_by_transitivity": len(implied - direct),
            "_rebuilt": (docs, src, kept)}


def _collect() -> list:
    from ..lib.validate import _project_code
    import pandas as pd
    out = []
    with _project_code():
        import corpora
        import protocols as P
        import run_sae_experiment as R
        for corpus, confounder, settings in BLOCKS:
            for key, name in settings:
                d = corpora.load(key)
                comp = _compose(key, d, P)
                cons = _construction(key, d, R, pd)
                if "_rebuilt" in cons:
                    docs, src, kept = cons.pop("_rebuilt")
                    # The loader's output order: source blocks in first appearance, then pairs.
                    blocks = list(dict.fromkeys(src))
                    order = np.concatenate([np.where(src == b)[0] for b in blocks])
                    pos = {int(r): i for i, r in enumerate(order)}
                    rebuilt_pairs = sorted(tuple(sorted((pos[a], pos[b]))) for m in kept
                                           for a, b in itertools.combinations(m, 2))
                    same = ([docs[r] for r in order] == list(d["texts"])
                            and list(src[order]) == list(map(str, d["source"]))
                            and rebuilt_pairs == sorted(tuple(sorted(map(int, p))) for p in d["pairs"]))
                    check(same, "a1_union_rebuilt", f"{key}: the step-by-step re-derivation of the pooled corpus "
                          f"equals corpora.load exactly (documents, sources, pairs)")
                    check(cons["retrieval_pairs_from_links"] + cons["retrieval_pairs_by_transitivity"]
                          == comp["n_retrieval_pairs"], "a1_union_pairs",
                          f"{key}: {cons['retrieval_pairs_from_links']} pairs recorded in a file + "
                          f"{cons['retrieval_pairs_by_transitivity']} by transitivity = "
                          f"{comp['n_retrieval_pairs']} retrieval pairs")
                check(set(comp["classes"]) == set(config.setting(key)["classes"]), "a1_registry_classes",
                      f"{key}: classes {comp['classes']} = manifest")
                pr = processed(key)
                check(pr["n_docs"] == comp["n_docs"] and pr["n_retrieval_pairs"] == comp["n_retrieval_pairs"]
                      and pr["n_unpaired"] == comp["n_unpaired"]
                      and pr["group_sizes"] == {str(k): v for k, v in (("2", comp["n_linked_pairs"]),
                                                                          ("3", comp["n_linked_triples"])) if v},
                      "a1_matches_shared_counts", f"{key}: documents, retrieval pairs, group sizes and unpaired "
                      f"counts equal lib.datasets.processed (used by Tables 1 and 2)")
                expected = corpora._KNOWN_SIZES.get(key)
                check(expected is None or expected == comp["n_docs"], "a1_known_size",
                      f"{key}: {comp['n_docs']} documents = registry size {expected}")
                out.append({"corpus": corpus, "confounder": confounder, "name": name, **comp,
                            "construction": cons})
    return out


# --------------------------------------------------------------------------- #
def _by_pair(r: dict) -> str:
    """Retrieval pairs by class pair of a three-class setting, as caption text."""
    v = r["pairs_by_combination"]
    if len(set(v.values())) == 1:
        return f"{_fmt(next(iter(v.values())))} pairs per {'language' if r['confounder'] == 'language' else 'source'} pair"
    return ", ".join(f"{'--'.join(ABBR.get(c, CLS[c]) for c in k.split('|'))} {_fmt(n)}" for k, n in v.items())


def build(ex) -> None:
    rows = _collect()
    body, csv_rows = [], []
    for bi, (corpus, confounder, settings) in enumerate(BLOCKS):
        block = [x for x in rows if x["corpus"] == corpus]
        classes = sorted({c for r in block for c in r["classes"]}, key=ORDER.get)
        check(len(classes) == 3, "a1_three_class_columns",
              f"{corpus}: its settings draw on exactly three classes ({classes}), one column each")
        if bi:
            body.append(r"\midrule")
        # The block row names the corpus and labels the three class columns for the rows below it.
        body.append([rf"\textbf{{{corpus}}}"] + [rf"\textit{{{HEAD[c]}}}" for c in classes] + ["", "", ""])
        for r in block:
            body.append([SHORT[r["setting"]]] + [_fmt(r["class_counts"][c]) if c in r["class_counts"] else "--"
                                       for c in classes]
                        + [_fmt(r["n_docs"]), _fmt(r["n_retrieval_pairs"]), _fmt(r["n_unpaired"])])
            check(r["n_linked_triples"] > 0 or r["n_retrieval_pairs"] == r["n_linked_pairs"] == r["n_groups"],
                  "a1_binary_groups_are_pairs", f"{r['setting']}: every linked group is one pair, so groups = "
                  f"retrieval pairs ({r['n_retrieval_pairs']}); the caption states this instead of a column")
            csv_rows.append({"corpus": corpus, "setting": r["setting"], "name": r["name"].replace("--", "-"),
                             "confounder": confounder, "classes": r["classes"],
                             **{f"docs_{c}": r["class_counts"][c] for c in ("de", "fr", "it", "oyez", "lexis", "wiki")
                                if c in r["class_counts"]},
                             "n_docs": r["n_docs"], "linked_pairs_2doc_groups": r["n_linked_pairs"],
                             "linked_triples_3doc_groups": r["n_linked_triples"], "linked_groups": r["n_groups"],
                             "retrieval_pairs": r["n_retrieval_pairs"],
                             **{f"retrieval_pairs_{k.replace('|', '_')}": v for k, v in r["pairs_by_combination"].items()},
                             "linked_docs": r["n_linked_docs"], "unpaired_docs": r["n_unpaired"],
                             **{f"unpaired_{c}": v for c, v in r["unpaired_by_class"].items()},
                             "split_unit": r["split_unit"], "split_path": r["split_path"],
                             "data_files": list(r["construction"]["files"]),
                             "data_sha256": list(r["construction"]["files"].values())})
    head = [["Setting", r"\multicolumn{3}{c}{Docs per class}", "Docs", "Pairs", "Unpaired"]]
    # One column wide: the tabular is set in \footnotesize, the caption stays \small like every exhibit's.
    lines = [r"{\footnotesize%"] + latex.tabular("@{}lrrrrrr@{}", head, body)
    lines[-1] += "}"
    sw = next(r for r in rows if r["setting"] == "swiltra_de_fr_it")
    un = next(r for r in rows if r["setting"] == "scotus_oyez_lexis_wiki")
    c = un["construction"]
    pairwise = [r for r in rows if "designated_pairs_lost_to_dedup" in r["construction"]]
    lost = [str(r["construction"]["designated_pairs_lost_to_dedup"]) for r in pairwise]
    bound = {SHORT[r["setting"]]: r["construction"]["n_pairs_parameter"] for r in pairwise}
    usual = Counter(bound.values()).most_common(1)[0][0]
    odd = "; ".join(f"{_fmt(v)} for {k}" for k, v in bound.items() if v != usual)
    caption = (
        r"Setting composition after exact-duplicate removal. SCOTUS sources: Oyez (O), LexisNexis (L), "
        r"Wikipedia (W); -- marks an absent class. \emph{Pairs}: linked document pairs (held-out retrieval "
        r"targets); \emph{Unpaired}: documents without a recorded link, used for fitting and as distractors. "
        r"Linked groups, never split between training and evaluation, are single pairs in the binary settings, "
        + _fmt(sw["n_linked_triples"]) + r" triples (one per decision) in DE--FR--IT, and "
        + _fmt(un["n_linked_pairs"]) + r" pairs plus " + _fmt(un["n_linked_triples"]) + r" triples in O--L--W ("
        + _by_pair(un) + r" pairs). SCOTUS files have no case IDs: as in \citet{fan2025medium}, row $i$ of one "
        r"source block is linked to row $i$ of the other for $i<" + _fmt(usual) + r"$ (" + odd + r"), and "
        r"deduplication breaks " + ", ".join(lost[:-1]) + r" and " + lost[-1] + r" of these links. O--L--W merges "
        r"the three files on (source, text) and joins links transitively (" + _fmt(c["retrieval_pairs_by_transitivity"])
        + r" pairs); " + str(c["ambiguous_components"]) + r" components with two summaries of one source ("
        + str(c["documents_in_ambiguous_components"]) + r" documents) stay unpaired.")
    latex.write_table(NAME, lines, caption, LABEL, ex, wide=False, size=r"\small", tabcolsep="3.5pt")
    latex.write_csv(NAME, csv_rows, ex)
    _STATE.update(rows=rows)


def _amb(rows) -> dict:
    return next(r for r in rows if r["setting"] == "scotus_oyez_lexis_wiki")["construction"]


def finalize(ex) -> None:
    rows = _STATE["rows"]
    L = [f"# Table A1 ({NAME}): provenance and validation", "",
         f"Build: `python -m paper_assets build {NAME}`. Counts come from `corpora.load` (the registry loaders) and "
         "`protocols.link_groups`; nothing is embedded, merged or relinked.", "", "## Data files", ""]
    files = {}
    for r in rows:
        files.update(r["construction"]["files"])
    L += [f"- `{p}` sha256 `{h}`" for p, h in sorted(files.items())]
    L += ["", "## Construction per setting", ""]
    for r in rows:
        c = {k: v for k, v in r["construction"].items() if k != "files"}
        L.append(f"### {r['setting']} ({r['name'].replace('--', '–')})")
        L.append(f"- composition: classes {r['class_counts']}; {r['n_docs']} documents; {r['n_linked_pairs']} "
                 f"linked pairs, {r['n_linked_triples']} linked triples; {r['n_retrieval_pairs']} retrieval pairs "
                 f"{r['pairs_by_combination']}; {r['n_unpaired']} unpaired {r['unpaired_by_class']}")
        L.append(f"- split unit: {r['split_unit']} via `{r['split_path']}`")
        for k, v in c.items():
            L.append(f"- {k}: {v}")
        L.append("")
    L += ["## Missing or unverified provenance (flagged, not asserted)", "",
          "- The three SCOTUS CSVs and the SwiLTra CSV entered the repository as uploads (git: 'Add files via "
          "upload', 2025-07-16). No download or build script, source revision or upstream identifier is recorded.",
          "- The SCOTUS CSVs hold only `case_summaries` and `source`: no case identifier. The pairing boundary "
          "(1,024; 780 for Oyez–Wikipedia) is the parent paper's construction. The loader docstring reports a "
          "TF-IDF check that recovers the row alignment for 95% of the rows before the boundary; that check is not "
          "re-run here.",
          "- The loader docstring also reports that the unpaired SCOTUS remainder still contains many cross-source "
          "matches (TF-IDF mutual best matches). Such documents are unpaired distractors, and under the document "
          "holdout a counterpart can sit on the other side of the split (audit_2026-09-28/outputs/"
          "C1_split_overlap.json).",
          "- SwiLTra: the columns are `de_bgeText`, `fr_bgeText`, `it_bgeText`. Which upstream SwiLTra-Bench field "
          "they are, and any filtering before upload, is not recorded.",
          "- Ambiguous SCOTUS components: the `load_case_summaries_union` docstring says 4 components and the "
          "pipeline's unresolved list says '12 ambiguous file pairs'. Both hold, as measured here: "
          f"{_amb(rows)['ambiguous_components']} components ({_amb(rows)['documents_in_ambiguous_components']} documents) "
          f"containing {_amb(rows)['file_links_in_ambiguous_components']} file links.",
          "", f"## Checks ({len(ex.checks)} passed)", ""] + [f"- {c['message']}" for c in ex.checks]
    L += ["", "## Issues", ""] + ([f"- **{i['level']}** `{i['code']}`: {i['message']}" for i in ex.issues] or ["None."])
    p = config.OUT_DIRS["reports"] / f"{NAME}_provenance.md"
    p.write_text("\n".join(L) + "\n")
    ex.output(p)
