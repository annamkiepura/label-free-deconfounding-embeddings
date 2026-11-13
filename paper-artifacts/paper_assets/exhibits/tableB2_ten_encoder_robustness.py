"""Appendix Table B2: robustness across ten encoders (the original refitting study).

Only the original ten-encoder experiment (`revised.run_encoder_cell`, files
encoders/encoder_{setting}_{encoder}.json, plus `alternatives.run_generous_k` on the same
splits for the silhouette k-means arm): cohort `encoder_study`, seeds 0-2, on the ORIGINAL
dataset versions -- SemEval (4 languages), SwiLTra DE-FR, Bills-Tweets, SCOTUS Oyez-Lexis,
CAP (3 sources). Every eraser is refit in each encoder's own space; nothing here is
frozen transfer, and no three-class composition enters.

Panel A: per encoder x corpus, the held-out linear-AUC gap to Metadata-LEACE of PC-1 /
k-means LEACE / SAE-LEACE, each formed within matched seeds and then averaged.
Panel B: per encoder x corpus, the semantic-utility gap to Metadata-LEACE of the same three
methods, formed and averaged the same way: R@10 of the paired retrieval (SemEval, SwiLTra,
SCOTUS) and CAP topic purity, the manuscript's primary CAP utility (manifest `utility`).
Bills-Tweets has no utility metric. CAP's change relative to Raw (purity and ARI, all four
methods) is exported in the CSVs, and the caption states how many encoders gain purity.
"""
from __future__ import annotations

import ast
import re

import numpy as np

from ..lib import config, latex
from ..lib.aggregate import paired_difference
from ..lib.datasets import processed
from ..lib.io import read
from ..lib.loaders import Cohort, validate_file
from ..lib.methods import ENCODERS
from ..lib.provenance import check, issue

NAME, KIND, TITLE = "tableB2_ten_encoder_robustness", "table", "Robustness across ten encoders"
LABEL = "tab:ten-encoder-robustness"
COHORT = "encoder_study"
#: The original corpora of the ten-encoder study that the paper's table shows (Bills-Tweets, which has no utility
#: measure, is not in it).
CORPORA = [("semeval", "SemEval"), ("swiltra", "SwiLTra"), ("scotus_oyez_lexis", "SCOTUS"), ("cap", "CAP")]
#: The dataset version each column must be (the original study's), by class set.
VERSION = {"semeval": ["de", "en", "es", "zh"], "swiltra": ["de", "fr"], "bills_tweets": ["bill", "tweet"],
           "scotus_oyez_lexis": ["lexis", "oyez"], "cap": ["bill", "newspaper", "order"]}
GAPS = ("pca_top1", "kmeans_auto", "sae_leace")
REF = "metadata_leace"
B_METHODS = ("pca_top1", "kmeans_auto", "metadata_leace", "sae_leace")
B_METRIC, B_ALT = "topic_purity", "topic_ari"          # CAP vs Raw: caption counts / exported alongside
#: Panel B columns: (setting, utility metric, header). The setting's own utility metric (manifest), so
#: Bills-Tweets (none) has no column.
UTIL = [("semeval", "recall@10", "SemEval R@10"), ("swiltra", "recall@10", "SwiLTra R@10"),
        ("scotus_oyez_lexis", "recall@10", "SCOTUS R@10"), ("cap", "topic_purity", "CAP purity")]
#: Short names as in the manuscript's encoder table (Table A2 / draft Table 5).
SHORT = {**ENCODERS, "mxb": "MXB"}
#: Visual grouping only (model families), not a statistical grouping.
FAMILY_BREAK_BEFORE = ("gist-small", "e5-small", "mpnet")
SPLIT_GROUPS = [CORPORA[:3], CORPORA[3:]]
DRAFT = config.REPO / "audit_2026-09-28" / "manuscript_v23.txt"
_STATE: dict = {}


def _registry() -> tuple[tuple, dict]:
    """Encoder order (experiments.PARENT_PAPER_ENCODERS) and model ids (MODEL_MAP), read
    from the source without importing the encoding stack."""
    def const(path, name):
        tree = ast.parse((config.REPO / "code" / path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
                return ast.literal_eval(node.value)
        raise KeyError(name)
    return const("experiments.py", "PARENT_PAPER_ENCODERS"), const("style_erasure_pipeline.py", "MODEL_MAP")


def _triplet(svs) -> str:
    return r"\,/\,".join(latex.num(v.mean, 3, signed=True) if v.ok else "---" for v in svs)


def _collect():
    order, models = _registry()
    check(len(order) == 10 and all(e in SHORT for e in order), "b2_registry",
          f"ten encoders from experiments.PARENT_PAPER_ENCODERS: {list(order)}")
    cells, absol, bcells, ucells, files = {}, {}, {}, {}, {}
    for enc in order:
        for key, _ in CORPORA:
            coh = Cohort(COHORT, key, enc)
            coh.compatibility()          # main vs generous-K: config, splits, shared arms identical
            for role, rel in coh.paths.items():
                d = validate_file(rel)
                files[rel] = d.get("experiment")
                check(d.get("experiment") in ("revised_encoder_cell", "generous_k") and not rel.startswith("ood"),
                      "b2_refit_study", f"{rel}: experiment `{d.get('experiment')}` (per-encoder refit), "
                      f"not frozen transfer")
                check(d.get("encoder") == enc and d.get("corpus") == key, "b2_file_identity",
                      f"{rel}: encoder {d.get('encoder')}, corpus {d.get('corpus')}")
                cls = sorted(d["corpus_stats"]["classes"])
                check(cls == VERSION[key], "b2_original_version",
                      f"{rel}: classes {cls} = original {config.setting(key)['long']} version")
            main = coh.file("main")
            p = processed(key)
            cs = main["corpus_stats"]
            check(cs["n_docs"] == p["n_docs"] and dict(zip(cs["classes"], cs["class_counts"]))
                  == dict(zip(p["classes"], p["class_counts"])), "b2_dataset_version",
                  f"{coh.paths['main']}: corpus statistics equal the processed definition of `{key}`")
            check(main["config"]["recipe"]["top_features"] == 32 and main["config"]["selection"] == "topn",
                  "b2_sae_config", f"{coh.paths['main']}: SAE-LEACE is MI top-32")
            gk = coh.file("gk")
            diag = [dg for s in gk["per_seed"] for dg in s["diagnostics"]]
            check(all(dg["k_auto"] == int(max(dg["silhouette"], key=dg["silhouette"].get)) for dg in diag),
                  "b2_kmeans_silhouette", f"{coh.paths['gk']}: kmeans_auto uses the training-silhouette K "
                  f"(K = {sorted({dg['k_auto'] for dg in diag})})")
            ref = coh.values(REF, "linear_auc")
            absol[(enc, key, REF)] = ref
            absol[(enc, key, "raw")] = coh.values("raw", "linear_auc")
            for m in GAPS:
                v = coh.values(m, "linear_auc")
                absol[(enc, key, m)] = v
                if v.ok and ref.ok:
                    check(v.seeds == ref.seeds, "b2_matched_seeds", f"{enc}/{key} {m}: seeds {v.seeds} "
                          f"matched with Metadata-LEACE")
                cells[(enc, key, m)] = (paired_difference(v, ref, method=f"{m}-{REF}") if v.ok and ref.ok
                                        else v)
                if not v.ok:
                    issue("missing", "b2_not_run", f"{enc}/{key} {m}: {v.reason}")
            for ukey, metric, _ in UTIL:
                if ukey != key:
                    continue
                check(config.setting(key).get("utility") == metric, "b2_utility_metric",
                      f"{key}: Panel B uses the setting's utility metric ({metric})")
                uref = coh.values(REF, metric)
                absol[(enc, key, REF, metric)] = uref
                for m in GAPS:
                    v = coh.values(m, metric)
                    absol[(enc, key, m, metric)] = v
                    ucells[(enc, key, m)] = (paired_difference(v, uref, method=f"{m}-{REF}")
                                             if v.ok and uref.ok else v)
                    if not v.ok:
                        issue("missing", "b2_not_run", f"{enc}/{key} {m} {metric}: {v.reason}")
            if key == "cap":
                for metric in (B_METRIC, B_ALT):
                    raw = coh.values("raw", metric)
                    absol[(enc, key, "raw", metric)] = raw
                    for m in B_METHODS:
                        v = coh.values(m, metric)
                        absol[(enc, key, m, metric)] = v
                        bcells[(enc, m, metric)] = (paired_difference(v, raw, method=f"{m}-raw")
                                                    if v.ok and raw.ok else v)
    # CAP protocol identical across encoders: audit definition and the per-seed split.
    caps = [Cohort(COHORT, "cap", e).file("main") for e in order]
    check(all(c["config"]["audit"] == caps[0]["config"]["audit"] for c in caps), "b2_cap_protocol",
          f"CAP: audit protocol (21 topic clusters, k-means n_init {caps[0]['config']['audit']['kmeans_n_init']}) "
          f"identical across the ten encoders")
    check(all([p["split"] for p in c["per_seed"]] == [p["split"] for p in caps[0]["per_seed"]] for c in caps),
          "b2_cap_split", "CAP: per-seed held-out split identical across the ten encoders")
    return order, models, cells, absol, bcells, ucells, files


def _draft_table11() -> dict:
    """Absolute linear AUCs of the draft's ten-encoder Table 11 (Raw, Metadata-LEACE,
    SAE-LEACE), for a sanity comparison only."""
    if not DRAFT.exists():
        return {}
    lines = DRAFT.read_text().splitlines()
    try:
        i = next(k for k, l in enumerate(lines) if l.startswith("Table 11: Absolute linear AUC"))
    except StopIteration:
        return {}
    out, block = {}, None
    names = {v: k for k, v in SHORT.items()}
    for l in lines[i + 1:i + 45]:
        if l.strip() in ("Raw", "Metadata-LEACE", "SAE-LEACE"):
            block = {"Raw": "raw", "Metadata-LEACE": REF, "SAE-LEACE": "sae_leace"}[l.strip()]
            continue
        m = re.match(r"^(\S+) (.*)$", l)
        if block and m and m.group(1) in names:
            vals = re.findall(r"(\d\.\d{3})\s*±\s*(\d\.\d{3})", m.group(2))
            for (key, _), (mu, sd) in zip(CORPORA, vals):
                out[(names[m.group(1)], key, block)] = (float(mu), float(sd))
    return out


def _panel_a(order, cells, corpora, summary_rows=True, rows=("mean", "range")) -> list:
    body = []
    for enc in order:
        if enc in FAMILY_BREAK_BEFORE:
            body.append(r"\addlinespace[2pt]")
        body.append([SHORT[enc]] + [_triplet([cells[(enc, k, m)] for m in GAPS]) for k, _ in corpora])
    if summary_rows:
        body.append(r"\midrule")
        mean_row, range_row = ["Mean"], [r"Range (max$-$min)"]
        for k, _ in corpora:
            mu, rg = [], []
            for m in GAPS:
                v = [cells[(e, k, m)].mean for e in order if cells[(e, k, m)].ok]
                ok = len(v) == len(order)
                mu.append(latex.num(float(np.mean(v)), 3, signed=True) if ok else "---")
                rg.append(latex.num(float(np.max(v) - np.min(v)), 3) if ok else "---")
            mean_row.append(r"\,/\,".join(mu))
            range_row.append(r"\,/\,".join(rg))
        body += [r for r, k in ((mean_row, "mean"), (range_row, "range")) if k in rows]
    return body


def _panel_a_tabular(order, cells, corpora, tag="") -> list:
    head = [["Encoder"] + [rf"\multicolumn{{1}}{{c}}{{{lab}}}" for _, lab in corpora]]
    return latex.tabular("l" + "c" * len(corpora), head, _panel_a(order, cells, corpora))


def _panel_b_tabular(order, ucells) -> list:
    """Utility gaps to Metadata-LEACE, laid out exactly as Panel A (one triplet per corpus)."""
    cols = [(k, lab) for k, _, lab in UTIL]
    head = [["Encoder"] + [rf"\multicolumn{{1}}{{c}}{{{lab}}}" for _, lab in cols]]
    return latex.tabular("l" + "c" * len(cols), head, _panel_a(order, ucells, cols, rows=("mean",)))


def _width(lines, size=r"\small", sep="4pt") -> float | None:
    m = latex.measure(r"\sbox0{" + size + rf"\setlength{{\tabcolsep}}{{{sep}}}" + "\n".join(lines)
                      + r"}\typeout{MEASURE wd=\the\wd0}\typeout{MEASURE tw=\the\textwidth}\mbox{}")
    return (float(m["wd"].rstrip("pt")), float(m["tw"].rstrip("pt"))) if m else None


def build(ex) -> None:
    order, models, cells, absol, bcells, ucells, files = _collect()
    # ---- Layout: one Panel A at \small if it fits the text width, else two stacked subpanels.
    single = _panel_a_tabular(order, cells, CORPORA)
    w = _width(single)
    split = w is not None and w[0] > w[1]
    title_a = (r"\textbf{Panel A.} Linear AUC above Metadata-LEACE "
               r"(PC-1\,/\,$k$-means LEACE\,/\,SAE-LEACE; lower is better)")
    lines = [r"\par{" + title_a + r"}\par\smallskip"]
    if split:
        for gi, grp in enumerate(SPLIT_GROUPS):
            if gi:
                lines.append(r"\par\medskip")
            lines += _panel_a_tabular(order, cells, grp)
    else:
        lines += single
    lines += [r"\par\medskip",
              r"\par{\textbf{Panel B.} Semantic utility relative to Metadata-LEACE "
              r"(PC-1\,/\,$k$-means LEACE\,/\,SAE-LEACE; higher is better)}\par\smallskip"]
    lines += _panel_b_tabular(order, ucells)
    above = {m: (sum(bcells[(e, m, B_METRIC)].mean > 0 for e in order if bcells[(e, m, B_METRIC)].ok),
                 sum(bcells[(e, m, B_METRIC)].ok for e in order)) for m in B_METHODS}
    check(all(n == len(order) for _, n in above.values()), "b2_cap_vs_raw_complete",
          f"CAP purity relative to Raw available for every encoder and method ({above})")
    lab = {"pca_top1": "PC-1", "kmeans_auto": r"$k$-means LEACE", "metadata_leace": "Metadata-LEACE",
           "sae_leace": "SAE-LEACE"}
    groups = {}
    for m in B_METHODS:
        groups.setdefault(above[m][0], []).append(lab[m])
    join = lambda ms: " and ".join(ms) if len(ms) < 3 else ", ".join(ms[:-1]) + " and " + ms[-1]
    raw_txt = ", and ".join(f"in {k}/{len(order)} encoders for {join(ms)}"
                            for k, ms in sorted(groups.items(), reverse=True))
    caption = (
        r"Robustness across ten encoders in the original refitting study: every eraser is refit separately "
        r"in each encoder's embedding space, so these are robustness results, not frozen transfer. Corpora "
        r"are the study's original versions: SemEval (four languages), SwiLTra DE--FR, Bills--Tweets, SCOTUS "
        r"Oyez--Lexis and CAP (three sources); no three-class SwiLTra or SCOTUS composition enters. Each cell "
        r"is the gap to Metadata-LEACE of PC-1\,/\,$k$-means LEACE (training-silhouette $K$)\,/\,SAE-LEACE "
        r"($N{=}32$); 0 matches the supervised reference. Panel A: held-out linear AUC. Panel B: semantic "
        r"utility, R@10 of the paired retrieval (SemEval, SwiLTra, SCOTUS) or CAP topic purity; Bills--Tweets "
        r"has none. Relative to Raw, CAP topic purity rises " + raw_txt + r". Differences are "
        r"formed within each matched seed, then averaged over seeds 0--2; per-seed values, SDs and absolute "
        r"values are in the data files.")
    latex.write_table(NAME, lines, caption, LABEL, ex, wide=True, size=r"\small", tabcolsep="4pt")
    # ---- CSVs.
    per_seed, summary = [], []
    for (enc, key, m), d in cells.items():
        v, ref = absol[(enc, key, m)], absol[(enc, key, REF)]
        summary.append({"panel": "A", "encoder": enc, "encoder_label": SHORT[enc], "corpus": key, "method": m,
                        "quantity": "linear_auc_minus_metadata_leace", "status": d.status, "seeds": d.seeds,
                        "mean_gap": d.mean if d.ok else "", "sd_gap": d.sd if d.ok else "", "sd_ddof": 1,
                        "method_auc_mean": v.mean if v.ok else "", "method_auc_sd": v.sd if v.ok else "",
                        "metadata_auc_mean": ref.mean if ref.ok else "", "metadata_auc_sd": ref.sd if ref.ok else "",
                        "raw_auc_mean": absol[(enc, key, "raw")].mean, "source_files": [s["file"] for s in v.sources],
                        "run_ids": [s["run_id"] for s in v.sources], "reason": d.reason})
        if d.ok:
            for s, g, a, b in zip(d.seeds, d.values, v.values, ref.values):
                per_seed.append({"panel": "A", "encoder": enc, "corpus": key, "method": m, "seed": s,
                                 "method_linear_auc": a, "metadata_leace_linear_auc": b, "gap": g,
                                 "source_file": v.sources[0]["file"], "reference_file": ref.sources[0]["file"],
                                 "run_id": v.sources[0]["run_id"]})
    for (enc, key, m), d in ucells.items():
        metric = next(mt for k, mt, _ in UTIL if k == key)
        v, ref = absol[(enc, key, m, metric)], absol[(enc, key, REF, metric)]
        summary.append({"panel": "B", "encoder": enc, "encoder_label": SHORT[enc], "corpus": key, "method": m,
                        "quantity": f"{metric}_minus_metadata_leace", "status": d.status, "seeds": d.seeds,
                        "mean_gap": d.mean if d.ok else "", "sd_gap": d.sd if d.ok else "", "sd_ddof": 1,
                        "method_value_mean": v.mean if v.ok else "", "metadata_value_mean": ref.mean if ref.ok else "",
                        "source_files": [s["file"] for s in v.sources], "run_ids": [s["run_id"] for s in v.sources],
                        "reason": d.reason})
        if d.ok:
            for s, g, a, b in zip(d.seeds, d.values, v.values, ref.values):
                per_seed.append({"panel": "B", "encoder": enc, "corpus": key, "method": m, "metric": metric,
                                 "seed": s, "method_value": a, "metadata_leace_value": b, "gap": g,
                                 "source_file": v.sources[0]["file"], "reference_file": ref.sources[0]["file"],
                                 "run_id": v.sources[0]["run_id"]})
    for (enc, m, metric), d in bcells.items():
        v, raw = absol[(enc, "cap", m, metric)], absol[(enc, "cap", "raw", metric)]
        summary.append({"panel": "cap_vs_raw" if metric == B_METRIC else "cap_vs_raw_ari", "encoder": enc,
                        "encoder_label": SHORT[enc], "corpus": "cap", "method": m,
                        "quantity": f"{metric}_minus_raw", "printed": False, "in_caption": metric == B_METRIC,
                        "status": d.status,
                        "seeds": d.seeds, "mean_gap": d.mean if d.ok else "", "sd_gap": d.sd if d.ok else "",
                        "sd_ddof": 1, "method_value_mean": v.mean if v.ok else "", "raw_value_mean": raw.mean,
                        "source_files": [s["file"] for s in v.sources], "run_ids": [s["run_id"] for s in v.sources],
                        "reason": d.reason})
        if d.ok:
            for s, g, a, b in zip(d.seeds, d.values, v.values, raw.values):
                per_seed.append({"panel": "cap_vs_raw" if metric == B_METRIC else "cap_vs_raw_ari", "encoder": enc,
                                 "corpus": "cap",
                                 "method": m, "metric": metric, "seed": s, "method_value": a, "raw_value": b,
                                 "gap": g, "source_file": v.sources[0]["file"],
                                 "reference_file": raw.sources[0]["file"], "run_id": v.sources[0]["run_id"]})
    latex.write_csv(f"{NAME}_per_seed", per_seed, ex)
    latex.write_csv(f"{NAME}_summary", summary, ex)
    _STATE.update(order=order, models=models, cells=cells, absol=absol, bcells=bcells, ucells=ucells, files=files,
                  split=split, width=w, lines=lines, caption=caption)


# --------------------------------------------------------------------------- #
def _sanity(absol, order) -> dict:
    """Absolute AUCs vs (i) the draft's Table 11, (ii) the legacy ten-encoder CSV."""
    draft = _draft_table11()
    res = {"draft_n": 0, "draft_match": 0, "draft_diff": [], "legacy_n": 0, "legacy_match": 0, "legacy_diff": []}
    for (enc, key, m), (mu, sd) in draft.items():
        v = absol.get((enc, key, m))
        if v is None or not v.ok:
            continue
        res["draft_n"] += 1
        if abs(round(v.mean, 3) - mu) < 1e-9 and abs(round(v.sd, 3) - sd) < 1e-9:
            res["draft_match"] += 1
        else:
            res["draft_diff"].append((enc, key, m, (mu, sd), (round(v.mean, 4), round(v.sd, 4))))
    p = config.OUT_DIRS["data"] / "tab_encoders_legacy.csv"
    if p.exists():
        import csv
        for r in csv.DictReader(p.open()):
            if r["kind"] != "value" or r["status"] != "ok" or r["metric"] != "linear_auc":
                continue
            v = absol.get((r["encoder"], r["setting"], r["method"]))
            if v is None:
                continue
            res["legacy_n"] += 1
            theirs = [float(x) for x in r["values"].split("|")]
            if theirs == v.values:
                res["legacy_match"] += 1
            else:
                res["legacy_diff"].append((r["encoder"], r["setting"], r["method"]))
    return res


def _layout(pdf) -> dict:
    import subprocess
    pages = subprocess.run(["gs", "-q", "-dNODISPLAY", "-dNOSAFER", "-c",
                            f"({pdf}) (r) file runpdfbegin pdfpagecount = quit"],
                           capture_output=True, text=True).stdout.strip()
    body = (r"\setbox0\vbox{\hsize\textwidth\small\setlength{\tabcolsep}{4pt}\centering "
            + "\n".join(_STATE["lines"]) + r"\par\vskip\abovecaptionskip\noindent Table B2: "
            + _STATE["caption"] + r"\par}\typeout{MEASURE ht=\the\dimexpr\ht0+\dp0\relax}"
            r"\typeout{MEASURE th=\the\textheight}\mbox{}")
    m = latex.measure(body)
    out = {"pages": int(pages) if pages.isdigit() else None}
    if m:
        out.update(ht=float(m["ht"].rstrip("pt")), th=float(m["th"].rstrip("pt")))
    return out


def finalize(ex) -> None:
    S = _STATE
    order, cells, bcells, absol = S["order"], S["cells"], S["bcells"], S["absol"]
    san = _sanity(absol, order)
    check(san["legacy_match"] == san["legacy_n"] and san["legacy_n"] > 0, "b2_matches_legacy_csv",
          f"{san['legacy_match']}/{san['legacy_n']} absolute AUC cells equal tab_encoders_legacy seed by seed")
    if san["draft_diff"]:
        issue("warning", "b2_draft_table11_differs", f"{len(san['draft_diff'])}/{san['draft_n']} absolute AUCs "
              f"differ from the draft Table 11 at 3 decimals (listed in the validation report)")
    else:
        check(san["draft_n"] > 0, "b2_matches_draft_table11",
              f"{san['draft_match']}/{san['draft_n']} Raw / Metadata-LEACE / SAE-LEACE absolute AUCs (mean and SD, "
              f"3 decimals) equal the draft's ten-encoder Table 11")
    pdf = config.OUT_DIRS["reports"] / "previews" / f"{NAME}.pdf"
    lay = _layout(pdf) if pdf.exists() else {}
    if lay.get("pages") is not None:
        check(lay["pages"] == 1, "b2_one_page", f"compiled preview has {lay['pages']} page(s)")
    if "ht" in lay:
        check(lay["ht"] <= lay["th"], "b2_fits_height", f"table + caption {lay['ht']:.0f}pt of the "
              f"{lay['th']:.0f}pt text height")
    ucells = S["ucells"]
    missing = [(k, v.reason) for k, v in {**cells, **bcells, **{("B",) + k: v for k, v in ucells.items()}}.items()
               if not v.ok]
    # ---- provenance report.
    P = [f"Table B2 ({NAME}) -- provenance", "=" * 60, "",
         f"Build: python -m paper_assets build {NAME}", "Reporting only: saved per-seed metrics; no model refit.", "",
         "Experiment: the original ten-encoder refitting study (cohort `encoder_study`, seeds 0-2).",
         "  main arm   : encoders/encoder_{setting}_{encoder}.json  (experiment revised_encoder_cell;",
         "               Raw, PC-1, Metadata-LEACE, SAE-LEACE)",
         "  k-means arm: alternatives/generous_k_{setting}_{encoder}.json  (experiment generous_k; kmeans_auto,",
         "               the training-silhouette K; same splits, shared arms identical seed by seed)", "",
         "Encoders (order = code/experiments.py PARENT_PAPER_ENCODERS; ids = style_erasure_pipeline.MODEL_MAP):"]
    P += [f"  {i + 1:2d}. {SHORT[e]:11s} {e:11s} {S['models'].get(e, '?')}" for i, e in enumerate(order)]
    P += ["", "Label note: the project-wide label for `mxb` is 'MXB-large'; this table prints 'MXB', the short",
          "name of the manuscript's encoder table (draft Table 5) and the requested row list.", "",
          "Dataset versions (classes): " + "; ".join(f"{k} {VERSION[k]}" for k, _ in CORPORA), "",
          "Panel B metrics: each setting's utility (manifest settings.<key>.utility): R@10 for SemEval, SwiLTra",
          "and SCOTUS (main arm: encoders/*; k-means arm: generous_k/*), CAP topic purity (manuscript primary CAP",
          "utility; plan 3 Sec. 3.1 'purity ... report ARI alongside'). Bills-Tweets has none. CAP relative to Raw",
          "(purity, ARI) is exported as panels cap_vs_raw / cap_vs_raw_ari and summarized in the caption.", "",
          f"Source files ({len(S['files'])}; run ID = path@sha256[:12] in the summary CSV):"]
    P += [f"  {rel}  [{exp}]" for rel, exp in sorted(S["files"].items())]
    p = config.OUT_DIRS["reports"] / f"{NAME}_provenance.txt"
    p.write_text("\n".join(P) + "\n")
    ex.output(p)
    # ---- validation report.
    V = [f"Table B2 ({NAME}) -- validation", "=" * 60, "",
         "1. Every value comes from the original ten-encoder experiment: "
         f"{all(e in ('revised_encoder_cell', 'generous_k') for e in S['files'].values())} "
         f"({len(S['files'])} files; experiments {sorted(set(S['files'].values()))}).",
         "2. SwiLTra is DE-FR and SCOTUS is Oyez-Lexis: checked from every file's corpus classes and against",
         "   the processed dataset definitions (document and class counts).",
         "3. Matched seeds/splits: every gap is a per-seed paired difference (seeds 0-2); the main and",
         "   generous-K files share config, per-seed split summaries and identical Raw/PC-1/Metadata-LEACE",
         "   values (Cohort.compatibility).",
         "4. CAP protocol: identical audit definition (21 topic clusters) and identical per-seed held-out",
         "   split across the ten encoders.",
         "5. No three-class results: no file of the three-class compositions is read (classes checked).",
         "6. No frozen-transfer results: no ood/ file is read; experiments are refits.",
         f"7. Absolute-AUC sanity check: {san['legacy_match']}/{san['legacy_n']} cells equal the legacy "
         f"ten-encoder CSV seed by seed; {san['draft_match']}/{san['draft_n']} Raw/Metadata-LEACE/SAE-LEACE "
         f"means and SDs equal the draft's Table 11 at three decimals."]
    V += [f"   differs from draft: {e}/{k} {m}: draft {a}, saved {b}" for e, k, m, a, b in san["draft_diff"]]
    V += ["", "Missing / not run:"] + ([f"  {k}: {r}" for k, r in missing] or ["  none (no '---' cell)."])
    V += ["", "Layout:",
          f"  Panel A single-row width at \\small: {S['width'][0]:.1f}pt vs text width {S['width'][1]:.1f}pt "
          f"-> {'split into two stacked subpanels (SemEval/SwiLTra/Bills-Tweets; SCOTUS/CAP)' if S['split'] else 'one panel'}."
          if S["width"] else "  width not measured"]
    if "ht" in lay:
        V.append(f"  table + caption {lay['ht']:.1f}pt of {lay['th']:.1f}pt text height "
                 f"({100 * lay['ht'] / lay['th']:.0f}%); preview pages: {lay.get('pages')}; body \\small, no color, "
                 f"no bold extrema.")
    V += ["", "Panel A cell means (gap to Metadata-LEACE; PC-1 / k-means / SAE):"]
    for enc in order:
        V.append(f"  {SHORT[enc]:11s} " + "  ".join(
            f"{k}: " + "/".join(f"{cells[(enc, k, m)].mean:+.3f}" if cells[(enc, k, m)].ok else "---" for m in GAPS)
            for k, _ in CORPORA))
    V += ["", "Panel B cell means (utility gap to Metadata-LEACE; PC-1 / k-means / SAE):"]
    for enc in order:
        V.append(f"  {SHORT[enc]:11s} " + "  ".join(
            f"{k}: " + "/".join(f"{ucells[(enc, k, m)].mean:+.3f}" if ucells[(enc, k, m)].ok else "---" for m in GAPS)
            for k, _, _ in UTIL))
    V += ["", "CAP relative to Raw (Delta purity | Delta ARI; PC-1, k-means, Metadata-LEACE, SAE-LEACE):"]
    for enc in order:
        V.append(f"  {SHORT[enc]:11s} " + "  ".join(
            f"{bcells[(enc, m, B_METRIC)].mean:+.3f}|{bcells[(enc, m, B_ALT)].mean:+.3f}" for m in B_METHODS))
    V += ["", f"Checks passed: {len(ex.checks)}", "Issues:"]
    V += [f"  [{i['level']}] {i['code']}: {i['message']}" for i in ex.issues] or ["  none"]
    p = config.OUT_DIRS["reports"] / f"{NAME}_validation.txt"
    p.write_text("\n".join(V) + "\n")
    ex.output(p)
