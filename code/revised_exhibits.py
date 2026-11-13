"""The tables and figures of an earlier version of the paper, from raw JSON (some runners call its helpers).

`build_all()` reads only ``<root>/raw/`` -- the JSONs `revised` writes -- and
writes

    <root>/tables/main/        New Tables 1-3                   (.tex + .md)
    <root>/tables/appendix/    Tables A1-A4, B1-B6, C1-C4, D1-D15 (.tex + .md)
    <root>/figures/main/       New Figures 1-2                  (.png + .pdf)
    <root>/figures/appendix/   Figures A1, C1, D1-D8            (.png + .pdf)
    <root>/README.md           index: exhibit -> files -> raw sources

Restyling or recaptioning an exhibit is a rebuild, never a rerun. An exhibit
whose raw inputs are missing is skipped and listed as such in the index; a
missing cell inside a built exhibit prints as NR ("not run"), never as an
estimate. Numbering follows the plan (``deconfounding_paper_revision_plan_3.md``,
Secs. 6-7), including its optional appendix items.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import re
import traceback

import numpy as np

import artifacts
import audit
import experiments as E
import ood
import protocols as P
import revised as R

MAIN, APPX = "main", "appendix"
COND = R.ID_CONDITIONS
LABEL = R.LABELS
CORPUS = {"semeval": "SemEval", "swiltra": "SwiLTra", "bills_tweets": "Bills–Tweets",
          "scotus_oyez_lexis": "SCOTUS", "cap": "CAP"}
ENCODER = {"mini": "MiniLM", "gist-small": "GIST-small", "gist-base": "GIST-base",
           "gist-large": "GIST-large", "e5-small": "E5-small", "e5-base": "E5-base",
           "e5-large": "E5-large", "mpnet": "MPNet", "nomic-v2": "Nomic-v2", "mxb": "MXB"}
CLASS = {"lexis": "Lexis", "oyez": "Oyez", "bill": "bills", "order": "orders",
         "newspaper": "news", "tweet": "tweets"}
CAP_CLASS = {"bill": "Bills", "order": "Executive orders", "newspaper": "Newspapers"}
#: Display order of a corpus's classes where the plan fixes one.
CLASS_ORDER = {"scotus_oyez_lexis": ("oyez", "lexis"), "cap": ("bill", "order", "newspaper")}


def ordered(key: str, classes) -> list:
    first = [c for c in CLASS_ORDER.get(key, ()) if c in classes]
    return first + [c for c in classes if c not in first]
MECH = {"activations": "Activation-LEACE", "directions_orth": "Directions-orthogonal",
        "directions_leace": "Directions-LEACE"}
SCORE = {"mi": "MI", "nmi": "Normalized MI", "corr_ratio": "Correlation ratio η²"}
WEIGHTED, UNWEIGHTED = "SAE selected (Σ-weighted)", "SAE selected (raw atoms)"

#: Palette and chrome of the parent paper's figures (ColorBrewer "Paired",
#: dashed grey y-grid, boxed axes, framed legends), so the two sets of figures
#: can be read side by side. Raw keeps the parent's "Original" salmon,
#: Metadata-LEACE their LEACE green and PCA-1 their baseline purple; SAE-LEACE,
#: which the parent has no counterpart for, takes the palette's blue, and k-means
#: LEACE its orange.
INK, MUTED, GRID = "black", "#4d4d4d", "#c7c7c7"
COLOR = {"raw": "#fb9a99", "pca_top1": "#9467bd", "metadata_leace": "#33a02c",
         "sae_leace": "#1f78b4", "kmeans_leace": "#ff7f00"}
MARKER = {"raw": "o", "pca_top1": "s", "metadata_leace": "D", "sae_leace": "o",
          "kmeans_leace": "P"}
MECH_COLOR = {"activations": "#ff7f00", "directions_orth": "#6a3d9a",
              "directions_leace": "#1f78b4"}
CORPUS_COLOR = {"semeval": "#1f78b4", "swiltra": "#ff7f00", "bills_tweets": "#33a02c",
                "scotus_oyez_lexis": "#6a3d9a", "cap": "#e31a1c"}
SEP = "---"                       # a \midrule inside a table body

#: The five OOD panels in the plan's order, and their titles.
OOD_KEYS = [f"{e}/{d}" for e, d in R.OOD_MATRIX]


def ood_title(key: str) -> str:
    enc, don = key.split("/")
    return f"{ENCODER[enc]} / {CORPUS[don]} donor"


# --------------------------------------------------------------------------- #
# Raw inputs, recorded per exhibit for the index
# --------------------------------------------------------------------------- #
_ACCESSED: list = []


class Missing(Exception):
    """An exhibit's raw input does not exist yet."""


def raw(folder: str, name: str):
    """A runner's saved JSON, or None. Every lookup is recorded for the index."""
    path = R.raw_path(folder, name)
    _ACCESSED.append((path, os.path.exists(path)))
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def req(obj, what: str):
    if obj is None:
        raise Missing(what)
    return obj


def id_result(corpus: str, encoder: str = "e5-small"):
    return raw("id", f"id_{corpus}_{encoder}.json")


def get(res, cond: str, key: str):
    try:
        return res["agg"][cond][key]
    except (KeyError, TypeError):
        return None


def id_kmeans(corpus: str, ref, encoder: str = "e5-small") -> tuple:
    """New Table 2's k-means LEACE for `corpus`, as ``(result, source)``.

    `revised.run_id_kmeans` scores it on every seed of the `run_id` result `ref`. Until
    that has run, the same arm of the erasure-alternatives run stands in: its seeds
    share `run_id`'s splits, and it is used only if its Raw, PCA-1, Metadata-LEACE and
    SAE-LEACE reproduce `ref`'s on each of them. Source is "id", "alternatives" or None.
    """
    r = raw("id", f"id_kmeans_{corpus}_{encoder}.json")
    if r is not None:
        return r, "id"
    alt = raw("alternatives", f"alternatives_{corpus}_{encoder}.json")
    if alt is None or ref is None or "kmeans_leace" not in alt["agg"]:
        return None, None
    same = {"raw": "raw", "pca_top1": "pca_top1", "metadata_leace": "metadata_leace",
            "sae_mi": "sae_leace"}
    for p in alt["per_seed"]:
        q = next((q for q in ref["per_seed"] if q["seed"] == p["seed"]), None)
        if q is None or p["split"] != q["split"] or any(
                abs(p["metrics"][a]["linear_auc"] - q["metrics"][b]["linear_auc"]) > 0.005
                for a, b in same.items()):
            return None, None
    return alt, "alternatives"


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def pm(stat, dec: int = 3) -> str:
    if stat is None:
        return "NR"
    return f"{stat['mean']:.{dec}f} ± {stat['std']:.{dec}f}"


def num(v, dec: int = 3) -> str:
    return "NR" if v is None else f"{v:.{dec}f}"


def signed(v, dec: int = 3) -> str:
    return "NR" if v is None else f"{v:+.{dec}f}".replace("-", "−")


def ci(lo_hi, dec: int = 3) -> str:
    return f"[{signed(lo_hi[0], dec)}, {signed(lo_hi[1], dec)}]"


def intc(n) -> str:
    return f"{int(n):,}"


def seed_desc(res) -> str:
    sp = res["config"]["split"]
    n = len(res["seeds"])
    return (f"{n} seeds × {sp['folds']} outer folds" if sp["scheme"] == "kfold"
            else f"{n} seeds")


def nseeds(results) -> str:
    """"3 seeds" (or "2-3 seeds") across the results given."""
    ns = sorted({len(r["seeds"]) for r in (results.values() if isinstance(results, dict)
                                             else results) if r})
    if not ns:
        return "NR seeds"
    if len(ns) > 1:
        return f"{ns[0]}–{ns[-1]} seeds"
    return f"{ns[0]} seed" + ("s" if ns[0] != 1 else "")


def seeds_sentence(results: dict) -> str:
    parts = [f"{CORPUS[k]} {seed_desc(r)}" for k, r in results.items() if r]
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
_TEX_CHARS = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
              "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}", "\\": r"\textbackslash{}"}
_TEX_UNICODE = {"η²": r"$\eta^2$", "±": r"$\pm$", "−": r"$-$", "–": "--", "—": "---",
                "Δ": r"$\Delta$", "≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$",
                "→": r"$\rightarrow$", "Σ": r"$\Sigma$", "μ": r"$\mu$", "ᵀ": r"$^\top$",
                "∈": r"$\in$", "·": r"$\cdot$", "≈": r"$\approx$", "…": r"\ldots{}",
                "ρ": r"$\rho$", "∩": r"$\cap$"}


def tex(s) -> str:
    s = "".join(_TEX_CHARS.get(ch, ch) for ch in str(s))
    for u, t in _TEX_UNICODE.items():
        s = s.replace(u, t)
    return s


def _is_num(cell) -> bool:
    c = str(cell).replace("−", "-").replace("+", "").replace(",", "").split(" ")[0].rstrip("*")
    if c in ("NR", "N/A", "--", "—", "[") or c.startswith("[") or re.fullmatch(r"\d+–\d+", c):
        return True
    try:
        float(c)
        return True
    except ValueError:
        return False


def _align(header, rows) -> str:
    out = ""
    for j in range(len(header)):
        cells = [r[j] for r in rows if r != SEP]
        out += "r" if cells and all(_is_num(c) for c in cells) else "l"
    return out


def _out(kind: str, where: str, stem: str) -> str:
    path = os.path.join(artifacts.BASE_DIR, kind, where, stem)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def write_table(stem: str, where: str, caption: str, panels: list, note: str | None = None,
                wide: bool = False, colspec: str | None = None,
                tabcolsep: float | None = None) -> list:
    """Write one table as LaTeX (booktabs) and Markdown.

    `panels` is a list of ``(title or None, header, rows)``, optionally with a fourth
    element `groups`; a row equal to `SEP` is a rule. `groups` is a list of
    ``(label or None, n_columns)`` covering the header: each labelled group gets a
    spanning header with a rule under it in LaTeX and prefixes its column names in
    Markdown. Panels with different column counts are stacked tabulars. `wide` sets
    the table across both text columns (``table*``). `colspec` overrides the LaTeX column
    specification; wrapping ``p{...}`` columns are set ragged-right. `tabcolsep` (pt)
    narrows the column padding of a table with many columns. In any cell a newline
    starts a new line, and `backticked` text is set as a file name.
    """
    env = "table*" if wide else "table"
    tx = [rf"\begin{{{env}}}[t]", r"  \centering", r"  \footnotesize" if wide else r"  \small"]
    if wide or tabcolsep:
        tx.append(rf"  \setlength{{\tabcolsep}}{{{tabcolsep or 3.5}pt}}")
    tx += [f"  \\caption{{{tex(caption)}}}", f"  \\label{{tab:{stem}}}"]
    md = [f"**{caption}**", ""]

    ragged = bool(colspec) and "p{" in colspec
    end = r" \tabularnewline" if colspec else r" \\"

    def inline(s):
        """Text, with `backticked` spans set as small typewriter file names."""
        parts = re.split(r"`([^`]*)`", str(s))
        return "".join(tex(p) if k % 2 == 0 else
                       r"{\scriptsize\texttt{" + tex(p).replace(r"\_", r"\_\allowbreak{}") + "}}"
                       for k, p in enumerate(parts))

    def head(h):
        """A header cell; a newline in the name stacks it over two lines in LaTeX."""
        parts = str(h).split("\n")
        out = (inline(parts[0]) if len(parts) == 1
               else r"\shortstack{" + r"\\".join(map(inline, parts)) + "}")
        return r"\raggedright " + out if ragged else out

    def cell(c):
        # Wide tables set the SD smaller beside the mean, so that the columns fit.
        m = re.fullmatch(r"(\S+) ± (\S+)( .*)?", str(c))
        if wide and m:
            return (f"{tex(m.group(1))}\\,{{\\scriptsize $\\pm${tex(m.group(2))}}}"
                    + tex(m.group(3) or ""))
        lines = str(c).split("\n")
        if len(lines) == 1:
            out = inline(lines[0])
        elif ragged:
            out = r"\newline ".join(map(inline, lines))
        else:
            out = r"\begin{tabular}[t]{@{}l@{}}" + r"\\".join(map(inline, lines)) + r"\end{tabular}"
        return r"\raggedright " + out if ragged else out

    for i, (title, header, rows, *extra) in enumerate(panels):
        groups = extra[0] if extra else None
        align = _align(header, rows)
        if title:
            tx.append(f"  \\par\\smallskip\\textit{{{tex(title)}}}\\par\\smallskip")
            md += [f"*{title}*", ""]
        tx += [f"  \\begin{{tabular}}{{{colspec or align}}}", r"    \toprule"]
        md_header = [str(h).replace("\n", " ") for h in header]
        if groups:
            if sum(n for _, n in groups) != len(header):
                raise ValueError(f"{stem}: column groups span {sum(n for _, n in groups)} "
                                 f"columns, the header has {len(header)}")
            cells, rules, col = [], [], 1
            for label, n in groups:
                if label:
                    cells.append(f"\\multicolumn{{{n}}}{{c}}{{{tex(label)}}}")
                    rules.append(f"\\cmidrule(lr){{{col}-{col + n - 1}}}")
                    for j in range(col - 1, col - 1 + n):
                        md_header[j] = f"{label}: {md_header[j]}"
                else:
                    cells += [""] * n
                col += n
            tx += ["    " + " & ".join(cells) + end, "    " + " ".join(rules)]
        tx += ["    " + " & ".join(head(h) for h in header) + end, r"    \midrule"]
        tx += [r"    \midrule" if r == SEP else "    " + " & ".join(cell(c) for c in r) + end
               for r in rows]
        tx += [r"    \bottomrule", r"  \end{tabular}"]
        if i < len(panels) - 1:
            tx.append(r"  \par\medskip")
        esc = lambda c: str(c).replace("|", r"\|").replace("\n", "<br>")                      # noqa: E731
        md += ["| " + " | ".join(map(esc, md_header)) + " |",
               "|" + "|".join("---:" if a == "r" else "---" for a in align) + "|"]
        md += ["| " + " | ".join(map(esc, r)) + " |" for r in rows if r != SEP]
        md.append("")
    if note:
        tx.append(f"  \\par\\smallskip{{\\footnotesize {tex(note)}\\par}}")
        md += [f"*Note.* {note}", ""]
    tx += [rf"\end{{{env}}}", ""]
    base = _out("tables", where, stem)
    for ext, body in ((".tex", "\n".join(tx)), (".md", "\n".join(md))):
        with open(base + ext, "w") as fh:
            fh.write(body)
    return [base + ".tex", base + ".md"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                         "axes.edgecolor": "black", "axes.linewidth": 0.8,
                         "axes.spines.top": True, "axes.spines.right": True,
                         "axes.axisbelow": True,
                         "grid.color": GRID, "grid.linestyle": "--", "grid.linewidth": 0.8,
                         "legend.frameon": True, "legend.framealpha": 1.0,
                         "legend.edgecolor": "0.6", "legend.fancybox": False,
                         "font.family": "DejaVu Sans"})
    return plt


def _nr(ax, text: str = "NR (not run)"):
    """An empty panel: a note, no axes."""
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color=MUTED)
    ax.axis("off")


def save_fig(fig, stem: str, where: str) -> list:
    import matplotlib.pyplot as plt
    base = _out("figures", where, stem)
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [base + ".png", base + ".pdf"]


# --------------------------------------------------------------------------- #
# The exhibit registry
# --------------------------------------------------------------------------- #
EXHIBITS: list = []
INDEX: list = []


def exhibit(ident: str, title: str, where: str):
    """Register a builder; `build_all` runs them in registration order."""
    def deco(fn):
        @functools.wraps(fn)
        def run():
            _ACCESSED.clear()
            try:
                files = fn() or []
                status = "built"
            except Missing as m:
                files, status = [], f"skipped -- missing {m}"
            except Exception as exc:                                    # noqa: BLE001
                files, status = [], f"FAILED -- {type(exc).__name__}: {exc}"
                traceback.print_exc()
            sources = sorted({p for p, ok in _ACCESSED if ok})
            INDEX.append({"id": ident, "title": title, "where": where, "files": files,
                          "sources": sources, "status": status})
            print(f"[{status.split(' ')[0]}] {ident}: {title}")
            return files
        EXHIBITS.append(run)
        return run
    return deco


# =========================================================================== #
# Main text
# =========================================================================== #
@exhibit("Table 1", "ID dataset inventory", MAIN)
def table_1():
    inv = req(raw("inventory", "dataset_inventory.json"), "inventory/dataset_inventory.json")
    rows = []
    for key in R.ID_CORPORA:
        if key not in inv["corpora"]:
            rows.append([CORPUS[key], "NR", "NR", "NR", "NR", "NR"])
            continue
        v = inv["corpora"][key]
        sem = (["Retrieval"] if v["paired"] else []) + (["graded similarity"] if v["has_gold"] else [])
        if v.get("n_topics"):
            sem.append(f"{v['n_topics']}-topic clustering")
        rows.append([CORPUS[key],
                     f"{v['confounder'].capitalize()}: "
                     + "/".join(CLASS.get(c, c) for c in ordered(key, v["classes"])),
                     intc(v["n_docs"]), intc(v["n_pairs"]) if v["paired"] else "N/A",
                     intc(v["n_unpaired"]), "; ".join(sem) or "N/A"])
    return write_table(
        "table1_datasets", MAIN,
        "ID datasets after preprocessing (exact duplicate texts removed).",
        [(None, ["Dataset", "Nuisance and classes", "# docs", "# pairs", "# unpaired",
                 "Semantic evaluation"], rows)],
        note="# pairs counts matched cross-nuisance pairs (retrieval targets); unpaired "
             "documents are distractors in the retrieval index and eraser-training data. "
             "CAP's 21 topic labels are the semantic target, not nuisance classes; they "
             "score clusterings and never enter a fit.")


#: New Table 2's rows: the plan's four conditions, with k-means LEACE -- the label-free
#: eraser without an SAE -- beside PCA-1.
TABLE2_ROWS = ("raw", "pca_top1", "kmeans_leace", "metadata_leace", "sae_leace")


@exhibit("Table 2", "Primary ID comparison (E5-small)", MAIN)
def table_2():
    res = {k: id_result(k) for k in R.ID_CORPORA}
    if not any(res.values()):
        raise Missing("id/id_<corpus>_e5-small.json")
    km = {k: id_kmeans(k, res[k]) for k in R.ID_CORPORA}
    for k, (r, _) in km.items():
        if res[k] is not None and r is not None:
            res[k] = {**res[k], "agg": {**res[k]["agg"], "kmeans_leace": r["agg"]["kmeans_leace"]}}
    rows = TABLE2_ROWS
    # Panel A: every corpus has a nuisance, so every probe applies to every column.
    a = []
    for key, probe in (("linear_auc", "Linear (logistic)"), ("gb_auc", "Nonlinear (GB)"),
                       ("mlp_auc", "Nonlinear (MLP)")):
        if a:
            a.append(SEP)
        for c in rows:
            a.append([probe if c == rows[0] else "", LABEL[c]]
                     + [pm(get(res[k], c, key)) for k in R.ID_CORPORA])
    head_a = ["Probe", "Method"] + [CORPUS[k] for k in R.ID_CORPORA]
    # Panel B: only the tasks each corpus has -- paired retrieval, and CAP's topics -- one
    # row per task metric, so no cell is empty and the methods sit side by side.
    tasks = ([(k, m, lab) for k in R.PAIRED_ORIGINAL
              for m, lab in (("recall@1", "R@1"), ("recall@10", "R@10"))]
             + [("cap", "topic_purity", "Topic purity"), ("cap", "topic_ari", "Topic ARI")])
    b = []
    for i, (k, m, lab) in enumerate(tasks):
        first = i == 0 or tasks[i - 1][0] != k
        if first and b:
            b.append(SEP)
        b.append([CORPUS[k] if first else "", lab] + [pm(get(res[k], c, m)) for c in rows])
    head_b = ["Corpus", "Metric"] + [LABEL[c] for c in rows]
    nr = ([f"no nonlinear probes on {', '.join(m)}" for m in
           [[CORPUS[k] for k in R.ID_CORPORA if res[k] and get(res[k], "raw", "gb_auc") is None]]
           if m]
          + [f"no k-means LEACE on {', '.join(m)}" for m in
             [[CORPUS[k] for k in R.ID_CORPORA if res[k] and km[k][0] is None]] if m])
    borrowed = [k for k in R.ID_CORPORA if res[k] and km[k][1] == "alternatives"]
    short = [f"{CORPUS[k]} {len(km[k][0]['seeds'])} of {len(res[k]['seeds'])}" for k in borrowed
             if len(km[k][0]["seeds"]) < len(res[k]["seeds"])]
    km_note = (" k-means LEACE: LEACE on the one-hot k-means grouping of the training "
               "documents, with K chosen by the training silhouette (the rule SAE-LEACE uses "
               "for its pseudo-groups); no SAE and no metadata.")
    if borrowed:
        km_note += (f" Its values on {', '.join(CORPUS[k] for k in borrowed)} come from the "
                    "erasure-alternatives run, whose fits share these splits (its Raw, PCA-1, "
                    "Metadata-LEACE and SAE-LEACE reproduce the rows here)"
                    + (f"; seeds: {', '.join(short)}" if short else "") + ".")
    return write_table(
        "table2_id_comparison", MAIN,
        "Primary ID comparison, multilingual E5-small. Mean ± SD across run seeds.",
        [("Panel A — Nuisance leakage: held-out probe AUC (lower is better; 0.5 = chance)",
          head_a, a),
         ("Panel B — Semantic utility (higher is better)", head_b, b)], wide=True,
        note=f"Seeds: {seeds_sentence(res)}. SemEval's five outer folds are combined within "
             "each seed (fold AUCs averaged; retrieval pooled over the folds' held-out pairs) "
             "before the seed summary. Every transformation is fit on the outer training split "
             "and frozen. Panel A: absolute ROC-AUC of fresh probes trained and tested by 5-fold "
             "cross-validation within the transformed held-out documents (matched pairs in one "
             "fold), predicting language (SemEval de/en/es/zh, SwiLTra de/fr) or source "
             "(Bills–Tweets, SCOTUS, CAP bills/orders/news); multi-class corpora are macro "
             "one-vs-rest. Linear: L2 logistic regression; nonlinear: histogram gradient "
             "boosting (GB, library defaults) and an MLP with hidden layers (256, 128) and early "
             "stopping, on the same folds. An AUC below 0.5 is not better erasure. Panel B: R@1 "
             "and R@10 are bidirectional, with held-out pairs as queries against the full "
             "transformed corpus, self excluded (Bills–Tweets and CAP have no pairs); CAP topic "
             "purity (cluster-size weighted) and ARI are from k-means (k = 21, n_init = 20) on the "
             "transformed held-out CAP documents against its 21 topic labels."
             + km_note
             + (f" NR: not evaluated in this run ({'; '.join(nr)})." if nr else ""))


#: New Table 3's rows from the erasure-alternatives run, after the three interventions:
#: the nonlinear way of removing S, then the two controls.
TABLE3_EXTRA = (("sae_recon_suppressed", "Decode with S zeroed"),
                ("sae_random", "Directions-LEACE, 32 random latents"),
                ("sae_recon", "Decode only (nothing zeroed)"))


#: The columns of New Table 3 and Table D3.
INTERVENTION_METRICS = ("linear_auc", "gb_auc", "mlp_auc", "recall@1", "recall@10")
INTERVENTION_HEADER = ["Corpus", "Intervention", "Linear\n(logistic)", "Nonlinear\n(GB)",
                       "Nonlinear\n(MLP)", "R@1", "R@10"]
INTERVENTION_GROUPS = [(None, 2), ("Nuisance leakage (AUC)", 3), ("Paired retrieval", 2)]


def _intervention_rows(corpora, names=None) -> tuple:
    """The rows of New Table 3 and Table D3: per corpus, the mechanism run's three
    interventions on S, then `TABLE3_EXTRA` from the erasure-alternatives run. Those join
    a corpus only where that run -- the recipe refit on the same splits and seeds --
    reproduces the mechanism run's Directions-LEACE. A corpus without pairs prints "—"
    for retrieval; `names` maps a corpus to its row label (default `CORPUS`). Returns
    ``(rows, mechanism results, alternatives results, reproduced corpora, largest
    per-seed difference per corpus)``."""
    names = names or CORPUS
    res = {k: raw("mechanism", f"mechanism_{k}_e5-small.json") for k in corpora}
    if not any(res.values()):
        raise Missing("mechanism/mechanism_<corpus>_e5-small.json")
    alt = {k: raw("alternatives", f"alternatives_{k}_e5-small.json") for k in corpora}
    drift = {}
    for k in corpora:
        if res[k] and alt[k] and alt[k]["seeds"] == res[k]["seeds"]:
            a_, m_ = alt[k]["agg"]["sae_mi"], res[k]["agg"]["directions_leace"]
            drift[k] = max(abs(x - y) for key in ("linear_auc", "recall@10") if key in m_
                           for x, y in zip(a_[key]["values"], m_[key]["values"]))
    ok = [k for k in corpora if drift.get(k, 1.0) <= 0.005]
    rows = []
    for k in corpora:
        if rows:
            rows.append(SEP)
        paired = (res[k] or alt[k] or {}).get("paired", True)
        arms = [(MECH[m], res[k]["agg"].get(m, {}) if res[k] else {}) for m in R.MECHANISMS]
        arms += [(label, alt[k]["agg"].get(arm, {}) if k in ok else {})
                 for arm, label in TABLE3_EXTRA]
        for i, (label, ag) in enumerate(arms):
            rows.append([names.get(k, k) if i == 0 else "", label]
                        + [pm(ag.get(m)) if paired or not m.startswith("recall") else "—"
                           for m in INTERVENTION_METRICS])
    return rows, res, alt, ok, drift


@exhibit("Table 3", "Controlled intervention comparison", MAIN)
def table_3():
    rows, res, alt, ok, drift = _intervention_rows(R.PAIRED_ORIGINAL)
    header, groups = INTERVENTION_HEADER, INTERVENTION_GROUPS
    same = all(p["same_selected_set"] for r in res.values() if r for p in r["per_seed"])
    nr = ([] if all("gb_auc" in r["agg"][m] for r in res.values() if r for m in R.MECHANISMS)
          else ["no nonlinear probes on the first three interventions"])
    nr += ([f"no erasure-alternatives rows on {', '.join(CORPUS[k] for k in R.PAIRED_ORIGINAL if k not in ok)}"]
           if len(ok) < len(R.PAIRED_ORIGINAL) else [])
    return write_table(
        "table3_mechanisms", MAIN,
        "Controlled intervention comparison, E5-small, mean ± SD over "
        f"{nseeds(list(res.values()) + [alt[k] for k in ok])}.",
        [(None, header, rows, groups)], wide=True,
        note="Within each corpus, one SAE recipe, MI ranking, selected set S (N = 32), set of "
             "training documents and evaluation split serve the first four rows; only what is "
             "done with S changes (selected sets identical across the three LEACE-type "
             f"interventions in every fit: {'yes' if same else 'NO'}). Activation-LEACE erases "
             "the selected activations; Directions-orthogonal projects out span(D_S); "
             "Directions-LEACE, the SAE-LEACE of New Table 2, erases the directions' projections "
             "X D_Sᵀ; Decode with S zeroed encodes each document with the SAE, sets the latents "
             "in S to zero and decodes, a nonlinear map. The last two rows are controls on the "
             "same SAE: Directions-LEACE on 32 latents drawn at random instead of by MI, and the "
             "SAE's reconstruction with nothing zeroed. "
             + (f"The last three rows come from the erasure-alternatives run, on the same splits "
                f"and seeds; its MI-selected Directions-LEACE reproduces the Directions-LEACE row "
                f"(largest per-seed difference in linear AUC or R@10: {max(drift[k] for k in ok):.3f}). "
                if ok else "")
             + "Nuisance leakage (lower is better; 0.5 = chance): independent held-out audit with "
               "the probes of New Table 2. Paired retrieval (higher is better): R@1 and R@10 use "
               "held-out pairs as queries, both directions, against the full transformed corpus, "
               "self excluded."
             + (f" NR: not evaluated in this run ({'; '.join(nr)})." if nr else ""))


# ------------------------------- OOD summary ------------------------------ #
#: Frozen maps the erasure-alternatives run fitted beside the plan's three, on the same
#: donors, seeds and training splits: name -> (file prefix, {its transform: ours}).
#: A rename of None takes every map of the family that the plan's runs do not have.
OOD_EXTRA = {"kmeans": ("ood_kmeans", {"kmeans_auto": "kmeans_leace"}),
             "ranks": ("ood_ranks", None)}
OOD_ORDER = ("pca_top1", "pca_top2", "pca_top3", "pca_top4", "pca_top8", "pca_top16",
             "pca_top32", "pca_matched_var", "kmeans_leace", "kmeans_count", "kmeans_generous",
             "metadata_leace", "sae_leace")


def _ood_merge(r: dict, e: dict, rename: dict, name: str) -> None:
    """Add `e`'s transforms to the OOD result `r`, in place, once both are shown to
    score the same frozen fits (seeds, training splits) against the same Raw."""
    if (e["seeds"] != r["seeds"]
            or [p["fit"]["split_id"] for p in e["per_seed"]]
            != [p["fit"]["split_id"] for p in r["per_seed"]]
            or any(e["raw"][t]["score"] != r["raw"][t]["score"] for t in r["raw"])):
        raise ValueError(f"{name} does not share the plan's OOD fits and Raw scores")
    for p, q in zip(r["per_seed"], e["per_seed"]):
        for theirs, ours in rename.items():
            p["scores"][ours] = q["scores"][theirs]
    for theirs, ours in rename.items():
        r["labels"][ours] = (LABEL.get(ours)
                             or ("PCA-" + ours[7:] if re.fullmatch(r"pca_top\d+", ours) else None)
                             or e["labels"].get(theirs, ours))
    have = set(r["transforms"]) | set(rename.values())
    r["transforms"] = [c for c in OOD_ORDER if c in have] + sorted(have - set(OOD_ORDER))


def _ood_extra_note(s) -> str:
    """What a summary's added maps are, for a table note."""
    trs, parts = _trs(s), []
    if "kmeans_leace" in trs:
        parts.append("k-means LEACE is LEACE on the one-hot k-means grouping of the donor's "
                     "training split, K chosen by silhouette, no SAE")
    ranks = [c[7:] for c in trs if re.fullmatch(r"pca_top\d+", c) and c != "pca_top1"]
    if ranks:
        parts.append(f"PCA-{ranks[0]} to PCA-{ranks[-1]} remove that many top principal "
                     "components")
    if "pca_matched_var" in trs:
        parts.append("the variance-matched PCA map removes, per seed, the fewest principal "
                     "directions carrying as much training variance as that seed's frozen "
                     "SAE-LEACE map removes")
    return (" " + "; ".join(parts) + ". These maps come from the erasure-alternatives run: "
            "fitted on the same seeds' donor training splits as the other maps, then frozen."
            if parts else "")


def ood_summary(n_boot: int = 10000, overwrite: bool = False, extra=()):
    """`ood.summarize` over the five settings, cached next to the raw results.

    `extra` names maps of `OOD_EXTRA` to add as transforms; each is added only when every
    setting has it. Adding transforms leaves the others' numbers unchanged: the
    bootstrap resamples are drawn per setting before any transform is scored.
    """
    results, paths = {}, []
    for key in OOD_KEYS:
        enc, don = key.split("/")
        r = raw("ood", f"ood_{enc}_{don}.json")
        if r is not None:
            results[key] = r
            paths.append(R.raw_path("ood", f"ood_{enc}_{don}.json"))
    if not results:
        return None
    used = []
    for name in extra:
        prefix, rename = OOD_EXTRA[name]
        files = {key: f"{prefix}_{key.replace('/', '_')}.json" for key in results}
        found = {key: raw("alternatives", f) for key, f in files.items()}
        if any(v is None for v in found.values()):
            continue
        if rename is None:
            common = set.intersection(*(set(e["transforms"]) for e in found.values()))
            plan = set(next(iter(results.values()))["transforms"])
            rename = {c: c for c in list(OOD_ORDER) + sorted(common - set(OOD_ORDER))
                      if c in common and c not in plan}
        if not rename or any(not set(rename) <= set(e["transforms"]) for e in found.values()):
            continue
        for key, e in found.items():
            _ood_merge(results[key], e, rename, files[key])
            paths.append(R.raw_path("alternatives", files[key]))
        used.append(name)
    transforms = next(iter(results.values()))["transforms"]
    out = R.raw_path("ood", "_".join(["ood_summary", *used]) + ".json")
    fresh = (os.path.exists(out) and not overwrite
             and all(os.path.getmtime(out) >= os.path.getmtime(p) for p in paths))
    if fresh:
        with open(out) as fh:
            s = json.load(fh)
        if (s.get("order") == list(results) and s.get("n_boot") == n_boot
                and s.get("transforms") == transforms):
            _ACCESSED.append((out, True))
            return s
    s = ood.summarize(results, n_boot=n_boot, save=False)
    with open(out, "w") as fh:
        json.dump(s, fh, indent=1)
    _ACCESSED.append((out, True))
    return s


def _task_order(v: dict) -> list:
    return [t for t in ood.LEGAL if t in v["legal"]]


_EXTRA_STYLE = [("#e34948", "v"), ("#4a3aa7", "^"), ("#eda100", "P"), ("#e87ba4", "X")]


def _trs(summary) -> list:
    """The transformations a summary carries, in its own order."""
    return list(summary.get("transforms", R.OOD_TRANSFORMS))


def _lab(summary, c: str) -> str:
    return summary.get("labels", {}).get(c) or LABEL.get(c, c)


def _style(summary, c: str) -> tuple:
    """(colour, marker): fixed for the plan's three, assigned in order otherwise."""
    if c in COLOR and c != "raw":
        return COLOR[c], MARKER[c]
    extra = [t for t in _trs(summary) if t not in COLOR]
    return _EXTRA_STYLE[extra.index(c) % len(_EXTRA_STYLE)]


def _offsets(summary) -> dict:
    trs = _trs(summary)
    offs = np.linspace(0.27, -0.27, len(trs)) if len(trs) > 1 else [0.0]
    return dict(zip(trs, offs))


def _ood_values(summary, transforms, keys, tasks_of) -> list:
    vals = [0.0]
    for key in keys:
        v = summary["settings"][key]
        for c in transforms:
            for t in tasks_of(v):
                ch = v["changes"][c][t]
                vals += [ch["delta"], *ch["ci"]]
    return vals


def _limits(vals, pad: float = 0.06) -> tuple:
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1e-3)
    return lo - pad * span, hi + pad * span


def _ood_panel(ax, summary, v, xlim, clip: bool, labels: bool):
    tasks = _task_order(v)
    y0 = np.arange(len(tasks))[::-1].astype(float)
    off = _offsets(summary)
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    for c in _trs(summary):
        col, mk = _style(summary, c)
        for i, t in enumerate(tasks):
            ch = v["changes"][c][t]
            d, (lo, hi), y = ch["delta"], ch["ci"], y0[i] + off[c]
            if clip and not (xlim[0] <= d <= xlim[1]):
                left = d < xlim[0]
                edge = xlim[0] if left else xlim[1]
                ax.plot(edge, y, marker="<" if left else ">", color=col, ms=5,
                        clip_on=False, zorder=3)
                ax.annotate(signed(d), (edge, y), xytext=(5 if left else -5, 0),
                            textcoords="offset points", ha="left" if left else "right",
                            va="center", fontsize=6, color=col)
                continue
            ax.plot([max(lo, xlim[0]), min(hi, xlim[1])], [y, y], color=col, lw=1.1,
                    zorder=2, solid_capstyle="butt")
            ax.plot(d, y, marker=mk, color=col, ms=4.2, mec="white", mew=0.5,
                    zorder=3, ls="none")
    ax.set_yticks(y0)
    ax.set_yticklabels([v["tasks"][t]["label"] for t in tasks] if labels else [])
    ax.set_ylim(-0.6, len(tasks) - 0.4)
    ax.set_xlim(*xlim)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))     # labels collide at 6
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9.5)


def _legend_panel(ax, summary):
    """The sixth panel: the series legend only. Everything the reader needs to
    interpret the marks -- what a point and a bar are, the resampling scheme, the
    off-scale arrows in the zoomed variant -- belongs in the figure caption, not
    on the figure."""
    from matplotlib.lines import Line2D
    ax.axis("off")
    handles = [Line2D([0], [0], color=_style(summary, c)[0], marker=_style(summary, c)[1],
                      mec="white", ms=7.5, lw=1.6, label=_lab(summary, c)) for c in _trs(summary)]
    handles.append(Line2D([0], [0], color=INK, lw=1.0, label="Raw (no change)"))
    ax.legend(handles=handles, loc="center", frameon=True, fontsize=10.5, handlelength=2.4,
              labelspacing=0.7, borderpad=0.7)


def _figure_1(summary, xlim, zoom: bool):
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6))
    slots = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for key, (i, j) in zip(OOD_KEYS, slots):
        ax = axes[i, j]
        if key not in summary["settings"]:
            _nr(ax)
            ax.set_title(ood_title(key))
            continue
        _ood_panel(ax, summary, summary["settings"][key], xlim, clip=zoom, labels=(j == 0))
        ax.set_title(ood_title(key))
    _legend_panel(axes[1, 2], summary)
    fig.supxlabel("ΔnDCG@10 (transformed − Raw)", fontsize=12, y=0.005)
    fig.tight_layout()
    return fig


def build_figure_1(s, suffix: str = "") -> list:
    keys = [k for k in OOD_KEYS if k in s["settings"]]
    full = _limits(_ood_values(s, _trs(s), keys, _task_order))
    files = save_fig(_figure_1(s, full, zoom=False), "figure1_ood_retrieval" + suffix, MAIN)
    near = _limits(_ood_values(s, [c for c in ("kmeans_leace", "metadata_leace", "sae_leace")
                                   if c in _trs(s)], keys, _task_order), pad=0.1)
    if (full[1] - full[0]) > 2.0 * (near[1] - near[0]):
        files += save_fig(_figure_1(s, near, zoom=True), "figure1_ood_retrieval_zoom" + suffix, MAIN)
    return files


@exhibit("Figure 1", "Task-level OOD retrieval changes", MAIN)
def figure_1():
    return build_figure_1(req(ood_summary(extra=("kmeans",)), "ood/ood_<encoder>_<donor>.json"))


def _shade(color: str, k: int, n: int) -> tuple:
    """`k`-th of `n` shades of `color`, light to dark, as the parent shades donors."""
    import matplotlib.colors as mc
    rgb = np.array(mc.to_rgb(color))
    f = 0.0 if n < 2 else 0.55 * (1 - k / (n - 1))          # 0.55 -> 0 lightening
    return tuple(rgb + (1.0 - rgb) * f)


def build_figure_1b(s, suffix: str = "") -> list:
    """The parent's display of the same runs: absolute nDCG@10 as grouped bars,
    one panel per encoder, donors as light-to-dark shades of each condition."""
    plt = _plt()
    trs = _trs(s)
    keys = [k for k in OOD_KEYS if k in s["settings"]]
    encoders = list(dict.fromkeys(k.split("/")[0] for k in keys))
    tasks = _task_order(s["settings"][keys[0]])
    fig, axes = plt.subplots(len(encoders), 1, figsize=(9.2, 3.1 * len(encoders)), squeeze=False)
    x = np.arange(len(tasks))
    for ax, enc in zip(axes.ravel(), encoders):
        donors = [k.split("/")[1] for k in keys if k.startswith(enc + "/")]
        n_bars = 1 + len(trs) * len(donors)
        w = 0.86 / n_bars
        raw = s["settings"][f"{enc}/{donors[0]}"]["scores"]["raw"]
        ax.bar(x - 0.43 + w / 2, [raw[t]["mean"] for t in tasks], w, color=COLOR["raw"],
               label="Raw", zorder=2)
        for ci, c in enumerate(trs):
            for di, don in enumerate(donors):
                sc = s["settings"][f"{enc}/{don}"]["scores"][c]
                pos = x - 0.43 + w / 2 + w * (1 + ci * len(donors) + di)
                ax.bar(pos, [sc[t]["mean"] for t in tasks], w,
                       yerr=[sc[t].get("std", 0.0) for t in tasks], error_kw=dict(lw=0.6, capsize=1.5),
                       color=_shade(COLOR.get(c, "#777777"), di, len(donors)), ecolor="black",
                       label=f"{_lab(s, c)} ({CORPUS.get(don, don)})", zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([s["settings"][keys[0]]["tasks"][t]["label"] for t in tasks],
                           rotation=20, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("NDCG@10")
        ax.set_title(f"{ENCODER.get(enc, enc)} model on legal retrieval tasks")
        ax.grid(axis="y")
        ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    return save_fig(fig, "figure1b_ood_absolute" + suffix, MAIN)


@exhibit("Figure 1b", "OOD absolute scores, parent-style bars", MAIN)
def figure_1b():
    return build_figure_1b(req(ood_summary(), "ood/ood_<encoder>_<donor>.json"))


#: New Figure 2's corpora: the three paired ones, then CAP (utility: topic purity).
#: Bills–Tweets has no utility task.
FIG2_CORPORA = ("semeval", "swiltra", "scotus_oyez_lexis", "cap")
#: series -> (legend label, colour, marker, line style). A point's x is the most
#: directions its eraser can remove: rank r for PCA, K − 1 for k-means LEACE, N for
#: SAE-LEACE.
FIG2_STYLE = {"pca": ("PCA removal (rank r)", COLOR["pca_top1"], "s", "-"),
              "kmeans": ("k-means LEACE, fixed K (rank K − 1)", COLOR["kmeans_leace"], "P", "-"),
              "mi": ("SAE-LEACE, MI top-N", COLOR["sae_leace"], "o", "-"),
              "random": ("SAE-LEACE, N random latents", "#8c510a", "^", "--")}


def _fig2_series(alt: dict, km, bud, key: str) -> dict:
    """{series: [(x, mean, sd)]} for one corpus and metric. SAE-LEACE's MI curve comes
    from the budget sweep where there is one (the same splits and fits on a finer
    grid), else from the erasure-alternatives run's budget arms."""
    def pts(table, name, xs, shift=0):
        out = []
        for x in xs:
            st = (table.get(name.format(x)) or {}).get(key)
            if st is not None:
                out.append((x - shift, st["mean"], st["std"]))
        return out
    return {"pca": pts(alt["agg"], "pca_top{}", alt["ranks"]),
            "kmeans": pts(km["agg"], "kmeans_k{}", km["ks"], shift=1) if km else [],
            "mi": (pts(bud["curve"], "{}", [int(n) for n in bud["grid"]]) if bud
                   else pts(alt["agg"], "sae_mi_n{}", alt["budget_grid"])),
            "random": pts(alt["agg"], "sae_random_n{}", alt["budget_grid"])}


#: Figure 2 is set across both text columns, where it prints at about 0.6 of its drawn
#: size; these sizes (pt, at drawn size) keep every text element at 7-10 pt in print.
FIG2_FONT = {"title": 13.5, "label": 13, "tick": 10, "legend": 12.5, "note": 10}
#: Decimal places of the y ticks in each row (leakage AUC, then utility).
FIG2_DECIMALS = (1, 2)


def _decimals(values) -> int:
    """The fewest decimal places (up to 4) that write every value exactly."""
    return max(next(d for d in range(5) if abs(v - round(v, d)) < 1e-9 or d == 4)
               for v in values)


@exhibit("Figure 2", "Leakage and utility against the number of erased directions", MAIN)
def figure_2():
    return build_figure_2(FIG2_CORPORA, "figure2_erased_directions", MAIN)


def build_figure_2(corpora, stem: str, where: str, titles=None) -> list:
    """New Figure 2 for any `corpora` (one column each), saved as `stem` in `where`;
    `titles` maps a corpus to its column title (default `CORPUS`)."""
    titles = titles or CORPUS
    alt = {k: raw("alternatives", f"alternatives_{k}_e5-small.json") for k in corpora}
    if not any(alt.values()):
        raise Missing("alternatives/alternatives_<corpus>_e5-small.json")
    km = {k: raw("alternatives", f"kmeans_count_{k}_e5-small.json") for k in corpora}
    bud = {k: raw("budget", f"budget_{k}_e5-small.json") for k in corpora}
    plt = _plt()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, MaxNLocator
    from matplotlib.transforms import Bbox
    fs = FIG2_FONT
    ticks = (1, 2, 4, 8, 16, 32, 64)
    # Reference lines: (condition, dash pattern, width). Dash patterns scale with width.
    refs = (("raw", (0, (1, 1.5)), 1.8), ("metadata_leace", (0, (4, 2)), 1.7))
    c1_style = dict(color="#8c8c8c", lw=1.2, ls=(0, (5, 2, 1, 2)))
    ring = dict(ls="none", mfc="none", mew=1.3, ms=10, zorder=4)
    line = dict(lw=1.3, ms=4.2)
    ylabel = {"linear_auc": "Linear probe AUC", "recall@10": "R@10",
              "topic_purity": "Topic purity"}
    fig, axes = plt.subplots(2, len(corpora), figsize=(11.8, 7.6), squeeze=False)
    spans = {0: [], 1: []}
    for j, k in enumerate(corpora):
        r = alt[k]
        for i, key in enumerate(("linear_auc", "topic_purity" if k == "cap" else "recall@10")):
            ax = axes[i, j]
            if r is None:
                _nr(ax)
                continue
            ax.axvline(r["n_classes"] - 1, zorder=1, **c1_style)
            # The axis spans what the panel draws -- the curves, the ringed pre-specified
            # settings and the Metadata-LEACE line -- so that nearby curves stay apart.
            ys = [r["agg"]["metadata_leace"][key]["mean"]]
            for name, pts in _fig2_series(r, km[k], bud[k], key).items():
                if pts:
                    _, col, mk, ls = FIG2_STYLE[name]
                    x, m, _ = (np.array(v) for v in zip(*pts))
                    ax.plot(x, m, color=col, marker=mk, ls=ls, zorder=3, **line)
                    ys += list(m)
            # The pre-specified settings, ringed: PCA-1, SAE-LEACE at N = 32, and k-means
            # LEACE at its silhouette K (x at the mean K − 1 where K varies across fits).
            pre = [("pca", 1, r["agg"]["pca_top1"]), ("mi", r["sae_budget"], r["agg"]["sae_mi"])]
            if km[k]:
                k_auto = [d["k_auto"] for p in km[k]["per_seed"] for d in p["diagnostics"]]
                pre.append(("kmeans", float(np.mean(k_auto)) - 1, r["agg"]["kmeans_leace"]))
            for name, x, ag in pre:
                if key in ag:
                    _, col, mk, _ = FIG2_STYLE[name]
                    ax.plot(x, ag[key]["mean"], marker=mk, color=col, ms=line["ms"], ls="none",
                            zorder=4)
                    ax.plot(x, ag[key]["mean"], marker="o", mec=col, **ring)
                    ys.append(ag[key]["mean"])
            lo, hi = min(ys), max(ys)
            pad = {"below": 0.08 * (hi - lo), "above": 0.08 * (hi - lo)}
            # Raw is drawn only where it falls inside that span; otherwise stretching the
            # axis to it would squeeze every curve together, so its value is printed on
            # the edge it lies beyond.
            for c, dash, lw in refs:
                v = r["agg"][c][key]["mean"]
                if lo <= v <= hi:
                    ax.axhline(v, color=COLOR[c], lw=lw, ls=dash, zorder=1)
                else:
                    below = v < lo
                    # Headroom on that edge, so the printed value clears every curve.
                    pad["below" if below else "above"] = 0.2 * (hi - lo)
                    ax.annotate(f"{LABEL[c]}: {v:.3f} " + ("↓" if below else "↑"),
                                (0.04, 0.04) if below else (0.96, 0.96),
                                xycoords="axes fraction", ha="left" if below else "right",
                                va="bottom" if below else "top", fontsize=fs["note"],
                                color=INK, zorder=5,
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
            ax.set_ylim(lo - pad["below"], hi + pad["above"])
            spans[i].append(ax)
            ax.set_xscale("log", base=2)
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
            ax.minorticks_off()
            ax.set_xlim(0.82, 78)
            ax.tick_params(axis="both", labelsize=fs["tick"])
            ax.set_ylabel(ylabel[key], fontsize=fs["label"], labelpad=2)
            if i == 0:
                ax.set_title(titles.get(k, k), fontsize=fs["title"], pad=6)
    # Every panel of a row writes its ticks with the row's number of decimals. A panel
    # keeps matplotlib's own tick values unless one of them cannot be written exactly
    # with that many decimals (0.825 at two); it then gets ticks in steps of 1, 2 or
    # 5 in the last decimal, over the same axis range. The values are frozen first, so
    # that the labels' width cannot change which values the locator picks.
    layout = dict(rect=(0, 0.15, 1, 1), h_pad=3.0, w_pad=3.0)
    fig.tight_layout(**layout)
    for i, row in spans.items():
        dec = FIG2_DECIMALS[i]
        for ax in row:
            lo, hi = ax.get_ylim()
            t = [v for v in ax.get_yticks() if lo <= v <= hi]
            if _decimals(t) > dec:
                t = [v for v in MaxNLocator(nbins=max(len(t) - 1, 3), steps=[1, 2, 5, 10])
                     .tick_values(lo, hi) if lo <= v <= hi]
            # A panel spanning less than the row's resolution (retrieval near 1.0 in a
            # saturated setting) would print two ticks with the same label; give it the
            # decimals its own ticks need instead.
            d = dec
            while len({f"{v:.{d}f}" for v in t}) < len(t) and d < 4:
                d += 1
            ax.yaxis.set_major_locator(FixedLocator(t))
            ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{d}f"))
    fig.tight_layout(**layout)
    # The shared x label and the legend go under the panels, the legend exactly as wide
    # as the panels together with their tick labels and axis labels.
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bb = Bbox.union([ax.get_tightbbox(rend) for ax in axes.flat]).transformed(inv)
    xl = fig.text((bb.x0 + bb.x1) / 2, bb.y0 - 0.012, "Erased directions (log₂ scale)",
                  ha="center", va="top", fontsize=fs["label"])
    xl_bb = xl.get_window_extent(rend).transformed(inv)
    handles = [Line2D([0], [0], color=col, marker=mk, ls=ls, label=label, **line)
               for label, col, mk, ls in FIG2_STYLE.values()]
    handles.append(Line2D([0], [0], marker="o", mec=INK, label="Pre-specified setting", **ring))
    handles += [Line2D([0], [0], color=COLOR[c], ls=dash, lw=lw, label=LABEL[c])
                for c, dash, lw in refs]
    handles.append(Line2D([0], [0], label="C − 1", **c1_style))
    # Three columns, filled so that the legend reads row by row: the three erasers,
    # then the random control with the ring and Metadata-LEACE, then Raw and C − 1.
    handles = [handles[i] for i in (0, 3, 5, 1, 4, 7, 2, 6)]
    fig.legend(handles=handles, loc="upper left", ncol=3, mode="expand", frameon=True,
               bbox_to_anchor=(bb.x0, 0.0, bb.x1 - bb.x0, xl_bb.y0 - 0.018),
               fontsize=fs["legend"], handlelength=2.4, handletextpad=0.5,
               labelspacing=0.4, borderpad=0.5, borderaxespad=0.0)
    return save_fig(fig, stem, where)


def build_figure_budget() -> list:
    """Figure D7 (the first draft's New Figure 2): SAE-LEACE against its budget N."""
    res = {k: raw("budget", f"budget_{k}_e5-small.json") for k in R.PAIRED_ORIGINAL}
    if not any(res.values()):
        raise Missing("budget/budget_<corpus>_e5-small.json")
    plt = _plt()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 4.5), sharey="row")
    for j, k in enumerate(R.PAIRED_ORIGINAL):
        r = res[k]
        for i, key in enumerate(("linear_auc", "recall@10")):
            ax = axes[i, j]
            if r is None:
                _nr(ax)
                continue
            grid = [int(n) for n in r["grid"]]
            m = np.array([r["curve"][str(n)][key]["mean"] for n in grid])
            s = np.array([r["curve"][str(n)][key]["std"] for n in grid])
            # Interleaved dash phases and widths: PCA-1 and Metadata-LEACE coincide to
            # within 0.002 on some retrieval panels, and a plain dashed line would hide
            # whichever is drawn first.
            for c, dash, lw in (("raw", (0, (1, 2.5)), 1.0),
                                ("pca_top1", (0, (6, 4)), 1.6),
                                ("metadata_leace", (5, (6, 4)), 1.1)):
                ax.axhline(r["references"][c][key]["mean"], color=COLOR[c], lw=lw, ls=dash,
                           zorder=1)
            if key == "linear_auc":
                ax.axhline(0.5, color=MUTED, lw=0.7, ls=":", zorder=1)
            ax.axvline(r["default_n"], color=MUTED, lw=0.7, ls=":", zorder=1)
            ax.fill_between(grid, m - s, m + s, color=COLOR["sae_leace"], alpha=0.18, lw=0)
            ax.plot(grid, m, "-o", color=COLOR["sae_leace"], ms=3, lw=1.2, zorder=3)
            d = grid.index(r["default_n"])
            ax.plot(grid[d], m[d], "o", mfc="white", mec=COLOR["sae_leace"], ms=7, mew=1.4,
                    zorder=4)
            ax.set_xscale("log", base=2)
            ax.set_xticks(grid)
            # All ten budgets stay labelled (the plan asks for them), horizontal;
            # the figure is wide enough that 12/16 and 24/32 do not touch.
            ax.set_xticklabels([str(n) for n in grid], fontsize=6.5)
            ax.minorticks_off()
            ax.grid(axis="y")
            if i == 0:
                ax.set_title(CORPUS[k], pad=8)
            else:
                ax.set_xlabel("Selected latents N (log₂ scale)")
        axes[0, 0].set_ylabel("Linear probe AUC")
        axes[1, 0].set_ylabel("R@10")
    handles = [Line2D([0], [0], color=COLOR["sae_leace"], marker="o", ms=3,
                      label="SAE-LEACE (mean ± 1 SD)"),
               Line2D([0], [0], color=COLOR["sae_leace"], marker="o", mfc="white", ms=7, lw=0,
                      label="N = 32 (fixed default)")]
    handles += [Line2D([0], [0], color=COLOR[c], ls=dash, lw=lw, label=f"{LABEL[c]} (same splits)")
                for c, dash, lw in (("raw", (0, (1, 2.5)), 1.0), ("pca_top1", (0, (6, 4)), 1.6),
                                    ("metadata_leace", (5, (6, 4)), 1.1))]
    handles.append(Line2D([0], [0], color=MUTED, ls=":", lw=0.7, label="AUC chance (0.5)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True,
               bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    return save_fig(fig, "figureD7_budget", APPX)


# =========================================================================== #
# Appendix A
# =========================================================================== #
@exhibit("Table A1", "CAP topics by source", APPX)
def table_a1():
    inv = req(raw("inventory", "dataset_inventory.json"), "inventory/dataset_inventory.json")
    comp = inv["cap_composition"]
    srcs = ("bill", "order", "newspaper")
    rows = [[str(r["topic_id"]), r["topic"]] + [intc(r[s]) for s in srcs]
            + [intc(sum(r[s] for s in srcs))] for r in comp]
    tot = [sum(r[s] for s in srcs) for r in comp]
    rows += [SEP, ["Total", f"All {len(comp)} topics"]
             + [intc(sum(r[s] for r in comp)) for s in srcs] + [intc(sum(tot))]]
    return write_table(
        "tableA1_cap_composition", APPX, "CAP topics by source (corpus counts).",
        [(None, ["Topic ID", "Topic name", "Bills", "Executive orders", "Newspapers", "Total"],
          rows)],
        note="Document counts of the CAP corpus after the preprocessing used in every "
             "experiment (not eraser-training counts). Source and topic labels describe and "
             "evaluate the corpus; neither enters SAE-LEACE fitting. A source–topic cell of 0 "
             "is a genuinely absent combination.")


@exhibit("Table A2", "Exact encoders and usage", APPX)
def table_a2():
    inv = req(raw("inventory", "encoder_inventory.json"), "inventory/encoder_inventory.json")
    rows, notes = [], set()
    for enc, v in inv["encoders"].items():
        rev = (v.get("revision") or "unrecorded")[:10]
        if not v.get("revision_source", "").startswith("model-cache snapshot"):
            rev += "*"
            notes.add("* Not in the model cache of the environment that built this table; "
                      "the Hub's current revision is shown instead of the one the runs used.")
        pool = (v.get("pooling") or "?") + (" + L2 normalize" if v.get("normalize") else "")
        rows.append([ENCODER.get(enc, enc), f"{v['model']} @ {rev}", str(v["dim"]),
                     v["prompt"], pool, v["coverage"]])
    return write_table(
        "tableA2_encoders", APPX, "Exact encoders and their usage.",
        [(None, ["Short name", "Model identifier @ revision", "d", "Prompt / prefix",
                 "Pooling / normalization", "Evaluation coverage"], rows)],
        note=" ".join(sorted(notes) + ["Revisions are the model snapshots the runs "
                                        "downloaded and used."]) + " ID corpora are encoded with the listed prompt "
             "(E5: \"query: \" on every document, the symmetric setting). OOD targets are "
             f"encoded in the \"{inv.get('ood_target_prompts', ood.PROMPT_STYLE)}\" style "
             "(ood.PROMPT_STYLES) identically for Raw and every transformation. Transfer is "
             "always within one exact encoder.")


@exhibit("Table A3", "Shared hyperparameters and where they are fitted", APPX)
def table_a3():
    import sae as S
    from concept_erasure import LeaceFitter
    from sae_confounder_pipeline import score_confounder_features
    sig = {k: v.default for k, v in inspect.signature(S.train_sae).parameters.items()}
    lf = {k: v.default for k, v in inspect.signature(LeaceFitter.__init__).parameters.items()}
    sc = {k: v.default for k, v in inspect.signature(score_confounder_features).parameters.items()}
    rec = E.RECIPE
    summary = ood_summary()
    n_boot = f"{summary['n_boot']:,}" if summary else "10,000"
    T, D, F = "Donor/ID training documents", "Fixed configuration", "Probe-training fold"
    kgrid = "{" + ", ".join(map(str, P.PSEUDO_GROUP_GRID)) + "}"
    rows = [
        ["SAE", "Input normalization", "none beyond the encoder output; pre-encoder bias "
         "initialised to the training mean", T],
        ["SAE", "Expansion factor", str(rec["expansion_factor"]), D],
        ["SAE", "Dictionary size", f"{rec['expansion_factor']}d", D],
        ["SAE", "TopK sparsity", f"k = {rec['k']}", D],
        ["SAE", "Epochs", str(rec["n_epochs"]), D],
        ["SAE", "Optimizer", "Adam", D],
        ["SAE", "Learning rate", f"{sig['lr']:g}", D],
        ["SAE", "Batch size", intc(sig["batch_size"]), D],
        ["SAE", "Decoder-row constraint", "unit L2 norm after every step", D],
        ["SAE", "AuxK coefficient", f"1/{round(1 / sig['aux_coef'])} (aux k = 2k)", D],
        ["SAE", "Dead-feature threshold", f"no firing for {sig['dead_epochs']} epochs", D],
        SEP,
        ["Selector", "Firing-frequency filter",
         f"[{sc['min_activation_freq']}, {sc['max_activation_freq']}]", T],
        ["Selector", "Firing definition", "TopK-gated activation > 0", T],
        ["Selector", "Score", "binary-firing MI I(F_j; G), plug-in, bits", T],
        ["Selector", "Pseudo-group candidates", f"K_proxy ∈ {kgrid}", D],
        ["Selector", "Silhouette selection", "k-means on ≤ 4,000 training embeddings; "
         "largest silhouette, ties to the smaller K", T],
        ["Selector", "Erasure budget", f"N = {rec['top_features']} (MI top-N)", D],
        SEP,
        ["Transformation", "SAE-LEACE concept", "X D_Sᵀ, columns standardized", T],
        ["Transformation", "Metadata-LEACE concept", "one-hot nuisance label", "Training documents and labels"],
        ["Transformation", "Covariance / centering",
         f"concept-erasure LEACE: mean-centred affine map, shrinkage = {lf['shrinkage']}, "
         f"svd_tol = {lf['svd_tol']}, constrain_cov_trace = {lf['constrain_cov_trace']}", T],
        ["Transformation", "PCA-1 operator", "T(x) = μ + (I − u uᵀ)(x − μ), μ and u from the "
         "training documents", T],
        ["Transformation", "Final normalization", "none; retrieval uses cosine similarity", "—"],
        SEP,
        ["Evaluator", "Probe family", "L2 logistic regression; one-vs-rest for > 2 classes", F],
        ["Evaluator", "Regularization", "C ∈ {" + ", ".join(f"{c:g}" for c in audit.C_GRID)
         + "}, inner 1/5 split, mean OvR AUC", F],
        ["Evaluator", "Probe scaling", "standardization with probe-training statistics; "
         f"coordinates with SD < {audit.REL_STD_FLOOR:g} × median SD treated as constant", F],
        ["Evaluator", "Audit folds", f"{audit.AUDIT_FOLDS}, group-stratified (pairs together); "
         "reduced only if a class would be missing", "Held-out documents"],
        ["Evaluator", "Nonlinear probes", "HistGradientBoosting (defaults); MLP "
         f"{audit.MLP_HIDDEN}, early stopping", F],
        ["Evaluator", "Topic clustering", f"k-means, K_topic = {audit.K_TOPIC}, "
         f"n_init = {audit.KMEANS_N_INIT}", "Held-out CAP documents (evaluation only)"],
        ["Evaluator", "ID retrieval", "R@1 / R@10, bidirectional, full transformed corpus as "
         "index, self excluded", "Held-out pairs"],
        ["Evaluator", "OOD bootstrap", f"paired, {n_boot} resamples of queries (AILA tasks "
         "jointly) or STS pairs", "Target tasks"],
    ]
    return write_table(
        "tableA3_hyperparameters", APPX, "Shared hyperparameters and where they are fitted.",
        [(None, ["Component", "Parameter / operation", "Value / rule", "Fitting / selection data"],
          rows)],
        note="Values are read from the code that produced the results (experiments.RECIPE, "
             "sae.train_sae, concept_erasure.LeaceFitter, audit). The K sweep tests exactly "
             f"{kgrid}; no K = 7 was run. The greedy selector of earlier drafts is not used.")


@exhibit("Figure A1", "Fitting, ID audit and OOD evaluation", APPX)
def figure_a1():
    plt = _plt()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    fit, id_, oo = "#eef4fb", "#eef8f3", "#fdf1ea"
    nodes = {                                   # x, y, w, h, label, colour
        1: (6, 38, 20, 9, "(1) Donor training\ndocuments", fit),
        2: (38, 38, 22, 9, "(2) Fit and freeze T", fit),
        3: (1, 20, 19, 9, "(3) Donor held-out\ndocuments", id_),
        4: (26, 20, 19, 9, "(4) Frozen outputs\nT(X_ID)", id_),
        5: (14, 2, 38, 9, "(5) Independent probe train/test folds\n+ ID task scoring", id_),
        6: (55, 20, 19, 9, "(6) Independent target\ndocuments (MTEB)", oo),
        7: (80, 20, 19, 9, "(7) Frozen outputs\nT(X_OOD) + scoring", oo),
    }
    for x, y, w, h, label, fc in nodes.values():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=MUTED, lw=0.8))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7, color=INK)

    def arrow(p, q, label=None, ls="-", dx=1.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=INK,
                                     lw=0.9, ls=ls, shrinkA=2, shrinkB=2))
        if label:
            ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + 0.6, label, fontsize=6.3,
                    color=MUTED, va="bottom")
    arrow((26.6, 42.5), (37.4, 42.5))
    arrow((20.6, 24.5), (25.4, 24.5))
    arrow((74.6, 24.5), (79.4, 24.5))
    arrow((35.5, 19.4), (35.5, 11.6))
    arrow((45, 37.4), (38, 29.6), "T", ls="--", dx=-4.5)
    arrow((56, 37.4), (86, 29.6), "the same T", ls="--")
    ax.text(62.5, 47.5, "(2) SAE, pseudo-grouping, selection and the affine\nmap all fit on "
            "training documents; PCA-1 uses the\nsame portion; only metadata-LEACE sees "
            "training\nnuisance labels.", fontsize=6.2, va="top", color=INK)
    ax.text(54, 6.5, "(5) Nuisance labels train and score audit probes\nwithin the held-out "
            "portion; event pairs / CAP topics\nscore ID tasks. The retrieval index is the full "
            "\ntransformed corpus; nothing flows back into fitting.", fontsize=6.2,
            va="center", color=INK)
    ax.text(55, 17.5, "(7) Queries and documents (or both STS sentences) are\ntransformed by "
            "the same map; no target refitting.", fontsize=6.2, va="top", color=INK)
    return save_fig(fig, "figureA1_protocol", APPX)


@exhibit("Table A4", "Evaluation-regime inventory", APPX)
def table_a4():
    ids = {k: id_result(k) for k in R.ID_CORPORA}
    mini = {k: id_result(k, "mini") for k in ("scotus_oyez_lexis", "cap")}
    sem = seed_desc(ids["semeval"]) if ids["semeval"] else "3 seeds × 5 outer folds"
    held = nseeds([ids[k] for k in ("swiltra", "bills_tweets", "scotus_oyez_lexis")])
    held = "5 seeds" if held.startswith("NR") else held
    cap = nseeds([ids["cap"], mini["cap"]])
    cap = "3 seeds" if cap.startswith("NR") else cap

    def runs(folder, stems):
        found = [raw(folder, f"{folder}_{k}_e5-small.json") if folder != "protocol"
                 else raw("protocol", f"protocol_{k}_e5-small.json") for k in stems]
        n = nseeds(found)
        return "3 seeds" if n.startswith("NR") else n
    sweeps = nseeds([raw(f, f"{f}_{k}_e5-small.json") for f in
                     ("budget", "mechanism", "pseudo_groups", "score_function") for k in R.ORIGINAL])
    sweeps = "3 seeds" if sweeps.startswith("NR") else sweeps
    size = runs("corpus_size", ("swiltra", "bills_tweets"))
    proto = runs("protocol", R.ORIGINAL)
    summ = ood_summary()
    ood_n = (f"{len(summ['settings'][summ['order'][0]]['seeds'])} seeds" if summ else "3 seeds")
    grid_n = nseeds(list(encoder_cells().values()))
    grid_n = "3 seeds" if grid_n.startswith("NR") else grid_n
    alts = nseeds([raw("alternatives", f"alternatives_{k}_e5-small.json") for k in R.ID_CORPORA])
    alts = "3 seeds" if alts.startswith("NR") else alts
    gk_n = nseeds([raw("alternatives", f"generous_k_{k}_{e}.json") for k in R.ID_CORPORA
                   for e in R.ENCODER_STUDY])
    gk_n = "3 seeds" if gk_n.startswith("NR") else gk_n
    rows = [
        ["Headline SemEval", "SemEval / E5-small", "5 outer folds (4/5 of documents)",
         "held-out fold: audit + retrieval + graded similarity", sem,
         "fold scores averaged within a seed"],
        ["Headline SwiLTra / Bills–Tweets / SCOTUS", "3 corpora / E5-small", "50/50 outer holdout",
         "held-out half: audit (+ retrieval if paired)", held, "primary ID evidence"],
        ["New CAP", "CAP / E5-small, MiniLM", "50/50 outer holdout",
         "held-out half: source audit"
         + (" (linear; GB and MLP on E5-small)" if get(ids["cap"], "raw", "gb_auc") is not None
            else "") + " + 21-topic k-means", cap, "primary ID evidence"],
        ["MiniLM source ID", "SCOTUS, CAP / MiniLM", "50/50 outer holdout",
         "held-out half: audit (+ retrieval on SCOTUS)",
         seed_desc(mini["scotus_oyez_lexis"]) if mini["scotus_oyez_lexis"] else "5 seeds",
         "second-encoder robustness"],
        ["Ten-encoder study", "4 corpora + CAP / 10 encoders", "registry split (SemEval 5-fold)",
         "held-out documents, revised audit", grid_n,
         "refitting robustness across encoders; not frozen transfer"],
        ["K / score / mechanism / N sweeps", "4 corpora / E5-small", "registry split",
         "held-out documents, revised audit", sweeps, "sensitivity; one SAE per split shared "
         "across variants"],
        ["Erasure alternatives / k-means group count", "5 corpora / E5-small",
         "New Table 2's splits, seeds 0–2", "held-out documents: linear, GB and MLP audit "
         "(+ retrieval / topics)", alts, "label-free baselines and controls on the same fits"],
        ["Generous-K confirmatory test", "5 corpora / E5-small (development) and 9 held-out "
         "encoders", "the ten-encoder study's splits, seeds 0–2", "held-out documents: linear "
         "audit (GB, MLP on three maps) + retrieval / topics; frozen transfer on the OOD donors",
         gk_n, "confirmatory: K fixed on E5-small before the held-out runs"],
        ["Nuisance geometry diagnostic", "5 corpora / 10 encoders",
         "the ten-encoder study's splits, seeds 0–2", "training documents only; nothing fitted "
         "or erased", gk_n, "descriptive; uses the labels to locate the nuisance"],
        ["CAP topics and source balance", "CAP / 10 encoders (source balance: E5-small, MiniLM)",
         "the ten-encoder study's splits and the saved ID maps, seeds 0–2", "held-out CAP "
         "documents: 21-topic k-means, and separately on source-balanced and skewed topics",
         gk_n, "topic utility across encoders; source-topic entanglement"],
        ["Corpus-size sweep", "SwiLTra, Bills–Tweets / E5-small",
         "50/50 within a size-fixed subsample", "held-out half: logistic, MLP, GB", size,
         "sample-size sensitivity"],
        ["Legacy protocol diagnostic", "4 corpora / E5-small", "one 50/50 split: all documents "
         "(links withheld) vs training half", "same audit on each arm's scored documents",
         proto, "protocol diagnostic, not headline evidence"],
        ["Descriptive latent / subspace fits", "5 corpora / E5-small", "full corpus, seed 0",
         "none (descriptive)", "1 fit", "no held-out performance claim"],
        ["OOD transfer", "5 donor/encoder settings", "the ID training split of seeds 0–2, "
         "frozen", "6 MTEB targets (nDCG@10, Spearman)", ood_n, "frozen transfer"],
    ]
    # The exhibits each regime produces, in the order of the plan. Table B3 gains a CAP
    # block only once CAP carries the nonlinear probes.
    cap_nonlinear = get(ids["cap"], "raw", "gb_auc") is not None
    # New Table 2's k-means row is the alternatives run's until run_id_kmeans has run.
    km_from_alt = any(id_kmeans(k, ids[k])[1] == "alternatives" for k in R.ID_CORPORA)
    produced = {
        "Headline SemEval": ("table2_id_comparison.tex", "tableB1_paired_tasks.tex",
                             "tableB3_nonlinear.tex", "tableB4_per_class.tex"),
        "Headline SwiLTra / Bills–Tweets / SCOTUS": ("table2_id_comparison.tex",
                                                     "tableB1_paired_tasks.tex",
                                                     "tableB3_nonlinear.tex"),
        "New CAP": (("table2_id_comparison.tex", "tableB2_cap.tex")
                    + (("tableB3_nonlinear.tex",) if cap_nonlinear else ())
                    + ("tableB4_per_class.tex",)),
        "MiniLM source ID": ("tableB1_paired_tasks.tex", "tableB2_cap.tex"),
        "Ten-encoder study": ("tableB5_encoders.tex", "tableB6_encoder_ranks.tex",
                              "figureD6_pc1_dominance.pdf"),
        "K / score / mechanism / N sweeps": ("figureD1_pseudo_groups.pdf",
                                             "tableD1_score_function.tex",
                                             "table3_mechanisms.tex", "tableD3_mechanisms.tex",
                                             "figure2_erased_directions.pdf",
                                             "figureD7_budget.pdf", "tableD2_budget.tex"),
        "Erasure alternatives / k-means group count": (
            (("table2_id_comparison.tex",) if km_from_alt else ())
            + ("table3_mechanisms.tex", "figure2_erased_directions.pdf",
               "tableD9_alternatives.tex", "tableD10_kmeans_count.tex",
               "tableD13_selector_overlap.tex")),
        "Corpus-size sweep": ("figureD2_corpus_size.pdf", "tableD4_corpus_size.tex"),
        "Legacy protocol diagnostic": ("tableD5_protocol.tex",),
        "Descriptive latent / subspace fits": ("tableD6_capture.tex",
                                               "figureD3_principal_angles.pdf",
                                               "tableD7_latents.tex", "tableD8_examples.tex",
                                               "figureD4_latent_distributions.pdf",
                                               "figureD5_pca_panels.pdf"),
        "OOD transfer": ("figure1_ood_retrieval.pdf", "tableC1_ood_inventory.tex",
                         "tableC2_ood_scores.tex", "tableC3_ood_changes.tex",
                         "tableC4_ood_ranks.tex", "figureC1_ood_sts.pdf"),
        "Generous-K confirmatory test": ("tableD11_generous_k.tex",),
        "Nuisance geometry diagnostic": ("tableD12_geometry.tex", "figureD8_geometry.pdf"),
        "CAP topics and source balance": ("tableD14_cap_topics.tex",
                                          "tableD15_cap_entanglement.tex"),
    }
    for row in rows:
        row[0] = "\n".join([row[0]] + [f"`{f}`" for f in produced[row[0]]])
    return write_table(
        "tableA4_regimes", APPX, "Evaluation-regime inventory.",
        [(None, ["Experiment (resulting exhibits)", "Corpora / encoders", "Eraser-training split",
                 "Evaluation documents / task", "Run seeds", "Interpretation"], rows)],
        wide=True, colspec="p{5.0cm}p{1.9cm}p{2.0cm}p{2.2cm}p{1.3cm}p{2.1cm}",
        note="Under each experiment, the file names of the tables (.tex) and figures (.pdf) it "
             "produces. The same method can carry different numbers across rows because the "
             "regimes differ; numbers are compared only within a row.")


# =========================================================================== #
# Appendix B
# =========================================================================== #
@exhibit("Table B1", "Complete paired-task results", APPX)
def table_b1():
    settings = [("e5-small", "semeval"), ("e5-small", "swiltra"),
                ("e5-small", "scotus_oyez_lexis"), ("mini", "scotus_oyez_lexis")]
    res = {s: id_result(s[1], s[0]) for s in settings}
    if not any(res.values()):
        raise Missing("id/id_<corpus>_<encoder>.json")
    km = {s: id_kmeans(s[1], res[s], s[0])[0] for s in settings}
    view = {s: (res[s] if res[s] is None or km[s] is None else
                {**res[s], "agg": {**res[s]["agg"], "kmeans_leace": km[s]["agg"]["kmeans_leace"]}})
            for s in settings}
    a = []
    for s in settings:
        if a:
            a.append(SEP)
        for c in TABLE2_ROWS:
            if c == "kmeans_leace" and get(view[s], c, "linear_auc") is None:
                continue
            a.append([ENCODER[s[0]], CORPUS[s[1]], LABEL[c], pm(get(view[s], c, "linear_auc")),
                      pm(get(view[s], c, "recall@1")), pm(get(view[s], c, "recall@10"))])
    sem = view[settings[0]]
    b = [[LABEL[c], pm(get(sem, c, "gold_sim_spearman")), seed_desc(res[settings[0]])]
         for c in TABLE2_ROWS
         if res[settings[0]] and get(sem, c, "gold_sim_spearman") is not None]
    return write_table(
        "tableB1_paired_tasks", APPX, "Complete paired-task results (mean ± SD across seeds).",
        [("Panel A — Nuisance leakage and paired retrieval",
          ["Encoder", "Corpus", "Method", "Linear AUC", "Recall@1", "Recall@10"], a),
         ("Panel B — SemEval graded similarity (E5-small)",
          ["Method", "Spearman (pair cosine vs. human rating)", "Seeds"], b)],
        note="Linear AUC: the independent held-out audit of New Table 2 for each corpus's "
             "nuisance (language on SemEval and SwiLTra, source on SCOTUS; 0.5: chance). The "
             "E5-small values repeat New Table 2; the MiniLM/SCOTUS values appear only here. "
             "Retrieval: held-out pairs as queries, both directions, against the full "
             "transformed corpus, self excluded. Graded similarity: Spearman correlation "
             "between each held-out pair's cosine similarity and its human rating, in the "
             "rating's original orientation (higher = more similar), pooled over the five "
             "outer folds of a seed. k-means LEACE appears wherever `revised.run_id_kmeans` "
             "has run: E5-small, and MiniLM once `class_combinations.ipynb` has run it.",
        wide=True)


@exhibit("Table B2", "Complete CAP topic/source evaluation", APPX)
def table_b2():
    res = {e: id_result("cap", e) for e in ("e5-small", "mini")}
    if not any(res.values()):
        raise Missing("id/id_cap_<encoder>.json")
    km = {e: id_kmeans("cap", res[e], e)[0] for e in res}
    view = {e: (res[e] if res[e] is None or km[e] is None else
                {**res[e], "agg": {**res[e]["agg"], "kmeans_leace": km[e]["agg"]["kmeans_leace"]}})
            for e in res}
    rows = []
    for e in ("e5-small", "mini"):
        if rows:
            rows.append(SEP)
        for c in TABLE2_ROWS:
            if c == "kmeans_leace" and get(view[e], c, "linear_auc") is None:
                continue
            rows.append([ENCODER[e], LABEL[c]] + [pm(get(view[e], c, m)) for m in
                         ("linear_auc", "gb_auc", "mlp_auc", "topic_purity", "topic_ari")])
    no_probe = [ENCODER[e] for e in res if res[e] and get(res[e], "raw", "gb_auc") is None]
    return write_table(
        "tableB2_cap", APPX, f"Complete CAP topic/source evaluation (mean ± SD over {nseeds(res)}).",
        [(None, ["Encoder", "Method", "Linear\n(logistic)", "Nonlinear\n(GB)", "Nonlinear\n(MLP)",
                 "Purity", "ARI"], rows,
          [(None, 2), ("Source AUC (lower is better)", 3), ("Topics (higher is better)", 2)])],
        wide=True,
        note="Source AUC: macro one-vs-rest over bills/orders/news, independent held-out audit "
             "(0.5: chance), with the linear, GB and MLP probes of New Table 2 on the same folds. "
             "Topic purity (cluster-size weighted) and ARI: k-means with K_topic = 21, n_init = "
             "20, on the transformed held-out documents; the topic labels never enter a fit."
             + " k-means LEACE appears wherever `revised.run_id_kmeans` has run, which is E5-small."
             + (f" NR: not evaluated in this run (no nonlinear probes on {', '.join(no_probe)})."
                if no_probe else ""))


@exhibit("Table B3", "Nonlinear nuisance-probe results", APPX)
def table_b3():
    res = {k: id_result(k) for k in R.ORIGINAL}
    if not any(res.values()):
        raise Missing("id/id_<corpus>_e5-small.json")
    rows = []
    for k in R.ORIGINAL:
        if rows:
            rows.append(SEP)
        for c in COND:
            rows.append([CORPUS[k], LABEL[c], pm(get(res[k], c, "gb_auc")),
                         pm(get(res[k], c, "mlp_auc"))])
    cap = id_result("cap")
    if cap and get(cap, "raw", "gb_auc"):
        rows.append(SEP)
        rows += [["CAP", LABEL[c], pm(get(cap, c, "gb_auc")), pm(get(cap, c, "mlp_auc"))]
                 for c in COND]
    return write_table(
        "tableB3_nonlinear", APPX, "Nonlinear nuisance-probe results, E5-small.",
        [(None, ["Corpus", "Method", "Gradient-boosted AUC", "MLP AUC"], rows)],
        note="Ordinary unflipped AUC on the same audit folds as the linear probe (0.5: "
             "chance), mean ± SD across seeds. A lower linear AUC does not imply the absence "
             "of nonlinearly recoverable nuisance information. NR: not evaluated.")


@exhibit("Table B4", "Per-class linear AUC", APPX)
def table_b4():
    rows = []
    for key, names in (("semeval", None), ("cap", CAP_CLASS)):
        r = id_result(key)
        if r is None:
            continue
        st = r["corpus_stats"]
        prior = {c: n / sum(st["class_counts"]) for c, n in zip(st["classes"], st["class_counts"])}
        if rows:
            rows.append(SEP)
        for cls in ordered(key, st["classes"]):
            rows.append([CORPUS[key], (names or {}).get(cls, cls), num(prior[cls])]
                        + [pm(get(r, c, "linear_auc_per_class").get(cls)
                              if get(r, c, "linear_auc_per_class") else None) for c in COND])
        rows.append([CORPUS[key], "Macro", "—"] + [pm(get(r, c, "linear_auc")) for c in COND])
    if not rows:
        raise Missing("id/id_semeval_e5-small.json, id/id_cap_e5-small.json")
    return write_table(
        "tableB4_per_class", APPX, "Per-class one-vs-rest linear AUC, E5-small.",
        [(None, ["Corpus", "Class", "Class prior", *[f"{LABEL[c]} AUC" for c in COND]], rows)],
        note="Mean ± SD across seeds under the protocol of New Table 2. The macro row is the "
             "equally weighted mean of the class AUCs; the full-corpus class prior is "
             "descriptive and is not a weight.")


def encoder_cells(corpora=R.ORIGINAL) -> dict:
    """``{(corpus, encoder): result}`` for the ten-encoder study cells on disk. The plan's
    study covers the four original corpora; CAP's cells (section 1k) are read by the
    exhibits that ask for them."""
    cells = {}
    for k in corpora:
        for e in R.ENCODER_STUDY:
            r = raw("encoders", f"encoder_{k}_{e}.json")
            if r is not None:
                cells[(k, e)] = r
    return cells


@exhibit("Table B5", "Absolute linear AUC across the ten encoders", APPX)
def table_b5():
    cells = encoder_cells(R.ID_CORPORA)                  # CAP's cells included (section 1k)
    if not cells:
        raise Missing("encoders/encoder_<corpus>_<encoder>.json")
    first = next(iter(cells.values()))
    # PCA removal, at every rank the cells carry, is Table B6.
    panels = []
    for c in [c for c in R.ENCODER_CONDITIONS if c in first["conditions"]]:
        rows = [[ENCODER.get(e, e)] + [pm(get(cells.get((k, e)), c, "linear_auc"))
                                       for k in R.ID_CORPORA] for e in R.ENCODER_STUDY]
        label = LABEL.get(c) or "PCA-" + c.replace("pca_top", "")
        panels.append((label, ["Encoder"] + [f"{CORPUS[k]} AUC" for k in R.ID_CORPORA], rows))
    return write_table(
        "tableB5_encoders", APPX, "Absolute linear AUC across the ten encoders.", panels, wide=True,
        note=f"Mean ± SD over {nseeds(cells)} per cell, under the audit of New Table 2 and "
             "each corpus's registry split (SemEval five outer folds per seed). Every cell "
             "refits every eraser on that encoder's embeddings: this is robustness of "
             "refitting across encoders, not frozen transfer. PCA removal on the same cells, "
             "rank by rank, is Table B6. NR: not run.")


@exhibit("Table B6", "PCA removal rank against SAE-LEACE across the ten encoders", APPX)
def table_b6():
    import alternatives as A
    try:
        return A.table_encoder_ranks(corpora=R.ID_CORPORA, stem="tableB6_encoder_ranks",
                                     where=APPX, wide=True)
    except FileNotFoundError as exc:
        raise Missing(exc) from None


# =========================================================================== #
# Appendix C
# =========================================================================== #
def _size(sz: dict) -> str:
    if "pairs" in sz:
        return f"{intc(sz['pairs'])} rated sentence pairs"
    return f"{intc(sz['queries'])} queries / {intc(sz['documents'])} documents / " \
           f"{intc(sz['judgements'])} judgements"


@exhibit("Table C1", "Frozen-transfer evaluation inventory", APPX)
def table_c1():
    s = req(ood_summary(), "ood/ood_<encoder>_<donor>.json")
    v0 = s["settings"][s["order"][0]]
    a = [[t, v0["tasks"][t]["revision"][:10], "test; English", _size(v0["tasks"][t]["size"]),
          "nDCG@10" if v0["tasks"][t]["kind"] == "retrieval" else "cosine Spearman"]
         for t in ood.TARGETS]
    b = []
    for key in OOD_KEYS:
        if key not in s["settings"]:
            continue
        v = s["settings"][key]
        b.append([ENCODER[v["encoder"]], CORPUS.get(v["donor"], v["donor"]),
                  ", ".join(v["split_ids"]), ", ".join(map(str, v["seeds"])), "all six targets"])
    return write_table(
        "tableC1_ood_inventory", APPX, "Frozen-transfer evaluation inventory.",
        [("Panel A — Targets", ["Task", "Dataset revision", "Split; language", "Evaluated size",
                                "Score"], a),
         ("Panel B — Donor/encoder settings", ["Encoder", "Donor mixture",
                                               "Donor training-split IDs", "Run seeds",
                                               "Transformed targets"], b)],
        note=f"MTEB {ood.MTEB_VERSION} task definitions, pinned dataset revisions. Split IDs "
             "fingerprint each seed's donor training documents (SHA-1 prefix); the maps are "
             "the ID fits of those seeds, frozen. No donor or target was chosen after seeing "
             "scores, and no donor/target text overlaps.")


def build_table_c2(s, suffix: str = "") -> list:
    trs = _trs(s)
    panels = []
    for key in OOD_KEYS:
        if key not in s["settings"]:
            continue
        v, sc = s["settings"][key], s["settings"][key]["scores"]
        rows = [[v["tasks"][t]["label"], "nDCG@10", num(sc["raw"][t]["mean"])]
                + [pm(sc[c][t]) for c in trs] for t in _task_order(v)]
        rows.append(["Legal macro-average", "Mean nDCG@10", num(sc["raw"]["legal_macro"]["mean"])]
                    + [pm(sc[c]["legal_macro"]) for c in trs])
        rows.append(SEP)
        rows += [[v["tasks"][t]["label"], "Spearman", num(sc["raw"][t]["mean"])]
                 + [pm(sc[c][t]) for c in trs] for t in v["sts"]]
        panels.append((ood_title(key), ["Target task", "Metric", "Raw",
                                         *[_lab(s, c) for c in trs]], rows))
    return write_table(
        "tableC2_ood_scores" + suffix, APPX, "All OOD task scores.", panels, wide=len(trs) > 3,
        note="Absolute scores, mean ± SD across the three frozen donor fits. Raw needs no fit "
             "and repeats across donor panels of one encoder. Legal macro-average: equal-weight "
             "mean of the five legal nDCG@10 scores within each seed, then mean ± SD; STS is "
             "not included in it." + _ood_extra_note(s))


@exhibit("Table C2", "All OOD task scores", APPX)
def table_c2():
    return build_table_c2(req(ood_summary(extra=("kmeans",)), "ood/ood_<encoder>_<donor>.json"))


def build_table_c3(s, suffix: str = "") -> list:
    a, b = [], []
    for key in OOD_KEYS:
        if key not in s["settings"]:
            continue
        v = s["settings"][key]
        if a:
            a.append(SEP)
            b.append(SEP)
        for c in _trs(s):
            ch = v["changes"][c]
            head = [ENCODER[v["encoder"]], CORPUS.get(v["donor"], v["donor"]), _lab(s, c)]
            w = ch["worst_task"]
            a.append(head + [signed(ch["legal_macro"]["delta"]), ci(ch["legal_macro"]["ci"]),
                             v["tasks"][w["task"]]["label"], signed(w["delta"])])
            for t in v["sts"]:
                b.append(head + [signed(ch[t]["delta"]), ci(ch[t]["ci"])])
    return write_table(
        "tableC3_ood_changes" + suffix, APPX, "OOD change summaries.",
        [("Panel A — Legal retrieval", ["Encoder", "Donor", "Transform", "Mean legal\nΔnDCG@10",
                                        "95% CI", "Smallest-change\ntask", "Task Δ"], a),
         ("Panel B — STSBenchmark", ["Encoder", "Donor", "Transform", "ΔSpearman", "95% CI"], b)],
        wide=True,
        note=f"Changes in the seed-averaged score (transformed − Raw). Paired bootstrap, "
             f"{s['n_boot']:,} resamples: legal macro intervals resample queries within each "
             "task (the two AILA tasks share one query set and are resampled jointly), "
             "recompute each task's change and average the five; STS intervals recompute both "
             "Spearman correlations on resampled pairs. Intervals are conditional on the three "
             "fitted maps (fit-seed variability: Table C2). The smallest-change task is "
             "descriptive, not a prespecified test; if every change is positive it is the "
             "smallest positive one." + _ood_extra_note(s))


@exhibit("Table C3", "OOD change summaries", APPX)
def table_c3():
    return build_table_c3(req(ood_summary(extra=("kmeans",)), "ood/ood_<encoder>_<donor>.json"))


#: Table C4's columns: PCA removal at every transferred rank, the variance-matched PCA
#: map, and SAE-LEACE for reference.
C4_MAPS = ("pca_top1", "pca_top2", "pca_top3", "pca_top4", "pca_top8", "pca_top16",
           "pca_top32", "pca_matched_var", "sae_leace")


@exhibit("Table C4", "Frozen-transfer cost by erased rank", APPX)
def table_c4():
    s = req(ood_summary(extra=("ranks",)), "ood/ood_<encoder>_<donor>.json")
    trs = _trs(s)

    def change(v, c, task):
        if c not in trs:
            return "NR"
        ch = v["changes"][c][task]
        return signed(ch["delta"]) + ("*" if ch["ci"][0] > 0 or ch["ci"][1] < 0 else "")
    a, b, matched = [], [], []
    for key in OOD_KEYS:
        if key not in s["settings"]:
            continue
        v = s["settings"][key]
        a.append([ood_title(key)] + [change(v, c, "legal_macro") for c in C4_MAPS])
        b.append([ood_title(key)] + [change(v, c, v["sts"][0]) for c in C4_MAPS])
        enc, don = key.split("/")
        f = raw("alternatives", f"ood_ranks_{enc}_{don}.json") or {}
        ranks = sorted({p["fit"]["matched_rank"] for p in f.get("per_seed", [])
                        if "matched_rank" in p.get("fit", {})})
        if ranks:
            matched.append(f"{ood_title(key)} {ranks[0]}"
                           + (f"–{ranks[-1]}" if len(ranks) > 1 else ""))
    header = (["Donor setting"] + [f"r = {c[7:]}" for c in C4_MAPS[:7]]
              + ["Variance-\nmatched", "SAE-\nLEACE"])
    groups = [(None, 1), ("PCA removal", 8), (None, 1)]
    return write_table(
        "tableC4_ood_ranks", APPX, "Frozen-transfer cost by erased rank.",
        [("Panel A — Legal retrieval: mean ΔnDCG@10", header, a, groups),
         ("Panel B — STSBenchmark: ΔSpearman", header, b, groups)], wide=True,
        note="Changes in the seed-averaged score (transformed − Raw). *: the paired bootstrap "
             f"95% interval ({s['n_boot']:,} resamples, as in Table C3) excludes zero. Every PCA "
             "map is refit on the donor's ID training split and frozen; r = 1 is the plan's PCA-1. "
             "The variance-matched map removes, per seed, the fewest principal directions whose "
             "share of the training variance reaches the share removed by that seed's frozen "
             "SAE-LEACE map"
             + (f" (rank by setting: {'; '.join(matched)})" if matched else "")
             + ". SAE-LEACE repeats Table C3. NR: not run.")


def build_figure_c1(s, suffix: str = "") -> list:
    plt = _plt()
    from matplotlib.lines import Line2D
    keys = [k for k in OOD_KEYS if k in s["settings"]]
    sts = s["settings"][keys[0]]["sts"][0]
    trs = _trs(s)
    vals = [0.0] + [x for k in keys for c in trs
                    for x in (s["settings"][k]["changes"][c][sts]["delta"],
                              *s["settings"][k]["changes"][c][sts]["ci"])]
    xlim = _limits(vals)
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    y0 = np.arange(len(keys))[::-1].astype(float)
    off = _offsets(s)
    ax.axvline(0, color=INK, lw=0.8)
    for i, key in enumerate(keys):
        for c in trs:
            ch = s["settings"][key]["changes"][c][sts]
            y = y0[i] + off[c]
            col, mk = _style(s, c)
            ax.plot(ch["ci"], [y, y], color=col, lw=1.1)
            ax.plot(ch["delta"], y, marker=mk, color=col, ms=4.5, mec="white", mew=0.5,
                    ls="none")
    ax.set_yticks(y0)
    ax.set_yticklabels([ood_title(k) for k in keys])
    ax.set_xlim(*xlim)
    ax.set_xlabel("ΔSpearman on STSBenchmark (transformed − Raw)")
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Line2D([0], [0], color=_style(s, c)[0], marker=_style(s, c)[1],
                              mec="white", ms=5, label=_lab(s, c)) for c in trs],
              loc="lower left", bbox_to_anchor=(1.0, 0.0), frameon=True)
    ax.set_title("Point: change in the seed-averaged cosine Spearman; bar: paired 95% bootstrap "
                 "interval", fontsize=7, color=MUTED, loc="left")
    fig.tight_layout()
    return save_fig(fig, "figureC1_ood_sts" + suffix, APPX)


@exhibit("Figure C1", "OOD semantic-similarity changes", APPX)
def figure_c1():
    return build_figure_c1(req(ood_summary(extra=("kmeans",)), "ood/ood_<encoder>_<donor>.json"))


# =========================================================================== #
# Appendix D
# =========================================================================== #
def _band(ax, xs, per_seed_values, color, label=None, marker="o"):
    arr = np.asarray(per_seed_values, float)
    m, s = arr.mean(axis=0), (arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros(arr.shape[1]))
    ax.errorbar(xs, m, yerr=s, color=color, marker=marker, ms=3.2, lw=1.1, capsize=2,
                label=label)
    return m


#: Figure D1 text sizes (pt, at drawn size), and its two reference lines: Metadata-LEACE
#: in a darker shade of its green, the evaluator's class count in red.
FIGD1_FONT = {"title": 11.5, "label": 11, "tick": 10, "legend": 10, "note": 10}
FIGD1_METADATA = dict(color="#1d6e19", ls="--", lw=1.7)
FIGD1_CLASSES = dict(color="#e31a1c", ls=(0, (1, 1.5)), lw=2.0)


@exhibit("Figure D1", "Pseudo-group-count sensitivity", APPX)
def figure_d1():
    res = {k: raw("pseudo_groups", f"pseudo_groups_{k}_e5-small.json") for k in R.ORIGINAL}
    if not any(res.values()):
        raise Missing("pseudo_groups/pseudo_groups_<corpus>_e5-small.json")
    plt = _plt()
    from collections import Counter
    from matplotlib.lines import Line2D
    # Unpaired corpora have no retrieval panel, so they go last and their empty
    # third cell carries the legend.
    keys = ([k for k in R.ORIGINAL if res[k] and res[k]["paired"]]
            + [k for k in R.ORIGINAL if res[k] and not res[k]["paired"]]
            + [k for k in R.ORIGINAL if not res[k]])
    fs = FIGD1_FONT
    fig, axes = plt.subplots(len(keys), 3, figsize=(7.7, 2.8 * len(keys)), squeeze=False)
    has_data = np.zeros((len(keys), 3), dtype=bool)
    for i, k in enumerate(keys):
        r = res[k]
        if r is None:
            for j in range(3):
                _nr(axes[i, j])
            axes[i, 0].set_title(CORPUS[k], loc="left", fontsize=fs["title"])
            continue
        ks, ps = r["ks"], r["per_seed"]
        _band(axes[i, 0], ks, [[p["silhouette"][str(K)] for K in ks] for p in ps], INK)
        counts = Counter(K for p in ps for K in p["selected_k"])
        n_fits = sum(counts.values())
        parts = [f"{K} ({n}/{n_fits})" for K, n in sorted(counts.items())]
        # Wrapped to the panel's width: SemEval picks four.
        note = ",\n".join(["selected K: " + parts[0]]
                          + [", ".join(parts[n:n + 2]) for n in range(1, len(parts), 2)])
        # Under the panel: anchored to the bottom of its x axis (ticks, and the label
        # where there is one), so it clears both.
        axes[i, 0].annotate(note, (0.5, 0), xycoords=lambda rend, a=axes[i, 0]:
                            a.xaxis.get_tightbbox(rend), xytext=(0, -3),
                            textcoords="offset points", ha="center", va="top",
                            fontsize=fs["note"], fontweight="bold", color="#262626",
                            linespacing=1.25)
        for K in counts:
            m = np.mean([p["silhouette"][str(K)] for p in ps])
            axes[i, 0].plot(K, m, "o", mfc="none", mec=COLOR["sae_leace"], ms=8, mew=1.3)
        has_data[i, 0] = True
        for j, key in ((1, "linear_auc"), (2, "recall@10")):
            ax = axes[i, j]
            if key == "recall@10" and not r["paired"]:
                ax.axis("off")                       # the legend goes here instead
                continue
            _band(ax, ks, [[p["points"][str(K)][key] for K in ks] for p in ps],
                  COLOR["sae_leace"], "SAE-LEACE")
            ax.axhline(np.mean([p["metadata_reference"][key] for p in ps]),
                       label="Metadata-LEACE", **FIGD1_METADATA)
            if key == "linear_auc":
                ax.axhline(0.5, color=MUTED, ls=":", lw=0.7)
            ax.axvline(r["n_classes"], **FIGD1_CLASSES)
            has_data[i, j] = True
        for j in range(3):
            if has_data[i, j]:
                axes[i, j].set_xticks(ks)
                axes[i, j].tick_params(axis="both", labelsize=fs["tick"])
        axes[i, 0].set_ylabel(f"{CORPUS[k]}\nsilhouette (train)", fontsize=fs["label"])
    for j, t in enumerate(("Training silhouette", "Held-out linear AUC", "Held-out R@10")):
        axes[0, j].set_title(t, fontsize=fs["title"])
        rows = np.flatnonzero(has_data[:, j])        # label the lowest panel that has data
        if len(rows):
            axes[rows[-1], j].set_xlabel("K_proxy", fontsize=fs["label"])
    handles = [Line2D([0], [0], color=COLOR["sae_leace"], marker="o", ms=4.5, lw=1.4),
               Line2D([0], [0], **FIGD1_METADATA),
               Line2D([0], [0], **FIGD1_CLASSES),
               Line2D([0], [0], mfc="none", mec=COLOR["sae_leace"], marker="o", ls="none",
                      ms=10, mew=1.3)]
    labels = ["SAE-LEACE\n(mean ± 1 SD)", "Metadata-LEACE\n(K-independent)",
              "true # classes\n(evaluator only)", "K chosen by the\ntraining silhouette"]
    free = [(i, j) for i in range(len(keys)) for j in range(3) if not has_data[i, j]]
    if free:
        # The legend takes the free cell's panel box, framed by that panel's own
        # spines, so its edges line up with the panels above it and beside it.
        i, j = free[-1]
        ax = axes[i, j]
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(plt.rcParams["legend.edgecolor"])
        ax.legend(handles, labels, loc="center", frameon=False, fontsize=fs["legend"],
                  handlelength=1.6, handletextpad=0.5, labelspacing=0.55, borderpad=0.4)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02),
                   fontsize=fs["legend"])
    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    return save_fig(fig, "figureD1_pseudo_groups", APPX)


@exhibit("Table D1", "Scoring-function comparison and selected-set agreement", APPX)
def table_d1():
    res = {k: raw("score_function", f"score_function_{k}_e5-small.json") for k in R.ORIGINAL}
    if not any(res.values()):
        raise Missing("score_function/score_function_<corpus>_e5-small.json")
    a, b = [], []
    for k in R.ORIGINAL:
        r = res[k]
        if r is None:
            continue
        if a:
            a.append(SEP)
        for v in R.SCORE_VARIANTS:
            ag = r["agg"][v]
            a.append([CORPUS[k], SCORE[v], str(r["top_n"]), pm(ag.get("linear_auc")),
                      pm(ag.get("gb_auc")), pm(ag.get("recall@10")) if r["paired"] else "N/A"])
        j = r["jaccard"]
        b.append([CORPUS[k], num(j["mi|nmi"]), num(j["mi|corr_ratio"]), num(j["nmi|corr_ratio"])])
    return write_table(
        "tableD1_score_function", APPX, "Scoring-function comparison and selected-set agreement.",
        [(f"Panel A — Outcomes (E5-small, mean ± SD over {nseeds(res)})",
          ["Corpus", "Score", "N", "Linear AUC", "GB AUC", "R@10"], a),
         ("Panel B — Jaccard overlap of the selected latent sets",
          ["Corpus", "J(MI, NMI)", "J(MI, η²)", "J(NMI, η²)"], b)],
        note="Same SAE, pseudo-grouping and frequency filter for every rule. Normalized MI is "
             "I(F_j; G)/H(F_j); η² is the correlation ratio of the continuous activation "
             "across pseudo-groups. Panel B averages the Jaccard index of the N = 32 selected "
             "sets over fits; it is a selector-agreement diagnostic, not evidence of semantic "
             "preservation.")


@exhibit("Table D2", "Full N-sweep values", APPX)
def table_d2():
    res = {k: raw("budget", f"budget_{k}_e5-small.json") for k in R.ORIGINAL}
    if not any(res.values()):
        raise Missing("budget/budget_<corpus>_e5-small.json")
    panels, refs = [], []
    for k in R.ORIGINAL:
        r = res[k]
        if r is None:
            continue
        rows = [[str(n), pm(r["curve"][str(n)]["linear_auc"]),
                 pm(r["curve"][str(n)]["recall@10"]) if r["paired"] else "N/A"]
                for n in r["grid"]]
        panels.append((f"{CORPUS[k]} — E5-small, {len(r['seeds'])} seeds",
                       ["N", "Linear AUC", "R@10"], rows))
        ref = r["references"]
        refs.append(f"{CORPUS[k]}: " + ", ".join(
            f"{LABEL[c]} AUC {num(ref[c]['linear_auc']['mean'])}"
            + (f" / R@10 {num(ref[c]['recall@10']['mean'])}" if r["paired"] else "")
            for c in ("raw", "pca_top1", "metadata_leace")))
    return write_table(
        "tableD2_budget", APPX, "Full N-sweep values (SAE direction-LEACE, nested top-N prefixes).",
        panels,
        note="Mean ± SD over seeds; one SAE and one MI ranking per split serve every N. "
             "Reference scores on the same splits and seeds (not assigned an N) — "
             + "; ".join(refs) + ".")


@exhibit("Table D3", "Full controlled intervention results", APPX)
def table_d3():
    rows, res, alt, ok, drift = _intervention_rows(R.ORIGINAL)
    same = all(p["same_selected_set"] for r in res.values() if r for p in r["per_seed"])
    no_probe = [CORPUS[k] for k in R.ORIGINAL
                if res[k] and not all("gb_auc" in res[k]["agg"][m] for m in R.MECHANISMS)]
    no_alt = [CORPUS[k] for k in R.ORIGINAL if k not in ok]
    unpaired = [CORPUS[k] for k in R.ORIGINAL if res[k] and not res[k]["paired"]]
    nr = ([f"no nonlinear probes on the first three interventions for {', '.join(no_probe)}"]
          if no_probe else []) + ([f"no erasure-alternatives rows on {', '.join(no_alt)}"]
                                  if no_alt else [])
    return write_table(
        "tableD3_mechanisms", APPX,
        "Full controlled intervention results, E5-small, mean ± SD over "
        f"{nseeds(list(res.values()) + [alt[k] for k in ok])}.",
        [(None, INTERVENTION_HEADER, rows, INTERVENTION_GROUPS)], wide=True, tabcolsep=2.8,
        note="New Table 3 on all four original corpora, with the same rows, fits and audit; the "
             "Bills–Tweets rows are the ones it does not show. Within each corpus and seed the "
             "first four rows share one SAE and selected set S (N = 32) (selected sets identical "
             f"across the three LEACE-type interventions in every fit: {'yes' if same else 'NO'}). "
             + (f"The last three rows come from the erasure-alternatives run, on the same splits "
                f"and seeds; its MI-selected Directions-LEACE reproduces the Directions-LEACE row "
                f"(largest per-seed difference in linear AUC or R@10: {max(drift[k] for k in ok):.3f}). "
                if ok else "")
             + (f"—: no paired retrieval task ({', '.join(unpaired)})." if unpaired else "")
             + (f" NR: not evaluated in this run ({'; '.join(nr)})." if nr else ""))


#: Figure D2 text sizes (pt, at drawn size).
FIGD2_FONT = {"title": 11.5, "label": 11, "tick": 10, "legend": 11}


@exhibit("Figure D2", "Linear and nonlinear AUC versus corpus size", APPX)
def figure_d2():
    res = {k: raw("corpus_size", f"corpus_size_{k}_e5-small.json") for k in ("swiltra", "bills_tweets")}
    if not any(res.values()):
        raise Missing("corpus_size/corpus_size_<corpus>_e5-small.json")
    plt = _plt()
    fs = FIGD2_FONT
    fig, axes = plt.subplots(2, 3, figsize=(7.7, 5.4), sharey=True)
    probes = (("linear_auc", "Logistic probe"), ("mlp_auc", "MLP probe"), ("gb_auc", "GB probe"))
    for i, k in enumerate(("swiltra", "bills_tweets")):
        r = res[k]
        for j, (key, title) in enumerate(probes):
            ax = axes[i, j]
            if r is None:
                _nr(ax)
                continue
            sizes = [str(s) for s in r["sizes"]]
            xs = [r["variants"][s]["n_docs"] for s in sizes]
            for c in ("raw", "metadata_leace", "sae_leace"):
                vals = [[p["metrics"][c][key] for p in r["variants"][s]["per_seed"]] for s in sizes]
                _band(ax, xs, np.array(vals).T, COLOR[c], LABEL[c], MARKER[c])
            ax.axhline(0.5, color=MUTED, ls=":", lw=0.7)
            ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x:,}" for x in xs])
            ax.minorticks_off()
            ax.tick_params(axis="both", labelsize=fs["tick"])
            if i == 0:
                ax.set_title(title, fontsize=fs["title"])
        axes[i, 0].set_ylabel(f"{CORPUS[k]}\nAUC", fontsize=fs["label"])
    xlabel = fig.supxlabel("Documents in the subsample before the 50/50 split (log scale)",
                           fontsize=fs["label"])
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COLOR[c], marker=MARKER[c], ms=5.5, lw=1.5,
                      label=LABEL[c]) for c in ("raw", "metadata_leace", "sae_leace")]
    handles.append(Line2D([0], [0], color=MUTED, ls=":", lw=0.7, label="chance (0.5)"))
    fig.tight_layout(w_pad=2.0)
    # The x label is centred on the panels, and the legend hangs below it spanning
    # them, from the left edge of the first column to the right edge of the last.
    x0, x1 = axes[0, 0].get_position().x0, axes[0, -1].get_position().x1
    xlabel.set_x((x0 + x1) / 2)
    fig.legend(handles=handles, loc="upper center", ncol=4, mode="expand", frameon=True,
               bbox_to_anchor=(x0, -0.015, x1 - x0, 0), borderaxespad=0,
               fontsize=fs["legend"], handlelength=2.8, borderpad=0.8)
    return save_fig(fig, "figureD2_corpus_size", APPX)


@exhibit("Table D4", "Size-sweep numeric companion", APPX)
def table_d4():
    res = {k: raw("corpus_size", f"corpus_size_{k}_e5-small.json") for k in ("swiltra", "bills_tweets")}
    if not any(res.values()):
        raise Missing("corpus_size/corpus_size_<corpus>_e5-small.json")
    rows = []
    for k, r in res.items():
        if r is None:
            continue
        for s in map(str, r["sizes"]):
            v = r["variants"][s]
            if rows:
                rows.append(SEP)
            for c in ("raw", "metadata_leace", "sae_leace"):
                ag = v["agg"][c]
                rows.append([CORPUS[k], intc(v["nominal"]), v["unit"], intc(v["n_docs"]), LABEL[c],
                             pm(ag["linear_auc"]), pm(ag.get("mlp_auc")), pm(ag.get("gb_auc"))])
    return write_table(
        "tableD4_corpus_size", APPX, "Size-sweep numeric companion, E5-small.",
        [(None, ["Corpus", "Nominal size", "Unit", "Actual docs", "Method", "Logistic AUC",
                 "MLP AUC", "GB AUC"], rows)],
        note="Nominal size is the requested number of pairs (SwiLTra) or documents "
             "(Bills–Tweets); actual docs is what the class-balanced subsample contains. The "
             "subsample at each size is fixed across the three pipeline seeds.")


@exhibit("Table D5", "Legacy fitting-protocol diagnostic", APPX)
def table_d5():
    res = {k: raw("protocol", f"protocol_{k}_e5-small.json") for k in R.ORIGINAL}
    if not any(res.values()):
        raise Missing("protocol/protocol_<corpus>_e5-small.json")
    rows = []
    for k in R.ORIGINAL:
        r = res[k]
        if r is None:
            continue
        if rows:
            rows.append(SEP)
        for c in ("raw", "metadata_leace", "sae_leace"):
            t, d = r["arms"]["pair"]["agg"][c]["linear_auc"], r["arms"]["document"]["agg"][c]["linear_auc"]
            rows.append([CORPUS[k], LABEL[c], pm(t), pm(d), signed(d["mean"] - t["mean"])])
    return write_table(
        "tableD5_protocol", APPX, "Legacy fitting-protocol diagnostic, E5-small.",
        [(None, ["Corpus", "Method", "AUC, transductive fit", "AUC, document holdout",
                 "Δ (holdout − transductive)"], rows)],
        note=f"Both arms use one 50/50 split per seed ({nseeds(res)}), so SemEval is not its "
             "five-fold headline. Transductive: the eraser is fit on every document with the "
             "held-out pairs' links withheld (the draft's \"pair holdout\"; Bills–Tweets has no "
             "pairs), and the audit probe is cross-validated over all documents, including "
             "those the eraser was fit on. Document holdout: fit on the training half only, "
             "audit on the held-out half. Recomputed with the revised audit, unflipped; these "
             "are not the draft's historical values. An AUC below 0.5 under the transductive "
             "arm is an artifact of fitting and scoring the same documents, not better "
             "erasure.")


def _desc(k):
    return raw("descriptive", f"descriptive_{k}_e5-small.json")


@exhibit("Table D6", "Capture of the nuisance covariance subspace", APPX)
def table_d6():
    res = {k: _desc(k) for k in R.ID_CORPORA}
    if not any(res.values()):
        raise Missing("descriptive/descriptive_<corpus>_e5-small.json")
    rows = []
    for k, r in res.items():
        if r is None:
            continue
        g = r["geometry"]
        rows.append([CORPUS[k], str(g["language_rank"]), str(g["n_selected"]),
                     num(g["candidates"][WEIGHTED]["capture"]),
                     num(g["candidates"][UNWEIGHTED]["capture"])])
    return write_table(
        "tableD6_capture", APPX, "Capture of the nuisance covariance subspace L = col(X_cᵀ Y_c).",
        [(None, ["Corpus", "rank(L)", "N", "Weighted\ncapture", "Unweighted\ncapture"],
          rows)],
        tabcolsep=2.5,
        note="Seed-0 full-corpus descriptive fits (no held-out protocol), one value per fit. "
             "Weighted (covariance-weighted): span(Σ D_Sᵀ), what directions-LEACE removes; unweighted: "
             "span(D_Sᵀ), what the orthogonal projection removes. Capture is the mean squared "
             "cosine of the principal angles to L, in [0, 1]. High capture alone does not "
             "establish useful erasure.")


@exhibit("Figure D3", "Principal angles for the two selected-direction spans", APPX)
def figure_d3():
    res = {k: _desc(k) for k in R.ID_CORPORA}
    if not any(res.values()):
        raise Missing("descriptive/descriptive_<corpus>_e5-small.json")
    plt = _plt()
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    keys = [k for k in R.ID_CORPORA if res[k]]
    y0 = np.arange(len(keys))[::-1].astype(float)
    style = {WEIGHTED: ("#1baf7a", "o", 0.14, "Covariance-weighted span(Σ D_Sᵀ)"),
             UNWEIGHTED: ("#4a3aa7", "^", -0.14, "Unweighted span(D_Sᵀ)")}
    for i, k in enumerate(keys):
        for cand, (col, mk, off, _) in style.items():
            ang = res[k]["geometry"]["candidates"][cand]["angles_deg"]
            y = y0[i] + off
            if len(ang) > 1:
                ax.plot([min(ang), max(ang)], [y, y], color=col, lw=0.9, alpha=0.6)
            ax.plot(ang, [y] * len(ang), mk, color=col, ms=6.5, mec="white", mew=0.6)
    ax.set_yticks(y0)
    ax.set_yticklabels([CORPUS[k] for k in keys])
    ax.set_xlim(0, 90)
    ax.set_xticks(range(0, 91, 15))
    ax.set_xlabel("Principal angle to the nuisance subspace L (degrees)", fontsize=11)
    ax.grid(axis="x")
    ax.tick_params(labelsize=10)
    ax.tick_params(axis="y", length=0)
    handles = [Line2D([0], [0], color=c, marker=m, ls="none", mec="white", ms=9, label=l)
               for c, m, _, l in style.values()]
    # Two columns fill column-first: this order puts both spans on the top row.
    handles.insert(1, Line2D([0], [0], color=MUTED, lw=1.2, label="angle range (not a CI)"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              fontsize=9.5, frameon=True)
    fig.tight_layout()
    return save_fig(fig, "figureD3_principal_angles", APPX)


@exhibit("Table D7", "Selected-latent examples with post-hoc label alignment", APPX)
def table_d7():
    res = {k: _desc(k) for k in R.ID_CORPORA}
    if not any(res.values()):
        raise Missing("descriptive/descriptive_<corpus>_e5-small.json")
    rows = []
    for k, r in res.items():
        if r is None:
            continue
        lat = sorted(r["latents"]["latents"], key=lambda x: (-x["mi_score"], x["latent"]))[:6]
        if rows:
            rows.append(SEP)
        for rank, x in enumerate(lat, 1):
            rows.append([CORPUS[k], str(rank), str(x["latent"]), f"{x['mi_score']:.4f}",
                         num(x["firing_freq"]), CLASS.get(x["aligned_class"], x["aligned_class"]),
                         num(x["purity"]), f"{x['purity_lift']:.2f}", num(x["dense_auc"])])
    return write_table(
        "tableD7_latents", APPX, "Selected-latent examples: the six highest-MI of the 32 "
        "selected latents per corpus.",
        [(None, ["Corpus", "MI rank", "Latent ID", "MI (bits)", "Firing\nfraction",
                 "Aligned\nclass", "Firing-class\nshare", "Lift", "Dense\nAUC"], rows)],
        wide=True,
        note="Seed-0 full-corpus descriptive fits. MI, rank and firing fraction are "
             "label-free; the rest is post-hoc: the aligned class is the largest class among "
             "the latent's firing documents, the share is that class's fraction of them, lift "
             "divides it by the class prior, and dense AUC is the one-vs-rest AUC of the "
             "latent's dense pre-activation for that same class. Rows are chosen by MI, never "
             "by alignment.")


@exhibit("Table D8", "High/low document example for the top-MI latent", APPX)
def table_d8():
    res = {k: _desc(k) for k in R.ID_CORPORA}
    if not any(res.values()):
        raise Missing("descriptive/descriptive_<corpus>_e5-small.json")
    rows = []
    for k, r in res.items():
        if r is None:
            continue
        for ex in r["examples"]:
            rows.append([CORPUS[k], str(r["top_latent"]), ex["end"], str(ex["doc_index"]),
                         CLASS.get(ex["class"], ex["class"]), f"{ex['pre_activation']:+.3f}"
                         .replace("-", "−"), ex["excerpt"]])
    return write_table(
        "tableD8_examples", APPX, "Highest- and lowest-ranked document for each corpus's "
        "highest-MI selected latent.",
        [(None, ["Corpus", "Latent", "End", "Doc ID", "Class", "Dense\npre-activation",
                 "Excerpt"], rows)],
        wide=True, colspec=r"lrlrlrp{0.44\textwidth}",
        note="Ranked by the dense pre-activation (the TopK-gated activation is zero for most "
             "documents), ties broken by the smaller document ID. Excerpts: the first 25 words "
             "(60 characters for Chinese), \"...\" marking truncation; the LaTeX version needs a "
             "CJK-capable engine (e.g. xeCJK) for the Chinese row. Seed-0 full-corpus fits. "
             "Examples illustrate a latent; they cannot show that topic information is "
             "preserved.")


#: Figure D4 text sizes (pt, at drawn size); the note matches the other figures' axis names.
FIGD4_FONT = {"title": 12.5, "tick": 10, "note": 11}


@exhibit("Figure D4", "Distribution across all selected latents", APPX)
def figure_d4():
    res = {k: _desc(k) for k in R.ID_CORPORA}
    if not any(res.values()):
        raise Missing("descriptive/descriptive_<corpus>_e5-small.json")
    plt = _plt()
    fs = FIGD4_FONT
    keys = [k for k in R.ID_CORPORA if res[k]]
    # Wide, with the two long titles on two lines: on one they collided at 7.4in.
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.9), sharey=True)
    rng = np.random.default_rng(0)
    y0 = np.arange(len(keys))[::-1].astype(float)
    for j, (key, title) in enumerate((("firing_freq", "Firing fraction"),
                                      ("purity", "Majority-class share\namong firing docs"),
                                      ("dense_auc", "Dense pre-activation AUC\n(aligned class)"))):
        ax = axes[j]
        for i, k in enumerate(keys):
            v = np.array([x[key] for x in res[k]["latents"]["latents"]], float)
            jit = rng.uniform(-0.18, 0.18, len(v))
            ax.plot(v, y0[i] + jit, "o", color=CORPUS_COLOR[k], ms=3, alpha=0.7, mec="none")
            ax.plot([np.median(v)] * 2, [y0[i] - 0.3, y0[i] + 0.3], color=INK, lw=1.4)
        if key == "dense_auc":
            ax.axvline(0.5, color=MUTED, ls=":", lw=0.7)
        ax.set_title(title, fontsize=fs["title"], pad=7)
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="both", labelsize=fs["tick"])
    axes[0].set_yticks(y0)
    axes[0].set_yticklabels([f"{CORPUS[k]} (n = {len(res[k]['latents']['latents'])})" for k in keys])
    # The note takes the x label's slot, so the layout leaves room for it; it is then
    # centred on the panels rather than on the figure.
    note = fig.supxlabel("One point per selected latent (fixed jitter); tick: median.\n"
                         "0.5 is a reference for AUC only; class shares depend on the class "
                         "priors.", fontsize=fs["note"], color=MUTED)
    fig.tight_layout(w_pad=1.5)
    note.set_x((axes[0].get_position().x0 + axes[-1].get_position().x1) / 2)
    return save_fig(fig, "figureD4_latent_distributions", APPX)


@exhibit("Figure D5", "Two-dimensional geometry illustration", APPX)
def figure_d5():
    res = {k: _desc(k) for k in R.ID_CORPORA}
    if not any(res.values()):
        raise Missing("descriptive/descriptive_<corpus>_e5-small.json")
    plt = _plt()
    keys = [k for k in R.ID_CORPORA if res[k]]
    conds = ("raw", "metadata_leace", "sae_leace")
    fig, axes = plt.subplots(len(keys), 3, figsize=(7.0, 2.2 * len(keys)))
    axes = np.atleast_2d(axes)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for i, k in enumerate(keys):
        pca = res[k]["pca"]
        labels = np.array(pca["labels"])
        idx = np.random.default_rng(0).permutation(len(labels))[:1500]   # one fixed subset
        classes = pca["classes"]
        for j, c in enumerate(conds):
            ax = axes[i, j]
            xy = np.asarray(pca["panels"][c]["coords"])[idx]
            ev = pca["panels"][c]["explained_variance_ratio"]
            for ci_, cls in enumerate(classes):
                m = labels[idx] == cls
                ax.scatter(xy[m, 0], xy[m, 1], s=2, color=palette[ci_ % 4], alpha=0.5,
                           label=CLASS.get(cls, cls), rasterized=True)
            ax.set_xlabel(f"PC1 ({ev[0]:.0%})", fontsize=6.5)
            ax.set_ylabel(f"PC2 ({ev[1]:.0%})", fontsize=6.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(LABEL[c])
        axes[i, 0].annotate(CORPUS[k], (-0.3, 0.5), xycoords="axes fraction", rotation=90,
                            va="center", ha="center", fontsize=8)
        axes[i, 2].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True,
                          markerscale=4, fontsize=6.5)
    fig.suptitle("PCA is fitted separately in every panel: coordinates are not comparable "
                 "across conditions.", fontsize=7.5, color=MUTED, y=1.0)
    fig.tight_layout()
    return save_fig(fig, "figureD5_pca_panels", APPX)


#: One colour per encoder (ColorBrewer "Paired" has exactly ten), so Figure D6 can
#: identify ten points per panel without per-point text labels, which collide.
ENCODER_PALETTE = ("#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
                   "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a")


@exhibit("Figure D6", "PC1 dominance and absolute leakage reduction", APPX)
def figure_d6():
    cells = encoder_cells(R.ID_CORPORA)                  # CAP's cells included (section 1k)
    if not cells:
        raise Missing("encoders/encoder_<corpus>_<encoder>.json")
    plt = _plt()
    from matplotlib.lines import Line2D
    from scipy.stats import pearsonr
    colour = {e: ENCODER_PALETTE[i % len(ENCODER_PALETTE)]
              for i, e in enumerate(R.ENCODER_STUDY)}
    # One panel per corpus in a two-column grid; the cell left over takes the legend.
    nrow = -(-len(R.ID_CORPORA) // 2)
    fig, axes = plt.subplots(nrow, 2, figsize=(6.6, 2.55 * nrow), sharex=True, sharey=True)
    pooled, seen = [], []
    for ax, k in zip(axes.ravel(), R.ID_CORPORA):
        pts = [(c["diagnostics"]["pc1_var_share"],
                c["agg"]["raw"]["linear_auc"]["mean"] - c["agg"]["sae_leace"]["linear_auc"]["mean"],
                e) for (kk, e), c in cells.items() if kk == k]
        pooled += [(x, y) for x, y, _ in pts]
        if not pts:
            _nr(ax)
            ax.set_title(CORPUS[k])
            continue
        for x, y, e in pts:
            ax.plot(x, y, "o", color=colour.get(e, MUTED), ms=6, mec="black", mew=0.4,
                    zorder=3)
            if e not in seen:
                seen.append(e)
        xs, ys, _ = zip(*pts)
        r = pearsonr(xs, ys).statistic if len(xs) > 2 else float("nan")
        ax.set_title(f"{CORPUS[k]} (descriptive r = {r:.2f}, n = {len(xs)})")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.grid(True)
    free = axes.ravel()[len(R.ID_CORPORA):]
    for ax in free:
        ax.axis("off")
    # The panel above an empty cell is the lowest in its column, so it carries the x axis.
    for ax in axes[-1]:
        low = axes[-2, list(axes[-1]).index(ax)] if ax in free else ax
        low.xaxis.set_tick_params(labelbottom=True)
        low.set_xlabel("PC1 variance share of the raw embeddings")
    for ax in axes[:, 0]:
        ax.set_ylabel("AUC_Raw − AUC_SAE")
    order = [e for e in R.ENCODER_STUDY if e in seen] + [e for e in seen if e not in R.ENCODER_STUDY]
    handles = [Line2D([0], [0], color=colour.get(e, MUTED), marker="o", ls="none", ms=6,
                      mec="black", mew=0.4, label=ENCODER.get(e, e)) for e in order]
    labels = [ENCODER.get(e, e) for e in order]
    r_all = pearsonr(*zip(*pooled)).statistic if len(pooled) > 2 else float("nan")
    fig.text(0.5, 0.005, f"Ten-encoder study, revised audit, {nseeds(cells)} per cell (means). "
             f"Pooled descriptive r = {r_all:.2f};\ncells share corpora and encoders, so they "
             "are not independent replications.", ha="center", va="bottom", fontsize=6.5,
             color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    if len(free):
        free[-1].legend(handles, labels, loc="center", ncol=2, title="Encoder")
    else:
        fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.10))
    return save_fig(fig, "figureD6_pc1_dominance", APPX)


@exhibit("Figure D7", "Erasure-budget response curves", APPX)
def figure_d7():
    return build_figure_budget()


@exhibit("Table D9", "Alternative erasers, label-free selectors and matched controls", APPX)
def table_d9():
    import alternatives as A
    try:
        return A.table_alternatives(stem="tableD9_alternatives", where=APPX, wide=True)
    except FileNotFoundError as exc:
        raise Missing(exc) from None


@exhibit("Table D10", "k-means LEACE by group count", APPX)
def table_d10():
    import alternatives as A
    try:
        return A.table_kmeans_count(stem="tableD10_kmeans_count", where=APPX, wide=True,
                                    keep="ABEGF", alt_name="Table D9")
    except FileNotFoundError as exc:
        raise Missing(exc) from None


@exhibit("Table D11", "Confirmatory test of a generous K on held-out encoders", APPX)
def table_d11():
    import alternatives as A
    try:
        return A.table_generous_k(stem="tableD11_generous_k", where=APPX, wide=True,
                                  verdicts=False)
    except FileNotFoundError as exc:
        raise Missing(exc) from None


@exhibit("Table D12", "Where the nuisance sits in each cell's geometry", APPX)
def table_d12():
    import alternatives as A
    try:
        return A.table_geometry(stem="tableD12_geometry", where=APPX, wide=True)
    except FileNotFoundError as exc:
        raise Missing(exc) from None


@exhibit("Figure D8", "Geometry against what the label-free erasers achieve", APPX)
def figure_d8():
    import alternatives as A
    try:
        return A.figure_geometry(stem="figureD8_geometry", where=APPX)
    except FileNotFoundError as exc:
        raise Missing(exc) from None


@exhibit("Table D14", "CAP topic organisation across the ten encoders", APPX)
def table_d14():
    import alternatives as A
    try:
        return A.table_cap_topics(stem="tableD14_cap_topics", where=APPX, wide=True)
    except FileNotFoundError as exc:
        raise Missing(exc) from None


@exhibit("Table D15", "CAP topics by source balance", APPX)
def table_d15():
    import alternatives as A
    try:
        return A.table_cap_entanglement(stem="tableD15_cap_entanglement", where=APPX, wide=True)
    except FileNotFoundError as exc:
        raise Missing(exc) from None


@exhibit("Table D13", "Agreement between the label-free selectors", APPX)
def table_d13():
    import alternatives as A
    try:
        return A.table_selector_overlap(stem="tableD13_selector_overlap", where=APPX, wide=True)
    except FileNotFoundError as exc:
        raise Missing(exc) from None


# =========================================================================== #
# Build everything, then the index
# =========================================================================== #
def write_index() -> str:
    root = artifacts.BASE_DIR
    lines = ["# Revised results", "",
             "Every exhibit of `deconfounding_paper_revision_plan_3.md`, rebuilt from the raw "
             "JSON in `raw/` by `code/revised_exhibits.py` (`build_all()`); the runs that "
             "produce `raw/` are in `notebooks/revised_experiments.ipynb` and "
             "`code/revised.py`. Tables come as LaTeX (`.tex`) and Markdown (`.md`), figures as "
             "PNG and PDF.", "",
             "| Exhibit | Title | Status | Files | Raw sources |", "|---|---|---|---|---|"]
    for e in INDEX:
        rel = lambda p: os.path.relpath(p, root)                       # noqa: E731
        lines.append(f"| {e['id']} | {e['title']} | {e['status']} | "
                     + "<br>".join(f"`{rel(p)}`" for p in e["files"]) + " | "
                     + "<br>".join(f"`{rel(p)}`" for p in e["sources"]) + " |")
    lines += ["", "## Layout", "",
              "- `raw/<experiment>/` — one JSON per run (per-seed values, aggregates, "
              "configuration); `raw/ood/ood_summary.json` is derived from the five OOD runs.",
              "- `tables/main`, `tables/appendix` — New Tables 1–3; Tables A1–D15.",
              "- `figures/main`, `figures/appendix` — New Figures 1–2; Figures A1, C1, D1–D8.",
              "", "NR in a table: that cell's run is missing, not estimated.", ""]
    path = os.path.join(root, "README.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


def build_all() -> list:
    """Rebuild every exhibit from the raw JSON under the active results root."""
    INDEX.clear()
    for run in EXHIBITS:
        run()
    path = write_index()
    built = sum(e["status"] == "built" for e in INDEX)
    print(f"\n{built}/{len(INDEX)} exhibits built; index: {path}")
    for e in INDEX:
        if e["status"] != "built":
            print(f"  {e['id']}: {e['status']}")
    return INDEX
