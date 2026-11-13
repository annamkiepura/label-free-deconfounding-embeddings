"""
Command-line entry point for the SAE-confounder-discovery experiment.

Usage examples
--------------
    # mixed-source corpus (bills vs tweets), unpaired
    python run_sae_experiment.py --dataset bills_tweets --embedding mini

    # cross-lingual news pairs (SemEval), paired + gold similarity
    python run_sae_experiment.py --dataset semeval --embedding e5-small --n 2000

    # multilingual Swiss legal summaries (de vs fr), paired
    python run_sae_experiment.py --dataset swiltra --langs de fr --embedding e5-small

    # US case summaries from two sources, unpaired
    python run_sae_experiment.py --dataset case_summaries \
        --csv data/case_summaries_oyez_vs_lexisnexis.csv --embedding mini

The script:
  1. loads the requested dataset into (texts, source_labels, pairs, gold_sim),
  2. embeds the documents with the chosen sentence encoder,
  3. runs the full SAE deconfounding pipeline,
  4. prints the discovered-confounder interpretation and the 3-way comparison
     (raw vs metadata-LEACE vs SAE-LEACE),
  5. saves a bar-chart of the key metrics to `sae_<dataset>_comparison.png`.
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
import torch

from sae_confounder_pipeline import (
    compute_sae_features,
    format_comparison,
    leakage_recall_sweep,
    run_sae_deconfounding,
)
from artifacts import artifact_path, load_result, save_result, savefig


def load_embedding_model(embedding, trust_remote_code=True):
    """Lazy wrapper around `style_erasure_pipeline.load_embedding_model`.

    That module imports sentence-transformers and seaborn at import time. The
    corpus registry (`corpora`) imports this file for its loaders, so a
    module-level import here would make every notebook session -- including one
    that only rebuilds tables from cached embeddings -- depend on the encoder
    stack. Resolving it on first use keeps the pipeline importable without it.
    """
    from style_erasure_pipeline import load_embedding_model as _load
    return _load(embedding, trust_remote_code=trust_remote_code)


def _artifact_scope(name):
    """Scope this runner's artifacts to ``run_artifacts/<name>/``."""
    import artifacts as _a
    return _a.experiment(name)


def pick_device(device=None):
    """Return 'cuda' on a GPU box (e.g. Colab A100), else 'cpu'."""
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Dataset loaders -> (texts, source_labels, pairs, gold_sim)
# --------------------------------------------------------------------------- #
def _balanced_head(df, col, n):
    """Take ~n rows total, balanced across the values of `col`, keeping all
    columns. (Avoids pandas 2.x groupby.apply dropping the grouping column.)"""
    if not n:
        return df
    per = max(1, n // df[col].nunique())
    return pd.concat([g.head(per) for _, g in df.groupby(col)]).reset_index(drop=True)


def _drop_duplicate_texts(df, col):
    """Drop rows whose text already appeared earlier in `df` (first kept).

    Every corpus shipped in `data/` contains some exact duplicates (67 in
    Bills--Tweets, 22 in SCOTUS Oyez/Lexis, ...). Under the document-level
    holdout a duplicate can land on both sides of the split, so an eraser or a
    k-means model fit on the training half has literally seen a test document;
    and a duplicate is trivially its own nearest neighbour, which inflates the
    within-source neighbourhood-retention metric. Deduplicating at load time
    removes both effects for every experiment at once.
    """
    keep = ~df[col].astype(str).duplicated(keep="first")
    return df[keep].reset_index(drop=True)


def _finalise(texts, sources, pair_id, keep):
    """Apply a keep-mask to a partially paired corpus and re-index its pairs.

    `pair_id[r]` is the linked group a row belongs to (-1 = unpaired): a pair in a
    two-class corpus, or one case or decision across three sources or languages.
    Rows are kept in order; every two surviving members of a group form a pair (a
    two-member group is one pair, a three-member group three), and a group left
    with one member makes it an unpaired document. Returns ``(texts, source,
    pairs)`` with pairs in output coordinates, earlier member first.
    """
    keep = np.asarray(keep, dtype=bool)
    pos = -np.ones(len(texts), dtype=int)
    pos[keep] = np.arange(int(keep.sum()))
    members: dict = {}
    for r in np.where(keep)[0]:
        if pair_id[r] >= 0:
            members.setdefault(int(pair_id[r]), []).append(int(pos[r]))
    pairs = [p for _, v in sorted(members.items()) for p in combinations(v, 2)]
    out_texts = [t for t, k in zip(texts, keep) if k]
    out_source = np.asarray(sources)[keep]
    return out_texts, out_source, (pairs or None)


def load_bills_tweets(n=None, path="data/bill_tweets_sample.jsonl"):
    df = _drop_duplicate_texts(pd.read_json(path, lines=True), "text")
    df = _balanced_head(df, "source", n)
    texts = df["text"].astype(str).tolist()
    source = df["source"].astype(str).to_numpy()
    return texts, source, None, None


CAP_SOURCES = ("bill", "order", "newspaper")


def load_cap(n=None, sources=CAP_SOURCES, path="data/cap_sample.jsonl", topics=False):
    """The parent paper's category-level data: congressional bill summaries,
    executive orders and Pennsylvania newspaper clips from the Comparative
    Agendas Project -- a three-way *source* confounder, unpaired, where every
    document also carries one of the 21 CAP major topics.

    The file is sized as the parent's Table 1: 1,301 bills, 598 orders and
    1,309 clips (`build_cap_data.table1_sample` explains how, and what differs
    from the parent). Each pairing the paper evaluates is this file restricted
    to two of `sources`. With `topics=True` a fifth value follows: each
    document's major topic, the category label clusters are scored against.
    """
    df = pd.read_json(path, lines=True)
    df = _drop_duplicate_texts(df[df["source"].isin(sources)], "text")
    df = _balanced_head(df, "source", n)
    out = (df["text"].astype(str).tolist(), df["source"].astype(str).to_numpy(), None, None)
    return (*out, df["topic"].astype(str).to_numpy()) if topics else out


def load_case_summaries(path, n=None, n_pairs=None):
    """US Supreme Court case summaries from two sources -- a *source* (not
    language) confounder, **partially paired** exactly as the parent paper
    constructs its event-level data.

    Each CSV is two source blocks in file order. The parent paper retained up
    to 1,024 case pairs per setting and pooled the rest as unpaired documents;
    the files reflect that construction: row i of the first block and row i of
    the second describe the same case for i < `n_pairs` (1,024 for Oyez/Lexis
    and Wiki/Lexis, 780 for Oyez/Wiki -- a TF-IDF best-match check recovers the
    row alignment for 95% of those rows and for none after), and the remaining
    rows are the unpaired remainder. The paper's Table 1 counts documents:
    2,048 paired + 2,075 unpaired for Oyez/Lexis.

    Two caveats a reader should know. The files carry no case identifier, so
    the boundary is taken from the paper rather than from the data. And the
    unpaired remainder still contains both members of many further cases (the
    paper says one member was discarded, but a TF-IDF mutual-best-match check
    finds hundreds of confident cross-source matches in the remainder), so
    "unpaired" means "not a retrieval target", not "no counterpart in the
    pool"; under the document holdout a remainder document's counterpart can
    sit in the training half. Exact duplicate texts are removed, so no test
    *text* is ever seen in training, and the effect is confined to distractors
    and to the probe's training pool. With `n_pairs=None` the corpus is
    unpaired, the older behaviour.
    """
    df = pd.read_csv(path)
    texts = df["case_summaries"].astype(str).tolist()
    source = df["source"].astype(str).to_numpy()
    blocks = list(dict.fromkeys(source))              # file order
    if len(blocks) != 2:
        raise ValueError(f"{path}: expected two sources, found {blocks}")
    rows = {b: np.where(source == b)[0] for b in blocks}
    pair_id = -np.ones(len(texts), dtype=int)
    if n_pairs:
        m = int(min(n_pairs, len(rows[blocks[0]]), len(rows[blocks[1]])))
        pair_id[rows[blocks[0]][:m]] = np.arange(m)
        pair_id[rows[blocks[1]][:m]] = np.arange(m)
    keep = ~pd.Series(texts).duplicated(keep="first").to_numpy()
    if n:                                              # head of each block, in file order
        per = max(1, int(n) // 2)
        for b in blocks:
            kept = [r for r in rows[b] if keep[r]]
            keep[kept[per:]] = False
    # Emit block A rows first, then block B, so (i, n_A + i) stays readable.
    order = np.concatenate([rows[b] for b in blocks])
    texts_o = [texts[r] for r in order]
    src_o, pid_o, keep_o = source[order], pair_id[order], keep[order]
    texts_f, source_f, pairs = _finalise(texts_o, src_o, pid_o, keep_o)
    return texts_f, source_f, pairs, None


#: The three pairwise SCOTUS files and the leading rows of each block that are case
#: pairs (see `load_case_summaries`); `load_case_summaries_union` pools them.
SCOTUS_FILES = (("data/case_summaries_oyez_vs_lexisnexis.csv", 1024),
                ("data/case_summaries_oyez_vs_wikipedia.csv", 780),
                ("data/case_summaries_wikipedia_vs_lexisnexis.csv", 1024))


def load_case_summaries_union(csvs=SCOTUS_FILES, n=None):
    """SCOTUS with all three sources in one corpus -- Oyez, LexisNexis and
    Wikipedia -- pooled from the three pairwise files, the way the parent paper
    pools every CAP document for its "all three sources" setting.

    `csvs` is a sequence of ``(path, n_pairs)``; each file is read exactly as
    `load_case_summaries` reads it. A document is its (source, text), so a
    summary that two files share (1,377 Oyez, 554 Lexis and all 1,497 Wikipedia
    summaries appear in two files) enters once. The files' pairs then link
    documents into cases: a connected component of the links is one case, and
    every two of its documents are a retrieval pair -- including the pair a case
    gets by transitivity, when an Oyez summary is paired with a Lexis summary in
    one file and with a Wikipedia summary in another. A component that holds two
    summaries from one source is ambiguous (4 in these files) and its documents
    are left unpaired. Every other document is an unpaired distractor, as in the
    pairwise files. Documents come in source blocks, in order of first
    appearance (Oyez, Lexis, Wikipedia), each block in order of first appearance.
    `n` keeps the head of each block, for smoke runs.
    """
    docs, src, index, links = [], [], {}, []
    for path, n_pairs in csvs:
        texts, source, pairs, _ = load_case_summaries(path, n_pairs=n_pairs)
        ids = []
        for t, s in zip(texts, source):
            ids.append(index.setdefault((str(s), t), len(docs)))
            if ids[-1] == len(docs):
                docs.append(t)
                src.append(str(s))
        links += [(ids[a], ids[b]) for a, b in (pairs or [])]
    src = np.asarray(src)
    parent = list(range(len(docs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        parent[find(b)] = find(a)
    members: dict = {}
    for i in range(len(docs)):
        members.setdefault(find(i), []).append(i)
    group = -np.ones(len(docs), dtype=int)
    for g, mem in enumerate(sorted(members.values(), key=min)):
        if len(mem) > 1 and len(set(src[mem])) == len(mem):
            group[mem] = g
    blocks = list(dict.fromkeys(src))
    order = np.concatenate([np.where(src == b)[0] for b in blocks])
    keep = np.ones(len(docs), dtype=bool)
    if n:
        per = max(1, int(n) // len(blocks))
        for b in blocks:
            keep[np.where(src == b)[0][per:]] = False
    texts_f, source_f, pairs = _finalise([docs[r] for r in order], src[order],
                                         group[order], keep[order])
    return texts_f, source_f, pairs, None


def _drop_duplicate_pairs(df):
    """Drop SemEval rows that re-use an article already in an earlier pair.

    The shipped CSV has 444 rows but only 861 distinct articles: 11 rows repeat
    an earlier pair outright (same `pair_id`) and 5 more share one article with
    an earlier pair. Left in, an exact duplicate of a query sits in the
    retrieval index at cosine 1.0 and outranks the true partner, so recall@1 is
    capped below 1 for those queries regardless of erasure; and under the
    document holdout the duplicate pair can land in the training half while the
    original is scored as held out, so the SAE and every eraser have seen the
    test document. First occurrence is kept, so pairs and gold stay aligned.
    """
    seen, keep = set(), []
    for t1, t2 in zip(df["text1"].astype(str), df["text2"].astype(str)):
        dup = t1 in seen or t2 in seen or t1 == t2
        keep.append(not dup)
        if not dup:
            seen.update((t1, t2))
    return df[np.asarray(keep)].reset_index(drop=True)


def load_semeval(n=None, path="data/semeval_en_vs_non_en.csv", invert_gold=True):
    """Cross-lingual news pairs. Stack text1 (lang1) and text2 (lang2); the
    confounder is the article *language*, and (i, n1+i) is a true match.

    SemEval's raw 'Overall' scale is inverted (1 = most similar). With
    `invert_gold=True` (default) we negate it at load time so that HIGHER gold
    = more similar, and therefore a HIGHER (positive) `gold_sim_spearman` is
    better everywhere downstream -- no more "-0.38 beats -0.30" footnotes.
    (Note: this flips the sign of gold_sim_spearman relative to runs from
    before this change.)

    Rows that repeat an article already used by an earlier pair are dropped
    (see `_drop_duplicate_pairs`): 428 pairs / 856 documents remain of the 444
    rows in the file."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["text1", "text2", "url1_lang", "url2_lang"])
    df = _drop_duplicate_pairs(df)
    if n:
        df = df.head(n)
    t1 = df["text1"].astype(str).tolist()
    t2 = df["text2"].astype(str).tolist()
    texts = t1 + t2
    source = np.array(
        df["url1_lang"].astype(str).tolist() + df["url2_lang"].astype(str).tolist()
    )
    n1 = len(t1)
    pairs = [(i, n1 + i) for i in range(n1)]
    gold = df["Overall"].to_numpy() if "Overall" in df.columns else None
    if gold is not None and invert_gold:
        gold = -gold
    return texts, source, pairs, gold


def load_swiltra(langs=("de", "fr"), n=None, path="data/swiltra_bench_decision_summaries.csv",
                 max_pairs=None, seed=0):
    """Multilingual Swiss legal summaries. Take two or three language columns
    (de, fr, it); rows are aligned (same decision), so the documents of one row
    are true cross-lingual matches: (i, n1+i) for two languages, and with three
    every two of (i, n1+i, 2*n1+i) -- three pairs per decision.

    The file is fully paired (2,784 decisions). The parent paper retained
    1,024 pairs and turned the remaining 1,760 decisions into unpaired
    documents by discarding one language version of each at random (its Table
    1: 2,048 paired + 1,760 unpaired documents). Pass `max_pairs=1024` to
    reproduce that construction (the first `max_pairs` rows stay paired; the
    rest keep one random side, seeded by `seed`; with three languages, one
    random language of each). The default keeps every pair, the larger fully
    paired variant this project uses so the corpus-size sweep can go up to
    2,784 pairs; the difference from the paper is documented in the corpus
    registry.
    """
    df = pd.read_csv(path)
    cols = [f"{lang}_bgeText" for lang in langs]
    df = df.dropna(subset=cols)
    if n:
        df = df.head(n)
    n1, k = len(df), len(cols)
    texts = [t for c in cols for t in df[c].astype(str).tolist()]
    source = np.array([lang for lang in langs for _ in range(n1)])
    pair_id = np.tile(np.arange(n1), k)
    keep = ~pd.Series(texts).duplicated(keep="first").to_numpy()
    if max_pairs is not None and max_pairs < n1:
        rng = np.random.default_rng(seed)
        for i in range(int(max_pairs), n1):
            rows = [j * n1 + i for j in range(k)]
            if k == 2:
                drop = [i if rng.random() < 0.5 else n1 + i]   # discard one side
            else:
                kept = int(rng.integers(k))                  # keep one language
                drop = [r for j, r in enumerate(rows) if j != kept]
            keep[drop] = False
            pair_id[rows] = -1                               # survivor is unpaired
    texts_f, source_f, pairs = _finalise(texts, source, pair_id, keep)
    return texts_f, source_f, pairs, None


def load_dataset(dataset, n=2000, csv=None, langs=("de", "fr"), n_pairs=None,
                 max_pairs=None, sources=CAP_SOURCES, topics=False, csvs=SCOTUS_FILES):
    """Dispatch to a loader. With `topics=True` a fifth value is appended: each
    document's category label, which only CAP has (None for the others)."""
    if dataset == "cap":
        return load_cap(n=n, sources=tuple(sources), topics=topics)
    if dataset == "bills_tweets":
        out = load_bills_tweets(n=n)
    elif dataset == "case_summaries":
        assert csv, "csv path required for case_summaries"
        out = load_case_summaries(csv, n=n, n_pairs=n_pairs)
    elif dataset == "case_summaries_union":
        out = load_case_summaries_union(tuple(csvs), n=n)
    elif dataset == "semeval":
        out = load_semeval(n=n)
    elif dataset == "swiltra":
        out = load_swiltra(langs=tuple(langs), n=n, max_pairs=max_pairs)
    else:
        raise ValueError(f"unknown dataset {dataset}")
    return (*out, None) if topics else out


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_comparison(res, dataset, show=False):
    import matplotlib.pyplot as plt

    conds = ["raw", "metadata_leace", "sae_leace"]
    keys = list(res.metrics["raw"].keys())
    x = np.arange(len(keys))
    width = 0.25
    colors = {"raw": "midnightblue", "metadata_leace": "darkorange", "sae_leace": "seagreen"}

    plt.figure(figsize=(max(8, 1.6 * len(keys)), 6))
    for offset, c in zip((-1, 0, 1), conds):
        vals = [res.metrics[c].get(k, np.nan) for k in keys]
        plt.bar(x + offset * width, vals, width, label=c, color=colors[c])
    plt.xticks(x, keys, rotation=30, ha="right")
    plt.ylabel("metric value")
    plt.title(f"Deconfounding comparison — {dataset}")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    savefig(plt.gcf(), f"sae_{dataset}_comparison", show=show)


def plot_comparison_metrics(metrics, dataset, show=False):
    """3-way bar chart from a plain metrics dict (raw/metadata/sae -> metric map).

    The dict-based twin of `plot_comparison`, so the figure can be redrawn from
    the saved ``threeway_*.json`` without re-running the pipeline."""
    import matplotlib.pyplot as plt

    metrics = load_result(metrics)
    if "metrics" in metrics:                       # a full threeway_*.json
        dataset = metrics.get("dataset", dataset)
        metrics = metrics["metrics"]
    conds = ["raw", "metadata_leace", "sae_leace"]
    keys = list(metrics["raw"].keys())
    x = np.arange(len(keys))
    width = 0.25
    colors = {"raw": "midnightblue", "metadata_leace": "darkorange",
              "sae_leace": "seagreen"}
    plt.figure(figsize=(max(8, 1.6 * len(keys)), 6))
    for offset, c in zip((-1, 0, 1), conds):
        vals = [metrics[c].get(k, np.nan) for k in keys]
        plt.bar(x + offset * width, vals, width, label=c, color=colors[c])
    plt.xticks(x, keys, rotation=30, ha="right")
    plt.ylabel("metric value")
    plt.title(f"Deconfounding comparison — {dataset}")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    savefig(plt.gcf(), f"sae_{dataset}_comparison", show=show)


def plot_sweeps(sweeps, dataset, show=False, out=None, x="leakage"):
    """Overlay one or more leakage-vs-quality trade-off curves on one axes.

    `sweeps` is a dict ``{label: sweep}`` (each from `leakage_recall_sweep`).
    y-axis: the quality metric (recall@k higher better, or cluster_purity).
    Each curve is annotated by #latents erased; a greedy curve's J-optimal point
    gets a ★ marker. The orange star is the metadata (true-label) oracle; the
    blue square is raw.

    `x` selects the leakage axis:

      * ``"leakage"``  — transductive flip-adjusted AUC (the historical axis);
      * ``"crossfit"`` — the cross-fitted AUC, where the eraser is refit inside
        each CV training fold. **This is the axis that agrees with the headline
        baselines table**; the two can order conditions differently, because the
        transductive fit makes a held-out probe anti-correlate with the label.

    Multi-seed sweeps (from `compare_sweeps(seeds=...)`) carry ``*_std`` keys and
    are drawn with error bars.
    """
    import matplotlib.pyplot as plt

    sweeps = load_result(sweeps)          # dict, or the name of a saved raw JSON
    sweeps = {k: v for k, v in sweeps.items() if not str(k).startswith("_")}
    palette = ["seagreen", "purple", "teal", "crimson", "darkgoldenrod"]
    any_sweep = next(iter(sweeps.values()))
    ym = any_sweep["y_metric"]
    if x != "leakage" and x not in any_sweep["raw"]:
        print(f"[plot_sweeps] no '{x}' in this sweep — falling back to 'leakage'. "
              f"Re-run with crossfit=True to get the cross-fitted axis.")
        x = "leakage"
    xsd, ysd = f"{x}_std", "y_std"

    plt.figure(figsize=(9, 6))
    for ci, (label, sweep) in enumerate(sweeps.items()):
        color = palette[ci % len(palette)]
        sae = sweep["sae"]
        xs = [p[x] for p in sae]
        ys = [p["y"] for p in sae]
        ns = [p["n_selected"] for p in sae]
        if any(xsd in p or ysd in p for p in sae):
            plt.errorbar(xs, ys, xerr=[p.get(xsd, 0.0) for p in sae],
                         yerr=[p.get(ysd, 0.0) for p in sae], fmt="-o", color=color,
                         ms=4, lw=1.2, elinewidth=0.8, capsize=2, alpha=0.9,
                         label=label, zorder=3)
        else:
            plt.plot(xs, ys, "-o", color=color, ms=4, label=label, zorder=3)
        # Annotate a few points to avoid clutter.
        step = max(1, len(sae) // 6)
        for idx in range(0, len(sae), step):
            plt.annotate(str(ns[idx]), (xs[idx], ys[idx]), textcoords="offset points",
                         xytext=(5, 4), fontsize=8, color=color)
        rec = sweep.get("recommended")
        if rec:  # mark greedy's J-optimal operating point
            match = [p for p in sae if p["n_selected"] == rec]
            if match:
                plt.scatter([match[0][x]], [match[0]["y"]], color=color,
                            marker="*", s=200, edgecolor="black", linewidth=0.5, zorder=5)

    plt.scatter([any_sweep["raw"][x]], [any_sweep["raw"]["y"]],
                color="midnightblue", marker="s", s=90, zorder=4, label="raw")
    plt.scatter([any_sweep["metadata"][x]], [any_sweep["metadata"]["y"]],
                color="darkorange", marker="*", s=280, zorder=4, label="metadata_leace (oracle)")
    plt.axvline(any_sweep["metadata"][x], color="darkorange", ls="--", lw=0.8, alpha=0.6)
    plt.axhline(any_sweep["metadata"]["y"], color="darkorange", ls="--", lw=0.8, alpha=0.6)

    xlabel = ("crossfit_leakage_auc (eraser refit per fold)" if x == "crossfit"
              else "source_leakage_auc (flip-adjusted, transductive)")
    plt.xlabel(f"{xlabel}  (← lower = more deconfounded)")
    ydir = "higher = better" if ym.startswith("recall") else "lower = better"
    plt.ylabel(f"{ym}  ({ydir})")
    n_seeds = any_sweep.get("n_seeds") or len(any_sweep.get("seeds", []) or [])
    seed_note = f" ({n_seeds} seeds, ±1 sd)" if n_seeds and n_seeds > 1 else ""
    plt.title(f"Leakage vs {ym} trade-off — {dataset}{seed_note}")
    plt.legend()
    plt.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    stem = (out or f"sae_{dataset}_sweep.png")
    stem = stem[:-4] if stem.endswith(".png") else stem
    if x != "leakage":
        stem = f"{stem}_{x}"
    savefig(plt.gcf(), stem, show=show)


def plot_sweeps_both(sweeps, dataset, show=False, out=None):
    """Draw the trade-off on both leakage axes (transductive and cross-fitted).

    Publishing only the transductive axis is what let the frontier figures and
    the headline baselines table disagree about whether the SAE beats the
    oracle; emitting both keeps that comparison honest and visible.
    """
    plot_sweeps(sweeps, dataset, show=show, out=out, x="leakage")
    probe = load_result(sweeps)
    ref = next((v for k, v in probe.items() if not str(k).startswith("_")), {})
    if "crossfit" in ref.get("raw", {}):
        plot_sweeps(probe, dataset, show=show, out=out, x="crossfit")


def plot_sweep(sweep, dataset, show=False):
    """Single-curve trade-off plot (thin wrapper around `plot_sweeps`)."""
    label = f"sae_leace ({sweep.get('method', 'topn')})"
    plot_sweeps({label: sweep}, dataset, show=show)


def _print_sweep_table(sweep):
    """Print a sweep as a table; shows the cross-fitted column when recorded."""
    ym = sweep["y_metric"]
    rec = sweep.get("recommended")
    has_cf = "crossfit" in sweep["raw"]

    def row(name, n, d, note=""):
        cf = f" {d['crossfit']:>9.4f}" if has_cf else ""
        print(f"{name:>14} {n:>11} {d['leakage']:>9.4f}{cf} {d['y']:>12.4f}{note}")

    cf_head = f" {'crossfit':>9}" if has_cf else ""
    print(f"\n{'condition':>14} {'n_selected':>11} {'flip':>9}{cf_head} {ym:>12}")
    print("-" * (50 + (10 if has_cf else 0)))
    row("raw", "-", sweep["raw"])
    row("metadata", "-", sweep["metadata"])
    for p in sweep["sae"]:
        note = "  <- recommended" if rec and p["n_selected"] == rec else ""
        row("sae@" + str(p["top_features"]), p["n_selected"], p, note)


def encode_texts(model, texts, embedding, device, batch_size, show_progress_bar=True):
    """Encode texts with model-specific prompt/task handling.

    Several encoders expect a prefix or a task flag for best results (and we use
    them symmetrically — every document is both a query and a passage — so we pick
    the symmetric-similarity setting):
      * e5 family (`e5-small/base/large`) → prepend ``"query: "``;
      * `nomic-v1.5` → prepend ``"search_document: "``; `nomic-v2` → `prompt_name="passage"`;
      * `jina` (jina-embeddings-v3, 8192-token) → `task="text-matching"`.
    Everything else is encoded plainly. Applied identically to all conditions, so
    it never biases the raw/metadata/sae comparison — it just uses each encoder
    the way it was trained (and lets long-context models see full documents).
    """
    kw = dict(show_progress_bar=show_progress_bar, convert_to_numpy=True,
              device=device, batch_size=batch_size)
    if embedding in ("e5-small", "e5-base", "e5-large"):
        return model.encode([f"query: {t}" for t in texts], **kw)
    if embedding == "nomic-v1.5":
        return model.encode([f"search_document: {t}" for t in texts], **kw)
    if embedding == "jina":
        return model.encode(texts, task="text-matching", **kw)
    if embedding == "nomic-v2":
        return model.encode(texts, prompt_name="passage", **kw)
    if embedding == "qwen3-0.6b":
        # Qwen3-Embedding's instruction prompt is for asymmetric (query->doc)
        # retrieval; our use is symmetric doc<->doc, so every text is encoded as
        # a plain document (no prompt), identically on both sides of a pair.
        return model.encode(texts, prompt_name=None, **kw)
    return model.encode(texts, **kw)


def _embed(dataset, embedding, n, csv, langs, device, batch_size):
    """Load + embed a dataset once; returns (embeddings, source, pairs)."""
    print(f"loading '{dataset}' and embedding with '{embedding}' on {device} ...")
    texts, source, pairs, gold = load_dataset(dataset, n=n, csv=csv, langs=langs)
    print(f"  {len(texts)} docs | sources={np.unique(source).tolist()} "
          f"| pairs={'yes' if pairs else 'no'}")
    model = load_embedding_model(embedding)
    emb = encode_texts(model, texts, embedding, device, batch_size)
    return emb, source, pairs


def sweep_top_features(
    dataset,
    embedding="mini",
    n=2000,
    csv=None,
    langs=("de", "fr"),
    method="topn",
    feature_grid=(8, 16, 32, 64, 96, 128),
    expansion_factor=8,
    k=32,
    epochs=200,
    pseudo_groups=2,
    recall_k=10,
    greedy_budget=48,
    greedy_pool=64,
    greedy_lambda=1.0,
    erase="directions_leace",
    crossfit=True,
    seed=0,
    device=None,
    batch_size=256,
    show_plot=True,
):
    """Notebook-friendly: sweep how many discovered latents we erase and plot the
    leakage-vs-quality trade-off, with raw + metadata reference points.

    `method="topn"` (MI top-N at each grid size) or `method="greedy"` (label-free
    greedy selection, whose trajectory becomes the curve). `erase` defaults to
    the headline mechanism and `crossfit` records the cross-fitted leakage axis.
    Returns the sweep dict.
    """
    device = pick_device(device)
    emb, source, pairs = _embed(dataset, embedding, n, csv, langs, device, batch_size)

    print(f"sweeping method={method} ...")
    sweep = leakage_recall_sweep(
        emb, source, pairs=pairs, method=method, feature_grid=feature_grid,
        expansion_factor=expansion_factor, k=k, n_epochs=epochs,
        pseudo_groups=pseudo_groups, recall_k=recall_k,
        greedy_budget=greedy_budget, greedy_pool=greedy_pool, greedy_lambda=greedy_lambda,
        erase=erase, crossfit=crossfit, seed=seed, device=device,
    )
    _print_sweep_table(sweep)
    plot_sweep(sweep, dataset, show=show_plot)
    return sweep


def _aggregate_sweeps(per_seed, seeds):
    """Merge per-seed sweeps of one variant into a single curve with error bars.

    Curves are aligned on `n_selected` (top-N walks a fixed grid; greedy's
    trajectory is indexed by prefix length), and only sizes present in *every*
    seed are kept, so each plotted point averages the same number of runs. Every
    numeric metric gains a ``<key>_std`` companion; the per-seed sweeps are kept
    verbatim under ``per_seed`` so nothing is lost from the raw artifact.
    """
    def _agg(dicts):
        out = {}
        for kk, v0 in dicts[0].items():
            if not isinstance(v0, (int, float)) or isinstance(v0, bool):
                continue
            vals = [d[kk] for d in dicts if kk in d]
            if len(vals) != len(dicts):
                continue
            out[kk] = float(np.mean(vals))
            if len(vals) > 1:
                out[f"{kk}_std"] = float(np.std(vals, ddof=1))
        return out

    base = per_seed[0]
    common = set(p["n_selected"] for p in base["sae"])
    for sw in per_seed[1:]:
        common &= set(p["n_selected"] for p in sw["sae"])
    sae = []
    for n_sel in sorted(common):
        pts = [next(p for p in sw["sae"] if p["n_selected"] == n_sel) for sw in per_seed]
        agg = _agg(pts)
        for bookkeeping in ("n_selected_std", "top_features_std"):
            agg.pop(bookkeeping, None)      # integer labels, not measurements
        agg["n_selected"] = int(n_sel)
        agg["top_features"] = int(pts[0]["top_features"])
        agg["n_seeds"] = len(pts)
        sae.append(agg)

    recs = [sw.get("recommended") for sw in per_seed if sw.get("recommended")]
    recommended = None
    if recs and sae:
        target = float(np.median(recs))
        recommended = int(min((p["n_selected"] for p in sae),
                              key=lambda v: abs(v - target)))
    return {
        "method": base["method"],
        "erase": base.get("erase"),
        "y_metric": base["y_metric"],
        "crossfit": base.get("crossfit", False),
        "raw": _agg([sw["raw"] for sw in per_seed]),
        "metadata": _agg([sw["metadata"] for sw in per_seed]),
        "sae": sae,
        "sae_info": base.get("sae_info", {}),
        "recommended": recommended,
        "seeds": list(seeds),
        "n_seeds": len(per_seed),
        "per_seed": per_seed,
    }


def compare_sweeps(
    dataset,
    embedding="e5-small",
    n=3000,
    csv=None,
    langs=("de", "fr"),
    variants=None,
    erase="directions_leace",
    crossfit=True,
    epochs=200,
    pseudo_groups=4,
    recall_k=10,
    seed=0,
    seeds=None,
    device=None,
    batch_size=256,
    show_plot=True,
    experiment: str | None = None,
):
    """Embed once, run several sweep `variants`, and overlay their frontiers.

    Each variant is a dict that overrides sweep kwargs, e.g.
    ``{"label": "greedy", "method": "greedy"}`` or
    ``{"label": "topn e16/k16", "expansion_factor": 16, "k": 16}``. Defaults to a
    2x2 of {MI top-N, label-free greedy} x {e8/k32, e16/k16} — the selector and
    dictionary-size ablations — all against the oracle.

    `erase` defaults to **`directions_leace`**, the mechanism the headline
    results use. (`leakage_recall_sweep` itself still defaults to `activations`
    for backward compatibility, which is why earlier frontier figures answered
    the ablation under the weaker mechanism; pass `erase=` explicitly to
    reproduce those.) A variant may override it.

    `crossfit=True` records the cross-fitted leakage alongside the transductive
    one, and both figures are drawn — see `plot_sweeps_both`.

    `seeds` (e.g. ``(0, 1, 2)``) reruns every variant per seed and aggregates
    into mean ± sd curves with error bars; the default single `seed` keeps the
    original single-run behaviour.

    Returns ``{label: sweep}``.
    """
    if variants is None:
        variants = [
            {"label": "topn (e8/k32)", "method": "topn", "expansion_factor": 8, "k": 32},
            {"label": "greedy (e8/k32)", "method": "greedy", "expansion_factor": 8, "k": 32},
            {"label": "topn (e16/k16)", "method": "topn", "expansion_factor": 16, "k": 16},
            {"label": "greedy (e16/k16)", "method": "greedy", "expansion_factor": 16, "k": 16},
        ]
    seed_list = list(seeds) if seeds else [seed]
    device = pick_device(device)
    emb, source, pairs = _embed(dataset, embedding, n, csv, langs, device, batch_size)

    sweeps = {}
    for v in variants:
        v = dict(v)
        label = v.pop("label")
        v.setdefault("erase", erase)
        v.setdefault("crossfit", crossfit)
        runs = []
        for sd in seed_list:
            tag = f" [seed {sd}]" if len(seed_list) > 1 else ""
            print(f"\n=== variant: {label}{tag} ({v['erase']}) ===")
            runs.append(leakage_recall_sweep(
                emb, source, pairs=pairs, n_epochs=epochs, pseudo_groups=pseudo_groups,
                recall_k=recall_k, seed=sd, device=device, **v,
            ))
        sweep = runs[0] if len(runs) == 1 else _aggregate_sweeps(runs, seed_list)
        _print_sweep_table(sweep)
        sweeps[label] = sweep

    sweeps["_meta"] = {"dataset": dataset, "figure_stem": f"sae_{dataset}_compare",
                       "erase": erase, "seeds": seed_list}
    with _artifact_scope(experiment or "selection_and_dictionary"):
        save_result(sweeps, f"sweeps_{dataset}.json")      # raw: redraw anytime
        plot_sweeps_both(sweeps, dataset, show=show_plot,
                         out=f"sae_{dataset}_compare.png")
    return sweeps


def compare_greedy_lambda(
    dataset,
    embedding="e5-small",
    n=3000,
    csv=None,
    langs=("de", "fr"),
    lambdas=(0.5, 1.0, 2.0, 4.0),
    erase="directions_leace",
    crossfit=True,
    expansion_factor=8,
    k=32,
    epochs=200,
    pseudo_groups=4,
    recall_k=10,
    greedy_budget=48,
    greedy_pool=64,
    seed=0,
    seeds=None,
    device=None,
    batch_size=256,
    show_plot=True,
    experiment: str | None = None,
):
    """Trace the greedy frontier at several `lambdas` and overlay them.

    `greedy_lambda` weights leakage in the objective `recall@k − lambda·leakage`:
    larger lambda makes greedy keep erasing latents (chasing lower leakage) at the
    cost of recall, so each lambda walks a different distance down the frontier.
    `erase` sets the erasure mechanism used for both the greedy objective and the
    reported metrics (defaults to `directions_leace`, the strongest one — this maps
    the near-oracle direction-erasure frontier; use "activations" for the original).
    The SAE is trained **once per seed** and reused across all lambdas (only the
    greedy search reruns). `seeds` aggregates into mean ± sd curves; `crossfit`
    adds the cross-fitted leakage axis. Returns ``{label: sweep}``.
    """
    seed_list = list(seeds) if seeds else [seed]
    device = pick_device(device)
    emb, source, pairs = _embed(dataset, embedding, n, csv, langs, device, batch_size)

    runs = {f"greedy λ={lam}": [] for lam in lambdas}
    for sd in seed_list:
        print(f"training SAE once for seed {sd}, reused across all lambdas ...")
        feats, scores, info, decoder = compute_sae_features(
            emb, expansion_factor=expansion_factor, k=k, n_epochs=epochs,
            pseudo_groups=pseudo_groups, seed=sd, device=device,
        )
        for lam in lambdas:
            tag = f" [seed {sd}]" if len(seed_list) > 1 else ""
            print(f"\n=== greedy lambda={lam} ({erase}){tag} ===")
            runs[f"greedy λ={lam}"].append(leakage_recall_sweep(
                emb, source, pairs=pairs, method="greedy", erase=erase,
                crossfit=crossfit, feats=feats, scores=scores, sae_info=info,
                decoder=decoder, pseudo_groups=pseudo_groups, recall_k=recall_k,
                greedy_budget=greedy_budget, greedy_pool=greedy_pool,
                greedy_lambda=lam, seed=sd, device=device,
            ))

    sweeps = {}
    for label, per_seed in runs.items():
        sweep = per_seed[0] if len(per_seed) == 1 else _aggregate_sweeps(per_seed, seed_list)
        _print_sweep_table(sweep)
        sweeps[label] = sweep

    sweeps["_meta"] = {"dataset": dataset,
                       "figure_stem": f"sae_{dataset}_lambda_{erase}",
                       "erase": erase, "seeds": seed_list}
    with _artifact_scope(experiment or "lambda_sweep"):
        save_result(sweeps, f"lambda_{dataset}_{erase}.json")
        plot_sweeps_both(sweeps, dataset, show=show_plot,
                         out=f"sae_{dataset}_lambda_{erase}.png")
    return sweeps


def compare_erasure_mechanisms(
    dataset,
    embedding="e5-small",
    n=3000,
    csv=None,
    langs=("de", "fr"),
    selection="greedy",
    grid=(1, 2, 4, 8, 12, 16, 24, 32, 48, 64),
    erasures=("activations", "directions_leace", "directions_orth"),
    expansion_factor=8,
    k=32,
    epochs=200,
    pseudo_groups=4,
    recall_k=10,
    greedy_budget=64,
    greedy_pool=80,
    greedy_lambda=1.0,
    crossfit=True,
    seed=0,
    seeds=None,
    device=None,
    batch_size=256,
    show_plot=True,
    experiment: str | None = None,
):
    """Hold the *selection* of confounder latents fixed; vary only the **erasure
    mechanism**, and overlay the resulting frontiers against the oracle.

    This isolates Step A: does removing the SAE-discovered confounder as a
    subspace of decoder *directions* (`directions_orth` / `directions_leace`)
    push leakage below the floor reached by erasing sparse *activations*
    (`activations`)? The same ordered set of latents is used for all three
    mechanisms, so any difference is due to the mechanism alone.

    `selection`: "greedy" (label-free greedy order) or "topn" (MI score order).
    `crossfit` adds the cross-fitted leakage axis; `seeds` aggregates over seeds
    with error bars. Returns ``{label: sweep}`` (compatible with `plot_sweeps`).
    """
    from sae_confounder_pipeline import (
        _SKIP,
        concept_matrix,
        crossfit_leakage_auc,
        leace_erase_with_directions,
        leace_erase_with_features,
        leace_erase_with_labels,
        one_hot_labels,
        paired_retrieval_recall,
        select_confounder_features_greedy,
        source_leakage_auc,
    )

    seed_list = list(seeds) if seeds else [seed]
    device = pick_device(device)
    emb, source, pairs = _embed(dataset, embedding, n, csv, langs, device, batch_size)
    y_metric = f"recall@{recall_k}"
    runs = {}

    for sd in seed_list:
        print(f"training SAE once (seed {sd}) ...")
        feats, scores, info, decoder = compute_sae_features(
            emb, expansion_factor=expansion_factor, k=k, n_epochs=epochs,
            pseudo_groups=pseudo_groups, seed=sd, device=device,
        )

        # Fix the selection order once (shared across all erasure mechanisms).
        if selection == "greedy":
            print("running greedy selection once (shared order) ...")
            order, _ = select_confounder_features_greedy(
                emb, feats, pairs, scores, budget=greedy_budget, candidate_pool=greedy_pool,
                lam=greedy_lambda, pseudo_groups=pseudo_groups, recall_k=recall_k, seed=sd,
            )
        else:
            order = list(np.argsort(scores)[::-1])
        order = list(order)
        cell_grid = [m for m in grid if m <= len(order)]

        def metrics_of(e, Z=_SKIP):
            out = {
                # flip-adjusted: below-chance raw AUC must not plot as "less leakage"
                "leakage": source_leakage_auc(e, source, sd, flip=True),
                "y": paired_retrieval_recall(e, pairs, ks=(recall_k,))[f"recall@{recall_k}"],
            }
            if crossfit and Z is not _SKIP:
                cf = crossfit_leakage_auc(emb, source, Z=Z, seed=sd)
                out["crossfit"] = cf["auc_flip"]
                out["crossfit_raw"] = cf["auc"]
            return out

        def erase(sel, mech):
            if mech == "activations":
                return leace_erase_with_features(emb, feats, sel)
            if mech == "directions_orth":
                return leace_erase_with_directions(emb, decoder, sel, mode="orthogonal")
            if mech == "directions_leace":
                return leace_erase_with_directions(emb, decoder, sel, mode="leace")
            raise ValueError(mech)

        raw = metrics_of(emb, Z=None)
        metadata = metrics_of(leace_erase_with_labels(emb, source),
                              Z=one_hot_labels(source))

        for mech in erasures:
            pts = []
            for m in cell_grid:
                sel = np.array(order[:m])
                e = erase(sel, mech)
                Z = concept_matrix(emb, feats, decoder, sel, mechanism=mech)
                pt = {"top_features": int(m), "n_selected": int(m), **metrics_of(e, Z=Z)}
                pts.append(pt)
                cf = f"  crossfit={pt['crossfit']:.4f}" if "crossfit" in pt else ""
                print(f"[{mech:16s}] n={m:4d}  leakage={pt['leakage']:.4f}{cf}  "
                      f"{y_metric}={pt['y']:.4f}")
            runs.setdefault(f"{mech} ({selection})", []).append({
                "method": mech, "erase": mech, "y_metric": y_metric,
                "crossfit": bool(crossfit), "sae_info": info,
                "raw": raw, "metadata": metadata, "sae": pts, "recommended": None,
            })

    sweeps = {label: (per_seed[0] if len(per_seed) == 1
                      else _aggregate_sweeps(per_seed, seed_list))
              for label, per_seed in runs.items()}
    sweeps["_meta"] = {"dataset": dataset, "figure_stem": f"sae_{dataset}_erasure",
                       "selection": selection, "seeds": seed_list}
    with _artifact_scope(experiment or "erasure_mechanism"):
        save_result(sweeps, f"erasure_{dataset}_{selection}.json")
        plot_sweeps_both(sweeps, dataset, show=show_plot,
                         out=f"sae_{dataset}_erasure.png")
    return sweeps


# --------------------------------------------------------------------------- #
# Notebook-friendly entry point (use this from Google Colab)
# --------------------------------------------------------------------------- #
def run_experiment(
    dataset,
    embedding="mini",
    n=2000,
    csv=None,
    langs=("de", "fr"),
    expansion_factor=8,
    k=32,
    epochs=200,
    top_features=32,
    pseudo_groups=2,
    erase="directions_leace",
    selection="auto",
    greedy_lambda=1.0,
    greedy_budget=64,
    greedy_pool=80,
    pair_holdout=0.5,
    nonlinear_leakage=True,
    nonlinear_probes=("mlp", "gb"),
    crossfit=True,
    seed=0,
    device=None,
    batch_size=256,
    show_plot=True,
    experiment: str | None = None,
):
    """Run the whole pipeline in one call and return the `ConfounderResult`.

    Designed for notebooks (e.g. Colab): no argparse, embeddings are computed on
    the GPU when one is available, and the comparison plot is both saved and
    displayed inline. `device`/`batch_size` are forwarded to the sentence encoder
    (an A100 happily handles batch_size=256+).
    """
    device = pick_device(device)
    print(f"[1/3] loading dataset '{dataset}' ...")
    texts, source, pairs, gold = load_dataset(dataset, n=n, csv=csv, langs=langs)
    print(f"      {len(texts)} docs | sources={np.unique(source).tolist()} "
          f"| pairs={'yes' if pairs else 'no'} | gold_sim={'yes' if gold is not None else 'no'}")

    print(f"[2/3] embedding with '{embedding}' on {device} ...")
    model = load_embedding_model(embedding)
    embeddings = encode_texts(model, texts, embedding, device, batch_size)

    print("[3/3] running SAE deconfounding pipeline ...")
    res = run_sae_deconfounding(
        embeddings, texts, source,
        pairs=pairs, gold_sim=gold,
        expansion_factor=expansion_factor, k=k,
        n_epochs=epochs, top_n_features=top_features,
        n_pseudo_groups=pseudo_groups, seed=seed,
        device=device, erase=erase, nonlinear_leakage=nonlinear_leakage,
        selection=selection, greedy_lambda=greedy_lambda,
        greedy_budget=greedy_budget, greedy_pool=greedy_pool,
        pair_holdout=pair_holdout, nonlinear_probes=nonlinear_probes,
        crossfit=crossfit,
    )

    if selection == "asymmetry":
        _sel = "asymmetry"
    elif selection == "greedy" or (selection == "auto" and pairs):
        _sel = "greedy"
    else:
        _sel = "topn"
    print(f"\n========= SELECTED CONFOUNDER LATENTS ({_sel}, erase={erase}, "
          f"n={len(res.selected_features)}) =========")
    for j in res.selected_features:
        align = res.label_alignment.get(int(j))
        align_str = f"  (AUC vs true label = {align:.3f})" if align is not None else ""
        print(f"  latent #{int(j):4d}  score={res.feature_scores[j]:.4f}{align_str}")

    print("\n================ 3-WAY COMPARISON ================")
    print(format_comparison(res))
    if pair_holdout and pairs:
        print("\n(recall@k / gold_sim measured on HELD-OUT pairs — greedy selected "
              "on a disjoint train split, so these are unbiased.)")
    print("\nLower is better: *_leakage_auc_flip (→0.5) and cluster_purity (→base rate).")
    print("Higher is better: recall@k, gold_sim_spearman (gold scale is "
          "de-inverted at load time, so positive & higher = better).")
    print("Compare conditions on the *_flip keys: raw below-chance AUCs are a "
          "transductive-eraser artifact, not extra-strong erasure "
          "(a score-flipping adversary recovers 1-AUC).")
    print("(nonlinear_leakage_auc = MLP probe; checks whether the confounder "
          "survives *non*-linearly, since erasure only removes it linearly.)")

    with _artifact_scope(experiment or "threeway"):
        save_result({"dataset": dataset, "embedding": embedding, "seed": seed,
                     "erase": erase, "selection": _sel,
                     "metrics": res.metrics,
                     "selected": [int(j) for j in res.selected_features],
                     "label_alignment": res.label_alignment,
                     "sae_info": {k: v for k, v in res.sae_info.items()
                                  if k != "loss_history"}},
                    f"threeway_{dataset}_{embedding}.json")
        plot_comparison(res, dataset, show=show_plot)
    return res


# --------------------------------------------------------------------------- #
# Multi-seed: error bars on the 3-way comparison
# --------------------------------------------------------------------------- #
def plot_comparison_ci(agg, dataset, keys, conds, show=False):
    """Grouped bar chart of the 3-way comparison with std error bars across seeds."""
    import matplotlib.pyplot as plt

    x = np.arange(len(keys))
    width = 0.25
    colors = {"raw": "midnightblue", "metadata_leace": "darkorange", "sae_leace": "seagreen"}
    plt.figure(figsize=(max(8, 1.6 * len(keys)), 6))
    for offset, c in zip((-1, 0, 1), conds):
        means = [agg[c][k]["mean"] for k in keys]
        stds = [agg[c][k]["std"] for k in keys]
        plt.bar(x + offset * width, means, width, yerr=stds, capsize=3,
                label=c, color=colors[c])
    plt.xticks(x, keys, rotation=30, ha="right")
    plt.ylabel("metric value (mean ± std)")
    plt.title(f"Deconfounding comparison — {dataset} ({agg['_n_seeds']} seeds)")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    savefig(plt.gcf(), f"sae_{dataset}_multiseed", show=show)


def run_experiment_multiseed(
    dataset,
    seeds=(0, 1, 2, 3, 4),
    embedding="e5-small",
    n=3000,
    csv=None,
    langs=("de", "fr"),
    expansion_factor=8,
    k=32,
    epochs=200,
    top_features=32,
    pseudo_groups=4,
    erase="directions_leace",
    selection="auto",
    greedy_lambda=1.0,
    greedy_budget=64,
    greedy_pool=80,
    pair_holdout=0.5,
    nonlinear_leakage=True,
    nonlinear_probes=("mlp", "gb"),
    crossfit=True,
    device=None,
    batch_size=256,
    show_plot=True,
    experiment: str | None = None,
):
    """Run the 3-way comparison over several `seeds` and report mean ± std.

    Embeds **once** (deterministic) and reuses the embeddings across seeds — only
    the seed-dependent parts rerun (SAE init, k-means pseudo-groups, greedy search,
    the pair holdout split, and the probe CV folds), which is exactly the variance
    the error bars should capture. Works on any dataset (e.g. `dataset="swiltra"`
    to replicate on the second language corpus). Returns
    ``{"agg": {cond: {metric: {mean,std,n,values}}}, "per_seed": [...], "seeds": [...]}``.
    """
    device = pick_device(device)
    print(f"[embed once] loading '{dataset}' and embedding with '{embedding}' on {device} ...")
    texts, source, pairs, gold = load_dataset(dataset, n=n, csv=csv, langs=langs)
    print(f"  {len(texts)} docs | sources={np.unique(source).tolist()} "
          f"| pairs={'yes' if pairs else 'no'}")
    model = load_embedding_model(embedding)
    embeddings = encode_texts(model, texts, embedding, device, batch_size)

    per_seed = []
    for s in seeds:
        print(f"\n===== seed {s} ({seeds.index(s) + 1}/{len(seeds)}) =====")
        res = run_sae_deconfounding(
            embeddings, texts, source, pairs=pairs, gold_sim=gold,
            expansion_factor=expansion_factor, k=k, n_epochs=epochs,
            top_n_features=top_features, n_pseudo_groups=pseudo_groups,
            erase=erase, selection=selection, greedy_lambda=greedy_lambda,
            greedy_budget=greedy_budget, greedy_pool=greedy_pool,
            pair_holdout=pair_holdout, nonlinear_leakage=nonlinear_leakage,
            nonlinear_probes=nonlinear_probes, crossfit=crossfit,
            device=device, seed=s, verbose=False,
        )
        per_seed.append(res.metrics)

    conds = ["raw", "metadata_leace", "sae_leace"]
    keys = []
    for c in conds:
        for kk in per_seed[0][c]:
            if kk not in keys:
                keys.append(kk)
    agg = {c: {} for c in conds}
    for c in conds:
        for kk in keys:
            vals = np.array([m[c][kk] for m in per_seed if kk in m[c]], dtype=float)
            agg[c][kk] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "n": int(len(vals)),
                "values": vals.tolist(),
            }
    agg["_n_seeds"] = len(seeds)

    # Print a mean ± std table.
    width = max(len(k) for k in keys) + 2
    header = f"{'metric':<{width}}" + "".join(f"{c:>22}" for c in conds)
    print(f"\n================ {len(seeds)}-SEED COMPARISON (mean ± std) ================")
    print(header)
    print("-" * len(header))
    for kk in keys:
        row = f"{kk:<{width}}"
        for c in conds:
            row += f"{agg[c][kk]['mean']:>10.4f} ± {agg[c][kk]['std']:.4f} "
        print(row)
    print("\nLower is better: *_leakage_auc (→0.5), cluster_purity. "
          "Higher: recall@k, gold_sim_spearman.")

    out = {"agg": agg, "per_seed": per_seed, "seeds": list(seeds),
           "dataset": dataset, "embedding": embedding, "keys": keys,
           "conditions": conds}
    with _artifact_scope(experiment or "multiseed"):
        save_result(out, f"multiseed_{dataset}_{embedding}.json")   # raw
        plot_comparison_ci(agg, dataset, keys, conds, show=show_plot)
    return out


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="SAE confounder discovery + LEACE")
    p.add_argument("--dataset", required=True,
                   choices=["bills_tweets", "case_summaries", "semeval", "swiltra"])
    p.add_argument("--csv", default=None, help="path for case_summaries dataset")
    p.add_argument("--langs", nargs=2, default=["de", "fr"], help="swiltra language pair")
    p.add_argument("--embedding", default="mini", help="key from load_embedding_model")
    p.add_argument("--n", type=int, default=2000, help="cap on #docs (per side/source)")
    p.add_argument("--expansion-factor", type=int, default=8)
    p.add_argument("--k", type=int, default=32, help="TopK sparsity")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--top-features", type=int, default=32,
                   help="how many SAE latents to treat as confounders (more = "
                        "stronger erasure but watch recall@k / gold_sim for over-erasure)")
    p.add_argument("--pseudo-groups", type=int, default=2,
                   help="k-means groups used to score confounder-likeness")
    p.add_argument("--batch-size", type=int, default=256, help="encoder batch size")
    p.add_argument("--device", default=None, help="cuda / cpu (auto-detected if unset)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_experiment(
        dataset=args.dataset, embedding=args.embedding, n=args.n, csv=args.csv,
        langs=args.langs, expansion_factor=args.expansion_factor, k=args.k,
        epochs=args.epochs, top_features=args.top_features,
        pseudo_groups=args.pseudo_groups, seed=args.seed,
        device=args.device, batch_size=args.batch_size, show_plot=False,
    )


if __name__ == "__main__":
    main()
