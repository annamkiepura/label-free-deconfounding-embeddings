"""The corpus registry: one place that defines every dataset the paper uses.

`CORPORA` maps a short key to everything an experiment needs to know about a
corpus -- how to load it, how many seeds to average over, and which
latent-selection strategy applies. The pseudo-group count is ``"auto"``
everywhere: K is chosen inside each fit by silhouette on the training
documents, so the number of confounder classes never enters the method.
Experiments take a key, never a pile of loader kwargs, so a corpus is
configured once and every table, sweep and ablation sees the same
configuration.

Three families, the first two matching Table 1:

* **Language-confounded, paired** -- SemEval (428 cross-lingual news pairs, 856
  documents once the 16 rows that re-use an article are dropped; the file has
  444 rows, which is the count the parent paper quotes) and SwiLTra-Bench de/fr
  (2,784 pairs, 5,568 documents). The confounder is the document's language;
  matched retrieval is well defined, so these take greedy selection and report
  recall@k.

* **Source-confounded** -- congressional bill summaries vs tweets by members of Congress
  (9,933 documents after exact duplicates are dropped; 10,000 in the file),
  which is genuinely unpaired and so takes top-N selection with leakage /
  cluster purity / within-source neighbourhood retention as its metrics; and
  Super-SCOTUS case summaries from two repositories (Oyez vs LexisNexis; the
  Oyez/Wikipedia and Wikipedia/LexisNexis variants are registered too), which
  are **partially paired exactly as the parent paper constructs them**: the
  first 1,024 rows of each source block (780 for Oyez/Wikipedia) are the
  retained case pairs and every other document is pooled as an unpaired
  distractor. Both the eraser and the retrieval index see the whole pool; only
  the pairs are retrieval targets. These therefore take greedy selection and
  report recall like the language corpora. See
  `run_sae_experiment.load_case_summaries` for the evidence and the caveats.

  SwiLTra is the one deliberate departure from the paper's construction: the
  paper kept 1,024 pairs and made the remaining 1,760 decisions unpaired by
  discarding one language side, while this registry keeps all 2,784 pairs so
  the corpus-size sweep can span 400 to 2,784 pairs. Set ``"max_pairs": 1024``
  in its loader entry to reproduce the paper's variant.

  **Class combinations.** Both paired corpora come with three classes: SwiLTra
  has an Italian version of every decision and SCOTUS a Wikipedia summary of
  many cases. As the parent does for CAP (its Table 2), each is registered in
  every pairing -- ``swiltra`` (de/fr), ``swiltra_de_it``, ``swiltra_fr_it``;
  ``scotus_oyez_lexis``, ``scotus_oyez_wiki``, ``scotus_wiki_lexis`` -- and
  with all three classes at once: ``swiltra_de_fr_it`` (each decision a linked
  triple) and ``scotus_oyez_lexis_wiki`` (the three files pooled, a case's
  summaries linked across them). In a three-class corpus a document can have
  two counterparts, so `pairs` holds every two members of a linked group; the
  split engine keeps a group on one side (`protocols.link_groups`) and
  retrieval ranks each target without the query's other counterparts.
  The main experiments run on all eight (notebooks/02_main_experiments.ipynb).

* **Source-confounded, topic-labelled** -- the parent paper's CAP data: 1,301
  bill summaries, 598 executive orders and 1,309 newspaper clips (3,208
  documents, the sizes of its Table 1; `build_cap_data.py` rebuilds and
  samples them). The confounder is the three-way source and there are no
  pairs, so it takes top-N selection like Bills--Tweets. Every document also
  carries one of the 21 CAP major topics, returned by `load` as ``topics``:
  the categories the parent scores clusters against. It is registered but not
  in `HEADLINE`, so no existing run includes it. ``cap_bills_news`` is the
  same sample restricted to bills and newspapers (2,610 documents), the
  parent's CAP eraser-fitting collection for its OOD experiments (`ood`).

Every loader drops exact duplicate texts first: a duplicate that straddles the
train/test split is a test document the eraser has seen, and a duplicate is
trivially its own nearest neighbour. Sizes default to the FULL deduplicated
corpus (`n=None`). Pass `n=` to subsample for a quick smoke run.

Split policy
------------
Each entry also declares how it should be split, because the right answer is not
the same for all four. The eraser is the estimation bottleneck: LEACE whitens
with a d x d covariance fit on the training documents, so what matters is the
ratio of training rows to embedding dimension. With e5-small (d = 384) a 50/50
split gives SwiLTra 2,784 rows (n/d = 7.3), Bills--Tweets 4,966 (12.9) and SCOTUS
2,050 (5.3) -- all comfortable -- but SemEval only 428 (1.11), barely more rows
than dimensions, which is exactly where a whitening estimator degrades. A
half-corpus fit there conflates genuine out-of-sample loss with simply having
halved the fitting data.

SemEval therefore uses 5-fold cross-fitting (685 training rows per fold, n/d =
1.78, and every document scored out of sample); the other three keep the single
50/50 holdout, as does CAP (1,604 training rows, n/d = 4.2). `split_config`
returns the policy and the arithmetic behind it. Every runner in `experiments`
reads this policy through `protocols.splits_for`, so a corpus's scheme is the
same in every section (only the deliberate protocol contrast pins a 50/50 split
on both arms).
"""

from __future__ import annotations

import numpy as np

from run_sae_experiment import load_dataset

#: dataset key -> loader kwargs + per-corpus experiment configuration.
CORPORA: dict[str, dict] = {
    # ---- language-confounded, paired ------------------------------------- #
    "semeval": {
        "load": {"dataset": "semeval", "n": None},
        "confounder": "language", "paired": True,
        "pseudo_groups": "auto", "seeds": (0, 1, 2), "selection": "greedy",
        "label": "SemEval", "has_gold": True,
        # 856 documents: a 50/50 split leaves 428 training rows against d = 384.
        "split": {"scheme": "kfold", "folds": 5},
    },
    "swiltra": {
        "load": {"dataset": "swiltra", "n": None, "langs": ("de", "fr")},
        "confounder": "language", "paired": True,
        "pseudo_groups": "auto", "seeds": (0, 1, 2, 3, 4), "selection": "greedy",
        "label": "SwiLTra (de/fr)", "has_gold": False,
        "split": {"scheme": "holdout", "holdout": 0.5},
    },
    # The other two language pairings of the parent's Table 4, and all three
    # languages at once: every decision in de, fr and it (a three-class
    # nuisance; each decision is a linked triple, i.e. three retrieval pairs).
    "swiltra_de_it": {
        "load": {"dataset": "swiltra", "n": None, "langs": ("de", "it")},
        "confounder": "language", "paired": True,
        "pseudo_groups": "auto", "seeds": (0, 1, 2, 3, 4), "selection": "greedy",
        "label": "SwiLTra (de/it)", "has_gold": False,
        "split": {"scheme": "holdout", "holdout": 0.5},
    },
    "swiltra_fr_it": {
        "load": {"dataset": "swiltra", "n": None, "langs": ("fr", "it")},
        "confounder": "language", "paired": True,
        "pseudo_groups": "auto", "seeds": (0, 1, 2, 3, 4), "selection": "greedy",
        "label": "SwiLTra (fr/it)", "has_gold": False,
        "split": {"scheme": "holdout", "holdout": 0.5},
    },
    "swiltra_de_fr_it": {
        "load": {"dataset": "swiltra", "n": None, "langs": ("de", "fr", "it")},
        "confounder": "language", "paired": True,
        "pseudo_groups": "auto", "seeds": (0, 1, 2, 3, 4), "selection": "greedy",
        "label": "SwiLTra (de/fr/it)", "has_gold": False,
        "split": {"scheme": "holdout", "holdout": 0.5},
    },
    # ---- source-confounded, unpaired ------------------------------------- #
    "bills_tweets": {
        "load": {"dataset": "bills_tweets", "n": None},
        "confounder": "source", "paired": False,
        "pseudo_groups": "auto", "seeds": (0, 1, 2, 3, 4), "selection": "topn",
        "label": "Bills--Tweets", "has_gold": False,
        "split": {"scheme": "holdout", "holdout": 0.5},
    },
    # The parent paper's partially paired construction: `n_pairs` leading
    # rows of each source block are case pairs, the rest are pooled unpaired.
    "scotus_oyez_lexis": {
        "load": {"dataset": "case_summaries", "n": None, "n_pairs": 1024,
                 "csv": "data/case_summaries_oyez_vs_lexisnexis.csv"},
        "confounder": "source", "paired": True,
        "pseudo_groups": "auto", "seeds": (0, 1, 2, 3, 4), "selection": "greedy",
        "label": "SCOTUS (Oyez/Lexis)", "has_gold": False,
        "split": {"scheme": "holdout", "holdout": 0.5},
    },
    "scotus_oyez_wiki": {
        "load": {"dataset": "case_summaries", "n": None, "n_pairs": 780,
                 "csv": "data/case_summaries_oyez_vs_wikipedia.csv"},
        "confounder": "source", "paired": True,
        "pseudo_groups": "auto", "seeds": (0, 1, 2, 3, 4), "selection": "greedy",
        "label": "SCOTUS (Oyez/Wiki)", "has_gold": False,
        "split": {"scheme": "holdout", "holdout": 0.5},
    },
    "scotus_wiki_lexis": {
        "load": {"dataset": "case_summaries", "n": None, "n_pairs": 1024,
                 "csv": "data/case_summaries_wikipedia_vs_lexisnexis.csv"},
        "confounder": "source", "paired": True,
        "pseudo_groups": "auto", "seeds": (0, 1, 2, 3, 4), "selection": "greedy",
        "label": "SCOTUS (Wiki/Lexis)", "has_gold": False,
        "split": {"scheme": "holdout", "holdout": 0.5},
    },
    # All three sources in one corpus: the union of the three files above, a
    # case's summaries linked across them (`load_case_summaries_union`).
    "scotus_oyez_lexis_wiki": {
        "load": {"dataset": "case_summaries_union", "n": None},
        "confounder": "source", "paired": True,
        "pseudo_groups": "auto", "seeds": (0, 1, 2, 3, 4), "selection": "greedy",
        "label": "SCOTUS (Oyez/Lexis/Wiki)", "has_gold": False,
        "split": {"scheme": "holdout", "holdout": 0.5},
    },
    # ---- source-confounded, topic-labelled ------------------------------- #
    # The parent paper's CAP data, all three sources, sized as its Table 1.
    # `load_cap` takes a `sources` subset for the paper's pairwise settings.
    "cap": {
        "load": {"dataset": "cap", "n": None},
        "confounder": "source", "paired": False,
        "pseudo_groups": "auto", "seeds": (0, 1, 2), "selection": "topn",
        "label": "CAP (bills/orders/news)", "has_gold": False,
        "split": {"scheme": "holdout", "holdout": 0.5},
    },
    # The parent's CAP eraser-fitting collection for its OOD section: the same
    # sample restricted to bills and newspapers (its Table 1: 2,613 documents).
    "cap_bills_news": {
        "load": {"dataset": "cap", "n": None, "sources": ("bill", "newspaper")},
        "confounder": "source", "paired": False,
        "pseudo_groups": "auto", "seeds": (0, 1, 2), "selection": "topn",
        "label": "CAP (bills/news)", "has_gold": False,
        "split": {"scheme": "holdout", "holdout": 0.5},
    },
}

#: The corpora the headline tables use, in reading order, and the two
#: families derived from the registry rather than kept by hand.
HEADLINE = ("semeval", "swiltra", "bills_tweets", "scotus_oyez_lexis")
PAIRED = tuple(k for k in HEADLINE if CORPORA[k]["paired"])
UNPAIRED = tuple(k for k in HEADLINE if not CORPORA[k]["paired"])


#: Embedding dimension per encoder, for the n/d diagnostic below. Anything not
#: listed falls back to the value measured at load time.
ENCODER_DIM = {"e5-small": 384, "mini": 384, "mpnet": 768, "gist-small": 384,
               "gist-base": 768, "gist-large": 1024, "e5-base": 768,
               "e5-large": 1024, "nomic-v2": 768, "mxb": 1024}


def split_config(key: str, dim: int | None = None) -> dict:
    """The split policy for `key`, with the training-size arithmetic behind it.

    `train_docs` is how many documents the eraser is fit on under that policy,
    and `n_over_d` is that count divided by the embedding dimension -- the ratio
    that decides whether the LEACE whitening is well estimated. Below about 2 the
    covariance is barely determined and held-out numbers understate the method.
    """
    cfg = config(key)
    sp = dict(cfg.get("split") or {"scheme": "holdout", "holdout": 0.5})
    d = dim or ENCODER_DIM.get("e5-small", 384)
    n = CORPORA[key]["load"].get("n")
    n_docs = n if n else _KNOWN_SIZES.get(key)
    if n_docs:
        share = (1.0 - 1.0 / sp["folds"]) if sp["scheme"] == "kfold" else (
            1.0 - sp.get("holdout", 0.5))
        sp["train_docs"] = int(round(share * n_docs))
        sp["n_over_d"] = round(sp["train_docs"] / d, 2)
        sp["n_docs"] = int(n_docs)
    sp["dim"] = d
    return sp


#: Corpus sizes AFTER the loaders' deduplication, so `split_config` can report
#: the arithmetic without loading the data. Verified against the loaders by the
#: notebook's Table 1 check.
_KNOWN_SIZES = {"semeval": 856, "swiltra": 5568, "bills_tweets": 9933,
                "scotus_oyez_lexis": 4101, "scotus_oyez_wiki": 3297,
                "scotus_wiki_lexis": 3536, "cap": 3208, "cap_bills_news": 2610,
                "swiltra_de_it": 5568, "swiltra_fr_it": 5568, "swiltra_de_fr_it": 8352,
                "scotus_oyez_lexis_wiki": 7506}


#: Session-wide selector override. ``None`` keeps each corpus's registered
#: selector (greedy where pairs exist, top-N otherwise); ``"topn"`` makes every
#: corpus use MI top-N, which skips the greedy search everywhere at once.
SELECTION_OVERRIDE: str | None = None


def set_selection(selection: str | None) -> str | None:
    """Override the selector for every corpus loaded from now on.

    `set_selection("topn")` is the one-line way to run the whole pipeline
    without the greedy search: every runner reads the selector from the corpus
    config it is handed, so corpora loaded after this call (re-run the embed
    cell) take top-N, the greedy-only experiments (selection comparison, the
    lambda sweep) become no-ops, and the budget frontier traces the top-N
    ordering alone. Pass ``None`` to restore the registry.
    """
    global SELECTION_OVERRIDE
    if selection not in (None, "topn", "greedy"):
        raise ValueError(f"selection must be None, 'topn' or 'greedy', got {selection!r}")
    SELECTION_OVERRIDE = selection
    return SELECTION_OVERRIDE


def config(key: str, **overrides) -> dict:
    """The registry entry for `key`, with `overrides` applied (never mutates).

    A session-wide `set_selection` override is applied first; greedy is never
    forced onto an unpaired corpus.
    """
    if key not in CORPORA:
        raise ValueError(f"unknown corpus {key!r}; known: {', '.join(CORPORA)}")
    cfg = {kk: (dict(v) if isinstance(v, dict) else v)
           for kk, v in CORPORA[key].items()}
    if SELECTION_OVERRIDE and (SELECTION_OVERRIDE != "greedy" or cfg["paired"]):
        cfg["selection"] = SELECTION_OVERRIDE
    for kk, v in overrides.items():
        if kk == "n":
            cfg["load"]["n"] = v
        else:
            cfg[kk] = v
    return cfg


def load(key: str, n: int | None = "keep") -> dict:
    """Load one corpus: ``{texts, source, pairs, gold, topics, ...}`` plus its config.

    `topics` is each document's category label -- the CAP major topic -- and
    None for every corpus without categories. `n=None` forces the full corpus;
    leave it at ``"keep"`` to use the registry's own size (also the full
    corpus, unless the entry says otherwise).
    """
    cfg = config(key)
    kw = dict(cfg["load"])
    if n != "keep":
        kw["n"] = n
    texts, source, pairs, gold, topics = load_dataset(**kw, topics=True)
    source = np.asarray(source)
    classes, counts = np.unique(source, return_counts=True)
    n_pairs = len(pairs) if pairs else 0
    # Documents in some pair: 2 per pair when pairs are disjoint, fewer when a
    # three-class corpus links a case's documents into a triple (three pairs).
    n_linked = int(len(np.unique(np.asarray(pairs)))) if pairs else 0
    return {
        "key": key, "label": cfg["label"], "texts": texts, "source": source,
        "pairs": pairs, "gold": gold, "topics": topics, "config": cfg,
        "n_docs": len(texts), "n_pairs": n_pairs,
        "n_unpaired": int(len(texts) - n_linked),
        "classes": classes.tolist(), "class_counts": counts.tolist(),
        "chance_purity": float(counts.max() / counts.sum()),
        "paired": bool(pairs),
    }


def stats(keys=HEADLINE, n: int | None = "keep") -> dict:
    """Corpus statistics for Table 1, recomputed from the loaders themselves."""
    out = {}
    for key in keys:
        d = load(key, n=n)
        out[key] = {kk: d[kk] for kk in
                    ("label", "n_docs", "n_pairs", "n_unpaired", "classes",
                     "class_counts", "chance_purity", "paired")}
        out[key]["confounder"] = d["config"]["confounder"]
        out[key]["has_gold"] = d["config"]["has_gold"]
        out[key]["selection"] = d["config"]["selection"]
        out[key]["pseudo_groups"] = d["config"]["pseudo_groups"]
        out[key]["split"] = split_config(key)
    return out


def format_stats(st: dict) -> str:
    w = max(len(v["label"]) for v in st.values()) + 2
    lines = [f"{'corpus':<{w}}{'confounder':>12}{'#docs':>9}{'#pairs':>8}"
             f"{'#unpaired':>11}{'classes':>9}{'chance':>9}{'selection':>11}",
             "-" * (w + 69)]
    for v in st.values():
        lines.append(
            f"{v['label']:<{w}}{v['confounder']:>12}{v['n_docs']:>9,}"
            f"{(v['n_pairs'] or '--'):>8}{v.get('n_unpaired', v['n_docs']):>11,}"
            f"{len(v['classes']):>9}"
            f"{v['chance_purity']:>9.3f}{v['selection']:>11}")
    return "\n".join(lines)
