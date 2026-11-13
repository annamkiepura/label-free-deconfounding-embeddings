"""Every experiment in the paper, as one runner per claim.

All of them go through `protocols.run_seed`, so the split protocol is a declared
argument rather than something each experiment re-implements: unless a runner is
explicitly asked for ``protocol="pair"``, every learned component is fit on the
training documents and every number is measured on held-out documents.

Each runner returns a JSON-serialisable dict and saves it under the active
artifact scope, so figures can be redrawn from `raw/` alone. The result shapes
are deliberately uniform:

* single-condition-set runs (`headline`, `encoder_generalization`) carry
  ``per_seed`` + ``agg`` + ``significance``;
* sweeps (`selection_comparison`, `erasure_comparison`, `lambda_sweep`,
  `pseudo_group_sweep`, `corpus_size_sweep`, `protocol_comparison`) carry a
  ``variants`` dict of ``{label: {per_seed, agg, config}}``.

`plots.py` dispatches on ``experiment`` and needs nothing but the JSON.
"""

from __future__ import annotations

import json
import os

import numpy as np

import corpora
import paths
import protocols as P
from artifacts import artifact_path, save_result
from run_sae_experiment import encode_texts, pick_device
from sae_confounder_pipeline import paired_bootstrap_diff, seed_paired_test

#: The ten sentence encoders of the parent paper (Fan et al., 2025), plus the
#: two extras this project adds. `PRIMARY` is the paper's headline encoder.
PRIMARY_ENCODER = "e5-small"
PARENT_PAPER_ENCODERS = (
    "mini",                                  # all-MiniLM-L6-v2
    "gist-small", "gist-base", "gist-large",  # GIST family
    "e5-small", "e5-base", "e5-large",        # multilingual E5 family
    "mpnet",                                 # all-mpnet-base-v2
    "nomic-v2",
    "mxb",                                   # mxbai-embed-large-v1
)
EXTRA_ENCODERS = ("jina", "qwen3-0.6b")
#: Encoders that need a smaller batch (long context or large model).
ENCODER_BATCH = {"jina": 8, "qwen3-0.6b": 16, "e5-large": 128, "gist-large": 128,
                 "mxb": 128, "nomic-v2": 64}

#: The method recipe of Section 4.1. One dict, imported by every runner and by
#: the notebook, so the prose and the runs cannot drift apart.
RECIPE = dict(
    expansion_factor=8,
    k=32,
    n_epochs=200,
    erase="directions_leace",
    greedy_lambda=1.0,
    greedy_budget=64,
    greedy_pool=80,
    recall_k=10,
    top_features=32,
)
#: Baseline arms of Section 4.5. `kmeans_auto` is LEACE on the silhouette-chosen
#: pseudo-grouping itself -- the label-free "no SAE" arm.
BASELINE_ARMS = dict(kmeans_grid=(2, 4, 8), kmeans_auto=True, pca_grid=(1, 2, 4, 8),
                     random_control=True)
#: The three conditions Section 5 reports.
CORE_CONDITIONS = ("raw", "metadata_leace", "sae_leace")

#: Headline metrics, in reading order, with their direction.
METRIC_LOWER_IS_BETTER = {
    "leakage_auc": True,
    "leakage_auc_flip": True,
    "crossfit_leakage_auc": True,
    "crossfit_leakage_auc_flip": True,
    "nonlinear_leakage_auc_mlp_flip": True,
    "nonlinear_leakage_auc_gb_flip": True,
    "cluster_purity": True,
    "recall@1": False,
    "recall@10": False,
    "gold_sim_spearman": False,
    "neighbor_retention@10": False,
}
#: Metrics `oracle_gain_recovery` can express as a fraction of the oracle's gain.
#: `neighbor_retention@10` is deliberately excluded: raw scores exactly 1.0 on it
#: by construction, so raw is the BEST condition and "the oracle's gain over raw"
#: is negative -- a ratio against it is meaningless (it reads as >100% recovery
#: for any condition that distorts neighbourhoods less than the oracle). It is a
#: preservation floor to read alongside the table, not a gain to recover.
RECOVERY_METRICS = tuple(kk for kk in METRIC_LOWER_IS_BETTER
                         if kk != "neighbor_retention@10")


# --------------------------------------------------------------------------- #
# Embedding, with a disk cache
# --------------------------------------------------------------------------- #
def _cache_path(corpus: str, encoder: str, n) -> str:
    tag = f"{corpus}__{encoder}__n{n if n is not None else 'full'}.npy"
    return os.path.join(EMBED_CACHE, tag)


#: Where encoded corpora are cached. Encoding is deterministic, so a cache hit is
#: byte-identical to a fresh encode -- it only saves GPU time across sessions.
EMBED_CACHE = paths.EMBED_CACHE


def set_embed_cache(path: str) -> str:
    global EMBED_CACHE
    EMBED_CACHE = path
    os.makedirs(EMBED_CACHE, exist_ok=True)
    return EMBED_CACHE


def embed(corpus: str, encoder: str = PRIMARY_ENCODER, n="keep",
          device: str | None = None, batch_size: int | None = None,
          cache: bool = True, verbose: bool = True) -> dict:
    """Load a corpus and embed it once. Returns the corpus dict plus ``emb``."""
    d = corpora.load(corpus, n=n)
    n_tag = d["n_docs"]
    path = _cache_path(corpus, encoder, n_tag)
    if cache and os.path.exists(path):
        d["emb"] = np.load(path)
        if verbose:
            print(f"[embed] {corpus}/{encoder}: cache hit {d['emb'].shape} <- {path}")
        return d
    device = pick_device(device)
    # Imported here rather than at module level: `style_erasure_pipeline` pulls
    # in sentence-transformers and seaborn, which a cache-hit session (or a
    # figure/table rebuild) does not need and may not have installed.
    from style_erasure_pipeline import load_embedding_model
    model = load_embedding_model(encoder)
    bs = batch_size or ENCODER_BATCH.get(encoder, 256)
    emb = np.asarray(encode_texts(model, d["texts"], encoder, device, bs,
                                  show_progress_bar=verbose), dtype=np.float32)
    del model
    d["emb"] = emb
    if cache:
        os.makedirs(EMBED_CACHE, exist_ok=True)
        np.save(path, emb)
        if verbose:
            print(f"[embed] {corpus}/{encoder}: {emb.shape} -> cached at {path}")
    return d


# --------------------------------------------------------------------------- #
# The generic multi-seed run every experiment is built from
# --------------------------------------------------------------------------- #
def run_multiseed(
    data: dict,
    seeds=(0, 1, 2),
    protocol: str = "document",
    holdout: float | None = None,
    scheme: str | None = None,
    folds: int | None = None,
    retrieval_index: str = "full",
    conditions=None,
    nonlinear_probes=("mlp", "gb"),
    n_boot: int = 10000,
    significance_vs: str = "sae_leace",
    device: str | None = None,
    verbose: bool = True,
    **fit_kwargs,
) -> dict:
    """Run one configuration over `seeds` (x folds) and aggregate.

    `data` is what `embed` returns. `fit_kwargs` go to `protocols.fit_pipeline`
    (recipe + baseline arms). Selection defaults to the corpus's registered
    strategy, so unpaired corpora take top-N without the caller remembering to.
    """
    cfg = data["config"]
    fit_kwargs.setdefault("selection", cfg["selection"])
    fit_kwargs.setdefault("pseudo_groups", cfg["pseudo_groups"])
    # The corpus's own split policy unless the caller overrides it: SemEval is
    # small enough that a half-corpus fit is estimation-limited, so it uses
    # K-fold while the larger corpora keep the single 50/50 holdout.
    sp = dict(cfg.get("split") or {"scheme": "holdout", "holdout": holdout})
    scheme = scheme or sp.get("scheme", "holdout")
    folds = folds or sp.get("folds", 5)
    if holdout is None:
        holdout = sp.get("holdout", 0.5)
    if scheme == "kfold":
        holdout = 1.0 / folds          # what each fold actually holds out

    per_seed = []
    for s in seeds:
        splits = P.splits_for(data["n_docs"], data["source"], pairs=data["pairs"],
                              gold_sim=data["gold"], scheme=scheme, holdout=holdout,
                              folds=folds, seed=s, protocol=protocol)
        if verbose:
            print(f"  -- seed {s}" + (f" ({len(splits)} folds)"
                                     if len(splits) > 1 else "") + " --")
        # One result per seed: a single split is scored directly, a seed's K
        # folds are fit one by one and scored fold-wise (see `protocols`).
        r = P.run_seed(
            data["emb"], data["source"], pairs=data["pairs"],
            gold_sim=data["gold"], seed=s, protocol=protocol,
            conditions=conditions, nonlinear_probes=nonlinear_probes,
            device=device, verbose=verbose, splits=splits,
            retrieval_index=retrieval_index, **fit_kwargs)
        per_seed.append(r)

    agg = P.aggregate(per_seed)
    out = {
        "corpus": data["key"], "label": data["label"], "protocol": protocol,
        "holdout": holdout, "scheme": scheme, "folds": folds if scheme == "kfold" else None,
        "retrieval_index": retrieval_index,
        "n_fits": len(per_seed) * (folds if scheme == "kfold" else 1),
        "seeds": list(seeds), "paired": data["paired"],
        "conditions": list(per_seed[0]["metrics"]),
        "per_seed": [_strip_hits(r) for r in per_seed],
        "agg": agg,
        "significance": _significance(per_seed, agg, significance_vs, n_boot,
                                      fit_kwargs.get("recall_k", 10)),
        "config": {**per_seed[0]["config"], "protocol": protocol,
                   "pseudo_groups_k_per_seed": [r["config"].get("pseudo_groups_k")
                                                for r in per_seed],
                   "holdout": holdout, "scheme": scheme,
                   "folds": folds if scheme == "kfold" else None,
                   "retrieval_index": retrieval_index,
                   "n_train_docs": int(per_seed[0]["split"]["n_train_docs"]),
                   "dim": int(np.asarray(data["emb"]).shape[1]),
                   "nonlinear_probes": list(nonlinear_probes)},
        "corpus_stats": {kk: data[kk] for kk in
                         ("n_docs", "n_pairs", "classes", "class_counts",
                          "chance_purity")},
    }
    return out


def _run_config(kw: dict, data: dict, protocol: str, split: dict | None = None,
                **extra) -> dict:
    """The config block saved with every result.

    Merged from the recipe, the corpus registry and the protocol, so
    `tables.table_hyperparameters` can report what a run actually used instead of
    relying on a hand-kept list. Sweeps override the field they vary. `split` is
    the policy the runner actually used (see `_split_policy`); it is recorded
    rather than assumed, because SemEval runs K-fold while the others run a
    single holdout.
    """
    split = split or _split_policy(data)
    return {**kw, "protocol": protocol,
            "scheme": split["scheme"],
            "holdout": split["holdout"],
            "folds": split["folds"] if split["scheme"] == "kfold" else None,
            "pseudo_groups": data["config"]["pseudo_groups"],
            "selection": data["config"]["selection"], **extra}


def _split_policy(data: dict, scheme: str | None = None, holdout: float | None = None,
                  folds: int | None = None) -> dict:
    """Resolve a runner's split policy: the corpus registry's, unless overridden.

    Every runner goes through this so that a corpus is split the same way in
    every section -- SemEval K-fold, the rest a 50/50 holdout -- instead of each
    runner quietly building its own 50/50 split. Returns
    ``{"scheme", "holdout", "folds"}`` ready for `protocols.splits_for`.
    """
    sp = dict(data["config"].get("split") or {"scheme": "holdout", "holdout": 0.5})
    scheme = scheme or sp.get("scheme", "holdout")
    folds = int(folds or sp.get("folds", 5))
    if holdout is None:
        holdout = sp.get("holdout", 0.5)
    if scheme == "kfold":
        holdout = 1.0 / folds
    return {"scheme": scheme, "holdout": float(holdout), "folds": folds}


def _splits(data: dict, seed: int, protocol: str, policy: dict) -> list:
    """The `Split`s one seed contributes under `policy` (one, or K folds)."""
    return P.splits_for(data["n_docs"], data["source"], pairs=data["pairs"],
                        gold_sim=data["gold"], scheme=policy["scheme"],
                        holdout=policy["holdout"], folds=policy["folds"],
                        seed=seed, protocol=protocol)


def _tag(split) -> str:
    return f"seed {split.seed}" + (f" fold {split.fold + 1}/{split.n_folds}"
                                   if split.fold is not None else "")


def _bases(data: dict, seed: int, protocol: str, policy: dict, kw: dict,
           pg: "int | str", device) -> tuple:
    """A seed's splits and one trained SAE base per split.

    `pg` is the pseudo-group policy: ``"auto"`` (silhouette on the training
    documents) or an explicit int for the K sweep.

    Sweeps fit the expensive, variant-independent base once per split and pass
    it to every variant, so a variant's effect is never confounded with a
    different dictionary or a different training half.
    """
    splits = _splits(data, seed, protocol, policy)
    bases = [P.fit_base(data["emb"], sp, expansion_factor=kw["expansion_factor"],
                        k=kw["k"], n_epochs=kw["n_epochs"], pseudo_groups=pg,
                        device=device, verbose=False) for sp in splits]
    return splits, bases


def _seed_tag(splits) -> str:
    return f"seed {splits[0].seed}" + (f" ({len(splits)} folds)"
                                       if len(splits) > 1 else "")


def _strip_hits(r: dict) -> dict:
    """Per-pair hit vectors are only needed for the bootstrap; keep JSON small."""
    return {kk: v for kk, v in r.items() if kk != "hits"}


def _significance(per_seed, agg, ref: str, n_boot: int, recall_k: int) -> dict:
    """Paired bootstrap over pairs (per seed) + a seed-level test, vs `ref`."""
    if ref not in per_seed[0]["metrics"]:
        return {}
    out = {}
    has_hits = bool(per_seed[0].get("hits"))
    for c in per_seed[0]["metrics"]:
        if c == ref:
            continue
        entry = {}
        if has_hits and c in per_seed[0]["hits"]:
            entry["recall_bootstrap_per_seed"] = [
                paired_bootstrap_diff(s["hits"][ref], s["hits"][c],
                                      n_boot=n_boot, seed=s["seed"])
                for s in per_seed]
        for kk in ("leakage_auc", f"recall@{recall_k}", "cluster_purity",
                   "neighbor_retention@10"):
            if kk not in agg.get(ref, {}) or kk not in agg.get(c, {}):
                continue
            a, b = agg[ref][kk]["values"], agg[c][kk]["values"]
            if len(a) < 2:      # a paired test over one seed is undefined
                continue
            entry[f"seed_test[{kk}]"] = seed_paired_test(a, b)
        out[f"{ref} vs {c}"] = entry
    return out


# --------------------------------------------------------------------------- #
# 1. Headline: raw / oracle / SAE + every baseline arm
# --------------------------------------------------------------------------- #
def headline(
    corpus: str,
    encoder: str = PRIMARY_ENCODER,
    seeds=None,
    protocol: str = "document",
    n="keep",
    device: str | None = None,
    data: dict | None = None,
    save: bool = True,
    verbose: bool = True,
    **overrides,
) -> dict:
    """Section 5's table: the three conditions plus the Section 4.5 baselines.

    Runs under the document-level holdout by default, so every number describes
    documents no fitted component ever saw.
    """
    data = data or embed(corpus, encoder, n=n, device=device, verbose=verbose)
    seeds = seeds or data["config"]["seeds"]
    kw = {**RECIPE, **BASELINE_ARMS, **overrides}
    if verbose:
        print(f"\n{'#' * 66}\n#  headline | {data['label']} | {encoder} | "
              f"{protocol} holdout | {len(seeds)} seeds\n{'#' * 66}")
    res = run_multiseed(data, seeds=seeds, protocol=protocol, device=device,
                        verbose=verbose, **kw)
    res["experiment"] = "headline"
    res["encoder"] = encoder
    if save:
        save_result(res, f"headline_{corpus}_{encoder}.json")
    return res


# --------------------------------------------------------------------------- #
# 2. Protocol comparison: pair holdout vs document holdout
# --------------------------------------------------------------------------- #
def protocol_comparison(
    corpus: str,
    encoder: str = PRIMARY_ENCODER,
    seeds=(0, 1, 2),
    n="keep",
    device: str | None = None,
    data: dict | None = None,
    save: bool = True,
    verbose: bool = True,
    **overrides,
) -> dict:
    """"Unseen documents": the same corpus, seeds and recipe under both protocols.

    The pair holdout withholds retrieval *links* but shows every document to the
    SAE, the k-means proxy, the selector and the erasers; the document holdout
    withholds the documents themselves. Running both isolates how much of the
    reported performance depends on having seen the evaluation documents.
    """
    data = data or embed(corpus, encoder, n=n, device=device, verbose=verbose)
    kw = {**RECIPE, **overrides}
    kw.setdefault("random_control", True)
    # Both arms are pinned to a single 50/50 holdout, overriding the corpus's own
    # split policy. The comparison isolates the *protocol*, so the split ratio has
    # to be identical on both sides -- and K-fold is undefined for the pair
    # protocol anyway, since it splits links rather than documents.
    kw.setdefault("scheme", "holdout")
    kw.setdefault("holdout", 0.5)
    variants = {}
    for proto in ("pair", "document"):
        if verbose:
            print(f"\n{'#' * 66}\n#  protocol={proto} | {data['label']} | "
                  f"{len(seeds)} seeds\n{'#' * 66}")
        r = run_multiseed(data, seeds=seeds, protocol=proto, device=device,
                          verbose=verbose, **kw)
        variants[proto] = {kk: r[kk] for kk in
                           ("per_seed", "agg", "significance", "config", "conditions")}
    res = {
        "experiment": "protocol_comparison", "corpus": corpus,
        "label": data["label"], "encoder": encoder, "seeds": list(seeds),
        "paired": data["paired"], "variants": variants,
        "variant_order": ["pair", "document"],
        "scheme": "holdout", "holdout": 0.5,
        "note": "both arms pinned to one 50/50 holdout so only the protocol varies",
        "corpus_stats": {kk: data[kk] for kk in ("n_docs", "n_pairs", "chance_purity")},
    }
    if save:
        save_result(res, f"protocol_{corpus}_{encoder}.json")
    return res


# --------------------------------------------------------------------------- #
# 3. Selection: top-N vs greedy, with a size-matched top-N arm
# --------------------------------------------------------------------------- #
def selection_comparison(
    corpus: str,
    encoder: str = PRIMARY_ENCODER,
    seeds=(0, 1, 2),
    protocol: str = "document",
    top_n: int = 32,
    n="keep",
    scheme: str | None = None,
    holdout: float | None = None,
    folds: int | None = None,
    device: str | None = None,
    data: dict | None = None,
    save: bool = True,
    verbose: bool = True,
    **overrides,
) -> dict:
    """Which selector, at equal erasure budget?

    Greedy chooses its own objective-optimal |S| (typically 12-21 latents) while
    top-N erases a fixed 32, so a bare top-N-vs-greedy comparison confounds the
    selector with the size of the erased subspace. This runs three arms per seed
    off ONE trained SAE:

      * ``topn`` -- the fixed top-`top_n` by MI score (the paper's top-N);
      * ``greedy`` -- the label-free greedy objective, |S| chosen by J;
      * ``topn_matched`` -- top-N truncated to greedy's |S| **for that seed**, so
        selector quality is compared at an identical erasure budget.

    Only the third arm can separate "greedy picks better latents" from "greedy
    picks fewer latents".
    """
    data = data or embed(corpus, encoder, n=n, device=device, verbose=verbose)
    if not data["paired"]:
        raise ValueError(f"{corpus} is unpaired; greedy selection needs pairs")
    kw = {**RECIPE, **overrides}
    kw["top_features"] = top_n
    pg = data["config"]["pseudo_groups"]
    policy = _split_policy(data, scheme, holdout, folds)

    per_variant: dict = {"topn": [], "greedy": [], "topn_matched": []}
    X, y = data["emb"], data["source"]
    for s in seeds:
        splits, bases = _bases(data, s, protocol, policy, kw, pg, device)
        if verbose:
            print(f"\n-- {data['label']} | selection | {_seed_tag(splits)} --")
        fit_kw = dict(pseudo_groups=pg, erase=kw["erase"], recall_k=kw["recall_k"],
                      greedy_lambda=kw["greedy_lambda"], greedy_budget=kw["greedy_budget"],
                      greedy_pool=kw["greedy_pool"], random_control=False,
                      device=device, verbose=False)
        # Greedy first: its |S| on each fold sets that fold's size-matched arm.
        fgs = [P.fit_pipeline(X, y, sp, selection="greedy", base=b, **fit_kw)
               for sp, b in zip(splits, bases)]
        arms = {
            "greedy": fgs,
            "topn": [P.fit_pipeline(X, y, sp, selection="topn", top_features=top_n,
                                    base=b, **fit_kw) for sp, b in zip(splits, bases)],
            "topn_matched": [P.fit_pipeline(X, y, sp, selection="topn",
                                            top_features=int(len(fg.selected)),
                                            base=b, **fit_kw)
                             for sp, b, fg in zip(splits, bases, fgs)],
        }
        for name, fitteds in arms.items():
            r = P.evaluate_splits(fitteds, X, y, recall_k=kw["recall_k"],
                                  conditions=CORE_CONDITIONS, verbose=False)
            per_variant[name].append(r)
            if verbose:
                m, rk = r["metrics"]["sae_leace"], kw["recall_k"]
                print(f"   {name:<14s} |S|={r['n_selected']:3d}  "
                      f"leak={m['leakage_auc']:.4f}  "
                      f"recall@{rk}={m['recall@' + str(rk)]:.4f}")

    variants = {name: {"per_seed": [_strip_hits(r) for r in rs],
                       "agg": P.aggregate(rs),
                       "n_selected": [r["n_selected"] for r in rs]}
                for name, rs in per_variant.items()}
    res = {
        "experiment": "selection_comparison", "corpus": corpus,
        "label": data["label"], "encoder": encoder, "protocol": protocol,
        "seeds": list(seeds), "variants": variants,
        "variant_order": ["topn", "topn_matched", "greedy"],
        "conditions": list(CORE_CONDITIONS),
        "config": _run_config(kw, data, protocol, policy, selection="topn/greedy"),
        **policy,
    }
    if save:
        save_result(res, f"selection_{corpus}_{encoder}.json")
    return res


# --------------------------------------------------------------------------- #
# 4. Erasure mechanism, on one fixed selection
# --------------------------------------------------------------------------- #
def erasure_comparison(
    corpus: str,
    encoder: str = PRIMARY_ENCODER,
    seeds=(0, 1, 2),
    protocol: str = "document",
    mechanisms=("activations", "directions_leace", "directions_orth"),
    n="keep",
    scheme: str | None = None,
    holdout: float | None = None,
    folds: int | None = None,
    device: str | None = None,
    data: dict | None = None,
    save: bool = True,
    verbose: bool = True,
    **overrides,
) -> dict:
    """Section 6.2: hold the selected latents fixed, vary only how they are removed.

    The selection is run once per seed under ``directions_leace`` and then reused
    verbatim by all three mechanisms, so any difference is attributable to the
    mechanism rather than to a different latent set.
    """
    data = data or embed(corpus, encoder, n=n, device=device, verbose=verbose)
    kw = {**RECIPE, **overrides}
    pg = data["config"]["pseudo_groups"]
    sel_strategy = data["config"]["selection"]
    policy = _split_policy(data, scheme, holdout, folds)
    per_variant = {m: [] for m in mechanisms}
    X, y = data["emb"], data["source"]

    for s in seeds:
        splits, bases = _bases(data, s, protocol, policy, kw, pg, device)
        if verbose:
            print(f"\n-- {data['label']} | erasure | {_seed_tag(splits)} --")
        # One selection per fold under the paper's mechanism, reused verbatim.
        fixed = [P.fit_pipeline(X, y, sp, base=b, selection=sel_strategy,
                                pseudo_groups=pg, erase="directions_leace",
                                top_features=kw["top_features"],
                                greedy_lambda=kw["greedy_lambda"],
                                greedy_budget=kw["greedy_budget"],
                                greedy_pool=kw["greedy_pool"], recall_k=kw["recall_k"],
                                random_control=False, device=device,
                                verbose=False).selected
                 for sp, b in zip(splits, bases)]
        for mech in mechanisms:
            fitteds = [
                _override_selection(
                    P.fit_pipeline(X, y, sp, base=b, selection="topn",
                                   pseudo_groups=pg, erase=mech,
                                   top_features=kw["top_features"],
                                   random_control=False, device=device,
                                   verbose=False),
                    sel, mech, X, sp)
                for sp, b, sel in zip(splits, bases, fixed)]
            r = P.evaluate_splits(fitteds, X, y, recall_k=kw["recall_k"],
                                  conditions=CORE_CONDITIONS, verbose=False)
            per_variant[mech].append(r)
            if verbose:
                m, rk = r["metrics"]["sae_leace"], kw["recall_k"]
                print(f"   {mech:<18s} |S|={r['n_selected']:3d}  "
                      f"leak={m['leakage_auc']:.4f}  "
                      f"recall@{rk}={m.get('recall@' + str(rk), float('nan')):.4f}")

    res = {
        "experiment": "erasure_comparison", "corpus": corpus,
        "label": data["label"], "encoder": encoder, "protocol": protocol,
        "seeds": list(seeds),
        "variants": {m: {"per_seed": [_strip_hits(r) for r in rs],
                         "agg": P.aggregate(rs)} for m, rs in per_variant.items()},
        "variant_order": list(mechanisms),
        "conditions": list(CORE_CONDITIONS),
        "config": _run_config(kw, data, protocol, policy, erase="varied"),
        "note": "one selection per fold (directions_leace), reused by all mechanisms",
        **policy,
    }
    if save:
        save_result(res, f"erasure_{corpus}_{encoder}.json")
    return res


def _override_selection(fitted, selected, mech, embeddings, split):
    """Rebuild only the `sae_leace` eraser so a fixed latent set is used."""
    X_tr = np.asarray(embeddings, np.float32)[split.train_docs]
    D = fitted.decoder[np.asarray(selected)]
    if mech == "directions_leace":
        fitted.erasers["sae_leace"] = P._fit_leace(X_tr, X_tr @ D.T)
    elif mech == "directions_orth":
        fitted.erasers["sae_leace"] = P._fit_orth(D)
    elif mech == "activations":
        fitted.erasers["sae_leace"] = P._fit_leace(
            X_tr, fitted.feats_train[:, np.asarray(selected)])
    else:
        raise ValueError(mech)
    fitted.concepts["sae_leace"] = lambda X, _D=D: np.asarray(X, np.float32) @ _D.T
    fitted.selected = np.asarray(selected)
    fitted.config = {**fitted.config, "erase": mech, "selection": "fixed"}
    return fitted


# --------------------------------------------------------------------------- #
# 5. The greedy objective's lambda
# --------------------------------------------------------------------------- #
def lambda_sweep(
    corpus: str,
    encoder: str = PRIMARY_ENCODER,
    lambdas=(0.5, 1.0, 2.0, 4.0),
    seeds=(0, 1, 2),
    protocol: str = "document",
    n="keep",
    scheme: str | None = None,
    holdout: float | None = None,
    folds: int | None = None,
    device: str | None = None,
    data: dict | None = None,
    save: bool = True,
    verbose: bool = True,
    **overrides,
) -> dict:
    """Is lambda=1 a tuned choice, or just a point on a monotone trade-off dial?"""
    data = data or embed(corpus, encoder, n=n, device=device, verbose=verbose)
    if not data["paired"]:
        raise ValueError(f"{corpus} is unpaired; the greedy objective needs pairs")
    kw = {**RECIPE, **overrides}
    pg = data["config"]["pseudo_groups"]
    policy = _split_policy(data, scheme, holdout, folds)
    per_variant = {f"lambda={lam:g}": [] for lam in lambdas}
    X, y = data["emb"], data["source"]

    for s in seeds:
        splits, bases = _bases(data, s, protocol, policy, kw, pg, device)
        for lam in lambdas:
            fitteds = [P.fit_pipeline(
                X, y, sp, base=b, selection="greedy", pseudo_groups=pg,
                erase=kw["erase"], greedy_lambda=lam,
                greedy_budget=kw["greedy_budget"], greedy_pool=kw["greedy_pool"],
                recall_k=kw["recall_k"], random_control=False, device=device,
                verbose=False) for sp, b in zip(splits, bases)]
            r = P.evaluate_splits(fitteds, X, y, recall_k=kw["recall_k"],
                                  conditions=CORE_CONDITIONS, verbose=False)
            per_variant[f"lambda={lam:g}"].append(r)
            if verbose:
                m, rk = r["metrics"]["sae_leace"], kw["recall_k"]
                print(f"  [{_seed_tag(splits)}] lambda={lam:<5g} |S|={r['n_selected']:3d}  "
                      f"leak={m['leakage_auc']:.4f}  "
                      f"recall@{rk}={m['recall@' + str(rk)]:.4f}")

    res = {
        "experiment": "lambda_sweep", "corpus": corpus, "label": data["label"],
        "encoder": encoder, "protocol": protocol, "seeds": list(seeds),
        "lambdas": list(lambdas),
        "variants": {kk: {"per_seed": [_strip_hits(r) for r in rs],
                          "agg": P.aggregate(rs),
                          "n_selected": [r["n_selected"] for r in rs]}
                     for kk, rs in per_variant.items()},
        "variant_order": [f"lambda={lam:g}" for lam in lambdas],
        "conditions": list(CORE_CONDITIONS),
        "config": _run_config(kw, data, protocol, policy, selection="greedy",
                              greedy_lambda="varied"),
        **policy,
    }
    if save:
        save_result(res, f"lambda_{corpus}_{encoder}.json")
    return res


# --------------------------------------------------------------------------- #
# 6. Pseudo-group count K
# --------------------------------------------------------------------------- #
def pseudo_group_sweep(
    corpus: str,
    encoder: str = PRIMARY_ENCODER,
    ks=(2, 3, 4, 5, 6, 8),
    seeds=(0, 1, 2),
    protocol: str = "document",
    n="keep",
    scheme: str | None = None,
    holdout: float | None = None,
    folds: int | None = None,
    device: str | None = None,
    data: dict | None = None,
    save: bool = True,
    verbose: bool = True,
    **overrides,
) -> dict:
    """Is the label-free choice of K load-bearing, and what does it pick?

    The pipeline chooses K by the silhouette of k-means on the TRAINING
    documents (`protocols.choose_pseudo_groups`). This sweep runs the full
    pipeline at every candidate K so the sensitivity of the result to K is
    visible, and reports the silhouette pick next to the number of confounder
    classes -- the latter is an evaluation-side reference the method never
    sees, so agreement is a finding and disagreement is not a failure.
    """
    data = data or embed(corpus, encoder, n=n, device=device, verbose=verbose)
    kw = {**RECIPE, **overrides}
    sel_strategy = data["config"]["selection"]
    policy = _split_policy(data, scheme, holdout, folds)
    per_variant = {f"K={K}": [] for K in ks}
    sil_per_seed = []
    X, y = data["emb"], data["source"]

    for s in seeds:
        # The SAE does not depend on K, so train it once per split and re-score
        # against each K's grouping instead of retraining six dictionaries.
        splits, bases0 = _bases(data, s, protocol, policy, kw, int(ks[0]), device)
        # The same label-free chooser the pipeline uses, on TRAINING documents.
        sil_folds = [P.choose_pseudo_groups(np.asarray(X, np.float32)[sp.train_docs],
                                            ks=ks, seed=s)[1] for sp in splits]
        sil_per_seed.append({int(K): float(np.mean([f[int(K)] for f in sil_folds]))
                             for K in ks})
        for K in ks:
            fitteds = [P.fit_pipeline(
                X, y, sp, base=P.rescore_base(b0, X, int(K)),
                selection=sel_strategy, pseudo_groups=K, erase=kw["erase"],
                top_features=kw["top_features"], greedy_lambda=kw["greedy_lambda"],
                greedy_budget=kw["greedy_budget"], greedy_pool=kw["greedy_pool"],
                recall_k=kw["recall_k"], random_control=False, device=device,
                verbose=False) for sp, b0 in zip(splits, bases0)]
            r = P.evaluate_splits(fitteds, X, y, recall_k=kw["recall_k"],
                                  conditions=CORE_CONDITIONS, verbose=False)
            per_variant[f"K={K}"].append(r)
            if verbose:
                m = r["metrics"]["sae_leace"]
                print(f"  [{_seed_tag(splits)}] K={K:<2d} |S|={r['n_selected']:3d}  "
                      f"leak={m['leakage_auc']:.4f}  purity={m['cluster_purity']:.4f}")

    sil_mean = {K: float(np.mean([sp[K] for sp in sil_per_seed])) for K in
                map(int, ks)}
    jac = _selection_jaccard({kk: [r["selected"] for r in rs]
                              for kk, rs in per_variant.items()})
    res = {
        "experiment": "pseudo_group_sweep", "corpus": corpus, "label": data["label"],
        "encoder": encoder, "protocol": protocol, "seeds": list(seeds),
        "ks": [int(K) for K in ks],
        "silhouette": sil_mean, "silhouette_per_seed": sil_per_seed,
        "silhouette_k": int(max(sorted(sil_mean), key=sil_mean.get)),
        "n_classes": int(len(data["classes"])),        # reference only
        "mean_jaccard_across_k": jac,
        "variants": {kk: {"per_seed": [_strip_hits(r) for r in rs],
                          "agg": P.aggregate(rs)} for kk, rs in per_variant.items()},
        "variant_order": [f"K={K}" for K in ks],
        "conditions": list(CORE_CONDITIONS),
        "config": _run_config(kw, data, protocol, policy, pseudo_groups="varied"),
        **policy,
    }
    if save:
        save_result(res, f"pseudo_groups_{corpus}_{encoder}.json")
    return res


def score_function_sweep(
    corpus: str,
    encoder: str = PRIMARY_ENCODER,
    variants=("mi", "nmi", "corr_ratio"),
    seeds=(0, 1, 2),
    protocol: str = "document",
    n="keep",
    scheme: str | None = None,
    holdout: float | None = None,
    folds: int | None = None,
    device: str | None = None,
    data: dict | None = None,
    save: bool = True,
    verbose: bool = True,
    **overrides,
) -> dict:
    """Does the scoring function matter? MI vs normalized MI vs correlation ratio.

    All three rank the SAE latents against the same label-free k-means
    pseudo-grouping of the TRAINING documents; only the statistic differs:

      * ``mi``         -- the paper's score, mutual information between a
                          latent's binary firing indicator and the grouping;
      * ``nmi``        -- the uncertainty coefficient ``I(F_j; G) / H(F_j)``,
                          which removes the ceiling raw MI inherits from each
                          latent's own firing entropy;
      * ``corr_ratio`` -- the correlation ratio (eta-squared) of the *continuous*
                          activation across groups, so magnitude counts and not
                          just whether the latent fired.

    One SAE per split is shared by all three, so a difference is attributable to
    the ranking and not to a different dictionary or training half. Selection is
    top-N at the recipe's budget throughout, and the erasure mechanism is fixed,
    so ``|S|`` is identical across variants and only *which* latents are chosen
    varies. The Jaccard overlap of the selected sets is recorded alongside.
    """
    from dataclasses import replace as _replace

    from sae_confounder_pipeline import score_confounder_features_variants

    data = data or embed(corpus, encoder, n=n, device=device, verbose=verbose)
    kw = {**RECIPE, **overrides}
    pg = data["config"]["pseudo_groups"]
    policy = _split_policy(data, scheme, holdout, folds)
    per_variant = {v: [] for v in variants}
    selected_by_variant = {v: [] for v in variants}
    X, y = data["emb"], data["source"]

    for s in seeds:
        splits, bases = _bases(data, s, protocol, policy, kw, pg, device)
        if verbose:
            print(f"\n-- {data['label']} | scoring function | {_seed_tag(splits)} --")
        # Score every latent three ways per split, against that split's own
        # training grouping -- no evaluation document touches the ranking.
        scored = []
        for sp, b in zip(splits, bases):
            X_tr = np.asarray(X, np.float32)[sp.train_docs]
            scored.append(score_confounder_features_variants(
                b.feats_train, X_tr, n_pseudo_groups=int(b.pseudo_groups),
                seed=sp.seed, pseudo_group=b.pseudo_group_train))
        for v in variants:
            fitteds = [P.fit_pipeline(
                X, y, sp, base=_replace(b, scores=np.asarray(sc[v])),
                selection="topn", pseudo_groups=pg, erase=kw["erase"],
                top_features=kw["top_features"], recall_k=kw["recall_k"],
                random_control=False, device=device, verbose=False)
                for sp, b, sc in zip(splits, bases, scored)]
            r = P.evaluate_splits(fitteds, X, y, recall_k=kw["recall_k"],
                                  conditions=CORE_CONDITIONS, verbose=False)
            per_variant[v].append(r)
            selected_by_variant[v].append(r["selected"])
            if verbose:
                m, rk = r["metrics"]["sae_leace"], kw["recall_k"]
                print(f"   {v:<12s} |S|={r['n_selected']:3d}  "
                      f"leak={m['leakage_auc']:.4f}  "
                      f"purity={m['cluster_purity']:.4f}  "
                      f"recall@{rk}={m.get('recall@' + str(rk), float('nan')):.4f}")

    # Pairwise overlap of the selected sets, averaged over seeds: two rankings
    # can score very differently and still erase almost the same latents.
    def _jac(a, b):
        a, b = set(map(int, a)), set(map(int, b))
        return len(a & b) / max(1, len(a | b))
    vs = list(variants)
    overlap = {f"{a}|{b}": float(np.mean([_jac(x, y_) for x, y_ in
                                          zip(selected_by_variant[a],
                                              selected_by_variant[b])]))
               for i, a in enumerate(vs) for b in vs[i + 1:]}

    res = {
        "experiment": "score_function_sweep", "corpus": corpus,
        "label": data["label"], "encoder": encoder, "protocol": protocol,
        "seeds": list(seeds),
        "variants": {v: {"per_seed": [_strip_hits(r) for r in rs],
                         "agg": P.aggregate(rs)} for v, rs in per_variant.items()},
        "variant_order": vs,
        "selection_jaccard": overlap,
        "conditions": list(CORE_CONDITIONS),
        "config": _run_config(kw, data, protocol, policy, selection="topn"),
        "note": "one SAE per split shared by all three rankings; top-N at a "
                "fixed budget, so only which latents are chosen varies",
        **policy,
    }
    if verbose:
        print("\n  Jaccard overlap of the selected sets: "
              + "  ".join(f"{k}={v:.2f}" for k, v in overlap.items()))
    if save:
        save_result(res, f"score_function_{corpus}_{encoder}.json")
    return res


def _selection_jaccard(sel_by_variant: dict) -> float:
    """Mean pairwise Jaccard of the selected latent sets across variants (seed 0,
    first fold)."""
    import itertools
    sets = [set(v[0]) for v in sel_by_variant.values() if v]
    if len(sets) < 2:
        return float("nan")
    vals = [len(a & b) / max(1, len(a | b)) for a, b in itertools.combinations(sets, 2)]
    return float(np.mean(vals))


# --------------------------------------------------------------------------- #
# 7. Erasure-budget frontier (leakage vs quality as |S| grows)
# --------------------------------------------------------------------------- #
def budget_frontier(
    corpus: str,
    encoder: str = PRIMARY_ENCODER,
    grid=(1, 2, 4, 8, 12, 16, 24, 32, 48, 64),
    seeds=(0, 1, 2),
    protocol: str = "document",
    selectors=("topn", "greedy"),
    n="keep",
    scheme: str | None = None,
    holdout: float | None = None,
    folds: int | None = None,
    device: str | None = None,
    data: dict | None = None,
    save: bool = True,
    verbose: bool = True,
    **overrides,
) -> dict:
    """Leakage vs downstream quality as more latents are erased.

    Traces both selectors' orderings over the same budget grid, with the raw and
    metadata-oracle reference points, all measured on held-out documents. The
    quality axis is recall@k on paired corpora and within-source neighbourhood
    retention on unpaired ones, so the frontier is meaningful either way.
    """
    data = data or embed(corpus, encoder, n=n, device=device, verbose=verbose)
    kw = {**RECIPE, **overrides}
    pg = data["config"]["pseudo_groups"]
    paired = data["paired"]
    y_key = f"recall@{kw['recall_k']}" if paired else "neighbor_retention@10"
    # Greedy needs pairs, and it is traced only where it is the corpus's
    # selector (so `corpora.set_selection("topn")` drops it here too).
    selectors = tuple(s for s in selectors
                      if s != "greedy" or (paired and data["config"]["selection"] == "greedy"))
    policy = _split_policy(data, scheme, holdout, folds)
    curves = {s: [] for s in selectors}
    refs = {"raw": [], "metadata_leace": []}
    X, y = data["emb"], data["source"]

    for s in seeds:
        splits, bases = _bases(data, s, protocol, policy, kw, pg, device)
        anchors = [P.fit_pipeline(X, y, sp, base=b, selection="topn", pseudo_groups=pg,
                                  erase=kw["erase"], top_features=kw["top_features"],
                                  random_control=False, device=device, verbose=False)
                   for sp, b in zip(splits, bases)]
        ref_r = P.evaluate_splits(anchors, X, y, recall_k=kw["recall_k"],
                                  conditions=("raw", "metadata_leace"), verbose=False)
        for c in refs:
            refs[c].append({"leakage": ref_r["metrics"][c]["leakage_auc"],
                            "y": ref_r["metrics"][c][y_key], "seed": s})

        # One ordering per fold for each selector; a budget point erases the
        # first m of each fold's ordering and is scored fold-wise as one result.
        orders = {sel: [] for sel in selectors}
        for sp, b in zip(splits, bases):
            if "topn" in selectors:
                orders["topn"].append([int(j) for j in np.argsort(b.scores)[::-1]])
            if "greedy" in selectors:
                g_order, _ = P.select_confounder_features_greedy(
                    np.asarray(X, np.float32)[sp.train_docs],
                    b.feats_train, sp.train_pairs, b.scores,
                    budget=max(grid), candidate_pool=max(kw["greedy_pool"], max(grid)),
                    lam=kw["greedy_lambda"], pseudo_groups=b.pseudo_groups,
                    recall_k=kw["recall_k"], erase=kw["erase"], decoder=b.decoder,
                    seed=s, verbose=False, pseudo_group=b.pseudo_group_train)
                orders["greedy"].append([int(j) for j in g_order])

        for sel, per_fold in orders.items():
            usable = [g for g in grid if all(g <= len(o) for o in per_fold)]
            for mgrid in usable:
                fitteds = [_override_selection(
                    P.fit_pipeline(X, y, sp, base=b, selection="topn",
                                   pseudo_groups=pg, erase=kw["erase"],
                                   top_features=1, random_control=False,
                                   device=device, verbose=False),
                    np.array(order[:mgrid]), kw["erase"], X, sp)
                    for sp, b, order in zip(splits, bases, per_fold)]
                r = P.evaluate_splits(fitteds, X, y, recall_k=kw["recall_k"],
                                      conditions=("sae_leace",),
                                      nonlinear_probes=(), verbose=False)
                m = r["metrics"]["sae_leace"]
                curves[sel].append({"seed": s, "n_selected": int(mgrid),
                                    "leakage": m["leakage_auc"], "y": m[y_key],
                                    "cluster_purity": m["cluster_purity"]})
            if verbose:
                print(f"  [{_seed_tag(splits)}] {sel}: {len(usable)} budget points done")

    res = {
        "experiment": "budget_frontier", "corpus": corpus, "label": data["label"],
        "encoder": encoder, "protocol": protocol, "seeds": list(seeds),
        "grid": list(grid), "y_metric": y_key, "paired": paired,
        "curves": {s: _mean_by_budget(pts) for s, pts in curves.items()},
        "curves_per_seed": curves,
        "references": {c: {"leakage": float(np.mean([p["leakage"] for p in v])),
                           "y": float(np.mean([p["y"] for p in v]))}
                       for c, v in refs.items()},
        "config": _run_config(kw, data, protocol, policy,
                              selection="/".join(selectors)),
        **policy,
    }
    if save:
        save_result(res, f"frontier_{corpus}_{encoder}.json")
    return res


def _mean_by_budget(points: list) -> list:
    """Average the frontier points at each budget.

    Folds of one seed are averaged first, so the spread is across seeds -- the
    same unit of replication `protocols.aggregate` uses -- rather than across
    correlated fold fits.
    """
    out = []
    for m in sorted({p["n_selected"] for p in points}):
        sub = [p for p in points if p["n_selected"] == m]
        seeds = sorted({p["seed"] for p in sub})
        per_seed = {kk: [float(np.mean([p[kk] for p in sub if p["seed"] == sd]))
                         for sd in seeds] for kk in ("leakage", "y")}
        out.append({
            "n_selected": m,
            "leakage": float(np.mean(per_seed["leakage"])),
            "leakage_std": (float(np.std(per_seed["leakage"], ddof=1))
                            if len(seeds) > 1 else 0.0),
            "y": float(np.mean(per_seed["y"])),
            "y_std": float(np.std(per_seed["y"], ddof=1)) if len(seeds) > 1 else 0.0,
            "n_seeds": len(seeds), "n_fits": len(sub)})
    return out


# --------------------------------------------------------------------------- #
# 8. Encoder generalisation
# --------------------------------------------------------------------------- #
def encoder_generalization(
    corpora_keys=corpora.HEADLINE,
    encoders=PARENT_PAPER_ENCODERS,
    seeds=(0, 1, 2),
    protocol: str = "document",
    n="keep",
    out_json: str = "encoder_grid.json",
    device: str | None = None,
    save: bool = True,
    verbose: bool = True,
    **overrides,
) -> dict:
    """The full encoder x corpus grid, under the document-level holdout.

    **Resumable**: every finished cell is written to `out_json` immediately and
    skipped on a rerun, so a disconnected Colab session resumes where it stopped.
    Delete the file (or one cell's key) to redo it.

    Also records a label-free *dominance diagnostic* per cell -- the PC1 variance
    share of the raw embeddings, a property of the corpus rather than a fitted
    component, so it is computed on the whole corpus -- letting the grid be read as
    "label-free discovery works where one structure dominates the geometry"
    rather than as an unexplained per-encoder table.
    """
    path = artifact_path(out_json)
    grid: dict = {}
    if os.path.exists(path):
        with open(path) as fh:
            grid = json.load(fh)
        print(f"[resume] {path} holds {len([k for k in grid if not k.startswith('_')])} cells")

    for ck in corpora_keys:
        for enc in encoders:
            key = f"{ck}|{enc}"
            if key in grid:
                print(f"[skip] {key}")
                continue
            try:
                data = embed(ck, enc, n=n, device=device, verbose=verbose)
            except Exception as exc:                     # noqa: BLE001
                print(f"[fail] {key}: could not embed -- {type(exc).__name__}: {exc}")
                grid[key] = {"error": f"{type(exc).__name__}: {exc}"}
                _write_json(grid, path)
                continue
            if verbose:
                print(f"\n{'#' * 66}\n#  {ck} | {enc} | {protocol} holdout\n{'#' * 66}")
            kw = {**RECIPE, **overrides}
            kw.setdefault("random_control", True)
            r = run_multiseed(data, seeds=seeds, protocol=protocol, device=device,
                              verbose=False, **kw)
            grid[key] = {
                "corpus": ck, "encoder": enc, "label": data["label"],
                "dim": int(data["emb"].shape[1]),
                "paired": data["paired"],
                "diagnostics": _dominance(data["emb"]),
                "agg": r["agg"], "per_seed": r["per_seed"],
                "significance": r["significance"], "config": r["config"],
                "conditions": r["conditions"],
            }
            grid["_meta"] = {"experiment": "encoder_generalization",
                             "protocol": protocol, "seeds": list(seeds),
                             "encoders": list(encoders),
                             "corpora": list(corpora_keys)}
            _write_json(grid, path)
            if verbose:
                a = grid[key]["agg"]
                print(f"  raw leak={a['raw']['leakage_auc']['mean']:.3f} -> "
                      f"sae {a['sae_leace']['leakage_auc']['mean']:.3f} "
                      f"(oracle {a['metadata_leace']['leakage_auc']['mean']:.3f})")
    return grid


def _write_json(obj, path):
    from artifacts import _json_default
    with open(path, "w") as fh:
        json.dump(obj, fh, default=_json_default, indent=1)
    print(f"[saved raw] {path}")


def _dominance(embeddings: np.ndarray, ks=(2, 4)) -> dict:
    """Label-free measures of how strongly one coarse structure dominates."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = np.asarray(embeddings, np.float32)
    Xc = X - X.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    var = sv ** 2
    out = {"pc1_var_share": float(var[0] / var.sum()),
           "top5_var_share": float(var[:5].sum() / var.sum())}
    idx = np.random.default_rng(0).permutation(len(X))[: min(4000, len(X))]
    for K in ks:
        lab = KMeans(n_clusters=K, random_state=0, n_init="auto").fit_predict(X[idx])
        out[f"silhouette_k{K}"] = float(silhouette_score(X[idx], lab))
    return out


# --------------------------------------------------------------------------- #
# 9. Corpus size
# --------------------------------------------------------------------------- #
def corpus_size_sweep(
    corpus: str = "swiltra",
    encoder: str = PRIMARY_ENCODER,
    sizes=(400, 800, 1600, 2784),
    seeds=(0, 1, 2),
    protocol: str = "document",
    device: str | None = None,
    data: dict | None = None,
    save: bool = True,
    verbose: bool = True,
    **overrides,
) -> dict:
    """Does the result hold as the corpus grows? (Section: corpus size.)

    `sizes` counts **pairs** on a paired corpus (400 pairs = 800 documents, plus
    the same fraction of any unpaired remainder) and **documents** on an
    unpaired one. The subsample at each size is fixed by the
    size itself, not by the pipeline seed, so the three seeds vary the pipeline
    rather than the sample -- the variation the error bars are meant to describe.
    Linear and nonlinear leakage are reported for the oracle and for SAE-LEACE.
    """
    data = data or embed(corpus, encoder, n="keep", device=device, verbose=verbose)
    kw = {**RECIPE, **overrides}
    paired = data["paired"]
    unit = "pairs" if paired else "documents"
    n_unit = data["n_pairs"] if paired else data["n_docs"]
    # Clamp to what the corpus actually has, then de-duplicate: without this a
    # grid whose top end exceeds the corpus (or a subsampled QUICK run) collapses
    # several requested sizes onto the same value, and they would silently
    # overwrite each other under one key.
    sizes = sorted({int(min(s, n_unit)) for s in sizes})
    variants = {}

    for size in sizes:
        sub_rng = np.random.default_rng(size)          # fixed by size, not by seed
        if paired:
            # Subsample PAIRS and rebuild the document list from the pairs
            # array itself, rather than assuming the loader's (i, n1 + i)
            # layout -- that assumption silently breaks the moment a loader
            # drops or reorders rows. A partially paired corpus keeps its
            # unpaired remainder at the same fraction, stratified by class.
            arr = np.asarray(data["pairs"])
            sub = sub_rng.permutation(len(arr))[:size]
            unpaired = np.setdiff1d(np.arange(data["n_docs"]), np.unique(arr))
            y = np.asarray(data["source"])
            frac = size / len(arr)
            take_u = []
            for c in np.unique(y[unpaired]) if len(unpaired) else []:
                pool = sub_rng.permutation(unpaired[y[unpaired] == c])
                take_u.append(pool[: int(round(frac * len(pool)))])
            doc_idx = np.concatenate([arr[sub, 0], arr[sub, 1]] + take_u)
            sub_pairs = [(j, size + j) for j in range(size)]
        else:
            y = data["source"]
            take = []
            per = size // len(np.unique(y))
            for c in np.unique(y):
                idx = np.where(y == c)[0]
                take.append(sub_rng.permutation(idx)[:per])
            doc_idx = np.concatenate(take)
            sub_pairs = None
        sub_data = {
            **data, "emb": data["emb"][doc_idx],
            "source": np.asarray(data["source"])[doc_idx],
            "texts": [data["texts"][i] for i in doc_idx],
            "pairs": sub_pairs, "gold": None,
            "n_docs": int(len(doc_idx)), "n_pairs": int(size) if paired else 0,
        }
        if verbose:
            print(f"\n-- {data['label']} | size={size} {unit} "
                  f"({len(doc_idx)} docs) --")
        r = run_multiseed(sub_data, seeds=seeds, protocol=protocol, device=device,
                          verbose=False, random_control=False, **kw)
        variants[str(size)] = {
            "size": size, "unit": unit, "n_docs": int(len(doc_idx)),
            "per_seed": r["per_seed"], "agg": r["agg"], "config": r["config"],
            "scheme": r["scheme"], "holdout": r["holdout"], "folds": r["folds"]}
        if verbose:
            a = r["agg"]
            gb = a["sae_leace"].get("nonlinear_leakage_auc_gb", {}).get("mean", float("nan"))
            print(f"   raw {a['raw']['leakage_auc']['mean']:.3f} | "
                  f"oracle {a['metadata_leace']['leakage_auc']['mean']:.3f} | "
                  f"sae {a['sae_leace']['leakage_auc']['mean']:.3f}  (gb: {gb:.3f})")

    res = {
        "experiment": "corpus_size_sweep", "corpus": corpus, "label": data["label"],
        "encoder": encoder, "protocol": protocol, "seeds": list(seeds),
        "sizes": list(sizes), "unit": unit,
        "paired": paired, "variants": variants,
        "variant_order": [str(s) for s in sizes],
        "conditions": list(CORE_CONDITIONS),
        "config": _run_config(kw, data, protocol),
    }
    if save:
        save_result(res, f"corpus_size_{corpus}_{encoder}.json")
    return res


# --------------------------------------------------------------------------- #
# 10. Oracle-gain recovery (the framing Section 5 argues)
# --------------------------------------------------------------------------- #
def oracle_gain_recovery(result: dict, min_gain: float = 0.02,
                         sae: str = "sae_leace", oracle: str = "metadata_leace",
                         raw: str = "raw", save: bool = True) -> dict:
    """What fraction of the oracle's improvement over raw does the SAE recover?

        recovery = (raw - sae) / (raw - oracle)     # lower-is-better metrics
        recovery = (sae - raw) / (oracle - raw)     # higher-is-better metrics

    Computed per seed and then averaged, so the spread is the spread of the ratio
    and not a ratio of averages. Under K-fold the folds of one seed are averaged
    first (they share training documents and are not independent replicates),
    so the unit is the seed here exactly as in `protocols.aggregate`. A metric
    whose oracle gain is below `min_gain` is reported as n/a: a ratio with a
    denominator near zero says nothing.
    """
    per_seed = result["per_seed"]
    keys = [kk for kk in RECOVERY_METRICS
            if all(kk in s["metrics"].get(c, {}) for s in per_seed
                   for c in (raw, oracle, sae))]
    seeds = sorted({int(s["seed"]) for s in per_seed})
    out = {"experiment": "gain_recovery", "corpus": result.get("corpus"),
           "label": result.get("label"), "encoder": result.get("encoder"),
           "protocol": result.get("protocol"), "min_gain": min_gain,
           "n_seeds": len(seeds), "n_fits": len(per_seed), "metrics": {}}

    def per_seed_mean(cond, kk):
        return [float(np.mean([s["metrics"][cond][kk] for s in per_seed
                               if int(s["seed"]) == sd])) for sd in seeds]

    for kk in keys:
        lower = METRIC_LOWER_IS_BETTER[kk]
        recs, gains = [], []
        for r, o, a in zip(per_seed_mean(raw, kk), per_seed_mean(oracle, kk),
                           per_seed_mean(sae, kk)):
            gain = (r - o) if lower else (o - r)
            got = (r - a) if lower else (a - r)
            gains.append(gain)
            recs.append(got / gain if abs(gain) >= min_gain else float("nan"))
        arr = np.asarray(recs, float)
        usable = arr[~np.isnan(arr)]
        out["metrics"][kk] = {
            "lower_is_better": bool(lower),
            "raw": float(np.mean(per_seed_mean(raw, kk))),
            "oracle": float(np.mean(per_seed_mean(oracle, kk))),
            "sae": float(np.mean(per_seed_mean(sae, kk))),
            "oracle_gain": float(np.mean(gains)),
            "recovery": {"mean": float(usable.mean()) if len(usable) else None,
                         "std": float(usable.std(ddof=1)) if len(usable) > 1 else 0.0,
                         "values": [None if np.isnan(v) else float(v) for v in arr]},
        }
    if save:
        save_result(out, f"gain_recovery_{result.get('corpus')}_"
                         f"{result.get('encoder', PRIMARY_ENCODER)}.json")
    return out


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
HEADLINE_KEYS = ("leakage_auc", "nonlinear_leakage_auc_gb_flip", "cluster_purity",
                 "recall@1", "recall@10", "neighbor_retention@10",
                 "gold_sim_spearman")


def format_table(result: dict, keys=HEADLINE_KEYS, conditions=None) -> str:
    """Rows = conditions, columns = metrics (mean +/- std)."""
    agg = result["agg"]
    conds = list(conditions or result["conditions"])
    keys = [kk for kk in keys if any(kk in agg.get(c, {}) for c in conds)]
    w = max(len(c) for c in conds) + 2
    head = f"{'condition':<{w}}" + "".join(f"{kk[:20]:>22}" for kk in keys)
    lines = [head, "-" * len(head)]
    for c in conds:
        row = f"{c:<{w}}"
        for kk in keys:
            v = agg.get(c, {}).get(kk)
            row += (f"{v['mean']:>13.4f} ±{v['std']:.3f} " if v else f"{'--':>22}")
        lines.append(row)
    sp = result["per_seed"][0]["split"]
    if sp.get("n_folds") and sp.get("foldwise"):
        _s = len(result["seeds"])
        scope = (f"{sp['n_folds']}-fold x {_s} seed" + ("s" if _s != 1 else "") + "; "
                 f"{sp['n_train_docs']} train / {sp['n_test_docs_per_fold']} held out "
                 f"per fold, {sp['n_test_pairs']} pairs queried per seed")
    elif sp.get("n_folds"):
        _s = len(result["seeds"])
        scope = (f"{sp['n_folds']}-fold x {_s} seed" + ("s" if _s != 1 else "") + "; "
                 f"{sp['n_train_docs']} train / {sp['n_test_docs']} scored per fold, "
                 f"{sp['n_test_pairs']} pairs")
    else:
        scope = (f"{result.get('protocol', '?')} holdout; {sp['n_test_docs']} held-out "
                 f"docs, {sp['n_test_pairs']} held-out pairs; "
                 f"{len(result['seeds'])} seed"
                 + ("s" if len(result["seeds"]) != 1 else ""))
    idx = result.get("retrieval_index")
    ks = result.get("config", {}).get("pseudo_groups_k_per_seed")
    k_note = f"; pseudo-groups K = {sorted(set(ks))}" if ks and all(ks) else ""
    lines.append(f"({scope}" + (f"; retrieval index = {idx} corpus" if idx else "")
                 + k_note + ")")
    return "\n".join(lines)


def format_variants(result: dict, key: str = "leakage_auc",
                    condition: str = "sae_leace", keys=None) -> str:
    """One row per sweep variant, for the variant-shaped results."""
    keys = keys or (key, "recall@10", "neighbor_retention@10", "cluster_purity")
    order = result.get("variant_order") or list(result["variants"])
    w = max(len(v) for v in order) + 2
    head = f"{'variant':<{w}}{'|S|':>6}" + "".join(f"{kk[:18]:>20}" for kk in keys)
    lines = [head, "-" * len(head)]
    for v in order:
        e = result["variants"][v]
        agg = e["agg"].get(condition, {})
        sel = e.get("n_selected")
        sel_s = (f"{np.mean(sel):.0f}" if sel else
                 f"{np.mean([s['n_selected'] for s in e['per_seed']]):.0f}")
        row = f"{v:<{w}}{sel_s:>6}"
        for kk in keys:
            m = agg.get(kk)
            row += (f"{m['mean']:>12.4f} ±{m['std']:.3f}" if m else f"{'--':>20}")
        lines.append(row)
    return "\n".join(lines)


def format_gain_recovery(res: dict) -> str:
    rows = res["metrics"]
    w = max(len(kk) for kk in rows) + 2
    lines = [f"{'metric':<{w}}{'raw':>9}{'oracle':>9}{'SAE':>9}"
             f"{'oracle gain':>13}{'recovered':>18}", "-" * (w + 58)]
    for kk, m in rows.items():
        rec = m["recovery"]["mean"]
        s = "n/a" if rec is None else f"{rec:.1%} ±{m['recovery']['std']:.1%}"
        lines.append(f"{kk:<{w}}{m['raw']:>9.3f}{m['oracle']:>9.3f}{m['sae']:>9.3f}"
                     f"{m['oracle_gain']:>13.3f}{s:>18}")
    lines.append(f"(mean over {res['n_seeds']} seeds; 1.0 = matched the oracle, "
                 f"0.0 = no better than raw; n/a = oracle gain < {res['min_gain']})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 11. The estimation floor
# --------------------------------------------------------------------------- #
def estimation_floor(result: dict, sae: str = "sae_leace",
                     oracle: str = "metadata_leace", raw: str = "raw",
                     min_gain: float = 0.02, save: bool = True) -> dict:
    """Read the held-out table against the *achievable* floor, not against 0.5.

    The metadata oracle is label-supervised: it is handed the true confounder and
    removes it optimally in-sample. So whatever leakage it still shows on held-out
    documents is not a limitation of erasure -- it is the estimation error of
    fitting a d x d whitening on a finite training half. Chance (0.5) is the
    *asymptotic* floor; the oracle is the floor actually reachable at this corpus
    size and encoder dimension, and it is the right reference for the label-free
    method.

    Per metric this reports:

    ``floor``      the oracle's held-out value -- what is attainable here;
    ``headroom``   ``raw - floor``, how much of the metric is removable at all;
    ``attained``   ``raw - sae``, how much the label-free method removed;
    ``floor_gap``  ``sae - floor``, what separates it from the oracle;
    ``recovery``   ``attained / headroom``, the same ratio Section 5 quotes.

    Signs are oriented so that positive ``headroom`` and ``attained`` always mean
    "moved in the good direction", whichever way the metric points.

    The ``diagnostics`` block carries the training rows per fit and their ratio to
    the embedding dimension: below roughly 2 the whitening is barely determined
    and a low floor should be read as a sample-size result, not a method result.
    """
    per_seed = result["per_seed"]
    cfg = result.get("config", {})
    n_train = cfg.get("n_train_docs") or per_seed[0]["split"]["n_train_docs"]
    dim = cfg.get("dim")
    keys = [kk for kk in RECOVERY_METRICS
            if all(kk in s["metrics"].get(c, {}) for s in per_seed
                   for c in (raw, oracle, sae))]

    out = {"experiment": "estimation_floor", "corpus": result.get("corpus"),
           "label": result.get("label"), "encoder": result.get("encoder"),
           "protocol": result.get("protocol"), "scheme": result.get("scheme"),
           "n_fits": len(per_seed), "n_seeds": len(result.get("seeds") or []),
           "min_gain": min_gain, "metrics": {},
           "diagnostics": {"n_train_docs": int(n_train), "dim": dim,
                           "n_over_d": (round(n_train / dim, 2) if dim else None)}}
    for kk in keys:
        lower = METRIC_LOWER_IS_BETTER[kk]
        r, o, a = (float(np.mean([s["metrics"][c][kk] for s in per_seed]))
                   for c in (raw, oracle, sae))
        headroom = (r - o) if lower else (o - r)
        attained = (r - a) if lower else (a - r)
        gap = (a - o) if lower else (o - a)
        out["metrics"][kk] = {
            "lower_is_better": bool(lower), "raw": r, "floor": o, "sae": a,
            "headroom": headroom, "attained": attained, "floor_gap": gap,
            # Same guard as `oracle_gain_recovery`: a ratio whose denominator is
            # a rounding error is not a percentage, it is noise amplified.
            "recovery": (attained / headroom) if abs(headroom) >= min_gain else None,
        }
    if save:
        save_result(out, f"estimation_floor_{result.get('corpus')}_"
                         f"{result.get('encoder', PRIMARY_ENCODER)}.json")
    return out


def format_estimation_floor(res: dict) -> str:
    rows = res["metrics"]
    w = max(len(kk) for kk in rows) + 2
    d = res["diagnostics"]
    lines = [
        f"{res.get('label')} -- {res.get('scheme', 'holdout')} split, "
        f"{d['n_train_docs']} training documents"
        + (f", d={d['dim']} (n/d = {d['n_over_d']})" if d.get("dim") else ""),
        f"{'metric':<{w}}{'raw':>9}{'floor':>9}{'SAE':>9}"
        f"{'headroom':>11}{'attained':>10}{'to floor':>10}{'recovered':>11}",
        "-" * (w + 69)]
    for kk, m in rows.items():
        rec = m["recovery"]
        lines.append(
            f"{kk:<{w}}{m['raw']:>9.3f}{m['floor']:>9.3f}{m['sae']:>9.3f}"
            f"{m['headroom']:>11.3f}{m['attained']:>10.3f}{m['floor_gap']:>10.3f}"
            + (f"{rec:>11.0%}" if rec is not None else f"{'n/a':>11}"))
    lines.append("floor = the label-supervised oracle on the SAME held-out documents:")
    lines.append("what is attainable at this corpus size, not the asymptotic 0.5.")
    lines.append(f"n/a = headroom below {res.get('min_gain', 0.02)}, so the ratio "
                 "would be noise rather than a percentage.")
    if d.get("n_over_d") is not None and d["n_over_d"] < 2:
        lines.append(f"NOTE: n/d = {d['n_over_d']} < 2 -- the whitening is barely "
                     "determined, so the floor is sample-size limited.")
    return "\n".join(lines)
