"""The in-distribution and frozen-transfer experiments, one runner each.

Every runner fits through the existing engine -- `protocols.fit_pipeline` with
`experiments.RECIPE` and the corpus registry's split policy -- and scores
through `evaluate_fits`, which applies the protocol of `audit` (plan Sec. 3).
Nothing here re-implements the method.

====================  ================================================
Runner                Exhibits it feeds (plan numbering)
====================  ================================================
dataset_inventory     New Table 1; Table A1
encoder_inventory     Table A2
run_id                New Table 2; Tables B1-B4 (Raw, PCA-1,
                      Metadata-LEACE, SAE-LEACE); saves the frozen maps
run_id_kmeans         New Table 2 (k-means LEACE, no SAE, on run_id's splits)
run_ood               New Figure 1; Tables C1-C3; Figure C1 -- reuses
                      the maps `run_id` saved, fits nothing
run_budget            Figure D7; Table D2; New Figure 2's MI curve
run_mechanism         New Table 3; Table D3
run_pseudo_groups     Figure D1
run_score_function    Table D1
run_corpus_size       Figure D2; Table D4
run_protocol_diagnostic  Table D5
run_descriptive       Tables D6-D8; Figures D3-D5
run_encoder_cell      Tables B5-B6; Figure D6 (the ten-encoder study, rerun)
====================  ================================================

Each runner writes one JSON into ``<results root>/raw/<experiment>/`` (the
notebooks/02_main_experiments.ipynb sets the root to ``results/main``) and is resumable: a result
already on disk is loaded instead of recomputed unless ``overwrite=True``.
`revised_exhibits` builds every table and figure from those JSONs alone.

Every fit selects latents by MI top-N (N = 32), the plan's method; the
registry's greedy selector is never used here, whatever the session's
`corpora.set_selection` says.
"""

from __future__ import annotations

import functools
import json
import os
import time
from dataclasses import replace as _replace

import numpy as np

import artifacts
import audit
import corpora
import experiments as E
import ood
import protocols as P
from artifacts import artifact_path, load_result, save_result, save_text

#: The plan's selector (Sec. 3.2 of the paper): MI top-N on every corpus.
SELECTION = "topn"

#: The four comparison conditions, in the plan's order (Sec. 6.1).
ID_CONDITIONS = ("raw", "pca_top1", "metadata_leace", "sae_leace")
LABELS = {"raw": "Raw", "pca_top1": "PCA-1", "metadata_leace": "Metadata-LEACE",
          "sae_leace": "SAE-LEACE", "kmeans_leace": "k-means LEACE"}
#: The frozen transformations the OOD study transfers.
OOD_TRANSFORMS = ("pca_top1", "metadata_leace", "sae_leace")

#: The five ID corpora (New Table 1 order) and the four the draft already had.
ID_CORPORA = ("semeval", "swiltra", "bills_tweets", "scotus_oyez_lexis", "cap")
ORIGINAL = ("semeval", "swiltra", "bills_tweets", "scotus_oyez_lexis")
PAIRED_ORIGINAL = ("semeval", "swiltra", "scotus_oyez_lexis")
#: The donor/encoder combinations of plan Sec. 5.2 and their fixed seeds.
OOD_MATRIX = (("e5-small", "cap"), ("e5-small", "scotus_oyez_lexis"),
              ("e5-small", "swiltra"), ("mini", "cap"), ("mini", "scotus_oyez_lexis"))
OOD_SEEDS = (0, 1, 2)
#: Further (encoder, donor) settings whose `run_id` fits save their frozen maps for
#: `run_ood`, beside `OOD_MATRIX`; the notebooks register the three-class donors here, so
#: the plan's own exhibits (which read `OOD_MATRIX`) are unaffected.
MAP_DONORS: set = set()
#: Sweeps (plan Appendix D): three seeds, the registry's split policy.
SWEEP_SEEDS = (0, 1, 2)
BUDGET_GRID = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
K_GRID = (2, 3, 4, 5, 6, 8)
SCORE_VARIANTS = ("mi", "nmi", "corr_ratio")
MECHANISMS = ("activations", "directions_orth", "directions_leace")
#: The ten-encoder study (plan B.5): the parent paper's encoders, three
#: conditions, three seeds, the four original corpora.
ENCODER_STUDY = E.PARENT_PAPER_ENCODERS
ENCODER_CONDITIONS = ("raw", "metadata_leace", "sae_leace")
ENCODER_SEEDS = (0, 1, 2)
#: PCA removal ranks carried through the encoder study alongside those three.
#: Rank 1 is the plan's baseline; rank 3 is the small-rank variant that competes
#: with SAE-LEACE in distribution. Each costs one SVD per fit and no SAE.
ENCODER_PCA_RANKS = (1, 3)
#: The wider grid added to cached cells afterwards (no SAE refit), so each encoder
#: gets a PCA rank curve to compare with SAE-LEACE at matched leakage. Table B5
#: keeps showing `ENCODER_PCA_RANKS`; `alternatives.table_encoder_ranks` reads this.
ENCODER_PCA_GRID = (1, 2, 3, 4, 8, 16, 32)
#: Corpus-size grids: pairs for SwiLTra, documents for Bills--Tweets.
SIZE_GRID = {"swiltra": (400, 800, 1600, 2784), "bills_tweets": (1000, 2500, 5000, 9933)}

#: The raw-results sub-folder of every runner (``<root>/raw/<folder>/``).
FOLDERS = {"dataset_inventory": "inventory", "encoder_inventory": "inventory",
           "run_encoder_cell": "encoders", "run_id": "id", "run_ood": "ood",
           "run_budget": "budget", "run_mechanism": "mechanism",
           "extend_id_nonlinear": "id", "run_id_kmeans": "id",
           "extend_mechanism_nonlinear": "mechanism",
           "run_pseudo_groups": "pseudo_groups", "run_score_function": "score_function",
           "run_corpus_size": "corpus_size", "run_protocol_diagnostic": "protocol",
           "run_descriptive": "descriptive"}


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #
def scoped(fn):
    """Run `fn` with artifacts routed to ``<root>/{raw,...}/<its folder>/``."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with artifacts.experiment(FOLDERS[fn.__name__]):
            return fn(*args, **kwargs)
    return wrapper


def raw_path(folder: str, name: str) -> str:
    """Where a runner's JSON lives, independent of the active scope."""
    return os.path.join(artifacts.BASE_DIR, artifacts.KIND_DIRS["raw"], folder, name)


class Erasers:
    """Minimal stand-in for a fitted pipeline: a split plus frozen erasers.

    `evaluate_fits` needs only these attributes, so an arm that fits no SAE (PCA
    removal, say) can be scored on exactly the same split as one that does.
    """

    def __init__(self, split, erasers):
        self.split, self.erasers = split, erasers
        self.selected, self.config = [], {"pseudo_groups_k": 0}


def kmeans_eraser(X_tr, K: int, seed: int) -> tuple:
    """LEACE on the one-hot k-means grouping of the training documents (rank K - 1) --
    the fit `protocols.fit_pipeline` uses for its k-means arms; no SAE. Returns
    ``(eraser, labels)``."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=int(K), random_state=int(seed), n_init="auto").fit(X_tr)
    eraser = P._fit_leace(X_tr, P._one_hot(km.labels_, np.arange(int(K))), standardise=False)
    return eraser, km.labels_


def _cached(name: str, overwrite: bool):
    path = artifact_path(name)
    if not overwrite and os.path.exists(path):
        print(f"[cached] {path}")
        return load_result(path)
    return None


def load(corpus: str, encoder: str, device=None, verbose: bool = False) -> dict:
    """`experiments.embed` with the plan's top-N selector pinned in the config."""
    data = E.embed(corpus, encoder, device=device, verbose=verbose)
    data["config"] = {**data["config"], "selection": SELECTION}
    return data


def _splits(data: dict, seed: int) -> list:
    """The registry's split policy for one seed (SemEval 5-fold, else 50/50)."""
    return E._splits(data, seed, "document", E._split_policy(data))


def _kw(overrides: dict) -> dict:
    return {**E.RECIPE, **overrides}


def fit_seed(data: dict, seed: int, *, pca: bool = True, splits=None,
             device: str | None = None, **overrides) -> tuple:
    """One seed's fits under the recipe: ``(splits, fitted pipelines)``.

    Fits the plan's conditions only -- the metadata-LEACE reference, SAE-LEACE
    and (with `pca`) PCA-1 -- and none of the draft's other baseline arms.
    """
    kw = _kw(overrides)
    splits = splits or _splits(data, seed)
    fitteds = [P.fit_pipeline(data["emb"], data["source"], sp, **kw,
                              selection=data["config"]["selection"],
                              pseudo_groups=data["config"]["pseudo_groups"],
                              pca_grid=(1,) if pca else (), kmeans_grid=(),
                              kmeans_auto=False, random_control=False, device=device)
               for sp in splits]
    return splits, fitteds


def _refit(data: dict, sp, base, *, top_features: int, erase: str = "directions_leace",
           pseudo_groups=None, pca: bool = False, device=None):
    """A pipeline on a pre-trained base (the sweeps share one SAE per split)."""
    return P.fit_pipeline(data["emb"], data["source"], sp, base=base, selection="topn",
                          top_features=int(top_features), erase=erase,
                          pseudo_groups=pseudo_groups or data["config"]["pseudo_groups"],
                          pca_grid=(1,) if pca else (), kmeans_grid=(), kmeans_auto=False,
                          random_control=False, device=device)


def _bases(data: dict, splits, pseudo_groups=None, device=None, **overrides) -> list:
    kw = _kw(overrides)
    return [P.fit_base(data["emb"], sp, expansion_factor=kw["expansion_factor"], k=kw["k"],
                       n_epochs=kw["n_epochs"],
                       pseudo_groups=pseudo_groups or data["config"]["pseudo_groups"],
                       device=device) for sp in splits]


def _config(data: dict, overrides: dict, **extra) -> dict:
    """What a run used, saved with it so the tables need no hand-kept list."""
    return {"recipe": _kw(overrides), "selection": data["config"]["selection"],
            "pseudo_groups": data["config"]["pseudo_groups"],
            "split": E._split_policy(data), "audit": audit.PROTOCOL, **extra}


def _corpus_stats(data: dict) -> dict:
    classes, counts = np.unique(np.asarray(data["source"]), return_counts=True)
    linked = len(np.unique(np.asarray(data["pairs"]))) if data.get("pairs") else 0
    out = {"n_docs": int(data["n_docs"]), "n_pairs": int(data["n_pairs"]),
           "n_unpaired": int(data["n_docs"] - linked),
           "classes": [str(c) for c in classes], "class_counts": counts.tolist(),
           "confounder": data["config"]["confounder"]}
    if data.get("topics") is not None:
        out["n_topics"] = int(len(np.unique(data["topics"])))
    return out


def _stat(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "values": v.tolist(), "n": int(len(v))}


# --------------------------------------------------------------------------- #
# The plan-3 evaluator
# --------------------------------------------------------------------------- #
def _unit(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


#: Most entries of the query x corpus similarity matrix `pair_hits` holds at once (0.5 GiB
#: of float32). Larger problems are scored in blocks of queries, which bounds the memory and
#: changes nothing else: a query's rank depends on its own row only. Every corpus of the
#: revision fits in one block (the largest, SwiLTra DE/FR/IT, needs 35M entries); the full
#: Fedlex corpus (about 10^5 queries x 2 x 10^5 documents) does not.
PAIR_HITS_MAX_ENTRIES = 1 << 27


def pair_hits(E_full, pairs, ks=(1, 10)) -> dict:
    """Per-pair retrieval outcomes in {0, 0.5, 1} at each k.

    Held-out pairs are queries against the whole transformed corpus, the query
    itself excluded, in both directions -- `sae_confounder_pipeline.
    pair_hit_vector`, computed for several k from one similarity pass. In a
    three-class corpus a query's other counterpart is excluded too, so each
    target is ranked against non-matching documents only (`pair_partners`).
    Queries are processed in blocks of at most `PAIR_HITS_MAX_ENTRIES` similarities.
    """
    from sae_confounder_pipeline import pair_partners

    U, pr = _unit(E_full), np.asarray(pairs)
    out = {k: np.zeros(len(pr)) for k in ks}
    nb = pair_partners(pr)
    block = max(1, PAIR_HITS_MAX_ENTRIES // max(1, len(U)))
    for side in (0, 1):
        for lo in range(0, len(pr), block):
            q, t = pr[lo:lo + block, side], pr[lo:lo + block, 1 - side]
            rows = np.arange(len(q))
            S = U[q] @ U.T
            S[rows, q] = -np.inf
            if nb is not None:
                others = [(r, o) for r, (a, b) in enumerate(zip(q, t))
                          for o in nb[int(a)] if o != int(b)]
                if others:
                    rr, cc = np.asarray(others).T
                    S[rr, cc] = -np.inf
            rank = (S > S[rows, t][:, None]).sum(axis=1)
            for k in ks:
                out[k][lo:lo + block] += (rank < k) / 2.0
    return out


def pair_kinds(y, pairs) -> np.ndarray:
    """Each pair's class pair, e.g. ``"de|fr"`` (the two classes sorted)."""
    y = np.asarray(y)
    return np.asarray(["|".join(sorted((str(y[a]), str(y[b])))) for a, b in pairs])


def _pair_cosines(E_full, pairs) -> np.ndarray:
    U, pr = _unit(E_full), np.asarray(pairs)
    return (U[pr[:, 0]] * U[pr[:, 1]]).sum(axis=1)


def evaluate_fits(fitteds: list, X, y, *, conditions=ID_CONDITIONS, topics=None,
                  nonlinear=(), nonlinear_conditions=None, recall_k: int = 10) -> dict:
    """One seed's fits -- a single holdout split or K folds -- scored per plan Sec. 3.

    Per fit and condition, the frozen map transforms the whole corpus; the
    held-out documents get the independent linear audit (and, if asked, the
    nonlinear probes on the same folds and the k = 21 topic clustering), and
    the held-out pairs are retrieval queries against the whole transformed
    corpus. `nonlinear_conditions` restricts the nonlinear probes to a subset of
    the conditions, for runs where they would otherwise dominate the cost. Under K-fold the per-fold audit, probe and topic scores are
    averaged, and the retrieval hits (and the pair cosines behind SemEval's
    graded similarity) are concatenated, so every pair is a query once.
    """
    from scipy.stats import spearmanr

    X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
    splits = [f.split for f in fitteds]
    seed = int(splits[0].seed)
    gold = (np.concatenate([sp.gold_test for sp in splits])
            if all(sp.has_pairs and sp.gold_test is not None for sp in splits) else None)
    out = {"seed": seed, "metrics": {}, "per_fold": {}, "hits": {}, "audit_C": {},
           "split": {"scheme": "kfold" if len(splits) > 1 else "holdout",
                     "n_folds": len(splits),
                     "n_train_docs": [int(len(sp.train_docs)) for sp in splits],
                     "n_test_docs": [int(len(sp.test_docs)) for sp in splits],
                     "n_test_pairs": [int(len(sp.test_pairs or [])) for sp in splits]},
           "selected": [[int(j) for j in f.selected] for f in fitteds],
           "n_selected": [int(len(f.selected)) for f in fitteds],
           "pseudo_groups_k": [int(f.config.get("pseudo_groups_k", 0)) for f in fitteds]}
    for c in conditions:
        probes = tuple(nonlinear) if (nonlinear_conditions is None or c in nonlinear_conditions) else ()
        per = {"linear_auc": [], **{f"{p}_auc": [] for p in probes}}
        if topics is not None:
            per.update(topic_purity=[], topic_ari=[])
        per_class, chosen, folds_used, constant = [], [], [], []
        hits, cos, kinds = {1: [], recall_k: []}, [], []
        for f in fitteds:
            sp = f.split
            E_full = np.asarray(f.erasers[c](X), dtype=np.float32)
            E_te, y_te, g = E_full[sp.test_docs], y[sp.test_docs], P.probe_groups(sp)
            lin = audit.linear_audit(E_te, y_te, g, seed)
            per["linear_auc"].append(lin["auc"])
            per_class.append(lin["per_class"])
            chosen.append(lin["C"])
            folds_used.append(lin["n_folds"])
            constant.append(lin["n_constant"])
            for probe in probes:
                per[f"{probe}_auc"].append(audit.nonlinear_audit(E_te, y_te, g, seed, probe)["auc"])
            if topics is not None:
                ts = audit.topic_scores(E_te, np.asarray(topics)[sp.test_docs], seed=seed)
                per["topic_purity"].append(ts["topic_purity"])
                per["topic_ari"].append(ts["topic_ari"])
            if sp.has_pairs:
                h = pair_hits(E_full, sp.test_pairs_global, ks=(1, recall_k))
                hits[1].append(h[1])
                hits[recall_k].append(h[recall_k])
                kinds.append(pair_kinds(y, sp.test_pairs_global))
                if gold is not None:
                    cos.append(_pair_cosines(E_full, sp.test_pairs_global))
        m = {kk: float(np.mean(v)) for kk, v in per.items()}
        m["linear_auc_per_class"] = {cls: float(np.mean([pc[cls] for pc in per_class]))
                                     for cls in per_class[0]}
        m["audit_folds"] = int(min(folds_used))
        m["n_constant_coords"] = int(max(constant))
        if hits[recall_k]:
            m["recall@1"] = float(np.concatenate(hits[1]).mean())
            m[f"recall@{recall_k}"] = float(np.concatenate(hits[recall_k]).mean())
            out["hits"][c] = np.concatenate(hits[recall_k]).tolist()
            # A three-class corpus has three kinds of pair (de|fr, de|it, fr|it):
            # recall per kind, comparable with the matching two-class corpus.
            kind = np.concatenate(kinds)
            if len(np.unique(kind)) > 1:
                for kk in (1, recall_k):
                    hk = np.concatenate(hits[kk])
                    m[f"recall@{kk}_by_pair"] = {p: float(hk[kind == p].mean())
                                                 for p in np.unique(kind)}
            if gold is not None:
                m["gold_sim_spearman"] = float(spearmanr(np.concatenate(cos), gold).statistic)
        out["metrics"][c] = m
        out["per_fold"][c] = {kk: [float(x) for x in v] for kk, v in per.items()}
        out["audit_C"][c] = chosen
    return out


def aggregate(per_seed: list) -> dict:
    """mean / SD / values per condition and metric, across seeds (the unit of
    replication; folds were already averaged within each seed)."""
    agg = {}
    for c, m0 in per_seed[0]["metrics"].items():
        agg[c] = {}
        for kk, v in m0.items():
            if isinstance(v, dict):
                agg[c][kk] = {cls: _stat([s["metrics"][c][kk][cls] for s in per_seed])
                              for cls in v}
            else:
                agg[c][kk] = _stat([s["metrics"][c][kk] for s in per_seed])
    return agg


def _line(label: str, metrics: dict) -> str:
    keys = [k for k in ("linear_auc", "gb_auc", "mlp_auc", "recall@10", "topic_purity",
                        "topic_ari", "gold_sim_spearman") if k in metrics]
    return f"{label:16s} " + "  ".join(f"{k}={metrics[k]:.3f}" for k in keys)


# --------------------------------------------------------------------------- #
# Frozen maps, saved by run_id and reused by run_ood
# --------------------------------------------------------------------------- #
def _maps_path(encoder: str, corpus: str, seed: int) -> str:
    """Beside the embedding cache (git-ignored), one folder per results root so
    a QUICK run's maps never stand in for the full run's."""
    root = os.path.basename(os.path.normpath(artifacts.BASE_DIR))
    return os.path.join(E.EMBED_CACHE, "maps", root, f"{encoder}__{corpus}__seed{seed}.npz")


def save_maps(fitted, X, encoder: str, corpus: str) -> str:
    """Store one holdout fit's frozen maps as ``(A, c)`` plus the donor mean."""
    X = np.asarray(X, dtype=np.float32)
    tr = np.asarray(fitted.split.train_docs)
    arrays = {"mean": X[tr].astype(np.float64).mean(axis=0), "train_docs": tr}
    for c in OOD_TRANSFORMS:
        A, cvec = ood.affine(fitted.erasers[c], X.shape[1])
        arrays[f"{c}__A"], arrays[f"{c}__c"] = A.astype(np.float32), cvec.astype(np.float32)
    path = _maps_path(encoder, corpus, fitted.split.seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_maps(encoder: str, corpus: str, seed: int) -> tuple:
    """``(maps, donor mean, training documents)`` for one saved fit."""
    path = _maps_path(encoder, corpus, seed)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: no saved maps -- rerun run_id({corpus!r}, "
                                f"{encoder!r}, overwrite=True) with seed {seed}")
    with np.load(path) as z:
        maps = {c: (z[f"{c}__A"].astype(np.float64), z[f"{c}__c"].astype(np.float64))
                for c in OOD_TRANSFORMS}
        return maps, z["mean"].astype(np.float64), z["train_docs"].astype(int)


# --------------------------------------------------------------------------- #
# Inventories
# --------------------------------------------------------------------------- #
@scoped
def dataset_inventory(keys=ID_CORPORA, overwrite: bool = False) -> dict:
    """New Table 1 and Table A1: the loaded corpora, and CAP's topic x source counts."""
    res = _cached("dataset_inventory.json", overwrite)
    if res:
        return res
    rows = {}
    for key in keys:
        d = corpora.load(key)
        classes, counts = np.unique(d["source"], return_counts=True)
        rows[key] = {"label": d["label"], "confounder": d["config"]["confounder"],
                     "classes": [str(c) for c in classes], "class_counts": counts.tolist(),
                     "n_docs": d["n_docs"], "n_pairs": d["n_pairs"],
                     "n_unpaired": d["n_unpaired"], "paired": d["paired"],
                     "has_gold": d["config"]["has_gold"],
                     "n_topics": (int(len(np.unique(d["topics"])))
                                  if d.get("topics") is not None else None)}
    cap = corpora.load("cap")
    from build_cap_data import MAJOR_TOPICS
    topic_code = {name: code for code, name in MAJOR_TOPICS.items()}
    y, t = np.asarray(cap["source"]), np.asarray(cap["topics"])
    comp = [{"topic_id": code, "topic": name,
             **{s: int(((t == name) & (y == s)).sum()) for s in ("bill", "order", "newspaper")}}
            for name, code in sorted(topic_code.items(), key=lambda kv: kv[1])]
    res = {"experiment": "dataset_inventory", "corpora": rows, "cap_composition": comp}
    save_result(res, "dataset_inventory.json")
    return res


@scoped
def encoder_inventory(encoders=E.PARENT_PAPER_ENCODERS, overwrite: bool = False) -> dict:
    """Table A2: model id, revision, dimension, prompt and pooling per encoder.

    Call it after the runs: the revision recorded is the snapshot in this
    environment's model cache -- the one the runs downloaded and used. An
    encoder absent from that cache gets the Hub's current revision, flagged.
    """
    res = _cached("encoder_inventory.json", overwrite)
    if res:
        return res
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from style_erasure_pipeline import MODEL_MAP

    prompts = {"e5-small": "\"query: \" on every document", "e5-base": "\"query: \" on every document",
               "e5-large": "\"query: \" on every document",
               "nomic-v2": "prompt_name=\"passage\""}
    usage = {"e5-small": "primary ID + OOD; encoder study",
             "mini": "CAP/SCOTUS ID + OOD; encoder study"}
    rows = {}
    for enc in encoders:
        repo = MODEL_MAP[enc]
        try:
            revision = os.path.basename(snapshot_download(repo, local_files_only=True).rstrip("/"))
            source = "model-cache snapshot used by these runs"
        except Exception:                                   # noqa: BLE001
            try:
                revision = HfApi().model_info(repo).sha
                source = "current Hub revision (model not in this environment's cache)"
            except Exception:                               # noqa: BLE001
                revision, source = None, "unavailable offline"
        modules, pooling = [], None
        try:
            modules = [m["type"].split(".")[-1] for m in
                       json.load(open(hf_hub_download(repo, "modules.json")))]
            pool = json.load(open(hf_hub_download(repo, "1_Pooling/config.json")))
            pooling = next((k.replace("pooling_mode_", "").replace("_tokens", "")
                            for k, v in pool.items() if k.startswith("pooling_mode") and v), None)
        except Exception:                                   # noqa: BLE001
            pass
        rows[enc] = {"model": repo, "revision": revision, "revision_source": source,
                     "dim": corpora.ENCODER_DIM.get(enc),
                     "prompt": prompts.get(enc, "none"), "pooling": pooling,
                     "normalize": "Normalize" in modules, "modules": modules,
                     "coverage": usage.get(enc, "encoder study")}
    res = {"experiment": "encoder_inventory", "encoders": rows,
           "ood_target_prompts": ood.PROMPT_STYLE}
    save_result(res, "encoder_inventory.json")
    return res


# --------------------------------------------------------------------------- #
# ID: the four-condition comparison
# --------------------------------------------------------------------------- #
@scoped
def run_id(corpus: str, encoder: str = E.PRIMARY_ENCODER, seeds=None, nonlinear=(),
           save_maps_for_ood: bool = True, device: str | None = None,
           fit_device: str | None = None, overwrite: bool = False, verbose: bool = True,
           **overrides) -> dict:
    """Raw / PCA-1 / Metadata-LEACE / SAE-LEACE on one corpus, every registry seed.

    For the donors of `OOD_MATRIX` (and `MAP_DONORS`), the fits of the seeds in
    `OOD_SEEDS` save their frozen maps (`save_maps`) so `run_ood` transfers exactly these maps
    rather than refitting them (plan Sec. 5.2: the first three seeds, fixed in
    advance).
    """
    name = f"id_{corpus}_{encoder}.json"
    res = _cached(name, overwrite)
    if res:
        return res
    data = load(corpus, encoder, device=device, verbose=verbose)
    seeds = tuple(seeds or data["config"]["seeds"])
    X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
    per_seed = []
    for s in seeds:
        t0 = time.time()
        _, fitteds = fit_seed(data, s, device=fit_device, **overrides)
        r = evaluate_fits(fitteds, X, y, conditions=ID_CONDITIONS,
                          topics=data.get("topics"), nonlinear=nonlinear)
        if (save_maps_for_ood and len(fitteds) == 1
                and ((encoder, corpus) in OOD_MATRIX or (encoder, corpus) in MAP_DONORS)
                and s in OOD_SEEDS):
            r["maps_file"] = save_maps(fitteds[0], X, encoder, corpus)
        r["seconds"] = round(time.time() - t0, 1)
        per_seed.append(r)
        if verbose:
            print(f"[id] {corpus}/{encoder} seed {s} ({r['seconds']}s)")
            for c in ID_CONDITIONS:
                print("     " + _line(LABELS[c], r["metrics"][c]))
    res = {"experiment": "revised_id", "corpus": corpus, "label": data["label"],
           "encoder": encoder, "seeds": list(seeds), "conditions": list(ID_CONDITIONS),
           "labels": LABELS, "per_seed": per_seed, "agg": aggregate(per_seed),
           "corpus_stats": _corpus_stats(data),
           "config": _config(data, overrides, nonlinear=list(nonlinear))}
    save_result(res, name)
    return res


@scoped
def extend_id_nonlinear(corpus: str, encoder: str = E.PRIMARY_ENCODER, probes=("gb", "mlp"),
                        seeds=None, device=None, fit_device=None, overwrite: bool = False,
                        verbose: bool = True, **overrides) -> dict:
    """Add nonlinear probes to a cached `run_id` result, on the fits it was scored on.

    A seed whose frozen maps were saved (the OOD donors' seeds 0-2) is re-scored from
    those maps, so the probes see exactly the fitted transformations; any other seed is
    refitted with the given recipe. Either way its linear AUC must match the cached
    value before the probes are merged. Only the probe metrics are added: the cached
    linear, retrieval and topic scores are left as they are.
    """
    name = f"id_{corpus}_{encoder}.json"
    path = artifact_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run run_id({corpus!r}, {encoder!r}) first")
    res = load_result(path)
    keys, conds = [f"{p}_auc" for p in probes], list(res["conditions"])
    todo = [p for p in res["per_seed"] if (seeds is None or p["seed"] in seeds)
            and (overwrite or any(k not in p["metrics"][c] for c in conds for k in keys))]
    if not todo:
        print(f"[cached] {path} already has {keys}")
        return res
    data = load(corpus, encoder, device=device, verbose=False)
    X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
    checks = []
    for p in todo:
        s = int(p["seed"])
        splits, fits, source = _splits(data, s), None, "refit"
        if len(splits) == 1 and os.path.exists(_maps_path(encoder, corpus, s)):
            maps, _, tr = load_maps(encoder, corpus, s)
            if not np.array_equal(np.sort(tr), np.sort(np.asarray(splits[0].train_docs))):
                raise ValueError(f"{corpus} seed {s}: the saved maps were fitted on other "
                                 "training documents than this split")
            er = {"raw": lambda Z: np.asarray(Z, np.float32)}
            er.update({c: (lambda Z, _A=A, _c=cv: (np.asarray(Z, np.float64) @ _A + _c)
                           .astype(np.float32)) for c, (A, cv) in maps.items()})
            fits, source = [Erasers(splits[0], er)], "saved maps"
        if fits is None:
            _, fits = fit_seed(data, s, splits=splits, device=fit_device, **overrides)
        r = evaluate_fits(fits, X, y, conditions=tuple(conds), nonlinear=probes)
        diff = max(abs(r["metrics"][c]["linear_auc"] - p["metrics"][c]["linear_auc"]) for c in conds)
        if diff > 0.005:
            raise RuntimeError(
                f"{corpus} seed {s}: the {source} fit does not reproduce the cached linear AUC "
                f"(largest difference {diff:.4f}), so nothing was written. Rerun the corpus as a "
                f"whole: R.run_id({corpus!r}, {encoder!r}, nonlinear={tuple(probes)}, overwrite=True)")
        checks.append({"seed": s, "source": source, "max_linear_auc_diff": float(diff)})
        for c in conds:
            for k in keys:
                p["metrics"][c][k] = r["metrics"][c][k]
                p["per_fold"][c][k] = r["per_fold"][c][k]
        if verbose:
            print(f"[id+] {corpus}/{encoder} seed {s} ({source}, linear AUC within {diff:.4f}): "
                  + " | ".join(f"{LABELS[c]} " + " ".join(f"{k} {p['metrics'][c][k]:.3f}" for k in keys)
                               for c in conds))
    res["agg"] = aggregate(res["per_seed"])
    res["config"]["nonlinear"] = sorted(set(res["config"].get("nonlinear") or []) | set(probes))
    res.setdefault("extended", []).append(
        {"added": keys, "fits": checks,
         "note": "probes added on the cached fits' held-out documents and audit folds"})
    save_result(res, name)
    return res


@scoped
def run_id_kmeans(corpus: str, encoder: str = E.PRIMARY_ENCODER, seeds=None,
                  nonlinear=("gb", "mlp"), device: str | None = None,
                  overwrite: bool = False, verbose: bool = True) -> dict:
    """k-means LEACE on the splits of a cached `run_id` result: New Table 2's k-means row.

    Per split, K is chosen by the training silhouette -- the rule SAE-LEACE uses for
    its pseudo-groups -- and LEACE erases the one-hot k-means grouping
    (`kmeans_eraser`); no SAE is trained and no metadata is used. Every seed of the
    `run_id` result is scored by `evaluate_fits` with the given probes, and each
    seed's split must be the cached one. Where the erasure-alternatives run scored the
    same arm on the same split (seeds 0-2), its linear AUC must be reproduced before
    anything is written. Saved beside the `run_id` result, which is left untouched,
    as ``id_kmeans_<corpus>_<encoder>.json``.
    """
    name = f"id_kmeans_{corpus}_{encoder}.json"
    res = _cached(name, overwrite)
    if res:
        return res
    ref_path = artifact_path(f"id_{corpus}_{encoder}.json")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"{ref_path}: run run_id({corpus!r}, {encoder!r}) first")
    ref = {p["seed"]: p for p in load_result(ref_path)["per_seed"]}
    alt_path = raw_path("alternatives", f"alternatives_{corpus}_{encoder}.json")
    alt = ({p["seed"]: p for p in load_result(alt_path)["per_seed"]}
           if os.path.exists(alt_path) else {})
    seeds = tuple(seeds or sorted(ref))
    if set(seeds) - set(ref):
        raise ValueError(f"seeds {sorted(set(seeds) - set(ref))} are not in the cached run_id result")
    data = load(corpus, encoder, device=device, verbose=verbose)
    X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
    per_seed, checks = [], []
    for s in seeds:
        t0 = time.time()
        fits, diag = [], []
        for sp in _splits(data, s):
            tr = np.asarray(sp.train_docs)
            K, sil = P.choose_pseudo_groups(X[tr], seed=sp.seed)
            fits.append(Erasers(sp, {"kmeans_leace": kmeans_eraser(X[tr], K, sp.seed)[0]}))
            diag.append({"k_auto": int(K), "silhouette": {str(k): v for k, v in sil.items()}})
        r = evaluate_fits(fits, X, y, conditions=("kmeans_leace",), topics=data.get("topics"),
                          nonlinear=nonlinear)
        if r["split"] != ref[s]["split"]:
            raise RuntimeError(f"{corpus} seed {s}: the split differs from the cached run_id "
                               "result's, so nothing was written")
        auc, check = r["metrics"]["kmeans_leace"]["linear_auc"], {"seed": int(s)}
        if "kmeans_leace" in alt.get(s, {}).get("metrics", {}):
            other = alt[s]["metrics"]["kmeans_leace"]["linear_auc"]
            if abs(auc - other) > 0.005:
                raise RuntimeError(f"{corpus} seed {s}: linear AUC {auc:.4f} does not reproduce "
                                   f"the alternatives run's k-means LEACE ({other:.4f}), so "
                                   "nothing was written")
            check["alternatives_linear_auc_diff"] = float(abs(auc - other))
        checks.append(check)
        r["diagnostics"] = diag
        r["seconds"] = round(time.time() - t0, 1)
        per_seed.append(r)
        if verbose:
            print(f"[id-kmeans] {corpus}/{encoder} seed {s} ({r['seconds']}s, silhouette K "
                  f"{[d['k_auto'] for d in diag]})")
            print("     " + _line(LABELS["kmeans_leace"], r["metrics"]["kmeans_leace"]))
    res = {"experiment": "revised_id_kmeans", "corpus": corpus, "label": data["label"],
           "encoder": encoder, "seeds": list(seeds), "conditions": ["kmeans_leace"],
           "labels": {"kmeans_leace": LABELS["kmeans_leace"]}, "per_seed": per_seed,
           "agg": aggregate(per_seed), "checks": checks, "corpus_stats": _corpus_stats(data),
           "config": _config(data, {}, nonlinear=list(nonlinear), sae=None)}
    save_result(res, name)
    return res


# --------------------------------------------------------------------------- #
# OOD: the ID maps, frozen, on the MTEB targets
# --------------------------------------------------------------------------- #
@scoped
def run_ood(encoder: str, donor: str, seeds=OOD_SEEDS, device: str | None = None,
            overwrite: bool = False, verbose: bool = True) -> dict:
    """Apply the maps `run_id` fitted on the donor's ID training split, unchanged,
    to every target (plan Sec. 5.2-5.3). Returns an `ood.summarize`-ready result."""
    name = f"ood_{encoder}_{donor}.json"
    res = _cached(name, overwrite)
    if res:
        return res
    data = load(donor, encoder, device=device, verbose=False)
    targets = {t: ood.load_target(t) for t in ood.TARGETS}
    embs = ood.embed_targets(encoder, device=device, verbose=verbose)
    overlap = ood.donor_target_overlap(data["texts"], targets)
    if any(overlap.values()):
        raise ValueError(f"{donor} shares texts with the targets {overlap}")
    raw = {t: ood.score_target(targets[t], embs[t]) for t in targets}
    X = np.asarray(data["emb"], np.float32)
    per_seed = []
    for s in seeds:
        maps, mu, tr = load_maps(encoder, donor, s)
        checks = {c: ood.check_operator(maps[c], mu) for c in OOD_TRANSFORMS}
        scores = {c: {t: ood.score_target(targets[t], embs[t], maps[c]) for t in targets}
                  for c in OOD_TRANSFORMS}
        per_seed.append({"seed": int(s), "maps_file": _maps_path(encoder, donor, s),
                         "fit": {"fit_on": "train", "split_id": ood._split_id(tr),
                                 "n_train_docs": int(len(tr))},
                         "operator_checks": checks, "scores": scores})
        if verbose:
            print(f"[ood] {encoder}/{donor} seed {s}: " + " | ".join(
                f"{LABELS[c]} " + " ".join(f"{scores[c][t]['score']:.3f}" for t in targets)
                for c in OOD_TRANSFORMS))
    res = {"experiment": "ood", "encoder": encoder, "donor": donor,
           "donor_label": data["label"], "donor_role": None, "seeds": [int(s) for s in seeds],
           "fit_on": "train", "transforms": list(OOD_TRANSFORMS),
           "labels": {c: LABELS[c] for c in ("raw",) + OOD_TRANSFORMS},
           "tasks": ood.task_records(targets), "raw": raw, "per_seed": per_seed,
           "protocol": {"mteb_version": ood.MTEB_VERSION,
                        "encoder_model": ood.encoder_revision(encoder),
                        "prompt_style": ood.PROMPT_STYLE,
                        "target_prompts": ood._prompts(encoder),
                        "maps": "the ID maps of run_id, fit on the donor's ID training "
                                "split and frozen", "donor_target_overlap": overlap,
                        "normalization": "maps applied without renormalization; cosine"}}
    save_result(res, name)
    return res


# --------------------------------------------------------------------------- #
# Sweeps (Appendix D and New Figure 2 / Table 3)
# --------------------------------------------------------------------------- #
@scoped
def run_budget(corpus: str, encoder: str = E.PRIMARY_ENCODER, grid=BUDGET_GRID,
               seeds=SWEEP_SEEDS, device=None, overwrite: bool = False,
               verbose: bool = True, **overrides) -> dict:
    """SAE direction-LEACE at nested top-N prefixes of one MI ranking per split.

    One SAE per split serves every N. Raw, PCA-1 and Metadata-LEACE are scored
    on the same splits as N-independent references.
    """
    name = f"budget_{corpus}_{encoder}.json"
    res = _cached(name, overwrite)
    if res:
        return res
    data = load(corpus, encoder, device=device, verbose=False)
    X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
    kw = _kw(overrides)
    per_seed = []
    for s in seeds:
        splits = _splits(data, s)
        bases = _bases(data, splits, device=device, **overrides)
        refs = evaluate_fits([_refit(data, sp, b, top_features=kw["top_features"], pca=True)
                              for sp, b in zip(splits, bases)], X, y,
                             conditions=("raw", "pca_top1", "metadata_leace"),
                             topics=data.get("topics"))
        points = {}
        for m in grid:
            r = evaluate_fits([_refit(data, sp, b, top_features=m, erase=kw["erase"])
                               for sp, b in zip(splits, bases)], X, y,
                              conditions=("sae_leace",), topics=data.get("topics"))
            points[str(m)] = r["metrics"]["sae_leace"]
        per_seed.append({"seed": int(s), "points": points,
                         "references": refs["metrics"]})
        if verbose:
            print(f"[budget] {corpus} seed {s}: " + ", ".join(
                f"N={m} {points[str(m)]['linear_auc']:.3f}" for m in grid))
    curve = {str(m): {kk: _stat([p["points"][str(m)][kk] for p in per_seed])
                      for kk in per_seed[0]["points"][str(m)]
                      if not isinstance(per_seed[0]["points"][str(m)][kk], dict)}
             for m in grid}
    references = {c: {kk: _stat([p["references"][c][kk] for p in per_seed])
                      for kk, v in per_seed[0]["references"][c].items()
                      if not isinstance(v, dict)}
                  for c in per_seed[0]["references"]}
    res = {"experiment": "revised_budget", "corpus": corpus, "label": data["label"],
           "encoder": encoder, "seeds": list(seeds), "grid": list(grid),
           "paired": data["paired"], "per_seed": per_seed, "curve": curve,
           "references": references, "default_n": kw["top_features"],
           "config": _config(data, overrides)}
    save_result(res, name)
    return res


@scoped
def run_mechanism(corpus: str, encoder: str = E.PRIMARY_ENCODER, mechanisms=MECHANISMS,
                  seeds=SWEEP_SEEDS, nonlinear=(), device=None, overwrite: bool = False,
                  verbose: bool = True, **overrides) -> dict:
    """The same SAE, ranking and selected set (top-N), three interventions.

    `nonlinear` adds the GB/MLP probes on the audit folds of the linear probe.
    """
    name = f"mechanism_{corpus}_{encoder}.json"
    res = _cached(name, overwrite)
    if res:
        return res
    data = load(corpus, encoder, device=device, verbose=False)
    X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
    kw = _kw(overrides)
    per_seed = []
    for s in seeds:
        splits = _splits(data, s)
        bases = _bases(data, splits, device=device, **overrides)
        row = {}
        for mech in mechanisms:
            fits = [_refit(data, sp, b, top_features=kw["top_features"], erase=mech)
                    for sp, b in zip(splits, bases)]
            r = evaluate_fits(fits, X, y, conditions=("sae_leace",), topics=data.get("topics"),
                              nonlinear=nonlinear)
            row[mech] = {**r["metrics"]["sae_leace"], "selected": r["selected"]}
        same = all(row[m]["selected"] == row[mechanisms[0]]["selected"] for m in mechanisms)
        per_seed.append({"seed": int(s), "mechanisms": row, "same_selected_set": same})
        if verbose:
            print(f"[mechanism] {corpus} seed {s}: " + " | ".join(
                f"{m} {row[m]['linear_auc']:.3f}" for m in mechanisms))
    agg = {m: {kk: _stat([p["mechanisms"][m][kk] for p in per_seed])
               for kk, v in per_seed[0]["mechanisms"][m].items()
               if isinstance(v, float)} for m in mechanisms}
    res = {"experiment": "revised_mechanism", "corpus": corpus, "label": data["label"],
           "encoder": encoder, "seeds": list(seeds), "mechanisms": list(mechanisms),
           "paired": data["paired"], "per_seed": per_seed, "agg": agg,
           "config": _config(data, overrides, nonlinear=list(nonlinear))}
    save_result(res, name)
    return res


def _mechanism_agg(per_seed: list, mechanisms) -> dict:
    return {m: {kk: _stat([p["mechanisms"][m][kk] for p in per_seed])
                for kk, v in per_seed[0]["mechanisms"][m].items() if isinstance(v, float)}
            for m in mechanisms}


@scoped
def extend_mechanism_nonlinear(corpus: str, encoder: str = E.PRIMARY_ENCODER,
                               probes=("gb", "mlp"), device=None, overwrite: bool = False,
                               verbose: bool = True, **overrides) -> dict:
    """Add nonlinear probes to a cached `run_mechanism` result.

    The interventions are not stored, so each seed's SAEs are refitted with the given
    recipe. The refit must select exactly the latent sets the cached run selected and
    reproduce its linear AUC before the probes are merged; otherwise nothing is written
    and the experiment has to be rerun as a whole. Only the probe metrics are added.
    """
    name = f"mechanism_{corpus}_{encoder}.json"
    path = artifact_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run run_mechanism({corpus!r}, {encoder!r}) first")
    res = load_result(path)
    keys, mechs = [f"{p}_auc" for p in probes], list(res["mechanisms"])
    if not overwrite and all(k in p["mechanisms"][m] for p in res["per_seed"]
                             for m in mechs for k in keys):
        print(f"[cached] {path} already has {keys}")
        return res
    data = load(corpus, encoder, device=device, verbose=False)
    X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
    kw = _kw(overrides)
    checks = []
    for p in res["per_seed"]:
        s = int(p["seed"])
        splits = _splits(data, s)
        bases = _bases(data, splits, device=device, **overrides)
        for m in mechs:
            fits = [_refit(data, sp, b, top_features=kw["top_features"], erase=m)
                    for sp, b in zip(splits, bases)]
            r = evaluate_fits(fits, X, y, conditions=("sae_leace",), nonlinear=probes)
            new, old = r["metrics"]["sae_leace"], p["mechanisms"][m]
            same = r["selected"] == old["selected"]
            diff = abs(new["linear_auc"] - old["linear_auc"])
            if not same or diff > 0.005:
                raise RuntimeError(
                    f"{corpus} seed {s} {m}: the refit does not reproduce the cached fit (same "
                    f"selected sets: {same}; linear AUC difference {diff:.4f}), so nothing was "
                    f"written. Rerun the experiment as a whole: R.run_mechanism({corpus!r}, "
                    f"{encoder!r}, nonlinear={tuple(probes)}, overwrite=True)")
            checks.append({"seed": s, "mechanism": m, "same_selected_sets": same,
                           "linear_auc_diff": float(diff)})
            for k in keys:
                old[k] = new[k]
        if verbose:
            print(f"[mechanism+] {corpus} seed {s}: " + " | ".join(
                f"{m} " + " ".join(f"{k} {p['mechanisms'][m][k]:.3f}" for k in keys) for m in mechs))
    res["agg"] = _mechanism_agg(res["per_seed"], mechs)
    res["config"]["nonlinear"] = sorted(set(res["config"].get("nonlinear") or []) | set(probes))
    res.setdefault("extended", []).append(
        {"added": keys, "fits": checks,
         "note": "SAEs refitted; identical selected sets and linear AUC verified before merging"})
    save_result(res, name)
    return res


@scoped
def run_pseudo_groups(corpus: str, encoder: str = E.PRIMARY_ENCODER, ks=K_GRID,
                      seeds=SWEEP_SEEDS, device=None, overwrite: bool = False,
                      verbose: bool = True, **overrides) -> dict:
    """SAE-LEACE at every candidate K, with the training silhouette per K."""
    name = f"pseudo_groups_{corpus}_{encoder}.json"
    res = _cached(name, overwrite)
    if res:
        return res
    data = load(corpus, encoder, device=device, verbose=False)
    X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
    kw = _kw(overrides)
    per_seed = []
    for s in seeds:
        splits = _splits(data, s)
        chosen = [P.choose_pseudo_groups(X[sp.train_docs], ks=ks, seed=s) for sp in splits]
        bases0 = _bases(data, splits, pseudo_groups=int(ks[0]), device=device, **overrides)
        meta = evaluate_fits([_refit(data, sp, b, top_features=kw["top_features"])
                              for sp, b in zip(splits, bases0)], X, y,
                             conditions=("metadata_leace",))["metrics"]["metadata_leace"]
        points = {}
        for K in ks:
            fits = [_refit(data, sp, P.rescore_base(b, X, int(K)),
                           top_features=kw["top_features"], erase=kw["erase"],
                           pseudo_groups=int(K)) for sp, b in zip(splits, bases0)]
            points[str(K)] = evaluate_fits(fits, X, y, conditions=("sae_leace",))["metrics"]["sae_leace"]
        per_seed.append({"seed": int(s), "points": points, "metadata_reference": meta,
                         "selected_k": [int(c[0]) for c in chosen],
                         "silhouette": {str(K): float(np.mean([c[1][int(K)] for c in chosen]))
                                        for K in ks}})
        if verbose:
            print(f"[K] {corpus} seed {s}: selected K {per_seed[-1]['selected_k']}")
    res = {"experiment": "revised_pseudo_groups", "corpus": corpus, "label": data["label"],
           "encoder": encoder, "seeds": list(seeds), "ks": [int(K) for K in ks],
           "paired": data["paired"], "n_classes": int(len(np.unique(y))),
           "per_seed": per_seed, "config": _config(data, overrides)}
    save_result(res, name)
    return res


@scoped
def run_score_function(corpus: str, encoder: str = E.PRIMARY_ENCODER,
                       variants=SCORE_VARIANTS, seeds=SWEEP_SEEDS, device=None,
                       overwrite: bool = False, verbose: bool = True, **overrides) -> dict:
    """MI vs normalized MI vs correlation ratio on the same SAE and grouping."""
    from sae_confounder_pipeline import score_confounder_features_variants

    name = f"score_function_{corpus}_{encoder}.json"
    res = _cached(name, overwrite)
    if res:
        return res
    data = load(corpus, encoder, device=device, verbose=False)
    X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
    kw = _kw(overrides)
    per_seed = []
    for s in seeds:
        splits = _splits(data, s)
        bases = _bases(data, splits, device=device, **overrides)
        scored = [score_confounder_features_variants(
            b.feats_train, X[sp.train_docs], n_pseudo_groups=int(b.pseudo_groups),
            seed=sp.seed, pseudo_group=b.pseudo_group_train) for sp, b in zip(splits, bases)]
        row = {}
        for v in variants:
            fits = [_refit(data, sp, _replace(b, scores=np.asarray(sc[v])),
                           top_features=kw["top_features"], erase=kw["erase"])
                    for sp, b, sc in zip(splits, bases, scored)]
            r = evaluate_fits(fits, X, y, conditions=("sae_leace",), nonlinear=("gb",))
            row[v] = {**r["metrics"]["sae_leace"], "selected": r["selected"]}
        per_seed.append({"seed": int(s), "variants": row})
        if verbose:
            print(f"[score] {corpus} seed {s}: " + " | ".join(
                f"{v} {row[v]['linear_auc']:.3f}" for v in variants))

    def jac(a, b):
        return len(set(a) & set(b)) / max(1, len(set(a) | set(b)))

    jaccard = {f"{a}|{b}": float(np.mean([np.mean([jac(fa, fb) for fa, fb in
                                                    zip(p["variants"][a]["selected"],
                                                        p["variants"][b]["selected"])])
                                           for p in per_seed]))
               for i, a in enumerate(variants) for b in variants[i + 1:]}
    agg = {v: {kk: _stat([p["variants"][v][kk] for p in per_seed])
               for kk, x in per_seed[0]["variants"][v].items() if isinstance(x, float)}
           for v in variants}
    res = {"experiment": "revised_score_function", "corpus": corpus, "label": data["label"],
           "encoder": encoder, "seeds": list(seeds), "variants": list(variants),
           "paired": data["paired"], "per_seed": per_seed, "agg": agg,
           "jaccard": jaccard, "top_n": kw["top_features"],
           "config": _config(data, overrides)}
    save_result(res, name)
    return res


def _subsample(data: dict, size: int) -> dict:
    """The fixed subsample of `experiments.corpus_size_sweep` (seeded by the size)."""
    rng = np.random.default_rng(size)
    y = np.asarray(data["source"])
    if data["paired"]:
        arr = np.asarray(data["pairs"])
        sub = rng.permutation(len(arr))[:size]
        unpaired = np.setdiff1d(np.arange(data["n_docs"]), np.unique(arr))
        frac = size / len(arr)
        take_u = [rng.permutation(unpaired[y[unpaired] == c])[: int(round(frac * (y[unpaired] == c).sum()))]
                  for c in (np.unique(y[unpaired]) if len(unpaired) else [])]
        idx = np.concatenate([arr[sub, 0], arr[sub, 1]] + take_u)
        pairs = [(j, size + j) for j in range(size)]
    else:
        per = size // len(np.unique(y))
        idx = np.concatenate([rng.permutation(np.where(y == c)[0])[:per] for c in np.unique(y)])
        pairs = None
    return {**data, "emb": np.asarray(data["emb"])[idx], "source": y[idx],
            "texts": [data["texts"][i] for i in idx], "pairs": pairs, "gold": None,
            "topics": None, "n_docs": int(len(idx)), "n_pairs": int(size) if pairs else 0}


@scoped
def run_corpus_size(corpus: str, encoder: str = E.PRIMARY_ENCODER, sizes=None,
                    seeds=SWEEP_SEEDS, device=None, overwrite: bool = False,
                    verbose: bool = True, **overrides) -> dict:
    """Raw / Metadata-LEACE / SAE-LEACE and all three probes at each corpus size."""
    name = f"corpus_size_{corpus}_{encoder}.json"
    res = _cached(name, overwrite)
    if res:
        return res
    data = load(corpus, encoder, device=device, verbose=False)
    unit = "pairs" if data["paired"] else "documents"
    cap = data["n_pairs"] if data["paired"] else data["n_docs"]
    sizes = [int(s) for s in (sizes or SIZE_GRID[corpus])]
    variants = {}
    for size in sizes:
        sub = _subsample(data, min(size, cap))
        X, y = np.asarray(sub["emb"], np.float32), np.asarray(sub["source"])
        per_seed = []
        for s in seeds:
            _, fitteds = fit_seed(sub, s, pca=False, device=device, **overrides)
            per_seed.append(evaluate_fits(fitteds, X, y,
                                          conditions=("raw", "metadata_leace", "sae_leace"),
                                          nonlinear=("mlp", "gb")))
        variants[str(size)] = {"nominal": size, "unit": unit, "n_docs": sub["n_docs"],
                               "per_seed": per_seed, "agg": aggregate(per_seed)}
        if verbose:
            a = variants[str(size)]["agg"]
            print(f"[size] {corpus} {size} {unit} ({sub['n_docs']} docs): " + " | ".join(
                f"{c} lin {a[c]['linear_auc']['mean']:.3f} gb {a[c]['gb_auc']['mean']:.3f}"
                for c in ("raw", "metadata_leace", "sae_leace")))
    res = {"experiment": "revised_corpus_size", "corpus": corpus, "label": data["label"],
           "encoder": encoder, "seeds": list(seeds), "sizes": sizes, "unit": unit,
           "variants": variants, "config": _config(data, overrides)}
    save_result(res, name)
    return res


@scoped
def run_protocol_diagnostic(corpus: str, encoder: str = E.PRIMARY_ENCODER,
                            seeds=SWEEP_SEEDS, device=None, overwrite: bool = False,
                            verbose: bool = True, **overrides) -> dict:
    """Legacy fitting-protocol diagnostic: one 50/50 split per seed, the eraser
    fit on all documents ("pair" holdout, links withheld) versus on the training
    documents only ("document"); the same audit scores both."""
    name = f"protocol_{corpus}_{encoder}.json"
    res = _cached(name, overwrite)
    if res:
        return res
    data = load(corpus, encoder, device=device, verbose=False)
    X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
    arms = {}
    for proto in ("pair", "document"):
        per_seed = []
        for s in seeds:
            sp = P.make_split(data["n_docs"], y, pairs=data["pairs"], gold_sim=data["gold"],
                              holdout=0.5, seed=s, protocol=proto)
            _, fitteds = fit_seed(data, s, pca=False, splits=[sp], device=device, **overrides)
            per_seed.append(evaluate_fits(fitteds, X, y,
                                          conditions=("raw", "metadata_leace", "sae_leace")))
        arms[proto] = {"per_seed": per_seed, "agg": aggregate(per_seed)}
        if verbose:
            a = arms[proto]["agg"]
            print(f"[protocol] {corpus} {proto}: " + " | ".join(
                f"{c} {a[c]['linear_auc']['mean']:.3f}" for c in ("raw", "metadata_leace", "sae_leace")))
    res = {"experiment": "revised_protocol", "corpus": corpus, "label": data["label"],
           "encoder": encoder, "seeds": list(seeds), "arms": arms,
           "note": "both arms use one 50/50 split per seed; 'pair' fits on every "
                   "document (links withheld), 'document' on the training half only",
           "config": _config(data, overrides)}
    save_result(res, name)
    return res


# --------------------------------------------------------------------------- #
# Descriptive fits (Appendix D.5-D.6)
# --------------------------------------------------------------------------- #
def _excerpt(text: str, words: int = 25, cjk_chars: int = 60) -> str:
    """The first `words` words, or `cjk_chars` characters of an unspaced script."""
    import re
    t = " ".join(str(text).split())
    if len(re.findall(r"[぀-ヿ一-鿿가-힯]", t)) > 0.3 * max(len(t), 1):
        return t[:cjk_chars] + (" ..." if len(t) > cjk_chars else "")
    w = t.split(" ")
    return " ".join(w[:words]) + (" ..." if len(w) > words else "")


@scoped
def run_descriptive(corpus: str, encoder: str = E.PRIMARY_ENCODER, seed: int = 0,
                    device=None, overwrite: bool = False, **overrides) -> dict:
    """The seed-0 full-corpus fit behind Tables D6-D8 and Figures D3-D5.

    Descriptive only: one fit on every document, no holdout, no performance
    number. Saves every selected latent's label-free and post-hoc quantities,
    the subspace geometry of the two same-atom interventions, the 2-d PCA
    panels, and the top-MI latent's highest/lowest documents.
    """
    import analysis as A

    name = f"descriptive_{corpus}_{encoder}.json"
    res = _cached(name, overwrite)
    if res:
        return res
    data = load(corpus, encoder, device=device, verbose=False)
    bundle = A.fit_descriptive(corpus, encoder, seed=seed, data=data, device=device,
                               verbose=False, **overrides)
    align = A.latent_alignment(bundle, top_m=len(bundle["selected"]), save=False)
    geometry = A.subspace_geometry(bundle, save=False)
    pca = A.pca_projection(bundle, save=False)
    top = int(max(bundle["selected"], key=lambda j: bundle["scores"][int(j)]))
    pre = np.asarray(bundle["pre"][:, top], dtype=float)
    ids = np.arange(len(pre))                   # ties go to the smaller document index
    ends = {"highest": int(np.lexsort((ids, -pre))[0]), "lowest": int(np.lexsort((ids, pre))[0])}
    examples = [{"end": end, "doc_index": i, "class": str(bundle["source"][i]),
                 "pre_activation": float(pre[i]), "excerpt": _excerpt(bundle["texts"][i])}
                for end, i in ends.items()]
    res = {"experiment": "revised_descriptive", "corpus": corpus, "label": data["label"],
           "encoder": encoder, "seed": seed, "latents": align, "geometry": geometry,
           "pca": pca, "top_latent": top, "examples": examples,
           "config": bundle["config"], "sae_info": bundle["sae_info"]}
    save_result(res, name)
    save_text(A.latent_reports(bundle, save=False), f"latent_report_{corpus}_{encoder}.md")
    return res


# --------------------------------------------------------------------------- #
# The ten-encoder study (Table B5, Figure D6)
# --------------------------------------------------------------------------- #
def _pc_shares(X) -> dict:
    """Variance shares of the leading principal components of the raw embeddings."""
    Xc = np.asarray(X, np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    var = np.linalg.svd(Xc, compute_uv=False) ** 2
    return {"pc1_var_share": float(var[0] / var.sum()),
            "top5_var_share": float(var[:5].sum() / var.sum())}


def _extend_pca(res: dict, name: str, corpus: str, encoder: str, missing: list,
                device=None, verbose: bool = True) -> dict:
    """Add PCA removal arms to a cached encoder cell, reusing its splits.

    PCA needs no SAE, and the splits and audit folds follow from the seed, so the
    new arms are measured on exactly the documents and folds the cached ones were.
    Cheaper and less wasteful than recomputing the cell.
    """
    data = load(corpus, encoder, device=device, verbose=False)
    X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
    ranks = [int(c.replace("pca_top", "")) for c in missing]
    for i, seed in enumerate(res["seeds"]):
        fits = []
        for sp in _splits(data, seed):
            mu = X[sp.train_docs].mean(axis=0, keepdims=True)
            Vt = np.linalg.svd(np.asarray(X[sp.train_docs], np.float32) - mu,
                               full_matrices=False)[2]

            def make(r, _mu=mu, _Vt=Vt):
                B = _Vt[:int(min(r, X.shape[1] - 1))]
                return lambda Z: (np.asarray(Z, np.float32) - _mu
                                  - ((np.asarray(Z, np.float32) - _mu) @ B.T) @ B
                                  + _mu).astype(np.float32)
            fits.append(Erasers(sp, {f"pca_top{r}": make(r) for r in ranks}))
        add = evaluate_fits(fits, X, y, conditions=tuple(missing),
                            topics=data.get("topics"))
        for key in ("metrics", "per_fold", "audit_C"):
            res["per_seed"][i].setdefault(key, {}).update(add[key])
    res["conditions"] = list(res["conditions"]) + list(missing)
    res["pca_ranks"] = sorted(set(res.get("pca_ranks", [])) | set(ranks))
    res["agg"] = aggregate(res["per_seed"])
    res.setdefault("extended", []).append({"added": list(missing),
                                           "note": "PCA arms fitted on the cached "
                                                   "cell's splits; no SAE refit"})
    save_result(res, name)
    if verbose:
        print(f"[encoders] {corpus}/{encoder}: added " + ", ".join(
            f"{c} {res['agg'][c]['linear_auc']['mean']:.3f}" for c in missing))
    return res


@scoped
def run_encoder_cell(corpus: str, encoder: str, seeds=ENCODER_SEEDS,
                     pca_ranks=ENCODER_PCA_RANKS, device=None, overwrite: bool = False,
                     verbose: bool = True, **overrides) -> dict:
    """One corpus x encoder cell of the ten-encoder study, run from scratch.

    Raw, Metadata-LEACE and SAE-LEACE (plan B.5's three conditions) plus PCA
    removal at each rank in `pca_ranks`, under the registry's split policy and
    the revised audit, with the raw embeddings' PC1 variance share for Figure D6.
    The PCA arms cost one SVD per fit and let the breadth table speak to the
    baseline comparison as well; pass ``pca_ranks=()`` for the plan's three alone.
    Per-pair retrieval hits are not kept: the study reports leakage only.
    """
    name = f"encoder_{corpus}_{encoder}.json"
    conditions = ENCODER_CONDITIONS + tuple(f"pca_top{int(r)}" for r in pca_ranks)
    res = _cached(name, overwrite)
    if res:
        missing = [c for c in conditions if c not in res["conditions"]]
        if not missing:
            return res
        if any(not c.startswith("pca_top") for c in missing):
            print(f"[encoders] {name} lacks {missing}, which need the SAE; "
                  "rerun with overwrite=True to recompute the cell")
            return res
        return _extend_pca(res, name, corpus, encoder, missing, device, verbose)
    data = load(corpus, encoder, device=device, verbose=verbose)
    X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
    kw = _kw(overrides)
    per_seed = []
    for s in seeds:
        t0 = time.time()
        fitteds = [P.fit_pipeline(X, y, sp, **kw, selection=data["config"]["selection"],
                                  pseudo_groups=data["config"]["pseudo_groups"],
                                  pca_grid=tuple(int(r) for r in pca_ranks), kmeans_grid=(),
                                  kmeans_auto=False, random_control=False, device=device)
                   for sp in _splits(data, s)]
        r = evaluate_fits(fitteds, X, y, conditions=conditions, topics=data.get("topics"))
        r.pop("hits", None)
        r["seconds"] = round(time.time() - t0, 1)
        per_seed.append(r)
    res = {"experiment": "revised_encoder_cell", "corpus": corpus, "label": data["label"],
           "encoder": encoder, "dim": int(X.shape[1]), "seeds": [int(s) for s in seeds],
           "conditions": list(conditions), "pca_ranks": [int(r) for r in pca_ranks],
           "per_seed": per_seed,
           "agg": aggregate(per_seed), "diagnostics": _pc_shares(X),
           "corpus_stats": _corpus_stats(data), "config": _config(data, overrides)}
    if verbose:
        a = res["agg"]
        print(f"[encoders] {corpus}/{encoder}: " + " | ".join(
            f"{LABELS.get(c) or 'PCA-' + c.replace('pca_top', '')} "
            f"{a[c]['linear_auc']['mean']:.3f}" for c in conditions)
            + f" | PC1 share {res['diagnostics']['pc1_var_share']:.3f}")
    save_result(res, name)
    return res
