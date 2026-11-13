"""Frozen cross-corpus transfer: does a fitted eraser preserve other corpora?

The parent paper (Fan et al., 2025, Sec. 5) fits an eraser on one corpus and
applies it, unchanged, to MTEB tasks encoded by the same model. This module
replicates that design as closely as its description allows and adds
SAE-LEACE: each eraser is learned on one of the parent's two donor
collections -- CAP Bills--Newspapers ("CAP") and SCOTUS LexisNexis--Wikipedia
("Legal") -- with the parent's encoders (MiniLM, E5-base-v2) plus this
project's primary E5-small, and scored on six MTEB targets: the five legal
retrieval tasks of the parent's Tables 7 and 11, by nDCG@10, and
STSBenchmark, by cosine Spearman. `PARENT` holds the parent's reported Raw and
LEACE numbers and `parent_comparison` sets ours beside them.

The transformations
-------------------
* ``pca_top1`` -- **PCA-1**, removal of the first principal direction, the
  operator the parent describes and its own ``remove_first_principal_component``
  computes: ``T(x) = mu + (I - u u^T)(x - mu)`` with the donor's mean and first
  principal direction, frozen (plan Sec. 5.4).
* ``metadata_leace`` -- LEACE on the donor's source/language labels.
* ``sae_leace`` -- the metadata-free pipeline of `protocols.fit_pipeline`.

Further transformations can be added with `register_transform`: each is fitted
on the same donor documents, frozen the same way and scored on every target.

Protocol
--------
* **Fit on the donor, then freeze.** A donor fit is `protocols.fit_pipeline`
  with the recipe of the headline run (`experiments.RECIPE`, the session's
  selector -- `corpora.set_selection`, top-N by default as in the notebook and
  the saved ID results). ``fit_on="full"`` (the default, as the parent fits its
  erasers) uses every donor document; ``fit_on="train"`` uses the donor's 50/50
  ID training split, which makes the OOD maps the ID maps. The SAE,
  pseudo-grouping, selection and LEACE statistics of SAE-LEACE, the donor mean
  and direction of PCA-1, and the source-label fit of metadata-LEACE see donor
  documents only.
* **Nothing is fitted on a target.** Every map is affine, ``T(x) = x A + c``;
  it is read off the frozen fit once (`affine`), saved, and applied to queries
  and documents alike (both sentences of an STS pair). No target PCA,
  re-centering, SAE or selection, and no choice of donor per target.
* **Same encoder, same prompts across conditions.** Donor and target are
  encoded by the same model. Target texts are encoded plainly, as the parent
  evaluated (``PROMPT_STYLE = "parent"``; it reproduces the parent's Raw
  scores exactly), in every condition, Raw included; ``"mteb"`` switches to
  mteb's registered prompts. Donor documents are encoded by
  `experiments.embed`, which gives E5-base-v2 no prefix, as the parent's
  pipeline encodes, and E5-small the ID pipeline's ``"query: "``.
* **Normalization order.** Every encoder here ends in a Normalize module, so
  the map receives unit vectors. Its output is not renormalized; scoring is
  cosine, which a final renormalization would not change.
* **Scoring reproduces mteb 2.20.11.** Targets are read at the dataset
  revisions mteb pins; retrieval is cosine ranking with pytrec_eval's tie-break
  (document id, descending) and nDCG@10 with linear gains; STS is the Spearman
  correlation of pair cosines with the gold scores. Per-query nDCG and per-pair
  cosines are kept for the paired bootstrap in `summarize`.

`run_donor` runs one donor/encoder; `run_matrix` runs `MATRIX`; `summarize`
turns the saved results into the numbers behind Tables C1-C3 and Figures 1/C1
(`tables.table_ood_*`, `plots.plot_ood_*`). From the repository root::

    python code/ood.py run --all          # every donor/encoder in MATRIX
    python code/ood.py run --encoder mini --donor cap_bills_news
    python code/ood.py report             # summary, tables and figures
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time

import numpy as np

import experiments as E
import protocols as P
from artifacts import load_result, save_result

#: The version of mteb whose task definitions the targets below were resolved
#: from (dataset repository, pinned revision, evaluation split, main score).
MTEB_VERSION = "2.20.11"
#: The six fixed targets, in reporting order: the five legal retrieval tasks,
#: then STSBenchmark. `label` is the short name the tables and figures use.
TARGETS = {
    "AILACasedocs": dict(
        kind="retrieval", label="AILACasedocs", license="cc-by-4.0",
        repo="mteb/AILA_casedocs", revision="4106e6bcc72e0698d714ea8b101355e3e238431a"),
    "AILAStatutes": dict(
        kind="retrieval", label="AILAStatutes", license="cc-by-4.0",
        repo="mteb/AILA_statutes", revision="ebfcd844eadd3d667efa3c57fc5c8c87f5c2867e"),
    "LegalBenchConsumerContractsQA": dict(
        kind="retrieval", label="ConsumerContractsQA", license="cc-by-nc-4.0",
        repo="mteb/legalbench_consumer_contracts_qa",
        revision="b23590301ec94e8087e2850b21d43d4956b1cca9"),
    "LegalBenchCorporateLobbying": dict(
        kind="retrieval", label="CorporateLobbying", license="cc-by-4.0",
        repo="mteb/legalbench_corporate_lobbying",
        revision="f69691c650464e62546d7f2a4536f8f87c891e38"),
    "LegalSummarization": dict(
        kind="retrieval", label="LegalSummarization", license="apache-2.0",
        repo="mteb/legal_summarization", revision="3bb1a05c66872889662af04c5691c14489cebd72"),
    "STSBenchmark": dict(
        kind="sts", label="STSBenchmark", license="not specified",
        repo="mteb/stsbenchmark-sts", revision="b0fddb56ed78048fa8b90373c8a3cfc37b684831"),
}
METRIC = {"retrieval": "nDCG@10", "sts": "Spearman"}
#: Every target's evaluation split in mteb.
SPLIT = "test"
LEGAL = tuple(t for t, v in TARGETS.items() if v["kind"] == "retrieval")
STS = "STSBenchmark"
#: Where the target files are downloaded (git-ignored, like the CAP raw files).
TARGET_DIR = os.path.join("data", "raw", "mteb")

#: The parent's two eraser-fitting collections, "CAP" and "Legal".
DONORS = ("cap_bills_news", "scotus_wiki_lexis")
DONOR_ROLE = {"cap_bills_news": "cap", "scotus_wiki_lexis": "legal"}
#: The parent's OOD encoders, then this project's primary encoder.
ENCODERS = ("mini", "e5-base-v2", "e5-small")
#: Every encoder x donor, in the order of Table C1B (encoders down, donors across).
MATRIX = tuple((enc, don) for enc in ENCODERS for don in DONORS)
#: The fitting seeds, fixed in advance. A full-donor fit makes PCA-1 and
#: metadata-LEACE seed-independent; the seed still varies the SAE.
SEEDS = (0, 1, 2)
#: What a donor fit uses: every donor document (the parent) or its ID training split.
FIT_ON = "full"
#: The frozen transformations compared against Raw (see the module docstring).
TRANSFORMS = ("pca_top1", "metadata_leace", "sae_leace")
CONDITIONS = ("raw",) + TRANSFORMS
#: Display names, carried into every result so tables and figures need no code.
LABELS = {"raw": "Raw", "pca_top1": "PCA-1", "metadata_leace": "Metadata-LEACE",
          "sae_leace": "SAE-LEACE"}

#: Transformations added at run time by `register_transform`: name -> fit(X_train)
#: returning ``((A, c), audit)``. Empty unless something registers one.
EXTRA_TRANSFORMS: dict = {}


def register_transform(name: str, fit, label: str) -> None:
    """Add a frozen transformation to every OOD run in this session.

    `fit` receives the donor documents a fit sees and returns the affine map
    ``(A, c)`` -- ``T(x) = x A + c``, any output dimension -- and a dict for the
    audit. It is fitted, frozen and scored exactly like the built-in three.
    """
    EXTRA_TRANSFORMS[name] = fit
    LABELS[name] = label


def transforms() -> tuple:
    """The built-in transformations, then any registered ones."""
    return TRANSFORMS + tuple(EXTRA_TRANSFORMS)

#: Target-side prompts, per style. "parent" encodes every text plainly, as the
#: parent evaluated: it reproduces the parent's Raw scores exactly, E5-base-v2
#: included (0.292 / 0.186 / 0.720 / 0.915 / 0.577), where mteb's E5 prompts
#: give 0.272 / 0.196 / 0.719 / 0.919 / 0.587. "mteb" uses mteb 2.20's
#: registered prompts: E5 takes "query: " / "passage: " on retrieval queries /
#: documents and nothing on STS sentences (mteb passes STS no prompt type).
_NONE = {"query": "", "document": "", "sts": ""}
_E5 = {"query": "query: ", "document": "passage: ", "sts": ""}
PROMPT_STYLES = {
    "parent": {"e5-small": _NONE, "e5-base-v2": _NONE, "mini": _NONE},
    "mteb": {"e5-small": _E5, "e5-base-v2": _E5, "mini": _NONE},
}
PROMPT_STYLE = "parent"
PROMPTS = PROMPT_STYLES[PROMPT_STYLE]

#: The parent paper's reported MTEB scores (Fan et al., 2025): Raw and
#: metadata-LEACE fit on each donor, from its Tables 7 (legal nDCG@10) and 9
#: (STSBenchmark Spearman). Per task: (raw, leace_cap, leace_legal).
PARENT = {
    "mini": {
        "AILACasedocs": (0.197, 0.197, 0.218),
        "AILAStatutes": (0.205, 0.196, 0.205),
        "LegalBenchConsumerContractsQA": (0.656, 0.659, 0.654),
        "LegalBenchCorporateLobbying": (0.864, 0.865, 0.863),
        "LegalSummarization": (0.590, 0.591, 0.592),
        "STSBenchmark": (0.820, 0.820, 0.820),
    },
    "e5-base-v2": {
        "AILACasedocs": (0.292, 0.290, 0.292),
        "AILAStatutes": (0.186, 0.191, 0.193),
        "LegalBenchConsumerContractsQA": (0.720, 0.712, 0.720),
        "LegalBenchCorporateLobbying": (0.915, 0.914, 0.913),
        "LegalSummarization": (0.577, 0.576, 0.578),
        "STSBenchmark": (0.855, 0.855, 0.855),
    },
}


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def _files(task: str) -> tuple:
    if TARGETS[task]["kind"] == "sts":
        return (f"{SPLIT}.jsonl.gz",)
    return ("corpus.jsonl", "queries.jsonl", f"qrels/{SPLIT}.jsonl")


def fetch_target(task: str) -> str:
    """The task's directory, downloading its files at the pinned revision once."""
    t, local = TARGETS[task], os.path.join(TARGET_DIR, task)
    if not all(os.path.exists(os.path.join(local, f)) for f in _files(task)):
        from huggingface_hub import snapshot_download
        snapshot_download(t["repo"], repo_type="dataset", revision=t["revision"],
                          local_dir=local, allow_patterns=list(_files(task)))
    return local


def _read_jsonl(path: str) -> list:
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_target(task: str) -> dict:
    """One target as mteb evaluates it.

    Retrieval: the judged queries, the corpus (title and text joined as mteb
    joins them; every title here is empty) and the graded judgements. STS: the
    sentence pairs and their gold scores.
    """
    local = fetch_target(task)
    if TARGETS[task]["kind"] == "sts":
        rows = _read_jsonl(os.path.join(local, f"{SPLIT}.jsonl.gz"))
        return {"task": task, "kind": "sts",
                "s1": [r["sentence1"] for r in rows], "s2": [r["sentence2"] for r in rows],
                "gold": np.array([r["score"] for r in rows], dtype=float)}
    qrels: dict = {}
    for r in _read_jsonl(os.path.join(local, "qrels", f"{SPLIT}.jsonl")):
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    queries = [q for q in _read_jsonl(os.path.join(local, "queries.jsonl"))
               if str(q["_id"]) in qrels]
    corpus = _read_jsonl(os.path.join(local, "corpus.jsonl"))
    return {"task": task, "kind": "retrieval",
            "query_ids": [str(q["_id"]) for q in queries],
            "queries": [q["text"] for q in queries],
            "doc_ids": [str(d["_id"]) for d in corpus],
            "docs": [f"{d['title']} {d['text']}".strip() if d.get("title") else d["text"]
                     for d in corpus],
            "qrels": qrels}


def target_size(target: dict) -> dict:
    """What Table C1 reports: queries / documents / judgements, or rated pairs."""
    if target["kind"] == "sts":
        return {"pairs": len(target["gold"])}
    return {"queries": len(target["query_ids"]), "documents": len(target["doc_ids"]),
            "judgements": sum(g > 0 for rel in target["qrels"].values() for g in rel.values())}


def _prompts(encoder: str, style: str | None = None) -> dict:
    prompts = PROMPT_STYLES[style or PROMPT_STYLE]
    if encoder not in prompts:
        raise ValueError(f"no target prompt convention for {encoder!r}; add it to "
                         f"ood.PROMPT_STYLES")
    return prompts[encoder]


def embed_targets(encoder: str, tasks=tuple(TARGETS), device: str | None = None,
                  batch_size: int | None = None, style: str | None = None,
                  verbose: bool = True) -> dict:
    """Encode each target once per encoder and prompt style; cached beside the
    corpus embeddings.

    Returns ``{task: {"q", "d"}}`` for retrieval (queries, documents) and
    ``{task: {"a", "b"}}`` for STS (first and second sentences).
    """
    style = style or PROMPT_STYLE
    pr, out, model = _prompts(encoder, style), {}, None
    for task in tasks:
        kind = TARGETS[task]["kind"]
        parts = ("q", "d") if kind == "retrieval" else ("a", "b")
        paths = {p: os.path.join(E.EMBED_CACHE, f"ood__{task}__{encoder}__{style}__{p}.npy")
                 for p in parts}
        if all(os.path.exists(p) for p in paths.values()):
            out[task] = {p: np.load(path) for p, path in paths.items()}
            continue
        if model is None:
            from style_erasure_pipeline import load_embedding_model
            model = load_embedding_model(encoder)
        tgt = load_target(task)
        if kind == "retrieval":
            texts = {"q": [pr["query"] + t for t in tgt["queries"]],
                     "d": [pr["document"] + t for t in tgt["docs"]]}
        else:
            texts = {"a": [pr["sts"] + t for t in tgt["s1"]],
                     "b": [pr["sts"] + t for t in tgt["s2"]]}
        bs = batch_size or E.ENCODER_BATCH.get(encoder, 64)
        out[task] = {}
        for p, tx in texts.items():
            emb = np.asarray(model.encode(tx, convert_to_numpy=True, batch_size=bs,
                                          device=device or E.pick_device(None),
                                          show_progress_bar=False), dtype=np.float32)
            os.makedirs(E.EMBED_CACHE, exist_ok=True)
            np.save(paths[p], emb)
            out[task][p] = emb
        if verbose:
            print(f"[ood] {task}/{encoder}: "
                  + ", ".join(f"{p} {v.shape}" for p, v in out[task].items()))
    return out


def encoder_revision(encoder: str) -> dict:
    """The model id and its resolved Hugging Face commit, for the reproducibility
    tables: transfer is only meaningful within one exact encoder."""
    from style_erasure_pipeline import load_embedding_model
    cfg = load_embedding_model(encoder)[0].auto_model.config
    revision = getattr(cfg, "_commit_hash", None)
    if revision is None:
        revision = _cached_revision(cfg._name_or_path)
    return {"model": cfg._name_or_path, "revision": revision}


def _cached_revision(repo: str):
    """The commit of the model snapshot in the local Hugging Face cache, or None. transformers
    5.19 no longer keeps the commit on the model config (5.17 did); the model has just been
    loaded, so its snapshot is cached, and the snapshot folder is named after the commit."""
    try:
        from huggingface_hub import snapshot_download
        return os.path.basename(snapshot_download(repo, local_files_only=True).rstrip("/"))
    except Exception:                                    # not a Hub repository, or no cache
        return None


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _unit(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def ndcg_at_10(Q, D, query_ids, doc_ids, qrels) -> np.ndarray:
    """Per-query nDCG@10 exactly as pytrec_eval scores an mteb run.

    Documents are ranked by cosine similarity held in single precision --
    trec_eval stores run scores as C floats, so values that differ only past
    float32 resolution are ties there -- with ties broken by document id in
    descending order; gains are the judged relevance grades (linear), the
    discount is 1 / log2(rank + 1), and the ideal ranking is over the query's
    judged relevant documents.
    """
    S = (_unit(Q) @ _unit(D).T).astype(np.float32)
    id_rank = np.argsort(np.argsort(np.asarray(doc_ids)))
    disc = 1.0 / np.log2(np.arange(2, 12))
    out = np.empty(len(query_ids))
    for i, qid in enumerate(query_ids):
        top = np.lexsort((-id_rank, -S[i]))[:10]
        rel = qrels[qid]
        gains = np.array([max(rel.get(doc_ids[j], 0), 0) for j in top], dtype=float)
        ideal = np.sort([g for g in rel.values() if g > 0])[::-1][:10]
        idcg = float(np.dot(ideal, disc[:len(ideal)]))
        out[i] = float(np.dot(gains, disc[:len(gains)])) / idcg if idcg > 0 else 0.0
    return out


def pair_cosines(A, B) -> np.ndarray:
    return (_unit(A) * _unit(B)).sum(axis=1)


def spearman(gold, pred) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(gold, pred).statistic)


def score_target(target: dict, emb: dict, amap=None) -> dict:
    """Score one target under a frozen map (None = Raw).

    The same ``(A, c)`` transforms queries and documents (both STS sentences),
    so there is one map per condition and no side-specific adjustment.
    """
    tf = (lambda X: X) if amap is None else (lambda X: np.asarray(X, np.float64) @ amap[0] + amap[1])
    if target["kind"] == "retrieval":
        per = ndcg_at_10(tf(emb["q"]), tf(emb["d"]), target["query_ids"],
                         target["doc_ids"], target["qrels"])
        return {"score": float(per.mean()), "per_unit": per}
    cos = pair_cosines(tf(emb["a"]), tf(emb["b"]))
    return {"score": spearman(target["gold"], cos), "per_unit": cos}


# --------------------------------------------------------------------------- #
# Donor fits: frozen affine maps
# --------------------------------------------------------------------------- #
def affine(fn, dim: int) -> tuple:
    """``(A, c)`` with ``fn(X) = X @ A + c``, read off a frozen affine eraser."""
    c = np.asarray(fn(np.zeros((1, dim), np.float32)), dtype=np.float64)[0]
    A = np.asarray(fn(np.eye(dim, dtype=np.float32)), dtype=np.float64) - c
    return A, c


def check_operator(amap, mu, X_check=None, fn=None) -> dict:
    """The operator audit of plan Sec. 5.4, for any of the frozen maps.

    With ``T(x) = x A + c`` the removed part is ``M = I - A``. Every eraser here
    is a projection (orthogonal for PCA-1, oblique for LEACE), so `M` should be
    idempotent, its rank is the number of directions removed (1 for PCA-1), and
    the map should leave the donor training mean where it was. `affine_err`
    checks that the stored ``(A, c)`` reproduces the eraser on real documents.
    """
    A, c = amap
    M = np.eye(len(A)) - A
    sv = np.linalg.svd(M, compute_uv=False)
    out = {"dim_in": int(A.shape[0]), "dim_out": int(A.shape[1]),
           # A projection's nonzero singular values are >= 1, so 0.5 separates them.
           "removed_rank": int((sv > 0.5).sum()),
           "idempotent_err": float(np.abs(M @ M - M).max()),
           "symmetric_err": float(np.abs(M - M.T).max()),
           "mean_preserving_err": float(np.abs(mu @ A + c - mu).max())}
    if X_check is not None and fn is not None:
        out["affine_err"] = float(np.abs(np.asarray(fn(X_check), np.float64)
                                         - (np.asarray(X_check, np.float64) @ A + c)).max())
    return out


def _split_id(train_docs) -> str:
    """A short fingerprint of the training documents -- the "split ID" of Table C1."""
    return hashlib.sha1(np.sort(np.asarray(train_docs, np.int64)).tobytes()).hexdigest()[:10]


#: The transformations `protocols.fit_pipeline` fits.
_PIPELINE_ERASERS = ("pca_top1", "metadata_leace", "sae_leace")


def _donor_split(data: dict, seed: int, fit_on: str):
    """The documents a donor fit sees: all of them, or the ID training split."""
    if fit_on == "full":
        return P.make_split(data["n_docs"], data["source"], pairs=data["pairs"],
                            gold_sim=data["gold"], seed=seed, protocol="none")
    if fit_on != "train":
        raise ValueError(f"fit_on must be 'full' or 'train', got {fit_on!r}")
    splits = E._splits(data, seed, "document", E._split_policy(data))
    if len(splits) != 1:
        raise ValueError(f"{data['key']}: OOD donors are fit on one holdout split, "
                         f"its registry gives {len(splits)} folds")
    return splits[0]


def fit_donor(data: dict, seed: int, fit_on: str = FIT_ON, fit_device: str | None = None,
              verbose: bool = False) -> dict:
    """Fit and freeze one seed's maps on the donor (see `FIT_ON`).

    `data` is what `experiments.embed` returns for the donor corpus. Returns the
    maps as ``{condition: (A, c)}`` with the fit's provenance and operator audit.
    """
    cfg = data["config"]
    split = _donor_split(data, seed, fit_on)
    t0 = time.time()
    fitted = P.fit_pipeline(
        data["emb"], data["source"], split, **E.RECIPE,
        selection=cfg["selection"], pseudo_groups=cfg["pseudo_groups"],
        pca_grid=(1,), kmeans_grid=(), kmeans_auto=False, random_control=False,
        device=fit_device, verbose=verbose)
    X = np.asarray(data["emb"], np.float32)
    X_tr, X_chk = X[split.train_docs], X[split.test_docs][:256]
    mu = X_tr.astype(np.float64).mean(axis=0)
    maps = {c: affine(fitted.erasers[c], X.shape[1]) for c in _PIPELINE_ERASERS}
    checks = {c: check_operator(maps[c], mu, X_chk, fitted.erasers[c])
              for c in _PIPELINE_ERASERS}
    for name, fit in EXTRA_TRANSFORMS.items():          # registered ones, same data
        maps[name], checks[name] = fit(X_tr)
    return {
        "seed": int(seed), "maps": maps, "mean": mu,
        "fit": {"fit_on": fit_on, "split_id": _split_id(split.train_docs),
                "n_train_docs": int(len(split.train_docs)),
                "n_heldout_docs": int(len(split.test_docs)),
                "selection": fitted.selection,
                "n_selected": int(len(fitted.selected)),
                "selected": [int(j) for j in fitted.selected],
                "pseudo_groups_k": int(fitted.config["pseudo_groups_k"]),
                "sae": fitted.sae_info, "seconds": round(time.time() - t0, 1)},
        "operator_checks": checks,
    }


def _maps_path(encoder: str, donor: str, seed: int) -> str:
    return os.path.join(E.EMBED_CACHE, "ood_maps", f"{encoder}__{donor}__seed{seed}.npz")


def save_maps(fit: dict, encoder: str, donor: str) -> str:
    """Store one seed's frozen maps so they can be re-applied without refitting."""
    path = _maps_path(encoder, donor, fit["seed"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {"mean": fit["mean"]}
    for c, (A, cvec) in fit["maps"].items():
        arrays[f"{c}__A"], arrays[f"{c}__c"] = A.astype(np.float32), cvec.astype(np.float32)
    np.savez_compressed(path, **arrays)
    return path


def load_maps(encoder: str, donor: str, seed: int) -> dict:
    """Every map stored for one fit, as ``{condition: (A, c)}``."""
    with np.load(_maps_path(encoder, donor, seed)) as z:
        names = [k[:-3] for k in z.files if k.endswith("__A")]
        return {c: (z[f"{c}__A"].astype(np.float64), z[f"{c}__c"].astype(np.float64))
                for c in names}


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
def _norm_text(t) -> str:
    return " ".join(str(t).lower().split())


def donor_target_overlap(donor_texts, targets: dict) -> dict:
    """Target texts that also occur in the donor corpus (whitespace/case-folded).

    The plan excludes known donor/target duplicates. The count is recorded with
    every run so a non-zero one cannot pass unnoticed; with the current donors
    and targets it is zero everywhere.
    """
    seen = {_norm_text(t) for t in donor_texts}
    out = {}
    for task, tgt in targets.items():
        texts = (tgt["queries"] + tgt["docs"] if tgt["kind"] == "retrieval"
                 else tgt["s1"] + tgt["s2"])
        out[task] = int(sum(_norm_text(t) in seen for t in texts))
    return out


def task_records(targets: dict) -> dict:
    """Per-target metadata a result carries: definition, size, and what the
    bootstrap needs (query ids for retrieval, gold scores for STS)."""
    return {t: {**{kk: TARGETS[t][kk] for kk in ("kind", "label", "repo", "revision",
                                                  "license")},
                "metric": METRIC[TARGETS[t]["kind"]], "split": SPLIT,
                "size": target_size(tgt),
                "unit_ids": tgt["query_ids"] if tgt["kind"] == "retrieval" else None,
                "gold": tgt["gold"] if tgt["kind"] == "sts" else None}
            for t, tgt in targets.items()}


def _result_name(encoder: str, donor: str, fit_on: str) -> str:
    return f"ood_{encoder}_{donor}_{fit_on}.json"


def run_donor(encoder: str, donor: str, seeds=SEEDS, tasks=tuple(TARGETS),
              fit_on: str = FIT_ON, device: str | None = None,
              fit_device: str | None = None, save: bool = True,
              verbose: bool = True) -> dict:
    """Fit one donor/encoder's maps per seed and score every target.

    Raw is scored once (it does not depend on the donor or the seed). Saves
    ``ood_<encoder>_<donor>_<fit_on>.json`` in the active artifact scope and the
    maps under ``<embedding cache>/ood_maps/``.
    """
    data = E.embed(donor, encoder, device=device, verbose=verbose)
    targets = {t: load_target(t) for t in tasks}
    embs = embed_targets(encoder, tasks, device=device, verbose=verbose)
    overlap = donor_target_overlap(data["texts"], targets)
    if any(overlap.values()):
        raise ValueError(f"{donor} shares texts with the targets {overlap}; "
                         "exclude them before scoring")
    raw = {t: score_target(targets[t], embs[t]) for t in tasks}
    names = transforms()
    per_seed = []
    for s in seeds:
        if verbose:
            print(f"[ood] {encoder}/{donor}: fitting seed {s} ...")
        fit = fit_donor(data, s, fit_on=fit_on, fit_device=fit_device)
        maps_file = save_maps(fit, encoder, f"{donor}_{fit_on}") if save else None
        scores = {c: {t: score_target(targets[t], embs[t], fit["maps"][c]) for t in tasks}
                  for c in names}
        per_seed.append({"seed": int(s), "fit": fit["fit"],
                         "operator_checks": fit["operator_checks"],
                         "maps_file": maps_file, "scores": scores})
        if verbose:
            print(f"[ood]   seed {s}: {fit['fit']['seconds']}s | " + " | ".join(
                f"{c} " + " ".join(f"{scores[c][t]['score']:.3f}" for t in tasks)
                for c in names))
    res = {
        "experiment": "ood", "encoder": encoder, "donor": donor,
        "donor_label": data["label"], "donor_role": DONOR_ROLE.get(donor),
        "seeds": [int(s) for s in seeds], "fit_on": fit_on,
        "transforms": list(names),
        "labels": {c: LABELS[c] for c in ("raw",) + names},
        "tasks": task_records(targets),
        "raw": raw, "per_seed": per_seed,
        "protocol": {"mteb_version": MTEB_VERSION, "encoder_model": encoder_revision(encoder),
                     "prompt_style": PROMPT_STYLE, "target_prompts": _prompts(encoder),
                     "donor_encoding": "experiments.embed (encode_texts: E5-base-v2 "
                                       "unprefixed as in the parent, E5-small 'query: ')",
                     "normalization": "encoder outputs are unit vectors; maps are applied "
                                      "without renormalization; similarity is cosine",
                     "fit": ("protocols.fit_pipeline on every donor document" if fit_on == "full"
                             else "protocols.fit_pipeline on the donor's ID training split"),
                     "recipe": dict(E.RECIPE),
                     "selection": data["config"]["selection"],
                     "pseudo_groups": data["config"]["pseudo_groups"],
                     "split": E._split_policy(data) if fit_on == "train" else None,
                     "donor_target_overlap": overlap},
    }
    if save:
        save_result(res, _result_name(encoder, donor, fit_on))
    return res


def run_matrix(matrix=MATRIX, seeds=SEEDS, fit_on: str = FIT_ON,
               device: str | None = None, fit_device: str | None = None,
               verbose: bool = True) -> dict:
    """Every donor/encoder combination; returns ``{"<encoder>/<donor>": result}``."""
    return {f"{enc}/{don}": run_donor(enc, don, seeds=seeds, fit_on=fit_on, device=device,
                                      fit_device=fit_device, verbose=verbose)
            for enc, don in matrix}


# --------------------------------------------------------------------------- #
# Summary: absolute scores, changes and paired bootstrap intervals
# --------------------------------------------------------------------------- #
def _spearman_rows(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Spearman correlation of every row pair (tie-averaged ranks); NaN for a row
    whose predictions are all tied, where the correlation is undefined."""
    from scipy.stats import rankdata
    rg, rp = rankdata(gold, axis=1), rankdata(pred, axis=1)
    rg -= rg.mean(axis=1, keepdims=True)
    rp -= rp.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rg * rp).sum(axis=1) / np.sqrt((rg ** 2).sum(axis=1) * (rp ** 2).sum(axis=1))


def _ci(values: np.ndarray, level: float) -> list:
    """Percentile interval over the resamples where the statistic is defined.

    Only a degenerate predictor -- one that ties most pairs -- leaves undefined
    (NaN) resamples; they are dropped rather than allowed to void the interval.
    """
    a = (1.0 - level) / 2.0
    return [float(np.nanquantile(values, a)), float(np.nanquantile(values, 1.0 - a))]


def summarize(results: dict, n_boot: int = 10000, level: float = 0.95, seed: int = 0,
              save: bool = True) -> dict:
    """The numbers behind Tables C2/C3 and Figures 1/C1.

    `results` maps ``"<encoder>/<donor>"`` to a `run_donor` result (or its
    saved JSON). Absolute scores are mean +- SD over the fitting seeds (Raw has
    no fit, so one value). Changes are in the seed-averaged score, with paired
    bootstrap intervals conditional on the fitted maps: the same resampled
    queries (pairs) serve every condition, tasks are resampled separately and
    their deltas averaged for the legal macro-average -- except that the two
    AILA tasks share one query set, which is resampled jointly -- and STS
    Spearman is recomputed on each resample rather than averaged per pair.
    """
    results = {k: load_result(v) for k, v in results.items()}
    first = next(iter(results.values()))
    transforms = first.get("transforms", list(TRANSFORMS))
    out = {"experiment": "ood_summary", "n_boot": n_boot, "level": level,
           "bootstrap_seed": seed, "order": list(results), "transforms": transforms,
           "labels": {**LABELS, **first.get("labels", {})}, "settings": {}}
    for key, r in results.items():
        tasks = list(r["tasks"])
        legal = [t for t in tasks if r["tasks"][t]["kind"] == "retrieval"]
        rng = np.random.default_rng(seed)
        draws, shared = {}, {}
        for t in legal:                          # one draw per distinct query set
            ids = tuple(r["tasks"][t]["unit_ids"])
            if ids not in shared:
                shared[ids] = rng.integers(0, len(ids), size=(n_boot, len(ids)))
            draws[t] = shared[ids]
        raw = {t: np.asarray(r["raw"][t]["per_unit"], float) for t in tasks}
        seeds = r["per_seed"]

        scores = {"raw": {t: {"mean": r["raw"][t]["score"]} for t in tasks}}
        scores["raw"]["legal_macro"] = {"mean": float(np.mean([r["raw"][t]["score"]
                                                              for t in legal]))}
        changes = {}
        for c in transforms:
            per_seed = {t: np.array([s["scores"][c][t]["score"] for s in seeds]) for t in tasks}
            macro = np.mean([per_seed[t] for t in legal], axis=0)
            scores[c] = {t: {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1
                             else 0.0, "values": v.tolist()} for t, v in per_seed.items()}
            scores[c]["legal_macro"] = {"mean": float(macro.mean()),
                                        "std": float(macro.std(ddof=1)) if len(macro) > 1 else 0.0,
                                        "values": macro.tolist()}
            ch, boot_legal = {}, []
            for t in legal:
                unit = np.mean([np.asarray(s["scores"][c][t]["per_unit"], float)
                                for s in seeds], axis=0) - raw[t]
                boot = unit[draws[t]].mean(axis=1)
                boot_legal.append(boot)
                ch[t] = {"delta": float(unit.mean()), "ci": _ci(boot, level)}
            macro_boot = np.mean(boot_legal, axis=0)
            ch["legal_macro"] = {"delta": float(np.mean([ch[t]["delta"] for t in legal])),
                                 "ci": _ci(macro_boot, level)}
            worst = min(legal, key=lambda t: ch[t]["delta"])
            ch["worst_task"] = {"task": worst, "delta": ch[worst]["delta"]}
            changes[c] = ch

        sts = [t for t in tasks if r["tasks"][t]["kind"] == "sts"]
        for t in sts:
            gold = np.asarray(r["tasks"][t]["gold"], float)
            srng = np.random.default_rng(seed + 1)
            idx = srng.integers(0, len(gold), size=(n_boot, len(gold)))
            boot_raw = np.concatenate([_spearman_rows(gold[i], raw[t][i])
                                       for i in np.array_split(idx, max(1, n_boot // 500))])
            for c in transforms:
                preds = [np.asarray(s["scores"][c][t]["per_unit"], float) for s in seeds]
                boot_c = np.mean([np.concatenate([_spearman_rows(gold[i], p[i])
                                                  for i in np.array_split(idx, max(1, n_boot // 500))])
                                  for p in preds], axis=0)
                changes[c][t] = {"delta": float(scores[c][t]["mean"] - scores["raw"][t]["mean"]),
                                 "ci": _ci(boot_c - boot_raw, level)}
        out["settings"][key] = {
            "encoder": r["encoder"], "donor": r["donor"], "donor_label": r["donor_label"],
            "donor_role": r.get("donor_role"), "fit_on": r.get("fit_on", "train"),
            "seeds": r["seeds"], "tasks": {t: {kk: r["tasks"][t][kk] for kk in
                                               ("kind", "label", "metric", "revision", "size")}
                                           for t in tasks},
            "legal": legal, "sts": sts,
            "n_train_docs": [s["fit"]["n_train_docs"] for s in seeds],
            "split_ids": [s["fit"]["split_id"] for s in seeds],
            "scores": scores, "changes": changes}
    out["parent_comparison"] = parent_comparison(out)
    if save:
        save_result(out, "ood_summary.json")
    return out


def parent_comparison(summary: dict) -> list:
    """Our Raw and metadata-LEACE scores beside the parent's (Tables 7 and 9).

    Only settings the parent ran (MiniLM and E5-base-v2, on its two donors)
    have a counterpart. A check that data, encoders, fits and evaluation
    reproduce the parent's pipeline.
    """
    rows = []
    for key in summary["order"]:
        v = summary["settings"][key]
        ref, role = PARENT.get(v["encoder"]), v.get("donor_role")
        if ref is None or role is None:
            continue
        j = {"cap": 0, "legal": 1}[role]
        for t in v["legal"] + v["sts"]:
            raw_p, lc, ll = ref[t]
            rows.append({"setting": key, "encoder": v["encoder"], "donor": v["donor"],
                         "role": role, "task": t, "label": v["tasks"][t]["label"],
                         "metric": v["tasks"][t]["metric"],
                         "ours": {c: v["scores"][c][t]["mean"]
                                  for c in ("raw", "metadata_leace")},
                         "parent": {"raw": raw_p, "metadata_leace": (lc, ll)[j]}})
    return rows


def format_parent_comparison(rows: list) -> str:
    """The replication check as plain text, for a terminal or notebook."""
    lines = [f"{'encoder':11s} {'donor':6s} {'task':20s} {'raw':>13s} {'meta-LEACE':>13s}",
             " " * 40 + "  ours parent   ours parent"]
    for r in rows:
        o, p = r["ours"], r["parent"]
        lines.append(f"{r['encoder']:11s} {r['role']:6s} {r['label']:20s} "
                     f"{o['raw']:6.3f} {p['raw']:6.3f} {o['metadata_leace']:6.3f} "
                     f"{p['metadata_leace']:6.3f}")
    return "\n".join(lines)


def load_saved(matrix=MATRIX, fit_on: str = FIT_ON) -> dict:
    """The saved per-donor results in the active artifact scope, in `MATRIX` order."""
    out = {}
    for enc, don in matrix:
        try:
            out[f"{enc}/{don}"] = load_result(_result_name(enc, don, fit_on))
        except FileNotFoundError:
            print(f"[ood] no saved result for {enc}/{don} ({fit_on} fit)")
    return out


def report(results: dict | None = None, n_boot: int = 10000, fit_on: str = FIT_ON,
           zoom=(-0.08, 0.05), show: bool = False) -> dict:
    """Summary JSON, Tables C1-C3, the parent comparison and Figures 1/C1.

    Figure 1 is drawn at full scale and again zoomed to `zoom`, so that
    near-zero changes stay legible whatever the largest change is.
    """
    import plots
    import tables
    results = results or load_saved(fit_on=fit_on)
    summary = summarize(results, n_boot=n_boot)
    tables.table_ood_inventory(summary)
    tables.table_ood_scores(summary)
    tables.table_ood_changes(summary)
    tables.table_ood_parent(summary)
    plots.plot_ood_retrieval_changes(summary, show=show)
    if zoom:
        plots.plot_ood_retrieval_changes(summary, xlim=zoom, show=show)
    plots.plot_ood_sts_changes(summary, show=show)
    print(format_parent_comparison(parent_comparison(summary)))
    return summary


def main():
    import artifacts
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="fit donors and score the targets")
    run.add_argument("--all", action="store_true", help="every combination in MATRIX")
    run.add_argument("--encoder", choices=ENCODERS)
    run.add_argument("--donor", choices=sorted({d for _, d in MATRIX}))
    run.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    run.add_argument("--device", default=None, help="encoding device (cuda, mps, cpu)")
    run.add_argument("--selector", choices=("topn", "registry"), default="topn",
                     help="latent selector of the donor fits; 'topn' (the default) is "
                          "the notebook's SELECTOR and what the ID results use, "
                          "'registry' restores greedy selection on paired corpora")
    rep = sub.add_parser("report", help="summary, tables and figures from saved runs")
    rep.add_argument("--n-boot", type=int, default=10000)
    for p in (run, rep):
        p.add_argument("--fit-on", choices=("full", "train"), default=FIT_ON,
                       help="fit donors on every document (the parent) or on the "
                            "ID training split")
    args = parser.parse_args()

    if args.cmd == "run":
        import corpora
        corpora.set_selection(None if args.selector == "registry" else "topn")
    artifacts.set_section("C")
    with artifacts.experiment("ood"):
        if args.cmd == "report":
            report(n_boot=args.n_boot, fit_on=args.fit_on)
        elif args.all:
            run_matrix(seeds=tuple(args.seeds), fit_on=args.fit_on, device=args.device)
        elif args.encoder and args.donor:
            run_donor(args.encoder, args.donor, seeds=tuple(args.seeds),
                      fit_on=args.fit_on, device=args.device)
        else:
            parser.error("run needs --all, or both --encoder and --donor")


if __name__ == "__main__":
    main()
