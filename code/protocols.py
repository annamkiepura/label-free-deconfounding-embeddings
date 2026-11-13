"""Dataset splitting and the fit -> freeze -> apply evaluation engine.

Every experiment in the paper runs through this module, so the split protocol is
declared once and recorded in every result JSON rather than being re-implemented
(and silently varied) per experiment.

Two protocols
-------------
``"document"`` -- **the default.** A 50/50 split of the corpus; every learned
component (the SAE, the k-means pseudo-grouping, the latent scores, the latent
selector and every eraser, baselines included) is fit on the training documents,
frozen, and applied to the held-out documents. All metrics are computed on the
held-out half only. On paired corpora the split is taken at the *pair* level so
both members of a pair land on the same side and held-out retrieval is well
defined among held-out documents; on unpaired corpora it is stratified by the
confounder class.

``"pair"`` -- the weaker, transductive protocol: the retrieval *links* are split
but every document embedding is visible to the pipeline. Kept because it is what
the standard pair-holdout reports, and because running both protocols on the same
corpus is itself an experiment (`experiments.protocol_comparison`).

``"none"`` -- no holdout at all: one fit on the whole corpus, every pair visible
to the selector. This exists for the **descriptive** analyses of "what is being
removed?" (reading the selected latents, the geometry of the removed subspace),
where the object of study is what the method finds on a corpus rather than how
well it generalises. Nothing evaluative may be quoted from such a fit, and
`evaluate` refuses to score one unless explicitly overridden -- the guard is
there so a descriptive run cannot quietly become a performance number.

Partially paired corpora
------------------------
A corpus may hold documents that belong to no pair (the parent paper's SCOTUS
construction: 1,024 case pairs plus an unpaired remainder pooled with them).
Under ``"document"`` the pairs are split at the pair level and the unpaired
documents are split stratified by class; both parts are pooled on each side, so
the eraser is fit on paired and unpaired training documents alike and the
unpaired held-out documents count towards leakage, purity and retention while
only the held-out pairs are retrieval targets -- which is how the paper uses
its pooled data.

K-fold: fold-wise evaluation, one result per seed
--------------------------------------------------
When a corpus is split into K folds (`make_folds`), `evaluate_folds` scores a
seed as **one** result. Retrieval is done fold by fold with a single eraser per
index -- the fold's held-out pairs as queries against the whole corpus erased by
that fold's eraser -- and the per-pair hits are concatenated, so every pair is
a query exactly once and the paired bootstrap runs over all of them. The
probe-based metrics (linear and nonlinear leakage) and cluster purity are
computed inside each held-out fold, cross-validated over that fold's documents,
and averaged over folds. Neighbourhood retention scores the held-out documents
of each fold against neighbours drawn from the whole corpus erased by that
fold's eraser, so a small class is not reduced to a dozen candidates.

Why the probe is NOT pooled over folds. Pooling the out-of-fold erased
documents into one matrix and cross-validating a single probe over it looks
more powerful, but it is biased below chance: each fold's residual after
erasure is that fold's class-mean shift minus the other folds' estimate, so the
fold residuals sum to zero over the corpus, and a probe trained on one part of
the pooled matrix meets residuals anti-correlated with the part it is tested
on. On synthetic data where the true out-of-sample leakage of the oracle was
0.71 (probe on a fresh sample), the per-fold probe read 0.58 and the pooled
probe 0.25. The per-fold probe is therefore the honest choice; it errs on the
conservative side on small folds, which is a limitation of corpus size that the
n/d diagnostic flags rather than something a different estimator can remove.

Retrieval index
---------------
Recall is measured with the held-out pairs as **queries** against the eraser
applied to the **whole corpus** (``retrieval_index="full"``, the default). This
is the deployment setting: freeze the eraser, transform the corpus, retrieve
inside it. It stays fully out of sample -- the eraser never saw the held-out
documents -- while the training documents act as additional distractors, which
makes the task harder rather than leakier. It also keeps the candidate pool the
same size no matter how the corpus was split, so recall is comparable across
protocols and across fold counts.

``retrieval_index="heldout"`` restores the older behaviour of ranking only within
the held-out half. That pool shrinks as the evaluation share shrinks, so recall
under it is not comparable across split schemes; it is kept only for
reproducing earlier runs.

Under ``"document"`` no held-out document can influence any fitted component, so
the *systematic* below-chance anti-correlation that motivates flip-adjustment and
cross-fitting cannot arise from the eraser. A second such mechanism lives in
the probe itself on paired corpora: after a good erasure a document and its
counterpart are near-identical vectors with opposite labels, so a probe whose
folds split pairs at random meets the twin of almost every test document in
its training portion and reads below chance. The probe folds therefore keep
pairs together (`probe_groups`), and with that ``leakage_auc`` is the honest
number. (An individual AUC can still land a little under 0.5 by sampling
noise; what is gone are the mechanisms that drove the oracle to 0.07.)
``leakage_auc_flip`` is reported alongside for continuity;
``crossfit_leakage_auc`` is reported **only** under
``"pair"``, where the eraser saw everything and cross-fitting is the repair --
under ``"document"`` it would refit the eraser inside the evaluation half, which
is a different and weaker estimator, not a robustness check.

Label discipline
----------------
`fit_pipeline` receives `source_labels` **only** to fit the `metadata_leace`
oracle condition, which is defined as label-supervised. Steps (1)-(4) of the
method -- decompose, score, select, erase -- never see them, and neither do
they see the *number* of confounder classes: the pseudo-group count K is chosen
by silhouette on the training documents (`choose_pseudo_groups`), so the SAE
condition's eraser is built from the corpus geometry alone. `evaluate` uses
labels freely: that is evaluation, not fitting. (The split is stratified by
class so that both halves are balanced; that is protocol, not fitting.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from concept_erasure import LeaceEraser
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sae import encode_features, train_sae
from sae_confounder_pipeline import (
    _cosine,
    cluster_purity,
    crossfit_leakage_auc,
    gold_similarity_spearman,
    pair_hit_vector,
    paired_retrieval_recall,
    probe_leakage,
    score_confounder_features,
    select_confounder_features,
    select_confounder_features_greedy,
)

PROTOCOLS = ("document", "pair", "none")

#: Candidate pseudo-group counts for the label-free choice of K.
PSEUDO_GROUP_GRID = (2, 3, 4, 5, 6, 8)


def choose_pseudo_groups(X: np.ndarray, ks=PSEUDO_GROUP_GRID, seed: int = 0,
                         cap: int = 4000) -> tuple:
    """Pick the number of k-means pseudo-groups K without labels.

    The coarse grouping G that the MI score is computed against must not be
    told how many confounder classes there are -- that would inject knowledge
    of the confounder into the one step meant to be blind to it. K is therefore
    chosen by the silhouette score of a k-means clustering of the TRAINING
    embeddings over the candidate grid, the largest silhouette winning (ties to
    the smaller K). A random subset of at most `cap` documents keeps the
    silhouette's pairwise distances affordable. Returns ``(K, {K: silhouette})``.
    """
    from sklearn.metrics import silhouette_score

    X = np.asarray(X, dtype=np.float32)
    idx = np.random.default_rng(seed).permutation(len(X))[: min(cap, len(X))]
    Xs = X[idx]
    sil = {}
    for K in ks:
        K = int(K)
        if K >= len(Xs):
            continue
        labels = KMeans(n_clusters=K, random_state=seed, n_init="auto").fit_predict(Xs)
        sil[K] = float(silhouette_score(Xs, labels))
    if not sil:                       # degenerate corpus smaller than the grid
        return 2, {}
    best = max(sorted(sil), key=lambda K: sil[K])
    return int(best), sil


def _resolve_pseudo_groups(pseudo_groups, X_tr: np.ndarray, seed: int) -> tuple:
    """``"auto"`` -> silhouette choice; an int is used as given (for sweeps)."""
    if pseudo_groups == "auto" or pseudo_groups is None:
        return choose_pseudo_groups(X_tr, seed=seed)
    return int(pseudo_groups), {}


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    """One seeded train/evaluation split of a corpus.

    `train_docs` / `test_docs` index the FULL corpus. `train_pairs` /
    `test_pairs` are pairs of positions *within* those index arrays, so a caller
    slices once (``X[split.train_docs]``) and then works in local coordinates.
    Under ``protocol="pair"`` the two document arrays are both the whole corpus
    and the pair lists are the two halves of the pairing.
    """

    protocol: str
    seed: int
    holdout: float
    train_docs: np.ndarray
    test_docs: np.ndarray
    train_pairs: list | None = None
    test_pairs: list | None = None
    gold_test: np.ndarray | None = None
    #: The same held-out pairs in FULL-corpus coordinates. Retrieval uses these
    #: so a held-out query can be ranked against the whole erased corpus rather
    #: than only against its own half -- see `evaluate`'s `retrieval_index`.
    test_pairs_global: list | None = None
    #: Fold index and count when the split came from `make_folds`.
    fold: int | None = None
    n_folds: int | None = None

    @property
    def has_pairs(self) -> bool:
        return bool(self.test_pairs)

    def summary(self) -> dict:
        n_tr_p, n_te_p = len(self.train_pairs or []), len(self.test_pairs or [])
        return {
            "protocol": self.protocol, "seed": int(self.seed),
            "holdout": float(self.holdout),
            "n_train_docs": int(len(self.train_docs)),
            "n_test_docs": int(len(self.test_docs)),
            "n_train_pairs": int(n_tr_p),
            "n_test_pairs": int(n_te_p),
            "n_test_unpaired": int(len(self.test_docs) - _n_linked(self.test_pairs))
            if self.protocol == "document" else None,
            "documents_disjoint": bool(self.protocol == "document"),
            "performance_valid": bool(self.protocol != "none"),
            "fold": self.fold,
            "n_folds": self.n_folds,
        }


def _unpaired_docs(n_docs: int, pairs) -> np.ndarray:
    """Indices of documents that belong to no pair."""
    if pairs is None or len(pairs) == 0:
        return np.arange(n_docs)
    return np.setdiff1d(np.arange(n_docs), np.unique(np.asarray(pairs)))


def _n_linked(pairs) -> int:
    """How many documents belong to some pair (2 per pair when pairs are disjoint)."""
    return int(len(np.unique(np.asarray(pairs)))) if pairs else 0


def _disjoint(pairs) -> bool:
    """Whether every document belongs to at most one pair."""
    arr = np.asarray(pairs)
    return len(np.unique(arr)) == arr.size


def link_groups(pairs) -> tuple:
    """The linked groups of a pairing: ``(group of each pair, documents of each group)``.

    A group is a connected component of the pair links -- one case summarised by
    several sources, one decision in several languages. Groups are numbered in
    order of their first pair and list their documents in order of appearance.
    With disjoint pairs every pair is its own group, so ``group[j] == j`` and the
    documents of group j are pair j itself.
    """
    arr = np.asarray(pairs, dtype=int).reshape(-1, 2)
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in arr:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    ids, of_pair, members = {}, [], []
    for a, b in arr:
        g = ids.setdefault(find(int(a)), len(members))
        if g == len(members):
            members.append([])
        of_pair.append(g)
        for d in (int(a), int(b)):
            if d not in members[g]:
                members[g].append(d)
    return np.asarray(of_pair, dtype=int), [np.asarray(m, dtype=int) for m in members]


def _group_sides(arr: np.ndarray, groups: list, of_pair: np.ndarray, test_groups,
                 train_groups, te_u: np.ndarray, tr_u: np.ndarray) -> tuple:
    """Documents and pairs of a group-level split, both sides.

    Each side lists its groups' documents (in the order the groups are given), then
    its unpaired documents; its pairs are the pairs of its groups, in local and (for
    the test side) full-corpus coordinates.
    """
    out = []
    for gs, unp in ((train_groups, tr_u), (test_groups, te_u)):
        docs = np.concatenate([groups[g] for g in gs] + [np.asarray(unp, dtype=int)])
        pos = {int(d): i for i, d in enumerate(docs)}
        sel = arr[np.isin(of_pair, np.asarray(gs, dtype=int))]
        out.append((docs, [(pos[int(a)], pos[int(b)]) for a, b in sel],
                    [(int(a), int(b)) for a, b in sel]))
    (tr_docs, tr_pairs, _), (te_docs, te_pairs, te_glob) = out
    return tr_docs, te_docs, tr_pairs, te_pairs, te_glob


def _make_group_split(n_docs: int, y: np.ndarray, pairs, holdout: float, seed: int,
                      protocol: str) -> Split:
    """`make_split` for a pairing whose groups are larger than a pair.

    In a three-class corpus a case's documents are linked into a triple, so a
    document belongs to two pairs. The split is taken over groups instead of
    pairs: a group's documents -- and with them all of its pairs -- land on the
    same side, so no held-out document has a counterpart the eraser was fit on.
    Unpaired documents are split stratified by class, as for disjoint pairs.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(pairs, dtype=int)
    of_pair, groups = link_groups(arr)
    perm = rng.permutation(len(groups))
    n_te = int(np.clip(round(holdout * len(groups)), 1, len(groups) - 1))
    te_g, tr_g = perm[:n_te], perm[n_te:]
    as_list = lambda sel: [tuple(map(int, p)) for p in sel]            # noqa: E731
    if protocol == "none":
        allidx = np.arange(n_docs)
        return Split(protocol="none", seed=seed, holdout=0.0, train_docs=allidx,
                     test_docs=allidx, train_pairs=as_list(arr), test_pairs=as_list(arr),
                     test_pairs_global=as_list(arr))
    if protocol == "pair":
        allidx = np.arange(n_docs)
        te_glob = as_list(arr[np.isin(of_pair, te_g)])
        return Split(protocol="pair", seed=seed, holdout=holdout, train_docs=allidx,
                     test_docs=allidx, train_pairs=as_list(arr[np.isin(of_pair, tr_g)]),
                     test_pairs=te_glob, test_pairs_global=te_glob)
    tr_u, te_u = _stratified_holdout(_unpaired_docs(n_docs, arr), y, holdout, rng)
    tr_docs, te_docs, tr_pairs, te_pairs, te_glob = _group_sides(
        arr, groups, of_pair, te_g, tr_g, te_u, tr_u)
    return Split(protocol="document", seed=seed, holdout=holdout,
                 train_docs=tr_docs, test_docs=te_docs, train_pairs=tr_pairs,
                 test_pairs=te_pairs, test_pairs_global=te_glob)


def _stratified_holdout(idx: np.ndarray, y: np.ndarray, holdout: float, rng):
    """Split `idx` into (train, test) with the class balance of `y[idx]` kept."""
    tr, te = [], []
    for c in np.unique(y[idx]):
        sub = rng.permutation(idx[y[idx] == c])
        n_te = int(np.clip(round(holdout * len(sub)), 1, max(len(sub) - 1, 0)))
        te.append(sub[:n_te])
        tr.append(sub[n_te:])
    return (np.concatenate(tr) if tr else np.array([], int),
            np.concatenate(te) if te else np.array([], int))


def _stratified_blocks(idx: np.ndarray, y: np.ndarray, folds: int, rng) -> list:
    """Split `idx` into `folds` blocks, each keeping the class balance."""
    per_class = []
    for c in np.unique(y[idx]):
        per_class.append(np.array_split(rng.permutation(idx[y[idx] == c]), folds))
    return [np.concatenate([pc[f] for pc in per_class]) if per_class
            else np.array([], int) for f in range(folds)]


def make_split(
    n_docs: int,
    source_labels,
    pairs=None,
    gold_sim=None,
    holdout: float = 0.5,
    seed: int = 0,
    protocol: str = "document",
) -> Split:
    """Build a `Split` for one seed. See the module docstring for the protocols.

    When a document belongs to more than one pair (a three-class corpus), the
    split is taken over linked groups instead (`_make_group_split`).
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    rng = np.random.default_rng(seed)
    y = np.asarray(source_labels)
    has_pairs = pairs is not None and len(pairs) > 0
    if has_pairs and not _disjoint(pairs):
        if gold_sim is not None:
            raise ValueError("graded similarity is defined per pair; it needs disjoint pairs")
        return _make_group_split(n_docs, y, pairs, holdout, seed, protocol)

    if has_pairs:
        arr = np.asarray(pairs)
        perm = rng.permutation(len(arr))
        n_te = int(np.clip(round(holdout * len(arr)), 1, len(arr) - 1))
        te_p, tr_p = arr[perm[:n_te]], arr[perm[n_te:]]
        gold_test = (np.asarray(gold_sim)[perm[:n_te]] if gold_sim is not None else None)
    else:
        te_p = tr_p = gold_test = None

    if protocol == "none":
        # Descriptive: one fit on everything, no holdout of documents or links.
        allidx = np.arange(n_docs)
        allpairs = ([tuple(map(int, p)) for p in np.asarray(pairs)]
                    if has_pairs else None)
        return Split(
            protocol="none", seed=seed, holdout=0.0,
            train_docs=allidx, test_docs=allidx,
            train_pairs=allpairs, test_pairs=allpairs,
            test_pairs_global=allpairs,
            gold_test=(np.asarray(gold_sim) if gold_sim is not None else None),
        )

    if protocol == "pair":
        # Transductive: every document is visible; only the links are split.
        allidx = np.arange(n_docs)
        te_glob = [tuple(map(int, p)) for p in te_p] if has_pairs else None
        return Split(
            protocol="pair", seed=seed, holdout=holdout,
            train_docs=allidx, test_docs=allidx,
            train_pairs=([tuple(map(int, p)) for p in tr_p] if has_pairs else None),
            test_pairs=te_glob, test_pairs_global=te_glob, gold_test=gold_test,
        )

    # protocol == "document": the held-out documents leave the pipeline entirely.
    # Documents outside every pair (the whole corpus when there are no pairs)
    # are split stratified by the confounder class and pooled with the pairs'
    # documents on each side.
    tr_u, te_u = _stratified_holdout(_unpaired_docs(n_docs, pairs), y, holdout, rng)
    if has_pairs:
        # Pair members first, so a pair is (j, m + j) in local coordinates,
        # then the unpaired documents of that side.
        train_docs = np.concatenate([tr_p[:, 0], tr_p[:, 1], tr_u])
        test_docs = np.concatenate([te_p[:, 0], te_p[:, 1], te_u])
        m_tr, m_te = len(tr_p), len(te_p)
        train_pairs = [(j, m_tr + j) for j in range(m_tr)]
        test_pairs = [(j, m_te + j) for j in range(m_te)]
        test_pairs_global = [tuple(map(int, p)) for p in te_p]
    else:
        train_docs, test_docs = tr_u, te_u
        train_pairs = test_pairs = test_pairs_global = None

    return Split(
        protocol="document", seed=seed, holdout=holdout,
        train_docs=train_docs, test_docs=test_docs,
        train_pairs=train_pairs, test_pairs=test_pairs,
        test_pairs_global=test_pairs_global, gold_test=gold_test,
    )


def make_folds(
    n_docs: int,
    source_labels,
    pairs=None,
    gold_sim=None,
    folds: int = 5,
    seed: int = 0,
    protocol: str = "document",
) -> list:
    """K disjoint evaluation folds, each with the other K-1 as its training set.

    Why prefer this to a single 50/50 split: the eraser is the estimation
    bottleneck, not the metrics. LEACE whitens with a d x d covariance fit on the
    training documents, so on a small corpus a half-corpus fit is unstable and
    the held-out numbers understate the method -- with e5-small (d=384), SemEval's
    888 documents give only 444 training rows, barely more than the dimension.
    Five folds raise that to 710 while still scoring every document out of
    sample, which lowers bias and variance at once. The cost is K fits per seed.

    On paired corpora the folds are taken over **pairs**, so both members of a
    pair always land on the same side (over linked groups when a document belongs
    to several pairs). On unpaired corpora they are stratified by the confounder
    class. Returns a list of `Split`s tagged with `fold`.
    """
    if protocol != "document":
        raise ValueError("K-fold is only meaningful for protocol='document'; "
                         f"got {protocol!r}")
    if folds < 2:
        raise ValueError(f"folds must be >= 2, got {folds}")
    rng = np.random.default_rng(seed)
    y = np.asarray(source_labels)
    has_pairs = pairs is not None and len(pairs) > 0
    if has_pairs and not _disjoint(pairs):
        if gold_sim is not None:
            raise ValueError("graded similarity is defined per pair; it needs disjoint pairs")
        arr = np.asarray(pairs, dtype=int)
        of_pair, groups = link_groups(arr)
        blocks = np.array_split(rng.permutation(len(groups)), folds)
        u_blocks = _stratified_blocks(_unpaired_docs(n_docs, pairs), y, folds, rng)
        out = []
        for f in range(folds):
            tr_g = np.concatenate([blocks[g] for g in range(folds) if g != f])
            tr_u = np.concatenate([u_blocks[g] for g in range(folds) if g != f])
            tr_docs, te_docs, tr_pairs, te_pairs, te_glob = _group_sides(
                arr, groups, of_pair, blocks[f], tr_g, u_blocks[f], tr_u)
            out.append(Split(protocol="document", seed=seed, holdout=1.0 / folds,
                             train_docs=tr_docs, test_docs=te_docs, train_pairs=tr_pairs,
                             test_pairs=te_pairs, test_pairs_global=te_glob,
                             fold=f, n_folds=folds))
        return out

    if has_pairs:
        arr = np.asarray(pairs)
        perm = rng.permutation(len(arr))
        blocks = np.array_split(perm, folds)
    # Unpaired documents (all of them when there are no pairs): K stratified
    # blocks, zipped so every fold keeps the corpus's class balance.
    u_blocks = _stratified_blocks(_unpaired_docs(n_docs, pairs), y, folds, rng)

    out = []
    for f in range(folds):
        te_u = u_blocks[f]
        tr_u = np.concatenate([u_blocks[g] for g in range(folds) if g != f])
        if has_pairs:
            te_rows = blocks[f]
            tr_rows = np.concatenate([blocks[g] for g in range(folds) if g != f])
            te_p, tr_p = arr[te_rows], arr[tr_rows]
            train_docs = np.concatenate([tr_p[:, 0], tr_p[:, 1], tr_u])
            test_docs = np.concatenate([te_p[:, 0], te_p[:, 1], te_u])
            m_tr, m_te = len(tr_p), len(te_p)
            sp = Split(
                protocol="document", seed=seed, holdout=1.0 / folds,
                train_docs=train_docs, test_docs=test_docs,
                train_pairs=[(j, m_tr + j) for j in range(m_tr)],
                test_pairs=[(j, m_te + j) for j in range(m_te)],
                test_pairs_global=[tuple(map(int, p)) for p in te_p],
                gold_test=(np.asarray(gold_sim)[te_rows]
                           if gold_sim is not None else None),
                fold=f, n_folds=folds)
        else:
            sp = Split(protocol="document", seed=seed, holdout=1.0 / folds,
                       train_docs=tr_u, test_docs=te_u,
                       fold=f, n_folds=folds)
        out.append(sp)
    return out


def splits_for(
    n_docs: int,
    source_labels,
    pairs=None,
    gold_sim=None,
    scheme: str = "holdout",
    holdout: float = 0.5,
    folds: int = 5,
    seed: int = 0,
    protocol: str = "document",
) -> list:
    """The splits one seed contributes, under either scheme.

    ``scheme="holdout"`` yields a single split; ``scheme="kfold"`` yields `folds`
    of them. Every runner iterates over this, so switching a corpus from one
    scheme to the other is a registry edit rather than a code change.
    """
    if scheme == "kfold":
        return make_folds(n_docs, source_labels, pairs=pairs, gold_sim=gold_sim,
                          folds=folds, seed=seed, protocol=protocol)
    if scheme != "holdout":
        raise ValueError(f"unknown scheme {scheme!r}; expected 'holdout' or 'kfold'")
    return [make_split(n_docs, source_labels, pairs=pairs, gold_sim=gold_sim,
                       holdout=holdout, seed=seed, protocol=protocol)]


# --------------------------------------------------------------------------- #
# Erasers: fit on train, return a frozen apply(X) -> X_erased
# --------------------------------------------------------------------------- #
def _leace_apply(eraser):
    return lambda X: eraser(torch.from_numpy(np.asarray(X, np.float32)).float()).numpy()


def _fit_leace(X_tr: np.ndarray, Z_tr: np.ndarray, standardise: bool = True):
    """Fit LEACE on training data; return a frozen ``apply(X) -> X_erased``.

    `standardise` scales each concept column to unit variance before the fit.
    LEACE's projection depends only on the column space of the whitened
    cross-covariance, so this is a numerical convenience (it keeps the
    library's singular-value cut-off meaningful for continuous concepts) and
    does not change which subspace is removed. One-hot concepts skip it.
    """
    X_tr = np.asarray(X_tr, dtype=np.float32)
    Z_tr = np.asarray(Z_tr, dtype=np.float32)
    if standardise:
        scaler = StandardScaler().fit(Z_tr)
        Z_fit = scaler.transform(Z_tr).astype(np.float32)
    else:
        Z_fit = Z_tr
    eraser = LeaceEraser.fit(torch.from_numpy(X_tr).float(),
                             torch.from_numpy(Z_fit).float())
    return _leace_apply(eraser)


def _fit_orth(D: np.ndarray):
    """Hard orthogonal projection out of span(D). Frozen by construction."""
    _, s, Vt = np.linalg.svd(np.asarray(D, np.float32), full_matrices=False)
    B = Vt[: int((s > 1e-6).sum())]
    return lambda X: (np.asarray(X, np.float32)
                      - (np.asarray(X, np.float32) @ B.T) @ B).astype(np.float32)


def _fit_pca_removal(X_tr: np.ndarray, r: int):
    """Fit the top-r PCs on TRAIN; project them out of anything applied later."""
    X_tr = np.asarray(X_tr, dtype=np.float32)
    mu = X_tr.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X_tr - mu, full_matrices=False)
    B = Vt[:r]

    def apply(X):
        Xc = np.asarray(X, np.float32) - mu
        return (Xc - (Xc @ B.T) @ B + mu).astype(np.float32)

    return apply, B


def _one_hot(labels, classes=None) -> np.ndarray:
    classes = np.unique(labels) if classes is None else np.asarray(classes)
    idx = np.searchsorted(classes, np.asarray(labels))
    return np.eye(len(classes), dtype=np.float32)[idx]


# --------------------------------------------------------------------------- #
# The fitted pipeline
# --------------------------------------------------------------------------- #
@dataclass
class TrainedBase:
    """The expensive, variant-independent half of a fit: SAE + grouping + scores.

    These depend only on the corpus and the seed -- not on which latents get
    selected, which erasure mechanism is used, or what lambda the greedy
    objective runs with. Sweeps over those knobs therefore fit the base once per
    (corpus, seed) and pass it to `fit_pipeline(base=...)`, which cuts a
    four-variant sweep from four SAE trainings per seed to one without weakening
    the protocol: the base is still fit on the training documents alone.
    """

    split: Split
    sae: object
    decoder: np.ndarray
    feats_train: np.ndarray
    kmeans: object
    pseudo_group_train: np.ndarray
    scores: np.ndarray
    sae_info: dict = field(default_factory=dict)
    #: K actually used for the pseudo-grouping, and the silhouette per
    #: candidate when it was chosen label-free (empty when K was given).
    pseudo_groups: int = 2
    silhouette: dict = field(default_factory=dict)


def fit_base(
    embeddings: np.ndarray,
    split: Split,
    *,
    expansion_factor: int = 8,
    k: int = 32,
    n_epochs: int = 200,
    pseudo_groups="auto",
    device: str | None = None,
    verbose: bool = False,
) -> TrainedBase:
    """Steps (1) and (2) -- decompose and score -- on the training documents.

    `pseudo_groups` is ``"auto"`` (K chosen by silhouette on the training
    documents, see `choose_pseudo_groups`) or an int for an explicit sweep.
    """
    X_tr = np.asarray(embeddings, dtype=np.float32)[split.train_docs]
    sae, info = train_sae(X_tr, expansion_factor=expansion_factor, k=k,
                          n_epochs=n_epochs, seed=split.seed, verbose=verbose,
                          device=device)
    feats_tr = encode_features(sae, X_tr)
    decoder = sae.W_dec.detach().cpu().numpy()
    K, sil = _resolve_pseudo_groups(pseudo_groups, X_tr, split.seed)
    km = KMeans(n_clusters=K, random_state=split.seed, n_init="auto").fit(X_tr)
    scores, _, _ = score_confounder_features(
        feats_tr, X_tr, n_pseudo_groups=K, seed=split.seed,
        pseudo_group=km.labels_)
    if verbose:
        print(f"[base] pseudo-groups K={K}" + (f" (silhouette {sil})" if sil else ""))
    return TrainedBase(
        split=split, sae=sae, decoder=decoder, feats_train=feats_tr, kmeans=km,
        pseudo_group_train=km.labels_, scores=scores,
        sae_info={kk: v for kk, v in info.items() if kk != "loss_history"},
        pseudo_groups=K, silhouette=sil)


def rescore_base(base: TrainedBase, embeddings: np.ndarray,
                 pseudo_groups: int) -> TrainedBase:
    """Re-run step (2) at an explicit `pseudo_groups`, reusing the SAE.

    The dictionary does not depend on K -- only the k-means pseudo-grouping and
    the mutual-information scores against it do. A K sweep that calls `fit_base`
    per K therefore retrains the same SAE once per K for nothing; this refits
    just the grouping and the scores, on the training documents as always.
    """
    X_tr = np.asarray(embeddings, dtype=np.float32)[base.split.train_docs]
    K = int(pseudo_groups)
    km = KMeans(n_clusters=K, random_state=base.split.seed, n_init="auto").fit(X_tr)
    scores, _, _ = score_confounder_features(
        base.feats_train, X_tr, n_pseudo_groups=K,
        seed=base.split.seed, pseudo_group=km.labels_)
    return TrainedBase(split=base.split, sae=base.sae, decoder=base.decoder,
                       feats_train=base.feats_train, kmeans=km,
                       pseudo_group_train=km.labels_, scores=scores,
                       sae_info=base.sae_info, pseudo_groups=K,
                       silhouette=base.silhouette)


@dataclass
class FittedPipeline:
    """Everything learned from the training documents, frozen and ready to apply."""

    split: Split
    sae: object = None
    decoder: np.ndarray = None
    feats_train: np.ndarray = None
    scores: np.ndarray = None
    selected: np.ndarray = None
    selection: str = ""
    kmeans: object = None
    greedy_order: list | None = None                 # greedy's full add order
    erasers: dict = field(default_factory=dict)      # name -> apply(X) -> X
    concepts: dict = field(default_factory=dict)     # name -> concept_fn(X) or None
    sae_info: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)


def fit_pipeline(
    embeddings: np.ndarray,
    source_labels,
    split: Split,
    *,
    expansion_factor: int = 8,
    k: int = 32,
    n_epochs: int = 200,
    pseudo_groups="auto",
    erase: str = "directions_leace",
    selection: str = "auto",
    top_features: int = 32,
    greedy_lambda: float = 1.0,
    greedy_budget: int = 64,
    greedy_pool: int = 80,
    recall_k: int = 10,
    kmeans_grid=(),
    kmeans_auto: bool = False,
    pca_grid=(),
    random_control: bool = True,
    device: str | None = None,
    verbose: bool = False,
    base: "TrainedBase | None" = None,
) -> FittedPipeline:
    """Steps (1)-(4) of the method, fit on `split.train_docs` only.

    Returns a `FittedPipeline` whose `erasers` are frozen affine maps: applying
    one to unseen documents involves no refitting, which is what makes the
    document-level holdout meaningful.

    `selection` is ``"greedy"``, ``"topn"`` or ``"auto"`` (greedy when the split
    has training pairs, top-N otherwise -- unpaired corpora have no pairing term
    in the greedy objective, so they take top-N as the paper states).

    `pseudo_groups` is ``"auto"`` (K chosen label-free by silhouette) or an int.
    `kmeans_auto` adds the ``kmeans_leace_auto`` baseline: LEACE on the one-hot
    pseudo-grouping G itself, i.e. the same label-free grouping the SAE latents
    are scored against, with no dictionary in between. If it matches the SAE
    condition, the SAE is justified only by interpretability.

    Pass `base` (from `fit_base`) to reuse an SAE already trained for this split.
    """
    X = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(source_labels)
    X_tr, y_tr = X[split.train_docs], y[split.train_docs]

    # (1) Decompose and (2) score -- both on the training documents only.
    if base is None:
        base = fit_base(X, split, expansion_factor=expansion_factor, k=k,
                        n_epochs=n_epochs, pseudo_groups=pseudo_groups,
                        device=device, verbose=verbose)
    sae, decoder, feats_tr = base.sae, base.decoder, base.feats_train
    km, g_tr, scores, info = base.kmeans, base.pseudo_group_train, base.scores, base.sae_info
    K = int(base.pseudo_groups)

    # (3) Select -- greedy on the training pairs, or MI top-N.
    use_greedy = selection == "greedy" or (selection == "auto" and split.train_pairs)
    if selection == "greedy" and not split.train_pairs:
        raise ValueError("selection='greedy' needs pairs; use 'topn' on unpaired corpora")
    if use_greedy:
        order, best_len = select_confounder_features_greedy(
            X_tr, feats_tr, split.train_pairs, scores,
            budget=greedy_budget, candidate_pool=greedy_pool, lam=greedy_lambda,
            pseudo_groups=K, recall_k=recall_k, erase=erase,
            decoder=decoder, seed=split.seed, verbose=verbose, pseudo_group=g_tr,
        )
        selected, sel_name = np.array(order[:best_len]), "greedy"
        greedy_order = [int(j) for j in order]
    else:
        selected = select_confounder_features(scores, top_n=top_features)
        sel_name, greedy_order = "topn", None

    # (4) Erase -- every eraser fit on the training documents, then frozen.
    erasers: dict = {"raw": (lambda X: np.asarray(X, np.float32))}
    concepts: dict = {"raw": None}
    classes = np.unique(y)

    erasers["metadata_leace"] = _fit_leace(X_tr, _one_hot(y_tr, classes), standardise=False)
    concepts["metadata_leace"] = None    # its concept is the label, not a function of X

    D_sel = decoder[selected]
    if erase == "directions_leace":
        erasers["sae_leace"] = _fit_leace(X_tr, X_tr @ D_sel.T)
    elif erase == "directions_orth":
        erasers["sae_leace"] = _fit_orth(D_sel)
    elif erase == "activations":
        erasers["sae_leace"] = _fit_leace(X_tr, feats_tr[:, selected])
    else:
        raise ValueError(f"unknown erase mechanism {erase!r}")
    concepts["sae_leace"] = lambda X, _D=D_sel: np.asarray(X, np.float32) @ _D.T

    for kk in kmeans_grid:
        km_b = KMeans(n_clusters=kk, random_state=split.seed, n_init="auto").fit(X_tr)
        erasers[f"kmeans_leace_k{kk}"] = _fit_leace(
            X_tr, _one_hot(km_b.labels_, np.arange(kk)), standardise=False)
        concepts[f"kmeans_leace_k{kk}"] = (
            lambda X, _m=km_b, _k=kk: _one_hot(_m.predict(np.asarray(X, np.float32)),
                                               np.arange(_k)))
    if kmeans_auto:
        # The pseudo-grouping G as a LEACE pseudo-label, no SAE anywhere.
        erasers["kmeans_leace_auto"] = _fit_leace(
            X_tr, _one_hot(g_tr, np.arange(K)), standardise=False)
        concepts["kmeans_leace_auto"] = (
            lambda X, _m=km, _k=K: _one_hot(_m.predict(np.asarray(X, np.float32)),
                                            np.arange(_k)))

    # The fixed grid plus one arm matched to |S|. Without the matched arm the
    # SAE erases 14-32 directions while PCA erases at most 8, so the comparison
    # would confound the selection signal with the size of the removed subspace
    # -- exactly the confound the size-matched top-N arm exists to remove. The
    # matched arm always gets its own ``_matched`` name: it is a distinct
    # condition that tables and plots look up by name, and it must not vanish
    # on the seeds where |S| happens to coincide with a grid value.
    for r in pca_grid:
        r = int(min(r, X_tr.shape[1] - 1))
        apply_fn, B = _fit_pca_removal(X_tr, r)
        erasers[f"pca_top{r}"] = apply_fn
        concepts[f"pca_top{r}"] = lambda X, _B=B: np.asarray(X, np.float32) @ _B.T
    pca_matched_r = None
    if pca_grid:
        # A fixed name, because |S| (and so r) differs between seeds and folds
        # and the arm has to line up across fits when results are pooled.
        pca_matched_r = int(min(len(selected), X_tr.shape[1] - 1))
        apply_fn, B = _fit_pca_removal(X_tr, pca_matched_r)
        erasers["pca_matched"] = apply_fn
        concepts["pca_matched"] = lambda X, _B=B: np.asarray(X, np.float32) @ _B.T

    if random_control:
        freq = (feats_tr > 0).mean(axis=0)
        alive = np.where(freq >= 0.02)[0]
        m = min(len(selected), len(alive))
        rand_ids = np.random.default_rng(split.seed).choice(alive, size=m, replace=False)
        D_rand = decoder[rand_ids]
        erasers["random_latents"] = _fit_leace(X_tr, X_tr @ D_rand.T)
        concepts["random_latents"] = lambda X, _D=D_rand: np.asarray(X, np.float32) @ _D.T

    return FittedPipeline(
        split=split, sae=sae, decoder=decoder, feats_train=feats_tr, scores=scores,
        selected=selected, selection=sel_name, kmeans=km,
        greedy_order=greedy_order, erasers=erasers,
        concepts=concepts,
        sae_info=dict(info),
        config={"expansion_factor": expansion_factor, "k": k, "n_epochs": n_epochs,
                "pseudo_groups": pseudo_groups, "pseudo_groups_k": K,
                "pseudo_group_silhouette": {str(a): b for a, b in base.silhouette.items()},
                "erase": erase,
                "selection": sel_name, "top_features": top_features,
                "greedy_lambda": greedy_lambda, "greedy_budget": greedy_budget,
                "greedy_pool": greedy_pool, "recall_k": recall_k,
                "dict_size": int(decoder.shape[0]),
                "pca_matched_r": pca_matched_r},
    )


# --------------------------------------------------------------------------- #
# Label-free semantic preservation, for corpora with no pairs
# --------------------------------------------------------------------------- #
def within_source_neighbor_retention(
    raw: np.ndarray, erased: np.ndarray, source_labels, k: int = 10,
    cap: int = 3000, seed: int = 0, queries=None,
) -> float:
    """How much of the *within-source* neighbourhood structure survives erasure.

    Unpaired corpora have no retrieval target, so leakage and confounder purity
    are the only metrics -- and both are trivially minimised by destroying the
    embedding. This is the missing quality axis: for each query document we
    compare its top-`k` nearest neighbours **among documents of the same source**
    before and after erasure, and report the mean Jaccard overlap. Restricting to
    one source at a time means removing the *between*-source axis is not
    penalised; only damage to the semantic ordering a practitioner would
    actually use is.

    `queries` restricts which documents are scored (the held-out ones under the
    document protocol) while the neighbour pool stays the whole of `raw` /
    `erased` -- the corpus erased by one eraser, as in the retrieval index. That
    keeps the candidate pool large even for a small class: with neighbours drawn
    from the held-out half alone, a class with barely more than `k` held-out
    documents makes nearly every document a neighbour of every other and the
    metric reads close to 1 by construction. Classes with `k + 1` or fewer
    documents in the pool are skipped outright.

    1.0 = the within-source ranking is untouched (raw scores exactly 1.0 by
    construction); 0.0 = it was destroyed. Source labels are used here for
    evaluation only, never for fitting.
    """
    raw, erased = np.asarray(raw, np.float32), np.asarray(erased, np.float32)
    y = np.asarray(source_labels)
    rng = np.random.default_rng(seed)
    is_query = np.zeros(len(y), dtype=bool)
    is_query[np.arange(len(y)) if queries is None else np.asarray(queries)] = True
    out = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) <= k + 1:
            continue
        if len(idx) > cap:
            # Keep the queries, fill the pool with random other documents.
            q, rest = idx[is_query[idx]], idx[~is_query[idx]]
            if len(q) > cap:
                q = rng.choice(q, size=cap, replace=False)
            fill = rng.choice(rest, size=min(cap - len(q), len(rest)), replace=False)
            idx = np.concatenate([q, fill])
        A, B = _cosine(raw[idx], raw[idx]), _cosine(erased[idx], erased[idx])
        np.fill_diagonal(A, -np.inf)
        np.fill_diagonal(B, -np.inf)
        ta = np.argpartition(-A, k, axis=1)[:, :k]
        tb = np.argpartition(-B, k, axis=1)[:, :k]
        for r in np.where(is_query[idx])[0]:
            sa, sb = set(ta[r].tolist()), set(tb[r].tolist())
            out.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(out)) if out else float("nan")


# --------------------------------------------------------------------------- #
# Evaluation on held-out documents
# --------------------------------------------------------------------------- #
def evaluate(
    fitted: FittedPipeline,
    embeddings: np.ndarray,
    source_labels,
    *,
    recall_k: int = 10,
    nonlinear_probes=("mlp", "gb"),
    neighbor_k: int = 10,
    crossfit: bool = True,
    conditions=None,
    verbose: bool = False,
    allow_descriptive: bool = False,
    retrieval_index: str = "full",
) -> dict:
    """Apply every frozen eraser to the evaluation documents and score them.

    Under ``protocol="document"`` the evaluation documents were never seen by any
    fitted component, so ``leakage_auc`` is already artifact-free and is the
    primary number; ``crossfit_leakage_auc`` is skipped (it would refit the
    eraser inside the evaluation half, a different estimator). Under
    ``protocol="pair"`` the eraser saw everything, so the flip-adjusted and
    cross-fitted variants are reported alongside as the paper describes.
    """
    split = fitted.split
    if split.protocol == "none" and not allow_descriptive:
        raise ValueError(
            "this pipeline was fit with protocol='none' (no holdout), so every "
            "metric it could produce is a training-set number. Descriptive "
            "analyses should read the fit itself (see analysis.py); pass "
            "allow_descriptive=True only if you intend an explicitly in-sample "
            "diagnostic and will label it as one.")
    X = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(source_labels)
    X_te, y_te = X[split.test_docs], y[split.test_docs]
    seed = split.seed
    ck = max(2, len(np.unique(y)))
    names = list(conditions) if conditions else list(fitted.erasers)
    groups = probe_groups(split)          # pairs stay together in the probe folds

    if retrieval_index not in ("full", "heldout"):
        raise ValueError(f"retrieval_index must be 'full' or 'heldout', "
                         f"got {retrieval_index!r}")
    out = {"metrics": {}, "hits": {}, "split": split.summary(),
           "retrieval_index": retrieval_index,
           "n_selected": int(len(fitted.selected)),
           "selected": [int(j) for j in fitted.selected],
           "selection": fitted.selection, "seed": int(seed)}

    for name in names:
        # One eraser applied to the whole corpus: the held-out rows are scored,
        # the rest serve as the retrieval index and the neighbour pool.
        E_full = fitted.erasers[name](X)
        E = E_full[split.test_docs]
        lin = probe_leakage(E, y_te, seed=seed, probe="linear", groups=groups)
        m = {"leakage_auc": lin["auc"], "leakage_auc_flip": lin["auc_flip"]}
        for probe in nonlinear_probes:
            nl = probe_leakage(E, y_te, seed=seed, probe=probe, groups=groups)
            m[f"nonlinear_leakage_auc_{probe}"] = nl["auc"]
            m[f"nonlinear_leakage_auc_{probe}_flip"] = nl["auc_flip"]
        m["cluster_purity"] = cluster_purity(E, y_te, ck, seed)
        m[f"neighbor_retention@{neighbor_k}"] = within_source_neighbor_retention(
            X, E_full, y, k=neighbor_k, seed=seed, queries=split.test_docs)

        if split.protocol == "pair" and crossfit:
            Z = None
            if name == "metadata_leace":
                Z = _one_hot(y_te, np.unique(y))
            elif fitted.concepts.get(name) is not None:
                Z = fitted.concepts[name](X_te)
            cf = crossfit_leakage_auc(X_te, y_te, Z=Z, seed=seed, groups=groups)
            m["crossfit_leakage_auc"] = cf["auc"]
            m["crossfit_leakage_auc_flip"] = cf["auc_flip"]

        if split.has_pairs:
            # Held-out pairs as queries; the index is the whole erased corpus by
            # default, so the candidate pool does not change with the split.
            if retrieval_index == "full" and split.test_pairs_global:
                index, qpairs = E_full, split.test_pairs_global
            else:
                index, qpairs = E, split.test_pairs
            m.update(paired_retrieval_recall(index, qpairs, ks=(1, recall_k)))
            m["retrieval_pool"] = int(len(index))
            if split.gold_test is not None:
                m["gold_sim_spearman"] = gold_similarity_spearman(
                    index, qpairs, split.gold_test)
            out["hits"][name] = pair_hit_vector(index, qpairs, k=recall_k)

        out["metrics"][name] = m
        if verbose:
            print(f"  [{name:>18s}] leak={m['leakage_auc']:.4f}  "
                  f"purity={m['cluster_purity']:.4f}  "
                  f"recall@{recall_k}={m.get(f'recall@{recall_k}', float('nan')):.4f}  "
                  f"nbr={m[f'neighbor_retention@{neighbor_k}']:.4f}")
    return out


def probe_groups(split: Split) -> np.ndarray:
    """Probe-fold group ids aligned with `split.test_docs`: the documents of a
    linked group -- a pair, or a case's triple in a three-class corpus -- share an
    id, every other document has its own. Under the transductive protocols
    every document is scored, so every pair (train and test alike) is a group;
    under the document protocol only the held-out pairs are present. A group's
    id is ``n + j`` for the last pair j it contains -- with disjoint pairs, pair
    j's id is ``n + j``, the ids the fold assignment has always used."""
    n = len(split.test_docs)
    groups = np.arange(n)
    if split.protocol == "document":
        pairs = split.test_pairs or []
    else:
        pairs = list(split.train_pairs or []) + list(split.test_pairs or [])
    if not len(pairs):
        return groups
    of_pair = link_groups(pairs)[0]
    last = {int(g): j for j, g in enumerate(of_pair)}
    for j, (a, b) in enumerate(pairs):
        groups[a] = groups[b] = n + last[int(of_pair[j])]
    return groups


def _pair_cosines(index: np.ndarray, pairs) -> np.ndarray:
    idx = np.asarray(pairs)
    a = index[idx[:, 0]] / (np.linalg.norm(index[idx[:, 0]], axis=1, keepdims=True) + 1e-8)
    b = index[idx[:, 1]] / (np.linalg.norm(index[idx[:, 1]], axis=1, keepdims=True) + 1e-8)
    return (a * b).sum(axis=1)


def evaluate_folds(
    fitteds: list,
    embeddings: np.ndarray,
    source_labels,
    *,
    recall_k: int = 10,
    nonlinear_probes=("mlp", "gb"),
    neighbor_k: int = 10,
    conditions=None,
    verbose: bool = False,
    retrieval_index: str = "full",
) -> dict:
    """Score one seed's K folds as a single result (see the module docstring).

    `fitteds` holds one `FittedPipeline` per fold of the same seed, all under
    ``protocol="document"`` with disjoint held-out sets. Per fold, one eraser is
    applied to the whole corpus; leakage, purity and retention are computed for
    that fold's held-out documents and averaged over folds, while the per-pair
    retrieval hits (and the pair cosines behind the gold correlation) are
    concatenated so every pair is a query once. Returns the same shape as
    `evaluate`, plus the per-fold values under ``per_fold``.
    """
    if not fitteds:
        raise ValueError("evaluate_folds needs at least one fitted fold")
    splits = [f.split for f in fitteds]
    if any(sp.protocol != "document" for sp in splits):
        raise ValueError("fold-wise evaluation is only defined for protocol='document'")
    if retrieval_index not in ("full", "heldout"):
        raise ValueError(f"retrieval_index must be 'full' or 'heldout', "
                         f"got {retrieval_index!r}")
    X = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(source_labels)
    scored = np.concatenate([sp.test_docs for sp in splits])
    if len(np.unique(scored)) != len(scored):
        raise ValueError("folds must hold out disjoint documents")
    seed = int(splits[0].seed)
    ck = max(2, len(np.unique(y)))
    names = list(conditions) if conditions else list(fitteds[0].erasers)
    has_pairs = any(sp.has_pairs for sp in splits)
    gold_all = (np.concatenate([sp.gold_test for sp in splits])
                if has_pairs and all(sp.gold_test is not None for sp in splits)
                else None)
    n_sel = [int(len(f.selected)) for f in fitteds]
    n_te_pairs = int(sum(len(sp.test_pairs or []) for sp in splits))
    out = {"metrics": {}, "hits": {}, "per_fold": {},
           "split": {
               "protocol": "document", "seed": seed,
               "holdout": float(splits[0].holdout),
               "scheme": "kfold" if len(splits) > 1 else "holdout",
               "n_folds": len(splits), "foldwise": True, "fold": None,
               "n_train_docs": int(round(np.mean([len(sp.train_docs) for sp in splits]))),
               "n_test_docs": int(len(scored)),
               "n_test_docs_per_fold": int(round(np.mean([len(sp.test_docs) for sp in splits]))),
               "n_train_pairs": int(round(np.mean([len(sp.train_pairs or []) for sp in splits]))),
               "n_test_pairs": n_te_pairs,
               "n_test_unpaired": int(len(scored) - sum(_n_linked(sp.test_pairs)
                                                        for sp in splits)),
               "documents_disjoint": True, "performance_valid": True},
           "retrieval_index": retrieval_index,
           "n_selected": int(round(np.mean(n_sel))), "n_selected_per_fold": n_sel,
           "selected": [int(j) for j in fitteds[0].selected],
           "selected_per_fold": [[int(j) for j in f.selected] for f in fitteds],
           "selection": fitteds[0].selection, "seed": seed}

    fold_keys = (["leakage_auc", "leakage_auc_flip"]
                 + [f"nonlinear_leakage_auc_{p}{sfx}" for p in nonlinear_probes
                    for sfx in ("", "_flip")]
                 + ["cluster_purity", f"neighbor_retention@{neighbor_k}"])
    for name in names:
        per = {kk: [] for kk in fold_keys}
        hits_k, hits_1, cos, pools = [], [], [], []
        for f in fitteds:
            sp = f.split
            E_full = f.erasers[name](X)            # one eraser, whole corpus
            E_te, y_te = E_full[sp.test_docs], y[sp.test_docs]
            groups = probe_groups(sp)
            lin = probe_leakage(E_te, y_te, seed=seed, probe="linear", groups=groups)
            per["leakage_auc"].append(lin["auc"])
            per["leakage_auc_flip"].append(lin["auc_flip"])
            for probe in nonlinear_probes:
                nl = probe_leakage(E_te, y_te, seed=seed, probe=probe, groups=groups)
                per[f"nonlinear_leakage_auc_{probe}"].append(nl["auc"])
                per[f"nonlinear_leakage_auc_{probe}_flip"].append(nl["auc_flip"])
            per["cluster_purity"].append(cluster_purity(E_te, y_te, ck, seed))
            per[f"neighbor_retention@{neighbor_k}"].append(
                within_source_neighbor_retention(X, E_full, y, k=neighbor_k, seed=seed,
                                                 queries=sp.test_docs))
            if sp.has_pairs:
                if retrieval_index == "full":
                    index, qp = E_full, sp.test_pairs_global
                else:
                    index, qp = E_te, sp.test_pairs
                hits_k.append(pair_hit_vector(index, qp, k=recall_k))
                hits_1.append(pair_hit_vector(index, qp, k=1))
                cos.append(_pair_cosines(index, qp))
                pools.append(len(index))
        m = {kk: float(np.mean(v)) for kk, v in per.items()}
        if has_pairs and hits_k:
            hk = np.concatenate(hits_k)
            m["recall@1"] = float(np.concatenate(hits_1).mean())
            m[f"recall@{recall_k}"] = float(hk.mean())
            m["retrieval_pool"] = int(round(np.mean(pools)))
            if gold_all is not None:
                from scipy.stats import spearmanr
                m["gold_sim_spearman"] = float(spearmanr(np.concatenate(cos), gold_all)[0])
            out["hits"][name] = hk
        out["metrics"][name] = m
        out["per_fold"][name] = {kk: [float(x) for x in v] for kk, v in per.items()}
        if verbose:
            print(f"  [{name:>18s}] leak={m['leakage_auc']:.4f}  "
                  f"purity={m['cluster_purity']:.4f}  "
                  f"recall@{recall_k}={m.get(f'recall@{recall_k}', float('nan')):.4f}  "
                  f"nbr={m[f'neighbor_retention@{neighbor_k}']:.4f}")
    return out


def evaluate_splits(fitteds: list, embeddings: np.ndarray, source_labels, **kw) -> dict:
    """`evaluate` for a single fit, `evaluate_folds` for a seed's K folds.

    The one entry point runners use, so a corpus's split scheme decides how it
    is scored without the runner knowing. The result carries the first fit's
    SAE diagnostics and config, plus the per-fold selection sizes under K-fold.
    """
    fitteds = list(fitteds)
    if len(fitteds) == 1:
        res = evaluate(fitteds[0], embeddings, source_labels, **kw)
    else:
        kw.pop("crossfit", None)
        res = evaluate_folds(fitteds, embeddings, source_labels, **kw)
    res["sae_info"] = fitteds[0].sae_info
    res["config"] = {**fitteds[0].config,
                     "n_selected_per_fold": [int(len(f.selected)) for f in fitteds],
                     "pseudo_groups_k_per_fold": [int(f.config.get("pseudo_groups_k", 0))
                                                  for f in fitteds]}
    return res


def run_seed(
    embeddings: np.ndarray,
    source_labels,
    pairs=None,
    gold_sim=None,
    *,
    seed: int = 0,
    protocol: str = "document",
    holdout: float = 0.5,
    recall_k: int = 10,
    nonlinear_probes=("mlp", "gb"),
    neighbor_k: int = 10,
    crossfit: bool = True,
    conditions=None,
    verbose: bool = False,
    retrieval_index: str = "full",
    split: "Split | None" = None,
    splits: "list | None" = None,
    **fit_kwargs,
) -> dict:
    """One seed end to end: fit on train -> evaluate on held-out.

    Pass `splits` (a seed's folds from `make_folds`) to fit every fold and score
    them fold-wise as one result, or `split` for one prebuilt split; otherwise a
    single split is built from `protocol`/`holdout`/`seed`.
    """
    if splits is None:
        splits = [split or make_split(len(embeddings), source_labels, pairs=pairs,
                                      gold_sim=gold_sim, holdout=holdout, seed=seed,
                                      protocol=protocol)]
    fitteds = [fit_pipeline(embeddings, source_labels, sp, recall_k=recall_k,
                            verbose=verbose, **fit_kwargs) for sp in splits]
    return evaluate_splits(fitteds, embeddings, source_labels, recall_k=recall_k,
                           nonlinear_probes=nonlinear_probes, neighbor_k=neighbor_k,
                           crossfit=crossfit, conditions=conditions, verbose=verbose,
                           retrieval_index=retrieval_index)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(per_seed: list) -> dict:
    """mean / std / values per (condition, metric), respecting the fold structure.

    Folds are **not** independent replicates: with 5 folds any two share 60% of
    their training documents, so pooling seed x fold fits into one standard
    deviation treats 15 correlated numbers as 15 samples and reports an error bar
    that is too small. When the entries carry fold tags this averages within each
    seed first and takes the spread **across seeds**, which is the honest unit of
    replication; ``std_pooled`` keeps the naive across-all-fits value for
    reference, and ``n_seeds`` / ``n_folds`` say what the two are over.

    With a single split per seed the two agree and nothing changes.
    """
    conds, keys = [], []
    for s in per_seed:
        for c in s["metrics"]:
            if c not in conds:
                conds.append(c)
            for kk in s["metrics"][c]:
                if kk not in keys:
                    keys.append(kk)

    folded = any(s.get("split", {}).get("fold") is not None for s in per_seed)
    seeds = sorted({int(s["seed"]) for s in per_seed})
    n_folds = max((s["split"].get("n_folds") or 1) for s in per_seed)

    agg = {}
    for c in conds:
        agg[c] = {}
        for kk in keys:
            allv = [s["metrics"][c][kk] for s in per_seed
                    if c in s["metrics"] and kk in s["metrics"][c]]
            if not allv:
                continue
            allv = np.asarray(allv, dtype=float)
            if folded:
                # one number per seed = the mean over that seed's folds
                per = [np.mean([s["metrics"][c][kk] for s in per_seed
                                if int(s["seed"]) == sd
                                and c in s["metrics"] and kk in s["metrics"][c]])
                       for sd in seeds]
                per = np.asarray([v for v in per if np.isfinite(v)], dtype=float)
                unit = per
            else:
                unit = allv
            agg[c][kk] = {
                "mean": float(unit.mean()),
                "std": float(unit.std(ddof=1)) if len(unit) > 1 else 0.0,
                "values": unit.tolist(),
                "n_units": int(len(unit)),
                "std_pooled": (float(allv.std(ddof=1)) if len(allv) > 1 else 0.0),
                "n_fits": int(len(allv)),
            }
    if folded:
        agg["_replication"] = {"unit": "seed", "n_seeds": len(seeds),
                               "n_folds": int(n_folds),
                               "note": "std is across seeds (folds averaged within "
                                       "a seed); folds share training data and are "
                                       "not independent replicates"}
    return agg
