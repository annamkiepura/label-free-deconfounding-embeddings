"""
SAE-based confounder discovery and deconfounding.

This module implements the *extension* described in the project README: instead
of assuming we already know the confounder (e.g. "source" or "language") and
have a metadata label for it, we

  1. train a TopK SAE on document embeddings (see `sae.py`),
  2. score each SAE latent for how much it looks like a latent confounder,
     using only unsupervised signals (no metadata labels),
  3. interpret the top latents with their highest-activating documents,
  4. use the selected latents' activations as *pseudo-confounder labels* and
     erase them from the original embeddings with LEACE,
  5. evaluate retrieval / clustering / source-leakage, and compare
        raw  vs  metadata-LEACE  vs  SAE-discovered-LEACE.

The whole point of step (2) is that it never touches the true source/language
label. We do, however, *report* the correlation of each discovered latent with
the true label afterwards, purely as a sanity check / interpretability aid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from concept_erasure import LeaceEraser
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from sae import encode_features, encode_preactivations, train_sae


# --------------------------------------------------------------------------- #
# 1-2. Confounder scoring (unsupervised)
# --------------------------------------------------------------------------- #
def _binary_mutual_information(active: np.ndarray, group: np.ndarray) -> float:
    """MI (in bits) between a binary feature mask and a discrete pseudo-group."""
    n = len(active)
    mi = 0.0
    for a in (0, 1):
        pa = np.mean(active == a)
        if pa == 0:
            continue
        for g in np.unique(group):
            pg = np.mean(group == g)
            pag = np.mean((active == a) & (group == g))
            if pag > 0:
                mi += pag * np.log2(pag / (pa * pg))
    return mi


def score_confounder_features(
    feats: np.ndarray,
    embeddings: np.ndarray,
    n_pseudo_groups: int = 2,
    min_activation_freq: float = 0.02,
    max_activation_freq: float = 0.98,
    seed: int = 0,
    pseudo_group: np.ndarray | None = None,
):
    """Score every SAE latent for "confounder-likeness" without any labels.

    Intuition (from the parent paper): document-level confounders such as source
    or language dominate the *global* geometry of the embedding space, so the
    coarsest clusters of the raw embeddings are largely organised by the
    confounder rather than by meaning. A latent that "is" a confounder will
    therefore (a) fire on a large, coherent block of documents and (b) line up
    strongly with that coarse clustering.

    We build an unsupervised pseudo-grouping by running k-means (k =
    `n_pseudo_groups`) on the raw embeddings, then score each latent by the
    mutual information between "is this latent active" and the pseudo-group.
    Latents that fire almost never / almost always carry no partitioning signal
    and are filtered out by the activation-frequency band.

    Returns
    -------
    scores : np.ndarray, shape (n_latents,)
        MI-based confounder score per latent (0 for filtered latents).
    pseudo_group : np.ndarray, shape (n_docs,)
        The k-means pseudo-grouping used for scoring (handy for plots/debug).
    freq : np.ndarray, shape (n_latents,)
        Activation frequency of each latent.

    Pass `pseudo_group` to score against a grouping produced elsewhere -- e.g. a
    k-means model fit on the TRAINING documents and used to `predict` on unseen
    ones, which is what the document-level holdout protocol requires (no test
    document may influence any fitted component). When it is None the grouping is
    fit here on `embeddings`, the transductive behaviour.
    """
    if pseudo_group is None:
        km = KMeans(n_clusters=n_pseudo_groups, random_state=seed, n_init="auto")
        pseudo_group = km.fit_predict(embeddings)
    else:
        pseudo_group = np.asarray(pseudo_group)

    active = feats > 0
    freq = active.mean(axis=0)
    scores = np.zeros(feats.shape[1])
    for j in range(feats.shape[1]):
        if freq[j] < min_activation_freq or freq[j] > max_activation_freq:
            continue
        scores[j] = _binary_mutual_information(active[:, j].astype(int), pseudo_group)
    return scores, pseudo_group, freq


def score_confounder_features_variants(
    feats: np.ndarray,
    embeddings: np.ndarray,
    n_pseudo_groups: int = 2,
    min_activation_freq: float = 0.02,
    max_activation_freq: float = 0.98,
    seed: int = 0,
    pseudo_group: np.ndarray | None = None,
) -> dict:
    """Three confounder scores per latent, for the selection-score ablation.

      * ``mi``         — the paper's score: MI between binary firing and the
                         k-means pseudo-group (bits; plug-in estimator).
      * ``nmi``        — normalized MI, the uncertainty coefficient
                         ``I(F_j; G) / H(F_j)`` in [0, 1]. Removes the cap that
                         raw MI inherits from each latent's own firing entropy
                         and makes scores comparable across pseudo-group counts.
      * ``corr_ratio`` — magnitude-aware alternative: the correlation ratio
                         (eta-squared) of the *continuous* activation across
                         pseudo-groups — between-group variance / total variance.

    All three apply the same activation-frequency band (rare/ubiquitous latents
    scored 0). Returns ``{"mi", "nmi", "corr_ratio", "freq", "pseudo_group"}``.

    Pass `pseudo_group` to score against a grouping fit elsewhere -- the k-means
    model of the split's TRAINING documents, as the document-level holdout
    requires -- exactly as in `score_confounder_features`. When it is None the
    grouping is fit here on `embeddings`, the transductive behaviour.
    """
    if pseudo_group is None:
        km = KMeans(n_clusters=n_pseudo_groups, random_state=seed, n_init="auto")
        pseudo_group = km.fit_predict(embeddings)
    else:
        pseudo_group = np.asarray(pseudo_group)
    groups = np.unique(pseudo_group)
    masks = [pseudo_group == g for g in groups]
    p_g = np.array([m.mean() for m in masks])

    active = feats > 0
    freq = active.mean(axis=0)
    n_lat = feats.shape[1]
    mi = np.zeros(n_lat)
    nmi = np.zeros(n_lat)
    eta2 = np.zeros(n_lat)
    for j in range(n_lat):
        f = freq[j]
        if f < min_activation_freq or f > max_activation_freq:
            continue
        I = _binary_mutual_information(active[:, j].astype(int), pseudo_group)
        mi[j] = I
        H = -(f * np.log2(f) + (1 - f) * np.log2(1 - f))
        nmi[j] = I / H if H > 0 else 0.0
        z = feats[:, j]
        mu = z.mean()
        total = ((z - mu) ** 2).mean()
        if total > 0:
            between = sum(p * (z[m].mean() - mu) ** 2 for p, m in zip(p_g, masks))
            eta2[j] = between / total
    return {"mi": mi, "nmi": nmi, "corr_ratio": eta2,
            "freq": freq, "pseudo_group": pseudo_group}


def select_confounder_features(
    scores: np.ndarray,
    top_n: int | None = None,
    score_quantile: float | None = 0.99,
):
    """Pick the confounder latents: either a fixed top-N or a score quantile."""
    order = np.argsort(scores)[::-1]
    if top_n is not None:
        return order[:top_n]
    thresh = np.quantile(scores[scores > 0], score_quantile) if np.any(scores > 0) else 0
    return np.array([j for j in order if scores[j] >= thresh and scores[j] > 0])


# --------------------------------------------------------------------------- #
# 3-4. Interpretation via high- and low-activation examples
# --------------------------------------------------------------------------- #
def _example(feats, texts, source_labels, i, j, snippet_chars):
    ex = {
        "doc_index": int(i),
        "activation": float(feats[i, j]),
        "snippet": texts[i][:snippet_chars].replace("\n", " "),
    }
    if source_labels is not None:
        ex["source"] = source_labels[i]
    return ex


def interpret_features(
    feats: np.ndarray,
    texts: list[str],
    feature_ids,
    top_examples: int = 6,
    snippet_chars: int = 160,
    source_labels: np.ndarray | None = None,
    pre: np.ndarray | None = None,
):
    """Return, for each selected latent, its highest- AND lowest-activating docs.

    The top examples characterise what the latent *fires on*. For the bottom
    examples, pass `pre` (the dense pre-activation matrix from
    `sae.encode_preactivations`): with TopK sparsity the post-activation matrix
    is exactly 0 on ~97% of entries, so ranking by post-activations makes the
    "bottom" an arbitrary tie-break among zeros. Ranking by *pre*-activations
    instead surfaces the documents that most strongly anti-activate the latent —
    the genuine other side of the confounder. Seeing, say, all-tweets at the top
    and all-bills at the bottom is exactly the qualitative signature of a
    confounder latent. (Without `pre` we fall back to the old post-activation
    ranking, with the tie-break caveat.)

    `source_labels` (if available) is *only* used to annotate the printout so a
    human can eyeball whether the latent tracks a confounder. It is never used
    for selection.

    Returns
    -------
    report : dict[int, dict]
        ``{latent_id: {"top": [examples...], "bottom": [examples...]}}``, each
        example sorted from strongest to weakest activation.
    """
    report = {}
    for j in feature_ids:
        top_idx = np.argsort(feats[:, j])[::-1][:top_examples]
        bottom_rank = pre[:, j] if pre is not None else feats[:, j]
        bottom_idx = np.argsort(bottom_rank)[:top_examples]
        report[int(j)] = {
            "top": [_example(feats, texts, source_labels, i, j, snippet_chars) for i in top_idx],
            "bottom": [_example(feats, texts, source_labels, i, j, snippet_chars) for i in bottom_idx],
        }
        if pre is not None:
            for ex, i in zip(report[int(j)]["bottom"], bottom_idx):
                ex["preactivation"] = float(pre[i, j])
    return report


def print_interpretation(report: dict) -> None:
    def _rows(examples):
        for ex in examples:
            src = f" [{ex['source']}]" if "source" in ex else ""
            print(f"  act={ex['activation']:7.3f}{src}  {ex['snippet']}")

    for j, groups in report.items():
        print(f"\n=== SAE latent #{j} — top activations (fires on) ===")
        _rows(groups["top"])
        print(f"--- SAE latent #{j} — bottom activations (silent on) ---")
        _rows(groups["bottom"])


# --------------------------------------------------------------------------- #
# 5-6. Pseudo-label LEACE
# --------------------------------------------------------------------------- #
def leace_erase_with_features(
    embeddings: np.ndarray,
    feats: np.ndarray,
    feature_ids,
) -> np.ndarray:
    """Erase the selected SAE latents from the embeddings with LEACE.

    The selected latents' (standardised) activations form a multi-dimensional
    pseudo-confounder Z. `LeaceEraser.fit(X, Z)` then removes from X exactly the
    subspace that is linearly predictable from Z — i.e. everything the SAE-
    discovered confounder features can explain — while changing X as little as
    possible (the LEACE optimality guarantee from the parent paper).
    """
    X = torch.from_numpy(np.asarray(embeddings)).float()
    Z = feats[:, np.asarray(list(feature_ids))]
    Z = StandardScaler().fit_transform(Z)
    Z = torch.from_numpy(Z).float()
    eraser = LeaceEraser.fit(X, Z)
    return eraser(X).numpy()


def leace_erase_with_directions(
    embeddings: np.ndarray,
    decoder: np.ndarray,
    feature_ids,
    mode: str = "orthogonal",
) -> np.ndarray:
    """Erase the SAE-discovered confounder as a **subspace of decoder directions**.

    `leace_erase_with_features` treats the selected latents' *sparse activations*
    as the concept to remove. Because those activations are gated (zero on most
    documents), LEACE only "sees" the confounder where the latent fires, so it can
    leave a lot of the underlying geometric axis in place. Here we instead remove
    the actual **decoder directions** `W_dec[j]` of the selected latents — the
    dense axes along which those features write into the embedding — which should
    cover more of the confounder's linear subspace.

    Two modes:
      * ``"orthogonal"`` — hard orthogonal projection onto the complement of the
        span of the selected decoder rows (removes the whole subspace; maximally
        aggressive, may take some content with it).
      * ``"leace"`` — the LEACE-oblique analogue: use each embedding's projection
        onto the decoder directions as the concept Z, so LEACE removes the same
        subspace but with its minimal-distortion (whitened) guarantee. This is the
        apples-to-apples counterpart of `leace_erase_with_features`, swapping
        sparse activations for dense direction-projections.
    """
    X = np.asarray(embeddings, dtype=np.float32)
    ids = np.asarray(list(feature_ids))
    D = np.asarray(decoder, dtype=np.float32)[ids]  # (m, d) unit-norm decoder rows

    if mode == "orthogonal":
        # Orthonormal basis B (r, d) of span(D), then project it out of X.
        _, s, Vt = np.linalg.svd(D, full_matrices=False)
        r = int((s > 1e-6).sum())
        B = Vt[:r]
        return (X - (X @ B.T) @ B).astype(np.float32)

    if mode == "leace":
        Xt = torch.from_numpy(X).float()
        Z = X @ D.T  # (n, m): coordinate of each doc along each decoder direction
        Z = StandardScaler().fit_transform(Z)
        Z = torch.from_numpy(Z).float()
        eraser = LeaceEraser.fit(Xt, Z)
        return eraser(Xt).numpy()

    raise ValueError(f"unknown mode {mode!r} (use 'orthogonal' or 'leace')")


def leace_erase_with_labels(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Metadata baseline: erase the true source/language label with LEACE.

    For >2 classes the label is **one-hot encoded** before fitting. This matters:
    LEACE erases the subspace linearly predictable from Z, so a nominal variable
    with C categories needs its full (C-1)-dim one-hot representation. Passing raw
    ordinal class indices (0,1,2,...) would only erase the single spurious
    direction correlated with that arbitrary ordering and badly under-erase the
    confounder. (Binary labels are already 1-D, so they need no one-hot.)
    """
    X = torch.from_numpy(np.asarray(embeddings)).float()
    classes, y = np.unique(labels, return_inverse=True)
    if len(classes) > 2:
        Z = np.eye(len(classes), dtype=np.float32)[y]  # one-hot, shape (n, C)
        Z = torch.from_numpy(Z)
    else:
        Z = torch.from_numpy(y).long()
    eraser = LeaceEraser.fit(X, Z)
    return eraser(X).numpy()


#: Sentinel for "this condition has no concept matrix, skip the cross-fit"
#: (``None`` already means "probe the unerased embeddings", so it cannot serve).
_SKIP = object()


def one_hot_labels(labels: np.ndarray) -> np.ndarray:
    """(n, n_classes) one-hot matrix — the concept matrix of the label oracle."""
    classes, y = np.unique(np.asarray(labels), return_inverse=True)
    return np.eye(len(classes), dtype=np.float32)[y]


def concept_matrix(embeddings, feats, decoder, selected, mechanism="directions_leace"):
    """The concept matrix ``Z`` describing *what* an erasure mechanism removes.

    `crossfit_leakage_auc` needs this: it refits the eraser inside every CV
    training fold, so it must be told which concept to erase rather than being
    handed already-erased embeddings. The mapping mirrors `erase_embeddings`:

      * ``activations``      -> the selected latents' sparse codes;
      * ``directions_leace`` -> the projections of the embeddings on the
        selected decoder rows (the subspace LEACE is asked to neutralise);
      * ``directions_orth``  -> the same subspace (the cross-fitted reference
        for an orthogonal projection is the LEACE fit on that subspace).

    Returns ``None`` for an empty selection (nothing is erased).
    """
    selected = np.asarray(selected)
    if selected.size == 0:
        return None
    if mechanism == "activations":
        return np.asarray(feats, dtype=np.float32)[:, selected]
    if mechanism in ("directions_leace", "directions_orth"):
        D = np.asarray(decoder, dtype=np.float32)[selected]
        return np.asarray(embeddings, dtype=np.float32) @ D.T
    raise ValueError(f"unknown mechanism {mechanism!r}")


def erase_embeddings(embeddings, feats, decoder, feature_ids, mechanism="directions_leace"):
    """Dispatch the SAE-based erasure by mechanism (one entry point for all of it).

    * ``"activations"``      — `leace_erase_with_features` (sparse activations as Z).
    * ``"directions_leace"`` — `leace_erase_with_directions(mode="leace")`
      (LEACE on the decoder-direction projections — the best mechanism in our
      experiments, so the default here).
    * ``"directions_orth"``  — `leace_erase_with_directions(mode="orthogonal")`.

    Direction mechanisms need `decoder`; the activation mechanism needs `feats`.
    """
    if mechanism == "activations":
        return leace_erase_with_features(embeddings, feats, feature_ids)
    if mechanism == "directions_leace":
        return leace_erase_with_directions(embeddings, decoder, feature_ids, mode="leace")
    if mechanism == "directions_orth":
        return leace_erase_with_directions(embeddings, decoder, feature_ids, mode="orthogonal")
    raise ValueError(
        f"unknown erase mechanism {mechanism!r} "
        "(use 'activations', 'directions_leace', or 'directions_orth')"
    )


# --------------------------------------------------------------------------- #
# 7. Evaluation
# --------------------------------------------------------------------------- #
def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T


def _auc_raw_and_flip(y: np.ndarray, proba: np.ndarray, n_classes: int) -> tuple[float, float]:
    """(raw AUC, flip-adjusted AUC) for binary or multi-class (macro-OVR) probes.

    Flip-adjustment reports ``max(auc, 1 - auc)`` (per class for multi-class,
    then macro-averaged). Rationale: an eraser fit on the *full* sample forces
    full-sample covariance with the label to zero, so a cross-validated probe's
    held-out scores become systematically *anti*-correlated with the label —
    below-chance AUC (e.g. the metadata oracle's 0.077 on SemEval). An adversary
    who flips the score recovers the concept with AUC 1 - 0.077 = 0.923, so raw
    below-chance AUC must never be read as "less leakage than chance". The
    flip-adjusted value is the honest headline number; the raw value is kept for
    reference. (The fully clean alternative is `crossfit_leakage_auc`, which
    refits the eraser inside each CV fold so the artifact never arises.)
    """
    if n_classes == 2:
        a = float(roc_auc_score(y, proba if proba.ndim == 1 else proba[:, 1]))
        return a, float(max(a, 1.0 - a))
    per_class = [float(roc_auc_score((y == c).astype(int), proba[:, c])) for c in range(n_classes)]
    raw = float(np.mean(per_class))
    flip = float(np.mean([max(a, 1.0 - a) for a in per_class]))
    return raw, flip


def _probe_cv(seed: int, n_splits: int, groups):
    """Seeded, shuffled probe folds; group-aware when `groups` is given."""
    if groups is None:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _cv_probe_proba(clf, embeddings, y, seed: int = 0, n_splits: int = 5,
                    groups=None) -> np.ndarray:
    """Cross-validated predicted probabilities with *seeded, shuffled* folds
    (so multi-seed runs actually vary the probe folds, as the error bars claim).

    `groups` keeps the members of a matched pair in the same probe fold. This
    matters on paired corpora: after a good erasure a document and its
    counterpart are near-identical vectors carrying *opposite* labels, so a
    probe that saw one of them in training is pushed to mislabel the other, and
    a randomly folded probe reads systematically below chance -- which
    flip-adjustment would then turn into spurious leakage. An adversary's new
    documents do not come with their twins in its training set, so the probe's
    folds must not either.
    """
    cv = _probe_cv(seed, n_splits, groups)
    kw = {"groups": np.asarray(groups)} if groups is not None else {}
    return cross_val_predict(clf, embeddings, y, cv=cv, method="predict_proba", **kw)


def probe_leakage(
    embeddings: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
    probe: str = "linear",
    hidden=(256, 128),
    max_iter: int = 300,
    groups=None,
) -> dict:
    """Cross-validated leakage of `labels` from `embeddings` for one probe.

    Returns ``{"auc": raw, "auc_flip": flip-adjusted}`` — see `_auc_raw_and_flip`
    for why both exist. `probe` is 'linear' or any of the nonlinear probes from
    `_make_nonlinear_probe` ('mlp', 'gb', 'rbf'). `groups` (one id per row;
    both members of a pair share one) keeps pairs together in the probe folds
    -- see `_cv_probe_proba` for why that is required on paired corpora.
    """
    classes, y = np.unique(labels, return_inverse=True)
    if probe == "linear":
        clf = LogisticRegression(max_iter=2000, C=1.0)
    else:
        clf = _make_nonlinear_probe(probe, seed, hidden, max_iter)
    proba = _cv_probe_proba(clf, embeddings, y, seed=seed, groups=groups)
    raw, flip = _auc_raw_and_flip(y, proba, len(classes))
    return {"auc": raw, "auc_flip": flip}


def source_leakage_auc(
    embeddings: np.ndarray, labels: np.ndarray, seed: int = 0, flip: bool = False
) -> float:
    """How recoverable is the source/language from the embeddings? (lower=better).

    A logistic probe is cross-validated; we report ROC-AUC (binary) or macro
    one-vs-rest AUC (multi-class). With ``flip=True`` (the paper's headline
    setting) the AUC is flip-adjusted — ``max(auc, 1-auc)`` — so a below-chance
    probe cannot masquerade as successful erasure (see `_auc_raw_and_flip`).
    0.5 means the confounder is gone; that reading is only valid flip-adjusted.
    """
    out = probe_leakage(embeddings, labels, seed=seed, probe="linear")
    return out["auc_flip"] if flip else out["auc"]


def crossfit_leakage_auc(
    embeddings: np.ndarray,
    labels: np.ndarray,
    Z: np.ndarray | None = None,
    probe: str = "linear",
    n_splits: int = 5,
    seed: int = 0,
    hidden=(256, 128),
    max_iter: int = 300,
    groups=None,
) -> dict:
    """Leakage AUC with the eraser *refit inside each CV training fold*.

    This is the statistically clean guardedness protocol: for each fold we fit
    a LEACE eraser on the training documents only (using concept matrix `Z`),
    apply it to both splits, train the probe on the erased training docs, and
    score the erased held-out docs. Because held-out documents never influence
    the eraser, the below-chance anti-correlation artifact of the transductive
    protocol cannot arise — a clean erasure yields AUC ≈ 0.5, full stop.

    Parameters
    ----------
    Z : (n, d_z) concept matrix to erase (continuous columns are standardised
        per training fold), or ``None`` to probe the *raw* embeddings (no
        erasure — the reference condition).
        For the metadata oracle pass the one-hot label matrix; for the SAE
        condition pass the decoder-direction projections ``X @ D_S.T`` (or the
        selected activations). Note the SAE itself / the latent selection may
        still have seen all documents — this cross-fits the *eraser and probe*,
        which is where the artifact lives.

    `groups` keeps matched pairs in one fold (see `_cv_probe_proba`).
    Returns ``{"auc": raw, "auc_flip": flip-adjusted}``.
    """
    X = np.asarray(embeddings, dtype=np.float32)
    classes, y = np.unique(labels, return_inverse=True)
    proba = np.zeros((len(y), len(classes)), dtype=np.float64)
    cv = _probe_cv(seed, n_splits, groups)
    split_iter = (cv.split(X, y, groups=np.asarray(groups)) if groups is not None
                  else cv.split(X, y))
    for tr, te in split_iter:
        X_tr, X_te = X[tr], X[te]
        if Z is not None:
            Zs = np.asarray(Z, dtype=np.float32)
            scaler = StandardScaler().fit(Zs[tr])
            Z_tr = torch.from_numpy(scaler.transform(Zs[tr])).float()
            eraser = LeaceEraser.fit(torch.from_numpy(X_tr).float(), Z_tr)
            X_tr = eraser(torch.from_numpy(X_tr).float()).numpy()
            X_te = eraser(torch.from_numpy(X_te).float()).numpy()
        if probe == "linear":
            clf = LogisticRegression(max_iter=2000, C=1.0)
        else:
            clf = _make_nonlinear_probe(probe, seed, hidden, max_iter)
        clf.fit(X_tr, y[tr])
        proba[te] = clf.predict_proba(X_te)
    raw, flip = _auc_raw_and_flip(y, proba if len(classes) > 2 else proba[:, 1], len(classes))
    return {"auc": raw, "auc_flip": flip}


def _make_nonlinear_probe(probe: str, seed: int, hidden, max_iter: int):
    """Build a nonlinear classifier for the leakage probe."""
    from sklearn.pipeline import make_pipeline

    if probe == "mlp":
        from sklearn.neural_network import MLPClassifier
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=hidden, max_iter=max_iter,
                          early_stopping=True, random_state=seed),
        )
    if probe == "gb":  # gradient-boosted trees — fast, very different inductive bias
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=seed)
    if probe == "rbf":  # kernel SVM — powerful but slow (probability calibration)
        from sklearn.svm import SVC
        return make_pipeline(
            StandardScaler(), SVC(kernel="rbf", probability=True, random_state=seed)
        )
    raise ValueError(f"unknown probe {probe!r} (use 'mlp', 'gb', or 'rbf')")


def nonlinear_leakage_auc(
    embeddings: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
    probe: str = "mlp",
    hidden=(256, 128),
    max_iter: int = 300,
    flip: bool = False,
) -> float:
    """Nonlinear analogue of `source_leakage_auc`: a cross-validated nonlinear probe.

    LEACE (and our direction erasure) only guarantee *linear* guardedness, so a
    below-chance *linear* leakage AUC does not prove the confounder is truly gone —
    a nonlinear classifier may still recover it. We therefore also report nonlinear
    probes. `probe` selects the classifier: ``"mlp"`` (neural net), ``"gb"``
    (gradient-boosted trees — a very different inductive bias, a good second
    opinion), or ``"rbf"`` (kernel SVM; strongest but slow). Report it for every
    condition, **including the oracle**, so the linear and nonlinear pictures can be
    compared fairly. 0.5 is chance / macro one-vs-rest chance (multi-class).
    ``flip=True`` flip-adjusts (see `_auc_raw_and_flip`).
    """
    out = probe_leakage(embeddings, labels, seed=seed, probe=probe,
                        hidden=hidden, max_iter=max_iter)
    return out["auc_flip"] if flip else out["auc"]


def cluster_purity(embeddings: np.ndarray, labels: np.ndarray, k: int, seed: int = 0) -> float:
    """Mean cluster purity w.r.t. the source label (lower=better, chance≈base rate)."""
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    clusters = km.fit_predict(embeddings)
    classes, y = np.unique(labels, return_inverse=True)
    purities = []
    for c in range(k):
        members = y[clusters == c]
        if len(members) == 0:
            continue
        counts = np.bincount(members, minlength=len(classes))
        purities.append(counts.max() / counts.sum())
    return float(np.mean(purities))


def pair_partners(pairs) -> dict | None:
    """Every document's counterparts under `pairs`, or None when no document has
    more than one (disjoint pairs, where there is nothing to filter).

    In a three-class corpus a case's documents form a triple, so a query has two
    counterparts. Retrieval then ranks each target without the query's other
    counterpart (the "filtered" ranking of multi-answer retrieval): a query whose
    Italian version outranks its French one has not missed the French one.
    """
    nb: dict = {}
    for i, j in pairs:
        nb.setdefault(int(i), set()).add(int(j))
        nb.setdefault(int(j), set()).add(int(i))
    return nb if any(len(v) > 1 for v in nb.values()) else None


def _filtered_row(sims: np.ndarray, q: int, t: int, nb) -> np.ndarray:
    """Query `q`'s similarity row, with its counterparts other than `t` removed."""
    row = sims[q]
    if nb is not None and len(nb[q]) > 1:
        row = row.copy()
        row[[o for o in nb[q] if o != t]] = -np.inf
    return row


def paired_retrieval_recall(embeddings: np.ndarray, pairs, ks=(1, 5, 10)) -> dict:
    """Recall@k for matched cross-source/language pairs (higher=better).

    `pairs` is a list of (i, j) indices that should retrieve each other. For each
    query i we rank all other documents by cosine similarity and check whether
    its partner j lands in the top-k. We average over both directions. A query
    with further counterparts (a three-class corpus) is ranked without them
    (`pair_partners`).
    """
    sims = _cosine(embeddings, embeddings)
    np.fill_diagonal(sims, -np.inf)
    nb = pair_partners(pairs)
    out = {}
    for k in ks:
        hits = 0
        total = 0
        for i, j in pairs:
            topk_i = np.argpartition(-_filtered_row(sims, i, j, nb), k)[:k]
            topk_j = np.argpartition(-_filtered_row(sims, j, i, nb), k)[:k]
            hits += int(j in topk_i) + int(i in topk_j)
            total += 2
        out[f"recall@{k}"] = hits / total
    return out


def pair_hit_vector(embeddings: np.ndarray, pairs, k: int = 10) -> np.ndarray:
    """Per-pair retrieval outcome, for significance testing.

    Returns a (n_pairs,) vector in {0, 0.5, 1}: for pair (i, j), the fraction of
    the two directed queries (i→j, j→i) whose partner lands in the top-k. The
    mean of this vector equals `paired_retrieval_recall`'s recall@k, but keeping
    the per-pair outcomes lets us bootstrap the *difference* between two
    conditions over the same pairs (a paired test — much more powerful than
    comparing two aggregate recalls).
    """
    sims = _cosine(embeddings, embeddings)
    np.fill_diagonal(sims, -np.inf)
    nb = pair_partners(pairs)
    hits = np.zeros(len(pairs))
    for p, (i, j) in enumerate(pairs):
        topk_i = np.argpartition(-_filtered_row(sims, i, j, nb), k)[:k]
        topk_j = np.argpartition(-_filtered_row(sims, j, i, nb), k)[:k]
        hits[p] = (int(j in topk_i) + int(i in topk_j)) / 2.0
    return hits


def paired_bootstrap_diff(
    hits_a: np.ndarray,
    hits_b: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap over queries for the recall difference of two conditions.

    `hits_a`, `hits_b` are per-pair hit vectors (`pair_hit_vector`) computed on
    the SAME pairs. Resamples pairs with replacement and reports the mean
    difference (a − b), a 95% CI, and a two-sided bootstrap p-value for
    diff = 0. Use it for every headline recall comparison — with only a few
    hundred test pairs, gaps of a few points can be well within noise.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(hits_a, dtype=float) - np.asarray(hits_b, dtype=float)
    n = len(d)
    boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # two-sided p: how often the resampled mean difference crosses zero
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {
        "diff": float(d.mean()),
        "ci95": (float(lo), float(hi)),
        "p_boot": float(min(1.0, p)),
        "n_pairs": int(n),
    }


def seed_paired_test(values_a, values_b) -> dict:
    """Paired t-test + Wilcoxon over per-seed metric values of two conditions.

    For metrics without a per-query decomposition (leakage AUC, purity), the
    unit of variation is the seed. Feed the per-seed values of two conditions
    (same seeds, same order). With ≤5 seeds these tests are underpowered — treat
    p-values as descriptive and lean on the bootstrap where available.
    """
    from scipy.stats import ttest_rel, wilcoxon

    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    t_stat, t_p = ttest_rel(a, b)
    try:
        w_stat, w_p = wilcoxon(a, b)
    except ValueError:  # all differences zero
        w_stat, w_p = 0.0, 1.0
    return {
        "mean_diff": float((a - b).mean()),
        "t_p": float(t_p),
        "wilcoxon_p": float(w_p),
        "n_seeds": int(len(a)),
    }


def gold_similarity_spearman(embeddings: np.ndarray, pairs, gold_sim) -> float:
    """Spearman between embedding cosine of a pair and a human similarity score.

    A semantic-preservation check (higher=better): deconfounding should not
    destroy genuine meaning-level similarity. Returns NaN if unavailable.
    """
    from scipy.stats import spearmanr

    cos = np.array([
        float(_cosine(embeddings[i : i + 1], embeddings[j : j + 1])[0, 0])
        for i, j in pairs
    ])
    rho, _ = spearmanr(cos, np.asarray(gold_sim))
    return float(rho)


def evaluate_embeddings(
    embeddings: np.ndarray,
    source_labels: np.ndarray,
    pairs=None,
    gold_sim=None,
    cluster_k: int | None = None,
    seed: int = 0,
    nonlinear_leakage: bool = True,
    nonlinear_probes=("mlp", "gb"),
) -> dict:
    """Bundle the metrics that matter for deconfounding into one dict.

    When `nonlinear_leakage` is True, a nonlinear leakage probe is reported for
    each classifier in `nonlinear_probes` (default MLP + gradient-boosting) —
    worth having since our erasure only guarantees *linear* removal, and two
    different nonlinear probes agreeing is stronger evidence the confounder is
    truly gone. The MLP result keeps the key `nonlinear_leakage_auc`; other probes
    get `nonlinear_leakage_auc_<probe>`.

    Every leakage key also gets a ``*_flip`` companion — the flip-adjusted AUC
    ``max(auc, 1-auc)``, which is the value to compare across conditions (raw
    below-chance AUCs are an artifact of the transductive eraser, not evidence
    of extra-strong erasure; see `_auc_raw_and_flip`).
    """
    n_sources = len(np.unique(source_labels))
    cluster_k = cluster_k or max(2, n_sources)
    lin = probe_leakage(embeddings, source_labels, seed=seed, probe="linear")
    metrics = {
        "source_leakage_auc": lin["auc"],
        "source_leakage_auc_flip": lin["auc_flip"],
    }
    if nonlinear_leakage:
        for probe in nonlinear_probes:
            key = "nonlinear_leakage_auc" if probe == "mlp" else f"nonlinear_leakage_auc_{probe}"
            out = probe_leakage(embeddings, source_labels, seed=seed, probe=probe)
            metrics[key] = out["auc"]
            metrics[f"{key}_flip"] = out["auc_flip"]
    metrics["cluster_purity"] = cluster_purity(embeddings, source_labels, cluster_k, seed)
    if pairs is not None and len(pairs) > 0:
        metrics.update(paired_retrieval_recall(embeddings, pairs))
        if gold_sim is not None:
            metrics["gold_sim_spearman"] = gold_similarity_spearman(
                embeddings, pairs, gold_sim
            )
    return metrics


# --------------------------------------------------------------------------- #
# Orchestration: the full 3-way comparison
# --------------------------------------------------------------------------- #
def split_pairs(pairs, gold_sim=None, holdout: float = 0.5, seed: int = 0):
    """Split matched pairs into (train_pairs, test_pairs, gold_test).

    Used so pair-optimising selectors (greedy) never see the pairs they are
    evaluated on. With ``holdout`` in (0, 1), a seeded permutation reserves that
    fraction as the test half; ``holdout=0`` returns the full list for both
    roles (transductive). `gold_sim` (if given) is subset alongside the test
    pairs. Shared by `run_sae_deconfounding` and the baseline comparison so all
    conditions are scored on the identical held-out pairs.
    """
    if pairs is None or len(pairs) == 0 or not holdout or holdout <= 0:
        return pairs, pairs, gold_sim
    idx = np.random.default_rng(seed).permutation(len(pairs))
    n_test = int(np.clip(round(holdout * len(pairs)), 1, len(pairs) - 1))
    pairs_arr = np.asarray(pairs)
    test_pairs = [tuple(int(x) for x in pairs_arr[i]) for i in idx[:n_test]]
    train_pairs = [tuple(int(x) for x in pairs_arr[i]) for i in idx[n_test:]]
    gold_test = np.asarray(gold_sim)[idx[:n_test]] if gold_sim is not None else None
    return train_pairs, test_pairs, gold_test


@dataclass
class ConfounderResult:
    metrics: dict = field(default_factory=dict)          # condition -> metric dict
    selected_features: np.ndarray = None
    feature_scores: np.ndarray = None
    interpretation: dict = field(default_factory=dict)
    sae_info: dict = field(default_factory=dict)
    label_alignment: dict = field(default_factory=dict)  # feature -> AUC vs true label


def run_sae_deconfounding(
    embeddings: np.ndarray,
    texts: list[str],
    source_labels: np.ndarray,
    pairs=None,
    gold_sim=None,
    expansion_factor: int = 8,
    k: int = 32,
    n_epochs: int = 200,
    top_n_features: int = 32,
    n_pseudo_groups: int = 2,
    cluster_k: int | None = None,
    seed: int = 0,
    verbose: bool = True,
    device: str | None = None,
    erase: str = "directions_leace",
    selection: str = "auto",
    recall_k: int = 10,
    greedy_lambda: float = 1.0,
    greedy_budget: int = 64,
    greedy_pool: int = 80,
    pair_holdout: float = 0.5,
    nonlinear_leakage: bool = True,
    nonlinear_probes=("mlp", "gb"),
    crossfit: bool = True,
) -> ConfounderResult:
    """End-to-end: train SAE, discover + interpret confounders, erase, compare.

    Returns a `ConfounderResult` with metrics for the three conditions
    ('raw', 'metadata_leace', 'sae_leace') plus the discovered features and
    their interpretations. `device` ('cuda'/'cpu') is forwarded to SAE training;
    if unset it is auto-detected (uses the GPU on Colab when available).

    `erase` selects how the `sae_leace` condition removes the discovered
    confounder: ``"directions_leace"`` (default — LEACE on the decoder directions,
    the strongest mechanism in our experiments), ``"activations"`` (the original
    sparse-activation erasure), or ``"directions_orth"``.

    `selection` chooses **which** latents to erase: ``"greedy"`` (the label-free
    greedy selector — best in our experiments; erases its objective-optimal number
    of latents, optimised for `erase`), ``"topn"`` (the top `top_n_features` by MI
    score), ``"asymmetry"`` (the top `top_n_features` by **pair asymmetry** — a
    dominance-free selector that does not use the k-means proxy, for when the
    confounder does not dominate the geometry; needs pairs), or ``"auto"`` (default
    — greedy when matched `pairs` are available, otherwise top-N). The greedy path
    uses `greedy_lambda/greedy_budget/greedy_pool` and `recall_k`.

    `crossfit` (default True) adds `crossfit_leakage_auc[_flip]` to every
    condition — the cross-fitted protocol in which the eraser is refit inside
    each CV training fold. It is the leakage number to compare conditions on;
    the transductive `source_leakage_auc[_flip]` is kept for continuity with
    earlier runs but is contaminated by the full-sample eraser fit.

    `pair_holdout` (default 0.5) guards against optimistic retrieval numbers: the
    greedy objective optimises `recall@k` on the pairs, so evaluating on the same
    pairs would be train-set recall. When >0, the pairs are split — greedy selects
    on the *train* half and **all three conditions** are evaluated on the held-out
    *test* half (a fair, unbiased estimate). Set to 0 to use all pairs for both.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    source_labels = np.asarray(source_labels)
    res = ConfounderResult()

    # (2) Train the SAE on the raw embeddings (on GPU when available).
    sae, info = train_sae(
        embeddings, expansion_factor=expansion_factor, k=k,
        n_epochs=n_epochs, seed=seed, verbose=verbose, device=device,
    )
    res.sae_info = info
    feats = encode_features(sae, embeddings)
    decoder = sae.W_dec.detach().cpu().numpy()

    # (3) Score + select confounder latents (no labels used here).
    scores, _, freq = score_confounder_features(
        feats, embeddings, n_pseudo_groups=n_pseudo_groups, seed=seed
    )
    has_pairs = pairs is not None and len(pairs) > 0
    if selection in ("greedy", "asymmetry") and not has_pairs:
        raise ValueError(f"selection={selection!r} needs matched pairs; use 'topn' for unpaired data")

    # Split pairs into a selection (train) half and an evaluation (test) half so
    # greedy can't inflate its reported recall by optimising the pairs it's scored on.
    train_pairs, test_pairs, gold_test = split_pairs(
        pairs, gold_sim, holdout=pair_holdout if has_pairs else 0, seed=seed
    )
    if has_pairs and pair_holdout and pair_holdout > 0 and verbose:
        print(f"[pairs] holdout: {len(train_pairs)} train (greedy selection) / "
              f"{len(test_pairs)} test (evaluation)")

    use_greedy = selection == "greedy" or (selection == "auto" and has_pairs)
    if selection == "asymmetry":
        selected = select_confounder_features_asymmetry(feats, train_pairs, top_n=top_n_features)
    elif use_greedy:
        order, best_len = select_confounder_features_greedy(
            embeddings, feats, train_pairs, scores,
            budget=greedy_budget, candidate_pool=greedy_pool, lam=greedy_lambda,
            pseudo_groups=n_pseudo_groups, recall_k=recall_k,
            erase=erase, decoder=decoder, seed=seed, verbose=verbose,
        )
        selected = np.array(order[:best_len])  # greedy's objective-optimal cut-off
    else:
        selected = select_confounder_features(scores, top_n=top_n_features)
    res.feature_scores = scores
    res.selected_features = selected

    # (4) Interpret the selected latents with high-activation examples (bottom
    # examples ranked by dense pre-activations — see `interpret_features`).
    pre = encode_preactivations(sae, embeddings)
    res.interpretation = interpret_features(
        feats, texts, selected, source_labels=source_labels, pre=pre
    )
    if verbose:
        print_interpretation(res.interpretation)

    # Sanity check: how well does each selected latent align with the TRUE label?
    classes, y_true = np.unique(source_labels, return_inverse=True)
    if len(classes) == 2:
        for j in selected:
            try:
                res.label_alignment[int(j)] = float(roc_auc_score(y_true, feats[:, j]))
            except ValueError:
                res.label_alignment[int(j)] = float("nan")

    # (6) Build the three embedding sets, each with the concept matrix that
    #     describes what it erased (needed for the cross-fitted probe).
    emb_raw = embeddings
    emb_meta = leace_erase_with_labels(embeddings, source_labels)
    emb_sae = erase_embeddings(embeddings, feats, decoder, selected, mechanism=erase)

    # (7) Evaluate each.
    for name, emb, Z in (
        ("raw", emb_raw, None),
        ("metadata_leace", emb_meta, one_hot_labels(source_labels)),
        ("sae_leace", emb_sae,
         concept_matrix(embeddings, feats, decoder, selected, mechanism=erase)),
    ):
        m = evaluate_embeddings(
            emb, source_labels, pairs=test_pairs, gold_sim=gold_test,
            cluster_k=cluster_k, seed=seed, nonlinear_leakage=nonlinear_leakage,
            nonlinear_probes=nonlinear_probes,
        )
        if crossfit:
            # The artifact-free guardedness number: eraser refit inside each CV
            # training fold, so held-out documents never shape the eraser.
            cf = crossfit_leakage_auc(embeddings, source_labels, Z=Z, seed=seed)
            m["crossfit_leakage_auc"] = cf["auc"]
            m["crossfit_leakage_auc_flip"] = cf["auc_flip"]
        res.metrics[name] = m
    return res


def format_comparison(res: ConfounderResult) -> str:
    """Pretty-print the 3-way metric table."""
    conds = ["raw", "metadata_leace", "sae_leace"]
    keys = []
    for c in conds:
        for kk in res.metrics[c]:
            if kk not in keys:
                keys.append(kk)
    width = max(len(k) for k in keys) + 2
    lines = [f"{'metric':<{width}}" + "".join(f"{c:>16}" for c in conds)]
    lines.append("-" * len(lines[0]))
    for kk in keys:
        row = f"{kk:<{width}}"
        for c in conds:
            v = res.metrics[c].get(kk, float("nan"))
            row += f"{v:>16.4f}"
        lines.append(row)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Label-free greedy selection (pair-aware)
# --------------------------------------------------------------------------- #
def pair_asymmetry(feats: np.ndarray, pairs) -> np.ndarray:
    """Per-latent "language-ness" score from matched pairs — **no labels used**.

    A pair (i, j) is the *same content in two languages*. A genuine confounder
    latent (language/style) fires on one side of the pair but not the other, so
    its within-pair activation difference is large. A content latent fires on
    both sides similarly, so its within-pair difference is small. We therefore
    score each latent by its mean absolute within-pair difference, normalised by
    its overall activation scale. High = looks like a nuisance, not meaning.

    This uses only the pairing structure of the dataset (which translations match
    which), never the language/source label itself.
    """
    pairs = np.asarray(pairs)
    a = feats[pairs[:, 0]]
    b = feats[pairs[:, 1]]
    within_diff = np.abs(a - b).mean(axis=0)
    scale = feats.mean(axis=0) + 1e-8
    return within_diff / scale


def select_confounder_features_asymmetry(
    feats: np.ndarray,
    pairs,
    top_n: int = 32,
    min_activation_freq: float = 0.02,
):
    """Select confounder latents purely from **pair asymmetry** — no k-means, no
    "confounder dominates the coarse geometry" assumption.

    The MI/greedy scorers use a k-means grouping of the raw embeddings as a
    label-free stand-in for the confounder; that only works when the confounder
    *dominates* the geometry (so the coarse clusters ≈ the confounder). On a
    strong, full-context encoder the geometry can be organised by content instead,
    and the k-means proxy points the selector at content latents.

    Pair asymmetry sidesteps this: a translation pair is the *same content in two
    languages*, so a latent that fires on one side but not the other is language /
    style regardless of whether language dominates clustering. We simply rank
    latents by `pair_asymmetry` and take the top `top_n` (dropping near-dead latents
    below `min_activation_freq`, whose asymmetry ratio is noise). This is the
    dominance-free discovery signal — the one expected to survive the low-confound
    (e.g. jina) regime.
    """
    asym = pair_asymmetry(feats, pairs)
    freq = (feats > 0).mean(axis=0)
    asym = np.where(freq >= min_activation_freq, asym, -np.inf)
    return np.argsort(asym)[::-1][:top_n]


def _fast_group_auc(emb: np.ndarray, group: np.ndarray, seed: int = 0) -> float:
    """Cheap label-free leakage proxy: train-set AUC of a probe predicting the
    unsupervised pseudo-group. No cross-validation (used inside the greedy inner
    loop where speed matters); the *reported* leakage still uses the CV probe."""
    classes = np.unique(group)
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(emb, group)
    if len(classes) == 2:
        return float(roc_auc_score(group, clf.predict_proba(emb)[:, 1]))
    return float(roc_auc_score(group, clf.predict_proba(emb), multi_class="ovr", average="macro"))


def select_confounder_features_greedy(
    embeddings: np.ndarray,
    feats: np.ndarray,
    pairs,
    base_scores: np.ndarray,
    budget: int = 48,
    candidate_pool: int = 64,
    lam: float = 1.0,
    patience: int = 8,
    pseudo_groups: int = 2,
    recall_k: int = 10,
    erase: str = "activations",
    decoder: np.ndarray | None = None,
    seed: int = 0,
    verbose: bool = True,
    pseudo_group: np.ndarray | None = None,
):
    """Greedily grow the set of latents to erase, optimising a **label-free**
    objective ``J = recall@k  -  lam * pseudo_leakage``.

    `erase` controls the erasure mechanism used *inside* the objective (so the
    selection is optimised for the same mechanism it will be evaluated under);
    direction-based mechanisms require `decoder`.

    Both terms are label-free: ``recall@k`` uses only the pairing structure, and
    ``pseudo_leakage`` is a probe predicting an unsupervised k-means grouping of
    the raw embeddings (the confounder proxy). At each step we tentatively add
    every still-unused latent from a shortlisted candidate pool, LEACE-erase the
    candidate set, and keep the latent that most improves J. Stops after
    `patience` non-improving steps.

    The candidate pool is pre-ranked by combining the MI confounder score with
    the pair-asymmetry score, so the search starts from latents that already look
    like language rather than meaning.

    `pseudo_group` supplies a grouping fit elsewhere (see
    `score_confounder_features`); when None it is fit here on `embeddings`.

    Returns
    -------
    order : list[int]
        Latents in the order greedy added them.
    best_len : int
        Prefix length of `order` that maximised J (the recommended cut-off).
    """
    from sklearn.cluster import KMeans

    if erase != "activations" and decoder is None:
        raise ValueError(f"greedy erase={erase!r} requires the SAE decoder (pass decoder=...)")

    g = (np.asarray(pseudo_group) if pseudo_group is not None else
         KMeans(n_clusters=pseudo_groups, random_state=seed,
                n_init="auto").fit_predict(embeddings))

    asym = pair_asymmetry(feats, pairs)
    def _z(x):
        s = x.std() + 1e-8
        return (x - x.mean()) / s
    rank = _z(base_scores) + _z(asym)
    pool = list(np.argsort(rank)[::-1][:candidate_pool])

    order, best_J, best_len, since = [], -np.inf, 0, 0
    for step in range(min(budget, len(pool))):
        best_j, best_step_J = None, -np.inf
        for j in pool:
            if j in order:
                continue
            cand = order + [j]
            erased = erase_embeddings(embeddings, feats, decoder, cand, mechanism=erase)
            q = paired_retrieval_recall(erased, pairs, ks=(recall_k,))[f"recall@{recall_k}"]
            leak = _fast_group_auc(erased, g, seed)
            J = q - lam * leak
            if J > best_step_J:
                best_step_J, best_j = J, j
        order.append(int(best_j))
        if best_step_J > best_J + 1e-4:
            best_J, best_len, since = best_step_J, len(order), 0
        else:
            since += 1
        if verbose:
            print(f"[greedy] step {step+1:3d}  add latent #{best_j:5d}  "
                  f"J={best_step_J:.4f}  |selected|={len(order)}")
        if since >= patience:
            break
    return order, best_len


# --------------------------------------------------------------------------- #
# Leakage-vs-quality trade-off sweep over how many latents we erase
# --------------------------------------------------------------------------- #
def compute_sae_features(
    embeddings: np.ndarray,
    expansion_factor: int = 8,
    k: int = 32,
    n_epochs: int = 200,
    pseudo_groups: int = 2,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = True,
    return_sae: bool = False,
):
    """Train the SAE and return ``(feats, scores, info, decoder)`` so callers can
    reuse a single trained SAE across many sweeps (e.g. a greedy-lambda sweep that
    should not retrain the identical dictionary for every lambda). ``decoder`` is
    the ``(n_latents, d_in)`` matrix of unit-norm decoder rows, needed for
    direction-based erasure (`leace_erase_with_directions`). With
    ``return_sae=True`` the trained model is appended as a fifth element (needed
    e.g. for `encode_preactivations` in the latent interpretability report)."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    sae, info = train_sae(
        embeddings, expansion_factor=expansion_factor, k=k,
        n_epochs=n_epochs, seed=seed, verbose=verbose, device=device,
    )
    feats = encode_features(sae, embeddings)
    scores, _, _ = score_confounder_features(
        feats, embeddings, n_pseudo_groups=pseudo_groups, seed=seed
    )
    decoder = sae.W_dec.detach().cpu().numpy()
    if return_sae:
        return feats, scores, info, decoder, sae
    return feats, scores, info, decoder


def leakage_recall_sweep(
    embeddings: np.ndarray,
    source_labels: np.ndarray,
    pairs=None,
    feature_grid=(8, 16, 32, 64, 96, 128),
    method: str = "topn",
    expansion_factor: int = 8,
    k: int = 32,
    n_epochs: int = 200,
    pseudo_groups: int = 2,
    recall_k: int = 10,
    cluster_k: int | None = None,
    greedy_budget: int = 48,
    greedy_pool: int = 64,
    greedy_lambda: float = 1.0,
    erase: str = "activations",
    crossfit: bool = True,
    feats: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    sae_info: dict | None = None,
    decoder: np.ndarray | None = None,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = True,
) -> dict:
    """Trace the deconfounding trade-off as we erase more SAE-discovered latents.

    Trains the SAE and scores the latents **once**, then erases an increasing
    number of latents and records, at each step:

      * `leakage` = `source_leakage_auc` (lower = more deconfounded), and
      * `y` = a downstream-quality metric: paired `recall@recall_k` when the
        dataset has matched pairs (higher = better), otherwise `cluster_purity`
        (lower = better) as a fallback.

    `method` controls *which* latents get added at each step:
      * ``"topn"``  — the latents with the highest MI confounder score, at each
        size in `feature_grid` (the original behaviour).
      * ``"greedy"`` — `select_confounder_features_greedy` grows the set by
        optimising a label-free objective; the trade-off curve is its trajectory
        (one point per added latent), and `recommended` marks the J-optimal size.

    It also records the `raw` and `metadata` (true-label LEACE) reference points,
    so a plot can show how close the label-free sweep gets to the oracle. Returns
    a plain dict (easy to print / feed to `plot_sweep` / `plot_sweeps`).
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    source_labels = np.asarray(source_labels)
    has_pairs = pairs is not None and len(pairs) > 0
    if method == "greedy" and not has_pairs:
        raise ValueError("method='greedy' needs matched pairs (use 'topn' otherwise)")
    y_metric = f"recall@{recall_k}" if has_pairs else "cluster_purity"
    ck = cluster_k or max(2, len(np.unique(source_labels)))

    def metrics_of(emb, Z=_SKIP):
        lk = probe_leakage(emb, source_labels, seed=seed, probe="linear")
        # `leakage` is the transductive flip-adjusted AUC; `leakage_raw` keeps
        # the raw CV AUC; `crossfit` is the artifact-free protocol (the eraser
        # is refit inside each CV training fold) and is the axis that agrees
        # with the headline baselines table. See `crossfit_leakage_auc`.
        out = {"leakage": lk["auc_flip"], "leakage_raw": lk["auc"]}
        if crossfit and Z is not _SKIP:
            cf = crossfit_leakage_auc(embeddings, source_labels, Z=Z, seed=seed)
            out["crossfit"] = cf["auc_flip"]
            out["crossfit_raw"] = cf["auc"]
        if has_pairs:
            rr = paired_retrieval_recall(emb, pairs, ks=(1, recall_k))
            out["y"] = rr[f"recall@{recall_k}"]
            out["recall@1"] = rr["recall@1"]  # stricter metric, same sweep
        else:
            out["y"] = cluster_purity(emb, source_labels, ck, seed)
        return out

    # Train + score once (or reuse a precomputed SAE passed in by the caller).
    if feats is None or scores is None:
        feats, scores, sae_info, decoder = compute_sae_features(
            embeddings, expansion_factor=expansion_factor, k=k, n_epochs=n_epochs,
            pseudo_groups=pseudo_groups, seed=seed, device=device, verbose=verbose,
        )
    if erase != "activations" and decoder is None:
        raise ValueError(
            f"erase={erase!r} requires the SAE decoder; pass decoder=... alongside "
            "feats/scores, or omit feats so it is computed here."
        )

    out = {
        "method": method,
        "erase": erase,
        "y_metric": y_metric,
        "crossfit": bool(crossfit),
        # Z=None -> cross-fitted probe on the *unerased* embeddings (the
        # reference); Z=one-hot -> the label oracle refit inside each fold.
        "raw": metrics_of(embeddings, Z=None),
        "metadata": metrics_of(leace_erase_with_labels(embeddings, source_labels),
                               Z=one_hot_labels(source_labels)),
        "sae": [],
        "sae_info": sae_info or {},
        "recommended": None,
    }

    if method == "topn":
        out["feature_grid"] = list(feature_grid)
        steps = [(tn, select_confounder_features(scores, top_n=tn)) for tn in feature_grid]
    elif method == "greedy":
        order, best_len = select_confounder_features_greedy(
            embeddings, feats, pairs, scores,
            budget=greedy_budget, candidate_pool=greedy_pool, lam=greedy_lambda,
            pseudo_groups=pseudo_groups, recall_k=recall_k,
            erase=erase, decoder=decoder, seed=seed, verbose=verbose,
        )
        out["recommended"] = int(best_len)
        # Evaluate the trajectory (prefix sizes) with the REAL metrics for plotting.
        grid = sorted(set(list(range(1, len(order) + 1))))
        steps = [(m, np.array(order[:m])) for m in grid]
    else:
        raise ValueError(f"unknown method {method!r} (use 'topn' or 'greedy')")

    for tn, selected in steps:
        emb_erased = erase_embeddings(embeddings, feats, decoder, selected, mechanism=erase)
        Z = concept_matrix(embeddings, feats, decoder, selected, mechanism=erase)
        out["sae"].append({
            "top_features": int(tn),
            "n_selected": int(len(selected)),
            **metrics_of(emb_erased, Z=Z),
        })
        if verbose:
            p = out["sae"][-1]
            cf = f"  crossfit={p['crossfit']:.4f}" if "crossfit" in p else ""
            print(f"[sweep:{method}/{erase}] n={tn:4d}  leakage={p['leakage']:.4f}{cf}  "
                  f"{y_metric}={p['y']:.4f}")
    return out
