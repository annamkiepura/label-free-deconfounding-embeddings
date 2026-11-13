"""The evaluation protocol: probe audits of leakage, and the utility measures.

Every number the revised tables report is computed here, so the protocol is
stated once.

Linear leakage (Sec. 3.2-3.3)
------------------------------
Absolute ROC-AUC of an L2-regularised logistic probe that predicts the true
nuisance label from the *transformed held-out* documents, under an independent
audit:

1. the transformation was fit on the outer training split and frozen (by the
   caller -- nothing here fits an eraser);
2. within the held-out set the probe is trained and tested by K-fold
   cross-validation, K = 5, reduced only when some class could not appear in
   every train and test fold (the K used is returned); matched pairs share a
   group and therefore a fold;
3. inputs are standardised with probe-training statistics only, and the
   regularisation C in {0.01, 0.1, 1, 10, 100} is chosen on an inner split of
   the probe-training data by mean one-vs-rest AUC;
4. AUC is computed within each audit test fold and averaged over folds.

Binary nuisances report ordinary ROC-AUC. Multiclass nuisances fit one-vs-rest
logistic classifiers and average their per-class AUCs with equal weights; the
per-class values are returned too. There is no flip adjustment: 0.5 is chance,
and an AUC below 0.5 is not better erasure.

Nonlinear probes (Sec. 3.4)
---------------------------
Gradient-boosted trees and an MLP on the same audit folds, ordinary unflipped
AUC -- appendix diagnostics of residual nonlinear recoverability.

Topic clustering (Sec. 3.1)
---------------------------
k-means with k = 21 and n_init = 20 on the transformed held-out CAP documents.
Purity weights each cluster by its size, ``(1/n) sum_k max_t |C_k & T_t|``;
ARI is the adjusted Rand index against the topic labels. Topic labels score
the clusters and never enter a fit.

Representation audited: the affine map's output ``T(x)``, not renormalised --
the vectors every probe and clustering here sees. Retrieval scores cosine
similarity, which a renormalisation would not change.

Numerically constant coordinates
--------------------------------
A coordinate whose standard deviation on the probe-training fold is below
`REL_STD_FLOOR` times the median coordinate's is treated as constant and left
out of that fold's probes -- scikit-learn's handling of zero-variance features
(``StandardScaler`` leaves them unscaled), extended to floating-point residue.
It matters for MiniLM: three of its coordinates are exactly constant in the raw
vectors, and after any affine map evaluated in float32 they carry round-off at
~1e-9 of the typical coordinate spread. That residue is a scaled-down copy of
the correction the map applied, so standardising it back to unit variance
lets a probe read the removed label from rounding error (metadata-LEACE on
MiniLM/SCOTUS audits at 0.98 with it, at the E5-small level without it). The
number of coordinates dropped is returned with every audit. A representation
whose coordinates are *all* constant carries nothing, and scores chance (0.5).
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

#: Logistic regularisation grid (plan Sec. 3.3, step 4).
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
#: Audit folds within the held-out set, before any reduction.
AUDIT_FOLDS = 5
#: The inner split for choosing C is one fold of this many (an 80/20 split).
INNER_FOLDS = 5
#: Topic clustering (plan Sec. 3.1): K_topic and k-means restarts.
K_TOPIC = 21
KMEANS_N_INIT = 20
#: The nonlinear probes' settings (as in the draft's diagnostics).
MLP_HIDDEN = (256, 128)
MLP_MAX_ITER = 300
#: Coordinates with a probe-training SD below this fraction of the median
#: coordinate SD are numerically constant (see the module docstring).
REL_STD_FLOOR = 1e-6

#: Recorded with every result, so a table can state the protocol it used.
PROTOCOL = {
    "probe": "L2 logistic regression, one-vs-rest for >2 classes",
    "standardisation": "probe-training statistics",
    "C_grid": list(C_GRID), "C_selection": "inner 1/5 split of the probe-training "
                                           "data, mean one-vs-rest AUC",
    "audit_folds": AUDIT_FOLDS, "fold_aggregation": "AUC per test fold, then mean",
    "pairs": "matched pairs kept in one fold", "flip_adjustment": None,
    "nonlinear": {"gb": "HistGradientBoostingClassifier (defaults)",
                  "mlp": f"StandardScaler + MLP{MLP_HIDDEN}, early stopping"},
    "topic_k": K_TOPIC, "kmeans_n_init": KMEANS_N_INIT,
    "representation": "affine output T(x), not renormalised",
    "constant_coordinates": f"probe-training SD < {REL_STD_FLOOR:g} x median SD: "
                            "treated as constant (dropped from that fold's probes)",
}


def _groups(n: int, groups) -> np.ndarray:
    return np.arange(n) if groups is None else np.asarray(groups)


def informative(X_tr) -> np.ndarray:
    """Mask of the coordinates that are not numerically constant on `X_tr`."""
    sd = np.asarray(X_tr, dtype=np.float64).std(axis=0)
    return sd > REL_STD_FLOOR * np.median(sd)


def audit_folds(y, groups=None, seed: int = 0, n_folds: int = AUDIT_FOLDS) -> tuple:
    """``(folds, K)``: stratified, group-aware audit folds and the K actually used.

    K starts at `n_folds` and is reduced only when some train or test fold would
    otherwise miss a class, which an AUC cannot be computed without.
    """
    y = np.asarray(y)
    g = _groups(len(y), groups)
    n_classes = len(np.unique(y))
    for k in range(n_folds, 1, -1):
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        folds = list(cv.split(np.zeros(len(y)), y, g))
        if all(len(np.unique(y[tr])) == n_classes and len(np.unique(y[te])) == n_classes
               for tr, te in folds):
            return folds, k
    raise ValueError("no audit split keeps every class in every fold")


def _fit_linear(X, y, C: float, classes):
    base = LogisticRegression(C=C, max_iter=5000)
    return (base if len(classes) == 2 else OneVsRestClassifier(base)).fit(X, y)


def _scores(model, X) -> np.ndarray:
    s = model.decision_function(X)
    return s[:, None] if s.ndim == 1 else s


def _class_aucs(y, S: np.ndarray, classes) -> dict:
    """One-vs-rest AUC per class from decision scores or probabilities.

    Binary problems carry one score column (for ``classes[1]``) and one AUC,
    reported under both classes so the macro mean is that AUC.
    """
    y = np.asarray(y)
    if len(classes) == 2:
        col = S[:, -1] if S.shape[1] > 1 else S[:, 0]
        a = float(roc_auc_score(y == classes[1], col))
        return {str(classes[0]): a, str(classes[1]): a}
    return {str(c): float(roc_auc_score(y == c, S[:, i])) for i, c in enumerate(classes)}


def _choose_C(X, y, groups, classes, seed: int, Cs=C_GRID) -> float:
    """The C with the best mean one-vs-rest AUC on an inner validation split."""
    for k in range(INNER_FOLDS, 1, -1):
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        itr, iva = next(cv.split(X, y, groups))
        if (len(np.unique(y[itr])) == len(classes)
                and len(np.unique(y[iva])) == len(classes)):
            break
    else:
        return 1.0
    scaler = StandardScaler().fit(X[itr])
    Xi, Xv = scaler.transform(X[itr]), scaler.transform(X[iva])
    val = {C: np.mean(list(_class_aucs(y[iva], _scores(_fit_linear(Xi, y[itr], C, classes),
                                                       Xv), classes).values()))
           for C in Cs}
    return float(max(Cs, key=lambda C: val[C]))


def linear_audit(E, y, groups=None, seed: int = 0, Cs=C_GRID) -> dict:
    """The independent linear-probe audit of one transformed held-out set.

    Returns ``{"auc", "per_class", "fold_aucs", "C", "n_folds", "n_constant"}``:
    the macro AUC averaged over audit folds, the per-class AUCs averaged the
    same way, each fold's macro AUC, the C chosen in each fold, the number of
    folds, and the most numerically constant coordinates any fold dropped.
    """
    E = np.asarray(E, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    g = _groups(len(y), groups)
    folds, k = audit_folds(y, g, seed)
    fold_macro, fold_class, chosen, dropped = [], [], [], []
    for tr, te in folds:
        keep = informative(E[tr])
        if not keep.any():
            # Every coordinate is constant on this fold: the representation carries
            # nothing, so every probe is the constant predictor and the AUC is chance.
            fold_class.append({str(c): 0.5 for c in classes})
            fold_macro.append(0.5)
            chosen.append(None)
            dropped.append(int(E.shape[1]))
            continue
        Etr, Ete = E[tr][:, keep], E[te][:, keep]
        C = _choose_C(Etr, y[tr], g[tr], classes, seed, Cs)
        scaler = StandardScaler().fit(Etr)
        model = _fit_linear(scaler.transform(Etr), y[tr], C, classes)
        per = _class_aucs(y[te], _scores(model, scaler.transform(Ete)), classes)
        fold_class.append(per)
        fold_macro.append(float(np.mean(list(per.values()))))
        chosen.append(C)
        dropped.append(int((~keep).sum()))
    return {"auc": float(np.mean(fold_macro)),
            "per_class": {c: float(np.mean([f[c] for f in fold_class]))
                          for c in map(str, classes)},
            "fold_aucs": fold_macro, "C": chosen, "n_folds": k,
            "n_constant": int(max(dropped))}


def _nonlinear(probe: str, seed: int):
    if probe == "gb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=seed)
    if probe == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=MLP_HIDDEN,
                                           max_iter=MLP_MAX_ITER, early_stopping=True,
                                           random_state=seed))
    raise ValueError(f"unknown nonlinear probe {probe!r}; use 'gb' or 'mlp'")


def nonlinear_audit(E, y, groups=None, seed: int = 0, probe: str = "gb") -> dict:
    """A nonlinear probe on the same audit folds as `linear_audit`; unflipped AUC."""
    E = np.asarray(E, dtype=np.float64)
    y = np.asarray(y)
    # Fit on integer codes: scikit-learn 1.6-1.7's MLP early-stopping score calls
    # np.isnan on the predicted labels and fails on string classes. Codes follow
    # the sorted order of `classes`, so column j of predict_proba is classes[j],
    # exactly as with the string labels.
    classes, codes = np.unique(y, return_inverse=True)
    folds, k = audit_folds(y, _groups(len(y), groups), seed)
    fold_macro, fold_class = [], []
    for tr, te in folds:
        keep = informative(E[tr])
        if not keep.any():                       # constant representation: chance
            fold_class.append({str(c): 0.5 for c in classes})
            fold_macro.append(0.5)
            continue
        clf = _nonlinear(probe, seed).fit(E[tr][:, keep], codes[tr])
        per = _class_aucs(y[te], clf.predict_proba(E[te][:, keep]), classes)
        fold_class.append(per)
        fold_macro.append(float(np.mean(list(per.values()))))
    return {"auc": float(np.mean(fold_macro)),
            "per_class": {c: float(np.mean([f[c] for f in fold_class]))
                          for c in map(str, classes)},
            "fold_aucs": fold_macro, "n_folds": k}


def purity(clusters, labels) -> float:
    """Cluster-size-weighted purity, ``(1/n) sum_k max_t |C_k & T_t|``."""
    clusters, labels = np.asarray(clusters), np.asarray(labels)
    total = 0
    for c in np.unique(clusters):
        _, counts = np.unique(labels[clusters == c], return_counts=True)
        total += int(counts.max())
    return float(total / len(labels))


def topic_scores(E, topics, seed: int = 0, k: int = K_TOPIC,
                 n_init: int = KMEANS_N_INIT) -> dict:
    """Topic purity and ARI of a k-means clustering of the transformed documents."""
    clusters = KMeans(n_clusters=k, n_init=n_init,
                      random_state=seed).fit_predict(np.asarray(E, dtype=np.float64))
    return {"topic_purity": purity(clusters, topics),
            "topic_ari": float(adjusted_rand_score(np.asarray(topics), clusters))}
