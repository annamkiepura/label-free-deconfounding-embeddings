"""The second round of runs added in the revision (`notebooks/04_extensions.ipynb`, Sections 3-8).

Each section is one experiment, on the same splits, recipe and audit as the published
runs. Nothing published is overwritten: every new run writes under the active results root (the
notebook uses ``results/extensions/``), and the seed extension under a sub-root of its own.

* **Table 3 on every setting, extended (`run_table3x`).** Main Table 3's rows -- direction
  selection (MI-selected vs random SAE latents vs Gaussian directions), operator (Activation-LEACE,
  orthogonal removal) and erasure extent (displacement-matched PCA) -- on one SAE per split, on
  three-language SwiLTra, three-source SCOTUS, SemEval, CAP and Fedlex, plus MiniLM on SCOTUS. New
  rows: orthogonal removal with the mean restored (centered), orthogonal removal of the subspace
  Directions-LEACE displaces (span(Sigma D_S^T)), and Gaussian directions matched to SAE-LEACE in
  displaced variance. Per fit it also records the displaced subspace of every map and its principal
  angles to the leading PCs, the MI prefix's eligibility, and two nuisance-orthogonal utilities.
* **SAE construction (`run_sae_configs`).** The core rows with three other TopK constructions, where MI's
  firing band allows N = 32.
* **Paired uncertainty (`figure3_paired`, `transfer_seed_variability`).** Two-stage bootstrap
  (seeds, then pairs) for Figure 3's pair-specific differences, and the fitting-seed spread beside
  the evaluation bootstrap of the frozen-transfer changes. Post-processing of published files.
* **MI prefix eligibility (`mi_prefix_report`).** Does any MI-selected prefix contain latents
  MI cannot score (zero score, firing rate outside [0.02, 0.98])? From the cached SAEs.
* **Diagnostics and label-free predictors (`collect_diagnostics`, `predictor_analysis`).** Selected
  K, ARI, PCA capture and fitting-time label-free quantities for every encoder-corpus cell, and
  whether the label-free ones predict the leakage gap. Post-processing, plus an optional k-means
  stability computed from the cached embeddings.
* **Encoder study on the three-class settings (`run_encoders_3class`).**
* **Five seeds (`run_seed_extension`, `merge_five_seeds`).** Seeds 3-4 for Figure 1, Figure 2, the
  SCOTUS and CAP headline runs and frozen transfer, merged with the published seeds 0-2.
* **Displacement-matched PCA in frozen transfer (`run_transfer_matched`, `transfer_matched_summary`).**
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import os
import shutil
import time

import numpy as np

import alternatives as A
import artifacts
import audit
import corpora
import experiments as E
import focused_additions as FA
import ood
import paths
import protocols as P
import revised as R

FEDLEX = "fedlex_domestic"
#: Figure 1's budget grid: the budgets at which MI prefixes are reported anywhere in the paper.
BUDGETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
SEL = "Direction selection (Directions-LEACE, N = 32)"
OPR = "Operator (the MI-selected set of row 1)"
#: Main Table 3's rows, the new rows, the references and Table 2's label-free baselines:
#: (block, label, condition). Conditions ending in a draw index are averaged into the bare name.
ROWS = (
    (SEL, "1. MI-selected SAE latents", "sae_mi"),
    (SEL, "2. Random SAE latents (MI band)", "sae_random_band"),
    (SEL, "3. Random Gaussian directions, N = 32", "gauss"),
    (SEL, "3m. Random Gaussian directions, displacement-matched N", "gauss_matched"),
    (OPR, "4. SAE activations + Activation-LEACE", "activations"),
    (OPR, "5. SAE directions + orthogonal removal (uncentered, as published)", "directions_orth"),
    (OPR, "5c. SAE directions + orthogonal removal, centered", "directions_orth_centered"),
    (OPR, "5s. Orthogonal removal of span(Sigma D_S^T), centered", "sigma_orth_centered"),
    ("Erasure extent", "6. PCA, displacement-matched", "pca_matched_var"),
    ("References", "Raw", "raw"),
    ("References", "Metadata-LEACE (supervised)", "metadata_leace"),
    ("Context: Table 2's label-free baselines", "PCA-1", "pca_top1"),
    ("Context: Table 2's label-free baselines", "k-means LEACE (silhouette K)", "kmeans_leace"),
)
DRAW_ARMS = ("sae_random_band", "gauss", "gauss_matched")
#: Which optional arm groups each preset fits (raw, Metadata-LEACE, MI SAE-LEACE and the
#: displacement-matched PCA are always fitted).
ARMSETS = {"full": ("operators", "band", "gauss", "context"), "extent": ("context",), "core": ("band",)}
#: Utility metrics per setting (Table 2's), then the nuisance-orthogonal ones of every setting.
UTILITY = {"semeval": ("recall@10", "gold_sim_spearman"),
           "cap": ("topic_purity", "topic_ari", "topic_p10_within", "topic_p10_cross")}
ORTHO = ("within_class_nn10",)
LABEL = {"linear_auc": "linear AUC ↓", "mlp_auc": "MLP AUC ↓", "recall@10": "R@10 ↑",
         "gold_sim_spearman": "graded sim. ρ ↑", "topic_purity": "topic purity ↑", "topic_ari": "topic ARI ↑",
         "topic_p10_within": "topic P@10 within source ↑", "topic_p10_cross": "topic P@10 across sources ↑",
         "within_class_nn10": "within-class NN@10 kept ↑"}


@contextlib.contextmanager
def results_root(path: str):
    """Write under the results root `path` for the block, then restore the previous one."""
    prev = artifacts.BASE_DIR
    artifacts.set_results_dir(path)
    try:
        yield
    finally:
        artifacts.set_results_dir(prev)


def _read(path: str):
    return artifacts.load_result(path) if path and os.path.exists(path) else None


def utility(setting: str) -> tuple:
    return UTILITY.get(setting, ("recall@10",))


def table_metrics(setting: str) -> tuple:
    """Main Table 3's columns: linear AUC, MLP AUC and the setting's first utility metric."""
    return ("linear_auc", "mlp_auc") + utility(setting)[:1]


# --------------------------------------------------------------------------- #
# Settings: the corpus registry, or Fedlex rebuilt from its manifest
# --------------------------------------------------------------------------- #
def load_setting(setting: str, encoder: str) -> tuple:
    """``(data, splits_of(seed), fedlex_run or None)``."""
    if setting == FEDLEX:
        import fedlex as FL
        if encoder != "e5-small":
            raise ValueError("Fedlex was run with E5-small only")
        data, run = FL.fedlex_corpus()
        return data, (lambda s: [FL.fedlex_split(data, run, int(s))]), run
    data = R.load(setting, encoder)
    return data, (lambda s: R._splits(data, int(s))), None


def _reference(setting: str, encoder: str, run=None) -> dict:
    """Per seed, what the published fits of this setting selected and scored, for the reproduction
    check: ``{seed: {"source", "selected": [per fold], "k": [...], "metrics": {condition: metrics}}}``."""
    out = {}
    if setting == FEDLEX:
        for p in run["per_seed"]:
            m = p["metrics"]
            out[int(p["seed"])] = {"source": "fedlex run", "selected": p["selected"], "k": p["pseudo_groups_k"],
                                   "metrics": {"sae_mi": m["sae_leace"], "raw": m["raw"], "metadata_leace": m["metadata_leace"],
                                               "pca_top1": m["pca_top1"], "kmeans_leace": m["kmeans_leace"]}}
        return out
    cov = FA._published(f"alternatives/alternatives_{setting}_{encoder}.json")
    if cov is not None:
        mech = FA._published(f"mechanism/mechanism_{setting}_{encoder}.json") or {"per_seed": []}
        band = _read(os.path.join(paths.EXTENSIONS, "raw", "band", f"band_{setting}_{encoder}.json")) or {"per_seed": []}
        slot = FA._published(f"random_directions/random_directions_{setting}_{encoder}.json") or {"per_seed": []}
        mech_by = {int(p["seed"]): p["mechanisms"] for p in mech["per_seed"]}
        band_by = {int(p["seed"]): p["metrics"] for p in band["per_seed"]}
        slot_by = {int(p["seed"]): p["metrics"] for p in slot["per_seed"]}
        for p in cov["per_seed"]:
            s, m = int(p["seed"]), p["metrics"]
            arms = {c: m[c] for c in ("sae_mi", "pca_matched_var", "raw", "metadata_leace", "pca_top1", "kmeans_leace") if c in m}
            arms.update({c: v for c, v in mech_by.get(s, {}).items() if c in ("activations", "directions_orth")})
            if "rband_mean_n32" in band_by.get(s, {}):
                arms["sae_random_band"] = band_by[s]["rband_mean_n32"]
            if FA.SLOT_CONDITION in slot_by.get(s, {}):
                arms["gauss0"] = slot_by[s][FA.SLOT_CONDITION]
            out[s] = {"source": "coverage", "selected": [d["selected_mi"] for d in p["diagnostics"]],
                      "k": [d["pseudo_groups_k"] for d in p["diagnostics"]], "metrics": arms}
        return out
    idr = FA._published(f"id/id_{setting}_{encoder}.json")
    km = FA._published(f"id/id_kmeans_{setting}_{encoder}.json") or {"per_seed": []}
    km_by = {int(p["seed"]): p["metrics"] for p in km["per_seed"]}
    for p in (idr or {"per_seed": []})["per_seed"]:
        s, m = int(p["seed"]), p["metrics"]
        arms = {"sae_mi": m["sae_leace"], "raw": m["raw"], "metadata_leace": m["metadata_leace"], "pca_top1": m["pca_top1"]}
        if "kmeans_leace" in km_by.get(s, {}):
            arms["kmeans_leace"] = km_by[s]["kmeans_leace"]
        out[s] = {"source": "id run", "selected": p["selected"], "k": p["pseudo_groups_k"], "metrics": arms}
    return out


# --------------------------------------------------------------------------- #
# Operators and their geometry
# --------------------------------------------------------------------------- #
def _orth_basis(rows) -> np.ndarray:
    """An orthonormal basis (columns) of the span of `rows`."""
    U, s, _ = np.linalg.svd(np.asarray(rows, np.float64).T, full_matrices=False)
    return U[:, s > 1e-8 * max(s.max(), 1e-300)]


def centered_orth(rows, mu):
    """``x -> mu + (I - Q Q^T)(x - mu)``: the span of `rows` removed orthogonally, the mean kept."""
    Q, mu = _orth_basis(rows), np.asarray(mu, np.float64)

    def apply(Z):
        Zc = np.asarray(Z, np.float64) - mu
        return (Zc - (Zc @ Q) @ Q.T + mu).astype(np.float32)
    return apply


def _affine_or_none(fn, X_chk: np.ndarray, d: int):
    """``(A, c)`` if `fn` is affine on real documents (to 1e-4), else None."""
    A_, c = ood.affine(fn, d)
    err = float(np.abs(np.asarray(fn(X_chk), np.float64) - (np.asarray(X_chk, np.float64) @ A_ + c)).max())
    return (A_, c) if err < 1e-4 else None


def displaced_basis(Amat: np.ndarray, tol: float = 0.5) -> np.ndarray:
    """Orthonormal basis of the subspace an affine map ``x A + c`` displaces documents into: the row
    space of ``M = I - A``. Every map here is a projection, so its nonzero singular values are >= 1."""
    _, sv, Vt = np.linalg.svd(np.eye(len(Amat)) - Amat)
    return Vt[: int((sv > tol).sum())].T


def capture(B: np.ndarray, P_r: np.ndarray) -> float:
    """Mean squared cosine of the principal angles between span(B) and span(P_r), normalized by dim B:
    the share of the displaced subspace that lies in the top-r principal subspace."""
    return float(np.linalg.norm(P_r.T @ B) ** 2 / max(B.shape[1], 1))


def max_angle_deg(B1: np.ndarray, B2: np.ndarray) -> float:
    """The largest principal angle between two subspaces of equal dimension, in degrees."""
    cos = np.linalg.svd(B1.T @ B2, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(cos.min(), -1.0, 1.0))))


def match_gaussian(X_tr: np.ndarray, G: np.ndarray, target: float) -> int:
    """Fewest leading rows of `G` whose Directions-LEACE map displaces at least `target` of the training
    variance. The displaced share trace(P_N Sigma)/trace(Sigma) grows with the nested span, so a
    bisection finds it."""
    memo = {}

    def share(N):
        if N not in memo:
            memo[N] = A._removed_variance(FA._fit_dirs(X_tr, G[:N]), X_tr)
        return memo[N]
    lo, hi = 1, len(G)
    if share(hi) < target:
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if share(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# --------------------------------------------------------------------------- #
# Nuisance-orthogonal utility
# --------------------------------------------------------------------------- #
def _unit(E_):
    E_ = np.asarray(E_, np.float64)
    return E_ / np.clip(np.linalg.norm(E_, axis=1, keepdims=True), 1e-12, None)


def _knn(U_q, U_pool, k: int, same: bool) -> np.ndarray:
    S = U_q @ U_pool.T
    if same:
        np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, k, axis=1)[:, :k]


def within_class_retention(E_raw, E_, y, k: int = 10) -> float:
    """Share of each document's k nearest same-class neighbours (cosine) that the map keeps: semantic
    structure inside one nuisance class, which an eraser of the nuisance should leave alone."""
    vals = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if len(idx) <= k:
            continue
        a, b = _knn(_unit(E_raw[idx]), _unit(E_raw[idx]), k, True), _knn(_unit(E_[idx]), _unit(E_[idx]), k, True)
        vals.extend(len(np.intersect1d(r, q, assume_unique=True)) / k for r, q in zip(a, b))
    return float(np.mean(vals)) if vals else float("nan")


def topic_precision(E_, y, topics, k: int = 10, same: bool = True) -> float:
    """Share of each document's k nearest neighbours that share its topic, among documents of the same
    source (`same`: nuisance held fixed) or of the other sources (across the nuisance)."""
    U, vals = _unit(E_), []
    for c in np.unique(y):
        q = np.flatnonzero(y == c)
        pool = q if same else np.flatnonzero(y != c)
        if len(pool) <= k:
            continue
        nn = _knn(U[q], U[pool], k, same)
        vals.extend((topics[pool][nn] == topics[q][:, None]).mean(axis=1))
    return float(np.mean(vals)) if vals else float("nan")


def _orthogonal_utility(fits: list, X, y, topics) -> dict:
    """Per condition, the nuisance-orthogonal metrics on the held-out documents, averaged over folds."""
    acc: dict = {}
    for f in fits:
        te = np.asarray(f.split.test_docs)
        X_te, y_te = np.asarray(X[te], np.float32), np.asarray(y)[te]
        t_te = None if topics is None else np.asarray(topics)[te]
        for c, fn in f.erasers.items():
            E_ = np.asarray(fn(X_te), np.float64)
            m = acc.setdefault(c, {})
            m.setdefault("within_class_nn10", []).append(within_class_retention(X_te, E_, y_te))
            if t_te is not None:
                m.setdefault("topic_p10_within", []).append(topic_precision(E_, y_te, t_te, same=True))
                m.setdefault("topic_p10_cross", []).append(topic_precision(E_, y_te, t_te, same=False))
    return {c: {k: float(np.mean(v)) for k, v in m.items()} for c, m in acc.items()}


# --------------------------------------------------------------------------- #
# Table 3, extended, on every setting
# --------------------------------------------------------------------------- #
T3X_FOLDER = "table3x"
#: (setting, encoder, arm preset): Table 3's two settings, the three complementary corpora, and
#: MiniLM on three-source SCOTUS for the intervention-extent control.
T3X_CELLS = (("swiltra_de_fr_it", "e5-small", "full"), ("scotus_oyez_lexis_wiki", "e5-small", "full"),
             ("semeval", "e5-small", "full"), ("cap", "e5-small", "full"), (FEDLEX, "e5-small", "full"),
             ("scotus_oyez_lexis_wiki", "mini", "extent"))
T3X_SEEDS = (0, 1, 2, 3, 4)


class BandTooSmall(ValueError):
    """The SAE has fewer latents in MI's firing band than the budget N. MI then cannot score N latents and the
    band-matched control cannot draw N, so the construction cannot be compared at that N."""

    def __init__(self, seed: int, fold: int, band: int, n: int):
        self.seed, self.fold, self.band, self.n = int(seed), int(fold), int(band), int(n)
        super().__init__(f"seed {seed} fold {fold}: only {band} latents have a training firing rate in MI's band "
                         f"{FA.R1_BAND}, fewer than N = {n}")


def _fold_maps(data, X, sp, fi: int, s: int, kw: dict, groups, n: int, draws: int, fit_device) -> tuple:
    """Every arm of one fit, on one SAE, and the fit's diagnostics."""
    tr = np.asarray(sp.train_docs)
    X_tr, d = np.asarray(X[tr], np.float32), X.shape[1]
    t0 = time.time()
    base = P.fit_base(X, sp, expansion_factor=kw["expansion_factor"], k=kw["k"], n_epochs=kw["n_epochs"],
                      pseudo_groups=data["config"]["pseudo_groups"], device=fit_device)
    sae_seconds = time.time() - t0
    main = R._refit(data, sp, base, top_features=n, erase=kw["erase"], device=fit_device)
    S = [int(j) for j in main.selected]
    D = np.asarray(base.decoder, np.float32)
    scores = np.asarray(base.scores, np.float64)
    freq = (np.asarray(base.feats_train) > 0).mean(axis=0)
    K = int(base.pseudo_groups)
    mu = X_tr.astype(np.float64).mean(axis=0)
    er = {"raw": main.erasers["raw"], "metadata_leace": main.erasers["metadata_leace"], "sae_mi": main.erasers["sae_leace"]}
    diag = {"fold": fi, "selected": S, "pseudo_groups_k": K, "sae_seconds": round(sae_seconds, 1),
            "sae_info": {k: v for k, v in base.sae_info.items() if isinstance(v, (int, float, str))}}
    if "operators" in groups:
        act = R._refit(data, sp, base, top_features=n, erase="activations", device=fit_device)
        orth = R._refit(data, sp, base, top_features=n, erase="directions_orth", device=fit_device)
        er["activations"], er["directions_orth"] = act.erasers["sae_leace"], orth.erasers["sae_leace"]
        diag["operators_same_set"] = [int(j) for j in act.selected] == S == [int(j) for j in orth.selected]
        Sigma = np.cov(X_tr.astype(np.float64), rowvar=False, bias=True)
        er["directions_orth_centered"] = centered_orth(D[S], mu)
        er["sigma_orth_centered"] = centered_orth((Sigma @ D[S].T.astype(np.float64)).T, mu)
    share = A._removed_variance(er["sae_mi"], X_tr)
    r_star = A._matched_rank(X_tr, share)
    er["pca_matched_var"] = A._pca_erasers(X_tr, (r_star,))[r_star]
    pools = FA._pools(freq)
    if "band" in groups:
        if len(pools["band"]) < n:
            raise BandTooSmall(s, fi, len(pools["band"]), n)
        for j in range(draws):
            er[f"sae_random_band{j}"] = FA._fit_dirs(X_tr, D[FA.latent_draw(pools["band"], n, s, fi, j)])
    if "gauss" in groups:
        n_star = []
        for j in range(draws):
            G = FA.gaussian_directions(d, d, s, j)       # nested: its first n rows are R1's N = n draw
            er[f"gauss{j}"] = FA._fit_dirs(X_tr, G[:n])
            N = match_gaussian(X_tr, G, share)
            er[f"gauss_matched{j}"] = FA._fit_dirs(X_tr, G[:N])
            n_star.append(int(N))
        diag["gauss_matched_n"] = n_star
    if "context" in groups:
        er["pca_top1"] = P._fit_pca_removal(X_tr, 1)[0]
        er["kmeans_leace"] = R.kmeans_eraser(X_tr, K, sp.seed)[0]
    # Diagnostics: displacement, displaced subspaces and their principal angles, MI prefix.
    X_chk = X_tr[:64].astype(np.float64)
    Xc = X_tr.astype(np.float64) - mu
    Vt_pca = np.linalg.svd(Xc, full_matrices=False)[2]
    P_n, P_r = Vt_pca[:n].T, Vt_pca[:r_star].T
    shares, sub = {}, {}
    for c, fn in er.items():
        if c == "raw":
            continue
        aff = _affine_or_none(fn, X_chk, d)
        if aff is None:
            shares[c] = None
            continue
        shares[c] = float(A._removed_variance(fn, X_tr))
        if c not in ("metadata_leace",):
            B = displaced_basis(aff[0])
            sub[c] = {"dim": int(B.shape[1]), f"capture_top{n}": capture(B, P_n), "capture_top_rstar": capture(B, P_r)}
            if c == "sae_mi":
                B_mi = B
    if "operators" in groups:
        B_sig = _orth_basis((Sigma @ D[S].T.astype(np.float64)).T)
        sub["sae_mi"]["max_angle_to_span_sigma_D_deg"] = (max_angle_deg(B_mi, B_sig) if B_mi.shape[1] == B_sig.shape[1]
                                                          else None)
        sub["decoder_span"] = {"dim": int(_orth_basis(D[S]).shape[1]), f"capture_top{n}": capture(_orth_basis(D[S]), P_n),
                               "capture_top_rstar": capture(_orth_basis(D[S]), P_r)}
    order = np.argsort(scores)[::-1]
    in_band = (freq >= FA.R1_BAND[0]) & (freq <= FA.R1_BAND[1])
    pos = scores > 0
    diag.update({
        "matched_rank": int(r_star), "sae_displacement_share": float(share), "displacement_share": shares,
        "subspaces": sub,
        "mi_prefix": {str(N): {"zero_score": int((~pos[order[:N]]).sum()), "out_of_band": int((~in_band[order[:N]]).sum())}
                      for N in BUDGETS if N <= len(order)},
        "mi_scores": {"n_latents": int(len(scores)), "n_positive": int(pos.sum()), "band_size": int(in_band.sum()),
                      "positive_out_of_band": int((pos & ~in_band).sum()),
                      "top_n_mass_share": float(scores[order[:n]].sum() / max(scores[pos].sum(), 1e-300))},
        "pool_sizes": {k: int(len(v)) for k, v in pools.items()}})
    return er, diag


def _seed_record(data, X, y, splits, s: int, kw: dict, groups, n: int, draws: int, nonlinear, fit_device,
                 verbose: bool) -> dict:
    fits, diags = [], []
    for fi, sp in enumerate(splits):
        er, dg = _fold_maps(data, X, sp, fi, s, kw, groups, n, draws, fit_device)
        f = R.Erasers(sp, er)
        f.selected, f.config = dg["selected"], {"pseudo_groups_k": dg["pseudo_groups_k"]}
        fits.append(f)
        diags.append(dg)
    conds = tuple(fits[0].erasers)
    res = R.evaluate_fits(fits, X, y, conditions=conds, topics=data.get("topics"), nonlinear=nonlinear)
    for c, m in _orthogonal_utility(fits, X, y, data.get("topics")).items():
        res["metrics"][c].update(m)
    for pre in DRAW_ARMS:
        parts = [c for c in conds if c.startswith(pre) and c[len(pre):].isdigit()]
        if parts:
            res["metrics"][pre] = FA._mean_metrics([res["metrics"][c] for c in parts])
            if all(c in res["hits"] for c in parts):
                res["hits"][pre] = np.mean([res["hits"][c] for c in parts], axis=0).tolist()
    res.update(seed=int(s), diagnostics=diags)
    if verbose:
        m = res["metrics"]
        show = [c for c in ("sae_mi", "sae_random_band", "gauss", "gauss_matched", "activations", "directions_orth",
                            "directions_orth_centered", "sigma_orth_centered", "pca_matched_var") if c in m]
        print("   linear AUC " + "  ".join(f"{c} {m[c]['linear_auc']:.3f}" for c in show))
    return res


def _reproduction(rec: dict, ref) -> dict:
    """The refit against the published fit of the same seed: MI selection, K and the shared rows."""
    if not ref:
        return {"reference": None}
    sel = [sorted(int(j) for j in d["selected"]) for d in rec["diagnostics"]]
    out = {"reference": ref["source"],
           "mi_selection_reproduced": sel == [sorted(int(j) for j in x) for x in ref["selected"]],
           "k_reproduced": [d["pseudo_groups_k"] for d in rec["diagnostics"]] == [int(k) for k in ref["k"]], "diff": {}}
    for c, mref in ref["metrics"].items():
        mine = rec["metrics"].get(c)
        if mine:
            out["diff"][c] = {k: float(mine[k] - mref[k]) for k in ("linear_auc", "mlp_auc", "recall@10", "topic_purity",
                                                                   "gold_sim_spearman") if k in mine and k in mref}
    return out


def run_table3x(setting: str, encoder: str = "e5-small", seeds=T3X_SEEDS, arms: str = "full", n: int = 32,
                draws: int = FA.BAND_DRAWS, nonlinear=("mlp",), fit_device=None, tag: str = "", overwrite: bool = False,
                verbose: bool = True, **overrides) -> dict:
    """Main Table 3's rows and the new ones on one setting, every arm on one SAE per split.

    The SAE, splits and audit are the published runs': `protocols.fit_base` with the recipe and seed,
    the registry's splits (Fedlex: the run's whole-law split, rebuilt and checked), and
    `revised.evaluate_fits`. Each seed is checked against the published fit of that seed -- MI
    selection, K and every row both have -- and checkpointed, so an interrupted run resumes."""
    groups = ARMSETS[arms]
    seeds = [int(s) for s in seeds]
    with artifacts.experiment(T3X_FOLDER):
        name = f"table3x_{setting}_{encoder}{tag}.json"
        path, part = artifacts.artifact_path(name), artifacts.artifact_path(name.replace(".json", ".partial.json"))
        design = {"arms": arms, "n": int(n), "draws": int(draws), "nonlinear": list(nonlinear), "overrides": overrides}
        if os.path.exists(path) and not overwrite:
            res = artifacts.load_result(path)
            if res["design"] != design or res["seeds"] != seeds:
                raise ValueError(f"{path} was run with {res['design']} on seeds {res['seeds']}; pass overwrite=True")
            print(f"[cached] {path}")
            return res
        data, splits_of, run = load_setting(setting, encoder)
        kw = R._kw(overrides)
        X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
        ref = _reference(setting, encoder, run)
        done = {}
        if os.path.exists(part) and not overwrite:
            prev = artifacts.load_result(part)
            if prev.get("design") == design:
                done = {int(p["seed"]): p for p in prev["per_seed"]}
                print(f"[table3x] resuming {setting}/{encoder}{tag}: seeds {sorted(done)} done")
        per_seed = []
        for s in seeds:
            if s in done:
                per_seed.append(done[s])
                continue
            t0 = time.time()
            if verbose:
                print(f"[table3x] {setting}/{encoder}{tag} seed {s}")
            rec = _seed_record(data, X, y, splits_of(s), s, kw, groups, n, draws, nonlinear, fit_device, verbose)
            rec["reproduction"] = _reproduction(rec, ref.get(s))
            rec["seconds"] = round(time.time() - t0, 1)
            per_seed.append(rec)
            artifacts.save_result({"design": design, "per_seed": per_seed}, part)
            if verbose:
                rp = rec["reproduction"]
                print(f"   ({rec['seconds']}s) published fit reproduced: MI selection {rp.get('mi_selection_reproduced')}, "
                      f"K {rp.get('k_reproduced')}; linear AUC diffs "
                      + ", ".join(f"{c} {v['linear_auc']:+.4f}" for c, v in rp.get("diff", {}).items() if "linear_auc" in v))
        lean = [{k: v for k, v in p.items() if k not in ("hits",)} for p in per_seed]
        out = {"experiment": "table3x", "corpus": setting, "encoder": encoder, "label": data.get("label", setting),
               "seeds": seeds, "design": design, "conditions": list(per_seed[0]["metrics"]),
               "metrics": list(table_metrics(setting)), "utility": list(utility(setting)) + list(ORTHO),
               "per_seed": per_seed, "agg": R.aggregate(lean),
               "config": {"recipe": kw, "selection": data["config"]["selection"],
                          "pseudo_groups": data["config"]["pseudo_groups"], "split": E._split_policy(data),
                          "audit": audit.PROTOCOL},
               "environment": _environment(fit_device)}
        artifacts.save_result(out, name)
        if os.path.exists(part):
            os.remove(part)
        return out


def _environment(fit_device) -> dict:
    import sklearn
    import torch
    return {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "torch": torch.__version__,
            "sae_device": fit_device or ("cuda" if torch.cuda.is_available() else "cpu")}


def load_table3x(setting: str, encoder: str = "e5-small", tag: str = ""):
    return _read(R.raw_path(T3X_FOLDER, f"table3x_{setting}_{encoder}{tag}.json"))


# ---- statistics shared by the tables ---------------------------------------- #
def seed_stats(v, level: float = 0.95) -> dict:
    """Mean, sample SD, signs and a t-interval over seeds (df = n - 1) of per-seed values."""
    from scipy import stats
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], np.float64)
    if not len(v):
        return {"n": 0, "mean": None}
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    half = (float(stats.t.ppf((1 + level) / 2, len(v) - 1)) * sd / np.sqrt(len(v))) if len(v) > 1 else float("nan")
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": sd, "n_pos": int((v > 0).sum()), "n_neg": int((v < 0).sum()),
            "t_ci": [float(v.mean() - half), float(v.mean() + half)]}


def hier_boot(diffs_by_seed, n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> dict:
    """Two-stage bootstrap of a mean per-unit difference: seeds resampled, then units within each seed.
    Both sources of variation -- the fitted maps (seeds) and the evaluation sample (pairs) -- enter."""
    rng = np.random.default_rng(seed)
    d = [np.asarray(x, np.float64) for x in diffs_by_seed if len(x)]
    if not d:
        return {"mean": None, "ci": [None, None], "n_seeds": 0}
    inner = np.stack([x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1) for x in d])
    boots = inner[rng.integers(0, len(d), (n_boot, len(d))), np.arange(n_boot)[:, None]].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
    return {"mean": float(np.mean([x.mean() for x in d])), "ci": [float(lo), float(hi)], "n_seeds": len(d),
            "n_units": [int(len(x)) for x in d]}


def _fmt(v, signed: bool = False, d: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"{v:+.{d}f}" if signed else f"{v:.{d}f}"


def _pm(st: dict, signed: bool = False) -> str:
    if not st or st.get("mean") is None:
        return "--"
    return _fmt(st["mean"], signed) + (f" ± {st['sd']:.3f}" if st["n"] > 1 else "")


def _values(res: dict, cond: str, metric: str) -> list:
    return [p["metrics"].get(cond, {}).get(metric) for p in res["per_seed"]]


def paired(res: dict, cond: str, anchor: str, metric: str, n_boot: int = 2000) -> dict:
    """Per-seed paired difference of `cond` to `anchor` on one metric, with the seed statistics and,
    for R@10, the two-stage bootstrap over seeds and held-out pairs."""
    a, b = _values(res, cond, metric), _values(res, anchor, metric)
    out = seed_stats([x - z if x is not None and z is not None else None for x, z in zip(a, b)])
    if metric == "recall@10" and all(cond in p.get("hits", {}) and anchor in p.get("hits", {}) for p in res["per_seed"]):
        out["boot"] = hier_boot([np.asarray(p["hits"][cond]) - np.asarray(p["hits"][anchor]) for p in res["per_seed"]],
                                n_boot)
    return out


def _cell_line(st: dict) -> str:
    if not st or st.get("mean") is None:
        return "--"
    ci = st.get("boot", {}).get("ci") if st.get("boot") else st.get("t_ci")
    tag = "" if not st.get("boot") else "ᵇ"
    ci_s = "" if not ci or ci[0] is None or not np.isfinite(ci[0]) else f" [{ci[0]:+.3f}, {ci[1]:+.3f}]{tag}"
    return f"{st['mean']:+.3f}{ci_s} ({st['n_pos']}/{st['n_neg']})"


def table_table3x(cells=T3X_CELLS, tag: str = "", n_boot: int = 2000) -> str:
    """Markdown: Main Table 3's layout per setting (seed mean ± SD), then each row's paired difference
    to row 1 (95% interval: two-stage bootstrap for R@10, t over seeds otherwise; signs over seeds)."""
    lines = []
    for setting, encoder, _ in cells:
        r = load_table3x(setting, encoder, tag)
        if r is None:
            lines += [f"### {setting} / {encoder}: not run", ""]
            continue
        mets = list(r["metrics"]) + [m for m in utility(setting)[1:]]
        rows = [x for x in ROWS if x[2] in r["agg"]]
        rep = [p["reproduction"] for p in r["per_seed"]]
        lines += [f"### {r['label']} / {encoder} (seeds {r['seeds']}; {r['design']['draws']} draws per random arm)", "",
                  "| | " + " | ".join(LABEL[m] for m in mets) + " |", "|---|" + "---|" * len(mets)]
        block = None
        for b, label, c in rows:
            if b != block:
                block = b
                lines.append(f"| *{b}* |" + " |" * len(mets))
            lines.append(f"| {label} | " + " | ".join(_pm(seed_stats(_values(r, c, m))) for m in mets) + " |")
        lines += ["", "Paired difference to row 1 (MI-selected SAE-LEACE, same SAE):", "",
                  "| | " + " | ".join(f"Δ {LABEL[m]}" for m in mets) + " |", "|---|" + "---|" * len(mets)]
        for b, label, c in rows:
            if c == "sae_mi":
                continue
            lines.append(f"| {label} | " + " | ".join(_cell_line(paired(r, c, "sae_mi", m, n_boot)) for m in mets) + " |")
        ok = [x.get("mi_selection_reproduced") for x in rep]
        lines += ["", f"Reproduction of the published fit per seed ({rep[0].get('reference')}): MI selection {ok}, "
                      f"K {[x.get('k_reproduced') for x in rep]}. Largest |linear-AUC difference| on the shared rows: "
                      + _fmt(max([abs(v['linear_auc']) for x in rep for v in x.get('diff', {}).values() if 'linear_auc' in v]
                                 or [float('nan')])) + f". Environment: {r['environment']}.",
                  "ᵇ two-stage bootstrap (seeds, then held-out pairs). In brackets: seeds with Δ > 0 / Δ < 0.", ""]
    md = "\n".join(lines)
    with artifacts.experiment(T3X_FOLDER):
        FA._save_md(md, f"table_table3x{tag}.md")
    return md


#: Item 1's comparison: SAE-LEACE against equally broad PCA and Table 2's label-free baselines.
EXTENT_ROWS = (("SAE-LEACE (MI, N = 32)", "sae_mi"), ("PCA, displacement-matched", "pca_matched_var"),
               ("PCA-1", "pca_top1"), ("k-means LEACE", "kmeans_leace"), ("Metadata-LEACE", "metadata_leace"),
               ("Raw", "raw"))
EXTENT_CELLS = (("cap", "e5-small"), ("semeval", "e5-small"), ("scotus_oyez_lexis_wiki", "mini"), (FEDLEX, "e5-small"),
                ("swiltra_de_fr_it", "e5-small"), ("scotus_oyez_lexis_wiki", "e5-small"))


def table_extent(cells=EXTENT_CELLS, n_boot: int = 2000) -> str:
    """Markdown: per setting, SAE-LEACE beside displacement-matched PCA, PCA-1, k-means LEACE and the
    references, on leakage and every utility metric, with matched PCA's paired difference to SAE-LEACE."""
    lines = []
    for setting, encoder in cells:
        r = load_table3x(setting, encoder)
        if r is None:
            lines += [f"### {setting} / {encoder}: not run", ""]
            continue
        mets = ["linear_auc", "mlp_auc"] + list(utility(setting)) + list(ORTHO)
        ranks = [d["matched_rank"] for p in r["per_seed"] for d in p["diagnostics"]]
        share = [d["sae_displacement_share"] for p in r["per_seed"] for d in p["diagnostics"]]
        lines += [f"### {r['label']} / {encoder}: SAE-LEACE displaces {min(share):.3f}-{max(share):.3f} of the training "
                  f"variance; matched PCA rank {min(ranks)}-{max(ranks)}", "",
                  "| | " + " | ".join(LABEL[m] for m in mets) + " |", "|---|" + "---|" * len(mets)]
        for label, c in EXTENT_ROWS:
            if c in r["agg"]:
                lines.append(f"| {label} | " + " | ".join(_pm(seed_stats(_values(r, c, m))) for m in mets) + " |")
        lines.append("| *Matched PCA − SAE-LEACE* | " + " | ".join(_cell_line(paired(r, "pca_matched_var", "sae_mi", m, n_boot))
                                                           for m in mets) + " |")
        lines.append("")
    md = "\n".join(lines)
    with artifacts.experiment(T3X_FOLDER):
        FA._save_md(md, "table_extent.md")
    return md


def table_operators(cells=T3X_CELLS) -> str:
    """Markdown: the displaced subspace of each Directions-LEACE arm -- its dimension, the training variance
    it displaces, and how much of it lies in the top-N and top-r* principal subspaces (mean squared cosine of
    the principal angles) -- and the check that SAE-LEACE displaces exactly span(Sigma D_S^T)."""
    lines = []
    arms = (("MI-selected SAE latents", "sae_mi"), ("Random SAE latents (MI band)", "sae_random_band"),
            ("Gaussian, N = 32", "gauss"), ("Gaussian, displacement-matched", "gauss_matched"),
            ("SAE decoder span itself (not displaced)", "decoder_span"),
            ("Orthogonal removal of span(Sigma D_S^T)", "sigma_orth_centered"),
            ("Orthogonal removal of span(D_S), centered", "directions_orth_centered"),
            ("PCA, displacement-matched", "pca_matched_var"))
    for setting, encoder, _ in cells:
        r = load_table3x(setting, encoder)
        if r is None:
            continue
        n = r["design"]["n"]
        diags = [d for p in r["per_seed"] for d in p["diagnostics"]]
        lines += [f"### {r['label']} / {encoder}", "",
                  f"| arm | dim | displaced share | capture by top-{n} PCs | capture by top-r* PCs |", "|---|---|---|---|---|"]
        for label, c in arms:
            def grab(key, c=c):
                v = []
                for dg in diags:
                    sub = dg["subspaces"]
                    keys = [c] if c in sub else [k for k in sub if k.startswith(c) and k[len(c):].isdigit()]
                    v += [sub[k][key] for k in keys if sub[k].get(key) is not None]
                return v
            dims = grab("dim")
            if not dims:
                continue
            sh = [dg["displacement_share"].get(c) for dg in diags if dg["displacement_share"].get(c) is not None]
            if not sh:
                sh = [dg["displacement_share"][k] for dg in diags for k in dg["displacement_share"]
                      if k.startswith(c) and k[len(c):].isdigit() and dg["displacement_share"][k] is not None]
            lines.append(f"| {label} | {min(dims)}-{max(dims)} | {_fmt(np.mean(sh) if sh else None)} | "
                         f"{_fmt(np.mean(grab(f'capture_top{n}')))} | {_fmt(np.mean(grab('capture_top_rstar')))} |")
        ang = [dg["subspaces"]["sae_mi"].get("max_angle_to_span_sigma_D_deg") for dg in diags]
        ang = [a for a in ang if a is not None]
        rs = [dg["matched_rank"] for dg in diags]
        nst = [x for dg in diags for x in dg.get("gauss_matched_n", [])]
        lines += ["", f"Matched PCA rank r* {min(rs)}-{max(rs)}; displacement-matched Gaussian N {min(nst) if nst else '--'}-"
                      f"{max(nst) if nst else '--'}. Largest principal angle between SAE-LEACE's displaced subspace and "
                      f"span(Sigma D_S^T): {max(ang):.2e} degrees." if ang else "", ""]
    md = "\n".join(lines)
    with artifacts.experiment(T3X_FOLDER):
        FA._save_md(md, "table_operators.md")
    return md


# --------------------------------------------------------------------------- #
# SAE construction: other dictionary sizes and sparsities
# --------------------------------------------------------------------------- #
#: Two alternatives to the paper's TopK SAE (8d latents, k = 32): sparser, and a larger dictionary.
#: Three other TopK constructions: a larger dictionary at the paper's density (k / latents = 1/96), denser
#: codes at the paper's dictionary size, and a smaller dictionary at the paper's density. MI scores only
#: latents with a training firing rate in [0.02, 0.98], and how many qualify swings with k / latents. On
#: SCOTUS-3 the paper's 8d, k = 32 has 64-80 such latents per fit, 4d, k = 16 about 15-27, 8d, k = 16 and
#: 16d, k = 32 3-4, 16d, k = 64 about 320 and 8d, k = 64 about 1,500. A construction with fewer than
#: N = 32 in some fit is skipped and reported (`BandTooSmall`): 4d, k = 16 on SCOTUS-3.
SAE_CONFIGS = ({"expansion_factor": 16, "k": 64}, {"expansion_factor": 8, "k": 64}, {"expansion_factor": 4, "k": 16})
SAE_SETTINGS = (("swiltra_de_fr_it", "e5-small"), ("scotus_oyez_lexis_wiki", "e5-small"))
SAE_SEEDS = (0, 1, 2)


def sae_tag(cfg: dict) -> str:
    return f"_sae_e{cfg['expansion_factor']}_k{cfg['k']}"


SAE_SKIPPED = "sae_configs_skipped.json"


def run_sae_configs(settings=SAE_SETTINGS, configs=SAE_CONFIGS, seeds=SAE_SEEDS, draws: int = FA.BAND_DRAWS,
                    nonlinear=("mlp",), fit_device=None, verbose: bool = True, **overrides) -> dict:
    """The core rows -- MI SAE-LEACE, band-matched random latents and the PCA matched to that SAE's
    displacement -- with each alternative SAE, on the same splits and audit as `run_table3x`.

    A construction whose MI band holds fewer than N latents in some fit cannot be compared at N; it is
    recorded in ``raw/table3x/sae_configs_skipped.json`` (and shown in the table) and the run moves on."""
    out, skipped = {}, []
    with artifacts.experiment(T3X_FOLDER):
        prev_path = artifacts.artifact_path(SAE_SKIPPED)
    prev = artifacts.load_result(prev_path)["skipped"] if os.path.exists(prev_path) else []
    for s, e in settings:
        for c in configs:
            done = [r for r in prev if r["setting"] == s and r["encoder"] == e and all(r.get(k) == v for k, v in c.items())]
            if done:                     # recorded as skipped by an earlier run: not refitted
                skipped.extend(done)
                print(f"[sae configs] {s}/{e}, {c['expansion_factor']}d latents, k = {c['k']}: skipped earlier "
                      f"({prev_path})")
                continue
            try:
                out[(s, e, sae_tag(c))] = run_table3x(s, e, seeds=seeds, arms="core", draws=draws, nonlinear=nonlinear,
                                                      fit_device=fit_device, tag=sae_tag(c), verbose=verbose,
                                                      **{**overrides, **c})
            except BandTooSmall as err:
                skipped.append({"setting": s, "encoder": e, **c, "seed": err.seed, "fold": err.fold,
                                "band_size": err.band, "n": err.n})
                print(f"[sae configs] {s}/{e}, {c['expansion_factor']}d latents, k = {c['k']}: skipped -- {err}. "
                      "MI cannot score N latents on this SAE, so the construction is not comparable at N.")
    with artifacts.experiment(T3X_FOLDER):
        artifacts.save_result({"skipped": skipped, "band": list(FA.R1_BAND)}, SAE_SKIPPED)
    return out


def _restrict(res: dict, seeds) -> dict:
    return {**res, "per_seed": [p for p in res["per_seed"] if int(p["seed"]) in set(seeds)], "seeds": list(seeds)}


def table_sae_configs(settings=SAE_SETTINGS, configs=SAE_CONFIGS, seeds=SAE_SEEDS, n_boot: int = 2000) -> str:
    """Markdown: per setting and SAE, the core rows on the same seeds, with each SAE's displacement and
    matched PCA rank, and the paired differences that carry Table 3's conclusions."""
    lines = []
    skipped = (_read(R.raw_path(T3X_FOLDER, SAE_SKIPPED)) or {}).get("skipped", [])
    for setting, encoder in settings:
        base = load_table3x(setting, encoder)
        variants = [("TopK, 8d latents, k = 32 (the paper's)", base and _restrict(base, seeds))]
        variants += [(f"TopK, {c['expansion_factor']}d latents, k = {c['k']}", load_table3x(setting, encoder, sae_tag(c)))
                     for c in configs]
        mets = list(table_metrics(setting))
        lines += [f"### {setting} / {encoder} (seeds {list(seeds)})", "",
                  "| SAE | arm | " + " | ".join(LABEL[m] for m in mets) + " | displaced share | matched rank |",
                  "|---|---|" + "---|" * (len(mets) + 2)]
        for name, r in variants:
            if not r or not r["per_seed"]:
                skip = next((z for z in skipped if z["setting"] == setting and z["encoder"] == encoder
                             and name.startswith(f"TopK, {z['expansion_factor']}d latents, k = {z['k']}")), None)
                why = ("not run" if skip is None else f"infeasible: seed {skip['seed']} has only {skip['band_size']} "
                       f"latents in MI's band (< N = {skip['n']})")
                lines.append(f"| {name} | {why} |" + " |" * (len(mets) + 2))
                continue
            dg = [d for p in r["per_seed"] for d in p["diagnostics"]]
            sh, rk = np.mean([d["sae_displacement_share"] for d in dg]), [d["matched_rank"] for d in dg]
            for label, c in (("SAE-LEACE (MI)", "sae_mi"), ("random SAE latents (MI band)", "sae_random_band"),
                             ("PCA, displacement-matched", "pca_matched_var")):
                lines.append(f"| {name} | {label} | " + " | ".join(_pm(seed_stats(_values(r, c, m))) for m in mets)
                             + (f" | {sh:.3f} | {min(rk)}-{max(rk)} |" if c == "sae_mi" else " | | |"))
            for label, c in (("random − MI", "sae_random_band"), ("matched PCA − MI", "pca_matched_var")):
                lines.append(f"| {name} | *Δ {label}* | " + " | ".join(_cell_line(paired(r, c, "sae_mi", m, n_boot))
                                                                    for m in mets) + " | | |")
        lines.append("")
    md = "\n".join(lines)
    with artifacts.experiment(T3X_FOLDER):
        FA._save_md(md, "table_sae_configs.md")
    return md


# --------------------------------------------------------------------------- #
# Paired uncertainty for the utility claims (post-processing of published runs)
# --------------------------------------------------------------------------- #
PAIRED_FOLDER = "paired_utility"
FIG3_SETTINGS = ("swiltra_de_fr_it", "scotus_oyez_lexis_wiki")
FIG3_METHODS = ("pca_top1", "kmeans_leace", "sae_leace", "metadata_leace")
FIG3_REFS = ("metadata_leace", "raw")
#: Figure 4 A-B's five donors and the published summary holding each.
FIG4_DONORS = (("e5-small", "cap", "ood/ood_summary_kmeans.json"),
               ("e5-small", "scotus_oyez_lexis_wiki", "combinations/ood_summary.json"),
               ("e5-small", "swiltra_de_fr_it", "combinations/ood_summary.json"),
               ("mini", "cap", "ood/ood_summary_kmeans.json"),
               ("mini", "scotus_oyez_lexis_wiki", "combinations/ood_summary.json"))
FIG4_TRANSFORMS = ("pca_top1", "kmeans_leace", "sae_leace", "metadata_leace")


def figure3_paired(settings=FIG3_SETTINGS, encoder: str = "e5-small", methods=FIG3_METHODS, refs=FIG3_REFS,
                   n_boot: int = 2000) -> dict:
    """Figure 3's pair-specific R@10 differences with paired uncertainty.

    From the published ID runs, which keep every held-out pair's outcome (k-means LEACE from the
    `run_id_kmeans` runs on the same splits): per pair type and method, the per-seed difference to the
    reference and a two-stage bootstrap over seeds and pairs. The saved per-type R@10 must be
    reproduced from the hits, and for SCOTUS -- whose figure cohort is the coverage run, seeds 0-2 --
    the ID run's per-type R@10 must equal the coverage run's on those seeds."""
    out = {"n_boot": n_boot, "settings": {}}
    for key in settings:
        idr = FA._published(f"id/id_{key}_{encoder}.json")
        ikm = FA._published(f"id/id_kmeans_{key}_{encoder}.json")
        cov = FA._published(f"alternatives/alternatives_{key}_{encoder}.json")
        data = corpora.load(key)
        y = np.asarray(data["source"])
        by_id, by_km = {int(p["seed"]): p for p in idr["per_seed"]}, {int(p["seed"]): p for p in ikm["per_seed"]}
        seeds = sorted(set(by_id) & set(by_km))
        hits, kinds, err = {}, {}, 0.0
        for s in seeds:
            kind = np.concatenate([R.pair_kinds(y, sp.test_pairs_global) for sp in R._splits(data, s)])
            h = {c: np.asarray(v, np.float64) for c, v in by_id[s]["hits"].items()}
            h["kmeans_leace"] = np.asarray(by_km[s]["hits"]["kmeans_leace"], np.float64)
            for c, m in list(by_id[s]["metrics"].items()) + [("kmeans_leace", by_km[s]["metrics"]["kmeans_leace"])]:
                if c in h and "recall@10_by_pair" in m:
                    if len(h[c]) != len(kind):
                        raise ValueError(f"{key} seed {s} {c}: {len(h[c])} hits for {len(kind)} pairs")
                    err = max(err, max(abs(h[c][kind == t].mean() - v) for t, v in m["recall@10_by_pair"].items()))
            hits[s], kinds[s] = h, kind
        cohort = None
        if cov is not None:
            pairs_ = {"sae_leace": "sae_mi", "pca_top1": "pca_top1", "raw": "raw", "metadata_leace": "metadata_leace"}
            cohort = max(abs(by_id[int(p["seed"])]["metrics"][a]["recall@10_by_pair"][t] - p["metrics"][b]["recall@10_by_pair"][t])
                         for p in cov["per_seed"] if int(p["seed"]) in by_id for a, b in pairs_.items()
                         for t in p["metrics"][b]["recall@10_by_pair"])
        types = sorted(np.unique(kinds[seeds[0]]).tolist()) + ["overall"]
        rows = []
        for m in methods:
            for ref in refs:
                if m == ref:
                    continue
                for t in types:
                    diffs = [(hits[s][m] - hits[s][ref]) if t == "overall" else (hits[s][m] - hits[s][ref])[kinds[s] == t]
                             for s in seeds]
                    st = seed_stats([float(x.mean()) for x in diffs])
                    rows.append({"method": m, "ref": ref, "pair": t, **st, "per_seed": [float(x.mean()) for x in diffs],
                                 "boot": hier_boot(diffs, n_boot)})
        out["settings"][key] = {"seeds": seeds, "rows": rows, "max_err_saved_by_pair": float(err),
                                "max_diff_id_vs_coverage_cohort": cohort,
                                "n_pairs": {t: int((kinds[seeds[0]] == t).sum()) for t in types[:-1]}}
    with artifacts.experiment(PAIRED_FOLDER):
        artifacts.save_result(out, "figure3_paired.json")
    return out


def table_figure3_paired(res: dict) -> str:
    lab = {"pca_top1": "PC-1", "kmeans_leace": "k-means LEACE", "sae_leace": "SAE-LEACE", "metadata_leace": "Metadata-LEACE"}
    lines = []
    for key, v in res["settings"].items():
        refs = sorted({r["ref"] for r in v["rows"]}, key=lambda z: z != "metadata_leace")
        lines += [f"### {key} (seeds {v['seeds']}; pairs per type {v['n_pairs']})", "",
                  "| method | pair type | " + " | ".join(f"Δ R@10 vs {lab.get(r, r)}" for r in refs) + " |",
                  "|---|---|" + "---|" * len(refs)]
        for m in dict.fromkeys(r["method"] for r in v["rows"]):
            for t in dict.fromkeys(r["pair"] for r in v["rows"]):
                cells = []
                for ref in refs:
                    row = next((r for r in v["rows"] if r["method"] == m and r["ref"] == ref and r["pair"] == t), None)
                    cells.append("--" if row is None else
                                 f"{row['mean']:+.3f} [{row['boot']['ci'][0]:+.3f}, {row['boot']['ci'][1]:+.3f}] "
                                 f"({row['n_pos']}/{row['n_neg']})")
                lines.append(f"| {lab.get(m, m)} | {t} | " + " | ".join(cells) + " |")
        lines += ["", f"Two-stage bootstrap (seeds, then held-out pairs), {res['n_boot']} resamples; in brackets the seeds "
                      f"with Δ > 0 / Δ < 0. Saved per-type R@10 reproduced from the hits to {v['max_err_saved_by_pair']:.1e}; "
                      f"ID run vs the figure's coverage cohort (seeds 0-2): largest difference "
                      f"{_fmt(v['max_diff_id_vs_coverage_cohort']) if v['max_diff_id_vs_coverage_cohort'] is not None else '--'}.",
                  ""]
    md = "\n".join(lines)
    with artifacts.experiment(PAIRED_FOLDER):
        FA._save_md(md, "table_figure3_paired.md")
    return md


def _verdict(per_seed, ci) -> str:
    per_seed = np.asarray(per_seed)
    excl = ci[0] > 0 or ci[1] < 0
    same = (per_seed > 0).all() or (per_seed < 0).all()
    return {(True, True): "stable: every fitted map, and the evaluation bootstrap",
            (True, False): "evaluation-stable, but the fitted maps disagree in sign",
            (False, True): "same sign in every fitted map, within evaluation noise",
            (False, False): "not distinguishable from 0"}[(bool(excl), bool(same))]


def transfer_seed_variability(donors=FIG4_DONORS, transforms=FIG4_TRANSFORMS, summaries: dict | None = None,
                              name: str = "transfer_seed_variability.json") -> dict:
    """Frozen transfer, per donor and map: the change from Raw for each fitted map (seed) beside the
    published evaluation bootstrap, and whether a change is stable across the fitted maps."""
    rows = []
    for enc, donor, rel in donors:
        key = f"{enc}/{donor}"
        v = (summaries or {}).get(key) or FA._published(rel)["settings"][key]
        legal, sts = list(v["legal"]), (v["sts"][0] if isinstance(v["sts"], list) else v["sts"])
        for c in transforms:
            if c not in v["scores"]:
                continue
            per_legal = np.mean([np.asarray(v["scores"][c][t]["values"]) - v["scores"]["raw"][t]["mean"] for t in legal], axis=0)
            per_sts = np.asarray(v["scores"][c][sts]["values"]) - v["scores"]["raw"][sts]["mean"]
            for target, per, ch in (("legal macro nDCG@10", per_legal, v["changes"][c]["legal_macro"]),
                                    ("STSBenchmark Spearman", per_sts, v["changes"][c][sts])):
                rows.append({"encoder": enc, "donor": donor, "transform": c, "target": target,
                             "per_seed": [float(x) for x in per], **seed_stats(per), "boot_delta": float(ch["delta"]),
                             "boot_ci": [float(x) for x in ch["ci"]], "verdict": _verdict(per, ch["ci"])})
    out = {"rows": rows}
    with artifacts.experiment(PAIRED_FOLDER):
        artifacts.save_result(out, name)
    return out


def table_transfer_variability(res: dict, name: str = "table_transfer_variability.md") -> str:
    lab = {"pca_top1": "PC-1", "pca_matched_var": "PCA, displacement-matched", "kmeans_leace": "k-means LEACE",
           "sae_leace": "SAE-LEACE", "metadata_leace": "Metadata-LEACE"}
    lines = ["| encoder | donor | map | target | Δ per fitted map (seeds) | seed SD | bootstrap Δ [95% CI] | reading |",
             "|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        lines.append(f"| {r['encoder']} | {r['donor']} | {lab.get(r['transform'], r['transform'])} | {r['target']} | "
                     + ", ".join(f"{x:+.3f}" for x in r["per_seed"]) + f" | {r['sd']:.3f} | {r['boot_delta']:+.3f} "
                     f"[{r['boot_ci'][0]:+.3f}, {r['boot_ci'][1]:+.3f}] | {r['verdict']} |")
    md = "\n".join(lines)
    with artifacts.experiment(PAIRED_FOLDER):
        FA._save_md(md, name)
    return md


# --------------------------------------------------------------------------- #
# MI prefix eligibility
# --------------------------------------------------------------------------- #
_ENC = "(e5-small|e5-base|e5-large|mini|gist-small|gist-base|gist-large|mpnet|nomic-v2|mxb)"


def _prefix_row(fit: str, scores, freq, budgets, source: str) -> dict:
    scores, freq = np.asarray(scores, np.float64), np.asarray(freq, np.float64)
    order = np.argsort(scores)[::-1]
    in_band, pos = (freq >= FA.R1_BAND[0]) & (freq <= FA.R1_BAND[1]), scores > 0
    return {"fit": fit, "source": source, "n_latents": int(len(scores)), "n_positive": int(pos.sum()),
            "band_size": int(in_band.sum()), "positive_out_of_band": int((pos & ~in_band).sum()),
            "zero_in_band": int((~pos & in_band).sum()),
            "prefix": {str(N): {"zero_score": int((~pos[order[:N]]).sum()), "out_of_band": int((~in_band[order[:N]]).sum())}
                       for N in budgets if N <= len(order)}}


def mi_prefix_report(cells=T3X_CELLS, folders=(os.path.join(paths.EXTENSIONS, FA.SAE_FOLDER),),
                     budgets=BUDGETS) -> dict:
    """Does any MI-selected prefix contain a latent MI cannot score? MI gives latents outside the firing-
    rate band [0.02, 0.98] a zero score instead of dropping them, so a prefix longer than the number of
    positive scores would include zero-scored (possibly out-of-band) latents. Read from every cached SAE
    (Sections 3-4) and every fit of `run_table3x`, at every budget the paper reports."""
    import re
    rows = []
    for folder in folders:
        for p in sorted(glob.glob(os.path.join(folder, "sae_*.npz"))):
            with np.load(p) as z:
                rows.append(_prefix_row(os.path.basename(p)[4:-4], z["scores"], z["freq"], budgets, p))
    for setting, encoder, _ in cells:
        r = load_table3x(setting, encoder)
        for q in (r or {"per_seed": []})["per_seed"]:
            for d in q["diagnostics"]:
                rows.append({"fit": f"{setting}_{encoder}_s{q['seed']}_f{d['fold']}", "source": "table3x",
                             **{k: d["mi_scores"][k] for k in ("n_latents", "n_positive", "band_size", "positive_out_of_band")},
                             "prefix": d["mi_prefix"]})
    for row in rows:
        m = re.match(rf"^(.*)_{_ENC}_s(\d+)_f(\d+)$", row["fit"])
        row["setting"], row["encoder"] = (m.group(1), m.group(2)) if m else (row["fit"], "?")
    out = {"budgets": list(budgets), "band": list(FA.R1_BAND), "rows": rows}
    with artifacts.experiment(PAIRED_FOLDER):
        artifacts.save_result(out, "mi_prefix_eligibility.json")
    return out


def table_mi_prefix(res: dict) -> str:
    budgets = [str(b) for b in res["budgets"]]
    groups: dict = {}
    for r in res["rows"]:
        groups.setdefault((r["setting"], r["encoder"]), []).append(r)
    lines = ["Largest count over the fits of each setting: zero-scored / out-of-band latents in the MI top-N.", "",
             "| setting | encoder | fits | positive scores | " + " | ".join(f"N = {b}" for b in budgets)
             + " | positive & out of band |", "|---|---|---|---|" + "---|" * (len(budgets) + 1)]
    worst = 0
    for (s, e), rs in sorted(groups.items()):
        cells = []
        for b in budgets:
            z = max(r["prefix"].get(b, {}).get("zero_score", 0) for r in rs)
            o = max(r["prefix"].get(b, {}).get("out_of_band", 0) for r in rs)
            worst = max(worst, z, o)
            cells.append(f"{z} / {o}")
        pos = [r["n_positive"] for r in rs]
        lines.append(f"| {s} | {e} | {len(rs)} | {min(pos)}-{max(pos)} | " + " | ".join(cells)
                     + f" | {max(r['positive_out_of_band'] for r in rs)} |")
    lines += ["", ("Answer: no MI-selected prefix at any reported budget contains a zero-scored or out-of-band latent "
                   "in these fits." if worst == 0 else
                   "Answer: some MI-selected prefixes contain zero-scored or out-of-band latents (non-zero cells above)."), ""]
    md = "\n".join(lines)
    with artifacts.experiment(PAIRED_FOLDER):
        FA._save_md(md, "table_mi_prefix.md")
    return md


# --------------------------------------------------------------------------- #
# Diagnostics and label-free predictors of success
# --------------------------------------------------------------------------- #
DIAG_FOLDER = "diagnostics"
STUDY_CORPORA = ("semeval", "swiltra", "bills_tweets", "scotus_oyez_lexis", "cap")
#: Table 2's cells: the six corpora on E5-small, and MiniLM on three-source SCOTUS.
HEADLINE_CELLS = (("semeval", "e5-small"), ("swiltra_de_fr_it", "e5-small"), ("bills_tweets", "e5-small"),
                  ("scotus_oyez_lexis_wiki", "e5-small"), ("cap", "e5-small"), (FEDLEX, "e5-small"),
                  ("scotus_oyez_lexis_wiki", "mini"))
#: Fitting-time quantities that need no nuisance labels (the SAE's pseudo-groups are k-means groups,
#: so its MI scores are label-free too). The spectrum is of the whole corpus, as in Figure D6.
PREDICTORS = (("k_auto", "selected K (silhouette)"), ("silhouette", "silhouette at the selected K"),
              ("silhouette_margin", "silhouette margin over the next-best K"),
              ("pc1_var_share", "PC1 variance share"), ("top5_var_share", "top-5 PCs variance share"),
              ("eigengap_12", "eigengap λ1/λ2"), ("eff_rank_share", "effective rank / d"),
              ("kmeans_stability", "k-means stability across seeds (ARI)"))
#: Label-based: the leakage gap of each label-free eraser to Metadata-LEACE (held-out linear AUC,
#: positive = more leakage than the supervised eraser) and the descriptive diagnostics of Table D12.
OUTCOMES = (("gap_sae", "SAE-LEACE gap"), ("gap_kmeans", "k-means LEACE gap"), ("gap_pc1", "PCA-1 gap"),
            ("ari_auto", "ARI(k-means groups, classes)"), ("capture_c1", "nuisance subspace in top C − 1 PCs"),
            ("rank_90", "PCs holding 90% of the nuisance subspace"))
#: Which predictor is the a-priori test for which outcome (the rest of the grid is exploratory).
PRIMARY = (("silhouette", "ari_auto"), ("silhouette", "gap_kmeans"), ("kmeans_stability", "gap_kmeans"),
           ("pc1_var_share", "gap_pc1"), ("eigengap_12", "gap_pc1"), ("silhouette", "gap_sae"))


def _folds_k(diags: list) -> dict:
    """Silhouette K, the silhouette at it and its margin over the best other K, averaged over a seed's
    folds (SemEval is cross-fitted; every other setting has one fold)."""
    rows = []
    for dg in diags:
        k, sil = int(dg["k_auto"]), {int(a): float(b) for a, b in dg["silhouette"].items()}
        others = [v for a, v in sil.items() if a != k]
        rows.append((k, sil.get(k), (sil[k] - max(others)) if others and k in sil else None))
    m = lambda i: (float(np.mean([r[i] for r in rows if r[i] is not None]))                   # noqa: E731
                   if any(r[i] is not None for r in rows) else None)
    return {"k_auto": m(0), "silhouette": m(1), "silhouette_margin": m(2)}


def _geo(folds: list, C: int) -> dict:
    mean = lambda f: float(np.mean([f(z) for z in folds]))                                    # noqa: E731
    return {"ari_auto": mean(lambda z: z["ari"]["auto"]), "capture_c1": mean(lambda z: z["capture"][str(C - 1)]),
            "rank_90": mean(lambda z: z["rank_90"]), "nuisance_var_share": mean(lambda z: z["nuisance_var_share"]),
            "train_pc1_var_share": mean(lambda z: z["pc1_var_share"])}


def spectrum(X) -> dict:
    """Label-free eigenspectrum of the whole corpus: PC1 and top-5 variance shares (as
    `revised._pc_shares`), the gap λ1/λ2 and the entropy effective rank over d."""
    Xc = np.asarray(X, np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    lam = np.clip(np.linalg.eigvalsh(Xc.T @ Xc / len(Xc))[::-1], 0, None)
    p = lam / lam.sum()
    return {"pc1_var_share": float(p[0]), "top5_var_share": float(p[:5].sum()), "eigengap_12": float(lam[0] / lam[1]),
            "eff_rank_share": float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))) / len(lam))}


def _gaps(m: dict, km: dict | None) -> dict:
    meta = m["metadata_leace"]["linear_auc"]
    out = {"gap_sae": m["sae_leace"]["linear_auc"] - meta, "gap_pc1": m["pca_top1"]["linear_auc"] - meta}
    if km is not None:
        out["gap_kmeans"] = km["linear_auc"] - meta
    return out


def _cell(corpus: str, encoder: str, kind: str) -> dict | None:
    """One cell's per-seed outcomes and fitting-time quantities from its published files (None if any is
    missing). `study`: the ten-encoder study's cell (`encoder_*`, `generous_k_*`, `geometry_*`, one set of
    splits); `headline`: Table 2's ID run, its k-means run and the geometry run on the same splits."""
    per_seed, extra = {}, {}
    if kind == "study":
        enc = FA._published(f"encoders/encoder_{corpus}_{encoder}.json")
        gk = FA._published(f"alternatives/generous_k_{corpus}_{encoder}.json")
        geo = FA._published(f"alternatives/geometry_{corpus}_{encoder}.json")
        if not (enc and gk and geo):
            return None
        C, g_by = int(geo["n_classes"]), {int(p["seed"]): p for p in geo["per_seed"]}
        k_by = {int(p["seed"]): p for p in gk["per_seed"]}
        for p in enc["per_seed"]:
            s = int(p["seed"])
            if s in g_by and s in k_by:
                per_seed[s] = {**_gaps(p["metrics"], k_by[s]["metrics"]["kmeans_auto"]),
                               **_folds_k(k_by[s]["diagnostics"]), **_geo(g_by[s]["folds"], C),
                               "sae_pseudo_k": float(np.mean(p["pseudo_groups_k"]))}
        extra, n_docs = dict(enc.get("diagnostics", {})), enc["corpus_stats"]["n_docs"]
    elif corpus == FEDLEX:
        run = _read(os.path.join(paths.FEDLEX, "raw", "id", f"id_{FEDLEX}_{encoder}.json"))
        if run is None:
            return None
        C, g_by = 3, {int(s): g for s, g in run["geometry"]["per_seed"].items()}
        n_docs = run["corpus_stats"]["n_docs"]
        for p in run["per_seed"]:
            s = int(p["seed"])
            per_seed[s] = {**_gaps(p["metrics"], p["metrics"]["kmeans_leace"]), **_folds_k([p["kmeans"]]),
                           **(_geo([g_by[s]], C) if s in g_by else {}), "sae_pseudo_k": float(np.mean(p["pseudo_groups_k"]))}
    else:
        idr = FA._published(f"id/id_{corpus}_{encoder}.json")
        km = FA._published(f"id/id_kmeans_{corpus}_{encoder}.json")
        geo = FA._published(f"alternatives/geometry_{corpus}_{encoder}.json")
        if not (idr and km and geo):
            return None
        C, g_by = int(geo["n_classes"]), {int(p["seed"]): p for p in geo["per_seed"]}
        k_by, n_docs = {int(p["seed"]): p for p in km["per_seed"]}, idr["corpus_stats"]["n_docs"]
        for p in idr["per_seed"]:
            s = int(p["seed"])
            if s in g_by and s in k_by:
                per_seed[s] = {**_gaps(p["metrics"], k_by[s]["metrics"]["kmeans_leace"]),
                               **_folds_k(k_by[s]["diagnostics"]), **_geo(g_by[s]["folds"], C),
                               "sae_pseudo_k": float(np.mean(p["pseudo_groups_k"]))}
    if not per_seed:
        return None
    keys = sorted({k for v in per_seed.values() for k in v})
    mean = {k: float(np.mean(vals)) if (vals := [v[k] for v in per_seed.values() if v.get(k) is not None]) else None
            for k in keys}
    return {"corpus": corpus, "encoder": encoder, "kind": kind, "seeds": sorted(per_seed), "n_classes": C,
            "n_docs": int(n_docs),
            "per_seed": {str(s): v for s, v in per_seed.items()}, "mean": {**mean, **extra}}


def _embeddings(corpus: str, encoder: str, n_docs: int):
    """The cell's cached embeddings (memory-mapped), or None; nothing is encoded here."""
    if corpus == FEDLEX:
        import fedlex as FL
        cands = [FL.FEDLEX["embeddings"]] if encoder == "e5-small" else []
    else:
        cands = [os.path.join(d, f"{corpus}__{encoder}__n{n_docs}.npy")
                 for d in dict.fromkeys((E.EMBED_CACHE, paths.EMBED_CACHE))]
    hit = next((c for c in cands if os.path.exists(c)), None)
    return np.load(hit, mmap_mode="r") if hit else None


def kmeans_stability(corpus: str, encoder: str, k_by_seed: dict) -> float | None:
    """Label-free: k-means (the protocol's n_init) at each seed's selected K, fitted on that seed's first
    training split and applied to the whole corpus; the mean pairwise ARI between the seeds' partitions."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    seeds = sorted(k_by_seed)
    if len(seeds) < 2:
        return None
    data, splits_of, _ = load_setting(corpus, encoder)
    X = np.asarray(data["emb"], np.float32)
    labs = [KMeans(n_clusters=int(round(k_by_seed[s])), n_init=audit.PROTOCOL.get("kmeans_n_init", 20),
                   random_state=int(s)).fit(X[np.asarray(splits_of(s)[0].train_docs)]).predict(X) for s in seeds]
    return float(np.mean([adjusted_rand_score(labs[i], labs[j]) for i in range(len(labs)) for j in range(i + 1, len(labs))]))


def collect_diagnostics(study: bool = True, headline: bool = True, use_spectrum: bool = True,
                        stability: bool = False, verbose: bool = True) -> dict:
    """Every encoder x corpus cell of the ten-encoder study and Table 2's cells: selected K, ARI and PCA
    capture (Item 13), the label-free fitting-time quantities, and the leakage gaps (Item 8). Reads the
    published runs; `use_spectrum` adds the eigenspectrum of each cell's cached embeddings (seconds), and
    `stability` the k-means stability (k-means fits on the cached embeddings; minutes in all)."""
    cells = []
    if study:
        cells += [c for c in (_cell(co, e, "study") for co in STUDY_CORPORA for e in R.ENCODER_STUDY) if c]
    if headline:
        cells += [c for c in (_cell(co, e, "headline") for co, e in HEADLINE_CELLS) if c]
    for c in cells:
        if use_spectrum:
            X = _embeddings(c["corpus"], c["encoder"], c["n_docs"])
            if X is not None and len(X) == c["n_docs"]:
                c["mean"].update(spectrum(X))
                c["spectrum_from"] = "embeddings"
        if stability:
            t0 = time.time()
            ks = {int(s): v["k_auto"] for s, v in c["per_seed"].items() if v.get("k_auto")}
            c["mean"]["kmeans_stability"] = kmeans_stability(c["corpus"], c["encoder"], {s: ks[s] for s in sorted(ks)[:3]})
            if verbose:
                print(f"[stability] {c['corpus']}/{c['encoder']} ({c['kind']}): "
                      f"{_fmt(c['mean']['kmeans_stability'])} ({time.time() - t0:.0f}s)")
    out = {"cells": cells, "predictors": [p for p, _ in PREDICTORS], "outcomes": [o for o, _ in OUTCOMES],
           "note": "seed means; study cells from the ten-encoder study (original binary SwiLTra/SCOTUS), "
                   "headline cells from Table 2's runs on the same splits"}
    if verbose:
        print(f"[diagnostics] {sum(c['kind'] == 'study' for c in cells)} study cells, "
              f"{sum(c['kind'] == 'headline' for c in cells)} headline cells")
    with artifacts.experiment(DIAG_FOLDER):
        artifacts.save_result(out, "diagnostics_cells.json")
    return out


def table_diagnostics(res: dict) -> str:
    """Item 13: per cell, the selected K (k-means LEACE and the SAE's pseudo-groups), ARI, PCA capture and
    the leakage gaps (seed means)."""
    lines = ["| corpus | encoder | set | C | K | SAE K | silhouette | ARI | capture C − 1 | rank 90% | PC1 share | "
             "gap SAE | gap k-means | gap PCA-1 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for kind in ("headline", "study"):
        for c in [c for c in res["cells"] if c["kind"] == kind]:
            m = c["mean"]
            lines.append(f"| {c['corpus']} | {c['encoder']} | {kind} | {c['n_classes']} | {_fmt(m.get('k_auto'), d=1)} | "
                         f"{_fmt(m.get('sae_pseudo_k'), d=1)} | {_fmt(m.get('silhouette'))} | {_fmt(m.get('ari_auto'), d=2)} | "
                         f"{_fmt(m.get('capture_c1'), d=2)} | {_fmt(m.get('rank_90'), d=1)} | {_fmt(m.get('pc1_var_share'))} | "
                         f"{_fmt(m.get('gap_sae'), True)} | {_fmt(m.get('gap_kmeans'), True)} | {_fmt(m.get('gap_pc1'), True)} |")
    md = "\n".join(lines)
    with artifacts.experiment(DIAG_FOLDER):
        FA._save_md(md, "table_diagnostics.md")
    return md


def _rho(x, y) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def _within(x, y, g) -> dict:
    return {co: _rho(x[g == co], y[g == co]) for co in np.unique(g)
            if (g == co).sum() >= 5 and len(np.unique(x[g == co])) > 2 and len(np.unique(y[g == co])) > 2}


def _holm(p: list) -> list:
    order, m, out, run = np.argsort(p), len(p), [None] * len(p), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        out[j] = run
    return out


def predictor_analysis(res: dict, kinds=("study",), n_boot: int = 5000, n_perm: int = 5000, seed: int = 0) -> dict:
    """Do the label-free quantities predict the outcomes across cells (seed means)?

    * pooled Spearman ρ over cells, with a bootstrap interval over cells;
    * within-corpus ρ (across encoders, the mean over corpora) with a permutation p-value that shuffles
      the outcome within corpora only, so between-corpus differences cannot create an association --
      the cells of one corpus share its data, so pooled p-values would overstate the evidence;
    * leave-one-corpus-out: a one-predictor linear fit on the other corpora, scored on the held-out
      corpus (Spearman and R² of the held-out predictions; R² < 0 = worse than the mean).
    Holm's adjustment is applied over the a-priori pairs (`PRIMARY`) and, separately, over the grid."""
    rng = np.random.default_rng(seed)
    cells = [c for c in res["cells"] if c["kind"] in kinds]
    rows = []
    for p, _ in PREDICTORS:
        for o, _ in OUTCOMES:
            pts = [(c["mean"].get(p), c["mean"].get(o), c["corpus"]) for c in cells]
            pts = [t for t in pts if t[0] is not None and t[1] is not None and np.isfinite(t[0]) and np.isfinite(t[1])]
            if len(pts) < 8 or len({t[0] for t in pts}) < 3 or len({t[1] for t in pts}) < 3:
                continue
            x, y, g = (np.array([t[i] for t in pts]) for i in range(3))
            rho = _rho(x, y)
            boot = [_rho(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))]
            within = _within(x, y, g)
            w_obs = float(np.mean(list(within.values()))) if within else None
            p_within = None
            if within:
                null = []
                for _ in range(n_perm):
                    yp = y.copy()
                    for co in np.unique(g):
                        idx = np.flatnonzero(g == co)
                        yp[idx] = rng.permutation(yp[idx])
                    wp = _within(x, yp, g)
                    null.append(np.mean(list(wp.values())) if wp else 0.0)
                p_within = float((1 + np.sum(np.abs(null) >= abs(w_obs))) / (1 + n_perm))
            pred = np.full(len(x), np.nan)
            for co in np.unique(g):
                tr = g != co
                if tr.sum() >= 5 and np.std(x[tr]) > 0:
                    b1, b0 = np.polyfit(x[tr], y[tr], 1)
                    pred[g == co] = b0 + b1 * x[g == co]
            ok = np.isfinite(pred)
            rows.append({"predictor": p, "outcome": o, "primary": (p, o) in PRIMARY, "n_cells": int(len(x)),
                         "n_corpora": int(len(np.unique(g))), "rho": rho,
                         "ci": [float(np.nanquantile(boot, 0.025)), float(np.nanquantile(boot, 0.975))],
                         "within_corpus": within, "within_mean": w_obs, "p_within": p_within,
                         "loco_rho": _rho(pred[ok], y[ok]) if ok.sum() >= 8 else None,
                         "loco_r2": float(1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))
                         if ok.sum() >= 8 else None})
    for key, keep in (("p_holm_primary", lambda r: r["primary"]), ("p_holm_grid", lambda r: True)):
        rs = [r for r in rows if keep(r) and r["p_within"] is not None]
        for r, q in zip(rs, _holm([r["p_within"] for r in rs])):
            r[key] = q
    out = {"kinds": list(kinds), "n_boot": n_boot, "n_perm": n_perm, "rows": rows}
    with artifacts.experiment(DIAG_FOLDER):
        artifacts.save_result(out, f"predictor_analysis_{'_'.join(kinds)}.json")
    return out


def table_predictors(res: dict, primary_only: bool = False) -> str:
    plab, olab = dict(PREDICTORS), dict(OUTCOMES)
    lines = [f"Cells: {', '.join(res['kinds'])}. Pooled Spearman ρ over cells [95% bootstrap interval]; within-corpus "
             "ρ = mean over corpora of the correlation across that corpus's encoders, with a permutation p that "
             "shuffles within corpora (Holm-adjusted over the a-priori pairs ★ and over the whole grid); "
             "leave-one-corpus-out (LOCO): Spearman / R² of the held-out predictions.", "",
             "| | label-free predictor | outcome | cells | pooled ρ [95% CI] | within-corpus ρ | p (Holm ★ / grid) | "
             "LOCO ρ / R² |", "|---|---|---|---|---|---|---|---|"]
    rows = [r for r in res["rows"] if r["primary"] or not primary_only]
    for r in sorted(rows, key=lambda z: (not z["primary"], z["outcome"], -abs(z["within_mean"] or 0))):
        w = "--" if r["within_mean"] is None else f"{r['within_mean']:+.2f}"
        p = "--" if r["p_within"] is None else (
            f"{r['p_within']:.3f} ({_fmt(r.get('p_holm_primary'), d=3) if r['primary'] else '·'} / "
            f"{_fmt(r.get('p_holm_grid'), d=3)})")
        lo = "--" if r["loco_rho"] is None else f"{r['loco_rho']:+.2f} / {r['loco_r2']:+.2f}"
        lines.append(f"| {'★' if r['primary'] else ''} | {plab[r['predictor']]} | {olab[r['outcome']]} | {r['n_cells']} | "
                     f"{r['rho']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | {w} | {p} | {lo} |")
    md = "\n".join(lines)
    with artifacts.experiment(DIAG_FOLDER):
        FA._save_md(md, f"table_predictors_{'_'.join(res['kinds'])}.md")
    return md


def figure_predictors(res: dict, analysis: dict, pairs=(("silhouette", "ari_auto"), ("silhouette", "gap_kmeans"),
                                                        ("pc1_var_share", "gap_pc1"), ("silhouette", "gap_sae"))) -> list:
    """Small multiples, one per a-priori pair: the ten-encoder cells (marker shape = corpus, neutral ink)
    and Table 2's cells (filled accent stars); the title carries the pooled and within-corpus ρ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plab, olab = dict(PREDICTORS), dict(OUTCOMES)
    markers = dict(zip(STUDY_CORPORA, ("o", "s", "^", "D", "v")))
    ink, muted, surface, grid, accent = "#2b2a27", "#6b6a66", "#ffffff", "#e7e6e2", "#2a78d6"
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.5 * len(pairs), 3.3), facecolor=surface)
    for ax, (p, o) in zip(np.atleast_1d(axes), pairs):
        ax.set_facecolor(surface)
        for c in res["cells"]:
            a, b = c["mean"].get(p), c["mean"].get(o)
            if a is None or b is None:
                continue
            if c["kind"] == "study":
                ax.scatter(a, b, marker=markers.get(c["corpus"], "o"), s=40, facecolors="none", edgecolors=muted, linewidths=1.1)
            else:
                ax.scatter(a, b, marker="*", s=120, color=accent, edgecolors=surface, linewidths=0.8, zorder=3)
        r = next((z for z in analysis["rows"] if z["predictor"] == p and z["outcome"] == o), None)
        sub = "" if r is None else (f"\npooled ρ {r['rho']:+.2f}, within-corpus ρ "
                                    f"{_fmt(r['within_mean'], True, 2)} (p {_fmt(r['p_within'], d=3)})")
        ax.set_title(olab[o] + sub, fontsize=8.5, loc="left", color=ink)
        ax.set_xlabel(plab[p], fontsize=8.5, color=ink)
        if o.startswith("gap"):
            ax.axhline(0, color=muted, lw=0.8, ls=(0, (3, 3)))
        ax.grid(color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted, labelsize=7.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
    handles = [Line2D([], [], marker=markers[c], ls="none", mfc="none", mec=muted, label=c) for c in STUDY_CORPORA]
    handles.append(Line2D([], [], marker="*", ls="none", color=accent, ms=11, label="Table 2 cell"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    base = os.path.join(artifacts.BASE_DIR, "figures", DIAG_FOLDER, "figure_predictors")
    os.makedirs(os.path.dirname(base), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=200, bbox_inches="tight", facecolor=surface)
    plt.close(fig)
    return [f"{base}.png", f"{base}.pdf"]


# --------------------------------------------------------------------------- #
# The encoder study on the three-class settings
# --------------------------------------------------------------------------- #
ENC3_SETTINGS = ("swiltra_de_fr_it", "scotus_oyez_lexis_wiki")
#: The reviewer's representative subset; R.ENCODER_STUDY is all ten.
ENC3_ENCODERS = ("e5-small", "e5-base", "e5-large", "mini", "gist-base", "nomic-v2", "mxb")


def run_encoders_3class(settings=ENC3_SETTINGS, encoders=ENC3_ENCODERS, seeds=R.ENCODER_SEEDS, gk_nonlinear=(),
                        device=None, verbose: bool = True, **overrides) -> dict:
    """Each cell exactly as the ten-encoder study ran it (`revised.run_encoder_cell`, then
    `alternatives.run_generous_k` and `run_geometry` on its splits), on the three-class settings. Cached per
    cell, so the study can run over several sessions. Embeddings are encoded on first use."""
    done = {}
    for key in settings:
        for enc in encoders:
            t0 = time.time()
            R.run_encoder_cell(key, enc, seeds=seeds, device=device, verbose=verbose, **overrides)
            A.run_generous_k(key, enc, seeds=seeds, nonlinear=gk_nonlinear, device=device, verbose=verbose)
            A.run_geometry(key, enc, seeds=seeds, device=device, verbose=verbose)
            done[(key, enc)] = round(time.time() - t0, 1)
            if verbose:
                print(f"[encoders] {key}/{enc} done ({done[(key, enc)]}s)")
    return done


def table_encoders_3class(settings=ENC3_SETTINGS, encoders=ENC3_ENCODERS) -> str:
    """Markdown in Table B2's terms: per encoder, the paired linear-AUC gap to Metadata-LEACE of PC-1,
    k-means LEACE (silhouette K) and SAE-LEACE, the R@10 gap, and the geometry diagnostics."""
    lines = []
    for key in settings:
        lines += [f"### {key} (seed mean ± SD of the per-seed gaps)", "",
                  "| encoder | Raw AUC | Meta AUC | gap PC-1 | gap k-means | gap SAE | R@10 gap SAE | R@10 gap PC-1 | K | "
                  "ARI | capture C − 1 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        col = {k: [] for k in ("pc1", "km", "sae")}
        for enc in encoders:
            cell = _read(R.raw_path("encoders", f"encoder_{key}_{enc}.json"))
            gk = _read(R.raw_path("alternatives", f"generous_k_{key}_{enc}.json"))
            geo = _read(R.raw_path("alternatives", f"geometry_{key}_{enc}.json"))
            if not (cell and gk and geo):
                lines.append(f"| {enc} | not run |" + " |" * 9)
                continue
            kb = {int(p["seed"]): p["metrics"] for p in gk["per_seed"]}
            g = lambda c, m, z: [p["metrics"][c][m] - p["metrics"][z][m] for p in cell["per_seed"]]   # noqa: E731
            km = [kb[int(p["seed"])]["kmeans_auto"]["linear_auc"] - kb[int(p["seed"])]["metadata_leace"]["linear_auc"]
                  for p in cell["per_seed"] if int(p["seed"]) in kb]
            st = {"pc1": seed_stats(g("pca_top1", "linear_auc", "metadata_leace")), "km": seed_stats(km),
                  "sae": seed_stats(g("sae_leace", "linear_auc", "metadata_leace"))}
            for k in col:
                if st[k]["mean"] is not None:
                    col[k].append(st[k]["mean"])
            ret = (seed_stats(g("sae_leace", "recall@10", "metadata_leace")), seed_stats(g("pca_top1", "recall@10", "metadata_leace"))) \
                if "recall@10" in cell["per_seed"][0]["metrics"]["raw"] else ({}, {})
            C = int(geo["n_classes"])
            folds = [f for p in geo["per_seed"] for f in p["folds"]]
            lines.append(f"| {enc} | {_fmt(np.mean([p['metrics']['raw']['linear_auc'] for p in cell['per_seed']]))} | "
                         f"{_fmt(np.mean([p['metrics']['metadata_leace']['linear_auc'] for p in cell['per_seed']]))} | "
                         f"{_pm(st['pc1'], True)} | {_pm(st['km'], True)} | {_pm(st['sae'], True)} | {_pm(ret[0], True)} | "
                         f"{_pm(ret[1], True)} | {np.median([f['k_auto'] for f in folds]):.0f} | "
                         f"{np.mean([f['ari']['auto'] for f in folds]):.2f} | {np.mean([f['capture'][str(C - 1)] for f in folds]):.2f} |")
        if any(col.values()):
            lines.append(f"| *mean over encoders* | | | {_fmt(np.mean(col['pc1']) if col['pc1'] else None, True)} | "
                         f"{_fmt(np.mean(col['km']) if col['km'] else None, True)} | "
                         f"{_fmt(np.mean(col['sae']) if col['sae'] else None, True)} | | | | | |")
        lines.append("")
    md = "\n".join(lines)
    with artifacts.experiment("encoders"):
        FA._save_md(md, "table_encoders_3class.md")
    return md


# --------------------------------------------------------------------------- #
# Five seeds: seeds 3-4 of the three-seed experiments, merged with seeds 0-2
# --------------------------------------------------------------------------- #
EXT_SEEDS = (3, 4)
FIVE_FOLDER = "five_seeds"
#: Figure 1's four settings (the coverage cohort) and Figure 2's six rows.
FIG1_SETTINGS = ("swiltra", "swiltra_de_fr_it", "scotus_oyez_lexis", "scotus_oyez_lexis_wiki")
FIG2_ROWS = (("swiltra", "e5-small"), ("swiltra_de_fr_it", "e5-small"), ("scotus_oyez_lexis", "e5-small"),
             ("scotus_oyez_lexis", "mini"), ("scotus_oyez_lexis_wiki", "e5-small"), ("scotus_oyez_lexis_wiki", "mini"))
#: Table 2 cells whose `run_id` holds seeds 0-2 only (every other cell already holds 0-4; CAP/MiniLM has no
#: k-means run).
HEADLINE_3SEED = (("semeval", "e5-small"), ("cap", "e5-small"), ("cap", "mini"))
TRANSFER_DONORS = tuple((e, d) for e, d, _ in FIG4_DONORS)
SEED_JOBS = ("figure1", "figure2", "headline", "transfer")


def extension_root(base: str | None = None) -> str:
    """Seeds 3-4 get a results root of their own inside the notebook's; its basename is unique (the saved
    maps are keyed by it), so a QUICK run's maps never stand in for the full run's."""
    base = os.path.normpath(base or artifacts.BASE_DIR)
    return os.path.join(base, f"{os.path.basename(base)}_seeds_{'_'.join(map(str, EXT_SEEDS))}")


@contextlib.contextmanager
def _published_first(root: str):
    """`focused_additions._published` looks in `root` before the published roots (the band control reads
    its seed's alternatives run there, so it is checked against the SAE of the same seed)."""
    prev = FA.PUBLISHED_ROOTS
    FA.PUBLISHED_ROOTS = (root,) + tuple(prev)
    try:
        yield
    finally:
        FA.PUBLISHED_ROOTS = prev


@contextlib.contextmanager
def _saving_maps(seeds, donors):
    """`revised.run_id` saves frozen maps for the seeds in `OOD_SEEDS` of the donors in `MAP_DONORS`."""
    prev, added = R.OOD_SEEDS, set(donors) - R.MAP_DONORS
    R.OOD_SEEDS = tuple(int(s) for s in seeds)
    R.MAP_DONORS.update(added)
    try:
        yield
    finally:
        R.OOD_SEEDS = prev
        R.MAP_DONORS.difference_update(added)


def _probe_arms(res: dict) -> tuple:
    m0 = res["per_seed"][0]["metrics"]
    return tuple(c for c in res["conditions"] if "mlp_auc" in m0.get(c, {}) or "gb_auc" in m0.get(c, {}))


def _coverage_kwargs(alt: dict, kc: dict) -> tuple:
    """`run_alternatives` and `run_kmeans_count` arguments that reproduce the published coverage runs."""
    a = {"ranks": tuple(alt["ranks"]), "budget_grid": tuple(alt["budget_grid"]),
         "nonlinear": tuple(alt["config"].get("nonlinear") or ()), "nonlinear_arms": _probe_arms(alt),
         "other_encoder": alt.get("other_encoder"), "selectors": "sae_stability" in alt["conditions"]}
    C = int(kc["n_classes"])
    count = kc.get("nonlinear_conditions") in (["kmeans_auto", f"kmeans_k{C}"], ("kmeans_auto", f"kmeans_k{C}"))
    k = {"ks": tuple(kc["ks"]), "nonlinear": tuple(kc["config"].get("nonlinear") or ()),
         "nonlinear_arms": "count" if count else kc.get("nonlinear_conditions")}
    return a, k


def _recipe_overrides(pub: dict, overrides: dict) -> dict:
    """The published recipe's keys that differ from the code default, plus the caller's overrides."""
    rec = pub["config"]["recipe"]
    return {**{k: v for k, v in rec.items() if E.RECIPE.get(k) != v}, **overrides}


def run_seed_extension(jobs=SEED_JOBS, seeds=EXT_SEEDS, fig1_settings=FIG1_SETTINGS, fig2_rows=FIG2_ROWS,
                       headline=HEADLINE_3SEED, donors=TRANSFER_DONORS, device=None, fit_device=None,
                       quick: bool = False, verbose: bool = True, **overrides) -> dict:
    """Seeds 3-4 of Figure 1 (coverage: alternatives, k-means group count, band control), Figure 2 (the
    geometry; its gaps already have five seeds), Table 2's three-seed cells (`run_id`, `run_id_kmeans`) and
    frozen transfer for Figure 4's donors (`run_id` saving maps, `run_ood`, `run_ood_kmeans`, the
    displacement-matched PCA map), each with the published run's settings, under `extension_root()`.
    Every runner caches its file, so an interrupted extension resumes. `quick` trims the grids for a smoke
    test (the merge then refuses the files unless asked not to check the configuration)."""
    root, done = extension_root(), {}
    seeds = tuple(int(s) for s in seeds)
    with results_root(root), _saving_maps(seeds, donors), _published_first(root):
        if "figure1" in jobs:
            for key in fig1_settings:
                pub_alt = FA._published(f"alternatives/alternatives_{key}_e5-small.json", roots=PUBLISHED)
                pub_kc = FA._published(f"alternatives/kmeans_count_{key}_e5-small.json", roots=PUBLISHED)
                pub_band = FA._published(f"band/band_{key}_e5-small.json", roots=(paths.EXTENSIONS,)) or {}
                a_kw, k_kw = _coverage_kwargs(pub_alt, pub_kc)
                if quick:
                    a_kw.update(ranks=a_kw["ranks"][:2], budget_grid=a_kw["budget_grid"][:2], nonlinear=(),
                                nonlinear_arms=None, selectors=False)
                    k_kw.update(ks=k_kw["ks"][:2], nonlinear=())
                t0 = time.time()
                A.run_alternatives(key, "e5-small", seeds=seeds, device=device, fit_device=fit_device,
                                   verbose=verbose, **a_kw, **_recipe_overrides(pub_alt, overrides))
                A.run_kmeans_count(key, "e5-small", seeds=seeds, verbose=verbose, **k_kw)
                FA.run_band_control(key, "e5-small", seeds=seeds,
                                    budgets=(pub_band.get("budgets") or pub_alt["budget_grid"])[: 2 if quick else None],
                                    draws=2 if quick else int(pub_band.get("draws", FA.BAND_DRAWS)),
                                    nonlinear=() if quick else tuple(pub_band.get("config", {}).get("nonlinear", ("mlp",))),
                                    fit_device=fit_device, verbose=verbose, **_recipe_overrides(pub_alt, overrides))
                done[("figure1", key)] = round(time.time() - t0, 1)
        if "figure2" in jobs:
            for key, enc in fig2_rows:
                A.run_geometry(key, enc, seeds=seeds, device=device, verbose=verbose)
                done[("figure2", key, enc)] = True
        if "headline" in jobs:
            for key, enc in headline:
                pub = FA._published(f"id/id_{key}_{enc}.json", roots=PUBLISHED)
                nl = () if quick else tuple(pub["config"].get("nonlinear") or ())
                R.run_id(key, enc, seeds=seeds, nonlinear=nl, device=device, fit_device=fit_device,
                         verbose=verbose, **_recipe_overrides(pub, overrides))
                pk = FA._published(f"id/id_kmeans_{key}_{enc}.json", roots=PUBLISHED)
                if pk is not None:
                    R.run_id_kmeans(key, enc, seeds=seeds, nonlinear=() if quick else tuple(pk["config"].get("nonlinear") or ()),
                                    device=device, verbose=verbose)
                done[("headline", key, enc)] = True
        if "transfer" in jobs:
            for enc, donor in donors:
                pub = FA._published(f"id/id_{donor}_{enc}.json", roots=PUBLISHED)
                R.run_id(donor, enc, seeds=seeds, nonlinear=() if quick else tuple(pub["config"].get("nonlinear") or ()),
                         device=device, fit_device=fit_device, verbose=verbose, **_recipe_overrides(pub, overrides))
                R.run_ood(enc, donor, seeds=seeds, device=device, verbose=verbose)
                A.run_ood_kmeans(matrix=[(enc, donor)], seeds=seeds, device=device, verbose=verbose)
                A.run_ood_ranks(matrix=[(enc, donor)], ranks=(), matched_var=True, seeds=seeds, device=device,
                                verbose=verbose)
                done[("transfer", enc, donor)] = True
    return {"root": root, "done": {"/".join(map(str, k)): v for k, v in done.items()}}


#: Where the published seeds 0-2 are read for merging (never the extension root).
PUBLISHED = FA.PUBLISHED_ROOTS


def _ext(rel: str) -> dict | None:
    return _read(os.path.join(extension_root(), "raw", rel))


def _strip(cfg: dict) -> dict:
    """A run's configuration without what legitimately differs between seed batches."""
    return {k: v for k, v in (cfg or {}).items() if k not in ("seeds", "nonlinear_note")}


def _common_metrics(per_seed: list) -> list:
    """Copies of `per_seed` restricted to the conditions and metrics every seed has (the band control's
    MI arms depend on whether a seed's refit reproduced the published selection)."""
    conds = set.intersection(*(set(p["metrics"]) for p in per_seed))
    keep = {c: set.intersection(*(set(p["metrics"][c]) for p in per_seed)) for c in conds}
    return [{**p, "metrics": {c: {k: v for k, v in p["metrics"][c].items() if k in keep[c]}
                              for c in p["metrics"] if c in conds}} for p in per_seed]


def _merge_seed_file(pub: dict, ext: dict, name: str, strict: bool = True) -> dict:
    """Seeds 0-2 and 3-4 of one runner's file, after checking that the seeds are disjoint, the runs
    share their configuration and that `revised.aggregate` reproduces the published aggregate."""
    s_pub, s_ext = [int(p["seed"]) for p in pub["per_seed"]], [int(p["seed"]) for p in ext["per_seed"]]
    if set(s_pub) & set(s_ext):
        raise ValueError(f"{name}: seeds {sorted(set(s_pub) & set(s_ext))} are in both batches")
    issues = [k for k in set(pub.get("config", {})) | set(ext.get("config", {}))
              if _strip(pub.get("config", {})).get(k) != _strip(ext.get("config", {})).get(k)]
    if issues and strict:
        raise ValueError(f"{name}: the batches differ in config {issues}")
    out = json.loads(json.dumps(pub))
    out["per_seed"] = sorted(pub["per_seed"] + ext["per_seed"], key=lambda p: int(p["seed"]))
    out["seeds"] = sorted(s_pub + s_ext)
    if "metrics" in pub["per_seed"][0] and "agg" in pub:
        again = R.aggregate(_common_metrics(pub["per_seed"]))
        if any(again[c][k] != pub["agg"].get(c, {}).get(k) for c in again for k in again[c]):
            raise RuntimeError(f"{name}: revised.aggregate does not reproduce the published aggregate")
        out["agg"] = R.aggregate(_common_metrics(out["per_seed"]))
    out["merged"] = {"seed_batches": [s_pub, s_ext], "config_differences": issues,
                     "note": "seeds 0-2 published, seeds 3-4 from focused_extensions.run_seed_extension"}
    return out


def run_rank_sweep_extension(seeds=EXT_SEEDS, matrix=R.OOD_MATRIX, device=None, verbose: bool = True) -> dict:
    """The PCA rank sweep of the original transfer donors on the extension seeds (Appendix Figure B3,
    panel C): `alternatives.run_ood_ranks` on its rank grid, without the displacement-matched map, which
    would need a saved SAE-LEACE map of these seeds. No SAE is trained and nothing is encoded when the
    donors and the targets are cached.

    Run it under a results root of its own: under a root that already holds an ``ood_ranks_*`` file of a
    donor (the published runs; the seed extension, for CAP) the family runner would extend that file in
    place on the seeds it holds, not write these seeds."""
    return A.run_ood_ranks(matrix=list(matrix), ranks=A.OOD_RANKS, matched_var=False, seeds=tuple(int(s) for s in seeds),
                           device=device, verbose=verbose)


def _geometry_agg(per_seed: list) -> dict:
    """`alternatives.run_geometry`'s aggregate over every fold of every seed."""
    flat = [f for p in per_seed for f in p["folds"]]
    mean = lambda key: float(np.mean([f[key] for f in flat]))                                 # noqa: E731
    return {"pc1_var_share": mean("pc1_var_share"), "nuisance_var_share": mean("nuisance_var_share"),
            "rank_90": mean("rank_90"), "rank_99": mean("rank_99"),
            "capture": {str(r): float(np.mean([f["capture"][str(r)] for f in flat])) for r in A.GEOMETRY_RANKS},
            "ari": {t: float(np.mean([f["ari"][t] for f in flat])) for t in ("auto", "count", "generous")},
            "k_auto": sorted({f["k_auto"] for f in flat})}


#: Raw transfer scores of two runs must agree to this (the same targets and embeddings scored on two
#: machines differ in the last bit only: 1e-16 between macOS and Colab).
RAW_TOL = 1e-9


def _raw_diff(a: dict, b: dict) -> float:
    """Largest difference between two runs' Raw scores, over task scores and per-unit values."""
    if set(a) != set(b):
        raise ValueError(f"the runs score different targets: {sorted(set(a) ^ set(b))}")
    d = 0.0
    for t in a:
        pa, pb = np.asarray(a[t]["per_unit"], np.float64), np.asarray(b[t]["per_unit"], np.float64)
        if pa.shape != pb.shape:
            raise ValueError(f"{t}: the runs score different units")
        d = max(d, abs(float(a[t]["score"]) - float(b[t]["score"])), float(np.abs(pa - pb).max()) if pa.size else 0.0)
    return d


def _revision_unrecorded(a, b) -> bool:
    """Two encoder records that name one model, at least one without its commit. A batch run under
    transformers 5.19 saved no commit (`ood.encoder_revision` read it from the model config, which no
    longer carries it): that is an unrecorded revision, not a different encoder."""
    return bool(a and b and a.get("model") == b.get("model") and None in (a.get("revision"), b.get("revision")))


def _merge_ood_seeds(pub: dict, ext: dict, name: str, transforms=None) -> dict:
    """Seeds 0-2 and 3-4 of one frozen-transfer file: same encoder, targets, prompts and Raw scores;
    restricted to the transforms both batches scored (or `transforms`).

    The Raw scores are those of the untransformed target embeddings, per query, so their agreement
    shows that both batches scored the same embeddings. An encoder commit that one batch did not
    record is therefore accepted, and noted in the merged file; two different commits are not."""
    if set(pub["seeds"]) & set(ext["seeds"]):
        raise ValueError(f"{name}: overlapping seeds")
    for k in ("encoder", "donor", "tasks"):
        if pub[k] != ext[k]:
            raise ValueError(f"{name}: the batches differ in {k}")
    raw_diff = _raw_diff(pub["raw"], ext["raw"])
    if raw_diff > RAW_TOL:
        raise ValueError(f"{name}: the batches' Raw scores differ by {raw_diff:.2e} (> {RAW_TOL:g})")
    unrecorded = None
    for k in ("mteb_version", "encoder_model", "prompt_style", "target_prompts"):
        a, b = pub["protocol"].get(k), ext["protocol"].get(k)
        if a == b:
            continue
        if k == "encoder_model" and _revision_unrecorded(a, b):
            unrecorded = {"seeds_%s" % "_".join(map(str, pub["seeds"])): a, "seeds_%s" % "_".join(map(str, ext["seeds"])): b}
            continue
        raise ValueError(f"{name}: the batches differ in protocol {k}: {a!r} vs {b!r}")
    keep = [c for c in (transforms or pub["transforms"]) if c in pub["transforms"] and c in ext["transforms"]]
    out = json.loads(json.dumps(pub))
    out["transforms"] = keep
    out["per_seed"] = sorted([{**p, "scores": {c: p["scores"][c] for c in keep}} for p in pub["per_seed"] + ext["per_seed"]],
                             key=lambda p: int(p["seed"]))
    out["seeds"] = sorted(int(p["seed"]) for p in out["per_seed"])
    out["merged"] = {"seed_batches": [pub["seeds"], ext["seeds"]], "transforms": keep,
                     "raw": "the published batch's (largest difference to the other batch: %.1e)" % raw_diff}
    if unrecorded:
        out["merged"]["encoder_model"] = {**unrecorded, "note": "one batch saved no encoder commit; the batches' Raw "
                                          "scores agree, so they scored the same target embeddings"}
        if out["protocol"]["encoder_model"].get("revision") is None:        # keep the recorded commit
            out["protocol"]["encoder_model"] = ext["protocol"]["encoder_model"]
    return out


def _id_reproduction(pub: dict, ext: dict) -> dict:
    """For seeds both batches ran (Table 2's five-seed runs refit for transfer): the largest metric
    difference per condition; closed-form maps must agree to numerical precision."""
    out = {}
    pb = {int(p["seed"]): p for p in pub["per_seed"]}
    for q in ext["per_seed"]:
        p = pb.get(int(q["seed"]))
        if p is None:
            continue
        out[str(q["seed"])] = {c: max(abs(p["metrics"][c][k] - q["metrics"][c][k]) for k in ("linear_auc", "recall@10")
                                      if k in p["metrics"][c] and k in q["metrics"][c])
                               for c in q["metrics"] if c in p["metrics"]}
    return out


def merge_five_seeds(jobs=SEED_JOBS, fig1_settings=FIG1_SETTINGS, fig2_rows=FIG2_ROWS, headline=HEADLINE_3SEED,
                     donors=TRANSFER_DONORS, strict: bool = True, verbose: bool = True) -> dict:
    """Merge every extension file with its published seeds 0-2 into ``<root>/raw/five_seeds/<folder>/``, in
    the published files' schema, so the reporting pipeline can be pointed at them. ID files whose published
    run already holds seeds 0-4 are not merged; their refit seeds 3-4 are compared with the published ones."""
    log = {"merged": [], "compared": {}, "missing": []}

    def put(folder: str, name: str, res: dict):
        with artifacts.experiment(os.path.join(FIVE_FOLDER, folder)):
            artifacts.save_result(res, name)
        log["merged"].append(f"{folder}/{name}")

    def pair(rel: str, pub_roots=PUBLISHED):
        pub, ext = FA._published(rel, roots=pub_roots), _ext(rel)
        if pub is None or ext is None:
            log["missing"].append(rel)
        return pub, ext

    if "figure1" in jobs:
        for key in fig1_settings:
            for rel in (f"alternatives/alternatives_{key}_e5-small.json", f"alternatives/kmeans_count_{key}_e5-small.json"):
                pub, ext = pair(rel)
                if pub and ext:
                    put("alternatives", os.path.basename(rel), _merge_seed_file(pub, ext, rel, strict))
            pub, ext = pair(f"band/band_{key}_e5-small.json", pub_roots=(paths.EXTENSIONS,))
            if pub and ext:
                put("band", f"band_{key}_e5-small.json", _merge_seed_file(pub, ext, f"band_{key}", strict))
    if "figure2" in jobs:
        for key, enc in fig2_rows:
            rel = f"alternatives/geometry_{key}_{enc}.json"
            pub, ext = pair(rel)
            if pub and ext:
                if _geometry_agg(pub["per_seed"]) != pub["agg"]:
                    raise RuntimeError(f"{rel}: the geometry aggregate is not reproduced from its seeds")
                m = _merge_seed_file({k: v for k, v in pub.items() if k != "agg"}, ext, rel, strict)
                m["agg"] = _geometry_agg(m["per_seed"])
                put("alternatives", os.path.basename(rel), m)
    if "headline" in jobs or "transfer" in jobs:
        cells = set(headline if "headline" in jobs else ()) | {(d, e) for e, d in (donors if "transfer" in jobs else ())}
        for key, enc in sorted(cells):
            for rel in (f"id/id_{key}_{enc}.json", f"id/id_kmeans_{key}_{enc}.json"):
                pub, ext = FA._published(rel, roots=PUBLISHED), _ext(rel)
                if pub is None or ext is None:
                    if pub is not None and len(pub["seeds"]) < 5:
                        log["missing"].append(rel)
                    continue
                if set(pub["seeds"]) >= set(ext["seeds"]):
                    log["compared"][rel] = _id_reproduction(pub, ext)
                else:
                    put("id", os.path.basename(rel), _merge_seed_file(pub, ext, rel, strict))
    if "transfer" in jobs:
        for enc, donor in donors:
            for rel, roots in ((f"ood/ood_{enc}_{donor}.json", PUBLISHED),
                               (f"alternatives/ood_kmeans_{enc}_{donor}.json", PUBLISHED),
                               (f"alternatives/ood_ranks_{enc}_{donor}.json", PUBLISHED + (artifacts.BASE_DIR,))):
                pub, ext = pair(rel, roots)
                if pub and ext:
                    trs = ("pca_matched_var",) if "ood_ranks" in rel else None
                    put(os.path.dirname(rel), os.path.basename(rel), _merge_ood_seeds(pub, ext, rel, trs))
    with artifacts.experiment(FIVE_FOLDER):
        artifacts.save_result(log, "merge_log.json")
    if verbose:
        print(f"[five seeds] merged {len(log['merged'])} files; compared {len(log['compared'])}; "
              f"missing {log['missing'] or 'none'}")
        for rel, d in log["compared"].items():
            print(f"   {rel}: refit seeds 3-4 vs published, largest |Δ| per condition: " + "; ".join(
                f"s{s} " + ", ".join(f"{c} {v:.4f}" for c, v in dd.items()) for s, dd in d.items()))
    return log


def _five(rel: str) -> dict | None:
    """A five-seed file: the merged one if it exists, else the published one if it already holds five seeds."""
    m = _read(os.path.join(artifacts.BASE_DIR, "raw", FIVE_FOLDER, rel))
    if m is not None:
        return m
    p = FA._published(rel, roots=PUBLISHED)
    return p if p is not None and len(p.get("seeds", [])) >= 5 else None


def _three_five(vals_by_seed: dict) -> str:
    v3 = [vals_by_seed[s] for s in (0, 1, 2) if s in vals_by_seed]
    v5 = [vals_by_seed[s] for s in sorted(vals_by_seed)]
    st = seed_stats(v5)
    ci = st.get("t_ci") or [None, None]
    return (f"{np.mean(v3):.3f} | {st['mean']:.3f} ± {st['sd']:.3f} | [{_fmt(ci[0])}, {_fmt(ci[1])}] | "
            f"{st['n']}" if v3 and st["mean"] is not None else "-- | -- | -- | --")


TABLE2_CELLS = (("semeval", "e5-small"), ("swiltra_de_fr_it", "e5-small"), ("bills_tweets", "e5-small"),
                ("scotus_oyez_lexis_wiki", "e5-small"), ("cap", "e5-small"), ("scotus_oyez_lexis_wiki", "mini"),
                ("cap", "mini"))


def table_five_seeds_headline(cells=TABLE2_CELLS) -> str:
    """Table 2's cells at five seeds: per method the three-seed mean, the five-seed mean ± SD and t-interval
    (df = 4 at five seeds), and the paired gap to Metadata-LEACE with its sign count."""
    lines = ["| setting | encoder | method | metric | mean, seeds 0-2 | mean ± SD, seeds 0-4 | 95% t-interval (df = seeds − 1) | "
             "seeds | gap to Meta (0-4) | seeds above Meta |", "|---|---|---|---|---|---|---|---|---|---|"]
    for key, enc in cells:
        idr, km = _five(f"id/id_{key}_{enc}.json"), _five(f"id/id_kmeans_{key}_{enc}.json")
        if idr is None:
            lines.append(f"| {key} | {enc} | not yet five seeds | | | | | | | |")
            continue
        by = {int(p["seed"]): p["metrics"] for p in idr["per_seed"]}
        kb = {int(p["seed"]): p["metrics"]["kmeans_leace"] for p in (km or {"per_seed": []})["per_seed"]}
        for m in ("linear_auc",) + utility(key)[:1]:
            for c, lab in (("pca_top1", "PCA-1"), ("kmeans_leace", "k-means LEACE"), ("sae_leace", "SAE-LEACE"),
                           ("metadata_leace", "Metadata-LEACE")):
                src = kb if c == "kmeans_leace" else {s: v[c] for s, v in by.items()}
                vals = {s: v[m] for s, v in src.items() if m in v}
                if not vals:
                    continue
                gaps = [vals[s] - by[s]["metadata_leace"][m] for s in sorted(vals) if s in by]
                g = seed_stats(gaps) if c != "metadata_leace" else {}
                above = f"{g['n_pos']}/{g['n']}" if g else ""
                lines.append(f"| {key} | {enc} | {lab} | {m} | {_three_five(vals)} | {_pm(g, True) if g else ''} | {above} |")
    md = "\n".join(lines)
    with artifacts.experiment(FIVE_FOLDER):
        FA._save_md(md, "table_five_seeds_headline.md")
    return md


def table_five_seeds_figures(fig1_settings=FIG1_SETTINGS, fig2_rows=FIG2_ROWS) -> str:
    """Figure 1's key points and Figure 2's diagnostics at five seeds, beside the three-seed means."""
    lines = ["**Figure 1 (coverage cohort), selected points**", "",
             "| setting | curve | metric | mean, seeds 0-2 | mean ± SD, seeds 0-4 | 95% t-interval (df = seeds − 1) | seeds |",
             "|---|---|---|---|---|---|---|"]
    for key in fig1_settings:
        alt, kc = _five(f"alternatives/alternatives_{key}_e5-small.json"), _five(f"alternatives/kmeans_count_{key}_e5-small.json")
        band = _read(os.path.join(artifacts.BASE_DIR, "raw", FIVE_FOLDER, "band", f"band_{key}_e5-small.json"))
        if alt is None:
            lines.append(f"| {key} | not yet five seeds | | | | | |")
            continue
        C = int(alt["n_classes"])
        pca = [("pca_top1", "PCA, r = 1")] + ([(f"pca_top{C - 1}", f"PCA, r = C − 1 = {C - 1}")] if C > 2 else [])
        pts = [(alt, c, lab) for c, lab in pca + [("pca_top8", "PCA, r = 8"), ("pca_top32", "PCA, r = 32"),
                                                  ("sae_mi_n8", "SAE MI, N = 8"), ("sae_mi_n32", "SAE MI, N = 32"),
                                                  ("sae_mi_n64", "SAE MI, N = 64"), ("metadata_leace", "Metadata-LEACE")]]
        pts += [(kc, f"kmeans_k{C}", f"k-means, K = C = {C}")] if kc else []
        pts += [(band, f"rband_mean_n{n}", f"random band, N = {n}") for n in (8, 32, 64)] if band else []
        for src, c, lab in pts:
            for m in ("linear_auc", "recall@10"):
                vals = {int(p["seed"]): p["metrics"][c][m] for p in src["per_seed"]
                        if c in p["metrics"] and m in p["metrics"][c]}
                if vals:
                    lines.append(f"| {key} | {lab} | {m} | {_three_five(vals)} |")
    lines += ["", "**Figure 2 (geometry diagnostics; the gaps come from Table 2's five-seed runs)**", "",
              "| setting | encoder | quantity | mean, seeds 0-2 | mean ± SD, seeds 0-4 | 95% t-interval (df = seeds − 1) | seeds |",
              "|---|---|---|---|---|---|---|"]
    for key, enc in fig2_rows:
        geo = _five(f"alternatives/geometry_{key}_{enc}.json")
        if geo is None:
            lines.append(f"| {key} | {enc} | geometry: not yet five seeds | | | | |")
        else:
            C = int(geo["n_classes"])
            for lab, f in ((f"capture[C − 1 = {C - 1}]", lambda z: z["capture"][str(C - 1)]),
                           ("capture[1]", lambda z: z["capture"]["1"]), ("ARI (silhouette K)", lambda z: z["ari"]["auto"])):
                vals = {int(p["seed"]): float(np.mean([f(z) for z in p["folds"]])) for p in geo["per_seed"]}
                lines.append(f"| {key} | {enc} | {lab} | {_three_five(vals)} |")
        idr, km = _five(f"id/id_{key}_{enc}.json"), _five(f"id/id_kmeans_{key}_{enc}.json")
        if idr:
            by = {int(p["seed"]): p["metrics"] for p in idr["per_seed"]}
            kb = {int(p["seed"]): p["metrics"]["kmeans_leace"] for p in (km or {"per_seed": []})["per_seed"]}
            for c, lab in (("pca_top1", "gap PC-1"), ("kmeans_leace", "gap k-means"), ("sae_leace", "gap SAE-LEACE")):
                src = kb if c == "kmeans_leace" else {s: v[c] for s, v in by.items()}
                vals = {s: src[s]["linear_auc"] - by[s]["metadata_leace"]["linear_auc"] for s in src if s in by}
                if vals:
                    lines.append(f"| {key} | {enc} | {lab} | {_three_five(vals)} |")
    md = "\n".join(lines)
    with artifacts.experiment(FIVE_FOLDER):
        FA._save_md(md, "table_five_seeds_figures.md")
    return md


# --------------------------------------------------------------------------- #
# Displacement-matched PCA in frozen transfer (Figure 4 A-B's donors)
# --------------------------------------------------------------------------- #
TRANSFER_3CLASS = (("e5-small", "scotus_oyez_lexis_wiki"), ("e5-small", "swiltra_de_fr_it"),
                   ("mini", "scotus_oyez_lexis_wiki"))
TRANSFER_FOLDER = "transfer_matched"


def _stage_published_maps(pairs, seeds) -> list:
    """Make the published run's frozen maps (the ones `run_ood` transferred, recorded in its file) visible
    under the active results root, after checking that each was fit on that seed's training split."""
    staged = []
    for enc, donor in pairs:
        pub = FA._published(f"ood/ood_{enc}_{donor}.json", roots=PUBLISHED)
        by = {int(p["seed"]): p for p in pub["per_seed"]}
        for s in seeds:
            src, dst = by[int(s)]["maps_file"], R._maps_path(enc, donor, s)
            if not os.path.exists(src):
                raise FileNotFoundError(f"{src}: the published frozen map (sync embedding_cache/.../maps/)")
            with np.load(src) as z:
                split_id = ood._split_id(np.asarray(z["train_docs"]))
            if split_id != by[int(s)]["fit"]["split_id"]:
                raise ValueError(f"{src}: fit on split {split_id}, the transfer run recorded {by[int(s)]['fit']['split_id']}")
            if os.path.abspath(src) != os.path.abspath(dst):
                if os.path.exists(dst):
                    if hashlib.sha256(open(src, "rb").read()).digest() != hashlib.sha256(open(dst, "rb").read()).digest():
                        raise ValueError(f"{dst} exists and differs from the published map {src}")
                else:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
            staged.append(dst)
    return staged


def run_transfer_matched(pairs=TRANSFER_3CLASS, seeds=R.OOD_SEEDS, device=None, verbose: bool = True) -> dict:
    """The displacement-matched PCA map, fitted on each donor's ID training split and frozen, matched per
    seed to that seed's published (transferred) SAE-LEACE map, on the six MTEB targets
    (`alternatives.run_ood_ranks` with no fixed ranks). CAP's donors already have it. The published maps are
    staged only for the donors whose matched map is not saved yet."""
    todo = []
    for enc, donor in pairs:
        p = R.raw_path(A.FOLDER, f"ood_ranks_{enc}_{donor}.json")
        d = artifacts.load_result(p) if os.path.exists(p) else None
        if not (d and "pca_matched_var" in d["transforms"] and {int(s) for s in seeds} <= set(d["seeds"])):
            todo.append((enc, donor))
    _stage_published_maps(todo, seeds)
    return A.run_ood_ranks(matrix=list(pairs), ranks=(), matched_var=True, seeds=tuple(seeds), device=device,
                           verbose=verbose)


def _transfer_inputs(enc: str, donor: str, five: bool = False) -> dict:
    """The published transfer run of one donor with k-means LEACE and the displacement-matched PCA map
    merged in (`revised_exhibits._ood_merge`: same seeds, split fingerprints and Raw scores required)."""
    import revised_exhibits as X
    base = FA._published(f"ood/ood_{enc}_{donor}.json", roots=PUBLISHED)
    km = FA._published(f"alternatives/ood_kmeans_{enc}_{donor}.json", roots=PUBLISHED)
    mt = FA._published(f"alternatives/ood_ranks_{enc}_{donor}.json", roots=PUBLISHED)
    if mt is None or "pca_matched_var" not in mt["transforms"]:
        mt = _read(R.raw_path("alternatives", f"ood_ranks_{enc}_{donor}.json"))
    if five:
        base = _read(os.path.join(artifacts.BASE_DIR, "raw", FIVE_FOLDER, "ood", f"ood_{enc}_{donor}.json"))
        km = _read(os.path.join(artifacts.BASE_DIR, "raw", FIVE_FOLDER, "alternatives", f"ood_kmeans_{enc}_{donor}.json"))
        mt = _read(os.path.join(artifacts.BASE_DIR, "raw", FIVE_FOLDER, "alternatives", f"ood_ranks_{enc}_{donor}.json"))
    if base is None:
        raise FileNotFoundError(f"{enc}/{donor}: no {'five-seed ' if five else ''}transfer run")
    r = json.loads(json.dumps(base))

    def same_raw(e: dict, name: str) -> dict:
        # `_ood_merge` requires bit-identical Raw scores; a run scored in another session may differ in the
        # last bit, so check the difference here and hand it the base run's Raw.
        d = _raw_diff(r["raw"], e["raw"])
        if d > RAW_TOL:
            raise ValueError(f"{name}: Raw scores differ from the transfer run's by {d:.2e}")
        return {**e, "raw": r["raw"]}

    if km is not None:
        X._ood_merge(r, same_raw(km, f"ood_kmeans_{enc}_{donor}"), {"kmeans_auto": "kmeans_leace"}, f"ood_kmeans_{enc}_{donor}")
    if mt is not None and "pca_matched_var" in mt["transforms"]:
        X._ood_merge(r, same_raw(mt, f"ood_ranks_{enc}_{donor}"), {"pca_matched_var": "pca_matched_var"},
                     f"ood_ranks_{enc}_{donor}")
        for p, q in zip(r["per_seed"], mt["per_seed"]):
            p["fit"]["matched_rank"], p["fit"]["sae_removed_variance"] = q["fit"]["matched_rank"], q["fit"]["sae_removed_variance"]
    return r


def transfer_matched_summary(pairs=TRANSFER_DONORS, five: bool = False, n_boot: int = 10000) -> dict:
    """Figure 4 A-B's changes from Raw with the displacement-matched PCA map beside the four methods:
    the paired evaluation bootstrap (`ood.summarize`, as published) and the change per fitted map."""
    results = {f"{e}/{d}": _transfer_inputs(e, d, five) for e, d in pairs}
    summ = ood.summarize(results, n_boot=n_boot, save=False)
    trs = FIG4_TRANSFORMS + ("pca_matched_var",)
    var = transfer_seed_variability(donors=[(e, d, None) for e, d in pairs], transforms=trs, summaries=summ["settings"],
                                    name=f"transfer_matched_variability{'_five' if five else ''}.json")
    ranks = {k: [{"seed": int(p["seed"]), "matched_rank": p["fit"].get("matched_rank"),
                  "sae_removed_variance": p["fit"].get("sae_removed_variance")} for p in r["per_seed"]]
             for k, r in results.items()}
    out = {"five_seeds": five, "n_boot": n_boot, "seeds": {k: r["seeds"] for k, r in results.items()},
           "matched": ranks, "variability": var["rows"],
           "changes": {k: {c: summ["settings"][k]["changes"][c] for c in trs if c in summ["settings"][k]["changes"]}
                       for k in summ["settings"]}}
    with artifacts.experiment(TRANSFER_FOLDER):
        artifacts.save_result(out, f"transfer_matched{'_five' if five else ''}.json")
    return out


def table_transfer_matched(res: dict) -> str:
    lab = {"pca_top1": "PC-1", "pca_matched_var": "PCA, displacement-matched", "kmeans_leace": "k-means LEACE",
           "sae_leace": "SAE-LEACE", "metadata_leace": "Metadata-LEACE"}
    lines = [f"Seeds per donor: {', '.join(f'{k} {v}' for k, v in res['seeds'].items())}. Change from Raw; bootstrap "
             f"= paired evaluation bootstrap ({res['n_boot']:,} resamples, conditional on the maps); per fitted map = "
             "the change of each seed's frozen map; r* = rank of the matched PCA map per seed.", "",
             "| donor | map | legal macro Δ [95% CI] | per fitted map | STS Δ [95% CI] | per fitted map | reading (legal) |",
             "|---|---|---|---|---|---|---|"]
    rows = {(r["encoder"], r["donor"], r["transform"], r["target"]): r for r in res["variability"]}
    for key, chs in res["changes"].items():
        enc, donor = key.split("/")
        rr = res["matched"][key]
        for c in FIG4_TRANSFORMS + ("pca_matched_var",):
            if c not in chs:
                continue
            lg, st = rows.get((enc, donor, c, "legal macro nDCG@10")), rows.get((enc, donor, c, "STSBenchmark Spearman"))
            name = lab[c] + (" (r* = " + "/".join(str(x["matched_rank"]) for x in rr) + ")" if c == "pca_matched_var" else "")
            fmt = lambda r: ("--", "--") if r is None else (                                         # noqa: E731
                f"{r['boot_delta']:+.3f} [{r['boot_ci'][0]:+.3f}, {r['boot_ci'][1]:+.3f}]",
                ", ".join(f"{x:+.3f}" for x in r["per_seed"]))
            (a, b), (cc, d) = fmt(lg), fmt(st)
            lines.append(f"| {key} | {name} | {a} | {b} | {cc} | {d} | {lg['verdict'] if lg else '--'} |")
    md = "\n".join(lines)
    with artifacts.experiment(TRANSFER_FOLDER):
        FA._save_md(md, f"table_transfer_matched{'_five' if res['five_seeds'] else ''}.md")
    return md


# --------------------------------------------------------------------------- #
# Label-free intervention quantities of the Table 3 runs (Item 8, the headline cells)
# --------------------------------------------------------------------------- #
def table_intervention_quantities(cells=T3X_CELLS) -> str:
    """Per Table 3 cell, label-free quantities of the fitted SAE-LEACE intervention -- displaced variance,
    the matched PCA rank, MI concentration -- beside its leakage gap. Six cells: descriptive only."""
    lines = ["| setting | encoder | SAE displacement share | matched rank r* | MI: positive scores | MI: top-32 share of "
             "the MI mass | capture of SAE span by top-32 PCs | gap SAE-LEACE − Meta (linear AUC) |",
             "|---|---|---|---|---|---|---|---|"]
    for setting, encoder, _ in cells:
        r = load_table3x(setting, encoder)
        if r is None:
            lines.append(f"| {setting} | {encoder} | not run | | | | | |")
            continue
        ds = [d for q in r["per_seed"] for d in q["diagnostics"]]
        gap = [q["metrics"]["sae_mi"]["linear_auc"] - q["metrics"]["metadata_leace"]["linear_auc"] for q in r["per_seed"]]
        sub = [d["subspaces"]["sae_mi"].get("capture_top32") for d in ds if "sae_mi" in d.get("subspaces", {})]
        lines.append(f"| {setting} | {encoder} | {np.mean([d['sae_displacement_share'] for d in ds]):.3f} | "
                     f"{np.mean([d['matched_rank'] for d in ds]):.1f} | {np.mean([d['mi_scores']['n_positive'] for d in ds]):.0f} | "
                     f"{np.mean([d['mi_scores']['top_n_mass_share'] for d in ds]):.2f} | "
                     f"{_fmt(float(np.mean([x for x in sub if x is not None])) if any(x is not None for x in sub) else None, d=2)} | "
                     f"{_pm(seed_stats(gap), True)} |")
    md = "\n".join(lines)
    with artifacts.experiment(DIAG_FOLDER):
        FA._save_md(md, "table_intervention_quantities.md")
    return md
