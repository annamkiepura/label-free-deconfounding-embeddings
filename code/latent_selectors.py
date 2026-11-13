"""Choosing the confounder-like SAE latents without k-means (`notebooks/05_latent_selectors.ipynb`).

SAE-LEACE ranks the latents of a TopK dictionary by the mutual information between a latent's firing
indicator and a k-means partition of the embeddings, and erases the top N decoder directions with
LEACE. Where k-means misses the nuisance (MiniLM on SCOTUS, CAP, SemEval) the ranking inherits the
miss. This module scores the same latents of the same SAE with rules that never see a partition,
erases each rule's top-N latents with the same operator, and audits the result on the published
splits. It extends the direction-selection rows (1-3) of Main Table 3.

Notation: ``a_j`` is the binary firing indicator of latent j on the training documents, ``p_j`` its
training firing rate and ``d_j`` its decoder direction.

Selectors (`SCORERS`; each scores every eligible latent, and its budget-N set is the top N):

* ``mi``         -- ``I(a_j; G)``, G the silhouette-K k-means grouping: the paper's rule, the reference.
* ``entropy``    -- ``H(a_j)``: prefers ``p_j`` closest to 0.5. No reference variable. Since
  ``I(a_j; G) <= H(a_j)``, it isolates how much of MI's ranking is firing density.
* ``dip``        -- Hartigan's dip statistic of the training coordinates ``X_tr d_j``, the coordinates
  Directions-LEACE consumes. Scale-invariant, so it can flag a categorical split along a low-variance
  direction; each direction is scored on its own, so no K is chosen.
* ``cofire``     -- ``mean_{l != j} I(a_j; a_l)`` over the eligible latents: how much a latent's firing
  tells about the other latents' (a hub of the co-firing graph). The same MI estimator as ``mi``,
  with the dictionary itself as the reference instead of the k-means partition.
* ``cofire_neg`` -- the same sum restricted to the latents negatively correlated with j: mutual
  exclusivity, the signature of categorical alternatives.
* ``oracle``     -- ``I(a_j; y)``, y the true classes. **Post hoc, not a method**: it uses the labels,
  like the ARI and PCA-capture diagnostics. It gives a reference for what the dictionary can reach
  at each N, and the yardstick the label-free rankings are compared with.

Controls on the same SAE: ``random`` (uniform draws from the eligible latents, Table 3 row 2's
control and RNG) and ``gauss`` (Gaussian directions, row 3's). References: Raw, Metadata-LEACE and
k-means LEACE at the silhouette K.

Held fixed across selectors:

* the SAE of each split (`protocols.fit_base`, the published recipe and seed), cached per split;
* the splits and the audit (`revised._splits`, `revised.evaluate_fits`), on `SEEDS`;
* the eligible pool: the latents MI can score, training firing rate in `BAND`;
* nested prefixes: a selector's budget-N set is the first N latents of one ranking;
* the operator: Directions-LEACE, `protocols._fit_leace(X_tr, X_tr @ D_S^T)`.

No selector has a hyperparameter tuned on labels. The dip uses every training document (no
subsample) and the statistic itself (no p-value, no normalization); ties go to the lower latent id.

Per fit the run also records, for every selector: the ranking, the scores, the top-N overlap and
the rank correlation with every other selector (so with the oracle and with MI-k-means), the mean
firing rate of the selected latents, and the normalized displacement of every map
(`alternatives._removed_variance`, the statistic displacement-matched PCA uses).

Outputs, under the active results root (the notebook uses ``results/selectors/``)::

    raw/selectors/     selectors_<setting>_<encoder>.json
    tables/selectors/  table_*.md
    sae/               sae_<setting>_<encoder>_s<seed>_f<fold>.npz   (the per-split SAE cache)
"""

from __future__ import annotations

import json
import os
import platform
import time
from importlib.metadata import version

import numpy as np

import alternatives as A
import artifacts
import audit
import experiments as E
import focused_additions as FA
import focused_extensions as FX
import ood
import paths
import protocols as P
import revised as R

FOLDER = "selectors"
SAE_FOLDER = "sae"
#: The budgets reported, and the ones the headline tables show: at N = 32 MI and random selection
#: already differ by only 0.010-0.014 linear AUC, so selectors can separate at small N only.
BUDGETS = (1, 2, 4, 8, 16, 32)
HEADLINE = (4, 8)
#: Budgets that also get the nonlinear probe and the detailed table (N = 32 is Main Table 3's).
DETAIL = (8, 32)
#: Seeds 0-2 are the coverage cohort's; 3-4 extend them with the same split rule. They are the
#: published id runs' seeds 3-4 for SwiLTra, SCOTUS, Bills-Tweets and Fedlex; SemEval and CAP were
#: published with seeds 0-2 only.
SEEDS = (0, 1, 2, 3, 4)
DRAWS = FA.BAND_DRAWS
GAUSS_DRAWS = 1
#: The eligible pool: the latents MI scores (`sae_confounder_pipeline.score_confounder_features`).
BAND = FA.R1_BAND
#: The published runs' budget, at which a refit is compared with them.
PUBLISHED_N = int(E.RECIPE["top_features"])
#: Decoder rows cached per SAE beside the eligible latents': MI's leading latents by the published rule.
MI_KEEP = 64

SELECTORS = ("mi", "entropy", "dip", "cofire", "cofire_neg", "oracle")
#: The selectors that use neither a partition nor the labels.
KMEANS_FREE = ("entropy", "dip", "cofire", "cofire_neg")
REFERENCES = ("raw", "metadata_leace", "kmeans_leace")
#: Every arm with a budget, in table order: (name, row label, column label).
ARMS = (("mi", "MI vs k-means (the paper's rule)", "MI-k-means"),
        ("entropy", "Firing entropy", "Entropy"),
        ("dip", "Dip of the decoder coordinates", "Dip"),
        ("cofire", "Co-firing centrality", "Co-firing"),
        ("cofire_neg", "Co-firing centrality, exclusive pairs", "Co-firing (−)"),
        ("random", "Random SAE latents (eligible pool)", "Random"),
        ("gauss", "Gaussian directions", "Gaussian"),
        ("oracle", "Oracle: MI vs the true classes (post hoc)", "Oracle"))
ROW = {a: row for a, row, _ in ARMS}
COL = {a: col for a, _, col in ARMS}
REF_LABEL = {"raw": "Raw", "metadata_leace": "Metadata-LEACE", "kmeans_leace": "k-means LEACE"}
METRIC = {"linear_auc": "linear AUC ↓", "mlp_auc": "MLP AUC ↓", "recall@10": "R@10 ↑",
          "gold_sim_spearman": "graded sim. ρ ↑", "topic_purity": "topic purity ↑",
          "within_class_nn10": "within-class NN@10 kept ↑"}

#: The settings: every dataset with E5-small, every class combination of SwiLTra and SCOTUS, and
#: MiniLM on SCOTUS, where k-means LEACE trails SAE-LEACE most clearly.
SWILTRA = ("swiltra", "swiltra_de_it", "swiltra_fr_it", "swiltra_de_fr_it")
SCOTUS = ("scotus_oyez_lexis", "scotus_oyez_wiki", "scotus_wiki_lexis", "scotus_oyez_lexis_wiki")
E5_CELLS = tuple((k, "e5-small") for k in ("semeval",) + SWILTRA + ("bills_tweets",) + SCOTUS + ("cap", FX.FEDLEX))
MINI_CELLS = tuple((k, "mini") for k in SCOTUS)
CELLS = E5_CELLS + MINI_CELLS
SHORT = {"semeval": "SemEval", "swiltra": "SwiLTra DE-FR", "swiltra_de_it": "SwiLTra DE-IT",
         "swiltra_fr_it": "SwiLTra FR-IT", "swiltra_de_fr_it": "SwiLTra DE-FR-IT", "bills_tweets": "Bills-Tweets",
         "scotus_oyez_lexis": "SCOTUS Oyez-Lexis", "scotus_oyez_wiki": "SCOTUS Oyez-Wiki",
         "scotus_wiki_lexis": "SCOTUS Wiki-Lexis", "scotus_oyez_lexis_wiki": "SCOTUS Oyez-Lexis-Wiki",
         "cap": "CAP", FX.FEDLEX: "Fedlex DE-FR-IT"}
ENCODER = {"e5-small": "E5-small", "mini": "MiniLM"}
#: The settings of the extended Table 3 (`focused_extensions.T3X_CELLS`, E5-small).
TABLE3_CELLS = tuple((k, "e5-small") for k in ("swiltra_de_fr_it", "scotus_oyez_lexis_wiki", "semeval", "cap", FX.FEDLEX))
#: Main Table 3's direction-selection rows with the selectors between them: (row, label, arm).
TABLE3_ROWS = (("1", "MI-selected SAE directions", "mi"), ("7", "Entropy-selected SAE directions", "entropy"),
               ("8", "Dip-selected SAE directions", "dip"), ("9", "Co-firing-selected SAE directions", "cofire"),
               ("10", "Co-firing-selected, exclusive pairs", "cofire_neg"), ("2", "Random SAE directions", "random"),
               ("3", "Random non-SAE directions", "gauss"), ("", "Oracle (MI with the true classes)", "oracle"))
TABLE3_REFERENCES = (("raw", "Raw"), ("metadata_leace", "Metadata-LEACE (supervised)"))
#: Utility columns of the detailed table (Table 2's metric, then the nuisance-orthogonal one).
UTILITY = {"semeval": ("recall@10", "gold_sim_spearman"), "cap": ("topic_purity",), "bills_tweets": ()}


def utility(setting: str) -> tuple:
    return UTILITY.get(setting, ("recall@10",)) + ("within_class_nn10",)


def title(setting: str, encoder: str) -> str:
    return f"{SHORT.get(setting, setting)} / {ENCODER.get(encoder, encoder)}"


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #
def binary_entropy(p) -> np.ndarray:
    """``H(a)`` in bits of an indicator with firing rate `p`."""
    p = np.asarray(p, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return np.where((p > 0) & (p < 1), h, 0.0)


def _plogp(count, pa, pb, n: int) -> np.ndarray:
    """One cell of the plug-in MI sum, ``p log2(p / (pa pb))`` with ``p = count / n`` and 0 log 0 = 0."""
    joint = count / n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = joint * np.log2(joint / (pa * pb))
    return np.where(count > 0, t, 0.0)


def group_mi(F, groups) -> np.ndarray:
    """``I(a_j; G)`` in bits for every column of the binary matrix `F`: the plug-in estimator of
    `sae_confounder_pipeline._binary_mutual_information`, all latents at once."""
    F, groups = np.asarray(F, np.float64), np.asarray(groups)
    n, c1 = len(F), F.sum(axis=0)
    mi = np.zeros(F.shape[1])
    for g in np.unique(groups):
        mask = groups == g
        n_g, c1g = int(mask.sum()), F[mask].sum(axis=0)
        mi += _plogp(c1g, c1 / n, n_g / n, n) + _plogp(n_g - c1g, 1.0 - c1 / n, n_g / n, n)
    return mi


def pairwise_mi(F) -> tuple:
    """``(I(a_j; a_l), [corr(a_j, a_l) < 0])`` for every pair of columns of the binary matrix `F`,
    with the same plug-in estimator. The diagonal holds ``H(a_j)`` and is never negative."""
    F = np.asarray(F, np.float64)
    n, c1 = len(F), F.sum(axis=0)
    n11 = np.rint(F.T @ F)
    a, b = c1[:, None], c1[None, :]
    mi = (_plogp(n11, a / n, b / n, n) + _plogp(a - n11, a / n, 1.0 - b / n, n)
          + _plogp(b - n11, 1.0 - a / n, b / n, n) + _plogp(n - a - b + n11, 1.0 - a / n, 1.0 - b / n, n))
    return mi, n * n11 < a * b


def cofiring_centrality(mi: np.ndarray, negative: np.ndarray, exclusive: bool = False) -> np.ndarray:
    """Mean over the other latents of ``I(a_j; a_l)``; `exclusive` keeps only the negatively
    correlated pairs in the sum (the divisor stays the number of other latents)."""
    m = max(len(mi) - 1, 1)
    return (mi * negative).sum(axis=1) / m if exclusive else (mi.sum(axis=1) - np.diag(mi)) / m


def dip_scores(X_tr, D) -> np.ndarray:
    """Hartigan's dip statistic of the training coordinates ``X_tr d_j`` of every row of `D`, on
    every training document."""
    # Imported here, after torch: diptest's macOS wheel bundles its own OpenMP runtime, and loading it
    # before torch makes the SAE fit crash.
    try:
        import diptest
    except ImportError as err:
        raise ImportError("the dip selector needs the `diptest` package: pip install diptest") from err
    Z = np.asarray(X_tr, np.float64) @ np.asarray(D, np.float64).T
    return np.array([diptest.dipstat(np.ascontiguousarray(Z[:, j])) for j in range(Z.shape[1])])


def _pair(c: dict) -> tuple:
    if "pair" not in c:
        c["pair"] = pairwise_mi(c["sae"]["firing"])
    return c["pair"]


#: name -> score of every eligible latent, from ``{"X_tr", "y_tr", "sae"}``. A new selector is one entry.
SCORERS = {
    "mi": lambda c: np.asarray(c["sae"]["scores"], np.float64)[c["sae"]["pool"]],
    "entropy": lambda c: binary_entropy(c["sae"]["freq"][c["sae"]["pool"]]),
    "dip": lambda c: dip_scores(c["X_tr"], decoder_rows(c["sae"], c["sae"]["pool"])),
    "cofire": lambda c: cofiring_centrality(*_pair(c)),
    "cofire_neg": lambda c: cofiring_centrality(*_pair(c), exclusive=True),
    "oracle": lambda c: group_mi(c["sae"]["firing"], c["y_tr"]),
}


def score_latents(X_tr, y_tr, sae: dict, selectors=SELECTORS) -> dict:
    """Every selector's score of the eligible latents, aligned with ``sae["pool"]``. Only ``oracle``
    reads `y_tr`."""
    ctx = {"X_tr": X_tr, "y_tr": y_tr, "sae": sae}
    return {sel: np.asarray(SCORERS[sel](ctx), np.float64) for sel in selectors}


def rank_latents(score, pool) -> np.ndarray:
    """The eligible latents by decreasing score; ties go to the lower latent id."""
    return np.asarray(pool)[np.argsort(-np.asarray(score, np.float64), kind="stable")]


# --------------------------------------------------------------------------- #
# One SAE per split, cached
# --------------------------------------------------------------------------- #
def fit_sae(X, sp, setting: str, encoder: str, fold: int, kw: dict, pseudo_groups="auto", fit_device=None,
            verbose: bool = True) -> dict:
    """The split's SAE as the selectors need it, fit as every published run fits it
    (`protocols.fit_base`, same recipe and seed) and cached in ``<results>/sae/``.

    Holds the MI scores and firing rates of every latent, the eligible pool, the pool's firing
    matrix on the training documents, the k-means grouping MI is scored against, and the decoder
    rows of the pool and of MI's `MI_KEEP` leading latents (``latents`` names the rows kept)."""
    folder = os.path.join(artifacts.get_results_dir(), SAE_FOLDER)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"sae_{setting}_{encoder}_s{int(sp.seed)}_f{int(fold)}.npz")
    split_id = ood._split_id(np.asarray(sp.train_docs))
    recipe = json.dumps({k: kw[k] for k in ("expansion_factor", "k", "n_epochs")}, sort_keys=True)
    if os.path.exists(path):
        with np.load(path) as z:
            sae = {k: z[k] for k in z.files}
        if str(sae["split_id"]) != split_id or str(sae["recipe"]) != recipe:
            raise ValueError(f"{path}: cached SAE was fit on split {sae['split_id']} with {sae['recipe']}, "
                             f"not on {split_id} with {recipe}")
    else:
        t0 = time.time()
        base = P.fit_base(X, sp, expansion_factor=kw["expansion_factor"], k=kw["k"], n_epochs=kw["n_epochs"],
                          pseudo_groups=pseudo_groups, device=fit_device)
        fired = np.asarray(base.feats_train) > 0
        freq, scores = fired.mean(axis=0), np.asarray(base.scores, np.float64)
        pool = FA._pools(freq)["band"]
        keep = np.union1d(pool, np.argsort(scores)[::-1][:MI_KEEP])
        sae = {"latents": keep, "decoder": np.asarray(base.decoder, np.float32)[keep], "scores": scores, "freq": freq,
               "pool": pool, "firing": np.packbits(fired[:, pool], axis=1),
               "groups": np.asarray(base.pseudo_group_train, np.int32), "k_auto": np.int64(base.pseudo_groups),
               "silhouette": np.array(json.dumps(base.silhouette)), "split_id": np.array(split_id),
               "recipe": np.array(recipe), "fit_seconds": np.float64(time.time() - t0),
               "info": np.array(json.dumps({k: v for k, v in base.sae_info.items() if isinstance(v, (int, float, str))}))}
        tmp = path[:-len(".npz")] + ".tmp.npz"
        np.savez_compressed(tmp, **sae)
        os.replace(tmp, path)
        if verbose:
            print(f"[sae] {setting}/{encoder} seed {sp.seed} fold {fold}: fit in {sae['fit_seconds']:.0f}s, "
                  f"{len(pool)} of {len(scores)} latents eligible -> {path}")
    sae["firing"] = np.unpackbits(sae["firing"], axis=1, count=len(sae["pool"])).astype(bool)
    return {**sae, "path": path}


def decoder_rows(sae: dict, ids) -> np.ndarray:
    """The decoder directions of latents `ids` (they must be among the rows the cache keeps)."""
    ids = np.asarray(ids, np.int64)
    pos = np.minimum(np.searchsorted(sae["latents"], ids), len(sae["latents"]) - 1)
    if not np.array_equal(sae["latents"][pos], ids):
        raise KeyError(f"{sae['path']} keeps the decoder rows of the eligible latents and of MI's {MI_KEEP} leading "
                       f"latents only; latents {ids[sae['latents'][pos] != ids].tolist()} are neither")
    return sae["decoder"][pos]


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def _budget(cond: str):
    """The budget N of a condition named ``<arm>_n<N>``; None for a reference."""
    head, _, tail = cond.rpartition("_n")
    return int(tail) if head and tail.isdigit() else None


def _spearman(a, b):
    from scipy.stats import spearmanr
    rho = float(spearmanr(a, b).statistic) if len(a) > 2 else float("nan")
    return rho if np.isfinite(rho) else None


def _fold_maps(X, y, sp, fi: int, s: int, sae: dict, budgets, selectors, draws: int, gauss_draws: int) -> tuple:
    """Every arm of one fit on its SAE, and the fit's diagnostics."""
    from sklearn.metrics import adjusted_rand_score

    tr = np.asarray(sp.train_docs)
    X_tr, y_tr, d = np.asarray(X[tr], np.float32), np.asarray(y)[tr], X.shape[1]
    pool, freq, K, top = sae["pool"], sae["freq"], int(sae["k_auto"]), max(budgets)
    score = score_latents(X_tr, y_tr, sae, selectors)
    order = {sel: rank_latents(score[sel], pool) for sel in selectors}
    mi_order = np.argsort(sae["scores"])[::-1]              # = sae_confounder_pipeline.select_confounder_features
    if "mi" in order:
        order["mi"] = mi_order
    draw = {f"random{j}": FA.latent_draw(pool, top, s, fi, j) for j in range(draws)}
    er = {"raw": (lambda Z: np.asarray(Z, np.float32)),
          "metadata_leace": P._fit_leace(X_tr, P._one_hot(y_tr, np.unique(y)), standardise=False),
          "kmeans_leace": R.kmeans_eraser(X_tr, K, sp.seed)[0]}
    for arm, o in {**order, **draw}.items():
        for N in budgets:
            er[f"{arm}_n{N}"] = FA._fit_dirs(X_tr, decoder_rows(sae, o[:N]))
    for j in range(gauss_draws):
        G = FA.gaussian_directions(d, top, s, j)            # nested: its first N rows are R1's budget-N draw
        for N in budgets:
            er[f"gauss{j}_n{N}"] = FA._fit_dirs(X_tr, G[:N])

    sets = {arm: [int(j) for j in o[:top]] for arm, o in {**order, **draw}.items()}
    rand = [f"random{j}" for j in range(draws)]

    def overlap(a: str, b: str) -> dict:
        return {str(N): len(set(sets[a][:N]) & set(sets[b][:N])) / N for N in budgets}

    def over_draws(per_draw: list) -> dict:
        return {str(N): float(np.mean([v[str(N)] for v in per_draw])) for N in budgets}

    ov = {a: {b: overlap(a, b) for b in order if b != a} for a in order}
    mean_freq = {a: {str(N): float(freq[sets[a][:N]].mean()) for N in budgets} for a in sets}
    if rand:
        ov["random"] = {b: over_draws([overlap(a, b) for a in rand]) for b in order}
        mean_freq["random"] = over_draws([mean_freq[a] for a in rand])
    disp = {c: float(A._removed_variance(fn, X_tr)) for c, fn in er.items() if c != "raw"}
    for arm, k in (("random", draws), ("gauss", gauss_draws)):
        for N in budgets if k else ():
            disp[f"{arm}_n{N}"] = float(np.mean([disp[f"{arm}{j}_n{N}"] for j in range(k)]))
    diag = {
        "fold": fi, "n_train_docs": int(len(tr)), "sae_file": sae["path"], "sae_seconds": float(sae["fit_seconds"]),
        "n_latents": int(len(sae["scores"])), "pool_size": int(len(pool)), "pseudo_groups_k": K,
        "kmeans_ari": float(adjusted_rand_score(y_tr, sae["groups"])),
        "pool": [int(j) for j in pool], "freq": [float(f) for f in freq[pool]],
        "scores": {sel: v.tolist() for sel, v in score.items()}, "selected": sets,
        "mi_top": [int(j) for j in mi_order[:PUBLISHED_N]],
        "mi_prefix": {str(N): {"zero_score": int((sae["scores"][mi_order[:N]] <= 0).sum()),
                               "out_of_pool": int((~np.isin(mi_order[:N], pool)).sum())} for N in budgets},
        # The vectorized estimator against the pipeline's own MI scores, on the same grouping.
        "mi_estimator_max_diff": float(np.abs(group_mi(sae["firing"], sae["groups"]) - sae["scores"][pool]).max()),
        "overlap": ov, "spearman": {a: {b: _spearman(score[a], score[b]) for b in order if b != a} for a in order},
        "mean_freq": mean_freq, "displacement": disp}
    return er, diag


def _seed_record(data, X, y, splits, s: int, setting: str, encoder: str, kw: dict, budgets, selectors, draws: int,
                 gauss_draws: int, nonlinear, nonlinear_budgets, fit_device, verbose: bool) -> dict:
    saes = [fit_sae(X, sp, setting, encoder, fi, kw, data["config"]["pseudo_groups"], fit_device, verbose)
            for fi, sp in enumerate(splits)]
    sizes = [len(sae["pool"]) for sae in saes]
    feasible = [N for N in budgets if N <= min(sizes)]      # a budget above the pool is no selection at all
    if not feasible:
        raise FX.BandTooSmall(s, int(np.argmin(sizes)), min(sizes), min(budgets))
    fits, diags = [], []
    for fi, (sp, sae) in enumerate(zip(splits, saes)):
        er, dg = _fold_maps(X, y, sp, fi, s, sae, feasible, selectors, draws, gauss_draws)
        f = R.Erasers(sp, er)
        f.selected, f.config = dg["mi_top"], {"pseudo_groups_k": dg["pseudo_groups_k"]}
        fits.append(f)
        diags.append(dg)
    conds = tuple(fits[0].erasers)
    probed = [c for c in conds if c in REFERENCES or _budget(c) in nonlinear_budgets]
    res = R.evaluate_fits(fits, X, y, conditions=conds, topics=data.get("topics"), nonlinear=nonlinear,
                          nonlinear_conditions=probed)
    res.pop("hits", None)
    m = res["metrics"]
    for c, v in FX._orthogonal_utility(fits, X, y, data.get("topics")).items():
        m[c].update(v)
    for arm, k in (("random", draws), ("gauss", gauss_draws)):
        for N in feasible if k else ():
            m[f"{arm}_n{N}"] = FA._mean_metrics([m[f"{arm}{j}_n{N}"] for j in range(k)])
    res.update(seed=int(s), budgets=feasible, diagnostics=diags)
    return res


def _reference(setting: str, encoder: str, run, draws: int, gauss_draws: int) -> dict:
    """Per seed, what the published fits of this setting selected and scored, under this run's
    condition names: ``{seed: {"source", "selected", "k", "metrics"}}``."""
    out = {}
    for s, r in FX._reference(setting, encoder, run).items():
        m = r["metrics"]
        mets = {c: m[c] for c in REFERENCES if c in m}
        if "sae_mi" in m:
            mets[f"mi_n{PUBLISHED_N}"] = m["sae_mi"]
        if gauss_draws and "gauss0" in m:
            mets[f"gauss0_n{PUBLISHED_N}"] = m["gauss0"]
        out[s] = {"source": r["source"], "selected": r["selected"], "k": r["k"], "metrics": mets}
    if setting == FX.FEDLEX:
        return out
    alt = FA._published(f"alternatives/alternatives_{setting}_{encoder}.json")
    band = FX._read(os.path.join(paths.EXTENSIONS, "raw", FA.BAND_FOLDER, f"band_{setting}_{encoder}.json"))
    for src, theirs, mine in ((alt, "sae_mi_n", "mi_n"),
                              (band if band and band.get("draws") == draws else None, "rband_mean_n", "random_n")):
        for p in (src or {"per_seed": []})["per_seed"]:
            for c, v in p["metrics"].items():
                if int(p["seed"]) in out and c.startswith(theirs) and c[len(theirs):].isdigit():
                    out[int(p["seed"])]["metrics"][mine + c[len(theirs):]] = v
    # Seeds beyond the coverage run: the published id run of the same splits, where it has them.
    idr = FA._published(f"id/id_{setting}_{encoder}.json") or {"per_seed": []}
    km = FA._published(f"id/id_kmeans_{setting}_{encoder}.json") or {"per_seed": []}
    km_by = {int(p["seed"]): p["metrics"] for p in km["per_seed"]}
    for p in idr["per_seed"]:
        s, m = int(p["seed"]), p["metrics"]
        if s not in out:
            mets = {"raw": m["raw"], "metadata_leace": m["metadata_leace"], f"mi_n{PUBLISHED_N}": m["sae_leace"]}
            if "kmeans_leace" in km_by.get(s, {}):
                mets["kmeans_leace"] = km_by[s]["kmeans_leace"]
            out[s] = {"source": "id run", "selected": p["selected"], "k": p["pseudo_groups_k"], "metrics": mets}
    return out


def _reproduction(rec: dict, ref) -> dict:
    """The refit against the published fit of the same seed: MI's top-32, K, and every shared row."""
    if not ref:
        return {"reference": None}
    out = {"reference": ref["source"],
           "mi_selection_reproduced": ([sorted(d["mi_top"]) for d in rec["diagnostics"]]
                                       == [sorted(int(j) for j in x) for x in ref["selected"]]),
           "k_reproduced": [d["pseudo_groups_k"] for d in rec["diagnostics"]] == [int(k) for k in ref["k"]], "diff": {}}
    for c, mref in ref["metrics"].items():
        mine = rec["metrics"].get(c)
        if mine:
            out["diff"][c] = {k: float(mine[k] - mref[k]) for k in ("linear_auc", "mlp_auc", "recall@10", "topic_purity",
                                                                   "gold_sim_spearman") if k in mine and k in mref}
    return out


def aggregate(per_seed: list) -> dict:
    """mean / SD / values per condition and metric over the seeds that have the condition (a budget
    above a seed's pool is missing in that seed)."""
    agg = {}
    for c in dict.fromkeys(c for p in per_seed for c in p["metrics"]):
        have = [p["metrics"][c] for p in per_seed if c in p["metrics"]]
        agg[c] = {}
        for k, v in have[0].items():
            if isinstance(v, dict):
                agg[c][k] = {kk: R._stat([m[k][kk] for m in have if kk in m.get(k, {})]) for kk in v}
            elif isinstance(v, (int, float)):
                agg[c][k] = R._stat([m[k] for m in have if k in m])
    return agg


def _progress(rec: dict, n: int) -> str:
    m, d = rec["metrics"], rec["diagnostics"]
    arms = [a for a, _, _ in ARMS if f"{a}_n{n}" in m]
    pool = [x["pool_size"] for x in d]
    return (f"pool {min(pool)}-{max(pool)}, K {sorted({x['pseudo_groups_k'] for x in d})}, "
            f"k-means ARI {np.mean([x['kmeans_ari'] for x in d]):.2f} | linear AUC at N = {n}: "
            + "  ".join(f"{a} {m[f'{a}_n{n}']['linear_auc']:.3f}" for a in arms))


def run_selectors(setting: str, encoder: str = "e5-small", seeds=SEEDS, budgets=BUDGETS, selectors=SELECTORS,
                  draws: int = DRAWS, gauss_draws: int = GAUSS_DRAWS, nonlinear=("mlp",), nonlinear_budgets=DETAIL,
                  fit_device=None, overwrite: bool = False, verbose: bool = True, **overrides) -> dict:
    """Every selector at every budget on one setting, all on one SAE per split.

    The SAE, the splits and the audit are the published runs': `protocols.fit_base` with the recipe
    and the seed, the registry's splits (Fedlex: the run's whole-law split, rebuilt and checked) and
    `revised.evaluate_fits`. The linear audit runs on every condition; the `nonlinear` probes on the
    references and on the `nonlinear_budgets` only. ``random_n<N>`` and ``gauss_n<N>`` average their
    draws within a seed. Each seed is compared with the published fit of that seed (MI's top-32, K
    and every row both have) and checkpointed, so an interrupted run resumes after the last seed.
    Asking for more seeds than a finished run has keeps its seeds and runs only the new ones."""
    unknown = [sel for sel in selectors if sel not in SCORERS]
    if unknown:
        raise ValueError(f"unknown selectors {unknown}; known: {list(SCORERS)}")
    seeds, budgets = [int(s) for s in seeds], sorted(int(n) for n in budgets)
    with artifacts.experiment(FOLDER):
        name = f"selectors_{setting}_{encoder}.json"
        path, part = artifacts.artifact_path(name), artifacts.artifact_path(name.replace(".json", ".partial.json"))
        design = {"budgets": budgets, "selectors": list(selectors), "draws": int(draws), "gauss_draws": int(gauss_draws),
                  "nonlinear": list(nonlinear), "nonlinear_budgets": sorted(int(n) for n in nonlinear_budgets),
                  "band": list(BAND), "overrides": overrides}
        done = {}
        if os.path.exists(path) and not overwrite:
            res = artifacts.load_result(path)
            if res["design"] != design or not set(res["seeds"]) <= set(seeds):
                raise ValueError(f"{path} was run with {res['design']} on seeds {res['seeds']}; pass overwrite=True")
            if set(res["seeds"]) == set(seeds):
                print(f"[cached] {path}")
                return res
            done = {int(p["seed"]): p for p in res["per_seed"]}     # more seeds asked: the finished ones are kept
        data, splits_of, run = FX.load_setting(setting, encoder)
        kw = R._kw(overrides)
        X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
        ref = _reference(setting, encoder, run, draws, gauss_draws)
        if os.path.exists(part) and not overwrite:
            prev = artifacts.load_result(part)
            if prev.get("design") == design:
                done.update({int(p["seed"]): p for p in prev["per_seed"]})
        if done:
            print(f"[selectors] {setting}/{encoder}: seeds {sorted(s for s in done if s in seeds)} already done, "
                  f"running {[s for s in seeds if s not in done]}")
        per_seed = []
        for s in seeds:
            if s in done:
                per_seed.append(done[s])
                continue
            t0 = time.time()
            rec = _seed_record(data, X, y, splits_of(s), s, setting, encoder, kw, budgets, selectors, draws,
                               gauss_draws, nonlinear, design["nonlinear_budgets"], fit_device, verbose)
            rec["reproduction"] = _reproduction(rec, ref.get(s))
            rec["seconds"] = round(time.time() - t0, 1)
            per_seed.append(rec)
            artifacts.save_result({"design": design, "per_seed": per_seed}, part)
            if verbose:
                rp = rec["reproduction"]
                show = max([n for n in rec["budgets"] if n <= max(HEADLINE)] or rec["budgets"][:1])
                print(f"[selectors] {setting}/{encoder} seed {s} ({rec['seconds']}s) {_progress(rec, show)}")
                if rp.get("reference"):
                    print(f"   published fit ({rp['reference']}): MI top-{PUBLISHED_N} reproduced "
                          f"{rp.get('mi_selection_reproduced')}, K {rp.get('k_reproduced')}; largest |linear AUC diff| "
                          + _fmt(max([abs(v["linear_auc"]) for v in rp.get("diff", {}).values() if "linear_auc" in v],
                                     default=None), d=4))
                else:
                    print("   no published fit of this seed to compare with")
        out = {"experiment": "latent_selectors", "corpus": setting, "encoder": encoder, "label": data.get("label", setting),
               "seeds": seeds, "design": design, "budgets": sorted(set.intersection(*(set(p["budgets"]) for p in per_seed))),
               "selectors": list(selectors), "conditions": list(per_seed[0]["metrics"]), "per_seed": per_seed,
               "agg": aggregate(per_seed),
               "config": {"recipe": kw, "selection": data["config"]["selection"],
                          "pseudo_groups": data["config"]["pseudo_groups"], "split": E._split_policy(data),
                          "audit": audit.PROTOCOL,
                          "operator": "protocols._fit_leace(X_tr, X_tr @ D_S^T) (Directions-LEACE, as SAE-LEACE)",
                          "pool": f"latents with training firing rate in {list(BAND)}: the set MI scores",
                          "ranking": "top N of one ranking per selector (nested); ties to the lower latent id; "
                                     "mi: the published rule, argsort(scores)[::-1]",
                          "dip": "diptest.dipstat on X_tr d_j, every training document",
                          "random": f"{draws} draws per SAE; draw j of seed s, fold f: uniform permutation of the pool "
                                    f"from default_rng([{FA.R1_SEED}, s, j, 1, f]), prefixes (focused_additions.latent_draw)",
                          "gauss": f"{gauss_draws} draws; unit-norm N(0, I) rows from default_rng([{FA.R1_SEED}, s, j, 0])"},
               "corpus_stats": R._corpus_stats(data),
               "environment": {**FX._environment(fit_device), "scipy": version("scipy"), "diptest": version("diptest"),
                               "python": platform.python_version(), "platform": platform.platform()}}
        artifacts.save_result(out, name)
        if os.path.exists(part):
            os.remove(part)
        return out


def run_cells(cells=CELLS, **kw) -> dict:
    """`run_selectors` on every ``(setting, encoder)`` in turn. A setting whose SAE leaves no eligible
    latent is reported and skipped; anything else raises."""
    out = {}
    for setting, encoder in cells:
        try:
            out[(setting, encoder)] = run_selectors(setting, encoder, **kw)
        except FX.BandTooSmall as err:
            print(f"[selectors] {setting}/{encoder} skipped: {err}")
    return out


def load(setting: str, encoder: str = "e5-small"):
    return FX._read(R.raw_path(FOLDER, f"selectors_{setting}_{encoder}.json"))


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _fmt(v, signed: bool = False, d: int = 3) -> str:
    return FX._fmt(v, signed, d)


def _vals(res: dict, cond: str, metric: str) -> list:
    return [p["metrics"].get(cond, {}).get(metric) for p in res["per_seed"]]


def _mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def _pm(v, n_seeds: int | None = None, d: int = 3) -> str:
    """Seed mean ± SD; ``(n = k)`` when only k of the seeds have the value."""
    v = [x for x in v if x is not None]
    if not v:
        return "--"
    s = f"{np.mean(v):.{d}f}" + (f" ± {np.std(v, ddof=1):.{d}f}" if len(v) > 1 else "")
    return s + (f" (n = {len(v)})" if n_seeds and len(v) < n_seeds else "")


def _diag(res: dict, get) -> list:
    """Per seed, the mean over its fits of ``get(diagnostics)`` (None where no fit has it)."""
    out = []
    for p in res["per_seed"]:
        v = []
        for dg in p["diagnostics"]:
            try:
                x = get(dg)
            except KeyError:
                x = None
            if x is not None:
                v.append(x)
        out.append(float(np.mean(v)) if v else None)
    return out


def _arms(res: dict) -> list:
    """The budgeted arms a run has, in table order."""
    have = {c.rpartition("_n")[0] for c in res["conditions"]}
    return [a for a, _, _ in ARMS if a in have]


def _runs(cells) -> list:
    """``(setting, encoder, result or None)`` per cell."""
    return [(k, e, load(k, e)) for k, e in cells]


def _all_arms(runs: list) -> list:
    have = {a for _, _, r in runs if r for a in _arms(r)}
    return [a for a, _, _ in ARMS if a in have]


def _best(res: dict, n: int, metric: str = "linear_auc"):
    """The k-means-free selector with the lowest seed-mean `metric` at budget n."""
    means = {a: _mean(_vals(res, f"{a}_n{n}", metric)) for a in KMEANS_FREE}
    means = {a: v for a, v in means.items() if v is not None}
    return min(means, key=means.get) if means else None


def _save(lines: list, name: str) -> str:
    md = "\n".join(lines)
    with artifacts.experiment(FOLDER):
        FA._save_md(md, name)
    return md


def table_settings(cells=CELLS) -> str:
    """Markdown: per setting, the fit sizes, the eligible pool, k-means' K and its agreement with the
    true classes (ARI on the training documents, post hoc), and the references' linear AUC."""
    lines = ["| Setting | classes | train docs per fit | eligible latents | K (silhouette) | k-means ARI | Raw | "
             "k-means LEACE | Metadata-LEACE | seeds |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, e, r in _runs(cells):
        if r is None:
            lines.append(f"| {title(k, e)} | not run | | | | | | | | |")
            continue
        dg = [d for p in r["per_seed"] for d in p["diagnostics"]]
        pool, K = [d["pool_size"] for d in dg], sorted({d["pseudo_groups_k"] for d in dg})
        lines.append(f"| {title(k, e)} | {len(r['corpus_stats']['classes'])} | {int(np.mean([d['n_train_docs'] for d in dg])):,} | "
                     f"{min(pool)}-{max(pool)} | {', '.join(map(str, K))} | {_pm(_diag(r, lambda d: d['kmeans_ari']), d=2)} | "
                     + " | ".join(_pm(_vals(r, c, "linear_auc")) for c in ("raw", "kmeans_leace", "metadata_leace"))
                     + f" | {r['seeds']} |")
    lines += ["", f"Eligible latents: training firing rate in {list(BAND)}, the set MI scores and every selector ranks "
              "(range over fits). k-means ARI: the silhouette-K grouping against the true classes on the training "
              "documents, seed mean ± SD (post hoc). The last three columns are linear AUC, seed mean ± SD."]
    return _save(lines, "table_settings.md")


def table_curves(cells=CELLS, metric: str = "linear_auc") -> str:
    """Markdown: per setting, `metric` of every arm at every budget (seed mean ± SD)."""
    lines = []
    for k, e, r in _runs(cells):
        if r is None:
            lines += [f"### {title(k, e)}: not run", ""]
            continue
        budgets = sorted({n for p in r["per_seed"] for n in p["budgets"]})
        best = {n: _best(r, n, metric) for n in budgets}
        lines += [f"### {title(k, e)} (seeds {r['seeds']})", "",
                  "References: " + "; ".join(f"{REF_LABEL[c]} {_pm(_vals(r, c, metric))}" for c in REFERENCES) + ".", "",
                  f"| {METRIC.get(metric, metric)} | " + " | ".join(f"N = {n}" for n in budgets) + " |",
                  "|---|" + "---|" * len(budgets)]
        for a in _arms(r):
            row = []
            for n in budgets:
                txt = _pm(_vals(r, f"{a}_n{n}", metric), len(r["seeds"]))
                row.append(f"**{txt}**" if a == best[n] else txt)
            lines.append(f"| {ROW[a]} | " + " | ".join(row) + " |")
        lines.append("")
    lines += ["Seed mean ± SD. Bold: the lowest seed mean among the k-means-free selectors "
              f"({', '.join(COL[a] for a in KMEANS_FREE)}). Random and Gaussian rows average their draws within a seed."]
    return _save(lines, f"table_curves_{metric}.md")


def table_headline(cells=CELLS, n: int = 8, metric: str = "linear_auc") -> str:
    """Markdown: `metric` at budget n, one row per setting and one column per arm (seed mean ± SD)."""
    runs = _runs(cells)
    arms = _all_arms(runs)
    head = ["Raw"] + [COL[a] for a in arms] + ["k-means LEACE", "Metadata-LEACE"]
    lines = [f"**{METRIC.get(metric, metric)} at N = {n}**", "", "| Setting | " + " | ".join(head) + " |",
             "|---|" + "---|" * len(head)]
    for k, e, r in runs:
        if r is None:
            lines.append(f"| {title(k, e)} | not run |" + " |" * (len(head) - 1))
            continue
        best, row = _best(r, n, metric), [_pm(_vals(r, "raw", metric))]
        for a in arms:
            txt = _pm(_vals(r, f"{a}_n{n}", metric), len(r["seeds"]))
            row.append(f"**{txt}**" if a == best else txt)
        row += [_pm(_vals(r, c, metric)) for c in ("kmeans_leace", "metadata_leace")]
        lines.append(f"| {title(k, e)} | " + " | ".join(row) + " |")
    lines += ["", "Seed mean ± SD. Bold: the lowest seed mean among the k-means-free selectors. `--`: the budget exceeds "
              "the eligible pool of that setting."]
    return _save(lines, f"table_headline_{metric}_n{n}.md")


def table_paired(cells=CELLS, n: int = 8, anchor: str = "mi", metric: str = "linear_auc") -> str:
    """Markdown: every arm's per-seed paired difference to `anchor` at budget n on `metric`: mean,
    95% t-interval over seeds and the signs (seeds above / below zero)."""
    runs = _runs(cells)
    arms = [a for a in _all_arms(runs) if a != anchor]
    lines = [f"**Δ {METRIC.get(metric, metric)} at N = {n}: arm − {COL[anchor]}**", "",
             "| Setting | " + " | ".join(COL[a] for a in arms) + " |", "|---|" + "---|" * len(arms)]
    for k, e, r in runs:
        if r is None:
            lines.append(f"| {title(k, e)} | not run |" + " |" * (len(arms) - 1))
            continue
        ref, row = _vals(r, f"{anchor}_n{n}", metric), []
        for a in arms:
            v = _vals(r, f"{a}_n{n}", metric)
            row.append(FX._cell_line(FX.seed_stats([x - z if x is not None and z is not None else None
                                                    for x, z in zip(v, ref)])))
        lines.append(f"| {title(k, e)} | " + " | ".join(row) + " |")
    lines += ["", "Per-seed paired difference on the same SAE and split: mean [95% t-interval over seeds] (seeds with a "
              "positive / negative difference). For leakage, negative means less leakage than the anchor."]
    return _save(lines, f"table_paired_{metric}_n{n}_vs_{anchor}.md")


def table_quality(cells=CELLS, reference: str = "oracle", budgets=DETAIL) -> str:
    """Markdown: every selector's agreement with `reference` -- the rank correlation of their scores
    over the eligible latents, and the share of its top-N set that is in the reference's top-N.
    ``reference="oracle"`` is the selector-quality diagnostic (post hoc); ``"mi"`` measures the
    coupling with the k-means rule."""
    runs = _runs(cells)
    sels = [a for a in _all_arms(runs) if a not in (reference, "gauss", "random")]
    lines = [f"**Rank correlation with {ROW[reference]}** (Spearman ρ of the scores over the eligible latents)", "",
             "| Setting | " + " | ".join(COL[a] for a in sels) + " |", "|---|" + "---|" * len(sels)]
    for k, e, r in runs:
        if r is None:
            lines.append(f"| {title(k, e)} | not run |" + " |" * (len(sels) - 1))
            continue
        lines.append(f"| {title(k, e)} | " + " | ".join(
            _pm(_diag(r, lambda d, a=a: d["spearman"][a][reference]), d=2) for a in sels) + " |")
    for n in budgets:
        lines += ["", f"**Top-{n} overlap with {ROW[reference]}** (share of the selector's top {n} in the reference's top {n})",
                  "", "| Setting | " + " | ".join(COL[a] for a in sels) + " | Random | chance N / pool |",
                  "|---|" + "---|" * (len(sels) + 2)]
        for k, e, r in runs:
            if r is None:
                lines.append(f"| {title(k, e)} | not run |" + " |" * (len(sels) + 1))
                continue
            row = [_pm(_diag(r, lambda d, a=a: d["overlap"][a][reference][str(n)]), len(r["seeds"]), 2)
                   for a in sels + ["random"]]
            lines.append(f"| {title(k, e)} | " + " | ".join(row)
                         + f" | {_fmt(_mean(_diag(r, lambda d: n / d['pool_size'] if n <= d['pool_size'] else None)), d=2)} |")
    lines += ["", "Seed mean ± SD of the fits' values. Random: the mean over the random draws."]
    return _save(lines, f"table_quality_vs_{reference}.md")


def table_displacement(cells=CELLS, budgets=DETAIL) -> str:
    """Markdown: per budget, the normalized displacement of every arm's map (the share of training
    variance it moves, `alternatives._removed_variance`) and the mean training firing rate of the
    latents each selector picks."""
    runs = _runs(cells)
    arms = _all_arms(runs)
    sae = [a for a in arms if a != "gauss"]
    lines = []
    for n in budgets:
        lines += [f"**Normalized displacement at N = {n}**", "", "| Setting | " + " | ".join(COL[a] for a in arms) + " |",
                  "|---|" + "---|" * len(arms)]
        for k, e, r in runs:
            if r is None:
                lines.append(f"| {title(k, e)} | not run |" + " |" * (len(arms) - 1))
                continue
            lines.append(f"| {title(k, e)} | " + " | ".join(
                _pm(_diag(r, lambda d, c=f"{a}_n{n}": d["displacement"][c]), len(r["seeds"])) for a in arms) + " |")
        lines += ["", f"**Mean training firing rate of the selected latents at N = {n}**", "",
                  "| Setting | " + " | ".join(COL[a] for a in sae) + " | eligible pool |", "|---|" + "---|" * (len(sae) + 1)]
        for k, e, r in runs:
            if r is None:
                lines.append(f"| {title(k, e)} | not run |" + " |" * len(sae))
                continue
            lines.append(f"| {title(k, e)} | " + " | ".join(
                _pm(_diag(r, lambda d, a=a: d["mean_freq"][a][str(n)]), len(r["seeds"])) for a in sae)
                + f" | {_pm(_diag(r, lambda d: float(np.mean(d['freq']))))} |")
        lines.append("")
    lines += ["Seed mean ± SD of the fits' values. Displacement: tr((I − A)ᵀ Σ (I − A)) / tr Σ on the training documents, the "
              "statistic displacement-matched PCA is matched on. A selector that displaces more may suppress more for that "
              "reason alone."]
    return _save(lines, "table_displacement.md")


def table_detail(cells=CELLS, budgets=DETAIL) -> str:
    """Markdown: Main Table 3's direction-selection block per setting, extended by the selectors:
    linear AUC, the nonlinear probes, the utility metrics and the displacement at each of `budgets`."""
    lines = []
    for k, e, r in _runs(cells):
        if r is None:
            lines += [f"### {title(k, e)}: not run", ""]
            continue
        probes = [f"{p}_auc" for p in r["design"]["nonlinear"]]
        mets = ["linear_auc"] + probes + list(utility(k))
        head = [f"N = {n}: {METRIC.get(m, m)}" for n in budgets for m in mets + ["displaced"]]
        lines += [f"### {title(k, e)} (seeds {r['seeds']}; {r['design']['draws']} random draws, "
                  f"{r['design']['gauss_draws']} Gaussian)", "", "| Selection | " + " | ".join(head) + " |",
                  "|---|" + "---|" * len(head)]
        for a in _arms(r):
            row = []
            for n in budgets:
                row += [_pm(_vals(r, f"{a}_n{n}", m), len(r["seeds"])) for m in mets]
                row.append(_pm(_diag(r, lambda d, c=f"{a}_n{n}": d["displacement"][c]), len(r["seeds"])))
            lines.append(f"| {ROW[a]} | " + " | ".join(row) + " |")
        for c in REFERENCES:
            row = []
            for _ in budgets:
                row += [_pm(_vals(r, c, m)) for m in mets]
                row.append("--" if c == "raw" else _pm(_diag(r, lambda d, c=c: d["displacement"][c])))
            lines.append(f"| {REF_LABEL[c]} | " + " | ".join(row) + " |")
        lines.append("")
    lines += ["Seed mean ± SD. Every budgeted row is Directions-LEACE on N directions of one SAE per split; the references "
              "do not depend on N. Within-class NN@10 kept: the share of each held-out document's 10 nearest same-class "
              "neighbours that the map keeps."]
    return _save(lines, "table_detail.md")


def table_table3(cells=TABLE3_CELLS, n: int = PUBLISHED_N) -> str:
    """Markdown: Main Table 3's direction-selection block and references at budget n, with the
    selectors as rows 7-10 -- one sub-table per column of Table 3 (linear AUC, MLP AUC, utility) and
    one column per setting. Every row of a setting is on the same SAEs, splits and audit."""
    runs = _runs(cells)

    def primary(setting: str) -> str:
        return (UTILITY.get(setting, ("recall@10",)) or ("within_class_nn10",))[0]

    metrics = [("linear_auc", lambda k: "linear_auc", METRIC["linear_auc"])]
    if any(r and "mlp" in r["design"]["nonlinear"] and n in r["design"]["nonlinear_budgets"] for _, _, r in runs):
        metrics.append(("mlp_auc", lambda k: "mlp_auc", METRIC["mlp_auc"]))
    metrics.append(("utility", primary, "Utility ↑"))
    lines = [f"**Main Table 3's direction-selection block at N = {n}, with the k-means-free selectors**", ""]
    for name, metric_of, label in metrics:
        head = [title(k, e) + (f": {METRIC[primary(k)]}" if name == "utility" else "") for k, e, _ in runs]
        lines += [f"*{label}*", "", "| | Condition | " + " | ".join(head) + " |", "|---|---|" + "---|" * len(runs)]
        for num, row, arm in TABLE3_ROWS:
            lines.append(f"| {num} | {row} | " + " | ".join(
                "not run" if r is None else _pm(_vals(r, f"{arm}_n{n}", metric_of(k)), len(r["seeds"]))
                for k, _, r in runs) + " |")
        for c, row in TABLE3_REFERENCES:
            lines.append(f"| | {row} | " + " | ".join("not run" if r is None else _pm(_vals(r, c, metric_of(k)))
                                                      for k, _, r in runs) + " |")
        lines.append("")
    lines += ["Seed mean ± SD; seeds " + "; ".join(f"{title(k, e)} {r['seeds']}" for k, e, r in runs if r) + ". "
              "Rows 1-3 are Main Table 3's direction-selection rows, recomputed with rows 7-10 on one SAE per split "
              "(operator fixed: Directions-LEACE); the random rows average their draws within a seed. The oracle row "
              "uses the labels: a post hoc reference, not a label-free method."]
    return _save(lines, f"table_table3_n{n}.md")


def table_reproduction(cells=CELLS) -> str:
    """Markdown: the refit SAEs against the published fits of the same seeds."""
    lines = ["| Setting | published fit | MI top-32 reproduced | K reproduced | MI rows | references | references, utility | "
             "random SAE | Gaussian | estimator check | MI prefix |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    util = ("recall@10", "topic_purity", "gold_sim_spearman")

    def worst(reps: list, pick, metrics=("linear_auc",)) -> str:
        v = [abs(m[k]) for rp in reps for c, m in rp.get("diff", {}).items() if pick(c) for k in metrics if k in m]
        return f"{max(v):.4f} ({len(v)})" if v else "--"

    for k, e, r in _runs(cells):
        if r is None:
            lines.append(f"| {title(k, e)} | not run | | | | | | | | | |")
            continue
        by_source: dict = {}
        for p in r["per_seed"]:
            by_source.setdefault(p["reproduction"].get("reference") or "none", []).append(str(p["seed"]))
        reps = [p["reproduction"] for p in r["per_seed"] if p["reproduction"].get("reference")]
        dg = [d for p in r["per_seed"] for d in p["diagnostics"]]
        bad = sum(v["zero_score"] + v["out_of_pool"] for d in dg for v in d["mi_prefix"].values())
        sources = "; ".join(src + ": seeds " + ", ".join(v) for src, v in by_source.items())
        lines.append(
            f"| {title(k, e)} | {sources} | "
            f"{sum(bool(rp.get('mi_selection_reproduced')) for rp in reps)}/{len(reps)} | "
            f"{sum(bool(rp.get('k_reproduced')) for rp in reps)}/{len(reps)} | "
            f"{worst(reps, lambda c: c.startswith('mi_n'))} | {worst(reps, lambda c: c in REFERENCES)} | "
            f"{worst(reps, lambda c: c in REFERENCES, util)} | "
            f"{worst(reps, lambda c: c.startswith('random_n'))} | {worst(reps, lambda c: c.startswith('gauss'))} | "
            f"{max(d['mi_estimator_max_diff'] for d in dg):.1e} | {'ok' if not bad else f'{bad} ineligible'} |")
    lines += ["", "Published fit: the run each seed is compared with (`none`: that seed was never published). "
              "Reproduced: seeds whose refit matches the published fit, of the seeds that have one. *MI rows*, *references*, *random SAE* and "
              "*Gaussian*: the largest |linear AUC difference| to the published value over seeds and budgets (number of "
              "values compared). *References, utility*: the same for the references' utility metric; the references use no "
              "SAE, so a zero here confirms the published splits and maps, and a nonzero linear-AUC difference beside it "
              "comes from the audit (the scikit-learn version). *Estimator check*: the largest difference between the "
              "vectorized MI estimator and the pipeline's MI scores on the same k-means grouping. *MI prefix*: whether any "
              "MI top-N set contains a latent MI cannot score (zero score or outside the pool)."]
    return _save(lines, "table_reproduction.md")
