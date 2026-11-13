"""Alternative erasers, alternative label-free selectors, and matched controls.

The plan's comparison is PCA at rank 1 against SAE-LEACE at N = 32. That leaves
several questions open, and this module answers them on the *same* splits and
under the same audit as `revised.run_id`, so every number here is comparable to
New Table 2 cell by cell:

**How far does the PCA baseline go?**   `pca_top{r}` for r in `RANKS`, plus
`pca_matched_var`, whose rank is chosen per split so that PCA removes the same
share of training variance as SAE-LEACE does -- equal distortion rather than
equal rank.

**Does the MI selector earn its keep?**  `sae_random` removes 32 randomly chosen
active latents through the same mechanism: same SAE, same rank, same operator,
no selection signal. `kmeans_leace` drops the SAE altogether and erases the
one-hot silhouette-chosen k-means grouping, the proxy the MI score is computed
against.

**Are there label-free selectors that beat MI-against-k-means?**  Two that use
neither labels nor the pairing (which is label-derived: pair links are
cross-class by construction, and two-colouring them recovers a binary nuisance):

* `sae_stability` -- a document-level factor should be rediscovered by an SAE
  trained on a different subsample. Each latent scores by how well its firing
  pattern is matched by some latent of SAEs trained on disjoint halves of the
  training documents.
* `sae_crossenc` -- a document-level factor lives in the text, so it should
  appear in a second encoder's dictionary too. Each latent scores by its best
  firing-pattern match among the latents of an SAE trained on another encoder's
  embeddings of the same documents.

**Is LEACE doing the work, or would suppressing the latents suffice?**
`sae_recon_suppressed` zeroes the selected latents and decodes, using the SAE
reconstruction as the deconfounded embedding; `sae_recon` decodes without
suppressing anything, which separates the autoencoder's own lossiness from the
effect of suppression. Both are nonlinear maps, unlike every other arm here, so
they cannot be transferred as frozen affine maps.

Outputs live beside the plan's, in their own folders::

    <root>/raw/alternatives/   alternatives_<corpus>_<encoder>.json
    <root>/tables/extra/       table_alternatives.{tex,md}, table_selector_overlap.{tex,md}
    <root>/figures/extra/      figure_frontier.{png,pdf}

Plan Sec. 5.4 rules out a PCA rank sweep as an argument for *superiority*. The
use here is the opposite: to find out what the baselines and controls can do.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace as _replace

import numpy as np

import artifacts
import experiments as E
import ood
import protocols as P
import revised as R
from sae import encode_features, train_sae

#: PCA removal ranks. 1 is the plan's PCA-1; 32 matches the SAE budget N.
RANKS = (1, 2, 3, 4, 8, 16, 32, 64)
CORPORA = ("semeval", "cap", "swiltra", "bills_tweets", "scotus_oyez_lexis")
FOLDER = "alternatives"
#: Firing-frequency band for the new selectors, as in the MI scorer.
FREQ_BAND = (0.02, 0.98)
#: Disjoint training halves used by the stability selector.
N_RESAMPLES = 2

#: Arms other than the PCA ranks, in reading order, with their labels.
ARMS = {
    "raw": "Raw",
    "pca_matched_var": "PCA, variance-matched",
    "kmeans_leace": "k-means LEACE (no SAE)",
    "sae_random": "SAE-LEACE, 32 random latents",
    "sae_mi": "SAE-LEACE, MI top-32",
    "sae_stability": "SAE-LEACE, stability top-32",
    "sae_crossenc": "SAE-LEACE, cross-encoder top-32",
    "sae_recon": "SAE decode only (nothing zeroed)",
    "sae_recon_suppressed": "SAE decode, MI top-32 zeroed",
    "metadata_leace": "Metadata-LEACE (supervised)",
}
#: A cheaper subset for `nonlinear_arms`, when probing every arm costs too much.
#: The default is None -- gradient-boosted and MLP probes on every condition.
NONLINEAR_ARMS = ("raw", "pca_top1", "pca_top2", "pca_top3", "pca_matched_var",
                  "kmeans_leace", "sae_mi", "sae_stability", "sae_crossenc",
                  "sae_recon_suppressed", "metadata_leace")


#: A split plus frozen erasers, which is all `revised.evaluate_fits` needs.
_Fit = R.Erasers


# --------------------------------------------------------------------------- #
# Label-free selectors
# --------------------------------------------------------------------------- #
def _firing(feats) -> np.ndarray:
    return (np.asarray(feats) > 0).astype(np.float32)


def _best_match(F_a: np.ndarray, F_b: np.ndarray) -> np.ndarray:
    """For every column of `F_a`, its largest Pearson correlation with a column
    of `F_b` (both are binary firing patterns over the same documents)."""
    A = F_a - F_a.mean(axis=0, keepdims=True)
    B = F_b - F_b.mean(axis=0, keepdims=True)
    A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-8
    B /= np.linalg.norm(B, axis=0, keepdims=True) + 1e-8
    return (A.T @ B).max(axis=1)


def _band(feats, band=FREQ_BAND) -> np.ndarray:
    freq = (np.asarray(feats) > 0).mean(axis=0)
    return (freq >= band[0]) & (freq <= band[1])


def stability_scores(base, X, n_resamples: int = N_RESAMPLES, device=None, **kw) -> np.ndarray:
    """How reliably each latent is rediscovered by SAEs trained on disjoint
    halves of the training documents (mean best firing-pattern match)."""
    tr = np.asarray(base.split.train_docs)
    rng = np.random.default_rng(int(base.split.seed) + 991)
    halves = np.array_split(rng.permutation(len(tr)), n_resamples)
    F_main = _firing(base.feats_train)
    scores = np.zeros(F_main.shape[1], dtype=np.float64)
    for h in halves:
        sae_h, _ = train_sae(np.asarray(X, np.float32)[tr[h]], expansion_factor=kw["expansion_factor"],
                             k=kw["k"], n_epochs=kw["n_epochs"], seed=int(base.split.seed),
                             device=device, verbose=False)
        scores += _best_match(F_main, _firing(encode_features(sae_h, np.asarray(X, np.float32)[tr])))
    return (scores / len(halves)) * _band(base.feats_train)


def cross_encoder_scores(base, other_X, device=None, **kw) -> np.ndarray:
    """Each latent's best firing-pattern match in a dictionary learned on another
    encoder's embeddings of the same training documents."""
    tr = np.asarray(base.split.train_docs)
    Z = np.asarray(other_X, np.float32)[tr]
    sae_o, _ = train_sae(Z, expansion_factor=kw["expansion_factor"], k=kw["k"],
                         n_epochs=kw["n_epochs"], seed=int(base.split.seed), device=device,
                         verbose=False)
    return _best_match(_firing(base.feats_train), _firing(encode_features(sae_o, Z))) * _band(base.feats_train)


# --------------------------------------------------------------------------- #
# Erasers
# --------------------------------------------------------------------------- #
def _pca_erasers(X_tr: np.ndarray, ranks) -> dict:
    """All PCA removals from one SVD: T(x) = mu + (I - B'B)(x - mu)."""
    X_tr = np.asarray(X_tr, np.float32)
    mu = X_tr.mean(axis=0, keepdims=True)
    Vt = np.linalg.svd(X_tr - mu, full_matrices=False)[2]

    def make(r):
        B = Vt[:r]
        return lambda Z: (np.asarray(Z, np.float32) - mu
                          - ((np.asarray(Z, np.float32) - mu) @ B.T) @ B + mu).astype(np.float32)
    return {r: make(int(min(r, X_tr.shape[1] - 1))) for r in ranks}


def _removed_variance(eraser, X_tr: np.ndarray) -> float:
    """Share of the training variance an affine eraser removes."""
    d = X_tr.shape[1]
    A = ood.affine(eraser, d)[0]
    M = np.eye(d) - A.T
    Xc = np.asarray(X_tr, np.float64) - np.asarray(X_tr, np.float64).mean(axis=0)
    S = Xc.T @ Xc / len(Xc)
    return float(np.trace(M @ S @ M.T) / np.trace(S))


def _matched_rank(X_tr: np.ndarray, share: float) -> int:
    """Fewest principal directions whose variance share reaches `share`. The tolerance
    keeps a share that equals a cumulative share (a map that is itself a PCA removal)
    from rounding up to the next rank: float32 erasers put such a share up to ~2e-8
    off, while consecutive cumulative shares of real embeddings sit >= ~1e-3 apart."""
    Xc = np.asarray(X_tr, np.float64) - np.asarray(X_tr, np.float64).mean(axis=0)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    return int(np.searchsorted(np.cumsum(lam) / lam.sum(), share - 1e-6) + 1)


def _recon_eraser(base, suppress=()):
    """Decode the SAE code, optionally with `suppress` latents zeroed. Nonlinear."""
    W = np.asarray(base.decoder, np.float32)
    b = base.sae.b_pre.detach().cpu().numpy().astype(np.float32)
    drop = np.asarray(suppress, dtype=int)

    def apply(Z):
        F = encode_features(base.sae, np.asarray(Z, np.float32)).copy()
        if len(drop):
            F[:, drop] = 0.0
        return (F @ W + b).astype(np.float32)
    return apply


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run_alternatives(corpus: str, encoder: str = E.PRIMARY_ENCODER, ranks=RANKS, seeds=None,
                     budget_grid=(), nonlinear=("gb", "mlp"), nonlinear_arms=None,
                     other_encoder: str | None = None, selectors: bool = True,
                     device=None, fit_device=None, overwrite: bool = False,
                     verbose: bool = True, **overrides) -> dict:
    """Every arm on one corpus, sharing one SAE per split.

    `nonlinear` probes run on every condition unless `nonlinear_arms` restricts
    them (`NONLINEAR_ARMS` is a ready-made cheaper subset). `budget_grid` adds
    the MI and random-latent arms at several budgets N, which is what shows
    whether MI's advantage over random selection holds along the curve rather
    than only at N = 32.
    """
    with artifacts.experiment(FOLDER):
        name = f"alternatives_{corpus}_{encoder}.json"
        path = artifacts.artifact_path(name)
        if os.path.exists(path) and not overwrite:
            print(f"[cached] {path}")
            return artifacts.load_result(path)
        kw = R._kw(overrides)
        data = R.load(corpus, encoder, device=device, verbose=verbose)
        seeds = tuple(seeds or data["config"]["seeds"])
        X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
        other = other_encoder or ("mini" if encoder != "mini" else "e5-small")
        other_X = (np.asarray(R.load(corpus, other, device=device, verbose=False)["emb"], np.float32)
                   if selectors else None)
        conditions = (["raw"] + [f"pca_top{r}" for r in ranks]
                      + [a for a in ARMS if a != "raw"
                         and (selectors or a not in ("sae_stability", "sae_crossenc"))]
                      + [f"sae_{kind}_n{n}" for n in budget_grid for kind in ("mi", "random")])
        per_seed = []
        for s in seeds:
            t0 = time.time()
            fits, diag = [], []
            for sp in R._splits(data, s):
                base = P.fit_base(X, sp, expansion_factor=kw["expansion_factor"], k=kw["k"],
                                  n_epochs=kw["n_epochs"],
                                  pseudo_groups=data["config"]["pseudo_groups"], device=fit_device)
                main = P.fit_pipeline(X, y, sp, base=base, selection="topn",
                                      top_features=kw["top_features"], erase=kw["erase"],
                                      kmeans_auto=True, random_control=True, pca_grid=(),
                                      kmeans_grid=(), device=fit_device)
                er = {"raw": main.erasers["raw"], "metadata_leace": main.erasers["metadata_leace"],
                      "sae_mi": main.erasers["sae_leace"],
                      "kmeans_leace": main.erasers["kmeans_leace_auto"],
                      "sae_random": main.erasers["random_latents"],
                      "sae_recon": _recon_eraser(base),
                      "sae_recon_suppressed": _recon_eraser(base, main.selected)}
                pca = _pca_erasers(X[sp.train_docs], ranks)
                er.update({f"pca_top{r}": fn for r, fn in pca.items()})
                share = _removed_variance(er["sae_mi"], X[sp.train_docs])
                r_star = _matched_rank(X[sp.train_docs], share)
                er["pca_matched_var"] = _pca_erasers(X[sp.train_docs], (r_star,))[r_star]
                for n in budget_grid:
                    grid_fit = P.fit_pipeline(X, y, sp, base=base, selection="topn",
                                              top_features=int(n), erase=kw["erase"],
                                              pca_grid=(), kmeans_grid=(), kmeans_auto=False,
                                              random_control=True, device=fit_device)
                    er[f"sae_mi_n{n}"] = grid_fit.erasers["sae_leace"]
                    er[f"sae_random_n{n}"] = grid_fit.erasers["random_latents"]
                row = {"sae_removed_variance": share, "matched_rank": r_star,
                       "selected_mi": [int(j) for j in main.selected],
                       "pseudo_groups_k": int(base.pseudo_groups)}
                if selectors:
                    for arm, score in (("sae_stability",
                                        stability_scores(base, X, device=fit_device, **kw)),
                                       ("sae_crossenc",
                                        cross_encoder_scores(base, other_X, device=fit_device, **kw))):
                        fitted = P.fit_pipeline(X, y, sp, base=_replace(base, scores=np.asarray(score)),
                                                selection="topn", top_features=kw["top_features"],
                                                erase=kw["erase"], pca_grid=(), kmeans_grid=(),
                                                kmeans_auto=False, random_control=False,
                                                device=fit_device)
                        er[arm] = fitted.erasers["sae_leace"]
                        row[f"selected_{arm.split('_')[1]}"] = [int(j) for j in fitted.selected]
                fits.append(_Fit(sp, er))
                diag.append(row)
            res = R.evaluate_fits(fits, X, y, conditions=tuple(conditions),
                                  topics=data.get("topics"), nonlinear=nonlinear,
                                  nonlinear_conditions=nonlinear_arms)
            res.pop("hits", None)
            res["diagnostics"] = diag
            res["seconds"] = round(time.time() - t0, 1)
            per_seed.append(res)
            if verbose:
                m = res["metrics"]
                print(f"[alt] {corpus}/{encoder} seed {s} ({res['seconds']}s) "
                      f"matched rank {diag[0]['matched_rank']}: "
                      + "  ".join(f"{a} {m[a]['linear_auc']:.3f}"
                                  for a in ("pca_top1", "pca_top3", "pca_matched_var", "sae_mi",
                                            "sae_random", "kmeans_leace", "sae_recon_suppressed")
                                  if a in m))
        out = {"experiment": "alternatives", "corpus": corpus, "label": data["label"],
               "encoder": encoder, "other_encoder": other if selectors else None,
               "seeds": [int(s) for s in seeds], "ranks": list(ranks),
               "conditions": list(conditions), "budget_grid": [int(n) for n in budget_grid],
               "arms": {k: v for k, v in ARMS.items() if k in conditions},
               "n_classes": int(len(np.unique(y))), "class_rank": int(len(np.unique(y)) - 1),
               "paired": data["paired"], "sae_budget": int(kw["top_features"]),
               "per_seed": per_seed, "agg": R.aggregate(per_seed),
               "selector_overlap": _overlap(per_seed),
               "corpus_stats": R._corpus_stats(data),
               "config": R._config(data, overrides, nonlinear=list(nonlinear))}
        artifacts.save_result(out, name)
        return out


def _overlap(per_seed: list) -> dict:
    """Mean Jaccard overlap between the selectors' chosen latent sets."""
    keys = [k for k in ("mi", "stability", "crossenc") if f"selected_{k}" in per_seed[0]["diagnostics"][0]]
    out = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            vals = [len(set(d[f"selected_{a}"]) & set(d[f"selected_{b}"]))
                    / max(1, len(set(d[f"selected_{a}"]) | set(d[f"selected_{b}"])))
                    for s in per_seed for d in s["diagnostics"]]
            out[f"{a}|{b}"] = float(np.mean(vals))
    return out


def run_all(corpora=CORPORA, encoder: str = E.PRIMARY_ENCODER, **kw) -> dict:
    return {c: run_alternatives(c, encoder, **kw) for c in corpora}


# --------------------------------------------------------------------------- #
# Frozen transfer for PCA at several ranks
# --------------------------------------------------------------------------- #
#: Ranks transferred out of distribution. Rank 1 repeats the plan's PCA-1, as a
#: check that this path reproduces `revised.run_ood`.
OOD_RANKS = (1, 2, 3, 4, 8, 16, 32)


def _run_ood_family(prefix: str, matrix, seeds, transforms, labels: dict, fit_maps,
                    maps_note: str, device=None, overwrite: bool = False,
                    verbose: bool = True) -> dict:
    """Frozen transfer of one family of label-free maps: each fitted on the donor's ID
    training split (the split `revised.run_id` used), frozen, and scored on the six
    MTEB targets like every other frozen map.

    ``fit_maps(todo, data, X, split, encoder, donor, seed)`` returns
    ``({transform: eraser}, {fit diagnostics})``. A cached setting that lacks some of
    `transforms` is extended in place: only the missing maps are fitted and scored, on
    the seeds it already holds, after checking each seed's training split against the
    cached fingerprint; its Raw scores are kept.
    """
    with artifacts.experiment(FOLDER):
        targets = {t: ood.load_target(t) for t in ood.TARGETS}
        out = {}
        for enc, donor in matrix:
            key, name = f"{enc}/{donor}", f"{prefix}_{enc}_{donor}.json"
            path = artifacts.artifact_path(name)
            res = artifacts.load_result(path) if os.path.exists(path) and not overwrite else None
            todo = [c for c in transforms if res is None or c not in res["transforms"]]
            if not todo:
                print(f"[cached] {path}")
                out[key] = res
                continue
            data = R.load(donor, enc, device=device, verbose=verbose)
            X = np.asarray(data["emb"], np.float32)
            embs = ood.embed_targets(enc, device=device, verbose=verbose)
            cached = res is not None
            if not cached:
                overlap = ood.donor_target_overlap(data["texts"], targets)
                if any(overlap.values()):
                    raise ValueError(f"{donor} shares texts with the targets {overlap}")
                res = {"experiment": prefix, "encoder": enc, "donor": donor,
                       "donor_label": data["label"], "donor_role": None,
                       "seeds": [int(s) for s in seeds], "fit_on": "train",
                       "transforms": [], "labels": {"raw": "Raw"},
                       "tasks": ood.task_records(targets),
                       "raw": {t: ood.score_target(targets[t], embs[t]) for t in targets},
                       "per_seed": [{"seed": int(s), "maps_file": None, "fit": {},
                                     "operator_checks": {}, "scores": {}} for s in seeds],
                       "protocol": {"mteb_version": ood.MTEB_VERSION,
                                    "encoder_model": ood.encoder_revision(enc),
                                    "prompt_style": ood.PROMPT_STYLE,
                                    "target_prompts": ood._prompts(enc), "maps": maps_note,
                                    "donor_target_overlap": overlap,
                                    "normalization": "maps applied without renormalization; "
                                                     "cosine"}}
            elif [int(s) for s in seeds] != res["seeds"]:
                print(f"[{prefix}] {name}: adding {todo} on its cached seeds {res['seeds']}")
            for p in res["per_seed"]:
                sp = R._splits(data, p["seed"])[0]
                tr = np.asarray(sp.train_docs)
                split_id = ood._split_id(tr)
                if p["fit"].get("split_id", split_id) != split_id:
                    raise RuntimeError(f"{name} seed {p['seed']}: the cached maps were fitted on "
                                       "another training split, so nothing was written")
                fns, extra = fit_maps(todo, data, X, sp, enc, donor, int(p["seed"]))
                maps = {c: ood.affine(fns[c], X.shape[1]) for c in todo}
                mu = X[tr].astype(np.float64).mean(axis=0)
                p["fit"].update({"fit_on": "train", "split_id": split_id,
                                 "n_train_docs": int(len(tr)), **extra})
                p["operator_checks"].update({c: ood.check_operator(maps[c], mu, X[tr][:256], fns[c])
                                             for c in todo})
                p["scores"].update({c: {t: ood.score_target(targets[t], embs[t], maps[c])
                                        for t in targets} for c in todo})
                if verbose:
                    print(f"[{prefix}] {key} seed {p['seed']}" + (f" {extra}" if extra else "")
                          + ": " + "  ".join(f"{c} " + " ".join(
                              f"{p['scores'][c][t]['score']:.3f}" for t in ood.LEGAL) for c in todo))
            if cached:
                res.setdefault("extended", []).append(
                    {"added": todo, "note": "fitted on the cached seeds' training splits "
                                            "(fingerprints checked); cached Raw scores kept"})
            have = set(res["transforms"]) | set(todo)
            res["transforms"] = ([c for c in transforms if c in have]
                                 + [c for c in res["transforms"] if c not in transforms])
            res["labels"].update({c: labels[c] for c in todo})
            artifacts.save_result(res, name)
            out[key] = res
        return out


def run_ood_ranks(matrix=R.OOD_MATRIX, ranks=OOD_RANKS, matched_var: bool = True,
                  seeds=R.OOD_SEEDS, device=None, overwrite: bool = False,
                  verbose: bool = True) -> dict:
    """PCA removal at each rank, fitted on the donor's ID training split and frozen, on
    the six MTEB targets: how the transfer cost grows with the number of erased
    directions. With `matched_var`, also the PCA map matched to SAE-LEACE's removed
    variance -- per seed, the fewest principal directions whose share of the training
    variance reaches the share removed by that seed's frozen SAE-LEACE map, read from
    the maps `revised.run_id` saved, so it is matched to exactly the map that was
    transferred. No SAE is trained. Files from a smaller grid are extended, not redone.
    """
    ranks = [int(r) for r in ranks]
    transforms = [f"pca_top{r}" for r in ranks] + (["pca_matched_var"] if matched_var else [])
    labels = {**{f"pca_top{r}": f"PCA, top {r} removed" for r in ranks},
              "pca_matched_var": "PCA, variance-matched to SAE-LEACE"}

    def fit_maps(todo, data, X, sp, enc, donor, seed):
        tr = np.asarray(sp.train_docs)
        want = [int(c.replace("pca_top", "")) for c in todo if c.startswith("pca_top")]
        fns = {f"pca_top{r}": fn for r, fn in _pca_erasers(X[tr], want).items()} if want else {}
        extra = {}
        if "pca_matched_var" in todo:
            maps, _, tr_saved = R.load_maps(enc, donor, seed)
            if not np.array_equal(np.sort(np.asarray(tr_saved)), np.sort(tr)):
                raise ValueError(f"{enc}/{donor} seed {seed}: the saved SAE-LEACE map was fitted "
                                 "on other training documents than this split")
            A_sae, c_sae = maps["sae_leace"]
            share = _removed_variance(lambda Z: np.asarray(Z, np.float64) @ A_sae + c_sae, X[tr])
            r_star = _matched_rank(X[tr], share)
            fns["pca_matched_var"] = _pca_erasers(X[tr], (r_star,))[r_star]
            extra = {"sae_removed_variance": share, "matched_rank": int(r_star)}
        return fns, extra

    return _run_ood_family("ood_ranks", matrix, seeds, transforms, labels, fit_maps,
                           "PCA removal refit on the donor's ID training split (the split "
                           "revised.run_id used), then frozen", device, overwrite, verbose)


def _ood_table(prefix: str, stem: str, caption: str, note: str, matrix, n_boot: int,
               plan_transforms) -> list:
    """A frozen-transfer change table for one family of maps, with the plan's own
    transferred maps (`plan_transforms`, read from its OOD summary) beside it."""
    import revised_exhibits as X

    results = {}
    with artifacts.experiment(FOLDER):
        for enc, donor in matrix:
            p = R.raw_path(FOLDER, f"{prefix}_{enc}_{donor}.json")
            if os.path.exists(p):
                results[f"{enc}/{donor}"] = artifacts.load_result(p)
    if not results:
        raise FileNotFoundError(f"no {prefix} results; run run_{prefix}() first")
    summary = ood.summarize(results, n_boot=n_boot, save=False)
    plan = None
    p = R.raw_path("ood", "ood_summary.json")
    if os.path.exists(p):
        plan = artifacts.load_result(p)
    rows = []
    for key in summary["order"]:
        v = summary["settings"][key]
        if rows:
            rows.append(X.SEP)
        head = [X.ENCODER.get(v["encoder"], v["encoder"]), X.CORPUS.get(v["donor"], v["donor"])]
        for c in summary["transforms"]:
            ch = v["changes"][c]
            w = ch["worst_task"]
            rows.append(head + [summary["labels"][c], X.signed(ch["legal_macro"]["delta"]),
                                X.ci(ch["legal_macro"]["ci"]), v["tasks"][w["task"]]["label"],
                                X.signed(w["delta"]),
                                X.signed(ch[v["sts"][0]]["delta"]) if v["sts"] else "NR"])
            head = ["", ""]
        if plan and key in plan.get("settings", {}):
            pv = plan["settings"][key]
            for c in plan_transforms:
                if c not in pv["changes"]:
                    continue
                ch = pv["changes"][c]
                w = ch["worst_task"]
                rows.append(["", "", plan["labels"].get(c, c), X.signed(ch["legal_macro"]["delta"]),
                             X.ci(ch["legal_macro"]["ci"]), pv["tasks"][w["task"]]["label"],
                             X.signed(w["delta"]),
                             X.signed(ch[pv["sts"][0]]["delta"]) if pv["sts"] else "NR"])
    header = ["Encoder", "Donor", "Transform", "Mean legal ΔnDCG@10", "95% CI",
              "Smallest-change task", "Task Δ", "STS ΔSpearman"]
    with artifacts.experiment(None):
        return X.write_table(stem, "extra", caption, [(None, header, rows)], note=note)


def table_ood_ranks(matrix=R.OOD_MATRIX, n_boot: int = 10000) -> list:
    """Transfer cost by PCA rank, beside the plan's three transferred maps."""
    return _ood_table(
        "ood_ranks", "table_ood_ranks",
        "Frozen transfer by PCA removal rank, with the plan's two other transferred "
        "maps for comparison.",
        f"Changes in the seed-averaged score (transformed − Raw) on the six MTEB "
        f"targets, paired bootstrap with {n_boot:,} resamples, over the same three "
        "donor fits. PCA maps are refit on each donor's ID training split and "
        "frozen, exactly as the other maps are; rank 1 repeats the plan's PCA-1 and "
        "should match Table C3. Metadata-LEACE and SAE-LEACE rows are read from the "
        "plan's OOD summary.", matrix, n_boot, ("metadata_leace", "sae_leace"))


def table_ood_kmeans(matrix=R.OOD_MATRIX, n_boot: int = 10000) -> list:
    """Transfer cost of k-means LEACE, beside the plan's three transferred maps."""
    return _ood_table(
        "ood_kmeans", "table_ood_kmeans",
        "Frozen transfer of k-means LEACE (no SAE), with the plan's three transferred "
        "maps for comparison.",
        f"Changes in the seed-averaged score (transformed − Raw) on the six MTEB "
        f"targets, paired bootstrap with {n_boot:,} resamples, over the same three "
        "donor fits. Both k-means maps are fitted on each donor's ID training split and "
        "frozen: LEACE on the one-hot k-means grouping, with K chosen by silhouette or set "
        "to the number of donor classes C (count-only: it uses how many sources there "
        "are, never which document is which). PCA-1, Metadata-LEACE and SAE-LEACE rows "
        "are read from the plan's OOD summary.", matrix, n_boot,
        ("pca_top1", "metadata_leace", "sae_leace"))


# --------------------------------------------------------------------------- #
# Exhibits
# --------------------------------------------------------------------------- #
def _results(corpora=CORPORA, encoder: str = E.PRIMARY_ENCODER) -> dict:
    out = {}
    for c in corpora:
        p = R.raw_path(FOLDER, f"alternatives_{c}_{encoder}.json")
        if os.path.exists(p):
            out[c] = artifacts.load_result(p)
    return out


def _util_key(corpus: str) -> str:
    return "topic_purity" if corpus == "cap" else "recall@10"


def table_alternatives(corpora=CORPORA, encoder: str = E.PRIMARY_ENCODER,
                       stem: str = "table_alternatives", where: str = "extra",
                       wide: bool = False) -> list:
    """Every arm's leakage and utility, on the splits of New Table 2. `stem`, `where`
    and `wide` place it among the paper's exhibits (Table D9)."""
    import revised_exhibits as X

    res = _results(corpora, encoder)
    if not res:
        raise FileNotFoundError("no alternatives results; run run_all() first")
    keys = [c for c in corpora if c in res]

    def block(metric_of, cols=keys):
        rows = []
        for r in RANKS:
            row = [f"PCA, top {r} removed"]
            for c in cols:
                note = " †" if r == res[c]["class_rank"] else ""
                note += " ‡" if r == res[c]["sae_budget"] else ""
                row.append(X.pm(res[c]["agg"].get(f"pca_top{r}", {}).get(metric_of(c))) + note)
            rows.append(row)
        for arm in ARMS:
            if arm == "raw":
                continue
            if arm == "kmeans_leace":
                rows.append(X.SEP)
            rows.append([ARMS[arm]] + [X.pm(res[c]["agg"].get(arm, {}).get(metric_of(c)))
                                       for c in cols])
            if arm == "pca_matched_var":
                ranks = {c: sorted({d["matched_rank"] for s in res[c]["per_seed"]
                                    for d in s.get("diagnostics", []) if "matched_rank" in d})
                         for c in cols}
                rows.append(["Its rank r (range across fits)"]
                            + [("NR" if not ranks[c] else str(ranks[c][0]) if len(ranks[c]) == 1
                                else f"{ranks[c][0]}–{ranks[c][-1]}") for c in cols])
        # The budgets every column carries. A corpus rerun with a finer grid (the class
        # combinations use the full N sweep) would otherwise add rows that are NR in
        # every other column; its own sweep is Table D2 and the coverage tables.
        grids = [{int(n) for n in res[c].get("budget_grid", []) if int(n) != res[c]["sae_budget"]}
                 for c in cols]
        grids = [g for g in grids if g]
        grid = sorted(set.intersection(*grids)) if grids else []
        if grid:
            rows.append(X.SEP)
            for n in grid:
                for kind, lab in (("mi", f"SAE-LEACE, MI top-{n}"),
                                  ("random", f"SAE-LEACE, {n} random latents")):
                    rows.append([lab] + [X.pm(res[c]["agg"].get(f"sae_{kind}_n{n}", {})
                                              .get(metric_of(c))) for c in cols])
        rows.append(X.SEP)
        rows.append(["Raw (no erasure)"] + [X.pm(res[c]["agg"]["raw"].get(metric_of(c))) for c in cols])
        return rows

    a = block(lambda c: "linear_auc")
    ukeys = [c for c in keys if _util_key(c) in res[c]["agg"]["raw"]]
    b = block(_util_key, ukeys)
    gb = block(lambda c: "gb_auc")
    head_a = ["Condition"] + [f"{X.CORPUS[c]} AUC" for c in keys]
    head_b = ["Condition"] + [f"{X.CORPUS[c]} " + ("topic purity" if c == "cap" else "R@10")
                              for c in ukeys]
    mlp = block(lambda c: "mlp_auc")
    panels = [("Panel A — Linear probe AUC (0.5: chance)", head_a, a),
              ("Panel B — Semantic utility (higher is better)", head_b, b),
              ("Panel C — Gradient-boosted probe AUC (0.5: chance)", head_a, gb),
              ("Panel D — MLP probe AUC (0.5: chance)", head_a, mlp)]
    caption = ("Alternative erasers, label-free selectors and matched controls, "
               + X.ENCODER.get(encoder, encoder) + ". Mean ± SD over "
               + X.nseeds(res) + ", on the splits of New Table 2.")
    note = ("† rank equal to C − 1; ‡ rank equal to the SAE budget N = 32. The "
            "variance-matched PCA arm picks, per split, the fewest directions whose "
            "variance share reaches the share SAE-LEACE removes (the row below it gives that "
            "rank). The random-latent arm uses the same SAE, rank and operator as "
            "SAE-LEACE with the selection signal removed; the k-means arm erases the "
            "one-hot pseudo-grouping directly, with no SAE. The two reconstruction arms "
            "decode the SAE code, with and without the selected latents zeroed; both are "
            "nonlinear maps and cannot be transferred as frozen affine maps. Every arm is "
            "fitted on the training documents and audited on the held-out documents. "
            "Panels C and D use the same audit folds as Panel A, unflipped. The MI and "
            "random rows at N < 32 share each split's SAE with the N = 32 arms. Bills–Tweets "
            "has neither pairs nor topic labels, so it has no Panel B column. NR: not "
            "evaluated in this run.")
    with artifacts.experiment(None):
        return X.write_table(stem, where, caption, panels, note=note, wide=wide)


def table_selector_overlap(corpora=CORPORA, encoder: str = E.PRIMARY_ENCODER,
                           stem: str = "table_selector_overlap", where: str = "extra",
                           wide: bool = False) -> list:
    """How much the three label-free selectors agree on what to erase (Table D13)."""
    import revised_exhibits as X

    res = _results(corpora, encoder)
    pairs = sorted({k for r in res.values() for k in r.get("selector_overlap", {})})
    if not pairs:
        raise FileNotFoundError("no selector overlaps; run with selectors=True")
    label = {"mi": "MI", "stability": "stability", "crossenc": "cross-encoder"}
    header = ["Corpus"] + [" ∩ ".join(label[x] for x in p.split("|")) for p in pairs]
    rows = [[X.CORPUS[c]] + [X.num(res[c]["selector_overlap"].get(p)) for p in pairs]
            for c in res]
    with artifacts.experiment(None):
        return X.write_table(
            stem, where,
            "Agreement between the label-free selectors (mean Jaccard overlap of the "
            "32 selected latents).", [(None, header, rows)],
            note="Averaged over every fit. A low overlap with comparable outcomes means "
                 "different latent sets support the same erasure; a high overlap means the "
                 "selectors are finding the same structure by different routes.", wide=wide)


def table_encoder_ranks(corpora=R.ORIGINAL, encoders=R.ENCODER_STUDY,
                        stem: str = "table_encoder_ranks", where: str = "extra",
                        wide: bool = False) -> list:
    """PCA at every rank of `R.ENCODER_PCA_GRID` beside SAE-LEACE, per encoder (Table B6).

    PCA-1 alone is not budget-matched to SAE-LEACE, which erases a subspace of up to
    32 directions. Each row here gives the PCA rank
    curve, the smallest rank whose linear AUC reaches SAE-LEACE's, and on the paired
    corpora the R@10 PCA keeps at that rank against SAE-LEACE's.
    """
    import revised_exhibits as X

    cells = {key: c for key, c in X.encoder_cells(corpora).items()
             if key[0] in corpora and key[1] in encoders and len(c.get("pca_ranks", [])) > 2}
    if not cells:
        raise FileNotFoundError("no encoder cells carry the PCA rank grid; run "
                                "run_encoder_cell with pca_ranks=R.ENCODER_PCA_GRID first")
    ranks = sorted({r for c in cells.values() for r in c["pca_ranks"]})
    got = lambda c, cond, k: (c["agg"].get(cond, {}).get(k) or {}).get("mean")  # noqa: E731
    panels = []
    for corpus in corpora:
        paired = any(got(c, "sae_leace", "recall@10") is not None
                     for (k, _), c in cells.items() if k == corpus)
        header = (["Encoder"] + [f"PCA-{r}" for r in ranks] + ["SAE-\nLEACE", "Rank to\nmatch"]
                  + (["SAE-\nLEACE", "PCA at\nthat rank", "Δ (PCA\n− SAE)"] if paired else []))
        groups = ([(None, 1), ("Linear probe AUC", len(ranks) + 1), (None, 1)]
                  + ([("R@10", 3)] if paired else []))
        rows = []
        for enc in encoders:
            c = cells.get((corpus, enc))
            if c is None:
                rows.append([X.ENCODER.get(enc, enc)] + ["NR"] * (len(header) - 1))
                continue
            auc = {r: got(c, f"pca_top{r}", "linear_auc") for r in ranks}
            sae = got(c, "sae_leace", "linear_auc")
            match = next((r for r in ranks if None not in (auc[r], sae) and auc[r] <= sae), None)
            top = max((r for r in ranks if auc[r] is not None), default=0)
            row = ([X.ENCODER.get(enc, enc)] + [X.num(auc[r]) for r in ranks]
                   + [X.num(sae), str(match) if match else f"> {top}"])
            if paired:
                s_r = got(c, "sae_leace", "recall@10")
                p_r = got(c, f"pca_top{match}", "recall@10") if match else None
                row += [X.num(s_r), X.num(p_r) if match else "—",
                        X.signed(p_r - s_r) if None not in (p_r, s_r) else "—"]
            rows.append(row)
        panels.append((X.CORPUS[corpus], header, rows, groups))
    with artifacts.experiment(None):
        return X.write_table(
            stem, where,
            "PCA removal rank against SAE-LEACE across the ten encoders (held-out linear AUC, "
            "mean over " + X.nseeds(cells) + ").", panels,
            note="Rank to match: the smallest PCA rank whose linear AUC is at or below "
                 "SAE-LEACE's in the same cell, on the same splits and audit folds; '> r' means "
                 "no rank in the grid reaches it. The R@10 columns set SAE-LEACE against PCA at "
                 "that rank, which removes at least as much linear leakage, so a positive Δ "
                 "means PCA matches SAE-LEACE's leakage while keeping more retrieval. The PCA "
                 "arms were added to the cached cells without refitting the SAE. NR: not run.",
            wide=wide, tabcolsep=2.5 if wide else None)


def _label_points(ax, labels, obstacles=(), fontsize: float = 5.5) -> None:
    """Label curve points without collisions: try four offsets around each point,
    in the order given, and leave a label out when none is clear of the labels
    already placed, the markers and the frame. Call after the layout is final."""
    px = ax.figure.dpi / 72.0
    frame = ax.get_window_extent()
    taken = []
    for xy, half in obstacles:
        cx, cy = ax.transData.transform(xy)
        taken.append((cx - half * px, cy - half * px, cx + half * px, cy + half * px))
    for x, y, text, colour in labels:
        cx, cy = ax.transData.transform((x, y))
        w, h = 0.6 * fontsize * len(text) * px, 0.8 * fontsize * px
        for dx, dy, ha, va in ((3, 2, "left", "bottom"), (3, -2, "left", "top"),
                               (-3, 2, "right", "bottom"), (-3, -2, "right", "top")):
            x0 = cx + dx * px - (w if ha == "right" else 0.0)
            y0 = cy + dy * px - (h if va == "top" else 0.0)
            box = (x0, y0, x0 + w, y0 + h)
            inside = frame.x0 <= box[0] and box[2] <= frame.x1 and frame.y0 <= box[1] and box[3] <= frame.y1
            if inside and all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]
                              for b in taken):
                taken.append(box)
                ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, fontsize=fontsize, color=colour)
                break


def figure_frontier(corpora=CORPORA, encoder: str = E.PRIMARY_ENCODER) -> list:
    """Leakage against utility: the PCA rank curve, the SAE budget curves, and
    every other affine arm as a point. Better is up and to the left.

    The two reconstruction arms are not drawn: they sit at AUC >= 0.99 with
    utility at or below Raw's (Table Alternatives, Panels A-B), and drawing them
    would squeeze every other arm into a thin band. For the same reason a Raw
    point far below every eraser is marked on the panel's lower edge, with its value.
    """
    import revised_exhibits as X

    res = _results(corpora, encoder)
    if not res:
        raise FileNotFoundError("no alternatives results; run run_all() first")
    # Bills–Tweets has neither pairs nor topic labels, so it has no utility axis.
    keys = [c for c in corpora if c in res and _util_key(c) in res[c]["agg"]["raw"]]
    plt = X._plt()
    ncol = min(2, len(keys))
    nrow = -(-len(keys) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.4 * nrow), squeeze=False)
    sweeps = (("pca", "#9467bd", "#6a3d9a", "s", "PCA, rank sweep"),
              ("mi", "#1f78b4", "#1f78b4", "o", "SAE-LEACE, MI, N sweep"),
              ("random", "#b15928", "#b15928", "o", "SAE-LEACE, random, N sweep"))
    points = {"sae_mi": ("#1f78b4", "o"), "sae_stability": ("#a6cee3", "^"),
              "sae_crossenc": ("#33a02c", "v"), "sae_random": ("#b15928", "X"),
              "kmeans_leace": ("#ff7f00", "P"), "metadata_leace": ("#e31a1c", "D"),
              "pca_matched_var": ("#9467bd", "s")}
    pending = []                                  # labels are placed once the layout is final
    for ax, c in zip(axes.ravel(), keys):
        r, uk = res[c], _util_key(c)
        got = lambda arm, k: (r["agg"].get(arm, {}).get(k) or {}).get("mean")  # noqa: E731
        arms = {"pca": [(k, f"pca_top{k}") for k in r["ranks"]],
                "mi": [(n, f"sae_mi_n{n}") for n in r.get("budget_grid", [])],
                "random": [(n, f"sae_random_n{n}") for n in r.get("budget_grid", [])]}
        labels, obstacles, ys, drawn = [], [], [], set()
        for kind, colour, ink, mk, lab in sweeps:
            pts = [(t, got(a, "linear_auc"), got(a, uk)) for t, a in arms[kind]]
            pts = [p for p in pts if p[1] is not None and p[2] is not None]
            if len(pts) < 2:
                continue
            ax.plot([p[1] for p in pts], [p[2] for p in pts], "-" + mk, color=colour,
                    ms=3.5 if mk == "s" else 3, lw=1.1, label=lab, zorder=2)
            labels += [(x, y, str(t), ink) for t, x, y in pts]
            obstacles += [((x, y), 2.0) for _, x, y in pts]
            ys += [y for _, _, y in pts]
            drawn.add(kind)
        # The plan's budget sweep, only when this run has no MI curve of its own
        # (that one is on identical splits, so it is preferred).
        budget = R.raw_path("budget", f"budget_{c}_{encoder}.json")
        if "mi" not in drawn and os.path.exists(budget):
            bud = artifacts.load_result(budget)
            bx = [bud["curve"][str(n)]["linear_auc"]["mean"] for n in bud["grid"]]
            by = [bud["curve"][str(n)].get(uk, {}).get("mean") for n in bud["grid"]]
            if all(v is not None for v in by):
                ax.plot(bx, by, "-o", color="#1f78b4", ms=3, lw=1.1, label="SAE-LEACE, N sweep",
                        zorder=2)
                ys += by
        for arm, (colour, mk) in points.items():
            x, y = got(arm, "linear_auc"), got(arm, uk)
            if x is not None and y is not None:
                ax.plot(x, y, mk, color=colour, ms=5.5, mec="black", mew=0.4, ls="none",
                        label=ARMS.get(arm, arm), zorder=3)
                obstacles.append(((x, y), 3.2))
                ys.append(y)
        rx, ry = got("raw", "linear_auc"), got("raw", uk)
        lo, hi = min(ys), max(ys)
        far = ry < lo - 1.25 * (hi - lo)
        lo = lo if far else min(lo, ry)
        pad = 0.07 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0.48, 1.02)
        raw_style = dict(color=X.COLOR["raw"], mec="black", mew=0.4, ls="none", zorder=4)
        if far:
            ax.plot(rx, lo - pad, "v", ms=6, clip_on=False, **raw_style)
            ax.annotate(f"Raw: {ry:.2f}", (rx, lo - pad), xytext=(-5, 4),
                        textcoords="offset points", ha="right", va="bottom", fontsize=6.5,
                        color=X.MUTED)
            ax.plot([], [], "o", ms=5.5, label="Raw", **raw_style)
        else:
            ax.plot(rx, ry, "o", ms=5.5, label="Raw", **raw_style)
            obstacles.append(((rx, ry), 3.2))
        ax.axvline(0.5, color="black", ls=":", lw=0.7)
        ax.set_title(X.CORPUS[c])
        ax.set_xlabel("Linear probe AUC (lower is better)")
        ax.set_ylabel("Topic purity" if c == "cap" else "R@10")
        ax.grid(True)
        pending.append((ax, labels, obstacles))
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    seen, handles = set(), []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append((h, l))
    rows = -(-len(handles) // 3)
    fig.tight_layout(rect=(0, (0.17 * rows + 0.12) / fig.get_figheight(), 1, 1))
    fig.legend([h for h, _ in handles], [l for _, l in handles], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.0))
    for ax, labels, obstacles in pending:
        _label_points(ax, labels, obstacles)
    with artifacts.experiment(None):
        return X.save_fig(fig, "figure_frontier", "extra")


def figure_pc1_recall(log_y: bool = False) -> list:
    """The parent's Figure 5, for every map the ten-encoder study ran: PC1
    variance share against the percentage increase in Recall@1, one point per
    corpus x encoder setting.

    Theirs covers metadata-LEACE only, on full-corpus fits and more dataset
    mixtures; ours is the held-out, frozen-map protocol on the ten-encoder study's
    three paired corpora, and it adds the metadata-free eraser and PCA removal at
    ranks 1 and 3.
    """
    import revised_exhibits as X
    from scipy import stats

    cells = {}
    for k in R.ORIGINAL:
        for e in R.ENCODER_STUDY:
            c = X.raw("encoders", f"encoder_{k}_{e}.json")
            if c and "recall@1" in c["agg"]["raw"]:
                cells[(k, e)] = c
    if not cells:
        raise FileNotFoundError("no encoder cells with recall@1; run the ten-encoder study")
    conds = [c for c in ("metadata_leace", "sae_leace", "pca_top1", "pca_top3")
             if all(c in v["conditions"] for v in cells.values())]
    title = {**X.LABEL, "pca_top3": "PCA-3"}
    plt = X._plt()
    ncol = min(2, len(conds))
    nrow = -(-len(conds) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.1 * nrow), sharex=True,
                             sharey=True, squeeze=False)
    for ax, cond in zip(axes.ravel(), conds):
        xs, ys, ks = [], [], []
        for (k, _e), c in cells.items():
            raw = c["agg"]["raw"]["recall@1"]["mean"]
            xs.append(c["diagnostics"]["pc1_var_share"])
            ys.append((c["agg"][cond]["recall@1"]["mean"] - raw) / raw * 100)
            ks.append(k)
        x, y = np.asarray(xs), np.asarray(ys)
        for k in dict.fromkeys(ks):                       # one colour per corpus
            m = np.array([kk == k for kk in ks])
            ax.plot(x[m], y[m], "o", color=X.CORPUS_COLOR[k], ms=5, mec="black", mew=0.4,
                    ls="none", label=X.CORPUS[k], zorder=3)
        # least squares fit with a 95% interval for the mean, as the parent draws it
        fit = stats.linregress(x, y)
        gx = np.linspace(x.min(), x.max(), 100)
        resid = y - (fit.intercept + fit.slope * x)
        se = np.sqrt((resid ** 2).sum() / (len(x) - 2)) * np.sqrt(
            1 / len(x) + (gx - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
        t = stats.t.ppf(0.975, len(x) - 2)
        ax.plot(gx, fit.intercept + fit.slope * gx, color="#e31a1c", lw=1.6, zorder=2)
        ax.fill_between(gx, fit.intercept + fit.slope * gx - t * se,
                        fit.intercept + fit.slope * gx + t * se, color="#e31a1c", alpha=0.15,
                        lw=0, zorder=1)
        r = stats.pearsonr(x, y)
        ax.set_title(f"{title.get(cond, cond)}\ncorrelation: {r.statistic:.2f} (p = {r.pvalue:.3f})")
        ax.grid(True)
        if log_y:
            ax.set_yscale("log")
    for ax in axes.ravel()[len(conds):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Ratio of total variance explained by first PC")
    for ax in axes[:, 0]:
        ax.set_ylabel("Percentage increase at Recall@1")
    axes.ravel()[len(conds) - 1].legend(loc="lower right")
    fig.tight_layout()
    with artifacts.experiment(None):
        return X.save_fig(fig, "figure_pc1_recall" + ("_log" if log_y else ""), "extra")


# --------------------------------------------------------------------------- #
# k-means LEACE without an SAE: the group count, and frozen transfer
# --------------------------------------------------------------------------- #
#: LEACE on the one-hot k-means grouping of the training documents, ``(eraser, labels)``;
#: shared with `revised.run_id_kmeans`, which scores the same arm on New Table 2's fits.
_kmeans_eraser = R.kmeans_eraser


def run_kmeans_count(corpus: str, encoder: str = E.PRIMARY_ENCODER, ks=R.K_GRID, seeds=None,
                     nonlinear=("gb", "mlp"), nonlinear_arms="count", overwrite: bool = False,
                     verbose: bool = True) -> dict:
    """k-means LEACE with K chosen by silhouette, at every K in `ks`, and at the
    number of classes C, on the splits of `run_alternatives`. No SAE is trained.

    K = C is not metadata-free -- it uses how many sources there are, though never
    which document belongs to which -- so it is a count-only reference. The
    silhouette arm repeats `kmeans_leace` of `run_alternatives` on the same splits.
    ``nonlinear_arms="count"`` runs the GB and MLP probes on the silhouette and
    K = C arms only; None probes every arm. Each grouping's adjusted Rand index
    with the true classes is recorded after fitting, for description only.
    """
    from sklearn.metrics import adjusted_rand_score

    with artifacts.experiment(FOLDER):
        name = f"kmeans_count_{corpus}_{encoder}.json"
        path = artifacts.artifact_path(name)
        if os.path.exists(path) and not overwrite:
            print(f"[cached] {path}")
            return artifacts.load_result(path)
        data = R.load(corpus, encoder, verbose=verbose)
        seeds = tuple(seeds or data["config"]["seeds"])
        X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
        n_classes = int(len(np.unique(y)))
        ks = tuple(sorted({int(k) for k in ks} | {n_classes}))
        conditions = ("kmeans_auto",) + tuple(f"kmeans_k{k}" for k in ks)
        probe_on = (("kmeans_auto", f"kmeans_k{n_classes}") if nonlinear_arms == "count"
                    else nonlinear_arms)
        per_seed = []
        for s in seeds:
            t0 = time.time()
            fits, diag = [], []
            for sp in R._splits(data, s):
                tr = np.asarray(sp.train_docs)
                K, sil = P.choose_pseudo_groups(X[tr], seed=sp.seed)
                er, ari = {}, {}
                er["kmeans_auto"], labels = _kmeans_eraser(X[tr], K, sp.seed)
                ari["auto"] = float(adjusted_rand_score(y[tr], labels))
                for k in ks:
                    er[f"kmeans_k{k}"], labels = _kmeans_eraser(X[tr], k, sp.seed)
                    ari[str(k)] = float(adjusted_rand_score(y[tr], labels))
                fits.append(_Fit(sp, er))
                diag.append({"k_auto": int(K), "silhouette": {str(k): v for k, v in sil.items()},
                             "ari_with_classes": ari})
            res = R.evaluate_fits(fits, X, y, conditions=conditions, topics=data.get("topics"),
                                  nonlinear=nonlinear, nonlinear_conditions=probe_on)
            res.pop("hits", None)
            res["diagnostics"] = diag
            res["seconds"] = round(time.time() - t0, 1)
            per_seed.append(res)
            if verbose:
                m = res["metrics"]
                print(f"[kmeans] {corpus} seed {s} ({res['seconds']}s), silhouette K "
                      f"{[d['k_auto'] for d in diag]}: " + "  ".join(
                          f"{c.replace('kmeans_', '')} {m[c]['linear_auc']:.3f}" for c in conditions))
        out = {"experiment": "kmeans_count", "corpus": corpus, "label": data["label"],
               "encoder": encoder, "seeds": [int(s) for s in seeds], "ks": list(ks),
               "n_classes": n_classes, "conditions": list(conditions),
               "nonlinear_conditions": None if probe_on is None else list(probe_on),
               "paired": data["paired"], "per_seed": per_seed, "agg": R.aggregate(per_seed),
               "corpus_stats": R._corpus_stats(data),
               "config": R._config(data, {}, nonlinear=list(nonlinear), sae=None)}
        artifacts.save_result(out, name)
        return out


def run_ood_kmeans(matrix=R.OOD_MATRIX, seeds=R.OOD_SEEDS, generous: bool = True,
                   k_generous: int | None = None, device=None, overwrite: bool = False,
                   verbose: bool = True) -> dict:
    """k-means LEACE fitted on each donor's ID training split and frozen, applied to the
    six MTEB targets: K chosen by silhouette, K = C, and (with `generous`) the
    pre-specified K = `GENEROUS_K` of the confirmatory test. No SAE: each map is one
    k-means and one LEACE fit. Files without the generous map are extended, not redone.
    """
    k_generous = int(k_generous or GENEROUS_K)
    transforms = ["kmeans_auto", "kmeans_count"] + (["kmeans_generous"] if generous else [])
    labels = {"kmeans_auto": "k-means LEACE, silhouette K",
              "kmeans_count": "k-means LEACE, K = C (count-only)",
              "kmeans_generous": f"k-means LEACE, K = {k_generous} (pre-specified)"}

    def fit_maps(todo, data, X, sp, enc, donor, seed):
        tr = np.asarray(sp.train_docs)
        n_classes = int(len(np.unique(data["source"])))
        fns, extra = {}, {"n_classes": n_classes}
        if "kmeans_auto" in todo:
            K, _ = P.choose_pseudo_groups(X[tr], seed=sp.seed)
            fns["kmeans_auto"], extra["k_auto"] = _kmeans_eraser(X[tr], K, sp.seed)[0], int(K)
        if "kmeans_count" in todo:
            fns["kmeans_count"] = _kmeans_eraser(X[tr], n_classes, sp.seed)[0]
        if "kmeans_generous" in todo:
            fns["kmeans_generous"] = _kmeans_eraser(X[tr], k_generous, sp.seed)[0]
            extra["k_generous"] = k_generous
        return fns, extra

    return _run_ood_family("ood_kmeans", matrix, seeds, transforms, labels, fit_maps,
                           "LEACE on a k-means grouping of the donor's ID training split (the "
                           "split revised.run_id used), then frozen", device, overwrite, verbose)


def table_kmeans_count(corpora=CORPORA, encoder: str = E.PRIMARY_ENCODER,
                       stem: str = "table_kmeans_count", where: str = "extra",
                       wide: bool = False, keep: str = "ABCDEFG",
                       alt_name: str = "Table Alternatives") -> list:
    """k-means LEACE by group count, beside SAE-LEACE and Metadata-LEACE. `keep` names
    the panels to print (relettered in order); `alt_name` is how the table refers to
    `table_alternatives`, which it reads its SAE-LEACE and Metadata-LEACE values from."""
    import revised_exhibits as X

    km = {}
    with artifacts.experiment(FOLDER):
        for c in corpora:
            p = R.raw_path(FOLDER, f"kmeans_count_{c}_{encoder}.json")
            if os.path.exists(p):
                km[c] = artifacts.load_result(p)
    if not km:
        raise FileNotFoundError("no kmeans_count results; run run_kmeans_count() first")
    alt = _results(corpora, encoder)
    keys = [c for c in corpora if c in km]

    def block(metric_of, cols):
        rows = []
        for c in cols:
            r, ref = km[c], alt.get(c, {}).get("agg", {})
            m, C = metric_of(c), r["n_classes"]
            k_auto = sorted({d["k_auto"] for s in r["per_seed"] for d in s["diagnostics"]})
            rows.append([X.CORPUS[c], str(C),
                         str(k_auto[0]) if len(k_auto) == 1 else f"{k_auto[0]}–{k_auto[-1]}",
                         X.pm(r["agg"]["kmeans_auto"].get(m)), X.pm(r["agg"][f"kmeans_k{C}"].get(m)),
                         X.pm(ref.get("sae_mi", {}).get(m)),
                         X.pm(ref.get("metadata_leace", {}).get(m))])
        return rows

    head = ["Corpus", "C", "Silhouette\nK", "k-means LEACE,\nsilhouette K",
            "k-means LEACE,\nK = C", "SAE-LEACE", "Metadata-\nLEACE"]
    ukeys = [c for c in keys if _util_key(c) in km[c]["agg"]["kmeans_auto"]]
    ks = sorted({k for c in keys for k in km[c]["ks"]})
    diags = {c: [d for s in km[c]["per_seed"] for d in s["diagnostics"]] for c in keys}
    by_k = [[X.CORPUS[c]] + [(X.pm(km[c]["agg"][f"kmeans_k{k}"].get("linear_auc"))
                              + (" †" if k == km[c]["n_classes"] else ""))
                             if k in km[c]["ks"] else "NR" for k in ks] for c in keys]
    by_u = [[X.CORPUS[c]] + [(X.pm(km[c]["agg"][f"kmeans_k{k}"].get(_util_key(c)))
                              + (" †" if k == km[c]["n_classes"] else ""))
                             if k in km[c]["ks"] else "NR" for k in ks] for c in ukeys]
    ari = [[X.CORPUS[c]] + [X.num(float(np.mean([d["ari_with_classes"][key] for d in diags[c]])))
                            if all(key in d["ari_with_classes"] for d in diags[c]) else "NR"
                            for key in ["auto"] + [str(k) for k in ks]] for c in keys]
    titled = {"A": ("Linear probe AUC (0.5: chance)", head, block(lambda c: "linear_auc", keys)),
              "B": ("Semantic utility (R@10; CAP: topic purity)", head, block(_util_key, ukeys)),
              "C": ("Gradient-boosted probe AUC", head, block(lambda c: "gb_auc", keys)),
              "D": ("MLP probe AUC", head, block(lambda c: "mlp_auc", keys)),
              "E": ("k-means LEACE linear AUC at each fixed K († K = C)",
                    ["Corpus"] + [f"K = {k}" for k in ks], by_k),
              "F": ("Adjusted Rand index of the grouping with the true classes (post hoc)",
                    ["Corpus", "Silhouette K"] + [f"K = {k}" for k in ks], ari),
              "G": ("k-means LEACE utility at each fixed K († K = C; R@10, CAP: topic purity)",
                    ["Corpus"] + [f"K = {k}" for k in ks], by_u)}
    # G sits beside E, the same sweep scored on the other axis.
    letter = {p: "ABCDEFG"[i] for i, p in enumerate(p for p in "ABCDEGF" if p in keep)}
    panels = [(f"Panel {letter[p]} — {titled[p][0]}", *titled[p][1:]) for p in letter]
    with artifacts.experiment(None):
        return X.write_table(
            stem, where,
            "k-means LEACE without an SAE, by group count, E5-small. Mean ± SD over "
            + X.nseeds(km) + f", on the splits of {alt_name}.", panels, wide=wide,
            note="k-means LEACE erases the one-hot k-means grouping of the training documents "
                 "(rank K − 1); no SAE is involved. The silhouette arm is the metadata-free "
                 f"method and repeats {alt_name}'s k-means row. K = C sets the group count "
                 "to the number of sources, so it is a count-only reference rather than a "
                 "metadata-free method. Silhouette K is the range chosen across fits. SAE-LEACE "
                 f"and Metadata-LEACE are {alt_name}'s values on the same splits. "
                 + ("The GB and MLP probes ran on the silhouette and K = C arms. "
                    if {"C", "D"} & set(letter) else "")
                 + (f"Panel {letter['G']} scores the same sweep on the utility axis. "
                    if "G" in letter else "")
                 + (f"Panel {letter['F']} compares each grouping with the true classes after "
                    "fitting, to show how closely it matches the nuisance; nothing is fitted "
                    "on it. " if "F" in letter else "")
                 + "Bills–Tweets has no utility column. NR: not run.")


# --------------------------------------------------------------------------- #
# Confirmatory test: a generous group count on held-out encoders
# --------------------------------------------------------------------------- #
#: Fixed on the development encoder alone (E5-small; Table D10, Figure 2) before any
#: held-out encoder was run, and not to be changed after seeing those results. On
#: E5-small, K = 8 -- the largest group count of `R.K_GRID` -- brought k-means LEACE
#: within 0.002 linear AUC of Metadata-LEACE on both multi-class corpora, and moved the
#: binary corpora by at most 0.005 AUC and 0.008 R@10 against K = 2.
GENEROUS_K = 8
GK_DEV_ENCODER = E.PRIMARY_ENCODER
GK_HELD_OUT = tuple(e for e in R.ENCODER_STUDY if e != E.PRIMARY_ENCODER)
GK_CORPORA = R.ID_CORPORA
GK_SEEDS = R.ENCODER_SEEDS
#: The pre-specified hypotheses, each judged per held-out corpus x encoder cell:
#:   H1, multi-class nuisance (C > 2): at K = GENEROUS_K the held-out linear AUC is at or
#:       below the silhouette K's and at most GK_META_MARGIN above Metadata-LEACE's, and
#:       utility (R@10; CAP: topic purity) is at most GK_UTILITY_DROP below the silhouette K's.
#:   H2, binary nuisance (C = 2): against the silhouette K, K = GENEROUS_K raises linear AUC
#:       by at most GK_BINARY_AUC and lowers utility by at most GK_UTILITY_DROP.
#:   H3, frozen transfer on the held-out encoder's donors: the 95% interval of the legal
#:       nDCG@10 macro-average change under K = GENEROUS_K includes zero.
#: The GB and MLP probes are reported, not tested.
GK_META_MARGIN = 0.03
GK_BINARY_AUC = 0.02
GK_UTILITY_DROP = 0.02
GK_CONDITIONS = ("raw", "pca_top1", "metadata_leace", "kmeans_auto", "kmeans_count",
                 "kmeans_generous")
GK_NONLINEAR_CONDITIONS = ("metadata_leace", "kmeans_auto", "kmeans_generous")


def run_generous_k(corpus: str, encoder: str, k: int = GENEROUS_K, seeds=GK_SEEDS,
                   nonlinear=("gb", "mlp"), device=None, overwrite: bool = False,
                   verbose: bool = True) -> dict:
    """k-means LEACE at the pre-specified K = `k` on one corpus x encoder cell, beside the
    silhouette K, K = C (a count-only reference), Metadata-LEACE and PCA-1, on the
    splits of the ten-encoder study. No SAE is trained.

    Where `revised.run_encoder_cell` has a cell for this corpus and encoder, every seed's
    split must match it and its Raw, PCA-1 and Metadata-LEACE linear AUCs must be
    reproduced before anything is written, so that the cell's SAE-LEACE can be set beside
    these arms. The GB and MLP probes run on `GK_NONLINEAR_CONDITIONS` only.
    """
    with artifacts.experiment(FOLDER):
        name = f"generous_k_{corpus}_{encoder}.json"
        path = artifacts.artifact_path(name)
        if os.path.exists(path) and not overwrite:
            print(f"[cached] {path}")
            return artifacts.load_result(path)
        data = R.load(corpus, encoder, device=device, verbose=verbose)
        X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
        classes = np.unique(y)
        cell_path = R.raw_path("encoders", f"encoder_{corpus}_{encoder}.json")
        cell = artifacts.load_result(cell_path) if os.path.exists(cell_path) else None
        ref = {p["seed"]: p for p in cell["per_seed"]} if cell else {}
        per_seed, checks = [], []
        for s in seeds:
            t0 = time.time()
            fits, diag = [], []
            for sp in R._splits(data, s):
                tr = np.asarray(sp.train_docs)
                K, sil = P.choose_pseudo_groups(X[tr], seed=sp.seed)
                er = {"raw": lambda Z: np.asarray(Z, np.float32),
                      "pca_top1": _pca_erasers(X[tr], (1,))[1],
                      "metadata_leace": P._fit_leace(X[tr], P._one_hot(y[tr], classes),
                                                     standardise=False),
                      "kmeans_auto": _kmeans_eraser(X[tr], K, sp.seed)[0],
                      "kmeans_count": _kmeans_eraser(X[tr], len(classes), sp.seed)[0],
                      "kmeans_generous": _kmeans_eraser(X[tr], k, sp.seed)[0]}
                fits.append(_Fit(sp, er))
                diag.append({"k_auto": int(K), "silhouette": {str(kk): v for kk, v in sil.items()}})
            res = R.evaluate_fits(fits, X, y, conditions=GK_CONDITIONS, topics=data.get("topics"),
                                  nonlinear=nonlinear, nonlinear_conditions=GK_NONLINEAR_CONDITIONS)
            res.pop("hits", None)
            if s in ref:
                q = ref[s]
                if res["split"] != q["split"]:
                    raise RuntimeError(f"{corpus}/{encoder} seed {s}: the split differs from the "
                                       "ten-encoder cell's, so nothing was written")
                shared = [c for c in ("raw", "pca_top1", "metadata_leace") if c in q["metrics"]]
                diff = max(abs(res["metrics"][c]["linear_auc"] - q["metrics"][c]["linear_auc"])
                           for c in shared)
                if diff > 0.005:
                    raise RuntimeError(f"{corpus}/{encoder} seed {s}: {shared} do not reproduce the "
                                       f"ten-encoder cell (largest linear AUC difference {diff:.4f}), "
                                       "so nothing was written")
                checks.append({"seed": int(s), "reproduced": shared,
                               "max_linear_auc_diff": float(diff)})
            res["diagnostics"] = diag
            res["seconds"] = round(time.time() - t0, 1)
            per_seed.append(res)
            if verbose:
                m = res["metrics"]
                print(f"[generous-k] {corpus}/{encoder} seed {s} ({res['seconds']}s, silhouette K "
                      f"{[d['k_auto'] for d in diag]}): " + "  ".join(
                          f"{c} {m[c]['linear_auc']:.3f}" for c in
                          ("metadata_leace", "kmeans_auto", "kmeans_count", "kmeans_generous")))
        out = {"experiment": "generous_k", "corpus": corpus, "label": data["label"],
               "encoder": encoder,
               "role": "development" if encoder == GK_DEV_ENCODER else "held-out",
               "k_generous": int(k), "n_classes": int(len(classes)),
               "seeds": [int(s) for s in seeds], "conditions": list(GK_CONDITIONS),
               "nonlinear_conditions": list(GK_NONLINEAR_CONDITIONS) if nonlinear else [],
               "paired": data["paired"], "per_seed": per_seed, "agg": R.aggregate(per_seed),
               "reference_cell": os.path.basename(cell_path) if cell else None,
               "checks": checks,
               "criteria": {"meta_margin": GK_META_MARGIN, "binary_auc": GK_BINARY_AUC,
                            "utility_drop": GK_UTILITY_DROP},
               "corpus_stats": R._corpus_stats(data),
               "config": R._config(data, {}, nonlinear=list(nonlinear), sae=None)}
        artifacts.save_result(out, name)
        return out


def _sae_reference(corpus: str, encoder: str, like=None) -> tuple:
    """SAE-LEACE for one corpus x encoder cell on the ten-encoder study's splits, as
    ``(result, source)``. That study's cell where it exists; otherwise (CAP, which the
    study does not cover) the in-distribution run for the same corpus and encoder -- used
    only if its seeds and every seed's split are those of `like`, a result on the same
    cell, and its Metadata-LEACE linear AUC reproduces there. Source is "encoders", "id"
    or None."""
    import revised_exhibits as X

    cell = X.raw("encoders", f"encoder_{corpus}_{encoder}.json")
    if cell is not None:
        return cell, "encoders"
    idr = X.raw("id", f"id_{corpus}_{encoder}.json")
    if idr is None or like is None or idr["seeds"] != like["seeds"]:
        return None, None
    for p_, q in zip(idr["per_seed"], like["per_seed"]):
        if p_["split"] != q["split"] or abs(p_["metrics"]["metadata_leace"]["linear_auc"]
                                            - q["metrics"]["metadata_leace"]["linear_auc"]) > 0.005:
            return None, None
    return idr, "id"


def generous_k_verdict(r: dict) -> tuple:
    """``(hypothesis, passed)`` for one `run_generous_k` cell under the pre-specified
    criteria: H1 on a multi-class nuisance, H2 on a binary one."""
    def mean(c, key):
        return (r["agg"].get(c, {}).get(key) or {}).get("mean")
    util = next((m for m in ("topic_purity", "recall@10") if m in r["agg"]["raw"]), None)
    gen, auto = mean("kmeans_generous", "linear_auc"), mean("kmeans_auto", "linear_auc")
    keeps = (util is None
             or mean("kmeans_generous", util) >= mean("kmeans_auto", util) - GK_UTILITY_DROP)
    if r["n_classes"] > 2:
        return "H1", bool(gen <= auto and gen <= mean("metadata_leace", "linear_auc") + GK_META_MARGIN
                          and keeps)
    return "H2", bool(gen <= auto + GK_BINARY_AUC and keeps)


def table_generous_k(corpora=GK_CORPORA, stem: str = "table_generous_k", where: str = "extra",
                     wide: bool = False, n_boot: int = 10000, verdicts: bool = True) -> list:
    """`verdicts=False` drops the per-row "Met" column; Panel F still counts the cells
    meeting each criterion.

    The confirmatory generous-K test: each corpus's cells, the pre-specified hypotheses'
    outcomes on the held-out encoders, the nonlinear probes, and frozen transfer."""
    import revised_exhibits as X

    encoders = (GK_DEV_ENCODER,) + GK_HELD_OUT
    res = {}
    with artifacts.experiment(FOLDER):
        for c in corpora:
            for e in encoders:
                p = R.raw_path(FOLDER, f"generous_k_{c}_{e}.json")
                if os.path.exists(p):
                    res[(c, e)] = artifacts.load_result(p)
    if not res:
        raise FileNotFoundError("no generous_k results; run run_generous_k() first")
    k = next(iter(res.values()))["k_generous"]
    gk = f"K = {k}"

    def mean(r, c, key):
        return (r["agg"].get(c, {}).get(key) or {}).get("mean") if r else None

    panels, outcomes, nonlinear, letters = [], [], [], iter("ABCDEFGHIJ")
    for corpus in corpora:
        cells = {e: res.get((corpus, e)) for e in encoders}
        first = next((r for r in cells.values() if r), None)
        if first is None:
            continue
        multi = first["n_classes"] > 2
        util = next((m for m in ("topic_purity", "recall@10") if m in first["agg"]["raw"]), None)
        header = (["Encoder", "Silhouette\nK chosen", "Silhouette\nK", gk, "K = C", "Metadata-\nLEACE",
                   "SAE-\nLEACE"]
                  + (["Silhouette\nK", gk, "Metadata-\nLEACE", "SAE-\nLEACE"] if util else [])
                  + (["Met"] if verdicts else []))
        groups = ([(None, 2), ("Linear probe AUC", 5)]
                  + ([("Topic purity" if util == "topic_purity" else "R@10", 4)] if util else [])
                  + ([(None, 1)] if verdicts else []))
        rows, passed, tested = [], 0, 0
        for e in encoders:
            r = cells[e]
            name = X.ENCODER.get(e, e) + (" (dev.)" if e == GK_DEV_ENCODER else "")
            if r is None:
                rows.append([name] + ["NR"] * (len(header) - 1))
                continue
            cell = _sae_reference(corpus, e, r)[0]
            ks = sorted({d["k_auto"] for p in r["per_seed"] for d in p["diagnostics"]})
            row = [name, str(ks[0]) if len(ks) == 1 else f"{ks[0]}–{ks[-1]}"]
            row += [X.num(mean(r, c, "linear_auc"))
                    for c in ("kmeans_auto", "kmeans_generous", "kmeans_count", "metadata_leace")]
            row.append(X.num(mean(cell, "sae_leace", "linear_auc")))
            if util:
                row += [X.num(mean(r, c, util)) for c in ("kmeans_auto", "kmeans_generous",
                                                          "metadata_leace")]
                row.append(X.num(mean(cell, "sae_leace", util)))
            if e == GK_DEV_ENCODER:
                mark = "—"
            else:
                ok = generous_k_verdict(r)[1]
                passed, tested = passed + ok, tested + 1
                mark = "pass" if ok else "fail"
            rows.append(row + ([mark] if verdicts else []))
        panels.append((f"Panel {next(letters)} — {X.CORPUS[corpus]} (C = {first['n_classes']}; "
                       f"{'multi-class' if multi else 'binary'} criterion)", header, rows, groups))
        if tested:
            outcomes.append([X.CORPUS[corpus], "multi-class" if multi else "binary",
                             f"{passed}/{tested}"])
        held = [cells[e] for e in GK_HELD_OUT
                if cells[e] and "gb_auc" in cells[e]["agg"].get("kmeans_generous", {})]
        if held:
            nonlinear.append([X.CORPUS[corpus], str(len(held))]
                             + [X.num(float(np.mean([mean(r, c, key) for r in held])))
                                for key in ("gb_auc", "mlp_auc")
                                for c in ("kmeans_auto", "kmeans_generous", "metadata_leace")])

    transfer, passed3, tested3, ood_res = [], 0, 0, {}
    with artifacts.experiment(FOLDER):
        for enc, donor in R.OOD_MATRIX:
            p = R.raw_path(FOLDER, f"ood_kmeans_{enc}_{donor}.json")
            if os.path.exists(p):
                o = artifacts.load_result(p)
                if "kmeans_generous" in o["transforms"]:
                    ood_res[f"{enc}/{donor}"] = o
    donor_k = sorted({p["fit"]["k_auto"] for o in ood_res.values() for p in o["per_seed"]
                      if p["fit"].get("k_auto")})
    if ood_res:
        s = ood.summarize(ood_res, n_boot=n_boot, save=False)
        for key in s["order"]:
            v = s["settings"][key]
            if transfer:
                transfer.append(X.SEP)
            for c in ("kmeans_auto", "kmeans_count", "kmeans_generous"):
                ch = v["changes"][c]
                lo, hi = ch["legal_macro"]["ci"]
                verdict = "—"
                if c == "kmeans_generous" and v["encoder"] != GK_DEV_ENCODER:
                    ok = lo <= 0 <= hi
                    passed3, tested3 = passed3 + ok, tested3 + 1
                    verdict = "pass" if ok else "fail"
                transfer.append([X.ood_title(key) if c == "kmeans_auto" else "", s["labels"].get(c, c),
                                 X.signed(ch["legal_macro"]["delta"]), X.ci(ch["legal_macro"]["ci"]),
                                 X.signed(ch[v["sts"][0]]["delta"]) if v["sts"] else "NR"]
                                + ([verdict] if verdicts else []))
        if tested3:
            outcomes.append(["Frozen transfer (held-out encoder's donors)", "transfer",
                             f"{passed3}/{tested3}"])
    if outcomes:
        panels.append((f"Panel {next(letters)} — Pre-specified criteria on the held-out encoders",
                       ["Corpus / evidence", "Criterion", "Cells meeting it"], outcomes))
    if nonlinear:
        panels.append((f"Panel {next(letters)} — Nonlinear probes, mean over the held-out encoders "
                       "(descriptive)",
                       ["Corpus", "Encoders", "Silhouette\nK", gk, "Metadata-\nLEACE", "Silhouette\nK",
                        gk, "Metadata-\nLEACE"], nonlinear,
                       [(None, 2), ("GB probe AUC", 3), ("MLP probe AUC", 3)]))
    if transfer:
        panels.append((f"Panel {next(letters)} — Frozen transfer of k-means LEACE",
                       ["Donor setting", "Map", "Mean legal\nΔnDCG@10", "95% CI", "STS\nΔSpearman"]
                       + (["Met"] if verdicts else []), transfer))
    note = (f"k-means LEACE at K = {k}, fixed on the development encoder (dev., E5-small; Table "
            "D10) before any held-out encoder was run, beside the silhouette K, K = C (a "
            "count-only reference) and Metadata-LEACE; every map is fitted on the ten-encoder "
            "study's training splits (seeds 0–2) and audited on its held-out documents, and no "
            "SAE is trained. SAE-LEACE is read from the ten-encoder study's cell on the same "
            "splits, whose Raw, PCA-1 and Metadata-LEACE linear AUCs each cell reproduced before "
            "it was written, CAP included. Utility is R@10 on the paired corpora and "
            "topic purity on CAP; Bills–Tweets has none. Three criteria were fixed with K. On a "
            f"multi-class nuisance: at K = {k}, linear AUC at or below the silhouette K's and at "
            f"most {GK_META_MARGIN} above Metadata-LEACE's, and utility at most {GK_UTILITY_DROP} "
            f"below the silhouette K's. On a binary nuisance: against the silhouette K, linear AUC "
            f"at most {GK_BINARY_AUC} higher and utility at most {GK_UTILITY_DROP} lower. Under "
            f"frozen transfer: on the held-out encoder's donors, the 95% interval of the legal "
            f"macro-average change under K = {k} includes zero (paired bootstrap, {n_boot:,} "
            f"resamples, as in Table C3)."
            + (" 'Met': the cell satisfies the criterion for its corpus." if verdicts else "")
            + " The "
            "development rows are not tested; the nonlinear probes are descriptive."
            + (f" The silhouette chose K = {'–'.join(map(str, donor_k))} in every donor fit."
               if donor_k else "") + " NR: not run.")
    with artifacts.experiment(None):
        return X.write_table(
            stem, where, "Confirmatory test of a generous K on held-out encoders (means over "
            + X.nseeds(res) + ").", panels, note=note, wide=wide,
            tabcolsep=2.5 if wide else None)


# --------------------------------------------------------------------------- #
# Where the nuisance sits in the geometry
# --------------------------------------------------------------------------- #
#: Ranks at which the capture of the nuisance subspace is reported.
GEOMETRY_RANKS = (1, 2, 3, 4, 8, 16, 32)


def run_geometry(corpus: str, encoder: str, seeds=GK_SEEDS, device=None,
                 overwrite: bool = False, verbose: bool = True) -> dict:
    """Where the nuisance sits in the geometry of one corpus x encoder cell.

    On the training split of every fit the ten-encoder study used: how much of the
    subspace Metadata-LEACE removes -- the span of the label cross-covariance, rank
    C - 1 -- lies in the leading principal directions, how much of the total variance
    that subspace carries, and how closely the k-means groupings match the classes.

    This is descriptive and uses the labels, so it explains where label-free erasers
    can find the nuisance at all; it fits no eraser, trains no SAE and runs no probe.
    """
    from sklearn.metrics import adjusted_rand_score

    with artifacts.experiment(FOLDER):
        name = f"geometry_{corpus}_{encoder}.json"
        path = artifacts.artifact_path(name)
        if os.path.exists(path) and not overwrite:
            print(f"[cached] {path}")
            return artifacts.load_result(path)
        data = R.load(corpus, encoder, device=device, verbose=verbose)
        X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
        classes = np.unique(y)
        per_seed = []
        for s in seeds:
            t0, folds = time.time(), []
            for sp in R._splits(data, s):
                tr = np.asarray(sp.train_docs)
                Xc = np.asarray(X[tr], np.float64)
                Xc = Xc - Xc.mean(axis=0)
                lam = np.linalg.svd(Xc, compute_uv=False) ** 2
                Vt = np.linalg.svd(Xc, full_matrices=False)[2]
                Z = P._one_hot(y[tr], classes).astype(np.float64)
                B = _orth(Xc.T @ (Z - Z.mean(axis=0)))      # what Metadata-LEACE removes
                capture = np.cumsum(((Vt @ B) ** 2).sum(axis=1)) / B.shape[1]
                K = int(P.choose_pseudo_groups(X[tr], seed=sp.seed)[0])
                ari = {tag: float(adjusted_rand_score(y[tr], _kmeans_eraser(X[tr], k, sp.seed)[1]))
                       for tag, k in (("auto", K), ("count", len(classes)),
                                      ("generous", GENEROUS_K))}
                folds.append({
                    "k_auto": K, "pc1_var_share": float(lam[0] / lam.sum()),
                    "nuisance_var_share": float(((Xc @ B) ** 2).sum() / (Xc ** 2).sum()),
                    "capture": {str(r): float(capture[min(int(r), len(capture)) - 1])
                                for r in GEOMETRY_RANKS},
                    "rank_90": int(np.searchsorted(capture, 0.90) + 1),
                    "rank_99": int(np.searchsorted(capture, 0.99) + 1), "ari": ari})
            per_seed.append({"seed": int(s), "folds": folds,
                             "seconds": round(time.time() - t0, 1)})
        flat = [f for p_ in per_seed for f in p_["folds"]]
        mean = lambda key: float(np.mean([f[key] for f in flat]))              # noqa: E731
        agg = {"pc1_var_share": mean("pc1_var_share"),
               "nuisance_var_share": mean("nuisance_var_share"),
               "rank_90": mean("rank_90"), "rank_99": mean("rank_99"),
               "capture": {str(r): float(np.mean([f["capture"][str(r)] for f in flat]))
                           for r in GEOMETRY_RANKS},
               "ari": {tag: float(np.mean([f["ari"][tag] for f in flat]))
                       for tag in ("auto", "count", "generous")},
               "k_auto": sorted({f["k_auto"] for f in flat})}
        out = {"experiment": "geometry", "corpus": corpus, "label": data["label"],
               "encoder": encoder, "n_classes": int(len(classes)), "class_rank": int(len(classes) - 1),
               "seeds": [int(s) for s in seeds], "k_generous": GENEROUS_K,
               "ranks": list(GEOMETRY_RANKS), "per_seed": per_seed, "agg": agg,
               "corpus_stats": R._corpus_stats(data), "config": R._config(data, {}, sae=None)}
        if verbose:
            print(f"[geometry] {corpus}/{encoder}: PC1 share {agg['pc1_var_share']:.3f} | "
                  f"nuisance in PC1 {agg['capture']['1']:.2f}, in top 8 {agg['capture']['8']:.2f} | "
                  f"rank for 90% {agg['rank_90']:.1f} | variance share {agg['nuisance_var_share']:.4f} | "
                  f"ARI silhouette {agg['ari']['auto']:.2f}, K={GENEROUS_K} {agg['ari']['generous']:.2f}")
        artifacts.save_result(out, name)
        return out


def _geometry_cells(corpora=GK_CORPORA, encoders=None) -> dict:
    """``{(corpus, encoder): (geometry, generous-K result, encoder-study cell)}``."""
    encoders = tuple(encoders or (GK_DEV_ENCODER,) + GK_HELD_OUT)
    import revised_exhibits as X

    out = {}
    with artifacts.experiment(FOLDER):
        for c in corpora:
            for e in encoders:
                p = R.raw_path(FOLDER, f"geometry_{c}_{e}.json")
                if not os.path.exists(p):
                    continue
                gk = R.raw_path(FOLDER, f"generous_k_{c}_{e}.json")
                gkr = artifacts.load_result(gk) if os.path.exists(gk) else None
                out[(c, e)] = (artifacts.load_result(p), gkr, _sae_reference(c, e, gkr)[0])
    return out


#: (diagnostic, eraser) pairs Table D12 and Figure D8 relate: does the diagnostic predict
#: how far that eraser stays above Metadata-LEACE?
GEOMETRY_TESTS = (("capture_1", "pca_top1", "share of the nuisance subspace in PC1", "PCA-1"),
                  ("rank_90", "pca_top1", "rank that captures 90% of it", "PCA-1"),
                  ("ari_auto", "kmeans_auto", "ARI of the silhouette grouping", "k-means, silhouette K"),
                  ("ari_generous", "kmeans_generous", "ARI of the K = 8 grouping", "k-means, K = 8"),
                  ("rank_90", "sae_leace", "rank that captures 90% of it", "SAE-LEACE"))


def _geometry_points(cells: dict) -> dict:
    """``{(diagnostic, eraser): [(x, gap to Metadata-LEACE, corpus)]}`` over the cells."""
    pts = {}
    for (corpus, _enc), (g, gk, cell) in cells.items():
        if gk is None:
            continue
        meta = (gk["agg"].get("metadata_leace", {}).get("linear_auc") or {}).get("mean")
        diag = {"capture_1": g["agg"]["capture"]["1"], "rank_90": g["agg"]["rank_90"],
                "ari_auto": g["agg"]["ari"]["auto"], "ari_generous": g["agg"]["ari"]["generous"]}
        for key, arm, _dl, _al in GEOMETRY_TESTS:
            src = cell if arm == "sae_leace" else gk
            auc = ((src or {}).get("agg", {}).get(arm, {}).get("linear_auc") or {}).get("mean")
            if None not in (auc, meta):
                pts.setdefault((key, arm), []).append((diag[key], auc - meta, corpus))
    return pts


def table_geometry(corpora=GK_CORPORA, encoders=None, stem: str = "table_geometry",
                   where: str = "extra", wide: bool = False) -> list:
    """Where each cell's nuisance sits in the geometry, beside how far each label-free
    eraser stays above Metadata-LEACE there (Table D12)."""
    import revised_exhibits as X
    from scipy import stats

    cells = _geometry_cells(corpora, encoders)
    if not cells:
        raise FileNotFoundError("no geometry results; run run_geometry() first")
    encoders = tuple(encoders or (GK_DEV_ENCODER,) + GK_HELD_OUT)
    rows = []
    for corpus in corpora:
        for e in encoders:
            if (corpus, e) not in cells:
                continue
            g, gk, cell = cells[(corpus, e)]
            meta = ((gk or {}).get("agg", {}).get("metadata_leace", {}).get("linear_auc")
                    or {}).get("mean")
            gap = lambda src, arm: (                                            # noqa: E731
                "NR" if src is None
                or (src["agg"].get(arm, {}).get("linear_auc") or {}).get("mean") is None
                or meta is None else
                X.signed(src["agg"][arm]["linear_auc"]["mean"] - meta))
            rows.append([X.CORPUS[corpus] if e == encoders[0] else "", X.ENCODER.get(e, e),
                         str(g["n_classes"]),
                         X.num(g["agg"]["capture"]["1"], 2), X.num(g["agg"]["capture"]["8"], 2),
                         f"{g['agg']['rank_90']:.0f}", X.num(g["agg"]["nuisance_var_share"], 4),
                         X.num(g["agg"]["ari"]["auto"], 2), X.num(g["agg"]["ari"]["generous"], 2),
                         gap(gk, "pca_top1"), gap(gk, "kmeans_auto"), gap(gk, "kmeans_generous"),
                         gap(cell, "sae_leace")])
        if rows and rows[-1] != X.SEP and corpus != corpora[-1]:
            rows.append(X.SEP)
    header = ["Corpus", "Encoder", "C", "in PC1", "in top 8", "rank for\n90%", "variance\nshare",
              "Silh.\nK", "K = 8", "PCA-1", "Silh.\nK", "K = 8", "SAE-\nLEACE"]
    groups = [(None, 3), ("Nuisance subspace", 4), ("Grouping ARI", 2),
              ("Linear AUC above Metadata-LEACE", 4)]
    pts = _geometry_points(cells)
    corr = []
    for key, arm, dlabel, alabel in GEOMETRY_TESTS:
        v = pts.get((key, arm), [])
        if len(v) > 2:
            rho = stats.spearmanr([x for x, _, _ in v], [y for _, y, _ in v])
            corr.append([dlabel, alabel, str(len(v)), X.signed(rho.statistic, 2),
                         f"{rho.pvalue:.3f}"])
    panels = [("Panel A — Each cell's geometry and what the erasers achieve there", header, rows,
               groups)]
    if corr:
        panels.append(("Panel B — Rank correlation across the cells",
                       ["Diagnostic", "Eraser", "Cells", "Spearman ρ", "p"], corr))
    with artifacts.experiment(None):
        return X.write_table(
            stem, where, "Where the nuisance sits in each cell's geometry (means over "
            + X.nseeds({k: v[0] for k, v in cells.items()}) + ").", panels, wide=wide,
            tabcolsep=2.5 if wide else None,
            note="Descriptive and label-using: nothing here is fitted or erased. The nuisance "
                 "subspace is the rank C − 1 span of the label cross-covariance on each fit's "
                 "training documents -- what Metadata-LEACE removes. 'In PC1' and 'in top 8' are "
                 "the share of that subspace lying in the leading principal directions, 'rank "
                 "for 90%' the fewest principal directions holding 90% of it, and 'variance "
                 "share' the fraction of total training variance along it (the embeddings' own "
                 "PC1 variance share is Figure D6's x-axis). Grouping ARI compares "
                 "the k-means grouping with the true classes after fitting. The last four "
                 "columns are each eraser's held-out linear AUC minus Metadata-LEACE's in the "
                 "same cell (positive: more leakage than the supervised eraser); k-means and "
                 "PCA-1 come from Table D11's run, SAE-LEACE from the ten-encoder study's cell "
                 "on the same splits (CAP included). Panel B correlates each diagnostic with that gap across "
                 "cells; the correlations are descriptive, and the cells share corpora and "
                 "encoders, so they are not independent.")


def figure_geometry(corpora=GK_CORPORA, encoders=None, stem: str = "figure_geometry",
                    where: str = "extra") -> list:
    """Does the geometry predict where a label-free eraser works? Each panel: one
    diagnostic against how far its eraser stays above Metadata-LEACE (Figure D8)."""
    import revised_exhibits as X
    from scipy import stats

    cells = _geometry_cells(corpora, encoders)
    if not cells:
        raise FileNotFoundError("no geometry results; run run_geometry() first")
    pts = _geometry_points(cells)
    shown = [(k, a, dl, al) for k, a, dl, al in GEOMETRY_TESTS if len(pts.get((k, a), [])) > 2]
    if not shown:
        raise FileNotFoundError("no geometry/erasure pairs to plot")
    plt = X._plt()
    from matplotlib.lines import Line2D
    ncol = min(3, len(shown))                      # a page is 3 panels wide at this size
    nrow = -(-len(shown) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.45 * ncol, 2.75 * nrow), squeeze=False)
    seen = []
    for ax, (key, arm, dlabel, alabel) in zip(axes.ravel(), shown):
        v = pts[(key, arm)]
        for corpus in dict.fromkeys(c for _, _, c in v):
            xs = [x for x, _, c in v if c == corpus]
            ys = [y for _, y, c in v if c == corpus]
            ax.plot(xs, ys, "o", color=X.CORPUS_COLOR[corpus], ms=5, mec="black", mew=0.4,
                    ls="none", zorder=3)
            if corpus not in seen:
                seen.append(corpus)
        rho = stats.spearmanr([x for x, _, _ in v], [y for _, y, _ in v])
        ax.axhline(0, color=X.MUTED, lw=0.7, ls=":")
        ax.set_title(f"{alabel}\nρ = {rho.statistic:+.2f} (n = {len(v)})", fontsize=8.5)
        ax.set_xlabel(dlabel, fontsize=7.5)
        if key == "rank_90":
            ax.set_xscale("log", base=2)
        ax.grid(True)
    for ax in axes.ravel()[len(shown):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Linear AUC − Metadata-LEACE", fontsize=8)
    fig.legend([Line2D([0], [0], color=X.CORPUS_COLOR[c], marker="o", ls="none", ms=5,
                       mec="black", mew=0.4) for c in seen], [X.CORPUS[c] for c in seen],
               loc="lower center", ncol=len(seen), bbox_to_anchor=(0.5, -0.05 / nrow),
               frameon=True)
    fig.tight_layout()
    with artifacts.experiment(None):
        return X.save_fig(fig, stem, where)


# --------------------------------------------------------------------------- #
# CAP: topic organisation across encoders, and source-topic entanglement
# --------------------------------------------------------------------------- #
#: The maps Table D14 compares on CAP topics, in display order; SAE-LEACE is added from
#: `_sae_reference`.
CAP_TOPIC_MAPS = (("raw", "Raw"), ("pca_top1", "PCA-1"), ("kmeans_auto", "k-means,\nsilhouette K"),
                  ("kmeans_generous", f"k-means,\nK = {GENEROUS_K}"),
                  ("metadata_leace", "Metadata-\nLEACE"))


def table_cap_topics(encoders=None, stem: str = "table_cap_topics", where: str = "extra",
                     wide: bool = False) -> list:
    """CAP topic organisation after each map, across the ten encoders (Table D14): topic
    purity, topic ARI and source AUC on the generous-K run's splits, with SAE-LEACE from
    the ten-encoder study's CAP cells, or the in-distribution run where those are missing."""
    import revised_exhibits as X

    encoders = tuple(encoders or (GK_DEV_ENCODER,) + GK_HELD_OUT)
    cells = {}
    with artifacts.experiment(FOLDER):
        for e in encoders:
            p = R.raw_path(FOLDER, f"generous_k_cap_{e}.json")
            if os.path.exists(p):
                cells[e] = artifacts.load_result(p)
    if not cells:
        raise FileNotFoundError("no generous_k CAP results; run run_generous_k('cap', ...) first")
    sae = {e: _sae_reference("cap", e, cells.get(e))[0] for e in encoders}
    keys = [c for c, _ in CAP_TOPIC_MAPS] + ["sae_leace"]
    header = ["Encoder"] + [lab for _, lab in CAP_TOPIC_MAPS] + ["SAE-\nLEACE"]

    def mean(r, c, m):
        return ((((r or {}).get("agg") or {}).get(c) or {}).get(m) or {}).get("mean")

    panels = []
    for metric, title, higher in (("topic_purity", "Topic purity (higher is better)", True),
                                  ("topic_ari", "Topic ARI (higher is better)", True),
                                  ("linear_auc", "Source linear AUC (lower is better; 0.5 = chance)",
                                   False)):
        rows, beats = [], {c: [0, 0] for c in keys[1:]}
        for e in encoders:
            vals = [mean(cells.get(e), c, metric) for c in keys[:-1]] + [mean(sae[e], "sae_leace", metric)]
            rows.append([X.ENCODER.get(e, e) + (" (dev.)" if e == GK_DEV_ENCODER else "")]
                        + [X.num(v) for v in vals])
            for c, v in zip(keys[1:], vals[1:]):
                if None not in (v, vals[0]):
                    beats[c][0] += (v > vals[0]) if higher else (v < vals[0])
                    beats[c][1] += 1
        if higher:
            rows += [X.SEP, ["Encoders above Raw", "—"]
                     + [f"{beats[c][0]}/{beats[c][1]}" if beats[c][1] else "NR" for c in keys[1:]]]
        panels.append((f"Panel {'ABC'[len(panels)]} — {title}", header, rows))
    with artifacts.experiment(None):
        return X.write_table(
            stem, where, "CAP topic organisation after each map, across the ten encoders (means "
            "over " + X.nseeds(cells) + ").", panels, wide=wide,
            note="Held-out CAP documents on the ten-encoder study's splits. Topic purity "
                 "(cluster-size weighted) and ARI come from k-means (K_topic = 21, n_init = 20) on "
                 "the transformed held-out documents; the topic labels never enter a fit. Raw, "
                 "PCA-1, both k-means maps and Metadata-LEACE come from the run of Table D11; "
                 "SAE-LEACE from the ten-encoder study's CAP cells on the same splits. "
                 "'Encoders above Raw' counts the encoders in which a map's value "
                 "exceeds Raw's. dev.: the development encoder of Table D11. NR: not run.")


#: Experiment 3's split of CAP's topics, fixed before the run: a topic is source-balanced
#: when every source supplies at least this share of its documents in the whole corpus
#: (half of an even three-way split), and source-skewed otherwise. On CAP it falls in a gap
#: of the per-topic shares (0.12 -> 0.17), giving 12 balanced and 9 skewed topics.
CAP_BALANCE_SHARE = 1 / 6
#: The maps Experiment 3 scores, in display order.
ENTANGLE_MAPS = (("raw", "Raw"), ("pca_top1", "PCA-1"), ("kmeans_auto", "k-means LEACE, silhouette K"),
                 ("kmeans_generous", f"k-means LEACE, K = {GENEROUS_K}"),
                 ("metadata_leace", "Metadata-LEACE"), ("sae_leace", "SAE-LEACE"))


def run_cap_entanglement(encoder: str, seeds=R.OOD_SEEDS, min_share: float = CAP_BALANCE_SHARE,
                         device=None, overwrite: bool = False, verbose: bool = True) -> dict:
    """Does erasure lose CAP topic structure where the source carries topic information?

    CAP's 21 topics are split by how evenly the three sources cover them in the whole corpus:
    source-balanced (every source supplies at least `min_share` of the topic's documents) and
    source-skewed (the rest, where knowing a document's source says something about its
    topic). For each seed's held-out documents the maps `revised.run_id` saved for this
    encoder -- PCA-1, Metadata-LEACE, SAE-LEACE -- are applied unchanged, beside Raw and the
    two k-means LEACE maps refitted on the same training split (silhouette K, K =
    GENEROUS_K). Each topic group is then clustered on its own, with k equal to its number of
    topics and the audit's k-means settings, and scored by topic purity and ARI. The "all"
    group repeats the main protocol, and its agreement with the saved in-distribution run is
    recorded as a check. Labels only score; nothing but the two k-means maps is fitted.
    """
    import audit

    with artifacts.experiment(FOLDER):
        name = f"cap_entanglement_{encoder}.json"
        path = artifacts.artifact_path(name)
        if os.path.exists(path) and not overwrite:
            print(f"[cached] {path}")
            return artifacts.load_result(path)
        data = R.load("cap", encoder, device=device, verbose=verbose)
        X = np.asarray(data["emb"], np.float32)
        src, top = np.asarray(data["source"]), np.asarray(data["topics"])
        sources, all_topics = np.unique(src), np.unique(top)
        share = {str(t): min(float(np.mean(src[top == t] == s)) for s in sources) for t in all_topics}
        groups = {"balanced": [t for t in all_topics if share[str(t)] >= min_share],
                  "skewed": [t for t in all_topics if share[str(t)] < min_share],
                  "all": list(all_topics)}
        id_path = R.raw_path("id", f"id_cap_{encoder}.json")
        ref = ({p["seed"]: p for p in artifacts.load_result(id_path)["per_seed"]}
               if os.path.exists(id_path) else {})
        per_seed, checks = [], []
        for s in seeds:
            sp = R._splits(data, s)[0]
            tr, te = np.asarray(sp.train_docs), np.asarray(sp.test_docs)
            maps, _, tr_saved = R.load_maps(encoder, "cap", s)
            if not np.array_equal(np.sort(np.asarray(tr_saved)), np.sort(tr)):
                raise ValueError(f"cap/{encoder} seed {s}: the saved maps were fitted on other "
                                 "training documents than this split")
            erasers = {"raw": lambda Z: np.asarray(Z, np.float32)}
            erasers.update({c: (lambda Z, _A=A_, _c=c_: (np.asarray(Z, np.float64) @ _A + _c)
                                .astype(np.float32)) for c, (A_, c_) in maps.items()})
            K, _ = P.choose_pseudo_groups(X[tr], seed=sp.seed)
            erasers["kmeans_auto"] = _kmeans_eraser(X[tr], K, sp.seed)[0]
            erasers["kmeans_generous"] = _kmeans_eraser(X[tr], GENEROUS_K, sp.seed)[0]
            metrics = {}
            for c, fn in erasers.items():
                Z = fn(X[te])
                metrics[c] = {}
                for g, keep in groups.items():
                    idx = np.isin(top[te], keep)
                    metrics[c][g] = audit.topic_scores(Z[idx], top[te][idx], seed=int(sp.seed),
                                                       k=int(len(np.unique(top[te][idx]))))
            diff = [abs(metrics[c]["all"]["topic_purity"] - ref[s]["metrics"][c]["topic_purity"])
                    for c in ("raw", "pca_top1", "metadata_leace", "sae_leace")
                    if s in ref and c in metrics and c in ref[s]["metrics"]]
            checks.append({"seed": int(s), "max_all_topics_purity_diff_vs_id": max(diff) if diff else None})
            per_seed.append({"seed": int(s), "k_auto": int(K), "metrics": metrics,
                             "n_test": {g: int(np.isin(top[te], keep).sum()) for g, keep in groups.items()}})
            if verbose:
                print(f"[cap-entangle] {encoder} seed {s} (silhouette K={K}; all-topics purity within "
                      f"{checks[-1]['max_all_topics_purity_diff_vs_id']} of the ID run): " + "  ".join(
                          f"{c} {metrics[c]['balanced']['topic_purity']:.3f}/"
                          f"{metrics[c]['skewed']['topic_purity']:.3f}" for c in metrics))
        agg = {c: {g: {m: R._stat([p["metrics"][c][g][m] for p in per_seed])
                       for m in ("topic_purity", "topic_ari")} for g in groups}
               for c in per_seed[0]["metrics"]}
        out = {"experiment": "cap_entanglement", "corpus": "cap", "encoder": encoder,
               "seeds": [int(s) for s in seeds], "min_share": float(min_share),
               "topic_min_source_share": share,
               "groups": {g: [str(t) for t in keep] for g, keep in groups.items()},
               "per_seed": per_seed, "agg": agg, "checks": checks,
               "config": R._config(data, {}, sae=None)}
        artifacts.save_result(out, name)
        return out


def table_cap_entanglement(encoders=("e5-small", "mini"), stem: str = "table_cap_entanglement",
                           where: str = "extra", wide: bool = False) -> list:
    """Topic organisation on CAP's source-balanced and source-skewed topics (Table D15)."""
    import revised_exhibits as X

    res = {}
    with artifacts.experiment(FOLDER):
        for e in encoders:
            p = R.raw_path(FOLDER, f"cap_entanglement_{e}.json")
            if os.path.exists(p):
                res[e] = artifacts.load_result(p)
    if not res:
        raise FileNotFoundError("no cap_entanglement results; run run_cap_entanglement() first")
    first = next(iter(res.values()))
    nb, ns = len(first["groups"]["balanced"]), len(first["groups"]["skewed"])
    header = ["Map", "Purity", "ARI", "Purity", "ARI", "Purity", "ARI"]
    groups = [(None, 1), (f"Source-balanced topics ({nb})", 2), (f"Source-skewed topics ({ns})", 2),
              (f"All {nb + ns} topics", 2)]
    panels = []
    for e, r in res.items():
        rows = [[lab] + [X.pm(((r["agg"].get(c) or {}).get(g) or {}).get(m))
                         for g in ("balanced", "skewed", "all") for m in ("topic_purity", "topic_ari")]
                for c, lab in ENTANGLE_MAPS]
        panels.append((f"Panel {'ABCD'[len(panels)]} — {X.ENCODER.get(e, e)}", header, rows, groups))
    diffs = [c["max_all_topics_purity_diff_vs_id"] for r in res.values() for c in r["checks"]
             if c["max_all_topics_purity_diff_vs_id"] is not None]
    with artifacts.experiment(None):
        return X.write_table(
            stem, where, "CAP topic organisation on source-balanced and source-skewed topics (mean "
            "± SD over " + X.nseeds(res) + ").", panels, wide=wide,
            note=f"A topic is source-balanced when each of the three sources supplies at least "
                 f"{first['min_share']:.3f} of its documents in the whole corpus (half an even "
                 f"split; fixed before the run), and source-skewed otherwise: {nb} and {ns} of the "
                 "21 topics. Each group of held-out documents is clustered on its own with k equal "
                 "to its number of topics (n_init = 20), so values compare maps within a column, "
                 "not groups across columns. PCA-1, Metadata-LEACE and SAE-LEACE are the frozen "
                 "maps saved by the in-distribution run for seeds 0–2; both k-means maps are "
                 "refitted on the same training splits. The 'all topics' columns repeat the main "
                 "protocol" + (f" and reproduce the in-distribution run's purity to within "
                 f"{max(diffs):.3f}" if diffs else "") + ".")


# --------------------------------------------------------------------------- #
# Label-free rules for the erasure budget N
# --------------------------------------------------------------------------- #
#: Thresholds fixed before any run. Rule A stops once the pseudo-grouping's
#: covariance subspace lies inside what directions-LEACE removes; rule C once
#: k-means on the erased training documents no longer recovers the grouping.
RULE_CAPTURE = 0.99
RULE_ARI = 0.10
#: The corpora of the plan's budget sweep (Table D2), which the rules are read against.
RULE_CORPORA = ("semeval", "swiltra", "bills_tweets", "scotus_oyez_lexis")


def _orth(A: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the column space of `A`."""
    U, s, _ = np.linalg.svd(np.asarray(A, np.float64), full_matrices=False)
    return U[:, s > (s.max() if s.size else 0.0) * 1e-10]


def _rule_context(X_tr: np.ndarray, base) -> dict:
    """What both signals need from one split, computed once for every N."""
    Xd = np.asarray(X_tr, np.float64)
    Xc = Xd - Xd.mean(axis=0, keepdims=True)
    K = int(base.pseudo_groups)
    G = P._one_hot(base.pseudo_group_train, np.arange(K)).astype(np.float64)
    return {"X_tr": np.asarray(X_tr, np.float32), "K": K, "seed": int(base.split.seed),
            "groups": np.asarray(base.pseudo_group_train),
            "decoder": np.asarray(base.decoder, np.float64), "Sigma": Xc.T @ Xc / len(Xc),
            "Q_G": _orth(Xc.T @ (G - G.mean(axis=0, keepdims=True)))}


def _rule_signals(ctx: dict, fitted) -> dict:
    """Both stopping signals for one top-N eraser, from training documents only.

    capture -- Table D6's capture with the k-means grouping G in place of the
    labels: the share of col(X_cᵀ G_c) lying in span(Σ D_Sᵀ), the subspace
    directions-LEACE removes. ari -- agreement between G and a k-means clustering
    (same K and seed) of the erased training embeddings. removed_variance -- the
    share of training variance the eraser removes, for the distortion guardrail.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    fn = fitted.erasers["sae_leace"]
    Q_E = _orth(ctx["Sigma"] @ ctx["decoder"][np.asarray(fitted.selected)].T)
    cos = np.clip(np.linalg.svd(ctx["Q_G"].T @ Q_E, compute_uv=False), 0, 1)
    E_tr = np.asarray(fn(ctx["X_tr"]), np.float32)
    labels = KMeans(n_clusters=ctx["K"], random_state=ctx["seed"], n_init="auto").fit_predict(E_tr)
    d = ctx["X_tr"].shape[1]
    M = np.eye(d) - ood.affine(fn, d)[0].T
    S = ctx["Sigma"]
    return {"capture": float((cos ** 2).sum() / max(ctx["Q_G"].shape[1], 1)),
            "ari": float(adjusted_rand_score(ctx["groups"], labels)),
            "removed_variance": float(np.trace(M @ S @ M.T) / np.trace(S))}


def choose_budget(seed_result: dict, grid, rule: str, capture: float = RULE_CAPTURE,
                  ari: float = RULE_ARI) -> tuple:
    """The N a rule picks for one seed: the smallest N in `grid` whose signal,
    averaged over the seed's folds, passes the threshold. Returns ``(N, reached)``;
    when no N passes, the largest N with ``reached=False``."""
    for n in grid:
        sig = seed_result["signals"][str(n)]
        if ((rule == "capture" and np.mean(sig["capture"]) >= capture)
                or (rule == "ari" and np.mean(sig["ari"]) <= ari)):
            return int(n), True
    return int(grid[-1]), False


def run_budget_rules(corpus: str, encoder: str = E.PRIMARY_ENCODER, grid=R.BUDGET_GRID,
                     seeds=R.SWEEP_SEEDS, device=None, overwrite: bool = False,
                     verbose: bool = True, **overrides) -> dict:
    """The plan's budget sweep with both label-free stopping signals recorded at every N.

    Same splits, one SAE per split, nested top-N prefixes of one MI ranking and the
    same held-out scoring as `revised.run_budget`, so the curve should reproduce
    Table D2; the signals see only the training documents. The rules are applied
    afterwards by `choose_budget`, so thresholds can be varied without a rerun, and
    every N is scored with the SAE that would have chosen it.
    """
    with artifacts.experiment(FOLDER):
        name = f"budget_rules_{corpus}_{encoder}.json"
        path = artifacts.artifact_path(name)
        if os.path.exists(path) and not overwrite:
            print(f"[cached] {path}")
            return artifacts.load_result(path)
        data = R.load(corpus, encoder, device=device, verbose=False)
        X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
        kw = R._kw(overrides)
        per_seed = []
        for s in seeds:
            t0 = time.time()
            splits = R._splits(data, s)
            bases = R._bases(data, splits, device=device, **overrides)
            ctxs = [_rule_context(X[sp.train_docs], b) for sp, b in zip(splits, bases)]
            points, signals = {}, {}
            for n in grid:
                fitted = [R._refit(data, sp, b, top_features=n, erase=kw["erase"])
                          for sp, b in zip(splits, bases)]
                sig = [_rule_signals(ctx, f) for ctx, f in zip(ctxs, fitted)]
                signals[str(n)] = {k: [v[k] for v in sig] for k in sig[0]}
                points[str(n)] = R.evaluate_fits(fitted, X, y, conditions=("sae_leace",),
                                                 topics=data.get("topics"))["metrics"]["sae_leace"]
            per_seed.append({"seed": int(s), "points": points, "signals": signals,
                             "pseudo_groups_k": [int(b.pseudo_groups) for b in bases],
                             "seconds": round(time.time() - t0, 1)})
            if verbose:
                e = per_seed[-1]
                print(f"[rules] {corpus} seed {s} ({e['seconds']}s): rule A picks "
                      f"N={choose_budget(e, grid, 'capture')[0]}, rule C picks "
                      f"N={choose_budget(e, grid, 'ari')[0]} | " + ", ".join(
                          f"N={n}: capture {np.mean(signals[str(n)]['capture']):.3f} "
                          f"ARI {np.mean(signals[str(n)]['ari']):.2f} "
                          f"AUC {points[str(n)]['linear_auc']:.3f}" for n in grid))
        out = {"experiment": "budget_rules", "corpus": corpus, "label": data["label"],
               "encoder": encoder, "seeds": [int(s) for s in seeds], "grid": [int(n) for n in grid],
               "paired": data["paired"], "per_seed": per_seed,
               "thresholds": {"capture": RULE_CAPTURE, "ari": RULE_ARI},
               "config": R._config(data, overrides)}
        artifacts.save_result(out, name)
        return out


def table_budget_rules(corpora=RULE_CORPORA, encoder: str = E.PRIMARY_ENCODER,
                       capture: float = RULE_CAPTURE, ari: float = RULE_ARI) -> list:
    """What the two label-free budget rules pick, and what that N gives held out."""
    import revised_exhibits as X

    runs = {}
    with artifacts.experiment(FOLDER):
        for c in corpora:
            p = R.raw_path(FOLDER, f"budget_rules_{c}_{encoder}.json")
            if os.path.exists(p):
                runs[c] = artifacts.load_result(p)
    if not runs:
        raise FileNotFoundError("no budget_rules results; run run_budget_rules() first")
    grid_all = sorted({n for r in runs.values() for n in r["grid"]})
    rows, sig_rows = [], []
    for c, r in runs.items():
        grid, seeds = r["grid"], r["per_seed"]
        util = "recall@10" if r["paired"] else None

        def at(picks, key, seeds=seeds):
            vals = [s["points"][str(n)].get(key) for s, n in zip(seeds, picks)]
            return None if key is None or None in vals else R._stat(vals)

        choices = []
        for label, rule in ((f"Rule A: capture ≥ {capture:g}", "capture"),
                            (f"Rule C: ARI ≤ {ari:g}", "ari")):
            picked = [choose_budget(s, grid, rule, capture, ari) for s in seeds]
            choices.append((label, [n for n, _ in picked],
                            ", ".join(f"{n}" + ("" if ok else "*") for n, ok in picked)))
        if 32 in grid:
            choices.append(("Fixed N = 32", [32] * len(seeds), ", ".join(["32"] * len(seeds))))
        if util:
            best = [max(grid, key=lambda n, s=s: s["points"][str(n)][util]) for s in seeds]
            choices.append(("Best N by held-out R@10 (uses pairs)", best, ", ".join(map(str, best))))
        if rows:
            rows.append(X.SEP)
        head = X.CORPUS.get(c, c)
        for label, ns, tag in choices:
            removed = R._stat([float(np.mean(s["signals"][str(n)]["removed_variance"]))
                               for s, n in zip(seeds, ns)])
            rows.append([head, label, tag, X.pm(at(ns, "linear_auc")),
                         X.pm(at(ns, util)) if util else "N/A", X.pm(removed)])
            head = ""
        plan = R.raw_path("budget", f"budget_{c}_{encoder}.json")
        if os.path.exists(plan):
            b = artifacts.load_result(plan)["curve"].get("32", {})
            rows.append(["", "Table D2 run, N = 32 (reproduction check)", "32",
                         X.pm(b.get("linear_auc")), X.pm(b.get(util)) if util else "N/A", "NR"])
        for key, label in (("capture", "Capture of the grouping subspace"),
                           ("ari", "ARI with the original grouping")):
            sig_rows.append([X.CORPUS.get(c, c) if key == "capture" else "", label] + [
                X.num(float(np.mean([np.mean(s["signals"][str(n)][key]) for s in seeds])))
                if n in grid else "NR" for n in grid_all])
    header = ["Corpus", "Choice of N", "N per seed", "Linear AUC", "R@10", "Removed variance"]
    with artifacts.experiment(None):
        return X.write_table(
            "table_budget_rules", "extra",
            "Label-free rules for the erasure budget N, E5-small. Mean ± SD over "
            + X.nseeds(runs) + " at each seed's chosen N.",
            [("Panel A — Chosen N and its held-out results", header, rows),
             ("Panel B — Rule signals by N (training documents, mean over seeds and folds)",
              ["Corpus", "Signal"] + [f"N = {n}" for n in grid_all], sig_rows)],
            note=f"Both rules see only the training documents: the SAE, its MI ranking and the "
                 f"silhouette-chosen k-means grouping G. Rule A picks the smallest N whose "
                 f"erased subspace span(Σ D_Sᵀ) captures at least {capture:g} of G's covariance "
                 f"subspace (Table D6's measure with G in place of the labels); rule C the "
                 f"smallest N at which k-means with the same K and seed, run on the erased "
                 f"training embeddings, agrees with G at ARI ≤ {ari:g}. Signals are averaged "
                 "over a seed's folds, so one N is chosen per seed; * marks a seed where no N "
                 "in the grid passed and the largest was used. The thresholds were fixed "
                 "before the runs. Every N is scored with the SAE that chose it, on the "
                 "held-out documents of Table D2's splits; the best-N row uses the held-out "
                 "pairs and is a reference, not a rule. Removed variance is the share of "
                 "training variance the eraser removes. The check row repeats Table D2.")


def build(corpora=CORPORA, encoder: str = E.PRIMARY_ENCODER) -> list:
    files = table_alternatives(corpora, encoder)
    try:
        files += table_selector_overlap(corpora, encoder)
    except FileNotFoundError as exc:
        print(f"[alternatives] {exc}")
    files += figure_frontier(corpora, encoder)
    try:
        files += figure_pc1_recall()
    except FileNotFoundError as exc:
        print(f"[alternatives] {exc}")
    try:
        files += table_encoder_ranks()
    except FileNotFoundError as exc:
        print(f"[alternatives] {exc}")
    for extra in (lambda: table_kmeans_count(corpora, encoder), table_ood_kmeans,
                  lambda: table_budget_rules(encoder=encoder), table_generous_k,
                  table_geometry, figure_geometry, table_cap_topics,
                  table_cap_entanglement):
        try:
            files += extra()
        except FileNotFoundError as exc:
            print(f"[alternatives] {exc}")
    return files
