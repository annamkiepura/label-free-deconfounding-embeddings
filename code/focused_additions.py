"""Focused additions: Table 3's SAE-free random directions (R1) and band-matched random SAE latents
(`notebooks/04_extensions.ipynb`, Sections 1-2), from the runs a review audit proposed.

**R1 -- SAE-free random directions vs random SAE latents vs MI-selected latents (E1, E3).**
Does the learned dictionary matter, or would any N covariance-weighted directions do? Every
arm goes through the same operator as SAE-LEACE -- LEACE on the concept ``Z = X_tr R^T`` with
standardized columns and the library defaults (`protocols._fit_leace`) -- on the same splits
and audit as the coverage cohort, at every budget N in `R1_BUDGETS`:

1. ``mi_n{N}``       the top-N latents of one MI ranking (nested prefixes; the SAE-LEACE of
                     the coverage runs at N = 32);
2. ``rsae{j}_n{N}``  N latents of the same SAE drawn uniformly without replacement from the
                     latents MI can score (training firing rate in `R1_BAND`), `R1_DRAWS`
                     draws, each a nested prefix of one permutation;
3. ``gauss{j}_n{N}`` N Gaussian directions, rows ~ N(0, I_d) scaled to unit norm, not
                     orthogonalized, drawn independently of the data and labels; `R1_DRAWS`
                     draws, each a nested prefix of one draw of `max(R1_BUDGETS)` rows.

Seeds: the Gaussian directions of seed s and draw j come from
``default_rng([R1_SEED, s, j, 0])`` and are shared by every fold of that seed; the latent
permutation of seed s, fold f and draw j from ``default_rng([R1_SEED, s, j, 1, f])``. The SAE,
split and k-means use the run seed exactly as every other run. ``rsae_mean_n{N}`` and
``gauss_mean_n{N}`` average the draws within a seed; seeds stay the unit of replication.
Per arm and fit the run also records the measured rank of ``I - A``, the displaced-variance
share (`alternatives._removed_variance`), the variance-reduction share and -- labels used post
hoc, descriptive only -- the capture of the label subspace L by ``span(Sigma R^T)``.

The notebook's default (``R1_FULL = False``) runs only what Main Table 3's pending row needs:
the N = 32 Gaussian arm, one draw per seed, no SAE and no reference arms (``budgets=(32,),
draws=1, sae_arms=False, references=False``) -- rows 1-2, Raw and Metadata-LEACE come from the
coverage runs on the same splits. ``R1_FULL = True`` runs the full design above.

`export_table3_slot` then writes the N = 32 Gaussian arm (mean over draws, per seed) into the
manuscript's pending Table 3 row, in the schema `paper_assets` reads
(``[pending.random_nonsae_directions]`` in `paper_assets/manifest.toml`), after checking seeds,
splits and audit against the coverage cohort file.

**R5 -- Prompt-consistent E5-small frozen transfer (I2).** E5-small donor documents were
encoded with ``"query: "`` (`run_sae_experiment.encode_texts`), the MTEB targets without a
prompt. R5 re-encodes every target text -- queries, documents and both STS sentences -- with
``"query: "`` (prompt style ``"donor_match"``) and re-scores the *published* frozen maps (the
``maps_file`` each published OOD run recorded, checked against its training-split
fingerprint). k-means LEACE (silhouette K) and the optional PCA-rank maps are refit on the same
donor training split exactly as `alternatives.run_ood_kmeans` / `run_ood_ranks` fit them. Every
map is also re-scored under the published (unprompted) targets and compared with the published
scores, so the only difference between the two summaries is the target prompt. MiniLM uses no
prompt on either side and is not rerun.

Outputs go to the active results root (the notebook uses ``results/extensions/``)::

    raw/r1/r1_<setting>_<encoder>.json         every arm, draw and diagnostic
    raw/r5/r5_<encoder>_<donor>[_ranks].json   both prompt styles, reproduction check
    raw/r5/r5_summary[_ranks]_<style>.json     ood.summarize of each style
    tables/r1/, tables/r5/                     Markdown comparison tables

Nothing here modifies a published result: the Table 3 slot file is new, and it is written
only on request.

**Effective erasure rank (`run_effective_rank`).** Main Figure 1 plots every eraser against a
*nominal* rank (PCA r, k-means K - 1, SAE budget N). This refits every map of the Figure 1 sweep on
the coverage splits -- PCA r, variance-matched PCA, fixed-K and silhouette k-means LEACE,
Metadata-LEACE, SAE-LEACE with MI top-N, with the coverage runs' random latents and with
band-matched random latents -- and reads off the fitted ``T(x) = x A + c``:
``rank(I - A)`` (every eraser here is a projection, so its nonzero singular values are >= 1 and
the rank is unambiguous; the gap and ranks at relative tolerances 1e-2..1e-4 are recorded), the
displaced-variance share, the number of leading PCs that displace the same share
(PCA-equivalent rank), and the participation ratio and 90% count of the displaced-variance
spectrum. No audit runs; the only cost is one SAE per split.

**Band-matched random SAE control (`run_band_control`).** MI scores only latents with training
firing rate in [0.02, 0.98]; the coverage runs' random control drew from firing rate >= 0.02.
This redraws the random latents from exactly the MI band, `BAND_DRAWS` draws per SAE (nested
prefixes of one permutation, the RNG of R1's ``rsae`` arm), at every budget of the published
random curve, and runs the audit on them (linear everywhere, MLP at N = 32). The SAE is refit with
the coverage recipe and seed; whether it reproduces the published MI selection is checked per split,
and where it does not, the MI arms are re-evaluated at every budget so the comparison stays
within one SAE.

Both share one SAE per split, cached as ``<results>/sae/sae_<setting>_<encoder>_s<seed>_f<fold>.npz``
(decoder, MI scores, firing rates), so the SAE is trained once however often the notebook runs.

**Paired gaps to Metadata-LEACE (`paired_gaps`).** Post-processing only. For the exhibits that
compare label-free erasers with Metadata-LEACE (Table 2, Figure 2 C, Table B1; optionally Table
B2), per seed ``Delta_s = m_s(label-free) - m_s(Metadata-LEACE)`` on the same split, read through
`paper_assets`' cohort loader (the exhibits' own files, seeds and compatibility checks), with a
t-interval over seeds and the number of seeds on each side of zero.

More outputs::

    raw/rank/effective_rank_<setting>_<encoder>.json   every map's rank diagnostics, per split
    raw/band/band_<setting>_<encoder>.json             band-matched arms, audited
    raw/paired/paired_gaps.json                        paired differences and intervals
    tables/{rank,band,paired}/, figures/rank/          Markdown tables, the rank figure
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

import alternatives as A
import artifacts
import ood
import paths
import protocols as P
import revised as R

# --------------------------------------------------------------------------- #
# R1
# --------------------------------------------------------------------------- #
R1_FOLDER = "r1"
R1_BUDGETS = (4, 8, 16, 32)
R1_DRAWS = 5
R1_SEED = 1000
#: The latents MI scores (`sae_confounder_pipeline.score_confounder_features`): the random
#: SAE arm draws from exactly this band (the coverage runs' random control used >= 0.02 only).
R1_BAND = (0.02, 0.98)
#: Main Table 3's settings (the pending row lives there), and the corpora R1 first proposed.
TABLE3_SETTINGS = ("swiltra_de_fr_it", "scotus_oyez_lexis_wiki")
R1_ORIGINAL = ("semeval", "scotus_oyez_lexis", "cap")
SLOT_FOLDER = "random_directions"
SLOT_CONDITION = "random_dirs_n32"
#: Where the published runs are looked up, in order (the Table 3 slot is written to the first).
PUBLISHED_ROOTS = (paths.MAIN,)


def gaussian_directions(dim: int, n_max: int, seed: int, draw: int) -> np.ndarray:
    """`n_max` unit-norm Gaussian rows; the first N rows are the budget-N set (nested)."""
    rng = np.random.default_rng([R1_SEED, int(seed), int(draw), 0])
    G = rng.standard_normal((int(n_max), int(dim)))
    return G / np.linalg.norm(G, axis=1, keepdims=True)


def latent_draw(pool, n_max: int, seed: int, fold: int, draw: int) -> np.ndarray:
    """One uniform permutation of the eligible latents; its first N entries are the budget-N set."""
    rng = np.random.default_rng([R1_SEED, int(seed), int(draw), 1, int(fold)])
    return rng.permutation(np.asarray(pool))[: int(n_max)]


def _label_subspace(X_tr: np.ndarray, y_tr, classes) -> tuple:
    """Centered training embeddings, their covariance and L = col(X_c^T Y_c) (rank C - 1)."""
    Xc = np.asarray(X_tr, np.float64) - np.asarray(X_tr, np.float64).mean(axis=0)
    Z = P._one_hot(y_tr, classes).astype(np.float64)
    return Xc, Xc.T @ Xc / len(Xc), A._orth(Xc.T @ (Z - Z.mean(axis=0)))


def _map_stats(eraser, X_tr: np.ndarray, S: np.ndarray, directions: np.ndarray, L: np.ndarray) -> dict:
    """Measured rank, displaced and variance-reduction shares, capture of L (post hoc)."""
    d = X_tr.shape[1]
    Amat = ood.affine(eraser, d)[0]
    sv = np.linalg.svd(np.eye(d) - Amat, compute_uv=False)
    Q = A._orth(S @ np.asarray(directions, np.float64).T)
    return {"rank": int((sv > 0.5).sum()),
            "displaced_share": float(A._removed_variance(eraser, X_tr)),
            "variance_reduction_share": float(1.0 - np.trace(Amat.T @ S @ Amat) / np.trace(S)),
            "capture_L": float(((Q.T @ L) ** 2).sum() / L.shape[1])}


def _mean_metrics(dicts: list) -> dict:
    """Average metric dicts over draws: numbers directly, nested dicts per key."""
    out = {}
    for k, v in dicts[0].items():
        if isinstance(v, dict):
            out[k] = {kk: float(np.mean([d[k][kk] for d in dicts])) for kk in v}
        elif isinstance(v, (int, float)):
            out[k] = float(np.mean([d[k] for d in dicts]))
    return out


def run_r1(setting: str, encoder: str = "e5-small", seeds=(0, 1, 2), budgets=R1_BUDGETS,
           draws: int = R1_DRAWS, nonlinear=("mlp",), sae_arms: bool = True, references: bool = True,
           device=None, fit_device=None, overwrite: bool = False, verbose: bool = True, **overrides) -> dict:
    """R1 on one setting: MI, random-SAE and Gaussian arms at every budget, on the registry
    splits of the coverage cohort. `nonlinear` probes run on the largest budget's arms (and Raw,
    Metadata-LEACE); the smaller budgets get the linear probe. `sae_arms=False` runs only the
    Gaussian arms, which need no SAE; `references=False` skips Raw and Metadata-LEACE (already in
    the coverage cohort, on the same splits). The fastest run that still fills Table 3's row is
    ``budgets=(32,), draws=1, sae_arms=False, references=False, nonlinear=("mlp",)``."""
    with artifacts.experiment(R1_FOLDER):
        name = f"r1_{setting}_{encoder}.json"
        path = artifacts.artifact_path(name)
        if os.path.exists(path) and not overwrite:
            print(f"[cached] {path}")
            return artifacts.load_result(path)
        kw = R._kw(overrides)
        budgets = sorted(int(n) for n in budgets)
        n_max = budgets[-1]
        data = R.load(setting, encoder, device=device, verbose=verbose)
        X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
        classes = np.unique(y)
        per_seed = []
        for s in seeds:
            t0, fits, diag = time.time(), [], []
            gauss = [gaussian_directions(X.shape[1], n_max, s, j) for j in range(draws)]
            for fi, sp in enumerate(R._splits(data, s)):
                tr = np.asarray(sp.train_docs)
                X_tr = X[tr]
                _, S, L = _label_subspace(X_tr, y[tr], classes)
                er = ({"raw": (lambda Z: np.asarray(Z, np.float32)),
                       "metadata_leace": P._fit_leace(X_tr, P._one_hot(y[tr], classes), standardise=False)}
                      if references else {})
                row = {"fold": fi, "n_train_docs": int(len(tr)), "label_rank": int(L.shape[1]), "arms": {}}
                arms = []                                  # (condition, directions)
                if sae_arms:
                    base = P.fit_base(X, sp, expansion_factor=kw["expansion_factor"], k=kw["k"],
                                      n_epochs=kw["n_epochs"], pseudo_groups=data["config"]["pseudo_groups"],
                                      device=fit_device)
                    D = np.asarray(base.decoder, np.float64)
                    freq = (base.feats_train > 0).mean(axis=0)
                    pool = np.where((freq >= R1_BAND[0]) & (freq <= R1_BAND[1]))[0]
                    if len(pool) < n_max:
                        raise ValueError(f"{setting} seed {s} fold {fi}: only {len(pool)} latents in the "
                                         f"MI band, fewer than N = {n_max}")
                    mi = np.argsort(base.scores)[::-1][:n_max]      # = select_confounder_features
                    arms += [(f"mi_n{n}", D[mi[:n]]) for n in budgets]
                    latents = [latent_draw(pool, n_max, s, fi, j) for j in range(draws)]
                    arms += [(f"rsae{j}_n{n}", D[latents[j][:n]]) for j in range(draws) for n in budgets]
                    row.update({"pool_size": int(len(pool)), "mi_selected": [int(i) for i in mi],
                                "random_latents": {f"draw{j}": [int(i) for i in lat]
                                                   for j, lat in enumerate(latents)},
                                "pseudo_groups_k": int(base.pseudo_groups),
                                "sae_info": {k: v for k, v in base.sae_info.items()
                                             if isinstance(v, (int, float, str))}})
                arms += [(f"gauss{j}_n{n}", gauss[j][:n]) for j in range(draws) for n in budgets]
                for c, Dir in arms:
                    er[c] = P._fit_leace(X_tr, X_tr @ np.asarray(Dir, np.float32).T)
                    row["arms"][c] = _map_stats(er[c], X_tr, S, Dir, L)
                fits.append(R.Erasers(sp, er))
                diag.append(row)
            conds = tuple(fits[0].erasers)
            big = [c for c in conds if c.endswith(f"_n{n_max}") or c in ("raw", "metadata_leace")]
            res = R.evaluate_fits(fits, X, y, conditions=conds, topics=data.get("topics"),
                                  nonlinear=nonlinear, nonlinear_conditions=big)
            res.pop("hits", None)
            m = res["metrics"]
            for kind in (("rsae", "gauss") if sae_arms else ("gauss",)):
                for n in budgets:
                    m[f"{kind}_mean_n{n}"] = _mean_metrics([m[f"{kind}{j}_n{n}"] for j in range(draws)])
            res["diagnostics"], res["seconds"] = diag, round(time.time() - t0, 1)
            per_seed.append(res)
            if verbose:
                show = [c for c in (f"mi_n{n_max}", f"rsae_mean_n{n_max}", f"gauss_mean_n{n_max}",
                                    "metadata_leace", "raw") if c in m]
                print(f"[r1] {setting}/{encoder} seed {s} ({res['seconds']}s) linear AUC at N = {n_max}: "
                      + "  ".join(f"{c} {m[c]['linear_auc']:.3f}" for c in show))
        out = {"experiment": "r1_random_directions", "corpus": setting, "label": data["label"],
               "encoder": encoder, "seeds": [int(s) for s in seeds], "budgets": budgets, "draws": int(draws),
               "sae_arms": bool(sae_arms), "references": bool(references),
               "conditions": list(per_seed[0]["metrics"]),
               "design": {"operator": "protocols._fit_leace(X_tr, X_tr @ R.T): LEACE, standardized concept "
                                      "columns, concept-erasure defaults",
                          "gaussian": "rows ~ N(0, I_d), unit norm, not orthogonalized, data-independent; "
                                      f"rng default_rng([{R1_SEED}, seed, draw, 0]), shared by a seed's folds",
                          "random_sae": f"uniform without replacement from latents with training firing rate in "
                                        f"{list(R1_BAND)}; rng default_rng([{R1_SEED}, seed, draw, 1, fold])",
                          "mi": "top-N of the MI ranking of the split's SAE (nested prefixes)",
                          "nesting": "budget-N sets are prefixes of one draw of max(budgets) directions",
                          "draw_mean": "<kind>_mean_n<N> averages the draws within a seed",
                          "nonlinear": f"{list(nonlinear)} on the N = {n_max} arms"
                                       + (", Raw and Metadata-LEACE" if references else "")},
               "per_seed": per_seed, "agg": R.aggregate(per_seed),
               "config": R._config(data, overrides, nonlinear=list(nonlinear)),
               "corpus_stats": R._corpus_stats(data)}
        out["reference_check"] = reference_check(out)
        artifacts.save_result(out, name)
        return out


def _published_path(rel: str, roots=None) -> str | None:
    """The first ``<root>/raw/<rel>`` that exists, across `PUBLISHED_ROOTS`."""
    for root in roots or PUBLISHED_ROOTS:
        p = os.path.join(root, "raw", rel)
        if os.path.exists(p):
            return p
    return None


def _published(rel: str, roots=None) -> dict | None:
    p = _published_path(rel, roots)
    if p is None:
        return None
    with open(p) as fh:
        return json.load(fh)


def reference_check(r1: dict, roots=None) -> dict:
    """Compare an R1 run with the coverage cohort's file for the same setting: identical
    splits/audit (needed for the Table 3 slot) and, if the SAE arms ran, whether the refit SAE
    reproduced the published MI selection and SAE-LEACE scores."""
    rel = f"alternatives/alternatives_{r1['corpus']}_{r1['encoder']}.json"
    ref = _published(rel, roots)
    if ref is None:
        return {"reference": None, "splits_match": False, "note": "no coverage-cohort file for this setting"}
    by_seed = {p["seed"]: p for p in ref["per_seed"]}
    out = {"reference": _published_path(rel, roots),
           "config_split_equal": r1["config"]["split"] == ref["config"]["split"],
           "config_audit_equal": r1["config"]["audit"] == ref["config"]["audit"], "seeds": {}}
    for p in r1["per_seed"]:
        q = by_seed.get(p["seed"])
        if q is None:
            out["seeds"][p["seed"]] = {"in_reference": False}
            continue
        row = {"in_reference": True, "split_equal": p.get("split") == q.get("split")}
        if r1["sae_arms"]:
            saved = [d.get("selected_mi") for d in q.get("diagnostics", [])]
            mine = [d["mi_selected"][:len(s_ or [])] for d, s_ in zip(p["diagnostics"], saved)]
            row["mi_selection_reproduced"] = bool(saved) and all(s_ is not None and list(a) == list(s_)
                                                                 for a, s_ in zip(mine, saved))
            n32 = f"mi_n{ref.get('sae_budget', 32)}"
            if n32 in p["metrics"] and "sae_mi" in q["metrics"]:
                row["sae_leace_linear_auc_diff"] = float(p["metrics"][n32]["linear_auc"]
                                                         - q["metrics"]["sae_mi"]["linear_auc"])
        out["seeds"][p["seed"]] = row
    out["splits_match"] = (out["config_split_equal"] and out["config_audit_equal"]
                           and all(v.get("split_equal") for v in out["seeds"].values()))
    return out


def export_table3_slot(setting: str, encoder: str = "e5-small", root: str = PUBLISHED_ROOTS[0],
                       overwrite: bool = False) -> str:
    """Write the N = 32 Gaussian arm (mean over draws, per seed) as the manuscript's pending
    Table 3 row, only if the run matches the coverage cohort in seeds, splits and audit."""
    with artifacts.experiment(R1_FOLDER):
        r1 = artifacts.load_result(artifacts.artifact_path(f"r1_{setting}_{encoder}.json"))
    ref = _published(f"alternatives/alternatives_{setting}_{encoder}.json", (root,))
    if ref is None:
        raise FileNotFoundError(f"{root}/raw/alternatives/alternatives_{setting}_{encoder}.json: no "
                                "coverage-cohort reference, so the slot cannot be checked")
    chk = reference_check(r1, (root,))
    seeds_ok = [p["seed"] for p in r1["per_seed"]] == [p["seed"] for p in ref["per_seed"]]
    if not (chk["splits_match"] and seeds_ok):
        raise ValueError(f"{setting}/{encoder}: R1 run does not match the coverage cohort "
                         f"(seeds {seeds_ok}; check {chk}); nothing was written")
    n = 32
    if f"gauss_mean_n{n}" not in r1["per_seed"][0]["metrics"]:
        raise ValueError(f"{setting}/{encoder}: the R1 run has no N = {n} Gaussian arm")
    slot = {"experiment": "random_directions", "corpus": setting, "encoder": encoder, "label": r1["label"],
            "seeds": r1["seeds"], "condition": SLOT_CONDITION,
            "definition": (f"LEACE on X R^T for N = {n} unit-norm Gaussian directions (Directions-LEACE "
                           "operator); " + ("one independent draw per seed" if r1["draws"] == 1 else
                                            f"per seed the mean over {r1['draws']} independent draws; draw "
                                            "values are kept as random_dirs_n32_draw<j>")),
            "per_seed": [{"seed": p["seed"], "split": p["split"],
                          "metrics": {SLOT_CONDITION: p["metrics"][f"gauss_mean_n{n}"],
                                      **{f"{SLOT_CONDITION}_draw{j}": p["metrics"][f"gauss{j}_n{n}"]
                                         for j in range(r1["draws"])}}}
                         for p in r1["per_seed"]],
            "config": r1["config"], "corpus_stats": r1["corpus_stats"], "design": r1["design"],
            "reference_check": chk,
            "source": os.path.join(artifacts.get_results_dir(), "raw", R1_FOLDER, f"r1_{setting}_{encoder}.json")}
    out = os.path.join(root, "raw", SLOT_FOLDER, f"random_directions_{setting}_{encoder}.json")
    if os.path.exists(out) and not overwrite:
        raise FileExistsError(f"{out} exists; pass overwrite=True to replace it")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(slot, fh, indent=1, default=artifacts._json_default)
    print(f"[r1] Table 3 slot written: {out}")
    return out


def _save_md(md: str, name: str) -> str:
    """A Markdown table in the active scope's ``tables/`` directory, under its own name."""
    p = artifacts.artifact_path(name, kind="table")
    with open(p, "w") as fh:
        fh.write(md + "\n")
    print(f"[saved table] {p}")
    return p


def _pm(vals) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    return f"{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"


def table_r1(settings=TABLE3_SETTINGS, encoder: str = "e5-small") -> str:
    """Markdown: per setting and budget, MI vs random SAE vs Gaussian (seed mean ± SD of the
    per-seed draw means), with the displaced-variance share and capture of L."""
    lines = []
    with artifacts.experiment(R1_FOLDER):
        for setting in settings:
            p = artifacts.artifact_path(f"r1_{setting}_{encoder}.json")
            if not os.path.exists(p):
                lines += [f"### {setting} / {encoder}: not run", ""]
                continue
            r = artifacts.load_result(p)
            first = next(iter(r["per_seed"][0]["metrics"].values()))
            util = "recall@10" if "recall@10" in first else ("topic_purity" if "topic_purity" in first else None)
            kinds = [("MI top-N", "mi_n{n}", "mi_n{n}"), ("random SAE", "rsae_mean_n{n}", "rsae{j}_n{n}"),
                     ("Gaussian", "gauss_mean_n{n}", "gauss{j}_n{n}")]
            kinds = [k for k in kinds if k[1].format(n=r["budgets"][0]) in r["per_seed"][0]["metrics"]]
            head = ["N", "arm", "linear AUC"] + ([util] if util else []) + ["MLP AUC", "displaced share",
                                                                             "capture of L"]
            lines += [f"### {r['label']} / {encoder} (seeds {r['seeds']}, {r['draws']} draws)", "",
                      "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            for ref in [c for c in ("raw", "metadata_leace") if c in r["per_seed"][0]["metrics"]]:
                v = [q["metrics"][ref] for q in r["per_seed"]]
                lines.append(f"| -- | {ref} | {_pm([x['linear_auc'] for x in v])} | "
                             + (f"{_pm([x.get(util) for x in v])} | " if util else "")
                             + f"{_pm([x.get('mlp_auc') for x in v])} | -- | -- |")
            for n in r["budgets"]:
                for label, cond, arm in kinds:
                    v = [q["metrics"][cond.format(n=n)] for q in r["per_seed"]]
                    st = [d["arms"][arm.format(n=n, j=j)] for q in r["per_seed"] for d in q["diagnostics"]
                          for j in (range(r["draws"]) if "{j}" in arm else (0,))]
                    lines.append(f"| {n} | {label} | {_pm([x['linear_auc'] for x in v])} | "
                                 + (f"{_pm([x.get(util) for x in v])} | " if util else "")
                                 + f"{_pm([x.get('mlp_auc') for x in v])} | "
                                 f"{np.mean([x['displaced_share'] for x in st]):.3f} | "
                                 f"{np.mean([x['capture_L'] for x in st]):.3f} |")
            chk = r.get("reference_check", {})
            lines += ["", f"Reference check vs the coverage cohort: splits match = {chk.get('splits_match')}; "
                          + "; ".join(f"seed {s}: " + ", ".join(f"{k}={v}" for k, v in d.items() if k != "in_reference")
                                      for s, d in chk.get("seeds", {}).items()), ""]
    md = "\n".join(lines)
    with artifacts.experiment(R1_FOLDER):
        _save_md(md, "table_r1.md")
    return md


# --------------------------------------------------------------------------- #
# R5
# --------------------------------------------------------------------------- #
R5_FOLDER = "r5"
R5_STYLE = "donor_match"
PUBLISHED_STYLE = "parent"
#: Figure B3 A-B's E5-small donors, and C's original donors for the rank sweep.
R5_DONORS = ("cap", "scotus_oyez_lexis_wiki", "swiltra_de_fr_it")
R5_RANK_DONORS = ("cap", "scotus_oyez_lexis", "swiltra")
R5_RANKS = (2, 3, 4, 8, 16, 32)
R5_LABELS = {"raw": "Raw", "pca_top1": "PC-1", "kmeans_leace": "k-means LEACE",
             "metadata_leace": "Metadata-LEACE", "sae_leace": "SAE-LEACE"}


def register_prompt_style() -> dict:
    """``"donor_match"``: E5 targets prefixed with ``"query: "`` everywhere, as the donors were."""
    e5 = {"query": "query: ", "document": "query: ", "sts": "query: "}
    ood.PROMPT_STYLES[R5_STYLE] = {"e5-small": e5, "e5-base-v2": e5, "mini": dict(ood._NONE)}
    return ood.PROMPT_STYLES[R5_STYLE]


def _load_published_maps(p_pub: dict, split_id: str, d: int, allow_refit: bool, X_tr, y_tr, classes) -> tuple:
    """The published PC-1 / Metadata-LEACE / SAE-LEACE maps of one seed, checked against the
    training split; or, for smoke tests only, refit PC-1 and Metadata-LEACE (no SAE-LEACE)."""
    path = p_pub.get("maps_file")
    if path and os.path.exists(path):
        with np.load(path) as z:
            if ood._split_id(z["train_docs"]) != split_id:
                raise ValueError(f"{path}: fitted on another training split than {split_id}")
            maps = {c: (z[f"{c}__A"].astype(np.float64), z[f"{c}__c"].astype(np.float64))
                    for c in R.OOD_TRANSFORMS}
            return maps, z["mean"].astype(np.float64), "published"
    if not allow_refit:
        raise FileNotFoundError(f"{path}: published maps not found. Run from the repository root with "
                                "the embedding cache (embedding_cache/corpora/maps/...) present.")
    maps = {"pca_top1": ood.affine(P._fit_pca_removal(X_tr, 1)[0], d),
            "metadata_leace": ood.affine(P._fit_leace(X_tr, P._one_hot(y_tr, classes), standardise=False), d)}
    return maps, np.asarray(X_tr, np.float64).mean(axis=0), "refit (smoke test: PC-1, Metadata-LEACE only)"


def run_r5(donor: str, encoder: str = "e5-small", ranks=(), tasks=tuple(ood.TARGETS), device=None,
           allow_refit_maps: bool = False, overwrite: bool = False, verbose: bool = True) -> dict:
    """Score the published frozen maps of one donor under the published (unprompted) and the
    donor-matched (``"query: "``) target encodings; both in `ood.summarize`'s input format."""
    register_prompt_style()
    with artifacts.experiment(R5_FOLDER):
        name = f"r5_{encoder}_{donor}{'_ranks' if ranks else ''}.json"
        path = artifacts.artifact_path(name)
        if os.path.exists(path) and not overwrite:
            print(f"[cached] {path}")
            return artifacts.load_result(path)
        pub_path = _published_path(f"ood/ood_{encoder}_{donor}.json")
        if pub_path is None:
            raise FileNotFoundError(f"ood/ood_{encoder}_{donor}.json not found under any of {PUBLISHED_ROOTS}")
        with open(pub_path) as fh:
            pub = json.load(fh)
        km_pub = _published(f"alternatives/ood_kmeans_{encoder}_{donor}.json")
        rk_pub = _published(f"alternatives/ood_ranks_{encoder}_{donor}.json") if ranks else None
        data = R.load(donor, encoder, device=device, verbose=verbose)
        X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
        classes, d = np.unique(y), X.shape[1]
        targets = {t: ood.load_target(t) for t in tasks}
        styles = (PUBLISHED_STYLE, R5_STYLE)
        embs = {st: ood.embed_targets(encoder, tasks=tuple(tasks), device=device, style=st, verbose=verbose)
                for st in styles}
        raw = {st: {t: ood.score_target(targets[t], embs[st][t]) for t in tasks} for st in styles}
        per_seed = {st: [] for st in styles}
        repro = {"raw": max(abs(raw[PUBLISHED_STYLE][t]["score"] - pub["raw"][t]["score"]) for t in tasks)}
        for p_pub in pub["per_seed"]:
            s = int(p_pub["seed"])
            sp = R._splits(data, s)[0]
            tr = np.asarray(sp.train_docs)
            split_id = ood._split_id(tr)
            if split_id != p_pub["fit"]["split_id"]:
                raise ValueError(f"{encoder}/{donor} seed {s}: rebuilt split {split_id} != published "
                                 f"{p_pub['fit']['split_id']}")
            maps, mu, source = _load_published_maps(p_pub, split_id, d, allow_refit_maps, X[tr], y[tr], classes)
            K = int(P.choose_pseudo_groups(X[tr], seed=sp.seed)[0])
            maps["kmeans_leace"] = ood.affine(R.kmeans_eraser(X[tr], K, sp.seed)[0], d)
            for r in ranks:
                maps[f"pca_top{int(r)}"] = ood.affine(A._pca_erasers(X[tr], (int(r),))[int(r)], d)
            order = [c for c in ("pca_top1", "kmeans_leace", "metadata_leace", "sae_leace") if c in maps] + \
                    [f"pca_top{int(r)}" for r in ranks]
            checks = {c: ood.check_operator(maps[c], mu) for c in order}
            fit = {"fit_on": "train", "split_id": split_id, "n_train_docs": int(len(tr)), "k_auto": K,
                   "maps_source": source}
            for st in styles:
                per_seed[st].append({"seed": s, "maps_file": p_pub.get("maps_file"), "fit": fit,
                                     "operator_checks": checks,
                                     "scores": {c: {t: ood.score_target(targets[t], embs[st][t], maps[c])
                                                    for t in tasks} for c in order}})
            # Reproduction under the published prompt style: every map against its published scores.
            mine = per_seed[PUBLISHED_STYLE][-1]["scores"]
            pairs = [(c, p_pub["scores"].get(c)) for c in R.OOD_TRANSFORMS if c in mine]
            if km_pub:
                q = next((x for x in km_pub["per_seed"] if int(x["seed"]) == s), None)
                if q and "kmeans_auto" in q["scores"]:
                    pairs.append(("kmeans_leace", q["scores"]["kmeans_auto"]))
                    fit["k_auto_published"] = q["fit"].get("k_auto")
            if rk_pub:
                q = next((x for x in rk_pub["per_seed"] if int(x["seed"]) == s), None)
                pairs += [(f"pca_top{int(r)}", q["scores"].get(f"pca_top{int(r)}")) for r in ranks if q]
            repro[f"seed{s}"] = {c: (None if ref is None else
                                     float(max(abs(mine[c][t]["score"] - ref[t]["score"]) for t in tasks)))
                                 for c, ref in pairs}
            if verbose:
                print(f"[r5] {encoder}/{donor} seed {s} ({source}, K={K}): max |score - published| "
                      + ", ".join(f"{c} {v:.1e}" if v is not None else f"{c} n/a"
                                  for c, v in repro[f"seed{s}"].items()))
        results = {}
        for st in styles:
            results[st] = {"experiment": "r5_frozen_transfer", "encoder": encoder, "donor": donor,
                           "donor_label": data["label"], "donor_role": None,
                           "seeds": [p["seed"] for p in per_seed[st]], "fit_on": "train",
                           "transforms": list(per_seed[st][0]["scores"]),
                           "labels": {c: R5_LABELS.get(c, c.replace("pca_top", "PCA-")) for c in
                                      ["raw"] + list(per_seed[st][0]["scores"])},
                           "tasks": ood.task_records(targets), "raw": raw[st], "per_seed": per_seed[st],
                           "protocol": {"mteb_version": ood.MTEB_VERSION, "prompt_style": st,
                                        "target_prompts": ood._prompts(encoder, st),
                                        "donor_prompt": "query: " if encoder.startswith("e5") else "none",
                                        "maps": "published run_id maps (PC-1, Metadata-LEACE, SAE-LEACE); "
                                                "k-means LEACE and PCA ranks refit on the same donor training "
                                                "split", "normalization": "maps applied without renormalization; cosine"}}
        out = {"experiment": "r5", "encoder": encoder, "donor": donor, "styles": results,
               "reproduction_max_abs_diff": repro, "published_run": pub_path}
        artifacts.save_result(out, name)
        return out


def summarize_r5(donors=R5_DONORS, encoder: str = "e5-small", n_boot: int = 10000, level: float = 0.95,
                 tag: str = "") -> dict:
    """`ood.summarize` (paired query bootstrap, seed 0, as published) for each prompt style.
    `tag` selects the run files (``"_ranks"`` for the rank sweep) and names the summaries."""
    out = {}
    with artifacts.experiment(R5_FOLDER):
        for st in (PUBLISHED_STYLE, R5_STYLE):
            res = {f"{encoder}/{dn}": artifacts.load_result(artifacts.artifact_path(f"r5_{encoder}_{dn}{tag}.json"))
                   ["styles"][st] for dn in donors}
            out[st] = ood.summarize(res, n_boot=n_boot, level=level, seed=0, save=False)
            artifacts.save_result(out[st], f"r5_summary{tag}_{st}.json")
    return out


def table_r5(summaries: dict, tag: str = "") -> str:
    """Markdown: legal macro and STS change per donor and map, published vs donor-matched prompts."""
    pub, new = summaries[PUBLISHED_STYLE], summaries[R5_STYLE]
    lab = lambda c: R5_LABELS.get(c, c.replace("pca_top", "PCA-"))
    fmt = lambda ch: f"{ch['delta']:+.4f} [{ch['ci'][0]:+.4f}, {ch['ci'][1]:+.4f}]" + (
        " *" if ch["ci"][0] > 0 or ch["ci"][1] < 0 else "")
    lines = ["| donor | map | legal Δ, unprompted targets | legal Δ, \"query: \" targets | "
             "STS Δ, unprompted | STS Δ, \"query: \" |", "|---|---|---|---|---|---|"]
    excl = {st: {} for st in (PUBLISHED_STYLE, R5_STYLE)}
    for key in pub["order"]:
        for c in pub["transforms"]:
            a, b = pub["settings"][key]["changes"][c], new["settings"][key]["changes"][c]
            sts = pub["settings"][key]["sts"][0] if pub["settings"][key]["sts"] else None
            lines.append(f"| {key} | {lab(c)} | {fmt(a['legal_macro'])} | {fmt(b['legal_macro'])} | "
                         + (f"{fmt(a[sts])} | {fmt(b[sts])} |" if sts else "-- | -- |"))
            for st, ch in ((PUBLISHED_STYLE, a), (R5_STYLE, b)):
                lo, hi = ch["legal_macro"]["ci"]
                excl[st].setdefault(c, []).append(lo > 0 or hi < 0)
    raw_lines = ["", "Raw legal macro nDCG@10: " + "; ".join(
        f"{k}: {pub['settings'][k]['scores']['raw']['legal_macro']['mean']:.4f} unprompted, "
        f"{new['settings'][k]['scores']['raw']['legal_macro']['mean']:.4f} \"query: \"" for k in pub["order"])]
    count = ["", "Legal-macro intervals excluding zero (*), per map: " + "; ".join(
        f"{lab(c)} {sum(excl[PUBLISHED_STYLE][c])}/{len(excl[PUBLISHED_STYLE][c])} unprompted, "
        f"{sum(excl[R5_STYLE][c])}/{len(excl[R5_STYLE][c])} \"query: \"" for c in excl[PUBLISHED_STYLE])]
    md = "\n".join(lines + raw_lines + count)
    with artifacts.experiment(R5_FOLDER):
        _save_md(md, f"table_r5{tag}.md")
    return md


# --------------------------------------------------------------------------- #
# Shared: one SAE per coverage split, cached
# --------------------------------------------------------------------------- #
SAE_FOLDER = "sae"
#: Main Figure 1's settings (its rank sweep and random curves); they include Table 3's two.
FIG1_SETTINGS = ("swiltra", "swiltra_de_fr_it", "scotus_oyez_lexis", "scotus_oyez_lexis_wiki")
#: The coverage runs' random-latent pool (`protocols.fit_pipeline`): firing rate >= 0.02, no upper bound.
LEGACY_FLOOR = 0.02


def _coverage_files(setting: str, encoder: str) -> tuple:
    """The coverage cohort's alternatives and k-means-count runs of one setting."""
    alt = _published(f"alternatives/alternatives_{setting}_{encoder}.json")
    kc = _published(f"alternatives/kmeans_count_{setting}_{encoder}.json")
    if alt is None or kc is None:
        raise FileNotFoundError(f"{setting}/{encoder}: the coverage cohort's alternatives and kmeans_count "
                                f"files are needed (looked in {PUBLISHED_ROOTS})")
    return alt, kc


def split_sae(X, sp, setting: str, encoder: str, fold: int, kw: dict, pseudo_groups="auto",
              fit_device=None, verbose: bool = True) -> dict:
    """The split's SAE as the directions arms need it -- decoder rows, MI scores, training firing
    rates, silhouette K -- fit exactly as the coverage runs fit it (`protocols.fit_base`, same recipe
    and seed) and cached in ``<results>/sae/``, so every analysis here shares one fit per split."""
    folder = os.path.join(artifacts.get_results_dir(), SAE_FOLDER)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"sae_{setting}_{encoder}_s{int(sp.seed)}_f{int(fold)}.npz")
    split_id = ood._split_id(np.asarray(sp.train_docs))
    if os.path.exists(path):
        with np.load(path) as z:
            sae = {k: z[k] for k in z.files}
        if str(sae["split_id"]) != split_id:
            raise ValueError(f"{path}: cached SAE was fit on split {sae['split_id']}, not {split_id}")
        return {**sae, "path": path}
    t0 = time.time()
    base = P.fit_base(X, sp, expansion_factor=kw["expansion_factor"], k=kw["k"], n_epochs=kw["n_epochs"],
                      pseudo_groups=pseudo_groups, device=fit_device)
    sae = {"decoder": np.asarray(base.decoder, np.float32), "scores": np.asarray(base.scores, np.float64),
           "freq": (np.asarray(base.feats_train) > 0).mean(axis=0), "k_auto": np.int64(base.pseudo_groups),
           "split_id": np.array(split_id), "fit_seconds": np.float64(time.time() - t0),
           "info": np.array(json.dumps({k: v for k, v in base.sae_info.items() if isinstance(v, (int, float, str))}))}
    np.savez(path, **sae)
    if verbose:
        print(f"[sae] {setting}/{encoder} seed {sp.seed} fold {fold}: fit in {sae['fit_seconds']:.0f}s -> {path}")
    return {**sae, "path": path}


def _pools(freq) -> dict:
    """MI's band (the latents it can score) and the coverage runs' random pool."""
    freq = np.asarray(freq)
    return {"band": np.where((freq >= R1_BAND[0]) & (freq <= R1_BAND[1]))[0],
            "legacy": np.where(freq >= LEGACY_FLOOR)[0]}


def _mi_order(sae: dict) -> np.ndarray:
    return np.argsort(sae["scores"])[::-1]          # = sae_confounder_pipeline.select_confounder_features


def _legacy_random(pool, n: int, split_seed: int) -> np.ndarray:
    """The coverage runs' random latents at budget n (`protocols.fit_pipeline`, one draw per budget)."""
    return np.random.default_rng(int(split_seed)).choice(pool, size=min(int(n), len(pool)), replace=False)


def _fit_dirs(X_tr, D) -> object:
    """The SAE-LEACE operator: LEACE on ``X_tr D^T``, standardized columns (`protocols.fit_pipeline`)."""
    return P._fit_leace(X_tr, X_tr @ np.asarray(D, np.float32).T)


def _sae_check(sae: dict, pub_diag: dict | None) -> dict:
    """Does the refit SAE reproduce the published fit (MI top-32 selection, silhouette K)?"""
    if not pub_diag or "selected_mi" not in pub_diag:
        return {"mi_set_reproduced": None, "mi_order_reproduced": None, "k_auto_reproduced": None}
    saved = [int(j) for j in pub_diag["selected_mi"]]
    mine = [int(j) for j in _mi_order(sae)[:len(saved)]]
    return {"mi_set_reproduced": set(mine) == set(saved), "mi_order_reproduced": mine == saved,
            "k_auto_reproduced": int(sae["k_auto"]) == int(pub_diag.get("pseudo_groups_k", -1))}


# --------------------------------------------------------------------------- #
# Effective erasure rank
# --------------------------------------------------------------------------- #
RANK_FOLDER = "rank"
RANK_EPS = (1e-2, 1e-3, 1e-4)
#: The curves of Main Figure 1 (and the extra maps), with the nominal x each is plotted at.
RANK_FAMILIES = {"pca": "PCA removal (r)", "pca_matched_var": "PCA, variance-matched (r*)",
                 "kmeans": "k-means LEACE, fixed K (K - 1)", "kmeans_leace": "k-means LEACE, silhouette K (K - 1)",
                 "metadata_leace": "Metadata-LEACE (C - 1)", "sae_mi": "SAE-LEACE, MI top-N (N)",
                 "sae_random": "SAE-LEACE, coverage random N (N)",
                 "sae_band": "SAE-LEACE, band-matched random N (N; mean over draws)"}


def _spectrum(X_tr) -> dict:
    """Centered training data, covariance and the cumulative PCA variance shares."""
    Xc = np.asarray(X_tr, np.float64) - np.asarray(X_tr, np.float64).mean(axis=0)
    S = Xc.T @ Xc / len(Xc)
    lam = np.clip(np.linalg.eigvalsh(S)[::-1], 0, None)
    return {"S": S, "tr": float(np.trace(S)), "cum": np.cumsum(lam) / lam.sum()}


def rank_stats(Amat: np.ndarray, spec: dict) -> dict:
    """Rank and size of the erasure ``x -> x A + c`` on the training data.

    ``rank``: the numerical rank of ``I - A``. Every map here is a projection (orthogonal for PCA,
    oblique for LEACE), whose nonzero singular values are >= 1, so 0.5 separates them (as
    `ood.check_operator`); the smallest kept and largest dropped singular values show the gap, and
    ``rank_eps`` repeats the count at tolerances relative to the largest singular value.
    ``displaced_share``: tr Cov(x - T(x)) / tr Cov(x) (`alternatives._removed_variance`).
    ``pca_equivalent_rank``: the fewest leading PCs that displace at least that share
    (`alternatives._matched_rank`). ``participation_ratio`` and ``k90``: how many directions carry
    the displaced variance (``(sum l)^2 / sum l^2`` and the count reaching 90%)."""
    d = Amat.shape[0]
    M = np.eye(d) - np.asarray(Amat, np.float64)
    sv = np.linalg.svd(M, compute_uv=False)
    r = int((sv > 0.5).sum())
    lam = np.clip(np.linalg.eigvalsh(M.T @ spec["S"] @ M)[::-1], 0, None)
    tot = float(lam.sum())
    share = tot / spec["tr"]
    return {"rank": r, "rank_eps": {f"{e:g}": int((sv > e * sv[0]).sum()) for e in RANK_EPS},
            "sv_kept_min": float(sv[r - 1]) if r else None, "sv_dropped_max": float(sv[r]) if r < d else None,
            "displaced_share": share,
            "pca_equivalent_rank": int(np.searchsorted(spec["cum"], share - 1e-6) + 1),
            "participation_ratio": float(tot ** 2 / (lam ** 2).sum()) if tot > 0 else 0.0,
            "k90": int(np.searchsorted(np.cumsum(lam) / tot, 0.9) + 1) if tot > 0 else 0}


def _family(cond: str) -> tuple:
    """``(family, nominal x, draw)`` of a map name."""
    for pat, fam, f in ((r"pca_top(\d+)", "pca", int), (r"kmeans_k(\d+)", "kmeans", lambda v: int(v) - 1),
                        (r"sae_mi_n(\d+)", "sae_mi", int), (r"sae_random_n(\d+)", "sae_random", int)):
        m = re.fullmatch(pat, cond)
        if m:
            return fam, f(m[1]), None
    m = re.fullmatch(r"sae_band(\d+)_n(\d+)", cond)
    if m:
        return "sae_band", int(m[2]), int(m[1])
    return cond, None, None


def run_effective_rank(setting: str, encoder: str = "e5-small", seeds=(0, 1, 2), draws: int = 5,
                       fit_device=None, overwrite: bool = False, verbose: bool = True, **overrides) -> dict:
    """Rank diagnostics of every map in Main Figure 1's sweep, refit on the coverage splits.

    The grids are the published runs' own (the alternatives file's PCA ranks and SAE budgets, the
    k-means-count file's K grid), so every plotted point has its map. PCA, k-means and
    Metadata-LEACE are deterministic refits; the SAE maps use the split's SAE (`split_sae`), and
    ``checks`` records whether it reproduces the published MI selection, silhouette K and
    variance-matched rank. Where the id run saved its SAE-LEACE map (``maps_file``) on the same
    split, that published map's diagnostics are recorded too, as a refit-free check at N = 32."""
    with artifacts.experiment(RANK_FOLDER):
        name = f"effective_rank_{setting}_{encoder}.json"
        path = artifacts.artifact_path(name)
        if os.path.exists(path) and not overwrite:
            print(f"[cached] {path}")
            return artifacts.load_result(path)
        kw = R._kw(overrides)
        alt, kc = _coverage_files(setting, encoder)
        idr = _published(f"id/id_{setting}_{encoder}.json")
        ranks, budgets, ks = alt["ranks"], alt["budget_grid"] or [alt.get("sae_budget", 32)], kc["ks"]
        n_max = max(budgets)
        data = R.load(setting, encoder, verbose=verbose)
        X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
        classes, d = np.unique(y), X.shape[1]
        fits = []
        for s in seeds:
            pub = next((p for p in alt["per_seed"] if int(p["seed"]) == int(s)), None)
            pkc = next((p for p in kc["per_seed"] if int(p["seed"]) == int(s)), None)
            pid = next((p for p in (idr or {}).get("per_seed", []) if int(p["seed"]) == int(s)), None)
            for fi, sp in enumerate(R._splits(data, s)):
                t0 = time.time()
                tr = np.asarray(sp.train_docs)
                X_tr = X[tr]
                spec = _spectrum(X_tr)
                sae = split_sae(X, sp, setting, encoder, fi, kw, data["config"]["pseudo_groups"], fit_device, verbose)
                D, mi, pools = sae["decoder"], _mi_order(sae), _pools(sae["freq"])
                K = int(P.choose_pseudo_groups(X_tr, seed=sp.seed)[0])
                maps = {f"pca_top{r}": fn for r, fn in A._pca_erasers(X_tr, ranks).items()}
                maps["kmeans_leace"] = R.kmeans_eraser(X_tr, K, sp.seed)[0]
                maps.update({f"kmeans_k{k}": R.kmeans_eraser(X_tr, k, sp.seed)[0] for k in ks})
                maps["metadata_leace"] = P._fit_leace(X_tr, P._one_hot(y[tr], classes), standardise=False)
                band = [latent_draw(pools["band"], n_max, s, fi, j) for j in range(draws)]
                for n in budgets:
                    maps[f"sae_mi_n{n}"] = _fit_dirs(X_tr, D[mi[:n]])
                    maps[f"sae_random_n{n}"] = _fit_dirs(X_tr, D[_legacy_random(pools["legacy"], n, sp.seed)])
                    maps.update({f"sae_band{j}_n{n}": _fit_dirs(X_tr, D[band[j][:n]]) for j in range(draws)})
                stats = {c: rank_stats(ood.affine(fn, d)[0], spec) for c, fn in maps.items()}
                r_star = int(stats[f"sae_mi_n{alt.get('sae_budget', 32)}"]["pca_equivalent_rank"])
                stats["pca_matched_var"] = rank_stats(ood.affine(A._pca_erasers(X_tr, (r_star,))[r_star], d)[0], spec)
                pdiag = (pub or {}).get("diagnostics", [{}])[fi] if pub else None
                checks = {**_sae_check(sae, pdiag),
                          "matched_rank_reproduced": (None if not pdiag else r_star == int(pdiag.get("matched_rank", -1))),
                          "k_auto_matches_kmeans_count": (None if not pkc else K == int(pkc["diagnostics"][fi]["k_auto"]))}
                published_map = None
                mf = (pid or {}).get("maps_file")
                if mf and os.path.exists(mf):
                    with np.load(mf) as z:
                        if ood._split_id(z["train_docs"]) == ood._split_id(tr):
                            published_map = {"file": mf, **rank_stats(z["sae_leace__A"].astype(np.float64), spec)}
                fits.append({"seed": int(s), "fold": fi, "split_id": ood._split_id(tr), "n_train_docs": int(len(tr)),
                             "k_auto": K, "n_classes": int(len(classes)), "pool_sizes": {k: int(len(v)) for k, v in pools.items()},
                             "checks": checks, "published_sae_leace_map": published_map, "sae_file": sae["path"],
                             "maps": stats, "seconds": round(time.time() - t0, 1)})
                if verbose:
                    mis = [stats[f"sae_mi_n{n}"]["rank"] for n in budgets]
                    print(f"[rank] {setting}/{encoder} seed {s} fold {fi} ({fits[-1]['seconds']}s): MI top-N ranks "
                          f"{dict(zip(budgets, mis))}; checks {checks}")
        out = {"experiment": "effective_rank", "corpus": setting, "label": data["label"], "encoder": encoder,
               "seeds": [int(s) for s in seeds], "ranks": list(ranks), "budgets": list(budgets), "ks": list(ks),
               "draws": int(draws), "families": RANK_FAMILIES, "fits": fits,
               "design": {"rank": "numerical rank of I - A (singular values > 0.5; projections have 0 or >= 1)",
                          "sae_band": f"latents drawn from MI's band {list(R1_BAND)} (R1's rsae RNG), nested prefixes",
                          "sae_random": f"the coverage runs' draw: firing rate >= {LEGACY_FLOOR}, "
                                        "default_rng(split seed).choice per budget",
                          "grids": "the published alternatives (ranks, budget_grid) and kmeans_count (ks) files"},
               "config": R._config(data, overrides), "published": {"alternatives": _published_path(
                   f"alternatives/alternatives_{setting}_{encoder}.json"), "kmeans_count": _published_path(
                   f"alternatives/kmeans_count_{setting}_{encoder}.json")}}
        artifacts.save_result(out, name)
        return out


def rank_summary(r: dict) -> list:
    """One row per (family, nominal x): rank range and means of the size diagnostics over fits
    (band-matched draws pooled)."""
    rows = {}
    for f in r["fits"]:
        for cond, st in f["maps"].items():
            fam, x, _ = _family(cond)
            if fam == "metadata_leace":
                x = f["n_classes"] - 1
            elif fam == "kmeans_leace":
                x = f["k_auto"] - 1
            elif fam == "pca_matched_var":
                x = st["rank"]
            rows.setdefault((fam, x), []).append(st)
    out = []
    for (fam, x), sts in rows.items():
        rk = [s["rank"] for s in sts]
        out.append({"family": fam, "nominal": x, "n": len(sts), "rank_min": min(rk), "rank_max": max(rk),
                    "rank_equals_nominal": sum(v == x for v in rk),
                    **{k: float(np.mean([s[k] for s in sts])) for k in
                       ("pca_equivalent_rank", "displaced_share", "participation_ratio", "k90")},
                    "pca_equivalent_rank_min": min(s["pca_equivalent_rank"] for s in sts),
                    "pca_equivalent_rank_max": max(s["pca_equivalent_rank"] for s in sts),
                    "sv_gap": [min(s["sv_kept_min"] or np.inf for s in sts),
                               max(s["sv_dropped_max"] or 0.0 for s in sts)]})
    order = list(RANK_FAMILIES)
    return sorted(out, key=lambda o: (order.index(o["family"]) if o["family"] in order else 99, o["nominal"]))


def table_effective_rank(settings=FIG1_SETTINGS, encoder: str = "e5-small") -> str:
    """Markdown: per setting, nominal vs numerical rank, PCA-equivalent rank and displaced share."""
    lines = []
    with artifacts.experiment(RANK_FOLDER):
        for setting in settings:
            p = artifacts.artifact_path(f"effective_rank_{setting}_{encoder}.json")
            if not os.path.exists(p):
                lines += [f"### {setting} / {encoder}: not run", ""]
                continue
            r = artifacts.load_result(p)
            pubm = [(f["published_sae_leace_map"], f["maps"]["sae_mi_n32"]) for f in r["fits"]
                    if f.get("published_sae_leace_map") and "sae_mi_n32" in f["maps"]]
            eps_agree = all(v == st["rank"] for f in r["fits"] for st in f["maps"].values()
                            for v in st["rank_eps"].values())
            lines += [f"### {r['label']} / {encoder} (seeds {r['seeds']}, {len(r['fits'])} fits, "
                      f"{r['draws']} band-matched draws)", "",
                      "| map | nominal | numerical rank | = nominal | PCA-equivalent rank | displaced share "
                      "| participation ratio | k90 |", "|---|---|---|---|---|---|---|---|"]
            for o in rank_summary(r):
                rk = f"{o['rank_min']}" if o["rank_min"] == o["rank_max"] else f"{o['rank_min']}-{o['rank_max']}"
                pe = (f"{o['pca_equivalent_rank']:.1f} ({o['pca_equivalent_rank_min']}-{o['pca_equivalent_rank_max']})")
                lines.append(f"| {RANK_FAMILIES.get(o['family'], o['family'])} | {o['nominal']} | {rk} | "
                             f"{o['rank_equals_nominal']}/{o['n']} | {pe} | {o['displaced_share']:.3f} | "
                             f"{o['participation_ratio']:.1f} | {o['k90']:.1f} |")
            gaps = [o["sv_gap"] for o in rank_summary(r)]
            lines += ["", f"Singular values of I - A over all maps: smallest kept {min(g[0] for g in gaps):.3f}, largest "
                          f"dropped {max(g[1] for g in gaps):.2e}; ranks at relative tolerances "
                          f"{', '.join(f'{e:g}' for e in RANK_EPS)} {'all equal' if eps_agree else 'DIFFER for some maps'}.",
                      "Refit checks per fit: " + "; ".join(
                          f"seed {f['seed']} fold {f['fold']}: " + ", ".join(f"{k}={v}" for k, v in f["checks"].items())
                          for f in r["fits"])]
            if pubm:
                lines.append("Published SAE-LEACE maps (id runs, same split): rank "
                             + ", ".join(f"{a['rank']} (refit {b['rank']})" for a, b in pubm)
                             + "; displaced share " + ", ".join(f"{a['displaced_share']:.3f} (refit {b['displaced_share']:.3f})"
                                                                 for a, b in pubm))
            lines.append("")
    md = "\n".join(lines)
    with artifacts.experiment(RANK_FOLDER):
        _save_md(md, "table_effective_rank.md")
    return md


def figure_effective_rank(settings=FIG1_SETTINGS, encoder: str = "e5-small") -> str | None:
    """Numerical rank (top) and PCA-equivalent rank (bottom) against Figure 1's nominal x, per setting;
    the dotted line is y = x. Means over fits, bars the range."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    runs = []
    with artifacts.experiment(RANK_FOLDER):
        for s in settings:
            p = artifacts.artifact_path(f"effective_rank_{s}_{encoder}.json")
            if os.path.exists(p):
                runs.append(artifacts.load_result(p))
    if not runs:
        return None
    style = {"pca": ("#4C72B0", "o"), "kmeans": ("#55A868", "s"), "sae_mi": ("#C44E52", "D"),
             "sae_random": ("#8C510A", "^"), "sae_band": ("#DD8452", "v"), "metadata_leace": ("black", "*")}
    fig, axes = plt.subplots(2, len(runs), figsize=(max(3.0 * len(runs), 7.0), 6.0), squeeze=False)
    for j, r in enumerate(runs):
        summ = [o for o in rank_summary(r) if o["family"] in style]
        for i, (key, lo, hi, ylab) in enumerate((("rank", "rank_min", "rank_max", "numerical rank of I - A"),
                                                 ("pca_equivalent_rank", "pca_equivalent_rank_min",
                                                  "pca_equivalent_rank_max", "PCA-equivalent rank"))):
            ax = axes[i, j]
            for fam, (col, mk) in style.items():
                pts = sorted((o for o in summ if o["family"] == fam), key=lambda o: o["nominal"])
                if not pts:
                    continue
                x = np.array([o["nominal"] for o in pts], float)
                m = np.array([o["rank_min"] + o["rank_max"] for o in pts]) / 2 if key == "rank" else \
                    np.array([o[key] for o in pts])
                err = np.array([[mm - o[lo], o[hi] - mm] for mm, o in zip(m, pts)]).T
                ax.errorbar(x, m, yerr=err, color=col, marker=mk, ms=4, lw=1, capsize=2,
                            label=RANK_FAMILIES[fam])
            ax.plot([1, 64], [1, 64], ":", color="grey", lw=0.8)
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32, 64]))
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                axis.set_minor_locator(NullLocator())
            ax.set_title(r["label"] if i == 0 else "", fontsize=9)
            ax.set_xlabel("nominal rank (Figure 1 x)" if i == 1 else "", fontsize=8)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8)
            ax.tick_params(labelsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    with artifacts.experiment(RANK_FOLDER):
        return artifacts.savefig(fig, "figure_effective_rank")


# --------------------------------------------------------------------------- #
# Band-matched random SAE control
# --------------------------------------------------------------------------- #
BAND_FOLDER = "band"
BAND_DRAWS = 5
BAND_SLOT_FOLDER = "random_band"


def run_band_control(setting: str, encoder: str = "e5-small", seeds=(0, 1, 2), budgets=None,
                     draws: int = BAND_DRAWS, nonlinear=("mlp",), nonlinear_budget: int = 32, mi: str = "auto",
                     fit_device=None, overwrite: bool = False, verbose: bool = True, **overrides) -> dict:
    """Random SAE latents drawn from exactly MI's eligible set, audited on the coverage splits.

    ``rband{j}_n{N}`` erases N latents of the split's SAE, the budget-N prefix of draw j's uniform
    permutation of the latents with training firing rate in `R1_BAND`; ``rband_mean_n{N}`` averages
    the draws within a seed. `budgets` defaults to the published random curve's grid. Linear audit
    everywhere, `nonlinear` probes at `nonlinear_budget` (Table 3's N = 32).

    ``mi="auto"`` also audits MI top-32 as a reproduction check when the refit SAE reproduces the
    published MI selection in every fold of the seed, and MI top-N at every budget when it does not,
    so the MI-vs-random comparison always has both arms on one SAE. ``"all"`` / ``"check"`` force
    either. Each finished seed is checkpointed; a rerun resumes after the last one."""
    with artifacts.experiment(BAND_FOLDER):
        name = f"band_{setting}_{encoder}.json"
        path = artifacts.artifact_path(name)
        if os.path.exists(path) and not overwrite:
            print(f"[cached] {path}")
            return artifacts.load_result(path)
        part_path = artifacts.artifact_path(f"band_{setting}_{encoder}.partial.json")
        kw = R._kw(overrides)
        alt, _ = _coverage_files(setting, encoder)
        budgets = sorted(int(n) for n in (budgets or alt["budget_grid"] or R1_BUDGETS))
        n_max, check_n = budgets[-1], int(alt.get("sae_budget", 32))
        data = R.load(setting, encoder, verbose=verbose)
        X, y = np.asarray(data["emb"], np.float32), np.asarray(data["source"])
        done = {}
        if os.path.exists(part_path) and not overwrite:
            prev = artifacts.load_result(part_path)
            if prev.get("budgets") == budgets and prev.get("draws") == draws:
                done = {int(p["seed"]): p for p in prev["per_seed"]}
                print(f"[band] resuming {setting}/{encoder}: seeds {sorted(done)} already done")
        per_seed = []
        for s in seeds:
            if int(s) in done:
                per_seed.append(done[int(s)])
                continue
            t0 = time.time()
            pub = next((p for p in alt["per_seed"] if int(p["seed"]) == int(s)), None)
            splits = R._splits(data, s)
            saes = [split_sae(X, sp, setting, encoder, fi, kw, data["config"]["pseudo_groups"], fit_device, verbose)
                    for fi, sp in enumerate(splits)]
            checks = [_sae_check(sae, pub["diagnostics"][fi] if pub else None) for fi, sae in enumerate(saes)]
            reproduced = all(c["mi_set_reproduced"] for c in checks)
            mi_budgets = {"all": budgets, "check": [check_n]}.get(mi, [check_n] if reproduced else budgets)
            fits, diag = [], []
            for fi, (sp, sae) in enumerate(zip(splits, saes)):
                X_tr = X[np.asarray(sp.train_docs)]
                D, order, pools = sae["decoder"], _mi_order(sae), _pools(sae["freq"])
                if len(pools["band"]) < n_max:
                    raise ValueError(f"{setting} seed {s} fold {fi}: {len(pools['band'])} latents in the MI band < N = {n_max}")
                lat = [latent_draw(pools["band"], n_max, s, fi, j) for j in range(draws)]
                er = {f"mi_n{n}": _fit_dirs(X_tr, D[order[:n]]) for n in mi_budgets}
                er.update({f"rband{j}_n{n}": _fit_dirs(X_tr, D[lat[j][:n]]) for j in range(draws) for n in budgets})
                legacy32 = _legacy_random(pools["legacy"], check_n, sp.seed)
                fits.append(R.Erasers(sp, er))
                diag.append({"fold": fi, "sae_file": sae["path"], "checks": checks[fi],
                             "pool_sizes": {k: int(len(v)) for k, v in pools.items()},
                             "legacy_pool_above_band": int((sae["freq"][pools["legacy"]] > R1_BAND[1]).sum()),
                             "legacy_random_n32_outside_band": int((sae["freq"][legacy32] > R1_BAND[1]).sum()),
                             "mi_top_n_in_band": bool(np.isin(order[:n_max], pools["band"]).all()),
                             "random_latents": {f"draw{j}": [int(i) for i in l] for j, l in enumerate(lat)}})
            conds = tuple(fits[0].erasers)
            res = R.evaluate_fits(fits, X, y, conditions=conds, topics=data.get("topics"), nonlinear=nonlinear,
                                  nonlinear_conditions=[c for c in conds if c.endswith(f"_n{nonlinear_budget}")])
            res.pop("hits", None)
            m = res["metrics"]
            for n in budgets:
                m[f"rband_mean_n{n}"] = _mean_metrics([m[f"rband{j}_n{n}"] for j in range(draws)])
            res.update(diagnostics=diag, mi_budgets=list(mi_budgets), seconds=round(time.time() - t0, 1))
            if pub and f"mi_n{check_n}" in m:
                res["mi_n32_minus_published"] = {mt: float(m[f"mi_n{check_n}"][mt] - pub["metrics"]["sae_mi"][mt])
                                                 for mt in ("linear_auc", "mlp_auc", "recall@10", "topic_purity")
                                                 if mt in m[f"mi_n{check_n}"] and mt in pub["metrics"]["sae_mi"]}
            per_seed.append(res)
            artifacts.save_result({"budgets": budgets, "draws": draws, "per_seed": per_seed}, part_path)
            if verbose:
                show = [c for c in (f"mi_n{check_n}", f"rband_mean_n{check_n}") if c in m]
                was = (f"  | published sae_mi {pub['metrics']['sae_mi']['linear_auc']:.3f}, "
                       f"sae_random {pub['metrics']['sae_random']['linear_auc']:.3f}") if pub else ""
                print(f"[band] {setting}/{encoder} seed {s} ({res['seconds']}s, MI reproduced: {reproduced}) "
                      f"linear AUC at N = {check_n}: " + "  ".join(f"{c} {m[c]['linear_auc']:.3f}" for c in show) + was)
        out = {"experiment": "band_random_sae", "corpus": setting, "label": data["label"], "encoder": encoder,
               "seeds": [int(s) for s in seeds], "budgets": budgets, "draws": int(draws),
               "conditions": list(per_seed[0]["metrics"]),
               "design": {"operator": "protocols._fit_leace(X_tr, X_tr @ D_S^T) (Directions-LEACE, as SAE-LEACE)",
                          "pool": f"latents with training firing rate in {list(R1_BAND)}: the set MI scores "
                                  "(sae_confounder_pipeline.score_confounder_features)",
                          "draws": f"{draws} per SAE; draw j of seed s, fold f: uniform permutation from "
                                   f"default_rng([{R1_SEED}, s, j, 1, f]); budget-N set = its first N latents (nested)",
                          "draw_mean": "rband_mean_n<N> averages the draws within a seed; seeds are the unit",
                          "mi": f"mi={mi!r}: MI top-{check_n} as a reproduction check, every budget if the refit "
                                "SAE does not reproduce the published MI selection",
                          "nonlinear": f"{list(nonlinear)} at N = {nonlinear_budget}",
                          "sae": "protocols.fit_base with the coverage recipe and seed, cached per split"},
               "per_seed": per_seed, "agg": R.aggregate(per_seed),
               "config": R._config(data, overrides, nonlinear=list(nonlinear)),
               "corpus_stats": R._corpus_stats(data)}
        out["reference_check"] = _band_reference(out, alt)
        artifacts.save_result(out, name)
        if os.path.exists(part_path):
            os.remove(part_path)
        return out


def _band_reference(r: dict, alt: dict) -> dict:
    """Same splits and audit as the coverage file; the refit SAE's MI selection per fit."""
    by_seed = {int(p["seed"]): p for p in alt["per_seed"]}
    seeds = {int(p["seed"]): {"split_equal": p.get("split") == by_seed.get(int(p["seed"]), {}).get("split"),
                              "mi_reproduced": all(d["checks"]["mi_set_reproduced"] for d in p["diagnostics"]),
                              "mi_n32_minus_published": p.get("mi_n32_minus_published")}
             for p in r["per_seed"]}
    return {"reference": _published_path(f"alternatives/alternatives_{r['corpus']}_{r['encoder']}.json"),
            "config_split_equal": r["config"]["split"] == alt["config"]["split"],
            "config_audit_equal": r["config"]["audit"] == alt["config"]["audit"], "seeds": seeds,
            "splits_match": (r["config"]["split"] == alt["config"]["split"] and r["config"]["audit"] == alt["config"]["audit"]
                             and all(v["split_equal"] for v in seeds.values())),
            "mi_reproduced_everywhere": all(v["mi_reproduced"] for v in seeds.values())}


def table_band(settings=FIG1_SETTINGS, encoder: str = "e5-small") -> str:
    """Markdown: per budget, MI top-N (published, and refit where audited) against the coverage
    runs' random latents and the band-matched draws (seed mean ± SD of the per-seed draw means)."""
    lines = []
    with artifacts.experiment(BAND_FOLDER):
        for setting in settings:
            p = artifacts.artifact_path(f"band_{setting}_{encoder}.json")
            if not os.path.exists(p):
                lines += [f"### {setting} / {encoder}: not run", ""]
                continue
            r = artifacts.load_result(p)
            alt, _ = _coverage_files(setting, encoder)
            pub = {int(q["seed"]): q["metrics"] for q in alt["per_seed"]}
            first = r["per_seed"][0]["metrics"][f"rband_mean_n{r['budgets'][-1]}"]
            util = "recall@10" if "recall@10" in first else ("topic_purity" if "topic_purity" in first else None)
            head = ["N", "arm", "linear AUC"] + ([util] if util else []) + ["MLP AUC"]
            lines += [f"### {r['label']} / {encoder} (seeds {r['seeds']}, {r['draws']} band-matched draws per SAE)", "",
                      "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            for n in r["budgets"]:
                arms = [("MI top-N (published)", [pub[s].get(f"sae_mi_n{n}") for s in r["seeds"]]),
                        ("MI top-N (refit)", [q["metrics"].get(f"mi_n{n}") for q in r["per_seed"]]),
                        ("random, coverage pool (published)", [pub[s].get(f"sae_random_n{n}") for s in r["seeds"]]),
                        ("random, MI band (mean of draws)", [q["metrics"][f"rband_mean_n{n}"] for q in r["per_seed"]])]
                for label, v in arms:
                    if any(x is None for x in v):
                        continue
                    lines.append(f"| {n} | {label} | {_pm([x['linear_auc'] for x in v])} | "
                                 + (f"{_pm([x.get(util) for x in v])} | " if util else "")
                                 + f"{_pm([x.get('mlp_auc') for x in v])} |")
            chk = r["reference_check"]
            outside = [d["legacy_random_n32_outside_band"] for q in r["per_seed"] for d in q["diagnostics"]]
            lines += ["", f"Reference check: splits match = {chk['splits_match']}; MI selection reproduced in every fit = "
                          f"{chk['mi_reproduced_everywhere']}; " + "; ".join(
                              f"seed {s}: MI top-32 refit - published = {v['mi_n32_minus_published']}" for s, v in chk["seeds"].items()),
                      f"Published random N = 32 latents above the band (firing rate > {R1_BAND[1]}), per fit: {outside}", ""]
    md = "\n".join(lines)
    with artifacts.experiment(BAND_FOLDER):
        _save_md(md, "table_band.md")
    return md


def export_band(setting: str, encoder: str = "e5-small", root: str = PUBLISHED_ROOTS[0],
                allow_refit_sae: bool = False, overwrite: bool = False) -> str:
    """Write the band-matched random arm as a result file beside the published runs,
    ``<root>/raw/random_band/random_band_<setting>_<encoder>.json``: per seed ``sae_random_band_n<N>``
    (the draw mean), each draw as ``..._draw<j>``, and ``sae_random_band`` = N = 32 -- the schema of the
    coverage files, for Table 3's row 2 and Figure 1's random curve. Refuses unless the run's splits
    and audit equal the coverage cohort's and (unless `allow_refit_sae`) the refit SAE reproduced
    the published MI selection in every fit; with `allow_refit_sae`, the refit MI arms are written
    as ``sae_mi_refit_n<N>`` so both rows can come from one SAE."""
    with artifacts.experiment(BAND_FOLDER):
        r = artifacts.load_result(artifacts.artifact_path(f"band_{setting}_{encoder}.json"))
    chk = r["reference_check"]
    if not chk["splits_match"]:
        raise ValueError(f"{setting}/{encoder}: band run does not match the coverage cohort's splits/audit; nothing written")
    if not (chk["mi_reproduced_everywhere"] or allow_refit_sae):
        raise ValueError(f"{setting}/{encoder}: the refit SAE did not reproduce the published MI selection in every "
                         "fit, so the band arm is not on the published SAE. Pass allow_refit_sae=True to export it with "
                         "the refit MI arms; nothing written")
    per_seed = []
    for q in r["per_seed"]:
        m = q["metrics"]
        met = {f"sae_random_band_n{n}": m[f"rband_mean_n{n}"] for n in r["budgets"]}
        met.update({f"sae_random_band_n{n}_draw{j}": m[f"rband{j}_n{n}"] for n in r["budgets"] for j in range(r["draws"])})
        if 32 in r["budgets"]:
            met["sae_random_band"] = m["rband_mean_n32"]
        if allow_refit_sae:
            met.update({f"sae_mi_refit_n{n}": m[f"mi_n{n}"] for n in q["mi_budgets"]})
        per_seed.append({"seed": q["seed"], "split": q["split"], "metrics": met})
    slot = {"experiment": "random_band", "corpus": setting, "encoder": encoder, "label": r["label"], "seeds": r["seeds"],
            "budgets": r["budgets"], "draws": r["draws"], "condition": "sae_random_band",
            "definition": (f"Directions-LEACE on N SAE decoder rows drawn uniformly from the latents MI scores "
                           f"(training firing rate in {list(R1_BAND)}); per seed the mean over {r['draws']} draws"),
            "same_sae_as_published": bool(chk["mi_reproduced_everywhere"]),
            "per_seed": per_seed, "config": r["config"], "corpus_stats": r["corpus_stats"], "design": r["design"],
            "reference_check": chk,
            "source": os.path.join(artifacts.get_results_dir(), "raw", BAND_FOLDER, f"band_{setting}_{encoder}.json")}
    out = os.path.join(root, "raw", BAND_SLOT_FOLDER, f"random_band_{setting}_{encoder}.json")
    if os.path.exists(out) and not overwrite:
        raise FileExistsError(f"{out} exists; pass overwrite=True to replace it")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(slot, fh, indent=1, default=artifacts._json_default)
    print(f"[band] written: {out}")
    return out


# --------------------------------------------------------------------------- #
# Paired gaps to Metadata-LEACE (post-processing)
# --------------------------------------------------------------------------- #
PAIRED_FOLDER = "paired"
PAIRED_METHODS = ("pca_top1", "kmeans_auto", "sae_leace")
PAIRED_REF = "metadata_leace"
#: Reporting margins for "within +-delta of Metadata-LEACE": the ones the generous-K verdict uses
#: (`alternatives.GK_META_MARGIN` for AUC, `GK_UTILITY_DROP` for utility).
PAIRED_MARGINS = {"auc": A.GK_META_MARGIN, "utility": A.GK_UTILITY_DROP}
PAIRED_GROUPS = ("table2", "fig2c", "tableB1")


def _paper_assets():
    """`paper_assets` (at the repository root), for the exhibits' own cohorts and loaders."""
    root = os.getcwd()
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from paper_assets.lib import config as pa_config, loaders, aggregate, provenance
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"paired_gaps reads the exhibits' cohorts through paper_assets/, which is not in "
                                  f"{root}. Copy the repository's paper_assets/ folder there, or run this "
                                  "post-processing step locally (it needs no GPU or embeddings).") from e
    return pa_config, loaders, aggregate, provenance


def _paired_cells(groups) -> dict:
    """(setting, encoder, cohort) -> the exhibits it appears in, from the exhibit modules themselves."""
    pa_config, *_ = _paper_assets()
    from paper_assets.exhibits import table2_primary_id as T2, tableB1_extended_id as B1
    cells = {}

    def add(setting, enc, cohort, ex):
        cells.setdefault((setting, enc, cohort), []).append(ex)
    for g in groups:
        if g == "table2":
            for k, enc in T2.COLUMNS:
                add(k, enc, pa_config.cohort_for(k, enc), "Table 2")
        elif g == "fig2c":                     # Figure 2's six rows (fig2_geometry_discovery.LABELS)
            for k, enc in (("swiltra", "e5-small"), ("swiltra_de_fr_it", "e5-small"), ("scotus_oyez_lexis", "e5-small"),
                           ("scotus_oyez_lexis", "mini"), ("scotus_oyez_lexis_wiki", "e5-small"),
                           ("scotus_oyez_lexis_wiki", "mini")):
                add(k, enc, pa_config.cohort_for(k, enc), "Figure 2C")
        elif g == "tableB1":
            for _, _, items in B1.BLOCKS:
                for k, _ in items:
                    add(k, B1.ENCODER, pa_config.cohort_for(k, B1.ENCODER), "Table B1")
        elif g == "tableB2":
            from paper_assets.exhibits import tableB2_ten_encoder_robustness as B2
            import experiments as E
            for k, _ in B2.CORPORA:
                for enc in E.PARENT_PAPER_ENCODERS:
                    add(k, enc, B2.COHORT, "Table B2")
        else:
            raise ValueError(f"unknown group {g!r}")
    return cells


def _t_interval(v, level: float) -> tuple:
    from scipy import stats
    v = np.asarray(v, float)
    if len(v) < 2:
        return None, None
    h = stats.t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean() - h), float(v.mean() + h)


def paired_gaps(groups=PAIRED_GROUPS, methods=PAIRED_METHODS, level: float = 0.95, margins=None) -> dict:
    """Per-seed paired differences to Metadata-LEACE, with a t-interval over seeds.

    Unit: the fitting seed (each seed has its own split and fits; folds are already combined
    within a seed, as in every exhibit). ``Delta_s = m_s(method) - m_s(Metadata-LEACE)`` on the same
    split; the interval is ``mean +- t_{n-1} SD / sqrt(n)``, so it covers split-and-fit variability,
    not the sampling of documents within a split. ``verdict`` compares the interval with zero
    ("above"/"below") and with the reporting margin (+-0.03 AUC, +-0.02 utility): "within" when the
    whole interval lies inside it, otherwise "inconclusive" when it crosses zero."""
    pa_config, loaders, aggregate, provenance = _paper_assets()
    margins = {**PAIRED_MARGINS, **(margins or {})}
    n_issues = len(provenance.ISSUES)
    rows = []
    for (setting, enc, cohort), exhibits in _paired_cells(groups).items():
        coh = loaders.Cohort(cohort, setting, enc)
        coh.compatibility()                   # not quietly: its errors are reported below
        util = pa_config.setting(setting).get("utility") or None
        metrics = [("linear_auc", "auc"), ("mlp_auc", "auc")] + ([(util, "utility")] if util else [])
        metrics += [(m, "utility") for m in ("gold_sim_spearman", "topic_ari") if m != util]
        for metric, kind in metrics:
            ref = coh.values(PAIRED_REF, metric, register=False, quiet=True)
            if not ref.ok:
                continue
            for meth in methods:
                a = coh.values(meth, metric, register=False, quiet=True)
                if not a.ok:
                    continue
                dv = aggregate.paired_difference(a, ref)
                if not dv.ok:
                    continue
                v = np.asarray(dv.values, float)
                lo, hi = _t_interval(v, level)
                delta = margins[kind]
                side = None if lo is None else ("above" if lo > 0 else "below" if hi < 0 else None)
                within = lo is not None and lo >= -delta and hi <= delta
                verdict = ("n/a" if lo is None else f"{side}, within ±{delta:g}" if side and within
                           else side if side else f"within ±{delta:g}" if within else "inconclusive")
                rows.append({"setting": setting, "encoder": enc, "cohort": cohort, "exhibits": exhibits,
                             "method": meth, "metric": metric, "kind": kind, "seeds": list(dv.seeds),
                             "method_values": list(a.values), "metadata_values": list(ref.values),
                             "delta": [float(x) for x in v], "mean": float(v.mean()),
                             "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "ci": [lo, hi], "level": level,
                             "n_above": int((v > 0).sum()), "n_below": int((v < 0).sum()),
                             "margin": delta, "verdict": verdict,
                             "sources": [s["file"] for s in dv.sources]})
    errors = [i for i in provenance.ISSUES[n_issues:] if i["level"] == "error"]
    out = {"experiment": "paired_gaps", "groups": list(groups), "methods": list(methods), "reference": PAIRED_REF,
           "level": level, "margins": margins, "rows": rows, "loader_errors": errors,
           "design": {"unit": "fitting seed (own split and fits); folds combined within seed as in the exhibits",
                      "difference": "method - Metadata-LEACE per seed, same split (paper_assets paired_difference)",
                      "interval": f"t-interval over seeds at level {level} (df = n - 1)",
                      "not_covered": "within-split sampling of documents/pairs (probe predictions are not saved)"}}
    with artifacts.experiment(PAIRED_FOLDER):
        artifacts.save_result(out, "paired_gaps.json")
    return out


def table_paired(res: dict) -> str:
    """Markdown: one row per setting x encoder x method x metric, with the exhibits it feeds."""
    lab = {"pca_top1": "PC-1", "kmeans_auto": "k-means LEACE", "sae_leace": "SAE-LEACE"}
    short = {"Table 2": "T2", "Figure 2C": "F2C", "Table B1": "B1", "Table B2": "B2"}
    lines = [f"Δ = method − Metadata-LEACE per seed (same split); {int(res['level'] * 100)}% t-interval over seeds "
             f"(the fitting seed is the unit). AUC: Δ > 0 = more leakage than Metadata-LEACE; utility: Δ > 0 = "
             f"better utility. Verdict: interval above/below 0, and/or inside ±{res['margins']['auc']:g} (AUC) / "
             f"±{res['margins']['utility']:g} (utility). Loader errors: {len(res['loader_errors'])}.", "",
             "| setting | encoder | seeds | exhibits | method | metric | Δ mean | 95% CI | seeds Δ>0 / Δ<0 | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        ci = "--" if r["ci"][0] is None else f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"
        lines.append(f"| {r['setting']} | {r['encoder']} | {len(r['seeds'])} | "
                     f"{', '.join(short.get(e, e) for e in r['exhibits'])} | {lab.get(r['method'], r['method'])} | "
                     f"{r['metric']} | {r['mean']:+.3f} | {ci} | {r['n_above']} / {r['n_below']} | {r['verdict']} |")
    md = "\n".join(lines)
    with artifacts.experiment(PAIRED_FOLDER):
        _save_md(md, "table_paired_gaps.md")
    return md
