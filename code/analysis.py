"""What is being removed? -- descriptive analyses of one fit per corpus.

These are **descriptive, not evaluative**. Each analysis here uses a single SAE
per corpus (seed 0, the primary encoder, the Section 4.1 configuration) fit on
the *whole* corpus with no pair holdout, because the object of study is what the
method finds on a corpus rather than how well it generalises to unseen
documents. `fit_descriptive` therefore builds its pipeline with
``protocol="none"``, which `protocols.evaluate` refuses to score: no performance
estimate can be quoted from anything in this module.

Two questions:

**Reading the selected latents.** Do latents chosen without labels turn out,
after the fact, to track the confounder? For each selected latent we record the
label-free quantities that drove its selection -- mutual-information score,
pair asymmetry, firing frequency -- and then compute its one-vs-rest ROC-AUC
against each class from the latent's activation. Labels enter only here, after
selection is finished, and never feed back into it.

**Geometry of the removed subspace.** Does the subspace SAE-LEACE removes line up
with the linear language signal? The target is the *language subspace*
``L = col(X_c^T Y_c)`` -- the span of the embedding-label cross-covariance,
which is what LEACE's guardedness guarantee concerns, of rank at most ``C-1``.
Four size-matched candidates are compared against it by principal angles and by
normalised capture ``||Q_E^T Q_L||_F^2 / rank(L)``, the mean squared cosine of
those angles: 1 means the candidate contains all of L, 0 means orthogonality.
"""

from __future__ import annotations

import numpy as np

import protocols as P
from artifacts import artifact_path, save_result, save_text
from experiments import PRIMARY_ENCODER, RECIPE, embed
from sae import encode_preactivations
from sae_confounder_pipeline import pair_asymmetry

#: Conditions the PCA panel shows, in reading order.
PCA_CONDITIONS = ("raw", "metadata_leace", "sae_leace")


# --------------------------------------------------------------------------- #
# The single descriptive fit each analysis reads
# --------------------------------------------------------------------------- #
def fit_descriptive(
    corpus: str,
    encoder: str = PRIMARY_ENCODER,
    seed: int = 0,
    n="keep",
    device: str | None = None,
    data: dict | None = None,
    verbose: bool = True,
    **overrides,
) -> dict:
    """One SAE per corpus on the full corpus, no holdout. Returns a bundle.

    The bundle carries everything the analyses below need -- embeddings, texts,
    labels, the trained SAE and its dense pre-activations, the decoder, the
    label-free scores, the selected latents, and the erased embeddings for each
    condition -- so a notebook fits once per corpus and passes the bundle around.
    """
    data = data or embed(corpus, encoder, n=n, device=device, verbose=verbose)
    kw = {**RECIPE, **overrides}
    split = P.make_split(data["n_docs"], data["source"], pairs=data["pairs"],
                         gold_sim=data["gold"], seed=seed, protocol="none")
    fitted = P.fit_pipeline(
        data["emb"], data["source"], split,
        expansion_factor=kw["expansion_factor"], k=kw["k"], n_epochs=kw["n_epochs"],
        pseudo_groups=data["config"]["pseudo_groups"], erase=kw["erase"],
        selection=data["config"]["selection"], top_features=kw["top_features"],
        greedy_lambda=kw["greedy_lambda"], greedy_budget=kw["greedy_budget"],
        greedy_pool=kw["greedy_pool"], recall_k=kw["recall_k"],
        random_control=False, device=device, verbose=verbose)

    X = np.asarray(data["emb"], dtype=np.float32)
    feats = fitted.feats_train                     # train == the whole corpus here
    pre = encode_preactivations(fitted.sae, X)
    # Read the conditions off the fitted erasers rather than re-deriving them, so
    # the panels show exactly what the pipeline produces.
    conditions = {name: fitted.erasers[name](X) for name in PCA_CONDITIONS}
    if verbose:
        print(f"[descriptive] {data['label']}: |S|={len(fitted.selected)} latents "
              f"({fitted.selection}), dictionary {fitted.decoder.shape[0]}, "
              f"no holdout -- descriptive only")
    return {
        "corpus": corpus, "label": data["label"], "encoder": encoder, "seed": seed,
        "emb": X, "texts": data["texts"], "source": np.asarray(data["source"]),
        "pairs": data["pairs"], "paired": data["paired"],
        "sae": fitted.sae, "feats": feats, "pre": pre, "decoder": fitted.decoder,
        "scores": fitted.scores, "selected": np.asarray(fitted.selected),
        "selection": fitted.selection, "conditions": conditions,
        "config": {**fitted.config, "protocol": "none", "holdout": 0.0,
                   "descriptive_only": True},
        "sae_info": fitted.sae_info,
    }


# --------------------------------------------------------------------------- #
# 1. Reading the selected latents
# --------------------------------------------------------------------------- #
def _ovr_auc(activation: np.ndarray, labels: np.ndarray) -> dict:
    """One-vs-rest ROC-AUC of one latent's activation against each class."""
    from sklearn.metrics import roc_auc_score

    out = {}
    for c in np.unique(labels):
        try:
            out[str(c)] = float(roc_auc_score((labels == c).astype(int), activation))
        except ValueError:                       # a class with no negatives
            out[str(c)] = float("nan")
    return out


def _finite_max(d: dict) -> float:
    vals = [v for v in d.values() if np.isfinite(v)]
    return float(max(vals)) if vals else float("nan")


def latent_alignment(bundle: dict, top_m: int = 6, save: bool = True) -> dict:
    """Post-hoc: how well does each selected latent track the true confounder?

    Records, per selected latent, three label-free quantities (the MI score
    against the k-means pseudo-grouping, which top-N ranks by; pair asymmetry,
    which only the greedy selector also uses; firing frequency) and then, with
    labels, two alignment measures:

      * **firing-set purity** -- of the documents the latent fires on, the share
        belonging to each class; the maximum and the class attaining it define
        the latent's alignment. Reported with its *lift* over the class prior,
        because purity 0.5 against a 50% class is chance while 0.5 against an
        8% class is a six-fold enrichment.
      * **dense AUC** -- one-vs-rest ROC-AUC of the *pre-activation*, which is
        dense and therefore ranks every document.

    The one-vs-rest AUC of the TopK-gated activation is kept for continuity but
    is NOT an alignment measure. The gated value is exactly zero for most
    documents, and an AUC over a score that ties at zero for nearly everyone is
    capped at ``0.5 + min(1, f/p)/2`` for firing rate ``f`` and class prior
    ``p``: a perfectly class-pure latent firing on 3% of documents cannot exceed
    0.53 against a 50% class. Every selected latent sat at that ceiling on the
    binary corpora and read as "chance" until the ceiling was noticed. The
    ceiling is saved per latent so the artifact stays visible in the JSON.

    Labels enter only here, after selection is complete, and never feed back.
    `top_m` sets how many rows the printed and LaTeX tables show, ranked by
    purity; the full list is always saved.
    """
    feats, pre, sel = bundle["feats"], bundle["pre"], bundle["selected"]
    y = np.asarray(bundle["source"])
    classes = [str(c) for c in np.unique(y)]
    prior = {c: float(np.mean(y == c)) for c in classes}
    freq = (feats > 0).mean(axis=0)
    asym = pair_asymmetry(feats, bundle["pairs"]) if bundle["paired"] else None

    rows = []
    for j in sel:
        j = int(j)
        fires = feats[:, j] > 0
        n_fire = int(fires.sum())
        purity = {c: (float(np.mean(y[fires] == c)) if n_fire else float("nan"))
                  for c in classes}
        best = max(classes, key=lambda c: purity[c] if np.isfinite(purity[c]) else -1.0)
        dense = _ovr_auc(pre[:, j], y)
        gated = _ovr_auc(feats[:, j], y)
        f, p = float(freq[j]), prior[best]
        rows.append({
            "latent": j,
            "mi_score": float(bundle["scores"][j]),
            "pair_asymmetry": float(asym[j]) if asym is not None else None,
            "firing_freq": f,
            "n_firing": n_fire,
            "aligned_class": best,
            "class_prior": p,
            "purity": float(purity[best]),
            "purity_lift": float(purity[best] / p) if p > 0 else float("nan"),
            "purity_per_class": purity,
            "dense_auc": float(dense[best]),
            "max_dense_auc": _finite_max(dense),
            "dense_auc_per_class": dense,
            "max_gated_auc": _finite_max(gated),
            "gated_auc_per_class": gated,
            "gated_auc_ceiling": (0.5 + 0.5 * min(1.0, f / p)) if p > 0 else float("nan"),
        })
    rows.sort(key=lambda r: ((r["purity"] if np.isfinite(r["purity"]) else -1.0),
                             r["dense_auc"]), reverse=True)
    res = {
        "experiment": "latent_alignment", "corpus": bundle["corpus"],
        "label": bundle["label"], "encoder": bundle["encoder"],
        "seed": bundle["seed"], "selection": bundle["selection"],
        "n_selected": int(len(sel)), "classes": classes, "class_prior": prior,
        "top_m": top_m, "latents": rows, "config": bundle["config"],
    }
    if save:
        save_result(res, f"latent_alignment_{bundle['corpus']}.json")
    return res


def format_latent_alignment(res: dict, top_m: int | None = None) -> str:
    rows = res["latents"][: (top_m or res["top_m"])]
    head = (f"{'latent':>7}{'MI':>9}{'pair asym':>11}{'firing':>9}"
            f"{'purity':>9}{'lift':>7}{'dense AUC':>11}  aligned class")
    lines = [f"{res['label']} -- {len(res['latents'])} selected latents "
             f"({res['selection']}, seed {res['seed']}, no holdout)", head,
             "-" * (len(head) + 4)]
    for r in rows:
        asym = f"{r['pair_asymmetry']:.2f}" if r["pair_asymmetry"] is not None else "--"
        lines.append(f"{r['latent']:>7}{r['mi_score']:>9.4f}{asym:>11}"
                     f"{r['firing_freq']:>9.3f}{r['purity']:>9.3f}"
                     f"{r['purity_lift']:>7.1f}{r['dense_auc']:>11.3f}  "
                     f"{r['aligned_class']}")
    lines.append("(purity = share of the latent's firing set in its aligned class; "
                 "lift = purity / class prior; dense AUC = one-vs-rest AUC of the "
                 "pre-activation. Post-hoc: labels never entered scoring or selection.)")
    return "\n".join(lines)


def latent_reports(bundle: dict, top_examples: int = 8, snippet: int = 220,
                   save: bool = True) -> str:
    """The appendix report: what each selected latent fires on, and against.

    Top examples rank by the TopK-gated activation. Bottom examples rank by the
    **dense pre-activation** instead: with TopK sparsity the gated activation is
    exactly zero for most documents, so it gives no ordering among inactive ones
    and the "bottom" would be an arbitrary tie-break among zeros. The
    pre-activation surfaces the documents that most strongly anti-activate the
    latent -- the genuine other side of the confounder.
    """
    feats, pre, sel = bundle["feats"], bundle["pre"], bundle["selected"]
    texts, y = bundle["texts"], bundle["source"]
    align = {r["latent"]: r for r in latent_alignment(bundle, save=False)["latents"]}

    def esc(t):
        t = " ".join(str(t).split())
        return (t[:snippet] + "...") if len(t) > snippet else t

    info = bundle["sae_info"]
    lines = [
        f"# Selected latents -- {bundle['label']} / {bundle['encoder']}", "",
        f"SAE: expansion x{bundle['config']['expansion_factor']}, "
        f"k={bundle['config']['k']}, {bundle['config']['n_epochs']} epochs, "
        f"R2={info.get('r2', float('nan')):.3f}, "
        f"dead={info.get('frac_dead', 0):.1%}. "
        f"Selection: **{bundle['selection']}** ({len(sel)} latents), "
        f"seed={bundle['seed']}.", "",
        "Fit on the **full corpus with no holdout** -- this report is descriptive "
        "and no performance number is quoted from it.", "",
        "Purity (share of the latent's firing set in its aligned class), lift over "
        "the class prior and the dense-pre-activation AUC are computed *after* "
        "selection and never feed back into it. Bottom examples are ranked by "
        "dense pre-activations, since the TopK-gated activation is exactly zero "
        "for most documents -- the same reason its AUC is not reported here.", "",
        "| latent | MI | pair asym | firing | purity | lift | dense AUC | class |",
        "|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for j in sorted((int(x) for x in sel), key=lambda j: -align[j]["purity"]):
        r = align[j]
        a = f"{r['pair_asymmetry']:.2f}" if r["pair_asymmetry"] is not None else "--"
        lines.append(f"| {j} | {r['mi_score']:.4f} | {a} | {r['firing_freq']:.3f} "
                     f"| {r['purity']:.2f} | {r['purity_lift']:.1f} "
                     f"| {r['dense_auc']:.3f} | {r['aligned_class']} |")

    for j in sorted((int(x) for x in sel), key=lambda j: -align[j]["purity"]):
        lines += ["", f"## Latent {j}",
                  f"(MI {align[j]['mi_score']:.4f}, firing {align[j]['firing_freq']:.3f}, "
                  f"purity {align[j]['purity']:.2f} for `{align[j]['aligned_class']}`, "
                  f"lift {align[j]['purity_lift']:.1f}, dense AUC {align[j]['dense_auc']:.3f})",
                  "", "**Fires on (highest activations):**", ""]
        for i in np.argsort(feats[:, j])[::-1][:top_examples]:
            lines.append(f"- `act={feats[i, j]:.3f}` **[{y[i]}]** {esc(texts[i])}")
        lines += ["", "**Anti-activates on (lowest pre-activations):**", ""]
        for i in np.argsort(pre[:, j])[:top_examples]:
            lines.append(f"- `pre={pre[i, j]:.3f}` **[{y[i]}]** {esc(texts[i])}")

    body = "\n".join(lines)
    if save:
        save_text(body, f"latent_report_{bundle['corpus']}.md")
    return body


# --------------------------------------------------------------------------- #
# 2. Geometry of the removed subspace
# --------------------------------------------------------------------------- #
def _orth(A: np.ndarray) -> np.ndarray:
    """Orthonormal basis of col-span(A), numerically rank-truncated."""
    U, s, _ = np.linalg.svd(np.asarray(A, dtype=np.float64), full_matrices=False)
    r = int((s > 1e-8 * max(s[0], 1e-30)).sum())
    return U[:, :r]


def subspace_language_angles(embeddings, source_labels, decoder, selected,
                             feats=None, pca_r: int | None = None,
                             seed: int = 0) -> dict:
    """Principal angles between the confounder subspace and each candidate
    removed subspace. (Moved here from `paper_extras` so that this module does
    not import the legacy pipeline, which pulls in the encoder stack.)

    Confounder subspace: span of the centred cross-covariance between the
    embeddings and the one-hot labels (rank C-1) -- exactly the subspace whose
    removal LEACE's guarantee concerns. Candidates:
      * SAE selected, Sigma-weighted -- span(Sigma D_S^T): what
        `directions_leace` actually removes;
      * SAE selected, raw atoms      -- span(D_S): what `directions_orth` removes;
      * SAE random, Sigma-weighted   -- size-matched random-latent control;
      * PCA top-r                    -- top principal directions (r = |S|).

    Reports the principal angles (deg) and the capture fraction
    ||Q_E^T Q_L||_F^2 / rank(L) in [0, 1].
    """
    X = np.asarray(embeddings, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    labels = np.asarray(source_labels)
    classes, yi = np.unique(labels, return_inverse=True)
    Y = np.eye(len(classes))[yi]
    Q_L = _orth(Xc.T @ (Y - Y.mean(axis=0, keepdims=True)))
    Sigma = (Xc.T @ Xc) / len(X)

    selected = np.asarray(selected)
    D_sel = np.asarray(decoder, dtype=np.float64)[selected].T  # (d, m)
    rng = np.random.default_rng(seed)
    if feats is not None:
        alive = np.where((feats > 0).mean(axis=0) >= 0.02)[0]
        pool = alive if len(alive) >= len(selected) else np.arange(decoder.shape[0])
    else:
        pool = np.arange(decoder.shape[0])
    rand = rng.choice(pool, size=len(selected), replace=False)
    D_rand = np.asarray(decoder, dtype=np.float64)[rand].T

    r = pca_r or len(selected)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    candidates = {
        "SAE selected (Σ-weighted)": Sigma @ D_sel,
        "SAE selected (raw atoms)": D_sel,
        "SAE random (Σ-weighted)": Sigma @ D_rand,
        f"PCA top-{r}": Vt[:r].T,
    }
    out = {"classes": [str(c) for c in classes], "language_rank": Q_L.shape[1],
           "n_selected": int(len(selected)), "candidates": {}}
    for name, A in candidates.items():
        Qe = _orth(A)
        cos = np.clip(np.linalg.svd(Q_L.T @ Qe, compute_uv=False), 0, 1)
        out["candidates"][name] = {
            "angles_deg": [float(np.degrees(np.arccos(c))) for c in cos],
            "capture": float((cos ** 2).sum() / Q_L.shape[1]),
            "dim": int(Qe.shape[1]),
        }
    return out


def pca_projection(bundle: dict, conditions=PCA_CONDITIONS, save: bool = True) -> dict:
    """2-D PCA of each condition, coloured by the true class.

    Stores the projected coordinates rather than the embeddings, so the scatter
    redraws without re-running erasure or re-fitting the PCA.
    """
    from sklearn.decomposition import PCA

    y = bundle["source"]
    out = {"experiment": "pca_projection", "corpus": bundle["corpus"],
           "label": bundle["label"], "encoder": bundle["encoder"],
           "labels": [str(v) for v in y], "classes": [str(c) for c in np.unique(y)],
           "panels": {}, "config": bundle["config"]}
    for name in conditions:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(np.asarray(bundle["conditions"][name], np.float64))
        out["panels"][name] = {
            "coords": np.asarray(coords, float).round(4).tolist(),
            "explained_variance_ratio": [float(v) for v in
                                         pca.explained_variance_ratio_[:2]],
        }
    if save:
        save_result(out, f"pca_projection_{bundle['corpus']}.json")
    return out


def subspace_geometry(bundle: dict, save: bool = True) -> dict:
    """Principal angles and capture between the language subspace and four
    size-matched candidate removal subspaces.

    The candidates, all of dimension ``|S|``:
      (i)   the covariance-weighted selected directions ``span(Sigma D_S^T)`` --
            what ``directions_leace`` actually removes;
      (ii)  the same decoder atoms unweighted, ``span(D_S)`` -- their Euclidean
            geometry, i.e. what ``directions_orth`` removes;
      (iii) covariance-weighted *randomly* selected latents -- the size-matched
            control that separates the selection signal from the dimension count;
      (iv)  the top ``|S|`` principal components of the raw embeddings.
    """
    res = subspace_language_angles(
        bundle["emb"], bundle["source"], bundle["decoder"], bundle["selected"],
        feats=bundle["feats"], seed=bundle["seed"])
    # `subspace_language_angles` names its key "language_rank"; the same
    # construction is the *source* subspace on the unpaired corpora, so carry the
    # corpus's own confounder name for anything that prints it.
    import corpora
    res.update({"experiment": "subspace_geometry", "corpus": bundle["corpus"],
                "label": bundle["label"], "encoder": bundle["encoder"],
                "seed": bundle["seed"], "config": bundle["config"],
                "confounder": corpora.CORPORA[bundle["corpus"]]["confounder"],
                "subspace_rank": int(res["language_rank"])})
    if save:
        save_result(res, f"subspace_geometry_{bundle['corpus']}.json")
    return res


def format_subspace_geometry(res: dict) -> str:
    w = max(len(n) for n in res["candidates"]) + 2
    conf = res.get("confounder", "confounder")
    rank = res.get("subspace_rank", res["language_rank"])
    lines = [f"{res.get('label', res.get('corpus'))}: {conf} subspace rank "
             f"{rank} (classes {', '.join(res['classes'])}), "
             f"|S| = {res['n_selected']}",
             f"{'candidate subspace':<{w}}{'dim':>5}{'capture':>10}   "
             f"principal angles (deg)",
             "-" * (w + 45)]
    for name, c in res["candidates"].items():
        angles = ", ".join(f"{a:.1f}" for a in c["angles_deg"][:6])
        lines.append(f"{name:<{w}}{c['dim']:>5}{c['capture']:>10.3f}   {angles}")
    lines.append(f"(capture = mean squared cosine of the principal angles from "
                 f"the {conf} subspace; 1.0 = fully contained, 0.0 = orthogonal)")
    return "\n".join(lines)
