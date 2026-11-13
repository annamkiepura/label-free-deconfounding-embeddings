"""One method-name and style mapping for every exhibit.

Canonical method keys are the saved condition names of the ID runs (run_id /
the OOD runs), with the coverage run's aliases resolved by the manifest
(`sae_mi` -> `sae_leace`, `kmeans_leace` -> `kmeans_auto`). Colours are the
palette of the revised-results figures (code/revised_exhibits.py: COLOR, MARKER,
MECH_COLOR), so both figure sets read the same way; every method also has its own
marker and line style so figures survive greyscale printing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    key: str
    label: str            # manuscript name (LaTeX-safe)
    short: str            # narrow table header
    color: str
    marker: str
    linestyle: str = "-"
    uses_labels: bool = False
    family: str = ""


METHODS: dict[str, Method] = {m.key: m for m in [
    Method("raw", "Raw", "Raw", "#FB9A99", "o", ":", family="reference"),
    Method("pca_top1", "PC-1", "PC-1", "#9467BD", "s", "-", family="pca"),
    Method("kmeans_auto", "$k$-means LEACE", "$k$-means", "#FF7F00", "P", "-", family="kmeans"),
    Method("sae_leace", "SAE-LEACE", "SAE-LEACE", "#1F78B4", "o", "-", family="sae"),
    Method("metadata_leace", "Metadata-LEACE", "Metadata", "#33A02C", "D", "--", uses_labels=True,
           family="reference"),
    Method("sae_random", "SAE-LEACE, random latents", "SAE rand.", "#8C510A", "^", "--", family="sae"),
    Method("pca_matched_var", "PCA, variance-matched", "PCA var.", "#56B4E9", "P", "-", family="pca"),
    Method("sae_recon", "SAE decode", "Decode", "#8C564B", "X", "-", family="sae"),
    Method("sae_recon_suppressed", "SAE decode, latents zeroed", "Decode-0", "#8C564B", "x", "--",
           family="sae"),
    Method("activations", "Activation-LEACE", "Act.-LEACE", "#FF7F00", "<", "-", family="mechanism"),
    Method("directions_orth", "Directions, orthogonal", "Dir.-orth.", "#6A3D9A", ">", "-",
           family="mechanism"),
    Method("directions_leace", "Directions-LEACE (= SAE-LEACE)", "Dir.-LEACE", "#1F78B4", "o", "-",
           family="mechanism"),
    Method("random_nonsae_directions", "Random directions, no SAE", "Rand. dir.", "#999999", "*",
           "-.", family="pending"),
]}

#: The comparison set of the headline, composition and transfer exhibits.
CORE = ("raw", "pca_top1", "kmeans_auto", "sae_leace", "metadata_leace")
#: Label-free erasers whose gap to Metadata-LEACE the geometry exhibit plots.
LABEL_FREE = ("pca_top1", "kmeans_auto", "sae_leace")

METRICS = {
    "linear_auc": ("Linear AUC", "lower"),
    "gb_auc": ("GB AUC", "lower"),
    "mlp_auc": ("MLP AUC", "lower"),
    "recall@1": ("R@1", "higher"),
    "recall@10": ("R@10", "higher"),
    "topic_purity": ("Topic purity", "higher"),
    "topic_ari": ("Topic ARI", "higher"),
    "gold_sim_spearman": (r"Gold sim.\ $\rho$", "higher"),
}

ENCODERS = {"e5-small": "E5-small", "mini": "MiniLM", "e5-base": "E5-base", "e5-large": "E5-large",
            "gist-small": "GIST-small", "gist-base": "GIST-base", "gist-large": "GIST-large",
            "mpnet": "MPNet", "nomic-v2": "Nomic-v2", "mxb": "MXB-large"}
ENCODER_ORDER = ("e5-small", "e5-base", "e5-large", "gist-small", "gist-base", "gist-large",
                 "mini", "mpnet", "nomic-v2", "mxb")

CLASS_LABEL = {"de": "DE", "fr": "FR", "it": "IT", "en": "EN", "es": "ES", "zh": "ZH",
               "oyez": "Oyez", "lexis": "Lexis", "wiki": "Wikipedia",
               "bill": "Bills", "tweet": "Tweets", "order": "Orders", "newspaper": "News"}


def method(key: str) -> Method:
    if key in METHODS:
        return METHODS[key]
    m = re.fullmatch(r"pca_top(\d+)", key)
    if m:
        return Method(key, f"PCA-{m[1]}", f"PCA-{m[1]}", METHODS["pca_top1"].color, "s", family="pca")
    m = re.fullmatch(r"sae_(mi|random)_n(\d+)", key)
    if m:
        base = METHODS["sae_leace" if m[1] == "mi" else "sae_random"]
        return Method(key, f"{base.label} ($N{{=}}{m[2]}$)", f"{base.short} {m[2]}", base.color,
                      base.marker, base.linestyle, family="sae")
    m = re.fullmatch(r"kmeans_k(\d+)", key)
    if m:
        return Method(key, f"$k$-means LEACE ($K{{=}}{m[1]}$)", f"KM-{m[1]}",
                      METHODS["kmeans_auto"].color, "P", family="kmeans")
    raise KeyError(f"no style for method {key!r}")


def pair_label(kind: str) -> str:
    """``"lexis|oyez"`` -> ``"Lexis--Oyez"``."""
    return "--".join(CLASS_LABEL.get(c, c) for c in kind.split("|"))


def metric_label(metric: str) -> str:
    return METRICS[metric][0]


def direction(metric: str) -> str:
    return METRICS[metric][1]
