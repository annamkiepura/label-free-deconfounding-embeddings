"""Where the experiments read and write, relative to the repository root (the notebooks run from there).

    results/main/        the core experiments (notebooks/02_main_experiments.ipynb)
    results/fedlex/      Fedlex: dataset record, primary run, ten-encoder study (notebooks/03_fedlex.ipynb)
    results/extensions/  Table 3's runs, SAE construction, the three-class encoder study, displacement-matched
                         transfer and the seeds 3-4 extension with the five-seed merges (notebooks/04_extensions.ipynb)
    results/selectors/   the k-means-free latent selectors of Table 3 (notebooks/05_latent_selectors.ipynb)

Each results folder has the layout `artifacts` writes: ``<root>/raw/<experiment>/<file>.json``. Embeddings and the
frozen maps of the transfer runs are cached under ``embedding_cache/`` (not in git; rebuilt by the notebooks). The
maps are filed under the basename of the results folder that saved them, so the folders' basenames are distinct.
"""
import os

RESULTS = "results"
MAIN = os.path.join(RESULTS, "main")
FEDLEX = os.path.join(RESULTS, "fedlex")
EXTENSIONS = os.path.join(RESULTS, "extensions")
SELECTORS = os.path.join(RESULTS, "selectors")

EMBED_CACHE = os.path.join("embedding_cache", "corpora")
FEDLEX_CACHE = os.path.join("embedding_cache", "fedlex")

QUICK = False


def quick(root: str) -> str:
    """The results folder of a QUICK smoke run (``results/main`` -> ``results/main_quick``): never mixed with the
    real results, and its saved maps never stand in for the real ones."""
    return os.path.normpath(root) + "_quick"


def configure(quick_run: bool = False) -> None:
    """Point every results folder at its QUICK twin (or back). Call it before importing the experiment modules:
    some of them read these folders into module constants at import time."""
    global MAIN, FEDLEX, EXTENSIONS, SELECTORS, QUICK
    QUICK = bool(quick_run)
    base = {"MAIN": "main", "FEDLEX": "fedlex", "EXTENSIONS": "extensions", "SELECTORS": "selectors"}
    for name, sub in base.items():
        root = os.path.join(RESULTS, sub)
        globals()[name] = quick(root) if QUICK else root
