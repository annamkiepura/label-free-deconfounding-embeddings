"""Shared artifact I/O: results are filed by paper section, raw results first.

Every JSON, figure, table and report a run produces is written through this
module, so the output tree always looks the same:

    results/
      A/  raw/  figures/  tables/     # Sec. A -- does metadata-free erasure work?
      B/  raw/  figures/  tables/     # Sec. B -- which components produce it?
      C/  raw/  figures/  tables/     # Sec. C -- how far does it generalise?
      D/  raw/  figures/  tables/     # Sec. D -- what exactly is removed?

A notebook declares which section it is producing, once, at the top:

    import artifacts
    artifacts.set_section("A")        # -> results/A/{raw,figures,tables}

after which `save_result` lands in ``raw/``, `savefig` in ``figures/`` and
`save_table` in ``tables/`` without any call site having to know the layout.

**Raw results are the primary artifact; figures and tables are provisional.**
Each analysis saves a self-contained JSON holding everything its figure needs,
so a plot can be redrawn -- restyled, recolored, recaptioned -- without
repeating the expensive compute (SAE training, greedy selection, LEACE fits).

Call `artifact_path()` / `save_result()` rather than importing `RESULTS_DIR`
directly: the directory is a module global that `set_section` rebinds, so a
`from artifacts import RESULTS_DIR` would capture a stale copy.
"""

from __future__ import annotations

import json
import os

import numpy as np

#: Root of the results tree.
BASE_DIR = "results"
#: The four paper sections. `set_section` accepts these (case-insensitive).
SECTIONS = ("A", "B", "C", "D")
#: Sub-directory per artifact kind, inside ``<BASE_DIR>/<section>/``.
KIND_DIRS = {"raw": "raw", "figure": "figures", "table": "tables"}

#: Active section letter, or None to write to ``<BASE_DIR>/{raw,figures,tables}``.
SECTION: str | None = None
#: Active experiment name, or None. Nests INSIDE each kind directory, so one
#: experiment's raw JSON, figures and tables sit in matching sub-folders:
#: ``results/A/raw/<experiment>/`` and ``results/A/figures/<experiment>/``.
EXPERIMENT: str | None = None
#: Effective section directory: ``BASE_DIR`` or ``BASE_DIR/<section>``.
RESULTS_DIR = BASE_DIR


def _refresh(create: bool = False) -> str:
    """Recompute RESULTS_DIR. Directories are created on first write (see
    `artifact_path`), so merely pointing at a section leaves no empty folders
    behind; `create=True` materialises the section's three sub-directories."""
    global RESULTS_DIR
    RESULTS_DIR = BASE_DIR if not SECTION else os.path.join(BASE_DIR, SECTION)
    if create:
        for sub in KIND_DIRS.values():
            os.makedirs(os.path.join(RESULTS_DIR, sub), exist_ok=True)
    return RESULTS_DIR


def set_results_dir(path: str) -> str:
    """Set the ROOT of the results tree (default ``results``).

    Rarely needed -- change it only to write a run somewhere other than the
    repository's `results/` directory. The section layout is unaffected.
    """
    global BASE_DIR
    BASE_DIR = path
    _refresh(create=bool(SECTION))
    print(f"[artifacts] results root: {os.path.abspath(BASE_DIR)}")
    return RESULTS_DIR


def set_section(letter: str | None) -> str:
    """Route every subsequent artifact into ``results/<letter>/``.

    ``set_section(None)`` writes to the root's own raw/figures/tables instead,
    which is the right choice for scratch runs that do not belong to a section.
    """
    global SECTION
    if letter is not None:
        letter = str(letter).strip().upper()
        if letter not in SECTIONS:
            raise ValueError(
                f"unknown section {letter!r}; expected one of {', '.join(SECTIONS)} "
                "(or None for the results root)")
    SECTION = letter
    _refresh(create=letter is not None)
    where = f"section {SECTION}" if SECTION else "results root"
    print(f"[artifacts] writing {where}: {os.path.abspath(RESULTS_DIR)}"
          f"/{{{','.join(KIND_DIRS.values())}}}")
    return RESULTS_DIR


class section:
    """Context manager scoping artifacts to one paper section.

        with artifacts.section("D"):
            geometry_figures(prep)      # -> results/D/{raw,figures}/...

    The previous section is restored on exit, and scopes never compound: an
    inner scope is still ``results/<inner>``, never ``results/<outer>/<inner>``.
    """

    def __init__(self, letter: str | None):
        self.letter, self._prev = letter, None

    def __enter__(self):
        self._prev = SECTION
        return set_section(self.letter)

    def __exit__(self, *exc):
        set_section(self._prev)
        return False


class experiment:
    """Context manager nesting artifacts in one experiment's sub-folder.

        with artifacts.experiment("headline"):
            run_headline(...)      # -> results/A/raw/headline/...
                                   #    results/A/figures/headline/...

    The nesting happens INSIDE each kind directory, so a single experiment's
    raw JSON and its figures stay side by side under matching paths -- which is
    what makes "redraw every figure from raw" a per-folder operation. Names may
    contain ``/`` to nest further (``"selection/semeval"``).

    Scopes do not compound: an inner scope replaces the outer one rather than
    appending to it, and the previous scope is restored on exit.
    """

    def __init__(self, name: str | None = None):
        self.name, self._prev = name, None

    def __enter__(self):
        self._prev = EXPERIMENT
        return set_experiment(self.name)

    def __exit__(self, *exc):
        set_experiment(self._prev)
        return False


def set_experiment(name: str | None) -> str:
    """Route subsequent artifacts into ``<kind>/<name>/``. ``None`` clears it."""
    global EXPERIMENT
    EXPERIMENT = str(name).strip("/") if name else None
    return RESULTS_DIR


def get_results_dir() -> str:
    """The active section directory (the parent of raw/, figures/, tables/)."""
    return RESULTS_DIR


def _kind_dir(kind: str, experiment: bool = True) -> str:
    d = os.path.join(RESULTS_DIR, KIND_DIRS[kind])
    return os.path.join(d, EXPERIMENT) if (experiment and EXPERIMENT) else d


def raw_dir(experiment: bool = True) -> str:
    """The active raw/ directory. ``experiment=False`` gives the section root."""
    return _kind_dir("raw", experiment)


def figures_dir(experiment: bool = True) -> str:
    return _kind_dir("figure", experiment)


def tables_dir(experiment: bool = True) -> str:
    return _kind_dir("table", experiment)


def artifact_path(name: str, kind: str = "raw") -> str:
    """Path for `name` inside the active section's `kind` sub-directory.

    `kind` is one of ``raw`` (JSON results, markdown reports), ``figure``
    (PNG/PDF) or ``table`` (LaTeX). A name that already contains a directory
    separator is passed through untouched, so a caller can still write
    somewhere specific on purpose.
    """
    if kind not in KIND_DIRS:
        raise ValueError(f"unknown artifact kind {kind!r}; expected one of "
                         f"{', '.join(KIND_DIRS)}")
    if os.path.dirname(name):
        os.makedirs(os.path.dirname(name), exist_ok=True)
        return name
    d = os.path.join(RESULTS_DIR, KIND_DIRS[kind])
    if EXPERIMENT:
        d = os.path.join(d, EXPERIMENT)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(f"not JSON-serialisable: {type(o)}")


def save_result(obj, name: str) -> str:
    """Write a result dict to the section's ``raw/`` directory (numpy-safe)."""
    p = artifact_path(name if name.endswith(".json") else name + ".json", kind="raw")
    with open(p, "w") as fh:
        json.dump(obj, fh, default=_json_default, indent=1)
    print(f"[saved raw] {p}")
    return p


def save_table(body: str, name: str) -> str:
    """Write a LaTeX table to the section's ``tables/`` directory."""
    p = artifact_path(name if name.endswith(".tex") else name + ".tex", kind="table")
    with open(p, "w") as fh:
        fh.write(body)
    print(f"[saved table] {p}")
    return p


def save_text(body: str, name: str) -> str:
    """Write a plain-text/markdown artifact (e.g. a latent report) to ``raw/``."""
    p = artifact_path(name, kind="raw")
    with open(p, "w") as fh:
        fh.write(body)
    print(f"[saved raw] {p}")
    return p


def load_result(name_or_path):
    """Load a saved result. Accepts a dict (passed through), a bare file name
    resolved inside the section's ``raw/`` directory, or an explicit path -- so
    plotting functions can be called on either live results or a saved JSON."""
    if isinstance(name_or_path, dict):
        return name_or_path
    if os.path.exists(str(name_or_path)):
        p = name_or_path
    else:
        stem = str(name_or_path)
        p = artifact_path(stem if stem.endswith(".json") else f"{stem}.json", kind="raw")
    with open(p) as fh:
        return json.load(fh)


def savefig(fig, stem: str, show: bool = False, dpi: int = 200) -> str:
    """Save a figure as PNG+PDF in the section's ``figures/`` directory."""
    import matplotlib.pyplot as plt

    base = artifact_path(stem, kind="figure")
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=dpi, bbox_inches="tight")
    print(f"[saved fig] {base}.png/.pdf")
    if show:
        plt.show()
    plt.close(fig)
    return base


_refresh()
