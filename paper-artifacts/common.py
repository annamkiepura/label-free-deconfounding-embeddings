"""Shared pieces of the build: the build profile, readers and table helpers.

Everything the build writes goes under ``paper-artifacts/`` (tables/, figures/, data/, derived/, reports/).
`profile` points the reporting library (``paper-artifacts/paper_assets/``) at this folder's manifest and output
directories for the duration of a build.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                # paper-artifacts/
REPO = HERE.parent
if str(HERE) not in sys.path:                         # the reporting library, paper-artifacts/paper_assets/
    sys.path.insert(0, str(HERE))

from paper_assets.lib import config, io, latex, loaders, provenance      # noqa: E402
from paper_assets.lib.aggregate import SeedValues                         # noqa: E402
from paper_assets.lib.provenance import Exhibit, check, current           # noqa: E402

REL = HERE.relative_to(REPO).as_posix()               # "paper-artifacts"
COMMAND = f"python {REL}/build.py"
MAIN = "results/main/raw"                              # notebooks/02_main_experiments.ipynb
FIVE = "results/extensions/raw/five_seeds"             # merged seeds 0-4 (focused_extensions.merge_five_seeds)
EXT = "results/extensions/extensions_seeds_3_4/raw"    # the raw seeds 3-4
FOCUSED = "results/extensions/raw"                     # notebooks/04_extensions.ipynb
SELECTORS = "results/selectors/raw/selectors"          # notebooks/05_latent_selectors.ipynb
FEDLEX = "results/fedlex/raw"                          # notebooks/03_fedlex.ipynb
DERIVED = HERE / "derived"
SEEDS5 = [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# The build profile
# --------------------------------------------------------------------------- #
def _reset() -> None:
    config.manifest.cache_clear()
    io.clear_cache()
    loaders._VALIDATED.clear()
    loaders.Cohort._instances.clear()


@contextlib.contextmanager
def profile():
    """Route the paper_assets library to this folder: its manifest (five-seed cohorts), its output
    directories and its five-seed random-latent files."""
    saved = (config.MANIFEST_PATH, dict(config.OUT_DIRS), loaders.RANDOM_BAND_FILE)
    config.MANIFEST_PATH = HERE / "manifest.toml"
    config.OUT_DIRS.update({k: HERE / k for k in ("figures", "tables", "data", "reports")})
    loaders.RANDOM_BAND_FILE = f"{REL}/derived/random_band/random_band_{{setting}}_{{encoder}}.json"
    _reset()
    config.ensure_dirs()
    try:
        yield
    finally:
        config.MANIFEST_PATH = saved[0]
        config.OUT_DIRS.update(saved[1])
        loaders.RANDOM_BAND_FILE = saved[2]
        _reset()


# --------------------------------------------------------------------------- #
# Reading saved results outside the manifest's roots, with provenance
# --------------------------------------------------------------------------- #
_READ: dict = {}


def read(rel: str) -> dict:
    """A saved result by repository-relative path, registered with the active exhibit (path, SHA-256,
    run ID), as `paper_assets.lib.io.read` registers the manifest's files."""
    if rel not in _READ:
        path = REPO / rel
        if not path.exists():
            raise FileNotFoundError(f"{rel} not found")
        raw = path.read_bytes()
        _READ[rel] = (json.loads(raw), hashlib.sha256(raw).hexdigest(), len(raw))
    data, sha, size = _READ[rel]
    ctx = current()
    if ctx is not None:
        ctx.add_source(rel, sha, size, data)
    return data


def source(rel: str, condition: str, metric: str) -> dict:
    read(rel)
    return {"file": rel, "run_id": io.run_id(rel, _READ[rel][1]), "condition": condition,
            "path": f"per_seed[*].metrics.{condition}.{metric}"}


def seed_values(rel: str, setting: str, encoder: str, condition: str, metric: str, *, cohort: str,
                seeds=None, method: str | None = None, sub: str | None = None) -> SeedValues:
    """One saved condition's per-seed values of `metric` (``sub`` picks a key of a dict metric); a
    cell that the run does not have is returned as not applicable, never estimated."""
    d = read(rel)
    per = [p for p in d["per_seed"] if seeds is None or int(p["seed"]) in seeds]
    vals = []
    for p in per:
        v = p["metrics"].get(condition, {}).get(metric)
        if isinstance(v, dict):
            v = v.get(sub) if sub else None
        vals.append(v)
    if not per or any(v is None for v in vals):
        return SeedValues(setting, encoder, method or condition, metric, cohort=cohort, status="not_applicable",
                          reason=f"{rel}: {condition}.{metric} not recorded in every seed")
    return SeedValues(setting, encoder, method or condition, metric, [int(p["seed"]) for p in per],
                      [float(v) for v in vals], cohort=cohort, sources=[source(rel, condition, metric)]).register()


def derived(values, setting: str, encoder: str, method: str, metric: str, seeds, *, cohort: str, sources) -> SeedValues:
    """Per-seed values computed here from saved per-seed values (a paired difference, for instance)."""
    return SeedValues(setting, encoder, method, metric, [int(s) for s in seeds], [float(v) for v in values],
                      cohort=cohort, sources=list(sources)).register()


def paired(a: SeedValues, b: SeedValues, method: str | None = None) -> SeedValues:
    """a - b per matched seed (both must hold the same seeds)."""
    if not (a.ok and b.ok):
        return SeedValues(a.setting, a.encoder, method or f"{a.method}-{b.method}", a.metric, cohort=a.cohort,
                          status="not_applicable", reason="an operand is not available")
    check(list(a.seeds) == list(b.seeds), "paired_same_seeds",
          f"{a.setting}/{a.encoder} {a.metric}: {a.method} and {b.method} are paired on seeds {list(a.seeds)}")
    return derived([x - y for x, y in zip(a.values, b.values)], a.setting, a.encoder,
                   method or f"{a.method}-{b.method}", a.metric, a.seeds, cohort=a.cohort, sources=a.sources + b.sources)


def t_interval(sv: SeedValues, level: float = 0.95):
    """Two-sided t-interval of the seed mean (df = n - 1); None with fewer than two seeds."""
    from scipy import stats
    if not sv.ok or sv.n < 2:
        return None
    half = float(stats.t.ppf((1 + level) / 2, sv.n - 1)) * sv.sd / sv.n ** 0.5
    return sv.mean - half, sv.mean + half


# --------------------------------------------------------------------------- #
# Writing: tables, figures, and the paths of this folder
# --------------------------------------------------------------------------- #
def _fix_paths(path: Path, name: str) -> None:
    """The library writes include paths and rebuild commands for paper_assets/; point them here."""
    text = path.read_text()
    for old, new in ((f"python -m paper_assets build {name}", f"{COMMAND} {name}"),
                     ("paper_assets/tables/", f"{REL}/tables/"), ("paper_assets/figures/", f"{REL}/figures/"),
                     ("paper_assets/data/", f"{REL}/data/")):
        text = text.replace(old, new)
    path.write_text(text)


def fix_outputs(name: str, kind: str) -> None:
    folder = HERE / ("tables" if kind == "table" else "figures")
    for p in (folder / f"{name}.tex", folder / f"{name}.include.tex"):
        if p.exists():
            _fix_paths(p, name)


def rename_outputs(old: str, new: str, kind: str, ex) -> None:
    """Give an exhibit's outputs the manuscript's file name: its LaTeX, include file and data CSVs. The old name is
    replaced in the include file and in the LaTeX comment lines (the table itself never names its file)."""
    folder = HERE / ("tables" if kind == "table" else "figures")
    data = HERE / "data"
    files = [folder / f"{old}{ext}" for ext in (".tex", ".include.tex", ".pdf", ".png")]
    # A new name that extends the old one (tableB1_extended_id -> tableB1_extended_id_block) must not catch the
    # renamed files of an earlier build.
    files += [data / f"{old}.csv"] + sorted(q for q in data.glob(f"{old}_*.csv") if not q.name.startswith(new))
    moved = {}
    for p in files:
        if p.exists():
            q = p.with_name(new + p.name[len(old):])
            p.replace(q)
            moved[p] = q
    for q in moved.values():
        if q.name.endswith(".include.tex"):
            q.write_text(q.read_text().replace(old, new))
        elif q.suffix == ".tex":
            q.write_text("\n".join(l.replace(old, new) if l.lstrip().startswith("%") else l
                                   for l in q.read_text().split("\n")))
    rel = {str(p.relative_to(HERE)): str(q.relative_to(HERE)) for p, q in moved.items()}
    ex.outputs[:] = [rel.get(o, o) for o in ex.outputs]
    ex.name = new                                     # the provenance file is named after the exhibit
    ex.note(f"Outputs renamed from {old} to {new} (the manuscript's file name).")


def patch_caption(name: str, kind: str, replacements, ex) -> None:
    """Replace caption wording that the original exhibit states for its three-seed cohorts. Every
    old string must be present exactly once, so a change in the exhibit's text cannot pass silently."""
    folder = HERE / ("tables" if kind == "table" else "figures")
    target = folder / (f"{name}.tex" if kind == "table" else f"{name}.include.tex")
    text = target.read_text()
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError(f"{name}: expected exactly one occurrence of {old!r} in {target.name}, found {text.count(old)}")
        text = text.replace(old, new)
        ex.caption = ex.caption.replace(old, new)
    target.write_text(text)


def write_table(name: str, lines: list, caption: str, label: str, ex, **kw) -> Path:
    p = latex.write_table(name, lines, caption, label, ex, **kw)
    fix_outputs(name, "table")
    return p


def preview(name: str, ex, stubs: dict | None = None) -> None:
    """Compile a table on the TACL page (width and float checks). `stubs` gives the number shown for
    labels that a standalone preview cannot resolve; by default the manuscript's section labels and
    this update's own table labels, except those the table defines itself."""
    import re
    tex = (HERE / "tables" / f"{name}.tex").read_text()
    if stubs is None:
        own = set(re.findall(r"\\label\{([^}]+)\}", tex))
        stubs = {k: v for k, v in {**MANUSCRIPT_LABELS, **UPDATE_LABELS}.items() if k not in own}
    pre = ("\\makeatletter\n" + "".join(f"\\@namedef{{r@{k}}}{{{{{v}}}{{}}}}\n" for k, v in stubs.items())
           + "\\makeatother\n") if stubs else ""
    if name in FULL_WIDTH:
        pre += "\\onecolumn\n"
    latex.compile_preview(name, pre + tex, ex)


#: Non-floating tables that the manuscript typesets in a one-column region: previewed at the full text width.
FULL_WIDTH = {"tableB2_ten_encoder_robustness_singlepage"}


#: Section labels of the manuscript that captions refer to (the manuscript resolves the real ones).
MANUSCRIPT_LABELS = {"sec:appendix_A3": "A.3", "sec:appendix_A4": "A.4"}
#: Labels of this update's own exhibits that other captions refer to, as the number a standalone preview shows.
UPDATE_LABELS = {"tab:erasure-extent": "4", "tab:mechanism-controls": "3"}


def finish(ex, name: str) -> dict:
    """Write the exhibit's provenance and return a one-line status for the build log."""
    ex.write(f"{COMMAND} {name}")
    errors = [i for i in ex.issues if i["level"] in ("error", "missing")]
    return {"name": name, "checks": len(ex.checks), "errors": errors,
            "warnings": [i for i in ex.issues if i["level"] == "warning"],
            "outputs": [o for o in ex.outputs if not o.endswith(".provenance.json")]}


def setting_label(key: str) -> str:
    return {"swiltra_de_fr_it": "SwiLTra (DE/FR/IT)", "scotus_oyez_lexis_wiki": "SCOTUS (3 sources)",
            "semeval": "SemEval", "cap": "CAP", "fedlex_domestic": "Fedlex (DE/FR/IT)",
            "bills_tweets": "Bills--Tweets", "swiltra": "SwiLTra DE--FR", "swiltra_de_it": "SwiLTra DE--IT",
            "swiltra_fr_it": "SwiLTra FR--IT", "scotus_oyez_lexis": "SCOTUS Oyez--Lexis",
            "scotus_oyez_wiki": "SCOTUS Oyez--Wiki", "scotus_wiki_lexis": "SCOTUS Lexis--Wiki"}[key]
