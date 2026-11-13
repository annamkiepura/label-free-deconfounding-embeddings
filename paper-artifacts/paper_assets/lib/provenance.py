"""Per-exhibit provenance context and the shared validation-issue log.

An exhibit build runs inside ``with Exhibit(name) as ex:``. Everything the
loaders read is registered on it (source files, SHA-256, run IDs, dataset
versions, seeds, cohorts); checks and issues are recorded on it; at the end it
writes ``data/<name>.provenance.json``. Issues are also appended to a global
log that `validate` turns into ``reports/validation_report.{md,json}``.
"""
from __future__ import annotations

import contextvars
import datetime as _dt
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict

from . import config

_CURRENT: contextvars.ContextVar = contextvars.ContextVar("exhibit", default=None)
ISSUES: list[dict] = []          # every issue raised in this process, all exhibits
CHECKS: list[dict] = []          # every passed check, all exhibits

LEVELS = ("error", "missing", "warning", "info")


def current():
    return _CURRENT.get()


def issue(level: str, code: str, message: str, **detail) -> dict:
    """Record a validation issue (in the active exhibit and globally)."""
    assert level in LEVELS, level
    ctx = current()
    rec = {"level": level, "code": code, "message": message,
           "exhibit": ctx.name if ctx else None, **detail}
    if rec not in ISSUES:
        ISSUES.append(rec)
    if ctx is not None and rec not in ctx.issues:
        ctx.issues.append(rec)
    return rec


def check(ok: bool, code: str, message: str, **detail) -> bool:
    """A consistency check: passes silently into the log, or records an error."""
    ctx = current()
    if ok:
        rec = {"code": code, "message": message, "exhibit": ctx.name if ctx else None}
        CHECKS.append(rec)
        if ctx is not None:
            ctx.checks.append(rec)
    else:
        issue("error", code, message, **detail)
    return ok


def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=config.REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                   # noqa: BLE001
        return None


def environment() -> dict:
    import matplotlib
    import numpy
    return {"python": sys.version.split()[0], "numpy": numpy.__version__,
            "matplotlib": matplotlib.__version__, "platform": platform.platform(),
            "git_head": _git("rev-parse", "HEAD"),
            "git_dirty_paths": len((_git("status", "--porcelain") or "").splitlines())}


@dataclass
class Exhibit:
    name: str
    kind: str                     # "table" or "figure"
    title: str = ""
    sources: dict = field(default_factory=dict)
    cells: list = field(default_factory=list)       # SeedValues provenance records
    checks: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    caption: str = ""

    def __enter__(self):
        self._token = _CURRENT.set(self)
        return self

    def __exit__(self, *exc):
        _CURRENT.reset(self._token)
        return False

    # -- registration -------------------------------------------------------- #
    def add_source(self, rel: str, sha: str, size: int, data: dict) -> None:
        if rel in self.sources:
            return
        from .io import run_id
        meta = {k: data.get(k) for k in ("experiment", "corpus", "encoder", "donor", "seeds",
                                         "label", "donor_label", "fit_on")
                if data.get(k) is not None}
        if "merged" in data:
            meta["merged_note"] = data["merged"]
        self.sources[rel] = {"sha256": sha, "bytes": size, "run_id": run_id(rel, sha), **meta}

    def add_cell(self, record: dict) -> None:
        self.cells.append(record)

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def output(self, path) -> None:
        rel = (str(path.relative_to(config.HOME)) if hasattr(path, "is_relative_to")
               and path.is_relative_to(config.HOME) else str(path))
        if rel not in self.outputs:
            self.outputs.append(rel)

    # -- summary ----------------------------------------------------------- #
    def cohorts(self) -> dict:
        out: dict = {}
        for c in self.cells:
            key = f"{c['setting']}/{c['encoder']}"
            ent = out.setdefault(key, {"cohorts": set(), "seeds": set(), "dataset_version": c.get("dataset_version")})
            ent["cohorts"].add(c.get("cohort"))
            ent["seeds"].add(tuple(c["seeds"]))
        return {k: {"cohorts": sorted(x for x in v["cohorts"] if x),
                    "seed_sets": [list(s) for s in sorted(v["seeds"])],
                    "dataset_version": v["dataset_version"]} for k, v in sorted(out.items())}

    def write(self, build_command: str) -> None:
        conv = config.manifest()["conventions"]
        doc = {"exhibit": self.name, "kind": self.kind, "title": self.title,
               "build_command": build_command,
               "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
               "results_root": str(config.results_root().relative_to(config.REPO))
               if config.results_root().is_relative_to(config.REPO) else str(config.results_root()),
               "manifest_sha256": __import__("hashlib").sha256(config.MANIFEST_PATH.read_bytes()).hexdigest(),
               "conventions": conv, "caption": self.caption,
               "settings_and_seeds": self.cohorts(),
               "sources": dict(sorted(self.sources.items())),
               "notes": self.notes, "checks_passed": len(self.checks),
               "checks": self.checks, "issues": self.issues,
               "cells": self.cells, "outputs": sorted(self.outputs),
               "environment": environment()}
        path = config.OUT_DIRS["data"] / f"{self.name}.provenance.json"
        path.write_text(json.dumps(doc, indent=1, default=_json_default) + "\n")
        self.output(path)


def _json_default(o):
    if isinstance(o, (set, tuple)):
        return list(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    raise TypeError(type(o))
