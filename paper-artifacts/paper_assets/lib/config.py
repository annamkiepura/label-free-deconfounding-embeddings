"""Paths and the result-selection manifest."""
from __future__ import annotations

import functools
import os
import tomllib
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]          # paper-artifacts/paper_assets/
HOME = PKG.parent                                  # paper-artifacts/
REPO = HOME.parent                                 # repository root
MANIFEST_PATH = HOME / "manifest.toml"

OUT_DIRS = {"figures": HOME / "figures", "tables": HOME / "tables",
            "data": HOME / "data", "reports": HOME / "reports"}

#: Manuscript geometry measured from tacl2021v1.sty (11pt, a4paper, two-column):
#: \columnwidth = 219.08612pt, \textwidth = 455.24408pt (1in = 72.27pt).
COLUMN_WIDTH_IN = 219.08612 / 72.27
TEXT_WIDTH_IN = 455.24408 / 72.27
#: \textheight of the same page (702.8pt, as measured with tacl2021v1.sty in the B1/B2 reports). Used,
#: with the widths above, only by the stand-in page when the style file is not available.
TEXT_HEIGHT_PT = 702.8


@functools.lru_cache(maxsize=1)
def manifest() -> dict:
    with open(MANIFEST_PATH, "rb") as f:
        return tomllib.load(f)


def results_root() -> Path:
    override = os.environ.get("PAPER_ASSETS_RESULTS")
    return Path(override).resolve() if override else (REPO / manifest()["results_root"])


def setting(key: str) -> dict:
    s = manifest()["settings"].get(key)
    if s is None:
        raise KeyError(f"setting {key!r} is not declared in manifest.toml")
    return {"key": key, **s}


def settings(role: str | None = None) -> list[str]:
    return [k for k, v in manifest()["settings"].items() if role is None or v["role"] == role]


def cohort_for(setting_key: str, encoder: str) -> str:
    a = manifest()["assignments"]
    return a.get(f"{setting_key}/{encoder}", a["default"])


def ensure_dirs() -> None:
    for p in OUT_DIRS.values():
        p.mkdir(parents=True, exist_ok=True)
