"""Read-only access to saved raw results, with provenance for every file touched.

Every JSON is read through `read`, which hashes the bytes (SHA-256) and records
the file in the active exhibit's provenance. Raw files are opened read-only and
never written. A saved run has no run identifier of its own, so the run ID used
throughout is ``<relative path>@<first 12 hex of SHA-256>``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import config
from .provenance import current

_CACHE: dict[str, tuple[dict, str, int]] = {}


class MissingArtifact(FileNotFoundError):
    """A result the manifest selects does not exist on disk."""


def run_id(rel: str, sha: str) -> str:
    return f"{rel}@{sha[:12]}"


def path_of(rel: str) -> Path:
    """``<results_root>/<rel>``, or ``<repo>/<rel>`` for a file under one of the manifest's `extra_roots`."""
    if any(rel.startswith(root.rstrip("/") + "/") for root in config.manifest().get("extra_roots", [])):
        return config.REPO / rel
    return config.results_root() / rel


def exists(rel: str) -> bool:
    return path_of(rel).exists()


def read(rel: str) -> dict:
    """Parsed JSON of the saved result `rel` (see `path_of`); registers it with the active exhibit."""
    if rel not in _CACHE:
        path = path_of(rel)
        if not path.exists():
            raise MissingArtifact(rel)
        raw = path.read_bytes()
        _CACHE[rel] = (json.loads(raw), hashlib.sha256(raw).hexdigest(), len(raw))
    data, sha, size = _CACHE[rel]
    ctx = current()
    if ctx is not None:
        ctx.add_source(rel, sha, size, data)
    return data


def sha256(rel: str) -> str:
    read(rel)
    return _CACHE[rel][1]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clear_cache() -> None:
    _CACHE.clear()
