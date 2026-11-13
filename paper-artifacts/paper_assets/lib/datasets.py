"""Counts from the processed dataset definitions (the project's own loaders).

`processed(key)` calls `corpora.load(key)`: the registry's loader with its
deduplication and linking. It then derives, without any new rule:

* documents, and documents per nuisance class;
* retrieval pairs: the registered pair list, one undirected document-document link each;
* linked groups: connected components of those links (`protocols.link_groups`), by size;
* unpaired documents: documents in no registered link;
* topics, and whether graded gold similarity exists.

Nothing is embedded or fitted, and nothing is taken from split or query counts.

`from_manifest(key)` does the same for a corpus built outside the registry (Fedlex),
from the cohort manifest its notebook writes.
"""
from __future__ import annotations

import csv
import functools
import gzip
import hashlib
from collections import Counter

import numpy as np

from . import config
from .io import file_sha256


@functools.lru_cache(maxsize=None)
def from_manifest(key: str) -> dict:
    """Counts of a corpus built outside the registry, from its `cohort_manifest`.

    The manifest has one row per complete article tuple (one document per class), so a
    row holds C documents and C(C-1)/2 retrieval pairs. Also recomputes the cohort
    fingerprint the building notebook records: SHA-256 over the ordered
    "<article_id> <triple_sha256>" lines."""
    s = config.setting(key)
    rel = s["cohort_manifest"]
    with gzip.open(config.REPO / rel, "rt", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    n, C = len(rows), len(s["classes"])
    lines = "\n".join(f"{r['article_id']} {r['triple_sha256']}" for r in rows)
    return {"setting": key, "manifest": rel, "n_tuples": n, "n_unique_ids": len({r["article_id"] for r in rows}),
            "n_docs": C * n, "classes": list(s["classes"]), "class_counts": [n] * C,
            "n_retrieval_pairs": n * C * (C - 1) // 2,
            "cohort_sha256": hashlib.sha256(lines.encode("utf-8")).hexdigest(),
            "data_files": {f: file_sha256(config.REPO / f) for f in s.get("data_files", [])}}


@functools.lru_cache(maxsize=None)
def processed(key: str) -> dict:
    from .validate import _project_code
    with _project_code():
        import corpora
        import protocols as P
        d = corpora.load(key)
    y = np.asarray(d["source"])
    classes, counts = np.unique(y, return_counts=True)
    pairs = [tuple(map(int, p)) for p in (d["pairs"] or [])]
    linked = {i for p in pairs for i in p}
    sizes, edges = Counter(), Counter()
    if pairs:
        of_pair, members = P.link_groups(pairs)
        sizes = Counter(len(m) for m in members)
        per_group = Counter(int(g) for g in of_pair)            # links inside each group
        for g, m in enumerate(members):
            edges[(len(m), per_group[g])] += 1
    s = config.setting(key)
    return {"setting": key, "loader_label": d["label"], "confounder": d["config"]["confounder"],
            "n_docs": int(d["n_docs"]), "n_docs_texts": len(d["texts"]),
            "classes": [str(c) for c in classes], "class_counts": [int(c) for c in counts],
            "n_retrieval_pairs": len(pairs), "loader_n_pairs": int(d["n_pairs"]),
            "group_sizes": {str(k): int(v) for k, v in sorted(sizes.items())},
            "n_groups": int(sum(sizes.values())),
            "groups_by_size_and_links": {f"{a} docs/{b} links": int(v) for (a, b), v in sorted(edges.items())},
            "n_linked_docs": len(linked), "n_unpaired": int(d["n_docs"]) - len(linked),
            "loader_n_unpaired": int(d["n_unpaired"]),
            "n_topics": int(len(np.unique(d["topics"]))) if d.get("topics") is not None else None,
            "has_gold": d.get("gold") is not None,
            "registry_load": d["config"].get("load") or {},
            "data_files": {f: file_sha256(config.REPO / f) for f in s.get("data_files", [])}}
