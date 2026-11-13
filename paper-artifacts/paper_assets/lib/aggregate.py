"""Seed-level values and their summaries.

`SeedValues` carries one metric for one method in one setting/encoder, as the
list of per-seed values it was read from, with full provenance. Summaries are
the arithmetic mean and the sample SD with the manifest's ddof (1); nothing is
rounded here. Differences are formed per matched seed before summarising.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import config
from .provenance import current, issue

OK, MISSING, NOT_APPLICABLE, PENDING = "ok", "missing", "not_applicable", "pending"


@dataclass
class SeedValues:
    setting: str
    encoder: str
    method: str
    metric: str
    seeds: list = field(default_factory=list)
    values: list = field(default_factory=list)
    cohort: str | None = None
    sources: list = field(default_factory=list)     # [{file, run_id, condition, path}]
    dataset_version: str | None = None
    status: str = OK
    reason: str = ""

    # -- summaries --------------------------------------------------------- #
    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.status == OK else math.nan

    @property
    def sd(self) -> float:
        ddof = config.manifest()["conventions"]["sd_ddof"]
        if self.status != OK:
            return math.nan
        return float(np.std(self.values, ddof=ddof)) if self.n > ddof else 0.0

    @property
    def ok(self) -> bool:
        return self.status == OK

    def record(self, **extra) -> dict:
        return {"setting": self.setting, "encoder": self.encoder, "method": self.method,
                "metric": self.metric, "cohort": self.cohort, "status": self.status,
                "reason": self.reason, "seeds": list(self.seeds), "n": self.n,
                "values": list(self.values),
                "mean": None if not self.ok else self.mean,
                "sd": None if not self.ok else self.sd,
                "sd_ddof": config.manifest()["conventions"]["sd_ddof"],
                "dataset_version": self.dataset_version, "sources": self.sources, **extra}

    def register(self, **extra) -> "SeedValues":
        ctx = current()
        if ctx is not None:
            ctx.add_cell(self.record(**extra))
        return self


def missing(setting, encoder, method, metric, reason, cohort=None, level="missing",
            dataset_version=None) -> SeedValues:
    sv = SeedValues(setting, encoder, method, metric, cohort=cohort, status=MISSING,
                    reason=reason, dataset_version=dataset_version)
    issue(level, "missing_result", f"{setting}/{encoder} {method} {metric}: {reason}",
          setting=setting, encoder=encoder, method=method, metric=metric, cohort=cohort)
    return sv


def not_applicable(setting, encoder, method, metric, reason, cohort=None) -> SeedValues:
    return SeedValues(setting, encoder, method, metric, cohort=cohort, status=NOT_APPLICABLE,
                      reason=reason)


def pending(setting, encoder, method, metric, key: str) -> SeedValues:
    p = config.manifest()["pending"][key]
    ctx = current()
    assert p["status"] == "pending" and (ctx is None or ctx.name in p["exhibits"]), \
        f"{key} is not a planned pending entry for this exhibit"
    return SeedValues(setting, encoder, method, metric, status=PENDING, reason=p["reason"])


def paired_difference(a: SeedValues, b: SeedValues, method: str | None = None) -> SeedValues:
    """a - b per matched seed. Unmatched seed sets are a compatibility failure."""
    name = method or f"{a.method}-{b.method}"
    if not (a.ok and b.ok):
        return SeedValues(a.setting, a.encoder, name, a.metric, cohort=a.cohort,
                          status=MISSING if MISSING in (a.status, b.status) else a.status,
                          reason="an operand is not available")
    if a.setting != b.setting or a.encoder != b.encoder or a.metric != b.metric:
        raise ValueError(f"cannot difference {a.setting}/{a.encoder}/{a.metric} against "
                         f"{b.setting}/{b.encoder}/{b.metric}")
    if list(a.seeds) != list(b.seeds):
        issue("error", "unmatched_seeds",
              f"{a.setting}/{a.encoder} {a.metric}: {a.method} seeds {a.seeds} vs {b.method} "
              f"seeds {b.seeds}; difference not formed")
        return SeedValues(a.setting, a.encoder, name, a.metric, cohort=a.cohort, status=MISSING,
                          reason="unmatched seeds")
    vals = [float(x) - float(y) for x, y in zip(a.values, b.values)]
    return SeedValues(a.setting, a.encoder, name, a.metric, list(a.seeds), vals,
                      cohort=a.cohort if a.cohort == b.cohort else f"{a.cohort}|{b.cohort}",
                      sources=a.sources + b.sources, dataset_version=a.dataset_version)


def stat(values, ddof: int | None = None) -> dict:
    ddof = config.manifest()["conventions"]["sd_ddof"] if ddof is None else ddof
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=ddof)) if len(v) > ddof else 0.0,
            "n": int(len(v))}
