"""Shared reporting infrastructure: manifest, loaders, aggregation, styles, LaTeX."""


def reset() -> None:
    """Forget every cached file, validation, cohort and logged issue (tests, re-runs)."""
    from . import io, loaders, provenance
    io.clear_cache()
    loaders._VALIDATED.clear()
    loaders.Cohort._instances.clear()
    provenance.ISSUES.clear()
    provenance.CHECKS.clear()
