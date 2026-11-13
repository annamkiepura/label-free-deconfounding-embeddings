"""Loaders for the saved raw results, with integrity and compatibility checks.

`Cohort(name, setting, encoder)` resolves the manifest's cohort for one
setting/encoder: which files, which saved condition each canonical method maps
to, and which seeds. `Cohort.values(method, metric)` returns a registered
`SeedValues`. Before any value leaves a file the file is checked once:

* ``seeds`` equals the per-seed records' seeds;
* every saved aggregate equals the per-seed values, their mean, and their
  sample SD with the manifest's ddof (this is what pins ddof = 1);
* fold combination: each seed's AUC/purity equals the mean of its saved fold
  values, and its R@10 equals the mean of its saved per-pair hits;
* the audit applied no AUC flip.

Before files are combined within a cohort: corpus and encoder fields, split
policy, audit protocol, selection rule and recipe agree; the per-seed split
summaries agree; the corpus statistics agree; and conditions present in two
files hold identical per-seed values.
"""
from __future__ import annotations

import math
import re

import numpy as np

from . import config
from .aggregate import SeedValues, missing, not_applicable
from .io import MissingArtifact, exists, read, run_id, sha256
from .provenance import check, issue

TOL = 1e-12
_VALIDATED: set[str] = set()


# --------------------------------------------------------------------------- #
# File-level integrity
# --------------------------------------------------------------------------- #
def container(d: dict) -> str:
    p0 = d["per_seed"][0]
    return "mechanisms" if "mechanisms" in p0 else "metrics"


def _close(a, b, tol=TOL) -> bool:
    return (a is None and b is None) or (a is not None and b is not None
                                         and math.isclose(float(a), float(b), rel_tol=0, abs_tol=tol))


def validate_file(rel: str) -> dict:
    d = read(rel)
    if rel in _VALIDATED:
        return d
    _VALIDATED.add(rel)
    ddof = config.manifest()["conventions"]["sd_ddof"]
    ps = d.get("per_seed")
    if not isinstance(ps, list) or not ps or "seed" not in ps[0]:
        return d
    seeds = [int(p["seed"]) for p in ps]
    if d.get("seeds") is not None:
        check(list(d["seeds"]) == seeds, "seeds_match", f"{rel}: seeds field equals per-seed records",
              saved=d["seeds"], per_seed=seeds)
    check(len(set(seeds)) == len(seeds), "seeds_unique", f"{rel}: no duplicated seed")
    audit = (d.get("config") or {}).get("audit") or {}
    if "flip_adjustment" in audit:
        check(audit["flip_adjustment"] is None and not config.manifest()["conventions"]["auc_flip"],
              "auc_unflipped", f"{rel}: AUCs saved without flip adjustment")
    box = container(d) if "metrics" in ps[0] or "mechanisms" in ps[0] else None
    if box and "agg" in d:
        bad = []
        for cond, metrics in d["agg"].items():
            for metric, a in metrics.items():
                if not isinstance(a, dict):
                    continue
                items = [(None, a)] if "values" in a else [(k, v) for k, v in a.items()
                                                            if isinstance(v, dict) and "values" in v]
                for sub, s in items:
                    try:
                        vals = [p[box][cond][metric] if sub is None else p[box][cond][metric][sub]
                                for p in ps]
                    except (KeyError, TypeError):
                        bad.append(f"{cond}.{metric}{'.' + sub if sub else ''}: absent from a seed")
                        continue
                    sd = float(np.std(vals, ddof=ddof)) if len(vals) > ddof else 0.0
                    if not (len(vals) == s["n"] == len(s["values"])
                            and all(_close(x, y) for x, y in zip(vals, s["values"]))
                            and _close(np.mean(vals), s["mean"]) and _close(sd, s["std"])):
                        bad.append(f"{cond}.{metric}{'.' + sub if sub else ''}")
        check(not bad, "agg_matches_per_seed",
              f"{rel}: saved aggregates equal per-seed values, mean and SD (ddof={ddof})", bad=bad[:20])
    # Fold combination within seed, and hits -> recall.
    fold_bad, hit_bad = [], []
    for p in ps:
        n_folds = (p.get("split") or {}).get("n_folds")
        for cond, per in (p.get("per_fold") or {}).items():
            for metric, fv in per.items():
                if not fv:
                    continue
                if n_folds is not None and len(fv) != n_folds:
                    fold_bad.append(f"seed {p['seed']} {cond}.{metric}: {len(fv)} fold values, {n_folds} folds")
                saved = p[box][cond].get(metric)
                if saved is not None and not _close(np.mean(fv), saved):
                    fold_bad.append(f"seed {p['seed']} {cond}.{metric}")
        for cond, h in (p.get("hits") or {}).items():
            saved = p[box][cond].get("recall@10")
            if saved is not None and not _close(np.mean(h), saved):
                hit_bad.append(f"seed {p['seed']} {cond}")
    if any(p.get("per_fold") for p in ps):
        check(not fold_bad, "fold_combination",
              f"{rel}: each seed's AUC/purity equals the mean of its saved fold values", bad=fold_bad[:20])
    if any(p.get("hits") for p in ps):
        check(not hit_bad, "hits_recall",
              f"{rel}: each seed's R@10 equals the mean of its saved per-pair hits", bad=hit_bad[:20])
    return d


def dataset_version(setting: str, d: dict | None = None) -> str:
    s = config.setting(setting)
    cs = (d or {}).get("corpus_stats") or {}
    n = f"; {cs['n_docs']:,} docs" if cs.get("n_docs") else ""
    return f"{setting} [{s['long']}; classes {'/'.join(s['classes'])}{n}]"


# --------------------------------------------------------------------------- #
# Cohorts
# --------------------------------------------------------------------------- #
class Cohort:
    """The saved runs one manifest cohort contributes to a setting/encoder."""

    _instances: dict = {}

    def __new__(cls, name, setting, encoder):
        key = (name, setting, encoder)
        if key not in cls._instances:
            obj = super().__new__(cls)
            obj._init(name, setting, encoder)
            cls._instances[key] = obj
        return cls._instances[key]

    def _init(self, name, setting, encoder):
        spec = config.manifest()["cohorts"][name]
        self.name, self.setting, self.encoder, self.spec = name, setting, encoder, spec
        root = spec.get("root")              # a cohort outside results_root names its tree
        if root is not None and root not in config.manifest().get("extra_roots", []):
            raise KeyError(f"cohort {name!r}: root {root!r} is not one of the manifest's extra_roots")
        self.paths = {role: (f"{root}/" if root else "") + pat.format(setting=setting, encoder=encoder)
                      for role, pat in spec["files"].items()}
        self._compat_done = False

    @classmethod
    def assigned(cls, setting, encoder):
        return cls(config.cohort_for(setting, encoder), setting, encoder)

    # -- resolution --------------------------------------------------------- #
    def resolve(self, method: str) -> tuple[str, str]:
        """``(file role, saved condition)`` for a canonical or parameterised method."""
        if method in self.spec["methods"]:
            role, cond = self.spec["methods"][method].split(":")
            return role, cond
        for pat, role in (self.spec.get("patterns") or {}).items():
            if re.fullmatch(pat, method):
                return role, method
        raise KeyError(f"cohort {self.name!r} has no rule for method {method!r}")

    def file(self, role: str) -> dict:
        rel = self.paths[role]
        d = validate_file(rel)
        check(d.get("corpus") == self.setting and d.get("encoder") == self.encoder, "file_identity",
              f"{rel}: corpus/encoder fields are {self.setting}/{self.encoder}",
              corpus=d.get("corpus"), encoder=d.get("encoder"))
        return d

    def available(self, role: str) -> bool:
        return exists(self.paths[role])

    def seeds_for(self, d: dict) -> list[int]:
        saved = [int(p["seed"]) for p in d["per_seed"]]
        want = self.spec["seeds"]
        return saved if want == "saved" else [s for s in want]

    # -- compatibility across the cohort's files --------------------------- #
    def compatibility(self) -> None:
        if self._compat_done:
            return
        self._compat_done = True
        files = {r: self.file(r) for r in self.paths if self.available(r)}
        if len(files) < 2:
            return
        roles = list(files)
        ref_role, ref = roles[0], files[roles[0]]
        for role in roles[1:]:
            d = files[role]
            tag = f"{self.name}:{self.setting}/{self.encoder} {ref_role}~{role}"
            for key in ("split", "audit", "selection", "pseudo_groups"):
                a, b = (ref.get("config") or {}).get(key), (d.get("config") or {}).get(key)
                if a is not None and b is not None:
                    check(a == b, "config_compatible", f"{tag}: config.{key} identical", a=a, b=b)
            ra, rb = (ref.get("config") or {}).get("recipe"), (d.get("config") or {}).get("recipe")
            if ra and rb:
                diff = {k: (ra.get(k), rb.get(k)) for k in set(ra) | set(rb) if ra.get(k) != rb.get(k)}
                check(not diff, "recipe_compatible", f"{tag}: SAE/eraser recipe identical", diff=diff)
            ca, cb = ref.get("corpus_stats"), d.get("corpus_stats")
            if ca and cb:
                check(ca == cb, "corpus_compatible", f"{tag}: corpus statistics identical")
            sa = {p["seed"]: p.get("split") for p in ref["per_seed"]}
            sb = {p["seed"]: p.get("split") for p in d["per_seed"]}
            common = sorted(set(sa) & set(sb))
            if all(sa[s] and sb[s] for s in common):
                check(all(sa[s] == sb[s] for s in common), "split_compatible",
                      f"{tag}: per-seed split summaries identical on seeds {common}")
            # Conditions both files hold must agree seed by seed.
            ba, bb = container(ref), container(d)
            shared_bad, shared_n = [], 0
            conds = set(ref["per_seed"][0][ba]) & set(d["per_seed"][0][bb])
            for cond in conds:
                for s in common:
                    ma = next(p for p in ref["per_seed"] if p["seed"] == s)[ba][cond]
                    mb = next(p for p in d["per_seed"] if p["seed"] == s)[bb][cond]
                    for metric in set(ma) & set(mb):
                        if isinstance(ma[metric], (int, float)) and isinstance(mb[metric], (int, float)):
                            shared_n += 1
                            if not _close(ma[metric], mb[metric]):
                                shared_bad.append(f"seed {s} {cond}.{metric}")
            if shared_n:
                check(not shared_bad, "shared_values_identical",
                      f"{tag}: {shared_n} shared per-seed values identical", bad=shared_bad[:20])
        # Coverage cohort: the mechanism run's Directions-LEACE must be the main run's SAE-LEACE.
        if "mech" in files and "main" in files:
            m, a = files["mech"], files["main"]
            bad = []
            for pm in m["per_seed"]:
                pa = next((p for p in a["per_seed"] if p["seed"] == pm["seed"]), None)
                if pa is None:
                    continue
                for metric, v in pm["mechanisms"]["directions_leace"].items():
                    w = pa["metrics"]["sae_mi"].get(metric)
                    if isinstance(v, float) and isinstance(w, float) and not _close(v, w):
                        bad.append(f"seed {pm['seed']} {metric}")
            check(not bad, "mechanism_matches_main",
                  f"{self.setting}/{self.encoder}: mechanism Directions-LEACE equals coverage SAE-LEACE",
                  bad=bad)
            check(all(p.get("same_selected_set", False) for p in m["per_seed"]), "same_selected_set",
                  f"{self.setting}/{self.encoder}: the three mechanisms share the selected latents")

    # -- values ------------------------------------------------------------- #
    def values(self, method: str, metric: str, sub: str | None = None,
               register: bool = True, quiet: bool = False) -> SeedValues:
        """Per-seed values of one metric (``sub`` selects a key of a dict metric).

        ``quiet`` probes availability without logging a missing result (used by
        cross-checks that scan metrics a setting may never have)."""
        name = f"{metric}[{sub}]" if sub else metric
        if quiet:
            from .provenance import ISSUES
            n0 = len(ISSUES)
            try:
                return self.values(method, metric, sub, register=False)
            finally:
                from .provenance import current
                del ISSUES[n0:]
                ctx = current()
                if ctx is not None:
                    ctx.issues[:] = [i for i in ctx.issues if i in ISSUES]
        try:
            role, cond = self.resolve(method)
        except KeyError as e:
            return missing(self.setting, self.encoder, method, name, str(e), cohort=self.name)
        rel = self.paths[role]
        try:
            d = self.file(role)
        except MissingArtifact:
            return missing(self.setting, self.encoder, method, name, f"file not found: {rel}",
                           cohort=self.name)
        self.compatibility()
        box = container(d)
        by_seed = {int(p["seed"]): p for p in d["per_seed"]}
        seeds = self.seeds_for(d)
        absent = [s for s in seeds if s not in by_seed]
        if absent:
            return missing(self.setting, self.encoder, method, name,
                           f"{rel} lacks seeds {absent} required by cohort {self.name}", cohort=self.name)
        vals = []
        for s in seeds:
            m = by_seed[s][box].get(cond)
            v = None if m is None else m.get(metric)
            if isinstance(v, dict):
                v = v.get(sub) if sub else None
            if v is None:
                return missing(self.setting, self.encoder, method, name,
                               f"{rel}: per_seed[seed={s}].{box}.{cond}.{name} not recorded",
                               cohort=self.name, dataset_version=dataset_version(self.setting, d))
            vals.append(float(v))
        path = f"per_seed[seed in {seeds}].{box}.{cond}.{metric}" + (f".{sub}" if sub else "")
        sv = SeedValues(self.setting, self.encoder, method, name, seeds, vals, cohort=self.name,
                        sources=[{"file": rel, "run_id": run_id(rel, sha256(rel)),
                                  "condition": cond, "path": path}],
                        dataset_version=dataset_version(self.setting, d))
        return sv.register() if register else sv


def utility_metric(setting: str) -> str:
    return config.setting(setting)["utility"]


def utility(cohort: Cohort, method: str) -> SeedValues:
    u = utility_metric(cohort.setting)
    if not u:
        return not_applicable(cohort.setting, cohort.encoder, method, "utility",
                              "no semantic utility metric (unpaired corpus without topic labels)",
                              cohort=cohort.name)
    return cohort.values(method, u)


# --------------------------------------------------------------------------- #
# Planned results (manifest [pending.*] entries with a result_file slot)
# --------------------------------------------------------------------------- #
def planned(key: str, reference: Cohort, metric: str, method: str) -> SeedValues:
    """Values of a planned control, or a "pending" placeholder while its file is absent.

    The file named by the manifest's `result_file` pattern must match `reference` (the
    cohort the row is compared with) in seeds, split policy, audit protocol and per-seed
    split summaries, or the cell is reported missing -- never silently mixed in."""
    from .aggregate import pending
    spec = config.manifest()["pending"][key]
    rel = spec["result_file"].format(setting=reference.setting, encoder=reference.encoder)
    if not exists(rel):
        return pending(reference.setting, reference.encoder, method, metric, key)
    d = validate_file(rel)
    ref = reference.file("main")
    tag = f"{rel} vs {reference.paths['main']}"
    ok = (d.get("corpus") == reference.setting and d.get("encoder") == reference.encoder
          and [p["seed"] for p in d["per_seed"]] == reference.seeds_for(ref)
          and d["config"]["split"] == ref["config"]["split"] and d["config"]["audit"] == ref["config"]["audit"]
          and all(p.get("split") == q.get("split") for p, q in zip(d["per_seed"], ref["per_seed"])))
    check(ok, "planned_compatible", f"{tag}: planned result matches the reference cohort (seeds, splits, audit)")
    if not ok:
        return missing(reference.setting, reference.encoder, method, metric,
                       f"{rel} exists but is not compatible with {reference.paths['main']}")
    vals = [p["metrics"][spec["condition"]].get(metric) for p in d["per_seed"]]
    if any(v is None for v in vals):
        return missing(reference.setting, reference.encoder, method, metric, f"{rel}: {metric} not recorded")
    return SeedValues(reference.setting, reference.encoder, method, metric,
                      [p["seed"] for p in d["per_seed"]], [float(v) for v in vals], cohort="planned:" + key,
                      sources=[{"file": rel, "run_id": run_id(rel, sha256(rel)), "condition": spec["condition"],
                                "path": f"per_seed[*].metrics.{spec['condition']}.{metric}"}],
                      dataset_version=dataset_version(reference.setting, d)).register()


# --------------------------------------------------------------------------- #
# The 5-draw random-latent rerun (code/focused_additions.py, export_band)
# --------------------------------------------------------------------------- #
#: Random SAE-LEACE on the coverage splits and SAEs, five draws per seed from the latents MI scores.
#: Read directly by the exhibits that use it (Figure 1's random curve, Table 3 row 2) -- not as a
#: cohort file, so no other exhibit opens it -- after the checks the cohort applies to its own files.
RANDOM_BAND_FILE = "random_band/random_band_{setting}_{encoder}.json"
RANDOM_BAND_DRAWS, RANDOM_BAND_POOL = 5, [0.02, 0.98]


def random_band(coh: "Cohort") -> tuple[str, dict]:
    """The rerun's file for one coverage setting, checked against the coverage run: identity, seeds,
    split policy, audit, recipe, corpus, per-seed splits, and its recorded design (draws, pool, and
    the published SAEs -- the rerun reproduced every fit's MI selection)."""
    rel = RANDOM_BAND_FILE.format(setting=coh.setting, encoder=coh.encoder)
    d, main = validate_file(rel), coh.file("main")
    cfg, ref = d.get("config") or {}, main["config"]
    rc = d.get("reference_check") or {}
    seeds = coh.seeds_for(main)
    check(d.get("corpus") == coh.setting and d.get("encoder") == coh.encoder
          and [int(p["seed"]) for p in d["per_seed"]] == seeds
          and all(cfg.get(k) == ref.get(k) for k in ("split", "audit", "recipe", "selection", "pseudo_groups"))
          and d.get("corpus_stats") == main.get("corpus_stats")
          and all(p.get("split") == next(q for q in main["per_seed"] if q["seed"] == p["seed"]).get("split")
                  for p in d["per_seed"]),
          "random_band_compatible", f"{rel}: same setting, seeds {seeds}, split policy, audit, recipe, "
          f"corpus and per-seed splits as {coh.paths['main']}")
    check(d.get("draws") == RANDOM_BAND_DRAWS and d.get("same_sae_as_published") is True
          and rc.get("splits_match") is True and rc.get("mi_reproduced_everywhere") is True
          and str(RANDOM_BAND_POOL) in d.get("definition", ""),
          "random_band_design", f"{rel}: {RANDOM_BAND_DRAWS} draws per seed from the MI band "
          f"{RANDOM_BAND_POOL}, on the published SAEs (MI selection reproduced in every fit)")
    return rel, d


def random_band_values(coh: "Cohort", rel: str, d: dict, cond: str, metric: str) -> SeedValues:
    """Per-seed values (each the mean of the draws) of one saved condition of the rerun."""
    vals = [float(p["metrics"][cond][metric]) for p in d["per_seed"]]
    return SeedValues(coh.setting, coh.encoder, cond, metric, [int(p["seed"]) for p in d["per_seed"]], vals,
                      cohort="coverage (5-draw random rerun)",
                      sources=[{"file": rel, "run_id": run_id(rel, sha256(rel)), "condition": cond,
                                "path": f"per_seed[*].metrics.{cond}.{metric}"}],
                      dataset_version=dataset_version(coh.setting, d)).register()


# --------------------------------------------------------------------------- #
# Geometry (alternatives.run_geometry), seeds x folds
# --------------------------------------------------------------------------- #
def geometry(setting: str, encoder: str) -> dict | None:
    rel = f"alternatives/geometry_{setting}_{encoder}.json"
    if not exists(rel):
        issue("missing", "missing_result", f"{rel}: geometry run not found", setting=setting,
              encoder=encoder)
        return None
    d = validate_file(rel)
    check(d.get("corpus") == setting and d.get("encoder") == encoder, "file_identity",
          f"{rel}: corpus/encoder fields are {setting}/{encoder}")
    seeds = [int(p["seed"]) for p in d["per_seed"]]

    def per_seed(get):
        return [float(np.mean([get(f) for f in p["folds"]])) for p in d["per_seed"]]

    out = {"file": rel, "run_id": run_id(rel, sha256(rel)), "seeds": seeds,
           "n_classes": d["n_classes"], "class_rank": d["class_rank"],
           "capture": {r: per_seed(lambda f, r=r: f["capture"][r]) for r in d["agg"]["capture"]},
           "ari": {k: per_seed(lambda f, k=k: f["ari"][k]) for k in d["agg"]["ari"]},
           "k_auto": [[int(f["k_auto"]) for f in p["folds"]] for p in d["per_seed"]],
           "rank_90": per_seed(lambda f: f["rank_90"])}
    bad = [f"capture[{r}]" for r, v in out["capture"].items() if not _close(np.mean(v), d["agg"]["capture"][r])]
    bad += [f"ari[{k}]" for k, v in out["ari"].items() if not _close(np.mean(v), d["agg"]["ari"][k])]
    check(not bad, "geometry_agg", f"{rel}: saved geometry means equal the per-seed fold means", bad=bad)
    return out


# --------------------------------------------------------------------------- #
# OOD frozen transfer
# --------------------------------------------------------------------------- #
OOD_SUMMARIES = ("combinations/ood_summary.json", "ood/ood_summary_kmeans.json",
                 "ood/ood_summary.json", "ood/ood_summary_ranks.json")


def ood_setting(summary_rel: str, key: str) -> dict:
    s = read(summary_rel)
    if key not in s["settings"]:
        raise MissingArtifact(f"{summary_rel}: no setting {key}")
    return {"summary": s, "setting": s["settings"][key]}


def ood_verify_point(summary_rel: str, key: str, transform: str, run_rel: str,
                     run_transform: str | None = None) -> float:
    """Recompute a saved legal-macro change from the per-query records and check it.

    Change = mean over tasks of (mean over queries of (seed-averaged transformed
    score - raw score)), the estimand `ood.summarize` saves. Returns the value.
    """
    st = read(summary_rel)["settings"][key]
    run = validate_file(run_rel)
    base_rel = f"ood/ood_{st['encoder']}_{st['donor']}.json"
    base = read(base_rel)
    check([p["fit"]["split_id"] for p in run["per_seed"]] == st["split_ids"], "ood_split_ids",
          f"{run_rel}: donor split fingerprints equal the summary's {st['split_ids']}")
    rt = run_transform or transform
    deltas = []
    for task in st["legal"]:
        raw = np.asarray(run["raw"][task]["per_unit"], float)
        if run_rel != base_rel:
            check(run["tasks"][task]["unit_ids"] == base["tasks"][task]["unit_ids"]
                  and np.allclose(raw, base["raw"][task]["per_unit"], atol=TOL, rtol=0),
                  "ood_raw_identical", f"{run_rel} {task}: same query ids and raw scores as {base_rel}")
        tr = np.mean([p["scores"][rt][task]["per_unit"] for p in run["per_seed"]], axis=0)
        deltas.append(float(np.mean(tr - raw)))
    delta = float(np.mean(deltas))
    saved = st["changes"][transform]["legal_macro"]["delta"]
    check(_close(delta, saved, 1e-10), "ood_delta_recomputed",
          f"{summary_rel} {key} {transform}: legal-macro change recomputed from per-query records "
          f"({delta:+.6f})", saved=saved)
    return delta


# --------------------------------------------------------------------------- #
# Inventories
# --------------------------------------------------------------------------- #
def inventory(setting: str) -> dict:
    combo = read("combinations/inventory.json")["settings"]
    if setting in combo:
        return {"source": "combinations/inventory.json", **combo[setting]}
    inv = read("inventory/dataset_inventory.json")["corpora"]
    return {"source": "inventory/dataset_inventory.json", **inv[setting]}
