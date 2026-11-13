"""Cross-exhibit validation and the validation report.

* `cross_cohort_agreement` compares, for every setting/encoder that has both an
  `id` and a `coverage` run, the values the two cohorts share on common seeds
  (Raw, PC-1, Metadata-LEACE, SAE-LEACE, k-means LEACE). This is what licenses
  the SCOTUS coverage-cohort decision and states, for SwiLTra, whether the
  five-seed headline and the three-seed controls use the same fits.
* `rederive_splits` (opt-in, ``--splits``) rebuilds every split with the
  project's split engine from data/ (deterministic; no embeddings, no model),
  checks the saved split summaries and OOD split fingerprints, verifies that no
  registered linked group straddles a train/test boundary, and recomputes the
  pair-specific R@10 of the pooled three-class settings from the saved per-pair
  hits.
* `write_report` writes reports/validation_report.{md,json} and
  reports/result_sources.md.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import config
from .io import _CACHE, exists, read, run_id, sha256
from .loaders import Cohort, validate_file
from .methods import CORE
from .provenance import CHECKS, ISSUES, Exhibit, check, issue

SHARED = {"raw": "raw", "pca_top1": "pca_top1", "metadata_leace": "metadata_leace",
          "sae_leace": "sae_leace", "kmeans_auto": "kmeans_auto"}


def cross_cohort_agreement() -> list[dict]:
    out = []
    for key in config.settings():
        for enc in ("e5-small", "mini"):
            a, b = Cohort("id", key, enc), Cohort("coverage", key, enc)
            if not (a.available("main") and b.available("main")):
                continue
            for meth in CORE:
                for metric in ("linear_auc", "gb_auc", "mlp_auc", "recall@10", "topic_purity"):
                    va, vb = a.values(meth, metric, quiet=True), b.values(meth, metric, quiet=True)
                    if not (va.ok and vb.ok):
                        continue
                    common = sorted(set(va.seeds) & set(vb.seeds))
                    da, db = dict(zip(va.seeds, va.values)), dict(zip(vb.seeds, vb.values))
                    diff = max(abs(da[s] - db[s]) for s in common)
                    out.append({"setting": key, "encoder": enc, "method": meth, "metric": metric,
                                "id_seeds": va.seeds, "coverage_seeds": vb.seeds, "common_seeds": common,
                                "max_abs_diff": diff, "identical": diff < 1e-12,
                                "id_mean_all_seeds": float(np.mean(va.values)),
                                "id_mean_common_seeds": float(np.mean([da[s] for s in common])),
                                "coverage_mean": float(np.mean(vb.values))})
    for r in out:
        tag = f"{r['setting']}/{r['encoder']} {r['method']} {r['metric']}"
        if config.cohort_for(r["setting"], r["encoder"]) == "coverage":
            check(r["identical"], "authoritative_cohort_agrees",
                  f"{tag}: coverage cohort equals run_id on seeds {r['common_seeds']}",
                  max_abs_diff=r["max_abs_diff"])
        elif not r["identical"]:
            issue("warning", "cohorts_differ",
                  f"{tag}: run_id and the coverage run differ on common seeds {r['common_seeds']} "
                  f"(max |diff| {r['max_abs_diff']:.2e}); ID exhibits use run_id, controls use the "
                  f"coverage run", **r)
    return out


# --------------------------------------------------------------------------- #
# Deterministic split re-derivation
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _project_code():
    code = str(config.REPO / config.manifest()["code_dir"])
    old = os.getcwd()
    sys.path.insert(0, code)
    os.chdir(config.REPO)
    try:
        import warnings
        warnings.filterwarnings("ignore")
        yield
    finally:
        os.chdir(old)
        sys.path.remove(code)


def rederive_splits() -> list[dict]:
    """See module docstring. Returns one record per (file, seed)."""
    with _project_code():
        import corpora
        import experiments as E
        import protocols as P
        import ood as O
        from revised import pair_kinds

        records = []
        files = sorted({rel for rel in _CACHE if rel.startswith(("id/", "alternatives/alternatives_",
                                                                  "alternatives/kmeans_count_",
                                                                  "encoders/", "alternatives/generous_k_"))})
        data_cache: dict = {}
        ood_ids = {}
        for summ in ("combinations/ood_summary.json", "ood/ood_summary_kmeans.json", "ood/ood_summary.json"):
            if exists(summ):
                for k, st in read(summ)["settings"].items():
                    ood_ids[(st["encoder"], st["donor"])] = dict(zip(st["seeds"], st["split_ids"]))
        for rel in files:
            d = read(rel)
            key = d.get("corpus")
            if key not in data_cache:
                data_cache[key] = corpora.load(key)
            data = data_cache[key]
            pol = d["config"]["split"]
            pairs = data["pairs"]
            of_pair, members = P.link_groups(pairs) if pairs else (None, [])
            for p in d["per_seed"]:
                s = int(p["seed"])
                sps = E._splits(data, s, "document", pol)
                summ = {"scheme": "kfold" if len(sps) > 1 else "holdout", "n_folds": len(sps),
                        "n_train_docs": [int(len(x.train_docs)) for x in sps],
                        "n_test_docs": [int(len(x.test_docs)) for x in sps],
                        "n_test_pairs": [int(len(x.test_pairs or [])) for x in sps]}
                ok_summary = p.get("split") == summ
                straddle = 0
                for sp in sps:
                    te = np.zeros(data["n_docs"], bool)
                    te[np.asarray(sp.test_docs)] = True
                    for g in members:
                        side = te[g]
                        straddle += int(side.any() and not side.all())
                rec = {"file": rel, "setting": key, "encoder": d.get("encoder"), "seed": s,
                       "summary_matches": ok_summary, "straddling_groups": straddle,
                       "n_groups": len(members), "split_id": O._split_id(sps[0].train_docs)
                       if len(sps) == 1 else None}
                ref = ood_ids.get((d.get("encoder"), key), {}).get(s)
                if ref and rec["split_id"]:
                    rec["ood_split_id_matches"] = rec["split_id"] == ref
                # Pair-specific R@10 from the saved per-pair hits (pooled three-class runs).
                if p.get("hits") and "recall@10_by_pair" in next(iter(p["metrics"].values())):
                    kinds = np.concatenate([pair_kinds(data["source"], x.test_pairs_global) for x in sps])
                    bad = []
                    for cond, h in p["hits"].items():
                        h = np.asarray(h)
                        if len(h) != len(kinds):
                            bad.append(f"{cond}: {len(h)} hits vs {len(kinds)} pairs")
                            continue
                        for kind, v in p["metrics"][cond]["recall@10_by_pair"].items():
                            if abs(h[kinds == kind].mean() - v) > 1e-12:
                                bad.append(f"{cond} {kind}")
                    rec["pair_recall_recomputed"] = not bad
                    rec["pair_recall_bad"] = bad
                records.append(rec)
        for r in records:
            tag = f"{r['file']} seed {r['seed']}"
            check(r["summary_matches"], "split_rederived", f"{tag}: rebuilt split matches the saved summary")
            check(r["straddling_groups"] == 0, "linked_groups_intact",
                  f"{tag}: no registered linked group ({r['n_groups']}) straddles train/test")
            if "ood_split_id_matches" in r:
                check(r["ood_split_id_matches"], "split_id_matches_ood",
                      f"{tag}: rebuilt split fingerprint equals the OOD donor fit's")
            if "pair_recall_recomputed" in r:
                check(r["pair_recall_recomputed"], "pair_recall_recomputed",
                      f"{tag}: pair-specific R@10 recomputed from per-pair hits", bad=r["pair_recall_bad"])
        return records


# --------------------------------------------------------------------------- #
# Static provenance findings (documented in the repository; not re-derivable here)
# --------------------------------------------------------------------------- #
UNRESOLVED = [
    ("No run identifiers", "Saved raw JSON carry no run or job ID, timestamp or code hash. The "
     "reporting pipeline therefore identifies each run as <relative path>@<SHA-256 prefix>."),
    ("SCOTUS case identifiers", "The SCOTUS CSVs have no case ID. Linked groups are the registered "
     "pairs and their transitive closure (12 ambiguous file pairs dropped in the pooled corpus); "
     "unregistered matching cases can still fall on both sides of a split "
     "(audit_2026-09-28/outputs/C1_split_overlap.json)."),
    ("Encoder revisions", "encoder_inventory.json records the model-cache snapshot only for E5-small "
     "and MiniLM; the other eight encoders carry the Hub revision current at inventory time, not a "
     "run-resolved one."),
    ("Colab merge", "alternatives_swiltra_e5-small.json and alternatives_scotus_oyez_lexis_e5-small.json "
     "carry a documented `merged` key: selector arms and nonlinear probes folded in from an earlier "
     "run with identical seeds, splits and SAE."),
    ("SAE reproducibility", "SAE fits are not bit-reproducible across hardware; locally saved maps are "
     "not the published SAE maps (audit map_identity.json). Nothing here refits, so reported values "
     "are the saved Colab ones."),
    ("OOD prompt convention", "E5 donors were encoded with the 'query: ' prefix while MTEB targets "
     "are encoded plainly (ood.PROMPT_STYLE='parent'); reported as saved."),
    ("Prior draft seed policy", "writing/tacl_analysis_draft reported every main comparison on seeds "
     "0-2. This pipeline follows the current instruction instead: SCOTUS/E5-small uses the coverage "
     "cohort (0-2), while SwiLTra and MiniLM keep all saved seeds (0-4). SwiLTra headline means "
     "therefore differ from that draft's."),
]


def write_report(exhibits: list[Exhibit], agreement: list[dict], splits: list[dict] | None) -> None:
    rep = config.OUT_DIRS["reports"]
    by_level = Counter(i["level"] for i in ISSUES)
    used = {}
    for ex in exhibits:
        for rel, meta in ex.sources.items():
            used.setdefault(rel, {**meta, "exhibits": []})["exhibits"].append(ex.name)
    all_raw = sorted(str(p.relative_to(config.results_root())) for p in config.results_root().rglob("*.json"))
    unused = [r for r in all_raw if r not in used]
    doc = {"summary": {"exhibits": [e.name for e in exhibits], "checks_passed": len(CHECKS),
                       "issues_by_level": dict(by_level), "raw_files_total": len(all_raw),
                       "raw_files_used": len(used), "split_rederivation_run": splits is not None},
           "conventions": config.manifest()["conventions"],
           "issues": ISSUES, "cross_cohort_agreement": agreement,
           "split_rederivation": splits, "unresolved_provenance": dict(UNRESOLVED),
           "pending": config.manifest().get("pending", {}),
           "sources_used": used, "raw_files_unused": unused}
    (rep / "validation_report.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")

    L = ["# Validation report", "",
         f"Exhibits built: {', '.join(e.name for e in exhibits)}.", "",
         f"- Consistency checks passed: **{len(CHECKS)}**",
         f"- Errors: **{by_level.get('error', 0)}**, missing results: **{by_level.get('missing', 0)}**, "
         f"warnings: **{by_level.get('warning', 0)}**, info: {by_level.get('info', 0)}",
         f"- Raw JSON files: {len(all_raw)} under `{config.manifest()['results_root']}`, "
         f"{len(used)} read by the exhibits",
         f"- Split re-derivation: {'run' if splits is not None else 'not run (use `validate --splits`)'}",
         "", "## Conventions", ""]
    for k, v in config.manifest()["conventions"].items():
        L.append(f"- `{k}`: {v}")
    L += ["", "## Errors and missing results", ""]
    bad = [i for i in ISSUES if i["level"] in ("error", "missing")]
    L += [f"- **{i['level']}** [{i['exhibit']}] `{i['code']}`: {i['message']}" for i in bad] or ["None."]
    L += ["", "## Warnings", ""]
    L += [f"- [{i['exhibit']}] `{i['code']}`: {i['message']}" for i in ISSUES if i["level"] == "warning"] or ["None."]
    L += ["", "## Scope notes (info)", ""]
    L += [f"- [{i['exhibit']}] {i['message']}" for i in ISSUES if i["level"] == "info"] or ["None."]
    L += ["", "## Planned controls (Pending wherever their result file is absent)", ""]
    for k, p in config.manifest().get("pending", {}).items():
        L.append(f"- `{k}` ({', '.join(p['exhibits'])}): {p['reason']}")
    L += ["", "## Cohort agreement: run_id vs the coverage run on common seeds", "",
          "| Setting | Encoder | Method | Metric | run_id seeds | coverage seeds | max abs diff | used by ID exhibits |",
          "|---|---|---|---|---|---|---:|---|"]
    for r in agreement:
        L.append(f"| {r['setting']} | {r['encoder']} | {r['method']} | {r['metric']} | "
                 f"{r['id_seeds']} | {r['coverage_seeds']} | {r['max_abs_diff']:.1e} | "
                 f"{config.cohort_for(r['setting'], r['encoder'])} |")
    if splits is not None:
        n = len(splits)
        L += ["", "## Split re-derivation", "",
              f"{n} (file, seed) splits rebuilt from data/ with the project's split engine.",
              f"- summaries match: {sum(r['summary_matches'] for r in splits)}/{n}",
              f"- linked-group straddles: {sum(r['straddling_groups'] for r in splits)}",
              f"- OOD split fingerprints matched: {sum(r.get('ood_split_id_matches') is True for r in splits)}"
              f"/{sum('ood_split_id_matches' in r for r in splits)}",
              f"- pair-specific R@10 recomputed from hits: {sum(r.get('pair_recall_recomputed') is True for r in splits)}"
              f"/{sum('pair_recall_recomputed' in r for r in splits)}"]
    L += ["", "## Unresolved provenance issues", ""]
    L += [f"- **{k}.** {v}" for k, v in UNRESOLVED]
    L += ["", "## Checks by exhibit", ""]
    for e in exhibits:
        L.append(f"- `{e.name}`: {len(e.checks)} checks passed, {len(e.issues)} issues")
    (rep / "validation_report.md").write_text("\n".join(L) + "\n")

    S = ["# Result sources", "",
         f"Raw results root: `{config.manifest()['results_root']}`. Run ID = `<path>@<SHA-256[:12]>`.", "",
         "| Run ID | Experiment | Setting/donor | Encoder | Seeds | Exhibits |", "|---|---|---|---|---|---|"]
    for rel, m in sorted(used.items()):
        S.append(f"| `{m['run_id']}` | {m.get('experiment', '')} | {m.get('corpus') or m.get('donor') or ''} | "
                 f"{m.get('encoder', '')} | {m.get('seeds', '')} | {', '.join(sorted(set(m['exhibits'])))} |")
    S += ["", f"## Raw files not used by any exhibit ({len(unused)})", ""]
    groups = defaultdict(list)
    for r in unused:
        groups[r.split("/")[0]].append(r.split("/", 1)[1])
    for g, fs in sorted(groups.items()):
        S.append(f"- `{g}/` ({len(fs)}): " + ", ".join(f"`{f}`" for f in fs))
    (rep / "result_sources.md").write_text("\n".join(S) + "\n")
