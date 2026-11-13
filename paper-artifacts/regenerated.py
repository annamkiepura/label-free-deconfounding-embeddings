"""Tables 2, A1, A2 and B1 and Figures 1, 2 and B3, built with the first-draft exhibit code
(`paper_assets/exhibits/`) on the five-seed results.

Each exhibit runs unchanged under the build profile (`common.profile`: the manifest and the output
directories of paper-artifacts/). What an exhibit states for its three-seed cohort is adjusted here,
explicitly: a module constant (Figure 1's seed list), a loader (Figure 2's geometry file) or a caption
sentence, each replaced by exact match so that a change in the original cannot pass unnoticed.
"""
from __future__ import annotations

import copy
import csv
import dataclasses
import importlib
import json
import re
import sys
import types

import numpy as np

import common as C
import tables_md
from paper_assets.lib import config, io, loaders
from paper_assets.lib.loaders import RANDOM_BAND_DRAWS, RANDOM_BAND_POOL
from paper_assets.lib.provenance import check

FIG1_SETTINGS = ("swiltra", "swiltra_de_fr_it", "scotus_oyez_lexis", "scotus_oyez_lexis_wiki")


# --------------------------------------------------------------------------- #
# Derived input: the five-seed random-latent files of Figure 1
# --------------------------------------------------------------------------- #
def derive_band_slots() -> list:
    """The band control at five seeds in the schema `lib.loaders.random_band` reads (the schema of
    the first draft's ``random_band/`` files, which `focused_additions.export_band` wrote for seeds 0-2).
    A reformatting of the merged band runs: nothing is recomputed. A file is written only if both
    seed batches ran on the coverage splits and on the SAEs of the coverage run of the same seed."""
    out = []
    for key in FIG1_SETTINGS:
        merged = json.loads((C.REPO / C.FIVE / "band" / f"band_{key}_e5-small.json").read_text())
        batches = [json.loads((C.REPO / root / "band" / f"band_{key}_e5-small.json").read_text())["reference_check"]
                   for root in (C.FOCUSED, C.EXT)]
        if not all(b["splits_match"] and b["mi_reproduced_everywhere"] for b in batches):
            raise ValueError(f"{key}: a band batch is not on its coverage run's splits and SAEs; no file written")
        per_seed = []
        for q in merged["per_seed"]:
            m = q["metrics"]
            met = {f"sae_random_band_n{n}": m[f"rband_mean_n{n}"] for n in merged["budgets"]}
            met.update({f"sae_random_band_n{n}_draw{j}": m[f"rband{j}_n{n}"]
                        for n in merged["budgets"] for j in range(merged["draws"])})
            met["sae_random_band"] = m["rband_mean_n32"]
            per_seed.append({"seed": q["seed"], "split": q["split"], "metrics": met})
        slot = {"experiment": "random_band", "corpus": key, "encoder": "e5-small", "label": merged["label"],
                "seeds": merged["seeds"], "budgets": merged["budgets"], "draws": merged["draws"],
                "condition": "sae_random_band",
                "definition": (f"Directions-LEACE on N SAE decoder rows drawn uniformly from the latents MI scores "
                               f"(training firing rate in {RANDOM_BAND_POOL}); per seed the mean over "
                               f"{merged['draws']} draws"),
                "same_sae_as_published": True, "per_seed": per_seed, "config": merged["config"],
                "corpus_stats": merged["corpus_stats"], "design": merged["design"],
                "reference_check": {"splits_match": True, "mi_reproduced_everywhere": True,
                                    "batches": {"seeds_0_2": batches[0], "seeds_3_4": batches[1]},
                                    "note": "each batch was checked against the coverage run of its own seeds: seeds "
                                            "0-2 against the published run, seeds 3-4 against the seed extension's"},
                "source": f"{C.FIVE}/band/band_{key}_e5-small.json"}
        path = C.DERIVED / "random_band" / f"random_band_{key}_e5-small.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(slot, indent=1) + "\n")
        out.append(path)
    return out


# --------------------------------------------------------------------------- #
# Running an original exhibit under the profile
# --------------------------------------------------------------------------- #
def _run(name: str, *, before=None, captions=(), expected=None, after=None, module: str | None = None,
         out: str | None = None) -> dict:
    """Build the original exhibit `name` into this folder. `before(mod)` adjusts the module and returns
    an undo function; `captions` are exact caption replacements; `expected` maps the code of a check
    that cannot hold here to ``(why, substitute)``, the substitute being checked in its place; `out` renames
    the outputs (the manuscript's file name)."""
    mod = importlib.import_module(f"paper_assets.exhibits.{module or name}")
    undo = before(mod) if before else None
    try:
        with C.Exhibit(mod.NAME, mod.KIND, mod.TITLE) as ex:
            mod.build(ex)
            for code, (why, substitute) in (expected or {}).items():
                hit = [i for i in ex.issues if i["code"] == code]
                if not hit:
                    raise ValueError(f"{name}: check {code!r} was expected not to hold here, but it did")
                ex.issues[:] = [i for i in ex.issues if i["code"] != code]
                ex.note(f"Check `{code}` of the original exhibit does not apply here: {why}")
                substitute(ex)
            C.fix_outputs(name, mod.KIND)
            if captions:
                C.patch_caption(name, mod.KIND, captions(ex) if callable(captions) else captions, ex)
            if after:
                after(ex, mod)
            if out:
                C.rename_outputs(name, out, mod.KIND, ex)
                name = out
            if mod.KIND == "table":
                C.preview(name, ex)
                tables_md.write(name, ex)
            if hasattr(mod, "finalize"):
                mod.finalize(ex)
                rep = C.HERE / "reports" / f"{name}_validation.md"
                if rep.exists():
                    rep.write_text(rep.read_text().replace(f"python -m paper_assets build {name}", f"{C.COMMAND} {name}"))
            return C.finish(ex, name)
    finally:
        if undo:
            undo()


def tableA1() -> dict:
    return _run("tableA1_dataset_composition")


def tableA2() -> dict:
    return _run("tableA2_encoders")


def table2() -> dict:
    return _run("table2_primary_id", out="table_2_primary_id", captions=[
        (r"SCOTUS with E5-small uses the final three-seed coverage cohort. ",
         r"SCOTUS with E5-small uses the coverage experiment, and SemEval and CAP their primary runs, each extended "
         r"from seeds 0--2 by seeds 3--4. ")])


def tableB1() -> dict:
    def substitute(ex):
        files = sorted({s["file"] for c in ex.cells for s in c["sources"]
                        if c["setting"] in ("scotus_oyez_lexis", "scotus_oyez_lexis_wiki")})
        check(all(f.startswith(f"{C.FIVE}/alternatives/") for f in files), "b1_coverage_five_seeds",
              f"SCOTUS Oyez-Lexis and Oyez-Lexis-Wikipedia: every cell comes from the five-seed coverage run ({files})")
    return _run("tableB1_extended_id", out="tableB1_extended_id_block",
                expected={"b1_coverage_only": ("it tests for the published coverage file's path; the five-seed coverage "
                                               f"files live under {C.FIVE}/", substitute)},
                captions=[(r"; SCOTUS, the three-seed coverage cohort of Table~\ref{tab:primary-id}, ",
                           r"; SCOTUS, its five-seed runs, ")])


def fig1() -> dict:
    def before(mod):
        old = mod.SEEDS
        mod.SEEDS = list(C.SEEDS5)
        return lambda: setattr(mod, "SEEDS", old)

    def captions(ex):
        with (C.HERE / "data" / "fig1_rank_composition_summary.csv").open() as f:
            sd = max(float(r["sd"]) for r in csv.DictReader(f) if r["curve"] == "sae" and r["metric"] == "recall@10")
        return [(r"Points are means over seeds 0--2 of the coverage experiment (the random curve: a rerun on its "
                 r"splits and SAEs)",
                 r"Points are means over seeds 0--4 of the coverage experiment (the random curve: a rerun on its "
                 r"splits and SAEs)"),
                ("up to 0.19 in R@10", f"up to {sd:.2f} in R@10")]
    return _run("fig1_rank_composition", before=before, captions=captions)


def _geometry5(setting: str, encoder: str):
    """`lib.loaders.geometry` on the five-seed geometry run of the setting."""
    rel = f"{C.FIVE}/alternatives/geometry_{setting}_{encoder}.json"
    d = loaders.validate_file(rel)
    check(d.get("corpus") == setting and d.get("encoder") == encoder and d["seeds"] == C.SEEDS5, "file_identity",
          f"{rel}: the geometry run of {setting}/{encoder}, seeds {C.SEEDS5}")

    def per_seed(get):
        return [float(np.mean([get(f) for f in p["folds"]])) for p in d["per_seed"]]

    out = {"file": rel, "run_id": io.run_id(rel, io.sha256(rel)), "seeds": [int(p["seed"]) for p in d["per_seed"]],
           "n_classes": d["n_classes"], "class_rank": d["class_rank"],
           "capture": {r: per_seed(lambda f, r=r: f["capture"][r]) for r in d["agg"]["capture"]},
           "ari": {k: per_seed(lambda f, k=k: f["ari"][k]) for k in d["agg"]["ari"]},
           "k_auto": [[int(f["k_auto"]) for f in p["folds"]] for p in d["per_seed"]],
           "rank_90": per_seed(lambda f: f["rank_90"])}
    bad = [f"capture[{r}]" for r, v in out["capture"].items() if abs(np.mean(v) - d["agg"]["capture"][r]) > 1e-12]
    bad += [f"ari[{k}]" for k, v in out["ari"].items() if abs(np.mean(v) - d["agg"]["ari"][k]) > 1e-12]
    check(not bad, "geometry_agg", f"{rel}: saved geometry means equal the per-seed fold means", bad=bad)
    return out


def fig2() -> dict:
    def before(mod):
        scatter = importlib.import_module("paper_assets.exhibits.fig2_geometry_discovery_scatter")
        old = scatter.geometry
        scatter.geometry = _geometry5
        return lambda: setattr(scatter, "geometry", old)

    def after(ex, mod):
        # The standalone panels' include files point at paper_assets/figures/.
        for panel in mod.PANELS.values():
            p = C.HERE / "figures" / f"{panel}.include.tex"
            if p.exists():
                C._fix_paths(p, mod.NAME)
    return _run("fig2_geometry_discovery", before=before, after=after, captions=[
        (r"A and B use seeds 0--2 of the geometry run; C uses each setting's performance cohort (SCOTUS/E5-small: "
         r"coverage seeds 0--2; SwiLTra and MiniLM: seeds 0--4), so the two do not always share a seed population. ",
         r"A, B and C all use seeds 0--4: the geometry run and each setting's performance cohort, both extended "
         r"from seeds 0--2 where they had only three. ")])


# --------------------------------------------------------------------------- #
# Appendix Figure B3 on seeds 0-4: its five-seed inputs, then the original exhibit on them
# --------------------------------------------------------------------------- #
OOD = "ood"                                              # under this update's derived/
PRIMARY = [("e5-small", "cap"), ("e5-small", "scotus_oyez_lexis_wiki"), ("e5-small", "swiltra_de_fr_it"),
           ("mini", "cap"), ("mini", "scotus_oyez_lexis_wiki")]
ORIGINAL = [("e5-small", "cap"), ("e5-small", "scotus_oyez_lexis"), ("e5-small", "swiltra"), ("mini", "cap"),
            ("mini", "scotus_oyez_lexis")]
RANK_EXT = "results/extensions/rank_sweep_seeds_3_4/raw"
N_BOOT, LEVEL, BOOT_SEED = 10000, 0.95, 0
SUMMARY = {"primary": "ood_summary_primary_five_seeds.json", "ranks": "ood_summary_ranks_five_seeds.json"}


def _derived(name: str) -> str:
    """Repository-relative path of a file under this update's derived/ood/."""
    return f"{C.REL}/derived/{OOD}/{name}"


def _run_files(kind: str, enc: str, donor: str) -> tuple:
    """(main run, run that adds transforms to it) of one Figure B3 setting on seeds 0-4."""
    if kind == "primary":                                 # the main run with the matched-PCA map added to it
        return _derived(f"ood_primary_{enc}_{donor}.json"), f"{C.FIVE}/alternatives/ood_kmeans_{enc}_{donor}.json"
    rel = _derived(f"ood_ranks_{enc}_{donor}.json")       # the rank sweep holds its own Raw scores: nothing to add
    return rel, rel


# --------------------------------------------------------------------------- #
# Derived inputs of Figure B3
# --------------------------------------------------------------------------- #
def derive_transfer(force: bool = False) -> list:
    """The five-seed inputs of Figure B3: per original donor the rank sweep of seeds 0-2 (published) and 3-4
    (the extension) merged, and the two `ood.summarize` summaries (paired bootstrap over queries, as the
    saved three-seed summaries). Skipped when the files exist and their inputs have not changed."""
    out_dir = C.HERE / "derived" / OOD
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = ([f"{C.MAIN}/alternatives/ood_ranks_{e}_{d}.json" for e, d in ORIGINAL]
              + [f"{RANK_EXT}/alternatives/ood_ranks_{e}_{d}.json" for e, d in ORIGINAL]
              + [f"{C.FIVE}/{sub}_{e}_{d}.json" for e, d in PRIMARY
                 for sub in ("ood/ood", "alternatives/ood_kmeans", "alternatives/ood_ranks")] + ["code/ood.py"])
    key = io.hashlib.sha256("".join(io.hashlib.sha256((C.REPO / f).read_bytes()).hexdigest() for f in inputs).encode()
                            + json.dumps([N_BOOT, LEVEL, BOOT_SEED]).encode()).hexdigest()
    stamp = out_dir / "derived_key.json"
    files = ([out_dir / f"ood_ranks_{e}_{d}.json" for e, d in ORIGINAL] + [out_dir / f"ood_primary_{e}_{d}.json" for e, d in PRIMARY]
             + [out_dir / v for v in SUMMARY.values()])
    if not force and stamp.exists() and json.loads(stamp.read_text()).get("key") == key and all(f.exists() for f in files):
        return files
    from paper_assets.lib.validate import _project_code
    with _project_code():
        import focused_extensions as FX
        import ood
        ranks = {}
        for enc, donor in ORIGINAL:
            pub = json.loads((C.REPO / f"{C.MAIN}/alternatives/ood_ranks_{enc}_{donor}.json").read_text())
            ext = json.loads((C.REPO / f"{RANK_EXT}/alternatives/ood_ranks_{enc}_{donor}.json").read_text())
            sweep = [c for c in pub["transforms"] if re.fullmatch(r"pca_top\d+", c)]
            if sweep != [c for c in ext["transforms"] if re.fullmatch(r"pca_top\d+", c)]:
                raise ValueError(f"{enc}/{donor}: the two seed batches swept different ranks")
            m = FX._merge_ood_seeds(pub, ext, f"ood_ranks_{enc}_{donor}", transforms=sweep)
            m["merged"]["sources"] = [f"{C.MAIN}/alternatives/ood_ranks_{enc}_{donor}.json",
                                      f"{RANK_EXT}/alternatives/ood_ranks_{enc}_{donor}.json"]
            (out_dir / f"ood_ranks_{enc}_{donor}.json").write_text(json.dumps(m, indent=1) + "\n")
            ranks[f"{enc}/{donor}"] = m
        primary = {}
        for enc, donor in PRIMARY:
            main, matched, extra = (json.loads((C.REPO / f"{C.FIVE}/{sub}_{enc}_{donor}.json").read_text())
                                    for sub in ("ood/ood", "alternatives/ood_ranks", "alternatives/ood_kmeans"))
            for other, what in ((matched, "matched-PCA"), (extra, "k-means")):
                if (other["seeds"] != main["seeds"] or [p["fit"]["split_id"] for p in other["per_seed"]]
                        != [p["fit"]["split_id"] for p in main["per_seed"]]):
                    raise ValueError(f"{enc}/{donor}: the {what} transfer run is not on the main run's seeds and splits")
            if FX._raw_diff(main["raw"], matched["raw"]) > FX.RAW_TOL:
                raise ValueError(f"{enc}/{donor}: the matched-PCA run scored other target embeddings than the main run")
            r = copy.deepcopy(main)
            for p_, q in zip(r["per_seed"], matched["per_seed"]):
                p_["scores"]["pca_matched_var"] = q["scores"]["pca_matched_var"]
                p_["fit"].update({k: q["fit"][k] for k in ("matched_rank", "sae_removed_variance")})
            r["transforms"] = list(r["transforms"]) + ["pca_matched_var"]
            r["labels"] = {**r.get("labels", {}), "pca_matched_var": matched["labels"]["pca_matched_var"]}
            r["merged"]["matched_pca"] = f"{C.FIVE}/alternatives/ood_ranks_{enc}_{donor}.json"
            (out_dir / f"ood_primary_{enc}_{donor}.json").write_text(json.dumps(r, indent=1) + "\n")
            r = copy.deepcopy(r)
            for p_, q in zip(r["per_seed"], extra["per_seed"]):
                p_["scores"]["kmeans_leace"] = q["scores"]["kmeans_auto"]
            r["transforms"] = list(r["transforms"]) + ["kmeans_leace"]
            primary[f"{enc}/{donor}"] = r
        for kind, runs in (("primary", primary), ("ranks", ranks)):
            s = ood.summarize(runs, n_boot=N_BOOT, level=LEVEL, seed=BOOT_SEED, save=False)
            s["derived"] = {"by": f"{C.REL}/regenerated.py derive_transfer", "evaluator": "code/ood.py summarize",
                            "run_files": {k: list(_run_files(kind, *k.split("/"))) for k in runs}}
            (out_dir / SUMMARY[kind]).write_text(json.dumps(s, indent=1, default=float) + "\n")
    stamp.write_text(json.dumps({"key": key, "inputs": inputs}, indent=1) + "\n")
    return files


def figB3() -> dict:
    """The original exhibit code on the five-seed inputs: its file names, the seed check of its summary
    reader and its reproduction cache are pointed at this update, and the caption's seed range is replaced.

    Panels A-B also show PCA matched to SAE-LEACE's displacement, which the 2026-10-06 transfer table held;
    that table is not built any more. The original module draws four markers per donor at fixed offsets, so
    it is run from a copy of its source in which that one line spaces any number of markers."""
    import new_tables as T
    derive_transfer()
    name = "figB3_frozen_transfer_rank"
    summary = {k: _derived(v) for k, v in SUMMARY.items()}
    source = config.PKG / "exhibits" / f"{name}.py"
    src, fixed = source.read_text(), "                y = yi + (ti - 1.5) * 0.19\n"
    if src.count(fixed) != 1:
        raise ValueError(f"{source.name}: the marker-offset line of panels A-B has changed")
    variant = types.ModuleType(f"paper_assets.exhibits.{name}_matched")
    variant.__package__, variant.__file__ = "paper_assets.exhibits", str(source)
    sys.modules[variant.__name__] = variant
    exec(compile(src.replace(fixed, "                y = yi + (ti - (len(TRANSFORMS) - 1) / 2) * 0.76 / len(TRANSFORMS)\n"),
                 str(source), "exec"), variant.__dict__)

    def before(mod):
        saved = {k: getattr(mod, k) for k in ("PRIMARY", "RANKS_SUMMARY", "CACHE", "_inputs", "_setting", "TRANSFORMS",
                                              "LEGEND", "METHODS")}
        mod.TRANSFORMS = tuple(mod.TRANSFORMS) + ("pca_matched_var",)
        mod.LEGEND = {**mod.LEGEND, "pca_matched_var": "PCA, displacement-matched"}
        mod.METHODS = {**mod.METHODS, "pca_matched_var": dataclasses.replace(mod.METHODS["pca_matched_var"], marker="^")}
        mod.PRIMARY = [(e, d, summary["primary"]) for e, d in PRIMARY]
        mod.RANKS_SUMMARY = summary["ranks"]
        mod.CACHE = C.HERE / "data" / f"{name}_reproduction.json"
        mod._inputs = _run_files

        def setting(summ, enc, donor):
            kind = "ranks" if summ == summary["ranks"] else "primary"
            sm = mod.read(summ)
            check(sm["n_boot"] == mod.N_BOOT and sm["level"] == mod.LEVEL and sm["bootstrap_seed"] == mod.BOOT_SEED,
                  "b3_bootstrap_spec", f"{summ}: paired bootstrap, {mod.N_BOOT} resamples, level {mod.LEVEL}, seed {mod.BOOT_SEED}")
            st = sm["settings"][f"{enc}/{donor}"]
            check(st["seeds"] == C.SEEDS5 and st["fit_on"] == "train", "b3_frozen_maps",
                  f"{summ} {enc}/{donor}: maps fitted on the donor's ID training split, seeds 0-4, then frozen")
            check(len(config.setting(donor)["classes"]) == mod.N_CLASSES[donor], "b3_donor_version",
                  f"{enc}/{donor}: donor has {mod.N_CLASSES[donor]} classes ({mod.DONOR[donor]})")
            check(len(st["legal"]) == 5 and mod.STS in st["sts"], "b3_targets",
                  f"{enc}/{donor}: five legal retrieval tasks {st['legal']} and {mod.STS}")
            run_rel = _run_files(kind, enc, donor)[0]
            run = mod.read(run_rel)
            ids = run["tasks"]["AILACasedocs"]["unit_ids"]
            check(ids == run["tasks"]["AILAStatutes"]["unit_ids"], "b3_aila_joint",
                  f"{run_rel}: AILACasedocs and AILAStatutes share the same {len(ids)} query ids in the same order, so the "
                  "evaluator draws one joint resample for both")
            check([p["fit"]["split_id"] for p in run["per_seed"]] == st["split_ids"], "b3_split_ids",
                  f"{run_rel}: donor split fingerprints equal the summary's")
            check(run["seeds"] == C.SEEDS5 and run.get("merged", {}).get("seed_batches") == [[0, 1, 2], [3, 4]],
                  "b3_five_seed_merge", f"{run_rel}: seeds 0-2 and 3-4 merged")
            return st
        mod._setting = setting
        return lambda: [setattr(mod, k, v) for k, v in saved.items()]

    def captions(ex):
        ranks = {}
        for enc, donor in PRIMARY:
            run = json.loads((C.REPO / _run_files("primary", enc, donor)[0]).read_text())
            ranks[(enc, donor)] = T._rng([p["fit"]["matched_rank"] for p in run["per_seed"]])
        short = {"cap": "CAP", "scotus_oyez_lexis_wiki": "SCOTUS", "swiltra_de_fr_it": "SwiLTra"}
        text = "; ".join(("E5-small " if enc == "e5-small" else "MiniLM ")
                         + ", ".join(f"{short[d]} {ranks[(e, d)]}" for e, d in PRIMARY if e == enc)
                         for enc in dict.fromkeys(e for e, _ in PRIMARY))
        return [(r"training split (fitting seeds 0--2) and applied unchanged to the MTEB targets",
                 r"training split (fitting seeds 0--4) and applied unchanged to the MTEB targets"),
                (r"only Metadata-LEACE uses confounder labels. ",
                 r"only Metadata-LEACE uses confounder labels. Displacement-matched PCA removes the fewest leading principal "
                 r"components whose displacement of the donor's training embeddings reaches that seed's SAE-LEACE map (ranks: "
                 + text + r"). ")]

    def after(ex, mod):
        """The numbers of the former transfer table: checked against the figure's, then saved with the count of
        fitted maps that raise or lower each score, which the figure does not show."""
        rows = T.transfer_rows()
        sm = json.loads((C.REPO / summary["primary"]).read_text())["settings"]
        task = {"legal macro nDCG@10": "legal_macro", "STSBenchmark Spearman": "STSBenchmark"}
        worst = max(max(abs(r["delta"] - sm[f"{r['encoder']}/{r['donor']}"]["changes"][r["map"]][task[r["target"]]]["delta"]),
                        *(abs(a - b) for a, b in zip((r["ci_low"], r["ci_high"]),
                                                     sm[f"{r['encoder']}/{r['donor']}"]["changes"][r["map"]][task[r["target"]]]["ci"])))
                    for r in rows)
        check(worst < 1e-9, "b3_matches_saved_transfer", f"{T.TRANSFER}: its changes and bootstrap intervals for the five "
              f"maps, donors and both targets equal the figure's (largest difference {worst:.1e})")
        from paper_assets.lib import latex
        latex.write_csv(f"{name}_seed_signs", rows, ex)

    return _run(name, before=before, captions=captions, after=after, module=f"{name}_matched")


#: In build order: Table B1 compares its cells with the figures' data files, so it comes after them.
EXHIBITS = {"tableA1_dataset_composition": tableA1, "tableA2_encoders": tableA2, "table_2_primary_id": table2,
            "fig1_rank_composition": fig1, "fig2_geometry_discovery": fig2, "tableB1_extended_id_block": tableB1,
            "figB3_frozen_transfer_rank": figB3}
