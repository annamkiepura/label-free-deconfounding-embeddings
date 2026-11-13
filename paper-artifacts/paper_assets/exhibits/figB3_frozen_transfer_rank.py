"""Appendix Figure B3: frozen transfer and erasure-rank sensitivity.

A/B: the five primary donor settings (E5-small CAP, three-source SCOTUS, three-language
SwiLTra; MiniLM CAP, three-source SCOTUS), four transformed methods, change from Raw in
(A) the equal-weight macro nDCG@10 of the five MTEB legal retrieval tasks and (B)
STSBenchmark cosine Spearman. C: the original-donor PCA rank sweeps (CAP, binary SCOTUS
Oyez-Lexis, binary SwiLTra DE-FR), legal macro change against the evaluated ranks.

Every value and interval is the saved `ood.summarize` output: paired percentile bootstrap,
10,000 resamples, seed 0, conditional on the frozen maps (fitted on each donor's ID
training split, seeds 0-2, never refit here). The same resampled queries (STS pairs)
serve every method and seed; the two AILA tasks share their query set and are
resampled jointly by query id; STS Spearman is recomputed per seed on each resample and
then averaged over seeds. The intervals are reproduced from the saved per-query outputs
with that same evaluator (`ood.summarize` on the run files, merged as the project
merges them); the reproduction is cached against the SHA-256 of every input and of
code/ood.py and recomputed when any changes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..lib import config, latex, style
from ..lib.io import read, run_id, sha256
from ..lib.loaders import ood_verify_point, validate_file
from ..lib.methods import ENCODERS, METHODS
from ..lib.provenance import check, issue

NAME, KIND, TITLE = "figB3_frozen_transfer_rank", "figure", "Frozen transfer and erasure-rank sensitivity"
LABEL = "fig:frozen-transfer-rank"
FONT = {"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"}
N_BOOT, LEVEL, BOOT_SEED = 10000, 0.95, 0
STS = "STSBenchmark"
PRIMARY = [("e5-small", "cap", "ood/ood_summary_kmeans.json"),
           ("e5-small", "scotus_oyez_lexis_wiki", "combinations/ood_summary.json"),
           ("e5-small", "swiltra_de_fr_it", "combinations/ood_summary.json"),
           ("mini", "cap", "ood/ood_summary_kmeans.json"),
           ("mini", "scotus_oyez_lexis_wiki", "combinations/ood_summary.json")]
ORIGINAL = [("e5-small", "cap"), ("e5-small", "scotus_oyez_lexis"), ("e5-small", "swiltra"),
            ("mini", "cap"), ("mini", "scotus_oyez_lexis")]
RANKS_SUMMARY = "ood/ood_summary_ranks.json"
TRANSFORMS = ("pca_top1", "kmeans_leace", "metadata_leace", "sae_leace")
STYLE_KEY = {"kmeans_leace": "kmeans_auto"}
LEGEND = {"pca_top1": "PCA-1", "kmeans_leace": "k-means LEACE (silhouette K)",
          "metadata_leace": "Metadata-LEACE", "sae_leace": "SAE-LEACE (N = 32)"}
DONOR = {"cap": "CAP", "scotus_oyez_lexis_wiki": "SCOTUS (3 sources)", "swiltra_de_fr_it": "SwiLTra (3 languages)",
         "scotus_oyez_lexis": "SCOTUS Oyez–Lexis (binary)", "swiltra": "SwiLTra DE–FR (binary)"}
#: Class set each donor must be (three-class primaries vs the original binary versions).
N_CLASSES = {"cap": 3, "scotus_oyez_lexis_wiki": 3, "swiltra_de_fr_it": 3, "scotus_oyez_lexis": 2, "swiltra": 2}
CACHE = config.OUT_DIRS["data"] / f"{NAME}_reproduction.json"
_STATE: dict = {}


# --------------------------------------------------------------------------- #
# Reproduction of the saved intervals with the project's evaluator
# --------------------------------------------------------------------------- #
def _merge(r: dict, e: dict, rename: dict, name: str) -> None:
    """`revised_exhibits._ood_merge`: add `e`'s transforms to `r` once both score the
    same frozen fits (seeds, donor split fingerprints) against the same Raw."""
    ok = (e["seeds"] == r["seeds"]
          and [p["fit"]["split_id"] for p in e["per_seed"]] == [p["fit"]["split_id"] for p in r["per_seed"]]
          and all(e["raw"][t]["score"] == r["raw"][t]["score"] for t in r["raw"]))
    check(ok, "b3_merge_same_fits", f"{name}: same seeds, donor split fingerprints and Raw scores as the "
          f"plan's OOD run")
    for p, q in zip(r["per_seed"], e["per_seed"]):
        for theirs, ours in rename.items():
            p["scores"][ours] = q["scores"][theirs]
    r["transforms"] = list(r["transforms"]) + [v for v in rename.values() if v not in r["transforms"]]


def _inputs(kind: str, enc: str, donor: str) -> tuple[str, str]:
    extra = "ood_kmeans" if kind == "primary" else "ood_ranks"
    return f"ood/ood_{enc}_{donor}.json", f"alternatives/{extra}_{enc}_{donor}.json"


def _rename(kind: str, main: dict, extra: dict) -> dict:
    if kind == "primary":
        return {"kmeans_auto": "kmeans_leace"}
    return {c: c for c in extra["transforms"] if c not in main["transforms"]}      # OOD_EXTRA["ranks"]


def _cache_key(jobs) -> str:
    h = hashlib.sha256((config.REPO / "code" / "ood.py").read_bytes())
    for kind, enc, donor in jobs:
        for rel in _inputs(kind, enc, donor):
            h.update(rel.encode() + sha256(rel).encode())
    h.update(json.dumps([N_BOOT, LEVEL, BOOT_SEED]).encode())
    return h.hexdigest()


def _reproduce(jobs) -> dict:
    key = _cache_key(jobs)
    if CACHE.exists():
        c = json.loads(CACHE.read_text())
        if c.get("key") == key:
            return {**c, "cached": True}
    from ..lib.validate import _project_code
    with _project_code():
        import ood
        out = {}
        for kind, enc, donor in jobs:
            main_rel, extra_rel = _inputs(kind, enc, donor)
            r, e = copy.deepcopy(read(main_rel)), read(extra_rel)
            _merge(r, e, _rename(kind, r, e), extra_rel)
            s = ood.summarize({f"{enc}/{donor}": r}, n_boot=N_BOOT, level=LEVEL, seed=BOOT_SEED, save=False)
            st = s["settings"][f"{enc}/{donor}"]
            out[f"{kind}:{enc}/{donor}"] = {c: {t: st["changes"][c][t] for t in ("legal_macro", STS)}
                                           for c in st["changes"]}
    res = {"key": key, "evaluator": "code/ood.py summarize", "n_boot": N_BOOT, "level": LEVEL,
           "bootstrap_seed": BOOT_SEED, "settings": out}
    CACHE.write_text(json.dumps(res, indent=1))
    return {**res, "cached": False}


# --------------------------------------------------------------------------- #
def _setting(summ: str, enc: str, donor: str) -> dict:
    sm = read(summ)
    check(sm["n_boot"] == N_BOOT and sm["level"] == LEVEL and sm["bootstrap_seed"] == BOOT_SEED,
          "b3_bootstrap_spec", f"{summ}: paired bootstrap, {N_BOOT} resamples, level {LEVEL}, seed {BOOT_SEED}")
    st = sm["settings"][f"{enc}/{donor}"]
    check(st["seeds"] == [0, 1, 2] and st["fit_on"] == "train", "b3_frozen_maps",
          f"{summ} {enc}/{donor}: maps fitted on the donor's ID training split, seeds 0-2, then frozen")
    check(len(config.setting(donor)["classes"]) == N_CLASSES[donor], "b3_donor_version",
          f"{enc}/{donor}: donor has {N_CLASSES[donor]} classes ({DONOR[donor]})")
    check(len(st["legal"]) == 5 and STS in st["sts"], "b3_targets",
          f"{enc}/{donor}: five legal retrieval tasks {st['legal']} and {STS}")
    main_rel, _ = _inputs("primary", enc, donor)
    run = validate_file(main_rel)
    ids = run["tasks"]["AILACasedocs"]["unit_ids"]
    check(ids == run["tasks"]["AILAStatutes"]["unit_ids"], "b3_aila_joint",
          f"{main_rel}: AILACasedocs and AILAStatutes share the same {len(ids)} query ids in the same order, "
          f"so the evaluator draws one joint resample for both")
    check([p["fit"]["split_id"] for p in run["per_seed"]] == st["split_ids"], "b3_split_ids",
          f"{main_rel}: donor split fingerprints equal the summary's")
    return st


def build(ex) -> None:
    style.apply()
    with plt.rc_context(FONT):
        _draw(ex)


def _draw(ex) -> None:
    jobs = [("primary", e, d) for e, d, _ in PRIMARY] + [("ranks", e, d) for e, d in ORIGINAL]
    rep = _reproduce(jobs)
    ab_rows, c_rows = [], []
    # ---------------- data: panels A/B
    prim = {}
    for enc, donor, summ in PRIMARY:
        st = _setting(summ, enc, donor)
        prim[(enc, donor)] = st
        for t in TRANSFORMS:
            extra_rel = _inputs("primary", enc, donor)[1] if t == "kmeans_leace" else _inputs("primary", enc, donor)[0]
            ood_verify_point(summ, f"{enc}/{donor}", t, extra_rel,
                             run_transform="kmeans_auto" if t == "kmeans_leace" else None)
            for task in ("legal_macro", STS):
                ch = st["changes"][t][task]
                rp = rep["settings"][f"primary:{enc}/{donor}"][t][task]
                same = abs(rp["delta"] - ch["delta"]) < 1e-12 and all(abs(a - b) < 1e-12 for a, b in zip(rp["ci"], ch["ci"]))
                check(same, "b3_interval_reproduced", f"{enc}/{donor} {t} {task}: saved change and 95% interval "
                      f"reproduced from the per-query outputs with ood.summarize", saved=ch, reproduced=rp)
                if task == STS:
                    raw = st["scores"]["raw"][task]["mean"]
                    check(abs(np.mean(st["scores"][t][task]["values"]) - raw - ch["delta"]) < 1e-10,
                          "b3_sts_delta", f"{enc}/{donor} {t}: STS change = mean seed Spearman - Raw")
                ab_rows.append({
                    "panel": "A" if task == "legal_macro" else "B", "encoder": enc, "donor": donor,
                    "donor_classes": N_CLASSES[donor], "method": t, "task": task,
                    "tasks_in_score": st["legal"] if task == "legal_macro" else [task],
                    "delta": ch["delta"], "ci_low": ch["ci"][0], "ci_high": ch["ci"][1],
                    "reproduced_delta": rp["delta"], "reproduced_ci_low": rp["ci"][0],
                    "reproduced_ci_high": rp["ci"][1], "interval": "paired percentile bootstrap",
                    "ci_level": LEVEL, "n_boot": N_BOOT, "bootstrap_seed": BOOT_SEED,
                    "raw_score": st["scores"]["raw"][task]["mean"],
                    "seed_scores": st["scores"][t][task]["values"], "seeds": st["seeds"],
                    "donor_split_ids": st["split_ids"], "summary_file": summ,
                    "summary_run_id": run_id(summ, sha256(summ)),
                    "run_files": list(_inputs("primary", enc, donor)),
                    "run_ids": [run_id(p, sha256(p)) for p in _inputs("primary", enc, donor)]})
    # ---------------- data: panel C
    sweep = {}
    for enc, donor in ORIGINAL:
        st = _setting(RANKS_SUMMARY, enc, donor)
        ranks = sorted(int(m[1]) for c in st["changes"] if (m := re.fullmatch(r"pca_top(\d+)", c)))
        extra = read(_inputs("ranks", enc, donor)[1])
        evaluated = sorted({1} | {int(m[1]) for c in extra["transforms"] if (m := re.fullmatch(r"pca_top(\d+)", c))})
        check(ranks == evaluated, "b3_evaluated_ranks", f"{enc}/{donor}: plotted ranks {ranks} are exactly the "
              f"evaluated ones (rank 1 from the plan run, the rest from the rank-sweep run)")
        pts = []
        for r in ranks:
            c = f"pca_top{r}"
            ch = st["changes"][c]["legal_macro"]
            rp = rep["settings"][f"ranks:{enc}/{donor}"][c]["legal_macro"]
            same = abs(rp["delta"] - ch["delta"]) < 1e-12 and all(abs(a - b) < 1e-12 for a, b in zip(rp["ci"], ch["ci"]))
            check(same, "b3_interval_reproduced", f"{enc}/{donor} PCA-{r} legal macro: saved change and 95% "
                  f"interval reproduced with ood.summarize", saved=ch, reproduced=rp)
            pts.append((r, ch["delta"], ch["ci"][0], ch["ci"][1]))
            c_rows.append({"panel": "C", "encoder": enc, "donor": donor, "donor_classes": N_CLASSES[donor],
                           "label": "original-donor rank sweep", "pca_rank": r, "method": c,
                           "delta_legal_macro_ndcg10": ch["delta"], "ci_low": ch["ci"][0], "ci_high": ch["ci"][1],
                           "reproduced_delta": rp["delta"], "reproduced_ci_low": rp["ci"][0],
                           "reproduced_ci_high": rp["ci"][1], "interval": "paired percentile bootstrap",
                           "ci_level": LEVEL, "n_boot": N_BOOT, "bootstrap_seed": BOOT_SEED,
                           "seeds": st["seeds"], "donor_split_ids": st["split_ids"],
                           "seed_scores": st["scores"][c]["legal_macro"]["values"],
                           "summary_file": RANKS_SUMMARY, "summary_run_id": run_id(RANKS_SUMMARY, sha256(RANKS_SUMMARY)),
                           "run_files": list(_inputs("ranks", enc, donor)),
                           "run_ids": [run_id(p, sha256(p)) for p in _inputs("ranks", enc, donor)]})
        sweep[(enc, donor)] = pts
    # ---------------- figure
    fig = plt.figure(figsize=(style.TEXT, 6.0))
    gs = fig.add_gridspec(2, 5, height_ratios=[2.35, 1.55], left=0.235, right=0.985, top=0.945, bottom=0.08,
                          hspace=0.85, wspace=0.12)
    axA = fig.add_subplot(gs[0, 0:2])
    axB = fig.add_subplot(gs[0, 2:5], sharey=axA)
    # A/B use the top row; B is wider only by gridspec arithmetic -- equalise explicitly.
    pa, pb = axA.get_position(), axB.get_position()
    w = (pb.x1 - pa.x0 - 0.035) / 2
    axA.set_position([pa.x0, pa.y0, w, pa.height])
    axB.set_position([pa.x0 + w + 0.035, pb.y0, w, pb.height])
    n = len(PRIMARY)
    for ax, task, title, xlabel in ((axA, "legal_macro", "A  Legal retrieval (5 tasks)",
                                     "Δ macro nDCG@10 vs Raw"),
                                    (axB, STS, "B  Semantic similarity", "Δ Spearman vs Raw (STSBenchmark)")):
        ax.axvline(0, color="#000000", linewidth=0.7, linestyle="--", zorder=1)
        for yi, (enc, donor, _) in enumerate(PRIMARY):
            st = prim[(enc, donor)]
            for ti, t in enumerate(TRANSFORMS):
                ch = st["changes"][t][task]
                mt = METHODS[STYLE_KEY.get(t, t)]
                y = yi + (ti - 1.5) * 0.19
                lo, hi = ch["ci"]
                ax.errorbar(ch["delta"], y, xerr=[[ch["delta"] - lo], [hi - ch["delta"]]], fmt=mt.marker,
                            color=mt.color, markersize=4.6 if t != "kmeans_leace" else 5.4, elinewidth=0.9,
                            capsize=1.8, zorder=3)
        for yi in range(1, n):
            ax.axhline(yi - 0.5, color="#9A9A9A" if yi == 3 else "#E2E2E2", linewidth=0.8 if yi == 3 else 0.5,
                       zorder=0)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
        ax.tick_params(labelsize=7.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(False)
    axA.set_ylim(n - 0.5, -0.5)
    axA.set_yticks(range(n), [f"{ENCODERS[e]}\n{DONOR[d]}" for e, d, _ in PRIMARY], fontsize=7.5)
    axB.tick_params(axis="y", labelleft=False, length=0)
    handles = [Line2D([], [], color=METHODS[STYLE_KEY.get(t, t)].color, marker=METHODS[STYLE_KEY.get(t, t)].marker,
                      linestyle="none", markersize=5, label=LEGEND[t]) for t in TRANSFORMS]
    handles.append(Line2D([], [], color="#444444", marker="|", linestyle="-", markersize=6, linewidth=0.9,
                          label="paired 95% CI (bootstrap)"))
    y_leg = axA.get_position().y0 - 0.075
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7.4,
               bbox_to_anchor=(0.54, y_leg), columnspacing=1.0, handletextpad=0.4)
    # Panel C: one small multiple per original donor, shared y.
    axes_c = [fig.add_subplot(gs[1, i]) for i in range(len(ORIGINAL))]
    for a in axes_c[1:]:
        a.sharey(axes_c[0])
    col = METHODS["pca_top1"].color
    for a, (enc, donor) in zip(axes_c, ORIGINAL):
        pts = np.asarray(sweep[(enc, donor)])
        a.axhline(0, color="#000000", linewidth=0.7, linestyle="--", zorder=1)
        a.plot(pts[:, 0], pts[:, 1], color=col, linewidth=1.0, zorder=2)
        a.errorbar(pts[:, 0], pts[:, 1], yerr=[pts[:, 1] - pts[:, 2], pts[:, 3] - pts[:, 1]], fmt="s",
                   color=col, markersize=3.4, elinewidth=0.8, capsize=1.5, zorder=3)
        a.set_xscale("log", base=2)
        a.set_xlim(0.8, 40)
        a.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
        a.minorticks_off()
        a.tick_params(labelsize=6.8)
        a.set_title(f"{ENCODERS[enc]}\n{DONOR[donor].replace(' (binary)', '')}"
                    + ("\n(binary)" if N_CLASSES[donor] == 2 else "\n(3 sources)"), fontsize=7.2)
        a.grid(False)
    for a in axes_c[1:]:
        a.tick_params(axis="y", labelleft=False)
    axes_c[0].set_ylabel("Δ macro nDCG@10\nvs Raw (legal)", fontsize=7.5)
    x0, x1 = axes_c[0].get_position().x0, axes_c[-1].get_position().x1
    fig.text(x0, axes_c[0].get_position().y1 + 0.075, "C  Original-donor rank sweeps: PCA removal rank r",
             fontsize=9, fontweight="bold", ha="left")
    fig.text((x0 + x1) / 2, 0.018, "PCA removal rank r (log₂ scale; evaluated ranks only)", ha="center", fontsize=7.5)
    caption = (
        r"Frozen transfer and erasure-rank sensitivity. Every map is fitted once on the donor's in-distribution "
        r"training split (fitting seeds 0--2) and applied unchanged to the MTEB targets; nothing is refitted on a "
        r"target. (A) Change from Raw in the equal-weight mean nDCG@10 over the five legal retrieval tasks "
        r"(AILACasedocs, AILAStatutes, LegalBenchConsumerContractsQA, LegalBenchCorporateLobbying, "
        r"LegalSummarization); (B) change in STSBenchmark cosine Spearman. Donors are the primary settings: CAP "
        r"(three sources), three-source SCOTUS and three-language SwiLTra with E5-small, and CAP and three-source "
        r"SCOTUS with MiniLM. Points are changes in the seed-averaged score; bars are paired 95\% percentile "
        r"bootstrap intervals (10{,}000 resamples, conditional on the frozen maps). The same resampled queries or "
        r"sentence pairs serve every method and seed; the two AILA tasks share their 50 queries and are resampled "
        r"jointly by query ID; STS Spearman is recomputed for each seed within every resample and then averaged "
        r"over seeds. (C) Original-donor rank sweeps: legal macro change against the number $r$ of removed "
        r"principal components, for the original donors CAP, binary SCOTUS Oyez--Lexis and binary SwiLTra DE--FR "
        r"(not the three-class settings of A--B), at the evaluated ranks only, with the same paired bootstrap "
        r"intervals. $k$-means LEACE uses the training-silhouette $K$; SAE-LEACE $N{=}32$; only Metadata-LEACE uses "
        r"confounder labels. An interval that includes zero does not show that a method is harmless or "
        r"equivalent to Raw.")
    style.save(fig, NAME, ex)
    latex.write_figure_include(NAME, caption, LABEL, ex, wide=True)
    latex.write_csv(f"{NAME}_panels_ab", ab_rows, ex)
    latex.write_csv(f"{NAME}_panel_c", c_rows, ex)
    meta = {"interval": "paired percentile bootstrap, conditional on the frozen maps", "evaluator": "code/ood.py summarize",
            "n_boot": N_BOOT, "level": LEVEL, "bootstrap_seed": BOOT_SEED,
            "legal": "per query: seed-averaged transformed nDCG@10 minus Raw; queries resampled with one draw per "
                     "distinct query-id set (AILACasedocs and AILAStatutes share one), reused for every method; "
                     "task means averaged with equal weight on each resample",
            "sts": "sentence pairs resampled (one draw, rng seed bootstrap_seed + 1, reused for every method and "
                   "seed); Spearman recomputed per seed on each resample, averaged over seeds, minus the Raw "
                   "resample Spearman",
            "maps": "fitted on each donor's ID training split (revised.run_id), seeds 0-2, frozen; not refit",
            "reproduction": {"cache_file": str(CACHE.relative_to(config.HOME)), "cache_key_sha256": rep["key"],
                             "cached": rep["cached"]}}
    p = config.OUT_DIRS["data"] / f"{NAME}_interval_metadata.json"
    p.write_text(json.dumps(meta, indent=1))
    ex.output(p)
    ex.output(CACHE)
    _STATE.update(ab=ab_rows, c=c_rows, rep=rep, sweep=sweep)


def finalize(ex) -> None:
    S = _STATE
    L = [f"# Figure B3 ({NAME}): provenance and validation", "",
         f"Build: `python -m paper_assets build {NAME}`. Reporting only: no donor map fitted, no target re-encoded.", "",
         "## Sources", ""]
    for enc, donor, summ in PRIMARY:
        L.append(f"- A/B {ENCODERS[enc]} · {DONOR[donor]}: `{summ}` + `{_inputs('primary', enc, donor)[0]}` + "
                 f"`{_inputs('primary', enc, donor)[1]}` (k-means arm)")
    for enc, donor in ORIGINAL:
        L.append(f"- C {ENCODERS[enc]} · {DONOR[donor]}: `{RANKS_SUMMARY}` + `{_inputs('ranks', enc, donor)[0]}` + "
                 f"`{_inputs('ranks', enc, donor)[1]}`; ranks {[p[0] for p in S['sweep'][(enc, donor)]]}")
    rep = S["rep"]
    L += ["", "## Intervals", "",
          f"- Paired percentile bootstrap, {N_BOOT} resamples, level {LEVEL}, seed {BOOT_SEED}, conditional on the "
          "frozen maps (all read from the saved summaries).",
          f"- Reproduced from the per-query run files with `ood.summarize` "
          f"({'loaded from the cache' if rep['cached'] else 'recomputed this build'}; cache key "
          f"`{rep['key'][:16]}…` = SHA-256 of code/ood.py and every input run file). Every plotted change and "
          "interval equals the reproduction to 1e-12 (checks `b3_interval_reproduced`).",
          "- AILA: both tasks list the same 50 query ids in the same order in every run file, so the evaluator "
          "draws one joint resample for them.",
          "- STS: per-seed Spearman recomputed within every resample and averaged over seeds; endpoints are never "
          "averaged.",
          "- Panel C uses the same bootstrap intervals (no seed SD is shown anywhere).", "",
          "## Values", "", "| Panel | Encoder | Donor | Method | Δ | 95% CI |", "|---|---|---|---|---:|---|"]
    for r in S["ab"]:
        L.append(f"| {r['panel']} | {r['encoder']} | {r['donor']} | {r['method']} | {r['delta']:+.4f} | "
                 f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] |")
    for r in S["c"]:
        L.append(f"| C | {r['encoder']} | {r['donor']} | PCA-{r['pca_rank']} | {r['delta_legal_macro_ndcg10']:+.4f} | "
                 f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] |")
    L += ["", f"## Checks ({len(ex.checks)} passed)", ""] + [f"- {c['message']}" for c in ex.checks]
    L += ["", "## Issues", ""] + ([f"- **{i['level']}** `{i['code']}`: {i['message']}" for i in ex.issues] or ["None."])
    p = config.OUT_DIRS["reports"] / f"{NAME}_validation.md"
    p.write_text("\n".join(L) + "\n")
    ex.output(p)
