"""Appendix Table A2: encoder configurations and evaluation coverage.

The table is rendered from a structured configuration record (data/tableA2_encoders.json)
built here from what the runs recorded, never typed by hand:

* model identifier, dimension, pooling and normalization: `inventory/encoder_inventory.json`
  (the Sentence-Transformers modules each model loads), cross-checked against the code's
  MODEL_MAP and the `dim` saved in every result file;
* revision: the `protocol.encoder_model` an OOD run saved (resolved when it loaded the
  model), or the inventory's model-cache snapshot where the inventory says the runs used
  it. The inventory's current-Hub fallback is NOT a run revision and is reported as not
  recorded;
* prompts: the inventory's document prompt (the rule in `run_sae_experiment.encode_texts`)
  for in-distribution documents, and the OOD runs' saved `target_prompts` for OOD targets;
* coverage: a scan of every saved result file's `encoder`, `experiment` and setting.

One row per distinct configuration: an encoder used with different prompts (E5-small: ID
documents vs OOD targets) gets one row per prompt setting.
"""
from __future__ import annotations

import ast
import json
from collections import defaultdict

from ..lib import config, latex
from ..lib.io import read, run_id, sha256
from ..lib.methods import ENCODERS
from ..lib.provenance import check, issue

NAME, KIND, TITLE = "tableA2_encoders", "table", "Encoder configurations and evaluation coverage"
LABEL = "tab:encoders"
INVENTORY = "inventory/encoder_inventory.json"
SHORT = {**ENCODERS, "mxb": "MXB"}            # manuscript short names (draft Table 5)
NR = "not recorded"
OOD_EXPERIMENTS = {"ood", "ood_kmeans", "ood_ranks"}
STUDY_EXPERIMENTS = {"revised_encoder_cell", "generous_k"}
_STATE: dict = {}


def _registry():
    def const(path, name):
        tree = ast.parse((config.REPO / "code" / path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
                return ast.literal_eval(node.value)
        raise KeyError(name)
    return const("experiments.py", "PARENT_PAPER_ENCODERS"), const("style_erasure_pipeline.py", "MODEL_MAP")


def _scan() -> dict:
    """Every saved result file that names an encoder: {encoder: [(family, experiment, setting, rel)]}."""
    root = config.results_root()
    out = defaultdict(list)
    for p in sorted(root.rglob("*.json")):
        rel = str(p.relative_to(root))
        if rel.startswith("inventory/"):
            continue
        try:
            head = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(head, dict) or not head.get("encoder"):
            continue
        d = read(rel)
        exp, enc = d.get("experiment", "?"), d["encoder"]
        setting = d.get("corpus") or d.get("donor") or "?"
        if exp in OOD_EXPERIMENTS:
            fam = "ood"
        elif exp in STUDY_EXPERIMENTS or (exp == "geometry" and enc not in ("e5-small", "mini")):
            fam = "study"
        else:
            fam = "id"
        out[enc].append({"family": fam, "experiment": exp, "setting": setting, "file": rel,
                         "run_id": run_id(rel, sha256(rel)), "dim": d.get("dim"),
                         "protocol": d.get("protocol") if fam == "ood" else None})
    return out


def _record() -> list:
    order, model_map = _registry()
    inv = read(INVENTORY)
    runs = _scan()
    check(set(runs) == set(order), "a2_encoders_in_runs",
          f"the saved runs use exactly the ten registry encoders ({sorted(runs)})")
    configs = []
    for enc in order:
        e = inv["encoders"][enc]
        rs = runs[enc]
        check(e["model"] == model_map[enc], "a2_model_id", f"{enc}: inventory model id {e['model']} = code MODEL_MAP")
        dims = {r["dim"] for r in rs if r["dim"] is not None}
        check(dims <= {e["dim"]} and dims, "a2_dim", f"{enc}: every result file that saves `dim` has {e['dim']} "
              f"({sum(r['dim'] is not None for r in rs)} files)")
        cached = e["revision_source"].startswith("model-cache snapshot")
        hub_config = not cached
        ood = [r for r in rs if r["family"] == "ood"]
        ood_revs = {json.dumps(r["protocol"]["encoder_model"], sort_keys=True) for r in ood if r["protocol"]}
        ood_prompts = {json.dumps(r["protocol"]["target_prompts"], sort_keys=True) for r in ood if r["protocol"]}
        if ood:
            check(len(ood_revs) == 1 and len(ood_prompts) == 1, "a2_ood_consistent",
                  f"{enc}: all {len(ood)} OOD files record one model revision and one target-prompt setting")
            rec = json.loads(next(iter(ood_revs)))
            check(rec["model"] == e["model"], "a2_ood_model", f"{enc}: OOD runs record model {rec['model']}")
            if cached:
                check(rec["revision"] == e["revision"], "a2_ood_revision_matches_cache",
                      f"{enc}: OOD-recorded revision {rec['revision'][:12]} = the inventory's cache snapshot")
        doc_prompt = e["prompt"]
        id_like = [r for r in rs if r["family"] in ("id", "study")]

        def coverage(rows):
            fam = defaultdict(set)
            for r in rows:
                fam[r["family"]].add(r["setting"])
            return {k: sorted(v) for k, v in fam.items()}

        base = {"short_name": SHORT[enc], "registry_key": enc, "model": e["model"], "dimension": e["dim"],
                "pooling": e["pooling"], "normalization": "L2 (Normalize module)" if e["normalize"]
                else "none (no Normalize module)",
                "pooling_normalization_source": ("inventory: modules of the cached snapshot used by the runs"
                                                 if cached else "inventory: modules of the current Hub "
                                                 "configuration (not run-attributed)"),
                "max_length": NR, "max_length_note": "max_seq_length is not set in code (model default); the "
                                                     "value was not saved by any run",
                "trust_remote_code": True, "similarity": "cosine"}
        # Configuration 1: in-distribution documents (primary ID, ten-encoder study, OOD donor maps).
        cfg = {**base, "configuration": "ID documents",
               "document_prompt": doc_prompt, "query_prompt": "same as document (symmetric: every query is a "
                                                              "document)",
               "revision": e["revision"] if cached else NR,
               "revision_source": ("inventory: model-cache snapshot attributed to these runs; not saved in the ID "
                                   "or encoder-study result files" if cached else
                                   "not recorded (inventory holds only the current Hub revision "
                                   f"{e['revision'][:12]}, excluded)"),
               "coverage": coverage(id_like), "ood_donor_maps": sorted({r["setting"] for r in ood}),
               "files": [(r["file"], r["run_id"]) for r in id_like]}
        if not cached:
            issue("warning", "a2_revision_not_recorded", f"{enc}: model revision not recorded by any run "
                  f"(inventory fallback {e['revision'][:12]} is the current Hub revision)")
        configs.append(cfg)
        # Configuration 2: OOD target encoding, if its prompts differ from the ID document prompt.
        if ood:
            tp = json.loads(next(iter(ood_prompts)))
            same = all(v == "" for v in tp.values()) and doc_prompt == "none"
            target = {**base, "configuration": "OOD targets",
                      "document_prompt": "none" if tp["document"] == "" else repr(tp["document"]),
                      "query_prompt": "none" if tp["query"] == "" else repr(tp["query"]),
                      "sts_prompt": "none" if tp["sts"] == "" else repr(tp["sts"]),
                      "revision": rec["revision"],
                      "revision_source": "recorded by every OOD run (protocol.encoder_model)",
                      "coverage": {"ood_targets": ["5 MTEB legal retrieval tasks", "STSBenchmark"]},
                      "ood_donors": sorted({r["setting"] for r in ood}),
                      "files": [(r["file"], r["run_id"]) for r in ood]}
            if same:
                # Same (empty) prompt as the ID documents: one configuration, OOD recorded on it.
                cfg["revision_source"] = ("recorded by every OOD run (protocol.encoder_model); for the ID and "
                                          "encoder-study runs, the inventory's cache snapshot (same hash)")
                cfg["coverage"]["ood_targets"] = target["coverage"]["ood_targets"]
                cfg["files"] += target["files"]
                cfg["merged_with_ood_targets"] = "OOD target prompts are empty, identical to the ID setting"
            else:
                configs.append(target)
    return configs


# --------------------------------------------------------------------------- #
#: In-distribution document prompts, shown as a mark on the encoder name (defined in the caption).
PROMPT_MARK = {"none": "", '"query: " on every document': "$^a$", 'prompt_name="passage"': "$^b$"}
PRIMARY = ("mini", "e5-small")


def build(ex) -> None:
    configs = _record()
    # One line per encoder: its in-distribution configuration. The OOD-target configuration differs
    # only in its prompts, which are empty for every encoder (checked here, stated in the caption).
    # A configuration merged with its OOD targets was merged only because their recorded prompts are empty.
    separate = [c for c in configs if c["configuration"] == "OOD targets"]
    merged = [c for c in configs if c.get("merged_with_ood_targets")]
    check(bool(separate or merged) and all(c[k] == "none" for c in separate
                                           for k in ("document_prompt", "query_prompt", "sts_prompt")),
          "a2_ood_targets_unprompted", f"OOD targets are encoded without any prompt for every OOD encoder "
          f"({sorted({c['short_name'] for c in separate + merged})}), as the caption states")
    id_cfgs = [c for c in configs if c["configuration"] == "ID documents"]
    check(set(PROMPT_MARK) >= {c["document_prompt"] for c in id_cfgs}, "a2_prompt_marks",
          f"every in-distribution document prompt has a caption mark ({sorted({c['document_prompt'] for c in id_cfgs})})")
    primary = [c for c in id_cfgs if c["registry_key"] in PRIMARY]
    study = [c for c in id_cfgs if c["registry_key"] not in PRIMARY]
    rows = primary + study

    def line(c):
        return [c["short_name"] + PROMPT_MARK[c["document_prompt"]],
                r"{\scriptsize " + latex.esc(c["model"]) + "}", str(c["dimension"])]
    body = [line(c) for c in primary] + [r"\midrule"] + [line(c) for c in study]
    head = [["Encoder", "Hugging Face model", "Dim."]]
    # One column wide: \footnotesize tabular, identifiers in roman \scriptsize (Courier would not fit).
    lines = [r"{\footnotesize%"] + latex.tabular("@{}llr@{}", head, body)
    lines[-1] += "}"
    caption = (
        r"Encoders (Sentence-Transformers, \texttt{trust\_remote\_code}; cosine similarity). Above the rule: also "
        r"used in the primary experiments and OOD transfer; below: ten-encoder study only. In-distribution "
        r"documents (and so queries) carry $^a$the prefix \texttt{query:\textvisiblespace} or $^b$the named "
        r"prompt \texttt{passage}, otherwise no prompt; OOD targets never do. Maximum sequence length: model "
        r"default, not recorded. Recorded model revisions are in the accompanying data file.")
    latex.write_table(NAME, lines, caption, LABEL, ex, wide=False, size=r"\small", tabcolsep="4pt")
    rec = [{k: v for k, v in c.items() if k != "files"} for c in configs]
    p = config.OUT_DIRS["data"] / f"{NAME}.json"
    p.write_text(json.dumps({"source": "built by paper_assets/exhibits/tableA2_encoders.py from the saved runs",
                             "configurations": rec}, indent=1))
    ex.output(p)
    flat = []
    for c in rec:
        flat.append({**{k: v for k, v in c.items() if not isinstance(v, (dict, list))},
                     "coverage_id_settings": c["coverage"].get("id", []),
                     "coverage_study_settings": c["coverage"].get("study", []),
                     "coverage_ood_targets": c["coverage"].get("ood_targets", []),
                     "ood_donor_maps": c.get("ood_donor_maps", []), "ood_donors": c.get("ood_donors", [])})
    latex.write_csv(NAME, flat, ex)
    _STATE.update(configs=configs, inventory=read(INVENTORY))


def finalize(ex) -> None:
    configs = _STATE["configs"]
    L = [f"# Table A2 ({NAME}): configurations, sources and run IDs", "",
         f"Build: `python -m paper_assets build {NAME}`. The LaTeX is rendered from `data/{NAME}.json`, which this "
         "build derives from the saved runs. Run ID = `<path>@<sha256 prefix>`.", "",
         f"- Inventory: `{INVENTORY}` ({run_id(INVENTORY, sha256(INVENTORY))})",
         "- Code: `code/style_erasure_pipeline.py` (MODEL_MAP, `SentenceTransformer(id, trust_remote_code=True)`, "
         "no revision pin); `code/run_sae_experiment.py` `encode_texts` (document prompts); `code/ood.py` "
         "(`PROMPT_STYLE = \"parent\"`, target prompts, `encoder_revision`).", ""]
    for c in configs:
        L += [f"## {c['short_name']} — {c['configuration']}", ""]
        for k in ("model", "revision", "revision_source", "dimension", "pooling", "normalization",
                  "pooling_normalization_source", "document_prompt", "query_prompt", "sts_prompt", "max_length",
                  "max_length_note", "coverage", "ood_donor_maps", "ood_donors", "merged_with_ood_targets"):
            if k in c:
                L.append(f"- {k}: {c[k]}")
        L.append(f"- source files ({len(c['files'])}):")
        L += [f"  - {rid}" for _, rid in c["files"]]
        L.append("")
    L += ["## Not recorded (listed, not filled)", ""]
    for c in configs:
        miss = [k for k in ("revision", "max_length") if c.get(k) == NR]
        if c["pooling_normalization_source"].endswith("(not run-attributed)"):
            miss.append("run-attributed pooling/normalization (current configuration shown)")
        if miss:
            L.append(f"- {c['short_name']} ({c['configuration']}): {', '.join(miss)}")
    L += ["- No run records the Sentence-Transformers or transformers version used for the ID and encoder-study "
          "embeddings; OOD runs record the mteb version (2.20.11).",
          "", f"## Checks ({len(ex.checks)} passed)", ""] + [f"- {c['message']}" for c in ex.checks]
    L += ["", "## Issues", ""] + ([f"- **{i['level']}** `{i['code']}`: {i['message']}" for i in ex.issues] or ["None."])
    p = config.OUT_DIRS["reports"] / f"{NAME}_provenance.md"
    p.write_text("\n".join(L) + "\n")
    ex.output(p)
