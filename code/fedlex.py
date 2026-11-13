"""Fedlex DE/FR/IT (Swiss domestic federal law): the corpus, its whole-law splits, the primary run and the
ten-encoder study.

This is the code of the original Fedlex notebook (`fedlex_minimal_experiments.ipynb`), moved here unchanged
in substance so that `notebooks/03_fedlex.ipynb` only calls it:

* `prepare` reads the SwissLawTranslations article-level parquet files (`data/raw/fedlex/`), keeps domestic
  federal law (SR 1-9 without the 26 cantonal constitutions), drops exact-duplicate texts at triple level,
  draws the seeded 2,784-triple cohort and checks it against `data/fedlex_domestic_n2784_manifest.csv.gz`;
* `law_split` keeps every law (rsNr) whole on one side of the 50/50 article split;
* `run_primary` fits Raw / PCA-1 / PCA-2 / k-means LEACE / Metadata-LEACE / SAE-LEACE on seeds 0-4
  (checkpointed per seed; a checkpoint is reused only when its run hash matches the configuration);
* `encoder_cell` is one encoder of the ten-encoder study (Table B2).

`fedlex_corpus` and `fedlex_split` rebuild the corpus and a seed's split from the manifest and the saved run,
for the Table 3 run on Fedlex (`focused_extensions.load_setting`).
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import time

import numpy as np

import artifacts
import audit
import experiments as E
import paths
import protocols as P
import revised as R
import run_sae_experiment as RSE

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
HF_REPO = "joelniklaus/SwissLawTranslations"          # SwiLTra-Bench's Fedlex translations
HF_CONFIG = "article_level"
HF_REVISION = "03080f362948f9695475a1ece4bb17e33f5c2af6"   # the Hub's main on 2026-10-02 (last commit 2025-03-04)
HF_LAST_MODIFIED = "2025-03-04 08:44:30+00:00"
HF_DIR = os.path.join("data", "raw", "fedlex")        # the parquet files (train.parquet is downloaded on first use)
#: The three parquet files of the configuration at that revision, with their SHA-256.
HF_FILES = {"train": ("article_level/train.parquet", "35b8990a163eca74f7943c820b779db25b51c6336a101030008367cfc27c05fe"),
            "validation": ("article_level/val.parquet", "aa0ca03b357de17d013a912b747e4b4a43ea45a09c0a88d8e67e42e91d297ce3"),
            "test": ("article_level/test.parquet", "19961bcf80a5593e57cfae49237338d6ca801d90901ceffcd40c7a793080be55")}
LANGS = ("de", "fr", "it")
SCOPE_LABEL = "domestic federal law"                  # SR 1-9 without the 26 cantonal constitutions
N_TRIPLES = 2784                                      # downsample to SwiLTra DE/FR/IT's size; None = every clean triple
DATASET_SEED = 42                                     # draws the N_TRIPLES sample (and QUICK's subsample)
QUICK_TRIPLES = 1500


def manifest_path(n_triples=N_TRIPLES) -> str:
    """ids + text hashes of the cohort."""
    return os.path.join("data", f"fedlex_domestic_{'all' if n_triples is None else f'n{n_triples}'}_manifest.csv.gz")


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #
CORPUS = "fedlex_domestic"                            # names the embeddings, checkpoints and result files
ENCODER = "e5-small"                                  # intfloat/multilingual-e5-small, the primary encoder
HOLDOUT = 0.5                                         # share of ARTICLES held out; whole laws on each side
SMOKE_SEED = 0
CONDITIONS = ("raw", "pca_top1", "pca_top2", "kmeans_leace", "metadata_leace", "sae_leace")
PRIMARY = ("raw", "pca_top1", "kmeans_leace", "metadata_leace", "sae_leace")   # the paper's five
LABELS = {**R.LABELS, "pca_top2": "PCA-2", "sae_leace": "SAE-LEACE (N=32)"}
PAIR_LABEL = {"de|fr": "DE-FR", "de|it": "DE-IT", "fr|it": "FR-IT"}
RUN_SEEDS = (0, 1, 2, 3, 4)                           # as the SwiLTra and SCOTUS primary cohorts
NONLINEAR = ("mlp",)                                  # Table B1 reports the MLP AUC

#: The ten-encoder study, as `revised.run_encoder_cell` runs it on the other corpora.
ENCODER_STUDY = R.ENCODER_STUDY
ENC_SEEDS = R.ENCODER_SEEDS                           # (0, 1, 2)
ENC_PCA_RANKS = R.ENCODER_PCA_GRID                    # (1, 2, 3, 4, 8, 16, 32)
ENC_NONLINEAR = ()                                    # the cells are audited linearly
SHARED_WITH_PRIMARY = {"raw": "raw", "pca_top1": "pca_top1", "pca_top2": "pca_top2", "kmeans_auto": "kmeans_leace",
                       "metadata_leace": "metadata_leace", "sae_leace": "sae_leace"}

SPLIT_POLICY = {"scheme": "holdout", "holdout": HOLDOUT, "groups": "rsNr (whole law)",
                "rule": "laws in default_rng(seed).permutation order; held-out side = the prefix whose "
                        "article count is closest to holdout x articles; sides via protocols._group_sides",
                "probe_groups": "article triples (protocols.probe_groups)"}
CODE_FILES = ("protocols.py", "revised.py", "audit.py", "sae.py", "sae_confounder_pipeline.py",
              "run_sae_experiment.py", "experiments.py")
PROMPT = {"e5-small": "query: ", "e5-base": "query: ", "e5-large": "query: "}   # encode_texts' plain prefixes

#: What `focused_extensions` needs to rebuild the corpus of the published run.
FEDLEX = {"corpus": CORPUS, "root": paths.FEDLEX, "langs": LANGS, "holdout": HOLDOUT, "manifest": manifest_path(),
          "embeddings": os.path.join(paths.FEDLEX_CACHE, f"{CORPUS}__{ENCODER}__n{3 * N_TRIPLES}.npy")}


def k_extra() -> dict:
    """`alternatives.run_generous_k`'s K = C and K = 8 arms."""
    import alternatives as A
    return {"kmeans_count": len(LANGS), "kmeans_generous": A.GENEROUS_K}


def enc_conditions() -> tuple:
    return (("raw",) + tuple(f"pca_top{r}" for r in ENC_PCA_RANKS) + ("kmeans_auto",) + tuple(k_extra())
            + ("metadata_leace", "sae_leace"))


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def texts_sha256(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def triple_sha256(df):
    """sha256 of each article's three texts (DE, FR, IT, joined by U+001F)."""
    return [hashlib.sha256("\x1f".join(t).encode("utf-8")).hexdigest() for t in zip(*(df[f"{l}_text"] for l in LANGS))]


def cohort_fingerprint(df):
    """sha256 over the ordered (article_id, triple hash) lines of a cohort table."""
    lines = (a + " " + h for a, h in zip(df.article_id, triple_sha256(df)))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# The corpus
# --------------------------------------------------------------------------- #
def parquet_files(hf_dir: str = HF_DIR) -> tuple:
    """``(paths, provenance)``: the three parquet files, downloaded at `HF_REVISION` only if they are not on
    disk, each checked against its recorded SHA-256."""
    import huggingface_hub
    import pyarrow
    import pyarrow.parquet as pq
    paths_, prov = {}, {"repo": HF_REPO, "config": HF_CONFIG, "revision": HF_REVISION, "last_modified": HF_LAST_MODIFIED,
                        "huggingface_hub": huggingface_hub.__version__, "pyarrow": pyarrow.__version__, "files": {}}
    for split, (rel, sha) in HF_FILES.items():
        p = os.path.join(hf_dir, rel)
        if not os.path.exists(p):
            p = huggingface_hub.hf_hub_download(HF_REPO, rel, repo_type="dataset", revision=HF_REVISION, local_dir=hf_dir)
        got = sha256_file(p)
        if got != sha:
            raise ValueError(f"{p}: SHA-256 {got[:12]}... is not the file of revision {HF_REVISION[:12]} "
                             f"({sha[:12]}...); delete it to download it again")
        paths_[split] = p
        prov["files"][split] = {"path": rel, "rows": pq.ParquetFile(p).metadata.num_rows, "bytes": os.path.getsize(p),
                                "sha256": got}
    return paths_, prov


def prepare(n_triples=N_TRIPLES, dataset_seed: int = DATASET_SEED, quick: bool = False, hf_dir: str = HF_DIR,
            verbose: bool = True) -> dict:
    """The cohort and the document-level corpus, with every check of the original notebook. Returns a dict
    (``ctx``) that the run functions below read."""
    import pandas as pd
    import pyarrow.parquet as pq
    paths_, prov = parquet_files(hf_dir)
    say = print if verbose else (lambda *a, **k: None)
    # The dataset card's article_level fields, and what the experiment needs.
    card_fields = (["abbreviation", "artNr", "lawTitle"]
                   + [f"{l}_{f}" for f in ("artTitle", "artText", "artHtml") for l in ("de", "fr", "it", "rm", "en")])
    needed = ["rsNr", "artNr"] + [f"{l}_artText" for l in LANGS]
    schema = {split: pq.read_schema(p) for split, p in paths_.items()}
    names = schema["train"].names
    assert all(s.names == names for s in schema.values()), "the splits have different columns"
    missing = [c for c in needed if c not in names]
    assert not missing, f"needed columns missing from the parquet: {missing}"
    assert all(str(schema["train"].field(c).type) in ("string", "large_string") for c in needed)
    meta_cols = ("artTitle", "abbreviation", "dateApplicability", "lawTitle", "url")
    cols = ["rsNr", "artNr"] + [f"{l}_{f}" for l in LANGS for f in ("artText",) + meta_cols]
    frames = []
    for split, p in paths_.items():
        t = pq.read_table(p, columns=cols).to_pandas()
        t.insert(0, "hf_split", split)
        t.insert(1, "hf_row", np.arange(len(t)))
        frames.append(t)
    src = pd.concat(frames, ignore_index=True)
    nulls = src[cols].isna().sum()
    assert not nulls[needed].any(), f"null values in needed columns: {nulls[needed][nulls[needed] > 0].to_dict()}"

    for l in LANGS:
        src[f"{l}_text"] = src[f"{l}_artText"].str.strip()
    changed = {l: int((src[f"{l}_text"] != src[f"{l}_artText"]).sum()) for l in LANGS}
    present = pd.DataFrame({l: src[f"{l}_text"] != "" for l in LANGS})
    src["complete"] = present.all(axis=1)
    # The linked-article key.
    assert int(src.duplicated(["rsNr", "artNr"], keep=False).sum()) == 0, "(rsNr, artNr) is not unique"
    assert not (src.rsNr.str.contains(":").any() or src.artNr.str.contains(":").any())
    src["article_id"] = src.rsNr + ":" + src.artNr
    # Scope: the part of the Systematic Compilation each record belongs to.
    intl, cantonal, federal = "international law (SR 0.*)", "cantonal constitutions (SR 131.2)", "domestic federal law"
    src["sr_part"] = np.select([src.rsNr.str.startswith("0."), src.rsNr.str.match(r"131\.2")], [intl, cantonal], federal)
    const = src[src.sr_part == cantonal]
    const_titles = const[const.de_lawTitle != ""].groupby("rsNr").de_lawTitle.first()
    assert set(const_titles.index) == set(const.rsNr) and const_titles.str.startswith("Verfassung").all(), \
        "an SR 131.2 number that is not a cantonal constitution"
    src["in_scope"] = src.sr_part == federal
    complete = src[src.complete & src.in_scope].reset_index(drop=True)

    dup = {}
    for l in LANGS:
        s = complete[f"{l}_text"]
        dup[l.upper()] = {"records whose text occurs more than once": int(s.duplicated(keep=False).sum()),
                          "occurrences beyond the first": int(s.duplicated().sum()),
                          "distinct duplicated texts": int(s[s.duplicated(keep=False)].nunique())}
    lang_of = {}
    for l in LANGS:
        for t in set(complete[f"{l}_text"]):
            lang_of.setdefault(t, set()).add(l)
    dup["cross_language"] = {"texts found in two or more language columns": sum(len(v) > 1 for v in lang_of.values())}
    # Triple-level exact-duplicate removal (first occurrence kept, in file order).
    seen, reason = set(), []
    for tri in zip(*(complete[f"{l}_text"] for l in LANGS)):
        if len(set(tri)) < len(tri):
            reason.append("two identical texts within the record")
        elif any(t in seen for t in tri):
            reason.append("re-uses a text of an earlier record")
        else:
            reason.append("")
            seen.update(tri)
    complete["drop_reason"] = reason
    clean = complete[complete.drop_reason == ""].reset_index(drop=True)
    dropped = complete.drop_reason.value_counts().drop("", errors="ignore").to_dict()
    pooled = pd.concat([clean[f"{l}_text"] for l in LANGS], ignore_index=True)
    assert not pooled.duplicated().any(), "a text occurs twice among the kept documents"
    assert all(len(set(t)) == 3 for t in zip(*(clean[f"{l}_text"] for l in LANGS)))

    cohort_cols = (["article_id", "rsNr", "artNr", "hf_split", "hf_row"] + [f"{l}_abbreviation" for l in LANGS]
                   + ["de_dateApplicability", "de_lawTitle"] + [f"{l}_artTitle" for l in LANGS] + ["de_url"]
                   + [f"{l}_text" for l in LANGS])
    manifest_cols = ["article_id", "rsNr", "artNr", "hf_split", "hf_row", "triple_sha256"]
    if n_triples is None:
        cohort = clean[cohort_cols].reset_index(drop=True)              # every clean triple: no downsampling
    else:
        assert len(clean) >= n_triples, (f"only {len(clean):,} clean complete triples, fewer than {n_triples:,}")
        pick = np.sort(np.random.default_rng(dataset_seed).choice(len(clean), size=n_triples, replace=False))
        cohort = clean.iloc[pick][cohort_cols].reset_index(drop=True)   # downsampled; file order kept
    n_cohort, n_cohort_laws = len(cohort), int(cohort.rsNr.nunique())
    manifest = cohort[manifest_cols[:-1]].assign(triple_sha256=triple_sha256(cohort)).astype(str)
    mpath = manifest_path(n_triples)
    if os.path.exists(mpath):
        saved = pd.read_csv(mpath, dtype=str, keep_default_na=False)
        same = (list(saved.columns) == manifest_cols and saved.shape == manifest.shape
                and bool((saved.to_numpy() == manifest.to_numpy()).all()))
        assert same, (f"{mpath} differs from the cohort rebuilt here (revision {HF_REVISION[:12]}). The manifest "
                      "describes the corpus the results were computed on: find out why before replacing it.")
        say(f"[cohort] rebuilt cohort matches {mpath}: {len(manifest):,} articles, ids and text hashes identical")
    else:
        manifest.to_csv(mpath, index=False, compression={"method": "gzip", "mtime": 0})
        say(f"[cohort] wrote {mpath}: {len(manifest):,} articles")
    if quick and len(cohort) > QUICK_TRIPLES:
        pick = np.sort(np.random.default_rng(dataset_seed).choice(len(cohort), size=QUICK_TRIPLES, replace=False))
        cohort = cohort.iloc[pick].reset_index(drop=True)
        say(f"[QUICK] continuing on a seeded {len(cohort):,}-triple subsample, not the cohort")
    cohort_sha = cohort_fingerprint(cohort)

    n = len(cohort)
    assert n == (min(QUICK_TRIPLES, n_cohort) if quick else n_cohort) and cohort.article_id.is_unique
    for l in LANGS:
        assert (cohort[f"{l}_text"].str.len() > 0).all()
    # The document-level corpus, built as run_sae_experiment.load_swiltra builds SwiLTra DE/FR/IT.
    texts = [t for l in LANGS for t in cohort[f"{l}_text"]]
    source = np.array([l for l in LANGS for _ in range(n)])
    pair_id = np.tile(np.arange(n), len(LANGS))
    texts, source, pairs = RSE._finalise(texts, source, pair_id, np.ones(len(texts), dtype=bool))
    y = np.asarray(source)
    law = np.tile(cohort.rsNr.to_numpy(dtype=str), len(LANGS))            # law of every document
    article = np.tile(cohort.article_id.to_numpy(dtype=str), len(LANGS))  # linked article of every document
    classes, counts = np.unique(y, return_counts=True)
    of_pair, groups = P.link_groups(pairs)
    assert len(texts) == 3 * n and len(pairs) == 3 * n and len(groups) == n
    assert list(classes) == sorted(LANGS) and list(counts) == [n] * 3
    assert all(sorted(y[g]) == sorted(LANGS) and len(set(article[g])) == 1 for g in groups)
    assert len(set(texts)) == len(texts)
    law_size = cohort.groupby("rsNr").size()
    is_intl, is_cantonal = cohort.rsNr.str.startswith("0."), cohort.rsNr.str.match(r"131\.2")
    chars = pd.DataFrame({l.upper(): cohort[f"{l}_text"].str.len() for l in LANGS})
    stats = {
        "scope_label": SCOPE_LABEL, "downsampled": n_triples is not None, "quick_subsample": bool(quick),
        "triples": n, "documents": len(texts), "retrieval_pairs": len(pairs),
        "laws": int(law_size.size), "articles_per_law": {"median": float(law_size.median()),
                                                         "mean": float(law_size.mean()), "max": int(law_size.max())},
        "laws_with_one_article": int((law_size == 1).sum()),
        "international_law_articles": int(is_intl.sum()), "international_law_laws": int(cohort[is_intl].rsNr.nunique()),
        "cantonal_constitution_articles": int(is_cantonal.sum()),
        "domestic_federal_articles": int((~is_intl & ~is_cantonal).sum()),
        "hf_split": cohort.hf_split.value_counts().to_dict(),
        "chars": {c: {"median": float(chars[c].median()), "mean": float(chars[c].mean()),
                      "p95": float(chars[c].quantile(0.95)), "max": int(chars[c].max())} for c in chars},
    }
    assert stats["domestic_federal_articles"] == n, "an article outside domestic federal law entered the cohort"
    ledger = pd.DataFrame([
        ("source rows (train + validation + test)", len(src)),
        *[(f"  out of scope: {p}", int((src.sr_part[~src.in_scope] == p).sum()))
          for p in (intl, cantonal) if (src.sr_part[~src.in_scope] == p).any()],
        (f"in scope: {SCOPE_LABEL}", int(src.in_scope.sum())),
        *[(f"  with a non-empty {l.upper()} text", int((present[l] & src.in_scope).sum())) for l in LANGS],
        ("complete DE/FR/IT records", len(complete)),
        *[(f"  dropped: {k}", v) for k, v in dropped.items()],
        ("clean triples", len(clean)),
        ("  laws among the clean triples", int(clean.rsNr.nunique())),
        ((f"cohort: seeded sample (N_TRIPLES = {n_triples:,}, DATASET_SEED = {dataset_seed})" if n_triples
          else "cohort: every clean triple (no downsampling)"), n_cohort),
        ("  documents (3 per triple)", 3 * n_cohort),
        ("  laws in the cohort", n_cohort_laws),
        *([(f"QUICK smoke-test subsample (DATASET_SEED = {dataset_seed}; not the cohort)", n)] if quick else []),
    ], columns=["step", "count"])
    say(f"{n:,} triples -> {len(texts):,} documents, {len(pairs):,} retrieval pairs, {len(groups):,} linked groups "
        f"of 3, from {stats['laws']:,} laws: all checks passed")
    dataset = {"experiment": "fedlex_dataset", "hf": prov, "schema_columns": names,
               "card_vs_parquet": {"in_parquet_not_on_card": [c for c in names if c not in card_fields],
                                   "on_card_not_in_parquet": [c for c in card_fields if c not in names]},
               "cleaning": {"strip_changed_texts": changed, "rule": "triple-level exact-duplicate removal, "
                            "first occurrence kept in file order; pooled across languages"},
               "scope": {"label": SCOPE_LABEL, "rule": "international law = SR 0.*; "
                         "cantonal constitutions = SR 131.2; domestic federal law = everything else (kept)"},
               "article_id": '"<rsNr>:<artNr>", unique over all source rows',
               "duplicates": dup, "dropped": dropped, "ledger": ledger.to_dict("records"),
               "cohort": (f"seeded sample of {n_triples} clean triples (DATASET_SEED = {dataset_seed}), file order"
                          if n_triples else "every clean triple, file order (no downsampling)"),
               "n_triples": n_cohort, "n_clean_triples": len(clean),
               "manifest": mpath, "quick_subsample": {"n_triples": n, "seed": dataset_seed} if quick else None,
               "cohort_sha256": cohort_sha, "cohort_stats": stats}
    return {"n_triples": n_triples, "dataset_seed": dataset_seed, "quick": quick, "provenance": prov,
            "cohort": cohort, "cohort_sha": cohort_sha, "texts": texts, "texts_sha": texts_sha256(texts), "y": y,
            "pairs": pairs, "law": law, "article": article, "classes": classes, "n": n, "stats": stats,
            "ledger": ledger, "dataset": dataset}


def save_dataset(ctx: dict, overwrite: bool = False) -> dict:
    """The dataset record (`raw/dataset/fedlex_dataset.json`). An existing record is kept when it describes the
    same cohort."""
    path = R.raw_path("dataset", "fedlex_dataset.json")
    if os.path.exists(path) and not overwrite:
        saved = artifacts.load_result(path)
        if saved["cohort_sha256"] != ctx["cohort_sha"]:
            raise ValueError(f"{path} describes another cohort; pass overwrite=True to replace it")
        print(f"[cached] {path}")
        return saved
    artifacts.save_result(ctx["dataset"], path)
    return ctx["dataset"]


def embed_corpus(ctx: dict, encoder: str, device=None) -> tuple:
    """The corpus embedded with `encoder`, cached and fingerprinted: `experiments.embed`, step by step."""
    import torch
    texts, y = ctx["texts"], ctx["y"]
    path = E._cache_path(CORPUS, encoder, len(texts))     # <cache>/<CORPUS>__<encoder>__n<docs>.npy
    meta_path = path[:-len(".npy")] + ".json"
    if os.path.exists(path):
        meta = json.load(open(meta_path))
        assert meta["texts_sha256"] == ctx["texts_sha"], f"{path} was encoded from other texts: delete it and rerun"
        Z = np.load(path)
        print(f"[embed] {CORPUS}/{encoder}: cache hit {Z.shape} <- {path}")
        return Z, meta
    from huggingface_hub import snapshot_download
    from style_erasure_pipeline import MODEL_MAP
    dev = RSE.pick_device(device)
    model = RSE.load_embedding_model(encoder)
    bs = E.ENCODER_BATCH.get(encoder, 256)
    t0 = time.time()
    Z = np.asarray(RSE.encode_texts(model, texts, encoder, dev, bs, show_progress_bar=False), dtype=np.float32)
    seconds = round(time.time() - t0, 1)
    meta = {"encoder": encoder, "model": MODEL_MAP[encoder], "modules": [type(m).__name__ for m in model],
            "max_seq_length": int(model.max_seq_length), "batch_size": bs, "device": dev, "dtype": "float32",
            "prompt": (f'"{PROMPT[encoder]}" on every document' if encoder in PROMPT else
                       "the encoder's own setting in run_sae_experiment.encode_texts"),
            "seconds": seconds, "texts_sha256": ctx["texts_sha"], "n_docs": len(texts),
            "torch": torch.__version__, "environment": platform.platform()}
    try:
        meta["model_revision"] = os.path.basename(snapshot_download(MODEL_MAP[encoder], local_files_only=True).rstrip("/"))
    except Exception:                                   # noqa: BLE001
        meta["model_revision"] = None
    try:                                                # a diagnostic only: never stops the run
        n_tok = np.array([len(ids) for ids in model.tokenizer([PROMPT.get(encoder, "") + t for t in texts],
                                                              add_special_tokens=True)["input_ids"]])
        meta["tokens"] = {"median": float(np.median(n_tok)), "p95": float(np.quantile(n_tok, 0.95)),
                          "share_over_max_seq_length": float((n_tok > model.max_seq_length).mean()),
                          "share_over_by_language": {l: float((n_tok[y == l] > model.max_seq_length).mean())
                                                     for l in LANGS}}
    except Exception as err:                            # noqa: BLE001
        meta["tokens"] = {"error": repr(err)}
    np.save(path, Z)
    json.dump(meta, open(meta_path, "w"), indent=1)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[embed] {CORPUS}/{encoder}: encoded {Z.shape} on {dev} in {seconds}s -> {path}")
    return Z, meta


def data_record(ctx: dict, X, seeds) -> dict:
    """A registry-style corpus entry, as `revised.load` returns it (MI top-N pinned)."""
    label = f"Fedlex, {SCOPE_LABEL} (de/fr/it)"
    return {"key": CORPUS, "label": label, "texts": ctx["texts"], "source": ctx["y"], "pairs": ctx["pairs"],
            "gold": None, "topics": None, "emb": X, "n_docs": len(ctx["texts"]), "n_pairs": len(ctx["pairs"]),
            "paired": True,
            "config": {"confounder": "language", "paired": True, "pseudo_groups": "auto", "seeds": tuple(seeds),
                       "selection": R.SELECTION, "label": label, "has_gold": False,
                       "split": {"scheme": "holdout", "holdout": HOLDOUT}}}


# --------------------------------------------------------------------------- #
# Whole-law splits
# --------------------------------------------------------------------------- #
def law_split(pairs, law_of_doc, seed: int, holdout: float = HOLDOUT):
    """One outer split that keeps every law (rsNr) whole on one side.

    The linked groups are the articles (`protocols.link_groups`). The laws are put in a seeded random order;
    the held-out side is the prefix of that order whose article count is closest to `holdout` of all articles
    (ties to the shorter prefix). `protocols._group_sides` then assembles both sides exactly as
    `protocols._make_group_split` does for a three-class corpus.
    """
    arr = np.asarray(pairs, dtype=int)
    of_pair, groups = P.link_groups(arr)
    law_of_group = np.array([law_of_doc[g[0]] for g in groups])
    laws, size = np.unique(law_of_group, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(laws))
    cum = np.cumsum(size[order])
    n_te = int(np.clip(np.argmin(np.abs(cum - holdout * len(groups))) + 1, 1, len(laws) - 1))
    members = {law: np.flatnonzero(law_of_group == law) for law in laws}
    te_g = np.concatenate([members[law] for law in laws[order[:n_te]]])
    tr_g = np.concatenate([members[law] for law in laws[order[n_te:]]])
    none = np.array([], dtype=int)              # every document is in a triple: nothing unpaired
    tr_docs, te_docs, tr_pairs, te_pairs, te_glob = P._group_sides(arr, groups, of_pair, te_g, tr_g, none, none)
    return P.Split(protocol="document", seed=int(seed), holdout=float(holdout), train_docs=tr_docs,
                   test_docs=te_docs, train_pairs=tr_pairs, test_pairs=te_pairs, test_pairs_global=te_glob)


def check_split(ctx: dict, sp) -> dict:
    """The integrity checks of the design; returns the split's summary."""
    import pandas as pd
    y, law, article, n = ctx["y"], ctx["law"], ctx["article"], ctx["n"]
    tr, te = np.asarray(sp.train_docs), np.asarray(sp.test_docs)
    assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == len(y)
    assert set(law[tr]).isdisjoint(law[te]), "a law is on both sides"
    assert set(article[tr]).isdisjoint(article[te]), "an article is on both sides"
    for side in (tr, te):
        cnt = pd.Series(y[side]).value_counts()
        assert len(cnt) == 3 and cnt.nunique() == 1, f"languages unbalanced: {cnt.to_dict()}"
    n_te_art = len(set(article[te]))
    assert len(sp.test_pairs_global) == len(sp.test_pairs) == 3 * n_te_art
    assert all(article[a] == article[b] and y[a] != y[b] for a, b in sp.test_pairs_global)
    g = P.probe_groups(sp)                                      # what revised.evaluate_fits hands the audit
    grp = pd.DataFrame({"g": g, "article": article[te], "lang": y[te]}).groupby("g")
    assert (grp.size() == 3).all() and (grp.article.nunique() == 1).all() and (grp.lang.nunique() == 3).all()
    folds, k = audit.audit_folds(y[te], g, sp.seed)
    assert all(set(g[a]).isdisjoint(g[b]) for a, b in folds), "an audit fold splits an article triple"
    te_laws = pd.Series(law[te]).value_counts() // 3
    return {"seed": int(sp.seed), "train_articles": len(set(article[tr])), "test_articles": n_te_art,
            "train_laws": len(set(law[tr])), "test_laws": len(set(law[te])),
            "train_docs": int(len(tr)), "test_docs": int(len(te)), "test_pairs": len(sp.test_pairs_global),
            "test_article_share": round(n_te_art / n, 4), "largest_test_law": f"{te_laws.index[0]} ({te_laws.iloc[0]} articles)",
            "audit_folds": int(k), "test_laws_list": sorted(set(law[te]))}


def splits(ctx: dict, seeds) -> tuple:
    """``(splits, split_info)`` for `seeds`, each checked and rebuilt twice (deterministic)."""
    out, info = {}, {}
    for s in sorted(set(seeds)):
        out[s] = law_split(ctx["pairs"], ctx["law"], s)
        info[s] = check_split(ctx, out[s])
        assert law_split(ctx["pairs"], ctx["law"], s).test_docs.tolist() == out[s].test_docs.tolist()
    return out, info


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def geometry_record(classes, X_tr, y_tr, seed):
    """alternatives.run_geometry's per-split record, on a given training split."""
    import alternatives as A
    from sklearn.metrics import adjusted_rand_score
    Xc = np.asarray(X_tr, np.float64)
    Xc = Xc - Xc.mean(axis=0)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    Vt = np.linalg.svd(Xc, full_matrices=False)[2]
    Z = P._one_hot(y_tr, classes).astype(np.float64)
    B = A._orth(Xc.T @ (Z - Z.mean(axis=0)))             # what Metadata-LEACE removes
    capture = np.cumsum(((Vt @ B) ** 2).sum(axis=1)) / B.shape[1]
    K = int(P.choose_pseudo_groups(X_tr, seed=seed)[0])
    ari = {tag: float(adjusted_rand_score(y_tr, R.kmeans_eraser(X_tr, k, seed)[1]))
           for tag, k in (("auto", K), ("count", len(classes)), ("generous", A.GENEROUS_K))}
    return {"k_auto": K, "pc1_var_share": float(lam[0] / lam.sum()),
            "nuisance_var_share": float(((Xc @ B) ** 2).sum() / (Xc ** 2).sum()),
            "capture": {str(r): float(capture[min(int(r), len(capture)) - 1]) for r in A.GEOMETRY_RANKS},
            "rank_90": int(np.searchsorted(capture, 0.90) + 1), "rank_99": int(np.searchsorted(capture, 0.99) + 1),
            "ari": ari}


def _code_sha() -> dict:
    return {f: sha256_file(os.path.join("code", f)) for f in CODE_FILES}


def _environment(fit_device) -> dict:
    import sklearn
    import torch
    return {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
            "scikit-learn": sklearn.__version__, "torch": torch.__version__,
            "sae_device": fit_device or ("cuda" if torch.cuda.is_available() else "cpu")}


# --------------------------------------------------------------------------- #
# The primary run (Tables 2 and B1, Table 3's Fedlex cell)
# --------------------------------------------------------------------------- #
def run_primary(ctx: dict, seeds=RUN_SEEDS, nonlinear=NONLINEAR, device=None, fit_device=None,
                overwrite: bool = False, verbose: bool = True, **overrides) -> dict:
    """Raw, PCA-1, PCA-2 (auxiliary), k-means LEACE (silhouette K), Metadata-LEACE and SAE-LEACE on the
    whole-law splits of `seeds`, with the smoke test of seed 0 and the post-hoc geometry; saved as
    ``raw/id/id_fedlex_domestic_e5-small.json``. Each seed is checkpointed under ``raw/id/checkpoints/``; a
    checkpoint is reused only when its run hash (cohort, texts, embeddings, recipe, conditions, probes, split
    rule, audit) equals this configuration's. An existing result file is loaded unless `overwrite`."""
    import alternatives as A
    from scipy import stats as st
    from sklearn.metrics import adjusted_rand_score
    final = R.raw_path("id", f"id_{CORPUS}_{ENCODER}.json")
    if os.path.exists(final) and not overwrite:
        print(f"[cached] {final}")
        return artifacts.load_result(final)
    y, classes = ctx["y"], ctx["classes"]
    X, emb_meta = embed_corpus(ctx, ENCODER, device=device)
    assert X.shape == (len(ctx["texts"]), 384) and X.dtype == np.float32 and np.isfinite(X).all()
    emb_sha = hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()
    data = data_record(ctx, X, seeds)
    split_of, split_info = splits(ctx, set(seeds) | {SMOKE_SEED})

    # Smoke test: the closed-form conditions on seed 0, before any SAE is trained.
    smoke_conditions = ("raw", "pca_top1", "metadata_leace")
    sp = split_of[SMOKE_SEED]
    X_tr, y_tr = X[sp.train_docs], y[sp.train_docs]
    smoke_fit = R.Erasers(sp, {"raw": lambda Z: np.asarray(Z, np.float32),
                               "pca_top1": P._fit_pca_removal(X_tr, 1)[0],
                               "metadata_leace": P._fit_leace(X_tr, P._one_hot(y_tr, classes), standardise=False)})
    smoke = R.evaluate_fits([smoke_fit], X, y, conditions=smoke_conditions)
    ceiling = smoke["metrics"]["raw"]["recall@10"] >= 0.98

    recipe = {**E.RECIPE, **overrides}
    run_config = {"corpus": CORPUS, "cohort_sha256": ctx["cohort_sha"], "texts_sha256": ctx["texts_sha"],
                  "embeddings_sha256": emb_sha, "encoder": ENCODER, "recipe": recipe, "selection": R.SELECTION,
                  "pseudo_groups": "auto", "conditions": list(CONDITIONS), "nonlinear": list(nonlinear),
                  "split": SPLIT_POLICY, "audit": audit.PROTOCOL}
    run_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True, default=str).encode()).hexdigest()
    code_sha, env = _code_sha(), _environment(fit_device)
    per_seed = {}
    for s in seeds:
        path, sp = R.raw_path("id", os.path.join("checkpoints", f"id_{CORPUS}_{ENCODER}_seed{s}.json")), split_of[s]
        if os.path.exists(path) and not overwrite:
            r = artifacts.load_result(path)
            if r.get("run_hash") != run_hash:
                raise RuntimeError(f"{path} was computed under another configuration (cohort, embeddings, recipe, "
                                   "conditions, split or audit). Pass overwrite=True to recompute, or remove it.")
            changed = [f for f, h in code_sha.items() if r["code_sha256"].get(f) != h]
            print(f"[cached] seed {s} <- {path}" + (f"   NOTE: code/ changed since: {changed}" if changed else ""))
            per_seed[s] = r
            continue
        t0 = time.time()
        _, (fitted,) = R.fit_seed(data, s, splits=[sp], device=fit_device, **overrides)
        tr = np.asarray(sp.train_docs)
        K, sil = P.choose_pseudo_groups(X[tr], seed=sp.seed)               # label-free: training silhouette
        km_eraser, km_labels = R.kmeans_eraser(X[tr], K, sp.seed)
        fitted.erasers["kmeans_leace"] = km_eraser
        fitted.erasers["pca_top2"] = P._fit_pca_removal(X[tr], 2)[0]
        assert K == fitted.config["pseudo_groups_k"], "k-means LEACE and the SAE chose different K"
        r = R.evaluate_fits([fitted], X, y, conditions=CONDITIONS, nonlinear=nonlinear)
        r["kmeans"] = {"k_auto": int(K), "silhouette": {str(k): float(v) for k, v in sil.items()},
                       "ari_with_sae_pseudo_groups": float(adjusted_rand_score(fitted.kmeans.labels_, km_labels))}
        r["sae"] = {"pseudo_groups_k": int(fitted.config["pseudo_groups_k"]), "dict_size": fitted.config["dict_size"],
                    "info": fitted.sae_info}
        r["split_summary"] = split_info[s]
        r.update(seconds=round(time.time() - t0, 1), run_hash=run_hash, code_sha256=code_sha, environment=env)
        artifacts.save_result(r, path)
        per_seed[s] = r
        if verbose:
            print(f"[fedlex] seed {s} ({r['seconds']}s, silhouette K {K}, |S| = {r['n_selected'][0]})")
            for c in CONDITIONS:
                print("     " + R._line(LABELS[c], r["metrics"][c]))
    seeds_done = [s for s in seeds if s in per_seed]
    assert seeds_done == list(seeds)
    if SMOKE_SEED in per_seed:          # the smoke test's three rows must be reproduced
        diff = max(abs(per_seed[SMOKE_SEED]["metrics"][c][k] - smoke["metrics"][c][k])
                   for c in smoke_conditions for k in ("linear_auc", "recall@1", "recall@10"))
        # Exactly in the environment that fitted the seed; to rounding (1e-4) when the checkpoint comes from
        # another machine (the probes' solvers differ in the last digits between platforms).
        ck_env = per_seed[SMOKE_SEED].get("environment", {})
        same = all(ck_env.get(k) == env.get(k) for k in ("platform", "python", "numpy", "scikit-learn", "torch"))
        tol = 1e-6 if same else 1e-4
        assert diff < tol, f"seed {SMOKE_SEED} does not reproduce the smoke test (max difference {diff:.2e})"
        print(f"seed {SMOKE_SEED} reproduces the smoke test for {smoke_conditions} (max |difference| {diff:.1e}"
              + ("" if same else ", the checkpoint was computed in another environment") + ")")

    # Post-hoc geometry of every training split (no method uses it).
    geo = {}
    for s in seeds_done:
        tr = np.asarray(split_of[s].train_docs)
        geo[s] = geometry_record(classes, X[tr], y[tr], s)
        assert geo[s]["k_auto"] == per_seed[s]["kmeans"]["k_auto"], "geometry K differs from k-means LEACE K"
        lab = R.kmeans_eraser(X[tr], geo[s]["k_auto"], s)[1]
        geo[s]["cluster_languages"] = {str(k): {str(l): int(((lab == k) & (y[tr] == l)).sum())
                                                for l in classes if ((lab == k) & (y[tr] == l)).any()}
                                       for k in np.unique(lab)}
        if geo[s]["k_auto"] != len(classes):          # where the silhouette misses K = C: K = C with 1 and 10 starts
            import inspect
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            cap = inspect.signature(P.choose_pseudo_groups).parameters["cap"].default
            idx = np.random.default_rng(s).permutation(len(tr))[:min(cap, len(tr))]
            Xs, ys = X[tr][idx], y[tr][idx]
            geo[s]["k_check"] = {}
            for n_init in ("auto", 10):
                lab_ = KMeans(n_clusters=len(classes), random_state=s, n_init=n_init).fit_predict(Xs)
                geo[s]["k_check"][str(n_init)] = {"silhouette": float(silhouette_score(Xs, lab_)),
                                                  "ari": float(adjusted_rand_score(ys, lab_)),
                                                  "sizes": np.bincount(lab_).tolist()}
    sil_at = lambda s, k: per_seed[s]["kmeans"]["silhouette"].get(str(k), np.nan)    # noqa: E731
    diag = {s: {"silhouette K": g["k_auto"], "silhouette at K = 2": sil_at(s, 2), "silhouette at K = 3": sil_at(s, 3),
                "ARI (silhouette K)": g["ari"]["auto"], "ARI (K = C = 3)": g["ari"]["count"],
                f"ARI (K = {A.GENEROUS_K})": g["ari"]["generous"],
                "capture[1]": g["capture"]["1"], "capture[C-1 = 2]": g["capture"]["2"], "PCs for 90%": g["rank_90"],
                "PC1 variance share": g["pc1_var_share"], "language variance share": g["nuisance_var_share"],
                "k-means = SAE pseudo-groups (ARI)": per_seed[s]["kmeans"]["ari_with_sae_pseudo_groups"],
                "train articles": split_info[s]["train_articles"], "test articles": split_info[s]["test_articles"],
                "train laws": split_info[s]["train_laws"], "test laws": split_info[s]["test_laws"]}
            for s, g in geo.items()}
    keys = list(next(iter(diag.values())))
    geo_agg = {k: R._stat(np.array([diag[s][k] for s in diag], dtype=float)) for k in keys if k != "silhouette K"}
    geo_agg["silhouette K"] = [int(diag[s]["silhouette K"]) for s in diag]

    agg = R.aggregate([per_seed[s] for s in seeds_done])
    metric_keys = [k for k in ("linear_auc", "mlp_auc", "recall@1", "recall@10") if k in agg["metadata_leace"]]

    def paired(c, k, ref="metadata_leace"):
        d = np.array([per_seed[s]["metrics"][c][k] - per_seed[s]["metrics"][ref][k] for s in seeds_done])
        m, sd = float(d.mean()), (float(d.std(ddof=1)) if len(d) > 1 else float("nan"))
        h = float(st.t.ppf(0.975, len(d) - 1) * sd / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        return {"mean": m, "sd": sd, "ci95": [m - h, m + h], "n": int(len(d)), "values": d.tolist()}

    gaps = {c: {k: paired(c, k) for k in metric_keys} for c in CONDITIONS if c != "metadata_leace"}
    res = {"experiment": "fedlex_id", "corpus": CORPUS, "label": data["label"], "encoder": ENCODER, "quick": ctx["quick"],
           "seeds": seeds_done, "conditions": list(CONDITIONS), "primary": list(PRIMARY),
           "labels": {c: LABELS[c] for c in CONDITIONS},
           # The per-pair retrieval hits (about 10^5 per condition and seed) stay in the checkpoints only.
           "per_seed": [{k: v for k, v in per_seed[s].items() if k != "hits"} for s in seeds_done], "agg": agg,
           "paired_to_metadata_leace": gaps, "geometry": {"per_seed": geo, "agg": geo_agg},
           "smoke": {"seed": SMOKE_SEED, "metrics": smoke["metrics"], "ceiling_flag": bool(ceiling)},
           "splits": split_info, "corpus_stats": R._corpus_stats(data),
           "dataset": {"manifest": manifest_path(ctx["n_triples"]), "cohort_sha256": ctx["cohort_sha"],
                       "texts_sha256": ctx["texts_sha"], "hf": ctx["provenance"], "stats": ctx["stats"],
                       "ledger": ctx["ledger"].to_dict("records")},
           "embeddings": {**emb_meta, "sha256": emb_sha, "file": E._cache_path(CORPUS, ENCODER, len(ctx["texts"]))},
           "config": R._config(data, overrides, nonlinear=list(nonlinear), split_groups=SPLIT_POLICY,
                               run_hash=run_hash, dataset_seed=ctx["dataset_seed"])}
    artifacts.save_result(res, final)
    return res


# --------------------------------------------------------------------------- #
# The ten-encoder study (Table B2)
# --------------------------------------------------------------------------- #
def encoder_cell(ctx: dict, encoder: str, seeds=ENC_SEEDS, primary: dict | None = None, device=None,
                 fit_device=None, overwrite: bool = False, verbose: bool = True, **overrides) -> dict:
    """One Fedlex x encoder cell: Raw, PCA ranks 1-32, k-means LEACE (silhouette K, K = C, K = 8),
    Metadata-LEACE and SAE-LEACE on the whole-law splits of `seeds`; resumable per seed (checkpoints with a
    run hash). E5-small's cell must reproduce the primary run on the arms they share."""
    final = R.raw_path("encoders", f"encoder_{CORPUS}_{encoder}.json")
    if os.path.exists(final) and not overwrite:
        print(f"[cached] {final}")
        return artifacts.load_result(final)
    import alternatives as A
    y = ctx["y"]
    Z, meta = embed_corpus(ctx, encoder, device=device)
    z_sha = hashlib.sha256(np.ascontiguousarray(Z).tobytes()).hexdigest()
    data = data_record(ctx, Z, seeds)
    split_of, _ = splits(ctx, seeds)
    recipe, kx, conds = {**E.RECIPE, **overrides}, k_extra(), enc_conditions()
    cfg = {"corpus": CORPUS, "cohort_sha256": ctx["cohort_sha"], "texts_sha256": ctx["texts_sha"],
           "embeddings_sha256": z_sha, "encoder": encoder, "recipe": recipe, "selection": R.SELECTION,
           "pseudo_groups": "auto", "conditions": list(conds), "k_extra": kx, "nonlinear": list(ENC_NONLINEAR),
           "split": SPLIT_POLICY, "audit": audit.PROTOCOL}
    run_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    code_sha, env = _code_sha(), _environment(fit_device)
    per_seed = []
    for s in seeds:
        ck = R.raw_path("encoders", os.path.join("checkpoints", f"encoder_{CORPUS}_{encoder}_seed{s}.json"))
        if os.path.exists(ck) and not overwrite:
            r = artifacts.load_result(ck)
            if r.get("run_hash") != run_hash:
                raise RuntimeError(f"{ck} was computed under another configuration (cohort, embeddings, recipe, arms, "
                                   "split or audit). Pass overwrite=True to recompute, or remove it.")
            print(f"[cached] {encoder} seed {s} <- {ck}")
            per_seed.append(r)
            continue
        sp, t0 = split_of[s], time.time()
        tr = np.asarray(sp.train_docs)
        fitted = P.fit_pipeline(Z, y, sp, **recipe, selection=R.SELECTION, pseudo_groups="auto", pca_grid=(),
                                kmeans_grid=(), kmeans_auto=True, random_control=False, device=fit_device)
        fitted.feats_train = None                      # the SAE activations (n x 8d) are not needed any more
        fitted.erasers["kmeans_auto"] = fitted.erasers.pop("kmeans_leace_auto")
        fitted.erasers.update({f"pca_top{r}": fn for r, fn in A._pca_erasers(Z[tr], ENC_PCA_RANKS).items()})
        fitted.erasers.update({c: R.kmeans_eraser(Z[tr], k, sp.seed)[0] for c, k in kx.items()})
        r = R.evaluate_fits([fitted], Z, y, conditions=conds, nonlinear=ENC_NONLINEAR)
        r.pop("hits", None)                            # as run_encoder_cell: the cells keep no per-pair hits
        r["geometry"] = geometry_record(ctx["classes"], Z[tr], y[tr], s)
        r["kmeans"] = {"k_auto": int(fitted.config["pseudo_groups_k"]),
                       "silhouette": fitted.config["pseudo_group_silhouette"]}
        assert r["geometry"]["k_auto"] == r["kmeans"]["k_auto"], "geometry K differs from the cell K"
        r["sae"] = {"dict_size": int(fitted.config["dict_size"]), "info": fitted.sae_info}
        r.update(seconds=round(time.time() - t0, 1), run_hash=run_hash, code_sha256=code_sha, environment=env)
        artifacts.save_result(r, ck)
        per_seed.append(r)
        if verbose:
            m = r["metrics"]
            print(f"[encoders] {encoder} seed {s} ({r['seconds']}s, K {r['kmeans']['k_auto']}): linear AUC "
                  + ", ".join(f"{c} {m[c]['linear_auc']:.3f}" for c in ("raw", "pca_top1", "kmeans_auto",
                                                                       "metadata_leace", "sae_leace") if c in m))
        del fitted
        gc.collect()
    res = {"experiment": "fedlex_encoder_cell", "corpus": CORPUS, "label": data["label"], "encoder": encoder,
           "dim": int(Z.shape[1]), "seeds": [int(s) for s in seeds], "conditions": list(conds),
           "pca_ranks": [int(r) for r in ENC_PCA_RANKS], "k_extra": kx, "per_seed": per_seed,
           "agg": R.aggregate(per_seed), "diagnostics": R._pc_shares(Z),
           "embeddings": {**meta, "sha256": z_sha}, "corpus_stats": R._corpus_stats(data),
           "config": R._config(data, overrides, nonlinear=list(ENC_NONLINEAR), split_groups=SPLIT_POLICY,
                               run_hash=run_hash, k_extra=kx)}
    if encoder == ENCODER and primary is not None:     # the shared arms must be the primary run's fits
        prim = {int(p["seed"]): p for p in primary["per_seed"]}
        diffs = [abs(p["metrics"][c][k] - prim[p["seed"]]["metrics"][pc][k])
                 for p in per_seed if p["seed"] in prim for c, pc in SHARED_WITH_PRIMARY.items()
                 if c in p["metrics"] for k in ("linear_auc", "recall@1", "recall@10")]
        res["primary_check"] = {"max_abs_diff": max(diffs) if diffs else None, "n_values": len(diffs)}
        if diffs and max(diffs) > 0.005:
            raise RuntimeError(f"the {encoder} cell does not reproduce the primary run (largest difference {max(diffs):.4f})")
    artifacts.save_result(res, final)
    return res


# --------------------------------------------------------------------------- #
# The corpus of the published run, for the Table 3 run on Fedlex
# --------------------------------------------------------------------------- #
def _run_seed(result: dict, seed: int) -> dict:
    return next(p for p in result["per_seed"] if p["seed"] == seed)


def fedlex_corpus() -> tuple:
    """``(data, run)``: the Fedlex corpus as `prepare` builds it (rebuilt from the manifest), checked against
    the published run. `data` holds what `revised.fit_seed`, `revised._refit` and `revised.evaluate_fits` read,
    plus each document's law (``law``) for the whole-law split."""
    import pandas as pd
    run = artifacts.load_result(os.path.join(FEDLEX["root"], "raw", "id", f"id_{CORPUS}_{ENCODER}.json"))
    if run["config"]["recipe"] != E.RECIPE:
        raise ValueError("Fedlex: the run's recipe is not experiments.RECIPE")
    cohort = pd.read_csv(FEDLEX["manifest"])
    n, langs = len(cohort), FEDLEX["langs"]
    source = np.array([lang for lang in langs for _ in range(n)])
    _, source, pairs = RSE._finalise([""] * len(source), source, np.tile(np.arange(n), len(langs)),
                                     np.ones(len(source), dtype=bool))
    X = np.load(FEDLEX["embeddings"])
    if hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest() != run["embeddings"]["sha256"]:
        raise ValueError(f"{FEDLEX['embeddings']}: not the embeddings the run was fitted on")
    data = {"key": CORPUS, "label": "Fedlex DE/FR/IT", "emb": X, "source": np.asarray(source),
            "pairs": pairs, "gold": None, "topics": None, "paired": True, "n_docs": int(len(source)),
            "n_pairs": int(len(pairs)), "law": np.tile(cohort.rsNr.to_numpy(dtype=str), len(langs)),
            "config": {"selection": run["config"]["selection"], "pseudo_groups": run["config"]["pseudo_groups"],
                       "split": run["config"]["split"], "confounder": "language", "seeds": tuple(run["seeds"])}}
    return data, run


def fedlex_split(data: dict, run: dict, seed: int):
    """The run's whole-law split of `seed`, rebuilt; it must hold out exactly the run's laws."""
    sp = law_split(data["pairs"], data["law"], seed)
    ref = _run_seed(run, seed)
    if sorted(set(data["law"][np.asarray(sp.test_docs)])) != sorted(ref["split_summary"]["test_laws_list"]):
        raise ValueError(f"Fedlex seed {seed}: the rebuilt split does not hold out the run's laws")
    return sp
