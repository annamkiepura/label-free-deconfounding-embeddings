# When Does Label-Free Deconfounding of Document Embeddings Work?

Code, data and results for the paper *When Does Label-Free Deconfounding of Document Embeddings Work?*
<!-- TODO before publishing: add the pre-print link here, e.g. "Paper: https://arxiv.org/abs/XXXX.XXXXX". -->

Document embeddings often encode a confounder: the language a text is written in, or the source it comes from.
The paper asks when such a confounder can be erased **without its labels**, and compares label-free erasers
against Metadata-LEACE, which uses the labels:

* **PCA-1**: removing the first principal component;
* **k-means LEACE**: LEACE against k-means pseudo-groups, with K chosen by the training silhouette;
* **SAE-LEACE**: a TopK sparse autoencoder (SAE) on the embeddings; the 32 latents with the highest mutual
  information with the k-means pseudo-groups; LEACE on their decoder directions;
* controls that take SAE-LEACE apart: random latents, random non-SAE directions, other erasure operators, and PCA
  matched to SAE-LEACE's displacement.

Each eraser is audited for leakage (the AUC of probes trained to recover the confounder on held-out documents) and
for its cost (retrieval of a document's translation or counterpart, topic purity): in distribution, across ten
encoders, and in frozen transfer to six MTEB tasks.

This repository reproduces every table and figure of the paper, at two levels: from the saved raw results in
minutes, or by rerunning every experiment.

## Quick start: rebuild the paper's tables and figures

```bash
git clone https://github.com/annamkiepura/label-free-deconfounding-embeddings.git
cd label-free-deconfounding-embeddings
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python paper-artifacts/build.py
```

This reads the raw results committed under `results/` and rebuilds every table (`paper-artifacts/tables/*.tex`)
and figure (`paper-artifacts/figures/*.pdf`), together with the numbers behind each one (`paper-artifacts/data/`).
It takes a few minutes on a laptop and needs no GPU, no embeddings and no data downloads. The rebuilt files are identical to
the committed ones (the figures pixel for pixel). See [Reproducing the paper](#reproducing-the-paper) to rerun the
experiments themselves.

## Requirements

* **Python 3.12** (3.13 works too) and the packages in `requirements.txt`. Use the pinned `scikit-learn==1.6.1` and
  `numpy==2.1.3`: the probe audits (their cross-validation folds and solvers) depend on them, and they are the
  versions of the published runs.
* **To rerun the experiments**, in addition:
  * internet access on first use, to download the ten encoders from Hugging Face (roughly 10 GB) and three large
    raw data files that are not in the repository (about 450 MB; see [Data](#data));
  * a CUDA GPU: the published runs used a Colab A100 (80 GB); see the run times below;
  * about 2 GB of disk for the embedding cache (`embedding_cache/`, created by the notebooks).
* `pdflatex` (optional): the build uses it to compile a preview PDF of each table; without it the previews are
  skipped.

## Reproducing the paper

There are two ways to reproduce the paper's tables and figures.

### A. Rebuild them from the saved results

The raw results of every run are committed in `results/`, so the tables and figures can be rebuilt without
rerunning any experiment:

```bash
python paper-artifacts/build.py               # or run notebooks/06_paper_artifacts.ipynb
python paper-artifacts/build.py list          # the exhibits
python paper-artifacts/build.py <name>        # rebuild one
```

The build writes everything under `paper-artifacts/`: the tables and figures, the numbers behind them, and their
provenance. Every exhibit records the checks it ran (seeds, splits, configurations, reproductions) and every results
file it read (with its SHA-256) in `paper-artifacts/data/<name>.provenance.json`; the build fails on any error.
`paper-artifacts/manifest.toml` says which saved run feeds which cell of which table.

### B. Rerun the experiments

The experiments are run by six notebooks, in order. Every step saves its result as soon as it finishes and
**reloads the saved file instead of recomputing it** when the file exists. With the results of this repository in
place, the notebooks therefore run in minutes and recompute nothing. To rerun the experiments, move the saved results
aside first:

```bash
mv results results_published                  # keep the published runs to compare with
```

then run the notebooks in order (01 to 06); notebook 06 rebuilds the tables and figures from the new results. A
notebook can be split over several sessions, since every finished step is saved. To rerun a single experiment
instead, delete its file under `results/`; the steps that read it, such as the five-seed merges of notebook 04,
are then rerun the same way.

| Notebook | What it does | Results | Time (from scratch) |
|---|---|---|---|
| `01_data.ipynb` | Checks that CAP is rebuilt byte for byte from its sources; writes the dataset and encoder inventories | `results/main/raw/inventory/` | minutes |
| `02_main_experiments.ipynb` | The in-distribution comparison (Raw, PCA-1, k-means LEACE, Metadata-LEACE, SAE-LEACE) on every setting; sweeps over erasure rank and latent budget; the geometry of the confounder subspace; frozen transfer to the MTEB tasks; the ten-encoder study on the original corpora | `results/main/` | about 30 h on a GPU |
| `03_fedlex.ipynb` | The Fedlex corpus (`code/fedlex.py`): its cohort, primary run and ten-encoder study | `results/fedlex/` | about 16 h on a GPU |
| `04_extensions.ipynb` | Table 3's controls on six settings; other SAE constructions; the ten-encoder study on the three-class settings; displacement-matched PCA in frozen transfer; seeds 3–4 and the five-seed merges | `results/extensions/` | about 40 h on a GPU |
| `05_latent_selectors.ipynb` | Latent selection rules that need no k-means (Table 3, rows A5–A9); SAEs fitted on a CPU | `results/selectors/` | under an hour on a CPU |
| `06_paper_artifacts.ipynb` | Builds every table and figure (`paper-artifacts/build.py`) | `paper-artifacts/` | minutes |

The times are the sums of the per-run timings saved in the results (mostly on a Colab A100). Embeddings are computed
on first use and cached under `embedding_cache/` (not in git), together with the frozen maps the transfer runs reuse.

**What to expect.** With the pinned `scikit-learn` and `numpy`, the conditions that use no SAE (Raw, PCA, k-means
LEACE, Metadata-LEACE) are reproduced exactly. SAE training is deterministic within one environment but not across
hardware, so the conditions that use an SAE come out close to the published values, not identical. To see what
changed, compare the rebuilt files with the committed ones (`git diff --stat paper-artifacts/`) and the new raw
results with `results_published/`; the build's checks report any value outside their tolerances.

**Smoke test.** Set `QUICK = True` in the second code cell of notebooks 02–05 to run every section end to end in
minutes on a subset (two seeds, a 20-epoch SAE, a few settings). QUICK runs write to `results/*_quick/` and never
touch the real results; notebook 04's QUICK run reads the QUICK results of 02 and 03. They test the code, not the
numbers: the tables are built from the full results only.

### Running on Google Colab

The published runs were made on Colab. Clone the repository into your Google Drive (so that results persist across
sessions), for instance from a Colab cell:

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/annamkiepura/label-free-deconfounding-embeddings.git /content/drive/MyDrive/label-free-deconfounding-embeddings
%cd /content/drive/MyDrive/label-free-deconfounding-embeddings
```

Then open the notebooks from that folder. The first code cell of each notebook mounts Drive, moves to `REPO` (set
it if your copy is elsewhere) and installs the requirements; outside Colab it does nothing.

## How the experiments are organized

* **Settings.** A setting is a corpus with a confounder of two or three classes: SemEval (four languages),
  SwiLTra (German, French, Italian), Bills–Tweets (bill or tweet), SCOTUS (Oyez, LexisNexis or Wikipedia), CAP
  (bill, executive order or newspaper clip) and Fedlex (German, French, Italian). SwiLTra and SCOTUS are used in
  every pairing of their classes and with all three classes at once. `code/corpora.py` registers every setting.
* **Encoders.** E5-small (multilingual) is the primary encoder; MiniLM is the second encoder on the English
  settings; the ten-encoder study refits every eraser in each of ten encoders' embedding spaces.
* **Splits and seeds.** Each seed draws an outer split (50/50, linked documents kept on one side; five folds for
  SemEval), and every eraser is fitted on the training side and audited on the held-out side. Some experiments were
  first run with seeds 0–2 and later extended with seeds 3–4 (`04_extensions.ipynb`, Section 7); the paper reports
  the five-seed results.
* **Frozen transfer.** Maps fitted on a donor corpus are frozen and applied to six MTEB tasks (five legal retrieval
  tasks and STSBenchmark).

Some code comments refer to "the parent paper", Fan et al. (2025; see [Origin and licence](#origin-and-licence)),
whose data, encoders and protocol this work builds on, and to "the revision" of the paper, during which most
experiments were added.

## The paper's tables and figures

| Exhibit (LaTeX label) | File in `paper-artifacts/` | Results it reads (notebook, section) |
|---|---|---|
| Table 2: primary in-distribution leakage and utility (`tab:primary-id`) | `tables/table_2_primary_id.tex` | main (02 §1–2), fedlex (03), five-seed merges (04 §7) |
| Table 3: candidate directions, intervention operators and erasure extent (`tab:mechanism-controls`) | `tables/table3_mechanism_controls.tex` | extensions (04 §3), selectors (05) |
| Erasure extent: SAE-LEACE against displacement-matched PCA (`tab:erasure-extent`) | `tables/table_erasure_extent.tex` | extensions (04 §3), plus the in-distribution runs |
| SAE construction (`tab:sae-construction`) | `tables/table_SAE_construction_block.tex` | extensions (04 §4) |
| Dataset composition (`tab:dataset-composition`) | `tables/tableA1_dataset_composition.tex` | the corpora in `data/` |
| Encoders (`tab:encoders`) | `tables/tableA2_encoders.tex` | inventory (01) |
| Extended in-distribution results (`tab:extended-id`) | `tables/tableB1_extended_id_block.tex` | main (02 §1–2), fedlex (03), five-seed merges (04 §7) |
| Robustness across embedding models (`tab:ten-encoder-robustness`) | `tables/tableB2_ten_encoder_robustness_singlepage.tex` | main (02 §5), extensions (04 §5), fedlex (03) |
| Figure 1: erasure rank across compositions (`fig:rank-composition`) | `figures/fig1_rank_composition.pdf` | five-seed merges (02 §2, 04 §2 and §7) |
| Figure 2: geometry and label-free erasure (`fig:geometry-discovery`) | `figures/fig2_geometry_discovery.pdf` | five-seed merges (02 §1 and §3, 04 §7) |
| Frozen transfer and erasure-rank sensitivity (`fig:frozen-transfer-rank`) | `figures/figB3_frozen_transfer_rank.pdf` | main (02 §4), extensions (04 §6–8) |

Each table also has a Markdown copy (`tables/<name>.md`) and each figure a PNG. The robustness table is a
non-floating block (`\captionof`); the others are floats.

## Repository layout

```
code/              the experiment code (Python modules; the notebooks only call it)
data/              the corpora, processed (CSV/JSONL), and data/raw/ with the downloaded sources
notebooks/         01–06: run them in order to reproduce every result and every exhibit
results/           the raw results the notebooks write (JSON), one folder per notebook
  main/              02_main_experiments
  fedlex/            03_fedlex
  extensions/        04_extensions (the seeds 3-4 runs in extensions_seeds_3_4/, merged into raw/five_seeds/)
  selectors/         05_latent_selectors
paper-artifacts/   the code that builds each table and figure from results/, and the built files
  build.py           python paper-artifacts/build.py  rebuilds everything
  manifest.toml      which saved run feeds which cell
  tables/  figures/  the paper's tables (.tex, .md) and figures (.pdf, .png)
  data/              the numbers behind each exhibit (CSV) and its provenance (sources, checks)
  derived/           inputs the build computes from results/ (five-seed transfer summaries, Figure 1's random curve)
  paper_assets/      the reporting library (loaders, checks, LaTeX)
```

| Module in `code/` | Role |
|---|---|
| `corpora.py` | The corpus registry: every setting, its loader, classes, pairs, seeds and split policy |
| `protocols.py` | Splits (linked documents kept on one side) and the fit → freeze → apply engine of every eraser |
| `sae.py`, `sae_confounder_pipeline.py` | The TopK SAE and the SAE-based confounder discovery (pseudo-groups, latent scoring) |
| `audit.py` | The evaluation protocol: linear, gradient-boosting and MLP probe AUC, topic purity |
| `run_sae_experiment.py`, `style_erasure_pipeline.py`, `experiments.py` | Data loaders, encoders and encoding, the embedding cache, the shared fitting recipe |
| `revised.py` | The in-distribution runners (`run_id`, `run_id_kmeans`, `run_mechanism`, `run_ood`, `run_encoder_cell`) |
| `alternatives.py` | Rank and budget sweeps, k-means variants, geometry, transfer of PCA ranks and k-means maps |
| `ood.py` | Frozen transfer to the MTEB tasks and its paired bootstrap |
| `focused_additions.py`, `focused_extensions.py` | Table 3's runs and controls, SAE construction, the three-class encoder study, the seeds 3–4 extension and merges |
| `latent_selectors.py` | The k-means-free latent selectors of Table 3 |
| `fedlex.py` | The Fedlex cohort, whole-law splits, primary run and encoder cells |
| `build_cap_data.py` | Rebuilds the CAP corpora from their public sources |
| `paths.py`, `artifacts.py` | The results folders and caches; result I/O |
| `analysis.py`, `plots.py`, `tables.py`, `revised_exhibits.py` | Helpers that some runners call (summaries and the exhibit code of an earlier version) |

## Data

| Corpus | Confounder | Source | In this repository |
|---|---|---|---|
| SemEval | language (EN, DE, ES, ZH) | cross-lingual news article pairs of SemEval-2022 Task 8 (multilingual news article similarity), as distributed with the code of Fan et al. (2025) | `data/semeval_en_vs_non_en.csv` |
| SwiLTra | language (DE, FR, IT) | summaries of Swiss Federal Supreme Court decisions in three languages (SwiLTra-Bench), as distributed with the code of Fan et al. (2025) | `data/swiltra_bench_decision_summaries.csv` |
| Bills–Tweets | source | congressional bill summaries (Hoyle et al., 2022) and tweets by members of Congress ([congresstweets](https://github.com/alexlitel/congresstweets), 2017-06-21 to 2019-01-02); a 10,000-document sample from the code of Fan et al. (2025) | `data/bill_tweets_sample.jsonl` |
| SCOTUS | source (Oyez, LexisNexis, Wikipedia) | Supreme Court case summaries (Super-SCOTUS), paired as in Fan et al. (2025), whose code distributes them | `data/case_summaries_*.csv` |
| CAP | source (bill, executive order, newspaper) | Comparative Agendas Project: newspaper clips and US executive orders (v21.3), bill summaries (Hoyle et al., 2022), order texts from the Federal Register API; rebuilt by `code/build_cap_data.py` | `data/cap_*.jsonl.gz`, `data/cap_sample.jsonl`; sources in `data/raw/cap/` (the two bill files are downloaded on first use) |
| Fedlex | language (DE, FR, IT) | Swiss domestic federal law, article level, from [SwissLawTranslations](https://huggingface.co/datasets/joelniklaus/SwissLawTranslations) (revision `03080f36`); a seeded 2,784-article cohort | `data/raw/fedlex/` (the large `train.parquet` is downloaded on first use); cohort in `data/fedlex_domestic_n2784_manifest.csv.gz` |
| MTEB tasks (transfer) | — | AILACasedocs, AILAStatutes, LegalBenchConsumerContractsQA, LegalBenchCorporateLobbying, LegalSummarization and STSBenchmark, from the `mteb/` datasets on Hugging Face at pinned revisions (`code/ood.py`) | `data/raw/mteb/` |

The loaders drop exact duplicate texts; `data/raw/` holds the downloaded sources the processed files are built from.
Three of them are too large for the repository and are downloaded the first time a notebook needs them, each
checked against its recorded SHA-256 (a changed upstream file stops the run instead of silently changing the data):

| File | Downloaded from | By | When |
|---|---|---|---|
| `data/raw/fedlex/article_level/train.parquet` (318 MB) | [SwissLawTranslations](https://huggingface.co/datasets/joelniklaus/SwissLawTranslations) at revision `03080f36` | `code/fedlex.py` | `03_fedlex.ipynb` |
| `data/raw/cap/bills_train.metadata.jsonl`, `bills_test.metadata.jsonl` (130 MB) | the data release of Hoyle et al. (2022) | `code/build_cap_data.py` | `01_data.ipynb` (the CAP check) |

Rebuilding the tables and figures (`paper-artifacts/build.py`) needs none of them.
Each dataset keeps the licence and terms of use of its original source.

## Notes on reproducibility

* SAE training is deterministic within one environment (the minibatch order is drawn on the training device), so a
  rerun on other hardware reproduces the conditions without an SAE exactly and the SAE conditions closely. The
  published runs fitted the SAEs on a Colab A100 GPU, except the latent selectors (notebook 05) and six cells of the
  three-class encoder study (GIST-small, GIST-large and MPNet), which were fitted on a CPU; the result files record
  where each run was made.
* The rank and budget sweeps of SwiLTra DE–FR and SCOTUS Oyez–Lexis in `results/main/` hold seeds 0–2 as
  `02_main_experiments` writes them; seeds 3–4 are added by `04_extensions` (`results/extensions/raw/five_seeds/`).
* `paper-artifacts/derived/` holds inputs the build computes from `results/`; it is rebuilt when its inputs change.

## Origin and licence

This repository builds on the code and data of Fan et al. (2025), *The Medium Is Not the Message: Deconfounding
Document Embeddings via Linear Concept Erasure* (EMNLP 2025), [github.com/y-fn/deconfounding-embeddings](https://github.com/y-fn/deconfounding-embeddings),
which is released under the Apache License 2.0. The code here is released under the same licence (see `LICENSE`).
The datasets keep the licences and terms of use of their original sources (see [Data](#data)).

## Citation

If you use this code or data, please cite the paper.
<!-- TODO before publishing: add the BibTeX entry of the pre-print (authors, year, arXiv identifier). -->
