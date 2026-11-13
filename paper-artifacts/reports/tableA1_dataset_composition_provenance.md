# Table A1 (tableA1_dataset_composition): provenance and validation

Build: `python -m paper_assets build tableA1_dataset_composition`. Counts come from `corpora.load` (the registry loaders) and `protocols.link_groups`; nothing is embedded, merged or relinked.

## Data files

- `data/case_summaries_oyez_vs_lexisnexis.csv` sha256 `6c00ceab52190f77d1c29e9e2e5dcacd4e9fc6a1fc8aa8bcaaaa725f7daf3084`
- `data/case_summaries_oyez_vs_wikipedia.csv` sha256 `d0cef4f97bf898e339f4dae0108af055efd019c0d1a88e80d5792559f98cf45c`
- `data/case_summaries_wikipedia_vs_lexisnexis.csv` sha256 `2d1826c0229cc5d099e4e0de3ba3c91f55855156905fe773cd0d748560d52bb7`
- `data/swiltra_bench_decision_summaries.csv` sha256 `e595becacbf89265d14738b82453bb63cf63543ea523b068e4714bb3126bd2a0`

## Construction per setting

### swiltra (DE–FR)
- composition: classes {'de': 2784, 'fr': 2784}; 5568 documents; 2784 linked pairs, 0 linked triples; 2784 retrieval pairs {'de|fr': 2784}; 0 unpaired {'de': 0, 'fr': 0}
- split unit: linked pair via `protocols.make_split (disjoint pairs)`
- rule: row-aligned decisions; every two language versions of a row are linked
- raw_rows: 2784
- rows_with_all_languages: 2784
- texts_before_dedup: 5568
- exact_duplicates_dropped: 0
- max_pairs: None

### swiltra_de_it (DE–IT)
- composition: classes {'de': 2784, 'it': 2784}; 5568 documents; 2784 linked pairs, 0 linked triples; 2784 retrieval pairs {'de|it': 2784}; 0 unpaired {'de': 0, 'it': 0}
- split unit: linked pair via `protocols.make_split (disjoint pairs)`
- rule: row-aligned decisions; every two language versions of a row are linked
- raw_rows: 2784
- rows_with_all_languages: 2784
- texts_before_dedup: 5568
- exact_duplicates_dropped: 0
- max_pairs: None

### swiltra_fr_it (FR–IT)
- composition: classes {'fr': 2784, 'it': 2784}; 5568 documents; 2784 linked pairs, 0 linked triples; 2784 retrieval pairs {'fr|it': 2784}; 0 unpaired {'fr': 0, 'it': 0}
- split unit: linked pair via `protocols.make_split (disjoint pairs)`
- rule: row-aligned decisions; every two language versions of a row are linked
- raw_rows: 2784
- rows_with_all_languages: 2784
- texts_before_dedup: 5568
- exact_duplicates_dropped: 0
- max_pairs: None

### swiltra_de_fr_it (DE–FR–IT)
- composition: classes {'de': 2784, 'fr': 2784, 'it': 2784}; 8352 documents; 0 linked pairs, 2784 linked triples; 8352 retrieval pairs {'de|fr': 2784, 'de|it': 2784, 'fr|it': 2784}; 0 unpaired {'de': 0, 'fr': 0, 'it': 0}
- split unit: linked group (pair or triple) via `protocols._make_group_split`
- rule: row-aligned decisions; every two language versions of a row are linked
- raw_rows: 2784
- rows_with_all_languages: 2784
- texts_before_dedup: 8352
- exact_duplicates_dropped: 0
- max_pairs: None

### scotus_oyez_lexis (Oyez–Lexis)
- composition: classes {'oyez': 2067, 'lexis': 2034}; 4101 documents; 1017 linked pairs, 0 linked triples; 1017 retrieval pairs {'oyez|lexis': 1017}; 2067 unpaired {'oyez': 1050, 'lexis': 1017}
- split unit: linked pair via `protocols.make_split (disjoint pairs)`
- rule: row i of each source block pairs with row i of the other for i < 1024; later rows pooled unpaired
- block_order: ['oyez', 'lexis']
- raw_rows_by_source: {'oyez': 2075, 'lexis': 2048}
- n_pairs_parameter: 1024
- designated_pairs: 1024
- exact_duplicates_dropped: 22
- designated_pairs_lost_to_dedup: 7

### scotus_oyez_wiki (Oyez–Wikipedia)
- composition: classes {'oyez': 1800, 'wiki': 1497}; 3297 documents; 772 linked pairs, 0 linked triples; 772 retrieval pairs {'oyez|wiki': 772}; 1753 unpaired {'oyez': 1028, 'wiki': 725}
- split unit: linked pair via `protocols.make_split (disjoint pairs)`
- rule: row i of each source block pairs with row i of the other for i < 780; later rows pooled unpaired
- block_order: ['oyez', 'wiki']
- raw_rows_by_source: {'oyez': 1804, 'wiki': 1518}
- n_pairs_parameter: 780
- designated_pairs: 780
- exact_duplicates_dropped: 25
- designated_pairs_lost_to_dedup: 8

### scotus_wiki_lexis (Lexis–Wikipedia)
- composition: classes {'lexis': 2039, 'wiki': 1497}; 3536 documents; 1017 linked pairs, 0 linked triples; 1017 retrieval pairs {'lexis|wiki': 1017}; 1502 unpaired {'lexis': 1022, 'wiki': 480}
- split unit: linked pair via `protocols.make_split (disjoint pairs)`
- rule: row i of each source block pairs with row i of the other for i < 1024; later rows pooled unpaired
- block_order: ['wiki', 'lexis']
- raw_rows_by_source: {'wiki': 1518, 'lexis': 2048}
- n_pairs_parameter: 1024
- designated_pairs: 1024
- exact_duplicates_dropped: 30
- designated_pairs_lost_to_dedup: 7

### scotus_oyez_lexis_wiki (All three)
- composition: classes {'oyez': 2490, 'lexis': 3519, 'wiki': 1497}; 7506 documents; 1286 linked pairs, 633 linked triples; 3185 retrieval pairs {'oyez|lexis': 1285, 'oyez|wiki': 768, 'lexis|wiki': 1132}; 3035 unpaired {'oyez': 1070, 'lexis': 1735, 'wiki': 230}
- split unit: linked group (pair or triple) via `protocols._make_group_split`
- rule: document = (source, text); the three files' pair links joined transitively into connected components; a component holding two summaries of one source is left unpaired
- per_file: {'data/case_summaries_oyez_vs_lexisnexis.csv': {'documents': 4101, 'pairs': 1017, 'new_documents': 4101, 'already_seen': 0}, 'data/case_summaries_oyez_vs_wikipedia.csv': {'documents': 3297, 'pairs': 772, 'new_documents': 1920, 'already_seen': 1377}, 'data/case_summaries_wikipedia_vs_lexisnexis.csv': {'documents': 3536, 'pairs': 1017, 'new_documents': 1485, 'already_seen': 2051}}
- union_documents: 7506
- summaries_in_two_files_by_source: {'oyez': 1377, 'lexis': 554, 'wiki': 1497}
- file_links: 2806
- distinct_links: 2806
- components_with_links: 1923
- ambiguous_components: 4
- ambiguous_component_sizes: [4, 4, 4, 4]
- documents_in_ambiguous_components: 16
- file_links_in_ambiguous_components: 12
- retrieval_pairs_from_links: 2794
- retrieval_pairs_by_transitivity: 391

## Missing or unverified provenance (flagged, not asserted)

- The three SCOTUS CSVs and the SwiLTra CSV entered the repository as uploads (git: 'Add files via upload', 2025-07-16). No download or build script, source revision or upstream identifier is recorded.
- The SCOTUS CSVs hold only `case_summaries` and `source`: no case identifier. The pairing boundary (1,024; 780 for Oyez–Wikipedia) is the parent paper's construction. The loader docstring reports a TF-IDF check that recovers the row alignment for 95% of the rows before the boundary; that check is not re-run here.
- The loader docstring also reports that the unpaired SCOTUS remainder still contains many cross-source matches (TF-IDF mutual best matches). Such documents are unpaired distractors, and under the document holdout a counterpart can sit on the other side of the split (audit_2026-09-28/outputs/C1_split_overlap.json).
- SwiLTra: the columns are `de_bgeText`, `fr_bgeText`, `it_bgeText`. Which upstream SwiLTra-Bench field they are, and any filtering before upload, is not recorded.
- Ambiguous SCOTUS components: the `load_case_summaries_union` docstring says 4 components and the pipeline's unresolved list says '12 ambiguous file pairs'. Both hold, as measured here: 4 components (16 documents) containing 12 file links.

## Checks (87 passed)

- swiltra: class counts {'de': 2784, 'fr': 2784} sum to the 5568 documents
- swiltra: every linked document belongs to exactly one linked group (5568 documents)
- swiltra: no linked group holds two documents of one language class
- swiltra: group sizes {2: 2784} (pairs and triples only)
- swiltra: 2784 retrieval pairs = sum over groups of C(size, 2) = 2784; every group is fully linked, no pair repeated
- swiltra: 0 unpaired documents (no link) = the loader's n_unpaired; pairs = loader n_pairs
- swiltra: classes ['de', 'fr'] = manifest
- swiltra: documents, retrieval pairs, group sizes and unpaired counts equal lib.datasets.processed (used by Tables 1 and 2)
- swiltra: 5568 documents = registry size 5568
- swiltra_de_it: class counts {'de': 2784, 'it': 2784} sum to the 5568 documents
- swiltra_de_it: every linked document belongs to exactly one linked group (5568 documents)
- swiltra_de_it: no linked group holds two documents of one language class
- swiltra_de_it: group sizes {2: 2784} (pairs and triples only)
- swiltra_de_it: 2784 retrieval pairs = sum over groups of C(size, 2) = 2784; every group is fully linked, no pair repeated
- swiltra_de_it: 0 unpaired documents (no link) = the loader's n_unpaired; pairs = loader n_pairs
- swiltra_de_it: classes ['de', 'it'] = manifest
- swiltra_de_it: documents, retrieval pairs, group sizes and unpaired counts equal lib.datasets.processed (used by Tables 1 and 2)
- swiltra_de_it: 5568 documents = registry size 5568
- swiltra_fr_it: class counts {'fr': 2784, 'it': 2784} sum to the 5568 documents
- swiltra_fr_it: every linked document belongs to exactly one linked group (5568 documents)
- swiltra_fr_it: no linked group holds two documents of one language class
- swiltra_fr_it: group sizes {2: 2784} (pairs and triples only)
- swiltra_fr_it: 2784 retrieval pairs = sum over groups of C(size, 2) = 2784; every group is fully linked, no pair repeated
- swiltra_fr_it: 0 unpaired documents (no link) = the loader's n_unpaired; pairs = loader n_pairs
- swiltra_fr_it: classes ['fr', 'it'] = manifest
- swiltra_fr_it: documents, retrieval pairs, group sizes and unpaired counts equal lib.datasets.processed (used by Tables 1 and 2)
- swiltra_fr_it: 5568 documents = registry size 5568
- swiltra_de_fr_it: class counts {'de': 2784, 'fr': 2784, 'it': 2784} sum to the 8352 documents
- swiltra_de_fr_it: every linked document belongs to exactly one linked group (8352 documents)
- swiltra_de_fr_it: no linked group holds two documents of one language class
- swiltra_de_fr_it: group sizes {3: 2784} (pairs and triples only)
- swiltra_de_fr_it: 8352 retrieval pairs = sum over groups of C(size, 2) = 8352; every group is fully linked, no pair repeated
- swiltra_de_fr_it: 0 unpaired documents (no link) = the loader's n_unpaired; pairs = loader n_pairs
- swiltra_de_fr_it: classes ['de', 'fr', 'it'] = manifest
- swiltra_de_fr_it: documents, retrieval pairs, group sizes and unpaired counts equal lib.datasets.processed (used by Tables 1 and 2)
- swiltra_de_fr_it: 8352 documents = registry size 8352
- scotus_oyez_lexis: class counts {'oyez': 2067, 'lexis': 2034} sum to the 4101 documents
- scotus_oyez_lexis: every linked document belongs to exactly one linked group (2034 documents)
- scotus_oyez_lexis: no linked group holds two documents of one source class
- scotus_oyez_lexis: group sizes {2: 1017} (pairs and triples only)
- scotus_oyez_lexis: 1017 retrieval pairs = sum over groups of C(size, 2) = 1017; every group is fully linked, no pair repeated
- scotus_oyez_lexis: 2067 unpaired documents (no link) = the loader's n_unpaired; pairs = loader n_pairs
- scotus_oyez_lexis: classes ['oyez', 'lexis'] = manifest
- scotus_oyez_lexis: documents, retrieval pairs, group sizes and unpaired counts equal lib.datasets.processed (used by Tables 1 and 2)
- scotus_oyez_lexis: 4101 documents = registry size 4101
- scotus_oyez_wiki: class counts {'oyez': 1800, 'wiki': 1497} sum to the 3297 documents
- scotus_oyez_wiki: every linked document belongs to exactly one linked group (1544 documents)
- scotus_oyez_wiki: no linked group holds two documents of one source class
- scotus_oyez_wiki: group sizes {2: 772} (pairs and triples only)
- scotus_oyez_wiki: 772 retrieval pairs = sum over groups of C(size, 2) = 772; every group is fully linked, no pair repeated
- scotus_oyez_wiki: 1753 unpaired documents (no link) = the loader's n_unpaired; pairs = loader n_pairs
- scotus_oyez_wiki: classes ['oyez', 'wiki'] = manifest
- scotus_oyez_wiki: documents, retrieval pairs, group sizes and unpaired counts equal lib.datasets.processed (used by Tables 1 and 2)
- scotus_oyez_wiki: 3297 documents = registry size 3297
- scotus_wiki_lexis: class counts {'lexis': 2039, 'wiki': 1497} sum to the 3536 documents
- scotus_wiki_lexis: every linked document belongs to exactly one linked group (2034 documents)
- scotus_wiki_lexis: no linked group holds two documents of one source class
- scotus_wiki_lexis: group sizes {2: 1017} (pairs and triples only)
- scotus_wiki_lexis: 1017 retrieval pairs = sum over groups of C(size, 2) = 1017; every group is fully linked, no pair repeated
- scotus_wiki_lexis: 1502 unpaired documents (no link) = the loader's n_unpaired; pairs = loader n_pairs
- scotus_wiki_lexis: classes ['lexis', 'wiki'] = manifest
- scotus_wiki_lexis: documents, retrieval pairs, group sizes and unpaired counts equal lib.datasets.processed (used by Tables 1 and 2)
- scotus_wiki_lexis: 3536 documents = registry size 3536
- scotus_oyez_lexis_wiki: class counts {'oyez': 2490, 'lexis': 3519, 'wiki': 1497} sum to the 7506 documents
- scotus_oyez_lexis_wiki: every linked document belongs to exactly one linked group (4471 documents)
- scotus_oyez_lexis_wiki: no linked group holds two documents of one source class
- scotus_oyez_lexis_wiki: group sizes {3: 633, 2: 1286} (pairs and triples only)
- scotus_oyez_lexis_wiki: 3185 retrieval pairs = sum over groups of C(size, 2) = 3185; every group is fully linked, no pair repeated
- scotus_oyez_lexis_wiki: 3035 unpaired documents (no link) = the loader's n_unpaired; pairs = loader n_pairs
- scotus_oyez_lexis_wiki: the step-by-step re-derivation of the pooled corpus equals corpora.load exactly (documents, sources, pairs)
- scotus_oyez_lexis_wiki: 2794 pairs recorded in a file + 391 by transitivity = 3185 retrieval pairs
- scotus_oyez_lexis_wiki: classes ['oyez', 'lexis', 'wiki'] = manifest
- scotus_oyez_lexis_wiki: documents, retrieval pairs, group sizes and unpaired counts equal lib.datasets.processed (used by Tables 1 and 2)
- scotus_oyez_lexis_wiki: 7506 documents = registry size 7506
- SwiLTra: its settings draw on exactly three classes (['de', 'fr', 'it']), one column each
- swiltra: every linked group is one pair, so groups = retrieval pairs (2784); the caption states this instead of a column
- swiltra_de_it: every linked group is one pair, so groups = retrieval pairs (2784); the caption states this instead of a column
- swiltra_fr_it: every linked group is one pair, so groups = retrieval pairs (2784); the caption states this instead of a column
- swiltra_de_fr_it: every linked group is one pair, so groups = retrieval pairs (8352); the caption states this instead of a column
- SCOTUS: its settings draw on exactly three classes (['oyez', 'lexis', 'wiki']), one column each
- scotus_oyez_lexis: every linked group is one pair, so groups = retrieval pairs (1017); the caption states this instead of a column
- scotus_oyez_wiki: every linked group is one pair, so groups = retrieval pairs (772); the caption states this instead of a column
- scotus_wiki_lexis: every linked group is one pair, so groups = retrieval pairs (1017); the caption states this instead of a column
- scotus_oyez_lexis_wiki: every linked group is one pair, so groups = retrieval pairs (3185); the caption states this instead of a column
- tableA1_dataset_composition: every citation and reference resolves
- tableA1_dataset_composition: the float fits the TACL text block
- tableA1_dataset_composition: no overfull boxes in the TACL preview

## Issues

- **warning** `tacl_style_standin`: writing/tacl_analysis_draft/tacl2021v1.sty not found: measured on a stand-in page with the TACL geometry (455.24pt x 702.8pt text, 219.09pt columns, Times)
- **warning** `no_bibliography`: tableA1_dataset_composition: writing/tacl_analysis_draft/references.bib not found; the preview shows citation keys
