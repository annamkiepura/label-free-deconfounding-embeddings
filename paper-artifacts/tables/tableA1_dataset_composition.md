<!-- Generated from tableA1_dataset_composition.tex by `python paper-artifacts/build.py tableA1_dataset_composition`; the LaTeX file is the reference. -->

| Setting | Docs per class | Docs per class | Docs per class | Docs | Pairs | Unpaired |
|---|---|---|---|---|---|---|
| **SwiLTra** | *DE* | *FR* | *IT* |  |  |  |
| DE–FR | 2,784 | 2,784 | – | 5,568 | 2,784 | 0 |
| DE–IT | 2,784 | – | 2,784 | 5,568 | 2,784 | 0 |
| FR–IT | – | 2,784 | 2,784 | 5,568 | 2,784 | 0 |
| DE–FR–IT | 2,784 | 2,784 | 2,784 | 8,352 | 8,352 | 0 |
| **SCOTUS** | *Oyez* | *Lexis* | *Wiki* |  |  |  |
| O–L | 2,067 | 2,034 | – | 4,101 | 1,017 | 2,067 |
| O–W | 1,800 | – | 1,497 | 3,297 | 772 | 1,753 |
| L–W | – | 2,039 | 1,497 | 3,536 | 1,017 | 1,502 |
| O–L–W | 2,490 | 3,519 | 1,497 | 7,506 | 3,185 | 3,035 |

Setting composition after exact-duplicate removal. SCOTUS sources: Oyez (O), LexisNexis (L), Wikipedia (W); – marks an absent class. *Pairs*: linked document pairs (held-out retrieval targets); *Unpaired*: documents without a recorded link, used for fitting and as distractors. Linked groups, never split between training and evaluation, are single pairs in the binary settings, 2,784 triples (one per decision) in DE–FR–IT, and 1,286 pairs plus 633 triples in O–L–W (O–L 1,285, O–W 768, L–W 1,132 pairs). SCOTUS files have no case IDs: as in \citetfan2025medium, row i of one source block is linked to row i of the other for i<1,024 (780 for O–W), and deduplication breaks 7, 8 and 7 of these links. O–L–W merges the three files on (source, text) and joins links transitively (391 pairs); 4 components with two summaries of one source (16 documents) stay unpaired.
