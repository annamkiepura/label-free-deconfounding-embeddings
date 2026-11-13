# Build report

Rebuild: `python paper-artifacts/build.py`. One row per exhibit; a check is an assertion about the saved results that the exhibit records in its provenance file (`data/<name>.provenance.json`).

| Exhibit | Group | Kind | Checks | Errors | Warnings | Title |
|---|---|---|---:|---:|---:|---|
| `tableA1_dataset_composition` | regenerated | table | 87 | 0 | 2 | Table A1: dataset composition and linked groups |
| `tableA2_encoders` | regenerated | table | 32 | 0 | 9 | Table A2: encoders |
| `table_2_primary_id` | regenerated | table | 277 | 0 | 1 | Table 2: the in-distribution comparison, seeds 0-4 |
| `fig1_rank_composition` | regenerated | figure | 767 | 0 | 0 | Figure 1: rank and composition, seeds 0-4 |
| `fig2_geometry_discovery` | regenerated | figure | 182 | 0 | 0 | Figure 2: geometry and discovery, seeds 0-4 |
| `tableB1_extended_id_block` | regenerated | table | 455 | 0 | 3 | Table B1: the extended in-distribution comparison, seeds 0-4 |
| `figB3_frozen_transfer_rank` | regenerated | figure | 434 | 0 | 0 | Figure B3: frozen transfer, seeds 0-4, with displacement-matched PCA |
| `table3_mechanism_controls` | new | table | 39 | 0 | 1 | Table 3: mechanism controls on seeds 0-4, with the latent-selector rows |
| `table_erasure_extent` | new | table | 46 | 0 | 1 | Table 4: SAE-LEACE against displacement-matched PCA |
| `table_SAE_construction_block` | new | table | 54 | 0 | 1 | SAE construction: other TopK dictionaries |
| `tableB2_ten_encoder_robustness_singlepage` | new | table | 3498 | 0 | 1 | Table B2: the ten-encoder study relative to Raw |

## Warnings

- a2_revision_not_recorded: e5-base: model revision not recorded by any run (inventory fallback d12875059715 is the current Hub revision)
- a2_revision_not_recorded: e5-large: model revision not recorded by any run (inventory fallback 3d7cfbdacd47 is the current Hub revision)
- a2_revision_not_recorded: gist-base: model revision not recorded by any run (inventory fallback bf6b2e55e92f is the current Hub revision)
- a2_revision_not_recorded: gist-large: model revision not recorded by any run (inventory fallback 7831200e2f78 is the current Hub revision)
- a2_revision_not_recorded: gist-small: model revision not recorded by any run (inventory fallback 75e62fd210b9 is the current Hub revision)
- a2_revision_not_recorded: mpnet: model revision not recorded by any run (inventory fallback e8c3b32edf54 is the current Hub revision)
- a2_revision_not_recorded: mxb: model revision not recorded by any run (inventory fallback b33106f585b9 is the current Hub revision)
- a2_revision_not_recorded: nomic-v2: model revision not recorded by any run (inventory fallback 1066b6599d09 is the current Hub revision)
- b1_main_csv_absent: not compared (CSV not built): ['table2_primary_id', 'fig3_pair_specific_retrieval', 'tab_composition']
- b1_no_preview: preview not compiled; one-page fit not verified
- no_bibliography: tableA1_dataset_composition: writing/tacl_analysis_draft/references.bib not found; the preview shows citation keys
- tacl_style_standin: writing/tacl_analysis_draft/tacl2021v1.sty not found: measured on a stand-in page with the TACL geometry (455.24pt x 702.8pt text, 219.09pt columns, Times)
