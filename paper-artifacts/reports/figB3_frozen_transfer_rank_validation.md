# Figure B3 (figB3_frozen_transfer_rank): provenance and validation

Build: `python paper-artifacts/build.py figB3_frozen_transfer_rank`. Reporting only: no donor map fitted, no target re-encoded.

## Sources

- A/B E5-small · CAP: `paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json` + `paper-artifacts/derived/ood/ood_primary_e5-small_cap.json` + `results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_cap.json` (k-means arm)
- A/B E5-small · SCOTUS (3 sources): `paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json` + `paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json` + `results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_scotus_oyez_lexis_wiki.json` (k-means arm)
- A/B E5-small · SwiLTra (3 languages): `paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json` + `paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json` + `results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_swiltra_de_fr_it.json` (k-means arm)
- A/B MiniLM · CAP: `paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json` + `paper-artifacts/derived/ood/ood_primary_mini_cap.json` + `results/extensions/raw/five_seeds/alternatives/ood_kmeans_mini_cap.json` (k-means arm)
- A/B MiniLM · SCOTUS (3 sources): `paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json` + `paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json` + `results/extensions/raw/five_seeds/alternatives/ood_kmeans_mini_scotus_oyez_lexis_wiki.json` (k-means arm)
- C E5-small · CAP: `paper-artifacts/derived/ood/ood_summary_ranks_five_seeds.json` + `paper-artifacts/derived/ood/ood_ranks_e5-small_cap.json` + `paper-artifacts/derived/ood/ood_ranks_e5-small_cap.json`; ranks [1, 2, 3, 4, 8, 16, 32]
- C E5-small · SCOTUS Oyez–Lexis (binary): `paper-artifacts/derived/ood/ood_summary_ranks_five_seeds.json` + `paper-artifacts/derived/ood/ood_ranks_e5-small_scotus_oyez_lexis.json` + `paper-artifacts/derived/ood/ood_ranks_e5-small_scotus_oyez_lexis.json`; ranks [1, 2, 3, 4, 8, 16, 32]
- C E5-small · SwiLTra DE–FR (binary): `paper-artifacts/derived/ood/ood_summary_ranks_five_seeds.json` + `paper-artifacts/derived/ood/ood_ranks_e5-small_swiltra.json` + `paper-artifacts/derived/ood/ood_ranks_e5-small_swiltra.json`; ranks [1, 2, 3, 4, 8, 16, 32]
- C MiniLM · CAP: `paper-artifacts/derived/ood/ood_summary_ranks_five_seeds.json` + `paper-artifacts/derived/ood/ood_ranks_mini_cap.json` + `paper-artifacts/derived/ood/ood_ranks_mini_cap.json`; ranks [1, 2, 3, 4, 8, 16, 32]
- C MiniLM · SCOTUS Oyez–Lexis (binary): `paper-artifacts/derived/ood/ood_summary_ranks_five_seeds.json` + `paper-artifacts/derived/ood/ood_ranks_mini_scotus_oyez_lexis.json` + `paper-artifacts/derived/ood/ood_ranks_mini_scotus_oyez_lexis.json`; ranks [1, 2, 3, 4, 8, 16, 32]

## Intervals

- Paired percentile bootstrap, 10000 resamples, level 0.95, seed 0, conditional on the frozen maps (all read from the saved summaries).
- Reproduced from the per-query run files with `ood.summarize` (loaded from the cache; cache key `a5ef96e99a109b21…` = SHA-256 of code/ood.py and every input run file). Every plotted change and interval equals the reproduction to 1e-12 (checks `b3_interval_reproduced`).
- AILA: both tasks list the same 50 query ids in the same order in every run file, so the evaluator draws one joint resample for them.
- STS: per-seed Spearman recomputed within every resample and averaged over seeds; endpoints are never averaged.
- Panel C uses the same bootstrap intervals (no seed SD is shown anywhere).

## Values

| Panel | Encoder | Donor | Method | Δ | 95% CI |
|---|---|---|---|---:|---|
| A | e5-small | cap | pca_top1 | +0.0018 | [-0.0017, +0.0057] |
| B | e5-small | cap | pca_top1 | +0.0002 | [-0.0002, +0.0006] |
| A | e5-small | cap | kmeans_leace | +0.0017 | [-0.0016, +0.0056] |
| B | e5-small | cap | kmeans_leace | +0.0003 | [-0.0001, +0.0006] |
| A | e5-small | cap | metadata_leace | +0.0040 | [-0.0004, +0.0087] |
| B | e5-small | cap | metadata_leace | +0.0008 | [+0.0003, +0.0014] |
| A | e5-small | cap | sae_leace | -0.0093 | [-0.0164, -0.0026] |
| B | e5-small | cap | sae_leace | -0.0051 | [-0.0080, -0.0024] |
| A | e5-small | cap | pca_matched_var | -0.0057 | [-0.0141, +0.0026] |
| B | e5-small | cap | pca_matched_var | -0.0049 | [-0.0081, -0.0018] |
| A | e5-small | scotus_oyez_lexis_wiki | pca_top1 | -0.0003 | [-0.0051, +0.0039] |
| B | e5-small | scotus_oyez_lexis_wiki | pca_top1 | +0.0012 | [+0.0007, +0.0017] |
| A | e5-small | scotus_oyez_lexis_wiki | kmeans_leace | +0.0006 | [-0.0047, +0.0053] |
| B | e5-small | scotus_oyez_lexis_wiki | kmeans_leace | +0.0011 | [+0.0005, +0.0017] |
| A | e5-small | scotus_oyez_lexis_wiki | metadata_leace | -0.0001 | [-0.0056, +0.0049] |
| B | e5-small | scotus_oyez_lexis_wiki | metadata_leace | +0.0011 | [+0.0005, +0.0017] |
| A | e5-small | scotus_oyez_lexis_wiki | sae_leace | -0.0094 | [-0.0170, -0.0022] |
| B | e5-small | scotus_oyez_lexis_wiki | sae_leace | -0.0029 | [-0.0048, -0.0011] |
| A | e5-small | scotus_oyez_lexis_wiki | pca_matched_var | -0.0120 | [-0.0209, -0.0033] |
| B | e5-small | scotus_oyez_lexis_wiki | pca_matched_var | -0.0031 | [-0.0057, -0.0007] |
| A | e5-small | swiltra_de_fr_it | pca_top1 | -0.0009 | [-0.0035, +0.0014] |
| B | e5-small | swiltra_de_fr_it | pca_top1 | +0.0002 | [-0.0001, +0.0004] |
| A | e5-small | swiltra_de_fr_it | kmeans_leace | -0.0003 | [-0.0028, +0.0019] |
| B | e5-small | swiltra_de_fr_it | kmeans_leace | +0.0004 | [-0.0000, +0.0008] |
| A | e5-small | swiltra_de_fr_it | metadata_leace | -0.0003 | [-0.0028, +0.0019] |
| B | e5-small | swiltra_de_fr_it | metadata_leace | +0.0004 | [-0.0000, +0.0008] |
| A | e5-small | swiltra_de_fr_it | sae_leace | -0.0074 | [-0.0146, -0.0005] |
| B | e5-small | swiltra_de_fr_it | sae_leace | -0.0017 | [-0.0032, -0.0001] |
| A | e5-small | swiltra_de_fr_it | pca_matched_var | -0.0078 | [-0.0157, -0.0002] |
| B | e5-small | swiltra_de_fr_it | pca_matched_var | -0.0023 | [-0.0044, -0.0002] |
| A | mini | cap | pca_top1 | -0.0015 | [-0.0064, +0.0032] |
| B | mini | cap | pca_top1 | -0.0001 | [-0.0006, +0.0004] |
| A | mini | cap | kmeans_leace | -0.0004 | [-0.0056, +0.0047] |
| B | mini | cap | kmeans_leace | -0.0003 | [-0.0013, +0.0007] |
| A | mini | cap | metadata_leace | -0.0024 | [-0.0079, +0.0026] |
| B | mini | cap | metadata_leace | -0.0022 | [-0.0035, -0.0009] |
| A | mini | cap | sae_leace | -0.0195 | [-0.0282, -0.0110] |
| B | mini | cap | sae_leace | -0.0013 | [-0.0041, +0.0015] |
| A | mini | cap | pca_matched_var | -0.0170 | [-0.0263, -0.0076] |
| B | mini | cap | pca_matched_var | +0.0002 | [-0.0026, +0.0030] |
| A | mini | scotus_oyez_lexis_wiki | pca_top1 | -0.0030 | [-0.0092, +0.0028] |
| B | mini | scotus_oyez_lexis_wiki | pca_top1 | +0.0002 | [-0.0005, +0.0008] |
| A | mini | scotus_oyez_lexis_wiki | kmeans_leace | -0.0019 | [-0.0077, +0.0040] |
| B | mini | scotus_oyez_lexis_wiki | kmeans_leace | -0.0005 | [-0.0015, +0.0005] |
| A | mini | scotus_oyez_lexis_wiki | metadata_leace | +0.0051 | [-0.0015, +0.0124] |
| B | mini | scotus_oyez_lexis_wiki | metadata_leace | -0.0007 | [-0.0013, +0.0000] |
| A | mini | scotus_oyez_lexis_wiki | sae_leace | -0.0178 | [-0.0271, -0.0085] |
| B | mini | scotus_oyez_lexis_wiki | sae_leace | -0.0028 | [-0.0052, -0.0004] |
| A | mini | scotus_oyez_lexis_wiki | pca_matched_var | -0.0161 | [-0.0266, -0.0056] |
| B | mini | scotus_oyez_lexis_wiki | pca_matched_var | -0.0014 | [-0.0038, +0.0009] |
| C | e5-small | cap | PCA-1 | +0.0018 | [-0.0017, +0.0057] |
| C | e5-small | cap | PCA-2 | +0.0033 | [-0.0012, +0.0080] |
| C | e5-small | cap | PCA-3 | -0.0004 | [-0.0045, +0.0037] |
| C | e5-small | cap | PCA-4 | -0.0012 | [-0.0056, +0.0032] |
| C | e5-small | cap | PCA-8 | -0.0037 | [-0.0099, +0.0023] |
| C | e5-small | cap | PCA-16 | -0.0038 | [-0.0110, +0.0032] |
| C | e5-small | cap | PCA-32 | -0.0172 | [-0.0270, -0.0077] |
| C | e5-small | scotus_oyez_lexis | PCA-1 | -0.0012 | [-0.0066, +0.0038] |
| C | e5-small | scotus_oyez_lexis | PCA-2 | -0.0025 | [-0.0076, +0.0022] |
| C | e5-small | scotus_oyez_lexis | PCA-3 | -0.0027 | [-0.0093, +0.0033] |
| C | e5-small | scotus_oyez_lexis | PCA-4 | -0.0070 | [-0.0136, -0.0010] |
| C | e5-small | scotus_oyez_lexis | PCA-8 | -0.0061 | [-0.0143, +0.0020] |
| C | e5-small | scotus_oyez_lexis | PCA-16 | -0.0078 | [-0.0165, +0.0009] |
| C | e5-small | scotus_oyez_lexis | PCA-32 | -0.0174 | [-0.0265, -0.0087] |
| C | e5-small | swiltra | PCA-1 | +0.0007 | [-0.0010, +0.0024] |
| C | e5-small | swiltra | PCA-2 | -0.0015 | [-0.0043, +0.0011] |
| C | e5-small | swiltra | PCA-3 | -0.0037 | [-0.0080, +0.0001] |
| C | e5-small | swiltra | PCA-4 | -0.0031 | [-0.0079, +0.0014] |
| C | e5-small | swiltra | PCA-8 | -0.0000 | [-0.0062, +0.0060] |
| C | e5-small | swiltra | PCA-16 | -0.0050 | [-0.0127, +0.0024] |
| C | e5-small | swiltra | PCA-32 | -0.0105 | [-0.0199, -0.0014] |
| C | mini | cap | PCA-1 | -0.0015 | [-0.0064, +0.0032] |
| C | mini | cap | PCA-2 | -0.0012 | [-0.0065, +0.0039] |
| C | mini | cap | PCA-3 | -0.0004 | [-0.0065, +0.0055] |
| C | mini | cap | PCA-4 | -0.0004 | [-0.0073, +0.0064] |
| C | mini | cap | PCA-8 | -0.0040 | [-0.0111, +0.0032] |
| C | mini | cap | PCA-16 | -0.0130 | [-0.0223, -0.0039] |
| C | mini | cap | PCA-32 | -0.0274 | [-0.0394, -0.0153] |
| C | mini | scotus_oyez_lexis | PCA-1 | -0.0034 | [-0.0096, +0.0025] |
| C | mini | scotus_oyez_lexis | PCA-2 | -0.0056 | [-0.0134, +0.0021] |
| C | mini | scotus_oyez_lexis | PCA-3 | -0.0059 | [-0.0131, +0.0016] |
| C | mini | scotus_oyez_lexis | PCA-4 | -0.0054 | [-0.0124, +0.0016] |
| C | mini | scotus_oyez_lexis | PCA-8 | -0.0108 | [-0.0195, -0.0017] |
| C | mini | scotus_oyez_lexis | PCA-16 | -0.0194 | [-0.0297, -0.0088] |
| C | mini | scotus_oyez_lexis | PCA-32 | -0.0300 | [-0.0430, -0.0169] |

## Checks (434 passed)

- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json: paired bootstrap, 10000 resamples, level 0.95, seed 0
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json e5-small/cap: maps fitted on the donor's ID training split, seeds 0-4, then frozen
- e5-small/cap: donor has 3 classes (CAP)
- e5-small/cap: five legal retrieval tasks ['AILACasedocs', 'AILAStatutes', 'LegalBenchConsumerContractsQA', 'LegalBenchCorporateLobbying', 'LegalSummarization'] and STSBenchmark
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json: AILACasedocs and AILAStatutes share the same 50 query ids in the same order, so the evaluator draws one joint resample for both
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json: donor split fingerprints equal the summary's
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json: seeds 0-2 and 3-4 merged
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json: seeds field equals per-seed records
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json: no duplicated seed
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json: donor split fingerprints equal the summary's ['9bb1882d89', '2ef8fc5b56', 'ef4e3dce59', 'df9bc9767e', '365bcd6f71']
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json AILACasedocs: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json AILAStatutes: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json LegalSummarization: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json e5-small/cap pca_top1: legal-macro change recomputed from per-query records (+0.001799)
- e5-small/cap pca_top1 legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/cap pca_top1 STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/cap pca_top1: STS change = mean seed Spearman - Raw
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_cap.json: seeds field equals per-seed records
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_cap.json: no duplicated seed
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_cap.json: donor split fingerprints equal the summary's ['9bb1882d89', '2ef8fc5b56', 'ef4e3dce59', 'df9bc9767e', '365bcd6f71']
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_cap.json AILACasedocs: same query ids and raw scores as ood/ood_e5-small_cap.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_cap.json AILAStatutes: same query ids and raw scores as ood/ood_e5-small_cap.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_cap.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_e5-small_cap.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_cap.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_e5-small_cap.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_cap.json LegalSummarization: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json e5-small/cap kmeans_leace: legal-macro change recomputed from per-query records (+0.001748)
- e5-small/cap kmeans_leace legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/cap kmeans_leace STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/cap kmeans_leace: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json: donor split fingerprints equal the summary's ['9bb1882d89', '2ef8fc5b56', 'ef4e3dce59', 'df9bc9767e', '365bcd6f71']
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json AILACasedocs: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json AILAStatutes: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json LegalSummarization: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json e5-small/cap metadata_leace: legal-macro change recomputed from per-query records (+0.004017)
- e5-small/cap metadata_leace legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/cap metadata_leace STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/cap metadata_leace: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json: donor split fingerprints equal the summary's ['9bb1882d89', '2ef8fc5b56', 'ef4e3dce59', 'df9bc9767e', '365bcd6f71']
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json AILACasedocs: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json AILAStatutes: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json LegalSummarization: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json e5-small/cap sae_leace: legal-macro change recomputed from per-query records (-0.009341)
- e5-small/cap sae_leace legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/cap sae_leace STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/cap sae_leace: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json: donor split fingerprints equal the summary's ['9bb1882d89', '2ef8fc5b56', 'ef4e3dce59', 'df9bc9767e', '365bcd6f71']
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json AILACasedocs: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json AILAStatutes: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_primary_e5-small_cap.json LegalSummarization: same query ids and raw scores as ood/ood_e5-small_cap.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json e5-small/cap pca_matched_var: legal-macro change recomputed from per-query records (-0.005701)
- e5-small/cap pca_matched_var legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/cap pca_matched_var STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/cap pca_matched_var: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json: paired bootstrap, 10000 resamples, level 0.95, seed 0
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json e5-small/scotus_oyez_lexis_wiki: maps fitted on the donor's ID training split, seeds 0-4, then frozen
- e5-small/scotus_oyez_lexis_wiki: donor has 3 classes (SCOTUS (3 sources))
- e5-small/scotus_oyez_lexis_wiki: five legal retrieval tasks ['AILACasedocs', 'AILAStatutes', 'LegalBenchConsumerContractsQA', 'LegalBenchCorporateLobbying', 'LegalSummarization'] and STSBenchmark
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json: AILACasedocs and AILAStatutes share the same 50 query ids in the same order, so the evaluator draws one joint resample for both
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json: donor split fingerprints equal the summary's
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json: seeds 0-2 and 3-4 merged
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json: seeds field equals per-seed records
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json: no duplicated seed
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json: donor split fingerprints equal the summary's ['992a84d7c9', 'a884540068', 'ac2c2f3ca7', '0658791625', '24a5d3f8d1']
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json AILACasedocs: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json AILAStatutes: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json LegalSummarization: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json e5-small/scotus_oyez_lexis_wiki pca_top1: legal-macro change recomputed from per-query records (-0.000295)
- e5-small/scotus_oyez_lexis_wiki pca_top1 legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/scotus_oyez_lexis_wiki pca_top1 STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/scotus_oyez_lexis_wiki pca_top1: STS change = mean seed Spearman - Raw
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_scotus_oyez_lexis_wiki.json: seeds field equals per-seed records
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_scotus_oyez_lexis_wiki.json: no duplicated seed
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_scotus_oyez_lexis_wiki.json: donor split fingerprints equal the summary's ['992a84d7c9', 'a884540068', 'ac2c2f3ca7', '0658791625', '24a5d3f8d1']
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_scotus_oyez_lexis_wiki.json AILACasedocs: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_scotus_oyez_lexis_wiki.json AILAStatutes: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_scotus_oyez_lexis_wiki.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_scotus_oyez_lexis_wiki.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_scotus_oyez_lexis_wiki.json LegalSummarization: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json e5-small/scotus_oyez_lexis_wiki kmeans_leace: legal-macro change recomputed from per-query records (+0.000567)
- e5-small/scotus_oyez_lexis_wiki kmeans_leace legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/scotus_oyez_lexis_wiki kmeans_leace STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/scotus_oyez_lexis_wiki kmeans_leace: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json: donor split fingerprints equal the summary's ['992a84d7c9', 'a884540068', 'ac2c2f3ca7', '0658791625', '24a5d3f8d1']
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json AILACasedocs: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json AILAStatutes: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json LegalSummarization: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json e5-small/scotus_oyez_lexis_wiki metadata_leace: legal-macro change recomputed from per-query records (-0.000125)
- e5-small/scotus_oyez_lexis_wiki metadata_leace legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/scotus_oyez_lexis_wiki metadata_leace STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/scotus_oyez_lexis_wiki metadata_leace: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json: donor split fingerprints equal the summary's ['992a84d7c9', 'a884540068', 'ac2c2f3ca7', '0658791625', '24a5d3f8d1']
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json AILACasedocs: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json AILAStatutes: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json LegalSummarization: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json e5-small/scotus_oyez_lexis_wiki sae_leace: legal-macro change recomputed from per-query records (-0.009445)
- e5-small/scotus_oyez_lexis_wiki sae_leace legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/scotus_oyez_lexis_wiki sae_leace STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/scotus_oyez_lexis_wiki sae_leace: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json: donor split fingerprints equal the summary's ['992a84d7c9', 'a884540068', 'ac2c2f3ca7', '0658791625', '24a5d3f8d1']
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json AILACasedocs: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json AILAStatutes: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_e5-small_scotus_oyez_lexis_wiki.json LegalSummarization: same query ids and raw scores as ood/ood_e5-small_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json e5-small/scotus_oyez_lexis_wiki pca_matched_var: legal-macro change recomputed from per-query records (-0.012017)
- e5-small/scotus_oyez_lexis_wiki pca_matched_var legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/scotus_oyez_lexis_wiki pca_matched_var STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/scotus_oyez_lexis_wiki pca_matched_var: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json: paired bootstrap, 10000 resamples, level 0.95, seed 0
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json e5-small/swiltra_de_fr_it: maps fitted on the donor's ID training split, seeds 0-4, then frozen
- e5-small/swiltra_de_fr_it: donor has 3 classes (SwiLTra (3 languages))
- e5-small/swiltra_de_fr_it: five legal retrieval tasks ['AILACasedocs', 'AILAStatutes', 'LegalBenchConsumerContractsQA', 'LegalBenchCorporateLobbying', 'LegalSummarization'] and STSBenchmark
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json: AILACasedocs and AILAStatutes share the same 50 query ids in the same order, so the evaluator draws one joint resample for both
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json: donor split fingerprints equal the summary's
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json: seeds 0-2 and 3-4 merged
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json: seeds field equals per-seed records
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json: no duplicated seed
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json: donor split fingerprints equal the summary's ['eba6fc84ce', 'fbb2240430', '4933d049ff', 'b7ecf30998', '1ccaa91ee4']
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json AILACasedocs: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json AILAStatutes: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json LegalSummarization: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json e5-small/swiltra_de_fr_it pca_top1: legal-macro change recomputed from per-query records (-0.000896)
- e5-small/swiltra_de_fr_it pca_top1 legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/swiltra_de_fr_it pca_top1 STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/swiltra_de_fr_it pca_top1: STS change = mean seed Spearman - Raw
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_swiltra_de_fr_it.json: seeds field equals per-seed records
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_swiltra_de_fr_it.json: no duplicated seed
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_swiltra_de_fr_it.json: donor split fingerprints equal the summary's ['eba6fc84ce', 'fbb2240430', '4933d049ff', 'b7ecf30998', '1ccaa91ee4']
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_swiltra_de_fr_it.json AILACasedocs: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_swiltra_de_fr_it.json AILAStatutes: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_swiltra_de_fr_it.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_swiltra_de_fr_it.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_e5-small_swiltra_de_fr_it.json LegalSummarization: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json e5-small/swiltra_de_fr_it kmeans_leace: legal-macro change recomputed from per-query records (-0.000321)
- e5-small/swiltra_de_fr_it kmeans_leace legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/swiltra_de_fr_it kmeans_leace STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/swiltra_de_fr_it kmeans_leace: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json: donor split fingerprints equal the summary's ['eba6fc84ce', 'fbb2240430', '4933d049ff', 'b7ecf30998', '1ccaa91ee4']
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json AILACasedocs: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json AILAStatutes: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json LegalSummarization: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json e5-small/swiltra_de_fr_it metadata_leace: legal-macro change recomputed from per-query records (-0.000321)
- e5-small/swiltra_de_fr_it metadata_leace legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/swiltra_de_fr_it metadata_leace STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/swiltra_de_fr_it metadata_leace: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json: donor split fingerprints equal the summary's ['eba6fc84ce', 'fbb2240430', '4933d049ff', 'b7ecf30998', '1ccaa91ee4']
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json AILACasedocs: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json AILAStatutes: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json LegalSummarization: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json e5-small/swiltra_de_fr_it sae_leace: legal-macro change recomputed from per-query records (-0.007396)
- e5-small/swiltra_de_fr_it sae_leace legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/swiltra_de_fr_it sae_leace STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/swiltra_de_fr_it sae_leace: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json: donor split fingerprints equal the summary's ['eba6fc84ce', 'fbb2240430', '4933d049ff', 'b7ecf30998', '1ccaa91ee4']
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json AILACasedocs: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json AILAStatutes: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_primary_e5-small_swiltra_de_fr_it.json LegalSummarization: same query ids and raw scores as ood/ood_e5-small_swiltra_de_fr_it.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json e5-small/swiltra_de_fr_it pca_matched_var: legal-macro change recomputed from per-query records (-0.007769)
- e5-small/swiltra_de_fr_it pca_matched_var legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/swiltra_de_fr_it pca_matched_var STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- e5-small/swiltra_de_fr_it pca_matched_var: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json: paired bootstrap, 10000 resamples, level 0.95, seed 0
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json mini/cap: maps fitted on the donor's ID training split, seeds 0-4, then frozen
- mini/cap: donor has 3 classes (CAP)
- mini/cap: five legal retrieval tasks ['AILACasedocs', 'AILAStatutes', 'LegalBenchConsumerContractsQA', 'LegalBenchCorporateLobbying', 'LegalSummarization'] and STSBenchmark
- paper-artifacts/derived/ood/ood_primary_mini_cap.json: AILACasedocs and AILAStatutes share the same 50 query ids in the same order, so the evaluator draws one joint resample for both
- paper-artifacts/derived/ood/ood_primary_mini_cap.json: donor split fingerprints equal the summary's
- paper-artifacts/derived/ood/ood_primary_mini_cap.json: seeds 0-2 and 3-4 merged
- paper-artifacts/derived/ood/ood_primary_mini_cap.json: seeds field equals per-seed records
- paper-artifacts/derived/ood/ood_primary_mini_cap.json: no duplicated seed
- paper-artifacts/derived/ood/ood_primary_mini_cap.json: donor split fingerprints equal the summary's ['9bb1882d89', '2ef8fc5b56', 'ef4e3dce59', 'df9bc9767e', '365bcd6f71']
- paper-artifacts/derived/ood/ood_primary_mini_cap.json AILACasedocs: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_primary_mini_cap.json AILAStatutes: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_primary_mini_cap.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_primary_mini_cap.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_primary_mini_cap.json LegalSummarization: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json mini/cap pca_top1: legal-macro change recomputed from per-query records (-0.001519)
- mini/cap pca_top1 legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/cap pca_top1 STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/cap pca_top1: STS change = mean seed Spearman - Raw
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_mini_cap.json: seeds field equals per-seed records
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_mini_cap.json: no duplicated seed
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_mini_cap.json: donor split fingerprints equal the summary's ['9bb1882d89', '2ef8fc5b56', 'ef4e3dce59', 'df9bc9767e', '365bcd6f71']
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_mini_cap.json AILACasedocs: same query ids and raw scores as ood/ood_mini_cap.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_mini_cap.json AILAStatutes: same query ids and raw scores as ood/ood_mini_cap.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_mini_cap.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_mini_cap.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_mini_cap.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_mini_cap.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_mini_cap.json LegalSummarization: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json mini/cap kmeans_leace: legal-macro change recomputed from per-query records (-0.000377)
- mini/cap kmeans_leace legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/cap kmeans_leace STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/cap kmeans_leace: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_primary_mini_cap.json: donor split fingerprints equal the summary's ['9bb1882d89', '2ef8fc5b56', 'ef4e3dce59', 'df9bc9767e', '365bcd6f71']
- paper-artifacts/derived/ood/ood_primary_mini_cap.json AILACasedocs: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_primary_mini_cap.json AILAStatutes: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_primary_mini_cap.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_primary_mini_cap.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_primary_mini_cap.json LegalSummarization: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json mini/cap metadata_leace: legal-macro change recomputed from per-query records (-0.002447)
- mini/cap metadata_leace legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/cap metadata_leace STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/cap metadata_leace: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_primary_mini_cap.json: donor split fingerprints equal the summary's ['9bb1882d89', '2ef8fc5b56', 'ef4e3dce59', 'df9bc9767e', '365bcd6f71']
- paper-artifacts/derived/ood/ood_primary_mini_cap.json AILACasedocs: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_primary_mini_cap.json AILAStatutes: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_primary_mini_cap.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_primary_mini_cap.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_primary_mini_cap.json LegalSummarization: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json mini/cap sae_leace: legal-macro change recomputed from per-query records (-0.019458)
- mini/cap sae_leace legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/cap sae_leace STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/cap sae_leace: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_primary_mini_cap.json: donor split fingerprints equal the summary's ['9bb1882d89', '2ef8fc5b56', 'ef4e3dce59', 'df9bc9767e', '365bcd6f71']
- paper-artifacts/derived/ood/ood_primary_mini_cap.json AILACasedocs: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_primary_mini_cap.json AILAStatutes: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_primary_mini_cap.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_primary_mini_cap.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_primary_mini_cap.json LegalSummarization: same query ids and raw scores as ood/ood_mini_cap.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json mini/cap pca_matched_var: legal-macro change recomputed from per-query records (-0.017030)
- mini/cap pca_matched_var legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/cap pca_matched_var STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/cap pca_matched_var: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json: paired bootstrap, 10000 resamples, level 0.95, seed 0
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json mini/scotus_oyez_lexis_wiki: maps fitted on the donor's ID training split, seeds 0-4, then frozen
- mini/scotus_oyez_lexis_wiki: donor has 3 classes (SCOTUS (3 sources))
- mini/scotus_oyez_lexis_wiki: five legal retrieval tasks ['AILACasedocs', 'AILAStatutes', 'LegalBenchConsumerContractsQA', 'LegalBenchCorporateLobbying', 'LegalSummarization'] and STSBenchmark
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json: AILACasedocs and AILAStatutes share the same 50 query ids in the same order, so the evaluator draws one joint resample for both
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json: donor split fingerprints equal the summary's
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json: seeds 0-2 and 3-4 merged
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json: seeds field equals per-seed records
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json: no duplicated seed
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json: donor split fingerprints equal the summary's ['992a84d7c9', 'a884540068', 'ac2c2f3ca7', '0658791625', '24a5d3f8d1']
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json AILACasedocs: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json AILAStatutes: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json LegalSummarization: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json mini/scotus_oyez_lexis_wiki pca_top1: legal-macro change recomputed from per-query records (-0.002996)
- mini/scotus_oyez_lexis_wiki pca_top1 legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/scotus_oyez_lexis_wiki pca_top1 STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/scotus_oyez_lexis_wiki pca_top1: STS change = mean seed Spearman - Raw
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_mini_scotus_oyez_lexis_wiki.json: seeds field equals per-seed records
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_mini_scotus_oyez_lexis_wiki.json: no duplicated seed
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_mini_scotus_oyez_lexis_wiki.json: donor split fingerprints equal the summary's ['992a84d7c9', 'a884540068', 'ac2c2f3ca7', '0658791625', '24a5d3f8d1']
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_mini_scotus_oyez_lexis_wiki.json AILACasedocs: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_mini_scotus_oyez_lexis_wiki.json AILAStatutes: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_mini_scotus_oyez_lexis_wiki.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_mini_scotus_oyez_lexis_wiki.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- results/extensions/raw/five_seeds/alternatives/ood_kmeans_mini_scotus_oyez_lexis_wiki.json LegalSummarization: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json mini/scotus_oyez_lexis_wiki kmeans_leace: legal-macro change recomputed from per-query records (-0.001906)
- mini/scotus_oyez_lexis_wiki kmeans_leace legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/scotus_oyez_lexis_wiki kmeans_leace STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/scotus_oyez_lexis_wiki kmeans_leace: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json: donor split fingerprints equal the summary's ['992a84d7c9', 'a884540068', 'ac2c2f3ca7', '0658791625', '24a5d3f8d1']
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json AILACasedocs: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json AILAStatutes: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json LegalSummarization: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json mini/scotus_oyez_lexis_wiki metadata_leace: legal-macro change recomputed from per-query records (+0.005131)
- mini/scotus_oyez_lexis_wiki metadata_leace legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/scotus_oyez_lexis_wiki metadata_leace STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/scotus_oyez_lexis_wiki metadata_leace: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json: donor split fingerprints equal the summary's ['992a84d7c9', 'a884540068', 'ac2c2f3ca7', '0658791625', '24a5d3f8d1']
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json AILACasedocs: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json AILAStatutes: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json LegalSummarization: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json mini/scotus_oyez_lexis_wiki sae_leace: legal-macro change recomputed from per-query records (-0.017751)
- mini/scotus_oyez_lexis_wiki sae_leace legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/scotus_oyez_lexis_wiki sae_leace STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/scotus_oyez_lexis_wiki sae_leace: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json: donor split fingerprints equal the summary's ['992a84d7c9', 'a884540068', 'ac2c2f3ca7', '0658791625', '24a5d3f8d1']
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json AILACasedocs: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json AILAStatutes: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json LegalBenchConsumerContractsQA: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json LegalBenchCorporateLobbying: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_primary_mini_scotus_oyez_lexis_wiki.json LegalSummarization: same query ids and raw scores as ood/ood_mini_scotus_oyez_lexis_wiki.json
- paper-artifacts/derived/ood/ood_summary_primary_five_seeds.json mini/scotus_oyez_lexis_wiki pca_matched_var: legal-macro change recomputed from per-query records (-0.016132)
- mini/scotus_oyez_lexis_wiki pca_matched_var legal_macro: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/scotus_oyez_lexis_wiki pca_matched_var STSBenchmark: saved change and 95% interval reproduced from the per-query outputs with ood.summarize
- mini/scotus_oyez_lexis_wiki pca_matched_var: STS change = mean seed Spearman - Raw
- paper-artifacts/derived/ood/ood_summary_ranks_five_seeds.json: paired bootstrap, 10000 resamples, level 0.95, seed 0
- paper-artifacts/derived/ood/ood_summary_ranks_five_seeds.json e5-small/cap: maps fitted on the donor's ID training split, seeds 0-4, then frozen
- e5-small/cap: donor has 3 classes (CAP)
- e5-small/cap: five legal retrieval tasks ['AILACasedocs', 'AILAStatutes', 'LegalBenchConsumerContractsQA', 'LegalBenchCorporateLobbying', 'LegalSummarization'] and STSBenchmark
- paper-artifacts/derived/ood/ood_ranks_e5-small_cap.json: AILACasedocs and AILAStatutes share the same 50 query ids in the same order, so the evaluator draws one joint resample for both
- paper-artifacts/derived/ood/ood_ranks_e5-small_cap.json: donor split fingerprints equal the summary's
- paper-artifacts/derived/ood/ood_ranks_e5-small_cap.json: seeds 0-2 and 3-4 merged
- e5-small/cap: plotted ranks [1, 2, 3, 4, 8, 16, 32] are exactly the evaluated ones (rank 1 from the plan run, the rest from the rank-sweep run)
- e5-small/cap PCA-1 legal macro: saved change and 95% interval reproduced with ood.summarize
- e5-small/cap PCA-2 legal macro: saved change and 95% interval reproduced with ood.summarize
- e5-small/cap PCA-3 legal macro: saved change and 95% interval reproduced with ood.summarize
- e5-small/cap PCA-4 legal macro: saved change and 95% interval reproduced with ood.summarize
- e5-small/cap PCA-8 legal macro: saved change and 95% interval reproduced with ood.summarize
- e5-small/cap PCA-16 legal macro: saved change and 95% interval reproduced with ood.summarize
- e5-small/cap PCA-32 legal macro: saved change and 95% interval reproduced with ood.summarize
- paper-artifacts/derived/ood/ood_summary_ranks_five_seeds.json: paired bootstrap, 10000 resamples, level 0.95, seed 0
- paper-artifacts/derived/ood/ood_summary_ranks_five_seeds.json e5-small/scotus_oyez_lexis: maps fitted on the donor's ID training split, seeds 0-4, then frozen
- e5-small/scotus_oyez_lexis: donor has 2 classes (SCOTUS Oyez–Lexis (binary))
- e5-small/scotus_oyez_lexis: five legal retrieval tasks ['AILACasedocs', 'AILAStatutes', 'LegalBenchConsumerContractsQA', 'LegalBenchCorporateLobbying', 'LegalSummarization'] and STSBenchmark
- paper-artifacts/derived/ood/ood_ranks_e5-small_scotus_oyez_lexis.json: AILACasedocs and AILAStatutes share the same 50 query ids in the same order, so the evaluator draws one joint resample for both
- paper-artifacts/derived/ood/ood_ranks_e5-small_scotus_oyez_lexis.json: donor split fingerprints equal the summary's
- paper-artifacts/derived/ood/ood_ranks_e5-small_scotus_oyez_lexis.json: seeds 0-2 and 3-4 merged
- e5-small/scotus_oyez_lexis: plotted ranks [1, 2, 3, 4, 8, 16, 32] are exactly the evaluated ones (rank 1 from the plan run, the rest from the rank-sweep run)
- e5-small/scotus_oyez_lexis PCA-1 legal macro: saved change and 95% interval reproduced with ood.summarize
- e5-small/scotus_oyez_lexis PCA-2 legal macro: saved change and 95% interval reproduced with ood.summarize
- e5-small/scotus_oyez_lexis PCA-3 legal macro: saved change and 95% interval reproduced with ood.summarize
- e5-small/scotus_oyez_lexis PCA-4 legal macro: saved change and 95% interval reproduced with ood.summarize
- e5-small/scotus_oyez_lexis PCA-8 legal macro: saved change and 95% interval reproduced with ood.summarize
- e5-small/scotus_oyez_lexis PCA-16 legal macro: saved change and 95% interval reproduced with ood.summarize
- e5-small/scotus_oyez_lexis PCA-32 legal macro: saved change and 95% interval reproduced with ood.summarize
- paper-artifacts/derived/ood/ood_summary_ranks_five_seeds.json: paired bootstrap, 10000 resamples, level 0.95, seed 0
- paper-artifacts/derived/ood/ood_summary_ranks_five_seeds.json e5-small/swiltra: maps fitted on the donor's ID training split, seeds 0-4, then frozen
- e5-small/swiltra: donor has 2 classes (SwiLTra DE–FR (binary))
- e5-small/swiltra: five legal retrieval tasks ['AILACasedocs', 'AILAStatutes', 'LegalBenchConsumerContractsQA', 'LegalBenchCorporateLobbying', 'LegalSummarization'] and STSBenchmark
- paper-artifacts/derived/ood/ood_ranks_e5-small_swiltra.json: AILACasedocs and AILAStatutes share the same 50 query ids in the same order, so the evaluator draws one joint resample for both
- paper-artifacts/derived/ood/ood_ranks_e5-small_swiltra.json: donor split fingerprints equal the summary's
- paper-artifacts/derived/ood/ood_ranks_e5-small_swiltra.json: seeds 0-2 and 3-4 merged
- e5-small/swiltra: plotted ranks [1, 2, 3, 4, 8, 16, 32] are exactly the evaluated ones (rank 1 from the plan run, the rest from the rank-sweep run)
- e5-small/swiltra PCA-1 legal macro: saved change and 95% interval reproduced with ood.summarize
- e5-small/swiltra PCA-2 legal macro: saved change and 95% interval reproduced with ood.summarize
- e5-small/swiltra PCA-3 legal macro: saved change and 95% interval reproduced with ood.summarize
- e5-small/swiltra PCA-4 legal macro: saved change and 95% interval reproduced with ood.summarize
- e5-small/swiltra PCA-8 legal macro: saved change and 95% interval reproduced with ood.summarize
- e5-small/swiltra PCA-16 legal macro: saved change and 95% interval reproduced with ood.summarize
- e5-small/swiltra PCA-32 legal macro: saved change and 95% interval reproduced with ood.summarize
- paper-artifacts/derived/ood/ood_summary_ranks_five_seeds.json: paired bootstrap, 10000 resamples, level 0.95, seed 0
- paper-artifacts/derived/ood/ood_summary_ranks_five_seeds.json mini/cap: maps fitted on the donor's ID training split, seeds 0-4, then frozen
- mini/cap: donor has 3 classes (CAP)
- mini/cap: five legal retrieval tasks ['AILACasedocs', 'AILAStatutes', 'LegalBenchConsumerContractsQA', 'LegalBenchCorporateLobbying', 'LegalSummarization'] and STSBenchmark
- paper-artifacts/derived/ood/ood_ranks_mini_cap.json: AILACasedocs and AILAStatutes share the same 50 query ids in the same order, so the evaluator draws one joint resample for both
- paper-artifacts/derived/ood/ood_ranks_mini_cap.json: donor split fingerprints equal the summary's
- paper-artifacts/derived/ood/ood_ranks_mini_cap.json: seeds 0-2 and 3-4 merged
- mini/cap: plotted ranks [1, 2, 3, 4, 8, 16, 32] are exactly the evaluated ones (rank 1 from the plan run, the rest from the rank-sweep run)
- mini/cap PCA-1 legal macro: saved change and 95% interval reproduced with ood.summarize
- mini/cap PCA-2 legal macro: saved change and 95% interval reproduced with ood.summarize
- mini/cap PCA-3 legal macro: saved change and 95% interval reproduced with ood.summarize
- mini/cap PCA-4 legal macro: saved change and 95% interval reproduced with ood.summarize
- mini/cap PCA-8 legal macro: saved change and 95% interval reproduced with ood.summarize
- mini/cap PCA-16 legal macro: saved change and 95% interval reproduced with ood.summarize
- mini/cap PCA-32 legal macro: saved change and 95% interval reproduced with ood.summarize
- paper-artifacts/derived/ood/ood_summary_ranks_five_seeds.json: paired bootstrap, 10000 resamples, level 0.95, seed 0
- paper-artifacts/derived/ood/ood_summary_ranks_five_seeds.json mini/scotus_oyez_lexis: maps fitted on the donor's ID training split, seeds 0-4, then frozen
- mini/scotus_oyez_lexis: donor has 2 classes (SCOTUS Oyez–Lexis (binary))
- mini/scotus_oyez_lexis: five legal retrieval tasks ['AILACasedocs', 'AILAStatutes', 'LegalBenchConsumerContractsQA', 'LegalBenchCorporateLobbying', 'LegalSummarization'] and STSBenchmark
- paper-artifacts/derived/ood/ood_ranks_mini_scotus_oyez_lexis.json: AILACasedocs and AILAStatutes share the same 50 query ids in the same order, so the evaluator draws one joint resample for both
- paper-artifacts/derived/ood/ood_ranks_mini_scotus_oyez_lexis.json: donor split fingerprints equal the summary's
- paper-artifacts/derived/ood/ood_ranks_mini_scotus_oyez_lexis.json: seeds 0-2 and 3-4 merged
- mini/scotus_oyez_lexis: plotted ranks [1, 2, 3, 4, 8, 16, 32] are exactly the evaluated ones (rank 1 from the plan run, the rest from the rank-sweep run)
- mini/scotus_oyez_lexis PCA-1 legal macro: saved change and 95% interval reproduced with ood.summarize
- mini/scotus_oyez_lexis PCA-2 legal macro: saved change and 95% interval reproduced with ood.summarize
- mini/scotus_oyez_lexis PCA-3 legal macro: saved change and 95% interval reproduced with ood.summarize
- mini/scotus_oyez_lexis PCA-4 legal macro: saved change and 95% interval reproduced with ood.summarize
- mini/scotus_oyez_lexis PCA-8 legal macro: saved change and 95% interval reproduced with ood.summarize
- mini/scotus_oyez_lexis PCA-16 legal macro: saved change and 95% interval reproduced with ood.summarize
- mini/scotus_oyez_lexis PCA-32 legal macro: saved change and 95% interval reproduced with ood.summarize
- figB3_frozen_transfer_rank: smallest font 6.8 pt >= 6.5 pt
- figB3_frozen_transfer_rank: no text outside the canvas
- results/extensions/raw/transfer_matched/transfer_matched_five.json: every donor's maps are those of seeds 0-4
- e5-small/cap pca_top1 legal_macro: the change is the mean of the five fitted maps' changes
- e5-small/cap pca_top1 STSBenchmark: the change is the mean of the five fitted maps' changes
- e5-small/cap kmeans_leace legal_macro: the change is the mean of the five fitted maps' changes
- e5-small/cap kmeans_leace STSBenchmark: the change is the mean of the five fitted maps' changes
- e5-small/cap metadata_leace legal_macro: the change is the mean of the five fitted maps' changes
- e5-small/cap metadata_leace STSBenchmark: the change is the mean of the five fitted maps' changes
- e5-small/cap sae_leace legal_macro: the change is the mean of the five fitted maps' changes
- e5-small/cap sae_leace STSBenchmark: the change is the mean of the five fitted maps' changes
- e5-small/cap pca_matched_var legal_macro: the change is the mean of the five fitted maps' changes
- e5-small/cap pca_matched_var STSBenchmark: the change is the mean of the five fitted maps' changes
- e5-small/scotus_oyez_lexis_wiki pca_top1 legal_macro: the change is the mean of the five fitted maps' changes
- e5-small/scotus_oyez_lexis_wiki pca_top1 STSBenchmark: the change is the mean of the five fitted maps' changes
- e5-small/scotus_oyez_lexis_wiki kmeans_leace legal_macro: the change is the mean of the five fitted maps' changes
- e5-small/scotus_oyez_lexis_wiki kmeans_leace STSBenchmark: the change is the mean of the five fitted maps' changes
- e5-small/scotus_oyez_lexis_wiki metadata_leace legal_macro: the change is the mean of the five fitted maps' changes
- e5-small/scotus_oyez_lexis_wiki metadata_leace STSBenchmark: the change is the mean of the five fitted maps' changes
- e5-small/scotus_oyez_lexis_wiki sae_leace legal_macro: the change is the mean of the five fitted maps' changes
- e5-small/scotus_oyez_lexis_wiki sae_leace STSBenchmark: the change is the mean of the five fitted maps' changes
- e5-small/scotus_oyez_lexis_wiki pca_matched_var legal_macro: the change is the mean of the five fitted maps' changes
- e5-small/scotus_oyez_lexis_wiki pca_matched_var STSBenchmark: the change is the mean of the five fitted maps' changes
- e5-small/swiltra_de_fr_it pca_top1 legal_macro: the change is the mean of the five fitted maps' changes
- e5-small/swiltra_de_fr_it pca_top1 STSBenchmark: the change is the mean of the five fitted maps' changes
- e5-small/swiltra_de_fr_it kmeans_leace legal_macro: the change is the mean of the five fitted maps' changes
- e5-small/swiltra_de_fr_it kmeans_leace STSBenchmark: the change is the mean of the five fitted maps' changes
- e5-small/swiltra_de_fr_it metadata_leace legal_macro: the change is the mean of the five fitted maps' changes
- e5-small/swiltra_de_fr_it metadata_leace STSBenchmark: the change is the mean of the five fitted maps' changes
- e5-small/swiltra_de_fr_it sae_leace legal_macro: the change is the mean of the five fitted maps' changes
- e5-small/swiltra_de_fr_it sae_leace STSBenchmark: the change is the mean of the five fitted maps' changes
- e5-small/swiltra_de_fr_it pca_matched_var legal_macro: the change is the mean of the five fitted maps' changes
- e5-small/swiltra_de_fr_it pca_matched_var STSBenchmark: the change is the mean of the five fitted maps' changes
- mini/cap pca_top1 legal_macro: the change is the mean of the five fitted maps' changes
- mini/cap pca_top1 STSBenchmark: the change is the mean of the five fitted maps' changes
- mini/cap kmeans_leace legal_macro: the change is the mean of the five fitted maps' changes
- mini/cap kmeans_leace STSBenchmark: the change is the mean of the five fitted maps' changes
- mini/cap metadata_leace legal_macro: the change is the mean of the five fitted maps' changes
- mini/cap metadata_leace STSBenchmark: the change is the mean of the five fitted maps' changes
- mini/cap sae_leace legal_macro: the change is the mean of the five fitted maps' changes
- mini/cap sae_leace STSBenchmark: the change is the mean of the five fitted maps' changes
- mini/cap pca_matched_var legal_macro: the change is the mean of the five fitted maps' changes
- mini/cap pca_matched_var STSBenchmark: the change is the mean of the five fitted maps' changes
- mini/scotus_oyez_lexis_wiki pca_top1 legal_macro: the change is the mean of the five fitted maps' changes
- mini/scotus_oyez_lexis_wiki pca_top1 STSBenchmark: the change is the mean of the five fitted maps' changes
- mini/scotus_oyez_lexis_wiki kmeans_leace legal_macro: the change is the mean of the five fitted maps' changes
- mini/scotus_oyez_lexis_wiki kmeans_leace STSBenchmark: the change is the mean of the five fitted maps' changes
- mini/scotus_oyez_lexis_wiki metadata_leace legal_macro: the change is the mean of the five fitted maps' changes
- mini/scotus_oyez_lexis_wiki metadata_leace STSBenchmark: the change is the mean of the five fitted maps' changes
- mini/scotus_oyez_lexis_wiki sae_leace legal_macro: the change is the mean of the five fitted maps' changes
- mini/scotus_oyez_lexis_wiki sae_leace STSBenchmark: the change is the mean of the five fitted maps' changes
- mini/scotus_oyez_lexis_wiki pca_matched_var legal_macro: the change is the mean of the five fitted maps' changes
- mini/scotus_oyez_lexis_wiki pca_matched_var STSBenchmark: the change is the mean of the five fitted maps' changes
- results/extensions/raw/transfer_matched/transfer_matched_five.json: its changes and bootstrap intervals for the five maps, donors and both targets equal the figure's (largest difference 0.0e+00)

## Issues

None.
