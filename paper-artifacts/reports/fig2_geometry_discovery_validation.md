# Figure 2 (three panels + composite): provenance and validation

Build: `python paper-artifacts/build.py fig2_geometry_discovery`. Values from the shared collector (identical to the earlier Figure 2 versions). Descriptive only.

| # | Setting | C | K | capture(C−1) | PCs 90% | ARI | Meta AUC | PC-1 gap | k-means gap | SAE gap | geometry seeds | performance |
|---:|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | E5 · SwiLTra DE–FR | 2 | 2 | 0.997 | 1 | 0.995 | 0.600 | 0.032 | 0.000 | 0.025 | [0, 1, 2, 3, 4] | id [0, 1, 2, 3, 4] |
| 2 | E5 · SwiLTra DE–FR–IT | 3 | 3 | 0.992 | 2 | 0.996 | 0.577 | 0.329 | 0.000 | 0.038 | [0, 1, 2, 3, 4] | id [0, 1, 2, 3, 4] |
| 3 | E5 · SCOTUS Oyez–Lexis | 2 | 2 | 0.984 | 1 | 0.846 | 0.622 | 0.093 | 0.050 | 0.079 | [0, 1, 2, 3, 4] | coverage [0, 1, 2, 3, 4] |
| 4 | MiniLM · SCOTUS Oyez–Lexis | 2 | 2 | 0.049 | 22–29 | 0.002 | 0.622 | 0.355 | 0.356 | 0.219 | [0, 1, 2, 3, 4] | id [0, 1, 2, 3, 4] |
| 5 | E5 · SCOTUS 3 sources | 3 | 3 | 0.870 | 2–3 | 0.743 | 0.615 | 0.267 | 0.084 | 0.112 | [0, 1, 2, 3, 4] | coverage [0, 1, 2, 3, 4] |
| 6 | MiniLM · SCOTUS 3 sources | 3 | 2–5 | 0.271 | 17–19 | 0.081 | 0.611 | 0.375 | 0.340 | 0.250 | [0, 1, 2, 3, 4] | id [0, 1, 2, 3, 4] |

## Cohorts

Geometry (A, B) uses seeds 0–2 in every row; C uses each setting's locked cohort. The populations differ in:

## Checks (182 passed)

- results/extensions/raw/five_seeds/alternatives/geometry_swiltra_e5-small.json: seeds field equals per-seed records
- results/extensions/raw/five_seeds/alternatives/geometry_swiltra_e5-small.json: no duplicated seed
- results/extensions/raw/five_seeds/alternatives/geometry_swiltra_e5-small.json: AUCs saved without flip adjustment
- results/extensions/raw/five_seeds/alternatives/geometry_swiltra_e5-small.json: the geometry run of swiltra/e5-small, seeds [0, 1, 2, 3, 4]
- results/extensions/raw/five_seeds/alternatives/geometry_swiltra_e5-small.json: saved geometry means equal the per-seed fold means
- swiltra/e5-small: geometry run has C=2, nuisance subspace rank 1
- swiltra/e5-small: geometry uses the same corpus version and split policy
- swiltra/e5-small: capture recorded at rank 1 and at C-1=1
- id/id_swiltra_e5-small.json: seeds field equals per-seed records
- id/id_swiltra_e5-small.json: no duplicated seed
- id/id_swiltra_e5-small.json: AUCs saved without flip adjustment
- id/id_swiltra_e5-small.json: saved aggregates equal per-seed values, mean and SD (ddof=1)
- id/id_swiltra_e5-small.json: each seed's AUC/purity equals the mean of its saved fold values
- id/id_swiltra_e5-small.json: each seed's R@10 equals the mean of its saved per-pair hits
- id/id_swiltra_e5-small.json: corpus/encoder fields are swiltra/e5-small
- id/id_swiltra_e5-small.json: corpus/encoder fields are swiltra/e5-small
- id/id_kmeans_swiltra_e5-small.json: seeds field equals per-seed records
- id/id_kmeans_swiltra_e5-small.json: no duplicated seed
- id/id_kmeans_swiltra_e5-small.json: AUCs saved without flip adjustment
- id/id_kmeans_swiltra_e5-small.json: saved aggregates equal per-seed values, mean and SD (ddof=1)
- id/id_kmeans_swiltra_e5-small.json: each seed's AUC/purity equals the mean of its saved fold values
- id/id_kmeans_swiltra_e5-small.json: each seed's R@10 equals the mean of its saved per-pair hits
- id/id_kmeans_swiltra_e5-small.json: corpus/encoder fields are swiltra/e5-small
- id:swiltra/e5-small main~kmeans: config.split identical
- id:swiltra/e5-small main~kmeans: config.audit identical
- id:swiltra/e5-small main~kmeans: config.selection identical
- id:swiltra/e5-small main~kmeans: config.pseudo_groups identical
- id:swiltra/e5-small main~kmeans: SAE/eraser recipe identical
- id:swiltra/e5-small main~kmeans: corpus statistics identical
- id:swiltra/e5-small main~kmeans: per-seed split summaries identical on seeds [0, 1, 2, 3, 4]
- id/id_swiltra_e5-small.json: corpus/encoder fields are swiltra/e5-small
- id/id_kmeans_swiltra_e5-small.json: corpus/encoder fields are swiltra/e5-small
- id/id_swiltra_e5-small.json: corpus/encoder fields are swiltra/e5-small
- swiltra/e5-small: silhouette K identical in geometry and performance runs on seeds [0, 1, 2, 3, 4]
- results/extensions/raw/five_seeds/alternatives/geometry_swiltra_de_fr_it_e5-small.json: seeds field equals per-seed records
- results/extensions/raw/five_seeds/alternatives/geometry_swiltra_de_fr_it_e5-small.json: no duplicated seed
- results/extensions/raw/five_seeds/alternatives/geometry_swiltra_de_fr_it_e5-small.json: AUCs saved without flip adjustment
- results/extensions/raw/five_seeds/alternatives/geometry_swiltra_de_fr_it_e5-small.json: the geometry run of swiltra_de_fr_it/e5-small, seeds [0, 1, 2, 3, 4]
- results/extensions/raw/five_seeds/alternatives/geometry_swiltra_de_fr_it_e5-small.json: saved geometry means equal the per-seed fold means
- swiltra_de_fr_it/e5-small: geometry run has C=3, nuisance subspace rank 2
- swiltra_de_fr_it/e5-small: geometry uses the same corpus version and split policy
- swiltra_de_fr_it/e5-small: capture recorded at rank 1 and at C-1=2
- id/id_swiltra_de_fr_it_e5-small.json: corpus/encoder fields are swiltra_de_fr_it/e5-small
- id/id_swiltra_de_fr_it_e5-small.json: corpus/encoder fields are swiltra_de_fr_it/e5-small
- id/id_kmeans_swiltra_de_fr_it_e5-small.json: corpus/encoder fields are swiltra_de_fr_it/e5-small
- id/id_swiltra_de_fr_it_e5-small.json: corpus/encoder fields are swiltra_de_fr_it/e5-small
- swiltra_de_fr_it/e5-small: silhouette K identical in geometry and performance runs on seeds [0, 1, 2, 3, 4]
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_e5-small.json: seeds field equals per-seed records
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_e5-small.json: no duplicated seed
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_e5-small.json: AUCs saved without flip adjustment
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_e5-small.json: the geometry run of scotus_oyez_lexis/e5-small, seeds [0, 1, 2, 3, 4]
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_e5-small.json: saved geometry means equal the per-seed fold means
- scotus_oyez_lexis/e5-small: geometry run has C=2, nuisance subspace rank 1
- scotus_oyez_lexis/e5-small: geometry uses the same corpus version and split policy
- scotus_oyez_lexis/e5-small: capture recorded at rank 1 and at C-1=1
- results/extensions/raw/five_seeds/alternatives/alternatives_scotus_oyez_lexis_e5-small.json: corpus/encoder fields are scotus_oyez_lexis/e5-small
- results/extensions/raw/five_seeds/alternatives/alternatives_scotus_oyez_lexis_e5-small.json: corpus/encoder fields are scotus_oyez_lexis/e5-small
- results/extensions/raw/five_seeds/alternatives/alternatives_scotus_oyez_lexis_e5-small.json: corpus/encoder fields are scotus_oyez_lexis/e5-small
- results/extensions/raw/five_seeds/alternatives/alternatives_scotus_oyez_lexis_e5-small.json: corpus/encoder fields are scotus_oyez_lexis/e5-small
- scotus_oyez_lexis/e5-small: silhouette K identical in geometry and performance runs on seeds [0, 1, 2, 3, 4]
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_mini.json: seeds field equals per-seed records
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_mini.json: no duplicated seed
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_mini.json: AUCs saved without flip adjustment
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_mini.json: the geometry run of scotus_oyez_lexis/mini, seeds [0, 1, 2, 3, 4]
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_mini.json: saved geometry means equal the per-seed fold means
- scotus_oyez_lexis/mini: geometry run has C=2, nuisance subspace rank 1
- scotus_oyez_lexis/mini: geometry uses the same corpus version and split policy
- scotus_oyez_lexis/mini: capture recorded at rank 1 and at C-1=1
- id/id_scotus_oyez_lexis_mini.json: seeds field equals per-seed records
- id/id_scotus_oyez_lexis_mini.json: no duplicated seed
- id/id_scotus_oyez_lexis_mini.json: AUCs saved without flip adjustment
- id/id_scotus_oyez_lexis_mini.json: saved aggregates equal per-seed values, mean and SD (ddof=1)
- id/id_scotus_oyez_lexis_mini.json: each seed's AUC/purity equals the mean of its saved fold values
- id/id_scotus_oyez_lexis_mini.json: each seed's R@10 equals the mean of its saved per-pair hits
- id/id_scotus_oyez_lexis_mini.json: corpus/encoder fields are scotus_oyez_lexis/mini
- id/id_scotus_oyez_lexis_mini.json: corpus/encoder fields are scotus_oyez_lexis/mini
- id/id_kmeans_scotus_oyez_lexis_mini.json: seeds field equals per-seed records
- id/id_kmeans_scotus_oyez_lexis_mini.json: no duplicated seed
- id/id_kmeans_scotus_oyez_lexis_mini.json: AUCs saved without flip adjustment
- id/id_kmeans_scotus_oyez_lexis_mini.json: saved aggregates equal per-seed values, mean and SD (ddof=1)
- id/id_kmeans_scotus_oyez_lexis_mini.json: each seed's AUC/purity equals the mean of its saved fold values
- id/id_kmeans_scotus_oyez_lexis_mini.json: each seed's R@10 equals the mean of its saved per-pair hits
- id/id_kmeans_scotus_oyez_lexis_mini.json: corpus/encoder fields are scotus_oyez_lexis/mini
- id:scotus_oyez_lexis/mini main~kmeans: config.split identical
- id:scotus_oyez_lexis/mini main~kmeans: config.audit identical
- id:scotus_oyez_lexis/mini main~kmeans: config.selection identical
- id:scotus_oyez_lexis/mini main~kmeans: config.pseudo_groups identical
- id:scotus_oyez_lexis/mini main~kmeans: SAE/eraser recipe identical
- id:scotus_oyez_lexis/mini main~kmeans: corpus statistics identical
- id:scotus_oyez_lexis/mini main~kmeans: per-seed split summaries identical on seeds [0, 1, 2, 3, 4]
- id/id_scotus_oyez_lexis_mini.json: corpus/encoder fields are scotus_oyez_lexis/mini
- id/id_kmeans_scotus_oyez_lexis_mini.json: corpus/encoder fields are scotus_oyez_lexis/mini
- id/id_scotus_oyez_lexis_mini.json: corpus/encoder fields are scotus_oyez_lexis/mini
- scotus_oyez_lexis/mini: silhouette K identical in geometry and performance runs on seeds [0, 1, 2, 3, 4]
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_wiki_e5-small.json: seeds field equals per-seed records
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_wiki_e5-small.json: no duplicated seed
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_wiki_e5-small.json: AUCs saved without flip adjustment
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_wiki_e5-small.json: the geometry run of scotus_oyez_lexis_wiki/e5-small, seeds [0, 1, 2, 3, 4]
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_wiki_e5-small.json: saved geometry means equal the per-seed fold means
- scotus_oyez_lexis_wiki/e5-small: geometry run has C=3, nuisance subspace rank 2
- scotus_oyez_lexis_wiki/e5-small: geometry uses the same corpus version and split policy
- scotus_oyez_lexis_wiki/e5-small: capture recorded at rank 1 and at C-1=2
- results/extensions/raw/five_seeds/alternatives/alternatives_scotus_oyez_lexis_wiki_e5-small.json: corpus/encoder fields are scotus_oyez_lexis_wiki/e5-small
- results/extensions/raw/five_seeds/alternatives/alternatives_scotus_oyez_lexis_wiki_e5-small.json: corpus/encoder fields are scotus_oyez_lexis_wiki/e5-small
- results/extensions/raw/five_seeds/alternatives/alternatives_scotus_oyez_lexis_wiki_e5-small.json: corpus/encoder fields are scotus_oyez_lexis_wiki/e5-small
- results/extensions/raw/five_seeds/alternatives/alternatives_scotus_oyez_lexis_wiki_e5-small.json: corpus/encoder fields are scotus_oyez_lexis_wiki/e5-small
- scotus_oyez_lexis_wiki/e5-small: silhouette K identical in geometry and performance runs on seeds [0, 1, 2, 3, 4]
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_wiki_mini.json: seeds field equals per-seed records
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_wiki_mini.json: no duplicated seed
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_wiki_mini.json: AUCs saved without flip adjustment
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_wiki_mini.json: the geometry run of scotus_oyez_lexis_wiki/mini, seeds [0, 1, 2, 3, 4]
- results/extensions/raw/five_seeds/alternatives/geometry_scotus_oyez_lexis_wiki_mini.json: saved geometry means equal the per-seed fold means
- scotus_oyez_lexis_wiki/mini: geometry run has C=3, nuisance subspace rank 2
- scotus_oyez_lexis_wiki/mini: geometry uses the same corpus version and split policy
- scotus_oyez_lexis_wiki/mini: capture recorded at rank 1 and at C-1=2
- id/id_scotus_oyez_lexis_wiki_mini.json: corpus/encoder fields are scotus_oyez_lexis_wiki/mini
- id/id_scotus_oyez_lexis_wiki_mini.json: corpus/encoder fields are scotus_oyez_lexis_wiki/mini
- id/id_kmeans_scotus_oyez_lexis_wiki_mini.json: corpus/encoder fields are scotus_oyez_lexis_wiki/mini
- id/id_scotus_oyez_lexis_wiki_mini.json: corpus/encoder fields are scotus_oyez_lexis_wiki/mini
- scotus_oyez_lexis_wiki/mini: silhouette K identical in geometry and performance runs on seeds [0, 1, 2, 3, 4]
- standalone A: smallest text 7.5 pt >= 7.5 pt
- standalone A: no text outside the canvas
- standalone A: no two value annotations overlap
- standalone A: no annotation covers its marker
- standalone A: header ends 0.045 in above the plot
- standalone A: setting labels fully on the canvas
- standalone B: smallest text 7.5 pt >= 7.5 pt
- standalone B: no text outside the canvas
- standalone B: no two value annotations overlap
- standalone B: no annotation covers its marker
- standalone B: header ends 0.170 in above the plot
- standalone B: setting labels fully on the canvas
- standalone C: smallest text 7.5 pt >= 7.5 pt
- standalone C: no text outside the canvas
- standalone C: no two value annotations overlap
- standalone C: no annotation covers its marker
- standalone C: the method key overlaps no title or subtitle
- standalone C: header ends 0.200 in above the plot
- standalone C: setting labels fully on the canvas
- composite part A: smallest text 7.5 pt >= 7.5 pt
- composite part A: no text outside the canvas
- composite part A: no two value annotations overlap
- composite part A: no annotation covers its marker
- composite part A: the method key overlaps no title or subtitle
- composite part A: header ends 0.045 in above the plot
- composite part A: setting labels fully on the canvas
- composite part B: smallest text 7.5 pt >= 7.5 pt
- composite part B: no text outside the canvas
- composite part B: no two value annotations overlap
- composite part B: no annotation covers its marker
- composite part B: header ends 0.170 in above the plot
- composite part C: smallest text 7.5 pt >= 7.5 pt
- composite part C: no text outside the canvas
- composite part C: no two value annotations overlap
- composite part C: no annotation covers its marker
- composite part C: header ends 0.200 in above the plot
- composite width 6.30 in = manuscript text width 6.30 in
- composite built from the three finished panel PDFs
- id/id_swiltra_e5-small.json: corpus/encoder fields are swiltra/e5-small
- id/id_swiltra_e5-small.json: corpus/encoder fields are swiltra/e5-small
- id/id_kmeans_swiltra_e5-small.json: corpus/encoder fields are swiltra/e5-small
- id/id_swiltra_e5-small.json: corpus/encoder fields are swiltra/e5-small
- id/id_swiltra_de_fr_it_e5-small.json: corpus/encoder fields are swiltra_de_fr_it/e5-small
- id/id_swiltra_de_fr_it_e5-small.json: corpus/encoder fields are swiltra_de_fr_it/e5-small
- id/id_kmeans_swiltra_de_fr_it_e5-small.json: corpus/encoder fields are swiltra_de_fr_it/e5-small
- id/id_swiltra_de_fr_it_e5-small.json: corpus/encoder fields are swiltra_de_fr_it/e5-small
- results/extensions/raw/five_seeds/alternatives/alternatives_scotus_oyez_lexis_e5-small.json: corpus/encoder fields are scotus_oyez_lexis/e5-small
- results/extensions/raw/five_seeds/alternatives/alternatives_scotus_oyez_lexis_e5-small.json: corpus/encoder fields are scotus_oyez_lexis/e5-small
- results/extensions/raw/five_seeds/alternatives/alternatives_scotus_oyez_lexis_e5-small.json: corpus/encoder fields are scotus_oyez_lexis/e5-small
- results/extensions/raw/five_seeds/alternatives/alternatives_scotus_oyez_lexis_e5-small.json: corpus/encoder fields are scotus_oyez_lexis/e5-small
- id/id_scotus_oyez_lexis_mini.json: corpus/encoder fields are scotus_oyez_lexis/mini
- id/id_scotus_oyez_lexis_mini.json: corpus/encoder fields are scotus_oyez_lexis/mini
- id/id_kmeans_scotus_oyez_lexis_mini.json: corpus/encoder fields are scotus_oyez_lexis/mini
- id/id_scotus_oyez_lexis_mini.json: corpus/encoder fields are scotus_oyez_lexis/mini
- results/extensions/raw/five_seeds/alternatives/alternatives_scotus_oyez_lexis_wiki_e5-small.json: corpus/encoder fields are scotus_oyez_lexis_wiki/e5-small
- results/extensions/raw/five_seeds/alternatives/alternatives_scotus_oyez_lexis_wiki_e5-small.json: corpus/encoder fields are scotus_oyez_lexis_wiki/e5-small
- results/extensions/raw/five_seeds/alternatives/alternatives_scotus_oyez_lexis_wiki_e5-small.json: corpus/encoder fields are scotus_oyez_lexis_wiki/e5-small
- results/extensions/raw/five_seeds/alternatives/alternatives_scotus_oyez_lexis_wiki_e5-small.json: corpus/encoder fields are scotus_oyez_lexis_wiki/e5-small
- id/id_scotus_oyez_lexis_wiki_mini.json: corpus/encoder fields are scotus_oyez_lexis_wiki/mini
- id/id_scotus_oyez_lexis_wiki_mini.json: corpus/encoder fields are scotus_oyez_lexis_wiki/mini
- id/id_kmeans_scotus_oyez_lexis_wiki_mini.json: corpus/encoder fields are scotus_oyez_lexis_wiki/mini
- id/id_scotus_oyez_lexis_wiki_mini.json: corpus/encoder fields are scotus_oyez_lexis_wiki/mini

## Issues

None.
