<!-- Generated from table_2_primary_id.tex by `python paper-artifacts/build.py table_2_primary_id`; the LaTeX file is the reference. -->

|  | E5-small: SemEval C=4 | E5-small: SwiLTra C=3 | E5-small: Fedlex C=3 | E5-small: Bills–Tweets C=2 | E5-small: SCOTUS C=3 | E5-small: CAP C=3 | MiniLM SCOTUS C=3 |
|---|---|---|---|---|---|---|---|
| *Panel A: linear-probe nuisance ROC-AUC ↓ (one-vs-rest macro average for C>2)* |  |  |  |  |  |  |  |
| Raw | 1.000 ± .000 | .999 ± .000 | 1.000 ± .000 | 1.000 ± .000 | .999 ± .000 | 1.000 ± .000 | .986 ± .000 |
| PC-1 | .927 ± .006 | .906 ± .004 | .911 ± .006 | .678 ± .014 | .882 ± .004 | .892 ± .008 | .986 ± .000 |
| k-means LEACE | .797 ± .050 | .577 ± .011 | .644 ± .016 | .612 ± .011 | .699 ± .085 | .879 ± .012 | .950 ± .047 |
| Metadata-LEACE | .649 ± .022 | .577 ± .011 | .644 ± .016 | .609 ± .011 | .615 ± .004 | .658 ± .024 | .611 ± .005 |
| SAE-LEACE (N=32) | .699 ± .017 | .615 ± .010 | .672 ± .016 | .653 ± .012 | .727 ± .009 | .713 ± .022 | .860 ± .017 |
| *Panel B: semantic utility ↑ (R@10; CAP: topic purity)* |  |  |  |  |  |  |  |
| Raw | .597 ± .000 | .492 ± .004 | .784 ± .009 | — | .626 ± .005 | .242 ± .009 | .697 ± .006 |
| PC-1 | .811 ± .003 | .897 ± .003 | .950 ± .004 | — | .721 ± .008 | .318 ± .019 | .696 ± .006 |
| k-means LEACE | .814 ± .020 | .969 ± .003 | .981 ± .002 | — | .787 ± .034 | .320 ± .009 | .715 ± .023 |
| Metadata-LEACE | .877 ± .001 | .969 ± .003 | .981 ± .002 | — | .800 ± .008 | .352 ± .012 | .749 ± .006 |
| SAE-LEACE (N=32) | .811 ± .007 | .916 ± .003 | .938 ± .005 | — | .770 ± .005 | .263 ± .033 | .712 ± .010 |

Primary in-distribution leakage and semantic utility. Cells are the mean ± sample SD (ddof=1) across fitting seeds. Each seed's score is computed within that seed; SemEval's five outer folds are combined within each seed (fold AUCs averaged, held-out retrieval hits pooled), and nothing is pooled across seeds. Seeds: SemEval 5 seeds (0–4); SwiLTra 5 seeds (0–4); Fedlex 5 seeds (0–4); Bills–Tweets 5 seeds (0–4); SCOTUS 5 seeds (0–4); CAP 5 seeds (0–4); SCOTUS MiniLM 5 seeds (0–4). SCOTUS with E5-small uses the coverage experiment, and SemEval and CAP their primary runs, each extended from seeds 0–2 by seeds 3–4. Fedlex is a seeded sample of 2,784 German/French/Italian article triples of Swiss domestic federal law (SwiLTra's size), with outer splits by whole law. MiniLM results are shown only for three-source SCOTUS, not for the other corpora. AUCs are unflipped. k-means LEACE uses the training-silhouette K; only Metadata-LEACE uses nuisance labels. Bills–Tweets has no semantic-utility measure (—). Equal rounded values are not evidence of equivalence.
