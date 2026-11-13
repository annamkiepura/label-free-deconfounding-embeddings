<!-- Generated from table3_mechanism_controls.tex by `python paper-artifacts/build.py table3_mechanism_controls`; the LaTeX file is the reference. -->

|  | Condition | SwiLTra (DE/FR/IT): Lin. AUC↓ | SwiLTra (DE/FR/IT): MLP AUC↓ | SwiLTra (DE/FR/IT): R@10↑ | SCOTUS (3 sources): Lin. AUC↓ | SCOTUS (3 sources): MLP AUC↓ | SCOTUS (3 sources): R@10↑ |
|---|---|---|---|---|---|---|---|
| **A. Direction selection (Directions-LEACE; N=32 except A4)** |  |  |  |  |  |  |  |
| A1 | MI-selected SAE directions | .616 ± .010 | .838 ± .007 | .914 ± .004 | .727 ± .013 | .797 ± .014 | .774 ± .005 |
| A2 | Random SAE directions | .630 ± .010 | .842 ± .007 | .909 ± .003 | .742 ± .002 | .814 ± .005 | .772 ± .005 |
| A3 | Random non-SAE directions | .819 ± .015 | .913 ± .007 | .923 ± .003 | .887 ± .010 | .916 ± .006 | .771 ± .006 |
| A4 | Non-SAE directions, displ.-matched | .781 ± .012 | .893 ± .007 | .911 ± .003 | .861 ± .009 | .894 ± .006 | .767 ± .004 |
| A1† | MI-selected, fits of A5–A9 | .617 ± .012 | .842 ± .009 | .918 ± .004 | .737 ± .008 | .811 ± .006 | .772 ± .003 |
| A5† | Entropy-selected SAE directions | .618 ± .013 | .832 ± .015 | .912 ± .006 | .752 ± .006 | .819 ± .008 | .768 ± .003 |
| A6† | Dip-selected SAE directions | .617 ± .011 | .840 ± .010 | .917 ± .005 | .756 ± .011 | .826 ± .012 | .769 ± .004 |
| A7† | Co-firing-selected SAE directions | .623 ± .017 | .835 ± .006 | .914 ± .003 | .751 ± .014 | .815 ± .007 | .769 ± .004 |
| A8† | Co-firing-selected, exclusive pairs | .620 ± .014 | .840 ± .006 | .917 ± .005 | .745 ± .009 | .811 ± .008 | .769 ± .004 |
| A9† | Oracle (MI with the true classes) | .616 ± .011 | .841 ± .009 | .918 ± .004 | .732 ± .003 | .806 ± .005 | .773 ± .004 |
| **B. Operator (same MI-selected latent set S as row A1)** |  |  |  |  |  |  |  |
| B1 | SAE activations + Activation-LEACE | .812 ± .018 | .901 ± .009 | .933 ± .004 | .862 ± .005 | .896 ± .006 | .768 ± .003 |
| B2 | SAE directions + orthogonal removal | .998 ± .001 | .999 ± .000 | .894 ± .017 | .995 ± .001 | .996 ± .001 | .746 ± .008 |
| B3 | Orthogonal removal, mean kept | .998 ± .001 | .999 ± .000 | .894 ± .017 | .995 ± .001 | .996 ± .001 | .746 ± .008 |
| B4 | Orth. removal of span(Σ D_Sᵀ) | .616 ± .010 | .838 ± .006 | .914 ± .004 | .727 ± .013 | .796 ± .011 | .774 ± .005 |
| **C. Erasure extent** |  |  |  |  |  |  |  |
| C1 | PCA, displacement-matched | .600 ± .013 | .817 ± .010 | .882 ± .005 | .677 ± .003 | .756 ± .006 | .774 ± .005 |
| **References** |  |  |  |  |  |  |  |
|  | Raw | .999 ± .000 | .999 ± .000 | .492 ± .004 | .999 ± .000 | .999 ± .000 | .626 ± .005 |
|  | Metadata-LEACE (supervised) | .577 ± .011 | .923 ± .003 | .969 ± .003 | .615 ± .004 | .880 ± .008 | .800 ± .008 |

**Candidate directions, intervention operators, and erasure extent.** E5-small results on three-class SwiLTra and SCOTUS (mean ± sample SD, seeds 0–4). AUCs are held-out macro one-vs-rest probe scores; R@10 measures paired retrieval. Random SAE selection (row A2) draws uniformly from the same latents that MI scores (training firing rate in [0.02, 0.98]; Appendix `sec:appendix_A3`). Row A4 uses the smallest number of random non-SAE directions whose Directions-LEACE map reaches or exceeds row A1's normalized squared training displacement (N: SwiLTra 40–43; SCOTUS 40–41). The random rows average five draws per seed. Rows A5–A8 rank the same eligible latents without k-means: by the entropy of a latent's firing indicator, by Hartigan's dip statistic of the training coordinates along its decoder direction, and by its mean firing mutual information with the other eligible latents (all pairs; negatively correlated pairs only). The oracle (row A9) ranks by mutual information with the true classes: a post hoc reference, not a label-free method. Row B3 is row B2 with the training mean restored, and row B4 removes orthogonally the subspace that Directions-LEACE displaces, span(Σ D_Sᵀ), also with the mean restored. Displacement-matched PCA (row C1) removes the fewest principal directions whose training displacement reaches that of row A1 (Appendix `sec:appendix_A4`; ranks: SwiLTra 22–23; SCOTUS 22); this matches displacement, not rank or total-variance reduction, and Table `tab:erasure-extent` repeats the comparison on six settings. The SAEs of this table were fitted separately from those of Table `tab:primary-id`, and those of the rows marked † separately again (same recipe, seeds and splits throughout); row A1† is row A1 on the latter fits, the baseline for rows A5–A9. The reference rows are the runs of Table `tab:primary-id`.
