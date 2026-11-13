<!-- Generated from tableB1_extended_id_block.tex by `python paper-artifacts/build.py tableB1_extended_id_block`; the LaTeX file is the reference. -->

| Setting | Metric | Raw | PC-1 | k-means LEACE | Metadata-LEACE | SAE-LEACE (N=32) |
|---|---|---|---|---|---|---|
| **SwiLTra** confounder: language; 5 seeds (0–4), primary cohort |  |  |  |  |  |  |
| **DE–FR** | Linear AUC ↓ | .999 ± .000 | .632 ± .009 | .600 ± .010 | .600 ± .011 | .625 ± .006 |
| C=2 | MLP AUC ↓ | .999 ± .000 | .964 ± .004 | .963 ± .006 | .961 ± .006 | .871 ± .010 |
|  | R@1 ↑ | .106 ± .003 | .656 ± .011 | .658 ± .010 | .658 ± .010 | .559 ± .012 |
|  | R@10 ↑ | .324 ± .006 | .963 ± .004 | .964 ± .004 | .964 ± .004 | .884 ± .009 |
| **DE–IT** | Linear AUC ↓ | 1.000 ± .000 | .613 ± .018 | .589 ± .017 | .589 ± .016 | .604 ± .016 |
| C=2 | MLP AUC ↓ | 1.000 ± .000 | .961 ± .002 | .962 ± .003 | .961 ± .002 | .870 ± .005 |
|  | R@1 ↑ | .054 ± .001 | .647 ± .010 | .649 ± .011 | .649 ± .011 | .536 ± .002 |
|  | R@10 ↑ | .190 ± .005 | .957 ± .004 | .957 ± .004 | .957 ± .005 | .880 ± .006 |
| **FR–IT** | Linear AUC ↓ | .999 ± .001 | .611 ± .016 | .577 ± .015 | .576 ± .015 | .606 ± .016 |
| C=2 | MLP AUC ↓ | .999 ± .001 | .883 ± .012 | .882 ± .011 | .883 ± .011 | .764 ± .017 |
|  | R@1 ↑ | .722 ± .010 | .978 ± .002 | .978 ± .002 | .978 ± .002 | .965 ± .006 |
|  | R@10 ↑ | .966 ± .003 | 1.000 ± .000 | 1.000 ± .000 | 1.000 ± .000 | .999 ± .000 |
| **DE–FR–IT** | Linear AUC ↓ | .999 ± .000 | .906 ± .004 | .577 ± .011 | .577 ± .011 | .615 ± .010 |
| C=3 | MLP AUC ↓ | .999 ± .000 | .997 ± .001 | .924 ± .003 | .923 ± .003 | .838 ± .008 |
|  | R@1 ↑ | .294 ± .004 | .585 ± .006 | .759 ± .007 | .759 ± .007 | .688 ± .004 |
|  | R@10 ↑ | .492 ± .004 | .897 ± .003 | .969 ± .003 | .969 ± .003 | .916 ± .003 |
| **Fedlex** confounder: language; 5 seeds (0–4), split by whole law |  |  |  |  |  |  |
| **DE–FR–IT** | Linear AUC ↓ | 1.000 ± .000 | .911 ± .006 | .644 ± .016 | .644 ± .016 | .672 ± .016 |
| C=3 | MLP AUC ↓ | 1.000 ± .000 | .990 ± .001 | .958 ± .003 | .958 ± .002 | .851 ± .007 |
|  | R@1 ↑ | .456 ± .009 | .690 ± .009 | .804 ± .008 | .804 ± .008 | .731 ± .010 |
|  | R@10 ↑ | .784 ± .009 | .950 ± .004 | .981 ± .002 | .981 ± .002 | .938 ± .005 |
| **SCOTUS** confounder: source; 5 seeds (0–4), coverage cohort |  |  |  |  |  |  |
| **Oyez–Lexis** | Linear AUC ↓ | 1.000 ± .000 | .714 ± .015 | .672 ± .016 | .622 ± .017 | .701 ± .022 |
| C=2 | MLP AUC ↓ | .998 ± .000 | .892 ± .010 | .890 ± .010 | .874 ± .009 | .748 ± .012 |
|  | R@1 ↑ | .539 ± .013 | .797 ± .012 | .800 ± .010 | .801 ± .010 | .792 ± .013 |
|  | R@10 ↑ | .811 ± .014 | .942 ± .004 | .943 ± .003 | .943 ± .004 | .931 ± .002 |
| **Oyez–Wikipedia** | Linear AUC ↓ | 1.000 ± .000 | .668 ± .015 | .633 ± .005 | .633 ± .010 | .658 ± .022 |
| C=2 | MLP AUC ↓ | .999 ± .000 | .855 ± .017 | .855 ± .010 | .853 ± .009 | .709 ± .020 |
|  | R@1 ↑ | .380 ± .011 | .691 ± .005 | .694 ± .006 | .694 ± .005 | .716 ± .010 |
|  | R@10 ↑ | .652 ± .005 | .915 ± .007 | .918 ± .009 | .918 ± .009 | .917 ± .007 |
| **Lexis–Wikipedia** | Linear AUC ↓ | .999 ± .000 | .756 ± .011 | .706 ± .021 | .648 ± .015 | .722 ± .007 |
| C=2 | MLP AUC ↓ | .997 ± .001 | .864 ± .018 | .847 ± .013 | .833 ± .008 | .761 ± .014 |
|  | R@1 ↑ | .230 ± .010 | .379 ± .007 | .380 ± .008 | .370 ± .011 | .352 ± .008 |
|  | R@10 ↑ | .505 ± .011 | .680 ± .009 | .681 ± .009 | .680 ± .009 | .635 ± .015 |
| **All three sources** | Linear AUC ↓ | .999 ± .000 | .882 ± .004 | .699 ± .085 | .615 ± .004 | .727 ± .009 |
| C=3 | MLP AUC ↓ | .999 ± .000 | .979 ± .000 | .905 ± .037 | .880 ± .008 | .803 ± .004 |
|  | R@1 ↑ | .350 ± .006 | .506 ± .007 | .560 ± .033 | .572 ± .009 | .563 ± .007 |
|  | R@10 ↑ | .626 ± .005 | .721 ± .008 | .787 ± .034 | .800 ± .008 | .770 ± .005 |
| **Supplementary: other primary corpora** 5 seeds (0–4), primary cohort |  |  |  |  |  |  |
| **SemEval** C=4 | Graded sim. ρ ↑ | .274 ± .000 | .340 ± .003 | .342 ± .022 | .388 ± .001 | .401 ± .003 |
| **CAP** C=3 | Topic ARI ↑ | .061 ± .004 | .108 ± .009 | .112 ± .010 | .146 ± .010 | .075 ± .021 |

Extended in-distribution results for the eight E5-small composition settings and Fedlex. Cells are the mean ± sample SD (ddof=1) across fitting seeds; each seed is scored on its own held-out split and nothing is pooled across seeds. Cohorts: SwiLTra, the primary cohort, 5 seeds (0–4), in all four settings; Fedlex, as in Table `tab:primary-id` (outer splits by whole law), 5 seeds (0–4); SCOTUS, its five-seed runs, 5 seeds (0–4). Within a setting all methods and metrics come from the same saved run and splits. Metrics are defined in Appendix `app:extended-id-metrics`. k-means LEACE uses the training-silhouette K; only Metadata-LEACE uses confounder labels.
