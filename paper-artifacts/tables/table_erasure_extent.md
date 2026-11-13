<!-- Generated from table_erasure_extent.tex by `python paper-artifacts/build.py table_erasure_extent`; the LaTeX file is the reference. -->

| Setting | Displ. | r\* | Linear AUC↓: SAE | Linear AUC↓: PCA | Linear AUC↓: PCA − SAE | Utility↑: SAE | Utility↑: PCA | Utility↑: PCA − SAE |
|---|---|---|---|---|---|---|---|---|
| SwiLTra (DE/FR/IT) | .36 | 22–23 | .616 | .600 | −.016 [−.021, −.011] | .914 | .882 | −.032 [−.037, −.026] |
| SCOTUS (3 sources) | .30 | 22 | .727 | .677 | −.049 [−.065, −.033] | .774 | .774 | +.001 [−.003, +.005] |
| SemEval | .42 | 18–21 | .702 | .684 | −.018 [−.035, .000] | .807 | .814 | +.007 [−.001, +.015] |
| CAP | .33 | 20–21 | .713 | .687 | −.026 [−.035, −.017] | .284 | .215 | −.069 [−.135, −.003] |
| Fedlex (DE/FR/IT) | .36 | 21 | .674 | .665 | −.009 [−.015, −.002] | .939 | .926 | −.013 [−.019, −.008] |
| SCOTUS (3 sources), MiniLM | .36 | 19–21 | .840 | .789 | −.051 [−.078, −.023] | .713 | .735 | +.022 [+.012, +.031] |

**Erasure extent: SAE-LEACE against PCA matched in displacement.** Per setting (E5-small unless marked; seeds 0–4), SAE-LEACE at N=32 and PCA removal at the smallest rank r\* whose normalized squared training displacement reaches SAE-LEACE's (Displ.: that displacement, mean over fits). PCA − SAE is the per-seed paired difference with its 95% t-interval over seeds. Utility is R@10, and topic purity for CAP. Both maps of a seed are fitted on the same split and audited on the same held-out documents. The first two rows are rows A1 and C1 of Table `tab:mechanism-controls`. As there, the SAEs are fitted separately from those of Table `tab:primary-id` (same recipe, seeds and splits), so the SAE column differs from that table's SAE-LEACE by .021 on SCOTUS (3 sources) with MiniLM and by at most .003 elsewhere.
