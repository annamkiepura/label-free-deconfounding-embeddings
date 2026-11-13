<!-- Generated from table_SAE_construction_block.tex by `python paper-artifacts/build.py table_SAE_construction_block`; the LaTeX file is the reference. -->

| TopK SAE | Eligible | Displ. | r\* | Linear AUC↓: MI-selected | Linear AUC↓: Random − MI | Linear AUC↓: Matched PCA − MI |
|---|---|---|---|---|---|---|
| **SwiLTra (DE/FR/IT)** |  |  |  |  |  |  |
| 8d latents, k=32 (the paper's) | 76–93 | .36 | 22 | .611 ± .010 | +.015 [+.008, +.022] | −.016 [−.025, −.008] |
| 16d latents, k=64 | 303–351 | .35 | 22 | .604 ± .006 | +.045 [+.020, +.070] | −.009 [−.019, +.002] |
| 8d latents, k=64 | 1470–1493 | .35 | 21–22 | .611 ± .011 | +.063 [+.037, +.088] | −.015 [−.019, −.012] |
| 4d latents, k=16 | 42–56 | .36 | 23–24 | .604 ± .011 | +.005 [.000, +.011] | −.009 [−.025, +.008] |
| **SCOTUS (3 sources)** |  |  |  |  |  |  |
| 8d latents, k=32 (the paper's) | 65–80 | .30 | 22 | .732 ± .008 | +.010 [−.012, +.032] | −.055 [−.079, −.031] |
| 16d latents, k=64 | 287–318 | .30 | 22 | .715 ± .003 | +.034 [+.007, +.060] | −.038 [−.051, −.024] |
| 8d latents, k=64 | 1479–1549 | .29 | 21 | .721 ± .016 | +.059 [+.027, +.091] | −.044 [−.091, +.003] |
| 4d latents, k=16 | 22 | *not comparable at N=32: too few eligible latents (seed 0)* |  |  |  |  |

**SAE construction.** Table `tab:mechanism-controls`'s core comparisons with other TopK dictionaries (latents as a multiple of the embedding dimension d, and k active latents; E5-small, seeds 0–2, N=32). Eligible: latents per fit with a training firing rate in [0.02, 0.98], the set MI scores. Displ.: normalized squared training displacement of MI-selected SAE-LEACE, and r\* the rank of the PCA matched to it. Linear AUC of MI-selected SAE-LEACE (mean ± sample SD), and the per-seed paired differences to it of random SAE directions (five draws per seed) and of the matched PCA, with 95% t-intervals over seeds. A construction with fewer than 32 eligible latents in some fit cannot be compared at N=32.
