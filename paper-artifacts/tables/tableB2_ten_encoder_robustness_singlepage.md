<!-- Generated from tableB2_ten_encoder_robustness_singlepage.tex by `python paper-artifacts/build.py tableB2_ten_encoder_robustness_singlepage`; the LaTeX file is the reference. -->

**Panel A.** Linear AUC relative to Raw (PCA-1/k-means LEACE/SAE-LEACE; lower is better)

| Encoder | SemEval | SwiLTra DE–FR | SCOTUS O–L |
|---|---|---|---|
| MiniLM | −.028/−.323/−.312 | −.346/−.379/−.353 | −.001/.000/−.127 |
| GIST-small | −.008/−.235/−.283 | −.333/−.391/−.370 | −.001/.000/−.186 |
| GIST-base | −.007/−.123/−.279 | −.332/−.363/−.337 | .000/.000/−.204 |
| GIST-large | −.007/−.182/−.281 | −.322/−.351/−.327 | .000/.000/−.154 |
| E5-small | −.070/−.204/−.303 | −.371/−.401/−.377 | −.276/−.318/−.283 |
| E5-base | −.040/−.228/−.259 | −.377/−.389/−.372 | −.244/−.310/−.258 |
| E5-large | −.085/−.219/−.275 | −.348/−.377/−.345 | −.174/−.200/−.234 |
| MPNet | −.003/−.152/−.228 | −.303/−.340/−.319 | .000/−.002/−.143 |
| Nomic-v2 | −.005/−.005/−.086 | −.312/−.383/−.280 | .000/.000/−.137 |
| MXB | −.015/−.263/−.288 | −.321/−.345/−.322 | −.001/.000/−.183 |
| Mean | −.027/−.194/−.260 | −.336/−.372/−.340 | −.070/−.083/−.191 |
| Range (max−min) | .082/.318/.226 | .075/.061/.096 | .276/.318/.157 |

| Encoder | CAP | Fedlex |
|---|---|---|
| MiniLM | −.077/−.124/−.275 | −.088/−.262/−.269 |
| GIST-small | −.074/−.117/−.268 | −.080/−.182/−.282 |
| GIST-base | −.073/−.107/−.248 | −.080/−.298/−.273 |
| GIST-large | −.071/−.106/−.246 | −.071/−.278/−.256 |
| E5-small | −.113/−.129/−.296 | −.091/−.361/−.329 |
| E5-base | −.082/−.114/−.259 | −.082/−.111/−.314 |
| E5-large | −.050/−.077/−.235 | −.069/−.105/−.308 |
| MPNet | −.076/−.104/−.242 | −.081/−.231/−.239 |
| Nomic-v2 | −.076/−.109/−.244 | .000/.000/−.181 |
| MXB | −.077/−.106/−.253 | −.069/−.285/−.260 |
| Mean | −.077/−.109/−.257 | −.071/−.211/−.271 |
| Range (max−min) | .063/.052/.061 | .091/.361/.149 |

**Panel B.** Semantic utility relative to Raw (PCA-1/k-means LEACE/SAE-LEACE; R@10, for CAP topic purity; higher is better)

| Encoder | SemEval | SwiLTra DE–FR | SCOTUS O–L |
|---|---|---|---|
| MiniLM | +.067/+.243/+.182 | +.066/+.065/+.053 | −.010/−.010/−.015 |
| GIST-small | +.036/+.100/+.008 | +.169/+.173/+.158 | +.002/+.001/+.005 |
| GIST-base | +.041/+.093/+.035 | +.280/+.278/+.251 | −.004/−.005/+.007 |
| GIST-large | +.033/+.088/+.063 | +.235/+.237/+.184 | +.001/.000/+.005 |
| E5-small | +.212/+.217/+.214 | +.643/+.645/+.567 | +.134/+.135/+.121 |
| E5-base | +.039/+.082/+.050 | +.381/+.381/+.361 | +.158/+.158/+.148 |
| E5-large | +.141/+.166/+.144 | +.100/+.100/+.098 | +.119/+.121/+.099 |
| MPNet | +.015/+.077/+.064 | +.097/+.099/+.064 | +.002/+.003/+.007 |
| Nomic-v2 | −.008/−.005/−.017 | +.078/+.079/+.062 | −.005/−.002/+.009 |
| MXB | +.076/+.197/+.141 | +.295/+.294/+.245 | +.006/+.001/+.017 |
| Mean | +.065/+.126/+.088 | +.234/+.235/+.204 | +.040/+.040/+.040 |

| Encoder | CAP | Fedlex |
|---|---|---|
| MiniLM | +.084/+.119/−.027 | +.100/+.155/+.117 |
| GIST-small | +.137/+.146/−.016 | +.081/+.125/+.121 |
| GIST-base | +.134/+.153/−.002 | +.141/+.303/+.214 |
| GIST-large | +.156/+.157/−.018 | +.162/+.306/+.216 |
| E5-small | +.085/+.085/+.032 | +.169/+.201/+.155 |
| E5-base | +.096/+.096/+.034 | +.040/+.041/+.033 |
| E5-large | +.055/+.077/+.005 | +.016/+.017/+.018 |
| MPNet | +.114/+.117/−.034 | +.106/+.166/+.122 |
| Nomic-v2 | +.130/+.126/+.022 | −.004/−.004/−.002 |
| MXB | +.114/+.149/+.034 | +.222/+.397/+.295 |
| Mean | +.110/+.123/+.003 | +.103/+.171/+.129 |

\newpage **Panel C.** k-means ARI/PCA capture (post hoc)

| Encoder | SemEval | SwiLTra DE–FR | SCOTUS O–L | CAP | Fedlex |
|---|---|---|---|---|---|
| MiniLM | .55/.78 | .99/.99 | .00/.05 | .59/.89 | .84/.91 |
| GIST-small | .37/.71 | .99/.99 | .00/.09 | .61/.92 | .71/.91 |
| GIST-base | .33/.72 | 1.00/1.00 | .01/.09 | .65/.95 | 1.00/.98 |
| GIST-large | .34/.68 | .99/1.00 | .01/.09 | .62/.96 | 1.00/.99 |
| E5-small | .77/.93 | 1.00/1.00 | .84/.98 | .68/.97 | 1.00/.97 |
| E5-base | .76/.79 | 1.00/1.00 | .88/.98 | .68/.93 | .57/.54 |
| E5-large | .78/.80 | 1.00/.99 | .64/.94 | .60/.94 | .55/.49 |
| MPNet | .30/.75 | .98/.99 | .01/.04 | .62/.91 | .83/.79 |
| Nomic-v2 | .01/.20 | .98/.97 | .00/.03 | .68/.94 | .00/.01 |
| MXB | .52/.70 | .99/1.00 | .01/.15 | .63/.96 | 1.00/.99 |

**Robustness across embedding models.** All erasers are refit in each model's embedding space (not frozen transfer). Each triple reports the change in linear AUC from Raw for PCA-1/k-means LEACE (silhouette-selected K)/SAE-LEACE (N=32); 0 means no change from the raw embeddings. Values average matched differences over seeds 0–2; means and ranges summarize available models. SwiLTra and SCOTUS are shown in the version, among those run with all ten models, where the label-free erasers lower linear AUC most: DE–FR and Oyez–Lexis (mean over models and erasers −.350 and −.115; DE–FR–IT and Oyez–Lexis–Wikipedia: −.220 and −.086).
