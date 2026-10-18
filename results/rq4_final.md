Nodes: 71674 from 1350 answers; emotional nodes: 23273; hallucinated nodes: 5457.

## RQ4 Model 1. Reliability (logit P) and emotional content, linear mixed model

| Term | Coefficient (per SD) | 95% CI | p |
| --- | --- | --- | --- |
| abs_valence | -0.028 | [-0.039, -0.016] | 3.08e-06 |
| n_emotions | -0.018 | [-0.029, -0.006] | 0.00294 |
| frame_valence_spread | +0.513 | [+0.503, +0.523] | 0 |
| zipf | -0.212 | [-0.222, -0.202] | 0 |
| length | -0.150 | [-0.160, -0.140] | 3.12e-178 |
| log_degree | +0.211 | [+0.201, +0.221] | 0 |

## RQ4 Model 2. Does emotional content change how well confidence detects hallucination?

Logistic regression, standard errors clustered by answer. Odds ratios per SD of (1 − P).

| Term | Odds ratio | 95% CI | p |
| --- | --- | --- | --- |
| risk_z | 1.826 | [1.725, 1.932] | 1.2e-96 |
| emotional | 0.843 | [0.783, 0.907] | 4.79e-06 |
| risk_z:emotional | 0.895 | [0.830, 0.964] | 0.00329 |

| Nodes | AUROC of 1 − P | Hallucinated / total |
| --- | --- | --- |
| emotional | 0.623 | 1565 / 23273 |
| neutral | 0.654 | 3892 / 48401 |
