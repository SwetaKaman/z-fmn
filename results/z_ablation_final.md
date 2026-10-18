## A. Fusion ablation: AUROC pooled over datasets (mean of per-dataset AUROC)

| Combiner | Edge AUROC | Node AUROC | Edge: Z-FMN − this | Node: Z-FMN − this |
| --- | --- | --- | --- | --- |
| Z-FMN fusion (base opinion) | 0.793 | 0.762 |  |  |
| Z-FMN fusion (context opinion) | 0.741 | 0.748 | +0.052 [+0.043, +0.061] * | +0.015 [+0.008, +0.022] * |
| Z-FMN fusion (base), reliability removed | 0.778 | 0.750 | +0.015 [+0.011, +0.020] * | +0.012 [+0.010, +0.015] * |
| Z-FMN fusion (context), reliability removed | 0.759 | 0.753 | +0.034 [+0.027, +0.042] * | +0.009 [+0.002, +0.016] * |
| Mean of the three risks | 0.787 | 0.772 | +0.005 [+0.002, +0.009] * | -0.010 [-0.013, -0.006] * |
| Max of the three risks | 0.685 | 0.704 | +0.108 [+0.092, +0.123] * | +0.058 [+0.050, +0.066] * |
| SelfCheckGPT-NLI alone | 0.725 | 0.729 | +0.068 [+0.053, +0.083] * | +0.033 [+0.018, +0.047] * |
| NLI not-entailed alone | 0.767 | 0.753 | +0.026 [+0.013, +0.040] * | +0.009 [-0.006, +0.024] |
| Z-FMN model opinion alone | 0.664 | 0.643 | +0.129 [+0.112, +0.145] * | +0.120 [+0.105, +0.135] * |
| Z-FMN context opinion alone | 0.583 | 0.606 | +0.210 [+0.189, +0.231] * | +0.156 [+0.142, +0.171] * |
| Logistic stacking (supervised, CV) | 0.801 | 0.786 | -0.009 [-0.017, -0.001] * | -0.024 [-0.034, -0.016] * |

* = 95% CI of the paired difference (bootstrap over answers, stratified by dataset) excludes 0.

## A2. Per dataset, edges (AUROC)

| Dataset | Z-FMN fusion (base opinion) | Z-FMN fusion (base), reliability removed | Mean of the three risks | Logistic stacking (supervised, CV) | SelfCheckGPT-NLI alone | Wrong / total |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-2-7B / Data2txt | 0.756 | 0.757 | 0.754 | 0.759 | 0.743 | 2299 / 26751 |
| Llama-2-7B / QA | 0.775 | 0.754 | 0.767 | 0.802 | 0.677 | 2076 / 24126 |
| Llama-2-7B / Summary | 0.771 | 0.760 | 0.776 | 0.765 | 0.722 | 1056 / 19645 |
| Mistral-7B / Data2txt | 0.799 | 0.798 | 0.796 | 0.804 | 0.769 | 1877 / 24065 |
| Mistral-7B / QA | 0.883 | 0.850 | 0.863 | 0.914 | 0.729 | 924 / 13935 |
| Mistral-7B / Summary | 0.849 | 0.827 | 0.843 | 0.857 | 0.788 | 1772 / 21104 |
| Llama-2-13B / Data2txt | 0.730 | 0.720 | 0.731 | 0.744 | 0.716 | 2913 / 33264 |
| Llama-2-13B / QA | 0.828 | 0.807 | 0.827 | 0.854 | 0.737 | 1516 / 20560 |
| Llama-2-13B / Summary | 0.743 | 0.725 | 0.730 | 0.715 | 0.643 | 542 / 18332 |

## B. Reliability ablation (does the B / uncertainty part matter?)

Each source keeps its probability; only its uncertainty (reliability) is made constant.

| Element | Z-FMN − same fusion with reliability removed |
| --- | --- |
| edge | +0.015 [+0.011, +0.020] * |
| node | +0.012 [+0.010, +0.015] * |

## C. Telling contradicted from invented content (hallucinated elements only; AUROC for Conflict)

| Element | −uncertainty u | disbelief d | 1 − P | NLI contradiction | −NLI neutral | u − best single NLI score | Conflict / Baseless |
| --- | --- | --- | --- | --- | --- | --- | --- |
| edge | 0.704 | 0.403 | 0.398 | 0.582 | 0.704 | +0.000 [+0.000, +0.000] (−NLI neutral) | 5075 / 9775 |
| node | 0.691 | 0.385 | 0.381 | 0.570 | 0.691 | -0.000 [-0.000, +0.000] (−NLI neutral) | 1881 / 3552 |

AUROC for Conflict vs Baseless; 0.5 = cannot tell them apart. Low u marks a contradiction.


## D. Calibration of the probability of hallucination (lower is better; mean over datasets)

| Probability | Edge ECE | Edge Brier | Node ECE | Node Brier |
| --- | --- | --- | --- | --- |
| Z-FMN fusion (base opinion) | 0.349 | 0.192 | 0.262 | 0.146 |
| Z-FMN fusion (context opinion) | 0.223 | 0.121 | 0.173 | 0.104 |
| Mean of the three risks | 0.319 | 0.168 | 0.257 | 0.138 |
| Logistic stacking (supervised, CV) | 0.011 | 0.057 | 0.012 | 0.060 |
| SelfCheckGPT-NLI alone | 0.172 | 0.128 | 0.186 | 0.139 |
