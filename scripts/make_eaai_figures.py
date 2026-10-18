"""Extra figure for the journal version: share of hallucinated links a reviewer finds when checking
only the riskiest fraction of links (mean over the nine model x task datasets).

Usage:
  python make_eaai_figures.py OUT_DIR benchmark/ragtruth_fmn_elements.csv.gz
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, INK3, GRID = "#1d1d1b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
plt.rcParams.update({
    "font.family": "Inter", "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.edgecolor": INK3, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlesize": 8,
})

out, path = sys.argv[1], sys.argv[2]
d = pd.read_csv(path)
e = d[d.element_type == "edge"].dropna(subset=["score_zfmn_full_P", "score_nli_entail", "score_selfcheck_nli"])
scores = {"Z-FMN (ours)": (1 - e.score_zfmn_full_P, BLUE, 1.8),
          "Source NLI": (1 - e.score_nli_entail, AQUA, 1.1),
          "SelfCheckGPT-NLI": (e.score_selfcheck_nli, ORANGE, 1.1),
          "Token log-prob": (-e.score_mean_logprob, VIOLET, 1.1)}
budgets = np.linspace(0, 0.5, 101)
rng = np.random.default_rng(0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.3), gridspec_kw={"wspace": 0.32})
for name, (s, col, lw) in scores.items():
    e["_s"] = s.values + rng.uniform(0, 1e-9, len(e))
    rec, prec = [], []
    for _, g in e.groupby(["model", "task"]):
        y = g.hallucinated.values[np.argsort(-g["_s"].values)]
        cum = np.concatenate([[0], np.cumsum(y)])
        n = np.round(budgets * len(y)).astype(int)
        rec.append(cum[n] / y.sum())
        prec.append(np.where(n > 0, cum[n] / np.maximum(n, 1), np.nan))
    rec, prec = np.mean(rec, 0), np.nanmean(prec, 0)
    a1.plot(budgets * 100, rec * 100, color=col, lw=lw, label=name, zorder=3 if "ours" in name else 2)
    a2.plot(budgets[2:] * 100, prec[2:] * 100, color=col, lw=lw, zorder=3 if "ours" in name else 2)
a1.plot([0, 50], [0, 50], color=INK3, lw=0.7, ls=":", label="Random order")
a2.axhline(7.4, color=INK3, lw=0.7, ls=":")
a2.text(49, 4.6, "base rate 7.4%", ha="right", fontsize=6.3, color=INK3)
for a, t, yl in ((a1, "Hallucinated links found", "Recall (%)"), (a2, "Precision of the checked links", "Precision (%)")):
    a.set_title(t)
    a.set_xlabel("Links checked, riskiest first (% of all links)")
    a.set_ylabel(yl)
    a.set_xlim(0, 50)
    a.grid(color=GRID, lw=0.6)
    a.set_axisbelow(True)
a1.set_ylim(0, 85)
a2.set_ylim(0, 50)
for b in (10, 20):
    a1.axvline(b, color=GRID, lw=1.2, zorder=1)
h, l = a1.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, handlelength=1.8, columnspacing=1.6)
fig.savefig(f"{out}/fig12_budget.pdf")
fig.savefig(f"{out}/fig12_budget.png", dpi=200)
print("wrote fig12_budget")
