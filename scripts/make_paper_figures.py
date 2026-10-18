"""Camera-ready figures for the Z-FMN paper (no GPU).

Usage:
  python make_paper_figures.py OUT_DIR EXTRA_JSON RQ4_MD Llama-2-7B=runs/full_llama-2-7b-chat Mistral-7B=... Llama-2-13B=...

EXTRA_JSON is the output of extra_analysis.py and RQ4_MD the output of rq4_emotion_analysis.py.
Writes one PDF per figure (vector, fonts embedded) plus a PNG preview, and numbers.json with every
value the figures show. Scores and pooling are those of z_ablation.py.
"""
import json
import os
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

warnings.filterwarnings("ignore")
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "z_ablation.py")).read()
exec(_src.split("\nres, cal, sep")[0])  # measures, stacked, auc_boot, ece, fuse, P, opinion_from_evidence, rng, NB

# ---------------- style ----------------
INK, INK2, INK3, GRID, SURF = "#1d1d1b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100"
GREY = "#b9b8b2"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_ORANGE = ["#fde3d6", "#f7b99c", "#f19066", "#eb6834", "#c4501f", "#8f3a16"]
COL_B, COL_D, COL_U = BLUE, ORANGE, "#c9c8c2"  # belief, disbelief, uncertainty
plt.rcParams.update({
    "font.family": "Inter", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.edgecolor": INK3,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": 300,
    "mathtext.fontset": "custom", "mathtext.rm": "Inter", "mathtext.it": "Inter:italic",
    "mathtext.bf": "Inter:bold", "axes.titleweight": "semibold", "axes.titlelocation": "left",
})
COL1, COL2 = 3.03, 6.3  # ACL single and double column widths (in)

out, extra_path, rq4_path = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
EXTRA = json.load(open(extra_path))
NUM = {}


def save(fig, name):
    fig.savefig(f"{out}/{name}.pdf")
    fig.savefig(f"{out}/{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name, file=sys.stderr)


def hgrid(ax):
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ---------------- Figure 1: method overview ----------------
def opinion_bar(ax, x, y, w, h, b, d, u):
    for start, width, col in ((0, b, COL_B), (b, d, COL_D), (b + d, u, COL_U)):
        ax.add_patch(Rectangle((x + start * w, y), width * w - 0.004, h, fc=col, ec="none"))


def card(ax, x, y, w, h, fc="#ffffff", ec=GRID, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.018", fc=fc, ec=ec, lw=lw))


def arrow(ax, a, b, col=INK3, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=col,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def mini_net(ax, cx, cy, s, risks=None):
    pts = {"a": (-1, 0.4), "b": (-0.2, 0.9), "c": (0.1, -0.1), "d": (1, 0.5), "e": (0.7, -0.8), "f": (-0.8, -0.7)}
    E = [("a", "b"), ("b", "c"), ("a", "c"), ("c", "d"), ("c", "e"), ("a", "f"), ("f", "c"), ("d", "e")]
    cmap = LinearSegmentedColormap.from_list("r", SEQ_ORANGE)
    for i, (p, q) in enumerate(E):
        (x1, y1), (x2, y2) = pts[p], pts[q]
        col = INK3 if risks is None else cmap(risks[i])
        lw = 0.9 if risks is None else 0.6 + 2.2 * risks[i]
        ax.plot([cx + s * x1, cx + s * x2], [cy + s * y1, cy + s * y2], color=col, lw=lw, solid_capstyle="round", zorder=2)
    for x, y in pts.values():
        ax.add_patch(Circle((cx + s * x, cy + s * y), s * 0.2, fc="#ffffff", ec=INK2, lw=0.7, zorder=3))


fig = plt.figure(figsize=(COL2, 2.35))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.34)
ax.axis("off")
stages = [(0.005, 0.29, "1  Build the network"), (0.305, 0.39, "2  Evidence as subjective-logic opinions"),
          (0.705, 0.29, "3  Fuse and score")]
for x, w, t in stages:
    ax.add_patch(FancyBboxPatch((x, 0.008), w, 0.322, boxstyle="round,pad=0,rounding_size=0.012",
                                fc="#f4f3f0", ec="none"))
    ax.text(x + 0.012, 0.312, t, fontsize=7.6, fontweight="semibold", va="center", color=INK)
# stage 1
for i, (t, sub) in enumerate((("Source passage", "retrieved context"), ("LLM answer", "token log-probs"),
                              ("10 resamples", "T = 0.7"))):
    y = 0.225 - i * 0.085
    card(ax, 0.015, y, 0.125, 0.065)
    ax.text(0.0775, y + 0.041, t, ha="center", fontsize=6.6, fontweight="medium")
    ax.text(0.0775, y + 0.017, sub, ha="center", fontsize=5.8, color=INK3)
card(ax, 0.16, 0.07, 0.125, 0.22)
mini_net(ax, 0.2225, 0.222, 0.024)
ax.text(0.2225, 0.15, "Textual forma\nmentis network", ha="center", fontsize=6.4, fontweight="medium", linespacing=1.15)
ax.text(0.2225, 0.084, "concepts + links\nwithin three\ndependency hops", ha="center", fontsize=5.6, color=INK3,
        linespacing=1.15)
arrow(ax, (0.14, 0.1735), (0.16, 0.1735))
# stage 2
ops = [("Model opinion", "token confidence pooled over\nanswer + resamples, r + s = 11", (0.52, 0.33, 0.15)),
       ("Source opinion", "DeBERTa-MNLI vs source:\nr = 10e,  s = 10c  (neutral \u2192 u)", (0.18, 0.47, 0.35)),
       ("Consistency opinion", "SelfCheckGPT-NLI score s:\nr = 10(1\u2212s),  s = 10s", (0.28, 0.55, 0.17))]
for i, (t, sub, (b, d, u)) in enumerate(ops):
    y = 0.218 - i * 0.092
    card(ax, 0.315, y, 0.37, 0.076)
    ax.text(0.325, y + 0.054, t, fontsize=6.7, fontweight="medium", va="center")
    ax.text(0.325, y + 0.024, sub, fontsize=5.6, color=INK3, va="center", linespacing=1.2)
    opinion_bar(ax, 0.535, y + 0.03, 0.14, 0.018, b, d, u)
    ax.text(0.535, y + 0.06, f"b {b:.2f}", fontsize=5.5, color=BLUE, va="center")
    ax.text(0.584, y + 0.06, f"d {d:.2f}", fontsize=5.5, color="#c4501f", va="center")
    ax.text(0.633, y + 0.06, f"u {u:.2f}", fontsize=5.5, color=INK2, va="center")
    arrow(ax, (0.285, 0.18), (0.315, y + 0.038), rad=0.0)
ax.text(0.325, 0.02, "each opinion: b + d + u = 1,  probability P = b + u/2", fontsize=5.8, color=INK2)
# stage 3
ax.add_patch(Circle((0.745, 0.172), 0.021, fc="#ffffff", ec=BLUE, lw=1.1, zorder=3))
ax.text(0.745, 0.171, "\u2295", ha="center", va="center", fontsize=10, color=BLUE, zorder=4)
ax.text(0.745, 0.06, "cumulative\nfusion", ha="center", fontsize=6.0, color=INK2, linespacing=1.1)
for i in range(3):
    arrow(ax, (0.685, 0.256 - i * 0.092), (0.725, 0.172 + (1 - i) * 0.012), rad=0.0)
card(ax, 0.8, 0.178, 0.188, 0.118, ec=BLUE, lw=0.9)
ax.text(0.81, 0.274, "Z-number  Z = (A, B)", fontsize=6.9, fontweight="semibold", color=INK, va="center")
ax.text(0.81, 0.25, "A  value across the 11 texts", fontsize=5.7, color=INK3, va="center")
ax.text(0.81, 0.23, "B  reliability (b, P, b + u)", fontsize=5.7, color=INK3, va="center")
opinion_bar(ax, 0.81, 0.19, 0.168, 0.016, 0.21, 0.72, 0.07)
card(ax, 0.8, 0.04, 0.188, 0.122)
mini_net(ax, 0.838, 0.101, 0.02, risks=[0.1, 0.85, 0.3, 0.15, 0.95, 0.2, 0.25, 0.4])
ax.text(0.875, 0.138, "risk = 1 \u2212 P", fontsize=6.6, fontweight="medium", va="center")
ax.text(0.875, 0.112, "per concept, link", fontsize=5.7, color=INK3, va="center")
ax.text(0.875, 0.09, "answer: max over", fontsize=5.7, color=INK3, va="center")
ax.text(0.875, 0.072, "its links", fontsize=5.7, color=INK3, va="center")
arrow(ax, (0.765, 0.186), (0.8, 0.236))
arrow(ax, (0.765, 0.158), (0.8, 0.1))
for lab, col, x in (("belief", COL_B, 0.758), ("disbelief", COL_D, 0.818), ("uncertainty", COL_U, 0.895)):
    ax.add_patch(Rectangle((x, 0.016), 0.01, 0.008, fc=col, ec="none"))
    ax.text(x + 0.014, 0.0195, lab, fontsize=5.6, color=INK2, va="center")
save(fig, "fig1_overview")

# ---------------- load data and score ----------------
OURS = "Z-FMN fusion (base opinion)"
import pickle  # noqa: E402
CACHE = f"{out}/.scored.pkl"  # scoring takes minutes; delete this file after the data change
if os.path.exists(CACHE):
    res, names, rocs, nli_ops, cal_pts, frames = pickle.load(open(CACHE, "rb"))
else:
    res, names, rocs, nli_ops, cal_pts, loso = {}, [], [], [], [], {}
    frames = {}
    for arg in sys.argv[4:]:
        model, folder = arg.split("=", 1)
        full = pd.read_csv(f"{folder}/elements.csv")
        full = full.dropna(subset=["nli_e", "selfcheck", "ctx_u"])
        full["types"] = full["types"].fillna("")
        frames[model] = full
        for (task, kind), d in full.groupby(["task_type", "kind"]):
            name = f"{model} / {task}"
            if kind == "edge":
                names.append(name)
            ids = pd.factorize(d.response_id)[0]
            nr = ids.max() + 1
            Wt = np.stack([np.bincount(rng.integers(0, nr, nr), minlength=nr) for _ in range(NB)])
            y = d.wrong.values.astype(int)
            ms, extra = measures(d)
            ms["Logistic stacking (supervised, CV)"] = stacked(d)
            ms["Token log-prob"] = -d.mean_lp.values
            base = (d.sl_b.values, 1 - d.sl_b.values - d.sl_u.values, d.sl_u.values)
            nli = opinion_from_evidence(10 * d.nli_e.values, 10 * d.nli_c.values)
            sc = opinion_from_evidence(10 * (1 - d.selfcheck.values), 10 * d.selfcheck.values)
            ms["without model opinion"] = 1 - P(fuse(nli, sc))
            ms["without source opinion"] = 1 - P(fuse(base, sc))
            ms["without consistency opinion"] = 1 - P(fuse(base, nli))
            for k, s in ms.items():
                res[(name, kind, k)] = auc_boot(np.asarray(s, float), y, ids, Wt)
            if kind == "edge":
                rocs.append(pd.DataFrame({"y": y, "ours": ms[OURS],
                                          "sc": ms["SelfCheckGPT-NLI alone"], "nli": ms["NLI not-entailed alone"],
                                          "lp": ms["Token log-prob"], "model": ms["Z-FMN model opinion alone"],
                                          "stack": ms["Logistic stacking (supervised, CV)"], "ds": name}))
                tc, tb = d.types.str.contains("Conflict").values, d.types.str.contains("Baseless").values
                cls = np.where(y == 0, "clean", np.where(tc, np.where(tb, "both", "conflict"), "baseless"))
                nli_ops.append(pd.DataFrame({"b": nli[0], "d": nli[1], "u": nli[2], "cls": cls}))
                cal_pts.append(pd.DataFrame({"y": y, "ours": ms[OURS], "sc": ms["SelfCheckGPT-NLI alone"],
                                             "stack": ms["Logistic stacking (supervised, CV)"]}))
        print(model, "scored", file=sys.stderr)
    pickle.dump((res, names, rocs, nli_ops, cal_pts, frames), open(CACHE, "wb"))


def ci(b):
    return float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


def pooled(kind, k, base=None):
    p = np.mean([res[(n, kind, k)][0] for n in names])
    bs = np.nanmean([res[(n, kind, k)][1] for n in names], axis=0)
    if base is None:
        return float(p), *ci(bs)
    pb = np.mean([res[(n, kind, base)][0] for n in names])
    bb = np.nanmean([res[(n, kind, base)][1] for n in names], axis=0)
    return float(p - pb), *ci(bs - bb)


# ---------------- Figure 2: main comparison (forest) ----------------
SHOW = [(OURS, "Z-FMN (ours)", BLUE), ("Logistic stacking (supervised, CV)", "Supervised stacking\u2020", INK2),
        ("Mean of the three risks", "Mean of the three risks", INK3), ("NLI not-entailed alone", "Source NLI", AQUA),
        ("SelfCheckGPT-NLI alone", "SelfCheckGPT-NLI", ORANGE), ("Max of the three risks", "Max of the three risks", INK3),
        ("Token log-prob", "Token log-probability", VIOLET), ("Z-FMN model opinion alone", "Model opinion only", INK3)]
fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.25), sharey=True, gridspec_kw={"wspace": 0.08})
NUM["main"] = {}
for ax, kind, title in ((axes[0], "edge", "Hallucinated links"), (axes[1], "node", "Hallucinated concepts")):
    yy = np.arange(len(SHOW))[::-1]
    for yi, (k, lab, col) in zip(yy, SHOW):
        p, lo, hi = pooled(kind, k)
        NUM["main"].setdefault(lab, {})[kind] = [p, lo, hi]
        if k == OURS:
            ax.axhspan(yi - 0.45, yi + 0.45, color="#eaf2fc", lw=0, zorder=0)
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.6, solid_capstyle="round", zorder=2)
        ax.scatter([p], [yi], s=26 if k == OURS else 18, color=col, ec="white", lw=0.8, zorder=3)
        ax.text(hi + 0.006, yi, f"{p:.3f}", va="center", fontsize=6.6,
                fontweight="semibold" if k == OURS else "normal", color=INK if k == OURS else INK2)
    ours = pooled(kind, OURS)[0]
    ax.axvline(ours, color=BLUE, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.set_yticks(yy, [lab for _, lab, _ in SHOW])
    for tl, (k, _, _) in zip(ax.get_yticklabels(), SHOW):
        if k == OURS:
            tl.set_fontweight("semibold")
            tl.set_color(INK)
    ax.set_xlim(0.55, 0.845)
    ax.set_xticks([0.6, 0.65, 0.7, 0.75, 0.8])
    hgrid(ax)
    ax.set_xlabel("AUROC, mean over 9 datasets (95% CI)")
    ax.set_title(title)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
save(fig, "fig2_main")
NUM["delta_vs_ours"] = {kind: {k: list(pooled(kind, OURS, k)) for k in sorted({x[2] for x in res}) if k != OURS}
                        for kind in ("edge", "node")}


# ---------------- Figure 3: ROC and PR ----------------
def roc_pr(s, y):
    o = np.argsort(-s, kind="mergesort")
    y = y[o]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    return (np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]], tp / tp[-1], tp / (tp + fp))


R = pd.concat(rocs, ignore_index=True)
CURVES = [("ours", "Z-FMN (ours)", BLUE, "-", 1.6), ("nli", "Source NLI", AQUA, "-", 1.1),
          ("sc", "SelfCheckGPT-NLI", ORANGE, "-", 1.1), ("lp", "Token log-prob", VIOLET, "-", 1.0),
          ("model", "Model opinion only", GREY, "-", 1.0)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL1, 1.75), gridspec_kw={"wspace": 0.42})
for c, lab, col, ls, lw in CURVES[::-1]:
    m = R[c].notna().values
    f, t, rc, pr = roc_pr(R[c].values[m], R.y.values[m])
    a1.plot(f, t, color=col, lw=lw, ls=ls, label=lab, solid_capstyle="round")
    step = max(1, len(rc) // 400)
    a2.plot(rc[::step], pr[::step], color=col, lw=lw, ls=ls, solid_capstyle="round")
base = R.y.mean()
a1.plot([0, 1], [0, 1], color=GRID, lw=0.7, zorder=0)
a2.axhline(base, color=GRID, lw=0.7, zorder=0)
a1.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
a2.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 0.6))
a1.set_title("ROC")
a2.set_title("Precision\u2013recall")
h, l = a1.get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.0, handlelength=1.4)
for ax in (a1, a2):
    ax.set_aspect("auto")
save(fig, "fig3_roc_pr")
NUM["pooled_curves_base_rate"] = float(base)


# ---------------- Figure 4: per-dataset gain heatmaps ----------------
models = list(dict.fromkeys(n.split(" / ")[0] for n in names))
tasks = ["QA", "Summary", "Data2txt"]
TL = {"QA": "QA", "Summary": "Sum.", "Data2txt": "D2T"}
cmap_div = LinearSegmentedColormap.from_list("div", ["#c43c3b", "#e98e8d", "#f0efec", "#86b6ef", "#1c5cab"])
fig, axes = plt.subplots(1, 2, figsize=(COL1, 1.5), gridspec_kw={"wspace": 0.08})
NUM["heat"] = {}
for ax, (b, lab) in zip(axes, (("SelfCheckGPT-NLI alone", "vs SelfCheckGPT"), ("NLI not-entailed alone", "vs source NLI"))):
    M = np.array([[res[(f"{m} / {t}", "edge", OURS)][0] - res[(f"{m} / {t}", "edge", b)][0] for t in tasks] for m in models])
    NUM["heat"][lab] = M.round(4).tolist()
    ax.imshow(M, cmap=cmap_div, norm=TwoSlopeNorm(0, -0.16, 0.16), aspect="auto")
    for i in range(len(models)):
        for j in range(len(tasks)):
            ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=6.4,
                    color="white" if abs(M[i, j]) > 0.1 else INK)
    ax.set_xticks(range(3), [TL[t] for t in tasks])
    ax.set_yticks(range(len(models)), models if ax is axes[0] else [])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(f"Z-FMN {lab}", fontsize=7.0, loc="center")
save(fig, "fig4_heatmap")

# ---------------- Figure 5: ablation ----------------
AB = [("without model opinion", "Drop model opinion"), ("without source opinion", "Drop source opinion"),
      ("without consistency opinion", "Drop consistency opinion"),
      ("Z-FMN fusion (base), reliability removed", "Remove reliability (constant u)"),
      ("Mean of the three risks", "Average instead of fuse"), ("Max of the three risks", "Max instead of fuse"),
      ("Z-FMN fusion (context opinion)", "Context-conditioned model opinion")]
fig, ax = plt.subplots(figsize=(COL1, 2.2))
NUM["ablation"] = {}
yy = np.arange(len(AB))[::-1]
off = {"edge": 0.19, "node": -0.19}
for kind, col, lab in (("edge", BLUE, "links"), ("node", "#86b6ef", "concepts")):
    for yi, (k, _) in zip(yy, AB):
        dlt, lo, hi = pooled(kind, k, OURS)
        NUM["ablation"].setdefault(k, {})[kind] = [dlt, lo, hi]
        ax.barh(yi + off[kind], dlt, height=0.34, color=col, lw=0, label=lab if yi == yy[0] else None)
        ax.plot([lo, hi], [yi + off[kind]] * 2, color=INK2, lw=0.7)
        ax.text(max(dlt, hi, 0) + 0.003, yi + off[kind], f"{dlt:+.3f}", ha="left", va="center", fontsize=5.8, color=INK2)
ax.axvline(0, color=INK2, lw=0.7)
ax.set_yticks(yy, [lab for _, lab in AB])
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(-0.13, 0.03)
ax.set_xticks([-0.1, -0.05, 0])
hgrid(ax)
ax.set_xlabel("Change in AUROC relative to full Z-FMN (95% CI)")
ax.legend(loc="lower left", ncol=2, handlelength=1, bbox_to_anchor=(0.0, 1.0))
save(fig, "fig5_ablation")

# ---------------- Figure 6: opinion simplex of the source opinion ----------------
O = pd.concat(nli_ops, ignore_index=True)
S3 = np.sqrt(3) / 2


def bary(b, d, u):  # disbelief bottom-left, belief bottom-right, uncertainty top
    return b + u / 2, u * S3


fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.0), gridspec_kw={"wspace": 0.05})
cm_blue = LinearSegmentedColormap.from_list("sb", ["#fcfcfb"] + SEQ_BLUE)
NUM["simplex"] = {}
for ax, (cls, title) in zip(axes, (("clean", "Correct links"), ("conflict", "Contradicted links"),
                                   ("baseless", "Invented (baseless) links"))):
    g = O[O.cls == cls]
    x, y = bary(g.b.values, g.d.values, g.u.values)
    ax.add_patch(Polygon([(0, 0), (1, 0), (0.5, S3)], closed=True, fc="#f7f7f5", ec=INK3, lw=0.7, zorder=0))
    hb = ax.hexbin(x, y, gridsize=22, extent=(0, 1, 0, S3), cmap=cm_blue, mincnt=1, lw=0.1, edgecolors="none",
                   C=np.full(len(x), 100.0 / len(x)), reduce_C_function=np.sum, vmin=0, vmax=3.0)
    hb.set_clip_path(Polygon([(0, 0), (1, 0), (0.5, S3)], closed=True, transform=ax.transData))
    mb, md, mu = g.b.mean(), g.d.mean(), g.u.mean()
    NUM["simplex"][cls] = {"n": int(len(g)), "b": float(mb), "d": float(md), "u": float(mu)}
    mx, my = bary(mb, md, mu)
    ax.scatter([mx], [my], marker="o", s=28, color=ORANGE, ec="white", lw=0.9, zorder=4)
    ax.text(0.5, -0.215, f"mean  b {mb:.2f} \u00b7 d {md:.2f} \u00b7 u {mu:.2f}", ha="center", fontsize=6.2, color=INK, zorder=5)
    umin = 2 / 12
    ax.plot([umin * 0.5, 1 - umin * 0.5], [umin * S3] * 2, color=INK3, lw=0.5, ls=(0, (2, 2)))
    ax.text(-0.02, -0.03, "disbelief", ha="left", va="top", fontsize=6.2, color=INK2)
    ax.text(1.02, -0.03, "belief", ha="right", va="top", fontsize=6.2, color=INK2)
    ax.text(0.5, S3 + 0.03, "uncertainty", ha="center", va="bottom", fontsize=6.2, color=INK2)
    ax.set_title(f"{title}\n(n = {len(g):,})", loc="center", fontsize=7.0, pad=10, linespacing=1.3)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.25, S3 + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
cax = fig.add_axes([0.93, 0.25, 0.008, 0.5])
cb = fig.colorbar(hb, cax=cax)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=5.6, length=2)
cb.set_label("% of links per cell", fontsize=6)
save(fig, "fig6_simplex")

# ---------------- Figure 7: hallucination subtypes ----------------
ST = [("Evident Conflict", "Evident\nconflict"), ("Subtle Conflict", "Subtle\nconflict*"),
      ("Evident Baseless Info", "Evident\nbaseless"), ("Subtle Baseless Info", "Subtle\nbaseless")]
MK = [("zfmn", "Z-FMN (ours)", BLUE), ("nli", "Source NLI", AQUA), ("selfcheck", "SelfCheckGPT-NLI", ORANGE),
      ("logprob", "Token log-prob", VIOLET)]
fig, ax = plt.subplots(figsize=(COL1, 1.85))
w = 0.19
for j, (k, lab, col) in enumerate(MK):
    xs = np.arange(len(ST)) + (j - 1.5) * w
    v = [EXTRA["subtype"][s][k] for s, _ in ST]
    ax.bar(xs, [x["auc"] - 0.5 for x in v], bottom=0.5, width=w - 0.025, color=col, lw=0, label=lab)
    ax.errorbar(xs, [x["auc"] for x in v], yerr=[[x["auc"] - x["ci"][0] for x in v], [x["ci"][1] - x["auc"] for x in v]],
                fmt="none", ecolor=INK2, elinewidth=0.6, capsize=0)
ax.set_xticks(range(len(ST)), [s for _, s in ST])
ax.set_ylim(0.5, 0.9)
ax.set_ylabel("Link AUROC (95% CI)")
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="x", length=0)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, handlelength=0.9, columnspacing=0.8)
save(fig, "fig7_subtypes")

# ---------------- Figure 8: sensitivity ----------------
sens = np.array(EXTRA["sensitivity"]["auc"])
K = EXTRA["sensitivity"]["k_nli"]
fig, ax = plt.subplots(figsize=(1.95, 1.75))
im = ax.imshow(sens, cmap=LinearSegmentedColormap.from_list("sq", SEQ_BLUE[:6]), origin="lower", aspect="auto")
for i in range(len(K)):
    for j in range(len(K)):
        ax.text(j, i, f"{sens[i, j]:.3f}"[1:], ha="center", va="center", fontsize=5.6,
                color="white" if sens[i, j] > np.percentile(sens, 55) else INK)
i0 = K.index(10)
ax.add_patch(Rectangle((i0 - 0.5, i0 - 0.5), 1, 1, fill=False, ec=ORANGE, lw=1.4))
ax.set_xticks(range(len(K)), K)
ax.set_yticks(range(len(K)), K)
ax.set_xlabel("Consistency evidence weight")
ax.set_ylabel("Source evidence weight k")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
cb.ax.tick_params(labelsize=5.8, length=2)
cb.outline.set_visible(False)
cb.set_label("Link AUROC", fontsize=6.2)
save(fig, "fig8_sensitivity")

# ---------------- Figure 9: calibration ----------------
Kc = pd.concat(cal_pts, ignore_index=True)
fig = plt.figure(figsize=(1.6, 1.85))
gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1], hspace=0.08)
a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
bins = np.linspace(0, 1, 11)
NUM["calibration"] = {}
for c, lab, col in (("ours", "Z-FMN", BLUE), ("sc", "SelfCheckGPT", ORANGE), ("stack", "Stacking\u2020", INK3)):
    p = np.clip(Kc[c].values, 0, 1)
    idx = np.minimum(np.digitize(p, bins) - 1, 9)
    keep = [b for b in range(10) if (idx == b).sum() >= 30]
    xs = [p[idx == b].mean() for b in keep]
    ys = [Kc.y.values[idx == b].mean() for b in keep]
    e = ece(p, Kc.y.values)
    NUM["calibration"][lab] = float(e)
    a1.plot(xs, ys, "-o", color=col, ms=2.8, lw=1.1, mec="white", mew=0.5, label=f"{lab} (ECE {e:.3f})")
    a2.hist(p, bins=40, range=(0, 1), histtype="step", color=col, lw=0.9, density=True)
a1.plot([0, 1], [0, 1], color=GRID, lw=0.7, zorder=0)
a1.set(xlim=(0, 1), ylim=(0, 0.75), ylabel="Observed rate")
a1.set_xticklabels([])
a1.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), fontsize=5.6, handlelength=1.2, labelspacing=0.25)
a2.set(xlim=(0, 1), xlabel="Predicted probability of hallucination")
a2.set_yticks([])
a2.spines["left"].set_visible(False)
save(fig, "fig9_calibration")

# ---------------- Figure 10: emotion ----------------
txt = open(rq4_path).read()
coef = {}
for line in txt.splitlines():
    m = re.match(r"\| (abs_valence|n_emotions|frame_valence_spread|zipf|length|log_degree) \| ([+-]?\d\.\d+) \| \[([+-]?\d\.\d+), ([+-]?\d\.\d+)\]", line)
    if m:
        coef[m.group(1)] = tuple(float(m.group(i)) for i in (2, 3, 4))
aur = dict(re.findall(r"\| (emotional|neutral) \| (\d\.\d+) \|", txt))
inter = re.search(r"\| risk_z:emotional \| (\d\.\d+) \| \[(\d\.\d+), (\d\.\d+)\]", txt)
NUM["rq4"] = {"coef": coef, "auroc": aur, "interaction": [float(inter.group(i)) for i in (1, 2, 3)]}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(2.75, 1.6), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.6})
lab = {"abs_valence": "|valence|", "n_emotions": "# emotions", "frame_valence_spread": "frame valence spread",
       "zipf": "word frequency", "length": "word length", "log_degree": "log degree"}
order = ["abs_valence", "n_emotions", "frame_valence_spread", "zipf", "length", "log_degree"]
yy = np.arange(len(order))[::-1]
for yi, k in zip(yy, order):
    c, lo, hi = coef[k]
    col = BLUE if k in ("abs_valence", "n_emotions", "frame_valence_spread") else INK3
    a1.plot([lo, hi], [yi, yi], color=col, lw=1.4)
    a1.scatter([c], [yi], s=14, color=col, ec="white", lw=0.6, zorder=3)
a1.axvline(0, color=INK2, lw=0.6)
a1.set_yticks(yy, [lab[k] for k in order])
a1.tick_params(axis="y", length=0)
a1.spines["left"].set_visible(False)
a1.set_xlabel("Effect on logit P (per SD)")
a1.set_title("Reliability model")
hgrid(a1)
vals = [float(aur["emotional"]), float(aur["neutral"])]
a2.bar([0, 1], [v - 0.5 for v in vals], bottom=0.5, color=[ORANGE, INK3], width=0.62, lw=0)
for x, v in zip([0, 1], vals):
    a2.text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.2)
a2.set_xticks([0, 1], ["with\nemotion", "neutral"])
a2.set_ylim(0.5, 0.68)
a2.set_ylabel("Concept AUROC of 1 \u2212 P")
a2.set_title("Detection")
a2.tick_params(axis="x", length=0)
save(fig, "fig10_emotion")

# ---------------- Figure 11: example network ----------------
full = frames[list(frames)[0]]
g = full.groupby("response_id").agg(n=("kind", "size"), w=("wrong", "sum"), t=("task_type", "first"))
cand = g[(g.n.between(30, 55)) & (g.w.between(4, 14))]
rid = cand.index[0] if len(cand) else g.w.idxmax()
d = full[full.response_id == rid]
ms, _ = measures(d)
d = d.assign(risk=ms[OURS])
G = nx.Graph()
for _, r in d[d.kind == "node"].iterrows():
    G.add_node(r.key, wrong=r.wrong)
for _, r in d[d.kind == "edge"].iterrows():
    a, b = str(r.key).split("|", 1)
    G.add_edge(a, b, risk=r.risk, wrong=r.wrong)
pos = nx.kamada_kawai_layout(G)
fig, ax = plt.subplots(figsize=(COL1, 2.35))
cmap_r = LinearSegmentedColormap.from_list("r", ["#e9e8e4"] + SEQ_ORANGE[1:])
ed = sorted(G.edges(data=True), key=lambda e: e[2]["risk"])
for a, b, e in ed:
    (x1, y1), (x2, y2) = pos[a], pos[b]
    ax.plot([x1, x2], [y1, y2], color=cmap_r(e["risk"]), lw=0.6 + 2.6 * e["risk"], solid_capstyle="round", zorder=1)
    if e["wrong"]:
        ax.plot([x1, x2], [y1, y2], color=INK, lw=0.6, ls=(0, (2.2, 1.6)), zorder=2)
for n, (x, y) in pos.items():
    hal = G.nodes[n].get("wrong")
    ax.text(x, y, n, ha="center", va="center", fontsize=6.0, zorder=4, color=INK,
            bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.35", fc="#fde3d6" if hal else "white",
                      ec="#c4501f" if hal else INK3, lw=0.6))
sm_ = plt.cm.ScalarMappable(cmap=cmap_r, norm=plt.Normalize(0, 1))
cb = fig.colorbar(sm_, ax=ax, fraction=0.035, pad=0.01, shrink=0.75)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=5.8, length=2)
cb.set_label("Z-FMN risk 1 \u2212 P", fontsize=6.2)
ax.plot([], [], color=INK, lw=0.6, ls=(0, (2.2, 1.6)), label="link inside an annotated hallucination")
ax.scatter([], [], marker="s", s=22, fc="#fde3d6", ec="#c4501f", lw=0.6, label="hallucinated concept")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=5.9, handlelength=1.6)
ax.axis("off")
ax.margins(0.06)
NUM["example"] = {"model": list(frames)[0], "response_id": str(rid), "task": d.task_type.iloc[0],
                  "nodes": G.number_of_nodes(), "links": G.number_of_edges(),
                  "hallucinated_links": int(sum(e["wrong"] for _, _, e in G.edges(data=True))),
                  "auroc": float(auc_boot(d[d.kind == "edge"].risk.values, d[d.kind == "edge"].wrong.values,
                                          np.zeros(len(d[d.kind == "edge"]), int), np.ones((1, 1)))[0])}
save(fig, "fig11_example")

json.dump(NUM, open(f"{out}/numbers.json", "w"), indent=1, default=float)
print(json.dumps(NUM, indent=1, default=float)[:4000])
