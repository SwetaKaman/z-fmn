"""Full-study analysis for Z-FMN (v3 outputs): element- and answer-level AUROC, paired tests, pooling.

Usage:
  python full_study_analysis.py Mistral-7B=runs/mistral Llama-2-7B=runs/llama7 Llama-2-13B=runs/llama13 > tables.md

Each argument is Model=folder, where the folder holds elements.csv (and responses.csv, for the
answer-level tables). Every model's rows are split by task, so a dataset is one (model, task) pair.
CIs come from a bootstrap that resamples answers (elements of one answer stay together); pooled
effects average the per-dataset differences with the bootstrap stratified by dataset. Measures
missing from a folder (for example NLI in pre-v3 runs) are left blank.
"""
import csv
import sys
from collections import defaultdict

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

NB = 2000
rng = np.random.default_rng(7)

# score name -> (label, function of the column getter); higher score = more likely hallucinated
ELEMENT = {
    "logprob": ("Token log-prob", lambda c: -c("mean_lp")),
    "selfcheck": ("SelfCheckGPT-NLI", lambda c: c("selfcheck")),
    "nli_c": ("NLI contradiction", lambda c: c("nli_c")),
    "nli_ne": ("NLI not-entailed", lambda c: 1 - c("nli_e")),
    "P": ("Z-FMN P (model only)", lambda c: 1 - c("sl_P")),
    "u": ("Z-FMN u", lambda c: c("ctx_u")),
    "fP": ("Z-FMN + NLI P", lambda c: 1 - c("nli_fused_P")),
    "fd": ("Z-FMN + NLI d", lambda c: c("nli_fused_d")),
    "tP": ("Z-FMN + NLI + SelfCheck P", lambda c: 1 - c("tri_fused_P")),
    "freq": ("Resample frequency", lambda c: 1 - c("freq")),
}
ANSWER = {
    "logprob": ("Mean token log-prob", "resp_neg_mean_logprob"),
    "selfcheck": ("SelfCheckGPT-NLI", "resp_selfcheck_nli"),
    "nli_c": ("NLI max contradiction", "resp_nli_contra_max"),
    "nli_ne": ("NLI not-entailed", "resp_nli_not_entailed"),
    "P": ("Z-FMN max(1-P) edges", "resp_edge_P_max"),
    "Pn": ("Z-FMN max(1-P) nodes", "resp_node_P_max"),
    "fP": ("Z-FMN + NLI max(1-P) edges", "resp_edge_nli_fused_P_max"),
    "fPn": ("Z-FMN + NLI max(1-P) nodes", "resp_node_nli_fused_P_max"),
    "tP": ("Z-FMN + NLI + SelfCheck max(1-P) edges", "resp_edge_tri_fused_P_max"),
}
PAIRS = [("tP", "selfcheck"), ("tP", "fP"), ("P", "freq"), ("fP", "selfcheck"), ("fP", "nli_ne"), ("fP", "nli_c"), ("fP", "P"), ("P", "logprob"), ("selfcheck", "logprob")]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def read(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def auc_boot(s, y, rid, W):
    """AUROC of s for labels y on the full sample, then on each bootstrap weight row of W."""
    m = ~np.isnan(s)
    s, y, rid = s[m], y[m], rid[m]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan, np.full(len(W), np.nan)
    o = np.argsort(s, kind="mergesort")
    s, y, rid = s[o], y[o], rid[o]
    _, g = np.unique(s, return_inverse=True)
    ng = g.max() + 1
    w = np.vstack([np.ones(len(s)), W[:, rid]])
    out = np.empty(len(w))
    for b in range(len(w)):
        pos = np.bincount(g[y == 1], w[b, y == 1], ng)
        neg = np.bincount(g[y == 0], w[b, y == 0], ng)
        cn = np.cumsum(neg) - neg
        den = pos.sum() * neg.sum()
        out[b] = (pos * (cn + .5 * neg)).sum() / den if den else np.nan
    return out[0], out[1:]


def datasets():
    for arg in sys.argv[1:]:
        model, folder = arg.split("=", 1)
        el, rs = read(f"{folder}/elements.csv"), read(f"{folder}/responses.csv")
        for task in sorted({r["task_type"] for r in el}):
            yield model, task, [r for r in el if r["task_type"] == task], [r for r in rs if r["task_type"] == task]


res = {}  # (dataset, level, kind, subset, measure) -> (point, boot)
counts = {}
names = []
for model, task, el, rs in datasets():
    name = f"{model} / {task}"
    names.append(name)
    ids = sorted({r["response_id"] for r in el} | {r["response_id"] for r in rs})
    pos = {k: i for i, k in enumerate(ids)}
    W = np.stack([np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)) for _ in range(NB)])

    rid = np.array([pos[r["response_id"]] for r in el])
    y = np.array([int(r["wrong"]) for r in el])
    kind = np.array([r["kind"] for r in el])
    types = np.array([r["types"] or "" for r in el])
    col = lambda k: np.array([num(r.get(k)) for r in el])  # noqa: E731
    subsets = {"all": np.ones(len(el), bool),
               "conflict": (y == 0) | (np.char.find(types, "Conflict") >= 0),
               "baseless": (y == 0) | (np.char.find(types, "Baseless") >= 0)}
    for kd in ("edge", "node"):
        for sb, sm in subsets.items():
            m = (kind == kd) & sm
            counts[(name, kd, sb)] = (int(y[m].sum()), int(m.sum()))
            for key, (_, f) in ELEMENT.items():
                res[(name, "el", kd, sb, key)] = auc_boot(f(col)[m], y[m], rid[m], W)

    if rs:
        ra = np.array([pos[r["response_id"]] for r in rs])
        ya = np.array([int(r["wrong"]) for r in rs])
        counts[(name, "answer")] = (int(ya.sum()), len(ya))
        for key, (_, c) in ANSWER.items():
            res[(name, "ans", "-", "all", key)] = auc_boot(np.array([num(r.get(c)) for r in rs]), ya, ra, W)
    print(name, "elements", len(el), "answers", len(rs), file=sys.stderr)

ci = lambda b: (np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)) if np.isfinite(b).any() else (np.nan, np.nan)  # noqa: E731
star = lambda lo, hi: " *" if lo > 0 or hi < 0 else ""  # noqa: E731


def cell(p, b):
    if np.isnan(p):
        return ""
    lo, hi = ci(b)
    return f"{p:.3f} [{lo:.2f}, {hi:.2f}]"


def dcell(p, b):
    if np.isnan(p) or not np.isfinite(b).any():
        return ""
    lo, hi = ci(b)
    return f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}]{star(lo, hi)}"


def pooled(group, level, kd, sb, a, b):
    """Mean over datasets of AUROC(a) − AUROC(b), stratified bootstrap; datasets missing either are skipped."""
    ds = [n for n in group if (n, level, kd, sb, a) in res and not np.isnan(res[(n, level, kd, sb, a)][0])
          and not np.isnan(res[(n, level, kd, sb, b)][0])]
    if not ds:
        return np.nan, np.array([np.nan])
    p = np.mean([res[(n, level, kd, sb, a)][0] - res[(n, level, kd, sb, b)][0] for n in ds])
    bs = np.nanmean([res[(n, level, kd, sb, a)][1] - res[(n, level, kd, sb, b)][1] for n in ds], axis=0)
    return p, bs


L = []
L.append("## Table A. Element-level AUROC, all hallucinations (95% CI, bootstrap over answers)\n")
L.append("| Dataset | Element | " + " | ".join(v[0] for v in ELEMENT.values()) + " | Wrong / total |")
L.append("| --- | --- | " + " | ".join("---" for _ in ELEMENT) + " | --- |")
for n in names:
    for kd in ("edge", "node"):
        w, t = counts[(n, kd, "all")]
        L.append(f"| {n} | {kd} | " + " | ".join(cell(*res[(n, "el", kd, "all", k)]) for k in ELEMENT) + f" | {w} / {t} |")

tasks = sorted({n.split(" / ")[1] for n in names})
groups = [("All datasets", names)] + [(f"Task {t}", [n for n in names if n.endswith(" / " + t)]) for t in tasks]
L.append("\n## Table B. Pooled paired differences, element level (mean of per-dataset AUROC differences; * = CI excludes 0)\n")
L.append("| Pool | Element | Subset | " + " | ".join(f"{ELEMENT[a][0]} − {ELEMENT[b][0]}" for a, b in PAIRS) + " |")
L.append("| --- | --- | --- | " + " | ".join("---" for _ in PAIRS) + " |")
for gname, g in groups:
    for kd in ("edge", "node"):
        for sb in ("all", "conflict", "baseless"):
            L.append(f"| {gname} | {kd} | {sb} | " + " | ".join(dcell(*pooled(g, "el", kd, sb, a, b)) for a, b in PAIRS) + " |")

L.append("\n## Table C. Edge − node (same measure, pooled over all datasets)\n")
L.append("| Measure | Edge − node |")
L.append("| --- | --- |")
for k in ("P", "fP", "tP", "selfcheck", "nli_ne"):
    ds = [n for n in names if not np.isnan(res[(n, "el", "edge", "all", k)][0])]
    if ds:
        p = np.mean([res[(n, "el", "edge", "all", k)][0] - res[(n, "el", "node", "all", k)][0] for n in ds])
        bs = np.nanmean([res[(n, "el", "edge", "all", k)][1] - res[(n, "el", "node", "all", k)][1] for n in ds], axis=0)
        L.append(f"| {ELEMENT[k][0]} | {dcell(p, bs)} |")

ans = [n for n in names if (n, "answer") in counts and 0 < counts[(n, "answer")][0] < counts[(n, "answer")][1]]
if ans:
    L.append("\n## Table D. Answer-level AUROC: does the answer contain any hallucination? (needs clean answers)\n")
    L.append("| Dataset | " + " | ".join(v[0] for v in ANSWER.values()) + " | Hallucinated / answers |")
    L.append("| --- | " + " | ".join("---" for _ in ANSWER) + " | --- |")
    for n in ans:
        w, t = counts[(n, "answer")]
        L.append(f"| {n} | " + " | ".join(cell(*res[(n, "ans", "-", "all", k)]) for k in ANSWER) + f" | {w} / {t} |")
    apairs = [("tP", "selfcheck"), ("fP", "selfcheck"), ("fP", "nli_ne"), ("fP", "P"), ("P", "logprob")]
    L.append("\n## Table E. Pooled paired differences, answer level\n")
    L.append("| Pool | " + " | ".join(f"{ANSWER[a][0]} − {ANSWER[b][0]}" for a, b in apairs) + " |")
    L.append("| --- | " + " | ".join("---" for _ in apairs) + " |")
    for gname, g in groups:
        g = [n for n in g if n in ans]
        if g:
            L.append(f"| {gname} | " + " | ".join(dcell(*pooled(g, "ans", "-", "all", a, b)) for a, b in apairs) + " |")
print("\n".join(L))
