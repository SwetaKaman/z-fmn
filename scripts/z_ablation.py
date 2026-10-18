"""Does the Z-number / subjective-logic machinery matter? Ablations on Z-FMN outputs (no GPU).

Usage:
  python z_ablation.py Llama-2-7B=runs/full_llama-2-7b-chat Mistral-7B=runs/full_mistral-7B-instruct ... > z_ablation.md

All measures are scored on the same elements (those with an NLI verdict). A dataset is one model x task.

A. Fusion ablation. The same three evidence sources (the model's own opinion, source entailment,
   SelfCheck resample consistency) combined by:
     - Z-FMN subjective-logic fusion (training-free), with the base model opinion (constant u)
       and with the context-conditioned opinion, whose uncertainty varies per element;
     - mean and max of the three risk scores (training-free);
     - logistic regression stacking, trained with 5-fold cross-validation grouped by answer
       (supervised: an upper bound that needs labels Z-FMN does not use).
B. Reliability ablation. Z-FMN fusion where every source keeps its projected probability but its
   uncertainty is flattened to a constant, so only the reliability (B) information is removed.
C. Type separation. Among hallucinated elements: does disbelief d or uncertainty u tell
   contradicted (Conflict) from invented (Baseless) content better than the scalar P?
D. Calibration. Expected calibration error and Brier score of each probability for "hallucinated".
"""
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
NB = 1000
rng = np.random.default_rng(11)
COLS = ["kind", "wrong", "types", "sl_b", "sl_u", "sl_P", "ctx_b", "ctx_d", "ctx_u", "ctx_P",
        "nli_e", "nli_c", "selfcheck", "tri_fused_P", "response_id", "task_type"]


def opinion_from_evidence(r, s, W=2.0):
    k = r + s + W
    return r / k, s / k, W / k


def fuse(o1, o2):
    (b1, d1, u1), (b2, d2, u2) = o1, o2
    k = u1 + u2 - u1 * u2
    b = (b1 * u2 + b2 * u1) / k
    u = u1 * u2 / k
    return b, 1 - b - u, u


def P(o, a=0.5):
    return o[0] + a * o[2]


def flatten(o, u_const):
    """Same projected probability, constant uncertainty: removes the reliability information."""
    p = np.clip(P(o), 0, 1)
    b = np.clip(p - 0.5 * u_const, 0, 1 - u_const)
    return b, 1 - u_const - b, np.full_like(b, u_const)


def measures(d, strength=10.0, n_samples=10, W=2.0):
    base = (d.sl_b.values, 1 - d.sl_b.values - d.sl_u.values, d.sl_u.values)
    ctx = (d.ctx_b.values, d.ctx_d.values, d.ctx_u.values)
    nli = opinion_from_evidence(strength * d.nli_e.values, strength * d.nli_c.values, W)
    sc = opinion_from_evidence(n_samples * (1 - d.selfcheck.values), n_samples * d.selfcheck.values, W)
    tri_base = fuse(fuse(base, nli), sc)
    tri_ctx = fuse(fuse(ctx, nli), sc)
    uc = [float(np.mean(o[2])) for o in (ctx, nli, sc)]
    tri_flat = fuse(fuse(flatten(ctx, uc[0]), flatten(nli, uc[1])), flatten(sc, uc[2]))
    ub = [float(np.mean(o[2])) for o in (base, nli, sc)]
    tri_base_flat = fuse(fuse(flatten(base, ub[0]), flatten(nli, ub[1])), flatten(sc, ub[2]))
    risks = np.vstack([1 - P(base), 1 - P(nli), d.selfcheck.values])
    return {
        "Z-FMN fusion (base opinion)": 1 - P(tri_base),
        "Z-FMN fusion (context opinion)": 1 - P(tri_ctx),
        "Z-FMN fusion (base), reliability removed": 1 - P(tri_base_flat),
        "Z-FMN fusion (context), reliability removed": 1 - P(tri_flat),
        "Mean of the three risks": risks.mean(0),
        "Max of the three risks": risks.max(0),
        "SelfCheckGPT-NLI alone": d.selfcheck.values,
        "NLI not-entailed alone": 1 - d.nli_e.values,
        "Z-FMN model opinion alone": 1 - d.sl_P.values,
        "Z-FMN context opinion alone": 1 - d.ctx_P.values,
    }, {"d_ctx": tri_ctx[1], "u_ctx": tri_ctx[2], "d_base": tri_base[1], "u_base": tri_base[2],
        "nli_c": d.nli_c.values, "nli_neutral": 1 - d.nli_e.values - d.nli_c.values}


def stacked(d, folds=5):
    """Cross-validated logistic stacking of the same evidence, grouped by answer."""
    X = np.column_stack([d.sl_P, d.ctx_P, d.ctx_u, d.nli_e, d.nli_c, d.selfcheck]).astype(float)
    X = sm.add_constant((X - X.mean(0)) / (X.std(0) + 1e-9))
    y = d.wrong.values
    ids = pd.factorize(d.response_id)[0]
    fold = rng.permutation(ids.max() + 1)[ids] % folds
    out = np.full(len(y), np.nan)
    for f in range(folds):
        tr, te = fold != f, fold == f
        try:
            m = sm.Logit(y[tr], X[tr]).fit(disp=0, maxiter=200)
            out[te] = m.predict(X[te])
        except Exception:
            pass
    return out


def auc_boot(s, y, rid, Wt):
    m = ~np.isnan(s)
    s, y, rid = s[m], y[m], rid[m]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan, np.full(len(Wt), np.nan)
    o = np.argsort(s, kind="mergesort")
    s, y, rid = s[o], y[o], rid[o]
    _, g = np.unique(s, return_inverse=True)
    ng = g.max() + 1
    w = np.vstack([np.ones(len(s)), Wt[:, rid]])
    res = np.empty(len(w))
    for b in range(len(w)):
        pos = np.bincount(g[y == 1], w[b, y == 1], ng)
        neg = np.bincount(g[y == 0], w[b, y == 0], ng)
        cn = np.cumsum(neg) - neg
        den = pos.sum() * neg.sum()
        res[b] = (pos * (cn + .5 * neg)).sum() / den if den else np.nan
    return res[0], res[1:]


def ece(p, y, bins=15):
    p = np.clip(p, 0, 1)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return tot


res, cal, sep, counts, names = {}, {}, {}, {}, []
for arg in sys.argv[1:]:
    model, folder = arg.split("=", 1)
    full = pd.read_csv(f"{folder}/elements.csv", usecols=lambda c: c in COLS)
    full = full.dropna(subset=["nli_e", "selfcheck", "ctx_u"])
    full["types"] = full["types"].fillna("")
    for (task, kind), d in full.groupby(["task_type", "kind"]):
        name = (f"{model} / {task}", kind)
        if kind == "edge":
            names.append(name[0])
        ids = pd.factorize(d.response_id)[0]
        nr = ids.max() + 1
        Wt = np.stack([np.bincount(rng.integers(0, nr, nr), minlength=nr) for _ in range(NB)])
        y = d.wrong.values.astype(int)
        ms, extra = measures(d)
        ms["Logistic stacking (supervised, CV)"] = stacked(d)
        counts[name] = (int(y.sum()), len(y))
        for k, s in ms.items():
            res[(name, k)] = auc_boot(np.asarray(s, float), y, ids, Wt)
            cal[(name, k)] = (ece(np.asarray(s, float), y), float(np.mean((np.clip(s, 0, 1) - y) ** 2)))
        # C: among hallucinated elements, Conflict (1) vs Baseless (0)
        hm = (y == 1) & (d.types.str.contains("Conflict") ^ d.types.str.contains("Baseless"))
        yt = d.types[hm].str.contains("Conflict").values.astype(int)
        for k, s in (("disbelief d", extra["d_base"]), ("−uncertainty u", -extra["u_base"]),
                     ("1 − P", ms["Z-FMN fusion (base opinion)"]), ("NLI contradiction", extra["nli_c"]),
                     ("−NLI neutral", -extra["nli_neutral"])):
            sep[(name, k)] = auc_boot(np.asarray(s, float)[hm], yt, ids[hm], Wt)
        sep[(name, "n")] = (int(yt.sum()), int(len(yt) - yt.sum()))
    print(model, "done", file=sys.stderr)

ci = lambda b: (np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5))  # noqa: E731
MS = list(dict.fromkeys(k for (_, k) in res))


def pooled(kind, a, b, table=res):
    ds = [n for n in names if not np.isnan(table[((n, kind), a)][0]) and not np.isnan(table[((n, kind), b)][0])]
    p = np.mean([table[((n, kind), a)][0] - table[((n, kind), b)][0] for n in ds])
    bs = np.nanmean([table[((n, kind), a)][1] - table[((n, kind), b)][1] for n in ds], axis=0)
    lo, hi = ci(bs)
    return f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" *" if lo > 0 or hi < 0 else "")


L = []
OURS = "Z-FMN fusion (base opinion)"
L.append("## A. Fusion ablation: AUROC pooled over datasets (mean of per-dataset AUROC)\n")
L.append("| Combiner | Edge AUROC | Node AUROC | Edge: Z-FMN − this | Node: Z-FMN − this |")
L.append("| --- | --- | --- | --- | --- |")
for k in MS:
    e = np.mean([res[((n, "edge"), k)][0] for n in names])
    nd = np.mean([res[((n, "node"), k)][0] for n in names])
    de = "" if k == OURS else pooled("edge", OURS, k)
    dn = "" if k == OURS else pooled("node", OURS, k)
    L.append(f"| {k} | {e:.3f} | {nd:.3f} | {de} | {dn} |")
L.append("\n* = 95% CI of the paired difference (bootstrap over answers, stratified by dataset) excludes 0.\n")

L.append("## A2. Per dataset, edges (AUROC)\n")
show = [OURS, "Z-FMN fusion (base), reliability removed", "Mean of the three risks",
        "Logistic stacking (supervised, CV)", "SelfCheckGPT-NLI alone"]
L.append("| Dataset | " + " | ".join(show) + " | Wrong / total |")
L.append("| --- | " + " | ".join("---" for _ in show) + " | --- |")
for n in names:
    w, t = counts[(n, "edge")]
    L.append(f"| {n} | " + " | ".join(f"{res[((n, 'edge'), k)][0]:.3f}" for k in show) + f" | {w} / {t} |")

L.append("\n## B. Reliability ablation (does the B / uncertainty part matter?)\n")
L.append("Each source keeps its probability; only its uncertainty (reliability) is made constant.\n")
L.append("| Element | Z-FMN − same fusion with reliability removed |")
L.append("| --- | --- |")
for kind in ("edge", "node"):
    L.append(f"| {kind} | {pooled(kind, OURS, 'Z-FMN fusion (base), reliability removed')} |")

L.append("\n## C. Telling contradicted from invented content (hallucinated elements only; AUROC for Conflict)\n")
CK = ("−uncertainty u", "disbelief d", "1 − P", "NLI contradiction", "−NLI neutral")
L.append("| Element | " + " | ".join(CK) + " | u − best single NLI score | Conflict / Baseless |")
L.append("| --- | " + " | ".join("---" for _ in CK) + " | --- | --- |")
for kind in ("edge", "node"):
    cells = [f"{np.nanmean([sep[((n, kind), k)][0] for n in names]):.3f}" for k in CK]
    best = max(("NLI contradiction", "−NLI neutral"), key=lambda k: np.nanmean([sep[((n, kind), k)][0] for n in names]))
    nc = sum(sep[((n, kind), 'n')][0] for n in names)
    nb = sum(sep[((n, kind), 'n')][1] for n in names)
    L.append(f"| {kind} | " + " | ".join(cells) + f" | {pooled(kind, '−uncertainty u', best, sep)} ({best}) | {nc} / {nb} |")
L.append("\nAUROC for Conflict vs Baseless; 0.5 = cannot tell them apart. Low u marks a contradiction.\n")
L.append("\n## D. Calibration of the probability of hallucination (lower is better; mean over datasets)\n")
L.append("| Probability | Edge ECE | Edge Brier | Node ECE | Node Brier |")
L.append("| --- | --- | --- | --- | --- |")
for k in (OURS, "Z-FMN fusion (context opinion)", "Mean of the three risks", "Logistic stacking (supervised, CV)",
          "SelfCheckGPT-NLI alone"):
    cells = []
    for kind in ("edge", "node"):
        cells += [f"{np.mean([cal[((n, kind), k)][0] for n in names]):.3f}",
                  f"{np.mean([cal[((n, kind), k)][1] for n in names]):.3f}"]
    L.append(f"| {k} | " + " | ".join(cells) + " |")
print("\n".join(L))
