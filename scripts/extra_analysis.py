"""Extra analyses for the paper (no GPU): hallucination subtypes, model scale, evidence-weight
sensitivity and answer-level comparisons against plain NLI.

Usage:
  python extra_analysis.py OUT.json Llama-2-7B=runs/full_llama-2-7b-chat Mistral-7B=... Llama-2-13B=...

Pooling follows z_ablation.py: a dataset is one model x task, AUROC is averaged over datasets and
CIs come from a bootstrap over answers stratified by dataset.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "z_ablation.py")).read()
exec(_src.split("\nres, cal, sep")[0])  # measures, auc_boot, rng, NB, COLS

OURS = "Z-FMN fusion (base opinion)"
SUBTYPES = ["Evident Conflict", "Subtle Conflict", "Evident Baseless Info", "Subtle Baseless Info"]
METHODS = {"zfmn": OURS, "selfcheck": "SelfCheckGPT-NLI alone", "nli": "NLI not-entailed alone",
           "model": "Z-FMN model opinion alone"}
GRID_K = [1, 2, 5, 10, 20, 50]

out = {"subtype": {}, "scale": {}, "sensitivity": {}, "answer": {}}
sub_res, scale_res, sens, ans_res = {}, {}, {}, {}
names = []
for arg in sys.argv[2:]:
    model, folder = arg.split("=", 1)
    full = pd.read_csv(f"{folder}/elements.csv", usecols=lambda c: c in COLS + ["mean_lp"])
    full = full.dropna(subset=["nli_e", "selfcheck", "ctx_u"])
    full["types"] = full["types"].fillna("")
    for task, d in full[full.kind == "edge"].groupby("task_type"):
        name = (model, task)
        names.append(name)
        ids = pd.factorize(d.response_id)[0]
        nr = ids.max() + 1
        Wt = np.stack([np.bincount(rng.integers(0, nr, nr), minlength=nr) for _ in range(NB)])
        y = d.wrong.values.astype(int)
        ms, _ = measures(d)
        ms["logprob"] = -d.mean_lp.values
        for st in SUBTYPES:  # clean links vs links of this hallucination type only
            m = (y == 0) | d.types.str.contains(st).values
            for key, col in list(METHODS.items()) + [("logprob", "logprob")]:
                sub_res[(name, st, key)] = auc_boot(np.asarray(ms[col], float)[m], y[m], ids[m], Wt)
        for key, col in METHODS.items():
            scale_res[(name, key)] = auc_boot(np.asarray(ms[col], float), y, ids, Wt)
        for k in GRID_K:
            for ksc in GRID_K:
                s, _ = measures(d, strength=float(k), n_samples=ksc)
                sens[(name, k, ksc)] = auc_boot(np.asarray(s[OURS], float), y, ids, Wt[:1])[0]
    rs = pd.read_csv(f"{folder}/responses.csv")
    for task, r in rs.groupby("task_type"):
        ids = np.arange(len(r))
        Wt = np.stack([np.bincount(rng.integers(0, len(r), len(r)), minlength=len(r)) for _ in range(NB)])
        y = r.wrong.values.astype(int)
        for key, col in (("zfmn", "resp_edge_tri_fused_P_max"), ("selfcheck", "resp_selfcheck_nli"),
                         ("nli", "resp_nli_not_entailed"), ("logprob", "resp_neg_mean_logprob"),
                         ("zfmn_mean", "resp_edge_tri_fused_P_mean")):
            ans_res[((model, task), key)] = auc_boot(r[col].values.astype(float), y, ids, Wt)
    print(model, "done", file=sys.stderr)


def ci(b):
    return [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]


def pool(table, keys_a, keys_b=None):
    """Mean over datasets; datasets where any measure is undefined (no positives) are skipped."""
    ok = [i for i in range(len(keys_a)) if not np.isnan(table[keys_a[i]][0])
          and (keys_b is None or not np.isnan(table[keys_b[i]][0]))]
    pa = [table[keys_a[i]][0] for i in ok]
    ba = np.nanmean([table[keys_a[i]][1] for i in ok], axis=0)
    if keys_b is None:
        return {"auc": float(np.mean(pa)), "ci": ci(ba), "datasets": len(ok)}
    pb = [table[keys_b[i]][0] for i in ok]
    bb = np.nanmean([table[keys_b[i]][1] for i in ok], axis=0)
    return {"diff": float(np.mean(pa) - np.mean(pb)), "ci": ci(ba - bb), "datasets": len(ok)}


for st in SUBTYPES:
    out["subtype"][st] = {key: pool(sub_res, [(n, st, key) for n in names])
                          for key in list(METHODS) + ["logprob"]}
    out["subtype"][st]["zfmn_minus_selfcheck"] = pool(sub_res, [(n, st, "zfmn") for n in names],
                                                      [(n, st, "selfcheck") for n in names])
for model in dict.fromkeys(n[0] for n in names):
    g = [n for n in names if n[0] == model]
    out["scale"][model] = {key: pool(scale_res, [(n, key) for n in g]) for key in METHODS}
    out["scale"][model]["zfmn_minus_selfcheck"] = pool(scale_res, [(n, "zfmn") for n in g],
                                                       [(n, "selfcheck") for n in g])
out["sensitivity"] = {"k_nli": GRID_K, "k_selfcheck": GRID_K,
                      "auc": [[float(np.mean([sens[(n, k, ksc)] for n in names])) for ksc in GRID_K]
                              for k in GRID_K]}
an = list(dict.fromkeys(n for n, _ in ans_res))
for key in ("zfmn", "selfcheck", "nli", "logprob", "zfmn_mean"):
    out["answer"][key] = pool(ans_res, [(n, key) for n in an])
for b in ("selfcheck", "nli", "logprob"):
    out["answer"][f"zfmn_minus_{b}"] = pool(ans_res, [(n, "zfmn") for n in an], [(n, b) for n in an])
out["answer"]["zfmn_mean_minus_nli"] = pool(ans_res, [(n, "zfmn_mean") for n in an], [(n, "nli") for n in an])
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
