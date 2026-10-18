"""RQ4: does epistemic reliability covary with the emotional content EmoAtlas measures?

Usage:
  python rq4_emotion_analysis.py Mistral-7B=runs/mistral Llama-2-7B=runs/llama7 ... > rq4_tables.md

Uses the node rows of each folder's elements.csv (EmoAtlas features exist for concepts only).

Model 1 (reliability). Linear mixed model, random intercept per answer:
  logit(P) ~ |valence| + emotion count + frame valence spread + Zipf word frequency + word length
             + log(1 + degree) + task + model
Model 2 (detection). Logistic regression with answer-clustered standard errors:
  hallucinated ~ (1 − P) × emotional + controls
  The interaction asks whether confidence detects hallucinations worse for emotionally loaded concepts.
Continuous predictors are z-scored, so coefficients are per standard deviation.
"""
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from wordfreq import zipf_frequency

warnings.filterwarnings("ignore")

frames = []
for arg in sys.argv[1:]:
    model, folder = arg.split("=", 1)
    d = pd.read_csv(f"{folder}/elements.csv")
    d = d[d["kind"] == "node"].copy()
    d["model"] = model
    frames.append(d)
d = pd.concat(frames, ignore_index=True)
d["answer"] = d["model"] + ":" + d["response_id"].astype(str)
P = d["sl_P"].clip(1e-3, 1 - 1e-3)
d["logit_P"] = np.log(P / (1 - P))
d["risk"] = 1 - d["sl_P"]
d["abs_valence"] = d["valence"].abs()
d["emotional"] = ((d["abs_valence"] > 0) | (d["n_emotions"] > 0)).astype(int)
d["zipf"] = [zipf_frequency(str(k), "en") for k in d["key"]]
d["length"] = d["key"].astype(str).str.len()
d["log_degree"] = np.log1p(d["degree"])
d = d.dropna(subset=["abs_valence", "n_emotions", "frame_valence_spread", "logit_P"])
for c in ("abs_valence", "n_emotions", "frame_valence_spread", "zipf", "length", "log_degree", "risk"):
    d[c + "_z"] = (d[c] - d[c].mean()) / (d[c].std() or 1)
multi = lambda c: f" + C({c})" if d[c].nunique() > 1 else ""  # noqa: E731
controls = "zipf_z + length_z + log_degree_z" + multi("task_type") + multi("model")

L = [f"Nodes: {len(d)} from {d['answer'].nunique()} answers; emotional nodes: {int(d['emotional'].sum())}; "
     f"hallucinated nodes: {int(d['wrong'].sum())}.\n"]

m1 = smf.mixedlm(f"logit_P ~ abs_valence_z + n_emotions_z + frame_valence_spread_z + {controls}",
                 d, groups=d["answer"]).fit(reml=True)
L.append("## RQ4 Model 1. Reliability (logit P) and emotional content, linear mixed model\n")
L.append("| Term | Coefficient (per SD) | 95% CI | p |")
L.append("| --- | --- | --- | --- |")
ci = m1.conf_int()
for t in ("abs_valence_z", "n_emotions_z", "frame_valence_spread_z", "zipf_z", "length_z", "log_degree_z"):
    L.append(f"| {t[:-2]} | {m1.params[t]:+.3f} | [{ci.loc[t, 0]:+.3f}, {ci.loc[t, 1]:+.3f}] | {m1.pvalues[t]:.3g} |")

m2 = smf.logit(f"wrong ~ risk_z * emotional + {controls}", d).fit(
    disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d["answer"])[0]})
L.append("\n## RQ4 Model 2. Does emotional content change how well confidence detects hallucination?\n")
L.append("Logistic regression, standard errors clustered by answer. Odds ratios per SD of (1 − P).\n")
L.append("| Term | Odds ratio | 95% CI | p |")
L.append("| --- | --- | --- | --- |")
ci = m2.conf_int()
for t in ("risk_z", "emotional", "risk_z:emotional"):
    L.append(f"| {t} | {np.exp(m2.params[t]):.3f} | [{np.exp(ci.loc[t, 0]):.3f}, {np.exp(ci.loc[t, 1]):.3f}] | {m2.pvalues[t]:.3g} |")

# descriptive check: AUROC of 1 − P on emotional vs neutral nodes
def auroc(s, y):
    r = pd.Series(s).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


L.append("\n| Nodes | AUROC of 1 − P | Hallucinated / total |")
L.append("| --- | --- | --- |")
for lab, g in (("emotional", d[d["emotional"] == 1]), ("neutral", d[d["emotional"] == 0])):
    L.append(f"| {lab} | {auroc(g['risk'].to_numpy(), g['wrong'].to_numpy()):.3f} | {int(g['wrong'].sum())} / {len(g)} |")
print("\n".join(L))
