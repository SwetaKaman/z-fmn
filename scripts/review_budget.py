"""Review-budget analysis (Section 7 of the journal paper): share of hallucinated links found when only
the riskiest q% of links are checked, per method, averaged over the nine model x task datasets, with
bootstrap CIs (answers resampled within each dataset) for the gains of Z-FMN.

Usage:
  python scripts/review_budget.py data/ragtruth_fmn_elements.csv.gz results/review_budget.json
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

d = pd.read_csv(sys.argv[1])
e=d[d.element_type=='edge'].copy()
e=e.dropna(subset=['score_zfmn_full_P','score_nli_entail','score_selfcheck_nli'])
S={'zfmn':1-e.score_zfmn_full_P,'nli':1-e.score_nli_entail,'selfcheck':e.score_selfcheck_nli,'logprob':-e.score_mean_logprob}
for k,v in S.items(): e[k]=v.values
rng=np.random.default_rng(0)
out={}
for k in S:
    aucs=[];rec={b:[] for b in (0.05,0.10,0.20,0.30)};prec={b:[] for b in rec}
    for _,g in e.groupby(['model','task']):
        y=g.hallucinated.values; s=g[k].values+rng.uniform(0,1e-9,len(g))
        aucs.append(roc_auc_score(y,s))
        o=np.argsort(-s)
        for b in rec:
            n=int(round(b*len(g))); top=y[o[:n]]
            rec[b].append(top.sum()/y.sum()); prec[b].append(top.mean())
    out[k]={'auc':np.mean(aucs),**{f'rec{int(b*100)}':np.mean(rec[b]) for b in rec},**{f'prec{int(b*100)}':np.mean(prec[b]) for b in prec}}
print(pd.DataFrame(out).T.round(3))
json.dump(out,open(sys.argv[2],'w'),indent=1)

NB=1000
groups=[]
for _,g in e.groupby(['model','task']):
    ids=pd.factorize(g.response_id)[0]
    groups.append((ids,g.hallucinated.values,{k:g[k].values for k in S}))
def recall(y,s,w,b):
    o=np.argsort(-s); ww=w[o]; cum=np.cumsum(ww); n=b*ww.sum()
    sel=cum<=n
    return (ww[sel]*y[o][sel]).sum()/(w*y).sum()
res={}
for b in (0.10,0.20):
    for base in ('selfcheck','nli'):
        diffs=[]
        for it in range(NB):
            dd=[]
            for ids,y,sc in groups:
                nr=ids.max()+1; cnt=np.bincount(rng.integers(0,nr,nr),minlength=nr); w=cnt[ids].astype(float)
                dd.append(recall(y,sc['zfmn'],w,b)-recall(y,sc[base],w,b))
            diffs.append(np.mean(dd))
        res[f'rec{int(b*100)}_minus_{base}']=[out['zfmn'][f'rec{int(b*100)}']-out[base][f'rec{int(b*100)}'],float(np.percentile(diffs,2.5)),float(np.percentile(diffs,97.5))]
print(res)
out['diffs']=res
json.dump(out,open(sys.argv[2],'w'),indent=1)
