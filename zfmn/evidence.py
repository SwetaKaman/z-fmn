"""Pool a target text and its resampled siblings into evidence, opinions and Z-numbers,
and map human hallucination spans onto the target's nodes and edges."""

import math

from .zops import ZNumber, from_evidence, reliability_tfn, tfn_from_values


def _overlaps(span, spans):
    s, e = span
    return any(s < le and e > ls for ls, le in spans)


def _occ_conf(occs, kind):
    vals = []
    for o in occs:
        if kind == "node":
            if o["stats"] is not None:
                vals.append(o["stats"]["conf"])
        elif o["path_conf"] is not None:
            vals.append(o["path_conf"])
    return sum(vals) / len(vals) if vals else None


def element_records(target, samples, spans, W=2.0, a=0.5):
    """One record per node and edge of the target TFMN.

    target, samples: TFMN dicts from network.build_tfmn (built with token stats).
    spans: human hallucination spans [(start, end)] in the target text.

    Evidence over T = 1 + len(samples) texts: r = sum of the element's confidence in each
    text where it appears (0 where absent), s = T - r.
    """
    T = 1 + len(samples)
    recs = []
    for kind, key_name in (("node", "nodes"), ("edge", "edges")):
        for key, occs in target[key_name].items():
            per_text = [_occ_conf(occs, kind)]
            present = 1
            for smp in samples:
                o = smp[key_name].get(key)
                per_text.append(_occ_conf(o, kind) if o else None)
                present += 1 if o else 0
            confs = [c if c is not None else 0.0 for c in per_text]
            r = sum(confs)
            op = from_evidence(r, T - r, W=W, a=a)
            op_freq = from_evidence(present, T - present, W=W, a=a)  # beta-only baseline
            if kind == "node":
                stats = [o["stats"] for o in occs if o["stats"] is not None]
                wrong = any(_overlaps(o["span"], spans) for o in occs)
                mean_lp = sum(s["mean_lp"] for s in stats) / len(stats) if stats else None
                min_lp = min((s["min_lp"] for s in stats), default=None)
                ent = sum(s["entropy"] for s in stats) / len(stats) if stats else None
            else:
                wrong = any(_overlaps(sp, spans) for o in occs for sp in o["spans"])
                pc = [o["path_conf"] for o in occs if o["path_conf"] is not None]
                mean_lp = min_lp = ent = None
                if pc:
                    mean_lp = sum(math.log(max(c, 1e-12)) for c in pc) / len(pc)
            # A: the element's confidence across texts (0 where absent) as (min, median, max)
            A = tfn_from_values(confs)
            z = ZNumber(A, reliability_tfn(op))
            recs.append({
                "kind": kind, "key": key if kind == "node" else "|".join(key),
                "wrong": int(wrong), "n_occ": len(occs),
                "target_conf": per_text[0], "mean_lp": mean_lp, "min_lp": min_lp,
                "entropy": ent, "freq": present / T,
                "sl_b": op.b, "sl_u": op.u, "sl_P": op.projected,
                "freq_P": op_freq.projected, "z_kang": z.kang().centroid,
            })
    return recs


# Scores where higher means "more likely hallucinated". None-safe.
SCORERS = {
    "neg_mean_logprob": lambda r: None if r["mean_lp"] is None else -r["mean_lp"],
    "neg_min_logprob": lambda r: None if r["min_lp"] is None else -r["min_lp"],
    "entropy": lambda r: r["entropy"],
    "beta_frequency_only": lambda r: 1 - r["freq_P"],
    "subjective_logic_P": lambda r: 1 - r["sl_P"],
    "z_fmn_hybrid": lambda r: 1 - r["z_kang"],
}
