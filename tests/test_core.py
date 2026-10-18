"""Checks that need no language model: python -m pytest tests (or python tests/test_core.py)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from zfmn.metrics import auroc  # noqa: E402
from zfmn.network import build_tfmn  # noqa: E402
from zfmn.zops import Opinion, discount, from_evidence, fuse  # noqa: E402


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_opinions_sum_to_one():
    o1, o2 = from_evidence(8, 2), from_evidence(3, 5)
    for o in (o1, o2, discount(o1, o2), fuse(o1, o2)):
        assert close(o.b + o.d + o.u, 1.0)
    # discounting can only lower belief; fusion lowers uncertainty
    assert discount(o1, o2).b <= min(o1.b, o2.b)
    assert fuse(o1, o2).u < min(o1.u, o2.u)


def test_fusion_matches_pooled_evidence():
    # cumulative fusion of two evidence-based opinions equals the opinion from pooled evidence
    f = fuse(from_evidence(4, 1), from_evidence(2, 3))
    p = from_evidence(6, 4)
    assert close(f.b, p.b) and close(f.u, p.u)


def test_auroc():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5


def test_alignment_and_edges():
    text = "Photosynthesis converts light energy into chemical energy."
    # fake LM tokens: split every word in two halves with known log-probs
    toks, i = [], 0
    for w in text.split(" "):
        s = text.index(w, i)
        h = s + len(w) // 2
        toks += [{"start": s, "end": h, "logprob": -1.0, "entropy": 1.0},
                 {"start": h, "end": s + len(w), "logprob": -3.0, "entropy": 2.0}]
        i = s + len(w)
    net = build_tfmn(text, toks)
    assert "photosynthesis" in net["nodes"] and "light" in net["nodes"]
    st = net["nodes"]["photosynthesis"][0]["stats"]
    assert close(st["joint_lp"], -4.0) and close(st["mean_lp"], -2.0) and close(st["min_lp"], -3.0)
    assert any("photosynthesis" in k for k in net["edges"])
    for occs in net["edges"].values():
        assert all(o["path_conf"] is not None for o in occs)


def test_grounding():
    from zfmn.grounding import element_nli, flatten_source, fused_fields, nli_record
    from zfmn.network import get_nlp

    def fake_nli(pairs):  # entailed if the hypothesis words all occur in the premise, else contradicted
        out = []
        for prem, hyp in pairs:
            ok = all(w.strip(".").lower() in prem.lower() for w in hyp.split())
            out.append((0.9, 0.05, 0.05) if ok else (0.05, 0.05, 0.9))
        return out

    src = "The bridge opened in 1932. It is made of steel."
    resp = "The bridge opened in 1932. It is made of wood."
    rec = nli_record(resp, src, [src, "The bridge opened in 1932."], get_nlp(), fake_nli)
    assert len(rec["sents"]) == 2
    assert rec["src"][0][0] > 0.5 and rec["src"][1][2] > 0.5  # second sentence contradicted
    wood = resp.index("wood")
    en = element_nli([(wood, wood + 4)], rec)
    assert en["nli_c"] > 0.5 and en["selfcheck"] > 0.5
    f = fused_fields(0.8, 0.1, en["nli_e"], en["nli_c"])
    assert f["nli_fused_d"] > 0.3 and f["nli_fused_P"] < 0.65  # contradiction pulls a confident element down
    assert "name: Joe's." in flatten_source('{"name": "Joe\'s", "tags": []}')


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
