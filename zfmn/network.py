"""Textual forma mentis network (TFMN) construction with word-level confidence.

Mirrors the EmoAtlas construction (content words linked when they are within K hops in the
dependency tree) but keeps character offsets and dependency paths, which EmoAtlas does not
return and which the token-to-concept alignment needs.
"""

import math
from collections import defaultdict

import networkx as nx

CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}

_NLP = None


def get_nlp(model="en_core_web_sm"):
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load(model)
    return _NLP


def word_confidence(doc, token_stats):
    """Aggregate sub-word token stats to spaCy words (Technical Challenge 1, step 1).

    For each spaCy token returns: joint log-prob (sum), mean log-prob, min log-prob,
    mean entropy and conf = exp(mean log-prob), the geometric-mean token probability in [0, 1].
    """
    out = []
    j = 0
    n = len(token_stats)
    for w in doc:
        s, e = w.idx, w.idx + len(w.text)
        # LM offsets are sorted; skip tokens that end before this word
        while j < n and token_stats[j]["end"] <= s:
            j += 1
        k, lps, hs = j, [], []
        while k < n and token_stats[k]["start"] < e:
            if token_stats[k]["end"] > s:
                lps.append(token_stats[k]["logprob"])
                hs.append(token_stats[k]["entropy"])
            k += 1
        if lps:
            mean_lp = sum(lps) / len(lps)
            out.append({
                "joint_lp": sum(lps), "mean_lp": mean_lp, "min_lp": min(lps),
                "entropy": sum(hs) / len(hs), "conf": math.exp(mean_lp),
            })
        else:
            out.append(None)
    return out


def is_content(w, include_numbers=True):
    if w.is_stop or w.is_punct or w.is_space:
        return False
    if w.pos_ == "NUM" or w.like_num:
        return include_numbers
    return w.pos_ in CONTENT_POS and w.is_alpha


def node_key(w):
    return w.text if (w.like_num or w.pos_ == "NUM") else w.lemma_.lower()


def build_tfmn(text, token_stats=None, max_distance=3, include_numbers=True):
    """Build one text's TFMN.

    Returns {"nodes": {key: [occurrence]}, "edges": {(a, b): [occurrence]}} where a node
    occurrence holds its char span and word stats, and an edge occurrence holds the char spans
    of both endpoints and the confidence of the weakest word on its dependency path
    (Technical Challenge 1, step 3).
    """
    doc = get_nlp()(text)
    wstats = word_confidence(doc, token_stats) if token_stats is not None else [None] * len(doc)
    nodes, edges = defaultdict(list), defaultdict(list)
    for sent in doc.sents:
        g = nx.Graph()
        for w in sent:
            g.add_node(w.i)
            if w.head.i != w.i:
                g.add_edge(w.i, w.head.i)
        content = [w for w in sent if is_content(w, include_numbers)]
        for w in content:
            nodes[node_key(w)].append({
                "span": (w.idx, w.idx + len(w.text)), "stats": wstats[w.i],
            })
        idx = {w.i for w in content}
        for w in content:
            paths = nx.single_source_shortest_path(g, w.i, cutoff=max_distance)
            for v_i, path in paths.items():
                if v_i <= w.i or v_i not in idx:
                    continue
                v = doc[v_i]
                a, b = node_key(w), node_key(v)
                if a == b:
                    continue
                confs = [wstats[i]["conf"] for i in path if wstats[i] is not None]
                key = (a, b) if a < b else (b, a)
                edges[key].append({
                    "spans": [(w.idx, w.idx + len(w.text)), (v.idx, v.idx + len(v.text))],
                    "path_conf": min(confs) if confs else None,
                    "distance": len(path) - 1,
                })
    return {"nodes": dict(nodes), "edges": dict(edges)}
