"""Network-aware evidence and propagation for Z-FMN.

Three additions on top of evidence.py:

1. Context-conditioned evidence. A resample that does not mention a concept is not automatically
   evidence against it. If the resample talks about the same context (the concept's network
   neighbours) but leaves the concept out, that is disbelief. If it never touches that context,
   it is silence, which adds to uncertainty only. Subjective logic keeps the two apart; plain
   frequency cannot.
2. Relational propagation. A concept's reliability is fused with the support it receives from
   its neighbours through the links that connect them (trust discounting along each link,
   averaging fusion across links). Hallucinated material tends to form small, weakly supported
   clusters, which this exposes.
3. Emotional value. The fuzzy value A of a concept is the valence of its semantic frame (the
   mean EmoAtlas valence of its neighbours) across the answer and its resamples.
"""

from collections import defaultdict

from .zops import Opinion, TFN, ZNumber, discount, from_evidence, reliability_tfn, tfn_from_values

# ---- lexicons --------------------------------------------------------------------------------

_VALENCE = None
_EMOTIONS = None


def load_emoatlas_lexicons():
    """Valence and Plutchik emotion lexicons shipped with EmoAtlas (pip install emoatlas --no-deps)."""
    global _VALENCE, _EMOTIONS
    if _VALENCE is None:
        import json
        import os
        import runpy
        import importlib.util

        root = importlib.util.find_spec("emoatlas").submodule_search_locations[0]
        v = runpy.run_path(os.path.join(root, "valence", "english.py"))
        _VALENCE = {w: 1.0 for w in v["_positive"]}
        _VALENCE.update({w: -1.0 for w in v["_negative"]})
        with open(os.path.join(root, "langs", "english.json")) as f:
            _EMOTIONS = json.load(f)
    return _VALENCE, _EMOTIONS


def valence(word):
    val, _ = load_emoatlas_lexicons()
    return val.get(word, 0.0)


def emotions(word):
    _, emo = load_emoatlas_lexicons()
    return emo.get(word, [])


# ---- helpers ---------------------------------------------------------------------------------


def _conf(occs, kind):
    vals = []
    for o in occs:
        c = o["stats"]["conf"] if kind == "node" and o["stats"] else o.get("path_conf")
        if c is not None:
            vals.append(c)
    return sum(vals) / len(vals) if vals else 0.0


def neighbours(net):
    nb = defaultdict(set)
    for a, b in net["edges"]:
        nb[a].add(b)
        nb[b].add(a)
    return nb


def average_fuse(ops):
    """Averaging fusion (Josang) of several opinions: for dependent sources such as links."""
    ops = [o for o in ops if o is not None]
    if not ops:
        return None
    if any(o.u == 0 for o in ops):
        dog = [o for o in ops if o.u == 0]
        b = sum(o.b for o in dog) / len(dog)
        return Opinion(b, 1 - b, 0.0, dog[0].a)
    inv = sum(1 / o.u for o in ops)
    b = sum(o.b / o.u for o in ops) / inv
    u = len(ops) / inv
    return Opinion(b, 1 - b - u, u, sum(o.a for o in ops) / len(ops))


def _overlaps(span, spans):
    s, e = span
    return any(s < le and e > ls for ls, le in spans)


# ---- context-conditioned opinions --------------------------------------------------------------


def basic_opinions(target, samples, W=2.0, a=0.5):
    """Opinions where every resample that omits the element counts against it."""
    T = 1 + len(samples)
    node_ops, edge_ops = {}, {}
    for key_name, kind, ops in (("nodes", "node", node_ops), ("edges", "edge", edge_ops)):
        for k, occs in target[key_name].items():
            r = _conf(occs, kind) + sum(
                _conf(s[key_name][k], kind) for s in samples if k in s[key_name])
            ops[k] = from_evidence(r, T - r, W=W, a=a)
    return node_ops, edge_ops


def context_opinions(target, samples, W=2.0, a=0.5):
    """Opinions for every node and edge of the target using context-conditioned evidence."""
    nb = neighbours(target)
    node_ops, edge_ops = {}, {}
    for v, occs in target["nodes"].items():
        r, s = _conf(occs, "node"), 0.0
        ctx = nb.get(v, set())
        for smp in samples:
            if v in smp["nodes"]:
                r += _conf(smp["nodes"][v], "node")
            elif ctx:
                s += sum(1 for u in ctx if u in smp["nodes"]) / len(ctx)
            # no context at all: silence, adds nothing
        node_ops[v] = from_evidence(r, s, W=W, a=a)
    for (x, y), occs in target["edges"].items():
        r, s = _conf(occs, "edge"), 0.0
        for smp in samples:
            if (x, y) in smp["edges"]:
                r += _conf(smp["edges"][(x, y)], "edge")
            else:
                s += 0.5 * ((x in smp["nodes"]) + (y in smp["nodes"]))
        edge_ops[(x, y)] = from_evidence(r, s, W=W, a=a)
    return node_ops, edge_ops


def propagate(node_ops, edge_ops, nb):
    """Fuse each node's own opinion with relational support from its neighbours.

    Support from neighbour u through link e = discount(op(e), op(u)): u vouches for v only as far
    as the link between them is itself reliable. Link supports are combined by averaging fusion,
    then fused with the node's own opinion. Edges get the discounted support of both endpoints.
    """
    new_nodes = {}
    for v, own in node_ops.items():
        sup = []
        for u in nb.get(v, ()):
            e = edge_ops.get((v, u) if v < u else (u, v))
            if e is not None and u in node_ops:
                sup.append(discount(e, node_ops[u]))
        support = average_fuse(sup)
        new_nodes[v] = own if support is None else average_fuse([own, support])
    new_edges = {}
    for (x, y), own in edge_ops.items():
        ends = average_fuse([node_ops.get(x), node_ops.get(y)])
        new_edges[(x, y)] = own if ends is None else average_fuse([own, discount(ends, own)])
    return new_nodes, new_edges


def source_opinions(target, source, W=2.0, a=0.5, strength=2.0):
    """External evidence from the source network (the document the model was given).

    Node in the source: belief. Node absent from the source: disbelief.
    Edge in the source network: belief. Both endpoints in the source but never linked there:
    disbelief (the answer relates two source concepts in a way the source does not). Endpoints
    missing from the source: silence (uncertainty), already covered by the node opinions.
    `strength` is how many resamples one source observation is worth.
    """
    node_ops, edge_ops = {}, {}
    for v in target["nodes"]:
        hit = v in source["nodes"]
        node_ops[v] = from_evidence(strength * hit, strength * (not hit), W=W, a=a)
    for (x, y) in target["edges"]:
        if (x, y) in source["edges"]:
            r, s = strength, 0.0
        elif x in source["nodes"] and y in source["nodes"]:
            r, s = 0.0, strength
        else:
            r, s = 0.0, 0.0
        edge_ops[(x, y)] = from_evidence(r, s, W=W, a=a)
    return node_ops, edge_ops


# ---- records -----------------------------------------------------------------------------------


def _src_fields(src_op, model_op, ctx_op):
    """Source opinion alone and fused with the model's own opinion (cumulative fusion)."""
    from .zops import fuse

    if src_op is None:
        return {}
    f = fuse(model_op, src_op)
    fc = fuse(ctx_op, src_op)
    return {"src_P": src_op.projected, "src_d": src_op.d, "src_u": src_op.u,
            "fused_P": f.projected, "fused_d": f.d, "fused_u": f.u,
            "fused_ctx_P": fc.projected, "fused_ctx_d": fc.d, "fused_ctx_u": fc.u}


def frame_valence(net, v, nb):
    vals = [valence(u) for u in nb.get(v, ())]
    return sum(vals) / len(vals) if vals else 0.0


def _types(spans_list, spans, span_types):
    """RAGTruth hallucination types touching any of the given char spans."""
    if not span_types:
        return []
    return sorted({t for (ls, le), t in zip(spans, span_types)
                   for s, e in spans_list if s < le and e > ls})


def advanced_records(target, samples, spans, W=2.0, a=0.5, span_types=None, source=None):
    nb_t = neighbours(target)
    node_ops, edge_ops = context_opinions(target, samples, W, a)
    p_nodes, p_edges = propagate(node_ops, edge_ops, nb_t)
    b_nodes, b_edges = basic_opinions(target, samples, W, a)
    pb_nodes, pb_edges = propagate(b_nodes, b_edges, nb_t)
    if source is not None:
        s_nodes, s_edges = source_opinions(target, source, W, a)
    else:
        s_nodes = s_edges = {}
    nbs = [neighbours(s) for s in samples]
    recs = []
    for v, occs in target["nodes"].items():
        # A: frame valence across the answer and the resamples that mention v
        fv = [frame_valence(target, v, nb_t)] + [
            frame_valence(s, v, nb) for s, nb in zip(samples, nbs) if v in s["nodes"]]
        A = tfn_from_values(fv)
        z = ZNumber(A, reliability_tfn(p_nodes[v]))
        recs.append({
            "kind": "node", "key": v, "wrong": int(any(_overlaps(o["span"], spans) for o in occs)),
            "types": _types([o["span"] for o in occs], spans, span_types),
            "ctx_P": node_ops[v].projected, "ctx_u": node_ops[v].u, "ctx_d": node_ops[v].d,
            "ctx_b": node_ops[v].b, "prop_P": p_nodes[v].projected,
            "basic_prop_P": pb_nodes[v].projected,
            **_src_fields(s_nodes.get(v), b_nodes[v], node_ops[v]),
            "valence": valence(v), "n_emotions": len(emotions(v)),
            "frame_valence": A.centroid, "frame_valence_spread": A.r - A.l,
            "degree": len(nb_t.get(v, ())),
            "z": [A.l, A.m, A.r, z.B.l, z.B.m, z.B.r],
        })
    for (x, y), occs in target["edges"].items():
        recs.append({
            "kind": "edge", "key": f"{x}|{y}",
            "wrong": int(any(_overlaps(sp, spans) for o in occs for sp in o["spans"])),
            "types": _types([sp for o in occs for sp in o["spans"]], spans, span_types),
            "ctx_P": edge_ops[(x, y)].projected, "ctx_u": edge_ops[(x, y)].u,
            "ctx_d": edge_ops[(x, y)].d, "ctx_b": edge_ops[(x, y)].b,
            "prop_P": p_edges[(x, y)].projected, "basic_prop_P": pb_edges[(x, y)].projected,
            **_src_fields(s_edges.get((x, y)), b_edges[(x, y)], edge_ops[(x, y)]),
        })
    return recs


ADVANCED_SCORERS = {
    "context_conditioned_P": lambda r: 1 - r["ctx_P"],
    "network_propagated_P": lambda r: 1 - r["prop_P"],
}
