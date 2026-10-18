"""Hybrid Z-numbers: fuzzy value A plus a reliability B derived from a subjective-logic opinion.

B is combined along paths with subjective-logic discounting and across sources with cumulative
fusion; A uses triangular fuzzy numbers. Kang et al. (2012) converts Z = (A, B) to a single
fuzzy number for ranking.
"""

import math
from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class Opinion:
    b: float  # belief
    d: float  # disbelief
    u: float  # uncertainty
    a: float = 0.5  # base rate

    @property
    def projected(self):
        return self.b + self.a * self.u


def from_evidence(r, s, W=2.0, a=0.5):
    """Opinion from positive evidence r and negative evidence s (beta-distribution mapping)."""
    r, s = max(r, 0.0), max(s, 0.0)
    k = r + s + W
    return Opinion(r / k, s / k, W / k, a)


def discount(o1, o2):
    """Trust discounting along i -> j -> k: reliability can only drop along a path."""
    return Opinion(o1.b * o2.b, o1.b * o2.d, o1.d + o1.u + o1.b * o2.u, o2.a)


def fuse(o1, o2):
    """Cumulative fusion of two independent opinions about the same element."""
    den = o1.u + o2.u - o1.u * o2.u
    if den == 0:  # both dogmatic
        return Opinion((o1.b + o2.b) / 2, (o1.d + o2.d) / 2, 0.0, (o1.a + o2.a) / 2)
    b = (o1.b * o2.u + o2.b * o1.u) / den
    u = o1.u * o2.u / den
    den_a = o1.u + o2.u - 2 * o1.u * o2.u
    a = (
        (o1.a * o2.u + o2.a * o1.u - (o1.a + o2.a) * o1.u * o2.u) / den_a
        if den_a != 0 else (o1.a + o2.a) / 2
    )
    return Opinion(b, 1 - b - u, u, a)


@dataclass(frozen=True)
class TFN:
    """Triangular fuzzy number (l, m, r)."""
    l: float
    m: float
    r: float

    @property
    def centroid(self):
        return (self.l + self.m + self.r) / 3

    def scale(self, k):
        return TFN(self.l * k, self.m * k, self.r * k)


def tfn_from_values(values):
    """(min, median, max) of observed values."""
    v = sorted(values)
    n = len(v)
    med = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
    return TFN(v[0], med, v[-1])


def reliability_tfn(o):
    """B as a fuzzy number: support [belief, plausibility], peak at the projected probability."""
    return TFN(o.b, o.projected, o.b + o.u)


@dataclass(frozen=True)
class ZNumber:
    A: TFN
    B: TFN

    def kang(self):
        """Kang et al. (2012): weight A by sqrt of B's centroid to get one fuzzy number."""
        return self.A.scale(math.sqrt(max(self.B.centroid, 0.0)))


# ---- Z-weighted network measures ---------------------------------------------------------


def to_graph(nodes, edges):
    """nodes: {key: (ZNumber, Opinion)}, edges: {(a, b): (ZNumber, Opinion)} -> nx.Graph."""
    g = nx.Graph()
    for k, (z, o) in nodes.items():
        g.add_node(k, z=z, op=o)
    for (a, b), (z, o) in edges.items():
        g.add_edge(a, b, z=z, op=o,
                   rel_cost=-math.log(max(o.b, 1e-9)),
                   dist=max(1.0 - z.kang().centroid, 1e-6))
    return g


def z_strength(g, n):
    """Sum of edge A (defuzzified) and the fused opinion of those edges."""
    strength, op = 0.0, None
    for _, _, data in g.edges(n, data=True):
        strength += data["z"].A.centroid
        op = data["op"] if op is None else fuse(op, data["op"])
    return strength, op


def most_reliable_path(g, src, dst):
    """Path maximising product of beliefs (Dijkstra on -log b) and its discounted opinion."""
    path = nx.dijkstra_path(g, src, dst, weight="rel_cost")
    op = None
    for a, b in zip(path, path[1:]):
        e = g.edges[a, b]["op"]
        op = e if op is None else discount(op, e)
    return path, op


def z_closeness(g):
    """Harmonic closeness on distance 1 - Kang-defuzzified edge Z."""
    return nx.harmonic_centrality(g, distance="dist")


def fragile_core(g, central_q=0.75, reliable_q=0.25):
    """Nodes that are central (top strength quantile) but weakly held (bottom reliability quantile)."""
    if g.number_of_nodes() == 0:
        return []
    strength = {n: z_strength(g, n)[0] for n in g}
    rel = {n: g.nodes[n]["op"].projected for n in g}

    def q(vals, p):
        v = sorted(vals)
        return v[min(int(p * len(v)), len(v) - 1)]

    s_cut, r_cut = q(strength.values(), central_q), q(rel.values(), reliable_q)
    return [n for n in g if strength[n] >= s_cut and rel[n] <= r_cut]
