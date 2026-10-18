"""AUROC with a bootstrap that resamples whole responses (clustered)."""

import random


def auroc(scores, labels):
    """Mann-Whitney AUROC with average ranks for ties. None if one class is missing."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # ranks are 1-based
        rank_sum += avg_rank * sum(l for _, l in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def clustered_auroc(groups, n_boot=1000, seed=0):
    """groups: list of (scores, labels) per response. Returns (auroc, ci_low, ci_high)."""
    flat_s = [s for g in groups for s in g[0]]
    flat_l = [l for g in groups for l in g[1]]
    point = auroc(flat_s, flat_l)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        pick = [groups[rng.randrange(len(groups))] for _ in groups]
        a = auroc([s for g in pick for s in g[0]], [l for g in pick for l in g[1]])
        if a is not None:
            boots.append(a)
    boots.sort()
    if not boots:
        return point, None, None
    return point, boots[int(0.025 * len(boots))], boots[min(int(0.975 * len(boots)), len(boots) - 1)]
