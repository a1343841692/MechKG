"""Exact endpoint-matched sampler, extracted unchanged from trusted_pipeline/robustness.py."""
from collections import Counter
import numpy as np


def endpoint_matched(positive, forbidden, rng, max_restarts=20):
    """Random bipartite rewiring; preserve BOTH endpoint occurrence counts.

    Repair conflicts by swapping gene assignments. Never relax constraints.
    Sampling is a controlled matched design, not uniform over all feasible graphs.
    """
    positive = np.asarray(positive, dtype=np.int64)
    n = len(positive)
    if n < 2:
        raise ValueError('Exact matching requires at least two positive pairs')
    for restart in range(max_restarts):
        result = positive.copy()
        rng.shuffle(result[:, 1])
        counts = Counter(map(tuple, result))
        for sweep in range(30):
            invalid = [i for i, p in enumerate(result)
                       if tuple(p) in forbidden or counts[tuple(p)] > 1]
            if not invalid:
                for axis in (0, 1):
                    assert Counter(result[:, axis]) == Counter(positive[:, axis])
                assert len(counts) == n
                assert not set(counts) & forbidden
                return result
            changed = False
            for i in rng.permutation(invalid):
                old_i = tuple(result[i])
                if old_i not in forbidden and counts[old_i] == 1:
                    continue
                for j in rng.integers(n, size=min(2000, max(100, n))):
                    if i == j:
                        continue
                    old_j = tuple(result[j])
                    a = (old_i[0], old_j[1]); b = (old_j[0], old_i[1])
                    if a == b or a in forbidden or b in forbidden:
                        continue
                    if any(counts[p] - int(p == old_i) - int(p == old_j) > 0 for p in (a, b)):
                        continue
                    counts.subtract([old_i, old_j])
                    for p in (old_i, old_j):
                        if counts[p] == 0:
                            del counts[p]
                    counts.update([a, b])
                    result[i, 1], result[j, 1] = old_j[1], old_i[1]
                    changed = True
                    break
            if not changed:
                break
    raise ValueError('Unable to realize unique matched negatives; constraints were NOT relaxed')
