"""Evaluation metrics: F1 with margin and segmentation covering.

Ported verbatim from TCPDBench analysis/scripts/metrics.py and summarize.py
(G.J.J. van den Burg, Copyright (c) 2020 The Alan Turing Institute, MIT License),
so our scores are directly comparable with the published TCPDBench results.
The greedy matching iterates Python sets, exactly as the original does.
"""


def clean_cps(locations, n_obs):
    """Keep predictions in [1, n_obs - 2], as TCPDBench does before scoring."""
    return sorted({x for x in locations if 1 <= x < n_obs - 1})


def true_positives(T, X, margin=5):
    """Compute true positives without double counting."""
    X = set(list(X))
    TP = set()
    for tau in T:
        close = [(abs(tau - x), x) for x in X if abs(tau - x) <= margin]
        close.sort()
        if not close:
            continue
        dist, xstar = close[0]
        TP.add(tau)
        X.remove(xstar)
    return TP


def f_measure(annotations, predictions, margin=5, alpha=0.5, return_PR=False):
    """F-measure against multiple annotators (dict annotator -> cp list); 0 counts as a cp."""
    Tks = {k + 1: set(annotations[uid]) for k, uid in enumerate(annotations)}
    for Tk in Tks.values():
        Tk.add(0)

    X = set(predictions)
    X.add(0)

    Tstar = set()
    for Tk in Tks.values():
        for tau in Tk:
            Tstar.add(tau)

    K = len(Tks)

    P = len(true_positives(Tstar, X, margin=margin)) / len(X)

    TPk = {k: true_positives(Tks[k], X, margin=margin) for k in Tks}
    R = 1 / K * sum(len(TPk[k]) / len(Tks[k]) for k in Tks)

    F = P * R / (alpha * R + (1 - alpha) * P)
    if return_PR:
        return F, P, R
    return F


def overlap(A, B):
    """Jaccard index of two sets."""
    return len(A.intersection(B)) / len(A.union(B))


def partition_from_cps(locations, n_obs):
    """Partition of [0, n_obs - 1] defined by change point locations."""
    T = n_obs
    partition = []
    current = set()

    all_cps = iter(sorted(set(locations)))
    cp = next(all_cps, None)
    for i in range(T):
        if i == cp:
            if current:
                partition.append(current)
            current = set()
            cp = next(all_cps, None)
        current.add(i)
    partition.append(current)
    return partition


def cover_single(S, Sprime):
    """Covering of segmentation S by Sprime (Arbelaez 2010, eq. 8)."""
    T = sum(map(len, Sprime))
    assert T == sum(map(len, S))
    C = 0
    for R in S:
        C += len(R) * max(overlap(R, Rprime) for Rprime in Sprime)
    C /= T
    return C


def covering(annotations, predictions, n_obs):
    """Average segmentation covering against the human annotations."""
    Ak = {
        k + 1: partition_from_cps(annotations[uid], n_obs)
        for k, uid in enumerate(annotations)
    }
    pX = partition_from_cps(predictions, n_obs)

    Cs = [cover_single(Ak[k], pX) for k in Ak]
    return sum(Cs) / len(Cs)


def score(annotations, predictions, n_obs, margin=5):
    """Clean the predictions, then return (f1, cover)."""
    cps = clean_cps(predictions, n_obs)
    return f_measure(annotations, cps, margin=margin), covering(annotations, cps, n_obs)
