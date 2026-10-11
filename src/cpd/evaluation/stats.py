"""Paired tests over series (DESIGN D6): Wilcoxon signed-rank + Holm correction."""
import numpy as np
from scipy.stats import wilcoxon


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, same order as the input."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def paired_test(a, b):
    """a vs b over the same series: (median diff, wins, ties, losses, two-sided Wilcoxon p)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    wins, ties, losses = int((d > 1e-12).sum()), int((np.abs(d) <= 1e-12).sum()), int((d < -1e-12).sum())
    p = 1.0 if wins + losses == 0 else float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    return float(np.median(d)), wins, ties, losses, p
