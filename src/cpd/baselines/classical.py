"""PELT and Binary Segmentation (ruptures) with a Gaussian cost.

cost="mean": sum of squared deviations (change in mean, unit variance; data is z-scored).
cost="meanvar": Gaussian negative log-likelihood (change in mean and variance).
mbic=True adds log(segment length) per segment; with pen = 3 log n this is the MBIC
penalty of R `changepoint`, and reproduces TCPDBench's default PELT/BinSeg exactly.
Change points are returned as the first index of each new segment (TCPD convention).
"""
import numpy as np
import ruptures as rpt
from ruptures.base import BaseCost

VAR_FLOOR = 1e-8  # meanvar cost on a constant segment


class GaussianCost(BaseCost):
    model = "gaussian"

    def __init__(self, cost="mean", mbic=False):
        self.cost, self.mbic = cost, mbic
        self.min_size = 1 if cost == "mean" else 2

    def fit(self, signal):
        s = self.signal = signal.reshape(len(signal), -1)
        zero = np.zeros((1, s.shape[1]))
        self.c1 = np.vstack([zero, np.cumsum(s, 0)])
        self.c2 = np.vstack([zero, np.cumsum(s ** 2, 0)])
        return self

    def error(self, start, end):
        n = end - start
        s1, s2 = self.c1[end] - self.c1[start], self.c2[end] - self.c2[start]
        sse = s2 - s1 ** 2 / n
        err = sse.sum() if self.cost == "mean" else n * np.log(np.maximum(sse / n, VAR_FLOOR)).sum()
        return err + (np.log(n) if self.mbic else 0.0)


def _cps(bkps):
    return [int(b) for b in bkps[:-1]]


def pelt(z, pen, cost="mean", mbic=False):
    c = GaussianCost(cost, mbic)
    return _cps(rpt.Pelt(custom_cost=c, min_size=c.min_size, jump=1).fit(z).predict(pen=pen))


def binseg_path(z, cost="mean", mbic=False):
    """Fitted Binseg; reuse it for several penalties (ruptures caches the splits)."""
    c = GaussianCost(cost, mbic)
    return rpt.Binseg(custom_cost=c, min_size=c.min_size, jump=1).fit(z)


def binseg(algo, pen, max_cps=None):
    bkps = algo.predict(pen=pen)
    if max_cps is not None and len(bkps) - 1 > max_cps:
        bkps = algo.predict(n_bkps=max_cps)
    return _cps(bkps)
