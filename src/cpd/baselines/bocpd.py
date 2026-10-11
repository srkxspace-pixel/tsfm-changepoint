"""Bayesian online change point detection (Adams & MacKay 2007).

Gaussian likelihood with unknown mean and variance (Normal-Inverse-Gamma prior),
constant hazard 1/lam, channels treated as independent. The recursion, run-length
truncation and MAP change-point path follow the R package `ocp` (onlineCPD +
findCPprobs) exactly as TCPDBench calls it, so results are comparable.
"""
import numpy as np
from scipy.special import gammaln

TINY = np.finfo(float).tiny  # R's .Machine$double.xmin


def bocpd(x, lam=100.0, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0, trunc=1e-4):
    """x: (T, C) z-scored series. Returns 0-based change points (segment starts, 0 included)."""
    x = np.asarray(x, dtype=float)
    T, C = x.shape
    H = 1.0 / lam
    R = np.ones(1)  # run-length probabilities before the current point
    mu, kappa = np.full((1, C), mu0), np.full(1, kappa0)
    alpha, beta = np.full(1, alpha0), np.full((1, C), beta0)
    # MAP path over boundaries (1-based positions as in ocp): best[p], cps[p]
    best = np.zeros(T + 2)
    cps = [[], [1]]
    for t in range(1, T + 1):
        if len(R) > 1:  # drop run lengths whose tail mass is below trunc
            keep = int(np.sum(np.cumsum(R[::-1])[::-1] >= trunc))
            R, mu, kappa, alpha, beta = R[:keep], mu[:keep], kappa[:keep], alpha[:keep], beta[:keep]
        xt = x[t - 1]
        nu = 2 * alpha[:, None]
        var = beta * (kappa[:, None] + 1) / (alpha[:, None] * kappa[:, None])
        logpred = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * var)
                   - (nu + 1) / 2 * np.log1p((xt - mu) ** 2 / (nu * var))).sum(axis=1)
        p = np.exp(np.maximum(np.log(R) + logpred, np.log(TINY)))
        newR = np.concatenate([[np.sum(p * H)], p * (1 - H)])
        newR /= newR.sum()
        newR = np.maximum(newR, TINY)
        # posterior update; run length 0 restarts from the prior
        beta = np.vstack([np.full((1, C), beta0), beta + kappa[:, None] * (xt - mu) ** 2 / (2 * (kappa[:, None] + 1))])
        mu = np.vstack([np.full((1, C), mu0), (kappa[:, None] * mu + xt) / (kappa[:, None] + 1)])
        kappa = np.concatenate([[kappa0], kappa + 1])
        alpha = np.concatenate([[alpha0], alpha + 0.5])
        # findCPprobs: candidate start p = t + 1 - r for run length r; ties -> smallest p
        L = len(newR) - 1
        starts = np.arange(t + 1 - L, t + 2)
        scores = np.log(newR[::-1]) + best[starts - 1]
        k = int(np.argmax(scores))
        best[t + 1] = scores[k]
        cps.append(cps[starts[k] - 1] + [int(starts[k])])
        R = newR
    return sorted({c - 1 for c in cps[-1]})
