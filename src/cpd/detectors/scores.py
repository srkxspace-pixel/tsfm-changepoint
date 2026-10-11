"""Model-agnostic TSFM scores (DESIGN D1, D2). The model is passed in as a function,
so this file runs on CPU in tests and on GPU in the Colab notebook unchanged."""
import numpy as np


def residual_scores(z, forecast_fn, L, min_context, eps):
    """One-step forecast surprise per time step.

    z: (T, C) z-scored series. forecast_fn(list of (C, l) contexts) -> (q10, q50, q90), each (n, C).
    s_t = mean_c |z_tc - q50| / max(q90 - q10, eps), using context z[max(0, t - L):t].
    """
    T = len(z)
    s = np.zeros(T)
    ts = np.arange(min_context, T)
    if len(ts) == 0:
        return s
    contexts = [z[max(0, t - L):t].T.astype(np.float32) for t in ts]
    q10, q50, q90 = (np.asarray(q, dtype=float) for q in forecast_fn(contexts))
    s[ts] = (np.abs(z[ts] - q50) / np.maximum(q90 - q10, eps)).mean(axis=1)
    return s


def embedding_scores(z, embed_fn, w):
    """Split-window embedding distance per time step.

    embed_fn(list of (C, 2w) windows) -> (n, P, d) patch-token embeddings in time order
    (already averaged over channels). For the window [t - w, t + w), the score is
    1 - cos(mean of left-half tokens, mean of right-half tokens). Both halves share one
    normalisation, so level and scale shifts stay visible.
    """
    T = len(z)
    s = np.zeros(T)
    ts = np.arange(w, T - w + 1)
    if len(ts) == 0:
        return s
    tokens = np.asarray(embed_fn([z[t - w:t + w].T.astype(np.float32) for t in ts]), dtype=float)
    half = tokens.shape[1] // 2
    left, right = tokens[:, :half].mean(1), tokens[:, half:].mean(1)
    cos = (left * right).sum(1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-12)
    s[ts] = 1 - cos
    return s
