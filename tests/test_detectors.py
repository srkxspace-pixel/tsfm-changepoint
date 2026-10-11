import numpy as np

from cpd.data.loaders import zscore
from cpd.detectors.peaks import scores_to_cps
from cpd.detectors.scores import embedding_scores, residual_scores


def step_series(seed=0, cp=150, n=300):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 2))
    x[cp:] += 5
    return zscore(x)


def naive_forecaster(contexts):
    """Stand-in for a TSFM: median = mean of the last 8 points, spread = 2 sd of the context."""
    last = np.array([c[:, -8:].mean(1) for c in contexts])
    sd = np.array([c.std(1) + 1e-3 for c in contexts])
    return last - 1.28 * sd, last, last + 1.28 * sd


def level_embedder(windows):
    """Stand-in for Chronos-2 embed(): normalise the whole window (like the real model),
    then one 2-d token per 16-step patch."""
    out = []
    for w in windows:
        w = (w - w.mean()) / (w.std() + 1e-8)
        patches = w.mean(0).reshape(-1, 16)
        out.append(np.stack([patches.mean(1), np.ones(len(patches))], 1))
    return np.array(out)


def test_residual_score_peaks_at_the_change():
    z = step_series()
    s = residual_scores(z, naive_forecaster, L=64, min_context=8, eps=1e-3)
    assert s.shape == (300,) and (s[:8] == 0).all()
    assert abs(int(np.argmax(s)) - 150) <= 1


def test_residual_uses_expanding_then_sliding_context():
    seen = []
    def spy(contexts):
        seen.extend(c.shape for c in contexts)
        n = len(contexts)
        return np.zeros((n, 1)), np.zeros((n, 1)), np.ones((n, 1))
    residual_scores(np.zeros((20, 1)), spy, L=10, min_context=3, eps=1e-3)
    assert seen[0] == (1, 3) and seen[6] == (1, 9) and seen[-1] == (1, 10)


def test_embedding_score_peaks_at_the_change():
    z = step_series()
    s = embedding_scores(z, level_embedder, w=32)
    assert (s[:32] == 0).all() and (s[-31:] == 0).all()
    assert abs(int(np.argmax(s)) - 150) <= 8


def test_short_series_gets_zero_scores():
    assert not embedding_scores(np.zeros((20, 1)), level_embedder, w=16).any()
    assert not residual_scores(np.zeros((5, 1)), naive_forecaster, 32, 8, 1e-3).any()


def test_peak_picking():
    s = np.zeros(200)
    s[50], s[52], s[120] = 10, 9, 8
    s += np.random.default_rng(0).normal(0, 0.1, 200)
    assert scores_to_cps(s, tau=3, smooth=1, min_gap=5) == [50, 120]
    assert scores_to_cps(np.zeros(50), tau=3, smooth=3, min_gap=5) == []
