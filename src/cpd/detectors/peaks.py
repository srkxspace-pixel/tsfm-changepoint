"""Shared score -> change point conversion (DESIGN D3)."""
import numpy as np
from scipy.signal import find_peaks

MAD_TO_SD = 1.4826


def scores_to_cps(s, tau, smooth, min_gap):
    """Robust z-score the curve, smooth it, and return peak indices above tau."""
    s = np.asarray(s, dtype=float)
    mad = np.median(np.abs(s - np.median(s)))
    z = (s - np.median(s)) / (MAD_TO_SD * mad + 1e-12)
    if smooth > 1:
        z = np.convolve(z, np.ones(smooth) / smooth, mode="same")
    peaks, _ = find_peaks(z, height=tau, distance=min_gap)
    return peaks.tolist()
