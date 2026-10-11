"""Shared score -> change point conversion (DESIGN D3)."""
import numpy as np
from scipy.signal import find_peaks

MAD_TO_SD = 1.4826


def peak_curve(s, smooth):
    """Robust z-score of the score curve, then a centred moving average of width `smooth`."""
    s = np.asarray(s, dtype=float)
    mad = np.median(np.abs(s - np.median(s)))
    z = (s - np.median(s)) / (MAD_TO_SD * mad + 1e-12)
    if smooth > 1:
        z = np.convolve(z, np.ones(smooth) / smooth, mode="same")
    return z


def scores_to_cps(s, tau, smooth, min_gap):
    """Peak indices of the robust, smoothed curve that rise above tau."""
    peaks, _ = find_peaks(peak_curve(s, smooth), height=tau, distance=min_gap)
    return peaks.tolist()
