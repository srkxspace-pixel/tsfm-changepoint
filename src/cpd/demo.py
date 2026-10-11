"""Run every method on one series. Shared by scripts/demo.py (CLI) and app.py (Streamlit).

Classical methods run live with their TCPDBench default settings; TSFM detectors reuse the
scores saved by the Colab notebook (results/tsfm_raw/), so no GPU is needed.
"""
import time

import numpy as np

from cpd import ROOT, load_config
from cpd.baselines.bocpd import bocpd
from cpd.baselines.classical import binseg, binseg_path, pelt
from cpd.data.loaders import zscore
from cpd.detectors.peaks import peak_curve, scores_to_cps
from cpd.evaluation.metrics import clean_cps, score

# key -> (label, kind); colours are the validated categorical slots used in the paper figures
TSFM = {"chronos2_residual": ("Chronos-2 residual", "residual"),
        "timesfm25_residual": ("TimesFM-2.5 residual", "residual"),
        "chronos2_embedding": ("Chronos-2 embedding", "embedding")}
COLOR = {"Chronos-2 residual": "#2a78d6", "TimesFM-2.5 residual": "#eb6834",
         "Chronos-2 embedding": "#1baf7a"}
CLASSICAL_COLOR = "#52514e"


def classical_defaults(values):
    """PELT, BinSeg and BOCPD at their default settings, timed. name -> {cps, runtime, where}."""
    z, n = zscore(values), len(values)
    cfg = load_config("baselines")
    p, b, o = cfg["pelt"]["default"], cfg["binseg"]["default"], cfg["bocpd"]["default"]
    pen = 3 * np.log(n)  # MBIC, as in configs/baselines.yaml ("pen: mbic")
    runs = {"PELT": lambda: pelt(z, pen, p["cost"], p["mbic"]),
            "BinSeg": lambda: binseg(binseg_path(z, b["cost"], b["mbic"]), pen, b["max_cps"]),
            "BOCPD": lambda: clean_cps(bocpd(z, **o), n)}
    out = {}
    for name, fn in runs.items():
        t0 = time.perf_counter()
        cps = fn()
        out[name] = dict(cps=cps, runtime=time.perf_counter() - t0, where="live · CPU")
    return out


def tsfm_saved(series_id, tau=None, context=None, half_window=None):
    """Saved Colab scores -> change points. name -> {cps, runtime, where, curve, param}.

    tau / context / half_window default to the values fixed in configs/detectors.yaml.
    `curve` is exactly what peak picking thresholds (robust z, smoothed)."""
    det = load_config("detectors")
    pk = det["peaks"]
    tau = pk["default_tau"] if tau is None else tau
    params = {"residual": context or det["residual"]["default_context"],
              "embedding": half_window or det["embedding"]["default_half_window"]}
    out = {}
    for key, (label, kind) in TSFM.items():
        path = ROOT / "results/tsfm_raw" / key / f"{series_id.replace('/', '__')}.npz"
        if not path.exists():
            continue
        arr = np.load(path)
        p = params[kind]
        out[label] = dict(cps=scores_to_cps(arr[f"p{p}"], tau, pk["smooth"], pk["min_gap"]),
                          runtime=float(arr[f"runtime_p{p}"]), where="saved · Colab T4 GPU",
                          curve=peak_curve(arr[f"p{p}"], pk["smooth"]), param=p)
    return out


def score_rows(series, results):
    """One row per method: detected change points, F1, covering, runtime."""
    margin = load_config("datasets")["margin"][series["dataset"]]
    rows = []
    for name, r in results.items():
        f1, cover = score(series["annotations"], r["cps"], series["n_obs"], margin)
        rows.append(dict(Method=name, Detected=" ".join(map(str, r["cps"])) or "–", F1=round(f1, 3),
                         Cover=round(cover, 3), Runtime_s=round(r["runtime"], 2), Where=r["where"]))
    return rows
