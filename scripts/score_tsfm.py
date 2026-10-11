"""Colab scores (results/tsfm_raw/) -> change points via the shared peak picking (DESIGN D3).

Writes results/tsfm_cp/grid.csv with the same columns as results/classical/grid*.csv.
Refuses to run if the Colab output is missing or incomplete (no partial results).
"""
import csv
import itertools
import json
import sys

import numpy as np

from cpd import ROOT, load_config
from cpd.data.loaders import load_processed
from cpd.detectors.peaks import scores_to_cps
from cpd.evaluation.metrics import score

RAW, OUT = ROOT / "results/tsfm_raw", ROOT / "results/tsfm_cp/grid.csv"
FIELDS = ["series_id", "dataset", "method", "config", "is_default", "n_cps", "cps", "runtime_s", "f1", "cover"]
DETECTORS = {"chronos2_residual": "residual", "timesfm25_residual": "residual", "chronos2_embedding": "embedding"}


def main():
    det = load_config("detectors")
    margins = load_config("datasets")["margin"]
    data = load_processed(ROOT / load_config("datasets")["processed_dir"])
    manifest = RAW / "manifest.json"
    if not manifest.exists():
        sys.exit(f"{manifest} not found: run notebooks/colab/01_tsfm_scores.ipynb and unzip tsfm_raw.zip here")
    m = json.loads(manifest.read_text())
    missing = [f"{d}/{sid}" for d in DETECTORS for sid in data
               if not (RAW / d / f"{sid.replace('/', '__')}.npz").exists()]
    if missing or m["series_written"] != m["series_expected"]:
        sys.exit(f"incomplete Colab output ({len(missing)} files missing, e.g. {missing[:3]})")

    pk = det["peaks"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for d, kind in DETECTORS.items():
            cfg = det[kind]
            params = cfg["context_lengths"] if kind == "residual" else cfg["half_windows"]
            default = cfg["default_context"] if kind == "residual" else cfg["default_half_window"]
            for sid, s in data.items():
                arr = np.load(RAW / d / f"{sid.replace('/', '__')}.npz")
                for p, tau in itertools.product(params, pk["tau_grid"]):
                    cps = scores_to_cps(arr[f"p{p}"], tau, pk["smooth"], pk["min_gap"])
                    f1, cover = score(s["annotations"], cps, s["n_obs"], margins[s["dataset"]])
                    w.writerow(dict(series_id=sid, dataset=s["dataset"], method=d,
                                    config=json.dumps({"param": p, "tau": tau}),
                                    is_default=(p == default and tau == pk["default_tau"]),
                                    n_cps=len(cps), cps=" ".join(map(str, cps)),
                                    runtime_s=f"{float(arr[f'runtime_p{p}']):.4f}",
                                    f1=f"{f1:.6f}", cover=f"{cover:.6f}"))
            print(f"{d}: {len(data)} series x {len(params) * len(pk['tau_grid'])} configs")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
