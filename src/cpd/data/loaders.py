"""Load TCPD and SKAB into one format: values (T, C) + change points per annotator."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_tcpd(raw_dir, exclude=()):
    raw_dir = Path(raw_dir)
    annotations = json.loads((raw_dir / "annotations.json").read_text())
    series = {}
    for name in sorted(annotations):
        if name in exclude:
            continue
        d = json.loads((raw_dir / "datasets" / name / f"{name}.json").read_text())
        values = np.array([s["raw"] for s in d["series"]], dtype=np.float64).T
        series[f"tcpd/{name}"] = dict(
            dataset="tcpd", values=values,
            annotations={a: sorted(cps) for a, cps in annotations[name].items()})
    return series


def load_skab(raw_dir, groups, sep=";"):
    series = {}
    for group in groups:
        for path in sorted((Path(raw_dir) / group).glob("*.csv"), key=lambda p: int(p.stem)):
            df = pd.read_csv(path, sep=sep, index_col="datetime")
            cps = np.flatnonzero(df.pop("changepoint").to_numpy() == 1).tolist()
            df.pop("anomaly")
            series[f"skab/{group}_{path.stem}"] = dict(
                dataset="skab", values=df.to_numpy(np.float64), annotations={"skab": cps})
    return series


def check_series(series_id, s):
    """Label checks: finite (T, C) values, every change point inside [0, T)."""
    v = s["values"]
    assert v.ndim == 2 and len(v) > 0, f"{series_id}: bad shape {v.shape}"
    assert np.isfinite(v).all(), f"{series_id}: non-finite values"
    for annotator, cps in s["annotations"].items():
        assert all(0 <= c < len(v) for c in cps), f"{series_id}: annotator {annotator} has a cp outside [0, {len(v)})"


def save_processed(series, out_dir):
    out_dir = Path(out_dir)
    index = {}
    for sid, s in series.items():
        path = out_dir / f"{sid}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, values=s["values"])
        index[sid] = dict(dataset=s["dataset"], n_obs=len(s["values"]),
                          n_dim=s["values"].shape[1], annotations=s["annotations"])
    (out_dir / "series.json").write_text(json.dumps(index, indent=1))


def load_processed(processed_dir):
    """Return {series_id: {dataset, n_obs, n_dim, annotations, values}}."""
    processed_dir = Path(processed_dir)
    index = json.loads((processed_dir / "series.json").read_text())
    return {sid: dict(meta, values=np.load(processed_dir / f"{sid}.npz")["values"])
            for sid, meta in index.items()}
