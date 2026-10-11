"""Run PELT, BinSeg and BOCPD (default + oracle grid) on every series.

Writes results/classical/grid*.csv: one row per (series, method, config), with the
change points, runtime and F1/covering. Resumable: finished (series, method) pairs are skipped.
Parallel: `run_classical.py <shard> <n_shards>` handles every n-th series into grid_<shard>.csv.
"""
import csv
import sys
import itertools
import json
import time

import numpy as np

from cpd import ROOT, load_config
from cpd.baselines.bocpd import bocpd
from cpd.baselines.classical import binseg, binseg_path, pelt
from cpd.data.loaders import load_processed, zscore
from cpd.evaluation.metrics import score

OUT_DIR = ROOT / "results/classical"
FIELDS = ["series_id", "dataset", "method", "config", "is_default", "n_cps", "cps", "runtime_s", "f1", "cover"]


def pen_value(pen, n):
    return 3 * np.log(n) if pen == "mbic" else float(pen)


def timed(fn, *args, **kw):
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    return out, time.perf_counter() - t0


def runs_pelt(z, cfg):
    n = len(z)
    d = cfg["pelt"]["default"]
    yield True, d, *timed(pelt, z, pen_value(d["pen"], n), d["cost"], d["mbic"])
    g = cfg["segmentation_grid"]
    for cost, pen in itertools.product(g["costs"], np.logspace(**g["pen_log10"])):
        yield False, dict(cost=cost, mbic=False, pen=float(pen)), *timed(pelt, z, pen, cost)


def runs_binseg(z, cfg):
    n = len(z)
    d = cfg["binseg"]["default"]
    algo, fit_t = timed(binseg_path, z, d["cost"], d["mbic"])
    cps, t = timed(binseg, algo, pen_value(d["pen"], n), d["max_cps"])
    yield True, d, cps, fit_t + t
    g = cfg["segmentation_grid"]
    for cost in g["costs"]:
        algo, fit_t = timed(binseg_path, z, cost)
        for pen, q in itertools.product(np.logspace(**g["pen_log10"]), g["binseg_max_cps"]):
            cps, t = timed(binseg, algo, pen, q)
            yield False, dict(cost=cost, mbic=False, pen=float(pen), max_cps=q), cps, fit_t + t


def runs_bocpd(z, cfg):
    d = cfg["bocpd"]["default"]
    yield True, d, *timed(bocpd, z, **d)
    g = cfg["bocpd"]["grid"]
    for lam, a, b, k in itertools.product(g["lam"], g["prior"], g["prior"], g["prior"]):
        c = dict(d, lam=lam, alpha0=a, beta0=b, kappa0=k)
        yield False, c, *timed(bocpd, z, **c)


METHODS = {"pelt": runs_pelt, "binseg": runs_binseg, "bocpd": runs_bocpd}


def main():
    cfg = load_config("baselines")
    margins = load_config("datasets")["margin"]
    data = load_processed(ROOT / load_config("datasets")["processed_dir"])
    shard, n_shards = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (0, 1)
    out = OUT_DIR / (f"grid_{shard}.csv" if n_shards > 1 else "grid.csv")
    done = set()
    for path in OUT_DIR.glob("grid*.csv"):
        with path.open() as f:
            done |= {(r["series_id"], r["method"]) for r in csv.DictReader(f) if r["cover"]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for sid, s in list(data.items())[shard::n_shards]:
            z = zscore(s["values"])
            for method, runs in METHODS.items():
                if (sid, method) in done:
                    continue
                t0 = time.perf_counter()
                rows = []
                for is_default, config, cps, runtime in runs(z, cfg):
                    f1, cover = score(s["annotations"], cps, s["n_obs"], margins[s["dataset"]])
                    rows.append(dict(series_id=sid, dataset=s["dataset"], method=method,
                                     config=json.dumps(config), is_default=is_default, n_cps=len(cps),
                                     cps=" ".join(map(str, cps)), runtime_s=f"{runtime:.4f}",
                                     f1=f"{f1:.6f}", cover=f"{cover:.6f}"))
                w.writerows(rows)
                f.flush()
                print(f"{sid:28s} {method:7s} {len(rows):4d} configs {time.perf_counter() - t0:7.1f}s", flush=True)


if __name__ == "__main__":
    main()
