import json

import numpy as np
import pytest

from cpd import ROOT
from cpd.baselines.bocpd import bocpd
from cpd.baselines.classical import binseg, binseg_path, pelt
from cpd.data.loaders import load_processed, zscore
from cpd.evaluation.metrics import clean_cps

SUMMARIES = ROOT / "data/raw/TCPDBench/analysis/output/summaries"
PROCESSED = ROOT / "data/processed"


def step_series(seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0, 1, 100), rng.normal(4, 1, 100)])[:, None]


def test_baselines_find_an_obvious_mean_shift():
    z = zscore(step_series())
    pen = 3 * np.log(len(z))
    assert pelt(z, pen, mbic=True) == [100]
    assert binseg(binseg_path(z, mbic=True), pen, max_cps=5) == [100]
    cps = clean_cps(bocpd(z), len(z))
    assert len(cps) == 1 and abs(cps[0] - 100) <= 1


def test_zscore_keeps_constant_channel_finite():
    z = zscore(np.column_stack([np.arange(5.0), np.ones(5)]))
    assert np.isfinite(z).all() and (z[:, 1] == 0).all()


@pytest.mark.skipif(not SUMMARIES.exists() or not (PROCESSED / "series.json").exists(),
                    reason="needs TCPDBench summaries and processed data")
@pytest.mark.parametrize("name", ["nile", "brent_spot", "gdp_iran", "businv", "bank"])
def test_defaults_reproduce_tcpdbench(name):
    """Default PELT/BinSeg/BOCPD give TCPDBench's stored change points (R changepoint stores cp - 1)."""
    s = load_processed(PROCESSED)[f"tcpd/{name}"]
    ref = json.loads((SUMMARIES / f"summary_{name}.json").read_text())["results"]
    z, n = zscore(s["values"]), s["n_obs"]
    pen = 3 * np.log(n)
    stored = {m: sorted(ref[f"default_{m}"][0]["cplocations"]) for m in ("pelt", "binseg", "bocpd")}
    assert clean_cps([c - 1 for c in pelt(z, pen, mbic=True)], n) == stored["pelt"]
    assert clean_cps([c - 1 for c in binseg(binseg_path(z, mbic=True), pen, 5)], n) == stored["binseg"]
    assert clean_cps(bocpd(z), n) == stored["bocpd"]
