import json

import pytest

from cpd import ROOT
from cpd.evaluation.metrics import clean_cps, covering, f_measure, partition_from_cps, true_positives

SUMMARIES = ROOT / "data/raw/TCPDBench/analysis/output/summaries"


def test_hand_computed_examples():
    """Doctest values from TCPDBench metrics.py, checked by hand."""
    assert true_positives({1, 10, 20, 23}, {3, 8, 20}) == {1, 10, 20}
    assert f_measure({1: [10, 20], 2: [11, 20], 3: [10], 4: [0, 5]}, [10, 20]) == 1.0
    assert f_measure({1: [], 2: [10], 3: [50]}, [10]) == pytest.approx(0.9090909090909091)
    # no predictions: P = 1 (only index 0), R = (1 + 1/2 + 1/2) / 3 = 2/3 -> F1 = 0.8
    assert f_measure({1: [], 2: [10], 3: [50]}, []) == pytest.approx(0.8)
    assert partition_from_cps([3, 5], 8) == [{0, 1, 2}, {3, 4}, {5, 6, 7}]
    assert covering({1: [10, 20], 2: [10], 3: [0, 5]}, [10, 20], 45) == pytest.approx(0.7962962962962963)
    assert covering({1: [], 2: [10], 3: [40]}, [], 45) == pytest.approx(0.8189300411522634)
    assert clean_cps([0, 1, 5, 98, 99, 5], 100) == [1, 5, 98]


@pytest.mark.skipif(not SUMMARIES.exists(), reason="TCPDBench summaries not downloaded")
def test_reproduces_tcpdbench_stored_scores():
    """Our port must give TCPDBench's stored F1/cover for every default run of every method."""
    checked = 0
    for path in sorted(SUMMARIES.glob("summary_*.json")):
        d = json.loads(path.read_text())
        for name, runs in d["results"].items():
            if not name.startswith("default_"):
                continue
            for run in runs or []:
                if run["status"] != "SUCCESS":
                    continue
                cps = clean_cps(run["cplocations"], d["dataset_nobs"])
                assert f_measure(d["annotations"], cps) == pytest.approx(run["scores"]["f1"]), (path.name, name)
                assert covering(d["annotations"], cps, d["dataset_nobs"]) == pytest.approx(run["scores"]["cover"]), (path.name, name)
                checked += 1
    assert checked > 400
