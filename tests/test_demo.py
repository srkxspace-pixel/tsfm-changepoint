import numpy as np

from cpd.demo import classical_defaults, score_rows, tsfm_saved


def test_demo_runs_classical_and_scores_rows():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0, 1, 100), rng.normal(4, 1, 100)])[:, None]
    res = classical_defaults(values)
    assert set(res) == {"PELT", "BinSeg", "BOCPD"}
    assert all(any(abs(c - 100) <= 1 for c in r["cps"]) for r in res.values())
    series = dict(dataset="tcpd", n_obs=200, annotations={"1": [100], "2": [101]})
    rows = score_rows(series, res)
    assert [r["Method"] for r in rows] == ["PELT", "BinSeg", "BOCPD"] and all(r["F1"] == 1.0 for r in rows)


def test_tsfm_saved_skips_unknown_series():
    assert tsfm_saved("tcpd/does_not_exist") == {}
