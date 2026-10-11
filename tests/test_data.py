import numpy as np
import pytest

from cpd import ROOT, load_config
from cpd.data.loaders import check_series, load_processed, save_processed

PROCESSED = ROOT / load_config("datasets")["processed_dir"]


def test_round_trip(tmp_path):
    series = {"tcpd/toy": dict(dataset="tcpd", values=np.arange(12.0).reshape(6, 2),
                               annotations={"6": [0, 3], "7": []})}
    save_processed(series, tmp_path)
    back = load_processed(tmp_path)["tcpd/toy"]
    assert back["n_obs"] == 6 and back["n_dim"] == 2
    assert back["annotations"] == {"6": [0, 3], "7": []}
    np.testing.assert_array_equal(back["values"], series["tcpd/toy"]["values"])


def test_check_rejects_bad_labels():
    with pytest.raises(AssertionError):
        check_series("x", dict(values=np.zeros((5, 1)), annotations={"a": [5]}))
    with pytest.raises(AssertionError):
        check_series("x", dict(values=np.array([[np.nan]]), annotations={}))


@pytest.mark.skipif(not (PROCESSED / "series.json").exists(), reason="run scripts/prepare_data.py first")
def test_processed_matches_published_counts():
    """Counts verified at M0 against the TCPD paper and SKAB v0.9 (docs/M0_FINDINGS.md)."""
    data = load_processed(PROCESSED)
    tcpd = {k: v for k, v in data.items() if v["dataset"] == "tcpd"}
    skab = {k: v for k, v in data.items() if v["dataset"] == "skab"}
    assert len(tcpd) == 36 and sum(v["n_dim"] > 1 for v in tcpd.values()) == 4
    assert all(len(v["annotations"]) == 5 for v in tcpd.values())
    assert data["tcpd/nile"]["n_obs"] == 100
    assert len(skab) == 34 and all(v["n_dim"] == 8 for v in skab.values())
    assert sum(len(v["annotations"]["skab"]) for v in skab.values()) == 129
    for sid, s in data.items():
        check_series(sid, s)
