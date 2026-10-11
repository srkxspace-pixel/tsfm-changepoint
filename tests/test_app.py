import pytest

from cpd import ROOT

NEEDS = [ROOT / "data/processed/series.json", ROOT / "results/tsfm_raw/manifest.json", ROOT / "results/eval/table_main.csv"]


@pytest.mark.skipif(not all(p.exists() for p in NEEDS), reason="needs processed data and results")
def test_streamlit_app_runs_without_errors():
    """Smoke test of all three tabs plus two interactions (slider, dataset switch)."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600).run()
    assert not at.exception
    at.sidebar.select_slider[0].set_value(2.0).run()
    at.sidebar.radio[0].set_value("skab").run()
    assert not at.exception
    assert len(at.dataframe) == 6
