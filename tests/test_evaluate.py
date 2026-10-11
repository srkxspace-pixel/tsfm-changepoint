import sys

import numpy as np
import pandas as pd

from cpd import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import per_series  # noqa: E402


def test_default_and_oracle_selection():
    grid = pd.DataFrame([
        dict(series_id="tcpd/a", method="pelt", config="d", is_default=True, f1=0.5, cover=0.6, runtime_s=1.0),
        dict(series_id="tcpd/a", method="pelt", config="g1", is_default=False, f1=0.9, cover=0.4, runtime_s=2.0),
        dict(series_id="tcpd/a", method="pelt", config="g2", is_default=False, f1=0.2, cover=0.8, runtime_s=2.0),
    ])
    data = {"tcpd/a": dict(dataset="tcpd", n_obs=50, n_dim=1, annotations={"1": [], "2": [25]},
                           values=np.zeros((50, 1)))}
    df = per_series(grid, data).set_index("method")
    pelt = df.loc["pelt"]
    assert (pelt.default_f1, pelt.default_cover, pelt.runtime_s) == (0.5, 0.6, 1.0)
    assert (pelt.oracle_f1, pelt.oracle_cover) == (0.9, 0.8)   # maximised separately, as TCPDBench
    zero = df.loc["zero"]                                       # no detections: P = 1, R = (1 + 1/2) / 2
    assert abs(zero.default_f1 - 2 * 0.75 / 1.75) < 1e-9 and zero.dims == "uni" and zero.length == "<100"
