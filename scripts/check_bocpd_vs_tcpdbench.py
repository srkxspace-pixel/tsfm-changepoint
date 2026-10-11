"""Compare our BOCPD (default params) with TCPDBench's stored R/ocp change points."""
import json

import numpy as np

from cpd import ROOT
from cpd.baselines.bocpd import bocpd
from cpd.data.loaders import load_processed
from cpd.evaluation.metrics import clean_cps

data = load_processed(ROOT / "data/processed")
summ = ROOT / "data/raw/TCPDBench/analysis/output/summaries"
same = 0
for sid, s in data.items():
    if s["dataset"] != "tcpd":
        continue
    name = sid.split("/")[1]
    ref = json.loads((summ / f"summary_{name}.json").read_text())["results"]["default_bocpd"][0]["cplocations"]
    v = s["values"]
    z = (v - v.mean(0)) / v.std(0, ddof=1)
    ours = clean_cps(bocpd(z), len(v))
    same += ours == sorted(ref)
    if ours != sorted(ref):
        print(f"{name:20s} ours={ours} tcpdbench={sorted(ref)}")
print(f"identical on {same}/{sum(s['dataset'] == 'tcpd' for s in data.values())} TCPD series")
