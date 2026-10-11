"""Raw TCPD + SKAB -> data/processed/ (unified format), with label checks and a summary."""
from cpd import ROOT, load_config
from cpd.data.loaders import check_series, load_skab, load_tcpd, save_processed

cfg = load_config("datasets")
series = load_tcpd(ROOT / cfg["tcpd"]["raw_dir"], exclude=cfg["tcpd"]["exclude"])
series |= load_skab(ROOT / cfg["skab"]["raw_dir"], cfg["skab"]["groups"], cfg["skab"]["sep"])
for sid, s in series.items():
    check_series(sid, s)
save_processed(series, ROOT / cfg["processed_dir"])

for name in ("tcpd", "skab"):
    sub = [s for s in series.values() if s["dataset"] == name]
    lengths = sorted(len(s["values"]) for s in sub)
    multi = sum(s["values"].shape[1] > 1 for s in sub)
    n_cps = sum(len(c) for s in sub for c in s["annotations"].values())
    print(f"{name}: {len(sub)} series ({multi} multivariate), length {lengths[0]}-{lengths[-1]}, "
          f"{n_cps} annotated change points (all annotators)")
print("excluded:", {k: v for k, v in cfg["tcpd"]["exclude"].items()})
