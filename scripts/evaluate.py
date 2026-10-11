"""All grids -> results/eval/: per-series default/oracle scores and the summary tables (SPECS E1-E4)."""
import pandas as pd

from cpd import ROOT, load_config
from cpd.data.loaders import load_processed
from cpd.evaluation.metrics import score

EVAL = ROOT / "results/eval"
BASELINES = ["zero", "pelt", "binseg", "bocpd"]
TSFM = ["chronos2_residual", "timesfm25_residual", "chronos2_embedding"]
LENGTH_BINS, LENGTH_LABELS = [0, 100, 256, 512, 10**9], ["<100", "100-255", "256-511", ">=512"]


def load_grids():
    paths = sorted((ROOT / "results/classical").glob("grid*.csv"))
    tsfm = ROOT / "results/tsfm_cp/grid.csv"
    paths += [tsfm] if tsfm.exists() else []
    g = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return g.drop_duplicates(["series_id", "method", "config"])


def per_series(grid, data):
    """Default = the method's default config; oracle = per-series max over its grid (F1 and cover separately)."""
    rows = []
    for (sid, method), g in grid.groupby(["series_id", "method"]):
        d = g[g["is_default"]].iloc[0]
        rows.append(dict(series_id=sid, method=method, default_f1=d.f1, default_cover=d.cover,
                         oracle_f1=g.f1.max(), oracle_cover=g.cover.max(), runtime_s=d.runtime_s,
                         n_configs=len(g)))
    margins = load_config("datasets")["margin"]
    for sid, s in data.items():  # "zero": no change points at all (TCPDBench reference method)
        f1, cover = score(s["annotations"], [], s["n_obs"], margins[s["dataset"]])
        rows.append(dict(series_id=sid, method="zero", default_f1=f1, default_cover=cover,
                         oracle_f1=f1, oracle_cover=cover, runtime_s=0.0, n_configs=1))
    df = pd.DataFrame(rows)
    meta = pd.DataFrame([dict(series_id=k, dataset=v["dataset"], n_obs=v["n_obs"], n_dim=v["n_dim"])
                         for k, v in data.items()])
    df = df.merge(meta, on="series_id")
    df["dims"] = df.n_dim.map(lambda n: "uni" if n == 1 else "multi")
    df["length"] = pd.cut(df.n_obs, LENGTH_BINS, labels=LENGTH_LABELS, right=False).astype(str)
    return df


def summary(df, by):
    cols = ["default_f1", "default_cover", "oracle_f1", "oracle_cover"]
    out = df.groupby(by + ["method"])[cols].agg(["mean", "median"])
    out.columns = [f"{c}_{a}" for c, a in out.columns]
    return out.join(df.groupby(by + ["method"]).size().rename("n_series")).reset_index().round(4)


def main():
    data = load_processed(ROOT / load_config("datasets")["processed_dir"])
    df = per_series(load_grids(), data)
    complete = df.groupby("method").series_id.nunique()
    for m, n in complete.items():
        if n != len(data):
            print(f"warning: {m} has {n}/{len(data)} series (still running?)")
    EVAL.mkdir(parents=True, exist_ok=True)
    df.to_csv(EVAL / "per_series.csv", index=False)
    s = summary(df, ["dataset", "dims"])
    s[s.method.isin(BASELINES)].to_csv(EVAL / "baselines.csv", index=False)
    if df.method.isin(TSFM).any():
        s[s.method.isin(TSFM)].to_csv(EVAL / "tsfm.csv", index=False)
    summary(df, ["dataset", "length"]).to_csv(EVAL / "breakdown.csv", index=False)
    rt = df[df.method != "zero"].groupby(["dataset", "method"]).runtime_s.agg(["mean", "median", "max"])
    rt.round(4).reset_index().to_csv(EVAL / "runtime.csv", index=False)
    pd.set_option("display.width", 200)
    print(s[["dataset", "dims", "method", "n_series", "default_f1_mean", "default_cover_mean",
             "oracle_f1_mean", "oracle_cover_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
