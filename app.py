"""Streamlit demo: do zero-shot time-series foundation models detect change points?

    streamlit run app.py

Live tab: pick a series; PELT / BinSeg / BOCPD run live on CPU, the Chronos-2 and TimesFM-2.5
detectors re-use the scores saved by the Colab GPU run (adjust tau / context length instantly).
Results tab: the benchmark tables and figures from results/eval/. Every number comes from results/.
"""
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from cpd import ROOT, load_config
from cpd.data.loaders import load_processed, zscore
from cpd.demo import CLASSICAL_COLOR, COLOR, classical_defaults, score_rows, tsfm_saved

alt.data_transformers.disable_max_rows()
INK, MUTED, ANNOT = "#0b0b0b", "#52514e", "#8a8984"
EVAL, FIGS = ROOT / "results/eval", ROOT / "figures"
DATASETS = {"tcpd": "TCPD · real-world (36 series)", "skab": "SKAB · industrial testbed (34 files)"}

st.set_page_config(page_title="Zero-shot TSFM change points", page_icon="📈", layout="wide")


@st.cache_data
def series_data():
    return load_processed(ROOT / load_config("datasets")["processed_dir"])


@st.cache_data
def classical(series_id):
    return classical_defaults(series_data()[series_id]["values"])


@st.cache_data
def tsfm(series_id, tau, context, half_window):
    return tsfm_saved(series_id, tau=tau, context=context, half_window=half_window)


@st.cache_data
def csv(name):
    return pd.read_csv(EVAL / name)


def need(path, how):
    if not path.exists():
        st.error(f"Missing `{path.relative_to(ROOT)}`. {how}")
        st.stop()


need(ROOT / "data/processed/series.json", "Run `python scripts/download_data.py` and `python scripts/prepare_data.py` first.")
need(ROOT / "results/tsfm_raw/manifest.json", "Unzip the Colab output `tsfm_raw.zip` into `results/tsfm_raw/` (README, Step 7).")
data, det = series_data(), load_config("detectors")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Choose a series")
    ds = st.radio("Dataset", list(DATASETS), format_func=DATASETS.get)
    ids = [k for k, v in data.items() if v["dataset"] == ds]
    default = ids.index("tcpd/nile") if "tcpd/nile" in ids else 0
    sid = st.selectbox("Series", ids, index=default,
                       format_func=lambda k: f"{k.split('/')[1]}  ·  {data[k]['n_obs']} steps  ·  {data[k]['n_dim']} ch")
    st.header("TSFM detector settings")
    tau = st.select_slider("Threshold τ (robust z)", det["peaks"]["tau_grid"], value=det["peaks"]["default_tau"],
                           help="A score peak above τ is reported as a change point.")
    L = st.select_slider("Context length L (residual)", det["residual"]["context_lengths"],
                         value=det["residual"]["default_context"], help="History the forecaster sees.")
    w = st.select_slider("Half-window w (embedding)", det["embedding"]["half_windows"],
                         value=det["embedding"]["default_half_window"])
    st.caption("Defaults (τ=3, L=128, w=32) were fixed before any result was seen. "
               "Classical methods always use their TCPDBench default settings.")

st.title("Do zero-shot time-series foundation models detect change points?")
st.caption("Chronos-2 and TimesFM-2.5, used with no training, against PELT, Binary Segmentation and BOCPD "
           "on human-labelled change points (TCPD, SKAB).")
live, results_tab, about = st.tabs(["🔍 Live detection", "📊 Benchmark results", "ℹ️ About the project"])

# ------------------------------------------------------------------ live tab
with live:
    s = data[sid]
    annotated = sorted({c for a in s["annotations"].values() for c in a})
    votes = {c: sum(c in a for a in s["annotations"].values()) for c in annotated}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Length", f"{s['n_obs']} steps")
    c2.metric("Channels", s["n_dim"])
    c3.metric("Annotators", len(s["annotations"]))
    c4.metric("Annotated change points", len(annotated))

    with st.spinner("Running PELT, BinSeg and BOCPD live on CPU…"):
        res = {"Zero (no CPs)": dict(cps=[], runtime=0.0, where="reference")} | classical(sid)
    tsfm_res = tsfm(sid, tau, L, w)
    res |= tsfm_res
    order = list(res)
    x = alt.X("t:Q", title="time step", scale=alt.Scale(domain=[0, s["n_obs"] - 1], nice=False))
    rules = alt.Chart(pd.DataFrame({"t": annotated, "annotators": [votes[c] for c in annotated]})).mark_rule(
        color=ANNOT, strokeDash=[5, 3], strokeWidth=1.5).encode(
        x="t:Q", tooltip=[alt.Tooltip("t:Q", title="annotated change at"), alt.Tooltip("annotators:Q", title="marked by")])

    # 1. the series (channels stacked so each stays readable)
    z = zscore(s["values"])
    T, C = z.shape
    long = pd.DataFrame({"t": np.tile(np.arange(T), C), "channel": np.repeat([f"ch {c + 1}" for c in range(C)], T),
                         "value": z.T.ravel(), "y": (z - 5 * np.arange(C)).T.ravel()})
    series_chart = alt.Chart(long).mark_line(strokeWidth=1, color=INK).encode(
        x=x, y=alt.Y("y:Q", title="z-scored value" if z.shape[1] == 1 else "channels (stacked)",
                     axis=None if z.shape[1] > 1 else alt.Axis()),
        detail="channel:N", tooltip=["t:Q", "channel:N", alt.Tooltip("value:Q", format=".2f")])
    st.subheader(f"{sid}  —  dashed lines are the human-annotated change points")
    st.altair_chart((series_chart + rules).properties(height=230 if z.shape[1] == 1 else 300),
                    width="stretch")

    # 2. detections of every method
    det_df = pd.DataFrame([dict(method=m, t=c) for m in order for c in res[m]["cps"]], columns=["method", "t"])
    colors = [COLOR.get(m, CLASSICAL_COLOR) for m in order]
    ticks = alt.Chart(det_df).mark_tick(thickness=4, size=24, opacity=1).encode(
        x=x, y=alt.Y("method:N", title=None, scale=alt.Scale(domain=order),   # keep rows with no detections
                     axis=alt.Axis(labelLimit=220, labelFontSize=12)),
        color=alt.Color("method:N", scale=alt.Scale(domain=order, range=colors), legend=None),
        tooltip=[alt.Tooltip("method:N"), alt.Tooltip("t:Q", title="detected at")])
    st.markdown("**Detected change points** (hover a mark for its time step)")
    st.altair_chart((ticks + rules).properties(height=40 * len(order)), width="stretch")

    # 3. scores
    table = pd.DataFrame(score_rows(s, res))
    st.markdown("**Accuracy against the human labels** (F1: a detection counts if within 5 steps; "
                "Cover: overlap of predicted and true segments; best values highlighted)")
    st.dataframe(table.style.format({"F1": "{:.3f}", "Cover": "{:.3f}", "Runtime_s": "{:.2f}"})
                 .highlight_max(subset=["F1", "Cover"], color="#d7e8fb"),
                 hide_index=True, width="stretch",
                 column_config={"Detected": st.column_config.TextColumn("Detected at (time steps)", width="large"),
                                "Runtime_s": "Runtime (s)", "Where": st.column_config.TextColumn(width="medium")})
    st.markdown("**What the foundation model sees**")
    left, right = st.columns([1, 4])
    with left:
        pick = st.radio("Score curve of", list(tsfm_res))
        st.caption(f"Peaks above the grey line (τ = {tau}) become detections. Move τ and L in the sidebar "
                   "to see the detections change.")
    with right:
        curve = tsfm_res[pick]["curve"]
        cdf = pd.DataFrame({"t": np.arange(len(curve)), "score": curve})
        log = curve.max() > 20
        ychart = alt.Y("score:Q", title="score (robust z)", scale=alt.Scale(type="symlog") if log else alt.Scale())
        line = alt.Chart(cdf).mark_line(color=COLOR[pick], strokeWidth=1.5).encode(
            x=x, y=ychart, tooltip=["t:Q", alt.Tooltip("score:Q", format=".2f")])
        thr = alt.Chart(pd.DataFrame({"tau": [tau]})).mark_rule(color=MUTED).encode(y="tau:Q")
        peaks = alt.Chart(cdf[cdf.t.isin(tsfm_res[pick]["cps"])]).mark_point(
            shape="triangle-down", filled=True, size=70, color=INK).encode(x=x, y=ychart, tooltip=["t:Q"])
        st.altair_chart((line + thr + peaks + rules).properties(height=230), width="stretch")
        if log:
            st.caption("Symlog y-axis: one very large score early in the series (tiny context).")

# ------------------------------------------------------------------ results tab
with results_tab:
    need(EVAL / "table_main.csv", "Run `python scripts/evaluate.py` and `python scripts/make_report.py`.")
    st.subheader("Main result: mean over all series")
    st.markdown("**Default** = one fixed setting for every series (the fair zero-shot test). "
                "**Oracle** = best setting per series (an upper bound).")
    main = csv("table_main.csv")
    num = main.columns[1:]
    st.dataframe(main.style.format({c: "{:.3f}" for c in num}).highlight_max(subset=list(num), color="#d7e8fb"),
                 hide_index=True, width="stretch")
    a, b = st.columns(2)
    with a:
        st.image(str(FIGS / "fig3_f1_boxplot.png"), caption="Per-series F1, default setting")
    with b:
        st.image(str(FIGS / "fig4_context_length.png"), caption="Residual detectors: effect of context length L")
    st.subheader("Is the difference significant?")
    st.caption("Paired Wilcoxon signed-rank test of per-series F1 vs the best classical method, Holm-corrected. "
               "W/T/L = series where the TSFM detector wins / ties / loses.")
    stats = csv("stats.csv")
    pfmt = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"  # noqa: E731
    st.dataframe(stats.style.format({"Median diff": "{:.3f}", "p": pfmt, "p (Holm)": pfmt}), hide_index=True,
                 width="stretch", height=35 * (len(stats) + 1) + 3)
    c, d = st.columns(2)
    with c:
        st.subheader("Runtime per series")
        st.dataframe(csv("table_runtime.csv").round(2), hide_index=True, width="stretch")
        st.caption("Classical: one CPU core. TSFM: one Colab T4 GPU.")
    with d:
        st.subheader("Our baselines reproduce TCPDBench")
        st.dataframe(csv("table_tcpdbench.csv").round(3), hide_index=True, width="stretch")
        st.caption("PELT differs only because TCPDBench stores each change point one index early.")
    st.subheader("TCPD by series length (default F1)")
    br = csv("breakdown.csv")
    pv = br[br.dataset == "tcpd"].pivot(index="method", columns="length", values="default_f1_mean")
    st.dataframe(pv[["<100", "100-255", "256-511", ">=512"]].round(3), width="stretch")

# ------------------------------------------------------------------ about tab
with about:
    st.markdown("""
### Research question
Can pretrained time-series foundation models (TSFMs), used **zero-shot** with no training, locate change
points in real labelled data as accurately as established change point detection (CPD) methods?

### Why it matters (the gap)
Uray et al. (EUROCAST 2026, arXiv 2607.12454) observed that TimesFM's forecast error spikes when an
industrial signal changes, and named change point detection as future work, but never measured it.
This project measures it on human-labelled benchmarks, against classical CPD, using the standard TCPDBench protocol.

### Detectors
* **Forecast residual** (Chronos-2, TimesFM-2.5): forecast each next value from the past `L` values;
  score = |actual − median forecast| / (90% − 10% forecast quantile). A change makes the model surprised.
* **Embedding distance** (Chronos-2): embed the window `[t−w, t+w)` and compare the model's
  representation of its left and right halves.
* Both: robust z-score → smooth → peaks above `τ` are change points.
* **Baselines**: PELT, Binary Segmentation, BOCPD (verified to reproduce TCPDBench) and *Zero* (no change points).

### Pipeline
```
TCPD + SKAB ─► prepare_data ─► classical baselines (CPU) ──────────────┐
                           └─► Colab T4: Chronos-2 / TimesFM-2.5 scores ─► peak picking ─► evaluate ─► tables & figures
```
""")
    m = json.loads((ROOT / "results/tsfm_raw/manifest.json").read_text())
    st.markdown("### GPU run record (from `results/tsfm_raw/manifest.json`)")
    st.json({k: m[k] for k in ("models", "packages", "gpu", "series_written", "started", "finished")})
