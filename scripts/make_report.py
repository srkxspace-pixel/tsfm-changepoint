"""results/eval + results/tsfm_* -> paper tables (LaTeX + CSV) and figures (PDF + PNG).

Every number written here comes from a CSV in results/; nothing is typed in by hand.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cpd import ROOT, load_config  # noqa: E402
from cpd.data.loaders import load_processed, zscore  # noqa: E402
from cpd.detectors.peaks import peak_curve, scores_to_cps  # noqa: E402
from cpd.evaluation.metrics import score  # noqa: E402
from cpd.evaluation.stats import holm, paired_test  # noqa: E402

EVAL, TABLES, FIGS = ROOT / "results/eval", ROOT / "paper/tables", ROOT / "figures"
BASELINES = ["zero", "pelt", "binseg", "bocpd"]
TSFM = ["chronos2_residual", "timesfm25_residual", "chronos2_embedding"]
LABEL = {"zero": "Zero (no CPs)", "pelt": "PELT", "binseg": "BinSeg", "bocpd": "BOCPD",
         "chronos2_residual": "Chronos-2 residual", "timesfm25_residual": "TimesFM-2.5 residual",
         "chronos2_embedding": "Chronos-2 embedding"}
DATASETS = {"tcpd": "TCPD", "skab": "SKAB"}
# Validated categorical slots (dataviz reference palette, light): blue, orange, aqua
COLOR = {"chronos2_residual": "#2a78d6", "timesfm25_residual": "#eb6834", "chronos2_embedding": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "savefig.bbox": "tight"})


def latex_tabular(df, fmt):
    esc = lambda v: str(v).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")  # noqa: E731
    cell = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) else esc(v)  # noqa: E731
    lines = ["\\begin{tabular}{l" + "r" * (len(df.columns) - 1) + "}", "\\toprule",
             " & ".join(esc(c) for c in df.columns) + " \\\\", "\\midrule"]
    lines += [" & ".join(cell(v) for v in row) + " \\\\" for row in df.itertuples(index=False)]
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}\n"])


def save_table(df, name, caption, label, fmt="{:.3f}"):
    df.to_csv(EVAL / f"table_{name}.csv", index=False)
    body = latex_tabular(df, fmt)
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
           "\\resizebox{\\columnwidth}{!}{%\n" + body + "}\n\\end{table}\n")
    (TABLES / f"{name}.tex").write_text(tex)


def save_fig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def table_datasets(data, margins):
    rows = []
    for ds, name in DATASETS.items():
        sub = [s for s in data.values() if s["dataset"] == ds]
        n = np.array([s["n_obs"] for s in sub])
        cps = [np.mean([len(c) for c in s["annotations"].values()]) for s in sub]
        rows.append({"Dataset": name, "Series": len(sub), "Multivariate": sum(s["n_dim"] > 1 for s in sub),
                     "Length min": int(n.min()), "Length median": int(np.median(n)), "Length max": int(n.max()),
                     "Annotators/series": int(np.median([len(s["annotations"]) for s in sub])),
                     "Mean CPs/annotator": round(float(np.mean(cps)), 2), "Margin": margins[ds]})
    save_table(pd.DataFrame(rows), "datasets", "Datasets used (after TCPDBench's exclusions).", "tab:datasets",
               fmt="{:.2f}")


def table_main(ps, methods):
    rows = []
    for m in methods:
        r = {"Method": LABEL[m]}
        for ds, name in DATASETS.items():
            d = ps[(ps.method == m) & (ps.dataset == ds)]
            for setting in ("default", "oracle"):
                r[f"{name} {setting} F1"] = d[f"{setting}_f1"].mean()
                r[f"{name} {setting} Cov"] = d[f"{setting}_cover"].mean()
        rows.append(r)
    save_table(pd.DataFrame(rows), "main", "Mean F1 (margin 5) and segmentation covering over series. "
               "Default: fixed parameters; oracle: best of each method's grid per series.", "tab:main")


def table_tcpdbench(ps):
    """Reproduction check against TCPDBench's own per-series scores (univariate TCPD, default setting)."""
    scores = ROOT / "data/raw/TCPDBench/analysis/output/scores"
    pub = {k: json.loads((scores / f"default_{k}_scores.json").read_text()) for k in ("f1", "cover")}
    rows = []
    for m in ["zero", "binseg", "pelt", "bocpd"]:
        d = ps[(ps.method == m) & (ps.dataset == "tcpd") & (ps.dims == "uni")]
        names = [sid.split("/")[1] for sid in d.series_id]
        rows.append({"Method": LABEL[m], "Ours F1": d.default_f1.mean(),
                     "TCPDBench F1": np.mean([pub["f1"][n][m] for n in names]),
                     "Ours Cov": d.default_cover.mean(),
                     "TCPDBench Cov": np.mean([pub["cover"][n][m] for n in names])})
    save_table(pd.DataFrame(rows), "tcpdbench", "Reproduction of TCPDBench defaults on the 32 univariate "
               "TCPD series. PELT differs only by TCPDBench storing each change point one index early.",
               "tab:tcpdbench")


def table_stats(ps, methods_tsfm):
    rows = []
    for ds, name in DATASETS.items():
        d = ps[ps.dataset == ds]
        for setting in ("default", "oracle"):
            col = f"{setting}_f1"
            means = d[d.method.isin(BASELINES[1:])].groupby("method")[col].mean()
            best = means.idxmax()
            ref = d[d.method == best].set_index("series_id")[col]
            block = []
            for m in methods_tsfm:
                x = d[d.method == m].set_index("series_id")[col].reindex(ref.index)
                med, w, t, l, p = paired_test(x.values, ref.values)
                block.append({"Dataset": name, "Setting": setting, "Detector": LABEL[m], "vs": LABEL[best],
                              "Median diff": med, "W/T/L": f"{w}/{t}/{l}", "p": p})
            for r, padj in zip(block, holm([r["p"] for r in block])):
                r["p (Holm)"] = padj
            rows += block
    df = pd.DataFrame(rows)
    df.to_csv(EVAL / "stats.csv", index=False)
    save_table(df, "stats", "Paired Wilcoxon signed-rank tests of per-series F1 against the best classical "
               "baseline of each dataset and setting (Holm-corrected over the three TSFM detectors).", "tab:stats")


def table_runtime(ps, methods):
    rt = ps[ps.method.isin(methods) & (ps.method != "zero")]
    df = rt.groupby(["method", "dataset"]).runtime_s.mean().unstack()
    df = df.rename(columns=DATASETS).reset_index()
    df["method"] = df.method.map(LABEL)
    df.columns = ["Method"] + [f"{c} (s/series)" for c in df.columns[1:]]
    save_table(df, "runtime", "Mean runtime per series at default parameters. Classical methods: one CPU "
               "core; TSFM detectors: one Colab T4 GPU (model inference only).", "tab:runtime", fmt="{:.2f}")


def fig_box(ps, methods):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, (ds, name) in zip(axes, DATASETS.items()):
        vals = [ps[(ps.method == m) & (ps.dataset == ds)].default_f1.values for m in methods]
        bp = ax.boxplot(vals, widths=0.55, patch_artist=True, medianprops={"color": INK, "linewidth": 1.2},
                        flierprops={"markersize": 3, "markeredgecolor": MUTED})
        for patch, m in zip(bp["boxes"], methods):
            patch.set(facecolor=COLOR.get(m, "#d6d5d0"), edgecolor=MUTED, linewidth=0.8)
        ax.set_xticks(range(1, len(methods) + 1), [LABEL[m] for m in methods], rotation=35, ha="right")
        ax.set_title(f"{name} ({len(vals[0])} series)", color=INK)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Per-series F1 (default setting)")
    save_fig(fig, "fig3_f1_boxplot")


def fig_context(methods_tsfm):
    grid = pd.read_csv(ROOT / "results/tsfm_cp/grid.csv")
    det = load_config("detectors")
    grid["param"] = grid.config.map(lambda c: json.loads(c)["param"])
    grid["tau"] = grid.config.map(lambda c: json.loads(c)["tau"])
    g = grid[(grid.tau == det["peaks"]["default_tau"]) & grid.method.str.endswith("_residual")]
    mean = g.groupby(["dataset", "method", "param"]).f1.mean().reset_index()
    mean.to_csv(EVAL / "context_sensitivity.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    for ax, (ds, name) in zip(axes, DATASETS.items()):
        for m in [m for m in methods_tsfm if m.endswith("_residual")]:
            d = mean[(mean.dataset == ds) & (mean.method == m)]
            ax.plot(d.param, d.f1, marker="o", markersize=4, linewidth=2, color=COLOR[m], label=LABEL[m])
        ax.set_xscale("log", base=2)
        ax.set_xticks(det["residual"]["context_lengths"], det["residual"]["context_lengths"])
        ax.set_xlabel("Context length L")
        ax.set_title(name, color=INK)
    axes[0].set_ylabel(f"Mean F1 (tau = {det['peaks']['default_tau']})")
    axes[0].legend(frameon=False, loc="lower right")
    save_fig(fig, "fig4_context_length")


def robustness(data):
    """Two checks raised at the M5 review, computed from saved results only (no re-runs)."""
    det, pk = load_config("detectors"), load_config("detectors")["peaks"]
    # (1) oracle with a matched budget: classical grids cut to 40 evenly spaced configs (= TSFM residual grid)
    g = pd.concat([pd.read_csv(p) for p in sorted((ROOT / "results/classical").glob("grid*.csv"))])
    budget = len(det["residual"]["context_lengths"]) * len(pk["tau_grid"])
    rows = []
    for (sid, m), gg in g.groupby(["series_id", "method"]):
        sub = gg.iloc[np.linspace(0, len(gg) - 1, budget).round().astype(int)]
        rows.append(dict(dataset=sid.split("/")[0], method=m, oracle_f1_full=gg.f1.max(),
                         oracle_f1_matched=sub.f1.max(), n_configs_full=len(gg)))
    pd.DataFrame(rows).groupby(["dataset", "method"]).mean(numeric_only=True).round(4).reset_index()         .to_csv(EVAL / "oracle_matched_budget.csv", index=False)
    # (2) residual detector without scores in the first 32 steps (tiny-context artefact check)
    rows = []
    for m in [m for m in TSFM if m.endswith("_residual")]:
        for sid, s in data.items():
            sc = np.load(ROOT / f"results/tsfm_raw/{m}/{sid.replace('/', '__')}.npz")[f"p{det['residual']['default_context']}"]
            skip = sc.copy()
            skip[:32] = 0
            f = [score_f1(s, scores_to_cps(x, pk["default_tau"], pk["smooth"], pk["min_gap"])) for x in (sc, skip)]
            rows.append(dict(dataset=s["dataset"], method=m, f1_default=f[0], f1_skip_first32=f[1],
                             true_cp_in_first32=any(c < 32 for a in s["annotations"].values() for c in a)))
    pd.DataFrame(rows).groupby(["dataset", "method"]).mean().round(4).reset_index()         .to_csv(EVAL / "early_context_check.csv", index=False)


def score_f1(s, cps):
    margins = load_config("datasets")["margin"]
    return score(s["annotations"], cps, s["n_obs"], margins[s["dataset"]])[0]


def fig_example(data, series_ids):
    det, raw = load_config("detectors"), ROOT / "results/tsfm_raw/timesfm25_residual"
    pk, L = det["peaks"], det["residual"]["default_context"]
    fig, axes = plt.subplots(2, len(series_ids), figsize=(7.0, 3.2), sharex="col",
                             gridspec_kw={"height_ratios": [1.2, 1]})
    for j, sid in enumerate(series_ids):
        s = data[sid]
        z = zscore(s["values"])
        score = np.load(raw / f"{sid.replace('/', '__')}.npz")[f"p{L}"]
        cps = scores_to_cps(score, pk["default_tau"], pk["smooth"], pk["min_gap"])
        truth = sorted({c for a in s["annotations"].values() for c in a})
        top, bot = axes[0, j], axes[1, j]
        offsets = np.arange(z.shape[1]) * 5.0           # stack channels so each stays readable
        top.plot(z + offsets, color=INK, linewidth=0.6)
        top.set_yticks([] if z.shape[1] > 1 else top.get_yticks())
        rz = peak_curve(score, pk["smooth"])          # exactly the curve peak picking sees
        bot.plot(rz, color=COLOR["timesfm25_residual"], linewidth=1.0)
        bot.axhline(pk["default_tau"], color=MUTED, linewidth=0.8)
        if rz.max() > 20:
            bot.set_yscale("symlog", linthresh=10)
        for ax in (top, bot):
            for c in truth:
                ax.axvline(c, color=MUTED, linestyle="--", linewidth=0.8)
        bot.plot(cps, rz[cps], "v", color=INK, markersize=5, linestyle="none")
        top.set_title(sid + (f" ({z.shape[1]} channels, stacked)" if z.shape[1] > 1 else ""), color=INK)
        bot.set_xlabel("time step")
    axes[0, 0].set_ylabel("z-scored value")
    axes[1, 0].set_ylabel("score (robust z)")
    save_fig(fig, "fig2_example")


def main():
    for p in (TABLES, FIGS):
        p.mkdir(parents=True, exist_ok=True)
    data = load_processed(ROOT / load_config("datasets")["processed_dir"])
    ps = pd.read_csv(EVAL / "per_series.csv")
    have_tsfm = ps.method.isin(TSFM).any()
    methods = BASELINES + (TSFM if have_tsfm else [])
    table_datasets(data, load_config("datasets")["margin"])
    table_tcpdbench(ps)
    table_main(ps, methods)
    table_runtime(ps, methods)
    fig_box(ps, methods)
    if have_tsfm:
        table_stats(ps, TSFM)
        fig_context(TSFM)
        fig_example(data, ["tcpd/nile", "skab/valve1_0"])   # fixed before seeing any TSFM result
        robustness(data)
    print("tables:", sorted(p.name for p in TABLES.glob("*.tex")))
    print("figures:", sorted(p.name for p in FIGS.glob("*.png")))


if __name__ == "__main__":
    main()
