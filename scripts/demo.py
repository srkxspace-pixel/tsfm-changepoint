"""Live demo for one series: classical CPD runs live on CPU, TSFM detectors use the saved Colab scores.

    python scripts/demo.py                  # Nile river (classic change point example)
    python scripts/demo.py skab/valve1_0    # an industrial SKAB file
    python scripts/demo.py --list           # all 70 series ids
    python scripts/demo.py tcpd/nile --show # also open the figure window

Prints each method's detected change points with F1 / covering against the human labels,
saves figures/demo_<series>.png, then prints the overall benchmark table.
"""
import argparse
import time

import matplotlib
import numpy as np
import pandas as pd

from cpd import ROOT, load_config
from cpd.baselines.bocpd import bocpd
from cpd.baselines.classical import binseg, binseg_path, pelt
from cpd.data.loaders import load_processed, zscore
from cpd.detectors.peaks import scores_to_cps
from cpd.evaluation.metrics import clean_cps, score

TSFM = {"chronos2_residual": ("Chronos-2 residual", "residual", "#2a78d6"),
        "timesfm25_residual": ("TimesFM-2.5 residual", "residual", "#eb6834"),
        "chronos2_embedding": ("Chronos-2 embedding", "embedding", "#1baf7a")}


def classical_defaults(z):
    cfg, n = load_config("baselines"), len(z)
    p, b, o = cfg["pelt"]["default"], cfg["binseg"]["default"], cfg["bocpd"]["default"]
    runs = {"PELT": lambda: pelt(z, 3 * np.log(n), p["cost"], p["mbic"]),
            "BinSeg": lambda: binseg(binseg_path(z, b["cost"], b["mbic"]), 3 * np.log(n), b["max_cps"]),
            "BOCPD": lambda: clean_cps(bocpd(z, **o), n)}
    out = {}
    for name, fn in runs.items():
        t0 = time.perf_counter()
        out[name] = (fn(), time.perf_counter() - t0, "live, CPU")
    return out


def tsfm_saved(sid):
    det = load_config("detectors")
    pk = det["peaks"]
    out = {}
    for key, (name, kind, _) in TSFM.items():
        path = ROOT / "results/tsfm_raw" / key / f"{sid.replace('/', '__')}.npz"
        if not path.exists():
            print(f"  (no saved scores for {name}: {path} missing)")
            continue
        p = det[kind]["default_context" if kind == "residual" else "default_half_window"]
        arr = np.load(path)
        cps = scores_to_cps(arr[f"p{p}"], pk["default_tau"], pk["smooth"], pk["min_gap"])
        out[name] = (cps, float(arr[f"runtime_p{p}"]), "saved, Colab T4 GPU")
    return out


def plot(sid, s, results, show):
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = zscore(s["values"])
    truth = sorted({c for a in s["annotations"].values() for c in a})
    names = list(results)
    fig, (top, bot) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    top.plot(z + np.arange(z.shape[1]) * 5.0, color="#0b0b0b", linewidth=0.7)
    top.set_title(f"{sid}: {s['n_obs']} steps, {s['n_dim']} channel(s); dashed = human-annotated change points")
    top.set_yticks([])
    colors = {name: c for name, _, c in TSFM.values()}
    for i, name in enumerate(names):
        cps = results[name][0]
        bot.plot(cps, [i] * len(cps), "|", markersize=14, markeredgewidth=2.2, color=colors.get(name, "#52514e"))
    bot.set_yticks(range(len(names)), names)
    bot.set_ylim(-0.7, len(names) - 0.3)
    bot.set_xlabel("time step")
    bot.set_title("Detected change points (default settings)")
    for ax in (top, bot):
        for c in truth:
            ax.axvline(c, color="#52514e", linestyle="--", linewidth=0.9)
    fig.tight_layout()
    out = ROOT / "figures" / f"demo_{sid.replace('/', '__')}.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("series", nargs="?", default="tcpd/nile")
    ap.add_argument("--list", action="store_true", help="list all series ids and exit")
    ap.add_argument("--show", action="store_true", help="open the figure window")
    args = ap.parse_args()
    data = load_processed(ROOT / load_config("datasets")["processed_dir"])
    if args.list:
        for sid, s in data.items():
            print(f"{sid:26s} T={s['n_obs']:5d}  channels={s['n_dim']}")
        return
    if args.series not in data:
        raise SystemExit(f"unknown series {args.series!r}; see --list")
    s = data[args.series]
    margin = load_config("datasets")["margin"][s["dataset"]]
    truth = {a: c for a, c in s["annotations"].items()}
    print(f"Series {args.series}: {s['n_obs']} steps, {s['n_dim']} channel(s)")
    print("Human annotations (per annotator):", truth)

    zero = {"Zero (no CPs)": ([], 0.0, "reference")}
    results = zero | classical_defaults(zscore(s["values"])) | tsfm_saved(args.series)
    rows = []
    for name, (cps, runtime, where) in results.items():
        f1, cover = score(s["annotations"], cps, s["n_obs"], margin)
        shown = " ".join(map(str, cps[:10])) + (" ..." if len(cps) > 10 else "")
        rows.append(dict(Method=name, Detected=shown or "-", F1=round(f1, 3), Cover=round(cover, 3),
                         Runtime_s=round(runtime, 2), Where=where))
    pd.set_option("display.width", 200)
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    plot(args.series, s, results, args.show)

    main_table = ROOT / "results/eval/table_main.csv"
    if main_table.exists():
        print("\nOverall benchmark (mean over all series, from results/eval/table_main.csv):")
        print(pd.read_csv(main_table).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
