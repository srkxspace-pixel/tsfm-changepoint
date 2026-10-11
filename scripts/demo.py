"""Live demo for one series (command line). For the web version run: streamlit run app.py

    python scripts/demo.py                  # Nile river (classic change point example)
    python scripts/demo.py skab/valve1_0    # an industrial SKAB file
    python scripts/demo.py --list           # all 70 series ids
    python scripts/demo.py tcpd/nile --show # also open the figure window

Classical methods run live on CPU; TSFM detectors use the saved Colab scores. Prints each method's
detected change points with F1 / covering against the human labels, saves figures/demo_<series>.png,
then prints the overall benchmark table.
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd

from cpd import ROOT, load_config
from cpd.data.loaders import load_processed, zscore
from cpd.demo import CLASSICAL_COLOR, COLOR, classical_defaults, score_rows, tsfm_saved


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
    for i, name in enumerate(names):
        cps = results[name]["cps"]
        bot.plot(cps, [i] * len(cps), "|", markersize=14, markeredgewidth=2.2, color=COLOR.get(name, CLASSICAL_COLOR))
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
    print(f"Series {args.series}: {s['n_obs']} steps, {s['n_dim']} channel(s)")
    print("Human annotations (per annotator):", s["annotations"])

    results = ({"Zero (no CPs)": dict(cps=[], runtime=0.0, where="reference")}
               | classical_defaults(s["values"]) | tsfm_saved(args.series))
    pd.set_option("display.width", 200)
    table = pd.DataFrame(score_rows(s, results))
    table["Detected"] = table.Detected.map(lambda d: d if len(d) < 45 else d[:42] + "...")
    print("\n" + table.to_string(index=False))
    plot(args.series, s, results, args.show)

    main_table = ROOT / "results/eval/table_main.csv"
    if main_table.exists():
        print("\nOverall benchmark (mean over all series, from results/eval/table_main.csv):")
        print(pd.read_csv(main_table).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
