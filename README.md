# Do Zero-Shot Time-Series Foundation Models Detect Change Points?

**Forecast-Residual and Embedding-Distance Detectors vs. Classical CPD on Real-World and Industrial Sensor Data**

*B.Tech final-year project · Track B (Research) · AI & Machine Learning*

---

## 1. What this project is, in plain words

A **change point** is a moment when a signal suddenly behaves differently. Examples are a river's flow dropping after a dam is built, or a pump's sensors changing when a valve fails. **Change point detection (CPD)** means finding those moments automatically.

**Time-series foundation models (TSFMs)** such as **Chronos-2** (Amazon) and **TimesFM-2.5** (Google) are large pretrained models that can forecast almost any time series without any training. This is called **zero-shot** use. A 2026 study (Uray et al., arXiv 2607.12454) noticed that a TSFM's forecast error jumps when a signal changes, and suggested using TSFMs for CPD. Nobody had tested this properly.

**This project tests it.** It answers one question:

> Can pretrained foundation models, used zero-shot, find change points in real labelled data as accurately as established CPD methods?

The project compares:

| Kind | Methods |
|---|---|
| **Our zero-shot TSFM detectors** (no training) | **Chronos-2 residual** and **TimesFM-2.5 residual**: flag a change where the model is most surprised by the next value. **Chronos-2 embedding**: flag a change where the model's internal representation of the left and right halves of a window differ. |
| **Classical CPD baselines** | **PELT**, **Binary Segmentation**, **BOCPD**, and **"Zero"** (predict no change at all, a reference) |
| **Data** (human-labelled change points) | **TCPD**: 36 real series (economics, climate, traffic and more), 5 human annotators each. **SKAB**: 34 recordings from an industrial water-pump testbed with 8 sensors. |
| **Scoring** | **F1** (a detection is correct if within 5 steps of a labelled change) and **segmentation covering** (how well the predicted segments overlap the true ones), exactly as in the standard TCPDBench benchmark |

## 2. What the final output is (and why there is no frontend)

This is a **research project**, not an app, so **no frontend or website is needed**. The final output is the **measured answer to the research question** plus the evidence behind it:

| Output | Where | What it shows |
|---|---|---|
| Results tables (CSV and LaTeX) | `results/eval/`, `paper/tables/` | Every method's F1 and covering on every series, the statistical tests and the runtime |
| Figures | `figures/` | Example detections (Fig. 2), F1 distribution (Fig. 3), effect of context length (Fig. 4) |
| Executed GPU notebook | `notebooks/colab/01_tsfm_scores.ipynb` | The real Chronos-2 and TimesFM-2.5 runs on a Colab T4 GPU, with all outputs visible |
| Live demo script | `scripts/demo.py` | Runs the methods on any one series and shows detections against the human labels |
| Tested code | `src/cpd/`, `tests/` | 19 automated tests, including checks that our numbers reproduce TCPDBench |
| Paper (draft) | `paper/main.tex` | IEEE conference format |

**Headline result** (mean over series, default settings; source: `results/eval/table_main.csv`):

| Method | TCPD F1 | TCPD Cover | SKAB F1 | SKAB Cover |
|---|---|---|---|---|
| Zero (no change points) | 0.638 | 0.554 | 0.349 | 0.364 |
| PELT | 0.652 | 0.640 | 0.166 | 0.255 |
| Binary Segmentation | **0.686** | **0.665** | 0.333 | **0.616** |
| BOCPD | 0.656 | 0.578 | 0.293 | 0.443 |
| **Chronos-2 residual (ours)** | 0.668 | 0.608 | 0.303 | 0.482 |
| **TimesFM-2.5 residual (ours)** | 0.641 | 0.583 | 0.244 | 0.468 |
| **Chronos-2 embedding (ours)** | 0.605 | 0.563 | 0.272 | 0.575 |

**In one sentence:** without any tuning, the zero-shot Chronos-2 residual detector is as good as classical CPD on TCPD (2nd of 6, statistically tied with the best). Once every method is tuned, the classical methods win, and on the industrial SKAB data no method beats "predict nothing" on F1.

## 3. How it works (pipeline)

```
 TCPD + SKAB (public, human-labelled)
        │  scripts/download_data.py  →  data/raw/
        │  scripts/prepare_data.py   →  data/processed/   (one format: values + labels)
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
 Classical baselines (your laptop, CPU)       TSFM inference (Google Colab, free T4 GPU)
 scripts/run_classical.py                     notebooks/colab/01_tsfm_scores.ipynb
 PELT · BinSeg · BOCPD, default + tuned       Chronos-2 & TimesFM-2.5 forecasts, Chronos-2 embeddings
        │                                              │  → results/tsfm_raw/  (score per time step)
        │                                     scripts/score_tsfm.py  (score → change points)
        ▼                                              ▼
 results/classical/grid*.csv                  results/tsfm_cp/grid.csv
        └──────────────────────┬───────────────────────┘
                               ▼
        scripts/evaluate.py     → results/eval/*.csv   (F1, covering, per series and averages)
        scripts/make_report.py  → paper/tables/*.tex, figures/*.pdf|png, statistical tests
        scripts/demo.py         → live demo on any one series
```

**Rule of the project:** your laptop never needs a GPU. Only the foundation-model step uses a GPU, and it runs in Google Colab.

## 4. Folder structure

```
p1-tsfm-changepoint/
├── README.md                  ← this file
├── CLAUDE.md, docs/           ← project rules, specs, design decisions, M0 research findings
├── requirements-local.txt     ← Python packages for your laptop (CPU)
├── requirements-colab.txt     ← packages the Colab notebook installs
├── pyproject.toml             ← makes `src/cpd` importable (pip install -e .)
├── configs/                   ← ALL parameters (datasets, models, detectors, baselines)
├── src/cpd/
│   ├── data/loaders.py        ← read TCPD and SKAB, label checks, z-scoring
│   ├── baselines/             ← classical.py (PELT, BinSeg), bocpd.py (BOCPD)
│   ├── detectors/             ← scores.py (residual and embedding scores), peaks.py (score → change points)
│   └── evaluation/            ← metrics.py (F1, covering), stats.py (Wilcoxon + Holm)
├── scripts/                   ← the commands you run, in order (see Section 6)
├── notebooks/colab/           ← 01_tsfm_scores.ipynb (executed, with outputs)
├── tests/                     ← pytest tests
├── data/raw/                  ← downloaded datasets (not in git)
├── data/processed/            ← unified series (not in git, rebuilt by prepare_data.py)
├── results/                   ← every number in the project comes from here
├── figures/                   ← generated figures
└── paper/                     ← IEEE paper source (main.tex, refs.bib, tables/)
```

## 5. Requirements

| Need | Version / note |
|---|---|
| **Python** | **3.11** (tested). 3.12 and 3.13 should also work. Get it from https://www.python.org/downloads/ (on Windows, tick *"Add Python to PATH"*). |
| **git** | Any recent version (https://git-scm.com/downloads). Used to download the datasets. |
| **Internet** | Needed once, to download the data (~40 MB, or ~350 MB with the optional TCPDBench files). |
| **Disk** | ~500 MB including the virtual environment |
| **GPU** | **Not needed on your laptop.** Only for the optional Colab step (free Google account). |
| OS | Windows 10/11, macOS or Linux |

## 6. Step-by-step: run the project

> **Two ways to run it:**
> - **A. Demo mode (about 10 minutes).** Use the results already in `results/`. Nothing heavy is re-run. Do Steps 1–4, then Step 10. **Use this on the demonstration laptop.**
> - **B. Full reproduction (about 4 hours, most of it unattended).** Re-run every experiment from scratch. Do Steps 1–10.

In the commands below, lines starting with `#` are comments. Run the commands from the project folder.

### Step 1: Get the project onto the laptop
Copy the whole project folder (USB, zip or `git clone`), then open a terminal in it:
```bash
cd p1-tsfm-changepoint
```
> The `results/` folder and the executed notebook travel with the project. Do **not** copy the `.venv` folder from another computer; create a fresh one in Step 2.

### Step 2: Create the Python environment and install the packages (~3 min)

**Windows (PowerShell or Command Prompt):**
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-local.txt
pip install -e .
```
**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-local.txt
pip install -e .
```
After `activate`, your prompt starts with `(.venv)`. **Activate it again in every new terminal** before running anything else.
> On PowerShell, if `activate` is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use Command Prompt.

### Step 3: Download the datasets (~5 min)
```bash
python scripts/download_data.py
```
This downloads TCPD and SKAB into `data/raw/` and builds the 10 TCPD series that TCPD fetches from external sources (archive.org and Yahoo Finance). It ends with `Done: 42/42 TCPD series, 35/35 SKAB files`.

If a series fails, it's usually a temporary website outage, so run the same command again. Finished parts are skipped.

Optional, only for the full validation tests (~310 MB):
```bash
python scripts/download_data.py --tcpdbench
```

### Step 4: Prepare the data (~1 min)
```bash
python scripts/prepare_data.py
```
Expected output:
```
tcpd: 36 series (4 multivariate), length 15-991, 443 annotated change points (all annotators)
skab: 34 series (34 multivariate), length 745-1327, 129 annotated change points (all annotators)
```

### Step 5: Run the tests (~1–5 min)
```bash
python -m pytest
```
Expected: `19 passed`. Tests that need the optional TCPDBench files are *skipped* (not failed) if you didn't download them.

These tests check, among other things, that:
- our F1 and covering code gives exactly TCPDBench's published scores;
- our BOCPD, PELT and BinSeg give exactly TCPDBench's change points.

### Step 6: Classical baselines (full reproduction only; ~2 h with 4 parallel workers)
Each of the 70 series runs with default settings and a tuning grid of 83–501 parameter sets per method. Open **4 terminals** (activate the environment in each) and run one command per terminal:
```bash
python scripts/run_classical.py 0 4
python scripts/run_classical.py 1 4
python scripts/run_classical.py 2 4
python scripts/run_classical.py 3 4
```
Results go to `results/classical/grid_0.csv … grid_3.csv`. If interrupted, run the same command again; finished work is skipped.

To re-run from scratch, delete `results/classical/grid*.csv` first.

### Step 7: Foundation-model scores in Google Colab (full reproduction only; ~1 h 40 min on a free T4)
1. Make the input bundle on your laptop:
   ```bash
   python scripts/make_colab_bundle.py
   ```
   This creates `colab_input.zip` (~1.3 MB).
2. In **Google Drive**, create a folder named **`P1-tsfm`** directly in *My Drive*, and upload `colab_input.zip` into it.
3. Open https://colab.research.google.com, choose **File → Upload notebook**, and pick `notebooks/colab/01_tsfm_scores.ipynb`.
4. Choose **Runtime → Change runtime type → T4 GPU → Save**.
5. Choose **Runtime → Run all**. When asked, click **Connect to Google Drive** and allow access.
   - A quick smoke test runs first.
   - Then 210 score files are written: 70 series × 3 detectors.
   - If Colab disconnects, choose *Run all* again; finished series are skipped.
6. When the last cell prints `210/210 files written`, download two files:
   - **`My Drive/P1-tsfm/tsfm_raw.zip`**
   - the executed notebook (**File → Download → Download .ipynb**)
7. Put them in the project:
   - unzip `tsfm_raw.zip` into `results/tsfm_raw/`, so that the file `results/tsfm_raw/manifest.json` exists;
   - replace `notebooks/colab/01_tsfm_scores.ipynb` with the downloaded copy.

### Step 8: Turn the TSFM scores into change points (~1 min)
```bash
python scripts/score_tsfm.py
```
This checks that all 210 Colab files are present, then writes `results/tsfm_cp/grid.csv`.

### Step 9: Evaluate everything and build the tables and figures (~2 min)
```bash
python scripts/evaluate.py
python scripts/make_report.py
```
- `evaluate.py` prints the summary table and writes `results/eval/*.csv`.
- `make_report.py` writes `paper/tables/*.tex` and `figures/fig2_example`, `fig3_f1_boxplot` and `fig4_context_length` (each as `.pdf` and `.png`).

### Step 10: Live demo (~30 s per series)
```bash
python scripts/demo.py                    # Nile river: the classic change point example
python scripts/demo.py skab/valve1_0      # an industrial pump recording
python scripts/demo.py --list             # all 70 series you can try
python scripts/demo.py tcpd/nile --show   # also open the figure in a window
```
For one series, the demo does the following:
1. runs PELT, BinSeg and BOCPD **live** on your CPU;
2. applies the saved Chronos-2 and TimesFM-2.5 scores from the Colab run;
3. prints every method's detected change points with F1 and covering against the human labels;
4. saves a figure to `figures/demo_<series>.png`;
5. prints the overall benchmark table.

## 7. What to show in the final project demonstration (suggested 10–12 min)

| # | Show | How | Point to make |
|---|---|---|---|
| 1 | The problem and the gap | 2 slides: what a change point is, what TSFMs are, and the gap from Uray et al. 2026 | "They saw forecast errors spike at changes and suggested CPD; nobody had measured it." |
| 2 | The pipeline | Section 3 diagram (or Fig. 1 in the paper) | Zero-shot: no model is trained. GPU work in Colab; everything else on the laptop. |
| 3 | Trustworthy evaluation | `python -m pytest` (live) and `results/eval/table_tcpdbench.csv` | Our metrics and baselines reproduce the published TCPDBench numbers. |
| 4 | Live detection | `python scripts/demo.py tcpd/nile --show`, then `python scripts/demo.py skab/valve1_0 --show` | Compare each method's detections with the dashed human labels. |
| 5 | Real GPU runs | Open `notebooks/colab/01_tsfm_scores.ipynb` and `results/tsfm_raw/manifest.json` | Real Chronos-2 and TimesFM-2.5 inference on a T4, pinned model versions, 210/210 outputs. |
| 6 | Results | `figures/fig3_f1_boxplot.png`, the table in Section 2, `results/eval/stats.csv` | The answer to the research question, with significance tests. |
| 7 | Insights | `figures/fig4_context_length.png`, `results/eval/breakdown.csv` | TSFMs are good on short series and weak on long ones; tuning favours classical methods; GPU cost is 10–70× higher. |
| 8 | Conclusion | 1 slide | A mixed, honest result: zero-shot TSFMs match untuned classical CPD on TCPD, but don't beat tuned classical CPD. Future work: better thresholds, multivariate TSFMs, other industrial data. |

*Optional:* re-run `python scripts/evaluate.py` live to show that every number is regenerated from the raw outputs.

## 8. Key terms (useful for the viva)

| Term | Meaning |
|---|---|
| **Zero-shot** | The model is used as downloaded, with no training or fine-tuning on our data. |
| **Forecast-residual detector** | Predict the next value; a big surprise (error relative to the model's own uncertainty band) suggests a change. |
| **Embedding detector** | Compare the model's internal representation of the left and right halves of a window. |
| **F1 with margin 5** | A detection counts if it is within 5 steps of a human-labelled change. Precision and recall are combined over the 5 annotators. |
| **Covering** | How well the predicted segments overlap the true segments (0 to 1). |
| **Default vs oracle** | *Default*: one fixed setting for all series, which is the fair zero-shot test. *Oracle*: the best setting per series, which is an upper bound. |
| **Zero baseline** | Predict no change points at all. It is surprisingly strong under F1, so a method must beat it to be useful. |
| **Wilcoxon + Holm** | A paired statistical test over series, corrected for multiple comparisons. |

## 9. Configuration

All parameters live in `configs/`. No numbers are hard-coded in the code.

| File | Contains |
|---|---|
| `datasets.yaml` | data paths, excluded series (with reasons), F1 margin |
| `models.yaml` | Hugging Face model IDs **and exact revisions** (verified) |
| `detectors.yaml` | context lengths, window sizes, threshold `tau`, tuning grids, Colab batch sizes |
| `baselines.yaml` | PELT/BinSeg penalties, BOCPD priors, tuning grids |

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found (Windows) | Use `py -3.11 …` or reinstall Python with *Add to PATH*. |
| `ModuleNotFoundError: cpd` | Activate the environment and run `pip install -e .` |
| TCPD build: *"Validating the file … failed"* | Re-run `python scripts/download_data.py`. This is a temporary yfinance or archive.org issue. The script already handles the Windows line-ending issue. |
| `bee_waggle_6 … didn't match a known checksum` | Harmless. TCPD documents tiny floating-point rounding differences on some machines. |
| Tests *skipped* | Optional TCPDBench files are missing. Run `python scripts/download_data.py --tcpdbench`. |
| Everything is very slow to start (Windows) | Antivirus real-time scanning slows Python imports. It's harmless, just slower. |
| `score_tsfm.py` says the Colab output is missing | Unzip `tsfm_raw.zip` so that `results/tsfm_raw/manifest.json` exists. |
| Colab: `Upload colab_input.zip …` assertion | The Drive folder must be exactly `My Drive/P1-tsfm/`, or edit the `DRIVE = …` line in the first code cell. |
| Colab disconnected mid-run | Run all again; finished files are skipped. |

## 11. Credits and citations

- **Data.**
  - TCPD (van den Burg & Williams, arXiv 2003.06222), https://github.com/alan-turing-institute/TCPD
  - SKAB (Katser & Kozitsin, Kaggle 2020, DOI 10.34740/KAGGLE/DSV/1693952), https://github.com/waico/SKAB
- **Evaluation protocol.** TCPDBench (MIT license). Its metric code is ported verbatim in `src/cpd/evaluation/metrics.py`.
- **Models.**
  - Chronos-2 (`amazon/chronos-2`, Apache-2.0)
  - TimesFM-2.5 (`google/timesfm-2.5-200m-pytorch`, Apache-2.0)
- **Gap source.** Uray et al., EUROCAST 2026 (arXiv 2607.12454).
- **IEEE base paper.** De Ryck, De Vos, Bertrand, *Change Point Detection in Time Series Data Using Autoencoders With a Time-Invariant Representation*, IEEE Trans. Signal Processing 69 (2021). Discussed; not re-run, because its code has no license.

Detailed design decisions and the research-gap verification are in `docs/DESIGN.md` and `docs/M0_FINDINGS.md`.
