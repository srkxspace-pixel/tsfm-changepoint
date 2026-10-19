# Do Zero-Shot Time-Series Foundation Models Detect Change Points?

**Forecast-Residual and Embedding-Distance Detectors vs. Classical CPD on Real-World and Industrial Sensor Data**

*B.Tech final-year project · Track B (Research) · AI & Machine Learning*

> **Short answer.** Partly. Used as downloaded, with no training and no tuning, the **Chronos-2 forecast-residual detector** finds change points about as well as the best classical method on real-world data (TCPD). The difference is not statistically significant. When every method is allowed to tune its settings, though, the classical methods are clearly and significantly better. On the industrial SKAB data, no method beats simply predicting "no change" on F1.

---

## Contents

1. [The project in plain words](#1-the-project-in-plain-words)
2. [The research gap](#2-the-research-gap)
3. [What is new in this project (novelty)](#3-what-is-new-in-this-project-novelty)
4. [How it works](#4-how-it-works)
5. [Results and findings](#5-results-and-findings)
6. [Folder structure](#6-folder-structure)
7. [One-time setup on a new laptop](#7-one-time-setup-on-a-new-laptop)
8. [Step-by-step: run the project](#8-step-by-step-run-the-project)
9. [Presenting the project (demo plan)](#9-presenting-the-project-demo-plan)
10. [Key terms (for the viva)](#10-key-terms-for-the-viva)
11. [Configuration](#11-configuration)
12. [Troubleshooting](#12-troubleshooting)
13. [Project status and limitations](#13-project-status-and-limitations)
14. [Credits and citations](#14-credits-and-citations)

---

## 1. The project in plain words

A **change point** is a moment when a signal starts to behave differently. A river's yearly flow drops after a dam is built. A pump's sensor readings shift when a valve starts to fail. A stock's volatility jumps after a crash. **Change point detection (CPD)** means finding those moments automatically.

**Time-series foundation models (TSFMs)** are large neural networks pretrained on huge collections of time series, so they can forecast almost any signal without being trained on it. Two examples are **Chronos-2** (Amazon) and **TimesFM-2.5** (Google). Using a model this way, as downloaded and with no training, is called **zero-shot**.

The idea we test is simple. If a model has learnt what "normal" looks like, it should be **surprised** when the signal changes. Its forecast error should jump exactly at a change point.

**Research question:**

> Can pretrained time-series foundation models, used zero-shot with no training, locate change points in real labelled data as accurately as established change-point detection methods?

What we compare:

| Kind | Methods |
|---|---|
| **Our zero-shot TSFM detectors** (no training) | **Chronos-2 residual** and **TimesFM-2.5 residual**: flag a change where the model is most surprised by the next value. **Chronos-2 embedding**: flag a change where the model's internal representation of the left and right halves of a window differ. |
| **Classical CPD baselines** | **PELT**, **Binary Segmentation (BinSeg)** and **BOCPD**, plus **Zero**, a reference method that predicts no change at all |
| **Data** (change points labelled by people) | **TCPD**: 36 real-world series (economics, climate, traffic and more), each labelled by 5 annotators. **SKAB**: 34 recordings from an industrial water-pump testbed with 8 sensors. |
| **Scoring** | The standard **TCPDBench** protocol: **F1** (a detection counts if it is within 5 steps of a labelled change) and **segmentation covering** (how well the predicted segments overlap the true ones) |

---

## 2. The research gap

**Where the idea comes from.** Uray et al. (EUROCAST 2026, [arXiv 2607.12454](https://arxiv.org/abs/2607.12454)) ran TimesFM zero-shot on the SWaT industrial testbed for *anomaly* detection. In one figure (Fig. 4b) they noticed that the forecast error **spikes at the start and end of an anomaly**. In their abstract and conclusion they name **change point detection** as a promising use of foundation models. **They never tested it.** Their study has no change point metric, no CPD baseline and only one dataset.

**What other work has done, and what it left open:**

| Work | What it does | What is missing for our question |
|---|---|---|
| Uray et al. 2026 ([2607.12454](https://arxiv.org/abs/2607.12454)) | TimesFM zero-shot anomaly detection on SWaT; error spikes at changes, shown in a figure | No CPD metric, no CPD baselines, one dataset. The idea is never measured. |
| Choi et al. 2026, ICLR TSALM workshop ([2604.16428](https://arxiv.org/abs/2604.16428)) | Supervised probes on Chronos-2 / MOMENT embeddings classify the *type* of shift (mean, variance or trend) | Synthetic AR(1) data only, with the change always in the middle of the window. It needs training and never *locates* a change. |
| GAS-FM, ICLR 2027 submission (OpenReview `mHedcrJ0TB`), concurrent | Training-free CPD from frozen TSFM **embeddings** | **No forecast-residual detector.** It does not use the TCPDBench multi-annotator protocol. |
| TSB-AD (NeurIPS 2024) and other TSFM anomaly benchmarks | Use TSFM forecast errors as anomaly scores | Score anomalous points or segments, **not change point locations** |
| TIRE (De Ryck et al., IEEE TSP 2021, our IEEE base paper), KL-CPD, TS-CP2 | Learned CPD with representation distances | Must be **trained** on every new data source |
| 98 papers citing TCPDBench since 2024 (checked at M0) | CPD benchmarks | **None uses a foundation model** |
| SKAB CPD leaderboard | Industrial CPD benchmark | **No foundation-model entry** |

**The gap, in one sentence:** nobody had measured whether a TSFM's **forecast error**, used zero-shot, finds change points on **human-labelled CPD benchmarks** under the standard protocol and against classical CPD. That is the question this project answers.

The full prior-art check is in [docs/M0_FINDINGS.md](docs/M0_FINDINGS.md).

---

## 3. What is new in this project (novelty)

1. **A zero-shot forecast-residual change point detector.** It measures the one-step-ahead forecast surprise, normalised by the model's own predictive interval, and is built on Chronos-2 and TimesFM-2.5. To our knowledge, this is the first evaluation of TSFM forecast residuals **as a change point detector** on labelled CPD benchmarks. It is the direct test of the observation by Uray et al.
2. **The first TSFM results under the TCPDBench protocol.** We use multi-annotator F1 and covering in both the default and oracle settings. The numbers can be compared directly with the published classical results, and our own baselines reproduce TCPDBench's numbers (Finding 7).
3. **The first TSFM change point results on SKAB**, a real industrial sensor testbed.
4. **An analysis of how TSFMs behave on short series.** Most TCPD series are short (half have fewer than 275 points), and we break the results down by series length.
5. **A fully reproducible, validated pipeline.** All GPU work is in one executed Colab notebook with pinned model revisions, and every number traces to a file in `results/`.

**What is *not* new (stated honestly):** using frozen embeddings for CPD is concurrent work (GAS-FM). Our **Chronos-2 embedding detector** is included only to compare the two ways of using a TSFM under one protocol. TIRE (the IEEE base paper) is cited and discussed but not re-run, because its code has no software license.

---

## 4. How it works

### 4.1 The detectors

**Forecast-residual detector (main contribution).** At each time step *t*, the model forecasts the next value from the last *L* values. The score is how far the real value lands from the forecast median, divided by the width of the model's own 10–90% prediction band:

```
s_t = | x_t − median_t | / max(q90_t − q10_t, 0.001)        (averaged over channels)
```

A change makes the new value unlike its past, so `s_t` should peak at the first point of the new segment. Chronos-2 reads all channels jointly. TimesFM-2.5 is univariate, so each channel is forecast separately and the channel scores are averaged.

**Embedding detector (comparison).** One window of width 2*w* centred on *t* goes into Chronos-2. The score is the cosine distance between the mean patch embeddings of its left and right halves. One shared window is used because Chronos-2 normalises every input on its own, and two separately embedded windows would hide a pure level shift.

**Shared peak picking.** Each score curve is standardised robustly ((s − median) / (1.4826·MAD)) and smoothed with a moving average of width 3. Every local peak above **τ = 3**, at least 5 steps from the next one, becomes a change point. Both TSFM detectors use exactly this step, so any difference between them comes from the scores alone.

**Baselines.**
- **PELT** and **BinSeg** come from the `ruptures` library.
- **BOCPD** is our own implementation, following the R `ocp` package.

All three reproduce the change points of the original TCPDBench R implementations.

### 4.2 Evaluation settings

- **Default**: one fixed setting for every series. It was chosen **before any result was seen**: L = 128, w = 32, τ = 3 for the TSFM detectors, and package defaults for the baselines. This is the fair zero-shot test.
- **Oracle**: the best setting per series from a grid, which is an upper bound on what tuning could achieve. The grids have 83, 165 and 501 settings for PELT, BinSeg and BOCPD, and 40 for each residual detector and 30 for the embedding detector.
- **Statistics**: a paired Wilcoxon signed-rank test over the series, with Holm correction, compares each TSFM detector with the best classical method.

### 4.3 Pipeline

```
 TCPD + SKAB (public, human-labelled)
        │  scripts/download_data.py  →  data/raw/
        │  scripts/prepare_data.py   →  data/processed/   (one format: values + labels)
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
 Classical baselines (laptop, CPU)            TSFM inference (Google Colab, free T4 GPU)
 scripts/run_classical.py                     notebooks/colab/01_tsfm_scores.ipynb
 PELT · BinSeg · BOCPD, default + oracle      Chronos-2 & TimesFM-2.5 forecasts, Chronos-2 embeddings
        │                                              │  → results/tsfm_raw/  (one score per time step)
        │                                     scripts/score_tsfm.py  (scores → change points)
        ▼                                              ▼
 results/classical/grid*.csv                  results/tsfm_cp/grid.csv
        └──────────────────────┬───────────────────────┘
                               ▼
        scripts/evaluate.py     → results/eval/*.csv   (F1, covering, per series and averages)
        scripts/make_report.py  → paper/tables/*.tex, figures/*.pdf|png, statistical tests
        app.py / scripts/demo.py → interactive demo on any series
```

**Compute rule:** the laptop never needs a GPU. Only the foundation-model step uses one, and it runs in Google Colab.

---

## 5. Results and findings

Every number below comes from a file in [`results/eval/`](results/eval/). Values are means over series.

### 5.1 Main table ([results/eval/table_main.csv](results/eval/table_main.csv))

| Method | TCPD F1 default | TCPD F1 oracle | TCPD Cover default | SKAB F1 default | SKAB F1 oracle | SKAB Cover default |
|---|---|---|---|---|---|---|
| Zero (no change points) | 0.638 | 0.638 | 0.554 | **0.349** | 0.349 | 0.364 |
| PELT | 0.652 | 0.864 | 0.640 | 0.166 | 0.581 | 0.255 |
| Binary Segmentation | **0.686** | 0.861 | **0.665** | 0.333 | 0.538 | **0.616** |
| BOCPD | 0.656 | **0.892** | 0.578 | 0.293 | **0.634** | 0.443 |
| **Chronos-2 residual (ours)** | 0.668 | 0.828 | 0.608 | 0.303 | 0.477 | 0.482 |
| **TimesFM-2.5 residual (ours)** | 0.641 | 0.827 | 0.583 | 0.244 | 0.426 | 0.468 |
| **Chronos-2 embedding (ours)** | 0.605 | 0.739 | 0.563 | 0.272 | 0.422 | 0.575 |

### 5.2 Findings

1. **Without tuning, Chronos-2 residual matches classical CPD on TCPD.** It is 2nd of the 6 detectors on F1 (0.668), behind only BinSeg (0.686) and ahead of BOCPD (0.656) and PELT (0.652). The difference from BinSeg is not significant: it wins on 14 series, ties on 3 and loses on 19, with Holm p = 0.74. None of the three TSFM detectors differs significantly from the best classical method in the default setting, on either dataset ([table_stats.csv](results/eval/table_stats.csv)).
2. **With tuning, classical CPD clearly wins.** In the oracle setting, BOCPD (0.892 on TCPD, 0.634 on SKAB) is significantly better than every TSFM detector:
   - on TCPD, Holm p = 0.012 for Chronos-2 residual, 0.016 for TimesFM-2.5 and below 0.001 for the embedding detector;
   - on SKAB, Holm p < 0.001 for all three.

   **This is not because the classical grids are larger.** Cut down to the same 40 settings as the TSFM grid, BOCPD still reaches 0.871 on TCPD and 0.600 on SKAB ([oracle_matched_budget.csv](results/eval/oracle_matched_budget.csv)).
3. **On the industrial SKAB data, no method beats "no change" on F1.** Zero scores 0.349. The best detector is BinSeg at 0.333, and the best TSFM detector is Chronos-2 residual at 0.303. On **covering**, the embedding detector comes 2nd (0.575), behind BinSeg (0.616), and well above Zero (0.364).
4. **The forecast-residual detector beats the embedding detector, and Chronos-2 beats TimesFM-2.5.**
   - **Residual vs embedding:** the residual is better on TCPD F1 (0.668 vs 0.605). The embedding detector even falls below Zero there.
   - **Chronos-2 vs TimesFM-2.5:** Chronos-2 is ahead on both datasets (TCPD 0.668 vs 0.641, SKAB 0.303 vs 0.244).
5. **TSFMs do well on short and medium series and badly on long ones** ([breakdown.csv](results/eval/breakdown.csv), TCPD default F1):

   | Series length | Number of series | Best method | Chronos-2 residual | TimesFM-2.5 residual |
   |---|---|---|---|---|
   | under 100 | 9 | BOCPD 0.833 | **0.811 (2nd)** | 0.807 (3rd) |
   | 100–255 | 9 | BinSeg 0.795 | 0.734 (3rd) | 0.729 |
   | 256–511 | 10 | **Chronos-2 residual 0.678** | **0.678 (1st)** | 0.626 |
   | 512 and over | 8 | Zero 0.702 | 0.421 | 0.376 |

   On the longest series, every detector falls below Zero. The lowest are TimesFM-2.5 (0.376), BOCPD (0.405) and Chronos-2 (0.421).
6. **The context length L matters only a little** ([context_sensitivity.csv](results/eval/context_sensitivity.csv)). On TCPD, Chronos-2 residual F1 stays between 0.653 and 0.679 for every L from 32 to 256. On SKAB, the longest context (L = 256) is best for both models: 0.322 for Chronos-2 and 0.281 for TimesFM-2.5.
7. **The evaluation is trustworthy: we reproduce TCPDBench** ([table_tcpdbench.csv](results/eval/table_tcpdbench.csv), 32 univariate series).

   | Method | Our F1 | Published F1 |
   |---|---|---|
   | Zero | 0.645 | 0.645 |
   | BinSeg | 0.698 | 0.698 |
   | BOCPD | 0.662 | 0.662 |
   | PELT | 0.657 | 0.674 |

   Our PELT returns exactly the same change points as TCPDBench's R `changepoint` run on all 32 series. Its F1 is lower only because TCPDBench stores PELT's change points one index early.
8. **The foundation models are much slower** ([table_runtime.csv](results/eval/table_runtime.csv), seconds per SKAB series). The classical methods run on a laptop CPU, and the TSFMs run on a Colab T4 GPU.

   | Method | Runs on | Seconds per series |
   |---|---|---|
   | BinSeg | laptop CPU | 0.36 |
   | BOCPD | laptop CPU | 0.70 |
   | PELT | laptop CPU | 2.9 |
   | Chronos-2 embedding | T4 GPU | 6.6 |
   | Chronos-2 residual | T4 GPU | 11.4 |
   | TimesFM-2.5 residual | T4 GPU | 25.5 |

9. **The early-context check rules out an artefact** ([early_context_check.csv](results/eval/early_context_check.csv)). The residual detector sees very little context in its first steps. Ignoring the first 32 steps changes SKAB F1 by less than 0.01. On TCPD, F1 drops instead (0.668 → 0.627), because 42% of TCPD series have a real labelled change in their first 32 steps.

### 5.3 Conclusion

The error-spike observation of Uray et al. holds up as a **usable, training-free change point detector**. Without any tuning, Chronos-2's forecast surprise is as good as untuned classical CPD on real-world series, and it is strongest on short and medium-length series. However, foundation models are **not a replacement** for classical CPD:
- once methods are tuned, the classical methods are significantly better;
- foundation models struggle on long series and on the industrial data;
- they are much slower: 11–25 s per SKAB series on a GPU, against under 1 s for BinSeg and BOCPD on a CPU.

Possible future work: adaptive thresholds instead of a fixed τ, better use of multivariate input, and more industrial datasets.

### 5.4 Figures ([figures/](figures/))

| File | Shows |
|---|---|
| `fig2_example.png` | Example series with the human labels and each method's detections |
| `fig3_f1_boxplot.png` | Per-series F1 distribution of every method, default and oracle |
| `fig4_context_length.png` | Effect of the context length L on the residual detectors |

---

## 6. Folder structure

```
p1-tsfm-changepoint/
├── README.md                      ← this file
├── app.py                         ← Streamlit web demo (streamlit run app.py)
├── requirements-local.txt         ← Python packages for the laptop (CPU only, pinned)
├── requirements-colab.txt         ← packages the Colab notebook installs
├── pyproject.toml                 ← makes src/cpd importable (pip install -e .)
├── configs/                       ← ALL parameters; no numbers are hard-coded in the code
│   ├── datasets.yaml              ←   data paths, excluded series (with reasons), F1 margin
│   ├── models.yaml                ←   Hugging Face model IDs and exact revisions
│   ├── detectors.yaml             ←   L, w, τ, tuning grids, Colab batch sizes
│   └── baselines.yaml             ←   PELT/BinSeg penalties, BOCPD priors, tuning grids
├── src/cpd/                       ← the project's Python package
│   ├── data/loaders.py            ←   read TCPD and SKAB, label checks, z-scoring
│   ├── baselines/                 ←   classical.py (PELT, BinSeg), bocpd.py (BOCPD)
│   ├── detectors/                 ←   scores.py (residual and embedding scores), peaks.py (score → change points)
│   ├── evaluation/                ←   metrics.py (F1, covering), stats.py (Wilcoxon + Holm)
│   └── demo.py                    ←   shared code for the web and terminal demos
├── scripts/                       ← the commands you run, in order (Section 8)
│   ├── download_data.py           ←   Step 3: download TCPD and SKAB
│   ├── prepare_data.py            ←   Step 4: build data/processed/
│   ├── run_classical.py           ←   Step 6: PELT, BinSeg, BOCPD on every series
│   ├── make_colab_bundle.py       ←   Step 7: zip the data for Colab
│   ├── score_tsfm.py              ←   Step 8: TSFM scores → change points
│   ├── evaluate.py                ←   Step 9: F1 and covering for everything
│   ├── make_report.py             ←   Step 9: tables, figures, statistics
│   ├── demo.py                    ←   Step 11: terminal demo
│   └── check_bocpd_vs_tcpdbench.py ←  optional: BOCPD vs TCPDBench check
├── notebooks/colab/
│   └── 01_tsfm_scores.ipynb       ← the GPU notebook, executed, with all outputs visible
├── tests/                         ← automated tests (pytest)
├── data/                          ← raw/ and processed/ (not in git; rebuilt by Steps 3–4)
├── results/                       ← every number in the project comes from here
│   ├── classical/                 ←   baseline runs: every series × every setting (+ run logs)
│   ├── tsfm_raw/                  ←   Colab output: per-step scores (210 files) + manifest.json
│   ├── tsfm_cp/                   ←   TSFM change points for every setting
│   └── eval/                      ←   final tables (CSV)
├── figures/                       ← generated figures (PDF and PNG)
├── paper/                         ← IEEE conference paper source
│   ├── main.tex                   ←   title, introduction, related work, method, setup
│   ├── sections/                  ←   abstract.tex; results.tex (results, discussion, conclusion)
│   ├── tables/                    ←   tables generated by make_report.py (never typed by hand)
│   └── refs.bib                   ←   references
├── docs/                          ← SPECS, ARCHITECTURE, DESIGN decisions, M0 research findings
├── .streamlit/config.toml         ← web demo theme
├── .claude/launch.json            ← preview config for Claude Code (optional)
└── CLAUDE.md                      ← project rules for the AI coding assistant
```

---

## 7. One-time setup on a new laptop

| Need | Version / note |
|---|---|
| **Python** | **3.11** exactly (tested with 3.11.9). Other Python versions can stay installed alongside it. |
| **git** | Any recent version. It moves the project to the laptop (Step 0) and downloads the data (Step 3). |
| **Internet** | Needed for Steps 2 and 3: about 300 MB of Python packages and about 350 MB of data. |
| **Disk space** | About 1–2 GB, mostly the Python environment and the downloaded data |
| **A short folder path** | For example `C:\projects\p1-tsfm-changepoint`. Windows limits file paths to 260 characters, and a deeply nested folder makes Step 2 fail. |
| **GPU** | **Not needed on the laptop.** Only the optional Colab step uses one (a free Google account is enough). |
| **Operating system** | Windows 10/11 (tested), macOS or Linux |

### 7.1 Install Python 3.11

🪟 **Windows:**
1. Open https://www.python.org/downloads/release/python-3119/ and scroll down to **Files**.
2. Click **Windows installer (64-bit)** to download it.
3. Run the installer. On the first screen, tick **"Add python.exe to PATH"**, then click **Install Now**.
4. Open a **new** PowerShell window and check:
   ```powershell
   py -3.11 --version
   ```
   It should print `Python 3.11.9`.

🍎 **macOS:** download the *macOS 64-bit universal2 installer* from the same page and run it. Check with `python3.11 --version`.

🐧 **Linux (Ubuntu/Debian):** `sudo apt install python3.11 python3.11-venv`. Check with `python3.11 --version`.

### 7.2 Install git

🪟 **Windows:**
1. Open https://git-scm.com/download/win and download **64-bit Git for Windows Setup**.
2. Run it and click **Next** on every screen; the defaults are fine.
3. Open a **new** PowerShell window and check:
   ```powershell
   git --version
   ```
   It should print `git version 2.…`.

🍎 **macOS:** run `xcode-select --install` in Terminal. 🐧 **Linux:** run `sudo apt install git`.

---

## 8. Step-by-step: run the project

### Choose your path

| Path | Time | Steps | When to use |
|---|---|---|---|
| **A. Demo** | about 15 min | 0 → 1 → 2 → 3 → 4 → 5 → 10 | Show the project using the results already saved in `results/`. **Use this on the presentation laptop.** |
| **B. End-to-end check** | about 20–30 min | 0 → 1 → 2 → 3 → 4 → 5 → 8 → 9 → 9b → 10 | **Use this to check the project on another laptop.** It builds everything from nothing (environment, data, tests) and regenerates every result table and figure from the saved model outputs. Step 9b then proves they came out identical. It skips only the two long compute steps, whose outputs are already saved in `results/`. |
| **C. Full re-run** | about 4 h, mostly unattended | 0 → 11, all of them | Re-run every experiment, including the 2-hour classical grid (Step 6) and the GPU run in Colab (Step 7) |

Path B was tested from scratch on 2026-10-19: a fresh clone from the bundle, a new environment, a fresh data download, the tests, and a rebuild of every result. Every table and CSV came out byte-for-byte identical.

### About the terminals

Every command block below says **which terminal** to use:

- 🪟 **PowerShell (Windows).** Press **Start**, type `PowerShell`, and open it.
- 🍎🐧 **Terminal (macOS / Linux).** Open the *Terminal* app.
- ☁️ **Google Colab.** A web page in your browser, not a terminal (Step 7 only).

From Step 1 on, all commands are run **from the project folder**. After Step 2, the prompt must start with **`(.venv)`**. If it doesn't, activate the environment again (Step 2b). **Every new terminal window must be activated once.**

---

### Step 0: Move the project to the other laptop (about 2 min)

The project is copied as **one file**, a *git bundle* (about 6 MB). The bundle contains:
- all the code;
- every saved result, including the GPU outputs;
- the executed Colab notebook;
- the paper;
- the full history.

It leaves out the things each laptop must build for itself: `.venv` and `data/`.

**0a. On the original laptop.** 🪟 PowerShell, in the project folder. The bundle holds only what is committed in git, so check first:
```powershell
git status
```
It must say `nothing to commit, working tree clean`. If it doesn't, commit your changes first. Then create the bundle:
```powershell
git bundle create ..\p1-tsfm.bundle --all
```
This writes `p1-tsfm.bundle` into the folder *above* the project folder. Copy that one file to the new laptop by USB stick, Google Drive or email.

**0b. On the new laptop.** 🪟 PowerShell. First make a short folder and go into it:
```powershell
mkdir C:\projects
```
```powershell
cd C:\projects
```
Put `p1-tsfm.bundle` into `C:\projects`, then unpack it:
```powershell
git clone -b master p1-tsfm.bundle p1-tsfm-changepoint
```
It prints `Cloning into 'p1-tsfm-changepoint'...`. The project is now in `C:\projects\p1-tsfm-changepoint`.

🍎🐧 On macOS or Linux, the same commands work in Terminal with `~/projects` in place of `C:\projects`.

> **Without git?** You can also copy the project folder itself (USB or zip). Leave out `.venv` and `data/`, and do **not** use a deeply nested folder. Step 9b needs the hidden `.git` folder, so copy that too.

---

### Step 1: Open a terminal in the project folder

🪟 **PowerShell (Windows):**
```powershell
cd C:\projects\p1-tsfm-changepoint
```
🍎🐧 **Terminal (macOS / Linux):**
```bash
cd ~/projects/p1-tsfm-changepoint
```
Shortcut on Windows: open the folder in File Explorer, click the address bar, type `powershell` and press Enter.

---

### Step 2: Create the Python environment and install the packages (about 5 min, first time only)

**2a. Create the environment** (only once):

🪟 **PowerShell (Windows):**
```powershell
py -3.11 -m venv .venv
```
🍎🐧 **Terminal (macOS / Linux):**
```bash
python3.11 -m venv .venv
```

**2b. Activate it** (in **every** new terminal):

🪟 **PowerShell (Windows):**
```powershell
.\.venv\Scripts\Activate.ps1
```
> If PowerShell says *"running scripts is disabled on this system"*, run this once in the same window and then activate again:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

🍎🐧 **Terminal (macOS / Linux):**
```bash
source .venv/bin/activate
```
The prompt now starts with `(.venv)`.

**2c. Install the packages** (only once, same commands on every OS, with `(.venv)` active):
```bash
python -m pip install --upgrade pip
pip install -r requirements-local.txt
pip install -e .
```
The last line should end with `Successfully installed cpd-0.1.0`.

---

### Step 3: Download the datasets (about 5 min)

With `(.venv)` active, run the command for your path.

**Paths B and C** also fetch the published TCPDBench results (about 310 MB). The reproduction tests in Step 5 and `make_report.py` in Step 9 need them:
```bash
python scripts/download_data.py --tcpdbench
```
**Path A** (demo only) needs just TCPD and SKAB:
```bash
python scripts/download_data.py
```
Either command downloads TCPD and SKAB into `data/raw/`. It also builds the 10 TCPD series that come from external websites (archive.org, Yahoo Finance). The last line is:
```
Done: 42/42 TCPD series, 35/35 SKAB files in C:\projects\p1-tsfm-changepoint\data\raw
```
If a series fails, a website was temporarily down. Run the same command again; finished parts are skipped.

> **Offline shortcut (Path A only):** copy the `data/processed/` folder (about 1.5 MB) from the original laptop into the same place in the new project. Then skip Steps 3 and 4.

---

### Step 4: Prepare the data (about 1 min)

```bash
python scripts/prepare_data.py
```
Expected output:
```
tcpd: 36 series (4 multivariate), length 15-991, 443 annotated change points (all annotators)
skab: 34 series (34 multivariate), length 745-1327, 129 annotated change points (all annotators)
excluded: {'quality_control_1': 'simulated annotator-check series, ...', ..., 'uk_coal_employ': 'contains missing values (excluded by TCPDBench as well)'}
```
The `excluded` line lists the 6 TCPD series left out, exactly as in TCPDBench: 5 simulated quality-control series and 1 series with missing values.

---

### Step 5: Run the tests (about 1 min; up to 15 min on a slow laptop)

```bash
python -m pytest
```
Expected: **`22 passed`**. On Path A (no TCPDBench download), the tests that need those files show as *skipped*, not failed. That is fine.

Among other things, the tests check that:
- our F1 and covering code gives **exactly** TCPDBench's published scores;
- our BOCPD, PELT and BinSeg return **exactly** TCPDBench's change points;
- the web demo and the terminal demo start without errors.

---

### Step 6: Classical baselines (Path C only, about 2 h)

Each of the 70 series runs with the default settings and the full tuning grid (83 to 501 settings per method).

The saved results are skipped automatically, so delete them first, or nothing will be recomputed.

🪟 **PowerShell:**
```powershell
Remove-Item results\classical\grid*.csv
```
🍎🐧 **Terminal:**
```bash
rm results/classical/grid*.csv
```
To finish faster, open **4 terminals** (repeat Steps 1 and 2b in each) and run **one** command per terminal:

| Terminal | Command |
|---|---|
| 1 | `python scripts/run_classical.py 0 4` |
| 2 | `python scripts/run_classical.py 1 4` |
| 3 | `python scripts/run_classical.py 2 4` |
| 4 | `python scripts/run_classical.py 3 4` |

Results go to `results/classical/grid_0.csv … grid_3.csv`. If a run is interrupted, run the same command again; finished series are skipped.

---

### Step 7: Foundation-model scores in Google Colab (Path C only, about 1 h 40 min on a free T4 GPU)

The notebook skips series whose output files already exist in Drive. Use an empty `P1-tsfm` folder so that everything is recomputed.

**7a. On the laptop**, make the input bundle (with `(.venv)` active):
```bash
python scripts/make_colab_bundle.py
```
This creates `colab_input.zip` (about 1.3 MB) in the project folder.

**7b. ☁️ In Google Drive** (https://drive.google.com):
1. In **My Drive**, create a folder named exactly **`P1-tsfm`**.
2. Upload `colab_input.zip` into it.

**7c. ☁️ In Google Colab** (https://colab.research.google.com):
1. **File → Upload notebook**, then choose `notebooks/colab/01_tsfm_scores.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. **Runtime → Run all**. When asked, click **Connect to Google Drive** and allow access.
   - A short smoke test runs first.
   - Then 210 score files are written (70 series × 3 detectors).
   - If Colab disconnects, choose **Run all** again; finished files are skipped.
4. Wait for the last cell to print `210/210 files written`.

**7d. Bring the results back to the laptop:**
1. From Drive, download **`My Drive/P1-tsfm/tsfm_raw.zip`**, and unzip it into `results/tsfm_raw/` so that `results/tsfm_raw/manifest.json` exists.
2. In Colab, choose **File → Download → Download .ipynb**, and replace `notebooks/colab/01_tsfm_scores.ipynb` with the downloaded copy.

---

### Step 8: Turn the TSFM scores into change points (Paths B and C, about 30 s)

```bash
python scripts/score_tsfm.py
```
It checks that all 210 Colab files are present, then prints:
```
chronos2_residual: 70 series x 40 configs
timesfm25_residual: 70 series x 40 configs
chronos2_embedding: 70 series x 30 configs
wrote C:\projects\p1-tsfm-changepoint\results\tsfm_cp\grid.csv
```

---

### Step 9: Evaluate everything and build the tables and figures (Paths B and C, about 1 min)

```bash
python scripts/evaluate.py
```
It prints the summary table, one row per dataset and method, and writes `results/eval/*.csv`.
```bash
python scripts/make_report.py
```
It writes `paper/tables/*.tex` and the figures `figures/fig2_example`, `fig3_f1_boxplot` and `fig4_context_length` (each as `.pdf` and `.png`), then prints:
```
tables: ['datasets.tex', 'main.tex', 'runtime.tex', 'stats.tex', 'tcpdbench.tex']
figures: ['fig2_example.png', 'fig3_f1_boxplot.png', 'fig4_context_length.png']
```

---

### Step 9b: Check that the results were reproduced (Paths B and C)

git compares every regenerated file with the saved one:
```bash
git status --short
```

**Path B: expected output, exactly these three lines:**
```
 M figures/fig2_example.pdf
 M figures/fig3_f1_boxplot.pdf
 M figures/fig4_context_length.pdf
```
A PDF stores the time it was created, so these three always show as changed. Nothing else may appear. If any file under `results/` or `paper/tables/` is listed, a number has changed. ✅ **No other lines means every table, CSV and number in the project was reproduced exactly.**

**Path C:** runtimes differ from computer to computer, so the classical grids, `per_series.csv` and the runtime tables will show as changed. Check the accuracy tables instead:
```bash
git diff results/eval/table_main.csv results/eval/table_stats.csv results/eval/table_tcpdbench.csv
```
The classical methods are deterministic, so their rows should not change. The TSFM rows can differ in the last digits after a new Colab run, because GPU arithmetic is not bit-exact.

To go back to the saved results afterwards:
```bash
git checkout -- results figures paper/tables
```

---

### Step 10: Web demo (Streamlit)

```bash
streamlit run app.py
```
The terminal prints `Local URL: http://localhost:8501` and your browser opens that page. If it doesn't, open the address yourself.
- On a new Windows laptop, a **Windows Security / firewall** window may ask whether to allow Python on the network. You can click **Cancel**: the demo only runs on your own computer and works either way.
- The **first load takes up to about 1–2 minutes** while Python loads its libraries. After that, every click takes 1–5 s.
- **Start it a few minutes before your presentation.**
- To stop it, press **Ctrl + C** in the terminal.

| Tab | What you can do |
|---|---|
| 🔍 **Live detection** | Pick a dataset and series in the sidebar. PELT, BinSeg and BOCPD run **live** on the CPU. Chronos-2 and TimesFM-2.5 use the saved Colab scores. You see: the series with the dashed human labels, every method's detections, an F1/covering table and the foundation model's score curve. Move **τ** or **L** and the detections update instantly. |
| 📊 **Benchmark results** | The main table, the box plot and context-length figures, the Wilcoxon tests, the runtimes, the TCPDBench reproduction check and the breakdown by series length |
| ℹ️ **About the project** | The research question, the gap, the detectors, the pipeline and the GPU run record (models, revisions, GPU, timings) |

---

### Step 11: Terminal demo (optional, about 30 s per series, no browser needed)

```bash
python scripts/demo.py
```
This runs the demo on the Nile river series, the classic change point example. Other options:

| Command | What it does |
|---|---|
| `python scripts/demo.py skab/valve1_0` | An industrial pump recording |
| `python scripts/demo.py --list` | Lists all 70 series you can try |
| `python scripts/demo.py tcpd/nile --show` | Also opens the figure in a window |

For one series, the demo:
1. runs PELT, BinSeg and BOCPD live on the CPU;
2. applies the saved Chronos-2 and TimesFM-2.5 scores;
3. prints every method's change points with F1 and covering;
4. saves a figure to `figures/demo_<series>.png`;
5. prints the overall benchmark table.

---

### Step 12: Build the paper PDF (optional, Overleaf, about 5 min)

No LaTeX installation is needed. Overleaf compiles the paper in the browser.

1. Make a folder named `paper_upload` anywhere on your computer and copy into it:
   - the whole `paper/` folder's contents: `main.tex`, `refs.bib`, `sections/` and `tables/`;
   - a new subfolder `figures/` holding the three PDFs `fig2_example.pdf`, `fig3_f1_boxplot.pdf` and `fig4_context_length.pdf` from the project's `figures/` folder.
2. Zip `paper_upload`.
3. ☁️ Go to https://www.overleaf.com and sign in. Choose **New Project → Upload Project** and select the zip.
4. Open `main.tex`, then click **Recompile**. The compiler must be **pdfLaTeX**, which is Overleaf's default.
5. Download the PDF with the **Download PDF** button.

`main.tex` looks for the figures in both `../figures/` (inside this project) and `figures/` (the Overleaf layout above), so neither layout needs any change.

---

## 9. Presenting the project (demo plan)

Suggested length: 10–12 min.

| # | Show | How | Point to make |
|---|---|---|---|
| 1 | The problem and the gap | 2 slides: what a change point is, what TSFMs are, and the Uray et al. 2026 observation (Section 2) | "They saw forecast errors spike at changes and suggested CPD. Nobody had measured it." |
| 2 | The pipeline | Section 4.3 diagram (Fig. 1 in the paper) | Zero-shot: no model is trained. GPU work runs in Colab; everything else runs on the laptop. |
| 3 | Trustworthy evaluation | Run `python -m pytest` live, then show [table_tcpdbench.csv](results/eval/table_tcpdbench.csv) | Our metrics and baselines reproduce the published TCPDBench numbers. |
| 4 | Live detection | Web demo → **Live detection**. Show `nile` (TCPD), switch to SKAB `valve1_0`, then move the **τ** slider | Compare the detections with the dashed human labels. τ trades missed changes against false alarms. |
| 5 | Real GPU runs | Open the executed notebook and `results/tsfm_raw/manifest.json` | Real Chronos-2 and TimesFM-2.5 inference on a T4, pinned model versions, 210/210 outputs |
| 6 | Results | Web demo → **Benchmark results** | The answer to the research question, with significance tests (Section 5) |
| 7 | Insights | `figures/fig4_context_length.png`, [breakdown.csv](results/eval/breakdown.csv) | TSFMs are good on short series and weak on long ones; tuning favours classical methods; compute cost is much higher. |
| 8 | Conclusion | 1 slide (Section 5.3) | An honest, mixed result, and what to try next |

---

## 10. Key terms (for the viva)

| Term | Meaning |
|---|---|
| **Change point** | The first time step of a new segment, where the series' behaviour (level, variance, trend) changes |
| **TSFM** | Time-series foundation model: a large model pretrained on many series that can forecast new ones |
| **Zero-shot** | The model is used as downloaded, with no training or fine-tuning on our data. |
| **Forecast-residual detector** | Predict the next value; a big surprise, measured against the model's own uncertainty band, signals a change. |
| **Embedding detector** | Compare the model's internal representation of the left and right halves of a window. |
| **Context length L** | How many past values the model sees when forecasting |
| **Threshold τ** | How high a standardised score peak must be to count as a change point |
| **F1 with margin 5** | A detection counts if it is within 5 steps of a labelled change. Precision uses all annotators together; recall is averaged over annotators. |
| **Covering** | How well the predicted segments overlap the true segments (from 0 to 1) |
| **Default vs oracle** | *Default*: one fixed setting for all series, which is the fair zero-shot test. *Oracle*: the best setting per series, which is an upper bound. |
| **Zero baseline** | Predict no change points at all. It is surprisingly strong under F1, so a useful method has to beat it. |
| **Wilcoxon + Holm** | A paired statistical test over the series, corrected because several comparisons are made at once |
| **PELT / BinSeg** | Offline methods that split the series where a cost function improves most, with a penalty against too many splits |
| **BOCPD** | Bayesian online change point detection: it tracks the probability of how long the current segment has lasted. |

---

## 11. Configuration

All parameters live in [`configs/`](configs/), so no numbers are hard-coded in the code. The default settings were fixed **before** any result was seen.

| File | Contains |
|---|---|
| `datasets.yaml` | data paths, excluded series (with reasons), F1 margin |
| `models.yaml` | Hugging Face model IDs **and exact revisions** (verified) |
| `detectors.yaml` | context lengths, window sizes, threshold τ, tuning grids, Colab batch sizes |
| `baselines.yaml` | PELT/BinSeg penalties, BOCPD priors, tuning grids |

---

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `python` or `py` not found (Windows) | Reinstall Python 3.11 with **"Add python.exe to PATH"** ticked (Section 7.1), then open a **new** terminal. |
| `py -3.11`: *"No suitable Python runtime found"* | Python 3.11 isn't installed; another version is. Install 3.11 as in Section 7.1. |
| `git` is not recognized | Install git (Section 7.2), then open a **new** terminal. |
| `[WinError 206] The filename or extension is too long` during Step 2 | The project folder's path is too long. Move the project to a short path such as `C:\projects\p1-tsfm-changepoint`, delete `.venv`, and redo Step 2. |
| `git clone … p1-tsfm.bundle`: *"does not exist"* | Run it in the folder that contains `p1-tsfm.bundle` (Step 0b). |
| Step 9b lists files under `results/` or `paper/tables/` | A number changed. Run `git diff <that file>` to see it, and check that Steps 3–9 all finished without errors. |
| *"running scripts is disabled"* when activating | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again (Step 2b). |
| `ModuleNotFoundError: No module named 'cpd'` | Activate the environment, then run `pip install -e .`. |
| `streamlit` not found | Activate the environment, then run `pip install -r requirements-local.txt` again. |
| TCPD build: *"Validating the file … failed"* | A temporary website problem. Run `python scripts/download_data.py` again. |
| `bee_waggle_6 … didn't match a known checksum` | Harmless. TCPD documents tiny rounding differences on some machines. |
| Tests *skipped* | The optional TCPDBench files are missing. Run `python scripts/download_data.py --tcpdbench`. |
| Everything is very slow to start (Windows) | Antivirus scanning slows down Python imports. It's harmless, just slower the first time. |
| Web demo shows *"Missing …"* | It names the missing file and the command that creates it. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| `score_tsfm.py` says the Colab output is missing | Unzip `tsfm_raw.zip` so that `results/tsfm_raw/manifest.json` exists. |
| Colab: `Upload colab_input.zip …` assertion | The Drive folder must be exactly `My Drive/P1-tsfm/`, or edit the `DRIVE = …` line in the first code cell. |
| Colab disconnected mid-run | **Run all** again; finished files are skipped. |

---

## 13. Project status and limitations

**Status**

| Milestone | Work | State |
|---|---|---|
| M0 | Prior-art search; verify the gap, models and data ([docs/M0_FINDINGS.md](docs/M0_FINDINGS.md)) | ✅ Done |
| M1 | TCPD and SKAB loaders, label checks, tests | ✅ Done |
| M2 | Metrics (validated against TCPDBench); PELT, BinSeg, BOCPD on all 70 series | ✅ Done |
| M3 | Colab GPU notebook: Chronos-2 and TimesFM-2.5 scores (executed, 210/210) | ✅ Done |
| M4 | Scores → change points; evaluation of the TSFM detectors | ✅ Done |
| M5 | Tables, figures, Wilcoxon tests, robustness checks; results reviewed | ✅ Done |
| — | Web demo (`app.py`) and terminal demo (`scripts/demo.py`) | ✅ Done |
| — | From-scratch check (Path B): fresh clone → install → download → 22 tests → every result rebuilt identically | ✅ Passed 2026-10-19 |
| M6 | Paper in `paper/` (IEEE conference template) | 🟡 Full draft written: every section from the Abstract to the Conclusion. **Still to do:** compile it on Overleaf (Step 12; there's no local LaTeX, so it has never been compiled), fill in the author affiliation, and proof-read. |

**Limitations**
- **Only two TSFMs** (Chronos-2, TimesFM-2.5). Moirai-2 and MOMENT were dropped at M0 because of install and license problems ([docs/M0_FINDINGS.md](docs/M0_FINDINGS.md) §3).
- **One fixed threshold (τ = 3) for every series.** Adaptive thresholds might help, especially on long series.
- **The oracle is an upper bound, not a held-out result.** This follows TCPDBench.
- **SKAB has one annotator** and labels the start and end of induced faults, which are not always sharp statistical changes.
- **Runtimes are not like-for-like.** The TSFMs ran on a T4 GPU and the baselines on a laptop CPU.
- **TIRE (the IEEE base paper) is not re-run**, because its reference code has no software license.

---

## 14. Credits and citations

- **Data**
  - **TCPD**: van den Burg & Williams, *An Evaluation of Change Point Detection Algorithms* (arXiv 2003.06222), https://github.com/alan-turing-institute/TCPD
  - **SKAB**: Katser & Kozitsin, Skoltech Anomaly Benchmark (Kaggle 2020, DOI 10.34740/KAGGLE/DSV/1693952), https://github.com/waico/SKAB
- **Evaluation protocol:** TCPDBench (MIT license). Its metric code is ported verbatim in `src/cpd/evaluation/metrics.py`.
- **Models**
  - **Chronos-2**: `amazon/chronos-2`, Apache-2.0
  - **TimesFM-2.5**: `google/timesfm-2.5-200m-pytorch`, Apache-2.0
- **Gap source:** Uray, Messineo, Kwitt, Huber, *Exploring Zero-Shot Foundation Models for Multivariate Time Series Anomaly Detection*, EUROCAST 2026 ([arXiv 2607.12454](https://arxiv.org/abs/2607.12454))
- **Related work we differentiate from**
  - Choi, Shook, Dubrawski, ICLR 2026 TSALM workshop ([arXiv 2604.16428](https://arxiv.org/abs/2604.16428))
  - GAS-FM, *Change-point detection using foundation models*, ICLR 2027 submission (OpenReview `mHedcrJ0TB`)
- **IEEE base paper:** De Ryck, De Vos, Bertrand, *Change Point Detection in Time Series Data Using Autoencoders With a Time-Invariant Representation*, IEEE Trans. Signal Processing 69 (2021), pp. 3513–3524. Discussed, not re-run (its code has no license).

Detailed design decisions are in [docs/DESIGN.md](docs/DESIGN.md), the specifications in [docs/SPECS.md](docs/SPECS.md), and the architecture in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
