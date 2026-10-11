# ARCHITECTURE.md: P1 Zero-Shot TSFM Change-Point Detection

## 1. Folder structure

```
p1-tsfm-changepoint/
├── CLAUDE.md, README.md
├── requirements-local.txt        # CPU only, pinned
├── requirements-colab.txt        # chronos-forecasting, timesfm (Colab GPU), pinned
├── configs/
│   ├── datasets.yaml             # sources, paths, exclusions
│   ├── models.yaml               # verified HF model IDs + revisions (M0)
│   ├── detectors.yaml            # context L, window w, peak picking, grids
│   └── baselines.yaml            # PELT/BinSeg penalties, BOCPD priors, grids
├── data/
│   ├── raw/                      # downloaded TCPD and SKAB, untouched (gitignored)
│   └── processed/                # <dataset>/<series_id>.npz + series.json
├── src/cpd/
│   ├── data/                     # loaders.py
│   ├── baselines/                # classical.py (ruptures), bocpd.py
│   ├── detectors/                # peaks.py (score → change points)
│   ├── evaluation/               # metrics.py (TCPDBench port), stats.py
│   └── reporting/                # tables.py, figures.py
├── notebooks/colab/01_tsfm_scores.ipynb   # replaced by the executed copy
├── results/
│   ├── classical/                # grid.csv: change points per series × method × config
│   ├── tsfm_raw/                 # scores exported from Colab (per model/detector, per series)
│   ├── tsfm_cp/                  # grid.csv: change points after peak picking
│   └── eval/                     # metric tables (CSV)
├── figures/
├── scripts/                      # prepare_data.py, run_classical.py, score_tsfm.py, evaluate.py, make_report.py
├── tests/
├── paper/
└── docs/
```

## 2. Pipeline

```
raw data ──► prepare_data (local) ──► data/processed/
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
          classical baselines (local CPU)           Colab notebook 01 (GPU)
                     │                                         │
                     ▼                                         ▼
          results/classical/grid.csv               results/tsfm_raw/  (scores)
                     │                                         │
                     │                          peak picking (local) ──► results/tsfm_cp/grid.csv
                     └────────────────────┬────────────────────┘
                                          ▼
                       evaluate (local) ──► results/eval/*.csv ──► tables, figures, paper
```

## 3. Local vs Colab boundary

| Runs locally (CPU) | Runs in Colab (GPU) |
|---|---|
| Dataset download and unification | Chronos-2 and TimesFM-2.5 rolling one-step forecasts |
| PELT, BinSeg, BOCPD | Chronos-2 window embeddings |
| Peak picking on exported scores | |
| Metrics, statistics, tables, figures | |

The boundary is the folder `results/tsfm_raw/`. Colab produces raw per-timestep scores, and everything after that is local.

## 4. Unified data format

- `data/processed/<dataset>/<series_id>.npz` holds `values`, a float64 array of shape `(T, C)`, raw and not z-scored. float64 is used because float32 loses precision on large values such as `us_population`.
- `data/processed/series.json` maps each `series_id` to `{dataset, n_obs, n_dim, annotations: {annotator_id: [cp indices]}}`.
  - TCPD keeps every annotator's set separately.
  - SKAB has one annotator.
- Excluded series and the reasons are listed in `configs/datasets.yaml`.

The same files are uploaded to Colab, so both sides read identical inputs.

## 5. Colab notebook contract

`notebooks/colab/01_tsfm_scores.ipynb` must:
1. **Header cell:** purpose, inputs, outputs, expected runtime on a free T4, date written.
2. **Setup cell:**
   - Mount Google Drive, where the input zip (`configs/` + `data/processed/`) has been uploaded.
   - Install the pinned `requirements-colab.txt`.
   - Print the GPU name and the torch and package versions.
3. **Config cell:** load the same `configs/*.yaml` the local code uses. No hard-coded parameters.
4. **Resumable loop:** process series one by one and skip any whose output already exists, so a disconnected session can be restarted.
5. **Outputs:**
   - Write `results/tsfm_raw/<model>_<detector>/<series_id>.npz`, holding one score array per grid value plus the runtime.
   - Write `manifest.json`.
   - Zip the folder for download.
6. **Final cell:** print a completeness check (series expected vs written) and the zip path.

**Hand-off:** Srikar runs the notebook, downloads the executed `.ipynb` and the results zip, replaces the notebook in `notebooks/colab/` with the executed copy, and unzips the results into `results/tsfm_raw/`.

## 6. Run manifest (`manifest.json`)

```json
{
  "notebook": "01_tsfm_scores.ipynb",
  "models": {"chronos-2": "amazon/chronos-2@<sha>", "timesfm-2.5": "google/timesfm-2.5-200m-pytorch@<sha>"},
  "packages": {"chronos-forecasting": "<version>", "timesfm": "<version>", "torch": "<version>"},
  "gpu": "Tesla T4",
  "seed": 42,
  "series_expected": 0,
  "series_written": 0,
  "started": "...", "finished": "..."
}
```
