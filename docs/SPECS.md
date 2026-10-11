# SPECS.md: P1 Zero-Shot TSFM Change-Point Detection

## 1. Objective

Measure whether zero-shot TSFM-based detectors match classical CPD methods on real, human-labelled change points (TCPD, SKAB).

## 2. Research questions

- **RQ1.** Do forecast-residual detectors built on Chronos-2 and TimesFM-2.5 locate change points as accurately as PELT, BinSeg and BOCPD?
- **RQ2.** How does a Chronos-2 embedding-distance detector compare with the forecast-residual detectors?
- **RQ3.** Where do TSFM detectors help or fail? Broken down by dataset (TCPD vs SKAB), by univariate vs multivariate series, and by series length.
- **RQ4.** What is the compute cost per series of each method?

## 3. Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | Load TCPD and SKAB into one unified format, keeping every annotator's label set |
| FR-2 | Run PELT, BinSeg and BOCPD on every series in the default and oracle settings |
| FR-3 | Colab notebook 01 produces per-timestep residual scores (Chronos-2, TimesFM-2.5) and embedding-distance scores (Chronos-2) for every series |
| FR-4 | Convert all TSFM scores to change points with one shared procedure |
| FR-5 | Compute F1 (margin 5) and segmentation covering per series and per method, using TCPDBench's definitions |
| FR-6 | Produce paper tables and figures from `results/eval/` only |
| FR-7 | Record runtime per series for every method |

## 4. Experiments

| ID | Experiment | Output |
|---|---|---|
| E1 | Classical baselines, default and oracle | `results/eval/baselines.csv` |
| E2 | TSFM detectors (2 residual + 1 embedding), default and oracle | `results/eval/tsfm.csv` |
| E3 | Breakdown by dataset, dimensionality and length bucket | `results/eval/breakdown.csv` |
| E4 | Runtime per series | `results/eval/runtime.csv` |

The context-length sensitivity is read off the oracle grid, which includes context length, so it needs no separate ablation run.

## 5. Paper deliverables

- Table 1: dataset statistics.
- Table 2: main results, F1 and covering, every method, default and oracle, both datasets.
- Table 3: Wilcoxon signed-rank test of each TSFM detector against the best baseline (Holm-corrected).
- Figure 1: pipeline diagram.
- Figure 2: example series with true change points, TSFM score curve and detected points.
- Figure 3: per-series F1 distribution (box plot) per method.

## 6. Acceptance criteria

- Every TCPD and SKAB series is processed by every method; any skipped series is listed with a reason.
- Our metric code reproduces TCPDBench's stored scores on their stored change points.
- Every table cell traces to a CSV in `results/eval/`.
- The executed Colab notebook is committed with visible outputs and a `manifest.json`.
- Tests pass on CPU: loaders, metrics (against hand-computed examples and TCPDBench), peak picking, BOCPD.

## 7. Out of scope

Fine-tuning any TSFM; new architectures; anomaly detection; real-time deployment; running TIRE; Moirai-2, MOMENT and SWaT.
