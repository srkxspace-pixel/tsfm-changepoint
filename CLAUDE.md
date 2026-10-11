# CLAUDE.md: P1 Zero-Shot TSFM Change-Point Detection

Read this file fully before any work. Then read `docs/SPECS.md`, `docs/ARCHITECTURE.md`, `docs/DESIGN.md` and `docs/M0_FINDINGS.md`.

## 1. Project identity (fixed, do not change)

- **Title:** Do Zero-Shot Time-Series Foundation Models Detect Change Points? Forecast-Residual and Embedding-Distance Detectors vs. Classical CPD on Real-World and Industrial Sensor Data
- **Track / Level / Domain:** Track B (Research) · B.Tech final-year (intermediate) · AI & Machine Learning
- **Research question:** Can pretrained time-series foundation models (TSFMs), used zero-shot with no training, locate change points in real labelled data as accurately as established change-point detection (CPD) methods?
- **Gap source:** https://arxiv.org/abs/2607.12454 (Uray et al., EUROCAST 2026). TimesFM is run zero-shot on SWaT, and forecast error spikes at the start and end of an anomaly (qualitative, Fig. 4b). The Abstract and §6 name CPD as a promising direction.
- **Work to differentiate from:**
  - https://arxiv.org/abs/2604.16428 (ICLR 2026 TSALM workshop). Supervised probes on Chronos-2 / MOMENT embeddings for shift-type classification. Synthetic AR(1) only, no localisation.
  - Concurrent: "Change-point detection using foundation models" (ICLR 2027 submission, OpenReview `mHedcrJ0TB`, code github.com/iclr456/anonymous_cpd, "GAS-FM"). Training-free CPD from frozen TSFM **embeddings**, with no forecast-residual detector.
  - Our main contribution is therefore the **forecast-residual** detector, scored with the TCPDBench protocol on TCPD and on SKAB.
- **IEEE base paper:** De Ryck, De Vos, Bertrand, *Change Point Detection in Time Series Data Using Autoencoders With a Time-Invariant Representation*, IEEE Trans. Signal Processing, Vol. 69, pp. 3513–3524. https://ieeexplore.ieee.org/document/9447888. It is cited and discussed but **not run**, because the authors' code has no license (decision 2026-10-11).

## 2. Compute rule (highest priority)

- **Never use or assume the local machine's GPU.** Do not read local hardware specs, do not call `torch.cuda` in local code, do not set device from local detection.
- **Local code is CPU-only:** data loading, classical baselines, metrics, tables, figures, tests.
- **Every TSFM step needs a GPU and therefore runs in Colab.** Write it as a notebook in `notebooks/colab/` following the notebook contract in `docs/ARCHITECTURE.md` §5.
- **Hand-off:** when a Colab notebook is ready, stop and tell Srikar which notebook to run. He runs it, replaces the notebook in the repo with the executed copy (outputs visible), and places the exported files in `results/`.
- **Never fabricate results.** No placeholder numbers, no "expected" values, no synthetic stand-ins for model outputs. If a required file in `results/` is missing, stop and ask for the run.

## 3. Scope lock (revised 2026-10-11 with Srikar's approval)

Exactly these, nothing added without Srikar's approval:

- **TSFMs:** Chronos-2 (forecast residual and embeddings) and TimesFM-2.5 (forecast residual). Moirai-2 and MOMENT were dropped. See `docs/M0_FINDINGS.md` §3.
- **Detectors:** forecast-residual scoring (main); embedding-distance scoring with Chronos-2 (secondary comparison).
- **Baselines:** PELT and Binary Segmentation (`ruptures`), BOCPD (own implementation). TIRE is cited only.
- **Datasets:** TCPD (Turing Change Point Dataset) and SKAB. SWaT is not used.
- **No model training or fine-tuning of any TSFM.**
- **Level:** B.Tech final-year project. Keep it simple: smallest code that answers the research question credibly, and no extra infrastructure.

## 4. Engineering rules

- Python 3.11 locally, dependencies pinned in `requirements-local.txt` (CPU).
- Colab runs its default runtime (Python 3.13), and each notebook installs `requirements-colab.txt`.
- All parameters in `configs/*.yaml`; no magic numbers in code.
- Fixed seeds. Every Colab run writes a small `manifest.json` (ARCHITECTURE §6).
- Hugging Face model IDs and revisions are pinned in `configs/models.yaml` (verified at M0). Do not guess IDs.
- Every number in `paper/` must trace to a file in `results/`.
- Tests in `tests/` must pass on CPU before any notebook is handed off.

## 5. Milestones and checkpoints

Stop at each ★ and wait for Srikar.

| ID | Work | Where |
|---|---|---|
| M0 | Prior-art search, verify gap / models / data; ★ report findings | Done 2026-10-11 (`docs/M0_FINDINGS.md`) |
| M1 | Dataset loaders for TCPD and SKAB into one series format, label checks, tests | Local CPU |
| M2 | Metrics (TCPDBench F1 and covering, validated against TCPDBench's stored scores); PELT, BinSeg and BOCPD on all series, default and oracle settings | Local CPU |
| M3 | ★ Colab notebook `01_tsfm_scores.ipynb`: residual scores (Chronos-2, TimesFM-2.5) and Chronos-2 embedding distances | Colab |
| M4 | Score-to-change-point conversion and evaluation of TSFM detectors from saved scores | Local CPU |
| M5 | Tables, figures, Wilcoxon test; ★ results review | Local CPU |
| M6 | Paper draft in `paper/` (IEEE conference template) | Local |
