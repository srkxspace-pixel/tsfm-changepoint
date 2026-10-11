# M0 Findings: gap, prior art, models, data (checked 2026-10-10)

All facts below were checked against primary sources on 2026-10-10. Nothing is from memory.

## 1. Gap source and differentiation papers

| Paper | What it actually is | Claim in CLAUDE.md |
|---|---|---|
| arXiv 2607.12454, Uray, Messineo, Kwitt, Huber, "Exploring Zero-Shot Foundation Models for Multivariate Time Series Anomaly Detection" (EUROCAST 2026, LNCS) | TimesFM 1.2 zero-shot on SWaT (51 ch). Residual detector F1 "close to zero"; embeddings + Isolation Forest F1 0.48 (baselines 0.78–0.83). Fig. 4b shows error spikes at the entry and exit of one anomaly (qualitative only). The Abstract and §6 Conclusion name CPD as a promising direction. | **Accurate.** Cite the Abstract and §6, not the §5 "Future Work" paragraph (it does not mention CPD). |
| arXiv 2604.16428, Choi, Shook, Dubrawski, "Non-Stationarity in the Embedding Space of Time Series Foundation Models" (ICLR 2026 TSALM workshop) | Supervised linear probes on Chronos-2 / MOMENT / TOTEM embeddings classify the shift type (mean / variance / trend) in synthetic AR(1) windows. The change is always at the window midpoint. No localisation, no real data, no CPD metrics. Simple statistics beat the TSFMs on strong shifts. | **Accurate.** |
| IEEE base paper, De Ryck et al., TIRE, IEEE TSP 69:3513–3524 (2021), DOI 10.1109/TSP.2021.3087031, arXiv 2008.09524 | Autoencoder CPD. Evaluated with AUC on simulated AR data, bee dance, HASC and well log. Code at github.com/deryckt/TIRE is TensorFlow, last updated 2021, **with no license**. `tire-cpd` 0.1.3 on PyPI (MIT, by Zhenxiang Cao of the MC-TIRE line) pins tensorflow==2.12, numpy==1.23.5. | Verified. |

## 2. Prior art: the gap is open but narrower

**Concurrent work (biggest threat): "Change-point detection using foundation models"**
- ICLR 2027 submission, anonymous, OpenReview id `mHedcrJ0TB`, submitted 2026-09-18.
- Code: https://github.com/iclr456/anonymous_cpd.
- Method (GAS-FM) is training-free:
  1. Take frozen **embeddings** from sliding windows.
  2. Apply TV denoising.
  3. Project to one dimension.
  4. Run the SCAN statistical change detector.
- Models: Chronos-T5, Chronos-Bolt, **Chronos-2**, Moirai-1, Moirai-MoE, **MOMENT**, TimesFM-1.0, **TimesFM-2.5**, plus DINO for video.
- It has **no forecast-residual detector**.
- Its datasets are not named in the code. The OpenReview page is behind a bot check, so the PDF has not been read: **Srikar should open it in a browser**.
- Its metric is plain F1 with a tolerance, not the TCPDBench multi-annotator F1/covering.

**Other related work**
- *Native Prediction Errors for Regime Detection* (OpenReview PpwzEfEMRu, ICLR 2027 submission). It uses trained world models, not TSFMs, but argues that flexible predictors "absorb" shifts. This is a risk for the residual idea.
- *In-Context Learning Under Regime Change* (arXiv 2604.16988). Theory on trained transformers. Low overlap.
- TSB-AD (NeurIPS 2024 D&B), plus several 2024–2026 studies, use TSFM residuals for **anomaly** detection. None reports change points. TSFMs are often matched by simple baselines.
- Trained representation CPD: KL-CPD, TS-CP2, TIRE, MC-TIRE, RECURVE. None uses a pretrained TSFM.
- 98 papers citing TCPDBench since 2024 were checked. **None uses a TSFM.**
- SKAB's CPD leaderboard has no foundation-model entry.

**What is still open**
1. **Forecast-residual CPD** with current TSFMs. This is the direct test of 2607.12454's boundary-spike observation.
2. TSFM detectors scored with the **TCPDBench protocol**: multi-annotator F1 and covering, against the published classical results.
3. TSFMs on the **SKAB** CPD task.
4. How TSFMs behave on **short** real series, which make up most of TCPD.

The embedding-distance arm is no longer novel on its own. It overlaps with GAS-FM and can only serve as a comparison.

## 3. Models (Hugging Face API, 2026-10-10)

| Model | HF ID @ sha | License | Package | Colab default (Py 3.13) |
|---|---|---|---|---|
| Chronos-2 (120M) | `amazon/chronos-2` @ `29ec3766d36d6f73f0696f85560a422f50e8498c` | Apache-2.0 | chronos-forecasting 2.3.2 | Works. Native multivariate input, `embed()`, context up to 8192. |
| TimesFM-2.5 (231M incl. quantile head) | `google/timesfm-2.5-200m-pytorch` @ `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` | Apache-2.0 | timesfm[torch] 3.0.2 | Works. Univariate. |
| Moirai-2 (small only, 11M) | `Salesforce/moirai-2.0-R-small` @ `30f43ff08c8494f4943ae1521e9d4e94a0fbb389` | CC-BY-NC-4.0 | uni2ts 2.0.0 | **Fails.** Pins torch<2.5, numpy 1.26 and scipy 1.11, so it needs the pinned Colab 2026.07 runtime (Py 3.12). Univariate only. |
| MOMENT-1-large (346M) | `AutonLab/MOMENT-1-large` @ `ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc` | MIT | momentfm (PyPI 0.1.4 is stale and does not install; GitHub main `38f7310` does) | Install from git. Untested on transformers 5.x. |

Colab facts:
- Colab's default runtime is now Python 3.13, torch 2.11 (cu130).
- The T4 is still supported.
- All four checkpoints are 1.4 GB or less in fp32.

## 4. Datasets and protocol

**TCPD**
- 37 evaluation series plus 5 simulated quality-control series (excluded).
- 33 series are univariate and 4 are multivariate.
- Each series has 5 annotations, drawn from a pool of 8 annotators.
- 10 series must be built from external sources (yfinance, archive.org).
- **Lengths:** 9 series are under 100 points, 19 under 256 and 29 under 512. The range is 15–991 points.

**TCPDBench**
- F1 uses margin 5. Index 0 is always counted as a change point. Precision is computed on the union of annotators; recall is macro-averaged over annotators.
- Covering is the mean over annotators.
- There are two settings:
  - **Default:** package defaults.
  - **Oracle:** per-series maximum over a parameter grid, scored on the same labels. This is **not** a held-out split, so DESIGN D5 is wrong on this point.
- The "zero" baseline (no detections) scores F1 0.645 on univariate series.
- Published averages use 32 univariate and 4 multivariate series.
- PELT and BinSeg come from the R `changepoint` package; BOCPD from R `ocp`.
- Per-series outputs for every method are in the repository (`analysis/output/`), so they can be reused for validation.

**SKAB v0.9**
- 34 labelled CSV files of 745–1,327 rows each, with 8 sensor channels at roughly 1 Hz.
- 129 change points in total, at the start and end of anomalies and their transition phases.
- License: GPL-3.0.
- Its own leaderboard uses a NAB score; the best entry is MSCRED at 32.42.
- It is real data from a lab Industrial-IoT testbed.

**SWaT**
- Free, but access requires an application to iTrust, and the requester's name is published.
- Openly downloadable industrial alternative: Petrobras 3W (CC BY 4.0).

## 5. Corrections needed in the docs

| Doc | Says | Should say |
|---|---|---|
| DESIGN D5 | "Tuned" means chosen on a held-out split, following TCPDBench | TCPDBench's tuned setting is an **oracle** (per-series grid maximum). |
| ARCH §4 | "all five annotators" | 5 annotations per series, from a pool of 8 annotators. |
| D2 / scope | Moirai-2 is a third independent family that installs normally | It is small-only, CC-BY-NC, univariate, and needs its own old runtime. |
| Title | "on Industrial Sensor Data" | Only SKAB is industrial. TCPD is mixed real-world data. |
| D4 | Use the authors' TIRE code if found | Their code has no license. Use the MIT-licensed `tire-cpd` package or reimplement. |
