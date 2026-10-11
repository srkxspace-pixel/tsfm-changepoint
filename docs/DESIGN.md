# DESIGN.md: P1 Zero-Shot TSFM Change-Point Detection

Each decision is marked **[Verified]** (from a cited source, see `docs/M0_FINDINGS.md`), **[Assumption]** (reasonable engineering choice, to revisit if results contradict it) or **[Proposed]** (this project's own method). All numbers are in `configs/*.yaml`.

## D1. Forecast-residual detector (main) [Proposed, motivated by 2607.12454]

For every time step `t`:
1. **Context:** `x[max(0, t-L) : t]`. Contexts are expanding while `t < L`, so short series are still scored. Steps with `t < min_context` get score 0.
2. **Forecast:** a one-step-ahead zero-shot forecast of `x_t`, giving quantiles q10, q50, q90.
3. **Score:** `s_t = |x_t − q50| / max(q90 − q10, eps)`. This measures how surprising the observed value is relative to the model's own uncertainty, so no extra scale estimate is needed. **[Assumption]**
4. **Multivariate series** (SKAB, 4 TCPD series):
   - Chronos-2 forecasts all channels jointly (native multivariate input) **[Verified]**.
   - TimesFM-2.5 forecasts each channel separately (it is univariate only) **[Verified]**.
   - The step score is the mean of the per-channel scores **[Assumption]**.

The context length `L` has a default value and a small grid (`configs/detectors.yaml`).

## D2. Embedding-distance detector (secondary) [Proposed]

The concurrent GAS-FM work (OpenReview `mHedcrJ0TB`) already does embedding-based CPD with a more complex pipeline. Ours is the simplest version, kept only to answer RQ2.

1. **Embed every window.** Embed each window `x[s : s+w]` with Chronos-2 `embed()`. Mean-pool the token embeddings (over patches and variates) into one vector `E[s]`.
2. **Score each step.** Set `d_t = 1 − cos(E[t−w], E[t])`, the distance between the window just before `t` and the window just after it. Each window is embedded once and reused.
3. **Edges.** Steps closer than `w` to either end get score 0. A series shorter than `2w+1` gets no detections and is flagged in the results.

## D3. Score to change points (shared by all TSFM detectors)

The same steps are used for every detector, so differences come from the scores and not from post-processing:
1. Robust z-score over the series: `(s − median) / (1.4826·MAD + eps)`.
2. Centred moving average of width `smooth`.
3. `scipy.signal.find_peaks(height=tau, distance=min_gap)`. Each peak index is a detected change point.

The default `tau` and `min_gap` are fixed before any results are seen **[Assumption]**. For the oracle setting, `tau` is part of the grid.

## D4. Baselines

- **Input.** Every channel is z-scored first, as in TCPDBench **[Verified]**.
- **PELT and BinSeg** (`ruptures`, `l2` cost).
  - The default penalty is `3·log(n)`, the main term of the MBIC penalty TCPDBench uses with R `changepoint` **[Assumption: close to, not identical to, R's MBIC]**.
  - BinSeg keeps at most 5 change points by default (TCPDBench Q=5).
  - Both run on multivariate data through the multivariate `l2` cost.
  - These choices are validated against TCPDBench's stored outputs (D5).
- **BOCPD** (`src/cpd/baselines/bocpd.py`): Adams & MacKay (2007).
  - Gaussian likelihood with unknown mean and variance (Normal-Inverse-Gamma prior) and a constant hazard `1/λ`.
  - Channels are treated as independent.
  - Defaults are the TCPDBench `ocp` ones: λ=100, μ0=0, κ0=α0=β0=1 **[Verified]**.
  - Change points come from back-tracking the MAP run-length path.
- **TIRE.** Not run (no code license). Cited and discussed as the IEEE base paper.

## D5. Evaluation protocol [Verified: TCPDBench]

**Metrics**
- F1 with a margin of 5 and segmentation covering.
- The code is ported verbatim from TCPDBench `analysis/scripts/metrics.py` (MIT, with attribution). Details:
  - Index 0 always counts as a change point.
  - Precision is computed on the union of annotators; recall is the mean over annotators.
  - Predictions are clipped to the range `1 … n−2`.
- **Check:** our port must reproduce TCPDBench's stored scores from their stored change points.

**Settings**, following TCPDBench:
- **Default:** fixed parameters for every series. This is the fair zero-shot comparison.
- **Oracle:** for each series, the best score over the method's parameter grid. This is an upper bound and **not** a held-out split. F1 and covering are maximised separately, as in TCPDBench.

**TCPD averages** use the same series as TCPDBench: 32 univariate and 4 multivariate series. The quality-control series are excluded, and `uk_coal_employ` is excluded because it has missing values. Exclusions are listed in the paper.

**SKAB**
- One annotator, the same margin of 5. **[Assumption]** SKAB change points are about 55 steps or more apart at 1 Hz, so margin windows never overlap.
- The whole series is scored, because nothing is trained.
- SKAB's NAB leaderboard is cited but not compared number for number (different metric and protocol).

## D6. Statistics

- Mean and median F1 and covering per dataset.
- A Wilcoxon signed-rank test over series for each TSFM detector against the best classical baseline, Holm-corrected over the 3 comparisons.

## D7. What this project does not claim

- Not a new model, no fine-tuning, not anomaly detection, no deployment claims.
- Not the first embedding-based TSFM change-point detector (see GAS-FM).
- Inference cost is reported but not optimised.
