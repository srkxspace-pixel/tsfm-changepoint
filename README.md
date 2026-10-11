# Do Zero-Shot Time-Series Foundation Models Detect Change Points? Forecast-Residual and Embedding-Distance Detectors vs. Classical CPD on Real-World and Industrial Sensor Data

This project is a B.Tech final-year research project (Track B). It tests whether pretrained time-series foundation models, used **zero-shot**, can find change points as accurately as classical change point detection (CPD) methods. The models are Chronos-2 and TimesFM-2.5, and the classical methods are PELT, BinSeg and BOCPD. Evaluation uses real labelled data, TCPD and SKAB, under the TCPDBench protocol.

Start with `CLAUDE.md`, then `docs/SPECS.md`, `docs/ARCHITECTURE.md`, `docs/DESIGN.md` and `docs/M0_FINDINGS.md`.

**Compute rule:** local code runs on CPU only. The only GPU step is `notebooks/colab/01_tsfm_scores.ipynb`, which runs in Google Colab.

## Reproduce

```bash
py -3.11 -m venv .venv && .venv/Scripts/python -m pip install -r requirements-local.txt && .venv/Scripts/python -m pip install -e .
```

1. **Data** (into `data/raw/`, untouched):
   - `git clone https://github.com/alan-turing-institute/TCPD`, then run its `build_tcpd.py collect`. This fetches 10 series from archive.org and yfinance. On Windows, run it with Unix newlines, otherwise the checksums fail.
   - `git clone https://github.com/waico/SKAB`.
   - `git clone --filter=blob:none --sparse https://github.com/alan-turing-institute/TCPDBench`, then sparse-checkout `analysis/`. This is only needed for the validation tests.
2. **Unify:** `python scripts/prepare_data.py` writes `data/processed/` (36 TCPD and 34 SKAB series).
3. **Tests:** `python -m pytest`. Among other things, they check that our metrics and baselines reproduce TCPDBench.
4. **Classical baselines** (CPU, a few hours; can be split into shards): `python scripts/run_classical.py <i> 4` for i in 0–3. Output goes to `results/classical/`.
5. **TSFM scores** (Colab T4):
   - Run `python scripts/make_colab_bundle.py`, then upload `colab_input.zip` to `MyDrive/p1_tsfm/`.
   - Run `notebooks/colab/01_tsfm_scores.ipynb`.
   - Unzip `tsfm_raw.zip` into `results/tsfm_raw/`.
6. **Evaluate:** run `python scripts/score_tsfm.py`, then `python scripts/evaluate.py`, then `python scripts/make_report.py`. This writes `results/eval/*.csv`, `paper/tables/*.tex` and `figures/*.pdf`.

## Layout

| Path | What |
|---|---|
| `src/cpd/data/loaders.py` | TCPD and SKAB loaders in one format, plus z-scoring |
| `src/cpd/evaluation/metrics.py` | F1 with margin and covering (verbatim TCPDBench port) |
| `src/cpd/baselines/` | PELT/BinSeg (`ruptures`, MBIC) and BOCPD (port of R `ocp`) |
| `src/cpd/detectors/` | residual and embedding scores (model-agnostic), and peak picking |
| `notebooks/colab/01_tsfm_scores.ipynb` | GPU inference (Chronos-2, TimesFM-2.5) |
| `results/` | every number in the paper comes from here |
| `paper/` | IEEE conference paper (`main.tex`) |
