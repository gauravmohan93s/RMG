# Changelog

All notable changes are recorded here. Format: [Keep a Changelog](https://keepachangelog.com/);
versioning: [SemVer](https://semver.org/) as defined in `docs/ROADMAP.md`.

## [2.2.0] — 2026-09-27

Requested: ranges that support decisions, reliability that matches the published numbers, an
executive-readable brief, and region-specific staffing productivity.

### Changed
- **Forecast ranges chosen on evidence** (`src/forecasting/ranges.py`): empirical vs centred
  robust construction, chosen per level group and window by the interval score in a sequential
  (origin-by-origin, leak-free) replay. Only a symmetric width factor (0.8–1.5) is applied, so
  ranges no longer carry a past bias as a one-sided tail.

  | Sequential test, 2026-09 data | before | after |
  |---|---|---|
  | Organisation receipts, next-quarter 80% range | −14% / +45% | about ±10% |
  | Organisation receipts, 12-month 80% range | −21% / +15% | about ±17% |

  Aggregate 80% coverage in the test was 80–87%.
- **Reliability** is measured on next-3-month totals of the published (reconciled) forecast:
  High ≤ 10%, Medium ≤ 20%, Low > 20% typical error; guardrail-capped series are marked Review.
- **Staffing FTE per region**, using region- and country-specific applications per counsellor
  (`Staffing_Productivity` sheet); country and organisation FTE are sums of their regions.
- **Point estimate `auto`**: a smearing-corrected mean is used only if it improves quarter totals
  across levels by ≥ 3% walk-forward. On 2026-09 data it did not (organisation better, countries
  worse), so the median is used.
- The executive brief was rewritten in plain language; the Excel workbook gains a Range Method
  Test sheet.

### Fixed
- The published range always contains the point forecast (fixes a round-off at zero width).

### Security
- The database password was removed from all local git history. Credentials are read only from the
  environment or `.env`.

### Evidence
- `docs/experiments/2026-09-27_training_window.md`: excluding 2020–21 COVID months was rejected.
- A uniform organisation-level bias factor was tested offline and gave mixed results, so it was
  not adopted.

## [2.1.0] — 2026-09-27

Requested improvements: all original models in a fair tournament, statistically derived
overlays instead of guesses, seasonality for staffing, intake view, prediction record,
human signals, better charts and a human-friendly workbook.

### Added
- **Fair 17-model tournament**: all v1 models re-implemented leak-free (LastYear_MoM, AvgMonthly_MoM,
  YoY_MoM, Rolling_MoM, AutoARIMA, ProphetStyle, Croston, XGBoost, neural net) plus STL_ETS; identical
  origins, horizons, data and metric for every model. Global ML models use the Received↔Submitted
  funnel as an internal signal.
- Policy `mean_screened`; policy comparison now includes all policies.
- **Statistical overlay**: recent-bias correction with empirical-Bayes shrinkage, adopted only if it
  wins walk-forward (on 2026-09 data it did not — it would have worsened error 20.2% → 21.2%).
- **Reconciliation `auto`**: bottom-up / WLS / MinT replayed walk-forward, most accurate used
  (2026-09: bottom-up; organisation receipts error 11.4% vs 18.2% forecasting the total alone).
  Walk-forward accuracy, calibration and Forecast vs Actual now use the reconciled (published) numbers.
- **Seasonality & staffing**: robust-STL seasonal index, High/Medium/Low months, year-by-year stability,
  12-month staffing calendar with capacity figure and optional FTE.
- **Measured policy events** with a non-overlapping placebo significance test.
- **Forecast vs Actual** record (walk-forward vintages + live forecasts) and **human inputs**
  (recorded, scored on identical months, blended only after beating the model for ≥ 6 months).
- **Intake extract** (`Intake`, `InYear`) and descriptive intake arrival view (which months each
  intake's applications arrive in). The forecast itself stays monthly.
- Reports: labelled seasonal/heatmap/forecast-vs-actual charts, plain-English Excel headers,
  colour-coded tab groups, native Excel charts, new analysis sheets.

### Changed
- Backtest runs local and global models in one pool, heaviest first; AutoARIMA re-selects its order
  every 6 origins (causal) — 17-model run 35 → 17 minutes.
- `fetch_data.py`: one incremental routine for both extracts; **database credentials only from
  environment / git-ignored `.env`** (hard-coded defaults removed — rotate the old password).

### Fixed (independent review)
- Placebo windows no longer overlap; human-vs-model scoring uses identical months; stale human
  inputs (> 120 days) expire.

## [2.0.1] — 2026-09-27

### Fixed
- Interval calibration was tested on errors that had also been used to build the intervals
  (found in independent code review). Now split-conformal (CQR): quantiles for the check are fitted
  only on errors realised before the walk-forward window; per-tail additive offsets are applied.
  Out-of-sample 80% coverage moved from 53–91% (before) to 80–88% (after).
- Calibration only widens a tail, never narrows it (one 12-month regime is not enough to remove
  downside risk).
- 3/6/12-month total ranges calibrated the same way (standardised units).
- Fiscal-period "actual to date" treats missing months as unknown instead of 0.
- Complete-block check requires horizons exactly 1..k.

## [2.0.0] — 2026-09-27

Complete rebuild of the forecasting engine after the v1 review (`docs/REVIEW_2026-09-27.md`).
**Numbers change materially**: on the Sep-2026 extract the next-12-month receipts forecast is
355,971 (80% range 247k–480k, +20% YoY) versus 732,850 (2.4× last year) in v1.

### Added
- `config/pipeline.toml`: single, versioned run configuration; typed loading with validation.
- Data layer: true-zero gaps, series status (Mature / Emerging / Inactive), data-quality register,
  Region → Country → Organisation hierarchy.
- Model registry (versioned): SNaive, SNaiveGrowth, ETS, Theta, SARIMA airline, TSB, GlobalLGBM (direct multi-horizon).
- Rolling-origin backtest (24 origins × 12 months) with leakage impossible by construction.
- Horizon-balanced scoring; forecast policies `mean_all` (default) and `top_k`; walk-forward
  evaluation of every policy each run (Policy Comparison); benchmark fallback.
- Empirical, calibrated 80/95% intervals per month and for 3/6/12-month totals (location-scale pooling).
- WLS / MinT-shrink reconciliation — regions add up exactly to countries and the organisation.
- Plausibility guardrails based on last year's level and recent momentum.
- Governed policy overlay (Approved, owned, dated rows only; baseline always kept).
- Fiscal-period outlook (actual to date + forecast) with like-for-like YoY.
- Governance: `run_manifest.json` (version, git commit, data SHA-256, config, timings),
  `prediction_history_v2.parquet` and realised track record by horizon.
- Executive Excel workbook (15 sheets: planning + evidence) and PDF brief (deterministic narrative).
- 39 pytest tests incl. a future-shock leakage test and an end-to-end run.
- Docs: METHODOLOGY, ROADMAP, policy experiment, review.

### Changed
- CLI: `forecast` replaces `eda`/`load-data`/`run-models`/`evaluate`; `report` builds from a saved run; new `version`.
- Dependencies: added lightgbm; removed prophet, pmdarima, xgboost, tensorflow, rich, tqdm, requests, python-dotenv.

### Removed
- v1 modules `src/models/*`, `src/eda.py`, `src/loader.py`, `src/reconciliation.py`,
  `src/policy_matrix.py`, `src/intake_calendar.py`, `src/tracking.py`,
  `src/reports/final_pdf_reports.py`, `test.py` — recoverable from tag `v1.0.0`.
- Ollama/LLM report commentary (replaced by deterministic, number-checked text).
- Hard-coded policy multipliers (now a Draft template requiring owner approval).

### Fixed (v1 review findings)
Selection-vs-use horizon mismatch; compounding month-on-month models; bands detached from
forecasts; backtest leakage (ensemble weights, LightGBM lags, EDA bands, full-series bounds,
bidirectional interpolation); interpolated zero months; past-dated forecasts for dead series;
generic selection reasons; mis-sorted governance scorecard; mislabelled intake summary.

## [1.0.0] — 2026-09-26
Legacy pipeline (per-model files, 3-month holdout "decathlon" selection). Tagged for reference.
