# Changelog

All notable changes are recorded here. Format: [Keep a Changelog](https://keepachangelog.com/);
versioning: [SemVer](https://semver.org/) as defined in `docs/ROADMAP.md`.

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
