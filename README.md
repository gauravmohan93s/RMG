# RMG Application-Flow Forecasting (v2.5)

Monthly 12-month forecast of applications **Received** and **Submitted** for the organisation,
every destination country and every region. Each forecast comes with ranges that have been tested
against what happened, evidence for every method choice, and an executive pack that reads the same way
every month.

## What you get each month
| Output | For |
|---|---|
| `Outcomes/Latest/` (and `Outcomes/<YYYY-MM> Forecast Pack/`) | Executives: the monthly pack, one PDF per destination, the workbook |
| `Reports/RMG_Monthly_Pack_<date>.pdf` | Leadership pack, read as a story: where we are heading, last month and accuracy, where growth comes from, pipeline and the coming intakes, capacity, can we trust it; then two pages per destination down to its regions |
| `Reports/RMG_Forecast_<date>.xlsx` | Planners and analysts: every number plus all evidence sheets (method decisions, tests, calibration) |
| `Forms/Live/<Country>.xlsx` | Country teams: a live sheet shared once. Region × month grid of actual, model and their view, plus staffing; read at every run and rolled forward after it |
| `Final_Forecast/final_forecast.csv` | Downstream systems: every region / country / organisation × month with 50/80/95% ranges |
| `run_manifest.json` | Audit: code version, data fingerprint, settings, method decisions |

## How it works
- **Models.** Seventeen methods compete under identical, leak-free conditions: benchmarks, ETS, Theta,
  SARIMA/AutoARIMA, STL, a Prophet-style model, intermittent-demand models, the v1 month-on-month models
  and three global machine-learning models. Each is re-run as if standing at 36 past month-ends.
- **Testing.** Every way of combining and correcting them is replayed walk-forward over the last 24
  months, using only information available at the time.
- **Adoption rule.** A component is used only if it makes next-quarter totals at least 3% more accurate.
  Choices are reviewed each quarter. Components:
  - the equal-weight average of all models;
  - a hedge with last year's level for series where models lose to it;
  - a recent-bias correction;
  - submissions forecast from receipts (the funnel);
  - a direction signal from the latest forecasts;
  - calendar and festival effects.
- **Adding up.** Regions sum to countries and the organisation.
- **Ranges** come from the errors actually made, with the construction chosen by a proper scoring rule.

Full detail: [docs/METHODOLOGY.md](docs/METHODOLOGY.md); evidence: [docs/experiments/](docs/experiments/).

## Quick start
```bash
python -m venv .venv_3.12
.venv_3.12\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest                          # 90 tests, synthetic data only
python run_pipeline.py all                # fetch -> transform -> forecast -> report -> publish -> forms (~25 min)
```

**Other commands:**
- `fetch-data`, `transform-data`, `forecast`, `report`, `publish`, `sync-forms`, `refresh-forms`, `notify`, `version`
- options: `--run-date YYYY-MM-DD`, `--skip-fetch`, `--skip-transform`
- stage-data discovery (read-only): `python fetch_data.py --discover-stages`

Database credentials (`.env`, git-ignored), mappings, forms and scheduling: [SETUP_GUIDE.md](SETUP_GUIDE.md).

## Repository layout
```
config/pipeline.toml               all modelling settings (versioned; stamped into each run)
src/data/prepare.py                extract -> panel: zeros, series status, hierarchy, latest-month check
src/forecasting/models.py          model registry (17 models)
src/forecasting/backtest.py        rolling-origin backtest (leak-free by construction)
src/forecasting/selection.py       scoring, combination policies, walk-forward evaluation
src/forecasting/bias_correction.py recent-bias correction and point-estimate test
src/forecasting/funnel.py          submissions from receipts
src/forecasting/stages.py          offers, deposits, visa stages: conversion chain from submissions
src/forecasting/vintage.py         direction signal from the latest forecasts
src/forecasting/calendar_fx.py     working-day and festival effects per 26th-25th cycle
src/forecasting/reconcile.py       bottom-up / WLS / MinT / middle-out, chosen on quarter totals
src/forecasting/ranges.py          ranges chosen by interval score, sequential and out-of-sample
src/forecasting/reliability.py     reliability on next-quarter totals of the published forecast
src/forecasting/engine.py          end-to-end run + artefacts
src/analysis/                      seasonality and staffing, signals (human inputs, events), intake view, intake running indicator
src/governance/                    manifest, prediction history, decision lock, extract snapshots
src/reports/                       story pack (pack.py) and its blocks (pdf.py), workbook, charts, live sheets, publishing, e-mails
tests/                             pytest suite (synthetic data only)
fetch_data.py, transform_data.py   SQL extract (26th-25th cycles, incl. stage data) and ID mapping
```

## Versioning
Semantic versioning; the version (`src/__init__.py`) is stamped into every output. Releases are git
tags (`v1.0.0` legacy pipeline, `v2.0.0` this engine, `v2.5.0` current). See
[CHANGELOG.md](CHANGELOG.md) and the plan in [docs/ROADMAP.md](docs/ROADMAP.md).

## Extending
Add a model as a `LocalModel` subclass in `src/forecasting/models.py`, register it and enable it in
`config/pipeline.toml`. It enters the tournament automatically and must earn its place in the
walk-forward test. Any new correction component must pass the common decision rule
(METHODOLOGY §7f) and be documented in `docs/experiments/`.
