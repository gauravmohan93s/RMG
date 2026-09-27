# RMG Application-Flow Forecasting (v2.1)

Monthly 12-month forecast of applications **Received** and **Submitted** for the organisation,
every destination country and every region — with calibrated ranges, evidence for every
modelling choice, and executive-ready reports.

## What you get each month
| Output | For |
|---|---|
| `Reports/RMG_Executive_Brief_<date>.pdf` | Leadership: headline numbers, ranges, key messages, country outlook, how far to trust it |
| `Reports/RMG_Forecast_<date>.xlsx` | Planners & analysts: forecasts, staffing calendar, seasonality, forecast vs actual, and all evidence sheets |
| `Final_Forecast/final_forecast.csv` | Downstream systems: every region/country/organisation × month with 80/95% ranges |
| `run_manifest.json` | Audit: code version, data fingerprint, settings |

## How it works (one paragraph)
Seventeen methods — benchmarks, ETS, Theta, SARIMA/AutoARIMA, STL, a Prophet-style model,
intermittent-demand models, all four v1 month-on-month models (rebuilt leak-free) and three global
machine-learning models (LightGBM, XGBoost, neural net) — are each re-run as if standing at 24 past
month-ends, forecasting the next 12 months every time, under identical conditions. Ways of combining
them are replayed walk-forward with only information available at the time; the equal-weight
average of all methods won and is used (re-tested every run). A statistical overlay corrects
recent bias only if that proves itself walk-forward. Ranges come from the errors actually made,
conformally calibrated. Regions, countries and the organisation are reconciled so they add up.
Seasonality (High/Medium/Low months), a staffing calendar, forecast-vs-actual history, manager
inputs and measured policy-event impacts are reported alongside. Full detail: [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Quick start
```bash
python -m venv .venv_3.12
.venv_3.12\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest                          # 59 tests
python run_pipeline.py all                # fetch -> transform -> forecast -> report (~20 min, 17 models)
```
Other commands: `fetch-data`, `transform-data`, `forecast`, `report`, `version`
(`--run-date YYYY-MM-DD`, `--skip-fetch`, `--skip-transform`). Setup details, database
credentials, mappings and scheduling: [SETUP_GUIDE.md](SETUP_GUIDE.md).

## Repository layout
```
config/pipeline.toml            all modelling settings (versioned; stamped into each run)
config/policy_overlay_template.csv
src/data/prepare.py             extract -> panel: zeros, series status, hierarchy
src/forecasting/models.py       model registry
src/forecasting/backtest.py     rolling-origin backtest (leak-free)
src/forecasting/selection.py    scoring, policies, walk-forward evaluation
src/forecasting/intervals.py    empirical calibrated intervals
src/forecasting/reconcile.py    WLS / MinT reconciliation
src/forecasting/guardrails.py   plausibility bounds, invariants
src/forecasting/overlay.py      governed business adjustments
src/forecasting/bias_correction.py  statistical overlay (recent-bias correction, walk-forward tested)
src/analysis/seasonality.py     seasonal index, High/Medium/Low months, staffing calendar
src/analysis/signals.py         forecast vs actual, human inputs, measured policy events
src/forecasting/engine.py       end-to-end run + artefacts
src/governance/                 run manifest, prediction history, track record
src/reports/                    Excel workbook, PDF brief, charts, narrative
tests/                          pytest suite (synthetic data only)
fetch_data.py, transform_data.py  SQL extract (26th-25th cycles) and ID mapping
```

## Versioning
Semantic versioning; the version (`src/__init__.py`) is stamped into every output.
Releases are git tags (`v1.0.0` = legacy pipeline, `v2.0.0` = this engine). See
[CHANGELOG.md](CHANGELOG.md), the plan in [docs/ROADMAP.md](docs/ROADMAP.md) and the review that
motivated v2 in [docs/REVIEW_2026-09-27.md](docs/REVIEW_2026-09-27.md).

## Extending
Add a model as a `LocalModel` subclass in `src/forecasting/models.py`, register it, enable it
in `config/pipeline.toml` — it enters the tournament automatically and must beat the benchmark
in the walk-forward test to earn its place. See the contribution rules in the roadmap.
