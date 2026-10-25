# Roadmap & Versioning

Versioning follows [Semantic Versioning](https://semver.org/):
**MAJOR** = methodology or output-schema change that alters numbers or breaks consumers,
**MINOR** = new capability that is backward compatible, **PATCH** = fixes.
The version lives in `src/__init__.py` and is stamped into every run manifest,
the prediction history and the reports. Each release is a git tag `vX.Y.Z`;
work happens on `release/*` or `feature/*` branches and is recorded in `CHANGELOG.md`.

## v2.0.0 — Production forecasting engine (released 2026-09-27)

Goal: numbers an executive can use, with evidence for every choice.

| # | Work item | Fixes review finding |
|---|---|---|
| 1 | Data layer: zero-fill true gaps, series status (Active / Emerging / Inactive), data-quality register | 9, 10, 15 |
| 2 | Hierarchy: Region → Country → Organisation nodes for Received & Submitted | 2 |
| 3 | Model registry with versioned, leak-free models: Seasonal Naive, Seasonal Naive × YoY growth, ETS (damped), Theta, SARIMA airline, TSB, Global LightGBM (direct multi-horizon) | 3, 4, 13, 14 |
| 4 | Rolling-origin backtest: 24 origins × 12-month horizon; models only ever see data ≤ origin | 1, 3, 11 |
| 5 | Scoring: horizon-balanced MAE vs seasonal-naive benchmark; policies `mean_all` (default — won the walk-forward comparison, see docs/experiments) and `top_k` | 1, 11 |
| 6 | Honest walk-forward evaluation of every policy each run (no selection bias) + benchmark fallback | 3 |
| 7 | Empirical, calibrated prediction intervals (80/95%) per horizon; monthly and 3/6/12-month totals | 2, 14 |
| 8 | Reconciliation (WLS / MinT-shrink) so regions sum to countries and organisation | 2 |
| 9 | Guardrails: 12-month plausibility ratio with documented fallback; invariants asserted | 1, 16 |
| 10 | Governed policy overlay: only *Approved*, owned, dated multipliers applied; baseline always shown | 6 |
| 11 | Period summary on fiscal quarters with like-for-like YoY (same months only) | 5 |
| 12 | Governance: run manifest (version, git SHA, data hash, config), prediction history v2, track record by horizon | 7, 8, 18, 19 |
| 13 | Executive Excel workbook + PDF brief (deterministic narrative), evidence sheets | 17 |
| 14 | pytest suite: leakage, intervals, reconciliation, guardrails, overlay, data prep | 18 |
| 15 | Config in `config/pipeline.toml`; legacy v1 modules removed (recoverable from tag `v1.0.0`) | 12, 15 |

## v2.1.0 — Full fair tournament, seasonality & staffing, statistical overlay, track record (released 2026-09-27)
Requested 2026-09-27. Every item keeps the v2 rule: identical origins, horizons, data and
metric for every model; nothing fitted on data after a forecast origin.

| # | Work item |
|---|---|
| 1 | All v1 models back in the tournament, re-implemented leak-free: LastYear_MoM, YoY_MoM (+seasonal prior), Rolling_MoM, AvgMonthly_MoM, Prophet, AutoARIMA, XGBoost (global direct), neural net (global direct MLP — TensorFlow is not installed, v1's LSTM already fell back to this), Croston. Plus suggested: STL+ETS, robust seasonal median |
| 2 | Policy `mean_screened`: average of models that beat the benchmark on past data only — robust when weak models join the pool; compared walk-forward with the other policies |
| 3 | Internal signals in the global ML models: Received→Submitted funnel lags, academic-calendar/intake-season flags |
| 4 | Seasonality & staffing: STL seasonal index per organisation/country, High / Medium / Low months, year-by-year stability, 12-month staffing calendar with capacity (upper-range) figures |
| 5 | Statistical overlay: recent level-shift (bias) correction estimated from realised errors with empirical-Bayes shrinkage; applied only if it improves walk-forward accuracy. Judgemental overlay stays optional and governed |
| 6 | Track record: simulated history backfilled from backtests + live history; Forecast-vs-Actual by month and vintage; feeds item 5 |
| 7 | Human signals: `Human_Inputs` template (expected volumes + reason) recorded every run and scored; combined statistically once ≥6 months of track record exist |
| 8 | Reports: fully labelled charts (seasonality heatmap, H/M/L profile, forecast vs actual, model league table); human-friendly Excel (plain-English headers, native charts, staffing calendar, grouped tabs) |

## v2.2.0 / v2.2.1 — Evidence-chosen ranges, deeper tests, funnel, monthly pack (released 2026-10-25)
- Shipped: evidence-chosen ranges, 24-origin walk-forward, health check, funnel model for submissions,
  fiscal-year outlook with narrowing range, monthly visual pack + country PDFs.

## v2.3.0 — Stage (pipeline) data (code released 2026-10-25; first data with the next fetch)
**Done:**
- Stage definitions from AppStatus2: offer, deposit, country-specific visa document, visa applied, visa
  granted.
- First-reach extract per cycle (`fetch_data.py`, SQL verified on synthetic history).
- Mapping (`transform-data`).
- Chain forecast gated against last year (`stages.py`).
- Pack page and workbook sheets.

**Next, once real stage history has been fetched:**
1. Review the Stage Evidence sheet: is the chain chosen, and are conversion rates and lags plausible per
   country?
2. Booking curve by intake: compare cumulative stage counts for an upcoming intake with the same point
   last year, as an early signal of a strong or weak intake.
3. Stage rows in the country input forms, if market leads want to give stage expectations.

## v2.4.0 — External drivers
- Intake extract and descriptive arrival view shipped in v2.1 (the forecast stays monthly).
- Test the intake mix of recent months as a leading signal for the global ML models (keep only if it
  improves walk-forward accuracy).
- Leading indicators as regressors (enquiries/leads, counsellor capacity, marketing, visa
  approval rates); each must prove uplift in the rolling-origin harness before use.
- Scenario engine: overlays as distributions rather than point multipliers.

## v2.5.0 — Operations (renumbered from the original plan)
- Scheduled monthly run after the 25th-cycle close; e-mail/Teams notification with the PDF brief.
- Drift monitor: alert when realised 80% coverage leaves 70–90% or WAPE degrades.
- Optional: statsforecast / hierarchicalforecast back-ends for speed at larger scale.

## Contribution rules (for future development)
1. A new model is a class in `src/forecasting/models.py` with `name`, `version`,
   `min_history` and `forecast(y, h, start_month)`; register it in `LOCAL_MODELS`
   and enable it in `config/pipeline.toml`. It competes automatically — never
   hand-pick winners.
2. A model may only be kept if it beats the seasonal-naive benchmark in the
   walk-forward evaluation on enough series (see `Accuracy_by_Horizon`).
3. Any change that moves numbers ⇒ bump MINOR (or MAJOR if methodology changes),
   update `CHANGELOG.md`, and compare against the previous release on the same data.
4. `pytest` must pass before tagging.
