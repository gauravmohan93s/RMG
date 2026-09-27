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

## v2.1.0 — Intake-true forecasting (next)
- Add `Intake`/`Intake-In-Year` to the SQL extract (`fetch_data.py`) and forecast
  applications *by target intake*, replacing month-of-application periods.
- Lead-time curves: share of an intake's applications received k months before start.

## v2.2.0 — Drivers & scenarios
- Leading indicators as regressors (enquiries/leads, counsellor capacity, visa
  approval rates) once history is clean; test uplift in the same rolling-origin harness.
- Scenario engine: overlays as distributions rather than point multipliers.

## v2.3.0 — Operations
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
