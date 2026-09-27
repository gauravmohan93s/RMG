# Methodology (v2)

This document is the scientific reference for the pipeline: what is done, why, and
where it lives in the code. Every number a report shows can be traced to a rule here.

## 1. Data (`src/data/prepare.py`)
- **Source**: `monthly_final_summary.xlsx` — applications Received (by submitted date)
  and Submitted (stage change) per Country group × Region per reporting month. A
  month is the **26th–25th cycle** named after its end month; only fully closed
  cycles are extracted (`fetch_data.py`).
- **Missing months are zeros.** The SQL only emits rows with activity, so an absent
  month is a true zero — never interpolated (v1 interpolated, inventing volume).
- **Series status** (per Country × Region, both metrics jointly):
  *Inactive* — no activity in the last 6 months → not forecast (listed in Data Quality);
  *Emerging* — fewer than 18 months since first activity → pooled-seasonality method;
  *Mature* — enters the model tournament.
- **Hierarchy**: active regions → country totals (countries with ≥2 active regions)
  → organisation. Each level is forecast in its own right, then reconciled (§6).
  Reporting history (YoY comparisons) includes inactive series, so a closed region
  correctly shows as lost volume.

## 2. Candidate models (`src/forecasting/models.py`) — 17, all competing fairly
| Family | Model | Idea | Min. months |
|---|---|---|---|
| Benchmark | SNaive | same month last year | 12 |
| Benchmark | SNaiveGrowth | same month last year × trailing 12-month growth, applied once | 24 |
| Statistical | ETS | Holt-Winters on log1p, damped/no trend, AICc choice | 24 |
| Statistical | Theta | Theta method on log1p, additive seasonal adjustment | 24 |
| Statistical | SARIMA_Airline | SARIMA(0,1,1)(0,1,1)12 on log1p | 36 |
| Statistical | AutoARIMA (v1 Model06) | pmdarima stepwise AICc search, D=1; order re-selected every 6 origins (causal) | 36 |
| Statistical | STL_ETS | robust STL seasonal adjustment + damped ETS | 24 |
| Statistical | ProphetStyle (v1 Model05) | Prophet's structure: piecewise-linear trend with L1-regularised changepoints + Fourier yearly seasonality (prophet 1.2.1 is incompatible with pandas 3, so fitted with scikit-learn) | 24 |
| Intermittent | TSB | Teunter-Syntetos-Babai rate | 6 |
| Intermittent | Croston (v1 Model11) | Croston with SBA bias correction | 6 |
| v1 month-on-month | LastYear_MoM (Model01), AvgMonthly_MoM (Model04), YoY_MoM (Model02), Rolling_MoM (Model03) | v1 logic re-implemented leak-free: every ratio from training data only | 24 |
| Machine learning | GlobalLGBM, GlobalXGB (v1 Model07), GlobalMLP (replaces v1 Model08 LSTM) | one model per horizon trained across all series (direct, not recursive); features: 12 scaled lags, YoY, the other metric's recent history (Received↔Submitted funnel), calendar month | 12 |

Fairness rules: every model of a series sees the same training data, origins, horizons and
metric; hyper-parameters are fixed in code/config, never tuned on test months; a model that
fails at any origin is removed for that series and logged (no silent advantage).
v1's LSTM ran as an MLP fallback because TensorFlow is not installed; the global MLP is the
honest equivalent. Results of the 2026-09-27 tournament (median error vs seasonal naive):
ETS 0.80, TSB 0.80, Theta 0.83, LastYear_MoM 0.84, GlobalXGB 0.86, GlobalLGBM 0.87 …
AutoARIMA 0.99, Croston 1.05, SNaiveGrowth 1.06, AvgMonthly_MoM 1.18 (Model Catalogue sheet).

## 3. Rolling-origin backtest (`src/forecasting/backtest.py`)
For each of the last **24 month-ends** (origins), every eligible model is fitted on data
up to that month only and forecasts the next **12** months. The harness is the only code
that slices history, so leakage is impossible by construction — and `tests/test_backtest.py`
proves it by shocking future values and checking that past forecasts do not change.
All models of a series are scored on identical (origin, horizon) pairs.

## 4. Scoring and the forecast policy (`src/forecasting/selection.py`)
- **Horizon-balanced MAE**: MAE at each horizon h = 1..12, averaged over horizons, so
  a model cannot win on next month and fail at month 9 (v1's flaw).
- **Policy** — how a series' forecast is formed from its candidates:
  `mean_all` (default): equal-weight average of every eligible model;
  `mean_screened`: average of the models that beat the benchmark on past data;
  `top_k`: average of the best k that beat the seasonal-naive benchmark.
- **Walk-forward (honest) evaluation**: for each of the last 12 origins e, every policy
  is replayed using only errors whose target month ≤ e. This measures the accuracy of
  the *procedure* with no selection bias, and is recomputed every run
  (`Evaluation/policy_comparison.csv`, Policy Comparison sheet).
- **Why `mean_all`**: in the 2026-09-27 walk-forward comparison it had the lowest median
  error (0.73× seasonal naive with 7 models, 0.67× with all 17) and beat top-3 selection,
  screened averaging and the best single model — the well-documented "forecast combination puzzle"
  (see `docs/experiments/2026-09-27_combination_policy.md`).
- **Benchmark fallback**: a series whose policy loses to seasonal naive by >5% in the
  walk-forward test uses seasonal naive instead.

## 5. Guardrails (`src/forecasting/guardrails.py`)
The 12-month total must lie within [0.5 × min(ref), 1.8 × max(ref)], where the references
are last-12-month actuals and last-12 × recent 3-month YoY momentum (clipped to 0.25–4).
Genuine growth passes (its momentum reference is high); model failures do not. A triggered
series is scaled to the bound (seasonal shape kept), marked Low confidence and flagged.

## 6. Reconciliation (`src/forecasting/reconcile.py`)
Base forecasts for all levels are combined with the MinT family estimator
ỹ = S (Sᵀ W⁻¹ S)⁻¹ Sᵀ W⁻¹ ŷ (Wickramasuriya, Athanasopoulos & Hyndman, 2019).
Default W = diagonal of one-step backtest error variances (WLS); `mint_shrink` uses the
Schäfer-Strimmer shrunk covariance when every node has complete residuals. Negative
bottom values are set to 0 and re-aggregated, so outputs are coherent and non-negative.

## 7. Prediction intervals (`src/forecasting/intervals.py`)
- **Monthly**: quantiles of relative (log1p) errors of the combined forecast from the
  backtest, per series and horizon (window h±1; pooled by metric/level if < 8 errors),
  forced to contain the point and to widen with horizon.
- **Calibration (split-conformal / CQR)**: quantiles are re-fitted on errors realised before the
  walk-forward window; on the walk-forward errors (never used for fitting) each tail receives an
  additive log-scale offset so it is missed at most (1-L)/2 of the time, per metric × level ×
  horizon band. Offsets only widen (never narrow) a tail. Coverage before/after is reported.
- **3/6/12-month totals** (monthly quantiles are not additive): location-scale pooling —
  each series' own monthly error scale × the pooled distribution of standardised block
  errors of all series of the metric, then conformally calibrated in the same way. One series has
  too few 12-month windows to estimate its own range; series differ in volatility, so raw pooling
  is also wrong.

## 7a. Statistical overlay (`src/forecasting/bias_correction.py`)
The data-driven replacement for guessed policy multipliers. Recent bias of each series =
median log error of the last 6 realised months (1–3 months ahead); empirical-Bayes
(James-Stein) shrinkage towards 0 so noisy series barely move; applied as a persistent
("level") or fading ("decay", ×0.85 per month) multiplier capped to ×0.67–×1.5. Both are
replayed walk-forward and adopted only if walk-forward WAPE improves by ≥ 1% (relative);
the test and every series' multiplier are reported (Statistical Overlay sheet).

## 7b. Measured policy events (`src/analysis/signals.py`)
For each event in `Settings.xlsx!Policy_Events`, impact = actual / counterfactual − 1 over the
next 6 months, where the counterfactual is an average of SNaive, SNaiveGrowth, ETS and Theta
fitted only on data before the event. A placebo test at 12 earlier dates gives the normal
error range; "significant" only if the impact lies outside its central 80%. Measured impacts
— not guesses — inform any judgemental overlay.

## 8. Emerging series
Recent 6-month deseasonalised level × the seasonal index of the parent (country if it is
mature, else organisation). Intervals use the pooled region-level errors. Always Low confidence.

## 9. Business overlay (`src/forecasting/overlay.py`)
Judgemental adjustments are applied only if Approved, owned, dated, within validity and
within ×0.5–×1.5. They are applied to regions and re-aggregated; the statistical baseline
is always reported alongside (`Forecast` vs `Forecast_Adjusted`).

## 9a. Seasonality and staffing (`src/analysis/seasonality.py`)
Robust STL (period 12) on log1p volume over the last 4 years → multiplicative index by
calendar month (mean 1.00). High ≥ 1.10, Low ≤ 0.90, else Medium. Stability: each month's
ratio to the centred 2×12 moving average (trend removed) per year; Consistency = share of years
in the same class. Staffing calendar: forecast / average forecast month, class, capacity plan =
80% high; optional FTE via `seasonality.apps_per_fte_month`.

## 9b. Human signals and the forecast record (`src/analysis/signals.py`)
Manager expectations (`Settings.xlsx!Human_Inputs`) are recorded each run, shown against the
model and scored against actuals; a person is blended in (inverse-error weight) only after ≥ 6
scored months in which they beat the model. Forecast vs Actual: every month's actual against the
forecasts made 1–12 months earlier — from the walk-forward test (what the method would have said
at the time) plus every live forecast stored in `prediction_history_v2.parquet`.

## 10. Reporting labels
- **Confidence**: High ≤ 15%, Medium ≤ 30%, Low > 30% walk-forward WAPE; emerging and
  guardrail-scaled series are always Low. These are communication thresholds, not model inputs.
- **YoY** always compares identical calendar months (like-for-like).
- **Fiscal periods**: FY starts in April; outlook = realised actuals + forecast remainder;
  only complete quarters/years are shown.

## 11. Governance
- `run_manifest.json`: pipeline version, git commit (and dirty flag), data SHA-256, row
  counts, full config, timings, invariant checks.
- `prediction_history_v2.parquet`: every forecast issued; later runs join actuals to report
  realised WAPE, bias and interval coverage by horizon (Track Record sheet). Horizon is
  counted from the last actual month.
- Hard invariants (the run fails rather than publish): no negatives, no past-dated rows,
  point inside both intervals, 95% ⊇ 80%, regions sum to totals.

## 12. Known limitations
- Univariate: turning points driven by policy, marketing or visa shocks cannot be foreseen
  from history alone — in the 2025→2026 rebound all methods under-forecast. Use the
  overlay for known events; v2.2 adds leading indicators.
- Forecasts are by month of application, not by target intake (v2.2 — needs the intake field in the extract).
- The log back-transform gives median-type forecasts (slightly below the mean for skewed counts).
