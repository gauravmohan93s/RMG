# Methodology (v2.2)

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
For each of the last **36 month-ends** (origins; v2.2, was 24), every eligible model is fitted on
data up to that month only and forecasts the next **12** months. The last **24** origins form the
walk-forward (honest) test window, covering both the 2024–25 dip and the 2025–26 surge. The harness is the only code
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
- **Walk-forward (honest) evaluation**: for each of the last 24 origins e, every policy
  is replayed using only errors whose target month ≤ e. This measures the accuracy of
  the *procedure* with no selection bias, and is recomputed every run
  (`Evaluation/policy_comparison.csv`, Policy Comparison sheet).
- **Why `mean_all`**: in the 2026-09-27 walk-forward comparison it had the lowest median
  error (0.73× seasonal naive with 7 models, 0.67× with all 17) and beat top-3 selection,
  screened averaging and the best single model — the well-documented "forecast combination puzzle"
  (see `docs/experiments/2026-09-27_combination_policy.md`).
- **Hedge, not a hard fallback** (v2.2): where the model average loses to seasonal naive by > 5% in the
  walk-forward test, the published forecast is w(h) × same month last year + (1 − w(h)) × model average,
  with w = 0.5 next month rising to 1 at 12 months. On those 11 series the next-quarter miss fell from
  14.3% (last year only) to 11.6%, and the 6-month miss from 14.4% to 10.9%.
- **Accuracy-weighted average** (1/error, shrunk 50% to equal weights) is a tested candidate. It lost to
  equal weights (quarter error 17.16% vs 17.06%), so it is reported only.

## 5. Guardrails (`src/forecasting/guardrails.py`)
The 12-month total must lie within [0.5 × min(ref), 1.8 × max(ref)], where the references
are last-12-month actuals and last-12 × recent 3-month YoY momentum (clipped to 0.25–4).
Genuine growth passes (its momentum reference is high); model failures do not. A triggered
series is scaled to the bound (seasonal shape kept), marked Low confidence and flagged.

## 6. Reconciliation (`src/forecasting/reconcile.py`)
**Choice (v2.2).** Five ways of making regions, countries and the organisation add up are replayed
walk-forward on **next-quarter totals per level**, the decision metric:
- `none`;
- `bottom_up` (sum of regions);
- `wls_var`;
- `mint_shrink`;
- `middle_out` (each country's own forecast split to its regions by 12-month share).

The lowest error averaged over levels wins. On 2026-09 data `bottom_up` won (16.9% vs `wls_var` 17.4%,
`middle_out` 17.7%). `wls_var` had won only on monthly error.

Base forecasts for all levels are combined with the MinT family estimator
ỹ = S (Sᵀ W⁻¹ S)⁻¹ Sᵀ W⁻¹ ŷ (Wickramasuriya, Athanasopoulos & Hyndman, 2019).
Default W = diagonal of one-step backtest error variances (WLS); `mint_shrink` uses the
Schäfer-Strimmer shrunk covariance when every node has complete residuals. Negative
bottom values are set to 0 and re-aggregated, so outputs are coherent and non-negative.

## 7. Forecast ranges (`src/forecasting/ranges.py`, v2.2)
Ranges are built from the past errors of the forecasts **as published** (regions summed bottom-up),
r = log(1+actual) − log(1+forecast), for each month ahead (window h±1) and for next-3/6/12-month
totals (from the totals' own errors, since monthly ranges do not add up).

- **Two candidate constructions**
  - *empirical*: recency-weighted quantiles of the series' own errors (half-life 6 origins), which
    allows skew;
  - *centred*: symmetric around the forecast, ± z × robust spread. The spread is the
    recency-weighted median absolute deviation × 1.4826, shrunk toward the pooled spread of
    comparable series (8 pseudo-observations).

  Series with fewer than 8 own errors use the pool of the same metric and level (regions as
  fallback).
- **Chosen on evidence.** Both constructions are replayed at each of the last 12 origins, fitted only
  on errors realised by then. They are scored with the **interval score** (Gneiting & Raftery 2007),
  a proper scoring rule: width plus 2/α × distance of any miss. Per level group
  (Organisation + Country / Region) and window, the lower mean score is used. On 2026-09 data,
  *centred* won for every aggregate window; regions use *empirical* for short horizons and *centred*
  for longer ones.
- **Symmetric calibration.** If the chosen construction's test coverage is off target by more than 5
  points (N ≥ 30), one width factor in [0.8, 1.5] scales both sides equally. A tail is never shifted
  on its own.
- **Why v2.1's method was replaced.** v2.1 widened each tail separately until it covered the last 12
  walk-forward months. Those months were one regime (a surge the models under-forecast), so a past
  bias became a permanent upside tail: the organisation quarter was −14% / +45% and one country
  was 0 / +304%. Bias corrections of the point forecast were tested and did not win (§7a), and the
  forecast errors changed sign between 2024–25 and 2025–26. Encoding that bias in the ranges would
  contradict the evidence.
- **Ranges come from the forecasts as published.** For any reconciliation method, the backtest is
  reconciled origin by origin with weights from errors known at the time (`reconcile.reconcile_backtest`).
- **Effective sample size.** Consecutive k-month windows overlap, so n of them count as n/k independent
  observations.
  - A series' own spread is used only with at least 4 independent windows.
  - The width factor needs at least 30 independent test results.
  - Longer windows without enough own evidence use location-scale pooling: the series' own quarter
    volatility × the pooled shape of k-month/quarter errors of comparable series.
  - Ranges never narrow (in %) as the window lengthens.
  - For the **Region group** (which includes single-region destinations such as Europe and Asia), the
    series' own long-window spread is blended with that estimate on the log scale, weighted by its
    independent windows (n/k against 2). Tested: Region 12-month interval score 5.27 → 5.07, coverage
    unchanged. For countries it made ranges worse, so it is not applied there (§12,
    `docs/experiments/2026-10-25_long_window_blend.md`).

  See `docs/experiments/2026-09-27_long_horizon_and_ranges.md`.
- **What width means.** A range can only be as narrow as the forecast is accurate. The 50% range is
  "most likely" and the 80% range is for planning. Narrower honest ranges require better point
  accuracy, for example leading indicators (ROADMAP).
- The earlier empirical/conformal functions remain in `intervals.py` for reference and tests.

## 7a. Statistical overlay (`src/forecasting/bias_correction.py`)
The data-driven replacement for guessed policy multipliers. Recent bias of each series =
median log error of the last 6 realised months (1–3 months ahead); empirical-Bayes
(James-Stein) shrinkage towards 0 so noisy series barely move; applied as a persistent
("level") or fading ("decay", ×0.85 per month) multiplier capped to ×0.67–×1.5. Both are
replayed walk-forward and adopted under the common rule (§7f). On 2026-09 data it was adopted (`decay`):
next-quarter miss for countries 18.8% → 14.6%, regions 22.3% → 20.4%. Until v2.2 it was judged on a 1%
monthly-WAPE gain, which flipped between runs; the test and every series' multiplier are reported
(Statistical Overlay sheet).

## 7c. Submissions from receipts: the funnel (`src/forecasting/funnel.py`)
Submissions follow receipts (12-month log changes correlate 0.97 at lag 0). The funnel forecast is
Submitted[t] = Σ w_l · Received[t−l], l = 0, 1, 2:
- the weights are non-negative least squares on the last 12 realised months, re-fitted at each origin;
- receipts are actual up to the origin and the published receipt forecast after it.

Model, funnel and a 50/50 blend compete under §7f. On 2026-09 data the funnel won (next-quarter miss:
organisation 13.0% → 10.6%, countries 20.9% → 19.0%).

## 7d. Direction signal from the latest vintages (`src/forecasting/vintage.py`)
Before a new forecast, the misses of the latest forecasts on the months already known are measured:
- the forecasts made 5, 4 and 3 months ago (full quarters);
- the forecasts made 2 and 1 months ago (the latest months).

The misses are recency-weighted and read at the country total. The published receipt forecasts of the
country's regions are corrected by half that gap, and totals are re-summed. Gated by §7f: it passed in
the first test (countries 18.4% → 14.2%) but not once the hedge and funnel removed most of the lag, so
it is currently off and re-tested every month. The pack's Direction Check page shows these misses.

## 7e. Calendar and festival effects (`src/forecasting/calendar_fx.py`)
For each 26th–25th cycle the difference from the same cycle last year is measured: working days and
moving festivals (Diwali, Holi, Eid al-Fitr). Effects are fitted walk-forward on organisation and
country receipt misses (+5.3% per working day, Diwali −8.5%, Eid −11.6%; bounded). The same factor
applies to every series in a month, so totals still add up. The component is adopted on month-level
accuracy for months 1–3 (≥ 3% better) and must not worsen quarters. On 2026-09 data the gain was 1.3%,
so it is off. **Maintenance:** extend the festival dates every year.

## 7f. One decision rule and the quarterly lock (`src/governance/decisions.py`)
**Rule.** Every component above, and the reconciliation and point estimate, is judged on next-quarter
totals per level, averaged over levels, and adopted only when at least 3% better. The calendar effects
are judged on single months instead.

**Lock.** Choices are re-decided in Jan, Apr, Jul and Oct. Between reviews a choice changes only if the
alternative is at least 6% better. Every decision, its scores and its reason are stored in
`method_decisions.json` and shown on the Method Decisions sheet. This stops numbers jumping because a
close test tipped the other way. In v2.1 one flip moved the organisation forecast by 12% with identical
data.

## 7g. Pipeline stages after submission (`src/forecasting/stages.py`)
**Definition.** Stages are those of AppStatus2 (CF Staff View). Each ApplicationStageId is flagged with
the milestones it implies, and an application crosses a milestone the first time it reaches any stage
flagged with it, counted per 26th–25th cycle:
- Offer;
- Deposit;
- Visa document: CAS (UK), Funds/COE (Australia), AIP (NZ), I-20 (USA), else any visa document;
- Visa applied;
- Visa granted;
- outcomes: Visa refused, Enrolled (enrolment confirmed, commission-earning), Deferred.

**Forecast.** Each stage comes from an earlier stage (from the published submission forecast onwards). The
upstream is chosen per destination by walk-forward quarter error, among upstreams whose fitted conversion
is at most 105%; a higher "conversion" means that stage really comes after this one there:
stage[t] = Σ w_l · upstream[t−l], l = 0..3, with non-negative weights fitted on the last 12 months at
each origin.

**Choice.** The chain, same month last year and their blend are replayed walk-forward and scored on
next-quarter totals (organisation and countries). The chain or blend must be at least 3% better than
last year's pattern.

**Ranges** are the 10/90 and 25/75 quantiles of the chosen variant's own errors by horizon band.
Change vs last year is shown only where the same months last year had at least 150 applications.

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

## 9b. Human signals and the forecast record (`src/analysis/signals.py`, `src/reports/forms.py`)
Each run writes one input file per destination (`Forms/<YYYY-MM>/To_Send`) with two sheets:
- staffing rates, or headcount (a rate is then derived from the last 12 months' volume);
- market expectations for the next 3 months, next to the model's forecast and range.

`collect-forms` validates the returned files and writes `Settings.xlsx` (backed up first). Manager
expectations (`Settings.xlsx!Human_Inputs`) are recorded each run, shown against the
model and scored against actuals; a person is blended in (inverse-error weight) only after ≥ 6
scored months in which they beat the model. Forecast vs Actual: every month's actual against the
forecasts made 1–12 months earlier — from the walk-forward test (what the method would have said
at the time) plus every live forecast stored in `prediction_history_v2.parquet`.

## 10. Reporting labels
- **Reliability** (`src/forecasting/reliability.py`) — measured the way the numbers are used: the typical
  (median) absolute % error of the **next-3-month total** of the **published** (reconciled) forecast in the
  walk-forward test. High ≤ 10%, Medium ≤ 10–20%, Low > 20%, New = fewer than 4 testable quarters,
  Review = guardrail-capped. Reliability describes how predictable a market is; every series already uses
  the most accurate method found, so "Low" never means a weaker method was used.
  (v2.1 labelled single-month, unreconciled errors, which overstated uncertainty for decisions.)
- **Ranges**: see §7. "Most likely" = 50% range, "plan for" = 80% range; a 12-month range wider than the
  forecast itself is labelled "direction only: plan on the next quarter".
- **YoY** always compares identical calendar months (like-for-like).
- **Fiscal periods**: FY starts in April; outlook = realised actuals + forecast remainder. Its range
  covers the remainder only, so it narrows every month as actuals arrive. Only complete quarters/years
  are shown.
- **Revisions**: "vs last run" = change against the previous month's forecast for the same target months
  (from the prediction history). Revisions are shown, not smoothed: damping them made countries worse
  (13.4% → 14.6%).
- **Staffing FTE**: computed per region with that region's applications-per-FTE rate
  (`Settings.xlsx!Staffing_Productivity`, most specific row wins); country / organisation FTE = sum of
  their regions' FTE (rates differ by country group and region).

## 11. Governance
- `run_manifest.json`: pipeline version, git commit (and dirty flag), data SHA-256, row
  counts, full config, timings, invariant checks.
- `prediction_history_v2.parquet`: every forecast issued; later runs join actuals to report
  realised WAPE, bias and interval coverage by horizon (Track Record sheet). Horizon is
  counted from the last actual month.
- `method_decisions.json`: the locked method choices (§7f).
- `Snapshots/extract_<YYYY-MM>.parquet`: the extract as used each month. Later runs measure how much a
  month's count grew after it was first reported (late entries). This is flagged in Data Quality when ≥ 3
  months show ≥ 3% uplift, and the latest month is checked for being partly loaded (red banner).
- **Publishing**: each `all` run copies the reports to `Outcomes/<YYYY-MM> Forecast Pack/` (kept) and
  `Outcomes/Latest/` for executives.
- Hard invariants (the run fails rather than publish): no negatives, no past-dated rows,
  point inside both intervals, 95% ⊇ 80%, regions sum to totals.

## 12. Known limitations
- **Turning points.** Turning points driven by policy, marketing or visa shocks cannot be foreseen
  from receipts history. The models lag them: +9% in the 2024–25 fall, −6% to −17% in the 2025–26 surge.
  Stage data (offers, deposits, CAS / Funds / AIP / I-20) is the planned lever (ROADMAP v2.3); the
  extract is prepared in `fetch_data.py`.
- **12-month country totals.** Typical misses are 34–42% for Australia, Canada, New Zealand and the
  USA, against 14% for the UK and 15% for the organisation. Plan countries on the next quarter.
- **Intake.** Forecasts are by month of application, not by target intake (the intake view is
  descriptive).
- The log back-transform gives median-type forecasts (slightly below the mean for skewed counts).
