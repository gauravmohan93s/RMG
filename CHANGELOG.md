# Changelog

All notable changes are recorded here. Format: [Keep a Changelog](https://keepachangelog.com/);
versioning: [SemVer](https://semver.org/) as defined in `docs/ROADMAP.md`.

## [2.3.1] — 2026-10-26

### Fixed
- **Stage upstream chosen per destination.** Each stage is now forecast from the earlier stage (Submitted,
  Offer, Deposit, ...) that tested best for that destination, among upstreams converting at most 105%.
  Previously one upstream was used for every destination, giving impossible conversions where the
  milestone order differs (Visa document from Deposit: Australia 210%, USA 133%). Australia and the USA
  now chain from Offer (39%). Walk-forward quarter error is equal or better (Visa applied 26% vs 39% for
  last year's pattern). The November run's stage files were recomputed; the main forecast is unchanged.
- **Country pages fit on one page** (chart heights reduced): the pack is 19 pages again, not 30.
- **Publish survives locked files**: when OneDrive or an open PDF locks `Outcomes/Latest`, files are
  overwritten in place instead of failing.
- **Run manifest records the git commit** when the scheduler has no git executable on PATH (read from `.git`).

### Changed
- `publish` also creates the country input forms (`Forms/<YYYY-MM>/To_Send`), so the orchestrator's
  separate steps deliver them.

## [2.3.0] — 2026-10-25

### Added
- **Pipeline stages after submission**: Offer, Deposit, Visa document, Visa applied, Visa granted.
  - **Definitions** come from `Region - Team.xlsx`, sheet AppStatus2 (CF Staff View). A milestone is
    crossed the first time an application reaches any stage flagged with it.
  - **Visa document is country-specific:** CAS (UK), Funds stage / COE (Australia), AIP (NZ), I-20 (USA).
  - **Extract** (`fetch_data.py`): the stage map is a join table, so one status counts for every
    milestone it implies. Verified on synthetic history in SQLite (`tests/test_stage_sql.py`).
  - **Forecast** (`src/forecasting/stages.py`): a chain from the published submission forecast, with
    lagged conversion fitted by non-negative least squares on the last 12 months. It competes with same
    month last year and a blend on next-quarter totals and is used only if at least 3% better. Ranges come
    from each stage's own walk-forward errors.
  - **Reports**: "Pipeline stages" page in the pack; Stage Forecast and Stage Evidence sheets.
  - **Robustness**: it runs automatically when `stage_final_summary.xlsx` exists. It is supplementary:
    a stage-data problem is logged in the manifest and never stops the main forecast.

## [2.2.1] — 2026-10-25

### Added
- `transform-data` also maps the stage extract to `stage_final_summary.xlsx` (skipped until
  `STAGE_GROUPS` is filled).

### Changed
- README and METHODOLOGY brought up to date with every v2.2 component (hedge, funnel, direction signal,
  calendar effects, decision rule and quarterly lock, snapshots, forms, publishing); ROADMAP v2.3 lists
  what is prepared and what remains for stage data.

### Data note
- `Settings.xlsx` was edited on 2026-10-22: a few countries moved from "Southeast & Central Asia" to
  "Asia" and "Central & Eastern Europe". The largest effect is 22 applications in a month between these
  groups; organisation totals are unchanged. Runs from October 2026 use the new grouping.

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
- **Monthly forecast pack** (`src/reports/pdf.py`) replaces the text-heavy brief. It has a fixed
  visual layout that is refilled every month:
  - organisation dashboard;
  - destinations at a glance (volume with range and reliability colour, and a growth sense-check of
    forecast vs the last 6 months' actual growth);
  - workload heatmap;
  - track record;
  - one page per destination, also saved as a separate PDF under `Reports/Countries/`;
  - a fixed "how to read" page.

  Numbers are written on the charts and in a 12-month strip, and every page footer carries the legend.
- **Deeper, fully out-of-sample testing:**
  - backtest origins 24 → 36; walk-forward test origins 12 → 24, covering both the 2024–25 dip and
    the 2025–26 surge;
  - new `range_tested_coverage.csv`: the whole range procedure (method choice and width factor)
    replayed with only earlier realised results, so the reported coverage has no in-sample element.
- **Ranges come from the published (reconciled) forecasts for every reconciliation method.** A
  `wls_var` run had used unreconciled aggregate errors, so the organisation's 6-month range held 39%.
  Also added: effective-sample-size rules for overlapping windows, location-scale pooling for long
  windows, and monotone widening across 3/6/12 months. A 12-month planning range wider than the
  forecast itself is labelled "direction only: plan on the next quarter".
- Experiment: long-horizon blend toward last year's level was **rejected** (no consistent gain), and
  12-month country misses of 34–42% were confirmed as real (policy shocks).
  See `docs/experiments/2026-09-27_long_horizon_and_ranges.md`.
- **Funnel model for submissions** (`src/forecasting/funnel.py`): Submitted = recent conversion ×
  receipts (actual to date, forecast after). It competes walk-forward with the model forecast and a
  blend, and is adopted only if at least 3% better. On 2026-09 data it won at every level (next-quarter
  miss: organisation 10.5 → 9.1%, countries 20.6 → 18.9%, regions 23.1 → 20.9%).
- **Diagnosis of under-prediction**: the models lag turning points (+9% in the 2024–25 fall, −6 to −17%
  in the 2025–26 surge). A momentum-gap correction was tested and rejected. See
  `docs/experiments/2026-09-28_bias_and_funnel.md`.
- **Hedge replaces the hard fallback**: series where the model average lost to "same month last year"
  now publish a horizon-weighted hedge (last-year weight 0.5 next month, rising to 1 at 12 months).
  Tested on those 11 series: next-quarter miss 14.3 → 11.6%, 6 months 14.4 → 10.9%.
- **Executive Outcomes folder**: every `all` run publishes the pack, the country PDFs, the workbook and
  the input form to `RMG Calculations/Outcomes/<YYYY-MM> Forecast Pack/` (kept) and `Outcomes/Latest/`
  (replaced each month) (`src/reports/publish.py`, `run_pipeline.py publish`).
- **Country input forms** (`src/reports/forms.py`):
  - `make-forms` writes one Excel file per destination (Staffing + Market_Inputs sheets) and runs at the
    end of every `all` run;
  - `collect-forms` validates the returned files, backs up `Settings.xlsx` and writes the
    `Staffing_Productivity` and `Human_Inputs` sheets;
  - a staffing rate can be given directly, or derived from current headcount.
- **Stage data, prepared** (`fetch_data.py`):
  - `--discover-stages` lists the stage IDs in use (read-only);
  - `STAGE_GROUPS` (offer, deposit, visa document, visa outcome; country-specific overrides) feeds a
    first-reach-per-cycle extract `monthly_stage_summary.xlsx`;
  - skipped until the IDs are filled in.
- **12-month ranges for single-region destinations** (Europe, Asia, ...): own long-window spread blended
  with the pooled estimate, for the Region group only (tested: better there, worse for countries).
  Europe's 12-month range −49%/+96% → −42%/+73%. See `docs/experiments/2026-10-25_long_window_blend.md`.
- **Improvement round** (`docs/experiments/2026-09-28_improvement_round.md`):
  - calendar and festival effects per 26th–25th cycle (working days, Diwali, Holi, Eid) for receipts;
  - quarterly lock of method decisions (Method Decisions sheet);
  - extract snapshots with late-entry revision measurement;
  - "vs last month's forecast" revisions in the pack;
  - pre-filled market input form (`Reports/Human_Inputs_Form_<date>.xlsx`);
  - tested and rejected: accuracy-weighted average, middle-out adding-up, revision damping.
- **One decision rule for every component**: the overlay and the reconciliation method are now also
  judged on next-quarter totals per level, adopted only if at least 3% better. The overlay's old 1%
  monthly-WAPE gate flipped between runs and moved the organisation forecast by 12% with identical data.
  On quarter totals the overlay wins decisively (countries 18.8 → 14.6%), and bottom-up reconciliation
  beats wls_var (level-average 16.9 vs 17.4%).
- **Direction signal from the latest vintages** (`src/forecasting/vintage.py`): the last 3 full
  quarters plus the latest 2 partial ones, read at country level and applied at half strength to
  receipts. Walk-forward next-quarter miss: countries 18.4 → 14.2%, regions 22.3 → 19.2%, organisation
  unchanged. Re-tested each run. Pack page 3 "Direction check" shows the vintage misses.
- **Proof page** (pack page 2): the forecast made 3 months ago vs what happened, per destination,
  regenerated every month.
- **Fiscal-year outlook with range**: actual to date + forecast remainder, where only the remainder is
  uncertain, so the range narrows every month. On pack page 1 and every country page.
- **Health check**: four acceptance criteria fixed in advance, scored on tested and live forecasts, in
  the pack and in Excel.
- **Data safeguard**: the latest month is flagged CRITICAL if its year-on-year change collapses or
  jumps against the previous 3 months (partial extract); shown as a banner on page 1.
- Report wording states the actual test depth. The retired v2.1 interval code is marked as such
  (independent review findings).
- The Excel workbook gains a Range Method Test sheet.

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
