# Changelog

All notable changes are recorded here. Format: [Keep a Changelog](https://keepachangelog.com/);
versioning: [SemVer](https://semver.org/) as defined in `docs/ROADMAP.md`.

## [2.6.1] — 2026-10-26
Version stamp and release notes for the 2.6 changes (the `v2.6.0` tag carries the code but still reported
2.5.2 in outputs).

## [2.6.0] — 2026-10-26

### Changed
- **Method decisions no longer flip on near-ties** (`governance/decisions.py`).
  - **Review:** in the quarter's review, a component switches away from its current choice only if the
    alternative is at least 3% better (the same bar as adopting any component).
  - **Once per quarter:** the review happens on the first run of the review month. Later runs that quarter
    need 6%.
  - **Why:** two runs on the same data on 26 Oct flipped three components on differences of about 2%,
    which is run-to-run noise. The stored decisions were restored to their September state before the
    October review; the backup is kept.
- **One sign convention in the pack:** + = actual came in above the forecast, everywhere (blue = above,
  orange = below).
- **Pack pages:**
  - **Capacity:** adds the offers workload calendar.
  - **Can we trust it:** plain-language method choices and their evidence, plus accuracy by months
    ahead for receipts and submissions.
  - **Accuracy page:** larger charts.

### Fixed
- `transform-data` did not write `stage_intake_summary.xlsx` (an edit had not been applied), so offers and
  deposits by intake season were missing. It is now written and covered by a test; the full stage history since
  2020 was re-extracted by intake on 26 Oct (355,706 rows).
- No projected final for an intake when its typical share-in exceeds 100%; chart error bars are never negative.

 — 2026-10-26

### Added
- **Pipeline chapter: offers and deposits by intake season** (ongoing season, per destination: so far vs last year
  at the same point) and the offers booking curve. Both appear automatically after the next fetch, which rebuilds
  the stage history since 2020 by intake (the extract's key now includes Intake / InYear). Until then the page
  shows the submissions curve.

## [2.5.1] — 2026-10-26

### Changed
- **The intake running indicator now works by intake season,** defined on the intake month (not the month an
  application was received or changed status): Spring = Jan–Mar, Summer = Apr–Jul, Fall = Aug–Dec intakes.
  - **Same seasons everywhere:** every destination and region shows the same seasons, so they compare
    directly. Before, each destination showed its own largest intake month.
  - **What the pack shows:**
    - the ongoing season and the next two (Spring 2027, Summer 2027, Fall 2027), side by side;
    - three booking curves and a destination × season grid (so far, vs last year, likely final);
    - the closing season (Fall 2026) as a strip;
    - on each destination's page B, the same grid for its regions.
  - **Projected final:** shown only where it tested better than last year's final and its 80% range spans
    at most 2.5×; otherwise "too early".
  - **Evidence (destinations, November):** at 3 months before a season starts, 14% miss vs 53% for "same as
    last year"; at 6 months, 34% vs 53%.

 — 2026-10-26

### Added
- **Monthly pack rebuilt as a story** (`src/reports/pack.py`).
  - **Organisation, six chapters:** where we are heading; last month and accuracy; where growth comes
    from; pipeline after submission and the coming intakes; capacity; can we trust it.
  - **Each destination, the same flow on two pages:** page A is the headline; page B covers regions /
    teams (table and small charts), the pipeline funnel, the intake curve and the track record.
  - Each chapter opens with a band carrying the one sentence that matters, and key messages are generated
    from the numbers. Pages are full and nothing overflows (30 pages for November).
  - Destination PDFs are the same two pages; the separate legend page is gone (it remains in the footer).
- **Intake running indicator** (`src/analysis/intake_pace.py`).
  - **What it shows:** for every running intake (organisation, destination, region; received, submitted
    and every stage), the count so far vs the same intake a year earlier at the same point, plus a
    projected final where it tested better than "same as last year".
  - **Evidence on November data:** at 3 months before the start, destination finals miss by 7% vs 52% for
    last year's final; at 6 months, 31% vs 50%. It is not published 11–12 months out, where it does not
    beat last year.
  - **Outputs:** the "Intake Running" and "Intake Test" workbook sheets.
- **Stage extract by intake.**
  - `fetch_data.py` adds Intake / InYear to the stage query. The next fetch rebuilds the stage history
    once, which also back-fills visa refused, enrolled and deferred.
  - `transform-data` writes `stage_intake_summary.xlsx`.
- **Live sheets cover every stage.**
  - **Market View:** a pipeline block of destination totals for offers, deposits, visa document, visa
    applied and visa granted.
  - **Staffing:** a region × stage grid (received, submitted, offers, deposits, visa applied), each with
    volume per month, FTE and applications per FTE. Destination rates per stage feed the FTE needed for
    stage forecasts (`seasonality.stage_staffing`); the capacity chapter shows it once teams enter FTE.
- **E-mails, built every run but OFF** (`src/reports/notify.py`, `[notify] enabled = false`).
  - **What is built:** a leadership summary (key messages, destination table, pack attached) and
    reminders to teams whose live sheet has not been reviewed for 45 days.
  - **Previews:** saved to `Reports/Email_preview/*.eml`.
  - **Sending needs:** `enabled = true`, recipients, SMTP settings in `.env`, and the Settings sheet
    `Live_Sheet_Contacts`. Command: `notify`.

### Fixed
- **Track record for single-region destinations** (Asia, Central & Eastern Europe, East Asia, Europe,
  Southeast & Central Asia) was blank. It now uses the region's tested history.
- **The "almost blank" second page** of every destination PDF (the legend) is removed.

 — 2026-10-26

### Added
- **Live country input sheets** (`src/reports/live_forms.py`), shared once and kept up to date by each team:
  `RMG Calculations/Forms/Live/<Country>.xlsx`.
  - **Market View:** a region × month grid with three rows per region: Actual (last 3 months), Model
    (next 12 months) and Your view (yellow input cells). Each row has a reason, name and last-reviewed date.
  - **Visual checks:** Your view turns orange or red when it is more than 15% or 30% away from the model;
    the date turns orange after 90 days without review. Model and actual cells are locked.
  - **Staffing:** one row per region with last-12-month volumes; the team enters current FTE and
    applications per FTE are calculated in the sheet.
  - **Read automatically** at the start of every `forecast` into Settings.xlsx (`Human_Inputs`,
    `Staffing_Productivity`, replaced per destination, Settings backed up daily). No collect step.
  - **Rolled forward** at every `publish`: new model numbers, new 12-month window, and every entry kept.
    Past months stay visible next to the actuals. Each refresh copies the old file to `Live/_history`.
    A file that is open or unreadable is skipped, never overwritten.
  - Commands: `sync-forms`, `refresh-forms`. The monthly `To_Send` / `Returned` forms are no longer made
    by `publish` (`make-forms` / `collect-forms` remain).
- **Outcome stages** from AppStatus2: Visa refused (flag Visa_Rejected), Enrolled (flag Commisionable:
  enrolment confirmed) and Deferred (flag Deffered). They are forecast like the other stages, each from the
  earlier stage that tests best per destination. Data arrives with the next fetch.
- **Country pages** show the next-quarter stage figures.
- **Health check** rows: "Recent under/over-forecast corrected" (statistical overlay status) and "Market
  view from every destination" (how many destinations have a live view).

### Changed
- Change vs last year is not shown for a stage whose same months last year had fewer than 150
  applications (it showed +300–500% on tiny bases).
- A destination without history for one stage no longer stops the later stages.
- Workbook stage columns are in pipeline order.

 — 2026-10-26

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
