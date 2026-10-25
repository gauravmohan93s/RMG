# RMG Pipeline Setup Guide

This guide walks you through setting up and configuring the RMG Forecasting Pipeline.

## 1️⃣ Initial Setup

### Clone the Repository
```bash
git clone https://github.com/gauravmohan93s/RMG.git
cd RMG
```

### Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 2️⃣ Configure Database Credentials

### Create `.env` File
```bash
# Copy the example file
cp .env.example .env
```

### Edit `.env` with Your Credentials
Open `.env` in your editor and update:

```env
# Database connection
DB_SERVER=your-sql-server.rds.amazonaws.com
DB_NAME=your_database_name
DB_USERNAME=your_username
DB_PASSWORD=your_password
```

**⚠️ Security Note**: Never commit the `.env` file to Git. It's in `.gitignore` by default.

### Test Database Connection
```bash
python -c "import fetch_data, sqlalchemy; e = fetch_data.make_engine(); e.connect().close(); print('Database connection OK')"
```
`make_engine()` reads `.env` automatically and stops with a clear message if a setting is missing.
Credentials are never stored in code.

## 3️⃣ Prepare Data Mappings

### Create the Settings File

You need to create a `Settings.xlsx` file in your data directory with two worksheets:

#### Worksheet 1: "Country"
Maps CountryID to country groupings.

| CountryId | Country Grouped |
|-----------|-----------------|
| 1         | UK              |
| 2         | USA             |
| 3         | Canada          |
| ... | ... |

**Note**: Any CountryID not in this sheet will be grouped as "Other"

#### Worksheet 2: "Region"
Maps TagGroupID to region names per country.

| id | UK | USA | Canada |
|----|----|----|--------|
| 1  | London | New York | Toronto |
| 2  | Manchester | California | Vancouver |
| ... | ... | ... | ... |

**Structure**:
- Column 1 (`id`): TagGroupID
- Other columns: Country names matching the "Country" sheet
- Cell values: Region names

**Note**: Any TagGroupID-Country combination not mapped will be grouped as "Other"

### Locate Settings File

Update the path in `.env`:
```env
# Example (adjust to your actual path)
RMG_DATA_ROOT=C:\Users\YourName\RMG\RMG Calculations
```

Ensure `{RMG_DATA_ROOT}/Settings.xlsx` exists.

## 4️⃣ Configure Data Paths

Update `.env` with your folder structure:

```env
# Where monthly raw data will be fetched
RMG_DATA_ROOT=C:\Users\YourName\RMG\RMG Calculations

# Optional: write one run somewhere else (default: {RMG_DATA_ROOT}\Predictions\<run-date>)
# RMG_OUT_DIR=C:\temp\rmg_test_run
```

### Directory Structure
```
RMG Calculations/
├── Settings.xlsx                    # Your mappings (+ optional Policy_Overlay sheet)
├── monthly_app_summary.xlsx         # Created by fetch-data
├── monthly_final_summary.xlsx       # Created by transform-data (pipeline input)
│
└── Predictions/
    ├── prediction_history_v2.parquet   # every forecast issued (track record)
    └── 2026-09-27/                     # one folder per run date
        ├── Reports/                    # executive PDF brief + Excel workbook
        ├── Final_Forecast/             # final_forecast.csv, totals, fiscal periods
        ├── Evaluation/                 # backtests, scores, selection evidence
        ├── Data/                       # series status, data quality
        ├── run_manifest.json
        └── pipeline.log
```

## 5️⃣ Configure Forecasting Parameters

All modelling settings live in **`config/pipeline.toml`** (version-controlled; every value is
copied into each run's `run_manifest.json`). The defaults are the validated production settings:

| Setting | Default | Meaning |
|---|---|---|
| `backtest.horizon` | 12 | months forecast **and** evaluated |
| `backtest.n_origins` | 24 | rolling forecast origins used to test every model (must be ≥ horizon) |
| `models.enabled` | 7 models | candidates in the tournament |
| `selection.policy` | `mean_all` | equal-weight average of all models (won the walk-forward comparison) |
| `intervals.levels` | 80, 95 | prediction-interval coverage targets |
| `reconciliation.method` | `wls_var` | how regions/countries/organisation are made to add up |
| `guardrails.min_ratio / max_ratio` | 0.5 / 1.8 | plausibility range for a 12-month total |

Changing any of these changes the numbers: record it in `CHANGELOG.md` and bump the version.
Environment overrides: `RMG_DATA_ROOT` (data folder), `RMG_OUT_DIR` (run folder), `RMG_CONFIG` (alternative TOML).

## 6️⃣ Business / Policy Adjustments (optional)

Add a sheet **`Policy_Overlay`** to `Settings.xlsx` with the columns of
`config/policy_overlay_template.csv`:
`Country, Region (* = all), Metric (* = both), Multiplier, Valid_From (YYYY-MM), Valid_To, Owner, Rationale, Source, Status, Approved_On`.

Only rows with `Status = Approved`, an `Owner`, an `Approved_On` date and a multiplier between
0.5 and 1.5 are applied — and only inside their validity window. The statistical baseline is
always kept; adjusted figures appear in separate `Adjusted` columns.

### Other optional sheets (v2.1)
| Sheet in `Settings.xlsx` | Columns | What happens |
|---|---|---|
| `Human_Inputs` | Country, Region (blank = country total; Country "Organisation" = whole org), Metric, Month (YYYY-MM), Expected_Volume, Submitted_By, Reason, Submitted_On | Shown next to the model, recorded every run and scored against actuals; blended in only for people with ≥ 6 scored months who beat the model |
| `Policy_Events` | Country, Metric (* = both), Event, Start_Month (YYYY-MM), Notes | The real impact of each event is measured against a counterfactual forecast, with a placebo significance test |
| `Staffing_Productivity` | Country, Region (* = all regions of the country), Metric (* = both), Apps_Per_FTE_Month, Owner, Notes | FTE per region with its own rate; country and organisation FTE are the sums of their regions |

Templates: `config/human_inputs_template.csv`, `config/policy_events_template.csv`.
For staffing in FTE, fill `Staffing_Productivity` (productivity differs by country group and region;
the config value `seasonality.apps_per_fte_month` is only a fallback for rows not listed).

## 7️⃣ Verify Setup

```bash
python run_pipeline.py version        # prints the pipeline version
python -m pytest                      # 59 tests: leakage, intervals, models, seasonality, overlays, signals, end-to-end
```

## 8️⃣ Run the Full Pipeline

Run once a month **after the 25th** (the reporting cycle closes on the 25th):

```bash
python run_pipeline.py all                     # fetch -> transform -> forecast -> report -> publish
python run_pipeline.py all --skip-fetch        # reuse the current extract
python run_pipeline.py forecast --run-date 2026-09-27
python run_pipeline.py report   --run-date 2026-09-27   # rebuild reports from a saved run
python run_pipeline.py publish  --run-date 2026-09-27   # copy a run's reports to Outcomes/
```

A run takes about 25 minutes. The 17-model rolling-origin backtest over 36 origins is the long step.

**Before sharing the pack, check two things:**
1. **Page 1 has no red "DATA WARNING" banner.** A banner means the latest month looks partly loaded;
   re-fetch before using the forecast.
2. **Track-record page health checks.** Tested should show Pass. Live shows "Not yet" until 3 months
   of published forecasts have been compared with actuals. Trust a use once Live passes for 3
   consecutive months.

### Country input forms (staffing rates and market inputs)
Every `all` run also writes `RMG Calculations/Forms/<YYYY-MM>/To_Send/<Country>.xlsx`, one per destination:
- **Staffing:** applications per counsellor per month per region, or just the current headcount.
- **Market_Inputs:** the next 3 months; expected volume where the market lead disagrees with the model,
  with a reason.

Send each file to its country lead and save the returned files in `Forms/<YYYY-MM>/Returned/`, then run:

```bash
python run_pipeline.py collect-forms
```

`Settings.xlsx` is backed up first, then its `Staffing_Productivity` and `Human_Inputs` sheets are updated.
The next run uses them.

### Stage data (offers, deposits, visa documents)
The stage groups in `fetch_data.py` (`STAGE_GROUPS`) are defined from `Region - Team.xlsx`, sheet AppStatus2.
If stage IDs are added or changed there, update `STAGE_GROUPS`; the read-only check

```bash
python fetch_data.py --discover-stages
```

lists the IDs in use. Every `all` run then:
1. fetches `monthly_stage_summary.xlsx`;
2. maps it to `stage_final_summary.xlsx`;
3. forecasts the stages (pack page "Pipeline stages", Stage Forecast sheet).

The first fetch pulls history from 2020 and takes longer.

### Executive outputs (`RMG Calculations/Outcomes/`)
Every `all` run publishes the reports for executives (no evidence files):

| Folder | Contents |
|---|---|
| `Outcomes/<YYYY-MM> Forecast Pack/` | that month's pack, country PDFs, workbook and input form, plus a README; kept permanently |
| `Outcomes/Latest/` | always the newest pack (replaced each month): the link to share with leadership |

The folder is named after the first forecast month (actuals to September 2026 = `2026-10 Forecast Pack`).
Override the location with `RMG_OUTCOMES_DIR`.

### Outputs (`Predictions/<run-date>/`)
| Path | Contents |
|---|---|
| `Reports/RMG_Monthly_Pack_<date>.pdf` | leadership pack: organisation, destinations, workload calendar, track record + health check, one page per destination |
| `Reports/Countries/<Country>.pdf` | the same destination page as a separate file, for country heads |
| `Reports/RMG_Forecast_<date>.xlsx` | planning sheets + all evidence sheets |
| `Final_Forecast/final_forecast.csv` | every region, country and organisation × month, with 50/80/95% ranges |
| `Final_Forecast/totals_summary.csv`, `period_summary.csv` | next 3/6/12-month totals, fiscal quarters/years |
| `Evaluation/*` | backtest predictions, model scores, selection evidence, policy comparison, calibration |
| `Data/*` | series status, data-quality register, history used |
| `run_manifest.json` | version, git commit, data SHA-256, config, timings |
| `../prediction_history_v2.parquet` | every forecast ever issued (feeds the Track Record sheet) |

## 9️⃣ Handling Unmapped Items

If you see warnings about unmapped CountryIDs or TagGroupIDs, add them to `Settings.xlsx`
(Country / Region sheets) and re-run with `--skip-fetch`. Until then they are grouped as
"Other" and listed in the **Data Quality** sheet.

## 🔟 Automate with Task Scheduler (Windows)

Save as `run_forecast.bat`:
```batch
@echo off
cd /d "C:\path\to\rmg_forecast_repo"
call .venv_3.12\Scripts\activate.bat
python run_pipeline.py all
```
Schedule it monthly (e.g. the 27th, 07:00) in Task Scheduler.

## 🛑 Troubleshooting

### ❌ `backtest.n_origins must be >= backtest.horizon`
Every forecast horizon must be tested. Keep `n_origins` ≥ `horizon` in `config/pipeline.toml`.

### ❌ `Forecast invariants violated`
The run stops rather than publish inconsistent numbers (e.g. a point outside its range).
Check `pipeline.log` in the run folder and open an issue with the run manifest attached.

### ❌ Database Connection Error
```
Error: Can't connect to database
```
**Solution**:
- Check `.env` credentials
- Verify SQL Server is accessible
- Ensure ODBC Driver 17 is installed: `pip install pyodbc`
- Test: `python -c "import pyodbc; pyodbc.connect(...)"`

### ❌ Settings.xlsx Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: '.../Settings.xlsx'
```
**Solution**:
- Create `Settings.xlsx` in `RMG_DATA_ROOT`
- Ensure "Country" and "Region" sheets exist
- Follow the structure in Section 3

### ❌ ODBC Driver Not Found
```
pyodbc.Error: ('01000', "[01000] [unixODBC]...")
```
**Solution**:
- Windows: Install "ODBC Driver 17 for SQL Server"
- Linux: `sudo apt install unixodbc-dev odbcinst1debian2`
- macOS: `brew install unixodbc`

## ✅ You're Ready!

Your pipeline is configured. Run your first forecast:

```bash
python run_pipeline.py all
```

---

**Questions?** Check the [README.md](README.md) or review `pipeline.log` for detailed error messages.
