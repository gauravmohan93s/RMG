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
python -c "
import os
from sqlalchemy import create_engine
from urllib.parse import quote_plus

os.environ['DB_SERVER'] = os.getenv('DB_SERVER')
os.environ['DB_USERNAME'] = os.getenv('DB_USERNAME')
os.environ['DB_PASSWORD'] = os.getenv('DB_PASSWORD')
os.environ['DB_NAME'] = os.getenv('DB_NAME')

from fetch_data import *
engine = create_engine(CONNECTION_URL)
print('✅ Database connection successful!')
"
```

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

# Where pipeline outputs (forecasts, reports) will be saved
RMG_OUT_DIR=C:\Users\YourName\RMG\RMG Calculations\Predictions
```

### Directory Structure
```
RMG Calculations/
├── Settings.xlsx                 # Your mappings (must exist)
├── monthly_app_summary.xlsx      # Created by fetch-data
├── monthly_final_summary.xlsx    # Created by transform-data
│
└── Predictions/
    ├── 2025-01-20/              # Run date folder (created automatically)
    │   ├── final_forecast.parquet
    │   ├── series_selection.csv
    │   ├── reports/
    │   │   └── *.pdf
    │   └── model_outputs/
    │       ├── model01_lastyear.parquet
    │       ├── model05_prophet.parquet
    │       └── ...
    │
    └── 2025-01-21/              # Next run
        └── ...
```

## 5️⃣ Configure Forecasting Parameters

Update `.env` for your forecast needs:

```env
# How many months to forecast into the future
FORECAST_HORIZON=12

# Random seed for reproducibility (keep same for consistent results)
RANDOM_SEED=42

# Data fetch parameters
FETCH_START_DATE=2020-01-01      # Historical data start
FETCH_END_DATE=                  # Leave empty for yesterday
```

## 6️⃣ (Optional) Setup LLM Report Commentary

If you want AI-generated commentary in PDF reports, install Ollama:

### Install Ollama
1. Download from https://ollama.ai
2. Install and run

### Pull a Small Model
```bash
ollama pull phi3:mini
# or
ollama pull qwen2:0.5b
```

### Configure in `.env`
```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=phi3:mini
OLLAMA_NUM_CTX=512
```

### Start Ollama
```bash
ollama serve
```

**Note**: If Ollama is not available, reports will be generated without LLM commentary (graceful fallback).

## 7️⃣ Verify Setup

### Run a Quick Test
```bash
# Test data loader
python run_pipeline.py load-data

# Expected output:
# ▶️  Testing data loader …
# loaded PREPARED table – X,XXX rows
# found artefacts: [...]
# ✅  Loader OK
```

### Check Log Output
```bash
# View pipeline log
tail -f pipeline.log
```

## 8️⃣ Run the Full Pipeline

### First Full Run
```bash
# Run for today
python run_pipeline.py all

# Or specify a date
python run_pipeline.py all --run-date 2025-01-20
```

### Expected Output Sequence
1. **Fetch data** from SQL Server (skips if no new months)
2. **Transform** to aggregated summary
3. **EDA** explores the data
4. **Load data** and validate
5. **Run models** (9 forecasting models)
6. **Evaluate** model performance
7. **Forecast** using best models
8. **Report** generates PDF files

### Check Results
```bash
# List output files
ls -la "C:\path\to\RMG Calculations\Predictions\2025-01-20"

# View forecasts
# → monthly_final_summary.xlsx
# → reports/*.pdf
```

## 9️⃣ Handling Unmapped Items

If you see warnings about unmapped CountryIDs or TagGroupIDs:

```
⚠️  Unmapped items detected (auto-grouped as 'Other'):
   Missing CountryIDs: [4, 7, 12]
   Missing TagGroupIDs: [15, 22]
```

### To Fix
1. Open `Settings.xlsx`
2. Add the missing IDs to appropriate sheets
3. Re-run the pipeline (it will use your mappings)

```bash
python run_pipeline.py all --skip-fetch
```

## 🔟 Automate with Task Scheduler (Windows)

### Create Batch File
Save as `run_forecast.bat`:
```batch
@echo off
cd C:\path\to\RMG
call venv\Scripts\activate.bat
python run_pipeline.py all --skip-fetch
pause
```

### Schedule Daily Run
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger to daily at desired time
4. Set action to run `run_forecast.bat`
5. Configure task settings:
   - Run whether user is logged in or not
   - Run with highest privileges

### Or with Cron (Linux/macOS)
```bash
# Edit crontab
crontab -e

# Add line (runs at 2 AM daily)
0 2 * * * cd /home/user/RMG && /home/user/RMG/venv/bin/python run_pipeline.py all --skip-fetch >> pipeline_cron.log 2>&1
```

## 🛑 Troubleshooting

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

### ❌ LSTM or XGBoost Out of Memory
**Solution**:
- Skip heavy models: `python run_pipeline.py all --exclude model07_xgboost,model08_lstm`
- Reduce data: Use fewer historical months by adjusting `FETCH_START_DATE`

### ❌ Ollama Connection Error
```
requests.exceptions.ConnectionError: Ollama not available
```
**Solution** (optional - not required):
- Install Ollama: https://ollama.ai
- Start Ollama: `ollama serve`
- Or disable: Reports will work without LLM commentary

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
