# RMG Forecasting Pipeline

A comprehensive **end-to-end forecasting pipeline** that predicts future values for "Received" and "Submitted" application metrics based on historical data. The pipeline automates data collection, transformation, exploratory analysis, multi-model forecasting, evaluation, and PDF report generation.

## 🎯 Features

- **Automated Data Pipeline**: Fetches data from SQL Server, transforms, and aggregates by month/country/region
- **Multi-Model Forecasting**: Includes 9 forecasting models:
  - **Baselines**: Last-year, Month-over-Month, Year-over-Year, Rolling Average
  - **Statistical**: Facebook Prophet, Seasonal ARIMA (SARIMA)
  - **Machine Learning**: XGBoost with feature engineering, LSTM neural networks
  - **Ensemble**: Weighted average based on model performance
- **Automatic Model Evaluation**: Uses holdout set to calculate MAE and sMAPE
- **PDF Report Generation**: Auto-generated reports with LLM commentary (Ollama integration)
- **Flexible Configuration**: Centralized settings with environment variable support
- **Non-Interactive Automation**: Runs end-to-end without manual intervention

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- SQL Server access (with stored credentials)
- Optional: Ollama with a small language model for report commentary

### Installation

```bash
# Clone the repository
git clone https://github.com/gauravmohan93s/RMG.git
cd RMG

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **Copy the environment template**:
   ```bash
   cp .env.example .env
   ```

2. **Update `.env` with your credentials**:
   ```
   # Database
   DB_SERVER=your-sql-server.rds.amazonaws.com
   DB_NAME=your_database
   DB_USERNAME=your_username
   DB_PASSWORD=your_password
   
   # Data Paths
   RMG_DATA_ROOT=C:\path\to\data
   RMG_OUT_DIR=C:\path\to\output
   
   # LLM (Optional, for report commentary)
   OLLAMA_HOST=http://localhost:11434
   OLLAMA_MODEL=phi3:mini
   ```

3. **Update `Settings.xlsx`** with your country and region mappings (in the RMG data folder)

## 📋 Usage

### Run the Full Pipeline

```bash
# Run entire pipeline for today's date
python run_pipeline.py all

# Run for a specific date
python run_pipeline.py all --run-date 2025-01-20

# Skip specific steps
python run_pipeline.py all --skip-fetch --skip-transform

# Run only specific models
python run_pipeline.py all --only model01_lastyear,model05_prophet
```

### Run Individual Steps

```bash
# Fetch data from SQL Server
python run_pipeline.py fetch-data

# Transform raw data to aggregated summary
python run_pipeline.py transform-data

# Run exploratory data analysis
python run_pipeline.py eda

# Test data loader
python run_pipeline.py load-data

# Run forecasting models
python run_pipeline.py run-models

# Evaluate model performance
python run_pipeline.py evaluate

# Generate final forecast
python run_pipeline.py forecast

# Generate PDF reports
python run_pipeline.py report
```

## 📁 Project Structure

```
RMG/
├── run_pipeline.py              # Main CLI orchestrator
├── fetch_data.py                # SQL Server data fetching
├── transform_data.py            # Data transformation & aggregation
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
├── README.md                    # This file
│
├── src/
│   ├── config.py               # Centralized configuration
│   ├── eda.py                  # Exploratory data analysis
│   ├── loader.py               # Data loading module
│   ├── reports/                # Report generation
│   │   ├── final_pdf_reports.py
│   │   └── ...
│   │
│   └── models/                 # Forecasting models
│       ├── model01_lastyear.py
│       ├── model02_mom_yoy.py
│       ├── model03_rollmom.py
│       ├── model04_avg_mom.py
│       ├── model05_prophet.py
│       ├── model06_sarima.py
│       ├── model07_xgboost.py
│       ├── model08_lstm.py
│       ├── model09_ensemble.py
│       ├── evaluator.py        # Model evaluation
│       └── final_forecast.py   # Final forecast assembly
│
└── test.py                      # Test scripts
```

## 🔄 Pipeline Workflow

```
Data Fetch (SQL Server)
    ↓
Data Transform (monthly aggregation)
    ↓
Exploratory Data Analysis (EDA)
    ↓
Data Loading & Validation
    ↓
Run Forecasting Models (9 models in parallel)
    ↓
Model Evaluation (MAE, sMAPE on holdout set)
    ↓
Ensemble & Final Forecast
    ↓
PDF Report Generation (with LLM commentary)
```

## 🔑 Key Configuration Files

### `.env` (Credentials & Paths)
```env
# Database credentials
DB_SERVER=your-server.rds.amazonaws.com
DB_NAME=database_name
DB_USERNAME=user
DB_PASSWORD=pass

# Data locations
RMG_DATA_ROOT=C:\RMG\RMG Calculations
RMG_OUT_DIR=C:\RMG\RMG Calculations\Predictions\{run_date}

# Optional: LLM settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=phi3:mini
```

### `Settings.xlsx` (Data Mappings)
Workbook with sheets:
- **Country**: Maps CountryID → Country Grouped
- **Region**: Maps TagGroupID → Region per Country Grouped

Unmapped items are automatically grouped as "Other" and logged for review.

## 📊 Model Evaluation

The `evaluator.py` script:
- Uses **last 3 months** as holdout test set
- Calculates **MAE** (Mean Absolute Error) and **sMAPE** (symmetric MAPE)
- Saves results to `series_selection.csv`
- Selects **best model per series** based on MAE for "Received" metric

Results guide ensemble weighting and final forecast.

## 🤖 LLM Commentary (Ollama)

Reports can include AI-generated commentary via Ollama:

```bash
# Start Ollama locally
ollama serve

# In another terminal, pull a small model
ollama pull phi3:mini

# Reports will now include LLM-generated insights
python run_pipeline.py report
```

Models tested: `phi3:mini`, `qwen2:0.5b`

## 📝 Logging

All pipeline events are logged to `pipeline.log`:
```bash
tail -f pipeline.log  # Monitor in real-time
```

## 🔐 Security

- **Never commit** `.env` files with real credentials
- Use `DB_PASSWORD` and other secrets via environment variables
- The repo includes `.env.example` as a safe template
- Credentials are read at runtime, not stored in code

## 🛠️ Development

### Adding a New Model

1. Create `src/models/model10_yourmodel.py`
2. Implement `main()` function
3. Save predictions to `RMG_OUT_DIR/model10_yourmodel.parquet`
4. Run `python run_pipeline.py run-models --only model10_yourmodel`

### Running Tests

```bash
python test.py
```

## 📦 Dependencies

Core libraries:
- **pandas**, **numpy**: Data manipulation
- **prophet**: Time series forecasting
- **statsmodels**: SARIMA
- **xgboost**: Gradient boosting
- **tensorflow/keras**: LSTM
- **matplotlib**, **seaborn**: Visualization
- **reportlab**: PDF generation
- **typer**: CLI
- **sqlalchemy**, **pyodbc**: Database access

See `requirements.txt` for full list.

## 🐛 Troubleshooting

### Database Connection Error
```
✗ Check DB_SERVER, DB_NAME, DB_USERNAME, DB_PASSWORD in .env
✗ Ensure ODBC Driver 17 for SQL Server is installed
✗ Test connection: python -c "from fetch_data import *; engine.connect()"
```

### Ollama Not Found (Report Generation)
```
✓ Optional feature - reports will skip LLM commentary if Ollama unavailable
✓ To use: Install Ollama from https://ollama.ai and run `ollama serve`
```

### Out of Memory (LSTM Training)
```
✓ Reduce batch size in model08_lstm.py
✓ Use --only to run subset of models
```

## 📚 References

- [Facebook Prophet Documentation](https://facebook.github.io/prophet/)
- [XGBoost API](https://xgboost.readthedocs.io/)
- [LSTM with Keras](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Statsmodels SARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)

## 📧 Support

For issues or questions:
1. Check `pipeline.log` for error details
2. Review configuration in `.env`
3. Verify `Settings.xlsx` mappings
4. Open an issue on GitHub

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data sourced from RMG application database
- Forecasting methodology combines classical and ML approaches for robustness
- Report generation enhanced with local LLM (Ollama) integration

---

**Last Updated**: January 2026  
**Maintainer**: Gaurav Mohan
