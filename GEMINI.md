# Project Overview

This project is a forecasting pipeline that predicts future values for "Received" and "Submitted" metrics based on historical data. It's structured as a series of Python scripts orchestrated by `run_pipeline.py`. The pipeline includes steps for Exploratory Data Analysis (EDA), data loading, running multiple forecasting models (including Prophet, SARIMA, and XGBoost), evaluating model performance, and generating final forecasts. The project uses `typer` for its command-line interface and `reportlab` to generate PDF reports. It also has an interesting feature to use a local LLM (Ollama) to generate commentary for the reports.

# Building and Running

The project is run via the command line using `run_pipeline.py`. Here are the main commands:

*   `python run_pipeline.py all`: Runs the entire pipeline.
*   `python run_pipeline.py eda`: Runs the EDA script.
*   `python run_pipeline.py load-data`: Tests the data loader.
*   `python run_pipeline.py run-models`: Runs the forecasting models.
*   `python run_pipeline.py evaluate`: Evaluates the models.
*   `python run_pipeline.py forecast`: Generates the final forecast.
*   `python run_pipeline.py report`: Generates PDF reports.

# Models

The project includes a variety of forecasting models, from simple baselines to more complex machine learning models:

*   **Baselines:** `model01_lastyear.py`, `model02_mom_yoy.py`, `model03_rollmom.py`, `model04_avg_mom.py`
*   **Statistical Models:** `model05_prophet.py` (using Facebook's Prophet library), `model06_sarima.py` (Seasonal ARIMA)
*   **Machine Learning Models:** `model07_xgboost.py` (XGBoost with feature engineering), `model08_lstm.py` (LSTM)
*   **Ensemble:** `model09_ensemble.py` combines the predictions of the other models using a weighted average based on their performance.

# Evaluation

The `evaluator.py` script is responsible for evaluating the performance of all models. It uses a holdout set to calculate Mean Absolute Error (MAE) and symmetric Mean Absolute Percentage Error (sMAPE) for each model on each time series. The best model for each series is then selected based on the MAE on the "Received" metric. The results of the evaluation are saved in `series_selection.csv`, which is then used by the ensemble model and the final forecast.

# Development Conventions

*   The code is well-structured with clear separation of concerns (config, loading, models, reports).
*   It uses type hints (`typing` module).
*   It uses a logging mechanism to log pipeline events to `pipeline.log`.
*   The models are organized in the `src/models` directory, and each model is in its own file.
*   The project uses a `__main__` block in each script, which allows them to be run independently.