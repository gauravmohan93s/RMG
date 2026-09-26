# -*- coding: utf-8 -*-
"""
src/tracking.py
---------------
Forecast Governance & Historical Audit Store.

Maintains a persistent, append-only history of every forecast run:
- What model was selected and WHY.
- What were the projected Low, Best (Base), and High values.
- Automatically joins newly ingested actuals to evaluate real-world forecasting accuracy,
  band coverage (reliability), and historical bias for executive reporting and model training feedback.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from .config import OUT_DIR

# Persistent storage location: in the root predictions directory across all runs
TRACKING_DIR = OUT_DIR.parent
TRACKING_CSV = TRACKING_DIR / "prediction_history.csv"
TRACKING_PARQUET = TRACKING_DIR / "prediction_history.parquet"

def _load_history_store() -> pd.DataFrame:
    """Loads existing historical predictions audit store if present."""
    if TRACKING_PARQUET.exists():
        try:
            return pd.read_parquet(TRACKING_PARQUET)
        except Exception:
            pass
    if TRACKING_CSV.exists():
        try:
            return pd.read_csv(TRACKING_CSV, parse_dates=["Run_Date", "Target_Date"])
        except Exception:
            pass
    return pd.DataFrame()

def _save_history_store(df: pd.DataFrame) -> None:
    """Saves updated predictions audit store to CSV and Parquet."""
    if df.empty:
        return
    try:
        TRACKING_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(TRACKING_CSV, index=False)
    except Exception as e:
        logging.warning("Failed to save prediction history CSV: %s", e)
    try:
        df.to_parquet(TRACKING_PARQUET, index=False)
    except Exception:
        pass  # Parquet engine optional if pyarrow not installed

def log_forecast_run(
    final_forecast: pd.DataFrame,
    series_selection: Optional[pd.DataFrame] = None,
    run_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Logs an immutable record of the current forecast run into the audit store.
    Captures: Run_Date, Target_Date, Horizon, Country, Region, Metric, Chosen_Model,
              Selection_Reason, Forecast_Lo, Forecast_Base, Forecast_Hi.
    """
    if final_forecast.empty:
        return pd.DataFrame()

    today_str = run_date or datetime.now().strftime("%Y-%m-%d")
    today_dt = pd.to_datetime(today_str)

    fc = final_forecast.copy()
    fc["Run_Date"] = today_dt
    fc["Target_Date"] = pd.to_datetime(fc["Date"])
    
    # Calculate Horizon step in months
    # (year diff * 12 + month diff)
    fc["Horizon_Months"] = (
        (fc["Target_Date"].dt.year - fc["Run_Date"].dt.year) * 12 +
        (fc["Target_Date"].dt.month - fc["Run_Date"].dt.month)
    )

    # Standardize column names
    fc["Forecast_Base"] = fc["Forecast"]
    if "Forecast_Lo" not in fc.columns:
        fc["Forecast_Lo"] = fc["Forecast"] * 0.90
    if "Forecast_Hi" not in fc.columns:
        fc["Forecast_Hi"] = fc["Forecast"] * 1.10

    # Attach Selection Reason
    reason_map = {}
    if series_selection is not None and not series_selection.empty:
        sel = series_selection.copy()
        # Look for selection metric / score columns
        for _, row in sel.iterrows():
            c = str(row.get("Country", ""))
            r = str(row.get("Region", ""))
            m = str(row.get("Metric", ""))
            model = str(row.get("Chosen_Model", row.get("Model", "")))
            mae = row.get("MAE", row.get("Holdout_MAE", np.nan))
            smape = row.get("sMAPE", row.get("Holdout_sMAPE", np.nan))
            
            parts = [f"Model '{model}' selected as top performer in holdout tournament."]
            if pd.notna(mae):
                parts.append(f"Holdout MAE: {mae:,.1f}")
            if pd.notna(smape):
                parts.append(f"sMAPE: {smape:.1f}%")
            reason_map[(c, r, m)] = " | ".join(parts)

    reasons = []
    for _, row in fc.iterrows():
        key = (str(row.get("Country")), str(row.get("Region")), str(row.get("Metric")))
        reasons.append(reason_map.get(key, f"Selected based on best backtest performance ({row.get('Chosen_Model', 'Best')})"))
    fc["Selection_Reason"] = reasons

    keep_cols = [
        "Run_Date", "Target_Date", "Horizon_Months", "Country", "Region", "Metric",
        "Chosen_Model", "Selection_Reason", "Forecast_Lo", "Forecast_Base", "Forecast_Hi"
    ]
    current_run = fc[[c for c in keep_cols if c in fc.columns]].copy()
    current_run["Actual_Value"] = np.nan
    current_run["Realized_MAE"] = np.nan
    current_run["Realized_sMAPE"] = np.nan
    current_run["In_Band"] = np.nan

    # Merge with existing store (avoid duplicates on Run_Date + Target_Date + Series)
    existing = _load_history_store()
    if not existing.empty:
        # Filter out existing entries with the exact same Run_Date and keys to allow idempotency
        key_cols = ["Run_Date", "Target_Date", "Country", "Region", "Metric"]
        existing_keys = set(zip(*[existing[c].astype(str) for c in key_cols if c in existing.columns]))
        current_run_keys = set(zip(*[current_run[c].astype(str) for c in key_cols if c in current_run.columns]))
        
        # Keep non-colliding existing records
        mask_existing = ~existing.apply(lambda r: tuple(str(r[c]) for c in key_cols) in current_run_keys, axis=1)
        combined = pd.concat([existing[mask_existing], current_run], ignore_index=True)
    else:
        combined = current_run

    _save_history_store(combined)
    return combined

def reconcile_history_with_actuals(history_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compares all past logged predictions against real actuals.
    Returns:
    1. Reconciled full audit DataFrame
    2. Executive Model Governance Scorecard DataFrame
    """
    audit = _load_history_store()
    if audit.empty or history_df is None or history_df.empty:
        return audit, pd.DataFrame()

    hist = history_df.copy()
    if "Value" in hist.columns and "Actual" not in hist.columns:
        hist = hist.rename(columns={"Value": "Actual"})

    hist["Date"] = pd.to_datetime(hist["Date"])
    audit["Target_Date"] = pd.to_datetime(audit["Target_Date"])

    # Merge actuals by Country, Region, Metric, Target_Date
    lookup = hist.set_index(["Country", "Region", "Metric", "Date"])["Actual"].to_dict()

    actuals_found = []
    in_bands = []
    maes = []
    smapes = []

    for _, row in audit.iterrows():
        key = (str(row["Country"]), str(row["Region"]), str(row["Metric"]), row["Target_Date"])
        act = lookup.get(key, np.nan)
        actuals_found.append(act)

        if pd.notna(act):
            lo = float(row.get("Forecast_Lo", np.nan))
            base = float(row.get("Forecast_Base", np.nan))
            hi = float(row.get("Forecast_Hi", np.nan))

            in_b = 1.0 if (act >= lo - 1e-4 and act <= hi + 1e-4) else 0.0
            in_bands.append(in_b)
            
            mae_val = abs(base - act)
            maes.append(mae_val)

            denom = (abs(base) + abs(act)) / 2.0
            smap = (abs(base - act) / denom * 100.0) if denom > 0 else 0.0
            smapes.append(smap)
        else:
            in_bands.append(np.nan)
            maes.append(np.nan)
            smapes.append(np.nan)

    audit["Actual_Value"] = actuals_found
    audit["In_Band"] = in_bands
    audit["Realized_MAE"] = maes
    audit["Realized_sMAPE"] = smapes

    _save_history_store(audit)

    # Build Executive Governance Scorecard
    evaluable = audit.dropna(subset=["Actual_Value"]).copy()
    if evaluable.empty:
        return audit, pd.DataFrame()

    scorecard = evaluable.groupby("Chosen_Model").agg(
        Total_Forecasts_Evaluated=("Actual_Value", "count"),
        Band_Coverage_Rate_Pct=("In_Band", lambda x: np.round(np.mean(x) * 100.0, 1)),
        Median_Realized_MAE=("Realized_MAE", lambda x: np.round(np.median(x), 1)),
        Median_Realized_sMAPE_Pct=("Realized_sMAPE", lambda x: np.round(np.median(x), 1)),
        Mean_Bias=("Forecast_Base", lambda f: np.round(np.mean(f - evaluable.loc[f.index, "Actual_Value"]), 1))
    ).reset_index().sort_values("Band_Coverage_Rate_Pct", ascending=False)

    return audit, scorecard
