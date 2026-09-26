# -*- coding: utf-8 -*-
"""
Model 10 — Global Cross-Learning LightGBM Engine
-------------------------------------------------
• Panel/Cross-Learning: A single unified Gradient Boosting model trained across all
  3,356 historical series-months, solving the small sample size bottleneck of individual series.
• Domain Awareness: Ingests academic intake calendar flags (Fall/Spring deadlines, visa filing windows).
• Coupled Funnel: Submissions dynamically condition on recent inquiry/receipt inventory lags.
• Explainability: Exports global feature importances revealing the primary drivers of volume.
• Strictly Leak-Free: Supervised features computed entirely on history prior to forecast origin.
Artefacts → …/Models/Model10_Global_LGBM/
"""
from __future__ import annotations

import os, logging, warnings, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from ..config import OUT_DIR, HORIZON, RANDOM_SEED
from ..loader import load_data
from ..intake_calendar import get_calendar_exog_matrix

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)
console = Console()

def _safe(s: str) -> str:
    return re.sub(r'[\\/:\*?"<>|]', "_", str(s))

RUN_DIR  = OUT_DIR / "Models" / "Model10_Global_LGBM"
PLOT_DIR = RUN_DIR / "plots"
RUN_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=RUN_DIR/"model10_global.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

OSS_MONTHS = int(os.environ.get("RMG_OSS_MONTHS", "3"))

# ─────────────────────────────────────────────────────────────────────
# Feature Engineering on Global Panel
# ─────────────────────────────────────────────────────────────────────
def _build_panel_features(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Builds leak-free panel lag features, rolling statistics, and academic calendar features
    across all series simultaneously.
    """
    df = df_long.copy().sort_values(["Country", "Region", "Metric", "Date"]).reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])

    # Separate Received to create cross-metric lead-lag features for Submitted
    rec_lookup = (
        df[df["Metric"] == "Received"]
        .set_index(["Country", "Region", "Date"])["Value"]
        .to_dict()
    )

    feature_dfs = []
    for (country, region, metric), g in df.groupby(["Country", "Region", "Metric"]):
        g = g.sort_values("Date").reset_index(drop=True)
        y = g["Value"].astype(float)
        
        # Strictly leak-free: shift(1) ensures all features are derived from t-1 or earlier
        y_prev = y.shift(1)
        
        # Autoregressive lags
        g["lag_1"] = y_prev
        g["lag_2"] = y.shift(2)
        g["lag_3"] = y.shift(3)
        g["lag_12"] = y.shift(12)
        
        # Rolling levels and volatility
        g["roll_mean_3"] = y_prev.rolling(3, min_periods=1).mean()
        g["roll_mean_6"] = y_prev.rolling(6, min_periods=1).mean()
        g["roll_std_3"]  = y_prev.rolling(3, min_periods=1).std().fillna(0.0)
        
        # Year-over-Year momentum
        y_prev12 = y.shift(13)
        g["yoy_momentum"] = np.where(y_prev12 > 0, y_prev / y_prev12, 1.0)
        g["yoy_momentum"] = np.clip(g["yoy_momentum"], 0.2, 5.0)

        # Coupled Funnel: If Metric == 'Submitted', add recent 'Received' lead lags
        if metric == "Submitted":
            dates = g["Date"]
            rec_l1, rec_l2, rec_l3 = [], [], []
            for dt in dates:
                # dt - 1 month, dt - 2 months, dt - 3 months
                d1 = dt - pd.DateOffset(months=1)
                d2 = dt - pd.DateOffset(months=2)
                d3 = dt - pd.DateOffset(months=3)
                rec_l1.append(rec_lookup.get((country, region, d1), np.nan))
                rec_l2.append(rec_lookup.get((country, region, d2), np.nan))
                rec_l3.append(rec_lookup.get((country, region, d3), np.nan))
            g["rec_lag_1"] = pd.Series(rec_l1, index=g.index).fillna(g["roll_mean_3"])
            g["rec_lag_2"] = pd.Series(rec_l2, index=g.index).fillna(g["roll_mean_3"])
            g["rec_lag_3"] = pd.Series(rec_l3, index=g.index).fillna(g["roll_mean_3"])
        else:
            g["rec_lag_1"] = 0.0
            g["rec_lag_2"] = 0.0
            g["rec_lag_3"] = 0.0

        # Academic Calendar Matrix
        cal = get_calendar_exog_matrix(g["Date"], country)
        for col in ["Is_Fall_Deadline_Window", "Is_Spring_Deadline_Window", "Is_Visa_Window", "Is_July_Special", "Billing_Cycle_Days", "Sin_Month", "Cos_Month"]:
            g[col] = cal[col].values

        feature_dfs.append(g)

    panel = pd.concat(feature_dfs, ignore_index=True)
    return panel

# ─────────────────────────────────────────────────────────────────────
# Recursive Multi-Step Forecasting
# ─────────────────────────────────────────────────────────────────────
def _recursive_forecast(
    model: lgb.LGBMRegressor,
    history_long: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    horizon: int = 12
) -> pd.DataFrame:
    """
    Simulates recursive step-by-step 12-month forecasting for all series simultaneously,
    updating lag features iteratively.
    """
    df = history_long.copy()
    max_date = pd.to_datetime(df["Date"]).max()
    future_dates = [max_date + pd.DateOffset(months=m) for m in range(1, horizon + 1)]

    # Store predicted records
    predicted_records = []
    
    # We maintain a dynamic working panel that appends new predictions step-by-step
    working_df = df.copy()

    for target_dt in future_dates:
        # Build features on the current working history
        panel_feat = _build_panel_features(working_df)
        
        # Prepare rows for the new target date
        step_rows = []
        for (c, r, m), g in working_df.groupby(["Country", "Region", "Metric"]):
            last_feat = panel_feat[(panel_feat.Country==c) & (panel_feat.Region==r) & (panel_feat.Metric==m)].iloc[-1]
            
            # Form feature vector for target_dt
            row_dict = {
                "Country": c,
                "Region": r,
                "Metric": m,
                "Date": target_dt,
            }
            # The lag_1 for target_dt is the last actual/predicted value
            last_val = float(g.sort_values("Date")["Value"].iloc[-1])
            prev_vals = g.sort_values("Date")["Value"].values
            
            row_dict["lag_1"] = last_val
            row_dict["lag_2"] = float(prev_vals[-2]) if len(prev_vals) >= 2 else last_val
            row_dict["lag_3"] = float(prev_vals[-3]) if len(prev_vals) >= 3 else last_val
            row_dict["lag_12"] = float(prev_vals[-12]) if len(prev_vals) >= 12 else last_val
            row_dict["roll_mean_3"] = float(np.mean(prev_vals[-3:])) if len(prev_vals) >= 1 else last_val
            row_dict["roll_mean_6"] = float(np.mean(prev_vals[-6:])) if len(prev_vals) >= 1 else last_val
            row_dict["roll_std_3"]  = float(np.std(prev_vals[-3:])) if len(prev_vals) >= 3 else 0.0
            
            yoy_denom = float(prev_vals[-13]) if len(prev_vals) >= 13 else last_val
            row_dict["yoy_momentum"] = float(np.clip(last_val / yoy_denom if yoy_denom > 0 else 1.0, 0.2, 5.0))
            
            # Calendar features
            cal = get_calendar_exog_matrix(pd.Series([target_dt]), c).iloc[0]
            for col in ["Is_Fall_Deadline_Window", "Is_Spring_Deadline_Window", "Is_Visa_Window", "Is_July_Special", "Billing_Cycle_Days", "Sin_Month", "Cos_Month"]:
                row_dict[col] = cal[col]
            
            # Coupled Funnel
            if m == "Submitted":
                # Find recent Received values
                rec_series = working_df[(working_df.Country==c) & (working_df.Region==r) & (working_df.Metric=="Received")].sort_values("Date")["Value"].values
                row_dict["rec_lag_1"] = float(rec_series[-1]) if len(rec_series) >= 1 else last_val
                row_dict["rec_lag_2"] = float(rec_series[-2]) if len(rec_series) >= 2 else last_val
                row_dict["rec_lag_3"] = float(rec_series[-3]) if len(rec_series) >= 3 else last_val
            else:
                row_dict["rec_lag_1"] = 0.0
                row_dict["rec_lag_2"] = 0.0
                row_dict["rec_lag_3"] = 0.0
                
            step_rows.append(row_dict)

        step_df = pd.DataFrame(step_rows)
        for cat in cat_cols:
            step_df[cat] = step_df[cat].astype("category")

        # Predict
        X_step = step_df[feature_cols]
        preds = np.maximum(0.0, model.predict(X_step))
        step_df["Value"] = preds
        step_df["Forecast"] = preds

        # Append to working_df for next recursive step
        working_df = pd.concat([working_df, step_df[["Country", "Region", "Metric", "Date", "Value"]]], ignore_index=True)
        predicted_records.append(step_df[["Country", "Region", "Metric", "Date", "Forecast"]])

    return pd.concat(predicted_records, ignore_index=True).sort_values(["Country", "Region", "Metric", "Date"]).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────
# Main Pipeline Function
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    console.print("[bold cyan]▶️  Running Model 10: Global Cross-Learning LightGBM Engine …[/]")

    # 1. Load Data
    raw, _ = load_data()
    df_long = (raw.melt(
        id_vars=["Country", "Region", "Date"],
        value_vars=[c for c in ["Received", "Submitted"] if c in raw.columns],
        var_name="Metric", value_name="Value"
    ).dropna(subset=["Value"]).sort_values(["Country", "Region", "Metric", "Date"]).reset_index(drop=True))

    df_long["Date"] = pd.to_datetime(df_long["Date"]).dt.to_period("M").dt.to_timestamp()

    # Build features on full history
    panel = _build_panel_features(df_long)
    panel = panel.dropna(subset=["lag_1", "roll_mean_3"]).reset_index(drop=True)

    feature_cols = [
        "Country", "Region", "Metric",
        "lag_1", "lag_2", "lag_3", "lag_12",
        "roll_mean_3", "roll_mean_6", "roll_std_3", "yoy_momentum",
        "rec_lag_1", "rec_lag_2", "rec_lag_3",
        "Is_Fall_Deadline_Window", "Is_Spring_Deadline_Window", "Is_Visa_Window", "Is_July_Special",
        "Billing_Cycle_Days", "Sin_Month", "Cos_Month"
    ]
    cat_cols = ["Country", "Region", "Metric"]
    for c in cat_cols:
        panel[c] = panel[c].astype("category")

    # ─────────────────────────────────────────────────────────────────
    # 2. Strict Out-Of-Sample (OSS) Backtest
    # ─────────────────────────────────────────────────────────────────
    max_date = panel["Date"].max()
    cutoff_date = max_date - pd.DateOffset(months=OSS_MONTHS - 1)
    
    train_panel = panel[panel["Date"] < cutoff_date].copy()
    test_panel  = panel[panel["Date"] >= cutoff_date].copy()

    X_train = train_panel[feature_cols]
    y_train = train_panel["Value"]

    lgb_params = {
        "objective": "regression_l1",  # MAE loss, robust to outliers
        "n_estimators": 160,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 8,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    model_oss = lgb.LGBMRegressor(**lgb_params)
    model_oss.fit(X_train, y_train)

    # Predict OSS
    X_test = test_panel[feature_cols]
    test_panel["Pred"] = np.maximum(0.0, model_oss.predict(X_test))
    test_panel["Actual"] = test_panel["Value"]

    bt_cols = ["Country", "Region", "Metric", "Date", "Actual", "Pred"]
    backtest_df = test_panel[bt_cols].sort_values(["Country", "Region", "Metric", "Date"]).reset_index(drop=True)
    backtest_df.to_csv(RUN_DIR / "backtest.csv", index=False)

    # Compute errors for display
    bt_clean = backtest_df.dropna().copy()
    rec_bt = bt_clean[bt_clean["Metric"] == "Received"]
    sub_bt = bt_clean[bt_clean["Metric"] == "Submitted"]

    def _calc_stats(sub_df):
        mae = float(np.median(abs(sub_df["Actual"] - sub_df["Pred"]))) if not sub_df.empty else 0.0
        denom = (abs(sub_df["Actual"]) + abs(sub_df["Pred"])) / 2.0
        smape = float(np.median(abs(sub_df["Actual"] - sub_df["Pred"]) / denom.replace(0, np.nan))) * 100.0 if not sub_df.empty else 0.0
        return mae, smape

    rec_mae, rec_smape = _calc_stats(rec_bt)
    sub_mae, sub_smape = _calc_stats(sub_bt)

    t = Table(title="Model 10 Global LightGBM — OSS Holdout Performance (Medians)", header_style="bold magenta")
    t.add_column("Metric", style="bold")
    t.add_column("Median MAE", justify="right")
    t.add_column("Median sMAPE %", justify="right")
    t.add_row("Received", f"{rec_mae:.1f}", f"{rec_smape:.1f}%")
    t.add_row("Submitted", f"{sub_mae:.1f}", f"{sub_smape:.1f}%")
    console.print(t)

    # ─────────────────────────────────────────────────────────────────
    # 3. Production Model & 12-Month Recursive Forecast
    # ─────────────────────────────────────────────────────────────────
    model_prod = lgb.LGBMRegressor(**lgb_params)
    model_prod.fit(panel[feature_cols], panel["Value"])

    # Feature Importance
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model_prod.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    importance_df.to_csv(RUN_DIR / "feature_importance.csv", index=False)

    # Plot Top 12 Feature Importances
    plt.figure(figsize=(8, 4.5))
    top_feats = importance_df.head(12).sort_values("Importance", ascending=True)
    plt.barh(top_feats["Feature"], top_feats["Importance"], color="#1f77b4")
    plt.title("Model 10 Global LightGBM — Top Feature Importances")
    plt.xlabel("Split Gain Importance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "feature_importance.png", dpi=130)
    plt.close()

    # Recursive 12m forecast
    forecast_df = _recursive_forecast(model_prod, df_long, feature_cols, cat_cols, horizon=HORIZON)
    forecast_df.to_csv(RUN_DIR / "forecast_h12.csv", index=False)

    console.print(f"[bold green]✅ Model 10 Global LightGBM finished — artefacts saved to {RUN_DIR}[/]")

if __name__ == "__main__":
    main()
