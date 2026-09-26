# -*- coding: utf-8 -*-
"""
src/intake_calendar.py
----------------------
Domain-specific intake calendar logic and consultant lead-time lag mechanics.

University Course Start Intakes:
- Fall Intake: August to December (includes July for Australia, New Zealand, USA).
- Spring Intake: January to March.
- Summer Intake: April to June (includes July for UK, Canada, Europe, Asia, etc.).

Consultant Lead Times:
- Applications submitted to universities 2 to 5 months prior to intake start.
- Inquiries / receipts gathered 4 to 12 months prior.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Countries where July course start aligns with Fall / Mid-Year intake
JULY_FALL_COUNTRIES = {
    "australia",
    "new zealand",
    "united states of america",
    "united states",
    "usa",
}

def get_intake_label(country: str, dt: pd.Timestamp) -> Tuple[str, int, str]:
    """
    Returns (IntakeName, IntakeYear, IntakeCode) for a given country and date.
    IntakeCode format: 'Fall 2026', 'Spring 2027', 'Summer 2027'
    """
    month = dt.month
    year = dt.year
    c_clean = str(country).strip().lower()

    if month in (8, 9, 10, 11, 12):
        name = "Fall"
        intake_year = year
    elif month in (1, 2, 3):
        name = "Spring"
        intake_year = year
    elif month in (4, 5, 6):
        name = "Summer"
        intake_year = year
    elif month == 7:
        if c_clean in JULY_FALL_COUNTRIES:
            name = "Fall"
            intake_year = year
        else:
            name = "Summer"
            intake_year = year
    else:
        name = "Unknown"
        intake_year = year

    return name, intake_year, f"{name} {intake_year}"

def assign_intake_columns(df: pd.DataFrame, date_col: str = "Date", country_col: str = "Country") -> pd.DataFrame:
    """
    Enriches a dataframe with Intake, Intake_Year, and Intake_Cycle columns.
    """
    if df.empty or date_col not in df.columns or country_col not in df.columns:
        return df

    out = df.copy()
    dates = pd.to_datetime(out[date_col])
    countries = out[country_col].astype(str)

    intake_names = []
    intake_years = []
    intake_cycles = []

    for c, d in zip(countries, dates):
        name, y, code = get_intake_label(c, d)
        intake_names.append(name)
        intake_years.append(y)
        intake_cycles.append(code)

    out["Intake"] = intake_names
    out["Intake_Year"] = intake_years
    out["Intake_Cycle"] = intake_cycles
    return out

def build_intake_summary(forecast_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates forecasts and previous intake actuals into an executive Intake Summary table
    with projected intake totals, uncertainty ranges, and YoY intake growth rates.
    """
    if forecast_df.empty:
        return pd.DataFrame()

    fc = assign_intake_columns(forecast_df, "Date", "Country")
    
    # Required grouping columns
    grp_cols = ["Country", "Region", "Metric", "Intake", "Intake_Year", "Intake_Cycle"]
    agg_dict = {"Forecast": "sum"}
    if "Forecast_Lo" in fc.columns:
        agg_dict["Forecast_Lo"] = "sum"
    if "Forecast_Hi" in fc.columns:
        agg_dict["Forecast_Hi"] = "sum"

    fc_summary = fc.groupby(grp_cols, as_index=False).agg(agg_dict)
    
    # Calculate months count in forecast for each intake
    month_counts = fc.groupby(grp_cols, as_index=False)["Date"].nunique().rename(columns={"Date": "Months_Forecasted"})
    fc_summary = fc_summary.merge(month_counts, on=grp_cols, how="left")

    # If history is provided, compute prior year same intake actuals for YoY comparison
    if history_df is not None and not history_df.empty:
        hist = history_df.copy()
        if "Value" in hist.columns and "Actual" not in hist.columns:
            hist = hist.rename(columns={"Value": "Actual"})
        
        hist = assign_intake_columns(hist, "Date", "Country")
        
        hist_agg = hist.groupby(["Country", "Region", "Metric", "Intake", "Intake_Year"], as_index=False)["Actual"].sum()
        hist_agg["Intake_Year_Next"] = hist_agg["Intake_Year"] + 1

        fc_summary = fc_summary.merge(
            hist_agg[["Country", "Region", "Metric", "Intake", "Intake_Year_Next", "Actual"]].rename(
                columns={"Intake_Year_Next": "Intake_Year", "Actual": "Prior_Year_Actual"}
            ),
            on=["Country", "Region", "Metric", "Intake", "Intake_Year"],
            how="left"
        )

        fc_summary["YoY_Growth_%"] = np.where(
            fc_summary["Prior_Year_Actual"] > 0,
            ((fc_summary["Forecast"] - fc_summary["Prior_Year_Actual"]) / fc_summary["Prior_Year_Actual"]) * 100.0,
            np.nan
        ).round(1)
    else:
        fc_summary["Prior_Year_Actual"] = np.nan
        fc_summary["YoY_Growth_%"] = np.nan

    # Operational & capacity alerts
    alerts = []
    for _, r in fc_summary.iterrows():
        g = r.get("YoY_Growth_%", np.nan)
        if pd.notna(g) and g >= 25.0:
            alerts.append(f"🚀 High Growth (+{g:.0f}%) — Staff Capacity Alert")
        elif pd.notna(g) and g <= -15.0:
            alerts.append(f"⚠️ Contraction ({g:.0f}%) — Marketing/Policy Review")
        else:
            alerts.append("⚖️ Normal Seasonal Range")
    fc_summary["Operational_Alert"] = alerts

    # Sort logically
    fc_summary = fc_summary.sort_values(["Country", "Region", "Metric", "Intake_Year", "Intake"]).reset_index(drop=True)
    return fc_summary

def get_calendar_exog_matrix(dates: pd.Series, country: str) -> pd.DataFrame:
    """
    Builds an academic intake calendar exogenous feature matrix for time-series / ML models.
    """
    dt_series = pd.to_datetime(dates)
    c_clean = str(country).strip().lower()
    
    records = []
    for d in dt_series:
        m = d.month
        # Academic Intake Windows
        is_fall_deadline = 1.0 if m in (4, 5, 6) else 0.0      # Consultant peak submissions for Fall
        is_spring_deadline = 1.0 if m in (10, 11) else 0.0     # Consultant peak submissions for Spring
        is_visa_window = 1.0 if m in (6, 7, 8) else 0.0         # Visa processing window
        
        # July handling
        is_july_special = 1.0 if (m == 7 and c_clean in JULY_FALL_COUNTRIES) else 0.0
        
        # Billing days (26th to 25th)
        # Approximate length of monthly billing cycle
        days_in_month = pd.Period(d, freq='M').days_in_month
        
        records.append({
            "Date": d,
            "Is_Fall_Deadline_Window": is_fall_deadline,
            "Is_Spring_Deadline_Window": is_spring_deadline,
            "Is_Visa_Window": is_visa_window,
            "Is_July_Special": is_july_special,
            "Billing_Cycle_Days": float(days_in_month),
            "Sin_Month": float(np.sin(2 * np.pi * m / 12)),
            "Cos_Month": float(np.cos(2 * np.pi * m / 12)),
        })
        
    return pd.DataFrame(records)
