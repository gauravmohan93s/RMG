# -*- coding: utf-8 -*-
"""
src/policy_matrix.py
--------------------
Visa Policy & Regulatory Headwinds Scenario Manager:
• Bridges purely algorithmic statistical forecasts with real-world geopolitical & immigration policy shifts.
• Reads user-configured multipliers from Settings.xlsx ('Policy_Scenarios' sheet) if available.
• Provides domain-grounded defaults for all major destinations:
    - Canada: 0.85 (IRCC study permit caps & PGWP tightening)
    - Australia: 0.90 (Visa fee hikes & GS requirement scrutiny)
    - United Kingdom: 0.95 (Dependent visa restrictions, offset by Graduate Route retention)
    - United States of America: 1.05 (Strong STEM OPT demand and expanding visa appointment slots)
    - Europe / Central & Eastern Europe: 1.08 (Affordable tuition alternatives)
    - Default / Others: 1.00 (Neutral baseline)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd

SETTINGS_FILE = Path(r"C:\Users\gsakhare\OneDrive - KC OVERSEAS EDUCATION PVT LTD\RMG\RMG Calculations\Settings.xlsx")

# Domain-grounded defaults for geopolitical & visa policies
DEFAULT_POLICY_MATRIX = pd.DataFrame([
    {
        "Country": "Canada",
        "Policy_Multiplier": 0.85,
        "Risk_Level": "High Headwind 🔴",
        "Regulatory_Context": "IRCC study permit quotas & strict spousal/PGWP eligibility limits"
    },
    {
        "Country": "Australia",
        "Policy_Multiplier": 0.90,
        "Risk_Level": "Moderate Headwind 🟡",
        "Regulatory_Context": "Substantial visa fee increases & Genuine Student (GS) assessment scrutiny"
    },
    {
        "Country": "United Kingdom",
        "Policy_Multiplier": 0.95,
        "Risk_Level": "Mild Headwind 🟡",
        "Regulatory_Context": "Postgraduate taught dependent restrictions, partially offset by Graduate Route"
    },
    {
        "Country": "United States of America",
        "Policy_Multiplier": 1.05,
        "Risk_Level": "Growth Tailwind 🟢",
        "Regulatory_Context": "Strong 3-year STEM OPT appeal, expanding visa interview waiver slots"
    },
    {
        "Country": "Centra & Eastern Europe",
        "Policy_Multiplier": 1.08,
        "Risk_Level": "Growth Tailwind 🟢",
        "Regulatory_Context": "Cost-effective Schengen access & medical/technical program surge"
    },
    {
        "Country": "Europe",
        "Policy_Multiplier": 1.06,
        "Risk_Level": "Growth Tailwind 🟢",
        "Regulatory_Context": "Expanding English-taught programs in Germany, France, and Ireland"
    },
    {
        "Country": "New Zealand",
        "Policy_Multiplier": 1.00,
        "Risk_Level": "Neutral ⚪",
        "Regulatory_Context": "Stable post-study work rights and steady university intake caps"
    },
    {
        "Country": "East Asia",
        "Policy_Multiplier": 1.02,
        "Risk_Level": "Neutral ⚪",
        "Regulatory_Context": "Emerging regional hub demand (Japan, South Korea, Malaysia)"
    },
    {
        "Country": "Southeast & Central Asia",
        "Policy_Multiplier": 1.00,
        "Risk_Level": "Neutral ⚪",
        "Regulatory_Context": "Steady intra-Asian mobility"
    },
    {
        "Country": "SNS",
        "Policy_Multiplier": 1.00,
        "Risk_Level": "Neutral ⚪",
        "Regulatory_Context": "Regional direct-admissions and non-traditional channels"
    },
    {
        "Country": "Asia",
        "Policy_Multiplier": 1.00,
        "Risk_Level": "Neutral ⚪",
        "Regulatory_Context": "Standard continental baseline"
    },
])

def get_policy_matrix() -> pd.DataFrame:
    """
    Returns the active policy matrix.
    If 'Policy_Scenarios' sheet exists in Settings.xlsx, it is loaded;
    otherwise, the robust domain defaults are used.
    """
    if SETTINGS_FILE.exists():
        try:
            xl = pd.ExcelFile(SETTINGS_FILE)
            if "Policy_Scenarios" in xl.sheet_names:
                df = pd.read_excel(SETTINGS_FILE, sheet_name="Policy_Scenarios")
                req = {"Country", "Policy_Multiplier"}
                if req.issubset(df.columns):
                    logging.info("Loaded custom policy scenarios from Settings.xlsx")
                    return df
        except Exception as e:
            logging.warning("Could not read Policy_Scenarios from Settings.xlsx: %s", e)

    return DEFAULT_POLICY_MATRIX.copy()

def get_country_multiplier_map() -> Dict[str, float]:
    """Returns mapping {Country -> Policy_Multiplier}."""
    df = get_policy_matrix()
    return dict(zip(df["Country"], df["Policy_Multiplier"]))

def apply_policy_scenarios(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Augments the forecast dataframe with policy multiplier and scenario values:
    - Policy_Multiplier
    - Regulatory_Risk_Level
    - Forecast_Policy_Adjusted (Point forecast * Policy Multiplier)
    - Forecast_Conservative (Low bound * min(1.0, Multiplier))
    - Forecast_Aggressive (High bound * max(1.0, Multiplier))
    """
    if forecast_df.empty:
        return forecast_df

    out = forecast_df.copy()
    mat = get_policy_matrix()
    mult_map = dict(zip(mat["Country"], mat["Policy_Multiplier"]))
    risk_map = dict(zip(mat["Country"], mat["Risk_Level"]))

    out["Policy_Multiplier"] = out["Country"].map(mult_map).fillna(1.00).astype(float)
    out["Regulatory_Risk"] = out["Country"].map(risk_map).fillna("Neutral ⚪")

    # Modulate forecasts
    out["Forecast_Policy_Adjusted"] = np.round(out["Forecast"] * out["Policy_Multiplier"], 1)

    if "Forecast_Lo" in out.columns:
        lo_mult = np.minimum(1.0, out["Policy_Multiplier"])
        out["Forecast_Conservative"] = np.round(out["Forecast_Lo"] * lo_mult, 1)
    else:
        out["Forecast_Conservative"] = np.round(out["Forecast"] * 0.90, 1)

    if "Forecast_Hi" in out.columns:
        hi_mult = np.maximum(1.0, out["Policy_Multiplier"])
        out["Forecast_Aggressive"] = np.round(out["Forecast_Hi"] * hi_mult, 1)
    else:
        out["Forecast_Aggressive"] = np.round(out["Forecast"] * 1.10, 1)

    return out
