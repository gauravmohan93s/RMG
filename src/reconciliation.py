# -*- coding: utf-8 -*-
# src/reconciliation.py
from __future__ import annotations
import pandas as pd
import numpy as np

SCEN_COLS = ["Forecast", "Forecast_Lo", "Forecast_Hi"]
COUNTRY_TOTAL_KEY = "_TOTAL"

def _mark_country(df: pd.DataFrame) -> pd.Series:
    r = df["Region"].astype(str).str.upper()
    return r.eq(COUNTRY_TOTAL_KEY) | r.eq("ALL") | r.eq("TOTAL")

def add_country_totals(df: pd.DataFrame) -> pd.DataFrame:
    """If no country total exists, create one as sum of regions per Date."""
    if df.empty:
        return df
    existing_cols = [c for c in SCEN_COLS if c in df.columns]
    if not existing_cols:
        return df
    
    # Check if country total already exists
    if _mark_country(df).any():
        return df

    gcols = ["Country", "Metric", "Date"]
    sums = df.groupby(gcols, as_index=False)[existing_cols].sum()
    sums["Region"] = COUNTRY_TOTAL_KEY
    if "Chosen_Model" in df.columns:
        sums["Chosen_Model"] = "Hierarchical_Sum"
    
    # Preserve date helpers if present
    extra_cols = [c for c in ["Date_ISO", "YearMonth", "Year", "Month", "Month_Name"] if c in df.columns]
    for ec in extra_cols:
        val_map = df.drop_duplicates("Date").set_index("Date")[ec]
        sums[ec] = sums["Date"].map(val_map)

    return pd.concat([df, sums], ignore_index=True)

def reconcile_to_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    If a country total row exists for (Country, Metric, Date),
    scale that date's regions proportionally so their sum equals the country total,
    for each forecast column independently.
    """
    if df.empty or not _mark_country(df).any():
        return df

    existing_cols = [c for c in SCEN_COLS if c in df.columns]
    if not existing_cols:
        return df

    out = df.copy()
    out["_is_country"] = _mark_country(out)

    def _reconcile_group(g: pd.DataFrame) -> pd.DataFrame:
        top = g[g["_is_country"]]
        reg = g[~g["_is_country"]].copy()
        if top.empty or reg.empty:
            return g.drop(columns=["_is_country"])
        for c in existing_cols:
            target = float(top[c].sum())
            s = float(reg[c].sum())
            if s > 0 and abs(target - s) > 1e-4:
                reg[c] = reg[c] * (target / s)
        return pd.concat([top, reg], ignore_index=True).drop(columns=["_is_country"])

    return (
        out.groupby(["Country", "Metric", "Date"], group_keys=False)
           .apply(_reconcile_group)
           .reset_index(drop=True)
    )

def apply_reconciliation(df: pd.DataFrame) -> pd.DataFrame:
    """Add country totals if absent, then reconcile regions to country totals."""
    return reconcile_to_country(add_country_totals(df))
