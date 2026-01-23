# src/reconciliation.py
from __future__ import annotations
import pandas as pd

SCEN_COLS = ["Forecast_min", "Forecast_best", "Forecast_max"]
COUNTRY_TOTAL_KEY = "_TOTAL"

def _mark_country(df: pd.DataFrame) -> pd.Series:
    r = df["Region"].astype(str).str.upper()
    return r.eq(COUNTRY_TOTAL_KEY) | r.eq("ALL")

def add_country_totals(df: pd.DataFrame) -> pd.DataFrame:
    """If no country total exists, create one as sum of regions per Date."""
    if _mark_country(df).any():
        return df
    gcols = ["Country", "BU", "Metric", "Date"]
    sums = (df.groupby(gcols, as_index=False)[SCEN_COLS].sum())
    sums["Region"] = COUNTRY_TOTAL_KEY
    cols = ["Country","Region","BU","Metric","Date"] + SCEN_COLS
    return pd.concat([df, sums[cols]], ignore_index=True)

def reconcile_to_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    If a country total row exists for (Country, BU, Metric, Date),
    scale that date's regions so their sum equals the country value,
    for each scenario column independently.
    """
    if not _mark_country(df).any():
        return df

    out = df.copy()
    out["_is_country"] = _mark_country(out)

    def _one(g):
        top = g[g["_is_country"]]
        reg = g[~g["_is_country"]]
        if top.empty or reg.empty:
            return g.drop(columns=["_is_country"])
        for c in SCEN_COLS:
            target = float(top[c].sum())
            s = float(reg[c].sum())
            if s > 0:
                reg.loc[:, c] = reg[c] * (target / s)
        return pd.concat([top, reg], ignore_index=True).drop(columns=["_is_country"])

    return (out
            .groupby(["Country","BU","Metric","Date"], group_keys=False)
            .apply(_one)
            .reset_index(drop=True))

def apply_reconciliation(df: pd.DataFrame) -> pd.DataFrame:
    """Add totals if absent, then reconcile."""
    x = add_country_totals(df)
    x = reconcile_to_country(x)
    return x
