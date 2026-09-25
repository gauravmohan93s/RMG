# -*- coding: utf-8 -*-
# src/models/_bt_utils.py
from __future__ import annotations
import pandas as pd

def standardize_backtest_tail(back_df: pd.DataFrame, tail_months: int = 18) -> pd.DataFrame:
    """
    Trim a model's backtest to the last `tail_months` calendar months (by Date).
    Expects columns: ["Date","Country","Region","Metric","Actual","Pred"].
    Returns a copy. If fewer than 3 rows after trim, returns the original.
    """
    if back_df is None or back_df.empty or "Date" not in back_df.columns:
        return back_df
    g = back_df.sort_values("Date")
    # unique month stamps
    uniq = pd.Index(pd.to_datetime(g["Date"]).unique()).sort_values()
    if len(uniq) <= tail_months:
        return g
    keep = set(uniq[-tail_months:])
    out = g[g["Date"].isin(keep)].copy()
    return out if len(out) >= 3 else g

def clip_forecast_nonneg(fc: pd.DataFrame) -> pd.DataFrame:
    """Clip Forecast/Lower/Upper to ≥ 0 if those columns exist."""
    if fc is None or fc.empty:
        return fc
    for col in ("Forecast", "Lower", "Upper"):
        if col in fc.columns:
            fc[col] = pd.to_numeric(fc[col], errors="coerce").clip(lower=0.0)
    return fc
