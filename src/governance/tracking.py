# -*- coding: utf-8 -*-
"""
Prediction history (append-only, one file across runs) and realised track record.

Every run stores what it predicted (point + 80/95% range + method) per node and
month. Each later run joins the actuals that have since arrived, giving the
real-world accuracy and interval coverage by horizon — the ultimate evidence.
A rerun on the same Run_Date replaces that date's rows (idempotent).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

KEEP = ["Run_Id", "Run_Date", "Pipeline_Version", "Last_Actual_Month", "Level", "Country", "Region",
        "Metric", "Date", "Horizon", "Forecast", "Lo80", "Hi80", "Lo95", "Hi95", "Method"]
KEYS = ["Level", "Country", "Region", "Metric", "Date"]


def _read(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=KEEP)


def previous_forecast(path: Path, last_actual: pd.Timestamp) -> pd.DataFrame:
    """What the most recent EARLIER month's run (actuals to an earlier month) forecast for each node x
    target month: KEYS + Previous_Forecast, Previous_Run_Date. Empty if there is no earlier month."""
    hist = _read(Path(path))
    if hist.empty:
        return pd.DataFrame(columns=KEYS + ["Previous_Forecast", "Previous_Run_Date"])
    la = pd.to_datetime(hist["Last_Actual_Month"])
    earlier = hist[la < pd.Timestamp(last_actual)]
    if earlier.empty:
        return pd.DataFrame(columns=KEYS + ["Previous_Forecast", "Previous_Run_Date"])
    last_run = pd.to_datetime(earlier["Run_Date"]).max()
    p = earlier[pd.to_datetime(earlier["Run_Date"]) == last_run][KEYS + ["Forecast"]].copy()
    p["Date"] = pd.to_datetime(p["Date"])
    return p.rename(columns={"Forecast": "Previous_Forecast"}).assign(Previous_Run_Date=last_run)


def log_run(fc: pd.DataFrame, run_id: str, run_date: str, version: str, last_actual: pd.Timestamp,
            path: Path) -> pd.DataFrame:
    cur = fc.copy()
    cur["Run_Id"], cur["Run_Date"] = run_id, pd.Timestamp(run_date)
    cur["Pipeline_Version"], cur["Last_Actual_Month"] = version, last_actual
    cur = cur[KEEP]
    hist = _read(path)
    if not hist.empty:
        hist = hist[pd.to_datetime(hist["Run_Date"]) != pd.Timestamp(run_date)]
    out = pd.concat([hist, cur], ignore_index=True) if not hist.empty else cur
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def track_record(actuals: pd.DataFrame, path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """actuals: Level, Country, Region, Metric, Date, Actual. Returns (joined rows, scorecard)."""
    hist = _read(path)
    if hist.empty:
        return pd.DataFrame(), pd.DataFrame()
    j = hist.merge(actuals[KEYS + ["Actual"]], on=KEYS, how="inner")
    if j.empty:
        return j, pd.DataFrame()
    j["AE"] = (j["Forecast"] - j["Actual"]).abs()
    j["E"] = j["Forecast"] - j["Actual"]
    j["In80"] = (j["Actual"] >= j["Lo80"]) & (j["Actual"] <= j["Hi80"])
    j["In95"] = (j["Actual"] >= j["Lo95"]) & (j["Actual"] <= j["Hi95"])
    j["Horizon_Bucket"] = pd.cut(j["Horizon"], [0, 3, 6, 12], labels=["1-3", "4-6", "7-12"])
    g = j.groupby(["Pipeline_Version", "Level", "Metric", "Horizon_Bucket"], observed=True)
    den = g["Actual"].sum().replace(0, np.nan)
    sc = pd.DataFrame({
        "Forecasts_Evaluated": g.size(),
        "WAPE_Pct": 100 * g["AE"].sum() / den,
        "Bias_Pct": 100 * g["E"].sum() / den,
        "Coverage80_Pct": 100 * g["In80"].mean(),
        "Coverage95_Pct": 100 * g["In95"].mean(),
    }).reset_index()
    sc["Coverage80_Status"] = np.where(sc["Coverage80_Pct"].between(70, 90), "Well calibrated",
                                       np.where(sc["Coverage80_Pct"] < 70, "Too narrow", "Too wide"))
    return j.drop(columns=["AE", "E"]), sc
