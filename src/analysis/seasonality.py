# -*- coding: utf-8 -*-
"""
Seasonality and staffing.

Seasonal index (per organisation / country / region and metric)
  Robust STL decomposition (period 12) of log1p volume over the most recent `years`
  years; the seasonal component averaged by calendar month and converted to a
  multiplicative index with mean 1.00 ("1.25" = a typical month is 25% above an
  average month). Robust STL down-weights one-off spikes.
Month class
  High   index >= high_threshold (default 1.10)
  Low    index <= low_threshold  (default 0.90)
  Medium otherwise
Stability
  Each month's volume / centred 2x12 moving average (classical ratio-to-moving-average,
  trend removed); the share of years in which the month falls in the same class as its
  index = Consistency_Pct.
Staffing calendar (next 12 months)
  Load index = forecast / average forecast month; class from the same thresholds;
  capacity plan = upper end of the 80% range; optional FTE = volume / apps per FTE-month.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def classify(x: float, high: float, low: float) -> str:
    if not np.isfinite(x):
        return ""
    return "High" if x >= high else ("Low" if x <= low else "Medium")


def seasonal_indices(history: pd.DataFrame, high: float = 1.10, low: float = 0.90, years: int = 4,
                     min_months: int = 36, levels: Sequence[str] = ("Organisation", "Country", "Region")):
    """Returns (index table: one row per series x calendar month, year table: series x year x month)."""
    from statsmodels.tsa.seasonal import STL

    idx_rows, year_rows = [], []
    for key, g in history.groupby(["Level", "Country", "Region", "Metric"], sort=False):
        if key[0] not in levels:
            continue
        s = g.sort_values("Date").set_index("Date")["Actual"].astype(float)
        nz = s[s > 0]
        if nz.empty or s.iloc[-6:].sum() == 0:
            continue
        s = s[s.index >= nz.index[0]]
        s = s.iloc[-years * 12:]
        if len(s) < min_months:
            continue
        res = STL(np.log1p(s.to_numpy()), period=12, robust=True).fit()
        seas = pd.Series(res.seasonal, index=s.index)
        by_m = seas.groupby(seas.index.month).mean().reindex(range(1, 13))
        ix = np.exp(by_m - by_m.mean())
        ix = ix / ix.mean()

        # classical ratio-to-moving-average (centred 2x12 MA) removes trend before comparing years
        df = s.to_frame("v")
        df["Year"], df["Month"] = df.index.year, df.index.month
        ma = s.rolling(12).mean().rolling(2).mean().shift(-6)
        df["Ratio"] = df["v"] / ma.replace(0, np.nan)
        df = df.dropna(subset=["Ratio"])
        for r in df.itertuples():
            year_rows.append({"Level": key[0], "Country": key[1], "Region": key[2], "Metric": key[3],
                              "Year": int(r.Year), "Month": int(r.Month), "Month_Name": MONTH_NAMES[r.Month - 1],
                              "Volume": float(r.v), "Ratio_to_Trend": float(r.Ratio)})
        ranks = ix.rank(ascending=False, method="first")
        for m in range(1, 13):
            cls = classify(float(ix[m]), high, low)
            yr = df[df["Month"] == m]["Ratio"].dropna()
            consistency = float(100 * np.mean([classify(v, high, low) == cls for v in yr])) if len(yr) else np.nan
            idx_rows.append({"Level": key[0], "Country": key[1], "Region": key[2], "Metric": key[3],
                             "Month": m, "Month_Name": MONTH_NAMES[m - 1], "Seasonal_Index": float(ix[m]),
                             "Class": cls, "Rank": int(ranks[m]), "Consistency_Pct": consistency,
                             "Years_Used": int(df["Year"].nunique()), "Months_Used": int(len(s))})
    return pd.DataFrame(idx_rows), pd.DataFrame(year_rows)


def staffing_calendar(fc: pd.DataFrame, sidx: pd.DataFrame, high: float = 1.10, low: float = 0.90,
                      apps_per_fte_month: Optional[float] = None,
                      levels: Sequence[str] = ("Organisation", "Country")) -> pd.DataFrame:
    """Next-12-month workload calendar per organisation/country and metric."""
    hist_cls = sidx.set_index(["Level", "Country", "Region", "Metric", "Month"])["Class"] if len(sidx) else None
    rows = []
    for key, g in fc[fc["Level"].isin(levels)].groupby(["Level", "Country", "Region", "Metric"], sort=False):
        g = g.sort_values("Date")
        avg = float(g["Forecast"].mean())
        for r in g.itertuples():
            load = r.Forecast / avg if avg > 0 else np.nan
            row = {"Level": key[0], "Country": key[1], "Region": key[2], "Metric": key[3],
                   "Label": getattr(r, "Label", key[1]), "Date": r.Date, "Month_Name": MONTH_NAMES[r.Date.month - 1],
                   "Forecast": r.Forecast, "Low_80": r.Lo80, "Capacity_Plan_80": r.Hi80,
                   "Load_Index": load, "Workload_Class": classify(load, high, low),
                   "Historical_Class": (hist_cls.get((*key, r.Date.month), "") if hist_cls is not None else "")}
            if apps_per_fte_month and apps_per_fte_month > 0:
                row["FTE_Base"] = r.Forecast / apps_per_fte_month
                row["FTE_Capacity_80"] = r.Hi80 / apps_per_fte_month
            rows.append(row)
    return pd.DataFrame(rows)
