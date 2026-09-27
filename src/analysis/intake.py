# -*- coding: utf-8 -*-
"""
Descriptive intake view (secondary to the monthly volume forecast).

Answers "for a given intake, in which months did its applications arrive?" from the intake
extract (Intake = target intake month name, InYear = its year):
  arrivals      intake x application month: count, share of the intake, cumulative share,
                months before the intake starts
  lead_profile  typical arrival curve per season (Spring Jan-Mar, Summer Apr-Jul, Fall Aug-Dec):
                average cumulative share by months before start, over completed intakes
  month_mix     for each calendar month of application, the share going to each season
Nothing here changes the forecast.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ..data.prepare import ORG, TOTAL

MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                          "nov", "dec"])}
SEASON_OF_MONTH = {1: "Spring", 2: "Spring", 3: "Spring", 4: "Summer", 5: "Summer", 6: "Summer", 7: "Summer",
                   8: "Fall", 9: "Fall", 10: "Fall", 11: "Fall", 12: "Fall"}
SEASON_TO_MONTH = {"spring": 1, "summer": 5, "fall": 9, "autumn": 9, "winter": 1}


def parse_intake_month(x) -> float:
    s = str(x).strip().lower()
    if not s or s in ("nan", "none"):
        return np.nan
    if s[:3] in MONTHS:
        return MONTHS[s[:3]]
    if s in SEASON_TO_MONTH:
        return SEASON_TO_MONTH[s]
    try:
        v = int(float(s))
        return v if 1 <= v <= 12 else np.nan
    except ValueError:
        return np.nan


def load_intake(path: Path, drop_regions: tuple = ()) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_excel(path, dtype={"Month": str, "Intake": str, "InYear": str})
    return clean_intake(df, drop_regions)


def clean_intake(df: pd.DataFrame, drop_regions: tuple = ()) -> pd.DataFrame:
    d = df.rename(columns={"Country Grouped": "Country", "Name of Region": "Region"}).copy()
    d = d[~d["Region"].isin(drop_regions)]
    d["Date"] = pd.to_datetime(d["Month"].astype(str).str[:7] + "-01", errors="coerce")
    d["Intake_Month"] = d["Intake"].map(parse_intake_month)
    d["Intake_Year"] = pd.to_numeric(d["InYear"], errors="coerce")
    d["Received"] = pd.to_numeric(d["ReceivedApps"], errors="coerce").fillna(0)
    d["Submitted"] = pd.to_numeric(d.get("SubmittedApps", 0), errors="coerce").fillna(0)
    ok = d["Date"].notna() & d["Intake_Month"].notna() & d["Intake_Year"].between(2000, 2100)
    d["Intake_Known"] = ok
    d.loc[ok, "Intake_Start"] = pd.to_datetime(dict(year=d.loc[ok, "Intake_Year"].astype(int),
                                                    month=d.loc[ok, "Intake_Month"].astype(int), day=1))
    return d


def _levels(d: pd.DataFrame) -> pd.DataFrame:
    """Country totals and the organisation total (regions are too thin for intake profiles)."""
    keys = ["Date", "Intake_Start", "Intake_Month"]
    c = d.groupby(["Country"] + keys, as_index=False)[["Received", "Submitted"]].sum()
    c["Level"], c["Region"] = "Country", TOTAL
    o = d.groupby(keys, as_index=False)[["Received", "Submitted"]].sum()
    o["Level"], o["Country"], o["Region"] = "Organisation", ORG, TOTAL
    return pd.concat([o, c], ignore_index=True)


def intake_views(d: pd.DataFrame, last_actual: pd.Timestamp, years: int = 3
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Returns (arrivals, lead_profile, month_mix, share of receipts with a usable intake)."""
    if d is None or d.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), np.nan
    known_share = float(d.loc[d["Intake_Known"], "Received"].sum() / max(d["Received"].sum(), 1))
    k = _levels(d[d["Intake_Known"]])
    k["Months_Before_Start"] = ((k["Intake_Start"].dt.year - k["Date"].dt.year) * 12
                                + (k["Intake_Start"].dt.month - k["Date"].dt.month))
    k["Season"] = k["Intake_Month"].astype(int).map(SEASON_OF_MONTH)
    k["Intake_Label"] = k["Intake_Start"].dt.strftime("%b %Y")
    cutoff = last_actual - pd.DateOffset(years=years)
    arr = k[k["Intake_Start"] >= cutoff].copy()
    tot = arr.groupby(["Level", "Country", "Intake_Start"])["Received"].transform("sum")
    arr["Share_of_Intake_Pct"] = 100 * arr["Received"] / tot.replace(0, np.nan)
    arr = arr.sort_values(["Level", "Country", "Intake_Start", "Date"])
    arr["Cumulative_Pct"] = arr.groupby(["Level", "Country", "Intake_Start"])["Share_of_Intake_Pct"].cumsum()
    arr["Intake_Status"] = np.where(arr["Intake_Start"] <= last_actual, "Started", "Upcoming (still filling)")

    # typical arrival curve from completed intakes (started at least 3 months ago)
    done = arr[arr["Intake_Start"] <= last_actual - pd.DateOffset(months=3)]
    lead = (done.groupby(["Level", "Country", "Season", "Intake_Start", "Months_Before_Start"])["Received"].sum()
                .reset_index())
    lead["Share"] = lead["Received"] / lead.groupby(["Level", "Country", "Season", "Intake_Start"])["Received"].transform("sum")
    grid = []
    for key, g in lead.groupby(["Level", "Country", "Season"]):
        per = g.pivot_table(index="Intake_Start", columns="Months_Before_Start", values="Share", aggfunc="sum").fillna(0)
        per = per[sorted(per.columns, reverse=True)].cumsum(axis=1)          # far -> near: cumulative arrived
        for mb in per.columns:
            grid.append({"Level": key[0], "Country": key[1], "Season": key[2], "Months_Before_Start": int(mb),
                         "Avg_Cumulative_Pct": 100 * float(per[mb].mean()), "Intakes_Used": int(len(per))})
    lead_profile = pd.DataFrame(grid)

    recent = k[k["Date"] > cutoff]
    mix = recent.groupby(["Level", "Country", recent["Date"].dt.month.rename("Application_Month"), "Season"])["Received"].sum()
    mix = (mix / mix.groupby(level=[0, 1, 2]).transform("sum") * 100).rename("Share_Pct").reset_index()
    cols = ["Level", "Country", "Intake_Label", "Season", "Intake_Start", "Intake_Status", "Date", "Months_Before_Start",
            "Received", "Submitted", "Share_of_Intake_Pct", "Cumulative_Pct"]
    return arr[cols], lead_profile, mix, known_share
