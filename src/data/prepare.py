# -*- coding: utf-8 -*-
"""
Raw extract -> modelling panel.

Rules (see docs/METHODOLOGY.md §1):
* A month missing from the extract is a true zero (the SQL only emits rows with
  activity) — never interpolated.
* A Country×Region series starts at its first month with activity.
* Status per series: Inactive (no activity in the last ``inactive_months``),
  Emerging (fewer than ``min_months_for_tournament`` months), else Mature.
* Modelling hierarchy = active regions -> country totals -> organisation total.
  Countries with a single active region are not duplicated as a separate node.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import DataCfg

METRICS = ("Received", "Submitted")
ORG = "_ORG"
TOTAL = "_TOTAL"

RENAMES = {
    "Month": "Date",
    "Country Grouped": "Country",
    "Name of Region": "Region",
    "ReceivedApps": "Received",
    "SubmittedApps": "Submitted",
}


def node_id(metric: str, country: str, region: str) -> str:
    return f"{metric}|{country}|{region}"


@dataclass
class Panel:
    months: pd.DatetimeIndex                 # global month-start index, first..last actual
    nodes: pd.DataFrame                      # one row per modelled node
    y: Dict[str, np.ndarray]                 # node_id -> values aligned to months[start:]
    start: Dict[str, int]                    # node_id -> index into months
    history: pd.DataFrame                    # actuals for reporting (all series, incl. inactive)
    dq: pd.DataFrame                         # data-quality register
    series_status: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def end(self) -> pd.Timestamp:
        return self.months[-1]

    def series(self, nid: str) -> pd.Series:
        s = self.start[nid]
        return pd.Series(self.y[nid], index=self.months[s:], name=nid)


# ────────────────────────────── raw extract ──────────────────────────────
def load_raw(path: Path, drop_regions: tuple = ()) -> pd.DataFrame:
    """Read the monthly extract; return Country, Region, Date, Received, Submitted."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw extract not found: {path}")
    df = pd.read_excel(path)
    df = df.rename(columns=RENAMES)
    missing = [c for c in ("Date", "Country", "Region", *METRICS) if c not in df.columns]
    if missing:
        raise ValueError(f"{Path(path).name} is missing columns: {missing}")
    return clean_raw(df, drop_regions)


def clean_raw(df: pd.DataFrame, drop_regions: tuple = ()) -> pd.DataFrame:
    out = df[["Date", "Country", "Region", *METRICS]].copy()
    out["Date"] = pd.to_datetime(out["Date"].astype(str), errors="coerce").dt.to_period("M").dt.to_timestamp()
    bad_dates = int(out["Date"].isna().sum())
    if bad_dates:
        raise ValueError(f"{bad_dates} rows have an unparseable Month value")
    out["Country"] = out["Country"].astype(str).str.strip()
    out["Region"] = out["Region"].astype(str).str.strip()
    for m in METRICS:
        out[m] = pd.to_numeric(out[m], errors="coerce").fillna(0.0).clip(lower=0.0)
    out = out[~out["Region"].isin(drop_regions)]
    # The extract is already aggregated; summing makes duplicates harmless.
    return out.groupby(["Country", "Region", "Date"], as_index=False)[list(METRICS)].sum()


# ────────────────────────────── panel ────────────────────────────────────
def build_panel(raw: pd.DataFrame, cfg: DataCfg) -> Panel:
    if raw.empty:
        raise ValueError("Raw extract is empty after filtering")
    months = pd.date_range(raw["Date"].min(), raw["Date"].max(), freq="MS")
    pos = {d: i for i, d in enumerate(months)}
    T = len(months)
    dq: List[dict] = []

    # 1) bottom series on the full grid (zeros where the extract has no row)
    grid: Dict[tuple, np.ndarray] = {}
    status_rows = []
    for (cty, reg), g in raw.groupby(["Country", "Region"], sort=True):
        arr = np.zeros((len(METRICS), T))
        idx = g["Date"].map(pos).to_numpy()
        for k, m in enumerate(METRICS):
            arr[k, idx] = g[m].to_numpy(dtype=float)
        activity = arr.sum(axis=0)
        first = int(np.argmax(activity > 0)) if activity.any() else T
        last = int(np.max(np.nonzero(activity > 0)[0])) if activity.any() else -1
        n_months = T - first
        zero_filled = int(n_months - len(set(idx[idx >= first]))) if n_months > 0 else 0
        recent = activity[max(0, T - cfg.inactive_months):].sum()
        if n_months <= 0 or recent == 0:
            status = "Inactive"
        elif n_months < cfg.min_months_for_tournament:
            status = "Emerging"
        else:
            status = "Mature"
        grid[(cty, reg)] = arr
        status_rows.append({
            "Country": cty, "Region": reg, "Status": status,
            "First_Month": months[first] if first < T else pd.NaT,
            "Last_Active_Month": months[last] if last >= 0 else pd.NaT,
            "Months_Of_History": max(n_months, 0),
            "Zero_Filled_Months": zero_filled,
            "Received_Last_12m": float(arr[0, -12:].sum()),
            "Submitted_Last_12m": float(arr[1, -12:].sum()),
        })
        if status == "Inactive":
            dq.append(_dq("Inactive series (not forecast)", cty, reg,
                          f"No activity in the last {cfg.inactive_months} months; "
                          f"last active {months[last].strftime('%b %Y') if last >= 0 else 'never'}"))
        elif status == "Emerging":
            dq.append(_dq("Emerging series (low confidence)", cty, reg,
                          f"{n_months} months of history (< {cfg.min_months_for_tournament}); "
                          "forecast with pooled seasonality, excluded from the tournament"))
        if zero_filled:
            dq.append(_dq("Months with no activity (set to 0)", cty, reg,
                          f"{zero_filled} month(s) absent from the extract inside the series span"))
        if reg == "Other" or cty == "Other":
            dq.append(_dq("Unmapped IDs grouped as 'Other'", cty, reg,
                          "Update Settings.xlsx Country/Region mapping"))
    series_status = pd.DataFrame(status_rows)

    # 2) modelling nodes
    nodes, y, start = [], {}, {}

    def _add(level, cty, reg, metric, arr, first, status, members):
        nid = node_id(metric, cty, reg)
        y[nid] = arr[first:].astype(float)
        start[nid] = first
        nodes.append({"node_id": nid, "Level": level, "Country": cty, "Region": reg,
                      "Metric": metric, "Status": status, "Start": months[first],
                      "Months_Of_History": T - first, "Members": members})

    active = series_status[series_status["Status"] != "Inactive"]
    for k, metric in enumerate(METRICS):
        org_sum, org_first, org_members = np.zeros(T), T, 0
        for cty, cg in active.groupby("Country", sort=True):
            c_sum, c_first = np.zeros(T), T
            for _, r in cg.iterrows():
                arr = grid[(cty, r["Region"])][k]
                first = months.get_loc(r["First_Month"])
                _add("Region", cty, r["Region"], metric, arr, first, r["Status"], 1)
                c_sum += arr
                c_first = min(c_first, first)
            if len(cg) >= 2:
                c_status = "Mature" if (T - c_first) >= cfg.min_months_for_tournament else "Emerging"
                _add("Country", cty, TOTAL, metric, c_sum, c_first, c_status, len(cg))
            org_sum += c_sum
            org_first = min(org_first, c_first)
            org_members += len(cg)
        _add("Organisation", ORG, TOTAL, metric, org_sum, org_first, "Mature", org_members)

    # 3) reporting history (all series incl. inactive) at every level
    hist_rows = []
    for (cty, reg), arr in grid.items():
        for k, metric in enumerate(METRICS):
            hist_rows.append(pd.DataFrame({"Level": "Region", "Country": cty, "Region": reg,
                                           "Metric": metric, "Date": months, "Actual": arr[k]}))
    history = pd.concat(hist_rows, ignore_index=True)
    ctry = history.groupby(["Country", "Metric", "Date"], as_index=False)["Actual"].sum()
    ctry["Level"], ctry["Region"] = "Country", TOTAL
    org = history.groupby(["Metric", "Date"], as_index=False)["Actual"].sum()
    org["Level"], org["Country"], org["Region"] = "Organisation", ORG, TOTAL
    history = pd.concat([history, ctry, org], ignore_index=True)[
        ["Level", "Country", "Region", "Metric", "Date", "Actual"]]

    return Panel(months=months, nodes=pd.DataFrame(nodes), y=y, start=start,
                 history=history, dq=pd.DataFrame(dq, columns=["Issue", "Country", "Region", "Detail"]),
                 series_status=series_status)


def _dq(issue: str, cty: str, reg: str, detail: str) -> dict:
    return {"Issue": issue, "Country": cty, "Region": reg, "Detail": detail}
