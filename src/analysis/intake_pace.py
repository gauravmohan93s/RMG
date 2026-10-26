# -*- coding: utf-8 -*-
"""
Intake running indicator ("booking curve"): how far each upcoming intake has filled, compared with the same
intake a year earlier at the same point, and what that implies for its final size.

For one series (organisation, destination or region x metric: Received, Submitted, Offer, Deposit, visa...)
and one intake I (target intake month, e.g. Sep 2027), with L = months before the intake starts:
  to date        applications for intake I counted up to the last closed month
  same point     the same intake a year earlier, counted up to L months before its start
  pace           to date / same point - 1     (ahead / in line / behind at +-5%)
  projection     to date / s, s = median share of the final count that the last 3 completed intakes of the
                 same month already had L months before their start ("final" = counted up to FINAL_AFTER
                 months after the start)

The projection is tested walk-forward on completed intakes: re-made at every lead with only earlier intakes'
shares and scored against their final count, next to the baseline "same as last year's final". It is
published for a level x metric x lead only where it beat the baseline by MIN_GAIN_PCT (the common decision
rule) on at least MIN_TESTS intakes; its range is the 10/90 quantiles of those test errors. Otherwise only the
pace is shown.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..data.prepare import ORG, TOTAL
from .intake import parse_intake_month

FINAL_AFTER = 3            # an intake's count is final 3 months after it starts (late and deferred entries)
MAX_LEAD = 12              # running intakes: start at most 12 months ahead ...
MIN_LEAD = -(FINAL_AFTER - 1)   # ... or started up to 2 months ago (still filling)
SHARE_YEARS = 3            # completed same-month intakes used for the typical share
MIN_SHARE = 0.10           # too early to project when under 10% of an intake is usually in
MIN_GAIN_PCT = 3.0
MIN_TESTS = 8
PACE_BAND = 5.0
MIN_INTAKE = {"Organisation": 100, "Country": 50, "Region": 50}   # last year's intake at least this big
MIN_INTAKE_SHARE = 0.05    # ... and at least 5% of the series' last-12-month volume


def _mi(d) -> int:
    d = pd.Timestamp(d)
    return d.year * 12 + d.month - 1


def _from_mi(i: int) -> pd.Timestamp:
    return pd.Timestamp(year=i // 12, month=i % 12 + 1, day=1)


# ─────────────────────────────── inputs ───────────────────────────────
def load_stage_intake(path: Path, drop_regions: tuple = ()) -> pd.DataFrame:
    """stage_intake_summary.xlsx -> Country, Region, Metric (= StageGroup), Date, Intake_Start, Apps."""
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    d = pd.read_excel(path, dtype={"Month": str, "Intake": str, "InYear": str})
    d = d.rename(columns={"Country Grouped": "Country", "Name of Region": "Region", "StageGroup": "Metric"})
    d = d[~d["Region"].isin(drop_regions)]
    return _intake_dates(d)


def _intake_dates(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["Date"] = pd.to_datetime(d["Month"].astype(str).str[:7] + "-01", errors="coerce")
    m = d["Intake"].map(parse_intake_month)
    y = pd.to_numeric(d["InYear"], errors="coerce")
    ok = d["Date"].notna() & m.notna() & y.between(2000, 2100)
    d = d[ok].copy()
    d["Intake_Start"] = pd.to_datetime(dict(year=y[ok].astype(int), month=m[ok].astype(int), day=1))
    d["Apps"] = pd.to_numeric(d["Apps"], errors="coerce").fillna(0.0)
    return d[["Country", "Region", "Metric", "Date", "Intake_Start", "Apps"]]


def intake_long(intake_df: pd.DataFrame, stage_df: pd.DataFrame = None) -> pd.DataFrame:
    """Receipts and submissions (intake extract, cleaned by analysis/intake.clean_intake) plus stage counts by
    intake, at region, destination and organisation level: Level, Country, Region, Metric, Date, Intake_Start, Apps."""
    parts = []
    if intake_df is not None and len(intake_df):
        k = intake_df[intake_df["Intake_Known"]] if "Intake_Known" in intake_df else intake_df
        parts.append(k.melt(id_vars=["Country", "Region", "Date", "Intake_Start"], value_vars=["Received", "Submitted"],
                            var_name="Metric", value_name="Apps"))
    if stage_df is not None and len(stage_df):
        parts.append(stage_df)
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    d = d[d["Apps"] > 0]
    keys = ["Metric", "Date", "Intake_Start"]
    reg = d.groupby(["Country", "Region"] + keys, as_index=False)["Apps"].sum().assign(Level="Region")
    cty = d.groupby(["Country"] + keys, as_index=False)["Apps"].sum().assign(Level="Country", Region=TOTAL)
    org = d.groupby(keys, as_index=False)["Apps"].sum().assign(Level="Organisation", Country=ORG, Region=TOTAL)
    return pd.concat([org, cty, reg], ignore_index=True)


# ─────────────────────────────── one series ───────────────────────────────
class _Series:
    """Counts of one series by intake (month index) and application month (month index)."""

    def __init__(self, g: pd.DataFrame):
        self.by: Dict[int, Dict[int, float]] = {}
        for (i, m), a in g.groupby([g["Intake_Start"].map(_mi), g["Date"].map(_mi)])["Apps"].sum().items():
            self.by.setdefault(int(i), {})[int(m)] = float(a)

    def cum(self, intake: int, upto: int) -> float:
        return float(sum(a for m, a in self.by.get(intake, {}).items() if m <= upto))

    def final(self, intake: int) -> float:
        return self.cum(intake, intake + FINAL_AFTER)

    def share_hat(self, intake: int, lead: int, known_by: int) -> Tuple[float, int]:
        """Median share already in at `lead`, over the last SHARE_YEARS same-month intakes complete by `known_by`."""
        s = []
        for k in range(1, 8):
            j = intake - 12 * k
            if j + FINAL_AFTER > known_by or j not in self.by:
                continue
            f = self.final(j)
            if f > 0:
                s.append(self.cum(j, j - lead) / f)
            if len(s) == SHARE_YEARS:
                break
        return (float(np.median(s)), len(s)) if s else (np.nan, 0)


def _backtest(series: Dict[tuple, _Series], last: int) -> pd.DataFrame:
    """Projection vs 'same as last year's final' on completed intakes, by level x metric x lead."""
    recs = []
    for key, s in series.items():
        lvl = key[0]
        for j in s.by:
            if j + FINAL_AFTER > last or j < last - 12 * 5:
                continue
            f, ly = s.final(j), s.final(j - 12)
            if f < MIN_INTAKE[lvl] or ly <= 0:
                continue
            for lead in range(MIN_LEAD, MAX_LEAD + 1):
                sh, n = s.share_hat(j, lead, j - lead)       # only intakes complete at that time
                if n < 2 or not np.isfinite(sh) or sh < MIN_SHARE:
                    continue
                p = s.cum(j, j - lead) / sh
                recs.append({"Level": lvl, "Metric": key[3], "Lead": lead, "Proj_APE": 100 * abs(p / f - 1),
                             "Base_APE": 100 * abs(ly / f - 1), "LogErr": np.log(f / p) if p > 0 else np.nan})
    r = pd.DataFrame(recs)
    if r.empty:
        return pd.DataFrame(columns=["Level", "Metric", "Lead", "N", "Projection_Miss_Pct", "Last_Year_Miss_Pct",
                                     "Q10", "Q90", "Used"])
    out = (r.groupby(["Level", "Metric", "Lead"])
             .agg(N=("Proj_APE", "size"), Projection_Miss_Pct=("Proj_APE", "median"),
                  Last_Year_Miss_Pct=("Base_APE", "median"),
                  Q10=("LogErr", lambda x: np.nanquantile(x, 0.1)), Q90=("LogErr", lambda x: np.nanquantile(x, 0.9)))
             .reset_index())
    gain = 100 * (out["Last_Year_Miss_Pct"] - out["Projection_Miss_Pct"]) / out["Last_Year_Miss_Pct"].replace(0, np.nan)
    out["Used"] = (out["N"] >= MIN_TESTS) & (gain >= MIN_GAIN_PCT)
    return out


def intake_pace(long: pd.DataFrame, last_actual: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (pace table, cumulative curves for charts, projection test)."""
    if long is None or long.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    last = _mi(last_actual)
    series, vol12 = {}, {}
    for key, g in long.groupby(["Level", "Country", "Region", "Metric"]):
        series[key] = _Series(g)
        vol12[key] = float(g.loc[g["Date"].map(_mi) > last - 12, "Apps"].sum())
    test = _backtest(series, last)
    use = {(r.Level, r.Metric, int(r.Lead)): (r.Q10, r.Q90, r.Projection_Miss_Pct, r.Last_Year_Miss_Pct)
           for r in test.itertuples() if r.Used}
    rows, curves = [], []
    for key, s in series.items():
        lvl, c, reg, met = key
        for i in sorted(set(s.by) | {j + 12 for j in s.by}):    # incl. intakes with nothing in yet
            lead = i - last
            if not MIN_LEAD <= lead <= MAX_LEAD:
                continue
            ly_final = s.final(i - 12) if i - 12 + FINAL_AFTER <= last else s.cum(i - 12, last)
            if ly_final < max(MIN_INTAKE[lvl], MIN_INTAKE_SHARE * vol12[key]):
                continue
            to_date, same = s.cum(i, last), s.cum(i - 12, last - 12)
            pace = 100 * (to_date / same - 1) if same > 0 else np.nan
            sh, _ = s.share_hat(i, lead, last)
            proj = lo = hi = miss = base = np.nan
            ev = use.get((lvl, met, lead))
            if ev is not None and np.isfinite(sh) and sh >= MIN_SHARE and to_date > 0:
                proj = to_date / sh
                lo, hi = max(proj * np.exp(ev[0]), to_date), max(proj * np.exp(ev[1]), to_date)
                miss, base = ev[2], ev[3]
            rows.append({"Level": lvl, "Country": c, "Region": reg, "Metric": met,
                         "Intake": f"{_from_mi(i):%b %Y}", "Intake_Start": _from_mi(i), "Months_To_Start": lead,
                         "To_Date": to_date, "Last_Year_Same_Point": same, "Pace_Pct": pace,
                         "Status": ("ahead" if pace >= PACE_BAND else "behind" if pace <= -PACE_BAND else "in line")
                         if np.isfinite(pace) else "new",
                         "Typical_Share_In_Pct": 100 * sh if np.isfinite(sh) else np.nan,
                         "Last_Year_Final": ly_final, "Projected_Final": proj, "Lo80": lo, "Hi80": hi,
                         "Projection_Miss_Pct": miss, "Last_Year_Miss_Pct": base})
            if lvl != "Region":
                for name, j in (("This intake", i), ("Last year", i - 12), ("Two years ago", i - 24)):
                    if j not in s.by:
                        continue
                    end = last if j == i else j + FINAL_AFTER
                    for ld in range(15, j - end - 1, -1):
                        curves.append({"Level": lvl, "Country": c, "Metric": met, "Intake": f"{_from_mi(i):%b %Y}",
                                       "Series": name, "Months_To_Start": ld, "Cumulative": s.cum(j, j - ld)})
    pace_df = pd.DataFrame(rows)
    if len(pace_df):
        pace_df = pace_df.sort_values(["Level", "Country", "Region", "Metric", "Intake_Start"]).reset_index(drop=True)
    return pace_df, pd.DataFrame(curves), test
