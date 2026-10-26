# -*- coding: utf-8 -*-
"""
Intake running indicator ("booking curve") by intake SEASON: how far each season's intake has filled, compared
with the same season a year earlier at the same point, and what that implies for its final size.

Seasons are defined on the INTAKE month (the month the student starts), not on the month an application was
received or changed status:
    Spring = Jan, Feb, Mar intakes      Summer = Apr, May, Jun, Jul intakes      Fall = Aug - Dec intakes
The same seasons are used for every destination and region, so they compare directly.

Seen from the last closed month, the seasons are labelled:
    Closing   the season the current month falls in (e.g. Fall 2026 in Oct 2026): nearly complete
    Ongoing   the next season to start (Spring 2027): the one being recruited for now
    Next      the two after it (Summer 2027, Fall 2027)

For one series (organisation, destination or region x metric: Received, Submitted, Offer, Deposit, visa...) and
one season S, with L = months from the last closed month to the season's first intake month:
  to date        applications for S's intakes counted up to the last closed month
  same point     the same season a year earlier, counted up to the same point (12 months earlier)
  pace           to date / same point - 1     (ahead / in line / behind at +-5%)
  projection     to date / s, s = median share of the final count that the last 3 completed same seasons already
                 had L months before their start ("final" = counted up to FINAL_AFTER months after the season's
                 last intake month)

The projection is tested walk-forward on completed seasons: re-made at every lead with only seasons complete at
the time and scored against their final count, next to the baseline "same as last year's final". It is published
for a level x metric x lead only where it beat that baseline by MIN_GAIN_PCT (the common decision rule) on at least
MIN_TESTS seasons and its range (the 10/90 quantiles of those test errors) spans at most MAX_RANGE_RATIO. Otherwise
only the pace is shown ("too early to project").
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..data.prepare import ORG, TOTAL
from .intake import parse_intake_month

SEASONS = (("Spring", 1, 3), ("Summer", 4, 7), ("Fall", 8, 12))    # name, first and last intake month
FINAL_AFTER = 3            # a season's count is final 3 months after its last intake month (late / deferred entries)
MAX_LEAD = 12
SHARE_YEARS = 3            # completed same seasons used for the typical share
MIN_SHARE = 0.10           # too early to project when under 10% of a season is usually in
MIN_GAIN_PCT = 3.0
MIN_TESTS = 8
MAX_RANGE_RATIO = 2.5      # a projection whose 80% range spans more than 2.5x is not usable: pace only
PACE_BAND = 5.0
MIN_INTAKE = {"Organisation": 100, "Country": 30, "Region": 30}   # last year's season at least this big
SHOW_NEXT = 3              # ongoing + the two after it


def _mi(d) -> int:
    d = pd.Timestamp(d)
    return d.year * 12 + d.month - 1


def _from_mi(i: int) -> pd.Timestamp:
    return pd.Timestamp(year=i // 12, month=i % 12 + 1, day=1)


def season_of(i: int) -> Tuple[int, str, int]:
    """Intake month index -> (season start month index, season name, season length in months)."""
    y, m = divmod(i, 12)
    for name, a, b in SEASONS:
        if a <= m + 1 <= b:
            return y * 12 + a - 1, name, b - a + 1
    raise ValueError(i)


def season_label(s: int) -> str:
    _, name, _ = season_of(s)
    return f"{name} {s // 12}"


def _end(s: int) -> int:
    start, _, n = season_of(s)
    return start + n - 1


def seasons_in_view(last: int) -> List[Tuple[int, str]]:
    """[(season start, phase)]: the season containing the last closed month (Closing), then the next SHOW_NEXT."""
    cur = season_of(last)[0]
    out, s = [(cur, "Closing")], cur
    for k in range(SHOW_NEXT):
        s = _end(s) + 1
        out.append((s, "Ongoing" if k == 0 else "Next"))
    return out


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
    """Counts of one series by intake season (season start month index) and application month (month index)."""

    def __init__(self, g: pd.DataFrame):
        self.by: Dict[int, Dict[int, float]] = {}
        seas = g["Intake_Start"].map(lambda d: season_of(_mi(d))[0])
        for (s, m), a in g.groupby([seas, g["Date"].map(_mi)])["Apps"].sum().items():
            self.by.setdefault(int(s), {})[int(m)] = float(a)

    def cum(self, s: int, upto: int) -> float:
        return float(sum(a for m, a in self.by.get(s, {}).items() if m <= upto))

    def final(self, s: int) -> float:
        return self.cum(s, _end(s) + FINAL_AFTER)

    def share_hat(self, s: int, lead: int, known_by: int) -> Tuple[float, int]:
        """Median share already in at `lead`, over the last SHARE_YEARS same seasons complete by `known_by`."""
        out = []
        for k in range(1, 8):
            j = s - 12 * k
            if _end(j) + FINAL_AFTER > known_by or j not in self.by:
                continue
            f = self.final(j)
            if f > 0:
                out.append(self.cum(j, j - lead) / f)
            if len(out) == SHARE_YEARS:
                break
        return (float(np.median(out)), len(out)) if out else (np.nan, 0)


def _backtest(series: Dict[tuple, _Series], last: int) -> pd.DataFrame:
    """Projection vs 'same as last year's final' on completed seasons, by level x metric x lead."""
    recs = []
    for key, s in series.items():
        lvl = key[0]
        for j in s.by:
            if _end(j) + FINAL_AFTER > last or j < last - 12 * 5:
                continue
            f, ly = s.final(j), s.final(j - 12)
            if f < MIN_INTAKE[lvl] or ly <= 0:
                continue
            for lead in range(-(_end(j) - j), MAX_LEAD + 1):
                sh, n = s.share_hat(j, lead, j - lead)       # only seasons complete at that time
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
    """Returns (pace table: one row per series x season in view, cumulative curves for charts, projection test)."""
    if long is None or long.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    last = _mi(last_actual)
    series = {key: _Series(g) for key, g in long.groupby(["Level", "Country", "Region", "Metric"])}
    test = _backtest(series, last)
    use = {(r.Level, r.Metric, int(r.Lead)): (r.Q10, r.Q90, r.Projection_Miss_Pct, r.Last_Year_Miss_Pct)
           for r in test.itertuples() if r.Used}
    view = seasons_in_view(last)
    rows, curves = [], []
    for key, s in series.items():
        lvl, c, reg, met = key
        for st, phase in view:
            lead = st - last
            ly = st - 12
            ly_final = s.final(ly) if _end(ly) + FINAL_AFTER <= last else s.cum(ly, last)
            if ly_final < MIN_INTAKE[lvl] and s.cum(st, last) < MIN_INTAKE[lvl]:
                continue
            to_date, same = s.cum(st, last), s.cum(ly, last - 12)
            pace = 100 * (to_date / same - 1) if same > 0 else np.nan
            sh, _ = s.share_hat(st, lead, last)
            proj = lo = hi = miss = base = np.nan
            ev = use.get((lvl, met, lead))
            if ev is not None and np.isfinite(sh) and sh >= MIN_SHARE and to_date > 0 and np.exp(ev[1] - ev[0]) <= MAX_RANGE_RATIO:
                proj = to_date / sh
                lo, hi = max(proj * np.exp(ev[0]), to_date), max(proj * np.exp(ev[1]), to_date)
                miss, base = ev[2], ev[3]
            _, name, _ = season_of(st)
            rows.append({"Level": lvl, "Country": c, "Region": reg, "Metric": met, "Intake": season_label(st),
                         "Season": name, "Season_Year": st // 12, "Season_Start": _from_mi(st), "Phase": phase,
                         "Months_To_Start": lead, "To_Date": to_date, "Last_Year_Same_Point": same, "Pace_Pct": pace,
                         "Status": ("ahead" if pace >= PACE_BAND else "behind" if pace <= -PACE_BAND else "in line")
                         if np.isfinite(pace) else "new",
                         "Typical_Share_In_Pct": 100 * sh if np.isfinite(sh) else np.nan,
                         "Last_Year_Final": ly_final, "Projected_Final": proj, "Lo80": lo, "Hi80": hi,
                         "Projection_Miss_Pct": miss, "Last_Year_Miss_Pct": base})
            if lvl != "Region":
                for name_, j in (("This intake", st), ("Last year", st - 12), ("Two years ago", st - 24)):
                    if j not in s.by:
                        continue
                    end = last if j == st else _end(j) + FINAL_AFTER
                    for ld in range(15, j - end - 1, -1):
                        curves.append({"Level": lvl, "Country": c, "Metric": met, "Intake": season_label(st),
                                       "Series": name_, "Months_To_Start": ld, "Cumulative": s.cum(j, j - ld)})
    pace_df = pd.DataFrame(rows)
    if len(pace_df):
        pace_df = pace_df.sort_values(["Level", "Country", "Region", "Metric", "Season_Start"]).reset_index(drop=True)
    return pace_df, pd.DataFrame(curves), test
