# -*- coding: utf-8 -*-
"""
Calendar and festival effects per reporting cycle (26th of the previous month to the 25th).

Seasonal models repeat last year's pattern, so what they miss is the DIFFERENCE vs the same cycle last
year:
  - working days (Mon-Fri) in the cycle;
  - moving festivals falling inside the cycle: Diwali, Holi, Eid al-Fitr.
On 2026-09 data (pooled, in-sample): +5.3% per extra working day, Diwali -8.5%, Eid -11.6%, Holi ~0.

log(actual / forecast) = b . (x_cycle - x_same_cycle_last_year)

The coefficients are fitted at every origin on realised misses of the published organisation and country
receipt forecasts only, and bounded. The resulting factor is the same for every series in a month, so
totals still add up. Purpose: the right MONTH for workload planning. The component is adopted when it
lowers the monthly error of months 1-3 by at least MIN_GAIN_PCT (level-averaged) and does not worsen
quarter totals. Submitted is not adjusted (applied directly it hurt countries).

MAINTENANCE: extend FESTIVALS every year (dates as observed in India).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

FESTIVALS: Dict[str, Tuple[str, ...]] = {
    "Diwali": ("2019-10-27", "2020-11-14", "2021-11-04", "2022-10-24", "2023-11-12", "2024-11-01",
               "2025-10-20", "2026-11-08", "2027-10-29", "2028-10-17", "2029-11-05"),
    "Holi": ("2019-03-21", "2020-03-10", "2021-03-29", "2022-03-18", "2023-03-08", "2024-03-25",
             "2025-03-14", "2026-03-04", "2027-03-22", "2028-03-11", "2029-03-01"),
    "Eid": ("2019-06-05", "2020-05-25", "2021-05-14", "2022-05-03", "2023-04-22", "2024-04-11",
            "2025-03-31", "2026-03-21", "2027-03-10", "2028-02-27", "2029-02-15"),
}
FEATURES = ("WD", "Diwali", "Holi", "Eid")
BOUNDS = {"WD": 0.10, "Diwali": 0.25, "Holi": 0.25, "Eid": 0.25}   # max |effect| on the log scale
MIN_ORIGINS = 6
MIN_GAIN_PCT = 3.0
METRIC = "Received"


def cycle_features(month: pd.Timestamp) -> Dict[str, int]:
    m = pd.Timestamp(month)
    a, b = (m - pd.DateOffset(months=1)).replace(day=26), m.replace(day=25)
    days = pd.date_range(a, b)
    f = {"WD": int((days.dayofweek < 5).sum())}
    for k, ds in FESTIVALS.items():
        f[k] = int(any(a <= pd.Timestamp(d) <= b for d in ds))
    return f


def deltas(months: pd.DatetimeIndex) -> pd.DataFrame:
    """Index = position in `months`; columns = feature difference vs 12 months earlier."""
    f = pd.DataFrame([cycle_features(m) for m in months])
    return (f - f.shift(12))[list(FEATURES)]


def _fit(e: np.ndarray, X: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(A, e, rcond=None)
    return np.array([np.clip(v, -BOUNDS[k], BOUNDS[k]) for k, v in zip(FEATURES, b[1:])])


def _frame(honest: pd.DataFrame, nodes: pd.DataFrame, D: pd.DataFrame, col: str) -> pd.DataFrame:
    lvl = nodes.set_index("node_id")["Level"]
    d = honest[honest["node_id"].str.startswith(METRIC + "|")].copy()
    d = d[d["node_id"].map(lvl).isin(["Organisation", "Country"])]
    d["t"] = d["o"] + d["h"]
    d = d[(d["Actual"] > 0) & (d[col] > 0)]
    X = D.reindex(d["t"].to_numpy()).to_numpy()
    ok = np.isfinite(X).all(axis=1)
    d, X = d[ok], X[ok]
    d["e"] = np.log(d["Actual"] / d[col])
    return d.assign(**{f"x_{k}": X[:, i] for i, k in enumerate(FEATURES)})


def coefficients(honest: pd.DataFrame, nodes: pd.DataFrame, D: pd.DataFrame, origins, col: str = "Policy"):
    """origin -> bounded coefficients fitted on misses realised by that origin (t <= origin)."""
    d = _frame(honest, nodes, D, col)
    xs = [f"x_{k}" for k in FEATURES]
    out = {}
    for o in origins:
        past = d[d["t"] <= o]
        if past["o"].nunique() >= MIN_ORIGINS:
            out[int(o)] = _fit(past["e"].to_numpy(), past[xs].to_numpy())
    return out


def factors(frame: pd.DataFrame, D: pd.DataFrame, coefs: dict, fixed_origin: int = None, t_offset: int = 0) -> np.ndarray:
    """Multiplier per row (Received rows only; 1 elsewhere or where no coefficients exist)."""
    o = frame["o"].to_numpy() if fixed_origin is None else np.full(len(frame), fixed_origin)
    t = o + frame["h"].to_numpy() + t_offset
    rec = frame["node_id"].str.startswith(METRIC + "|").to_numpy()
    X = D.reindex(t).to_numpy()
    f = np.ones(len(frame))
    for i in np.nonzero(rec)[0]:
        b = coefs.get(int(o[i]))
        if b is not None and np.isfinite(X[i]).all():
            f[i] = float(np.exp(X[i] @ b))
    return f


def walk_forward_test(honest: pd.DataFrame, nodes: pd.DataFrame, D: pd.DataFrame):
    """Monthly (months 1-3) and next-quarter errors by level, without and with calendar effects."""
    if honest is None or honest.empty:
        return pd.DataFrame(), None, {}
    origins = sorted(honest["o"].unique())
    coefs = coefficients(honest, nodes, D, origins)
    corrected = honest["Policy"] * factors(honest, D, coefs)
    lvl = nodes.set_index("node_id")["Level"]
    rows = []
    tested = set(coefs)
    for name, p in (("none", honest["Policy"]), ("calendar", corrected)):
        d = honest.assign(P=p)
        d = d[d["node_id"].str.startswith(METRIC + "|") & (d["h"] <= 3) & d["o"].isin(tested)]
        d = d.assign(Level=d["node_id"].map(lvl))
        m = d[d["Actual"] > 0]
        for L, g in m.groupby("Level"):
            rows.append({"Variant": name, "Level": L, "Window": "Month",
                         "Error_Pct": float((100 * (g["P"] - g["Actual"]).abs() / g["Actual"]).median())})
        full = d.groupby(["node_id", "o"])["h"].transform("nunique") == 3
        q = d[full].groupby(["node_id", "o", "Level"])[["P", "Actual"]].sum().reset_index()
        q = q[q["Actual"] > 0]
        for L, g in q.groupby("Level"):
            rows.append({"Variant": name, "Level": L, "Window": "Quarter",
                         "Error_Pct": float((100 * (g["P"] - g["Actual"]).abs() / g["Actual"]).median())})
    return pd.DataFrame(rows), corrected, coefs


def decide(test: pd.DataFrame, min_gain_pct: float = MIN_GAIN_PCT) -> bool:
    if test is None or test.empty:
        return False
    avg = test.groupby(["Window", "Variant"])["Error_Pct"].mean()
    try:
        m_gain = 100 * (avg[("Month", "none")] - avg[("Month", "calendar")]) / avg[("Month", "none")]
        q_worse = avg[("Quarter", "calendar")] > avg[("Quarter", "none")] * 1.005
    except KeyError:
        return False
    return bool(m_gain >= min_gain_pct and not q_worse)
