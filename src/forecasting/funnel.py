# -*- coding: utf-8 -*-
"""
Funnel model for submissions (v2.2): Received is a leading indicator of Submitted.

On 2026-09 data, 12-month log changes of organisation submissions correlate 0.97 with receipts in the
same month and 0.90 with receipts one month earlier. So

    Submitted[t] = w0 * Received[t] + w1 * Received[t-1] + w2 * Received[t-2]

The weights are non-negative least squares on the last WINDOW realised months, re-estimated at every
forecast origin. Receipts up to the origin are actuals; later receipts are the published Received
forecast made at that same origin, so nothing unknown at the time is used.

Three variants compete in a walk-forward test on next-3-month totals:
  model   the current Submitted forecast;
  funnel  the formula above;
  blend   the average of the two.
The one with the lowest error averaged over levels is published, but only if it beats `model` by at
least MIN_GAIN_PCT (the same rule used for the point estimate).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

LAGS = (0, 1, 2)
WINDOW = 12
MIN_GAIN_PCT = 3.0
VARIANTS = ("model", "funnel", "blend")


def _weights(R: np.ndarray, S: np.ndarray, o: int) -> np.ndarray:
    tt = np.arange(o - WINDOW + 1, o + 1)
    if tt[0] - max(LAGS) < 0:
        return np.full(len(LAGS), np.nan)
    X = np.column_stack([R[tt - L] for L in LAGS])
    if X.sum() <= 0:
        return np.full(len(LAGS), np.nan)
    # non-negative least squares by active-set elimination (3 coefficients)
    active = list(range(len(LAGS)))
    while active:
        w, *_ = np.linalg.lstsq(X[:, active], S[tt], rcond=None)
        if (w >= 0).all():
            out = np.zeros(len(LAGS))
            out[active] = w
            return out
        active.pop(int(np.argmin(w)))
    return np.zeros(len(LAGS))


def full_series(panel) -> Dict[str, np.ndarray]:
    """node_id -> actuals on the global month index (zeros before the series starts)."""
    T = len(panel.months)
    out = {}
    for nid, y in panel.y.items():
        a = np.zeros(T)
        a[panel.start[nid]:] = y
        out[nid] = a
    return out


def funnel_forecast(frame: pd.DataFrame, col: str, Y: Dict[str, np.ndarray]) -> pd.Series:
    """Funnel prediction for every Submitted row of `frame` (node_id, o, h, <col>), aligned to frame.index.
    Received for future months comes from the same frame (same origin), Received actuals from Y."""
    out = pd.Series(np.nan, index=frame.index)
    sub = frame[frame["node_id"].str.startswith("Submitted|")]
    rec_rows = frame[frame["node_id"].str.startswith("Received|")]
    rec_pred = {(n, o): g.set_index("h")[col] for (n, o), g in rec_rows.groupby(["node_id", "o"])}
    for (sid, o), g in sub.groupby(["node_id", "o"]):
        rid = "Received|" + sid.split("|", 1)[1]
        if rid not in Y or sid not in Y:
            continue
        w = _weights(Y[rid], Y[sid], int(o))
        if not np.isfinite(w).all():
            continue
        rp = rec_pred.get((rid, o))
        for idx, h in zip(g.index, g["h"]):
            val = 0.0
            for wi, L in zip(w, LAGS):
                t = int(o) + int(h) - L
                if t <= o:
                    val += wi * Y[rid][t]
                elif rp is not None and (t - o) in rp.index:
                    val += wi * float(rp.loc[t - o])
                else:
                    val = np.nan
                    break
            out.loc[idx] = val
    return out


def variant(frame: pd.DataFrame, col: str, fun: pd.Series, name: str) -> pd.Series:
    base = frame[col]
    if name == "model":
        return base
    f = fun.where(fun.notna(), base)
    return f if name == "funnel" else 0.5 * (base + f)


def walk_forward_test(honest: pd.DataFrame, nodes: pd.DataFrame, Y: Dict[str, np.ndarray],
                      k: int = 3) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """Median absolute % error of next-k-month totals of Submitted, per level, for each variant."""
    if honest is None or honest.empty:
        return pd.DataFrame(), {}
    fun = funnel_forecast(honest, "Policy", Y)
    lvl = nodes.set_index("node_id")["Level"]
    variants, rows = {}, []
    for v in VARIANTS:
        pv = variant(honest, "Policy", fun, v)
        variants[v] = pv
        d = honest.assign(P=pv)
        d = d[d["node_id"].str.startswith("Submitted|") & (d["h"] <= k)]
        full = d.groupby(["node_id", "o"])["h"].transform("nunique") == k
        q = d[full].groupby(["node_id", "o"])[["P", "Actual"]].sum().reset_index()
        q = q[q["Actual"] > 0]
        q["APE"] = 100 * (q["P"] - q["Actual"]).abs() / q["Actual"]
        q["Level"] = q["node_id"].map(lvl)
        for L, g in q.groupby("Level"):
            rows.append({"Variant": v, "Level": L, "Quarter_Error_Pct": float(g["APE"].median()), "N": len(g)})
    return pd.DataFrame(rows), variants


def decide(test: pd.DataFrame, min_gain_pct: float = MIN_GAIN_PCT) -> str:
    """Variant with the lowest level-averaged quarter error; 'model' unless the gain is clear."""
    if test.empty:
        return "model"
    avg = test.groupby("Variant")["Quarter_Error_Pct"].mean()
    best = str(avg.idxmin())
    if best == "model" or "model" not in avg:
        return "model"
    return best if 100 * (avg["model"] - avg[best]) / avg["model"] >= min_gain_pct else "model"
