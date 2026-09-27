# -*- coding: utf-8 -*-
"""
Reliability of the PUBLISHED forecast, measured the way it is used.

Decisions (staffing, targets, budgets) are taken on quarter and year totals, not on a single
month, and on the reconciled numbers that are actually published. Reliability is therefore the
typical absolute % error of the next-3-month total in the walk-forward test (forecasts made only
with data available at the time):

  High    typical quarter error <= 10%
  Medium  typical quarter error <= 20%
  Low     typical quarter error  > 20%   (usually small or very volatile markets)
  New     too little history to test

"Best method" and "reliability" answer different questions: every series uses the most accurate
method found in testing; reliability says how precisely that series can be forecast at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLDS = (10.0, 20.0)
MIN_TESTS = 4


def published_backtest(comb: pd.DataFrame, nodes: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Backtest predictions as they would have been published. With bottom-up reconciliation an
    aggregate's forecast is the sum of its regions' forecasts, so its errors (and therefore its
    ranges) must come from those sums, not from the aggregate's own unreconciled model.
    """
    if method != "bottom_up" or comb.empty:
        return comb
    meta = nodes.set_index("node_id")
    reg = nodes[nodes["Level"] == "Region"]
    out = [comb[comb["node_id"].isin(reg["node_id"])]]
    for _, a in nodes[nodes["Level"].isin(["Country", "Organisation"])].iterrows():
        members = reg[(reg["Metric"] == a["Metric"]) &
                      ((reg["Country"] == a["Country"]) if a["Level"] == "Country" else True)]["node_id"]
        mb = comb[comb["node_id"].isin(members)]
        if mb.empty:
            continue
        cnt = mb.groupby(["o", "h"])["node_id"].nunique()
        full = cnt[cnt == len(members)].index
        s = mb.set_index(["o", "h"]).loc[lambda d: d.index.isin(full)].groupby(level=["o", "h"])["Pred"].sum()
        own = comb[comb["node_id"] == a["node_id"]].set_index(["o", "h"])
        own = own[own.index.isin(s.index)].copy()
        if own.empty:
            continue
        own["Pred"] = s.reindex(own.index).to_numpy()
        out.append(own.reset_index())
    res = pd.concat(out, ignore_index=True)
    return res[comb.columns]


def reliability(honest: pd.DataFrame, nodes: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    cols = ["node_id", "Typical_Quarter_Error_Pct", "Quarter_Error_80_Pct", "Quarter_Tests", "Reliability"]
    if honest is None or honest.empty:
        return pd.DataFrame(columns=cols)
    b = honest[honest["h"] <= k]
    full = b.groupby(["node_id", "o"])["h"].transform("nunique") == k
    s = b[full].groupby(["node_id", "o"])[["Policy", "Actual"]].sum()
    s["APE"] = 100 * (s["Policy"] - s["Actual"]).abs() / s["Actual"].replace(0, np.nan)
    g = s.groupby(level="node_id")["APE"]
    r = pd.DataFrame({"Typical_Quarter_Error_Pct": g.median(), "Quarter_Error_80_Pct": g.quantile(0.8),
                      "Quarter_Tests": g.count()}).reset_index()
    hi, med = THRESHOLDS
    r["Reliability"] = np.where(r["Quarter_Tests"] < MIN_TESTS, "New",
                                np.where(r["Typical_Quarter_Error_Pct"] <= hi, "High",
                                         np.where(r["Typical_Quarter_Error_Pct"] <= med, "Medium", "Low")))
    missing = nodes.loc[~nodes["node_id"].isin(r["node_id"]), ["node_id"]].assign(
        Typical_Quarter_Error_Pct=np.nan, Quarter_Error_80_Pct=np.nan, Quarter_Tests=0, Reliability="New")
    return pd.concat([r, missing], ignore_index=True)[cols]


def label_text(rel: str, err: float) -> str:
    """Plain-English reliability for reports."""
    if rel == "New":
        return "New market — too little history to test"
    if not np.isfinite(err):
        return rel
    return f"{rel} — quarter totals typically within ±{err:.0f}%"
