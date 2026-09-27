# -*- coding: utf-8 -*-
"""
Plausibility guardrails and output invariants.

Guardrail: the 12-month forecast total is compared with two references built only
from actuals —
    level     = last 12 months actual total
    momentum  = level x (last 3 months / same 3 months a year earlier), clipped to [0.25, 4]
Plausible range = [min_ratio x min(level, momentum), max_ratio x max(level, momentum)].
A genuinely growing series passes (its momentum reference is high); a structural
failure (an exploding or collapsing model) does not. On a trigger the forecast
path is scaled to the violated bound (seasonal shape kept) and flagged for review.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def references(y: np.ndarray) -> Dict[str, float]:
    level = float(y[-12:].sum())
    prev3 = float(y[-15:-12].sum()) if len(y) >= 15 else 0.0
    g = float(np.clip(y[-3:].sum() / prev3, 0.25, 4.0)) if prev3 > 0 else 1.0
    return {"Actual_Last_12m": level, "Momentum_Reference": level * g}


def plausibility(base: pd.DataFrame, refs: pd.DataFrame, min_ratio: float, max_ratio: float) -> pd.DataFrame:
    """base: node_id, h, Pred. refs: index node_id, columns Actual_Last_12m, Momentum_Reference."""
    tot = base[base["h"] <= 12].groupby("node_id")["Pred"].sum().rename("Forecast_12m")
    df = refs.join(tot, how="inner")
    lo_ref = df[["Actual_Last_12m", "Momentum_Reference"]].min(axis=1)
    hi_ref = df[["Actual_Last_12m", "Momentum_Reference"]].max(axis=1)
    df["Lower_Bound"], df["Upper_Bound"] = min_ratio * lo_ref, max_ratio * hi_ref
    df["Ratio_vs_Last_12m"] = df["Forecast_12m"] / df["Actual_Last_12m"].replace(0, np.nan)
    df["Guardrail_Triggered"] = (df["Actual_Last_12m"] > 0) & (
        (df["Forecast_12m"] > df["Upper_Bound"]) | (df["Forecast_12m"] < df["Lower_Bound"]))
    df["Scale_Applied"] = np.where(
        df["Guardrail_Triggered"],
        np.where(df["Forecast_12m"] > df["Upper_Bound"], df["Upper_Bound"], df["Lower_Bound"])
        / df["Forecast_12m"].replace(0, np.nan), 1.0)
    return df.rename_axis("node_id").reset_index()


def check_invariants(fc: pd.DataFrame, last_actual: pd.Timestamp, levels=(80, 95)) -> Dict[str, int]:
    """Hard checks on the final forecast; raises AssertionError on violation."""
    problems = {}
    problems["negative_forecasts"] = int((fc["Forecast"] < -1e-9).sum())
    problems["past_dated_rows"] = int((fc["Date"] <= last_actual).sum())
    for L in levels:
        lo, hi = fc[f"Lo{L}"], fc[f"Hi{L}"]
        problems[f"point_outside_{L}"] = int(((fc["Forecast"] < lo - 1e-6) | (fc["Forecast"] > hi + 1e-6)).sum())
    problems["interval_not_nested"] = int(((fc["Lo95"] > fc["Lo80"] + 1e-6) | (fc["Hi95"] < fc["Hi80"] - 1e-6)).sum())
    bad = {k: v for k, v in problems.items() if v}
    assert not bad, f"Forecast invariants violated: {bad}"
    return problems


def coherence_gap(fc: pd.DataFrame) -> float:
    """Max absolute gap between each aggregate and the sum of its regions (should be ~0)."""
    reg = fc[fc["Level"] == "Region"]
    gaps = []
    for (metric, date), g in reg.groupby(["Metric", "Date"]):
        org = fc[(fc["Level"] == "Organisation") & (fc["Metric"] == metric) & (fc["Date"] == date)]
        if len(org):
            gaps.append(abs(float(org["Forecast"].iloc[0]) - float(g["Forecast"].sum())))
    return max(gaps) if gaps else 0.0
