# -*- coding: utf-8 -*-
"""
Direction signal from the latest forecast vintages (v2.2): detect a turning point quickly.

Before a new forecast is issued at month-end o, look at the most recent forecasts and what has since
happened:
  - vintages made at o-5, o-4, o-3: their three months are all known (three full quarters);
  - vintages made at o-2 and o-1: their first two months and first month are known (current momentum).
The signal is the recency-weighted log(actual / forecast) over those known months (a month counts half
as much for every HALF_LIFE months it is older). It is read at the COUNTRY total, where noise cancels,
and applied to the country's regions and to the country: forecast x exp(LAMBDA x signal).

The past vintages are re-created from scratch by the walk-forward replay in every run, so the signal
in production is computed exactly as in the test. Like every other component, it is re-tested on the
next-quarter totals each run and used only while it is at least MIN_GAIN_PCT better, averaged over levels.

Scope: Received only. Submitted inherits the correction through the funnel (receipts -> submissions);
correcting submissions directly made the organisation worse in testing.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

LAMBDA = 0.5          # half the measured gap: the gap is a noisy estimate of the true lag
HALF_LIFE = 2.0       # months
VINTAGES = 5          # o-5 .. o-1
MIN_PAIRS = 6
MIN_GAIN_PCT = 3.0
METRIC = "Received"


def _signal(g: pd.DataFrame, o: int, col: str) -> float:
    t = g["o"] + g["h"]
    x = g[(g["o"] >= o - VINTAGES) & (g["o"] <= o - 1) & (t <= o) & (g["h"] <= 3)]
    if len(x) < MIN_PAIRS:
        return np.nan
    w = 0.5 ** ((o - (x["o"] + x["h"])) / HALF_LIFE)
    a, f = float((w * x["Actual"]).sum()), float((w * x[col]).sum())
    return float(np.log(a / f)) if a > 0 and f > 0 else np.nan


def signal_node(nodes: pd.DataFrame) -> Dict[str, str]:
    """Series -> the series whose signal it uses: its country total (itself for single-region
    countries). The organisation and Submitted series are not corrected."""
    ids = set(nodes["node_id"])
    out = {}
    for r in nodes.itertuples(index=False):
        if r.Metric != METRIC or r.Level == "Organisation":
            continue
        cand = f"{r.Metric}|{r.Country}|_TOTAL"
        out[r.node_id] = cand if cand in ids else r.node_id
    return out


def signals(honest: pd.DataFrame, origins, col: str = "Policy") -> Dict[Tuple[str, int], float]:
    """(series, origin) -> signal, from vintages realised by that origin only."""
    by = {n: g for n, g in honest.groupby("node_id")}
    return {(n, o): _signal(g, o, col) for n, g in by.items() for o in origins}


def factors(frame: pd.DataFrame, nodes: pd.DataFrame, sig: Dict[Tuple[str, int], float],
            origin_col: str = "o", fixed_origin: int = None) -> np.ndarray:
    """Multiplier for every row of frame (1 where not corrected or no signal)."""
    src = signal_node(nodes)
    o = frame[origin_col].to_numpy() if fixed_origin is None else np.full(len(frame), fixed_origin)
    f = np.ones(len(frame))
    for i, (nid, oo) in enumerate(zip(frame["node_id"].to_numpy(), o)):
        s = src.get(nid)
        if s is None:
            continue
        v = sig.get((s, int(oo)), np.nan)
        if np.isfinite(v):
            f[i] = float(np.exp(LAMBDA * v))
    return f


def apply_published(frame: pd.DataFrame, col: str, nodes: pd.DataFrame, sig: Dict[Tuple[str, int], float],
                    fixed_origin: int = None) -> pd.Series:
    """Correct the published (coherent) forecast: each receipt REGION x its country's factor, then every
    receipt country and the organisation = the sum of its corrected regions, so the numbers still add up.
    Returns the corrected column aligned to frame.index."""
    meta = nodes.set_index("node_id")
    out = frame[col].astype(float).copy()
    lvl = frame["node_id"].map(meta["Level"])
    met = frame["node_id"].map(meta["Metric"])
    cty = frame["node_id"].map(meta["Country"])
    reg = (lvl == "Region") & (met == METRIC)
    fac = factors(frame[reg], nodes, sig, fixed_origin=fixed_origin)
    out.loc[reg] = frame.loc[reg, col].to_numpy() * fac
    keys = ["h"] if fixed_origin is not None or "o" not in frame.columns else ["o", "h"]
    r = frame.loc[reg, keys].assign(_v=out[reg].to_numpy(), _c=cty[reg].to_numpy(), _b=frame.loc[reg, col].to_numpy())
    by_c = r.groupby(keys + ["_c"])[["_v", "_b"]].sum()
    by_o = r.groupby(keys)[["_v", "_b"]].sum()
    for mask, table, extra in (((lvl == "Country") & (met == METRIC), by_c, True),
                               ((lvl == "Organisation") & (met == METRIC), by_o, False)):
        idx = frame.index[mask]
        if not len(idx):
            continue
        k = [tuple(x) for x in frame.loc[idx, keys].to_numpy()]
        if extra:
            k = [kk + (c,) for kk, c in zip(k, cty[idx])]
        t = table.reindex(pd.MultiIndex.from_tuples(k) if len(keys) + extra > 1 else [x[0] for x in k])
        ratio = (t["_v"] / t["_b"].replace(0, np.nan)).to_numpy()
        # scale the aggregate by its regions' combined correction (exactly the sum when bottom-up)
        out.loc[idx] = np.where(np.isfinite(ratio), frame.loc[idx, col].to_numpy() * ratio, frame.loc[idx, col].to_numpy())
    return out


def walk_forward_test(honest: pd.DataFrame, nodes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Next-quarter median abs % error by level, without and with the correction (Received), on the
    published forecasts."""
    if honest is None or honest.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    origins = sorted(honest["o"].unique())
    sig = signals(honest, origins)
    corrected = apply_published(honest, "Policy", nodes, sig)
    lvl = nodes.set_index("node_id")["Level"]
    rows = []
    for name, p in (("none", honest["Policy"]), ("vintage", corrected)):
        d = honest.assign(P=p)
        d = d[d["node_id"].str.startswith(METRIC + "|") & (d["h"] <= 3)]
        full = d.groupby(["node_id", "o"])["h"].transform("nunique") == 3
        q = d[full].groupby(["node_id", "o"])[["P", "Actual"]].sum().reset_index()
        q = q[(q["Actual"] > 0) & (q["o"] >= min(origins) + VINTAGES)]
        q["APE"] = 100 * (q["P"] - q["Actual"]).abs() / q["Actual"]
        q["Level"] = q["node_id"].map(lvl)
        for L, g in q.groupby("Level"):
            rows.append({"Variant": name, "Level": L, "Quarter_Error_Pct": float(g["APE"].median()), "N": len(g)})
    return pd.DataFrame(rows), corrected


def decide(test: pd.DataFrame, min_gain_pct: float = MIN_GAIN_PCT) -> bool:
    if test.empty:
        return False
    avg = test.groupby("Variant")["Quarter_Error_Pct"].mean()
    return bool("vintage" in avg and 100 * (avg["none"] - avg["vintage"]) / avg["none"] >= min_gain_pct)
