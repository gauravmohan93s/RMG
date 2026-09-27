# -*- coding: utf-8 -*-
"""
Empirical prediction intervals.

1. Residuals of the combined (selected) forecast from the rolling-origin backtest,
   on the log scale: r = log1p(actual) - log1p(forecast)  -> relative errors.
2. Per node and horizon h, quantiles of r from horizons h-1..h+1; if fewer than
   ``min_obs`` residuals, pool across nodes of the same metric and level.
3. Enforce q_lo <= 0 <= q_hi (the point lies inside) and widening with horizon.
4. Calibrate: in the honest walk-forward test, the smallest factor c >= 1 that
   makes the interval reach its nominal coverage is applied (per metric/level).
Totals over the next 3/6/12 months get their own quantiles from block errors,
because monthly quantiles cannot be added.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

BLOCKS = (3, 6, 12)


def _alphas(level: int) -> Tuple[float, float]:
    a = (1 - level / 100) / 2
    return a, 1 - a


def _qcols(levels: Iterable[int]):
    return [c for L in levels for c in (f"q_lo{L}", f"q_hi{L}")]


def _enforce(q: pd.DataFrame, levels: Sequence[int], by_h: bool = True) -> pd.DataFrame:
    q = q.copy()
    for L in levels:
        q[f"q_lo{L}"] = np.minimum(q[f"q_lo{L}"], 0.0)
        q[f"q_hi{L}"] = np.maximum(q[f"q_hi{L}"], 0.0)
    wide, narrow = max(levels), min(levels)
    q[f"q_lo{wide}"] = np.minimum(q[f"q_lo{wide}"], q[f"q_lo{narrow}"])
    q[f"q_hi{wide}"] = np.maximum(q[f"q_hi{wide}"], q[f"q_hi{narrow}"])
    if by_h:
        q = q.sort_values(["node_id", "h"])
        for L in levels:
            q[f"q_lo{L}"] = q.groupby("node_id")[f"q_lo{L}"].cummin()
            q[f"q_hi{L}"] = q.groupby("node_id")[f"q_hi{L}"].cummax()
    return q


def _quantiles(r: np.ndarray, levels: Sequence[int]) -> Dict[str, float]:
    out = {}
    for L in levels:
        lo, hi = _alphas(L)
        out[f"q_lo{L}"], out[f"q_hi{L}"] = float(np.quantile(r, lo)), float(np.quantile(r, hi))
    return out


def residual_quantiles(comb: pd.DataFrame, nodes: pd.DataFrame, levels: Sequence[int],
                       min_obs: int, horizon: int) -> pd.DataFrame:
    """comb: node_id, h, Pred, Actual (combined backtest)."""
    meta = nodes.set_index("node_id")[["Metric", "Level"]]
    d = comb.join(meta, on="node_id")
    d["r"] = np.log1p(d["Actual"].clip(lower=0)) - np.log1p(d["Pred"].clip(lower=0))
    pooled = {}
    for (metric, level), g in d.groupby(["Metric", "Level"]):
        for h in range(1, horizon + 1):
            r = g.loc[g["h"].between(h - 1, h + 1), "r"].to_numpy()
            if len(r):
                pooled[(metric, level, h)] = (_quantiles(r, levels), len(r))
    rows = []
    for nid, g in d.groupby("node_id"):
        metric, level = meta.loc[nid, "Metric"], meta.loc[nid, "Level"]
        for h in range(1, horizon + 1):
            r = g.loc[g["h"].between(h - 1, h + 1), "r"].to_numpy()
            if len(r) >= min_obs:
                rows.append({"node_id": nid, "h": h, "Source": "node", "N": len(r), **_quantiles(r, levels)})
            else:
                q, n = pooled_quantiles(pooled, metric, level, h)
                rows.append({"node_id": nid, "h": h, "Source": "pooled", "N": n, **q})
    return _enforce(pd.DataFrame(rows), levels), pooled


def pooled_quantiles(pooled: dict, metric: str, level: str, h: int):
    """Same metric/level at horizon h, else the Region pool, else the nearest shorter horizon."""
    for hh in range(h, 0, -1):
        for key in ((metric, level, hh), (metric, "Region", hh)):
            if key in pooled:
                return pooled[key]
    raise KeyError(f"No residuals to pool for {metric}/{level}/h<={h}")


def calibrate(q: pd.DataFrame, honest: pd.DataFrame, nodes: pd.DataFrame, levels: Sequence[int],
              max_factor: float, enabled: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scale quantiles so honest walk-forward coverage reaches nominal (never narrows)."""
    meta = nodes.set_index("node_id")[["Metric", "Level"]]
    rows = []
    q = q.join(meta, on="node_id")
    if honest.empty:
        return q.drop(columns=["Metric", "Level"]), pd.DataFrame()
    hz = honest.merge(q, on=["node_id", "h"], how="inner")
    hz["r"] = np.log1p(hz["Actual"].clip(lower=0)) - np.log1p(hz["Policy"].clip(lower=0))
    grid = np.round(np.arange(1.0, max_factor + 1e-9, 0.05), 2)
    for (metric, level), g in hz.groupby(["Metric", "Level"]):
        for L in levels:
            lo, hi, r = g[f"q_lo{L}"].to_numpy(), g[f"q_hi{L}"].to_numpy(), g["r"].to_numpy()
            cov = lambda c: float(np.mean((r >= c * lo) & (r <= c * hi)))
            before = cov(1.0)
            c = 1.0
            if enabled:
                c = next((x for x in grid if cov(x) >= L / 100), grid[-1])
            rows.append({"Metric": metric, "Level": level, "Interval": f"{L}%", "Factor": float(c),
                         "Coverage_Before_Pct": 100 * before, "Coverage_After_Pct": 100 * cov(c),
                         "Target_Pct": L, "N": len(r)})
    calib = pd.DataFrame(rows)
    for _, row in calib.iterrows():
        L = int(row["Interval"].rstrip("%"))
        m = (q["Metric"] == row["Metric"]) & (q["Level"] == row["Level"])
        q.loc[m, [f"q_lo{L}", f"q_hi{L}"]] *= row["Factor"]
    return q.drop(columns=["Metric", "Level"]), calib


def calibration_factor(calib: pd.DataFrame, metric: str, level: str, L: int) -> float:
    if calib.empty:
        return 1.0
    m = calib[(calib["Metric"] == metric) & (calib["Level"] == level) & (calib["Interval"] == f"{L}%")]
    if m.empty:
        m = calib[(calib["Metric"] == metric) & (calib["Level"] == "Region") & (calib["Interval"] == f"{L}%")]
    return float(m["Factor"].iloc[0]) if len(m) else 1.0


def block_quantiles(comb: pd.DataFrame, nodes: pd.DataFrame, levels: Sequence[int], min_obs: int,
                    calib: pd.DataFrame) -> pd.DataFrame:
    """
    Ranges for the next-k-month total (k = 3, 6, 12) by location-scale pooling.

    One series has at most ~13 heavily overlapping 12-month windows — too few to
    estimate its own range — and series differ greatly in volatility, so neither
    "own errors only" nor "pool raw errors" is sound. Instead:
        scale_i,k = std of series i's monthly log errors at horizons <= k   (many obs)
        z         = block log error / scale_i,k       pooled over all series of the metric
        range_i   = quantiles(z) x scale_i,k
    Series without backtest residuals (emerging) use the median scale of their level.
    If too few blocks exist to pool, z falls back to N(0,1) x scale (errors fully
    correlated across months — the conservative case).
    """
    from statistics import NormalDist

    meta = nodes.set_index("node_id")[["Metric", "Level"]]
    m = comb.copy()
    m["r"] = np.log1p(m["Actual"].clip(lower=0)) - np.log1p(m["Pred"].clip(lower=0))
    scale = {}
    for k in BLOCKS:
        s = m[m["h"] <= k].groupby("node_id")["r"].agg(["std", "size"])
        for nid, row in s.iterrows():
            if row["size"] >= min_obs and np.isfinite(row["std"]) and row["std"] > 0:
                scale[(nid, k)] = float(row["std"])
    zs = []
    for k in BLOCKS:
        b = m[m["h"] <= k]
        full = b.groupby(["node_id", "o"]).filter(lambda g: g["h"].nunique() == k)
        s = full.groupby(["node_id", "o"])[["Pred", "Actual"]].sum().reset_index()
        s["rb"] = np.log1p(s["Actual"]) - np.log1p(s["Pred"])
        s["k"] = k
        s["sc"] = [scale.get((n, k), np.nan) for n in s["node_id"]]
        zs.append(s.dropna(subset=["sc"]))
    z = pd.concat(zs, ignore_index=True).join(meta, on="node_id")
    z["z"] = z["rb"] / z["sc"]
    zq = {key: (_quantiles(g["z"].to_numpy(), levels), len(g))
          for key, g in z.groupby(["Metric", "k"]) if len(g) >= min_obs}
    med_scale = {}
    for (nid, k), v in scale.items():
        med_scale.setdefault((meta.loc[nid, "Metric"], meta.loc[nid, "Level"], k), []).append(v)
    med_scale = {key: float(np.median(v)) for key, v in med_scale.items()}

    rows = []
    for nid in nodes["node_id"]:
        metric, level = meta.loc[nid, "Metric"], meta.loc[nid, "Level"]
        for k in BLOCKS:
            sc = scale.get((nid, k))
            src = "own scale"
            if sc is None:
                sc = med_scale.get((metric, level, k)) or med_scale.get((metric, "Region", k))
                src = "median scale of level"
            if sc is None:
                continue
            if (metric, k) in zq:
                zz, n = zq[(metric, k)]
                src += f" x pooled z ({metric})"
            else:
                zz = {}
                for L in levels:
                    zval = NormalDist().inv_cdf(1 - (1 - L / 100) / 2)
                    zz[f"q_lo{L}"], zz[f"q_hi{L}"] = -zval, zval
                n, src = 0, src + " x normal (too few blocks)"
            q = {c: v * sc for c, v in zz.items()}
            for L in levels:
                f = calibration_factor(calib, metric, level, L)
                q[f"q_lo{L}"] = min(q[f"q_lo{L}"], 0.0) * f
                q[f"q_hi{L}"] = max(q[f"q_hi{L}"], 0.0) * f
            rows.append({"node_id": nid, "k": k, "Source": src, "N": n, **q})
    return pd.DataFrame(rows)


def apply_interval(point: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lp = np.log1p(np.clip(point, 0, None))
    return np.expm1(lp + q_lo), np.expm1(lp + q_hi)
