# -*- coding: utf-8 -*-
"""
Empirical, conformally calibrated prediction intervals.

Monthly intervals
1. Base quantiles: relative (log1p) errors of the combined forecast from the rolling-origin
   backtest, r = log1p(actual) - log1p(forecast), per node and horizon (window h±1; pooled by
   metric/level if fewer than ``min_obs``), forced to contain the point and widen with h.
2. Conformal calibration (CQR, Romano et al. 2019): base quantiles are re-fitted using only
   errors realised BEFORE the walk-forward window; on the walk-forward errors (never used to
   fit) each tail gets an additive offset so it is missed at most (1-L)/2 of the time:
       delta_lo = Q_{1-a}(q_lo - r),   delta_hi = Q_{1-a}(r - q_hi),   a = (1-L)/2
   per metric x level x horizon band (fallback metric x level). Offsets are added to the
   full-sample quantiles. Additive offsets handle bias flips (a tail that had collapsed to
   the point can still widen) — multiplicative scaling cannot.

Totals over the next 3/6/12 months (monthly quantiles are not additive)
   Location-scale pooling: z = block log error / series' monthly error scale, pooled over
   series of the metric; range = (Q(z) + conformal offset in z units) x scale.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BLOCKS = (3, 6, 12)
H_BANDS = [(1, 3, "1-3"), (4, 6, "4-6"), (7, 12, "7-12")]
MIN_GROUP = 30          # walk-forward errors needed to calibrate a group separately


def _alphas(level: int) -> Tuple[float, float]:
    a = (1 - level / 100) / 2
    return a, 1 - a


def _qcols(levels: Iterable[int]):
    return [c for L in levels for c in (f"q_lo{L}", f"q_hi{L}")]


def _band(h) -> str:
    for lo, hi, name in H_BANDS:
        if lo <= h <= hi:
            return name
    return H_BANDS[-1][2]


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


def _conformal_offsets(q_lo: np.ndarray, q_hi: np.ndarray, r: np.ndarray, level: int,
                       cap: float) -> Tuple[float, float]:
    """Additive offsets so each tail's miss rate on (r) is <= (1-L)/2 (finite-sample adjusted)."""
    a = (1 - level / 100) / 2
    n = len(r)
    p = min(1.0, np.ceil((n + 1) * (1 - a)) / n)
    d_lo = float(np.quantile(q_lo - r, p))
    d_hi = float(np.quantile(r - q_hi, p))
    return float(np.clip(d_lo, -cap, cap)), float(np.clip(d_hi, -cap, cap))


def residual_quantiles(comb: pd.DataFrame, nodes: pd.DataFrame, levels: Sequence[int],
                       min_obs: int, horizon: int):
    """comb: node_id, h, Pred, Actual (combined backtest). Returns (quantiles, pooled dict)."""
    meta = nodes.set_index("node_id")[["Metric", "Level"]]
    d = comb.join(meta, on="node_id", how="inner")
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
              max_offset: float, enabled: bool = True,
              q_fit: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Conformal calibration (see module docstring). Coverage is measured with `q_fit`
    (quantiles fitted only on errors realised before the walk-forward window); the
    resulting offsets are added to `q`. If `q_fit` is None, `q` is used (in-sample).
    """
    meta = nodes.set_index("node_id")[["Metric", "Level"]]
    q = q.join(meta, on="node_id")
    if honest.empty:
        return q.drop(columns=["Metric", "Level"]), pd.DataFrame()
    fit = (q_fit if q_fit is not None else q)[["node_id", "h"] + _qcols(levels)]
    hz = honest.merge(fit, on=["node_id", "h"], how="inner").join(meta, on="node_id")
    hz["r"] = np.log1p(hz["Actual"].clip(lower=0)) - np.log1p(hz["Policy"].clip(lower=0))
    hz["Band"] = hz["h"].map(_band)
    q["Band"] = q["h"].map(_band)
    rows = []
    for (metric, level), g_ml in hz.groupby(["Metric", "Level"]):
        for _, _, band in H_BANDS:
            g = g_ml[g_ml["Band"] == band]
            scope = band
            if len(g) < MIN_GROUP:
                g, scope = g_ml, "all"
            if g.empty:
                continue
            for L in levels:
                lo, hi, r = g[f"q_lo{L}"].to_numpy(), g[f"q_hi{L}"].to_numpy(), g["r"].to_numpy()
                d_lo, d_hi = _conformal_offsets(lo, hi, r, L, max_offset) if enabled else (0.0, 0.0)
                cov = lambda a, b: float(np.mean((r >= np.minimum(lo - a, 0)) & (r <= np.maximum(hi + b, 0))))
                rows.append({"Metric": metric, "Level": level, "Months_Ahead": band, "Calibrated_On": scope,
                             "Interval": f"{L}%", "Offset_Lo": d_lo, "Offset_Hi": d_hi,
                             "Coverage_Before_Pct": 100 * cov(0.0, 0.0),
                             "Coverage_After_Pct": 100 * cov(d_lo, d_hi), "Target_Pct": L, "N": len(r)})
    calib = pd.DataFrame(rows)
    for _, row in calib.iterrows():
        L = int(row["Interval"].rstrip("%"))
        m = (q["Metric"] == row["Metric"]) & (q["Level"] == row["Level"]) & (q["Band"] == row["Months_Ahead"])
        q.loc[m, f"q_lo{L}"] = q.loc[m, f"q_lo{L}"] - row["Offset_Lo"]
        q.loc[m, f"q_hi{L}"] = q.loc[m, f"q_hi{L}"] + row["Offset_Hi"]
    q = _enforce(q.drop(columns=["Metric", "Level", "Band"]), levels)
    return q, calib


def calibration_offsets(calib: pd.DataFrame, metric: str, level: str, h: int, L: int) -> Tuple[float, float]:
    """(offset_lo, offset_hi) for a metric/level/horizon; Region offsets if the level is absent."""
    if calib.empty:
        return 0.0, 0.0
    band = _band(h)
    for lvl in (level, "Region"):
        m = calib[(calib["Metric"] == metric) & (calib["Level"] == lvl) & (calib["Months_Ahead"] == band)
                  & (calib["Interval"] == f"{L}%")]
        if len(m):
            return float(m["Offset_Lo"].iloc[0]), float(m["Offset_Hi"].iloc[0])
    return 0.0, 0.0


def _block_errors(frame: pd.DataFrame, pred_col: str, k: int) -> pd.DataFrame:
    """Log error of the k-month total for every (node, origin) that has horizons 1..k."""
    b = frame[frame["h"] <= k]
    full = b.groupby(["node_id", "o"]).filter(lambda g: set(g["h"]) == set(range(1, k + 1)))
    s = full.groupby(["node_id", "o"])[[pred_col, "Actual"]].sum().reset_index()
    s["rb"] = np.log1p(s["Actual"]) - np.log1p(s[pred_col])
    s["k"] = k
    return s


def block_quantiles(comb: pd.DataFrame, nodes: pd.DataFrame, levels: Sequence[int], min_obs: int,
                    honest: Optional[pd.DataFrame] = None, max_offset: float = 1.5,
                    comb_fit: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ranges for the next-k-month total (k = 3, 6, 12) by location-scale pooling:
        scale_i,k = std of series i's monthly log errors at horizons <= k
        z         = block log error / scale_i,k, pooled over all series of the metric
        range_i   = (Q(z) + conformal offset) x scale_i,k
    The conformal offset (in z units) comes from walk-forward block errors, measured against
    z-quantiles fitted on `comb_fit` (errors realised before the walk-forward window).
    Series without residuals (emerging) use the median scale of their level.
    """
    from statistics import NormalDist

    meta = nodes.set_index("node_id")[["Metric", "Level"]]

    def scales(frame):
        m = frame.copy()
        m["r"] = np.log1p(m["Actual"].clip(lower=0)) - np.log1p(m["Pred"].clip(lower=0))
        out = {}
        for k in BLOCKS:
            s = m[m["h"] <= k].groupby("node_id")["r"].agg(["std", "size"])
            for nid, row in s.iterrows():
                if row["size"] >= min_obs and np.isfinite(row["std"]) and row["std"] > 0:
                    out[(nid, k)] = float(row["std"])
        return out

    def zquant(frame, sc):
        zs = []
        for k in BLOCKS:
            s = _block_errors(frame, "Pred", k)
            s["sc"] = [sc.get((n, k), np.nan) for n in s["node_id"]]
            zs.append(s.dropna(subset=["sc"]))
        z = pd.concat(zs, ignore_index=True).join(meta, on="node_id")
        z["z"] = z["rb"] / z["sc"]
        return {key: (_quantiles(g["z"].to_numpy(), levels), len(g))
                for key, g in z.groupby(["Metric", "k"]) if len(g) >= min_obs}

    scale = scales(comb)
    zq = zquant(comb, scale)

    # conformal offsets in z units from walk-forward block errors
    offsets, calib_rows = {}, []
    if honest is not None and not honest.empty:
        zq_fit = zquant(comb_fit, scales(comb_fit)) if comb_fit is not None and len(comb_fit) else zq
        hz = honest.rename(columns={"Policy": "Pred"})
        for k in BLOCKS:
            s = _block_errors(hz, "Pred", k).join(meta, on="node_id")
            s["sc"] = [scale.get((n, k), np.nan) for n in s["node_id"]]
            s = s.dropna(subset=["sc"])
            for metric, g in s.groupby("Metric"):
                if (metric, k) not in zq_fit or len(g) < 10:
                    continue
                zz = g["rb"].to_numpy() / g["sc"].to_numpy()
                base = zq_fit[(metric, k)][0]
                for L in levels:
                    lo = np.full(len(zz), base[f"q_lo{L}"])
                    hi = np.full(len(zz), base[f"q_hi{L}"])
                    d_lo, d_hi = _conformal_offsets(lo, hi, zz, L, max_offset * 4)
                    offsets[(metric, k, L)] = (d_lo, d_hi)
                    cov = lambda a, b: float(np.mean((zz >= lo - a) & (zz <= hi + b)))
                    calib_rows.append({"Metric": metric, "Window": f"Next {k} months", "Interval": f"{L}%",
                                       "Offset_Lo_z": d_lo, "Offset_Hi_z": d_hi,
                                       "Coverage_Before_Pct": 100 * cov(0, 0),
                                       "Coverage_After_Pct": 100 * cov(d_lo, d_hi), "N": len(zz)})

    med_scale: Dict[tuple, list] = {}
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
                zz, n = {}, 0
                for L in levels:
                    zval = NormalDist().inv_cdf(1 - (1 - L / 100) / 2)
                    zz[f"q_lo{L}"], zz[f"q_hi{L}"] = -zval, zval
                src += " x normal (too few blocks)"
            q = {}
            for L in levels:
                d_lo, d_hi = offsets.get((metric, k, L), (0.0, 0.0))
                q[f"q_lo{L}"] = min((zz[f"q_lo{L}"] - d_lo) * sc, 0.0)
                q[f"q_hi{L}"] = max((zz[f"q_hi{L}"] + d_hi) * sc, 0.0)
            wide, narrow = max(levels), min(levels)
            q[f"q_lo{wide}"] = min(q[f"q_lo{wide}"], q[f"q_lo{narrow}"])
            q[f"q_hi{wide}"] = max(q[f"q_hi{wide}"], q[f"q_hi{narrow}"])
            rows.append({"node_id": nid, "k": k, "Source": src + (" + conformal" if offsets else ""),
                         "N": n, **q})
    return pd.DataFrame(rows), pd.DataFrame(calib_rows)


def apply_interval(point: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lp = np.log1p(np.clip(point, 0, None))
    return np.expm1(lp + q_lo), np.expm1(lp + q_hi)
