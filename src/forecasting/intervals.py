# -*- coding: utf-8 -*-
"""
RETIRED FOR PRODUCTION RANGES (v2.2): the engine builds ranges with ``ranges.py``. From this module
it uses only the shared helpers ``BLOCKS``, ``_qcols``, ``_enforce``, ``_wq``, ``RECENCY_HALF_LIFE`` and
``apply_interval``. The conformal functions below are kept for reference and their tests; do not
re-wire them into the engine without re-running the range method test (docs/METHODOLOGY.md §7).

Empirical, conformally calibrated prediction intervals (v2.1 method).

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
   the point can still widen) — multiplicative scaling cannot. Offsets never narrow a tail
   (clipped at 0): the walk-forward window is one regime, and regimes flip; any resulting
   over-coverage is reported in the calibration table.

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
    lv = sorted(levels)
    for narrow, wide in zip(lv[:-1], lv[1:]):          # nest: 50% inside 80% inside 95%
        q[f"q_lo{wide}"] = np.minimum(q[f"q_lo{wide}"], q[f"q_lo{narrow}"])
        q[f"q_hi{wide}"] = np.maximum(q[f"q_hi{wide}"], q[f"q_hi{narrow}"])
    if by_h:
        q = q.sort_values(["node_id", "h"])
        for L in levels:
            q[f"q_lo{L}"] = q.groupby("node_id")[f"q_lo{L}"].cummin()
            q[f"q_hi{L}"] = q.groupby("node_id")[f"q_hi{L}"].cummax()
    return q


RECENCY_HALF_LIFE = 6.0   # forecast origins; errors from 6 origins ago count half (errors are not stationary)


def _wq(r: np.ndarray, w: np.ndarray, p: float) -> float:
    """Weighted empirical quantile."""
    o = np.argsort(r)
    r, w = r[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(p, c, r))


def _quantiles(r: np.ndarray, levels: Sequence[int], w: Optional[np.ndarray] = None) -> Dict[str, float]:
    out = {}
    for L in levels:
        lo, hi = _alphas(L)
        if w is None:
            out[f"q_lo{L}"], out[f"q_hi{L}"] = float(np.quantile(r, lo)), float(np.quantile(r, hi))
        else:
            out[f"q_lo{L}"], out[f"q_hi{L}"] = _wq(r, w, lo), _wq(r, w, hi)
    return out


def _recency(o: np.ndarray) -> np.ndarray:
    return 0.5 ** ((o.max() - o) / RECENCY_HALF_LIFE)


def _conformal_offsets(q_lo: np.ndarray, q_hi: np.ndarray, r: np.ndarray, level: int,
                       cap: float) -> Tuple[float, float]:
    """Additive offsets so each tail's miss rate on (r) is <= (1-L)/2 (finite-sample adjusted)."""
    a = (1 - level / 100) / 2
    n = len(r)
    p = min(1.0, np.ceil((n + 1) * (1 - a)) / n)
    d_lo = float(np.quantile(q_lo - r, p))
    d_hi = float(np.quantile(r - q_hi, p))
    # widen only: a tail that happened not to be missed during the last 12 months (one regime)
    # is not narrowed below its full-history width — regimes flip (2025 dip -> 2026 surge)
    return float(np.clip(d_lo, 0.0, cap)), float(np.clip(d_hi, 0.0, cap))


NARROW_FLOOR = 0.6        # a range is never narrowed below 60% of its fitted width
NARROW_MIN_N = 30         # walk-forward errors needed before any narrowing
NARROW_TRIGGER_PP = 5.0   # narrow only if coverage exceeds the target by more than 5 points


def _narrowing_factor(lo: np.ndarray, hi: np.ndarray, r: np.ndarray, level: int) -> float:
    """
    Common scale c in [NARROW_FLOOR, 1] for both tails when an interval clearly over-covers in the
    walk-forward test (wider than the evidence supports): the smallest c that still reaches the
    target. Scaling both tails together keeps the fitted asymmetry, so no tail collapses onto the point.
    """
    cov = lambda c: float(np.mean((r >= c * lo) & (r <= c * hi)))
    if len(r) < NARROW_MIN_N or 100 * cov(1.0) <= level + NARROW_TRIGGER_PP:
        return 1.0
    for c in np.round(np.arange(NARROW_FLOOR, 1.0 + 1e-9, 0.02), 2):
        if cov(c) >= level / 100:
            return float(c)
    return 1.0


def residual_quantiles(comb: pd.DataFrame, nodes: pd.DataFrame, levels: Sequence[int],
                       min_obs: int, horizon: int):
    """comb: node_id, h, Pred, Actual (combined backtest). Returns (quantiles, pooled dict)."""
    meta = nodes.set_index("node_id")[["Metric", "Level"]]
    d = comb.merge(meta.reset_index(), on="node_id", how="inner")
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
            sub = g[g["h"].between(h - 1, h + 1)]
            r = sub["r"].to_numpy()
            if len(r) >= min_obs:
                w = _recency(sub["o"].to_numpy()) if "o" in sub.columns else None
                rows.append({"node_id": nid, "h": h, "Source": "node (recency-weighted)", "N": len(r),
                             **_quantiles(r, levels, w)})
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
                c = _narrowing_factor(lo, hi, r, L) if enabled else 1.0
                d_lo, d_hi = _conformal_offsets(c * lo, c * hi, r, L, max_offset) if enabled else (0.0, 0.0)
                cov = lambda cc, a, b: float(np.mean((r >= np.minimum(cc * lo - a, 0))
                                                     & (r <= np.maximum(cc * hi + b, 0))))
                rows.append({"Metric": metric, "Level": level, "Months_Ahead": band, "Calibrated_On": scope,
                             "Interval": f"{L}%", "Narrowing_Factor": c, "Offset_Lo": d_lo, "Offset_Hi": d_hi,
                             "Coverage_Before_Pct": 100 * cov(1.0, 0.0, 0.0),
                             "Coverage_After_Pct": 100 * cov(c, d_lo, d_hi), "Target_Pct": L, "N": len(r)})
    calib = pd.DataFrame(rows)
    for _, row in calib.iterrows():
        L = int(row["Interval"].rstrip("%"))
        m = (q["Metric"] == row["Metric"]) & (q["Level"] == row["Level"]) & (q["Band"] == row["Months_Ahead"])
        q.loc[m, f"q_lo{L}"] = row["Narrowing_Factor"] * q.loc[m, f"q_lo{L}"] - row["Offset_Lo"]
        q.loc[m, f"q_hi{L}"] = row["Narrowing_Factor"] * q.loc[m, f"q_hi{L}"] + row["Offset_Hi"]
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
    Ranges for the next-k-month total (k = 3, 6, 12), from evidence about totals themselves:
      1. Own errors: recency-weighted empirical quantiles of the series' OWN k-month-total log errors
         (published forecasts, rolling-origin backtest) when it has >= min_obs windows. Recent errors
         weigh more (half-life RECENCY_HALF_LIFE origins): a market's 2024 collapse should not set the
         width of its 2027 range forever, but it still counts.
      2. Otherwise: pooled quantiles of comparable series (same metric and level; regions as fallback).
      3. Calibration on the walk-forward totals, per metric x group (aggregates = organisation +
         countries; regions): bounded narrowing if clearly too wide, then widen-only offsets so each
         tail is missed no more than its target — measured with quantiles fitted before the window.
    Returns (quantiles per node x k, calibration table).
    """
    meta = nodes.set_index("node_id")[["Metric", "Level"]]
    group_of = {"Organisation": "Aggregate", "Country": "Aggregate", "Region": "Region"}

    def fit(frame):
        own, pooled = {}, {}
        if frame is None or frame.empty:
            return own, pooled
        for k in BLOCKS:
            s = _block_errors(frame, "Pred", k).merge(meta.reset_index(), on="node_id", how="inner")
            for nid, g in s.groupby("node_id"):
                if len(g) >= min_obs:
                    own[(nid, k)] = (_quantiles(g["rb"].to_numpy(), levels, _recency(g["o"].to_numpy())), len(g))
            for (m, lvl), g in s.groupby(["Metric", "Level"]):
                if len(g) >= min_obs:
                    pooled[(m, lvl, k)] = (_quantiles(g["rb"].to_numpy(), levels), len(g))
            for m, g in s.groupby("Metric"):
                if len(g) >= 3:
                    pooled[(m, "*", k)] = (_quantiles(g["rb"].to_numpy(), levels), len(g))
        return own, pooled

    def base_q(own, pooled, nid, k):
        m, lvl = meta.loc[nid, "Metric"], meta.loc[nid, "Level"]
        if (nid, k) in own:
            return own[(nid, k)][0], own[(nid, k)][1], "own totals (recency-weighted)"
        for key, src in (((m, lvl, k), f"pooled {lvl}"), ((m, "Region", k), "pooled Region"),
                         ((m, "*", k), "pooled all series")):
            if key in pooled:
                return pooled[key][0], pooled[key][1], src
        for kk in sorted((b for b in BLOCKS if b < k), reverse=True):     # last resort: shorter window
            if (nid, kk) in own:
                return own[(nid, kk)][0], own[(nid, kk)][1], f"own {kk}-month totals (fallback)"
        return None, 0, ""

    own, pooled = fit(comb)
    own_f, pooled_f = fit(comb_fit) if comb_fit is not None and len(comb_fit) else (own, pooled)

    # sequential calibration on walk-forward totals (the same recency-weighted procedure, re-fitted
    # at every test origin with only the errors realised by then)
    adj, calib_rows = {}, pd.DataFrame()
    if honest is not None and not honest.empty:
        errs, evals = [], []
        hz = honest.rename(columns={"Policy": "Pred"})
        for k in BLOCKS:
            bb = _block_errors(comb, "Pred", k)
            errs.append(pd.DataFrame({"node_id": bb["node_id"], "key": k, "o": bb["o"], "t": bb["o"] + k, "r": bb["rb"]}))
            vv = _block_errors(hz, "Pred", k)
            evals.append(pd.DataFrame({"node_id": vv["node_id"], "key": k, "e": vv["o"], "r": vv["rb"]}))
        adj, calib_rows = sequential_calibration(pd.concat(errs, ignore_index=True), pd.concat(evals, ignore_index=True),
                                                 nodes, levels, min_obs, max_offset, window=0,
                                                 band_of=lambda k: f"{k}m", group_of=lambda l: group_of[l])
        if len(calib_rows):
            calib_rows = calib_rows.rename(columns={"Band": "Window"})
            calib_rows["Window"] = calib_rows["Window"].str.replace("m", "", regex=False).map(lambda x: f"Next {x} months")

    rows = []
    lv = sorted(levels)
    for nid in nodes["node_id"]:
        m, lvl = meta.loc[nid, "Metric"], meta.loc[nid, "Level"]
        for k in BLOCKS:
            q0, n, src = base_q(own, pooled, nid, k)
            if q0 is None:
                continue
            q = {}
            for L in levels:
                c, d_lo, d_hi = adj.get((m, group_of[lvl], f"{k}m", L), (1.0, 0.0, 0.0))
                q[f"q_lo{L}"] = min(c * q0[f"q_lo{L}"] - d_lo, 0.0)
                q[f"q_hi{L}"] = max(c * q0[f"q_hi{L}"] + d_hi, 0.0)
            for narrow, wide in zip(lv[:-1], lv[1:]):
                q[f"q_lo{wide}"] = min(q[f"q_lo{wide}"], q[f"q_lo{narrow}"])
                q[f"q_hi{wide}"] = max(q[f"q_hi{wide}"], q[f"q_hi{narrow}"])
            rows.append({"node_id": nid, "k": k, "Source": src + (" + calibrated" if adj else ""), "N": n, **q})
    return pd.DataFrame(rows), calib_rows


def apply_interval(point: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.clip(point, 0, None)
    lp = np.log1p(p)
    # min/max guard against round-off in expm1(log1p(x)) when an offset is exactly 0
    return np.minimum(np.expm1(lp + q_lo), p), np.maximum(np.expm1(lp + q_hi), p)


# ───────────────── sequential calibration (tests the procedure that is actually used) ─────────────────
def _seq_fit(err: dict, pooled_err: dict, nid: str, pool_key: tuple, key: int, at: int, levels, min_obs: int,
             window: int):
    """Recency-weighted quantiles from errors realised by `at` (t <= at) for key±window; pooled fallback."""
    for arr in (err.get(nid), pooled_err.get(pool_key)):
        if arr is None:
            continue
        k_, t_, o_, r_ = arr
        m = (np.abs(k_ - key) <= window) & (t_ <= at)
        if m.sum() >= min_obs:
            o = o_[m]
            w = 0.5 ** ((at - o) / RECENCY_HALF_LIFE)
            return _quantiles(r_[m], levels, w)
    return None


def sequential_calibration(err: pd.DataFrame, eval_: pd.DataFrame, nodes: pd.DataFrame, levels: Sequence[int],
                           min_obs: int, max_offset: float, window: int, band_of=None,
                           group_of=None) -> Tuple[dict, pd.DataFrame]:
    """
    err:   backtest errors of the PUBLISHED forecasts: node_id, key (h or k), o, t, r (log error)
    eval_: walk-forward errors: node_id, key, e (origin), r
    For every walk-forward error, the range procedure (recency-weighted own quantiles, pooled fallback)
    is re-fitted with only errors realised by e; its misses give a bounded narrowing factor and
    widen-only offsets per metric x group x band. Returns ({(metric, group, band, L): (c, d_lo, d_hi)}, table).
    """
    band_of = band_of or (lambda k: "all")
    group_of = group_of or (lambda lvl: lvl)
    meta = nodes.set_index("node_id")[["Metric", "Level"]]
    e2 = err.merge(meta.reset_index(), on="node_id", how="inner")
    by_node = {n: (g["key"].to_numpy(), g["t"].to_numpy(), g["o"].to_numpy(), g["r"].to_numpy())
               for n, g in e2.groupby("node_id")}
    by_pool = {(m, l): (g["key"].to_numpy(), g["t"].to_numpy(), g["o"].to_numpy(), g["r"].to_numpy())
               for (m, l), g in e2.groupby(["Metric", "Level"])}
    ev = eval_.merge(meta.reset_index(), on="node_id", how="inner")
    recs = []
    for row in ev.itertuples(index=False):
        q = _seq_fit(by_node, by_pool, row.node_id, (row.Metric, row.Level), row.key, row.e, levels, min_obs, window)
        if q is None:
            continue
        recs.append({"Metric": row.Metric, "Group": group_of(row.Level), "Band": band_of(row.key), "r": row.r, **q})
    adj, rows = {}, []
    if not recs:
        return adj, pd.DataFrame()
    d = pd.DataFrame(recs)
    for (m, grp, band), g in d.groupby(["Metric", "Group", "Band"]):
        if len(g) < 10:
            continue
        r = g["r"].to_numpy()
        for L in levels:
            lo, hi = g[f"q_lo{L}"].to_numpy(), g[f"q_hi{L}"].to_numpy()
            c = _narrowing_factor(lo, hi, r, L)
            d_lo, d_hi = _conformal_offsets(c * lo, c * hi, r, L, max_offset)
            adj[(m, grp, band, L)] = (c, d_lo, d_hi)
            cov = lambda cc, a, b: float(np.mean((r >= np.minimum(cc * lo - a, 0)) & (r <= np.maximum(cc * hi + b, 0))))
            rows.append({"Metric": m, "Group": grp, "Band": band, "Interval": f"{L}%", "Narrowing_Factor": c,
                         "Offset_Lo": d_lo, "Offset_Hi": d_hi, "Coverage_Before_Pct": 100 * cov(1.0, 0, 0),
                         "Coverage_After_Pct": 100 * cov(c, d_lo, d_hi), "Target_Pct": L, "N": len(r)})
    return adj, pd.DataFrame(rows)



def group_of_level(level: str) -> str:
    return "Aggregate" if level in ("Organisation", "Country") else "Region"


def monthly_calibration(q: pd.DataFrame, comb_pub: pd.DataFrame, honest: pd.DataFrame, nodes: pd.DataFrame,
                        levels: Sequence[int], min_obs: int, max_offset: float, enabled: bool = True):
    """Sequentially calibrated monthly ranges: returns (q adjusted, adjustments dict, table)."""
    if not enabled or honest is None or honest.empty:
        return q, {}, pd.DataFrame()
    lr = lambda a, p: np.log1p(a.clip(lower=0)) - np.log1p(p.clip(lower=0))
    err = pd.DataFrame({"node_id": comb_pub["node_id"], "key": comb_pub["h"], "o": comb_pub["o"], "t": comb_pub["t"],
                        "r": lr(comb_pub["Actual"], comb_pub["Pred"])})
    ev = pd.DataFrame({"node_id": honest["node_id"], "key": honest["h"], "e": honest["o"],
                       "r": lr(honest["Actual"], honest["Policy"])})
    adj, tab = sequential_calibration(err, ev, nodes, levels, min_obs, max_offset, window=1, band_of=_band,
                                      group_of=group_of_level)
    q = apply_adjustments(q, adj, nodes, levels)
    if len(tab):
        tab = tab.rename(columns={"Band": "Months_Ahead"})
    return q, adj, tab


def adjustment_for(adj: dict, metric: str, level: str, h: int, L: int):
    return adj.get((metric, group_of_level(level), _band(h), L)) or adj.get((metric, "Region", _band(h), L))         or (1.0, 0.0, 0.0)


def apply_adjustments(q: pd.DataFrame, adj: dict, nodes: pd.DataFrame, levels: Sequence[int]) -> pd.DataFrame:
    if not adj:
        return q
    meta = nodes.set_index("node_id")[["Metric", "Level"]]
    q = q.copy()
    for L in levels:
        f = [adjustment_for(adj, meta.loc[n, "Metric"], meta.loc[n, "Level"], h, L) for n, h in zip(q["node_id"], q["h"])]
        c = np.array([x[0] for x in f]); dl = np.array([x[1] for x in f]); dh = np.array([x[2] for x in f])
        q[f"q_lo{L}"] = c * q[f"q_lo{L}"].to_numpy() - dl
        q[f"q_hi{L}"] = c * q[f"q_hi{L}"].to_numpy() + dh
    return _enforce(q, levels)
