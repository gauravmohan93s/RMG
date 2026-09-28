# -*- coding: utf-8 -*-
"""
Statistical overlay: recent level-shift (bias) correction — the data-driven replacement for
guessed policy multipliers.

1. Recent bias of series i at origin e: median of r = log1p(actual) - log1p(forecast) over
   the last `window` realised target months (horizons 1..max_h, targets <= e only).
2. Empirical-Bayes (James-Stein) shrinkage towards 0:
       w_i = tau^2 / (tau^2 + se_i^2),  tau^2 = max(var(b) - mean(se^2), 0)
   Noisy series barely move; consistent biases across many observations count.
3. Correction  = exp(w_i * b_i)          mode "level" (persistent shift)
                 exp(w_i * b_i * phi^(h-1)) mode "decay" (fades with horizon, phi = 0.85)
   capped to [0.67, 1.5].
4. Both modes are replayed walk-forward (at each evaluation origin only past errors are
   used). The best mode is adopted only if it improves walk-forward WAPE by at least
   `min_gain_pct` overall; otherwise no correction is applied. Evidence is always reported.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

CAP = (np.log(0.67), np.log(1.5))
PHI = 0.85


def estimate(comb: pd.DataFrame, at: int, window: int = 6, max_h: int = 3) -> pd.DataFrame:
    """Shrunk recent bias per node using only targets <= `at`."""
    d = comb[(comb["t"] <= at) & (comb["h"] <= max_h)].copy()
    if d.empty:
        return pd.DataFrame(columns=["node_id", "Bias_Log", "N", "SE2", "Shrink_Weight", "Correction_Log"])
    d["r"] = np.log1p(d["Actual"].clip(lower=0)) - np.log1p(d["Pred"].clip(lower=0))
    last_t = d.groupby("node_id")["t"].transform(lambda s: np.sort(s.unique())[-window:].min())
    d = d[d["t"] >= last_t]
    g = d.groupby("node_id")["r"]
    est = pd.DataFrame({"Bias_Log": g.median(), "N": g.size(), "Var": g.var()}).reset_index()
    est["SE2"] = (np.pi / 2) * est["Var"].fillna(est["Var"].median()) / est["N"].clip(lower=1)   # var of a median
    # between-series variance of true biases; with <= 2 series it cannot be estimated -> no correction
    tau2 = max(float(est["Bias_Log"].var()) - float(est["SE2"].mean()), 0.0) if len(est) > 2 else 0.0
    est["Shrink_Weight"] = tau2 / (tau2 + est["SE2"]) if tau2 > 0 else 0.0
    est["Correction_Log"] = np.clip(est["Shrink_Weight"] * est["Bias_Log"], *CAP)
    return est.drop(columns=["Var"])


def factor(corr_log: np.ndarray, h: np.ndarray, mode: str) -> np.ndarray:
    if mode == "level":
        return np.exp(corr_log)
    if mode == "decay":
        return np.exp(corr_log * PHI ** (h - 1))
    return np.ones_like(corr_log, dtype=float)


def walk_forward_test(comb: pd.DataFrame, honest: pd.DataFrame, window: int = 6,
                      max_h: int = 3) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """Replay each mode at every walk-forward origin with corrections estimated from past errors only."""
    if honest.empty:
        return pd.DataFrame(), {}
    out = {"none": honest["Policy"].to_numpy(), "level": honest["Policy"].to_numpy().copy(),
           "decay": honest["Policy"].to_numpy().copy()}
    for e, idx in honest.groupby("o").groups.items():
        est = estimate(comb, e, window, max_h).set_index("node_id")["Correction_Log"]
        sub = honest.loc[idx]
        c = sub["node_id"].map(est).fillna(0.0).to_numpy()
        for mode in ("level", "decay"):
            pos = honest.index.get_indexer(idx)
            out[mode][pos] = sub["Policy"].to_numpy() * factor(c, sub["h"].to_numpy(), mode)
    a = honest["Actual"].to_numpy()
    rows = []
    for mode, p in out.items():
        ae = np.abs(p - a)
        wape = 100 * ae.sum() / a.sum() if a.sum() else np.nan
        per_node = pd.Series(ae).groupby(honest["node_id"].to_numpy()).mean()
        rows.append({"Mode": mode, "WAPE_Pct": wape, "Bias_Pct": 100 * (p - a).sum() / a.sum(),
                     "_node_mae": per_node})
    base_nodes = rows[0]["_node_mae"]
    for r in rows:
        r["Pct_Series_Improved"] = float(100 * (r.pop("_node_mae") < base_nodes - 1e-9).mean())
    return pd.DataFrame(rows), {k: pd.Series(v, index=honest.index) for k, v in out.items()}


def decide(test: pd.DataFrame, min_gain_pct: float = 1.0) -> str:
    """Mode to adopt ('none' unless a mode lowers walk-forward WAPE by >= min_gain_pct relative)."""
    if test.empty:
        return "none"
    base = float(test.loc[test["Mode"] == "none", "WAPE_Pct"].iloc[0])
    best = test[test["Mode"] != "none"].sort_values("WAPE_Pct").iloc[0]
    return best["Mode"] if (base - best["WAPE_Pct"]) / base * 100 >= min_gain_pct else "none"


# ───────────────── retransformation (mean vs median) correction ─────────────────
# Models on the log scale forecast the typical (median) month. For skewed counts the median is
# below the mean, and SUMS of regional medians under-state totals. Duan's smearing estimator
# rescales each series by mean(exp(residual)) so forecasts become expected values, which add up.
# It helps totals but can hurt single series (their absolute error is minimised by the median),
# so it is adopted only if it improves decision-level (quarter-total) accuracy across levels.
SMEAR_CAP = (0.8, 1.25)
SMEAR_MIN_OBS = 8


def smearing_factors(comb: pd.DataFrame, at: int) -> pd.Series:
    d = comb[comb["t"] <= at]
    r = np.log1p(d["Actual"].clip(lower=0)) - np.log1p(d["Pred"].clip(lower=0))
    g = pd.DataFrame({"node_id": d["node_id"], "e": np.exp(r)}).groupby("node_id")["e"]
    f = g.mean().where(g.size() >= SMEAR_MIN_OBS, 1.0)
    return f.clip(*SMEAR_CAP)


def _quarter_errors(h: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Median abs % error of next-3-month totals at region level and for bottom-up country/org sums."""
    reg = nodes[nodes["Level"] == "Region"][["node_id", "Country", "Metric"]]
    x = h.merge(reg, on="node_id")
    out = []
    for lvl, key in (("Region", "node_id"), ("Country", "Country"), ("Organisation", None)):
        y = x.assign(key=x[key] if key else "org").groupby(["Metric", "key", "o", "h"])[["Policy", "Actual"]].sum()
        y = y.reset_index()
        q = y[y["h"] <= 3]
        q = q[q.groupby(["Metric", "key", "o"])["h"].transform("nunique") == 3]
        b = q.groupby(["Metric", "key", "o"])[["Policy", "Actual"]].sum()
        b["APE"] = 100 * (b["Policy"] - b["Actual"]).abs() / b["Actual"].replace(0, np.nan)
        for m, v in b.groupby(level="Metric")["APE"].median().items():
            out.append({"Level": lvl, "Metric": m, "Quarter_Error_Pct": float(v)})
    return pd.DataFrame(out)


def overlay_quarter_test(honest: pd.DataFrame, variants: Dict[str, pd.Series], nodes: pd.DataFrame) -> pd.DataFrame:
    """Each overlay mode judged on next-quarter totals per level (bottom-up), like every other component."""
    rows = [_quarter_errors(honest, nodes).assign(Mode="none")]
    for mode, pol in (variants or {}).items():
        if mode != "none":
            rows.append(_quarter_errors(honest.assign(Policy=pol), nodes).assign(Mode=mode))
    return pd.concat(rows, ignore_index=True)


def decide_quarter(qtest: pd.DataFrame, min_gain_pct: float = 3.0) -> str:
    """'none' unless a mode lowers the level-averaged next-quarter error by >= min_gain_pct (relative)."""
    if qtest is None or qtest.empty:
        return "none"
    avg = qtest.groupby("Mode")["Quarter_Error_Pct"].mean()
    if "none" not in avg or len(avg) < 2:
        return "none"
    best = avg.drop("none").idxmin()
    return str(best) if (avg["none"] - avg[best]) / avg["none"] * 100 >= min_gain_pct else "none"


def smearing_test(comb: pd.DataFrame, honest: pd.DataFrame, nodes: pd.DataFrame):
    """Walk-forward: factors at each origin e use only targets <= e. Returns (table, smeared honest Policy)."""
    if honest is None or honest.empty:
        return pd.DataFrame(), None
    sm = honest["Policy"].to_numpy().copy()
    for e, idx in honest.groupby("o").groups.items():
        f = smearing_factors(comb, e)
        pos = honest.index.get_indexer(idx)
        sm[pos] = honest.loc[idx, "Policy"].to_numpy() * honest.loc[idx, "node_id"].map(f).fillna(1.0).to_numpy()
    base = _quarter_errors(honest, nodes).assign(Mode="median (typical month)")
    alt = _quarter_errors(honest.assign(Policy=sm), nodes).assign(Mode="mean (smearing-corrected)")
    return pd.concat([base, alt], ignore_index=True), pd.Series(sm, index=honest.index)


def decide_smearing(test: pd.DataFrame, min_gain_pct: float = 3.0) -> bool:
    if test is None or test.empty:
        return False
    avg = test.groupby("Mode")["Quarter_Error_Pct"].mean()
    b, a = avg.get("median (typical month)", np.nan), avg.get("mean (smearing-corrected)", np.nan)
    return bool(np.isfinite(a) and np.isfinite(b) and b > 0 and (b - a) / b * 100 >= min_gain_pct)
