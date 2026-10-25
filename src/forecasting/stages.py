# -*- coding: utf-8 -*-
"""
Pipeline stages after submission (v2.3): Offer, Deposit, Visa document, Visa applied, Visa granted.

Each stage is forecast from the stage before it, the same way submissions are forecast from receipts
(funnel.py), as a chain:

    Offer <- Submitted <- (published submission forecast)
    Deposit <- Offer,  Visa_Document <- Deposit,  Visa_Applied <- Visa_Document,  Visa_Granted <- Visa_Applied

    stage[t] = sum_l w_l * upstream[t - l],  l = 0..3

The weights are non-negative least squares on the last 12 realised months, re-fitted at every origin.
The upstream series is actual up to the origin and the upstream FORECAST after it (the published
submission forecast, then each stage's own published forecast), so nothing unknown at the time is used.

For every stage, three variants are replayed at each walk-forward origin and scored on next-quarter totals
(organisation and countries):
  snaive  same month last year (the baseline);
  chain   the conversion chain above;
  blend   their average.
The chosen variant is the best one, but chain or blend must beat snaive by at least MIN_GAIN_PCT; otherwise
snaive is used. Ranges are the 10/90 and 25/75 quantiles of the chosen variant's own walk-forward errors
(log scale, per stage and horizon band).

Stage counts come from fetch_data.py (first time an application reaches the stage in the 26th-25th cycle)
and transform_data.py (stage_final_summary.xlsx).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("rmg.stages")

CHAIN = [("Offer", "Submitted"), ("Deposit", "Offer"), ("Visa_Document", "Deposit"),
         ("Visa_Applied", "Visa_Document"), ("Visa_Granted", "Visa_Applied")]
LAGS = (0, 1, 2, 3)
WINDOW = 12
MIN_GAIN_PCT = 3.0
MIN_HISTORY = 24          # months of stage history needed before a series is forecast
VARIANTS = ("snaive", "chain", "blend")
BANDS = ((1, 3), (4, 6), (7, 12))
ORG = "_ORG"


def load_stage_actuals(path: Path, months: pd.DatetimeIndex) -> Dict[Tuple[str, str], np.ndarray]:
    """(Country, StageGroup) -> monthly counts on the panel's month index; organisation = sum of countries."""
    d = pd.read_excel(path, dtype={"Month": str})
    d["Date"] = pd.to_datetime(d["Month"].str[:7] + "-01", errors="coerce")
    d["Apps"] = pd.to_numeric(d["Apps"], errors="coerce").fillna(0.0)
    d = d.dropna(subset=["Date"])
    pos = {m: i for i, m in enumerate(months)}
    d = d[d["Date"].isin(pos)]
    out: Dict[Tuple[str, str], np.ndarray] = {}
    for (c, s), g in d.groupby(["Country Grouped", "StageGroup"]):
        a = np.zeros(len(months))
        gg = g.groupby("Date")["Apps"].sum()
        a[[pos[x] for x in gg.index]] = gg.to_numpy()
        out[(str(c), str(s))] = a
    for s in {k[1] for k in out}:
        out[(ORG, s)] = np.sum([v for (c, ss), v in out.items() if ss == s and c != ORG], axis=0)
    return out


def _nnls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    active = list(range(X.shape[1]))
    while active:
        w, *_ = np.linalg.lstsq(X[:, active], y, rcond=None)
        if (w >= 0).all():
            out = np.zeros(X.shape[1])
            out[active] = w
            return out
        active.pop(int(np.argmin(w)))
    return np.zeros(X.shape[1])


def _weights(up: np.ndarray, st: np.ndarray, o: int) -> Optional[np.ndarray]:
    tt = np.arange(o - WINDOW + 1, o + 1)
    if tt[0] - max(LAGS) < 0:
        return None
    X = np.column_stack([up[tt - L] for L in LAGS])
    if X.sum() <= 0:
        return None
    return _nnls(X, st[tt])


def _forecasts(up_act: np.ndarray, up_fc: Dict[int, float], st: np.ndarray, o: int, H: int) -> Dict[str, Dict[int, float]]:
    """Variants for horizons 1..H made at origin o (up_fc: upstream forecast by horizon at the same origin)."""
    w = _weights(up_act, st, o)
    out = {v: {} for v in VARIANTS}
    for h in range(1, H + 1):
        t = o + h
        sn = float(st[t - 12]) if t - 12 >= 0 else np.nan
        ch = np.nan
        if w is not None:
            val = 0.0
            for wi, L in zip(w, LAGS):
                u = t - L
                x = up_act[u] if u <= o else up_fc.get(u - o, np.nan)
                val += wi * x
            ch = float(val)
        out["snaive"][h] = sn
        out["chain"][h] = ch if np.isfinite(ch) else sn
        out["blend"][h] = 0.5 * (sn + out["chain"][h]) if np.isfinite(sn) else out["chain"][h]
    return out


def _submitted_sources(nodes: pd.DataFrame, Y: Dict[str, np.ndarray]) -> Dict[str, str]:
    """Country -> node_id of its published submissions (country total, or its single region)."""
    sub = nodes[nodes["Metric"] == "Submitted"]
    out = {ORG: f"Submitted|{ORG}|_TOTAL"}
    for c, g in sub.groupby("Country"):
        if c == ORG:
            continue
        tot = f"Submitted|{c}|_TOTAL"
        if tot in Y:
            out[c] = tot
        else:
            reg = g[g["Level"] == "Region"]["node_id"].tolist()
            if len(reg) == 1:
                out[c] = reg[0]
    return {c: n for c, n in out.items() if n in Y}


def run(stage_actual: Dict[Tuple[str, str], np.ndarray], Y: Dict[str, np.ndarray], nodes: pd.DataFrame,
        honest: pd.DataFrame, fc: pd.DataFrame, months_ext: pd.DatetimeIndex, T: int, H: int = 12):
    """Returns (stage_forecast, stage_test, stage_conversion)."""
    src = _submitted_sources(nodes, Y)
    origins = sorted(int(o) for o in honest["o"].unique()) if honest is not None and len(honest) else []
    hz = honest[honest["node_id"].isin(set(src.values()))] if len(origins) else honest
    sub_hon = ({(n, int(o)): g.set_index("h")["Policy"].to_dict() for (n, o), g in hz.groupby(["node_id", "o"])}
               if len(origins) else {})
    fsub = fc[(fc["Metric"] == "Submitted") & (fc["Region"] == "_TOTAL") & fc["Level"].isin(["Country", "Organisation"])]
    sub_final = {c: g.set_index("Horizon")["Forecast"].to_dict() for c, g in fsub.groupby("Country")}

    # upstream state per country: actual series, forecast by origin, final forecast
    up = {c: {"act": Y[n], "by_o": {o: sub_hon.get((n, o), {}) for o in origins}, "final": sub_final.get(c, {})}
          for c, n in src.items()}
    rows_fc, rows_test, rows_conv = [], [], []
    for stage, upstream in CHAIN:
        nxt, recs = {}, []
        for c, u in up.items():
            st = stage_actual.get((c, stage))
            if st is None or (st[:T] > 0).sum() < MIN_HISTORY:
                continue
            by_o = {}
            for o in origins:
                if o < WINDOW + max(LAGS) or not u["by_o"].get(o):
                    continue
                by_o[o] = _forecasts(u["act"], u["by_o"][o], st, o, H)
                if o + 3 <= T - 1:
                    a = float(st[o + 1:o + 4].sum())
                    if a > 0:
                        for v in VARIANTS:
                            f = sum(by_o[o][v].get(h, np.nan) for h in (1, 2, 3))
                            if np.isfinite(f):
                                recs.append({"Country": c, "o": o, "Variant": v, "APE": 100 * abs(f - a) / a})
            final = _forecasts(u["act"], u["final"], st, T - 1, H) if u["final"] else None
            nxt[c] = {"act": st, "by_o_all": by_o, "final_all": final}
        if not nxt:
            LOG.info("stages: no usable history for %s - chain stops here", stage)
            break
        r = pd.DataFrame(recs, columns=["Country", "o", "Variant", "APE"])
        r["Level"] = np.where(r["Country"] == ORG, "Organisation", "Country")
        score = r.groupby(["Variant", "Level"])["APE"].median().groupby("Variant").mean() if len(r) else pd.Series(dtype=float)
        best = str(score.idxmin()) if len(score) else "snaive"
        if best != "snaive" and "snaive" in score and 100 * (score["snaive"] - score[best]) / score["snaive"] < MIN_GAIN_PCT:
            best = "snaive"
        for v in VARIANTS:
            for lvl in ("Organisation", "Country"):
                x = r[(r["Variant"] == v) & (r["Level"] == lvl)]
                rows_test.append({"StageGroup": stage, "From": upstream, "Variant": v, "Level": lvl,
                                  "Quarter_Error_Pct": float(x["APE"].median()) if len(x) else np.nan,
                                  "N": int(len(x)), "Chosen": v == best})
        # chosen forecasts become the upstream of the next stage; collect errors for the ranges
        errs, new_up = [], {}
        for c, d in nxt.items():
            st = d["act"]
            by_o = {o: f[best] for o, f in d["by_o_all"].items()}
            for o, f in by_o.items():
                for h, v in f.items():
                    t = o + h
                    if t <= T - 1 and np.isfinite(v):
                        errs.append((h, np.log1p(max(st[t], 0)) - np.log1p(max(v, 0))))
            fin = d["final_all"][best] if d["final_all"] else {}
            new_up[c] = {"act": st, "by_o": by_o, "final": fin}
            w = _weights(up[c]["act"], st, T - 1)
            if w is not None and w.sum() > 0:
                rows_conv.append({"Country": "Organisation" if c == ORG else c, "StageGroup": stage, "From": upstream,
                                  "Conversion_Pct": round(100 * float(w.sum()), 1),
                                  **{f"Share_Lag{L}": round(100 * float(wi / w.sum()), 1) for L, wi in zip(LAGS, w)}})
        e = pd.DataFrame(errs, columns=["h", "r"])
        q = {}
        for lo, hi in BANDS:
            x = e[e["h"].between(lo, hi)]["r"].to_numpy()
            q[(lo, hi)] = np.quantile(x, [0.1, 0.25, 0.75, 0.9]) if len(x) >= 20 else np.array([np.nan] * 4)
        for c, d in new_up.items():
            st = d["act"]
            for h, f in d["final"].items():
                if not np.isfinite(f):
                    continue
                band = next(b for b in BANDS if b[0] <= h <= b[1])
                lp = np.log1p(max(f, 0))
                rng = {k: (float(np.expm1(lp + v)) if np.isfinite(v) else np.nan)
                       for k, v in zip(("Lo80", "Lo50", "Hi50", "Hi80"), q[band])}
                for k in ("Lo80", "Lo50"):
                    rng[k] = min(max(rng[k], 0.0), f) if np.isfinite(rng[k]) else np.nan
                for k in ("Hi50", "Hi80"):
                    rng[k] = max(rng[k], f) if np.isfinite(rng[k]) else np.nan
                t = T - 1 + h
                rows_fc.append({"Level": "Organisation" if c == ORG else "Country", "Country": c, "StageGroup": stage,
                                "Date": months_ext[t], "Horizon": h, "Forecast": f, **rng,
                                "Same_Month_Last_Year": float(st[t - 12]) if 0 <= t - 12 < len(st) else np.nan,
                                "Method": best})
        up = new_up
    return pd.DataFrame(rows_fc), pd.DataFrame(rows_test), pd.DataFrame(rows_conv)


def stage_totals(sf: pd.DataFrame) -> pd.DataFrame:
    """Next quarter and next 12 months per destination x stage, with last year's same months."""
    if sf is None or sf.empty:
        return pd.DataFrame()
    out = []
    for (lvl, c, s), g in sf.groupby(["Level", "Country", "StageGroup"], sort=False):
        g = g.sort_values("Horizon")
        for k in (3, 12):
            x = g[g["Horizon"] <= k]
            f, ly = float(x["Forecast"].sum()), float(x["Same_Month_Last_Year"].sum())
            out.append({"Level": lvl, "Country": c, "StageGroup": s, "Window": f"Next {k} months", "Forecast": f,
                        "Same_Months_Last_Year": ly, "YoY_Pct": 100 * (f / ly - 1) if ly > 0 else np.nan,
                        "Method": g["Method"].iloc[0]})
    return pd.DataFrame(out)
