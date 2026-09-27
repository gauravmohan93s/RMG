# -*- coding: utf-8 -*-
"""
Signals beyond the model's own history:

1. Forecast vs actual ("vintages") — for every past month: what was forecast 1..12 months
   earlier vs what happened. Built from the walk-forward test (what the method would have
   said at the time) plus live forecasts from the prediction history as they accumulate.
2. Human inputs — expectations entered by managers (Settings.xlsx sheet Human_Inputs).
   Recorded every run, shown next to the model, scored against actuals. A person's input is
   blended in (inverse-error weights) only after >= MIN_SCORED months in which they beat
   the model on the same months; until then it is informational.
3. Measured policy events (sheet Policy_Events) — impact = actual / counterfactual - 1 over
   the months after the event, where the counterfactual is forecast from data before the
   event. A placebo test (same method at earlier, NON-overlapping event-free windows) gives the
   normal error range; the impact is "significant" only if it falls outside that range
   (central 80% with >= 10 placebos, else min-max; fewer than 4 placebos -> not judged).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data.prepare import ORG, TOTAL

HUMAN_COLS = ["Country", "Region", "Metric", "Month", "Expected_Volume", "Submitted_By", "Reason", "Submitted_On"]
EVENT_COLS = ["Country", "Metric", "Event", "Start_Month", "Notes"]
MIN_SCORED = 6
COUNTERFACTUAL_MODELS = ("SNaive", "SNaiveGrowth", "ETS", "Theta")


# ───────────────────────────── sheets ─────────────────────────────
def load_sheet(settings_file: Path, sheet: str, template: Optional[Path], cols: Sequence[str]) -> Tuple[pd.DataFrame, str]:
    df, src = None, ""
    if settings_file is not None and Path(settings_file).exists():
        try:
            xl = pd.ExcelFile(settings_file)
            if sheet in xl.sheet_names:
                df, src = pd.read_excel(settings_file, sheet_name=sheet), f"{Path(settings_file).name}!{sheet}"
        except Exception as exc:
            src = f"unreadable {Path(settings_file).name}: {type(exc).__name__}"
    if df is None:
        df = pd.read_csv(template) if template is not None and Path(template).exists() else pd.DataFrame(columns=cols)
        src = src or (f"template {Path(template).name}" if template is not None else "none")
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[list(cols)].copy(), src


def _norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("Country", "Region", "Metric"):
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str).str.strip()
    out.loc[out["Country"].str.lower().isin(["organisation", "organization", "all", "*"]), "Country"] = ORG
    out["Region"] = out["Region"].replace({"": TOTAL, "*": TOTAL})
    out["Level"] = np.where(out["Country"] == ORG, "Organisation", np.where(out["Region"] == TOTAL, "Country", "Region"))
    return out


# ───────────────────────────── 1. forecast vs actual ─────────────────────────────
def forecast_vs_actual(honest: pd.DataFrame, nodes: pd.DataFrame, months_ext: pd.DatetimeIndex,
                       live: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    parts = []
    if honest is not None and not honest.empty:
        meta = nodes.set_index("node_id")[["Level", "Country", "Region", "Metric"]]
        d = honest.join(meta, on="node_id")
        wf = pd.DataFrame({
            "Level": d["Level"], "Country": d["Country"], "Region": d["Region"], "Metric": d["Metric"],
            "Target_Month": months_ext[d["t"].to_numpy()], "Made_At": months_ext[d["o"].to_numpy()],
            "Months_Ahead": d["h"], "Forecast": d["Policy"], "Actual": d["Actual"],
            "Source": "Walk-forward test (as if made at the time)"})
        parts.append(wf)
    if live is not None and not live.empty:
        lv = pd.DataFrame({
            "Level": live["Level"], "Country": live["Country"], "Region": live["Region"], "Metric": live["Metric"],
            "Target_Month": pd.to_datetime(live["Date"]), "Made_At": pd.to_datetime(live["Run_Date"]),
            "Months_Ahead": live["Horizon"], "Forecast": live["Forecast"], "Actual": live["Actual"],
            "Lo80": live.get("Lo80"), "Hi80": live.get("Hi80"),
            "Source": "Live forecast (run " + pd.to_datetime(live["Run_Date"]).dt.strftime("%Y-%m-%d") + ")"})
        parts.append(lv)
    if not parts:
        return pd.DataFrame()
    fva = pd.concat(parts, ignore_index=True)
    fva["Error"] = fva["Forecast"] - fva["Actual"]
    fva["Error_Pct"] = 100 * fva["Error"] / fva["Actual"].replace(0, np.nan)
    return fva.sort_values(["Level", "Country", "Region", "Metric", "Target_Month", "Months_Ahead"])


def fva_summary(fva: pd.DataFrame, level: str = "Organisation", horizons: Sequence[int] = (1, 3, 6, 12)) -> pd.DataFrame:
    """One row per target month and metric: actual and the forecasts made k months earlier."""
    if fva.empty:
        return pd.DataFrame()
    d = fva[(fva["Level"] == level) & fva["Source"].str.startswith("Walk-forward")]
    rows = []
    for (metric, cty, tgt), g in d.groupby(["Metric", "Country", "Target_Month"]):
        row = {"Metric": metric, "Country": cty, "Target_Month": tgt, "Actual": float(g["Actual"].iloc[0])}
        for k in horizons:
            s = g[g["Months_Ahead"] == k]
            if len(s):
                row[f"Forecast_{k}m_ahead"] = float(s["Forecast"].iloc[0])
                row[f"Error_{k}m_Pct"] = float(s["Error_Pct"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


# ───────────────────────────── 2. human inputs ─────────────────────────────
MAX_INPUT_AGE_DAYS = 120


def clean_human(hi: pd.DataFrame, today: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Valid rows only; inputs older than MAX_INPUT_AGE_DAYS (by Submitted_On) are treated as expired."""
    if hi.empty:
        return hi
    h = _norm_keys(hi.dropna(subset=["Expected_Volume", "Month"], how="any"))
    h["Date"] = pd.to_datetime(h["Month"].astype(str).str[:7] + "-01", errors="coerce")
    h["Expected_Volume"] = pd.to_numeric(h["Expected_Volume"], errors="coerce")
    h = h.dropna(subset=["Date", "Expected_Volume"])
    if today is not None:
        sub = pd.to_datetime(h["Submitted_On"], errors="coerce")
        h = h[sub.isna() | (sub >= pd.Timestamp(today) - pd.Timedelta(days=MAX_INPUT_AGE_DAYS))]
    return h


def record_human(hi: pd.DataFrame, run_date: str, path: Path) -> pd.DataFrame:
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if hi.empty:
        return old
    cur = hi.assign(Run_Date=pd.Timestamp(run_date))
    if not old.empty:
        old = old[pd.to_datetime(old["Run_Date"]) != pd.Timestamp(run_date)]
    out = pd.concat([old, cur], ignore_index=True) if not old.empty else cur
    out.astype({c: str for c in ("Month", "Submitted_On", "Reason", "Submitted_By")}).to_parquet(path, index=False)
    return out


def score_humans(hist: pd.DataFrame, actuals: pd.DataFrame, model_hist: pd.DataFrame) -> pd.DataFrame:
    """Per submitter: months scored, their WAPE vs the model's WAPE on the same months, blend weight."""
    cols = ["Submitted_By", "Months_Scored", "Human_WAPE_Pct", "Model_WAPE_Pct", "Eligible_For_Blend", "Blend_Weight"]
    if hist is None or hist.empty:
        return pd.DataFrame(columns=cols)
    keys = ["Level", "Country", "Region", "Metric", "Date"]
    j = hist.merge(actuals[keys + ["Actual"]], on=keys, how="inner")
    if j.empty:
        return pd.DataFrame(columns=cols)
    if model_hist is not None and not model_hist.empty:
        mh = model_hist.assign(Run_Date=pd.to_datetime(model_hist["Run_Date"]))[keys + ["Run_Date", "Forecast"]]
        j = j.merge(mh, on=keys + ["Run_Date"], how="left")
    else:
        j["Forecast"] = np.nan
    rows = []
    for who, g in j.groupby("Submitted_By"):
        gm = g.dropna(subset=["Forecast"])          # identical rows for both -> same months, same horizons
        a = gm["Actual"].sum()
        hw = 100 * (gm["Expected_Volume"] - gm["Actual"]).abs().sum() / a if a else np.nan
        mw = 100 * (gm["Forecast"] - gm["Actual"]).abs().sum() / a if a else np.nan
        n = int(gm["Date"].nunique())
        elig = bool(n >= MIN_SCORED and np.isfinite(mw) and hw < mw)
        rows.append({"Submitted_By": who, "Months_Scored": n, "Human_WAPE_Pct": hw, "Model_WAPE_Pct": mw,
                     "Eligible_For_Blend": elig, "Blend_Weight": (mw / (mw + hw)) if elig else 0.0})
    return pd.DataFrame(rows, columns=cols)


def attach_human(fc: pd.DataFrame, hi: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Adds Human_Expected, Human_Submitted_By, Human_vs_Model_Pct and Forecast_Human_Blend."""
    fc = fc.copy()
    fc["Human_Expected"], fc["Human_Submitted_By"], fc["Human_Reason"] = np.nan, "", ""
    fc["Forecast_Human_Blend"] = fc["Forecast"]
    if hi.empty:
        fc["Human_vs_Model_Pct"] = np.nan
        return fc
    keys = ["Level", "Country", "Region", "Metric", "Date"]
    latest = hi.sort_values("Submitted_On").drop_duplicates(keys, keep="last")
    m = fc.merge(latest[keys + ["Expected_Volume", "Submitted_By", "Reason"]], on=keys, how="left")
    fc["Human_Expected"] = m["Expected_Volume"].to_numpy()
    fc["Human_Submitted_By"] = m["Submitted_By"].fillna("").to_numpy()
    fc["Human_Reason"] = m["Reason"].fillna("").to_numpy()
    fc["Human_vs_Model_Pct"] = 100 * (fc["Human_Expected"] / fc["Forecast"].replace(0, np.nan) - 1)
    w = fc["Human_Submitted_By"].map(scores.set_index("Submitted_By")["Blend_Weight"] if len(scores) else {}).fillna(0.0)
    has = fc["Human_Expected"].notna() & (w > 0)
    fc.loc[has, "Forecast_Human_Blend"] = (1 - w[has]) * fc.loc[has, "Forecast"] + w[has] * fc.loc[has, "Human_Expected"]
    return fc


# ───────────────────────────── 3. measured policy events ─────────────────────────────
def _counterfactual(y: np.ndarray, h: int, start_month: int) -> Optional[np.ndarray]:
    from ..forecasting.models import LOCAL_MODELS
    fs = []
    for name in COUNTERFACTUAL_MODELS:
        try:
            fs.append(LOCAL_MODELS[name](y, h, start_month))
        except Exception:
            continue
    return np.mean(fs, axis=0) if fs else None


def measure_events(events: pd.DataFrame, history: pd.DataFrame, window: int = 6, n_placebo: int = 12) -> pd.DataFrame:
    cols = ["Country", "Metric", "Event", "Start_Month", "Months_Measured", "Actual", "Counterfactual",
            "Measured_Impact_Pct", "Normal_Error_Range_80_Pct", "Significant", "Notes"]
    if events is None or events.empty:
        return pd.DataFrame(columns=cols)
    ev = _norm_keys(events.dropna(subset=["Country", "Start_Month"]))
    rows = []
    for _, e in ev.iterrows():
        start = pd.to_datetime(str(e["Start_Month"])[:7] + "-01", errors="coerce")
        metrics = ["Received", "Submitted"] if e["Metric"] in ("", "*", "nan") else [e["Metric"]]
        for metric in metrics:
            s = history[(history["Level"] == e["Level"]) & (history["Country"] == e["Country"])
                        & (history["Region"] == e["Region"]) & (history["Metric"] == metric)]
            s = s.sort_values("Date").set_index("Date")["Actual"]
            if pd.isna(start) or s.empty or start not in s.index:
                rows.append({"Country": e["Country"], "Metric": metric, "Event": e["Event"], "Start_Month": start,
                             "Notes": "no data at the event month", "Significant": ""})
                continue
            i0 = s.index.get_loc(start)
            train = s.iloc[:i0].to_numpy()
            post = s.iloc[i0:i0 + window].to_numpy()
            k = len(post)
            if len(train) < 24 or k == 0:
                rows.append({"Country": e["Country"], "Metric": metric, "Event": e["Event"], "Start_Month": start,
                             "Notes": "fewer than 24 months before the event", "Significant": ""})
                continue
            cf = _counterfactual(train, k, start.month)
            impact = 100 * (post.sum() / cf.sum() - 1) if cf is not None and cf.sum() > 0 else np.nan
            plac = []
            for j in range(1, n_placebo + 1):
                cut = i0 - j * k                      # non-overlapping windows of length k before the event
                if cut < 24:
                    break
                tr, act = s.iloc[:cut].to_numpy(), s.iloc[cut:cut + k].to_numpy()
                c = _counterfactual(tr, k, s.index[cut].month)
                if c is not None and c.sum() > 0:
                    plac.append(100 * (act.sum() / c.sum() - 1))
            if len(plac) >= 10:
                lo, hi = np.quantile(plac, [0.1, 0.9])
            elif len(plac) >= 4:
                lo, hi = min(plac), max(plac)
            else:
                lo, hi = np.nan, np.nan
            sig = "" if not np.isfinite(lo) else ("Yes" if (impact < lo or impact > hi) else "No")
            rows.append({"Country": e["Country"], "Metric": metric, "Event": e["Event"], "Start_Month": start,
                         "Months_Measured": k, "Actual": float(post.sum()),
                         "Counterfactual": float(cf.sum()) if cf is not None else np.nan,
                         "Measured_Impact_Pct": impact,
                         "Normal_Error_Range_80_Pct": f"{lo:+.0f}% to {hi:+.0f}%" if np.isfinite(lo) else "",
                         "Significant": sig or "Not judged (history too short for placebo test)",
                         "Notes": str(e.get("Notes", "") or "")})
    return pd.DataFrame(rows, columns=cols)
