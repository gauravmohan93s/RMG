# -*- coding: utf-8 -*-
"""Headline numbers and deterministic narrative shared by the Excel and PDF reports."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..data.prepare import METRICS


def fmt_int(v) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:,.0f}"


def fmt_pct(v, signed=True) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    return f"{v:+.1f}%" if signed else f"{v:.1f}%"


def kpis(res) -> Dict[str, Dict[str, dict]]:
    """{metric: {"Next 3 months": row-dict, ...}} at organisation level."""
    t = res.totals[res.totals["Level"] == "Organisation"]
    return {m: {r["Window"]: r.to_dict() for _, r in t[t["Metric"] == m].iterrows()} for m in METRICS}


def conversion(res) -> Dict[str, float]:
    k = kpis(res)
    r12, s12 = k["Received"]["Next 12 months"], k["Submitted"]["Next 12 months"]
    return {"forecast": 100 * s12["Forecast"] / r12["Forecast"] if r12["Forecast"] else np.nan,
            "last_year": 100 * s12["Same_Months_Last_Year"] / r12["Same_Months_Last_Year"]
            if r12["Same_Months_Last_Year"] else np.nan}


def fiscal_year_outlook(res) -> pd.DataFrame:
    p = res.periods
    return p[(p["Level"] == "Organisation") & (p["Period_Type"] == "Fiscal year")]


def country_table(res) -> pd.DataFrame:
    """Organisation + countries: next 12 months, next quarter, range, change, reliability (receipts & submissions)."""
    t = res.totals[res.totals["Level"].isin(["Country", "Organisation"])]
    rows = []
    for (lvl, cty), g in t.groupby(["Level", "Country"], sort=False):
        row = {"Level": lvl, "Country": g["Label"].iloc[0]}
        for m in METRICS:
            r12 = g[(g["Metric"] == m) & (g["Window"] == "Next 12 months")]
            r3 = g[(g["Metric"] == m) & (g["Window"] == "Next 3 months")]
            if r12.empty:
                continue
            r12 = r12.iloc[0]
            row.update({f"{m}_Forecast": r12["Forecast"], f"{m}_Lo80": r12.get("Lo80", np.nan),
                        f"{m}_Hi80": r12.get("Hi80", np.nan), f"{m}_Last_Year": r12["Same_Months_Last_Year"],
                        f"{m}_YoY_Pct": r12["YoY_Pct"], f"{m}_Adjusted": r12["Forecast_Adjusted"],
                        f"{m}_Reliability": r12.get("Reliability", ""),
                        f"{m}_Quarter_Error_Pct": r12.get("Typical_Quarter_Error_Pct", np.nan)})
            if len(r3):
                row.update({f"{m}_Next_Quarter": r3.iloc[0]["Forecast"], f"{m}_NQ_Lo80": r3.iloc[0].get("Lo80", np.nan),
                            f"{m}_NQ_Hi80": r3.iloc[0].get("Hi80", np.nan)})
        rows.append(row)
    df = pd.DataFrame(rows)
    df["_o"] = (df["Level"] != "Organisation").astype(int)
    return df.sort_values(["_o", "Received_Forecast"], ascending=[True, False]).drop(columns="_o")


def reliability_summary(res, metric: str = "Received") -> pd.DataFrame:
    """Share of next-12-month volume (regions) in each reliability band."""
    fc = res.forecast
    r = fc[(fc["Level"] == "Region") & (fc["Metric"] == metric)]
    if r.empty or "Reliability" not in r.columns:
        return pd.DataFrame()
    v = r.groupby(["node_id", "Reliability"])["Forecast"].sum().reset_index()
    t = v.groupby("Reliability").agg(Series=("node_id", "nunique"), Volume=("Forecast", "sum"))
    t["Volume_Share_Pct"] = 100 * t["Volume"] / t["Volume"].sum()
    order = {"High": 0, "Medium": 1, "Low": 2, "New": 3, "Review": 4}
    return t.reset_index().sort_values("Reliability", key=lambda x: x.map(order))


def busy_months(res, metric: str = "Received"):
    si = getattr(res, "seasonal_index", None)
    if si is None or si.empty:
        return [], []
    o = si[(si["Level"] == "Organisation") & (si["Metric"] == metric)].sort_values("Seasonal_Index", ascending=False)
    return o[o["Class"] == "High"]["Month_Name"].tolist(), o[o["Class"] == "Low"]["Month_Name"].tolist()


def _join(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1] if xs else ""


def _round(v, step=100):
    return fmt_int(round(v / step) * step) if v is not None and np.isfinite(v) else "–"


def policy_label(res) -> str:
    from ..forecasting.selection import METHOD_LABELS
    pol = res.manifest.get("config", {}).get("selection", {}).get("policy", "mean_all")
    return METHOD_LABELS[{"mean_all": "Mean_All", "mean_screened": "Mean_Screened"}.get(pol, "Top_K")]


def accuracy_table(res, metric: str) -> pd.DataFrame:
    """Walk-forward WAPE / bias of the PUBLISHED procedure (combined, reconciled) vs the benchmark,
    by level x horizon bucket."""
    acc = res.accuracy
    if acc is None or acc.empty:
        return pd.DataFrame()
    a = acc[(acc["Metric"] == metric) & acc["Level"].isin(["Organisation", "Country", "Region"])].copy()
    a["Months_Ahead"] = pd.cut(a["h"], [0, 3, 6, 12], labels=["1-3", "4-6", "7-12"]).astype(str)
    def agg(method):
        x = a[a["Method"] == method]
        return x.groupby(["Level", "Months_Ahead"]).apply(
            lambda g: np.average(g["WAPE_Pct"], weights=g["N"]) if g["N"].sum() else np.nan, include_groups=False)
    out = pd.DataFrame({"WAPE_Pct": agg("Selected method (walk-forward)"),
                        "Benchmark_WAPE_Pct": agg("Seasonal naive (benchmark)")})
    n = a[a["Method"] == "Selected method (walk-forward)"].groupby(["Level", "Months_Ahead"])["N"].sum()
    out["N"] = n
    fva = getattr(res, "forecast_vs_actual", None)
    if fva is not None and len(fva):
        f = fva[(fva["Metric"] == metric) & fva["Source"].astype(str).str.startswith("Walk-forward")].copy()
        f["Months_Ahead"] = pd.cut(f["Months_Ahead"], [0, 3, 6, 12], labels=["1-3", "4-6", "7-12"]).astype(str)
        g = f.groupby(["Level", "Months_Ahead"])
        out["Bias_Pct"] = 100 * g["Error"].sum() / g["Actual"].sum()
    else:
        out["Bias_Pct"] = np.nan
    out = out.reset_index()
    order = {"Organisation": 0, "Country": 1, "Region": 2}
    return out.sort_values(["Level", "Months_Ahead"], key=lambda s: s.map(order) if s.name == "Level" else s)


def accuracy_buckets(res, metric: str, level: str = "Organisation") -> Dict[str, float]:
    t = accuracy_table(res, metric)
    t = t[t["Level"] == level] if len(t) else t
    return {r.Months_Ahead: float(r.WAPE_Pct) for r in t.itertuples()} if len(t) else {}


def overall_bias(res, metric: str, level: str = "Organisation") -> float:
    fva = getattr(res, "forecast_vs_actual", None)
    if fva is None or fva.empty:
        return np.nan
    f = fva[(fva["Level"] == level) & (fva["Metric"] == metric) & fva["Source"].astype(str).str.startswith("Walk-forward")]
    return float(100 * f["Error"].sum() / f["Actual"].sum()) if len(f) and f["Actual"].sum() else np.nan


def messages(res) -> List[str]:
    """Plain-language key messages for executives (no statistical jargon)."""
    k = kpis(res)
    out = []
    r12, s12, r3 = k["Received"]["Next 12 months"], k["Submitted"]["Next 12 months"], k["Received"].get("Next 3 months")
    out.append(f"We expect about {_round(r12['Forecast'])} applications to be received over the next 12 months "
               f"({r12['From']:%b %Y}–{r12['To']:%b %Y}), {fmt_pct(r12['YoY_Pct'])} compared with the same months last "
               f"year. Most likely between {_round(r12.get('Lo50'), 1000)} and {_round(r12.get('Hi50'), 1000)} "
               f"(an even chance); plan for {_round(r12.get('Lo80'), 1000)} to {_round(r12.get('Hi80'), 1000)} "
               f"(8 times in 10).")
    if r3 is not None:
        out.append(f"Next quarter ({r3['From']:%b}–{r3['To']:%b %Y}): about {_round(r3['Forecast'])} applications received "
                   f"(most likely {_round(r3.get('Lo50'), 1000)}–{_round(r3.get('Hi50'), 1000)}; plan for "
                   f"{_round(r3.get('Lo80'), 1000)}–{_round(r3.get('Hi80'), 1000)}).")
    c = conversion(res)
    out.append(f"About {_round(s12['Forecast'])} applications are expected to be submitted ({fmt_pct(s12['YoY_Pct'])}) — "
               f"roughly {c['forecast']:.0f} of every 100 received, in line with last year ({c['last_year']:.0f}).")
    ct = country_table(res)
    ct = ct[ct["Level"] == "Country"].copy()
    if len(ct):
        ct["chg"] = ct["Received_Forecast"] - ct["Received_Last_Year"]
        up = ct[ct["chg"] > 0].sort_values("chg", ascending=False).head(3)
        if len(up):
            out.append("Most of the growth comes from " + _join(
                [f"{r.Country} (+{_round(r.chg)})" for r in up.itertuples()]) + ".")
        dn = ct[ct["chg"] < 0].sort_values("chg").head(3)
        if len(dn):
            txt = "Expected to decline: " + _join([f"{r.Country} ({_round(r.chg)})" for r in dn.itertuples()]) + "."
            if len(up):
                out[-1] += " " + txt
            else:
                out.append(txt)
    busy, quiet = busy_months(res)
    if busy:
        out.append(f"Busiest months are typically {_join(busy)}; quietest are {_join(quiet)} — see the staffing calendar.")
    org = res.forecast[(res.forecast["Level"] == "Organisation") & (res.forecast["Metric"] == "Received")]
    err = float(org["Typical_Quarter_Error_Pct"].iloc[0]) if len(org) and "Typical_Quarter_Error_Pct" in org else np.nan
    if np.isfinite(err):
        out.append(f"How far to trust it: tested on the past year, our organisation-level quarterly forecasts were "
                   f"typically within ±{err:.0f}% of what actually happened.")
    b = overall_bias(res, "Received")
    if np.isfinite(b) and b <= -5:
        out.append(f"Actual volumes have recently been running above our forecasts (by about {abs(b):.0f}%), so these "
                   "numbers lean cautious — for peak months, plan capacity towards the upper end of the range.")
    elif np.isfinite(b) and b >= 5:
        out.append(f"Actual volumes have recently come in below our forecasts (by about {b:.0f}%), so treat the upper "
                   "end of the range with caution.")
    rs = reliability_summary(res)
    if len(rs):
        hm = rs[rs["Reliability"].isin(["High", "Medium"])]["Volume_Share_Pct"].sum()
        out.append(f"{hm:.0f}% of forecast volume is in markets we can forecast with high or medium reliability; the rest "
                   "is mostly small or new markets, where swings are proportionally larger.")
    applied = int((res.overlay_rules.get("Applied_Status", pd.Series(dtype=str)) == "Applied").sum())
    out.append("No manual adjustments have been made: all figures come directly from the data." if applied == 0 else
               f"{applied} approved management adjustment(s) are included in the 'Adjusted' figures.")
    return out
