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
    t = res.totals[(res.totals["Window"] == "Next 12 months") & res.totals["Level"].isin(["Country", "Organisation"])]
    rows = []
    for (lvl, cty), g in t.groupby(["Level", "Country"], sort=False):
        row = {"Level": lvl, "Country": g["Label"].iloc[0]}
        for m in METRICS:
            r = g[g["Metric"] == m]
            if r.empty:
                continue
            r = r.iloc[0]
            row.update({f"{m}_Forecast": r["Forecast"], f"{m}_Lo80": r.get("Lo80", np.nan),
                        f"{m}_Hi80": r.get("Hi80", np.nan), f"{m}_Last_Year": r["Same_Months_Last_Year"],
                        f"{m}_YoY_Pct": r["YoY_Pct"], f"{m}_Adjusted": r["Forecast_Adjusted"],
                        f"{m}_Confidence": r["Confidence"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    df["_o"] = (df["Level"] != "Organisation").astype(int)
    return df.sort_values(["_o", "Received_Forecast"], ascending=[True, False]).drop(columns="_o")


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
    k = kpis(res)
    ct = country_table(res)
    ct = ct[ct["Level"] == "Country"].copy()
    out = []
    for m, noun in (("Received", "Applications received"), ("Submitted", "Applications submitted")):
        r = k[m]["Next 12 months"]
        out.append(f"{noun} are forecast at {fmt_int(r['Forecast'])} over the next 12 months "
                   f"({r['From']:%b %Y}–{r['To']:%b %Y}); 80% range {fmt_int(r.get('Lo80'))}–{fmt_int(r.get('Hi80'))}, "
                   f"{fmt_pct(r['YoY_Pct'])} vs the same months last year ({fmt_int(r['Same_Months_Last_Year'])}).")
    c = conversion(res)
    out.append(f"Implied submission-to-receipt ratio: {fmt_pct(c['forecast'], False)} "
               f"(same months last year {fmt_pct(c['last_year'], False)}).")
    if len(ct):
        ct["chg"] = ct["Received_Forecast"] - ct["Received_Last_Year"]
        up = ct.sort_values("chg", ascending=False).head(3)
        up = up[up["chg"] > 0]
        if len(up):
            out.append("Largest growth in receipts: " + "; ".join(
                f"{r.Country} {fmt_int(r.chg)} ({fmt_pct(r.Received_YoY_Pct)})" for r in up.itertuples()) + ".")
        dn = ct[ct["chg"] < 0].sort_values("chg").head(3)
        if len(dn):
            out.append("Forecast declines: " + "; ".join(
                f"{r.Country} {fmt_int(r.chg)} ({fmt_pct(r.Received_YoY_Pct)})" for r in dn.itertuples()) + ".")
    acc = accuracy_buckets(res, "Received")
    if np.isfinite(acc.get("1-3", np.nan)):
        out.append(f"Tested on the past 12 forecast origins, the organisation receipts forecast was off by "
                   f"{acc['1-3']:.0f}% per month on average 1–3 months ahead and {acc.get('7-12', np.nan):.0f}% "
                   "7–12 months ahead — plan on the 80% range, not the single number, beyond 6 months.")
    b = overall_bias(res, "Received")
    if np.isfinite(b) and abs(b) >= 5:
        direction, why, stance = (("above", "growth outpaced history", "conservative") if b < 0 else
                                  ("below", "a slowdown the history did not show", "optimistic"))
        out.append(f"In that test, actual receipts ran on average {abs(b):.0f}% {direction} the forecast "
                   f"({why}), so treat the baseline as {stance}.")
    rt = getattr(res, "reconciliation_test", None)
    chosen = res.manifest.get("reconciliation_method_chosen", "")
    if rt is not None and len(rt) and chosen:
        o = rt[(rt["Level"] == "Organisation") & (rt["Metric"] == "Received")].set_index("Method")["WAPE_Pct"]
        if "none" in o and chosen in o and o["none"] > 0:
            label = {"bottom_up": "summing the regional forecasts", "wls_var": "variance-weighted reconciliation",
                     "mint_shrink": "MinT reconciliation"}.get(chosen, chosen)
            out.append(f"Organisation totals are built by {label}: {100 * (1 - o[chosen] / o['none']):.0f}% more accurate "
                       f"in testing ({o[chosen]:.0f}% vs {o['none']:.0f}% error) than forecasting the total on its own.")
    n_guard = int(res.guardrails["Guardrail_Triggered"].sum()) if len(res.guardrails) else 0
    n_emerg = int(((res.selection["Method"].str.startswith("Emerging")) &
                   (res.selection["node_id"].str.startswith("Received"))).sum())
    if n_guard or n_emerg:
        out.append(f"{n_guard} series were capped by plausibility guardrails and {n_emerg} new series have too "
                   "little history to test — both are marked low confidence (see Guardrails / Data Quality).")
    applied = int((res.overlay_rules.get("Applied_Status", pd.Series(dtype=str)) == "Applied").sum())
    out.append("No business/policy adjustments are approved: all figures are the statistical baseline."
               if applied == 0 else
               f"{applied} approved policy adjustment(s) applied — see 'Adjusted' columns and the Policy Overlay sheet.")
    return out
