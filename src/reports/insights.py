# -*- coding: utf-8 -*-
"""Headline numbers and deterministic narrative shared by the Excel and PDF reports."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.prepare import METRICS, TOTAL


def test_depth(res) -> Dict[str, int]:
    """Backtest origins and walk-forward test origins actually used in this run (from the manifest)."""
    cfg = res.manifest.get("config", {})
    return {"origins": int(cfg.get("backtest", {}).get("n_origins", 24)),
            "wf": int(cfg.get("selection", {}).get("honest_eval_origins", 12))}


def wf_period(res) -> str:
    n = test_depth(res)["wf"]
    return "the past year" if n <= 12 else f"the past {n // 12} years" if n % 12 == 0 else f"the past {n} months"


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
                        f"{m}_Quarter_Error_Pct": r12.get("Typical_Quarter_Error_Pct", np.nan),
                        f"{m}_Change_vs_Previous_Pct": r12.get("Change_vs_Previous_Pct", np.nan)})
            if len(r3):
                row.update({f"{m}_Next_Quarter": r3.iloc[0]["Forecast"], f"{m}_NQ_Lo80": r3.iloc[0].get("Lo80", np.nan),
                            f"{m}_NQ_Hi80": r3.iloc[0].get("Hi80", np.nan)})
        for m in METRICS:
            row[f"{m}_Recent_YoY_Pct"] = recent_yoy(res.history, lvl, cty, m)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["_o"] = (df["Level"] != "Organisation").astype(int)
    return df.sort_values(["_o", "Received_Forecast"], ascending=[True, False]).drop(columns="_o")


def recent_yoy(history: pd.DataFrame, level: str, country: str, metric: str, months: int = 6) -> float:
    """Actual growth of the last `months` months vs the same months a year earlier — what the market is
    doing now, to sense-check the growth the forecast assumes."""
    h = history[(history["Metric"] == metric) & (history["Country"] == country)]
    hh = h[(h["Level"] == level) & (h["Region"] == TOTAL)] if level != "Region" else h
    if hh.empty:                                   # single-region country: sum its region(s)
        hh = h[h["Level"] == "Region"]
    s = hh.groupby("Date")["Actual"].sum().sort_index()
    if len(s) < months + 12:
        return np.nan
    now, before = s.iloc[-months:].sum(), s.iloc[-months - 12:-12].sum()
    return float(100 * (now / before - 1)) if before > 0 else np.nan


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
    """Walk-forward bias in % of actual. Sign: Error = Forecast - Actual, so NEGATIVE = actuals came in
    ABOVE the forecast (under-forecast). messages() relies on this convention."""
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
        out.append(f"How far to trust it: tested on {wf_period(res)}, our organisation-level quarterly forecasts were "
                   f"typically within ±{err:.0f}% of what actually happened.")
    rt = getattr(res, "range_tested", None)
    if rt is not None and len(rt):
        x = rt[(rt["Level"] == "Organisation") & (rt["Window"] == "Next 3 months") & (rt["Interval"] == "80%")]
        if len(x) and x["N"].iloc[0] >= 10:
            out.append(f"Our 'plan for' quarter ranges were tested the same way: the actual organisation total fell inside "
                       f"them {x['Tested_Coverage_Pct'].iloc[0]:.0f}% of the time (aim: 80%).")
    b = overall_bias(res, "Received")
    if np.isfinite(b) and b <= -5:
        out.append(f"In testing over {wf_period(res)}, actual volumes came in above our forecasts by about {abs(b):.0f}% "
                   "on average, so these numbers lean cautious — for peak months, plan capacity towards the upper end "
                   "of the range.")
    elif np.isfinite(b) and b >= 5:
        out.append(f"In testing over {wf_period(res)}, actual volumes came in below our forecasts by about {b:.0f}% on "
                   "average, so treat the upper end of the range with caution.")
    rs = reliability_summary(res)
    if len(rs):
        hm = rs[rs["Reliability"].isin(["High", "Medium"])]["Volume_Share_Pct"].sum()
        out.append(f"{hm:.0f}% of forecast volume is in markets we can forecast with high or medium reliability; the rest "
                   "is mostly small or new markets, where swings are proportionally larger.")
    applied = int((res.overlay_rules.get("Applied_Status", pd.Series(dtype=str)) == "Applied").sum())
    out.append("No manual adjustments have been made: all figures come directly from the data." if applied == 0 else
               f"{applied} approved management adjustment(s) are included in the 'Adjusted' figures.")
    return out


# ───────────────────────── health check (acceptance criteria fixed in advance) ─────────────────────────
# The same checks every month. "Tested" = forecasts re-made for past months with only the data then
# available; "Live" = the forecasts we actually published, scored as their actuals arrive. The pipeline
# is trusted for a use once its live column passes for 3 consecutive months (shadow period).
HEALTH = {
    "quarter_miss": 10.0,      # organisation next-quarter total: typical miss at most ±10%
    "coverage": (70.0, 90.0),  # planning (80%) range should hold 70-90% of the time
    "bias": 5.0,               # average error at most ±5% of actual
    "live_min_months": 3,
}


def _status(ok: Optional[bool], near: bool = False) -> str:
    return "Not yet" if ok is None else "Pass" if ok else ("Watch" if near else "Fail")


def health_check(res) -> pd.DataFrame:
    rows = []
    org = res.totals[(res.totals["Level"] == "Organisation") & (res.totals["Metric"] == "Received")
                     & (res.totals["Window"] == "Next 3 months")]
    tq = float(org["Typical_Quarter_Error_Pct"].iloc[0]) if len(org) else np.nan
    fva = getattr(res, "forecast_vs_actual", None)
    live = fva[~fva["Source"].astype(str).str.startswith("Walk-forward")] if fva is not None and len(fva) else pd.DataFrame()
    lq, ln = np.nan, 0
    if len(live):
        q = live[(live["Level"] == "Organisation") & (live["Metric"] == "Received") & live["Months_Ahead"].between(1, 3)]
        g = q.groupby("Made_At").filter(lambda x: len(x) == 3).groupby("Made_At")[["Forecast", "Actual"]].sum()
        if len(g):
            lq, ln = float((100 * (g["Forecast"] - g["Actual"]).abs() / g["Actual"]).median()), len(g)
    lim = HEALTH["quarter_miss"]
    rows.append({"Check": "Organisation next-quarter total is close", "Target": f"typical miss up to ±{lim:.0f}%",
                 "Tested": f"±{tq:.1f}%" if np.isfinite(tq) else "–",
                 "Tested_Status": _status(tq <= lim if np.isfinite(tq) else None, np.isfinite(tq) and tq <= lim * 1.2),
                 "Live": f"±{lq:.1f}% ({ln} quarters)" if ln else "–",
                 "Live_Status": _status(lq <= lim if ln >= HEALTH["live_min_months"] else None)})

    lo, hi = HEALTH["coverage"]
    rt = getattr(res, "range_tested", None)
    tc = np.nan
    if rt is not None and len(rt):
        x = rt[rt["Level"].isin(["Organisation", "Country"]) & (rt["Window"] == "Next 3 months") & (rt["Interval"] == "80%")]
        if len(x) and x["N"].sum():
            tc = float(np.average(x["Tested_Coverage_Pct"], weights=x["N"]))
    sc = getattr(res, "track_scorecard", None)
    lc, lb, lnm = np.nan, np.nan, 0
    if sc is not None and len(sc) and "Coverage80_Pct" in sc:
        s = sc[(sc["Level"] == "Organisation") & (sc["Metric"] == "Received") & (sc["Horizon_Bucket"].astype(str) == "1-3")]
        if len(s):
            lnm = int(s["Forecasts_Evaluated"].sum())
            lc = float(np.average(s["Coverage80_Pct"], weights=s["Forecasts_Evaluated"]))
            lb = float(np.average(s["Bias_Pct"], weights=s["Forecasts_Evaluated"]))
    rows.append({"Check": "Planning range holds as promised", "Target": f"actual inside {lo:.0f}–{hi:.0f}% of the time",
                 "Tested": f"{tc:.0f}%" if np.isfinite(tc) else "–",
                 "Tested_Status": _status(lo <= tc <= hi if np.isfinite(tc) else None, np.isfinite(tc) and lo - 5 <= tc <= hi + 5),
                 "Live": f"{lc:.0f}% ({lnm} months)" if lnm else "–",
                 "Live_Status": _status(lo <= lc <= hi if lnm >= HEALTH["live_min_months"] else None)})

    b = overall_bias(res, "Received")
    bl = HEALTH["bias"]
    rows.append({"Check": "No persistent over- or under-forecast", "Target": f"average error within ±{bl:.0f}%",
                 "Tested": f"{-b:+.1f}%" if np.isfinite(b) else "–",
                 "Tested_Status": _status(abs(b) <= bl if np.isfinite(b) else None, np.isfinite(b) and abs(b) <= 2 * bl),
                 "Live": f"{-lb:+.1f}% ({lnm} months)" if lnm else "–",
                 "Live_Status": _status(abs(lb) <= bl if lnm >= HEALTH["live_min_months"] else None)})

    acc = accuracy_table(res, "Received")
    if len(acc):
        o = acc[(acc["Level"] == "Organisation") & (acc["Months_Ahead"] == "1-3")]
        if len(o):
            w, bw = float(o["WAPE_Pct"].iloc[0]), float(o["Benchmark_WAPE_Pct"].iloc[0])
            rows.append({"Check": "Better than 'same month last year'", "Target": "smaller monthly error",
                         "Tested": f"±{w:.0f}% vs ±{bw:.0f}%", "Tested_Status": _status(w < bw),
                         "Live": "–", "Live_Status": "Not yet"})

    dec = getattr(res, "decisions", None)
    if dec is not None and len(dec) and (dec["Component"] == "statistical_overlay").any():
        d = dec[dec["Component"] == "statistical_overlay"].iloc[0]
        rows.append({"Check": "Recent under/over-forecast corrected", "Target": "correction used when it tests better",
                     "Tested": "on" if str(d["Used"]) != "none" else "off",
                     "Tested_Status": _status(str(d["Used"]) == str(d["Test_Favours"])),
                     "Live": "reviewed Jan/Apr/Jul/Oct", "Live_Status": "Not yet"})

    fc = res.forecast
    dest = fc.loc[fc["Level"] == "Country", "Country"].nunique()
    if dest and "Human_Expected" in fc:
        k = fc.loc[fc["Level"].isin(["Country", "Region"]) & fc["Human_Expected"].notna(), "Country"].nunique()
        rows.append({"Check": "Market view from every destination", "Target": "live input sheet kept up to date",
                     "Tested": "–", "Tested_Status": "Not yet", "Live": f"{k} of {dest} destinations",
                     "Live_Status": _status(k == dest, k > 0)})
    return pd.DataFrame(rows)


def last_quarter_proof(res, months: int = 3) -> pd.DataFrame:
    """What the current procedure forecast `months` months ago for the months that have now happened:
    one row per destination x metric, with each month and the total. Uses only the walk-forward replay
    (forecasts made with data up to that month end)."""
    fva = getattr(res, "forecast_vs_actual", None)
    if fva is None or fva.empty:
        return pd.DataFrame()
    wf = fva[fva["Source"].astype(str).str.startswith("Walk-forward")]
    n_reg = wf[wf["Level"] == "Region"].groupby("Country")["Region"].nunique()
    single = wf[(wf["Level"] == "Region") & wf["Country"].isin(n_reg[n_reg == 1].index)
                & ~wf["Country"].isin(wf.loc[wf["Level"] == "Country", "Country"])].assign(Level="Country")
    f = pd.concat([wf[wf["Level"].isin(["Organisation", "Country"])], single], ignore_index=True)
    f["Made_At"] = pd.to_datetime(f["Made_At"])
    f["Target_Month"] = pd.to_datetime(f["Target_Month"])
    made = pd.Timestamp(res.last_actual) - pd.DateOffset(months=months)
    f = f[(f["Made_At"] == made) & f["Months_Ahead"].between(1, months)]
    if f.empty:
        return pd.DataFrame()
    rows = []
    for (lvl, cty, m), g in f.groupby(["Level", "Country", "Metric"], sort=False):
        g = g.sort_values("Target_Month")
        if len(g) < months:
            continue
        row = {"Level": lvl, "Country": "Organisation" if lvl == "Organisation" else cty, "Metric": m, "Made_At": made}
        for i, r in enumerate(g.itertuples(), start=1):
            row[f"M{i}"], row[f"F{i}"], row[f"A{i}"] = r.Target_Month, r.Forecast, r.Actual
        row["Forecast_Total"], row["Actual_Total"] = float(g["Forecast"].sum()), float(g["Actual"].sum())
        row["Miss_Pct"] = 100 * (row["Forecast_Total"] / row["Actual_Total"] - 1) if row["Actual_Total"] else np.nan
        rows.append(row)
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["_o"] = (d["Level"] != "Organisation").astype(int)
    return d.sort_values(["Metric", "_o", "Actual_Total"], ascending=[True, True, False]).drop(columns="_o")


def vintage_matrix(res, n_vintages: int = 5, metric: str = "Received") -> pd.DataFrame:
    """Miss (%) of each of the latest forecast vintages on the months already known, per destination.
    Vintages made 5, 4, 3 months ago cover a full quarter; 2 and 1 months ago, the latest 2 and 1 months.
    Columns are ordered oldest -> newest, so a turning point shows as a change of colour to the right."""
    fva = getattr(res, "forecast_vs_actual", None)
    if fva is None or fva.empty:
        return pd.DataFrame()
    wf = fva[fva["Source"].astype(str).str.startswith("Walk-forward") & (fva["Metric"] == metric)]
    n_reg = wf[wf["Level"] == "Region"].groupby("Country")["Region"].nunique()
    single = wf[(wf["Level"] == "Region") & wf["Country"].isin(n_reg[n_reg == 1].index)
                & ~wf["Country"].isin(wf.loc[wf["Level"] == "Country", "Country"])].assign(Level="Country")
    f = pd.concat([wf[wf["Level"].isin(["Organisation", "Country"])], single], ignore_index=True)
    f["Made_At"], f["Target_Month"] = pd.to_datetime(f["Made_At"]), pd.to_datetime(f["Target_Month"])
    last = pd.Timestamp(res.last_actual)
    cols, rows = [], {}
    for k in range(n_vintages, 0, -1):
        made = last - pd.DateOffset(months=k)
        x = f[(f["Made_At"] == made) & f["Months_Ahead"].between(1, 3) & (f["Target_Month"] <= last)]
        if x.empty:
            continue
        t0, t1 = x["Target_Month"].min(), x["Target_Month"].max()
        col = f"{made:%b} -> {t0:%b}" + (f"–{t1:%b}" if t1 != t0 else "")
        cols.append(col)
        g = x.groupby(["Level", "Country"])[["Forecast", "Actual"]].sum()
        for (lvl, cty), r in g.iterrows():
            key = ("Organisation" if lvl == "Organisation" else cty)
            rows.setdefault(key, {"Level": lvl, "Volume": 0.0})
            rows[key][col] = 100 * (r["Forecast"] / r["Actual"] - 1) if r["Actual"] > 0 else np.nan
            rows[key]["Volume"] += r["Actual"]
    d = pd.DataFrame.from_dict(rows, orient="index").rename_axis("Destination").reset_index()
    if d.empty:
        return d
    d["_o"] = (d["Level"] != "Organisation").astype(int)
    d = d.sort_values(["_o", "Volume"], ascending=[True, False]).drop(columns=["_o", "Volume"])
    return d[["Destination", "Level"] + cols]
