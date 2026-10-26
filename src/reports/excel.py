# -*- coding: utf-8 -*-
"""Executive Excel workbook — the working tool for planners, with full evidence sheets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .. import __version__
from ..data.prepare import ORG, TOTAL
from . import insights as ins

CONF_COLORS = {"High": "#d7f0dd", "Medium": "#fff1c7", "Low": "#fbd9d6", "New": "#eceff3", "Review": "#fde2c8"}
CLASS_FILL = {"High": "#cfe1f7", "Medium": "#f3f6fa", "Low": "#fdf1e3"}
TAB_COLORS = {"plan": "#2a78d6", "analysis": "#1baf7a", "evidence": "#8a8a85"}
SHEET_GROUP = {
    "Read Me": "plan", "Executive Summary": "plan", "Country Outlook": "plan", "Monthly Forecast": "plan",
    "Fiscal Periods": "plan", "Staffing Calendar": "plan",
    "Seasonality": "analysis", "Forecast vs Actual": "analysis", "Statistical Overlay": "analysis",
    "Human Inputs": "analysis", "Policy Events": "analysis", "Intake Profile": "analysis",
}
# plain-English column headers (the CSV files keep the technical names)
FRIENDLY = {
    "Label": "Series", "Date": "Month", "Horizon": "Months ahead", "Forecast": "Forecast",
    "Lo50": "Most likely low (50%)", "Hi50": "Most likely high (50%)",
    "Outlook_Lo80": "Outlook plan low (80%)", "Outlook_Hi80": "Outlook plan high (80%)",
    "Outlook_Lo50": "Outlook most likely low", "Outlook_Hi50": "Outlook most likely high",
    "Lo80": "Plan low (80% range)", "Hi80": "Plan high (80% range)", "Lo95": "Low (95% range)", "Hi95": "High (95% range)",
    "Forecast_Adjusted": "Adjusted forecast", "Adjusted_Lo80": "Adjusted low (80%)", "Adjusted_Hi80": "Adjusted high (80%)",
    "Overlay_Multiplier": "Business adjustment (x)", "Base_Forecast": "Before reconciliation",
    "Same_Months_Last_Year": "Same months last year", "YoY_Pct": "vs last year (%)", "Window": "Period",
    "Actual_To_Date": "Actual to date", "Forecast_Remaining": "Forecast remaining", "Outlook": "Outlook (actual + forecast)",
    "Outlook_Adjusted": "Adjusted outlook", "Same_Period_Last_Year": "Same period last year",
    "Months_Actual": "Months actual", "Months_Forecast": "Months forecast", "Period_Start": "Period start",
    "Selected_Models": "Models used", "Selection_Reason": "Why this method", "Best_Model": "Best single model",
    "Runner_Up": "Runner-up", "Margin_Best_vs_RunnerUp_Pct": "Best vs runner-up (% better)",
    "Best_HB_MAE": "Best model error (avg units/month)", "Benchmark_HB_MAE": "Same-month-last-year error",
    "Honest_WAPE_Policy": "Walk-forward error (method)", "Honest_WAPE_Benchmark": "Walk-forward error (benchmark)",
    "Honest_Skill_vs_Benchmark": "Skill vs benchmark", "Honest_N": "Walk-forward tests", "N_Candidates": "Models tested",
    "N_Origins": "Backtest origins", "Required_Train_Months": "Training months required",
    "Eligible_Models": "Eligible models", "Months_Of_History": "Months of history",
    "Statistical_Overlay": "Statistical overlay (x)", "HB_MAE": "Avg error (units/month, 1-12 ahead)",
    "Rel_to_Benchmark": "Error vs benchmark (x)", "WAPE": "WAPE", "Bias_Pct": "Bias (%)",
    "Seasonal_Index": "Seasonal index (1 = average month)", "Class": "Volume class",
    "Consistency_Pct": "Same class in % of years", "Month_Name": "Month", "Load_Index": "vs average month (x)",
    "Workload_Class": "Workload", "Historical_Class": "Usual for this month", "Capacity_Plan_80": "Plan for (80% high)",
    "Low_80": "Low (80%)", "FTE_Base": "FTE (forecast)", "FTE_Capacity_80": "FTE (80% high)",
    "Target_Month": "Month", "Reliability": "Reliability",
    "Typical_Quarter_Error_Pct": "Typical quarter error (±%)", "Apps_Per_FTE_Month": "Applications per FTE per month",
    "Productivity_Source": "Productivity source", "Made_At": "Forecast made at", "Months_Ahead": "Months ahead",
    "Error_Pct": "Error (%)", "Human_Expected": "Manager expectation", "Human_vs_Model_Pct": "Manager vs model (%)",
    "Forecast_Human_Blend": "Blended (proven managers only)", "Human_Submitted_By": "Submitted by",
    "Human_Reason": "Manager reason", "Measured_Impact_Pct": "Measured impact (%)",
    "Normal_Error_Range_80_Pct": "Normal forecast error range (80%)", "Coverage_Before_Pct": "Coverage before (%)",
    "Coverage_After_Pct": "Coverage after (%)", "Target_Pct": "Target (%)", "WAPE_Pct": "Error WAPE (%)",
}


def _hdr(c: str) -> str:
    return FRIENDLY.get(c, str(c).replace("_", " "))


def _friendly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("Country", "Region"):
        if c in df.columns:
            df[c] = df[c].replace({ORG: "Organisation", TOTAL: "Total"})
    return df


class _Book:
    def __init__(self, path: Path):
        self.w = pd.ExcelWriter(path, engine="xlsxwriter", datetime_format="mmm yyyy")
        wb = self.w.book
        self.f = {
            "title": wb.add_format({"bold": True, "font_size": 16, "font_color": "#0b0b0b"}),
            "h2": wb.add_format({"bold": True, "font_size": 12, "font_color": "#0b0b0b"}),
            "muted": wb.add_format({"font_color": "#52514e", "italic": True}),
            "wrap": wb.add_format({"text_wrap": True, "valign": "top"}),
            "hdr": wb.add_format({"bold": True, "bg_color": "#eef2f7", "border": 1, "border_color": "#d0d7e2",
                                  "text_wrap": True, "valign": "vcenter"}),
            "int": wb.add_format({"num_format": "#,##0"}),
            "int_b": wb.add_format({"num_format": "#,##0", "bold": True}),
            "pct": wb.add_format({"num_format": "+0.0%;-0.0%;0.0%"}),
            "pct_plain": wb.add_format({"num_format": "0.0"}),
            "dec": wb.add_format({"num_format": "0.00"}),
            "month": wb.add_format({"num_format": "mmm yyyy"}),
        }
        for k, col in CONF_COLORS.items():
            self.f[f"conf_{k}"] = wb.add_format({"bg_color": col})

    def table(self, name: str, df: pd.DataFrame, widths: Optional[dict] = None, formats: Optional[dict] = None,
              note: str = "", start_row: int = 0, conf_col: Optional[str] = None):
        df = _friendly(df).reset_index(drop=True)
        r0 = start_row + (2 if note else 0)
        df.to_excel(self.w, sheet_name=name, index=False, startrow=r0)
        ws = self.w.sheets[name]
        if note:
            ws.write(start_row, 0, note, self.f["muted"])
        for j, c in enumerate(df.columns):
            ws.write(r0, j, _hdr(c), self.f["hdr"])
            fmt = self.f.get((formats or {}).get(c, ""), None)
            if fmt is None:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    fmt = self.f["month"]
                elif pd.api.types.is_float_dtype(df[c]):
                    fmt = self.f["int"]
            ws.set_column(j, j, (widths or {}).get(c, max(11, min(40, len(_hdr(c)) + 2))), fmt)
        ws.freeze_panes(r0 + 1, 0)
        if len(df):
            ws.autofilter(r0, 0, r0 + len(df), len(df.columns) - 1)
        grp = SHEET_GROUP.get(name, "evidence")
        ws.set_tab_color(TAB_COLORS[grp])
        for cc in ("Class", "Workload_Class", "Historical_Class"):
            if cc in df.columns and len(df):
                j = list(df.columns).index(cc)
                for k, colr in CLASS_FILL.items():
                    ws.conditional_format(r0 + 1, j, r0 + len(df), j, {
                        "type": "cell", "criteria": "==", "value": f'"{k}"',
                        "format": self.w.book.add_format({"bg_color": colr})})
        if conf_col and conf_col in df.columns:
            j = list(df.columns).index(conf_col)
            for k in CONF_COLORS:
                ws.conditional_format(r0 + 1, j, r0 + len(df), j, {"type": "cell", "criteria": "==",
                                                                    "value": f'"{k}"', "format": self.f[f"conf_{k}"]})
        return ws

    def close(self):
        self.w.close()


def write_workbook(res, path: Path) -> Path:
    b = _Book(path)
    f = b.f
    k = ins.kpis(res)

    # ── Read Me ──────────────────────────────────────────────────────────
    ws = b.w.book.add_worksheet("Read Me")
    b.w.sheets["Read Me"] = ws
    ws.set_column(0, 0, 28); ws.set_column(1, 1, 110, f["wrap"])
    ws.write(0, 0, "Application Flow Forecast", f["title"])
    ws.write(1, 0, f"Data through {res.last_actual:%B %Y} · run {res.run_id} · pipeline v{__version__}", f["muted"])
    readme = [
        ("What this is", "12-month forecast of applications Received and Submitted for the organisation, each "
                         "country and each region. Months follow the 26th-to-25th reporting cycle."),
        ("Forecast", "The most likely value (statistical baseline). Regions add up exactly to countries and the organisation."),
        ("80% / 95% range", "Where the actual is expected to land 8 / 19.5 times in 10 — measured on past forecasts, "
                            "not assumed. Use the 80% range for planning, the 95% range for stress tests."),
        ("Adjusted", "Baseline × approved policy/business adjustments (Policy Overlay sheet). "
                     "Equal to the baseline when no adjustment is approved."),
        ("Reliability", "How precisely a series can be forecast, measured on past quarters: High = quarter totals "
                        "typically within ±10%, Medium ±10–20%, Low beyond ±20% (usually small or volatile markets), "
                        "New = too little history. Every series uses the most accurate method found; reliability is "
                        "about the market, not a weaker method. "
                       "new series and guardrail-capped series are always Low."),
        ("How models were chosen", f"Seventeen candidate methods were each tested on {ins.test_depth(res)['origins']} past forecast origins over the full "
                                   "12 months ahead. Ways of combining them were replayed walk-forward (Policy Comparison "
                                   "sheet); the equal-weight average of all methods won and is used. A series falls back to "
                                   "'same month last year' if the average did not beat it (Model Evidence sheet)."),
        ("Sheets", "Executive Summary · Country Outlook · Monthly Forecast · Fiscal Periods (for planning) — "
                   "Staffing Calendar · Seasonality · Forecast vs Actual · Statistical Overlay · Human Inputs · "
                   "Policy Events · Intake Profile (analysis) — Model Evidence · Model Scores · Accuracy by Horizon · Policy Comparison · "
                   "Health Check · Range Method Test · Interval Calibration · Model Catalogue · "
                   "Guardrails · Policy Overlay · Data Quality · Track Record · Run Manifest (for audit)."),
        ("Reproducibility", "Run Manifest records the code version (git commit), the data file fingerprint (SHA-256) "
                            "and every setting; the same inputs reproduce the same numbers."),
    ]
    for i, (a, t) in enumerate(readme):
        ws.write(3 + i, 0, a, f["h2"]); ws.write(3 + i, 1, t, f["wrap"])

    # ── Executive Summary ────────────────────────────────────────────────
    ws = b.w.book.add_worksheet("Executive Summary")
    b.w.sheets["Executive Summary"] = ws
    ws.set_column(0, 0, 26); ws.set_column(1, 8, 15)
    ws.write(0, 0, "Executive Summary — Organisation", f["title"])
    ws.write(1, 0, f"Data through {res.last_actual:%B %Y}. Baseline forecast with 80% range.", f["muted"])
    hdr = ["Window", "Metric", "Forecast", "Low (80%)", "High (80%)", "Same months LY", "YoY", "Adjusted"]
    for j, h in enumerate(hdr):
        ws.write(3, j, h, f["hdr"])
    r = 4
    for m in ("Received", "Submitted"):
        for wname, row in k[m].items():
            ws.write(r, 0, wname); ws.write(r, 1, m)
            ws.write_number(r, 2, row["Forecast"], f["int_b"])
            for j, key in ((3, "Lo80"), (4, "Hi80"), (5, "Same_Months_Last_Year"), (7, "Forecast_Adjusted")):
                v = row.get(key, np.nan)
                if v is not None and np.isfinite(v):
                    ws.write_number(r, j, v, f["int"])
            if np.isfinite(row["YoY_Pct"]):
                ws.write_number(r, 6, row["YoY_Pct"] / 100, f["pct"])
            r += 1
    r += 1
    fy = ins.fiscal_year_outlook(res)
    if len(fy):
        ws.write(r, 0, "Fiscal-year outlook (actual to date + forecast)", f["h2"]); r += 1
        for j, h in enumerate(["Fiscal year", "Metric", "Actual to date", "Forecast remaining", "Outlook",
                               "Last year", "YoY"]):
            ws.write(r, j, h, f["hdr"])
        r += 1
        for row in fy.itertuples():
            ws.write(r, 0, row.Period); ws.write(r, 1, row.Metric)
            for j, v in ((2, row.Actual_To_Date), (3, row.Forecast_Remaining), (4, row.Outlook),
                         (5, row.Same_Period_Last_Year)):
                ws.write_number(r, j, v, f["int"])
            if np.isfinite(row.YoY_Pct):
                ws.write_number(r, 6, row.YoY_Pct / 100, f["pct"])
            r += 1
    r += 1
    ws.write(r, 0, "Key messages", f["h2"]); r += 1
    for msg in ins.messages(res):
        ws.merge_range(r, 0, r, 8, "• " + msg, f["wrap"]); ws.set_row(r, 30); r += 1
    ws.set_tab_color(TAB_COLORS["plan"])
    b.w.sheets["Read Me"].set_tab_color(TAB_COLORS["plan"])
    _summary_chart(b, ws, res, anchor="K4")

    # ── planning sheets ──────────────────────────────────────────────────
    ct = ins.country_table(res)
    b.table("Country Outlook", ct, widths={"Country": 28},
            formats={c: "pct_plain" for c in ct.columns if c.endswith("YoY_Pct")},
            note="Next 12 months by country. YoY % compares with the same 12 months last year (all series, incl. closed ones).")

    fc = res.forecast
    mcols = ["Label", "Level", "Country", "Region", "Metric", "Date", "Horizon", "Forecast", "Lo80", "Hi80",
             "Lo95", "Hi95", "Forecast_Adjusted", "Adjusted_Lo80", "Adjusted_Hi80", "Overlay_Multiplier",
             "Reliability", "Typical_Quarter_Error_Pct", "Method", "Models", "Base_Forecast"]
    b.table("Monthly Forecast", fc[[c for c in mcols if c in fc.columns]],
            widths={"Label": 36, "Method": 34, "Models": 34, "Date": 11},
            formats={"Overlay_Multiplier": "dec", "Horizon": ""}, conf_col="Reliability",
            note="Base_Forecast = before reconciliation. Forecast = reconciled baseline (use this).")
    b.table("Fiscal Periods", res.periods, widths={"Label": 36},
            formats={"YoY_Pct": "pct_plain", "Months_Actual": "", "Months_Forecast": ""},
            note="Fiscal year starts in April. Outlook = actuals already realised + forecast for the remaining months.")

    # ── staffing & analysis sheets ──────────────────────────────────────
    st = getattr(res, "staffing", pd.DataFrame())
    if st is not None and len(st):
        stc = st.drop(columns=["Country", "Region"], errors="ignore")
        wss = b.table("Staffing Calendar", stc, widths={"Label": 30},
                      formats={"Load_Index": "dec", "FTE_Base": "dec", "FTE_Capacity_80": "dec"},
                      note="Next 12 months. Workload High / Low = at least 10% above / below the average forecast month. "
                           "Plan capacity on 'Plan for (80% high)'. Set apps_per_fte_month in config to get FTE columns.")
        _staffing_chart(b, wss, st)
    si = getattr(res, "seasonal_index", pd.DataFrame())
    if si is not None and len(si):
        piv = si[si["Level"].isin(["Organisation", "Country"])].copy()
        piv["Series"] = [ev_label(l, c, r_) for l, c, r_ in zip(piv["Level"], piv["Country"], piv["Region"])]
        wide = piv.pivot_table(index=["Series", "Metric"], columns="Month_Name", values="Seasonal_Index", sort=False)
        wide = wide[[m for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                     if m in wide.columns]].reset_index()
        wsz = b.table("Seasonality", wide, widths={"Series": 30}, formats={m: "dec" for m in wide.columns[2:]},
                      note="Seasonal index by calendar month (1.00 = average month; 1.25 = 25% busier). "
                           "Blue = busier, orange = quieter. Robust STL on the last 4 years.")
        if len(wide):
            wsz.conditional_format(3, 2, 2 + len(wide), len(wide.columns) - 1, {
                "type": "3_color_scale", "min_color": "#fdf1e3", "mid_color": "#ffffff", "max_color": "#9ec5f0",
                "mid_type": "num", "mid_value": 1.0})
        long = si.drop(columns=["Rank"], errors="ignore").copy()
        long["Series"] = [ev_label(l, c, r_) for l, c, r_ in zip(long["Level"], long["Country"], long["Region"])]
        b.table("Seasonality", long[["Series", "Level", "Metric", "Month_Name", "Seasonal_Index", "Class",
                                     "Consistency_Pct", "Years_Used"]],
                start_row=len(wide) + 5, widths={"Series": 30},
                formats={"Seasonal_Index": "dec", "Consistency_Pct": "pct_plain", "Years_Used": ""},
                note="Every series: class of each calendar month and how consistently it held across years.")
    fs = getattr(res, "fva_summary", pd.DataFrame())
    fv = getattr(res, "forecast_vs_actual", pd.DataFrame())
    if fs is not None and len(fs):
        fs2 = fs.drop(columns=["Country"], errors="ignore")
        wsf = b.table("Forecast vs Actual", fs2, formats={c: "pct_plain" for c in fs2.columns if c.startswith("Error_")},
                      note="Organisation: actual vs the forecast made 1, 3, 6 and 12 months earlier (walk-forward: what "
                           "the current method would have said at the time). Live forecasts are added as actuals arrive.")
        _fva_chart(b, wsf, fs2)
        if fv is not None and len(fv):
            full = fv.copy()
            full["Series"] = [ev_label(l, c, r_) for l, c, r_ in zip(full["Level"], full["Country"], full["Region"])]
            cols = ["Series", "Metric", "Target_Month", "Made_At", "Months_Ahead", "Forecast", "Actual", "Error_Pct",
                    "Source"]
            b.table("Forecast vs Actual", full[[c for c in cols if c in full.columns]], start_row=len(fs2) + 24,
                    widths={"Series": 30, "Source": 38}, formats={"Error_Pct": "pct_plain", "Months_Ahead": ""},
                    note="Every forecast vs actual, all series (filterable).")
    bt_ = getattr(res, "bias_test", pd.DataFrame())
    be = getattr(res, "bias_estimates", pd.DataFrame())
    if bt_ is not None and len(bt_):
        mode = res.manifest.get("statistical_overlay_mode", "none")
        b.table("Statistical Overlay", bt_, formats={"WAPE_Pct": "pct_plain", "Bias_Pct": "pct_plain",
                                                     "Pct_Series_Improved": "pct_plain"},
                note=f"Recent-bias correction tested walk-forward. Adopted mode: '{mode}' ('none' = not applied because "
                     "it did not improve accuracy by at least the configured threshold).")
        if be is not None and len(be):
            be2 = be.merge(res.nodes[["node_id", "Level", "Country", "Region", "Metric"]], on="node_id", how="left")
            be2["Series"] = [ev_label(l, c, r_) for l, c, r_ in zip(be2["Level"], be2["Country"], be2["Region"])]
            be2["Multiplier"] = np.exp(be2["Correction_Log"])
            b.table("Statistical Overlay", be2[["Series", "Metric", "Bias_Log", "N", "Shrink_Weight", "Multiplier",
                                                 "Mode_Adopted"]],
                    start_row=len(bt_) + 5, widths={"Series": 30},
                    formats={"Bias_Log": "dec", "Shrink_Weight": "dec", "Multiplier": "dec", "N": ""},
                    note="Per series: recent log bias (actual vs forecast), empirical-Bayes shrink weight, resulting multiplier.")
    hs = getattr(res, "human_scores", pd.DataFrame())
    hcols = ["Label", "Metric", "Date", "Forecast", "Human_Expected", "Human_vs_Model_Pct", "Forecast_Human_Blend",
             "Human_Submitted_By", "Human_Reason"]
    hrows = (fc[fc["Human_Expected"].notna()][[c for c in hcols if c in fc.columns]]
             if "Human_Expected" in fc.columns else pd.DataFrame())
    if len(hrows) or (hs is not None and len(hs)):
        b.table("Human Inputs", hrows if len(hrows) else pd.DataFrame({"Status": ["No manager inputs this run."]}),
                widths={"Label": 30, "Human_Reason": 50, "Status": 60}, formats={"Human_vs_Model_Pct": "pct_plain"},
                note="Manager expectations (Settings.xlsx sheet Human_Inputs) next to the model. Blended only for people "
                     "with >= 6 scored months who beat the model on those months.")
        if hs is not None and len(hs):
            b.table("Human Inputs", hs, start_row=max(len(hrows), 1) + 5,
                    formats={"Human_WAPE_Pct": "pct_plain", "Model_WAPE_Pct": "pct_plain", "Blend_Weight": "dec",
                             "Months_Scored": ""}, note="Track record of each person's inputs.")
    else:
        b.table("Human Inputs", pd.DataFrame({"How to use": [
            "Add a sheet 'Human_Inputs' to Settings.xlsx with columns: Country, Region (blank = country total), Metric, "
            "Month (YYYY-MM), Expected_Volume, Submitted_By, Reason, Submitted_On.",
            "Every run records the inputs, shows them next to the model and scores them against actuals.",
            "After 6 scored months, a person who beats the model on the same months is blended in automatically "
            "(inverse-error weights)."]}), widths={"How to use": 120})
    evn = getattr(res, "events", pd.DataFrame())
    b.table("Policy Events", evn if evn is not None and len(evn) else pd.DataFrame({"Status": ["No events listed."]}),
            widths={"Event": 40, "Notes": 50, "Normal_Error_Range_80_Pct": 22},
            formats={"Measured_Impact_Pct": "pct_plain", "Months_Measured": ""},
            note="Measured impact of policy events (Settings.xlsx sheet Policy_Events): actual vs a counterfactual "
                 "forecast made just before the event. 'Significant' = outside the normal forecast error range "
                 "(placebo test at 12 earlier dates). Use measured impacts, not guesses, for future adjustments.")

    ia = getattr(res, "intake_arrivals", pd.DataFrame())
    if ia is not None and len(ia):
        il = getattr(res, "intake_lead_profile", pd.DataFrame())
        im = getattr(res, "intake_month_mix", pd.DataFrame())
        org_l = il[il["Level"] == "Organisation"] if len(il) else il
        if len(org_l):
            curve = org_l.pivot_table(index="Months_Before_Start", columns="Season", values="Avg_Cumulative_Pct")
            curve = curve.sort_index(ascending=False).reset_index()
            b.table("Intake Profile", curve, formats={c: "pct_plain" for c in curve.columns if c != "Months_Before_Start"},
                    note="Typical arrival curve (organisation): % of an intake's applications already received N months "
                         "before the intake starts, by season. Descriptive only — the forecast itself is monthly.")
        start = (len(org_l.pivot_table(index="Months_Before_Start", columns="Season", values="Avg_Cumulative_Pct"))
                 if len(org_l) else 0) + 5
        ia2 = ia.copy()
        ia2["Series"] = np.where(ia2["Level"] == "Organisation", "Organisation", ia2["Country"])
        b.table("Intake Profile", ia2[["Series", "Intake_Label", "Season", "Intake_Status", "Date", "Months_Before_Start",
                                       "Received", "Share_of_Intake_Pct", "Cumulative_Pct"]],
                start_row=start, widths={"Series": 26}, formats={"Share_of_Intake_Pct": "pct_plain",
                                                               "Cumulative_Pct": "pct_plain", "Months_Before_Start": ""},
                note="For each intake: applications received in each month (filter by Series / Intake).")
        if len(im):
            mm = im[im["Level"] == "Organisation"].pivot_table(index="Application_Month", columns="Season",
                                                               values="Share_Pct").reset_index()
            b.table("Intake Profile", mm, start_row=start + len(ia2) + 5,
                    formats={c: "pct_plain" for c in mm.columns if c != "Application_Month"},
                    note="Intake mix of each application month (organisation, last 3 years): % going to each season.")
    else:
        b.table("Intake Profile", pd.DataFrame({"Status": [
            "Intake extract not available yet. Run: python run_pipeline.py fetch-data, then transform-data "
            "(creates intake_final_summary.xlsx from the Intake / InYear fields)."]}), widths={"Status": 120})

    # ── evidence sheets ─────────────────────────────────────────────────
    meta = res.nodes[["node_id", "Level", "Country", "Region", "Metric", "Months_Of_History"]]
    ev = meta.merge(res.selection, on="node_id", how="right")
    ev["Label"] = [ev_label(l, c, r_) for l, c, r_ in zip(ev["Level"], ev["Country"], ev["Region"])]
    conf = fc.drop_duplicates("node_id").set_index("node_id")["Reliability"]
    ev["Reliability"] = ev["node_id"].map(conf)
    ecols = ["Label", "Metric", "Level", "Method", "Selected_Models", "Reliability", "Selection_Reason", "Best_Model",
             "Runner_Up", "Margin_Best_vs_RunnerUp_Pct", "Best_HB_MAE", "Benchmark_HB_MAE", "Honest_WAPE_Policy",
             "Honest_WAPE_Benchmark", "Honest_Skill_vs_Benchmark", "Honest_N", "N_Candidates", "N_Origins",
             "Required_Train_Months", "Eligible_Models", "Months_Of_History"]
    b.table("Model Evidence", ev[[c for c in ecols if c in ev.columns]],
            widths={"Label": 36, "Selection_Reason": 90, "Method": 34, "Selected_Models": 32, "Eligible_Models": 50},
            formats={"Margin_Best_vs_RunnerUp_Pct": "pct_plain", "Honest_WAPE_Policy": "dec",
                     "Honest_WAPE_Benchmark": "dec", "Honest_Skill_vs_Benchmark": "dec", "Honest_N": "",
                     "N_Candidates": "", "N_Origins": "", "Required_Train_Months": "", "Months_Of_History": ""},
            conf_col="Reliability",
            note="Why each series uses its method. Honest_* = walk-forward replay of the selection rule (no selection bias). "
                 "Skill > 0 means better than 'same month last year'.")
    sc = meta.merge(res.scores, on="node_id", how="right")
    sc["Label"] = [ev_label(l, c, r_) for l, c, r_ in zip(sc["Level"], sc["Country"], sc["Region"])]
    b.table("Model Scores", sc[["Label", "Metric", "Model", "Rank", "HB_MAE", "Rel_to_Benchmark", "WAPE", "Bias_Pct",
                                "MAE_h1-3", "MAE_h4-6", "MAE_h7-12", "N_Origins", "N_Pairs"]],
            widths={"Label": 36}, formats={"Rel_to_Benchmark": "dec", "WAPE": "dec", "Bias_Pct": "pct_plain",
                                           "Rank": "", "N_Origins": "", "N_Pairs": ""},
            note="Every candidate for every series. HB_MAE = MAE averaged over horizons 1-12 across all rolling origins. "
                 "Rel_to_Benchmark < 1 beats seasonal naive.")
    acc = res.accuracy
    if len(acc):
        pv = acc.pivot_table(index=["Level", "Metric", "h"], columns="Method", values="WAPE_Pct").reset_index()
        b.table("Accuracy by Horizon", pv, formats={c: "pct_plain" for c in pv.columns if c not in ("Level", "Metric", "h")},
                note=f"Weighted absolute % error (WAPE) of forecasts made at the last {ins.test_depth(res)['wf']} origins, by months ahead. "
                     "'Selected method' is the walk-forward replay of the full procedure.")
    pc = res.policy_comparison
    if pc is not None and len(pc):
        b.table("Policy Comparison", pc, widths={"Method": 34},
                formats={"WAPE_Pct": "pct_plain", "Bias_Pct": "pct_plain", "Median_Error_vs_Benchmark": "dec",
                         "Pct_Series_Beating_Benchmark": "pct_plain", "N": ""},
                note="Walk-forward test of each way of forming the forecast (re-run every month). The configured policy "
                     f"is '{ins.policy_label(res)}'. Bias < 0 = actuals above forecast.")
    cal_fmt = {"Offset_Lo": "dec", "Offset_Hi": "dec", "Offset_Lo_z": "dec", "Offset_Hi_z": "dec",
               "Coverage_Before_Pct": "pct_plain", "Coverage_After_Pct": "pct_plain", "Target_Pct": "", "N": ""}
    rt = getattr(res, "reconciliation_test", pd.DataFrame())
    if rt is not None and len(rt):
        rtw = rt.pivot_table(index=["Metric", "Level"], columns="Method", values="WAPE_Pct").reset_index()
        b.table("Reconciliation Test", rtw, formats={c: "pct_plain" for c in rtw.columns[2:]},
                note="Walk-forward error (WAPE %) of each way of making regions, countries and the organisation add up; "
                     f"'none' = each level forecast on its own. Chosen: '{res.manifest.get('reconciliation_method_chosen', '')}'.")
    cal_fmt["Width_Factor"] = "dec"
    b.table("Health Check", ins.health_check(res), widths={"Check": 42, "Target": 34, "Tested": 18, "Live": 22},
            note="Acceptance checks fixed in advance and scored every month. Tested = forecasts re-made for past months "
                 "with only the data then available; Live = published forecasts as their actuals arrive. Trust a use once "
                 "the live column passes for 3 consecutive months.")
    st_tot = getattr(res, "stage_totals", None)
    if st_tot is not None and len(st_tot):
        from ..forecasting.stages import CHAIN
        order = [s for s, _ in CHAIN if s in set(st_tot["StageGroup"])]            # pipeline order, not alphabetical
        stw = st_tot.pivot_table(index=["Level", "Country", "Window"], columns="StageGroup", values="Forecast",
                                 aggfunc="sum")[order].reset_index()
        b.table("Stage Forecast", stw, formats={c: "int" for c in stw.columns[3:]},
                note="Applications expected to reach each stage after submission (first time in the 26th-25th cycle), "
                     "forecast from the stage before it (conversion chain) or same month last year, whichever tested "
                     "better on next-quarter totals.")
        yoy = st_tot.pivot_table(index=["Level", "Country", "Window"], columns="StageGroup", values="YoY_Pct",
                                 dropna=False).reindex(columns=order).reset_index()
        b.table("Stage Forecast", yoy, start_row=len(stw) + 5, formats={c: "pct_plain" for c in yoy.columns[3:]},
                note="Change vs the same months last year (%); left blank where last year had fewer than 150.")
        conv = getattr(res, "stage_conversion", None)
        if conv is not None and len(conv):
            b.table("Stage Forecast", conv, start_row=len(stw) + len(yoy) + 10,
                    note="Conversion from the previous stage over the last 12 months (%), and how it is spread over the "
                         "months after (lag 0 = same month).")
        stt = getattr(res, "stage_test", None)
        if stt is not None and len(stt):
            b.table("Stage Evidence", stt, formats={"Quarter_Error_Pct": "pct_plain", "N": ""},
                    note="Walk-forward test per stage: typical miss (%) on next-quarter totals for same month last year "
                         "(snaive), the conversion chain and their blend. Chosen = used (must beat snaive by >= 3%).")
    ipace = getattr(res, "intake_pace", None)
    if ipace is not None and len(ipace):
        cols = ["Level", "Country", "Region", "Metric", "Intake", "Phase", "Months_To_Start", "To_Date", "Last_Year_Same_Point",
                "Pace_Pct", "Status", "Typical_Share_In_Pct", "Projected_Final", "Lo80", "Hi80", "Last_Year_Final",
                "Projection_Miss_Pct", "Last_Year_Miss_Pct"]
        b.table("Intake Running", ipace[[c for c in cols if c in ipace.columns]],
                formats={c: "int" for c in ("To_Date", "Last_Year_Same_Point", "Projected_Final", "Lo80", "Hi80",
                                            "Last_Year_Final")} | {c: "pct_plain" for c in ("Pace_Pct",
                                            "Typical_Share_In_Pct", "Projection_Miss_Pct", "Last_Year_Miss_Pct")},
                note="Each intake season (by intake month: Spring Jan-Mar, Summer Apr-Jul, Fall Aug-Dec; Closing = the "
                     "season of the current month, Ongoing = the next to start, Next = the two after): applications so "
                     "far vs the same season a year earlier at the same point (pace). Projected final = so far / the share the last three "
                     "same-month intakes had at this point, only where that tested better than 'same as last year'.")
        itest = getattr(res, "intake_pace_test", None)
        if itest is not None and len(itest):
            b.table("Intake Test", itest, formats={"Projection_Miss_Pct": "pct_plain", "Last_Year_Miss_Pct": "pct_plain",
                                                   "N": ""},
                    note="Walk-forward test on completed intakes: typical miss (%) of the projected final at each lead "
                         "(months before the start) vs 'same as last year's final'. Used = published (>= 3% better, "
                         ">= 8 intakes).")
    ctest = getattr(res, "calendar_test", None)
    if ctest is not None and len(ctest):
        ctw = ctest.pivot_table(index=["Window", "Level"], columns="Variant", values="Error_Pct").reset_index()
        b.table("Calendar Effects Test", ctw, formats={c: "pct_plain" for c in ctw.columns[2:]},
                note="Receipts: typical miss (%) of single months 1-3 ahead and of the next quarter, without and with "
                     "working-day and festival effects (Diwali, Holi, Eid) per 26th-25th cycle. "
                     f"Used: {res.manifest.get('calendar_effects_used', False)}; effects (log): "
                     f"{res.manifest.get('calendar_effects_log_coefficients', {})}.")
    dtab = getattr(res, "decisions", None)
    if dtab is not None and len(dtab):
        b.table("Method Decisions", dtab, widths={"Reason": 60, "Scores": 60},
                note="Each method choice is re-tested every run but only changed in the first month of a quarter "
                     "(Jan/Apr/Jul/Oct), or earlier if the evidence is at least 6% in favour. Scores = next-quarter "
                     "(or monthly, for calendar effects) error averaged over levels; lower is better.")
    vtest = getattr(res, "vintage_test", None)
    if vtest is not None and len(vtest):
        vtw = vtest.pivot_table(index="Level", columns="Variant", values="Quarter_Error_Pct").reset_index()
        b.table("Direction Signal Test", vtw, formats={c: "pct_plain" for c in vtw.columns[1:]},
                note="Receipts: next-quarter typical miss (%) without and with the direction signal from the latest "
                     "forecast vintages (last 3 full quarters + latest 2 partial, at country level, half strength). "
                     f"Used: {res.manifest.get('vintage_signal_used', False)}. Latest signal by country (% actual vs "
                     f"forecast): {res.manifest.get('vintage_signal_pct_latest', {})}.")
    ft = getattr(res, "funnel_test", None)
    if ft is not None and len(ft):
        ftw = ft.pivot_table(index="Level", columns="Variant", values="Quarter_Error_Pct").reset_index()
        b.table("Submitted Funnel Test", ftw, formats={c: "pct_plain" for c in ftw.columns[1:]},
                note="Walk-forward typical miss (%) on next-quarter submissions. model = time-series forecast; "
                     "funnel = recent conversion rate x receipts (actual to date, forecast after); blend = average. "
                     f"Used: '{res.manifest.get('submitted_method', 'model')}'.")
    rtest = getattr(res, "range_test", None)
    if rtest is not None and len(rtest):
        b.table("Range Method Test", rtest,
                formats={"Coverage_Pct": "pct_plain", "Median_Width_Pct": "pct_plain", "Interval_Score": "dec", "N": ""},
                note="How each way of building a range would have performed, replayed at each of the last test origins "
                     "with only the errors known at the time. Interval score = width + penalty for misses (lower is "
                     "better; a proper scoring rule). Chosen = used for that group and window. 'centred' = symmetric "
                     "around the forecast; 'empirical' = past error quantiles (may be skewed).")
    rt2 = getattr(res, "range_tested", None)
    if rt2 is not None and len(rt2):
        b.table("Range Method Test", rt2, start_row=(len(rtest) + 5 if rtest is not None and len(rtest) else 0),
                formats={"Tested_Coverage_Pct": "pct_plain", "Target_Pct": "", "N": ""},
                note="Fully out-of-sample check of the whole range procedure: at each test origin the method and width "
                     "were decided only from earlier realised results, then compared with what happened. "
                     "Tested coverage should be close to the target.")
    b.table("Interval Calibration", res.calibration, formats=cal_fmt,
            note="Monthly ranges. Coverage = share of test actuals inside the range fitted only on earlier errors. "
                 "Width factor (0.8-1.5) scales both sides equally when coverage was clearly off target.")
    bc = getattr(res, "block_calibration", None)
    if bc is not None and len(bc):
        b.table("Interval Calibration", bc, formats=cal_fmt, start_row=len(res.calibration) + 5,
                note="Ranges for next 3/6/12-month totals.")
    b.table("Model Catalogue", res.catalogue, widths={"Description": 90},
            formats={"Median_Error_vs_Benchmark": "dec", "Pct_Series_Beating_Benchmark": "pct_plain",
                     "Min_History_Months": "", "Series_Using_Model": ""})
    g = res.guardrails.merge(meta, on="node_id", how="left") if len(res.guardrails) else res.guardrails
    if len(g):
        g["Label"] = [ev_label(l, c, r_) for l, c, r_ in zip(g["Level"], g["Country"], g["Region"])]
        g = g[["Label", "Metric", "Guardrail_Triggered", "Forecast_12m", "Actual_Last_12m", "Momentum_Reference",
               "Lower_Bound", "Upper_Bound", "Ratio_vs_Last_12m", "Scale_Applied"]].sort_values(
            ["Guardrail_Triggered", "Ratio_vs_Last_12m"], ascending=[False, False])
    b.table("Guardrails", g, widths={"Label": 36}, formats={"Ratio_vs_Last_12m": "dec", "Scale_Applied": "dec"},
            note="Plausible 12-month range from last year's level and recent 3-month momentum. Triggered = capped and flagged.")
    b.table("Policy Overlay", res.overlay_rules, widths={"Rationale": 60, "Applied_Status": 40},
            formats={"Multiplier": "dec"},
            note=f"Source: {res.manifest.get('overlay_source', '')}. Only Approved rows with an owner and date are applied.")
    dq = res.dq
    b.table("Data Quality", dq, widths={"Detail": 90, "Issue": 34})
    row_next = len(dq) + 5
    dr = getattr(res, "data_revisions", None)
    if dr is not None and len(dr):
        b.table("Data Quality", dr, start_row=row_next,
                formats={"Typical_Revision_Pct": "pct_plain", "Months_Evidence": ""},
                note="Late entries: how much a month's count grew after it was first reported (from saved extracts).")
        row_next += len(dr) + 5
    ss = res.series_status
    b.table("Data Quality", ss, start_row=row_next,
            formats={"Months_Of_History": "", "Zero_Filled_Months": ""}, note="Status of every Country × Region series")
    ts = res.track_scorecard
    if ts is None or ts.empty:
        ts = pd.DataFrame({"Status": ["No realised forecasts yet — the track record fills in as future months' actuals arrive."]})
    b.table("Track Record", ts, formats={c: "pct_plain" for c in ts.columns if c.endswith("_Pct")},
            widths={"Status": 90, "Coverage80_Status": 18},
            note="Real-world accuracy of earlier runs vs actuals. 80% coverage should sit between 70% and 90%.")
    man = pd.DataFrame(_flatten(res.manifest), columns=["Key", "Value"])
    b.table("Run Manifest", man, widths={"Key": 45, "Value": 100})
    b.close()
    return path


def _chart_block(ws, b, r0, c0, header, rows):
    ws.write_row(r0, c0, header, b.f["hdr"])
    for i, row in enumerate(rows, start=1):
        ws.write_datetime(r0 + i, c0, row[0].to_pydatetime(), b.f["month"])
        for j, v in enumerate(row[1:], start=1):
            if v is not None and np.isfinite(v):
                ws.write_number(r0 + i, c0 + j, float(v), b.f["int"])
    return len(rows)


def _summary_chart(b, ws, res, anchor="K4"):
    # Native Excel line chart: organisation receipts, last 24 months actual + 12-month forecast with 80% range.
    h = res.history[(res.history["Level"] == "Organisation") & (res.history["Metric"] == "Received")].sort_values("Date").tail(24)
    f = res.forecast[(res.forecast["Level"] == "Organisation") & (res.forecast["Metric"] == "Received")].sort_values("Date")
    if h.empty or f.empty:
        return
    c0, r0 = 30, 3
    rows = [(d, a, None, None, None) for d, a in zip(h["Date"], h["Actual"])]
    rows += [(d, None, x, lo, hi) for d, x, lo, hi in zip(f["Date"], f["Forecast"], f["Lo80"], f["Hi80"])]
    n = _chart_block(ws, b, r0, c0, ["Month", "Actual", "Forecast", "Low (80%)", "High (80%)"], rows)
    ch = b.w.book.add_chart({"type": "line"})
    for j, (name, color, width, dash) in enumerate([("Actual", "#52514e", 2.25, None), ("Forecast", "#2a78d6", 2.5, None),
                                                    ("Low (80%)", "#9ec5f0", 1.25, "dash"),
                                                    ("High (80%)", "#9ec5f0", 1.25, "dash")], start=1):
        line = {"color": color, "width": width}
        if dash:
            line["dash_type"] = dash
        ch.add_series({"name": name, "categories": ["Executive Summary", r0 + 1, c0, r0 + n, c0],
                       "values": ["Executive Summary", r0 + 1, c0 + j, r0 + n, c0 + j], "line": line})
    ch.set_title({"name": "Organisation — applications received per month", "name_font": {"size": 11}})
    ch.set_y_axis({"num_format": "#,##0", "name": "Applications",
                   "major_gridlines": {"visible": True, "line": {"color": "#e4e3df"}}})
    ch.set_x_axis({"date_axis": True, "num_format": "mmm yy", "name": "Month"})
    ch.set_legend({"position": "bottom"})
    ch.set_size({"width": 820, "height": 360})
    ws.insert_chart(anchor, ch)


def _staffing_chart(b, ws, st):
    o = st[(st["Level"] == "Organisation") & (st["Metric"] == "Received")]
    if o.empty:
        return
    c0, r0 = 30, 3
    n = _chart_block(ws, b, r0, c0, ["Month", "Forecast", "Plan for (80% high)"],
                     [(r.Date, r.Forecast, r.Capacity_Plan_80) for r in o.itertuples()])
    ch = b.w.book.add_chart({"type": "column"})
    ch.add_series({"name": "Forecast", "categories": ["Staffing Calendar", r0 + 1, c0, r0 + n, c0],
                   "values": ["Staffing Calendar", r0 + 1, c0 + 1, r0 + n, c0 + 1], "fill": {"color": "#2a78d6"},
                   "gap": 60})
    ln = b.w.book.add_chart({"type": "line"})
    ln.add_series({"name": "Plan for (80% high)", "categories": ["Staffing Calendar", r0 + 1, c0, r0 + n, c0],
                   "values": ["Staffing Calendar", r0 + 1, c0 + 2, r0 + n, c0 + 2],
                   "line": {"color": "#eb6834", "width": 2, "dash_type": "dash"}})
    ch.combine(ln)
    ch.set_title({"name": "Organisation receipts — workload plan, next 12 months", "name_font": {"size": 11}})
    ch.set_y_axis({"num_format": "#,##0", "name": "Applications"})
    ch.set_x_axis({"num_format": "mmm yy", "name": "Month"})
    ch.set_legend({"position": "bottom"})
    ch.set_size({"width": 760, "height": 320})
    ws.insert_chart("P3", ch)


def _fva_chart(b, ws, fs):
    o = fs[fs["Metric"] == "Received"].sort_values("Target_Month").tail(12)
    if o.empty or "Forecast_1m_ahead" not in o.columns:
        return
    c0, r0 = 30, 3
    cols = ["Target_Month", "Actual", "Forecast_1m_ahead"] + (["Forecast_3m_ahead"] if "Forecast_3m_ahead" in o.columns else [])
    names = ["Month", "Actual", "Forecast 1 month before", "Forecast 3 months before"][:len(cols)]
    n = _chart_block(ws, b, r0, c0, names, list(o[cols].itertuples(index=False)))
    ch = b.w.book.add_chart({"type": "line"})
    for j, (name, color, dash) in enumerate([("Actual", "#52514e", None), ("Forecast 1 month before", "#2a78d6", None),
                                             ("Forecast 3 months before", "#2a78d6", "dash")][:len(cols) - 1], start=1):
        line = {"color": color, "width": 2}
        if dash:
            line["dash_type"] = dash
        ch.add_series({"name": name, "categories": ["Forecast vs Actual", r0 + 1, c0, r0 + n, c0],
                       "values": ["Forecast vs Actual", r0 + 1, c0 + j, r0 + n, c0 + j], "line": line,
                       "marker": {"type": "circle", "size": 4}})
    ch.set_title({"name": "Organisation receipts — forecast vs actual (last 12 months)", "name_font": {"size": 11}})
    ch.set_y_axis({"num_format": "#,##0", "name": "Applications"})
    ch.set_x_axis({"num_format": "mmm yy", "name": "Month"})
    ch.set_legend({"position": "bottom"})
    ch.set_size({"width": 760, "height": 320})
    ws.insert_chart(len(fs) + 5, 0, ch)


def ev_label(level, country, region) -> str:
    if level == "Organisation":
        return "Organisation"
    if level == "Country":
        return str(country)
    return f"{country} – {region}"


def _flatten(d, prefix="") -> List[tuple]:
    out = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.extend(_flatten(v, key + "."))
        else:
            out.append((key, json.dumps(v, default=str) if isinstance(v, (list, tuple)) else str(v)))
    return out
