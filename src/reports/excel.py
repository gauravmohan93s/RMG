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

CONF_COLORS = {"High": "#d7f0dd", "Medium": "#fff1c7", "Low": "#fbd9d6"}


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
            ws.write(r0, j, c, self.f["hdr"])
            fmt = self.f.get((formats or {}).get(c, ""), None)
            if fmt is None:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    fmt = self.f["month"]
                elif pd.api.types.is_float_dtype(df[c]):
                    fmt = self.f["int"]
            ws.set_column(j, j, (widths or {}).get(c, max(10, min(40, len(str(c)) + 2))), fmt)
        ws.freeze_panes(r0 + 1, 0)
        if len(df):
            ws.autofilter(r0, 0, r0 + len(df), len(df.columns) - 1)
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
        ("Confidence", "High ≤15%, Medium ≤30%, Low >30% average monthly error in the walk-forward test; "
                       "new series and guardrail-capped series are always Low."),
        ("How models were chosen", "Seven candidate methods were each tested on 24 past forecast origins over the full "
                                   "12 months ahead. Ways of combining them were replayed walk-forward (Policy Comparison "
                                   "sheet); the equal-weight average of all methods won and is used. A series falls back to "
                                   "'same month last year' if the average did not beat it (Model Evidence sheet)."),
        ("Sheets", "Executive Summary · Country Outlook · Monthly Forecast · Fiscal Periods (for planning) — "
                   "Model Evidence · Model Scores · Accuracy by Horizon · Policy Comparison · Interval Calibration · "
                   "Model Catalogue · "
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

    # ── planning sheets ──────────────────────────────────────────────────
    ct = ins.country_table(res)
    b.table("Country Outlook", ct, widths={"Country": 28},
            formats={c: "pct_plain" for c in ct.columns if c.endswith("YoY_Pct")},
            note="Next 12 months by country. YoY % compares with the same 12 months last year (all series, incl. closed ones).")

    fc = res.forecast
    mcols = ["Label", "Level", "Country", "Region", "Metric", "Date", "Horizon", "Forecast", "Lo80", "Hi80",
             "Lo95", "Hi95", "Forecast_Adjusted", "Adjusted_Lo80", "Adjusted_Hi80", "Overlay_Multiplier",
             "Confidence", "Method", "Models", "Base_Forecast"]
    b.table("Monthly Forecast", fc[[c for c in mcols if c in fc.columns]],
            widths={"Label": 36, "Method": 34, "Models": 34, "Date": 11},
            formats={"Overlay_Multiplier": "dec", "Horizon": ""}, conf_col="Confidence",
            note="Base_Forecast = before reconciliation. Forecast = reconciled baseline (use this).")
    b.table("Fiscal Periods", res.periods, widths={"Label": 36},
            formats={"YoY_Pct": "pct_plain", "Months_Actual": "", "Months_Forecast": ""},
            note="Fiscal year starts in April. Outlook = actuals already realised + forecast for the remaining months.")

    # ── evidence sheets ─────────────────────────────────────────────────
    meta = res.nodes[["node_id", "Level", "Country", "Region", "Metric", "Months_Of_History"]]
    ev = meta.merge(res.selection, on="node_id", how="right")
    ev["Label"] = [ev_label(l, c, r_) for l, c, r_ in zip(ev["Level"], ev["Country"], ev["Region"])]
    conf = fc.drop_duplicates("node_id").set_index("node_id")["Confidence"]
    ev["Confidence"] = ev["node_id"].map(conf)
    ecols = ["Label", "Metric", "Level", "Method", "Selected_Models", "Confidence", "Selection_Reason", "Best_Model",
             "Runner_Up", "Margin_Best_vs_RunnerUp_Pct", "Best_HB_MAE", "Benchmark_HB_MAE", "Honest_WAPE_Policy",
             "Honest_WAPE_Benchmark", "Honest_Skill_vs_Benchmark", "Honest_N", "N_Candidates", "N_Origins",
             "Required_Train_Months", "Eligible_Models", "Months_Of_History"]
    b.table("Model Evidence", ev[[c for c in ecols if c in ev.columns]],
            widths={"Label": 36, "Selection_Reason": 90, "Method": 34, "Selected_Models": 32, "Eligible_Models": 50},
            formats={"Margin_Best_vs_RunnerUp_Pct": "pct_plain", "Honest_WAPE_Policy": "dec",
                     "Honest_WAPE_Benchmark": "dec", "Honest_Skill_vs_Benchmark": "dec", "Honest_N": "",
                     "N_Candidates": "", "N_Origins": "", "Required_Train_Months": "", "Months_Of_History": ""},
            conf_col="Confidence",
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
                note="Weighted absolute % error (WAPE) of forecasts made at the last 12 origins, by months ahead. "
                     "'Selected method' is the walk-forward replay of the full procedure.")
    pc = res.policy_comparison
    if pc is not None and len(pc):
        b.table("Policy Comparison", pc, widths={"Method": 34},
                formats={"WAPE_Pct": "pct_plain", "Bias_Pct": "pct_plain", "Median_Error_vs_Benchmark": "dec",
                         "Pct_Series_Beating_Benchmark": "pct_plain", "N": ""},
                note="Walk-forward test of each way of forming the forecast (re-run every month). The configured policy "
                     f"is '{ins.policy_label(res)}'. Bias < 0 = actuals above forecast.")
    b.table("Interval Calibration", res.calibration,
            formats={"Factor": "dec", "Coverage_Before_Pct": "pct_plain", "Coverage_After_Pct": "pct_plain",
                     "Target_Pct": "", "N": ""},
            note="Share of walk-forward actuals inside the range. Factor = widening applied so coverage reaches the target.")
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
    ss = res.series_status
    b.table("Data Quality", ss, start_row=len(dq) + 5,
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
