# -*- coding: utf-8 -*-
"""Executive PDF brief: 3 pages of decisions-first content + a country appendix."""
from __future__ import annotations

import io
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from .. import __version__
from ..data.prepare import ORG, TOTAL
from . import charts
from . import insights as ins

INK, INK2, LINE, HEAD = (colors.HexColor(c) for c in ("#0b0b0b", "#52514e", "#d0d7e2", "#eef2f7"))
CONF = {"High": colors.HexColor("#d7f0dd"), "Medium": colors.HexColor("#fff1c7"), "Low": colors.HexColor("#fbd9d6")}


def _styles():
    s = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("h1", parent=s["Heading1"], fontSize=18, textColor=INK, spaceAfter=2),
        "h2": ParagraphStyle("h2", parent=s["Heading2"], fontSize=12, textColor=INK, spaceBefore=6, spaceAfter=4),
        "body": ParagraphStyle("body", parent=s["BodyText"], fontSize=9, leading=12.5, textColor=INK),
        "small": ParagraphStyle("small", parent=s["BodyText"], fontSize=7.5, leading=10, textColor=INK2),
        "kpi_v": ParagraphStyle("kv", fontSize=17, leading=20, textColor=INK, fontName="Helvetica-Bold"),
        "kpi_l": ParagraphStyle("kl", fontSize=7.5, leading=10, textColor=INK2),
        "cell": ParagraphStyle("cell", fontSize=7.5, leading=9.5, textColor=INK),
    }


def _img(png: bytes, w_cm: float) -> Image:
    im = Image(io.BytesIO(png))
    ratio = im.imageHeight / im.imageWidth
    im.drawWidth, im.drawHeight = w_cm * cm, w_cm * cm * ratio
    return im


def _grid_style(n_rows: int, zebra=True):
    st = [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5), ("FONT", (0, 1), (-1, -1), "Helvetica", 7.5),
          ("BACKGROUND", (0, 0), (-1, 0), HEAD), ("LINEBELOW", (0, 0), (-1, 0), 0.6, LINE),
          ("LINEBELOW", (0, 1), (-1, -1), 0.25, LINE), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    return st


def _kpi(label: str, value: str, sub: str, S) -> Table:
    t = Table([[Paragraph(label, S["kpi_l"])], [Paragraph(value, S["kpi_v"])], [Paragraph(sub, S["kpi_l"])]],
              colWidths=[6.1 * cm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, LINE), ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 3)]))
    return t


def _node(fc: pd.DataFrame, hist: pd.DataFrame, level: str, country: str, metric: str):
    region = TOTAL
    f = fc[(fc["Level"] == level) & (fc["Country"] == country) & (fc["Region"] == region) & (fc["Metric"] == metric)]
    h = hist[(hist["Level"] == level) & (hist["Country"] == country) & (hist["Region"] == region)
             & (hist["Metric"] == metric)]
    return h, f


def write_brief(res, path: Path) -> Path:
    S = _styles()
    doc = SimpleDocTemplate(str(path), pagesize=landscape(A4), leftMargin=1.4 * cm, rightMargin=1.4 * cm,
                            topMargin=1.1 * cm, bottomMargin=1.1 * cm,
                            title="Application Flow Forecast — Executive Brief", author="RMG forecasting pipeline")
    story: List = []
    k = ins.kpis(res)
    conv = ins.conversion(res)
    r12, s12, r3 = k["Received"]["Next 12 months"], k["Submitted"]["Next 12 months"], k["Received"]["Next 3 months"]

    # ── page 1: the answer ──────────────────────────────────────────────
    story += [Paragraph("Application Flow Forecast — Executive Brief", S["h1"]),
              Paragraph(f"Data through {res.last_actual:%B %Y} (26th–25th cycle) · forecast "
                        f"{r12['From']:%b %Y}–{r12['To']:%b %Y} · run {res.run_id} · pipeline v{__version__}", S["small"]),
              Spacer(0, 8)]
    tiles = [
        _kpi("Applications received · next 12 months", ins.fmt_int(r12["Forecast"]),
             f"80% range {ins.fmt_int(r12.get('Lo80'))}–{ins.fmt_int(r12.get('Hi80'))} · {ins.fmt_pct(r12['YoY_Pct'])} YoY", S),
        _kpi("Applications submitted · next 12 months", ins.fmt_int(s12["Forecast"]),
             f"80% range {ins.fmt_int(s12.get('Lo80'))}–{ins.fmt_int(s12.get('Hi80'))} · {ins.fmt_pct(s12['YoY_Pct'])} YoY", S),
        _kpi("Received · next 3 months", ins.fmt_int(r3["Forecast"]),
             f"80% range {ins.fmt_int(r3.get('Lo80'))}–{ins.fmt_int(r3.get('Hi80'))} · {ins.fmt_pct(r3['YoY_Pct'])} YoY", S),
        _kpi("Submitted ÷ received (next 12 months)", ins.fmt_pct(conv["forecast"], False),
             f"same months last year {ins.fmt_pct(conv['last_year'], False)}", S),
    ]
    row = Table([tiles], colWidths=[6.55 * cm] * 4)
    row.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [row, Spacer(0, 8)]
    imgs = []
    for m in ("Received", "Submitted"):
        h, f = _node(res.forecast, res.history, "Organisation", ORG, m)
        imgs.append(_img(charts.history_forecast(h, f, m, f"Organisation — applications {m.lower()} per month"), 12.9))
    t = Table([imgs], colWidths=[13.3 * cm, 13.3 * cm])
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [t, Spacer(0, 4), Paragraph("Key messages", S["h2"])]
    story += [Paragraph("• " + m, S["body"]) for m in ins.messages(res)]
    story.append(PageBreak())

    # ── page 2: where ───────────────────────────────────────────────────
    story.append(Paragraph("Outlook by destination country — next 12 months", S["h2"]))
    ct = ins.country_table(res)
    data = [["Country", "Received", "80% range", "vs last year", "Submitted", "80% range", "vs last year", "Confidence"]]
    styles = _grid_style(len(ct) + 1)
    for i, r in enumerate(ct.itertuples(), start=1):
        data.append([r.Country, ins.fmt_int(r.Received_Forecast),
                     f"{ins.fmt_int(r.Received_Lo80)}–{ins.fmt_int(r.Received_Hi80)}", ins.fmt_pct(r.Received_YoY_Pct),
                     ins.fmt_int(getattr(r, "Submitted_Forecast", np.nan)),
                     f"{ins.fmt_int(getattr(r, 'Submitted_Lo80', np.nan))}–{ins.fmt_int(getattr(r, 'Submitted_Hi80', np.nan))}",
                     ins.fmt_pct(getattr(r, "Submitted_YoY_Pct", np.nan)), r.Received_Confidence])
        if r.Level == "Organisation":
            styles.append(("FONT", (0, i), (-1, i), "Helvetica-Bold", 7.5))
        if r.Received_Confidence in CONF:
            styles.append(("BACKGROUND", (7, i), (7, i), CONF[r.Received_Confidence]))
    tbl = Table(data, colWidths=[5.2 * cm, 2.6 * cm, 3.6 * cm, 2.5 * cm, 2.6 * cm, 3.6 * cm, 2.5 * cm, 2.4 * cm],
                repeatRows=1)
    tbl.setStyle(TableStyle(styles))
    story += [tbl, Spacer(0, 6),
              Paragraph("Confidence reflects the average monthly error of this method in walk-forward testing "
                        "(High ≤15%, Medium ≤30%, Low >30%). Country totals include every region; the full "
                        "region-level detail is in the Excel workbook.", S["small"]),
              PageBreak()]

    # ── page: seasonality & staffing ────────────────────────────────────
    sidx = getattr(res, "seasonal_index", pd.DataFrame())
    syear = getattr(res, "seasonal_by_year", pd.DataFrame())
    staff = getattr(res, "staffing", pd.DataFrame())
    if sidx is not None and len(sidx):
        so = sidx[sidx["Level"] == "Organisation"]
        yo = syear[syear["Level"] == "Organisation"] if len(syear) else syear
        story.append(Paragraph("Seasonality and staffing — when volume is high, medium and low", S["h2"]))
        imgs = [_img(charts.seasonal_profile(so, "Received", "Applications received: typical month vs average"), 12.9),
                _img(charts.seasonal_profile(so, "Submitted", "Applications submitted: typical month vs average"), 12.9)]
        t = Table([imgs], colWidths=[13.3 * cm, 13.3 * cm])
        t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        story.append(t)
        left = (_img(charts.seasonal_heatmap(yo, "Received", "Is the pattern stable? Receipts vs trend, by year "
                                             "(1.00 = on trend)", size=(5.6, 2.6)), 12.6) if len(yo) else "")
        right = ""
        if len(staff):
            so_st = staff[(staff["Level"] == "Organisation") & (staff["Metric"] == "Received")]
            data = [["Month", "Forecast", "Plan for (80% high)", "vs avg month", "Workload", "Usual"]]
            sty = _grid_style(len(so_st) + 1)
            for i, r in enumerate(so_st.itertuples(), start=1):
                data.append([f"{r.Date:%b %Y}", ins.fmt_int(r.Forecast), ins.fmt_int(r.Capacity_Plan_80),
                             f"{100 * (r.Load_Index - 1):+.0f}%", r.Workload_Class, r.Historical_Class])
                col = {"High": colors.HexColor("#cfe1f7"), "Low": colors.HexColor("#eef2f7")}.get(r.Workload_Class)
                if col is not None:
                    sty.append(("BACKGROUND", (4, i), (4, i), col))
            sty.append(("FONT", (0, 1), (-1, -1), "Helvetica", 7))
            tb = Table(data, colWidths=[2.1 * cm, 1.9 * cm, 2.7 * cm, 2.1 * cm, 1.8 * cm, 1.7 * cm], repeatRows=1)
            tb.setStyle(TableStyle(sty))
            right = [Paragraph("Staffing calendar — organisation receipts, next 12 months", S["h3"] if "h3" in S else S["body"]),
                     tb]
        two = Table([[left, right]], colWidths=[13.0 * cm, 13.6 * cm])
        two.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story += [two, Paragraph("High / Low = at least 10% above / below an average month. Plan capacity for the 80% "
                                 "high figure; 'Usual' is the historical class of that calendar month. Country calendars "
                                 "are in the workbook (Staffing Calendar sheet).", S["small"])]
        story.append(PageBreak())

    # ── page: forecast vs actual ────────────────────────────────────────
    fsum = getattr(res, "fva_summary", pd.DataFrame())
    if fsum is not None and len(fsum):
        story.append(Paragraph("What we forecast vs what happened", S["h2"]))
        imgs = [_img(charts.forecast_vs_actual(fsum, m, f"Organisation — applications {m.lower()}"), 12.9)
                for m in ("Received", "Submitted")]
        t = Table([imgs], colWidths=[13.3 * cm, 13.3 * cm])
        t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        story += [t, Paragraph(
            "Forecasts shown were produced by the current method using only the data available at the time "
            "(walk-forward test). From now on every live forecast is stored, and the Forecast vs Actual sheet adds "
            "them automatically as actuals arrive — the running record of how good the forecasts are.", S["small"])]
        bt_ = getattr(res, "bias_test", pd.DataFrame())
        if bt_ is not None and len(bt_):
            mode = res.manifest.get("statistical_overlay_mode", "none")
            base = bt_[bt_["Mode"] == "none"].iloc[0]
            best = bt_[bt_["Mode"] != "none"].sort_values("WAPE_Pct").iloc[0]
            verdict = (f"adopted ('{mode}')." if mode != "none" else
                       "not applied — it made accuracy worse." if best.WAPE_Pct >= base.WAPE_Pct else
                       "not applied — the improvement was below the 1% threshold.")
            txt = (f"Statistical overlay (recent-bias correction): walk-forward error {base.WAPE_Pct:.1f}% without, "
                   f"{best.WAPE_Pct:.1f}% with the best ('{best.Mode}') correction — " + verdict)
            story.append(Paragraph(txt, S["body"]))
        story.append(PageBreak())

    # ── page 3: how far to trust it ─────────────────────────────────────
    story.append(Paragraph("How far can these numbers be trusted?", S["h2"]))
    imgs = [_img(charts.accuracy_by_horizon(res.accuracy, m), 12.9) for m in ("Received", "Submitted")]
    t = Table([imgs], colWidths=[13.3 * cm, 13.3 * cm])
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(t)
    acc_rows = [["Level", "Months ahead", "Receipts error", "Same month LY", "Receipts bias",
                 "Submissions error", "Same month LY", "Submissions bias", "Tests (N)"]]
    ar, asub = ins.accuracy_table(res, "Received"), ins.accuracy_table(res, "Submitted")
    if len(ar):
        m = ar.merge(asub, on=["Level", "Months_Ahead"], suffixes=("_r", "_s"))
        for r in m.itertuples():
            acc_rows.append([r.Level, r.Months_Ahead, f"{r.WAPE_Pct_r:.0f}%", f"{r.Benchmark_WAPE_Pct_r:.0f}%",
                             ins.fmt_pct(r.Bias_Pct_r), f"{r.WAPE_Pct_s:.0f}%", f"{r.Benchmark_WAPE_Pct_s:.0f}%",
                             ins.fmt_pct(r.Bias_Pct_s), f"{int(r.N_r):,}"])
        at = Table(acc_rows, colWidths=[3.2 * cm, 2.4 * cm] + [2.7 * cm] * 6 + [2.2 * cm], repeatRows=1)
        at.setStyle(TableStyle(_grid_style(len(acc_rows))))
        story += [Spacer(0, 4), at,
                  Paragraph("Error = weighted average absolute % error per month in walk-forward tests (lower is better); "
                            "bias < 0 means actuals came in above the forecast.", S["small"])]
    cal = res.calibration
    if len(cal):
        c80 = cal[cal["Interval"] == "80%"]
        parts = []
        for m, g in c80.groupby("Metric"):
            before = np.average(g["Coverage_Before_Pct"], weights=g["N"])
            after = np.average(g["Coverage_After_Pct"], weights=g["N"])
            parts.append(f"{m} {before:.0f}% before → {after:.0f}% after calibration")
        cal_txt = ("Monthly 80% ranges, tested on the last 12 months with ranges built only from earlier errors, "
                   "contained the actual: " + "; ".join(parts) + " (target 80%). The widening is mostly on the "
                   "upside, because recent actuals ran above forecasts.")
    else:
        cal_txt = ""
    guard = int(res.guardrails["Guardrail_Triggered"].sum()) if len(res.guardrails) else 0
    method_counts = res.selection[res.selection["node_id"].str.startswith("Received")]["Method"].value_counts()
    story += [
        Paragraph("Method, in one paragraph", S["h2"]),
        Paragraph(
            "Seven forecasting methods — from 'same month last year' to exponential smoothing, Theta, SARIMA, an "
            "intermittent-demand model and a machine-learning model trained across all series — were each re-run as if "
            "standing at 24 different past month-ends, forecasting the following 12 months every time. Several ways of "
            "turning these into one forecast (picking the best model, averaging the best three, averaging all) were "
            "replayed month by month using only information available at the time; the equal-weight average of all "
            "methods was the most accurate and is used, and it is re-tested every run. Series where it does not beat "
            "'same month last year' fall back to that benchmark. Ranges come from the errors actually made in these "
            "tests, widened until they achieved their stated coverage. Regional forecasts are reconciled so they add "
            "up to country and organisation totals.", S["body"]),
        Spacer(0, 4), Paragraph(cal_txt, S["body"]),
        Paragraph("Methods in use (receipts series): " + "; ".join(f"{k_} {v}" for k_, v in method_counts.items())
                  + f". Guardrail-capped series: {guard}.", S["body"]),
        Paragraph(f"Reproducibility: code {res.manifest.get('git', {}).get('describe', '')}, data file SHA-256 "
                  f"{str(res.manifest.get('data', {}).get('sha256', ''))[:16]}…, settings in the Run Manifest sheet.",
                  S["small"]),
        PageBreak(),
    ]

    # ── appendix: country charts ────────────────────────────────────────
    story.append(Paragraph("Appendix — monthly outlook by country (applications received, 80% range)", S["h2"]))
    countries = [c for c in ct["Country"] if c != "Organisation"]
    cells = []
    fc, hist = res.forecast, res.history
    for c in countries:
        f = fc[(fc["Level"] == "Country") & (fc["Label"] == c) & (fc["Metric"] == "Received")]
        if f.empty:
            continue
        h = hist[(hist["Level"] == "Country") & (hist["Country"] == f["Country"].iloc[0]) & (hist["Metric"] == "Received")]
        cells.append(_img(charts.history_forecast(h, f, "Received", c, months_back=24, size=(4.2, 2.1)), 8.5))
    for i in range(0, len(cells), 3):
        row = cells[i:i + 3] + [""] * (3 - len(cells[i:i + 3]))
        t = Table([row], colWidths=[8.8 * cm] * 3)
        t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        story.append(KeepTogether(t))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return path


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(INK2)
    canvas.drawString(1.4 * cm, 0.6 * cm, "Statistical baseline unless marked 'Adjusted'. Evidence: RMG_Forecast workbook.")
    canvas.drawRightString(landscape(A4)[0] - 1.4 * cm, 0.6 * cm, f"Page {doc.page}")
    canvas.restoreState()
