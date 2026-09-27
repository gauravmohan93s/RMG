# -*- coding: utf-8 -*-
"""
Executive PDF brief — written for non-technical readers.

  1 The forecast at a glance      4 Our track record
  2 By destination                5 How reliable is this forecast?
  3 Planning the workload         6 Methodology notes (for analysts) + country appendix
"""
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
from ..forecasting.reliability import label_text
from . import charts
from . import insights as ins

INK, INK2, LINE, HEAD = (colors.HexColor(c) for c in ("#0b0b0b", "#52514e", "#d0d7e2", "#eef2f7"))
REL = {"High": colors.HexColor("#d7f0dd"), "Medium": colors.HexColor("#fff1c7"), "Low": colors.HexColor("#fbd9d6"),
       "New": colors.HexColor("#eceff3"), "Review": colors.HexColor("#fde2c8")}
REL_MEANING = {
    "High": "Quarter totals typically within ±10% of what happens",
    "Medium": "Quarter totals typically within ±10–20%",
    "Low": "Quarter totals often more than ±20% out — usually small or very volatile markets",
    "New": "Too little history to test — shown for information",
    "Review": "Forecast was outside the plausible range and has been capped — check with the market team",
}


def _styles():
    s = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("h1", parent=s["Heading1"], fontSize=18, textColor=INK, spaceAfter=2),
        "h2": ParagraphStyle("h2", parent=s["Heading2"], fontSize=13, textColor=INK, spaceBefore=4, spaceAfter=4),
        "h3": ParagraphStyle("h3", parent=s["Heading3"], fontSize=10, textColor=INK, spaceBefore=2, spaceAfter=3),
        "body": ParagraphStyle("body", parent=s["BodyText"], fontSize=9.5, leading=13, textColor=INK),
        "small": ParagraphStyle("small", parent=s["BodyText"], fontSize=7.8, leading=10.5, textColor=INK2),
        "kpi_v": ParagraphStyle("kv", fontSize=18, leading=21, textColor=INK, fontName="Helvetica-Bold"),
        "kpi_l": ParagraphStyle("kl", fontSize=8, leading=10.5, textColor=INK2),
        "cell": ParagraphStyle("cell", fontSize=7.8, leading=9.8, textColor=INK),
    }


def _img(png: bytes, w_cm: float) -> Image:
    im = Image(io.BytesIO(png))
    ratio = im.imageHeight / im.imageWidth
    im.drawWidth, im.drawHeight = w_cm * cm, w_cm * cm * ratio
    return im


def _grid_style(n_rows: int):
    return [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.8), ("FONT", (0, 1), (-1, -1), "Helvetica", 7.8),
            ("BACKGROUND", (0, 0), (-1, 0), HEAD), ("LINEBELOW", (0, 0), (-1, 0), 0.6, LINE),
            ("LINEBELOW", (0, 1), (-1, -1), 0.25, LINE), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]


def _kpi(label: str, value: str, sub: str, S, width=6.1) -> Table:
    t = Table([[Paragraph(label, S["kpi_l"])], [Paragraph(value, S["kpi_v"])], [Paragraph(sub, S["kpi_l"])]],
              colWidths=[width * cm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, LINE), ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 3)]))
    return t


def _row(cells, widths):
    t = Table([cells], colWidths=[w * cm for w in widths])
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def _node(fc, hist, level, country, metric):
    f = fc[(fc["Level"] == level) & (fc["Country"] == country) & (fc["Region"] == TOTAL) & (fc["Metric"] == metric)]
    h = hist[(hist["Level"] == level) & (hist["Country"] == country) & (hist["Region"] == TOTAL)
             & (hist["Metric"] == metric)]
    return h, f


def _k(v, step=1000):
    return ins._round(v, step)


def write_brief(res, path: Path) -> Path:
    S = _styles()
    doc = SimpleDocTemplate(str(path), pagesize=landscape(A4), leftMargin=1.4 * cm, rightMargin=1.4 * cm,
                            topMargin=1.1 * cm, bottomMargin=1.1 * cm,
                            title="Application Flow Forecast — Executive Brief", author="RMG forecasting pipeline")
    story: List = []
    k = ins.kpis(res)
    r12, s12, r3 = k["Received"]["Next 12 months"], k["Submitted"]["Next 12 months"], k["Received"]["Next 3 months"]
    org = res.forecast[(res.forecast["Level"] == "Organisation") & (res.forecast["Metric"] == "Received")]
    org_rel = org["Reliability"].iloc[0] if len(org) and "Reliability" in org else ""
    org_err = float(org["Typical_Quarter_Error_Pct"].iloc[0]) if len(org) and "Typical_Quarter_Error_Pct" in org else np.nan

    # ── 1. the forecast at a glance ─────────────────────────────────────
    story += [Paragraph("Application Flow Forecast", S["h1"]),
              Paragraph(f"Based on data up to {res.last_actual:%B %Y} · covers {r12['From']:%B %Y} to {r12['To']:%B %Y} · "
                        f"version {__version__}", S["small"]), Spacer(0, 8)]
    tiles = [
        _kpi("Applications received — next 12 months", _k(r12["Forecast"], 100),
             f"most likely {_k(r12.get('Lo50'))}–{_k(r12.get('Hi50'))} · plan for {_k(r12.get('Lo80'))}–{_k(r12.get('Hi80'))}"
             f"<br/>{ins.fmt_pct(r12['YoY_Pct'])} vs last year", S),
        _kpi(f"Received — next quarter ({r3['From']:%b}–{r3['To']:%b})", _k(r3["Forecast"], 100),
             f"most likely {_k(r3.get('Lo50'))}–{_k(r3.get('Hi50'))} · plan for {_k(r3.get('Lo80'))}–{_k(r3.get('Hi80'))}"
             f"<br/>{ins.fmt_pct(r3['YoY_Pct'])} vs last year", S),
        _kpi("Applications submitted — next 12 months", _k(s12["Forecast"], 100),
             f"most likely {_k(s12.get('Lo50'))}–{_k(s12.get('Hi50'))} · plan for {_k(s12.get('Lo80'))}–{_k(s12.get('Hi80'))}"
             f"<br/>{ins.fmt_pct(s12['YoY_Pct'])} vs last year", S),
        _kpi("How reliable?", f"{org_rel}" if org_rel else "–",
             (f"organisation quarter totals were typically within ±{org_err:.0f}% in testing"
              if np.isfinite(org_err) else ""), S),
    ]
    story += [_row(tiles, [6.55] * 4), Spacer(0, 8)]
    imgs = []
    for m in ("Received", "Submitted"):
        h, f = _node(res.forecast, res.history, "Organisation", ORG, m)
        imgs.append(_img(charts.history_forecast(h, f, m, f"Applications {m.lower()} per month — actual and forecast",
                                                size=(6.2, 1.95)), 12.9))
    story += [_row(imgs, [13.3, 13.3]), Spacer(0, 4), Paragraph("What this means", S["h2"])]
    story += [Paragraph("• " + m, S["body"]) for m in ins.messages(res)]
    story.append(PageBreak())

    # ── 2. by destination ───────────────────────────────────────────────
    story.append(Paragraph("By destination — next 12 months", S["h2"]))
    ct = ins.country_table(res)
    data = [["Destination", "Received", "Planning range", "vs last year", "Next quarter", "Submitted", "vs last year",
             "Reliability"]]
    sty = _grid_style(len(ct) + 1)
    for i, r in enumerate(ct.itertuples(), start=1):
        rel = getattr(r, "Received_Reliability", "")
        err = getattr(r, "Received_Quarter_Error_Pct", np.nan)
        data.append([r.Country, _k(r.Received_Forecast, 100), f"{_k(r.Received_Lo80, 100)}–{_k(r.Received_Hi80, 100)}",
                     ins.fmt_pct(r.Received_YoY_Pct), _k(getattr(r, "Received_Next_Quarter", np.nan), 100),
                     _k(getattr(r, "Submitted_Forecast", np.nan), 100), ins.fmt_pct(getattr(r, "Submitted_YoY_Pct", np.nan)),
                     Paragraph(label_text(rel, err), S["cell"])])
        if r.Level == "Organisation":
            sty.append(("FONT", (0, i), (-2, i), "Helvetica-Bold", 7.8))
        if rel in REL:
            sty.append(("BACKGROUND", (7, i), (7, i), REL[rel]))
    tbl = Table(data, colWidths=[4.6 * cm, 2.3 * cm, 3.2 * cm, 2.2 * cm, 2.4 * cm, 2.3 * cm, 2.2 * cm, 7.4 * cm],
                repeatRows=1)
    tbl.setStyle(TableStyle(sty))
    story += [tbl, Spacer(0, 6), Paragraph(
        "<b>Reading this table.</b> 'Planning range' is where we expect the actual 12-month total to land 8 times out of "
        "10 (it was tested that way on past forecasts). "
        "Every destination uses the forecasting method that proved most accurate for it; <b>reliability</b> tells you "
        "how precisely that destination can be forecast at all, measured on past quarters. Small or fast-changing "
        "markets move proportionally more, so plan them on the range rather than the single number. "
        "Region-level detail is in the Excel workbook.", S["small"]), PageBreak()]

    # ── 3. planning the workload ────────────────────────────────────────
    sidx = getattr(res, "seasonal_index", pd.DataFrame())
    syear = getattr(res, "seasonal_by_year", pd.DataFrame())
    staff = getattr(res, "staffing", pd.DataFrame())
    if sidx is not None and len(sidx):
        so = sidx[sidx["Level"] == "Organisation"]
        yo = syear[syear["Level"] == "Organisation"] if len(syear) else syear
        story.append(Paragraph("Planning the workload — busy and quiet months", S["h2"]))
        story.append(_row([_img(charts.seasonal_profile(so, "Received", "Applications received: a typical month compared with average"), 12.9),
                           _img(charts.seasonal_profile(so, "Submitted", "Applications submitted: a typical month compared with average"), 12.9)],
                          [13.3, 13.3]))
        left = (_img(charts.seasonal_heatmap(yo, "Received", "Does the pattern repeat every year? (1.00 = normal level)",
                                             size=(5.6, 2.6)), 12.6) if len(yo) else "")
        right = ""
        if len(staff):
            so_st = staff[(staff["Level"] == "Organisation") & (staff["Metric"] == "Received")]
            has_fte = "FTE_Base" in so_st.columns and so_st["FTE_Base"].notna().any()
            hdr = ["Month", "Expected", "Plan for", "vs average", "Workload"] + (["Staff (FTE)"] if has_fte else [])
            data = [hdr]
            sty = _grid_style(len(so_st) + 1)
            for i, r in enumerate(so_st.itertuples(), start=1):
                row = [f"{r.Date:%b %Y}", _k(r.Forecast, 100), _k(r.Capacity_Plan_80, 100),
                       f"{100 * (r.Load_Index - 1):+.0f}%", r.Workload_Class]
                if has_fte:
                    row.append(f"{r.FTE_Base:,.0f}–{r.FTE_Capacity_80:,.0f}" if np.isfinite(r.FTE_Base) else "–")
                data.append(row)
                col = {"High": colors.HexColor("#cfe1f7"), "Low": colors.HexColor("#eef2f7")}.get(r.Workload_Class)
                if col is not None:
                    sty.append(("BACKGROUND", (4, i), (4, i), col))
            tb = Table(data, colWidths=[2.2 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 1.9 * cm] + ([2.6 * cm] if has_fte else []),
                       repeatRows=1)
            tb.setStyle(TableStyle(sty))
            right = [Paragraph("Staffing calendar — applications received, next 12 months", S["h3"]), tb]
        story += [_row([left, right], [13.0, 13.6]), Paragraph(
            "High / Low month = at least 10% busier / quieter than an average month. 'Plan for' is the upper end of the "
            "likely range — use it for capacity. Staff (FTE) appears once applications-per-counsellor rates are entered "
            "per country and region (Settings.xlsx, sheet Staffing_Productivity); country and region calendars are in "
            "the workbook.", S["small"]), PageBreak()]

    # ── 4. track record ─────────────────────────────────────────────────
    fsum = getattr(res, "fva_summary", pd.DataFrame())
    if fsum is not None and len(fsum):
        story.append(Paragraph("Our track record — what we forecast and what actually happened", S["h2"]))
        story.append(_row([_img(charts.forecast_vs_actual(fsum, m, f"Organisation — applications {m.lower()}"), 12.9)
                           for m in ("Received", "Submitted")], [13.3, 13.3]))
        story.append(Paragraph(
            "The lines show forecasts produced by today's method using only the information that was available at the "
            "time, one and three months before each month. From now on every forecast we publish is stored and compared "
            "with the actual figure as it arrives (workbook sheet 'Forecast vs Actual').", S["small"]))
        bt_ = getattr(res, "bias_test", pd.DataFrame())
        if bt_ is not None and len(bt_):
            mode = res.manifest.get("statistical_overlay_mode", "none")
            story.append(Paragraph(
                "Every month we also test whether adjusting the forecast for the most recent trend would have made it "
                "more accurate. " + ("It would, so the adjustment is applied." if mode != "none" else
                                     "This month it would not, so no adjustment is applied."), S["body"]))
        story.append(PageBreak())

    # ── 5. how reliable is this forecast? ───────────────────────────────
    story.append(Paragraph("How reliable is this forecast?", S["h2"]))
    rs = ins.reliability_summary(res)
    if len(rs):
        data = [["Reliability", "What it means", "Regions", "Share of forecast volume"]]
        sty = _grid_style(len(rs) + 1)
        for i, r in enumerate(rs.itertuples(), start=1):
            data.append([r.Reliability, Paragraph(REL_MEANING.get(r.Reliability, ""), S["cell"]), str(int(r.Series)),
                         f"{r.Volume_Share_Pct:.0f}%"])
            if r.Reliability in REL:
                sty.append(("BACKGROUND", (0, i), (0, i), REL[r.Reliability]))
        t = Table(data, colWidths=[2.6 * cm, 12 * cm, 2.2 * cm, 4 * cm], repeatRows=1)
        t.setStyle(TableStyle(sty))
        story += [t, Spacer(0, 6)]
    acc = ins.accuracy_table(res, "Received")
    if len(acc):
        o = acc[acc["Level"] == "Organisation"].set_index("Months_Ahead")
        c = acc[acc["Level"] == "Country"].set_index("Months_Ahead")
        rows = [["Looking ahead", "Organisation — typical monthly error", "'Same as last year' would have been",
                 "Countries — typical monthly error"]]
        for b, lab in (("1-3", "1–3 months"), ("4-6", "4–6 months"), ("7-12", "7–12 months")):
            if b in o.index:
                rows.append([lab, f"±{o.loc[b, 'WAPE_Pct']:.0f}%", f"±{o.loc[b, 'Benchmark_WAPE_Pct']:.0f}%",
                             f"±{c.loc[b, 'WAPE_Pct']:.0f}%" if b in c.index else "–"])
        t = Table(rows, colWidths=[3.2 * cm, 6 * cm, 6 * cm, 6 * cm])
        t.setStyle(TableStyle(_grid_style(len(rows))))
        story += [Paragraph("Tested on the past year (forecasts made as if at the time)", S["h3"]), t, Spacer(0, 6)]
    story += [Paragraph("What we do to keep the forecast dependable", S["h3"])] + [Paragraph("• " + x, S["body"]) for x in [
        "Seventeen forecasting methods are re-tested every month on the past two years; the combination that proved "
        "most accurate is used — nothing is chosen by opinion.",
        "Regional forecasts are added up to give country and organisation totals, which tested markedly more accurate "
        "than forecasting totals directly.",
        "Each 'likely range' is based on the errors we actually made in the past, and the way it is built was itself tested: it is as narrow as the evidence allows while still containing the actual about 8 times in 10.",
        "Forecasts that fall outside what recent history makes plausible are capped and flagged for review.",
        "Every published forecast is recorded and compared with what happens, so accuracy is tracked month by month.",
        "Known one-off events (for example a visa policy change) can be measured from the data and, once approved by "
        "the market owner, applied as a documented adjustment — never as a guess."]]
    story.append(PageBreak())

    # ── 6. methodology notes (analysts) ─────────────────────────────────
    story.append(Paragraph("Methodology notes (for analysts)", S["h2"]))
    story.append(_row([_img(charts.accuracy_by_horizon(res.accuracy, m), 12.9) for m in ("Received", "Submitted")],
                      [13.3, 13.3]))
    method_counts = res.selection[res.selection["node_id"].str.startswith("Received")]["Method"].value_counts()
    story += [Paragraph(
        "Rolling-origin backtest (24 origins × 12-month horizon, no information after each origin) of 17 models; "
        "combination policy chosen walk-forward (equal-weight mean of all models); benchmark fallback where the "
        f"combination loses to seasonal naive; hierarchy reconciled with '{res.manifest.get('reconciliation_method_chosen', '')}' "
        "(chosen walk-forward); ranges = past errors of the published forecasts turned into a range by the method "
        "(centred robust spread or empirical quantiles) with the better interval score when replayed origin by origin, "
        "with a symmetric width factor (0.8–1.5) where coverage was off target; reliability = median absolute % error of next-3-month totals in the walk-forward test. "
        "Methods in use (receipts series): " + "; ".join(f"{k_} {v}" for k_, v in method_counts.items()) + ". "
        f"Code {res.manifest.get('git', {}).get('describe', '')}; data SHA-256 "
        f"{str(res.manifest.get('data', {}).get('sha256', ''))[:16]}…; full evidence in the workbook.", S["small"]),
        PageBreak()]

    # ── appendix: country charts ────────────────────────────────────────
    story.append(Paragraph("Appendix — monthly outlook by destination (applications received, likely range shaded)", S["h2"]))
    cells = []
    fc, hist = res.forecast, res.history
    for c in [c for c in ct["Country"] if c != "Organisation"]:
        f = fc[(fc["Level"] == "Country") & (fc["Label"] == c) & (fc["Metric"] == "Received")]
        if f.empty:
            continue
        h = hist[(hist["Level"] == "Country") & (hist["Country"] == f["Country"].iloc[0]) & (hist["Metric"] == "Received")]
        cells.append(_img(charts.history_forecast(h, f, "Received", c, months_back=24, size=(4.2, 2.1)), 8.5))
    for i in range(0, len(cells), 3):
        story.append(KeepTogether(_row(cells[i:i + 3] + [""] * (3 - len(cells[i:i + 3])), [8.8] * 3)))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return path


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(INK2)
    canvas.drawString(1.4 * cm, 0.6 * cm, "Data-driven forecast. Full detail and evidence: RMG_Forecast workbook.")
    canvas.drawRightString(landscape(A4)[0] - 1.4 * cm, 0.6 * cm, f"Page {doc.page}")
    canvas.restoreState()
