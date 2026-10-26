# -*- coding: utf-8 -*-
"""
The monthly pack as a story, read top to bottom (v2.5). Every page is full; every chapter opens with a band
(number, title, the one sentence that matters) and the same flow repeats for each destination, going down to
its regions / teams.

  Organisation
    1  Where we are heading      tiles, key messages, received + submitted, the next 12 months as numbers
    2  Last month & accuracy     last month vs forecast, proof on the last quarter, direction check, track record
    3  Where growth comes from   destinations: next 12 months, growth vs the market now, table
    4  Pipeline & intakes        funnel after submission, stage table; how the coming intakes are filling
    5  Capacity                  workload calendar (and FTE needed once teams enter headcount)
    6  Can we trust it           health check, what the method uses and why, reliability
  Destinations (2 pages each, also saved as one PDF per destination)
    A  Where it is heading       tiles, key messages, received + submitted, months
    B  Regions, pipeline, intakes  region table and small charts, funnel, intake curve, track record
  How to read this pack
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, Spacer, Table, TableStyle

from .. import __version__
from ..data.prepare import ORG, TOTAL
from ..forecasting.stages import MIN_YOY_BASE
from . import charts
from . import insights as ins
from .pdf import (FULL_W, HEAD, INK, INK2, LINE, MISS_BG, REL_BG, STAGE_LABEL, STAGE_ORDER, _countries, _critical_banner,
                  _destination_table, _doc, _footer, _health_table, _img, _miss_bg, _month_strip, _n, _page_legend,
                  _pct, _row, _series, _styles, _tested_coverage, _tile, _tiles, _tot)

NAVY = colors.HexColor("#1f3864")
SOFT = colors.HexColor("#f3f6fb")
HALF = FULL_W / 2 - 0.15
THIRD = FULL_W / 3 - 0.2
PACE_BG = [(-10, "#f5ad7c"), (-5, "#fbd9bf"), (5, "#ffffff"), (10, "#cfe1f7"), (1e9, "#8fbbe9")]


def _s():
    S = _styles()
    S["band_t"] = ParagraphStyle("bt", fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=INK)
    S["band_s"] = ParagraphStyle("bs", fontName="Helvetica", fontSize=8.6, leading=11, textColor=INK2)
    S["no"] = ParagraphStyle("no", fontName="Helvetica-Bold", fontSize=17, leading=20, textColor=colors.white,
                             alignment=1)
    S["msg"] = ParagraphStyle("msg", fontName="Helvetica", fontSize=8.2, leading=10.6, textColor=INK, leftIndent=8,
                              bulletIndent=0)
    S["note"] = ParagraphStyle("note", fontName="Helvetica", fontSize=6.9, leading=8.6, textColor=INK2)
    return S


# ───────────────────────────── page furniture ─────────────────────────────
def _band(no: str, title: str, sub: str, res, S) -> Table:
    """Chapter band: number block, title and the sentence that matters, run details on the right."""
    fr, to = res.forecast["Date"].min(), res.forecast["Date"].max()
    meta = (f"Actuals to {res.last_actual:%b %Y}<br/>Forecast {fr:%b %Y} – {to:%b %Y}<br/>"
            f"Run {res.run_date} · v{__version__}")
    t = Table([[Paragraph(no, S["no"]), [Paragraph(title, S["band_t"]), Paragraph(sub, S["band_s"])],
                Paragraph(meta, S["meta"])]], colWidths=[1.3 * cm, (FULL_W - 6.1) * cm, 4.8 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("LINEBELOW", (0, 0), (-1, 0), 1.4, NAVY), ("LEFTPADDING", (1, 0), (1, 0), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def _messages(msgs: List[str], S, cols: int = 2):
    """Key messages in a soft box, in columns."""
    if not msgs:
        return Spacer(0, 0)
    paras = [Paragraph(m, S["msg"], bulletText="•") for m in msgs]
    per = int(np.ceil(len(paras) / cols))
    grid = [paras[i * per:(i + 1) * per] for i in range(cols)]
    w = FULL_W / cols
    t = Table([grid], colWidths=[w * cm] * cols)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SOFT), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LINEBEFORE", (0, 0), (0, -1), 3, NAVY)]))
    return t


def _table(data, widths, extra=(), font=7.2, align="CENTER") -> Table:
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", font), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font),
                           ("BACKGROUND", (0, 0), (-1, 0), HEAD), ("ALIGN", (1, 0), (-1, -1), align),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, LINE),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6)] + list(extra)))
    return t


def _pace_bg(v):
    if not np.isfinite(v):
        return colors.white
    return colors.HexColor(next(c for lim, c in PACE_BG if v <= lim))


def _name(level: str, country: str) -> str:
    return "Organisation" if level == "Organisation" else country


# ───────────────────────────── shared sections ─────────────────────────────
def _charts_pair(res, level: str, country: str, h: float = 2.55) -> Table:
    imgs = []
    for m in ("Received", "Submitted"):
        hs, f = _series(res, level, country, m)
        imgs.append(_img(charts.pack_forecast(hs, f, m, f"Applications {m.lower()} per month", size=(6.4, h)), HALF))
    return _row(imgs, [HALF + 0.3, HALF])


def _headline(res, S, level: str, country: str, no: str, title: str) -> List:
    msgs = ins.story_messages(res, level, country)
    sub = msgs[0].replace("<b>", "").replace("</b>", "") if msgs else ""
    out = [_band(no, title, sub, res, S)]
    if level == "Organisation":
        out += _critical_banner(res, S)
    return out + [Spacer(0, 5), _tiles(res, level, country, S), Spacer(0, 5), _messages(msgs[1:] or msgs, S),
                  Spacer(0, 5), _charts_pair(res, level, country, 2.7),
                  Spacer(0, 3), _month_strip(res, level, country, S), PageBreak()]


def _funnel_steps(res, country: str, level: str):
    """[(label, next-quarter forecast, same months last year)] from submitted onwards."""
    sub = _tot(res, level, country, "Submitted", "Next 3 months")
    ly = sub["Forecast"] / (1 + sub["YoY_Pct"] / 100) if sub and np.isfinite(sub.get("YoY_Pct", np.nan)) else np.nan
    steps = [("Submitted", sub.get("Forecast", np.nan) if sub else np.nan, ly)]
    tot = getattr(res, "stage_totals", None)
    if tot is not None and len(tot):
        q = tot[(tot["Window"] == "Next 3 months") & (tot["Country"] == country)].set_index("StageGroup")
        for s in STAGE_ORDER:
            if s in q.index and s not in ("Visa_Refused", "Deferred"):
                steps.append((STAGE_LABEL[s].rstrip("*†"), float(q.loc[s, "Forecast"]),
                              float(q.loc[s, "Same_Months_Last_Year"])))
    return steps


def _intake_chart(res, level: str, country: str, metric: str, k: int = 0, size=(6.4, 2.8)):
    rows = ins.intake_rows(res, level, country, metric)
    cur = getattr(res, "intake_curves", None)
    if rows.empty or cur is None or cur.empty or k >= len(rows):
        return None
    r = rows.iloc[k]
    c = cur[(cur["Level"] == level) & (cur["Country"] == country) & (cur["Metric"] == metric) & (cur["Intake"] == r["Intake"])]
    if c.empty:
        return None
    return charts.intake_curve(c, r.to_dict(), f"{r['Intake']} intake — applications {metric.lower()} so far",
                               metric, size=size)


def _intake_table(res, S, level: str = "Country", country: str = None, per: int = 2, metric: str = "Received",
                  max_rows: int = 14):
    """Running intakes: so far, vs last year at the same point (coloured), likely final with its range."""
    p = getattr(res, "intake_pace", None)
    if p is None or p.empty:
        return None
    x = p[(p["Metric"] == metric) & (p["Months_To_Start"] >= 1)]
    if country is None:
        x = x[x["Level"].isin(["Organisation", "Country"])]
    else:
        x = x[(x["Country"] == country) & x["Level"].isin(["Country", "Region"])]
        if x.loc[x["Level"] == "Region", "Region"].nunique() < 2:      # single region: it is the destination
            x = x[x["Level"] == "Country"]
    if x.empty:
        return None
    x = x.assign(_o=(x["Level"] != "Organisation").astype(int), _t=(x["Region"] != TOTAL).astype(int))
    x = x.sort_values(["_o", "Country", "_t", "Region", "Last_Year_Final"], ascending=[True, True, True, True, False])
    x = x.groupby(["Level", "Country", "Region"], sort=False).head(per)
    if country is not None:
        x = x.head(max_rows)
    hdr = ["Destination" if country is None else "Region / team", "Intake", "Months to go", "So far",
           "Last year, same point", "Pace", "Likely final (range)", "Last year final"]
    data, st = [hdr], []
    for i, r in enumerate(x.itertuples(), start=1):
        who = _name(r.Level, r.Country) if country is None else ("All regions" if r.Region == TOTAL else r.Region)
        fin = (f"{_n(r.Projected_Final)}  ({_n(r.Lo80)}–{_n(r.Hi80)})" if np.isfinite(r.Projected_Final)
               else "too early to project")
        data.append([who, r.Intake, f"{int(r.Months_To_Start)}", _n(r.To_Date, True), _n(r.Last_Year_Same_Point, True),
                     _pct(r.Pace_Pct), fin, _n(r.Last_Year_Final)])
        st.append(("BACKGROUND", (5, i), (5, i), _pace_bg(r.Pace_Pct)))
        if r.Level == "Organisation" or (country is not None and r.Region == TOTAL):
            st.append(("FONT", (0, i), (-1, i), "Helvetica-Bold", 7.0))
    return _table(data, [5.0, 2.2, 2.0, 2.6, 3.4, 2.0, 6.0, 3.2], st, font=7.0)


# ───────────────────────────── organisation chapters ─────────────────────────────
def _ch2_accuracy(res, S) -> List:
    lm = ins.last_month(res)
    proof = ins.last_quarter_proof(res)
    pr = proof[proof["Metric"] == "Received"] if len(proof) else proof
    org = pr[pr["Level"] == "Organisation"] if len(pr) else pr
    r12 = _tot(res, "Organisation", ORG, "Received", "Next 12 months")
    err = r12.get("Typical_Quarter_Error_Pct", np.nan) if r12 else np.nan
    cov = _tested_coverage(res, "Organisation")
    w = (FULL_W - 0.3) / 4
    made = pd.Timestamp(pr["Made_At"].iloc[0]) if len(pr) else None
    tiles = [
        _tile(f"{lm['Month']:%B}: actual vs forecast" if lm else "Last month",
              _pct(100 * (lm["Actual"] / lm["Forecast"] - 1)) if lm else "–",
              [f"{_n(lm['Actual'])} received vs {_n(lm['Forecast'])} forecast" if lm else "", "+ = more than forecast"],
              S, w),
        _tile(f"Last quarter (forecast made {made:%b %Y})" if made is not None else "Last quarter",
              _pct(-org["Miss_Pct"].iloc[0]) if len(org) else "–",
              [f"{_n(org['Actual_Total'].iloc[0])} actual vs {_n(org['Forecast_Total'].iloc[0])} forecast" if len(org) else "",
               "+ = more than forecast"], S, w),
        _tile("Typical miss on a quarter (tested)", f"±{err:.0f}%" if np.isfinite(err) else "–",
              [f"forecasts re-made for {ins.wf_period(res)}"], S, w),
        _tile("Plan-for range held (tested)", f"{cov:.0f}%" if np.isfinite(cov) else "–", ["of quarters; aim 80%"], S, w)]
    left = None
    if len(pr):
        data, st = [["Last quarter", "Forecast", "Actual", "Miss"]], []
        for i, r in enumerate(pr.itertuples(), start=1):
            small = r.Actual_Total < 5000
            data.append([_name(r.Level, r.Country), _n(r.Forecast_Total, small), _n(r.Actual_Total, small), _pct(r.Miss_Pct)])
            bg = next(c for lim, c in MISS_BG if abs(r.Miss_Pct) <= lim) if np.isfinite(r.Miss_Pct) else REL_BG["New"]
            st += [("BACKGROUND", (3, i), (3, i), bg), ("TEXTCOLOR", (3, i), (3, i), colors.white)]
            if r.Level == "Organisation":
                st.append(("FONT", (0, i), (-1, i), "Helvetica-Bold", 7.0))
        left = _table(data, [5.2, 2.6, 2.6, 2.4], st, font=7.0)
    right = None
    d = ins.vintage_matrix(res, metric="Received")
    if len(d):
        vcols = [c for c in d.columns if c not in ("Destination", "Level")]
        data, st = [["Miss by forecast made in"] + [c.replace("->", "for") for c in vcols]], []
        for i, r in enumerate(d.itertuples(index=False), start=1):
            vals = [r[2 + j] for j in range(len(vcols))]
            data.append([r[0]] + [_pct(v) for v in vals])
            for j, v in enumerate(vals, start=1):
                st.append(("BACKGROUND", (j, i), (j, i), _miss_bg(v)))
            if r[1] == "Organisation":
                st.append(("FONT", (0, i), (-1, i), "Helvetica-Bold", 7.0))
        w0 = 4.0
        right = _table(data, [w0] + [(HALF - w0) / len(vcols)] * len(vcols), st, font=6.6)
    fva = res.forecast_vs_actual
    ch = [_img(charts.track_record(fva, "Organisation", ORG, m, f"Track record — applications {m.lower()}",
                                   size=(6.4, 2.2)), HALF) for m in ("Received", "Submitted")]
    out = [_band("2", "Last month and how accurate we have been",
                 "What we forecast against what happened: last month, the last full quarter, and every recent forecast.",
                 res, S), Spacer(0, 5), _row(tiles, [w + 0.08] * 4), Spacer(0, 6)]
    if left is not None or right is not None:
        out.append(_row([left or "", right or ""], [HALF + 0.3, HALF]))
        out.append(Paragraph("Left: quarter forecast re-made at the time with today's method; green within 10%, amber 10–20%, "
                             "red over 20% (− = actual came in higher). Right: each forecast's miss on the months since; blue "
                             "= actual higher, orange = lower, darker = bigger.", S["note"]))
    return out + [Spacer(0, 5), _row(ch, [HALF + 0.3, HALF]), PageBreak()]


def _ch3_destinations(res, S) -> List:
    ct = ins.country_table(res)
    msgs = ins.story_messages(res)
    sub = next((m for m in msgs if "Growth comes from" in m), "").replace("<b>", "").replace("</b>", "")
    return [_band("3", "Where growth comes from", sub or "Destinations over the next 12 months.", res, S), Spacer(0, 5),
            _row([_img(charts.destination_bars(ct, "Received", "Next 12 months by destination", size=(6.4, 3.1)), HALF),
                  _img(charts.growth_check(ct, "Received", "Growth: forecast vs what the market is doing now",
                                           size=(6.4, 3.1)), HALF)], [HALF + 0.3, HALF]),
            Spacer(0, 4), _destination_table(res),
            Paragraph("Market now = applications received in the last 6 months vs the same 6 months a year earlier; "
                      "orange = differs from the forecast growth by more than 25 points: confirm the assumption with the "
                      "market team.", S["note"]), PageBreak()]


def _stage_table(res):
    tot = getattr(res, "stage_totals", None)
    if tot is None or tot.empty:
        return None
    q = tot[tot["Window"] == "Next 3 months"]
    stages = [s for s in STAGE_ORDER if s in set(q["StageGroup"])]
    rows = q.pivot_table(index=["Level", "Country"], columns="StageGroup", values="Forecast", aggfunc="sum").reset_index()
    yoy = q.pivot_table(index=["Level", "Country"], columns="StageGroup", values="YoY_Pct", aggfunc="first")
    rows["_o"] = (rows["Level"] != "Organisation").astype(int)
    rows = rows.sort_values(["_o", stages[0]], ascending=[True, False]).reset_index(drop=True)
    data, st = [["Next quarter"] + [STAGE_LABEL[s] for s in stages]], []
    for i, r in rows.iterrows():
        lvl, cty = r["Level"], r["Country"]
        cells = []
        for s in stages:
            v = r[s]
            y = yoy.loc[(lvl, cty), s] if (lvl, cty) in yoy.index and s in yoy.columns else np.nan
            cells.append("–" if not np.isfinite(v) else f"{_n(v)}  ({_pct(y)})" if np.isfinite(y) else _n(v))
        data.append([_name(lvl, cty)] + cells)
        if lvl == "Organisation":
            st.append(("FONT", (0, i + 1), (-1, i + 1), "Helvetica-Bold", 7.0))
    w0 = 5.0
    return _table(data, [w0] + [(FULL_W - w0) / len(stages)] * len(stages), st, font=7.0)


def _ch4_pipeline(res, S) -> List:
    steps = _funnel_steps(res, ORG, "Organisation")
    out = [_band("4", "Pipeline after submission", "Where this quarter's submissions are expected to go next: offers, "
                 "deposits, visas and enrolment.", res, S), Spacer(0, 5)]
    left = _img(charts.funnel(steps, "Organisation — next quarter, applications reaching each stage", size=(6.4, 3.0)), HALF)
    ic = _intake_chart(res, "Organisation", ORG, "Submitted", size=(6.4, 3.0))
    out.append(_row([left, _img(ic, HALF) if ic else ""], [HALF + 0.3, HALF]))
    t = _stage_table(res)
    if t is not None:
        out += [Spacer(0, 5), t]
    out.append(Paragraph(
        "First time each application reaches the stage in the next three months (change vs the same months last year, "
        f"left out below {MIN_YOY_BASE}). Each stage is forecast from the earlier stage that tested best for that "
        "destination. Visa document = CAS (UK), Funds/COE (Australia), AIP (NZ), I-20 (USA), else equals visa applied. "
        "Enrolled = enrolment confirmed (commission-earning).", S["note"]))
    return out + [PageBreak()]


def _ch4_intakes(res, S) -> List:
    rows = ins.intake_rows(res)
    if rows.empty:
        return []
    r = rows.iloc[0]
    sub = (f"{r['Intake']} intake: {_n(r['To_Date'])} received so far, {_pct(r['Pace_Pct'])} vs last year at the same "
           "point" + (f"; likely final {_n(r['Projected_Final'])}." if np.isfinite(r["Projected_Final"]) else "."))
    imgs = [_intake_chart(res, "Organisation", ORG, "Received", 0, size=(6.4, 2.6)),
            _intake_chart(res, "Organisation", ORG, "Received", 1, size=(6.4, 2.6))]
    out = [_band("4", "The coming intakes", sub, res, S), Spacer(0, 5),
           _row([_img(i, HALF) if i else "" for i in imgs], [HALF + 0.3, HALF]), Spacer(0, 5)]
    t = _intake_table(res, S)
    if t is not None:
        out.append(t)
    out.append(Paragraph(
        "Applications for each intake counted so far, against the same intake a year earlier at the same number of "
        "months before its start (blue = ahead, orange = behind). Likely final = so far ÷ the share that the last three "
        "intakes of that month already had at this point; shown only where that tested better than 'same as last "
        "year' on past intakes, with the range those tests give (8 in 10).", S["note"]))
    return out + [PageBreak()]


def _fte_table(res):
    st = getattr(res, "staffing", None)
    if st is None or st.empty or "FTE_Base" not in st or not st["FTE_Base"].notna().any():
        return None
    x = st[st["Level"].isin(["Organisation", "Country"]) & st["FTE_Base"].notna()].copy()
    x["Date"] = pd.to_datetime(x["Date"])
    q = x[x["Date"] < x["Date"].min() + pd.DateOffset(months=3)]
    piv = q.pivot_table(index=["Level", "Label"], columns="Metric", values="FTE_Capacity_80", aggfunc="max").reset_index()
    mets = [m for m in ["Received", "Submitted"] + list(STAGE_ORDER) if m in piv.columns]
    data = [["FTE needed at the busiest month of next quarter (plan for)"] + [STAGE_LABEL.get(m, m) for m in mets]]
    for _, r in piv.sort_values("Level", ascending=False).iterrows():
        data.append([r["Label"]] + [f"{r[m]:.1f}" if np.isfinite(r[m]) else "–" for m in mets])
    return _table(data, [7.0] + [(FULL_W - 7.0) / len(mets)] * len(mets), font=7.0)


def _ch5_capacity(res, S) -> List:
    st = getattr(res, "staffing", None)
    if st is None or st.empty:
        return []
    out = [_band("5", "Capacity", "Busy and quiet months ahead for each destination, so staffing can be planned before "
                 "the peak.", res, S), Spacer(0, 4)]
    for m in ("Received", "Submitted"):
        out.append(_img(charts.workload_heatmap(
            st, m, f"Applications {m.lower()} per month.  Blue = busier than that destination's average month, "
                   "grey = quieter; bold = at least 10% busier.", size=(13.2, 3.2)), FULL_W))
    t = _fte_table(res)
    out += ([Spacer(0, 4), t] if t is not None else
            [Paragraph("FTE needed appears here once destination teams enter today's headcount per stage in their live "
                       "sheets (Forms/Live).", S["note"])])
    return out + [PageBreak()]


METHOD_LABEL = {"statistical_overlay": "Correction for recent over/under-forecasting",
                "point_estimate": "Single number taken from the forecast range", "submitted_funnel": "Submissions forecast",
                "reconciliation": "Regions add up to destinations", "vintage_signal": "Direction signal from latest forecasts",
                "calendar_effects": "Working-day and festival effects", "hedge": "Hedge with last year where models lose"}


def _ch6_trust(res, S) -> List:
    out = [_band("6", "Can we trust it", "The same acceptance checks every month, on forecasts re-made for past months "
                 "and on the forecasts we actually published.", res, S), Spacer(0, 6), _health_table(res), Spacer(0, 8)]
    dec = getattr(res, "decisions", None)
    if dec is not None and len(dec):
        small = ParagraphStyle("x", fontName="Helvetica", fontSize=6.8, leading=8.2)
        data = [["What the method uses", "Used", "Test favours", "Decided", "Why (next-quarter error, % — lower is better)"]]
        for r in dec.itertuples():
            data.append([METHOD_LABEL.get(r.Component, r.Component), str(r.Used), str(r.Test_Favours), str(r.Decided_On),
                         Paragraph(str(r.Scores), small)])
        out += [_table(data, [7.0, 2.6, 2.6, 2.4, FULL_W - 14.6], font=7.0, align="LEFT"),
                Paragraph("Every component must make next-quarter totals at least 3% more accurate in the walk-forward "
                          "test to be used; choices are locked for a quarter and reviewed in Jan, Apr, Jul and Oct.", S["note"])]
    rs = ins.reliability_summary(res)
    if len(rs):
        data = [["Reliability (regions)", "Series", "Share of forecast volume"]]
        for r in rs.itertuples():
            data.append([r.Reliability, str(r.Series), f"{r.Volume_Share_Pct:.0f}%"])
        rel = _table(data, [4.2, 2.6, 4.2], font=7.0)
        acc = getattr(res, "accuracy", None)
        img = (_img(charts.accuracy_by_horizon(acc, "Received", "Country", size=(6.4, 2.5)), HALF)
               if acc is not None and len(acc) else "")
        out += [Spacer(0, 6), _row([rel, img], [HALF + 0.3, HALF])]
    return out + [PageBreak()]


# ───────────────────────────── destination chapter, page B ─────────────────────────────
def _region_table(res, country: str, top: int = 12):
    t = res.totals
    sel = lambda m, w: t[(t["Level"] == "Region") & (t["Country"] == country) & (t["Metric"] == m) & (t["Window"] == w)]
    r12 = sel("Received", "Next 12 months")
    if r12["Region"].nunique() < 2:
        return None
    x = (r12.set_index("Region")[["Forecast", "YoY_Pct", "Reliability"]]
         .join(sel("Received", "Next 3 months").set_index("Region")["Forecast"].rename("Q"))
         .join(sel("Submitted", "Next 12 months").set_index("Region")["Forecast"].rename("S"))
         .sort_values("Forecast", ascending=False))
    rest = x.iloc[top:]
    data, st = [["Region / team", "Next quarter", "Next 12 m", "vs last year", "Submitted 12 m", "Reliability"]], []
    for i, (reg, r) in enumerate(x.iloc[:top].iterrows(), start=1):
        data.append([reg, _n(r["Q"], True), _n(r["Forecast"], True), _pct(r["YoY_Pct"]), _n(r["S"], True), r["Reliability"]])
        if r["Reliability"] in REL_BG:
            st += [("BACKGROUND", (5, i), (5, i), REL_BG[r["Reliability"]]), ("TEXTCOLOR", (5, i), (5, i), colors.white)]
    if len(rest):
        data.append([f"{len(rest)} smaller regions", _n(rest["Q"].sum(), True), _n(rest["Forecast"].sum(), True), "",
                     _n(rest["S"].sum(), True), ""])
    return _table(data, [4.6, 1.9, 1.9, 1.8, 2.1, 1.6], st, font=6.8)


def _destination_b(res, S, country: str, label: str) -> List:
    msgs = ins.story_messages(res, "Country", country)
    sub = next((m for m in msgs if "Biggest" in m or "intake" in m),
               "Regions, pipeline after submission and the coming intakes.").replace("<b>", "").replace("</b>", "")
    out = [_band("B", f"{label} — regions, pipeline and intakes", sub, res, S), Spacer(0, 5)]
    rt = _region_table(res, country)
    if rt is not None:
        h, f, t = res.history, res.forecast, res.totals
        hr = h[(h["Level"] == "Region") & (h["Country"] == country) & (h["Metric"] == "Received")].assign(
            Date=lambda d: pd.to_datetime(d["Date"]))
        fr = f[(f["Level"] == "Region") & (f["Country"] == country) & (f["Metric"] == "Received")].assign(
            Date=lambda d: pd.to_datetime(d["Date"]))
        top = (t[(t["Level"] == "Region") & (t["Country"] == country) & (t["Metric"] == "Received")
                 & (t["Window"] == "Next 12 months")].sort_values("Forecast", ascending=False)["Region"].head(6).tolist())
        mult = _img(charts.region_multiples(hr, fr, top, "Largest regions — received, actual and forecast", ncols=3,
                                            size=(6.4, 2.9)), HALF)
        out.append(_row([rt, mult], [HALF + 0.3, HALF]))
    else:
        it = _intake_table(res, S, "Country", country, per=3)
        out.append(it if it is not None else Paragraph("Single region: the destination page shows it in full.", S["note"]))
    fva = getattr(res, "forecast_vs_actual", None)
    cells = [_img(charts.funnel(_funnel_steps(res, country, "Country"), "Pipeline — next quarter", size=(6.4, 3.4)), THIRD)]
    ic = _intake_chart(res, "Country", country, "Received", size=(6.4, 3.4))
    cells.append(_img(ic, THIRD) if ic else Paragraph("No running intake large enough to track yet.", S["note"]))
    cells.append(_img(charts.track_record(fva, "Country", country, "Received", "Track record — received",
                                          size=(6.4, 3.4)), THIRD) if fva is not None and len(fva) else "")
    out += [Spacer(0, 6), _row(cells, [THIRD + 0.3] * 3)]
    if rt is None:                                   # single region: a second row instead of the region view
        more = [_intake_chart(res, "Country", country, "Received", 1, size=(6.4, 3.4)),
                _intake_chart(res, "Country", country, "Submitted", 0, size=(6.4, 3.4)),
                charts.track_record(fva, "Country", country, "Submitted", "Track record — submitted", size=(6.4, 3.4))
                if fva is not None and len(fva) else None]
        out += [Spacer(0, 6), _row([_img(m, THIRD) if m else "" for m in more], [THIRD + 0.3] * 3)]
    if rt is not None:
        it = _intake_table(res, S, "Country", country, per=1, max_rows=7)
        if it is not None:
            out += [Spacer(0, 5), it]
    return out + [PageBreak()]


def _destination(res, S, country: str, label: str) -> List:
    return _headline(res, S, "Country", country, "A", label) + _destination_b(res, S, country, label)


# ───────────────────────────── documents ─────────────────────────────
def write_story(res, path: Path) -> Path:
    S = _s()
    story: List = _headline(res, S, "Organisation", ORG, "1", "Where we are heading — Organisation")
    story += _ch2_accuracy(res, S) + _ch3_destinations(res, S) + _ch4_pipeline(res, S) + _ch4_intakes(res, S)
    story += _ch5_capacity(res, S) + _ch6_trust(res, S)
    for c, lab in _countries(res):
        story += _destination(res, S, c, lab)
    story += _page_legend(res, S)
    _doc(path, "Application Flow Forecast — Monthly Pack").build(story, onFirstPage=_footer, onLaterPages=_footer)
    return path


def write_country_pack(res, country: str, label: str, path: Path) -> Path:
    S = _s()
    story = _destination(res, S, country, label)
    if isinstance(story[-1], PageBreak):
        story = story[:-1]
    _doc(path, f"Application Flow Forecast — {label}").build(story, onFirstPage=_footer, onLaterPages=_footer)
    return path


def write_country_packs(res, folder: Path) -> Dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for c, lab in _countries(res):
        safe = "".join(ch if ch.isalnum() else "_" for ch in lab).strip("_")
        out[lab] = write_country_pack(res, c, lab, folder / f"{safe[:28]}.pdf")
    return out
