# -*- coding: utf-8 -*-
"""
Monthly forecast pack (PDF): one fixed layout, filled from the run's data every month.
Visual first, numbers written on the charts, almost no prose; every page carries its own legend,
so nothing has to be looked up elsewhere.

  1  Organisation dashboard           4   Track record (forecast vs actual)
  2  Destinations at a glance         5+  One page per destination (also saved as a separate PDF)
  3  Workload calendar, next 12 months     last  How to read this pack (fixed)
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .. import __version__
from ..data.prepare import ORG, TOTAL
from . import charts
from . import insights as ins

PAGE_W, PAGE_H = landscape(A4)
MARGIN = 1.2 * cm
FULL_W = (PAGE_W - 2 * MARGIN) / cm          # usable width in cm (~27.3)
INK, INK2, LINE, HEAD = (colors.HexColor(c) for c in ("#0b0b0b", "#52514e", "#d0d7e2", "#eef2f7"))
REL_BG = {k: colors.HexColor(v) for k, v in charts.REL_COLOR.items()}
REL_TEXT = {"High": "quarter totals typically within ±{e:.0f}%",
            "Medium": "quarter totals typically within ±{e:.0f}%",
            "Low": "quarter totals typically within ±{e:.0f}% — plan on the range",
            "New": "too little history to test",
            "Review": "capped by plausibility check — confirm with market team"}


def _styles() -> Dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=16, leading=19, textColor=INK),
        "meta": ParagraphStyle("m", fontName="Helvetica", fontSize=7.5, leading=9.5, textColor=INK2, alignment=2),
        "tl": ParagraphStyle("tl", fontName="Helvetica", fontSize=7.3, leading=9, textColor=INK2),
        "tv": ParagraphStyle("tv", fontName="Helvetica-Bold", fontSize=15.5, leading=18.5, textColor=INK),
        "ts": ParagraphStyle("ts", fontName="Helvetica", fontSize=7.3, leading=9.2, textColor=INK),
        "legend": ParagraphStyle("l", fontName="Helvetica", fontSize=8.5, leading=12, textColor=INK),
    }


# ───────────────────────────── building blocks ─────────────────────────────
def _img(png: bytes, w_cm: float) -> Image:
    im = Image(io.BytesIO(png))
    ratio = im.imageHeight / im.imageWidth
    im.drawWidth, im.drawHeight = w_cm * cm, w_cm * cm * ratio
    return im


def _row(cells, widths) -> Table:
    t = Table([cells], colWidths=[w * cm for w in widths])
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def _n(v, small: bool = False) -> str:
    """Whole numbers people can plan with: rounded to 100 (to 10 for small series)."""
    if v is None or not np.isfinite(v):
        return "–"
    step = 10 if small or abs(v) < 2000 else 100
    return f"{round(v / step) * step:,.0f}"


def _pct(v) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:+.0f}%"


def _header(title: str, res, S) -> Table:
    fr, to = res.forecast["Date"].min(), res.forecast["Date"].max()
    meta = (f"Actuals to {res.last_actual:%b %Y}  ·  Forecast {fr:%b %Y} – {to:%b %Y}<br/>"
            f"Run {res.run_date}  ·  pipeline v{__version__}")
    t = Table([[Paragraph(title, S["title"]), Paragraph(meta, S["meta"])]], colWidths=[17 * cm, (FULL_W - 17) * cm])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 1.2, INK), ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def _tile(label: str, value: str, lines: List[str], S, width: float, accent=None) -> Table:
    lines = [x for x in lines if x][:3]
    lines += [" "] * (3 - len(lines))          # equal tile heights
    rows = [[Paragraph(label, S["tl"])], [Paragraph(value, S["tv"])]] + [[Paragraph(x, S["ts"])] for x in lines]
    t = Table(rows, colWidths=[width * cm])
    st = [("BOX", (0, 0), (-1, -1), 0.6, LINE), ("BACKGROUND", (0, 0), (-1, -1), colors.white),
          ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]
    if accent is not None:
        st.append(("LINEBEFORE", (0, 0), (0, -1), 4, accent))
    t.setStyle(TableStyle(st))
    return t


def _tot(res, level: str, country: str, metric: str, window: str) -> dict:
    t = res.totals
    r = t[(t["Level"] == level) & (t["Country"] == country) & (t["Region"] == TOTAL) & (t["Metric"] == metric)
          & (t["Window"] == window)]
    return r.iloc[0].to_dict() if len(r) else {}


def _series(res, level: str, country: str, metric: str):
    hi, fc = res.history, res.forecast
    h = hi[(hi["Level"] == level) & (hi["Country"] == country) & (hi["Region"] == TOTAL) & (hi["Metric"] == metric)]
    f = fc[(fc["Level"] == level) & (fc["Country"] == country) & (fc["Region"] == TOTAL) & (fc["Metric"] == metric)]
    if h.empty:   # single-region destination: its history is its region
        h = (hi[(hi["Level"] == "Region") & (hi["Country"] == country) & (hi["Metric"] == metric)]
             .groupby("Date", as_index=False)["Actual"].sum())
    return h, f


def _tested_coverage(res, level: str) -> float:
    rt = getattr(res, "range_tested", None)
    if rt is None or rt.empty:
        return np.nan
    x = rt[(rt["Level"] == level) & (rt["Window"] == "Next 3 months") & (rt["Interval"] == "80%")]
    return float(x["Tested_Coverage_Pct"].iloc[0]) if len(x) and x["N"].iloc[0] >= 10 else np.nan


def _tiles(res, level: str, country: str, S) -> Table:
    """Five tiles, identical for the organisation and every destination."""
    r3 = _tot(res, level, country, "Received", "Next 3 months")
    r12 = _tot(res, level, country, "Received", "Next 12 months")
    s12 = _tot(res, level, country, "Submitted", "Next 12 months")
    small = bool(r12) and r12.get("Forecast", 0) < 5000
    w = (FULL_W - 0.5) / 6

    def vol(label, r):
        if not r:
            return _tile(label, "–", [], S, w)
        width = (r.get("Hi80", np.nan) - r.get("Lo80", np.nan)) / r["Forecast"] if r["Forecast"] else np.nan
        last = (f"{_pct(r.get('YoY_Pct'))} vs last year · <b>direction only: plan on the next quarter</b>"
                if np.isfinite(width) and width > WIDE_RANGE else f"{_pct(r.get('YoY_Pct'))} vs same months last year")
        return _tile(label, _n(r["Forecast"], small),
                     [f"Most likely <b>{_n(r.get('Lo50'), small)} – {_n(r.get('Hi50'), small)}</b>",
                      f"Plan for <b>{_n(r.get('Lo80'), small)} – {_n(r.get('Hi80'), small)}</b>", last], S, w)

    q_lab = (f"Received · next quarter ({pd.Timestamp(r3['From']):%b}–{pd.Timestamp(r3['To']):%b})" if r3
             else "Received · next quarter")
    rel = r12.get("Reliability", "New") if r12 else "New"
    err = r12.get("Typical_Quarter_Error_Pct", np.nan) if r12 else np.nan
    rel_lines = [REL_TEXT.get(rel, "").format(e=err if np.isfinite(err) else 0)]
    cov = _tested_coverage(res, level)
    if np.isfinite(cov):
        rel_lines.append(f"plan-for ranges held {cov:.0f}% of the time (aim 80%)")
    recent = ins.recent_yoy(res.history, level, country, "Received")
    trend = [f"forecast assumes <b>{_pct(r12.get('YoY_Pct'))}</b> for the next 12 months" if r12 else "",
             "if very different, confirm with the market"]
    tiles = [vol(q_lab, r3), _fy_tile(res, level, country, "Received", S, w, small),
             vol("Received · next 12 months", r12), _fy_tile(res, level, country, "Submitted", S, w, small),
             _tile("Reliability", rel, rel_lines, S, w, accent=REL_BG.get(rel)),
             _tile("Market now: last 6 months vs a year earlier", _pct(recent), trend, S, w)]
    return _row(tiles, [w + 0.08] * 6)


def _fy_tile(res, level: str, country: str, metric: str, S, w: float, small: bool) -> Table:
    """Current fiscal year = actual to date + forecast for the rest; only the rest is uncertain."""
    p = getattr(res, "periods", None)
    if p is None or p.empty:
        return _tile(f"{metric} · this fiscal year", "–", [], S, w)
    x = p[(p["Level"] == level) & (p["Country"] == country) & (p["Region"] == TOTAL) & (p["Metric"] == metric)
          & (p["Period_Type"] == "Fiscal year") & (p["Months_Forecast"] > 0)].sort_values("Period_Start")
    if x.empty:
        return _tile(f"{metric} · this fiscal year", "–", [], S, w)
    r = x.iloc[0]
    lines = [f"{_n(r['Actual_To_Date'], small)} done ({int(r['Months_Actual'])} m) + {_n(r['Forecast_Remaining'], small)} to come"]
    if "Outlook_Lo80" in r and np.isfinite(r.get("Outlook_Lo80", np.nan)):
        lines.append(f"Plan for <b>{_n(r['Outlook_Lo80'], small)} – {_n(r['Outlook_Hi80'], small)}</b>")
    lines.append(f"{_pct(r['YoY_Pct'])} vs last fiscal year")
    return _tile(f"{metric} · {r['Period']} outlook", _n(r["Outlook"], small), lines, S, w)


def _month_strip(res, level: str, country: str, S) -> Table:
    """The next 12 months as numbers, so the chart never has to be read off by eye."""
    _, fr = _series(res, level, country, "Received")
    _, fs = _series(res, level, country, "Submitted")
    fr, fs = fr.sort_values("Date"), fs.sort_values("Date")
    small = fr["Forecast"].sum() < 5000
    data = [["Month"] + [f"{d:%b %y}" for d in fr["Date"]],
            ["Received"] + [_n(v, small) for v in fr["Forecast"]],
            ["  plan for"] + [f"{_n(a, small)}–{_n(b, small)}" for a, b in zip(fr["Lo80"], fr["Hi80"])],
            ["Submitted"] + [_n(v, small) for v in fs.set_index("Date")["Forecast"].reindex(fr["Date"])]]
    if "Previous_Forecast" in fr and fr["Previous_Forecast"].notna().any():
        data.insert(3, ["  last month's fcst"] + [_n(v, small) if np.isfinite(v) else "new" for v in fr["Previous_Forecast"]])
    first = 2.0
    t = Table(data, colWidths=[first * cm] + [(FULL_W - first) / 12 * cm] * 12)
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", 6.6), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 6.8),
          ("FONT", (0, 1), (-1, 1), "Helvetica-Bold", 7), ("BACKGROUND", (0, 0), (-1, 0), HEAD),
          ("TEXTCOLOR", (0, 2), (-1, 2), INK2), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
          ("LINEBELOW", (0, 0), (-1, -1), 0.25, LINE), ("TOPPADDING", (0, 0), (-1, -1), 1.6),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6)]
    if len(fr):
        peak = int(np.argmax(fr["Forecast"].to_numpy())) + 1
        st.append(("BACKGROUND", (peak, 1), (peak, 1), colors.HexColor("#cfe1f7")))
    t.setStyle(TableStyle(st))
    return t


def _destination_table(res) -> Table:
    ct = ins.country_table(res)
    hdr = ["Destination", "Next quarter", "Next 12 months", "Plan for (12 m)", "vs last year", "Market now*",
           "vs last run", "Submitted (12 m)", "Reliability"]
    data, st = [hdr], []
    for i, r in enumerate(ct.itertuples(), start=1):
        small = r.Received_Forecast < 5000
        rel = getattr(r, "Received_Reliability", "New")
        err = getattr(r, "Received_Quarter_Error_Pct", np.nan)
        data.append([r.Country, _n(getattr(r, "Received_Next_Quarter", np.nan), small), _n(r.Received_Forecast, small),
                     f"{_n(r.Received_Lo80, small)} – {_n(r.Received_Hi80, small)}", _pct(r.Received_YoY_Pct),
                     _pct(getattr(r, "Received_Recent_YoY_Pct", np.nan)),
                     _pct(getattr(r, "Received_Change_vs_Previous_Pct", np.nan)),
                     _n(getattr(r, "Submitted_Forecast", np.nan), small),
                     f"{rel}" + (f"  ±{err:.0f}%" if np.isfinite(err) else "")])
        if rel in REL_BG:
            st += [("BACKGROUND", (8, i), (8, i), REL_BG[rel]), ("TEXTCOLOR", (8, i), (8, i), colors.white)]
        gap = abs(r.Received_YoY_Pct - getattr(r, "Received_Recent_YoY_Pct", np.nan))
        if np.isfinite(gap) and gap > GROWTH_GAP_PP:
            st.append(("BACKGROUND", (5, i), (5, i), colors.HexColor("#fde2c8")))
        if r.Level == "Organisation":
            st.append(("FONT", (0, i), (-2, i), "Helvetica-Bold", 7.3))
    w = [4.4, 2.4, 2.6, 3.5, 2.2, 2.3, 2.9, 2.7, 3.2]
    t = Table(data, colWidths=[x * cm for x in w], repeatRows=1)
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 7.3), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.3),
                           ("BACKGROUND", (0, 0), (-1, 0), HEAD), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                           ("ALIGN", (8, 0), (8, -1), "CENTER"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, LINE),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.8), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8)] + st))
    return t


# ───────────────────────────── pages ─────────────────────────────
def _page_dashboard(res, S, level: str, country: str, name: str, chart_h: float = 2.9) -> List:
    story = [_header(name, res, S), Spacer(0, 5), _tiles(res, level, country, S), Spacer(0, 5)]
    half = FULL_W / 2 - 0.15
    imgs = []
    for m in ("Received", "Submitted"):
        h, f = _series(res, level, country, m)
        imgs.append(_img(charts.pack_forecast(h, f, m, f"Applications {m.lower()} per month", size=(6.4, chart_h)),
                         half))
    return story + [_row(imgs, [half + 0.3, half]), Spacer(0, 3), _month_strip(res, level, country, S)]


def _critical_banner(res, S) -> List:
    dq = getattr(res, "dq", None)
    if dq is None or dq.empty:
        return []
    c = dq[dq["Issue"].astype(str).str.startswith("CRITICAL")]
    if c.empty:
        return []
    txt = "<b>DATA WARNING — check before using this forecast:</b> " + " ".join(
        f"{r.Country}: {r.Detail}" for r in c.head(3).itertuples())
    t = Table([[Paragraph(txt, ParagraphStyle("w", parent=S["ts"], textColor=colors.white))]], colWidths=[FULL_W * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), STATUS_BG["Fail"]), ("LEFTPADDING", (0, 0), (-1, -1), 6)]))
    return [t, Spacer(0, 3)]


def _page_org(res, S) -> List:
    page = _page_dashboard(res, S, "Organisation", ORG, "Application flow forecast — Organisation")
    return page[:1] + _critical_banner(res, S) + page[1:] + [PageBreak()]


MISS_BG = [(10, colors.HexColor("#2e9e5b")), (20, colors.HexColor("#e0a100")), (1e9, colors.HexColor("#d6453d"))]


def _page_proof(res, S) -> List:
    """Proof on the latest completed quarter: what this procedure forecast 3 months ago vs what happened."""
    d = ins.last_quarter_proof(res)
    if d.empty:
        return []
    made = pd.Timestamp(d["Made_At"].iloc[0])
    mcols = [pd.Timestamp(d[f"M{i}"].iloc[0]) for i in (1, 2, 3)]
    out = [_header(f"Proof: forecast made in {made:%b %Y} vs actual", res, S), Spacer(0, 5)]
    for metric in ("Received", "Submitted"):
        x = d[d["Metric"] == metric]
        if x.empty:
            continue
        hdr = ["Applications " + metric.lower()] + [f"{m:%b %Y} forecast / actual" for m in mcols] +               ["Quarter forecast", "Quarter actual", "Miss"]
        data, st = [hdr], []
        for i, r in enumerate(x.itertuples(), start=1):
            small = r.Actual_Total < 5000
            data.append([r.Country] + [f"{_n(getattr(r, f'F{j}'), small)} / {_n(getattr(r, f'A{j}'), small)}" for j in (1, 2, 3)]
                        + [_n(r.Forecast_Total, small), _n(r.Actual_Total, small), _pct(r.Miss_Pct)])
            bg = next(c for lim, c in MISS_BG if abs(r.Miss_Pct) <= lim) if np.isfinite(r.Miss_Pct) else REL_BG["New"]
            st += [("BACKGROUND", (6, i), (6, i), bg), ("TEXTCOLOR", (6, i), (6, i), colors.white)]
            if r.Level == "Organisation":
                st.append(("FONT", (0, i), (-2, i), "Helvetica-Bold", 7.3))
        t = Table(data, colWidths=[x_ * cm for x_ in (5.4, 3.7, 3.7, 3.7, 3.1, 3.1, 2.6)], repeatRows=1)
        t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 7.3), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.1),
                               ("BACKGROUND", (0, 0), (-1, 0), HEAD), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, LINE),
                               ("TOPPADDING", (0, 0), (-1, -1), 1.7), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.7)] + st))
        out += [t, Spacer(0, 6)]
    out.append(Paragraph(
        f"Forecasts re-made with the method used today and only the data available at the end of {made:%B %Y}. "
        "Miss: green up to 10%, amber 10–20%, red over 20% ( − = actual came in higher). Regenerated every month.",
        S["tl"]))
    return out + [PageBreak()]


def _miss_bg(v: float):
    """Diverging colour: blue = forecast too low (actual higher), orange = too high; darker = bigger miss."""
    if not np.isfinite(v):
        return colors.white
    steps = [(5, ("#eef4fb", "#fdf1e7")), (10, ("#cfe1f7", "#fbd9bf")), (20, ("#8fbbe9", "#f5ad7c")),
             (1e9, ("#2a78d6", "#e0662a"))]
    for lim, (lo, hi) in steps:
        if abs(v) <= lim:
            return colors.HexColor(lo if v < 0 else hi)


def _page_direction(res, S) -> List:
    """Latest vintages vs actuals: where the forecast has been running low or high, most recent on the right."""
    out = [_header("Direction check: our latest forecasts vs what happened", res, S), Spacer(0, 5)]
    any_ = False
    for metric in ("Received", "Submitted"):
        d = ins.vintage_matrix(res, metric=metric)
        if d.empty:
            continue
        any_ = True
        vcols = [c for c in d.columns if c not in ("Destination", "Level")]
        data = [[f"Applications {metric.lower()}: miss by forecast made in"] + [c.replace("->", "for") for c in vcols]]
        st = []
        for i, r in enumerate(d.itertuples(index=False), start=1):
            vals = [getattr(r, c) if False else r[2 + j] for j, c in enumerate(vcols)]
            data.append([r[0]] + [_pct(v) for v in vals])
            for j, v in enumerate(vals, start=1):
                st.append(("BACKGROUND", (j, i), (j, i), _miss_bg(v)))
                if np.isfinite(v) and abs(v) > 20:
                    st.append(("TEXTCOLOR", (j, i), (j, i), colors.white))
            if r[1] == "Organisation":
                st.append(("FONT", (0, i), (-1, i), "Helvetica-Bold", 7.3))
        w0 = 7.0
        t = Table(data, colWidths=[w0 * cm] + [(FULL_W - w0) / max(len(vcols), 1) * cm] * len(vcols), repeatRows=1)
        t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 7.3), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.1),
                               ("BACKGROUND", (0, 0), (-1, 0), HEAD), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                               ("LINEBELOW", (0, 0), (-1, -1), 0.25, LINE), ("TOPPADDING", (0, 0), (-1, -1), 1.7),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 1.7)] + st))
        out += [t, Spacer(0, 6)]
    if not any_:
        return []
    out.append(Paragraph(
        "Each column is one forecast, re-made with today's method and only the data available when it was made, "
        "compared with the months that have since happened (first three columns: a full quarter; last two: the "
        "latest two months and the latest month). Blue = actual came in higher than forecast, orange = lower; darker "
        "= bigger miss. A colour change towards the right is an early sign of a turning point; the latest columns "
        "feed the direction signal applied to this month's receipts forecast.", S["tl"]))
    return out + [PageBreak()]


def _page_destinations(res, S) -> List:
    ct = ins.country_table(res)
    half = FULL_W / 2 - 0.15
    return [_header("Destinations at a glance — applications received", res, S), Spacer(0, 5),
            _row([_img(charts.destination_bars(ct, "Received", "Next 12 months by destination", size=(6.4, 3.3)), half),
                  _img(charts.growth_check(ct, "Received", "Growth: forecast vs what the market is doing now",
                                           size=(6.4, 3.3)), half)], [half + 0.3, half]),
            Spacer(0, 4), _destination_table(res),
            Paragraph("* Market now = actual applications received in the last 6 months vs the same 6 months a year "
                      f"earlier. Shaded orange = differs from the forecast growth by more than {GROWTH_GAP_PP:.0f} points: "
                      "confirm the assumption with the market team.", S["tl"]), PageBreak()]


def _page_workload(res, S) -> List:
    st = getattr(res, "staffing", None)
    if st is None or st.empty:
        return []
    out = [_header("Workload calendar — next 12 months", res, S), Spacer(0, 4)]
    for m in ("Received", "Submitted"):
        out.append(_img(charts.workload_heatmap(
            st, m, f"Applications {m.lower()} per month.  Blue = busier than that destination's average month, "
                   "grey = quieter; bold = at least 10% busier.", size=(13.2, 3.55)), FULL_W))
    return out + [PageBreak()]


def _page_track(res, S) -> List:
    fva = getattr(res, "forecast_vs_actual", None)
    if fva is None or fva.empty:
        return []
    r12 = _tot(res, "Organisation", ORG, "Received", "Next 12 months")
    err = r12.get("Typical_Quarter_Error_Pct", np.nan) if r12 else np.nan
    cov, cov_c = _tested_coverage(res, "Organisation"), _tested_coverage(res, "Country")
    bias = ins.overall_bias(res, "Received")      # Forecast - Actual, % of actual
    w = (FULL_W - 0.3) / 4
    tiles = [_tile("Typical miss on an organisation quarter", f"±{err:.0f}%" if np.isfinite(err) else "–",
                   [f"tested on {ins.wf_period(res)}, forecasts made as if at the time"], S, w),
             _tile("Plan-for range held — organisation", f"{cov:.0f}%" if np.isfinite(cov) else "–",
                   ["of test quarters (aim 80%)"], S, w),
             _tile("Plan-for range held — destinations", f"{cov_c:.0f}%" if np.isfinite(cov_c) else "–",
                   ["of test quarters (aim 80%)"], S, w),
             _tile("Actual vs forecast on average", _pct(-bias) if np.isfinite(bias) else "–",
                   ["+ = actuals came in above the forecast"], S, w)]
    half = FULL_W / 2 - 0.15
    ch = [_img(charts.track_record(fva, "Organisation", ORG, m, f"Organisation — applications {m.lower()}",
                                   size=(6.4, 3.0)), half) for m in ("Received", "Submitted")]
    return [_header("Track record — what we forecast and what actually happened", res, S), Spacer(0, 5),
            _row(tiles, [w + 0.08] * 4), Spacer(0, 6), _row(ch, [half + 0.3, half]), Spacer(0, 6),
            _health_table(res), PageBreak()]


WIDE_RANGE = 1.0          # planning range wider than the forecast itself -> direction only
GROWTH_GAP_PP = 25.0      # forecast vs recent growth differ by more than this -> confirm with the market


STATUS_BG = {"Pass": colors.HexColor("#2e9e5b"), "Watch": colors.HexColor("#e0a100"),
             "Fail": colors.HexColor("#d6453d"), "Not yet": colors.HexColor("#9a9a95")}


def _health_table(res) -> Table:
    """Fixed acceptance checks: tested on past months, and live on published forecasts as actuals arrive."""
    h = ins.health_check(res)
    data = [["Health check (same checks every month)", "Target", "Tested on past months", "", "Live forecasts", ""]]
    st = []
    for i, r in enumerate(h.itertuples(), start=1):
        data.append([r.Check, r.Target, r.Tested, r.Tested_Status, r.Live, r.Live_Status])
        for col, stt in ((3, r.Tested_Status), (5, r.Live_Status)):
            st += [("BACKGROUND", (col, i), (col, i), STATUS_BG.get(stt, STATUS_BG["Not yet"])),
                   ("TEXTCOLOR", (col, i), (col, i), colors.white)]
    t = Table(data, colWidths=[x * cm for x in (7.6, 5.4, 4.0, 1.9, 4.2, 1.9)])
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 7.4), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.4),
                           ("BACKGROUND", (0, 0), (-1, 0), HEAD), ("SPAN", (2, 0), (3, 0)), ("SPAN", (4, 0), (5, 0)),
                           ("ALIGN", (2, 0), (-1, -1), "CENTER"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, LINE),
                           ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2)] + st))
    return t


def _page_country(res, S, country: str, label: str) -> List:
    story = _page_dashboard(res, S, "Country", country, label, chart_h=2.45)
    half = FULL_W / 2 - 0.15
    tot = res.totals[res.totals["Country"] == country]
    n_reg = tot[(tot["Level"] == "Region") & (tot["Metric"] == "Received")]["Region"].nunique()
    cells = []
    if n_reg > 1:
        cells.append(_img(charts.region_bars(tot, "Received", "Regions — received, next 12 months",
                                             size=(6.4, 2.2)), half))
    fva = getattr(res, "forecast_vs_actual", None)
    if fva is not None and len(fva) and ((fva["Level"] == "Country") & (fva["Country"] == country)).any():
        cells.append(_img(charts.track_record(fva, "Country", country, "Received", "Track record — received",
                                              size=(6.4, 2.2)), half))
    if cells:
        cells += [""] * (2 - len(cells))
        story += [Spacer(0, 4), _row(cells, [half + 0.3, half])]
    return story


def _page_legend(res, S) -> List:
    items = [
        ("Forecast", "The single most likely number, from the method that tested most accurate for that series."),
        ("Most likely range", "The actual should land inside it about 1 time in 2 (dark band on charts)."),
        ("Plan for range", "The actual should land inside it about 8 times in 10 (light band). Use the upper end for "
                           "capacity and the lower end for commitments."),
        ("Reliability", "Typical miss on a quarter total in testing:  High up to 10%   ·   Medium 10–20%   ·   "
                        "Low over 20%   ·   New = too little history."),
        ("Market now", "Actual growth in the last 6 months. If it differs a lot from the forecast growth, the forecast "
                       "is assuming a change of pace — confirm with the market team."),
        ("Tested", "Every accuracy figure comes from forecasts re-made for past months using only the data "
                   "available at the time, then compared with what actually happened."),
        ("More detail", "Regions, monthly figures, staffing and all evidence: RMG_Forecast workbook (Excel)."),
    ]
    data = [[Paragraph(f"<b>{a}</b>", S["legend"]), Paragraph(b, S["legend"])] for a, b in items]
    t = Table(data, colWidths=[4.5 * cm, (FULL_W - 4.5) * cm])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.3, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return [_header("How to read this pack", res, S), Spacer(0, 8), t]


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 6.8)
    canvas.setFillColor(INK2)
    canvas.drawString(MARGIN, 0.5 * cm, "Dark band = most likely (1 in 2)   ·   light band = plan for (8 in 10)   ·   "
                                        "Reliability = typical miss on a quarter total: High up to 10%, Medium 10-20%, "
                                        "Low over 20%")
    canvas.drawRightString(PAGE_W - MARGIN, 0.5 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _doc(path: Path, title: str) -> SimpleDocTemplate:
    return SimpleDocTemplate(str(path), pagesize=landscape(A4), leftMargin=MARGIN, rightMargin=MARGIN,
                             topMargin=0.9 * cm, bottomMargin=1.0 * cm, title=title, author="RMG forecasting pipeline")


def _countries(res) -> List[tuple]:
    t = res.totals[(res.totals["Level"] == "Country") & (res.totals["Metric"] == "Received")
                   & (res.totals["Window"] == "Next 12 months")].sort_values("Forecast", ascending=False)
    return list(zip(t["Country"], t["Label"]))


def write_brief(res, path: Path) -> Path:
    """The monthly pack: organisation, destinations, workload, track record, one page per destination."""
    S = _styles()
    story: List = _page_org(res, S) + _page_proof(res, S) + _page_direction(res, S) + _page_destinations(res, S) + _page_workload(res, S) + _page_track(res, S)
    for c, lab in _countries(res):
        story += _page_country(res, S, c, lab) + [PageBreak()]
    story += _page_legend(res, S)
    _doc(path, "Application Flow Forecast — Monthly Pack").build(story, onFirstPage=_footer, onLaterPages=_footer)
    return path


def write_country_pack(res, country: str, label: str, path: Path) -> Path:
    S = _styles()
    story = _page_country(res, S, country, label) + [PageBreak()] + _page_legend(res, S)
    _doc(path, f"Application Flow Forecast — {label}").build(story, onFirstPage=_footer, onLaterPages=_footer)
    return path


def write_country_packs(res, folder: Path) -> Dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for c, lab in _countries(res):
        safe = "".join(ch if ch.isalnum() else "_" for ch in lab).strip("_")
        # short names: the run folder already carries the date, and Windows paths stop at 260 characters
        out[lab] = write_country_pack(res, c, lab, folder / f"{safe[:28]}.pdf")
    return out
