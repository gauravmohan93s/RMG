# -*- coding: utf-8 -*-
"""
Live country input sheets: shared once, kept up to date by each destination team, read at every run.

  Forms/Live/<Country>.xlsx
    Market View   region x month grid. For every region (and the destination total) three rows:
                    Actual     the last 3 months
                    Model      the forecast for the next 12 months
                    Your view  yellow cells: the team's expected volume where it differs from the model
                  then Reason / driver, Name, Last reviewed. Blocks: applications received, applications
                  submitted, and the pipeline after submission (offers, deposits, visa... destination total).
                  Orange / red = more than 15% / 30% away from the model; an orange date = not reviewed for
                  90 days.
    Staffing      region x stage grid: for Received, Submitted, Offers, Deposits and Visa applied, the
                  last-12-month volume per month, the team's FTE on that stage (yellow; a person's time can be
                  split, e.g. 0.5 + 0.5) and applications per FTE (calculated).
    Read me

  sync_live(folder, settings_file)   before each forecast: reads every sheet into Settings.xlsx
                                     (Human_Inputs, Staffing_Productivity) for those destinations.
  refresh_live(res, folder)          after each run: rebuilds every sheet on the new 12-month window with the
                                     new model numbers, carrying over everything the team entered.

The window rolls forward each month; past months keep the team's entries (locked) next to what actually
happened. A file that is open (locked) or unreadable is never overwritten: it is skipped and reported. Each
refresh first copies the previous file to Live/_history.
"""
from __future__ import annotations

import os
import shutil
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell

from ..data.prepare import METRICS, TOTAL

HEADER_ROW = 6                     # 0-based row of the table header in Market View and Staffing
TOTAL_LABEL = "All regions (total)"
PAST = 3                           # actual months shown before the forecast window
DIFF_WATCH, DIFF_FLAG = 0.15, 0.30
STALE_DAYS = 90
NOTE_COLS = ["Reason / driver", "Name", "Last reviewed"]
STAFF_STAGES = [("Received", "Received"), ("Submitted", "Submitted"), ("Offer", "Offers"), ("Deposit", "Deposits"),
                ("Visa_Applied", "Visa applied")]
PIPELINE = ["Offer", "Deposit", "Visa_Document", "Visa_Applied", "Visa_Granted", "Visa_Refused", "Enrolled", "Deferred"]
STAGE_TITLE = {"Offer": "Offers", "Deposit": "Deposits", "Visa_Document": "Visa document", "Visa_Applied": "Visa applied",
               "Visa_Granted": "Visa granted", "Visa_Refused": "Visa refused", "Enrolled": "Enrolled", "Deferred": "Deferred"}
SETTINGS_STAFF = ["Country", "Region", "Metric", "Apps_Per_FTE_Month", "Owner", "Notes"]
SETTINGS_HUMAN = ["Country", "Region", "Metric", "Month", "Expected_Volume", "Submitted_By", "Reason", "Submitted_On"]
METRIC_TITLE = {"Received": "Applications received", "Submitted": "Applications submitted"}

README = [
    "Your destination's view of the coming months. Keep it up to date whenever you learn something; the forecast "
    "reads it at every monthly run. There is no need to send it back.",
    "Market View: for each region, 'Model' is the forecast. Type your expected number in the yellow 'Your view' cell "
    "only where you expect something different, with the reason, your name and the date in 'Last reviewed'. "
    "Leave it blank where you agree with the model. The last block covers the stages after submission (offers, "
    "deposits, visas) for the destination as a whole.",
    "Colours: orange = your view is more than 15% away from the model, red = more than 30% (please make sure the "
    "reason explains it). An orange date = not reviewed for 90 days; views older than 120 days stop counting.",
    "Staffing: for each region and stage, enter how many full-time people (FTE) work on it today. Split a person's "
    "time if they cover several stages (e.g. 0.5 offers + 0.5 deposits). Applications per FTE are calculated from "
    "the last 12 months and turn the forecast into staffing needs.",
    "Every month the model rows are refreshed and the months roll forward; your entries are kept. Past months stay "
    "visible (locked) next to what actually happened, so you can see how close your view was.",
    "Every view is scored against what happens. After 6 scored months, views from people who beat the model are "
    "blended into the forecast automatically.",
    "Please do not add rows or columns, rename sheets, rename or move this file.",
]


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")[:40]


def _mlabel(d) -> str:
    return f"{pd.Timestamp(d):%b %Y}"


def _mkey(label) -> str:
    """'Nov 2026' (or a date the user's Excel turned it into) -> '2026-11'; '' when not a month."""
    if isinstance(label, (pd.Timestamp, date)):
        return f"{pd.Timestamp(label):%Y-%m}"
    ts = pd.to_datetime(str(label).strip(), format="%b %Y", errors="coerce")
    return "" if pd.isna(ts) else f"{ts:%Y-%m}"


def _as_date(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip()):
        return None
    ts = pd.to_datetime(v, errors="coerce", dayfirst=True)
    return None if pd.isna(ts) else ts.to_pydatetime()


def _text(v) -> str:
    return "" if v is None or pd.isna(v) or str(v).strip().lower() == "nan" else str(v).strip()


def _num(v) -> float:
    x = pd.to_numeric(v, errors="coerce")
    return float(x) if np.isfinite(x) else np.nan


def _staff_cols(label: str) -> Tuple[str, str, str]:
    return f"{label} / month", f"{label} FTE", f"{label} per FTE"


def load_stage_actuals(stage_file: Optional[Path]) -> pd.DataFrame:
    """stage_final_summary.xlsx -> Country, Region, StageGroup, Date, Apps (empty when absent)."""
    if stage_file is None or not Path(stage_file).exists():
        return pd.DataFrame(columns=["Country", "Region", "StageGroup", "Date", "Apps"])
    d = pd.read_excel(stage_file, dtype={"Month": str})
    d = d.rename(columns={"Country Grouped": "Country", "Name of Region": "Region"})
    d["Date"] = pd.to_datetime(d["Month"].str[:7] + "-01", errors="coerce")
    d["Apps"] = pd.to_numeric(d["Apps"], errors="coerce").fillna(0.0)
    return d.dropna(subset=["Date"])[["Country", "Region", "StageGroup", "Date", "Apps"]]


# ─────────────────────────────── data for one destination ───────────────────────────────
def _country_data(res, country: str, stages: pd.DataFrame):
    h = res.history.copy()
    h["Date"] = pd.to_datetime(h["Date"])
    f = res.forecast.copy()
    f["Date"] = pd.to_datetime(f["Date"])
    last = pd.Timestamp(res.last_actual)
    past = list(pd.date_range(last - pd.DateOffset(months=PAST - 1), periods=PAST, freq="MS"))
    fut = sorted(pd.Timestamp(d) for d in f.loc[f["Horizon"].between(1, 12), "Date"].unique())
    hc = h[(h["Country"] == country) & (h["Level"] == "Region")]
    fr = f[f["Country"] == country]
    recent = hc[hc["Date"] > last - pd.DateOffset(months=12)]
    regions = (recent[recent["Metric"] == METRICS[0]].groupby("Region")["Actual"].sum()       # largest first
               .reindex(hc["Region"].unique(), fill_value=0).sort_values(ascending=False).index.tolist())
    blocks = []                          # (title, [(region label, metric, actual by month, model by month)])
    for m in METRICS:
        rows = [(TOTAL_LABEL, m, hc[hc["Metric"] == m].groupby("Date")["Actual"].sum(),
                 fr[(fr["Level"] == "Country") & (fr["Region"] == TOTAL) & (fr["Metric"] == m)].groupby("Date")["Forecast"].sum())]
        if len(regions) > 1:
            for r in regions:
                rows.append((r, m, hc[(hc["Metric"] == m) & (hc["Region"] == r)].groupby("Date")["Actual"].sum(),
                             fr[(fr["Level"] == "Region") & (fr["Region"] == r) & (fr["Metric"] == m)]
                             .groupby("Date")["Forecast"].sum()))
        blocks.append((METRIC_TITLE[m], rows))
    sc = stages[stages["Country"] == country]
    sf = getattr(res, "stage_forecast", None)
    sf = sf[(sf["Country"] == country) & (sf["Level"] == "Country")] if sf is not None and len(sf) else pd.DataFrame()
    pipe = []
    for s in PIPELINE:
        mod = (sf[sf["StageGroup"] == s].assign(Date=lambda x: pd.to_datetime(x["Date"])).groupby("Date")["Forecast"].sum()
               if len(sf) else pd.Series(dtype=float))
        act = sc[sc["StageGroup"] == s].groupby("Date")["Apps"].sum()
        if len(mod) or act.sum() > 0:
            pipe.append((TOTAL_LABEL, s, act, mod))
    if pipe:
        blocks.append(("Pipeline after submission — destination total", pipe))
    avg = recent.groupby(["Region", "Metric"])["Actual"].sum().div(12).to_dict()
    srec = sc[sc["Date"] > last - pd.DateOffset(months=12)]
    avg.update(srec.groupby(["Region", "StageGroup"])["Apps"].sum().div(12).to_dict())
    staff_stages = [(k, lab) for k, lab in STAFF_STAGES if k in METRICS or (srec["StageGroup"] == k).any()]
    staff = [(r, {k: float(avg.get((r, k), 0.0)) for k, _ in staff_stages}) for r in regions]
    return past, fut, blocks, staff_stages, staff


# ─────────────────────────────── reading a live file ───────────────────────────────
def _read_raw(path: Path) -> dict:
    """Everything the team entered: views by (region, metric, 'YYYY-MM'), notes by (region, metric), staffing by
    region ({"fte": {stage: FTE}, "avg": {stage: per month}, "notes", "name", "date"}). Raises if unreadable."""
    with pd.ExcelFile(path) as x:
        key = pd.read_excel(x, "_key", header=None) if "_key" in x.sheet_names else pd.DataFrame()
        mv = pd.read_excel(x, "Market View", header=HEADER_ROW)
        st = pd.read_excel(x, "Staffing", header=HEADER_ROW)
    out = {"country": _text(key.iloc[0, 1]) if len(key) else "", "label": _text(key.iloc[1, 1]) if len(key) > 1 else "",
           "views": {}, "notes": {}, "staff": {}}
    mv = mv[mv["Row"].astype(str).str.strip() == "Your view"]
    months = {c: _mkey(c) for c in mv.columns}
    for vals in mv.itertuples(index=False, name=None):
        row = dict(zip(mv.columns, vals))
        reg, met = _text(row["Region"]), _text(row["Metric"])
        if met not in METRICS:                    # pipeline rows show the stage name; they are destination totals
            reg = TOTAL_LABEL
        for c, k in months.items():
            v = _num(row[c]) if k else np.nan
            if k and np.isfinite(v):
                out["views"][(reg, met, k)] = v
        note = tuple(row.get(c) for c in NOTE_COLS)
        if _text(note[0]) or _text(note[1]) or _as_date(note[2]):
            out["notes"][(reg, met)] = note
    for vals in st.itertuples(index=False, name=None):
        row = dict(zip(st.columns, vals))
        reg = _text(row.get("Region"))
        if not reg or reg.lower() == "total":
            continue
        fte, avg = {}, {}
        for k, lab in STAFF_STAGES:
            c_avg, c_fte, _ = _staff_cols(lab)
            if c_fte in row:
                fte[k], avg[k] = _num(row[c_fte]), _num(row.get(c_avg))
        if "Current FTE" in row:                                 # layout 1 (one FTE for receiving + submitting)
            fte["Received"] = _num(row["Current FTE"])
            avg["Received"] = _num(row.get("Received per month (last 12m)"))
        out["staff"][reg] = {"fte": fte, "avg": avg, "notes": row.get("Notes"), "name": row.get("Name"),
                             "date": row.get("Last reviewed")}
    return out


# ─────────────────────────────── writing a live file ───────────────────────────────
def _write(path: Path, country: str, label: str, past, fut, blocks, staff_stages, staff, carry: dict,
           refreshed: date, actual_to: pd.Timestamp) -> None:
    wb = xlsxwriter.Workbook(str(path))
    F = lambda **k: wb.add_format({"font_name": "Calibri", "font_size": 10, "valign": "vcenter", **k})
    edge = {"border": 1, "border_color": "#e3e6ea"}
    title, sub = F(bold=True, font_size=15, font_color="#1f3864"), F(font_color="#52514e", italic=True)
    head = F(bold=True, bg_color="#1f3864", font_color="white", border=1, border_color="#c9d3e3", align="center",
             text_wrap=True)
    head_l = F(bold=True, bg_color="#1f3864", font_color="white", border=1, border_color="#c9d3e3", text_wrap=True)
    head_past = F(bold=True, bg_color="#7f8c9d", font_color="white", border=1, border_color="#c9d3e3", align="center",
                  text_wrap=True)
    band = F(bold=True, font_size=11, bg_color="#dde6f3", font_color="#1f3864")
    lab, lab_soft = F(bold=True, top=1, top_color="#9fb0c8"), F(font_color="#9a9a95")
    kind = {"Actual": F(font_color="#52514e", top=1, top_color="#9fb0c8"), "Model": F(font_color="#1f3864"),
            "Your view": F(bold=True, font_color="#7a5c00")}
    actual = F(num_format="#,##0", bg_color="#eef1f5", font_color="#52514e", **edge)
    model = F(num_format="#,##0", font_color="#1f3864", **edge)
    blank = F(bg_color="#f6f6f4", **edge)
    yellow = {"bg_color": "#fff3b0", "border": 1, "border_color": "#d6c06a", "locked": False}
    inp = F(num_format="#,##0", **yellow)
    inp_txt = F(text_wrap=True, **yellow)
    inp_date = F(num_format="dd mmm yyyy", **yellow)
    past_view = F(num_format="#,##0", bg_color="#f3eed8", font_color="#7a5c00", **edge)
    warn, flag = wb.add_format({"bg_color": "#f7c58b"}), wb.add_format({"bg_color": "#ef8a80"})
    swatch = {"Your input": inp, "Actual": actual, "Model forecast": model,
              f"Over {DIFF_WATCH:.0%} from model": F(bg_color="#f7c58b"),
              f"Over {DIFF_FLAG:.0%} from model": F(bg_color="#ef8a80")}

    # ── Market View
    ws = wb.add_worksheet("Market View")
    months = list(past) + list(fut)
    c0 = 3                                   # first month column
    cn = c0 + len(months)                    # first note column
    ws.write(0, 0, f"Market view — {label}", title)
    ws.write(1, 0, f"Model refreshed {refreshed:%d %b %Y} · actuals to {actual_to:%B %Y} · "
                   "fill only the yellow cells · see 'Read me'", sub)
    ws.write(3, 0, "Key", F(bold=True))
    for j, (k, fm) in enumerate(swatch.items()):
        ws.merge_range(3, c0 + 2 * j, 3, c0 + 2 * j + 1, k, fm)
    ws.merge_range(5, c0, 5, c0 + len(past) - 1, "Last 3 months — actual", head_past)
    ws.merge_range(5, c0 + len(past), 5, cn - 1, "Next 12 months — forecast", head)
    for j, h in enumerate(["Region", "Metric", "Row"]):
        ws.write(HEADER_ROW, j, h, head_l)
    for j, d in enumerate(months):
        ws.write_string(HEADER_ROW, c0 + j, _mlabel(d), head_past if j < len(past) else head)
    for j, h in enumerate(NOTE_COLS):
        ws.write(HEADER_ROW, cn + j, h, head_l)
    r = HEADER_ROW + 1
    for block_title, rows in blocks:
        ws.merge_range(r, 0, r, cn + len(NOTE_COLS) - 1, block_title, band)
        r += 1
        for reg, met, act, mod in rows:
            shown = STAGE_TITLE.get(met, met)
            for i, row in enumerate(("Actual", "Model", "Your view")):
                ws.write(r + i, 0, reg if met in METRICS else shown, lab if i == 0 else lab_soft)
                ws.write(r + i, 1, met, lab_soft)
                ws.write(r + i, 2, row, kind[row])
            for j, d in enumerate(months):
                k, is_past = f"{d:%Y-%m}", j < len(past)
                a, mv = act.get(d, np.nan), mod.get(d, np.nan)
                if is_past and np.isfinite(a):
                    ws.write_number(r, c0 + j, round(float(a)), actual)
                else:
                    ws.write_blank(r, c0 + j, None, actual if is_past else blank)
                if not is_past and np.isfinite(mv):
                    ws.write_number(r + 1, c0 + j, round(float(mv)), model)
                else:
                    ws.write_blank(r + 1, c0 + j, None, blank)
                v = carry["views"].get((reg, met, k))
                fm = past_view if is_past else inp
                if v is not None:
                    ws.write_number(r + 2, c0 + j, v, fm)
                else:
                    ws.write_blank(r + 2, c0 + j, None, fm)
            reason, name, when = carry["notes"].get((reg, met), (None, None, None))
            for j, txt in ((0, _text(reason)), (1, _text(name))):
                ws.write_string(r + 2, cn + j, txt, inp_txt) if txt else ws.write_blank(r + 2, cn + j, None, inp_txt)
            d = _as_date(when)
            ws.write_datetime(r + 2, cn + 2, d, inp_date) if d else ws.write_blank(r + 2, cn + 2, None, inp_date)
            for j in range(len(NOTE_COLS)):
                ws.write_blank(r, cn + j, None, blank)
                ws.write_blank(r + 1, cn + j, None, blank)
            # checks and colours on the future cells of 'Your view', compared with the model cell above
            first, mcell = xl_rowcol_to_cell(r + 2, c0 + len(past)), xl_rowcol_to_cell(r + 1, c0 + len(past))
            rng = f"{first}:{xl_rowcol_to_cell(r + 2, cn - 1)}"
            ws.data_validation(rng, {"validate": "decimal", "criteria": ">=", "value": 0,
                                     "error_message": "Enter an expected number of applications (0 or more)."})
            ws.conditional_format(rng, {"type": "formula", "format": flag, "stop_if_true": True,
                                        "criteria": f"=AND(ISNUMBER({first}),{mcell}>0,ABS({first}/{mcell}-1)>{DIFF_FLAG})"})
            ws.conditional_format(rng, {"type": "formula", "format": warn,
                                        "criteria": f"=AND(ISNUMBER({first}),{mcell}>0,ABS({first}/{mcell}-1)>{DIFF_WATCH})"})
            dcell = xl_rowcol_to_cell(r + 2, cn + 2)
            ws.data_validation(dcell, {"validate": "date", "criteria": ">", "value": date(2020, 1, 1),
                                       "error_message": "Enter the date you last reviewed this row."})
            ws.conditional_format(dcell, {"type": "formula", "format": warn,
                                          "criteria": f"=AND(ISNUMBER({dcell}),TODAY()-{dcell}>{STALE_DAYS})"})
            r += 3
    ws.set_column(0, 0, 28)
    ws.set_column(1, 2, 10)
    ws.set_column(c0, cn - 1, 9.5)
    ws.set_column(cn, cn, 42)
    ws.set_column(cn + 1, cn + 1, 16)
    ws.set_column(cn + 2, cn + 2, 13)
    ws.set_row(HEADER_ROW, 30)
    ws.freeze_panes(HEADER_ROW + 1, c0)
    ws.protect("", {"format_columns": True, "format_rows": True})
    ws.activate()

    # ── Staffing: region x stage
    ss = wb.add_worksheet("Staffing")
    ss.write(0, 0, f"Staffing — {label}", title)
    ss.write(1, 0, "For each region and stage: how many full-time people (FTE) work on it today. Split a person's "
                   "time across stages if needed (e.g. 0.5 + 0.5). Applications per FTE are calculated.", sub)
    ss.write(HEADER_ROW, 0, "Region", head_l)
    num = F(num_format="#,##0", bg_color="#eef1f5", **edge)
    rate = F(num_format="#,##0.0", font_color="#1f3864", **edge)
    fte = F(num_format="0.0", **yellow)
    r0, cols = HEADER_ROW + 1, {}
    for j, (k, lab_) in enumerate(staff_stages):
        c = 1 + 3 * j
        cols[k] = c
        ss.merge_range(HEADER_ROW - 1, c, HEADER_ROW - 1, c + 2, lab_, head)
        for i, h in enumerate(_staff_cols(lab_)):
            ss.write(HEADER_ROW, c + i, h, head)
    cn2 = 1 + 3 * len(staff_stages)
    for j, h in enumerate(("Notes", "Name", "Last reviewed")):
        ss.write(HEADER_ROW, cn2 + j, h, head_l)
    for i, (reg, avgs) in enumerate(staff):
        r = r0 + i
        cr = carry["staff"].get(reg, {"fte": {}, "notes": None, "name": None, "date": None})
        ss.write(r, 0, reg, F(bold=True, **edge))
        for k, c in cols.items():
            a, v = avgs.get(k, 0.0), cr["fte"].get(k, np.nan)
            ss.write_number(r, c, round(a), num)
            ss.write_number(r, c + 1, float(v), fte) if np.isfinite(v) else ss.write_blank(r, c + 1, None, fte)
            D, A = xl_rowcol_to_cell(r, c + 1), xl_rowcol_to_cell(r, c)
            cached = round(a / v, 1) if np.isfinite(v) and v > 0 else ""
            ss.write_formula(r, c + 2, f'=IF(AND(ISNUMBER({D}),{D}>0),{A}/{D},"")', rate, cached)
            ss.data_validation(r, c + 1, r, c + 1, {"validate": "decimal", "criteria": ">=", "value": 0,
                                                    "error_message": "Enter the FTE (0 or more)."})
        for j, txt in ((0, _text(cr["notes"])), (1, _text(cr["name"]))):
            ss.write_string(r, cn2 + j, txt, inp_txt) if txt else ss.write_blank(r, cn2 + j, None, inp_txt)
        d = _as_date(cr["date"])
        ss.write_datetime(r, cn2 + 2, d, inp_date) if d else ss.write_blank(r, cn2 + 2, None, inp_date)
    rt = r0 + len(staff)
    ss.write(rt, 0, "Total", F(bold=True, top=1))
    for k, c in cols.items():
        for i in (0, 1):
            a, b = xl_rowcol_to_cell(r0, c + i), xl_rowcol_to_cell(rt - 1, c + i)
            ss.write_formula(rt, c + i, f"=SUM({a}:{b})", F(bold=True, num_format="#,##0.0" if i else "#,##0", top=1))
        D, A = xl_rowcol_to_cell(rt, c + 1), xl_rowcol_to_cell(rt, c)
        ss.write_formula(rt, c + 2, f'=IF(AND(ISNUMBER({D}),{D}>0),{A}/{D},"")', F(bold=True, num_format="#,##0.0", top=1))
    ss.set_column(0, 0, 28)
    ss.set_column(1, cn2 - 1, 11)
    ss.set_column(cn2, cn2, 36)
    ss.set_column(cn2 + 1, cn2 + 2, 15)
    ss.set_row(HEADER_ROW, 30)
    ss.freeze_panes(HEADER_ROW + 1, 1)
    ss.protect("", {"format_columns": True, "format_rows": True})

    # ── Read me, and the hidden key that ties the file to its destination
    rm = wb.add_worksheet("Read me")
    rm.write(0, 0, f"How to use this sheet — {label}", title)
    wrap = F(text_wrap=True, valign="top")
    for i, t in enumerate(README, start=2):
        rm.write(i, 0, f"{i - 1}.", F(bold=True, valign="top"))
        rm.write(i, 1, t, wrap)
        rm.set_row(i, 48)
    rm.set_column(0, 0, 4)
    rm.set_column(1, 1, 120)
    k = wb.add_worksheet("_key")
    for i, (a, b) in enumerate((("country", country), ("label", label), ("layout", "2"))):
        k.write_string(i, 0, a)
        k.write_string(i, 1, b)
    k.hide()
    wb.close()


# ─────────────────────────────── public ───────────────────────────────
def refresh_live(res, folder: Path, stage_file: Optional[Path] = None) -> pd.DataFrame:
    """Create or roll forward every destination's live sheet, keeping the team's entries."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stages = load_stage_actuals(stage_file)
    t = res.totals
    countries = t[(t["Level"] == "Country") & (t["Metric"] == "Received")][["Country", "Label"]].drop_duplicates()
    log = []
    for country, label in countries.itertuples(index=False):
        p = folder / f"{_safe(label)}.xlsx"
        carry, existed = {"views": {}, "notes": {}, "staff": {}}, p.exists()
        if existed:
            try:
                carry = _read_raw(p)
            except Exception as exc:                             # noqa: BLE001 - never overwrite what we cannot read
                log.append({"Destination": label, "Status": "skipped", "Note": f"unreadable, left as is: {exc}"})
                continue
            (folder / "_history").mkdir(exist_ok=True)
            shutil.copy2(p, folder / "_history" / f"{p.stem}_{date.today():%Y%m%d}.xlsx")
        past, fut, blocks, staff_stages, staff = _country_data(res, country, stages)
        tmp = p.with_name(p.stem + ".tmp.xlsx")
        _write(tmp, country, label, past, fut, blocks, staff_stages, staff, carry, date.today(),
               pd.Timestamp(res.last_actual))
        try:
            os.replace(tmp, p)
        except PermissionError:
            tmp.unlink(missing_ok=True)
            log.append({"Destination": label, "Status": "skipped", "Note": "file open / locked: refreshed next run"})
            continue
        n_fte = sum(1 for v in carry["staff"].values() for x in v["fte"].values() if np.isfinite(x))
        log.append({"Destination": label, "Status": "refreshed" if existed else "created",
                    "Note": f"{len(carry['views'])} views and {n_fte} FTE entries kept" if existed else ""})
    return pd.DataFrame(log, columns=["Destination", "Status", "Note"])


def read_live(folder: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(Human_Inputs rows, Staffing_Productivity rows, log) from every live sheet in folder. Staffing: one rate
    per region x stage, plus a destination rate (Region '*') per stage after submission, where the stage
    forecast is made for the destination as a whole."""
    human, staff, log = [], [], []
    for p in sorted(Path(folder).glob("*.xlsx")):
        if p.name.startswith("~$") or p.name.endswith(".tmp.xlsx"):
            continue
        try:
            raw = _read_raw(p)
            if not raw["country"]:
                raise ValueError("no _key sheet: not a live sheet")
        except Exception as exc:                                 # noqa: BLE001 - report and continue
            log.append({"File": p.name, "Country": "", "Views": 0, "FTE_Rows": 0, "Note": f"unreadable: {exc}"})
            continue
        country, team = raw["country"], f"{raw['label'] or raw['country']} team"
        n_views = 0
        for (reg, met, k), v in raw["views"].items():
            if v < 0:
                continue
            reason, name, when = raw["notes"].get((reg, met), (None, None, None))
            d = _as_date(when)
            human.append({"Country": country, "Region": "" if reg == TOTAL_LABEL else reg, "Metric": met, "Month": k,
                          "Expected_Volume": v, "Submitted_By": _text(name) or team, "Reason": _text(reason),
                          "Submitted_On": (d.date() if d else date.today()).isoformat()})
            n_views += 1
        n_fte, pooled = 0, {}
        for reg, e in raw["staff"].items():
            for k, fte in e["fte"].items():
                a = e["avg"].get(k, np.nan)
                if not (np.isfinite(fte) and fte > 0 and np.isfinite(a) and a > 0):
                    continue
                n_fte += 1
                staff.append({"Country": country, "Region": reg, "Metric": k, "Apps_Per_FTE_Month": round(a / fte, 1),
                              "Owner": _text(e["name"]), "Notes": f"live sheet: FTE {fte:g}. {_text(e['notes'])}".strip()})
                pooled.setdefault(k, [0.0, 0.0])
                pooled[k][0] += a
                pooled[k][1] += fte
        for k, (a, fte) in pooled.items():
            if k not in METRICS:
                staff.append({"Country": country, "Region": "*", "Metric": k, "Apps_Per_FTE_Month": round(a / fte, 1),
                              "Owner": "", "Notes": f"live sheet: destination total, FTE {fte:g}"})
        log.append({"File": p.name, "Country": country, "Views": n_views, "FTE_Rows": n_fte, "Note": ""})
    return (pd.DataFrame(human, columns=SETTINGS_HUMAN), pd.DataFrame(staff, columns=SETTINGS_STAFF),
            pd.DataFrame(log, columns=["File", "Country", "Views", "FTE_Rows", "Note"]))


def sync_live(folder: Path, settings_file: Path) -> pd.DataFrame:
    """Write the live sheets into Settings.xlsx: for every destination with a readable live sheet, its
    Human_Inputs and Staffing_Productivity rows are replaced by the sheet's content (others are kept).
    Settings.xlsx is backed up once a day before the first write."""
    settings_file = Path(settings_file)
    human, staff, log = read_live(folder)
    live = set(log.loc[log["Note"] == "", "Country"])
    if not live or not settings_file.exists():
        return log
    with pd.ExcelFile(settings_file) as xl:
        existing = {s: pd.read_excel(xl, s) for s in ("Staffing_Productivity", "Human_Inputs") if s in xl.sheet_names}
    sheets = {}
    for sheet, new, cols in (("Staffing_Productivity", staff, SETTINGS_STAFF), ("Human_Inputs", human, SETTINGS_HUMAN)):
        old = existing.get(sheet, pd.DataFrame(columns=cols))
        for c in cols:
            if c not in old.columns:
                old[c] = np.nan
        old = old[~old["Country"].astype(str).isin(live)]
        parts = [x[cols] for x in (old, new) if len(x)]
        sheets[sheet] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    backup = settings_file.with_name(f"{settings_file.stem}_backup_{date.today():%Y%m%d}{settings_file.suffix}")
    if not backup.exists():
        shutil.copy2(settings_file, backup)
    with pd.ExcelWriter(settings_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        for sheet, df in sheets.items():
            df.to_excel(w, sheet_name=sheet, index=False)
    return log
