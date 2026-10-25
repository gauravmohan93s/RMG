# -*- coding: utf-8 -*-
"""
Country input forms: send one Excel file per destination, collect them back into Settings.xlsx.

  make_forms(res, folder)          -> folder/To_Send/<Country>.xlsx, one per destination, each with
                                        Instructions | Staffing | Market_Inputs
  collect_forms(folder, settings)  reads folder/Returned/*.xlsx, validates every row, backs up
                                        Settings.xlsx and writes sheets Staffing_Productivity and Human_Inputs.

Staffing: a country lead gives either Apps_Per_FTE_Month directly, or only Current_FTE (current
headcount handling that region). In the second case the rate is derived as average monthly volume over the
last 12 months / FTE, and the row is marked as derived. The forecast then turns volumes into FTE needs per
region (analysis/seasonality.py).

Market inputs: the lead's expected volume where they expect something different from the model, with a
reason. These are recorded, scored against what happens and blended in only for people who have beaten the
model for 6 scored months (analysis/signals.py).
"""
from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..data.prepare import TOTAL

STAFF_COLS = ["Country", "Region", "Metric", "Avg_Monthly_Last_12m", "Current_FTE", "Apps_Per_FTE_Month",
              "Owner", "Notes"]
INPUT_COLS = ["Country", "Region", "Metric", "Month", "Model_Forecast", "Plan_Low", "Plan_High",
              "Expected_Volume", "Submitted_By", "Reason", "Submitted_On"]
STAFF_EDIT = ("Current_FTE", "Apps_Per_FTE_Month", "Owner", "Notes")
INPUT_EDIT = ("Expected_Volume", "Submitted_By", "Reason", "Submitted_On")
SETTINGS_STAFF = ["Country", "Region", "Metric", "Apps_Per_FTE_Month", "Owner", "Notes"]
SETTINGS_HUMAN = ["Country", "Region", "Metric", "Month", "Expected_Volume", "Submitted_By", "Reason", "Submitted_On"]
NUMERIC_SHOWN = ("Avg_Monthly_Last_12m", "Model_Forecast", "Plan_Low", "Plan_High")


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")[:40]


def _staff_rows(res, country: str) -> pd.DataFrame:
    h = res.history
    r = h[(h["Level"] == "Region") & (h["Country"] == country)]
    last = pd.Timestamp(res.last_actual)
    r = r[pd.to_datetime(r["Date"]) > last - pd.DateOffset(months=12)]
    avg = r.groupby(["Region", "Metric"])["Actual"].sum().div(12).reset_index()
    avg = avg[avg["Actual"] > 0].sort_values(["Metric", "Actual"], ascending=[True, False])
    return pd.DataFrame({"Country": country, "Region": avg["Region"].to_numpy(), "Metric": avg["Metric"].to_numpy(),
                         "Avg_Monthly_Last_12m": avg["Actual"].round(0).to_numpy(),
                         **{c: None for c in STAFF_EDIT}})[STAFF_COLS]


def _input_rows(res, country: str, months: int = 3) -> pd.DataFrame:
    fc = res.forecast
    f = fc[(fc["Country"] == country) & (fc["Horizon"] <= months)]
    tot = f[(f["Level"] == "Country") & (f["Region"] == TOTAL)]
    reg = f[(f["Level"] == "Region") & (f["Metric"] == "Received")]
    if tot.empty:                      # single-region destination: its region is the total
        tot = f[f["Level"] == "Region"]
        reg = reg.iloc[0:0]
    x = pd.concat([tot.assign(_o=0), reg.assign(_o=1)]).sort_values(["_o", "Metric", "Region", "Date"])
    return pd.DataFrame({
        "Country": country, "Region": np.where(x["_o"] == 0, "", x["Region"]), "Metric": x["Metric"].to_numpy(),
        "Month": [f"{d:%Y-%m}" for d in pd.to_datetime(x["Date"])],
        "Model_Forecast": x["Forecast"].round(-1).to_numpy(), "Plan_Low": x["Lo80"].round(-1).to_numpy(),
        "Plan_High": x["Hi80"].round(-1).to_numpy(), **{c: None for c in INPUT_EDIT}})[INPUT_COLS]


INSTRUCTIONS = [
    ("What this is", "Two short inputs for the application-flow forecast for {country}. Please fill only the "
                     "yellow cells and return this file to the forecasting team by {due}."),
    ("Sheet 'Staffing'", "For each region: how many applications one counsellor handles in a typical month "
                         "(Apps_Per_FTE_Month). If you do not know the rate, enter only Current_FTE (today's "
                         "headcount working on that region) and the rate will be calculated from the last 12 months' "
                         "volume shown. Owner = your name. Used to turn forecast volumes into staffing needs."),
    ("Sheet 'Market_Inputs'", "The model's forecast for the next 3 months with its planning range (8 in 10). Where "
                              "you expect something DIFFERENT, enter your Expected_Volume, your name, the reason (new "
                              "partner, visa change, campaign...) and today's date. Leave rows blank where you agree "
                              "with the model."),
    ("How inputs are used", "Every input is recorded and compared with what actually happens. After 6 months, "
                            "inputs from people whose numbers beat the model are blended into the forecast "
                            "automatically. Region blank = the whole destination."),
    ("Do not", "Change column names, add sheets or rename the file."),
]


def _write_country(path: Path, label: str, staff: pd.DataFrame, inputs: pd.DataFrame, due: str) -> None:
    with pd.ExcelWriter(path, engine="xlsxwriter") as w:
        wb = w.book
        bold = wb.add_format({"bold": True, "font_size": 13})
        wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        hb = wb.add_format({"bold": True, "valign": "top"})
        fill = wb.add_format({"bg_color": "#fff3b0", "border": 1, "border_color": "#d0d7e2"})
        num = wb.add_format({"num_format": "#,##0", "font_color": "#52514e"})
        ws = wb.add_worksheet("Instructions")
        ws.write(0, 0, f"Forecast inputs — {label}", bold)
        for i, (a, b) in enumerate(INSTRUCTIONS, start=2):
            ws.write(i, 0, a, hb)
            ws.write(i, 1, b.format(country=label, due=due), wrap)
            ws.set_row(i, 48)
        ws.set_column(0, 0, 22)
        ws.set_column(1, 1, 110)
        for name, df, edit in (("Staffing", staff, STAFF_EDIT), ("Market_Inputs", inputs, INPUT_EDIT)):
            df.to_excel(w, sheet_name=name, index=False)
            s = w.sheets[name]
            for j, c in enumerate(df.columns):
                s.set_column(j, j, max(12, min(30, len(c) + 4)), num if c in NUMERIC_SHOWN else None)
                if c in edit:
                    for r in range(len(df)):
                        s.write_blank(r + 1, j, None, fill)
                if c in ("Current_FTE", "Apps_Per_FTE_Month", "Expected_Volume") and len(df):
                    s.data_validation(1, j, len(df), j, {"validate": "decimal", "criteria": ">=", "value": 0,
                                                         "error_message": "Enter a number of 0 or more."})
            s.freeze_panes(1, 0)


def make_forms(res, folder: Path, due_days: int = 7) -> Dict[str, Path]:
    """One file per destination in folder/To_Send; an empty folder/Returned for the collected files."""
    out_dir = Path(folder) / "To_Send"
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path(folder) / "Returned").mkdir(parents=True, exist_ok=True)
    due = (pd.Timestamp(date.today()) + pd.Timedelta(days=due_days)).strftime("%d %b %Y")
    t = res.totals
    countries = t[(t["Level"] == "Country") & (t["Metric"] == "Received")][["Country", "Label"]].drop_duplicates()
    out = {}
    for c, lab in countries.itertuples(index=False):
        p = out_dir / f"{_safe(lab)}.xlsx"
        _write_country(p, lab, _staff_rows(res, c), _input_rows(res, c), due)
        out[lab] = p
    return out


def _read_returned(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.ExcelFile(path)
    staff = pd.read_excel(x, "Staffing") if "Staffing" in x.sheet_names else pd.DataFrame(columns=STAFF_COLS)
    inp = (pd.read_excel(x, "Market_Inputs", dtype={"Month": str}) if "Market_Inputs" in x.sheet_names
           else pd.DataFrame(columns=INPUT_COLS))
    x.close()
    for c in STAFF_COLS:
        if c not in staff.columns:
            staff[c] = np.nan
    for c in INPUT_COLS:
        if c not in inp.columns:
            inp[c] = np.nan
    return staff, inp


def collect_forms(folder: Path, settings_file: Path) -> pd.DataFrame:
    """Validate returned files and write Staffing_Productivity + Human_Inputs into Settings.xlsx (backed up
    first). A country that returned a file replaces its own staffing rows; other countries are kept.
    Market inputs are appended (the pipeline keeps the latest per series and month)."""
    ret = sorted((Path(folder) / "Returned").glob("*.xlsx"))
    log: List[dict] = []
    staff_rows, input_rows = [], []
    for p in ret:
        try:
            s, i = _read_returned(p)
        except Exception as exc:                                # noqa: BLE001 - report and continue
            log.append({"File": p.name, "Sheet": "-", "Accepted": 0, "Rejected": 0, "Note": f"unreadable: {exc}"})
            continue
        s = s.copy()
        rate = pd.to_numeric(s["Apps_Per_FTE_Month"], errors="coerce")
        fte = pd.to_numeric(s["Current_FTE"], errors="coerce")
        avg = pd.to_numeric(s["Avg_Monthly_Last_12m"], errors="coerce")
        derived = rate.isna() & (fte > 0) & (avg > 0)
        rate = rate.where(~derived, avg / fte)
        ok = rate > 0
        s["Apps_Per_FTE_Month"] = rate.round(1)
        notes = s["Notes"].fillna("").astype(str)
        s["Notes"] = np.where(derived, "derived: last-12-month average / Current_FTE " + fte.astype(str) + ". " + notes,
                              notes)
        staff_rows.append(s[ok][SETTINGS_STAFF])
        log.append({"File": p.name, "Sheet": "Staffing", "Accepted": int(ok.sum()),
                    "Rejected": int(((fte.notna() | s["Apps_Per_FTE_Month"].notna()) & ~ok).sum()),
                    "Note": f"{int(derived.sum())} rate(s) derived from headcount"})
        i = i.astype({"Submitted_On": object, "Submitted_By": object, "Reason": object})
        ev = pd.to_numeric(i["Expected_Volume"], errors="coerce")
        filled = ev.notna()
        good = filled & (ev >= 0) & i["Submitted_By"].notna()
        i.loc[good & i["Submitted_On"].isna(), "Submitted_On"] = date.today().isoformat()
        input_rows.append(i[good][SETTINGS_HUMAN])
        log.append({"File": p.name, "Sheet": "Market_Inputs", "Accepted": int(good.sum()),
                    "Rejected": int((filled & ~good).sum()), "Note": "rejected rows lack a name or have a negative volume"})
    report = pd.DataFrame(log, columns=["File", "Sheet", "Accepted", "Rejected", "Note"])
    if not ret:
        return report
    settings_file = Path(settings_file)
    backup = settings_file.with_name(f"{settings_file.stem}_backup_{date.today():%Y%m%d}{settings_file.suffix}")
    shutil.copy2(settings_file, backup)
    new_staff = pd.concat(staff_rows, ignore_index=True) if staff_rows else pd.DataFrame(columns=SETTINGS_STAFF)
    new_human = pd.concat(input_rows, ignore_index=True) if input_rows else pd.DataFrame(columns=SETTINGS_HUMAN)
    with pd.ExcelFile(settings_file) as xl:
        existing = {s: pd.read_excel(xl, s) for s in ("Staffing_Productivity", "Human_Inputs") if s in xl.sheet_names}
    sheets = {}
    for sheet, new, cols in (("Staffing_Productivity", new_staff, SETTINGS_STAFF), ("Human_Inputs", new_human, SETTINGS_HUMAN)):
        old = existing.get(sheet, pd.DataFrame(columns=cols))
        for c in cols:
            if c not in old.columns:
                old[c] = np.nan
        if sheet == "Staffing_Productivity":       # a returned country replaces its own rows
            old = old[~old["Country"].astype(str).isin(set(new["Country"].astype(str)))]
        sheets[sheet] = pd.concat([old[cols], new[cols]], ignore_index=True)
    with pd.ExcelWriter(settings_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        for sheet, df in sheets.items():
            df.to_excel(w, sheet_name=sheet, index=False)
    report.attrs["backup"] = str(backup)
    return report
