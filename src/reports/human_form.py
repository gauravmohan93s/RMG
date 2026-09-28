# -*- coding: utf-8 -*-
"""
Monthly market-input form: one row per destination x month for the next quarter, pre-filled with the
model forecast and planning range. Market heads fill Expected_Volume, Submitted_By and Reason (yellow),
then the rows are pasted into Settings.xlsx, sheet Human_Inputs. The pipeline records every input,
scores it against the actuals on identical months, and blends it in only for people who have beaten
the model for at least 6 scored months (analysis/signals.py).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.prepare import ORG, TOTAL

FORM_COLS = ["Country", "Region", "Metric", "Month", "Expected_Volume", "Submitted_By", "Reason", "Submitted_On"]


def form_rows(res, months: int = 3) -> pd.DataFrame:
    fc = res.forecast
    f = fc[fc["Level"].isin(["Organisation", "Country"]) & (fc["Region"] == TOTAL) & (fc["Horizon"] <= months)].copy()
    f["_o"] = (f["Level"] != "Organisation").astype(int)
    f = f.sort_values(["Metric", "_o", "Country", "Date"])
    return pd.DataFrame({
        "Country": ["Organisation" if c == ORG else c for c in f["Country"]],
        "Region": "",
        "Metric": f["Metric"].to_numpy(),
        "Month": [f"{d:%Y-%m}" for d in pd.to_datetime(f["Date"])],
        "Expected_Volume": None, "Submitted_By": None, "Reason": None, "Submitted_On": None,
        "Model_Forecast": f["Forecast"].round(-1).to_numpy(),
        "Plan_Low": f["Lo80"].round(-1).to_numpy(), "Plan_High": f["Hi80"].round(-1).to_numpy(),
    })


def write_form(res, path: Path) -> Path:
    d = form_rows(res)
    with pd.ExcelWriter(path, engine="xlsxwriter") as w:
        d.to_excel(w, sheet_name="Human_Inputs", index=False, startrow=3)
        wb, ws = w.book, w.sheets["Human_Inputs"]
        title = wb.add_format({"bold": True, "font_size": 13})
        note = wb.add_format({"font_color": "#52514e", "text_wrap": True, "valign": "top"})
        fill = wb.add_format({"bg_color": "#fff3b0", "border": 1, "border_color": "#d0d7e2"})
        grey = wb.add_format({"font_color": "#52514e", "num_format": "#,##0"})
        ws.write(0, 0, f"Market input form — forecast run {res.run_date}", title)
        ws.merge_range(1, 0, 2, 10,
                       "Fill the yellow cells where you expect something different from the model (leave the rest "
                       "blank): your expected number of applications, your name and the reason (e.g. new partner, "
                       "visa change, campaign). Submitted_On = today's date. Paste the filled rows into Settings.xlsx, "
                       "sheet Human_Inputs (same columns). Every input is scored against what happens; after 6 scored "
                       "months, inputs from people who beat the model are blended into the forecast automatically.",
                       note)
        for r in range(len(d)):
            for c in (4, 5, 6, 7):
                ws.write_blank(4 + r, c, None, fill)
        ws.set_column(0, 0, 26)
        ws.set_column(1, 3, 11)
        ws.set_column(4, 7, 18)
        ws.set_column(8, 10, 15, grey)
        ws.set_row(1, 30)
        ws.freeze_panes(4, 0)
    return path
