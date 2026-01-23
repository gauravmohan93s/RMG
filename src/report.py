"""
Stakeholder report generator
────────────────────────────
Creates
    •  organisation_report.xlsx           (Country‑level)
    •  reports/<Country>.xlsx (one per)   (Region‑level)
Run automatically by run_pipeline.report()
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from textwrap import dedent

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .config import OUT_DIR, HORIZON
from .loader import load_data                     # clean history

REPORT_DIR = OUT_DIR / "Reports"
REPORT_DIR.mkdir(exist_ok=True)

# the three artefacts produced by your modelling stage ↓
FINAL_FC_PATH   = OUT_DIR / "Final_Forecast" / "final_forecast.xlsx"
RANKING_PATH    = OUT_DIR / "Evaluation" / "series_model_ranking.csv"
MODEL_LOG_PATH  = OUT_DIR / "Evaluation" /"country_best_models.csv"   # ensure you write this




# ────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────
def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """history, ranking, forecast"""
    hist, _ = load_data()
    ranking = pd.read_csv(RANKING_PATH)
    fc      = pd.read_excel(FINAL_FC_PATH, parse_dates=["Date"])
    return hist, ranking, fc


def _tl_dr_narrative(fc: pd.DataFrame, scope: str) -> str:
    """Return a stakeholder‑friendly paragraph."""
    # crude examples – customise the language as you wish
    tot_recv  = fc.groupby("Date")["Received"].sum()
    yoy       = tot_recv.iloc[-1] / tot_recv.iloc[0] - 1
    direction = "increase" if yoy > 0 else "decrease"
    pct       = f"{abs(yoy)*100:,.1f}%"
    return dedent(
        f"""
        • Over the coming {HORIZON} months we expect a **{direction} of {pct}**
          in aggregated Received applications.

        • Seasonality and recent momentum are the dominant drivers; external
          regressors contribute ~{fc.filter(like='x_').shape[1]} features.

        • Forecast risk is skewed towards {{'upside' if yoy>0 else 'downside'}}:
          80 % prediction‑interval width averages
          {fc['Received_lo80'].sub(fc['Received_hi80']).abs().mean():,.0f} apps.

        • Next step: focus on the five fastest‑growing countries
          to validate regional assumptions.
        """
    ).strip()


def _add_chart(ws, g: pd.DataFrame, start_row: int) -> None:
    """
    Write a tiny area chart (history + forecast) into the worksheet at row.
    Expects columns: Date, Value_hist, Value_fc
    """
    # write data
    for col_i, col in enumerate(g.columns):
        ws.write_column(start_row, col_i, g[col])
    # build chart
    chart = ws.book.add_chart({"type": "line"})
    chart.add_series(
        {"categories": [ws.name, start_row, 0, start_row + len(g) - 1, 0],
         "values":     [ws.name, start_row, 1, start_row + len(g) - 1, 1],
         "name": "Hist"})
    chart.add_series(
        {"categories": [ws.name, start_row, 0, start_row + len(g) - 1, 0],
         "values":     [ws.name, start_row, 2, start_row + len(g) - 1, 2],
         "name": "Fc"})
    chart.set_legend({"none": True})
    ws.insert_chart(start_row, 4, chart, {"x_scale": .6, "y_scale": .6})


# ────────────────────────────────────────────────────────────────────
# core builder
# ────────────────────────────────────────────────────────────────────
def _build_book(scope_df: pd.DataFrame,
                ranking:   pd.DataFrame,
                fc:        pd.DataFrame,
                book_path: Path,
                scope_name: str) -> None:
    """
    scope_df : history slice (either whole org or one country)
    ranking  : model ranking (filtered beforehand)
    fc       : forecast slice             "

    Writes an Excel book to book_path.
    """
    with pd.ExcelWriter(book_path, engine="xlsxwriter") as w:
        wb = w.book

        # TLDR sheet --------------------------------------------------
        tldr_txt = _tl_dr_narrative(fc, scope_name)
        pd.DataFrame({"TL;DR": [tldr_txt]}).to_excel(w, "TLDR", index=False)
        ws_tldr = w.sheets["TLDR"]
        ws_tldr.set_column(0, 0, 120)

        # Model log ---------------------------------------------------
        ranking.to_excel(w, "Model_log", index=False)

        # Forecast table ---------------------------------------------
        fc.to_excel(w, "Forecast", index=False)

        # Raw output (history + fc) ----------------------------------
        scope_df.to_excel(w, "History", index=False)

        # Charts sheet -----------------------------------------------
        ws_ch = wb.add_worksheet("Charts")
        w.sheets["Charts"] = ws_ch
        ws_ch.write_row(0, 0, ["Date", "Hist", "Fc"])
        row_ptr = 1
        for key, g_hist in scope_df.groupby(["Country", "Region"]):
            g_fc = fc[(fc["Country"], fc["Region"]) == key] if "Region" in fc else fc
            g = pd.DataFrame({
                "Date": g_hist["Date"],
                "Hist": g_hist["Received"],
                "Fc":   pd.concat([g_hist[["Date", "Received"]],
                                   g_fc[["Date", "Received_fc"]]], ignore_index=True
                                  )["Received_fc"].reindex_like(g_hist)
            })
            _add_chart(ws_ch, g, row_ptr)
            row_ptr += len(g) + 3


def make_reports() -> None:
    """Public entry‑point used by run_pipeline.report()."""
    hist, ranking, fc_all = _load_inputs()

    # 1) organisation‑wide
    _build_book(
        hist, ranking, fc_all,
        REPORT_DIR / "organisation_report.xlsx",
        "Organisation level"
    )

    # 2) each country
    for country, g_hist in hist.groupby("Country", sort=False):
        g_fc     = fc_all[fc_all["Country"] == country]
        g_rank   = ranking[ranking["Country"] == country]
        _build_book(
            g_hist, g_rank, g_fc,
            REPORT_DIR / f"{country}.xlsx",
            country
        )

    print(f"📒  Reports written to {REPORT_DIR}")


# if run standalone ----------------------------------------------------
if __name__ == "__main__":
    make_reports()
