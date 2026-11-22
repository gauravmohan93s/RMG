from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.reports import insights as ins


def test_revisions_compare_only_months_both_forecasts_cover():
    d = pd.date_range("2026-11-01", periods=3, freq="MS")
    fc = pd.DataFrame([{"Level": lvl, "Country": c, "Region": "_TOTAL", "Metric": "Received", "Date": x,
                        "Forecast": f, "Previous_Forecast": (p if i < 2 else np.nan)}
                       for lvl, c, f, p in (("Organisation", "_ORG", 110, 100), ("Country", "UK", 70, 60),
                                            ("Country", "Asia", 40, 40))
                       for i, x in enumerate(d)])
    r = ins.revisions(SimpleNamespace(forecast=fc))
    assert r["months"] == 2 and np.isclose(r["change"], 20) and np.isclose(r["pct"], 10)
    by = r["by_destination"].set_index("Country")["Change"]
    assert by["UK"] == 20 and by["Asia"] == 0


def test_track_marks_each_month_right_close_or_missed():
    months = pd.date_range("2026-05-01", periods=6, freq="MS")
    actual = [100, 100, 100, 100, 100, 100]
    fcst = [100, 95, 88, 85, 70, 130]                   # actual vs forecast: 0%, +5%, +14%, +18%, +43%, -23%
    fva = pd.DataFrame([{"Level": "Organisation", "Country": "_ORG", "Region": "_TOTAL", "Metric": "Received",
                         "Months_Ahead": 1, "Target_Month": m, "Forecast": f, "Actual": a,
                         "Source": "Live forecast (run x)"} for m, f, a in zip(months, fcst, actual)])
    t = ins._track(SimpleNamespace(forecast_vs_actual=fva), "Organisation", "_ORG")
    assert [x["verdict"] for x in t] == ["right", "right", "close", "close", "missed", "missed"]
