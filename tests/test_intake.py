"""Descriptive intake view on synthetic data."""
import numpy as np
import pandas as pd

from src.analysis import intake as itk

# Fall (September) intakes receive 10/20/30/40% of their applications 4/3/2/1 months before start
PROFILE = {4: 0.1, 3: 0.2, 2: 0.3, 1: 0.4}


def synthetic(years=(2023, 2024, 2025), n=1000):
    rows = []
    for y in years:
        for lead, share in PROFILE.items():
            month = pd.Timestamp(y, 9, 1) - pd.DateOffset(months=lead)
            rows.append({"Month": month.strftime("%Y-%m"), "Country Grouped": "UK", "Name of Region": "North",
                         "Intake": "September", "InYear": str(y), "ReceivedApps": n * share, "SubmittedApps": 0})
    rows.append({"Month": "2025-05", "Country Grouped": "UK", "Name of Region": "North", "Intake": "",
                 "InYear": "", "ReceivedApps": 50, "SubmittedApps": 0})           # unknown intake
    return pd.DataFrame(rows)


def test_parse_intake_month_variants():
    assert itk.parse_intake_month("September") == 9 and itk.parse_intake_month("sep") == 9
    assert itk.parse_intake_month("Fall") == 9 and itk.parse_intake_month("1") == 1
    assert np.isnan(itk.parse_intake_month("")) and np.isnan(itk.parse_intake_month("unknown"))


def test_arrivals_lead_profile_and_mix():
    d = itk.clean_intake(synthetic())
    arr, lead, mix, known = itk.intake_views(d, last_actual=pd.Timestamp("2026-06-01"), years=3)
    assert 0.98 < known < 0.99                                   # 3000 of 3050 have a usable intake
    org = arr[(arr["Level"] == "Organisation") & (arr["Intake_Label"] == "Sep 2025")].sort_values("Date")
    assert np.allclose(org["Share_of_Intake_Pct"], [10, 20, 30, 40])
    assert np.isclose(org["Cumulative_Pct"].iloc[-1], 100)
    assert list(org["Months_Before_Start"]) == [4, 3, 2, 1]
    curve = lead[(lead["Level"] == "Organisation") & (lead["Season"] == "Fall")].set_index("Months_Before_Start")
    assert np.isclose(curve.loc[3, "Avg_Cumulative_Pct"], 30) and np.isclose(curve.loc[1, "Avg_Cumulative_Pct"], 100)
    m = mix[mix["Level"] == "Organisation"]
    assert np.allclose(m.groupby("Application_Month")["Share_Pct"].sum(), 100)
