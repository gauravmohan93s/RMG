import numpy as np
import pandas as pd

from src.analysis import intake_pace as ip

# every Sep intake fills the same way: 10% of its final count arrives in each of the 10 months before the
# start (leads 10..1); the size grows 20% a year, so the right answer is always "+20%, final = 1.2 x last year"
CURVE = {lead: 0.10 for lead in range(1, 11)}


def _cleaned(last="2026-10-01", years=range(2020, 2028)):
    rows = []
    for country in ("UK", "Canada", "Australia"):        # enough completed intakes for the evidence gate
        for y in years:
            start = pd.Timestamp(f"{y}-09-01")
            size = 1000 * 1.2 ** (y - 2020)
            for lead, share in CURVE.items():
                d = start - pd.DateOffset(months=lead)
                if d <= pd.Timestamp(last):
                    rows.append({"Country": country, "Region": "North", "Date": d, "Intake_Start": start,
                                 "Intake_Known": True, "Received": size * share, "Submitted": 0.6 * size * share})
    return pd.DataFrame(rows)


def test_pace_and_projection_recover_the_truth():
    pace, curves, test = ip.intake_pace(ip.intake_long(_cleaned()), pd.Timestamp("2026-10-01"))
    sep27 = pace[(pace["Level"] == "Country") & (pace["Metric"] == "Received") & (pace["Intake"] == "Sep 2027")]
    assert sep27["Months_To_Start"].iloc[0] == 11 and sep27["To_Date"].iloc[0] == 0   # nothing in yet

    # seen from April 2027 (5 months before the start): leads 10..5 are in (60%), pace +20%, final = 1.2 x last year
    p, c, t = ip.intake_pace(ip.intake_long(_cleaned(last="2027-04-01")), pd.Timestamp("2027-04-01"))
    r = p[(p["Country"] == "UK") & (p["Level"] == "Country") & (p["Metric"] == "Received") & (p["Intake"] == "Sep 2027")].iloc[0]
    assert np.isclose(r["Pace_Pct"], 20.0) and r["Status"] == "ahead"
    assert np.isclose(r["Typical_Share_In_Pct"], 60.0)
    assert np.isclose(r["Projected_Final"], 1000 * 1.2 ** 7, rtol=1e-6)   # tested: exact shares beat "last year"
    assert r["Lo80"] <= r["Projected_Final"] <= r["Hi80"]
    assert t[(t["Level"] == "Country") & (t["Lead"] == 5) & (t["Metric"] == "Received")]["Used"].iloc[0]
    assert set(c["Series"]) == {"This intake", "Last year", "Two years ago"}
    assert set(p["Level"]) == {"Organisation", "Country", "Region"}


def test_stage_counts_by_intake_join_the_same_view(tmp_path):
    f = tmp_path / "stage_intake_summary.xlsx"
    pd.DataFrame({"Month": ["2026-09", "2026-10"], "Country Grouped": ["UK", "UK"], "Name of Region": ["North"] * 2,
                  "StageGroup": ["Offer", "Offer"], "Intake": ["Jan", "Spring"], "InYear": ["2027", "2027"],
                  "Apps": [5, 7]}).to_excel(f, index=False)
    s = ip.load_stage_intake(f)
    assert (s["Intake_Start"] == pd.Timestamp("2027-01-01")).all() and s["Apps"].sum() == 12   # 'Spring' = Jan
    long = ip.intake_long(_cleaned(), s)
    assert {"Received", "Submitted", "Offer"} <= set(long["Metric"])
