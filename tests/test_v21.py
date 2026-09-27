"""v2.1 components: seasonality/staffing, statistical overlay, human inputs, event impacts, vintages."""
import numpy as np
import pandas as pd

from src.analysis import seasonality as seas
from src.analysis import signals as sig
from src.forecasting import bias_correction as bc

PATTERN = np.array([1.0, 1.4, 1.0, 1.05, 0.95, 1.0, 0.85, 1.0, 1.3, 0.95, 0.9, 0.6])


def org_history(n=48, shock_at=None, shock=1.0, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="MS")
    v = 1000 * PATTERN[dates.month - 1] * np.exp(rng.normal(0, 0.03, n))
    if shock_at is not None:
        v[shock_at:] *= shock
    return pd.DataFrame({"Level": "Organisation", "Country": "_ORG", "Region": "_TOTAL", "Metric": "Received",
                         "Date": dates, "Actual": v})


def test_seasonal_index_recovers_known_pattern():
    ix, yr = seas.seasonal_indices(org_history())
    ix = ix.set_index("Month")
    assert ix.loc[2, "Class"] == "High" and ix.loc[9, "Class"] == "High"
    assert ix.loc[12, "Class"] == "Low" and ix.loc[1, "Class"] == "Medium"
    assert abs(ix["Seasonal_Index"].mean() - 1) < 1e-9
    assert ix.loc[2, "Consistency_Pct"] == 100 and len(yr) > 0


def test_staffing_calendar_load_index_and_region_specific_fte():
    dates = pd.date_range("2027-01-01", periods=12, freq="MS")
    f = 1000 * PATTERN
    base = dict(Metric="Received", Date=dates, Lo80=f * 0.8)
    fc = pd.concat([
        pd.DataFrame({"Level": "Organisation", "Country": "_ORG", "Region": "_TOTAL", "Label": "Organisation",
                      "Forecast": 2 * f, "Hi80": 2 * f * 1.2, **base}),
        pd.DataFrame({"Level": "Country", "Country": "UK", "Region": "_TOTAL", "Label": "UK",
                      "Forecast": 2 * f, "Hi80": 2 * f * 1.2, **base}),
        pd.DataFrame({"Level": "Region", "Country": "UK", "Region": "North", "Label": "UK – North",
                      "Forecast": f, "Hi80": f * 1.2, **base}),
        pd.DataFrame({"Level": "Region", "Country": "UK", "Region": "South", "Label": "UK – South",
                      "Forecast": f, "Hi80": f * 1.2, **base}),
    ])
    ix, _ = seas.seasonal_indices(org_history())
    prod = pd.DataFrame([{"Country": "UK", "Region": "North", "Metric": "*", "Apps_Per_FTE_Month": 100},
                         {"Country": "UK", "Region": "*", "Metric": "*", "Apps_Per_FTE_Month": 50}])
    st = seas.staffing_calendar(fc, ix, productivity=prod)
    feb = st[st["Month_Name"] == "Feb"].set_index(["Level", "Region"])
    assert feb.loc[("Organisation", "_TOTAL"), "Workload_Class"] == "High"
    n, s_ = feb.loc[("Region", "North"), "FTE_Base"], feb.loc[("Region", "South"), "FTE_Base"]
    assert np.isclose(n, 1400 / 100) and np.isclose(s_, 1400 / 50)          # region-specific rates
    assert np.isclose(feb.loc[("Country", "_TOTAL"), "FTE_Base"], n + s_)     # country = sum of regions
    assert np.isclose(feb.loc[("Organisation", "_TOTAL"), "FTE_Base"], n + s_)
    assert np.isclose(st[st["Level"] == "Organisation"]["Load_Index"].mean(), 1.0)
    no_rates = seas.staffing_calendar(fc, ix)
    assert "FTE_Base" not in no_rates.columns


def comb_frame(bias_node_ratio=1.2, n_nodes=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_nodes):
        for o in range(30):
            for h in (1, 2, 3):
                pred = 100.0
                ratio = bias_node_ratio if i == 0 else np.exp(rng.normal(0, 0.10))
                rows.append({"node_id": f"n{i}", "o": o, "h": h, "t": o + h, "Pred": pred, "Actual": pred * ratio})
    return pd.DataFrame(rows)


def test_bias_estimate_shrinks_noise_keeps_consistent_shift():
    est = bc.estimate(comb_frame(), at=32).set_index("node_id")
    assert est.loc["n0", "Correction_Log"] > 0.8 * np.log(1.2)
    assert est.drop("n0")["Correction_Log"].abs().max() < 0.06


def test_bias_mode_adopted_only_when_it_helps():
    comb = comb_frame(bias_node_ratio=1.25)
    honest = comb[comb["o"] >= 24].rename(columns={"Pred": "Policy"}).reset_index(drop=True)
    test, _ = bc.walk_forward_test(comb, honest)
    assert bc.decide(test) in ("level", "decay")
    unbiased = comb_frame(bias_node_ratio=1.0)
    h2 = unbiased[unbiased["o"] >= 24].rename(columns={"Pred": "Policy"}).reset_index(drop=True)
    t2, _ = bc.walk_forward_test(unbiased, h2)
    assert bc.decide(t2, min_gain_pct=1.0) == "none"


def test_human_input_blended_only_after_proving_itself():
    dates = pd.date_range("2026-01-01", periods=8, freq="MS")
    hist = pd.DataFrame({"Level": "Country", "Country": "Canada", "Region": "_TOTAL", "Metric": "Received",
                         "Date": dates, "Expected_Volume": 100.0, "Submitted_By": "Head Canada",
                         "Run_Date": dates - pd.DateOffset(months=1)})
    actuals = hist[["Level", "Country", "Region", "Metric", "Date"]].assign(Actual=102.0)
    model = hist[["Level", "Country", "Region", "Metric", "Date", "Run_Date"]].assign(Forecast=130.0)
    sc = sig.score_humans(hist, actuals, model).set_index("Submitted_By")
    assert sc.loc["Head Canada", "Eligible_For_Blend"] and sc.loc["Head Canada", "Blend_Weight"] > 0.5
    few = sig.score_humans(hist.head(3), actuals, model).set_index("Submitted_By")
    assert not few.loc["Head Canada", "Eligible_For_Blend"]
    fc = pd.DataFrame({"Level": ["Country"], "Country": ["Canada"], "Region": ["_TOTAL"], "Metric": ["Received"],
                       "Date": [pd.Timestamp("2026-10-01")], "Forecast": [130.0]})
    hi = sig.clean_human(pd.DataFrame([{"Country": "Canada", "Region": "", "Metric": "Received", "Month": "2026-10",
                                        "Expected_Volume": 100, "Submitted_By": "Head Canada", "Reason": "cap",
                                        "Submitted_On": "2026-09-20"}]))
    out = sig.attach_human(fc, hi, sc.reset_index())
    assert out["Human_Expected"].iloc[0] == 100 and 100 < out["Forecast_Human_Blend"].iloc[0] < 130


def test_event_impact_detects_real_shock_not_placebo():
    ev = pd.DataFrame([{"Country": "Organisation", "Metric": "Received", "Event": "cap",
                        "Start_Month": "2029-01", "Notes": ""}])          # index 84 of a 96-month history
    shocked = sig.measure_events(ev, org_history(n=96, shock_at=84, shock=0.6)).iloc[0]
    assert shocked["Measured_Impact_Pct"] < -30 and shocked["Significant"] == "Yes"
    calm = sig.measure_events(ev, org_history(n=96)).iloc[0]
    assert abs(calm["Measured_Impact_Pct"]) < 10 and calm["Significant"] == "No"
    short = pd.DataFrame([{"Country": "Organisation", "Metric": "Received", "Event": "cap",
                           "Start_Month": "2025-05", "Notes": ""}])       # only 40 months before
    assert sig.measure_events(short, org_history(n=48)).iloc[0]["Significant"].startswith("Not judged")


def test_forecast_vs_actual_summary():
    months = pd.date_range("2024-01-01", periods=40, freq="MS")
    honest = pd.DataFrame({"node_id": "Received|_ORG|_TOTAL", "o": [20, 20, 21], "h": [1, 2, 1], "t": [21, 22, 22],
                           "Policy": [90.0, 95.0, 99.0], "Actual": [100.0, 100.0, 100.0]})
    nodes = pd.DataFrame([{"node_id": "Received|_ORG|_TOTAL", "Level": "Organisation", "Country": "_ORG",
                           "Region": "_TOTAL", "Metric": "Received"}])
    fva = sig.forecast_vs_actual(honest, nodes, months)
    s = sig.fva_summary(fva, horizons=(1, 2)).set_index("Target_Month")
    assert np.isclose(s.loc[months[22], "Forecast_1m_ahead"], 99.0) and np.isclose(s.loc[months[22], "Error_2m_Pct"], -5.0)
