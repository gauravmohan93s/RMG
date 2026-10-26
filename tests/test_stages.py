import numpy as np
import pandas as pd

from src.forecasting import stages as sg


def _setup(T=60, H=12):
    t = np.arange(T + H)
    sub = 1000 * (1.02 ** t) * (1 + 0.2 * np.sin(2 * np.pi * t / 12))          # growing, seasonal submissions
    offer = np.r_[0, 0.6 * sub[:-1]]                                         # 60% one month later
    deposit = np.r_[0, 0, 0.5 * offer[:-2]]                                  # 50% two months later
    nodes = pd.DataFrame([{"node_id": "Submitted|_ORG|_TOTAL", "Level": "Organisation", "Country": "_ORG",
                           "Region": "_TOTAL", "Metric": "Submitted"}])
    Y = {"Submitted|_ORG|_TOTAL": sub[:T]}
    stage_actual = {("_ORG", "Offer"): offer[:T], ("_ORG", "Deposit"): deposit[:T]}
    origins = range(T - 25, T - 1)
    honest = pd.DataFrame([{"node_id": "Submitted|_ORG|_TOTAL", "o": o, "h": h, "Policy": sub[o + h]}
                           for o in origins for h in range(1, H + 1)])          # perfect submission forecasts
    fc = pd.DataFrame([{"Level": "Organisation", "Country": "_ORG", "Region": "_TOTAL", "Metric": "Submitted",
                        "Horizon": h, "Forecast": sub[T - 1 + h]} for h in range(1, H + 1)])
    months = pd.date_range("2021-01-01", periods=T + H, freq="MS")
    return stage_actual, Y, nodes, honest, fc, months, T, offer, deposit


def test_chain_recovers_conversion_and_beats_last_year():
    stage_actual, Y, nodes, honest, fc, months, T, offer, deposit = _setup()
    sf, test, conv = sg.run(stage_actual, Y, nodes, honest, fc, months, T)
    chosen = test[test["Chosen"]].drop_duplicates("StageGroup").set_index("StageGroup")["Variant"]
    assert chosen["Offer"] in ("chain", "blend") and chosen["Deposit"] in ("chain", "blend")
    c = conv.set_index("StageGroup")
    assert c.loc["Offer", "From"] == "Submitted" and abs(c.loc["Offer", "Conversion_Pct"] - 60) < 1
    # deposits are exactly 50% of offers two months later = 30% of submissions three months later
    expected = 50 if c.loc["Deposit", "From"] == "Offer" else 30
    assert abs(c.loc["Deposit", "Conversion_Pct"] - expected) < 1
    o = sf[sf["StageGroup"] == "Offer"].set_index("Horizon")["Forecast"]
    assert np.allclose(o.to_numpy(), offer[T:T + 12], rtol=0.02)            # downstream of perfect submissions
    d = sf[sf["StageGroup"] == "Deposit"].set_index("Horizon")["Forecast"]
    assert np.allclose(d.to_numpy(), deposit[T:T + 12], rtol=0.03)
    assert (sf["Lo80"] <= sf["Forecast"] + 1e-9).all() and (sf["Hi80"] >= sf["Forecast"] - 1e-9).all()
    tot = sg.stage_totals(sf)
    assert set(tot["Window"]) == {"Next 3 months", "Next 12 months"}


def test_chain_stops_cleanly_without_stage_history():
    stage_actual, Y, nodes, honest, fc, months, T, *_ = _setup()
    sf, test, conv = sg.run({}, Y, nodes, honest, fc, months, T)
    assert sf.empty and test.empty and conv.empty
