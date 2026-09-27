import numpy as np
import pandas as pd

from src.forecasting import ranges as rg

LEVELS = [50, 80, 95]


def _nodes(n_reg=6):
    rows = [{"node_id": f"Received|C|R{i}", "Level": "Region", "Country": "C", "Region": f"R{i}",
             "Metric": "Received", "Status": "Mature"} for i in range(n_reg)]
    rows.append({"node_id": "Received|C|_TOTAL", "Level": "Country", "Country": "C", "Region": "_TOTAL",
                 "Metric": "Received", "Status": "Mature"})
    rows.append({"node_id": "Received|C|NEW", "Level": "Region", "Country": "C", "Region": "NEW",
                 "Metric": "Received", "Status": "Emerging"})
    return pd.DataFrame(rows)


def _backtest(nodes, bias=0.0, sd=0.15, origins=range(56, 80), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for nid in nodes.loc[nodes["Status"] == "Mature", "node_id"]:
        for o in origins:
            for h in range(1, 13):
                pred = 1000.0
                rows.append({"node_id": nid, "o": o, "h": h, "t": o + h, "Pred": pred,
                             "Actual": pred * np.exp(bias + rng.normal(0, sd))})
    return pd.DataFrame(rows)


def test_centred_range_is_symmetric_in_log_space():
    q = rg._fit("centred", np.random.default_rng(1).normal(0.3, 0.1, 40), np.arange(40), np.array([]), 40, LEVELS, 8)
    for L in LEVELS:
        assert abs(q[f"q_lo{L}"] + q[f"q_hi{L}"]) < 1e-12


def test_build_gives_every_node_nested_ranges_and_evidence():
    nodes = _nodes()
    q, bq, test, cal_m, cal_t = rg.build(_backtest(nodes), nodes, LEVELS, 8, 12)
    assert set(q["node_id"]) == set(nodes["node_id"])            # emerging node gets pooled ranges
    assert set(bq["k"]) == {3, 6, 12}
    for L_n, L_w in ((50, 80), (80, 95)):
        assert (q[f"q_lo{L_w}"] <= q[f"q_lo{L_n}"] + 1e-12).all() and (q[f"q_hi{L_w}"] >= q[f"q_hi{L_n}"] - 1e-12).all()
    assert test["Chosen"].any() and {"Interval_Score", "Coverage_Pct"} <= set(test.columns)
    # well-behaved symmetric errors: 80% ranges cover roughly 80% of test errors
    chosen = test[test["Chosen"] & (test["Interval"] == "80%") & (test["Group"] == "Region") &
                  test["Window"].isin(["Months 1-3 ahead", "Next 3 months"])]
    assert chosen["Coverage_Pct"].between(65, 95).all()
    assert (cal_m["Width_Factor"].between(0.8, 1.5)).all()


def test_persistent_bias_is_not_hidden():
    """If actuals are always above forecasts, the published 80% range must still cover them ~80% of the time."""
    nodes = _nodes()
    _, _, _, cal_m, cal_t = rg.build(_backtest(nodes, bias=0.25, sd=0.08), nodes, LEVELS, 8, 12)
    reg = cal_m[(cal_m["Group"] == "Region") & (cal_m["Interval"] == "80%")]
    assert (reg["Coverage_After_Pct"] >= 70).all()
