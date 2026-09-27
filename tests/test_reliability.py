"""Reliability on quarter totals of published forecasts; published (bottom-up) backtest."""
import numpy as np
import pandas as pd

from src.forecasting import reliability as rl


def honest_frame(errors):
    """errors: node -> multiplicative monthly error (forecast = actual * err)."""
    rows = []
    for nid, err in errors.items():
        for o in range(10):
            for h in range(1, 4):
                rows.append({"node_id": nid, "o": o, "h": h, "Policy": 100 * err, "Actual": 100.0})
    return pd.DataFrame(rows)


def test_reliability_bands_from_quarter_totals():
    nodes = pd.DataFrame({"node_id": ["a", "b", "c", "d"]})
    hz = honest_frame({"a": 1.05, "b": 0.85, "c": 1.4})
    r = rl.reliability(hz, nodes).set_index("node_id")
    assert r.loc["a", "Reliability"] == "High" and np.isclose(r.loc["a", "Typical_Quarter_Error_Pct"], 5)
    assert r.loc["b", "Reliability"] == "Medium" and r.loc["c", "Reliability"] == "Low"
    assert r.loc["d", "Reliability"] == "New"
    assert rl.label_text("High", 5.0).startswith("High") and "±5%" in rl.label_text("High", 5.0)


def test_monthly_noise_cancels_in_quarter_reliability():
    rows = []
    for o in range(10):
        for h, e in zip((1, 2, 3), (1.25, 0.75, 1.0)):          # ±25% monthly, 0% on the quarter
            rows.append({"node_id": "x", "o": o, "h": h, "Policy": 100 * e, "Actual": 100.0})
    r = rl.reliability(pd.DataFrame(rows), pd.DataFrame({"node_id": ["x"]})).iloc[0]
    assert r["Reliability"] == "High" and r["Typical_Quarter_Error_Pct"] < 1e-9


def test_published_backtest_sums_regions_for_aggregates():
    nodes = pd.DataFrame([
        {"node_id": "R|org", "Level": "Organisation", "Country": "_ORG", "Region": "_TOTAL", "Metric": "R"},
        {"node_id": "R|A", "Level": "Country", "Country": "A", "Region": "_TOTAL", "Metric": "R"},
        {"node_id": "R|A1", "Level": "Region", "Country": "A", "Region": "1", "Metric": "R"},
        {"node_id": "R|A2", "Level": "Region", "Country": "A", "Region": "2", "Metric": "R"},
        {"node_id": "R|B1", "Level": "Region", "Country": "B", "Region": "1", "Metric": "R"},
    ])
    preds = {"R|org": 999.0, "R|A": 555.0, "R|A1": 100.0, "R|A2": 90.0, "R|B1": 110.0}
    comb = pd.DataFrame([{"node_id": n, "o": 0, "h": 1, "t": 1, "Pred": p, "Actual": 1.0} for n, p in preds.items()])
    pub = rl.published_backtest(comb, nodes, "bottom_up").set_index("node_id")["Pred"]
    assert pub["R|A"] == 190.0 and pub["R|org"] == 300.0 and pub["R|A1"] == 100.0
    assert rl.published_backtest(comb, nodes, "none").equals(comb)


def test_published_backtest_reconciles_for_other_methods():
    """With wls_var the aggregates' backtest errors must be those of the RECONCILED (published) numbers:
    coherent (aggregate = sum of its regions), not the aggregate's own unreconciled model."""
    ids = [("Received|_ORG|_TOTAL", "Organisation", "_ORG", "_TOTAL"), ("Received|A|_TOTAL", "Country", "A", "_TOTAL"),
           ("Received|A|1", "Region", "A", "1"), ("Received|A|2", "Region", "A", "2"), ("Received|B|1", "Region", "B", "1")]
    nodes = pd.DataFrame([{"node_id": n, "Level": l, "Country": c, "Region": r, "Metric": "Received", "Status": "Mature"}
                          for n, l, c, r in ids])
    base = {"Received|_ORG|_TOTAL": 330.0, "Received|A|_TOTAL": 200.0, "Received|A|1": 100.0, "Received|A|2": 90.0,
            "Received|B|1": 110.0}
    rows = []
    for o in range(6):
        for h in (1, 2):
            for n, p in base.items():
                rows.append({"node_id": n, "o": o, "h": h, "t": o + h, "Pred": p * (1 + 0.01 * o), "Actual": p})
    comb = pd.DataFrame(rows)
    pub = rl.published_backtest(comb, nodes, "wls_var")
    g = pub[(pub["o"] == 5) & (pub["h"] == 1)].set_index("node_id")["Pred"]
    assert np.isclose(g["Received|A|_TOTAL"], g["Received|A|1"] + g["Received|A|2"])
    assert np.isclose(g["Received|_ORG|_TOTAL"], g["Received|A|1"] + g["Received|A|2"] + g["Received|B|1"])
