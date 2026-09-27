import numpy as np
import pandas as pd

from src.forecasting import intervals as iv

LEVELS = (80, 95)


def residual_frame(n_nodes=3, n_orig=20, h=12, seed=0):
    rng = np.random.default_rng(seed)
    rows, nodes = [], []
    for i in range(n_nodes):
        nid = f"Received|C{i}|R"
        nodes.append({"node_id": nid, "Metric": "Received", "Level": "Region"})
        for o in range(n_orig):
            for k in range(1, h + 1):
                pred = 100.0
                actual = pred * np.exp(rng.normal(0, 0.05 * np.sqrt(k)))
                rows.append({"node_id": nid, "o": o, "h": k, "Pred": pred, "Actual": actual})
    return pd.DataFrame(rows), pd.DataFrame(nodes)


def test_quantiles_contain_point_widen_and_nest():
    comb, nodes = residual_frame()
    q, _ = iv.residual_quantiles(comb, nodes, LEVELS, min_obs=8, horizon=12)
    assert (q["q_lo80"] <= 0).all() and (q["q_hi80"] >= 0).all()
    assert (q["q_lo95"] <= q["q_lo80"]).all() and (q["q_hi95"] >= q["q_hi80"]).all()
    for _, g in q.groupby("node_id"):
        g = g.sort_values("h")
        assert (np.diff(g["q_hi80"]) >= -1e-12).all() and (np.diff(g["q_lo80"]) <= 1e-12).all()


def test_calibration_widens_to_nominal_coverage():
    comb, nodes = residual_frame()
    q, _ = iv.residual_quantiles(comb, nodes, LEVELS, min_obs=8, horizon=12)
    # honest errors twice as large as the in-sample ones
    rng = np.random.default_rng(1)
    hz = comb.rename(columns={"Pred": "Policy"}).copy()
    hz["Actual"] = hz["Policy"] * np.exp(rng.normal(0, 0.10 * np.sqrt(hz["h"])))
    q2, cal = iv.calibrate(q, hz, nodes, LEVELS, max_factor=3.0)
    row = cal[cal["Interval"] == "80%"].iloc[0]
    assert row["Factor"] > 1.0 and row["Coverage_After_Pct"] >= 80 > row["Coverage_Before_Pct"]


def test_apply_interval_orders_bounds():
    lo, hi = iv.apply_interval(np.array([100.0, 0.0]), np.array([-0.2, -0.1]), np.array([0.3, 0.1]))
    assert lo[0] < 100 < hi[0] and lo[1] <= 0 <= hi[1] + 1e-12


def test_block_quantiles_are_pooled_and_signed():
    comb, nodes = residual_frame()
    bq = iv.block_quantiles(comb, nodes, LEVELS, min_obs=8, calib=pd.DataFrame())
    assert set(bq["k"]) == {3, 6, 12}
    assert (bq["q_lo80"] <= 0).all() and (bq["q_hi80"] >= 0).all()
    assert bq["Source"].str.contains("scale").all()
