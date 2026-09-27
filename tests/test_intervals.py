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


def _band_rows(cal, L=80):
    return cal[cal["Interval"] == f"{L}%"]


def test_conformal_calibration_reaches_nominal_coverage():
    comb, nodes = residual_frame()
    q, _ = iv.residual_quantiles(comb, nodes, LEVELS, min_obs=8, horizon=12)
    rng = np.random.default_rng(1)                         # walk-forward errors twice as large
    hz = comb.rename(columns={"Pred": "Policy"}).copy()
    hz["Actual"] = hz["Policy"] * np.exp(rng.normal(0, 0.10 * np.sqrt(hz["h"])))
    _, cal = iv.calibrate(q, hz, nodes, LEVELS, max_offset=1.5)
    rows = _band_rows(cal)
    assert (rows["Coverage_After_Pct"] >= 80).all() and (rows["Coverage_Before_Pct"] < 80).all()
    assert (rows["Offset_Hi"] > 0).all() and (rows["Offset_Lo"] >= 0).all()


def test_conformal_widens_a_collapsed_tail_after_a_bias_flip():
    """Upper quantile at 0 (history over-forecast), then actuals run 20% above: hi must widen."""
    comb, nodes = residual_frame()
    q, _ = iv.residual_quantiles(comb, nodes, LEVELS, min_obs=8, horizon=12)
    q[["q_hi80", "q_hi95"]] = 0.0
    hz = comb.rename(columns={"Pred": "Policy"}).copy()
    hz["Actual"] = hz["Policy"] * 1.2
    q2, cal = iv.calibrate(q, hz, nodes, LEVELS, max_offset=1.5)
    expected = np.log1p(hz["Actual"]) - np.log1p(hz["Policy"])
    assert (q2["q_hi80"].to_numpy() >= expected.max() - 1e-9).all()


def test_calibration_measures_coverage_with_out_of_sample_quantiles():
    """Offsets must be measured with q_fit (fitted before the test window), then added to q."""
    comb, nodes = residual_frame()
    q, _ = iv.residual_quantiles(comb, nodes, LEVELS, min_obs=8, horizon=12)
    hz = comb.rename(columns={"Pred": "Policy"})
    narrow = q.copy()
    narrow[iv._qcols(LEVELS)] *= 0.5                       # an interval that under-covers
    _, cal_in = iv.calibrate(q, hz, nodes, LEVELS, max_offset=1.5)
    q2, cal_out = iv.calibrate(q, hz, nodes, LEVELS, max_offset=1.5, q_fit=narrow)
    assert (_band_rows(cal_out)["Offset_Hi"].to_numpy() > _band_rows(cal_in)["Offset_Hi"].to_numpy()).all()
    assert (q2["q_hi80"].to_numpy() >= q["q_hi80"].to_numpy() - 1e-12).all()


def test_apply_interval_orders_bounds():
    lo, hi = iv.apply_interval(np.array([100.0, 0.0]), np.array([-0.2, -0.1]), np.array([0.3, 0.1]))
    assert lo[0] < 100 < hi[0] and lo[1] <= 0 <= hi[1] + 1e-12


def test_block_quantiles_are_pooled_and_signed():
    comb, nodes = residual_frame()
    bq, _ = iv.block_quantiles(comb, nodes, LEVELS, min_obs=8)
    assert set(bq["k"]) == {3, 6, 12}
    assert (bq["q_lo80"] <= 0).all() and (bq["q_hi80"] >= 0).all()
    assert bq["Source"].str.contains("own totals|pooled").all()


def test_over_wide_intervals_are_narrowed_within_bounds():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.05, 400)
    wide_lo, wide_hi = np.full(400, -0.2), np.full(400, 0.2)          # ~100% coverage: far too wide
    c = iv._narrowing_factor(wide_lo, wide_hi, r, 80)
    assert iv.NARROW_FLOOR <= c < 1.0
    ok_lo, ok_hi = np.full(400, -0.064), np.full(400, 0.064)          # ~80% coverage: leave alone
    assert iv._narrowing_factor(ok_lo, ok_hi, r, 80) == 1.0
    assert iv._narrowing_factor(wide_lo[:10], wide_hi[:10], r[:10], 80) == 1.0   # too little evidence
