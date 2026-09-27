import copy

import numpy as np

from src.forecasting.backtest import global_origins, node_plan, run_backtest

MODELS = ["SNaive", "SNaiveGrowth", "TSB", "ETS", "GlobalLGBM"]


def _run(panel, ids):
    return run_backtest(panel, ids, MODELS, horizon=12, n_origins=14,
                        lgbm_params={"n_estimators": 30}, seed=1, n_jobs=1)


def test_no_future_leakage(panel):
    """Changing values AFTER origin o* must not change any forecast made at origins <= o*."""
    ids = panel.nodes.loc[panel.nodes["Status"] == "Mature", "node_id"].tolist()
    base = _run(panel, ids)
    T = len(panel.months)
    o_star = global_origins(T, 14)[5]
    shocked = copy.deepcopy(panel)
    for nid in ids:
        s = panel.start[nid]
        v = shocked.y[nid].copy()
        v[max(0, o_star + 1 - s):] *= 10.0
        shocked.y[nid] = v
    after = _run(shocked, ids)
    key = ["node_id", "Model", "o", "h"]
    a = base.preds[base.preds["o"] <= o_star].sort_values(key).reset_index(drop=True)
    b = after.preds[after.preds["o"] <= o_star].sort_values(key).reset_index(drop=True)
    assert len(a) > 0 and len(a) == len(b)
    assert np.allclose(a["Pred"].to_numpy(), b["Pred"].to_numpy())
    # ...while forecasts after the shock do change (the test is sensitive)
    assert not np.allclose(base.preds[base.preds["o"] > o_star]["Pred"].to_numpy(),
                           after.preds[after.preds["o"] > o_star]["Pred"].to_numpy())


def test_fair_comparison_same_pairs_for_every_model(panel):
    ids = panel.nodes.loc[panel.nodes["Status"] == "Mature", "node_id"].tolist()
    bt = _run(panel, ids)
    for nid, g in bt.preds.groupby("node_id"):
        sets = g.groupby("Model").apply(lambda d: frozenset(zip(d["o"], d["h"])))
        assert sets.nunique() == 1, nid
    assert bt.failures.empty
    assert (bt.final.groupby(["node_id", "Model"]).size() == 12).all()


def test_node_plan_requires_enough_training():
    L, origins, elig = node_plan(n_months=30, s=50, origins=list(range(60, 80)), models=["SNaive", "ETS"])
    assert L == 18 and all(o - 50 + 1 >= 18 for o in origins)
    assert elig == ["SNaive"]                    # ETS needs 24 months at the first origin
