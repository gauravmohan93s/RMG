import numpy as np
import pandas as pd

from src.forecasting import reconcile as rc


def nodes():
    return pd.DataFrame([
        {"node_id": "org", "Level": "Organisation", "Country": "_ORG", "Region": "_TOTAL"},
        {"node_id": "A", "Level": "Country", "Country": "A", "Region": "_TOTAL"},
        {"node_id": "A1", "Level": "Region", "Country": "A", "Region": "1"},
        {"node_id": "A2", "Level": "Region", "Country": "A", "Region": "2"},
        {"node_id": "B1", "Level": "Region", "Country": "B", "Region": "1"},
    ])


def test_summing_matrix():
    S, ids, b = rc.summing_matrix(nodes())
    assert ids == ["org", "A", "A1", "A2", "B1"] and b == ["A1", "A2", "B1"]
    assert S.tolist() == [[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_reconciled_forecasts_are_coherent_and_nonnegative():
    S, ids, _ = rc.summing_matrix(nodes())
    yhat = np.array([[330.0], [210.0], [100.0], [90.0], [110.0]])        # incoherent base forecasts
    for method in ("wls_var", "bottom_up"):
        yt, used = rc.reconcile(yhat, S, method, var=np.array([400, 200, 50, 50, 60.0]))
        assert used == method
        assert np.isclose(yt[0, 0], yt[2:, 0].sum()) and np.isclose(yt[1, 0], yt[2, 0] + yt[3, 0])
        assert (yt >= 0).all()


def test_mint_falls_back_without_complete_residuals():
    S, _, _ = rc.summing_matrix(nodes())
    E = np.random.default_rng(0).normal(size=(24, 5))
    E[0, 2] = np.nan
    _, used = rc.reconcile(np.ones((5, 1)), S, "mint_shrink", var=np.ones(5), E=E)
    assert used == "wls_var"


def test_mint_shrink_runs_with_complete_residuals():
    S, _, _ = rc.summing_matrix(nodes())
    E = np.random.default_rng(0).normal(size=(24, 5))
    yt, used = rc.reconcile(np.array([[330.0], [210.0], [100.0], [90.0], [110.0]]), S, "mint_shrink", E=E)
    assert used == "mint_shrink" and np.isclose(yt[0, 0], yt[2:, 0].sum())
