import numpy as np
import pandas as pd

from src.forecasting import selection as sel


def toy_preds(switch_at=5, n_origins=10, h=3):
    """Model A is perfect before `switch_at`, model B is perfect from then on; S is mediocre."""
    rows = []
    for o in range(n_origins):
        for k in range(1, h + 1):
            actual = 100.0
            a = actual if o < switch_at else 160.0
            b = 160.0 if o < switch_at else actual
            for m, p in (("A", a), ("B", b), ("S", 120.0)):
                rows.append({"node_id": "n", "Model": m, "o": o, "h": k, "t": o + k, "Pred": p, "Actual": actual})
    return pd.DataFrame(rows)


def test_scores_horizon_balanced_and_relative():
    sc = sel.score_models(toy_preds(), "S").set_index("Model")
    assert np.isclose(sc.loc["S", "HB_MAE"], 20.0)
    assert np.isclose(sc.loc["S", "Rel_to_Benchmark"], 1.0)


def test_mean_all_uses_every_model_and_topk_requires_beating_benchmark():
    sc = sel.score_models(toy_preds(), "S")
    s_all = sel.select_models(sc, "S", 1, True, "mean_all").iloc[0]
    assert set(s_all["Selected_Models"].split(" + ")) == {"A", "B", "S"}
    bad = sc.copy()
    bad.loc[bad["Model"] != "S", "Rel_to_Benchmark"] = 1.5
    bad.loc[bad["Model"] != "S", "HB_MAE"] = 99.0
    assert sel.select_models(bad, "S", 2, True, "top_k").iloc[0]["Selected_Models"] == "S"


def test_walk_forward_selection_uses_only_the_past():
    """At origin 5 only A has a good record, so top-1 must pick A even though B is perfect from 5 on."""
    hz = sel.honest_evaluation(toy_preds(), "S", top_k=1, must_beat=True, n_eval_origins=5, policy="top_k")
    at5 = hz[hz["o"] == 5]
    assert set(at5["Top_K_Models"]) == {"A"}
    assert np.allclose(at5["Policy"], 160.0)


def test_combine_equal_weights():
    p = toy_preds()
    c = sel.combine(p, pd.Series({"n": "A + B"}))
    assert np.allclose(c["Pred"], 130.0)
