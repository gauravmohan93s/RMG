import numpy as np
import pandas as pd

from src.forecasting import funnel as fn


def _setup(rate=0.7, T=40):
    rng = np.random.default_rng(0)
    R = 1000 + 300 * np.sin(np.arange(T) / 2) + rng.normal(0, 20, T)
    S = rate * R
    Y = {"Received|C|R": R, "Submitted|C|R": S}
    nodes = pd.DataFrame({"node_id": list(Y), "Level": ["Region", "Region"]})
    return Y, nodes


def test_weights_recover_conversion_rate_without_future_data():
    Y, _ = _setup(0.7)
    w = fn._weights(Y["Received|C|R"], Y["Submitted|C|R"], 30)
    assert np.isclose(w.sum(), 0.7, atol=0.02) and (w >= 0).all()


def test_funnel_uses_actual_receipts_to_date_and_forecast_after():
    Y, nodes = _setup(0.5)
    o = 30
    rows = [{"node_id": n, "o": o, "h": h, "Policy": (999.0 if n.startswith("Submitted") else Y[n][o + h] * 1.0),
             "Actual": Y[n][o + h]} for n in Y for h in (1, 2, 3)]
    fr = pd.DataFrame(rows)
    f = fn.funnel_forecast(fr, "Policy", Y)
    sub = fr["node_id"].str.startswith("Submitted")
    # perfect receipt forecast + exact conversion -> funnel reproduces actual submissions
    assert np.allclose(f[sub], fr.loc[sub, "Actual"], rtol=0.03)
    t, _ = fn.walk_forward_test(fr, nodes, Y)
    assert fn.decide(t) == "funnel"                       # the 999 model forecast loses clearly
