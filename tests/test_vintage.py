import numpy as np
import pandas as pd

from src.forecasting import vintage as vt


def _nodes():
    return pd.DataFrame([
        {"node_id": "Received|_ORG|_TOTAL", "Level": "Organisation", "Country": "_ORG", "Metric": "Received"},
        {"node_id": "Received|C|_TOTAL", "Level": "Country", "Country": "C", "Metric": "Received"},
        {"node_id": "Received|C|R1", "Level": "Region", "Country": "C", "Metric": "Received"},
        {"node_id": "Submitted|C|R1", "Level": "Region", "Country": "C", "Metric": "Submitted"},
    ])


def _honest(under=0.8, origins=range(0, 12)):
    """Forecasts always `under` x actual (a persistent lag), actual = 100 every month."""
    rows = [{"node_id": n, "o": o, "h": h, "Policy": 100 * under, "Actual": 100.0}
            for n in _nodes()["node_id"] for o in origins for h in range(1, 4)]
    return pd.DataFrame(rows)


def test_signal_uses_only_months_known_at_the_origin():
    h = _honest()
    g = h[h["node_id"] == "Received|C|_TOTAL"].copy()
    s = vt._signal(g, 8, "Policy")
    assert np.isclose(s, np.log(1 / 0.8))
    # changing an actual AFTER the origin must not change the signal
    g.loc[(g["o"] + g["h"]) > 8, "Actual"] = 1e6
    assert np.isclose(vt._signal(g, 8, "Policy"), s)


def test_correction_applies_country_signal_to_receipts_only():
    h = _honest()
    sig = vt.signals(h, [8])
    f = vt.factors(pd.DataFrame({"node_id": _nodes()["node_id"], "o": 8}), _nodes(), sig)
    fac = dict(zip(_nodes()["node_id"], f))
    assert fac["Received|_ORG|_TOTAL"] == 1.0 and fac["Submitted|C|R1"] == 1.0
    assert np.isclose(fac["Received|C|R1"], np.exp(vt.LAMBDA * np.log(1.25)))
    t, _ = vt.walk_forward_test(h, _nodes())
    assert vt.decide(t)          # a persistent lag is corrected and the test adopts it
