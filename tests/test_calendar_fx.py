import numpy as np
import pandas as pd

from src.forecasting import calendar_fx as cx


def test_cycle_runs_26th_to_25th_and_finds_festivals():
    nov26 = cx.cycle_features(pd.Timestamp("2026-11-01"))     # 26 Oct - 25 Nov 2026: Diwali 8 Nov
    assert nov26["Diwali"] == 1 and nov26["Eid"] == 0 and 20 <= nov26["WD"] <= 23
    assert cx.cycle_features(pd.Timestamp("2025-10-01"))["Diwali"] == 1   # 26 Sep - 25 Oct 2025: Diwali 20 Oct
    assert cx.cycle_features(pd.Timestamp("2025-11-01"))["Diwali"] == 0   # 26 Oct - 25 Nov 2025


def test_same_factor_for_every_receipt_series_in_a_month_and_none_for_submissions():
    months = pd.date_range("2023-01-01", periods=48, freq="MS")
    D = cx.deltas(months)
    coefs = {30: np.array([0.05, -0.1, 0.0, -0.1])}
    fr = pd.DataFrame({"node_id": ["Received|_ORG|_TOTAL", "Received|C|R1", "Submitted|C|R1"], "o": 30, "h": 2})
    f = cx.factors(fr, D, coefs)
    assert np.isclose(f[0], f[1]) and f[2] == 1.0
