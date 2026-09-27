"""Synthetic data fixtures — no production data is used in tests."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.config import _RAW, load_settings
from src.data.prepare import build_panel, clean_raw

N_MONTHS = 48


def make_raw(months: int = N_MONTHS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=months, freq="MS")
    season = 1 + 0.4 * np.sin(2 * np.pi * (dates.month - 1) / 12)
    spec = [("Alpha", "North", 200, range(months)),
            ("Alpha", "South", 120, range(months)),
            ("Alpha", "New", 30, range(months - 8, months)),        # emerging: 8 months
            ("Beta", "All", 80, range(months)),
            ("Beta", "Closed", 50, range(months - 10))]              # inactive: stopped 10 months ago
    rows = []
    for c, r, lvl, idx in spec:
        for i in idx:
            rec = int(rng.poisson(lvl * season[i]))
            rows.append({"Date": dates[i], "Country": c, "Region": r, "Received": rec,
                         "Submitted": int(rng.binomial(rec, 0.7))})
    df = pd.DataFrame(rows)
    return df[~((df["Region"] == "South") & (df["Date"] == dates[20]))]   # a gap -> must become 0


def make_settings(**overrides):
    r = copy.deepcopy(_RAW)
    r["models"]["enabled"] = ["SNaive", "SNaiveGrowth", "TSB", "ETS", "GlobalLGBM"]
    r["models"]["lgbm"]["n_estimators"] = 30
    r["backtest"]["n_origins"] = 14
    r["selection"]["honest_eval_origins"] = 6
    r["runtime"]["n_jobs"] = 1
    for k, v in overrides.items():
        sec, key = k.split("__")
        r[sec][key] = v
    return load_settings(r)


@pytest.fixture
def raw():
    return clean_raw(make_raw())


@pytest.fixture
def settings():
    return make_settings()


@pytest.fixture
def panel(raw, settings):
    return build_panel(raw, settings.data)
