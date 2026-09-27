import numpy as np
import pytest

from src.forecasting.models import LOCAL_MODELS, SNaive, SNaiveGrowth, TSB, pooled_seasonal_forecast, seasonal_index


def seasonal_series(n=48, level=100.0, growth=0.0):
    t = np.arange(n)
    return level * (1 + growth) ** (t / 12) * (1 + 0.3 * np.sin(2 * np.pi * t / 12))


def test_snaive_repeats_last_year():
    y = seasonal_series()
    assert np.allclose(SNaive()(y, 12, 1), y[-12:])
    assert np.allclose(SNaive()(y, 14, 1)[12:], y[-12:-10])


def test_snaive_growth_applied_once_not_compounded():
    y = seasonal_series(growth=0.2)
    f = SNaiveGrowth()(y, 12, 1)
    ratio = f / y[-12:]
    assert np.allclose(ratio, ratio[0])                      # one constant factor
    assert 1.15 < ratio[0] < 1.25


def test_tsb_flat_and_nonnegative():
    y = np.array([0, 0, 3, 0, 1, 0, 0, 2, 0, 0, 0, 1] * 2, dtype=float)
    f = TSB()(y, 12, 1)
    assert np.allclose(f, f[0]) and f[0] >= 0


@pytest.mark.parametrize("name", list(LOCAL_MODELS))
def test_every_model_finite_nonnegative(name):
    y = seasonal_series(n=48)
    f = LOCAL_MODELS[name](y, 12, 1)
    assert f.shape == (12,) and np.all(np.isfinite(f)) and np.all(f >= 0)


@pytest.mark.parametrize("name", ["ETS", "Theta", "SARIMA_Airline"])
def test_statistical_models_track_a_clean_seasonal_series(name):
    y = seasonal_series(n=60)
    f = LOCAL_MODELS[name](y, 12, 1)
    assert np.mean(np.abs(f - y[-12:]) / y[-12:]) < 0.10


def test_min_history_enforced():
    with pytest.raises(ValueError):
        LOCAL_MODELS["SARIMA_Airline"](seasonal_series(n=20), 12, 1)


def test_pooled_seasonal_uses_parent_pattern():
    parent = seasonal_series(n=36)
    months = (np.arange(36) % 12) + 1
    idx = seasonal_index(parent, months)
    assert abs(idx.mean() - 1) < 1e-9
    child = parent[-6:] / 10
    f = pooled_seasonal_forecast(child, months[-6:], idx, months[:12])
    assert np.allclose(f / f.mean(), idx / idx.mean(), atol=0.02)
