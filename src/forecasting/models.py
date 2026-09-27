# -*- coding: utf-8 -*-
"""
Model registry.

Every local model implements ``forecast(y, h, start_month) -> np.ndarray`` where
``y`` is the training history ONLY (values up to and including the forecast
origin). The backtest harness is the only place that slices history, so no model
can see the future. Models are deterministic.

Count data are modelled on the log1p scale where a model is additive
(ETS, Theta, SARIMA) so seasonality is proportional to level and forecasts stay
non-negative after ``expm1``; the back-transform gives a median-type forecast.

To add a model: subclass ``LocalModel``, set name/version/min_history, register
it in ``LOCAL_MODELS`` and enable it in config/pipeline.toml.
"""
from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

SEASON = 12


class LocalModel:
    name: str = ""
    version: str = "1.0"
    family: str = ""
    min_history: int = SEASON
    description: str = ""

    def forecast(self, y: np.ndarray, h: int, start_month: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, y: np.ndarray, h: int, start_month: int) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if len(y) < self.min_history:
            raise ValueError(f"{self.name} needs {self.min_history} months, got {len(y)}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = np.asarray(self.forecast(y, h, start_month), dtype=float).reshape(-1)[:h]
        if out.shape[0] != h or not np.all(np.isfinite(out)):
            raise FloatingPointError(f"{self.name} produced a non-finite forecast")
        return np.clip(out, 0.0, None)


# ─────────────────────────────── benchmarks ───────────────────────────────
class SNaive(LocalModel):
    name, family, min_history = "SNaive", "Benchmark", SEASON
    description = "Seasonal naive: same month last year"

    def forecast(self, y, h, start_month):
        last = y[-SEASON:]
        return np.array([last[i % SEASON] for i in range(h)])


class SNaiveGrowth(LocalModel):
    name, family, min_history = "SNaiveGrowth", "Benchmark", 2 * SEASON
    description = "Same month last year x trailing 12-month YoY growth (applied once, not compounded)"

    def forecast(self, y, h, start_month):
        cur, prev = y[-SEASON:].sum(), y[-2 * SEASON:-SEASON].sum()
        g = float(np.clip(cur / prev, 0.5, 2.0)) if prev > 0 else 1.0
        return SNaive().forecast(y, h, start_month) * g


# ─────────────────────────────── statistical ──────────────────────────────
class ETS(LocalModel):
    name, family, min_history = "ETS", "Statistical", 2 * SEASON
    description = "Holt-Winters on log1p: additive seasonality, damped or no trend, chosen by AICc"

    def forecast(self, y, h, start_month):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        z = np.log1p(y)
        best, best_aicc = None, np.inf
        for trend, damped in (("add", True), (None, False)):
            try:
                fit = ExponentialSmoothing(z, trend=trend, damped_trend=damped, seasonal="add",
                                           seasonal_periods=SEASON,
                                           initialization_method="estimated").fit(
                    optimized=True, method="L-BFGS-B")   # default (basinhopping) is ~25x slower, same fit
            except Exception:
                continue
            aicc = fit.aicc if np.isfinite(fit.aicc) else fit.aic
            if np.isfinite(aicc) and aicc < best_aicc:
                best, best_aicc = fit, aicc
        if best is None:
            raise RuntimeError("ETS failed for all configurations")
        return np.expm1(best.forecast(h))


class Theta(LocalModel):
    name, family, min_history = "Theta", "Statistical", 2 * SEASON
    description = "Theta method on log1p with additive seasonal adjustment (M3 benchmark winner)"

    def forecast(self, y, h, start_month):
        from statsmodels.tsa.forecasting.theta import ThetaModel
        fit = ThetaModel(np.log1p(y), period=SEASON, deseasonalize=True,
                         method="additive", use_test=False).fit()
        return np.expm1(np.asarray(fit.forecast(h)))


class SARIMAAirline(LocalModel):
    name, family, min_history = "SARIMA_Airline", "Statistical", 3 * SEASON
    description = "SARIMA(0,1,1)(0,1,1)12 'airline' model on log1p"

    def forecast(self, y, h, start_month):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        fit = SARIMAX(np.log1p(y), order=(0, 1, 1), seasonal_order=(0, 1, 1, SEASON),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
        return np.expm1(np.asarray(fit.forecast(h)))


class TSB(LocalModel):
    name, family, min_history = "TSB", "Intermittent", 6
    description = "Teunter-Syntetos-Babai intermittent-demand model (flat rate), smoothing tuned in-sample"
    grid = (0.05, 0.1, 0.2, 0.3)

    @staticmethod
    def _run(y, a, b):
        occ = (y > 0).astype(float)
        p = occ[: min(len(y), SEASON)].mean() or 1e-6
        nz = y[y > 0]
        z = nz[: min(len(nz), 3)].mean() if len(nz) else 0.0
        err = 0.0
        for t in range(len(y)):
            err += (y[t] - p * z) ** 2
            p = p + b * (occ[t] - p)
            if y[t] > 0:
                z = z + a * (y[t] - z)
        return p * z, err

    def forecast(self, y, h, start_month):
        best = min(((self._run(y, a, b), a, b) for a in self.grid for b in self.grid),
                   key=lambda t: t[0][1])
        return np.full(h, best[0][0])


LOCAL_MODELS: Dict[str, LocalModel] = {
    m.name: m for m in (SNaive(), SNaiveGrowth(), ETS(), Theta(), SARIMAAirline(), TSB())
}


def model_catalogue(enabled: Iterable[str]) -> List[dict]:
    enabled = set(enabled)
    rows = [{"Model": m.name, "Version": m.version, "Family": m.family,
             "Min_History_Months": m.min_history, "Description": m.description}
            for m in LOCAL_MODELS.values() if m.name in enabled]
    if GlobalLGBMDirect.name in enabled:
        rows.append({"Model": GlobalLGBMDirect.name, "Version": GlobalLGBMDirect.version,
                     "Family": "Machine learning", "Min_History_Months": SEASON,
                     "Description": GlobalLGBMDirect.description})
    return rows


# ─────────────────────────────── global ML ────────────────────────────────
class GlobalLGBMDirect:
    """
    One LightGBM per horizon h, trained across all tournament series at once
    (scale-free log-ratio target), i.e. *direct* multi-horizon forecasting — no
    recursion, so errors cannot compound. For origin o a training row (t, h)
    is used only if its target month t+h <= o.
    """
    name = "GlobalLGBM"
    version = "1.0"
    min_history = SEASON
    description = ("Global LightGBM, one model per horizon (direct), log-ratio target, "
                   "12 scaled lags + YoY + calendar features, recency-weighted")

    def __init__(self, params: Optional[dict] = None, seed: int = 42):
        p = dict(params or {})
        self.half_life = float(p.pop("recency_half_life_months", 36))
        self.params = dict(objective="l1", n_estimators=400, learning_rate=0.03, num_leaves=15,
                           min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=1,
                           random_state=seed)
        self.params.update(p)

    @staticmethod
    def features(v: np.ndarray, s: int) -> Dict[str, np.ndarray]:
        """Features for every t >= s+11 using only v[..t]. v is the full-grid series."""
        T = len(v)
        ts = np.arange(s + SEASON - 1, T)
        if len(ts) == 0:
            return {"t": ts}
        lv = np.log1p(np.clip(v, 0, None))
        win = np.lib.stride_tricks.sliding_window_view(v, SEASON)[ts - SEASON + 1]   # v[t-11..t]
        lwin = np.lib.stride_tricks.sliding_window_view(lv, SEASON)[ts - SEASON + 1]
        level = np.log1p(win.mean(axis=1))
        lags = lwin[:, ::-1] - level[:, None]                                        # lag0..lag11
        s12 = win.sum(axis=1)
        prev = np.full(len(ts), np.nan)
        ok = ts - 2 * SEASON + 1 >= s
        csum = np.concatenate([[0.0], np.cumsum(v)])
        prev[ok] = csum[ts[ok] - SEASON + 1] - csum[ts[ok] - 2 * SEASON + 1]
        yoy = np.log1p(s12) - np.log1p(prev)
        return {"t": ts, "level": level, "lags": lags, "yoy": yoy}

    def fit_predict(self, series: Sequence[dict], cal_month: np.ndarray, origins: Sequence[int],
                    h: int) -> Dict[tuple, np.ndarray]:
        """
        series: dicts with keys id, v (full-grid values), s (start idx), metric_code, level_code.
        cal_month: calendar month (1-12) of every global index (length >= T + h).
        Returns {(id, origin): forecast array of length h} for series with >= 12 months at origin.
        """
        import lightgbm as lgb
        feats = []
        for sr in series:
            f = self.features(sr["v"], sr["s"])
            if len(f["t"]):
                feats.append((sr, f))
        out: Dict[tuple, np.ndarray] = {}
        for o in origins:
            preds_o: Dict[str, List[float]] = {}
            for step in range(1, h + 1):
                Xtr, ytr, wtr, Xte, ids = [], [], [], [], []
                for sr, f in feats:
                    t = f["t"]
                    m = t + step <= o
                    if m.any():
                        tt = t[m]
                        Xtr.append(self._design(f, m, tt + step, cal_month, sr, step))
                        ytr.append(np.log1p(sr["v"][tt + step]) - f["level"][m])
                        wtr.append(0.5 ** ((o - (tt + step)) / self.half_life))
                    k = np.nonzero(t == o)[0]
                    if len(k):
                        mm = np.zeros(len(t), bool)
                        mm[k] = True
                        Xte.append(self._design(f, mm, np.array([o + step]), cal_month, sr, step))
                        ids.append((sr["id"], f["level"][k[0]]))
                if not Xtr or not Xte:
                    continue
                model = lgb.LGBMRegressor(**self.params)
                model.fit(np.vstack(Xtr), np.concatenate(ytr), sample_weight=np.concatenate(wtr))
                yhat = model.predict(np.vstack(Xte))
                for (sid, lev), p in zip(ids, yhat):
                    preds_o.setdefault(sid, []).append(max(0.0, float(np.expm1(lev + p))))
            for sid, vals in preds_o.items():
                if len(vals) == h:
                    out[(sid, o)] = np.array(vals)
        return out

    @staticmethod
    def _design(f, mask, target_idx, cal_month, sr, step) -> np.ndarray:
        n = int(mask.sum())
        seasonal = f["lags"][mask][:, SEASON - step]            # value 12 months before target
        return np.column_stack([
            f["lags"][mask], f["level"][mask], f["yoy"][mask], seasonal,
            cal_month[target_idx], np.full(n, sr["metric_code"]), np.full(n, sr["level_code"]),
        ])


# ─────────────────────────────── emerging series ──────────────────────────
def seasonal_index(parent: np.ndarray, parent_months: np.ndarray, years: int = 3) -> np.ndarray:
    """Multiplicative month-of-year index (mean 1) from the parent's last `years` years."""
    n = min(len(parent), years * SEASON)
    v, mo = parent[-n:], parent_months[-n:]
    overall = v.mean() if v.mean() > 0 else 1.0
    idx = np.array([v[mo == m].mean() / overall if np.any(mo == m) else 1.0 for m in range(1, 13)])
    idx = np.where(idx > 0, idx, 1.0)
    return idx / idx.mean()


def pooled_seasonal_forecast(y: np.ndarray, y_months: np.ndarray, index: np.ndarray,
                             target_months: np.ndarray, k: int = 6) -> np.ndarray:
    """Recent deseasonalised level (last k months) x the parent's seasonal index."""
    k = min(k, len(y))
    level = float(np.mean(y[-k:] / index[y_months[-k:] - 1]))
    return np.clip(level * index[target_months - 1], 0.0, None)
