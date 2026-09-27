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

    uses_state: bool = False   # True: the harness passes a per-series dict kept across origins (in time order)

    def __call__(self, y: np.ndarray, h: int, start_month: int, state: Optional[dict] = None) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if len(y) < self.min_history:
            raise ValueError(f"{self.name} needs {self.min_history} months, got {len(y)}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = self.forecast(y, h, start_month, state=state) if self.uses_state else self.forecast(y, h, start_month)
            out = np.asarray(f, dtype=float).reshape(-1)[:h]
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


class Croston(LocalModel):
    name, family, min_history = "Croston", "Intermittent", 6
    description = "Croston's method with Syntetos-Boylan bias correction (SBA), flat rate"

    def forecast(self, y, h, start_month):
        a = 0.1
        nz = np.nonzero(y > 0)[0]
        if len(nz) == 0:
            return np.zeros(h)
        z, p, last = float(y[nz[0]]), float(nz[0] + 1), nz[0]
        for t in nz[1:]:
            z += a * (y[t] - z)
            p += a * ((t - last) - p)
            last = t
        return np.full(h, (1 - a / 2) * z / max(p, 1.0))


# ── v1 month-on-month family (re-implemented leak-free: every ratio from training data only) ──
def _mom(y: np.ndarray) -> np.ndarray:
    """Month-on-month ratios y[t]/y[t-1] (NaN where the previous month is 0)."""
    prev = y[:-1]
    return np.where(prev > 0, y[1:] / np.where(prev > 0, prev, 1.0), np.nan)


def _chain(last: float, ratios: np.ndarray) -> np.ndarray:
    out, lvl = [], float(last)
    for r in ratios:
        lvl *= r
        out.append(lvl)
    return np.array(out)


def _month_of(n: int, start_month: int) -> np.ndarray:
    """Calendar month (1-12) of each of the n training observations (last one = start_month-1)."""
    last = (start_month - 2) % 12 + 1
    return ((last - 1 - np.arange(n)[::-1]) % 12) + 1


class LastYearMoM(LocalModel):
    name, family, min_history = "LastYear_MoM", "v1 month-on-month", 2 * SEASON
    description = "v1 Model01: last year's month-on-month changes applied to the latest level (compounding, ±50%/step)"

    def forecast(self, y, h, start_month):
        r = _mom(y)[-SEASON:]
        r = np.clip(np.nan_to_num(r, nan=1.0), 0.5, 1.5)
        return _chain(y[-1], [r[i % SEASON] for i in range(h)])


class AvgMonthlyMoM(LocalModel):
    name, family, min_history = "AvgMonthly_MoM", "v1 month-on-month", 2 * SEASON
    description = "v1 Model04: average month-on-month change per calendar month (70% last 2 years / 30% older)"

    def _ratio_map(self, y, start_month):
        r = _mom(y)
        months = _month_of(len(y), start_month)[1:]
        recent = np.arange(len(r)) >= len(r) - 24
        out = np.ones(13)
        for m in range(1, 13):
            a = r[(months == m) & recent]
            b = r[(months == m) & ~recent]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) and len(b):
                out[m] = 0.7 * a.mean() + 0.3 * b.mean()
            elif len(a) or len(b):
                out[m] = (a if len(a) else b).mean()
        return np.clip(out, 0.5, 1.5)

    def forecast(self, y, h, start_month):
        rm = self._ratio_map(y, start_month)
        return _chain(y[-1], [rm[(start_month - 1 + i) % 12 + 1] for i in range(h)])


class YoYMoM(LocalModel):
    name, family, min_history = "YoY_MoM", "v1 month-on-month", 2 * SEASON
    description = "v1 Model02: blend of last year's and the long-run average month-on-month change (seasonal prior)"

    def forecast(self, y, h, start_month):
        last = np.clip(np.nan_to_num(_mom(y)[-SEASON:], nan=1.0), 0.5, 1.5)
        prior = AvgMonthlyMoM()._ratio_map(y, start_month)
        ratios = [0.5 * last[i % SEASON] + 0.5 * prior[(start_month - 1 + i) % 12 + 1] for i in range(h)]
        return _chain(y[-1], ratios)


class RollingMoM(LocalModel):
    name, family, min_history = "Rolling_MoM", "v1 month-on-month", 2 * SEASON
    description = "v1 Model03: rolling 12-month median month-on-month change blended with the calendar-month average"

    def forecast(self, y, h, start_month):
        r = _mom(y)[-SEASON:]
        med = float(np.nanmedian(r)) if np.isfinite(r).any() else 1.0
        prior = AvgMonthlyMoM()._ratio_map(y, start_month)
        ratios = [np.clip(0.5 * med + 0.5 * prior[(start_month - 1 + i) % 12 + 1], 0.5, 1.5) for i in range(h)]
        return _chain(y[-1], ratios)


# ── additional statistical models ──
class ProphetStyle(LocalModel):
    name, family, min_history = "ProphetStyle", "Statistical", 2 * SEASON
    description = ("Prophet model structure (v1 Model05) on log1p: piecewise-linear trend with 10 L1-regularised "
                   "changepoints (first 80% of history) + Fourier yearly seasonality (K=4). The prophet library "
                   "1.2.1 is incompatible with pandas 3, so the same model is fitted with scikit-learn")

    def forecast(self, y, h, start_month):
        from sklearn.linear_model import Lasso
        n = len(y)
        t = np.arange(n + h) / max(n - 1, 1)
        months = np.concatenate([_month_of(n, start_month), (start_month - 1 + np.arange(h)) % 12 + 1])
        cps = np.linspace(0, 0.8, 12)[1:-1]
        X = [t] + [np.maximum(t - c, 0) for c in cps]
        for k in range(1, 5):
            X += [np.sin(2 * np.pi * k * months / 12), np.cos(2 * np.pi * k * months / 12)]
        X = np.column_stack(X)
        fit = Lasso(alpha=1e-3, max_iter=5000, tol=1e-3).fit(X[:n], np.log1p(y))
        return np.expm1(fit.predict(X[n:]))


class AutoARIMA(LocalModel):
    name, family, min_history = "AutoARIMA", "Statistical", 3 * SEASON
    description = ("Seasonal auto-ARIMA (v1 Model06): pmdarima stepwise AICc search (D=1, order <= 4) on log1p; "
                   "in backtests the order is re-selected every 6 origins from data up to that origin and "
                   "parameters are refitted in between (causal, ~4x faster); the production fit on full "
                   "history always runs a fresh order search")
    uses_state = True
    reselect_every = 6

    def forecast(self, y, h, start_month, state=None):
        import pmdarima as pm
        z = np.log1p(y)
        if state is not None and "order" in state and len(y) - state["n_at_select"] < self.reselect_every:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            fit = SARIMAX(z, order=state["order"], seasonal_order=state["seasonal_order"],
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=100)
            return np.expm1(np.asarray(fit.forecast(h)))
        fit = pm.auto_arima(z, seasonal=True, m=SEASON, D=1, stepwise=True, information_criterion="aicc",
                            max_p=2, max_q=2, max_P=1, max_Q=1, max_order=4, maxiter=30,
                            suppress_warnings=True, error_action="ignore")
        if state is not None:
            state.update(order=fit.order, seasonal_order=fit.seasonal_order, n_at_select=len(y))
        return np.expm1(np.asarray(fit.predict(h)))


class STLETS(LocalModel):
    name, family, min_history = "STL_ETS", "Statistical", 2 * SEASON
    description = "Robust STL seasonal decomposition + damped-trend ETS on the seasonally adjusted series (log1p)"

    def forecast(self, y, h, start_month):
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        from statsmodels.tsa.forecasting.stl import STLForecast
        fit = STLForecast(np.log1p(y), ETSModel, model_kwargs={"error": "add", "trend": "add", "damped_trend": True},
                          period=SEASON, robust=True).fit(fit_kwargs={"disp": False})
        return np.expm1(np.asarray(fit.forecast(h)))


LOCAL_MODELS: Dict[str, LocalModel] = {
    m.name: m for m in (SNaive(), SNaiveGrowth(), ETS(), Theta(), SARIMAAirline(), TSB(), Croston(),
                        LastYearMoM(), AvgMonthlyMoM(), YoYMoM(), RollingMoM(), ProphetStyle(), AutoARIMA(), STLETS())
}


def model_catalogue(enabled: Iterable[str]) -> List[dict]:
    enabled = set(enabled)
    rows = [{"Model": m.name, "Version": m.version, "Family": m.family,
             "Min_History_Months": m.min_history, "Description": m.description}
            for m in LOCAL_MODELS.values() if m.name in enabled]
    rows += [{"Model": c.name, "Version": c.version, "Family": c.family, "Min_History_Months": c.min_history,
              "Description": c.description} for c in GLOBAL_MODELS.values() if c.name in enabled]
    return rows


# ─────────────────────────────── global ML ────────────────────────────────
class GlobalDirect:
    """
    One model per horizon h trained across all tournament series at once (scale-free
    log-ratio target): *direct* multi-horizon forecasting — no recursion, so errors cannot
    compound. For origin o a training row (t, h) is used only if its target month t+h <= o,
    and features at t use values <= t only. Internal signal: the partner metric's recent
    history (Received <-> Submitted funnel). Subclasses choose the learner.
    """
    name = "GlobalDirect"
    version = "1.1"
    min_history = SEASON
    family = "Machine learning"
    description = ""

    def __init__(self, params: Optional[dict] = None, seed: int = 42):
        p = dict(params or {})
        self.half_life = float(p.pop("recency_half_life_months", 36))
        self.seed = seed
        self.params = p

    def make_model(self):  # pragma: no cover
        raise NotImplementedError

    def fit_model(self, X, y, w):
        m = self.make_model()
        m.fit(X, y, sample_weight=w)
        return m

    @staticmethod
    def features(v: np.ndarray, s: int, pv: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Features for every t >= s+11 using only values up to t. v / pv are full-grid series."""
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
        partner = np.full((len(ts), 4), np.nan)
        if pv is not None:
            lp = np.log1p(np.clip(pv, 0, None))
            pwin = np.lib.stride_tricks.sliding_window_view(pv, SEASON)[ts - SEASON + 1]
            for k in range(3):
                partner[:, k] = lp[ts - k] - level
            partner[:, 3] = np.log1p(pwin.mean(axis=1)) - level
        return {"t": ts, "level": level, "lags": lags, "yoy": yoy, "partner": partner}

    def fit_predict(self, series: Sequence[dict], cal_month: np.ndarray, origins: Sequence[int],
                    h: int) -> Dict[tuple, np.ndarray]:
        """
        series: dicts with keys id, v (full-grid values), s (start idx), metric_code, level_code
        and optional pv (partner metric full-grid values).
        cal_month: calendar month (1-12) of every global index (length >= T + h).
        Returns {(id, origin): forecast array of length h} for series with >= 12 months at origin.
        """
        feats = []
        for sr in series:
            f = self.features(sr["v"], sr["s"], sr.get("pv"))
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
                model = self.fit_model(np.vstack(Xtr), np.concatenate(ytr), np.concatenate(wtr))
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
            f["lags"][mask], f["level"][mask], f["yoy"][mask], seasonal, f["partner"][mask],
            cal_month[target_idx], np.full(n, sr["metric_code"]), np.full(n, sr["level_code"]),
        ])


class GlobalLGBM(GlobalDirect):
    name = "GlobalLGBM"
    description = ("Global LightGBM, one model per horizon (direct), log-ratio target, scaled lags, YoY, "
                   "funnel (other metric) and calendar features, recency-weighted")

    def make_model(self):
        import lightgbm as lgb
        p = dict(objective="l1", n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=1,
                 random_state=self.seed)
        p.update(self.params)
        return lgb.LGBMRegressor(**p)


class GlobalXGB(GlobalDirect):
    name = "GlobalXGB"
    description = "Global XGBoost (v1 Model07, rebuilt direct & global), same features as GlobalLGBM"

    def make_model(self):
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:absoluteerror", n_estimators=400, learning_rate=0.03, max_depth=4,
                            min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                            random_state=self.seed, n_jobs=1, verbosity=0)


class GlobalMLP(GlobalDirect):
    name = "GlobalMLP"
    description = ("Global neural network (replaces v1 Model08 LSTM, which fell back to an MLP without "
                   "TensorFlow): 2 hidden layers, same features, direct multi-horizon")

    def fit_model(self, X, y, w):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        Xf = np.nan_to_num(X, nan=0.0)
        scaler = StandardScaler().fit(Xf)
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=400,
                           early_stopping=True, n_iter_no_change=20, random_state=self.seed)
        try:
            mlp.fit(scaler.transform(Xf), y, sample_weight=w)
        except TypeError:                     # older scikit-learn: no sample weights
            mlp.fit(scaler.transform(Xf), y)

        class _Fitted:
            def predict(self, Xn):
                return mlp.predict(scaler.transform(np.nan_to_num(Xn, nan=0.0)))
        return _Fitted()


GLOBAL_MODELS = {c.name: c for c in (GlobalLGBM, GlobalXGB, GlobalMLP)}
GlobalLGBMDirect = GlobalLGBM          # backward-compatible alias


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
