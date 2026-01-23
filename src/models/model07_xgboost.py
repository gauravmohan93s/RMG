# -*- coding: utf-8 -*-
"""
Model 07 — XGBoost (OSS 3m + 12m production, FAST by default)
-------------------------------------------------------------
• Two optional upgrades for short monthly counts:
  - Poisson objective (no log1p) → set XGB_USE_POISSON=True
  - Residual learning vs. seasonal prior → set XGB_USE_RESIDUAL_PRIOR=True
• Features: 12 lags, month cyclical, + YoY, rolling(3/6/12), 6m log-slope, optional exog (float32)
• Strict OSS: last 3 months (trained on y[:-3]); recursive 3-step (actual exog)
• Forecast: FULL history; recursive 12-step (ffill exog)
• FAST mode (RMG_FAST_XGB=1 default): 1 anchored split + optional early stopping
• FULL mode (RMG_FAST_XGB=0): small anchored CV grid
• Safety: clip to ≥ 0
Artefacts → …/Models/Model07_XGBoost/
"""
from __future__ import annotations

# std-lib
import os, warnings, logging, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression

# project
from ..config import OUT_DIR, HORIZON, RANDOM_SEED, ENABLE_EXOG
from ..loader import load_data

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)
console = Console()

# -------------------------- knobs --------------------------
FAST_MODE                 = os.getenv("RMG_FAST_XGB", "1") != "0"   # default fast
N_JOBS                    = int(os.getenv("RMG_XGB_JOBS", "-1"))    # -1 = all cores
OSS_MONTHS               = 3
N_LAGS                   = 12

# NEW toggles (recommended for short, count-like series)
XGB_USE_POISSON          = True   # use Poisson objective; turn off to revert to log1p + MSE
XGB_USE_RESIDUAL_PRIOR   = True   # learn residuals vs. a seasonal prior baseline

def _safe(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", s)

# ---------- simple seasonal prior (AvgMoM month map), fast & local ----------
def _avg_mom_map_from_series(y: pd.Series) -> Dict[int, float]:
    """
    Return month -> avg MoM% prior (clipped ±30%) computed from series y.
    Uses last 2y at 70% weight vs older at 30% when available.
    """
    pct = y.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    out = {}
    if pct.empty:
        return {m: 0.0 for m in range(1, 13)}
    end = pct.index.max()
    cutoff = end - pd.DateOffset(years=2)
    recent = pct[pct.index > cutoff]
    older  = pct[pct.index <= cutoff]
    for m in range(1, 13):
        r = recent[recent.index.month == m]
        o = older [older .index.month == m]
        if len(r) and len(o): val = 0.7*r.mean() + 0.3*o.mean()
        elif len(r):          val = r.mean()
        elif len(o):          val = o.mean()
        else:                 val = 0.0
        out[m] = float(np.clip(val, -0.3, 0.3))
    return out

def _apply_month_prior_step(prev: float, month: int, prior_map: Dict[int,float]) -> float:
    """One-step seasonal prior (multiplicative) from previous level."""
    pct = float(prior_map.get(int(month), 0.0))
    step = 1.0 + pct
    # light clamps for stability
    step = float(np.clip(step, 0.8, 1.2))
    return max(0.0, float(prev) * step)

# -------------------------- features --------------------------
def _month_feats(idx: pd.DatetimeIndex) -> pd.DataFrame:
    m = idx.month.values.astype(np.int16)
    return pd.DataFrame(
        {"month": m,
         "m_sin": np.sin(2*np.pi*m/12.0),
         "m_cos": np.cos(2*np.pi*m/12.0)},
        index=idx,
    ).astype({"month":"int16","m_sin":"float32","m_cos":"float32"})

def _extra_level_feats(y: pd.Series) -> pd.DataFrame:
    """
    Add YoY (level ratio vs t-12), rolling means (3/6/12), and 6m log-slope.
    These are cheap and robust for monthly data.
    """
    df = pd.DataFrame(index=y.index)
    # YoY level ratio - 1 (clip to ±100%)
    df["yoy"] = (y / y.shift(12) - 1.0).replace([np.inf,-np.inf], np.nan).clip(-1.0, 1.0)
    # rolling means (levels), normalized by last value to control scale
    df["roll3"]  = y.rolling(3).mean()
    df["roll6"]  = y.rolling(6).mean()
    df["roll12"] = y.rolling(12).mean()
    # normalize rolls by current level to produce dimensionless ratios
    with np.errstate(divide="ignore", invalid="ignore"):
        df["r3_ratio"]  = (df["roll3"]  / y).replace([np.inf,-np.inf], np.nan).clip(0, 5)
        df["r6_ratio"]  = (df["roll6"]  / y).replace([np.inf,-np.inf], np.nan).clip(0, 5)
        df["r12_ratio"] = (df["roll12"] / y).replace([np.inf,-np.inf], np.nan).clip(0, 5)
    df = df.drop(columns=["roll3","roll6","roll12"])
    # 6-month log-slope via simple linear regression on indices
    logy = np.log1p(y.replace(-np.inf, np.nan))
    slope = pd.Series(index=y.index, dtype="float32")
    for i in range(len(y)):
        j0 = max(0, i-5)
        seg = logy.iloc[j0:i+1].dropna()
        if len(seg) >= 3:
            X = np.arange(len(seg)).reshape(-1,1)
            lr = LinearRegression()
            lr.fit(X, seg.values)
            slope.iloc[i] = float(lr.coef_[0])
        else:
            slope.iloc[i] = np.nan
    df["log_slope6"] = slope.astype("float32")
    return df.astype("float32")

def _build_exog_slice(df_all: pd.DataFrame,
                      ex_cols: List[str],
                      cty: str, reg: str,
                      align_idx: pd.DatetimeIndex) -> Tuple[pd.DataFrame, List[str]]:
    """Align exog to index, drop unusable/near-constant columns."""
    if not ex_cols:
        return pd.DataFrame(index=align_idx), []
    ex = (
        df_all.loc[(df_all.Country==cty) & (df_all.Region==reg), ["Date"] + ex_cols]
             .drop_duplicates("Date")
             .set_index("Date")
             .reindex(align_idx)
             .interpolate("linear", limit_direction="both")
             .ffill().bfill()
             .astype("float32")
    )
    # drop non-finite & near-constant
    bad = []
    for c in list(ex.columns):
        arr = ex[c].to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            bad.append(c); continue
        if np.nanstd(arr) < 1e-8:
            bad.append(c)
    if bad:
        logging.warning("Dropping weak exog for %s/%s: %s", cty, reg, bad)
        ex = ex.drop(columns=bad, errors="ignore")
    return ex, list(ex.columns)

def _future_exog_from_last(base_exog: pd.DataFrame,
                           future_idx: pd.DatetimeIndex,
                           cols: List[str]) -> pd.DataFrame:
    if base_exog is None or base_exog.empty or not cols:
        return pd.DataFrame(0.0, index=future_idx, columns=cols, dtype="float32")
    last = base_exog.iloc[-1][cols]
    fut  = pd.DataFrame(index=future_idx, columns=cols, dtype="float32")
    for c in cols:
        v = float(last.get(c, 0.0))
        fut[c] = v if np.isfinite(v) else 0.0
    return fut

# -------------------------- supervised frames --------------------------
def make_supervised(
    y: pd.Series,
    exog: Optional[pd.DataFrame] = None,
    n_lags: int = N_LAGS,
    use_poisson: bool = XGB_USE_POISSON,
    use_residual: bool = XGB_USE_RESIDUAL_PRIOR,
) -> Tuple[pd.DataFrame, pd.Series, Dict[int,float]]:
    """
    Returns:
      X (float32), y_target (float32), prior_map (for residual mode)
    If use_residual=True, target = (y - prior_level) on level scale (clipped at 0 baseline).
    If use_poisson=True and not residual → target is y (nonnegative).
    """
    y = y.astype("float32")
    # prior map (from history) if residual mode
    prior_map = _avg_mom_map_from_series(y) if use_residual else {}

    # build base DF with lags
    df = pd.DataFrame({"y": y})
    for lag in range(1, n_lags+1):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # extra features
    df = df.join(_month_feats(df.index))
    df = df.join(_extra_level_feats(y))

    # add exog
    if exog is not None and not exog.empty:
        exog = exog.reindex(df.index).ffill().bfill()
        df = df.join(exog)

    # residual target or raw/log target
    if use_residual:
        # compute one-step prior level from t-1 → t using month prior
        prev = df["lag_1"]
        months = df.index.month
        prior_next = prev.copy()
        for i in range(len(df)):
            v_prev = float(prev.iloc[i]) if np.isfinite(prev.iloc[i]) else np.nan
            m = int(months[i])
            if np.isfinite(v_prev):
                prior_next.iloc[i] = _apply_month_prior_step(v_prev, m, prior_map)
            else:
                prior_next.iloc[i] = np.nan
        # residual (level scale). Keep residuals moderate: clip to ±200% of prior.
        df["target"] = (df["y"] - prior_next).clip(
            lower=-2.0 * prior_next.fillna(0.0), upper=2.0 * prior_next.fillna(0.0)
        )
    else:
        if use_poisson:
            df["target"] = df["y"].clip(lower=0.0)
        else:
            df["target"] = np.log1p(df["y"]).astype("float32")

    df = df.dropna()

    # final X/y
    X = df.drop(columns=["y","target"]).astype("float32")
    y_target = df["target"].astype("float32")
    return X, y_target, prior_map

# -------------------------- CV / ES helpers --------------------------
def _anchored_splits(n: int, fast: bool) -> List[Tuple[np.ndarray,np.ndarray]]:
    idx = np.arange(n)
    splits: List[Tuple[np.ndarray,np.ndarray]] = []
    if n < (N_LAGS + 20):
        return splits
    if fast:
        split = max(N_LAGS+8, int(n*0.80))
        splits.append((idx[:split], idx[split:]))
    else:
        for c in (0.65, 0.85):
            split = max(N_LAGS+8, int(n*c))
            if split < n-1:
                tr, va = idx[:split], idx[split:]
                if len(va) > 0:
                    splits.append((tr, va))
    return splits

def _mae_original(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def _new_model(**params) -> XGBRegressor:
    # stronger regularization for tiny monthly sets
    base = dict(
        objective="count:poisson" if XGB_USE_POISSON and not XGB_USE_RESIDUAL_PRIOR else "reg:squarederror",
        random_state=RANDOM_SEED,
        tree_method="hist",
        n_jobs=N_JOBS,
        max_depth=3,
        n_estimators=400 if FAST_MODE else 700,
        learning_rate=0.05 if FAST_MODE else 0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        reg_alpha=0.0 if FAST_MODE else 0.5,
        min_child_weight=6,
        max_bin=256,
    )
    base.update(params)
    return XGBRegressor(**base)

def _fit_with_optional_es(model, X_tr, y_tr, X_va=None, y_va=None, verbose=False):
    try:
        if X_va is not None and y_va is not None and len(X_va) > 0:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=verbose, early_stopping_rounds=60)
        else:
            model.fit(X_tr, y_tr, verbose=verbose)
    except TypeError:
        if X_va is not None and y_va is not None and len(X_va) > 0:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=verbose)
        else:
            model.fit(X_tr, y_tr, verbose=verbose)

def fit_params_with_cv(X: pd.DataFrame, y: pd.Series) -> Dict:
    splits = _anchored_splits(len(X), fast=FAST_MODE)
    if not splits:
        return {}
    if FAST_MODE:
        tr, va = splits[0]
        model = _new_model(n_estimators=900)
        _fit_with_optional_es(model, X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va], verbose=False)
        # recover effective best_n_estimators when available
        best_ne = None
        for attr in ("best_ntree_limit", "best_iteration", "best_iteration_"):
            v = getattr(model, attr, None)
            if isinstance(v, (int, np.integer)) and v > 0:
                best_ne = int(v)
                break
        return {"n_estimators": int(np.clip(best_ne or 400, 250, 900))}
    else:
        grid = {
            "max_depth": [2, 3, 4],
            "learning_rate": [0.03, 0.05],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.7, 0.9],
            "reg_lambda": [2.0, 5.0],
            "reg_alpha": [0.0, 0.5],
            "min_child_weight": [6, 10],
            "n_estimators": [500, 900],
        }
        best_params, best_score = None, np.inf
        for params in ParameterGrid(grid):
            scores = []
            for tr, va in splits:
                model = _new_model(**params)
                _fit_with_optional_es(model, X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va], verbose=False)
                pred = model.predict(X.iloc[va])
                scores.append(_mae_original(y.iloc[va].to_numpy(), pred))
            sc = float(np.mean(scores)) if scores else np.inf
            if sc < best_score:
                best_score, best_params = sc, params
        return best_params or {}

# -------------------------- data split --------------------------
def _split_oss(y: pd.Series, k: int = OSS_MONTHS) -> tuple[pd.Series, pd.DatetimeIndex]:
    y = y.sort_index()
    if len(y) <= k:
        return y.iloc[:0], y.index
    return y.iloc[:-k], y.index[-k:]

# -----------------------------------------------------------------------
RUN_DIR  = OUT_DIR / "Models" / "Model07_XGBoost"
PLOT_DIR = RUN_DIR / "plots"
RUN_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=RUN_DIR/"model07.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def _apply_model_prediction(model: XGBRegressor,
                            Xn: pd.DataFrame,
                            prev_level: float,
                            month: int,
                            prior_map: Dict[int,float],
                            use_residual: bool) -> float:
    """Turn model output into a level forecast (≥0) with residual vs. prior if enabled."""
    raw = float(model.predict(Xn)[0])
    if use_residual:
        baseline = _apply_month_prior_step(prev_level, month, prior_map)
        yhat = baseline + raw
    else:
        if XGB_USE_POISSON:
            # Poisson objective returns log-rate internally; sklearn wrapper returns mean
            yhat = max(0.0, raw)
        else:
            yhat = max(0.0, float(np.expm1(raw)))
    return max(0.0, yhat)

def main() -> None:
    # 1) LOAD
    df, ex_cols_all = load_data()
    value_vars = [c for c in ["Received","Submitted"] if c in df.columns]
    if not value_vars:
        raise RuntimeError("No metric columns found (need 'Received' and/or 'Submitted').")

    melt = (
        df.melt(id_vars=["Country","Region","Date"],
                value_vars=value_vars,
                var_name="Metric", value_name="Value")
          .drop_duplicates(subset=["Country","Region","Metric","Date"], keep="last")
          .sort_values(["Country","Region","Metric","Date"])
    )

    records_fore: List[pd.DataFrame] = []
    records_back: List[pd.DataFrame] = []

    for (cty, reg, metr), g in tqdm(melt.groupby(["Country","Region","Metric"], sort=False),
                                    desc="Model07_XGB (OSS+Prod)"):
        # complete monthly y
        y = (g.set_index("Date")["Value"]
               .asfreq("MS")
               .interpolate("linear", limit_direction="both")
               .astype("float32"))
        if len(y) < 24:
            logging.warning("Skip %s/%s/%s (len=%d)", cty, reg, metr, len(y))
            continue

        # exog
        use_exog = bool(ENABLE_EXOG and ex_cols_all)
        ex_full: Optional[pd.DataFrame] = None
        ex_cols_use: List[str] = []
        if use_exog:
            ex_full, ex_cols_use = _build_exog_slice(df, ex_cols_all, cty, reg, y.index)
            use_exog = bool(ex_cols_use)

        # ==================== STRICT OSS BACKTEST (last 3) =================
        y_train, oss_idx = _split_oss(y, OSS_MONTHS)
        best_params: Dict = {}
        prior_map_train: Dict[int,float] = _avg_mom_map_from_series(y_train) if XGB_USE_RESIDUAL_PRIOR else {}

        if not y_train.empty and len(oss_idx) > 0:
            ex_train = ex_full.loc[y_train.index, ex_cols_use] if use_exog else None
            ex_oss   = ex_full.loc[oss_idx,     ex_cols_use]   if use_exog else None

            X_tr_all, y_tr, _ = make_supervised(
                y_train, exog=ex_train, n_lags=N_LAGS,
                use_poisson=XGB_USE_POISSON, use_residual=XGB_USE_RESIDUAL_PRIOR
            )
            if len(X_tr_all) < 24:  # too short after lagging
                logging.warning("Not enough rows after lagging for %s/%s/%s", cty, reg, metr)
            else:
                best_params = fit_params_with_cv(X_tr_all, y_tr)
                model_oss = _new_model(**best_params)
                splits = _anchored_splits(len(X_tr_all), fast=True)
                if splits:
                    tr, va = splits[0]
                    _fit_with_optional_es(model_oss,
                                          X_tr_all.iloc[tr], y_tr.iloc[tr],
                                          X_tr_all.iloc[va], y_tr.iloc[va],
                                          verbose=False)
                else:
                    model_oss.fit(X_tr_all, y_tr, verbose=False)

                # recursive predict across oss_idx
                hist = y_train.copy()
                preds = []
                for dt in oss_idx:
                    feats = {f"lag_{lag}": float(hist.iloc[-lag]) for lag in range(1, N_LAGS+1)
                             if len(hist) >= lag}
                    if len(feats) < N_LAGS:
                        lastv = float(hist.iloc[-1])
                        for lag in range(len(feats)+1, N_LAGS+1):
                            feats[f"lag_{lag}"] = lastv
                    # month & extras for dt
                    mf = _month_feats(pd.DatetimeIndex([dt]))
                    ef = _extra_level_feats(hist).reindex([hist.index[-1]]).ffill()
                    # build one-row X
                    Xn = pd.DataFrame(feats, index=[dt]).astype("float32")
                    Xn = Xn.join(mf).astype("float32")
                    Xn = Xn.join(ef).astype("float32")
                    if use_exog:
                        for c in ex_cols_use:
                            Xn[c] = float(ex_oss.loc[dt, c])

                    yhat = _apply_model_prediction(model_oss, Xn, float(hist.iloc[-1]), dt.month, prior_map_train, XGB_USE_RESIDUAL_PRIOR)
                    preds.append(yhat)
                    hist.loc[dt] = yhat

                bt = pd.DataFrame({
                    "Country": cty, "Region": reg, "Metric": metr,
                    "Date": oss_idx, "Actual": y.loc[oss_idx].values, "Pred": preds
                })
                records_back.append(bt)

        # ==================== 12M PRODUCTION FORECAST =====================
        prior_map_full: Dict[int,float] = _avg_mom_map_from_series(y) if XGB_USE_RESIDUAL_PRIOR else {}
        X_all, y_all, _ = make_supervised(
            y, exog=(ex_full if use_exog else None), n_lags=N_LAGS,
            use_poisson=XGB_USE_POISSON, use_residual=XGB_USE_RESIDUAL_PRIOR
        )
        if not best_params:
            best_params = fit_params_with_cv(X_all, y_all)

        model_full = _new_model(**best_params)
        # small validation tail for optional ES
        if len(X_all) > 30:
            _fit_with_optional_es(model_full,
                                  X_all.iloc[:-12], y_all.iloc[:-12],
                                  X_all.iloc[-12:],  y_all.iloc[-12:],
                                  verbose=False)
        else:
            model_full.fit(X_all, y_all, verbose=False)

        fut_idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=HORIZON, freq="MS")
        ex_future = _future_exog_from_last(ex_full, fut_idx, ex_cols_use) if use_exog else None

        hist = y.copy()
        fut_preds = []
        for next_date in fut_idx:
            feats = {f"lag_{lag}": float(hist.iloc[-lag]) for lag in range(1, N_LAGS+1)}
            mf = _month_feats(pd.DatetimeIndex([next_date]))
            ef = _extra_level_feats(hist).reindex([hist.index[-1]]).ffill()
            Xn = pd.DataFrame(feats, index=[next_date]).astype("float32")
            Xn = Xn.join(mf).astype("float32")
            Xn = Xn.join(ef).astype("float32")
            if ex_future is not None and not ex_future.empty:
                for c in ex_cols_use:
                    Xn[c] = float(ex_future.loc[next_date, c])

            yhat = _apply_model_prediction(model_full, Xn, float(hist.iloc[-1]), next_date.month, prior_map_full, XGB_USE_RESIDUAL_PRIOR)
            fut_preds.append(yhat)
            hist.loc[next_date] = yhat

        fc = pd.DataFrame({
            "Country": cty, "Region": reg, "Metric": metr,
            "Date": fut_idx, "Forecast": fut_preds
        })
        records_fore.append(fc)

        # ----------------------------- Plot ------------------------------
        fig, ax = plt.subplots(figsize=(6.4,3.2))
        ax.plot(y.index, y.values, label="Actual")
        if 'bt' in locals() and not bt.empty:
            ax.plot(bt["Date"], bt["Pred"], "--", label=f"OSS backtest ({OSS_MONTHS}m)")
        ax.plot(fc["Date"], fc["Forecast"], ":", label=f"Forecast {HORIZON}m")
        ax.set_title(f"{cty} – {reg} – {metr}")
        ax.set_ylabel(metr); ax.grid(alpha=.3); ax.legend(loc="upper left")
        for lbl in ax.get_xticklabels(): lbl.set_rotation(45)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{_safe(cty)}_{_safe(reg)}_{_safe(metr)}.png", dpi=120)
        plt.close(fig)

    # 2) SAVE ARTEFACTS
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    fore_df = pd.concat(records_fore, ignore_index=True) if records_fore else pd.DataFrame()
    back_df = pd.concat(records_back, ignore_index=True) if records_back else pd.DataFrame()
    if not fore_df.empty: fore_df.to_csv(RUN_DIR/"forecast_h12.csv", index=False)
    if not back_df.empty: back_df.to_csv(RUN_DIR/"backtest.csv", index=False)

    # per-series OSS scores
    if not back_df.empty:
        tmp = back_df.copy()
        tmp["abs_err"] = (tmp["Actual"] - tmp["Pred"]).abs()
        denom = tmp["Actual"].replace(0, np.nan).abs()
        tmp["ape"] = (tmp["abs_err"] / denom).replace([np.inf, -np.inf], np.nan) * 100.0
        perf = (
            tmp.groupby(["Country","Region","Metric"], as_index=False)
               .agg(MAE=("abs_err","mean"), MAPE=("ape","mean"))
        )
        perf.to_excel(RUN_DIR/"series_scores.xlsx", index=False)
    else:
        perf = pd.DataFrame()

    # TL;DR
    tbl = Table(title=f"Model 07 XGBoost — OSS last {OSS_MONTHS} months (median errors)", pad_edge=False)
    tbl.add_column("Metric"); tbl.add_column("Median MAE", justify="right")
    tbl.add_column("Median MAPE %", justify="right")
    for metr in ["Received","Submitted"]:
        sub = perf[perf["Metric"]==metr]
        if not sub.empty:
            tbl.add_row(metr, f"{sub['MAE'].median():,.0f}", f"{sub['MAPE'].median():.1f}")
    console.print(tbl)
    (RUN_DIR/"tldr.txt").write_text(tbl.__str__())
    console.print(f"[bold green]✅ All artefacts saved →  {RUN_DIR}[/]")

if __name__=="__main__":
    main()
