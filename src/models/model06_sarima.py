# -*- coding: utf-8 -*-
"""
Model 06 — SARIMA (OSS 3m + 12m production, EDA-aware)
------------------------------------------------------
• auto_arima (m=12), optional exogenous regressors
• Strict OSS backtest: fit on history EXCLUDING last 3 months, forecast those 3 months
• Production forecast: fit on FULL history, forecast next 12 months
• Month-of-year step clipping from EDA mom_bands (fallback [0.8, 1.2])
• All predictions clipped to ≥ 0
Artefacts → …/Models/Model06_SARIMA/
"""
from __future__ import annotations

import os, warnings, logging, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
import pmdarima as pm

from ..config import OUT_DIR, HORIZON, RANDOM_SEED, ENABLE_EXOG
from ..loader import load_data

# ─────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)
console = Console()

RUN_DIR  = OUT_DIR / "Models" / "Model06_SARIMA"
PLOT_DIR = RUN_DIR / "plots"
RUN_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=RUN_DIR/"model06_oss.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

EDA_DIR = OUT_DIR / "EDA"
OSS_MONTHS = int(os.environ.get("RMG_OSS_MONTHS", "3"))

# ── helpers ──────────────────────────────────────────────────────────
def _safe(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", s)

def _split_oss(y: pd.Series, k: int = OSS_MONTHS) -> tuple[pd.Series, pd.DatetimeIndex]:
    y = y.sort_index()
    if len(y) <= k:
        return y.iloc[:0], y.index
    return y.iloc[:-k], y.index[-k:]

def _load_need_diff() -> Optional[pd.DataFrame]:
    p = EDA_DIR / "need_diff_flags.parquet"
    try:
        return pd.read_parquet(p)
    except Exception:
        return None

def _load_mom_bands() -> Optional[pd.DataFrame]:
    p = EDA_DIR / "mom_bands.parquet"
    try:
        return pd.read_parquet(p)
    except Exception:
        return None

NEED_DIFF = _load_need_diff()
MOM_BANDS = _load_mom_bands()

def _hint_d(cty: str, reg: str, metr: str) -> Optional[int]:
    """Return 1/0 from need_diff flag if available, else None (auto decide)."""
    if NEED_DIFF is None:
        return None
    sub = NEED_DIFF[(NEED_DIFF.Country==cty) & (NEED_DIFF.Region==reg) & (NEED_DIFF.Metric==metr)]
    if sub.empty:
        return None
    try:
        return 1 if int(sub.need_diff.iloc[0]) == 1 else 0
    except Exception:
        return None

def _month_bands(cty: str, reg: str, metr: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build an index-aligned frame with lo_step, hi_step from EDA; fallback [0.8, 1.2].
    """
    lo = np.full(len(idx), 0.7, dtype=float)
    hi = np.full(len(idx), 1.5, dtype=float)
    if MOM_BANDS is not None:
        sub = MOM_BANDS[(MOM_BANDS.Country==cty) & (MOM_BANDS.Region==reg) & (MOM_BANDS.Metric==metr)]
        if not sub.empty:
            mapping = {int(r.Month): (max(0.2, 1.0+float(r.p05)),
                                      min(2.5, 1.0+float(r.p95)))
                       for _, r in sub.iterrows()}
            for i, m in enumerate(idx.month):
                if m in mapping:
                    lo[i], hi[i] = mapping[m]
                    if not np.isfinite(lo[i]) or not np.isfinite(hi[i]) or lo[i] >= hi[i]:
                        lo[i], hi[i] = 0.7, 1.5
    return pd.DataFrame({"lo_step": lo, "hi_step": hi}, index=idx)

def _build_exog(df_all: pd.DataFrame, ex_cols: List[str],
                cty: str, reg: str, align_idx: pd.DatetimeIndex) -> Tuple[pd.DataFrame, List[str]]:
    """
    Align exogenous features to align_idx and drop unusable columns.
    """
    if not ex_cols:
        return pd.DataFrame(index=align_idx), []
    ex = (
        df_all.loc[(df_all.Country==cty) & (df_all.Region==reg), ["Date"] + ex_cols]
             .drop_duplicates("Date")
             .set_index("Date")
             .reindex(align_idx)
             .interpolate("linear", limit_direction="both")
             .ffill()
             .bfill()
             .astype(float)
    )
    # drop columns with any non-finite values
    bad = [c for c in ex.columns if not np.isfinite(ex[c].to_numpy(dtype=float)).all()]
    if bad:
        logging.warning("Dropping non-finite exog columns: %s", bad)
        ex = ex.drop(columns=bad, errors="ignore")
    return ex, list(ex.columns)

def _future_exog_from_last(base_exog: pd.DataFrame, future_idx: pd.DatetimeIndex, cols: List[str]) -> pd.DataFrame:
    """
    Build future exog frame by forward-filling the last known row for given cols.
    """
    fut = pd.DataFrame(index=future_idx, columns=cols, dtype=float)
    if base_exog is None or base_exog.empty or not cols:
        return fut.fillna(0.0)
    last = base_exog.iloc[-1][cols]
    for c in cols:
        v = float(last.get(c, 0.0))
        if not np.isfinite(v):
            v = 0.0
        fut[c] = v
    return fut

def _clip_nonneg(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0, None)

# ── main pipeline ────────────────────────────────────────────────────
def main() -> None:
    df, ex_cols_all = load_data()

    # metric columns present
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
                                    desc="SARIMA (OSS+Prod)"):
        # complete monthly series
        y = (g.set_index("Date")["Value"]
               .astype(float)
               .asfreq("MS")
               .interpolate("linear", limit_direction="both"))

        if len(y) < 24:
            logging.warning("Skipping %s/%s/%s (len=%d)", cty, reg, metr, len(y))
            continue

        # ============== STRICT OSS BACKTEST (last 3 months) ==============
        y_train, oss_idx = _split_oss(y, OSS_MONTHS)
        if not y_train.empty and len(oss_idx) > 0:
            use_exog = bool(ENABLE_EXOG and ex_cols_all)
            X_train: Optional[pd.DataFrame] = None
            X_oss:   Optional[pd.DataFrame] = None
            use_cols: List[str] = []

            if use_exog:
                X_train, use_cols = _build_exog(df, ex_cols_all, cty, reg, y_train.index)
                if use_cols:
                    # For OSS months (historical dates) use the actual exog values aligned to those dates
                    X_oss, _ = _build_exog(df, use_cols, cty, reg, oss_idx)
                else:
                    use_exog = False

            d_hint = _hint_d(cty, reg, metr)

            try:
                model_oss = pm.auto_arima(
                    y_train, X=X_train if use_exog else None,
                    seasonal=True, m=12,
                    start_p=1, start_q=1, max_p=2, max_q=2,
                    d=d_hint, D=1,
                    start_P=1, start_Q=0, max_P=1, max_Q=1,
                    information_criterion="aicc",
                    trace=False, error_action="ignore",
                    suppress_warnings=True, stepwise=True,
                    max_d=2, max_D=1,
                    seasonal_test="ocsb",
                    random_state=RANDOM_SEED
                )
            except Exception as e:
                logging.exception("auto_arima(OSS) failed for %s/%s/%s: %s", cty, reg, metr, e)
                continue

            # Forecast exactly the OSS window (3m)
            try:
                yhat_oss = np.asarray(model_oss.predict(n_periods=len(oss_idx),
                                                        X=X_oss if use_exog else None))
            except Exception as e:
                logging.exception("predict(OSS) failed for %s/%s/%s: %s", cty, reg, metr, e)
                continue

            # Month-of-year step clipping (train→OSS)
            bands_oss = _month_bands(cty, reg, metr, oss_idx)
            hist_level = float(y_train.iloc[-1]) if np.isfinite(y_train.iloc[-1]) else 0.0
            level = max(1e-9, hist_level)
            clamped = []
            for i, dt in enumerate(oss_idx):
                v = max(0.0, float(yhat_oss[i]))
                step = v / max(1e-9, level)
                step = float(np.clip(step, bands_oss.loc[dt, "lo_step"], bands_oss.loc[dt, "hi_step"]))
                level = max(0.0, level * step)
                clamped.append(level)

            bt = pd.DataFrame({
                "Country": cty, "Region": reg, "Metric": metr,
                "Date": oss_idx,
                "Actual": y.loc[oss_idx].values,
                "Pred":   np.array(clamped, dtype=float)
            })
            records_back.append(bt)

        # ==================== 12M PRODUCTION FORECAST ====================
        use_exog_full = bool(ENABLE_EXOG and ex_cols_all)
        X_full: Optional[pd.DataFrame] = None
        use_cols_full: List[str] = []

        if use_exog_full:
            X_full, use_cols_full = _build_exog(df, ex_cols_all, cty, reg, y.index)
            if not use_cols_full:
                use_exog_full = False

        d_hint_full = _hint_d(cty, reg, metr)

        try:
            model_full = pm.auto_arima(
                y, X=X_full if use_exog_full else None,
                seasonal=True, m=12,
                start_p=1, start_q=1, max_p=2, max_q=2,
                d=d_hint_full, D=1,
                start_P=1, start_Q=0, max_P=1, max_Q=1,
                information_criterion="aicc",
                trace=False, error_action="ignore",
                suppress_warnings=True, stepwise=True,
                max_d=2, max_D=1,
                seasonal_test="ocsb",
                random_state=RANDOM_SEED
            )
        except Exception as e:
            logging.exception("auto_arima(full) failed for %s/%s/%s: %s", cty, reg, metr, e)
            continue

        fut_idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=HORIZON, freq="MS")
        if use_exog_full:
            X_future = _future_exog_from_last(X_full, fut_idx, use_cols_full)
            try:
                fc_vals = model_full.predict(n_periods=HORIZON, X=X_future)
            except Exception:
                fc_vals = model_full.predict(n_periods=HORIZON)
        else:
            fc_vals = model_full.predict(n_periods=HORIZON)

        fc_vals = _clip_nonneg(np.asarray(fc_vals, dtype=float))
        bands = _month_bands(cty, reg, metr, fut_idx)

        # step clamp on horizon
        hist_last = float(y.iloc[-1]) if np.isfinite(y.iloc[-1]) else 0.0
        level = max(1e-9, hist_last)
        clipped = []
        for i, dt in enumerate(fut_idx):
            v = float(fc_vals[i])
            step = v / max(1e-9, level)
            step = float(np.clip(step, bands.loc[dt, "lo_step"], bands.loc[dt, "hi_step"]))
            level = max(0.0, level * step)
            clipped.append(level)

        fc = pd.DataFrame({
            "Country": cty, "Region": reg, "Metric": metr,
            "Date": fut_idx, "Forecast": clipped
        })
        records_fore.append(fc)

        # ============================== Plot =============================
        fig, ax = plt.subplots(figsize=(6,3))
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

    # ── SAVE ARTEFACTS ───────────────────────────────────────────────────
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    fore_df = pd.concat(records_fore, ignore_index=True) if records_fore else pd.DataFrame()
    back_df = pd.concat(records_back, ignore_index=True) if records_back else pd.DataFrame()

    if not fore_df.empty:
        fore_df.to_csv(RUN_DIR / "forecast_h12.csv", index=False)
    back_df.to_csv(RUN_DIR / "backtest.csv", index=False)

    # OSS-only performance
    if not back_df.empty:
        tmp = back_df.copy()
        tmp["abs_err"] = (tmp["Actual"] - tmp["Pred"]).abs()
        denom = tmp["Actual"].replace(0, np.nan).abs()
        tmp["ape"] = (tmp["abs_err"] / denom).replace([np.inf, -np.inf], np.nan) * 100.0
        perf = (
            tmp.groupby(["Country","Region","Metric"], as_index=False)
               .agg(MAE=("abs_err","mean"), MAPE=("ape","mean"))
        )
        perf.to_excel(RUN_DIR / "series_scores.xlsx", index=False)
    else:
        perf = pd.DataFrame()

    # TL;DR
    tbl = Table(title=f"Model 06 SARIMA — OSS last {OSS_MONTHS} months (median errors)", pad_edge=False)
    tbl.add_column("Metric"); tbl.add_column("Median MAE", justify="right")
    tbl.add_column("Median MAPE %", justify="right")
    for metr in ["Received","Submitted"]:
        sub = perf[perf["Metric"]==metr]
        if not sub.empty:
            tbl.add_row(metr, f"{sub['MAE'].median():,.0f}", f"{sub['MAPE'].median():.1f}")
    console.print(tbl)
    (RUN_DIR/"tldr.txt").write_text(tbl.__str__())
    console.print(f"[bold green]✅ All artefacts saved → {RUN_DIR}[/]")

if __name__=="__main__":
    main()
