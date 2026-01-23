# -*- coding: utf-8 -*-
# src/models/model05_prophet_fast.py
from __future__ import annotations
import os, logging, warnings, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from prophet import Prophet

from ..config import OUT_DIR, HORIZON, RANDOM_SEED, ENABLE_EXOG
from ..loader import load_data

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)
console = Console()

def _safe(s: str) -> str:
    return re.sub(r'[\\/:\*?"<>|]', "_", s)

RUN_DIR  = OUT_DIR / "Models" / "Model05_Prophet_Fast"
PLOT_DIR = RUN_DIR / "plots"
RUN_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=RUN_DIR/"model05_fast.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ------------------------ settings & helpers ------------------------
OSS_MONTHS = int(os.environ.get("RMG_OSS_MONTHS", "3"))

def _series_bounds_24m(pct: pd.Series) -> tuple[float, float]:
    """
    5–95th pct of LAST 24 months of MoM% → multiplicative step bounds.
    Hard clamp to [0.75, 1.25]; fallback [0.85, 1.20].
    """
    clean = pct.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        return 0.85, 1.20
    end   = clean.index.max()
    start = end - pd.DateOffset(months=24)
    recent = clean[(clean.index >= start) & (clean.index <= end)]
    if len(recent) < 3:
        recent = clean
    lo, hi = recent.quantile([0.05, 0.95])
    lower = max(0.75, 1.0 + float(lo))
    upper = min(1.25, 1.0 + float(hi))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        return 0.85, 1.20
    return float(lower), float(upper)

def _split_oss(y: pd.Series, k: int = OSS_MONTHS) -> tuple[pd.Series, pd.DatetimeIndex]:
    """Return (train_upto, oss_idx) where the last k months are OSS targets."""
    y = y.sort_index()
    if len(y) <= k:
        return y.iloc[:0], y.index
    return y.iloc[:-k], y.index[-k:]

def _prepare_y(g: pd.DataFrame) -> pd.Series:
    """
    Monthly series with minimal smoothing:
      - MS freq
      - linear interpolation only for single-month gaps (limit=1)
      - ffill/bfill for remaining edge NAs
      - non-negative
    """
    y = (
        g.set_index("Date")["Value"]
         .astype(float)
         .asfreq("MS")
         .interpolate("linear", limit=1)
         .ffill().bfill()
         .clip(lower=0.0)
    )
    return y

def _build_exog_slice(df_all: pd.DataFrame,
                      ex_cols: List[str],
                      cty: str, reg: str,
                      index: pd.DatetimeIndex) -> Tuple[pd.DataFrame, List[str]]:
    """Align exogenous features to index and drop non-finite columns."""
    if not ex_cols:
        return pd.DataFrame(index=index), []
    ex = (
        df_all.loc[(df_all["Country"]==cty) & (df_all["Region"]==reg), ["Date"]+ex_cols]
              .drop_duplicates("Date")
              .set_index("Date")
              .reindex(index)
              .interpolate("linear", limit_direction="both")
              .ffill().bfill()
              .fillna(0.0)
              .astype(float)
    )
    bad = [c for c in ex.columns if not np.isfinite(ex[c].to_numpy(dtype=float)).all()]
    if bad:
        logging.warning("Dropping non-finite exog: %s", bad)
        ex = ex.drop(columns=bad, errors="ignore")
    return ex, list(ex.columns)

def _new_prophet() -> Prophet:
    """
    Prophet configured to allow curvature and multiplicative seasonality.
    Tuned for monthly data (no weekly/daily).
    """
    m = Prophet(
        changepoint_prior_scale=0.2,
        n_changepoints=25,
        seasonality_mode='multiplicative',
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    # richer annual & semiannual components for monthly cadence
    m.add_seasonality(name='yearly', period=365.25, fourier_order=10)
    m.add_seasonality(name='semiannual', period=182.6, fourier_order=5)
    return m

def _clip_step(prev_level: float, raw_next: float, lo: float, hi: float) -> float:
    base = max(1e-9, float(prev_level))
    step = float(raw_next) / base
    step = float(np.clip(step, lo, hi))
    return max(0.0, base * step)

# --------------------- main pipeline ---------------------
def main() -> None:
    df, ex_cols_all = load_data()

    # Defensively pick metrics present
    value_vars = [c for c in ["Received","Submitted"] if c in df.columns]
    if not value_vars:
        raise RuntimeError("No metric columns found (need at least one of 'Received','Submitted').")

    melt = (
        df.melt(
            id_vars=["Country","Region","Date"],
            value_vars=value_vars,
            var_name="Metric", value_name="Value"
        )
        .drop_duplicates(subset=["Country","Region","Metric","Date"], keep="last")
        .sort_values(["Country","Region","Metric","Date"])
    )

    records_fore: List[pd.DataFrame] = []
    records_back: List[pd.DataFrame] = []

    for (cty, reg, metr), g in tqdm(melt.groupby(["Country","Region","Metric"], sort=False),
                                    desc="Model05_Prophet"):
        y = _prepare_y(g)
        if len(y) < 24:
            logging.warning("Skip %s/%s/%s (len=%d)", cty, reg, metr, len(y))
            continue

        # bounds from last 24m MoM (to avoid flatlines/explosions)
        lo_step, hi_step = _series_bounds_24m(y.pct_change())

        # ----------------- STRICT OSS BACKTEST (last K months) -----------------
        y_train, oss_idx = _split_oss(y, OSS_MONTHS)
        bt_rows = []
        for t in oss_idx:
            # train strictly up to t-1 (no leakage)
            hist = y.loc[:t - pd.offsets.MonthBegin()]
            if hist.empty:
                continue

            use_exog = bool(ENABLE_EXOG and ex_cols_all)
            if use_exog:
                X_train, use_cols = _build_exog_slice(df, ex_cols_all, cty, reg, hist.index)
                use_exog = bool(use_cols)
            else:
                X_train, use_cols = pd.DataFrame(index=hist.index), []

            base = pd.DataFrame({"ds": hist.index, "y": hist.values})
            m = _new_prophet()
            if use_exog:
                for c in use_cols:
                    m.add_regressor(c)
                base = base.join(X_train[use_cols])

            try:
                m.fit(base, seed=RANDOM_SEED)
            except TypeError:
                m.fit(base)

            future = pd.DataFrame({"ds": [t]})
            if use_exog:
                last_vals = X_train.iloc[-1]
                for c in use_cols:
                    future[c] = last_vals.get(c, 0.0)

            pred = m.predict(future)
            raw_next = max(0.0, float(pred["yhat"].iloc[0]))
            prev_level = float(hist.iloc[-1])
            yhat = _clip_step(prev_level, raw_next, lo_step, hi_step)

            bt_rows.append({
                "Country": cty, "Region": reg, "Metric": metr,
                "Date": t, "Actual": float(y.loc[t]), "Pred": float(yhat)
            })

        back = pd.DataFrame(bt_rows)
        if not back.empty:
            back["Date"] = pd.to_datetime(back["Date"]).dt.to_period("M").dt.to_timestamp(how="S")
            records_back.append(back)

        # ----------------- PRODUCTION FORECAST (12m) ---------------------
        use_exog = bool(ENABLE_EXOG and ex_cols_all)
        if use_exog:
            X_all, use_cols = _build_exog_slice(df, ex_cols_all, cty, reg, y.index)
            use_exog = bool(use_cols)
        else:
            X_all, use_cols = pd.DataFrame(index=y.index), []

        base_all = pd.DataFrame({"ds": y.index, "y": y.values})
        m_f = _new_prophet()
        if use_exog:
            for c in use_cols:
                m_f.add_regressor(c)
            base_all = base_all.join(X_all[use_cols])

        try:
            m_f.fit(base_all, seed=RANDOM_SEED)
        except TypeError:
            m_f.fit(base_all)

        fut_idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=HORIZON, freq="MS")
        future_all = pd.DataFrame({"ds": fut_idx})
        if use_exog:
            X_future = pd.DataFrame(index=fut_idx, columns=use_cols, dtype=float)
            X_future = pd.concat([X_all[use_cols], X_future]).ffill().loc[fut_idx]
            future_all = future_all.join(X_future)

        pred_f = m_f.predict(future_all)
        raw_fore = pred_f["yhat"].to_numpy(dtype=float)

        # sequential step clipping over horizon
        clipped = []
        level = float(y.iloc[-1])
        for v in raw_fore:
            level = _clip_step(level, max(0.0, float(v)), lo_step, hi_step)
            clipped.append(level)

        fc = pd.DataFrame({
            "Country": cty, "Region": reg, "Metric": metr,
            "Date": fut_idx, "Forecast": clipped
        })
        fc["Date"] = pd.to_datetime(fc["Date"]).dt.to_period("M").dt.to_timestamp(how="S")
        records_fore.append(fc)

        # ----------------- PLOT -----------------
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(y.index, y.values, label="Actual", lw=1.2)
        if not back.empty:
            ax.plot(back["Date"], back["Pred"], "--", label=f"OSS ({OSS_MONTHS}m)", lw=1.0)
        ax.plot(fc["Date"], fc["Forecast"], ":", label=f"Forecast +{HORIZON}", lw=1.0)
        ax.set_title(f"{cty} – {reg} – {metr}")
        ax.set_ylabel(metr); ax.grid(alpha=.3); ax.legend(loc="upper left")
        for lbl in ax.get_xticklabels(): lbl.set_rotation(45)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{_safe(cty)}_{_safe(reg)}_{_safe(metr)}.png", dpi=120)
        plt.close(fig)

    # ----------------- SAVE ARTEFACTS -----------------
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    fore_df = pd.concat(records_fore, ignore_index=True) if records_fore else pd.DataFrame()
    back_df = pd.concat(records_back, ignore_index=True) if records_back else pd.DataFrame()

    if not fore_df.empty:
        fore_df.to_csv(RUN_DIR/"forecast_h12.csv", index=False)
    back_df.to_csv(RUN_DIR/"backtest.csv", index=False)

    # series-level OSS metrics
    if not back_df.empty:
        perf = (
            back_df.groupby(["Country","Region","Metric"], as_index=False)
            .apply(lambda d: pd.Series({
                "MAE" : (d["Actual"]-d["Pred"]).abs().mean(),
                "MAPE": (np.abs(d["Actual"]-d["Pred"]) / d["Actual"].replace(0,np.nan)).mean()*100.0
            }))
            .reset_index(drop=True)
        )
        perf.to_excel(RUN_DIR/"series_scores.xlsx", index=False)
    else:
        perf = pd.DataFrame()

    tbl = Table(title=f"Model 05 Prophet (enhanced) — OSS last {OSS_MONTHS}m", pad_edge=False)
    tbl.add_column("Metric"); tbl.add_column("Median MAE", justify="right")
    tbl.add_column("Median MAPE%", justify="right")
    for metr in ["Received","Submitted"]:
        s = perf[perf["Metric"]==metr]
        if not s.empty:
            tbl.add_row(metr, f"{s['MAE'].median():,.0f}", f"{s['MAPE'].median():.1f}")
    console.print(tbl)
    (RUN_DIR/"tldr.txt").write_text(tbl.__str__())

    console.print(f"[bold green]All artefacts saved → {RUN_DIR}[/]")
    logging.info("✅ Prophet (enhanced) finished — artefacts saved to %s", RUN_DIR)

if __name__ == "__main__":
    main()
