# -*- coding: utf-8 -*-
from __future__ import annotations
import os, logging, warnings, re
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from ..config import OUT_DIR, HORIZON, RANDOM_SEED
from ..loader import load_data

# ─────────────────────────────────────────────────────────────────────
# Optional EDA artefact (month-of-year MoM bands)
EDA_DIR = OUT_DIR / "EDA"

def _load_mom_bands() -> Optional[pd.DataFrame]:
    p = EDA_DIR / "mom_bands.parquet"
    try:
        df = pd.read_parquet(p)
        req = {"Country","Region","Metric","Month","p05","p95","rw_mean"}
        return df if req.issubset(df.columns) else None
    except Exception:
        return None

MOM_BANDS = _load_mom_bands()

# OSS (out-of-sample) holdout length for backtest (env-tunable)
OSS_MONTHS = int(os.environ.get("RMG_OSS_MONTHS", "3"))

# ─────────────────────────────────────────────────────────────────────
def _safe(s: str) -> str:
    return re.sub(r'[\\/:\*?"<>|]', "_", s)

def fiscal_year(ts: pd.Timestamp) -> int:
    return ts.year + (ts.month >= 4)

def _sdiv(n, d, default=0.0) -> float:
    """Safe division with fallback and finite guard."""
    try:
        d = float(d)
        if d == 0 or not np.isfinite(d):
            return default
        v = float(n) / d
        return v if np.isfinite(v) else default
    except Exception:
        return default

def _split_oss(y: pd.Series, k: int = OSS_MONTHS) -> tuple[pd.Series, pd.DatetimeIndex]:
    """
    Return (y_train, oss_idx) where the last k months of y are reserved for OSS.
    """
    y = y.sort_index()
    if len(y) <= k:
        return y.iloc[:0], y.index
    return y.iloc[:-k], y.index[-k:]

def _monthwise_bounds_and_prior(cty: str, reg: str, metr: str, month: int) -> tuple[float,float,float]:
    """
    Return (lo_step, hi_step, rw_mean) for this series+calendar month using EDA mom_bands
    when available; otherwise sensible defaults. lo/hi are multiplicative step bounds.
    """
    if MOM_BANDS is None:
        return 0.8, 1.2, 0.0
    sub = MOM_BANDS[
        (MOM_BANDS.Country == cty) &
        (MOM_BANDS.Region  == reg) &
        (MOM_BANDS.Metric  == metr) &
        (MOM_BANDS.Month   == int(month))
    ]
    if sub.empty:
        return 0.8, 1.2, 0.0
    p05, p95, rw = float(sub.p05.iloc[0]), float(sub.p95.iloc[0]), float(sub.rw_mean.iloc[0])
    lo = max(0.2, 1.0 + p05)  # allow real seasonal drops down to -80%
    hi = min(2.5, 1.0 + p95)  # allow real seasonal surges up to +150%
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = 0.7, 1.5
    rw = float(np.clip(rw, -0.6, 0.6))
    return lo, hi, rw

def _drift_factor(levels: pd.Series, t: pd.Timestamp, lam: float = 0.7) -> float:
    """
    Annual level momentum drift:
    Compare recent 3-month volume in current FY vs same period in previous FY.
    Uses strictly history up to t-1 (no leakage).
    """
    fy    = fiscal_year(t)
    start = pd.Timestamp(fy - 1, 4, 1)
    cur   = levels.loc[start : t - pd.offsets.MonthBegin()].dropna().tail(3)
    prev  = levels.loc[(start - pd.DateOffset(years=1)) : t - pd.DateOffset(months=13)].dropna().tail(3)
    if cur.empty or prev.empty:
        raw = 1.0
    else:
        s_cur = float(cur.sum())
        s_prev = float(prev.sum())
        raw = _sdiv(s_cur, s_prev, default=1.0) if s_prev > 0 else 1.0
    drift = lam * raw + (1.0 - lam) * 1.0
    return float(np.clip(drift, 0.8, 1.25))

# blend strength between seasonal prior and raw YoY MoM (additive on % scale)
PRIOR_WEIGHT = 0.4  # mom_blend = (1-PRIOR_WEIGHT)*raw_mom + PRIOR_WEIGHT*rw_mean

def _predict_one_step_mom_yoy(cty: str, reg: str, metr: str,
                              hist_levels: pd.Series,
                              hist_pct: pd.Series,
                              target_date: pd.Timestamp) -> tuple[float, float]:
    """
    One-step prediction at target_date using:
      • raw YoY MoM from hist_levels (needs t-12 & t-13), clipped ±50%
      • seasonal prior (rw_mean) blended in
      • FY drift computed from hist_levels (history only)
      • month-specific bands from EDA (or defaults)
    Returns (yhat, mom_blend_used)
    """
    t = pd.Timestamp(target_date).to_period("M").to_timestamp(how="S")

    if (t - pd.DateOffset(years=1) in hist_levels.index) and (t - pd.DateOffset(months=13) in hist_levels.index):
        raw_mom = _sdiv(hist_levels.loc[t - pd.DateOffset(years=1)],
                        hist_levels.loc[t - pd.DateOffset(months=13)],
                        default=1.0) - 1.0
    else:
        raw_mom = 0.0
    raw_mom = float(np.clip(raw_mom, -0.5, 0.5))

    lo, hi, rw = _monthwise_bounds_and_prior(cty, reg, metr, t.month)
    mom_blend  = (1.0 - PRIOR_WEIGHT) * raw_mom + PRIOR_WEIGHT * rw

    drift = _drift_factor(hist_levels, t)
    step  = float(np.clip((1.0 + mom_blend) * drift, lo, hi))

    prev = float(hist_levels.iloc[-1]) if len(hist_levels) else 0.0
    yhat = max(0.0, prev * step)
    return yhat, mom_blend

# ─────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)
console = Console()

RUN_DIR  = OUT_DIR / "Models" / "Model02_MoM_YoY"
PLOT_DIR = RUN_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=RUN_DIR / "model02.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def main() -> None:
    # Load & reshape
    df, _ = load_data()
    value_vars = [c for c in ["Received", "Submitted"] if c in df.columns]
    melt = (
        df.melt(id_vars=["Country", "Region", "Date"],
                value_vars=value_vars,
                var_name="Metric", value_name="Value")
          .drop_duplicates(subset=["Country", "Region", "Metric", "Date"], keep="last")
          .sort_values(["Country", "Region", "Metric", "Date"])
    )

    records_fore: List[pd.DataFrame] = []
    records_back: List[pd.DataFrame] = []
    mgmt_tables: Dict[Tuple[str, str, str], pd.DataFrame] = {}

    for (cty, reg, metr), g in tqdm(melt.groupby(["Country", "Region", "Metric"], sort=False),
                                    desc="Model02_MoM_YoY"):
        # 1) Prepare series
        y = (
            g.set_index("Date")["Value"]
             .astype("float64")
             .asfreq("MS")
             .interpolate("linear", limit_direction="both")
             .clip(lower=0.0)
        )
        if len(y) < 24:
            logging.warning("%s/%s/%s skipped (len<24)", cty, reg, metr)
            continue

        # ============== STRICT OSS BACKTEST (last K months) ==============
        y_train, oss_idx = _split_oss(y, OSS_MONTHS)
        pct_hist = y_train.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

        bt_rows = []
        hist_levels = y_train.copy()
        hist_pct    = pct_hist.copy()

        for t in oss_idx:
            if hist_levels.empty:
                break
            yhat, mom_blend = _predict_one_step_mom_yoy(cty, reg, metr, hist_levels, hist_pct, t)
            bt_rows.append({"Date": t, "Actual": float(y.loc[t]), "Pred": float(yhat)})
            # extend history with prediction (recursive, still no leakage)
            hist_levels.loc[t] = yhat
            hist_pct.loc[t]    = mom_blend

        bt = pd.DataFrame(bt_rows).dropna(subset=["Date","Actual","Pred"])
        if not bt.empty:
            bt.insert(0, "Country", cty)
            bt.insert(1, "Region",  reg)
            bt.insert(2, "Metric",  metr)
            bt["Date"] = pd.to_datetime(bt["Date"]).dt.to_period("M").dt.to_timestamp(how="S")
            records_back.append(bt)

        # ==================== 12M PRODUCTION FORECAST =====================
        pct_full  = y.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        hist_full = y.copy()
        fut_vals  = []
        fut_idx   = pd.date_range(hist_full.index[-1] + pd.offsets.MonthBegin(), periods=HORIZON, freq="MS")

        for t in fut_idx:
            yhat, mom_blend = _predict_one_step_mom_yoy(cty, reg, metr, hist_full, pct_full, t)
            hist_full.loc[t] = yhat
            pct_full.loc[t]  = mom_blend
            fut_vals.append(yhat)

        fc_df = pd.DataFrame({
            "Country": cty, "Region": reg, "Metric": metr,
            "Date": fut_idx, "Forecast": fut_vals
        })
        fc_df["Date"]         = pd.to_datetime(fc_df["Date"]).dt.to_period("M").dt.to_timestamp(how="S")
        fc_df["MoM%"]         = fc_df["Forecast"].pct_change() * 100.0
        fc_df["Flag_Outlier"] = fc_df["MoM%"].abs() > (25.0 if MOM_BANDS is not None else 30.0)
        records_fore.append(fc_df)

        # 4) Management table (historical YoY with MoM clip)
        rows = []
        for d_curr in y.index[13:]:
            d_prevFY  = d_curr - pd.DateOffset(years=1)
            d_prevMon = d_curr - pd.DateOffset(months=1)
            if d_prevFY not in y.index or d_prevMon not in y.index:
                continue
            val_prevFY = float(y.loc[d_prevFY])
            val_prevMo = float(y.loc[d_prevMon])
            val_curr   = float(y.loc[d_curr])
            raw_mom    = _sdiv(val_curr, val_prevMo, default=1.0) - 1.0
            mom        = float(np.clip(raw_mom, -0.5, 0.5))
            pred       = max(0.0, val_prevFY * (1.0 + mom))
            rows.append([
                d_curr.date(), cty, reg, metr,
                val_prevFY, val_curr,
                round(mom * 100.0, 2), pred,
                round(pred - val_curr, 2)
            ])
        if rows:
            mgmt_tables[(cty, reg, metr)] = pd.DataFrame(
                rows, columns=[
                    "Month", "Country", "Region", "Metric",
                    "Actual_prevFY", "Actual_currFY",
                    "MoM%_currFY", "Pred_from_prevFY", "Error"
                ]
            )

        # 5) Plot
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(y.index, y.values, label="Actual", lw=1.2)
        if not bt.empty:
            ax.plot(bt["Date"], bt["Pred"], "--", label=f"OSS ({OSS_MONTHS}m)", lw=1.0)
        ax.plot(fc_df["Date"], fc_df["Forecast"], ":", label=f"Forecast +{HORIZON}", lw=1.0)
        ax.set_title(f"{cty} – {reg} – {metr}")
        ax.set_ylabel(metr)
        ax.grid(alpha=.3); ax.legend()
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{_safe(cty)}_{_safe(reg)}_{_safe(metr)}.png", dpi=120)
        plt.close(fig)

    # 6) SAVE ARTEFACTS
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    fore_df = pd.concat(records_fore, ignore_index=True) if records_fore else pd.DataFrame()
    back_df = pd.concat(records_back, ignore_index=True) if records_back else pd.DataFrame()
    mgmt_df = pd.concat(mgmt_tables.values(), ignore_index=True) if mgmt_tables else pd.DataFrame()

    if not fore_df.empty:
        fore_df.to_csv(RUN_DIR / "forecast_h12.csv", index=False)
    back_df.to_csv(RUN_DIR / "backtest.csv", index=False)

    with pd.ExcelWriter(RUN_DIR / "mgmt_table.xlsx") as w:
        (mgmt_df if not mgmt_df.empty else
         pd.DataFrame({"info": ["no rows produced"]})
         ).to_excel(w, sheet_name="Mgmt_Table", index=False)

    # 7) PERFORMANCE & TL;DR (on OSS only; robust aggregation)
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

    tbl = Table(title=f"Model 02 — OSS last {OSS_MONTHS} months (median errors)", pad_edge=False)
    tbl.add_column("Metric"); tbl.add_column("MAE", justify="right")
    tbl.add_column("MAPE%", justify="right")
    for metr in ["Received", "Submitted"]:
        s = perf[perf["Metric"] == metr]
        if not s.empty:
            tbl.add_row(metr, f"{s['MAE'].median():,.0f}", f"{s['MAPE'].median():.1f}")
    console.print(tbl)
    (RUN_DIR / "tldr.txt").write_text(tbl.__str__())
    console.print(f"[bold green]All artefacts saved → {RUN_DIR}[/]")
    logging.info("✅ Model02 saved to %s", RUN_DIR)

if __name__ == "__main__":
    main()
