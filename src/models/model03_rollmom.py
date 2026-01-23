# -*- coding: utf-8 -*-
from __future__ import annotations

import os, warnings, logging, re
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

# weight for blending seasonal prior vs. rolling median
PRIOR_WEIGHT = 0.3  # mu_blend = (1-PRIOR_WEIGHT)*rolling + PRIOR_WEIGHT*rw_mean

# ─────────────────────────────────────────────────────────────────────
# Helpers
def _safe(s: str) -> str:
    return re.sub(r'[\\/:\*?"<>|]', "_", s)

def _split_oss(y: pd.Series, k: int = OSS_MONTHS) -> tuple[pd.Series, pd.DatetimeIndex]:
    """
    Return (y_train, oss_idx) where the last k months are reserved for OSS.
    """
    y = y.sort_index()
    if len(y) <= k:
        return y.iloc[:0], y.index
    return y.iloc[:-k], y.index[-k:]

def _series_bounds_fallback(pct: pd.Series) -> tuple[float, float]:
    """
    Fallback bounds from 5th/95th pct of last 12M MoM% as multiplicative step,
    clamped to [0.8, 1.2].
    """
    clean = pct.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return 0.8, 1.2
    end    = clean.index.max()
    start  = end - pd.DateOffset(months=12)
    recent = clean[(clean.index >= start) & (clean.index <= end)]
    if len(recent) < 2:
        recent = clean
    lo_q, hi_q = recent.quantile([0.05, 0.95])
    lower = max(0.8, 1.0 + float(lo_q))
    upper = min(1.2, 1.0 + float(hi_q))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        lower, upper = 0.8, 1.2
    return float(lower), float(upper)

def _monthwise_from_eda(cty: str, reg: str, metr: str, month: int) -> tuple[float, float, float]:
    """
    From mom_bands: return (lo_step, hi_step, rw_mean) for series+month.
    Fallback to (0.8, 1.2, 0.0) if EDA absent/missing row.
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
    lo = max(0.8, 1.0 + p05)
    hi = min(1.2, 1.0 + p95)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = 0.8, 1.2
    # safety on rw_mean as % (cap ±50%)
    rw = float(np.clip(rw, -0.5, 0.5))
    return lo, hi, rw

def _rolling_median(pct: pd.Series, t: pd.Timestamp) -> float:
    """
    Median MoM% over the last 12 months, winsorized to ±30%.
    Uses only data up to t-1 (no leakage).
    """
    t = pd.Timestamp(t).to_period("M").to_timestamp(how="S")
    end    = t - pd.offsets.MonthBegin()
    start  = end - pd.DateOffset(months=12)
    window = pct.loc[(pct.index > start) & (pct.index <= end)].dropna()
    if window.empty:
        return 0.0
    wins = window.clip(lower=-0.3, upper=0.3)
    val = float(wins.median())
    return 0.0 if not np.isfinite(val) else val

def _predict_one_step_rollmom(cty: str, reg: str, metr: str,
                              hist_levels: pd.Series,
                              hist_pct: pd.Series,
                              target_date: pd.Timestamp,
                              lo_fb: float, hi_fb: float) -> tuple[float, float]:
    """
    One-step prediction for target_date using rolling median MoM blended with EDA prior,
    bounded by month-specific bands (EDA or fallback). Returns (yhat, mu_blend).
    """
    t = pd.Timestamp(target_date).to_period("M").to_timestamp(how="S")
    mu = _rolling_median(hist_pct, t)  # MoM% from history only

    lo_m, hi_m, rw = _monthwise_from_eda(cty, reg, metr, t.month)
    if MOM_BANDS is None:
        lo_m, hi_m = lo_fb, hi_fb
        rw = 0.0

    mu_blend = (1.0 - PRIOR_WEIGHT) * mu + PRIOR_WEIGHT * rw
    step     = float(np.clip(1.0 + mu_blend, lo_m, hi_m))

    prev = float(hist_levels.iloc[-1]) if len(hist_levels) else 0.0
    yhat = max(0.0, prev * step)
    return yhat, mu_blend

# ─────────────────────────────────────────────────────────────────────
# Pipeline
warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)
console = Console()

RUN_DIR  = OUT_DIR / "Models" / "Model03_RollMoM"
PLOT_DIR = RUN_DIR / "plots"
RUN_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=RUN_DIR / "model03.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def main() -> None:
    # 1) Load & reshape
    df, _ = load_data()
    value_vars = [c for c in ["Received","Submitted"] if c in df.columns]
    melt = (
        df.melt(id_vars=["Country","Region","Date"],
                value_vars=value_vars,
                var_name="Metric", value_name="Value")
          .drop_duplicates(subset=["Country","Region","Metric","Date"], keep="last")
          .sort_values(["Country","Region","Metric","Date"])
    )

    records_fore: List[pd.DataFrame] = []
    records_back: List[pd.DataFrame] = []
    mgmt_tables: Dict[Tuple[str,str,str], pd.DataFrame] = {}

    # 2) Loop series
    for (cty, reg, metr), g in tqdm(melt.groupby(["Country","Region","Metric"], sort=False),
                                    desc="Model03_RollMoM"):
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

        # Fallback band if EDA missing (computed on FULL series; used in both OSS & forecast)
        lo_fb, hi_fb = _series_bounds_fallback(y.pct_change())

        # ============== STRICT OSS BACKTEST (last K months) ==============
        y_train, oss_idx = _split_oss(y, OSS_MONTHS)
        hist_levels = y_train.copy()
        hist_pct    = y_train.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

        bt_rows = []
        for t in oss_idx:
            if hist_levels.empty:
                break
            yhat, mu_blend = _predict_one_step_rollmom(cty, reg, metr, hist_levels, hist_pct, t, lo_fb, hi_fb)
            bt_rows.append({"Date": t, "Actual": float(y.loc[t]), "Pred": float(yhat)})
            # extend history with prediction (recursive, no leakage)
            hist_levels.loc[t] = yhat
            hist_pct.loc[t]    = mu_blend

        bt = pd.DataFrame(bt_rows).dropna(subset=["Date","Actual","Pred"])
        if not bt.empty:
            bt.insert(0, "Country", cty)
            bt.insert(1, "Region",  reg)
            bt.insert(2, "Metric",  metr)
            bt["Date"] = pd.to_datetime(bt["Date"]).dt.to_period("M").dt.to_timestamp(how="S")
            records_back.append(bt)

        # ==================== 12M PRODUCTION FORECAST =====================
        hist_full = y.copy()
        pct_full  = y.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        fut_idx   = pd.date_range(hist_full.index[-1] + pd.offsets.MonthBegin(), periods=HORIZON, freq="MS")
        fut_vals  = []

        for t in fut_idx:
            yhat, mu_blend = _predict_one_step_rollmom(cty, reg, metr, hist_full, pct_full, t, lo_fb, hi_fb)
            hist_full.loc[t] = yhat
            pct_full.loc[t]  = mu_blend
            fut_vals.append(yhat)

        fc_df = pd.DataFrame({
            "Country":  cty,
            "Region":   reg,
            "Metric":   metr,
            "Date":     fut_idx,
            "Forecast": fut_vals
        })
        fc_df["Date"] = (
            pd.to_datetime(fc_df["Date"], errors="coerce")
            .dt.to_period("M")
            .dt.to_timestamp()   # start of month
        )
        fc_df = fc_df.dropna(subset=["Date"])

        # simple diagnostics
        fc_df["MoM%"] = fc_df["Forecast"].pct_change() * 100.0
        if MOM_BANDS is None:
            low_pct = (lo_fb - 1.0) * 100.0
            hi_pct  = (hi_fb - 1.0) * 100.0
            fc_df["Flag_Outlier"] = (fc_df["MoM%"] < low_pct) | (fc_df["MoM%"] > hi_pct)
        else:
            fc_df["Flag_Outlier"] = False

        records_fore.append(fc_df)

        # management table (YoY with MoM clip on history)
        rows = []
        for d_curr in y.index[13:]:
            d_prevFY  = d_curr - pd.DateOffset(years=1)
            d_prevMon = d_curr - pd.DateOffset(months=1)
            if d_prevFY not in y.index or d_prevMon not in y.index:
                continue
            prev_valFY = float(y.loc[d_prevFY])
            prev_valMo = float(y.loc[d_prevMon])
            curr_val   = float(y.loc[d_curr])
            raw_mom    = (curr_val / prev_valMo - 1.0) if prev_valMo != 0 else 0.0
            mom_clip   = float(np.clip(raw_mom, -0.5, 0.5))
            pred_prev  = max(0.0, prev_valFY * (1.0 + mom_clip))
            rows.append([
                d_curr.date(), cty, reg, metr,
                prev_valFY, curr_val,
                round(mom_clip*100.0, 2), pred_prev,
                round(pred_prev - curr_val, 2)
            ])
        if rows:
            mgmt_tables[(cty,reg,metr)] = pd.DataFrame(
                rows, columns=[
                    "Month","Country","Region","Metric",
                    "Actual_prevFY","Actual_currFY",
                    "MoM%_currFY","Pred_from_prevFY","Error"
                ]
            )

        # plot Actual vs OSS backtest vs Forecast
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(y.index, y.values, label="Actual", lw=1.2)
        if not bt.empty:
            ax.plot(bt["Date"], bt["Pred"], "--", label=f"OSS ({OSS_MONTHS}m)", lw=1.0)
        ax.plot(fc_df["Date"], fc_df["Forecast"], ":", label=f"Forecast +{HORIZON}", lw=1.0)
        ax.set_title(f"{cty} – {reg} – {metr}")
        ax.set_ylabel(metr)
        ax.grid(alpha=0.3); ax.legend()
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
        fig.tight_layout()
        fig.savefig(PLOT_DIR/f"{_safe(cty)}_{_safe(reg)}_{_safe(metr)}.png", dpi=120)
        plt.close(fig)

    # 3) Save artefacts
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    fore_df = pd.concat(records_fore, ignore_index=True) if records_fore else pd.DataFrame()
    back_df = pd.concat(records_back, ignore_index=True) if records_back else pd.DataFrame()
    mgmt_df = pd.concat(mgmt_tables.values(), ignore_index=True) if mgmt_tables else pd.DataFrame()

    if not fore_df.empty:
        fore_df.to_csv(RUN_DIR/"forecast_h12.csv", index=False)
    back_df.to_csv(RUN_DIR/"backtest.csv", index=False)

    with pd.ExcelWriter(RUN_DIR/"mgmt_table.xlsx") as w:
        (mgmt_df if not mgmt_df.empty else
         pd.DataFrame({"info":["no rows produced"]})
        ).to_excel(w, sheet_name="Mgmt_Table", index=False)

    # 4) Performance & TL;DR (OSS only)
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

    tbl = Table(title=f"Model03 — OSS last {OSS_MONTHS} months (median errors)", pad_edge=False)
    tbl.add_column("Metric"); tbl.add_column("Median MAE", justify="right")
    tbl.add_column("Median MAPE %", justify="right")
    for metr in ["Received","Submitted"]:
        sub = perf[perf["Metric"]==metr]
        if not sub.empty:
            tbl.add_row(
                metr,
                f"{sub['MAE'].median():,.0f}",
                f"{sub['MAPE'].median():.1f}"
            )
    console.print(tbl)
    (RUN_DIR/"tldr.txt").write_text(tbl.__str__())
    console.print(f"[bold green]All artefacts saved → {RUN_DIR}[/]")
    logging.info("✅ Model03 saved to %s", RUN_DIR)

if __name__ == "__main__":
    main()
