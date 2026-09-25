# -*- coding: utf-8 -*-
"""
Model 04 — Recency-weighted seasonal MoM% (EDA-aware, OSS-eval)
----------------------------------------------------------------
Backtest: STRICT OSS (last K months only), month-map learned from history
before that window; predictions done recursively (no leakage).
Forecast: 12 months ahead using month-map built on full history.

If OUT_DIR/EDA/mom_bands.parquet is present, per-month rw_mean (seasonal prior)
and p05/p95 bands are used. Otherwise, we fallback to:
  • pct (seasonal MoM prior) = recency-weighted mean (last 2y vs older, clipped ±30%)
  • bounds from month-wise 5–95th pct (min ±20% overall)

Artefacts → …/Models/Model04_AvgMoM/
"""
from __future__ import annotations

import os, logging, warnings, re
from pathlib import Path
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
OSS_MONTHS = int(os.environ.get("RMG_OSS_MONTHS", "3"))  # strict out-of-sample months

def _safe(s: str) -> str:
    return re.sub(r'[\\/:\*?"<>|]', "_", s)

def _split_oss(y: pd.Series, k: int = OSS_MONTHS) -> tuple[pd.Series, pd.DatetimeIndex]:
    """Return (y_train, oss_idx) where the last k months are reserved for OSS."""
    y = y.sort_index()
    if len(y) <= k:
        return y.iloc[:0], y.index
    return y.iloc[:-k], y.index[-k:]

# Optional EDA artefacts
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

def _avg_mom_map(cty: str, reg: str, metr: str, y: pd.Series) -> dict[int, Tuple[float, float, float]]:
    """
    Build {month -> (pct, lo, hi)} using ONLY the series passed in.
      pct = seasonal MoM prior for that month (on % scale)
      lo,hi = multiplicative step bounds (e.g., 0.92..1.15)
    Prefers EDA's mom_bands (rw_mean, p05, p95). Fallback computes:
      pct via recency-weighted mean (last 2y vs older, clipped ±30%)
      lo/hi via month-wise 5–95% of MoM%, clamped to [0.8, 1.2]
    """
    out: dict[int, Tuple[float, float, float]] = {}

    # EDA path
    if MOM_BANDS is not None:
        sub = MOM_BANDS[
            (MOM_BANDS.Country == cty) &
            (MOM_BANDS.Region  == reg) &
            (MOM_BANDS.Metric  == metr)
        ]
        if not sub.empty:
            for _, r in sub.iterrows():
                m  = int(r["Month"])
                rw = float(r["rw_mean"])
                p05, p95 = float(r["p05"]), float(r["p95"])
                lo, hi = max(0.2, 1.0 + p05), min(2.5, 1.0 + p95)
                if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                    lo, hi = 0.7, 1.5
                out[m] = (float(np.clip(rw, -0.6, 0.6)), lo, hi)

    if len(out) == 12:
        return out

    # Fallback from the given series
    pct = y.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if pct.empty:
        return {m: (0.0, 0.7, 1.5) for m in range(1, 13)}

    end    = pct.index.max()
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
        rw = float(np.clip(val, -0.6, 0.6))
        # bounds from month-wise distribution (or default)
        ref = pct[pct.index.month == m]
        if len(ref) >= 3:
            q05, q95 = float(ref.quantile(0.05)), float(ref.quantile(0.95))
            lo, hi = max(0.2, 1.0 + q05), min(2.5, 1.0 + q95)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                lo, hi = 0.7, 1.5
        else:
            lo, hi = 0.7, 1.5
        out[m] = (rw, lo, hi)

    return out

def _predict_one_step_monthmap(hist: pd.Series,
                               t: pd.Timestamp,
                               month_map: dict[int, Tuple[float,float,float]]) -> float:
    """
    One-step prediction at date t using the given month_map and last hist value.
    month_map entries are (pct_on_%_scale, lo, hi) → step = clip(1+pct, lo, hi).
    """
    t = pd.Timestamp(t).to_period("M").to_timestamp(how="S")
    pct, lo, hi = month_map.get(int(t.month), (0.0, 0.8, 1.2))
    step = float(np.clip(1.0 + float(pct), float(lo), float(hi)))
    prev = float(hist.iloc[-1]) if len(hist) else 0.0
    return max(0.0, prev * step)

# ─────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)
console = Console()

RUN_DIR  = OUT_DIR / "Models" / "Model04_AvgMoM"
PLOT_DIR = RUN_DIR / "plots"
RUN_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=RUN_DIR/"model04_oss.log",
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

def main() -> None:
    # 1) Load & reshape
    df, _ = load_data()
    value_vars = [c for c in ["Received","Submitted"] if c in df.columns]
    if not value_vars:
        raise RuntimeError("No metric columns found (need at least one of 'Received','Submitted').")

    melt = (
        df.melt(id_vars=["Country","Region","Date"],
                value_vars=value_vars,
                var_name="Metric", value_name="Value")
          .drop_duplicates(subset=["Country","Region","Metric","Date"], keep="last")
          .sort_values(["Country","Region","Metric","Date"])
    )

    records_back: List[pd.DataFrame] = []
    records_fore: List[pd.DataFrame] = []
    mgmt_tables: Dict[Tuple[str,str,str], pd.DataFrame] = {}

    # 2) Loop each series
    for (cty, reg, metr), g in tqdm(
        melt.groupby(["Country","Region","Metric"], sort=False),
        desc="Model04_AvgMoM"
    ):
        y = (g.set_index("Date")["Value"]
               .astype("float64")
               .asfreq("MS")
               .interpolate("linear", limit_direction="both")
               .clip(lower=0.0))
        if len(y) < 24:
            logging.warning("Skipping %s/%s/%s (len<24)", cty, reg, metr)
            continue

        # ================== STRICT OSS BACKTEST (last K) ==================
        y_train, oss_idx = _split_oss(y, OSS_MONTHS)
        bt_rows: List[dict] = []
        if not y_train.empty and len(oss_idx) > 0:
            # month map from TRAIN ONLY (no leakage)
            month_map_train = _avg_mom_map(cty, reg, metr, y_train)
            hist = y_train.copy()
            for t in oss_idx:
                if hist.empty:
                    break
                yhat = _predict_one_step_monthmap(hist, t, month_map_train)
                bt_rows.append({"Date": t, "Actual": float(y.loc[t]), "Pred": float(yhat)})
                hist.loc[pd.Timestamp(t)] = yhat  # recursive extension
        bt = pd.DataFrame(bt_rows).dropna(subset=["Date","Actual","Pred"])
        if not bt.empty:
            bt.insert(0, "Country", cty)
            bt.insert(1, "Region",  reg)
            bt.insert(2, "Metric",  metr)
            bt["Date"] = pd.to_datetime(bt["Date"]).dt.to_period("M").dt.to_timestamp(how="S")
            records_back.append(bt)

        # =================== 12M PRODUCTION FORECAST =====================
        # month map from FULL history for production forecast
        month_map_full = _avg_mom_map(cty, reg, metr, y)
        fut_idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=HORIZON, freq="MS")
        hist_f = y.copy()
        fut_vals: List[float] = []
        for t in fut_idx:
            yhat = _predict_one_step_monthmap(hist_f, t, month_map_full)
            fut_vals.append(yhat)
            hist_f.loc[pd.Timestamp(t)] = yhat
        fc_df = pd.DataFrame({
            "Country":  cty,
            "Region":   reg,
            "Metric":   metr,
            "Date":     fut_idx,
            "Forecast": fut_vals
        })
        fc_df["Date"] = pd.to_datetime(fc_df["Date"]).dt.to_period("M").dt.to_timestamp(how="S")

        # month-wise outlier flag using the same bands
        mom = fc_df["Forecast"].pct_change()
        bounds = fc_df["Date"].map(lambda d: month_map_full.get(d.month, (0.0, 0.8, 1.2)))
        lo_pct = bounds.map(lambda t: (t[1] - 1.0) * 100.0)
        hi_pct = bounds.map(lambda t: (t[2] - 1.0) * 100.0)
        fc_df["MoM%"] = mom * 100.0
        fc_df["Flag_Outlier"] = (fc_df["MoM%"] < lo_pct) | (fc_df["MoM%"] > hi_pct)
        records_fore.append(fc_df)

        # ================ Management table (historical YoY) ==============
        rows = []
        for i in range(1, len(y)):
            d      = y.index[i]
            prevFY = d - pd.DateOffset(years=1)
            prevMo = d - pd.DateOffset(months=1)
            if prevFY not in y.index or prevMo not in y.index:
                continue
            pct, lo, hi = month_map_full.get(d.month, (0.0, 0.8, 1.2))
            step   = float(np.clip(1.0 + pct, lo, hi))
            predPY = max(0.0, float(y.loc[prevMo]) * step)
            rows.append([
                d.date(), cty, reg, metr,
                float(y.loc[prevFY]), float(y.loc[d]),
                round(pct * 100.0, 2), predPY,
                round(predPY - float(y.loc[d]), 2)
            ])
        if rows:
            mgmt_tables[(cty,reg,metr)] = pd.DataFrame(
                rows, columns=[
                    "Month","Country","Region","Metric",
                    "Actual_prevFY","Actual_currFY",
                    "Avg_MoM%","Pred_from_prevFY","Error"
                ]
            )

        # ============================== Plot =============================
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(y.index, y.values, label="Actual", lw=1.2)
        if not bt.empty:
            ax.plot(bt["Date"], bt["Pred"], "--", label=f"OSS ({OSS_MONTHS}m)", lw=1.0)
        ax.plot(fc_df["Date"], fc_df["Forecast"], ":", label=f"Forecast +{HORIZON}", lw=1.0)
        ax.set_title(f"{cty} – {reg} – {metr}")
        ax.set_ylabel(metr)
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
        fig.tight_layout()
        fig.savefig(PLOT_DIR/f"{_safe(cty)}_{_safe(reg)}_{_safe(metr)}.png", dpi=120)
        plt.close(fig)

    # 3) Save backtest & forecast CSVs
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    back_df     = pd.concat(records_back, ignore_index=True) if records_back else pd.DataFrame()
    forecast_df = pd.concat(records_fore, ignore_index=True) if records_fore else pd.DataFrame()
    back_df.to_csv(RUN_DIR/"backtest.csv", index=False)
    forecast_df.to_csv(RUN_DIR/"forecast_h12.csv", index=False)

    # 4) Save management tables
    mgmt_df = pd.concat(mgmt_tables.values(), ignore_index=True) if mgmt_tables else pd.DataFrame()
    with pd.ExcelWriter(RUN_DIR/"mgmt_table.xlsx") as w:
        (mgmt_df if not mgmt_df.empty else
         pd.DataFrame({"info":["no rows"]})
        ).to_excel(w, sheet_name="Mgmt_Table", index=False)

    if not mgmt_df.empty:
        styled = (mgmt_df.style
                  .format({"Avg_MoM%":"{:.1f}%", "Error":"{:+,.0f}"})
                  .background_gradient(subset="Error", cmap="RdYlGn_r")
                  .hide(axis="index"))
        Path(RUN_DIR/"mgmt_table.html").write_text(styled.to_html(table_uuid="tbl", notebook=False),
                                                   encoding="utf-8")

    # 5) Series-level performance (OSS only) & TL;DR
    if not back_df.empty:
        tmp = back_df.assign(
            e_abs=lambda d: (d["Actual"] - d["Pred"]).abs(),
            denom=lambda d: d["Actual"].replace(0, np.nan).abs()
        )
        tmp["ape"] = (tmp["e_abs"] / tmp["denom"]).replace([np.inf, -np.inf], np.nan) * 100.0
        perf = (
            tmp.groupby(["Country","Region","Metric"], as_index=False)
               .agg(MAE=("e_abs","mean"), MAPE=("ape","mean"))
        )
        perf.to_excel(RUN_DIR/"series_scores.xlsx", index=False)

        tbl = Table(title=f"Model 04 — OSS last {OSS_MONTHS} months (median errors)", pad_edge=False)
        tbl.add_column("Metric"); tbl.add_column("Median MAE", justify="right")
        tbl.add_column("Median MAPE %", justify="right")
        for metr in ["Received","Submitted"]:
            sub = perf[perf["Metric"]==metr]
            if not sub.empty:
                tbl.add_row(f"{metr}", f"{sub.MAE.median():,.0f}", f"{sub.MAPE.median():.1f}")
        console.print(tbl)
        Path(RUN_DIR/"tldr.txt").write_text(tbl.__str__())

    console.print(f"[bold green]✅ All artefacts saved → {RUN_DIR}[/]")
    logging.info("✅ Model04 OSS pipeline complete → %s", RUN_DIR)

if __name__ == "__main__":
    main()
