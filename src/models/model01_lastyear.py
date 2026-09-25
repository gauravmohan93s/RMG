# -*- coding: utf-8 -*-
from __future__ import annotations
import os, logging, warnings, re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from ..config import OUT_DIR, HORIZON, RANDOM_SEED
from ..loader import load_data

# ─────────────────────────────────────────────────────────────────────
def _safe(label: str) -> str:
    return re.sub(r'[\\/:\*?"<>|]', "_", label)

def fiscal_year(ts: pd.Timestamp) -> int:
    return ts.year + (ts.month >= 4)

def _safe_div(n: float, d: float, default: float = 0.0) -> float:
    """Finite & zero-safe division."""
    try:
        n = float(n); d = float(d)
        if d == 0 or not np.isfinite(d):
            return default
        v = n / d
        return v if np.isfinite(v) else default
    except Exception:
        return default

# --- OSS (out-of-sample) holdout settings (env-tunable) ---
OSS_MONTHS = int(os.environ.get("RMG_OSS_MONTHS", "3"))

def split_oss(y: pd.Series, k: int = OSS_MONTHS):
    """
    Return (y_train, oss_idx) where the last k months are reserved as
    out-of-sample targets. y_train excludes those months.
    """
    y = y.sortindex() if hasattr(y, "sortindex") else y.sort_index()
    if len(y) <= k:
        return y.iloc[:0], y.index   # degenerate but safe
    return y.iloc[:-k], y.index[-k:]

# ── INIT ─────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)
console = Console()

RUN_DIR  = OUT_DIR / "Models" / "Model01_LastYear"
PLOT_DIR = RUN_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=RUN_DIR/"model01.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ── CORE FUNCTIONS ───────────────────────────────────────────────────
def _predict_one_step_lastyear(hist: pd.Series, target_date: pd.Timestamp) -> float:
    """
    Predict level at `target_date` using last-year MoM step applied to the
    last available level in `hist`. No peeking at actuals beyond hist end.
    MoM is clipped to ±50% and result to ≥ 0.
    """
    if hist.empty:
        return 0.0
    t = pd.Timestamp(target_date).to_period("M").to_timestamp(how="S")
    d12 = t - pd.DateOffset(years=1)
    d13 = t - pd.DateOffset(months=13)
    prev = float(hist.iloc[-1])
    if (d12 in hist.index) and (d13 in hist.index):
        mom = _safe_div(hist.loc[d12], hist.loc[d13], default=1.0) - 1.0
        mom = float(np.clip(mom, -0.5, 0.5))
        return max(0.0, prev * (1.0 + mom))
    # fallback: carry last level (rare, very short histories)
    return max(0.0, prev)

def _last_year_mom_forecast(y: pd.Series, steps: int) -> pd.Series:
    """
    Forecast next 'steps' values using last-year MoM growth applied to
    the last observed level. Non-negative. Returns MS-indexed series.
    """
    y = y.copy()
    if y.empty:
        return pd.Series(dtype=float)
    fut_idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq="MS")
    history = y.copy()
    future_vals = []
    for t in fut_idx:
        yhat = _predict_one_step_lastyear(history, t)
        history.loc[t] = yhat  # recursive (still zero leakage)
        future_vals.append(yhat)
    return pd.Series(future_vals, index=fut_idx, dtype="float64")

# ── MAIN ─────────────────────────────────────────────────────────────
def main() -> None:
    # load once (already monthly & capped by EDA/loader)
    df, _ = load_data()

    melt = (
        df.melt(id_vars=["Country","Region","Date"],
                value_vars=[c for c in ["Received","Submitted"] if c in df.columns],
                var_name="Metric", value_name="Value")
          .drop_duplicates(subset=["Country","Region","Metric","Date"], keep="last")
          .sort_values(["Country","Region","Metric","Date"])
    )

    records_fore: List[pd.DataFrame] = []
    records_back: List[pd.DataFrame] = []
    mgmt_tables: Dict[Tuple[str,str,str], pd.DataFrame] = {}

    for (cty, reg, metr), g in tqdm(
        melt.groupby(["Country","Region","Metric"], sort=False),
        desc="Model01_LastYear"
    ):
        # monthly, continuous, interpolated
        y = (
            g.set_index("Date")["Value"]
             .astype("float64")
             .asfreq("MS")
             .interpolate("linear", limit_direction="both")
             .clip(lower=0.0)
        )
        if y.empty:
            continue

        # ============== STRICT OSS BACKTEST (last 3 months) ==============
        y_train, oss_idx = split_oss(y, OSS_MONTHS)

        bt_rows = []
        hist = y_train.copy()  # extends recursively with predictions
        for t in oss_idx:
            if hist.empty:
                break
            yhat = _predict_one_step_lastyear(hist, t)
            bt_rows.append({"Date": t, "Actual": float(y.loc[t]), "Pred": float(yhat)})
            hist.loc[t] = yhat  # recursive, no leakage

        bt = pd.DataFrame(bt_rows)
        if not bt.empty:
            bt.insert(0, "Country", cty)
            bt.insert(1, "Region",  reg)
            bt.insert(2, "Metric",  metr)
            bt = bt.dropna(subset=["Date","Actual","Pred"])
            bt["Date"] = pd.to_datetime(bt["Date"]).dt.to_period("M").dt.to_timestamp(how="S")
            records_back.append(bt)

        # ==================== 12M PRODUCTION FORECAST =====================
        fc_ser = _last_year_mom_forecast(y, HORIZON)
        if not fc_ser.empty:
            fc_df = pd.DataFrame({
                "Country": cty, "Region": reg, "Metric": metr,
                "Date": fc_ser.index, "Forecast": fc_ser.values
            })
            # simple MoM diagnostics
            fc_df["FCast_MoM%"]   = fc_df["Forecast"].pct_change() * 100.0
            fc_df["Flag_Outlier"] = fc_df["FCast_MoM%"].abs() > 50.0
            records_fore.append(fc_df)

        # — Management table (FY YoY view on *historical* actuals)
        rows = []
        for d_curr in y.index[13:]:
            d_prevFY  = d_curr - pd.DateOffset(years=1)
            d_prevMon = d_curr - pd.DateOffset(months=1)
            if d_prevFY not in y.index or d_prevMon not in y.index:
                continue
            val_prevFY = float(y.loc[d_prevFY])
            val_prevMo = float(y.loc[d_prevMon])
            val_curr   = float(y.loc[d_curr])
            mom_pct    = _safe_div(val_curr, val_prevMo, default=1.0) - 1.0
            pred_from_prevFY = max(0.0, val_prevFY * (1.0 + mom_pct))
            rows.append([
                d_curr.date(), cty, reg, metr,
                val_prevFY, val_curr,
                round(mom_pct*100.0, 2),
                pred_from_prevFY,
                round(pred_from_prevFY - val_curr, 2)
            ])
        if rows:
            mgmt_tables[(cty,reg,metr)] = pd.DataFrame(
                rows, columns=[
                    "Month","Country","Region","Metric",
                    "Actual_prevFY","Actual_currFY",
                    "MoM%_currFY","Pred_from_prevFY","Error"
                ]
            )

        # — Plot actual, OSS backtest fit (last 3), forecast —
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(y.index, y.values, label="Actual", lw=1.2)
        if not bt.empty:
            ax.plot(bt["Date"], bt["Pred"], "--", label=f"OSS ({OSS_MONTHS}m)", lw=1.0)
        if not fc_ser.empty:
            ax.plot(fc_ser.index, fc_ser.values, ":", label=f"Forecast +{HORIZON}", lw=1.0)
        ax.set_title(f"{cty} – {reg} – {metr}")
        ax.set_ylabel(metr); ax.grid(alpha=.3); ax.legend()
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
        fig.tight_layout()
        fig.savefig(PLOT_DIR/f"{_safe(cty)}_{_safe(reg)}_{_safe(metr)}.png", dpi=120)
        plt.close(fig)

    # ── SAVE ARTEFACTS ────────────────────────────────────────────────
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

    # ── PERF & TL;DR (on OSS only) ────────────────────────────────────
    if not back_df.empty:
        perf = (
            back_df.groupby(["Country","Region","Metric"], as_index=False)
                   .apply(lambda d: pd.Series({
                       "MAE" : (d["Actual"]-d["Pred"]).abs().mean(),
                       "MAPE": (np.abs(d["Actual"]-d["Pred"]) /
                                d["Actual"].replace(0, np.nan)).mean()*100.0
                   }))
                   .reset_index(drop=True)
        )
        perf.to_excel(RUN_DIR/"series_scores.xlsx", index=False)
    else:
        perf = pd.DataFrame()

    tbl = Table(title=f"Model 01 TL;DR (OSS last {OSS_MONTHS} months)", pad_edge=False)
    tbl.add_column("Metric")
    tbl.add_column("Med MAE", justify="right")
    tbl.add_column("Med MAPE%", justify="right")
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

if __name__ == "__main__":
    main()
