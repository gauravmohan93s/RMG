# -*- coding: utf-8 -*-
"""
Model 11 — Intermittent Demand Engine (Croston & TSB: Teunter-Syntetos-Babai)
-----------------------------------------------------------------------------
Tailored for sparse, zero-inflated, and micro-branch time series:
• Decomposes intermittent demand into Demand Probability (d_t) and Demand Magnitude (z_t).
• Smoothly updates probability every period (TSB), preventing obsolete inventory/capacity projections.
• Poisson/Compound uncertainty bounds (5th and 95th percentiles).
• Evaluated with strict OSS backtest and dynamic 12-month production forecasts.

Artefacts → …/Models/Model11_Croston_TSB/
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

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)
console = Console()

RUN_DIR  = OUT_DIR / "Models" / "Model11_Croston_TSB"
PLOT_DIR = RUN_DIR / "plots"
RUN_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=RUN_DIR / "model11_tsb.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

OSS_MONTHS = int(os.environ.get("RMG_OSS_MONTHS", "3"))

def _safe(s: str) -> str:
    return re.sub(r'[\\/:\*?"<>|]', "_", str(s))

def _split_oss(y: pd.Series, k: int = OSS_MONTHS) -> tuple[pd.Series, pd.DatetimeIndex]:
    """Return (y_train, oss_idx) where the last k months are reserved for OSS."""
    y = y.sort_index()
    if len(y) <= k:
        return y.iloc[:0], y.index
    return y.iloc[:-k], y.index[-k:]

# ─────────────────────────────────────────────────────────────────────
# Teunter-Syntetos-Babai (TSB) Core Logic
# ─────────────────────────────────────────────────────────────────────
def fit_predict_tsb(
    history: np.ndarray,
    horizon: int = 12,
    alpha: float = 0.15,
    beta: float = 0.10
) -> Tuple[float, float, float]:
    """
    Fits TSB on historical series and returns (point_forecast, forecast_lo, forecast_hi).
    alpha: smoothing parameter for demand size (magnitude)
    beta: smoothing parameter for demand occurrence (probability)
    """
    y = np.asarray(history, dtype=float)
    y = np.maximum(0.0, y)
    n = len(y)
    if n == 0:
        return 0.0, 0.0, 0.0

    non_zeros = y[y > 0]
    if len(non_zeros) == 0:
        return 0.0, 0.0, 0.0

    # Initialization
    first_nonzero_idx = np.where(y > 0)[0][0]
    z = float(y[first_nonzero_idx])
    d = float(len(non_zeros) / n)

    # Recursive TSB updates
    for t in range(n):
        yt = y[t]
        if yt > 0:
            z = alpha * yt + (1.0 - alpha) * z
            d = beta * 1.0 + (1.0 - beta) * d
        else:
            # magnitude unchanged; probability decays
            d = (1.0 - beta) * d

    # Point forecast: expected demand per period
    point_forecast = max(0.0, float(d * z))

    # Uncertainty bounds via Poisson / zero-inflated variance
    # Var(Y) = d * Var(Z) + d * (1 - d) * E[Z]^2
    expected_var = max(0.2, point_forecast + (1.0 - d) * (point_forecast ** 2))
    std_err = np.sqrt(expected_var)

    forecast_lo = max(0.0, point_forecast - 1.645 * std_err)
    forecast_hi = max(point_forecast, point_forecast + 1.645 * std_err)

    return point_forecast, forecast_lo, forecast_hi

# ─────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    console.print("[bold cyan]▶️  Running Model 11: Intermittent Demand Engine (Croston & TSB) …[/]")
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
    summary_rows: List[dict] = []

    for (cty, reg, metr), g in tqdm(melt.groupby(["Country", "Region", "Metric"], sort=False),
                                    desc="Model11_Croston_TSB"):
        y = (
            g.set_index("Date")["Value"]
             .astype("float64")
             .asfreq("MS")
             .fillna(0.0)
             .clip(lower=0.0)
        )
        if len(y) < 12:
            logging.warning("Skip %s/%s/%s (len=%d < 12)", cty, reg, metr, len(y))
            continue

        # ── 1. Strict Out-of-Sample Backtest ──
        y_train, oss_idx = _split_oss(y, OSS_MONTHS)
        if len(y_train) >= 6 and len(oss_idx) > 0:
            hist_vals = y_train.values
            pred_point, _, _ = fit_predict_tsb(hist_vals, horizon=len(oss_idx))
            
            bt_df = pd.DataFrame({
                "Country": cty,
                "Region": reg,
                "Metric": metr,
                "Date": oss_idx,
                "Actual": y.loc[oss_idx].values,
                "Pred": np.full(len(oss_idx), pred_point, dtype=float)
            })
            bt_df["Date"] = pd.to_datetime(bt_df["Date"]).dt.to_period("M").dt.to_timestamp(how="S")
            records_back.append(bt_df)

        # ── 2. Production Forecast (12m ahead) ──
        full_vals = y.values
        fut_point, fut_lo, fut_hi = fit_predict_tsb(full_vals, horizon=HORIZON)
        
        fut_idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=HORIZON, freq="MS")
        fc_df = pd.DataFrame({
            "Country": cty,
            "Region": reg,
            "Metric": metr,
            "Date": fut_idx,
            "Forecast": np.full(HORIZON, fut_point, dtype=float),
            "Forecast_Lo": np.full(HORIZON, fut_lo, dtype=float),
            "Forecast_Hi": np.full(HORIZON, fut_hi, dtype=float),
        })
        records_fore.append(fc_df)

        # ── 3. Diagnostic Plots for Sparse Series ──
        sparsity = float(np.mean(full_vals == 0.0))
        if sparsity > 0.25:  # Only generate plots for genuinely sparse series to keep execution ultra-fast
            fig, ax = plt.subplots(figsize=(6.5, 3.2))
            ax.plot(y.index, y.values, label="Actuals", marker="o", markersize=3, lw=1.2, color="#2b5c8f")
            ax.plot(fut_idx, fc_df["Forecast"], label="TSB Demand Rate", color="#e26d5c", lw=1.5, ls="--")
            ax.fill_between(fut_idx, fc_df["Forecast_Lo"], fc_df["Forecast_Hi"], color="#e26d5c", alpha=0.15, label="90% Intermittent Band")
            ax.set_title(f"{cty} - {reg} - {metr} (Sparsity: {sparsity*100:.0f}%)", fontsize=10)
            ax.grid(alpha=0.25)
            ax.legend(loc="upper left", fontsize=7)
            fig.tight_layout()
            fig.savefig(PLOT_DIR / f"{_safe(cty)}_{_safe(reg)}_{_safe(metr)}_tsb.png", dpi=100)
            plt.close(fig)

    if not records_back:
        console.print("[red]❌ Model 11 produced no backtests[/]")
        return

    # Export deliverables
    back_all = pd.concat(records_back, ignore_index=True)
    fore_all = pd.concat(records_fore, ignore_index=True)

    back_all.to_csv(RUN_DIR / "backtest.csv", index=False)
    fore_all.to_csv(RUN_DIR / "forecast_h12.csv", index=False)

    # Median performance
    back_all["Abs_Err"] = (back_all["Actual"] - back_all["Pred"]).abs()
    back_all["Denom"] = (back_all["Actual"].abs() + back_all["Pred"].abs())
    back_all["sMAPE"] = np.where(back_all["Denom"] > 0, 200.0 * back_all["Abs_Err"] / back_all["Denom"], 0.0)

    med_rec_mae = back_all[back_all["Metric"] == "Received"]["Abs_Err"].median()
    med_rec_smape = back_all[back_all["Metric"] == "Received"]["sMAPE"].median()
    med_sub_mae = back_all[back_all["Metric"] == "Submitted"]["Abs_Err"].median()
    med_sub_smape = back_all[back_all["Metric"] == "Submitted"]["sMAPE"].median()

    tbl = Table(title="Model 11 Croston & TSB — Intermittent Engine Performance", pad_edge=False)
    tbl.add_column("Metric")
    tbl.add_column("Median MAE", justify="right")
    tbl.add_column("Median sMAPE %", justify="right")
    tbl.add_row("Received", f"{med_rec_mae:,.2f}", f"{med_rec_smape:.1f}%")
    tbl.add_row("Submitted", f"{med_sub_mae:,.2f}", f"{med_sub_smape:.1f}%")
    console.print(tbl)

    logging.info("Model 11 TSB completed successfully. Median Rec MAE: %.2f", med_rec_mae)
    console.print(f"[bold green]✅ Model 11 Croston & TSB finished — artefacts saved to {RUN_DIR}[/]")

if __name__ == "__main__":
    main()
