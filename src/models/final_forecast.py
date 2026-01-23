# -*- coding: utf-8 -*-
from __future__ import annotations
import logging, warnings, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console

from ..config import OUT_DIR, RANDOM_SEED, HORIZON
from ..loader import load_data

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)
console = Console()

# ─────────────────────────────────────────────────────────────────────
BASE         = OUT_DIR / "Models"
EVAL_DIR     = OUT_DIR / "Evaluation"
FORECAST_DIR = OUT_DIR / "Final_Forecast"
PLOT_DIR     = FORECAST_DIR / "plots"
EDA_DIR      = OUT_DIR / "EDA"

FORECAST_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=FORECAST_DIR/"forecast.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Human-friendly model names → folders
MODEL_DIRS: Dict[str, Path] = {
    "LastYear_MoM":                 BASE/"Model01_LastYear",
    "YoY_MoM_with_SeasonalPrior":   BASE/"Model02_MoM_YoY",
    "Rolling_MoM":                  BASE/"Model03_RollMoM",
    "AvgMonthly_MoM":               BASE/"Model04_AvgMoM",
    "Prophet_Enhanced":             BASE/"Model05_Prophet_Fast",
    "SARIMA_auto":                  BASE/"Model06_SARIMA",
    "XGBoost_Lags":                 BASE/"Model07_XGBoost",
    "LSTM_Univariate":              BASE/"Model08_LSTM",
    "Ensemble_Weighted_AntiFlat":   BASE/"Model09_Ensemble",
}

def _safe(s: str) -> str:
    return re.sub(r'[\/:*?"<>|]', "_", str(s))

# ── DATE HELPERS ─────────────────────────────────────────────────────
def _normalize_month_start(s: pd.Series) -> pd.Series:
    """
    Ensure monthly first-of-month timestamps, naive, 00:00:00.
    This prevents locale flips like 2025-10-01 → 10-01-2025 (MM-DD-YYYY).
    """
    s = pd.to_datetime(s, errors="coerce")
    s = s.dt.to_period("M").dt.to_timestamp(how="start")
    return s.dt.normalize()

def _add_date_helpers(df: pd.DataFrame) -> pd.DataFrame:
    """Add locale-proof helper columns for downstream tools (Excel/CSV)."""
    out = df.copy()
    out["Date"]       = _normalize_month_start(out["Date"])
    out["Date_ISO"]   = out["Date"].dt.strftime("%Y-%m-%d")
    out["YearMonth"]  = out["Date"].dt.to_period("M").astype(str)   # YYYY-MM
    out["Year"]       = out["Date"].dt.year.astype(int)
    out["Month"]      = out["Date"].dt.month.astype(int)
    out["Month_Name"] = out["Date"].dt.strftime("%b")               # Jan, Feb, ...
    return out

# ── loaders ──────────────────────────────────────────────────────────
def _load_series_selection() -> pd.DataFrame:
    p = EVAL_DIR / "series_selection.csv"
    if not p.exists():
        raise FileNotFoundError(f"Per-series selection not found: {p}")
    sel = pd.read_csv(p)
    need = {"Country","Region","Best_Model_Received"}
    if not need.issubset(sel.columns):
        raise ValueError(f"{p.name} must have columns {need}")
    return sel

def _load_forecasts_all() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for mname, d in MODEL_DIRS.items():
        f = d / "forecast_h12.csv"
        if not f.exists():
            logging.warning("Missing forecast for %s at %s", mname, f)
            continue
        try:
            dfc = pd.read_csv(f, parse_dates=["Date"])
            need = {"Country","Region","Metric","Date","Forecast"}
            if not need.issubset(dfc.columns):
                logging.error("%s forecast missing columns %s", mname, need - set(dfc.columns))
                continue
            dfc["Model"] = mname
            # normalize right here to be safe
            dfc["Date"] = _normalize_month_start(dfc["Date"])
            out[mname] = dfc
        except Exception as e:
            logging.exception("Reading %s failed: %s", f, e)
    if not out:
        raise RuntimeError("No forecasts found in any model folders.")
    return out

def _load_backtests_all() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for mname, d in MODEL_DIRS.items():
        f = d / "backtest.csv"
        if not f.exists():
            out[mname] = pd.DataFrame()
            continue
        try:
            dfb = pd.read_csv(f, parse_dates=["Date"])
            dfb["Date"] = _normalize_month_start(dfb["Date"])
            out[mname] = dfb
        except Exception:
            out[mname] = pd.DataFrame()
    return out

# ── EDA bands ───────────────────────────────────────────────────────
def _mom_bands() -> Optional[pd.DataFrame]:
    p = EDA_DIR / "mom_bands.parquet"
    try:
        df = pd.read_parquet(p)
        req = {"Country","Region","Metric","Month","p05","p95","rw_mean"}
        return df if req.issubset(df.columns) else None
    except Exception:
        return None

MOM_BANDS = _mom_bands()

def _apply_bands_trajectory(country: str, region: str, metric: str,
                            history_last: float,
                            fc: pd.DataFrame) -> Tuple[np.ndarray,np.ndarray]:
    """Lower/upper trajectories via month-wise p05/p95 step bands."""
    lo = np.zeros(len(fc), dtype=float)
    hi = np.zeros(len(fc), dtype=float)
    level_lo = float(history_last)
    level_hi = float(history_last)

    for i, dt in enumerate(fc["Date"]):
        if MOM_BANDS is not None:
            sub = MOM_BANDS[(MOM_BANDS.Country==country) &
                            (MOM_BANDS.Region==region) &
                            (MOM_BANDS.Metric==metric) &
                            (MOM_BANDS.Month==int(pd.Timestamp(dt).month))]
            if not sub.empty:
                lo_step = max(0.8, 1.0 + float(sub.p05.iloc[0]))
                hi_step = min(1.2, 1.0 + float(sub.p95.iloc[0]))
            else:
                lo_step, hi_step = 0.8, 1.2
        else:
            lo_step, hi_step = 0.8, 1.2

        level_lo = max(0.0, level_lo * lo_step)
        level_hi = max(0.0, level_hi * hi_step)
        lo[i] = level_lo
        hi[i] = level_hi

    return lo, hi

def _mape_band(country: str, region: str, metric: str,
               model: str, bt_all: Dict[str,pd.DataFrame]) -> float:
    """Fallback band width as median APE from model's backtest; clamp to 5–50%."""
    bt = bt_all.get(model)
    if bt is None or bt.empty: return 0.20
    s = bt[(bt.Country==country) & (bt.Region==region) & (bt.Metric==metric)].copy()
    if s.empty: return 0.20
    denom = s["Actual"].replace(0, np.nan).abs()
    ape = (s["Actual"] - s["Pred"]).abs() / denom
    mape = float(np.nanmedian(ape)) if np.isfinite(ape).any() else 0.20
    if not np.isfinite(mape) or mape<=0: mape = 0.20
    return float(np.clip(mape, 0.05, 0.50))

# ── plotting helpers (titles, labels, annotations) ───────────────────
def _annotate_points(ax, x, y, n_tail: int = 3, fmt: str = "{:,.0f}", color=None):
    """Annotate last n actual points."""
    if len(x) == 0: return
    sel_idx = np.arange(max(0, len(x)-n_tail), len(x))
    for i in sel_idx:
        ax.annotate(fmt.format(float(y[i])),
                    (x[i], y[i]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=7,
                    color=color)

def _annotate_first_last(ax, x, y, fmt="{:,.0f}", color=None):
    if len(x) == 0: return
    idxs = [0, len(x)-1] if len(x) > 1 else [0]
    for i in idxs:
        ax.annotate(fmt.format(float(y[i])),
                    (x[i], y[i]),
                    textcoords="offset points",
                    xytext=(0, 6 if i==0 else -10),
                    ha="center",
                    fontsize=7,
                    color=color)

# ─────────────────────────────────────────────────────────────────────
def main():
    console.print("[bold]🔮 Assembling final 12-month per-series forecast (Received-based selection)…[/]\n")

    sel        = _load_series_selection()         # Country, Region → Best_Model_Received (human names)
    forecasts  = _load_forecasts_all()            # model → forecast_h12.csv
    backtests  = _load_backtests_all()

    # history (for last actual & plotting)
    raw,_ = load_data()
    hist = (raw.melt(
        id_vars=["Country","Region","Date"],
        value_vars=[c for c in ["Received","Submitted"] if c in raw.columns],
        var_name="Metric", value_name="Value")
        .sort_values(["Country","Region","Metric","Date"])
    )
    hist["Date"] = _normalize_month_start(hist["Date"])

    last_actual = (hist.dropna(subset=["Value"])
                        .groupby(["Country","Region","Metric"], as_index=False)
                        .last())  # last row per series

    assembled: List[pd.DataFrame] = []

    for _, row in sel.iterrows():
        cty  = row["Country"]
        reg  = row["Region"]
        mdl  = row["Best_Model_Received"]  # already a human-friendly name

        dfm = forecasts.get(mdl)
        if dfm is None or dfm.empty:
            logging.warning("Selected model %s has no forecast slice; skip %s/%s", mdl, cty, reg)
            continue

        # Use SAME selected model for both metrics in the region
        sub = dfm[(dfm.Country==cty) & (dfm.Region==reg)].copy()
        if sub.empty:
            logging.warning("Model %s no rows for %s/%s", mdl, cty, reg)
            continue
        sub["Chosen_Model"] = mdl
        sub["Date"] = _normalize_month_start(sub["Date"])

        # Add uncertainty bands
        out_rows = []
        for metr, g in sub.groupby("Metric", sort=False):
            g = g.sort_values("Date").copy()
            try:
                hist_last = float(last_actual[(last_actual.Country==cty) &
                                              (last_actual.Region==reg) &
                                              (last_actual.Metric==metr)]["Value"].iloc[-1])
            except Exception:
                hist_last = float("nan")
            if not np.isfinite(hist_last):
                hist_last = float(g["Forecast"].iloc[0])

            if MOM_BANDS is not None:
                lo_traj, hi_traj = _apply_bands_trajectory(cty, reg, metr, hist_last, g)
            else:
                p = _mape_band(cty, reg, metr, mdl, backtests)
                lo_traj = g["Forecast"].to_numpy(dtype=float) * (1.0 - p)
                hi_traj = g["Forecast"].to_numpy(dtype=float) * (1.0 + p)

            g["Forecast_Lo"] = np.maximum(0.0, lo_traj)
            g["Forecast_Hi"] = np.maximum(0.0, hi_traj)

            # Add date helper columns on this slice
            g = _add_date_helpers(g)
            out_rows.append(g)

        out_df = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
        if not out_df.empty:
            assembled.append(out_df)

    if not assembled:
        console.print("[red]❌ No per-series assembled rows — check series_selection.csv vs model forecasts.[/]")
        return

    final_fc = (pd.concat(assembled, ignore_index=True)
                  .sort_values(["Country","Region","Metric","Date"]))

    # ── CSV (locale-proof columns included) ───────────────────────────
    final_fc.to_csv(FORECAST_DIR/"final_forecast.csv", index=False)

    # ── XLSX (explicit formats & widths) ──────────────────────────────
    try:
        with pd.ExcelWriter(FORECAST_DIR/"final_forecast.xlsx", engine="xlsxwriter", datetime_format="yyyy-mm-dd") as w:
            # Forecast sheet
            final_fc.to_excel(w, sheet_name="Forecast_12m", index=False)
            ws = w.sheets["Forecast_12m"]
            cols = list(final_fc.columns)
            def col_idx(name): return cols.index(name) if name in cols else None
            # Set widths if columns exist
            for name, width in [
                ("Date", 12),
                ("Date_ISO", 12),
                ("YearMonth", 9),
                ("Month_Name", 9),
                ("Forecast", 14),
                ("Forecast_Lo", 14),
                ("Forecast_Hi", 14),
            ]:
                ci = col_idx(name)
                if ci is not None:
                    ws.set_column(ci, ci, width)
            ws.freeze_panes(1, 0)

            # History sheet
            hist2 = hist.copy()
            hist2 = _add_date_helpers(hist2.rename(columns={"Value":"Actual"}))
            hist2.to_excel(w, sheet_name="History", index=False)
            w.sheets["History"].freeze_panes(1, 0)

            # Selection sheet
            sel.to_excel(w, sheet_name="Series_Selection", index=False)
            w.sheets["Series_Selection"].freeze_panes(1, 0)
    except Exception as e:
        logging.exception("Writing XLSX failed: %s", e)

    # ── Plots: fully labeled with data annotations ────────────────────
    last_hist_date = hist["Date"].max()

    for (c,r,m), g in hist.groupby(["Country","Region","Metric"], sort=False):
        f = final_fc[(final_fc.Country==c)&(final_fc.Region==r)&(final_fc.Metric==m)].copy()
        if f.empty:
            continue

        # composed title & subtitle
        chosen = f["Chosen_Model"].iloc[0]
        title = f"{c} — {r} — {m}"
        subtitle = f"Model: {chosen} · Horizon: {HORIZON} months"

        fig, ax = plt.subplots(figsize=(7.5, 4.0))

        # Actuals
        ax.plot(g["Date"], g["Value"], label="Actual", lw=1.4)
        _annotate_points(ax, g["Date"].to_numpy(), g["Value"].to_numpy(), n_tail=3, fmt="{:,.0f}")

        # Forecast + band
        ax.plot(f["Date"], f["Forecast"], ":", label="Forecast", lw=1.6)
        _annotate_first_last(ax, f["Date"].to_numpy(), f["Forecast"].to_numpy(), fmt="{:,.0f}")
        if {"Forecast_Lo","Forecast_Hi"}.issubset(f.columns):
            ax.fill_between(f["Date"], f["Forecast_Lo"], f["Forecast_Hi"],
                            alpha=0.16, label="Forecast band")

        # Vertical split at forecast start
        if pd.notna(last_hist_date):
            ax.axvline(last_hist_date, color="black", ls=":", lw=0.9)
            ax.text(last_hist_date, ax.get_ylim()[1]*0.98, "Forecast start",
                    va="top", ha="right", fontsize=8)

        ax.set_title(title, fontsize=12, pad=6)
        ax.set_xlabel("Date")
        ax.set_ylabel(m)
        ax.legend(loc="upper left", fontsize=8, title=subtitle)
        ax.grid(alpha=0.3)
        for t in ax.get_xticklabels(): t.set_rotation(45)
        fig.tight_layout()
        fig.savefig(PLOT_DIR/f"{_safe(c)}_{_safe(r)}_{_safe(m)}.png", dpi=140)
        plt.close(fig)

    console.print(f"[bold green]✅ Final per-series forecast, bands, and labeled plots → {FORECAST_DIR}[/]")

if __name__ == "__main__":
    main()
