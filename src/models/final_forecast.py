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
from ..reconciliation import add_country_totals
from ..intake_calendar import build_intake_summary
from ..tracking import log_forecast_run, reconcile_history_with_actuals

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
    "Global_LightGBM":              BASE/"Model10_Global_LGBM",
    "Croston_TSB_Intermittent":     BASE/"Model11_Croston_TSB",
    "Ensemble_Weighted_AntiFlat":   BASE/"Model09_Ensemble",
}

from ..policy_matrix import apply_policy_scenarios, get_policy_matrix

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
    if "Best_Model_Submitted" not in sel.columns:
        sel["Best_Model_Submitted"] = sel["Best_Model_Received"]
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

def _enforce_pipeline_inventory_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Logically mature funnel constraint:
    1. Acknowledges consultant lead time: In peak submission deadline months
       (April-June for Fall, October-November for Spring), Submitted can legitimately
       exceed Received in that specific month due to accumulated backlog.
    2. Bounds monthly spikes by the rolling 3-month receipt inventory:
       Submitted_t <= 1.40 * max(Received_t, Rolling_3M_Avg(Received))
    3. Enforces 12-month annual conservation invariant:
       Sum_12m(Submitted) <= Sum_12m(Received)
       If cumulative 12m Submitted exceeds Received, scales submissions down proportionally.
    """
    if df.empty:
        return df

    rec = df[df["Metric"] == "Received"].copy()
    sub = df[df["Metric"] == "Submitted"].copy()

    if rec.empty or sub.empty:
        return df

    merge_keys = ["Country", "Region", "Date"]
    val_cols = [c for c in ["Forecast", "Forecast_Lo", "Forecast_Hi"] if c in rec.columns and c in sub.columns]

    # Sort both by date
    rec = rec.sort_values(merge_keys).reset_index(drop=True)
    sub = sub.sort_values(merge_keys).reset_index(drop=True)

    # Compute rolling 3-month receipt inventory baseline per series
    rec["_rec_roll3m"] = rec.groupby(["Country", "Region"])["Forecast"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    rec["_rec_cap"] = np.maximum(rec["Forecast"], rec["_rec_roll3m"]) * 1.40

    merged = sub.merge(
        rec[merge_keys + val_cols + ["_rec_cap"]],
        on=merge_keys,
        suffixes=("", "_rec")
    )

    # 1. Monthly rolling inventory cap: allow surge up to 1.4x rolling receipts
    spike_mask = merged["Forecast"] > merged["_rec_cap"]
    spike_count = int(spike_mask.sum())
    if spike_count > 0:
        logging.info("Pipeline inventory funnel: tempered %d runaway submission spikes beyond 1.4x rolling receipts.", spike_count)
        merged.loc[spike_mask, "Forecast"] = merged.loc[spike_mask, "_rec_cap"]

    # 2. Cumulative 12-month conservation: Sum_12m(Submitted) <= Sum_12m(Received)
    scaled_sub_list = []
    for (c, r), g in merged.groupby(["Country", "Region"]):
        tot_rec = g["Forecast_rec"].sum()
        tot_sub = g["Forecast"].sum()
        if tot_sub > tot_rec and tot_sub > 0:
            scale_factor = tot_rec / tot_sub
            g["Forecast"] = g["Forecast"] * scale_factor
            if "Forecast_Hi" in g.columns:
                g["Forecast_Hi"] = g["Forecast_Hi"] * scale_factor
            if "Forecast_Lo" in g.columns:
                g["Forecast_Lo"] = g["Forecast_Lo"] * scale_factor
            logging.info("Annual pipeline conservation: scaled submissions for %s / %s by %.3f (total sub %d -> %d)",
                         c, r, scale_factor, int(tot_sub), int(tot_rec))
        scaled_sub_list.append(g)

    merged_clean = pd.concat(scaled_sub_list, ignore_index=True)
    clean_sub = merged_clean[sub.columns]
    other = df[~df["Metric"].isin(["Received", "Submitted"])]
    out = pd.concat([rec.drop(columns=["_rec_roll3m", "_rec_cap"], errors="ignore"), clean_sub, other], ignore_index=True)
    return out.sort_values(["Country", "Region", "Metric", "Date"]).reset_index(drop=True)

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
        
        out_rows = []
        for metr in ["Received", "Submitted"]:
            mdl_col = "Best_Model_Received" if metr == "Received" else "Best_Model_Submitted"
            mdl = row[mdl_col]
            
            dfm = forecasts.get(mdl)
            if dfm is None or dfm.empty:
                logging.warning("Selected model %s has no forecast slice; skip %s/%s for metric %s", mdl, cty, reg, metr)
                continue
                
            g = dfm[(dfm.Country==cty) & (dfm.Region==reg) & (dfm.Metric==metr)].copy()
            if g.empty:
                logging.warning("Model %s no rows for %s/%s/%s", mdl, cty, reg, metr)
                continue
                
            g["Chosen_Model"] = mdl
            g["Date"] = _normalize_month_start(g["Date"])
            g = g.sort_values("Date")
            
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

        if out_rows:
            out_df = pd.concat(out_rows, ignore_index=True)
            assembled.append(out_df)

    if not assembled:
        console.print("[red]❌ No per-series assembled rows — check series_selection.csv vs model forecasts.[/]")
        return

    final_fc = (pd.concat(assembled, ignore_index=True)
                  .sort_values(["Country","Region","Metric","Date"]))

    # Enforce cumulative pipeline inventory funnel (allows April-June/Nov surges, caps annual conservation)
    final_fc = _enforce_pipeline_inventory_funnel(final_fc)

    # Apply Visa Policy & Regulatory Scenarios (Headwinds / Tailwinds)
    final_fc = apply_policy_scenarios(final_fc)
    policy_matrix = get_policy_matrix()

    # Hierarchical roll-up: generate country totals
    all_with_totals = add_country_totals(final_fc)
    country_totals = all_with_totals[all_with_totals["Region"] == "_TOTAL"].copy()

    # Domain Intake Summary: Fall, Spring, Summer with lead time and YoY comparisons
    intake_summary = build_intake_summary(final_fc, hist)

    # Forecast Governance & Audit Store: Log current run and reconcile with realized actuals
    try:
        log_forecast_run(final_fc, sel)
        audit_history, governance_scorecard = reconcile_history_with_actuals(hist)
    except Exception as e:
        logging.exception("Forecast governance tracking error: %s", e)
        audit_history, governance_scorecard = pd.DataFrame(), pd.DataFrame()

    # ── CSV (locale-proof columns included) ───────────────────────────
    final_fc.to_csv(FORECAST_DIR/"final_forecast.csv", index=False)
    if not intake_summary.empty:
        intake_summary.to_csv(FORECAST_DIR/"intake_summary.csv", index=False)

    # ── XLSX (explicit formats & widths) ──────────────────────────────
    try:
        with pd.ExcelWriter(FORECAST_DIR/"final_forecast.xlsx", engine="xlsxwriter", datetime_format="yyyy-mm-dd") as w:
            # 1. Forecast sheet
            final_fc.to_excel(w, sheet_name="Forecast_12m", index=False)
            ws = w.sheets["Forecast_12m"]
            cols = list(final_fc.columns)
            def col_idx(name): return cols.index(name) if name in cols else None
            for name, width in [
                ("Date", 12), ("Date_ISO", 12), ("YearMonth", 9), ("Month_Name", 9),
                ("Forecast", 14), ("Forecast_Lo", 14), ("Forecast_Hi", 14),
                ("Policy_Multiplier", 15), ("Regulatory_Risk", 18),
                ("Forecast_Policy_Adjusted", 20), ("Forecast_Conservative", 18), ("Forecast_Aggressive", 18),
            ]:
                ci = col_idx(name)
                if ci is not None: ws.set_column(ci, ci, width)
            ws.freeze_panes(1, 0)

            # 2. Intake Summary sheet (Fall, Spring, Summer)
            if not intake_summary.empty:
                intake_summary.to_excel(w, sheet_name="Intake_Summary", index=False)
                ws_int = w.sheets["Intake_Summary"]
                cols_int = list(intake_summary.columns)
                def col_idx_int(name): return cols_int.index(name) if name in cols_int else None
                for name, width in [
                    ("Country", 16), ("Region", 18), ("Metric", 12),
                    ("Intake_Cycle", 14), ("Forecast", 14), ("Forecast_Lo", 14),
                    ("Forecast_Hi", 14), ("Prior_Year_Actual", 16), ("YoY_Growth_%", 14)
                ]:
                    ci = col_idx_int(name)
                    if ci is not None: ws_int.set_column(ci, ci, width)
                ws_int.freeze_panes(1, 0)

            # 3. Policy Scenarios sheet (Regulatory Headwinds & Tailwinds Matrix)
            if not policy_matrix.empty:
                policy_matrix.to_excel(w, sheet_name="Policy_Scenarios", index=False)
                ws_pol = w.sheets["Policy_Scenarios"]
                ws_pol.freeze_panes(1, 0)
                ws_pol.set_column(0, 0, 18)
                ws_pol.set_column(1, 1, 16)
                ws_pol.set_column(2, 2, 22)
                ws_pol.set_column(3, 3, 70)

            # 4. Country Totals sheet
            if not country_totals.empty:
                country_totals.to_excel(w, sheet_name="Country_Totals", index=False)
                ws_tot = w.sheets["Country_Totals"]
                cols_tot = list(country_totals.columns)
                def col_idx_tot(name): return cols_tot.index(name) if name in cols_tot else None
                for name, width in [
                    ("Date", 12), ("Date_ISO", 12), ("YearMonth", 9), ("Month_Name", 9),
                    ("Forecast", 14), ("Forecast_Lo", 14), ("Forecast_Hi", 14),
                    ("Forecast_Policy_Adjusted", 20), ("Forecast_Conservative", 18), ("Forecast_Aggressive", 18),
                ]:
                    ci = col_idx_tot(name)
                    if ci is not None: ws_tot.set_column(ci, ci, width)
                ws_tot.freeze_panes(1, 0)

            # 5. Model Governance & Reliability Scorecard
            if not governance_scorecard.empty:
                governance_scorecard.to_excel(w, sheet_name="Model_Governance", index=False)
                w.sheets["Model_Governance"].freeze_panes(1, 0)

            # 6. Full Prediction Audit Trail (what model, why, low, best, high, actuals)
            if not audit_history.empty:
                audit_history.to_excel(w, sheet_name="Prediction_Audit", index=False)
                w.sheets["Prediction_Audit"].freeze_panes(1, 0)

            # 7. History sheet
            hist2 = hist.copy()
            hist2 = _add_date_helpers(hist2.rename(columns={"Value":"Actual"}))
            hist2.to_excel(w, sheet_name="History", index=False)
            w.sheets["History"].freeze_panes(1, 0)

            # 8. Selection sheet
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
