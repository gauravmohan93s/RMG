# -*- coding: utf-8 -*-
from __future__ import annotations
import logging, warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table   import Table

from ..config import OUT_DIR, RANDOM_SEED
from ..loader import load_data

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)
console = Console()

EVAL_DIR = OUT_DIR/"Evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=EVAL_DIR/"evaluator.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ------------------------- helpers -------------------------
def _to_month_start(s: pd.Series) -> pd.Series:
    """
    Robustly coerce any datetime-like series to month-start timestamps.
    Works across pandas versions (no explicit 'MS' token).
    """
    s = pd.to_datetime(s, errors="coerce")
    # PeriodIndex → Timestamp at start of period
    return s.dt.to_period("M").dt.to_timestamp()

def _smape(a: np.ndarray, f: np.ndarray) -> float:
    den = np.abs(a) + np.abs(f)
    s = 200.0 * np.abs(a - f) / np.where(den==0, np.nan, den)
    return float(np.nanmean(s))

def _long_history() -> pd.DataFrame:
    df, _ = load_data()
    long = (
        df.melt(id_vars=["Country","Region","Date"],
                value_vars=[c for c in ["Received","Submitted"] if c in df.columns],
                var_name="Metric", value_name="Value")
          .sort_values(["Country","Region","Metric","Date"])
    )
    # normalize to month-start for safety
    long["Date"] = _to_month_start(long["Date"])
    return long

def _last_k_window(hist_long: pd.DataFrame, k: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    last_dt = hist_long["Date"].max()
    start = (last_dt - pd.DateOffset(months=k-1)).replace(day=1)
    end   = last_dt
    return start, end

# Human-friendly names → model folders (including Ensemble)
MODEL_DIRS: Dict[str, Path] = {
    "LastYear_MoM":               OUT_DIR/"Models"/"Model01_LastYear",
    "YoY_MoM_with_SeasonalPrior": OUT_DIR/"Models"/"Model02_MoM_YoY",
    "Rolling_MoM":                OUT_DIR/"Models"/"Model03_RollMoM",
    "AvgMonthly_MoM":             OUT_DIR/"Models"/"Model04_AvgMoM",
    "Prophet_Enhanced":           OUT_DIR/"Models"/"Model05_Prophet_Fast",
    "SARIMA_auto":                OUT_DIR/"Models"/"Model06_SARIMA",
    "XGBoost_Lags":               OUT_DIR/"Models"/"Model07_XGBoost",
    "LSTM_Univariate":            OUT_DIR/"Models"/"Model08_LSTM",
    "Ensemble_Weighted_AntiFlat": OUT_DIR/"Models"/"Model09_Ensemble",
}

HOLDOUT_MONTHS    = 3
MIN_SERIES_LEN    = 24
MIN_HOLDOUT_ROWS  = 2
FALLBACK_TAIL_M   = 12  # allow a longer overlap check if strict window is too short

def _load_backtests() -> Dict[str, pd.DataFrame]:
    bt: Dict[str, pd.DataFrame] = {}
    for model_name, dpath in MODEL_DIRS.items():
        f = dpath/"backtest.csv"
        if not f.exists():
            logging.warning("Backtest not found for %s at %s", model_name, f)
            bt[model_name] = pd.DataFrame()
            continue
        try:
            df = pd.read_csv(f, parse_dates=["Date"])  # ISO-safe
            need = {"Country","Region","Metric","Date","Actual","Pred"}
            if not need.issubset(df.columns):
                logging.error("%s missing columns: %s", model_name, need - set(df.columns))
                df = pd.DataFrame()
            else:
                # normalize to month-start to match history
                df["Date"] = _to_month_start(df["Date"])
                # drop dupes just in case
                df = (df
                      .drop_duplicates(subset=["Country","Region","Metric","Date"], keep="last")
                      .sort_values(["Country","Region","Metric","Date"]))
            bt[model_name] = df
        except Exception as e:
            logging.exception("Failed reading %s: %s", f, e)
            bt[model_name] = pd.DataFrame()
    return bt

# --------------------------- main ---------------------------
def main() -> None:
    # 1) Load normalized history and backtests
    hist_long = _long_history()
    bt_all    = _load_backtests()

    if hist_long.empty:
        console.print("[red]❌ No history from loader; aborting evaluator.[/]")
        raise SystemExit(1)

    # Establish holdout window on normalized dates
    h_start, h_end = _last_k_window(hist_long, HOLDOUT_MONTHS)
    logging.info("Holdout window (for selection) = %s → %s", h_start.date(), h_end.date())
    # Universe: series with enough history (both metrics evaluated)
    universe = (
        hist_long.groupby(["Country","Region","Metric"], as_index=False)
                 .agg(n=("Date","count"))
    )
    universe = universe[universe["n"] >= MIN_SERIES_LEN].copy()

    # 2) Scores (strict window; fallback to last 12m overlap if strict < 2)
    rows_scores: List[dict] = []
    drop_notes:  List[dict] = []

    for _, srow in universe.iterrows():
        c, r, metr = srow["Country"], srow["Region"], srow["Metric"]

        actual_strict = (
            hist_long.query("Country==@c & Region==@r & Metric==@metr & Date>=@h_start & Date<=@h_end")
                     .loc[:, ["Date","Value"]]
                     .rename(columns={"Value":"Actual"})
        )

        # fallback window: last 12 months of history for this series
        series_hist = hist_long.query("Country==@c & Region==@r & Metric==@metr").copy()
        if series_hist.empty:
            continue
        tail_start = (series_hist["Date"].max() - pd.DateOffset(months=FALLBACK_TAIL_M-1)).replace(day=1)
        actual_fallback = (
            series_hist[series_hist["Date"] >= tail_start]
            .loc[:, ["Date","Value"]].rename(columns={"Value":"Actual"})
        )

        for mdl_name, bt in bt_all.items():
            if bt is None or bt.empty:
                drop_notes.append({"Country": c, "Region": r, "Metric": metr,
                                   "Model": mdl_name, "Reason": "No backtest"})
                continue

            bt_slice = bt.query("Country==@c & Region==@r & Metric==@metr").loc[:,["Date","Pred"]]
            if bt_slice.empty:
                drop_notes.append({"Country": c, "Region": r, "Metric": metr,
                                   "Model": mdl_name, "Reason": "No series rows in backtest"})
                continue

            # strict overlap
            merged = actual_strict.merge(
                bt_slice[(bt_slice["Date"]>=h_start) & (bt_slice["Date"]<=h_end)],
                on="Date", how="inner"
            )
            # if strict too short, try a 12m tail overlap
            if len(merged) < MIN_HOLDOUT_ROWS:
                merged = actual_fallback.merge(
                    bt_slice[bt_slice["Date"]>=tail_start],
                    on="Date", how="inner"
                )

            if len(merged) < MIN_HOLDOUT_ROWS:
                drop_notes.append({"Country": c, "Region": r, "Metric": metr,
                                   "Model": mdl_name,
                                   "Reason": "Insufficient overlap (strict<2 and fallback<2)"})
                continue

            a = merged["Actual"].to_numpy(dtype=float)
            f = merged["Pred"].to_numpy(dtype=float)
            mae   = float(np.mean(np.abs(a - f)))
            smape = _smape(a, f)
            rows_scores.append({
                "Country": c, "Region": r, "Metric": metr, "Model": mdl_name,
                "Holdout_N": int(len(merged)),
                "MAE": mae,
                "sMAPE": smape,
            })

    scores_df = pd.DataFrame(rows_scores)
    scores_df.to_csv(EVAL_DIR/"holdout_series_scores_all.csv", index=False)

    if drop_notes:
        pd.DataFrame(drop_notes).to_csv(EVAL_DIR/"_selection_notes_all.csv", index=False)

    if scores_df.empty:
        console.print("[red]❌ No valid holdout scores after date normalization; see _selection_notes_all.csv[/]")
        raise SystemExit(1)

    # Write out split received & submitted tables for compatibility
    scores_df[scores_df["Metric"] == "Received"].to_csv(EVAL_DIR/"holdout_series_scores_received.csv", index=False)
    scores_df[scores_df["Metric"] == "Received"].to_csv(EVAL_DIR/"series_scores_received.csv", index=False)
    scores_df[scores_df["Metric"] == "Submitted"].to_csv(EVAL_DIR/"series_scores_submitted.csv", index=False)

    # 3) Winners per (Country, Region) calculated independently per metric
    winners_rec = (
        scores_df[scores_df["Metric"] == "Received"]
                 .sort_values(["Country","Region","MAE","sMAPE","Model"])
                 .groupby(["Country","Region"], as_index=False)
                 .first()[["Country","Region","Model"]]
                 .rename(columns={"Model":"Best_Model_Received"})
    )
    winners_sub = (
        scores_df[scores_df["Metric"] == "Submitted"]
                 .sort_values(["Country","Region","MAE","sMAPE","Model"])
                 .groupby(["Country","Region"], as_index=False)
                 .first()[["Country","Region","Model"]]
                 .rename(columns={"Model":"Best_Model_Submitted"})
    )
    # Outer join so we have columns Best_Model_Received and Best_Model_Submitted
    winners = pd.merge(winners_rec, winners_sub, on=["Country","Region"], how="outer")
    winners.to_csv(EVAL_DIR/"series_selection.csv", index=False)

    # TL;DR on Received (for backward compatible table printing)
    summary_rec = (
        scores_df[scores_df["Metric"] == "Received"]
        .groupby("Model", as_index=False)
        .agg(
            Series_Won=("MAE", "count"),
            Med_MAE=("MAE","median"),
            Med_sMAPE=("sMAPE","median")
        )
        .sort_values(["Med_MAE","Med_sMAPE","Model"])
    )

    tbl = Table(title="Evaluator TL;DR — per-series (Received)", pad_edge=False)
    tbl.add_column("Model"); tbl.add_column("Series Won", justify="right")
    tbl.add_column("Median MAE", justify="right"); tbl.add_column("Median sMAPE", justify="right")
    for _, row in summary_rec.iterrows():
        tbl.add_row(row.Model, f"{int(row.Series_Won)}",
                    f"{row.Med_MAE:,.2f}", f"{row.Med_sMAPE:.2f}%")
    console.print(tbl)
    (EVAL_DIR/"tldr_series_received.txt").write_text(tbl.__str__())
    logging.info("Evaluator completed series-level selection independently on Received and Submitted metrics.")
    console.print(f"[bold green]✅ Series selection complete → {EVAL_DIR/'series_selection.csv'}[/]")

if __name__ == "__main__":
    main()
