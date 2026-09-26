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
    "Global_LightGBM":            OUT_DIR/"Models"/"Model10_Global_LGBM",
    "Croston_TSB_Intermittent":   OUT_DIR/"Models"/"Model11_Croston_TSB",
    "Ensemble_Weighted_AntiFlat": OUT_DIR/"Models"/"Model09_Ensemble",
}

HOLDOUT_MONTHS    = 3
MIN_SERIES_LEN    = 12  # Support emerging branches with >= 12 months of data
MATURE_SERIES_LEN = 24  # Threshold for full heavy-model competition
FALLBACK_TAIL_M   = 12  # allow a longer overlap check if strict window is too short
MIN_HOLDOUT_ROWS  = 2   # minimum months of overlap required for scoring
TIER1_MODELS      = {"LastYear_MoM", "AvgMonthly_MoM", "Rolling_MoM", "Global_LightGBM", "Croston_TSB_Intermittent"}

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

        is_emerging = (srow["n"] < MATURE_SERIES_LEN)
        for mdl_name, bt in bt_all.items():
            if is_emerging and mdl_name not in TIER1_MODELS:
                # Gating: Emerging series with <24 months only compete with low-parameter baselines & Global LightGBM
                drop_notes.append({"Country": c, "Region": r, "Metric": metr,
                                   "Model": mdl_name, "Reason": f"Emerging series gating (n={srow['n']} < {MATURE_SERIES_LEN})"})
                continue

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

            # ── 3-Test Tournament Scoring: Peak, Normal, and Low Regimes ──
            hist_vals = series_hist["Value"].to_numpy(dtype=float)
            p33 = float(np.percentile(hist_vals, 33.3)) if len(hist_vals) >= 3 else float(np.min(hist_vals))
            p67 = float(np.percentile(hist_vals, 66.7)) if len(hist_vals) >= 3 else float(np.max(hist_vals))
            if p33 >= p67:
                p33 = float(np.min(hist_vals))
                p67 = float(np.max(hist_vals))

            merged["Actual"] = merged["Actual"].astype(float)
            merged["Pred"] = merged["Pred"].astype(float)
            merged["Error"] = np.abs(merged["Actual"] - merged["Pred"])

            # Overall metrics
            mae_overall = float(merged["Error"].mean())
            smape_overall = _smape(merged["Actual"].to_numpy(), merged["Pred"].to_numpy())

            # Regime splits based on volume
            peak_mask = merged["Actual"] >= p67
            low_mask  = merged["Actual"] <= p33
            norm_mask = (~peak_mask) & (~low_mask)

            # Test 1: Peak / High-Volume Regime
            if peak_mask.any():
                peak_sub = merged[peak_mask]
                mae_peak = float(peak_sub["Error"].mean())
                smape_peak = _smape(peak_sub["Actual"].to_numpy(), peak_sub["Pred"].to_numpy())
                n_peak = int(peak_mask.sum())
            else:
                mae_peak = mae_overall
                smape_peak = smape_overall
                n_peak = 0

            # Test 2: Normal / Baseline Regime
            if norm_mask.any():
                norm_sub = merged[norm_mask]
                mae_norm = float(norm_sub["Error"].mean())
                smape_norm = _smape(norm_sub["Actual"].to_numpy(), norm_sub["Pred"].to_numpy())
                n_norm = int(norm_mask.sum())
            else:
                mae_norm = mae_overall
                smape_norm = smape_overall
                n_norm = 0

            # Test 3: Low-Volume / Off-Peak Regime
            if low_mask.any():
                low_sub = merged[low_mask]
                mae_low = float(low_sub["Error"].mean())
                smape_low = _smape(low_sub["Actual"].to_numpy(), low_sub["Pred"].to_numpy())
                n_low = int(low_mask.sum())
            else:
                mae_low = mae_overall
                smape_low = smape_overall
                n_low = 0

            # Weighted Decathlon Score: 40% Peak + 35% Normal + 25% Low
            decathlon_score = 0.40 * mae_peak + 0.35 * mae_norm + 0.25 * mae_low

            rows_scores.append({
                "Country": c, "Region": r, "Metric": metr, "Model": mdl_name,
                "Holdout_N": int(len(merged)),
                "MAE": mae_overall,
                "sMAPE": smape_overall,
                "MAE_Peak": mae_peak,
                "sMAPE_Peak": smape_peak,
                "N_Peak": n_peak,
                "MAE_Normal": mae_norm,
                "sMAPE_Normal": smape_norm,
                "N_Normal": n_norm,
                "MAE_Low": mae_low,
                "sMAPE_Low": smape_low,
                "N_Low": n_low,
                "Decathlon_Score": decathlon_score,
            })

    scores_df = pd.DataFrame(rows_scores)
    scores_df.to_csv(EVAL_DIR/"holdout_series_scores_all.csv", index=False)
    scores_df.to_csv(EVAL_DIR/"tournament_3test_scores.csv", index=False)

    if drop_notes:
        pd.DataFrame(drop_notes).to_csv(EVAL_DIR/"_selection_notes_all.csv", index=False)

    if scores_df.empty:
        console.print("[red]❌ No valid holdout scores after date normalization; see _selection_notes_all.csv[/]")
        raise SystemExit(1)

    # Write out split received & submitted tables for compatibility
    scores_df[scores_df["Metric"] == "Received"].to_csv(EVAL_DIR/"holdout_series_scores_received.csv", index=False)
    scores_df[scores_df["Metric"] == "Received"].to_csv(EVAL_DIR/"series_scores_received.csv", index=False)
    scores_df[scores_df["Metric"] == "Submitted"].to_csv(EVAL_DIR/"series_scores_submitted.csv", index=False)

    # 3) Winners per (Country, Region) calculated independently via 3-Test Decathlon Score
    best_rec = (
        scores_df[scores_df["Metric"] == "Received"]
        .sort_values(["Country", "Region", "Decathlon_Score", "sMAPE", "Model"])
        .groupby(["Country", "Region"], as_index=False)
        .first()
    )
    best_rec["Selection_Reason"] = best_rec.apply(
        lambda x: f"Decathlon Winner (Score: {x['Decathlon_Score']:.1f} | Peak: {x['MAE_Peak']:.1f}, Norm: {x['MAE_Normal']:.1f}, Low: {x['MAE_Low']:.1f})",
        axis=1
    )
    rec_winners = best_rec[["Country", "Region", "Model", "Decathlon_Score", "MAE_Peak", "MAE_Normal", "MAE_Low", "Selection_Reason"]].rename(
        columns={
            "Model": "Best_Model_Received",
            "Decathlon_Score": "Decathlon_Score_Rec",
            "MAE_Peak": "Peak_MAE_Rec",
            "MAE_Normal": "Norm_MAE_Rec",
            "MAE_Low": "Low_MAE_Rec",
            "Selection_Reason": "Selection_Reason_Received"
        }
    )

    best_sub = (
        scores_df[scores_df["Metric"] == "Submitted"]
        .sort_values(["Country", "Region", "Decathlon_Score", "sMAPE", "Model"])
        .groupby(["Country", "Region"], as_index=False)
        .first()
    )
    best_sub["Selection_Reason"] = best_sub.apply(
        lambda x: f"Decathlon Winner (Score: {x['Decathlon_Score']:.1f} | Peak: {x['MAE_Peak']:.1f}, Norm: {x['MAE_Normal']:.1f}, Low: {x['MAE_Low']:.1f})",
        axis=1
    )
    sub_winners = best_sub[["Country", "Region", "Model", "Decathlon_Score", "MAE_Peak", "MAE_Normal", "MAE_Low", "Selection_Reason"]].rename(
        columns={
            "Model": "Best_Model_Submitted",
            "Decathlon_Score": "Decathlon_Score_Sub",
            "MAE_Peak": "Peak_MAE_Sub",
            "MAE_Normal": "Norm_MAE_Sub",
            "MAE_Low": "Low_MAE_Sub",
            "Selection_Reason": "Selection_Reason_Submitted"
        }
    )

    winners = pd.merge(rec_winners, sub_winners, on=["Country", "Region"], how="outer")
    winners.to_csv(EVAL_DIR/"series_selection.csv", index=False)

    # TL;DR Decathlon Leaderboard on Received
    summary_rec = (
        scores_df[scores_df["Metric"] == "Received"]
        .groupby("Model", as_index=False)
        .agg(
            Series_Won=("Decathlon_Score", "count"),
            Med_Decathlon=("Decathlon_Score", "median"),
            Med_Peak_MAE=("MAE_Peak", "median"),
            Med_Norm_MAE=("MAE_Normal", "median"),
            Med_Low_MAE=("MAE_Low", "median"),
            Med_sMAPE=("sMAPE", "median")
        )
        .sort_values(["Med_Decathlon", "Med_sMAPE", "Model"])
    )

    tbl = Table(title="🏆 Tournament Decathlon Leaderboard — 3-Test Regime (Received)", pad_edge=False)
    tbl.add_column("Model")
    tbl.add_column("Series Evaluated", justify="right")
    tbl.add_column("Decathlon Score", justify="right")
    tbl.add_column("Peak MAE", justify="right")
    tbl.add_column("Normal MAE", justify="right")
    tbl.add_column("Low MAE", justify="right")
    tbl.add_column("Median sMAPE", justify="right")
    for _, row in summary_rec.iterrows():
        tbl.add_row(
            row.Model,
            f"{int(row.Series_Won)}",
            f"{row.Med_Decathlon:,.2f}",
            f"{row.Med_Peak_MAE:,.2f}",
            f"{row.Med_Norm_MAE:,.2f}",
            f"{row.Med_Low_MAE:,.2f}",
            f"{row.Med_sMAPE:.2f}%"
        )
    console.print(tbl)
    try:
        (EVAL_DIR/"tldr_tournament_decathlon.txt").write_text(tbl.__str__(), encoding="utf-8")
    except Exception:
        pass
    logging.info("Evaluator completed 3-Test Tournament (Peak, Normal, Low) selection independently on Received and Submitted.")
    console.print(f"[bold green]✅ 3-Test Decathlon Tournament complete → {EVAL_DIR/'series_selection.csv'}[/]")

if __name__ == "__main__":
    main()
