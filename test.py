# quick_eval_check.py
import os, re
import pathlib as pl
import pandas as pd

# 0) Base Predictions folder from your pipeline
BASE = pl.Path(r"C:\Users\AdmiN\Downloads\rmg_forecast_repo\RMG Calculations\Predictions")

def _latest_run_folder(base: pl.Path) -> pl.Path:
    if not base.exists():
        raise FileNotFoundError(f"Predictions base not found: {base}")
    date_dirs = []
    pat = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for p in base.iterdir():
        if p.is_dir() and pat.match(p.name):
            date_dirs.append(p)
    if not date_dirs:
        raise FileNotFoundError(f"No YYYY-MM-DD folders under {base}")
    return sorted(date_dirs)[-1]

# Resolve OUT_DIR (env if available, else latest)
OUT_DIR = pl.Path(os.environ["RMG_OUT_DIR"]) if "RMG_OUT_DIR" in os.environ else _latest_run_folder(BASE)
print("OUT_DIR:", OUT_DIR)

notes_path = OUT_DIR / "Evaluation" / "_received_selection_notes.csv"
if not notes_path.exists():
    raise FileNotFoundError(f"Notes file not found: {notes_path}")

notes = pd.read_csv(notes_path)
print("\n=== Top 'Reason' counts in _received_selection_notes.csv ===")
print(notes["Reason"].value_counts(dropna=False))

# Determine the evaluator’s 3-month holdout window from history
from src.loader import load_data
hist,_ = load_data()
long = (
    hist.melt(id_vars=["Country","Region","Date"],
              value_vars=[c for c in ["Received","Submitted"] if c in hist.columns],
              var_name="Metric", value_name="Value")
        .sort_values(["Country","Region","Metric","Date"])
)
h_end   = long["Date"].max()
h_start = (h_end - pd.DateOffset(months=2)).replace(day=1)
print(f"\nHoldout window used by evaluator: {h_start.date()} → {h_end.date()}\n")

# Per-model strict-window coverage (helps spot date misalignment)
models = [
    ("LastYear_MoM",               OUT_DIR/"Models"/"Model01_LastYear"/"backtest.csv"),
    ("YoY_MoM_with_SeasonalPrior", OUT_DIR/"Models"/"Model02_MoM_YoY"/"backtest.csv"),
    ("Rolling_MoM",                OUT_DIR/"Models"/"Model03_RollMoM"/"backtest.csv"),
    ("AvgMonthly_MoM",             OUT_DIR/"Models"/"Model04_AvgMoM"/"backtest.csv"),
    ("Prophet_Enhanced",           OUT_DIR/"Models"/"Model05_Prophet_Fast"/"backtest.csv"),
    ("SARIMA_auto",                OUT_DIR/"Models"/"Model06_SARIMA"/"backtest.csv"),
    ("XGBoost_Lags",               OUT_DIR/"Models"/"Model07_XGBoost"/"backtest.csv"),
    ("LSTM_Univariate",            OUT_DIR/"Models"/"Model08_LSTM"/"backtest.csv"),
    # ("Ensemble_Weighted_AntiFlat",  OUT_DIR/"Models"/"Model09_Ensemble"/"backtest.csv"),  # excluded if desired
]

for name, path in models:
    if not path.exists():
        print(f"[MISS] {name}: no backtest at {path}")
        continue
    df = pd.read_csv(path, parse_dates=["Date"])
    mask = (df["Metric"]=="Received") & (df["Date"].between(h_start, h_end))
    print(f"{name:28s} rows in strict 3m window: {mask.sum():3d}")
    if mask.sum()==0:
        # Peek a few dates to reveal month alignment issues
        print("   ↪ sample dates:", df.head(3)[["Date","Metric"]].to_dict("records"))
