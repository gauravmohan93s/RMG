# -*- coding: utf-8 -*-
"""
Model 08 — LSTM (univariate, FAST by default)
----------------------------------------------
• Transform: log1p(y) → z-score
• Backtest: STRICT OOS = last 3 months (train on y[:-3]), recursive
• Forecast: train on FULL history, recursive HORIZON steps
• EDA-aware: month-wise step clipping via OUT_DIR/EDA/mom_bands.parquet (fallback [0.8,1.2])
• No negative outputs (clip ≥ 0)
Artefacts → …/Models/Model08_LSTM/
"""
from __future__ import annotations

# std-lib
import os, warnings, logging, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from sklearn.preprocessing import StandardScaler

# tensorflow (configure BEFORE importing TF heavy modules)
RMG_LSTM_CPU  = os.getenv("RMG_LSTM_CPU", "0") != "0"
if RMG_LSTM_CPU:
    # optional CPU-only to avoid GPU init / contention on servers
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

HAS_TENSORFLOW = False
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except Exception as e:
    TF_IMPORT_ERROR = e
    # Define dummy names to avoid NameErrors/type errors if inspected
    Sequential = object
    LSTM = object
    Dense = object
    Dropout = object
    EarlyStopping = object
    ReduceLROnPlateau = object

# project
from ..config import OUT_DIR, HORIZON, RANDOM_SEED
from ..loader import load_data

# ─────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)
if HAS_TENSORFLOW:
    tf.random.set_seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

# runtime knobs
FAST_MODE   = os.getenv("RMG_FAST_LSTM", "1") != "0"   # default ON
OSS_MONTHS  = 3
N_LAGS      = int(os.getenv("RMG_LSTM_LAGS", "12"))

# TF threads (tune if needed)
if HAS_TENSORFLOW:
    if os.getenv("TF_NUM_INTEROP_THREADS"):
        tf.config.threading.set_inter_op_parallelism_threads(int(os.getenv("TF_NUM_INTEROP_THREADS")))
    if os.getenv("TF_NUM_INTRAOP_THREADS"):
        tf.config.threading.set_intra_op_parallelism_threads(int(os.getenv("TF_NUM_INTRAOP_THREADS")))

console = Console()

def _safe(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", s)

def fiscal_year(ts: pd.Timestamp) -> int:
    return ts.year + (ts.month >= 4)

# ── sequences & model ────────────────────────────────────────────────
def make_sequences(series_1d: np.ndarray, n_lags: int = N_LAGS) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a 1D array into (X,y) for LSTM: X.shape=(n, n_lags, 1), y.shape=(n,)."""
    X, y = [], []
    for i in range(n_lags, len(series_1d)):
        X.append(series_1d[i-n_lags:i])
        y.append(series_1d[i])
    if not X:
        return np.empty((0, n_lags, 1), dtype=np.float32), np.empty((0,), dtype=np.float32)
    X = np.asarray(X, dtype=np.float32).reshape(-1, n_lags, 1)
    y = np.asarray(y, dtype=np.float32)
    return X, y

def build_model(n_lags: int = N_LAGS) -> Sequential:
    """Small, stable LSTM. FAST mode is lighter."""
    units = 24 if FAST_MODE else 48
    drop  = 0.1 if FAST_MODE else 0.2
    model = Sequential([
        LSTM(units, input_shape=(n_lags, 1), activation="tanh", recurrent_dropout=0.0),
        Dropout(drop),
        Dense(1, activation="linear")
    ])
    model.compile(loss="mae", optimizer="adam")
    return model

class MLPFallbackModel:
    def __init__(self, n_lags: int):
        from sklearn.neural_network import MLPRegressor
        units = 24 if FAST_MODE else 48
        max_iter = 100 if FAST_MODE else 200
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(units,),
            activation="tanh",
            solver="adam",
            max_iter=max_iter,
            random_state=RANDOM_SEED,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.n_lags = n_lags

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_2d = X.reshape(X.shape[0], X.shape[1])
        self.mlp.fit(X_2d, y)
        return self

    def predict(self, X: np.ndarray, verbose=0):
        X_2d = X.reshape(X.shape[0], X.shape[1])
        preds = self.mlp.predict(X_2d)
        return preds.reshape(-1, 1)

def fit_lstm(X: np.ndarray, y: np.ndarray) -> Optional[Sequential]:
    if len(X) < 8:
        return None
    if not HAS_TENSORFLOW:
        return MLPFallbackModel(n_lags=X.shape[1]).fit(X, y)
    es_pat = 6 if FAST_MODE else 10
    rl_pat = 3 if FAST_MODE else 5
    epochs = 40 if FAST_MODE else 90
    bs     = 16 if FAST_MODE else 32
    es = EarlyStopping(patience=es_pat, restore_best_weights=True, verbose=0, monitor="val_loss")
    rl = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=rl_pat, min_lr=1e-4, verbose=0)
    model = build_model(n_lags=X.shape[1])
    # IMPORTANT: shuffle=False to respect time order
    model.fit(
        X, y,
        epochs=epochs, batch_size=bs,
        validation_split=0.1, shuffle=False,
        callbacks=[es, rl], verbose=0
    )
    return model

# ── EDA integration: month-wise step bands ───────────────────────────
EDA_DIR = OUT_DIR / "EDA"
def _load_mom_bands() -> Optional[pd.DataFrame]:
    p = EDA_DIR / "mom_bands.parquet"
    try:
        df = pd.read_parquet(p)
        req = {"Country","Region","Metric","Month","p05","p95"}
        return df if req.issubset(df.columns) else None
    except Exception:
        return None
MOM_BANDS = _load_mom_bands()

def _month_step_bounds(cty: str, reg: str, metr: str, month: int) -> Tuple[float, float]:
    """Return (lo_step, hi_step) for series+calendar month; fallback [0.8, 1.2]."""
    if MOM_BANDS is None:
        return 0.8, 1.2
    sub = MOM_BANDS[(MOM_BANDS.Country==cty) &
                    (MOM_BANDS.Region==reg) &
                    (MOM_BANDS.Metric==metr) &
                    (MOM_BANDS.Month==int(month))]
    if sub.empty:
        return 0.8, 1.2
    p05, p95 = float(sub.p05.iloc[0]), float(sub.p95.iloc[0])
    lo = max(0.8, 1.0 + p05)
    hi = min(1.2, 1.0 + p95)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        lo, hi = 0.8, 1.2
    return lo, hi

def _clip_step(prev_level: float, raw_next: float, lo: float, hi: float) -> float:
    """Clip step = raw_next/prev_level to [lo,hi], return clipped next level ≥ 0."""
    base = max(1e-9, float(prev_level))
    step = float(raw_next) / base
    step = float(np.clip(step, lo, hi))
    return max(0.0, base * step)

# ── OSS split ────────────────────────────────────────────────────────
def _split_oss(y: pd.Series, k: int = OSS_MONTHS) -> tuple[pd.Series, pd.DatetimeIndex]:
    y = y.sort_index()
    if len(y) <= k:
        return y.iloc[:0], y.index
    return y.iloc[:-k], y.index[-k:]

# ─────────────────────────────────────────────────────────────────────
RUN_DIR  = OUT_DIR / "Models" / "Model08_LSTM"
PLOT_DIR = RUN_DIR / "plots"
RUN_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=RUN_DIR/"model08.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def main() -> None:
    if not HAS_TENSORFLOW:
        logging.warning("TensorFlow is not available. Using Scikit-Learn MLP Neural Network fallback for Model 08 LSTM. Error: %s", TF_IMPORT_ERROR)
        console.print("[yellow]⚠️  TensorFlow not available. Running Model 08 using Scikit-Learn MLP Neural Network fallback.[/]")

    # 1) Load & reshape
    df, _ = load_data()
    value_vars = [c for c in ["Received","Submitted"] if c in df.columns]
    if not value_vars:
        raise RuntimeError("No metric columns found (need 'Received' and/or 'Submitted').")

    melt = (
        df.melt(id_vars=["Country","Region","Date"],
                value_vars=value_vars,
                var_name="Metric", value_name="Value")
          .drop_duplicates(subset=["Country","Region","Metric","Date"], keep="last")
          .sort_values(["Country","Region","Metric","Date"])
    )

    records_back: List[pd.DataFrame] = []
    records_fore: List[pd.DataFrame] = []
    mgmt_tables: List[pd.DataFrame] = []

    for (cty, reg, metr), g in tqdm(melt.groupby(["Country","Region","Metric"], sort=False),
                                    desc="Model08_LSTM (OSS+Prod)"):
        # monthly series (mild interpolation)
        y = (g.set_index("Date")["Value"]
               .astype(float)
               .asfreq("MS")
               .interpolate("linear", limit_direction="both"))
        if len(y) < max(24, N_LAGS + 6):
            logging.warning("skip %s/%s/%s len=%d", cty, reg, metr, len(y))
            continue

        # ===================== STRICT OSS BACKTEST (last 3) =====================
        y_train, oss_idx = _split_oss(y, OSS_MONTHS)
        if not y_train.empty and len(oss_idx) > 0:
            # scale on TRAIN only (log1p → zscore)
            y_tr_log = np.log1p(y_train.values).reshape(-1, 1)
            scaler_bt = StandardScaler().fit(y_tr_log)
            y_tr_z = scaler_bt.transform(y_tr_log).flatten().astype(np.float32)

            X_tr, y_tr = make_sequences(y_tr_z, n_lags=N_LAGS)
            model_bt = fit_lstm(X_tr, y_tr) if len(X_tr) >= 8 else None

            hist_z = y_tr_z.copy()
            prev_level = float(y_train.iloc[-1])
            back_rows = []
            for dt in oss_idx:
                if (model_bt is not None) and (len(hist_z) >= N_LAGS):
                    x_in = np.asarray(hist_z[-N_LAGS:], dtype=np.float32).reshape(1, N_LAGS, 1)
                    yhat_z = float(model_bt.predict(x_in, verbose=0)[0, 0])
                    raw_next = float(np.expm1(scaler_bt.inverse_transform([[yhat_z]])[0, 0]))
                else:
                    raw_next = prev_level  # naive

                lo, hi = _month_step_bounds(cty, reg, metr, dt.month)
                pred = _clip_step(prev_level, raw_next, lo, hi)

                back_rows.append({
                    "Country": cty, "Region": reg, "Metric": metr,
                    "Date": dt, "Actual": float(y.loc[dt]), "Pred": pred
                })

                # update prev_level & z-history with CLIPPED prediction
                prev_level = pred
                yhat_log_clip = np.log1p(pred)
                yhat_z_clip   = float(scaler_bt.transform([[yhat_log_clip]])[0,0])
                hist_z = np.append(hist_z, yhat_z_clip)

            bt = pd.DataFrame(back_rows)
            if not bt.empty:
                records_back.append(bt)

        # ===================== 12M PRODUCTION FORECAST =====================
        y_all_log = np.log1p(y.values).reshape(-1, 1)
        scaler_f  = StandardScaler().fit(y_all_log)
        y_all_z   = scaler_f.transform(y_all_log).flatten().astype(np.float32)
        X_all, y_all = make_sequences(y_all_z, n_lags=N_LAGS)

        model_f = fit_lstm(X_all, y_all) if len(X_all) >= 8 else None

        fut_vals = []
        level = float(y.iloc[-1])  # last actual
        hist_z_f = y_all_z.copy()
        fut_idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=HORIZON, freq="MS")

        for dt in fut_idx:
            if (model_f is not None) and (len(hist_z_f) >= N_LAGS):
                x_in = np.asarray(hist_z_f[-N_LAGS:], dtype=np.float32).reshape(1, N_LAGS, 1)
                yhat_z = float(model_f.predict(x_in, verbose=0)[0, 0])
                raw_next = float(np.expm1(scaler_f.inverse_transform([[yhat_z]])[0, 0]))
            else:
                raw_next = level  # naive if too short

            lo, hi = _month_step_bounds(cty, reg, metr, dt.month)
            level = _clip_step(level, raw_next, lo, hi)
            fut_vals.append(level)

            yhat_log_clip = np.log1p(level)
            yhat_z_clip   = float(scaler_f.transform([[yhat_log_clip]])[0,0])
            hist_z_f = np.append(hist_z_f, yhat_z_clip)

        fc = pd.DataFrame({
            "Country": cty, "Region": reg, "Metric": metr,
            "Date": fut_idx, "Forecast": np.array(fut_vals, dtype=float)
        })
        records_fore.append(fc)

        # ============================ Management table (optional) ============================
        if 'bt' in locals() and isinstance(bt, pd.DataFrame) and not bt.empty:
            rows = []
            dfb = bt.set_index("Date").copy()
            dfb["FY"] = dfb.index.to_series().apply(fiscal_year)
            for yr in sorted(dfb["FY"].unique()):
                if yr-1 not in dfb["FY"].unique():
                    continue
                curr = dfb[dfb.FY==yr]
                prev = dfb[dfb.FY==yr-1]
                for d in curr.index:
                    d_prev = d - pd.DateOffset(years=1)
                    if d_prev not in prev.index:
                        continue
                    rows.append([
                        d.date(), cty, reg, metr,
                        float(prev.loc[d_prev,"Actual"]),
                        float(curr.loc[d,"Actual"]),
                        float(curr.loc[d,"Pred"]),
                        round(float(curr.loc[d,"Pred"]-curr.loc[d,"Actual"]), 2)
                    ])
            if rows:
                mgmt_tables.append(pd.DataFrame(
                    rows, columns=[
                        "Month","Country","Region","Metric",
                        "Actual_prevFY","Actual_currFY",
                        "LSTM_pred","Error"
                    ]
                ))

        # =============================== Plot ===============================
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(y.index, y.values, label="Actual")
        if 'bt' in locals() and isinstance(bt, pd.DataFrame) and not bt.empty:
            ax.plot(bt["Date"], bt["Pred"], "--", label=f"OSS backtest ({OSS_MONTHS}m)")
        ax.plot(fc["Date"], fc["Forecast"], ":", label=f"Forecast {HORIZON}m")
        ax.set_title(f"{cty} – {reg} – {metr}")
        ax.grid(alpha=.3); ax.legend(loc="upper left")
        for l in ax.get_xticklabels(): l.set_rotation(45)
        fig.tight_layout()
        fig.savefig(PLOT_DIR/f"{_safe(cty)}_{_safe(reg)}_{_safe(metr)}.png", dpi=120)
        plt.close(fig)

    # 2) Save artefacts
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    back_all = pd.concat(records_back, ignore_index=True) if records_back else pd.DataFrame()
    fore_all = pd.concat(records_fore, ignore_index=True) if records_fore else pd.DataFrame()
    mgmt_all = pd.concat(mgmt_tables,    ignore_index=True) if mgmt_tables else pd.DataFrame()

    if not fore_all.empty:
        fore_all.to_csv(RUN_DIR/"forecast_h12.csv", index=False)
    back_all.to_csv(RUN_DIR/"backtest.csv", index=False)

    with pd.ExcelWriter(RUN_DIR/"mgmt_table.xlsx") as w:
        (mgmt_all if not mgmt_all.empty else
         pd.DataFrame({"info":["no rows produced"]})
        ).to_excel(w, sheet_name="Mgmt_Table", index=False)

    # 3) OSS performance & TL;DR (median over series)
    if not back_all.empty:
        perf = (
            back_all.groupby(["Country","Region","Metric"])
            .apply(lambda d: pd.Series({
                "MAE" : (d.Actual-d.Pred).abs().mean(),
                "MAPE": (np.abs(d.Actual-d.Pred)/d.Actual.replace(0,np.nan)).mean()*100.0
            })).reset_index()
        )
        perf.to_excel(RUN_DIR/"series_scores.xlsx", index=False)
    else:
        perf = pd.DataFrame()

    tbl = Table(title=f"Model 08 LSTM — OSS last {OSS_MONTHS} months (median errors)", pad_edge=False)
    tbl.add_column("Metric"); tbl.add_column("Median MAE", justify="right")
    tbl.add_column("Median MAPE %", justify="right")
    for metr in ["Received","Submitted"]:
        sub = perf[perf["Metric"]==metr]
        if not sub.empty:
            tbl.add_row(metr, f"{sub['MAE'].median():,.0f}", f"{sub['MAPE'].median():.1f}")
    console.print(tbl)
    (RUN_DIR/"tldr.txt").write_text(tbl.__str__())
    console.print(f"[bold green]✅ All artefacts saved →  {RUN_DIR}[/]")

if __name__=="__main__":
    main()
