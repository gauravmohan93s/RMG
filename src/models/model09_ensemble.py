# -*- coding: utf-8 -*-
# src/models/model09_ensemble.py
from __future__ import annotations

import logging, warnings, re
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from ..config import OUT_DIR, RANDOM_SEED
from ..loader import load_data

# ─────────────────────────────────────────────────────────────────────
def _safe(s: str) -> str:
    return re.sub(r'[\\/:\*?"<>|]', "_", str(s))

def _normcols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    mapping = {
        "country":"Country","region":"Region","metric":"Metric","date":"Date",
        "actual":"Actual","pred":"Pred","forecast":"Forecast",
        "model":"Model","chosen_model":"Chosen_Model",
    }
    return df.rename(columns={c: mapping.get(str(c).strip().lower(), c) for c in df.columns})

def _has(df: pd.DataFrame, req: set[str]) -> bool:
    return (df is not None) and (not df.empty) and req.issubset(df.columns)

def _coefvar(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    mu = np.mean(x)
    if mu == 0:
        return np.inf if np.std(x) > 0 else 0.0
    return float(np.std(x) / abs(mu))

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)
console = Console()

BASE       = OUT_DIR / "Models"
GROUP1     = {
    "Model01_LastYear": BASE / "Model01_LastYear",
    "Model02_MoM_YoY":  BASE / "Model02_MoM_YoY",
    "Model03_RollMoM":  BASE / "Model03_RollMoM",
    "Model04_AvgMoM":   BASE / "Model04_AvgMoM",
}
GROUP2     = {
    "Model05_Prophet":      BASE / "Model05_Prophet_Fast",
    "Model06_SARIMA":       BASE / "Model06_SARIMA",
    "Model07_XGBoost":      BASE / "Model07_XGBoost",
    "Model08_LSTM":         BASE / "Model08_LSTM",
    "Model10_Global_LGBM":  BASE / "Model10_Global_LGBM",
    "Model11_Croston_TSB":  BASE / "Model11_Croston_TSB",
}
ALL_MODELS = {**GROUP1, **GROUP2}

RUN_DIR  = BASE / "Model09_Ensemble"
PLOT_DIR = RUN_DIR / "plots"
RUN_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=RUN_DIR/"model09.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

REQ_BT = {"Country","Region","Metric","Date","Actual","Pred"}
REQ_FC = {"Country","Region","Metric","Date","Forecast"}

MAX_WEIGHT = 0.70      # cap any single model weight after normalization
FLAT_CV_TH = 0.005     # 0.5% coefficient of variation → treat as near-flat

def main() -> None:
    # ─────────────────────────────────────────────────────────────────
    # 1) LOAD backtests & forecasts
    # ─────────────────────────────────────────────────────────────────
    backtests: Dict[str, pd.DataFrame] = {}
    forecasts: Dict[str, pd.DataFrame] = {}

    for name, dpath in ALL_MODELS.items():
        bt_file = dpath / "backtest.csv"
        fc_file = dpath / "forecast_h12.csv"

        if bt_file.exists():
            try:
                bt = pd.read_csv(bt_file, parse_dates=["Date"], dayfirst=True)
                bt = _normcols(bt)
                if _has(bt, REQ_BT):
                    bt = bt.loc[:, list(REQ_BT)].copy()
                    bt = bt.dropna(subset=["Date"]).sort_values("Date")
                    backtests[name] = bt
                else:
                    logging.warning("Skipping backtest for %s: missing %s", name, REQ_BT - set(bt.columns))
            except Exception as e:
                logging.exception("Backtest read failed for %s: %s", name, e)
        else:
            logging.warning("Backtest missing for %s: %s", name, bt_file)

        if fc_file.exists():
            try:
                fc = pd.read_csv(fc_file, parse_dates=["Date"], dayfirst=True)
                fc = _normcols(fc)
                if _has(fc, REQ_FC):
                    fc = fc.loc[:, list(REQ_FC)].copy()
                    fc = fc.dropna(subset=["Date"]).sort_values("Date")
                    forecasts[name] = fc
                else:
                    logging.warning("Skipping forecast for %s: missing %s", name, REQ_FC - set(fc.columns))
            except Exception as e:
                logging.exception("Forecast read failed for %s: %s", name, e)
        else:
            logging.warning("Forecast missing for %s: %s", name, fc_file)

    if not backtests:
        console.print("[yellow]⚠️  No valid backtests — skipping ensemble.[/]")
        for fn in ("backtest.csv","forecast_h12.csv"): pd.DataFrame().to_csv(RUN_DIR/fn, index=False)
        Path(RUN_DIR/"tldr.txt").write_text("No ensemble performed")
        return

    # ─────────────────────────────────────────────────────────────────
    # 2) MAE per (Country,Region) using ONLY Received
    # ─────────────────────────────────────────────────────────────────
    mae_tables: List[pd.DataFrame] = []
    valid_models: List[str] = []

    for model_name, bt in backtests.items():
        try:
            rec = bt[bt["Metric"]=="Received"]
            if rec.empty:
                logging.warning("%s: no Received rows in backtest; skip for selection.", model_name)
                continue
            m = (rec.assign(err=lambda d: (d["Actual"]-d["Pred"]).abs())
                     .groupby(["Country","Region"], as_index=False)
                     .agg(**{model_name: ("err","mean")}))
            if _has(m, {"Country","Region"}):
                mae_tables.append(m)
                valid_models.append(model_name)
        except Exception as e:
            logging.exception("MAE build failed for %s: %s", model_name, e)

    if not mae_tables:
        console.print("[yellow]⚠️  No MAE tables on Received; cannot select models.[/]")
        for fn in ("backtest.csv","forecast_h12.csv"): pd.DataFrame().to_csv(RUN_DIR/fn, index=False)
        Path(RUN_DIR/"tldr.txt").write_text("No ensemble performed (no Received backtests)")
        return

    def _outer_merge(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        keys = ["Country","Region"]
        if not set(keys).issubset(a.columns) or not set(keys).issubset(b.columns):
            return a if len(a) >= len(b) else b
        return a.merge(b, on=keys, how="outer")

    mae_df = reduce(_outer_merge, mae_tables)
    for col in valid_models:
        if col in mae_df.columns:
            mae_df[col] = pd.to_numeric(mae_df[col], errors="coerce")

    # ─────────────────────────────────────────────────────────────────
    # 3) SELECT 1 from GROUP1 + top-2 from GROUP2 per (Country,Region)
    # ─────────────────────────────────────────────────────────────────
    def _pick_best(row: pd.Series, names: List[str], k: int) -> List[str]:
        names = [n for n in names if n in row.index]
        vals = row[names].astype(float).dropna()
        return vals.nsmallest(min(k, len(vals))).index.tolist()

    selections = []
    for _, row in mae_df.iterrows():
        c, r = row.get("Country", np.nan), row.get("Region", np.nan)
        if pd.isna(c) or pd.isna(r): 
            continue
        best1 = _pick_best(row, list(GROUP1.keys()), 1)
        best2 = _pick_best(row, list(GROUP2.keys()), 2)
        chosen: List[str] = []
        if best1: chosen += best1
        if best2: chosen += best2
        if len(chosen) < 2:
            rest = [n for n in valid_models if n not in chosen and n in row.index]
            if rest:
                chosen += row[rest].astype(float).dropna().nsmallest(2-len(chosen)).index.tolist()
        if len(chosen) >= 2:
            selections.append([c, r] + chosen[:3])

    selection_df = pd.DataFrame(selections, columns=["Country","Region","M1","M2","M3"])
    selection_df.to_csv(RUN_DIR/"region_selection_on_received.csv", index=False)

    if selection_df.empty:
        console.print("[yellow]⚠️  0 regions selected — no ensemble to build.[/]")
        for fn in ("backtest.csv","forecast_h12.csv"): pd.DataFrame().to_csv(RUN_DIR/fn, index=False)
        Path(RUN_DIR/"tldr.txt").write_text("No ensemble performed (no selections)")
        return

    console.print(f"▶️  Regions selected for ensembling: {len(selection_df)}")

    mae_idx = mae_df.set_index(["Country","Region"], drop=True)

    def _inv_mae_weights(key: Tuple[str,str], models: List[str]) -> np.ndarray:
        ws = []
        for m in models:
            v = np.nan
            if (key in mae_idx.index) and (m in mae_idx.columns):
                try:
                    v = float(mae_idx.loc[key, m])
                except Exception:
                    v = np.nan
            w = 1.0 / max(1e-6, float(v)) if pd.notna(v) else np.nan
            ws.append(w)
        ws = np.array(ws, dtype=float)
        ws = np.where(np.isfinite(ws), ws, 0.0)
        if ws.sum() == 0:
            ws = np.ones_like(ws)
        ws = ws / ws.sum()
        # cap dominance
        if ws.max() > MAX_WEIGHT:
            excess = ws.max() - MAX_WEIGHT
            i_max = int(np.argmax(ws))
            ws[i_max] = MAX_WEIGHT
            others = np.arange(len(ws)) != i_max
            if others.any():
                denom = ws[others].sum()
                if denom > 0:
                    ws[others] = ws[others] + excess * (ws[others] / denom)
                else:
                    ws[others] = ws[others] + excess / others.sum()
        return ws / ws.sum()

    # ─────────────────────────────────────────────────────────────────
    # 4) BUILD ENSEMBLE with deterministic model order + anti-flat fallback
    # ─────────────────────────────────────────────────────────────────
    def _weighted_union_average(
        panels: List[pd.DataFrame], models_order: List[str], weights: np.ndarray, prefix: str
    ) -> pd.Series:
        """
        Align by union of dates; for each date compute weighted average over available
        models, renormalizing weights for those present. Column order strictly follows
        models_order to guarantee weight alignment.
        """
        if not panels:
            return pd.Series(dtype=float)

        # start with an empty union index
        idx_union = pd.Index([])
        for p in panels:
            idx_union = idx_union.union(p.index)
        idx_union = pd.DatetimeIndex(idx_union).sort_values()

        # build matrix in order of models_order
        mat_cols = []
        for m in models_order:
            col = f"{prefix}{m}"
            # find panel for this model
            p = next((pp for pp in panels if col in pp.columns), None)
            if p is None:
                # create an all-NaN column aligned to union
                mat_cols.append(pd.Series(np.nan, index=idx_union))
            else:
                mat_cols.append(p[col].reindex(idx_union))
        M = np.column_stack([c.to_numpy(dtype=float) for c in mat_cols])

        out = np.full(len(idx_union), np.nan, dtype=float)
        for i in range(len(idx_union)):
            row = M[i, :]
            mask = np.isfinite(row)
            if not mask.any():
                continue
            w = weights[mask]
            w = w / w.sum() if w.sum() > 0 else np.ones(mask.sum()) / mask.sum()
            out[i] = float(np.dot(row[mask], w))
        return pd.Series(out, index=idx_union)

    records_back: List[pd.DataFrame] = []
    records_fore: List[pd.DataFrame] = []

    raw_df, _ = load_data()
    melt_raw = raw_df.melt(
        id_vars=["Country","Region","Date"],
        value_vars=[c for c in ["Received","Submitted"] if c in raw_df.columns],
        var_name="Metric", value_name="Value"
    )

    for _, sel in tqdm(selection_df.iterrows(), total=len(selection_df), desc="Ensembling"):
        cty, reg, m1, m2, m3 = sel.tolist()
        chosen = [m for m in [m1, m2, m3] if isinstance(m, str)]
        key = (cty, reg)
        w_base = _inv_mae_weights(key, chosen)

        for metr in ["Received","Submitted"]:
            # ---------- Backtest: panels in chosen order ----------
            bt_panels = []
            actual_union = pd.Series(dtype=float)
            for mdl in chosen:
                bt = backtests.get(mdl)
                if bt is None or bt.empty: 
                    continue
                s = (bt.query("Country==@cty & Region==@reg & Metric==@metr")[["Date","Actual","Pred"]]
                       .drop_duplicates("Date").set_index("Date").sort_index())
                if s.empty:
                    continue
                if actual_union.empty:
                    actual_union = s["Actual"].copy()
                else:
                    actual_union = actual_union.combine_first(s["Actual"])
                bt_panels.append(s[["Pred"]].rename(columns={"Pred": f"Pred_{mdl}"}))

            if bt_panels:
                ens_pred = _weighted_union_average(bt_panels, chosen, w_base, prefix="Pred_")
                dfb = pd.DataFrame({
                    "Country": cty, "Region": reg, "Metric": metr,
                    "Date": ens_pred.index,
                    "Actual": actual_union.reindex(ens_pred.index).to_numpy(),
                    "Pred": ens_pred.values
                }).dropna(subset=["Date"])
                if not dfb.empty:
                    records_back.append(dfb)

            # ---------- Forecast: panels in chosen order + anti-flat ----------
            fc_panels = []
            for mdl in chosen:
                fc = forecasts.get(mdl)
                if fc is None or fc.empty: 
                    continue
                s = (fc.query("Country==@cty & Region==@reg & Metric==@metr")[["Date","Forecast"]]
                       .drop_duplicates("Date").set_index("Date").sort_index())
                if s.empty:
                    continue
                fc_panels.append(s.rename(columns={"Forecast": f"Forecast_{mdl}"}))

            if fc_panels:
                ens_fc = _weighted_union_average(fc_panels, chosen, w_base, prefix="Forecast_")

                # anti-flat fallback
                vals = ens_fc.to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                need_fallback = False
                if vals.size <= 1:
                    need_fallback = True
                else:
                    cv = _coefvar(vals)
                    if np.allclose(vals, vals[0]) or (cv is not None and cv < FLAT_CV_TH):
                        need_fallback = True

                if need_fallback:
                    # pick best single (lowest Received MAE) among chosen that actually has forecast
                    ranking = [
                        (m, mae_idx.loc[key, m] if (key in mae_idx.index and m in mae_idx.columns) else np.inf)
                        for m in chosen
                    ]
                    ranking.sort(key=lambda x: x[1])
                    best_fc: Optional[pd.Series] = None
                    for best_single, _ in ranking:
                        src = forecasts.get(best_single)
                        if src is None or src.empty:
                            continue
                        s = (src.query("Country==@cty & Region==@reg & Metric==@metr")[["Date","Forecast"]]
                               .drop_duplicates("Date").set_index("Date")["Forecast"])
                        if not s.empty:
                            best_fc = s.sort_index()
                            logging.info("Flat ensemble %s/%s/%s → fallback to %s", cty, reg, metr, best_single)
                            break
                    if best_fc is not None:
                        ens_fc = best_fc

                dff = pd.DataFrame({
                    "Country": cty, "Region": reg, "Metric": metr,
                    "Date": ens_fc.index, "Forecast": ens_fc.values
                }).dropna(subset=["Date"])
                if not dff.empty:
                    records_fore.append(dff)

    back_df     = pd.concat(records_back, ignore_index=True) if records_back else pd.DataFrame()
    forecast_df = pd.concat(records_fore, ignore_index=True) if records_fore else pd.DataFrame()

    # ─────────────────────────────────────────────────────────────────
    # 5) MANAGEMENT TABLE (optional)
    # ─────────────────────────────────────────────────────────────────
    mgmt_rows = []
    if not back_df.empty and not melt_raw.empty:
        for (cty, reg, metr), grp in back_df.groupby(["Country","Region","Metric"]):
            raw_series = melt_raw.query("Country==@cty & Region==@reg & Metric==@metr").set_index("Date")["Value"]
            for _, row in grp.iterrows():
                dt = row["Date"]
                prevFY = dt - pd.DateOffset(years=1)
                prevMo = dt - pd.DateOffset(months=1)
                if prevFY not in raw_series.index or prevMo not in raw_series.index:
                    continue
                val_prev = float(raw_series.loc[prevFY])
                val_curr = float(row["Actual"])
                prev_mo_val = float(raw_series.loc[prevMo])
                mom_pct  = float(val_curr / prev_mo_val - 1.0) if prev_mo_val != 0 else 0.0
                pred     = float(row["Pred"])
                err      = pred - val_curr
                mgmt_rows.append([dt.date(), cty, reg, metr, val_prev, val_curr, round(mom_pct*100,2), pred, round(err,2)])

    mgmt_df = (pd.DataFrame(mgmt_rows, columns=[
        "Month","Country","Region","Metric",
        "Actual_prevFY","Actual_currFY","MoM%_currFY",
        "Pred_from_prevFY","Error"
    ]) if mgmt_rows else pd.DataFrame())

    # ─────────────────────────────────────────────────────────────────
    # 6) PLOTS
    # ─────────────────────────────────────────────────────────────────
    hist_long = load_data()[0].melt(
        id_vars=["Country","Region","Date"],
        value_vars=[c for c in ["Received","Submitted"] if c in raw_df.columns],
        var_name="Metric", value_name="Value"
    )

    for (c,r,m), h in hist_long.groupby(["Country","Region","Metric"], sort=False):
        b = back_df.query("Country==@c & Region==@r & Metric==@m").sort_values("Date")
        f = forecast_df.query("Country==@c & Region==@r & Metric==@m").sort_values("Date")
        if b.empty and f.empty:
            continue
        fig, ax = plt.subplots(figsize=(6,3))
        if not h.empty:
            ax.plot(h["Date"], h["Value"], label="Actual", lw=1)
        if not b.empty:
            ax.plot(b["Date"], b["Pred"], "--", label="Ensemble OOS fit", lw=1)
        if not f.empty:
            ax.plot(f["Date"], f["Forecast"], ":", label="Ensemble Forecast", lw=1)
        if not h.empty:
            last_hist = h["Date"].max()
            ax.axvline(last_hist, color="black", ls=":", lw=0.8)
        ax.set_title(f"{c} — {r} — {m}")
        ax.grid(alpha=.3); ax.legend(fontsize=7, loc="upper left")
        for t in ax.get_xticklabels(): t.set_rotation(45)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{_safe(c)}_{_safe(r)}_{_safe(m)}.png", dpi=120)
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────
    # 7) SAVE ARTEFACTS
    # ─────────────────────────────────────────────────────────────────
    back_df.to_csv(RUN_DIR/"backtest.csv", index=False)
    forecast_df.to_csv(RUN_DIR/"forecast_h12.csv", index=False)
    with pd.ExcelWriter(RUN_DIR/"mgmt_table.xlsx") as w:
        (mgmt_df if not mgmt_df.empty else pd.DataFrame({"info":["no rows"]})
        ).to_excel(w, sheet_name="Mgmt_Table", index=False)

    if not back_df.empty:
        perf = (
            back_df
            .assign(err=lambda d: (d.Actual - d.Pred).abs(),
                    ape=lambda d: np.abs(d.Actual - d.Pred)/d.Actual.replace(0, np.nan))
            .groupby(["Country","Region","Metric"], as_index=False)
            .agg(MAE=("err","mean"), MAPE=("ape", lambda x: x.mean()*100))
        )
        perf.to_excel(RUN_DIR/"series_scores.xlsx", index=False)
        tbl = Table(title="Model 09 Ensemble TL;DR (selection on Received)", pad_edge=False)
        tbl.add_column("Metric"); tbl.add_column("Median MAE", justify="right")
        tbl.add_column("Median MAPE %", justify="right")
        for metr in ["Received","Submitted"]:
            sub = perf[perf.Metric==metr]
            if not sub.empty:
                tbl.add_row(metr, f"{sub.MAE.median():,.0f}", f"{sub.MAPE.median():.1f}")
        console.print(tbl)
        (RUN_DIR/"tldr.txt").write_text(tbl.__str__())
    else:
        Path(RUN_DIR/"tldr.txt").write_text("No ensemble backtests")

    console.print(f"[bold green]✅ Ensemble artefacts saved → {RUN_DIR}[/]")
    logging.info("✅ Ensemble artefacts saved to %s", RUN_DIR)

if __name__ == "__main__":
    main()
