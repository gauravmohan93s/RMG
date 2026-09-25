# -*- coding: utf-8 -*-
#  src/eda.py – Exploratory-Data Artefact generator (monthly-only)
from __future__ import annotations

import json, warnings, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / non-interactive
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from rich.console import Console
from rich.table    import Table

from .config import OUT_DIR, RANDOM_SEED
from .loader import load_data

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)
console = Console()

EDA_DIR = OUT_DIR / "EDA"
EDA_DIR.mkdir(parents=True, exist_ok=True)

METRIC_COLS_DEFAULT = ["Received", "Submitted"]
ID_KEYS = ["Country", "Region", "Metric"]

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _adf_p(series: pd.Series) -> float:
    """Return ADF p-value; short/failed → 1.0 (non-stationary)."""
    s = series.dropna()
    if len(s) < 15:
        return 1.0
    try:
        return float(adfuller(s, autolag="AIC")[1])
    except Exception:
        return 1.0

def _series_profile(g: pd.DataFrame) -> dict:
    y = g["Value"]
    d = g["Date"]
    n = len(y)
    miss = int(y.isna().sum())
    return {
        "n_months": n,
        "n_missing": miss,
        "pct_missing": round(100.0 * miss / n, 1) if n else 0.0,
        "first": d.min().date() if len(d) else None,
        "last":  d.max().date() if len(d) else None,
        "variance": float(y.var()) if n else 0.0,
        "short_series": int(n < 24),
    }

def _detect_caps(g: pd.DataFrame) -> pd.DataFrame:
    """Flag spikes using Q3 + 3*IQR cap per series; floor at 0 (non-negative apps)."""
    q1, q3 = g["Value"].quantile([0.25, 0.75])
    iqr    = q3 - q1
    hi_cap = q3 + 3.0 * iqr
    flag = g["Value"] > hi_cap
    g = g.copy()
    g["shock_flag"]   = flag.astype("uint8")
    capped = np.where(flag, hi_cap, g["Value"])
    g["Value_capped"] = np.clip(capped, 0.0, np.inf)
    return g

def _mom_bands(g: pd.DataFrame, recent_years: int = 2) -> pd.DataFrame:
    """
    Compute month-wise MoM bands & recency-weighted mean for ONE series g:
        ['Country','Region','Metric','Month','p05','p95','rw_mean']
    Uses capped values, monthly frequency, and 0.7/0.3 recency weighting.
    """
    # Ensure monthly, continuous
    y = (
        g.set_index("Date")["Value"]
         .asfreq("MS")
         .interpolate("linear", limit_direction="both")
    ).clip(lower=0)
    pct = y.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    out = pd.DataFrame({
        "Country": g["Country"].iloc[0],
        "Region":  g["Region"].iloc[0],
        "Metric":  g["Metric"].iloc[0],
        "Month":   np.arange(1, 13, dtype=int),
        "p05":     np.zeros(12, dtype=float),
        "p95":     np.zeros(12, dtype=float),
        "rw_mean": np.zeros(12, dtype=float),
    })

    if pct.empty or not isinstance(pct.index, pd.DatetimeIndex):
        return out

    end = pct.index.max()
    cutoff = end - pd.DateOffset(years=recent_years)
    recent = pct[pct.index > cutoff]
    older  = pct[pct.index <= cutoff]

    rows = []
    for m in range(1, 13):
        pm_all = pct[pct.index.month == m]
        pm_r   = recent[recent.index.month == m]
        pm_o   = older [older .index.month == m]

        if pm_all.empty:
            p05 = p95 = 0.0
        else:
            p05, p95 = pm_all.quantile([0.05, 0.95]).astype(float).tolist()

        if len(pm_r) and len(pm_o):
            rw = 0.7*pm_r.mean() + 0.3*pm_o.mean()
        elif len(pm_r):
            rw = pm_r.mean()
        elif len(pm_o):
            rw = pm_o.mean()
        else:
            rw = 0.0

        rw = float(np.clip(rw, -0.3, 0.3))
        rows.append([m, float(p05), float(p95), rw])

    out[["Month","p05","p95","rw_mean"]] = pd.DataFrame(rows, columns=["Month","p05","p95","rw_mean"])
    return out

# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    # 1) Load cleaned monthly table
    df, ex_cols = load_data()

    # Build id_vars defensively
    cols = set(df.columns)
    id_vars = [c for c in ["Country","Region","Date","FY_Month"] if c in cols]
    if ex_cols:
        id_vars += [c for c in ex_cols if c in cols]

    metric_cols = [c for c in METRIC_COLS_DEFAULT if c in cols]
    if not metric_cols:
        raise RuntimeError("No metric columns found (expected one of 'Received','Submitted').")

    # Long form (monthly-only)
    long = (
        pd.melt(
            df, id_vars=id_vars, value_vars=metric_cols,
            var_name="Metric", value_name="Value"
        )
        .sort_values(["Country","Region","Metric","Date"])
        .reset_index(drop=True)
    )

    # 1a) Series calendar (for availability & sanity)
    cal = (
        long.groupby(ID_KEYS, as_index=False)
            .agg(first=("Date","min"), last=("Date","max"), n_months=("Date","count"))
    )
    cal.to_csv(EDA_DIR / "series_calendar.csv", index=False)

    # 2) Series profile
    profile = (
        long.groupby(ID_KEYS, sort=False)
            .apply(_series_profile).apply(pd.Series)
            .reset_index()
    )

    xl_path   = EDA_DIR / "series_profile.xlsx"
    html_path = EDA_DIR / "series_profile.html"
    csv_path  = EDA_DIR / "series_profile.csv"
    profile.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xl_path) as w:
        profile.to_excel(w, index=False)
    html = (
        profile.style
               .background_gradient(cmap="Reds", subset=["pct_missing"])
               .bar(subset=["n_months"], color="#5fba7d")
               .hide(axis="index")
               .to_html()
    )
    html_path.write_text(html, encoding="utf-8")

    # 3) ADF stationarity test (p-values) → differencing hint
    pvals = (
        long.groupby(ID_KEYS, sort=False)["Value"]
            .apply(_adf_p)
            .rename("adf_p")
            .reset_index()
    )
    # histogram
    fig, ax = plt.subplots(figsize=(4,3))
    ax.hist(pvals["adf_p"], bins=20, edgecolor="k")
    ax.axvline(0.05, color="red", ls="--")
    ax.set_title("ADF p-values across series")
    ax.set_xlabel("p-value"); ax.set_ylabel("Count")
    fig.tight_layout()
    adf_fig_path = EDA_DIR / "adf_hist.png"
    fig.savefig(adf_fig_path, dpi=120)
    plt.close(fig)

    need_diff = pvals.assign(need_diff=lambda d: (d.adf_p > 0.05).astype("uint8"))
    need_diff_path = EDA_DIR / "need_diff_flags.parquet"
    need_diff.to_parquet(need_diff_path, index=False)

    # 4) Spike capping + flags
    capped_parts, outlier_rows = [], []
    for _, g in long.groupby(ID_KEYS, sort=False):
        g2 = _detect_caps(g)
        capped_parts.append(g2)
        outlier_rows.append(g2[g2["shock_flag"] == 1])

    capped   = pd.concat(capped_parts, ignore_index=True) if capped_parts else long.copy()
    outliers = pd.concat(outlier_rows, ignore_index=True) if outlier_rows else capped.iloc[0:0]
    outliers.to_csv(EDA_DIR / "outliers.csv", index=False)
    capped_path = EDA_DIR / "data_with_shock_flag.parquet"
    capped.to_parquet(capped_path, index=False)

    # 5) Month-of-year MoM bands (on capped values)
    mom_parts = []
    for _, g in capped.groupby(ID_KEYS, sort=False):
        g_cap = g.assign(Value=g["Value_capped"]) if "Value_capped" in g.columns else g
        mom_parts.append(_mom_bands(g_cap))

    mom_bands = pd.concat(mom_parts, ignore_index=True) if mom_parts else pd.DataFrame(
        columns=["Country","Region","Metric","Month","p05","p95","rw_mean"]
    )
    mom_bands_path = EDA_DIR / "mom_bands.parquet"
    if not mom_bands.empty:
        mom_bands.to_parquet(mom_bands_path, index=False)

    # 6) Persist modelling table (capped values only) for all downstream models
    prepared_path = OUT_DIR / "prepared_data.parquet"
    (
        capped.pivot_table(index=["Country","Region","Date"],
                           columns="Metric", values="Value_capped", aggfunc="first")
              .reset_index()
              .to_parquet(prepared_path, index=False)
    )

    # 7) Evaluation windows (for 15+3 and last-3 selection weighting)
    global_last = long["Date"].max()
    holdout_start = (global_last - pd.DateOffset(months=2)).replace(day=1)  # last 3 calendar months
    holdout_end   = global_last
    backtest_start = (global_last - pd.DateOffset(months=17)).replace(day=1)  # 18 months window
    windows = {
        "holdout_3m": {"start": holdout_start.strftime("%Y-%m-%d"), "end": holdout_end.strftime("%Y-%m-%d")},
        "backtest_18m": {"start": backtest_start.strftime("%Y-%m-%d"), "end": holdout_end.strftime("%Y-%m-%d")},
        "latest_month": global_last.strftime("%Y-%m-%d"),
    }
    (EDA_DIR / "evaluation_windows.json").write_text(json.dumps(windows, indent=2), encoding="utf-8")

    # 8) EDA action-plan JSON
    plan = {
        "run_timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "artefacts": {
            "series_profile_csv":      str(csv_path),
            "series_profile_xlsx":     str(xl_path),
            "series_profile_html":     str(html_path),
            "series_calendar":         str(EDA_DIR / "series_calendar.csv"),
            "adf_histogram":           str(adf_fig_path),
            "need_diff_flags":         str(need_diff_path),
            "outlier_list":            str(EDA_DIR / "outliers.csv"),
            "shock_flag_data":         str(capped_path),
            "mom_bands":               str(mom_bands_path),
            "prepared_data_parquet":   str(prepared_path),
            "evaluation_windows_json": str(EDA_DIR / "evaluation_windows.json"),
        },
        "stats": {
            "n_series":         int(len(profile)),
            "n_short":          int(profile["short_series"].sum()),
            "n_high_missing":   int((profile["pct_missing"] > 20).sum()),
            "pct_stationary":   round(100 * (1 - need_diff["need_diff"].mean()), 1),
            "pct_outlier_rows": round(100 * len(outliers) / max(1, len(long)), 2),
        },
        "parameters": {
            "stationarity_alpha":      0.05,
            "short_series_cut":        24,
            "outlier_cap_rule":        "Q3 + 3*IQR (floor 0)",
            "mom_bands_recent_years":  2,
            "windows": windows,
        },
    }
    json_path = EDA_DIR / "eda_action_plan.json"
    json_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    # 9) TL;DR
    tbl = Table(title=f"EDA summary  ({plan['run_timestamp']})", pad_edge=False)
    tbl.add_column("Item", style="cyan bold")
    tbl.add_column("Value", style="green")
    tbl.add_row("Time-series total",          str(plan["stats"]["n_series"]))
    tbl.add_row("Short (<24 mo)",             f"{plan['stats']['n_short']}  ▸ check baselines")
    tbl.add_row(">20 % missing",              str(plan["stats"]["n_high_missing"]))
    tbl.add_row("Stationary already",         f"{plan['stats']['pct_stationary']} %")
    tbl.add_row("Rows flagged as spike",      f"{plan['stats']['pct_outlier_rows']} %")
    tbl.add_section()
    tbl.add_row("Need-diff flags",            Path(plan["artefacts"]["need_diff_flags"]).name)
    tbl.add_row("Capped data + shock_flag",   Path(plan["artefacts"]["shock_flag_data"]).name)
    tbl.add_row("MoM bands (p05/p95/rw)",     Path(plan["artefacts"]["mom_bands"]).name)
    tbl.add_row("Prepared table",             Path(plan["artefacts"]["prepared_data_parquet"]).name)
    tbl.add_row("Eval windows",               Path(plan["artefacts"]["evaluation_windows_json"]).name)
    console.print(tbl)
    (EDA_DIR / "tldr.txt").write_text(tbl.__str__(), encoding="utf-8")

    print(f"✅ prepared_data.parquet written → {prepared_path}")
    console.print(f"[bold green]✅ EDA artefacts saved → {EDA_DIR}[/]")

if __name__ == "__main__":
    main()
