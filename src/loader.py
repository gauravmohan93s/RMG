# ──────────────────────────────────────────────────────────────
#  src/loader.py
#     public functions:
#        - load_data()            -> (clean monthly df, exog_columns)
#        - load_prepared_data()   -> (prepared df from EDA, meta dict)
# ──────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import (
    RAW_FILE, EXOG_FILE, FREQ,          # e.g. FREQ="MS"
    ENABLE_EXOG, FIN_YEAR_START, OUT_DIR
)

# ──────────────────────────────────────────────────────────────
# Business filters & schema maps
DROP_REGIONS = {"B2C", "Test", "Unclaimed"}   # optional business filter
RENAMES = {                                   # raw → canonical
    "Month"           : "Date",
    "Country Grouped" : "Country",
    "Name of Region"  : "Region",
    "ReceivedApps"    : "Received",
    "SubmittedApps"   : "Submitted",
}
CORE_COLS    = ["Country", "Region", "Date", "Received", "Submitted"]
NUMERIC_COLS = ["Received", "Submitted"]
# ──────────────────────────────────────────────────────────────


# ────────── helpers ───────────────────────────────────────────

def _to_month_start_series(idxlike) -> pd.Index:
    """
    Coerce a scalar, list-like, Series or Index of datetimes to month-start timestamps.
    Always returns a DatetimeIndex aligned to the input length.
    """
    dt = pd.to_datetime(idxlike, errors="coerce")

    # If we already have an Index/DatetimeIndex, operate directly (no .dt)
    if isinstance(dt, (pd.DatetimeIndex, pd.Index)) and not isinstance(dt, pd.Series):
        return dt.to_period("M").to_timestamp(how="start")

    # Otherwise make a Series so .dt works
    s = pd.Series(dt)
    return s.dt.to_period("M").dt.to_timestamp(how="start")


def _to_month_start_df(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Normalize a date column to month-start (00:00) safely."""
    out = df.copy()
    out[date_col] = _to_month_start_series(out[date_col])
    return out


def _dedupe_last(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """
    Keep the last non-null record per key combo (stable, explicit).
    Assumes df is already sorted in file order; we re-sort after.
    """
    df = df.copy()
    df["_nn"] = df[NUMERIC_COLS].notna().sum(axis=1)
    df.sort_values(keys + ["_nn"], inplace=True)  # worst → best by completeness
    df = df.drop_duplicates(subset=keys, keep="last").drop(columns=["_nn"])
    df.sort_values(keys, inplace=True)
    return df


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # OPTIONAL: non-negativity guard (default ON)
    if os.environ.get("RMG_CLIP_NONNEG", "1") == "1":
        for c in cols:
            if c in out.columns:
                out[c] = out[c].clip(lower=0)
    return out


def _trim_to_observed_window(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Treat all months present in RAW as already-completed.
    Do NOT drop the last month based on 'today'; simply ensure dates are within
    the observed min..max window of the raw file after month-start normalization.
    """
    out = _to_month_start_df(df, date_col)
    if out.empty:
        return out
    min_ms = out[date_col].min()
    max_ms = out[date_col].max()
    return out[(out[date_col] >= min_ms) & (out[date_col] <= max_ms)].copy()


# ────────── core readers / mergers ───────────────────────────

def _read_main(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"RAW_FILE not found: {path}")

    df = pd.read_excel(path)
    df.rename(columns=RENAMES, inplace=True, errors="ignore")

    # minimal schema guard
    missing = [c for c in ["Date", "Country", "Region"] if c not in df.columns]
    if missing:
        raise ValueError(f"❌ missing required column(s) in {path.name}: {missing}")

    # normalize date to month start
    df = _to_month_start_df(df, "Date")

    # ensure metric columns exist (if absent in this export)
    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # coerce numerics & clip negatives (optional)
    df = _coerce_numeric(df, NUMERIC_COLS)

    # business filters & keep only core columns
    df = df[~df["Region"].isin(DROP_REGIONS)].copy()
    df = df[[c for c in CORE_COLS if c in df.columns]]

    # de-duplicate on (Country, Region, Date) keeping last non-null
    df = _dedupe_last(df, ["Country", "Region", "Date"])

    # IMPORTANT: keep ALL months present in RAW (treat as completed)
    df = _trim_to_observed_window(df, "Date")
    return df


def _read_exog(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    if not ENABLE_EXOG or not path.exists():
        return pd.DataFrame(), []

    ex = pd.read_excel(path)
    # allow 'Month' or 'Date'
    if "Date" not in ex.columns and "Month" not in ex.columns:
        raise ValueError("❌ EXOG file must contain a 'Date' or 'Month' column.")
    if "Date" not in ex.columns:
        ex = ex.rename(columns={"Month": "Date"})

    ex = _to_month_start_df(ex, "Date")
    ex_cols = [c for c in ex.columns if c != "Date"]

    # coerce all exog to numeric (non-numeric → NaN)
    for c in ex_cols:
        ex[c] = pd.to_numeric(ex[c], errors="coerce")

    ex = ex.dropna(subset=["Date"]).sort_values("Date")
    return ex, ex_cols


def _merge_exog(df: pd.DataFrame, ex: pd.DataFrame, ex_cols: List[str]) -> pd.DataFrame:
    if ex.empty or not ex_cols:
        return df
    return df.merge(ex[["Date"] + ex_cols], on="Date", how="left")


def _complete_monthly_index(df: pd.DataFrame, ex_cols: List[str]) -> pd.DataFrame:
    """
    Ensure perfect monthly continuity per (Country, Region),
    interpolate core metrics internally, and ffill/bfill exog.
    """
    pieces = []
    for (cty, reg), g in df.groupby(["Country", "Region"], sort=False):
        g = g.sort_values("Date").reset_index(drop=True)

        # full monthly index from first to last month present for this (cty,reg)
        idx = pd.date_range(g["Date"].min(), g["Date"].max(), freq=FREQ)  # e.g., 'MS'
        g2 = (
            g.set_index("Date")
             .reindex(idx)
             .rename_axis("Date")
             .reset_index()
        )
        g2[["Country", "Region"]] = (cty, reg)

        # interpolate metrics within range; fill edges by bfill/ffill to stay stable
        g2[NUMERIC_COLS] = g2[NUMERIC_COLS].interpolate("linear", limit_direction="both")

        # exogenous: ffill/bfill; fallback to 0.0 if still NaN
        if ex_cols:
            g2[ex_cols] = g2[ex_cols].ffill().bfill()
            for c in ex_cols:
                if g2[c].isna().any():
                    g2[c] = g2[c].fillna(0.0)

        pieces.append(g2)

    out = (
        pd.concat(pieces, ignore_index=True)
          .sort_values(["Country", "Region", "Date"])
          .reset_index(drop=True)
    )
    return out


# ────────── public loader ───────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, list[str]]:
    """
    Returns
    -------
    df : pd.DataFrame
        Cleaned, *monthly* table with columns:
        [Country, Region, Date, Received, Submitted, (exog...), FY_Month]
        NOTE: Every month present in RAW is treated as a completed month.
              No dropping based on today's date.
    ex_cols : list[str]
        Names of exogenous regressor columns merged (possibly empty).
    """
    # 1) main (treat all months present as completed)
    df = _read_main(RAW_FILE)

    # 2) exog merge (optional)
    ex, ex_cols = _read_exog(EXOG_FILE)
    df = _merge_exog(df, ex, ex_cols)

    # 3) complete monthly index per Country×Region
    df = _complete_monthly_index(df, ex_cols)

    # 4) fiscal-month tag AFTER all date ops
    df["FY_Month"] = (
        (df["Date"].dt.month - int(FIN_YEAR_START)) % 12 + 1
    ).astype("int8")  # 1..12

    # 5) final schema guard & sort
    want = ["Country", "Region", "Date", "Received", "Submitted"] + ex_cols + ["FY_Month"]
    df = df[want].sort_values(["Country", "Region", "Date"]).reset_index(drop=True)

    # Dev-time safety: ensure monthly-only invariant
    assert (df["Date"].dt.day == 1).all(), "Non-month-start dates slipped into loader output."

    return df, ex_cols


def load_prepared_data() -> tuple[pd.DataFrame, dict]:
    """
    Return the parquet table produced by src.eda plus meta-info dict
    from EDA (if present). This path is used when --use-prepared is on.
    """
    pp = OUT_DIR / "prepared_data.parquet"
    jp = OUT_DIR / "EDA" / "eda_action_plan.json"

    if not pp.exists():
        raise FileNotFoundError(
            f"prepared_data.parquet not found at {pp} — run `python -m src.eda` first."
        )
    df = pd.read_parquet(pp)

    # Enforce month-start & sort
    df = _to_month_start_df(df, "Date")
    df.sort_values(["Country", "Region", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Optional meta
    if jp.exists():
        try:
            meta = json.loads(jp.read_text())
        except Exception:
            meta = {}
    else:
        meta = {}

    return df, meta
