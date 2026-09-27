# -*- coding: utf-8 -*-
"""
Governed business overlay (policy / regulatory adjustments).

The statistical baseline is never overwritten. An adjustment is applied only if
the row is Status = "Approved", has an Owner and an Approved_On date, is inside
its validity window (Valid_From..Valid_To, YYYY-MM, blank = open) and its
multiplier is within [0.5, 1.5]. Everything else is listed as "not applied" with
the reason, so executives see exactly which judgement entered the numbers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

COLUMNS = ["Country", "Region", "Metric", "Multiplier", "Valid_From", "Valid_To",
           "Owner", "Rationale", "Source", "Status", "Approved_On"]
MULT_RANGE = (0.5, 1.5)


def load_overlay(settings_file: Path, sheet: str, template: Path) -> Tuple[pd.DataFrame, str]:
    df, source = None, ""
    if settings_file.exists():
        try:
            xl = pd.ExcelFile(settings_file)
            if sheet in xl.sheet_names:
                df, source = pd.read_excel(settings_file, sheet_name=sheet), f"{settings_file.name}!{sheet}"
        except Exception as exc:  # unreadable workbook -> fall back, but say so
            source = f"template (Settings.xlsx unreadable: {type(exc).__name__})"
    if df is None:
        df = pd.read_csv(template) if template.exists() else pd.DataFrame(columns=COLUMNS)
        source = source or f"template {template.name}"
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS].copy()
    for c in ("Country", "Region", "Metric", "Status", "Owner", "Rationale", "Source",
              "Valid_From", "Valid_To", "Approved_On"):
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["Region"] = df["Region"].replace("", "*")
    df["Metric"] = df["Metric"].replace("", "*")
    df["Multiplier"] = pd.to_numeric(df["Multiplier"], errors="coerce")
    return df, source


def validate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    reasons = []
    for _, r in out.iterrows():
        if r["Status"].lower() != "approved":
            reasons.append(f"Not applied: status '{r['Status'] or 'blank'}'")
        elif not r["Owner"] or not r["Approved_On"]:
            reasons.append("Not applied: missing Owner or Approved_On")
        elif not np.isfinite(r["Multiplier"]) or not (MULT_RANGE[0] <= r["Multiplier"] <= MULT_RANGE[1]):
            reasons.append(f"Not applied: multiplier outside {MULT_RANGE}")
        else:
            reasons.append("Applied")
    out["Applied_Status"] = reasons
    return out


def _period(s: str, default: pd.Timestamp) -> pd.Timestamp:
    if not s:
        return default
    return pd.Timestamp(s[:7] + "-01")


def multipliers(rows: pd.DataFrame, rules: pd.DataFrame) -> np.ndarray:
    """Multiplier for every forecast row (Country, Region, Metric, Date) — product of matching rules."""
    mult = np.ones(len(rows))
    ok = rules[rules["Applied_Status"] == "Applied"]
    for _, r in ok.iterrows():
        lo = _period(r["Valid_From"], pd.Timestamp.min)
        hi = _period(r["Valid_To"], pd.Timestamp.max)
        m = ((rows["Country"] == r["Country"])
             & ((r["Region"] == "*") | (rows["Region"] == r["Region"]))
             & ((r["Metric"] == "*") | (rows["Metric"] == r["Metric"]))
             & (rows["Date"] >= lo) & (rows["Date"] <= hi)).to_numpy()
        mult[m] *= float(r["Multiplier"])
    return mult
