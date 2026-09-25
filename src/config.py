# -*- coding: utf-8 -*-
"""
Global configuration for RMG pipeline.
- Cross-platform paths
- Safe environment variable overrides
- Reproducibility knobs (seed, horizon, etc.)
- One place to tweak run-level settings
"""

from __future__ import annotations

import os
import datetime as dt
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────
def _env_bool(name: str, default: bool) -> bool:
    """Read boolean env var with common truthy/falsey strings."""
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)

def _mk(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    return parent

# ─────────────────────────────────────────────────────────────────────
# ROOTS & DATA LOCATIONS
#   You can override these via environment variables for CI/servers:
#   - RMG_DATA_ROOT
#   - RMG_OUT_DIR
# ─────────────────────────────────────────────────────────────────────
# Repo root (…/src/..)
CODE_ROOT: Path = Path(__file__).resolve().parents[1]

# Data root (user default, but overrideable)
# Adjusted to be relative to the project root
DEFAULT_DATA_ROOT = Path(r"C:\Users\gsakhare\OneDrive - KC OVERSEAS EDUCATION PVT LTD\RMG\RMG Calculations")
DATA_ROOT: Path = Path(_env_str("RMG_DATA_ROOT", str(DEFAULT_DATA_ROOT)))

# Input files (keep names centralized)
RAW_FILE:  Path = DATA_ROOT / "monthly_final_summary.xlsx"
EXOG_FILE: Path = DATA_ROOT / "external_regressors.xlsx"

# ─────────────────────────────────────────────────────────────────────
# RUN OUTPUT FOLDERS
# - Default OUT_DIR → {DATA_ROOT}/Predictions/{YYYY-MM-DD}_{tag}
# - You can fully override with RMG_OUT_DIR
# - Optional RMG_RUN_TAG to append a short suffix (e.g., "debug", "A/B")
# ─────────────────────────────────────────────────────────────────────
def get_out_dir(base: Optional[Path] = None, tag: Optional[str] = None) -> Path:
    base = base or (DATA_ROOT / "Predictions")
    date_str = dt.date.today().isoformat()
    suffix = f"_{tag}" if tag else ""
    return base / f"{date_str}{suffix}"

RUN_TAG = _env_str("RMG_RUN_TAG", "").strip() or None

# Highest priority: explicit override
_out_override = os.environ.get("RMG_OUT_DIR")
if _out_override:
    OUT_DIR: Path = Path(_out_override)
else:
    OUT_DIR: Path = get_out_dir(tag=RUN_TAG)

_mk(OUT_DIR)

# Convenience subfolders used by several modules
MODELS_DIR: Path      = _mk(OUT_DIR / "Models")
EVAL_DIR: Path        = _mk(OUT_DIR / "Evaluation")
FINAL_FC_DIR: Path    = _mk(OUT_DIR / "Final_Forecast")
EDA_DIR: Path         = _mk(OUT_DIR / "EDA")          # EDA artefacts (mom_bands, need_diff, etc.)
REPORTS_DIR: Path     = _mk(OUT_DIR / "Reports")      # Excel & PDF outputs root
PDF_REPORTS_DIR: Path = _mk(REPORTS_DIR / "PDF")      # per-country and org-level PDFs

# ─────────────────────────────────────────────────────────────────────
# RUN-LEVEL SWITCHES (all env-overridable)
#   RMG_FREQ, RMG_FIN_YEAR_START, RMG_HORIZON, RMG_RANDOM_SEED,
#   RMG_ENABLE_EXOG, RMG_PARALLEL_JOBS
# ─────────────────────────────────────────────────────────────────────
FREQ: str           = _env_str("RMG_FREQ", "MS")           # monthly start (do not change casually)
FIN_YEAR_START: int = _env_int("RMG_FIN_YEAR_START", 4)    # April–March FY
HORIZON: int        = _env_int("RMG_HORIZON", 12)          # months ahead
RANDOM_SEED: int    = _env_int("RMG_RANDOM_SEED", 42)
ENABLE_EXOG: bool   = _env_bool("RMG_ENABLE_EXOG", False)  # flip on when regressors are clean
PARALLEL_JOBS: int  = _env_int("RMG_PARALLEL_JOBS", -1)    # use all logical CPUs

# ─────────────────────────────────────────────────────────────────────
# Loader/report-adjacent toggles (read in other modules)
# (These are *documented* here but read where relevant.)
#   RMG_INCLUDE_CURRENT_MONTH=1  → keep the in-progress month in loader
#   RMG_CLIP_NONNEG=0/1          → clip negatives in core metrics to 0
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# Handy utility for fiscal-year labeling (shared by models)
# ─────────────────────────────────────────────────────────────────────
def fiscal_year(ts) -> int:
    """Return fiscal year number given FIN_YEAR_START (e.g., 4 for April)."""
    # Accepts pandas Timestamp or datetime.date
    month = ts.month
    year  = ts.year
    return year if month < FIN_YEAR_START else year + 1

# ─────────────────────────────────────────────────────────────────────
# Sanity hints (non-fatal): useful when files move
# ─────────────────────────────────────────────────────────────────────
if not RAW_FILE.exists():
    # Do not raise here; loader.py can provide clearer error messages.
    print(f"[config] NOTE: RAW_FILE not found at {RAW_FILE}")
if ENABLE_EXOG and not EXOG_FILE.exists():
    print(f"[config] NOTE: EXOG_FILE not found at {EXOG_FILE} (ENABLE_EXOG=True)")
