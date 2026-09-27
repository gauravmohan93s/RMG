# -*- coding: utf-8 -*-
"""
Run configuration.

Values come from ``config/pipeline.toml`` (override the file with env ``RMG_CONFIG``).
Paths can be overridden with ``RMG_DATA_ROOT`` and ``RMG_OUT_DIR``.
Importing this module has no side effects (no folders are created).
"""
from __future__ import annotations

import datetime as dt
import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

CODE_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_FILE: Path = Path(os.environ.get("RMG_CONFIG", CODE_ROOT / "config" / "pipeline.toml"))
DEFAULT_DATA_ROOT = Path(r"C:\Users\gsakhare\OneDrive - KC OVERSEAS EDUCATION PVT LTD\RMG\RMG Calculations")


def _read_toml(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


_RAW: Dict[str, Any] = _read_toml(CONFIG_FILE)

DATA_ROOT: Path = Path(
    os.environ.get("RMG_DATA_ROOT") or _RAW["paths"].get("data_root") or DEFAULT_DATA_ROOT
)
RAW_FILE: Path = DATA_ROOT / _RAW["paths"]["raw_file"]
INTAKE_FILE: Path = DATA_ROOT / _RAW["paths"].get("intake_file", "intake_final_summary.xlsx")
SETTINGS_FILE: Path = DATA_ROOT / _RAW["paths"]["settings_file"]
PREDICTIONS_ROOT: Path = DATA_ROOT / "Predictions"


@dataclass(frozen=True)
class DataCfg:
    drop_regions: Tuple[str, ...]
    inactive_months: int
    min_months_for_tournament: int
    fin_year_start_month: int


@dataclass(frozen=True)
class BacktestCfg:
    horizon: int
    n_origins: int


@dataclass(frozen=True)
class ModelCfg:
    enabled: Tuple[str, ...]
    lgbm: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionCfg:
    policy: str
    benchmark: str
    top_k: int
    must_beat_benchmark: bool
    honest_eval_origins: int
    fallback_if_skill_below: float


@dataclass(frozen=True)
class IntervalCfg:
    levels: Tuple[int, ...]
    min_obs: int
    calibrate: bool
    max_offset: float


@dataclass(frozen=True)
class GuardCfg:
    min_ratio: float
    max_ratio: float


@dataclass(frozen=True)
class BiasCfg:
    enabled: bool
    window: int
    max_h: int
    min_gain_pct: float


@dataclass(frozen=True)
class SeasonalityCfg:
    high_threshold: float
    low_threshold: float
    years: int
    apps_per_fte_month: float


@dataclass(frozen=True)
class Settings:
    data: DataCfg
    backtest: BacktestCfg
    models: ModelCfg
    selection: SelectionCfg
    intervals: IntervalCfg
    reconciliation_method: str
    guardrails: GuardCfg
    overlay_sheet: str
    overlay_template: Path
    bias_correction: BiasCfg
    seasonality: SeasonalityCfg
    random_seed: int
    n_jobs: int

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["overlay_template"] = str(self.overlay_template)
        return d


def load_settings(raw: Optional[Dict[str, Any]] = None) -> Settings:
    """Build typed settings from the TOML dict (or a dict passed in tests)."""
    r = raw if raw is not None else _RAW
    if int(r["backtest"]["n_origins"]) < int(r["backtest"]["horizon"]):
        raise ValueError("backtest.n_origins must be >= backtest.horizon so every horizon is evaluated")
    if r["selection"].get("policy", "mean_all") not in ("mean_all", "mean_screened", "top_k"):
        raise ValueError("selection.policy must be 'mean_all', 'mean_screened' or 'top_k'")
    return Settings(
        data=DataCfg(
            drop_regions=tuple(r["data"]["drop_regions"]),
            inactive_months=int(r["data"]["inactive_months"]),
            min_months_for_tournament=int(r["data"]["min_months_for_tournament"]),
            fin_year_start_month=int(r["data"]["fin_year_start_month"]),
        ),
        backtest=BacktestCfg(
            horizon=int(r["backtest"]["horizon"]),
            n_origins=int(r["backtest"]["n_origins"]),
        ),
        models=ModelCfg(
            enabled=tuple(r["models"]["enabled"]),
            lgbm=dict(r["models"].get("lgbm", {})),
        ),
        selection=SelectionCfg(
            policy=str(r["selection"].get("policy", "mean_all")),
            benchmark=str(r["selection"]["benchmark"]),
            top_k=int(r["selection"]["top_k"]),
            must_beat_benchmark=bool(r["selection"]["must_beat_benchmark"]),
            honest_eval_origins=int(r["selection"]["honest_eval_origins"]),
            fallback_if_skill_below=float(r["selection"]["fallback_if_skill_below"]),
        ),
        intervals=IntervalCfg(
            levels=tuple(int(x) for x in r["intervals"]["levels"]),
            min_obs=int(r["intervals"]["min_obs"]),
            calibrate=bool(r["intervals"]["calibrate"]),
            max_offset=float(r["intervals"]["max_offset"]),
        ),
        reconciliation_method=str(r["reconciliation"]["method"]),
        guardrails=GuardCfg(
            min_ratio=float(r["guardrails"]["min_ratio"]),
            max_ratio=float(r["guardrails"]["max_ratio"]),
        ),
        overlay_sheet=str(r["overlay"]["sheet"]),
        overlay_template=CODE_ROOT / r["overlay"]["template"],
        bias_correction=BiasCfg(
            enabled=bool(r.get("statistical_overlay", {}).get("enabled", True)),
            window=int(r.get("statistical_overlay", {}).get("window_months", 6)),
            max_h=int(r.get("statistical_overlay", {}).get("max_horizon", 3)),
            min_gain_pct=float(r.get("statistical_overlay", {}).get("min_gain_pct", 1.0)),
        ),
        seasonality=SeasonalityCfg(
            high_threshold=float(r.get("seasonality", {}).get("high_threshold", 1.10)),
            low_threshold=float(r.get("seasonality", {}).get("low_threshold", 0.90)),
            years=int(r.get("seasonality", {}).get("years", 4)),
            apps_per_fte_month=float(r.get("seasonality", {}).get("apps_per_fte_month", 0)),
        ),
        random_seed=int(r["runtime"]["random_seed"]),
        n_jobs=int(r["runtime"]["n_jobs"]),
    )


def run_dir(run_date: Optional[str] = None) -> Path:
    """Output folder for a run: ``RMG_OUT_DIR`` or ``Predictions/<YYYY-MM-DD>``."""
    override = os.environ.get("RMG_OUT_DIR")
    if override:
        return Path(override)
    d = dt.date.fromisoformat(run_date) if run_date else dt.date.today()
    return PREDICTIONS_ROOT / d.isoformat()
