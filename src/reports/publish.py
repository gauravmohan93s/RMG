# -*- coding: utf-8 -*-
"""
Publish a run's reports to the executive Outcomes folder.

  Outcomes/
    2026-10 Forecast Pack/          one folder per month, kept permanently (the monthly record)
      RMG_Monthly_Pack_<run>.pdf     leadership pack
      RMG_Forecast_<run>.xlsx        all numbers + evidence
      Human_Inputs_Form_<run>.xlsx   market input form
      Countries/<Country>.pdf        one page per destination
      README.txt
    Latest/                          always the most recent pack (same files, replaced each month)

The folder is named after the first forecast month: a pack built on actuals to September 2026 is the
"2026-10 Forecast Pack". Only report files are copied; the evidence stays in Predictions/<run-date>/.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict

import pandas as pd

README = """RMG application-flow forecast — {label}
Actuals to {actual:%B %Y}; forecast {start:%B %Y} to {end:%B %Y}. Run {run_date}, pipeline v{version}.

RMG_Monthly_Pack_*.pdf     The leadership pack: organisation, proof on the last quarter, direction check,
                           destinations, workload calendar, track record, one page per destination.
Countries/*.pdf            The destination page on its own, for country heads.
RMG_Forecast_*.xlsx        Every number (organisation, countries, regions, months), plus all the
                           evidence behind each method choice.
Human_Inputs_Form_*.xlsx   Market heads: fill the yellow cells where you expect something different and
                           return it to the forecasting team.

How to read the ranges: "most likely" = the actual lands inside it about 1 time in 2; "plan for" = about
8 times in 10. Reliability = how close quarter totals have typically been to what happened.
"""


def _copy(src: Path, dst: Path) -> None:
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def publish(res, report_dir: Path, outcomes_root: Path, version: str) -> Dict[str, Path]:
    report_dir, outcomes_root = Path(report_dir), Path(outcomes_root)
    files = [p for p in report_dir.iterdir() if p.suffix.lower() in (".pdf", ".xlsx") or p.is_dir()]
    if not files:
        raise FileNotFoundError(f"No reports to publish in {report_dir}")
    start, end = pd.Timestamp(res.forecast["Date"].min()), pd.Timestamp(res.forecast["Date"].max())
    label = f"{start:%Y-%m} Forecast Pack"
    text = README.format(label=label, actual=pd.Timestamp(res.last_actual), start=start, end=end,
                         run_date=res.run_date, version=version)
    out = {}
    for target in (outcomes_root / label, outcomes_root / "Latest"):
        if target.name == "Latest" and target.exists():
            shutil.rmtree(target)                       # Latest = exactly the newest pack, nothing stale
        target.mkdir(parents=True, exist_ok=True)
        for f in files:
            _copy(f, target / f.name)
        (target / "README.txt").write_text(text, encoding="utf-8")
        out[target.name] = target
    return out
