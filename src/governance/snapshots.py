# -*- coding: utf-8 -*-
"""
Extract snapshots and data revisions (late entries).

If applications are entered or back-dated after the 25th cut-off, the latest month is under-counted at
fetch time. Every model then starts from a number that is too low, and the actual "comes in higher"
later: a systematic under-forecast that no model can see. Each run stores the extract it used (one file
per last-actual month). Later runs compare how the same months were counted then and now:

    Revision_Pct = 100 * (count now / count when that month was the latest) - 1

Once at least MIN_EVIDENCE months have been revised, the typical uplift per country and metric is
reported (Data Revisions sheet). The latest month is flagged when its uplift is material. The
forecast is not adjusted automatically: a completeness adjustment would be a separate, tested component.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("Received", "Submitted")
MIN_EVIDENCE = 3
MATERIAL_PCT = 3.0


def save(raw: pd.DataFrame, folder: Path, last_actual: pd.Timestamp) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"extract_{pd.Timestamp(last_actual):%Y-%m}.parquet"
    raw[["Country", "Region", "Date", *METRICS]].to_parquet(p, index=False)
    return p


def revisions(raw: pd.DataFrame, folder: Path, last_actual: pd.Timestamp) -> pd.DataFrame:
    """How each earlier 'latest month' was counted then vs in today's extract (by country)."""
    folder = Path(folder)
    cols = ["Country", "Metric", "Month", "First_Count", "Latest_Count", "Revision_Pct"]
    if not folder.exists():
        return pd.DataFrame(columns=cols)
    now = raw.groupby(["Country", "Date"])[list(METRICS)].sum()
    rows = []
    for p in sorted(folder.glob("extract_*.parquet")):
        m = pd.Timestamp(p.stem.split("_", 1)[1] + "-01")
        if m >= pd.Timestamp(last_actual):
            continue
        old = pd.read_parquet(p)
        old = old[pd.to_datetime(old["Date"]) == m].groupby("Country")[list(METRICS)].sum()
        for cty, r in old.iterrows():
            for met in METRICS:
                first = float(r[met])
                latest = float(now[met].get((cty, m), np.nan)) if (cty, m) in now.index else np.nan
                if first > 0 and np.isfinite(latest):
                    rows.append({"Country": cty, "Metric": met, "Month": m, "First_Count": first,
                                 "Latest_Count": latest, "Revision_Pct": 100 * (latest / first - 1)})
    return pd.DataFrame(rows, columns=cols)


def summary(rev: pd.DataFrame) -> pd.DataFrame:
    """Typical late-entry uplift per country x metric (median over revised months)."""
    if rev is None or rev.empty:
        return pd.DataFrame(columns=["Country", "Metric", "Months_Evidence", "Typical_Revision_Pct", "Material"])
    g = rev.groupby(["Country", "Metric"])["Revision_Pct"]
    s = pd.DataFrame({"Months_Evidence": g.size(), "Typical_Revision_Pct": g.median()}).reset_index()
    s["Material"] = (s["Months_Evidence"] >= MIN_EVIDENCE) & (s["Typical_Revision_Pct"].abs() >= MATERIAL_PCT)
    return s
