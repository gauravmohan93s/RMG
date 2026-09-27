"""End-to-end smoke test on synthetic data: engine + evidence + reports."""
import json

import numpy as np
import pandas as pd

from src import __version__
from src.forecasting.engine import ARTEFACTS, load_result, run_forecast
from src.reports import build_reports


def test_end_to_end(raw, settings, tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    hist = tmp_path / "history.parquet"
    kw = dict(raw_file=tmp_path / "none.xlsx", settings_file=tmp_path / "none.xlsx", history_path=hist, raw=raw)
    res = run_forecast(settings, out, "2026-01-15", **kw)
    fc = res.forecast

    # invariants and coherence
    assert all(v == 0 for v in res.manifest["invariants"].values())
    assert res.manifest["coherence_max_gap"] < 1e-6
    assert (fc["Date"] > res.last_actual).all()
    assert not fc["Region"].eq("Closed").any()
    for L in (80, 95):
        assert ((fc[f"Lo{L}"] <= fc["Forecast"] + 1e-6) & (fc["Forecast"] <= fc[f"Hi{L}"] + 1e-6)).all()
    # single-region country gets a country row; every node has 12 months
    assert ((fc["Level"] == "Country") & (fc["Country"] == "Beta")).any()
    assert (fc.groupby(["Level", "Country", "Region", "Metric"]).size() == 12).all()
    # evidence
    assert res.selection["Selection_Reason"].str.len().gt(20).all()
    assert not res.policy_comparison.empty and not res.accuracy.empty
    org = res.totals[(res.totals["Level"] == "Organisation") & (res.totals["Window"] == "Next 12 months")]
    assert ((org["Lo80"] <= org["Forecast"]) & (org["Forecast"] <= org["Hi80"])).all()
    man = json.loads((out / "run_manifest.json").read_text())
    assert man["pipeline_version"] == __version__ and "config" in man and "git" in man
    for rel in ARTEFACTS.values():
        assert (out / rel).exists(), rel

    # reports from the saved run (the `report` CLI path)
    paths = build_reports(load_result(out))
    assert all(p.exists() and p.stat().st_size > 5_000 for p in paths.values())

    # idempotent history: rerunning the same date replaces, not duplicates
    n1 = len(pd.read_parquet(hist))
    run_forecast(settings, out, "2026-01-15", **kw)
    assert len(pd.read_parquet(hist)) == n1
