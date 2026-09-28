import pandas as pd

from src.governance import snapshots as snp


def _raw(months, sep_count):
    rows = []
    for m in months:
        v = sep_count if m == pd.Timestamp("2026-09-01") else 100.0
        rows.append({"Country": "UK", "Region": "R", "Date": m, "Received": v, "Submitted": v / 2})
    return pd.DataFrame(rows)


def test_late_entries_are_measured_from_saved_extracts(tmp_path):
    sep, octo = pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-01")
    snp.save(_raw([sep], 90.0), tmp_path, sep)                          # September as first reported
    now = _raw([sep, octo], 99.0)                                       # a month later: September grew 10%
    rev = snp.revisions(now, tmp_path, octo)
    r = rev[rev["Metric"] == "Received"].iloc[0]
    assert r["Month"] == sep and abs(r["Revision_Pct"] - 10.0) < 1e-9
    s = snp.summary(rev)
    assert not s["Material"].any()                                      # 1 month of evidence is not enough yet
