from types import SimpleNamespace

import pandas as pd

from src.reports.publish import publish


def _res():
    fc = pd.DataFrame({"Date": pd.date_range("2026-10-01", periods=12, freq="MS")})
    return SimpleNamespace(forecast=fc, last_actual=pd.Timestamp("2026-09-01"), run_date="2026-09-28")


def test_publish_creates_monthly_folder_and_replaces_latest(tmp_path):
    rep = tmp_path / "Reports"
    (rep / "Countries").mkdir(parents=True)
    (rep / "RMG_Monthly_Pack_2026-09-28.pdf").write_bytes(b"pdf")
    (rep / "Countries" / "United_Kingdom.pdf").write_bytes(b"uk")
    out = tmp_path / "Outcomes"
    (out / "Latest").mkdir(parents=True)
    (out / "Latest" / "old_pack.pdf").write_bytes(b"stale")
    paths = publish(_res(), rep, out, "2.2.0")
    month = out / "2026-10 Forecast Pack"
    assert paths["2026-10 Forecast Pack"] == month
    assert (month / "RMG_Monthly_Pack_2026-09-28.pdf").exists() and (month / "Countries" / "United_Kingdom.pdf").exists()
    assert "Actuals to September 2026" in (month / "README.txt").read_text(encoding="utf-8")
    assert not (out / "Latest" / "old_pack.pdf").exists()          # Latest holds only the newest pack
    assert (out / "Latest" / "RMG_Monthly_Pack_2026-09-28.pdf").exists()
