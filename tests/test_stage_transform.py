import pandas as pd

import transform_data as td


def test_stage_summary_maps_ids_and_skips_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(td, "STAGE_PATH", tmp_path / "monthly_stage_summary.xlsx")
    monkeypatch.setattr(td, "STAGE_FINAL_PATH", tmp_path / "stage_final_summary.xlsx")
    monkeypatch.setattr(td, "SETTINGS_PATH", tmp_path / "Settings.xlsx")
    assert td.generate_stage_summary() is False                      # no extract yet -> skipped quietly

    with pd.ExcelWriter(tmp_path / "Settings.xlsx") as w:
        pd.DataFrame({"CountryId": [1, 2], "Country Grouped": ["UK", "UK"]}).to_excel(w, sheet_name="Country", index=False)
        pd.DataFrame({"id": [10, 11], "UK": ["North", "South"]}).to_excel(w, sheet_name="Region", index=False)
    pd.DataFrame({"ReportMonth": ["2026-10"] * 4, "CountryId": [1, 2, 1, 9], "TagGroupID": [10, 10, 11, 10],
                  "StageGroup": ["Offer", "Offer", "Deposit", "Offer"], "Apps": [5, 7, 3, 2]}).to_excel(
        tmp_path / "monthly_stage_summary.xlsx", index=False)
    assert td.generate_stage_summary() is True
    out = pd.read_excel(tmp_path / "stage_final_summary.xlsx", dtype={"Month": str})
    north_offer = out[(out["Country Grouped"] == "UK") & (out["Name of Region"] == "North") & (out["StageGroup"] == "Offer")]
    assert north_offer["Apps"].iloc[0] == 12                        # two CountryIds of the same group summed
    assert (out["Country Grouped"] == "Other").any()                 # unmapped CountryId kept as 'Other'
