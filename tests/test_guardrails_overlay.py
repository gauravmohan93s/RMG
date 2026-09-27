import numpy as np
import pandas as pd
import pytest

from src.forecasting import guardrails as gr
from src.forecasting import overlay as ov


def _base(nid, monthly):
    return pd.DataFrame({"node_id": nid, "h": np.arange(1, 13), "Pred": monthly})


def test_explosion_is_capped_but_genuine_growth_passes():
    flat = np.full(24, 100.0)
    growing = np.concatenate([np.full(12, 100.0), np.linspace(150, 400, 12)])
    refs = pd.DataFrame({"flat": gr.references(flat), "grow": gr.references(growing)}).T
    base = pd.concat([_base("flat", np.full(12, 1000.0)),        # 10x last year: model failure
                      _base("grow", np.full(12, 400.0))])        # continues recent momentum
    g = gr.plausibility(base, refs, 0.5, 1.8).set_index("node_id")
    assert g.loc["flat", "Guardrail_Triggered"] and not g.loc["grow", "Guardrail_Triggered"]
    assert np.isclose(g.loc["flat", "Forecast_12m"] * g.loc["flat", "Scale_Applied"], g.loc["flat", "Upper_Bound"])


def test_invariants_catch_point_outside_interval():
    fc = pd.DataFrame({"Forecast": [10.0], "Lo80": [11.0], "Hi80": [12.0], "Lo95": [9.0], "Hi95": [13.0],
                       "Date": [pd.Timestamp("2027-01-01")]})
    with pytest.raises(AssertionError):
        gr.check_invariants(fc, pd.Timestamp("2026-12-01"))


def rules(**kw):
    base = {"Country": "Canada", "Region": "*", "Metric": "*", "Multiplier": 0.8, "Valid_From": "2027-01",
            "Valid_To": "2027-03", "Owner": "Head of Canada", "Rationale": "cap", "Source": "IRCC",
            "Status": "Approved", "Approved_On": "2026-09-01"}
    base.update(kw)
    return ov.validate(pd.DataFrame([base]))


def test_overlay_governance():
    assert rules(Status="Draft")["Applied_Status"].iloc[0].startswith("Not applied")
    assert rules(Owner="")["Applied_Status"].iloc[0].startswith("Not applied")
    assert rules(Multiplier=3.0)["Applied_Status"].iloc[0].startswith("Not applied")
    assert rules()["Applied_Status"].iloc[0] == "Applied"


def test_overlay_applies_only_inside_validity_window():
    rows = pd.DataFrame({"Country": ["Canada"] * 3 + ["UK"], "Region": ["North"] * 4, "Metric": ["Received"] * 4,
                         "Date": pd.to_datetime(["2026-12-01", "2027-02-01", "2027-04-01", "2027-02-01"])})
    assert ov.multipliers(rows, rules()).tolist() == [1.0, 0.8, 1.0, 1.0]


def test_template_loads_as_draft(tmp_path):
    from src.config import CODE_ROOT
    df, src = ov.load_overlay(tmp_path / "missing.xlsx", "Policy_Overlay", CODE_ROOT / "config" / "policy_overlay_template.csv")
    v = ov.validate(df)
    assert "template" in src and (v["Applied_Status"] != "Applied").all()
