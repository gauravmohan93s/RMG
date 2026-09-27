import numpy as np

from src.data.prepare import ORG, TOTAL, node_id


def _status(panel, region):
    return panel.series_status.set_index("Region").loc[region, "Status"]


def test_missing_month_is_zero_not_interpolated(panel):
    y = panel.series("Received|Alpha|South")
    assert y.iloc[20] == 0.0
    assert panel.series_status.set_index("Region").loc["South", "Zero_Filled_Months"] == 1


def test_series_status(panel):
    assert _status(panel, "Closed") == "Inactive"
    assert _status(panel, "New") == "Emerging"
    assert _status(panel, "North") == "Mature"
    assert not panel.nodes["Region"].eq("Closed").any()          # inactive series are not forecast


def test_hierarchy_sums(panel):
    for m in ("Received", "Submitted"):
        org = panel.series(node_id(m, ORG, TOTAL))
        alpha = panel.series(node_id(m, "Alpha", TOTAL))
        parts = sum(panel.series(node_id(m, "Alpha", r)).reindex(alpha.index, fill_value=0)
                    for r in ("North", "South", "New"))
        assert np.allclose(alpha.values, parts.values)
        beta = panel.series(node_id(m, "Beta", "All")).reindex(org.index, fill_value=0)
        assert np.allclose(org.values, alpha.reindex(org.index, fill_value=0).values + beta.values)
    # a country with one active region has no separate country node
    assert node_id("Received", "Beta", TOTAL) not in panel.y


def test_history_keeps_inactive_series(panel):
    h = panel.history
    assert (h["Region"] == "Closed").any()
    assert set(h["Level"]) == {"Region", "Country", "Organisation"}
