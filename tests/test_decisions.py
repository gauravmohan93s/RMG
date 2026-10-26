from src.governance.decisions import DecisionLock


def test_choice_held_between_reviews_and_switched_in_review_month(tmp_path):
    p = tmp_path / "d.json"
    d = DecisionLock(p, "2026-10-28")                      # October = review month, first decision
    assert d.decide("recon", "bottom_up", {"bottom_up": 16.9, "wls_var": 17.4}) == "bottom_up"
    d.save()
    d = DecisionLock(p, "2026-11-28")                      # off-cycle, small edge for wls -> held
    assert d.decide("recon", "wls_var", {"bottom_up": 17.0, "wls_var": 16.7}) == "bottom_up"
    d.save()
    d = DecisionLock(p, "2026-12-28")                      # off-cycle, overwhelming evidence -> switch
    assert d.decide("recon", "wls_var", {"bottom_up": 18.0, "wls_var": 16.0}) == "wls_var"
    d.save()
    d = DecisionLock(p, "2027-01-28")                      # review: a near-tie does not switch back
    assert d.decide("recon", "bottom_up", {"bottom_up": 16.5, "wls_var": 16.6}) == "wls_var"
    log = d.save()
    assert list(log["Component"]) == ["recon"] and "kept" in log["Reason"].iloc[0]


def test_review_switches_on_a_clear_margin_once_per_quarter(tmp_path):
    p = tmp_path / "d.json"
    d = DecisionLock(p, "2026-09-28")
    d.decide("recon", "bottom_up", {"bottom_up": 14.4, "middle_out": 14.6})
    d.save()
    d = DecisionLock(p, "2026-10-26")                      # review: 2% better is noise -> kept
    assert d.decide("recon", "middle_out", {"bottom_up": 14.4, "middle_out": 14.1}) == "bottom_up"
    d.save()
    d = DecisionLock(p, "2026-10-27")                      # re-run in the same quarter: 4% is not enough off-cycle
    assert d.decide("recon", "middle_out", {"bottom_up": 14.6, "middle_out": 14.0}) == "bottom_up"
    d.save()
    d = DecisionLock(p, "2027-01-26")                      # next review: 4% better -> switch
    assert d.decide("recon", "middle_out", {"bottom_up": 14.6, "middle_out": 14.0}) == "middle_out"
