from src.governance.decisions import DecisionLock


def test_choice_held_between_reviews_and_switched_in_review_month(tmp_path):
    p = tmp_path / "d.json"
    d = DecisionLock(p, "2026-10-28")                      # October = review month
    assert d.decide("recon", "bottom_up", {"bottom_up": 16.9, "wls_var": 17.4}) == "bottom_up"
    d.save()
    d = DecisionLock(p, "2026-11-28")                      # off-cycle, small edge for wls -> held
    assert d.decide("recon", "wls_var", {"bottom_up": 17.0, "wls_var": 16.7}) == "bottom_up"
    d.save()
    d = DecisionLock(p, "2026-12-28")                      # off-cycle, overwhelming evidence -> switch
    assert d.decide("recon", "wls_var", {"bottom_up": 18.0, "wls_var": 16.0}) == "wls_var"
    d.save()
    d = DecisionLock(p, "2027-01-28")                      # review month -> evidence adopted
    assert d.decide("recon", "bottom_up", {"bottom_up": 16.5, "wls_var": 16.6}) == "bottom_up"
    log = d.save()
    assert list(log["Component"]) == ["recon"] and "review month" in log["Reason"].iloc[0]
