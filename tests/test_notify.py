from datetime import date, datetime
from email.message import EmailMessage
from types import SimpleNamespace

import pandas as pd
import pytest

from src.reports import notify


def test_reminders_only_for_stale_or_unreviewed_sheets():
    status = pd.DataFrame([
        {"Country": "UK", "Label": "United Kingdom", "File": "United_Kingdom.xlsx", "Views": 3, "FTE": 2,
         "Last_Reviewed": datetime(2026, 10, 20)},                                # fresh: no reminder
        {"Country": "Canada", "Label": "Canada", "File": "Canada.xlsx", "Views": 0, "FTE": 0, "Last_Reviewed": None},
        {"Country": "Asia", "Label": "Asia", "File": "Asia.xlsx", "Views": 1, "FTE": 0,
         "Last_Reviewed": datetime(2026, 7, 1)}])                                 # stale, but no contact
    contacts = pd.DataFrame({"Country": ["UK", "Canada", "Canada"], "Email": ["a@x.com", "b@x.com", "c@x.com"]})
    msgs = notify.reminder_emails(status, contacts, "Forms/Live", date(2026, 10, 26))
    assert len(msgs) == 1 and msgs[0]["To"] == "b@x.com, c@x.com" and "not reviewed yet" in msgs[0].get_content()


def test_nothing_is_sent_while_switched_off(tmp_path, monkeypatch):
    def summary(res, cfg, pack, outcomes):
        m = EmailMessage()
        m["Subject"] = "pack"
        m.set_content("x")
        return m
    monkeypatch.setattr(notify, "summary_email", summary)
    monkeypatch.setattr(notify, "_send", lambda msgs: pytest.fail("must not send while disabled"))
    res = SimpleNamespace(out_dir=tmp_path)
    r = notify.run(res, {"enabled": False, "to": ["boss@x.com"]}, None, tmp_path, tmp_path / "Live", None)
    assert r["sent"] == 0 and r["built"] == 1 and not r["enabled"]
    assert (tmp_path / "Reports" / "Email_preview" / "00_summary.eml").exists()     # preview always written
