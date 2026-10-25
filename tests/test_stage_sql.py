"""The stage extract SQL, executed on synthetic history in SQLite (T-SQL specifics translated)."""
import re
import sqlite3
from datetime import datetime

import pandas as pd

import fetch_data as fd


def _to_sqlite(sql: str) -> str:
    sql = re.sub(r"DATEADD\(MINUTE,330,([\w\.]+)\)", r"datetime(\1, '+330 minutes')", sql)
    sql = sql.replace("SELECT * FROM (VALUES", "SELECT column1 AS StageGroup, column2 AS CountryId, column3 AS StageId FROM (VALUES")
    return re.sub(r"\)\s*v\(StageGroup, CountryId, StageId\)", ")", sql)


def _run(history, staff, month="2026-10"):
    con = sqlite3.connect(":memory:")
    pd.DataFrame(history, columns=["AcknowledgementId", "ApplicationStageId", "ChangedOn"]).to_sql("ApplicationHistory", con)
    pd.DataFrame(staff, columns=["AcknowledgementId", "CountryId", "TagGroupID"]).to_sql("Application_View_Staff", con)
    # cycle 26 Sep - 25 Oct 2026 in IST; ChangedOn is stored in UTC (IST = UTC + 5:30)
    sql = fd.build_stage_sql(datetime(2026, 9, 26), datetime(2026, 10, 25, 23, 59, 59), {}, [4, 81], month)
    out = pd.read_sql(_to_sqlite(sql), con)
    return {(int(r.CountryId), r.StageGroup): int(r.Apps) for r in out.itertuples()}


def test_first_reach_per_milestone_with_country_specific_documents():
    history = [
        ("A", 5, "2026-08-10 10:00:00"),     # UK: offer in August (outside the cycle)
        ("A", 17, "2026-10-05 10:00:00"),    # CAS received -> deposit + UK visa document
        ("A", 20, "2026-10-20 10:00:00"),    # visa received -> visa applied + granted
        ("A", 5, "2026-10-21 10:00:00"),     # re-enters an offer status: not a new offer
        ("B", 18, "2026-10-01 10:00:00"),    # USA: I-20 received -> offer + USA visa document
        ("C", 18, "2026-10-02 10:00:00"),    # other country: I-20 counts as a visa document by default
        ("D", 18, "2026-10-03 10:00:00"),    # UK: I-20 status is NOT the UK visa document (CAS is)
    ]
    staff = [("A", 4, 1), ("B", 1, 1), ("C", 9, 1), ("D", 4, 1)]
    c = _run(history, staff)
    assert (4, "Offer") in c and c[(4, "Offer")] == 1          # only D's first offer is in October
    assert c[(4, "Deposit")] == 1 and c[(4, "Visa_Document")] == 1 and c[(4, "Visa_Granted")] == 1
    assert c[(1, "Offer")] == 1 and c[(1, "Visa_Document")] == 1 and (1, "Deposit") not in c
    assert c[(9, "Visa_Document")] == 1
    assert c[(4, "Visa_Applied")] == 1
