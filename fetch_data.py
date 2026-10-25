"""
Fetch monthly application extracts from SQL Server (26th-25th reporting cycles).

Two incremental extracts (only fully closed, not-yet-fetched months are queried):
  monthly_app_summary.xlsx     ReportMonth, CountryId, TagGroupID, Received/Submitted apps & students
                               -> the forecasting input (unchanged)
  monthly_intake_summary.xlsx  ReportMonth, CountryId, TagGroupID, Intake, InYear, Received/Submitted apps
                               -> descriptive intake view (which intake each month's applications target)
  monthly_stage_summary.xlsx   ReportMonth, CountryId, TagGroupID, StageGroup, Apps (v2.3, off until
                               STAGE_GROUPS is filled): applications reaching each later stage (offer,
                               deposit, visa document...) for the FIRST time in the cycle

Find the stage IDs first:  python fetch_data.py --discover-stages   (writes stage_discovery.xlsx)

Credentials come ONLY from environment variables (or a local, git-ignored `.env` file in the
repo root): DB_SERVER, DB_NAME, DB_USERNAME, DB_PASSWORD. Nothing secret is stored in code.
"""
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import pytz

try:
    from src.config import DATA_ROOT
except Exception:
    DATA_ROOT = Path(r"C:\Users\gsakhare\OneDrive - KC OVERSEAS EDUCATION PVT LTD\RMG\RMG Calculations")

# -----------------------------------------------------------
# 1) Operator inputs
# -----------------------------------------------------------
START_DATE_STR = '2020-01-01'   # YYYY-MM-DD (initial lower bound)
END_DATE_STR   = ''             # '' => up to yesterday (IST), capped at the last closed cycle
OUTPUT_PATH = Path(os.environ.get("RMG_OUTPUT_PATH", str(DATA_ROOT / "monthly_app_summary.xlsx")))
INTAKE_OUTPUT_PATH = Path(os.environ.get("RMG_INTAKE_OUTPUT_PATH", str(DATA_ROOT / "monthly_intake_summary.xlsx")))
FETCH_INTAKE = os.environ.get("RMG_FETCH_INTAKE", "1") != "0"

# Per-country "submitted" stage overrides
COUNTRY_SUBMITTED_STAGES = {
    # e.g. 1: [4],
    #       2: [4,81],
}
DEFAULT_SUBMITTED_STAGES = [4, 81]

# Later stages to extract (v2.3). Fill the ApplicationStageId lists from stage_discovery.xlsx.
# "*" = all countries; a CountryId key overrides "*" for that country (country-specific visa documents).
# Leave a group empty to skip it. Nothing is fetched while every list is empty.
STAGE_GROUPS = {
    "Offer":        {"*": []},          # conditional / unconditional offer
    "Deposit":      {"*": []},          # deposit / tuition fee paid
    "Visa_Document": {"*": []},         # e.g. {<UK CountryId>: [CAS id], <Australia>: [Funds/COE id],
                                        #       <New Zealand>: [AIP id], <USA>: [I-20 id]}
    "Visa_Outcome": {"*": []},          # visa granted
}
STAGE_OUTPUT_PATH = Path(os.environ.get("RMG_STAGE_OUTPUT_PATH", str(DATA_ROOT / "monthly_stage_summary.xlsx")))


# -----------------------------------------------------------
# 2) Helpers
# -----------------------------------------------------------
def month_ranges(start_date: date, end_date: date):
    """
    Yield (month_start_dt, month_end_dt, 'YYYY-MM') for each custom 26th→25th cycle.
    IMPORTANT: Only yield **fully completed cycles** (period_end <= end_date).
      For ReportMonth='2025-07', start=2025-06-26 00:00:00, end=2025-07-25 23:59:59.
    """
    year, month = start_date.year, start_date.month
    while True:
        prev_year, prev_month = (year - 1, 12) if month == 1 else (year, month - 1)
        period_start = date(prev_year, prev_month, 26)
        period_end = date(year, month, 25)
        if period_end < start_date:
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
            continue
        if period_end > end_date:
            print(f"⏭️  Skipping incomplete cycle {year:04d}-{month:02d} "
                  f"({period_start}→{period_end}); effective end date is {end_date}")
            break
        ms = datetime.combine(max(period_start, start_date), time(0, 0, 0))
        me = datetime.combine(period_end, time(23, 59, 59))
        yield ms, me, f"{year}-{month:02d}"
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def last_closed_cycle_end(today_ist: date) -> date:
    """Most recent fully closed 26th→25th cycle end (2026-02-26 -> 2026-02-25; 2026-02-25 -> 2026-01-25)."""
    if today_ist.day >= 26:
        return date(today_ist.year, today_ist.month, 25)
    if today_ist.month == 1:
        return date(today_ist.year - 1, 12, 25)
    return date(today_ist.year, today_ist.month - 1, 25)


def _stage_clause(submitted_stages_dict, default_stages) -> str:
    conditions = []
    for cid, stages in submitted_stages_dict.items():
        stage_list = ",".join(str(st) for st in stages)
        conditions.append(f"(avs.CountryId = {cid} AND ah.ApplicationStageId IN ({stage_list}))")
    default_stage_list = ",".join(str(st) for st in default_stages)
    if submitted_stages_dict:
        overridden = ",".join(str(cid) for cid in submitted_stages_dict.keys())
        conditions.append(f"(avs.CountryId NOT IN ({overridden}) AND ah.ApplicationStageId IN ({default_stage_list}))")
    else:
        conditions.append(f"ah.ApplicationStageId IN ({default_stage_list})")
    return " AND (" + " OR ".join(conditions) + ")"


def build_monthly_sql(start_dt, end_dt, submitted_stages_dict, default_stages, report_month):
    """Monthly aggregation over the 26→25 window; ReportMonth = month of the cycle end day."""
    s = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    e = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    stage_where_clause = _stage_clause(submitted_stages_dict, default_stages)
    return f"""
WITH
  Received AS (
    SELECT avs.CountryId, avs.TagGroupID,
      COUNT(DISTINCT avs.AcknowledgementNumber) AS ReceivedApps,
      COUNT(DISTINCT avs.StudentId)             AS ReceivedStudents
    FROM Application_View_Staff avs
    WHERE avs.CountryId IS NOT NULL
      AND DATEADD(MINUTE,330,avs.SubmittedDate) BETWEEN '{s}' AND '{e}'
    GROUP BY avs.CountryId, avs.TagGroupID
  ),
  Submitted AS (
    SELECT avs.CountryId, avs.TagGroupID,
      COUNT(DISTINCT ah.AcknowledgementId) AS SubmittedApps,
      COUNT(DISTINCT avs.StudentId)        AS SubmittedStudents
    FROM ApplicationHistory ah
    JOIN Application_View_Staff avs ON ah.AcknowledgementId = avs.AcknowledgementId
    WHERE avs.CountryId IS NOT NULL
      {stage_where_clause}
      AND DATEADD(MINUTE,330,ah.ChangedOn) BETWEEN '{s}' AND '{e}'
    GROUP BY avs.CountryId, avs.TagGroupID
  )
SELECT
  '{report_month}' AS ReportMonth,
  COALESCE(r.CountryId, s.CountryId)   AS CountryId,
  COALESCE(r.TagGroupID, s.TagGroupID) AS TagGroupID,
  ISNULL(r.ReceivedApps,0)      AS ReceivedApps,
  ISNULL(r.ReceivedStudents,0)  AS ReceivedStudents,
  ISNULL(s.SubmittedApps,0)     AS SubmittedApps,
  ISNULL(s.SubmittedStudents,0) AS SubmittedStudents
FROM (SELECT CountryId, TagGroupID FROM Received UNION SELECT CountryId, TagGroupID FROM Submitted) AS d
LEFT JOIN Received  r ON d.CountryId=r.CountryId AND d.TagGroupID=r.TagGroupID
LEFT JOIN Submitted s ON d.CountryId=s.CountryId AND d.TagGroupID=s.TagGroupID
ORDER BY CountryId, ReportMonth, TagGroupID;
"""


def build_intake_sql(start_dt, end_dt, submitted_stages_dict, default_stages, report_month):
    """Same windows and definitions as build_monthly_sql, additionally split by target Intake / InYear."""
    s = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    e = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    stage_where_clause = _stage_clause(submitted_stages_dict, default_stages)
    intake = "ISNULL(CAST(avs.Intake AS NVARCHAR(50)), '')"
    inyear = "ISNULL(CAST(avs.InYear AS NVARCHAR(10)), '')"
    return f"""
WITH
  Received AS (
    SELECT avs.CountryId, avs.TagGroupID, {intake} AS Intake, {inyear} AS InYear,
      COUNT(DISTINCT avs.AcknowledgementNumber) AS ReceivedApps
    FROM Application_View_Staff avs
    WHERE avs.CountryId IS NOT NULL
      AND DATEADD(MINUTE,330,avs.SubmittedDate) BETWEEN '{s}' AND '{e}'
    GROUP BY avs.CountryId, avs.TagGroupID, {intake}, {inyear}
  ),
  Submitted AS (
    SELECT avs.CountryId, avs.TagGroupID, {intake} AS Intake, {inyear} AS InYear,
      COUNT(DISTINCT ah.AcknowledgementId) AS SubmittedApps
    FROM ApplicationHistory ah
    JOIN Application_View_Staff avs ON ah.AcknowledgementId = avs.AcknowledgementId
    WHERE avs.CountryId IS NOT NULL
      {stage_where_clause}
      AND DATEADD(MINUTE,330,ah.ChangedOn) BETWEEN '{s}' AND '{e}'
    GROUP BY avs.CountryId, avs.TagGroupID, {intake}, {inyear}
  )
SELECT
  '{report_month}' AS ReportMonth,
  COALESCE(r.CountryId, s.CountryId)   AS CountryId,
  COALESCE(r.TagGroupID, s.TagGroupID) AS TagGroupID,
  COALESCE(r.Intake, s.Intake)         AS Intake,
  COALESCE(r.InYear, s.InYear)         AS InYear,
  ISNULL(r.ReceivedApps,0)  AS ReceivedApps,
  ISNULL(s.SubmittedApps,0) AS SubmittedApps
FROM Received r
FULL OUTER JOIN Submitted s
  ON r.CountryId=s.CountryId AND r.TagGroupID=s.TagGroupID AND r.Intake=s.Intake AND r.InYear=s.InYear
ORDER BY CountryId, TagGroupID, Intake, InYear;
"""


def _stage_groups_active(groups=None) -> dict:
    groups = STAGE_GROUPS if groups is None else groups
    return {g: {k: v for k, v in m.items() if v} for g, m in groups.items() if any(m.values())}


def _stage_case(groups) -> str:
    """CASE expression mapping (CountryId, ApplicationStageId) -> StageGroup; country keys before '*'."""
    whens = []
    for g, m in groups.items():
        for cid, ids in m.items():
            if cid == "*":
                continue
            whens.append(f"WHEN avs.CountryId = {int(cid)} AND ah.ApplicationStageId IN ({','.join(str(int(i)) for i in ids)}) "
                         f"THEN '{g}'")
        if m.get("*"):
            excl = [int(c) for c in m if c != "*"]
            cond = f"avs.CountryId NOT IN ({','.join(map(str, excl))}) AND " if excl else ""
            whens.append(f"WHEN {cond}ah.ApplicationStageId IN ({','.join(str(int(i)) for i in m['*'])}) THEN '{g}'")
    return "CASE " + " ".join(whens) + " END"


def build_stage_sql(start_dt, end_dt, submitted_stages_dict, default_stages, report_month, groups=None):
    """Applications reaching each stage group for the FIRST time inside the 26->25 window (an application
    that re-enters a stage later is not counted twice)."""
    s = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    e = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    grp = _stage_case(_stage_groups_active(groups))
    return f"""
WITH
  StageEvents AS (
    SELECT avs.CountryId, avs.TagGroupID, ah.AcknowledgementId, {grp} AS StageGroup,
           MIN(DATEADD(MINUTE,330,ah.ChangedOn)) AS FirstReached
    FROM ApplicationHistory ah
    JOIN Application_View_Staff avs ON ah.AcknowledgementId = avs.AcknowledgementId
    WHERE avs.CountryId IS NOT NULL AND {grp} IS NOT NULL
      AND DATEADD(MINUTE,330,ah.ChangedOn) <= '{e}'   -- first reach up to the cycle end
    GROUP BY avs.CountryId, avs.TagGroupID, ah.AcknowledgementId, {grp}
  )
SELECT '{report_month}' AS ReportMonth, CountryId, TagGroupID, StageGroup,
       COUNT(DISTINCT AcknowledgementId) AS Apps
FROM StageEvents
WHERE FirstReached BETWEEN '{s}' AND '{e}'
GROUP BY CountryId, TagGroupID, StageGroup
ORDER BY CountryId, TagGroupID, StageGroup;
"""


STAGE_DISCOVERY_SQL = """
-- read-only: which stage IDs are in use (last 12 months), with names if a stage master table exists
SELECT ah.ApplicationStageId, COUNT(DISTINCT ah.AcknowledgementId) AS Apps_Last_12m,
       MIN(DATEADD(MINUTE,330,ah.ChangedOn)) AS First_Seen, MAX(DATEADD(MINUTE,330,ah.ChangedOn)) AS Last_Seen
FROM ApplicationHistory ah
WHERE ah.ChangedOn >= DATEADD(MONTH,-12,GETDATE())
GROUP BY ah.ApplicationStageId
ORDER BY Apps_Last_12m DESC;
"""

STAGE_TABLES_SQL = """
SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME LIKE '%Stage%' ORDER BY TABLE_NAME, ORDINAL_POSITION;
"""


def discover_stages(engine, out_path: Path = DATA_ROOT / "stage_discovery.xlsx") -> Path:
    """Stage IDs in use + candidate stage master tables, to fill STAGE_GROUPS."""
    used = execute_query(STAGE_DISCOVERY_SQL, engine)
    tables = execute_query(STAGE_TABLES_SQL, engine)
    names = pd.DataFrame()
    for (schema, table), cols in tables.groupby(["TABLE_SCHEMA", "TABLE_NAME"]):
        c = set(cols["COLUMN_NAME"])
        idc = next((x for x in ("ApplicationStageId", "StageId", "Id") if x in c), None)
        namec = next((x for x in ("StageName", "Name", "Description", "Title") if x in c), None)
        if idc and namec:
            try:
                names = execute_query(f"SELECT {idc} AS ApplicationStageId, {namec} AS StageName "
                                      f"FROM [{schema}].[{table}]", engine).assign(Source=f"{schema}.{table}")
                break
            except Exception as exc:                        # noqa: BLE001 - try the next candidate table
                print(f"  (could not read {schema}.{table}: {exc})")
    if len(names):
        used = used.merge(names, on="ApplicationStageId", how="left")
    with pd.ExcelWriter(out_path) as w:
        used.to_excel(w, sheet_name="Stages_In_Use", index=False)
        tables.to_excel(w, sheet_name="Stage_Tables", index=False)
    print(f"✅ Stage discovery written to {out_path}: fill STAGE_GROUPS in fetch_data.py from it.")
    return out_path


def execute_query(sql, engine):
    t0 = datetime.now()
    df = pd.read_sql(sql, engine)
    print(f"  → fetched {len(df)} rows in {(datetime.now()-t0).total_seconds():.1f}s")
    return df


def _load_env_file(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines); existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_engine():
    from sqlalchemy import create_engine
    _load_env_file(Path(__file__).resolve().parent / ".env")
    missing = [k for k in ("DB_SERVER", "DB_NAME", "DB_USERNAME", "DB_PASSWORD") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Missing database settings {missing}: set them as environment variables or in a "
                         f"git-ignored .env file in the repo root (see SETUP_GUIDE.md).")
    url = (f"mssql+pyodbc://{os.environ['DB_USERNAME']}:{quote_plus(os.environ['DB_PASSWORD'])}"
           f"@{os.environ['DB_SERVER']}/{os.environ['DB_NAME']}"
           "?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes")
    return create_engine(url)


def fetch_missing(engine, sql_builder, output_path: Path, key_cols, start_date: date, end_date: date, label: str):
    """Query only fully closed months not yet present in `output_path`, append and save."""
    if output_path.exists():
        existing = pd.read_excel(output_path, dtype=str)
        if not {"ReportMonth", *key_cols}.issubset(existing.columns):
            print(f"⚠️  {output_path.name} is missing key columns; starting fresh.")
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()
    existing_months = set(existing["ReportMonth"].astype(str)) if len(existing) else set()
    if existing_months:
        y, m = map(int, max(existing_months).split("-"))
        start_date = max(start_date, date(y, m, 26))
    missing = [c for c in month_ranges(start_date, end_date) if c[2] not in existing_months]
    if not missing:
        print(f"ℹ️  [{label}] no missing closed months — nothing to fetch.")
        return
    new = []
    for ms, me, mstr in missing:
        print(f"\n-- [{label}] Month {mstr} ({ms.date()}→{me.date()}) --")
        df = execute_query(sql_builder(ms, me, COUNTRY_SUBMITTED_STAGES, DEFAULT_SUBMITTED_STAGES, mstr), engine)
        if len(df):
            new.append(df.astype(str))
    if not new:
        print(f"ℹ️  [{label}] no new rows.")
        return
    combined = pd.concat([existing, *new], ignore_index=True) if len(existing) else pd.concat(new, ignore_index=True)
    combined = combined.drop_duplicates(subset=["ReportMonth", *key_cols], keep="last")
    combined.to_excel(output_path, index=False)
    print(f"\n✅ [{label}] saved {sum(len(n) for n in new)} new rows to {output_path}")


# -----------------------------------------------------------
# 3) Main
# -----------------------------------------------------------
def main():
    start_date = datetime.strptime(START_DATE_STR, "%Y-%m-%d").date()
    ist = pytz.timezone("Asia/Kolkata")
    today_ist = datetime.now(ist).date()
    hard_cutoff = last_closed_cycle_end(today_ist)
    requested_end = (datetime.strptime(END_DATE_STR, "%Y-%m-%d").date() if END_DATE_STR.strip()
                     else (datetime.now(ist) - timedelta(days=1)).date())
    end_date = min(requested_end, hard_cutoff)       # never beyond the last fully closed 26->25 cycle
    print(f"ℹ️  Requested end date: {requested_end} | Effective end date: {end_date} | Hard cutoff: {hard_cutoff}")
    engine = make_engine()
    fetch_missing(engine, build_monthly_sql, OUTPUT_PATH, ["CountryId", "TagGroupID"], start_date, end_date, "monthly")
    if FETCH_INTAKE:
        fetch_missing(engine, build_intake_sql, INTAKE_OUTPUT_PATH, ["CountryId", "TagGroupID", "Intake", "InYear"],
                      start_date, end_date, "intake")
    if _stage_groups_active():
        fetch_missing(engine, build_stage_sql, STAGE_OUTPUT_PATH, ["CountryId", "TagGroupID", "StageGroup"],
                      start_date, end_date, "stages")
    else:
        print("ℹ️  [stages] STAGE_GROUPS is empty: stage extract skipped (run --discover-stages first).")


if __name__ == "__main__":
    import sys
    if "--discover-stages" in sys.argv:
        discover_stages(make_engine())
    else:
        main()
