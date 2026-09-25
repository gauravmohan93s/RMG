import os
from pathlib import Path
import pandas as pd
from datetime import datetime, date, time, timedelta
import pytz
from urllib.parse import quote_plus
from sqlalchemy import create_engine

# Try to load settings from config if available, otherwise fallback
try:
    from src.config import DATA_ROOT
    DEFAULT_OUTPUT_PATH = DATA_ROOT / "monthly_app_summary.xlsx"
except Exception:
    DEFAULT_OUTPUT_PATH = Path(r"C:\Users\gsakhare\OneDrive - KC OVERSEAS EDUCATION PVT LTD\RMG\RMG Calculations\monthly_app_summary.xlsx")

# -----------------------------------------------------------
# 1) Operator inputs — set these if needed
# -----------------------------------------------------------
START_DATE_STR = '2020-01-01'   # YYYY-MM-DD (initial lower bound)
END_DATE_STR   = ''             # '' => up to yesterday (IST)
# OUTPUT_PATH absolute location as requested
OUTPUT_PATH = Path(os.environ.get("RMG_OUTPUT_PATH", str(DEFAULT_OUTPUT_PATH)))


# Per-country “submitted” stage overrides
COUNTRY_SUBMITTED_STAGES = {
    # e.g. 1: [4],
    #       2: [4,81],
}
DEFAULT_SUBMITTED_STAGES = [4, 81]


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
        # The cycle that ENDS on 25th of (year, month) starts on 26th of previous month
        if month == 1:
            prev_year, prev_month = year - 1, 12
        else:
            prev_year, prev_month = year, month - 1

        period_start = date(prev_year, prev_month, 26)
        period_end   = date(year, month, 25)

        # If this cycle ended before our window starts, advance
        if period_end < start_date:
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
            continue

        # If this cycle hasn't fully completed by end_date, STOP (no partial month)
        if period_end > end_date:
            print(
                f"⏭️  Skipping incomplete cycle {year:04d}-{month:02d} "
                f"({period_start}→{period_end}); effective end date is {end_date}"
            )
            break

        # Clamp to overall start (start of cycle may be before our start_date)
        ms = datetime.combine(max(period_start, start_date), time(0, 0, 0))
        me = datetime.combine(period_end, time(23,59,59))  # full completed cycle

        yield ms, me, f"{year}-{month:02d}"

        # Next cycle (next calendar month)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def last_closed_cycle_end(today_ist: date) -> date:
    """
    Return the most recent fully closed 26th→25th cycle end date.
    Examples:
      - 2026-02-26 -> 2026-02-25
      - 2026-02-25 -> 2026-01-25
    """
    if today_ist.day >= 26:
        return date(today_ist.year, today_ist.month, 25)

    if today_ist.month == 1:
        return date(today_ist.year - 1, 12, 25)
    return date(today_ist.year, today_ist.month - 1, 25)


def build_monthly_sql(start_dt, end_dt, submitted_stages_dict, default_stages, report_month):
    """
    Monthly aggregation over your custom 26→25 window; output key is ReportMonth='YYYY-MM'
    (the month of the cycle end day). Columns fully-qualified to avoid ambiguity.
    """
    s = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    e = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Build stage filter dynamically supporting per-country overrides
    conditions = []
    for cid, stages in submitted_stages_dict.items():
        stage_list = ",".join(str(st) for st in stages)
        conditions.append(f"(avs.CountryId = {cid} AND ah.ApplicationStageId IN ({stage_list}))")
    
    default_stage_list = ",".join(str(st) for st in default_stages)
    if submitted_stages_dict:
        overridden_countries = ",".join(str(cid) for cid in submitted_stages_dict.keys())
        conditions.append(f"(avs.CountryId NOT IN ({overridden_countries}) AND ah.ApplicationStageId IN ({default_stage_list}))")
    else:
        conditions.append(f"ah.ApplicationStageId IN ({default_stage_list})")
        
    stage_where_clause = " AND (" + " OR ".join(conditions) + ")"

    return f"""
WITH
  Received AS (
    SELECT
      avs.CountryId,
      avs.TagGroupID,
      COUNT(DISTINCT avs.AcknowledgementNumber)    AS ReceivedApps,
      COUNT(DISTINCT avs.StudentId)                AS ReceivedStudents
    FROM Application_View_Staff avs
    WHERE avs.CountryId IS NOT NULL
      AND DATEADD(MINUTE,330,avs.SubmittedDate) BETWEEN '{s}' AND '{e}'
    GROUP BY
      avs.CountryId, avs.TagGroupID
  ),
  Submitted AS (
    SELECT
      avs.CountryId,
      avs.TagGroupID,
      COUNT(DISTINCT ah.AcknowledgementId)    AS SubmittedApps,
      COUNT(DISTINCT avs.StudentId)           AS SubmittedStudents
    FROM ApplicationHistory ah
    JOIN Application_View_Staff avs
      ON ah.AcknowledgementId = avs.AcknowledgementId
    WHERE avs.CountryId IS NOT NULL
      {stage_where_clause}
      AND DATEADD(MINUTE,330,ah.ChangedOn) BETWEEN '{s}' AND '{e}'
    GROUP BY
      avs.CountryId, avs.TagGroupID
  )
SELECT
  '{report_month}' AS ReportMonth,
  COALESCE(r.CountryId, s.CountryId)       AS CountryId,
  COALESCE(r.TagGroupID, s.TagGroupID)     AS TagGroupID,
  ISNULL(r.ReceivedApps,0)      AS ReceivedApps,
  ISNULL(r.ReceivedStudents,0)  AS ReceivedStudents,
  ISNULL(s.SubmittedApps,0)     AS SubmittedApps,
  ISNULL(s.SubmittedStudents,0) AS SubmittedStudents
FROM (
  SELECT CountryId, TagGroupID FROM Received
  UNION
  SELECT CountryId, TagGroupID FROM Submitted
) AS d
LEFT JOIN Received  r ON d.CountryId=r.CountryId AND d.TagGroupID=r.TagGroupID
LEFT JOIN Submitted s ON d.CountryId=s.CountryId AND d.TagGroupID=s.TagGroupID
ORDER BY CountryId, ReportMonth, TagGroupID;
"""


def execute_query(sql, engine):
    t0 = datetime.now()
    df = pd.read_sql(sql, engine)
    print(f"  → fetched {len(df)} rows in {(datetime.now()-t0).total_seconds():.1f}s")
    return df


# -----------------------------------------------------------
# 3) Main (MONTHLY, query ONLY missing & fully completed months)
# -----------------------------------------------------------
def main():
    # parse dates
    start_date = datetime.strptime(START_DATE_STR, "%Y-%m-%d").date()
    ist = pytz.timezone("Asia/Kolkata")
    today_ist = datetime.now(ist).date()
    hard_cutoff = last_closed_cycle_end(today_ist)

    if END_DATE_STR.strip():
        requested_end_date = datetime.strptime(END_DATE_STR, "%Y-%m-%d").date()
    else:
        # keep legacy behavior as the requested upper bound
        requested_end_date = (datetime.now(ist) - timedelta(days=1)).date()

    # Hard guard: never allow fetching beyond the last fully closed 26->25 cycle.
    end_date = min(requested_end_date, hard_cutoff)
    print(f"ℹ️  Requested end date: {requested_end_date} | Effective end date: {end_date} | Hard cutoff: {hard_cutoff}")

    # build connection
    SERVER   = os.environ.get("DB_SERVER", '***REMOVED***')
    DATABASE = os.environ.get("DB_NAME", '***REMOVED***')
    USERNAME = os.environ.get("DB_USERNAME", '***REMOVED***')
    PASSWORD = os.environ.get("DB_PASSWORD", '***REMOVED***')
    pwd_enc  = quote_plus(PASSWORD)
    CONNECTION_URL = (
        f"mssql+pyodbc://{USERNAME}:{pwd_enc}@{SERVER}/{DATABASE}"
        "?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
    )
    engine = create_engine(CONNECTION_URL)

    # load existing (monthly) and build skip sets
    cols = ["ReportMonth","CountryId","TagGroupID",
            "ReceivedApps","ReceivedStudents","SubmittedApps","SubmittedStudents"]
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_excel(OUTPUT_PATH, dtype=str)
        if not set(["ReportMonth","CountryId","TagGroupID"]).issubset(existing.columns):
            print("⚠️  Existing file missing key columns; starting fresh monthly sheet.")
            existing = pd.DataFrame(columns=cols)
    else:
        existing = pd.DataFrame(columns=cols)

    if len(existing):
        existing["ReportMonth"] = existing["ReportMonth"].astype(str)
        existing["CountryId"]   = existing["CountryId"].astype(int)
        existing["TagGroupID"]  = existing["TagGroupID"].astype(int)
        existing_months = set(existing["ReportMonth"].unique())
        skip_keys = set(zip(existing["ReportMonth"], existing["CountryId"], existing["TagGroupID"]))
    else:
        existing_months = set()
        skip_keys = set()

    # --- OPTIONAL ACCELERATION ---
    # If you already have months, move start_date to the next cycle after the latest ReportMonth (26th of that month)
    if existing_months:
        last_month_str = max(existing_months)  # 'YYYY-MM'
        y, m = map(int, last_month_str.split("-"))
        next_cycle_start = date(y, m, 26)  # day after that cycle's end-of-month window starts
        if next_cycle_start > start_date:
            start_date = next_cycle_start

    # Build only **fully completed** cycles and then filter to **missing** months
    all_cycles = list(month_ranges(start_date, end_date))  # [(ms, me, 'YYYY-MM'), ...] completed only
    missing_cycles = [t for t in all_cycles if t[2] not in existing_months]

    if not missing_cycles:
        print("\nℹ️  No missing full months detected — nothing to fetch.")
        return

    # fetch ONLY missing full months in bulk (all countries in a single query per cycle)
    all_new   = []
    for ms, me, mstr in missing_cycles:
        print(f"\n-- Month {mstr} ({ms.date()}→{me.date()}) --")
        df = execute_query(build_monthly_sql(ms, me, COUNTRY_SUBMITTED_STAGES, DEFAULT_SUBMITTED_STAGES, mstr), engine)

        if df.empty:
            print("   ↳ no rows")
            continue

        # Guard: skip any row that already exists (in case partial prior data)
        if skip_keys:
            df = df.loc[
                [ (r.ReportMonth, r.CountryId, r.TagGroupID) not in skip_keys
                  for r in df.itertuples() ]
            ]
        if df.empty:
            print("   ↳ no new rows")
            continue

        for r in df.itertuples():
            skip_keys.add((r.ReportMonth, r.CountryId, r.TagGroupID))
        all_new.append(df)
        print(f"   ↳ {len(df)} new rows")

    # write back (monthly sheet)
    if all_new:
        new_df   = pd.concat(all_new, ignore_index=True)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(
            subset=["ReportMonth","CountryId","TagGroupID"],
            keep="last", inplace=True
        )
        combined.to_excel(OUTPUT_PATH, index=False)
        print(f"\n✅ Saved {len(new_df)} new rows to {OUTPUT_PATH}")
    else:
        print("\nℹ️  No new data to append.")


if __name__ == "__main__":
    main()
