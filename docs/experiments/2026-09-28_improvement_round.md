# Improvement round for the Oct 2026 output (2026-09-28)

Every candidate was tested walk-forward (24 origins, only information available at each origin) on the
decision metric: next-quarter totals per level, or single months where the purpose is monthly workload.

| # | Idea | Result | Decision |
|---|---|---|---|
| 1 | Accuracy-weighted model average (1/error, shrunk 50% to equal) | quarter error 17.16 vs 17.06 (equal weights) | **Rejected**, kept as a candidate in Policy Comparison |
| 2 | Middle-out adding-up (country forecast split to regions by 12-month share) | 17.7 vs 16.9 (bottom-up); regions better (21.3 vs 22.3), countries and organisation worse | **Rejected**, stays a candidate in the quarterly reconciliation choice |
| 3 | Revision damping (blend with last month's forecast for the same month) | countries 13.4 → 14.6 (α = 0.75) | **Rejected**: revisions carry new actuals. Revisions are now **shown** instead (see below) |
| 4 | Calendar and festival effects per 26th–25th cycle | +5.3% per extra working day, Diwali −8.5%, Eid −11.6%, Holi ~0. Received single months: organisation 6.4 → 5.3, countries 13.7 → 13.4; quarters 8.9 → 8.8 | **Adopted** for receipts on month-level accuracy (≥ 3% better, quarters not worse); submissions not adjusted (countries worse) |
| 5 | Quarterly decision lock | all gated choices changed only in Jan/Apr/Jul/Oct, or off-cycle if at least 6% better | **Adopted** (`governance/decisions.py`, Method Decisions sheet) |
| 6 | Extract snapshots and late-entry revisions | each run stores its extract; from the second month the growth of earlier "latest months" is measured per country | **Adopted** (measurement; flagged when ≥ 3 months show ≥ 3% uplift) |
| 7 | Market input form | pre-filled next-quarter form per destination; rows paste into Settings.xlsx → Human_Inputs, then scored and blended once proven | **Adopted** |

Revisions shown in the pack:
- "vs last month's forecast" column on the destinations table;
- "last month's fcst" row in each 12-month strip;
- Previous_Forecast and Change_vs_Previous_Pct in the totals.

Typical month-to-month revision of the same target month on the walk-forward: organisation about 6%,
countries about 10%, regions about 13%.
