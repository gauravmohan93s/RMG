# Intake pace as an input to the next-quarter receipts forecast — rejected

**Date:** 2026-11-22 · **Data:** run 2026-10-26 (actuals to Oct 2026) · **Decision:** not adopted

## Question
The intake running indicator projects a season's final count much better than "same as last year" (destinations,
3 months before the start: 14% vs 53% miss). Does nudging the next-quarter receipts forecast towards the ongoing
season's pace make the forecast more accurate?

## Method (walk-forward, honest)
- At each of the last 22 evaluated month-ends, for the organisation and every destination:
  - **Model forecast:** the next-quarter forecast re-made at that month (`walk_forward_evaluation`).
  - **Pace:** the ongoing season's pace, recomputed from intake data known at that month only.
- **Candidate:** `F × ((1 + pace) / (1 + model growth vs last year)) ^ w`, with w = 0, ¼, ½, ¾, 1 (0 = today).
- **Score:** the common decision rule, the level-averaged median miss on next-quarter totals; adopt only if at
  least 3% better.

## Result (230 tests)

| w | Typical miss (score) | Destinations | Organisation | vs today |
|---|---|---|---|---|
| 0 | 11.9% | 15.4% | 8.4% | — |
| ¼ | 15.1% | 19.4% | 10.8% | 27% worse |
| ½ | 17.9% | 24.0% | 11.7% | 50% worse |
| ¾ | 20.9% | 29.2% | 12.5% | 75% worse |
| 1 | 26.3% | 34.5% | 18.2% | 121% worse |

## Why
A month's receipts are applications for several seasons at once. The ongoing season's pace says how *that* season
is filling, not how many applications will arrive in total next quarter. The model already uses the recent level of
receipts, which carries the same information without the season mix.

## Decision
Not adopted. The pace stays where it is strong: the intake running indicator (pages 4–5, "At a glance").
