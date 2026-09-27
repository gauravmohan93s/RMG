# Experiment: how to form a forecast from the candidate models (2026-09-27)

**Question.** Should each series use its best model, the best three, or an average of all?
v1 picked a single winner per series; the first v2 draft averaged the top 3 that beat the benchmark.

**Design.** Walk-forward replay on the production extract (data through Sep 2026):
94 mature series, 7 candidate models, 24 rolling origins × 12-month horizon. For each of the
last 12 origins the policy was re-decided using only errors known at that time, and its
forecasts scored against actuals. Metric: WAPE by level × horizon bucket, and per-series
horizon-balanced MAE relative to seasonal naive.

**Results (WAPE %, lower is better).**

| Level | Months ahead | Best single | Top-3 | Mean of all | Trimmed mean | Seasonal naive |
|---|---|---|---|---|---|---|
| Organisation | 1-3 | 13.2 | 13.7 | 13.4 | 13.5 | 15.7 |
| Organisation | 4-6 | 16.2 | 17.5 | 18.1 | 18.3 | 16.9 |
| Organisation | 7-12 | 24.6 | 29.3 | 25.7 | 26.8 | 23.8 |
| Country | 1-3 | 17.0 | 18.9 | 18.2 | 17.5 | 37.7 |
| Country | 4-6 | 23.3 | 22.8 | 22.0 | 22.2 | 31.9 |
| Country | 7-12 | 30.8 | 26.2 | 27.2 | 27.7 | 31.8 |
| Region | 1-3 | 22.8 | 21.3 | 20.6 | 20.0 | 38.9 |
| Region | 4-6 | 28.7 | 25.9 | 24.6 | 25.1 | 34.3 |
| Region | 7-12 | 34.2 | 31.3 | 28.3 | 28.9 | 33.8 |

| Per-series summary | Best single | Top-3 | Mean of all | Seasonal naive |
|---|---|---|---|---|
| Median error vs seasonal naive | 0.83 | 0.79 | **0.73** | 1.00 |
| Series beating seasonal naive | 68% | 69% | **84%** | — |

Also tested: filtered means (only models with relative error ≤ 1.25 or ≤ 1.0) — no better than the plain mean.

**Decision.** Default policy `mean_all` (equal-weight average of all eligible models). It is the
simplest, has no selection step to overfit, and wins or ties on most cells — consistent with the
"forecast combination puzzle" (Claeskens et al., 2016; M4 competition findings).

**Caveats.** (1) At organisation level 7–12 months ahead, seasonal naive was competitive: the
evaluation window contains the 2025 dip → 2026 surge turning point, which trend-following methods
extrapolated wrongly. (2) All policies were biased low (actuals 10–18% above forecasts) for the
same reason. (3) The comparison is recomputed every run (Policy Comparison sheet); if another
policy starts winning consistently, change `selection.policy` and record it in the CHANGELOG.
