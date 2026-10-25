# Experiment: narrower 12-month ranges for steady series such as Europe (2026-10-25)

**Problem** (open item from 2026-09-27_long_horizon_and_ranges.md §4). For long windows without enough
independent own history, the range is "own quarter volatility × pooled shape of 12-month errors". Some
series have noisy quarters but steady 12-month totals, so their ranges were overstated:
- Europe's 12-month 80% range was −70% / +236% in run16.
- After the v2.2 hedge it was −49% / +96%.
- Its real 12-month misses: typically 13%, worst −18% / +77%.

**Change tested.** Blend the series' own 12-month spread with the location-scale spread on the log scale
(the natural way to shrink a scale estimate), weighted by independent evidence:

  w = n_eff / (n_eff + 2),  n_eff = own 12-month windows / 12

**Data.** September 2026 run, 36 origins, sequential replay (only errors known at each origin).

| Group | Window | Interval score before | after | Out-of-sample 80% coverage before | after |
|---|---|---|---|---|---|
| Region (incl. single-region destinations: Europe, Asia, CEE, SE & C Asia, East Asia) | 12 months | 5.274 | **5.071** | – | – |
| Region | 6 months | 3.264 | **2.791** | 79.3% | 78.7% |
| Countries + organisation | 6 months | **1.388** | 1.502 | country 65.7% | 57.1% |
| Countries + organisation | 12 months | **2.091** | 2.116 | – | – |

**Decision.** Apply the blend to the **Region group only** (`ranges.LS_BLEND_GROUPS`). There it is better
on the proper score with unchanged coverage; for countries and the organisation it is worse, so they keep
the current method.

Result for Europe's 12-month 80% range: −49% / +96% → **−42% / +73%**. The UK, Canada and the
organisation are unchanged by this decision.
