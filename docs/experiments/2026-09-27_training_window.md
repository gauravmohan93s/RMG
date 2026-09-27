# Experiment: exclude the COVID-era months from training? (2026-09-27)

**Question.** 2020 to mid-2021 contain lockdowns and visa freezes. Does training only on data from
July 2021 onwards give more accurate forecasts?

**Design.** Full pipeline (17 models, 24 rolling origins × 12 months, walk-forward evaluation of the
published, reconciled forecast) run twice on the same extract: all history (Jan 2020+) vs
`train_start = 2021-07-01`. Same code, same settings otherwise.

**Results** (walk-forward, last 12 origins; lower is better; bias < 0 = forecasts too low)

| Level | Metric | Monthly WAPE all / 2021-07 | Quarter-total median error all / 2021-07 | Bias all / 2021-07 |
|---|---|---|---|---|
| Organisation | Received | **11.4** / 12.1 | **8.6** / 8.9 | **−8.1** / −9.3 |
| Organisation | Submitted | **10.9** / 11.5 | 9.1 / **8.5** | **−3.3** / −4.6 |
| Country | Received | **18.8** / 20.3 | **14.1** / 14.2 | **−8.6** / −9.7 |
| Country | Submitted | **18.7** / 20.0 | **16.6** / 17.1 | **−4.4** / −5.6 |
| Region | Received | **22.6** / 23.9 | 17.6 / 17.6 | **−8.0** / −9.2 |
| Region | Submitted | **23.3** / 24.6 | **19.1** / 19.9 | **−3.2** / −4.6 |

Median series error vs seasonal naive: all history 0.657, from 2021-07 0.682.

**Decision.** Keep all history (`train_start = ""`). Excluding the COVID-era months made monthly error
worse at every level (+0.6 to +1.5 points) and the under-forecast bias larger; the models learn useful
level/recovery dynamics from those months. The option stays in config for future re-tests.
