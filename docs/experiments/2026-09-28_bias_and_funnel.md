# Experiment: why do forecasts under-predict, and what fixes it? (2026-09-28)

## 1. Diagnosis: the bias flips with the market direction

Bias of the published forecast, (forecast − actual) / actual, walk-forward (made as if at the time):

| Received | next 1–3 months | 4–6 | 7–12 |
|---|---|---|---|
| Organisation 2024–25 (falling market) | **+8.7%** | +11.2% | +7.5% |
| Organisation 2025–26 (surge) | **−6.3%** | −9.5% | −15.9% |
| Countries 2024–25 | +8.7% | +12.4% | +10.4% |
| Countries 2025–26 | −6.0% | −9.7% | −17.0% |

Share of organisation quarters that were under-forecast: 33% in 2024–25, 79% in 2025–26.

- **Conclusion.** The models do not "always under-predict". They lag turning points: they keep
  extrapolating the previous direction and catch up over several months, and the lag grows with the
  horizon. The last year was a surge, so the lag showed up as under-prediction. Over the full two-year
  test the bias averages about −2%.
- **Why history alone cannot fix it.** At a turning point the data cannot yet say whether the change
  will last. Only information that arrives before the volume moves can.

## 2. Tested: momentum-gap correction (rejected)

**Idea.** Correct each forecast by β × (recent actual YoY growth − growth implied by the forecast), with
β fitted walk-forward.

- **Signal.** The correlation of the miss with the gap is weak (0.1–0.4).
- **Median quarter error:**

  | Level | before | after |
  |---|---|---|
  | Organisation | 10.3% | 10.4–10.8% |
  | Countries | 18.7% | 18.0–18.2% |
  | Regions | 22.6% | 21.6–21.8% |

  There is no gain at 6 and 12 months.
- **Decision:** rejected.

## 3. Tested: funnel for submissions (adopted)

**Idea.** Receipts lead submissions.
- 12-month log changes of organisation submissions vs receipts correlate 0.97 at lag 0, 0.90 at lag 1
  and 0.82 at lag 2.
- The funnel forecast is Submitted[t] = Σ w_l · Received[t−l] (l = 0, 1, 2).
- The weights are non-negative least squares on the last 12 realised months, re-fitted at each origin.
- Receipts are actuals up to the origin and the published Received forecast after it.

**Walk-forward result** (median abs % error, next-quarter submissions, 24 origins):

| Level | model | funnel | blend |
|---|---|---|---|
| Organisation | 10.5 | **9.1** | 10.0 |
| Country | 20.6 | **18.9** | 19.3 |
| Region | 23.1 | **20.9** | 22.2 |

On the last 18 origins, countries at 6 months: 23.1 → 15.6.

**Decision:** adopted automatically by the walk-forward rule (`funnel.decide`: best level-average, ≥ 3%
better). It is re-tested every run, so it switches back if it stops winning.

## 4. Practical outcome for decisions

- **Fiscal-year outlook** = actuals to date + forecast of the remaining months. Only the remainder is
  uncertain, so the range narrows every month as actuals arrive. It is shown on page 1 and on each
  country page.
- **Stage data is the next lever for Received and for turning points.** Offers, deposits and the
  country-specific visa stages (CAS UK, Funds Australia, AIP NZ, I-20 USA) each measure how many
  applications cross a later stage. Two models then become possible:
  - a stage-by-stage conversion model, like the funnel above;
  - a "booking curve" by intake: progress this year vs last year at the same point.

  Both see changes before they reach the monthly totals.
