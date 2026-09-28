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

## 5. Replacing the hard "same month last year" fallback (adopted)

**Scope.** 11 series (4 Received, 7 Submitted; e.g. Europe, UK Franchise) where the model average lost
to seasonal naive on monthly error in the walk-forward test. Until now they switched entirely to last
year's numbers.

**Candidates.** Median absolute % error, walk-forward:

| Window | model average | last year only (old) | 50/50 | **horizon hedge** (last-year weight 0.5 → 1 over 12 months) |
|---|---|---|---|---|
| Next month | 12.8 | 17.3 | 13.0 | 13.0 |
| Next quarter | 15.0 | 14.3 | 11.2 | **11.6** |
| Next 6 months | 19.0 | 14.4 | 11.3 | **10.9** |
| 12 months | 21.2 | **12.1** | 15.1 | 14.1 |

A per-series choice made nested (only on earlier results) was worse than a fixed rule: 14.5 at the
quarter for Europe. With about 20 realised quarters per series, the choice is noise.

**Decision:** horizon hedge. It gains 2.7–3.5 points where decisions are made (quarter, half-year,
fiscal-year remainder) and gives up 2 points at 12 months.

## 6. Proof page: the last completed quarter (pack page 2)

Each month the pack shows what the current procedure forecast 3 months earlier for the months that
have now happened, next to the actuals.

For Jul–Sep 2026, forecast at end of June:

| Series | Miss |
|---|---|
| Organisation, received | −9% |
| Organisation, submitted | −5% |
| UK received | −17% |
| NZ received | −19% |
| Canada received | −26% |
| USA received | +2% |

The surge accelerated (September +34% over its forecast month) and the forecast lagged, as diagnosed in §1.

These months are **not** tuned to look right. Fitting them would make this page look perfect and the
next quarter fail. The page is regenerated unchanged every month, so the live record accumulates.

## 7. Direction signal from the latest vintages (adopted; the user's design)

**Idea.** Before issuing the forecast at month-end o, check how the most recent forecasts did on the
months already known:
- the vintages of o-5, o-4 and o-3 (three full quarters);
- o-2 (2 months known) and o-1 (1 month known), for current momentum.

The signal is the recency-weighted log(actual / forecast), where a month counts half as much for every 2
months it is older. It is read at the **country total**, where noise cancels, and applied at half strength
to the country's regions and to the country: forecast × exp(0.5 × signal).

**Why the earlier overlay failed where this works.** The earlier overlay corrected each series from its
own raw-model errors. Empirical-Bayes shrinkage then cancelled most of the correction, and it was judged
on monthly error. This design:
- reads the published vintages;
- pools at country level;
- is judged on the next-quarter total.

**Walk-forward result** (next-quarter median abs % error, received, 17 origins, signal from earlier
vintages only):

| Level | none | country signal × 0.5 | own signal × 0.5 | organisation signal × 0.5 |
|---|---|---|---|---|
| Country | 18.4 | **14.2** | 14.2 | 18.5 |
| Region | 22.3 | **19.2** | 20.9 | 23.5 |
| Organisation (bottom-up of corrected regions) | 9.2 | **9.3** | 9.7 | 9.5 |

- **Submitted:** correcting it directly made the organisation worse (11.7 → 13.5). Submissions therefore
  inherit the correction through the funnel instead.
- **Selection note:** six variants were compared on the same origins. Country × 0.5 was chosen as the
  theory-preferred option (pooling plus partial adjustment), not only the best number. It is re-tested
  every run (`vintage.decide`, ≥ 3% level-averaged gain) and switches off if it stops winning.

**Report.** Pack page 3 ("Direction check") shows the miss of each of the last 5 vintages per
destination, coloured blue (actual higher) or orange (lower), with the newest on the right.

## 8. One decision rule for every component (consistency fix)

**Problem found.** With the same data and identical backtests, the organisation's next-quarter forecast
moved from 86.2k (run17) to 96.5k (run19). The recent-bias overlay switched itself on because its gate
was a 1% gain in monthly WAPE (25.6 → 25.2), and a small change elsewhere (the hedge) tipped it over.
A number that moves 12% because a threshold flickered cannot be trusted.

**Rule now used by every component** (overlay, point estimate, funnel, direction signal, hedge,
reconciliation): judged on next-quarter totals per level, averaged over levels, and adopted only when at
least 3% better.

**Reconciliation on quarter totals** (median abs % error, 24 origins):

| Metric | Level | bottom_up | wls_var | mint_shrink | none |
|---|---|---|---|---|---|
| Received | Country | **18.8** | 19.5 | 19.9 | 20.4 |
| Received | Organisation | 9.9 | **9.8** | 10.2 | 11.8 |
| Submitted | Country | **18.8** | 20.0 | 20.4 | 20.9 |
| Submitted | Organisation | **9.1** | 9.7 | 9.8 | 13.0 |
| All levels | Average | **16.9** | 17.4 | 17.7 | 18.5 |

`wls_var` had won only on monthly WAPE. The method is now chosen with `reconcile.choose_method_quarter`.

**Overlay on quarter totals** (bottom-up):

| Metric | Level | none | level | decay |
|---|---|---|---|---|
| Received | Country | 18.8 | 14.6 | 14.6 |
| Received | Organisation | 9.9 | 8.1 | 8.7 |
| Received | Region | 22.3 | 20.9 | 20.4 |

It clears the 3% bar decisively, so the correction is real. Only its gate was unstable.

**Direction signal (§7), moved.** It now corrects the published (reconciled) forecasts, with regions
corrected and totals re-summed. On run19's published forecasts it did not help (country 13.1 → 13.9),
because the hedge and funnel had already removed most of the lag. It stays gated and re-tested monthly.
