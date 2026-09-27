# Experiment: how uncertain are 12-month totals, and can the forecast be made better there? (2026-09-27)

**Question.** With the deeper test (36 backtest origins; the last 24 used for testing, covering the
2024–25 dip and the 2025–26 surge), country 12-month ranges came out very wide (Australia up to +160%).
Is that a flaw in how the ranges are built, or is it what really happened? If it is real, can the
point forecast be improved at long horizons?

## 1. What actually happened: typical miss of past 12-month forecasts (received)

These are published (reconciled) forecasts, made as if at the time.

| Series | Quarter (typical / worst) | 12 months (typical / worst) |
|---|---|---|
| Organisation | 9% / −21%, +17% | 15% / −24%, +15% |
| United Kingdom | 14% / −28%, +35% | 14% / −15%, +29% |
| Canada | 20% | 34% / −47%, +44% |
| New Zealand | 23% | 38% / −47%, +102% |
| Australia | 26% | 39% / −56%, +76% |
| USA | 25% | 42% / −64%, −11% |

The large 12-month misses are real, and they run in both directions. Australia is the clearest case:
- Before the 2024 visa caps the models extrapolated growth: 68k forecast vs 30k actual.
- After the collapse they extrapolated the fall: 14k forecast vs 25k actual.

The ranges must reflect this, not hide it.

## 2. Can the long-horizon forecast be improved by leaning on last year's level?

**Design.** The weight on "same month last year" grows with the horizon, w(h) = w_max·(h−1)/11, and
the result is reconciled bottom-up. Scored as the median absolute % error of quarter and 12-month totals
over the last 24 origins.

| Window | Level | w_max 0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|
| Quarter | Country | **18.8** | 19.2 | 19.4 | 20.1 | 21.4 |
| Quarter | Organisation | 9.9 | **9.8** | 9.9 | 10.1 | 10.0 |
| Quarter | Region | **22.3** | 22.6 | 22.8 | 23.2 | 23.5 |
| 12 months | Country | 27.3 | 27.1 | 27.3 | 26.7 | **25.6** |
| 12 months | Organisation | 10.3 | 11.0 | **9.6** | 9.8 | 10.1 |
| 12 months | Region | **35.3** | 35.7 | 36.5 | 38.1 | 40.9 |

(Received shown; Submitted gives the same picture.)

**Decision: rejected.** It makes quarters and regions worse and gives no consistent gain at 12 months.
The current combination is as good as these simpler alternatives. The remaining 12-month uncertainty
comes from policy shocks that history alone cannot anticipate. Leading indicators (leads, enquiries)
or measured events and approved market input are the only levers.

## 3. Range fixes adopted (v2.2)

1. **Ranges come from the forecasts actually published.** For reconciliation methods other than
   bottom-up, country and organisation ranges had been built from their unreconciled errors. As a
   result, the organisation's 6-month range held only 39% of the time. `reconcile_backtest` now replays
   the chosen method at every origin, using weights from errors known by then.
2. **Effective sample size.** Consecutive k-month windows share k−1 months, so n windows carry about
   n/k independent observations.
   - A series uses its own spread only with at least 4 independent windows' worth.
   - The symmetric width factor needs at least 30 independent test results.
   - 24 overlapping 12-month windows are about 2 episodes, too few for a country-specific 12-month
     spread.
3. **Location-scale pooling for long windows.** A series' own quarter volatility (well estimated) is
   multiplied by the pooled shape of 12-month-vs-quarter errors across comparable series. Calm markets
   keep narrow ranges and shock-hit markets keep wide ones.
4. **Monotone in the window.** A 12-month range is never narrower, in %, than the 6-month range.

**Result:** final numbers are in the run16 manifest and the Range Method Test sheet. For use:
- plan countries on next-quarter numbers and roll them monthly;
- treat country 12-month totals as direction with a wide range, and report them as such;
- the UK and the organisation are the exceptions, and their 12-month numbers are plannable.

## 4. Known limitation (open)

For series that publish the "same month last year" fallback (4 series; e.g. Europe), quarter errors
are noisy but 12-month totals are steady: Europe's 12-month total typically missed by 8%. The
location-scale step assumes 12-month error grows with quarter volatility, so it OVERSTATES these
ranges: Europe's 12-month 80% range is −70% / +236% against realised misses of −33% / +56%. The ranges
are too wide, never too narrow. The pack labels them "direction only".

Next step: blend the series' own 12-month spread (weighted by its effective sample) with the pooled
estimate, and test that in the sequential replay before adopting it.
