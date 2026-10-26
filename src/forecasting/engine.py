# -*- coding: utf-8 -*-
"""
End-to-end forecasting run:

  extract -> panel (status, hierarchy) -> rolling-origin backtest of all models
  -> scores -> selection (+ honest walk-forward check, benchmark fallback)
  -> final base forecasts (+ guardrails, emerging-series method)
  -> empirical calibrated intervals -> reconciliation -> governed overlay
  -> totals / fiscal periods -> invariants -> history, manifest, artefacts.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import __version__
from ..config import CODE_ROOT, INTAKE_FILE, RAW_FILE, SETTINGS_FILE, STAGE_FILE, STAGE_INTAKE_FILE, Settings
from ..data.prepare import METRICS, ORG, TOTAL, Panel, build_panel, load_raw, node_id
from ..governance import manifest as mf
from ..governance import tracking
from ..governance.decisions import DecisionLock
from ..governance import snapshots as snp
from ..analysis import seasonality as seas
from ..analysis import signals as sig
from ..analysis import intake as itk
from ..analysis import intake_pace as ip
from . import bias_correction as bc
from . import calendar_fx as cx
from . import funnel as fn
from . import stages as sg
from . import vintage as vt
from . import guardrails as gr
from . import intervals as iv
from . import overlay as ov
from . import reconcile as rc
from . import ranges as rg
from . import reliability as rl
from . import selection as sel
from .backtest import run_backtest
from .models import model_catalogue, pooled_seasonal_forecast, seasonal_index

LOG = logging.getLogger("rmg.engine")
DECISIONS_FILE = "method_decisions.json"
HISTORY_FILE = "prediction_history_v2.parquet"
EMERGING_METHOD = "PooledSeasonal"


@dataclass
class RunResult:
    out_dir: Path
    run_id: str
    run_date: str
    last_actual: pd.Timestamp
    forecast: pd.DataFrame
    totals: pd.DataFrame
    periods: pd.DataFrame
    history: pd.DataFrame
    nodes: pd.DataFrame
    series_status: pd.DataFrame
    dq: pd.DataFrame
    scores: pd.DataFrame
    selection: pd.DataFrame
    accuracy: pd.DataFrame
    policy_comparison: pd.DataFrame
    calibration: pd.DataFrame
    block_calibration: pd.DataFrame
    range_test: pd.DataFrame
    range_tested: pd.DataFrame
    guardrails: pd.DataFrame
    overlay_rules: pd.DataFrame
    catalogue: pd.DataFrame
    failures: pd.DataFrame
    track_scorecard: pd.DataFrame
    bias_test: pd.DataFrame
    bias_estimates: pd.DataFrame
    reconciliation_test: pd.DataFrame
    point_estimate_test: pd.DataFrame
    funnel_test: pd.DataFrame
    vintage_test: pd.DataFrame
    decisions: pd.DataFrame
    calendar_test: pd.DataFrame
    stage_forecast: pd.DataFrame
    stage_totals: pd.DataFrame
    stage_test: pd.DataFrame
    stage_conversion: pd.DataFrame
    data_revisions: pd.DataFrame
    seasonal_index: pd.DataFrame
    seasonal_by_year: pd.DataFrame
    staffing: pd.DataFrame
    forecast_vs_actual: pd.DataFrame
    fva_summary: pd.DataFrame
    human_scores: pd.DataFrame
    events: pd.DataFrame
    intake_arrivals: pd.DataFrame
    intake_lead_profile: pd.DataFrame
    intake_month_mix: pd.DataFrame
    manifest: dict
    intake_pace: pd.DataFrame = None          # intake running indicator (v2.5); empty for older runs
    intake_curves: pd.DataFrame = None
    intake_pace_test: pd.DataFrame = None


# where each table lives inside the run folder
ARTEFACTS = {
    "forecast": "Final_Forecast/final_forecast.csv",
    "totals": "Final_Forecast/totals_summary.csv",
    "periods": "Final_Forecast/period_summary.csv",
    "overlay_rules": "Final_Forecast/overlay_rules.csv",
    "history": "Data/history.csv",
    "nodes": "Data/nodes.csv",
    "series_status": "Data/series_status.csv",
    "dq": "Data/data_quality.csv",
    "scores": "Evaluation/model_scores.csv",
    "selection": "Evaluation/selection.csv",
    "accuracy": "Evaluation/accuracy_by_horizon.csv",
    "policy_comparison": "Evaluation/policy_comparison.csv",
    "calibration": "Evaluation/interval_calibration.csv",
    "block_calibration": "Evaluation/total_interval_calibration.csv",
    "range_test": "Evaluation/range_method_test.csv",
    "range_tested": "Evaluation/range_tested_coverage.csv",
    "guardrails": "Evaluation/guardrails.csv",
    "catalogue": "Evaluation/model_catalogue.csv",
    "failures": "Evaluation/model_failures.csv",
    "track_scorecard": "Evaluation/track_record_scorecard.csv",
    "bias_test": "Evaluation/statistical_overlay_test.csv",
    "bias_estimates": "Evaluation/statistical_overlay_estimates.csv",
    "reconciliation_test": "Evaluation/reconciliation_test.csv",
    "point_estimate_test": "Evaluation/point_estimate_test.csv",
    "funnel_test": "Evaluation/submitted_funnel_test.csv",
    "vintage_test": "Evaluation/vintage_signal_test.csv",
    "decisions": "Evaluation/method_decisions.csv",
    "calendar_test": "Evaluation/calendar_effects_test.csv",
    "stage_forecast": "Final_Forecast/stage_forecast.csv",
    "stage_totals": "Final_Forecast/stage_totals.csv",
    "stage_test": "Evaluation/stage_test.csv",
    "stage_conversion": "Evaluation/stage_conversion.csv",
    "data_revisions": "Data/data_revisions.csv",
    "seasonal_index": "Analysis/seasonal_index.csv",
    "seasonal_by_year": "Analysis/seasonal_by_year.csv",
    "staffing": "Analysis/staffing_calendar.csv",
    "forecast_vs_actual": "Evaluation/forecast_vs_actual.csv",
    "fva_summary": "Evaluation/forecast_vs_actual_summary.csv",
    "human_scores": "Analysis/human_input_scores.csv",
    "events": "Analysis/policy_event_impacts.csv",
    "intake_arrivals": "Analysis/intake_arrivals.csv",
    "intake_lead_profile": "Analysis/intake_lead_profile.csv",
    "intake_month_mix": "Analysis/intake_month_mix.csv",
    "intake_pace": "Analysis/intake_pace.csv",
    "intake_curves": "Analysis/intake_curves.csv",
    "intake_pace_test": "Evaluation/intake_pace_test.csv",
}
DATE_COLS = ("Date", "Start", "First_Month", "Last_Active_Month", "Period_Start", "From", "To",
             "Target_Month", "Made_At", "Start_Month", "Intake_Start")


def label(level: str, country: str, region: str) -> str:
    if level == "Organisation":
        return "Organisation"
    if level == "Country":
        return country
    return f"{country} – {region}"


# ─────────────────────────────────────────────────────────────────────────────
def run_forecast(settings: Settings, out_dir: Path, run_date: str, raw_file: Path = RAW_FILE,
                 settings_file: Path = SETTINGS_FILE, history_path: Optional[Path] = None,
                 raw: Optional[pd.DataFrame] = None, intake_file: Optional[Path] = INTAKE_FILE,
                 intake_raw: Optional[pd.DataFrame] = None, stage_file: Optional[Path] = STAGE_FILE,
                 stage_intake_file: Optional[Path] = STAGE_INTAKE_FILE) -> RunResult:
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    # method decisions are locked per quarter (see governance/decisions.py)
    lock = DecisionLock((history_path.parent if history_path else out_dir.parent) / DECISIONS_FILE, run_date)

    def _avg(tbl, key):
        return tbl.groupby(key)["Quarter_Error_Pct"].mean().to_dict() if tbl is not None and len(tbl) else {}

    def lap(name):
        nonlocal t0
        timings[name] = round(time.perf_counter() - t0, 1)
        LOG.info("stage %-14s %6.1fs", name, timings[name])
        t0 = time.perf_counter()

    H, B = settings.backtest.horizon, settings.selection.benchmark
    levels = settings.intervals.levels

    # 1) data ───────────────────────────────────────────────────────────────
    if raw is None:
        raw = load_raw(raw_file, settings.data.drop_regions)
    panel = build_panel(raw, settings.data)
    nodes = panel.nodes
    T = len(panel.months)
    months_ext = pd.date_range(panel.months[0], periods=T + H, freq="MS")
    last_actual = panel.end
    # extract snapshot + late-entry revisions of earlier months (measured from the second month on)
    snap_dir = (history_path.parent if history_path else out_dir.parent) / "Snapshots"
    data_rev = snp.summary(snp.revisions(raw, snap_dir, last_actual))
    try:
        snp.save(raw, snap_dir, last_actual)
    except OSError as exc:
        LOG.warning("could not save extract snapshot: %s", exc)
    for r in data_rev[data_rev["Material"]].itertuples():
        panel.dq = pd.concat([panel.dq, pd.DataFrame([{
            "Issue": "Late entries: recent months get revised", "Country": r.Country, "Region": TOTAL,
            "Detail": f"{r.Metric}: a month's count rose by {r.Typical_Revision_Pct:+.1f}% (median of "
                      f"{r.Months_Evidence} months) after it was first reported; the latest month is likely "
                      "under-counted by about as much."}])], ignore_index=True)
    tour = nodes[nodes["Status"] == "Mature"]
    emer = nodes[nodes["Status"] == "Emerging"]
    lap("data")

    # 2) backtest ───────────────────────────────────────────────────────────
    bt = run_backtest(panel, tour["node_id"].tolist(), settings.models.enabled, H,
                      settings.backtest.n_origins, settings.models.lgbm, settings.random_seed,
                      settings.n_jobs)
    lap("backtest")

    # 3) selection + honest evaluation ──────────────────────────────────────
    sc = settings.selection
    scores = sel.score_models(bt.preds, B)
    selection = sel.select_models(scores, B, sc.top_k, sc.must_beat_benchmark, sc.policy)
    honest = sel.honest_evaluation(bt.preds, B, sc.top_k, sc.must_beat_benchmark, sc.honest_eval_origins, sc.policy)
    policy_cmp = sel.policy_comparison(honest, nodes)
    selection = selection.merge(sel.honest_summary(honest), on="node_id", how="left")
    default_method = {"mean_all": "Average of all models",
                      "mean_screened": "Average of models beating the benchmark"}.get(sc.policy, "Top-k combination")
    selection["Method"] = np.where(selection["Selected_Models"] == B, "Benchmark (no model beat it)", default_method)
    fb = ((selection["Honest_N"] >= sel.MIN_KNOWN_PAIRS)
          & (selection["Honest_Skill_vs_Benchmark"] < settings.selection.fallback_if_skill_below)
          & (selection["Selected_Models"] != B))
    # v2.2: a hedge instead of a hard switch. Where the model average lost to 'same month last year' in
    # the walk-forward test, publish w(h) x last year + (1-w(h)) x model average, with w rising from 0.5
    # (next month) to 1 (12 months ahead). Tested on these series: next-quarter miss 14.3% -> 11.6%,
    # 6 months 14.4% -> 10.9%, 12 months 12.1% -> 14.1% (docs/experiments/2026-09-28_bias_and_funnel.md)
    hedge_ids = set(selection.loc[fb, "node_id"])
    selection.loc[fb, "Selection_Reason"] += (" Walk-forward test: the combination lost to seasonal naive -> "
                                               "published as a hedge: last year's level weighted 50% next month "
                                               "rising to 100% at 12 months, the model average for the rest.")
    selection.loc[fb, "Method"] = "Hedge: model average + same month last year (walk-forward)"
    selection = selection.merge(bt.plan, on="node_id", how="left")
    lap("selection")

    # 4) base forecasts + guardrails ────────────────────────────────────────
    selected = selection.set_index("node_id")["Selected_Models"]
    base = sel.combine(bt.final, selected)
    comb_bt = sel.combine(bt.preds, selected)
    if hedge_ids:
        base = _hedge(base, bt.final, hedge_ids, B)
        comb_bt = _hedge(comb_bt, bt.preds, hedge_ids, B)
        hm = honest["node_id"].isin(hedge_ids)
        w = hedge_weight(honest.loc[hm, "h"].to_numpy())
        honest.loc[hm, "Policy"] = w * honest.loc[hm, "Benchmark"] + (1 - w) * honest.loc[hm, "Mean_All"]

    # statistical overlay: recent-bias correction, adopted only if it wins walk-forward
    bcfg = settings.bias_correction
    bias_test, variants = bc.walk_forward_test(comb_bt, honest, bcfg.window, bcfg.max_h)
    # decided on next-quarter totals with the common >= 3% rule (a 1% monthly-WAPE gate flipped between
    # runs and moved the organisation forecast by 12%); the monthly table is kept as evidence
    bias_qtest = bc.overlay_quarter_test(honest, variants, tour)
    bias_mode = (lock.decide("statistical_overlay", bc.decide_quarter(bias_qtest), _avg(bias_qtest, "Mode"))
                 if bcfg.enabled else "none")
    if len(bias_test) and len(bias_qtest):
        bias_test = bias_test.merge(bias_qtest.groupby("Mode")["Quarter_Error_Pct"].mean().rename(
            "Quarter_Error_Pct_Level_Avg").reset_index(), on="Mode", how="left")
    bias_est = bc.estimate(comb_bt, T - 1, bcfg.window, bcfg.max_h)
    bias_est["Mode_Adopted"] = bias_mode
    if bias_mode != "none":
        corr = base["node_id"].map(bias_est.set_index("node_id")["Correction_Log"]).fillna(0.0).to_numpy()
        base["Pred"] = base["Pred"].to_numpy() * bc.factor(corr, base["h"].to_numpy(), bias_mode)
        honest = honest.assign(Policy=variants[bias_mode])
        mult = bias_est.set_index("node_id")["Correction_Log"].map(np.exp)
        for nid, mlt in mult.items():
            if abs(mlt - 1) >= 0.005:
                m = selection["node_id"] == nid
                selection.loc[m, "Selection_Reason"] += (
                    f" Statistical overlay ({bias_mode}): recent actuals vs forecasts imply x{mlt:.3f} "
                    f"(shrunk estimate from the last {bcfg.window} realised months).")
    selection["Statistical_Overlay"] = selection["node_id"].map(
        bias_est.set_index("node_id")["Correction_Log"].map(np.exp) if bias_mode != "none" else {}).fillna(1.0)

    # point estimate: typical month (median) vs expected value (smearing-corrected mean) — the one
    # that gives more accurate quarter totals across levels in the walk-forward test is published
    smear_test, smear_honest = bc.smearing_test(comb_bt, honest, tour)
    pe = settings.point_estimate
    if pe == "auto":
        pe_scores = {("mean" if "mean" in k else "median"): v for k, v in _avg(smear_test, "Mode").items()}
        use_mean = lock.decide("point_estimate", "mean" if bc.decide_smearing(smear_test) else "median",
                               pe_scores) == "mean"
    else:
        use_mean = pe == "mean"
    if use_mean and smear_honest is not None:
        f_now = bc.smearing_factors(comb_bt, T - 1)
        base["Pred"] = base["Pred"].to_numpy() * base["node_id"].map(f_now).fillna(1.0).to_numpy()
        comb_bt = comb_bt.assign(Pred=comb_bt["Pred"].to_numpy() * comb_bt["node_id"].map(f_now).fillna(1.0).to_numpy())
        honest = honest.assign(Policy=smear_honest)
    point_estimate_used = "mean (smearing-corrected)" if use_mean else "median (typical month)"

    # funnel: submissions follow receipts (a leading indicator); adopted only if it wins walk-forward
    Y_full = fn.full_series(panel)
    funnel_test, funnel_var = fn.walk_forward_test(honest, tour, Y_full)
    funnel_used = lock.decide("submitted_funnel", fn.decide(funnel_test), _avg(funnel_test, "Variant"))
    if funnel_used != "model":
        honest = honest.assign(Policy=funnel_var[funnel_used])
        f_bt = fn.funnel_forecast(comb_bt, "Pred", Y_full)
        comb_bt = comb_bt.assign(Pred=fn.variant(comb_bt, "Pred", f_bt, funnel_used))
        b_o = base.assign(o=T - 1)
        base["Pred"] = fn.variant(b_o, "Pred", fn.funnel_forecast(b_o, "Pred", Y_full), funnel_used).to_numpy()
        m = selection["node_id"].str.startswith("Submitted|")
        selection.loc[m, "Selection_Reason"] += (f" Submissions: '{funnel_used}' (recent conversion from receipts) "
                                                 "replaced the model forecast after winning the walk-forward test.")
    refs = pd.DataFrame({nid: gr.references(panel.y[nid]) for nid in tour["node_id"]}).T
    guards = gr.plausibility(base, refs, settings.guardrails.min_ratio, settings.guardrails.max_ratio)
    for _, g in guards[guards["Guardrail_Triggered"]].iterrows():
        nid = g["node_id"]
        base.loc[base["node_id"] == nid, "Pred"] *= g["Scale_Applied"]
        m = selection["node_id"] == nid
        selection.loc[m, "Selection_Reason"] += (
            f" Guardrail: 12-month total ({g['Forecast_12m']:,.0f}) was outside the plausible range "
            f"{g['Lower_Bound']:,.0f}-{g['Upper_Bound']:,.0f} implied by last year and recent momentum "
            f"-> scaled to the bound; review this series.")
        selection.loc[m, "Method"] = "Guardrail-adjusted combination"

    # emerging series: recent level x parent's seasonal index
    tgt_months = months_ext[T:T + H].month.to_numpy()
    em_rows, em_sel = [], []
    for _, n in emer.iterrows():
        parent = node_id(n["Metric"], n["Country"], TOTAL)
        if parent not in panel.y or parent == n["node_id"] or \
                nodes.set_index("node_id").loc[parent, "Status"] != "Mature":
            parent = node_id(n["Metric"], ORG, TOTAL)
        idx = seasonal_index(panel.y[parent], panel.months[panel.start[parent]:].month.to_numpy())
        y = panel.y[n["node_id"]]
        f = pooled_seasonal_forecast(y, panel.months[panel.start[n["node_id"]]:].month.to_numpy(), idx, tgt_months)
        em_rows.append(pd.DataFrame({"node_id": n["node_id"], "h": np.arange(1, H + 1), "t": np.arange(T, T + H),
                                     "Target": months_ext[T:T + H], "Pred": f}))
        em_sel.append({"node_id": n["node_id"], "Selected_Models": EMERGING_METHOD, "N_Selected": 1,
                       "Method": "Emerging series (pooled seasonality)",
                       "Selection_Reason": (f"Only {n['Months_Of_History']} months of history: recent 6-month "
                                            f"level x seasonal pattern of {parent.split('|')[1].replace(ORG, 'the organisation')}. "
                                            "Not enough data to test models; low confidence.")})
    if em_rows:
        base = pd.concat([base, *em_rows], ignore_index=True)
        selection = pd.concat([selection, pd.DataFrame(em_sel)], ignore_index=True)
    lap("forecast")

    # reconciliation method: replayed walk-forward, chosen on evidence ("auto") — and from here on the
    # walk-forward forecasts are the reconciled ones, i.e. exactly the procedure that is published
    recon_test, recon_detail = rc.walk_forward_test(honest, comb_bt, nodes, return_detail=True)
    # chosen on next-quarter totals per level (how the numbers are used), not monthly error
    rq = rc.quarter_errors(recon_detail)
    recon_method = (lock.decide("reconciliation", rc.choose_method_quarter(recon_detail), _avg(rq, "Method"))
                    if settings.reconciliation_method == "auto" else settings.reconciliation_method)
    if len(recon_test) and len(rq):
        recon_test = recon_test.merge(rq, on=["Metric", "Level", "Method"], how="left")
    if len(recon_detail) and recon_method in set(recon_detail["Method"]):
        rd = recon_detail[recon_detail["Method"] == recon_method][["node_id", "o", "h", "Pred"]]
        honest = honest.merge(rd, on=["node_id", "o", "h"], how="left")
        honest["Policy"] = honest["Pred"].fillna(honest["Policy"])
        honest = honest.drop(columns="Pred")

    # direction signal from the latest PUBLISHED vintages (receipts): the last three full quarters plus
    # the latest two partial ones, read at country level; regions corrected, totals re-summed; adopted
    # only if it wins the next-quarter test
    vintage_test, vintage_corr = vt.walk_forward_test(honest, tour)
    v_scores = {("on" if k == "vintage" else "off"): v for k, v in _avg(vintage_test, "Variant").items()}
    vintage_used = lock.decide("direction_signal", "on" if vt.decide(vintage_test) else "off", v_scores) == "on"
    vintage_signal: Dict[str, float] = {}
    v_sig: dict = {}
    if vintage_used:
        v_sig = vt.signals(honest, sorted(set(honest["o"]) | {T - 1}))
        honest = honest.assign(Policy=vintage_corr)
        vintage_signal = {k[0]: round(100 * float(np.expm1(v)), 1) for k, v in v_sig.items()
                          if k[1] == T - 1 and np.isfinite(v) and "_TOTAL" in k[0]}

    # calendar & festival effects per 26th-25th cycle (receipts): same factor for every series in a month,
    # so totals still add up; adopted on month-level accuracy (months 1-3), never at the cost of quarters
    cal_D = cx.deltas(months_ext)
    calendar_test, cal_corr, cal_coefs = cx.walk_forward_test(honest, tour, cal_D)
    cal_scores = ({("on" if v == "calendar" else "off"): e for v, e in
                   calendar_test[calendar_test["Window"] == "Month"].groupby("Variant")["Error_Pct"].mean().items()}
                  if len(calendar_test) else {})
    calendar_used = lock.decide("calendar_effects", "on" if cx.decide(calendar_test) else "off", cal_scores) == "on"
    cal_final: dict = {}
    if calendar_used:
        cal_final = cx.coefficients(honest, tour, cal_D, [T - 1])
        honest = honest.assign(Policy=cal_corr)

    # 5) intervals ──────────────────────────────────────────────────────────
    # ranges are built from the errors of the forecasts as PUBLISHED (e.g. regions summed bottom-up)
    comb_pub = rl.published_backtest(comb_bt, tour, recon_method)
    if vintage_used:
        comb_pub = comb_pub.assign(Pred=vt.apply_published(comb_pub, "Pred", tour, v_sig).to_numpy())
    if calendar_used:
        comb_pub = comb_pub.assign(Pred=comb_pub["Pred"].to_numpy() * cx.factors(comb_pub, cal_D, cal_coefs))
    # the way errors become a range (empirical quantiles vs centred robust spread) is chosen per level
    # group and window by a proper scoring rule, replayed with only the errors realised at each origin
    q, bq, range_test, calib, block_calib, range_tested = rg.build(comb_pub, nodes, levels, settings.intervals.min_obs, H,
                                                     settings.selection.honest_eval_origins)
    lap("intervals")

    # 6) reconciliation ─────────────────────────────────────────────────────
    piv = base.pivot_table(index="node_id", columns="h", values="Pred")
    resid_h1 = comb_bt[comb_bt["h"] == 1].assign(e=lambda d: d["Actual"] - d["Pred"])
    level_hint = base.groupby("node_id")["Pred"].mean()
    rec_parts, recon_used = [], {}
    for metric in METRICS:
        mn = nodes[nodes["Metric"] == metric]
        S, all_ids, b_ids = rc.summing_matrix(mn)
        Y = piv.loc[all_ids].to_numpy()
        var = rc.node_variances(all_ids, resid_h1, level_hint)
        E = resid_h1.pivot_table(index="o", columns="node_id", values="e").reindex(columns=all_ids).to_numpy()
        last12 = {b: float(panel.y[b][-12:].sum()) if b in panel.y else 0.0 for b in b_ids}
        Yt, used = rc.reconcile(Y, S, recon_method, var=var, E=E, shares=rc.region_shares(S, b_ids, last12))
        recon_used[metric] = used
        for i, nid in enumerate(all_ids):
            rec_parts.append(pd.DataFrame({"node_id": nid, "h": np.arange(1, H + 1),
                                           "Forecast": Yt[i], "Base_Forecast": Y[i]}))
    rec = pd.concat(rec_parts, ignore_index=True)
    if vintage_used:
        rec["Forecast"] = vt.apply_published(rec, "Forecast", nodes, v_sig, fixed_origin=T - 1).to_numpy()
    if calendar_used and cal_final:
        rec["Forecast"] = rec["Forecast"].to_numpy() * cx.factors(rec, cal_D, cal_final, fixed_origin=T - 1)
    lap("reconcile")

    # 7) assemble ───────────────────────────────────────────────────────────
    fc = rec.merge(nodes[["node_id", "Level", "Country", "Region", "Metric", "Status"]], on="node_id")
    fc["Date"] = months_ext[T - 1 + fc["h"].to_numpy()]
    fc["Horizon"] = fc["h"]
    fc = fc.merge(q[["node_id", "h"] + iv._qcols(levels)], on=["node_id", "h"], how="left")
    for L in levels:
        fc[f"Lo{L}"], fc[f"Hi{L}"] = iv.apply_interval(fc["Forecast"].to_numpy(), fc[f"q_lo{L}"].to_numpy(),
                                                       fc[f"q_hi{L}"].to_numpy())
    fc = fc.drop(columns=iv._qcols(levels))
    selc = selection.set_index("node_id")
    fc["Method"] = fc["node_id"].map(selc["Method"])
    fc["Models"] = fc["node_id"].map(selc["Selected_Models"])
    relt = rl.reliability(honest, nodes).set_index("node_id")
    fc["Reliability"] = fc["node_id"].map(relt["Reliability"]).fillna("New")
    fc["Typical_Quarter_Error_Pct"] = fc["node_id"].map(relt["Typical_Quarter_Error_Pct"])
    guard_ids = set(guards.loc[guards["Guardrail_Triggered"], "node_id"]) if len(guards) else set()
    fc.loc[fc["node_id"].isin(guard_ids), "Reliability"] = "Review"
    # must be set BEFORE the single-region-country copies below: copied Country rows keep their region's
    # node_id here, which is how _totals finds their ranges
    fc["Source_Node"] = fc["node_id"]

    # single-region countries: country total = its region
    reg_counts = nodes[nodes["Level"] == "Region"].groupby(["Metric", "Country"]).size()
    single = [c for (m, c), k in reg_counts.items() if k == 1 and m == METRICS[0]]
    copies = fc[(fc["Level"] == "Region") & fc["Country"].isin(single)].copy()
    copies["Level"], copies["Region"] = "Country", TOTAL
    copies["node_id"] = [node_id(m, c, TOTAL) for m, c in zip(copies["Metric"], copies["Country"])]
    fc = pd.concat([fc, copies], ignore_index=True)

    # 8) governed overlay ───────────────────────────────────────────────────
    rules_raw, overlay_source = ov.load_overlay(settings_file, settings.overlay_sheet, settings.overlay_template)
    rules = ov.validate(rules_raw)
    fc = _apply_overlay(fc, rules, levels)
    fc["Label"] = [label(l, c, r) for l, c, r in zip(fc["Level"], fc["Country"], fc["Region"])]

    # human inputs: shown next to the model, blended only for proven forecasters
    hist_dir = history_path.parent if history_path else out_dir.parent
    hi_raw, human_source = sig.load_sheet(settings_file, "Human_Inputs", CODE_ROOT / "config" / "human_inputs_template.csv",
                                          sig.HUMAN_COLS)
    hi = sig.clean_human(hi_raw, today=pd.Timestamp(run_date))
    human_hist = sig.record_human(hi, run_date, hist_dir / "human_inputs_history.parquet")
    model_hist_path = history_path or (out_dir.parent / HISTORY_FILE)
    model_hist = pd.read_parquet(model_hist_path) if model_hist_path.exists() else pd.DataFrame()
    human_scores = sig.score_humans(human_hist, panel.history, model_hist)
    fc = sig.attach_human(fc, hi, human_scores)
    fc = fc.sort_values(["Metric", "Level", "Country", "Region", "Date"], key=_level_sort).reset_index(drop=True)

    invariants = gr.check_invariants(fc, last_actual, levels)

    # pipeline stages after submission (offer, deposit, visa...): supplementary, never blocks the forecast
    stage_fc = stage_tot = stage_test = stage_conv = pd.DataFrame()
    stage_status = "no stage extract"
    if stage_file is not None and Path(stage_file).exists():
        try:
            sa = sg.load_stage_actuals(Path(stage_file), panel.months)
            stage_fc, stage_test, stage_conv = sg.run(sa, Y_full, nodes, honest, fc, months_ext, T, H)
            stage_tot = sg.stage_totals(stage_fc)
            stage_status = (f"forecast: {', '.join(stage_fc['StageGroup'].unique())}" if len(stage_fc)
                            else "extract present, too little history")
        except Exception as exc:                                  # noqa: BLE001 - log and continue
            LOG.exception("stage forecast failed")
            stage_status = f"failed: {type(exc).__name__}: {exc}"
    lap("stages")
    coherence = gr.coherence_gap(fc)
    lap("assemble")

    # 9) summaries ──────────────────────────────────────────────────────────
    # revisions: last month's forecast for the same target months, shown next to this month's
    prev = tracking.previous_forecast(history_path or (out_dir.parent / HISTORY_FILE), last_actual)
    fc = fc.merge(prev[tracking.KEYS + ["Previous_Forecast"]], on=tracking.KEYS, how="left") if len(prev) else         fc.assign(Previous_Forecast=np.nan)
    totals = _totals(fc, bq, panel.history, levels)
    periods = _periods(fc, panel.history, settings.data.fin_year_start_month, bq, levels)
    accuracy = sel.accuracy_by_horizon(honest, bt.preds, nodes)
    scfg = settings.seasonality
    sidx, syear = seas.seasonal_indices(panel.history, scfg.high_threshold, scfg.low_threshold, scfg.years)
    prod, prod_source = sig.load_sheet(settings_file, "Staffing_Productivity",
                                       CODE_ROOT / "config" / "staffing_productivity_template.csv",
                                       seas.PRODUCTIVITY_COLS)
    staffing = seas.staffing_calendar(fc, sidx, scfg.high_threshold, scfg.low_threshold, scfg.apps_per_fte_month,
                                      productivity=prod)
    st_stage = seas.stage_staffing(stage_fc, prod, scfg.high_threshold, scfg.low_threshold)   # offers, deposits, visa
    if len(st_stage):
        staffing = pd.concat([staffing, st_stage], ignore_index=True)
    ev_raw, events_source = sig.load_sheet(settings_file, "Policy_Events",
                                           CODE_ROOT / "config" / "policy_events_template.csv", sig.EVENT_COLS)
    events = sig.measure_events(ev_raw, panel.history)
    intake_df = (itk.clean_intake(intake_raw, settings.data.drop_regions) if intake_raw is not None
                 else itk.load_intake(intake_file, settings.data.drop_regions) if intake_file else pd.DataFrame())
    i_arr, i_lead, i_mix, i_known = itk.intake_views(intake_df, last_actual)
    try:                                           # intake running indicator: supplementary, never blocks
        stage_intake = ip.load_stage_intake(stage_intake_file, settings.data.drop_regions)
        i_pace, i_curves, i_test = ip.intake_pace(ip.intake_long(intake_df, stage_intake), last_actual)
    except Exception:                              # noqa: BLE001 - log and continue
        LOG.exception("intake running indicator failed")
        i_pace = i_curves = i_test = pd.DataFrame()
    lap("summaries")

    # 10) governance ────────────────────────────────────────────────────────
    git = mf.git_info(CODE_ROOT)
    run_id = f"{run_date}_{git.get('commit', 'nogit')[:7]}"
    history_path = history_path or (out_dir.parent / HISTORY_FILE)
    out_cols = ["Level", "Country", "Region", "Metric", "Date", "Horizon", "Forecast",
                "Lo80", "Hi80", "Lo95", "Hi95", "Method"]
    tracking.log_run(fc[out_cols], run_id, run_date, __version__, last_actual, history_path)
    live_joined, track_sc = tracking.track_record(panel.history, history_path)
    if live_joined is not None and len(live_joined):
        live_joined = live_joined[pd.to_datetime(live_joined["Run_Date"]) != pd.Timestamp(run_date)]
    fva = sig.forecast_vs_actual(honest, nodes, months_ext, live_joined)
    fva_sum = sig.fva_summary(fva)
    catalogue = pd.DataFrame(model_catalogue(settings.models.enabled))
    catalogue = catalogue.merge(_model_usage(selection, scores), on="Model", how="left")

    decisions = lock.save()
    manifest = mf.write_manifest(out_dir / "run_manifest.json", {
        "pipeline_version": __version__, "run_id": run_id, "run_date": run_date,
        "git": git, "packages": mf.package_versions(),
        "data": {**(mf.file_fingerprint(raw_file) if raw_file and Path(raw_file).exists() else {}),
                 "rows": int(len(raw)), "first_month": str(panel.months[0].date()),
                 "last_actual_month": str(last_actual.date()),
                 "series_by_status": panel.series_status["Status"].value_counts().to_dict()},
        "config": settings.as_dict(),
        "reconciliation_method_used": recon_used,
        "reconciliation_method_chosen": recon_method,
        "point_estimate_used": point_estimate_used,
        "submitted_method": funnel_used,
        "vintage_signal_used": vintage_used,
        "calendar_effects_used": calendar_used,
        "calendar_effects_log_coefficients": ({k: round(float(v), 4) for k, v in zip(cx.FEATURES, cal_final[T - 1])}
                                              if cal_final.get(T - 1) is not None else {}),
        "method_decisions": decisions.to_dict("records") if len(decisions) else [],
        "vintage_signal_pct_latest": vintage_signal,
        "range_methods_chosen": ({f"{g} | {w}": m for g, w, m in range_test.loc[range_test["Chosen"], ["Group", "Window", "Method"]]
                                  .drop_duplicates(["Group", "Window"]).itertuples(index=False)} if len(range_test) else {}),
        "overlay_source": overlay_source,
        "overlay_rules_applied": int((rules["Applied_Status"] == "Applied").sum()),
        "guardrails_triggered": int(guards["Guardrail_Triggered"].sum()),
        "model_failures": int(len(bt.failures)),
        "invariants": invariants, "coherence_max_gap": coherence,
        "timings_seconds": timings,
        "prediction_history": str(history_path),
        "statistical_overlay_mode": bias_mode,
        "human_inputs_source": human_source, "human_inputs_rows": int(len(hi)),
        "policy_events_source": events_source,
        "staffing_productivity_source": prod_source,
        "stages": stage_status,
        "intake_extract": (str(intake_file) if intake_df is not None and len(intake_df) else "not available"),
        "intake_known_share": i_known,
    })

    res = RunResult(out_dir=out_dir, run_id=run_id, run_date=run_date, last_actual=last_actual,
                    forecast=fc, totals=totals, periods=periods, history=panel.history, nodes=nodes,
                    series_status=panel.series_status, dq=panel.dq, scores=scores, selection=selection,
                    accuracy=accuracy, policy_comparison=policy_cmp, calibration=calib, block_calibration=block_calib, range_test=range_test, range_tested=range_tested, guardrails=guards, overlay_rules=rules,
                    catalogue=catalogue, failures=bt.failures, track_scorecard=track_sc,
                    bias_test=bias_test, bias_estimates=bias_est, reconciliation_test=recon_test, point_estimate_test=smear_test, funnel_test=funnel_test, vintage_test=vintage_test, decisions=decisions, calendar_test=calendar_test, stage_forecast=stage_fc, stage_totals=stage_tot, stage_test=stage_test, stage_conversion=stage_conv, data_revisions=data_rev, seasonal_index=sidx, seasonal_by_year=syear,
                    staffing=staffing, forecast_vs_actual=fva, fva_summary=fva_sum, human_scores=human_scores,
                    events=events, intake_arrivals=i_arr, intake_lead_profile=i_lead, intake_month_mix=i_mix,
                    manifest=manifest, intake_pace=i_pace, intake_curves=i_curves, intake_pace_test=i_test)
    save_result(res, bt.preds, honest)
    lap("save")
    return res


# ─────────────────────────────── helpers ────────────────────────────────────
_LEVEL_ORDER = {"Organisation": 0, "Country": 1, "Region": 2}


def _level_sort(s: pd.Series) -> pd.Series:
    return s.map(_LEVEL_ORDER) if s.name == "Level" else s


def _apply_overlay(fc: pd.DataFrame, rules: pd.DataFrame, levels) -> pd.DataFrame:
    fc = fc.copy()
    reg = fc["Level"] == "Region"
    fc["Overlay_Multiplier"] = 1.0
    fc.loc[reg, "Overlay_Multiplier"] = ov.multipliers(fc[reg], rules)
    fc["Forecast_Adjusted"] = fc["Forecast"] * fc["Overlay_Multiplier"]
    regs = fc[reg]
    by_c = regs.groupby(["Metric", "Country", "Date"])["Forecast_Adjusted"].sum()
    by_o = regs.groupby(["Metric", "Date"])["Forecast_Adjusted"].sum()
    cm = fc["Level"] == "Country"
    fc.loc[cm, "Forecast_Adjusted"] = [by_c.get((m, c, d), f) for m, c, d, f in
                                       zip(fc.loc[cm, "Metric"], fc.loc[cm, "Country"], fc.loc[cm, "Date"],
                                           fc.loc[cm, "Forecast"])]
    om = fc["Level"] == "Organisation"
    fc.loc[om, "Forecast_Adjusted"] = [by_o.get((m, d), f) for m, d, f in
                                       zip(fc.loc[om, "Metric"], fc.loc[om, "Date"], fc.loc[om, "Forecast"])]
    ratio = (fc["Forecast_Adjusted"] / fc["Forecast"].replace(0, np.nan)).fillna(1.0)
    fc.loc[~reg, "Overlay_Multiplier"] = ratio[~reg].round(4)
    for L in levels:
        fc[f"Adjusted_Lo{L}"] = fc[f"Lo{L}"] * ratio
        fc[f"Adjusted_Hi{L}"] = fc[f"Hi{L}"] * ratio
    return fc


def _totals(fc: pd.DataFrame, bq: pd.DataFrame, history: pd.DataFrame, levels) -> pd.DataFrame:
    """Next 3/6/12-month totals with block intervals and like-for-like YoY."""
    hist = history.set_index(["Level", "Country", "Region", "Metric", "Date"])["Actual"]
    bqi = bq.set_index(["node_id", "k"])
    rows = []
    for key, g in fc.groupby(["Level", "Country", "Region", "Metric"], sort=False):
        g = g.sort_values("Date")
        src = g["Source_Node"].iloc[0]
        for k in iv.BLOCKS:
            gk = g.head(k)
            point, adj = float(gk["Forecast"].sum()), float(gk["Forecast_Adjusted"].sum())
            pv = gk["Previous_Forecast"] if "Previous_Forecast" in gk else pd.Series(dtype=float)
            prev_total = float(pv.sum()) if len(pv) and pv.notna().all() else np.nan
            ly_dates = gk["Date"] - pd.DateOffset(years=1)
            ly = float(sum(hist.get((*key, d), np.nan) for d in ly_dates))
            row = {"Level": key[0], "Country": key[1], "Region": key[2], "Metric": key[3],
                   "Label": g["Label"].iloc[0], "Window": f"Next {k} months",
                   "From": gk["Date"].min(), "To": gk["Date"].max(),
                   "Forecast": point, "Forecast_Adjusted": adj, "Same_Months_Last_Year": ly,
                   "Previous_Forecast": prev_total,
                   "Change_vs_Previous_Pct": 100 * (point / prev_total - 1) if prev_total and prev_total > 0 else np.nan,
                   "YoY_Pct": 100 * (point / ly - 1) if ly > 0 else np.nan,
                   "Reliability": g["Reliability"].iloc[0],
                   "Typical_Quarter_Error_Pct": g["Typical_Quarter_Error_Pct"].iloc[0]}
            if (src, k) in bqi.index:
                qq = bqi.loc[(src, k)]
                for L in levels:
                    lo, hi = iv.apply_interval(np.array([point]), np.array([qq[f"q_lo{L}"]]),
                                               np.array([qq[f"q_hi{L}"]]))
                    row[f"Lo{L}"], row[f"Hi{L}"] = float(lo[0]), float(hi[0])
            rows.append(row)
    return pd.DataFrame(rows)


def _fiscal(d: pd.Timestamp, start: int):
    fy = d.year + (1 if d.month >= start else 0)
    q = (d.month - start) % 12 // 3 + 1
    return fy, q


def hedge_weight(h: np.ndarray, horizon: int = 12) -> np.ndarray:
    """Weight on 'same month last year': 0.5 next month, rising linearly to 1 at `horizon` months."""
    return 0.5 + 0.5 * (np.asarray(h, dtype=float) - 1) / max(horizon - 1, 1)


def _hedge(frame: pd.DataFrame, preds: pd.DataFrame, ids: set, benchmark: str) -> pd.DataFrame:
    keys = [c for c in ("node_id", "o", "h") if c in frame.columns and c in preds.columns]
    sn = preds[(preds["Model"] == benchmark) & preds["node_id"].isin(ids)][keys + ["Pred"]].rename(columns={"Pred": "_SN"})
    out = frame.merge(sn, on=keys, how="left")
    m = out["node_id"].isin(ids) & out["_SN"].notna()
    w = hedge_weight(out.loc[m, "h"].to_numpy())
    out.loc[m, "Pred"] = w * out.loc[m, "_SN"] + (1 - w) * out.loc[m, "Pred"]
    return out.drop(columns="_SN")[frame.columns]


def _periods(fc: pd.DataFrame, history: pd.DataFrame, fy_start: int, bq: Optional[pd.DataFrame] = None,
             levels=(80,)) -> pd.DataFrame:
    """Fiscal quarter / year outlook = actuals to date + forecast remainder, vs same period last year.
    Range: only the forecast remainder is uncertain. It uses the total-range of the smallest tested window
    (3/6/12 months from now) that covers the remainder; months already actual add no uncertainty, so the
    year-end range narrows every month as actuals arrive."""
    hist = history.set_index(["Level", "Country", "Region", "Metric", "Date"])["Actual"]
    bqi = bq.set_index(["node_id", "k"]) if bq is not None and len(bq) else None
    first_fc, last_fc = fc["Date"].min(), fc["Date"].max()
    rows = []
    for key, g in fc.groupby(["Level", "Country", "Region", "Metric"], sort=False):
        f = g.set_index("Date")
        periods: Dict[tuple, List[pd.Timestamp]] = {}
        d = pd.Timestamp(first_fc)
        # start from the beginning of the fiscal year containing the first forecast month
        fy0, _ = _fiscal(d, fy_start)
        start = pd.Timestamp(year=fy0 - 1, month=fy_start, day=1)
        for dd in pd.date_range(start, last_fc, freq="MS"):
            fy, q = _fiscal(dd, fy_start)
            periods.setdefault((f"FY{fy % 100:02d} Q{q}", "Quarter"), []).append(dd)
            periods.setdefault((f"FY{fy % 100:02d}", "Fiscal year"), []).append(dd)
        for (name, kind), ds in periods.items():
            need = 3 if kind == "Quarter" else 12
            if len(ds) < need or max(ds) < first_fc:
                continue            # incomplete period or entirely in the past
            act = [d for d in ds if d < first_fc]
            fcs = [d for d in ds if d >= first_fc]
            a = float(sum(hist.get((*key, d), np.nan) for d in act))   # missing -> unknown, not 0
            fsum = float(f.loc[fcs, "Forecast"].sum())
            fadj = float(f.loc[fcs, "Forecast_Adjusted"].sum())
            ly = float(sum(hist.get((*key, d - pd.DateOffset(years=1)), np.nan) for d in ds))
            rng = {}
            if bqi is not None and fcs:
                n_end = (max(fcs).year - first_fc.year) * 12 + max(fcs).month - first_fc.month + 1
                kk = next((k for k in iv.BLOCKS if k >= n_end), None)
                src = g["Source_Node"].iloc[0] if "Source_Node" in g else None
                if kk is not None and (src, kk) in bqi.index:
                    qq = bqi.loc[(src, kk)]
                    for L in levels:
                        lo, hi = iv.apply_interval(np.array([fsum]), np.array([qq[f"q_lo{L}"]]), np.array([qq[f"q_hi{L}"]]))
                        rng[f"Outlook_Lo{L}"], rng[f"Outlook_Hi{L}"] = a + float(lo[0]), a + float(hi[0])
            rows.append({**rng, "Level": key[0], "Country": key[1], "Region": key[2], "Metric": key[3],
                         "Label": g["Label"].iloc[0], "Period": name, "Period_Type": kind,
                         "Period_Start": min(ds), "Months_Actual": len(act), "Months_Forecast": len(fcs),
                         "Actual_To_Date": a, "Forecast_Remaining": fsum, "Outlook": a + fsum,
                         "Outlook_Adjusted": a + fadj, "Same_Period_Last_Year": ly,
                         "YoY_Pct": 100 * ((a + fsum) / ly - 1) if ly > 0 else np.nan})
    return pd.DataFrame(rows)


def _model_usage(selection: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    used = selection["Selected_Models"].dropna().str.split(r" \+ ").explode().value_counts()
    rel = scores.groupby("Model")["Rel_to_Benchmark"].median()
    beat = scores.assign(b=scores["Rel_to_Benchmark"] < 1).groupby("Model")["b"].mean() * 100
    return pd.DataFrame({"Series_Using_Model": used, "Median_Error_vs_Benchmark": rel,
                         "Pct_Series_Beating_Benchmark": beat}).rename_axis("Model").reset_index()


# ─────────────────────────────── persistence ─────────────────────────────────
def save_result(res: RunResult, preds: pd.DataFrame, honest: pd.DataFrame) -> None:
    for attr, rel in ARTEFACTS.items():
        p = res.out_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        df = getattr(res, attr)
        (df if df is not None else pd.DataFrame()).to_csv(p, index=False)
    preds.to_parquet(res.out_dir / "Evaluation" / "backtest_predictions.parquet", index=False)
    if not honest.empty:
        honest.to_parquet(res.out_dir / "Evaluation" / "walk_forward_evaluation.parquet", index=False)


def load_result(out_dir: Path) -> RunResult:
    import json
    kw = {}
    for attr, rel in ARTEFACTS.items():
        p = out_dir / rel
        df = pd.read_csv(p) if p.exists() and p.stat().st_size > 2 else pd.DataFrame()
        for c in DATE_COLS:
            if c in df.columns:
                if attr in ("stage_test", "stage_conversion") and c in ("From", "To"):
                    continue
                try:
                    df[c] = pd.to_datetime(df[c])
                except Exception as exc:  # keep the run going, but never silently: a date column left as text sorts / compares wrongly
                    print(f"[load_result] WARNING: {rel} column '{c}' is not a date column ({exc}); left as text")
        kw[attr] = df
    man = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    return RunResult(out_dir=out_dir, run_id=man["run_id"], run_date=man["run_date"],
                     last_actual=pd.Timestamp(man["data"]["last_actual_month"]), manifest=man, **kw)
