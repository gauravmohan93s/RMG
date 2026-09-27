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
from ..config import CODE_ROOT, INTAKE_FILE, RAW_FILE, SETTINGS_FILE, Settings
from ..data.prepare import METRICS, ORG, TOTAL, Panel, build_panel, load_raw, node_id
from ..governance import manifest as mf
from ..governance import tracking
from ..analysis import seasonality as seas
from ..analysis import signals as sig
from ..analysis import intake as itk
from . import bias_correction as bc
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
                 intake_raw: Optional[pd.DataFrame] = None) -> RunResult:
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

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
    selection.loc[fb, "Selection_Reason"] += (" Walk-forward test: the combination lost to seasonal naive "
                                               "-> benchmark used instead.")
    selection.loc[fb, "Selected_Models"] = B
    selection.loc[fb, "Method"] = "Benchmark fallback (failed walk-forward test)"
    selection = selection.merge(bt.plan, on="node_id", how="left")
    lap("selection")

    # 4) base forecasts + guardrails ────────────────────────────────────────
    selected = selection.set_index("node_id")["Selected_Models"]
    base = sel.combine(bt.final, selected)
    comb_bt = sel.combine(bt.preds, selected)

    # statistical overlay: recent-bias correction, adopted only if it wins walk-forward
    bcfg = settings.bias_correction
    bias_test, variants = bc.walk_forward_test(comb_bt, honest, bcfg.window, bcfg.max_h)
    bias_mode = bc.decide(bias_test, bcfg.min_gain_pct) if bcfg.enabled else "none"
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
    use_mean = pe == "mean" or (pe == "auto" and bc.decide_smearing(smear_test))
    if use_mean and smear_honest is not None:
        f_now = bc.smearing_factors(comb_bt, T - 1)
        base["Pred"] = base["Pred"].to_numpy() * base["node_id"].map(f_now).fillna(1.0).to_numpy()
        comb_bt = comb_bt.assign(Pred=comb_bt["Pred"].to_numpy() * comb_bt["node_id"].map(f_now).fillna(1.0).to_numpy())
        honest = honest.assign(Policy=smear_honest)
    point_estimate_used = "mean (smearing-corrected)" if use_mean else "median (typical month)"
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
    recon_method = (rc.choose_method(recon_test) if settings.reconciliation_method == "auto"
                    else settings.reconciliation_method)
    if len(recon_detail) and recon_method in set(recon_detail["Method"]):
        rd = recon_detail[recon_detail["Method"] == recon_method][["node_id", "o", "h", "Pred"]]
        honest = honest.merge(rd, on=["node_id", "o", "h"], how="left")
        honest["Policy"] = honest["Pred"].fillna(honest["Policy"])
        honest = honest.drop(columns="Pred")

    # 5) intervals ──────────────────────────────────────────────────────────
    # ranges are built from the errors of the forecasts as PUBLISHED (e.g. regions summed bottom-up)
    comb_pub = rl.published_backtest(comb_bt, tour, recon_method)
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
        S, all_ids, _ = rc.summing_matrix(mn)
        Y = piv.loc[all_ids].to_numpy()
        var = rc.node_variances(all_ids, resid_h1, level_hint)
        E = resid_h1.pivot_table(index="o", columns="node_id", values="e").reindex(columns=all_ids).to_numpy()
        Yt, used = rc.reconcile(Y, S, recon_method, var=var, E=E)
        recon_used[metric] = used
        for i, nid in enumerate(all_ids):
            rec_parts.append(pd.DataFrame({"node_id": nid, "h": np.arange(1, H + 1),
                                           "Forecast": Yt[i], "Base_Forecast": Y[i]}))
    rec = pd.concat(rec_parts, ignore_index=True)
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
    coherence = gr.coherence_gap(fc)
    lap("assemble")

    # 9) summaries ──────────────────────────────────────────────────────────
    totals = _totals(fc, bq, panel.history, levels)
    periods = _periods(fc, panel.history, settings.data.fin_year_start_month)
    accuracy = sel.accuracy_by_horizon(honest, bt.preds, nodes)
    scfg = settings.seasonality
    sidx, syear = seas.seasonal_indices(panel.history, scfg.high_threshold, scfg.low_threshold, scfg.years)
    prod, prod_source = sig.load_sheet(settings_file, "Staffing_Productivity",
                                       CODE_ROOT / "config" / "staffing_productivity_template.csv",
                                       seas.PRODUCTIVITY_COLS)
    staffing = seas.staffing_calendar(fc, sidx, scfg.high_threshold, scfg.low_threshold, scfg.apps_per_fte_month,
                                      productivity=prod)
    ev_raw, events_source = sig.load_sheet(settings_file, "Policy_Events",
                                           CODE_ROOT / "config" / "policy_events_template.csv", sig.EVENT_COLS)
    events = sig.measure_events(ev_raw, panel.history)
    intake_df = (itk.clean_intake(intake_raw, settings.data.drop_regions) if intake_raw is not None
                 else itk.load_intake(intake_file, settings.data.drop_regions) if intake_file else pd.DataFrame())
    i_arr, i_lead, i_mix, i_known = itk.intake_views(intake_df, last_actual)
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
        "intake_extract": (str(intake_file) if intake_df is not None and len(intake_df) else "not available"),
        "intake_known_share": i_known,
    })

    res = RunResult(out_dir=out_dir, run_id=run_id, run_date=run_date, last_actual=last_actual,
                    forecast=fc, totals=totals, periods=periods, history=panel.history, nodes=nodes,
                    series_status=panel.series_status, dq=panel.dq, scores=scores, selection=selection,
                    accuracy=accuracy, policy_comparison=policy_cmp, calibration=calib, block_calibration=block_calib, range_test=range_test, range_tested=range_tested, guardrails=guards, overlay_rules=rules,
                    catalogue=catalogue, failures=bt.failures, track_scorecard=track_sc,
                    bias_test=bias_test, bias_estimates=bias_est, reconciliation_test=recon_test, point_estimate_test=smear_test, seasonal_index=sidx, seasonal_by_year=syear,
                    staffing=staffing, forecast_vs_actual=fva, fva_summary=fva_sum, human_scores=human_scores,
                    events=events, intake_arrivals=i_arr, intake_lead_profile=i_lead, intake_month_mix=i_mix,
                    manifest=manifest)
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
            ly_dates = gk["Date"] - pd.DateOffset(years=1)
            ly = float(sum(hist.get((*key, d), np.nan) for d in ly_dates))
            row = {"Level": key[0], "Country": key[1], "Region": key[2], "Metric": key[3],
                   "Label": g["Label"].iloc[0], "Window": f"Next {k} months",
                   "From": gk["Date"].min(), "To": gk["Date"].max(),
                   "Forecast": point, "Forecast_Adjusted": adj, "Same_Months_Last_Year": ly,
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


def _periods(fc: pd.DataFrame, history: pd.DataFrame, fy_start: int) -> pd.DataFrame:
    """Fiscal quarter / year outlook = actuals to date + forecast remainder, vs same period last year."""
    hist = history.set_index(["Level", "Country", "Region", "Metric", "Date"])["Actual"]
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
            rows.append({"Level": key[0], "Country": key[1], "Region": key[2], "Metric": key[3],
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
        getattr(res, attr).to_csv(p, index=False)
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
                df[c] = pd.to_datetime(df[c])
        kw[attr] = df
    man = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    return RunResult(out_dir=out_dir, run_id=man["run_id"], run_date=man["run_date"],
                     last_actual=pd.Timestamp(man["data"]["last_actual_month"]), manifest=man, **kw)
