# -*- coding: utf-8 -*-
"""
Rolling-origin backtest (time-series cross-validation).

For each forecast origin o (the last month of training data) every eligible
model is fitted on history[..o] only and forecasts months o+1..o+H. This is
repeated over the last ``n_origins`` month-ends, so every model is judged on the
full 12-month horizon it will be used for, over many origins (not one short window).

Per-node plan (fair comparison — all models of a node see identical origins):
    L = max(12, min(n_months - 12, 36))   training months required at the first origin
    origins  = global origins with >= L training months
    eligible = models whose min_history <= L
A model that fails at any origin is removed for that node and logged.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..data.prepare import Panel
from .models import GLOBAL_MODELS, LOCAL_MODELS, SEASON

LEVEL_CODE = {"Region": 0, "Country": 1, "Organisation": 2}
METRIC_CODE = {"Received": 0, "Submitted": 1}


@dataclass
class BacktestResult:
    preds: pd.DataFrame        # node_id, Model, Origin, h, Target, Pred, Actual
    final: pd.DataFrame        # node_id, Model, h, Target, Pred
    failures: pd.DataFrame     # node_id, Model, Error
    plan: pd.DataFrame         # node_id, Required_Train_Months, N_Origins, Eligible_Models


def global_origins(T: int, n_origins: int) -> List[int]:
    """Origin indices T-1-n .. T-2 (the latest origin still has one realised target)."""
    return list(range(max(SEASON - 1, T - 1 - n_origins), T - 1))


def node_plan(n_months: int, s: int, origins: Sequence[int], models: Sequence[str]) -> tuple:
    L = max(SEASON, min(n_months - SEASON, 3 * SEASON))
    node_origins = [o for o in origins if o - s + 1 >= L]
    eligible = [m for m in models if _min_history(m) <= L]
    return L, node_origins, eligible


def _min_history(model: str) -> int:
    if model in GLOBAL_MODELS:
        return GLOBAL_MODELS[model].min_history
    return LOCAL_MODELS[model].min_history


def _local_task(nid: str, y: np.ndarray, s: int, T: int, node_origins: Sequence[int],
                models: Sequence[str], h: int, cal_month: np.ndarray):
    rows, finals, fails = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in models:
            model = LOCAL_MODELS[m]
            m_rows = []
            state: dict = {}                      # carried forward in time order only (causal)
            try:
                for o in node_origins:
                    f = model(y[: o - s + 1], h, int(cal_month[o + 1]), state=state)
                    for k in range(h):
                        tgt = o + 1 + k
                        if tgt <= T - 1:
                            m_rows.append((nid, m, o, k + 1, tgt, f[k], y[tgt - s]))
                f_final = model(y, h, int(cal_month[T]))
            except Exception as exc:          # recorded, never silently dropped
                fails.append((nid, m, f"{type(exc).__name__}: {exc}"[:240]))
                continue
            rows.extend(m_rows)
            finals.extend((nid, m, k + 1, T + k, f_final[k]) for k in range(h))
    return rows, finals, fails


def _global_task(name, series, cal_month, origin, h, params, seed):
    return GLOBAL_MODELS[name](params, seed).fit_predict(series, cal_month, [origin], h)


def _job(kind, args):
    """Single entry point so local and global jobs share one worker pool."""
    if kind == "local":
        return kind, args[0], _local_task(*args)
    return kind, (args[0], args[3]), _global_task(*args)


def run_backtest(panel: Panel, node_ids: Sequence[str], models: Sequence[str], horizon: int,
                 n_origins: int, lgbm_params: dict, seed: int = 42, n_jobs: int = -1) -> BacktestResult:
    T = len(panel.months)
    months_ext = pd.date_range(panel.months[0], periods=T + horizon, freq="MS")
    cal_month = months_ext.month.to_numpy()
    origins = global_origins(T, n_origins)
    unknown = [m for m in models if m not in LOCAL_MODELS and m not in GLOBAL_MODELS]
    if unknown:
        raise ValueError(f"Unknown models in config: {unknown}")
    local_models = [m for m in models if m in LOCAL_MODELS]
    global_models = [m for m in models if m in GLOBAL_MODELS]

    plans = {}
    for nid in node_ids:
        s = panel.start[nid]
        L, n_orig, elig = node_plan(T - s, s, origins, models)
        plans[nid] = (s, L, n_orig, elig)

    rows, finals, fails = [], [], []
    jobs = []
    # local models: one job per node (cost ~ origins x models)
    for nid in node_ids:
        lm = [m for m in plans[nid][3] if m in local_models]
        jobs.append((len(plans[nid][2]) * len(lm), "local",
                     (nid, panel.y[nid], plans[nid][0], T, plans[nid][2], lm, horizon, cal_month)))
    series = []
    if global_models:
        meta = panel.nodes.set_index("node_id")
        grid = {}
        for nid in node_ids:
            v = np.zeros(T)
            v[plans[nid][0]:] = panel.y[nid]
            grid[nid] = v
        series = []
        for nid in node_ids:
            m_, c_, r_ = nid.split("|")
            partner = f"{'Submitted' if m_ == 'Received' else 'Received'}|{c_}|{r_}"
            series.append({"id": nid, "v": grid[nid], "s": plans[nid][0], "pv": grid.get(partner),
                           "metric_code": METRIC_CODE[meta.loc[nid, "Metric"]],
                           "level_code": LEVEL_CODE[meta.loc[nid, "Level"]]})
        # global models: one job per (model, origin), incl. the final origin T-1 — identical origins
        for g in global_models:
            for o in origins + [T - 1]:
                jobs.append((1e6, "global", (g, series, cal_month, o, horizon,
                                             lgbm_params if g == "GlobalLGBM" else {}, seed)))
    jobs.sort(key=lambda j: -j[0])                    # heaviest first -> better load balance
    outs = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_job)(kind, args) for _, kind, args in jobs)
    merged: Dict[str, Dict[tuple, np.ndarray]] = {g: {} for g in global_models}
    for kind, key, res in outs:
        if kind == "local":
            r, f, e = res
            rows.extend(r); finals.extend(f); fails.extend(e)
        else:
            merged[key[0]].update(res)
    if global_models:
        for g in global_models:
            for nid in node_ids:
                s, _, n_orig, elig = plans[nid]
                if g not in elig:
                    continue
                need = list(n_orig) + [T - 1]
                if not all((nid, o) in merged[g] for o in need):
                    fails.append((nid, g, "missing predictions for some origins"))
                    continue
                y = panel.y[nid]
                for o in n_orig:
                    f = merged[g][(nid, o)]
                    for k in range(horizon):
                        tgt = o + 1 + k
                        if tgt <= T - 1:
                            rows.append((nid, g, o, k + 1, tgt, f[k], y[tgt - s]))
                f = merged[g][(nid, T - 1)]
                finals.extend((nid, g, k + 1, T + k, f[k]) for k in range(horizon))

    preds = pd.DataFrame(rows, columns=["node_id", "Model", "o", "h", "t", "Pred", "Actual"])
    preds["Origin"] = months_ext[preds["o"].to_numpy()]
    preds["Target"] = months_ext[preds["t"].to_numpy()]
    final = pd.DataFrame(finals, columns=["node_id", "Model", "h", "t", "Pred"])
    final["Target"] = months_ext[final["t"].to_numpy()]
    failures = pd.DataFrame(fails, columns=["node_id", "Model", "Error"])
    # a model without a final forecast cannot be used for that node
    ok = final[["node_id", "Model"]].drop_duplicates()
    preds = preds.merge(ok, on=["node_id", "Model"], how="inner")
    plan = pd.DataFrame([
        {"node_id": nid, "Required_Train_Months": p[1], "N_Origins": len(p[2]),
         "Eligible_Models": ", ".join(p[3])} for nid, p in plans.items()
    ])
    return BacktestResult(
        preds=preds[["node_id", "Model", "o", "Origin", "h", "t", "Target", "Pred", "Actual"]],
        final=final[["node_id", "Model", "h", "t", "Target", "Pred"]],
        failures=failures, plan=plan)
