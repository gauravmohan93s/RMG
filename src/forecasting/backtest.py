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
from .models import LOCAL_MODELS, SEASON, GlobalLGBMDirect

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
    if model == GlobalLGBMDirect.name:
        return GlobalLGBMDirect.min_history
    return LOCAL_MODELS[model].min_history


def _local_task(nid: str, y: np.ndarray, s: int, T: int, node_origins: Sequence[int],
                models: Sequence[str], h: int, cal_month: np.ndarray):
    rows, finals, fails = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in models:
            model = LOCAL_MODELS[m]
            m_rows = []
            try:
                for o in node_origins:
                    f = model(y[: o - s + 1], h, int(cal_month[o + 1]))
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


def _lgbm_task(series, cal_month, origin, h, params, seed):
    return GlobalLGBMDirect(params, seed).fit_predict(series, cal_month, [origin], h)


def run_backtest(panel: Panel, node_ids: Sequence[str], models: Sequence[str], horizon: int,
                 n_origins: int, lgbm_params: dict, seed: int = 42, n_jobs: int = -1) -> BacktestResult:
    T = len(panel.months)
    months_ext = pd.date_range(panel.months[0], periods=T + horizon, freq="MS")
    cal_month = months_ext.month.to_numpy()
    origins = global_origins(T, n_origins)
    local_models = [m for m in models if m in LOCAL_MODELS]
    use_lgbm = GlobalLGBMDirect.name in models

    plans = {}
    for nid in node_ids:
        s = panel.start[nid]
        L, n_orig, elig = node_plan(T - s, s, origins, models)
        plans[nid] = (s, L, n_orig, elig)

    # local models: one task per node
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_local_task)(nid, panel.y[nid], plans[nid][0], T, plans[nid][2],
                             [m for m in plans[nid][3] if m in local_models], horizon, cal_month)
        for nid in node_ids
    )
    rows, finals, fails = [], [], []
    for r, f, e in results:
        rows.extend(r); finals.extend(f); fails.extend(e)

    # global model: one task per origin (+ the final origin T-1)
    if use_lgbm:
        series = []
        for nid in node_ids:
            s = plans[nid][0]
            v = np.zeros(T)
            v[s:] = panel.y[nid]
            meta = panel.nodes.set_index("node_id").loc[nid]
            series.append({"id": nid, "v": v, "s": s, "metric_code": METRIC_CODE[meta["Metric"]],
                           "level_code": LEVEL_CODE[meta["Level"]]})
        lg = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_lgbm_task)(series, cal_month, o, horizon, lgbm_params, seed)
            for o in origins + [T - 1]
        )
        merged: Dict[tuple, np.ndarray] = {}
        for d in lg:
            merged.update(d)
        for nid in node_ids:
            s, _, n_orig, elig = plans[nid]
            if GlobalLGBMDirect.name not in elig:
                continue
            need = list(n_orig) + [T - 1]
            if not all((nid, o) in merged for o in need):
                fails.append((nid, GlobalLGBMDirect.name, "missing predictions for some origins"))
                continue
            y = panel.y[nid]
            for o in n_orig:
                f = merged[(nid, o)]
                for k in range(horizon):
                    tgt = o + 1 + k
                    if tgt <= T - 1:
                        rows.append((nid, GlobalLGBMDirect.name, o, k + 1, tgt, f[k], y[tgt - s]))
            f = merged[(nid, T - 1)]
            finals.extend((nid, GlobalLGBMDirect.name, k + 1, T + k, f[k]) for k in range(horizon))

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
