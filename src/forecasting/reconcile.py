# -*- coding: utf-8 -*-
"""
Forecast reconciliation: regions, country totals and the organisation total are
forecast independently, then combined so they add up exactly.

    y_tilde = S (S' W^-1 S)^-1 S' W^-1 y_hat         (Wickramasuriya et al., 2019)

* ``wls_var``     W = diag(one-step error variance of each node)   [default]
* ``mint_shrink`` W = shrunk covariance of one-step errors (Schäfer-Strimmer)
* ``bottom_up``   aggregates = sums of the region forecasts
Negative bottom-level values are set to 0 and aggregates re-summed (coherent, non-negative).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def summing_matrix(nodes: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
    """nodes for ONE metric. Rows: organisation, countries, regions. Columns: regions."""
    bottom = nodes[nodes["Level"] == "Region"]
    aggs = pd.concat([nodes[nodes["Level"] == "Organisation"], nodes[nodes["Level"] == "Country"]])
    b_ids = bottom["node_id"].tolist()
    all_ids = aggs["node_id"].tolist() + b_ids
    S = np.zeros((len(all_ids), len(b_ids)))
    for i, (_, a) in enumerate(aggs.iterrows()):
        if a["Level"] == "Organisation":
            S[i, :] = 1.0
        else:
            S[i, :] = (bottom["Country"] == a["Country"]).to_numpy(dtype=float)
    S[len(aggs):, :] = np.eye(len(b_ids))
    return S, all_ids, b_ids


def shrink_covariance(E: np.ndarray) -> np.ndarray:
    """Schäfer-Strimmer shrinkage of the error covariance towards its diagonal."""
    n = E.shape[0]
    X = E - E.mean(axis=0)
    S = X.T @ X / n
    sd = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    Z = X / sd
    R = Z.T @ Z / n
    W = (Z[:, :, None] * Z[:, None, :]) - R
    var_r = (W ** 2).sum(axis=0) * n / (n - 1) ** 3
    off = ~np.eye(len(sd), dtype=bool)
    lam = float(np.clip(var_r[off].sum() / max((R[off] ** 2).sum(), 1e-12), 0.0, 1.0))
    return lam * np.diag(np.diag(S)) + (1 - lam) * S


def region_shares(S: np.ndarray, b_ids: List[str], last12: dict) -> np.ndarray:
    """Each region's share of its country over the last 12 realised months (equal split if no history)."""
    n_agg = S.shape[0] - S.shape[1]
    v = np.array([max(float(last12.get(b, 0.0)), 0.0) for b in b_ids])
    sh = np.full(len(b_ids), np.nan)
    for i in range(1, n_agg):                          # country rows (row 0 = organisation)
        m = S[i] == 1
        tot = v[m].sum()
        sh[m] = v[m] / tot if tot > 0 else 1.0 / m.sum()
    return sh


def reconcile(yhat: np.ndarray, S: np.ndarray, method: str, var: Optional[np.ndarray] = None,
              E: Optional[np.ndarray] = None, shares: Optional[np.ndarray] = None) -> Tuple[np.ndarray, str]:
    """yhat: (n_nodes, H) base forecasts in the row order of S. Returns (y_tilde, method_used).
    middle_out: each country keeps its own forecast, split to its regions by `shares`; regions of
    single-region countries keep their own; the organisation is the sum."""
    n_agg = S.shape[0] - S.shape[1]
    used = method
    if method == "middle_out":
        if shares is None:
            used = "bottom_up"
        else:
            b = yhat[n_agg:].copy()
            for i in range(1, n_agg):
                m = S[i] == 1
                b[m] = yhat[i][None, :] * shares[m][:, None] if b.ndim == 2 else yhat[i] * shares[m]
            b = np.clip(b, 0.0, None)
            return S @ b, used
    if method == "mint_shrink" and (E is None or np.isnan(E).any() or E.shape[0] < 12):
        used = "wls_var"
    if used == "bottom_up":
        b = yhat[n_agg:]
    else:
        if used == "mint_shrink":
            W = shrink_covariance(E)
        else:
            if var is None:
                raise ValueError("wls_var needs error variances")
            W = np.diag(np.clip(var, 1e-9, None))
        Winv = np.linalg.pinv(W)
        P = np.linalg.solve(S.T @ Winv @ S, S.T @ Winv)
        b = P @ yhat
    b = np.clip(b, 0.0, None)
    return S @ b, used


def node_variances(ids: List[str], resid_h1: pd.DataFrame, level_hint: pd.Series) -> np.ndarray:
    """
    One-step error variance per node from the backtest; nodes without residuals
    (emerging) get the median relative variance x their level^2.
    """
    v = resid_h1.groupby("node_id")["e"].var()
    lvl = level_hint.reindex(ids).fillna(0.0)
    rel = (v / (lvl.reindex(v.index) + 1.0) ** 2).replace([np.inf, -np.inf], np.nan).dropna()
    med = float(rel.median()) if len(rel) else 0.05
    out = []
    for nid in ids:
        if nid in v.index and np.isfinite(v[nid]) and v[nid] > 0:
            out.append(float(v[nid]))
        else:
            out.append(med * (float(lvl.get(nid, 0.0)) + 1.0) ** 2)
    return np.array(out)


METHODS = ("none", "bottom_up", "wls_var", "mint_shrink", "middle_out")


def walk_forward_test(honest: pd.DataFrame, comb: pd.DataFrame, nodes: pd.DataFrame, return_detail: bool = False):
    """
    Replay every reconciliation method on the walk-forward forecasts: at each evaluation origin e
    the weights come only from one-step errors realised by e (targets <= e). Mature series only
    (emerging series have no walk-forward forecasts). Returns WAPE by metric x level x method.
    """
    if honest is None or honest.empty:
        return (pd.DataFrame(), pd.DataFrame()) if return_detail else pd.DataFrame()
    meta = nodes.set_index("node_id")
    resid = comb[comb["h"] == 1].assign(e=lambda d: d["Actual"] - d["Pred"])
    act_t = comb.drop_duplicates(["node_id", "t"]).pivot_table(index="t", columns="node_id", values="Actual")
    out = []
    for metric in ("Received", "Submitted"):
        mn = nodes[(nodes["Metric"] == metric) & (nodes["Status"] == "Mature")]
        if mn.empty:
            continue
        S, ids, b_ids = summing_matrix(mn)
        h_m = honest[honest["node_id"].isin(ids)]
        for (e, h), g in h_m.groupby(["o", "h"]):
            g = g.set_index("node_id").reindex(ids)
            if g["Policy"].isna().any():
                continue
            yhat = g["Policy"].to_numpy()[:, None]
            act = g["Actual"].to_numpy()
            past = resid[resid["t"] <= e]
            var = node_variances(ids, past, pd.Series(yhat[:, 0], index=ids))
            E = past.pivot_table(index="o", columns="node_id", values="e").reindex(columns=ids).dropna().to_numpy()
            win = act_t.loc[(act_t.index <= e) & (act_t.index > e - 12)]
            sh = region_shares(S, b_ids, win.sum().to_dict())
            for method in METHODS:
                if method == "none":
                    yt, used = yhat, "none"
                else:
                    yt, used = reconcile(yhat, S, method, var=var, E=E if len(E) else None, shares=sh)
                for i, nid in enumerate(ids):
                    out.append((nid, metric, meta.loc[nid, "Level"], method, used, e, h, float(yt[i, 0]), float(act[i])))
    if not out:
        return (pd.DataFrame(), pd.DataFrame()) if return_detail else pd.DataFrame()
    d = pd.DataFrame(out, columns=["node_id", "Metric", "Level", "Method", "Method_Used", "o", "h", "Pred", "Actual"])
    d["AE"] = (d["Pred"] - d["Actual"]).abs()
    d["Fell_Back"] = (d["Method_Used"] != d["Method"]).astype(float)
    g = d.groupby(["Metric", "Level", "Method"])
    res = (100 * g["AE"].sum() / g["Actual"].sum()).rename("WAPE_Pct").reset_index()
    res["N"] = g.size().to_numpy()
    res["Fell_Back_To_WLS_Pct"] = (100 * g["Fell_Back"].mean()).to_numpy()
    return (res, d) if return_detail else res


def reconcile_backtest(comb: pd.DataFrame, nodes: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Every backtest forecast as it would have been PUBLISHED with `method`: at origin o the weights use
    only one-step errors realised by o (targets <= o), exactly as in walk_forward_test. Mature series only.
    """
    resid = comb[comb["h"] == 1].assign(e=lambda d: d["Actual"] - d["Pred"])
    act_t = comb.drop_duplicates(["node_id", "t"]).pivot_table(index="t", columns="node_id", values="Actual")
    parts = []
    for metric in ("Received", "Submitted"):
        mn = nodes[(nodes["Metric"] == metric) & (nodes["Status"] == "Mature")]
        if mn.empty:
            continue
        S, ids, b_ids = summing_matrix(mn)
        cm = comb[comb["node_id"].isin(ids)]
        for (o, h), g in cm.groupby(["o", "h"]):
            gi = g.set_index("node_id").reindex(ids)
            if gi["Pred"].isna().any():
                continue
            yhat = gi["Pred"].to_numpy()[:, None]
            past = resid[resid["t"] <= o]
            var = node_variances(ids, past, pd.Series(yhat[:, 0], index=ids))
            E = past.pivot_table(index="o", columns="node_id", values="e").reindex(columns=ids).dropna().to_numpy()
            win = act_t.loc[(act_t.index <= o) & (act_t.index > o - 12)]
            yt, _ = reconcile(yhat, S, method, var=var, E=E if len(E) else None,
                              shares=region_shares(S, b_ids, win.sum().to_dict()))
            out = gi.reset_index()
            out["Pred"] = yt[:, 0]
            parts.append(out)
    if not parts:
        return comb
    res = pd.concat(parts, ignore_index=True)
    return res[comb.columns]


def quarter_errors(detail: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Median abs % error of next-k-month totals per metric x level x method (the decision metric)."""
    if detail is None or detail.empty:
        return pd.DataFrame()
    d = detail[detail["h"] <= k]
    full = d.groupby(["node_id", "Method", "o"])["h"].transform("nunique") == k
    q = d[full].groupby(["node_id", "Metric", "Level", "Method", "o"])[["Pred", "Actual"]].sum().reset_index()
    q = q[q["Actual"] > 0]
    q["APE"] = 100 * (q["Pred"] - q["Actual"]).abs() / q["Actual"]
    return q.groupby(["Metric", "Level", "Method"])["APE"].median().rename("Quarter_Error_Pct").reset_index()


def choose_method_quarter(detail: pd.DataFrame, candidates=("bottom_up", "wls_var", "mint_shrink", "middle_out")) -> str:
    """Method with the lowest next-quarter error averaged over metric x level (each level counts equally)."""
    q = quarter_errors(detail)
    q = q[q["Method"].isin(candidates)] if len(q) else q
    if q.empty:
        return "bottom_up"
    return str(q.groupby("Method")["Quarter_Error_Pct"].mean().idxmin())


def choose_method(test: pd.DataFrame, candidates=("bottom_up", "wls_var", "mint_shrink")) -> str:
    """Method with the lowest WAPE averaged over metric x level (each level counts equally)."""
    t = test[test["Method"].isin(candidates)]
    if t.empty:
        return "wls_var"
    return str(t.groupby("Method")["WAPE_Pct"].mean().idxmin())
