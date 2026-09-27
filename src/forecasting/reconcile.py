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


def reconcile(yhat: np.ndarray, S: np.ndarray, method: str, var: Optional[np.ndarray] = None,
              E: Optional[np.ndarray] = None) -> Tuple[np.ndarray, str]:
    """yhat: (n_nodes, H) base forecasts in the row order of S. Returns (y_tilde, method_used)."""
    n_agg = S.shape[0] - S.shape[1]
    used = method
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
