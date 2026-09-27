# -*- coding: utf-8 -*-
"""
Model scoring, selection and honest evaluation of the selection policy.

Score  = horizon-balanced MAE: the MAE at each horizon h=1..12 (over all origins),
         averaged across horizons, so a model that is good next month but explodes
         at month 9 cannot win.
Policy = how a node's forecast is formed from the candidates:
         ``mean_all`` equal-weight average of every eligible model (default), or
         ``top_k``    equal-weight average of the best k that beat the benchmark.
         Equal-weight averaging is robust to selection noise (the "forecast
         combination puzzle"; M3/M4 evidence) — and on this data it won the
         walk-forward comparison (docs/experiments/2026-09-27_combination_policy.md).
Honest = every policy is replayed walk-forward: at each evaluation origin e the
         choice uses only errors whose target month is <= e. These errors estimate
         the real accuracy of each *procedure*, free of selection bias, and are
         recomputed every run (Policy Comparison sheet).
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

MIN_KNOWN_PAIRS = 12
H_BUCKETS = {"1-3": (1, 3), "4-6": (4, 6), "7-12": (7, 12)}
POLICIES = ("mean_all", "top_k")
METHOD_LABELS = {
    "Policy": "Selected method (walk-forward)",
    "Mean_All": "Equal-weight mean of all models",
    "Top_K": "Top-k combination",
    "Best_Single": "Best single model",
    "Benchmark": "Seasonal naive (benchmark)",
}


def _hb_mae(df: pd.DataFrame, by: List[str]) -> pd.Series:
    """Horizon-balanced MAE for groups in `by` (df has Pred, Actual, h)."""
    e = df.assign(AE=(df["Pred"] - df["Actual"]).abs())
    per_h = e.groupby(by + ["h"])["AE"].mean()
    return per_h.groupby(level=list(range(len(by)))).mean()


def score_models(preds: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    e = preds.assign(AE=(preds["Pred"] - preds["Actual"]).abs(), E=preds["Pred"] - preds["Actual"])
    g = e.groupby(["node_id", "Model"])
    sc = pd.DataFrame({
        "HB_MAE": _hb_mae(preds, ["node_id", "Model"]),
        "WAPE": g["AE"].sum() / g["Actual"].sum().replace(0, np.nan),
        "Bias_Pct": 100 * g["E"].sum() / g["Actual"].sum().replace(0, np.nan),
        "N_Origins": g["o"].nunique(),
        "N_Pairs": g.size(),
    })
    for name, (a, b) in H_BUCKETS.items():
        sub = e[e["h"].between(a, b)]
        sc[f"MAE_h{name}"] = sub.groupby(["node_id", "Model"])["AE"].mean()
    sc = sc.reset_index()
    bench = sc[sc["Model"] == benchmark].set_index("node_id")["HB_MAE"]
    sc["Rel_to_Benchmark"] = sc["HB_MAE"] / sc["node_id"].map(bench).replace(0, np.nan)
    sc["Rank"] = sc.groupby("node_id")["HB_MAE"].rank(method="first").astype(int)
    return sc.sort_values(["node_id", "Rank"]).reset_index(drop=True)


def _choose(rank_tbl: pd.DataFrame, benchmark: str, top_k: int, must_beat: bool) -> List[str]:
    """rank_tbl: Model, HB_MAE, Rel_to_Benchmark sorted ascending by HB_MAE."""
    cand = rank_tbl
    if must_beat:
        cand = cand[cand["Rel_to_Benchmark"].fillna(np.inf) <= 1.0]
    chosen = cand["Model"].head(top_k).tolist()
    return chosen or [benchmark]


def select_models(scores: pd.DataFrame, benchmark: str, top_k: int, must_beat: bool,
                  policy: str = "mean_all") -> pd.DataFrame:
    if policy not in POLICIES:
        raise ValueError(f"Unknown selection policy '{policy}'; use one of {POLICIES}")
    rows = []
    for nid, g in scores.groupby("node_id", sort=False):
        g = g.sort_values("HB_MAE")
        best, runner = g.iloc[0], (g.iloc[1] if len(g) > 1 else None)
        margin = (100 * (runner["HB_MAE"] - best["HB_MAE"]) / best["HB_MAE"]
                  if runner is not None and best["HB_MAE"] > 0 else np.nan)
        rel = g.set_index("Model")["Rel_to_Benchmark"]
        n_orig = int(g["N_Origins"].max())
        ranking = ", ".join(f"{m} {rel[m]:.2f}" for m in g["Model"])
        if policy == "mean_all":
            chosen = g["Model"].tolist()
            reason = (f"Equal-weight average of all {len(chosen)} eligible models (policy validated walk-forward, "
                      f"see Policy Comparison). Backtest over {n_orig} rolling origins x 12 months; error relative "
                      f"to seasonal naive: {ranking}.")
        else:
            chosen = _choose(g, benchmark, top_k, must_beat)
            if chosen == [benchmark] and (len(g) == 1 or rel.drop(benchmark, errors="ignore").min() > 1):
                reason = (f"No candidate beat the seasonal-naive benchmark over {n_orig} origins x 12 months; "
                          f"benchmark used.")
            else:
                rels = ", ".join(f"{m} {rel[m]:.2f}" for m in chosen)
                reason = (f"Equal-weight combination of the best {len(chosen)} of {len(g)} candidates by "
                          f"12-month horizon-balanced MAE over {n_orig} rolling origins; error relative to "
                          f"seasonal naive: {rels}.")
        rows.append({
            "node_id": nid, "Selected_Models": " + ".join(chosen), "N_Selected": len(chosen),
            "N_Candidates": len(g), "Best_Model": best["Model"], "Best_HB_MAE": best["HB_MAE"],
            "Runner_Up": runner["Model"] if runner is not None else "",
            "Margin_Best_vs_RunnerUp_Pct": margin,
            "Benchmark_HB_MAE": float(g.loc[g["Model"] == benchmark, "HB_MAE"].iloc[0])
            if (g["Model"] == benchmark).any() else np.nan,
            "Selection_Reason": reason,
        })
    return pd.DataFrame(rows)


def combine(preds: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    """Equal-weight mean of each node's selected models. `selected`: node_id -> 'A + B'."""
    sel = selected.rename_axis("node_id").str.split(r" \+ ").explode().rename("Model").reset_index()
    m = preds.merge(sel, on=["node_id", "Model"], how="inner")
    keys = [c for c in ("node_id", "o", "Origin", "h", "t", "Target") if c in m.columns]
    agg = {"Pred": "mean"}
    if "Actual" in m.columns:
        agg["Actual"] = "first"
    return m.groupby(keys, as_index=False).agg(agg)


def honest_evaluation(preds: pd.DataFrame, benchmark: str, top_k: int, must_beat: bool,
                      n_eval_origins: int, policy: str = "mean_all") -> pd.DataFrame:
    """Walk-forward replay of every policy (see module docstring). `Policy` = the configured one."""
    out = []
    for nid, g in preds.groupby("node_id", sort=False):
        origins = np.sort(g["o"].unique())
        if benchmark not in set(g["Model"]):
            continue
        wide = g.pivot_table(index=["o", "h", "t"], columns="Model", values="Pred")
        act = g.drop_duplicates(["o", "h"]).set_index(["o", "h", "t"])["Actual"].reindex(wide.index)
        o_idx = wide.index.get_level_values("o").to_numpy()
        t_idx = wide.index.get_level_values("t").to_numpy()
        h_idx = wide.index.get_level_values("h").to_numpy()
        ae = (wide.sub(act, axis=0)).abs()
        for e in origins[-n_eval_origins:]:
            known = (o_idx < e) & (t_idx <= e)
            if known.sum() < MIN_KNOWN_PAIRS:
                chosen, best = [benchmark], benchmark
            else:
                hb = ae[known].groupby(h_idx[known]).mean().mean()
                tbl = pd.DataFrame({"Model": hb.index, "HB_MAE": hb.values})
                tbl["Rel_to_Benchmark"] = tbl["HB_MAE"] / hb[benchmark] if hb[benchmark] > 0 else np.nan
                tbl = tbl.sort_values("HB_MAE")
                chosen = _choose(tbl, benchmark, top_k, must_beat)
                best = tbl["Model"].iloc[0]
            at = o_idx == e
            w = wide[at]
            mean_all = w.mean(axis=1).to_numpy()
            top = w[chosen].mean(axis=1).to_numpy()
            out.append(pd.DataFrame({
                "node_id": nid, "o": e, "h": h_idx[at], "t": t_idx[at],
                "Actual": act[at].to_numpy(),
                "Policy": mean_all if policy == "mean_all" else top,
                "Mean_All": mean_all,
                "Top_K": top,
                "Best_Single": w[best].to_numpy(),
                "Benchmark": w[benchmark].to_numpy(),
                "Top_K_Models": " + ".join(chosen),
            }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def policy_comparison(honest: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward WAPE of each policy by level x metric x horizon bucket, plus node-level skill."""
    if honest.empty:
        return pd.DataFrame()
    meta = nodes.set_index("node_id")[["Level", "Metric"]]
    hz = honest.join(meta, on="node_id")
    hz["Bucket"] = pd.cut(hz["h"], [0, 3, 6, 12], labels=list(H_BUCKETS))
    cols = [c for c in ("Mean_All", "Top_K", "Best_Single", "Benchmark") if c in hz.columns]
    rows = []
    for (lvl, met, b), g in hz.groupby(["Level", "Metric", "Bucket"], observed=True):
        den = g["Actual"].sum()
        for c in cols:
            rows.append({"Level": lvl, "Metric": met, "Months_Ahead": str(b), "Method": METHOD_LABELS[c],
                         "WAPE_Pct": 100 * (g[c] - g["Actual"]).abs().sum() / den if den else np.nan,
                         "Bias_Pct": 100 * (g[c] - g["Actual"]).sum() / den if den else np.nan, "N": len(g)})
    # scale-free summary: per-node horizon-balanced MAE relative to the benchmark
    ae = {c: (hz[c] - hz["Actual"]).abs() for c in cols}
    per = pd.DataFrame({"node_id": hz["node_id"], "h": hz["h"], **ae}).groupby(["node_id", "h"]).mean()
    per = per.groupby(level=0).mean()
    rel = per.div(per["Benchmark"].replace(0, np.nan), axis=0)
    for c in cols:
        rows.append({"Level": "All series", "Metric": "Both", "Months_Ahead": "1-12", "Method": METHOD_LABELS[c],
                     "Median_Error_vs_Benchmark": float(rel[c].median()),
                     "Pct_Series_Beating_Benchmark": float(100 * (rel[c] < 1).mean()), "N": int(rel[c].notna().sum())})
    return pd.DataFrame(rows)


def honest_summary(honest: pd.DataFrame) -> pd.DataFrame:
    if honest.empty:
        return pd.DataFrame(columns=["node_id", "Honest_N", "Honest_WAPE_Policy", "Honest_WAPE_Benchmark",
                                     "Honest_WAPE_BestSingle", "Honest_Skill_vs_Benchmark"])
    e = honest.assign(P=(honest["Policy"] - honest["Actual"]).abs(),
                      B=(honest["Benchmark"] - honest["Actual"]).abs(),
                      S=(honest["Best_Single"] - honest["Actual"]).abs())
    g = e.groupby("node_id")
    den = g["Actual"].sum().replace(0, np.nan)
    s = pd.DataFrame({
        "Honest_N": g.size(),
        "Honest_WAPE_Policy": g["P"].sum() / den,
        "Honest_WAPE_Benchmark": g["B"].sum() / den,
        "Honest_WAPE_BestSingle": g["S"].sum() / den,
    })
    s["Honest_Skill_vs_Benchmark"] = 1 - g["P"].mean() / g["B"].mean().replace(0, np.nan)
    return s.reset_index()


def accuracy_by_horizon(honest: pd.DataFrame, preds: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """WAPE by level x metric x horizon for the policy, benchmark and every single model (honest)."""
    meta = nodes.set_index("node_id")[["Level", "Metric"]]
    rows = []
    if not honest.empty:
        hz = honest.join(meta, on="node_id")
        for col, label in METHOD_LABELS.items():
            if col not in hz.columns:
                continue
            hz["AE"] = (hz[col] - hz["Actual"]).abs()
            t = hz.groupby(["Level", "Metric", "h"]).agg(AE=("AE", "sum"), A=("Actual", "sum"), N=("AE", "size"))
            t["Method"] = label
            rows.append(t.reset_index())
        keys = honest[["node_id", "o"]].drop_duplicates()
        pm = preds.merge(keys, on=["node_id", "o"]).join(meta, on="node_id")
        pm["AE"] = (pm["Pred"] - pm["Actual"]).abs()
        t = pm.groupby(["Level", "Metric", "h", "Model"]).agg(AE=("AE", "sum"), A=("Actual", "sum"),
                                                              N=("AE", "size")).reset_index()
        rows.append(t.rename(columns={"Model": "Method"}))
    if not rows:
        return pd.DataFrame()
    acc = pd.concat(rows, ignore_index=True)
    acc["WAPE_Pct"] = 100 * acc["AE"] / acc["A"].replace(0, np.nan)
    return acc[["Level", "Metric", "h", "Method", "WAPE_Pct", "N"]]
