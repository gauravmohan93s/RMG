# -*- coding: utf-8 -*-
"""
Forecast ranges chosen on evidence (v2.2).

Why this exists: in v2.1 each tail of a range was widened separately until it covered the last
12 months of walk-forward errors. Those 12 months were one regime (a surge the models under-
forecast), so a past bias became a permanent upside tail (e.g. quarter −14% / +45%). The point
forecast is not bias-corrected, because bias corrections did not win the walk-forward test; the
ranges must be consistent with that: a bias that flips sign between regimes is uncertainty, not a
direction.

Two candidate ways to turn past log errors r = log(1+actual) − log(1+forecast) into a range:
  empirical  recency-weighted empirical quantiles of r (skew allowed)
  centred    centred on the forecast: ± z × robust scale, where the scale is the recency-weighted
             median absolute deviation (×1.4826), shrunk toward the pooled scale of
             comparable series
Both are replayed sequentially: at each of the last `n_test` origins the range is fitted with only
the errors realised by then and scored against what happened, with the interval score (Gneiting &
Raftery 2007) — a proper scoring rule that rewards narrow ranges and penalises misses in proportion
to how far outside they fall. Per level group (Organisation+Country / Region) and window, the method
with the lower score is used. A single symmetric width factor in [0.8, 1.5] then corrects clear
over- or under-coverage (N >= 30, off by > 5 points); it never shifts one tail.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from .intervals import BLOCKS, RECENCY_HALF_LIFE, _band, _enforce, _wq

METHODS = ("empirical", "centred")
SHRINK_N0 = 8.0                 # pseudo-observations of the pooled scale in the centred method
SCALE_BOUNDS = (0.8, 1.5)       # symmetric width factor
SCALE_MIN_N = 30
SCALE_TRIGGER_PP = 5.0
MONTH_WINDOW = 1                # monthly ranges use errors at h-1..h+1
REF_KEY = 3                     # quarter totals: the reference volatility for location-scale pooling


def group_of(level: str) -> str:
    return "Aggregate" if level in ("Organisation", "Country") else "Region"


def _errors(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Log errors per (node, origin, key): key = h for months, k for k-month totals."""
    if kind == "month":
        d = frame[["node_id", "o", "h", "Pred", "Actual"]].rename(columns={"h": "key"}).copy()
        d["end"] = d["o"] + d["key"]
    else:
        parts = []
        for k in BLOCKS:
            b = frame[frame["h"] <= k]
            full = b.groupby(["node_id", "o"])["h"].transform("nunique") == k
            s = b[full].groupby(["node_id", "o"], as_index=False)[["Pred", "Actual"]].sum()
            s["key"], s["end"] = k, s["o"] + k
            parts.append(s)
        d = pd.concat(parts, ignore_index=True)
    d["r"] = np.log1p(d["Actual"].clip(lower=0)) - np.log1p(d["Pred"].clip(lower=0))
    return d


class _Store:
    """Errors indexed by node and by (Metric, Level) pool, as numpy arrays."""

    def __init__(self, err: pd.DataFrame, meta: pd.DataFrame):
        e = err.merge(meta, left_on="node_id", right_index=True, how="inner")
        cols = ["key", "o", "end", "r"]
        self.node = {n: tuple(g[c].to_numpy() for c in cols) for n, g in e.groupby("node_id")}
        self.pool = {k: tuple(g[c].to_numpy() for c in cols) for k, g in e.groupby(["Metric", "Level"])}
        self.members = {k: g["node_id"].unique().tolist() for k, g in e.groupby(["Metric", "Level"])}
        self._cache: dict = {}

    def ref_scale(self, nid, at, ref_key=REF_KEY):
        """Robust spread of the series' own quarter-total errors realised by `at` (well estimated)."""
        r, _ = self.pick(self.node.get(nid), ref_key, at, 0)
        if len(r) < 6:
            return np.nan
        return 1.4826 * float(np.median(np.abs(r - np.median(r))))

    def scaled_pool(self, pool_key, key, at):
        """Location-scale pooling: every member's k-month errors divided by that member's own quarter
        spread, so volatile and calm series can share one SHAPE of long-window uncertainty."""
        ck = (pool_key, key, at)
        if ck not in self._cache:
            z = []
            for m in self.members.get(pool_key, []):
                s = self.ref_scale(m, at)
                r, _ = self.pick(self.node.get(m), key, at, 0)
                if np.isfinite(s) and s > 0 and len(r):
                    z.append(r / s)
            self._cache[ck] = np.concatenate(z) if z else np.empty(0)
        return self._cache[ck]

    @staticmethod
    def pick(arr, key, at, window):
        if arr is None:
            return np.empty(0), np.empty(0)
        k_, o_, end_, r_ = arr
        m = (np.abs(k_ - key) <= window) & (end_ <= at)
        return r_[m], o_[m]


MIN_INDEPENDENT = 4    # k-month totals: own errors used only with >= 4 non-overlapping windows' worth
LS_OWN_BLEND = True    # v2.2.1: blend own long-window spread into the location-scale estimate
LS_PRIOR_N = 2.0       # weight of the pooled estimate, in independent windows
# Tested on 2026-09 data (docs/experiments/2026-10-25_long_window_blend.md): better for the Region group
# (interval score 12 m 5.27 -> 5.07, 6 m 3.26 -> 2.79, coverage unchanged), worse for countries and the
# organisation (6 m 1.39 -> 1.50, country coverage 66% -> 57%). So it applies to the Region group only.
LS_BLEND_GROUPS = ("Region",)


def _own_enough(n: int, min_obs: int, overlap: int) -> bool:
    """Consecutive k-month windows share k-1 months, so n of them carry only about n/k independent
    observations. A series' own spread is used only when that effective sample is large enough;
    otherwise the pool of comparable series is used (a 12-month range from ~2 episodes is noise)."""
    return n >= min_obs and (overlap <= 1 or n / overlap >= MIN_INDEPENDENT)


def _fit(method: str, own_r, own_o, pool_r, at, levels, min_obs, overlap: int = 1) -> Dict[str, float]:
    use_own = _own_enough(len(own_r), min_obs, overlap)
    if not use_own and len(pool_r) < 3:
        return {}
    out = {}
    if method == "empirical":
        r, w = (own_r, 0.5 ** ((at - own_o) / RECENCY_HALF_LIFE)) if use_own else (pool_r, np.ones(len(pool_r)))
        for L in levels:
            a = (100 - L) / 200
            out[f"q_lo{L}"], out[f"q_hi{L}"] = _wq(r, w, a), _wq(r, w, 1 - a)
        return out
    # spread = recency-weighted median absolute deviation. Bias is not built in (it flips between
    # regimes); if a bias persisted, the sequential test would show under-coverage and the width
    # factor (or the empirical method) would take over — tested, not assumed.
    ps = None
    if len(pool_r) >= 3:
        ps = 1.4826 * float(np.median(np.abs(pool_r - np.median(pool_r))))
    if use_own:
        w = 0.5 ** ((at - own_o) / RECENCY_HALF_LIFE)
        c = _wq(own_r, w, 0.5)
        s = 1.4826 * _wq(np.abs(own_r - c), w, 0.5)
        if ps is not None:
            n = len(own_r)
            s = float(np.sqrt((n * s ** 2 + SHRINK_N0 * ps ** 2) / (n + SHRINK_N0)))
    else:
        s = ps
    for L in levels:
        z = float(norm.ppf(0.5 + L / 200))
        out[f"q_lo{L}"], out[f"q_hi{L}"] = -z * s, z * s
    return out


def _inputs(store: "_Store", kind: str, nid: str, pool_key: tuple, key: int, at: int, min_obs: int):
    """(own errors, own origins, pool errors, multiplier, overlap, source). For long totals without enough
    independent own evidence: pooled standardised errors x the series' own quarter spread."""
    window = MONTH_WINDOW if kind == "month" else 0
    ov = key if kind == "total" else 1
    own_r, own_o = store.pick(store.node.get(nid), key, at, window)
    pool_r, _ = store.pick(store.pool.get(pool_key), key, at, window)
    if kind == "total" and key > REF_KEY and not _own_enough(len(own_r), min_obs, ov):
        s = store.ref_scale(nid, at)
        z = store.scaled_pool(pool_key, key, at)
        if np.isfinite(s) and s > 0 and len(z) >= 3:
            mult, how = s, "own quarter volatility x pooled shape"
            if LS_OWN_BLEND and len(own_r) >= 6 and group_of(pool_key[1]) in LS_BLEND_GROUPS:
                # blend the series' OWN k-month spread with the location-scale spread, on the log scale,
                # weighted by independent evidence (n/k windows vs LS_PRIOR_N). Series whose long totals are
                # steadier than their quarters (e.g. last-year-anchored forecasts) are no longer overstated.
                s_z = 1.4826 * float(np.median(np.abs(z - np.median(z))))
                s_own = 1.4826 * float(np.median(np.abs(own_r - np.median(own_r))))
                if s_z > 0 and s_own > 0:
                    n_eff = len(own_r) / ov
                    w = n_eff / (n_eff + LS_PRIOR_N)
                    s_blend = float(np.exp(w * np.log(s_own) + (1 - w) * np.log(s_z * s)))
                    mult, how = s_blend / s_z, "own long-window spread blended with pooled shape"
            return np.empty(0), np.empty(0), z, mult, ov, how
    own = _own_enough(len(own_r), min_obs, ov)
    return own_r, own_o, pool_r, 1.0, ov, ("own errors" if own else "comparable series")


def _scaled(q: Dict[str, float], mult: float) -> Dict[str, float]:
    return {k: v * mult for k, v in q.items()} if mult != 1.0 else q


def _interval_score(lo, hi, a, L):
    al = 1 - L / 100
    return (hi - lo) + 2 / al * np.maximum(lo - a, 0) + 2 / al * np.maximum(a - hi, 0)


def _window_name(kind: str, key: int) -> str:
    return f"Months {_band(key)} ahead" if kind == "month" else f"Next {key} months"


def sequential_test(err: pd.DataFrame, kind: str, nodes: pd.DataFrame, levels: Sequence[int], min_obs: int,
                    n_test: int) -> pd.DataFrame:
    """Replay both methods at each of the last n_test origins. One row per (test error, method, level)."""
    meta = nodes.set_index("node_id")[["Metric", "Level"]]
    store = _Store(err, meta)
    origins = np.sort(err["o"].unique())[-n_test:]
    test = err[err["o"].isin(origins)].merge(meta, left_on="node_id", right_index=True, how="inner")
    recs = []
    for x in test.itertuples(index=False):
        own_r, own_o, pool_r, mult, ov, _src = _inputs(store, kind, x.node_id, (x.Metric, x.Level), x.key, x.o, min_obs)
        for meth in METHODS:
            q = _scaled(_fit(meth, own_r, own_o, pool_r, x.o, levels, min_obs, ov), mult)
            if not q:
                continue
            for L in levels:
                lo, hi = min(q[f"q_lo{L}"], 0.0), max(q[f"q_hi{L}"], 0.0)
                P = max(float(x.Pred), 0.0)
                Lo, Hi = np.expm1(np.log1p(P) + lo), np.expm1(np.log1p(P) + hi)
                recs.append({"Group": group_of(x.Level), "Level": x.Level, "Window": _window_name(kind, x.key),
                             "Method": meth, "o": x.o, "end": x.end, "L": L, "r": x.r, "lo": lo, "hi": hi, "Covered": lo <= x.r <= hi,
                             "Width_Pct": 100 * (Hi - Lo) / max(P, 1.0),
                             "Score": _interval_score(Lo, Hi, float(x.Actual), L) / max(float(x.Actual), 1.0)})
    return pd.DataFrame(recs)


def choose(test: pd.DataFrame, levels: Sequence[int]) -> Tuple[Dict[Tuple[str, str], str], pd.DataFrame,
                                                                  Dict[Tuple[str, str, int], float], pd.DataFrame]:
    """Method per (Group, Window) by mean interval score; then a symmetric width factor per level."""
    if test.empty:
        return {}, pd.DataFrame(), {}, pd.DataFrame()
    summ = (test.groupby(["Group", "Window", "Method", "L"])
            .agg(Coverage_Pct=("Covered", lambda v: 100 * v.mean()), Median_Width_Pct=("Width_Pct", "median"),
                 Interval_Score=("Score", "mean"), N=("Covered", "size")).reset_index())
    overall = summ.groupby(["Group", "Window", "Method"])["Interval_Score"].mean()
    chosen = {}
    for (grp, win), s in overall.groupby(level=["Group", "Window"]):
        chosen[(grp, win)] = s.droplevel(["Group", "Window"]).idxmin()
    summ["Chosen"] = [chosen.get((g, w)) == m for g, w, m in zip(summ["Group"], summ["Window"], summ["Method"])]
    summ["Interval"] = summ["L"].astype(str) + "%"

    scale, rows = {}, []
    for (grp, win), meth in chosen.items():
        for L in levels:
            g = test[(test["Group"] == grp) & (test["Window"] == win) & (test["Method"] == meth) & (test["L"] == L)]
            if g.empty:
                continue
            need = _need(g)
            before = 100 * float(np.mean(need <= 1.0))
            f = _width_factor(need, L, _overlap(win))
            scale[(grp, win, L)] = f
            rows.append({"Group": grp, "Window": win, "Method": meth, "Interval": f"{L}%", "Width_Factor": round(f, 3),
                         "Coverage_Before_Pct": before, "Coverage_After_Pct": 100 * float(np.mean(need <= f)),
                         "Target_Pct": L, "N": len(need)})
    return chosen, summ, scale, pd.DataFrame(rows)


def _need(g: pd.DataFrame) -> np.ndarray:
    """Width multiple each error would have needed to fall inside its range."""
    r, lo, hi = g["r"].to_numpy(), g["lo"].to_numpy(), g["hi"].to_numpy()
    need = np.where(r >= 0, r / np.where(hi > 0, hi, np.inf), r / np.where(lo < 0, lo, -np.inf))
    return np.where(np.isfinite(need), need, np.inf)


def _overlap(window: str) -> int:
    """'Next 12 months' -> 12 (consecutive windows overlap); monthly windows -> 1."""
    return int(window.split()[1]) if window.startswith("Next ") else 1


def _width_factor(need: np.ndarray, L: int, overlap: int = 1) -> float:
    """Symmetric width correction, only with enough INDEPENDENT test results (overlapping k-month
    windows count as n/k)."""
    before = 100 * float(np.mean(need <= 1.0)) if len(need) else L
    if len(need) / overlap >= SCALE_MIN_N and abs(before - L) > SCALE_TRIGGER_PP:
        return float(np.clip(np.quantile(need, L / 100), *SCALE_BOUNDS))
    return 1.0


NESTED_MIN = 20     # earlier realised test results needed before the nested replay decides anything


def nested_coverage(test: pd.DataFrame, levels: Sequence[int]) -> pd.DataFrame:
    """
    Fully out-of-sample check of the WHOLE range procedure. At each test origin, the method and the width
    factor are decided from earlier test results that had been realised by then (end <= origin), exactly
    as they would have been in production; coverage is then measured at that origin. Unlike
    Coverage_After_Pct (fitted on the same results it reports), nothing here is in-sample.
    """
    if test.empty:
        return pd.DataFrame()
    out = []
    for (grp, win), g in test.groupby(["Group", "Window"]):
        for o in np.sort(g["o"].unique()):
            prior = g[g["end"] <= o]
            if prior["o"].nunique() < 2 or len(prior) < NESTED_MIN * len(METHODS) * len(levels):
                continue
            sc = prior.groupby("Method")["Score"].mean()
            meth = sc.idxmin() if len(sc) == len(METHODS) else "empirical"
            now = g[(g["o"] == o) & (g["Method"] == meth)]
            for L in levels:
                f = _width_factor(_need(prior[(prior["Method"] == meth) & (prior["L"] == L)]), L, _overlap(win))
                x = now[now["L"] == L]
                if len(x):
                    out.append(pd.DataFrame({"Group": grp, "Level": x["Level"].to_numpy(), "Window": win, "L": L,
                                             "Covered": _need(x) <= f}))
    if not out:
        return pd.DataFrame()
    d = pd.concat(out, ignore_index=True)
    res = (d.groupby(["Level", "Window", "L"])["Covered"].agg(Tested_Coverage_Pct=lambda v: 100 * v.mean(), N="size")
           .reset_index())
    res["Interval"] = res["L"].astype(str) + "%"
    return res.rename(columns={"L": "Target_Pct"})[["Level", "Window", "Interval", "Target_Pct", "Tested_Coverage_Pct", "N"]]


def final_quantiles(err: pd.DataFrame, kind: str, nodes: pd.DataFrame, levels: Sequence[int], min_obs: int,
                    chosen: dict, scale: dict, keys: Sequence[int]) -> pd.DataFrame:
    """Ranges for the live forecast: all realised errors, chosen method, width factor. Every node in
    `nodes` gets a row (series without enough own history use the pool of comparable series)."""
    meta = nodes.set_index("node_id")[["Metric", "Level"]]
    store = _Store(err, meta)
    at = int(err["end"].max()) if len(err) else 0
    rows = []
    for nid, (m, lvl) in meta.iterrows():
        grp = group_of(lvl)
        for key in keys:
            win = _window_name(kind, key)
            meth = chosen.get((grp, win)) or chosen.get(("Region", win)) or "empirical"
            for pk in ((m, lvl), (m, "Region")):       # comparable series; regions as fallback
                own_r, own_o, pool_r, mult, ov, how = _inputs(store, kind, nid, pk, key, at, min_obs)
                if how != "comparable series" or len(pool_r) >= 3:
                    break
            q = _scaled(_fit(meth, own_r, own_o, pool_r, at, levels, min_obs, ov), mult)
            if not q:
                continue
            for L in levels:
                f = scale.get((grp, win, L), 1.0)
                q[f"q_lo{L}"], q[f"q_hi{L}"] = f * q[f"q_lo{L}"], f * q[f"q_hi{L}"]
            rows.append({"node_id": nid, ("h" if kind == "month" else "k"): key, "Source": f"{meth} ({how})",
                         "N": int(len(own_r) if how == "own errors" else len(pool_r)), **q})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if kind == "month":
        return _enforce(out, levels, by_h=True)
    # a longer window is never shown as more certain (in %) than a shorter one
    return _enforce(out.rename(columns={"k": "h"}), levels, by_h=True).rename(columns={"h": "k"})


def build(comb_pub: pd.DataFrame, nodes: pd.DataFrame, levels: Sequence[int], min_obs: int, horizon: int,
          n_test: int = 12):
    """Monthly and total ranges plus the evidence tables."""
    out, nested = {}, []
    for kind, keys in (("month", range(1, horizon + 1)), ("total", BLOCKS)):
        err = _errors(comb_pub, kind)
        test = sequential_test(err, kind, nodes, levels, min_obs, n_test)
        chosen, summ, scale, calib = choose(test, levels)
        nested.append(nested_coverage(test, levels))
        out[kind] = (final_quantiles(err, kind, nodes, levels, min_obs, chosen, scale, list(keys)), summ, calib)
    q, summ_m, cal_m = out["month"]
    bq, summ_t, cal_t = out["total"]
    nested_tab = pd.concat(nested, ignore_index=True)
    test_tab = pd.concat([summ_m, summ_t], ignore_index=True)
    if len(test_tab):
        test_tab = test_tab[["Group", "Window", "Method", "Interval", "Coverage_Pct", "Median_Width_Pct",
                             "Interval_Score", "N", "Chosen"]]
    return q, bq, test_tab, cal_m, cal_t, nested_tab
