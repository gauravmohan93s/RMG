# -*- coding: utf-8 -*-
"""Static charts for the PDF brief (reference data-viz palette, light surface)."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = {"Received": "#2a78d6", "Submitted": "#eb6834", "Benchmark": "#52514e"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=7)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def history_forecast(hist: pd.DataFrame, fc: pd.DataFrame, metric: str, title: str,
                     months_back: int = 36, size=(6.2, 2.6)) -> bytes:
    """Actuals (last N months) + forecast with 50% and 80% bands for one node/metric."""
    color = SERIES.get(metric, SERIES["Received"])
    h = hist.sort_values("Date").tail(months_back)
    f = fc.sort_values("Date")
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.plot(h["Date"], h["Actual"], color=INK_2, lw=1.6, label="Actual")
    ax.fill_between(f["Date"], f["Lo80"], f["Hi80"], color=color, alpha=0.13, lw=0, label="Planning range (8 in 10)")
    if "Lo50" in f and f["Lo50"].notna().any():
        ax.fill_between(f["Date"], f["Lo50"], f["Hi50"], color=color, alpha=0.28, lw=0, label="Most likely (1 in 2)")
    ax.plot(f["Date"], f["Forecast"], color=color, lw=2.0, label="Forecast")
    if len(h) and len(f):   # visually join actual to forecast
        ax.plot([h["Date"].iloc[-1], f["Date"].iloc[0]], [h["Actual"].iloc[-1], f["Forecast"].iloc[0]],
                color=color, lw=1.2, ls=":")
        ax.axvline(h["Date"].iloc[-1], color=GRID, lw=1)
    ax.set_title(title, fontsize=9, color=INK, loc="left", pad=4)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", ncols=4)
    ax.set_ylim(bottom=0, top=1.18 * max(h["Actual"].max() if len(h) else 0, f["Hi80"].max() if len(f) else 0))
    ax.set_ylabel(f"Applications {metric.lower()} / month", fontsize=7, color=INK_2)
    if len(f):   # label the forecast peak and the first/last forecast month
        for idx in {f["Forecast"].idxmax(), f.index[0], f.index[-1]}:
            ax.annotate(f"{f.loc[idx, 'Forecast']:,.0f}", (f.loc[idx, "Date"], f.loc[idx, "Forecast"]),
                        textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6.5, color=INK)
    if len(h):
        ax.annotate(f"{h['Actual'].iloc[-1]:,.0f}", (h["Date"].iloc[-1], h["Actual"].iloc[-1]),
                    textcoords="offset points", xytext=(0, -11), ha="center", fontsize=6.5, color=INK_2)
    return _png(fig)


CLASS_COLORS = {"High": "#2a78d6", "Medium": "#9ec5f0", "Low": "#dde6f0"}


def seasonal_profile(sidx: pd.DataFrame, metric: str, title: str, high: float = 1.10, low: float = 0.90,
                     size=(6.2, 2.6)) -> bytes:
    """Bars = seasonal index by calendar month, shaded by class, with the ±10% thresholds labelled."""
    s = sidx[sidx["Metric"] == metric].sort_values("Month")
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    colors = [CLASS_COLORS.get(c, GRID) for c in s["Class"]]
    bars = ax.bar(s["Month_Name"], s["Seasonal_Index"], color=colors, width=0.7, edgecolor=SURFACE, linewidth=2)
    for b, v, c in zip(bars, s["Seasonal_Index"], s["Class"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK_2)
    for y, lab in ((high, f"High ≥ {high:.2f}"), (1.0, "Average month = 1.00"), (low, f"Low ≤ {low:.2f}")):
        ax.axhline(y, color=INK_2 if y == 1.0 else GRID, lw=0.8, ls="--" if y != 1.0 else "-")
        ax.text(11.6, y, lab, fontsize=6, color=INK_2, va="center", ha="left")
    ax.set_ylim(max(0, s["Seasonal_Index"].min() - 0.15), s["Seasonal_Index"].max() + 0.12)
    ax.set_ylabel("Volume vs average month", fontsize=7, color=INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[k]) for k in ("High", "Medium", "Low")]
    ax.legend(handles, ["High volume", "Medium", "Low volume"], fontsize=6.5, frameon=False, loc="upper left", ncols=3)
    ax.set_title(title, fontsize=9, color=INK, loc="left", pad=4)
    return _png(fig)


def seasonal_heatmap(syear: pd.DataFrame, metric: str, title: str, size=(6.2, 2.2)) -> bytes:
    """Month x year: volume relative to trend (1.00 = on trend); shows how stable the pattern is."""
    s = syear[syear["Metric"] == metric]
    pv = s.pivot_table(index="Year", columns="Month", values="Ratio_to_Trend").reindex(columns=range(1, 13))
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(pv.to_numpy(), cmap="Blues", aspect="auto", vmin=0.6, vmax=1.5)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=7)
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels(pv.index, fontsize=7)
    for i in range(pv.shape[0]):
        for j in range(12):
            v = pv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 1.2 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("vs trend", fontsize=6)
    ax.set_title(title, fontsize=9, color=INK, loc="left", pad=4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return _png(fig)


def forecast_vs_actual(summary: pd.DataFrame, metric: str, title: str, size=(6.2, 2.6)) -> bytes:
    """Actual vs what was forecast 1 and 3 months earlier (walk-forward), last 12 months."""
    s = summary[summary["Metric"] == metric].sort_values("Target_Month").tail(12)
    color = SERIES.get(metric, SERIES["Received"])
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.plot(s["Target_Month"], s["Actual"], color=INK_2, lw=2, marker="o", ms=3.5, label="Actual")
    if "Forecast_1m_ahead" in s:
        ax.plot(s["Target_Month"], s["Forecast_1m_ahead"], color=color, lw=1.8, marker="o", ms=3,
                label="Forecast made 1 month before")
    if "Forecast_3m_ahead" in s:
        ax.plot(s["Target_Month"], s["Forecast_3m_ahead"], color=color, lw=1.2, ls="--", label="Forecast made 3 months before")
    ax.set_ylim(bottom=0)
    ax.set_ylabel(f"Applications {metric.lower()}", fontsize=7, color=INK_2)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left", ncols=3)
    ax.set_title(title, fontsize=9, color=INK, loc="left", pad=4)
    for t in ax.get_xticklabels():
        t.set_fontsize(6.5)
    return _png(fig)


def accuracy_by_horizon(acc: pd.DataFrame, metric: str, level: str = "Region", size=(4.6, 2.4)) -> bytes:
    """Walk-forward WAPE by months ahead: selected method vs seasonal-naive benchmark (level with most data)."""
    a = acc[(acc["Level"] == level) & (acc["Metric"] == metric)]
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    for method, color, lab in (("Selected method (walk-forward)", SERIES.get(metric), "Selected method"),
                               ("Seasonal naive (benchmark)", SERIES["Benchmark"], "Same month last year")):
        s = a[a["Method"] == method].sort_values("h")
        if len(s):
            ax.plot(s["h"], s["WAPE_Pct"], color=color, lw=2, marker="o", ms=3.5, label=lab)
    ax.set_xlabel("Months ahead", fontsize=7, color=INK_2)
    ax.set_xticks(range(1, 13))
    scope = "all regions" if level == "Region" else level.lower()
    ax.set_title(f"{metric}: average monthly error by months ahead ({scope})", fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax.set_ylim(bottom=0)
    return _png(fig)


# ───────────────────────── monthly forecast pack (v2.2) ─────────────────────────
REL_COLOR = {"High": "#2e9e5b", "Medium": "#e0a100", "Low": "#d6453d", "New": "#9a9a95", "Review": "#b36b00"}


def _k(v) -> str:
    """Compact number: 386,891 -> 387k, 4,810 -> 4.8k, 812 -> 812."""
    if v is None or not np.isfinite(v):
        return "–"
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.1f}M"
    if a >= 1e4:
        return f"{v / 1e3:.0f}k"
    if a >= 1e3:
        return f"{v / 1e3:.1f}k"
    return f"{v:.0f}"


def _kfmt(ax, axis="y"):
    fmt = FuncFormatter(lambda v, _: _k(v) if v else "0")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _month_axis(ax):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    for t in ax.get_xticklabels():
        t.set_fontsize(6.5)


def pack_forecast(hist: pd.DataFrame, fc: pd.DataFrame, metric: str, title: str, size=(6.4, 2.9),
                  months_back: int = 24) -> bytes:
    """Actual (last 24 months) and forecast (next 12) with 'most likely' and 'plan for' bands, direct labels."""
    color = SERIES.get(metric, SERIES["Received"])
    h = hist.sort_values("Date").tail(months_back)
    f = fc.sort_values("Date")
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    _kfmt(ax)
    if len(f):
        ax.fill_between(f["Date"], f["Lo80"], f["Hi80"], color=color, alpha=0.13, lw=0, label="Plan between (8 in 10)")
        if "Lo50" in f:
            ax.fill_between(f["Date"], f["Lo50"], f["Hi50"], color=color, alpha=0.30, lw=0, label="Most likely (1 in 2)")
        ax.plot(f["Date"], f["Forecast"], color=color, lw=2.0, label="Forecast")
    if len(h):
        ax.plot(h["Date"], h["Actual"], color=INK_2, lw=1.6, label="Actual")
        if len(f):   # same months last year, drawn under the forecast for direct comparison
            ly = h[h["Date"] >= f["Date"].min() - pd.DateOffset(years=1)]
            ax.plot(ly["Date"] + pd.DateOffset(years=1), ly["Actual"], color=INK_2, lw=0.9, ls=":",
                    label="Same months last year")
    if len(h) and len(f):
        ax.plot([h["Date"].iloc[-1], f["Date"].iloc[0]], [h["Actual"].iloc[-1], f["Forecast"].iloc[0]],
                color=color, lw=1.1, ls=":")
        ax.axvline(h["Date"].iloc[-1], color=GRID, lw=1)
        top = max(h["Actual"].max(), f["Hi80"].max())
        ax.text(h["Date"].iloc[-1], top * 1.14, " forecast →", fontsize=6.5, color=INK_2, va="top")
        ax.text(h["Date"].iloc[-1], top * 1.14, "actual ← ", fontsize=6.5, color=INK_2, va="top", ha="right")
        ax.annotate(_k(h["Actual"].iloc[-1]), (h["Date"].iloc[-1], h["Actual"].iloc[-1]), textcoords="offset points",
                    xytext=(-6, -11), ha="right", fontsize=6.5, color=INK_2)
        pk = f.loc[f["Forecast"].idxmax()]
        pts = [(f["Date"].iloc[0], f["Forecast"].iloc[0])]
        if pk["Date"] != f["Date"].iloc[0]:
            pts.append((pk["Date"], pk["Forecast"]))
        for d, v in pts:
            ax.annotate(f"{_k(v)} {d:%b}", (d, v), textcoords="offset points", xytext=(0, 7), ha="center",
                        fontsize=6.5, color=color, fontweight="bold")
        ax.set_ylim(0, top * 1.2)
    _month_axis(ax)
    ax.set_title(title, fontsize=9, color=INK, loc="left", pad=16)
    ax.legend(fontsize=6, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncols=5, handlelength=1.4,
              borderaxespad=0.1)
    ax.set_ylabel(f"Applications {metric.lower()} per month", fontsize=6.5, color=INK_2)
    return _png(fig)


def destination_bars(ct: pd.DataFrame, metric: str = "Received", title: str = "", size=(6.4, 4.2)) -> bytes:
    """Next-12-month forecast per destination: bar = forecast (colour = reliability), whisker = plan-for
    range, black tick = same months last year; value and change written at the end of each bar."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    d = ct[ct["Level"] == "Country"].sort_values(f"{metric}_Forecast")
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    y = np.arange(len(d))
    f = d[f"{metric}_Forecast"].to_numpy(dtype=float)
    ax.barh(y, f, color=[REL_COLOR.get(r, REL_COLOR["New"]) for r in d[f"{metric}_Reliability"]], height=0.62,
            alpha=0.9)
    lo, hi = d[f"{metric}_Lo80"].to_numpy(dtype=float), d[f"{metric}_Hi80"].to_numpy(dtype=float)
    ax.errorbar(f, y, xerr=[np.clip(f - lo, 0, None), np.clip(hi - f, 0, None)], fmt="none", ecolor=INK, lw=0.8,
                capsize=2)
    ax.scatter(d[f"{metric}_Last_Year"], y, marker="|", s=110, color=INK, zorder=3)
    xmax = float(np.nanmax(hi)) if len(hi) else 1.0
    for yi, fi, hii, yoy in zip(y, f, hi, d[f"{metric}_YoY_Pct"]):
        lab = f"{_k(fi)}  ({yoy:+.0f}%)" if np.isfinite(yoy) else _k(fi)
        ax.text(np.nanmax([hii, fi]) + 0.02 * xmax, yi, lab, va="center", fontsize=6.8, color=INK)
    ax.set_yticks(y, d["Country"], fontsize=7)
    ax.set_xlim(0, xmax * 1.3)
    _kfmt(ax, "x")
    handles = [Patch(color=REL_COLOR[k], label=f"{k} confidence") for k in ("High", "Medium", "Low")]
    handles += [Line2D([], [], color=INK, lw=0.8, label="Plan between"),
                Line2D([], [], color=INK, marker="|", ls="", ms=9, label="Same months last year")]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="lower right")
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    return _png(fig)


def growth_check(ct: pd.DataFrame, metric: str = "Received", title: str = "", size=(6.4, 4.2),
                 cap: float = 150) -> bytes:
    """Forecast growth vs how the market actually grew in the last 6 months (sense-check of the assumption)."""
    d = ct[ct["Level"].isin(["Country", "Organisation"])].iloc[::-1]
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    y = np.arange(len(d))
    fy = d[f"{metric}_YoY_Pct"].to_numpy(dtype=float)
    col = f"{metric}_Recent_YoY_Pct"
    ry = d[col].to_numpy(dtype=float) if col in d else np.full(len(d), np.nan)
    c = SERIES.get(metric)
    for yi, a, b in zip(y, fy, ry):
        if np.isfinite(a) and np.isfinite(b):
            ax.plot([np.clip(a, -cap, cap), np.clip(b, -cap, cap)], [yi, yi], color=GRID, lw=2.2, zorder=1)
    ax.scatter(np.clip(ry, -cap, cap), y, s=34, facecolors=SURFACE, edgecolors=INK_2, lw=1.3, zorder=2,
               label="Actual: last 6 months vs a year earlier")
    ax.scatter(np.clip(fy, -cap, cap), y, s=34, color=c, zorder=3, label="Forecast: next 12 months vs last year")
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_yticks(y, d["Country"], fontsize=7)
    ax.set_ylim(-0.7, len(d) + 0.4)
    ax.set_xlim(-100, cap + 10)
    # exact values in two fixed columns on the right (dots beyond the axis are shown at the edge)
    tr = ax.get_yaxis_transform()
    ax.text(1.04, len(d) - 0.1, "Forecast", transform=tr, fontsize=6.3, color=c, ha="center", fontweight="bold")
    ax.text(1.16, len(d) - 0.1, "Now", transform=tr, fontsize=6.3, color=INK_2, ha="center", fontweight="bold")
    for yi, a, b in zip(y, fy, ry):
        ax.text(1.04, yi, f"{a:+.0f}%" if np.isfinite(a) else "–", transform=tr, fontsize=6.5, color=c, ha="center",
                va="center")
        ax.text(1.16, yi, f"{b:+.0f}%" if np.isfinite(b) else "–", transform=tr, fontsize=6.5, color=INK_2,
                ha="center", va="center")
    ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.08), ncols=2)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    return _png(fig)


def workload_heatmap(staff: pd.DataFrame, metric: str = "Received", title: str = "", size=(13.0, 4.4)) -> bytes:
    """Next 12 months x destination. Cell = forecast applications; colour = busier (blue) or quieter
    (grey) than that destination's own average month, so small and large markets read the same way."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    s = staff[(staff["Metric"] == metric) & staff["Level"].isin(["Organisation", "Country"])].copy()
    s["Date"] = pd.to_datetime(s["Date"])
    order = s.groupby("Label")["Forecast"].sum().sort_values(ascending=False).index.tolist()
    if "Organisation" in order:
        order.remove("Organisation")
        order = ["Organisation"] + order
    piv = s.pivot_table(index="Label", columns="Date", values="Forecast").reindex(order)
    rel = piv.div(piv.mean(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list("load", ["#dcdfe4", "#ffffff", "#2a78d6"])
    ax.imshow(rel.to_numpy(dtype=float), aspect="auto", cmap=cmap, norm=TwoSlopeNorm(vmin=0.6, vcenter=1.0, vmax=1.4))
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v, r = piv.iat[i, j], rel.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, _k(v), ha="center", va="center", fontsize=6.6,
                        color="white" if r >= 1.28 else INK, fontweight="bold" if r >= 1.10 else "normal")
    ax.set_xticks(range(piv.shape[1]), [f"{d:%b}\n{d:%Y}" for d in piv.columns], fontsize=6.8)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=7)
    ax.axhline(0.5, color=INK_2, lw=0.8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.tick_top()
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=24)
    return _png(fig)


def region_bars(tot: pd.DataFrame, metric: str, title: str, size=(6.4, 3.2), top: int = 12) -> bytes:
    """Regions of one destination: next-12-month forecast (colour = reliability), plan-for whisker."""
    d = tot[(tot["Level"] == "Region") & (tot["Metric"] == metric) & (tot["Window"] == "Next 12 months")]
    d = d.sort_values("Forecast", ascending=False)
    rest = d.iloc[top:]
    d = d.iloc[:top].iloc[::-1]
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    y = np.arange(len(d))
    f = d["Forecast"].to_numpy(dtype=float)
    ax.barh(y, f, color=[REL_COLOR.get(r, REL_COLOR["New"]) for r in d["Reliability"]], height=0.6, alpha=0.9)
    lo, hi = d["Lo80"].to_numpy(dtype=float), d["Hi80"].to_numpy(dtype=float)
    ax.errorbar(f, y, xerr=[np.clip(f - lo, 0, None), np.clip(hi - f, 0, None)], fmt="none", ecolor=INK, lw=0.7,
                capsize=2)
    xmax = float(np.nanmax(hi)) if len(hi) else 1.0
    for yi, fi, hii, yoy in zip(y, f, hi, d["YoY_Pct"]):
        ax.text(np.nanmax([hii, fi]) + 0.02 * xmax, yi, f"{_k(fi)} ({yoy:+.0f}%)" if np.isfinite(yoy) else _k(fi),
                va="center", fontsize=6.3)
    ax.set_yticks(y, d["Region"], fontsize=6.6)
    ax.set_xlim(0, xmax * 1.38)
    _kfmt(ax, "x")
    t = title + (f"  (top {top}; {len(rest)} smaller regions total {_k(rest['Forecast'].sum())})" if len(rest) else "")
    ax.set_title(t, fontsize=8.5, color=INK, loc="left")
    return _png(fig)


def track_record(fva: pd.DataFrame, level: str, country: str, metric: str, title: str, size=(6.4, 2.6)) -> bytes:
    """Actual vs what we forecast 1 and 3 months before (walk-forward, last 12 months)."""
    f = fva[(fva["Level"] == level) & (fva["Country"] == country) & (fva["Metric"] == metric)].copy()
    if level == "Country" and f["Target_Month"].nunique() < 6:
        # single-region destination: it is tested as its one region (the country rows are copies, live only)
        r = fva[(fva["Level"] == "Region") & (fva["Country"] == country) & (fva["Metric"] == metric)]
        if r["Region"].nunique() == 1:
            f = r.copy()
    f["Target_Month"] = pd.to_datetime(f["Target_Month"])
    color = SERIES.get(metric)
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    _kfmt(ax)
    act = f.groupby("Target_Month")["Actual"].first().sort_index().tail(12)
    ax.plot(act.index, act.values, color=INK_2, lw=2, marker="o", ms=3.2, label="Actual")
    for ma, ls, lab in ((1, "-", "Forecast made 1 month before"), (3, "--", "Forecast made 3 months before")):
        s = f[f["Months_Ahead"] == ma].groupby("Target_Month")["Forecast"].first().reindex(act.index)
        if s.notna().any():
            ax.plot(s.index, s.values, color=color, lw=1.6 if ma == 1 else 1.1, ls=ls, label=lab)
    ax.set_ylim(bottom=0, top=max(float(act.max()) if len(act) else 1.0, 1.0) * 1.3)
    _month_axis(ax)
    ax.legend(fontsize=6, frameon=False, loc="upper left", ncols=3)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    return _png(fig)


STAGE_COLOR = "#1f3864"


def funnel(steps, title: str, size=(6.4, 2.8)) -> bytes:
    """Pipeline for the next quarter: one bar per stage (forecast), a tick for the same months last year, and
    how many of every 100 submitted applications reach the stage. steps = [(label, forecast, last_year)],
    first step = submitted."""
    from matplotlib.lines import Line2D
    steps = [s for s in steps if s[1] is not None and np.isfinite(s[1])]
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    if not steps:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No stage data yet", ha="center", va="center", color=INK_2, fontsize=8)
        return _png(fig)
    y = np.arange(len(steps))[::-1]
    f = np.array([s[1] for s in steps], dtype=float)
    ly = np.array([s[2] if s[2] is not None else np.nan for s in steps], dtype=float)
    cols = [SERIES["Submitted"]] + [STAGE_COLOR] * (len(steps) - 1)
    ax.barh(y, f, color=cols, height=0.62, alpha=0.88)
    ax.scatter(ly, y, marker="|", s=120, color=INK, zorder=3)
    base = f[0] if f[0] > 0 else np.nan
    xmax = float(np.nanmax(np.r_[f, ly]))
    for i, (yi, (lab, fi, li)) in enumerate(zip(y, steps)):
        chg = f"  ({100 * (fi / li - 1):+.0f}%)" if li and np.isfinite(li) and li >= 150 else ""
        per = f"   ·  {100 * fi / base:.0f} per 100 submitted" if i and np.isfinite(base) else ""
        ax.text(max(fi, li if np.isfinite(li) else 0) + 0.02 * xmax, yi, f"{_k(fi)}{chg}{per}", va="center",
                fontsize=6.6, color=INK)
    ax.set_yticks(y, [s[0] for s in steps], fontsize=7)
    ax.set_xlim(0, xmax * 1.75)
    _kfmt(ax, "x")
    ax.legend(handles=[Line2D([], [], color=INK, marker="|", ls="", ms=9, label="Same months last year")],
              fontsize=6, frameon=False, loc="lower right")
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    return _png(fig)


def intake_curve(curves: pd.DataFrame, pace: dict, title: str, metric: str = "Received", size=(6.4, 2.8)) -> bytes:
    """Booking curve of one intake: applications for it counted so far, by months before it starts, against the
    same intake one and two years earlier; the gap at today's point is the pace."""
    color = SERIES.get(metric, STAGE_COLOR)
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    _kfmt(ax)
    styles = {"Two years ago": dict(color="#b9b8b3", lw=1.0, ls=":"), "Last year": dict(color=INK_2, lw=1.3, ls="--"),
              "This intake": dict(color=color, lw=2.4, ls="-")}
    for name, st in styles.items():
        c = curves[curves["Series"] == name].sort_values("Months_To_Start", ascending=False)
        if len(c):
            ax.plot(c["Months_To_Start"], c["Cumulative"], label=name, **st)
    ax.invert_xaxis()
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.axvline(0, color=GRID, lw=1)
    top = float(curves["Cumulative"].max()) if len(curves) else 1.0
    ax.text(0, top * 0.02, " intake starts", fontsize=6.3, color=INK_2, va="bottom")
    now = pace.get("Months_To_Start")
    if now is not None and np.isfinite(pace.get("To_Date", np.nan)):
        ax.scatter([now], [pace["To_Date"]], s=28, color=color, zorder=4)
        p = pace.get("Pace_Pct", np.nan)
        lab = f"today: {_k(pace['To_Date'])}" + (f"  ({p:+.0f}% vs last year)" if np.isfinite(p) else "")
        ax.annotate(lab, (now, pace["To_Date"]), textcoords="offset points", xytext=(6, 6), fontsize=6.6,
                    color=color, fontweight="bold")
    pf = pace.get("Projected_Final", np.nan)
    if np.isfinite(pf):
        x_end = -2
        ax.errorbar([x_end], [pf], yerr=[[max(pf - pace["Lo80"], 0)], [max(pace["Hi80"] - pf, 0)]], fmt="o",
                    color=color, ms=4, capsize=3, lw=1)
        ax.annotate(f"likely final {_k(pf)}", (x_end, pf), textcoords="offset points", xytext=(-4, 8), ha="right",
                    fontsize=6.4, color=color)
        top = max(top, float(pace["Hi80"]))
    ax.set_ylim(0, top * 1.15)
    ax.set_xlabel("months before the intake starts", fontsize=6.5, color=INK_2)
    ax.legend(fontsize=6, frameon=False, loc="upper left", ncols=3)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    return _png(fig)


def region_multiples(hist: pd.DataFrame, fc: pd.DataFrame, regions, title: str, metric: str = "Received",
                     ncols: int = 4, size=(13.0, 3.4), months_back: int = 18) -> bytes:
    """Small multiples: one panel per region, actual (last 18 months) and forecast (next 12) with the plan-for
    band, the same scale style in every panel and the 12-month change written in the corner."""
    color = SERIES.get(metric, SERIES["Received"])
    n = len(regions)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0], size[1] * nrows / 2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.set_axis_off()
            continue
        reg = regions[k]
        h = hist[hist["Region"] == reg].sort_values("Date").tail(months_back)
        f = fc[fc["Region"] == reg].sort_values("Date")
        _style(ax)
        _kfmt(ax)
        ax.tick_params(labelsize=5.5)
        if len(f):
            ax.fill_between(f["Date"], f["Lo80"], f["Hi80"], color=color, alpha=0.15, lw=0)
            ax.plot(f["Date"], f["Forecast"], color=color, lw=1.5)
        if len(h):
            ax.plot(h["Date"], h["Actual"], color=INK_2, lw=1.1)
        ly = hist[(hist["Region"] == reg) & (hist["Date"] >= f["Date"].min() - pd.DateOffset(years=1))
                  & (hist["Date"] <= f["Date"].max() - pd.DateOffset(years=1))]["Actual"].sum() if len(f) else np.nan
        chg = 100 * (f["Forecast"].sum() / ly - 1) if len(f) and ly else np.nan
        ax.set_title(f"{reg}", fontsize=7, color=INK, loc="left", pad=2)
        if np.isfinite(chg):
            ax.text(0.98, 0.92, f"{_k(f['Forecast'].sum())}  {chg:+.0f}%", transform=ax.transAxes, ha="right",
                    va="top", fontsize=6.2, color=color, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.set_xticks([])
    fig.suptitle(title, fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _png(fig)


# ───────────────────────── "At a glance" visuals (operations / HR readers) ─────────────────────────
UP, DOWN, FLAT = "#2a78d6", "#e0662a", "#9a9a95"
CONF_DOTS = {"High": 3, "Medium": 2, "Low": 1}
VERDICT = {"right": ("✓", "#2e9e5b"), "close": ("~", "#e0a100"), "missed": ("✗", "#d6453d")}


def _pace_color(v, band: float = 5.0):
    return FLAT if v is None or not np.isfinite(v) or abs(v) < band else (UP if v > 0 else DOWN)


def _arrow(v) -> str:
    return "" if v is None or not np.isfinite(v) else ("▲" if v > 0 else "▼")


def kpi_cards(g: dict, size=(13.2, 2.25)) -> bytes:
    """Four big cards: next quarter (or its direction where forecasts are too uncertain for a number), busiest
    month (12 months as small bars; staff needed once headcount is known), the ongoing intake's pace, and 'how
    sure are we' with the last six months as right / close / missed."""
    from matplotlib.patches import Circle, FancyBboxPatch
    fig = plt.figure(figsize=size)
    fig.patch.set_facecolor("white")
    W = 0.235
    for i in range(4):
        ax = fig.add_axes([0.005 + i * (W + 0.014), 0.04, W, 0.92])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=SURFACE, ec=GRID, lw=1))
        if i == 0 and g.get("next_quarter"):
            q = g["next_quarter"]
            c = _pace_color(q["yoy"])
            ax.text(0.07, 0.83, f"Applications, next 3 months ({q['from']:%b}–{q['to']:%b})", fontsize=8.5, color=INK_2)
            if g.get("direction_only"):
                word = "Rising" if q["yoy"] > 5 else "Falling" if q["yoy"] < -5 else "Flat"
                ax.text(0.07, 0.42, f"{_arrow(q['yoy'])} {word}", fontsize=25, fontweight="bold", color=c)
                ax.text(0.07, 0.17, f"about {round(q['value'], -1):,.0f} ({q['yoy']:+.0f}% vs last year)", fontsize=9.5,
                        color=INK)
                ax.text(0.07, 0.05, f"direction only: forecasts here miss by about ±{q['miss']:.0f}%", fontsize=7.3,
                        color=INK_2)
            else:
                ax.text(0.07, 0.42, f"{round(q['value'], -2 if q['value'] >= 2000 else -1):,.0f}", fontsize=27,
                        fontweight="bold", color=INK)
                ax.text(0.07, 0.17, f"{_arrow(q['yoy'])} {q['yoy']:+.0f}% vs last year", fontsize=11, color=c,
                        fontweight="bold")
                ax.text(0.07, 0.05, f"plan between {_k(q['lo'])} and {_k(q['hi'])}", fontsize=7.5, color=INK_2)
        elif i == 1 and g.get("peak"):
            p, m = g["peak"], g["months"]
            ax.text(0.07, 0.83, "Busiest month ahead", fontsize=8.5, color=INK_2)
            ax.text(0.07, 0.55, f"{p['date']:%b %Y}", fontsize=22, fontweight="bold", color=INK)
            v_ = round(p["value"], -2 if p["value"] >= 2000 else -1)
            sub = f"about {v_:,.0f} applications"
            if g.get("staff"):
                sub += f" · ~{g['staff']['need']:.0f} staff (today ~{g['staff']['today']:.0f})"
            ax.text(0.07, 0.38, sub, fontsize=8.6, color=INK_2)
            if len(m):
                v = m["Forecast"].to_numpy(dtype=float)
                xs = 0.07 + np.arange(len(v)) * (0.86 / len(v))
                h = 0.2 * v / v.max()
                for x, hh, d in zip(xs, h, m["Date"]):
                    ax.add_patch(FancyBboxPatch((x, 0.07), 0.86 / len(v) * 0.7, hh, boxstyle="square,pad=0",
                                                fc=UP if d == p["date"] else "#c9d6e8", ec="none"))
        elif i == 2:
            v = g.get("intake_pace", np.nan)
            ax.text(0.07, 0.83, f"{g.get('season') or 'Coming intake'}: applications so far", fontsize=8.5, color=INK_2)
            ax.text(0.07, 0.42, f"{_arrow(v)} {v:+.0f}%" if np.isfinite(v) else "–", fontsize=27, fontweight="bold",
                    color=_pace_color(v))
            ax.text(0.07, 0.17, "vs last year at this point", fontsize=10, color=INK_2)
        elif i == 3 and g.get("next_quarter"):
            q = g["next_quarter"]
            n = CONF_DOTS.get(q["confidence"], 0)
            col = REL_COLOR.get(q["confidence"], REL_COLOR["New"])
            ax.text(0.07, 0.83, "How sure are we?", fontsize=8.5, color=INK_2)
            for k in range(3):
                ax.add_patch(Circle((0.12 + k * 0.1, 0.64), 0.035, fc=col if k < n else "#e4e3df", ec="none"))
            ax.text(0.45, 0.63, q["confidence"], fontsize=15, fontweight="bold", color=col, va="center")
            if np.isfinite(q["miss"]):
                ax.text(0.07, 0.44, f"usually within ±{q['miss']:.0f}%", fontsize=10, color=INK, fontweight="bold")
            tr = g.get("track") or []
            if tr:
                ax.text(0.07, 0.3, "Last months, forecast vs actual:", fontsize=7, color=INK_2)
                for k, t in enumerate(tr):
                    sym, cc = VERDICT[t["verdict"]]
                    x = 0.1 + k * 0.145
                    ax.text(x, 0.15, sym, fontsize=13, fontweight="bold", color=cc, ha="center", va="center")
                    ax.text(x, 0.05, f"{t['month']:%b}", fontsize=6.3, color=INK_2, ha="center")
    return _png(fig)


LEAK_LABEL = {"Received": "Applications received", "Submitted": "Submitted", "Offer": "Offers", "Deposit": "Deposits",
              "Visa_Document": "Visa documents", "Visa_Applied": "Visa applied", "Visa_Granted": "Visas granted"}


def stage_leak(leak, season: str, size=(6.4, 3.3)) -> bytes:
    """One bar per stage of the ongoing intake: change vs last year at the same point. Falling bars from left to
    right = growth leaking out of the pipeline."""
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor("white")
    _style(ax)
    ax.set_facecolor("white")
    if not leak:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Stage data by intake not available yet", ha="center", va="center", color=INK_2)
        return _png(fig)
    labs = [LEAK_LABEL.get(m, m) for m, _ in leak]
    vals = np.array([v for _, v in leak], dtype=float)
    shown = np.clip(vals, -100, 150)
    y = np.arange(len(vals))[::-1]
    ax.barh(y, shown, color=[_pace_color(v) for v in vals], height=0.62)
    ax.axvline(0, color=INK_2, lw=0.8)
    for yi, v, s in zip(y, vals, shown):
        ax.text(s + (3 if s >= 0 else -3), yi, f"{v:+.0f}%", va="center", ha="left" if s >= 0 else "right",
                fontsize=8.5, fontweight="bold", color=_pace_color(v))
    ax.set_yticks(y, labs, fontsize=8)
    lim = max(20, np.nanmax(np.abs(shown)) * 1.35)
    ax.set_xlim(-lim, lim)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_title(f"{season}: each stage vs last year at the same point", fontsize=9.5, color=INK, loc="left")
    return _png(fig)


def traffic_grid(grid: pd.DataFrame, season: str, title: str = "Destinations at a glance", size=(6.4, 3.3)) -> bytes:
    """Rows (destinations, or a destination's regions) x four signals as coloured tiles: blue = above last year,
    orange = below, grey = about the same; how sure we are as green / amber / red."""
    from matplotlib.patches import FancyBboxPatch
    cols = [("Next quarter", "Next 3 months\nvs last year"), ("Intake", f"{season}\napplications"),
            ("Deposits", f"{season}\ndeposits"), ("Confidence", "How sure\nare we")]
    n = len(grid)
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor("white")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, max(n, 6) + 1.1)
    x0, w = 0.36, 0.155
    top = max(n, 6)
    for j, (_, lab) in enumerate(cols):
        ax.text(x0 + j * w + w / 2, top + 0.35, lab, ha="center", va="center", fontsize=6.8, color=INK_2)
    for i, r in enumerate(grid.to_dict("records")):
        yy = top - 1 - i
        name = r["Destination"] if len(r["Destination"]) <= 26 else r["Destination"][:25] + "…"
        ax.text(0.0, yy, name, va="center", fontsize=7.6, color=INK)
        for j, (key, _) in enumerate(cols):
            v = r[key]
            if key == "Confidence":
                fc, txt = REL_COLOR.get(v, REL_COLOR["New"]), f"±{r['Miss']:.0f}%" if np.isfinite(r["Miss"]) else v
            else:
                fc = _pace_color(v) if np.isfinite(v) else "#f1f1ef"
                txt = (f"{_arrow(v)}{min(v, 999):+.0f}%" if np.isfinite(v) else "–")
            ax.add_patch(FancyBboxPatch((x0 + j * w + 0.008, yy - 0.36), w - 0.016, 0.72,
                                        boxstyle="round,pad=0,rounding_size=0.02", fc=fc, ec="none"))
            ax.text(x0 + j * w + w / 2, yy, txt, ha="center", va="center", fontsize=7.4, fontweight="bold",
                    color="white" if fc not in ("#f1f1ef", FLAT) else INK)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    return _png(fig)


def revision_bars(rev: dict, size=(6.4, 2.6)) -> bytes:
    """What changed since last month's forecast, by destination (same months in both forecasts)."""
    d = rev["by_destination"]
    d = d[d["Change"].abs() >= 1]
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor("white")
    _style(ax)
    ax.set_facecolor("white")
    y = np.arange(len(d))
    v = d["Change"].to_numpy(dtype=float)
    ax.barh(y, v, color=[UP if x > 0 else DOWN for x in v], height=0.62)
    ax.axvline(0, color=INK_2, lw=0.8)
    lim = float(np.nanmax(np.abs(v))) * 1.35 if len(v) else 1.0
    for yi, x in zip(y, v):
        ax.text(x + (0.02 * lim if x >= 0 else -0.02 * lim), yi, f"{'+' if x > 0 else '-'}{_k(abs(x))}", va="center",
                ha="left" if x >= 0 else "right", fontsize=7, color=UP if x > 0 else DOWN, fontweight="bold")
    ax.set_yticks(y, d["Country"], fontsize=7)
    ax.set_xlim(-lim, lim)
    _kfmt(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    sign = "+" if rev["change"] >= 0 else "-"
    ax.set_title(f"What changed since last month's forecast: {sign}{_k(abs(rev['change']))} ({rev['pct']:+.0f}%) "
                 f"over the same {rev['months']} months", fontsize=8.6, color=INK, loc="left")
    return _png(fig)
