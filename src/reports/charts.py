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
