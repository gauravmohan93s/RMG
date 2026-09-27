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
    """Actuals (last N months) + forecast with 80% band for one node/metric."""
    color = SERIES.get(metric, SERIES["Received"])
    h = hist.sort_values("Date").tail(months_back)
    f = fc.sort_values("Date")
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.plot(h["Date"], h["Actual"], color=INK_2, lw=1.6, label="Actual")
    ax.fill_between(f["Date"], f["Lo80"], f["Hi80"], color=color, alpha=0.18, lw=0, label="80% range")
    ax.plot(f["Date"], f["Forecast"], color=color, lw=2.0, label="Forecast")
    if len(h) and len(f):   # visually join actual to forecast
        ax.plot([h["Date"].iloc[-1], f["Date"].iloc[0]], [h["Actual"].iloc[-1], f["Forecast"].iloc[0]],
                color=color, lw=1.2, ls=":")
        ax.axvline(h["Date"].iloc[-1], color=GRID, lw=1)
    ax.set_title(title, fontsize=9, color=INK, loc="left", pad=4)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncols=3)
    ax.set_ylim(bottom=0)
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
