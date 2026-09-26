# -*- coding: utf-8 -*-
# src/reports/final_pdf_reports.py
from __future__ import annotations

import json, logging, os, io, math, time, textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless, faster
import matplotlib.pyplot as plt

# ReportLab (Platypus for proper pagination/layout)
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image, PageBreak, KeepTogether
)

from ..config import OUT_DIR, HORIZON
from ..loader import load_data

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

REPORT_DIR = OUT_DIR / "Reports" / "PDF"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_CSV  = OUT_DIR / "Final_Forecast" / "final_forecast.csv"
FORECAST_XLSX = OUT_DIR / "Final_Forecast" / "final_forecast.xlsx"
SERIES_SEL    = OUT_DIR / "Evaluation" / "series_selection.csv"
SERIES_SCORES = OUT_DIR / "Evaluation" / "series_scores_received.csv"  # optional for reasoning

ENABLE_LLM    = os.environ.get("RMG_ENABLE_LLM", "0").lower() in {"1", "true", "yes"}
OLLAMA_URL    = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL  = os.environ.get("OLLAMA_MODEL", "phi3:mini")   # or qwen2:0.5b
OLLAMA_TIMEOUT_S = float(os.environ.get("OLLAMA_TIMEOUT_S", "4.0"))  # keep short for speed
LLM_CACHE        = REPORT_DIR / "llm_cache.json"

# ----------------------------- utils / IO -----------------------------
def _safe(s: str) -> str:
    return "".join(("_" if ch in r'\/:*?"<>|' else ch) for ch in str(s))

def _load_final_forecast() -> pd.DataFrame:
    if FORECAST_CSV.exists():
        df = pd.read_csv(FORECAST_CSV, parse_dates=["Date"])
    elif FORECAST_XLSX.exists():
        # prefer sheet Forecast_12m if present
        try:
            df = pd.read_excel(FORECAST_XLSX, sheet_name="Forecast_12m", parse_dates=["Date"])
        except Exception:
            df = pd.read_excel(FORECAST_XLSX, sheet_name=0, parse_dates=["Date"])
    else:
        raise FileNotFoundError("Final forecast not found (csv/xlsx).")

    need = {"Country","Region","Metric","Date","Forecast"}
    if not need.issubset(df.columns):
        raise ValueError(f"Final forecast missing columns: {need - set(df.columns)}")

    # Optional bands + chosen model columns
    for c in ["Forecast_Lo","Forecast_Hi","Chosen_Model"]:
        if c not in df.columns:
            df[c] = np.nan
    return df

def _history_long() -> pd.DataFrame:
    wide, _ = load_data()
    long = (
        wide.melt(id_vars=["Country","Region","Date"],
                  value_vars=[c for c in ["Received","Submitted"] if c in wide.columns],
                  var_name="Metric", value_name="Value")
            .sort_values(["Country","Region","Metric","Date"])
            .reset_index(drop=True)
    )
    return long

def _load_series_selection() -> pd.DataFrame:
    if not SERIES_SEL.exists():
        # still allow report creation but reasoning will be thin
        return pd.DataFrame(columns=["Country","Region","Best_Model_Received"])
    df = pd.read_csv(SERIES_SEL)
    need = {"Country","Region","Best_Model_Received"}
    if not need.issubset(df.columns):
        return pd.DataFrame(columns=list(need))
    return df

def _load_scores() -> pd.DataFrame:
    if SERIES_SCORES.exists():
        try:
            s = pd.read_csv(SERIES_SCORES)
            return s
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def _month_label(d: pd.Timestamp) -> str:
    return pd.Timestamp(d).strftime("%b-%Y")

# ----------------------------- plotting ------------------------------
def _series_plot_png(hist: pd.DataFrame, fc: pd.DataFrame,
                     country: str, region: str, metric: str,
                     width_cm: float = 18.0, height_cm: float = 6.0, dpi: int = 120) -> bytes:
    h = hist[(hist.Country==country) & (hist.Region==region) & (hist.Metric==metric)].copy()
    f = fc[(fc.Country==country) & (fc.Region==region) & (fc.Metric==metric)].copy()
    fig_w, fig_h = width_cm/2.54, height_cm/2.54

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    if not h.empty:
        ax.plot(h["Date"], h["Value"], label="Actual", lw=1.2)
        last_hist = h["Date"].max()
        ax.axvline(last_hist, color="black", ls=":", lw=0.8)
    if not f.empty:
        label = "Forecast"
        if "Chosen_Model" in f.columns and f["Chosen_Model"].notna().any():
            label = f"Forecast ({f['Chosen_Model'].dropna().iloc[0]})"
        ax.plot(f["Date"], f["Forecast"], ":", label=label, lw=1.4)
        if {"Forecast_Lo","Forecast_Hi"}.issubset(f.columns):
            if f["Forecast_Lo"].notna().any() and f["Forecast_Hi"].notna().any():
                ax.fill_between(f["Date"], f["Forecast_Lo"], f["Forecast_Hi"],
                                alpha=0.18, label="Uncertainty band")

    ax.set_title(f"{country} — {region} — {metric}")
    ax.set_xlabel("Date"); ax.set_ylabel(metric)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    for t in ax.get_xticklabels():
        t.set_rotation(45)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ----------------------------- LLM (Ollama) --------------------------
def _load_llm_cache() -> Dict[str, str]:
    if LLM_CACHE.exists():
        try:
            return json.loads(LLM_CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_llm_cache(cache: Dict[str, str]) -> None:
    try:
        LLM_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def _ollama_generate(prompt: str, model: str, timeout_s: float) -> Optional[str]:
    """
    Lightweight Ollama call with short timeout; returns None on any failure.
    """
    import urllib.request, json as _json, socket
    data = _json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        url=f"{OLLAMA_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
            obj = _json.loads(raw.decode("utf-8", errors="ignore"))
            txt = obj.get("response", "")
            return (txt or "").strip()[:1200]
    except Exception:
        return None

def _mk_prompt(country: str, region: str, metric: str,
               chosen_model: str,
               holdout_mae: Optional[float],
               holdout_smape: Optional[float],
               hist_tail: pd.DataFrame,
               fc_series: pd.DataFrame) -> str:
    """
    Keep it short/structured so tiny models can handle it quickly.
    """
    trend = []
    if not hist_tail.empty:
        v = hist_tail["Value"].to_list()
        if len(v) >= 2:
            delta = v[-1] - v[0]
            pct = (v[-1]/v[0]-1.0)*100.0 if v[0] else 0.0
            trend.append(f"Last 12m change: {delta:+.0f} ({pct:+.1f}%).")
    if not fc_series.empty:
        first, last = float(fc_series.iloc[0]), float(fc_series.iloc[-1])
        pct = (last/first-1.0)*100.0 if first>0 else 0.0
        trend.append(f"Next {len(fc_series)}m outlook vs first forecast month: {pct:+.1f}%.")

    parts = [
        f"You are writing a short, neutral analyst note for operations users.",
        f"Country: {country}. Region: {region}. Metric: {metric}.",
        f"Chosen model: {chosen_model}.",
        f"Holdout (3m) errors: MAE={'' if holdout_mae is None else f'{holdout_mae:.1f}'}, sMAPE={'' if holdout_smape is None else f'{holdout_smape:.1f}%'} (lower is better).",
        "Context:",
        " • History shows seasonality month-on-month; values are non-negative and can be spiky.",
        " • Forecast uses uncertainty bands based on month-wise historical limits.",
        "Write 4–6 concise sentences:",
        "1) Summarize recent pattern.",
        "2) Explain why the selected model is sensible.",
        "3) Describe direction of the 12-month outlook and any risk.",
        "4) Keep it concrete, avoid buzzwords. No bullets; 1 short paragraph."
    ]
    if trend:
        parts.append("Quick stats: " + " ".join(trend))
    return "\n".join(parts)

def _llm_comment(country: str, region: str, metric: str,
                 chosen_model: str,
                 hist_tail: pd.DataFrame,
                 fc: pd.DataFrame,
                 scores: pd.DataFrame,
                 cache: Dict[str,str]) -> str:
    key = f"{country}|{region}|{metric}|{chosen_model}"
    if key in cache and cache[key]:
        return cache[key]

    # find holdout errors for context
    mae = smape = None
    if not scores.empty:
        s = scores[(scores.Country==country) &
                   (scores.Region==region) &
                   (scores.Metric=="Received") &
                   (scores.Model.notna())]
        if not s.empty:
            # use row of chosen model if available, else best row
            if chosen_model in set(s["Model"]):
                row = s[s.Model==chosen_model].sort_values("MAE").head(1)
            else:
                row = s.sort_values("MAE").head(1)
            if not row.empty:
                mae = float(row["MAE"].iloc[0]) if "MAE" in row.columns else None
                smape = float(row["sMAPE"].iloc[0]) if "sMAPE" in row.columns else None

    # prep prompt (last 12 months of history for quick gist)
    tail_hist = hist_tail.tail(12)
    fc_vals = fc["Forecast"].dropna()
    txt = None
    if ENABLE_LLM:
        prompt = _mk_prompt(country, region, metric, chosen_model, mae, smape, tail_hist, fc_vals)
        txt = _ollama_generate(prompt, model=OLLAMA_MODEL, timeout_s=OLLAMA_TIMEOUT_S)

    if not txt:
        # High-quality deterministic summary with real calculated numbers
        tot_fc = float(fc_vals.sum()) if not fc_vals.empty else 0.0
        avg_fc = float(fc_vals.mean()) if not fc_vals.empty else 0.0
        if not fc_vals.empty and len(fc.loc[fc_vals.index]) > 0:
            peak_idx = fc_vals.idxmax()
            peak_dt = fc.loc[peak_idx, "Date"]
            peak_month = pd.Timestamp(peak_dt).strftime("%b %Y")
            peak_val = float(fc_vals.max())
        else:
            peak_month, peak_val = "N/A", 0.0

        err_str = f"Holdout MAE: {mae:,.1f}, sMAPE: {smape:.1f}%. " if mae is not None and smape is not None else ""
        txt = (
            f"{metric} in {region}, {country} is modeled using **{chosen_model}** ({err_str.strip()}). "
            f"The 12-month projected total volume is {tot_fc:,.0f} (monthly average: {avg_fc:,.0f}), "
            f"with projected peak volume of {peak_val:,.0f} occurring in {peak_month}. "
            f"Forecast uncertainty bands reflect empirical month-of-year seasonal ranges."
        )

    cache[key] = txt
    _save_llm_cache(cache)
    return txt

# ----------------------------- tables --------------------------------
def _monthly_table_for_region(fc: pd.DataFrame, country: str, region: str) -> pd.DataFrame:
    """
    Build a single table combining Received + Submitted columns for 12 months:
    Columns: Month | Rec | Rec_Lo | Rec_Hi | Sub | Sub_Lo | Sub_Hi | Row_Total
    Rounded to int. Missing metrics are filled with zeros.
    """
    month_idx = sorted(fc[(fc.Country==country)&(fc.Region==region)]["Date"].drop_duplicates())
    rows = []
    def _cols(metric: str) -> Tuple[List[float], List[float], List[float]]:
        s = fc[(fc.Country==country)&(fc.Region==region)&(fc.Metric==metric)].sort_values("Date")
        vals = s["Forecast"].to_list()
        los  = s["Forecast_Lo"].fillna(s["Forecast"]*0.9).to_list()
        his  = s["Forecast_Hi"].fillna(s["Forecast"]*1.1).to_list()
        return vals, los, his
    rec, rec_lo, rec_hi = _cols("Received")
    sub, sub_lo, sub_hi = _cols("Submitted")

    # align lengths to month_idx
    def _pad(a, n): return (a + [0.0]*(n-len(a)))[:n]
    n = len(month_idx)
    rec, rec_lo, rec_hi = _pad(rec,n), _pad(rec_lo,n), _pad(rec_hi,n)
    sub, sub_lo, sub_hi = _pad(sub,n), _pad(sub_lo,n), _pad(sub_hi,n)

    for i, d in enumerate(month_idx):
        r = round(float(rec[i] or 0))
        rl= round(float(rec_lo[i] or 0))
        rh= round(float(rec_hi[i] or 0))
        s = round(float(sub[i] or 0))
        sl= round(float(sub_lo[i] or 0))
        sh= round(float(sub_hi[i] or 0))
        rows.append([_month_label(d), r, rl, rh, s, sl, sh, r+s])

    tbl = pd.DataFrame(rows, columns=[
        "Month","Received","Received_Lo","Received_Hi",
        "Submitted","Submitted_Lo","Submitted_Hi","Row_Total"
    ])
    return tbl

def _totals_row(df: pd.DataFrame) -> List:
    return ["12-mo Total",
            int(df["Received"].sum()), int(df["Received_Lo"].sum()), int(df["Received_Hi"].sum()),
            int(df["Submitted"].sum()), int(df["Submitted_Lo"].sum()), int(df["Submitted_Hi"].sum()),
            int(df["Row_Total"].sum())]

def _platypus_table(df: pd.DataFrame) -> Table:
    # convert to strings to avoid scientific notation; compact widths
    data = [list(df.columns)] + df.astype(str).values.tolist()
    # add totals at the bottom
    data.append(_totals_row(df))
    t = Table(data, colWidths=[2.4*cm, 2.0*cm, 2.0*cm, 2.0*cm, 2.0*cm, 2.0*cm, 2.0*cm, 2.4*cm])
    t.setStyle(TableStyle([
        ("FONT", (0,0), (-1,0), "Helvetica-Bold", 9),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F0F3F6")),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("FONT", (0,1), (-1,-2), "Helvetica", 8),
        ("FONT", (0,-1), (-1,-1), "Helvetica-Bold", 9),
        ("LINEBELOW", (0,0), (-1,0), 0.5, colors.black),
        ("LINEABOVE", (0,-1), (-1,-1), 0.7, colors.black),
        ("ROWBACKGROUNDS", (0,1), (-1,-2), [colors.whitesmoke, colors.HexColor("#FBFBFB")]),
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("LEFTPADDING", (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    return t

# ----------------------------- document builders ---------------------
@dataclass
class ReportStyles:
    h1: ParagraphStyle
    h2: ParagraphStyle
    h3: ParagraphStyle
    body: ParagraphStyle
    small: ParagraphStyle

def _styles() -> ReportStyles:
    styles = getSampleStyleSheet()
    return ReportStyles(
        h1=ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=6, spaceBefore=6),
        h2=ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceAfter=4, spaceBefore=6),
        h3=ParagraphStyle("h3", parent=styles["Heading3"], fontSize=11, spaceAfter=2, spaceBefore=4),
        body=ParagraphStyle("body", parent=styles["BodyText"], fontSize=9, leading=12),
        small=ParagraphStyle("small", parent=styles["BodyText"], fontSize=8, leading=11, textColor=colors.grey),
    )

def _country_summary_plot(hist: pd.DataFrame, fc: pd.DataFrame, country: str,
                          metric: str = "Received") -> bytes:
    h = hist[(hist.Country==country) & (hist.Metric==metric)].groupby("Date", as_index=False)["Value"].sum()
    f = fc[(fc.Country==country) & (fc.Metric==metric)].groupby("Date", as_index=False)["Forecast"].sum()
    idx = pd.Index(sorted(set(h.Date).union(set(f.Date))))
    hs = h.set_index("Date").reindex(idx)
    fs = f.set_index("Date").reindex(idx)

    fig, ax = plt.subplots(figsize=(18/2.54, 6/2.54))
    if hs["Value"].notna().any(): ax.plot(hs.index, hs["Value"], label="Actual", lw=1.4)
    if fs["Forecast"].notna().any(): ax.plot(fs.index, fs["Forecast"], ":", label="Forecast", lw=1.4)
    if not hs.empty: ax.axvline(hs.index.max(), color="black", ls=":", lw=0.8)
    ax.set_title(f"{country} — Organisation view ({metric})")
    ax.set_xlabel("Date"); ax.set_ylabel(metric)
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)
    for t in ax.get_xticklabels(): t.set_rotation(45)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def _country_story(flow: List, styles: ReportStyles,
                   country: str, hist: pd.DataFrame, fc: pd.DataFrame,
                   selection: pd.DataFrame) -> None:
    # Header
    flow.append(Paragraph(f"{country} — Forecast Report (LLM-assisted)", styles.h1))
    flow.append(Paragraph("Evaluation: 3-Test Decathlon Tournament. Horizon: next 12 months. Scenarios: Baseline vs Policy Adjusted.", styles.small))
    flow.append(Spacer(0, 4))

    # Visa & Immigration Policy Context
    try:
        from ..policy_matrix import get_policy_matrix
        pol_df = get_policy_matrix()
        c_pol = pol_df[pol_df["Country"] == country]
        if not c_pol.empty:
            p_row = c_pol.iloc[0]
            mult = float(p_row["Policy_Multiplier"])
            risk = str(p_row["Risk_Level"])
            ctx  = str(p_row["Regulatory_Context"])
            diff_pct = (mult - 1.0) * 100.0
            diff_str = f"+{diff_pct:.0f}%" if diff_pct > 0 else f"{diff_pct:.0f}%"
            pol_banner = [
                [f"Visa & Immigration Policy Status: {risk}", f"Regulatory Factor: {mult:.2f}x ({diff_str})"],
                [f"Key Policy Driver: {ctx}", ""]
            ]
            t_pol = Table(pol_banner, colWidths=[12.5*cm, 5.5*cm])
            bg_col = colors.HexColor("#FDF2E9") if mult < 1.0 else (colors.HexColor("#EAF2F8") if mult > 1.0 else colors.HexColor("#F4F6F7"))
            bd_col = colors.HexColor("#DC7633") if mult < 1.0 else (colors.HexColor("#5499C7") if mult > 1.0 else colors.HexColor("#BDC3C7"))
            t_pol.setStyle(TableStyle([
                ("FONT", (0,0), (-1,-1), "Helvetica", 7.5),
                ("FONT", (0,0), (0,0), "Helvetica-Bold", 8),
                ("BACKGROUND", (0,0), (-1,-1), bg_col),
                ("BOX", (0,0), (-1,-1), 0.5, bd_col),
                ("SPAN", (0,1), (1,1)),
                ("TOPPADDING", (0,0), (-1,-1), 2),
                ("BOTTOMPADDING", (0,0), (-1,-1), 2),
            ]))
            flow.append(t_pol)
            flow.append(Spacer(0, 4))
    except Exception:
        pass

    # Chosen model map for each region (Received-based)
    if not selection.empty:
        sel = selection[selection.Country==country][["Region","Best_Model_Received"]].sort_values("Region")
        if not sel.empty:
            data = [["Region","Selected model (Received)"]] + sel.values.tolist()
            t = Table(data, colWidths=[6*cm, 10*cm])
            t.setStyle(TableStyle([
                ("FONT", (0,0), (-1,0), "Helvetica-Bold", 9),
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F0F3F6")),
                ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
                ("FONT", (0,1), (-1,-1), "Helvetica", 8),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ]))
            flow.append(t)
            flow.append(Spacer(0, 6))

    # Country summary plot
    png = _country_summary_plot(hist, fc, country, metric="Received")
    flow.append(Image(io.BytesIO(png), width=18*cm, height=6*cm))
    flow.append(Spacer(0, 8))

def _region_section(flow: List, styles: ReportStyles,
                    country: str, region: str, hist: pd.DataFrame, fc: pd.DataFrame,
                    selection: pd.DataFrame, scores: pd.DataFrame, cache: Dict[str,str]) -> None:
    # Resolve chosen model for the region
    chosen = None
    if not selection.empty:
        srow = selection[(selection.Country==country) & (selection.Region==region)]
        if not srow.empty:
            chosen = srow["Best_Model_Received"].iloc[0]

    flow.append(Paragraph(f"{country} — {region}", styles.h2))

    # Two metrics (if present)
    for metr in ["Received","Submitted"]:
        f = fc[(fc.Country==country) & (fc.Region==region) & (fc.Metric==metr)].sort_values("Date")
        if f.empty:
            continue

        # Insert model name onto the slice if not present
        if "Chosen_Model" in f.columns and f["Chosen_Model"].notna().any():
            chosen_here = f["Chosen_Model"].dropna().iloc[0]
        else:
            chosen_here = chosen or "Model (not recorded)"
            f = f.assign(Chosen_Model=chosen_here)

        # LLM brief
        hist_series = hist[(hist.Country==country) & (hist.Region==region) & (hist.Metric==metr)].copy()
        llm_text = _llm_comment(country, region, metr, chosen_here, hist_series[["Date","Value"]], f, scores, cache)
        flow.append(Paragraph(f"{metr}", styles.h3))
        flow.append(Paragraph(llm_text, styles.body))
        flow.append(Spacer(0, 4))

        # Plot
        png = _series_plot_png(hist, fc, country, region, metr)
        flow.append(Image(io.BytesIO(png), width=18*cm, height=6*cm))
        flow.append(Spacer(0, 4))

    # Monthly numeric table (both metrics together)
    tbl_df = _monthly_table_for_region(fc, country, region)
    flow.append(Paragraph("12-month monthly distribution (rounded)", styles.h3))
    flow.append(_platypus_table(tbl_df))
    flow.append(Spacer(0, 10))

# ----------------------------- main entry ----------------------------
def build_pdf_reports() -> None:
    LOG.info("Building PDF reports …")
    hist = _history_long()
    fc   = _load_final_forecast()
    selection = _load_series_selection()
    scores = _load_scores()
    cache = _load_llm_cache()
    styles = _styles()

    # Ensure Chosen_Model is present per row (fallback to selection map)
    if "Chosen_Model" not in fc.columns or fc["Chosen_Model"].isna().all():
        if not selection.empty:
            fc = fc.merge(selection, on=["Country","Region"], how="left")
            fc["Chosen_Model"] = fc["Chosen_Model"].fillna(fc["Best_Model_Received"])
            fc = fc.drop(columns=["Best_Model_Received"])

    # Organisation-wide report
    org_path = REPORT_DIR / "organisation.pdf"
    doc = SimpleDocTemplate(str(org_path), pagesize=landscape(A4),
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.2*cm, bottomMargin=1.2*cm)
    story: List = []
    story.append(Paragraph("Organisation — Backtest & Forecast Report", styles.h1))
    story.append(Paragraph("Evaluation: strict 3-month holdout. Forecast horizon: next 12 months. Bands show uncertainty.", styles.small))
    story.append(Spacer(0, 6))

    # Overall chart
    png = _country_summary_plot(hist, fc.assign(Country="Organisation"), country="Organisation", metric="Received")
    story.append(Image(io.BytesIO(png), width=22*cm, height=7*cm))
    story.append(Spacer(0, 8))

    # Simple model selection overview table
    if not selection.empty:
        data = [["Country","Regions","Distinct models used"]]
        for c in sorted(selection["Country"].unique()):
            sub = selection[selection.Country==c]
            data.append([c, str(sub["Region"].nunique()), str(sub["Best_Model_Received"].nunique())])
        t = Table(data, colWidths=[8*cm, 6*cm, 6*cm])
        t.setStyle(TableStyle([
            ("FONT", (0,0), (-1,0), "Helvetica-Bold", 9),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F0F3F6")),
            ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("FONT", (0,1), (-1,-1), "Helvetica", 8),
        ]))
        story.append(Paragraph("Model selection summary (based on Received):", styles.h3))
        story.append(t)
        story.append(PageBreak())

    # Per-country reports
    for country in sorted(hist["Country"].unique()):
        country_doc = SimpleDocTemplate(
            str(REPORT_DIR / f"{_safe(country)}.pdf"),
            pagesize=landscape(A4),
            leftMargin=1.5*cm, rightMargin=1.5*cm,
            topMargin=1.2*cm, bottomMargin=1.2*cm
        )
        flows: List = []
        _country_story(flows, styles, country, hist, fc, selection)

        for region in sorted(hist[hist.Country==country]["Region"].unique()):
            flows.append(Spacer(0, 4))
            _region_section(flows, styles, country, region, hist, fc, selection, scores, cache)

        country_doc.build(flows)

    # Build org doc last
    doc.build(story)
    LOG.info("PDF reports written → %s", REPORT_DIR)
