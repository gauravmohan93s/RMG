# -*- coding: utf-8 -*-
"""
Monthly e-mails (built every run, SENT ONLY when switched on).

  summary     to the leadership list: the headline numbers and key messages, the destination table, where to find
              the pack (Outcomes/Latest), the pack PDF attached
  reminders   to each destination team whose live sheet has not been reviewed for REMIND_DAYS (or never)

Every run writes the messages as .eml previews to Reports/Email_preview/ (open them in Outlook to check).
Sending needs config/pipeline.toml [notify] enabled = true, recipients, and SMTP settings in the environment or
the git-ignored .env: RMG_SMTP_HOST, RMG_SMTP_PORT (587), RMG_SMTP_USER, RMG_SMTP_PASSWORD, RMG_MAIL_FROM.
Team addresses: Settings.xlsx sheet Live_Sheet_Contacts (Country, Email; several rows per country allowed).
"""
from __future__ import annotations

import html
import logging
import os
import smtplib
from datetime import date
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.prepare import ORG
from . import insights as ins

LOG = logging.getLogger("rmg.notify")
REMIND_DAYS = 45
MAX_ATTACH_MB = 15


def _env(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE); existing environment variables win."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _k(v) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{round(v / 100) * 100:,.0f}"


def summary_email(res, cfg: dict, pack: Optional[Path], outcomes_dir: Path) -> EmailMessage:
    first = pd.Timestamp(res.forecast["Date"].min())
    q = res.totals[(res.totals["Level"] == "Organisation") & (res.totals["Metric"] == "Received")
                   & (res.totals["Window"] == "Next 3 months")]
    head = f"next quarter {_k(q['Forecast'].iloc[0])} ({q['YoY_Pct'].iloc[0]:+.0f}% vs last year)" if len(q) else ""
    msg = EmailMessage()
    msg["Subject"] = f"{cfg.get('subject_prefix', 'RMG forecast')} — {first:%B %Y} pack: {head}"
    msg["From"] = os.environ.get("RMG_MAIL_FROM", "")
    msg["To"] = ", ".join(cfg.get("to", []))
    if cfg.get("cc"):
        msg["Cc"] = ", ".join(cfg["cc"])
    bullets = "".join(f"<li>{m}</li>" for m in ins.story_messages(res, "Organisation", ORG, n=5))
    ct = ins.country_table(res)
    rows = "".join(
        f"<tr><td>{html.escape('Organisation' if r.Level == 'Organisation' else r.Country)}</td>"
        f"<td align=right>{_k(getattr(r, 'Received_Next_Quarter', np.nan))}</td><td align=right>{_k(r.Received_Forecast)}</td>"
        f"<td align=right>{r.Received_YoY_Pct:+.0f}%</td><td>{html.escape(str(r.Received_Reliability))}</td></tr>"
        for r in ct.itertuples())
    body = f"""<html><body style="font-family:Calibri,Arial,sans-serif;font-size:11pt;color:#0b0b0b">
<p>The application-flow forecast for <b>{first:%B %Y}</b> onwards is ready (actuals to {res.last_actual:%B %Y}).</p>
<ul>{bullets}</ul>
<table cellpadding=4 cellspacing=0 style="border-collapse:collapse;font-size:10pt">
<tr style="background:#eef2f7"><th align=left>Destination</th><th>Next quarter</th><th>Next 12 months</th><th>vs last year</th>
<th>Reliability</th></tr>{rows}</table>
<p>The full pack (one story from the organisation down to each destination's regions), the destination PDFs and the
workbook are in <b>{html.escape(str(outcomes_dir))}</b>{' — the pack is attached' if pack else ''}.</p>
<p style="color:#52514e;font-size:9pt">Ranges: "plan for" holds about 8 times in 10. Sent automatically by the RMG
forecasting pipeline.</p></body></html>"""
    msg.set_content("The monthly forecast pack is ready. Open this message in an HTML-capable mail client.")
    msg.add_alternative(body, subtype="html")
    if pack and Path(pack).exists() and Path(pack).stat().st_size < MAX_ATTACH_MB * 2 ** 20:
        msg.add_attachment(Path(pack).read_bytes(), maintype="application", subtype="pdf", filename=Path(pack).name)
    return msg


def live_status(folder: Path) -> pd.DataFrame:
    """Per live sheet: destination, views entered, FTE entries, last reviewed date (any row)."""
    from .live_forms import _as_date, _read_raw
    rows = []
    for p in (sorted(Path(folder).glob("*.xlsx")) if Path(folder).exists() else []):
        if p.name.startswith("~$") or p.name.endswith(".tmp.xlsx"):
            continue
        try:
            raw = _read_raw(p)
        except Exception:                                        # noqa: BLE001 - unreadable sheet: reported as such
            rows.append({"Country": "", "Label": p.stem, "File": p.name, "Views": 0, "FTE": 0, "Last_Reviewed": None})
            continue
        dates = [_as_date(n[2]) for n in raw["notes"].values()] + [_as_date(s["date"]) for s in raw["staff"].values()]
        dates = [d for d in dates if d]
        rows.append({"Country": raw["country"], "Label": raw["label"], "File": p.name, "Views": len(raw["views"]),
                     "FTE": sum(1 for s in raw["staff"].values() for v in s["fte"].values() if np.isfinite(v)),
                     "Last_Reviewed": max(dates) if dates else None})
    return pd.DataFrame(rows, columns=["Country", "Label", "File", "Views", "FTE", "Last_Reviewed"])


def reminder_emails(status: pd.DataFrame, contacts: pd.DataFrame, folder: Path, today: date) -> List[EmailMessage]:
    out = []
    if status.empty or contacts is None or contacts.empty:
        return out
    for r in status.itertuples():
        last = r.Last_Reviewed
        if last is not None and not pd.isna(last) and (pd.Timestamp(today) - pd.Timestamp(last)).days < REMIND_DAYS:
            continue
        to = contacts.loc[contacts["Country"].astype(str) == str(r.Country), "Email"].dropna().astype(str).tolist()
        if not to:
            continue
        msg = EmailMessage()
        msg["Subject"] = f"Please review your market view — {r.Label or r.Country}"
        msg["From"] = os.environ.get("RMG_MAIL_FROM", "")
        msg["To"] = ", ".join(to)
        when = (f"last reviewed {pd.Timestamp(last):%d %b %Y}" if last is not None and not pd.isna(last)
                else "not reviewed yet")
        msg.set_content(
            f"Hello,\n\nThe forecast for {r.Label or r.Country} has been refreshed. Your live sheet is {when}.\n"
            f"Please open {Path(folder) / r.File}, check the yellow 'Your view' cells against the new model numbers, "
            "update the headcount (FTE) per stage on the Staffing sheet, and put today's date in 'Last reviewed'.\n\n"
            "It takes a few minutes and is read automatically at the next monthly run.\n\nRMG forecasting pipeline")
        out.append(msg)
    return out


def _send(msgs: List[EmailMessage]) -> int:
    host = os.environ.get("RMG_SMTP_HOST")
    if not host:
        raise RuntimeError("RMG_SMTP_HOST is not set (environment or .env)")
    with smtplib.SMTP(host, int(os.environ.get("RMG_SMTP_PORT", "587")), timeout=60) as s:
        s.starttls()
        if os.environ.get("RMG_SMTP_USER"):
            s.login(os.environ["RMG_SMTP_USER"], os.environ.get("RMG_SMTP_PASSWORD", ""))
        for m in msgs:
            s.send_message(m)
    return len(msgs)


def run(res, cfg: dict, pack: Optional[Path], outcomes_dir: Path, live_folder: Path, settings_file: Optional[Path],
        env_file: Optional[Path] = None) -> Dict[str, object]:
    """Build the month's e-mails, save previews, and send them only when cfg['enabled'] is true."""
    if env_file is not None:
        _env(env_file)
    msgs = [summary_email(res, cfg, pack, outcomes_dir)]
    contacts = pd.DataFrame()
    if settings_file is not None and Path(settings_file).exists():
        with pd.ExcelFile(settings_file) as xl:
            if "Live_Sheet_Contacts" in xl.sheet_names:
                contacts = pd.read_excel(xl, "Live_Sheet_Contacts")
    if cfg.get("remind_teams", True):
        msgs += reminder_emails(live_status(live_folder), contacts, live_folder, date.today())
    prev = Path(res.out_dir) / "Reports" / "Email_preview"
    prev.mkdir(parents=True, exist_ok=True)
    for f in prev.glob("*.eml"):
        f.unlink()
    for i, m in enumerate(msgs):
        (prev / f"{i:02d}_{'summary' if i == 0 else 'reminder'}.eml").write_bytes(bytes(m))
    sent = _send(msgs) if cfg.get("enabled", False) and cfg.get("to") else 0
    return {"built": len(msgs), "sent": sent, "preview": prev, "enabled": bool(cfg.get("enabled", False))}
