# -*- coding: utf-8 -*-
"""
Quarterly lock for method decisions (consistency).

Each gated component (overlay, point estimate, funnel, direction signal, reconciliation) is re-tested
every run, but a new choice is ADOPTED only:
  - in a review month (the first month of each quarter: Jan, Apr, Jul, Oct), or
  - when no decision is stored yet, or
  - off-cycle, when the stored choice has become clearly worse: the best choice beats it by at least
    OVERRIDE_PCT (twice the usual 3% adoption bar).
So the published numbers do not jump because a close test tipped the other way for one month. Every
decision is written to `method_decisions.json` next to the prediction history, and to the run manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

REVIEW_MONTHS = (1, 4, 7, 10)
OVERRIDE_PCT = 6.0


class DecisionLock:
    def __init__(self, path: Path, run_date: str):
        self.path = Path(path)
        self.run_date = pd.Timestamp(run_date)
        self.state: Dict[str, dict] = {}
        if self.path.exists():
            try:
                self.state = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.state = {}
        self.log = []

    @property
    def review_month(self) -> bool:
        return self.run_date.month in REVIEW_MONTHS

    def decide(self, component: str, evidence_choice: str, scores: Optional[Dict[str, float]] = None) -> str:
        """scores: choice -> level-averaged next-quarter error (lower is better), from this run's test."""
        scores = {k: float(v) for k, v in (scores or {}).items() if v == v}
        prev = self.state.get(component)
        if prev is None or self.review_month or prev.get("choice") == evidence_choice:
            used, why = evidence_choice, ("first decision" if prev is None else
                                          "review month" if self.review_month and prev.get("choice") != evidence_choice
                                          else "confirmed")
        else:
            held = prev["choice"]
            s_held, s_best = scores.get(held), scores.get(evidence_choice)
            gain = (100 * (s_held - s_best) / s_held) if s_held and s_best is not None else 0.0
            if gain >= OVERRIDE_PCT:
                used, why = evidence_choice, f"off-cycle switch: {gain:.1f}% better (>= {OVERRIDE_PCT:.0f}%)"
            else:
                used, why = held, f"held until next review month (test favours '{evidence_choice}' by {gain:.1f}%)"
        changed = prev is None or prev.get("choice") != used
        entry = {"choice": used, "decided": (self.run_date.strftime("%Y-%m-%d") if changed else prev.get("decided")),
                 "last_checked": self.run_date.strftime("%Y-%m-%d"), "evidence_choice": evidence_choice,
                 "scores": scores, "reason": why}
        hist = (prev or {}).get("history", [])
        if changed:
            hist = hist + [{"date": self.run_date.strftime("%Y-%m-%d"), "choice": used, "reason": why}]
        entry["history"] = hist[-24:]
        self.state[component] = entry
        self.log.append({"Component": component, "Used": used, "Test_Favours": evidence_choice,
                         "Decided_On": entry["decided"], "Reason": why,
                         "Scores": ", ".join(f"{k} {v:.1f}" for k, v in scores.items())})
        return used

    def save(self) -> pd.DataFrame:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        except OSError:
            pass
        return pd.DataFrame(self.log)
