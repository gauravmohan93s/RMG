"""Executive reporting: Excel workbook (planning + evidence) and PDF brief."""
from __future__ import annotations

from pathlib import Path
from typing import Dict


def build_reports(res) -> Dict[str, Path]:
    from .excel import write_workbook
    from .pdf import write_brief

    out = res.out_dir / "Reports"
    out.mkdir(parents=True, exist_ok=True)
    return {
        "workbook": write_workbook(res, out / f"RMG_Forecast_{res.run_date}.xlsx"),
        "brief": write_brief(res, out / f"RMG_Executive_Brief_{res.run_date}.pdf"),
    }
