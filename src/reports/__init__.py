"""Executive reporting: Excel workbook (planning + evidence) and PDF brief."""
from __future__ import annotations

from pathlib import Path
from typing import Dict


def build_reports(res) -> Dict[str, Path]:
    from .excel import write_workbook
    from .human_form import write_form
    from .pack import write_country_packs, write_story

    out = res.out_dir / "Reports"
    out.mkdir(parents=True, exist_ok=True)
    return {
        "workbook": write_workbook(res, out / f"RMG_Forecast_{res.run_date}.xlsx"),
        "brief": write_story(res, out / f"RMG_Monthly_Pack_{res.run_date}.pdf"),
        "human_form": write_form(res, out / f"Human_Inputs_Form_{res.run_date}.xlsx"),
        "countries": write_country_packs(res, out / "Countries"),
    }
