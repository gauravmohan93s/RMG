# run_pipeline.py — command-line entry point for the RMG application-flow forecast (v2)
from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import typer

from src import __version__
from src.config import DATA_ROOT, OUTCOMES_ROOT, PREDICTIONS_ROOT, RAW_FILE, SETTINGS_FILE, load_settings, run_dir

app = typer.Typer(help=f"RMG application-flow forecasting pipeline v{__version__}: "
                       "fetch -> transform -> forecast -> report", add_completion=False)
LOG = logging.getLogger("rmg")


def _setup_logging(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        handlers=[logging.FileHandler(folder / "pipeline.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)], force=True)
    for noisy in ("cmdstanpy", "prophet", "matplotlib", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _run_date(run_date: Optional[str]) -> str:
    try:
        return (dt.date.fromisoformat(run_date) if run_date else dt.date.today()).isoformat()
    except ValueError:
        raise typer.BadParameter(f"Invalid --run-date '{run_date}'. Use YYYY-MM-DD.")


RunDate = Annotated[Optional[str], typer.Option(help="Run folder date YYYY-MM-DD (default: today).")]


@app.command("fetch-data")
def fetch_data_cmd() -> None:
    """Pull the monthly extract from SQL Server (26th-25th cycles)."""
    import fetch_data
    fetch_data.main()


@app.command("transform-data")
def transform_data_cmd(
    interactive: Annotated[bool, typer.Option("--interactive", help="Pause to fix unmapped IDs in Settings.xlsx.")] = False,
) -> None:
    """Map CountryId/TagGroupID to country groups and regions (Settings.xlsx)."""
    import importlib
    import transform_data
    while True:
        missing_c, missing_t = transform_data.generate_final_summary()
        transform_data.generate_intake_summary()
        transform_data.generate_stage_summary()
        if not missing_c and not missing_t:
            typer.echo("Transformation complete (all IDs mapped).")
            return
        typer.echo(f"Unmapped IDs grouped as 'Other' — CountryIDs: {missing_c}  TagGroupIDs: {missing_t}")
        typer.echo(f"Mapping file: {transform_data.SETTINGS_PATH}")
        if interactive and sys.stdin.isatty() and typer.confirm("Update Settings.xlsx and retry?", default=False):
            typer.prompt("Press Enter when the file is saved")
            importlib.reload(transform_data)
            continue
        typer.echo("Proceeding; 'Other' series are listed in the Data Quality sheet.")
        return


@app.command("forecast")
def forecast_cmd(run_date: RunDate = None) -> None:
    """Backtest, select, forecast, reconcile and save all evidence for a run."""
    from src.forecasting.engine import run_forecast
    rd = _run_date(run_date)
    out = run_dir(rd)
    _setup_logging(out)
    LOG.info("RMG pipeline v%s — forecast run %s -> %s (data: %s)", __version__, rd, out, RAW_FILE)
    res = run_forecast(load_settings(), out, rd)
    tot = res.totals[(res.totals["Level"] == "Organisation") & (res.totals["Window"] == "Next 12 months")]
    for r in tot.itertuples():
        typer.echo(f"  {r.Metric:9s} next 12 months: {r.Forecast:,.0f}  (80% {r.Lo80:,.0f}–{r.Hi80:,.0f}, "
                   f"{r.YoY_Pct:+.1f}% YoY)")
    typer.echo(f"Forecast complete -> {out}")


@app.command("report")
def report_cmd(run_date: RunDate = None) -> None:
    """Build the Excel workbook and PDF brief from a saved forecast run."""
    from src.forecasting.engine import load_result
    from src.reports import build_reports
    rd = _run_date(run_date)
    out = run_dir(rd)
    _setup_logging(out)
    paths = build_reports(load_result(out))
    for name, p in paths.items():
        typer.echo(f"  {name}: {p}")


@app.command("publish")
def publish_cmd(run_date: RunDate = None) -> None:
    """Copy a run's pack, country PDFs, workbook and input form to the executive Outcomes folder."""
    from src.forecasting.engine import load_result
    from src.reports.publish import publish
    rd = _run_date(run_date)
    out = run_dir(rd)
    for name, p in publish(load_result(out), out / "Reports", OUTCOMES_ROOT, __version__).items():
        typer.echo(f"  published ({name}): {p}")


def _latest_run() -> Path:
    runs = sorted(p for p in PREDICTIONS_ROOT.iterdir() if (p / "run_manifest.json").exists())
    if not runs:
        raise typer.BadParameter(f"No forecast run found in {PREDICTIONS_ROOT}")
    return runs[-1]


@app.command("make-forms")
def make_forms_cmd(run_date: RunDate = None) -> None:
    """One input file per country (staffing rates + market inputs) to send to country leads."""
    from src.forecasting.engine import load_result
    from src.reports.forms import make_forms
    out = run_dir(_run_date(run_date)) if run_date else _latest_run()
    res = load_result(out)
    folder = DATA_ROOT / "Forms" / f"{pd_month(res)}"
    files = make_forms(res, folder)
    typer.echo(f"  {len(files)} country files in {folder / 'To_Send'}")
    typer.echo(f"  Put the returned files in {folder / 'Returned'}, then run: python run_pipeline.py collect-forms")


def pd_month(res) -> str:
    import pandas as pd
    return f"{pd.Timestamp(res.forecast['Date'].min()):%Y-%m}"


@app.command("collect-forms")
def collect_forms_cmd(
    folder: Annotated[Optional[Path], typer.Option(help="Forms/<YYYY-MM> folder (default: newest).")] = None,
) -> None:
    """Validate returned country files and write them into Settings.xlsx (a backup is made first)."""
    from src.reports.forms import collect_forms
    root = DATA_ROOT / "Forms"
    folder = folder or max((p for p in root.iterdir() if p.is_dir()), default=None)
    if folder is None:
        raise typer.BadParameter(f"No forms folder in {root}; run make-forms first.")
    rep = collect_forms(folder, SETTINGS_FILE)
    if rep.empty:
        typer.echo(f"  No returned files in {folder / 'Returned'}")
        return
    typer.echo(rep.to_string(index=False))
    typer.echo(f"  Settings updated: {SETTINGS_FILE} (backup: {rep.attrs.get('backup')})")


@app.command("all")
def all_cmd(
    run_date: RunDate = None,
    skip_fetch: Annotated[bool, typer.Option("--skip-fetch", help="Use the existing extract.")] = False,
    skip_transform: Annotated[bool, typer.Option("--skip-transform", help="Use the existing final summary.")] = False,
    interactive: Annotated[bool, typer.Option("--interactive")] = False,
) -> None:
    """Full monthly run: fetch -> transform -> forecast -> report -> publish -> country input forms."""
    if not skip_fetch:
        fetch_data_cmd()
    if not skip_transform:
        transform_data_cmd(interactive=interactive)
    forecast_cmd(run_date)
    report_cmd(run_date)
    publish_cmd(run_date)
    make_forms_cmd(run_date)
    typer.echo("Pipeline complete.")


@app.command("version")
def version_cmd() -> None:
    """Print the pipeline version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
