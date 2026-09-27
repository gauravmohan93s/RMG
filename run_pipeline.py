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
from src.config import RAW_FILE, load_settings, run_dir

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


@app.command("all")
def all_cmd(
    run_date: RunDate = None,
    skip_fetch: Annotated[bool, typer.Option("--skip-fetch", help="Use the existing extract.")] = False,
    skip_transform: Annotated[bool, typer.Option("--skip-transform", help="Use the existing final summary.")] = False,
    interactive: Annotated[bool, typer.Option("--interactive")] = False,
) -> None:
    """Full monthly run: fetch -> transform -> forecast -> report."""
    if not skip_fetch:
        fetch_data_cmd()
    if not skip_transform:
        transform_data_cmd(interactive=interactive)
    forecast_cmd(run_date)
    report_cmd(run_date)
    typer.echo("Pipeline complete.")


@app.command("version")
def version_cmd() -> None:
    """Print the pipeline version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
