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
from src.config import (CODE_ROOT, DATA_ROOT, LIVE_FORMS_DIR, NOTIFY, OUTCOMES_ROOT, PREDICTIONS_ROOT, RAW_FILE,
                        SETTINGS_FILE, STAGE_FILE, load_settings, run_dir)

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
    _sync_live()
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
    """Copy a run's reports to the executive Outcomes folder and roll the live country sheets forward."""
    from src.forecasting.engine import load_result
    from src.reports.live_forms import refresh_live
    from src.reports.publish import publish
    rd = _run_date(run_date)
    out = run_dir(rd)
    res = load_result(out)
    for name, p in publish(res, out / "Reports", OUTCOMES_ROOT, __version__).items():
        typer.echo(f"  published ({name}): {p}")
    log = refresh_live(res, LIVE_FORMS_DIR, STAGE_FILE)  # the shared live sheets get the new model numbers and window
    typer.echo(f"  live country sheets in {LIVE_FORMS_DIR}:")
    typer.echo(log.to_string(index=False))
    _notify(res, out, rd)


def _notify(res, out: Path, rd: str) -> None:
    """Build the month's e-mails (previews always; sent only when [notify] enabled = true). Never blocks."""
    from src.reports import notify
    try:
        r = notify.run(res, NOTIFY, out / "Reports" / f"RMG_Monthly_Pack_{rd}.pdf", OUTCOMES_ROOT / "Latest",
                       LIVE_FORMS_DIR, SETTINGS_FILE, CODE_ROOT / ".env")
        typer.echo(f"  e-mails: {r['built']} built, {r['sent']} sent ({'on' if r['enabled'] else 'off'}); "
                   f"previews in {r['preview']}")
    except Exception as exc:                  # noqa: BLE001 - e-mail problems never stop the delivery
        typer.echo(f"  e-mails not built/sent: {type(exc).__name__}: {exc}")


@app.command("notify")
def notify_cmd(run_date: RunDate = None) -> None:
    """Build (and, when enabled in config/pipeline.toml [notify], send) the month's e-mails."""
    from src.forecasting.engine import load_result
    out = run_dir(_run_date(run_date)) if run_date else _latest_run()
    _notify(load_result(out), out, out.name)


def _sync_live() -> None:
    """Read the live country sheets into Settings.xlsx before forecasting (never blocks the run)."""
    from src.reports.live_forms import sync_live
    if not LIVE_FORMS_DIR.exists():
        return
    try:
        log = sync_live(LIVE_FORMS_DIR, SETTINGS_FILE)
        LOG.info("live country sheets read:\n%s", log.to_string(index=False))
    except Exception as exc:                  # noqa: BLE001 - e.g. Settings.xlsx open in Excel
        LOG.warning("live country sheets not read (%s: %s); using Settings.xlsx as it is", type(exc).__name__, exc)


@app.command("sync-forms")
def sync_forms_cmd() -> None:
    """Read the live country sheets into Settings.xlsx now (the forecast step also does this)."""
    from src.reports.live_forms import sync_live
    typer.echo(sync_live(LIVE_FORMS_DIR, SETTINGS_FILE).to_string(index=False))


@app.command("refresh-forms")
def refresh_forms_cmd(run_date: RunDate = None) -> None:
    """Create or roll forward the live country sheets from a run (the publish step also does this)."""
    from src.forecasting.engine import load_result
    from src.reports.live_forms import refresh_live
    out = run_dir(_run_date(run_date)) if run_date else _latest_run()
    typer.echo(refresh_live(load_result(out), LIVE_FORMS_DIR, STAGE_FILE).to_string(index=False))


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
    """Full monthly run: fetch -> transform -> forecast (reads live sheets) -> report -> publish (refreshes them)."""
    if not skip_fetch:
        fetch_data_cmd()
    if not skip_transform:
        transform_data_cmd(interactive=interactive)
    forecast_cmd(run_date)
    report_cmd(run_date)
    publish_cmd(run_date)
    typer.echo("Pipeline complete.")


@app.command("version")
def version_cmd() -> None:
    """Print the pipeline version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
