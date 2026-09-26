# run_pipeline.py  –  one-stop CLI for the whole workflow (Ollama report only)
from __future__ import annotations
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Set a non-interactive backend for matplotlib to prevent GUI-related crashes
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────── SET OUT_DIR EARLY ───────────────────────────────
import os, datetime as dt
from pathlib import Path

def _resolve_run_folder(run_date: str | None) -> Path:
    """Return the run folder for a given YYYY-MM-DD or today if None."""
    # Resolve Predictions root dynamically from env if available
    data_root_str = os.environ.get("RMG_DATA_ROOT")
    if data_root_str:
        base = Path(data_root_str) / "Predictions"
    else:
        base = Path(r"C:\Users\gsakhare\OneDrive - KC OVERSEAS EDUCATION PVT LTD\RMG\RMG Calculations\Predictions")
        
    if run_date:
        try:
            d = dt.date.fromisoformat(run_date)
        except ValueError:
            raise SystemExit(f"Invalid --run-date '{run_date}'. Use YYYY-MM-DD.")
    else:
        d = dt.date.today()
    return base / d.isoformat()

# Default (overridden in CLI if user passes --run-date)
RUN_FOLDER = _resolve_run_folder(run_date=None)
os.environ["RMG_OUT_DIR"] = str(RUN_FOLDER)   # <-- key line for src.config consumers
RUN_FOLDER.mkdir(parents=True, exist_ok=True)

import src.config as cfg  # after env is set
print(f"ℹ️  All artefacts will be written to\n   {RUN_FOLDER}\n")

# ───────────────────────────────── imports ────────────────────────────────────────
import importlib
import logging
import sys
from typing import Optional, List
from typing import Annotated
import typer

# ──────────────────────────────── logging ────────────────────────────────────────
LOG_FILE = Path("pipeline.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

app = typer.Typer(help="🔄  End-to-end pipeline: Fetch → Transform → EDA → Loader → Models → Eval → Forecast → Report (Ollama LLM)")

# Global default
DEFAULT_USE_PREPARED = True  # flip to False if you deliberately want raw data

# ─────────────────────────────── utilities ───────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "src" / "models"

def _set_out_dir(run_date: Optional[str]) -> Path:
    """Set env var for the rest of the run; return the folder."""
    folder = _resolve_run_folder(run_date)
    os.environ["RMG_OUT_DIR"] = str(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def _discover_models() -> List[str]:
    """Return module paths for model files in src/models/ named modelNN_*.py."""
    mods = []
    for p in sorted(MODELS_DIR.glob("model[0-9][0-9]_*.py")):
        mods.append(f"src.models.{p.stem}")
    # Ensure ensemble runs after all base models so it can consume their predictions
    ensemble = [m for m in mods if "ensemble" in m.lower()]
    others = [m for m in mods if "ensemble" not in m.lower()]
    return others + ensemble

def _filter_models(mods: List[str], only: Optional[str], exclude: Optional[str]) -> List[str]:
    """Filter model module list by comma-separated short names (from file stems)."""
    chosen = mods[:]
    if only:
        requested = {m.strip().lower() for m in only.split(",") if m.strip()}
        chosen = [
            m for m in mods
            if m.split(".")[-1].lower() in requested
            or m.split(".")[-1].lower().replace("model", "") in requested
        ]
    if exclude:
        blocked = {m.strip().lower() for m in exclude.split(",") if m.strip()}
        chosen = [
            m for m in chosen
            if m.split(".")[-1].lower() not in blocked
            and m.split(".")[-1].lower().replace("model","") not in blocked
        ]
    return chosen

# ─────────────────────────────── 0) F E T C H  &  T R A N S F O R M ────────────────
@app.command("fetch-data")
def fetch_data_cmd() -> None:
    """Fetch data from SQL Server."""
    typer.echo("▶️  Fetching data ...")
    try:
        import fetch_data
        fetch_data.main()
        typer.echo("[bold green]✅  Fetch complete[/]")
    except Exception:
        typer.echo("[red]❌  Fetch failed — see pipeline.log/console[/]")
        logging.exception("Fetch error")
        sys.exit(1)

@app.command("transform-data")
def transform_data_cmd(
    interactive: Annotated[bool, typer.Option("--interactive", help="Prompt to pause and update Settings.xlsx if unmapped items are found.")] = False
) -> None:
    """Transform fetched data into final summary."""
    typer.echo("▶️  Transforming data ...")
    try:
        import transform_data
        
        while True:
            # Run transformation and get missing IDs
            missing_c, missing_t = transform_data.generate_final_summary()
            
            if not missing_c and not missing_t:
                typer.echo("[bold green]✅  Transformation complete (No unmapped items)[/]")
                break
            
            # Alert user about unmapped items
            typer.echo("\n[yellow]⚠️  Unmapped items detected (grouped as 'Other'):[/]")
            if missing_c:
                typer.echo(f"   Missing CountryIDs: {missing_c}")
            if missing_t:
                typer.echo(f"   Missing TagGroupIDs: {missing_t}")
            
            typer.echo(f"\n   Mapping file: {transform_data.SETTINGS_PATH}")
            
            # Only prompt if interactive mode is requested and terminal is interactive
            if interactive and sys.stdin.isatty():
                should_retry = typer.confirm("🔄  Do you want to pause, update Settings.xlsx, and RETRY the transformation?", default=False)
                if should_retry:
                    typer.echo("   (Pausing... Update the Excel file, then save it.)")
                    typer.prompt("   Press Enter when you are ready to rerun the transformation")
                    importlib.reload(transform_data)
                    continue
                else:
                    typer.echo("[yellow]   Skipping map update. Proceeding with 'Other' group.[/]")
                    break
            else:
                typer.echo("[yellow]   Non-interactive run: proceeding with 'Other' group for unmapped items.[/]")
                break

    except Exception:
        typer.echo("[red]❌  Transformation failed — see pipeline.log/console[/]")
        logging.exception("Transformation error")
        sys.exit(1)

# ─────────────────────────────── 1) E D A ────────────────────────────────────────
@app.command("eda")
def eda(
    run_date: Annotated[Optional[str], typer.Option(help="Run date folder (YYYY-MM-DD). Defaults to today.")] = None
) -> None:
    """Run the single exploratory-data script `src.eda`."""
    _set_out_dir(run_date)
    typer.echo("▶️  Running EDA …")
    try:
        eda_mod = importlib.import_module("src.eda")
        if hasattr(eda_mod, "main"):
            eda_mod.main()  # idempotent
        logging.info("EDA succeeded")
        typer.echo("[bold green]✅  EDA complete[/]")
    except Exception:
        typer.echo("[red]❌  EDA failed — see pipeline.log[/]")
        logging.exception("EDA error")
        sys.exit(1)

# ───────────────────────────── 2) L O A D E R ────────────────────────────────────
@app.command("load-data")
def load_data_cmd(
    run_date: Annotated[Optional[str], typer.Option(help="Run date folder (YYYY-MM-DD).")] = None,
    use_prepared: Annotated[bool, typer.Option("--use-prepared/--use-raw", help="Use EDA's prepared parquet instead of raw clean loader.")] = DEFAULT_USE_PREPARED
) -> None:
    """Import loader just to confirm it works."""
    _set_out_dir(run_date)
    typer.echo("▶️  Testing data loader …")
    try:
        if use_prepared:
            from src.loader import load_prepared_data as _load
            df, meta = _load()
            typer.echo(f"loaded PREPARED table – {len(df):,} rows")
            typer.echo(f"found artefacts: {list(meta.get('artefacts', {}).keys())}")
        else:
            from src.loader import load_data as _load
            df, _ = _load()
            typer.echo(f"loaded RAW-cleaned table – {len(df):,} rows")
        logging.info("Loader succeeded")
        typer.echo("[bold green]✅  Loader OK[/]")
    except Exception:
        typer.echo("[red]❌  Loader failed — see pipeline.log[/]")
        logging.exception("Loader error")
        sys.exit(1)

# ─────────────────────────────── 3) M O D E L S ──────────────────────────────────
@app.command("run-models")
def run_models(
    run_date: Annotated[Optional[str], typer.Option(help="Run date folder (YYYY-MM-DD).")] = None,
    only: Annotated[Optional[str], typer.Option("--only", help="Comma-separated model stems to run (e.g. 'model02_mom_yoy,model07_xgboost').")] = None,
    exclude: Annotated[Optional[str], typer.Option("--exclude", help="Comma-separated model stems to skip.")] = None,
) -> None:
    _set_out_dir(run_date)
    all_mods = _discover_models()
    chosen = _filter_models(all_mods, only, exclude)
    typer.echo(f"▶️  Found {len(all_mods)} model(s). Running {len(chosen)}: {', '.join(chosen) if chosen else '(none)'}")

    if not chosen:
        typer.echo("[yellow]No models selected to run.[/]")
        return

    for mod in chosen:
        typer.echo(f"   ↪ Running {mod} …")
        try:
            m = importlib.import_module(mod)
            if hasattr(m, "main"):
                m.main()
            logging.info("Model %s succeeded", mod)
            typer.echo("      ✅ done")
        except Exception:
            typer.echo(f"[red]❌  {mod} failed — see pipeline.log[/]")
            logging.exception("Model error in %s", mod)
            sys.exit(1)
    typer.echo("[bold green]✅  All models finished[/]")

# ─────────────────────────────── 4) E V A L U A T E ──────────────────────────────
@app.command("evaluate")
def evaluate(
    run_date: Annotated[Optional[str], typer.Option(help="Run date folder (YYYY-MM-DD).")] = None
) -> None:
    _set_out_dir(run_date)
    typer.echo("▶️  Evaluating …")
    try:
        import src.models.evaluator as ev
        if hasattr(ev, "main"):
            ev.main()
        logging.info("Evaluator succeeded")
        typer.echo("[bold green]✅  Evaluation complete[/]")
    except Exception:
        typer.echo("[red]❌  Evaluator failed — see pipeline.log[/]")
        logging.exception("Evaluator error")
        sys.exit(1)

# ─────────────────────────────── 5) F O R E C A S T ──────────────────────────────
@app.command("forecast")
def forecast(
    run_date: Annotated[Optional[str], typer.Option(help="Run date folder (YYYY-MM-DD).")] = None
) -> None:
    _set_out_dir(run_date)
    typer.echo("▶️  Generating final forecast …")
    try:
        import src.models.final_forecast as ff
        if hasattr(ff, "main"):
            ff.main()
        logging.info("Forecast succeeded")
        typer.echo("[bold green]✅  Forecast ready[/]")
    except Exception:
        typer.echo("[red]❌  Forecast failed — see pipeline.log[/]")
        logging.exception("Forecast error")
        sys.exit(1)

# ─────────────────────────────── 6) R E P O R T (OLLAMA) ─────────────────────────
@app.command("report")
def report(
    run_date: Annotated[Optional[str], typer.Option(help="Run date folder (YYYY-MM-DD).")] = None
) -> None:
    """
    Build reports using the LLM (Ollama) commentary engine.
    Expects Ollama running locally and a small model available (e.g., phi3:mini or qwen2:0.5b).
    """
    _set_out_dir(run_date)
    typer.echo("▶️  Building reports (Ollama LLM) …")
    try:
        # Optionally ensure defaults for local setups:
        os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")
        os.environ.setdefault("OLLAMA_MODEL_ORDER", "phi3:mini,qwen2:0.5b")
        os.environ.setdefault("OLLAMA_NUM_CTX", "512")

        from src.reports.final_pdf_reports import build_pdf_reports
        build_pdf_reports()
        logging.info("Reports succeeded")
        typer.echo("[bold green]✅  Reports complete[/]")
    except Exception:
        typer.echo("[red]❌  Report failed — see pipeline.log[/]")
        logging.exception("Report error")
        sys.exit(1)

# ─────────────────────────────── 7) A L L  S T E P S ─────────────────────────────
@app.command("all")
def all_steps(
    run_date: Annotated[Optional[str], typer.Option(help="Run date folder (YYYY-MM-DD).")] = None,
    use_prepared: Annotated[bool, typer.Option("--use-prepared/--use-raw", help="Use EDA prepared parquet for loader step.")] = DEFAULT_USE_PREPARED,
    only: Annotated[Optional[str], typer.Option("--only", help="Run only these model stems, comma-separated.")] = None,
    exclude: Annotated[Optional[str], typer.Option("--exclude", help="Skip these model stems, comma-separated.")] = None,
    skip_fetch: Annotated[bool, typer.Option("--skip-fetch", help="Skip data fetching.")] = False,
    skip_transform: Annotated[bool, typer.Option("--skip-transform", help="Skip data transformation.")] = False,
    interactive: Annotated[bool, typer.Option("--interactive", help="Enable interactive pause for unmapped items.")] = False,
) -> None:
    """Run the full pipeline in one go. Reports are generated via Ollama."""
    _set_out_dir(run_date)

    # 0a) Fetch
    if not skip_fetch:
        fetch_data_cmd()
    
    # 0b) Transform
    if not skip_transform:
        transform_data_cmd(interactive=interactive)

    # 1) EDA
    eda(run_date=run_date)

    # 2) Loader smoke-test
    load_data_cmd(run_date=run_date, use_prepared=use_prepared)

    # 3) Models
    run_models(run_date=run_date, only=only, exclude=exclude)

    # 4) Evaluate (last 3 months OOS scoring)
    evaluate(run_date=run_date)

    # 5) Forecast (assemble 12-month forecast using evaluator's BU selections)
    forecast(run_date=run_date)

    # 6) Reports (PDF org + per-BU, with LLM rationale & plots via Ollama)
    report(run_date=run_date)

    typer.echo("[bold magenta]🎉  Pipeline complete![/]")

if __name__ == "__main__":
    app()
