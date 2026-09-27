# Project notes for coding agents

Forecasting pipeline (v2) for monthly applications Received/Submitted by region, country and
organisation. Read `README.md` and `docs/METHODOLOGY.md` first.

## Run
- `python run_pipeline.py all | forecast | report | fetch-data | transform-data | version`
- `python -m pytest` must pass before any commit that changes numbers.

## Rules
- Models only ever see history up to the forecast origin — slicing happens exclusively in
  `src/forecasting/backtest.py`. Never compute bounds, scalers or features on full series inside a model.
- Never hand-pick a model for a series; add candidates to the registry and let the
  rolling-origin tournament and walk-forward policy evaluation decide.
- All settings live in `config/pipeline.toml`; a change that moves numbers needs a version bump
  and a `CHANGELOG.md` entry.
- Outputs must satisfy the invariants in `src/forecasting/guardrails.py::check_invariants`.
- Legacy v1 code is only in git tag `v1.0.0`.
