# pollgrid

A tool for reading UK opinion polls honestly. Users toggle turnout assumptions by
demographic group and see how party vote share and seat counts change.

Uncertainty is a first-class output, never a footnote.

## Architecture

Two halves, kept separate:

- **Batch half** (`ingest/`, `model/`) — runs offline. Ingests polls, extracts
  crosstabs, builds the poststratification frame, fits the multilevel
  regression. Writes a frozen artifact of cell-level estimates to disk.
- **Live half** (`engine/`) — runs while a human drags a slider. Pure
  functions only: reads the frozen artifact, applies scenario weights,
  aggregates to constituencies and seats. No I/O, no model fitting.

See `CLAUDE.md` for the full vocabulary, architecture rules, and non-negotiables.

## Setup

```bash
uv sync
```

## Commands

```bash
uv run pytest              # tests
uv run ruff check --fix    # lint
uv run ruff format         # format
uv run python -m pollgrid.ingest.polls   # refresh poll data
```
