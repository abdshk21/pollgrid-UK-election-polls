# pollgrid

A tool for reading UK opinion polls honestly. Users toggle turnout assumptions by
demographic group and see how party vote share and seat counts change.

The point of this project is to stop people treating polls as gospel. That means
uncertainty is a first-class output, never a footnote.

## Vocabulary

Use these words precisely. Do not invent synonyms.

- **topline** — the headline vote share a poll reports for each party.
- **crosstab** — a poll's breakdown of support by one variable (age, education,
  region). Polls publish crosstabs one variable at a time, never combined.
- **cell** — one combination of demographic characteristics, e.g.
  `age=18-24, education=degree, region=Scotland`. Cells are the atomic unit
  of this codebase.
- **frame** — the poststratification frame. A table of how many real people
  occupy each cell, per constituency, from census data.
- **house effect** — a pollster's average distance from the polling average.
  Measures distance from other polls, NOT from truth.
- **bias** — a pollster's average distance from an actual election result.
  Different from house effect. Never use the two interchangeably.
- **scenario** — a user-specified set of turnout rates per cell.

## Architecture

Two halves. Keep them separate.

**Batch half** (`ingest/`, `model/`) — runs offline, may take twenty minutes.
Ingests polls, extracts crosstabs, builds the frame, fits the multilevel
regression. Writes a frozen artifact of cell-level estimates to disk.

**Live half** (`engine/`) — runs while a human drags a slider. Must complete in
milliseconds. Reads the frozen artifact, applies scenario weights, aggregates
to constituencies and seats.

Rules:
- `engine/` never fits a model, never touches the network, never reads a PDF.
- `engine/` contains pure functions: inputs in, numbers out, no I/O, no globals.
- If a change would make the live half slow, it belongs in the batch half.

## Non-negotiables

- **Every estimate carries its own uncertainty.** A cell built from 8
  respondents and a cell built from 400 must not be presented identically.
  Any function returning a point estimate without an interval is incomplete.
- **Never impute silently.** If a value is modelled rather than measured, the
  record says so in a field, not a comment.
- **No partisan adjustment.** Calibration happens only against real election
  results. Never against expectation, vibe, or a prior about who "should" win.
- **Data is never committed.** `data/` is gitignored. Commit the fetcher, not
  the fetch.

## Known landmines

- UK constituency boundaries were redrawn for 2024. Only about 80 seats are
  unchanged from 2019, so "previous result in this seat" mostly does not exist
  and must be handled explicitly, not assumed.
- Pollster crosstab categories are inconsistent between firms (age bands
  especially). Normalise at ingest, keep the raw value alongside.
- Weighted and unweighted bases are both published and mean different things.
  Store both.
- The frame (`ingest/frame.py`) currently covers England & Wales only (575 of
  650 UK constituencies), via ONS/Nomis. Scotland (NRS, 2022 census) and
  Northern Ireland (NISRA) publish census data through separate systems and
  need their own ingest paths. National seat totals are invalid until those
  are added.

## Conventions

- Python 3.12+, managed with `uv`. Never call `pip` directly.
- `polars`, not `pandas`.
- `pydantic` models for every data structure that crosses a module boundary.
- Type hints on all public functions.
- `ruff` for lint and format. No other formatter.
- Tests: `pytest`. Test the arithmetic in `engine/` exhaustively — it has
  hand-checkable right answers. Do not write assertion-heavy tests for model
  output; test its shape and invariants instead.

## Commands

```bash
uv run pytest              # tests
uv run ruff check --fix    # lint
uv run ruff format         # format
uv run python -m pollgrid.ingest.polls   # refresh poll data
uv run python -m pollgrid.ingest.frame    # rebuild the poststratification frame
```

## Working with me on this

- Propose a plan before multi-file changes. I want to read it first.
- One concern per commit. No omnibus commits.
- Do not add dependencies without saying why the stdlib is insufficient.
- Do not scaffold a web frontend, Docker setup, or CI pipeline unless asked.
- If a statistical shortcut would make a number look more confident than the
  data supports, say so instead of taking it.
