# pollgrid

A tool for reading UK opinion polls honestly. Users toggle turnout assumptions by
demographic group and see how party vote share and seat counts change.

The point of this project is to stop people treating polls as gospel.

## Vocabulary

- **topline** — the headline vote share a poll reports for each party.
- **crosstab** — a poll's breakdown of support by one variable (age, education,
  region). Polls publish crosstabs one variable at a time.
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

**Batch half** (`ingest/`, `model/`) — runs offline, may take twenty minutes.
Ingests polls, extracts crosstabs, builds the frame, fits the multilevel
regression. Writes a frozen artifact of cell-level estimates to disk.

**Live half** (`engine/`) — runs while a human drags a slider. Must complete in
milliseconds. Reads the frozen artifact, applies scenario weights, aggregates
to constituencies and seats.

Rules:
- `engine/` contains pure functions, no globals.
- If a change would make the live half slow, it belongs in the batch half.

## Non-negotiables

- **Every estimate carries its own uncertainty.** A cell built from 8
  respondents and a cell built from 400 must not be presented identically.
  Any function returning a point estimate without an interval is incomplete.
- **Never impute silently.** If a value is modelled rather than measured, the
  record says so in a field, not a comment.
- **No partisan adjustment.** Calibration happens only against real election
  results.
- **Data is never committed.** `data/` is gitignored. Commit the fetcher, not
  the fetch. Exception: `data/crosstabs.yaml` : it's hand-curated from
  published tables, not fetched, so it's tracked like source code.

## Known landmines

- UK constituency boundaries were redrawn for 2024. Only about 80 seats are
  unchanged from 2019, so "previous result in this seat" mostly does not exist
  and must be handled explicitly.
- Pollster crosstab categories are inconsistent between firms (age bands
  especially). Normalise at ingest, keep the raw value alongside.
- Weighted and unweighted bases are both published and mean different things.
  Store both.
- The frame (`ingest/frame.py`) currently covers England & Wales only (575 of
  650 UK constituencies), via ONS/Nomis. Scotland (NRS, 2022 census) and
  Northern Ireland (NISRA) publish census data through separate systems and
  need their own ingest paths. National seat totals are invalid until those
  are added.
- britpol's `pollbase` dataset (`ingest/polls.py`) is a HISTORICAL dataset,
  not a live spine — it stops in December 2021 and only ever reports
  Con/Lab/LD shares, so it predates Reform UK's rise and the 2024 election
  entirely. Every poll parsed from it carries Reform and Green as
  `absent_parties`. Its job is providing a large, clean sample for
  validating `house_effects` against — not feeding current toplines. Current
  support numbers come from the hand-curated `data/crosstabs.yaml`.

## Conventions

- Python 3.12+, managed with `uv`. Never call `pip` directly.
- Used `polars`, not `pandas`.
- `pydantic` models for every data structure that crosses a module boundary.
- Type hints on all public functions.
- `ruff` for lint and format.
- Tests: `pytest`. Test the arithmetic in `engine/` exhaustively — it has
  hand-checkable right answers. 

## Commands

```bash
uv run pytest              # tests
uv run ruff check --fix    # lint
uv run ruff format         # format
uv run python -m pollgrid.ingest.polls   # refresh poll data
uv run python -m pollgrid.ingest.frame    # rebuild the poststratification frame
```
