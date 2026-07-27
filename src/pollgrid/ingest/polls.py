"""HISTORICAL poll data from britpol (github.com/jackobailey/britpol) — NOT a
live spine.

britpol's `pollbase` dataset is a long-format list of individual historical
UK voting-intention polls (as opposed to `pollbasepro`, a smoothed daily
poll-of-polls trend with no per-poll pollster/sample-size detail — not what
we want here). Its job in this codebase is providing a large, clean sample
to validate the house_effects() arithmetic against — it stops in December
2021 and only ever reports Con/Lab/LD, so it cannot and does not feed
current app toplines. Current support numbers come from the hand-curated
`data/crosstabs.yaml` (see load_crosstabs / crosstabs_to_support_table
below).

Two landmines worth knowing before using this data:
- britpol ships `pollbase` as R's binary .rda format, not CSV — there is no
  stdlib or pure-CSV path to read it, hence the `rdata` dependency (pure
  Python, no R runtime required).
- `pollbase` only ever reports Con/Lab/LD shares, and its coverage ends in
  December 2021 — before Reform UK's rise and the 2024 election. Every poll
  parsed from it will carry Reform and Green in `absent_parties`.
"""

import urllib.request
from collections import defaultdict
from datetime import date, timedelta
from math import sqrt
from pathlib import Path

import polars as pl
import rdata
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from pollgrid.engine.bridge import SupportTable
from pollgrid.ingest.frame import VOTING_BANDS

POLLBASE_URL = "https://raw.githubusercontent.com/jackobailey/britpol/master/data/pollbase.rda"
POLLS_ARTIFACT_PATH = Path("data/polls.parquet")
CROSSTABS_PATH = Path("data/crosstabs.yaml")

# R's Date columns are stored as days since this epoch; britpol's `start`/
# `end` columns are floats in that encoding, not python dates.
R_DATE_EPOCH = date(1970, 1, 1)

# britpol's raw column -> our canonical party name.
BRITPOL_PARTY_COLUMNS: dict[str, str] = {"con": "Con", "lab": "Lab", "lib": "LD"}

# The parties we track coverage for. pollbase structurally only ever
# populates Con/Lab/LD (see module docstring), so Reform and Green will show
# up as absent_parties on every single row until a newer source is added.
TRACKED_PARTIES = ("Con", "Lab", "LD", "Reform", "Green")

POLLS_SCHEMA = {
    "poll_id": pl.Utf8,
    "pollster": pl.Utf8,
    "pollster_raw": pl.Utf8,
    "fieldwork_start": pl.Date,
    "fieldwork_end": pl.Date,
    "sample_size": pl.Int64,
    "party": pl.Utf8,
    "share": pl.Float64,
}

# Known case/spacing variants of the SAME pollster brand (e.g. YouGov spelled
# a few different ways). Deliberately does NOT merge distinct brand names
# that happen to share a corporate lineage (ComRes -> Savanta ComRes, MORI ->
# Ipsos MORI) — that's an editorial call about company history, not a
# spelling fix, and this module never makes that call silently.
POLLSTER_ALIASES: dict[str, str] = {
    "yougov": "YouGov",
    "you gov": "YouGov",
    "gfk": "GfK",
    "comres": "ComRes",
    "savanta comres": "Savanta ComRes",
    "ipsos mori": "Ipsos MORI",
    "orb": "ORB",
    "nop": "NOP",
    "tns": "TNS",
    "icm": "ICM",
    "bmg": "BMG",
}


def normalise_pollster(raw: str) -> str:
    """Canonical pollster name for a raw published string.

    Only fixes case/spacing variants of the same brand (see
    POLLSTER_ALIASES) — never merges different brand names. Unknown strings
    pass through stripped, unchanged, so the raw value is never silently
    reinterpreted.
    """
    return POLLSTER_ALIASES.get(raw.strip().lower(), raw.strip())


class PollTopline(BaseModel):
    """One poll's fieldwork, sample size, and per-party vote-share topline.

    `shares` holds only the parties this poll actually reported —
    `absent_parties` names the rest of TRACKED_PARTIES rather than the
    model inventing a share for them.
    """

    poll_id: str
    pollster: str
    pollster_raw: str
    fieldwork_start: date
    fieldwork_end: date
    sample_size: int | None = Field(default=None, ge=0)
    shares: dict[str, float]
    absent_parties: list[str]

    @field_validator("shares")
    @classmethod
    def _shares_in_unit_interval(cls, shares: dict[str, float]) -> dict[str, float]:
        for party, share in shares.items():
            if not 0.0 <= share <= 1.0:
                raise ValueError(f"share for {party!r} out of [0, 1]: {share}")
        return shares

    @model_validator(mode="after")
    def _fieldwork_start_before_end(self) -> "PollTopline":
        if self.fieldwork_start > self.fieldwork_end:
            raise ValueError(
                f"fieldwork_start {self.fieldwork_start} is after "
                f"fieldwork_end {self.fieldwork_end}"
            )
        return self


def fetch_pollbase_raw() -> pl.DataFrame:
    """Download britpol's `pollbase` dataset as-is: one row per poll, columns
    id/start/end/pollster/n/con/lab/lib exactly as britpol publishes them.
    """
    with urllib.request.urlopen(POLLBASE_URL, timeout=60) as response:
        body = response.read()
    parsed = rdata.parser.parse_data(body, extension=".rda")
    pollbase = rdata.conversion.convert(parsed)["pollbase"]
    return pl.from_pandas(pollbase)


def parse_pollbase(raw: pl.DataFrame) -> list[PollTopline]:
    """Validate and normalise raw pollbase rows into PollToplines.

    Pure function, no I/O. Rows with no pollster, no fieldwork dates, or no
    party shares at all are dropped — that's the ordinary shape of scraped
    historical data, not a structural integrity failure the way a missing
    frame lookup would be, so we drop and count rather than raise.
    """
    polls: list[PollTopline] = []
    for row in raw.iter_rows(named=True):
        poll_id = row["id"]
        pollster_raw = row["pollster"]
        start_days = row["start"]
        end_days = row["end"]
        if not poll_id or not pollster_raw or start_days is None or end_days is None:
            continue

        shares = {
            party: float(row[column])
            for column, party in BRITPOL_PARTY_COLUMNS.items()
            if row[column] is not None
        }
        if not shares:
            continue

        sample_size = row["n"]
        polls.append(
            PollTopline(
                poll_id=poll_id,
                pollster=normalise_pollster(pollster_raw),
                pollster_raw=pollster_raw,
                fieldwork_start=R_DATE_EPOCH + timedelta(days=int(start_days)),
                fieldwork_end=R_DATE_EPOCH + timedelta(days=int(end_days)),
                sample_size=int(sample_size) if sample_size is not None else None,
                shares=shares,
                absent_parties=sorted(set(TRACKED_PARTIES) - set(shares)),
            )
        )
    return polls


def polls_to_frame(polls: list[PollTopline]) -> pl.DataFrame:
    """Long-format table for storage: one row per (poll, party reported)."""
    rows = [
        {
            "poll_id": poll.poll_id,
            "pollster": poll.pollster,
            "pollster_raw": poll.pollster_raw,
            "fieldwork_start": poll.fieldwork_start,
            "fieldwork_end": poll.fieldwork_end,
            "sample_size": poll.sample_size,
            "party": party,
            "share": share,
        }
        for poll in polls
        for party, share in poll.shares.items()
    ]
    return pl.DataFrame(rows, schema=POLLS_SCHEMA)


def house_effects(polls: list[PollTopline], window_days: int) -> dict[str, dict[str, float]]:
    """Per-pollster, per-party average signed distance from the
    cross-pollster mean, over a trailing window ending at the most recent
    poll's fieldwork_end.

    This is HOUSE EFFECT, not bias: it measures each pollster's distance
    from other pollsters' average over the window, not from any actual
    election result. Computing bias requires real election results and is
    out of scope for this function.

    Each poll contributes equally to the cross-pollster mean regardless of
    sample size — a simplifying choice that keeps this hand-checkable, not a
    claim that sample size doesn't matter.
    """
    if not polls:
        return {}

    latest_end = max(poll.fieldwork_end for poll in polls)
    window_start = latest_end - timedelta(days=window_days)
    in_window = [poll for poll in polls if window_start <= poll.fieldwork_end <= latest_end]

    party_shares: dict[str, list[float]] = defaultdict(list)
    for poll in in_window:
        for party, share in poll.shares.items():
            party_shares[party].append(share)
    overall_mean = {party: sum(shares) / len(shares) for party, shares in party_shares.items()}

    deviations: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for poll in in_window:
        for party, share in poll.shares.items():
            deviations[poll.pollster][party].append(share - overall_mean[party])

    return {
        pollster: {party: sum(devs) / len(devs) for party, devs in party_devs.items()}
        for pollster, party_devs in deviations.items()
    }


class CrosstabRow(BaseModel):
    """One (pollster, age_band, party) cell from a hand-typed published
    crosstab — see data/crosstabs.yaml for the field-by-field provenance and
    the schema this is loaded from.
    """

    pollster: str
    fieldwork_start: date
    fieldwork_end: date
    age_band: str
    party: str
    support: float = Field(ge=0, le=1)
    subsample_base: int = Field(gt=0)

    @field_validator("age_band")
    @classmethod
    def _age_band_is_a_voting_band(cls, age_band: str) -> str:
        if age_band not in VOTING_BANDS:
            raise ValueError(
                f"age_band {age_band!r} is not one of the four voting bands: {VOTING_BANDS}"
            )
        return age_band

    @model_validator(mode="after")
    def _fieldwork_start_before_end(self) -> "CrosstabRow":
        if self.fieldwork_start > self.fieldwork_end:
            raise ValueError(
                f"fieldwork_start {self.fieldwork_start} is after "
                f"fieldwork_end {self.fieldwork_end}"
            )
        return self


def load_crosstabs(path: Path) -> list[CrosstabRow]:
    """Load and validate data/crosstabs.yaml — hand-curated, not fetched.

    Raises if any row's age_band isn't one of the four voting bands from
    ingest/frame.py's collapse_to_voting_bands, or if any (pollster,
    fieldwork_start, fieldwork_end, age_band) group's support values don't
    sum to 1.0 within tolerance once Other is included — the same
    "close the loop honestly, don't silently rescale" invariant
    engine.cells_to_frame enforces.
    """
    with path.open() as f:
        raw_rows = yaml.safe_load(f) or []

    rows = [CrosstabRow(**raw_row) for raw_row in raw_rows]

    totals: dict[tuple[str, date, date, str], float] = defaultdict(float)
    for row in rows:
        key = (row.pollster, row.fieldwork_start, row.fieldwork_end, row.age_band)
        totals[key] += row.support

    bad = {key: total for key, total in totals.items() if abs(total - 1.0) > 1e-6}
    if bad:
        raise ValueError(f"crosstab support values do not sum to 1.0: {bad}")

    return rows


def crosstabs_to_support_table(crosstabs: list[CrosstabRow], pollster: str) -> SupportTable:
    """Build the {age_band: {party: (support, se)}} shape
    engine.bridge.frame_to_cells expects, from one pollster's crosstab rows.

    SE is derived the same way as engine.placeholder_support: sqrt(p*(1-p)/n)
    from the row's own subsample_base. That base is a real published figure
    here (unlike placeholder_support's invented one), but the SE itself is
    still derived, not published — pollsters don't print a margin of error
    per crosstab cell.
    """
    rows = [row for row in crosstabs if row.pollster == pollster]
    if not rows:
        raise ValueError(f"no crosstab rows found for pollster {pollster!r}")

    table: SupportTable = {}
    for row in rows:
        table.setdefault(row.age_band, {})[row.party] = (
            row.support,
            sqrt(row.support * (1 - row.support) / row.subsample_base),
        )
    return table


def main() -> None:
    raw = fetch_pollbase_raw()
    polls = parse_pollbase(raw)
    rejected = raw.height - len(polls)

    POLLS_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    polls_to_frame(polls).write_parquet(POLLS_ARTIFACT_PATH)

    print(f"Wrote {len(polls)} polls ({rejected} rejected) to {POLLS_ARTIFACT_PATH}")
    print(
        "This is HISTORICAL data for validating house_effects, not a live spine: "
        "britpol's pollbase only reports Con/Lab/LD and stops in December 2021 — "
        "every poll here predates Reform UK's rise and the 2024 election, so Reform "
        "and Green will show up as absent_parties on every row. For current support "
        "numbers, see data/crosstabs.yaml."
    )


if __name__ == "__main__":
    main()
