import polars as pl
from pydantic import BaseModel, Field, field_validator

Z_95 = 1.959963984540054

CELL_SCHEMA = {
    "cell_id": pl.Utf8,
    "age_band": pl.Utf8,
    "region": pl.Utf8,
    "population": pl.Float64,
    "party": pl.Utf8,
    "support": pl.Float64,
    "se": pl.Float64,
}


class Cell(BaseModel):
    """One cell's per-party support estimate, e.g. age=18-24, region=Scotland."""

    cell_id: str
    age_band: str
    region: str
    population: float = Field(ge=0)
    party: str
    support: float = Field(ge=0, le=1)
    se: float = Field(ge=0)


class Scenario(BaseModel):
    """A user-specified turnout rate per age band."""

    turnout: dict[str, float]

    @field_validator("turnout")
    @classmethod
    def _rates_in_unit_interval(cls, turnout: dict[str, float]) -> dict[str, float]:
        for age_band, rate in turnout.items():
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"turnout rate for {age_band!r} out of [0, 1]: {rate}")
        return turnout


class Topline(BaseModel):
    """A party's national vote share estimate, with uncertainty."""

    party: str
    share: float = Field(ge=0, le=1)
    se: float = Field(ge=0)
    ci_low: float = Field(ge=0, le=1)
    ci_high: float = Field(ge=0, le=1)


def cells_to_frame(cells: list[Cell]) -> pl.DataFrame:
    """Convert validated Cell rows into the polars table the functions below expect.

    Party support values within a single cell_id must sum to 1.0 (within
    tolerance) — a cell where they don't is malformed input, not something to
    silently normalise.
    """
    frame = pl.DataFrame([c.model_dump() for c in cells], schema=CELL_SCHEMA)

    totals = frame.group_by("cell_id").agg(pl.col("support").sum().alias("total_support"))
    bad = totals.filter((pl.col("total_support") - 1.0).abs() > 1e-6)
    if bad.height:
        bad_totals = dict(bad.select("cell_id", "total_support").iter_rows())
        raise ValueError(f"cell support values do not sum to 1.0: {bad_totals}")

    return frame


def apply_scenario_weights(cells: pl.DataFrame, scenario: Scenario) -> pl.DataFrame:
    """Attach a `weight` column (population * turnout) to each cell row.

    Each cell's turnout rate is looked up by its age_band, so every cell
    sharing an age_band gets that band's rate broadcast to it. Every age_band
    present in `cells` must have a turnout rate in `scenario` — a cell with no
    assumed turnout would be a silent imputation, which this codebase never
    does.
    """
    age_bands = cells.get_column("age_band").unique().to_list()
    missing = [band for band in age_bands if band not in scenario.turnout]
    if missing:
        raise ValueError(f"scenario is missing turnout rates for age bands: {sorted(missing)}")

    turnout_frame = pl.DataFrame(
        {
            "age_band": list(scenario.turnout.keys()),
            "turnout": list(scenario.turnout.values()),
        }
    )
    return cells.join(turnout_frame, on="age_band", how="left").with_columns(
        (pl.col("population") * pl.col("turnout")).alias("weight")
    )


def national_vote_share(weighted_cells: pl.DataFrame) -> list[Topline]:
    """Aggregate weighted cells into a national vote share estimate per party.

    Combines per-cell uncertainty by treating cell-level support estimates as
    independent. That's a simplifying assumption, not a measured fact — real
    cells can share correlated error (e.g. a common pollster house effect) —
    so this likely understates true uncertainty until model/ can supply
    cross-cell covariances.
    """
    total_weight = weighted_cells.select("cell_id", "weight").unique().get_column("weight").sum()
    if total_weight <= 0:
        raise ValueError("total scenario weight is zero; check turnout and population inputs")

    per_party = weighted_cells.group_by("party").agg(
        (pl.col("weight") * pl.col("support")).sum().alias("weighted_support"),
        (pl.col("weight") ** 2 * pl.col("se") ** 2).sum().alias("weighted_variance"),
    )

    toplines = []
    for row in per_party.iter_rows(named=True):
        share = row["weighted_support"] / total_weight
        se = row["weighted_variance"] ** 0.5 / total_weight
        toplines.append(
            Topline(
                party=row["party"],
                share=share,
                se=se,
                ci_low=max(0.0, share - Z_95 * se),
                ci_high=min(1.0, share + Z_95 * se),
            )
        )
    return sorted(toplines, key=lambda t: t.party)


def topline_deltas(current: list[Topline], baseline: list[Topline]) -> dict[str, float]:
    """Per-party change in vote share, current minus baseline (share units, not pp).

    Every party in `current` must appear in `baseline` — comparing against a
    baseline that's silently missing a party would be a silent imputation.
    """
    baseline_by_party = {t.party: t.share for t in baseline}
    missing = [t.party for t in current if t.party not in baseline_by_party]
    if missing:
        raise ValueError(f"baseline is missing parties present in current: {sorted(missing)}")
    return {t.party: t.share - baseline_by_party[t.party] for t in current}
