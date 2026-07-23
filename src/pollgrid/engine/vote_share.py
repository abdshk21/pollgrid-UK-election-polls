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
    """A user-specified turnout rate per cell."""

    turnout: dict[str, float]

    @field_validator("turnout")
    @classmethod
    def _rates_in_unit_interval(cls, turnout: dict[str, float]) -> dict[str, float]:
        for cell_id, rate in turnout.items():
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"turnout rate for {cell_id!r} out of [0, 1]: {rate}")
        return turnout


class Topline(BaseModel):
    """A party's national vote share estimate, with uncertainty."""

    party: str
    share: float = Field(ge=0, le=1)
    se: float = Field(ge=0)
    ci_low: float = Field(ge=0, le=1)
    ci_high: float = Field(ge=0, le=1)


def cells_to_frame(cells: list[Cell]) -> pl.DataFrame:
    """Convert validated Cell rows into the polars table the functions below expect."""
    return pl.DataFrame([c.model_dump() for c in cells], schema=CELL_SCHEMA)


def apply_scenario_weights(cells: pl.DataFrame, scenario: Scenario) -> pl.DataFrame:
    """Attach a `weight` column (population * turnout) to each cell row.

    Every cell_id in `cells` must have a turnout rate in `scenario` — a cell
    with no assumed turnout would be a silent imputation, which this codebase
    never does.
    """
    cell_ids = cells.get_column("cell_id").unique().to_list()
    missing = [cid for cid in cell_ids if cid not in scenario.turnout]
    if missing:
        raise ValueError(f"scenario is missing turnout rates for cells: {sorted(missing)}")

    turnout_frame = pl.DataFrame(
        {
            "cell_id": list(scenario.turnout.keys()),
            "turnout": list(scenario.turnout.values()),
        }
    )
    return cells.join(turnout_frame, on="cell_id", how="left").with_columns(
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
