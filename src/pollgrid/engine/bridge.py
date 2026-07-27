import polars as pl

from pollgrid.engine.vote_share import Cell

SupportTable = dict[str, dict[str, tuple[float, float]]]


def frame_to_cells(frame: pl.DataFrame, support_table: SupportTable) -> list[Cell]:
    """Join the poststratification frame to a per-age-band support table.

    `frame` is the constituency x age_band population table produced by
    ingest/frame.py (post collapse_to_voting_bands — its age_band values
    must match support_table's keys). `support_table` maps age_band to
    {party: (support, se)}, e.g. engine.placeholder_support.PLACEHOLDER_SUPPORT.

    Produces one Cell per (constituency, age_band, party): every party in a
    given age_band's support_table entry gets a row sharing that
    constituency-age_band's cell_id and population.

    Pure function: no I/O, no network. Belongs to the live half.
    """
    present_bands = set(frame.get_column("age_band").unique().to_list())
    missing = present_bands - set(support_table)
    if missing:
        raise ValueError(f"support table is missing age bands present in frame: {sorted(missing)}")

    cells = []
    for row in frame.iter_rows(named=True):
        cell_id = f"{row['pcon_code']}/{row['age_band']}"
        for party, (support, se) in support_table[row["age_band"]].items():
            cells.append(
                Cell(
                    cell_id=cell_id,
                    age_band=row["age_band"],
                    region=row["pcon_name"],
                    population=float(row["population"]),
                    party=party,
                    support=support,
                    se=se,
                )
            )
    return cells
