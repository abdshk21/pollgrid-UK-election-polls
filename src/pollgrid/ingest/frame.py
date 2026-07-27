"""England & Wales only (575 of 650 UK constituencies).

Scotland (NRS, 2022 census) and Northern Ireland (NISRA) publish their own
census data under separate systems and need their own ingest paths — see
CLAUDE.md's Known landmines. Until those exist, national seat totals
computed from this frame are invalid; only E&W constituencies are covered.
"""

import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl
from pydantic import BaseModel, Field

NOMIS_DATASET_ID = "NM_2020_1"
NOMIS_GEOGRAPHY_TYPE = "TYPE151"  # 2021 LSOAs
NOMIS_AGE_CODES = [str(i) for i in range(1, 19)]  # excludes code 0, "Total"
NOMIS_MEASURE = "20100"  # value
NOMIS_URL = f"https://www.nomisweb.co.uk/api/v01/dataset/{NOMIS_DATASET_ID}.data.csv"

# LSOA (2021) to Westminster Parliamentary Constituency (July 2024) best fit
# lookup. "Best fit" is ONS's own methodology: an LSOA that straddles two
# constituencies is wholly assigned to the one it best matches, not split.
LOOKUP_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "LSOA21_PCON24_LAD21_EW_LU/FeatureServer/0/query"
)

FRAME_ARTIFACT_PATH = Path("data/frame.parquet")
VOTING_BANDS_ARTIFACT_PATH = Path("data/frame_voting_bands.parquet")

# UK pollsters publish age crosstabs against these four bands, not the
# eighteen five-year Census bands — this is the binding constraint on how
# fine-grained the support table can be.
VOTING_BANDS = ("18-24", "25-49", "50-64", "65+")

# Census bands with no electors. Dropped entirely: never counted in a
# vote-share denominator.
UNDER_15_CENSUS_BANDS = frozenset(
    {
        "Aged 4 years and under",
        "Aged 5 to 9 years",
        "Aged 10 to 14 years",
    }
)

# "Aged 15 to 19 years" straddles the voting-age line. This assumes ages are
# uniformly distributed within the band, so 18 and 19 year-olds are 2 of its
# 5 years — a stated modelling assumption, not a measured fact.
FIFTEEN_TO_NINETEEN_ADULT_FRACTION = 2 / 5

# Maps each Census band that contributes to a voting band -> (voting_band,
# fraction of that band's population counted as electors).
CENSUS_TO_VOTING_BAND: dict[str, tuple[str, float]] = {
    "Aged 15 to 19 years": ("18-24", FIFTEEN_TO_NINETEEN_ADULT_FRACTION),
    "Aged 20 to 24 years": ("18-24", 1.0),
    "Aged 25 to 29 years": ("25-49", 1.0),
    "Aged 30 to 34 years": ("25-49", 1.0),
    "Aged 35 to 39 years": ("25-49", 1.0),
    "Aged 40 to 44 years": ("25-49", 1.0),
    "Aged 45 to 49 years": ("25-49", 1.0),
    "Aged 50 to 54 years": ("50-64", 1.0),
    "Aged 55 to 59 years": ("50-64", 1.0),
    "Aged 60 to 64 years": ("50-64", 1.0),
    "Aged 65 to 69 years": ("65+", 1.0),
    "Aged 70 to 74 years": ("65+", 1.0),
    "Aged 75 to 79 years": ("65+", 1.0),
    "Aged 80 to 84 years": ("65+", 1.0),
    "Aged 85 years and over": ("65+", 1.0),
}


class FrameCell(BaseModel):
    """Population of one (constituency, age band) cell from the census."""

    pcon_code: str
    pcon_name: str
    age_band: str
    population: int = Field(ge=0)


def fetch_lsoa_age_counts(page_size: int = 25_000) -> pl.DataFrame:
    """Download Census 2021 usual-resident counts by age band, per LSOA.

    Source: Nomis dataset NM_2020_1 (published as TS007A), England & Wales
    only. Age bands are the 18 five-year bands Nomis publishes; the "Total"
    pseudo-band is excluded.
    """
    params = {
        "geography": NOMIS_GEOGRAPHY_TYPE,
        "c2021_age_19": ",".join(NOMIS_AGE_CODES),
        "measures": NOMIS_MEASURE,
        "select": "geography_code,c2021_age_19_name,obs_value,record_count",
        "RecordLimit": str(page_size),
        "RecordOffset": "0",
    }

    frames = []
    offset = 0
    total = None
    while total is None or offset < total:
        params["RecordOffset"] = str(offset)
        url = f"{NOMIS_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as response:
            body = response.read()
        page = pl.read_csv(io.BytesIO(body))
        frames.append(page)
        total = int(page["RECORD_COUNT"][0])
        offset += page.height

    counts = pl.concat(frames).select(
        pl.col("GEOGRAPHY_CODE").alias("lsoa_code"),
        pl.col("C2021_AGE_19_NAME").alias("age_band"),
        pl.col("OBS_VALUE").alias("population"),
    )
    return counts


def fetch_lsoa_to_constituency_lookup(page_size: int = 1000) -> pl.DataFrame:
    """Download the ONS LSOA (2021) -> Westminster constituency (2024) lookup."""
    params = {
        "where": "1=1",
        "outFields": "LSOA21CD,PCON24CD,PCON24NM",
        "f": "json",
        "resultRecordCount": str(page_size),
        "resultOffset": "0",
    }

    rows = []
    offset = 0
    while True:
        params["resultOffset"] = str(offset)
        url = f"{LOOKUP_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as response:
            body = json.load(response)
        features = body["features"]
        rows.extend(f["attributes"] for f in features)
        if len(features) < page_size:
            break
        offset += page_size

    lookup = pl.DataFrame(rows).select(
        pl.col("LSOA21CD").alias("lsoa_code"),
        pl.col("PCON24CD").alias("pcon_code"),
        pl.col("PCON24NM").alias("pcon_name"),
    )
    return lookup


def build_frame(lsoa_counts: pl.DataFrame, lookup: pl.DataFrame) -> pl.DataFrame:
    """Reshape LSOA-level age counts into (constituency, age band) cells.

    Pure aggregation, no I/O: sums each LSOA's population into the
    constituency it maps to in `lookup`.
    """
    lsoa_codes = lsoa_counts.get_column("lsoa_code").unique().to_list()
    lookup_codes = set(lookup.get_column("lsoa_code").to_list())
    missing = [code for code in lsoa_codes if code not in lookup_codes]
    if missing:
        raise ValueError(f"lookup is missing constituency mappings for LSOAs: {sorted(missing)}")

    return (
        lsoa_counts.join(lookup, on="lsoa_code", how="left")
        .group_by("pcon_code", "pcon_name", "age_band")
        .agg(pl.col("population").sum())
    )


def collapse_to_voting_bands(frame: pl.DataFrame) -> pl.DataFrame:
    """Collapse Census five-year age bands into the four voting bands
    (18-24, 25-49, 50-64, 65+) that UK pollsters publish crosstabs against.

    Bands under 15 are dropped entirely — children are not electors and must
    not sit in the vote-share denominator. "Aged 15 to 19 years" is split
    using FIFTEEN_TO_NINETEEN_ADULT_FRACTION (see its docstring for the
    uniformity assumption this relies on); every other contributing band
    counts in full.

    Pure aggregation, no I/O: belongs to the batch half. Never call this on
    `data/frame.parquet` in place — write its result to a separate artifact,
    since the engine only ever reads the collapsed one and CLAUDE.md requires
    raw census data to stay untouched.
    """
    present_bands = set(frame.get_column("age_band").unique().to_list())
    expected_bands = set(CENSUS_TO_VOTING_BAND)
    missing = expected_bands - present_bands
    if missing:
        raise ValueError(f"frame is missing expected census age bands: {sorted(missing)}")

    raw_total = frame.get_column("population").sum()

    collapsed = (
        frame.filter(~pl.col("age_band").is_in(UNDER_15_CENSUS_BANDS))
        .with_columns(
            pl.col("age_band")
            .replace_strict({band: target for band, (target, _) in CENSUS_TO_VOTING_BAND.items()})
            .alias("voting_band"),
            pl.col("age_band")
            .replace_strict(
                {band: fraction for band, (_, fraction) in CENSUS_TO_VOTING_BAND.items()},
                return_dtype=pl.Float64,
            )
            .alias("elector_fraction"),
        )
        .with_columns((pl.col("population") * pl.col("elector_fraction")).alias("population"))
        .group_by("pcon_code", "pcon_name", "voting_band")
        .agg(pl.col("population").sum())
        .rename({"voting_band": "age_band"})
    )

    collapsed_total = collapsed.get_column("population").sum()
    assert collapsed_total < raw_total, (
        f"collapsed population ({collapsed_total}) is not lower than raw ({raw_total}); "
        "expected removing under-15 bands to reduce the total"
    )

    present_voting_bands = set(collapsed.get_column("age_band").unique().to_list())
    empty_bands = set(VOTING_BANDS) - present_voting_bands
    assert not empty_bands, f"voting bands ended up empty after collapsing: {sorted(empty_bands)}"

    return collapsed


def main() -> None:
    lsoa_counts = fetch_lsoa_age_counts()
    lookup = fetch_lsoa_to_constituency_lookup()
    frame = build_frame(lsoa_counts, lookup)

    FRAME_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(FRAME_ARTIFACT_PATH)

    n_constituencies = frame.get_column("pcon_code").n_unique()
    print(
        f"Wrote {frame.height} cells ({n_constituencies} constituencies) to {FRAME_ARTIFACT_PATH}"
    )
    print(
        f"England & Wales only ({n_constituencies} of 650 UK constituencies) — "
        "Scotland and Northern Ireland are not yet covered; national seat totals are invalid."
    )

    voting_bands_frame = collapse_to_voting_bands(frame)
    voting_bands_frame.write_parquet(VOTING_BANDS_ARTIFACT_PATH)
    print(f"Wrote {voting_bands_frame.height} cells to {VOTING_BANDS_ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
