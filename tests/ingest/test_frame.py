import polars as pl
import pytest

from pollgrid.ingest.frame import (
    CENSUS_TO_VOTING_BAND,
    FIFTEEN_TO_NINETEEN_ADULT_FRACTION,
    VOTING_BANDS,
    build_frame,
    collapse_to_voting_bands,
)

# 3 LSOAs, 2 constituencies, 2 age bands:
#   lookup:      L1 -> PCON_A, L2 -> PCON_A, L3 -> PCON_B
#   lsoa_counts: each LSOA has a count for "18-24" and "65+"
# Expected sums:
#   PCON_A "18-24" = L1 + L2 = 100 + 200 = 300
#   PCON_A "65+"   = L1 + L2 =  10 +  20 =  30
#   PCON_B "18-24" = L3      = 300
#   PCON_B "65+"   = L3      =  30
LOOKUP = pl.DataFrame(
    {
        "lsoa_code": ["L1", "L2", "L3"],
        "pcon_code": ["PCON_A", "PCON_A", "PCON_B"],
        "pcon_name": ["Constituency A", "Constituency A", "Constituency B"],
    }
)

LSOA_COUNTS = pl.DataFrame(
    {
        "lsoa_code": ["L1", "L1", "L2", "L2", "L3", "L3"],
        "age_band": ["18-24", "65+", "18-24", "65+", "18-24", "65+"],
        "population": [100, 10, 200, 20, 300, 30],
    }
)


def test_build_frame_sums_lsoas_into_their_constituency():
    frame = build_frame(LSOA_COUNTS, LOOKUP)
    populations = dict(
        frame.select(
            pl.concat_str("pcon_code", "age_band", separator="/").alias("key"), "population"
        ).iter_rows()
    )

    assert populations == {
        "PCON_A/18-24": 300,
        "PCON_A/65+": 30,
        "PCON_B/18-24": 300,
        "PCON_B/65+": 30,
    }


def test_build_frame_rejects_lsoa_missing_from_lookup():
    lsoa_counts = pl.concat(
        [
            LSOA_COUNTS,
            pl.DataFrame({"lsoa_code": ["L4"], "age_band": ["18-24"], "population": [50]}),
        ]
    )
    with pytest.raises(ValueError, match="L4"):
        build_frame(lsoa_counts, LOOKUP)


# One constituency, every Census band the collapse needs plus the three
# under-15 bands it must drop. Chosen so the 0.4/0.6 apportionment of
# "Aged 15 to 19 years" is checkable by hand:
#   18-24 = 0.4*100 (15-19) + 50 (20-24)               = 90
#   25-49 = 10+10+10+10+10 (25-29 .. 45-49)            = 50
#   50-64 = 5+5+5 (50-54, 55-59, 60-64)                = 15
#   65+   = 3+3+2+1+1 (65-69 .. 85+)                   = 10
CENSUS_ROWS = {
    "Aged 4 years and under": 20,
    "Aged 5 to 9 years": 20,
    "Aged 10 to 14 years": 20,
    "Aged 15 to 19 years": 100,
    "Aged 20 to 24 years": 50,
    "Aged 25 to 29 years": 10,
    "Aged 30 to 34 years": 10,
    "Aged 35 to 39 years": 10,
    "Aged 40 to 44 years": 10,
    "Aged 45 to 49 years": 10,
    "Aged 50 to 54 years": 5,
    "Aged 55 to 59 years": 5,
    "Aged 60 to 64 years": 5,
    "Aged 65 to 69 years": 3,
    "Aged 70 to 74 years": 3,
    "Aged 75 to 79 years": 2,
    "Aged 80 to 84 years": 1,
    "Aged 85 years and over": 1,
}


def _census_frame(rows: dict[str, int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "pcon_code": ["PCON_A"] * len(rows),
            "pcon_name": ["Constituency A"] * len(rows),
            "age_band": list(rows.keys()),
            "population": list(rows.values()),
        }
    )


def test_collapse_to_voting_bands_apportions_15_to_19_band_by_hand():
    assert FIFTEEN_TO_NINETEEN_ADULT_FRACTION == pytest.approx(0.4)

    collapsed = collapse_to_voting_bands(_census_frame(CENSUS_ROWS))
    populations = dict(collapsed.select("age_band", "population").iter_rows())

    assert populations == pytest.approx({"18-24": 90.0, "25-49": 50.0, "50-64": 15.0, "65+": 10.0})


def test_collapse_to_voting_bands_excludes_under_15_bands():
    collapsed = collapse_to_voting_bands(_census_frame(CENSUS_ROWS))
    assert set(collapsed.get_column("age_band").unique().to_list()) == set(VOTING_BANDS)


def test_collapse_to_voting_bands_raises_when_census_band_missing():
    rows = {k: v for k, v in CENSUS_ROWS.items() if k != "Aged 65 to 69 years"}
    with pytest.raises(ValueError, match="Aged 65 to 69 years"):
        collapse_to_voting_bands(_census_frame(rows))


def test_collapse_to_voting_bands_covers_every_mapped_census_band():
    # Guards against the hand-built fixture and the production mapping drifting apart.
    assert set(CENSUS_ROWS) >= set(CENSUS_TO_VOTING_BAND)
