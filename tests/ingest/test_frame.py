import polars as pl
import pytest

from pollgrid.ingest.frame import build_frame

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
