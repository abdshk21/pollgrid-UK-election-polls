import polars as pl
import pytest

from pollgrid.engine.bridge import frame_to_cells

# 2 constituencies x 2 age bands, 2 parties per band (each pair sums to 1.0
# so the cells_to_frame invariant is satisfiable downstream).
FRAME = pl.DataFrame(
    {
        "pcon_code": ["PCON_A", "PCON_A", "PCON_B", "PCON_B"],
        "pcon_name": ["Constituency A", "Constituency A", "Constituency B", "Constituency B"],
        "age_band": ["18-24", "65+", "18-24", "65+"],
        "population": [1000, 2000, 1500, 2500],
    }
)

SUPPORT_TABLE = {
    "18-24": {"Lab": (0.6, 0.05), "Con": (0.4, 0.05)},
    "65+": {"Lab": (0.3, 0.02), "Con": (0.7, 0.02)},
}


def test_frame_to_cells_produces_one_cell_per_constituency_age_band_party():
    cells = frame_to_cells(FRAME, SUPPORT_TABLE)
    # 2 constituencies x 2 age bands x 2 parties
    assert len(cells) == 8


def test_frame_to_cells_preserves_population_per_constituency_age_band():
    cells = frame_to_cells(FRAME, SUPPORT_TABLE)
    populations = {(c.region, c.age_band): c.population for c in cells}
    assert populations == {
        ("Constituency A", "18-24"): 1000.0,
        ("Constituency A", "65+"): 2000.0,
        ("Constituency B", "18-24"): 1500.0,
        ("Constituency B", "65+"): 2500.0,
    }


def test_frame_to_cells_raises_when_age_band_missing_from_support_table():
    support_table = {"18-24": SUPPORT_TABLE["18-24"]}  # no entry for "65+"
    with pytest.raises(ValueError, match="65"):
        frame_to_cells(FRAME, support_table)
