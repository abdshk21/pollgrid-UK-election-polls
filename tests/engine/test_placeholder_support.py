import polars as pl
import pytest

from pollgrid.engine.bridge import frame_to_cells
from pollgrid.engine.placeholder_support import PLACEHOLDER_SUPPORT
from pollgrid.engine.vote_share import cells_to_frame


def test_placeholder_support_sums_to_one_per_band():
    for age_band, parties in PLACEHOLDER_SUPPORT.items():
        total = sum(support for support, _ in parties.values())
        assert total == pytest.approx(1.0), age_band


def test_placeholder_support_is_consumable_by_the_engine():
    age_bands = list(PLACEHOLDER_SUPPORT)
    frame = pl.DataFrame(
        {
            "pcon_code": ["PCON_A"] * len(age_bands),
            "pcon_name": ["Constituency A"] * len(age_bands),
            "age_band": age_bands,
            "population": [1000] * len(age_bands),
        }
    )
    cells = frame_to_cells(frame, PLACEHOLDER_SUPPORT)
    cells_to_frame(cells)  # raises if any cell's support doesn't sum to 1.0
