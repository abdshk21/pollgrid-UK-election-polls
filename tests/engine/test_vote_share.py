import math

import pytest

from pollgrid.engine import (
    Cell,
    Scenario,
    apply_scenario_weights,
    cells_to_frame,
    national_vote_share,
)

# Two cells, two parties, chosen so every downstream number is hand-checkable:
#   weight_A = population_A * turnout_A = 1000 * 0.5 = 500
#   weight_B = population_B * turnout_B = 2000 * 0.25 = 500
#   total_weight = 1000
#   share_Lab = (500*0.6 + 500*0.3) / 1000 = 0.45
#   share_Con = (500*0.4 + 500*0.7) / 1000 = 0.55
#   variance  = weight**2 * se**2 summed per party = 500**2*0.05**2 + 500**2*0.02**2 = 725
#   se        = sqrt(725) / 1000
CELLS = [
    Cell(
        cell_id="A",
        age_band="18-24",
        region="Scotland",
        population=1000,
        party="Lab",
        support=0.6,
        se=0.05,
    ),
    Cell(
        cell_id="A",
        age_band="18-24",
        region="Scotland",
        population=1000,
        party="Con",
        support=0.4,
        se=0.05,
    ),
    Cell(
        cell_id="B",
        age_band="65+",
        region="England",
        population=2000,
        party="Lab",
        support=0.3,
        se=0.02,
    ),
    Cell(
        cell_id="B",
        age_band="65+",
        region="England",
        population=2000,
        party="Con",
        support=0.7,
        se=0.02,
    ),
]
SCENARIO = Scenario(turnout={"A": 0.5, "B": 0.25})


def test_apply_scenario_weights_computes_population_times_turnout():
    weighted = apply_scenario_weights(cells_to_frame(CELLS), SCENARIO)
    weights = dict(weighted.select("cell_id", "weight").unique().iter_rows())
    assert weights == {"A": 500.0, "B": 500.0}


def test_apply_scenario_weights_rejects_cell_missing_from_scenario():
    scenario = Scenario(turnout={"A": 0.5})  # no rate for cell B
    with pytest.raises(ValueError, match="B"):
        apply_scenario_weights(cells_to_frame(CELLS), scenario)


def test_national_vote_share_matches_hand_computed_values():
    weighted = apply_scenario_weights(cells_to_frame(CELLS), SCENARIO)
    toplines = {t.party: t for t in national_vote_share(weighted)}

    expected_se = math.sqrt(725) / 1000

    assert toplines["Lab"].share == pytest.approx(0.45)
    assert toplines["Con"].share == pytest.approx(0.55)
    assert toplines["Lab"].se == pytest.approx(expected_se)
    assert toplines["Con"].se == pytest.approx(expected_se)
    assert toplines["Lab"].ci_low == pytest.approx(0.45 - 1.959963984540054 * expected_se)
    assert toplines["Lab"].ci_high == pytest.approx(0.45 + 1.959963984540054 * expected_se)


def test_shares_sum_to_one():
    weighted = apply_scenario_weights(cells_to_frame(CELLS), SCENARIO)
    toplines = national_vote_share(weighted)
    assert sum(t.share for t in toplines) == pytest.approx(1.0)


def test_national_vote_share_rejects_zero_total_weight():
    scenario = Scenario(turnout={"A": 0.0, "B": 0.0})
    weighted = apply_scenario_weights(cells_to_frame(CELLS), scenario)
    with pytest.raises(ValueError, match="zero"):
        national_vote_share(weighted)


def test_scenario_rejects_turnout_outside_unit_interval():
    with pytest.raises(ValueError):
        Scenario(turnout={"A": 1.5})


def test_confidence_interval_clips_to_unit_interval_for_extreme_cell():
    tiny_sample_cell = [
        Cell(
            cell_id="C",
            age_band="18-24",
            region="Wales",
            population=100,
            party="X",
            support=0.95,
            se=0.3,  # implausibly large SE, standing in for a tiny effective sample
        )
    ]
    weighted = apply_scenario_weights(
        cells_to_frame(tiny_sample_cell), Scenario(turnout={"C": 1.0})
    )
    [topline] = national_vote_share(weighted)

    # weight = 100, variance = 100**2 * 0.3**2 = 900, se = sqrt(900)/100 = 0.3
    # ci_high = 0.95 + 1.96*0.3 = 1.538 -> clipped to 1.0
    assert topline.se == pytest.approx(0.3)
    assert topline.ci_high == 1.0
    assert topline.ci_low == pytest.approx(0.95 - 1.959963984540054 * 0.3)
