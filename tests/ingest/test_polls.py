import math
from datetime import date, timedelta

import polars as pl
import pytest

from pollgrid.ingest.polls import (
    CROSSTABS_PATH,
    POLLS_SCHEMA,
    PollTopline,
    crosstabs_to_support_table,
    house_effects,
    load_crosstabs,
    normalise_pollster,
    parse_pollbase,
    polls_to_frame,
)

# Mirrors britpol's raw `pollbase` columns exactly: id/start/end/pollster/n/con/lab/lib.
# start/end are days since 1970-01-01, chosen small so the converted dates are
# hand-checkable without re-deriving the epoch arithmetic.
#   poll-1: complete row, every field present.
#   poll-2: no sample size (n) and no "lib" share — partial party coverage.
#   poll-3: no pollster -> must be rejected.
#   poll-4: no start date -> must be rejected.
RAW = pl.DataFrame(
    {
        "id": ["poll-1", "poll-2", "poll-3", "poll-4"],
        "start": [0.0, 2.0, 4.0, None],
        "end": [1.0, 3.0, 5.0, 6.0],
        "pollster": ["Yougov", "Opinium", None, "ICM"],
        "n": [1500.0, None, 2000.0, 1000.0],
        "con": [0.40, 0.42, 0.50, 0.45],
        "lab": [0.35, 0.33, 0.30, 0.35],
        "lib": [0.10, None, 0.10, 0.10],
    },
    schema={
        "id": pl.Utf8,
        "start": pl.Float64,
        "end": pl.Float64,
        "pollster": pl.Utf8,
        "n": pl.Float64,
        "con": pl.Float64,
        "lab": pl.Float64,
        "lib": pl.Float64,
    },
)


def test_parse_pollbase_rejects_rows_missing_pollster_or_dates():
    polls = parse_pollbase(RAW)
    assert {p.poll_id for p in polls} == {"poll-1", "poll-2"}


def test_parse_pollbase_normalises_pollster_and_keeps_raw_string():
    [poll_1, _] = parse_pollbase(RAW)
    assert poll_1.pollster == "YouGov"
    assert poll_1.pollster_raw == "Yougov"


def test_parse_pollbase_converts_r_date_encoding():
    [poll_1, _] = parse_pollbase(RAW)
    assert poll_1.fieldwork_start == date(1970, 1, 1)
    assert poll_1.fieldwork_end == date(1970, 1, 2)


def test_parse_pollbase_keeps_sample_size_when_published():
    [poll_1, poll_2] = parse_pollbase(RAW)
    assert poll_1.sample_size == 1500
    assert poll_2.sample_size is None


def test_parse_pollbase_records_absent_parties_instead_of_inventing_shares():
    [_, poll_2] = parse_pollbase(RAW)
    assert poll_2.shares == pytest.approx({"Con": 0.42, "Lab": 0.33})
    assert poll_2.absent_parties == ["Green", "LD", "Reform"]


@pytest.mark.parametrize(
    ("raw_name", "canonical"),
    [
        ("Yougov", "YouGov"),
        ("yougov", "YouGov"),
        ("You Gov", "YouGov"),
        ("Ipsos Mori", "Ipsos MORI"),
        ("Some New Pollster", "Some New Pollster"),
    ],
)
def test_normalise_pollster(raw_name: str, canonical: str):
    assert normalise_pollster(raw_name) == canonical


def test_polls_to_frame_is_long_format_one_row_per_poll_party():
    polls = parse_pollbase(RAW)
    frame = polls_to_frame(polls)
    assert frame.schema == pl.Schema(POLLS_SCHEMA)
    # poll-1 reports 3 parties, poll-2 reports 2 -> 5 rows total.
    assert frame.height == 5
    poll_1_shares = dict(
        frame.filter(pl.col("poll_id") == "poll-1").select("party", "share").iter_rows()
    )
    assert poll_1_shares == pytest.approx({"Con": 0.40, "Lab": 0.35, "LD": 0.10})


# Hand-computable house-effect fixture: two pollsters, two parties, all within
# the window.
#   overall mean Con = (0.40+0.44+0.38+0.42)/4 = 0.41
#   overall mean Lab = (0.30+0.28+0.34+0.32)/4 = 0.31
#   house effect X: Con = avg(-0.01, +0.03) = +0.01, Lab = avg(-0.01,-0.03) = -0.02
#   house effect Y: Con = avg(-0.03, +0.01) = -0.01, Lab = avg(+0.03,+0.01) = +0.02
def _poll(poll_id: str, pollster: str, day: int, shares: dict[str, float]) -> PollTopline:
    return PollTopline(
        poll_id=poll_id,
        pollster=pollster,
        pollster_raw=pollster,
        fieldwork_start=date(1970, 1, 1) + timedelta(days=day),
        fieldwork_end=date(1970, 1, 1) + timedelta(days=day),
        sample_size=1000,
        shares=shares,
        absent_parties=sorted(set(("Con", "Lab", "LD", "Reform", "Green")) - set(shares)),
    )


HOUSE_EFFECT_POLLS = [
    _poll("x1", "X", 10, {"Con": 0.40, "Lab": 0.30}),
    _poll("x2", "X", 12, {"Con": 0.44, "Lab": 0.28}),
    _poll("y1", "Y", 11, {"Con": 0.38, "Lab": 0.34}),
    _poll("y2", "Y", 13, {"Con": 0.42, "Lab": 0.32}),
    # Far outside any reasonable window — must not affect the mean or appear in output.
    _poll("z1", "Z", -1000, {"Con": 0.90, "Lab": 0.05}),
]


def test_house_effects_matches_hand_computed_values():
    effects = house_effects(HOUSE_EFFECT_POLLS, window_days=30)

    assert effects["X"] == pytest.approx({"Con": 0.01, "Lab": -0.02})
    assert effects["Y"] == pytest.approx({"Con": -0.01, "Lab": 0.02})
    assert "Z" not in effects


def test_house_effects_returns_empty_dict_for_no_polls():
    assert house_effects([], window_days=30) == {}


# One pollster, two age bands, each group's support summing to 1.0 with Other.
VALID_CROSSTABS_YAML = """
- pollster: "YouGov"
  fieldwork_start: 2026-07-18
  fieldwork_end: 2026-07-20
  age_band: "18-24"
  party: "Lab"
  support: 0.30
  subsample_base: 150
- pollster: "YouGov"
  fieldwork_start: 2026-07-18
  fieldwork_end: 2026-07-20
  age_band: "18-24"
  party: "Con"
  support: 0.20
  subsample_base: 150
- pollster: "YouGov"
  fieldwork_start: 2026-07-18
  fieldwork_end: 2026-07-20
  age_band: "18-24"
  party: "Other"
  support: 0.50
  subsample_base: 150
- pollster: "YouGov"
  fieldwork_start: 2026-07-18
  fieldwork_end: 2026-07-20
  age_band: "65+"
  party: "Lab"
  support: 0.25
  subsample_base: 200
- pollster: "YouGov"
  fieldwork_start: 2026-07-18
  fieldwork_end: 2026-07-20
  age_band: "65+"
  party: "Other"
  support: 0.75
  subsample_base: 200
"""


def test_load_crosstabs_parses_valid_rows(tmp_path):
    path = tmp_path / "crosstabs.yaml"
    path.write_text(VALID_CROSSTABS_YAML)

    rows = load_crosstabs(path)

    assert len(rows) == 5
    assert {row.age_band for row in rows} == {"18-24", "65+"}
    assert {row.party for row in rows} == {"Lab", "Con", "Other"}


def test_load_crosstabs_rejects_age_band_outside_voting_bands(tmp_path):
    path = tmp_path / "crosstabs.yaml"
    path.write_text(
        """
- pollster: "YouGov"
  fieldwork_start: 2026-07-18
  fieldwork_end: 2026-07-20
  age_band: "35-44"
  party: "Lab"
  support: 1.0
  subsample_base: 100
"""
    )

    with pytest.raises(ValueError, match="35-44"):
        load_crosstabs(path)


def test_load_crosstabs_rejects_bands_not_summing_to_one(tmp_path):
    path = tmp_path / "crosstabs.yaml"
    path.write_text(
        """
- pollster: "YouGov"
  fieldwork_start: 2026-07-18
  fieldwork_end: 2026-07-20
  age_band: "18-24"
  party: "Lab"
  support: 0.30
  subsample_base: 150
- pollster: "YouGov"
  fieldwork_start: 2026-07-18
  fieldwork_end: 2026-07-20
  age_band: "18-24"
  party: "Con"
  support: 0.20
  subsample_base: 150
"""
    )

    with pytest.raises(ValueError, match="sum to 1.0"):
        load_crosstabs(path)


def test_load_crosstabs_returns_empty_list_for_empty_file(tmp_path):
    path = tmp_path / "crosstabs.yaml"
    path.write_text("# just comments, no rows yet\n")

    assert load_crosstabs(path) == []


def test_shipped_crosstabs_yaml_loads_without_error():
    assert load_crosstabs(CROSSTABS_PATH) == []


def test_crosstabs_to_support_table_shape_and_derived_se(tmp_path):
    path = tmp_path / "crosstabs.yaml"
    path.write_text(VALID_CROSSTABS_YAML)
    rows = load_crosstabs(path)

    table = crosstabs_to_support_table(rows, "YouGov")

    assert set(table) == {"18-24", "65+"}
    assert set(table["18-24"]) == {"Lab", "Con", "Other"}

    support, se = table["18-24"]["Lab"]
    assert support == pytest.approx(0.30)
    assert se == pytest.approx(math.sqrt(0.30 * 0.70 / 150))


def test_crosstabs_to_support_table_raises_for_unknown_pollster(tmp_path):
    path = tmp_path / "crosstabs.yaml"
    path.write_text(VALID_CROSSTABS_YAML)
    rows = load_crosstabs(path)

    with pytest.raises(ValueError, match="Opinium"):
        crosstabs_to_support_table(rows, "Opinium")
