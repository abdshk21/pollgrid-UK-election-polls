"""Streamlit UI for the turnout-scenario slider.

Thin by design: every number shown here comes from `engine/` (pure
functions, no I/O). This file only loads the cached frame, wires slider
input into a Scenario, and formats/plots the engine's output.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import streamlit as st

from pollgrid.engine import (
    Scenario,
    Topline,
    apply_scenario_weights,
    cells_to_frame,
    national_vote_share,
    topline_deltas,
)
from pollgrid.engine.bridge import frame_to_cells
from pollgrid.engine.placeholder_support import PLACEHOLDER_SUPPORT
from pollgrid.ingest.frame import VOTING_BANDS

FRAME_VOTING_BANDS_PATH = Path("data/frame_voting_bands.parquet")

# Estimates from post-election survey work (e.g. British Election Study),
# not an official figure — the UK does not publish turnout by age.
BASELINE_TURNOUT: dict[str, float] = {
    "18-24": 0.40,
    "25-49": 0.55,
    "50-64": 0.68,
    "65+": 0.76,
}

# Fixed order and real party colours so bars don't reshuffle or repaint as
# the scenario changes, and readers can use the UK's familiar party colours
# rather than an arbitrary categorical palette.
PARTY_ORDER = ["Lab", "Con", "LD", "Reform", "Green", "Other"]
PARTY_COLORS = {
    "Lab": "#E4003B",
    "Con": "#0087DC",
    "LD": "#FAA61A",
    "Reform": "#12B6CF",
    "Green": "#6AB023",
    "Other": "#898781",
}

COVERAGE_CONSTITUENCIES = 575
COVERAGE_TOTAL = 650


@st.cache_data
def load_voting_bands_frame() -> pl.DataFrame:
    return pl.read_parquet(FRAME_VOTING_BANDS_PATH)


@st.cache_data
def load_cells_frame() -> pl.DataFrame:
    """Build and validate the engine's cell table once per session.

    Everything downstream of this (apply_scenario_weights,
    national_vote_share) is cheap enough to re-run on every slider drag;
    this — frame_to_cells over ~2,300 constituency/age_band rows times six
    parties, plus the cells_to_frame sum-to-one check — is not, so it's
    cached and must not re-run on slider movement.
    """
    frame = load_voting_bands_frame()
    cells = frame_to_cells(frame, PLACEHOLDER_SUPPORT)
    return cells_to_frame(cells)


@st.cache_data
def compute_baseline_toplines() -> list[Topline]:
    cells_frame = load_cells_frame()
    weighted = apply_scenario_weights(cells_frame, Scenario(turnout=BASELINE_TURNOUT))
    return national_vote_share(weighted)


def render_topline_chart(toplines: list[Topline]) -> plt.Figure:
    by_party = {t.party: t for t in toplines}
    parties = [p for p in PARTY_ORDER if p in by_party]
    shares = [by_party[p].share * 100 for p in parties]
    err_low = [(by_party[p].share - by_party[p].ci_low) * 100 for p in parties]
    err_high = [(by_party[p].ci_high - by_party[p].share) * 100 for p in parties]
    colors = [PARTY_COLORS[p] for p in parties]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    y_pos = range(len(parties))
    ax.barh(list(y_pos), shares, color=colors, height=0.6, zorder=2)
    ax.errorbar(
        shares,
        list(y_pos),
        xerr=[err_low, err_high],
        fmt="none",
        ecolor="#0b0b0b",
        capsize=4,
        linewidth=1.3,
        zorder=3,
    )
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(parties)
    ax.invert_yaxis()
    ax.set_xlabel("National vote share (%)")
    ax.set_xlim(0, max(h + e for h, e in zip(shares, err_high, strict=True)) + 6)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for y, share in zip(y_pos, shares, strict=True):
        ax.text(share + 1.2, y, f"{share:.1f}%", va="center", fontsize=9, color="#52514e")
    fig.tight_layout()
    return fig


def render_delta_chart(deltas: dict[str, float]) -> plt.Figure:
    parties = [p for p in PARTY_ORDER if p in deltas]
    values = [deltas[p] * 100 for p in parties]
    colors = [PARTY_COLORS[p] for p in parties]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    y_pos = range(len(parties))
    ax.barh(list(y_pos), values, color=colors, height=0.6, zorder=2)
    ax.axvline(0, color="#c3c2b7", linewidth=1, zorder=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(parties)
    ax.invert_yaxis()
    ax.set_xlabel("Change vs default-turnout baseline (percentage points)")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    span = max(abs(v) for v in values) if values else 1.0
    ax.set_xlim(-span - 1, span + 1)
    for y, value in zip(y_pos, values, strict=True):
        ha = "left" if value >= 0 else "right"
        offset = span * 0.03 if value >= 0 else -span * 0.03
        ax.text(
            value + offset, y, f"{value:+.2f}pp", va="center", ha=ha, fontsize=9, color="#52514e"
        )
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="pollgrid — turnout scenario explorer", layout="wide")
    st.title("pollgrid — turnout scenario explorer")

    if not FRAME_VOTING_BANDS_PATH.exists():
        st.error(
            f"No poststratification frame found at `{FRAME_VOTING_BANDS_PATH}`. "
            "Run the ingest step first, from the repo root:\n\n"
            "```\nuv run python -m pollgrid.ingest.frame\n```"
        )
        st.stop()

    st.warning(
        "**Read this before the numbers below.**\n"
        f"- **Coverage:** {COVERAGE_CONSTITUENCIES} of {COVERAGE_TOTAL} UK constituencies "
        "(England & Wales only). Scotland and Northern Ireland are not yet ingested — "
        "this is not a UK-wide figure.\n"
        "- **Support is a placeholder:** vote-share numbers come from a single "
        "hand-typed crosstab, not a fitted model (see `engine/placeholder_support.py`). "
        "Every constituency currently shares the same support profile — there is no "
        "geographic variation yet.\n"
        "- **Uncertainty is understated:** the error bars are 95% confidence intervals "
        "from sampling error only. They do not capture house effects or model "
        "uncertainty, so true uncertainty is wider than shown."
    )

    st.sidebar.header("Turnout scenario")
    st.sidebar.caption(
        "Defaults are approximate 2024 GE turnout by age band — estimates from "
        "post-election survey work, not an official figure; the UK does not "
        "publish turnout by age."
    )
    turnout: dict[str, float] = {}
    for age_band in VOTING_BANDS:
        turnout[age_band] = (
            st.sidebar.slider(
                age_band,
                min_value=0,
                max_value=100,
                value=round(BASELINE_TURNOUT[age_band] * 100),
                step=1,
                format="%d%%",
            )
            / 100
        )

    cells_frame = load_cells_frame()
    weighted = apply_scenario_weights(cells_frame, Scenario(turnout=turnout))
    toplines = national_vote_share(weighted)
    baseline_toplines = compute_baseline_toplines()
    deltas = topline_deltas(toplines, baseline_toplines)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("National vote share")
        st.pyplot(render_topline_chart(toplines))
    with col2:
        st.subheader("What your toggle changed")
        st.pyplot(render_delta_chart(deltas))


if __name__ == "__main__":
    main()
