"""PLACEHOLDER crosstab support figures — NOT model output.

There is no fitted multilevel regression yet (model/ is an empty stub).
Everything in this module is a single pollster's age-by-vote-intention
crosstab, hand-typed here as a stand-in for what batch/model/ will
eventually produce per cell. Replace PLACEHOLDER_SUPPORT with real
cell-level estimates the moment model/ exists — do not let this module
quietly become load-bearing.

The figures are plausible mid-2026 GB vote-intention numbers, not a
transcription of one named poll's exact published table.
"""

from math import sqrt

# Real published crosstabs report Lab/Con/LD/Reform/Green plus an "Other"
# residual — that's also where don't-know/won't-vote responses have already
# been excluded, per standard polling practice. We keep that residual as a
# named "Other" party rather than silently rescaling the five named parties
# up to sum to 1.0; the tolerance in cells_to_frame is satisfied honestly,
# not by inflating the parties we happen to name.
_PLACEHOLDER_SUPPORT_BY_BAND: dict[str, dict[str, float]] = {
    "18-24": {"Lab": 0.28, "Con": 0.10, "LD": 0.14, "Reform": 0.14, "Green": 0.24, "Other": 0.10},
    "25-49": {"Lab": 0.26, "Con": 0.14, "LD": 0.13, "Reform": 0.22, "Green": 0.15, "Other": 0.10},
    "50-64": {"Lab": 0.20, "Con": 0.19, "LD": 0.12, "Reform": 0.30, "Green": 0.09, "Other": 0.10},
    "65+": {"Lab": 0.16, "Con": 0.22, "LD": 0.11, "Reform": 0.36, "Green": 0.05, "Other": 0.10},
}

# Pollsters do not publish a standard error per crosstab cell. These
# per-band subsample sizes are a plausible base for a ~1300-respondent GB
# poll (younger bands are typically the smallest subsample in phone/online
# panels) — used only to *derive* an SE below via the standard
# sqrt(p*(1-p)/n) formula. They are not a published figure either.
PLACEHOLDER_SUBSAMPLE_N: dict[str, int] = {
    "18-24": 120,
    "25-49": 420,
    "50-64": 380,
    "65+": 380,
}


def _derive_se(support: float, n: int) -> float:
    """Standard error of a proportion from a plausible subsample size — derived, not published."""
    return sqrt(support * (1 - support) / n)


PLACEHOLDER_SUPPORT: dict[str, dict[str, tuple[float, float]]] = {
    age_band: {
        party: (support, _derive_se(support, PLACEHOLDER_SUBSAMPLE_N[age_band]))
        for party, support in parties.items()
    }
    for age_band, parties in _PLACEHOLDER_SUPPORT_BY_BAND.items()
}
