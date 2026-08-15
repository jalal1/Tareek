"""Smoke tests for the experiment report generator.

The report is read by people deciding what to change in a config, so a wrong
reading is worse than a missing one. These tests pin the two places where the
report draws a conclusion rather than printing a number:

  * demand grading, which must weigh trip LENGTH as well as trip COUNT — the
    two multiply into distance travelled, and an earlier version graded the
    count alone and reported demand as sound while both length ratios sat well
    outside their band; and
  * the network/reported distance correction, which makes the simulated and
    survey distance columns comparable in the first place.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import experiment_report as R  # noqa: E402


def _summary(**eval_over):
    """A minimal run summary; individual tests override single fields."""
    ev = {
        "aggregate_sim_obs_ratio": 0.949,
        "station_ratio_cv": 0.762,
        "geh_lt_5_pct": 20.2,
        "correlation": 0.78,
        "interquartile_mean_ratio": 0.890,
        "stations_over_simulated": 18,
        "stations_within_10pct": 3,
        "stations_under_simulated": 32,
        "station_ratio_p10": 0.45,
        "station_ratio_p90": 1.684,
        "num_devices": 36,
        "num_directional_counts": 53,
    }
    ev.update(eval_over)
    return {
        "created_at": "2026-08-13T21:12:17",
        "parameters": {"scaling_factor": 0.105},
        "matsim_output": {},
        "evaluation": ev,
    }


def _demand(tpp=0.96, median=1.49, mean=1.47):
    return {
        "primary_survey": "nhts",
        "surveys": {
            "nhts": {
                "label": "nhts 2022",
                "is_local": False,
                "survey_trips": 30462,
                "trips_per_person_survey": 2.82,
                "trips_per_person_simulated": 2.71,
                "trips_per_person_ratio": tpp,
                "survey_median_km": 6.93,
                "survey_mean_km": 13.70,
                "simulated_median_km": 10.32,
                "simulated_mean_km": 20.18,
                "median_km_ratio": median,
                "mean_km_ratio": mean,
            }
        },
    }


def _render(summary):
    run = {"summary": summary, "evaluation": summary["evaluation"],
           "dir": Path("run")}
    return R.build_markdown(run, None, Path("run"))


@pytest.mark.smoke
def test_trip_length_is_graded_not_only_trip_count():
    """Trip count inside its band must not certify demand on its own.

    The regression this pins: trips_per_person_ratio 0.96 passes, while both
    length ratios are far outside the band. The report must not claim demand
    is sound, and must show the length ratios being graded.
    """
    s = _summary()
    s["demand_validation"] = _demand(tpp=0.96, median=1.49, mean=1.47)
    md = _render(s)

    # The old wording declared the question settled. It must be gone.
    assert "Trip **generation** is about right" not in md
    assert "assignment or mode-choice question" not in md

    # Both length ratios are graded, and both land outside the band.
    assert "Median trip length" in md
    assert "Mean trip length" in md
    assert md.count("above the ±10% band") >= 2


@pytest.mark.smoke
def test_distance_travelled_combines_count_and_length():
    """Trips x length is reported, because that is what reaches the network."""
    s = _summary()
    s["demand_validation"] = _demand(tpp=0.96, median=1.49, mean=1.47)
    md = _render(s)
    # 0.96 * (1.47 / 1.25) = 1.129
    assert "Distance travelled per person: 1.13" in md


@pytest.mark.smoke
def test_detour_factor_is_applied_and_disablable():
    """The correction changes the graded value, and 1.0 turns it off."""
    s = _summary()
    s["demand_validation"] = _demand(median=1.49, mean=1.47)
    md = _render(s)
    # 1.49 / 1.25 = 1.192, shown alongside the raw figure.
    assert "1.49 raw, 1.19 after the distance correction" in md

    s["parameters"]["evaluation"] = {
        "report_tolerances": {"distance_detour_factor": 1.0}}
    off = _render(s)
    assert "distance correction" not in off
    assert "**Median trip length: 1.49**" in off


@pytest.mark.smoke
def test_length_within_band_after_correction_passes():
    """A run whose raw ratio only reflects the detour must not be flagged."""
    s = _summary()
    # 1.25 raw is exactly a correct-length trip once the detour is removed.
    s["demand_validation"] = _demand(tpp=1.0, median=1.25, mean=1.25)
    md = _render(s)
    assert "within the ±10% band" in md
    assert "above the ±10% band" not in md


@pytest.mark.smoke
def test_verdict_states_measurements_not_remedies():
    """No config lever is prescribed: those inferences do not travel regions."""
    md = _render(_summary())
    for phrase in ("would make the fit worse",
                   "Unmodelled demand is a plausible explanation",
                   "is a valid correction",
                   "Fix the spatial distribution first",
                   "departure-time question"):
        assert phrase not in md
    # The numbers themselves are still reported.
    assert "Total volume is 0.949x observed" in md
    assert "CV of 0.762" in md


@pytest.mark.smoke
def test_station_split_states_its_own_cutoffs():
    """The split counts use fixed 0.9/1.1, so the text must say so.

    Echoing the configurable volume band here would misdescribe the numbers in
    any region that widened it.
    """
    md = _render(_summary())
    assert "18 are above 1.1" in md
    assert "3 are between 0.9 and 1.1" in md
    assert "32 are below 0.9" in md


@pytest.mark.smoke
def test_no_duplicate_or_figure_covered_rows():
    """Supporting numbers carry only what is not already stated or drawn."""
    md = _render(_summary(hourly_ratio_spread=2.046, worst_hour=23,
                          worst_hour_ratio=2.05,
                          station_daily_geh_lt5_pct=1.9,
                          station_daily_geh_lt10_pct=1.9))
    # Repeated the headline metric under a second name.
    assert "Aggregate sim/obs" not in md
    # The hourly figures and the time-of-day table already show the profile.
    assert "Hourly ratio spread" not in md
    assert "Worst single hour" not in md
    # Strictly weaker than the <5 row printed beside it.
    assert "daily GEH < 10" not in md
    assert "% stations with daily GEH < 5" in md


@pytest.mark.smoke
def test_glossary_defines_every_short_name_used():
    md = _render(_summary())
    for term in ("sim/obs", "GEH", "CV", "iqr_mean", "p10 / p90", "MAE / RMSE"):
        assert f"<strong>{term}</strong>" in md


@pytest.mark.smoke
def test_survey_acronym_expanded_once_without_stutter():
    s = _summary()
    s["demand_validation"] = _demand()
    md = _render(s)
    assert "<strong>NHTS</strong> — The National Household Travel Survey" in md
    assert "NHTS — NHTS" not in md


@pytest.mark.smoke
def test_missing_metrics_do_not_raise():
    """A run that recorded nothing still produces a report."""
    md = R.build_markdown(
        {"summary": {"parameters": {}, "matsim_output": {}, "evaluation": {}},
         "evaluation": {}, "dir": Path("run")}, None, Path("run"))
    assert "# Experiment Report" in md
    assert "Demand validation" not in md


@pytest.mark.smoke
def test_retired_tolerances_in_config_are_ignored():
    """An old config must not resurrect a threshold the report stopped using."""
    tol = R.load_tolerances({"parameters": {"evaluation": {"report_tolerances": {
        "station_daily_geh_max": 50.0,
        "hourly_spread_max": 1.0,
        "volume_ratio_band": 0.15,
    }}}})
    assert "station_daily_geh_max" not in tol
    assert "hourly_spread_max" not in tol
    assert tol["volume_ratio_band"] == 0.15  # live keys still apply
