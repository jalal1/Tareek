"""Smoke tests for the OD diagnostics helpers: pure, deterministic logic.

Covers the distance helper, the trip-length statistics and the matrix statistics
that every run reports — the fields that make two runs comparable.
"""

import json

import numpy as np
import pandas as pd
import pytest

from data_sources.lodes_od import FlowTotals, flows_to_zone_matrix, matrix_stats
from utils.od_diagnostics import (
    ODDiagnostics,
    cosine_corrected_km,
    trip_length_stats,
)


# --------------------------------------------------------------------------
# cosine_corrected_km
# --------------------------------------------------------------------------

@pytest.mark.smoke
def test_latitude_separation_is_cosine_free():
    """A pure north-south separation is unaffected by the cosine correction."""
    d = cosine_corrected_km(np.array([[-93.0, 45.0]]), np.array([[-93.0, 46.0]]))
    # 1 degree of latitude is ~111.32 km everywhere.
    assert d.shape == (1, 1)
    assert d[0, 0] == pytest.approx(111.32, rel=1e-6)


@pytest.mark.smoke
def test_longitude_separation_shrinks_with_latitude():
    """East-west degrees cover less ground the further from the equator.

    This is the defect the correction fixes: without it, a degree of longitude
    is treated as a full 111.32 km at every latitude, overstating east-west
    separation by 1/cos(lat) — about 1.41x at 45 degrees N.
    """
    at_45 = cosine_corrected_km(np.array([[-93.0, 45.0]]), np.array([[-92.0, 45.0]]))[0, 0]
    at_25 = cosine_corrected_km(np.array([[-93.0, 25.0]]), np.array([[-92.0, 25.0]]))[0, 0]

    assert at_45 == pytest.approx(111.32 * np.cos(np.radians(45.0)), rel=1e-6)
    assert at_25 == pytest.approx(111.32 * np.cos(np.radians(25.0)), rel=1e-6)
    # Overstatement factor if the correction were omitted.
    assert 111.32 / at_45 == pytest.approx(1.414, abs=0.01)


@pytest.mark.smoke
def test_zero_distance_to_self():
    pt = np.array([[-93.26, 44.97]])
    assert cosine_corrected_km(pt, pt)[0, 0] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# trip_length_stats
# --------------------------------------------------------------------------

@pytest.mark.smoke
def test_trip_length_stats_are_trip_weighted():
    """Percentiles describe the average trip, not the average zone pair.

    Zone A->A carries 90 trips at 0 km and A->B carries 10 at ~111 km, so a
    trip-weighted median must land at 0, not halfway between the two pairs.
    """
    matrix = pd.DataFrame([[90.0, 10.0]], index=["A"], columns=["A", "B"])
    coords_h = {"A": (-93.0, 45.0)}
    coords_w = {"A": (-93.0, 45.0), "B": (-93.0, 46.0)}

    stats = trip_length_stats(matrix, coords_h, coords_w)

    assert stats["median_km"] == pytest.approx(0.0)
    # mean = (90*0 + 10*111.32) / 100
    assert stats["mean_km"] == pytest.approx(11.13, abs=0.01)
    assert stats["intrazonal_share"] == pytest.approx(0.9)


@pytest.mark.smoke
def test_trip_length_stats_handles_empty_matrix():
    matrix = pd.DataFrame([[0.0]], index=["A"], columns=["A"])
    stats = trip_length_stats(matrix, {"A": (-93.0, 45.0)}, {"A": (-93.0, 45.0)})
    assert stats["median_km"] is None
    assert stats["intrazonal_share"] is None


# --------------------------------------------------------------------------
# FlowTotals — the three shares
# --------------------------------------------------------------------------

@pytest.mark.smoke
def test_internal_share_is_resident_side():
    """internal_share must be I-I / (I-I + I-E).

    That is the definition the design docs quote and the one that explains the
    E1 agent drop; the job-side and two-sided variants are reported separately
    so the three can never be confused for one another.
    """
    t = FlowTotals(internal_ii=1_773_395, outbound_ie=109_568, inbound_ei=185_300)

    assert t.internal_share == pytest.approx(0.9418, abs=1e-4)
    assert t.job_side_share == pytest.approx(0.9054, abs=1e-4)
    assert t.two_sided_share == pytest.approx(0.8574, abs=1e-4)


@pytest.mark.smoke
def test_flow_totals_degenerate_region():
    """No flows at all must not divide by zero."""
    t = FlowTotals()
    assert t.internal_share == 0.0
    assert t.job_side_share == 0.0
    assert t.two_sided_share == 0.0


# --------------------------------------------------------------------------
# flows_to_zone_matrix
# --------------------------------------------------------------------------

@pytest.mark.smoke
def test_flows_aggregate_to_zones_and_keep_empty_zones():
    """Blocks roll up to their zone prefix; zones without flow survive as zeros.

    A work zone present in the location tables but absent from the flows is the
    E2 reindex case — it must stay in the matrix as an explicit zero so both
    sources present the same zone universe downstream.
    """
    flows = pd.DataFrame({
        # Two blocks in the same home zone (first 12 chars shared).
        "h_geocode": ["270030501071000", "270030501071001", "270030501072000"],
        "w_geocode": ["270030501081000", "270030501081001", "270030501081000"],
        "jobs": [5, 7, 3],
    })

    m = flows_to_zone_matrix(
        flows,
        home_zones=["270030501071", "270030501072", "270030501073"],
        work_zones=["270030501081", "270030501082"],
    )

    assert m.shape == (3, 2)
    # The two blocks of zone ...071 collapse into one cell.
    assert m.at["270030501071", "270030501081"] == 12
    assert m.at["270030501072", "270030501081"] == 3
    # Zones with no observed flow survive as zeros rather than disappearing.
    assert m.at["270030501073", "270030501081"] == 0
    assert m["270030501082"].sum() == 0
    assert m.to_numpy().sum() == 15


@pytest.mark.smoke
def test_flows_to_zone_matrix_empty_input():
    m = flows_to_zone_matrix(pd.DataFrame(columns=["h_geocode", "w_geocode", "jobs"]),
                             home_zones=["A"], work_zones=["B"])
    assert m.shape == (1, 1)
    assert m.to_numpy().sum() == 0


# --------------------------------------------------------------------------
# matrix_stats
# --------------------------------------------------------------------------

@pytest.mark.smoke
def test_matrix_stats_counts_zero_margins_and_stranded_residents():
    """A zero-row means those residents generate no work trip at all."""
    m = pd.DataFrame(
        [[1.0, 0.0],
         [0.0, 0.0]],
        index=["A", "B"], columns=["X", "Y"],
    )

    stats = matrix_stats(m, home_residents={"A": 10, "B": 250},
                         work_jobs={"X": 4, "Y": 7})

    assert stats["rows"] == 2 and stats["cols"] == 2
    assert stats["nonzero_pairs"] == 1
    assert stats["density"] == pytest.approx(0.25)
    assert stats["total_trips"] == 1.0
    assert stats["zero_rows"] == 1
    assert stats["zero_cols"] == 1
    # Zone B holds 250 residents who never travel — the red flag.
    assert stats["residents_in_zero_rows"] == 250
    assert stats["jobs_in_zero_cols"] == 7


# --------------------------------------------------------------------------
# ODDiagnostics
# --------------------------------------------------------------------------

@pytest.mark.smoke
def test_diagnostics_roundtrip_is_json_serialisable(tmp_path):
    """Numpy scalars must survive the write — they are pervasive upstream."""
    d = ODDiagnostics(source_requested="auto", geo_level="block_group")
    d.set_source("gravity", fallback_reason="no LODES coverage for 2099")
    d.update(matrix={"rows": np.int64(10), "density": np.float64(0.5)})
    d.set_runtime("assemble", 1.234)
    d.set_comparison_to_gravity(gravity_total=1_900_761, actual_total=1_773_395)

    path = d.write(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["source_requested"] == "auto"
    assert payload["source_used"] == "gravity"
    assert payload["fallback_reason"] == "no LODES coverage for 2099"
    assert payload["matrix"]["rows"] == 10
    assert payload["runtime_seconds"]["assemble"] == 1.23
    # E1 — the agent delta both arms must report.
    assert payload["comparison_to_gravity_base"]["ratio"] == pytest.approx(0.933, abs=1e-3)
    assert payload["comparison_to_gravity_base"]["agent_delta_pct"] == pytest.approx(-6.7, abs=0.01)


@pytest.mark.smoke
def test_diagnostics_has_stable_shape_across_sources(tmp_path):
    """Blocks that do not apply stay present as null, so runs diff cleanly."""
    d = ODDiagnostics(source_requested="gravity", geo_level="block_group")
    payload = json.loads(d.write(tmp_path).read_text(encoding="utf-8"))

    for key in ("source_used", "fallback_reason", "lodes", "flows", "matrix",
                "comparison_to_gravity_base", "survey_blend", "trip_length",
                "boundary", "demand_coverage"):
        assert key in payload, f"{key} must be present even when unset"
