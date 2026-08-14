"""Smoke tests for the stage-1 distance fixes in the gravity paths.

The load-bearing guarantee here is that ``legacy_degrees`` reproduces the
pre-stage-1 geometry exactly — ablation arm A depends on it — while
``cosine_km`` measures real kilometres and gives the diagonal an area-based
value instead of a centroid artefact.
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from models.od_matrix_v3 import (
    DISTANCE_COSINE_KM,
    DISTANCE_LEGACY_DEGREES,
    _apply_intrazonal_distance,
    create_gravity_model,
)
from utils.od_diagnostics import zone_sqrt_area_km


def _locs(zone_coords, employees):
    return {z: {"n_employees": employees[z], "centroid": c} for z, c in zone_coords.items()}


# Three zones spread across ~1 degree near 45 N, where the cosine correction
# matters most for the east-west axis.
_ZONES = {
    "A": (-93.0, 45.0),
    "B": (-92.0, 45.0),   # due east of A
    "C": (-93.0, 46.0),   # due north of A
}
_EMP = {"A": 100, "B": 100, "C": 100}


@pytest.mark.smoke
def test_legacy_mode_reproduces_original_geometry_exactly():
    """Arm A must stay reproducible after the fix lands.

    Replicates the pre-stage-1 code path verbatim — raw degrees, a guard that
    only replaces exact zeros with 0.1 — and requires a bit-for-bit match.
    """
    home = _locs(_ZONES, _EMP)
    work = _locs(_ZONES, _EMP)

    matrix, home_ids, work_ids = create_gravity_model(
        work, home, beta=1.5, max_iterations=200, convergence_threshold=0.03,
        distance_mode=DISTANCE_LEGACY_DEGREES,
    )

    # The original implementation, inlined.
    hc = np.array([home[g]["centroid"] for g in home_ids], dtype=float)
    wc = np.array([work[g]["centroid"] for g in work_ids], dtype=float)
    Oi = np.array([home[g]["n_employees"] for g in home_ids], dtype=float)
    Dj = np.array([work[g]["n_employees"] for g in work_ids], dtype=float)
    if abs(Oi.sum() - Dj.sum()) > 0.01:
        Dj = Dj * (Oi.sum() / Dj.sum())
    d = cdist(hc, wc, metric="euclidean")
    d = np.where(d == 0, 0.1, d)
    expected = np.power(d, -1.5)
    for _ in range(200):
        rs = expected.sum(axis=1, keepdims=True)
        expected = expected * np.divide(Oi.reshape(-1, 1), rs,
                                        out=np.ones_like(rs), where=rs != 0)
        cs = expected.sum(axis=0, keepdims=True)
        expected = expected * np.divide(Dj.reshape(1, -1), cs,
                                        out=np.ones_like(cs), where=cs != 0)
        if max((np.abs(expected.sum(1) - Oi) / (Oi + 1e-10)).max(),
               (np.abs(expected.sum(0) - Dj) / (Dj + 1e-10)).max()) < 0.03:
            break

    np.testing.assert_array_equal(matrix, expected)


@pytest.mark.smoke
def test_cosine_mode_differs_from_legacy():
    """The fix must actually change the geometry, not silently no-op."""
    home = _locs(_ZONES, _EMP)
    work = _locs(_ZONES, _EMP)
    kwargs = dict(beta=1.5, max_iterations=200, convergence_threshold=0.03)

    legacy, _, _ = create_gravity_model(work, home, distance_mode=DISTANCE_LEGACY_DEGREES, **kwargs)
    fixed, _, _ = create_gravity_model(
        work, home, distance_mode=DISTANCE_COSINE_KM,
        zone_sqrt_area={z: 1.0 for z in _ZONES}, intrazonal_factor=0.5, **kwargs)

    assert not np.allclose(legacy, fixed)


@pytest.mark.smoke
def test_cosine_mode_makes_east_west_pairs_closer_than_legacy_implies():
    """Anisotropy is the defect: a degree east is shorter than a degree north.

    A->B (1 degree east) and A->C (1 degree north) are equidistant in raw
    degrees, so legacy gives them the same friction. In real kilometres A->B is
    ~cos(45) = 0.71 of A->C, so it must attract strictly more trips.
    """
    home = _locs(_ZONES, _EMP)
    work = _locs(_ZONES, _EMP)
    kwargs = dict(beta=1.5, max_iterations=200, convergence_threshold=0.03)

    legacy, home_ids, work_ids = create_gravity_model(
        work, home, distance_mode=DISTANCE_LEGACY_DEGREES, **kwargs)
    fixed, _, _ = create_gravity_model(
        work, home, distance_mode=DISTANCE_COSINE_KM,
        zone_sqrt_area={z: 1.0 for z in _ZONES}, intrazonal_factor=0.5, **kwargs)

    ia, ib, ic = home_ids.index("A"), work_ids.index("B"), work_ids.index("C")

    # Legacy treats the two as identical; the fix does not.
    assert legacy[ia, ib] == pytest.approx(legacy[ia, ic], rel=1e-9)
    assert fixed[ia, ib] > fixed[ia, ic]


@pytest.mark.smoke
def test_unknown_distance_mode_is_rejected():
    home = _locs(_ZONES, _EMP)
    with pytest.raises(ValueError, match="distance_mode"):
        create_gravity_model(home, home, beta=1.5, distance_mode="furlongs")


# --------------------------------------------------------------------------
# _apply_intrazonal_distance
# --------------------------------------------------------------------------

@pytest.mark.smoke
def test_intrazonal_distance_comes_from_zone_area():
    """The diagonal is set from zone size, replacing the centroid artefact."""
    d = np.array([[0.001, 5.0],
                  [5.0, 0.002]])
    out = _apply_intrazonal_distance(
        d.copy(), home_geoids=["A", "B"], work_geoids=["A", "B"],
        zone_sqrt_area={"A": 2.0, "B": 4.0}, intrazonal_factor=0.5)

    assert out[0, 0] == pytest.approx(1.0)   # 0.5 * 2.0
    assert out[1, 1] == pytest.approx(2.0)   # 0.5 * 4.0
    # Off-diagonal pairs are untouched.
    assert out[0, 1] == pytest.approx(5.0)


@pytest.mark.smoke
def test_intrazonal_fix_floors_distances_so_friction_stays_finite():
    """A zero distance under a power law is infinite friction — never allow it."""
    d = np.array([[0.0]])
    out = _apply_intrazonal_distance(
        d.copy(), ["A"], ["A"], zone_sqrt_area={"A": 0.0}, intrazonal_factor=0.5)
    assert out[0, 0] >= 0.05
    assert np.isfinite(np.power(out, -1.5)).all()


@pytest.mark.smoke
def test_intrazonal_fix_without_area_leaves_matrix_usable():
    """Missing area data must degrade gracefully, not crash the run."""
    d = np.array([[0.001, 5.0], [5.0, 0.002]])
    out = _apply_intrazonal_distance(d.copy(), ["A", "B"], ["A", "B"],
                                     zone_sqrt_area=None, intrazonal_factor=0.5)
    np.testing.assert_array_equal(out, d)


@pytest.mark.smoke
def test_zone_not_present_as_destination_is_skipped():
    """A home zone with no matching work zone has no diagonal cell to fix."""
    d = np.array([[5.0], [6.0]])
    out = _apply_intrazonal_distance(d.copy(), ["A", "B"], ["C"],
                                     zone_sqrt_area={"A": 2.0}, intrazonal_factor=0.5)
    np.testing.assert_array_equal(out, d)


# --------------------------------------------------------------------------
# zone_sqrt_area_km
# --------------------------------------------------------------------------

@pytest.mark.smoke
def test_zone_area_grows_with_block_spread():
    """A zone whose blocks are spread further apart must measure larger."""
    tight = {"Z": [(-93.000, 45.000), (-93.002, 45.002)]}
    wide = {"Z": [(-93.00, 45.00), (-93.05, 45.05)]}

    assert zone_sqrt_area_km(wide)["Z"] > zone_sqrt_area_km(tight)["Z"]


@pytest.mark.smoke
def test_single_block_zone_falls_back_to_floor():
    """One block gives no spread — must still yield a positive distance."""
    out = zone_sqrt_area_km({"Z": [(-93.0, 45.0)]}, min_extent_km=0.05)
    assert out["Z"] == pytest.approx(0.05)


@pytest.mark.smoke
def test_empty_zone_falls_back_to_floor():
    out = zone_sqrt_area_km({"Z": []}, min_extent_km=0.05)
    assert out["Z"] == pytest.approx(0.05)
