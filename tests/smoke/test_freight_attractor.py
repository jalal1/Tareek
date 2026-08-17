"""Smoke tests for sector employment as the freight attractor (design.md item 12).

Total employment cannot tell a distribution centre from an insurance office of
the same headcount, but they attract entirely different amounts of truck
traffic. LODES publishes employment by NAICS sector in the *same* WAC file as
the C000 total, and the ETL used to discard those columns.

The tests that matter here are the discrimination (a warehouse must outweigh an
office) and the degradation (a database that predates the column must still run,
loudly, rather than silently sending every truck to a zero-attractor zone).

No network access: sector frames are built inline.
"""

import pandas as pd
import pytest

from models.freight.plans import (
    ATTRACTOR_EMPLOYMENT,
    ATTRACTOR_FREIGHT_EMPLOYMENT,
    VALID_ATTRACTORS,
    load_zones,
)
from models.freight.generator import FreightGenerationError
from models.work_locs_v2 import (
    ALL_CNS_COLUMNS,
    FREIGHT_SECTORS,
    freight_employment,
)


class _FakeConverter:
    def latlon_to_utm(self, lat, lon):
        return lon * 100_000.0, lat * 100_000.0

    def utm_to_latlon(self, x, y):
        return y / 100_000.0, x / 100_000.0


def _frame(rows):
    """A WAC-shaped frame with every CNS column present."""
    frame = pd.DataFrame(rows)
    for column in ALL_CNS_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0
    return frame


# ---------------------------------------------------------------------------
# the sector weighting
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_warehouse_outweighs_office_at_equal_headcount():
    """The whole point of item 12, as one assertion.

    CNS09 is transport/warehousing; CNS10 is finance. Same 100 jobs, and the
    attractor must not treat them alike.
    """
    frame = _frame([{'C000': 100, 'CNS09': 100},
                    {'C000': 100, 'CNS10': 100}])
    values = freight_employment(frame)
    assert values[0] > 0
    assert values[1] == 0


@pytest.mark.smoke
def test_weights_are_applied_per_sector():
    """Manufacturing 1.0, retail 0.5 — a weighted sum, not a headcount."""
    frame = _frame([{'C000': 100, 'CNS06': 50, 'CNS08': 50}])
    assert freight_employment(frame)[0] == pytest.approx(50 * 1.0 + 50 * 0.5)


@pytest.mark.smoke
def test_non_freight_sectors_contribute_nothing():
    """Finance, education, public administration draw no trucks."""
    frame = _frame([{'C000': 300, 'CNS10': 100, 'CNS15': 100, 'CNS20': 100}])
    assert freight_employment(frame)[0] == 0


@pytest.mark.smoke
def test_freight_sectors_are_a_strict_subset_of_what_lodes_publishes():
    """A typo in a sector key would silently drop that sector's contribution."""
    assert set(FREIGHT_SECTORS) < set(ALL_CNS_COLUMNS)


@pytest.mark.smoke
def test_missing_sector_columns_degrade_to_a_partial_sum():
    """LODES column sets vary by year and state; a partial sum beats nothing."""
    frame = pd.DataFrame([{'C000': 100, 'CNS06': 40, 'CNS09': 10}])
    assert freight_employment(frame)[0] == pytest.approx(40 * 1.0 + 10 * 1.2)


@pytest.mark.smoke
def test_no_sector_columns_at_all_returns_zeros_not_an_exception():
    frame = pd.DataFrame([{'C000': 100}, {'C000': 50}])
    values = freight_employment(frame)
    assert list(values) == [0, 0]


@pytest.mark.smoke
def test_nulls_in_sector_columns_are_treated_as_zero():
    frame = _frame([{'C000': 100, 'CNS06': None, 'CNS09': 10}])
    assert freight_employment(frame)[0] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# attractor selection in load_zones
# ---------------------------------------------------------------------------

def _patch_work_locations(monkeypatch, records):
    import models.work_locs_v2 as module
    monkeypatch.setattr(module, 'load_work_locations_by_counties',
                        lambda config: records)


@pytest.mark.smoke
def test_freight_employment_attractor_is_used_when_present(monkeypatch):
    _patch_work_locations(monkeypatch, {
        'g1': {'lat': 33.5, 'lon': -86.8, 'n_employees': 100,
               'n_employees_freight': 90},
        'g2': {'lat': 33.6, 'lon': -86.9, 'n_employees': 100,
               'n_employees_freight': 5},
    })
    zones = load_zones(
        {'freight': {'internal_attractor': ATTRACTOR_FREIGHT_EMPLOYMENT}},
        _FakeConverter())
    assert sorted(z.attractor for z in zones) == [5.0, 90.0]


@pytest.mark.smoke
def test_employment_attractor_ignores_the_sector_column(monkeypatch):
    _patch_work_locations(monkeypatch, {
        'g1': {'lat': 33.5, 'lon': -86.8, 'n_employees': 100,
               'n_employees_freight': 5},
    })
    zones = load_zones(
        {'freight': {'internal_attractor': ATTRACTOR_EMPLOYMENT}},
        _FakeConverter())
    assert zones[0].attractor == 100.0


@pytest.mark.smoke
def test_missing_sector_data_falls_back_to_total_employment(monkeypatch):
    """A database written before the column existed must still run.

    The dangerous alternative is a zero attractor everywhere, which would
    distribute every truck by distance alone while reporting success.
    """
    _patch_work_locations(monkeypatch, {
        'g1': {'lat': 33.5, 'lon': -86.8, 'n_employees': 100,
               'n_employees_freight': None},
        'g2': {'lat': 33.6, 'lon': -86.9, 'n_employees': 40,
               'n_employees_freight': None},
    })
    zones = load_zones(
        {'freight': {'internal_attractor': ATTRACTOR_FREIGHT_EMPLOYMENT}},
        _FakeConverter())
    assert sorted(z.attractor for z in zones) == [40.0, 100.0]


@pytest.mark.smoke
def test_partial_sector_coverage_still_uses_the_sector_column(monkeypatch):
    """Some rows populated is not the same as none, and must not trigger the
    whole-dataset fallback."""
    _patch_work_locations(monkeypatch, {
        'g1': {'lat': 33.5, 'lon': -86.8, 'n_employees': 100,
               'n_employees_freight': 90},
        'g2': {'lat': 33.6, 'lon': -86.9, 'n_employees': 100,
               'n_employees_freight': None},
    })
    zones = load_zones(
        {'freight': {'internal_attractor': ATTRACTOR_FREIGHT_EMPLOYMENT}},
        _FakeConverter())
    assert sorted(z.attractor for z in zones) == [0.0, 90.0]


@pytest.mark.smoke
def test_unknown_attractor_is_rejected(monkeypatch):
    _patch_work_locations(monkeypatch, {})
    with pytest.raises(FreightGenerationError, match='internal_attractor'):
        load_zones({'freight': {'internal_attractor': 'vehicle_miles'}},
                   _FakeConverter())


@pytest.mark.smoke
def test_both_attractors_are_valid_values():
    assert ATTRACTOR_EMPLOYMENT in VALID_ATTRACTORS
    assert ATTRACTOR_FREIGHT_EMPLOYMENT in VALID_ATTRACTORS


@pytest.mark.smoke
def test_default_attractor_is_total_employment(monkeypatch):
    """Omitting the key must not silently change an existing region's demand."""
    _patch_work_locations(monkeypatch, {
        'g1': {'lat': 33.5, 'lon': -86.8, 'n_employees': 77,
               'n_employees_freight': 1},
    })
    zones = load_zones({'freight': {}}, _FakeConverter())
    assert zones[0].attractor == 77.0
