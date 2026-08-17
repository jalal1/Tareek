"""Smoke tests for tier-3 classification coverage and hourly-profile checks.

See docs/freight/design.md §5.
"""

import zipfile

import pytest

from models.freight.classification import (
    ClassificationCoverage,
    check_coverage,
    compare_hourly_profile,
    validate_tier3,
)
from models.freight.departure import BUSINESS_DAY_PROFILE, normalise_profile


def _station_zip(path, rows, state_abbr='AL', year=2024):
    header = 'record_type|state_code|station_id|travel_dir|num_classes|county_code'
    lines = [header]
    for index, (county, num_classes) in enumerate(rows):
        lines.append(f"S|01|{index:06d}|1|{num_classes}|{county}")
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr(f"{state_abbr}_{year} (TMAS).STA", '\n'.join(lines))
    return path


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_full_13_class_stations_are_counted():
    coverage = ClassificationCoverage()
    assert not coverage.available


@pytest.mark.smoke
def test_coverage_detects_full_classification(tmp_path):
    station_zip = _station_zip(tmp_path / 's.zip',
                               [('073', '13'), ('073', '13'), ('073', '03')])
    coverage = check_coverage(station_zip, 'AL', ['073'])

    assert coverage.available
    assert coverage.n_full == 2
    assert coverage.n_volume_only == 1
    assert coverage.n_stations == 3


@pytest.mark.smoke
def test_coverage_is_false_where_only_volume_is_reported(tmp_path):
    """FL/LA/OK-style regions: tiers 1 and 2 must still stand there."""
    station_zip = _station_zip(tmp_path / 's.zip', [('073', '03')] * 5)
    coverage = check_coverage(station_zip, 'AL', ['073'])

    assert not coverage.available
    assert coverage.n_volume_only == 5


@pytest.mark.smoke
def test_partial_class_bins_count_as_usable(tmp_path):
    station_zip = _station_zip(tmp_path / 's.zip', [('073', '06')] * 3)
    coverage = check_coverage(station_zip, 'AL', ['073'])

    assert coverage.available
    assert coverage.n_usable == 3
    assert coverage.n_full == 0


@pytest.mark.smoke
def test_coverage_filters_by_county(tmp_path):
    station_zip = _station_zip(tmp_path / 's.zip',
                               [('073', '13'), ('999', '13')])
    coverage = check_coverage(station_zip, 'AL', ['073'])
    assert coverage.n_stations == 1


@pytest.mark.smoke
def test_missing_station_file_reports_unavailable(tmp_path):
    coverage = check_coverage(tmp_path / 'nope.zip', 'AL', ['073'])
    assert not coverage.available
    assert coverage.n_stations == 0


@pytest.mark.smoke
def test_coverage_serialises(tmp_path):
    import json
    station_zip = _station_zip(tmp_path / 's.zip', [('073', '13')])
    payload = json.loads(json.dumps(check_coverage(station_zip, 'AL', ['073']).to_dict()))
    assert payload['available'] is True


# ---------------------------------------------------------------------------
# hourly profile comparison
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_identical_profiles_match_perfectly():
    profile = list(normalise_profile(BUSINESS_DAY_PROFILE))
    result = compare_hourly_profile(profile, profile)

    assert result['comparable']
    assert result['correlation'] == pytest.approx(1.0)
    assert result['peak_hour_offset'] == 0
    assert validate_tier3(result).passed


@pytest.mark.smoke
def test_a_level_difference_alone_does_not_fail():
    """Tier 3 tests *when*, not *how many* — that is tier 2's job."""
    profile = list(normalise_profile(BUSINESS_DAY_PROFILE))
    doubled = [v * 2 for v in profile]
    result = compare_hourly_profile(doubled, profile)

    assert result['correlation'] == pytest.approx(1.0)
    assert validate_tier3(result).passed


@pytest.mark.smoke
def test_a_shifted_profile_fails():
    profile = list(normalise_profile(BUSINESS_DAY_PROFILE))
    shifted = profile[4:] + profile[:4]
    result = compare_hourly_profile(shifted, profile)

    assert result['peak_hour_offset'] == 4
    assert not validate_tier3(result).passed


@pytest.mark.smoke
def test_empty_profiles_are_reported_not_crashed():
    result = compare_hourly_profile([0.0] * 24, [1.0] * 24)
    assert not result['comparable']
    assert not validate_tier3(result).passed


@pytest.mark.smoke
@pytest.mark.parametrize('bad', [[0.1] * 23, [0.1] * 25])
def test_wrong_length_profiles_rejected(bad):
    with pytest.raises(ValueError):
        compare_hourly_profile(bad, [0.1] * 24)


@pytest.mark.smoke
def test_tier3_reports_missing_data_as_a_failure_not_a_pass():
    """An unavailable tier must never silently look like a passing one."""
    report = validate_tier3({'comparable': False, 'note': 'no data'})
    assert not report.passed
