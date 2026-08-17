"""Smoke tests for freight reporting metrics.

Each metric here is one that travel-model validation practice expects from a
model carrying more than one vehicle class, and that the existing evaluator
(total link volume only) cannot produce.

See docs/freight/design.md §5.
"""

import numpy as np
import pytest

from models.freight.cordons import Cordon, INBOUND, OUTBOUND
from models.freight.demand import CLASS_EXTERNAL_TO_INTERNAL, CLASS_THROUGH
from models.freight.events import LinkVolumes
from models.freight.generator import FreightTrip
from models.freight.reporting import (
    RMSE_TARGETS,
    VOLUME_GROUPS,
    build_report,
    cordon_screenline_check,
    hourly_class_shares,
    percent_rmse,
    rmse_by_volume_group,
    trip_length_distribution,
    truck_percentage_by_class,
    vmt_by_class,
)


def _streams():
    freight = LinkVolumes(volumes={'a': np.full(24, 10.0),
                                   'b': np.full(24, 5.0)})
    total = LinkVolumes(volumes={'a': np.full(24, 100.0),
                                 'b': np.full(24, 50.0)})
    car = LinkVolumes(volumes={'a': np.full(24, 90.0),
                               'b': np.full(24, 45.0)})
    return {'freight': freight, 'car': car, 'total': total}


# ---------------------------------------------------------------------------
# %RMSE
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_percent_rmse_is_zero_on_a_perfect_match():
    values = np.array([100.0, 200.0, 300.0])
    assert percent_rmse(values, values) == pytest.approx(0.0)


@pytest.mark.smoke
def test_percent_rmse_normalises_by_mean_observed():
    """What makes it comparable across volume groups and across studies."""
    simulated = np.array([110.0, 110.0])
    observed = np.array([100.0, 100.0])
    assert percent_rmse(simulated, observed) == pytest.approx(10.0)


@pytest.mark.smoke
def test_percent_rmse_handles_empty_and_zero():
    assert np.isnan(percent_rmse(np.array([]), np.array([])))
    assert np.isnan(percent_rmse(np.array([1.0]), np.array([0.0])))


@pytest.mark.smoke
def test_rmse_stratifies_by_volume_group():
    """A pooled figure hides that error is worse on low-volume links."""
    comparisons = [
        {'simulated': 1100.0, 'observed': 1000.0},     # under_5k
        {'simulated': 7000.0, 'observed': 7000.0},     # 5k_10k, perfect
        {'simulated': 60000.0, 'observed': 60000.0},   # over_50k, perfect
    ]
    result = rmse_by_volume_group(comparisons)

    assert set(result) == {'under_5k', '5k_10k', 'over_50k'}
    assert result['under_5k']['pct_rmse'] == pytest.approx(10.0)
    assert result['5k_10k']['pct_rmse'] == pytest.approx(0.0)
    assert result['5k_10k']['meets_target'] is True


@pytest.mark.smoke
def test_rmse_targets_loosen_for_low_volume_groups():
    """A few vehicles move the percentage on a quiet link."""
    assert RMSE_TARGETS['under_5k'] > RMSE_TARGETS['over_50k']
    assert len(VOLUME_GROUPS) == len(RMSE_TARGETS)


@pytest.mark.smoke
def test_rmse_flags_a_group_that_misses_its_target():
    comparisons = [{'simulated': 100_000.0, 'observed': 60_000.0}]
    result = rmse_by_volume_group(comparisons)
    assert result['over_50k']['meets_target'] is False


# ---------------------------------------------------------------------------
# VMT
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_vmt_by_class_computes_volume_times_length():
    streams = _streams()
    lengths = {'a': 1000.0, 'b': 2000.0}          # metres
    result = vmt_by_class(streams, lengths)

    # freight: 240 entries x 1 km + 120 x 2 km = 480 km
    assert result['freight']['vkt_km'] == pytest.approx(480.0)
    assert result['freight']['vmt_miles'] == pytest.approx(480.0 * 0.621371, rel=1e-3)


@pytest.mark.smoke
def test_vmt_reports_share_of_total():
    result = vmt_by_class(_streams(), {'a': 1000.0, 'b': 1000.0})
    assert result['freight']['share_of_total_pct'] == pytest.approx(10.0)


@pytest.mark.smoke
def test_vmt_counts_unmatched_links():
    result = vmt_by_class(_streams(), {'a': 1000.0})
    assert result['freight']['n_links_matched'] == 1
    assert result['freight']['n_links_total'] == 2


# ---------------------------------------------------------------------------
# truck percentage
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_truck_percentage_by_functional_class():
    """Closes the loop: demand is derived FROM a truck share."""
    streams = _streams()
    classes = {'a': 'interstate', 'b': 'arterial'}
    result = truck_percentage_by_class(streams, classes)

    assert result['interstate']['truck_pct'] == pytest.approx(10.0)
    assert result['arterial']['truck_pct'] == pytest.approx(10.0)


@pytest.mark.smoke
def test_truck_percentage_handles_unknown_class():
    result = truck_percentage_by_class(_streams(), {})
    assert 'unknown' in result


@pytest.mark.smoke
def test_truck_percentage_empty_without_streams():
    assert truck_percentage_by_class({}, {}) == {}


# ---------------------------------------------------------------------------
# trip length
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_trip_length_distribution_by_class():
    trips = [
        FreightTrip(CLASS_EXTERNAL_TO_INTERNAL, 45.0, -93.0, 45.1, -93.0, 3600.0),
        FreightTrip(CLASS_THROUGH, 45.0, -93.0, 45.5, -93.0, 3600.0),
    ]
    result = trip_length_distribution(trips)

    assert result[CLASS_EXTERNAL_TO_INTERNAL]['n_trips'] == 1
    assert result[CLASS_THROUGH]['mean_km'] > result[CLASS_EXTERNAL_TO_INTERNAL]['mean_km']
    assert result['all']['n_trips'] == 2


@pytest.mark.smoke
def test_trip_length_is_plausible_for_a_known_distance():
    """0.1 degree of latitude is ~11.1 km."""
    trips = [FreightTrip(CLASS_THROUGH, 45.0, -93.0, 45.1, -93.0, 0.0)]
    result = trip_length_distribution(trips)
    assert result['all']['mean_km'] == pytest.approx(11.1, abs=0.3)


@pytest.mark.smoke
def test_trip_length_handles_no_trips():
    assert trip_length_distribution([]) == {}


# ---------------------------------------------------------------------------
# screenline
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_screenline_passes_when_simulated_matches_expected():
    """For a boundary freight model the cordons ARE the screenline."""
    cordons = [Cordon('c1', 0, 0, INBOUND, 'N', ['a'], weight=240.0)]
    freight = LinkVolumes(volumes={'a': np.full(24, 10.0)})   # 240/day
    result = cordon_screenline_check(cordons, freight)

    assert result['n_within_tolerance'] == 1
    assert result['pct_within_tolerance'] == 100.0


@pytest.mark.smoke
def test_screenline_flags_a_cordon_that_carries_no_traffic():
    """Demand generated but never reaching the network."""
    cordons = [Cordon('c1', 0, 0, OUTBOUND, 'S', ['a'], weight=1000.0)]
    result = cordon_screenline_check(cordons, LinkVolumes())

    assert result['n_within_tolerance'] == 0
    assert result['worst'][0]['deviation_pct'] == pytest.approx(-100.0)


@pytest.mark.smoke
def test_screenline_tolerance_is_configurable():
    cordons = [Cordon('c1', 0, 0, INBOUND, 'N', ['a'], weight=100.0)]
    freight = LinkVolumes(volumes={'a': np.concatenate([[92.0], np.zeros(23)])})

    assert cordon_screenline_check(cordons, freight, 10.0)['n_within_tolerance'] == 1
    assert cordon_screenline_check(cordons, freight, 5.0)['n_within_tolerance'] == 0


@pytest.mark.smoke
def test_screenline_expectation_is_scaled_by_demand_scale():
    """Without this the check reports `demand_scale` back as an error.

    A cordon's weight is the *full* observed truck AADT, but a run at
    demand_scale=0.2 deliberately generates a fifth of it. Comparing the two
    directly makes a correctly-behaving model look 80% short — measured on
    Anoka the raw ratio was 0.281 against demand_scale 0.2 — and calibrating on
    that drives demand_scale to 1.0 no matter how good the model is.
    """
    cordons = [Cordon('c1', 0, 0, INBOUND, 'N', ['a'], weight=1000.0)]
    # The run generated 20% of observed, and all of it reached the cordon.
    freight = LinkVolumes(volumes={'a': np.concatenate([[200.0], np.zeros(23)])})

    unscaled = cordon_screenline_check(cordons, freight)
    assert unscaled['n_within_tolerance'] == 0, (
        "without demand_scale the deliberate reduction reads as an 80% error")

    scaled = cordon_screenline_check(cordons, freight, demand_scale=0.2)
    assert scaled['n_within_tolerance'] == 1
    assert scaled['total_expected'] == pytest.approx(200.0)


@pytest.mark.smoke
def test_screenline_still_detects_a_real_shortfall_when_scaled():
    """Scaling the expectation must not make the check unable to fail."""
    cordons = [Cordon('c1', 0, 0, INBOUND, 'N', ['a'], weight=1000.0)]
    freight = LinkVolumes(volumes={'a': np.concatenate([[50.0], np.zeros(23)])})

    result = cordon_screenline_check(cordons, freight, demand_scale=0.2)
    assert result['n_within_tolerance'] == 0
    assert result['worst'][0]['deviation_pct'] == pytest.approx(-75.0)


# ---------------------------------------------------------------------------
# hourly shares
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_hourly_shares_report_counts_alongside_share():
    """The share alone is the most misread number in truck validation."""
    result = hourly_class_shares(_streams())

    assert len(result['freight_counts']) == 24
    assert len(result['freight_share_pct']) == 24
    assert result['freight_share_pct'][0] == pytest.approx(10.0)
    assert 'caution' in result


@pytest.mark.smoke
def test_hourly_shares_handle_missing_streams():
    assert hourly_class_shares({}) == {}


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_build_report_assembles_available_metrics():
    import json

    streams = _streams()
    trips = [FreightTrip(CLASS_THROUGH, 45.0, -93.0, 45.2, -93.0, 3600.0)]
    cordons = [Cordon('c1', 0, 0, INBOUND, 'N', ['a'], weight=240.0)]

    report = build_report(
        streams, trips, cordons,
        link_lengths_m={'a': 1000.0, 'b': 1000.0},
        link_functional_class={'a': 'interstate', 'b': 'arterial'},
        observed_truck_aadt={'a': 240.0},
    )

    for key in ('hourly', 'trip_length', 'screenline', 'vmt',
                'truck_pct_by_functional_class', 'hpms_comparison',
                'rmse_by_volume_group'):
        assert key in report, key
    json.dumps(report)      # must survive the summary writer


@pytest.mark.smoke
def test_build_report_degrades_without_optional_inputs():
    """Network attributes and observed counts are not always available."""
    report = build_report(_streams(), [], [])

    assert 'hourly' in report
    assert 'vmt' not in report
    assert 'hpms_comparison' not in report


# ---------------------------------------------------------------------------
# network effect digest — what the experiment report shows
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_digest_separates_what_freight_did_from_what_was_generated():
    """The report's evidence half.

    Trip counts and truck share are inputs, knowable before MATSim runs. The
    digest carries only measured effects, so a reader cannot mistake a
    generated number for a validated one.
    """
    from models.freight.reporting import network_effect_digest

    streams = {
        'freight': LinkVolumes(volumes={'a': np.full(24, 10.0)}),
        'total': LinkVolumes(volumes={'a': np.full(24, 100.0)}),
    }
    report = {
        'vmt': {'freight': {'vmt_miles': 149.1, 'share_of_total_pct': 10.0}},
        'trip_length': {'all': {'mean_km': 28.9}},
        'screenline': {'total_expected': 240.0, 'total_simulated': 240.0,
                       'pct_within_tolerance': 100.0},
    }
    digest = network_effect_digest(report, streams)

    assert digest['link_entry_share_pct'] == 10.0
    assert digest['vmt_share_pct'] == 10.0
    assert digest['mean_trip_length_km'] == 28.9
    assert digest['screenline_ratio'] == 1.0


@pytest.mark.smoke
def test_digest_includes_tier2_when_observed_volumes_exist():
    """Tier 2 is the only figure judged against volumes the demand did not
    come from, so it must reach the report when available."""
    from models.freight.reporting import network_effect_digest

    streams = {'freight': LinkVolumes(), 'total': LinkVolumes()}
    report = {'hpms_comparison': {'n_links': 42, 'ratio': 0.95,
                                  'pct_geh_under_5': 61.0}}
    digest = network_effect_digest(report, streams)

    assert digest['hpms_n_links'] == 42
    assert digest['hpms_ratio'] == 0.95
    assert digest['hpms_pct_geh_under_5'] == 61.0


@pytest.mark.smoke
def test_digest_is_empty_rather_than_wrong_when_nothing_was_measured():
    """A run without an events file must report nothing, not zeros."""
    from models.freight.reporting import network_effect_digest

    assert network_effect_digest({}, {}) == {}
