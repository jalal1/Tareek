"""Smoke tests for freight events extraction and validation tiers.

Uses synthetic events and plans files, so the tier stays dependency-free. The
event names here are the ones a real MATSim events file uses — ``entered link``,
lowercase with a space, not the Java class name — which was verified against a
real output_events.xml.gz rather than assumed.

See docs/freight/design.md §5.
"""

import gzip
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from models.freight.cordons import Cordon, INBOUND, OUTBOUND
from models.freight.demand import (
    CLASS_EXTERNAL_TO_INTERNAL,
    CLASS_INTERNAL_TO_EXTERNAL,
    CLASS_THROUGH,
    FreightDemand,
)
from models.freight.events import (
    EVENT_LINK_ENTER,
    EventsExtractionError,
    LinkVolumes,
    compare_against_observed,
    compare_by_segment,
    extract_freight_and_total,
    extract_link_volumes,
    freight_vehicle_ids,
    geh,
)
from models.freight.generator import FreightTrip
from models.freight.validation import validate_tier1, validate_tier2


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _write_events(path, events, gzipped=False):
    """Write a MATSim-shaped events file."""
    root = ET.Element('events', version='1.0')
    for event in events:
        ET.SubElement(root, 'event', {k: str(v) for k, v in event.items()})
    data = ET.tostring(root, encoding='unicode')
    if gzipped:
        with gzip.open(path, 'wt', encoding='utf-8') as handle:
            handle.write(data)
    else:
        path.write_text(data, encoding='utf-8')
    return path


def _write_plans(path, freight_ids=(), car_ids=()):
    """Write a plans file with a freight subpopulation."""
    root = ET.Element('population')
    for person_id in car_ids:
        ET.SubElement(root, 'person', id=person_id)
    for person_id in freight_ids:
        person = ET.SubElement(root, 'person', id=person_id)
        attributes = ET.SubElement(person, 'attributes')
        attribute = ET.SubElement(attributes, 'attribute',
                                  {'name': 'subpopulation',
                                   'class': 'java.lang.String'})
        attribute.text = 'freight'
    ET.ElementTree(root).write(str(path), encoding='utf-8', xml_declaration=True)
    return path


# ---------------------------------------------------------------------------
# vehicle identification
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_freight_ids_read_from_the_plans_file(tmp_path):
    """Read the IDs, never assume a naming convention.

    run_experiment renumbers every person before writing plans.xml, so any
    convention-based guess breaks.
    """
    plans = _write_plans(tmp_path / 'plans.xml',
                         freight_ids=['person_7', 'person_9'],
                         car_ids=['person_1', 'person_2'])
    assert freight_vehicle_ids(plans) == {'person_7', 'person_9'}


@pytest.mark.smoke
def test_freight_ids_empty_when_no_subpopulation(tmp_path):
    plans = _write_plans(tmp_path / 'plans.xml', car_ids=['person_1'])
    assert freight_vehicle_ids(plans) == set()


@pytest.mark.smoke
def test_missing_plans_file_raises(tmp_path):
    with pytest.raises(EventsExtractionError):
        freight_vehicle_ids(tmp_path / 'nope.xml')


# ---------------------------------------------------------------------------
# link volume extraction
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_link_entries_are_binned_by_hour(tmp_path):
    events = _write_events(tmp_path / 'events.xml', [
        {'time': 3600.0, 'type': EVENT_LINK_ENTER, 'link': 'A', 'vehicle': 'v1'},
        {'time': 3700.0, 'type': EVENT_LINK_ENTER, 'link': 'A', 'vehicle': 'v2'},
        {'time': 7300.0, 'type': EVENT_LINK_ENTER, 'link': 'A', 'vehicle': 'v1'},
    ])
    volumes = extract_link_volumes(events)

    assert volumes.hourly('A')[1] == 2
    assert volumes.hourly('A')[2] == 1
    assert volumes.daily('A') == 3
    assert volumes.n_vehicles == 2


@pytest.mark.smoke
def test_only_requested_vehicles_are_counted(tmp_path):
    events = _write_events(tmp_path / 'events.xml', [
        {'time': 0, 'type': EVENT_LINK_ENTER, 'link': 'A', 'vehicle': 'truck'},
        {'time': 0, 'type': EVENT_LINK_ENTER, 'link': 'A', 'vehicle': 'car'},
    ])
    volumes = extract_link_volumes(events, vehicle_ids={'truck'})

    assert volumes.daily('A') == 1
    assert volumes.n_vehicles == 1


@pytest.mark.smoke
def test_transit_vehicles_are_excluded(tmp_path):
    """Buses emit the same link-enter event.

    Measured on a real run: transit was 35% of all link entries. Counting it
    into the car stream would corrupt the freight/car split.
    """
    events = _write_events(tmp_path / 'events.xml', [
        {'time': 0, 'type': 'TransitDriverStarts', 'vehicleId': 'bus1',
         'driverId': 'd1', 'transitLineId': 'L1', 'transitRouteId': 'R1',
         'departureId': 'D1'},
        {'time': 10, 'type': EVENT_LINK_ENTER, 'link': 'A', 'vehicle': 'bus1'},
        {'time': 20, 'type': EVENT_LINK_ENTER, 'link': 'A', 'vehicle': 'car1'},
    ])

    assert extract_link_volumes(events).daily('A') == 1
    assert extract_link_volumes(events, exclude_transit=False).daily('A') == 2


@pytest.mark.smoke
def test_gzipped_events_are_read(tmp_path):
    events = _write_events(tmp_path / 'events.xml.gz', [
        {'time': 0, 'type': EVENT_LINK_ENTER, 'link': 'A', 'vehicle': 'v1'},
    ], gzipped=True)
    assert extract_link_volumes(events).daily('A') == 1


@pytest.mark.smoke
def test_times_past_midnight_are_kept_but_held_out_of_the_profile(tmp_path):
    """Post-midnight traffic is real volume, but it is not hour-23 traffic.

    Folding it into hour 23 conserves the daily total while inventing a
    late-night peak: measured on a real run, hour 23 read 5.7% against hour
    22's 2.7% purely from the fold. Tier 3 compares the shape of the day, so
    that spike would look like a timing error in the model.
    """
    events = _write_events(tmp_path / 'events.xml', [
        {'time': 26 * 3600, 'type': EVENT_LINK_ENTER, 'link': 'A', 'vehicle': 'v1'},
        {'time': 22 * 3600, 'type': EVENT_LINK_ENTER, 'link': 'A', 'vehicle': 'v2'},
    ])
    volumes = extract_link_volumes(events)

    assert volumes.daily('A') == 2          # nothing dropped
    assert volumes.hourly('A')[23] == 0     # no fabricated peak
    assert volumes.hourly('A')[22] == 1
    assert volumes.overflow_total() == 1


@pytest.mark.smoke
def test_overflow_scales_with_the_rest(tmp_path):
    events = _write_events(tmp_path / 'events.xml', [
        {'time': 25 * 3600, 'type': EVENT_LINK_ENTER, 'link': 'A', 'vehicle': 'v1'},
    ])
    scaled = extract_link_volumes(events).scaled(10.0)
    assert scaled.daily('A') == pytest.approx(10.0)


@pytest.mark.smoke
def test_freight_and_car_overflow_also_difference(tmp_path):
    """The freight/car split must hold after midnight too."""
    events = _write_events(tmp_path / 'events.xml', [
        {'time': 25 * 3600, 'type': EVENT_LINK_ENTER, 'link': 'A', 'vehicle': 'truck1'},
        {'time': 25 * 3600, 'type': EVENT_LINK_ENTER, 'link': 'A', 'vehicle': 'car1'},
    ])
    plans = _write_plans(tmp_path / 'plans.xml', freight_ids=['truck1'],
                         car_ids=['car1'])
    streams = extract_freight_and_total(events, plans)

    assert streams['freight'].daily('A') == 1
    assert streams['car'].daily('A') == 1
    assert (streams['freight'].total() + streams['car'].total()
            == streams['total'].total())


@pytest.mark.smoke
def test_missing_events_file_raises_with_guidance(tmp_path):
    with pytest.raises(EventsExtractionError, match='writeEventsInterval'):
        extract_link_volumes(tmp_path / 'nope.xml.gz')


@pytest.mark.smoke
def test_freight_and_car_streams_sum_to_the_total(tmp_path):
    events = _write_events(tmp_path / 'events.xml', [
        {'time': 0, 'type': EVENT_LINK_ENTER, 'link': 'A', 'vehicle': 'truck1'},
        {'time': 0, 'type': EVENT_LINK_ENTER, 'link': 'A', 'vehicle': 'car1'},
        {'time': 0, 'type': EVENT_LINK_ENTER, 'link': 'B', 'vehicle': 'car2'},
    ])
    plans = _write_plans(tmp_path / 'plans.xml', freight_ids=['truck1'],
                         car_ids=['car1', 'car2'])

    streams = extract_freight_and_total(events, plans)

    assert streams['freight'].daily('A') == 1
    assert streams['car'].daily('A') == 1
    assert streams['total'].daily('A') == 2
    assert (streams['freight'].total() + streams['car'].total()
            == streams['total'].total())


@pytest.mark.smoke
def test_scale_factor_lifts_the_sample_to_real_world(tmp_path):
    """A 10% sample compared against AADT unscaled understates 10x."""
    events = _write_events(tmp_path / 'events.xml', [
        {'time': 0, 'type': EVENT_LINK_ENTER, 'link': 'A', 'vehicle': 'truck1'},
    ])
    plans = _write_plans(tmp_path / 'plans.xml', freight_ids=['truck1'])

    streams = extract_freight_and_total(events, plans, scale_factor=10.0)
    assert streams['freight'].daily('A') == pytest.approx(10.0)


@pytest.mark.smoke
def test_dataframe_matches_the_linkstats_shape(tmp_path):
    """Same columns as evaluator.load_linkstats, so matching code is reusable."""
    events = _write_events(tmp_path / 'events.xml', [
        {'time': 0, 'type': EVENT_LINK_ENTER, 'link': 'A', 'vehicle': 'v1'},
    ])
    frame = extract_link_volumes(events).to_dataframe()

    assert list(frame.columns) == ['link_id'] + [f'HRS{h}-{h+1}avg'
                                                 for h in range(24)]


# ---------------------------------------------------------------------------
# GEH
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_geh_is_zero_on_a_perfect_match():
    assert geh(100, 100) == pytest.approx(0.0)


@pytest.mark.smoke
def test_geh_tolerates_relative_error_more_at_low_volumes():
    """The reason GEH is used instead of a ratio."""
    small = geh(15, 10)      # 50% high on a small number
    large = geh(15_000, 10_000)   # 50% high on a large one
    assert small < large


@pytest.mark.smoke
def test_geh_handles_zero_volumes():
    assert geh(0, 0) == 0.0
    assert geh(0, 100) > 0


@pytest.mark.smoke
def test_comparison_reports_aggregate_ratio_and_geh():
    volumes = LinkVolumes(volumes={'A': np.full(24, 10.0)})   # 240/day
    result = compare_against_observed(volumes, {'A': 240.0})

    assert result['n_links'] == 1
    assert result['ratio'] == pytest.approx(1.0)
    assert result['pct_geh_under_5'] == 100.0


@pytest.mark.smoke
def test_comparison_with_no_observed_links_is_reported_not_crashed():
    assert compare_against_observed(LinkVolumes(), {})['n_links'] == 0


# ---------------------------------------------------------------------------
# tier 1 validation
# ---------------------------------------------------------------------------

def _trips_and_demand(n_each=200, seed=1):
    from models.freight.departure import DepartureSampler
    rng = np.random.default_rng(seed)
    sampler = DepartureSampler()
    trips = []
    for trip_class, origin, dest in (
            (CLASS_EXTERNAL_TO_INTERNAL, 'in_a', None),
            (CLASS_INTERNAL_TO_EXTERNAL, None, 'out_a'),
            (CLASS_THROUGH, 'in_a', 'out_a')):
        departures = sampler.sample(trip_class, rng, n_each)
        for departure in departures:
            trips.append(FreightTrip(
                trip_class=trip_class,
                origin_lat=33.5, origin_lon=-86.8,
                dest_lat=33.6, dest_lon=-86.7,
                departure_seconds=float(departure),
                origin_cordon=origin, dest_cordon=dest,
                origin_zone=None if origin else 'z1',
                dest_zone=None if dest else 'z1',
            ))
    demand = FreightDemand(
        total_trips=3 * n_each,
        trips_by_class={CLASS_EXTERNAL_TO_INTERNAL: n_each,
                        CLASS_INTERNAL_TO_EXTERNAL: n_each,
                        CLASS_THROUGH: n_each},
        source='hpms_cordon',
        detail={'class_shares': {CLASS_EXTERNAL_TO_INTERNAL: 1 / 3,
                                 CLASS_INTERNAL_TO_EXTERNAL: 1 / 3,
                                 CLASS_THROUGH: 1 / 3}},
    )
    cordons = [
        Cordon('in_a', 0, 10_000, INBOUND, 'N'),
        Cordon('out_a', 0, -10_000, OUTBOUND, 'S'),
    ]
    return trips, demand, cordons


@pytest.mark.smoke
def test_tier1_passes_on_well_formed_output():
    trips, demand, cordons = _trips_and_demand()
    report = validate_tier1(trips, demand, cordons)
    failed = [c.name for c in report.checks if not c.passed]
    assert report.passed, f"unexpected failures: {failed}"


@pytest.mark.smoke
def test_tier1_catches_a_trip_count_mismatch():
    trips, demand, cordons = _trips_and_demand()
    demand.total_trips += 50
    report = validate_tier1(trips, demand, cordons)

    assert not report.passed
    assert any(c.name == 'trip_count_matches_demand' and not c.passed
               for c in report.checks)


@pytest.mark.smoke
def test_tier1_catches_a_wrong_way_cordon():
    """The check that matters most: a truck on the wrong carriageway."""
    trips, demand, cordons = _trips_and_demand()
    trips[0].origin_cordon = 'out_a'          # entering on an exit cordon
    report = validate_tier1(trips, demand, cordons)

    assert any(c.name == 'cordon_direction_respected' and not c.passed
               for c in report.checks)


@pytest.mark.smoke
def test_tier1_catches_an_unknown_cordon():
    trips, demand, cordons = _trips_and_demand()
    trips[0].origin_cordon = 'ghost'
    report = validate_tier1(trips, demand, cordons)

    assert any(c.name == 'cordon_ids_valid' and not c.passed
               for c in report.checks)


@pytest.mark.smoke
def test_tier1_catches_a_degenerate_through_trip():
    trips, demand, cordons = _trips_and_demand()
    through = next(t for t in trips if t.trip_class == CLASS_THROUGH)
    through.dest_cordon = through.origin_cordon
    report = validate_tier1(trips, demand, cordons)

    assert any(c.name == 'through_trips_use_distinct_cordons' and not c.passed
               for c in report.checks)


@pytest.mark.smoke
def test_tier1_catches_an_out_of_range_departure():
    trips, demand, cordons = _trips_and_demand()
    trips[0].departure_seconds = -100.0
    report = validate_tier1(trips, demand, cordons)

    assert any(c.name == 'departure_times_in_range' and not c.passed
               for c in report.checks)


@pytest.mark.smoke
def test_tier1_reports_empty_output_as_a_failure():
    _, demand, cordons = _trips_and_demand()
    report = validate_tier1([], demand, cordons)
    assert not report.passed


@pytest.mark.smoke
def test_tier1_tolerance_scales_with_sample_size():
    """A small sample is noisy; a correct model must not fail for that."""
    small_trips, small_demand, cordons = _trips_and_demand(n_each=15)
    report = validate_tier1(small_trips, small_demand, cordons)
    profile_checks = [c for c in report.checks
                      if c.name.startswith('departure_profile')]
    assert profile_checks
    assert all(c.passed for c in profile_checks)


# ---------------------------------------------------------------------------
# tier 2 validation
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_check_results_are_json_serialisable():
    """numpy.bool_ is truthy but not JSON-serialisable.

    Every check is the result of a numpy comparison, so an uncoerced
    ``numpy.bool_`` sinks the whole summary file at write time while each
    assertion still appears to pass. Found on the first real run.
    """
    import json
    from models.freight.validation import CheckResult

    check = CheckResult(name='x', passed=np.bool_(True), detail='d',
                        value=np.float64(1.5), tolerance=np.float64(2.0))

    assert check.passed is True
    assert isinstance(check.passed, bool)
    json.dumps(check.to_dict())


@pytest.mark.smoke
def test_tier1_report_survives_json_round_trip():
    import json
    trips, demand, cordons = _trips_and_demand()
    report = validate_tier1(trips, demand, cordons)
    payload = json.loads(json.dumps(report.to_dict()))
    assert isinstance(payload['passed'], bool)


@pytest.mark.smoke
def test_tier2_reports_missing_observations_rather_than_passing():
    report = validate_tier2({'n_links': 0})
    assert not report.passed


@pytest.mark.smoke
def test_tier2_passes_on_a_good_match():
    report = validate_tier2({'n_links': 20, 'pct_geh_under_5': 80.0,
                             'ratio': 1.05})
    assert report.passed


@pytest.mark.smoke
def test_tier2_fails_when_volumes_are_far_off():
    report = validate_tier2({'n_links': 20, 'pct_geh_under_5': 10.0,
                             'ratio': 3.0})
    assert not report.passed


@pytest.mark.smoke
def test_comparison_truncates_links_by_default_but_can_return_all():
    """Stratified %RMSE over the top 50 describes the busiest links, not the
    model. Measured on Anoka: 2,503 links compared, 392 of them carrying no
    simulated trucks, none of which appear in the top 50."""
    volumes = LinkVolumes(volumes={f'L{i}': np.full(24, 10.0) for i in range(60)})
    observed = {f'L{i}': 240.0 for i in range(60)}

    assert len(compare_against_observed(volumes, observed)['links']) == 50
    assert len(compare_against_observed(
        volumes, observed, all_links=True)['links']) == 60


@pytest.mark.smoke
def test_comparison_reports_coverage_alongside_the_ratio():
    """A good ratio with many empty links is a coverage failure wearing an
    aggregate success, so the two must be reported together."""
    volumes = LinkVolumes(volumes={'A': np.full(24, 10.0)})
    result = compare_against_observed(volumes, {'A': 240.0, 'B': 240.0})

    assert result['n_links_zero_simulated'] == 1
    assert result['median_observed'] == 240.0


# ---------------------------------------------------------------------------
# per-segment aggregation
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_segment_aggregate_counts_one_aadt_per_segment_not_per_link():
    """The bug this function exists to fix.

    A divided highway is two links in the network and one segment in HPMS, and
    ``match_corridor_links`` credits each link with the segment's full AADT.
    Summing observed over links then counts that AADT twice. Measured on Anoka
    the effect was 2,503 links over 884 segments — a 2.8x inflation that drove
    the reported ratio to 0.52 when it was 0.334.
    """
    volumes = LinkVolumes(volumes={
        'fwd': np.full(24, 5.0),    # 120/day
        'rev': np.full(24, 5.0),    # 120/day
    })
    groups = [{'truck_aadt': 240.0, 'link_ids': ['fwd', 'rev']}]

    per_link = compare_against_observed(volumes, {'fwd': 240.0, 'rev': 240.0})
    per_segment = compare_by_segment(volumes, groups)

    # Per link, the same AADT is counted once per carriageway.
    assert per_link['observed_total'] == 480.0
    # Per segment, once.
    assert per_segment['n_segments'] == 1
    assert per_segment['observed_total'] == 240.0


@pytest.mark.smoke
def test_segment_simulated_volume_is_max_over_links_not_sum():
    """AADT is a flow across a cross-section, not a quantity to add along a
    road. Four consecutive links of one segment each carrying 120 trucks are
    the same 120 trucks, so the segment carries 120 — not 480."""
    volumes = LinkVolumes(volumes={
        f'L{i}': np.full(24, 5.0) for i in range(4)   # 120/day each
    })
    groups = [{'truck_aadt': 120.0, 'link_ids': ['L0', 'L1', 'L2', 'L3']}]

    result = compare_by_segment(volumes, groups)

    assert result['simulated_total'] == 120.0
    assert result['ratio'] == pytest.approx(1.0)


@pytest.mark.smoke
def test_segment_aggregate_counts_a_segment_empty_only_if_every_link_is():
    """A segment is only uncovered when none of its links carries a truck;
    one loaded carriageway means the corridor is served."""
    volumes = LinkVolumes(volumes={'fwd': np.full(24, 5.0)})
    groups = [
        {'truck_aadt': 240.0, 'link_ids': ['fwd', 'rev']},   # rev has none
        {'truck_aadt': 240.0, 'link_ids': ['other']},        # nothing at all
    ]

    result = compare_by_segment(volumes, groups)

    assert result['n_segments'] == 2
    assert result['n_segments_zero_simulated'] == 1


@pytest.mark.smoke
def test_segment_aggregate_reports_rather_than_crashes_without_grouping():
    """An older matcher returns no grouping; tier 2 must say so, not fail."""
    assert compare_by_segment(LinkVolumes(), [])['n_segments'] == 0
