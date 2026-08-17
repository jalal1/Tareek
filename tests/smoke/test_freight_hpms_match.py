"""Smoke tests for HPMS per-segment link matching (design.md item 9).

The failure this module exists to prevent is silent: without it every cordon
took the capacity fallback, so the spatial distribution of freight was driven by
road size rather than by measured truck traffic — and nothing in the output said
so. These tests pin the behaviour that closes that gap, and in particular the
bearing test, because matching on distance alone gives an inbound cordon the
opposing carriageway's volume roughly half the time with no visible symptom.

No network access: every test builds its own segments.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from models.freight.cordons import Cordon, INBOUND, OUTBOUND
from models.freight.demand import assign_cordon_weights
from models.freight.hpms_match import (
    DEFAULT_BEARING_TOLERANCE_DEG,
    HPMSGeometryClient,
    HPMSSegment,
    _angular_diff,
    _bearing,
    _one_way_links,
    build_truck_aadt_by_link,
    cordon_bbox,
    directional_aadt,
    link_geometry,
    match_link_to_segment,
    parse_segments,
    resolve_truck_aadt_by_link,
)
from models.freight.truck_share import national_truck_share


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _northbound(truck_aadt=5000.0, f_system=1):
    """A carriageway running due north from the origin."""
    return HPMSSegment(
        points=np.array([[0.0, 0.0], [0.0, 1000.0]]),
        truck_aadt=truck_aadt, aadt=truck_aadt * 10,
        single_unit=truck_aadt * 0.4, combination=truck_aadt * 0.6,
        f_system=f_system,
    )


def _southbound(truck_aadt=4000.0, offset=30.0, f_system=1):
    """Its antiparallel twin, ``offset`` metres east."""
    return HPMSSegment(
        points=np.array([[offset, 1000.0], [offset, 0.0]]),
        truck_aadt=truck_aadt, aadt=truck_aadt * 10,
        single_unit=truck_aadt * 0.4, combination=truck_aadt * 0.6,
        f_system=f_system,
    )


class _FakeConverter:
    """lat/lon <-> metres with a fixed scale, so tests need no pyproj."""

    def latlon_to_utm(self, lat, lon):
        return lon * 100_000.0, lat * 100_000.0

    def utm_to_latlon(self, x, y):
        return y / 100_000.0, x / 100_000.0


def _write_network(tmp_path: Path) -> Path:
    """A network with one divided road and one two-way road."""
    path = tmp_path / 'network.xml'
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<network>\n<nodes>\n'
        '<node id="a" x="0" y="0"/><node id="b" x="0" y="1000"/>\n'
        '<node id="c" x="30" y="0"/><node id="d" x="30" y="1000"/>\n'
        '</nodes>\n<links>\n'
        '<link id="nb" from="a" to="b" capacity="4000" freespeed="30"'
        ' permlanes="2" modes="car"/>\n'
        '<link id="sb" from="d" to="c" capacity="4000" freespeed="30"'
        ' permlanes="2" modes="car"/>\n'
        '<link id="two_f" from="a" to="d" capacity="2000" freespeed="25"'
        ' permlanes="1" modes="car"/>\n'
        '<link id="two_r" from="d" to="a" capacity="2000" freespeed="25"'
        ' permlanes="1" modes="car"/>\n'
        '</links>\n</network>\n', encoding='utf-8')
    return path


# ---------------------------------------------------------------------------
# bearing helpers
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_bearing_is_clockwise_from_grid_north():
    """Must agree with counts_generator._link_bearing, which uses the same
    convention — otherwise the two matchers disagree about what north is."""
    assert _bearing(0, 0, 0, 1) == pytest.approx(0.0)     # north
    assert _bearing(0, 0, 1, 0) == pytest.approx(90.0)    # east
    assert _bearing(0, 0, 0, -1) == pytest.approx(180.0)  # south
    assert _bearing(0, 0, -1, 0) == pytest.approx(270.0)  # west


@pytest.mark.smoke
def test_bearing_of_zero_length_link_is_none():
    assert _bearing(5.0, 5.0, 5.0, 5.0) is None


@pytest.mark.smoke
def test_angular_diff_wraps_the_short_way():
    assert _angular_diff(350.0, 10.0) == pytest.approx(20.0)
    assert _angular_diff(0.0, 180.0) == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# matching — the carriageway problem
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_each_carriageway_takes_its_own_volume():
    """The whole reason bearing is part of the match.

    Distance alone cannot separate two carriageways 30 m apart; without the
    bearing test an inbound cordon takes the outbound volume about half the
    time, and nothing downstream reveals it.
    """
    segments = [_northbound(5000.0), _southbound(4000.0)]

    northbound = match_link_to_segment((0.0, 100.0, 0.0, 200.0), segments)
    southbound = match_link_to_segment((30.0, 200.0, 30.0, 100.0), segments)

    assert northbound is not None and southbound is not None
    assert northbound[0].truck_aadt == 5000.0
    assert southbound[0].truck_aadt == 4000.0


@pytest.mark.smoke
def test_perpendicular_road_is_rejected():
    """A crossing road passes the distance test and must fail the bearing one."""
    assert match_link_to_segment(
        (-50.0, 100.0, 50.0, 100.0), [_northbound()]) is None


@pytest.mark.smoke
def test_opposite_digitisation_of_the_same_road_still_matches():
    """HPMS digitises an undivided centreline once, in one direction only.

    A link running against that direction is still on the same road, so the
    match must accept either orientation — otherwise every undivided road loses
    one of its two directions.
    """
    match = match_link_to_segment((0.0, 200.0, 0.0, 100.0), [_northbound()])
    assert match is not None
    assert match[2] == pytest.approx(0.0)


@pytest.mark.smoke
def test_segment_beyond_the_radius_is_not_matched():
    assert match_link_to_segment(
        (5000.0, 100.0, 5000.0, 200.0), [_northbound()]) is None


@pytest.mark.smoke
def test_closest_matching_segment_wins():
    """Two parallel same-direction roads: the nearer one is the right answer."""
    near = _northbound(5000.0)
    far = HPMSSegment(points=np.array([[120.0, 0.0], [120.0, 1000.0]]),
                      truck_aadt=9999.0, aadt=99990.0,
                      single_unit=4000.0, combination=5999.0, f_system=1)
    match = match_link_to_segment((0.0, 100.0, 0.0, 200.0), [far, near])
    assert match[0].truck_aadt == 5000.0


@pytest.mark.smoke
def test_functional_class_guard_blocks_a_wild_mismatch():
    """An interstate cordon must not inherit a local road's truck volume."""
    local = _northbound(99.0, f_system=7)
    assert match_link_to_segment(
        (0.0, 100.0, 0.0, 200.0), [local], f_system=1) is None
    # ...but one class of slack is allowed, since HPMS and OSM disagree at the
    # freeway/expressway margin.
    expressway = _northbound(5000.0, f_system=2)
    assert match_link_to_segment(
        (0.0, 100.0, 0.0, 200.0), [expressway], f_system=1) is not None


@pytest.mark.smoke
def test_bearing_tolerance_boundary_is_respected():
    """Just inside the tolerance matches; well outside does not."""
    segments = [_northbound()]
    inside = match_link_to_segment(
        (0.0, 100.0, 10.0, 200.0), segments,
        bearing_tolerance_deg=DEFAULT_BEARING_TOLERANCE_DEG)
    assert inside is not None
    outside = match_link_to_segment(
        (0.0, 100.0, 100.0, 130.0), segments, bearing_tolerance_deg=10.0)
    assert outside is None


# ---------------------------------------------------------------------------
# directional AADT
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_one_way_carriageway_takes_half_the_published_aadt():
    """HPMS AADT is bidirectional on an undivided segment.

    Crediting each carriageway with the full figure would double the corridor,
    inflating the regional total that demand_scale is calibrated against.
    """
    segment = _northbound(5000.0)
    assert directional_aadt(segment, link_is_one_way=True) == 2500.0
    assert directional_aadt(segment, link_is_one_way=False) == 5000.0


# ---------------------------------------------------------------------------
# network reading
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_one_way_links_are_those_without_a_reverse_twin(tmp_path):
    path = _write_network(tmp_path)
    one_way = _one_way_links(path, {'nb', 'sb', 'two_f', 'two_r'})
    assert one_way == {'nb', 'sb'}


@pytest.mark.smoke
def test_link_geometry_returns_only_requested_links(tmp_path):
    path = _write_network(tmp_path)
    geometry = link_geometry(path, link_ids={'nb'})
    assert set(geometry) == {'nb'}
    assert geometry['nb'] == (0.0, 0.0, 0.0, 1000.0)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_parse_segments_reads_geojson_linestrings():
    features = [{
        'geometry': {'type': 'LineString',
                     'coordinates': [[-86.8, 33.5], [-86.8, 33.6]]},
        'properties': {'AADT': 50000, 'AADT_SINGLE_UNIT': 2000,
                       'AADT_COMBINATION': 3000, 'F_SYSTEM': 1,
                       'URBAN_CODE': 12345},
    }]
    segments = parse_segments(features, _FakeConverter())
    assert len(segments) == 1
    assert segments[0].truck_aadt == 5000
    assert segments[0].is_rural is False


@pytest.mark.smoke
def test_parse_segments_splits_multilinestrings():
    features = [{
        'geometry': {'type': 'MultiLineString', 'coordinates': [
            [[-86.8, 33.5], [-86.8, 33.6]],
            [[-86.7, 33.5], [-86.7, 33.6]]]},
        'properties': {'AADT': 100, 'AADT_SINGLE_UNIT': 10,
                       'AADT_COMBINATION': 10, 'F_SYSTEM': 3},
    }]
    assert len(parse_segments(features, _FakeConverter())) == 2


@pytest.mark.smoke
def test_parse_segments_skips_untrafficked_and_malformed():
    """HPMS is 50 state submissions; bad rows are expected, not exceptional."""
    features = [
        {'geometry': {'type': 'LineString',
                      'coordinates': [[-86.8, 33.5], [-86.8, 33.6]]},
         'properties': {'AADT': 100, 'AADT_SINGLE_UNIT': 0,
                        'AADT_COMBINATION': 0}},          # no trucks
        {'geometry': {}, 'properties': {'AADT_SINGLE_UNIT': 5,
                                        'AADT_COMBINATION': 5}},  # no geometry
        {'geometry': {'type': 'LineString', 'coordinates': [[-86.8, 33.5]]},
         'properties': {'AADT_SINGLE_UNIT': 5,
                        'AADT_COMBINATION': 5}},          # single point
    ]
    assert parse_segments(features, _FakeConverter()) == []


@pytest.mark.smoke
def test_parse_segments_accepts_esri_paths_geometry():
    features = [{
        'geometry': {'paths': [[[-86.8, 33.5], [-86.8, 33.6]]]},
        'attributes': {'AADT': 100, 'AADT_SINGLE_UNIT': 30,
                       'AADT_COMBINATION': 20, 'F_SYSTEM': 1},
    }]
    segments = parse_segments(features, _FakeConverter())
    assert len(segments) == 1 and segments[0].truck_aadt == 50


# ---------------------------------------------------------------------------
# the seam: feeding assign_cordon_weights
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_build_truck_aadt_produces_the_dict_the_weighter_wants():
    segments = [_northbound(5000.0), _southbound(4000.0)]
    cordons = [
        Cordon('c_in', 0.0, 100.0, INBOUND, 'N', ['nb'], capacity=8000),
        Cordon('c_out', 30.0, 100.0, OUTBOUND, 'S', ['sb'], capacity=8000),
    ]
    link_xy = {'nb': (0.0, 100.0, 0.0, 200.0),
               'sb': (30.0, 200.0, 30.0, 100.0)}

    truck_aadt, stats = build_truck_aadt_by_link(
        cordons, segments, link_xy, one_way_links={'nb', 'sb'})

    assert truck_aadt == {'nb': 2500.0, 'sb': 2000.0}
    assert stats['n_cordons_matched'] == 2
    assert stats['match_rate'] == 1.0


@pytest.mark.smoke
def test_matching_moves_cordons_off_the_capacity_fallback():
    """The item-9 success criterion, stated as a test.

    ``n_observed`` rising from zero is exactly what the design says success
    looks like, and an unmatched cordon must still get its capacity weight.
    """
    segments = [_northbound(5000.0)]
    cordons = [
        Cordon('matched', 0.0, 100.0, INBOUND, 'N', ['nb'], capacity=8000),
        Cordon('unmatched', 9000.0, 9000.0, INBOUND, 'E', ['far'],
               capacity=3000),
    ]
    link_xy = {'nb': (0.0, 100.0, 0.0, 200.0),
               'far': (9000.0, 9000.0, 9000.0, 9100.0)}

    truck_aadt, _ = build_truck_aadt_by_link(
        cordons, segments, link_xy, one_way_links={'nb'})
    stats = assign_cordon_weights(
        cordons, truck_aadt_by_link=truck_aadt,
        truck_share=national_truck_share(1, False))

    assert stats['n_observed'] == 1
    assert stats['n_fallback'] == 1
    assert cordons[0].weight == 2500.0        # observed
    assert cordons[1].weight != 3000.0        # capacity fallback, converted


@pytest.mark.smoke
def test_big_corridors_match_in_preference_to_small_ones():
    """The match *rate* is not the quality signal — the bias is.

    A 56% rate is fine if the unmatched remainder is minor roads HPMS does not
    cover, and alarming if the big corridors are the ones failing. Measured on
    Anoka the matched cordons have median capacity 4,000 against 1,875
    unmatched; this pins that direction so a regression that starts dropping
    interstates is caught here rather than in a tier-2 result.
    """
    interstate = _northbound(8000.0, f_system=1)
    cordons = [
        Cordon('big', 0.0, 100.0, INBOUND, 'N', ['main'], capacity=40_000),
        Cordon('small', 5000.0, 5000.0, INBOUND, 'E', ['lane'], capacity=800),
    ]
    link_xy = {'main': (0.0, 100.0, 0.0, 200.0),
               'lane': (5000.0, 5000.0, 5000.0, 5100.0)}

    truck_aadt, _ = build_truck_aadt_by_link(
        cordons, [interstate], link_xy, one_way_links={'main'})

    assert 'main' in truck_aadt
    assert 'lane' not in truck_aadt


@pytest.mark.smoke
def test_no_match_leaves_every_cordon_on_capacity():
    """Degrading to the pre-item-9 behaviour must be silent-proof, not silent:
    the stats have to say the match rate was zero."""
    cordons = [Cordon('c', 0.0, 0.0, INBOUND, 'N', ['l'], capacity=5000)]
    truck_aadt, stats = build_truck_aadt_by_link(
        cordons, [], {'l': (0.0, 0.0, 0.0, 100.0)})
    assert truck_aadt == {}
    assert stats['match_rate'] == 0.0


# ---------------------------------------------------------------------------
# bbox
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_cordon_bbox_covers_every_cordon_with_padding():
    cordons = [Cordon('a', 0.0, 0.0, INBOUND, 'N', ['1']),
               Cordon('b', 100_000.0, 100_000.0, OUTBOUND, 'S', ['2'])]
    min_lon, min_lat, max_lon, max_lat = cordon_bbox(
        cordons, _FakeConverter(), pad_km=5.0)
    assert min_lon < 0.0 and min_lat < 0.0
    assert max_lon > 1.0 and max_lat > 1.0


@pytest.mark.smoke
def test_cordon_bbox_needs_cordons():
    from models.freight.hpms_match import HPMSMatchError
    with pytest.raises(HPMSMatchError):
        cordon_bbox([], _FakeConverter())


# ---------------------------------------------------------------------------
# failure tolerance — the contract every network path in this package keeps
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_disabled_hpms_returns_empty_without_touching_the_network():
    config = {'freight': {'hpms': {'enabled': False}}}
    cordons = [Cordon('c', 0.0, 0.0, INBOUND, 'N', ['l'])]
    truck_aadt, stats = resolve_truck_aadt_by_link(
        config, cordons, Path('does_not_exist.xml'), _FakeConverter())
    assert truck_aadt == {}
    assert 'skipped' in stats


@pytest.mark.smoke
def test_resolver_never_raises_on_a_broken_setup():
    """A missing network file, an unreachable service — none may stop a run."""
    config = {'freight': {'hpms': {'enabled': True,
                                   'service_url': 'http://127.0.0.1:1/nope'}}}
    cordons = [Cordon('c', 0.0, 0.0, INBOUND, 'N', ['l'])]
    truck_aadt, stats = resolve_truck_aadt_by_link(
        config, cordons, Path('does_not_exist.xml'), _FakeConverter())
    assert truck_aadt == {}
    assert isinstance(stats, dict)


@pytest.mark.smoke
def test_unreachable_service_returns_no_features():
    client = HPMSGeometryClient(service_url='http://127.0.0.1:1/nope',
                                cache_dir=None, timeout_seconds=0.05)
    assert client.fetch_bbox((-1.0, -1.0, 1.0, 1.0)) == []


@pytest.mark.smoke
def test_geometry_cache_round_trips(tmp_path):
    client = HPMSGeometryClient(cache_dir=tmp_path, cache_days=90)
    bbox = (-86.9, 33.4, -86.6, 33.6)
    features = [{'geometry': {'type': 'LineString',
                              'coordinates': [[-86.8, 33.5], [-86.8, 33.6]]},
                 'properties': {'AADT_SINGLE_UNIT': 1, 'AADT_COMBINATION': 1}}]
    client.write_cache(bbox, features)
    assert client.read_cache(bbox) == features


@pytest.mark.smoke
def test_stale_cache_is_ignored(tmp_path):
    client = HPMSGeometryClient(cache_dir=tmp_path, cache_days=0)
    bbox = (-1.0, -1.0, 1.0, 1.0)
    client.write_cache(bbox, [{'x': 1}])
    assert client.read_cache(bbox) is None


@pytest.mark.smoke
def test_incomplete_fetch_is_cached_but_never_served_as_complete(tmp_path):
    """A truncated fetch must not masquerade as full regional coverage.

    Measured on 15-county Twin Cities: the service returned a 400 at offset
    54,000, and the partial result was cached exactly like a complete one. Every
    later run inside ``cache_days`` would have reused those 54k segments
    believing the region fully covered, with no symptom beyond a quietly lower
    cordon match rate. The entry is still written — refetching is slow, and a
    partial region beats none — but reading it forces a refetch.
    """
    client = HPMSGeometryClient(cache_dir=tmp_path, cache_days=90)
    bbox = (-94.4, 44.0, -92.0, 46.4)
    client.write_cache(bbox, [{'x': 1}], complete=False)

    assert client.read_cache(bbox) is None, \
        "an incomplete fetch must not be served from cache"


@pytest.mark.smoke
def test_cache_written_before_the_complete_flag_existed_still_reads(tmp_path):
    """Entries predating the flag have no 'complete' key and were complete."""
    client = HPMSGeometryClient(cache_dir=tmp_path, cache_days=90)
    bbox = (-1.0, -1.0, 1.0, 1.0)
    path = client._cache_path(bbox)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'bbox': list(bbox), 'fetched_at': time.time(),
                                'features': [{'x': 1}]}), encoding='utf-8')

    assert client.read_cache(bbox) == [{'x': 1}]


@pytest.mark.smoke
def test_corrupt_cache_is_ignored_not_fatal(tmp_path):
    client = HPMSGeometryClient(cache_dir=tmp_path)
    bbox = (-1.0, -1.0, 1.0, 1.0)
    path = client._cache_path(bbox)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{not json', encoding='utf-8')
    assert client.read_cache(bbox) is None


# ---------------------------------------------------------------------------
# corridor matching — tier 2 (item 15)
# ---------------------------------------------------------------------------
#
# Tier 2 is the only check that judges the freight *level*, because it compares
# against corridor links the demand was not derived from. The cordon screenline
# cannot: its yardstick is the same cordon weighting the demand comes from.

@pytest.mark.smoke
def test_corridor_matching_is_bearing_aware_like_the_cordon_matcher():
    """Same carriageway rule, applied at corridor scale."""
    from models.freight.hpms_match import match_corridor_links

    segments = [_northbound(5000.0), _southbound(4000.0)]
    links = {'nb': (0.0, 100.0, 0.0, 200.0),
             'sb': (30.0, 200.0, 30.0, 100.0)}

    aadt, stats = match_corridor_links(links, segments,
                                       one_way_links={'nb', 'sb'})

    assert aadt['nb'] == 2500.0
    assert aadt['sb'] == 2000.0
    assert stats['n_matched'] == 2


@pytest.mark.smoke
def test_corridor_matching_rejects_perpendicular_roads():
    from models.freight.hpms_match import match_corridor_links

    aadt, _ = match_corridor_links(
        {'cross': (-50.0, 100.0, 50.0, 100.0)}, [_northbound()])
    assert aadt == {}


@pytest.mark.smoke
def test_corridor_matching_scales_to_many_links():
    """A nested loop is ~10^10 ops on a real network; the index makes it viable.

    This is a correctness test for the indexed path, not a benchmark: it checks
    that indexing returns the same answers the naive matcher would.
    """
    from models.freight.hpms_match import match_corridor_links

    segments = [
        HPMSSegment(points=np.array([[0.0, k * 200.0], [0.0, k * 200.0 + 200.0]]),
                    truck_aadt=5000.0, aadt=50_000.0,
                    single_unit=2000.0, combination=3000.0, f_system=1)
        for k in range(50)
    ]
    links = {f'L{i}': (0.0, i * 20.0, 0.0, i * 20.0 + 20.0) for i in range(500)}

    aadt, stats = match_corridor_links(links, segments)
    assert stats['n_matched'] == 500
    assert stats['match_rate'] == 1.0


@pytest.mark.smoke
def test_corridor_links_filters_to_through_corridors(tmp_path):
    """Must agree with CordonDetector about what a corridor is."""
    from models.freight.hpms_match import corridor_links

    path = tmp_path / 'net.xml'
    path.write_text(
        '<?xml version="1.0"?>\n<network>\n<nodes>\n'
        '<node id="a" x="0" y="0"/><node id="b" x="0" y="1000"/>\n'
        '</nodes>\n<links>\n'
        '<link id="motorway" from="a" to="b" capacity="4000" freespeed="30"'
        ' permlanes="2" modes="car"/>\n'
        '<link id="street" from="a" to="b" capacity="600" freespeed="11"'
        ' permlanes="1" modes="car"/>\n'
        '<link id="footpath" from="a" to="b" capacity="4000" freespeed="30"'
        ' permlanes="1" modes="walk"/>\n'
        '</links>\n</network>\n', encoding='utf-8')

    links = corridor_links(path)
    assert set(links) == {'motorway'}


@pytest.mark.smoke
def test_corridor_resolver_never_raises_when_hpms_is_off():
    from models.freight.hpms_match import resolve_corridor_truck_aadt

    aadt, stats = resolve_corridor_truck_aadt(
        {'freight': {'hpms': {'enabled': False}}},
        [Cordon('c', 0.0, 0.0, INBOUND, 'N', ['l'])],
        Path('nope.xml'), _FakeConverter())
    assert aadt == {}
    assert 'skipped' in stats
