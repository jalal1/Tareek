"""Smoke tests for the per-direction FHA counts pipeline.

Covers the pure, network-light logic that the per-direction model depends on:
  - link bearing / angular-difference math
  - bearing-based travel_dir -> link assignment, including the joint
    opposite-pair assignment (antiparallel links) and the missing-antiparallel
    drop case
  - the evaluator's countscompare.txt reader and station-base stripping

These use tiny synthetic networks (no DB / no Java) so they run in the smoke
tier.
"""

import pandas as pd
import pytest
from rtree import index
from shapely.geometry import LineString

from matsim.counts_generator import (
    CountsGenerator, DIR_TO_BEARING, OPPOSITE_DIRS, HOUR_COLS_UPPER,
    parse_f_system,
)
from matsim.evaluator import SimulationEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_generator():
    cfg = {'coordinates': {'utm_epsg': 'EPSG:26915'}, 'counts': {}, 'evaluation': {}}
    return CountsGenerator(cfg, db_manager=None)


def _load_links(g, links: pd.DataFrame):
    """Wire a tiny network into the generator (mimics load_network).

    Links may carry 'capacity'/'freespeed' for facility-class matching; when
    absent they default to freeway-grade so class-agnostic tests are unaffected.
    """
    g.network_links = links
    g.link_geometries = {
        r['link_id']: LineString([(r['from_x'], r['from_y']), (r['to_x'], r['to_y'])])
        for _, r in links.iterrows()
    }
    g._link_endpoints = {
        r['link_id']: (r['from_x'], r['from_y'], r['to_x'], r['to_y'])
        for _, r in links.iterrows()
    }
    g._link_attributes = {
        r['link_id']: (float(r.get('capacity', 8000.0)),
                       float(r.get('freespeed', 26.8)))
        for _, r in links.iterrows()
    }
    idx = index.Index()
    for i, r in links.iterrows():
        idx.insert(i, (min(r['from_x'], r['to_x']), min(r['from_y'], r['to_y']),
                       max(r['from_x'], r['to_x']), max(r['from_y'], r['to_y'])))
    g.spatial_index = idx
    # Reverse-node index (mirrors load_network) so get_reverse_link_id works.
    g._reverse_node_index = {
        (r['from_node'], r['to_node']): r['link_id'] for _, r in links.iterrows()
    }


def _station_rows(base, dirs_and_vols, f_system=None):
    """Build (volumes_df, stations_df) for one physical station with given dirs."""
    stations, volumes = [], []
    for d, vol in dirs_and_vols:
        lid = f"{base}_{d}"
        stations.append({'LOCAL_ID': lid, 'station_base': base, 'travel_dir': d,
                         'utm_x': 500, 'utm_y': 500, 'Latitude': 44.0, 'Longitude': -93.0,
                         'f_system': f_system})
        rec = {h: vol for h in HOUR_COLS_UPPER}
        rec['LOCAL_ID'] = lid
        volumes.append(rec)
    return pd.DataFrame(volumes), pd.DataFrame(stations)


# ---------------------------------------------------------------------------
# Bearing math
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_link_bearing_cardinals():
    g = _make_generator()
    assert g._link_bearing(0, 0, 0, 100) == pytest.approx(0.0)    # north
    assert g._link_bearing(0, 0, 100, 0) == pytest.approx(90.0)   # east
    assert g._link_bearing(0, 0, 0, -100) == pytest.approx(180.0) # south
    assert g._link_bearing(0, 0, -100, 0) == pytest.approx(270.0) # west


@pytest.mark.smoke
def test_link_bearing_zero_length_is_none():
    g = _make_generator()
    assert g._link_bearing(5, 5, 5, 5) is None


@pytest.mark.smoke
def test_angular_diff_wraps():
    g = _make_generator()
    assert g._angular_diff(350, 10) == pytest.approx(20.0)
    assert g._angular_diff(10, 350) == pytest.approx(20.0)
    assert g._angular_diff(0, 180) == pytest.approx(180.0)


@pytest.mark.smoke
def test_dir_to_bearing_and_opposite_consistent():
    # Cardinals only, and every direction has a distinct opposite.
    assert set(DIR_TO_BEARING) == {1, 3, 5, 7}
    for d, opp in OPPOSITE_DIRS.items():
        assert OPPOSITE_DIRS[opp] == d
        assert g_diff(DIR_TO_BEARING[d], DIR_TO_BEARING[opp]) == pytest.approx(180.0)


def g_diff(a, b):
    return CountsGenerator._angular_diff(a, b)


# ---------------------------------------------------------------------------
# Direction -> link matching
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_match_direction_picks_bearing_aligned_link():
    g = _make_generator()
    _load_links(g, pd.DataFrame([
        {'link_id': 'linkN', 'from_node': 'a', 'to_node': 'b',
         'from_x': 500, 'from_y': 480, 'to_x': 500, 'to_y': 520},
        {'link_id': 'linkS', 'from_node': 'c', 'to_node': 'd',
         'from_x': 505, 'from_y': 520, 'to_x': 505, 'to_y': 480},
    ]))
    link, _, method = g.match_direction_to_link(500, 500, 1)  # north
    assert link == 'linkN'
    assert method == 'bearing'


@pytest.mark.smoke
def test_match_direction_excludes_taken_link():
    g = _make_generator()
    _load_links(g, pd.DataFrame([
        {'link_id': 'linkN', 'from_node': 'a', 'to_node': 'b',
         'from_x': 500, 'from_y': 480, 'to_x': 500, 'to_y': 520},
        {'link_id': 'linkS', 'from_node': 'c', 'to_node': 'd',
         'from_x': 505, 'from_y': 520, 'to_x': 505, 'to_y': 480},
    ]))
    link, _, _ = g.match_direction_to_link(500, 500, 5, exclude_link_id='linkN')
    assert link == 'linkS'


@pytest.mark.smoke
def test_directional_pair_assigned_to_antiparallel_links():
    """The two opposite directions of one sensor land on distinct links."""
    g = _make_generator()
    _load_links(g, pd.DataFrame([
        {'link_id': 'linkN', 'from_node': 'a', 'to_node': 'b',
         'from_x': 500, 'from_y': 480, 'to_x': 500, 'to_y': 520},
        {'link_id': 'linkS', 'from_node': 'c', 'to_node': 'd',
         'from_x': 505, 'from_y': 520, 'to_x': 505, 'to_y': 480},
    ]))
    volumes, stations = _station_rows('FHA_27_000026', [(1, 100), (5, 40)])
    out = g.match_fha_directional_to_links(volumes, stations)
    assert len(out) == 2
    assert set(out['matched_link_id']) == {'linkN', 'linkS'}
    # Output keeps the columns downstream consumers need.
    for col in ('LOCAL_ID', 'travel_dir', 'matched_link_id', 'utm_x', 'utm_y',
                'Latitude', 'Longitude'):
        assert col in out.columns


# ---------------------------------------------------------------------------
# Facility-class (HPMS f_system) matching
# ---------------------------------------------------------------------------

def _freeway_and_frontage():
    """A 4-lane freeway pair 27 m away, plus a 1-lane frontage pair at 0 m.

    Mirrors the real TwinCities failure: the sensor coordinate sits ON the
    frontage road, so nearest-link matching binds an Interstate count to a
    600 veh/h service road.
    """
    return pd.DataFrame([
        # Frontage road carriageways — closest to the sensor, but tiny.
        {'link_id': 'frontN', 'from_node': 'f1', 'to_node': 'f2',
         'from_x': 500, 'from_y': 480, 'to_x': 500, 'to_y': 520,
         'capacity': 600.0, 'freespeed': 11.1},
        {'link_id': 'frontS', 'from_node': 'f2', 'to_node': 'f1',
         'from_x': 500, 'from_y': 520, 'to_x': 500, 'to_y': 480,
         'capacity': 600.0, 'freespeed': 11.1},
        # Freeway carriageways — 27 m east, 4 lanes, freeway speed.
        {'link_id': 'fwyN', 'from_node': 'w1', 'to_node': 'w2',
         'from_x': 527, 'from_y': 480, 'to_x': 527, 'to_y': 520,
         'capacity': 8000.0, 'freespeed': 26.8},
        {'link_id': 'fwyS', 'from_node': 'w2', 'to_node': 'w1',
         'from_x': 527, 'from_y': 520, 'to_x': 527, 'to_y': 480,
         'capacity': 8000.0, 'freespeed': 26.8},
    ])


@pytest.mark.smoke
def test_parse_f_system_variants():
    assert parse_f_system('1U') == 1
    assert parse_f_system('3R') == 3
    assert parse_f_system('2') == 2
    assert parse_f_system(4) == 4
    for blank in ('', '   ', None, 'X', float('nan')):
        assert parse_f_system(blank) is None


@pytest.mark.smoke
def test_interstate_station_skips_frontage_road():
    """An f_system=1 station binds to the freeway, not the closer frontage road."""
    g = _make_generator()
    _load_links(g, _freeway_and_frontage())
    volumes, stations = _station_rows('FHA_27_010794', [(1, 60000), (5, 60000)],
                                      f_system='1U')
    out = g.match_fha_directional_to_links(volumes, stations)
    assert len(out) == 2
    assert set(out['matched_link_id']) == {'fwyN', 'fwyS'}
    assert set(out['match_method']) == {'class'}


@pytest.mark.smoke
def test_local_station_still_uses_nearest_link():
    """Functional systems 5-7 are unconstrained — nearest-link wins as before."""
    g = _make_generator()
    _load_links(g, _freeway_and_frontage())
    volumes, stations = _station_rows('FHA_27_000001', [(1, 300), (5, 250)],
                                      f_system='7U')
    out = g.match_fha_directional_to_links(volumes, stations)
    assert set(out['matched_link_id']) == {'frontN', 'frontS'}
    assert set(out['match_method']) == {'nearest'}


@pytest.mark.smoke
def test_missing_f_system_falls_back_to_nearest():
    """No class signal => unchanged legacy behaviour, no station lost."""
    g = _make_generator()
    _load_links(g, _freeway_and_frontage())
    volumes, stations = _station_rows('FHA_27_000002', [(1, 300), (5, 250)],
                                      f_system=None)
    out = g.match_fha_directional_to_links(volumes, stations)
    assert len(out) == 2
    assert set(out['matched_link_id']) == {'frontN', 'frontS'}


@pytest.mark.smoke
def test_interstate_with_no_plausible_link_is_dropped():
    """Rather than bind an Interstate count to a service road, drop it."""
    g = _make_generator()
    _load_links(g, pd.DataFrame([
        {'link_id': 'frontN', 'from_node': 'f1', 'to_node': 'f2',
         'from_x': 500, 'from_y': 480, 'to_x': 500, 'to_y': 520,
         'capacity': 600.0, 'freespeed': 11.1},
        {'link_id': 'frontS', 'from_node': 'f2', 'to_node': 'f1',
         'from_x': 500, 'from_y': 520, 'to_x': 500, 'to_y': 480,
         'capacity': 600.0, 'freespeed': 11.1},
    ]))
    volumes, stations = _station_rows('FHA_27_010794', [(1, 60000), (5, 60000)],
                                      f_system='1U')
    out = g.match_fha_directional_to_links(volumes, stations)
    assert out.empty


@pytest.mark.smoke
def test_link_matches_fsystem_thresholds():
    g = _make_generator()
    _load_links(g, _freeway_and_frontage())
    # Interstate (1) rejects the frontage road, accepts the freeway.
    assert not g.link_matches_fsystem('frontN', 1)
    assert g.link_matches_fsystem('fwyN', 1)
    # Unconstrained classes accept anything.
    assert g.link_matches_fsystem('frontN', 7)
    assert g.link_matches_fsystem('frontN', None)


@pytest.mark.smoke
def test_directional_missing_antiparallel_drops_lighter_direction():
    """With only one link, keep the heavier direction and drop the other."""
    g = _make_generator()
    _load_links(g, pd.DataFrame([
        {'link_id': 'linkN', 'from_node': 'a', 'to_node': 'b',
         'from_x': 500, 'from_y': 480, 'to_x': 500, 'to_y': 520},
    ]))
    # dir 1 heavier (100) than dir 5 (40) -> dir 1 kept, dir 5 dropped.
    volumes, stations = _station_rows('FHA_27_999', [(1, 100), (5, 40)])
    out = g.match_fha_directional_to_links(volumes, stations)
    assert len(out) == 1
    assert out.iloc[0]['travel_dir'] == 1
    assert out.iloc[0]['matched_link_id'] == 'linkN'


@pytest.mark.smoke
def test_proximity_beats_bearing():
    """A far link with a perfect bearing must NOT win over the near road.

    Regression: the sensor sits ON the near road (an NE arterial). A distant
    pair of perfectly N/S local links exists too. The match must pick the near
    arterial + its antiparallel, not the far perfectly-aligned links.
    """
    g = _make_generator()
    _load_links(g, pd.DataFrame([
        # Near arterial at the station (~3 m), running NE (bearing ~45).
        {'link_id': 'near_fwd', 'from_node': 'a', 'to_node': 'b',
         'from_x': 497, 'from_y': 497, 'to_x': 540, 'to_y': 540},
        {'link_id': 'near_rev', 'from_node': 'b', 'to_node': 'a',
         'from_x': 540, 'from_y': 540, 'to_x': 497, 'to_y': 497},
        # Far links (~600 m away) perfectly N and S.
        {'link_id': 'far_N', 'from_node': 'c', 'to_node': 'd',
         'from_x': 500, 'from_y': 1080, 'to_x': 500, 'to_y': 1120},
        {'link_id': 'far_S', 'from_node': 'd', 'to_node': 'c',
         'from_x': 505, 'from_y': 1120, 'to_x': 505, 'to_y': 1080},
    ]))
    volumes, stations = _station_rows('FHA_27_777', [(1, 100), (5, 40)])
    out = g.match_fha_directional_to_links(volumes, stations)
    assert len(out) == 2
    # Both must land on the NEAR arterial's two carriageways, never the far links.
    assert set(out['matched_link_id']) == {'near_fwd', 'near_rev'}
    assert out['distance_m'].max() < 50  # near, not ~600 m away


# ---------------------------------------------------------------------------
# Evaluator: countscompare reader + station-base stripping
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_station_base_strips_direction_suffix():
    assert SimulationEvaluator._station_base('FHA_27_000026_1') == 'FHA_27_000026'
    assert SimulationEvaluator._station_base('FHA_27_011516_5') == 'FHA_27_011516'
    # No numeric direction suffix -> unchanged (custom/legacy ids).
    assert SimulationEvaluator._station_base('CUSTOM_ABC') == 'CUSTOM_ABC'
    # Combined cs_id keeps the first segment's base.
    assert SimulationEvaluator._station_base('FHA_27_1_1+FHA_27_2_5') == 'FHA_27_1'


@pytest.mark.smoke
def test_compare_from_countscompare_reads_per_station_rows(tmp_path):
    """The reader produces one row per station-hour with GEH taken verbatim."""
    f = tmp_path / "10.countscompare.txt"
    # Two stations (fwd/rev) x 2 hours, tab-separated like MATSim writes.
    f.write_text(
        "Link Id\tCount Station Id\tHour\tMATSIM volumes\tCount volumes\t"
        "Relative Error\tNormalized Relative Error\tGEH\n"
        "11013\tFHA_27_500_1\t1\t90\t100\t-0.1\t0.1\t1.02\n"
        "11013\tFHA_27_500_1\t2\t0\t50\t-1\t1\t9.99\n"
        "67812\tFHA_27_500_5\t1\t40\t40\t0\t0\t0.0\n"
        "67812\tFHA_27_500_5\t2\t30\t25\t0.2\t0.2\t0.95\n",
        encoding="utf-8",
    )
    ev = SimulationEvaluator(experiment_dir=tmp_path, ground_truth_data_dir=tmp_path)
    df = ev.compare_volumes_from_countscompare(f)

    assert len(df) == 4
    assert set(df['device_id']) == {'FHA_27_500_1', 'FHA_27_500_5'}
    # Hours converted 1..24 -> 0..23.
    assert set(df['hour']) == {0, 1}
    # GEH taken straight from the file (no recomputation).
    row = df[(df['device_id'] == 'FHA_27_500_1') & (df['hour'] == 0)].iloc[0]
    assert row['geh'] == pytest.approx(1.02)
    assert row['observed'] == pytest.approx(100)
    assert row['simulated'] == pytest.approx(90)
    # Zero-observed -> NaN pct_error; here all observed > 0.
    assert df['pct_error'].notna().all()
    # station_base collapses both directions to one physical station.
    assert set(df['station_base']) == {'FHA_27_500'}


@pytest.mark.smoke
def test_summary_metrics_num_devices_is_physical_stations(tmp_path):
    f = tmp_path / "10.countscompare.txt"
    f.write_text(
        "Link Id\tCount Station Id\tHour\tMATSIM volumes\tCount volumes\t"
        "Relative Error\tNormalized Relative Error\tGEH\n"
        "11013\tFHA_27_500_1\t1\t90\t100\t-0.1\t0.1\t1.02\n"
        "67812\tFHA_27_500_5\t1\t40\t40\t0\t0\t0.0\n",
        encoding="utf-8",
    )
    ev = SimulationEvaluator(experiment_dir=tmp_path, ground_truth_data_dir=tmp_path)
    df = ev.compare_volumes_from_countscompare(f)
    m = ev.calculate_summary_metrics(df)
    # Two directions, one physical station.
    assert m['num_devices'] == 1
    assert m['num_directional_counts'] == 2
