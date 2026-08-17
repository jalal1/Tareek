"""Smoke tests for freight cordon detection.

Built on synthetic networks rather than the cached region network, so the tier
stays dependency-free. The synthetic grid reproduces the two properties that
broke the original edge-band design and that the detector now has to handle:

  - the network's extreme nodes are outliers, so a min/max rectangle does not
    describe where the roads actually end;
  - a corridor can break in the interior, which looks like a terminus but is
    not a gateway.

See docs/freight/design.md §2.
"""

import xml.etree.ElementTree as ET

import pytest

from models.freight.cordons import (
    BIDIRECTIONAL,
    Cordon,
    CordonDetectionError,
    CordonDetector,
    DirectionalEnvelope,
    INBOUND,
    OUTBOUND,
    detector_from_config,
)

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_network(path, nodes, links):
    """Write a minimal MATSim network.xml."""
    root = ET.Element('network')
    nodes_elem = ET.SubElement(root, 'nodes')
    for node_id, (x, y) in nodes.items():
        ET.SubElement(nodes_elem, 'node',
                      {'id': node_id, 'x': str(x), 'y': str(y)})
    links_elem = ET.SubElement(root, 'links')
    for link in links:
        attribs = {
            'id': link['id'],
            'from': link['from'],
            'to': link['to'],
            'length': str(link.get('length', 1000.0)),
            'freespeed': str(link.get('freespeed', 27.0)),
            'capacity': str(link.get('capacity', 4000.0)),
            'permlanes': str(link.get('permlanes', 2.0)),
            'modes': link.get('modes', 'car'),
        }
        ET.SubElement(links_elem, 'link', attribs)
    ET.ElementTree(root).write(str(path), encoding='utf-8', xml_declaration=True)
    return path


def _cross_network(tmp_path, name='network.xml'):
    """A four-armed cross: one corridor entering and leaving on each side.

    Each arm is a pair of one-way links (an inbound and an outbound
    carriageway), so a correct detector finds four inbound and four outbound
    gateways, one per compass direction.
    """
    centre = (500_000.0, 5_000_000.0)
    arm_m = 20_000.0
    nodes = {'c': centre}
    links = []
    for name_, (dx, dy) in (('E', (1, 0)), ('N', (0, 1)),
                            ('W', (-1, 0)), ('S', (0, -1))):
        outer = (centre[0] + dx * arm_m, centre[1] + dy * arm_m)
        mid = (centre[0] + dx * arm_m / 2, centre[1] + dy * arm_m / 2)
        nodes[f'{name_}_outer'] = outer
        nodes[f'{name_}_mid'] = mid
        # inbound carriageway: outer -> mid -> centre
        links.append({'id': f'{name_}_in_1', 'from': f'{name_}_outer', 'to': f'{name_}_mid'})
        links.append({'id': f'{name_}_in_2', 'from': f'{name_}_mid', 'to': 'c'})
        # outbound carriageway: centre -> mid -> outer
        links.append({'id': f'{name_}_out_1', 'from': 'c', 'to': f'{name_}_mid'})
        links.append({'id': f'{name_}_out_2', 'from': f'{name_}_mid', 'to': f'{name_}_outer'})
    return _write_network(tmp_path / name, nodes, links)


# ---------------------------------------------------------------------------
# DirectionalEnvelope
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_envelope_is_not_distorted_by_a_single_outlier():
    """The failure that killed the edge-band method.

    One stray node must not define the network's reach: a percentile envelope
    ignores it, where a min/max rectangle would let it dictate a whole side.
    """
    points = np.array([[math_x, math_y]
                       for math_x in range(0, 10_000, 500)
                       for math_y in range(0, 10_000, 500)], dtype=float)
    with_outlier = np.vstack([points, [[500_000.0, 500_000.0]]])

    envelope = DirectionalEnvelope.build(with_outlier, n_bins=36, percentile=99.0)
    # The bulk of the grid must still score as peripheral despite the outlier.
    assert envelope.peripherality(0.0, 0.0) > 0.5


@pytest.mark.smoke
def test_envelope_peripherality_rises_towards_the_edge():
    points = np.array([[x, y] for x in range(-10_000, 10_001, 500)
                       for y in range(-10_000, 10_001, 500)], dtype=float)
    envelope = DirectionalEnvelope.build(points)
    assert envelope.peripherality(0.0, 0.0) == pytest.approx(0.0, abs=1e-9)
    assert envelope.peripherality(9_500.0, 0.0) > envelope.peripherality(4_000.0, 0.0)


@pytest.mark.smoke
def test_envelope_compass_octants():
    points = np.array([[x, y] for x in range(-1000, 1001, 100)
                       for y in range(-1000, 1001, 100)], dtype=float)
    envelope = DirectionalEnvelope.build(points)
    assert envelope.compass(1000.0, 0.0) == 'E'
    assert envelope.compass(0.0, 1000.0) == 'N'
    assert envelope.compass(-1000.0, 0.0) == 'W'
    assert envelope.compass(0.0, -1000.0) == 'S'


# ---------------------------------------------------------------------------
# CordonDetector
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_detects_one_gateway_per_arm_and_direction(tmp_path):
    """A four-armed cross has eight gateways: in and out on each arm."""
    network = _cross_network(tmp_path)
    cordons = CordonDetector().detect(network)

    assert len(cordons) == 8
    assert sum(1 for c in cordons if c.direction == INBOUND) == 4
    assert sum(1 for c in cordons if c.direction == OUTBOUND) == 4
    assert {c.compass for c in cordons} == {'E', 'N', 'W', 'S'}


@pytest.mark.smoke
def test_inbound_and_outbound_never_merge(tmp_path):
    """Opposite carriageways of one road stay separate cordons.

    They are metres apart, so any distance-only dedupe would merge them — and
    then an E->I truck could be injected onto an outbound link, travelling the
    wrong way up a divided highway.
    """
    network = _cross_network(tmp_path)
    cordons = CordonDetector(dedupe_radius_m=50_000).detect(network)

    for cordon in cordons:
        assert cordon.direction in (INBOUND, OUTBOUND)
    assert any(c.direction == INBOUND for c in cordons)
    assert any(c.direction == OUTBOUND for c in cordons)


@pytest.mark.smoke
def test_interior_corridor_break_is_not_a_gateway(tmp_path):
    """A corridor that breaks mid-region is a terminus but not a cordon.

    Measured on the real network, 98 of 155 raw termini were breaks like this.
    Peripherality is what separates them from real gateways.
    """
    centre = (500_000.0, 5_000_000.0)
    nodes = {
        'c': centre,
        'inner_a': (centre[0] + 1_000.0, centre[1]),
        'inner_b': (centre[0] + 2_000.0, centre[1]),
        'outer': (centre[0] + 20_000.0, centre[1]),
        'north': (centre[0], centre[1] + 20_000.0),
        'south': (centre[0], centre[1] - 20_000.0),
        'west': (centre[0] - 20_000.0, centre[1]),
    }
    links = [
        # a real gateway on the east arm
        {'id': 'east_in', 'from': 'outer', 'to': 'c'},
        # an interior stub: high capacity, but nowhere near the network edge
        {'id': 'interior', 'from': 'inner_a', 'to': 'inner_b'},
        # arms that give the network its extent
        {'id': 'north_in', 'from': 'north', 'to': 'c'},
        {'id': 'south_in', 'from': 'south', 'to': 'c'},
        {'id': 'west_in', 'from': 'west', 'to': 'c'},
    ]
    network = _write_network(tmp_path / 'interior.xml', nodes, links)

    cordons = CordonDetector().detect(network)
    covered = {link_id for c in cordons for link_id in c.link_ids}
    assert 'interior' not in covered
    assert 'east_in' in covered


@pytest.mark.smoke
def test_local_streets_are_not_cordons(tmp_path):
    """Below the freespeed/capacity floor, a link is not a through corridor."""
    network = _cross_network(tmp_path)
    tree = ET.parse(network)
    for link in tree.getroot().iter('link'):
        if link.get('id').startswith('E_'):
            link.set('freespeed', '11.0')      # ~40 km/h
            link.set('capacity', '600.0')
    tree.write(str(network))

    cordons = CordonDetector().detect(network)
    covered = {link_id for c in cordons for link_id in c.link_ids}
    assert not any(link_id.startswith('E_') for link_id in covered)


@pytest.mark.smoke
def test_non_car_links_are_ignored(tmp_path):
    """A rail-only corridor is not a truck gateway."""
    network = _cross_network(tmp_path)
    tree = ET.parse(network)
    for link in tree.getroot().iter('link'):
        if link.get('id').startswith('N_'):
            link.set('modes', 'rail')
    tree.write(str(network))

    cordons = CordonDetector().detect(network)
    assert 'N' not in {c.compass for c in cordons}


@pytest.mark.smoke
def test_zero_cordons_fails_loudly(tmp_path):
    """Zero freight reported as success is the failure hardest to notice."""
    network = _cross_network(tmp_path)
    with pytest.raises(CordonDetectionError):
        CordonDetector(min_capacity_vph=1_000_000).detect(network)


@pytest.mark.smoke
def test_fail_if_none_found_can_be_disabled(tmp_path):
    network = _cross_network(tmp_path)
    detector = CordonDetector(min_capacity_vph=1_000_000, fail_if_none_found=False)
    assert detector.detect(network) == []


@pytest.mark.smoke
def test_missing_network_raises(tmp_path):
    with pytest.raises(CordonDetectionError):
        CordonDetector().detect(tmp_path / 'does_not_exist.xml')


@pytest.mark.smoke
def test_detection_is_deterministic(tmp_path):
    """Same network, same cordons, same order — the summary must be stable."""
    network = _cross_network(tmp_path)
    first = [c.cordon_id for c in CordonDetector().detect(network)]
    second = [c.cordon_id for c in CordonDetector().detect(network)]
    assert first == second


@pytest.mark.smoke
def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        CordonDetector(min_peripherality=0.0)
    with pytest.raises(ValueError):
        CordonDetector(dedupe_radius_m=-1)
    with pytest.raises(ValueError):
        CordonDetector(envelope_bins=2)


# ---------------------------------------------------------------------------
# Cordon / config plumbing
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_cordon_direction_gates_trip_ends():
    inbound = Cordon('a', 0, 0, INBOUND, 'N')
    outbound = Cordon('b', 0, 0, OUTBOUND, 'S')
    both = Cordon('c', 0, 0, BIDIRECTIONAL, 'E')

    assert inbound.accepts_entry and not inbound.accepts_exit
    assert outbound.accepts_exit and not outbound.accepts_entry
    assert both.accepts_entry and both.accepts_exit


@pytest.mark.smoke
def test_cordon_to_dict_is_json_serialisable():
    import json
    cordon = Cordon('a', 1.234567, 2.0, INBOUND, 'N',
                    link_ids=['1'], capacity=4000.0, peripherality=0.812)
    payload = json.loads(json.dumps(cordon.to_dict()))
    assert payload['cordon_id'] == 'a'
    assert payload['direction'] == INBOUND


@pytest.mark.smoke
def test_detector_from_config_reads_the_freight_block():
    detector = detector_from_config({'freight': {'cordon': {
        'min_peripherality': 0.75,
        'dedupe_radius_m': 900,
        'min_capacity_vph': 2000,
    }}})
    assert detector.min_peripherality == 0.75
    assert detector.dedupe_radius_m == 900
    assert detector.min_capacity_vph == 2000


@pytest.mark.smoke
def test_detector_from_config_defaults_on_empty_config():
    detector = detector_from_config({})
    assert detector.min_peripherality == 0.6
    assert detector.fail_if_none_found is True
