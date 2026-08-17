"""Smoke tests for the freight vehicles.xml writer and PCE config wiring.

The MATSim facts these encode were verified against MATSim source, not assumed:
one vehicle type per network mode, named after the mode; an optional per-person
``vehicleTypes`` attribute overriding it; and startup failing outright if the
mode's type is missing.

See docs/freight/design.md §1 and §7.
"""

import json
import xml.etree.ElementTree as ET

import pytest

from matsim.config_manager import ConfigManager, VEHICLES_SOURCE
from models.freight.vehicles import (
    DEFAULT_PCE,
    build_vehicle_types,
    write_vehicles_file,
)

NS = '{http://www.matsim.org/files/dtd}'


def _config(pce_enabled=True, **freight):
    config = {
        'freight': {
            'enabled': True,
            'mode': 'car',
            'subpopulation': 'freight',
            'vehicle_mix': {'single_unit': 0.45, 'combination': 0.55},
            'pce': {'enabled': pce_enabled, 'single_unit': 1.5,
                    'combination': 2.5, 'car': 1.0},
        },
    }
    config['freight'].update(freight)
    return config


# ---------------------------------------------------------------------------
# vehicle type resolution
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_car_type_is_named_after_the_network_mode():
    """MATSim fails at startup if the mode's type is missing or misnamed."""
    types = build_vehicle_types(_config())
    assert 'car' in types
    assert types['car']['id'] == 'car'
    assert types['car']['network_mode'] == 'car'


@pytest.mark.smoke
def test_freight_type_routes_as_car():
    """A separate network mode is deliberately not used."""
    types = build_vehicle_types(_config())
    assert types['freight']['network_mode'] == 'car'


@pytest.mark.smoke
def test_freight_pce_is_the_mix_weighted_blend():
    types = build_vehicle_types(_config())
    expected = 0.45 * 1.5 + 0.55 * 2.5
    assert types['freight']['pce'] == pytest.approx(expected)


@pytest.mark.smoke
def test_freight_pce_tracks_the_vehicle_mix():
    """An all-combination fleet consumes more capacity than an all-SU one."""
    single = build_vehicle_types(
        _config(vehicle_mix={'single_unit': 1.0, 'combination': 0.0}))
    combination = build_vehicle_types(
        _config(vehicle_mix={'single_unit': 0.0, 'combination': 1.0}))

    assert single['freight']['pce'] == pytest.approx(1.5)
    assert combination['freight']['pce'] == pytest.approx(2.5)


@pytest.mark.smoke
def test_vehicle_mix_is_normalised():
    types = build_vehicle_types(
        _config(vehicle_mix={'single_unit': 45, 'combination': 55}))
    assert types['freight']['pce'] == pytest.approx(0.45 * 1.5 + 0.55 * 2.5)


@pytest.mark.smoke
def test_degenerate_mix_falls_back_to_defaults():
    types = build_vehicle_types(
        _config(vehicle_mix={'single_unit': 0.0, 'combination': 0.0}))
    assert types['freight']['pce'] > 1.0


@pytest.mark.smoke
def test_car_pce_is_one_by_default():
    assert build_vehicle_types(_config())['car']['pce'] == pytest.approx(1.0)
    assert DEFAULT_PCE['car'] == 1.0


@pytest.mark.smoke
def test_freight_is_longer_than_a_car():
    types = build_vehicle_types(_config())
    assert types['freight']['length'] > types['car']['length']


# ---------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_no_file_when_pce_is_off(tmp_path):
    """Stages 1-5 run at pce=1.0, so no vehicles.xml should appear."""
    path = tmp_path / 'vehicles.xml'
    assert write_vehicles_file(_config(pce_enabled=False), path) is None
    assert not path.exists()


@pytest.mark.smoke
def test_no_file_when_freight_is_off(tmp_path):
    path = tmp_path / 'vehicles.xml'
    config = _config()
    config['freight']['enabled'] = False
    assert write_vehicles_file(config, path) is None
    assert not path.exists()


@pytest.mark.smoke
def test_written_file_has_both_types_with_pce(tmp_path):
    path = tmp_path / 'vehicles.xml'
    write_vehicles_file(_config(), path)

    root = ET.parse(path).getroot()
    types = {t.get('id'): t for t in root.findall(f'{NS}vehicleType')}

    assert set(types) == {'car', 'freight'}
    for type_id, element in types.items():
        pce = element.find(f'{NS}passengerCarEquivalents')
        network_mode = element.find(f'{NS}networkMode')
        assert pce is not None and float(pce.get('pce')) > 0
        assert network_mode.get('networkMode') == 'car'

    assert float(types['freight'].find(f'{NS}passengerCarEquivalents').get('pce')) > \
        float(types['car'].find(f'{NS}passengerCarEquivalents').get('pce'))


@pytest.mark.smoke
def test_written_file_is_valid_xml_with_a_declaration(tmp_path):
    path = tmp_path / 'vehicles.xml'
    write_vehicles_file(_config(), path)
    text = path.read_text(encoding='utf-8')

    assert text.startswith('<?xml')
    ET.parse(path)


# ---------------------------------------------------------------------------
# config wiring
# ---------------------------------------------------------------------------

def _template(tmp_path):
    """A config tree with the modules configure_freight expects."""
    root = ET.Element('config')
    for name in ('controller', 'qsim', 'hermes', 'replanning'):
        ET.SubElement(root, 'module', name=name)
    scoring = ET.SubElement(root, 'module', name='scoring')
    ET.SubElement(scoring, 'parameterset', type='scoringParameters')
    strategy = ET.SubElement(
        [m for m in root.findall('module') if m.get('name') == 'replanning'][0],
        'parameterset', type='strategysettings')
    ET.SubElement(strategy, 'param', name='strategyName', value='SelectExpBeta')
    ET.SubElement(strategy, 'param', name='weight', value='0.6')
    return ET.ElementTree(root)


@pytest.mark.smoke
def test_pce_enabled_registers_the_vehicles_module(tmp_path):
    tree = _template(tmp_path)
    ConfigManager(_config()).configure_freight(tree, last_iteration=10)

    root = tree.getroot()
    vehicles = [m for m in root.findall('module') if m.get('name') == 'vehicles']
    assert vehicles, "vehicles module was not created"
    assert any(p.get('name') == 'vehiclesFile' and p.get('value') == 'vehicles.xml'
               for p in vehicles[0].findall('param'))


@pytest.mark.smoke
def test_pce_sets_vehicles_source_on_qsim_only_never_hermes(tmp_path):
    """``vehiclesSource`` is a QSim parameter, and hermes rejects it outright.

    This test previously asserted the opposite — that both mobsims get the
    parameter — and that is what let the bug ship. MATSim 25 aborts at
    config-parse time with "Module hermes ... doesn't accept unknown
    parameters. Parameter vehiclesSource is not part of the valid parameters:
    [mainMode, stuckTime, flowCapacityFactor, storageCapacityFactor, endTime,
    useDeterministicPt]", because ``HermesConfigGroup`` is a
    ``ReflectiveConfigGroup`` and treats an unknown parameter as fatal. Since
    this pipeline runs ``mobsim=hermes``, writing it there fails every PCE run
    one second after launch.

    Hermes still applies PCE: ``ScenarioImporter`` builds ``flowCapacityPCEs``
    from ``getVehicleTypes()``, which ``vehicles.vehiclesFile`` populates.
    """
    tree = _template(tmp_path)
    ConfigManager(_config()).configure_freight(tree, last_iteration=10)

    root = tree.getroot()
    qsim = [m for m in root.findall('module') if m.get('name') == 'qsim'][0]
    assert any(p.get('name') == 'vehiclesSource'
               and p.get('value') == VEHICLES_SOURCE
               for p in qsim.findall('param'))

    hermes = [m for m in root.findall('module') if m.get('name') == 'hermes'][0]
    assert not any(p.get('name') == 'vehiclesSource'
                   for p in hermes.findall('param')), \
        "vehiclesSource on hermes aborts MATSim at config-parse time"


@pytest.mark.smoke
def test_pce_disabled_leaves_no_vehicles_module(tmp_path):
    tree = _template(tmp_path)
    ConfigManager(_config(pce_enabled=False)).configure_freight(tree, last_iteration=10)

    root = tree.getroot()
    assert not [m for m in root.findall('module') if m.get('name') == 'vehicles']


def _strategy_blocks(tree):
    """(strategyName, subpopulation-or-None) for every strategysettings block."""
    replanning = [m for m in tree.getroot().findall('module')
                  if m.get('name') == 'replanning'][0]
    blocks = []
    for ps in replanning.findall("parameterset[@type='strategysettings']"):
        name = ps.find("param[@name='strategyName']")
        sub = ps.find("param[@name='subpopulation']")
        blocks.append((name.get('value') if name is not None else None,
                       sub.get('value') if sub is not None else None))
    return blocks


@pytest.mark.smoke
def test_passenger_strategies_stay_untagged(tmp_path):
    """The bug that killed the first real freight run.

    Tagging the passenger strategies `subpopulation="default"` mirrors the
    freight block and looks right, but MATSim reads an agent's subpopulation
    from a person attribute — and passenger persons have none, so it sees
    `null`, not `"default"`. The run then dies at the first replanning step
    with "No strategy found! ... Current subpopulation = null".

    An untagged block is the catch-all that actually matches those agents.
    """
    tree = _template(tmp_path)
    ConfigManager(_config()).configure_freight(tree, last_iteration=10)

    blocks = _strategy_blocks(tree)
    untagged = [name for name, sub in blocks if sub is None]
    assert 'SelectExpBeta' in untagged, (
        f"passenger strategies must stay untagged so null-subpopulation agents "
        f"match them; got {blocks}")
    assert not any(sub == 'default' for _, sub in blocks), (
        f"no strategy may be tagged 'default' — no agent carries that label; "
        f"got {blocks}")


@pytest.mark.smoke
def test_freight_strategies_are_tagged_freight(tmp_path):
    """Freight *is* labelled, so its strategies must be tagged to reach it."""
    tree = _template(tmp_path)
    ConfigManager(_config()).configure_freight(tree, last_iteration=10)

    freight = {name for name, sub in _strategy_blocks(tree) if sub == 'freight'}
    assert freight == {'ChangeExpBeta', 'ReRoute'}


@pytest.mark.smoke
def test_freight_never_gets_mode_or_time_mutation(tmp_path):
    """SubtourModeChoice could turn a truck into a pedestrian;
    TimeAllocationMutator would drift it off the sampled departure profile."""
    tree = _template(tmp_path)
    ConfigManager(_config()).configure_freight(tree, last_iteration=10)

    freight = {name for name, sub in _strategy_blocks(tree) if sub == 'freight'}
    assert 'SubtourModeChoice' not in freight
    assert 'TimeAllocationMutator' not in freight


@pytest.mark.smoke
def test_person_attribute_names_the_freight_type_when_pce_is_on():
    """The per-person override MATSim looks for, in the shape it expects."""
    from models.freight.generator import FreightTrip
    from models.freight.plans import trips_to_plans

    trip = FreightTrip('through', 33.5, -86.8, 33.6, -86.7, 3600.0)
    plans = trips_to_plans([trip], _config())
    attributes = {a.name: a for a in plans[0].attributes}

    assert 'vehicleTypes' in attributes
    assert attributes['vehicleTypes'].java_class == \
        'org.matsim.vehicles.PersonVehicleTypes'
    assert json.loads(attributes['vehicleTypes'].value) == {'car': 'freight'}


@pytest.mark.smoke
def test_no_vehicle_type_attribute_when_pce_is_off():
    """Naming a type that vehicles.xml does not define would fail at startup."""
    from models.freight.generator import FreightTrip
    from models.freight.plans import trips_to_plans

    trip = FreightTrip('through', 33.5, -86.8, 33.6, -86.7, 3600.0)
    plans = trips_to_plans([trip], _config(pce_enabled=False))
    names = {a.name for a in plans[0].attributes}

    assert 'vehicleTypes' not in names
    assert 'subpopulation' in names
