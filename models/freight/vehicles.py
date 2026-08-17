"""vehicles.xml — passenger-car equivalents for the freight stream.

**PCE does not need a separate network mode.** In MATSim it is a property of the
*vehicle type*, not of the mode, so trucks can route as ``car`` while consuming
the road capacity of a truck. The reference scenario ``equil-mixedTraffic``
demonstrates exactly this pattern.

**And it needs no per-agent vehicle list.** Verified against MATSim source,
``PrepareForSimImpl.createAndAddVehiclesForEveryNetworkMode()``: MATSim
auto-creates one vehicle per person per network mode and looks the type up *by
mode name*, so a single ``<vehicleType id="car">`` covers the entire passenger
population. It then checks an optional per-person ``vehicleTypes`` attribute and
substitutes that type when present. Freight persons set it; everyone else omits
it and silently gets ``car``.

That is the difference between the three ``vehiclesSource`` values:

  ``defaultVehicle``                  nothing to supply, no PCE control
  ``modeVehicleTypesFromVehiclesData`` one type per network mode, named after
                                       the mode; per-person override optional
  ``fromVehiclesData``                 every vehicle enumerated by hand

We use the middle one. One caveat follows from the same source: the type for
each mode must be **named exactly after the mode** (``id="car"``), or startup
fails with "Could not find requested vehicle type."

**This is deliberately staged last and must be its own run.** PCE changes
congestion, which changes routes, which changes counts on links that carry no
trucks at all. Introduced alongside new demand, neither effect is attributable.

See docs/freight/design.md §1 and §7.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional

from utils.logger import setup_logger

logger = setup_logger(__name__)

VEHICLES_NAMESPACE = 'http://www.matsim.org/files/dtd'
VEHICLES_SCHEMA = 'http://www.matsim.org/files/dtd/vehicleDefinitions_v2.0.xsd'

#: The vehiclesSource that reads one type per network mode.
VEHICLES_SOURCE = 'modeVehicleTypesFromVehiclesData'

#: Defaults sit in the FHWA passenger-car-equivalent range for level terrain.
#: network_generator already uses pce=2.8 for buses, so both the machinery and
#: the magnitude are consistent with existing practice in this repo.
DEFAULT_PCE = {'car': 1.0, 'single_unit': 1.5, 'combination': 2.5}

#: Physical lengths, which MATSim uses for queue occupancy alongside PCE.
DEFAULT_LENGTH_M = {'car': 7.5, 'single_unit': 9.0, 'combination': 15.0}


def build_vehicle_types(config: Dict) -> Dict[str, Dict]:
    """Resolve the vehicle types to write, from the freight config.

    The freight type is a **blend** of single-unit and combination weighted by
    ``vehicle_mix``, rather than two separate types. One type per mode is what
    ``modeVehicleTypesFromVehiclesData`` addresses, and a per-person override
    can only name one type; splitting trucks into two types would mean tagging
    each person with which kind it is, for a difference of about 0.4 PCE. The
    mix is recorded so the blend is auditable.
    """
    freight_config = config.get('freight', {})
    pce_config = freight_config.get('pce', {})
    mix = freight_config.get('vehicle_mix', {}) or {}

    single_share = float(mix.get('single_unit', 0.45))
    combination_share = float(mix.get('combination', 0.55))
    total_share = single_share + combination_share
    if total_share <= 0:
        single_share, combination_share, total_share = 0.45, 0.55, 1.0
    single_share /= total_share
    combination_share /= total_share

    single_pce = float(pce_config.get('single_unit', DEFAULT_PCE['single_unit']))
    combination_pce = float(pce_config.get('combination', DEFAULT_PCE['combination']))
    car_pce = float(pce_config.get('car', DEFAULT_PCE['car']))

    freight_pce = single_share * single_pce + combination_share * combination_pce
    freight_length = (single_share * DEFAULT_LENGTH_M['single_unit']
                      + combination_share * DEFAULT_LENGTH_M['combination'])

    subpopulation = freight_config.get('subpopulation', 'freight')
    mode = freight_config.get('mode', 'car')

    return {
        # MUST be named after the network mode, or MATSim fails at startup.
        mode: {
            'id': mode,
            'pce': car_pce,
            'length': DEFAULT_LENGTH_M['car'],
            'network_mode': mode,
        },
        subpopulation: {
            'id': subpopulation,
            'pce': round(freight_pce, 4),
            'length': round(freight_length, 2),
            'network_mode': mode,      # routes as car
            'mix': {'single_unit': round(single_share, 4),
                    'combination': round(combination_share, 4)},
        },
    }


def write_vehicles_file(config: Dict, output_path: Path) -> Optional[Path]:
    """Write vehicles.xml for the run, or None when PCE is off.

    Kept separate from the transit vehicles file: MATSim reads them through two
    different config parameters (``vehicles.vehiclesFile`` and
    ``transit.vehiclesFile``), so they do not collide, and coupling freight to
    the transit pipeline would break it whenever transit is disabled.
    """
    freight_config = config.get('freight', {})
    if not freight_config.get('enabled', False):
        return None
    if not freight_config.get('pce', {}).get('enabled', False):
        logger.info("Freight PCE is off; no vehicles.xml written (pce=1.0 run)")
        return None

    types = build_vehicle_types(config)
    output_path = Path(output_path)

    ET.register_namespace('', VEHICLES_NAMESPACE)
    root = ET.Element(f'{{{VEHICLES_NAMESPACE}}}vehicleDefinitions')
    root.set('{http://www.w3.org/2001/XMLSchema-instance}schemaLocation',
             f'{VEHICLES_NAMESPACE} {VEHICLES_SCHEMA}')

    for definition in types.values():
        element = ET.SubElement(root, f'{{{VEHICLES_NAMESPACE}}}vehicleType',
                                {'id': definition['id']})
        ET.SubElement(element, f'{{{VEHICLES_NAMESPACE}}}length',
                      {'meter': str(definition['length'])})
        ET.SubElement(element, f'{{{VEHICLES_NAMESPACE}}}passengerCarEquivalents',
                      {'pce': str(definition['pce'])})
        ET.SubElement(element, f'{{{VEHICLES_NAMESPACE}}}networkMode',
                      {'networkMode': definition['network_mode']})

    tree = ET.ElementTree(root)
    ET.indent(tree, space='    ')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_path), encoding='UTF-8', xml_declaration=True)

    described = ', '.join(f"{d['id']} pce={d['pce']}" for d in types.values())
    logger.info(f"Wrote {output_path.name}: {described}")
    return output_path
