"""Freight trips -> MATSim plans, and the end-to-end entry point.

A freight plan is two activities and one leg, carrying two person attributes:
``subpopulation`` (so the replanning strategies written in stage 1 apply to it)
and ``vehicleTypes`` (so PCE applies once stage 6 turns it on).

See docs/freight/design.md §1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from models.freight.cordons import (
    WEIGHT_BY_CAPACITY,
    detector_from_config,
    weight_by_from_config,
)
from models.freight.demand import assign_cordon_weights, resolve_demand
from models.freight.hpms_match import resolve_truck_aadt_by_link
from models.freight.departure import DepartureSampler, seconds_to_hms
from models.freight.generator import (
    ACTIVITY_DESTINATION,
    ACTIVITY_ORIGIN,
    FreightGenerationError,
    FreightTrip,
    FreightTripGenerator,
    Zone,
    anchor_cordons_to_zones,
)
from models.freight.truck_share import resolve_truck_share
from utils.logger import setup_logger

logger = setup_logger(__name__)


def trips_to_plans(trips: Sequence[FreightTrip], config: Dict) -> List:
    """Build Plan objects from freight trips.

    Imported lazily: plan_generator pulls in the whole modelling stack, and the
    freight package is otherwise importable without it.
    """
    from models.plan_generator import Activity, PersonAttribute, Plan
    from models.mode_choice import Leg

    freight_config = config.get('freight', {})
    subpopulation = freight_config.get('subpopulation', 'freight')
    mode = freight_config.get('mode', 'car')
    pce_config = freight_config.get('pce', {})
    pce_enabled = pce_config.get('enabled', False)

    plans = []
    for index, trip in enumerate(trips):
        attributes = [PersonAttribute('subpopulation', subpopulation)]
        if pce_enabled:
            # Names the vehicle type MATSim should substitute for this person.
            # Harmless while PCE is off, but it would name a type that does not
            # exist in vehicles.xml, so it is only written when PCE is on.
            attributes.append(PersonAttribute(
                'vehicleTypes',
                json.dumps({mode: subpopulation}),
                'org.matsim.vehicles.PersonVehicleTypes',
            ))

        plans.append(Plan(
            person_id=f"freight_{index}",
            activities=[
                Activity(
                    type=ACTIVITY_ORIGIN,
                    x=trip.origin_lon,
                    y=trip.origin_lat,
                    end_time=seconds_to_hms(trip.departure_seconds),
                ),
                Activity(
                    type=ACTIVITY_DESTINATION,
                    x=trip.dest_lon,
                    y=trip.dest_lat,
                ),
            ],
            legs=[Leg(mode=mode)],
            attributes=attributes,
        ))

    logger.info(f"Built {len(plans):,} freight plans (subpopulation={subpopulation})")
    return plans


#: What ``freight.internal_attractor`` may be set to.
#:
#: ``employment`` is total employment (LODES ``C000``) — every job weighted the
#: same, so a warehouse and an insurance office of equal headcount attract equal
#: freight, which is plainly wrong but was all the database held.
#:
#: ``freight_employment`` is the weighted sum over the goods-producing and
#: goods-handling ``CNS*`` sectors, which arrive in the same LODES download.
#: Memphis regresses truck trip ends on sector employment with R^2 0.77-1.00.
#: See ``work_locs_v2.FREIGHT_SECTORS``.
#:
#: Validated rather than ignored — a config naming an attractor that does not
#: exist should say so, not silently fall back to a different one.
ATTRACTOR_EMPLOYMENT = 'employment'
ATTRACTOR_FREIGHT_EMPLOYMENT = 'freight_employment'
VALID_ATTRACTORS = (ATTRACTOR_EMPLOYMENT, ATTRACTOR_FREIGHT_EMPLOYMENT)


def load_zones(config: Dict, converter) -> List[Zone]:
    """Load internal zones with their freight attractor.

    ``employment`` weights every job the same. ``freight_employment`` weights
    the goods-producing and goods-handling sectors, which is the difference
    between a distribution centre and an office park of the same headcount.

    The sector column is nullable: a database populated before it existed has
    no value for it, and re-downloading LODES must not be a precondition for
    running freight. So a request for ``freight_employment`` that the data
    cannot serve falls back to total employment **loudly**, rather than
    silently distributing every truck to zero-attractor zones.
    """
    from models.work_locs_v2 import load_work_locations_by_counties

    attractor = config.get('freight', {}).get(
        'internal_attractor', ATTRACTOR_EMPLOYMENT)
    if attractor not in VALID_ATTRACTORS:
        raise FreightGenerationError(
            f"freight.internal_attractor={attractor!r} is not implemented. "
            f"Valid values: {VALID_ATTRACTORS}."
        )

    work_locations = load_work_locations_by_counties(config)

    resolved = attractor
    if attractor == ATTRACTOR_FREIGHT_EMPLOYMENT:
        available = sum(
            1 for record in work_locations.values()
            if record.get('n_employees_freight') is not None)
        if available == 0:
            logger.warning(
                "freight.internal_attractor='freight_employment' was requested, "
                "but no work location carries sector employment — this database "
                "predates the CNS* columns. Falling back to total employment. "
                "Re-run the work-location ETL to populate it."
            )
            resolved = ATTRACTOR_EMPLOYMENT
        elif available < len(work_locations):
            logger.warning(
                f"Only {available:,}/{len(work_locations):,} work locations "
                f"carry sector employment; the rest contribute 0 to the freight "
                f"attractor. Re-run the work-location ETL for full coverage."
            )

    field = ('n_employees_freight' if resolved == ATTRACTOR_FREIGHT_EMPLOYMENT
             else 'n_employees')

    zones: List[Zone] = []
    for geoid, record in work_locations.items():
        lat, lon = record.get('lat'), record.get('lon')
        if lat is None or lon is None:
            continue
        x, y = converter.latlon_to_utm(lat, lon)
        zones.append(Zone(
            zone_id=geoid,
            x=x,
            y=y,
            lat=float(lat),
            lon=float(lon),
            attractor=float(record.get(field, 0) or 0),
        ))

    if not zones:
        raise FreightGenerationError(
            "No internal zones with coordinates were loaded; freight has "
            "nowhere to start or stop. Check that work locations are populated "
            "for the configured counties."
        )

    total_attractor = sum(z.attractor for z in zones)
    logger.info(
        f"Loaded {len(zones):,} freight zones, total attractor "
        f"{total_attractor:,.0f} ({resolved})"
    )
    if total_attractor <= 0:
        # Not fatal — the generator falls back to distance-only sampling — but
        # it means every internal trip end is placed by distance alone, which
        # is worth saying plainly rather than leaving in a debug log.
        logger.warning(
            f"Freight attractor '{resolved}' sums to zero across every zone; "
            f"internal trip ends will be distributed by distance alone."
        )
    return zones


def generate_freight_plans(
    config: Dict,
    network_path: Path,
    car_trips: Optional[int] = None,
    summary_path: Optional[Path] = None,
) -> List:
    """Generate the freight stream end to end.

    Returns a list of Plan objects ready to be combined with the work and
    non-work streams. Writes ``freight_summary.json`` when a path is given.
    """
    freight_config = config.get('freight', {})
    if not freight_config.get('enabled', False):
        return []

    from utils.coordinates import CoordinateConverter

    logger.info("=" * 60)
    logger.info("GENERATING BOUNDARY FREIGHT")
    logger.info("=" * 60)

    utm_epsg = config['coordinates']['utm_epsg']
    converter = CoordinateConverter(utm_epsg=utm_epsg)
    rng = np.random.default_rng(
        config.get('plan_generation', {}).get('random_seed', 42))

    # 1. Where trucks cross the boundary.
    detector = detector_from_config(config)
    cordons = detector.detect(Path(network_path))

    # 2. Which zones they start or stop at, and how far the cordons are from one.
    zones = load_zones(config, converter)
    anchor_stats = anchor_cordons_to_zones(cordons, zones)

    # 3. How much freight this region carries.
    region = config.get('region', {})
    counties = region.get('counties', [])
    state_fips = counties[0][:2] if counties else '01'
    county_codes = [geoid[2:5] for geoid in counties]
    truck_share = resolve_truck_share(
        config, state_fips, county_codes,
        state_abbr=_state_abbr(state_fips),
    )

    # Observed truck volume per cordon, where HPMS has a segment for it. This
    # is what makes the spatial distribution data-driven rather than
    # capacity-driven; it is total by contract, so an unreachable service simply
    # leaves every cordon on the capacity fallback.
    weight_by = weight_by_from_config(config)
    truck_aadt_by_link: Dict[str, float] = {}
    match_stats: Dict = {'skipped': "weight_by='capacity'"}
    if weight_by != WEIGHT_BY_CAPACITY:
        truck_aadt_by_link, match_stats = resolve_truck_aadt_by_link(
            config, cordons, Path(network_path), converter)

    weight_stats = assign_cordon_weights(
        cordons,
        truck_aadt_by_link=truck_aadt_by_link,
        truck_share=truck_share,
        weight_by=weight_by,
    )
    weight_stats['hpms_match'] = match_stats
    demand = resolve_demand(config, cordons, truck_share=truck_share,
                            car_trips=car_trips, rng=rng)

    # 4. The trips themselves.
    generator = FreightTripGenerator(cordons, zones, config, rng=rng)
    generator.set_converter(converter)
    trips = generator.generate(demand)

    plans = trips_to_plans(trips, config)

    if summary_path is not None:
        write_summary(
            summary_path,
            cordons=cordons,
            detector=detector,
            demand=demand,
            truck_share=truck_share,
            trips=trips,
            anchor_stats=anchor_stats,
            weight_stats=weight_stats,
            config=config,
        )

    return plans


def write_summary(
    path: Path,
    *,
    cordons,
    detector,
    demand,
    truck_share,
    trips,
    anchor_stats,
    weight_stats,
    config: Dict,
) -> None:
    """Write freight_summary.json.

    The recommender reads this the same way it reads every other summary, and
    the tier-1 consistency checks are computed from it, so it carries the
    *realised* distribution alongside the configured one rather than assuming
    they match.
    """
    from models.freight.validation import summarise, validate_tier1

    sampler = DepartureSampler.from_config(config)
    realised = sampler.realised_distribution([t.departure_seconds for t in trips])

    class_counts: Dict[str, int] = {}
    for trip in trips:
        class_counts[trip.trip_class] = class_counts.get(trip.trip_class, 0) + 1

    # Tier 1 runs here rather than in a separate step: it needs exactly the
    # objects this function already holds, costs milliseconds, and catches a
    # generator regression at the point it happens rather than after a
    # simulation has burned an hour.
    tier1 = validate_tier1(trips, demand, cordons, config=config)
    tier1.log()
    if not tier1.passed:
        logger.warning(
            "Freight tier-1 consistency checks FAILED. The generated demand "
            "does not match what was configured — see freight_summary.json."
        )

    cordon_usage: Dict[str, int] = {}
    for trip in trips:
        for cordon_id in (trip.origin_cordon, trip.dest_cordon):
            if cordon_id:
                cordon_usage[cordon_id] = cordon_usage.get(cordon_id, 0) + 1

    payload = {
        'enabled': True,
        'n_trips': len(trips),
        'demand': demand.to_dict(),
        'truck_share': truck_share.to_dict(),
        'cordons': {
            **detector.describe(cordons),
            'anchoring': anchor_stats,
            'weighting': weight_stats,
            'detail': [c.to_dict() for c in cordons],
        },
        'realised': {
            'class_counts': class_counts,
            'class_shares': {k: round(v / len(trips), 4)
                             for k, v in class_counts.items()} if trips else {},
            'departure_distribution': [round(v, 5) for v in realised],
            'cordon_usage': dict(sorted(cordon_usage.items(),
                                        key=lambda kv: -kv[1])),
        },
        'configured': {
            'departure_distribution': {
                name: [round(float(v), 5) for v in profile]
                for name, profile in sampler.profiles.items()
            },
        },
        'validation': summarise([tier1]),
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=_json_default),
                        encoding='utf-8')
        logger.info(f"Freight summary written to {path}")
    except Exception as exc:  # noqa: BLE001 - reporting must not fail a run
        logger.warning(f"Could not write freight summary (run unaffected): {exc}")


def _json_default(value):
    """Coerce numpy scalars and arrays that slipped through into JSON types.

    Values here come from numpy-heavy sampling code, and a single
    ``numpy.bool_`` or ``numpy.int64`` anywhere in the payload raises at
    serialisation time and costs the whole summary. Converting at the boundary
    is more robust than chasing each producer, and anything genuinely
    unserialisable still becomes a readable string rather than an exception.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _state_abbr(state_fips: str) -> Optional[str]:
    """Map a state FIPS code to its postal abbreviation."""
    from data_sources.fha_counts_manager import STATE_FIPS_TO_ABBR
    return STATE_FIPS_TO_ABBR.get(state_fips)
