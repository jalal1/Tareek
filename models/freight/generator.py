"""Boundary freight trip generation — the three classes.

Produces MATSim plans of two activities and one leg. No chain sampling, no POI
assignment, no mode choice, no duration model: a freight trip is an origin, a
destination and a departure time.

The distribution is **singly-constrained** and that is deliberate. The obvious
reuse, ``create_local_od_matrix``, is a doubly-constrained gravity model with
IPF, and it needs ``n_employees`` on *both* ends to build its margins. A cordon
has an inbound truck total, not a population, so IPF has nothing to converge
against on that side. What is actually needed is: distribute a known cordon
total across internal zones by attractiveness and distance. That is one weighted
sample, and ``compute_friction`` supplies the distance term.

See docs/freight/design.md §2 and §4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from models.freight.cordons import Cordon
from models.freight.demand import (
    CLASS_EXTERNAL_TO_INTERNAL,
    CLASS_INTERNAL_TO_EXTERNAL,
    CLASS_THROUGH,
    FreightDemand,
)
from models.freight.departure import DepartureSampler, seconds_to_hms
from models.od_matrix_v3 import compute_friction, FRICTION_EXPONENTIAL
from utils.logger import setup_logger

logger = setup_logger(__name__)

ACTIVITY_ORIGIN = 'freight_origin'
ACTIVITY_DESTINATION = 'freight_destination'


class FreightGenerationError(RuntimeError):
    """Raised when trips cannot be generated at all."""


@dataclass
class Zone:
    """An internal zone a freight trip can start or end at.

    Attributes:
        zone_id: the block-group GEOID.
        x, y: centroid in network coordinates (metres), so distances to cordons
            are computed in the same space the cordons live in.
        lat, lon: the same point in EPSG:4326, which is what a Plan carries.
        attractor: how much freight this zone draws. Total employment in v1;
            sector employment is a known, cheap upgrade (the LODES WAC download
            already contains the CNS* columns that work_locs_v2 discards).
    """
    zone_id: str
    x: float
    y: float
    lat: float
    lon: float
    attractor: float


@dataclass
class FreightTrip:
    """One generated boundary truck trip."""
    trip_class: str
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float
    departure_seconds: float
    origin_cordon: Optional[str] = None
    dest_cordon: Optional[str] = None
    origin_zone: Optional[str] = None
    dest_zone: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'trip_class': self.trip_class,
            'departure': seconds_to_hms(self.departure_seconds),
            'origin_cordon': self.origin_cordon,
            'dest_cordon': self.dest_cordon,
            'origin_zone': self.origin_zone,
            'dest_zone': self.dest_zone,
        }


def anchor_cordons_to_zones(
    cordons: Sequence[Cordon],
    zones: Sequence[Zone],
) -> Dict:
    """Attach each cordon to its nearest internal zone.

    A cordon sits where the road leaves the network, which may be well away from
    any block group — the zone system covers the demand region, the network
    covers more than that. The trip still has to start at the road, so the
    cordon keeps its own coordinates; the zone link exists to record *how far
    off* the nearest demand zone is.

    That distance is the diagnostic the design asks for: a large value means the
    corridor is being loaded from the wrong place, and it is the first thing to
    look at when a freeway is short of traffic.

    Modifies the cordons in place. Returns summary statistics.
    """
    if not zones:
        raise FreightGenerationError(
            "Cannot anchor cordons: no internal zones were loaded"
        )

    zone_points = np.array([[z.x, z.y] for z in zones], dtype=float)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(zone_points)
        query_points = np.array([[c.x, c.y] for c in cordons], dtype=float)
        distances, indices = tree.query(query_points)
    except ImportError:  # pragma: no cover - scipy is a hard dependency already
        distances, indices = _brute_force_nearest(cordons, zone_points)

    for cordon, distance, index in zip(cordons, distances, indices):
        cordon.zone_id = zones[int(index)].zone_id
        cordon.anchor_distance_m = float(distance)

    anchor_distances = [c.anchor_distance_m for c in cordons]
    stats = {
        'n_cordons': len(cordons),
        'anchor_distance_m': {
            'min': round(min(anchor_distances), 1),
            'median': round(float(np.median(anchor_distances)), 1),
            'max': round(max(anchor_distances), 1),
        },
    }
    logger.info(
        f"Anchored {len(cordons)} cordons to zones: median "
        f"{stats['anchor_distance_m']['median']:.0f} m, max "
        f"{stats['anchor_distance_m']['max']:.0f} m"
    )
    return stats


def _brute_force_nearest(cordons, zone_points):
    """Fallback nearest-neighbour when scipy is unavailable."""
    distances, indices = [], []
    for cordon in cordons:
        deltas = zone_points - np.array([cordon.x, cordon.y])
        squared = (deltas ** 2).sum(axis=1)
        best = int(squared.argmin())
        distances.append(math.sqrt(squared[best]))
        indices.append(best)
    return np.array(distances), np.array(indices)


class FreightTripGenerator:
    """Turns cordons, zones and a demand total into freight trips."""

    def __init__(
        self,
        cordons: Sequence[Cordon],
        zones: Sequence[Zone],
        config: Optional[Dict] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        config = config or {}
        freight_config = config.get('freight', {})
        od_config = freight_config.get('od_matrix', {})

        self.cordons = list(cordons)
        self.zones = list(zones)
        self.rng = rng or np.random.default_rng(
            config.get('plan_generation', {}).get('random_seed', 42))
        self.departures = DepartureSampler.from_config(config)

        self.friction_form = od_config.get('friction', FRICTION_EXPONENTIAL)
        # Beta is in **1/km**, not the QRFM 1/minute. QRFM Table 53 publishes
        # -0.030/min for heavy trucks; at ~80 km/h motorway running that is
        # about -0.0225/km. Passing the per-minute value straight through would
        # make the decay roughly 45x too strong.
        self.beta = float(od_config.get('beta', 0.02))

        if not self.cordons:
            raise FreightGenerationError("No cordons supplied; cannot generate freight")
        if not self.zones:
            raise FreightGenerationError("No zones supplied; cannot generate freight")

        self._entry_cordons = [c for c in self.cordons if c.accepts_entry]
        self._exit_cordons = [c for c in self.cordons if c.accepts_exit]

    # -- public API ---------------------------------------------------------

    def generate(self, demand: FreightDemand) -> List[FreightTrip]:
        """Generate the full trip list for a resolved demand."""
        trips: List[FreightTrip] = []

        trips.extend(self._generate_external_to_internal(
            demand.trips_by_class.get(CLASS_EXTERNAL_TO_INTERNAL, 0)))
        trips.extend(self._generate_internal_to_external(
            demand.trips_by_class.get(CLASS_INTERNAL_TO_EXTERNAL, 0)))
        trips.extend(self._generate_through(
            demand.trips_by_class.get(CLASS_THROUGH, 0)))

        logger.info(f"Generated {len(trips):,} freight trips")
        return trips

    # -- the three classes --------------------------------------------------

    def _generate_external_to_internal(self, n_trips: int) -> List[FreightTrip]:
        """Enters at an inbound cordon, stops at an internal zone."""
        if n_trips <= 0:
            return []
        if not self._entry_cordons:
            logger.warning(
                "No inbound cordon available; skipping E->I trips. Trucks would "
                "otherwise be injected travelling the wrong way."
            )
            return []

        cordons = self._sample_cordons(self._entry_cordons, n_trips)
        departures = self.departures.sample(
            CLASS_EXTERNAL_TO_INTERNAL, self.rng, n_trips)

        trips = []
        for cordon, departure in zip(cordons, departures):
            zone = self._sample_zone_near(cordon)
            trips.append(FreightTrip(
                trip_class=CLASS_EXTERNAL_TO_INTERNAL,
                origin_lat=self._cordon_lat(cordon),
                origin_lon=self._cordon_lon(cordon),
                dest_lat=zone.lat,
                dest_lon=zone.lon,
                departure_seconds=float(departure),
                origin_cordon=cordon.cordon_id,
                dest_zone=zone.zone_id,
            ))
        return trips

    def _generate_internal_to_external(self, n_trips: int) -> List[FreightTrip]:
        """Starts at an internal zone, leaves by an outbound cordon."""
        if n_trips <= 0:
            return []
        if not self._exit_cordons:
            logger.warning("No outbound cordon available; skipping I->E trips.")
            return []

        cordons = self._sample_cordons(self._exit_cordons, n_trips)
        departures = self.departures.sample(
            CLASS_INTERNAL_TO_EXTERNAL, self.rng, n_trips)

        trips = []
        for cordon, departure in zip(cordons, departures):
            zone = self._sample_zone_near(cordon)
            trips.append(FreightTrip(
                trip_class=CLASS_INTERNAL_TO_EXTERNAL,
                origin_lat=zone.lat,
                origin_lon=zone.lon,
                dest_lat=self._cordon_lat(cordon),
                dest_lon=self._cordon_lon(cordon),
                departure_seconds=float(departure),
                origin_zone=zone.zone_id,
                dest_cordon=cordon.cordon_id,
            ))
        return trips

    def _generate_through(self, n_trips: int) -> List[FreightTrip]:
        """Enters at one cordon and leaves by a different one.

        Pairs are weighted by the product of the two cordons' volumes, then
        rejected when the straight line between them does not cross the region
        interior: two adjacent cordons on the same side would otherwise produce
        a short hop along the boundary instead of a run across the region.
        """
        if n_trips <= 0:
            return []
        if len(self._entry_cordons) < 1 or len(self._exit_cordons) < 1:
            logger.warning("Too few cordons for E->E trips; skipping.")
            return []

        pairs = self._build_through_pairs()
        if not pairs:
            logger.warning(
                "No cordon pair crosses the region interior; skipping E->E "
                "trips. Check that cordons were found on more than one side."
            )
            return []

        weights = np.array([w for _, _, w in pairs], dtype=float)
        weights = weights / weights.sum()
        chosen = self.rng.choice(len(pairs), size=n_trips, p=weights)
        departures = self.departures.sample(CLASS_THROUGH, self.rng, n_trips)

        trips = []
        for index, departure in zip(chosen, departures):
            entry, exit_, _ = pairs[int(index)]
            trips.append(FreightTrip(
                trip_class=CLASS_THROUGH,
                origin_lat=self._cordon_lat(entry),
                origin_lon=self._cordon_lon(entry),
                dest_lat=self._cordon_lat(exit_),
                dest_lon=self._cordon_lon(exit_),
                departure_seconds=float(departure),
                origin_cordon=entry.cordon_id,
                dest_cordon=exit_.cordon_id,
            ))
        return trips

    # -- sampling internals -------------------------------------------------

    def _sample_cordons(self, cordons: Sequence[Cordon],
                        n_trips: int) -> List[Cordon]:
        """Draw cordons in proportion to their weights."""
        weights = np.array([max(0.0, c.weight) for c in cordons], dtype=float)
        if weights.sum() <= 0:
            # An unweighted set is still usable — uniform is a worse spatial
            # model, not a broken one — but it means the caller skipped
            # assign_cordon_weights, which is worth saying out loud.
            logger.warning(
                "Cordon weights are all zero; sampling uniformly. Call "
                "assign_cordon_weights first for a volume-weighted draw."
            )
            weights = np.ones(len(cordons))
        weights = weights / weights.sum()
        indices = self.rng.choice(len(cordons), size=n_trips, p=weights)
        return [cordons[int(i)] for i in indices]

    def _sample_zone_near(self, cordon: Cordon) -> Zone:
        """Pick an internal zone for a trip end, by attractor and distance.

        This is the singly-constrained step: the cordon's total is known and is
        distributed over zones by ``attractor x friction(distance)``.
        """
        weights = self._zone_weights(cordon)
        index = self.rng.choice(len(self.zones), p=weights)
        return self.zones[int(index)]

    def _zone_weights(self, cordon: Cordon) -> np.ndarray:
        """Cached per-cordon zone sampling weights."""
        if not hasattr(self, '_zone_weight_cache'):
            self._zone_weight_cache: Dict[str, np.ndarray] = {}
            self._zone_xy = np.array([[z.x, z.y] for z in self.zones], dtype=float)
            self._zone_attractors = np.array(
                [max(0.0, z.attractor) for z in self.zones], dtype=float)
            if self._zone_attractors.sum() <= 0:
                logger.warning(
                    "Every zone has a zero attractor; internal trip ends will be "
                    "distributed by distance alone."
                )
                self._zone_attractors = np.ones(len(self.zones))

        cached = self._zone_weight_cache.get(cordon.cordon_id)
        if cached is not None:
            return cached

        deltas = self._zone_xy - np.array([cordon.x, cordon.y])
        distances_km = np.sqrt((deltas ** 2).sum(axis=1)) / 1000.0

        # A bounded friction form matters here for the reason compute_friction's
        # docstring gives: a power law is unbounded as distance approaches zero,
        # so a cordon anchored close to its nearest zone would dominate every
        # draw. Guard the zero anyway, for the power-law ablation case.
        distances_km = np.maximum(distances_km, 1e-3)
        friction = compute_friction(distances_km, self.beta, form=self.friction_form)

        weights = self._zone_attractors * friction
        total = weights.sum()
        if total <= 0 or not np.isfinite(total):
            logger.warning(
                f"Zone weights for {cordon.cordon_id} degenerate; falling back "
                f"to attractor only"
            )
            weights = self._zone_attractors.copy()
            total = weights.sum()

        weights = weights / total
        self._zone_weight_cache[cordon.cordon_id] = weights
        return weights

    #: How far apart two cordons must be in bearing, as seen from the region's
    #: centre, for a trip between them to count as crossing rather than
    #: skirting. 90 degrees admits an interstate that enters north and leaves
    #: east — a real through movement — while still rejecting two neighbours on
    #: the same side, which is the failure the test exists for.
    MIN_THROUGH_SEPARATION_DEG = 90.0

    def _build_through_pairs(self) -> List[Tuple[Cordon, Cordon, float]]:
        """Cordon pairs that genuinely cross the region, with their weights.

        The test is **angular, not metric**. A distance-based version — "the
        chord must pass within the region's radius of its centre" — sounds
        right and is wrong: for cordons on the periphery, the chord between two
        perpendicular gateways passes at r/sqrt(2), which exceeds a typical
        zone-spread radius, so it rejects legitimate crossings and silently
        drops every through trip.

        What actually distinguishes a crossing from a skirt is whether the two
        cordons are on *different sides* of the region, which is a question
        about bearing from the centre.
        """
        centroid = self._region_centroid()

        pairs = []
        for entry in self._entry_cordons:
            for exit_ in self._exit_cordons:
                if entry.cordon_id == exit_.cordon_id:
                    continue
                separation = self._bearing_separation(entry, exit_, centroid)
                if separation < self.MIN_THROUGH_SEPARATION_DEG:
                    continue
                weight = max(entry.weight, 0.0) * max(exit_.weight, 0.0)
                if weight <= 0:
                    weight = 1.0
                pairs.append((entry, exit_, weight))

        if not pairs:
            # Rather than drop the class silently, fall back to the weakest
            # usable rule — any two distinct cordons — and say so. A region
            # whose gateways are genuinely all on one side is unusual but not
            # impossible, and losing 30% of the demand without a word is worse
            # than a geometrically imperfect through trip.
            logger.warning(
                f"No cordon pair is more than "
                f"{self.MIN_THROUGH_SEPARATION_DEG:.0f} degrees apart; falling "
                f"back to any distinct pair for E->E trips. Check the cordon "
                f"compass spread in the summary."
            )
            for entry in self._entry_cordons:
                for exit_ in self._exit_cordons:
                    if entry.cordon_id != exit_.cordon_id:
                        weight = max(entry.weight * exit_.weight, 1.0)
                        pairs.append((entry, exit_, weight))

        return pairs

    def _region_centroid(self) -> np.ndarray:
        if not hasattr(self, '_centroid_cache'):
            self._centroid_cache = np.array([
                float(np.mean([z.x for z in self.zones])),
                float(np.mean([z.y for z in self.zones])),
            ])
        return self._centroid_cache

    @staticmethod
    def _bearing_separation(a: Cordon, b: Cordon, centroid: np.ndarray) -> float:
        """Angle between two cordons as seen from the region centre, in degrees.

        Returns a value in [0, 180]: 0 means the same side, 180 means opposite
        sides of the region.
        """
        bearing_a = math.atan2(a.y - centroid[1], a.x - centroid[0])
        bearing_b = math.atan2(b.y - centroid[1], b.x - centroid[0])
        difference = abs(math.degrees(bearing_a - bearing_b)) % 360.0
        return 360.0 - difference if difference > 180.0 else difference

    # -- coordinates --------------------------------------------------------

    def _cordon_lat(self, cordon: Cordon) -> float:
        return self._cordon_latlon(cordon)[0]

    def _cordon_lon(self, cordon: Cordon) -> float:
        return self._cordon_latlon(cordon)[1]

    def _cordon_latlon(self, cordon: Cordon) -> Tuple[float, float]:
        """Cordon position as lat/lon.

        Plans carry lat/lon and the writer projects them, so a cordon's network
        coordinates have to come back out of the projection. Without a converter
        the anchor zone's coordinates stand in — the trip then starts at the
        zone rather than at the road, which is worse but not wrong enough to
        stop a run.
        """
        cache = getattr(self, '_latlon_cache', None)
        if cache is None:
            cache = self._latlon_cache = {}
        if cordon.cordon_id in cache:
            return cache[cordon.cordon_id]

        converter = getattr(self, 'converter', None)
        if converter is not None:
            latlon = converter.utm_to_latlon(cordon.x, cordon.y)
        else:
            zone = self._zone_by_id(cordon.zone_id)
            latlon = (zone.lat, zone.lon) if zone else (0.0, 0.0)
        cache[cordon.cordon_id] = latlon
        return latlon

    def _zone_by_id(self, zone_id: Optional[str]) -> Optional[Zone]:
        if zone_id is None:
            return None
        if not hasattr(self, '_zone_index'):
            self._zone_index = {z.zone_id: z for z in self.zones}
        return self._zone_index.get(zone_id)

    def set_converter(self, converter) -> None:
        """Supply the coordinate converter so cordons keep their real position."""
        self.converter = converter
        self._latlon_cache = {}
