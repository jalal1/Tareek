"""How many boundary freight trips, and of which class.

Turning an observed truck *share* or a set of cordon *volumes* into a number of
simulated trips has three traps, each of which silently changes the answer by a
factor:

1. **Crossings are not trips.** AADT counts vehicles passing a cross-section. An
   E->E truck crosses the cordon line twice — once entering, once leaving — so
   summing AADT over all cordons counts every through trip at both of its ends.
   The correction is ``1 / (1 + through_share)``, not a flat 0.5: E->I and I->E
   cross exactly once, only E->E is double. At the default 30% through share
   that is **0.77**, not 0.5.

2. **Freight must be scaled by ``scaling_factor``** like everything else.
   Birmingham runs 0.105 with ``countsScaleFactor = 1/flowCapacityFactor =
   9.52``, which MATSim applies to *every* simulated vehicle when comparing
   against counts. Freight generated at full volume would be multiplied by 9.52
   along with the rest and overstate trucks roughly tenfold.

3. **Flooring biases the total down.** At Birmingham's sample rate a cordon
   carrying 2,000 trucks/day yields ~210 agents; rounding each cordon's share
   down loses a meaningful fraction. Probabilistic rounding preserves the
   expected total, matching ``_apply_scaling_to_od_matrix``.

See docs/freight/design.md §3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from models.freight.cordons import (
    Cordon,
    VALID_WEIGHT_BY,
    WEIGHT_BY_CAPACITY,
    WEIGHT_BY_HPMS_TRUCK_AADT,
)
from models.freight.truck_share import TruckShare
from utils.logger import setup_logger

logger = setup_logger(__name__)


# How the daily freight total is derived.
#   hpms_cordon — sum observed truck AADT over the cordons themselves. Preferred,
#                 and self-calibrating: a region with two interstates through it
#                 gets more freight than one with none, from data, with no
#                 parameter to set. This is what the Memphis model does.
#   car_share   — a fraction of the car demand. The fallback for when no cordon
#                 can be matched to an observed volume.
#   absolute    — a number given directly in the config. For experiments.
DEMAND_HPMS_CORDON = 'hpms_cordon'
DEMAND_CAR_SHARE = 'car_share'
DEMAND_ABSOLUTE = 'absolute'
VALID_DEMAND_SOURCES = (DEMAND_HPMS_CORDON, DEMAND_CAR_SHARE, DEMAND_ABSOLUTE)

# The three trip classes. Internal (I->I) freight is out of scope.
CLASS_EXTERNAL_TO_INTERNAL = 'external_to_internal'
CLASS_INTERNAL_TO_EXTERNAL = 'internal_to_external'
CLASS_THROUGH = 'through'
TRIP_CLASSES = (CLASS_EXTERNAL_TO_INTERNAL, CLASS_INTERNAL_TO_EXTERNAL,
                CLASS_THROUGH)


@dataclass
class ClassShares:
    """How the freight total splits across the three classes.

    E->I and I->E are symmetric by construction — a truck that comes in
    generally goes out again — and Memphis's own external truck model supports
    it: 3,850 daily trucks produced internally against 4,068 attracted. The
    through share is the least certain number and is the second thing to
    calibrate after ``demand_scale``.
    """
    external_to_internal: float = 0.35
    internal_to_external: float = 0.35
    through: float = 0.30

    def __post_init__(self):
        for name in TRIP_CLASSES:
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"class_shares.{name} must be >= 0, got {value}")
        if self.total <= 0:
            raise ValueError("class_shares must not all be zero")

    @property
    def total(self) -> float:
        return self.external_to_internal + self.internal_to_external + self.through

    def normalised(self) -> Dict[str, float]:
        """The shares as fractions summing to 1.

        Hand-edited configs rarely sum to exactly 1, and treating a config that
        sums to 0.99 as if it summed to 1 would silently lose 1% of the demand.
        """
        total = self.total
        return {name: getattr(self, name) / total for name in TRIP_CLASSES}

    @classmethod
    def from_config(cls, config: Optional[Dict]) -> 'ClassShares':
        config = config or {}
        return cls(
            external_to_internal=float(config.get('external_to_internal', 0.35)),
            internal_to_external=float(config.get('internal_to_external', 0.35)),
            through=float(config.get('through', 0.30)),
        )


@dataclass
class FreightDemand:
    """The resolved trip totals, with every intermediate kept for audit.

    The arithmetic between an observed AADT and a simulated agent count passes
    through four multiplications, any of which is easy to get wrong by a factor.
    Recording each one is what makes a wrong total diagnosable from the summary
    rather than by re-deriving it.
    """
    total_trips: int
    trips_by_class: Dict[str, int]
    source: str
    gross_crossings: float = 0.0
    crossing_factor: float = 1.0
    demand_scale: float = 1.0
    scaling_factor: float = 1.0
    unscaled_trips: float = 0.0
    detail: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'total_trips': self.total_trips,
            'trips_by_class': dict(self.trips_by_class),
            'source': self.source,
            'gross_crossings': round(self.gross_crossings, 1),
            'crossing_factor': round(self.crossing_factor, 4),
            'demand_scale': self.demand_scale,
            'scaling_factor': self.scaling_factor,
            'unscaled_trips': round(self.unscaled_trips, 1),
            'detail': self.detail,
        }


def crossing_factor(through_share: float) -> float:
    """Convert cordon *crossings* into *trips*.

    Every E->E truck is counted at two cordons — the one it enters by and the one
    it leaves by — while E->I and I->E are counted once each. So the gross sum of
    cordon volumes over-counts by exactly the through share:

        crossings = trips * (1 - t) * 1 + trips * t * 2 = trips * (1 + t)

    hence ``trips = crossings / (1 + t)``.

    The flat 0.5 that first suggests itself is only correct when *every* trip is
    a through trip. At the default 30% through share the right factor is 0.77 —
    using 0.5 would throw away a third of the demand.
    """
    if not 0.0 <= through_share <= 1.0:
        raise ValueError(f"through_share must be in [0, 1], got {through_share}")
    return 1.0 / (1.0 + through_share)


def probabilistic_round(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Round preserving the expected total, rather than biasing it down.

    Matches ``_apply_scaling_to_od_matrix`` in the non-work generator: floor,
    then round up with probability equal to the fractional part.
    """
    floors = np.floor(values)
    return (floors + (rng.random(values.shape) < (values - floors))).astype(int)


def resolve_demand(
    config: Dict,
    cordons: Sequence[Cordon],
    truck_share: Optional[TruckShare] = None,
    car_trips: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> FreightDemand:
    """Work out how many freight trips to generate, and of which class.

    Args:
        config: the whole config dict; reads ``freight`` and ``plan_generation``.
        cordons: the detected gateways, carrying observed or fallback weights.
        truck_share: the resolved regional truck share. Needed by the
            ``car_share`` route, and used to report the vehicle mix.
        car_trips: how many car trips the rest of the pipeline generated. Only
            the ``car_share`` route needs it.
        rng: seeded generator, so a run is reproducible.

    Returns:
        A FreightDemand with the totals and the full audit trail.
    """
    freight_config = config.get('freight', {})
    plan_config = config.get('plan_generation', {})

    source = freight_config.get('demand_source', DEMAND_HPMS_CORDON)
    if source not in VALID_DEMAND_SOURCES:
        raise ValueError(
            f"freight.demand_source must be one of {VALID_DEMAND_SOURCES}, "
            f"got {source!r}"
        )

    shares = ClassShares.from_config(freight_config.get('class_shares'))
    demand_scale = float(freight_config.get('demand_scale', 1.0))
    scaling_factor = float(plan_config.get('scaling_factor', 1.0))
    if rng is None:
        rng = np.random.default_rng(plan_config.get('random_seed', 42))

    normalised = shares.normalised()
    factor = crossing_factor(normalised[CLASS_THROUGH])

    # -- the daily trip total, before sampling ------------------------------
    gross_crossings = 0.0
    detail: Dict = {'demand_source': source}

    if source == DEMAND_ABSOLUTE:
        trips_per_day = freight_config.get('trips_per_day')
        if trips_per_day is None:
            raise ValueError(
                "freight.demand_source='absolute' requires freight.trips_per_day"
            )
        unscaled = float(trips_per_day)
        detail['note'] = 'trips_per_day given directly in config'

    elif source == DEMAND_CAR_SHARE:
        if car_trips is None or truck_share is None:
            raise ValueError(
                "freight.demand_source='car_share' needs both car_trips and a "
                "resolved truck_share"
            )
        unscaled = float(car_trips) * truck_share.total
        detail.update({
            'car_trips': int(car_trips),
            'truck_share': round(truck_share.total, 5),
            'truck_share_source': truck_share.source,
        })

    else:  # DEMAND_HPMS_CORDON
        gross_crossings = float(sum(c.weight for c in cordons))
        if gross_crossings <= 0:
            # Falling back silently here would produce zero freight and report
            # success — the exact failure the cordon gate exists to prevent.
            raise ValueError(
                f"Cordon weights sum to {gross_crossings}; no freight could be "
                f"generated. Weights must be assigned (from observed truck AADT "
                f"or from link capacity) before demand is resolved."
            )
        unscaled = gross_crossings * factor
        detail.update({
            'n_cordons': len(cordons),
            'gross_crossings': round(gross_crossings, 1),
            'crossing_factor': round(factor, 4),
            'crossing_factor_note': (
                'trips = crossings / (1 + through_share); a flat 0.5 would be '
                'correct only if every trip were a through trip'
            ),
        })

    # -- scale into the simulated sample ------------------------------------
    scaled = unscaled * demand_scale * scaling_factor
    if scaled < 0:
        raise ValueError(f"Freight demand resolved to a negative total: {scaled}")

    per_class = np.array([scaled * normalised[name] for name in TRIP_CLASSES])
    counts = probabilistic_round(per_class, rng)
    trips_by_class = {name: int(count) for name, count in zip(TRIP_CLASSES, counts)}
    total = int(counts.sum())

    if truck_share is not None:
        detail['vehicle_mix'] = truck_share.mix
        detail.setdefault('truck_share_source', truck_share.source)
    detail['class_shares'] = {k: round(v, 4) for k, v in normalised.items()}

    demand = FreightDemand(
        total_trips=total,
        trips_by_class=trips_by_class,
        source=source,
        gross_crossings=gross_crossings,
        crossing_factor=factor if source == DEMAND_HPMS_CORDON else 1.0,
        demand_scale=demand_scale,
        scaling_factor=scaling_factor,
        unscaled_trips=unscaled,
        detail=detail,
    )

    logger.info(
        f"Freight demand: {total:,} trips "
        f"(E->I {trips_by_class[CLASS_EXTERNAL_TO_INTERNAL]:,}, "
        f"I->E {trips_by_class[CLASS_INTERNAL_TO_EXTERNAL]:,}, "
        f"E->E {trips_by_class[CLASS_THROUGH]:,})"
    )
    logger.info(
        f"  {unscaled:,.0f} unscaled x demand_scale {demand_scale} "
        f"x scaling_factor {scaling_factor} = {scaled:,.1f}"
    )
    if total == 0 and scaled > 0:
        logger.warning(
            f"Freight demand rounded to zero trips from {scaled:.2f} expected. "
            f"The sample rate may be too low for this region's freight volume."
        )

    return demand


# Daily volume as a multiple of hourly capacity, for the fallback only.
#
# Capacity is a road's *ceiling* in vehicles per hour; AADT is a *daily count*.
# They are different units and cannot be compared directly. The bridge is the
# design-hour factor K: peak-hour volume is conventionally ~9-10% of AADT, so
# AADT ~ peak_hour / 0.10. Taking the peak hour as roughly 80% of capacity on a
# corridor that is not permanently congested gives AADT ~ capacity * 8.
#
# This is a rough conversion and is only ever used when no observed volume
# exists for a cordon. Its purpose is to put the fallback on the *same scale*
# as an observed AADT so the two kinds of cordon can share one weighted draw —
# not to predict any particular road's traffic.
CAPACITY_TO_DAILY_VOLUME = 8.0


def assign_cordon_weights(
    cordons: Sequence[Cordon],
    truck_aadt_by_link: Optional[Dict[str, float]] = None,
    truck_share: Optional[TruckShare] = None,
    capacity_to_daily: float = CAPACITY_TO_DAILY_VOLUME,
    weight_by: str = WEIGHT_BY_HPMS_TRUCK_AADT,
) -> Dict:
    """Give every cordon the sampling weight that decides its share of trips.

    Weighting by **observed truck volume** rather than by capacity is the
    important choice: it gives an interstate cordon and a rural highway cordon
    the proportions the real corridors carry, in that specific region, from
    data. Capacity is only the fallback for a cordon no observation matches.

    **Units matter here and are easy to get wrong.** Observed weights are truck
    AADT — a daily count. Capacity is vehicles per hour. Mixing the two in one
    weighted draw without converting would make capacity-weighted cordons carry
    roughly an eighth of their proper share relative to observed ones, and the
    error would be invisible because both are "big numbers".

    Args:
        cordons: modified in place — each gets ``weight`` set.
        truck_aadt_by_link: observed truck AADT per network link, where known.
        truck_share: converts a capacity-derived *total* volume into a *truck*
            volume, so the fallback is comparable with an observed truck AADT.
        capacity_to_daily: hourly capacity -> daily volume factor. See
            CAPACITY_TO_DAILY_VOLUME.
        weight_by: ``hpms_truck_aadt`` prefers observed volume per cordon and
            falls back to capacity; ``capacity`` ignores observations entirely,
            which is the ablation control for "how much is the data doing?".

    Returns:
        A dict describing how many cordons got which kind of weight.
    """
    if weight_by not in VALID_WEIGHT_BY:
        raise ValueError(
            f"weight_by must be one of {VALID_WEIGHT_BY}, got {weight_by!r}")

    truck_aadt_by_link = truck_aadt_by_link or {}
    if weight_by == WEIGHT_BY_CAPACITY and truck_aadt_by_link:
        logger.info(
            "weight_by='capacity': ignoring the observed truck volumes supplied "
            "and weighting every cordon by capacity instead."
        )
        truck_aadt_by_link = {}

    observed = 0
    fallback = 0

    for cordon in cordons:
        matched = [truck_aadt_by_link[link_id] for link_id in cordon.link_ids
                   if link_id in truck_aadt_by_link]
        if matched:
            # The merged links are one carriageway of one road, so its volume is
            # that road's, not the sum over collinear segments of it.
            cordon.weight = float(max(matched))
            observed += 1
        else:
            share = truck_share.total if truck_share is not None else 1.0
            cordon.weight = float(cordon.capacity) * capacity_to_daily * share
            fallback += 1

    stats = {
        'n_observed': observed,
        'n_fallback': fallback,
        'total_weight': round(sum(c.weight for c in cordons), 1),
        'capacity_to_daily': capacity_to_daily,
        'weight_by': weight_by,
    }
    logger.info(
        f"Cordon weights ({weight_by}): {observed} from observed truck AADT, "
        f"{fallback} from capacity x {capacity_to_daily} x truck share"
    )
    if observed == 0 and cordons and weight_by != WEIGHT_BY_CAPACITY:
        logger.warning(
            "No cordon matched an observed truck volume; every weight is a "
            "capacity fallback. Cordon shares reflect road size, not measured "
            "freight, and the regional total rests on the capacity-to-daily "
            "conversion rather than on data. demand_scale is the knob to "
            "calibrate against observed counts."
        )
    return stats
