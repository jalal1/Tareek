"""When trucks depart — two profiles, chosen by trip class.

The FHWA/LTPP study identifies four national time-of-day patterns. Two are
trucks, and the distinction is exactly the one this module cares about:

**Business day** (local and regional trucks, classes 5-8) — "The vast majority
of trucking movements in the nation start and end during the normal, extended
business day (that is, between 6:00 a.m. and 6:00 p.m.)... The truck pattern
usually peaks earlier in the day [than cars] and drops earlier in the day." A
broad plateau 09:00-15:00, rising steeply from 06:00, falling after 16:00.

**Through** (long-haul, classes 11-12 and often 9) — "Long distance trucks are
not constrained to the normal business day. Many travel at night to avoid
traffic congestion... because many long distance drivers are paid by the mile
rather than by the hour, there is incentive to drive whenever possible... This
pattern has a fairly flat volume pattern." Nearly flat, ~4-4.5% per hour.

The assignment falls out with no extra parameters: E->I and I->E have a real
stop in the region, so they are business-day; E->E is long-haul by definition.

One caution worth recording so we do not misread our own validation: truck
*percentage* peaks at night — 30-50% of rural traffic in the small hours — but
that is because car volume collapses, not because truck volume rises. Validate
against truck *counts* per hour, never truck share per hour.

See docs/freight/design.md §4.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from utils.logger import setup_logger

logger = setup_logger(__name__)

HOURS_PER_DAY = 24
SECONDS_PER_HOUR = 3600

# Digitised by eye from LTPP Figures 6 and 7. They do **not** sum to 1 — they
# are read off a chart, not computed — which is precisely why the loader
# normalises rather than trusting the numbers. A hand-edited config will not sum
# to 1 either.
BUSINESS_DAY_PROFILE = [
    .008, .006, .005, .005, .008, .020,   # 00-05
    .038, .058, .070, .078, .080, .078,   # 06-11
    .075, .078, .076, .068, .052, .038,   # 12-17
    .028, .020, .015, .012, .010, .009,   # 18-23
]

THROUGH_PROFILE = [
    .038, .036, .035, .035, .036, .038,   # 00-05
    .042, .044, .045, .045, .045, .045,   # 06-11
    .045, .045, .045, .044, .043, .042,   # 12-17
    .041, .040, .040, .039, .039, .038,   # 18-23
]

PROFILE_BUSINESS_DAY = 'business_day'
PROFILE_THROUGH = 'through'


class DepartureProfileError(ValueError):
    """Raised when a configured profile cannot be used as a distribution."""


def normalise_profile(profile: Sequence[float], name: str = 'profile') -> np.ndarray:
    """Validate a 24-hour weight vector and normalise it to sum to 1.

    Rejects rather than repairs anything that is not a distribution: a negative
    weight or a wrong length is a config error, and silently coercing it would
    produce a plausible-looking but wrong time-of-day model.
    """
    values = np.asarray(list(profile), dtype=float)

    if values.shape != (HOURS_PER_DAY,):
        raise DepartureProfileError(
            f"{name} must have exactly {HOURS_PER_DAY} hourly weights, got "
            f"{values.size}"
        )
    if not np.isfinite(values).all():
        raise DepartureProfileError(f"{name} contains a non-finite value")
    if (values < 0).any():
        raise DepartureProfileError(
            f"{name} contains a negative weight at hour "
            f"{int(np.argmin(values))}"
        )

    total = values.sum()
    if total <= 0:
        raise DepartureProfileError(f"{name} sums to {total}; nothing to sample")

    return values / total


class DepartureSampler:
    """Samples a departure second-of-day from an hourly profile.

    Holds both profiles and picks between them by trip class, so callers never
    have to know which curve a class uses.
    """

    def __init__(
        self,
        business_day: Optional[Sequence[float]] = None,
        through: Optional[Sequence[float]] = None,
    ):
        self.profiles = {
            PROFILE_BUSINESS_DAY: normalise_profile(
                business_day if business_day else BUSINESS_DAY_PROFILE,
                PROFILE_BUSINESS_DAY),
            PROFILE_THROUGH: normalise_profile(
                through if through else THROUGH_PROFILE,
                PROFILE_THROUGH),
        }

    @classmethod
    def from_config(cls, config: Dict) -> 'DepartureSampler':
        distribution = (config.get('freight', {})
                        .get('departure_distribution', {}) or {})
        return cls(
            business_day=distribution.get(PROFILE_BUSINESS_DAY) or None,
            through=distribution.get(PROFILE_THROUGH) or None,
        )

    def profile_for_class(self, trip_class: str) -> str:
        """Which curve a trip class departs on.

        E->E is passing through and long-haul by definition; the other two have
        a real stop in the region and behave like business-day trucking.
        """
        from models.freight.demand import CLASS_THROUGH
        return (PROFILE_THROUGH if trip_class == CLASS_THROUGH
                else PROFILE_BUSINESS_DAY)

    def sample(self, trip_class: str, rng: np.random.Generator,
               size: int = 1) -> np.ndarray:
        """Draw departure times, in seconds from midnight.

        The hour comes from the profile; the position within the hour is uniform
        so departures do not stack on hour boundaries — MATSim would otherwise
        release a whole hour's trucks in the same second.
        """
        weights = self.profiles[self.profile_for_class(trip_class)]
        hours = rng.choice(HOURS_PER_DAY, size=size, p=weights)
        offsets = rng.random(size) * SECONDS_PER_HOUR
        return hours * SECONDS_PER_HOUR + offsets

    def realised_distribution(self, seconds: Sequence[float]) -> List[float]:
        """Hourly shares actually generated, for the tier-1 consistency check."""
        if len(seconds) == 0:
            return [0.0] * HOURS_PER_DAY
        hours = (np.asarray(seconds, dtype=float) // SECONDS_PER_HOUR).astype(int)
        hours = np.clip(hours, 0, HOURS_PER_DAY - 1)
        counts = np.bincount(hours, minlength=HOURS_PER_DAY)
        return (counts / counts.sum()).tolist()


def seconds_to_hms(seconds: float) -> str:
    """Format a second-of-day as MATSim's HH:MM:SS.

    Hours are not wrapped at 24: MATSim accepts times past midnight, and
    clamping them would silently move a late departure to the start of the day.
    """
    total = int(round(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"
