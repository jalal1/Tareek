"""Freight validation — the three tiers.

**Tier 1** — internal consistency. Pure Python, seconds to run, no MATSim and no
external data. Checks that what was generated is what was asked for: trip
counts, class split, departure profile, valid endpoints, reproducibility. Every
region gets this.

**Tier 2** — HPMS corridor check. Compares simulated truck volumes against
observed truck AADT per segment. Works for any region with Federal-Aid highways,
which is why it is the primary success metric rather than anything
region-specific. Needs the events file (see ``events.py``).

**Tier 3** — classification counts, where sensors report FHWA classes. Validates
the time-of-day model, which tier 2 cannot. Coverage is genuinely partial —
nationally about 49% of station-directions report all 13 classes, but it runs
from 100% in some states to zero in others — so it is checked per region rather
than assumed.

**What success means.** Freight is a single-digit to low-double-digit percentage
of traffic. It will not repair a large aggregate count error, and the module
must not be judged on whether total GEH improves. If the aggregate ratio was
already near 1.0, adding freight pushes it past — that is information about car
demand being too high, not a freight failure, which is why freight's
contribution is always reported separately.

See docs/freight/design.md §5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from models.freight.cordons import Cordon
from models.freight.demand import FreightDemand, TRIP_CLASSES
from models.freight.departure import DepartureSampler, HOURS_PER_DAY
from models.freight.generator import FreightTrip
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class CheckResult:
    """One validation check.

    ``passed`` is coerced to a built-in bool on construction. Most checks are
    the result of a numpy comparison, which yields ``numpy.bool_`` — truthy in
    Python but **not JSON-serialisable**, so it would sink the whole summary
    file at write time while every assertion still appeared to pass. Caught on
    the first real run; the writer's try/except meant the run survived but lost
    all its diagnostics.
    """
    name: str
    passed: bool
    detail: str
    value: Optional[float] = None
    tolerance: Optional[float] = None

    def __post_init__(self):
        self.passed = bool(self.passed)
        if self.value is not None:
            self.value = float(self.value)
        if self.tolerance is not None:
            self.tolerance = float(self.tolerance)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'passed': self.passed,
            'detail': self.detail,
            'value': (round(self.value, 5) if self.value is not None else None),
            'tolerance': self.tolerance,
        }


@dataclass
class ValidationReport:
    """The outcome of a validation tier."""
    tier: int
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)

    def to_dict(self) -> Dict:
        return {
            'tier': self.tier,
            'passed': self.passed,
            'n_checks': len(self.checks),
            'n_failed': sum(1 for c in self.checks if not c.passed),
            'checks': [c.to_dict() for c in self.checks],
        }

    def log(self) -> None:
        for check in self.checks:
            level = logger.info if check.passed else logger.warning
            level(f"  [{'PASS' if check.passed else 'FAIL'}] "
                  f"{check.name}: {check.detail}")


def validate_tier1(
    trips: Sequence[FreightTrip],
    demand: FreightDemand,
    cordons: Sequence[Cordon],
    zone_ids: Optional[set] = None,
    config: Optional[Dict] = None,
    profile_tolerance: float = 0.03,
    share_tolerance: float = 0.02,
) -> ValidationReport:
    """Internal consistency: is what was generated what was asked for?

    Args:
        profile_tolerance: allowed max absolute deviation per hour between the
            realised and configured departure distributions.
        share_tolerance: allowed absolute deviation on each class share.
    """
    report = ValidationReport(tier=1)
    config = config or {}
    n_trips = len(trips)

    # -- trip count matches the resolved demand ----------------------------
    report.add(CheckResult(
        name='trip_count_matches_demand',
        passed=n_trips == demand.total_trips,
        detail=f"generated {n_trips:,}, demand resolved {demand.total_trips:,}",
        value=float(n_trips - demand.total_trips),
    ))

    if n_trips == 0:
        report.add(CheckResult(
            name='non_empty',
            passed=False,
            detail="no freight trips were generated at all",
        ))
        return report

    # -- class split matches class_shares ----------------------------------
    counts = {name: 0 for name in TRIP_CLASSES}
    for trip in trips:
        counts[trip.trip_class] = counts.get(trip.trip_class, 0) + 1

    for name in TRIP_CLASSES:
        expected = demand.trips_by_class.get(name, 0)
        report.add(CheckResult(
            name=f'class_count_{name}',
            passed=counts[name] == expected,
            detail=f"{counts[name]:,} generated vs {expected:,} demanded",
            value=float(counts[name] - expected),
        ))

    configured_shares = demand.detail.get('class_shares', {})
    for name, configured in configured_shares.items():
        realised = counts.get(name, 0) / n_trips
        deviation = abs(realised - configured)
        report.add(CheckResult(
            name=f'class_share_{name}',
            passed=deviation <= share_tolerance,
            detail=f"realised {realised:.3f} vs configured {configured:.3f}",
            value=deviation,
            tolerance=share_tolerance,
        ))

    # -- departure profile matches the configuration -----------------------
    sampler = DepartureSampler.from_config(config)
    for trip_class in TRIP_CLASSES:
        class_trips = [t.departure_seconds for t in trips
                       if t.trip_class == trip_class]
        if not class_trips:
            continue
        realised = np.array(sampler.realised_distribution(class_trips))
        configured = np.array(sampler.profiles[sampler.profile_for_class(trip_class)])
        deviation = float(np.abs(realised - configured).max())
        # Small samples are noisy; scale the tolerance rather than failing a
        # correct model because only 40 trucks were drawn.
        tolerance = profile_tolerance + 1.0 / np.sqrt(len(class_trips))
        report.add(CheckResult(
            name=f'departure_profile_{trip_class}',
            passed=deviation <= tolerance,
            detail=(f"max hourly deviation {deviation:.4f} over "
                    f"{len(class_trips):,} trips"),
            value=deviation,
            tolerance=round(float(tolerance), 4),
        ))

    # -- endpoints are valid ------------------------------------------------
    cordon_ids = {c.cordon_id for c in cordons}
    entry_ids = {c.cordon_id for c in cordons if c.accepts_entry}
    exit_ids = {c.cordon_id for c in cordons if c.accepts_exit}

    bad_cordon = sum(
        1 for t in trips
        for cid in (t.origin_cordon, t.dest_cordon)
        if cid is not None and cid not in cordon_ids
    )
    report.add(CheckResult(
        name='cordon_ids_valid',
        passed=bad_cordon == 0,
        detail=f"{bad_cordon} trip ends reference an unknown cordon",
        value=float(bad_cordon),
    ))

    # Direction is the check that matters most here: a truck injected onto the
    # wrong carriageway either fails to route or loads the wrong side.
    wrong_direction = sum(
        1 for t in trips
        if (t.origin_cordon is not None and t.origin_cordon not in entry_ids)
        or (t.dest_cordon is not None and t.dest_cordon not in exit_ids)
    )
    report.add(CheckResult(
        name='cordon_direction_respected',
        passed=wrong_direction == 0,
        detail=(f"{wrong_direction} trips enter on an outbound cordon or "
                f"leave on an inbound one"),
        value=float(wrong_direction),
    ))

    if zone_ids is not None:
        bad_zone = sum(
            1 for t in trips
            for zid in (t.origin_zone, t.dest_zone)
            if zid is not None and zid not in zone_ids
        )
        report.add(CheckResult(
            name='zone_ids_valid',
            passed=bad_zone == 0,
            detail=f"{bad_zone} trip ends reference an unknown zone",
            value=float(bad_zone),
        ))

    # -- through trips genuinely go somewhere ------------------------------
    from models.freight.demand import CLASS_THROUGH
    degenerate = sum(1 for t in trips
                     if t.trip_class == CLASS_THROUGH
                     and t.origin_cordon == t.dest_cordon)
    report.add(CheckResult(
        name='through_trips_use_distinct_cordons',
        passed=degenerate == 0,
        detail=f"{degenerate} through trips start and end at the same cordon",
        value=float(degenerate),
    ))

    # -- departure times are inside a plausible day ------------------------
    departures = np.array([t.departure_seconds for t in trips])
    out_of_range = int(((departures < 0) | (departures >= 30 * 3600)).sum())
    report.add(CheckResult(
        name='departure_times_in_range',
        passed=out_of_range == 0,
        detail=f"{out_of_range} departures outside 00:00-30:00",
        value=float(out_of_range),
    ))

    return report


def validate_tier2(
    comparison: Dict,
    geh_target: float = 5.0,
    min_pct_under_target: float = 50.0,
) -> ValidationReport:
    """HPMS corridor check: do truck volumes match truck counts?

    This is the real test of the module. Note what it does *not* test: whether
    total volume improved. Freight is a small share of traffic and adding it to
    a model whose car demand is already too high will make the aggregate worse,
    which is a fact about car demand, not about freight.
    """
    report = ValidationReport(tier=2)

    n_links = comparison.get('n_links', 0)
    if n_links == 0:
        report.add(CheckResult(
            name='observed_volumes_available',
            passed=False,
            detail=('no link matched an observed truck volume; tier 2 cannot '
                    'run for this region'),
        ))
        return report

    pct_under = comparison.get('pct_geh_under_5', 0.0)
    report.add(CheckResult(
        name=f'geh_under_{geh_target:.0f}',
        passed=pct_under >= min_pct_under_target,
        detail=(f"{pct_under:.1f}% of {n_links} links have GEH < {geh_target:.0f} "
                f"(target {min_pct_under_target:.0f}%)"),
        value=pct_under,
        tolerance=min_pct_under_target,
    ))

    ratio = comparison.get('ratio')
    if ratio is not None:
        report.add(CheckResult(
            name='aggregate_truck_ratio',
            passed=0.7 <= ratio <= 1.3,
            detail=f"simulated/observed truck volume = {ratio:.3f}",
            value=ratio,
        ))

    return report


def summarise(reports: Sequence[ValidationReport]) -> Dict:
    """Fold several tier reports into the shape freight_summary.json wants."""
    return {
        'passed': all(report.passed for report in reports),
        'tiers': {f'tier{report.tier}': report.to_dict() for report in reports},
    }
