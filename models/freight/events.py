"""Freight link volumes from the MATSim events file.

**Why this exists.** No MATSim output distinguishes one vehicle from another at
link level. Verified against the code: ``evaluator.py`` reads `linkstats.txt.gz`
into `link_id, HRS0-1avg ... HRS23-24avg`, and MATSim's `countscompare.txt` is
one row per count station — **both report total link volume only**. There is no
breakdown by mode, vehicle type or subpopulation anywhere in the pipeline. So
"how many trucks used this link" cannot be answered from the normal outputs, and
tier-2 validation cannot run without this module.

The events file is the one output that carries vehicle identity. Filtering
``entered link`` events by the vehicle IDs belonging to freight persons gives a
freight-only link-volume table in the same shape as linkstats; differencing it
against the total gives the car stream.

**Event names verified against a real events file**, not assumed: the type
attribute is ``"entered link"`` — lowercase, with a space — not the Java class
name ``LinkEnterEvent``. For car agents MATSim names the auto-created vehicle
after the person, so a freight person's vehicle ID is its person ID; the
extractor accepts either, because that identity is a MATSim convention rather
than a guarantee.

Cost to budget for: events files are large and are only written when
``controller.writeEventsInterval`` emits them at the final iteration.

See docs/freight/design.md §5.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from utils.logger import setup_logger

logger = setup_logger(__name__)

HOURS_PER_DAY = 24

#: The event MATSim writes when a vehicle enters a link. Verified against a real
#: output_events.xml.gz; the Java class is LinkEnterEvent but the serialised
#: ``type`` attribute is this string.
EVENT_LINK_ENTER = 'entered link'
EVENT_VEHICLE_ENTERS_TRAFFIC = 'vehicle enters traffic'

#: Transit vehicles also traverse links and emit the same event. They are not
#: freight and must never be counted as such; this event announces them.
EVENT_TRANSIT_DRIVER_STARTS = 'TransitDriverStarts'


class EventsExtractionError(RuntimeError):
    """Raised when the events file needed for tier-2 validation is unusable."""


@dataclass
class LinkVolumes:
    """Hourly volumes per link for one vehicle stream.

    Attributes:
        volumes: ``{link_id: array of 24 hourly counts}``.
        n_vehicles: how many distinct vehicles contributed.
        n_events: how many link-enter events were counted.
        scale_factor: what the raw counts were multiplied by to reach
            real-world volumes. Simulated counts are a sample, so a comparison
            against observed AADT has to scale them the same way MATSim's
            ``countsScaleFactor`` does.
        overflow: entries at or after 24:00, kept **out** of the 24-hour
            profile. MATSim simulates past midnight, and folding those into
            hour 23 conserves the daily total but invents a late-night peak:
            measured on a real run, hour 23 read 5.7% against hour 22's 2.7%
            purely from the fold. Tier 3 compares the *shape* of the day, so a
            fabricated spike there would look like a timing error in the model.
            Reported separately instead, and added back for daily totals.
    """
    volumes: Dict[str, np.ndarray] = field(default_factory=dict)
    n_vehicles: int = 0
    n_events: int = 0
    scale_factor: float = 1.0
    overflow: Dict[str, float] = field(default_factory=dict)

    def daily(self, link_id: str) -> float:
        """Total volume on a link across the whole simulated day.

        Includes post-midnight overflow, because a comparison against observed
        AADT is a comparison of daily totals — dropping the overflow would
        understate the link.
        """
        series = self.volumes.get(link_id)
        base = float(series.sum()) if series is not None else 0.0
        return base + self.overflow.get(link_id, 0.0)

    def hourly(self, link_id: str) -> np.ndarray:
        """The 24-hour profile, **excluding** post-midnight overflow."""
        series = self.volumes.get(link_id)
        return series if series is not None else np.zeros(HOURS_PER_DAY)

    def total(self) -> float:
        """Every entry counted, overflow included."""
        return float(sum(series.sum() for series in self.volumes.values())
                     + sum(self.overflow.values()))

    def overflow_total(self) -> float:
        return float(sum(self.overflow.values()))

    def scaled(self, factor: float) -> 'LinkVolumes':
        """Return a copy with volumes multiplied up to real-world scale."""
        return LinkVolumes(
            volumes={k: v * factor for k, v in self.volumes.items()},
            n_vehicles=self.n_vehicles,
            n_events=self.n_events,
            scale_factor=self.scale_factor * factor,
            overflow={k: v * factor for k, v in self.overflow.items()},
        )

    def to_dataframe(self):
        """Same shape as ``evaluator.load_linkstats`` output, for reuse."""
        import pandas as pd
        if not self.volumes:
            return pd.DataFrame(
                columns=['link_id'] + [f'HRS{h}-{h+1}avg' for h in range(HOURS_PER_DAY)])
        rows = []
        for link_id, series in self.volumes.items():
            row = {'link_id': link_id}
            row.update({f'HRS{h}-{h+1}avg': float(series[h])
                        for h in range(HOURS_PER_DAY)})
            rows.append(row)
        return pd.DataFrame(rows)


def freight_vehicle_ids(plans_path: Path, subpopulation: str = 'freight') -> Set[str]:
    """Read the plans file for the person IDs in the freight subpopulation.

    Reads the file MATSim was *given*, so the IDs match whatever renumbering the
    pipeline applied before writing it. The alternative — assuming a naming
    convention — breaks the moment ``run_experiment`` renumbers persons, which
    it does.
    """
    plans_path = Path(plans_path)
    if not plans_path.exists():
        raise EventsExtractionError(f"Plans file not found: {plans_path}")

    ids: Set[str] = set()
    opener = gzip.open if plans_path.suffix == '.gz' else open

    with opener(str(plans_path), 'rt', encoding='utf-8') as handle:
        for _, elem in ET.iterparse(handle, events=('end',)):
            if elem.tag != 'person':
                continue
            attributes = elem.find('attributes')
            if attributes is not None:
                for attribute in attributes.findall('attribute'):
                    if (attribute.get('name') == 'subpopulation'
                            and (attribute.text or '').strip() == subpopulation):
                        ids.add(elem.get('id'))
                        break
            elem.clear()

    logger.info(
        f"Found {len(ids):,} persons in subpopulation '{subpopulation}' "
        f"in {plans_path.name}"
    )
    return ids


def extract_link_volumes(
    events_path: Path,
    vehicle_ids: Optional[Set[str]] = None,
    exclude_transit: bool = True,
) -> LinkVolumes:
    """Count link entries per hour, optionally for one set of vehicles.

    Args:
        events_path: the MATSim events file (.xml or .xml.gz).
        vehicle_ids: count only these vehicles. None counts every vehicle,
            which gives the total stream to difference against.
        exclude_transit: drop transit vehicles. They emit the same link-enter
            event as cars and trucks, so without this a bus route inflates the
            "car" stream that freight is differenced against.

    Returns:
        LinkVolumes with unscaled simulated counts.
    """
    events_path = Path(events_path)
    if not events_path.exists():
        raise EventsExtractionError(
            f"Events file not found: {events_path}. Tier-2 validation needs "
            f"events at the final iteration — check that "
            f"controller.writeEventsInterval is set (freight.require_events "
            f"does this) and that the run reached its last iteration."
        )

    volumes: Dict[str, np.ndarray] = {}
    overflow: Dict[str, float] = {}
    seen_vehicles: Set[str] = set()
    transit_vehicles: Set[str] = set()
    n_events = 0

    opener = gzip.open if events_path.suffix == '.gz' else open

    with opener(str(events_path), 'rt', encoding='utf-8') as handle:
        for _, elem in ET.iterparse(handle, events=('end',)):
            if elem.tag != 'event':
                continue

            event_type = elem.get('type')

            if event_type == EVENT_TRANSIT_DRIVER_STARTS:
                vehicle = elem.get('vehicleId') or elem.get('vehicle')
                if vehicle:
                    transit_vehicles.add(vehicle)
                elem.clear()
                continue

            if event_type != EVENT_LINK_ENTER:
                elem.clear()
                continue

            vehicle = elem.get('vehicle')
            if vehicle is None:
                elem.clear()
                continue
            if vehicle_ids is not None and vehicle not in vehicle_ids:
                elem.clear()
                continue
            if exclude_transit and vehicle in transit_vehicles:
                elem.clear()
                continue

            link_id = elem.get('link')
            try:
                hour = int(float(elem.get('time', 0.0)) // 3600)
            except (TypeError, ValueError):
                elem.clear()
                continue

            # Past-midnight traffic is real and must not be dropped, but it is
            # not hour-23 traffic either. Keeping it separate preserves the
            # daily total for tier 2 without fabricating a late-night peak in
            # the profile tier 3 compares.
            if hour >= HOURS_PER_DAY:
                overflow[link_id] = overflow.get(link_id, 0.0) + 1
            else:
                hour = max(hour, 0)
                if link_id not in volumes:
                    volumes[link_id] = np.zeros(HOURS_PER_DAY)
                volumes[link_id][hour] += 1

            seen_vehicles.add(vehicle)
            n_events += 1
            elem.clear()

    stream = 'freight' if vehicle_ids is not None else 'all'
    n_overflow = int(sum(overflow.values()))
    logger.info(
        f"Extracted {stream} link volumes: {n_events:,} link entries by "
        f"{len(seen_vehicles):,} vehicles over {len(volumes):,} links"
        + (f" ({n_overflow:,} after 24:00, held out of the hourly profile)"
           if n_overflow else "")
    )
    if vehicle_ids is not None and not seen_vehicles:
        logger.warning(
            "No freight vehicle appeared in the events file. Either no freight "
            "was generated, or the vehicle IDs do not match the person IDs in "
            "plans.xml."
        )

    return LinkVolumes(volumes=volumes, n_vehicles=len(seen_vehicles),
                       n_events=n_events, overflow=overflow)


def extract_freight_and_total(
    events_path: Path,
    plans_path: Path,
    subpopulation: str = 'freight',
    scale_factor: float = 1.0,
) -> Dict[str, LinkVolumes]:
    """Freight, total and derived car link volumes in one pass over the events.

    The events file is large, so this reads it once and splits, rather than
    calling ``extract_link_volumes`` twice.

    Args:
        scale_factor: multiply simulated counts up to real-world volumes,
            normally ``1 / flowCapacityFactor`` — the same figure MATSim applies
            as ``countsScaleFactor``. Comparing an unscaled sample against
            observed AADT would understate by that factor.
    """
    freight_ids = freight_vehicle_ids(plans_path, subpopulation)

    total = extract_link_volumes(events_path, vehicle_ids=None)
    freight = extract_link_volumes(events_path, vehicle_ids=freight_ids)

    car_volumes = {}
    for link_id, series in total.volumes.items():
        remainder = series - freight.hourly(link_id)
        car_volumes[link_id] = np.maximum(remainder, 0.0)

    car_overflow = {}
    for link_id, value in total.overflow.items():
        car_overflow[link_id] = max(0.0, value - freight.overflow.get(link_id, 0.0))

    car = LinkVolumes(
        volumes=car_volumes,
        n_vehicles=max(0, total.n_vehicles - freight.n_vehicles),
        n_events=max(0, total.n_events - freight.n_events),
        overflow=car_overflow,
    )

    result = {
        'freight': freight.scaled(scale_factor),
        'car': car.scaled(scale_factor),
        'total': total.scaled(scale_factor),
    }

    if total.total() > 0:
        share = freight.total() / total.total() * 100
        logger.info(f"Freight is {share:.2f}% of simulated link entries")

    return result


def geh(simulated: float, observed: float) -> float:
    """GEH statistic — the standard traffic-count goodness-of-fit measure.

    GEH < 5 is conventionally a good match, 5-10 marginal, >10 poor. It is used
    instead of a plain ratio because it tolerates larger relative error on small
    volumes, where a ratio is dominated by noise.
    """
    denominator = simulated + observed
    if denominator <= 0:
        return 0.0
    return float(np.sqrt(2.0 * (simulated - observed) ** 2 / denominator))


def compare_against_observed(
    freight_volumes: LinkVolumes,
    observed_truck_aadt: Dict[str, float],
    all_links: bool = False,
) -> Dict:
    """Tier-2: simulated truck volumes against observed truck AADT.

    Args:
        freight_volumes: already scaled to real-world volumes.
        observed_truck_aadt: ``{link_id: AADT_SINGLE_UNIT + AADT_COMBINATION}``.
        all_links: return every comparison under ``links`` rather than the 50
            busiest. Needed by anything computing a statistic from them —
            stratified %RMSE over the top 50 describes the busiest links, not
            the model.

    Returns:
        Per-link comparisons plus the aggregate GEH summary.
    """
    comparisons = []
    for link_id, observed in observed_truck_aadt.items():
        simulated = freight_volumes.daily(link_id)
        comparisons.append({
            'link_id': link_id,
            'simulated': round(simulated, 1),
            'observed': round(float(observed), 1),
            'ratio': round(simulated / observed, 3) if observed > 0 else None,
            'geh': round(geh(simulated, float(observed)), 2),
        })

    if not comparisons:
        return {'n_links': 0, 'note': 'no observed truck volumes to compare against'}

    geh_values = np.array([c['geh'] for c in comparisons])
    simulated_total = sum(c['simulated'] for c in comparisons)
    observed_total = sum(c['observed'] for c in comparisons)

    return {
        'n_links': len(comparisons),
        'geh_mean': round(float(geh_values.mean()), 2),
        'geh_median': round(float(np.median(geh_values)), 2),
        'pct_geh_under_5': round(float((geh_values < 5).mean() * 100), 1),
        'pct_geh_under_10': round(float((geh_values < 10).mean() * 100), 1),
        'simulated_total': round(simulated_total, 1),
        'observed_total': round(observed_total, 1),
        'ratio': (round(simulated_total / observed_total, 3)
                  if observed_total > 0 else None),
        # The 50 busiest links, for eyeballing in the summary file — unless the
        # caller needs all of them to compute from.
        'links': (sorted(comparisons, key=lambda c: -c['observed'])
                  if all_links
                  else sorted(comparisons, key=lambda c: -c['observed'])[:50]),
        # Distribution over *every* compared link, not just the top 50. The
        # truncated list above is fine for reading but wrong to compute from:
        # stratified %RMSE over the 50 busiest links describes the busiest
        # links, not the model. Measured on Anoka the two differ sharply —
        # 2,503 links compared, of which 392 carry no simulated trucks at all
        # and none of those appear in the top 50.
        'n_links_zero_simulated': int(sum(1 for c in comparisons
                                          if c['simulated'] == 0)),
        'median_observed': round(float(np.median(
            [c['observed'] for c in comparisons])), 1),
        'median_simulated': round(float(np.median(
            [c['simulated'] for c in comparisons])), 1),
    }


def compare_by_segment(
    freight_volumes: LinkVolumes,
    segment_groups: Sequence[Dict],
) -> Dict:
    """Tier-2 aggregate, counted once per HPMS segment rather than per link.

    **Why this exists.** A real network splits one HPMS segment into several
    links — the two carriageways of a divided highway, its ramps, and
    consecutive pieces of the same road. ``match_corridor_links`` credits each
    of those links with the segment's full AADT, which is right per link but
    makes the *aggregate* wrong in two different ways at once:

    - summing observed over links counts one segment's AADT once per link
      (measured on Anoka: 2,503 links over 884 segments, inflating the observed
      total 2.8x, which drove the reported ratio to 0.52 when it is 0.334);
    - summing simulated over links double-counts a truck that traverses several
      links of the same segment.

    AADT is a **flow across a cross-section, not a quantity to add up along a
    road**. So a segment's simulated volume is the ``max`` over its links, not
    the sum: the busiest cross-section is the one the published AADT describes.
    Parallel carriageways are the reason this is not simply "pick any link" —
    each carries part of the flow, and ``directional_aadt`` has already halved
    the observed figure for them, so max over links compares like with like.

    This does not replace the per-link comparison, which remains the right
    granularity for GEH on an individual link. It replaces the per-link
    *totals*.
    """
    if not segment_groups:
        return {'n_segments': 0,
                'note': 'no segment grouping available; matcher predates it'}

    observed_total = simulated_total = 0.0
    geh_values: List[float] = []
    n_zero = 0
    observed_values: List[float] = []
    simulated_values: List[float] = []

    for group in segment_groups:
        link_ids = group.get('link_ids') or []
        if not link_ids:
            continue
        observed = float(group.get('truck_aadt', 0.0))
        simulated = max(freight_volumes.daily(link_id) for link_id in link_ids)
        observed_total += observed
        simulated_total += simulated
        observed_values.append(observed)
        simulated_values.append(simulated)
        if simulated == 0:
            n_zero += 1
        if observed > 0:
            geh_values.append(geh(simulated, observed))

    if not observed_values:
        return {'n_segments': 0, 'note': 'no segment carried a matched link'}

    geh_array = np.array(geh_values) if geh_values else np.array([0.0])
    return {
        'n_segments': len(observed_values),
        'observed_total': round(observed_total, 1),
        'simulated_total': round(simulated_total, 1),
        'ratio': (round(simulated_total / observed_total, 3)
                  if observed_total > 0 else None),
        'geh_median': round(float(np.median(geh_array)), 2),
        'pct_geh_under_5': round(float((geh_array < 5).mean() * 100), 1),
        'pct_geh_under_10': round(float((geh_array < 10).mean() * 100), 1),
        'n_segments_zero_simulated': n_zero,
        'median_observed': round(float(np.median(observed_values)), 1),
        'median_simulated': round(float(np.median(simulated_values)), 1),
    }
