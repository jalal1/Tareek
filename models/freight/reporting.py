"""Freight reporting metrics — what practice expects a truck model to report.

The existing evaluator reports total link volume only: GEH, RMSE, MAE, %GEH<5
and correlation, all against `countscompare.txt`, which carries no vehicle
breakdown. That is the right set for a passenger model and an incomplete set for
one that carries freight.

This module adds the metrics that travel-model validation practice expects when
a model has more than one vehicle class. Each is standard, not invented here:

**VMT by vehicle class.** The headline output of practically every regional
model, and the basis on which emissions inventories are built. FHWA's VMT
forecasting method and SCAG's heavy-duty truck model both validate by comparing
class VMT against HPMS. Without it a freight model cannot be compared to any
published figure.

**Truck percentage by functional class.** The number HPMS publishes and the one
every DOT quotes. It is also the natural check on this module: we *derive*
demand from a truck share, so the realised share is a closed loop on the input.

**%RMSE by volume group.** The FHWA *Travel Model Validation and Reasonableness
Checking Manual* reports error stratified by volume group rather than pooled,
because a single aggregate figure hides the fact that error is systematically
worse on low-volume links. Standard practice, and absent here today.

**Screenline / cordon totals.** Peak-hour volumes crossing a cordon should be
within ~10% of counts (MDOT uses 5% for screenlines, 10% for cutlines). For a
boundary freight model the cordons *are* the screenline, which makes this the
most direct test of whether the module put the right number of trucks in.

**Trip length distribution.** Mean trip length by class, compared against the
QRFM/survey expectation. A model can match volumes at cordons and still have the
wrong spatial distribution; only trip length exposes that.

See docs/freight/design.md §5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from models.freight.events import HOURS_PER_DAY, LinkVolumes
from utils.logger import setup_logger

logger = setup_logger(__name__)

#: FHWA volume groups for stratified error reporting, from the Travel Model
#: Validation and Reasonableness Checking Manual. Error is reported per group
#: because a pooled figure hides systematically worse error on low-volume links.
VOLUME_GROUPS: Tuple[Tuple[str, float, float], ...] = (
    ('under_5k', 0.0, 5_000.0),
    ('5k_10k', 5_000.0, 10_000.0),
    ('10k_25k', 10_000.0, 25_000.0),
    ('25k_50k', 25_000.0, 50_000.0),
    ('over_50k', 50_000.0, float('inf')),
)

#: Maximum desirable %RMSE per volume group, same source. Low-volume links are
#: allowed far more relative error because a few vehicles move the percentage.
RMSE_TARGETS: Dict[str, float] = {
    'under_5k': 100.0,
    '5k_10k': 45.0,
    '10k_25k': 35.0,
    '25k_50k': 27.0,
    'over_50k': 25.0,
}


def percent_rmse(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Percent root mean square error, the standard assignment-validation stat.

    Normalised by the *mean observed* volume, which is what makes it comparable
    across volume groups and across studies.
    """
    simulated = np.asarray(simulated, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if simulated.size == 0 or observed.mean() <= 0:
        return float('nan')
    rmse = np.sqrt(np.mean((simulated - observed) ** 2))
    return float(rmse / observed.mean() * 100.0)


def rmse_by_volume_group(comparisons: Sequence[Dict]) -> Dict:
    """Stratify %RMSE by observed volume group, against FHWA targets.

    Args:
        comparisons: dicts with ``simulated`` and ``observed`` keys.
    """
    result: Dict[str, Dict] = {}
    for name, low, high in VOLUME_GROUPS:
        subset = [c for c in comparisons
                  if low <= float(c['observed']) < high]
        if not subset:
            continue
        simulated = np.array([float(c['simulated']) for c in subset])
        observed = np.array([float(c['observed']) for c in subset])
        value = percent_rmse(simulated, observed)
        target = RMSE_TARGETS[name]
        result[name] = {
            'n_links': len(subset),
            'pct_rmse': round(value, 1),
            'target': target,
            'meets_target': bool(value <= target) if np.isfinite(value) else False,
            'mean_observed': round(float(observed.mean()), 1),
            'mean_simulated': round(float(simulated.mean()), 1),
        }
    return result


def vmt_by_class(
    streams: Dict[str, LinkVolumes],
    link_lengths_m: Dict[str, float],
) -> Dict:
    """Vehicle miles travelled per vehicle class.

    The headline output of a regional model and the basis of every emissions
    inventory built on one. Reported in both miles and kilometres because US
    practice quotes VMT while the network is metric.

    Args:
        streams: ``{'freight': LinkVolumes, 'car': ..., 'total': ...}``.
        link_lengths_m: link length in metres, from the network.
    """
    result: Dict[str, Dict] = {}

    for name, volumes in streams.items():
        metres = 0.0
        matched_links = 0
        for link_id in volumes.volumes:
            length = link_lengths_m.get(link_id)
            if length is None:
                continue
            metres += volumes.daily(link_id) * length
            matched_links += 1

        km = metres / 1000.0
        result[name] = {
            'vmt_miles': round(km * 0.621371, 1),
            'vkt_km': round(km, 1),
            'n_links_matched': matched_links,
            'n_links_total': len(volumes.volumes),
        }

    total_vmt = result.get('total', {}).get('vmt_miles', 0.0)
    if total_vmt > 0:
        for name in result:
            result[name]['share_of_total_pct'] = round(
                result[name]['vmt_miles'] / total_vmt * 100.0, 2)

    return result


def truck_percentage_by_class(
    streams: Dict[str, LinkVolumes],
    link_functional_class: Dict[str, str],
) -> Dict:
    """Realised truck share per functional class — the HPMS-comparable number.

    This closes the loop on the module's own input: demand is *derived* from a
    truck share, so the realised share on the network is a direct check that
    the derivation and the assignment agree. A large gap means trucks are being
    routed onto different roads than the share was measured on.
    """
    freight = streams.get('freight')
    total = streams.get('total')
    if freight is None or total is None:
        return {}

    by_class: Dict[str, Dict[str, float]] = {}
    for link_id in total.volumes:
        functional_class = link_functional_class.get(link_id, 'unknown')
        entry = by_class.setdefault(functional_class,
                                    {'freight': 0.0, 'total': 0.0, 'n_links': 0})
        entry['freight'] += freight.daily(link_id)
        entry['total'] += total.daily(link_id)
        entry['n_links'] += 1

    return {
        name: {
            'truck_pct': round(values['freight'] / values['total'] * 100.0, 2)
            if values['total'] > 0 else 0.0,
            'freight_volume': round(values['freight'], 1),
            'total_volume': round(values['total'], 1),
            'n_links': int(values['n_links']),
        }
        for name, values in sorted(by_class.items())
    }


def trip_length_distribution(trips: Sequence, utm_converter=None) -> Dict:
    """Mean and distribution of trip length, per class.

    A model can match volumes at every cordon and still distribute trips
    wrongly inside the region; trip length is what exposes that. Compared
    against the QRFM expectation (heavy trucks ~33 min mean trip) it is also a
    check on the friction parameter.
    """
    from models.freight.demand import TRIP_CLASSES

    by_class: Dict[str, List[float]] = {name: [] for name in TRIP_CLASSES}

    for trip in trips:
        # Great-circle distance is enough here: this is a distribution check,
        # not a routing measurement, and the network distance is longer than
        # the straight line by a roughly constant factor.
        km = _haversine_km(trip.origin_lat, trip.origin_lon,
                           trip.dest_lat, trip.dest_lon)
        by_class.setdefault(trip.trip_class, []).append(km)

    result: Dict[str, Dict] = {}
    for name, lengths in by_class.items():
        if not lengths:
            continue
        array = np.array(lengths)
        result[name] = {
            'n_trips': len(array),
            'mean_km': round(float(array.mean()), 2),
            'median_km': round(float(np.median(array)), 2),
            'p10_km': round(float(np.percentile(array, 10)), 2),
            'p90_km': round(float(np.percentile(array, 90)), 2),
            'max_km': round(float(array.max()), 2),
        }

    all_lengths = [v for values in by_class.values() for v in values]
    if all_lengths:
        result['all'] = {
            'n_trips': len(all_lengths),
            'mean_km': round(float(np.mean(all_lengths)), 2),
            'median_km': round(float(np.median(all_lengths)), 2),
        }
    return result


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (np.sin(dphi / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2)
    return float(2 * radius * np.arcsin(np.sqrt(a)))


def cordon_screenline_check(
    cordons: Sequence,
    freight_volumes: LinkVolumes,
    tolerance_pct: float = 10.0,
    demand_scale: float = 1.0,
) -> Dict:
    """Simulated freight across each cordon against the volume it was given.

    For a boundary freight model the cordons **are** the screenline, so this is
    the most direct test that the module put the right number of trucks in.
    Practice puts screenline agreement within 5-10%.

    **``demand_scale`` must be passed or this check is circular.** A cordon's
    weight is the *full* observed truck AADT, but the model deliberately
    generates only ``demand_scale`` of it. Comparing the two directly reports
    that deliberate reduction as if it were an error: measured on Anoka at
    ``demand_scale=0.2``, the raw ratio came out 0.281 and would have driven
    calibration toward 1.0 no matter how well the model was performing.

    Scaling the expectation asks the question that actually matters: *given the
    demand we chose to generate, did it reach the network and cross the cordons
    it was assigned to?* A residual above 1.0 then means trucks cross more often
    than their trip count implies — re-crossing en route — rather than meaning
    the demand was too small.

    Args:
        demand_scale: the fraction of observed volume the run generated. Pass
            ``freight.demand_scale``; the default of 1.0 is correct only for a
            run that generates full observed volume.

    A failure here means the demand was generated but did not reach the network
    — trucks failing to route, or being assigned to a link the cordon does not
    cover.
    """
    rows = []
    for cordon in cordons:
        simulated = sum(freight_volumes.daily(link_id)
                        for link_id in cordon.link_ids)
        expected = float(cordon.weight) * float(demand_scale)
        deviation = ((simulated - expected) / expected * 100.0
                     if expected > 0 else float('nan'))
        rows.append({
            'cordon_id': cordon.cordon_id,
            'direction': cordon.direction,
            'compass': cordon.compass,
            'expected': round(expected, 1),
            'simulated': round(simulated, 1),
            'deviation_pct': (round(deviation, 1)
                              if np.isfinite(deviation) else None),
            'within_tolerance': bool(abs(deviation) <= tolerance_pct)
            if np.isfinite(deviation) else False,
        })

    valid = [r for r in rows if r['deviation_pct'] is not None]
    n_within = sum(1 for r in valid if r['within_tolerance'])

    return {
        'n_cordons': len(rows),
        'n_evaluated': len(valid),
        'n_within_tolerance': n_within,
        'pct_within_tolerance': (round(n_within / len(valid) * 100.0, 1)
                                 if valid else 0.0),
        'tolerance_pct': tolerance_pct,
        'total_expected': round(sum(r['expected'] for r in rows), 1),
        'total_simulated': round(sum(r['simulated'] for r in rows), 1),
        'worst': sorted(valid, key=lambda r: -abs(r['deviation_pct']))[:10],
    }


def hourly_class_shares(streams: Dict[str, LinkVolumes]) -> Dict:
    """Freight share of traffic per hour, and the caution that goes with it.

    Recorded because it is the number people reach for, and it is the one most
    easily misread: truck *percentage* peaks at night (30-50% of rural traffic
    in the small hours) because car volume collapses, not because truck volume
    rises. The absolute counts are reported alongside so the share is never
    read on its own.
    """
    freight = streams.get('freight')
    total = streams.get('total')
    if freight is None or total is None:
        return {}

    freight_hourly = np.zeros(HOURS_PER_DAY)
    total_hourly = np.zeros(HOURS_PER_DAY)
    for link_id in total.volumes:
        freight_hourly += freight.hourly(link_id)
        total_hourly += total.hourly(link_id)

    with np.errstate(divide='ignore', invalid='ignore'):
        share = np.where(total_hourly > 0,
                         freight_hourly / total_hourly * 100.0, 0.0)

    return {
        'freight_counts': [round(float(v), 1) for v in freight_hourly],
        'total_counts': [round(float(v), 1) for v in total_hourly],
        'freight_share_pct': [round(float(v), 2) for v in share],
        'peak_freight_count_hour': int(freight_hourly.argmax()),
        'peak_freight_share_hour': int(share.argmax()),
        'caution': ('Validate against truck COUNTS per hour, never truck share '
                    'per hour: the share peaks at night because car volume '
                    'collapses, not because truck volume rises.'),
    }


def build_report(
    streams: Dict[str, LinkVolumes],
    trips: Sequence,
    cordons: Sequence,
    link_lengths_m: Optional[Dict[str, float]] = None,
    link_functional_class: Optional[Dict[str, str]] = None,
    observed_truck_aadt: Optional[Dict[str, float]] = None,
    demand_scale: float = 1.0,
) -> Dict:
    """Assemble every freight metric into one reportable dict.

    Args:
        demand_scale: passed through to the screenline so its expectation is
            the volume this run *chose* to generate, not full observed AADT.
            See cordon_screenline_check.
    """
    from models.freight.events import compare_against_observed

    report: Dict = {
        'hourly': hourly_class_shares(streams),
        'trip_length': trip_length_distribution(trips),
        'screenline': cordon_screenline_check(cordons, streams['freight'],
                                              demand_scale=demand_scale),
    }

    if link_lengths_m:
        report['vmt'] = vmt_by_class(streams, link_lengths_m)
    if link_functional_class:
        report['truck_pct_by_functional_class'] = truck_percentage_by_class(
            streams, link_functional_class)
    if observed_truck_aadt:
        comparison = compare_against_observed(streams['freight'],
                                              observed_truck_aadt)
        report['hpms_comparison'] = comparison
        if comparison.get('links'):
            report['rmse_by_volume_group'] = rmse_by_volume_group(
                comparison['links'])

    report['digest'] = network_effect_digest(report, streams)
    return report


def network_effect_digest(report: Dict, streams: Dict[str, LinkVolumes]) -> Dict:
    """The handful of numbers that belong in an experiment report.

    ``build_report`` produces the full picture — per-link comparisons, per-class
    VMT, hourly profiles — which is the right level of detail for a
    freight-specific file and far too much for a report section someone reads
    once. This flattens it to what a reader needs to answer "what did freight
    actually do to this network?".

    Kept separate from the report writer so the same digest serves the JSON
    summary and the Markdown report without either owning the arithmetic.
    """
    freight = streams.get('freight')
    total = streams.get('total')

    digest: Dict = {}

    if freight is not None and total is not None:
        freight_entries = freight.total()
        total_entries = total.total()
        if total_entries > 0:
            digest['link_entry_share_pct'] = round(
                freight_entries / total_entries * 100.0, 2)

    vmt = report.get('vmt') or {}
    if 'freight' in vmt:
        digest['vmt_miles'] = vmt['freight'].get('vmt_miles')
        digest['vmt_share_pct'] = vmt['freight'].get('share_of_total_pct')

    trip_length = (report.get('trip_length') or {}).get('all') or {}
    if trip_length.get('mean_km') is not None:
        digest['mean_trip_length_km'] = trip_length['mean_km']

    screenline = report.get('screenline') or {}
    expected = screenline.get('total_expected')
    simulated = screenline.get('total_simulated')
    if expected:
        digest['screenline_ratio'] = round(simulated / expected, 3)
        digest['screenline_pct_within_tolerance'] = screenline.get(
            'pct_within_tolerance')

    hpms = report.get('hpms_comparison') or {}
    if hpms.get('n_links'):
        # Tier 2: the only figure here judged against volumes the demand was
        # not derived from, which is what makes it the real validation.
        digest['hpms_n_links'] = hpms.get('n_links')
        digest['hpms_ratio'] = hpms.get('ratio')
        digest['hpms_pct_geh_under_5'] = hpms.get('pct_geh_under_5')

    return digest
