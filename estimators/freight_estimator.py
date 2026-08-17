"""Freight Estimator — calibrates the boundary-freight parameters that have a
measured path, and refuses to guess at the ones that do not.

This replaces ``run_freight_tier2.py``, which measured ``demand_scale`` from a
finished experiment and printed it for a human to copy into a config by hand.
That is what an estimator does, and this repo already has the pattern, so tier 2
lives here now.

**What it estimates, and on what evidence**

===================  ===========  =================================================
parameter            mode         measured against
===================  ===========  =================================================
``truck_share``      cold start   HPMS, through the four-layer resolver. Pinning it
                                  with its provenance makes a published run
                                  reproducible with no network at all (design §3).
``vehicle_mix``      cold start   The single-unit / combination split from the same
                                  HPMS query. The shipped 45:55 measures ~28:72 on
                                  Alabama urban interstate, and the value becomes
                                  load-bearing the moment PCE is on.
``demand_scale``     feedback     Tier 2: simulated truck volumes against observed
                                  HPMS truck AADT, **per HPMS segment**, on
                                  corridors the demand was not derived from.
===================  ===========  =================================================

**What it deliberately does not estimate.** Each was measured before being
dropped, and the reasoning is recorded so it is not re-proposed:

- ``od_matrix.beta`` — inert at this scale. Sweeping it to its zero limit moves
  far-zone mass 8% relative and adds 5 effective zones out of 2,187.
- ``pce.enabled`` — never auto-enabled. At 2.05 against a 0.15-capacity network
  it gridlocked the fast tier (completed trips −22.6% by iteration 1).
- ``class_shares.through`` — **computed and reported, never written.** The
  generator produces through trips at exactly the configured rate, so measuring
  the output returns the input; writing it back would look like calibration
  while learning nothing. It is reported because a *divergence* is a real
  generator bug — the angular pairing test can silently drop every through trip.
  Settling it needs an external source (FAF5), which is not wired in.
- ``cordon.min_peripherality``, ``hpms_match.radius_m`` — sensitivity knobs with
  no ground truth to fit against. A looser radius matches more cordons and some
  of them are the wrong road, which fails silently.

**On the events file.** Tier 2 needs per-link truck volumes, and no MATSim
output carries them (see ``models/freight/link_volumes.py`` for what each output
does and does not hold). So the estimator parses the events file — but exactly
once per experiment, caching the result as ``freight_link_volumes.json``. A
re-run reads that in milliseconds instead of re-parsing 2.1 GB.

Usage:
    # Cold start — positional is the base config JSON. No experiment needed;
    # estimates truck_share and vehicle_mix from HPMS.
    python estimators/freight_estimator.py config/USA/TwinCities/config_twin.json

    # Feedback — positional is the region FOLDER; adds demand_scale from tier 2.
    python estimators/freight_estimator.py config/USA/TwinCities \
        --experiment-dir experiments/tc15_on_0816_203943

Output:
    Cold start:  <config_dir>/<stem>_estimated.json
    Feedback:    <region_folder>/config_estimated.json

See docs/freight/design.md §3 and §5.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from estimators.demand_estimator import (
    TeeWriter,
    apply_recommendations,
    resolve_estimator_inputs,
    _resolve_experiment_dir,
)

#: Below this the model puts materially less truck volume on corridors than HPMS
#: observes; above it, materially more. Inside the band demand_scale is left
#: alone — GEH tolerates more relative error than a ratio does, and moving a
#: level control by a few percent against a shape residual is noise.
RATIO_BAND = 0.15

#: A ratio computed from a handful of segments is noise, not a measurement.
MIN_SEGMENTS = 20


def _freight_enabled(config: Dict[str, Any]) -> bool:
    return bool((config.get('freight') or {}).get('enabled', False))


# ---------------------------------------------------------------------------
# Cold start: truck_share and vehicle_mix, from HPMS
# ---------------------------------------------------------------------------

def estimate_truck_share(config: Dict[str, Any]) -> Tuple[Optional[Any], List[Dict]]:
    """Resolve the region's truck share and SU/CU mix through the HPMS layers.

    Returns ``(TruckShare | None, recommendations)``. The resolver is total by
    contract — it always returns a usable value — so None here means only that
    the region has no counties configured to resolve for.
    """
    from models.freight.truck_share import (
        SOURCE_CONFIG_PINNED, SOURCE_NATIONAL_TABLE, resolve_truck_share)
    from models.freight.plans import _state_abbr

    counties = (config.get('region') or {}).get('counties') or []
    if not counties:
        print("  region.counties is empty; cannot resolve a truck share.")
        return None, []

    state_fips = counties[0][:2]
    county_codes = [geoid[2:5] for geoid in counties]

    share = resolve_truck_share(
        config, state_fips, county_codes, state_abbr=_state_abbr(state_fips))

    freight = config.get('freight') or {}
    recommendations: List[Dict] = []

    print(f"  truck share    : {share.total * 100:.2f}%  "
          f"(single-unit {share.single_unit * 100:.2f}%, "
          f"combination {share.combination * 100:.2f}%)")
    print(f"  source         : {share.source}")
    if share.f_system is not None:
        print(f"  functional class: f_system={share.f_system}"
              f"{'R' if share.is_rural else 'U'}")

    if share.source == SOURCE_CONFIG_PINNED:
        # Already pinned. Re-writing it would be a no-op that looks like an
        # estimate, and the resolver derived the split *from* the config, so
        # the mix below would be circular too.
        print("  truck_share is already pinned in config; nothing to estimate.")
        return share, []

    if share.source == SOURCE_NATIONAL_TABLE:
        # The 1997 national table understates freight-heavy regions ~2x
        # (Alabama urban interstate: 8.50% table vs 16.34% measured). Pinning
        # it would freeze that error into the config and hide its provenance.
        print("  Source is the 1997 national table, NOT a measurement for this")
        print("  region. Refusing to pin it — leaving truck_share null keeps the")
        print("  resolver free to reach HPMS on a later run.")
        return share, []

    current = freight.get('truck_share')
    if current is None:
        recommendations.append({
            'parameter': 'freight.truck_share',
            'current': None,
            'recommended': round(share.total, 5),
            'reason': (
                f"Resolved from {share.source}: "
                f"{share.total * 100:.2f}% truck share for f_system="
                f"{share.f_system}{'R' if share.is_rural else 'U'}. Pinning it "
                f"makes this run reproducible without network access. Set to "
                f"null to resolve live again."
            ),
        })

    # vehicle_mix must be written whenever truck_share is pinned. resolve_truck_share
    # reads vehicle_mix to split a pinned total back into SU and CU, so pinning
    # the total while leaving the mix at its 45:55 default would silently
    # re-impose that default over the measured split.
    mix = share.mix
    current_mix = freight.get('vehicle_mix') or {}
    current_su = float(current_mix.get('single_unit', 0.45))
    if abs(mix['single_unit'] - current_su) > 0.01:
        recommendations.append({
            'parameter': 'freight.vehicle_mix.single_unit',
            'current': current_su,
            'recommended': round(mix['single_unit'], 4),
            'reason': (
                f"Measured SU:CU split from {share.source} is "
                f"{mix['single_unit'] * 100:.0f}:{mix['combination'] * 100:.0f}, "
                f"against the configured "
                f"{current_su * 100:.0f}:{(1 - current_su) * 100:.0f}. Affects "
                f"reporting today and the capacity each truck consumes once "
                f"pce.enabled is true."
            ),
        })
        recommendations.append({
            'parameter': 'freight.vehicle_mix.combination',
            'current': round(1.0 - current_su, 4),
            'recommended': round(mix['combination'], 4),
            'reason': (
                f"Complement of the measured single-unit share "
                f"({share.source})."
            ),
        })

    return share, recommendations


# ---------------------------------------------------------------------------
# Feedback: demand_scale, from tier 2
# ---------------------------------------------------------------------------

def run_tier2(
    experiment_dir: Path,
    config: Dict[str, Any],
    *,
    force_reparse: bool = False,
) -> Optional[Dict]:
    """Tier-2 comparison for a finished experiment.

    Reads ``freight_link_volumes.json`` when it exists and extracts it from the
    events file when it does not — so the 20-minute parse happens once per
    experiment rather than once per estimator run.

    Returns the tier-2 payload, or None when the experiment cannot support it.
    """
    from models.freight.events import compare_against_observed, compare_by_segment
    from models.freight.hpms_match import resolve_corridor_truck_aadt
    from models.freight.link_volumes import (
        artefact_to_link_volumes, extract_link_volumes_artefact)
    from models.freight.reporting import rmse_by_volume_group
    from models.freight.validation import validate_tier2

    summary_path = experiment_dir / 'freight_summary.json'
    if not summary_path.exists():
        print(f"  No freight_summary.json in {experiment_dir} — was freight "
              f"enabled for this run?")
        return None

    summary = json.loads(summary_path.read_text(encoding='utf-8'))

    utm_epsg = (config.get('coordinates') or {}).get('utm_epsg')
    if not utm_epsg:
        print("  config has no coordinates.utm_epsg; cannot project HPMS "
              "segments to the network CRS.")
        return None

    scaling = float((config.get('plan_generation') or {}).get('scaling_factor', 1.0))
    scale_factor = 1.0 / scaling if scaling > 0 else 1.0

    from utils.coordinates import CoordinateConverter
    converter = CoordinateConverter(utm_epsg=utm_epsg)

    cordons = _load_cordons(summary)
    network = experiment_dir / 'network.xml'

    print(f"  Matching corridor links to HPMS (scale_factor {scale_factor:.3f})...")
    observed, match_stats = resolve_corridor_truck_aadt(
        config, cordons, network, converter)

    if not observed:
        print(f"  No corridor link matched an HPMS segment: {match_stats}")
        print("  Tier 2 cannot run for this region.")
        return None

    print(f"    {match_stats['n_matched']:,} of "
          f"{match_stats['n_corridor_links']:,} corridor links matched")

    artefact = extract_link_volumes_artefact(
        experiment_dir, scale_factor=scale_factor, force=force_reparse)
    freight_volumes = artefact_to_link_volumes(artefact)

    comparison = compare_against_observed(freight_volumes, observed, all_links=True)
    by_segment = compare_by_segment(freight_volumes,
                                    match_stats.get('segment_groups', []))
    report = validate_tier2(comparison)

    payload = {
        'tier2': report.to_dict(),
        'by_segment': by_segment,
        'comparison': {k: v for k, v in comparison.items() if k != 'links'},
        'match_stats': {k: v for k, v in match_stats.items()
                        if k != 'segment_groups'},
        'freight_share_pct': (artefact.get('totals') or {}).get('freight_share_pct'),
        'coverage': {
            'n_links_zero_simulated': comparison.get('n_links_zero_simulated'),
            'median_observed': comparison.get('median_observed'),
            'median_simulated': comparison.get('median_simulated'),
        },
    }
    if comparison.get('links'):
        payload['rmse_by_volume_group'] = rmse_by_volume_group(comparison['links'])
        payload['top_links'] = comparison['links'][:50]

    payload['_summary'] = summary
    return payload


def _load_cordons(summary: Dict):
    """Rebuild Cordon objects from freight_summary.json, for the HPMS bbox."""
    from models.freight.cordons import Cordon

    cordons = []
    for record in (summary.get('cordons') or {}).get('detail', []):
        cordons.append(Cordon(
            cordon_id=record['cordon_id'],
            x=record['x'], y=record['y'],
            direction=record['direction'], compass=record['compass'],
            link_ids=record.get('link_ids', []),
            capacity=record.get('capacity', 0.0),
            weight=record.get('weight', 0.0),
        ))
    return cordons


def print_tier2(payload: Dict) -> None:
    """Print the tier-2 result. Kept faithful to the retired runner's output,
    because these are the numbers recorded in the design doc's ground truth."""
    by_segment = payload.get('by_segment') or {}
    comparison = payload.get('comparison') or {}
    match_stats = payload.get('match_stats') or {}

    print()
    print('=' * 66)
    print('TIER 2 — simulated truck volumes vs observed HPMS truck AADT')
    print('=' * 66)

    if by_segment.get('n_segments'):
        # AADT is a flow across a cross-section, so it is counted once per HPMS
        # segment. The per-link view below locates error; it must not be used
        # for totals, which is the bug that produced a bogus implied 0.385.
        print('  PER SEGMENT (the aggregate to read)')
        print(f"  segments compared   : {by_segment['n_segments']:,}"
              f"  (from {comparison.get('n_links', 0):,} links,"
              f" {match_stats.get('links_per_segment')} per segment)")
        print(f"  simulated total     : {by_segment['simulated_total']:,.0f}")
        print(f"  observed total      : {by_segment['observed_total']:,.0f}")
        print(f"  ratio sim/observed  : {by_segment['ratio']}")
        print(f"  GEH median          : {by_segment['geh_median']}")
        print(f"  %seg GEH < 5        : {by_segment['pct_geh_under_5']}")
        zs = by_segment['n_segments_zero_simulated']
        print(f"  segments NO trucks  : {zs:,} "
              f"({zs / by_segment['n_segments'] * 100:.0f}%)")
        print()

    print('  PER LINK (for locating error, not for totals)')
    print(f"  links compared      : {comparison.get('n_links', 0):,}")
    print(f"  simulated total     : {comparison.get('simulated_total', 0):,.0f}")
    print(f"  observed total      : {comparison.get('observed_total', 0):,.0f}")
    print(f"  ratio sim/observed  : {comparison.get('ratio')}")
    print(f"  GEH median          : {comparison.get('geh_median')}")
    print(f"  %links GEH < 5      : {comparison.get('pct_geh_under_5')}")
    print()

    zero = comparison.get('n_links_zero_simulated') or 0
    n_links = comparison.get('n_links') or 1
    print(f"  links with NO trucks: {zero:,} of {n_links:,} "
          f"({zero / n_links * 100:.0f}%)")
    print(f"  median observed     : {comparison.get('median_observed'):,.0f}")
    print(f"  median simulated    : {comparison.get('median_simulated'):,.0f}")


def estimate_demand_scale(payload: Dict) -> List[Dict]:
    """Turn the tier-2 ratio into a demand_scale recommendation.

    Read from the **per-segment** aggregate. The per-link ratio's denominator
    counts one HPMS segment's AADT once per matched link, which on Anoka
    inflates the observed total 2.8x.
    """
    by_segment = payload.get('by_segment') or {}
    comparison = payload.get('comparison') or {}
    summary = payload.get('_summary') or {}

    ratio = by_segment.get('ratio')
    basis = 'per-segment'
    n_segments = by_segment.get('n_segments') or 0
    if not ratio:
        ratio = comparison.get('ratio')
        basis = 'per-link'
        n_segments = comparison.get('n_links') or 0

    if not ratio:
        print("\n  No usable ratio; demand_scale cannot be estimated.")
        return []

    current = float((summary.get('demand') or {}).get('demand_scale', 1.0) or 1.0)
    implied = current / ratio

    print()
    print('=' * 66)
    print('WHAT THIS SAYS ABOUT demand_scale')
    print('=' * 66)
    print(f"  current demand_scale : {current}")
    print(f"  ratio sim/observed   : {ratio:.3f}  ({basis})")
    print(f"  implied demand_scale : {implied:.3f}")
    print()

    if n_segments < MIN_SEGMENTS:
        print(f"  Only {n_segments} segments compared (need {MIN_SEGMENTS}).")
        print("  Too thin to calibrate against — reporting only.")
        return []

    # The caveat that must travel with this number. Measured on Anoka: applying
    # the implied scale made the aggregate perfect (ratio 1.001) and the fit
    # worse (GEH median 38.9 -> 42.6, %GEH<5 7.9 -> 3.3). One multiplier matches
    # the median by construction, overshoots the busiest quarter ~23%, and
    # leaves the bottom quartile at 2-38% of observed.
    caveat = (
        "demand_scale is a LEVEL control and the residual error is SHAPE: on "
        "Anoka, applying the implied value made the aggregate ratio 1.001 and "
        "the per-link fit worse (GEH median 38.9 -> 42.6). Defensible for total "
        "VMT; quote the per-link error next to it."
    )

    if abs(ratio - 1.0) <= RATIO_BAND:
        print(f"  Truck volumes agree with HPMS within {RATIO_BAND:.0%}. "
              f"demand_scale is about right as configured.")
        return []

    direction = ('MORE' if ratio > 1.0 else 'LESS')
    print(f"  The model puts {direction} truck volume on corridors than HPMS")
    print(f"  observes. {'Lower' if ratio > 1.0 else 'Raise'} demand_scale "
          f"toward the implied value.")
    print()
    print("  Unlike the cordon screenline, this compares against volumes the")
    print("  demand was NOT derived from, so it can judge the level.")
    print()
    print(f"  CAVEAT: {caveat}")

    return [{
        'parameter': 'freight.demand_scale',
        'current': current,
        'recommended': round(implied, 4),
        'reason': (
            f"Tier 2 ({basis}, {n_segments} segments): simulated truck volume "
            f"is {ratio:.3f} of observed HPMS truck AADT at demand_scale "
            f"{current}, implying {implied:.3f}. {caveat}"
        ),
    }]


# ---------------------------------------------------------------------------
# Reported, never written
# ---------------------------------------------------------------------------

def report_through_share(config: Dict[str, Any], summary: Dict) -> None:
    """Compare the configured through share against the realised one.

    **Not an estimate, and never written.** The generator draws E->E trips at
    exactly the configured rate, so the realised share reproduces the input;
    writing it back would be a no-op wearing the clothes of a calibration.

    It is reported because a *divergence* is a real defect. The design's angular
    pairing test is the failure mode to watch: a metric version of it rejects
    legitimate crossings and silently drops every through trip, which shows up
    here as a realised share of 0.00 against a configured 0.30 and nowhere else.

    Settling the value needs an external source. FAF5 was surveyed and deferred:
    its regions are whole metros, so a through trip across our boundary is
    internal to a FAF zone and invisible in zone-to-zone flows.
    """
    configured = ((config.get('freight') or {}).get('class_shares') or {})
    realised = ((summary.get('realised') or {}).get('class_shares') or {})
    if not realised:
        return

    print()
    print('=' * 66)
    print('CLASS SPLIT — reported, not estimated')
    print('=' * 66)
    drift = False
    for name in ('external_to_internal', 'internal_to_external', 'through'):
        want = configured.get(name)
        got = realised.get(name)
        if want is None or got is None:
            continue
        delta = abs(float(got) - float(want))
        flag = '  <-- DRIFT' if delta > 0.02 else ''
        if flag:
            drift = True
        print(f"  {name:22s} configured {float(want):.3f}   "
              f"realised {float(got):.3f}{flag}")

    print()
    if drift:
        print("  A realised share that differs from the configured one is a")
        print("  GENERATOR BUG, not a calibration signal. Check the cordon")
        print("  pairing — the angular test can drop every through trip.")
    else:
        print("  Realised matches configured, so the generator is doing what it")
        print("  was told. This is a health check, not a measurement: the")
        print("  realised share is derived FROM the configured one, so it can")
        print("  never disagree unless something is broken.")
    print()
    print("  class_shares.through is therefore NOT written. Measuring it needs")
    print("  an external source; FAF5 was surveyed and deferred (its regions")
    print("  are whole metros, so within-region through trips are invisible).")


# ---------------------------------------------------------------------------
# Config output
# ---------------------------------------------------------------------------

def update_estimated_config(
    config: Dict[str, Any],
    estimated_path: Path,
    recommendations: List[Dict[str, Any]],
) -> Path:
    """Merge recommendations onto the existing estimated config and write it.

    Merging rather than overwriting is what lets the orchestrator run several
    estimators in sequence without each one discarding the last one's work.

    **The freight block is seeded before merging when the base lacks it.**
    Measured on the server: an existing ``config_estimated.json`` written by the
    demand and mode-share estimators has no ``freight`` section, and
    ``apply_recommendations`` sets a dotted path by creating only the parents it
    needs — so ``freight.demand_scale`` alone produced a **three-key** freight
    block where the source config had twelve. Every unwritten key would then
    fall back to its default, which is exactly the failure the design warns
    about: a missing key does not fail, it falls back, and the run looks fine.
    """
    if estimated_path.exists():
        with open(estimated_path) as f:
            base = json.load(f)
        print(f"  Updating existing estimated config: {estimated_path.name}")
    else:
        base = copy.deepcopy(config)
        print(f"  Creating new estimated config: {estimated_path.name}")

    # Seed the whole freight section from the config actually read, so estimated
    # values land on a complete block rather than creating a partial one. Keys
    # already in the base win: a prior estimator's freight value is more current
    # than the source config's.
    source_freight = config.get('freight')
    if isinstance(source_freight, dict):
        base_freight = base.get('freight') or {}
        if not base_freight:
            print("  Seeding the full freight block — the base config had none, "
                  "and a partial block would silently fall back to defaults.")
        merged = copy.deepcopy(source_freight)
        merged.update(base_freight)
        base['freight'] = merged

    new_config = apply_recommendations(base, recommendations)
    new_config.pop("_config_dir", None)

    with open(estimated_path, "w") as f:
        json.dump(new_config, f, indent=2)

    return estimated_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freight estimator — truck_share and vehicle_mix from HPMS, "
                    "demand_scale from tier-2 validation"
    )
    parser.add_argument(
        "config_or_region",
        help="Cold start: path to config JSON. Feedback (with --experiment-dir): "
             "path to the region folder.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="A finished experiment folder. Enables tier 2 and the demand_scale "
             "estimate. The estimator reads <experiment-dir>/config_used.json as "
             "the state to update and writes <region>/config_estimated.json.",
    )
    parser.add_argument(
        "--force-reparse",
        action="store_true",
        help="Re-extract freight_link_volumes.json from the events file even if "
             "the artefact already exists (~20 min at 15-county scale).",
    )
    args = parser.parse_args()

    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tee = TeeWriter(logs_dir / f"freight_estimator_{timestamp}.log")
    sys.stdout = tee

    try:
        read_from, write_to = resolve_estimator_inputs(
            args.config_or_region, args.experiment_dir)

        with open(read_from) as f:
            config = json.load(f)

        print("=" * 70)
        print("  FREIGHT ESTIMATOR")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Config read : {read_from}")
        print(f"  Mode        : "
              f"{'FEEDBACK' if args.experiment_dir else 'COLD START'}")
        print("=" * 70)

        if not _freight_enabled(config):
            # Not an error: most regions run without freight, and an estimator
            # that wrote freight parameters into a config with freight off would
            # be changing a section the run never reads.
            print()
            print("  freight.enabled is false — nothing to estimate.")
            print("  Enable freight in the config to calibrate it.")
            sys.exit(0)

        recommendations: List[Dict] = []

        # -- cold-start parameters (also refreshed in feedback mode) ---------
        print()
        print("-" * 70)
        print("  TRUCK SHARE AND VEHICLE MIX (HPMS)")
        print("-" * 70)
        _, share_recs = estimate_truck_share(config)
        recommendations += share_recs

        # -- feedback-only parameters ---------------------------------------
        if args.experiment_dir:
            experiment_dir = _resolve_experiment_dir(args.experiment_dir)
            print()
            print("-" * 70)
            print("  TIER 2 — HPMS CORRIDOR VALIDATION")
            print("-" * 70)
            payload = run_tier2(experiment_dir, config,
                                force_reparse=args.force_reparse)
            if payload:
                print_tier2(payload)
                recommendations += estimate_demand_scale(payload)
                report_through_share(config, payload.get('_summary') or {})

                out = experiment_dir / 'freight_tier2.json'
                serialisable = {k: v for k, v in payload.items()
                                if not k.startswith('_')}
                out.write_text(json.dumps(serialisable, indent=2),
                               encoding='utf-8')
                print()
                print(f"  Tier-2 detail written to {out}")

        # -- write ----------------------------------------------------------
        print()
        print("=" * 70)
        print("  RECOMMENDATIONS")
        print("=" * 70)
        if not recommendations:
            print("  No parameter changed. config_estimated.json not modified.")
        else:
            for rec in recommendations:
                print(f"  {rec['parameter']}")
                print(f"    {rec['current']}  ->  {rec['recommended']}")
                print(f"    {rec['reason']}")
                print()
            path = update_estimated_config(config, write_to, recommendations)
            print(f"  Written: {path}")

        print()
    finally:
        sys.stdout = tee.terminal
        tee.close()


if __name__ == "__main__":
    main()
