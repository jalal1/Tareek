#!/usr/bin/env python3
"""
Experiment summary builder and standalone refresher.

Builds experiment_summary.json for a MATSim experiment. Used in two ways:

1) From run_experiment.py at the end of a live run, where most fields come
   from in-memory state (plan generation counters, runtime timers, etc.).

2) As a standalone CLI for refreshing the matsim_output section of a
   completed experiment after fixes to the extraction logic. The other
   sections (plans, population, runtime, evaluation, ...) are not
   reconstructable from disk - we read the existing experiment_summary.json
   and preserve them as-is.

Usage (standalone):
    python -m utils.experiment_summary <experiment_dir>
    python utils/experiment_summary.py <experiment_dir>

Examples:
    python -m utils.experiment_summary experiments/experiment_20260303_124251
    python -m utils.experiment_summary E:/jetstream2_experiments/april2026/chicago/experiment_20260503_025136

The standalone path:
    - Reads <experiment_dir>/experiment_summary.json
    - Recomputes the matsim_output section from MATSim output files
      (output/, ITERS/, output_events.xml.gz, scorestats.csv, etc.)
    - Writes the file back, preserving all other sections
"""

import argparse
import csv
import gzip
import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MATSim output extraction (the part that IS reconstructable from disk)
# ---------------------------------------------------------------------------

def extract_matsim_output_stats(experiment_dir: Path) -> Dict:
    """
    Extract statistics from MATSim output files for a completed experiment.

    Reads output/ subdir (or, if not present, the experiment_dir itself - some
    runs flatten outputs to the top level). Pulls scores, distances, mode
    splits, peak demand, and stuck-agent counts.

    Args:
        experiment_dir: Path to the experiment directory.

    Returns:
        Dict of extracted stats. Empty dict if output files are missing.
    """
    stats: Dict = {}

    # MATSim output may live in <exp>/output/ (default) or directly in <exp>/
    # (some flattened layouts). Pick whichever has the typical files.
    candidates = [experiment_dir / 'output', experiment_dir]
    output_dir: Optional[Path] = None
    for c in candidates:
        if c.exists() and ((c / 'scorestats.csv').exists() or (c / 'ITERS').exists()):
            output_dir = c
            break
    if output_dir is None:
        logger.debug("No MATSim output directory with stats files found.")
        return stats

    try:
        # 0. Counts from gzipped CSVs (number of lines minus header).
        for fname, key in (
            ('output_persons.csv.gz', 'output_persons_count'),
            ('output_trips.csv.gz', 'output_trips_count'),
            ('output_legs.csv.gz', 'output_legs_count'),
        ):
            path = output_dir / fname
            if path.exists():
                with gzip.open(path, 'rt') as f:
                    line_count = sum(1 for _ in f) - 1
                stats[key] = line_count

        # Plan counts from output_plans.xml.gz (line scan, plans are one-per-line).
        plans_xml_path = output_dir / 'output_plans.xml.gz'
        if plans_xml_path.exists():
            selected_plan_count = 0
            total_plan_count = 0
            with gzip.open(plans_xml_path, 'rt') as f:
                for line in f:
                    if '<plan ' in line:
                        total_plan_count += 1
                        if 'selected="yes"' in line:
                            selected_plan_count += 1
            stats['output_executed_plans_count'] = selected_plan_count
            stats['output_total_plans_count'] = total_plan_count

        # 1. Score stats - one row per iteration.
        scorestats_path = output_dir / 'scorestats.csv'
        if scorestats_path.exists():
            with open(scorestats_path, 'r') as f:
                rows = list(csv.DictReader(f, delimiter=';'))
            if rows:
                final_row = rows[-1]
                stats['score_final_executed'] = round(float(final_row.get('avg_executed', 0)), 2)
                stats['score_final_best'] = round(float(final_row.get('avg_best', 0)), 2)
                stats['score_final_worst'] = round(float(final_row.get('avg_worst', 0)), 2)
                stats['score_final_average'] = round(float(final_row.get('avg_average', 0)), 2)
                if len(rows) > 1:
                    first_row = rows[0]
                    stats['score_improvement'] = round(
                        float(final_row.get('avg_executed', 0))
                        - float(first_row.get('avg_executed', 0)),
                        2,
                    )

        # 2. Travel distance stats (values in meters).
        traveldist_path = output_dir / 'traveldistancestats.csv'
        if traveldist_path.exists():
            with open(traveldist_path, 'r') as f:
                rows = list(csv.DictReader(f, delimiter=';'))
            if rows:
                final_row = rows[-1]
                stats['avg_leg_distance_km'] = round(
                    float(final_row.get('avg. Average Leg distance', 0)) / 1000, 2
                )
                stats['avg_trip_distance_km'] = round(
                    float(final_row.get('avg. Average Trip distance', 0)) / 1000, 2
                )

        # 3. Person-km by mode (sum across modes for total).
        pkm_path = output_dir / 'pkm_modestats.csv'
        if pkm_path.exists():
            with open(pkm_path, 'r') as f:
                rows = list(csv.DictReader(f, delimiter=';'))
            if rows:
                final_row = rows[-1]
                total_pkm = 0.0
                for key, value in final_row.items():
                    if key != 'Iteration' and value:
                        try:
                            total_pkm += float(value)
                        except ValueError:
                            pass
                stats['total_person_km'] = int(total_pkm)

        # 4. Person-hours traveling (only *_travel columns; *_wait excluded).
        ph_path = output_dir / 'ph_modestats.csv'
        if ph_path.exists():
            with open(ph_path, 'r') as f:
                rows = list(csv.DictReader(f, delimiter=';'))
            if rows:
                final_row = rows[-1]
                total_travel_hours = 0.0
                for key, value in final_row.items():
                    if key != 'Iteration' and 'travel' in key.lower() and value:
                        try:
                            total_travel_hours += float(value)
                        except ValueError:
                            pass
                stats['total_person_hours_traveling'] = int(total_travel_hours)

        # 5. Per-iteration files (legdurations, legHistogram, events).
        iters_dir = output_dir / 'ITERS'
        if iters_dir.exists():
            iter_dirs = [d for d in iters_dir.iterdir()
                         if d.is_dir() and d.name.startswith('it.')]
            if iter_dirs:
                final_iter = max(iter_dirs, key=lambda x: int(x.name.split('.')[1]))
                final_iter_num = int(final_iter.name.split('.')[1])
                stats['final_iteration'] = final_iter_num

                # 5a. Average leg duration (from legdurations.txt footer line).
                legdur_path = final_iter / f'{final_iter_num}.legdurations.txt'
                if legdur_path.exists():
                    with open(legdur_path, 'r') as f:
                        for line in f:
                            if 'average leg duration' in line.lower():
                                m = re.search(r'(\d+\.?\d*)\s*seconds', line)
                                if m:
                                    stats['avg_leg_duration_min'] = round(
                                        float(m.group(1)) / 60, 2
                                    )
                                break

                # 5b. Peak demand from legHistogram. The file has TWO columns
                # named "time" (HH:MM:SS, then seconds); csv.DictReader collapses
                # duplicates and keeps the seconds form, so we read by index.
                leghist_path = final_iter / f'{final_iter_num}.legHistogram.txt'
                if leghist_path.exists():
                    _extract_leg_histogram(leghist_path, stats)

                # 5c. Stuck agents - authoritative source is the events file.
                # legHistogram's stuck_all column double-counts, so we stream
                # the gzipped events file (cheap even for multi-GB files) and
                # count "stuckAndAbort" event lines.
                stuck_events_path = output_dir / 'output_events.xml.gz'
                if not stuck_events_path.exists():
                    per_iter = final_iter / f'{final_iter_num}.events.xml.gz'
                    stuck_events_path = per_iter if per_iter.exists() else None
                if stuck_events_path and stuck_events_path.exists():
                    stats['total_stuck_agents'] = _count_stuck_events(stuck_events_path)

        # 6. Public-transit sub-mode breakdown (bus/rail/tram/subway/...).
        # Detects schedule-conversion regressions where a sub-mode has fleet
        # supply but zero realised legs (canonical signature: pt2matsim
        # PublicTransitMapper dropped routes whose <transportMode> lacked a
        # transportModeAssignment). See network_generator._SCHEDULE_MODE_TO_FAMILY.
        transit_section = _build_transit_section(experiment_dir, output_dir)
        if transit_section is not None:
            stats['transit'] = transit_section

        # Sanity check: peak departures cannot exceed total persons.
        persons_count = stats.get('output_persons_count')
        peak_dep = stats.get('peak_departures_per_5min')
        if persons_count and peak_dep and peak_dep > persons_count:
            logger.warning(
                f"peak_departures_per_5min ({peak_dep}) exceeds total persons "
                f"({persons_count}) - legHistogram values may be inflated."
            )

    except Exception as e:
        logger.warning(f"Error extracting MATSim output stats: {e}")

    return stats


def _extract_leg_histogram(leghist_path: Path, stats: Dict) -> None:
    """Populate peak_departures_per_5min, peak_hour, max_en_route in stats."""
    with open(leghist_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        if not header:
            return

        def col_idx(name: str) -> Optional[int]:
            try:
                return header.index(name)
            except ValueError:
                return None

        idx_time_str = 0  # First "time" column = HH:MM:SS
        idx_dep = col_idx('departures_all')
        idx_enroute = col_idx('en-route_all')
        if idx_dep is None:
            return

        max_departures = 0
        max_en_route = 0
        peak_hour: Optional[str] = None

        for row in reader:
            if not row:
                continue
            try:
                departures = int(row[idx_dep])
                if departures > max_departures:
                    max_departures = departures
                    peak_hour = row[idx_time_str]
                if idx_enroute is not None:
                    en_route = int(row[idx_enroute])
                    if en_route > max_en_route:
                        max_en_route = en_route
            except (ValueError, IndexError):
                pass

        stats['peak_departures_per_5min'] = max_departures
        stats['peak_hour'] = peak_hour
        stats['max_en_route'] = max_en_route


def _count_stuck_events(events_path: Path) -> int:
    """
    Stream-count stuckAndAbort events in a (possibly multi-GB) gzipped events
    file. Iterating gzip.open line-by-line keeps memory flat; one substring
    check per line is the cheapest possible filter.
    """
    total = 0
    try:
        with gzip.open(events_path, 'rt', encoding='utf-8', errors='replace') as f:
            for line in f:
                if 'stuckAndAbort' in line:
                    total += 1
    except (OSError, EOFError) as e:
        logger.warning(f"Could not read events file {events_path}: {e}")
    return total


# ---------------------------------------------------------------------------
# Demand-side leg mode breakdown (from plans.xml)
# ---------------------------------------------------------------------------

# Match `<leg mode="..."` anywhere on a line. Plans are pretty-printed so
# legs land on their own line, but be tolerant of single-line files too.
_LEG_MODE_RE = re.compile(r'<leg\b[^>]*\bmode="([^"]+)"')


def build_generated_plans_modes_section(experiment_dir: Path) -> Optional[Dict]:
    """
    Demand-side leg counts by mode, parsed from plans.xml.

    This is what plan generation emitted BEFORE MATSim ran any iterations -
    i.e. the mode shares produced by ModeChoiceModel. Use the matsim_output
    section's transit/legs counts for the realised post-replanning shares.

    Streams the file line-by-line with a regex - plans.xml is pretty-printed
    one element per line, and the file is plain text (not gzipped). Memory
    is flat regardless of file size.

    Returns None if plans.xml is missing.
    """
    plans_path = experiment_dir / 'plans.xml'
    if not plans_path.exists():
        return None

    counts: Dict[str, int] = {}
    try:
        with open(plans_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if '<leg' not in line:
                    continue
                m = _LEG_MODE_RE.search(line)
                if m:
                    mode = m.group(1)
                    counts[mode] = counts.get(mode, 0) + 1
    except OSError as e:
        logger.warning(f"Could not read plans.xml: {e}")
        return None

    total = sum(counts.values())
    if total == 0:
        return None

    shares_pct = {m: round(n / total * 100, 2) for m, n in counts.items()}
    return {
        '_comment': 'Generated-leg counts by mode, parsed from plans.xml. '
                   'These are the demand-side shares produced by ModeChoiceModel '
                   'BEFORE MATSim replanning. For realised post-simulation shares '
                   'see matsim_output (output_legs_count, transit.pt_legs_total).',
        'counts': counts,
        'counts_comment': 'Number of legs per mode summed across all generated plans',
        'shares_pct': shares_pct,
        'shares_pct_comment': 'Mode shares as percentages of total legs. Sums to ~100.',
        'total_legs': total,
        'total_legs_comment': 'Total legs across all generated plans (all purposes)',
    }


# ---------------------------------------------------------------------------
# Public-transit sub-mode breakdown
# ---------------------------------------------------------------------------
#
# MATSim agent legs collapse all transit to a single ``pt`` mode, but the
# schedule and vehicle files preserve the underlying GTFS sub-mode (bus,
# rail, tram, subway, ferry, ...). To detect schedule-conversion regressions
# (e.g. tram routes silently dropped by pt2matsim's PublicTransitMapper —
# see network_generator._SCHEDULE_MODE_TO_FAMILY), we need to compare:
#
#   - Supply: routes/departures per <transportMode> in transitSchedule.xml
#             and vehicles per type in transitVehicles.xml
#   - Demand: pt legs grouped by vehicle_id suffix in output_legs.csv.gz
#
# A healthy run shows non-zero demand for every supply mode that has
# vehicles. Tram=0 demand with hundreds of Tram vehicles in supply is the
# canonical "routes were dropped during mapping" signature.

_VEH_NS = '{http://www.matsim.org/files/dtd}'

# vehicle_id suffix → schedule submode label (the suffix is set by pt2matsim's
# Gtfs2TransitSchedule based on GtfsDefinitions$RouteType). We map every
# canonical pt2matsim mode here; unknown suffixes fall through to "other".
_VEH_SUFFIX_TO_SUBMODE = {
    'bus': 'bus',
    'rail': 'rail',
    'tram': 'tram',
    'subway': 'subway',
    'ferry': 'ferry',
    'cable_car': 'cable_car',
    'gondola': 'gondola',
    'funicular': 'funicular',
}


def _extract_transit_supply(experiment_dir: Path) -> Optional[Dict]:
    """Parse transitSchedule.xml + transitVehicles.xml for per-submode supply.

    Returns None if neither file is present (non-transit run). Otherwise
    returns a dict like::

        {
            'routes_by_mode':     {'bus': 3096, 'rail': 4, ...},
            'departures_by_mode': {'bus': 8505, 'rail': 6, ...},
            'vehicles_by_type':   {'Bus': 7466, 'Rail': 6, 'Tram': 406, ...},
            'total_routes': 3100,
            'total_departures': 8511,
            'total_vehicles': 7878,
        }

    The schedule lives in the experiment dir (alongside network.xml), not
    in output/ — pt2matsim writes it once at network-generation time.
    """
    sched = experiment_dir / 'transitSchedule.xml'
    veh = experiment_dir / 'transitVehicles.xml'
    if not sched.exists() and not veh.exists():
        return None

    result: Dict = {
        'routes_by_mode': {},
        'departures_by_mode': {},
        'vehicles_by_type': {},
    }

    if sched.exists():
        try:
            tree = ET.parse(sched)
            root = tree.getroot()
            for line in root.findall('transitLine'):
                for route in line.findall('transitRoute'):
                    tm_elem = route.find('transportMode')
                    mode = tm_elem.text.strip() if tm_elem is not None and tm_elem.text else 'unknown'
                    result['routes_by_mode'][mode] = result['routes_by_mode'].get(mode, 0) + 1
                    deps_elem = route.find('departures')
                    n_deps = len(deps_elem.findall('departure')) if deps_elem is not None else 0
                    result['departures_by_mode'][mode] = result['departures_by_mode'].get(mode, 0) + n_deps
            result['total_routes'] = sum(result['routes_by_mode'].values())
            result['total_departures'] = sum(result['departures_by_mode'].values())
        except (ET.ParseError, OSError) as e:
            logger.warning(f"Could not parse transitSchedule.xml: {e}")

    if veh.exists():
        try:
            tree = ET.parse(veh)
            root = tree.getroot()
            # transitVehicles uses the matsim DTD namespace, but pt2matsim
            # output sometimes drops it — try both.
            for vehicle in root.iter():
                if vehicle.tag in (f'{_VEH_NS}vehicle', 'vehicle'):
                    vtype = vehicle.get('type')
                    if vtype:
                        result['vehicles_by_type'][vtype] = result['vehicles_by_type'].get(vtype, 0) + 1
            result['total_vehicles'] = sum(result['vehicles_by_type'].values())
        except (ET.ParseError, OSError) as e:
            logger.warning(f"Could not parse transitVehicles.xml: {e}")

    if not result['routes_by_mode'] and not result['vehicles_by_type']:
        return None
    return result


def _extract_pt_legs_by_submode(output_dir: Path) -> Optional[Dict]:
    """Stream output_legs.csv.gz and group pt legs by vehicle_id suffix.

    The vehicle_id format pt2matsim emits is ``veh_<n>_<suffix>`` where
    suffix is one of bus / rail / tram / subway / ferry / cable_car /
    gondola / funicular. The leg's ``mode`` column collapses everything
    to ``pt``, so the suffix is the only signal that distinguishes which
    transit type actually carried the agent.

    Returns None if output_legs.csv.gz is missing.
    """
    legs_path = output_dir / 'output_legs.csv.gz'
    if not legs_path.exists():
        return None

    by_submode: Dict[str, int] = {}
    pt_total = 0
    try:
        with gzip.open(legs_path, 'rt', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                if row.get('mode') != 'pt':
                    continue
                pt_total += 1
                vid = row.get('vehicle_id', '') or ''
                suffix = vid.rsplit('_', 1)[-1] if '_' in vid else ''
                submode = _VEH_SUFFIX_TO_SUBMODE.get(suffix, 'other')
                by_submode[submode] = by_submode.get(submode, 0) + 1
    except (OSError, EOFError) as e:
        logger.warning(f"Could not read output_legs.csv.gz: {e}")
        return None

    if pt_total == 0:
        return {'pt_legs_total': 0, 'legs_by_submode': {}}
    return {'pt_legs_total': pt_total, 'legs_by_submode': by_submode}


def _build_transit_section(experiment_dir: Path, output_dir: Path) -> Optional[Dict]:
    """Combine supply (schedule + vehicles) and demand (output_legs) into one
    transit section. Returns None when this is a non-transit run.
    """
    supply = _extract_transit_supply(experiment_dir)
    demand = _extract_pt_legs_by_submode(output_dir)
    if supply is None and demand is None:
        return None

    # Per-submode rollup: { submode: {routes, departures, vehicles, legs} }.
    # Vehicle-type ids are capitalised in transitVehicles.xml ("Bus", "Rail",
    # "Tram", ...); fold them down to lowercase so they line up with the
    # schedule's transportMode strings and the leg vehicle_id suffixes.
    submodes: Dict[str, Dict[str, int]] = {}

    if supply:
        for mode, n in supply.get('routes_by_mode', {}).items():
            submodes.setdefault(mode, {})['routes'] = n
        for mode, n in supply.get('departures_by_mode', {}).items():
            submodes.setdefault(mode, {})['departures'] = n
        for vtype, n in supply.get('vehicles_by_type', {}).items():
            key = vtype.lower().replace(' ', '_')
            submodes.setdefault(key, {})['vehicles'] = n

    if demand:
        for mode, n in demand.get('legs_by_submode', {}).items():
            submodes.setdefault(mode, {})['legs'] = n

    # Flag any submode that has supply but zero demand — the canonical
    # "routes silently dropped during mapping" signature.
    orphans = []
    for mode, stats in submodes.items():
        has_supply = stats.get('vehicles', 0) > 0 or stats.get('routes', 0) > 0
        no_demand = stats.get('legs', 0) == 0
        if has_supply and no_demand:
            orphans.append(mode)

    section = {'submodes': submodes}
    if supply:
        section['total_routes'] = supply.get('total_routes', 0)
        section['total_departures'] = supply.get('total_departures', 0)
        section['total_vehicles'] = supply.get('total_vehicles', 0)
    if demand:
        section['pt_legs_total'] = demand.get('pt_legs_total', 0)
    if orphans:
        section['submodes_with_supply_but_no_demand'] = sorted(orphans)
    return section


# ---------------------------------------------------------------------------
# Matsim_output section builder (shared by live + standalone paths)
# ---------------------------------------------------------------------------

def build_matsim_output_section(experiment_dir: Path) -> Optional[Dict]:
    """
    Build the matsim_output section of experiment_summary.json from disk.

    Returns None if no MATSim output is available (so the caller can skip
    the section), else a fully-populated dict with comments.
    """
    s = extract_matsim_output_stats(experiment_dir)
    if not s:
        return None

    return {
        '_comment': 'Statistics extracted from MATSim output files (scorestats.csv, '
                    'traveldistancestats.csv, legHistogram.txt, etc.)',

        'output_persons_count': s.get('output_persons_count'),
        'output_persons_count_comment': 'Number of persons (agents) in final output. From output_persons.csv.gz.',
        'output_executed_plans_count': s.get('output_executed_plans_count'),
        'output_executed_plans_count_comment': 'Number of executed plans (selected="yes") in final iteration. '
                                              'This equals the number of persons - one executed plan per person.',
        'output_total_plans_count': s.get('output_total_plans_count'),
        'output_total_plans_count_comment': 'Total plans in output_plans.xml.gz including non-selected alternatives. '
                                           'MATSim keeps multiple plans per person for learning/replanning.',
        'output_trips_count': s.get('output_trips_count'),
        'output_trips_count_comment': 'Total trips completed in simulation. From output_trips.csv.gz.',
        'output_legs_count': s.get('output_legs_count'),
        'output_legs_count_comment': 'Total legs (mode segments) in simulation. From output_legs.csv.gz.',

        'score_final_executed': s.get('score_final_executed'),
        'score_final_executed_comment': 'Average executed plan score in final iteration. Higher = better agent utility.',
        'score_final_best': s.get('score_final_best'),
        'score_final_best_comment': 'Average best plan score per agent in final iteration.',
        'score_improvement': s.get('score_improvement'),
        'score_improvement_comment': 'Score improvement from iteration 0 to final iteration. Positive = agents found better routes.',

        'avg_leg_distance_km': s.get('avg_leg_distance_km'),
        'avg_leg_distance_km_comment': 'Average distance per leg (single mode segment) in kilometers.',
        'avg_trip_distance_km': s.get('avg_trip_distance_km'),
        'avg_trip_distance_km_comment': 'Average distance per trip (origin to destination) in kilometers.',

        'avg_leg_duration_min': s.get('avg_leg_duration_min'),
        'avg_leg_duration_min_comment': 'Average travel time per leg in minutes. From legdurations.txt.',

        'total_person_km': s.get('total_person_km'),
        'total_person_km_comment': 'Total person-kilometers traveled by all agents. From pkm_modestats.csv.',
        'total_person_hours_traveling': s.get('total_person_hours_traveling'),
        'total_person_hours_traveling_comment': 'Total person-hours spent traveling. From ph_modestats.csv. '
                                                '(Excludes wait time; sum of *_travel columns only.)',

        'peak_departures_per_5min': s.get('peak_departures_per_5min'),
        'peak_departures_per_5min_comment': 'Maximum departures in any 5-minute interval. Indicates peak demand.',
        'peak_hour': s.get('peak_hour'),
        'peak_hour_comment': 'Time of day with highest departure rate (HH:MM:SS format).',
        'max_en_route': s.get('max_en_route'),
        'max_en_route_comment': 'Maximum number of agents traveling simultaneously. Indicates network load.',

        'total_stuck_agents': s.get('total_stuck_agents'),
        'total_stuck_agents_comment': 'Count of stuckAndAbort events from the final-iteration events file '
                                     '(authoritative). Each event = one agent giving up on a leg. Counted on the '
                                     'simulated sample (multiply by 1/scaling_factor for population-equivalent). '
                                     'Should be near 0 for good simulations.',
        'final_iteration': s.get('final_iteration'),
        'final_iteration_comment': 'Last completed iteration number. Should match config iterations setting.',

        'transit': s.get('transit'),
        'transit_comment': 'Public-transit sub-mode breakdown. Per-submode counts of routes and departures '
                           '(from transitSchedule.xml), vehicles (from transitVehicles.xml), and realised '
                           'pt legs (from output_legs.csv.gz, grouped by vehicle_id suffix). MATSim collapses '
                           'all transit to mode="pt" in plans/legs, so the suffix is the only signal that '
                           'distinguishes bus / rail / tram / subway / etc. '
                           'submodes_with_supply_but_no_demand flags the canonical "routes dropped during '
                           'pt2matsim mapping" regression — e.g. tram vehicles defined but zero tram legs.',
    }


# ---------------------------------------------------------------------------
# Live-run path: build the full summary from in-memory state
# ---------------------------------------------------------------------------

def _calculate_runtime_minutes(runtime: Dict, start_key: str, end_key: str) -> Optional[float]:
    start = runtime.get(start_key)
    end = runtime.get(end_key)
    if start and end:
        return round((end - start).total_seconds() / 60, 2)
    return None


def build_summary(
    experiment_id: str,
    experiment_dir: Path,
    config: Dict,
    plan_stats: Dict,
    population_stats: Dict,
    data_quality_stats: Dict,
    network_stats: Dict,
    plans_file_size_mb: float,
    runtime: Dict,
    metadata: Dict,
    evaluation_metrics: Optional[Dict] = None,
) -> Dict:
    """
    Build the full experiment_summary.json dict from in-memory state.

    Used by run_experiment.py at the end of a live run. The matsim_output
    section is always read from disk via build_matsim_output_section.
    """
    def get_stat(d: Optional[Dict], key: str, default=0):
        return d.get(key, default) if d else default

    def get_plan_count(d: Optional[Dict]) -> int:
        return d.get('plans_generated', 0) if d else 0

    # Aggregate plan-generation counters across all purposes.
    all_stats = [plan_stats.get(k, {}) for k in plan_stats.keys()]
    total_chain_retries = sum(get_stat(s, 'chain_retries') for s in all_stats)
    total_chain_retries_too_short = sum(get_stat(s, 'chain_retries_too_short') for s in all_stats)
    total_chain_retries_bad_structure = sum(get_stat(s, 'chain_retries_bad_structure') for s in all_stats)
    total_chain_retries_missing_purpose = sum(get_stat(s, 'chain_retries_missing_purpose') for s in all_stats)
    total_chain_retries_missing_work = sum(get_stat(s, 'chain_retries_missing_work') for s in all_stats)
    total_chain_retries_has_work = sum(get_stat(s, 'chain_retries_has_work') for s in all_stats)
    total_chain_retries_too_many_work = sum(get_stat(s, 'chain_retries_too_many_work') for s in all_stats)
    total_chain_attempts = sum(get_stat(s, 'chain_attempts') for s in all_stats)
    total_poi_retries = sum(get_stat(s, 'poi_retries') for s in all_stats)
    total_time_retries = sum(get_stat(s, 'time_retries') for s in all_stats)
    total_failed = sum(get_stat(s, 'failed_plans') for s in all_stats)
    total_generated = sum(get_plan_count(s) for s in all_stats)
    total_requested = total_generated + total_failed
    overall_success_rate = round((total_generated / total_requested * 100), 2) if total_requested > 0 else 100.0

    scaling_factor = config.get('plan_generation', {}).get('scaling_factor', 0.1)
    matsim_params = config.get('matsim', {}).get('configurable_params', {})

    unscaled_work = get_stat(plan_stats.get('work'), 'unscaled_trips', 0)
    nonwork_purposes_config = config.get('nonwork_purposes', {})
    nonwork_purpose_keys = [
        p.lower() for p, cfg in nonwork_purposes_config.items()
        if isinstance(cfg, dict) and cfg.get('enabled', False)
    ]
    unscaled_nonwork = sum(
        get_stat(plan_stats.get(p), 'unscaled_trips', 0) for p in nonwork_purpose_keys
    )

    plans_dict = {'work': get_plan_count(plan_stats.get('work'))}
    for purpose_key in nonwork_purpose_keys:
        plans_dict[purpose_key] = get_plan_count(plan_stats.get(purpose_key))

    # Aggregate mode_choice stats across purposes (trip-weighted) so the
    # demand estimator can recover empirical chain_reduction on the next run.
    mode_choice_counts: Dict[str, int] = {}
    total_mode_choices = 0
    purpose_keys_all = ['work'] + nonwork_purpose_keys
    for pk in purpose_keys_all:
        mc = (plan_stats.get(pk) or {}).get('mode_choice') or {}
        n = mc.get('total_mode_choices', 0) or 0
        dist = mc.get('mode_distribution') or {}
        if n <= 0 or not dist:
            continue
        total_mode_choices += n
        for mode_name, share in dist.items():
            mode_choice_counts[mode_name] = mode_choice_counts.get(mode_name, 0) + share * n
    mode_distribution = {
        k: round(v / total_mode_choices, 6)
        for k, v in mode_choice_counts.items()
    } if total_mode_choices > 0 else {}
    plans_dict.update({
        'total': total_generated,
        'success_rate': overall_success_rate,
        'chain_retries': total_chain_retries,
        'chain_retries_too_short': total_chain_retries_too_short,
        'chain_retries_bad_structure': total_chain_retries_bad_structure,
        'chain_retries_missing_purpose': total_chain_retries_missing_purpose,
        'chain_retries_missing_work': total_chain_retries_missing_work,
        'chain_retries_has_work': total_chain_retries_has_work,
        'chain_retries_too_many_work': total_chain_retries_too_many_work,
        'chain_attempts': total_chain_attempts,
        'poi_retries': total_poi_retries,
        'time_retries': total_time_retries,
    })

    summary = {
        'experiment_id': experiment_id,
        'created_at': datetime.now().isoformat(),

        'population': {
            '_comment': 'Population counts from census home location data (home_locs_dict). '
                       'Summed from all census blocks within configured counties.',
            'total_population': population_stats.get('total_population', 0),
            'total_population_comment': 'Sum of total_employees + total_non_employees across all home blocks',
            'total_employees': population_stats.get('total_employees', 0),
            'total_employees_comment': 'Sum of n_employees field from each home block (workers who commute)',
            'total_non_employees': population_stats.get('total_non_employees', 0),
            'total_non_employees_comment': 'Sum of non_employees field from each home block (non-workers: retirees, students, etc.)',
        },

        'unscaled_trips': {
            '_comment': 'Total trips BEFORE applying scaling_factor. Represents full population trip demand.',
            'work': unscaled_work,
            'work_comment': 'Unscaled work trips = total_employees (one work trip per employee)',
            'nonwork': unscaled_nonwork,
            'nonwork_comment': 'Sum of unscaled trips across all enabled nonwork purposes. '
                              'Each purpose: non_employees * trip_rate from survey data',
        },

        'plans': plans_dict,
        'plans_comment': {
            '_comment': 'Plan counts AFTER applying scaling_factor. plans = unscaled_trips * scaling_factor',
            'work': 'Number of work plans generated (employees * scaling_factor)',
            'shopping': 'Number of shopping plans (non_employees * shopping_trip_rate * scaling_factor)',
            'school': 'Number of school plans (non_employees * school_trip_rate * scaling_factor)',
            'socialrecreation': 'Number of social/recreation plans (non_employees * rate * scaling_factor)',
            'dining': 'Number of dining plans (non_employees * dining_trip_rate * scaling_factor)',
            'other': 'Number of other purpose plans (non_employees * other_trip_rate * scaling_factor)',
            'total': 'Sum of all plan counts across all purposes (work + all nonwork)',
            'success_rate': 'Percentage of successfully generated plans: (total / (total + failed)) * 100',
            'chain_retries': 'Total retries due to invalid activity chains (sum of all reasons below)',
            'chain_retries_too_short': 'Retries due to chain having fewer than 3 activities',
            'chain_retries_bad_structure': 'Retries due to chain not starting/ending with Home',
            'chain_retries_missing_purpose': 'Retries due to chain missing target nonwork purpose',
            'chain_retries_missing_work': 'Retries due to chain missing Work activity (work plans)',
            'chain_retries_has_work': 'Retries due to chain containing Work (nonwork plans)',
            'chain_retries_too_many_work': 'Retries due to chain exceeding max Work activities',
            'chain_attempts': 'Total chain sampling attempts (retries + successes)',
            'poi_retries': 'Total retries due to POI selection failures (no suitable POI found nearby)',
            'time_retries': 'Total retries due to time constraint violations (activities overlapping or out of bounds)',
        },

        'parameters': {
            '_comment': 'Key simulation parameters from config file',
            'scaling_factor': scaling_factor,
            'scaling_factor_comment': 'Fraction of full population to simulate. '
                                     'E.g., 0.1 means 10% sample. From config.plan_generation.scaling_factor',
            'iterations': matsim_params.get('lastIteration'),
            'iterations_comment': 'Number of MATSim iterations for route/plan optimization. '
                                 'From config.matsim.configurable_params.lastIteration',
            'flow_capacity_factor': matsim_params.get('qsim.flowCapacityFactor'),
            'flow_capacity_factor_comment': 'MATSim QSim flow capacity scaling. Should roughly match scaling_factor. '
                                           'Controls how many vehicles can pass a link per time unit.',
            'storage_capacity_factor': matsim_params.get('qsim.storageCapacityFactor'),
            'storage_capacity_factor_comment': 'MATSim QSim storage capacity scaling. Should be >= flow_capacity_factor. '
                                              'Controls how many vehicles can be on a link simultaneously.',
            'modes': {
                mode_name: {
                    'config_rate': (config.get('modes', {}).get(mode_name, {}) or {}).get('config_rate'),
                    'blend_weight': (config.get('modes', {}).get(mode_name, {}) or {}).get('blend_weight'),
                }
                for mode_name in ('bus', 'rail')
            },
            'modes_comment': 'Snapshot of config.modes.{bus,rail}.{config_rate,blend_weight} used '
                             'for this run. Demand estimator reads this together with '
                             'mode_choice.mode_distribution to back-calculate empirical '
                             'chain_reduction for the next run.',
        },

        'mode_choice': {
            '_comment': 'Mode-choice outputs aggregated across all purposes (trip-weighted). '
                        'Used by the demand estimator to derive empirical chain_reduction.',
            'total_mode_choices': total_mode_choices,
            'mode_distribution': mode_distribution,
            'mode_distribution_comment': 'Share of legs assigned to each mode across all '
                                         'generated plans, weighted by purpose volume.',
        },

        'data_quality': {
            '_comment': 'Metrics about input data quality and coverage',
            'home_blocks': data_quality_stats.get('home_blocks', 0),
            'home_blocks_comment': 'Number of census blocks in home_locs_dict (filtered by configured counties)',
            'home_blocks_with_coords': data_quality_stats.get('home_blocks_with_coords', 0),
            'home_blocks_with_coords_comment': 'Census blocks that have valid lat/lon coordinates (should equal home_blocks)',
            'total_pois': data_quality_stats.get('total_pois', 0),
            'total_pois_comment': 'Total POIs loaded from database and filtered to county bounding box + 5km buffer',
            'poi_activity_types': data_quality_stats.get('poi_activity_types', 0),
            'poi_activity_types_comment': 'Number of distinct activity types in POI data (e.g., Shopping, School, Dining)',
        },

        'network': {
            '_comment': 'Road network statistics from network.xml',
            'num_nodes': network_stats.get('num_nodes', 0),
            'num_nodes_comment': 'Number of network nodes (intersections). Parsed from <nodes> element in network.xml',
            'num_links': network_stats.get('num_links', 0),
            'num_links_comment': 'Number of network links (road segments). Parsed from <links> element in network.xml',
            'file_size_mb': network_stats.get('file_size_mb', 0),
            'file_size_mb_comment': 'Size of network.xml file in megabytes',
        },

        'plans_file_size_mb': plans_file_size_mb,
        'plans_file_size_mb_comment': 'Size of plans.xml file in megabytes. Larger files = more agents simulated.',

        'runtime': {
            '_comment': 'Execution time measurements in minutes',
            'total_min': _calculate_runtime_minutes(runtime, 'start_time', 'end_time'),
            'total_min_comment': 'Total experiment runtime from start to finish (includes all steps)',
            'plans_min': _calculate_runtime_minutes(runtime, 'plans_start', 'plans_end'),
            'plans_min_comment': 'Time spent generating activity plans (work + all nonwork purposes)',
            'matsim_min': _calculate_runtime_minutes(runtime, 'matsim_start', 'matsim_end'),
            'matsim_min_comment': 'Time spent running MATSim simulation (all iterations)',
            'eval_min': _calculate_runtime_minutes(runtime, 'eval_start', 'eval_end'),
            'eval_min_comment': 'Time spent on evaluation (comparing sim output to ground truth)',
        },

        'paths': {
            '_comment': 'Relative paths to experiment outputs (from experiment directory)',
            'network': 'network.xml',
            'plans': 'plans.xml',
            'output': 'output/',
            'output_comment': 'MATSim simulation output directory (contains ITERS/, linkstats, etc.)',
            'evaluation': 'evaluation/',
            'evaluation_comment': 'Evaluation results directory (comparison CSVs, plots, maps)',
        },

        'simulation_status': metadata.get('simulation_status', 'unknown'),
        'simulation_status_comment': 'Final status: completed, failed, or skipped',
    }

    plan_modes_section = build_generated_plans_modes_section(experiment_dir)
    if plan_modes_section is not None:
        summary['generated_plans_modes'] = plan_modes_section

    matsim_section = build_matsim_output_section(experiment_dir)
    if matsim_section is not None:
        summary['matsim_output'] = matsim_section

    if evaluation_metrics is not None:
        summary['evaluation'] = _build_evaluation_section(evaluation_metrics)

    return summary


def _build_evaluation_section(evaluation_metrics: Dict) -> Dict:
    return {
        '_comment': 'Comparison of simulated traffic volumes vs ground truth traffic counts. '
                   'Fields are ordered by how much they should influence a calibration '
                   'decision: overall level first, then how the error splits across '
                   'stations, then across the day, then the GEH pass rates, and only '
                   'then the remaining station-hour metrics.',

        # ── 1. Overall level: is there too much or too little traffic? ────
        'aggregate_sim_obs_ratio': evaluation_metrics.get('aggregate_sim_obs_ratio', 0),
        'aggregate_sim_obs_ratio_comment': 'Total simulated / total observed across every station-hour. '
                                           'The single headline number: >1 means the model puts MORE traffic on '
                                           'the network than the sensors saw, <1 means less. Read it together '
                                           'with stations_over_simulated below — a ratio near 1.0 can also be '
                                           'over- and under-simulating stations cancelling each other out.',
        'total_observed_volume': evaluation_metrics.get('total_observed_volume', 0),
        'total_simulated_volume': evaluation_metrics.get('total_simulated_volume', 0),
        'interquartile_mean_ratio': evaluation_metrics.get('interquartile_mean_ratio', 0),
        'interquartile_mean_ratio_comment': 'Interquartile mean of per-station sim/obs total-volume ratios. '
                                            'Drops bottom & top 25% of stations and averages the middle 50%. '
                                            'Robust to boundary stations; 1.0 = perfect, <0.7 = under-simulation. '
                                            'Used as the primary signal for demand_estimator scaling-factor decisions.',

        # ── 2. How the error splits across stations ───────────────────────
        'stations_over_simulated': evaluation_metrics.get('stations_over_simulated', 0),
        'stations_within_10pct': evaluation_metrics.get('stations_within_10pct', 0),
        'stations_under_simulated': evaluation_metrics.get('stations_under_simulated', 0),
        'stations_split_comment': 'Count of stations whose daily total is >1.1x observed (over), within '
                                  '0.9-1.1x (ok), and <0.9x (under). This is what tells "the whole model is '
                                  'too high" apart from "a few corridors are wrong". Adding demand when most '
                                  'stations already over-simulate makes the fit worse, not better.',
        'station_ratio_cv': evaluation_metrics.get('station_ratio_cv', 0),
        'station_ratio_cv_comment': 'Coefficient of variation (std/mean) of the per-station sim/obs ratios — '
                                    'the concentrated-vs-uniform test. Below ~0.35 every station is off by a '
                                    'similar factor, so a single global lever (scaling_factor) can correct the '
                                    'region. Above it the error sits on particular corridors and a global '
                                    'multiplier would drag the already-correct stations off target. '
                                    'demand_estimator refuses to change scaling_factor when this is high or absent.',

        # ── 3. How the error splits across the day ────────────────────────
        'ratio_night': evaluation_metrics.get('ratio_night', 0),
        'ratio_morning': evaluation_metrics.get('ratio_morning', 0),
        'ratio_midday': evaluation_metrics.get('ratio_midday', 0),
        'ratio_evening': evaluation_metrics.get('ratio_evening', 0),
        'time_block_comment': 'Sim/obs ratio within each block: night 0-3, morning 4-9, midday 10-17, '
                              'evening 18-23. A model can match the daily total while being badly wrong '
                              'hour by hour. If midday is near 1.0 while the peaks are high and night is '
                              'near zero, the problem is WHEN trips depart, not how many there are — '
                              'a departure-time question, not a scaling_factor one.',
        'hourly_ratio_spread': evaluation_metrics.get('hourly_ratio_spread', 0),
        'hourly_ratio_spread_comment': 'Max minus min of the 24 hourly sim/obs ratios. Near 0 means the daily '
                                       'profile has the right shape and only its level is off; large means the '
                                       'shape itself is wrong.',
        'worst_hour': evaluation_metrics.get('worst_hour', 0),
        'worst_hour_ratio': evaluation_metrics.get('worst_hour_ratio', 0),

        # ── 4. Per-station GEH — pass rates, never an average ─────────────
        'station_daily_geh_lt5_pct': evaluation_metrics.get('station_daily_geh_lt5_pct', 0),
        'station_daily_geh_comment': 'Percent of stations whose 24-hour TOTAL scores GEH < 5. Each station is '
                                     'tested against the threshold on its own, so this is a count of passing '
                                     'stations rather than an average. No mean or median GEH is reported at any '
                                     'level: GEH grows with volume, so averaging it across stations of different '
                                     'size measures station size as much as model error, and the result is not '
                                     'the GEH of any real count. Individual per-station GEH values are in the '
                                     'device reports and spatial_overview.png.',

        # ── 5. Station-hour metrics — volume-sensitive, read with care ────
        'num_devices': evaluation_metrics.get('num_devices'),
        'num_devices_comment': 'Number of ground truth traffic count devices successfully matched to network links',
        'num_comparisons': evaluation_metrics.get('num_comparisons'),
        'num_comparisons_comment': 'Total hourly comparison points (num_devices * hours with valid data)',
        'geh_lt_5_pct': round(evaluation_metrics.get('geh_lt_5_pct', 0), 2),
        'geh_lt_5_pct_comment': 'Percentage of hourly link counts with GEH < 5 — the industry-standard form of '
                               'the convention, which is defined for one count over one period. Target >=85%.',
        'correlation': round(evaluation_metrics.get('correlation', 0), 3),
        'correlation_comment': 'Pearson correlation coefficient between simulated and observed volumes, on '
                              'station-hours. Aggregating to daily totals typically reads 0.04-0.05 higher, '
                              'so state which one is being quoted.',
        'mae': round(evaluation_metrics.get('mae', 0), 2),
        'mae_comment': 'Mean Absolute Error in vehicles/hour. Average |simulated - observed| across all comparisons.',
        'rmse': round(evaluation_metrics.get('rmse', 0), 2),
        'rmse_comment': 'Root Mean Square Error in vehicles/hour. Penalizes large errors more than MAE. '
                       'RMSE = sqrt(mean((simulated - observed)^2))',
        'mean_pct_error': round(evaluation_metrics.get('mean_pct_error', 0), 2),
        'mean_pct_error_comment': 'Mean percentage error: mean((simulated - observed) / observed * 100). '
                                 'Negative = underestimating traffic, Positive = overestimating traffic. '
                                 'Sensitive to boundary stations with through-traffic from outside the modeled '
                                 'area; prefer interquartile_mean_ratio for calibration decisions.',
        'median_station_ratio': evaluation_metrics.get('median_station_ratio', 0),
        'median_station_ratio_comment': 'Median of per-station sim/obs ratios. Companion to interquartile_mean_ratio.',
        'pct_stations_below_10pct': evaluation_metrics.get('pct_stations_below_10pct', 0),
        'pct_stations_below_10pct_comment': 'Percent of count stations whose simulated total is below 10% of observed. '
                                            'High values indicate boundary stations capturing through-traffic from '
                                            'outside the modeled region (cannot be fixed by raising demand).',
        'num_stations_below_10pct': evaluation_metrics.get('num_stations_below_10pct', 0),
        'station_ratio_p10': evaluation_metrics.get('station_ratio_p10', 0),
        'station_ratio_p90': evaluation_metrics.get('station_ratio_p90', 0),
        'station_ratio_p10_p90_comment': '10th and 90th percentile of per-station sim/obs ratios — the raw spread '
                                         'behind station_ratio_cv.',
        'num_stations_with_ratio': evaluation_metrics.get('num_stations_with_ratio', 0),
        'peak_hour_correlation': round(evaluation_metrics.get('peak_hour_correlation', 0), 3),
        'peak_hour_correlation_comment': 'Correlation restricted to peak hours (6-9, 15-18).',
        'geh_valid_count': evaluation_metrics.get('geh_valid_count', 0),
        'zero_observed_count': evaluation_metrics.get('zero_observed_count', 0),
        'num_directional_counts': evaluation_metrics.get('num_directional_counts', 0),
        'num_directional_counts_comment': 'Count locations INCLUDING direction (one physical sensor reports each '
                                          'travel direction separately), so this is typically 2x num_devices.',
        # Carry through anything the evaluator produced that is not named above,
        # so a new metric cannot silently fail to reach this file the way
        # station_ratio_cv did. Comment keys are skipped.
        **{k: v for k, v in evaluation_metrics.items()
           if k not in _EVALUATION_EXPLICIT_KEYS
           and k not in _EVALUATION_DROPPED_KEYS
           and not k.endswith('_comment')},
    }


# Fields _build_evaluation_section names explicitly (with their own comments).
# Everything else the evaluator emits is carried through verbatim.
_EVALUATION_EXPLICIT_KEYS = frozenset({
    'num_devices', 'num_comparisons', 'geh_lt_5_pct', 'correlation',
    'mae', 'rmse', 'mean_pct_error', 'interquartile_mean_ratio',
    'median_station_ratio', 'pct_stations_below_10pct', 'num_stations_below_10pct',
    'station_ratio_cv', 'station_ratio_p10', 'station_ratio_p90',
    'num_stations_with_ratio', 'peak_hour_correlation', 'geh_valid_count',
    'zero_observed_count', 'num_directional_counts',
    'aggregate_sim_obs_ratio', 'total_observed_volume', 'total_simulated_volume',
    'stations_over_simulated', 'stations_within_10pct', 'stations_under_simulated',
    'ratio_night', 'ratio_morning', 'ratio_midday', 'ratio_evening',
    'hourly_ratio_spread', 'worst_hour', 'worst_hour_ratio',
    'station_daily_geh_lt5_pct',
})

# Averaged GEH fields, retired. A GEH is defined for one count over one period
# and grows with volume, so any average across stations of different sizes
# measures station size as much as model error. These are dropped on the way
# into the summary so that a stale evaluator — or a re-summarised older run —
# cannot reintroduce them through the catch-all carry-through above.
_EVALUATION_DROPPED_KEYS = frozenset({
    'mean_geh', 'station_daily_geh_mean', 'station_daily_geh_median',
})


def write_summary(summary: Dict, summary_path: Path) -> None:
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Standalone path: refresh matsim_output in an existing summary
# ---------------------------------------------------------------------------

def refresh_matsim_output(experiment_dir: Path) -> Path:
    """
    Read existing experiment_summary.json, recompute the matsim_output and
    generated_plans_modes sections from disk, and write back. Other sections
    are preserved as-is because they are not reconstructable post-hoc (live
    timers, in-memory plan-gen counters, etc.).

    Returns the path of the refreshed summary file.
    """
    summary_path = experiment_dir / 'experiment_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(f"No experiment_summary.json found in {experiment_dir}")

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    matsim_section = build_matsim_output_section(experiment_dir)
    if matsim_section is None:
        logger.warning(
            f"No MATSim output files found under {experiment_dir} - "
            "matsim_output section not refreshed."
        )
    else:
        summary['matsim_output'] = matsim_section
        summary['matsim_output_refreshed_at'] = datetime.now().isoformat()

    plan_modes_section = build_generated_plans_modes_section(experiment_dir)
    if plan_modes_section is not None:
        summary['generated_plans_modes'] = plan_modes_section

    # Drop legacy section names superseded by current schema. Older runs
    # wrote `legs_by_mode` (in-memory aggregation, since removed in favour
    # of the plans.xml-based `generated_plans_modes`). Refresh removes it
    # so summaries don't carry stale or empty sections forward.
    summary.pop('legs_by_mode', None)

    write_summary(summary, summary_path)
    return summary_path


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Refresh the matsim_output section of an existing experiment_summary.json."
    )
    parser.add_argument(
        'experiment_dir',
        type=Path,
        help="Path to experiment directory (contains experiment_summary.json + MATSim output).",
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s %(name)s: %(message)s',
    )

    if not args.experiment_dir.exists():
        print(f"ERROR: experiment directory does not exist: {args.experiment_dir}",
              file=sys.stderr)
        return 1

    try:
        out_path = refresh_matsim_output(args.experiment_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Refreshed matsim_output section in: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(_main())
