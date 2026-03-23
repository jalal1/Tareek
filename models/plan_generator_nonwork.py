"""
MATSim Non-Work Plan Generator

Generates synthetic non-work activity plans (Shopping, Recreation, etc.) in MATSim XML format.
Integrates non-work OD matrices, POI weighting, time models, and chain sampling.

Key Differences from Work Trip Generator:
1. Uses non_employees instead of n_employees
2. Samples actual POI coordinates (not centroid + jitter)
3. Uses POI importance weighting
4. Different time distributions (midday/afternoon vs morning rush)
5. Shorter activity durations (~30-60 min vs 4-8 hours)
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import load_config
from data_sources.base_survey_trip import BaseSurveyTrip
from data_sources.survey_manager import SurveyManager
from models.home_locs_v2 import load_home_locations_by_counties
from models.chains import TripChainModel, BlendedTripChainModel, process_trip_chains, filter_chains_by_type
from models.time import TripDurationModel, ActivityDurationModel, BlendedTripDurationModel, BlendedActivityDurationModel
from models.poi_manager import POIManager
from models.od_matrix_nonwork import create_nonwork_od_matrix, generate_samples_from_od_matrix, calculate_blended_survey_trip_rate
from models.models import initialize_tables
from models.mode_choice import ModeChoiceModel, Leg
from models.mode_availability import Location
from models.gtfs_availability import GTFSAvailabilityManager
from utils.logger import setup_logger
from utils.poi_spatial_index import POISpatialIndex
from utils.poi_weighting import POIWeighting
from utils.coordinates import CoordinateConverter

logger = setup_logger(__name__)


def _filter_pois_by_bounds(poi_data: List[Dict], home_locs_dict: Dict, buffer_km: float = 5.0) -> List[Dict]:
    """
    Filter POIs to only include those within the bounding box of home locations.

    This prevents POIs from outside the selected counties being incorrectly
    assigned to blocks within the selected counties.

    Args:
        poi_data: List of POI dictionaries with 'lat' and 'lon' keys
        home_locs_dict: Dict of home blocks with lat/lon coordinates (already filtered by county)
        buffer_km: Buffer distance in km to add to bounding box (default 5km)

    Returns:
        Filtered list of POI dictionaries
    """
    if not home_locs_dict or not poi_data:
        return poi_data

    # Calculate bounding box from home locations
    lats = [data['lat'] for data in home_locs_dict.values() if data.get('lat')]
    lons = [data['lon'] for data in home_locs_dict.values() if data.get('lon')]

    if not lats or not lons:
        logger.warning("No valid coordinates in home locations, skipping POI filtering")
        return poi_data

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # Add buffer (approximate: 1 degree ≈ 111 km)
    buffer_deg = buffer_km / 111.0
    min_lat -= buffer_deg
    max_lat += buffer_deg
    min_lon -= buffer_deg
    max_lon += buffer_deg

    # Filter POIs within bounding box
    filtered_pois = [
        poi for poi in poi_data
        if (poi.get('lat') is not None and poi.get('lon') is not None and
            min_lat <= poi['lat'] <= max_lat and
            min_lon <= poi['lon'] <= max_lon)
    ]

    logger.info(f"  POI bounding box filter:")
    logger.info(f"    Lat: [{min_lat:.4f}, {max_lat:.4f}]")
    logger.info(f"    Lon: [{min_lon:.4f}, {max_lon:.4f}]")
    logger.info(f"    Buffer: {buffer_km} km")
    logger.info(f"    POIs before: {len(poi_data):,}")
    logger.info(f"    POIs after: {len(filtered_pois):,}")
    logger.info(f"    POIs filtered out: {len(poi_data) - len(filtered_pois):,}")

    return filtered_pois


@dataclass
class Activity:
    """Represents a single activity in a plan."""
    type: str
    x: float
    y: float
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    max_dur: Optional[str] = None

@dataclass
class Plan:
    """Represents a complete daily plan for one person."""
    person_id: str
    activities: List[Activity] = field(default_factory=list)
    legs: List[Leg] = field(default_factory=list)


def _worker_process_chunk_nonwork(args: Tuple) -> Tuple[List[Plan], Dict]:
    """
    Worker function for multiprocessing non-work plan generation.

    Re-initializes the worker generator in each process to avoid pickling issues
    with database connections and other unpicklable objects.

    Args:
        args: Tuple of (od_chunk, config_data, shared_data, log_file_path, purpose)
            - od_chunk: List of (origin_bg, dest_bg, num_trips) tuples
            - config_data: Configuration dictionary
            - shared_data: Dict with pre-loaded data (home_locs, poi_data, etc.)
            - log_file_path: Path to the main log file
            - purpose: Trip purpose (e.g., 'Shopping', 'Recreation')

    Returns:
        Tuple of (plans, stats) where:
            - plans: List of generated plans for this chunk
            - stats: Dict with worker statistics (failed_plans, retries, etc.)
    """
    import multiprocessing
    import logging
    import sys
    from pathlib import Path

    # Extract arguments
    od_chunk, config_data, shared_data, log_file_path, purpose = args

    # CRITICAL: Re-initialize logger for this worker process
    worker_id = multiprocessing.current_process().name

    # Get logging settings from config
    log_to_file = config_data.get('logging', {}).get('log_to_file', True)
    log_to_console = config_data.get('logging', {}).get('log_to_console', True)

    # Setup root logger for this worker
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Create formatter with worker ID
    formatter = logging.Formatter(
        f'%(asctime)s - [{worker_id}] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (only if log_to_console is True)
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)

    # File handler (only if log_to_file is True and log_file_path is provided)
    if log_to_file and log_file_path:
        try:
            file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            root_logger.addHandler(file_handler)
        except Exception as e:
            # File logging failed - fall back to stderr since logger may not be set up yet
            import sys as _sys
            print(f"Worker {worker_id}: Failed to setup file logging: {e}", file=_sys.stderr)

    # Get logger for this module
    worker_logger = logging.getLogger('models.plan_generator_nonwork')
    worker_logger.info(f"Worker {worker_id} started - processing {len(od_chunk)} OD pairs for {purpose}")

    # Create a worker generator instance for this process
    worker_gen = _WorkerNonWorkPlanGenerator(config_data, shared_data, purpose)

    chunk_plans = []
    for origin_bg, dest_bg, num_trips in od_chunk:
        plans = worker_gen.process_od_pair(origin_bg, dest_bg, num_trips)
        chunk_plans.extend(plans)

    worker_logger.info(f"Worker {worker_id} completed: generated {len(chunk_plans)} plans")

    # Return plans, stats, and mode choice stats from this worker
    worker_stats = worker_gen.stats
    worker_stats['mode_choice'] = worker_gen.mode_choice.get_stats_summary()
    return chunk_plans, worker_stats


class _WorkerNonWorkPlanGenerator:
    """
    Lightweight non-work plan generator for multiprocessing workers.

    Re-initializes only what's needed, avoiding unpicklable objects.
    Uses pre-loaded POI data to avoid database multi-connection issues.
    """

    def __init__(self, config: Dict, shared_data: Dict, purpose: str):
        """
        Initialize worker with shared data.

        Args:
            config: Configuration dictionary
            shared_data: Pre-loaded shared data from main process
            purpose: Trip purpose (e.g., 'Shopping', 'Recreation')
        """
        self.config = config
        self.purpose = purpose
        self.verbose = config.get('logging', {}).get('verbose', False)

        # Get purpose-specific configuration
        self.purpose_config = config.get('nonwork_purposes', {}).get(purpose, {})

        # Get shared data
        self.home_locs_dict = shared_data['home_locs_dict']
        self.poi_data_grouped = shared_data['poi_data_grouped']
        self.avg_trip_duration_min = shared_data['avg_trip_duration_min']

        # Re-initialize models with shared data
        survey_df = shared_data['survey_df']
        persons = shared_data['persons']
        chains_df = shared_data['chains_df']
        home_boost = config.get('chains', {}).get('home_boost_factor', 2.0)
        early_stop_exp = config.get('chains', {}).get('early_stop_exponent', 2.0)
        bw_method = config.get('time_models', {}).get('kde_bandwidth', 'scott')

        if 'per_source_data' in shared_data:
            # Multi-source: rebuild blended models
            per_source_data = shared_data['per_source_data']
            per_source_persons = shared_data['per_source_persons']
            blend_weights = shared_data['blend_weights']

            # Chain model — filter for purpose and exclude Work, then blend
            # Note: per_source_chains_dfs from nonwork shared data are unfiltered,
            # so they serve as both transition source (after purpose filtering)
            # and length distribution source.
            per_source_chains_dfs = shared_data['per_source_chains_dfs']
            chain_models = {}
            for name, cdf in per_source_chains_dfs.items():
                purpose_cdf = cdf[
                    cdf['pattern'].str.contains(self.purpose, regex=False)
                    & ~cdf['pattern'].str.contains(BaseSurveyTrip.ACT_WORK, regex=False)
                ].copy()
                if not purpose_cdf.empty:
                    chain_models[name] = TripChainModel(
                        purpose_cdf, home_boost_factor=home_boost,
                        length_distribution_df=cdf,
                        early_stop_exponent=early_stop_exp)
            if len(chain_models) > 1:
                self.chain_model = BlendedTripChainModel(chain_models, blend_weights)
            elif chain_models:
                self.chain_model = next(iter(chain_models.values()))
            else:
                purpose_chains = chains_df[
                    chains_df['pattern'].str.contains(self.purpose, regex=False)
                    & ~chains_df['pattern'].str.contains(BaseSurveyTrip.ACT_WORK, regex=False)
                ].copy()
                self.chain_model = TripChainModel(purpose_chains, home_boost_factor=home_boost,
                                                  length_distribution_df=chains_df,
                                                  early_stop_exponent=early_stop_exp)

            # Time models
            per_source_time = {name: TripDurationModel(df, config=config) for name, df in per_source_data.items()}
            if len(per_source_time) > 1:
                self.trip_duration_model = BlendedTripDurationModel(per_source_time, blend_weights)
            else:
                self.trip_duration_model = next(iter(per_source_time.values()))

            per_source_act = {
                name: ActivityDurationModel(p, bw_method=bw_method, config=config)
                for name, p in per_source_persons.items()
            }
            if len(per_source_act) > 1:
                self.activity_duration_model = BlendedActivityDurationModel(per_source_act, blend_weights)
            else:
                self.activity_duration_model = next(iter(per_source_act.values()))
        else:
            # Single-source path
            self.trip_duration_model = TripDurationModel(survey_df, config=config)
            self.activity_duration_model = ActivityDurationModel(
                persons, bw_method=bw_method, config=config
            )
            all_chains_df = shared_data.get('all_chains_df', chains_df)
            purpose_chains = chains_df[
                chains_df['pattern'].str.contains(self.purpose, regex=False)
                & ~chains_df['pattern'].str.contains(BaseSurveyTrip.ACT_WORK, regex=False)
            ].copy()
            self.chain_model = TripChainModel(purpose_chains, home_boost_factor=home_boost,
                                              length_distribution_df=all_chains_df,
                                              early_stop_exponent=early_stop_exp)

        # Build spatial index for fast POI lookups
        self.poi_spatial_index = POISpatialIndex(self.poi_data_grouped)

        # Initialize POI weighting
        self.poi_weighting = POIWeighting(config, purpose)

        # Rebuild GTFS availability indices from serialized stop data
        gtfs_avail_manager = self._rebuild_gtfs_indices(shared_data.get('gtfs_stop_data'))

        # Initialize mode choice model with survey data
        if 'per_source_data' in shared_data:
            self.mode_choice = ModeChoiceModel(
                config,
                survey_data=shared_data['per_source_data'],
                survey_weights=shared_data['blend_weights'],
                gtfs_avail_manager=gtfs_avail_manager,
            )
        else:
            self.mode_choice = ModeChoiceModel(
                config,
                survey_data={'default': survey_df},
                survey_weights={'default': 1.0},
                gtfs_avail_manager=gtfs_avail_manager,
            )

        # Stats (local to this worker)
        self.stats = {
            'total_plans': 0,
            'failed_plans': 0,
            'chain_retries': 0,
            'chain_retries_too_short': 0,
            'chain_retries_bad_structure': 0,
            'chain_retries_missing_purpose': 0,
            'chain_retries_has_work': 0,
            'chain_attempts': 0,
            'poi_retries': 0,
            'time_retries': 0
        }

    def _rebuild_gtfs_indices(self, gtfs_stop_data: Optional[Dict]) -> Optional[GTFSAvailabilityManager]:
        """
        Rebuild GTFS R-tree indices from serialized stop coordinates.

        Workers can't receive unpicklable STRtree objects, so we rebuild
        from raw coordinate data.

        Args:
            gtfs_stop_data: Dict mapping mode_name -> list of [lon, lat] pairs,
                           or None if no GTFS data

        Returns:
            GTFSAvailabilityManager with rebuilt indices, or None
        """
        if not gtfs_stop_data:
            return None

        import numpy as np
        from shapely.geometry import Point
        from shapely.strtree import STRtree
        from models.mode_types import ModeType

        avail_manager = GTFSAvailabilityManager()

        for mode_name, coords_list in gtfs_stop_data.items():
            try:
                mode_type = ModeType(mode_name)
            except ValueError:
                continue

            coords = np.array(coords_list)
            points = [Point(c[0], c[1]) for c in coords]
            tree = STRtree(points)

            avail_manager._indices[mode_type] = tree
            avail_manager._stop_points[mode_type] = coords
            avail_manager._stop_counts[mode_type] = len(points)

        logger.debug(f"Worker rebuilt GTFS indices: "
                    f"{', '.join(f'{k.value}={v} stops' for k, v in avail_manager._stop_counts.items())}")
        return avail_manager

    def process_od_pair(self, origin_bg: str, dest_bg: str, num_trips: int) -> List[Plan]:
        """
        Process one OD pair and generate plans.

        Args:
            origin_bg: Origin block group ID
            dest_bg: Destination block group ID
            num_trips: Number of trips to generate for this OD pair

        Returns:
            List of Plan objects
        """
        plans = []

        for _ in range(num_trips):
            # Sample home location
            home_loc = self._sample_home_location(origin_bg)
            if home_loc is None:
                self.stats['failed_plans'] += 1
                continue

            # Sample destination POI
            main_activity = self.purpose
            dest_poi = self._sample_poi_location(dest_bg, main_activity)
            if dest_poi is None:
                self.stats['failed_plans'] += 1
                self.stats['poi_retries'] += 1
                continue

            dest_loc = (dest_poi['lon'], dest_poi['lat'])

            # Generate plan
            plan = self._generate_single_plan(home_loc, dest_loc, main_activity)
            if plan is None:
                self.stats['failed_plans'] += 1
                continue

            plan.person_id = ""  # Will be assigned by orchestrator
            plans.append(plan)
            self.stats['total_plans'] += 1

        return plans

    def _sample_home_location(self, origin_bg: str) -> Optional[Tuple[float, float]]:
        """Sample a specific home block within the origin block group."""
        # Get all 15-digit blocks within this block group
        blocks_in_bg = {
            geoid: data for geoid, data in self.home_locs_dict.items()
            if geoid.startswith(origin_bg)
        }

        if not blocks_in_bg:
            return None

        # Weight by non_employees
        geoids = list(blocks_in_bg.keys())
        weights = np.array([blocks_in_bg[gid]['non_employees'] for gid in geoids])

        if weights.sum() == 0:
            return None

        # Sample block weighted by non_employees
        probs = weights / weights.sum()
        sampled_geoid = np.random.choice(geoids, p=probs)

        # Get coordinates and add jitter
        block_data = blocks_in_bg[sampled_geoid]
        lon, lat = block_data['lon'], block_data['lat']

        # Add small jitter (±0.001 degrees ≈ ±100m)
        lon += np.random.uniform(-0.001, 0.001)
        lat += np.random.uniform(-0.001, 0.001)

        return (lon, lat)

    def _sample_poi_location(self, dest_bg: str, activity_type: str) -> Optional[Dict]:
        """Sample a specific POI within the destination block group."""
        # Get all 15-digit blocks within this block group
        blocks_in_bg = [
            geoid for geoid in self.home_locs_dict.keys()
            if geoid.startswith(dest_bg)
        ]

        if not blocks_in_bg:
            return None

        # Find POIs in these blocks matching the activity type
        candidate_pois = []
        for block_geoid in blocks_in_bg:
            block_data = self.home_locs_dict[block_geoid]
            block_lat, block_lon = block_data['lat'], block_data['lon']

            # Use expanding radius search
            initial_radius = self.config.get('poi_assignment', {}).get('initial_radius_m', 1000)
            radius_increment = self.config.get('poi_assignment', {}).get('radius_increment_m', 500)
            max_retries = self.config.get('poi_assignment', {}).get('max_poi_retries', 3)

            for retry in range(max_retries):
                search_radius = initial_radius + (retry * radius_increment)

                # Find POIs within radius
                nearby_pois = self.poi_spatial_index.find_within_radius(
                    block_lat, block_lon, search_radius
                )

                # Filter by activity type
                matching_pois = [
                    poi for poi in nearby_pois
                    if poi.get('activity') == self.purpose
                ]

                if matching_pois:
                    candidate_pois.extend(matching_pois)
                    break

        if not candidate_pois:
            return None

        # Remove duplicates (same POI found from multiple blocks)
        seen = set()
        unique_pois = []
        for poi in candidate_pois:
            osm_id = poi['osm_id']
            if osm_id not in seen:
                seen.add(osm_id)
                unique_pois.append(poi)

        # Calculate weights using POI weighting
        if self.poi_weighting.is_enabled():
            weights = np.array([self.poi_weighting.calculate_weight(poi) for poi in unique_pois])
        else:
            weights = np.ones(len(unique_pois))

        # Sample POI weighted by importance
        probs = weights / weights.sum()
        sampled_poi = np.random.choice(unique_pois, p=probs)

        return sampled_poi

    def _generate_single_plan(self, home_loc: Tuple[float, float],
                             dest_loc: Tuple[float, float],
                             main_activity: str,
                             max_retries: int = 10) -> Optional[Plan]:
        """Generate one complete plan with home-purpose-home structure."""
        for retry in range(max_retries):
            try:
                # Sample chain containing the main activity
                chain_str = self._sample_valid_chain(main_activity)
                if not chain_str:
                    continue

                # Parse activities
                activity_types = [a.strip() for a in chain_str.split('-')]

                # Assign locations
                activity_objs = self._assign_locations(
                    activity_types, home_loc, dest_loc, main_activity
                )
                if not activity_objs:
                    continue

                # Assign times
                success = self._assign_times(activity_objs)
                if not success:
                    self.stats['time_retries'] += 1
                    continue

                # Create legs with mode choice
                locations = [Location(lat=act.y, lon=act.x) for act in activity_objs]
                legs = self.mode_choice.create_legs(activity_objs, locations)

                # Create plan
                plan = Plan(
                    person_id="",
                    activities=activity_objs,
                    legs=legs
                )

                return plan

            except Exception as e:
                if self.verbose:
                    logger.debug(f"Plan generation error (retry {retry+1}/{max_retries}): {e}")
                continue

        return None

    def _sample_valid_chain(self, main_activity: str) -> Optional[str]:
        """Sample chain containing the main activity."""
        max_retries = self.config.get('plan_generation', {}).get('max_chain_retries', 100)
        method = self.config.get('plan_generation', {}).get('chain_sampling_method', 'generated')
        max_length = self.config.get('chains', {}).get('max_length', None)
        min_length = self.config.get('chains', {}).get('min_length', 3)

        for attempt in range(max_retries):
            self.stats['chain_attempts'] += 1
            chain = self.chain_model.sample(
                method=method, max_length=max_length, min_length=min_length
            )

            activities = [a.strip() for a in chain.split('-')]

            if len(activities) < 3:
                self.stats['chain_retries'] += 1
                self.stats['chain_retries_too_short'] += 1
                continue

            # Check if the chain contains the target purpose activity
            if self.purpose not in activities:
                self.stats['chain_retries'] += 1
                self.stats['chain_retries_missing_purpose'] += 1
                continue

            # Skip chains that contain Work
            if BaseSurveyTrip.ACT_WORK in activities:
                self.stats['chain_retries'] += 1
                self.stats['chain_retries_has_work'] += 1
                continue

            return chain

        return None

    def _assign_locations(self, activity_types: List[str],
                         home_loc: Tuple[float, float],
                         dest_loc: Tuple[float, float],
                         main_activity: str) -> Optional[List[Activity]]:
        """Assign locations to all activities in chain."""
        activities = []
        current_location = home_loc
        purpose_activity_assigned = False

        for i, act_type in enumerate(activity_types):
            activity = Activity(type=act_type, x=0, y=0)

            if act_type == BaseSurveyTrip.ACT_HOME:
                activity.x, activity.y = home_loc

            elif act_type == self.purpose and not purpose_activity_assigned:
                activity.x, activity.y = dest_loc
                purpose_activity_assigned = True

            else:
                # Find nearby POI for other activities
                poi = self._assign_poi_nearby(current_location, act_type)
                if poi is None:
                    return None
                activity.x, activity.y = poi['lon'], poi['lat']

            activities.append(activity)
            current_location = (activity.x, activity.y)

        return activities

    def _assign_poi_nearby(self, current_location: Tuple[float, float],
                          activity_type: str) -> Optional[Dict]:
        """Find nearby POI for intermediate activities."""
        lat, lon = current_location[1], current_location[0]

        initial_radius = self.config.get('poi_assignment', {}).get('initial_radius_m', 1000)
        radius_increment = self.config.get('poi_assignment', {}).get('radius_increment_m', 500)
        max_retries = self.config.get('poi_assignment', {}).get('max_poi_retries', 3)

        for retry in range(max_retries):
            search_radius = initial_radius + (retry * radius_increment)

            nearby_pois = self.poi_spatial_index.find_within_radius(lat, lon, search_radius)

            matching_pois = [
                poi for poi in nearby_pois
                if poi.get('activity') == activity_type
            ]

            if matching_pois:
                return np.random.choice(matching_pois)

            self.stats['poi_retries'] += 1

        return None

    def _trim_activity_durations(self, activity_types: List[str],
                                  durations: List[float],
                                  excess: float) -> List[float]:
        """
        Trim activity durations using priority-based approach.
        Discretionary activities (higher trim_priority from config) are trimmed
        before mandatory activities (lower trim_priority).

        Within each priority tier, the cut is distributed proportionally to
        each activity's trimmable headroom (current - min_minutes).

        After trimming, uniform jitter is applied to prevent all compressed
        plans from landing on the exact same departure minute.
        """
        act_constraints = self.config.get('duration_constraints', {}).get('activity_durations', {})

        # Build per-activity metadata from config
        items = []
        for i, (atype, dur) in enumerate(zip(activity_types, durations)):
            cfg = act_constraints.get(atype, {})
            priority = cfg.get('trim_priority', 2)   # default middle priority
            min_dur = cfg.get('min_minutes', 5)
            items.append((i, priority, min_dur, dur))

        trimmed = list(durations)
        remaining_excess = excess

        # Process tiers from highest priority (trim first) to lowest
        for target_priority in sorted(set(p for _, p, _, _ in items), reverse=True):
            if remaining_excess <= 0:
                break

            tier_indices = [i for i, p, _, _ in items if p == target_priority]
            tier_trimmable = sum(max(0, trimmed[i] - items[i][2]) for i in tier_indices)

            if tier_trimmable <= 0:
                continue

            tier_cut = min(remaining_excess, tier_trimmable)
            for idx in tier_indices:
                _, _, min_dur, _ = items[idx]
                available = trimmed[idx] - min_dur
                if available <= 0:
                    continue
                share = available / tier_trimmable
                cut = tier_cut * share
                trimmed[idx] = max(min_dur, trimmed[idx] - cut)

            remaining_excess -= tier_cut

        # Add jitter (±jitter_minutes uniform) to prevent clustering
        jitter_minutes = self.config.get('duration_constraints', {}).get('trim_jitter_minutes', 15)
        if jitter_minutes > 0:
            for i in range(len(trimmed)):
                min_dur = items[i][2]
                jitter = np.random.uniform(-jitter_minutes, jitter_minutes)
                trimmed[i] = max(min_dur, trimmed[i] + jitter)

        return trimmed

    def _assign_times(self, activities: List[Activity]) -> bool:
        """Assign times using max_dur approach for realistic activity scheduling."""
        max_retries = self.config.get('time_models', {}).get('max_time_retries', 10)

        for retry in range(max_retries):
            try:
                if len(activities) < 2:
                    return False

                # Get departure time for leaving first activity
                first_depart_min, _ = self.trip_duration_model.sample_dep_arr_time(
                    activities[0].type, activities[1].type, n_samples=1
                )
                first_depart_min = first_depart_min[0]

                if first_depart_min > 1440 or first_depart_min < 0:
                    raise ValueError("First departure time out of range")

                # Add small jitter to departure time to prevent clustering
                depart_jitter = np.random.uniform(-5, 5)
                first_depart_min = np.clip(first_depart_min + depart_jitter, 0, 1440)

                # Set end_time for first activity
                activities[0].end_time = self._minutes_to_timestr(first_depart_min)
                activities[0].start_time = None
                activities[0].max_dur = None

                # Collect durations for intermediate activities
                # Track running clock to estimate arrival time at each activity
                running_clock_min = first_depart_min
                activity_durations = []
                for i in range(1, len(activities) - 1):
                    act = activities[i]

                    # Estimate arrival time at this activity
                    travel_to_act = self.trip_duration_model.mean_trip_duration(
                        activities[i - 1].type, act.type
                    )
                    arrival_min = running_clock_min + travel_to_act

                    act_duration_min = self.activity_duration_model.sample_duration(
                        act.type, n_samples=1,
                        arrival_hour=arrival_min / 60.0,
                    )[0]

                    if act_duration_min <= 0 or act_duration_min > 720:
                        raise ValueError(f"Activity duration out of range: {act_duration_min}")

                    activity_durations.append(act_duration_min)
                    running_clock_min = arrival_min + act_duration_min

                # Validate total schedule fits in 24 hours
                home_morning_duration = first_depart_min
                middle_activities_duration = sum(activity_durations)

                # Per-leg travel time estimates from survey
                estimated_travel_time = 0
                for leg_i in range(len(activities) - 1):
                    origin_type = activities[leg_i].type
                    dest_type = activities[leg_i + 1].type
                    estimated_travel_time += self.trip_duration_model.mean_trip_duration(origin_type, dest_type)

                max_travel_buffer = self.config.get('duration_constraints', {}).get('max_travel_buffer_minutes', 180)
                estimated_travel_time = min(estimated_travel_time, max_travel_buffer)

                min_evening_home = self.config.get('duration_constraints', {}).get('min_evening_home_minutes', 60)

                total_time_used = home_morning_duration + middle_activities_duration + estimated_travel_time + min_evening_home

                # If exceeds 24 hours, trim activities using priority-based approach
                if total_time_used > 1440:
                    excess = total_time_used - 1440

                    if excess >= middle_activities_duration:
                        raise ValueError("Insufficient time budget for activities")

                    intermediate_types = [act.type for act in activities[1:-1]]
                    activity_durations = self._trim_activity_durations(
                        intermediate_types, activity_durations, excess
                    )

                # Assign durations to activities
                for i, act in enumerate(activities[1:-1], start=0):
                    act.max_dur = self._minutes_to_timestr(activity_durations[i])
                    act.start_time = None
                    act.end_time = None

                # Last activity: no times
                activities[-1].start_time = None
                activities[-1].end_time = None
                activities[-1].max_dur = None

                return True

            except Exception as e:
                continue

        return False

    def _minutes_to_timestr(self, minutes: float) -> str:
        """Convert minutes since midnight to HH:MM:SS format."""
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        secs = int((minutes % 1) * 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"


class NonWorkPlanGenerator:
    """
    Generates non-work trip plans (Shopping, Recreation, etc.) for MATSim simulation.

    Workflow:
    1. Load OD matrix for specific purpose (Shopping, etc.)
    2. Sample origin/destination block groups from OD matrix
    3. Sample specific home block within origin BG (weighted by non_employees)
    4. Sample specific POI in destination BG (weighted by POI importance)
    5. Generate activity chain with sampled purpose as main activity
    6. Assign times using survey-based time models
    7. Generate MATSim XML plans
    """

    def __init__(self, config: Dict, purpose: str = 'Shopping', shared_data: Optional[Dict] = None):
        """
        Initialize Non-Work Plan Generator.

        Args:
            config: Configuration dictionary
            purpose: Trip purpose ('Shopping', 'Recreation', etc.)
            shared_data: Optional pre-loaded shared data dict with keys:
                - home_locs_dict: Home locations filtered by county
                - poi_data_flat: Flat list of POIs (filtered by bounds)
                - poi_data_grouped: POIs grouped by activity type
                - poi_block_mapping: Pre-computed POI osm_id to block_id mapping
                - survey_df: Survey DataFrame
                - persons: Processed persons list
                - chains_df: Pre-processed chains DataFrame
                - trip_duration_model: Pre-initialized TripDurationModel (optional)
                - activity_duration_model: Pre-initialized ActivityDurationModel (optional)
                - poi_spatial_index: Pre-built POISpatialIndex (optional)
                If None, data will be loaded from scratch.
        """
        self.config = config
        self.purpose = purpose
        self.verbose = config.get('logging', {}).get('verbose', False)
        self._shared_data = shared_data  # Store reference to shared data

        # Get purpose-specific configuration
        self.purpose_config = config.get('nonwork_purposes', {}).get(purpose, {})
        if not self.purpose_config.get('enabled', False):
            raise ValueError(f"Purpose '{purpose}' is not enabled in config")

        logger.info(f"Initialized NonWorkPlanGenerator for {purpose}")

        # Initialize coordinate converter
        utm_epsg = config['coordinates']['utm_epsg']
        self.coord_converter = CoordinateConverter(utm_epsg=utm_epsg)
        logger.info(f"  Coordinate system: WGS84 → {utm_epsg}")

        # Load or use shared data
        if shared_data is not None:
            logger.info("Using pre-loaded shared data...")
            self._use_shared_data(shared_data)
        else:
            logger.info("Loading data sources...")
            self._load_data()

        # Initialize time models (use shared if available)
        logger.info("Initializing time models...")
        self._initialize_time_models()

        # Initialize POI components (use shared if available)
        logger.info("Initializing POI components...")
        self._initialize_poi_components()

        # Initialize chain model (purpose-specific filtering)
        logger.info("Initializing chain model...")
        self._initialize_chain_model()

        # Initialize GTFS availability (Phase 3)
        logger.info("Initializing GTFS transit availability...")
        self.gtfs_avail_manager = self._init_gtfs_availability()

        # Initialize mode choice model
        logger.info("Initializing mode choice model...")
        self._initialize_mode_choice()

        # Create OD matrix (uses shared POI-to-block mapping if available)
        logger.info(f"Creating {purpose} OD matrix...")
        self._create_od_matrix()

        # Statistics
        self.stats = {
            'total_plans': 0,
            'failed_plans': 0,
            'chain_retries': 0,
            'chain_retries_too_short': 0,
            'chain_retries_bad_structure': 0,
            'chain_retries_missing_purpose': 0,
            'chain_retries_has_work': 0,
            'chain_attempts': 0,
            'poi_retries': 0,
            'time_retries': 0
        }

        logger.info("NonWorkPlanGenerator initialization complete")

    def _load_data(self):
        """Load home locations, POI data, and survey data."""
        # Load home locations (with non_employees) - filtered by counties in config
        self.home_locs_dict = load_home_locations_by_counties(self.config)
        logger.info(f"  Loaded {len(self.home_locs_dict):,} home blocks (county-filtered)")

        # Load POI data from database
        data_dir = self.config['data']['data_dir']
        db_manager = initialize_tables(data_dir)

        from sqlalchemy import text
        with db_manager.session_scope() as session:
            query = text("SELECT osm_id, name, activity, lat, lon, tags FROM pois")
            result = session.execute(query)
            rows = result.fetchall()

            # First, collect all POIs into a temporary list
            all_pois = []
            for row in rows:
                poi_dict = {
                    'osm_id': row[0],
                    'name': row[1],
                    'activity': row[2],
                    'lat': row[3],
                    'lon': row[4],
                    'tags': row[5]
                }
                all_pois.append(poi_dict)

            logger.info(f"  Loaded {len(all_pois):,} POIs from database (all regions)")

        # Filter POIs to only those within the county bounding box
        # This prevents POIs from outside selected counties being assigned to blocks within
        filtered_pois = _filter_pois_by_bounds(all_pois, self.home_locs_dict, buffer_km=5.0)

        # Group filtered POIs by activity type (required by POISpatialIndex)
        self.poi_data_grouped = {}
        self.poi_data_flat = []  # Keep flat list for OD matrix creation

        for poi_dict in filtered_pois:
            # Add to flat list
            self.poi_data_flat.append(poi_dict)

            # Group by activity for spatial index
            activity = poi_dict['activity']
            if activity not in self.poi_data_grouped:
                self.poi_data_grouped[activity] = []
            self.poi_data_grouped[activity].append(poi_dict)

        logger.info(f"  Final POI count: {len(self.poi_data_flat):,} POIs in {len(self.poi_data_grouped)} activity types")

        # Load survey data via SurveyManager
        survey_manager = SurveyManager(self.config)
        self.survey_df = survey_manager.get_survey_df()
        self.persons = survey_manager.get_persons()
        logger.info(f"  Loaded {len(self.survey_df):,} survey trips")
        logger.info(f"  Processed {len(self.persons):,} persons")

        # Process chains (needed for _initialize_chain_model)
        use_weight = self.config.get('chains', {}).get('use_weighted_chains', True)
        chains = process_trip_chains(self.persons, use_weight=use_weight)
        self.chains_df = pd.DataFrame(chains)
        logger.info(f"  Processed {len(self.chains_df):,} trip chains")

    def _use_shared_data(self, shared_data: Dict):
        """
        Use pre-loaded shared data instead of loading from scratch.

        Args:
            shared_data: Dictionary with pre-loaded data
        """
        self.home_locs_dict = shared_data['home_locs_dict']
        self.poi_data_flat = shared_data['poi_data_flat']
        self.poi_data_grouped = shared_data['poi_data_grouped']
        self.survey_df = shared_data['survey_df']
        self.persons = shared_data['persons']
        self.chains_df = shared_data['chains_df']

        logger.info(f"  Using {len(self.home_locs_dict):,} home blocks (pre-loaded)")
        logger.info(f"  Using {len(self.poi_data_flat):,} POIs (pre-loaded)")
        logger.info(f"  Using {len(self.survey_df):,} survey trips (pre-loaded)")
        logger.info(f"  Using {len(self.persons):,} persons (pre-loaded)")
        logger.info(f"  Using {len(self.chains_df):,} trip chains (pre-loaded)")

        # Log availability of optional shared components
        if 'poi_block_mapping' in shared_data:
            logger.info(f"  Using {len(shared_data['poi_block_mapping']):,} POI-to-block mappings (pre-computed)")
        if 'trip_duration_model' in shared_data:
            logger.info(f"  Using shared trip duration model (pre-initialized)")
        if 'activity_duration_model' in shared_data:
            logger.info(f"  Using shared activity duration model (pre-initialized)")
        if 'poi_spatial_index' in shared_data:
            logger.info(f"  Using shared POI spatial index (pre-built)")

    def _init_gtfs_availability(self) -> Optional[GTFSAvailabilityManager]:
        """
        Initialize GTFS data pipeline and build spatial indices.

        Reuses GTFS stop data from shared_data if available (avoids re-downloading),
        otherwise runs the full GTFS setup.

        Returns:
            GTFSAvailabilityManager with built indices, or None if no GTFS modes configured
        """
        modes_config = self.config.get('modes', {})

        # Check if any enabled mode uses GTFS availability (skip _comment/_help string entries)
        has_gtfs = any(
            isinstance(cfg, dict) and cfg.get('enabled', True)
            and isinstance(cfg.get('availability', ''), dict) and cfg['availability'].get('type') == 'gtfs'
            for cfg in modes_config.values()
        )
        if not has_gtfs:
            logger.info("  No GTFS-based modes configured, skipping GTFS setup")
            return None

        # Check if GTFS stop data is available from shared data
        if self._shared_data is not None and 'gtfs_stop_data' in self._shared_data:
            gtfs_stop_data = self._shared_data['gtfs_stop_data']
            if gtfs_stop_data:
                return self._rebuild_gtfs_from_shared(gtfs_stop_data)

        # Full GTFS setup (standalone mode or feeds not yet loaded)
        from data_sources.gtfs_manager import GTFSManager
        from models.models import initialize_tables

        db_manager = None
        try:
            data_dir = self.config['data']['data_dir']
            db_manager = initialize_tables(data_dir)

            gtfs_manager = GTFSManager(self.config, db_manager)

            # Only run full setup if feeds are not already in the DB
            if not gtfs_manager.has_feeds_loaded():
                logger.info("  GTFS feeds not yet in DB, running setup...")
                gtfs_manager.setup()
            else:
                logger.info("  GTFS feeds already in DB, skipping redundant setup")

            avail_manager = GTFSAvailabilityManager()
            avail_manager.build_indices(gtfs_manager, modes_config)

            stats = avail_manager.get_stats()
            logger.info(f"  GTFS availability ready: {stats['modes_indexed']} modes, "
                       f"{stats['total_stops']} stops indexed")
            return avail_manager

        except Exception as e:
            logger.error(f"GTFS initialization failed: {e}")
            logger.warning("Transit availability will fall back to universal (always available)")
            return None
        finally:
            if db_manager is not None:
                db_manager.close()

    def _rebuild_gtfs_from_shared(self, gtfs_stop_data: Dict) -> Optional[GTFSAvailabilityManager]:
        """Rebuild GTFS indices from pre-serialized stop data."""
        from shapely.geometry import Point
        from shapely.strtree import STRtree
        from models.mode_types import ModeType

        avail_manager = GTFSAvailabilityManager()

        for mode_name, coords_list in gtfs_stop_data.items():
            try:
                mode_type = ModeType(mode_name)
            except ValueError:
                continue

            coords = np.array(coords_list)
            points = [Point(c[0], c[1]) for c in coords]
            tree = STRtree(points)

            avail_manager._indices[mode_type] = tree
            avail_manager._stop_points[mode_type] = coords
            avail_manager._stop_counts[mode_type] = len(points)

        logger.info(f"  GTFS availability rebuilt from shared data: "
                   f"{', '.join(f'{k.value}={v} stops' for k, v in avail_manager._stop_counts.items())}")
        return avail_manager

    def _initialize_mode_choice(self):
        """Initialize mode choice model with survey data for mode rate computation."""
        if (self._shared_data is not None
                and 'per_source_data' in self._shared_data):
            # Multi-source: pass per-source survey data and weights
            self.mode_choice = ModeChoiceModel(
                self.config,
                survey_data=self._shared_data['per_source_data'],
                survey_weights=self._shared_data['blend_weights'],
                gtfs_avail_manager=self.gtfs_avail_manager,
            )
        else:
            # Single source: wrap survey_df in dict
            survey_name = 'default'
            for entry in self.config.get('data', {}).get('surveys', []):
                if entry.get('weight', 0) > 0:
                    survey_name = entry.get('type', 'default')
                    break
            self.mode_choice = ModeChoiceModel(
                self.config,
                survey_data={survey_name: self.survey_df},
                survey_weights={survey_name: 1.0},
                gtfs_avail_manager=self.gtfs_avail_manager,
            )

    def _initialize_time_models(self):
        """Initialize departure time and activity duration models.

        Uses pre-initialized models from shared_data if available,
        otherwise creates new models from survey data.  When per-source
        data is available (multi-source mode), builds blended wrappers.
        """
        bw_method = self.config.get('time_models', {}).get('kde_bandwidth', 'scott')

        # Check if shared models are available (pre-built, possibly blended)
        if self._shared_data is not None and 'trip_duration_model' in self._shared_data:
            self.trip_duration_model = self._shared_data['trip_duration_model']
            logger.info("  Using shared trip duration model")
        elif (self._shared_data is not None
              and 'per_source_data' in self._shared_data):
            # Multi-source: build blended model from per-source data
            per_source_data = self._shared_data['per_source_data']
            blend_weights = self._shared_data['blend_weights']
            per_source_time = {name: TripDurationModel(df, config=self.config) for name, df in per_source_data.items()}
            if len(per_source_time) > 1:
                self.trip_duration_model = BlendedTripDurationModel(per_source_time, blend_weights)
                logger.info("  Created blended trip duration model")
            else:
                self.trip_duration_model = next(iter(per_source_time.values()))
                logger.info("  Created single-source trip duration model")
        else:
            self.trip_duration_model = TripDurationModel(self.survey_df, config=self.config)
            logger.info("  Created new trip duration model")

        if self._shared_data is not None and 'activity_duration_model' in self._shared_data:
            self.activity_duration_model = self._shared_data['activity_duration_model']
            logger.info("  Using shared activity duration model")
        elif (self._shared_data is not None
              and 'per_source_persons' in self._shared_data):
            # Multi-source: build blended model from per-source persons
            per_source_persons = self._shared_data['per_source_persons']
            blend_weights = self._shared_data['blend_weights']
            per_source_act = {
                name: ActivityDurationModel(p, bw_method=bw_method, config=self.config)
                for name, p in per_source_persons.items()
            }
            if len(per_source_act) > 1:
                self.activity_duration_model = BlendedActivityDurationModel(per_source_act, blend_weights)
                logger.info("  Created blended activity duration model")
            else:
                self.activity_duration_model = next(iter(per_source_act.values()))
                logger.info("  Created single-source activity duration model")
        else:
            self.activity_duration_model = ActivityDurationModel(
                self.persons, bw_method=bw_method, config=self.config
            )
            logger.info("  Created new activity duration model")

        # Calculate average trip duration from survey for time budget estimation
        self.avg_trip_duration_min = self._calculate_avg_trip_duration()

        logger.info("  Time models initialized")
        logger.info(f"  Average trip duration: {self.avg_trip_duration_min:.1f} minutes")

    def _calculate_avg_trip_duration(self) -> float:
        """
        Calculate average trip duration from survey data.

        Returns:
            Average trip duration in minutes
        """
        # Get trip duration constraints from config
        trip_constraints = self.config.get('duration_constraints', {}).get('trip_durations', {}).get('default', {})
        min_duration = trip_constraints.get('min_minutes', 1)
        max_duration = trip_constraints.get('max_minutes', 180)

        # Convert duration from seconds to minutes and filter
        durations_min = (self.survey_df['duration_seconds'] / 60.0).dropna()
        durations_min = durations_min[(durations_min >= min_duration) & (durations_min <= max_duration)]

        if len(durations_min) == 0:
            logger.warning("No valid trip durations found in survey, using default 20 minutes")
            return 20.0

        avg_duration = durations_min.mean()
        return avg_duration

    def _initialize_poi_components(self):
        """Initialize POI spatial index and weighting.

        Uses pre-built spatial index from shared_data if available,
        otherwise creates a new one.
        """
        # Check if shared POI spatial index is available
        if self._shared_data is not None and 'poi_spatial_index' in self._shared_data:
            self.poi_spatial_index = self._shared_data['poi_spatial_index']
            stats = self.poi_spatial_index.get_stats()
            logger.info(f"  Using shared POI spatial index: {stats['num_activities']} activity types, {stats['total_pois']:,} POIs")
        else:
            # POI Spatial Index for efficient nearest-neighbor search (needs grouped data)
            self.poi_spatial_index = POISpatialIndex(self.poi_data_grouped)
            stats = self.poi_spatial_index.get_stats()
            logger.info(f"  POI spatial index built: {stats['num_activities']} activity types, {stats['total_pois']:,} POIs")

        # POI Weighting for importance-based sampling (purpose-specific, always created)
        self.poi_weighting = POIWeighting(self.config, self.purpose)
        logger.info(f"  POI weighting enabled: {self.poi_weighting.is_enabled()}")

    def _initialize_chain_model(self):
        """Initialize trip chain model with purpose-specific filtering.

        When per-source data is available (multi-source mode), builds a
        BlendedTripChainModel from per-source purpose-filtered chains.

        The chain length distribution is always learned from the full
        unfiltered survey so the Markov generator targets realistic lengths.
        """
        home_boost = self.config.get('chains', {}).get('home_boost_factor', 2.0)
        early_stop_exp = self.config.get('chains', {}).get('early_stop_exponent', 2.0)

        if (self._shared_data is not None
                and 'per_source_chains_dfs' in self._shared_data):
            # Multi-source: per-source purpose-filtered chain models
            # Note: per_source_chains_dfs from nonwork shared data are unfiltered
            per_source_chains_dfs = self._shared_data['per_source_chains_dfs']
            blend_weights = self._shared_data['blend_weights']

            chain_models = {}
            for name, cdf in per_source_chains_dfs.items():
                purpose_cdf = cdf[
                    cdf['pattern'].str.contains(self.purpose, regex=False)
                    & ~cdf['pattern'].str.contains(BaseSurveyTrip.ACT_WORK, regex=False)
                ].copy()
                if not purpose_cdf.empty:
                    chain_models[name] = TripChainModel(
                        purpose_cdf, home_boost_factor=home_boost,
                        length_distribution_df=cdf,
                        early_stop_exponent=early_stop_exp)
                    logger.info(f"  Chain model for '{name}': {len(purpose_cdf)} {self.purpose} patterns "
                                f"(excl. Work, length dist from {len(cdf)} unfiltered)")

            if len(chain_models) > 1:
                self.chain_model = BlendedTripChainModel(chain_models, blend_weights)
                logger.info(f"  Blended chain model with {len(chain_models)} sources")
            elif chain_models:
                self.chain_model = next(iter(chain_models.values()))
                logger.info(f"  Single-source chain model")
            else:
                # Fallback: use merged chains_df
                purpose_chains = self.chains_df[
                    self.chains_df['pattern'].str.contains(self.purpose, regex=False)
                    & ~self.chains_df['pattern'].str.contains(BaseSurveyTrip.ACT_WORK, regex=False)
                ].copy()
                self.chain_model = TripChainModel(purpose_chains, home_boost_factor=home_boost,
                                                  length_distribution_df=self.chains_df,
                                                  early_stop_exponent=early_stop_exp)
                logger.info(f"  Fallback chain model with {len(purpose_chains)} patterns (excl. Work)")
        else:
            # Single-source path
            purpose_chains = self.chains_df[
                self.chains_df['pattern'].str.contains(self.purpose, regex=False)
                & ~self.chains_df['pattern'].str.contains(BaseSurveyTrip.ACT_WORK, regex=False)
            ].copy()

            logger.info(f"  Filtered {len(purpose_chains):,} chains containing {self.purpose}")

            self.chain_model = TripChainModel(
                purpose_chains,
                home_boost_factor=home_boost,
                length_distribution_df=self.chains_df,
                early_stop_exponent=early_stop_exp
            )

            logger.info(f"  Chain model initialized with {len(purpose_chains):,} chains "
                        f"(length dist from {len(self.chains_df)} unfiltered)")

    def _create_od_matrix(self):
        """Create OD matrix for the purpose and apply scaling factor.

        Uses pre-computed POI-to-block mapping from shared_data if available,
        which significantly speeds up the POI density calculation.

        When multi-source data is available, computes a blended survey
        trip rate from location-capable surveys and overrides the
        ``survey_rate`` in the purpose config so
        ``create_nonwork_od_matrix`` uses it.
        """
        # Get pre-computed POI-to-block mapping if available
        poi_block_mapping = None
        if self._shared_data is not None and 'poi_block_mapping' in self._shared_data:
            poi_block_mapping = self._shared_data['poi_block_mapping']
            logger.info(f"  Using pre-computed POI-to-block mapping")

        # For multi-source: override survey_rate with blended rate (6e)
        effective_config = self.config
        if (self._shared_data is not None
                and 'per_source_data' in self._shared_data):
            per_source_data = self._shared_data['per_source_data']
            blend_weights = self._shared_data['blend_weights']

            blended_rate = calculate_blended_survey_trip_rate(
                per_source_data, blend_weights, self.purpose, self.config
            )
            if blended_rate > 0:
                import copy
                effective_config = copy.deepcopy(self.config)
                purpose_cfg = effective_config.get('nonwork_purposes', {}).get(self.purpose, {})
                trip_gen = purpose_cfg.get('trip_generation', {})
                trip_gen['survey_rate'] = blended_rate
                logger.info(
                    f"  Overriding survey_rate for {self.purpose} with "
                    f"blended rate: {blended_rate:.4f}"
                )

        self.od_matrix = create_nonwork_od_matrix(
            config=effective_config,
            home_locs_dict=self.home_locs_dict,
            poi_data=self.poi_data_flat,  # Use flat list for OD matrix creation
            survey_df=self.survey_df,
            purpose=self.purpose,
            poi_block_mapping=poi_block_mapping
        )

        # Store unscaled total for logging
        self.unscaled_total_trips = self.od_matrix.sum().sum()

        # Get scaling factor from config
        self.scaling_factor = self.config.get('plan_generation', {}).get('scaling_factor', 1.0)

        logger.info(f"  OD matrix shape: {self.od_matrix.shape}")
        logger.info(f"  Unscaled total trips: {self.unscaled_total_trips:,.0f}")
        logger.info(f"  Scaling factor: {self.scaling_factor}")

        # Apply scaling factor using probabilistic rounding (same method as work trips)
        if self.scaling_factor < 1.0:
            self._apply_scaling_to_od_matrix()

        self.scaled_total_trips = self.od_matrix.sum().sum()
        logger.info(f"  Scaled total trips: {self.scaled_total_trips:,.0f}")

    def _apply_scaling_to_od_matrix(self):
        """
        Apply scaling factor to OD matrix using probabilistic rounding.

        This uses the same method as the work trip generator to ensure
        consistent scaling across work and non-work trips.

        The probabilistic rounding preserves the expected total while
        maintaining the spatial distribution of trips.
        """
        # Set random seed for reproducibility
        random_seed = self.config.get('plan_generation', {}).get('random_seed', 42)
        np.random.seed(random_seed + hash(self.purpose) % 1000)  # Different seed per purpose

        # Vectorized probabilistic rounding (replaces slow cell-by-cell .loc loop)
        values = self.od_matrix.values
        scaled = values * self.scaling_factor
        floor_values = np.floor(scaled).astype(int)
        fractional_parts = scaled - floor_values

        # Probabilistically round up based on fractional part
        random_draws = np.random.random(scaled.shape)
        rounded = floor_values + (random_draws < fractional_parts).astype(int)

        # Only apply to cells that were originally > 0
        mask = values > 0
        result = np.where(mask, rounded, 0)
        self.od_matrix = pd.DataFrame(result, index=self.od_matrix.index, columns=self.od_matrix.columns)

    def sample_origin_destination_pairs(self, n_samples: int) -> List[Tuple[str, str, int]]:
        """
        Sample origin-destination block group pairs from OD matrix.

        Note: The OD matrix has already been scaled by scaling_factor in _create_od_matrix().
        When n_samples is "all", we use all trips from the already-scaled OD matrix.

        Args:
            n_samples: Number of trips to sample (or "all" to use full scaled OD matrix)

        Returns:
            List of (origin_bg, dest_bg, num_trips) tuples
        """
        # Use Hamilton's method for proportional allocation
        samples = generate_samples_from_od_matrix(self.od_matrix, n_samples)

        # Calculate actual total trips from samples
        total_trips = sum(count for _, _, count in samples)

        # Format n_samples for logging
        if isinstance(n_samples, str) and n_samples.lower() == "all":
            logger.info(f"  Using all {total_trips:,} trips from scaled OD matrix")
        else:
            logger.info(f"  Sampled {total_trips:,} trips from {len(samples)} OD pairs")

        return samples

    def sample_home_location(self, origin_bg: str) -> Optional[Tuple[float, float]]:
        """
        Sample a specific home block within the origin block group.

        Args:
            origin_bg: 12-digit block group ID

        Returns:
            (lon, lat) tuple or None if no valid blocks found
        """
        # Get all 15-digit blocks within this block group
        blocks_in_bg = {
            geoid: data for geoid, data in self.home_locs_dict.items()
            if geoid.startswith(origin_bg)
        }

        if not blocks_in_bg:
            return None

        # Weight by non_employees
        geoids = list(blocks_in_bg.keys())
        weights = np.array([blocks_in_bg[gid]['non_employees'] for gid in geoids])

        if weights.sum() == 0:
            return None

        # Sample block weighted by non_employees
        probs = weights / weights.sum()
        sampled_geoid = np.random.choice(geoids, p=probs)

        # Get coordinates and add jitter
        block_data = blocks_in_bg[sampled_geoid]
        lon, lat = block_data['lon'], block_data['lat']

        # Add small jitter (±0.001 degrees ≈ ±100m)
        lon += np.random.uniform(-0.001, 0.001)
        lat += np.random.uniform(-0.001, 0.001)

        return (lon, lat)

    def sample_poi_location(self, dest_bg: str, activity_type: str) -> Optional[Dict]:
        """
        Sample a specific POI within the destination block group.

        Uses POI importance weighting for realistic destination choice.

        Args:
            dest_bg: 12-digit block group ID
            activity_type: Activity type (e.g., 'Shopping')

        Returns:
            POI dictionary with keys: osm_id, name, activity, lat, lon, tags
        """
        # Get all 15-digit blocks within this block group
        blocks_in_bg = [
            geoid for geoid in self.home_locs_dict.keys()
            if geoid.startswith(dest_bg)
        ]

        if not blocks_in_bg:
            return None

        # Find POIs in these blocks matching the activity type
        candidate_pois = []
        for block_geoid in blocks_in_bg:
            block_data = self.home_locs_dict[block_geoid]
            block_lat, block_lon = block_data['lat'], block_data['lon']

            # Use expanding radius search
            initial_radius = self.config.get('poi_assignment', {}).get('initial_radius_m', 1000)
            radius_increment = self.config.get('poi_assignment', {}).get('radius_increment_m', 500)
            max_retries = self.config.get('poi_assignment', {}).get('max_poi_retries', 3)

            for retry in range(max_retries):
                search_radius = initial_radius + (retry * radius_increment)

                # Find POIs within radius
                nearby_pois = self.poi_spatial_index.find_within_radius(
                    block_lat, block_lon, search_radius
                )

                # Filter by activity type
                matching_pois = [
                    poi for poi in nearby_pois
                    if poi.get('activity') == self.purpose
                ]

                if matching_pois:
                    candidate_pois.extend(matching_pois)
                    break

        if not candidate_pois:
            return None

        # Remove duplicates (same POI found from multiple blocks)
        seen = set()
        unique_pois = []
        for poi in candidate_pois:
            osm_id = poi['osm_id']
            if osm_id not in seen:
                seen.add(osm_id)
                unique_pois.append(poi)

        # Calculate weights using POI weighting
        if self.poi_weighting.is_enabled():
            weights = np.array([self.poi_weighting.calculate_weight(poi) for poi in unique_pois])
        else:
            weights = np.ones(len(unique_pois))

        # Sample POI weighted by importance
        probs = weights / weights.sum()
        sampled_poi = np.random.choice(unique_pois, p=probs)

        return sampled_poi

    def generate_plans_list(self, n_plans: int, log_file_path: Optional[str] = None) -> Tuple[List[Plan], Dict]:
        """
        Generate non-work plans using multiprocessing and return as list.

        This method always uses multiprocessing for efficiency.
        Used when combining work and non-work plans into a single XML file.

        Args:
            n_plans: Number of plans to generate (or "all" to use scaled OD matrix)
            log_file_path: Optional path to log file for worker processes

        Returns:
            Tuple of (List of Plan objects, stats dictionary with generation metrics)
        """
        logger.info("=" * 60)
        logger.info(f"{self.purpose.upper()} PLAN GENERATION")
        logger.info("=" * 60)

        # Log scaling information
        logger.info(f"  Population base (non-employees in region): {sum(d.get('non_employees', 0) for d in self.home_locs_dict.values()):,}")
        logger.info(f"  Unscaled {self.purpose} trips: {self.unscaled_total_trips:,.0f}")
        logger.info(f"  Scaling factor: {self.scaling_factor}")
        logger.info(f"  Scaled {self.purpose} trips: {self.scaled_total_trips:,.0f}")
        expected_scaled = self.unscaled_total_trips * self.scaling_factor
        logger.info(f"  Expected scaled trips: {expected_scaled:,.0f}")
        if expected_scaled > 0:
            accuracy = (self.scaled_total_trips / expected_scaled) * 100
            logger.info(f"  Scaling accuracy: {accuracy:.1f}%")
        logger.info("")

        # Handle case where n_plans might be a string like "all"
        if isinstance(n_plans, str):
            logger.info(f"Generating {n_plans} {self.purpose} plans using multiprocessing...")
        else:
            logger.info(f"Generating {n_plans:,} {self.purpose} plans using multiprocessing...")

        # Sample OD pairs (from already-scaled OD matrix)
        od_samples = self.sample_origin_destination_pairs(n_plans)

        # Always use multiprocessing
        num_processes = self.config.get('plan_generation', {}).get('num_processes', 4)
        # Ensure at least 1 process
        num_processes = max(1, num_processes)

        logger.info(f"Using {num_processes} processes for {self.purpose} plan generation")

        # Generate plans using parallel processing
        plans = self._generate_plans_parallel(od_samples, num_processes, log_file_path)

        # Log statistics
        logger.info("")
        logger.info(f"{self.purpose} GENERATION COMPLETE")
        logger.info(f"  Plans generated: {self.stats['total_plans']:,}")
        logger.info(f"  Failed plans: {self.stats['failed_plans']:,}")
        logger.info(f"  Chain retries: {self.stats['chain_retries']:,} / {self.stats['chain_attempts']:,} attempts")
        if self.stats['chain_retries'] > 0:
            logger.info(f"    Too short: {self.stats['chain_retries_too_short']:,}")
            logger.info(f"    Bad structure (not Home-...-Home): {self.stats['chain_retries_bad_structure']:,}")
            logger.info(f"    Missing purpose: {self.stats['chain_retries_missing_purpose']:,}")
            logger.info(f"    Contains Work: {self.stats['chain_retries_has_work']:,}")
        logger.info(f"  POI retries: {self.stats['poi_retries']:,}")
        logger.info(f"  Time retries: {self.stats['time_retries']:,}")
        logger.info("=" * 60)

        # Log mode choice statistics
        self.mode_choice.log_stats_summary()

        # Calculate success rate
        total_requested = self.stats['total_plans'] + self.stats['failed_plans']
        success_rate = (self.stats['total_plans'] / total_requested * 100) if total_requested > 0 else 100.0

        # Build stats dict with all metrics
        generation_stats = {
            'purpose': self.purpose,
            'plans_requested': total_requested,
            'plans_generated': self.stats['total_plans'],
            'failed_plans': self.stats['failed_plans'],
            'success_rate': round(success_rate, 2),
            'chain_retries': self.stats['chain_retries'],
            'chain_retries_too_short': self.stats['chain_retries_too_short'],
            'chain_retries_bad_structure': self.stats['chain_retries_bad_structure'],
            'chain_retries_missing_purpose': self.stats['chain_retries_missing_purpose'],
            'chain_retries_has_work': self.stats['chain_retries_has_work'],
            'chain_attempts': self.stats['chain_attempts'],
            'poi_retries': self.stats['poi_retries'],
            'time_retries': self.stats['time_retries'],
            'unscaled_trips': int(self.unscaled_total_trips),  # Total trips before scaling
            'mode_choice': self.mode_choice.get_stats_summary(),  # Mode choice statistics
        }

        return plans, generation_stats

    def _serialize_gtfs_stop_data(self) -> Optional[Dict]:
        """
        Serialize GTFS stop coordinates per mode for passing to workers.

        STRtree objects are not picklable, so we pass raw coordinates
        and let workers rebuild the spatial indices.

        Returns:
            Dict mapping mode_name -> list of [lon, lat] pairs, or None
        """
        if self.gtfs_avail_manager is None:
            return None

        stop_data = {}
        for mode_type, points_array in self.gtfs_avail_manager._stop_points.items():
            if len(points_array) > 0:
                stop_data[mode_type.value] = points_array.tolist()

        if stop_data:
            logger.info(f"  Serialized GTFS stop data for workers: "
                       f"{', '.join(f'{k}={len(v)} stops' for k, v in stop_data.items())}")
        return stop_data if stop_data else None

    def _prepare_shared_data(self) -> Dict:
        """
        Prepare shared data for multiprocessing workers.

        Returns:
            Dictionary with all data needed by worker processes
        """
        # Get pre-processed chains DataFrame
        chains_df = self._get_processed_chains_df()

        shared = {
            'home_locs_dict': self.home_locs_dict,
            'poi_data_grouped': self.poi_data_grouped,
            'survey_df': self.survey_df,
            'persons': self.persons,
            'chains_df': chains_df,
            'all_chains_df': self.chains_df,  # Unfiltered chains for length distribution
            'avg_trip_duration_min': self.avg_trip_duration_min,
            'gtfs_stop_data': self._serialize_gtfs_stop_data(),
        }

        # Propagate per-source data for multi-source blending in workers
        if self._shared_data is not None and 'per_source_data' in self._shared_data:
            shared['per_source_data'] = self._shared_data['per_source_data']
            shared['per_source_persons'] = self._shared_data['per_source_persons']
            shared['per_source_chains_dfs'] = self._shared_data['per_source_chains_dfs']
            shared['blend_weights'] = self._shared_data['blend_weights']

        return shared

    def _get_processed_chains_df(self) -> pd.DataFrame:
        """
        Get pre-processed chains DataFrame for sharing with workers.

        Returns:
            DataFrame with chain patterns
        """
        return self.chain_model.chains_df

    def _aggregate_worker_stats(self, worker_stats: Dict) -> None:
        """Aggregate worker plan generation stats into main stats."""
        for key in ('total_plans', 'failed_plans', 'chain_retries',
                     'chain_retries_too_short', 'chain_retries_bad_structure',
                     'chain_retries_missing_purpose', 'chain_retries_has_work',
                     'chain_attempts', 'poi_retries', 'time_retries'):
            self.stats[key] += worker_stats.get(key, 0)

    def _aggregate_mode_choice_stats(self, worker_mode_stats: Optional[Dict]) -> None:
        """
        Aggregate mode choice statistics from a worker into the parent's ModeChoiceModel.

        Args:
            worker_mode_stats: Dict from worker's ModeChoiceModel.get_stats_summary()
        """
        if not worker_mode_stats:
            return

        from models.mode_types import ModeType

        # Aggregate mode sample counts
        worker_distribution = worker_mode_stats.get('mode_distribution', {})
        worker_total = worker_mode_stats.get('total_mode_choices', 0)

        for mode_str, share in worker_distribution.items():
            try:
                mode_type = ModeType(mode_str)
                count = int(round(share * worker_total))
                self.mode_choice.stats['mode_samples'][mode_type] += count
            except (ValueError, KeyError):
                pass

        # Aggregate other stats
        self.mode_choice.stats['fallback_used'] += worker_mode_stats.get('fallback_used', 0)
        self.mode_choice.stats['chain_retries'] += worker_mode_stats.get('chain_retries', 0)

        for purpose in worker_mode_stats.get('purposes_using_fallback_rates', []):
            self.mode_choice.stats['purposes_with_fallback'].add(purpose)

    def _generate_plans_parallel(self, od_pairs: List[Tuple[str, str, int]],
                                  num_processes: int,
                                  log_file_path: Optional[str] = None) -> List[Plan]:
        """
        Generate plans in parallel using multiprocessing.

        Args:
            od_pairs: List of (origin_bg, dest_bg, num_trips) tuples
            num_processes: Number of parallel processes to use
            log_file_path: Optional path to log file for worker processes

        Returns:
            List of all generated plans
        """
        logger.info(f"Pre-loading data for {self.purpose} multiprocessing...")

        # Prepare shared data for workers
        shared_data = self._prepare_shared_data()
        logger.info(f"  Prepared shared data with {len(shared_data['home_locs_dict']):,} home blocks")
        logger.info(f"  POI categories: {len(shared_data['poi_data_grouped'])}")
        logger.info(f"  Chain patterns: {len(shared_data['chains_df']):,}")

        # Split OD pairs into chunks for each process
        chunk_size = max(1, len(od_pairs) // num_processes)
        od_chunks = [od_pairs[i:i + chunk_size] for i in range(0, len(od_pairs), chunk_size)]

        logger.info(f"Split {len(od_pairs)} OD pairs into {len(od_chunks)} chunks")

        # Prepare arguments for each worker
        # Include purpose so workers can be reused for different trip types
        worker_args = [
            (chunk, self.config, shared_data, log_file_path, self.purpose)
            for chunk in od_chunks
        ]

        all_plans = []

        try:
            # Use multiprocessing pool
            with mp.Pool(processes=num_processes) as pool:
                show_progress = self.config.get('logging', {}).get('show_progress_bar', True)

                if show_progress:
                    # Use imap_unordered for progress tracking
                    pbar = tqdm(total=len(od_chunks), desc=f"Processing {self.purpose} chunks")
                    for chunk_plans, worker_stats in pool.imap_unordered(
                        _worker_process_chunk_nonwork, worker_args
                    ):
                        all_plans.extend(chunk_plans)
                        self._aggregate_worker_stats(worker_stats)
                        self._aggregate_mode_choice_stats(worker_stats.get('mode_choice'))
                        pbar.update(1)
                        pbar.set_postfix({
                            'plans': len(all_plans),
                            'failed': self.stats['failed_plans']
                        })
                    pbar.close()
                else:
                    # Process without progress bar
                    results = pool.map(_worker_process_chunk_nonwork, worker_args)
                    for chunk_plans, worker_stats in results:
                        all_plans.extend(chunk_plans)
                        self._aggregate_worker_stats(worker_stats)
                        self._aggregate_mode_choice_stats(worker_stats.get('mode_choice'))

        except Exception as e:
            logger.error(f"Multiprocessing failed for {self.purpose}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to generate {self.purpose} plans: {e}")

        return all_plans

    def generate_plans(self, n_plans: int, output_path: str):
        """
        Generate non-work plans and save to MATSim XML file.

        This method is for standalone non-work plan generation.
        For combined work+non-work plans, use generate_plans_list() instead.

        Args:
            n_plans: Number of plans to generate
            output_path: Path to save plans.xml
        """
        # Generate plans
        plans = self.generate_plans_list(n_plans)

        # Assign person IDs
        for i, plan in enumerate(plans):
            plan.person_id = f"nonwork_{self.purpose.lower()}_{i}"

        # Write to XML
        logger.info(f"Writing {len(plans):,} plans to {output_path}...")
        self._write_plans_xml(plans, output_path)

        # Log statistics
        logger.info("=" * 70)
        logger.info(f"{self.purpose} PLAN GENERATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total plans generated: {self.stats['total_plans']:,}")
        logger.info(f"Failed plans: {self.stats['failed_plans']:,}")
        logger.info(f"Chain retries: {self.stats['chain_retries']:,} / {self.stats['chain_attempts']:,} attempts")
        if self.stats['chain_retries'] > 0:
            logger.info(f"  Too short: {self.stats['chain_retries_too_short']:,}")
            logger.info(f"  Bad structure (not Home-...-Home): {self.stats['chain_retries_bad_structure']:,}")
            logger.info(f"  Missing purpose: {self.stats['chain_retries_missing_purpose']:,}")
            logger.info(f"  Contains Work: {self.stats['chain_retries_has_work']:,}")
        logger.info(f"POI retries: {self.stats['poi_retries']:,}")
        logger.info(f"Time retries: {self.stats['time_retries']:,}")
        logger.info("=" * 70)

    def _generate_single_plan(self, home_loc: Tuple[float, float],
                             dest_loc: Tuple[float, float],
                             main_activity: str,
                             max_retries: int = 10) -> Optional[Plan]:
        """
        Generate one complete plan with home-purpose-home structure.

        Args:
            home_loc: (lon, lat) for home
            dest_loc: (lon, lat) for main destination
            main_activity: Main activity type (e.g., 'Shopping')
            max_retries: Maximum retries

        Returns:
            Plan object or None if generation fails
        """
        for retry in range(max_retries):
            try:
                # Sample chain containing the main activity
                chain_str = self._sample_valid_chain(main_activity)
                if not chain_str:
                    if self.verbose:
                        logger.debug(f"Failed to sample valid chain for {main_activity} (retry {retry+1}/{max_retries})")
                    continue

                if self.verbose:
                    logger.debug(f"Sampled chain: {chain_str}")

                # Parse activities
                activity_types = [a.strip() for a in chain_str.split('-')]

                # Assign locations
                activity_objs = self._assign_locations(
                    activity_types, home_loc, dest_loc, main_activity
                )
                if not activity_objs:
                    if self.verbose:
                        logger.debug(f"Failed to assign locations (retry {retry+1}/{max_retries})")
                    continue

                # Assign times
                success = self._assign_times(activity_objs)
                if not success:
                    if self.verbose:
                        logger.debug(f"Failed to assign times (retry {retry+1}/{max_retries})")
                    self.stats['time_retries'] += 1
                    continue

                # Create legs with mode choice
                locations = [Location(lat=act.y, lon=act.x) for act in activity_objs]
                legs = self.mode_choice.create_legs(activity_objs, locations)

                # Create plan
                plan = Plan(
                    person_id="",  # Assigned later
                    activities=activity_objs,
                    legs=legs
                )

                if self.verbose:
                    logger.debug(f"Successfully generated plan with {len(activity_objs)} activities")

                return plan

            except Exception as e:
                if self.verbose:
                    logger.debug(f"Plan generation error (retry {retry+1}/{max_retries}): {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                continue

        if self.verbose:
            logger.debug(f"Failed to generate plan after {max_retries} retries")
        return None

    def _sample_valid_chain(self, main_activity: str) -> Optional[str]:
        """
        Sample chain containing the main activity.

        Valid chains must:
        1. Start with Home
        2. Contain the main activity (e.g., 'Shopping')
        3. End with Home

        Args:
            main_activity: Main activity to include (e.g., 'Shopping')

        Returns:
            Chain string or None
        """
        max_retries = self.config.get('plan_generation', {}).get('max_chain_retries', 100)
        method = self.config.get('plan_generation', {}).get('chain_sampling_method', 'generated')
        max_length = self.config.get('chains', {}).get('max_length', None)
        min_length = self.config.get('chains', {}).get('min_length', 3)

        for attempt in range(max_retries):
            self.stats['chain_attempts'] += 1
            chain = self.chain_model.sample(
                method=method, max_length=max_length, min_length=min_length
            )

            activities = [a.strip() for a in chain.split('-')]

            if len(activities) < 3:
                self.stats['chain_retries'] += 1
                self.stats['chain_retries_too_short'] += 1
                continue

            # Check if the chain contains the target purpose activity
            if self.purpose not in activities:
                self.stats['chain_retries'] += 1
                self.stats['chain_retries_missing_purpose'] += 1
                continue

            # For non-work plans, SKIP chains that contain Work
            # Work locations are handled differently (not as POIs)
            if BaseSurveyTrip.ACT_WORK in activities:
                self.stats['chain_retries'] += 1
                self.stats['chain_retries_has_work'] += 1
                continue

            return chain

        return None

    def _assign_locations(self, activity_types: List[str],
                         home_loc: Tuple[float, float],
                         dest_loc: Tuple[float, float],
                         main_activity: str) -> Optional[List[Activity]]:
        """
        Assign locations to all activities in chain.

        Logic:
        - Home → home_loc
        - Purpose activity → dest_loc (sampled POI)
        - Other activities → find nearby POI based on current location

        Args:
            activity_types: List of activity type strings
            home_loc: (lon, lat) for home
            dest_loc: (lon, lat) for main destination POI
            main_activity: Main activity type

        Returns:
            List of Activity objects or None if assignment fails
        """
        activities = []
        current_location = home_loc
        purpose_activity_assigned = False

        for i, act_type in enumerate(activity_types):
            activity = Activity(type=act_type, x=0, y=0)

            # Assign location
            if act_type == BaseSurveyTrip.ACT_HOME:
                activity.x, activity.y = home_loc

            elif act_type == self.purpose and not purpose_activity_assigned:
                activity.x, activity.y = dest_loc
                purpose_activity_assigned = True

            else:
                # Find nearby POI for other activities
                poi = self._assign_poi_nearby(current_location, act_type)
                if poi is None:
                    if self.verbose:
                        logger.warning(f"Failed to find POI for '{act_type}' near {current_location}")
                    return None

                activity.x, activity.y = poi['lon'], poi['lat']

            activities.append(activity)
            current_location = (activity.x, activity.y)

        return activities

    def _assign_poi_nearby(self, current_location: Tuple[float, float],
                          activity_type: str) -> Optional[Dict]:
        """
        Find nearby POI for intermediate activities.

        Args:
            current_location: (lon, lat)
            activity_type: Activity type string

        Returns:
            POI dictionary or None
        """
        lat, lon = current_location[1], current_location[0]

        initial_radius = self.config.get('poi_assignment', {}).get('initial_radius_m', 1000)
        radius_increment = self.config.get('poi_assignment', {}).get('radius_increment_m', 500)
        max_retries = self.config.get('poi_assignment', {}).get('max_poi_retries', 3)

        for retry in range(max_retries):
            search_radius = initial_radius + (retry * radius_increment)

            # Find POIs within radius
            nearby_pois = self.poi_spatial_index.find_within_radius(lat, lon, search_radius)

            # Filter by activity type
            matching_pois = [
                poi for poi in nearby_pois
                if poi.get('activity') == activity_type
            ]

            if matching_pois:
                # Sample randomly (or could use weighting here too)
                return np.random.choice(matching_pois)

            self.stats['poi_retries'] += 1

        return None

    def _trim_activity_durations(self, activity_types: List[str],
                                  durations: List[float],
                                  excess: float) -> List[float]:
        """
        Trim activity durations using priority-based approach.
        Discretionary activities (higher trim_priority from config) are trimmed
        before mandatory activities (lower trim_priority).

        Within each priority tier, the cut is distributed proportionally to
        each activity's trimmable headroom (current - min_minutes).

        After trimming, uniform jitter is applied to prevent all compressed
        plans from landing on the exact same departure minute.
        """
        act_constraints = self.config.get('duration_constraints', {}).get('activity_durations', {})

        # Build per-activity metadata from config
        items = []
        for i, (atype, dur) in enumerate(zip(activity_types, durations)):
            cfg = act_constraints.get(atype, {})
            priority = cfg.get('trim_priority', 2)   # default middle priority
            min_dur = cfg.get('min_minutes', 5)
            items.append((i, priority, min_dur, dur))

        trimmed = list(durations)
        remaining_excess = excess

        # Process tiers from highest priority (trim first) to lowest
        for target_priority in sorted(set(p for _, p, _, _ in items), reverse=True):
            if remaining_excess <= 0:
                break

            tier_indices = [i for i, p, _, _ in items if p == target_priority]
            tier_trimmable = sum(max(0, trimmed[i] - items[i][2]) for i in tier_indices)

            if tier_trimmable <= 0:
                continue

            tier_cut = min(remaining_excess, tier_trimmable)
            for idx in tier_indices:
                _, _, min_dur, _ = items[idx]
                available = trimmed[idx] - min_dur
                if available <= 0:
                    continue
                share = available / tier_trimmable
                cut = tier_cut * share
                trimmed[idx] = max(min_dur, trimmed[idx] - cut)

            remaining_excess -= tier_cut

        # Add jitter (±jitter_minutes uniform) to prevent clustering
        jitter_minutes = self.config.get('duration_constraints', {}).get('trim_jitter_minutes', 15)
        if jitter_minutes > 0:
            for i in range(len(trimmed)):
                min_dur = items[i][2]
                jitter = np.random.uniform(-jitter_minutes, jitter_minutes)
                trimmed[i] = max(min_dur, trimmed[i] + jitter)

        return trimmed

    def _assign_times(self, activities: List[Activity]) -> bool:
        """
        Assign times using max_dur approach for realistic activity scheduling.
        MATSim will calculate actual travel times based on network routing.

        Plan structure:
        - First activity: only end_time (when to leave home - sampled from survey)
        - Intermediate activities: only max_dur (how long to stay - sampled from survey)
        - Last activity: no times (agent stays until end of day)

        Returns:
            True if successful, False otherwise
        """
        max_retries = self.config.get('time_models', {}).get('max_time_retries', 10)

        for retry in range(max_retries):
            try:
                if len(activities) < 2:
                    return False

                # Get departure time for leaving first activity
                first_depart_min, _ = self.trip_duration_model.sample_dep_arr_time(
                    activities[0].type, activities[1].type, n_samples=1
                )
                first_depart_min = first_depart_min[0]

                if first_depart_min > 1440 or first_depart_min < 0:
                    raise ValueError("First departure time out of range")

                # Add small jitter to departure time to prevent clustering
                depart_jitter = np.random.uniform(-5, 5)
                first_depart_min = np.clip(first_depart_min + depart_jitter, 0, 1440)

                # Set end_time for first activity (when agent leaves home/first location)
                activities[0].end_time = self._minutes_to_timestr(first_depart_min)
                activities[0].start_time = None
                activities[0].max_dur = None

                # For intermediate activities (not first, not last): set max_dur
                # Track running clock to estimate arrival time at each activity
                running_clock_min = first_depart_min
                activity_durations = []
                for i in range(1, len(activities) - 1):
                    act = activities[i]

                    # Estimate arrival time at this activity
                    travel_to_act = self.trip_duration_model.mean_trip_duration(
                        activities[i - 1].type, act.type
                    )
                    arrival_min = running_clock_min + travel_to_act

                    act_duration_min = self.activity_duration_model.sample_duration(
                        act.type, n_samples=1,
                        arrival_hour=arrival_min / 60.0,
                    )[0]

                    if act_duration_min <= 0 or act_duration_min > 720:
                        raise ValueError(f"Activity duration out of range: {act_duration_min}")

                    activity_durations.append(act_duration_min)
                    running_clock_min = arrival_min + act_duration_min
                    logger.debug(f"Sampled duration for {act.type}: {act_duration_min:.1f} min")

                # Validate total schedule fits in 24 hours
                home_morning_duration = first_depart_min
                middle_activities_duration = sum(activity_durations)

                # Per-leg travel time estimates from survey
                estimated_travel_time = 0
                for leg_i in range(len(activities) - 1):
                    origin_type = activities[leg_i].type
                    dest_type = activities[leg_i + 1].type
                    estimated_travel_time += self.trip_duration_model.mean_trip_duration(origin_type, dest_type)

                max_travel_buffer = self.config.get('duration_constraints', {}).get('max_travel_buffer_minutes', 180)
                estimated_travel_time = min(estimated_travel_time, max_travel_buffer)

                min_evening_home = self.config.get('duration_constraints', {}).get('min_evening_home_minutes', 60)

                total_time_used = home_morning_duration + middle_activities_duration + estimated_travel_time + min_evening_home

                # If exceeds 24 hours, trim activities using priority-based approach
                if total_time_used > 1440:
                    excess = total_time_used - 1440

                    if excess >= middle_activities_duration:
                        logger.debug(f"Insufficient time budget: home_morning={home_morning_duration:.1f}, "
                                   f"travel={estimated_travel_time:.1f}, evening_home={min_evening_home:.1f}")
                        raise ValueError("Insufficient time budget for activities")

                    intermediate_types = [act.type for act in activities[1:-1]]
                    activity_durations = self._trim_activity_durations(
                        intermediate_types, activity_durations, excess
                    )

                    logger.debug(f"Trimmed activity durations (priority-based) to fit 24-hour constraint")
                    logger.debug(f"  Time budget: home_morning={home_morning_duration:.1f}min, "
                               f"activities={sum(activity_durations):.1f}min, "
                               f"travel={estimated_travel_time:.1f}min, "
                               f"evening_home={min_evening_home:.1f}min, "
                               f"excess_trimmed={excess:.1f}min")

                # Assign durations to activities
                for i, act in enumerate(activities[1:-1], start=0):
                    act.max_dur = self._minutes_to_timestr(activity_durations[i])
                    act.start_time = None
                    act.end_time = None

                # Last activity: no times at all (agent stays until simulation end)
                activities[-1].start_time = None
                activities[-1].end_time = None
                activities[-1].max_dur = None

                return True

            except Exception as e:
                if self.verbose:
                    logger.debug(f"Time assignment retry {retry+1}: {e}")
                continue

        return False

    def _minutes_to_timestr(self, minutes: float) -> str:
        """Convert minutes since midnight to HH:MM:SS format."""
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        secs = int((minutes % 1) * 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    def _write_plans_xml(self, plans: List[Plan], output_path: str):
        """
        Write plans to MATSim XML file.

        Coordinates are converted from WGS84 to UTM for MATSim.

        Args:
            plans: List of Plan objects
            output_path: Path to save XML file
        """
        # Create XML structure
        root = ET.Element('population')

        for plan in plans:
            person_elem = ET.SubElement(root, 'person', id=plan.person_id)
            plan_elem = ET.SubElement(person_elem, 'plan', selected='yes')

            for i, act in enumerate(plan.activities):
                # Convert coordinates to UTM
                x_utm, y_utm = self.coord_converter.transform(act.x, act.y)

                act_attribs = {
                    'type': act.type,
                    'x': f'{x_utm:.2f}',
                    'y': f'{y_utm:.2f}'
                }

                if act.end_time:
                    act_attribs['end_time'] = act.end_time
                if act.max_dur:
                    act_attribs['max_dur'] = act.max_dur

                ET.SubElement(plan_elem, 'activity', **act_attribs)

                # Add leg (except after last activity)
                if i < len(plan.activities) - 1:
                    leg_mode = plan.legs[i].mode if i < len(plan.legs) else 'car'
                    ET.SubElement(plan_elem, 'leg', mode=leg_mode)

        # Format and write
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent='  ')

        # Remove empty lines
        xml_lines = [line for line in xml_str.split('\n') if line.strip()]
        xml_str = '\n'.join(xml_lines)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)

        logger.info(f"Plans written to {output_path}")


if __name__ == '__main__':
    # Example usage
    config = load_config()

    # Generate Shopping plans
    generator = NonWorkPlanGenerator(config, purpose='Shopping')
    generator.generate_plans(n_plans=1000, output_path='../output/plans_shopping.xml')
