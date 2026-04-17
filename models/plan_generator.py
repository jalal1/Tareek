"""
MATSim Plan Generator

Generates synthetic daily activity plans in MATSim XML format for the Twin Cities region.
Integrates OD matrix, activity chains, time models, and POI assignment.

Coordinate System Handling:
- Internal processing: All coordinates are stored and processed in EPSG:4326 (WGS84 lat/lon)
- XML output: Coordinates are automatically converted to NAD83 UTM (auto-detected from county centroids)
- MATSim requires projected coordinates (UTM) for accurate distance calculations
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

# Ensure project root is in sys.path for module imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import load_config
from data_sources.base_survey_trip import BaseSurveyTrip
from data_sources.survey_manager import SurveyManager
from models.home_locs_v2 import load_home_locations_by_counties
from models.work_locs_v2 import load_work_locations_by_counties
from models.chains import TripChainModel, BlendedTripChainModel, filter_chains_by_type, process_trip_chains
from models.time import TripDurationModel, ActivityDurationModel, BlendedTripDurationModel, BlendedActivityDurationModel
from models.poi_manager import POIManager
from models.od_matrix_v3 import (
    create_local_od_matrix,
    create_survey_od_matrix_using_trip_weight,
    combine_od_matrices,
    blend_survey_od_matrices,
    generate_samples
)
from models.models import initialize_tables
from models.mode_choice import ModeChoiceModel, Leg
from models.mode_availability import Location
from models.gtfs_availability import GTFSAvailabilityManager
from data_sources.gtfs_manager import GTFSManager
from utils.logger import setup_logger, create_experiment_dir, get_current_experiment_dir, reconfigure_logger_to_experiment_dir
from utils.poi_spatial_index import POISpatialIndex
from utils.coordinates import CoordinateConverter

logger = setup_logger(__name__)

@dataclass
class Activity:
    """Represents a single activity in a plan."""
    type: str
    x: float
    y: float
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    max_dur: Optional[str] = None  # Maximum duration for intermediate activities


@dataclass
class Plan:
    """Represents a complete daily plan for one person.

    For n activities, there are n-1 legs (trips between activities).
    Each leg has its own mode assignment.
    """
    person_id: str
    activities: List[Activity] = field(default_factory=list)
    legs: List[Leg] = field(default_factory=list)


def _worker_process_chunk(args: Tuple) -> Tuple[List[Plan], Dict]:
    """
    Worker function for multiprocessing. Processes a chunk of OD pairs.

    Re-initializes the PlanGenerator in each worker to avoid pickling issues
    with database connections and other unpicklable objects.

    Args:
        args: Tuple of (od_chunk, config_data, data_dir, shared_data, log_file_path)
            - od_chunk: List of (origin_bg, dest_bg, num_trips) tuples
            - config_data: Configuration dictionary
            - data_dir: Path to data directory
            - shared_data: Dict with pre-loaded data (home_locs, work_locs, od_matrix)
            - log_file_path: Path to the main log file

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
    od_chunk, config_data, data_dir, shared_data, log_file_path = args

    # CRITICAL: Re-initialize logger for this worker process
    # Multiprocessing workers don't inherit the parent's logger handlers
    worker_id = multiprocessing.current_process().name

    # Get logging settings from config
    log_cfg = config_data.get('logging', {})
    log_to_file = log_cfg.get('log_to_file', True)
    log_to_console = log_cfg.get('log_to_console', True)
    log_level = getattr(logging, log_cfg.get('level', 'INFO').upper(), logging.INFO)

    # Setup root logger for this worker
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

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
        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)

    # File handler (only if log_to_file is True and log_file_path is provided)
    if log_to_file and log_file_path:
        try:
            file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            root_logger.addHandler(file_handler)
        except Exception as e:
            # File logging failed - fall back to stderr since logger may not be set up yet
            import sys
            print(f"Worker {worker_id}: Failed to setup file logging: {e}", file=sys.stderr)

    # Get logger for this module
    worker_logger = logging.getLogger('models.plan_generator')
    worker_logger.info(f"Worker {worker_id} started - processing {len(od_chunk)} OD pairs")

    # Create a minimal generator instance for this worker
    # This avoids pickling database connections
    worker_gen = _WorkerPlanGenerator(config_data, data_dir, shared_data)

    chunk_plans = []
    for origin_bg, dest_bg, num_trips in od_chunk:
        plans = worker_gen.process_od_pair(origin_bg, dest_bg, num_trips)
        chunk_plans.extend(plans)

    worker_logger.info(f"Worker {worker_id} completed: generated {len(chunk_plans)} plans")

    # Return plans, stats, and mode choice stats from this worker
    worker_stats = worker_gen.stats
    worker_stats['mode_choice'] = worker_gen.mode_choice.get_stats_summary()
    return chunk_plans, worker_stats


class _BasePlanGenerator:
    """
    Base class with shared plan generation logic.

    Contains all common methods used by both PlanGenerator and _WorkerPlanGenerator.
    Subclasses should implement _assign_poi_with_retry() to customize POI assignment.
    """

    def _generate_single_plan(self, home_loc: Tuple[float, float],
                             work_loc: Tuple[float, float],
                             max_retries: int = 10) -> Optional[Plan]:
        """
        Generate one complete plan.

        Args:
            home_loc: (lon, lat) geographic coordinates
            work_loc: (lon, lat) geographic coordinates
            max_retries: Maximum retries for failed generation

        Returns:
            Plan object or None if generation fails
        """
        for retry in range(max_retries):
            try:
                # Sample valid chain
                chain_str = self._sample_valid_chain()
                if not chain_str:
                    continue

                # Parse activities
                activities_types = [a.strip() for a in chain_str.split('-')]

                # Assign locations
                activity_objs = self._assign_locations(activities_types, home_loc, work_loc)
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
                    person_id="",  # Assigned later
                    activities=activity_objs,
                    legs=legs
                )

                return plan

            except Exception as e:
                # Debug logging if available
                if hasattr(logger, 'debug'):
                    logger.debug(f"Plan generation error (retry {retry+1}/{max_retries}): {e}")
                continue

        return None

    def _sample_valid_chain(self) -> Optional[str]:
        """Sample chain with retry until valid home-work-home pattern found.

        Valid chains must:
        1. Match home-work-home pattern (start home, contain work, end home)
        2. Contain between 1 and max_work_activities Work activities
        """
        max_retries = self.config.get('plan_generation', {}).get('max_chain_retries', 100)
        method = self.config.get('plan_generation', {}).get('chain_sampling_method', 'generated')
        max_length = self.config.get('chains', {}).get('max_length', None)
        min_length = self.config.get('chains', {}).get('min_length', 3)
        max_work = self.config.get('chains', {}).get('max_work_activities', 2)

        for attempt in range(max_retries):
            self.stats['chain_attempts'] += 1
            chain = self.chain_model.sample(
                method=method, max_length=max_length, min_length=min_length
            )

            activities = [a.strip() for a in chain.split('-')]

            # Check minimum length
            if len(activities) < 3:
                self.stats['chain_retries'] += 1
                self.stats['chain_retries_too_short'] += 1
                continue

            # Check contains Work
            work_count = sum(1 for a in activities if a == BaseSurveyTrip.ACT_WORK)
            if work_count == 0:
                self.stats['chain_retries'] += 1
                self.stats['chain_retries_missing_work'] += 1
                continue

            # Check Work count within allowed range
            if work_count > max_work:
                self.stats['chain_retries'] += 1
                self.stats['chain_retries_too_many_work'] += 1
                continue

            return chain

        return None

    def _assign_locations(self, activity_types: List[str],
                         home_loc: Tuple[float, float],
                         work_loc: Tuple[float, float]) -> Optional[List[Activity]]:
        """Assign locations to all activities in chain."""
        activities = []
        current_location = home_loc
        # logger.info(f"_assign_locations called with {len(activity_types)} activities")
        # logger.info(f"  Activity types: {activity_types}")

        for i, act_type in enumerate(activity_types):
            activity = Activity(
                type=act_type,
                x=0, y=0  # Set below
                # start_time and end_time set later in _assign_times()
            )

            # Assign location
            if act_type == BaseSurveyTrip.ACT_HOME:
                # All Home activities use home location (including intermediate returns)
                activity.x, activity.y = home_loc

            elif act_type == BaseSurveyTrip.ACT_WORK:
                activity.x, activity.y = work_loc

            else:
                # Assign POI
                # logger.info(f"  Activity {i}: '{act_type}' - searching for POI near {current_location}")
                poi = self._assign_poi_with_retry(current_location, act_type)
                if poi is None:
                    if self.verbose:
                        logger.warning(f"  Failed to find POI for activity '{act_type}' near {current_location}")
                    return None

                # Use POI lat/lon directly (no conversion needed)
                activity.x, activity.y = poi['lon'], poi['lat']  # x=lon, y=lat
                # logger.info(f"  Found POI: {poi.get('name', 'unknown')} at (lon={poi['lon']}, lat={poi['lat']})")

            activities.append(activity)
            current_location = (activity.x, activity.y)

        return activities

    def _has_work_activity(self, activities: List[Activity]) -> bool:
        """
        Check if activity chain contains Work activity (excluding first and last).

        Args:
            activities: List of Activity objects

        Returns:
            True if any intermediate activity is 'Work'
        """
        return any(act.type == BaseSurveyTrip.ACT_WORK for act in activities[1:-1])

    def _calculate_avg_trip_duration(self, survey_df: pd.DataFrame) -> float:
        """
        Calculate average trip duration from survey data.

        Args:
            survey_df: Survey DataFrame with duration_seconds column

        Returns:
            Average trip duration in minutes
        """
        # Get trip duration constraints from config
        trip_constraints = self.config.get('duration_constraints', {}).get('trip_durations', {}).get('default', {})
        min_duration = trip_constraints.get('min_minutes', 1)
        max_duration = trip_constraints.get('max_minutes', 180)

        # Convert duration from seconds to minutes and filter
        durations_min = (survey_df['duration_seconds'] / 60.0).dropna()
        durations_min = durations_min[(durations_min >= min_duration) & (durations_min <= max_duration)]

        if len(durations_min) == 0:
            fallback = (min_duration + max_duration) / 2.0
            logger.warning(
                f"FALLBACK: No valid trip durations found in survey after filtering "
                f"[{min_duration}, {max_duration}] min. Using midpoint={fallback:.1f} min. "
                f"Check that the survey data contains valid 'duration_seconds' values."
            )
            return fallback

        avg_duration = durations_min.mean()
        logger.debug(f"Calculated average trip duration from survey: {avg_duration:.1f} minutes")
        return avg_duration

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

        Args:
            activity_types: Types of intermediate activities (excluding first/last Home)
            durations: Current durations in minutes for each intermediate activity
            excess: Total minutes to trim from the schedule

        Returns:
            Trimmed durations list
        """
        act_constraints = self.config.get('duration_constraints', {}).get('activity_durations', {})

        # Build per-activity metadata from config
        items = []
        for i, (atype, dur) in enumerate(zip(activity_types, durations)):
            cfg = act_constraints.get(atype, {})
            if not cfg:
                logger.warning(
                    f"FALLBACK: Activity '{atype}' not in duration_constraints config. "
                    f"Using default trim_priority=2, min_minutes=5 for schedule trimming."
                )
            priority = cfg.get('trim_priority', 2)
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

    def _assign_poi_with_retry(self, current_location: Tuple[float, float],
                               activity: str) -> Optional[Dict]:
        """
        Assign POI with expanding radius retry logic.

        This method should be overridden by subclasses to customize POI assignment.
        """
        raise NotImplementedError("Subclasses must implement _assign_poi_with_retry")

    def _assign_times(self, activities: List[Activity]) -> bool:
        """
        Assign times using max_dur approach for realistic activity scheduling.
        MATSim will calculate actual travel times based on network routing.

        Plan structure:
        - First activity: only end_time (when to leave home - sampled from survey)
        - Intermediate activities: only max_dur (how long to stay - sampled from survey)
        - Last activity: no times (agent stays until end of day)

        This approach:
        1. Preserves temporal patterns (when people leave home)
        2. Preserves activity duration patterns (how long at work, shopping, etc.)
        3. Lets MATSim calculate realistic travel times based on actual network distances
        4. Avoids incorrect assumptions about arrival times
        """
        max_retries = self.config.get('time_models', {}).get('max_time_retries', 10)

        for retry in range(max_retries):
            try:
                # Sample when to leave the first activity (typically home)
                # Use departure time from survey data for first trip
                if len(activities) < 2:
                    return False

                # Get departure time for leaving first activity
                # If chain contains Work, use Home→Work departure time (work commute)
                # Otherwise, use actual first trip departure time
                if self._has_work_activity(activities):
                    # This is a work commute - use Home→Work departure time
                    first_depart_min, _ = self.time_model.sample_dep_arr_time(
                        BaseSurveyTrip.ACT_HOME, BaseSurveyTrip.ACT_WORK, n_samples=1
                    )
                else:
                    # Not a work trip - use actual first trip
                    first_depart_min, _ = self.time_model.sample_dep_arr_time(
                        activities[0].type, activities[1].type, n_samples=1
                    )
                first_depart_min = first_depart_min[0]

                # Validate: ensure departure happens during the day
                if first_depart_min > 1440 or first_depart_min < 0:
                    logger.debug(
                        f"CONSTRAINT: Sampled departure time {first_depart_min:.1f} min is outside "
                        f"[0, 1440] range — retrying (KDE can produce out-of-range samples)"
                    )
                    raise ValueError("First departure time out of range")

                # Add small jitter to departure time to prevent clustering
                # at KDE peak minutes (configurable, clamped to [0, 1440])
                depart_jitter_minutes = self.config.get('duration_constraints', {}).get('departure_jitter_minutes', 5)
                depart_jitter = np.random.uniform(-depart_jitter_minutes, depart_jitter_minutes)
                first_depart_min = np.clip(first_depart_min + depart_jitter, 0, 1440)

                # Set end_time for first activity (when agent leaves home/first location)
                activities[0].end_time = self._minutes_to_timestr(first_depart_min)
                activities[0].start_time = None  # No start time needed
                activities[0].max_dur = None     # No max_dur for first activity

                # For intermediate activities (not first, not last): set max_dur
                # Track running clock to estimate arrival time at each activity
                running_clock_min = first_depart_min

                # First, collect all durations
                activity_durations = []
                for i in range(1, len(activities) - 1):
                    act = activities[i]

                    # Estimate arrival time at this activity (depart prev + travel)
                    travel_to_act = self.time_model.mean_trip_duration(
                        activities[i - 1].type, act.type
                    )
                    arrival_min = running_clock_min + travel_to_act

                    # Sample how long to stay at this activity using activity duration model
                    act_duration_min = self.activity_duration_model.sample_duration(
                        act.type, n_samples=1,
                        arrival_hour=arrival_min / 60.0,
                    )[0]

                    # Validate duration is reasonable using per-activity config bounds
                    act_constraints = self.config.get('duration_constraints', {}).get('activity_durations', {})
                    act_cfg = act_constraints.get(act.type, {})
                    act_max = act_cfg.get('max_minutes', 720)
                    act_min = act_cfg.get('min_minutes', 0)
                    if act_duration_min <= act_min or act_duration_min > act_max:
                        raise ValueError(
                            f"Activity duration {act_duration_min:.1f} min for '{act.type}' "
                            f"outside config range [{act_min}, {act_max}]"
                        )

                    activity_durations.append(act_duration_min)
                    # Advance running clock: arrival + activity duration
                    running_clock_min = arrival_min + act_duration_min
                    logger.debug(f"Sampled duration for {act.type}: {act_duration_min:.1f} min")

                # Validate total schedule fits in 24 hours with proper time budget calculation
                # Time budget components:
                # 1. Home morning duration: 00:00 to first departure
                home_morning_duration = first_depart_min

                # 2. Middle activities duration
                middle_activities_duration = sum(activity_durations)

                # 3. Estimated travel time: sum of per-leg mean durations from survey
                estimated_travel_time = 0
                for leg_i in range(len(activities) - 1):
                    origin_type = activities[leg_i].type
                    dest_type = activities[leg_i + 1].type
                    estimated_travel_time += self.time_model.mean_trip_duration(origin_type, dest_type)

                # Clip travel time to realistic range from config
                max_travel_buffer = self.config.get('duration_constraints', {}).get('max_travel_buffer_minutes', 180)
                estimated_travel_time = min(estimated_travel_time, max_travel_buffer)

                # 4. Minimum evening home duration (from last arrival to 24:00)
                min_evening_home = self.config.get('duration_constraints', {}).get('min_evening_home_minutes', 60)

                # Calculate total time used
                total_time_used = home_morning_duration + middle_activities_duration + estimated_travel_time + min_evening_home

                # If exceeds 24 hours, trim activities using priority-based approach
                # Discretionary activities (Shopping, Dining, etc.) are trimmed before
                # mandatory activities (Work, School) based on trim_priority in config
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

                # Assign scaled durations to activities
                for i, act in enumerate(activities[1:-1], start=0):
                    act.max_dur = self._minutes_to_timestr(activity_durations[i])
                    act.start_time = None  # Let MATSim calculate from routing
                    act.end_time = None    # Let MATSim calculate from arrival + max_dur

                # Last activity: no times at all (agent stays until simulation end)
                activities[-1].start_time = None
                activities[-1].end_time = None
                activities[-1].max_dur = None

                return True

            except Exception as e:
                if hasattr(logger, 'debug'):
                    logger.debug(f"Time assignment retry {retry+1}: {e}")
                continue

        return False

    def _minutes_to_timestr(self, minutes: float) -> str:
        """Convert minutes since midnight to HH:MM:SS format."""
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        secs = int((minutes % 1) * 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"


class _WorkerPlanGenerator(_BasePlanGenerator):
    """
    Lightweight plan generator for multiprocessing workers.
    Re-initializes only what's needed, avoiding unpicklable objects.
    Uses pre-loaded POI data to avoid DuckDB multi-connection issues.
    """

    def __init__(self, config: dict, data_dir: str, shared_data: dict):
        """Initialize worker with shared data."""
        self.config = config
        self.data_dir = data_dir

        # Store verbose setting for controlling repetitive log messages
        self.verbose = config.get('logging', {}).get('verbose', True)

        # Get shared data
        self.blockid2homelocs = shared_data['blockid2homelocs']
        self.blockid2worklocs = shared_data['blockid2worklocs']
        self.geo_level = shared_data.get('geo_level', BaseSurveyTrip.GEO_BLOCK_GROUP)

        # Re-initialize models with shared survey data
        survey_df = shared_data['survey_df']

        # Build models — blended when per-source data is available
        home_boost = config.get('chains', {}).get('home_boost_factor', 2.0)
        early_stop_exp = config.get('chains', {}).get('early_stop_exponent', 2.0)
        bw = config.get('time_models', {}).get('kde_bandwidth', 'scott')

        if 'per_source_data' in shared_data:
            # Multi-source: rebuild blended models from per-source data
            per_source_data = shared_data['per_source_data']       # {name: df}
            per_source_persons = shared_data['per_source_persons'] # {name: persons}
            blend_weights = shared_data['blend_weights']           # {name: float}

            # Chain model — per-source TripChainModels wrapped in blended
            per_source_chains_dfs = shared_data['per_source_chains_dfs']  # {name: chains_df}
            per_source_all_chains_dfs = shared_data.get('per_source_all_chains_dfs', {})
            chain_models = {}
            for name, cdf in per_source_chains_dfs.items():
                if not cdf.empty:
                    all_cdf = per_source_all_chains_dfs.get(name)
                    chain_models[name] = TripChainModel(
                        cdf, home_boost_factor=home_boost,
                        length_distribution_df=all_cdf,
                        early_stop_exponent=early_stop_exp)
            if len(chain_models) > 1:
                self.chain_model = BlendedTripChainModel(chain_models, blend_weights)
            else:
                self.chain_model = next(iter(chain_models.values()))

            # Time model — per-source TripDurationModels
            per_source_time = {name: TripDurationModel(df, config=config) for name, df in per_source_data.items()}
            if len(per_source_time) > 1:
                self.time_model = BlendedTripDurationModel(per_source_time, blend_weights)
            else:
                self.time_model = next(iter(per_source_time.values()))

            # Activity duration model — per-source ActivityDurationModels
            per_source_act = {
                name: ActivityDurationModel(p, bw_method=bw, config=config)
                for name, p in per_source_persons.items()
            }
            if len(per_source_act) > 1:
                self.activity_duration_model = BlendedActivityDurationModel(per_source_act, blend_weights)
            else:
                self.activity_duration_model = next(iter(per_source_act.values()))
        else:
            # Single-source path (original behaviour)
            chains_df = shared_data['chains_df']
            all_chains_df = shared_data.get('all_chains_df')
            self.chain_model = TripChainModel(chains_df, home_boost_factor=home_boost,
                                              length_distribution_df=all_chains_df,
                                              early_stop_exponent=early_stop_exp)
            self.time_model = TripDurationModel(survey_df, config=config)
            self.activity_duration_model = ActivityDurationModel(
                shared_data['persons'],
                bw_method=bw,
                config=config
            )


        # Use pre-loaded POI data (avoids DuckDB multi-connection issues)
        poi_data = shared_data['poi_data']
        if not poi_data:
            logger.error("  WARNING: poi_data is EMPTY!")
        total_pois = sum(len(pois) for pois in poi_data.values()) if poi_data else 0
        logger.debug(f"Worker POI data: {len(poi_data)} categories, {total_pois} POIs")

        # Build spatial index for fast POI lookups
        self.poi_index = POISpatialIndex(poi_data)

        # Stats (local to this worker)
        self.stats = {
            'failed_plans': 0,
            'chain_retries': 0,
            'chain_retries_too_short': 0,
            'chain_retries_bad_structure': 0,
            'chain_retries_missing_work': 0,
            'chain_retries_too_many_work': 0,
            'chain_attempts': 0,
            'poi_retries': 0,
            'time_retries': 0
        }

        # Rebuild GTFS availability indices from serialized stop data
        gtfs_avail_manager = self._rebuild_gtfs_indices(shared_data.get('gtfs_stop_data'))

        # Initialize mode choice model with survey data
        if 'per_source_data' in shared_data:
            # Multi-source: use per-source data and blend weights
            self.mode_choice = ModeChoiceModel(
                config,
                survey_data=shared_data['per_source_data'],
                survey_weights=shared_data['blend_weights'],
                gtfs_avail_manager=gtfs_avail_manager,
            )
        else:
            # Single source: wrap survey_df
            self.mode_choice = ModeChoiceModel(
                config,
                survey_data={'default': survey_df},
                survey_weights={'default': 1.0},
                gtfs_avail_manager=gtfs_avail_manager,
            )

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
        """Process one OD pair (same as PlanGenerator._process_od_pair)."""
        samples = generate_samples(
            origin_bg,
            dest_bg,
            num_trips,
            self.blockid2homelocs,
            self.blockid2worklocs,
            geo_level=self.geo_level,
        )
        # logger.info(f"process_od_pair(), len(samples):  {len(samples)}")

        home_locs = samples['home_locations']
        work_locs = samples['work_locations']

        if len(home_locs) != len(work_locs):
            return []

        plans = []
        for home_loc, work_loc in zip(home_locs, work_locs):
            plan = self._generate_single_plan(home_loc, work_loc)
            if plan:
                plans.append(plan)
            else:
                self.stats['failed_plans'] += 1

        return plans

    def _assign_poi_with_retry(self, current_location: Tuple[float, float],
                               activity: str) -> Optional[Dict]:
        """Assign POI with retry using pre-loaded POI data.

        Args:
            current_location: (lon, lat) geographic coordinates
            activity: Activity type to search for

        Returns:
            POI dict or None
        """
        poi_config = self.config.get('poi_assignment', {})
        radius = poi_config.get('initial_radius_m', 1000)
        radius_increment = poi_config.get('radius_increment_m', 500)
        max_retries = poi_config.get('max_poi_retries', 5)

        # Use activity directly - POIs are already categorized in the database
        # logger.info(f"_assign_poi_with_retry: activity='{activity}'")

        lon, lat = current_location  # current_location is already (lon, lat)
        # logger.info(f"  Searching near lat={lat:.6f}, lon={lon:.6f}, initial_radius={radius}m")

        for retry_attempt in range(max_retries):
            poi = self._find_nearby_poi_from_cache(lat, lon, activity, radius)
            if poi:
                # logger.info(f"  Found POI on attempt {retry_attempt+1}/{max_retries} with radius={radius}m")
                return poi
            if self.verbose:
                logger.info(f"  Attempt {retry_attempt+1}/{max_retries}: No POI found with radius={radius}m, expanding...")
            radius += radius_increment
            self.stats['poi_retries'] += 1

        if self.verbose:
            logger.warning(f"  Failed to find POI for '{activity}' after {max_retries} retries (max radius={radius}m)")
        return None

    def _find_nearby_poi_from_cache(self, lat: float, lon: float, activity: str, radius_m: float) -> Optional[Dict]:
        """Find nearest POI from pre-loaded cache data using spatial index."""
        poi = self.poi_index.find_nearest(lat, lon, activity, radius_m)

        if poi:
            pass
            # logger.info(f"  Found POI for '{activity}' within {radius_m}m: {poi.get('name', 'unknown')}")
        else:
            if self.verbose:
                logger.info(f"  No POIs within {radius_m}m for '{activity}'")

        return poi


class PlanGenerator(_BasePlanGenerator):
    """
    Main plan generation orchestrator.

    Coordinates all components to generate MATSim plans from OD matrix,
    activity chains, time distributions, and POI locations.
    """

    def __init__(self, config_path: Optional[str] = None, data_dir: Optional[str] = None, experiment_dir: Optional[str] = None):
        """
        Initialize plan generator.

        Args:
            config_path: Path to configuration JSON file
            data_dir: Path to data directory (optional, defaults to config value)
            experiment_dir: Path to experiment directory (optional, auto-created if not provided)
        """
        self.experiment_dir_override = experiment_dir
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                import json
                self.config = json.load(f)
        else:
            self.config = load_config()

        # Use data_dir from config if not provided
        if data_dir is None:
            self.data_dir = self.config.get('data', {}).get('data_dir', '../data')
        else:
            self.data_dir = data_dir

        # Reconfigure logger based on log_to_file/log_to_console settings from config.
        # Skip when driven by ExperimentRunner (experiment_dir provided) — it manages the logger.
        if experiment_dir is None:
            log_to_file = self.config.get('logging', {}).get('log_to_file', True)
            log_to_console = self.config.get('logging', {}).get('log_to_console', True)
            self._reconfigure_logger(log_to_file, log_to_console)

        # Store verbose setting for controlling repetitive log messages
        self.verbose = self.config.get('logging', {}).get('verbose', True)

        logger.info("=" * 70)
        logger.info("INITIALIZING PLAN GENERATOR")
        logger.info("=" * 70)

        # Load data filtered by counties in config
        logger.info("Loading location data...")
        logger.info(f"  Using data directory: {self.data_dir}")
        logger.info(f"  Filtering by {len(self.config['region']['counties'])} counties from config")
        self.blockid2homelocs = load_home_locations_by_counties(self.config)
        self.blockid2worklocs = load_work_locations_by_counties(self.config)
        logger.info(f"  Loaded {len(self.blockid2homelocs)} home blocks")
        logger.info(f"  Loaded {len(self.blockid2worklocs)} work blocks")

        # Load survey data via SurveyManager
        logger.info("Loading survey data...")
        self.survey_manager = SurveyManager(self.config)
        self.survey_df = self.survey_manager.get_survey_df()
        self.persons = self.survey_manager.get_persons()
        logger.info(f"  Loaded {len(self.survey_df)} survey trips")
        logger.info(f"  Processed {len(self.persons)} persons")

        # Detect census geography level from survey location IDs
        self.geo_level = SurveyManager.detect_geo_level_from_df(self.survey_df)
        if self.geo_level is None:
            self.geo_level = BaseSurveyTrip.GEO_BLOCK_GROUP
        logger.info(f"  Detected survey geo level: {self.geo_level}")

        # Initialize models — blended wrappers when multiple sources are active
        logger.info("Initializing statistical models...")
        if self.survey_manager.has_multiple_sources():
            logger.info("  Multiple survey sources detected — building blended models")
            all_data = self.survey_manager.load_data()
            all_persons = self.survey_manager.process_persons()
            weights = self.survey_manager.get_blend_weights()

            # 6a: Blended trip chain model
            self.chain_model = self._init_blended_chain_model(all_persons, weights)

            # 6b: Blended trip duration model
            per_source_time = {name: TripDurationModel(df, config=self.config) for name, df in all_data.items()}
            self.time_model = BlendedTripDurationModel(per_source_time, weights)

            # 6c: Blended activity duration model
            bw = self.config.get('time_models', {}).get('kde_bandwidth', 'scott')
            per_source_act = {
                name: ActivityDurationModel(p, bw_method=bw, config=self.config)
                for name, p in all_persons.items()
            }
            self.activity_duration_model = BlendedActivityDurationModel(per_source_act, weights)

            # Store per-source data for OD matrix blending (6d)
            self._all_survey_data = all_data
            self._blend_weights = weights
        else:
            logger.info("  Single survey source — using direct models")
            self.chain_model = self._init_chain_model()
            self.time_model = TripDurationModel(self.survey_df, config=self.config)
            self.activity_duration_model = ActivityDurationModel(
                self.persons,
                bw_method=self.config.get('time_models', {}).get('kde_bandwidth', 'scott'),
                config=self.config
            )
            self._all_survey_data = None
            self._blend_weights = None

        # Initialize POI manager
        logger.info("Initializing POI manager...")
        self.poi_manager = self._init_poi_manager()

        # Generate OD matrix
        logger.info("Generating OD matrix...")
        self.od_matrix = self._generate_od_matrix()

        # Statistics
        self.stats = {
            'total_plans_generated': 0,
            'chain_retries': 0,
            'chain_retries_too_short': 0,
            'chain_retries_bad_structure': 0,
            'chain_retries_missing_work': 0,
            'chain_retries_too_many_work': 0,
            'chain_attempts': 0,
            'poi_retries': 0,
            'time_retries': 0,
            'failed_plans': 0
        }

        # Initialize GTFS data and availability indices (Phase 3)
        logger.info("Initializing GTFS transit availability...")
        self.gtfs_avail_manager = self._init_gtfs_availability()

        # Initialize mode choice model with survey data for mode rate computation
        logger.info("Initializing mode choice model...")
        if self._all_survey_data is not None:
            # Multi-source: pass per-source survey data and weights
            self.mode_choice = ModeChoiceModel(
                self.config,
                survey_data=self._all_survey_data,
                survey_weights=self._blend_weights,
                gtfs_avail_manager=self.gtfs_avail_manager,
            )
        else:
            # Single source: wrap survey_df in dict for consistent interface
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

        logger.info("Plan generator initialized successfully!")
        logger.info("=" * 70)

    def _reconfigure_logger(self, log_to_file: bool, log_to_console: bool):
        """
        Reconfigure existing logger handlers based on config settings.

        Args:
            log_to_file: If False, remove file handlers
            log_to_console: If False, remove console handlers
        """
        import logging
        root_logger = logging.getLogger()

        # Remove handlers based on config
        handlers_to_remove = []
        for handler in root_logger.handlers:
            if not log_to_file and isinstance(handler, logging.FileHandler):
                handlers_to_remove.append(handler)
            elif not log_to_console and isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handlers_to_remove.append(handler)

        for handler in handlers_to_remove:
            handler.close()
            root_logger.removeHandler(handler)

    def _init_chain_model(self) -> TripChainModel:
        """Initialize and fit trip chain model."""
        # Process chains with weighting
        use_weight = self.config.get('chains', {}).get('use_weighted_chains', True)
        chains = process_trip_chains(self.persons, use_weight=use_weight)
        all_chains_df = pd.DataFrame(chains)

        # Filter to chains containing Work (any structure allowed)
        chains_df = filter_chains_by_type(all_chains_df, 'contains_work')

        # Initialize model — use full survey for chain length distribution
        # so the Markov generator targets realistic lengths, not the shorter
        # lengths typical of purpose-filtered subsets.
        home_boost = self.config.get('chains', {}).get('home_boost_factor', 2.0)
        early_stop_exp = self.config.get('chains', {}).get('early_stop_exponent', 2.0)
        model = TripChainModel(chains_df, home_boost_factor=home_boost,
                               length_distribution_df=all_chains_df,
                               early_stop_exponent=early_stop_exp)

        logger.info(f"  Chain model fitted with {len(chains_df)} patterns "
                    f"(length dist from {len(all_chains_df)} unfiltered)")
        return model

    def _init_blended_chain_model(self, all_persons: Dict[str, Dict],
                                   weights: Dict[str, float]):
        """Build per-source TripChainModels and wrap in BlendedTripChainModel."""
        use_weight = self.config.get('chains', {}).get('use_weighted_chains', True)
        home_boost = self.config.get('chains', {}).get('home_boost_factor', 2.0)
        early_stop_exp = self.config.get('chains', {}).get('early_stop_exponent', 2.0)

        per_source = {}
        for name, persons in all_persons.items():
            chains = process_trip_chains(persons, use_weight=use_weight)
            all_chains_df = pd.DataFrame(chains)
            chains_df = filter_chains_by_type(all_chains_df, 'contains_work')
            if chains_df.empty:
                logger.warning(f"  No valid chains from source '{name}' — skipping")
                continue
            per_source[name] = TripChainModel(chains_df, home_boost_factor=home_boost,
                                              length_distribution_df=all_chains_df,
                                              early_stop_exponent=early_stop_exp)
            logger.info(f"  Chain model for '{name}': {len(chains_df)} patterns "
                        f"(length dist from {len(all_chains_df)} unfiltered)")

        if len(per_source) == 1:
            return next(iter(per_source.values()))

        return BlendedTripChainModel(per_source, weights)

    def _init_poi_manager(self) -> POIManager:
        """Initialize POI manager with database connection."""
        db_manager = initialize_tables(self.data_dir)
        poi_manager = POIManager(db_manager)
        stats = poi_manager.get_stats()
        logger.info(f"  POI manager initialized: {stats['total_pois']} POIs loaded")
        return poi_manager

    def _init_gtfs_availability(self) -> Optional[GTFSAvailabilityManager]:
        """
        Initialize GTFS data pipeline and build spatial indices.

        When called from run_experiment.py, GTFS feeds are already downloaded
        and loaded into the DB (during setup_network), so we skip the redundant
        setup() call and only build spatial indices from existing DB data.

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

        db_manager = None
        try:
            db_manager = initialize_tables(self.data_dir)

            gtfs_manager = GTFSManager(self.config, db_manager)

            # Only run full setup if feeds are not already in the DB.
            # run_experiment.setup_network() calls gtfs_manager.setup() before
            # network generation, so by the time PlanGenerator is created the
            # feeds are already downloaded and loaded — no need to repeat the
            # discovery/download/load cycle (saves 3-5 seconds per call).
            if not gtfs_manager.has_feeds_loaded():
                logger.info("  GTFS feeds not yet in DB, running setup...")
                gtfs_manager.setup()
            else:
                logger.info("  GTFS feeds already in DB, skipping redundant setup")

            # Build spatial indices
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

    def _generate_od_matrix(self) -> pd.DataFrame:
        """Generate combined OD matrix from gravity model and survey.

        When multiple surveys are configured, per-source survey OD matrices
        are built (only from sources with location data) and blended via
        ``blend_survey_od_matrices`` before combining with the gravity model.
        """
        # Group blocks by block group for OD matrix
        home_locs_bg_dict = self._group_by_blockgroup(self.blockid2homelocs, is_home=True)
        work_locs_bg_dict = self._group_by_blockgroup(self.blockid2worklocs, is_home=False)

        logger.info(f"  Aggregated to {len(home_locs_bg_dict)} home block groups")
        logger.info(f"  Aggregated to {len(work_locs_bg_dict)} work block groups")

        # Create local OD matrix (gravity model)
        od_config = self.config.get('od_matrix', {})
        result = create_local_od_matrix(
            work_locs_bg_dict,
            home_locs_bg_dict,
            beta=od_config.get('beta', 1.5),
            max_iterations=od_config.get('max_iterations', 200),
            convergence_threshold=od_config.get('convergence_threshold', 0.03)
        )
        local_od_matrix = result['od_matrix']
        alpha = od_config.get('alpha', 0.1)

        # Build survey OD matrix (possibly blended from multiple sources)
        if self._all_survey_data is not None:
            # Multi-source path: only surveys with location data contribute
            loc_surveys = self.survey_manager.get_surveys_with_locations(
                self._all_survey_data
            )

            if not loc_surveys:
                logger.info("  No survey has location data — using gravity model only")
                return local_od_matrix

            hw_query = (
                f"{BaseSurveyTrip.ORIGIN_PURPOSE} == '{BaseSurveyTrip.ACT_HOME}' and "
                f"{BaseSurveyTrip.DESTINATION_PURPOSE} == '{BaseSurveyTrip.ACT_WORK}'"
            )

            if len(loc_surveys) == 1:
                # Single location-capable survey — same as old path
                single_df = next(iter(loc_surveys.values()))
                survey_hw_df = single_df.query(hw_query)
                survey_od_matrix = create_survey_od_matrix_using_trip_weight(survey_hw_df)
            else:
                # Multiple location-capable surveys — blend
                per_source_ods = {}
                for name, df in loc_surveys.items():
                    hw_df = df.query(hw_query)
                    if not hw_df.empty:
                        per_source_ods[name] = create_survey_od_matrix_using_trip_weight(hw_df)
                        logger.info(f"  Survey OD from '{name}': {per_source_ods[name].shape}")

                if not per_source_ods:
                    logger.info("  No survey H→W trips — using gravity model only")
                    return local_od_matrix

                survey_od_matrix = blend_survey_od_matrices(
                    per_source_ods, self._blend_weights
                )
        else:
            # Single-source path (original behaviour)
            survey_hw_df = self.survey_df.query(
                f"{BaseSurveyTrip.ORIGIN_PURPOSE} == '{BaseSurveyTrip.ACT_HOME}' and "
                f"{BaseSurveyTrip.DESTINATION_PURPOSE} == '{BaseSurveyTrip.ACT_WORK}'"
            )
            survey_od_matrix = create_survey_od_matrix_using_trip_weight(survey_hw_df)

        # Combine survey and gravity matrices
        combined_matrix = combine_od_matrices(
            survey_od_matrix,
            local_od_matrix,
            alpha=alpha
        )

        logger.info(f"  Combined OD matrix shape: {combined_matrix.shape}")
        logger.info(f"  Total trips in matrix: {combined_matrix.sum().sum():,.0f}")

        # Save the final combined OD matrix (gravity + survey) to the experiment folder
        experiment_dir = get_current_experiment_dir()
        combined_matrix_path = experiment_dir / "combined_od_matrix.csv"
        combined_matrix.to_csv(combined_matrix_path)
        logger.info(f"  Saved combined OD matrix to: {combined_matrix_path}")

        return combined_matrix

    def _group_by_blockgroup(self, blockid_dict: Dict, is_home: bool = True) -> Dict:
        """Group blocks by census geography zone and calculate centroids.

        The aggregation level is determined by self.geo_level:
          block_group → first 12 chars (default)
          tract       → first 11 chars
        """
        from collections import defaultdict

        _prefix_map = {BaseSurveyTrip.GEO_BLOCK_GROUP: 12, BaseSurveyTrip.GEO_TRACT: 11}
        prefix_len = _prefix_map.get(self.geo_level, 12)

        bg_dict = defaultdict(lambda: {
            'n_employees': 0,
            'coords': []
        })

        for block_id, data in blockid_dict.items():
            bg_id = block_id[:prefix_len]
            bg_dict[bg_id]['n_employees'] += data.get('n_employees', 0)

            # Extract lat/lon coordinates (now stored directly in data)
            if 'lat' in data and 'lon' in data and data['lat'] is not None and data['lon'] is not None:
                # Store as (lon, lat) for consistency with old (x, y) format
                bg_dict[bg_id]['coords'].append((data['lon'], data['lat']))

        # Calculate centroids
        result = {}
        for bg_id, data in bg_dict.items():
            coords = data['coords']
            if coords:
                centroid_lon = np.mean([c[0] for c in coords])
                centroid_lat = np.mean([c[1] for c in coords])
                result[bg_id] = {
                    'n_employees': data['n_employees'],
                    'centroid': (centroid_lon, centroid_lat)  # (lon, lat)
                }

        return result

    def generate_plans(self, target_plans = None) -> Tuple[List[Plan], Dict]:
        """
        Generate plans using multiprocessing.

        Args:
            target_plans: Number of plans to generate, or "all" to generate all scaled OD pairs
                         (None = use config value)

        Returns:
            Tuple of (List of Plan objects, stats dictionary with generation metrics)
        """
        # Use provided experiment directory or create new one
        if self.experiment_dir_override:
            experiment_dir = Path(self.experiment_dir_override)
            experiment_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using provided experiment directory: {experiment_dir}")
        else:
            experiment_dir = create_experiment_dir()
            logger.info(f"Created experiment directory: {experiment_dir}")

        # Reconfigure logger to write to experiment directory instead of logs/
        log_file_path = reconfigure_logger_to_experiment_dir(experiment_dir, log_prefix='plan_generation')
        logger.info(f"All logs will be saved to: {log_file_path}")

        if target_plans is None:
            target_plans = self.config.get('plan_generation', {}).get('target_plans', 1000)

        scaling_factor = self.config.get('plan_generation', {}).get('scaling_factor', 0.001)

        # Note: experiment_metadata.json is no longer written here
        # All metrics are consolidated into experiment_summary.json by ExperimentRunner

        logger.info("=" * 70)
        logger.info(f"GENERATING PLANS")
        logger.info(f"Target plans: {target_plans}")
        logger.info(f"Scaling factor: {scaling_factor}")
        logger.info(f"Experiment directory: {experiment_dir}")
        logger.info("=" * 70)

        # Extract non-zero OD pairs using vectorized operations (much faster)
        od_pairs = []

        # Convert to long format and filter in one operation
        od_stacked = self.od_matrix.stack()
        od_nonzero = od_stacked[od_stacked > 0]

        # Scale all trips at once using probabilistic rounding
        # This preserves the expected total and maintains spatial distribution
        original_total = od_nonzero.sum()

        # Apply work-specific scaling multiplier (boosts work trips independently of non-work trips)
        # Default is 1.0 (no boost). Use values > 1.0 to increase work trips during peak hours.
        work_scaling_multiplier = self.config.get('plan_generation', {}).get('work_scaling_multiplier', 1.0)
        effective_scaling = scaling_factor * work_scaling_multiplier

        if work_scaling_multiplier != 1.0:
            logger.info(f"Work scaling multiplier: {work_scaling_multiplier}")
            logger.info(f"Effective scaling: {scaling_factor} × {work_scaling_multiplier} = {effective_scaling}")

        expected_total = original_total * effective_scaling

        # Set random seed for reproducibility
        random_seed = self.config.get('plan_generation', {}).get('random_seed', 42)
        np.random.seed(random_seed)

        # Scale to float
        od_scaled_float = od_nonzero * effective_scaling

        # Deterministic part (floor)
        od_scaled_int = np.floor(od_scaled_float).astype(int)

        # Probabilistic part (fractional)
        # Each OD pair has a probability equal to its fractional part of being rounded up
        fractional_part = od_scaled_float - od_scaled_int
        random_values = np.random.random(len(fractional_part))
        round_up = (random_values < fractional_part).astype(int)

        # Combine deterministic and probabilistic parts
        od_scaled = od_scaled_int + round_up

        # Filter out zeros (some might still be zero after probabilistic rounding)
        od_scaled = od_scaled[od_scaled > 0]

        # Convert to list of tuples
        od_pairs = [(origin, dest, count)
                    for (origin, dest), count in od_scaled.items()]
        total_trips_scaled = od_scaled.sum()

        logger.info(f"Initial OD pairs: {len(od_pairs)}")
        logger.info(f"Original total trips: {original_total:,.0f}")
        logger.info(f"Expected scaled trips: {expected_total:,.0f}")
        logger.info(f"Actual scaled trips: {total_trips_scaled:,.0f}")
        logger.info(f"Scaling accuracy: {total_trips_scaled/expected_total*100:.1f}%")

        # Determine whether to use all OD pairs or limit to target
        use_all_od_pairs = (target_plans == "all" or target_plans == "ALL")

        if use_all_od_pairs:
            # Use all non-zero scaled OD pairs - representative sample of entire OD matrix
            logger.info(f"Using ALL scaled OD pairs (representative sample)")
            logger.info(f"Total plans to generate: {total_trips_scaled:,.0f}")
        elif total_trips_scaled > target_plans:
            # Strategy: Sample OD pairs to reach approximately target_plans
            # Sort by trip count (descending) and take top pairs that sum to target
            od_pairs_sorted = sorted(od_pairs, key=lambda x: x[2], reverse=True)

            selected_pairs = []
            cumulative_trips = 0

            for origin, dest, count in od_pairs_sorted:
                if cumulative_trips >= target_plans:
                    break

                # Take this pair, potentially reducing its count
                remaining_needed = target_plans - cumulative_trips
                actual_count = min(count, remaining_needed)

                if actual_count > 0:
                    selected_pairs.append((origin, dest, actual_count))
                    cumulative_trips += actual_count

            od_pairs = selected_pairs
            logger.info(f"Reduced to {len(od_pairs)} OD pairs with ~{cumulative_trips} total trips")
            logger.info(f"od_pairs: {od_pairs}")

        # Use multiprocessing
        num_processes = self.config.get('plan_generation', {}).get('num_processes', 4)
        logger.info(f"Using {num_processes} processes")

        # Generate plans
        all_plans = []

        if num_processes > 1:
            all_plans = self._generate_plans_parallel(od_pairs, num_processes)
        else:
            all_plans = self._generate_plans_sequential(od_pairs)

        # Assign person IDs
        for i, plan in enumerate(all_plans):
            plan.person_id = f"p_{i}"

        # Get paths for final logging
        from utils.logger import get_current_log_file

        log_file = get_current_log_file()
        experiment_dir = get_current_experiment_dir()

        # Calculate success rate
        total_requested = len(all_plans) + self.stats['failed_plans']
        success_rate = (len(all_plans) / total_requested * 100) if total_requested > 0 else 100.0

        logger.info("=" * 70)
        logger.info(f"GENERATION COMPLETE")
        logger.info(f"  Plans requested: {total_requested:,}")
        logger.info(f"  Plans generated: {len(all_plans):,}")
        logger.info(f"  Failed plans: {self.stats['failed_plans']:,}")
        logger.info(f"  Success rate: {success_rate:.2f}%")
        logger.info(f"  Chain retries: {self.stats['chain_retries']:,} / {self.stats['chain_attempts']:,} attempts")
        if self.stats['chain_retries'] > 0:
            logger.info(f"    Too short: {self.stats['chain_retries_too_short']:,}")
            logger.info(f"    Bad structure (not Home-...-Home): {self.stats['chain_retries_bad_structure']:,}")
            logger.info(f"    Missing Work: {self.stats['chain_retries_missing_work']:,}")
            logger.info(f"    Too many Work: {self.stats['chain_retries_too_many_work']:,}")
        logger.info(f"  POI retries: {self.stats['poi_retries']:,}")
        logger.info(f"  Time retries: {self.stats['time_retries']:,}")
        logger.info(f"  Experiment directory: {experiment_dir}")
        logger.info(f"  Full logs saved to: {log_file}")
        logger.info("=" * 70)

        # Log mode choice statistics
        self.mode_choice.log_stats_summary()

        logger.info(f"Plan generation complete!")
        logger.info(f"  Generated: {len(all_plans):,} plans ({success_rate:.2f}% success rate)")
        logger.info(f"  Failed: {self.stats['failed_plans']:,} plans")
        logger.info(f"  Experiment directory: {experiment_dir}")
        logger.info(f"  Detailed logs: {log_file}")

        # Build stats dict with all metrics
        generation_stats = {
            'plans_requested': total_requested,
            'plans_generated': len(all_plans),
            'failed_plans': self.stats['failed_plans'],
            'success_rate': round(success_rate, 2),
            'chain_retries': self.stats['chain_retries'],
            'chain_retries_too_short': self.stats['chain_retries_too_short'],
            'chain_retries_bad_structure': self.stats['chain_retries_bad_structure'],
            'chain_retries_missing_work': self.stats['chain_retries_missing_work'],
            'chain_retries_too_many_work': self.stats['chain_retries_too_many_work'],
            'chain_attempts': self.stats['chain_attempts'],
            'poi_retries': self.stats['poi_retries'],
            'time_retries': self.stats['time_retries'],
            'unscaled_trips': int(original_total),  # Total trips before scaling
            'mode_choice': self.mode_choice.get_stats_summary(),  # Mode choice statistics
        }

        return all_plans, generation_stats

    def _generate_plans_sequential(self, od_pairs: List[Tuple[str, str, int]]) -> List[Plan]:
        """Generate plans sequentially with progress bar."""
        all_plans = []

        show_progress = self.config.get('logging', {}).get('show_progress_bar', True)

        if show_progress:
            pbar = tqdm(od_pairs, desc="Generating plans")
        else:
            pbar = od_pairs

        for origin_bg, dest_bg, num_trips in pbar:
            plans = self._process_od_pair(origin_bg, dest_bg, num_trips)
            all_plans.extend(plans)

            if show_progress and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'plans': len(all_plans),
                    'failed': self.stats['failed_plans']
                })

        return all_plans

    def _aggregate_worker_stats(self, worker_stats: Dict) -> None:
        """Aggregate worker plan generation stats into main stats."""
        for key in ('failed_plans', 'chain_retries', 'chain_retries_too_short',
                     'chain_retries_bad_structure', 'chain_retries_missing_work',
                     'chain_retries_too_many_work', 'chain_attempts',
                     'poi_retries', 'time_retries'):
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

    def _generate_plans_parallel(self, od_pairs: List[Tuple[str, str, int]], num_processes: int) -> List[Plan]:
        """
        Generate plans in parallel using multiprocessing.

        Args:
            od_pairs: List of (origin_bg, dest_bg, num_trips) tuples
            num_processes: Number of parallel processes to use

        Returns:
            List of all generated plans
        """
        logger.info("Pre-loading data for multiprocessing...")

        # Pre-load POI data
        poi_data = self._load_all_pois_to_memory()
        logger.info(f"  Loaded {sum(len(v) for v in poi_data.values())} POIs across {len(poi_data)} activities")

        # Pre-process chain model data (avoid re-processing in each worker)
        chains_df = self._get_processed_chains_df()
        logger.info(f"  Pre-processed {len(chains_df)} chain patterns")

        # Prepare shared data for workers
        shared_data = {
            'blockid2homelocs': self.blockid2homelocs,
            'blockid2worklocs': self.blockid2worklocs,
            'survey_df': self.survey_df,
            'persons': self.persons,
            'poi_data': poi_data,
            'chains_df': chains_df,  # Share pre-processed chains
            'gtfs_stop_data': self._serialize_gtfs_stop_data(),  # GTFS stop coords per mode
            'geo_level': self.geo_level,
        }

        # Share unfiltered chain length distribution for single-source workers
        if hasattr(self.chain_model, '_length_distribution_df') and self.chain_model._length_distribution_df is not None:
            shared_data['all_chains_df'] = self.chain_model._length_distribution_df

        # Add per-source data for multi-source blending in workers
        if self._all_survey_data is not None:
            all_persons = self.survey_manager.process_persons()
            use_weight = self.config.get('chains', {}).get('use_weighted_chains', True)

            per_source_chains_dfs = {}
            per_source_all_chains_dfs = {}
            for name, persons in all_persons.items():
                chains = process_trip_chains(persons, use_weight=use_weight)
                all_cdf = pd.DataFrame(chains)
                cdf = filter_chains_by_type(all_cdf, 'contains_work')
                per_source_chains_dfs[name] = cdf
                per_source_all_chains_dfs[name] = all_cdf

            shared_data['per_source_data'] = self._all_survey_data
            shared_data['per_source_persons'] = all_persons
            shared_data['per_source_chains_dfs'] = per_source_chains_dfs
            shared_data['per_source_all_chains_dfs'] = per_source_all_chains_dfs
            shared_data['blend_weights'] = self._blend_weights

        # Split OD pairs into chunks for each process
        chunk_size = max(1, len(od_pairs) // num_processes)
        od_chunks = [od_pairs[i:i + chunk_size] for i in range(0, len(od_pairs), chunk_size)]

        logger.info(f"Split {len(od_pairs)} OD pairs into {len(od_chunks)} chunks")

        # Get current log file path to pass to workers
        from utils.logger import get_current_log_file
        log_file_path = get_current_log_file()
        logger.info(f"Workers will log to: {log_file_path}")

        # Prepare arguments for each worker (now includes log file path)
        worker_args = [(chunk, self.config, self.data_dir, shared_data, log_file_path) for chunk in od_chunks]

        all_plans = []

        try:
            # Use multiprocessing pool
            with mp.Pool(processes=num_processes) as pool:
                # Process chunks and show progress
                show_progress = self.config.get('logging', {}).get('show_progress_bar', True)

                if show_progress:
                    # Use imap_unordered for progress tracking
                    pbar = tqdm(total=len(od_chunks), desc="Processing chunks")
                    for chunk_plans, worker_stats in pool.imap_unordered(_worker_process_chunk, worker_args):
                        all_plans.extend(chunk_plans)
                        # Aggregate worker stats into main stats
                        self._aggregate_worker_stats(worker_stats)
                        self._aggregate_mode_choice_stats(worker_stats.get('mode_choice'))
                        pbar.update(1)
                        pbar.set_postfix({'plans': len(all_plans), 'failed': self.stats['failed_plans']})
                    pbar.close()
                else:
                    # Process without progress bar
                    results = pool.map(_worker_process_chunk, worker_args)
                    for chunk_plans, worker_stats in results:
                        all_plans.extend(chunk_plans)
                        # Aggregate worker stats into main stats
                        self._aggregate_worker_stats(worker_stats)
                        self._aggregate_mode_choice_stats(worker_stats.get('mode_choice'))

        except Exception as e:
            logger.error(f"Multiprocessing failed: {e}")
            logger.warning("Falling back to sequential processing")
            all_plans = self._generate_plans_sequential(od_pairs)

        return all_plans

    def _serialize_gtfs_stop_data(self) -> Optional[Dict]:
        """
        Serialize GTFS stop coordinates per mode for passing to workers.

        STRtree objects are not picklable, so we pass raw coordinates
        and let workers rebuild the spatial indices.

        Returns:
            Dict mapping mode_name -> list of (lon, lat) tuples, or None
        """
        if self.gtfs_avail_manager is None:
            return None

        stop_data = {}
        for mode_type, points_array in self.gtfs_avail_manager._stop_points.items():
            if len(points_array) > 0:
                # Convert numpy array to list of tuples for pickling
                stop_data[mode_type.value] = points_array.tolist()

        if stop_data:
            logger.info(f"  Serialized GTFS stop data for workers: "
                       f"{', '.join(f'{k}={len(v)} stops' for k, v in stop_data.items())}")
        return stop_data if stop_data else None

    def _get_processed_chains_df(self) -> pd.DataFrame:
        """
        Get pre-processed chains DataFrame for sharing with workers.
        Extracts the chains_df from the existing chain_model.
        """
        # The chain_model already has the processed chains_df
        # We can access it directly from the model's chains_df attribute
        return self.chain_model.chains_df

    def _load_all_pois_to_memory(self) -> Dict[str, List[Dict]]:
        """
        Load POIs from database into memory for multiprocessing.
        Filters by configured counties using FIPS codes.
        Returns dict mapping activity type to list of POI dicts.
        """
        from models.poi_manager import load_pois_by_counties

        poi_data = {}

        try:
            poi_data = load_pois_by_counties(self.config)

            total = sum(len(pois) for pois in poi_data.values())
            logger.info(f"Loading POIs from database: found {total} POIs (county-filtered)")

            logger.info(f"Grouped POIs into {len(poi_data)} activity categories:")
            for activity, pois in poi_data.items():
                logger.info(f"  - {activity}: {len(pois)} POIs")

        except Exception as e:
            logger.error(f"Failed to pre-load POI data: {e}")
            logger.exception(e)  # Show full traceback
            # Return empty dict if loading fails
            poi_data = {}

        return poi_data

    def _process_od_pair(self, origin_bg: str, dest_bg: str, num_trips: int) -> List[Plan]:
        """Process one OD pair and generate plans."""
        # Sample home and work locations
        samples = generate_samples(
            origin_bg,
            dest_bg,
            num_trips,
            self.blockid2homelocs,
            self.blockid2worklocs,
            geo_level=self.geo_level,
        )

        home_locs = samples['home_locations']
        work_locs = samples['work_locations']

        if len(home_locs) != len(work_locs):
            logger.warning(f"Mismatch in sampled locations for {origin_bg}->{dest_bg}")
            return []

        plans = []
        for home_loc, work_loc in zip(home_locs, work_locs):
            plan = self._generate_single_plan(home_loc, work_loc)
            if plan:
                plans.append(plan)
            else:
                self.stats['failed_plans'] += 1

        return plans

    def _assign_poi_with_retry(self, current_location: Tuple[float, float],
                               activity: str) -> Optional[Dict]:
        """Assign POI with expanding radius retry logic using POI manager.

        Args:
            current_location: (lon, lat) geographic coordinates
            activity: Activity type to search for

        Returns:
            POI dict or None
        """
        poi_config = self.config.get('poi_assignment', {})
        radius = poi_config.get('initial_radius_m', 1000)
        radius_increment = poi_config.get('radius_increment_m', 500)
        max_retries = poi_config.get('max_poi_retries', 5)

        # Use activity directly - POIs are already categorized in the database
        # logger.info(f"_assign_poi_with_retry (sequential): activity='{activity}'")

        # current_location is already (lon, lat)
        lon, lat = current_location

        for attempt in range(max_retries):
            poi = self.poi_manager.sample_nearby_poi(lat, lon, activity, radius_m=radius)
            if poi:
                return poi

            if self.verbose:
                logger.info(f"  Attempt {attempt+1}/{max_retries}: No POI found with radius={radius}m for '{activity}', expanding...")
            radius += radius_increment
            self.stats['poi_retries'] += 1

        if self.verbose:
            logger.warning(f"No POI found for {activity} after {max_retries} retries")
        return None

    def write_xml(self, plans: List[Plan], output_path: Optional[str] = None):
        """
        Write plans to MATSim XML format with UTM coordinates.

        Args:
            plans: List of Plan objects (with lat/lon coordinates)
            output_path: Path to output XML file. If None, saves to current experiment directory.
        """
        # Default to experiment directory if no path provided
        if output_path is None:
            experiment_dir = get_current_experiment_dir()
            output_path = str(experiment_dir / 'plans.xml')

        logger.info(f"Writing {len(plans)} plans to {output_path}")

        utm_epsg = self.config['coordinates']['utm_epsg']

        # Initialize coordinate converter
        converter = CoordinateConverter(utm_epsg=utm_epsg)
        logger.info(f"Converting coordinates from lat/lon to {utm_epsg}")

        # Create XML structure
        root = ET.Element('population')

        for plan in plans:
            person = ET.SubElement(root, 'person', id=plan.person_id)
            plan_elem = ET.SubElement(person, 'plan', selected="yes")

            for i, activity in enumerate(plan.activities):
                # Convert lat/lon to UTM
                # activity.x is lon, activity.y is lat (from plan generation)
                utm_x, utm_y = converter.latlon_to_utm(activity.y, activity.x)

                # Add activity with UTM coordinates
                precision = self.config.get('coordinates', {}).get('precision_decimal_places', 2)

                # Build activity attributes (only include times if they exist)
                act_attribs = {
                    'type': activity.type,
                    'x': f"{utm_x:.{precision}f}",
                    'y': f"{utm_y:.{precision}f}"
                }

                # Only add start_time if it's not None
                if activity.start_time is not None:
                    act_attribs['start_time'] = activity.start_time

                # Only add end_time if it's not None
                if activity.end_time is not None:
                    act_attribs['end_time'] = activity.end_time

                # Only add max_dur if it's not None
                if activity.max_dur is not None:
                    act_attribs['max_dur'] = activity.max_dur

                act_elem = ET.SubElement(plan_elem, 'activity', **act_attribs)

                # Add leg (except after last activity)
                if i < len(plan.activities) - 1:
                    # Get mode from the corresponding leg
                    leg_mode = plan.legs[i].mode if i < len(plan.legs) else 'car'
                    leg_elem = ET.SubElement(plan_elem, 'leg', mode=leg_mode)

        # Pretty print and write
        xml_str = self._prettify_xml(root)

        # Add DOCTYPE with coordinate system reference
        xml_str = '<?xml version="1.0" ?>\n' + \
                 '<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">\n' + \
                 f'<!-- Coordinate System: {utm_epsg} -->\n' + \
                 xml_str

        with open(output_path, 'w') as f:
            f.write(xml_str)

        logger.info(f"Successfully wrote plans to {output_path} (coordinates in {utm_epsg})")

        experiment_dir = get_current_experiment_dir()
        if str(experiment_dir) in output_path:
            logger.info(f"Plans XML saved to experiment directory: {output_path}")
            logger.info(f"  Coordinate system: {utm_epsg}")

    def _prettify_xml(self, elem: ET.Element) -> str:
        """Return a pretty-printed XML string without XML declaration."""
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        # Get pretty XML and remove the auto-generated XML declaration
        pretty_xml = reparsed.toprettyxml(indent="    ")
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        # Filter out the XML declaration line that toprettyxml() adds
        return '\n'.join([line for line in lines if not line.startswith('<?xml')])

    def _write_experiment_metadata(self, experiment_dir: Path, target_plans, scaling_factor: float):
        """Write metadata file for the experiment.

        Args:
            experiment_dir: Path to experiment directory
            target_plans: Target number of plans (int or "all")
            scaling_factor: OD matrix scaling factor
        """
        from datetime import datetime
        import json

        metadata = {
            'timestamp': datetime.now().isoformat(),
            'target_plans': target_plans,
            'scaling_factor': scaling_factor,
            'config': {
                'num_processes': self.config.get('plan_generation', {}).get('num_processes', 4),
                'supported_chain_types': self.config.get('plan_generation', {}).get('supported_chain_types', []),
                'od_matrix': {
                    'beta': self.config.get('od_matrix', {}).get('beta'),
                    'alpha': self.config.get('od_matrix', {}).get('alpha')
                },
                'poi_assignment': {
                    'initial_radius_m': self.config.get('poi_assignment', {}).get('initial_radius_m'),
                    'max_poi_retries': self.config.get('poi_assignment', {}).get('max_poi_retries')
                }
            },
            'data_summary': {
                'num_home_blocks': len(self.blockid2homelocs),
                'num_work_blocks': len(self.blockid2worklocs),
                'survey_trips': len(self.survey_df),
                'survey_persons': len(self.persons)
            }
        }

        metadata_path = experiment_dir / 'experiment_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  Experiment metadata saved to: {metadata_path}")

# !!! DEBUGGING IS NOT FULLY WORKING BECAUSE THE CONNECTION TO DUCKDB FROM THE DEBUGGER !!!
# Use the below main func to run no Jetstream
if __name__ == "__main__":
    # Example usage
    generator = PlanGenerator()
    plans = generator.generate_plans()
    generator.write_xml(plans)
