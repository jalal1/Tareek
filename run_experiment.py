#!/usr/bin/env python3
"""
Main Experiment Runner for MATSim Twin Cities Simulation

This script runs complete experiments by:
1. Validating configuration
2. Generating or reusing cached networks
3. Generating activity plans
4. Running MATSim simulation

Usage:
    python run_experiment.py --config config/config.json [--experiment-id my_experiment]

Arguments:
    --config: Path to configuration JSON file (required)
    --experiment-id: Custom experiment ID (optional, auto-generated if not provided)
    --skip-simulation: Generate plans but don't run simulation (optional)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.logger import setup_logger, reconfigure_logger_to_experiment_dir
from utils.config_validator import ConfigValidator, ConfigValidationError
from models.plan_generator import PlanGenerator
from models.plan_generator_nonwork import NonWorkPlanGenerator
from models.od_matrix_nonwork import compute_poi_block_mapping
from matsim.network_generator import NetworkGenerator
from matsim.network_manager import NetworkManager
from matsim.orchestrator import MATSimOrchestrator
from matsim.evaluator import SimulationEvaluator
from matsim.counts_generator import CountsGenerator
from utils.experiment_tracker import ExperimentTracker
from utils.experiment_summary import build_summary, write_summary

logger = setup_logger(__name__)


def load_shared_nonwork_data(config: Dict) -> Dict:
    """
    Load shared data for non-work plan generation.

    This data is shared across all non-work purposes (Shopping, School, etc.)
    to avoid redundant database queries and processing.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with shared data:
        - home_locs_dict: Home locations filtered by county
        - poi_data_flat: Flat list of POIs (filtered by county FIPS)
        - poi_data_grouped: POIs grouped by activity type
        - survey_df: Survey DataFrame
        - persons: Processed persons list
        - chains_df: Pre-processed chains DataFrame
    """
    import pandas as pd
    from models.home_locs_v2 import load_home_locations_by_counties
    from data_sources.survey_manager import SurveyManager
    from models.chains import process_trip_chains

    logger.info("-" * 60)
    logger.info("LOADING SHARED DATA FOR NON-WORK PLANS")
    logger.info("-" * 60)

    # Load home locations (with non_employees) - filtered by counties in config
    home_locs_dict = load_home_locations_by_counties(config)
    logger.info(f"  Loaded {len(home_locs_dict):,} home blocks (county-filtered)")

    # Load POI data from database, filtered by configured counties
    from models.poi_manager import load_pois_by_counties
    poi_data_grouped = load_pois_by_counties(config)
    poi_data_flat = [poi for pois in poi_data_grouped.values() for poi in pois]
    logger.info(f"  Loaded {len(poi_data_flat):,} POIs in {len(poi_data_grouped)} activity types (county-filtered)")

    # Pre-compute POI-to-block mapping ONCE for all activity types
    # This avoids redundant spatial computations when creating OD matrices
    logger.info("  Pre-computing POI-to-block mapping...")
    poi_block_mapping = compute_poi_block_mapping(poi_data_flat, home_locs_dict, config)
    logger.info(f"  POI-to-block mapping complete: {len(poi_block_mapping):,} POIs mapped")

    # Load survey data via SurveyManager
    survey_manager = SurveyManager(config)
    survey_df = survey_manager.get_survey_df()
    persons = survey_manager.get_persons()
    logger.info(f"  Loaded {len(survey_df):,} survey trips")
    logger.info(f"  Processed {len(persons):,} persons")

    # Process chains
    use_weight = config.get('chains', {}).get('use_weighted_chains', True)
    chains = process_trip_chains(persons, use_weight=use_weight)
    chains_df = pd.DataFrame(chains)
    logger.info(f"  Processed {len(chains_df):,} trip chains")

    # Initialize shared time models ONCE (used by all activity types)
    logger.info("  Initializing shared time models...")
    from models.time import TripDurationModel, ActivityDurationModel
    trip_duration_model = TripDurationModel(survey_df, config=config)
    bw_method = config.get('time_models', {}).get('kde_bandwidth', 'scott')
    activity_duration_model = ActivityDurationModel(persons, bw_method=bw_method, config=config)
    logger.info(f"  Time models initialized")

    # Initialize shared POI spatial index ONCE (contains all activity types)
    logger.info("  Building shared POI spatial index...")
    from utils.poi_spatial_index import POISpatialIndex
    poi_spatial_index = POISpatialIndex(poi_data_grouped)
    stats = poi_spatial_index.get_stats()
    logger.info(f"  POI spatial index built: {stats['num_activities']} activity types, {stats['total_pois']:,} POIs")

    # Build per-source data for multi-source blending (if applicable)
    multi_source_data = {}
    if survey_manager.has_multiple_sources():
        logger.info("  Multi-source mode: building per-source data for blending...")
        all_data = survey_manager.load_data()
        all_persons = survey_manager.process_persons()
        blend_weights = survey_manager.get_blend_weights()

        # Build per-source chains DataFrames
        per_source_chains_dfs = {}
        for name, src_persons in all_persons.items():
            src_chains = process_trip_chains(src_persons, use_weight=use_weight)
            per_source_chains_dfs[name] = pd.DataFrame(src_chains)
            logger.info(f"    {name}: {len(per_source_chains_dfs[name]):,} chains")

        multi_source_data = {
            'per_source_data': all_data,
            'per_source_persons': all_persons,
            'per_source_chains_dfs': per_source_chains_dfs,
            'blend_weights': blend_weights,
        }
        logger.info(f"  Multi-source blending ready: {list(blend_weights.items())}")

    logger.info("-" * 60)

    result = {
        'home_locs_dict': home_locs_dict,
        'poi_data_flat': poi_data_flat,
        'poi_data_grouped': poi_data_grouped,
        'poi_block_mapping': poi_block_mapping,
        'survey_df': survey_df,
        'persons': persons,
        'chains_df': chains_df,
        'trip_duration_model': trip_duration_model,
        'activity_duration_model': activity_duration_model,
        'poi_spatial_index': poi_spatial_index,
    }
    result.update(multi_source_data)
    return result


class ExperimentRunner:
    """Main experiment orchestrator combining plan generation and MATSim simulation"""

    def __init__(self, config_path: Path, experiment_id: Optional[str] = None,
                 experiments_root: Optional[Path] = None):
        """
        Initialize experiment runner

        Args:
            config_path: Path to configuration JSON file
            experiment_id: Optional custom experiment ID
            experiments_root: Optional root directory under which the
                experiment directory is created. Defaults to
                project_root / 'experiments', overridable via the
                TAREEK_EXPERIMENTS_ROOT env var. Lets tests redirect output
                to a temp dir so real experiments are never touched.
        """
        self.config_path = Path(config_path)
        self.experiment_id = experiment_id or self._generate_experiment_id()
        if experiments_root is not None:
            self.experiments_root = Path(experiments_root)
        else:
            self.experiments_root = Path(
                os.environ.get('TAREEK_EXPERIMENTS_ROOT', project_root / 'experiments')
            )
        self.config = None
        self.validator = None

        # Initialize components (will be set up during run)
        self.plan_generator = None
        self.network_manager = None
        self.network_generator = None
        self.orchestrator = None

        # Experiment paths
        self.experiment_dir = None
        self.plans_path = None
        self.network_path = None
        self.counts_path = None

        # Metrics collection (populated during run)
        # plan_stats will be populated dynamically based on config's nonwork_purposes
        self.plan_stats = {
            'work': {},  # Work is always tracked
        }
        self.population_stats = {
            'total_population': 0,
            'total_employees': 0,
            'total_non_employees': 0,
        }
        self.data_quality_stats = {
            'home_blocks': 0,
            'home_blocks_with_coords': 0,
            'total_pois': 0,
            'poi_activity_types': 0,
        }
        self.network_stats = {
            'num_nodes': 0,
            'num_links': 0,
            'file_size_mb': 0,
        }
        self.counts_stats = {
            'num_devices_matched': 0,
            'num_count_locations': 0,
            'generated': False,
        }
        self.plans_file_size_mb = 0
        self.runtime = {
            'start_time': None,
            'plans_start': None,
            'plans_end': None,
            'matsim_start': None,
            'matsim_end': None,
            'eval_start': None,
            'eval_end': None,
            'end_time': None,
        }

    def _generate_experiment_id(self) -> str:
        """Generate timestamp-based experiment ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"experiment_{timestamp}"

    def _validate_network_file(self, network_path: Path) -> bool:
        """
        Validate that network.xml exists and is valid

        Args:
            network_path: Path to network.xml file

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check file exists
            if not network_path.exists():
                return False

            # Check file has content
            if network_path.stat().st_size == 0:
                logger.warning(f"Network file is empty: {network_path}")
                return False

            # Check XML is parseable (basic validation)
            import xml.etree.ElementTree as ET
            tree = ET.parse(network_path)
            root = tree.getroot()

            # Check it's a network file
            if root.tag != 'network':
                logger.warning(f"File is not a valid network XML: {network_path}")
                return False

            logger.debug(f"Network file validation passed: {network_path}")
            return True

        except ET.ParseError as e:
            logger.warning(f"Network file has invalid XML: {e}")
            return False
        except Exception as e:
            logger.warning(f"Network file validation failed: {e}")
            return False

    # Config sections that change the generated demand. Anything listed here
    # is part of the cache key; a change to any of it must miss the cache.
    _DEMAND_CACHE_SECTIONS = (
        'region',            # which counties
        'od_matrix',         # source, friction, distance, boundary policy, beta...
        'plan_generation',   # scaling factor, target plans, seed
        'nonwork_purposes',  # non-work trip rates and betas
        'chains',            # activity chains
        'time_models',       # departure times
        'modes',             # mode availability
        'mode_choice',       # mode choice parameters
        'poi_assignment',
        'duration_constraints',
        'freight',           # boundary truck demand: cordons, volumes, profiles
    )

    def _demand_cache_key(self) -> str:
        """Fingerprint of every config value that affects the generated demand.

        Deliberately conservative: it hashes whole config sections rather than
        a hand-picked list of keys, so a newly added parameter cannot silently
        fall outside the key and serve stale plans. Keys starting with '_' are
        ignored because they are documentation, not behaviour.
        """
        import hashlib
        import json as _json

        def strip_help(obj):
            if isinstance(obj, dict):
                return {k: strip_help(v) for k, v in sorted(obj.items())
                        if not str(k).startswith('_')}
            if isinstance(obj, list):
                return [strip_help(v) for v in obj]
            return obj

        payload = {s: strip_help(self.config.get(s, {}))
                   for s in self._DEMAND_CACHE_SECTIONS}
        # LODES year/job_type change the underlying flows, so include them too.
        payload['lodes'] = strip_help(self.config.get('data', {}).get('lodes', {}))

        blob = _json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _demand_cache_dir(self) -> Path:
        data_dir = self.config['data']['data_dir']
        if not Path(data_dir).is_absolute():
            data_dir = (self.config_path.parent / data_dir).resolve()
        return Path(data_dir) / 'demand_cache'

    def _try_reuse_cached_demand(self) -> bool:
        """Copy a previously generated plans.xml in, when the inputs match.

        Plan generation is the slow part of a run (~13 minutes at 15-county
        scale), and it is fully determined by the config. When only downstream
        settings change — simulation iterations, evaluation options — there is
        no reason to rebuild the same demand.

        Enabled with plan_generation.cache_demand = true. Off by default, so
        no existing workflow silently starts reusing plans.
        """
        if not self.config.get('plan_generation', {}).get('cache_demand', False):
            return False

        key = self._demand_cache_key()
        cached = self._demand_cache_dir() / f'plans_{key}.xml'

        if not cached.exists():
            logger.info(f"Demand cache: no entry for this configuration ({key})")
            logger.info(f"  Will generate and store at: {cached}")
            return False

        if not self._validate_plans_file(cached):
            logger.warning(f"Demand cache entry is invalid, regenerating: {cached}")
            return False

        import shutil
        shutil.copy2(cached, self.plans_path)

        size_mb = self.plans_path.stat().st_size / 1024 / 1024
        logger.info("=" * 60)
        logger.info("REUSING CACHED DEMAND")
        logger.info("=" * 60)
        logger.info(f"  Cache key : {key}")
        logger.info(f"  Source    : {cached}")
        logger.info(f"  Size      : {size_mb:.2f} MB")
        logger.info("  Plan generation skipped - the config values that affect")
        logger.info("  demand are unchanged since this was generated.")
        logger.info("  (set plan_generation.cache_demand=false to force a rebuild)")
        logger.info("")

        # Carry the matching diagnostics across so the run still reports how
        # its demand was built, rather than appearing to have none.
        for artefact in ('od_matrix_diagnostics.json', 'combined_od_matrix.csv',
                         'base_od_matrix.csv'):
            src = cached.parent / f'{key}_{artefact}'
            if src.exists():
                shutil.copy2(src, self.experiment_dir / artefact)

        self.plans_file_size_mb = round(size_mb, 2)
        self.runtime['plans_start'] = datetime.now()
        self.runtime['plans_end'] = datetime.now()
        return True

    def _store_cached_demand(self) -> None:
        """Save the finished demand for reuse by later runs."""
        if not self.config.get('plan_generation', {}).get('cache_demand', False):
            return

        try:
            key = self._demand_cache_key()
            cache_dir = self._demand_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            target = cache_dir / f'plans_{key}.xml'

            import shutil
            shutil.copy2(self.plans_path, target)

            # Keep the OD artefacts beside the plans so a cache hit can restore
            # the full picture of how this demand was produced.
            for artefact in ('od_matrix_diagnostics.json', 'combined_od_matrix.csv',
                             'base_od_matrix.csv'):
                src = self.experiment_dir / artefact
                if src.exists():
                    shutil.copy2(src, cache_dir / f'{key}_{artefact}')

            logger.info(f"Demand cached for reuse: {target}")
            logger.info(f"  Cache key: {key}")
        except Exception as e:  # noqa: BLE001 - caching must never fail a run
            logger.warning(f"Could not cache demand (run is unaffected): {e}")

    def _validate_plans_file(self, plans_path: Path) -> bool:
        """
        Validate that plans.xml exists and is valid

        Args:
            plans_path: Path to plans.xml file

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check file exists
            if not plans_path.exists():
                return False

            # Check file has content
            if plans_path.stat().st_size == 0:
                logger.warning(f"Plans file is empty: {plans_path}")
                return False

            # Check XML is parseable (basic validation)
            import xml.etree.ElementTree as ET
            tree = ET.parse(plans_path)
            root = tree.getroot()

            # Check it's a population/plans file
            if root.tag != 'population':
                logger.warning(f"File is not a valid plans XML: {plans_path}")
                return False

            logger.debug(f"Plans file validation passed: {plans_path}")
            return True

        except ET.ParseError as e:
            logger.warning(f"Plans file has invalid XML: {e}")
            return False
        except Exception as e:
            logger.warning(f"Plans file validation failed: {e}")
            return False

    def validate_config(self) -> Dict:
        """
        Validate configuration file

        Returns:
            Validated configuration dictionary

        Raises:
            ConfigValidationError: If validation fails
        """
        logger.info("="*60)
        logger.info("STEP 1: VALIDATING CONFIGURATION")
        logger.info("="*60)

        try:
            self.validator = ConfigValidator(self.config_path)
            self.config = self.validator.validate()
            self.config["_config_dir"] = str(self.config_path.parent)

            logger.info(f"Configuration file: {self.config_path}")
            logger.info(f"Experiment ID: {self.experiment_id}")
            logger.info("")

            return self.config

        except ConfigValidationError as e:
            logger.error(f"Configuration validation failed:")
            logger.error(f"  {str(e)}")
            logger.error("")
            logger.error("Please fix the errors in your config.json and try again.")
            raise

    def detect_coordinate_system(self):
        """
        Auto-detect the UTM EPSG code from the configured counties' centroids.

        Queries the counties table for intptlat/intptlon (Census internal point),
        computes the average centroid, and determines the NAD83 UTM zone.
        Stores the result in self.config['coordinates']['utm_epsg'].
        """
        logger.info("=" * 60)
        logger.info("DETECTING COORDINATE SYSTEM")
        logger.info("=" * 60)

        from utils.coordinates import detect_utm_epsg
        from utils.region_utils import ensure_counties_in_db
        from models.models import County
        from utils.duckdb_manager import DBManager

        county_geoids = self.config['region']['counties']
        data_dir = self.config['data']['data_dir']
        db_manager = DBManager(data_dir)

        ensure_counties_in_db(county_geoids, db_manager)

        with db_manager.Session() as session:
            counties = session.query(County).filter(
                County.geoid.in_(county_geoids)
            ).all()

            if not counties:
                raise RuntimeError(f"Failed to fetch counties for GEOIDs: {county_geoids}")

            # Extract data while session is open
            county_names = [c.county_name for c in counties]
            lats = [c.intptlat for c in counties if c.intptlat is not None]
            lons = [c.intptlon for c in counties if c.intptlon is not None]

        if not lats or not lons:
            raise RuntimeError(
                "Counties in database are missing intptlat/intptlon values. "
                "Re-run notebooks/0.setup_global_data.ipynb to populate county centroids."
            )

        avg_lat = sum(lats) / len(lats)
        avg_lon = sum(lons) / len(lons)
        utm_epsg = detect_utm_epsg(avg_lat, avg_lon)

        # Store in config so all downstream components can read it
        if 'coordinates' not in self.config:
            self.config['coordinates'] = {}
        self.config['coordinates']['utm_epsg'] = utm_epsg

        logger.info(f"  Counties: {', '.join(county_names)}")
        logger.info(f"  Average centroid: ({avg_lat:.4f}, {avg_lon:.4f})")
        logger.info(f"  Detected UTM EPSG: {utm_epsg}")
        logger.info(f"  All downstream components will use {utm_epsg}")
        logger.info("")

    def setup_matsim_config(self):
        """Write the experiment's MATSim config.xml right after the experiment
        directory exists, before network/counts/plans generation.

        Doing this early surfaces any errors in configurable_params (typos,
        stale keys, base-template drift) in milliseconds, instead of after
        plan generation has burned 30-90 minutes. Generating the XML only
        needs the JSON config and the coordinate system - it doesn't read
        the network, plans, or counts files (it just *references* them by
        relative filename, and MATSim opens them at simulation time).

        Step 5 will regenerate this file from the same inputs before running
        the simulation; the two writes are bit-identical, so this is safe
        even if a later step were to mutate state we depend on. The file
        also persists for inspection if a downstream step crashes.
        """
        from matsim.config_manager import ConfigManager

        logger.info("=" * 60)
        logger.info("STEP 2b: GENERATING MATSIM config.xml")
        logger.info("=" * 60)

        matsim_config = self.config.get('matsim', {})
        mode = matsim_config.get('mode', 'basic')
        custom_params = matsim_config.get('configurable_params', {}) or None
        coord_system = self.config.get('coordinates', {}).get('utm_epsg', 'EPSG:26915')

        config_dir = self.config.get('_config_dir')
        cm = ConfigManager(
            self.config,
            config_dir=Path(config_dir) if config_dir else None,
        )

        config_path = self.experiment_dir / 'config.xml'
        try:
            cm.generate_config(
                output_path=config_path,
                experiment_path=self.experiment_dir,
                coordinate_system=coord_system,
                mode=mode,
                custom_params=custom_params,
            )
        except Exception as e:
            logger.error(f"MATSim config generation failed: {e}")
            raise RuntimeError(
                f"MATSim config generation failed before any expensive work "
                f"was done. Fix the configurable_params entry and re-run. "
                f"Underlying error: {e}"
            ) from e

        # vehicles.xml must exist beside config.xml whenever PCE is on, because
        # generate_config has just pointed the vehicles module at it. Writing it
        # here keeps the two in step: neither is written without the other.
        from models.freight.vehicles import write_vehicles_file
        write_vehicles_file(self.config, self.experiment_dir / 'vehicles.xml')

        # Record the path so it can be used by subsequent steps and the
        # final metadata. Step 5 will overwrite the file with the same
        # content; the path stays valid.
        self.matsim_config_path = config_path
        logger.info(f"  Generated: {config_path}")
        logger.info("")

    def setup_experiment_directory(self):
        """Create experiment directory structure"""
        logger.info("="*60)
        logger.info("STEP 2: SETTING UP EXPERIMENT DIRECTORY")
        logger.info("="*60)

        # Create experiment directory
        self.experiment_dir = self.experiments_root / self.experiment_id

        # Check if directory already exists
        dir_exists = self.experiment_dir.exists()

        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Reconfigure logger to write to experiment directory
        log_file_path = reconfigure_logger_to_experiment_dir(self.experiment_dir, log_prefix='experiment')

        if dir_exists:
            logger.info(f"Reusing existing experiment directory: {self.experiment_dir}")
        else:
            logger.info(f"Created new experiment directory: {self.experiment_dir}")

        logger.info(f"All experiment logs will be saved to: {log_file_path}")
        logger.info("")

    def setup_network(self) -> Path:
        """
        Setup or reuse network based on configuration

        Returns:
            Path to network.xml file

        Raises:
            RuntimeError: If network generation fails
        """
        logger.info("="*60)
        logger.info("STEP 3: SETTING UP NETWORK")
        logger.info("="*60)

        try:
            # Target network path in experiment directory
            self.network_path = self.experiment_dir / 'network.xml'

            # Check if rebuild is forced
            rebuild_network = self.config.get('network', {}).get('rebuild_network', False)

            # Check if network.xml already exists in experiment directory
            if self.network_path.exists() and not rebuild_network:
                logger.info(f"Found existing network.xml in experiment directory")
                logger.info(f"  File: {self.network_path}")

                # Validate the existing network file
                if self._validate_network_file(self.network_path):
                    # Try to extract metadata from the file
                    import xml.etree.ElementTree as ET
                    try:
                        tree = ET.parse(self.network_path)
                        root = tree.getroot()
                        nodes = root.find('nodes')
                        links = root.find('links')
                        num_nodes = len(nodes) if nodes is not None else 0
                        num_links = len(links) if links is not None else 0

                        file_size_mb = self.network_path.stat().st_size / 1024 / 1024
                        logger.info(f"  Size: {file_size_mb:.2f} MB")
                        logger.info(f"  Nodes: {num_nodes}")
                        logger.info(f"  Links: {num_links}")
                        logger.info("Reusing existing network.xml")
                        logger.info("")

                        # Store network stats
                        self.network_stats = {
                            'num_nodes': num_nodes if isinstance(num_nodes, int) else 0,
                            'num_links': num_links if isinstance(num_links, int) else 0,
                            'file_size_mb': round(file_size_mb, 2),
                        }

                        return self.network_path
                    except Exception as e:
                        logger.debug(f"Could not extract metadata from network: {e}")
                        logger.info("Reusing existing network.xml")
                        logger.info("")
                        return self.network_path
                else:
                    logger.warning("Existing network.xml is invalid, will regenerate...")
            elif self.network_path.exists() and rebuild_network:
                logger.info("rebuild_network=true: ignoring existing network.xml, will regenerate")
                # Remove stale transit files from experiment dir so they don't get reused
                for fname in ('transitSchedule.xml', 'transitVehicles.xml'):
                    stale = self.experiment_dir / fname
                    if stale.exists():
                        stale.unlink()
                        logger.info(f"  Removed stale {fname}")

            # Initialize network components
            self.network_manager = NetworkManager()
            self.network_generator = NetworkGenerator(self.config)

            # Get network specification from config (returns FIPS codes)
            county_geoids, polygon = self.validator.get_network_spec()

            # Convert FIPS codes to county names for network generation
            county_names = None
            if county_geoids:
                from utils.region_utils import RegionHelper
                region_helper = RegionHelper(self.config)
                county_names = region_helper.get_county_names_for_network()
                logger.info(f"Converted {len(county_geoids)} FIPS codes to county names")

            # Get db_manager for transit network path (needs GTFS feed data)
            db_manager = None
            # Check if transit_network is enabled AND at least one transit mode is enabled
            modes_config = self.config.get('modes', {})
            has_enabled_transit = any(
                isinstance(cfg, dict) and cfg.get('enabled', True)
                and cfg.get('matsim_mode') == 'pt'
                for cfg in modes_config.values()
            )
            if self.config.get('matsim', {}).get('transit_network', False) and has_enabled_transit:
                from utils.duckdb_manager import DBManager
                from data_sources.gtfs_manager import GTFSManager
                data_dir = self.config['data']['data_dir']
                db_manager = DBManager(data_dir)

                # Download and load GTFS feeds before network generation
                logger.info("Transit network enabled - setting up GTFS feeds...")
                gtfs_manager = GTFSManager(self.config, db_manager)
                gtfs_manager.setup()
                logger.info("GTFS setup complete")
            elif self.config.get('matsim', {}).get('transit_network', False) and not has_enabled_transit:
                logger.warning("transit_network is true but no transit modes are enabled. "
                             "Skipping GTFS setup, will generate road-only network.")

            # Get or generate network
            network_path, network_metadata = self.network_manager.get_or_generate_network(
                network_generator=self.network_generator,
                config=self.config,
                output_path=self.network_path,
                counties=county_names,
                polygon=polygon,
                db_manager=db_manager,
            )

            logger.info(f"Network ready: {network_path}")
            logger.info(f"  Nodes: {network_metadata.get('num_nodes', 'N/A')}")
            logger.info(f"  Links: {network_metadata.get('num_links', 'N/A')}")
            logger.info(f"  Coordinate system: {network_metadata.get('coordinate_system', 'N/A')}")
            if network_metadata.get('transit_network'):
                experiment_dir = self.network_path.parent
                schedule_path = experiment_dir / 'transitSchedule.xml'
                vehicles_path = experiment_dir / 'transitVehicles.xml'
                logger.info(f"  Transit schedule: {schedule_path} (exists={schedule_path.exists()})")
                logger.info(f"  Transit vehicles: {vehicles_path} (exists={vehicles_path.exists()})")
            logger.info("")

            # Store network stats
            file_size_mb = self.network_path.stat().st_size / 1024 / 1024 if self.network_path.exists() else 0
            self.network_stats = {
                'num_nodes': network_metadata.get('num_nodes', 0),
                'num_links': network_metadata.get('num_links', 0),
                'file_size_mb': round(file_size_mb, 2),
                'transit_network': network_metadata.get('transit_network', False),
            }

            # If network generation fell back to road-only (e.g. no GTFS feeds
            # converted successfully), update the config so downstream steps
            # (config_manager, orchestrator) don't try to enable transit.
            if (self.config.get('matsim', {}).get('transit_network', False)
                    and not network_metadata.get('transit_network', False)):
                logger.warning(
                    "Network generation fell back to road-only. "
                    "Disabling transit_network in config for this experiment."
                )
                self.config['matsim']['transit_network'] = False

                # Without a transit network, MATSim falls back to teleported pt
                # (free, fast, no fare/waiting/transfers) which dominates car
                # during replanning. Disable every pt-mapped mode and strip pt
                # from subtourModeChoice.modes so replanning cannot insert pt.
                disabled_pt_modes = []
                for mode_name, mode_cfg in self.config.get('modes', {}).items():
                    if isinstance(mode_cfg, dict) and mode_cfg.get('matsim_mode') == 'pt' \
                            and mode_cfg.get('enabled', True):
                        mode_cfg['enabled'] = False
                        disabled_pt_modes.append(mode_name)
                if disabled_pt_modes:
                    logger.warning(
                        f"Disabled pt-mapped modes for this experiment: "
                        f"{', '.join(disabled_pt_modes)}"
                    )

                configurable = self.config['matsim'].setdefault('configurable_params', {})
                existing_modes = configurable.get('subtourModeChoice.modes', 'car,pt,walk')
                kept = [m for m in existing_modes.split(',') if m.strip() and m.strip() != 'pt']
                configurable['subtourModeChoice.modes'] = ','.join(kept)
                logger.warning(
                    f"Stripped pt from subtourModeChoice.modes: "
                    f"{existing_modes} -> {configurable['subtourModeChoice.modes']}"
                )

            return network_path

        except Exception as e:
            logger.error(f"Network setup failed: {e}")
            raise RuntimeError(f"Network setup failed: {e}")

    def _thin_transit_schedule(self) -> None:
        """
        Thin transit routes and vehicles to match the simulation scaling factor.

        pt2matsim generates one transitRoute per trip (each with 1 departure),
        so thinning operates at the route level. For a scaling_factor of 0.01,
        keeps every Nth route (N = ceil(1/scaling_factor)) within each
        transitLine, sorted by departure time to preserve temporal distribution.
        At least 1 route per transitLine always survives.

        Operates in-place on transitSchedule.xml and transitVehicles.xml in the
        experiment directory. Called after setup_network() so files are always
        present (whether freshly generated or copied from the network cache).
        """
        import math
        import xml.etree.ElementTree as ET

        # Guard: only applicable for transit network mode
        if not self.config.get('matsim', {}).get('transit_network', False):
            return

        scaling_factor = self.config['plan_generation'].get('scaling_factor', 1.0)
        if scaling_factor >= 1.0:
            logger.info("Scaling factor >= 1.0, no transit thinning needed")
            return

        schedule_path = self.experiment_dir / 'transitSchedule.xml'
        vehicles_path = self.experiment_dir / 'transitVehicles.xml'

        if not schedule_path.exists() or not vehicles_path.exists():
            logger.debug("Transit files not found in experiment dir, skipping thinning")
            return

        # Idempotency: skip if already thinned with same scaling factor
        thinning_marker = self.experiment_dir / 'transit_thinning.json'
        if thinning_marker.exists():
            try:
                with open(thinning_marker) as f:
                    marker = json.load(f)
                if marker.get('scaling_factor') == scaling_factor:
                    logger.info(f"Transit schedule already thinned at scaling_factor={scaling_factor}, skipping")
                    return
            except Exception:
                pass  # Corrupt marker, proceed with thinning

        logger.info("=" * 60)
        logger.info("THINNING TRANSIT SCHEDULE")
        logger.info("=" * 60)

        N = math.ceil(1.0 / scaling_factor)
        logger.info(f"  Scaling factor: {scaling_factor} -> keeping every {N}th departure")

        def _time_to_seconds(t: str) -> int:
            """Parse HH:MM:SS to seconds (supports hours >= 24 for overnight service)."""
            parts = t.split(':')
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

        # Phase 1: Thin transitSchedule.xml
        # pt2matsim creates one <transitRoute> per trip (1 departure each).
        # Thin at the route level: keep every Nth route per transitLine,
        # sorted by departure time, so temporal coverage is preserved.
        sched_tree = ET.parse(schedule_path)
        sched_root = sched_tree.getroot()

        surviving_vehicle_ids = set()
        total_original = 0
        total_kept = 0

        for line_elem in sched_root.findall('transitLine'):
            all_routes = line_elem.findall('transitRoute')
            if not all_routes:
                continue

            def _route_dep_time(route_elem):
                """Extract earliest departure time from a route (seconds)."""
                deps_elem = route_elem.find('departures')
                if deps_elem is None:
                    return 0
                deps = deps_elem.findall('departure')
                if not deps:
                    return 0
                return min(
                    _time_to_seconds(d.get('departureTime', '0:0:0'))
                    for d in deps
                )

            # Sort routes by departure time to preserve temporal distribution
            all_routes.sort(key=_route_dep_time)

            # Keep indices 0, N, 2N, ... (always keep at least 1 route per line)
            kept_indices = set(range(0, len(all_routes), N))

            for i, route_elem in enumerate(all_routes):
                if i in kept_indices:
                    # Collect vehicle IDs from surviving routes
                    deps_elem = route_elem.find('departures')
                    if deps_elem is not None:
                        for dep in deps_elem.findall('departure'):
                            vid = dep.get('vehicleRefId')
                            if vid:
                                surviving_vehicle_ids.add(vid)
                else:
                    line_elem.remove(route_elem)

            total_original += len(all_routes)
            total_kept += len(kept_indices)

        # Phase 1b: Remove orphaned stopFacilities
        # RAPTOR router crashes (NPE in calculateRouteStopTransfers) if a
        # stopFacility exists but no surviving route references it.
        referenced_stops = set()
        for stop in sched_root.iter('stop'):
            rid = stop.get('refId')
            if rid:
                referenced_stops.add(rid)

        transit_stops_elem = sched_root.find('transitStops')
        if transit_stops_elem is not None:
            orig_fac_count = len(transit_stops_elem.findall('stopFacility'))
            for fac in list(transit_stops_elem.findall('stopFacility')):
                if fac.get('id') not in referenced_stops:
                    transit_stops_elem.remove(fac)
            kept_fac_count = len(transit_stops_elem.findall('stopFacility'))
            logger.info(f"  StopFacilities: {orig_fac_count:,} -> {kept_fac_count:,} (removed {orig_fac_count - kept_fac_count:,} orphaned)")

        # Phase 1c: Remove orphaned minimalTransferTimes relations
        # RAPTOR crashes with "fromStop is null" if a transfer relation
        # references a stop facility that was removed.
        mtt_elem = sched_root.find('minimalTransferTimes')
        if mtt_elem is not None:
            orig_mtt = len(mtt_elem.findall('relation'))
            for rel in list(mtt_elem.findall('relation')):
                if rel.get('fromStop') not in referenced_stops or rel.get('toStop') not in referenced_stops:
                    mtt_elem.remove(rel)
            kept_mtt = len(mtt_elem.findall('relation'))
            logger.info(f"  TransferTimes: {orig_mtt:,} -> {kept_mtt:,} (removed {orig_mtt - kept_mtt:,} orphaned)")

        # Write back schedule with DOCTYPE
        _DOCTYPE = (
            '<!DOCTYPE transitSchedule SYSTEM '
            '"http://www.matsim.org/files/dtd/transitSchedule_v2.dtd">'
        )
        ET.indent(sched_tree, space='    ')
        xml_bytes = ET.tostring(sched_root, encoding='UTF-8', xml_declaration=True)
        xml_str = xml_bytes.decode('UTF-8')
        xml_str = xml_str.replace("?>\n", f"?>\n{_DOCTYPE}\n", 1)
        schedule_path.write_text(xml_str, encoding='UTF-8')

        # Phase 2: Thin transitVehicles.xml
        ns = '{http://www.matsim.org/files/dtd}'
        veh_tree = ET.parse(vehicles_path)
        veh_root = veh_tree.getroot()

        all_vehicles = list(veh_root.iter(f'{ns}vehicle'))
        total_orig_veh = len(all_vehicles)

        for v in all_vehicles:
            if v.get('id') not in surviving_vehicle_ids:
                veh_root.remove(v)

        total_kept_veh = len(list(veh_root.iter(f'{ns}vehicle')))

        ET.indent(veh_tree, space='    ')
        veh_tree.write(str(vehicles_path), xml_declaration=True, encoding='UTF-8')

        # Write idempotency marker
        marker_data = {
            'scaling_factor': scaling_factor,
            'stride': N,
            'original_routes': total_original,
            'kept_routes': total_kept,
            'original_stop_facilities': orig_fac_count if transit_stops_elem is not None else 0,
            'kept_stop_facilities': kept_fac_count if transit_stops_elem is not None else 0,
            'original_vehicles': total_orig_veh,
            'kept_vehicles': total_kept_veh,
        }
        with open(thinning_marker, 'w') as f:
            json.dump(marker_data, f, indent=2)

        # Log results
        route_pct = (total_kept / total_original * 100) if total_original > 0 else 0
        veh_pct = (total_kept_veh / total_orig_veh * 100) if total_orig_veh > 0 else 0
        logger.info(f"  Routes: {total_original:,} -> {total_kept:,} ({route_pct:.1f}% retained)")
        logger.info(f"  Vehicles:   {total_orig_veh:,} -> {total_kept_veh:,} ({veh_pct:.1f}% retained)")
        logger.info("")

    def generate_counts(self) -> Optional[Path]:
        """
        Generate counts.xml for MATSim validation against ground truth traffic counts.

        Uses FHA/TMAS data as the primary source (loaded from DB via FHACountsManager),
        with optional custom CSV counts blended in. Controlled by the 'counts' config section.

        Returns:
            Path to counts.xml file, or None if generation was skipped
        """
        logger.info("="*60)
        logger.info("STEP 3b: GENERATING COUNTS FILE")
        logger.info("="*60)

        # Check if counts generation is enabled
        counts_config = self.config.get('counts', {})
        counts_enabled = counts_config.get('enabled', True)

        if not counts_enabled:
            logger.info("Counts generation disabled in config (counts.enabled = false)")
            logger.info("")
            return None

        from data_sources.fha_counts_manager import FHACountsManager, FHASchemaError

        try:
            self.counts_path = self.experiment_dir / 'counts.xml'
            rebuild = counts_config.get('rebuild', True)

            # Check if counts.xml already exists and rebuild is not requested
            if not rebuild and self.counts_path.exists():
                logger.info(f"Found existing counts.xml in experiment directory")
                logger.info(f"  File: {self.counts_path}")
                file_size_kb = self.counts_path.stat().st_size / 1024
                logger.info(f"  Size: {file_size_kb:.2f} KB")
                logger.info("Reusing existing counts.xml (counts.rebuild = false)")
                logger.info("")
                self.counts_stats['generated'] = True
                return self.counts_path

            if rebuild and self.counts_path.exists():
                logger.info("Rebuilding counts.xml (counts.rebuild = true)")

            # Setup FHA counts data (ETL from zip â†’ DB) â€” skip when weight is 0
            fha_weight = counts_config.get('fha', {}).get('weight', 0.5)
            db_manager = None
            try:
                from models.models import initialize_tables
                data_dir = self.config['data']['data_dir']
                db_manager = initialize_tables(data_dir)

                if fha_weight > 0:
                    fha_manager = FHACountsManager(self.config, db_manager)
                    fha_success = fha_manager.setup(rebuild=rebuild)
                    if not fha_success:
                        logger.warning("FHA counts setup failed â€” continuing without FHA data")
                else:
                    logger.info("FHA counts setup skipped (weight=0)")
            except FHASchemaError:
                # An old-shape DB upgrades itself in _check_schema; reaching
                # here means that rebuild failed. Fatal and user-actionable:
                # do NOT continue without FHA data. Re-raise to stop the run.
                raise
            except Exception as e:
                logger.warning(f"FHA counts setup error: {e}")

            # Initialize counts generator with db_manager
            counts_generator = CountsGenerator(self.config, db_manager=db_manager)

            # Generate counts.xml
            counts_path, counts_metadata = counts_generator.generate_counts_xml(
                network_path=self.network_path,
                output_path=self.counts_path,
            )

            if counts_path is None:
                logger.warning("Could not generate counts.xml - continuing without counts validation")
                logger.info("")
                return None

            # Store counts stats
            self.counts_stats = {
                'num_devices_matched': counts_metadata.get('num_devices_matched', 0),
                'num_count_locations': counts_metadata.get('num_count_locations', 0),
                'num_fha_stations_matched': counts_metadata.get('num_fha_stations_matched', 0),
                'num_directional_counts': counts_metadata.get('num_directional_counts', 0),
                'num_custom_bidirectional': counts_metadata.get('num_custom_bidirectional', 0),
                'generated': True,
            }

            logger.info("")
            return counts_path

        except FHASchemaError:
            # The automatic per-direction rebuild failed. Fatal and
            # user-actionable: stop the run so the user migrates the DB.
            raise
        except Exception as e:
            logger.warning(f"Counts generation failed: {e}")
            logger.warning("Continuing without counts validation")
            logger.info("")
            return None

    def generate_plans(self) -> Path:
        """
        Generate activity plans (work + non-work combined)

        Returns:
            Path to plans.xml file

        Raises:
            RuntimeError: If plan generation fails
        """
        logger.info("="*60)
        logger.info("STEP 4: GENERATING ACTIVITY PLANS (WORK + NON-WORK)")
        logger.info("="*60)

        try:
            # Set plans path
            self.plans_path = self.experiment_dir / 'plans.xml'

            # Reuse demand from a previous experiment when the inputs that
            # determine it have not changed. skip_if_exists only ever looks
            # inside *this* experiment folder, and every run creates a fresh
            # timestamped one, so it never fires across runs — this does.
            if self._try_reuse_cached_demand():
                return self.plans_path

            # Check if plans.xml already exists and skip_if_exists is enabled
            skip_if_exists = self.config['plan_generation'].get('skip_if_exists', False)

            if self.plans_path.exists() and skip_if_exists:
                logger.info(f"Found existing plans.xml in experiment directory")
                logger.info(f"  File: {self.plans_path}")

                # Validate the existing plans file
                if self._validate_plans_file(self.plans_path):
                    file_size_mb = self.plans_path.stat().st_size / 1024 / 1024
                    logger.info(f"  Size: {file_size_mb:.2f} MB")
                    logger.info("Skipping plan generation (skip_if_exists=true)")
                    logger.info("  (set skip_if_exists=false to regenerate)")
                    logger.info("")
                    return self.plans_path
                else:
                    logger.warning("Existing plans.xml is invalid, regenerating...")
            elif self.plans_path.exists() and not skip_if_exists:
                logger.info(f"Found existing plans.xml, but skip_if_exists=false")
                logger.info(f"Regenerating plans...")
                logger.info("")

            # Get data directory (resolve relative paths)
            data_dir = self.config['data']['data_dir']
            if not Path(data_dir).is_absolute():
                data_dir = (self.config_path.parent / data_dir).resolve()

            # Update config with absolute data_dir
            self.config['data']['data_dir'] = str(data_dir)

            # Ensure home and work locations are available for all configured counties.
            # If the config specifies counties that are not yet in the database,
            # this will automatically run the LODES + Census ETL to populate them.
            from models.home_locs_v2 import ensure_home_locations
            from models.work_locs_v2 import ensure_work_locations
            from models.poi_manager import ensure_pois
            ensure_home_locations(self.config)
            ensure_work_locations(self.config)
            ensure_pois(self.config)

            # Get target plans from config
            target_plans = self.config['plan_generation'].get('target_plans', 1000)
            scaling_factor = self.config['plan_generation'].get('scaling_factor', 0.1)

            # Format target_plans for logging
            if isinstance(target_plans, str):
                logger.info(f"Target total plans: {target_plans}")
            else:
                logger.info(f"Target total plans: {target_plans:,}")
            logger.info(f"Scaling factor: {scaling_factor}")
            logger.info("")

            # ==================================================================
            # GENERATE WORK PLANS
            # ==================================================================
            logger.info("-" * 60)
            logger.info("GENERATING WORK PLANS")
            logger.info("-" * 60)

            # Track plan generation start time
            self.runtime['plans_start'] = datetime.now()

            # Save updated config to temp file for PlanGenerator
            import json
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(self.config, f, indent=2)
                temp_config_path = f.name

            try:
                self.plan_generator = PlanGenerator(
                    config_path=temp_config_path,
                    experiment_dir=str(self.experiment_dir)
                )
            finally:
                # Clean up temp config file
                Path(temp_config_path).unlink()

            # Generate work plans (use target_plans for work)
            work_plans, work_stats = self.plan_generator.generate_plans(target_plans=target_plans)
            self.plan_stats['work'] = work_stats
            logger.info(f"Generated {len(work_plans):,} work plans")
            logger.info("")

            # ==================================================================
            # GENERATE NON-WORK PLANS
            # ==================================================================
            all_nonwork_plans = []
            nonwork_purposes_config = self.config.get('nonwork_purposes', {})

            # Check if any non-work purposes are enabled
            enabled_purposes = [
                purpose for purpose, cfg in nonwork_purposes_config.items()
                if isinstance(cfg, dict) and cfg.get('enabled', False)
            ]

            # Load shared data ONCE if any non-work purposes are enabled
            shared_data = None
            if enabled_purposes:
                shared_data = load_shared_nonwork_data(self.config)

                # Reuse GTFS stop data from work plan generator (avoids re-downloading)
                gtfs_stop_data = self.plan_generator._serialize_gtfs_stop_data()
                if gtfs_stop_data:
                    shared_data['gtfs_stop_data'] = gtfs_stop_data
                    logger.info(f"Sharing GTFS stop data with non-work generators")

                # Collect population stats from home_locs_dict
                home_locs_dict = shared_data.get('home_locs_dict', {})
                total_employees = sum(d.get('n_employees', 0) for d in home_locs_dict.values())
                total_non_employees = sum(d.get('non_employees', 0) for d in home_locs_dict.values())
                self.population_stats = {
                    'total_population': total_employees + total_non_employees,
                    'total_employees': total_employees,
                    'total_non_employees': total_non_employees,
                }
                logger.info(f"Population stats: {self.population_stats['total_population']:,} total "
                           f"({total_employees:,} employees, {total_non_employees:,} non-employees)")

                # Collect data quality stats
                home_blocks_with_coords = sum(
                    1 for d in home_locs_dict.values()
                    if d.get('lat') is not None and d.get('lon') is not None
                )
                poi_data_flat = shared_data.get('poi_data_flat', [])
                poi_data_grouped = shared_data.get('poi_data_grouped', {})
                self.data_quality_stats = {
                    'home_blocks': len(home_locs_dict),
                    'home_blocks_with_coords': home_blocks_with_coords,
                    'total_pois': len(poi_data_flat),
                    'poi_activity_types': len(poi_data_grouped),
                }

            for purpose, purpose_config in nonwork_purposes_config.items():
                # Skip non-purpose entries (like _comment, etc.)
                if not isinstance(purpose_config, dict) or 'enabled' not in purpose_config:
                    continue

                if not purpose_config.get('enabled', False):
                    logger.info(f"Skipping {purpose} (disabled in config)")
                    continue

                logger.info("-" * 60)
                logger.info(f"GENERATING {purpose.upper()} PLANS")
                logger.info("-" * 60)

                try:
                    # Initialize non-work plan generator with pre-loaded shared data
                    nonwork_generator = NonWorkPlanGenerator(
                        self.config,
                        purpose=purpose,
                        shared_data=shared_data
                    )

                    # Calculate number of plans for this purpose based on trip rate and population
                    # This is automatically derived from the OD matrix (non_employees * trip_rate * scaling)
                    # We use "all" to generate all trips from the OD matrix
                    nonwork_plans, nonwork_stats = nonwork_generator.generate_plans_list(n_plans=target_plans)

                    # Store stats by purpose (lowercase for consistency)
                    purpose_key = purpose.lower()
                    self.plan_stats[purpose_key] = nonwork_stats

                    logger.info(f"Generated {len(nonwork_plans):,} {purpose} plans")
                    all_nonwork_plans.extend(nonwork_plans)
                    logger.info("")

                except Exception as e:
                    logger.warning(f"Failed to generate {purpose} plans: {e}")
                    logger.warning(f"Continuing without {purpose} plans...")
                    logger.info("")

            # ==================================================================
            # GENERATE BOUNDARY FREIGHT
            # ==================================================================
            # A third stream beside work and non-work: trucks with at least one
            # end outside the region. Off unless freight.enabled, and a failure
            # here must not lose the passenger demand that has just been built,
            # so it is caught rather than raised — except for the cordon gate,
            # which is deliberately fatal (see models/freight/cordons.py).
            freight_plans = []
            if self.config.get('freight', {}).get('enabled', False):
                from models.freight.plans import generate_freight_plans
                try:
                    freight_plans = generate_freight_plans(
                        self.config,
                        network_path=self.network_path,
                        car_trips=len(work_plans) + len(all_nonwork_plans),
                        summary_path=self.experiment_dir / 'freight_summary.json',
                    )
                except Exception as e:
                    logger.error(f"Freight generation failed: {e}")
                    logger.error("Continuing with passenger demand only. The run "
                                 "is valid but carries no boundary freight.")
                    freight_plans = []

            # ==================================================================
            # COMBINE ALL PLANS
            # ==================================================================
            logger.info("=" * 60)
            logger.info("COMBINING WORK + NON-WORK + FREIGHT PLANS")
            logger.info("=" * 60)

            all_plans = work_plans + all_nonwork_plans + freight_plans

            logger.info(f"Plan generation summary (scaling_factor={scaling_factor}):")
            logger.info(f"  Work plans: {len(work_plans):,}")
            logger.info(f"  Non-work plans: {len(all_nonwork_plans):,}")
            if freight_plans:
                logger.info(f"  Freight plans: {len(freight_plans):,} "
                            f"({len(freight_plans) / len(all_plans) * 100:.1f}% of total)")
            logger.info(f"  Combined total (scaled): {len(all_plans):,}")
            if scaling_factor < 1.0:
                # This represents the full population these scaled plans represent
                represented_population = len(all_plans) / scaling_factor
                logger.info(f"  Represents full population of: ~{represented_population:,.0f} travelers")
            logger.info("")

            # Assign unique person IDs (simple sequential format)
            logger.info("Assigning unique person IDs...")
            for i, plan in enumerate(all_plans):
                plan.person_id = f"person_{i}"

            # ==================================================================
            # WRITE COMBINED PLANS TO XML
            # ==================================================================
            logger.info("Writing combined plans to XML...")
            self.plan_generator.write_xml(all_plans)

            if not self.plans_path.exists():
                raise RuntimeError(f"Plans file was not created: {self.plans_path}")

            file_size_mb = self.plans_path.stat().st_size / 1024 / 1024
            logger.info(f"Plans XML ready: {self.plans_path}")
            logger.info(f"  File size: {file_size_mb:.2f} MB")
            logger.info(f"  Total persons: {len(all_plans):,}")
            logger.info("")

            # Store plans file size
            self.plans_file_size_mb = round(file_size_mb, 2)

            # Track plan generation end time
            self.runtime['plans_end'] = datetime.now()

            # Save the finished demand so a later run with the same inputs can
            # skip this step entirely.
            self._store_cached_demand()

            return self.plans_path

        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            raise RuntimeError(f"Plan generation failed: {e}")

    def run_simulation(self, skip_simulation: bool = False) -> Dict:
        """
        Run MATSim simulation

        Args:
            skip_simulation: If True, skip simulation and only setup

        Returns:
            Simulation metadata dictionary

        Raises:
            RuntimeError: If simulation fails
        """
        logger.info("="*60)
        logger.info("STEP 5: RUNNING MATSIM SIMULATION")
        logger.info("="*60)

        if skip_simulation:
            logger.info("Skipping simulation (--skip-simulation flag set)")
            logger.info("")
            return {'simulation_status': 'skipped'}

        try:
            # Track MATSim start time
            self.runtime['matsim_start'] = datetime.now()

            # Initialize orchestrator with the config we loaded
            self.orchestrator = MATSimOrchestrator(config_dict=self.config)

            # Get MATSim config
            matsim_config = self.config['matsim']
            mode = matsim_config.get('mode', 'basic')
            run_simulation = matsim_config.get('run_simulation', True)
            custom_params = matsim_config.get('configurable_params', {})

            logger.info(f"MATSim mode: {mode}")
            logger.info(f"Custom parameters: {custom_params}")
            logger.info("")

            # Run simulation
            metadata = self.orchestrator.create_and_run_experiment(
                experiment_id=self.experiment_id,
                mode=mode,
                generate_network=False,  # We already have network
                plans_file=self.plans_path,
                custom_params=custom_params if custom_params else None,
                run_simulation=run_simulation
            )

            logger.info("")
            logger.info(f"Simulation status: {metadata.get('simulation_status', 'unknown')}")
            logger.info("")

            # Track MATSim end time
            self.runtime['matsim_end'] = datetime.now()

            return metadata

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise RuntimeError(f"Simulation failed: {e}")

    def run_evaluation(self) -> Optional[Dict]:
        """
        Run simulation evaluation against ground truth traffic counts

        Returns:
            Evaluation summary metrics dictionary, or None if evaluation is disabled/fails

        Raises:
            RuntimeError: If evaluation fails critically
        """
        logger.info("="*60)
        logger.info("STEP 6: EVALUATING SIMULATION")
        logger.info("="*60)

        # Check if evaluation is enabled in config
        evaluation_config = self.config.get('evaluation', {})
        run_evaluation = evaluation_config.get('run_evaluation', True)

        if not run_evaluation:
            logger.info("Evaluation disabled in config (evaluation.run_evaluation = false)")
            logger.info("")
            return None

        # Check if counts were generated (required for evaluation)
        counts_enabled = self.config.get('counts', {}).get('enabled', True)
        if not counts_enabled:
            logger.warning("Evaluation requires counts generation, but counts.enabled = false")
            logger.warning("Skipping evaluation - set counts.enabled = true to enable")
            logger.info("")
            return None

        # Check if matched_devices.csv exists (generated during counts generation)
        matched_devices_path = self.experiment_dir / 'matched_devices.csv'
        if not matched_devices_path.exists():
            logger.warning("matched_devices.csv not found in experiment directory")
            logger.warning("This file is created during counts generation (Step 3b)")
            logger.warning("Skipping evaluation - ensure counts generation succeeded")
            logger.info("")
            return None

        try:
            # Track evaluation start time
            self.runtime['eval_start'] = datetime.now()

            # Get ground truth data directory
            ground_truth_dir = evaluation_config.get('ground_truth_data_dir', 'data/evaluation')

            # Check if ground truth data exists
            data_dir = Path(ground_truth_dir)
            if not data_dir.exists():
                logger.warning(f"Ground truth data directory not found: {data_dir}")
                logger.warning("Skipping evaluation")
                logger.info("")
                return None

            # Initialize evaluator
            logger.info(f"Initializing evaluator with ground truth data from: {data_dir}")
            evaluator = SimulationEvaluator(
                experiment_dir=self.experiment_dir,
                ground_truth_data_dir=data_dir
            )

            # Get evaluation options from config
            generate_spatial_maps = evaluation_config.get('generate_spatial_maps', True)
            generate_per_device_reports = evaluation_config.get('generate_per_device_reports', False)

            # Run evaluation (will auto-detect network and linkstats files)
            logger.info("Running evaluation...")
            logger.info("")

            comparison_df, summary_metrics = evaluator.run_evaluation(
                generate_spatial_maps=generate_spatial_maps,
                generate_per_device_reports=generate_per_device_reports
            )

            # Log summary
            logger.info("")
            logger.info("="*60)
            logger.info("EVALUATION RESULTS")
            logger.info("="*60)
            logger.info(f"Devices validated: {summary_metrics['num_devices']}")
            logger.info(f"Total comparisons: {summary_metrics['num_comparisons']:,}")

            if summary_metrics['num_comparisons'] > 0:
                logger.info(f"Mean Absolute Error (MAE): {summary_metrics['mae']:.2f} vehicles")
                logger.info(f"Root Mean Square Error (RMSE): {summary_metrics['rmse']:.2f} vehicles")
                logger.info(f"GEH < 5 (hourly counts, target >=85%): "
                            f"{summary_metrics['geh_lt_5_pct']:.1f}%")
                if 'station_daily_geh_lt5_pct' in summary_metrics:
                    logger.info(f"GEH < 5 (station daily totals): "
                                f"{summary_metrics['station_daily_geh_lt5_pct']:.1f}%")
                logger.info(f"Correlation: {summary_metrics['correlation']:.3f}")
                logger.info(f"Peak-hour correlation (6-9,15-18): {summary_metrics.get('peak_hour_correlation', 0):.3f}")
                logger.info("")
                logger.info(f"Results saved to: {evaluator.evaluation_dir}")
            else:
                logger.warning("No devices could be matched to network links")
                logger.warning("This may be because:")
                logger.warning("  - The network extent doesn't overlap with ground truth device locations")
                logger.warning("  - The simulation used a smaller region than the available ground truth data")

            logger.info("="*60)
            logger.info("")

            # Track evaluation end time
            self.runtime['eval_end'] = datetime.now()

            return summary_metrics

        except FileNotFoundError as e:
            logger.warning(f"Evaluation files not found: {e}")
            logger.warning("Skipping evaluation")
            logger.info("")
            return None

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            logger.warning("Continuing without evaluation")
            logger.info("")
            return None

    def save_experiment_summary(self, metadata: Dict, evaluation_metrics: Optional[Dict] = None):
        """
        Save experiment summary to JSON file.

        Delegates to utils.experiment_summary.build_summary so the same logic
        can be invoked standalone (see utils/experiment_summary.py for the
        CLI used to refresh matsim_output on completed experiments).
        """
        self.runtime['end_time'] = datetime.now()

        summary = build_summary(
            experiment_id=self.experiment_id,
            experiment_dir=self.experiment_dir,
            config=self.config,
            plan_stats=self.plan_stats,
            population_stats=self.population_stats,
            data_quality_stats=self.data_quality_stats,
            network_stats=self.network_stats,
            plans_file_size_mb=self.plans_file_size_mb,
            runtime=self.runtime,
            metadata=metadata,
            evaluation_metrics=evaluation_metrics,
        )

        # Demand validation vs the household travel survey. This is a separate
        # question from the count validation above: counts validate how demand
        # was assigned to the network, this validates the demand itself. It
        # reads MATSim's per-trip output, so it needs no extra simulation, and
        # it returns None rather than raising when survey or trip data is
        # missing — a run must still complete without it.
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent / 'scripts'))
            from validate_demand import compute_demand_validation, to_summary_section
            demand_results = compute_demand_validation(self.experiment_dir)
            if demand_results is not None:
                summary['demand_validation'] = to_summary_section(demand_results)
                logger.info("  Demand validation vs survey added to summary")
        except Exception as e:
            logger.warning(f"  Demand validation skipped: {e}")

        summary_path = self.experiment_dir / 'experiment_summary.json'
        write_summary(summary, summary_path)

        # Also save a copy of the config that produced this run.
        config_copy_path = self.experiment_dir / 'config_used.json'
        with open(config_copy_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Experiment summary saved to: {summary_path}")

        # Human-readable report for this run. Reads the summary just written,
        # so it must come after write_summary().
        self._generate_report()

    def _generate_report(self):
        """Write report.md / report.html summarising this experiment.

        Never fatal: a run that simulated and evaluated fine should not be
        reported as failed because a report could not be rendered. The PDF step
        is skipped automatically where no browser is installed (e.g. the
        server) — see scripts/experiment_report.py --html-to-pdf.
        """
        try:
            sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
            from experiment_report import (build_markdown, load_run,
                                           markdown_to_html)

            run = load_run(self.experiment_dir)
            md_text = build_markdown(run, baseline=None, embed_dir=self.experiment_dir)
            (self.experiment_dir / 'report.md').write_text(md_text, encoding='utf-8')
            html = markdown_to_html(md_text, self.experiment_dir.name,
                                    base_dir=self.experiment_dir)
            (self.experiment_dir / 'report.html').write_text(html, encoding='utf-8')
            logger.info(f"Experiment report saved to: {self.experiment_dir / 'report.html'}")
            logger.info("  For a PDF: python scripts/experiment_report.py "
                        "--html-to-pdf <that file>  (needs a browser)")
        except Exception as e:
            logger.warning(f"Could not generate the experiment report: {e}")

    def run(self, skip_simulation: bool = False) -> Dict:
        """
        Run complete experiment pipeline

        Args:
            skip_simulation: If True, generate plans but skip simulation

        Returns:
            Experiment metadata dictionary

        Raises:
            Exception: If any step fails
        """
        try:
            # Track experiment start time
            self.runtime['start_time'] = datetime.now()

            # Step 1: Validate configuration
            self.validate_config()

            # Step 1b: Auto-detect coordinate system from counties
            self.detect_coordinate_system()

            # Step 2: Setup experiment directory
            self.setup_experiment_directory()

            # Step 2b: Generate the MATSim config.xml right away. This
            # catches typos and stale keys in configurable_params in ms,
            # before the expensive network + counts + plans steps. The
            # generated XML references network.xml / plans.xml / counts.xml
            # by relative filename only - those files don't need to exist
            # yet. Step 5 will rewrite the same file with identical content
            # right before launching MATSim.
            self.setup_matsim_config()

            # Step 3: Setup network
            self.setup_network()

            # Step 3c: Thin transit schedule to match scaling factor
            # self._thin_transit_schedule()

            # Step 3b: Generate counts.xml (for MATSim validation)
            self.generate_counts()

            # Step 4: Generate plans
            self.generate_plans()

            # Step 5: Run simulation
            simulation_metadata = self.run_simulation(skip_simulation=skip_simulation)

            # Save config_used.json immediately after simulation (before evaluation)
            # This ensures the evaluator can load scaling factors from config
            config_copy_path = self.experiment_dir / 'config_used.json'
            with open(config_copy_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.debug(f"Saved config_used.json to: {config_copy_path}")

            # Step 6: Run evaluation (if enabled in config)
            # Note: Evaluation can run independently of simulation
            # It will check for existing output files and skip if not available
            evaluation_metrics = None
            evaluation_config = self.config.get('evaluation', {})
            run_evaluation = evaluation_config.get('run_evaluation', True)

            if run_evaluation:
                evaluation_metrics = self.run_evaluation()

            # Save experiment summary
            self.save_experiment_summary(simulation_metadata, evaluation_metrics)

            # Track experiment in comparison CSV
            try:
                tracker = ExperimentTracker()
                tracker_row = tracker.record_experiment(self.experiment_dir)
                logger.info(f"Experiment tracked in: {tracker.csv_path}")
                if tracker_row.get('suggestion'):
                    logger.info(f"Suggestion: {tracker_row['suggestion']}")
            except Exception as e:
                logger.warning(f"Failed to track experiment in CSV: {e}")

            # Final summary
            logger.info("="*60)
            logger.info("EXPERIMENT COMPLETE")
            logger.info("="*60)
            logger.info(f"Experiment ID: {self.experiment_id}")
            logger.info(f"Experiment directory: {self.experiment_dir}")
            logger.info(f"Status: {simulation_metadata.get('simulation_status', 'completed')}")
            logger.info("")
            logger.info("Results available at:")
            logger.info(f"  Network: {self.network_path.name}")
            logger.info(f"  Plans: {self.plans_path.name}")
            if simulation_metadata.get('simulation_status') == 'completed':
                logger.info(f"  Output: output/")
            if evaluation_metrics and evaluation_metrics.get('num_comparisons', 0) > 0:
                logger.info(f"  Evaluation: evaluation/")
            logger.info("="*60)

            return simulation_metadata

        except Exception as e:
            logger.error("="*60)
            logger.error("EXPERIMENT FAILED")
            logger.error("="*60)
            logger.error(f"Error: {e}")
            logger.error("")
            logger.error(f"Experiment ID: {self.experiment_id}")
            if self.experiment_dir:
                logger.error(f"Partial results may be in: {self.experiment_dir}")
            logger.error("="*60)
            raise


def main():
    """Main entry point for command-line execution"""
    parser = argparse.ArgumentParser(
        description='Run complete MATSim experiment (plan generation + simulation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiment with default config
  python run_experiment.py --config config/config.json

  # Run with custom experiment ID
  python run_experiment.py --config config/config.json --experiment-id my_test_001

  # Generate plans only, skip simulation
  python run_experiment.py --config config/config.json --skip-simulation

  # Run experiment with different config file
  python run_experiment.py --config config/config_large.json
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration JSON file'
    )

    parser.add_argument(
        '--experiment-id',
        type=str,
        default=None,
        help='Custom experiment ID (default: auto-generated timestamp)'
    )

    parser.add_argument(
        '--skip-simulation',
        action='store_true',
        help='Generate plans but skip MATSim simulation'
    )

    parser.add_argument(
        '--experiments-root',
        type=str,
        default=None,
        help='Root dir for experiment output (default: ./experiments, '
             'or TAREEK_EXPERIMENTS_ROOT env var)'
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # Run experiment
    try:
        runner = ExperimentRunner(
            config_path=config_path,
            experiment_id=args.experiment_id,
            experiments_root=Path(args.experiments_root) if args.experiments_root else None
        )

        runner.run(skip_simulation=args.skip_simulation)

        sys.exit(0)

    except ConfigValidationError as e:
        # Config validation errors already logged
        sys.exit(1)

    except Exception as e:
        # Other errors already logged
        sys.exit(1)


if __name__ == '__main__':
    main()
