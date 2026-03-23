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

    def __init__(self, config_path: Path, experiment_id: Optional[str] = None):
        """
        Initialize experiment runner

        Args:
            config_path: Path to configuration JSON file
            experiment_id: Optional custom experiment ID
        """
        self.config_path = Path(config_path)
        self.experiment_id = experiment_id or self._generate_experiment_id()
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
        from models.models import County
        from utils.duckdb_manager import DBManager

        county_geoids = self.config['region']['counties']
        data_dir = self.config['data']['data_dir']
        db_manager = DBManager(data_dir)

        with db_manager.Session() as session:
            counties = session.query(County).filter(
                County.geoid.in_(county_geoids)
            ).all()

            if not counties:
                raise RuntimeError(
                    f"No counties found in database for GEOIDs: {county_geoids}. "
                    "Run notebooks/0.setup_global_data.ipynb first."
                )

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

    def setup_experiment_directory(self):
        """Create experiment directory structure"""
        logger.info("="*60)
        logger.info("STEP 2: SETTING UP EXPERIMENT DIRECTORY")
        logger.info("="*60)

        # Create experiment directory
        experiments_root = project_root / 'experiments'
        self.experiment_dir = experiments_root / self.experiment_id

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

            # Setup FHA counts data (ETL from zip → DB) — skip when weight is 0
            fha_weight = counts_config.get('fha', {}).get('weight', 0.5)
            db_manager = None
            try:
                from models.models import initialize_tables
                data_dir = self.config['data']['data_dir']
                db_manager = initialize_tables(data_dir)

                if fha_weight > 0:
                    from data_sources.fha_counts_manager import FHACountsManager
                    fha_manager = FHACountsManager(self.config, db_manager)
                    fha_success = fha_manager.setup()
                    if not fha_success:
                        logger.warning("FHA counts setup failed — continuing without FHA data")
                else:
                    logger.info("FHA counts setup skipped (weight=0)")
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
                'num_bidirectional': counts_metadata.get('num_bidirectional', 0),
                'generated': True,
            }

            logger.info("")
            return counts_path

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
            from data_sources.survey_manager import ensure_surveys
            ensure_home_locations(self.config)
            ensure_work_locations(self.config)
            ensure_pois(self.config)
            ensure_surveys(self.config)

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
            # COMBINE ALL PLANS
            # ==================================================================
            logger.info("=" * 60)
            logger.info("COMBINING WORK + NON-WORK PLANS")
            logger.info("=" * 60)

            all_plans = work_plans + all_nonwork_plans

            logger.info(f"Plan generation summary (scaling_factor={scaling_factor}):")
            logger.info(f"  Work plans: {len(work_plans):,}")
            logger.info(f"  Non-work plans: {len(all_nonwork_plans):,}")
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
                logger.info(f"Mean GEH: {summary_metrics['mean_geh']:.2f}")
                logger.info(f"GEH < 5 (good matches): {summary_metrics['geh_lt_5_pct']:.1f}%")
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

    def _calculate_runtime_minutes(self, start_key: str, end_key: str) -> Optional[float]:
        """Calculate runtime in minutes between two timestamp keys"""
        start = self.runtime.get(start_key)
        end = self.runtime.get(end_key)
        if start and end:
            delta = end - start
            return round(delta.total_seconds() / 60, 2)
        return None

    def _extract_matsim_output_stats(self) -> Dict:
        """
        Extract statistics from MATSim output files.

        Parses various CSV/TXT files from the MATSim output directory to gather
        simulation performance metrics like scores, travel distances, and traffic patterns.

        Returns:
            Dictionary with extracted statistics, or empty dict if output not available
        """
        import csv

        stats = {}
        output_dir = self.experiment_dir / 'output'

        if not output_dir.exists():
            logger.debug("MATSim output directory not found, skipping output stats extraction")
            return stats

        try:
            # 0. Count persons and trips from output CSV files
            import gzip

            persons_path = output_dir / 'output_persons.csv.gz'
            if persons_path.exists():
                with gzip.open(persons_path, 'rt') as f:
                    # Count lines minus header
                    line_count = sum(1 for _ in f) - 1
                    stats['output_persons_count'] = line_count

            trips_path = output_dir / 'output_trips.csv.gz'
            if trips_path.exists():
                with gzip.open(trips_path, 'rt') as f:
                    # Count lines minus header
                    line_count = sum(1 for _ in f) - 1
                    stats['output_trips_count'] = line_count

            legs_path = output_dir / 'output_legs.csv.gz'
            if legs_path.exists():
                with gzip.open(legs_path, 'rt') as f:
                    # Count lines minus header
                    line_count = sum(1 for _ in f) - 1
                    stats['output_legs_count'] = line_count

            # Count selected (executed) plans from output_plans.xml.gz
            # Each person has multiple plans but only one is selected="yes" (the executed plan)
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

            # 1. Extract scorestats (agent utility scores across iterations)
            scorestats_path = output_dir / 'scorestats.csv'
            if scorestats_path.exists():
                with open(scorestats_path, 'r') as f:
                    reader = csv.DictReader(f, delimiter=';')
                    rows = list(reader)
                    if rows:
                        # Get final iteration stats
                        final_row = rows[-1]
                        stats['score_final_executed'] = round(float(final_row.get('avg_executed', 0)), 2)
                        stats['score_final_best'] = round(float(final_row.get('avg_best', 0)), 2)
                        stats['score_final_worst'] = round(float(final_row.get('avg_worst', 0)), 2)
                        stats['score_final_average'] = round(float(final_row.get('avg_average', 0)), 2)

                        # Calculate score improvement from first to last iteration
                        if len(rows) > 1:
                            first_row = rows[0]
                            score_improvement = float(final_row.get('avg_executed', 0)) - float(first_row.get('avg_executed', 0))
                            stats['score_improvement'] = round(score_improvement, 2)

            # 2. Extract travel distance stats
            traveldist_path = output_dir / 'traveldistancestats.csv'
            if traveldist_path.exists():
                with open(traveldist_path, 'r') as f:
                    reader = csv.DictReader(f, delimiter=';')
                    rows = list(reader)
                    if rows:
                        final_row = rows[-1]
                        # Convert meters to km
                        avg_leg_dist = float(final_row.get('avg. Average Leg distance', 0))
                        avg_trip_dist = float(final_row.get('avg. Average Trip distance', 0))
                        stats['avg_leg_distance_km'] = round(avg_leg_dist / 1000, 2)
                        stats['avg_trip_distance_km'] = round(avg_trip_dist / 1000, 2)

            # 3. Extract mode stats (person-km traveled)
            pkm_path = output_dir / 'pkm_modestats.csv'
            if pkm_path.exists():
                with open(pkm_path, 'r') as f:
                    reader = csv.DictReader(f, delimiter=';')
                    rows = list(reader)
                    if rows:
                        final_row = rows[-1]
                        # Sum all mode PKM (person-kilometers)
                        total_pkm = 0
                        for key, value in final_row.items():
                            if key != 'Iteration' and value:
                                try:
                                    total_pkm += float(value)
                                except ValueError:
                                    pass
                        stats['total_person_km'] = int(total_pkm)

            # 4. Extract person-hours traveled
            ph_path = output_dir / 'ph_modestats.csv'
            if ph_path.exists():
                with open(ph_path, 'r') as f:
                    reader = csv.DictReader(f, delimiter=';')
                    rows = list(reader)
                    if rows:
                        final_row = rows[-1]
                        # Sum travel time (in hours)
                        total_travel_hours = 0
                        for key, value in final_row.items():
                            if key != 'Iteration' and 'travel' in key.lower() and value:
                                try:
                                    total_travel_hours += float(value)
                                except ValueError:
                                    pass
                        stats['total_person_hours_traveling'] = int(total_travel_hours)

            # 5. Find final iteration and extract leg histogram stats
            iters_dir = output_dir / 'ITERS'
            if iters_dir.exists():
                # Find the highest iteration number
                iter_dirs = [d for d in iters_dir.iterdir() if d.is_dir() and d.name.startswith('it.')]
                if iter_dirs:
                    final_iter = max(iter_dirs, key=lambda x: int(x.name.split('.')[1]))
                    final_iter_num = int(final_iter.name.split('.')[1])
                    stats['final_iteration'] = final_iter_num

                    # Extract leg durations
                    legdur_path = final_iter / f'{final_iter_num}.legdurations.txt'
                    if legdur_path.exists():
                        with open(legdur_path, 'r') as f:
                            content = f.read()
                            # Find the average leg duration line
                            for line in content.split('\n'):
                                if 'average leg duration' in line.lower():
                                    # Parse: "average leg duration: 586.0481047872958 seconds = 00:09:46"
                                    import re
                                    match = re.search(r'(\d+\.?\d*)\s*seconds', line)
                                    if match:
                                        avg_leg_duration_sec = float(match.group(1))
                                        stats['avg_leg_duration_min'] = round(avg_leg_duration_sec / 60, 2)
                                    break

                    # Extract leg histogram for peak hour analysis
                    leghist_path = final_iter / f'{final_iter_num}.legHistogram.txt'
                    if leghist_path.exists():
                        with open(leghist_path, 'r') as f:
                            reader = csv.DictReader(f, delimiter='\t')
                            rows = list(reader)

                            # Find peak departures and stuck agents
                            max_departures = 0
                            max_en_route = 0
                            total_stuck = 0
                            peak_hour = None

                            for row in rows:
                                try:
                                    departures = int(row.get('departures_all', 0))
                                    en_route = int(row.get('en-route_all', 0))
                                    stuck = int(row.get('stuck_all', 0))

                                    if departures > max_departures:
                                        max_departures = departures
                                        peak_hour = row.get('time', '')

                                    if en_route > max_en_route:
                                        max_en_route = en_route

                                    total_stuck += stuck
                                except (ValueError, TypeError):
                                    pass

                            stats['peak_departures_per_5min'] = max_departures
                            stats['peak_hour'] = peak_hour
                            stats['max_en_route'] = max_en_route
                            stats['total_stuck_agents'] = total_stuck

        except Exception as e:
            logger.warning(f"Error extracting MATSim output stats: {e}")

        return stats

    def save_experiment_summary(self, metadata: Dict, evaluation_metrics: Optional[Dict] = None):
        """
        Save experiment summary to JSON file

        This is the single source of truth for experiment results.
        Contains all metrics needed by ExperimentTracker.

        Args:
            metadata: Experiment metadata to save
            evaluation_metrics: Optional evaluation metrics to include
        """
        summary_path = self.experiment_dir / 'experiment_summary.json'

        # Mark end time
        self.runtime['end_time'] = datetime.now()

        # Build plan stats with counts per purpose
        def get_plan_count(stats_dict):
            return stats_dict.get('plans_generated', 0) if stats_dict else 0

        def get_stat(stats_dict, key, default=0):
            return stats_dict.get(key, default) if stats_dict else default

        # Calculate totals across all purposes
        all_stats = [self.plan_stats.get(k, {}) for k in self.plan_stats.keys()]
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

        # Calculate overall success rate
        total_generated = sum(get_plan_count(s) for s in all_stats)
        total_requested = total_generated + total_failed
        overall_success_rate = round((total_generated / total_requested * 100), 2) if total_requested > 0 else 100.0

        # Get scaling factor from config
        scaling_factor = self.config.get('plan_generation', {}).get('scaling_factor', 0.1)

        # Get MATSim parameters
        matsim_params = self.config.get('matsim', {}).get('configurable_params', {})

        # Calculate unscaled trips from stats
        unscaled_work = get_stat(self.plan_stats.get('work'), 'unscaled_trips', 0)
        # Get nonwork purposes from config (dynamic)
        nonwork_purposes_config = self.config.get('nonwork_purposes', {})
        nonwork_purpose_keys = [
            p.lower() for p, cfg in nonwork_purposes_config.items()
            if isinstance(cfg, dict) and cfg.get('enabled', False)
        ]

        # Sum unscaled trips from all nonwork purposes
        unscaled_nonwork = sum(
            get_stat(self.plan_stats.get(p), 'unscaled_trips', 0)
            for p in nonwork_purpose_keys
        )

        # Build plans dict dynamically from config
        plans_dict = {
            'work': get_plan_count(self.plan_stats.get('work')),
        }
        # Add each enabled nonwork purpose from config
        for purpose_key in nonwork_purpose_keys:
            plans_dict[purpose_key] = get_plan_count(self.plan_stats.get(purpose_key))

        # Add aggregate stats
        plans_dict['total'] = total_generated
        plans_dict['success_rate'] = overall_success_rate
        plans_dict['chain_retries'] = total_chain_retries
        plans_dict['chain_retries_too_short'] = total_chain_retries_too_short
        plans_dict['chain_retries_bad_structure'] = total_chain_retries_bad_structure
        plans_dict['chain_retries_missing_purpose'] = total_chain_retries_missing_purpose
        plans_dict['chain_retries_missing_work'] = total_chain_retries_missing_work
        plans_dict['chain_retries_has_work'] = total_chain_retries_has_work
        plans_dict['chain_retries_too_many_work'] = total_chain_retries_too_many_work
        plans_dict['chain_attempts'] = total_chain_attempts
        plans_dict['poi_retries'] = total_poi_retries
        plans_dict['time_retries'] = total_time_retries

        summary = {
            'experiment_id': self.experiment_id,
            'created_at': datetime.now().isoformat(),

            # Population base - who are we simulating?
            'population': {
                '_comment': 'Population counts from census home location data (home_locs_dict). '
                           'Summed from all census blocks within configured counties.',
                'total_population': self.population_stats.get('total_population', 0),
                'total_population_comment': 'Sum of total_employees + total_non_employees across all home blocks',
                'total_employees': self.population_stats.get('total_employees', 0),
                'total_employees_comment': 'Sum of n_employees field from each home block (workers who commute)',
                'total_non_employees': self.population_stats.get('total_non_employees', 0),
                'total_non_employees_comment': 'Sum of non_employees field from each home block (non-workers: retirees, students, etc.)',
            },

            # Unscaled trips - raw trip generation before scaling
            'unscaled_trips': {
                '_comment': 'Total trips BEFORE applying scaling_factor. Represents full population trip demand.',
                'work': unscaled_work,
                'work_comment': 'Unscaled work trips = total_employees (one work trip per employee)',
                'nonwork': unscaled_nonwork,
                'nonwork_comment': 'Sum of unscaled trips across all enabled nonwork purposes. '
                                  'Each purpose: non_employees * trip_rate from survey data',
            },

            # Plan generation metrics (dynamic based on config)
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

            # Parameters used
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
            },

            # Data quality metrics
            'data_quality': {
                '_comment': 'Metrics about input data quality and coverage',
                'home_blocks': self.data_quality_stats.get('home_blocks', 0),
                'home_blocks_comment': 'Number of census blocks in home_locs_dict (filtered by configured counties)',
                'home_blocks_with_coords': self.data_quality_stats.get('home_blocks_with_coords', 0),
                'home_blocks_with_coords_comment': 'Census blocks that have valid lat/lon coordinates (should equal home_blocks)',
                'total_pois': self.data_quality_stats.get('total_pois', 0),
                'total_pois_comment': 'Total POIs loaded from database and filtered to county bounding box + 5km buffer',
                'poi_activity_types': self.data_quality_stats.get('poi_activity_types', 0),
                'poi_activity_types_comment': 'Number of distinct activity types in POI data (e.g., Shopping, School, Dining)',
            },

            # Network metrics
            'network': {
                '_comment': 'Road network statistics from network.xml',
                'num_nodes': self.network_stats.get('num_nodes', 0),
                'num_nodes_comment': 'Number of network nodes (intersections). Parsed from <nodes> element in network.xml',
                'num_links': self.network_stats.get('num_links', 0),
                'num_links_comment': 'Number of network links (road segments). Parsed from <links> element in network.xml',
                'file_size_mb': self.network_stats.get('file_size_mb', 0),
                'file_size_mb_comment': 'Size of network.xml file in megabytes',
            },

            # Plans file size
            'plans_file_size_mb': self.plans_file_size_mb,
            'plans_file_size_mb_comment': 'Size of plans.xml file in megabytes. Larger files = more agents simulated.',

            # Runtime metrics
            'runtime': {
                '_comment': 'Execution time measurements in minutes',
                'total_min': self._calculate_runtime_minutes('start_time', 'end_time'),
                'total_min_comment': 'Total experiment runtime from start to finish (includes all steps)',
                'plans_min': self._calculate_runtime_minutes('plans_start', 'plans_end'),
                'plans_min_comment': 'Time spent generating activity plans (work + all nonwork purposes)',
                'matsim_min': self._calculate_runtime_minutes('matsim_start', 'matsim_end'),
                'matsim_min_comment': 'Time spent running MATSim simulation (all iterations)',
                'eval_min': self._calculate_runtime_minutes('eval_start', 'eval_end'),
                'eval_min_comment': 'Time spent on evaluation (comparing sim output to ground truth)',
            },

            # Paths (relative to experiment dir for portability)
            'paths': {
                '_comment': 'Relative paths to experiment outputs (from experiment directory)',
                'network': 'network.xml',
                'plans': 'plans.xml',
                'output': 'output/',
                'output_comment': 'MATSim simulation output directory (contains ITERS/, linkstats, etc.)',
                'evaluation': 'evaluation/',
                'evaluation_comment': 'Evaluation results directory (comparison CSVs, plots, maps)',
            },

            # Simulation metadata (from orchestrator)
            'simulation_status': metadata.get('simulation_status', 'unknown'),
            'simulation_status_comment': 'Final status: completed, failed, or skipped',
        }

        # Extract and add MATSim output statistics
        matsim_output_stats = self._extract_matsim_output_stats()
        if matsim_output_stats:
            summary['matsim_output'] = {
                '_comment': 'Statistics extracted from MATSim output files (scorestats.csv, traveldistancestats.csv, legHistogram.txt, etc.)',

                # Output counts (from gzipped files)
                'output_persons_count': matsim_output_stats.get('output_persons_count'),
                'output_persons_count_comment': 'Number of persons (agents) in final output. From output_persons.csv.gz.',
                'output_executed_plans_count': matsim_output_stats.get('output_executed_plans_count'),
                'output_executed_plans_count_comment': 'Number of executed plans (selected="yes") in final iteration. '
                                                      'This equals the number of persons - one executed plan per person.',
                'output_total_plans_count': matsim_output_stats.get('output_total_plans_count'),
                'output_total_plans_count_comment': 'Total plans in output_plans.xml.gz including non-selected alternatives. '
                                                   'MATSim keeps multiple plans per person for learning/replanning.',
                'output_trips_count': matsim_output_stats.get('output_trips_count'),
                'output_trips_count_comment': 'Total trips completed in simulation. From output_trips.csv.gz.',
                'output_legs_count': matsim_output_stats.get('output_legs_count'),
                'output_legs_count_comment': 'Total legs (mode segments) in simulation. From output_legs.csv.gz.',

                # Score statistics (agent utility)
                'score_final_executed': matsim_output_stats.get('score_final_executed'),
                'score_final_executed_comment': 'Average executed plan score in final iteration. Higher = better agent utility.',
                'score_final_best': matsim_output_stats.get('score_final_best'),
                'score_final_best_comment': 'Average best plan score per agent in final iteration.',
                'score_improvement': matsim_output_stats.get('score_improvement'),
                'score_improvement_comment': 'Score improvement from iteration 0 to final iteration. Positive = agents found better routes.',

                # Travel distance statistics
                'avg_leg_distance_km': matsim_output_stats.get('avg_leg_distance_km'),
                'avg_leg_distance_km_comment': 'Average distance per leg (single mode segment) in kilometers.',
                'avg_trip_distance_km': matsim_output_stats.get('avg_trip_distance_km'),
                'avg_trip_distance_km_comment': 'Average distance per trip (origin to destination) in kilometers.',

                # Travel time statistics
                'avg_leg_duration_min': matsim_output_stats.get('avg_leg_duration_min'),
                'avg_leg_duration_min_comment': 'Average travel time per leg in minutes. From legdurations.txt.',

                # Aggregate travel statistics
                'total_person_km': matsim_output_stats.get('total_person_km'),
                'total_person_km_comment': 'Total person-kilometers traveled by all agents. From pkm_modestats.csv.',
                'total_person_hours_traveling': matsim_output_stats.get('total_person_hours_traveling'),
                'total_person_hours_traveling_comment': 'Total person-hours spent traveling. From ph_modestats.csv.',

                # Traffic pattern statistics
                'peak_departures_per_5min': matsim_output_stats.get('peak_departures_per_5min'),
                'peak_departures_per_5min_comment': 'Maximum departures in any 5-minute interval. Indicates peak demand.',
                'peak_hour': matsim_output_stats.get('peak_hour'),
                'peak_hour_comment': 'Time of day with highest departure rate (HH:MM:SS format).',
                'max_en_route': matsim_output_stats.get('max_en_route'),
                'max_en_route_comment': 'Maximum number of agents traveling simultaneously. Indicates network load.',

                # Simulation quality indicators
                'total_stuck_agents': matsim_output_stats.get('total_stuck_agents'),
                'total_stuck_agents_comment': 'Agents that could not complete trips (gridlock). Should be 0 for good simulations.',
                'final_iteration': matsim_output_stats.get('final_iteration'),
                'final_iteration_comment': 'Last completed iteration number. Should match config iterations setting.',
            }

        # Add evaluation metrics if available
        if evaluation_metrics is not None:
            # Only include the metrics we need (clean up redundant fields)
            summary['evaluation'] = {
                '_comment': 'Comparison of simulated traffic volumes vs ground truth traffic counts. '
                           'Metrics calculated from hourly volume comparisons at matched device locations.',
                'num_devices': evaluation_metrics.get('num_devices'),
                'num_devices_comment': 'Number of ground truth traffic count devices successfully matched to network links',
                'num_comparisons': evaluation_metrics.get('num_comparisons'),
                'num_comparisons_comment': 'Total hourly comparison points (num_devices * hours with valid data)',
                'geh_lt_5_pct': round(evaluation_metrics.get('geh_lt_5_pct', 0), 2),
                'geh_lt_5_pct_comment': 'Percentage of comparisons with GEH < 5 (industry standard for "good" match). '
                                       'Target: >85% for calibrated models. GEH formula: sqrt(2*(sim-obs)^2/(sim+obs))',
                'correlation': round(evaluation_metrics.get('correlation', 0), 3),
                'correlation_comment': 'Pearson correlation coefficient between simulated and observed volumes. '
                                      'Range: -1 to 1. Values > 0.7 indicate strong positive correlation.',
                'mean_geh': round(evaluation_metrics.get('mean_geh', 0), 2),
                'mean_geh_comment': 'Average GEH across all comparisons. Lower is better. '
                                   'GEH < 5 is considered acceptable for individual comparisons.',
                'mae': round(evaluation_metrics.get('mae', 0), 2),
                'mae_comment': 'Mean Absolute Error in vehicles/hour. Average |simulated - observed| across all comparisons.',
                'rmse': round(evaluation_metrics.get('rmse', 0), 2),
                'rmse_comment': 'Root Mean Square Error in vehicles/hour. Penalizes large errors more than MAE. '
                               'RMSE = sqrt(mean((simulated - observed)^2))',
                'mean_pct_error': round(evaluation_metrics.get('mean_pct_error', 0), 2),
                'mean_pct_error_comment': 'Mean percentage error: mean((simulated - observed) / observed * 100). '
                                         'Negative = underestimating traffic, Positive = overestimating traffic.',
            }

        # Save summary
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save copy of config used
        config_copy_path = self.experiment_dir / 'config_used.json'
        with open(config_copy_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Experiment summary saved to: {summary_path}")

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
            experiment_id=args.experiment_id
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
