"""
Configuration validator for experiment runner
Validates all config parameters upfront before starting experiments
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors"""
    pass


class ConfigValidator:
    """Validates experiment configuration"""

    REQUIRED_SECTIONS = [
        'data',
        'plan_generation',
        'network',
        'matsim'
    ]

    VALID_MATSIM_MODES = ['basic', 'uber']

    def __init__(self, config_path: Path):
        """
        Initialize validator

        Args:
            config_path: Path to config.json file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load and parse JSON config file"""
        if not self.config_path.exists():
            raise ConfigValidationError(f"Config file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded config from: {self.config_path}")
            return config
        except json.JSONDecodeError as e:
            raise ConfigValidationError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise ConfigValidationError(f"Error loading config: {e}")

    def validate(self) -> Dict:
        """
        Validate all configuration parameters

        Returns:
            Validated configuration dictionary

        Raises:
            ConfigValidationError: If validation fails
        """
        logger.info("Validating configuration...")

        # Validate required sections exist
        self._validate_required_sections()

        # Validate network configuration
        self._validate_network_config()

        # Validate MATSim configuration
        self._validate_matsim_config()

        # Validate plan generation configuration
        self._validate_plan_generation_config()

        # Validate data paths
        self._validate_data_paths()

        # Validate counts configuration
        self._validate_counts_config()

        logger.info("Configuration validation passed")
        return self.config

    def _validate_required_sections(self):
        """Validate all required sections exist"""
        missing = [s for s in self.REQUIRED_SECTIONS if s not in self.config]
        if missing:
            raise ConfigValidationError(
                f"Missing required config sections: {', '.join(missing)}"
            )

    def _validate_network_config(self):
        """Validate network configuration"""
        network_config = self.config.get('network', {})
        region_config = self.config.get('region', {})

        # Counties are now in region.counties, but we also check network.counties for backwards compatibility
        counties = region_config.get('counties') or network_config.get('counties')
        polygon = network_config.get('polygon')

        # Exactly one of counties or polygon must be provided
        if counties is None and polygon is None:
            raise ConfigValidationError(
                "Network configuration error: Either 'counties' or 'polygon' must be specified.\n"
                "Please update your config.json to include one of these options."
            )

        if counties is not None and polygon is not None:
            raise ConfigValidationError(
                "Network configuration error: Both 'counties' and 'polygon' are specified.\n"
                "Please use ONLY ONE option in your config.json:\n"
                "  - Set 'polygon' to null to use counties, OR\n"
                "  - Set 'counties' to null to use polygon"
            )

        # Validate counties format (GEOIDs)
        if counties is not None:
            if not isinstance(counties, list):
                raise ConfigValidationError(
                    "Region 'counties' must be a list of county GEOIDs"
                )

            if len(counties) == 0:
                raise ConfigValidationError(
                    "Region 'counties' list cannot be empty"
                )

            for i, geoid in enumerate(counties):
                if not isinstance(geoid, str):
                    raise ConfigValidationError(
                        f"Invalid county GEOID at index {i}: {geoid}\n"
                        f"Expected format: 5-character string (state_fips + county_fips)\n"
                        f"Example: \"27053\" for Hennepin County, Minnesota"
                    )

                if len(geoid) != 5:
                    raise ConfigValidationError(
                        f"Invalid county GEOID length at index {i}: {geoid}\n"
                        f"Expected 5 characters (2-digit state FIPS + 3-digit county FIPS)\n"
                        f"Got {len(geoid)} characters"
                    )

                if not geoid.isdigit():
                    raise ConfigValidationError(
                        f"Invalid county GEOID at index {i}: {geoid}\n"
                        f"GEOID must contain only digits"
                    )

            logger.info(f"Network config: Using {len(counties)} counties")

        # Validate polygon format
        if polygon is not None:
            if not isinstance(polygon, dict):
                raise ConfigValidationError(
                    "Network 'polygon' must be a dictionary with bbox or coordinates"
                )

            # Check for either bbox or coordinates
            if 'bbox' not in polygon and 'coordinates' not in polygon:
                raise ConfigValidationError(
                    "Network 'polygon' must contain either 'bbox' or 'coordinates'"
                )

            if 'bbox' in polygon:
                bbox = polygon['bbox']
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ConfigValidationError(
                        "Network polygon 'bbox' must be a list of 4 numbers: "
                        "[min_lon, min_lat, max_lon, max_lat]"
                    )

                try:
                    bbox = [float(x) for x in bbox]
                    min_lon, min_lat, max_lon, max_lat = bbox

                    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
                        raise ConfigValidationError(
                            f"Invalid longitude values in bbox: {min_lon}, {max_lon}"
                        )

                    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
                        raise ConfigValidationError(
                            f"Invalid latitude values in bbox: {min_lat}, {max_lat}"
                        )

                    if min_lon >= max_lon or min_lat >= max_lat:
                        raise ConfigValidationError(
                            f"Invalid bbox: min values must be less than max values"
                        )

                except (ValueError, TypeError) as e:
                    raise ConfigValidationError(f"Invalid bbox values: {e}")

            logger.info("Network config: Using polygon/bbox")

        # Validate clean_network flag
        clean_network = network_config.get('clean_network', True)
        if not isinstance(clean_network, bool):
            raise ConfigValidationError(
                "Network 'clean_network' must be a boolean (true/false)"
            )

    def _validate_matsim_config(self):
        """Validate MATSim configuration"""
        matsim_config = self.config.get('matsim', {})

        # Validate mode
        mode = matsim_config.get('mode')
        if not mode:
            raise ConfigValidationError("MATSim 'mode' is required")

        if mode not in self.VALID_MATSIM_MODES:
            raise ConfigValidationError(
                f"Invalid MATSim mode: {mode}. "
                f"Valid options: {', '.join(self.VALID_MATSIM_MODES)}"
            )

        # Validate run_simulation flag
        run_simulation = matsim_config.get('run_simulation')
        if run_simulation is not None and not isinstance(run_simulation, bool):
            raise ConfigValidationError(
                "MATSim 'run_simulation' must be a boolean (true/false)"
            )

        # Validate heap_size_gb
        heap_size = matsim_config.get('heap_size_gb', 4)
        if not isinstance(heap_size, (int, float)) or heap_size <= 0:
            raise ConfigValidationError(
                f"MATSim 'heap_size_gb' must be a positive number, got: {heap_size}"
            )

        # Validate configurable_params
        params = matsim_config.get('configurable_params', {})
        if not isinstance(params, dict):
            raise ConfigValidationError(
                "MATSim 'configurable_params' must be a dictionary"
            )

        # Validate lastIteration if present
        if 'lastIteration' in params:
            try:
                iterations = int(params['lastIteration'])
                if iterations < 0:
                    raise ConfigValidationError(
                        f"MATSim 'lastIteration' must be non-negative, got: {iterations}"
                    )
            except (ValueError, TypeError):
                raise ConfigValidationError(
                    f"MATSim 'lastIteration' must be a number, got: {params['lastIteration']}"
                )

        logger.info(f"MATSim config: mode={mode}, run_simulation={run_simulation}")

    def _validate_plan_generation_config(self):
        """Validate plan generation configuration"""
        plan_config = self.config.get('plan_generation', {})

        # Validate target_plans
        target_plans = plan_config.get('target_plans')
        if target_plans is not None:
            if isinstance(target_plans, str):
                if target_plans.lower() != "all":
                    raise ConfigValidationError(
                        f"plan_generation 'target_plans' must be a positive integer or 'all', got: {target_plans}"
                    )
            elif not isinstance(target_plans, int) or target_plans <= 0:
                raise ConfigValidationError(
                    f"plan_generation 'target_plans' must be a positive integer or 'all', got: {target_plans}"
                )

        # Validate scaling_factor
        scaling_factor = plan_config.get('scaling_factor', 1)
        if not isinstance(scaling_factor, (int, float)) or scaling_factor <= 0:
            raise ConfigValidationError(
                f"plan_generation 'scaling_factor' must be a positive number, got: {scaling_factor}"
            )

        # Validate num_processes
        num_processes = plan_config.get('num_processes', 1)
        if not isinstance(num_processes, int) or num_processes <= 0:
            raise ConfigValidationError(
                f"plan_generation 'num_processes' must be a positive integer, got: {num_processes}"
            )

        logger.info(f"Plan generation config: target_plans={target_plans}, scaling_factor={scaling_factor}")

    def _validate_data_paths(self):
        """Validate data directory paths exist"""
        data_config = self.config.get('data', {})
        data_dir = data_config.get('data_dir')

        if not data_dir:
            raise ConfigValidationError("data 'data_dir' is required")

        # Resolve relative to config file location
        if not Path(data_dir).is_absolute():
            data_dir_path = (self.config_path.parent / data_dir).resolve()
        else:
            data_dir_path = Path(data_dir)

        if not data_dir_path.exists():
            raise ConfigValidationError(
                f"Data directory does not exist: {data_dir_path}\n"
                f"Resolved from config value: {data_dir}"
            )

        # Update config with absolute path to avoid path resolution issues later
        self.config['data']['data_dir'] = str(data_dir_path)

        logger.info(f"Data directory validated: {data_dir_path}")

    def _validate_counts_config(self):
        """Validate counts configuration (optional section)"""
        counts_config = self.config.get('counts', {})
        if not counts_config:
            return  # counts section is optional

        enabled = counts_config.get('enabled', True)
        if not isinstance(enabled, bool):
            raise ConfigValidationError("counts 'enabled' must be a boolean")

        fha_config = counts_config.get('fha', {})
        if fha_config:
            year = fha_config.get('year')
            if year is not None and (not isinstance(year, int) or year < 2000):
                raise ConfigValidationError(
                    f"counts.fha 'year' must be an integer >= 2000, got: {year}"
                )

            month = fha_config.get('month')
            if month is not None and (not isinstance(month, int) or not 1 <= month <= 12):
                raise ConfigValidationError(
                    f"counts.fha 'month' must be an integer 1-12, got: {month}"
                )

            weight = fha_config.get('weight')
            if weight is not None and (not isinstance(weight, (int, float)) or weight < 0):
                raise ConfigValidationError(
                    f"counts.fha 'weight' must be a non-negative number, got: {weight}"
                )

        custom_config = counts_config.get('custom', {})
        if custom_config:
            custom_enabled = custom_config.get('enabled')
            if custom_enabled is not None and not isinstance(custom_enabled, bool):
                raise ConfigValidationError("counts.custom 'enabled' must be a boolean")

            weight = custom_config.get('weight')
            if weight is not None and (not isinstance(weight, (int, float)) or weight < 0):
                raise ConfigValidationError(
                    f"counts.custom 'weight' must be a non-negative number, got: {weight}"
                )

        logger.info(f"Counts config: enabled={enabled}, "
                    f"custom={custom_config.get('enabled', False)}")

    def get_network_spec(self) -> Tuple[Optional[List[str]], Optional[Dict]]:
        """
        Get network specification from config

        Returns:
            Tuple of (counties, polygon) where one is None
            counties is a list of GEOIDs (5-character strings)
        """
        network_config = self.config['network']
        region_config = self.config.get('region', {})

        # Counties are now in region.counties, but we also check network.counties for backwards compatibility
        counties = region_config.get('counties') or network_config.get('counties')
        polygon = network_config.get('polygon')

        return counties, polygon
