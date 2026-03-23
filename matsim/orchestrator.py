"""
MATSim Simulation Orchestrator
Coordinates network generation, config management, and simulation execution
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from matsim.network_generator import NetworkGenerator
from matsim.config_manager import ConfigManager
from matsim.runner import MATSimRunner
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MATSimOrchestrator:
    """Orchestrate MATSim simulation pipeline"""

    def __init__(self, config_path: Optional[Path] = None, config_dict: Optional[Dict] = None):
        """
        Initialize orchestrator

        Args:
            config_path: Path to config.json file. If None, uses default location
            config_dict: Config dictionary (takes precedence over config_path if provided)
        """
        if config_dict is not None:
            # Use provided config dictionary
            self.config = config_dict
        elif config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'config.json'
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            with open(config_path, 'r') as f:
                self.config = json.load(f)

        self.network_generator = NetworkGenerator(self.config)
        self.config_manager = ConfigManager(self.config)
        self.runner = MATSimRunner(self.config)

        logger.info("MATSim Orchestrator initialized")

    def create_experiment_directory(self, experiment_id: Optional[str] = None) -> Path:
        """
        Create directory for experiment

        Args:
            experiment_id: Experiment ID. If None, generates timestamp-based ID

        Returns:
            Path to experiment directory
        """
        if experiment_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"exp_{timestamp}"

        experiments_base = Path(__file__).parent.parent / 'experiments'
        experiment_path = experiments_base / experiment_id

        experiment_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created experiment directory: {experiment_path}")

        return experiment_path

    def generate_network_for_twin_cities(
        self,
        experiment_path: Path,
        state: str = "Minnesota",
        num_counties: Optional[int] = None
    ) -> Dict:
        """
        Generate network for Twin Cities counties

        Args:
            experiment_path: Path to experiment directory
            state: State name (deprecated, not used)
            num_counties: Number of counties to include (None = all counties, useful to limit for testing)

        Returns:
            Network metadata dictionary
        """
        from utils.region_utils import RegionHelper

        logger.info("Generating network for selected counties from config")

        # Create RegionHelper to convert GEOIDs to county names
        region_helper = RegionHelper(self.config)

        # Get county names for network generation (OSM download needs names)
        county_names = region_helper.get_county_names_for_network()

        # Limit counties if specified (useful for testing)
        if num_counties is not None:
            county_names = county_names[:num_counties]
            logger.info(f"Limiting to first {num_counties} counties for testing")

        logger.info(f"Counties to include: {len(county_names)} counties")
        for county_name, state_name in county_names:
            logger.info(f"  - {county_name}, {state_name}")

        # Generate network
        network_path = experiment_path / 'network.xml'
        metadata = self.network_generator.generate_network(
            counties=county_names,
            output_path=network_path
        )

        logger.info(f"Network generated: {metadata['num_nodes']} nodes, {metadata['num_links']} links")

        return metadata

    def setup_experiment(
        self,
        experiment_id: Optional[str] = None,
        mode: str = 'basic',
        generate_network: bool = True,
        plans_file: Optional[Path] = None,
        custom_params: Optional[Dict] = None
    ) -> Dict:
        """
        Set up a complete MATSim experiment

        Args:
            experiment_id: Experiment ID
            mode: Simulation mode ('basic', 'uber', etc.)
            generate_network: Whether to generate network (or use existing)
            plans_file: Path to existing plans.xml (if not generating)
            custom_params: Custom config parameters

        Returns:
            Dictionary with experiment metadata
        """
        logger.info(f"Setting up experiment: {experiment_id or 'auto-generated'}")

        # Create experiment directory
        experiment_path = self.create_experiment_directory(experiment_id)

        # Generate or copy network
        if generate_network:
            network_metadata = self.generate_network_for_twin_cities(experiment_path)
            coordinate_system = network_metadata['coordinate_system']
        else:
            # Assume network.xml already exists in experiment dir
            coordinate_system = self.config['coordinates']['utm_epsg']
            network_metadata = {'coordinate_system': coordinate_system}

        # Copy or check for plans.xml
        plans_path = experiment_path / 'plans.xml'
        if plans_file and plans_file.exists():
            import shutil
            # Check if source and destination are the same file
            if plans_file.resolve() != plans_path.resolve():
                shutil.copy(plans_file, plans_path)
                logger.info(f"Copied plans from {plans_file}")
            else:
                logger.info(f"Plans file already exists at destination: {plans_path}")
        elif not plans_path.exists():
            logger.warning(f"No plans.xml found in {experiment_path}. You need to generate or copy plans.xml before running simulation.")

        # Generate config.xml
        config_path = experiment_path / 'config.xml'
        self.config_manager.generate_config(
            output_path=config_path,
            experiment_path=experiment_path,
            coordinate_system=coordinate_system,
            mode=mode,
            custom_params=custom_params
        )

        # Create metadata file
        metadata = {
            'experiment_id': experiment_id or experiment_path.name,
            'created_at': datetime.now().isoformat(),
            'mode': mode,
            'coordinate_system': coordinate_system,
            'network_metadata': network_metadata,
            'custom_params': custom_params or {},
            'paths': {
                'experiment': str(experiment_path),
                'network': str(experiment_path / 'network.xml'),
                'plans': str(experiment_path / 'plans.xml'),
                'config': str(experiment_path / 'config.xml')
            }
        }

        # Note: experiment_metadata.json is no longer written here
        # All metrics are consolidated into experiment_summary.json by ExperimentRunner

        logger.info(f"Experiment setup complete: {experiment_path}")

        return metadata

    def run_experiment(
        self,
        experiment_path: Path,
        blocking: bool = True
    ) -> Optional[subprocess.Popen]:
        """
        Run MATSim simulation for an experiment

        Args:
            experiment_path: Path to experiment directory
            blocking: Whether to wait for simulation to complete

        Returns:
            Subprocess handle if not blocking, None if blocking
        """
        config_path = experiment_path / 'config.xml'

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Validate setup
        logger.info("Validating MATSim setup...")
        if not self.runner.validate_setup():
            raise RuntimeError("MATSim setup validation failed")

        # Validate config
        logger.info("Validating experiment configuration...")
        if not self.config_manager.validate_config(config_path):
            raise RuntimeError("Experiment configuration validation failed")

        # Run simulation
        log_file = experiment_path / 'matsim_output.log'
        logger.info(f"Starting MATSim simulation. Log file: {log_file}")

        import subprocess
        process = self.runner.run_simulation(
            config_path=config_path,
            log_file=log_file,
            blocking=blocking
        )

        if blocking:
            logger.info("Simulation completed")
            return None
        else:
            logger.info("Simulation started in background")
            return process

    def create_and_run_experiment(
        self,
        experiment_id: Optional[str] = None,
        mode: str = 'basic',
        generate_network: bool = True,
        plans_file: Optional[Path] = None,
        custom_params: Optional[Dict] = None,
        run_simulation: bool = None
    ) -> Dict:
        """
        Complete workflow: setup and run experiment

        Args:
            experiment_id: Experiment ID
            mode: Simulation mode
            generate_network: Whether to generate network
            plans_file: Path to plans.xml file
            custom_params: Custom parameters
            run_simulation: Whether to run simulation (None = use config default)

        Returns:
            Experiment metadata
        """
        # Setup experiment
        metadata = self.setup_experiment(
            experiment_id=experiment_id,
            mode=mode,
            generate_network=generate_network,
            plans_file=plans_file,
            custom_params=custom_params
        )

        # Run simulation if requested
        should_run = run_simulation if run_simulation is not None else self.config.get('matsim', {}).get('run_simulation', False)

        if should_run:
            experiment_path = Path(metadata['paths']['experiment'])
            self.run_experiment(experiment_path, blocking=True)
            metadata['simulation_status'] = 'completed'
        else:
            metadata['simulation_status'] = 'not_run'
            logger.info("Simulation not run (run_simulation=False)")

        return metadata


def main():
    """Example usage"""
    import subprocess

    # Initialize orchestrator
    orchestrator = MATSimOrchestrator()

    # Example 1: Setup experiment without running simulation
    logger.info("=== Example 1: Setup experiment ===")
    metadata = orchestrator.setup_experiment(
        experiment_id='test_twin_cities',
        mode='basic',
        generate_network=True
    )
    logger.info(f"Experiment created: {metadata['paths']['experiment']}")

    # Example 2: Run existing experiment
    # logger.info("=== Example 2: Run experiment ===")
    # experiment_path = Path(metadata['paths']['experiment'])
    # orchestrator.run_experiment(experiment_path)


if __name__ == '__main__':
    main()
