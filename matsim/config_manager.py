"""
MATSim configuration file manager
Handles loading templates and generating customized config.xml files
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manage MATSim configuration files"""

    def __init__(self, config: Dict):
        """
        Initialize config manager

        Args:
            config: Main configuration dictionary from config.json
        """
        self.config = config
        self.matsim_config = config.get('matsim', {})

    def get_template_path(self, mode: Optional[str] = None) -> Path:
        """
        Get path to config template for specified mode

        Args:
            mode: Simulation mode ('basic', 'uber', etc.). If None, uses config default

        Returns:
            Path to config template file
        """
        if mode is None:
            mode = self.matsim_config.get('mode', 'basic')

        # Config templates are in matsim/configs/{mode}/
        template_path = Path(__file__).parent / 'configs' / mode / 'config.xml'

        if not template_path.exists():
            raise FileNotFoundError(f"Config template not found: {template_path}")

        logger.info(f"Using config template: {template_path}")
        return template_path

    def load_template(self, mode: Optional[str] = None) -> ET.ElementTree:
        """
        Load config template XML

        Args:
            mode: Simulation mode

        Returns:
            ElementTree object
        """
        template_path = self.get_template_path(mode)
        tree = ET.parse(template_path)
        return tree

    def update_parameter(
        self,
        tree: ET.ElementTree,
        module_name: str,
        param_name: str,
        param_value: str
    ):
        """
        Update a parameter value in the config XML

        Args:
            tree: ElementTree object
            module_name: Name of the module (e.g., 'global', 'controler')
            param_name: Name of the parameter
            param_value: New value for the parameter
        """
        root = tree.getroot()

        # Find the module
        for module in root.findall('module'):
            if module.get('name') == module_name:
                # Find the parameter
                for param in module.findall('param'):
                    if param.get('name') == param_name:
                        param.set('value', str(param_value))
                        logger.debug(f"Updated {module_name}.{param_name} = {param_value}")
                        return

        logger.warning(f"Parameter not found: {module_name}.{param_name}")

    def _get_enabled_transit_modes(self) -> list:
        """Return list of enabled mode names that map to MATSim 'pt' (transit modes)."""
        modes_config = self.config.get('modes', {})
        transit_modes = []
        for mode_name, mode_cfg in modes_config.items():
            if not isinstance(mode_cfg, dict):
                continue
            if not mode_cfg.get('enabled', True):
                continue
            if mode_cfg.get('matsim_mode') == 'pt':
                transit_modes.append(mode_name)
        return transit_modes

    def _get_enabled_transit_matsim_modes(self) -> list:
        """Return deduplicated list of matsim_mode values for enabled transit modes.

        MATSim's ``transitModes`` parameter must list the mode strings that
        agents use in ``<leg mode="...">``.  Multiple config modes (bus, rail)
        may map to the same matsim_mode (``pt``), so we deduplicate.
        """
        modes_config = self.config.get('modes', {})
        seen = set()
        matsim_modes = []
        for mode_name, mode_cfg in modes_config.items():
            if not isinstance(mode_cfg, dict):
                continue
            if not mode_cfg.get('enabled', True):
                continue
            mm = mode_cfg.get('matsim_mode')
            if mm == 'pt' and mm not in seen:
                seen.add(mm)
                matsim_modes.append(mm)
        return matsim_modes

    def _enable_transit_module(self, tree: ET.ElementTree) -> None:
        """Add or enable the transit and transitRouter modules in the MATSim config.

        Called when matsim.transit_network is true. Sets ``transitModes`` to the
        MATSim mode strings that agents use in their legs (e.g. ``pt``), NOT
        the config mode names (e.g. ``bus``).
        """
        root = tree.getroot()

        # Check that at least one transit mode is enabled
        transit_mode_names = self._get_enabled_transit_modes()
        if not transit_mode_names:
            logger.warning("No enabled transit modes found, skipping transit module")
            return

        # transitModes must match what agents use in <leg mode="...">
        matsim_modes = self._get_enabled_transit_matsim_modes()
        transit_modes_str = ','.join(matsim_modes)

        # Remove any existing transit/transitRouter modules (shouldn't exist, but be safe)
        for module in list(root.findall('module')):
            if module.get('name') in ('transit', 'transitRouter'):
                root.remove(module)

        # Add transit module
        transit = ET.SubElement(root, 'module', name='transit')
        for name, value in [
            ('useTransit', 'true'),
            ('transitScheduleFile', 'transitSchedule.xml'),
            ('vehiclesFile', 'transitVehicles.xml'),
            ('transitModes', transit_modes_str),
        ]:
            ET.SubElement(transit, 'param', name=name, value=value)

        # Add transitRouter module
        router = ET.SubElement(root, 'module', name='transitRouter')
        for name, value in [
            ('additionalTransferTime', '0.0'),
            ('directWalkFactor', '1.0'),
            ('extensionRadius', '200.0'),
            ('maxBeelineWalkConnectionDistance', '500.0'),
            ('searchRadius', '1500.0'),
        ]:
            ET.SubElement(router, 'param', name=name, value=value)

        logger.info(f"Enabled transit module with transitModes={transit_modes_str} "
                     f"(from enabled modes: {transit_mode_names})")

    def generate_config(
        self,
        output_path: Path,
        experiment_path: Path,
        coordinate_system: str,
        mode: Optional[str] = None,
        custom_params: Optional[Dict] = None
    ) -> Path:
        """
        Generate customized config.xml for an experiment

        Args:
            output_path: Path where config.xml will be saved
            experiment_path: Path to experiment directory (for relative file paths)
            coordinate_system: EPSG code for coordinate system
            mode: Simulation mode ('basic', 'uber', etc.)
            custom_params: Dictionary of custom parameter overrides

        Returns:
            Path to generated config file
        """
        logger.info(f"Generating MATSim config for mode: {mode or 'default'}")

        # Load template
        tree = self.load_template(mode)

        # Get configurable parameters from main config
        configurable = self.matsim_config.get('configurable_params', {})

        # Update coordinate system
        coord_system = coordinate_system if configurable.get('coordinateSystem') == 'auto' else configurable.get('coordinateSystem')
        if coord_system:
            self.update_parameter(tree, 'global', 'coordinateSystem', coord_system)

        # Update last iteration
        last_iteration = configurable.get('lastIteration', 10)
        self.update_parameter(tree, 'controller', 'lastIteration', str(last_iteration))

        # Auto-set writeLinkStatsInterval to lastIteration (write stats only at final iteration)
        self.update_parameter(tree, 'linkStats', 'writeLinkStatsInterval', str(last_iteration))
        logger.info(f"Auto-set linkStats.writeLinkStatsInterval = {last_iteration} (matches lastIteration)")

        # Handle counts module based on counts config
        counts_config = self.config.get('counts', {})
        counts_enabled = counts_config.get('enabled', True)

        if counts_enabled:
            # Auto-set counts parameters
            # countsScaleFactor = 1 / flowCapacityFactor (to scale up simulated counts to real-world)
            # flowCapacityFactor reflects the true simulated-to-real traffic ratio,
            # which may differ from scaling_factor (population sample fraction)
            flow_capacity_factor = float(configurable.get('qsim.flowCapacityFactor', 0.1))
            counts_scale_factor = 1.0 / flow_capacity_factor if flow_capacity_factor > 0 else 10.0
            self.update_parameter(tree, 'counts', 'countsScaleFactor', str(counts_scale_factor))
            logger.info(f"Auto-set counts.countsScaleFactor = {counts_scale_factor} (1/{flow_capacity_factor} flowCapacityFactor)")

            # writeCountsInterval = lastIteration (write counts comparison only at final iteration)
            self.update_parameter(tree, 'counts', 'writeCountsInterval', str(last_iteration))
            logger.info(f"Auto-set counts.writeCountsInterval = {last_iteration} (matches lastIteration)")

            # averageCountsOverIterations defaults to lastIteration
            self.update_parameter(tree, 'counts', 'averageCountsOverIterations', str(last_iteration))
        else:
            # Remove counts module entirely when counts generation is disabled
            # MATSim will fail if counts module is present but counts.xml doesn't exist
            root = tree.getroot()
            for module in root.findall('module'):
                if module.get('name') == 'counts':
                    root.remove(module)
                    logger.info("Removed counts module from config (counts.enabled = false)")
                    break

        # Update output directory (relative to experiment path)
        output_dir = configurable.get('outputDirectory', 'output')
        self.update_parameter(tree, 'controller', 'outputDirectory', output_dir)

        # Apply all configurable_params that use module.parameter format
        for param_key, value in configurable.items():
            if '.' in param_key and param_key not in ['coordinateSystem', 'lastIteration', 'outputDirectory']:
                module, param = param_key.split('.', 1)
                self.update_parameter(tree, module, param, str(value))
                logger.info(f"Applied configurable param: {module}.{param} = {value}")

        # Apply custom parameters if provided (these override configurable_params)
        if custom_params:
            for module_param, value in custom_params.items():
                if '.' in module_param:
                    module, param = module_param.split('.', 1)
                    self.update_parameter(tree, module, param, str(value))

        # Ensure file paths are relative (network.xml, plans.xml are in same dir as config)
        self.update_parameter(tree, 'network', 'inputNetworkFile', 'network.xml')
        self.update_parameter(tree, 'plans', 'inputPlansFile', 'plans.xml')

        # Enable transit module when transit_network is true
        if self.matsim_config.get('transit_network', False):
            self._enable_transit_module(tree)

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write XML with proper formatting and DOCTYPE
        # Pretty print the tree first
        ET.indent(tree, space="    ")  # 4-space indent to match original templates

        # Use ElementTree's built-in write with XML declaration
        tree.write(
            output_path,
            encoding='unicode',
            xml_declaration=True,
            method='xml'
        )

        # Now add DOCTYPE after XML declaration
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Insert DOCTYPE after XML declaration
        lines = content.split('\n')
        if lines[0].startswith('<?xml'):
            # Add DOCTYPE after XML declaration
            lines.insert(1, '<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">')
            content = '\n'.join(lines)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

        logger.info(f"Config saved to: {output_path}")
        return output_path

    def validate_config(self, config_path: Path) -> bool:
        """
        Validate that required files referenced in config exist

        Args:
            config_path: Path to config.xml file

        Returns:
            True if valid, False otherwise
        """
        tree = ET.parse(config_path)
        root = tree.getroot()

        config_dir = config_path.parent
        required_files = []

        # Check network file
        for module in root.findall('module'):
            if module.get('name') == 'network':
                for param in module.findall('param'):
                    if param.get('name') == 'inputNetworkFile':
                        network_file = config_dir / param.get('value')
                        required_files.append(('network.xml', network_file))

            if module.get('name') == 'plans':
                for param in module.findall('param'):
                    if param.get('name') == 'inputPlansFile':
                        plans_file = config_dir / param.get('value')
                        required_files.append(('plans.xml', plans_file))

            if module.get('name') == 'transit':
                for param in module.findall('param'):
                    if param.get('name') == 'useTransit' and param.get('value') == 'true':
                        # Check for transit files
                        for transit_param in module.findall('param'):
                            if transit_param.get('name') in ['transitScheduleFile', 'vehiclesFile']:
                                transit_file = config_dir / transit_param.get('value')
                                required_files.append((transit_param.get('name'), transit_file))

        # Validate files exist
        all_valid = True
        for file_type, file_path in required_files:
            if not file_path.exists():
                logger.error(f"Required file missing: {file_type} at {file_path}")
                all_valid = False
            else:
                logger.info(f"Found {file_type}: {file_path}")

        return all_valid
