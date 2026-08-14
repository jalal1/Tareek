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

    def __init__(self, config: Dict, config_dir: Optional[Path] = None):
        """
        Initialize config manager

        Args:
            config: Main configuration dictionary from config.json
            config_dir: Directory containing the region's config.json. Used by
                _fixup_data_paths to resolve relative network/plans/transit
                file references. NOT used for config.xml template lookup -
                the template is always the global default at
                matsim/configs/{mode}/config.xml. The region's JSON config
                (specifically matsim.configurable_params) is the single
                source of truth for per-region parameter values.
        """
        self.config = config
        self.matsim_config = config.get('matsim', {})
        self.config_dir = config_dir

    def get_template_path(self, mode: Optional[str] = None) -> Path:
        """
        Get path to the base MATSim config template for the specified mode.

        Per-region parameter overrides live in the region's config.json under
        matsim.configurable_params and are applied on top of this template
        at experiment time (see generate_config). This method intentionally
        does NOT consult any region-specific config.xml - the JSON is the
        single source of truth.

        Args:
            mode: Simulation mode ('basic', 'uber', etc.). If None, uses
                  config default.

        Returns:
            Path to config template file
        """
        if mode is None:
            mode = self.matsim_config.get('mode', 'basic')

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

    def _set_or_add_parameter(
        self,
        tree: ET.ElementTree,
        module_name: str,
        param_name: str,
        param_value: str
    ):
        """Set a parameter, creating it if the template does not define it.

        Unlike update_parameter, a missing param is not an error here. Some
        templates leave optional MATSim settings out (or commented out), in
        which case MATSim applies its own default silently. For settings where
        that default is undesirable — per-iteration output being the case in
        point — the value has to be written explicitly regardless of which
        template revision is deployed.
        """
        root = tree.getroot()

        target_module = None
        for module in root.findall('module'):
            if module.get('name') == module_name:
                target_module = module
                break

        if target_module is None:
            logger.warning(
                f"Module '{module_name}' not in template; cannot set "
                f"'{param_name}'"
            )
            return

        for param in target_module.findall('param'):
            if param.get('name') == param_name:
                param.set('value', param_value)
                return

        ET.SubElement(target_module, 'param',
                      {'name': param_name, 'value': param_value})
        logger.debug(f"Added missing param {module_name}.{param_name} = {param_value}")

    def update_parameter(
        self,
        tree: ET.ElementTree,
        module_name: str,
        param_name: str,
        param_value: str
    ):
        """
        Update a parameter value in the config XML.

        Raises:
            KeyError: if the named module or param does not exist in the
                template. This is a hard error rather than a silent warning -
                a missing target almost always means a typo in
                configurable_params or a base template that is out of date.
                Add the <param> to matsim/configs/{mode}/config.xml or
                remove the offending key from configurable_params.

        Args:
            tree: ElementTree object
            module_name: Name of the module (e.g., 'global', 'controller')
            param_name: Name of the parameter
            param_value: New value for the parameter
        """
        root = tree.getroot()

        target_module = None
        for module in root.findall('module'):
            if module.get('name') == module_name:
                target_module = module
                break

        if target_module is None:
            raise KeyError(
                f"configurable_params target module not found in base "
                f"template: <module name=\"{module_name}\"> (param "
                f"{param_name}={param_value}). Add the module to "
                f"matsim/configs/<mode>/config.xml or remove "
                f"{module_name}.{param_name} from configurable_params."
            )

        for param in target_module.findall('param'):
            if param.get('name') == param_name:
                param.set('value', str(param_value))
                logger.debug(f"Updated {module_name}.{param_name} = {param_value}")
                return

        raise KeyError(
            f"configurable_params target param not found in base template: "
            f"<param name=\"{param_name}\"> inside <module name=\"{module_name}\"> "
            f"(value {param_value}). Add the <param> to "
            f"matsim/configs/<mode>/config.xml or remove "
            f"{module_name}.{param_name} from configurable_params."
        )

    def update_mode_param(
        self,
        tree: ET.ElementTree,
        mode: str,
        param_name: str,
        param_value: str,
    ):
        """Update a per-mode scoring parameter inside scoring/scoringParameters/modeParams.

        The MATSim scoring module nests mode-specific scoring under
        ``<parameterset type="scoringParameters"><parameterset type="modeParams">``
        keyed by ``<param name="mode" value="..."/>``. The flat
        ``module.parameter`` addressing in :meth:`update_parameter` cannot
        reach these nodes, so we walk into them explicitly and either update
        the existing param or insert a new one.
        """
        root = tree.getroot()
        scoring_module = None
        for module in root.findall('module'):
            if module.get('name') == 'scoring':
                scoring_module = module
                break
        if scoring_module is None:
            logger.warning(f"scoring module missing — cannot set modeParams.{mode}.{param_name}")
            return

        scoring_params = scoring_module.find("parameterset[@type='scoringParameters']")
        if scoring_params is None:
            logger.warning(f"scoringParameters missing — cannot set modeParams.{mode}.{param_name}")
            return

        target_block = None
        for ps in scoring_params.findall("parameterset[@type='modeParams']"):
            mode_param = ps.find("param[@name='mode']")
            if mode_param is not None and mode_param.get('value') == mode:
                target_block = ps
                break

        if target_block is None:
            target_block = ET.SubElement(scoring_params, 'parameterset', {'type': 'modeParams'})
            ET.SubElement(target_block, 'param', {'name': 'mode', 'value': mode})
            logger.info(f"Created new modeParams block for mode={mode}")

        existing = target_block.find(f"param[@name='{param_name}']")
        if existing is not None:
            existing.set('value', str(param_value))
        else:
            ET.SubElement(target_block, 'param', {'name': param_name, 'value': str(param_value)})
        logger.info(f"Applied scoring.modeParams.{mode}.{param_name} = {param_value}")

    def update_scoring_param(
        self,
        tree: ET.ElementTree,
        param_name: str,
        param_value: str,
    ):
        """Update a top-level scoringParameters param (e.g. waitingPt, performing).

        These live one level above modeParams: scoring/scoringParameters/<param>.
        """
        root = tree.getroot()
        scoring_module = None
        for module in root.findall('module'):
            if module.get('name') == 'scoring':
                scoring_module = module
                break
        if scoring_module is None:
            logger.warning(f"scoring module missing — cannot set scoring.{param_name}")
            return

        scoring_params = scoring_module.find("parameterset[@type='scoringParameters']")
        if scoring_params is None:
            logger.warning(f"scoringParameters missing — cannot set scoring.{param_name}")
            return

        existing = scoring_params.find(f"param[@name='{param_name}']")
        if existing is not None:
            existing.set('value', str(param_value))
        else:
            ET.SubElement(scoring_params, 'param', {'name': param_name, 'value': str(param_value)})
        logger.info(f"Applied scoring.{param_name} = {param_value}")

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

        If a region-specific config.xml (written by mode_share_estimator) already
        defines transit / transitRouter modules with calibrated values, those
        params are preserved — we only overwrite the ones we own
        (file paths, transitModes). Default-value modules are only synthesized
        when the loaded template has no transit/transitRouter block.
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

        def _set_param(module: ET.Element, name: str, value: str) -> None:
            for p in module.findall('param'):
                if p.get('name') == name:
                    p.set('value', value)
                    return
            ET.SubElement(module, 'param', name=name, value=value)

        # transit module: preserve existing if present (calibrated by estimator),
        # otherwise create from defaults. Always overwrite the params we own.
        transit = next(
            (m for m in root.findall('module') if m.get('name') == 'transit'),
            None,
        )
        if transit is None:
            transit = ET.SubElement(root, 'module', name='transit')
            transit_preserved = False
        else:
            transit_preserved = True
        _set_param(transit, 'useTransit', 'true')
        _set_param(transit, 'transitScheduleFile', 'transitSchedule.xml')
        _set_param(transit, 'vehiclesFile', 'transitVehicles.xml')
        _set_param(transit, 'transitModes', transit_modes_str)

        # transitRouter module: same policy. Defaults are only seeded when the
        # template has no transitRouter block at all; otherwise the region's
        # tuned values (extensionRadius, searchRadius, etc.) are kept.
        router = next(
            (m for m in root.findall('module') if m.get('name') == 'transitRouter'),
            None,
        )
        if router is None:
            router = ET.SubElement(root, 'module', name='transitRouter')
            for name, value in [
                ('additionalTransferTime', '0.0'),
                ('directWalkFactor', '1.0'),
                ('extensionRadius', '200.0'),
                ('maxBeelineWalkConnectionDistance', '100.0'),
                ('searchRadius', '1000.0'),
            ]:
                ET.SubElement(router, 'param', name=name, value=value)
            router_preserved = False
        else:
            router_preserved = True

        preserved_bits = []
        if transit_preserved:
            preserved_bits.append("transit")
        if router_preserved:
            preserved_bits.append("transitRouter")
        preserved_note = (
            f" (preserved existing params on: {', '.join(preserved_bits)})"
            if preserved_bits else ""
        )
        logger.info(f"Enabled transit module with transitModes={transit_modes_str} "
                    f"(from enabled modes: {transit_mode_names}){preserved_note}")

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
        logger.info(f"Auto-set writeLinkStatsInterval = {last_iteration} (matches lastIteration)")

        # Per-iteration output is expensive and mostly unread. At 15-county
        # scale MATSim writes a 256 MB compressed plans file EVERY iteration
        # (~40 s each), plus trips/legs/activities tables and PNG histograms.
        # Only the final iteration is normally looked at, so by default write
        # the bulky artefacts once at the end. Measured: roughly 79% of an
        # iteration's wall time is replanning + output, not the traffic
        # simulation itself.
        #
        # Set matsim.write_intermediate_output = true to restore MATSim's
        # default of writing everything every iteration (useful when debugging
        # how plans evolve across iterations).
        write_intermediate = self.matsim_config.get('write_intermediate_output', False)
        if not write_intermediate:
            # These params may be absent or commented out in a template, in
            # which case MATSim silently defaults to writing every iteration.
            # Set them explicitly, creating them when missing, so the behaviour
            # does not depend on which template revision is deployed.
            for param in ('writePlansInterval', 'writeEventsInterval',
                          'writeSnapshotsInterval'):
                self._set_or_add_parameter(tree, 'controller', param,
                                           str(last_iteration))
            logger.info(
                f"Auto-set write*Interval = {last_iteration} (bulky per-iteration "
                f"output written only at the final iteration; set "
                f"matsim.write_intermediate_output=true to write every iteration)"
            )

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

        # Apply all configurable_params that use module.parameter format.
        # Special-case the scoring module because its mode-specific params live
        # in nested parametersets that update_parameter cannot address:
        #   scoring.modeParams.<mode>.<param>   -> update_mode_param
        #   scoring.<param>                     -> update_scoring_param
        # All other keys continue to use the flat module.parameter format.
        for param_key, value in configurable.items():
            if '.' not in param_key or param_key in ['coordinateSystem', 'lastIteration', 'outputDirectory']:
                continue
            # Skip estimator-generated annotation keys (demand_estimator stores
            # rationale strings next to recommended values using the
            # `_estimator_<leaf>` convention).
            if param_key.startswith('_estimator_') or param_key.startswith('_info'):
                continue
            parts = param_key.split('.')
            if parts[0] == 'scoring' and len(parts) >= 4 and parts[1] == 'modeParams':
                # scoring.modeParams.<mode>.<param>  (param may itself contain dots)
                mode = parts[2]
                inner_param = '.'.join(parts[3:])
                self.update_mode_param(tree, mode, inner_param, str(value))
            elif parts[0] == 'scoring' and len(parts) == 2:
                # scoring.<top-level scoringParameters param>, e.g. scoring.waitingPt
                self.update_scoring_param(tree, parts[1], str(value))
            else:
                module, param = param_key.split('.', 1)
                self.update_parameter(tree, module, param, str(value))
                logger.info(f"Applied configurable param: {module}.{param} = {value}")
                # Capacity factors must be kept in sync between qsim and hermes
                # because mobsim=hermes reads its own module, not qsim.
                if module == 'qsim' and param in ('flowCapacityFactor', 'storageCapacityFactor'):
                    self.update_parameter(tree, 'hermes', param, str(value))
                    logger.info(f"Mirrored qsim.{param} -> hermes.{param} = {value}")

        # Apply custom parameters if provided (these override configurable_params)
        if custom_params:
            for module_param, value in custom_params.items():
                if '.' not in module_param:
                    continue
                # Skip estimator-generated annotation keys. These mirror the
                # filter in the configurable_params loop above; without it,
                # the strict update_parameter raises KeyError on the bogus
                # "_estimator_<leaf>" module names that demand_estimator and
                # mode_share_estimator attach as reason strings.
                if module_param.startswith('_estimator_') or module_param.startswith('_info'):
                    continue
                parts = module_param.split('.')
                if parts[0] == 'scoring' and len(parts) >= 4 and parts[1] == 'modeParams':
                    self.update_mode_param(tree, parts[2], '.'.join(parts[3:]), str(value))
                elif parts[0] == 'scoring' and len(parts) == 2:
                    self.update_scoring_param(tree, parts[1], str(value))
                else:
                    module, param = module_param.split('.', 1)
                    self.update_parameter(tree, module, param, str(value))
                    if module == 'qsim' and param in ('flowCapacityFactor', 'storageCapacityFactor'):
                        self.update_parameter(tree, 'hermes', param, str(value))
                        logger.info(f"Mirrored qsim.{param} -> hermes.{param} = {value}")

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
