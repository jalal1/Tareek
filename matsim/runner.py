"""
MATSim simulation runner
Handles execution of MATSim simulations with cross-platform support
"""

import subprocess
import platform
import os
from pathlib import Path
from typing import Dict, Optional
import logging
import time

logger = logging.getLogger(__name__)


class MATSimRunner:
    """Run MATSim simulations"""

    def __init__(self, config: Dict):
        """
        Initialize MATSim runner

        Args:
            config: Main configuration dictionary from config.json
        """
        self.config = config
        self.matsim_config = config.get('matsim', {})
        self.os_type = platform.system()

    def get_jar_path(self) -> Path:
        """
        Get path to MATSim JAR file

        Returns:
            Path to JAR file
        """
        matsim_version = self.matsim_config.get('version', 'matsim_25')
        base_path = Path(__file__).parent / matsim_version

        # Find the JAR file
        jar_files = list(base_path.glob('*.jar'))

        if not jar_files:
            raise FileNotFoundError(f"No JAR file found in {base_path}")

        # Use the first JAR file
        jar_path = jar_files[0]
        logger.info(f"Using JAR file: {jar_path}")

        return jar_path

    def get_lib_path(self) -> Optional[Path]:
        """
        Get path to MATSim library directory (if exists)

        Returns:
            Path to lib directory or None if not found
        """
        matsim_version = self.matsim_config.get('version', 'matsim_25')
        base_path = Path(__file__).parent / matsim_version

        # Find lib directory (assumes pattern: *_lib)
        lib_dirs = list(base_path.glob('*_lib'))

        if not lib_dirs:
            logger.info(f"No lib directory found in {base_path} (may not be needed)")
            return None

        lib_path = lib_dirs[0]
        logger.info(f"Using lib directory: {lib_path}")

        return lib_path

    def build_classpath(self) -> str:
        """
        Build Java classpath for MATSim

        Returns:
            Classpath string
        """
        jar_path = self.get_jar_path()
        lib_path = self.get_lib_path()

        # Platform-specific path separator
        sep = ';' if self.os_type == 'Windows' else ':'

        # Build classpath: jar + all jars in lib (if lib exists)
        if lib_path and lib_path.exists():
            classpath = f"{jar_path}{sep}{lib_path}/*"
        else:
            classpath = str(jar_path)

        return classpath

    def build_command(self, config_path: Path) -> list:
        """
        Build Java command to run MATSim

        Args:
            config_path: Path to config.xml file

        Returns:
            Command as list of strings
        """
        classpath = self.build_classpath()
        heap_size = self.matsim_config.get('heap_size_gb', 32)

        # Build command with JVM performance optimizations
        # Note: These are JVM flags, not MATSim arguments
        cmd = [
            'java',
            f'-Xmx{heap_size}g',  # Maximum heap size
            f'-Xms{heap_size}g',  # Initial heap size (same as max to avoid resizing)
            '-XX:+UseG1GC',  # G1 garbage collector (recommended for heaps > 4GB)
            '-XX:+ParallelRefProcEnabled',  # Parallel reference processing
            '-XX:MaxGCPauseMillis=200',  # Target max GC pause time (200ms)
            '-XX:+HeapDumpOnOutOfMemoryError',  # Create heap dump on OOM
            '-XX:HeapDumpPath=./heap_dumps',  # Where to save heap dumps
            '-cp', classpath,
            'org.matsim.core.controler.Controler',
            str(config_path.resolve())
        ]

        return cmd

    def run_simulation(
        self,
        config_path: Path,
        log_file: Optional[Path] = None,
        blocking: bool = True
    ) -> subprocess.Popen:
        """
        Run MATSim simulation

        Args:
            config_path: Path to config.xml file
            log_file: Optional path to log file for simulation output
            blocking: If True, wait for simulation to complete

        Returns:
            Subprocess handle
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info(f"Starting MATSim simulation with config: {config_path}")

        # Build command
        cmd = self.build_command(config_path)

        # Log command
        logger.info(f"Command: {' '.join(cmd)}")

        # Prepare output streams
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            stdout_file = open(log_file, 'w')
            stderr_file = subprocess.STDOUT
        else:
            stdout_file = subprocess.PIPE
            stderr_file = subprocess.PIPE

        # Run simulation
        start_time = time.time()

        try:
            process = subprocess.Popen(
                cmd,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                cwd=config_path.parent  # Run from experiment directory
            )

            if blocking:
                logger.info("Waiting for simulation to complete...")
                return_code = process.wait()

                elapsed_time = time.time() - start_time
                logger.info(f"Simulation completed in {elapsed_time:.2f} seconds")

                if return_code == 0:
                    logger.info("Simulation completed successfully")
                else:
                    logger.error(f"Simulation failed with return code: {return_code}")
                    if not log_file:
                        stdout, stderr = process.communicate()
                        if stdout:
                            logger.error(f"STDOUT: {stdout}")
                        if stderr:
                            logger.error(f"STDERR: {stderr}")

                if log_file and isinstance(stdout_file, type(open(__file__))):
                    stdout_file.close()

            return process

        except Exception as e:
            logger.error(f"Failed to run simulation: {e}")
            if log_file and isinstance(stdout_file, type(open(__file__))):
                stdout_file.close()
            raise

    def check_java_version(self) -> Dict:
        """
        Check if Java is installed and get version

        Returns:
            Dictionary with Java version info
        """
        try:
            result = subprocess.run(
                ['java', '-version'],
                capture_output=True,
                text=True
            )

            # Java version is in stderr
            version_output = result.stderr

            logger.info(f"Java version: {version_output.split()[0]}")

            return {
                'installed': True,
                'version': version_output,
                'status': 'ok'
            }

        except FileNotFoundError:
            logger.error("Java not found. Please install Java to run MATSim.")
            return {
                'installed': False,
                'version': None,
                'status': 'error'
            }

    def validate_setup(self) -> bool:
        """
        Validate that all required components are available

        Returns:
            True if valid, False otherwise
        """
        logger.info("Validating MATSim setup...")

        # Check Java
        java_info = self.check_java_version()
        if not java_info['installed']:
            return False

        # Check JAR file
        try:
            jar_path = self.get_jar_path()
            if not jar_path.exists():
                logger.error(f"JAR file not found: {jar_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to locate JAR file: {e}")
            return False

        # Check lib directory (optional for matsim_25)
        try:
            lib_path = self.get_lib_path()
            if lib_path and not lib_path.exists():
                logger.warning(f"Lib directory not found: {lib_path} (may not be needed)")
            elif lib_path:
                logger.info(f"Lib directory found: {lib_path}")
        except Exception as e:
            logger.warning(f"Could not check lib directory: {e} (may not be needed)")

        logger.info("MATSim setup validation passed")
        return True
