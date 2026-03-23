"""
Network Manager for caching and reusing MATSim networks
Manages network storage, metadata, and hash-based retrieval
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NetworkManager:
    """Manages network caching and reuse based on counties or polygon"""

    def __init__(self, networks_dir: Optional[Path] = None):
        """
        Initialize network manager

        Args:
            networks_dir: Directory to store networks (default: ../data/networks)
        """
        if networks_dir is None:
            networks_dir = Path(__file__).parent.parent / 'data' / 'networks'

        self.networks_dir = Path(networks_dir)
        self.networks_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.networks_dir / 'networks_metadata.json'
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load networks metadata from JSON file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata file: {e}. Starting fresh.")
                return {}
        return {}

    def _save_metadata(self):
        """Save networks metadata to JSON file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"Saved network metadata to {self.metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise

    def _compute_network_hash(
        self,
        counties: Optional[List[Tuple[str, str]]] = None,
        polygon: Optional[Dict] = None,
        network_config: Optional[Dict] = None,
    ) -> str:
        """
        Compute hash for network identification based on counties/polygon
        and network configuration (transit, clean_network, etc.)

        Args:
            counties: List of (county, state) tuples
            polygon: Dictionary with polygon coordinates
            network_config: Dict with keys that affect network generation:
                transit_network (bool), clean_network (bool),
                gtfs_sample_day (str)

        Returns:
            Hash string
        """
        if counties:
            # Sort counties to ensure consistent hashing
            sorted_counties = sorted([f"{county},{state}" for county, state in counties])
            hash_input = "counties:" + "|".join(sorted_counties)
        elif polygon:
            # Hash polygon coordinates
            hash_input = "polygon:" + json.dumps(polygon, sort_keys=True)
        else:
            raise ValueError("Either counties or polygon must be provided")

        # Include network config that affects generation output
        if network_config:
            config_str = json.dumps(network_config, sort_keys=True)
            hash_input += "|config:" + config_str

        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def get_network_path(
        self,
        counties: Optional[List[Tuple[str, str]]] = None,
        polygon: Optional[Dict] = None,
        network_config: Optional[Dict] = None,
    ) -> Optional[Tuple[Path, Dict]]:
        """
        Get path to existing network if it exists and is valid

        Args:
            counties: List of (county, state) tuples
            polygon: Dictionary with polygon coordinates
            network_config: Network config dict for hash computation

        Returns:
            Tuple of (network_path, metadata) if exists and valid, None otherwise
        """
        network_hash = self._compute_network_hash(counties, polygon, network_config)

        if network_hash not in self.metadata:
            logger.info(f"No existing network found for hash {network_hash}")
            return None

        network_info = self.metadata[network_hash]
        network_path = Path(network_info['network_path'])

        # Validate network file exists
        if not network_path.exists():
            logger.warning(f"Network file missing: {network_path}. Metadata will be regenerated.")
            del self.metadata[network_hash]
            self._save_metadata()
            return None

        # Validate transit files exist in cache if transit network
        if network_info.get('transit_network'):
            cache_dir = network_path.parent
            for fname in ('transitSchedule.xml', 'transitVehicles.xml'):
                if not (cache_dir / fname).exists():
                    logger.warning(f"Cached transit file missing: {cache_dir / fname}. Will regenerate.")
                    del self.metadata[network_hash]
                    self._save_metadata()
                    return None

        # Validate metadata integrity
        if not self._validate_metadata(network_info, counties, polygon):
            logger.warning(f"Metadata validation failed for {network_hash}. Will regenerate.")
            del self.metadata[network_hash]
            self._save_metadata()
            return None

        logger.info(f"Found existing network: {network_path}")
        logger.info(f"Network ID: {network_hash}")
        if counties:
            logger.info(f"Counties: {len(counties)} counties")
        else:
            logger.info(f"Polygon: {polygon.get('type', 'custom')}")

        return network_path, network_info

    def _validate_metadata(
        self,
        metadata: Dict,
        counties: Optional[List[Tuple[str, str]]],
        polygon: Optional[Dict]
    ) -> bool:
        """
        Validate that metadata matches the requested network configuration

        Args:
            metadata: Stored metadata
            counties: Requested counties
            polygon: Requested polygon

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields exist
            required_fields = ['network_hash', 'network_path', 'created_at']
            if not all(field in metadata for field in required_fields):
                return False

            # Validate counties match
            if counties:
                stored_counties = metadata.get('counties')
                if not stored_counties:
                    return False
                # Sort for comparison
                requested = sorted([f"{c},{s}" for c, s in counties])
                stored = sorted([f"{c},{s}" for c, s in stored_counties])
                if requested != stored:
                    return False

            # Validate polygon match
            if polygon:
                stored_polygon = metadata.get('polygon')
                if not stored_polygon:
                    return False
                if json.dumps(polygon, sort_keys=True) != json.dumps(stored_polygon, sort_keys=True):
                    return False

            return True

        except Exception as e:
            logger.warning(f"Metadata validation error: {e}")
            return False

    def save_network(
        self,
        network_path: Path,
        counties: Optional[List[Tuple[str, str]]] = None,
        polygon: Optional[Dict] = None,
        additional_metadata: Optional[Dict] = None,
        network_config: Optional[Dict] = None,
    ) -> str:
        """
        Save network to cache with metadata.
        Also caches transitSchedule.xml and transitVehicles.xml if present.

        Args:
            network_path: Path to the network.xml file to cache
            counties: List of (county, state) tuples
            polygon: Dictionary with polygon coordinates
            additional_metadata: Additional metadata to store (e.g., num_nodes, num_links)
            network_config: Network config dict for hash computation

        Returns:
            Network hash ID
        """
        from datetime import datetime

        if not network_path.exists():
            raise FileNotFoundError(f"Network file not found: {network_path}")

        network_hash = self._compute_network_hash(counties, polygon, network_config)

        # Create network directory
        network_dir = self.networks_dir / network_hash
        network_dir.mkdir(parents=True, exist_ok=True)

        # Copy network file to cache
        cached_network = network_dir / 'network.xml'
        shutil.copy2(network_path, cached_network)
        logger.info(f"Cached network to: {cached_network}")

        # Create metadata
        metadata = {
            'network_hash': network_hash,
            'network_path': str(cached_network),
            'created_at': datetime.now().isoformat(),
            'counties': counties if counties else None,
            'polygon': polygon if polygon else None,
        }

        # Add additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)

        # Cache transit files if this is a transit network
        if metadata.get('transit_network'):
            experiment_dir = network_path.parent
            for fname in ('transitSchedule.xml', 'transitVehicles.xml'):
                src = experiment_dir / fname
                if src.exists():
                    dst = network_dir / fname
                    shutil.copy2(src, dst)
                    logger.info(f"Cached {fname} to: {dst}")

            # Store cached paths (not experiment paths)
            metadata['transit_schedule_path'] = str(network_dir / 'transitSchedule.xml')
            metadata['transit_vehicles_path'] = str(network_dir / 'transitVehicles.xml')

        # Remove experiment-specific output_path — it's not useful in cache
        metadata.pop('output_path', None)

        # Store metadata
        self.metadata[network_hash] = metadata
        self._save_metadata()

        logger.info(f"Saved network with ID: {network_hash}")
        return network_hash

    def list_networks(self) -> List[Dict]:
        """
        List all cached networks

        Returns:
            List of network metadata dictionaries
        """
        return list(self.metadata.values())

    def delete_network(self, network_hash: str) -> bool:
        """
        Delete a cached network

        Args:
            network_hash: Network hash ID

        Returns:
            True if deleted, False if not found
        """
        if network_hash not in self.metadata:
            logger.warning(f"Network {network_hash} not found")
            return False

        network_dir = self.networks_dir / network_hash
        if network_dir.exists():
            shutil.rmtree(network_dir)
            logger.info(f"Deleted network directory: {network_dir}")

        del self.metadata[network_hash]
        self._save_metadata()

        logger.info(f"Deleted network: {network_hash}")
        return True

    @staticmethod
    def _build_network_config(config: Dict) -> Dict:
        """Build the config dict that affects network generation output,
        used for cache hash computation."""
        matsim_cfg = config.get('matsim', {})
        network_cfg = config.get('network', {})

        # Collect enabled transit mode names (affects network links and schedule)
        modes_config = config.get('modes', {})
        enabled_transit = sorted([
            name for name, cfg in modes_config.items()
            if isinstance(cfg, dict) and cfg.get('enabled', True)
            and cfg.get('matsim_mode') == 'pt'
        ])

        return {
            'transit_network': matsim_cfg.get('transit_network', False),
            'clean_network': network_cfg.get('clean_network', True),
            'gtfs_sample_day': matsim_cfg.get('gtfs_sample_day', 'dayWithMostTrips'),
            'enabled_transit_modes': enabled_transit,
        }

    def get_or_generate_network(
        self,
        network_generator,
        config: Dict,
        output_path: Path,
        counties: Optional[List[Tuple[str, str]]] = None,
        polygon: Optional[Dict] = None,
        db_manager=None,
    ) -> Tuple[Path, Dict]:
        """
        Get existing network or generate new one

        Args:
            network_generator: NetworkGenerator instance
            config: Configuration dictionary
            output_path: Where to place the network.xml for the experiment
            counties: List of (county, state) tuples
            polygon: Dictionary with polygon coordinates
            db_manager: DBManager instance (required when transit_network is true)

        Returns:
            Tuple of (network_path, metadata)
        """
        net_config = self._build_network_config(config)

        # Check if rebuild is forced
        rebuild_network = config.get('network', {}).get('rebuild_network', False)

        if rebuild_network:
            logger.info("rebuild_network=true: skipping cache, will regenerate network")
            # Delete existing cache entry if present so stale data doesn't linger
            network_hash = self._compute_network_hash(
                counties, polygon, net_config
            )
            if network_hash in self.metadata:
                self.delete_network(network_hash)
                logger.info(f"Deleted stale cache entry: {network_hash}")

        # Check if network exists in cache
        cached = None if rebuild_network else self.get_network_path(
            counties=counties, polygon=polygon, network_config=net_config
        )

        if cached:
            cached_path, cached_metadata = cached
            experiment_dir = output_path.parent

            # Copy network from cache to experiment directory
            shutil.copy2(cached_path, output_path)
            logger.info(f"Copied cached network to: {output_path}")

            # Copy transit files from cache if this is a transit network
            if cached_metadata.get('transit_network'):
                cache_dir = cached_path.parent
                for fname in ('transitSchedule.xml', 'transitVehicles.xml'):
                    src = cache_dir / fname
                    if src.exists():
                        dst = experiment_dir / fname
                        shutil.copy2(src, dst)
                        logger.info(f"Copied cached {fname} to: {dst}")
                    else:
                        logger.warning(
                            f"Transit file missing from cache: {src}. "
                            "Cache will be invalidated."
                        )
                        del self.metadata[cached_metadata['network_hash']]
                        self._save_metadata()
                        return self.get_or_generate_network(
                            network_generator, config, output_path,
                            counties, polygon, db_manager
                        )

            return output_path, cached_metadata

        # Generate new network
        logger.info("No cached network found. Generating new network...")

        # Extract network config
        network_cfg = config.get('network', {})
        coordinate_system = config['coordinates']['utm_epsg']
        clean_network = network_cfg.get('clean_network', True)

        # Generate network with full county+state info
        metadata = network_generator.generate_network(
            counties=counties if counties else None,
            output_path=output_path,
            coordinate_system=coordinate_system,
            bbox=polygon.get('bbox') if polygon else None,
            clean_network=clean_network,
            db_manager=db_manager,
        )

        # Cache the network (and transit files if present)
        network_hash = self.save_network(
            network_path=output_path,
            counties=counties,
            polygon=polygon,
            additional_metadata=metadata,
            network_config=net_config,
        )

        logger.info(f"Generated and cached new network: {network_hash}")
        return output_path, metadata
