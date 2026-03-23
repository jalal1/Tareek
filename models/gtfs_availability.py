"""
GTFS-based transit availability using spatial indices.

Builds R-tree (STRtree) indices per mode from GTFS stop data,
enabling fast "are there transit stops near this location?" queries.

Phase 3 Implementation:
- Separate spatial index per mode (bus stops != rail stops)
- Uses shapely STRtree for O(log n) spatial queries
- Buffer distance configurable per mode in config.json
"""

import logging
import time
from math import cos, radians
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree

from data_sources.gtfs_manager import GTFSManager
from models.mode_types import ModeType

logger = logging.getLogger(__name__)


class GTFSAvailabilityManager:
    """
    Manages spatial indices of transit stops for availability checking.

    Each transit mode (bus, rail) gets its own R-tree index built from
    stops that serve routes of the corresponding GTFS route_types.

    Usage:
        manager = GTFSAvailabilityManager()
        manager.build_indices(gtfs_manager, modes_config)
        has_bus = manager.has_stops_nearby(ModeType.BUS, lat, lon, buffer_m=800)
    """

    def __init__(self):
        # mode_type -> (STRtree, array of stop points)
        self._indices: Dict[ModeType, STRtree] = {}
        self._stop_points: Dict[ModeType, np.ndarray] = {}
        self._stop_counts: Dict[ModeType, int] = {}

    def build_indices(self, gtfs_manager: GTFSManager,
                      modes_config: Dict) -> None:
        """
        Build R-tree spatial indices for each GTFS-based mode.

        For each mode with availability.type='gtfs', queries the database
        for stops serving the configured route_types, then builds an STRtree.

        Args:
            gtfs_manager: GTFSManager with loaded feed data
            modes_config: The 'modes' section from config.json
        """
        logger.info("Building GTFS spatial indices...")
        t0 = time.time()

        for mode_name, mode_cfg in modes_config.items():
            if not isinstance(mode_cfg, dict):
                continue
            if not mode_cfg.get('enabled', True):
                continue
            avail = mode_cfg.get('availability', 'universal')
            if not isinstance(avail, dict) or avail.get('type') != 'gtfs':
                continue

            route_types = avail.get('route_types', [])
            if not route_types:
                logger.warning(f"Mode '{mode_name}' has GTFS availability but no route_types configured")
                continue

            try:
                mode_type = ModeType(mode_name.lower())
            except ValueError:
                logger.warning(f"Unknown mode type '{mode_name}', skipping")
                continue

            # Query stops for these route types
            t_mode = time.time()
            stops_df = gtfs_manager.get_stops_by_route_types(route_types)

            if stops_df.empty:
                logger.warning(f"No stops found for mode '{mode_name}' "
                             f"(route_types={route_types})")
                self._stop_counts[mode_type] = 0
                continue

            # Deduplicate stops (same stop may serve multiple routes of same type)
            stops_df = stops_df.drop_duplicates(subset=['stop_pk'])

            # Build STRtree from stop points
            points = [Point(row.lon, row.lat) for _, row in stops_df.iterrows()]
            tree = STRtree(points)

            self._indices[mode_type] = tree
            self._stop_points[mode_type] = np.array(
                [(row.lon, row.lat) for _, row in stops_df.iterrows()]
            )
            self._stop_counts[mode_type] = len(points)

            elapsed_mode = time.time() - t_mode
            logger.info(f"  Built index for {mode_name}: {len(points)} stops "
                       f"(route_types={route_types}, took {elapsed_mode:.2f}s)")

        elapsed = time.time() - t0
        total_stops = sum(self._stop_counts.values())
        logger.info(f"GTFS spatial indices built: {len(self._indices)} modes, "
                    f"{total_stops} total stops (took {elapsed:.1f}s)")

    def has_stops_nearby(self, mode_type: ModeType, lat: float, lon: float,
                         buffer_meters: float = 800) -> bool:
        """
        Check if there are transit stops of the given mode near a location.

        Converts the buffer distance from meters to approximate degrees,
        then queries the R-tree for any stops within that buffer.

        Args:
            mode_type: Transit mode to check
            lat: Latitude of the location
            lon: Longitude of the location
            buffer_meters: Search radius in meters

        Returns:
            True if at least one stop is within buffer distance
        """
        tree = self._indices.get(mode_type)
        if tree is None:
            return False

        # Convert buffer from meters to approximate degrees
        # 1 degree latitude ≈ 111,320 meters
        # 1 degree longitude ≈ 111,320 * cos(latitude) meters
        buffer_lat = buffer_meters / 111_320
        buffer_lon = buffer_meters / (111_320 * max(cos(radians(lat)), 0.01))

        # Create search envelope (point buffered to rectangle)
        query_point = Point(lon, lat)
        search_envelope = query_point.buffer(max(buffer_lat, buffer_lon))

        # Query R-tree
        result_indices = tree.query(search_envelope)

        return len(result_indices) > 0

    def has_mode(self, mode_type: ModeType) -> bool:
        """Check if we have an index for the given mode."""
        return mode_type in self._indices

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about loaded indices."""
        return {
            'modes_indexed': len(self._indices),
            'stop_counts': dict(self._stop_counts),
            'total_stops': sum(self._stop_counts.values()),
        }
