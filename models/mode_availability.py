"""
Mode availability checking for multi-modal transportation.

This module provides classes to determine which transportation modes
are available for a given origin-destination pair.

Phase 3 updates:
- DistanceBasedAvailability: real haversine distance calculation
- GTFSTransitAvailability: delegates to GTFSAvailabilityManager R-tree
- ModeAvailabilityManager: accepts optional GTFSAvailabilityManager
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from typing import Any, Dict, Optional, Set

from models.mode_types import ModeType, ModeConfig

logger = logging.getLogger(__name__)


@dataclass
class Location:
    """Geographic location for availability checks."""
    lat: float
    lon: float
    block_group_id: Optional[str] = None


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.

    Args:
        lat1, lon1: Coordinates of point 1 (degrees)
        lat2, lon2: Coordinates of point 2 (degrees)

    Returns:
        Distance in meters
    """
    R = 6_371_000  # Earth radius in meters
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


class ModeAvailabilityChecker(ABC):
    """Abstract base class for mode availability checking."""

    @abstractmethod
    def is_available(self, origin: Location, destination: Location) -> bool:
        """
        Check if mode is available for this origin-destination pair.

        Args:
            origin: Origin location
            destination: Destination location

        Returns:
            True if mode is available, False otherwise
        """
        pass

    @abstractmethod
    def get_mode_type(self) -> ModeType:
        """Return the mode type this checker handles."""
        pass


class UniversalAvailability(ModeAvailabilityChecker):
    """
    Mode is always available regardless of location.

    Used for car mode which has no geographic restrictions.
    """

    def __init__(self, mode_type: ModeType):
        self._mode_type = mode_type

    def is_available(self, origin: Location, destination: Location) -> bool:
        return True

    def get_mode_type(self) -> ModeType:
        return self._mode_type


class DistanceBasedAvailability(ModeAvailabilityChecker):
    """
    Mode available within maximum haversine distance.

    Used for walk and bike modes.
    """

    def __init__(self, mode_type: ModeType, max_distance_meters: float):
        self._mode_type = mode_type
        self.max_distance_meters = max_distance_meters

    def is_available(self, origin: Location, destination: Location) -> bool:
        """Check if haversine distance between origin and destination is within limit."""
        distance = haversine_meters(origin.lat, origin.lon, destination.lat, destination.lon)
        return distance <= self.max_distance_meters

    def get_mode_type(self) -> ModeType:
        return self._mode_type


class GTFSTransitAvailability(ModeAvailabilityChecker):
    """
    Mode available if GTFS stops exist near both origin and destination.

    Delegates to GTFSAvailabilityManager for R-tree spatial queries.
    """

    def __init__(self, mode_type: ModeType, gtfs_avail_manager,
                 access_buffer_meters: float):
        """
        Args:
            mode_type: The transit mode (BUS, RAIL)
            gtfs_avail_manager: GTFSAvailabilityManager with built indices
            access_buffer_meters: Max walking distance to a stop (meters)
        """
        self._mode_type = mode_type
        self._gtfs_avail = gtfs_avail_manager
        self.access_buffer_meters = access_buffer_meters

    def is_available(self, origin: Location, destination: Location) -> bool:
        """Check if transit stops exist near both origin and destination."""
        if self._gtfs_avail is None:
            return False

        origin_ok = self._gtfs_avail.has_stops_nearby(
            self._mode_type, origin.lat, origin.lon, self.access_buffer_meters
        )
        if not origin_ok:
            return False

        dest_ok = self._gtfs_avail.has_stops_nearby(
            self._mode_type, destination.lat, destination.lon, self.access_buffer_meters
        )
        return dest_ok

    def get_mode_type(self) -> ModeType:
        return self._mode_type


class ZoneListAvailability(ModeAvailabilityChecker):
    """
    Mode available only in specified zones.

    Used for rideshare, scooter services with limited coverage areas.
    Placeholder for future implementation.
    """

    def __init__(self, mode_type: ModeType, zones: list):
        self._mode_type = mode_type
        self.allowed_zones = set(zones)

    def is_available(self, origin: Location, destination: Location) -> bool:
        """Check if both locations are in allowed zones. Placeholder - returns True."""
        # TODO: Check block_group_id against allowed_zones
        return True

    def get_mode_type(self) -> ModeType:
        return self._mode_type


class ModeAvailabilityManager:
    """
    Manages all mode availability checkers.

    Initialized from config, creates appropriate checker for each mode.
    Accepts optional GTFSAvailabilityManager for transit modes.
    """

    def __init__(self, modes_config: Dict[str, Any],
                 gtfs_avail_manager=None):
        """
        Initialize availability checkers for each configured mode.

        Args:
            modes_config: The 'modes' section from config.json
            gtfs_avail_manager: Optional GTFSAvailabilityManager for transit modes.
                               If None, GTFS modes fall back to universal availability.
        """
        self.checkers: Dict[ModeType, ModeAvailabilityChecker] = {}
        self._gtfs_avail_manager = gtfs_avail_manager

        for mode_name, mode_cfg in modes_config.items():
            if not isinstance(mode_cfg, dict):
                continue
            if not mode_cfg.get('enabled', True):
                continue
            mode_config = ModeConfig.from_config(mode_name, mode_cfg)
            checker = self._create_checker(mode_config)
            self.checkers[mode_config.mode_type] = checker
            logger.debug(f"Registered availability checker for {mode_name}: {type(checker).__name__}")

    def _create_checker(self, config: ModeConfig) -> ModeAvailabilityChecker:
        """Create appropriate checker based on availability type."""
        avail_type = config.availability_type
        params = config.availability_params
        mode_type = config.mode_type

        if avail_type == 'universal':
            return UniversalAvailability(mode_type)

        elif avail_type == 'distance':
            return DistanceBasedAvailability(
                mode_type,
                max_distance_meters=params.get('max_distance_meters', 5000)
            )

        elif avail_type == 'gtfs':
            if self._gtfs_avail_manager is not None and self._gtfs_avail_manager.has_mode(mode_type):
                return GTFSTransitAvailability(
                    mode_type,
                    gtfs_avail_manager=self._gtfs_avail_manager,
                    access_buffer_meters=params.get('access_buffer_meters', 800),
                )
            else:
                # GTFS data not available — fall back to universal
                logger.warning(f"GTFS data not available for {mode_type.value}, "
                             f"falling back to universal availability")
                return UniversalAvailability(mode_type)

        elif avail_type == 'zone_list':
            return ZoneListAvailability(
                mode_type,
                zones=params.get('zones', [])
            )

        else:
            logger.warning(f"Unknown availability type '{avail_type}' for {mode_type}, defaulting to universal")
            return UniversalAvailability(mode_type)

    def get_available_modes(self, origin: Location, destination: Location) -> Set[ModeType]:
        """
        Get all modes available for this origin-destination pair.

        Args:
            origin: Origin location
            destination: Destination location

        Returns:
            Set of available ModeType values
        """
        available = set()
        for mode_type, checker in self.checkers.items():
            if checker.is_available(origin, destination):
                available.add(mode_type)
        return available

    def is_mode_available(self, mode_type: ModeType, origin: Location,
                          destination: Location) -> bool:
        """
        Check if a specific mode is available for this OD pair.

        Args:
            mode_type: Mode to check
            origin: Origin location
            destination: Destination location

        Returns:
            True if mode is available
        """
        checker = self.checkers.get(mode_type)
        if checker is None:
            return False
        return checker.is_available(origin, destination)
