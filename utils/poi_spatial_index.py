"""
POI Spatial Index Utility

Reusable R-tree based spatial index for fast POI lookups.
Can work with in-memory POI data (for multiprocessing) or database queries.
"""
from typing import Dict, List, Optional, Tuple
from math import radians, sin, cos, sqrt, atan2
from rtree import index


class POISpatialIndex:
    """
    R-tree based spatial index for POI searches.

    Provides fast nearest-neighbor searches using R-tree spatial indexing
    with exact haversine distance verification.
    """

    def __init__(self, poi_data: Dict[str, List[Dict]]):
        """
        Initialize spatial index with POI data.

        Args:
            poi_data: Dict mapping activity type to list of POI dicts.
                     Each POI dict must have 'lat', 'lon' keys.
        """
        self.poi_data = poi_data
        self.indices = {}
        self._build_indices()

    def _build_indices(self):
        """Build R-tree spatial indices for each activity type."""
        for activity, pois in self.poi_data.items():
            if not pois:
                continue

            # Create R-tree index
            idx = index.Index()

            # Insert each POI into the index
            for poi_idx, poi in enumerate(pois):
                lon, lat = poi['lon'], poi['lat']
                # Insert as point: (lon, lat, lon, lat)
                idx.insert(poi_idx, (lon, lat, lon, lat))

            self.indices[activity] = idx

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """
        Calculate distance between two points in meters using haversine formula.

        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates

        Returns:
            Distance in meters
        """
        R = 6371000  # Earth radius in meters
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    def find_nearest(self, lat: float, lon: float, activity: str,
                    radius_m: float) -> Optional[Dict]:
        """
        Find nearest POI of given activity within radius.

        Args:
            lat, lon: Query coordinates
            activity: Activity type to search for
            radius_m: Search radius in meters

        Returns:
            Nearest POI dict or None if not found
        """
        if activity not in self.indices:
            return None

        idx = self.indices[activity]
        poi_list = self.poi_data[activity]

        # Convert radius from meters to degrees (approximate)
        radius_lat = radius_m / 111000.0
        radius_lon = radius_m / (111000.0 * abs(cos(radians(lat))))

        # Query R-tree for POIs within bounding box
        bbox = (
            lon - radius_lon,
            lat - radius_lat,
            lon + radius_lon,
            lat + radius_lat
        )

        nearby_indices = list(idx.intersection(bbox))

        if not nearby_indices:
            return None

        # Calculate exact distances for candidates
        candidates = []
        for poi_idx in nearby_indices:
            poi = poi_list[poi_idx]
            dist = self._haversine_distance(lat, lon, poi['lat'], poi['lon'])

            if dist <= radius_m:
                candidates.append((dist, poi))

        if not candidates:
            return None

        # Return nearest POI
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def find_nearest_n(self, lat: float, lon: float, activity: str,
                      radius_m: float, limit: int = 5) -> List[Tuple[float, Dict]]:
        """
        Find N nearest POIs of given activity within radius.

        Args:
            lat, lon: Query coordinates
            activity: Activity type to search for
            radius_m: Search radius in meters
            limit: Maximum number of results

        Returns:
            List of (distance, poi_dict) tuples, sorted by distance
        """
        if activity not in self.indices:
            return []

        idx = self.indices[activity]
        poi_list = self.poi_data[activity]

        # Convert radius from meters to degrees (approximate)
        radius_lat = radius_m / 111000.0
        radius_lon = radius_m / (111000.0 * abs(cos(radians(lat))))

        # Query R-tree for POIs within bounding box
        bbox = (
            lon - radius_lon,
            lat - radius_lat,
            lon + radius_lon,
            lat + radius_lat
        )

        nearby_indices = list(idx.intersection(bbox))

        if not nearby_indices:
            return []

        # Calculate exact distances for candidates
        candidates = []
        for poi_idx in nearby_indices:
            poi = poi_list[poi_idx]
            dist = self._haversine_distance(lat, lon, poi['lat'], poi['lon'])

            if dist <= radius_m:
                candidates.append((dist, poi))

        # Sort by distance and limit
        candidates.sort(key=lambda x: x[0])
        return candidates[:limit]

    def find_within_radius(self, lat: float, lon: float, radius_m: float) -> List[Dict]:
        """
        Find all POIs within radius (any activity type).

        Args:
            lat, lon: Query coordinates
            radius_m: Search radius in meters

        Returns:
            List of POI dicts within radius
        """
        results = []

        # Search across all activity types
        for activity in self.indices.keys():
            if activity not in self.indices:
                continue

            idx = self.indices[activity]
            poi_list = self.poi_data[activity]

            # Convert radius from meters to degrees (approximate)
            radius_lat = radius_m / 111000.0
            radius_lon = radius_m / (111000.0 * abs(cos(radians(lat))))

            # Query R-tree for POIs within bounding box
            bbox = (
                lon - radius_lon,
                lat - radius_lat,
                lon + radius_lon,
                lat + radius_lat
            )

            nearby_indices = list(idx.intersection(bbox))

            # Calculate exact distances and filter
            for poi_idx in nearby_indices:
                poi = poi_list[poi_idx]
                dist = self._haversine_distance(lat, lon, poi['lat'], poi['lon'])

                if dist <= radius_m:
                    results.append(poi)

        return results

    def get_stats(self) -> Dict:
        """Get statistics about the spatial index."""
        return {
            'num_activities': len(self.indices),
            'activities': list(self.indices.keys()),
            'total_pois': sum(len(pois) for pois in self.poi_data.values()),
            'pois_per_activity': {
                activity: len(pois)
                for activity, pois in self.poi_data.items()
            }
        }
