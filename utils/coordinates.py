"""
Coordinate conversion utilities for MATSim plan generation.

Handles conversions between geographic (lat/lon) and projected (UTM) coordinate systems.
"""

from typing import Tuple, List
from pyproj import Transformer, CRS
import logging

logger = logging.getLogger(__name__)


# NAD83 UTM EPSG codes for US zones (zones 10-19 cover CONUS)
# NAD83 is the standard datum for US survey/Census data
_NAD83_UTM_EPSG = {
    10: "EPSG:26910",  # West coast (CA, OR, WA)
    11: "EPSG:26911",  # NV, ID, OR
    12: "EPSG:26912",  # AZ, UT, MT, ID
    13: "EPSG:26913",  # CO, NM, WY, MT
    14: "EPSG:26914",  # TX, NE, SD, ND, KS, OK
    15: "EPSG:26915",  # MN, WI, IA, MO, LA
    16: "EPSG:26916",  # MI, IN, IL, TN, AL, MS
    17: "EPSG:26917",  # OH, WV, VA, NC, SC, GA, FL
    18: "EPSG:26918",  # NY, PA, NJ, MD, DE, CT, VT
    19: "EPSG:26919",  # ME, NH, MA
}


def detect_utm_epsg(lat: float, lon: float) -> str:
    """
    Detect the appropriate NAD83 UTM EPSG code from geographic coordinates.

    Uses the standard UTM zone formula and maps to NAD83 EPSG codes
    (EPSG:269xx series), which is the standard datum for US data.

    Args:
        lat: Latitude in decimal degrees (WGS84)
        lon: Longitude in decimal degrees (WGS84)

    Returns:
        EPSG code string (e.g., 'EPSG:26915' for UTM Zone 15N)

    Raises:
        ValueError: If the UTM zone is outside the US range (10-19)
    """
    zone_number = int((lon + 180) / 6) + 1

    if zone_number not in _NAD83_UTM_EPSG:
        raise ValueError(
            f"UTM zone {zone_number} (from lon={lon:.2f}) is outside the supported "
            f"US range (zones 10-19). Only US counties are supported."
        )

    epsg_code = _NAD83_UTM_EPSG[zone_number]
    return epsg_code


class CoordinateConverter:
    """
    Coordinate converter between geographic (WGS84) and projected (UTM NAD83) systems.
    """

    def __init__(self, utm_epsg: str):
        """
        Initialize coordinate converter.

        Args:
            utm_epsg: EPSG code for UTM projection (e.g., 'EPSG:26915')
        """
        self.utm_epsg = utm_epsg
        self.geographic_epsg = "EPSG:4326"  # WGS84 lat/lon

        # Create transformers
        self.to_utm = Transformer.from_crs(
            self.geographic_epsg,
            self.utm_epsg,
            always_xy=True  # Ensures (lon, lat) input order
        )

        self.to_latlon = Transformer.from_crs(
            self.utm_epsg,
            self.geographic_epsg,
            always_xy=True  # Ensures (x, y) input order
        )

        logger.info(f"Initialized coordinate converter: {self.geographic_epsg} <-> {self.utm_epsg}")

    def latlon_to_utm(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert lat/lon to UTM coordinates.

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees

        Returns:
            Tuple of (x, y) in UTM meters

        Example:
            >>> converter = CoordinateConverter("EPSG:26915")
            >>> x, y = converter.latlon_to_utm(44.9778, -93.2650)  # Minneapolis
            >>> print(f"UTM: ({x:.2f}, {y:.2f})")
            UTM: (478237.51, 4980517.14)
        """
        x, y = self.to_utm.transform(lon, lat)
        return (x, y)

    def utm_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert UTM coordinates to lat/lon.

        Args:
            x: Easting in meters
            y: Northing in meters

        Returns:
            Tuple of (lat, lon) in decimal degrees

        Example:
            >>> converter = CoordinateConverter("EPSG:26915")
            >>> lat, lon = converter.utm_to_latlon(478237.51, 4980517.14)
            >>> print(f"Lat/Lon: ({lat:.4f}, {lon:.4f})")
            Lat/Lon: (44.9778, -93.2650)
        """
        lon, lat = self.to_latlon.transform(x, y)
        return (lat, lon)

    def batch_latlon_to_utm(self, coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Convert multiple lat/lon coordinates to UTM.

        Args:
            coords: List of (lat, lon) tuples

        Returns:
            List of (x, y) tuples in UTM
        """
        if not coords:
            return []

        lats, lons = zip(*coords)
        xs, ys = self.to_utm.transform(lons, lats)
        return list(zip(xs, ys))

    def batch_utm_to_latlon(self, coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Convert multiple UTM coordinates to lat/lon.

        Args:
            coords: List of (x, y) tuples in UTM

        Returns:
            List of (lat, lon) tuples
        """
        if not coords:
            return []

        xs, ys = zip(*coords)
        lons, lats = self.to_latlon.transform(xs, ys)
        return list(zip(lats, lons))

    def format_utm(self, x: float, y: float, precision: int = 6) -> Tuple[str, str]:
        """
        Format UTM coordinates as strings with specified precision.

        Args:
            x: Easting in meters
            y: Northing in meters
            precision: Number of decimal places

        Returns:
            Tuple of (x_str, y_str) formatted coordinates

        Example:
            >>> converter = CoordinateConverter("EPSG:26915")
            >>> x_str, y_str = converter.format_utm(478237.514, 4980517.135, precision=2)
            >>> print(f"X={x_str}, Y={y_str}")
            X=478237.51, Y=4980517.14
        """
        return (f"{x:.{precision}f}", f"{y:.{precision}f}")


def validate_utm_coordinates(x: float, y: float,
                             x_range: Tuple[float, float] = (400000, 600000),
                             y_range: Tuple[float, float] = (4900000, 5100000)) -> bool:
    """
    Validate UTM coordinates are within expected range for Twin Cities.

    Args:
        x: Easting in meters
        y: Northing in meters
        x_range: Valid range for x (easting)
        y_range: Valid range for y (northing)

    Returns:
        True if coordinates are valid, False otherwise

    Example:
        >>> validate_utm_coordinates(478237, 4980517)  # Minneapolis
        True
        >>> validate_utm_coordinates(1000, 2000)  # Invalid
        False
    """
    return (x_range[0] <= x <= x_range[1] and
            y_range[0] <= y <= y_range[1])


def validate_latlon_coordinates(lat: float, lon: float,
                                lat_range: Tuple[float, float] = (44.0, 46.0),
                                lon_range: Tuple[float, float] = (-95.0, -92.0)) -> bool:
    """
    Validate lat/lon coordinates are within expected range for Twin Cities.

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        lat_range: Valid range for latitude
        lon_range: Valid range for longitude

    Returns:
        True if coordinates are valid, False otherwise

    Example:
        >>> validate_latlon_coordinates(44.9778, -93.2650)  # Minneapolis
        True
        >>> validate_latlon_coordinates(0, 0)  # Invalid
        False
    """
    return (lat_range[0] <= lat <= lat_range[1] and
            lon_range[0] <= lon <= lon_range[1])


# Convenience functions for quick conversions
_default_converter = None


def get_converter(utm_epsg: str) -> CoordinateConverter:
    """Get or create default coordinate converter."""
    global _default_converter
    if _default_converter is None or _default_converter.utm_epsg != utm_epsg:
        _default_converter = CoordinateConverter(utm_epsg)
    return _default_converter


if __name__ == "__main__":
    # Test conversions
    print("=" * 60)
    print("Coordinate Conversion Tests")
    print("=" * 60)

    # Test auto-detection
    minneapolis_lat, minneapolis_lon = 44.9778, -93.2650
    detected = detect_utm_epsg(minneapolis_lat, minneapolis_lon)
    print(f"\nDetected EPSG for Minneapolis: {detected}")

    converter = CoordinateConverter(detected)

    print(f"Minneapolis (lat/lon): ({minneapolis_lat}, {minneapolis_lon})")

    x, y = converter.latlon_to_utm(minneapolis_lat, minneapolis_lon)
    print(f"Minneapolis (UTM): ({x:.2f}, {y:.2f})")

    lat, lon = converter.utm_to_latlon(x, y)
    print(f"Round-trip (lat/lon): ({lat:.4f}, {lon:.4f})")

    # Test validation
    print(f"\nValidation:")
    print(f"  UTM valid: {validate_utm_coordinates(x, y)}")
    print(f"  Lat/Lon valid: {validate_latlon_coordinates(lat, lon)}")

    # Test batch conversion
    test_coords = [
        (44.9778, -93.2650),  # Minneapolis
        (44.9537, -93.0900),  # St. Paul
        (44.8800, -93.2175),  # Bloomington
    ]

    print(f"\nBatch conversion of {len(test_coords)} coordinates:")
    utm_coords = converter.batch_latlon_to_utm(test_coords)
    for i, (orig, utm) in enumerate(zip(test_coords, utm_coords)):
        print(f"  {i+1}. {orig} -> {utm}")
