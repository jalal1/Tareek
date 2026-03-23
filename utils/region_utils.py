"""
Region utilities for handling county-based filtering across the pipeline.

Provides helpers to work with county GEOIDs from config.

IMPORTANT: Before using this module, you must run the setup notebook:
    notebooks/0.setup_global_data.ipynb
This notebook populates the database with all US states and counties.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from utils.duckdb_manager import DBManager

logger = logging.getLogger(__name__)

# Census TIGER cartographic boundary shapefile
_COUNTY_SHP_URL = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_500k.zip"
_COUNTY_SHP_DIR = "counties"
_COUNTY_SHP_FILE = "cb_2022_us_county_500k.shp"


def load_county_polygons(county_geoids: List[str],
                         data_dir: str,
                         utm_epsg: Optional[str] = None):
    """
    Load county boundary polygons from the Census TIGER shapefile.

    Downloads the shapefile on first use to {data_dir}/counties/.
    Filters by FIPS GEOID and optionally projects to UTM.

    Args:
        county_geoids: List of 5-character FIPS GEOIDs (e.g. ["27003", "27053"])
        data_dir: Path to the project data directory
        utm_epsg: If provided (e.g. "EPSG:32615"), project polygons to this CRS.
                  If None, returns polygons in WGS84 (EPSG:4326).

    Returns:
        List of Shapely polygon/multipolygon geometries, one per county.
        Empty list on failure.
    """
    import geopandas as gpd

    data_path = Path(data_dir)
    shp_dir = data_path / _COUNTY_SHP_DIR
    shp_file = shp_dir / _COUNTY_SHP_FILE

    # Download if not cached
    if not shp_file.exists():
        try:
            import requests
            from io import BytesIO
            import zipfile

            logger.info(f"Downloading US county boundaries to {shp_dir}...")
            shp_dir.mkdir(parents=True, exist_ok=True)
            response = requests.get(_COUNTY_SHP_URL, timeout=120)
            response.raise_for_status()
            z = zipfile.ZipFile(BytesIO(response.content))
            z.extractall(shp_dir)
            logger.info("County shapefile downloaded and extracted")
        except Exception as e:
            logger.warning(f"Failed to download county shapefile: {e}")
            return []

    # Load and filter by FIPS
    try:
        gdf = gpd.read_file(shp_file)
        gdf["FIPS"] = gdf["STATEFP"] + gdf["COUNTYFP"]
        selected = gdf[gdf["FIPS"].isin(county_geoids)]

        if selected.empty:
            logger.warning(f"No counties matched GEOIDs: {county_geoids}")
            return []

        logger.info(f"Loaded {len(selected)} county boundary polygons")

        # Project to UTM if requested
        if utm_epsg:
            selected = selected.to_crs(utm_epsg)

        return list(selected.geometry)

    except Exception as e:
        logger.warning(f"Failed to load county polygons: {e}")
        return []


class RegionHelper:
    """
    Helper class for working with county GEOIDs from config.

    All state and county data must be pre-populated in the database.
    Run notebooks/0.setup_global_data.ipynb before using this class.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RegionHelper with config.

        Args:
            config: Configuration dictionary with 'region.counties' section
                   counties should be a list of 5-character GEOIDs (state_fips + county_fips)
        """
        self.config = config
        self.county_geoids = config['region']['counties']  # List of GEOIDs like ["27003", "27053", ...]
        self.data_dir = config['data']['data_dir']
        self.db_manager = DBManager(self.data_dir)

    def get_county_fips_mapping(self) -> Dict[str, List[str]]:
        """
        Get state -> county FIPS mapping for counties in config.

        Returns:
            Dict mapping state FIPS to list of county FIPS codes
            Format: {'27': ['003', '053', ...], '55': ['093', '109']}
        """
        # Group counties by state FIPS
        fips_mapping: Dict[str, List[str]] = {}

        for geoid in self.county_geoids:
            # GEOID is 5 characters: first 2 are state FIPS, last 3 are county FIPS
            state_fips = geoid[:2]
            county_fips = geoid[2:5]

            if state_fips not in fips_mapping:
                fips_mapping[state_fips] = []
            fips_mapping[state_fips].append(county_fips)

        logger.info(f"Loaded {len(self.county_geoids)} counties across {len(fips_mapping)} state(s)")
        return fips_mapping

    def get_state_abbr_mapping(self) -> Dict[str, str]:
        """
        Get mapping of state FIPS to state abbreviation.

        Returns:
            Dict mapping state FIPS to state abbreviation
            Format: {'27': 'MN', '55': 'WI'}
        """
        from models.models import State

        with self.db_manager.Session() as session:
            # Get unique state FIPS from county GEOIDs
            state_fips_set = {geoid[:2] for geoid in self.county_geoids}

            # Query database for state abbreviations
            states = session.query(State).filter(State.state_fips.in_(state_fips_set)).all()

            return {s.state_fips: s.state_abbr for s in states}

    def get_county_names_for_network(self) -> List[tuple[str, str]]:
        """
        Get county names and state names for network generation (OSM download).

        Returns:
            List of (county_name, state_name) tuples
            Format: [("Hennepin County", "Minnesota"), ...]
        """
        from models.models import County, State

        with self.db_manager.Session() as session:
            # Build list of (state_fips, county_fips) tuples from GEOIDs
            county_parts = [(geoid[:2], geoid[2:5]) for geoid in self.county_geoids]

            result = []
            for state_fips, county_fips in county_parts:
                # Query for county and state info
                county = session.query(County, State).join(
                    State, County.state_fips == State.state_fips
                ).filter(
                    County.state_fips == state_fips,
                    County.county_fips == county_fips
                ).first()

                if county:
                    county_obj, state_obj = county
                    # Use full name from Census NAMELSAD (e.g., "Hennepin County", "Orleans Parish", "District of Columbia")
                    county_name = county_obj.county_name_full or (county_obj.county_name + " County")
                    state_name = state_obj.state_name
                    result.append((county_name, state_name))
                    logger.debug(f"  {county_name}, {state_name}")
                else:
                    logger.warning(f"County not found in database: {state_fips}{county_fips}")

            logger.info(f"Retrieved {len(result)} county names for network generation")
            return result
