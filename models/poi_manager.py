import json
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import overpy
from sqlalchemy import Column, String, Float, Integer, DateTime
from sqlalchemy.orm import declarative_base

from utils.duckdb_manager import DBManager
from utils.logger import setup_logger, CHECK_MARK
from utils.poi_spatial_index import POISpatialIndex
from models.models import POI

logger = setup_logger(__name__)

# ============================================================================
# ACTIVITY TO OSM TAG MAPPING
# ============================================================================

ACTIVITY_OSM_TAGS = {
    'Dining': {
        'amenity': ['restaurant', 'cafe', 'fast_food', 'bar', 'pub', 'bakery']
    },
    'School': {
        'amenity': ['school', 'university', 'college'],
        'building': ['school']
    },
    'Shopping': {
        'amenity': ['supermarket', 'grocery', 'shop', 'mall', 'pharmacy', 'bank', 'post_office'],
        'shop': ['supermarket', 'general', 'mall', 'clothes', 'furniture', 'electronics']
    },
    'Social': {
        'amenity': ['cinema', 'theatre', 'library', 'community_centre', 'bar', 'nightclub'],
        'leisure': ['park', 'playground', 'sports_centre', 'swimming_pool', 'gym']
    },
    'Escort': {
        'amenity': ['taxi', 'car_rental'],
        'tourism': ['hotel', 'guest_house', 'hostel']
    },
    'Other': {
        'amenity': ['parking', 'hospital', 'clinic']
    }
}

# ============================================================================
# POI MANAGER
# ============================================================================

class POIManager:
    def __init__(self,
                 db_manager: DBManager,
                 overpass_url: str = 'http://overpass-api.de/api/interpreter'):
        """
        Initialize POI Manager with existing DBManager.

        Args:
            db_manager: Existing DBManager instance
            overpass_url: Overpass API URL
        """
        self.db_manager = db_manager
        self.overpass = overpy.Overpass(url=overpass_url)
        # Set retry parameters for large queries
        overpy.Overpass.default_max_retry_count = 5
        overpy.Overpass.default_retry_timeout = 2.0

        # Spatial index for fast POI lookups (built on-demand)
        self._spatial_index = None
        self._spatial_index_cache_key = None

        logger.info("POIManager initialized")
    
    # ========================================================================
    # OSM QUERYING & PROCESSING
    # ========================================================================
    
    def _build_overpass_query(self, counties: List[str], state: str = "Minnesota",
                              fips_codes: Optional[List[str]] = None,
                              county_names_full: Optional[List[str]] = None) -> str:
        """Build Overpass query for counties in a specific state.

        Builds queries with specific tag=value pairs to avoid pulling too much data.
        Only queries for the specific amenity/shop/leisure types we care about.

        Args:
            counties: List of county names (e.g., ['Anoka', 'Hennepin'])
            state: State name (default: 'Minnesota')
            fips_codes: Optional list of full FIPS codes (e.g., ['27003', '27053']) for precise matching
            county_names_full: Optional list of full names from Census NAMELSAD
                (e.g., ['Hennepin County', 'Orleans Parish', 'District of Columbia'])
        """
        # Build specific tag=value filters from ACTIVITY_OSM_TAGS
        tag_filters = []
        for activity, tag_map in ACTIVITY_OSM_TAGS.items():
            for key, values in tag_map.items():
                for value in values:
                    tag_filters.append((key, value))

        # Remove duplicates
        tag_filters = list(set(tag_filters))

        # Prepare area clauses for each county
        area_lines = []
        for idx, c in enumerate(counties):
            # Use full name from Census if available (handles parishes, boroughs, DC, etc.)
            if county_names_full and idx < len(county_names_full):
                name = county_names_full[idx]
            else:
                name = f'{c.title()} County'
            # Use FIPS code if provided for precise matching (is_in:state tag doesn't exist in OSM)
            if fips_codes and idx < len(fips_codes):
                area_lines.append(f'  area["name"="{name}"]["admin_level"="6"]["nist:fips_code"="{fips_codes[idx]}"];')
            else:
                # Fallback to name-only matching (less reliable for counties with same names in different states)
                area_lines.append(f'  area["name"="{name}"]["admin_level"="6"];')

        area_block = "\n".join(area_lines)

        # Build specific tag queries (much more efficient than querying all with a key)
        element_queries = []
        for key, value in tag_filters:
            element_queries.append(f'  node(area.boundary)["{key}"="{value}"];')
            element_queries.append(f'  way(area.boundary)["{key}"="{value}"];')

        elements_block = "\n".join(element_queries)

        return f"""
[out:json][timeout:180];
(
{area_block}
)->.boundary;
(
{elements_block}
);
out center;
"""
    
    def _build_overpass_query_wikipedia(self, county_name_full: str, state_name: str) -> str:
        """Build Overpass query using Wikipedia tag for counties without FIPS codes in OSM.

        Args:
            county_name_full: Full county name from Census NAMELSAD (e.g., 'Jefferson County', 'District of Columbia')
            state_name: State name (e.g., 'Alabama')

        Returns:
            Overpass query string
        """
        # Build specific tag=value filters from ACTIVITY_OSM_TAGS
        tag_filters = []
        for activity, tag_map in ACTIVITY_OSM_TAGS.items():
            for key, values in tag_map.items():
                for value in values:
                    tag_filters.append((key, value))

        # Remove duplicates
        tag_filters = list(set(tag_filters))

        # Build specific tag queries
        element_queries = []
        for key, value in tag_filters:
            element_queries.append(f'  node(area.boundary)["{key}"="{value}"];')
            element_queries.append(f'  way(area.boundary)["{key}"="{value}"];')

        elements_block = "\n".join(element_queries)

        # Use Wikipedia tag format: "en:County Name, State Name"
        wikipedia_tag = f"en:{county_name_full}, {state_name}"

        return f"""
[out:json][timeout:180];
area["wikipedia"="{wikipedia_tag}"]->.boundary;
(
{elements_block}
);
out center;
"""

    def _get_county_bbox(self, county_name_full: str, state_name: str) -> Optional[Tuple[float, float, float, float]]:
        """Get bounding box for a county using OSMnx geocoding.

        Args:
            county_name_full: Full county name from Census NAMELSAD (e.g., 'McHenry County', 'District of Columbia')
            state_name: State name (e.g., 'Illinois')

        Returns:
            Tuple of (south, west, north, east) or None if geocoding fails
        """
        import osmnx as ox

        try:
            query = f"{county_name_full}, {state_name}, USA"
            gdf = ox.geocode_to_gdf(query)

            if gdf.empty:
                return None

            # Get bbox from geometry bounds
            bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
            bbox = (bounds[1], bounds[0], bounds[3], bounds[2])  # (south, west, north, east)
            return bbox

        except Exception as e:
            logger.warning(f"Failed to geocode {county_name_full}, {state_name}: {e}")
            return None

    def _build_overpass_query_bbox(self, county_name_full: str, state_name: str) -> str:
        """Build Overpass query using bounding box as fallback strategy.

        This is more reliable than tag-based queries when OSM data lacks proper
        administrative tags (FIPS codes, Wikipedia tags, etc.).

        Args:
            county_name_full: Full county name from Census NAMELSAD (e.g., 'McHenry County', 'District of Columbia')
            state_name: State name (e.g., 'Illinois')

        Returns:
            Overpass query string with bbox filter
        """
        # Build specific tag=value filters from ACTIVITY_OSM_TAGS
        tag_filters = []
        for activity, tag_map in ACTIVITY_OSM_TAGS.items():
            for key, values in tag_map.items():
                for value in values:
                    tag_filters.append((key, value))

        # Remove duplicates
        tag_filters = list(set(tag_filters))

        # Get bounding box for county
        bbox = self._get_county_bbox(county_name_full, state_name)
        if not bbox:
            raise Exception(f"Could not determine bounding box for {county_name_full}, {state_name}")

        south, west, north, east = bbox
        logger.info(f"Using bbox for {county_name_full}: ({south:.4f}, {west:.4f}, {north:.4f}, {east:.4f})")

        # Build specific tag queries with bbox
        element_queries = []
        for key, value in tag_filters:
            element_queries.append(f'  node["{key}"="{value}"]({south},{west},{north},{east});')
            element_queries.append(f'  way["{key}"="{value}"]({south},{west},{north},{east});')

        elements_block = "\n".join(element_queries)

        return f"""
[out:json][timeout:240];
(
{elements_block}
);
out center;
"""

    def _build_city_query(self, city_name: str) -> str:
        """Build Overpass query for a city."""
        return f"""
        [bbox];
        (
          area[name="{city_name}"][admin_level~"8|9|10"];
        )->.city;
        (
          node(area.city)["amenity"];
          node(area.city)["shop"];
          node(area.city)["leisure"];
          node(area.city)["tourism"];
          way(area.city)["amenity"];
          way(area.city)["shop"];
          way(area.city)["leisure"];
        );
        out center;
        """
    
    def _match_poi_activity(self, tags: Dict) -> str:
        """Match OSM tags to activity category."""
        for activity, tag_dict in ACTIVITY_OSM_TAGS.items():
            for key, values in tag_dict.items():
                if key in tags and tags[key] in values:
                    return activity
        return 'Other'
    
    def _extract_coords(self, element) -> Optional[Tuple[float, float]]:
        """Extract lat/lon from OSM element."""
        if hasattr(element, 'lat') and hasattr(element, 'lon'):
            return (element.lat, element.lon)
        elif hasattr(element, 'center_lat') and hasattr(element, 'center_lon'):
            return (element.center_lat, element.center_lon)
        return None
    
    @staticmethod
    def _valid_coords(coords: Tuple[float, float]) -> bool:
        """Validate latitude and longitude."""
        lat, lon = coords
        return -90 <= lat <= 90 and -180 <= lon <= 180

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """Calculate distance between two points in meters."""
        from math import radians, sin, cos, sqrt, atan2

        R = 6371000  # Earth radius in meters
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    def _ensure_spatial_index(self, activity: Optional[str] = None):
        """
        Build spatial index for POIs if not already built.

        Args:
            activity: Activity type to build index for (None = all POIs)
        """
        cache_key = activity if activity else '__all__'

        # Check if we need to rebuild the index
        if self._spatial_index is not None and self._spatial_index_cache_key == cache_key:
            return  # Already built for this activity

        # Load POIs from database and organize by activity
        if activity:
            pois = self.db_manager.query_all(POI, filters={'activity': activity})
            poi_data = {activity: []}
            for poi in pois:
                poi_data[activity].append({
                    'osm_id': poi.osm_id,
                    'name': poi.name,
                    'activity': poi.activity,
                    'lat': poi.lat,
                    'lon': poi.lon,
                    'tags': json.loads(poi.tags) if poi.tags else {},
                    'source': poi.source_name
                })
        else:
            # Load all POIs and group by activity
            all_pois = self.db_manager.query_all(POI)
            poi_data = {}
            for poi in all_pois:
                if poi.activity not in poi_data:
                    poi_data[poi.activity] = []
                poi_data[poi.activity].append({
                    'osm_id': poi.osm_id,
                    'name': poi.name,
                    'activity': poi.activity,
                    'lat': poi.lat,
                    'lon': poi.lon,
                    'tags': json.loads(poi.tags) if poi.tags else {},
                    'source': poi.source_name
                })

        if not poi_data or all(len(pois) == 0 for pois in poi_data.values()):
            logger.warning(f"No POIs found for activity '{activity}'")
            return

        # Create spatial index
        self._spatial_index = POISpatialIndex(poi_data)
        self._spatial_index_cache_key = cache_key

        stats = self._spatial_index.get_stats()
        logger.info(f"Built spatial index for '{cache_key}': {stats['total_pois']} POIs across {stats['num_activities']} activities")
    
    # ========================================================================
    # PUBLIC API - FETCH & STORE
    # ========================================================================

    def _is_county_processed_by_fips(self, state_fips: str, county_fips: str) -> bool:
        """
        Check if a county has already been processed by checking if any POI exists
        with that (state_fips, county_fips) pair.

        Args:
            state_fips: State FIPS code (e.g., '27')
            county_fips: County FIPS code (e.g., '053')

        Returns:
            True if at least one POI exists for this FIPS pair
        """
        try:
            from sqlalchemy import text
            with self.db_manager.session_scope() as session:
                query = text("SELECT COUNT(*) FROM pois WHERE state_fips = :state_fips AND county_fips = :county_fips LIMIT 1")
                result = session.execute(query, {'state_fips': state_fips, 'county_fips': county_fips}).fetchone()
                count = result[0] if result else 0
                return count > 0
        except Exception as e:
            logger.warning(f"Error checking if county is processed: {e}")
            return False

    def _check_osm_ids_exist(self, osm_ids: List[str]) -> set:
        """
        Efficiently check which OSM IDs already exist in the database.

        Args:
            osm_ids: List of OSM IDs to check

        Returns:
            Set of OSM IDs that already exist
        """
        if not osm_ids:
            return set()

        try:
            from sqlalchemy import text
            with self.db_manager.session_scope() as session:
                # Use IN clause for efficient batch checking
                placeholders = ','.join([f':id{i}' for i in range(len(osm_ids))])
                query = text(f"SELECT osm_id FROM pois WHERE osm_id IN ({placeholders})")
                params = {f'id{i}': osm_id for i, osm_id in enumerate(osm_ids)}
                result = session.execute(query, params).fetchall()
                return {row[0] for row in result}
        except Exception as e:
            logger.warning(f"Error checking existing OSM IDs: {e}")
            return set()

    def fetch_pois_by_counties(self, config: Dict, wait_time: float = 8.0, max_retries: int = 3):
        """
        Fetch POIs by counties from config and store in DuckDB.

        Uses FIPS codes from config['region']['counties'] to resolve county/state names
        via the County/State DB tables, then queries Overpass API.

        Args:
            config: Configuration dict with config['region']['counties'] GEOIDs
            wait_time: Seconds to wait between API calls (rate limiting) - default 8.0s
            max_retries: Number of retries on failure
        """
        from models.models import County, State

        county_geoids = config['region']['counties']

        # Resolve FIPS → county name + state name from DB
        counties_info = []  # List of (state_fips, county_fips, county_name, state_name)
        with self.db_manager.session_scope() as session:
            for geoid in county_geoids:
                state_fips = geoid[:2]
                county_fips = geoid[2:5]

                county_row = session.query(County).filter_by(geoid=geoid).first()
                state_row = session.query(State).filter_by(state_fips=state_fips).first()

                if not county_row or not state_row:
                    logger.warning(f"Could not resolve GEOID {geoid} to county/state names, skipping")
                    continue

                county_full = county_row.county_name_full or (county_row.county_name + " County")
                counties_info.append((state_fips, county_fips, county_row.county_name, state_row.state_name, county_full))

        if not counties_info:
            logger.warning("No counties could be resolved from config. Nothing to fetch.")
            return

        # Check which counties are already processed and skip them
        counties_to_process = []
        for state_fips, county_fips, county_name, state_name, county_full in counties_info:
            if self._is_county_processed_by_fips(state_fips, county_fips):
                logger.info(f"-->  Skipping {county_name} ({state_fips}{county_fips}) - already processed")
            else:
                counties_to_process.append((state_fips, county_fips, county_name, state_name, county_full))

        if not counties_to_process:
            logger.info(f"{CHECK_MARK} All counties already processed. Nothing to fetch.")
            return

        logger.info(f"Processing {len(counties_to_process)} counties")

        # Track failed counties for summary at the end
        failed_counties = []

        # Process each county individually
        for state_fips, county_fips, county_name, state_name, county_full in counties_to_process:
            logger.info(f"Fetching POIs for county: {county_name} in {state_name} (FIPS: {state_fips}{county_fips})")
            full_fips = f"{state_fips}{county_fips}"

            # Try multiple query strategies with fallback (FIPS → Wikipedia → BBox)
            strategies = [
                ('FIPS', lambda cn=county_name, sn=state_name, ff=full_fips, cf=county_full:
                    self._build_overpass_query([cn], sn, fips_codes=[ff], county_names_full=[cf])),
                ('Wikipedia', lambda cf=county_full, sn=state_name:
                    self._build_overpass_query_wikipedia(cf, sn)),
                ('BBox', lambda cf=county_full, sn=state_name:
                    self._build_overpass_query_bbox(cf, sn)),
            ]

            result = None
            for strategy_name, query_builder in strategies:
                try:
                    query = query_builder()
                except Exception as e:
                    logger.warning(f"{strategy_name} query builder failed for {county_name}: {e}")
                    continue  # Try next strategy

                for attempt in range(max_retries):
                    try:
                        time.sleep(wait_time)
                        logger.info(f"Trying {strategy_name} query strategy (attempt {attempt + 1}/{max_retries})")
                        result = self.overpass.query(query)

                        # Check if we got results
                        total_elements = len(result.nodes) + len(result.ways)
                        if total_elements == 0:
                            logger.warning(f"{strategy_name} query returned 0 results for {county_name}")
                            break  # Try next strategy

                        logger.info(f"✓ {strategy_name} query succeeded with {total_elements} elements")
                        self._process_osm_result(result, source='county', source_names=[county_name.lower()],
                                                 state_fips=state_fips, county_fips=county_fips)
                        result = 'success'  # Mark as successful
                        break  # Success, move to next county

                    except Exception as e:
                        logger.warning(f"{strategy_name} attempt {attempt + 1}/{max_retries} failed for {county_name}: {e}")
                        if attempt < max_retries - 1:
                            retry_wait = wait_time * (2 ** attempt)  # Exponential backoff
                            logger.info(f"Retrying in {retry_wait} seconds...")
                            time.sleep(retry_wait)
                        else:
                            logger.warning(f"{strategy_name} query failed after {max_retries} retries")
                            break  # Try next strategy

                if result == 'success':
                    break  # Move to next county

            # Allow partial failures - log error but continue with other counties
            if result != 'success':
                logger.error(f"All query strategies failed for {county_name} in {state_name}")
                logger.warning(f"PARTIAL FAILURE: Skipping {county_name} - experiment will continue without POIs for this county")
                failed_counties.append(f"{county_name} ({state_fips}{county_fips})")
                # Continue to next county instead of raising

        # Log summary of failures at the end
        if failed_counties:
            logger.warning(f"POI fetch completed with {len(failed_counties)} failed counties:")
            for failed in failed_counties:
                logger.warning(f"  - {failed}")
            logger.warning("Experiment will continue with partial POI data")
        else:
            logger.info(f"{CHECK_MARK} All counties processed successfully")
    
    def fetch_pois_by_city(self, city_name: str, state_fips: str, county_fips: str, wait_time: float = 5.0):
        """
        Fetch POIs by city name and store in DuckDB.

        Args:
            city_name: City name (e.g., 'Minneapolis')
            state_fips: State FIPS code for the city's county
            county_fips: County FIPS code for the city's county
            wait_time: Seconds to wait between API calls
        """
        logger.info(f"Fetching POIs for city: {city_name}")
        query = self._build_city_query(city_name)

        try:
            time.sleep(wait_time)
            result = self.overpass.query(query)
            self._process_osm_result(result, source='city', source_names=[city_name.lower()],
                                     state_fips=state_fips, county_fips=county_fips)
        except Exception as e:
            logger.error(f"Error fetching city {city_name}: {e}")
            raise
    
    def _process_osm_result(self, result, source: str, source_names: List[str],
                            state_fips: str = '', county_fips: str = ''):
        """Process Overpass result and store in DuckDB.

        Args:
            result: Overpass API result
            source: Source type ('county' or 'city')
            source_names: List of source names
            state_fips: State FIPS code to store with each POI
            county_fips: County FIPS code to store with each POI
        """
        # Join source names for multi-county queries
        source_name_str = ','.join(source_names)

        # First, collect all OSM IDs from the result
        all_osm_ids = []
        for node in result.nodes:
            all_osm_ids.append(str(node.id))
        for way in result.ways:
            all_osm_ids.append(str(way.id))

        # Efficiently check which IDs already exist in the database
        existing_ids = self._check_osm_ids_exist(all_osm_ids)
        logger.info(f"Found {len(all_osm_ids)} OSM elements, {len(existing_ids)} already in database")

        pois_to_insert = []

        # Process nodes
        for node in result.nodes:
            osm_id = str(node.id)
            if osm_id in existing_ids:
                continue

            coords = self._extract_coords(node)
            if not coords or not self._valid_coords(coords):
                continue

            activity = self._match_poi_activity(node.tags)
            poi_name = node.tags.get('name', f'Node_{node.id}')

            pois_to_insert.append({
                'osm_id': osm_id,
                'name': poi_name,
                'activity': activity,
                'lat': coords[0],
                'lon': coords[1],
                'tags': json.dumps(dict(node.tags)),
                'source': source,
                'source_name': source_name_str,
                'state_fips': state_fips,
                'county_fips': county_fips,
                'fetched_at': datetime.now()
            })

        # Process ways
        for way in result.ways:
            osm_id = str(way.id)
            if osm_id in existing_ids:
                continue

            coords = self._extract_coords(way)
            if not coords or not self._valid_coords(coords):
                continue

            activity = self._match_poi_activity(way.tags)
            poi_name = way.tags.get('name', f'Way_{way.id}')

            pois_to_insert.append({
                'osm_id': osm_id,
                'name': poi_name,
                'activity': activity,
                'lat': coords[0],
                'lon': coords[1],
                'tags': json.dumps(dict(way.tags)),
                'source': source,
                'source_name': source_name_str,
                'state_fips': state_fips,
                'county_fips': county_fips,
                'fetched_at': datetime.now()
            })

        # Batch insert into DuckDB
        if pois_to_insert:
            try:
                self.db_manager.insert_records(POI, pois_to_insert)
                logger.info(f"{CHECK_MARK} Inserted {len(pois_to_insert)} new POIs for {source_name_str}")
            except Exception as e:
                logger.error(f"Error inserting POIs: {e}")
                raise
        else:
            logger.info(f"{CHECK_MARK} No new POIs to insert for {source_name_str} (all already exist)")
    
    # ========================================================================
    # PUBLIC API - QUERY HELPERS
    # ========================================================================
    
    def find_nearest_pois(self,
                         lat: float,
                         lon: float,
                         radius_m: float = 1000,
                         activity: Optional[str] = None,
                         limit: int = 5) -> List[Dict]:
        """
        Find nearest POIs to a location using spatial index.

        Args:
            lat, lon: Query coordinates
            radius_m: Search radius in meters
            activity: Filter by activity (None = all)
            limit: Max results to return

        Returns:
            List of POI dicts with distance in meters
        """
        try:
            # Build spatial index if not already built
            self._ensure_spatial_index(activity)

            if self._spatial_index is None:
                logger.warning(f"No POIs found for activity '{activity}'")
                return []

            # Find nearest POIs using spatial index
            if activity:
                # Single activity search
                results = self._spatial_index.find_nearest_n(lat, lon, activity, radius_m, limit)
            else:
                # Search across all activities
                all_results = []
                for act in self._spatial_index.indices.keys():
                    act_results = self._spatial_index.find_nearest_n(lat, lon, act, radius_m, limit)
                    all_results.extend(act_results)
                # Sort by distance and limit
                all_results.sort(key=lambda x: x[0])
                results = all_results[:limit]

            # Format results
            formatted_results = []
            for dist, poi in results:
                formatted_results.append({
                    'osm_id': poi['osm_id'],
                    'name': poi['name'],
                    'activity': poi['activity'],
                    'lat': poi['lat'],
                    'lon': poi['lon'],
                    'distance_m': dist,
                    'source': poi['source'],
                    'tags': poi['tags']
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error finding nearest POIs: {e}")
            raise

    def find_pois_by_activity(self, activity: str, limit: int = 100) -> List[Dict]:
        """
        Get all POIs of a specific activity.
        
        Args:
            activity: Activity type (e.g., 'Dining', 'School')
            limit: Max results
        
        Returns:
            List of POI dicts
        """
        try:
            pois = self.db_manager.query_all(POI, filters={'activity': activity})
            
            return [
                {
                    'osm_id': p.osm_id,
                    'name': p.name,
                    'activity': p.activity,
                    'lat': p.lat,
                    'lon': p.lon,
                    'source': p.source_name,
                    'tags': json.loads(p.tags) if p.tags else {}
                }
                for p in pois[:limit]
            ]
        except Exception as e:
            logger.error(f"Error finding POIs by activity: {e}")
            raise
    
    def sample_nearby_poi(self, 
                         lat: float, 
                         lon: float, 
                         activity: str,
                         radius_m: float = 1000) -> Optional[Dict]:
        """
        Sample one nearby POI of given activity.
        
        Args:
            lat, lon: Query coordinates
            activity: Activity type
            radius_m: Search radius
        
        Returns:
            Single POI dict or None
        """
        pois = self.find_nearest_pois(lat, lon, activity=activity, 
                                     radius_m=radius_m, limit=1)
        return pois[0] if pois else None
    
    def get_pois_for_trip(self, 
                         locations: List[Tuple[float, float]],
                         activities: List[str],
                         radius_m: float = 1000) -> List[List[Dict]]:
        """
        Get POIs for each stop in a trip.
        
        Args:
            locations: List of (lat, lon) tuples for each stop
            activities: List of activities for each stop
            radius_m: Search radius for each stop
        
        Returns:
            List of POI lists, one for each stop
        """
        trip_pois = []
        for (lat, lon), activity in zip(locations, activities):
            pois = self.find_nearest_pois(lat, lon, activity=activity, 
                                         radius_m=radius_m, limit=3)
            trip_pois.append(pois)
            logger.info(f"Found {len(pois)} POIs for {activity} at ({lat}, {lon})")
        
        return trip_pois
    
    # ========================================================================
    # PUBLIC API - CUSTOM QUERIES
    # ========================================================================
    
    def execute_raw_query(self, query: str, params: Optional[Dict] = None) -> List[Tuple]:
        """
        Execute a raw SQL query for custom filtering.

        Args:
            query: SQL query string with named parameters (use :param_name syntax)
            params: Optional dictionary of query parameters

        Returns:
            List of result tuples

        Example:
            # Get POIs within a bounding box
            query = '''
            SELECT osm_id, name, activity, lat, lon
            FROM pois
            WHERE activity = :activity
              AND lat BETWEEN :lat_min AND :lat_max
              AND lon BETWEEN :lon_min AND :lon_max
            ORDER BY name
            LIMIT :limit
            '''
            results = poi_manager.execute_raw_query(query, {
                'activity': 'Dining',
                'lat_min': 44.9,
                'lat_max': 45.1,
                'lon_min': -93.3,
                'lon_max': -93.0,
                'limit': 10
            })
        """
        try:
            from sqlalchemy import text
            with self.db_manager.session_scope() as session:
                if params:
                    result = session.execute(text(query), params)
                else:
                    result = session.execute(text(query))
                return result.fetchall()
        except Exception as e:
            logger.error(f"Error executing raw query: {e}")
            raise
    
    # ========================================================================
    # UTILITIES & STATS
    # ========================================================================
    
    def get_stats(self) -> Dict:
        """Get stats on POIs in database."""
        try:
            all_pois = self.db_manager.query_all(POI)
            
            activities = {}
            sources = {}
            
            for poi in all_pois:
                activities[poi.activity] = activities.get(poi.activity, 0) + 1
                source_key = f"{poi.source}:{poi.source_name}"
                sources[source_key] = sources.get(source_key, 0) + 1
            
            return {
                'total_pois': len(all_pois),
                'by_activity': activities,
                'by_source': sources,
                'db_path': str(self.db_manager.db_path)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise
    
    def clear_pois(self, source: Optional[str] = None, source_name: Optional[str] = None):
        """
        Clear POIs from database.
        
        Args:
            source: Source type ('county' or 'city'), None to clear all
            source_name: Source name, None to clear all
        """
        try:
            if source and source_name:
                filters = {'source': source, 'source_name': source_name}
                self.db_manager.delete_records(POI, filters)
                logger.info(f"Cleared POIs from {source}:{source_name}")
            elif source:
                filters = {'source': source}
                self.db_manager.delete_records(POI, filters)
                logger.info(f"Cleared POIs from source {source}")
            else:
                self.db_manager.delete_records(POI, {})
                logger.info("Cleared all POIs")
        except Exception as e:
            logger.error(f"Error clearing POIs: {e}")
            raise
    
    def export_pois_geojson(self, filepath: str, activity: Optional[str] = None):
        """
        Export POIs to GeoJSON format.
        
        Args:
            filepath: Output file path
            activity: Filter by activity (None = all)
        """
        try:
            if activity:
                pois = self.db_manager.query_all(POI, filters={'activity': activity})
            else:
                pois = self.db_manager.query_all(POI)
            
            features = []
            for poi in pois:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [poi.lon, poi.lat]},
                    "properties": {
                        "osm_id": poi.osm_id,
                        "name": poi.name,
                        "activity": poi.activity,
                        "source": poi.source_name,
                        "tags": json.loads(poi.tags) if poi.tags else {}
                    }
                })
            
            geojson = {"type": "FeatureCollection", "features": features}
            with open(filepath, 'w') as f:
                json.dump(geojson, f, indent=2)
            
            logger.info(f"Exported {len(features)} POIs to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting to GeoJSON: {e}")
            raise

    def find_county_by_fips(self, state_fips: str, county_fips: str) -> Optional[str]:
        """
        Find the exact county name in OSM using FIPS codes.
        OSM areas use combined state+county FIPS codes (e.g., "27123" for Ramsey County, MN).

        Args:
            state_fips: State FIPS code (e.g., '27' for Minnesota)
            county_fips: County FIPS code (e.g., '123' for Ramsey)

        Returns:
            The exact county name as stored in OSM, or None if not found
        """
        try:
            full_fips = f"{state_fips}{county_fips.zfill(3)}"
            query = f"""
            [out:json][timeout:30];
            area["ref:fips"="{full_fips}"];
            out tags;
            """
            result = self.overpass.query(query)

            if result.areas:
                for area in result.areas:
                    name = area.tags.get('name')
                    if name:
                        logger.info(f"FIPS {full_fips} -> OSM name: '{name}'")
                        return name

            logger.warning(f"No area found for FIPS {full_fips}")
            return None
        except Exception as e:
            logger.error(f"Error finding county by FIPS: {e}")
            return None


# ============================================================================
# MODULE-LEVEL FUNCTIONS (following ensure_home_locations pattern)
# ============================================================================

def ensure_pois(config: Dict) -> None:
    """
    Ensure POIs exist in the database for all counties specified in config.
    If any counties are missing, fetches them from the Overpass API.

    Follows the same pattern as ensure_home_locations().

    Args:
        config: Configuration dict with config['region']['counties'] GEOIDs
    """
    from models.models import initialize_tables
    from sqlalchemy import tuple_

    county_geoids = config['region']['counties']
    if not county_geoids:
        logger.warning("No counties specified in config, skipping POI ensure")
        return

    data_dir = config['data']['data_dir']
    db_manager = initialize_tables(data_dir)

    # Build requested (state_fips, county_fips) pairs from config GEOIDs
    requested_pairs = {(geoid[:2], geoid[2:5]) for geoid in county_geoids}

    # Query DB for existing (state_fips, county_fips) pairs in the pois table
    with db_manager.Session() as session:
        existing_rows = session.query(
            POI.state_fips,
            POI.county_fips,
        ).filter(
            tuple_(POI.state_fips, POI.county_fips).in_(list(requested_pairs))
        ).distinct().all()
        existing_pairs = {(row[0], row[1]) for row in existing_rows}

    # Find missing pairs
    missing_pairs = requested_pairs - existing_pairs

    if not missing_pairs:
        logger.info(f"{CHECK_MARK} All {len(requested_pairs)} counties already have POIs in database")
        return

    missing_geoids = [f"{s}{c}" for s, c in missing_pairs]
    logger.info(f"Missing POIs for {len(missing_pairs)} counties: {missing_geoids}")
    logger.info(f"  Fetching POIs for {len(missing_pairs)} missing counties: {missing_geoids}")

    # Create a config copy with only the missing counties and fetch
    import copy
    etl_config = copy.deepcopy(config)
    etl_config['region']['counties'] = missing_geoids

    poi_manager = POIManager(db_manager)
    poi_manager.fetch_pois_by_counties(etl_config)

    logger.info(f"{CHECK_MARK} POI ensure complete")


def load_pois_by_counties(config: Dict) -> Dict[str, List[Dict]]:
    """
    Load POIs filtered by configured counties, grouped by activity.

    Args:
        config: Configuration dict with config['region']['counties'] GEOIDs

    Returns:
        Dict mapping activity type to list of POI dicts.
        Each POI dict has: osm_id, name, activity, lat, lon, tags
    """
    from models.models import initialize_tables
    from sqlalchemy import tuple_

    county_geoids = config['region']['counties']
    if not county_geoids:
        logger.warning("No counties specified in config")
        return {}

    data_dir = config['data']['data_dir']
    db_manager = initialize_tables(data_dir)

    county_pairs = [(geoid[:2], geoid[2:5]) for geoid in county_geoids]

    poi_data = {}
    with db_manager.Session() as session:
        results = session.query(POI).filter(
            tuple_(POI.state_fips, POI.county_fips).in_(county_pairs)
        ).all()

        for poi in results:
            activity = poi.activity
            poi_dict = {
                'osm_id': poi.osm_id,
                'name': poi.name,
                'activity': poi.activity,
                'lat': poi.lat,
                'lon': poi.lon,
                'tags': poi.tags
            }
            if activity not in poi_data:
                poi_data[activity] = []
            poi_data[activity].append(poi_dict)

    total_pois = sum(len(pois) for pois in poi_data.values())
    logger.info(f"Loaded {total_pois:,} POIs across {len(poi_data)} activity types (county-filtered)")
    return poi_data