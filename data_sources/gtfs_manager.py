"""
GTFS data manager for transit availability.

Discovers, downloads, and loads GTFS feeds into DuckDB for transit
availability checking. Uses the Mobility Database catalog to auto-discover
feeds that cover the configured region.

Phase 3 Implementation:
- Discovers feeds by bounding-box intersection with configured counties
- Downloads and caches GTFS zip files
- Parses routes.txt, stops.txt, trips.txt, stop_times.txt
- Loads into gtfs_* ORM tables
- Derives gtfs_stop_routes for fast availability queries
"""

import csv
import io
import logging
import os
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
from tqdm import tqdm

from sqlalchemy import func as sa_func

from models.models import (
    GTFSFeed, GTFSRoute, GTFSStop, GTFSStopRoute,
    GTFSStopTime, GTFSTrip,
)
from utils.duckdb_manager import DBManager

logger = logging.getLogger(__name__)


@dataclass
class BBox:
    """Bounding box with min/max lat/lon."""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def intersects(self, other: 'BBox') -> bool:
        """Check if two bounding boxes intersect."""
        if self.max_lon < other.min_lon:
            return False  # self is left of other
        if self.min_lon > other.max_lon:
            return False  # self is right of other
        if self.max_lat < other.min_lat:
            return False  # self is below other
        if self.min_lat > other.max_lat:
            return False  # self is above other
        return True


@dataclass
class FeedInfo:
    """Metadata for a single GTFS feed from the Mobility Database catalog."""
    feed_id: str
    provider: str
    country_code: str
    subdivision: str
    municipality: str
    download_url: str
    bbox: Optional[BBox]
    status: str


class GTFSManager:
    """
    Manages GTFS feed discovery, download, and database loading.

    Workflow:
    1. Compute region bounding box from configured counties
    2. Download Mobility Database catalog CSV (cached)
    3. Filter feeds by country, status, and bbox intersection
    4. Download matching GTFS feeds (cached)
    5. Parse and load into DuckDB tables
    6. Derive gtfs_stop_routes for fast queries
    """

    def __init__(self, config: Dict[str, Any], db_manager: DBManager):
        self.config = config
        self.db_manager = db_manager

        gtfs_config = config.get('gtfs', {})
        self.catalog_url = gtfs_config.get(
            'catalog_url',
            'https://files.mobilitydatabase.org/feeds_v2.csv'
        )
        self.catalog_max_age_days = gtfs_config.get('catalog_max_age_days', 7)
        self.feed_max_age_days = gtfs_config.get('feed_max_age_days', 30)
        self.cache_dir = Path(gtfs_config.get('cache_dir', 'data/gtfs'))
        self.country_filter = gtfs_config.get('country_filter', 'US')
        self.api_keys = gtfs_config.get('api_keys', {})  # domain → key (e.g., {"wmata.com": "YOUR_KEY"})

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Internal tracking flag set by load_feed_to_db
        self._feed_was_already_loaded = False

    # ── Region Bounding Box ──────────────────────────────────────────────

    def compute_region_bbox(self) -> BBox:
        """
        Compute bounding box for the configured counties using osmnx geocoding.

        Uses County table from DB to get county names, then osmnx for geometry.

        Returns:
            BBox for the entire region
        """
        from models.models import County, State

        t0 = time.time()
        county_geoids = self.config['region']['counties']

        all_bounds = []
        with self.db_manager.session_scope() as session:
            for geoid in county_geoids:
                state_fips = geoid[:2]
                county_fips = geoid[2:5]

                result = session.query(County, State).join(
                    State, County.state_fips == State.state_fips
                ).filter(
                    County.state_fips == state_fips,
                    County.county_fips == county_fips
                ).first()

                if not result:
                    logger.warning(f"County not found in DB: {geoid}")
                    continue

                county_obj, state_obj = result
                county_full = county_obj.county_name_full or (county_obj.county_name + " County")
                query = f"{county_full}, {state_obj.state_name}, USA"

                try:
                    gdf = ox.geocode_to_gdf(query)
                    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
                    all_bounds.append(bounds)
                    logger.debug(f"  {query}: bounds={bounds}")
                except Exception as e:
                    logger.warning(f"Could not geocode {query}: {e}")

        if not all_bounds:
            raise ValueError("Could not compute bounding box for any configured county")

        bbox = BBox(
            min_lat=min(b[1] for b in all_bounds),
            max_lat=max(b[3] for b in all_bounds),
            min_lon=min(b[0] for b in all_bounds),
            max_lon=max(b[2] for b in all_bounds),
        )

        elapsed = time.time() - t0
        logger.info(f"Region bbox: ({bbox.min_lon:.4f}, {bbox.min_lat:.4f}, "
                    f"{bbox.max_lon:.4f}, {bbox.max_lat:.4f}) (took {elapsed:.1f}s)")
        return bbox

    # ── Catalog Download & Feed Discovery ────────────────────────────────

    def _get_catalog_path(self) -> Path:
        """Path to the cached catalog CSV."""
        return self.cache_dir / 'feeds_v2.csv'

    def _is_catalog_fresh(self) -> bool:
        """Check if cached catalog is within max age."""
        catalog_path = self._get_catalog_path()
        if not catalog_path.exists():
            return False
        mtime = datetime.fromtimestamp(catalog_path.stat().st_mtime)
        age = datetime.now() - mtime
        return age < timedelta(days=self.catalog_max_age_days)

    def _download_catalog(self) -> Path:
        """Download the Mobility Database catalog CSV, with caching."""
        catalog_path = self._get_catalog_path()

        if self._is_catalog_fresh():
            logger.info(f"Using cached catalog: {catalog_path}")
            return catalog_path

        logger.info(f"Downloading Mobility Database catalog from {self.catalog_url}...")
        t0 = time.time()

        response = requests.get(self.catalog_url, timeout=60)
        response.raise_for_status()

        catalog_path.write_bytes(response.content)
        elapsed = time.time() - t0
        logger.info(f"Downloaded catalog ({len(response.content) / 1024:.0f} KB, took {elapsed:.1f}s)")

        return catalog_path

    def _is_local_feed(self, feed_bbox: BBox, region_bbox: BBox,
                       max_area_ratio: float = 25.0) -> bool:
        """
        Check if a feed's bbox is reasonably local to the region.

        Rejects feeds with bboxes vastly larger than the region (e.g., Amtrak
        spans the entire US but matches any US region by bbox intersection).

        Args:
            feed_bbox: Feed's bounding box
            region_bbox: Region's bounding box
            max_area_ratio: Maximum allowed ratio of feed area to region area

        Returns:
            True if the feed bbox is reasonably local
        """
        region_area = ((region_bbox.max_lat - region_bbox.min_lat) *
                       (region_bbox.max_lon - region_bbox.min_lon))
        if region_area <= 0:
            return True

        feed_area = ((feed_bbox.max_lat - feed_bbox.min_lat) *
                     (feed_bbox.max_lon - feed_bbox.min_lon))

        ratio = feed_area / region_area
        return ratio <= max_area_ratio

    def _extract_feeds_from_df(self, df: pd.DataFrame, region_bbox: BBox,
                               bbox_cols: Optional[Dict[str, str]],
                               url_col: str, provider_col: Optional[str],
                               id_col: Optional[str], subdiv_col: Optional[str],
                               munic_col: Optional[str], status_col: Optional[str],
                               check_locality: bool = True) -> List[FeedInfo]:
        """
        Extract matching FeedInfo entries from a catalog DataFrame.

        Args:
            df: Filtered catalog DataFrame
            region_bbox: Region bounding box
            bbox_cols: Column name mapping for bbox fields
            url_col: Column name for download URL
            provider_col: Column name for provider/agency name
            id_col: Column name for feed ID
            subdiv_col: Column name for subdivision
            munic_col: Column name for municipality
            status_col: Column name for status field
            check_locality: If True, reject feeds whose bbox is vastly larger than region

        Returns:
            List of FeedInfo for matching feeds
        """
        feeds = []
        skipped_too_broad = 0

        for _, row in df.iterrows():
            download_url = str(row.get(url_col, '')).strip()
            if not download_url or download_url == 'nan':
                continue

            feed_bbox = self._parse_feed_bbox(row, bbox_cols) if bbox_cols else None

            # Skip feeds with no bbox data or that don't intersect our region
            if feed_bbox is None or not feed_bbox.intersects(region_bbox):
                continue

            # Skip feeds with bboxes vastly larger than the region
            if check_locality and feed_bbox and not self._is_local_feed(feed_bbox, region_bbox):
                skipped_too_broad += 1
                continue

            feed_id = str(row.get(id_col, '')) if id_col else ''
            if not feed_id or feed_id == 'nan':
                provider = str(row.get(provider_col, 'unknown')) if provider_col else 'unknown'
                feed_id = provider.replace(' ', '_').lower()[:50]

            # Read actual status from catalog row
            row_status = 'active'
            if status_col:
                raw = str(row.get(status_col, '')).strip().lower()
                if raw and raw != 'nan':
                    row_status = raw

            feeds.append(FeedInfo(
                feed_id=feed_id,
                provider=str(row.get(provider_col, '')) if provider_col else '',
                country_code=self.country_filter,
                subdivision=str(row.get(subdiv_col, '')) if subdiv_col else '',
                municipality=str(row.get(munic_col, '')) if munic_col else '',
                download_url=download_url,
                bbox=feed_bbox,
                status=row_status,
            ))

        if skipped_too_broad > 0:
            logger.info(f"  Skipped {skipped_too_broad} feeds with bboxes too broad for region")

        return feeds

    def discover_feeds(self, region_bbox: BBox) -> List[FeedInfo]:
        """
        Discover GTFS feeds that intersect the region bounding box.

        Downloads the Mobility Database catalog, filters by country/status/data_type,
        then checks bounding box intersection. Rejects feeds whose bbox is vastly
        larger than the region (e.g., nationwide Amtrak).

        If no active feeds are found, retries including deprecated/inactive feeds
        as a fallback (some regional agencies are incorrectly marked in the catalog).

        Args:
            region_bbox: Bounding box of the configured region

        Returns:
            List of FeedInfo for feeds covering the region
        """
        catalog_path = self._download_catalog()

        t0 = time.time()
        df = pd.read_csv(catalog_path, dtype=str)
        logger.info(f"Catalog has {len(df)} total entries")

        # Normalize column names (the catalog uses varying conventions)
        df.columns = [c.strip().lower() for c in df.columns]

        # Filter by data type (schedule, not realtime)
        if 'data_type' in df.columns:
            df = df[df['data_type'].str.lower() == 'gtfs']
            logger.info(f"  After data_type=gtfs filter: {len(df)}")

        # Filter by country
        country_col = None
        for candidate in ['location.country_code', 'country_code', 'location.country']:
            if candidate in df.columns:
                country_col = candidate
                break
        if country_col:
            df = df[df[country_col].str.upper() == self.country_filter]
            logger.info(f"  After country={self.country_filter} filter: {len(df)}")

        # Identify status column
        status_col = None
        for candidate in ['status', 'feed_status']:
            if candidate in df.columns:
                status_col = candidate
                break

        # Find column mappings
        bbox_cols = self._find_bbox_columns(df)
        if not bbox_cols:
            logger.warning("No bounding box columns found in catalog. Using all filtered feeds.")

        url_col = self._find_url_column(df)
        if not url_col:
            raise ValueError("Could not find download URL column in catalog CSV")

        provider_col = self._find_provider_column(df)
        id_col = self._find_id_column(df)

        subdiv_col = None
        for candidate in ['location.subdivision_name', 'subdivision_name', 'location.subdivision']:
            if candidate in df.columns:
                subdiv_col = candidate
                break

        munic_col = None
        for candidate in ['location.municipality', 'municipality']:
            if candidate in df.columns:
                munic_col = candidate
                break

        # Phase 1: Try active feeds only (exclude deprecated/inactive)
        df_active = df
        if status_col:
            excluded = {'deprecated', 'inactive'}
            mask = ~df[status_col].str.lower().isin(excluded)
            mask = mask | df[status_col].isna()
            df_active = df[mask]
            logger.info(f"  After excluding deprecated/inactive: {len(df_active)}")

        feeds = self._extract_feeds_from_df(
            df_active, region_bbox, bbox_cols, url_col, provider_col,
            id_col, subdiv_col, munic_col, status_col,
        )

        # Phase 2: If no local feeds found, retry including deprecated/inactive
        if not feeds and status_col:
            n_excluded = len(df) - len(df_active)
            if n_excluded > 0:
                logger.warning(
                    f"No active GTFS feeds found for region. "
                    f"Retrying with {n_excluded} deprecated/inactive feeds included..."
                )
                feeds = self._extract_feeds_from_df(
                    df, region_bbox, bbox_cols, url_col, provider_col,
                    id_col, subdiv_col, munic_col, status_col,
                )
                if feeds:
                    for f in feeds:
                        logger.warning(
                            f"  Using {f.status} feed: {f.provider} ({f.feed_id})"
                        )

        elapsed = time.time() - t0
        logger.info(f"Found {len(feeds)} GTFS feeds for region (took {elapsed:.1f}s):")
        for f in feeds:
            status_tag = f" [{f.status}]" if f.status not in ('active', '') else ""
            logger.info(f"  - {f.provider} ({f.feed_id}): {f.subdivision} {f.municipality}{status_tag}")

        return feeds

    def _find_bbox_columns(self, df: pd.DataFrame) -> Optional[Dict[str, str]]:
        """Find bounding box column names in the catalog DataFrame."""
        # Try different naming conventions
        patterns = [
            {'min_lat': 'location.bounding_box.minimum_latitude',
             'max_lat': 'location.bounding_box.maximum_latitude',
             'min_lon': 'location.bounding_box.minimum_longitude',
             'max_lon': 'location.bounding_box.maximum_longitude'},
            {'min_lat': 'minimum_latitude', 'max_lat': 'maximum_latitude',
             'min_lon': 'minimum_longitude', 'max_lon': 'maximum_longitude'},
            {'min_lat': 'bbox_min_lat', 'max_lat': 'bbox_max_lat',
             'min_lon': 'bbox_min_lon', 'max_lon': 'bbox_max_lon'},
        ]
        for pattern in patterns:
            if all(col in df.columns for col in pattern.values()):
                return pattern
        return None

    def _find_url_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the download URL column."""
        candidates = [
            'urls.direct_download', 'direct_download_url',
            'url', 'urls.latest', 'download_url',
        ]
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _find_provider_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the provider/agency name column."""
        for c in ['provider', 'name', 'agency_name', 'feed_name']:
            if c in df.columns:
                return c
        return None

    def _find_id_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the feed ID column."""
        for c in ['mdb_source_id', 'id', 'feed_id', 'source_id']:
            if c in df.columns:
                return c
        return None

    def _parse_feed_bbox(self, row: pd.Series, bbox_cols: Dict[str, str]) -> Optional[BBox]:
        """Parse bounding box from a catalog row. Returns None if any value is missing/NaN."""
        import math
        try:
            vals = {
                'min_lat': float(row[bbox_cols['min_lat']]),
                'max_lat': float(row[bbox_cols['max_lat']]),
                'min_lon': float(row[bbox_cols['min_lon']]),
                'max_lon': float(row[bbox_cols['max_lon']]),
            }
            # float('nan') doesn't raise, so check explicitly
            if any(math.isnan(v) for v in vals.values()):
                return None
            return BBox(**vals)
        except (ValueError, TypeError, KeyError):
            return None

    # ── Feed Download ────────────────────────────────────────────────────

    def _get_feed_dir(self, feed_id: str) -> Path:
        """Directory for a specific feed's extracted files."""
        return self.cache_dir / feed_id

    def _is_feed_fresh(self, feed_id: str) -> bool:
        """Check if a downloaded feed is within max age."""
        feed_dir = self._get_feed_dir(feed_id)
        marker = feed_dir / '.downloaded_at'
        if not marker.exists():
            return False
        mtime = datetime.fromtimestamp(marker.stat().st_mtime)
        age = datetime.now() - mtime
        return age < timedelta(days=self.feed_max_age_days)

    def download_feed(self, feed: FeedInfo) -> Optional[Path]:
        """
        Download and extract a GTFS feed zip file.

        Args:
            feed: FeedInfo with download_url and feed_id

        Returns:
            Path to extracted feed directory, or None on failure
        """
        feed_dir = self._get_feed_dir(feed.feed_id)

        if self._is_feed_fresh(feed.feed_id):
            logger.info(f"Feed {feed.feed_id} is up to date (cached)")
            return feed_dir

        logger.info(f"Downloading feed {feed.feed_id} from {feed.download_url}...")
        t0 = time.time()

        try:
            url = feed.download_url
            # Check if an API key is configured for this feed's domain
            from urllib.parse import urlparse, urlencode, urlunparse, parse_qs
            domain = urlparse(url).hostname or ''
            api_key = None
            for key_domain, key_value in self.api_keys.items():
                if key_domain in domain and key_value and key_value != 'YOUR_KEY_HERE':
                    api_key = key_value
                    break
            if api_key:
                parsed = urlparse(url)
                params = parse_qs(parsed.query)
                params['api_key'] = [api_key]
                new_query = urlencode(params, doseq=True)
                url = urlunparse(parsed._replace(query=new_query))
                logger.debug(f"Added API key for {domain}")

            response = requests.get(url, timeout=120)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to download feed {feed.feed_id}: {e}")
            return None

        # Extract zip
        feed_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                zf.extractall(feed_dir)
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid zip file for feed {feed.feed_id}: {e}")
            return None

        # Write download marker
        marker = feed_dir / '.downloaded_at'
        marker.write_text(datetime.now().isoformat())

        elapsed = time.time() - t0
        size_kb = len(response.content) / 1024
        logger.info(f"Downloaded feed {feed.feed_id} ({size_kb:.0f} KB, took {elapsed:.1f}s)")

        return feed_dir

    # ── Feed Loading into DB ─────────────────────────────────────────────

    def _next_id(self, model_class) -> int:
        """Get the next available ID for a table (max(id) + 1, or 1 if empty)."""
        with self.db_manager.session_scope() as session:
            max_id = session.query(sa_func.max(model_class.id)).scalar()
        return (max_id or 0) + 1

    def _feed_exists_in_db(self, feed_id: str) -> bool:
        """Check if a feed is already loaded in the database."""
        results = self.db_manager.query_all(GTFSFeed, filters={'feed_id': feed_id})
        return len(results) > 0

    def load_feed_to_db(self, feed_path: Path, feed: FeedInfo) -> bool:
        """
        Parse GTFS files and load into database tables atomically.

        All inserts for a single feed happen in one transaction. If any step
        fails or the process is interrupted, the entire feed is rolled back,
        preventing partial data in the database.

        Loads: feed metadata, routes, stops, trips, stop_times, stop_routes.

        Args:
            feed_path: Path to extracted GTFS directory
            feed: FeedInfo metadata

        Returns:
            True if successful
        """
        feed_id = feed.feed_id
        t0 = time.time()
        self._feed_was_already_loaded = False

        # Check if already loaded
        if self._feed_exists_in_db(feed_id):
            logger.info(f"Feed {feed_id} already in database, skipping")
            self._feed_was_already_loaded = True
            return True

        logger.info(f"Loading feed {feed_id} into database...")

        try:
            with self.db_manager.write_session_scope() as session:
                # 1. Insert feed metadata
                feed_obj = GTFSFeed(**self.db_manager.handle_binary_data(GTFSFeed, {
                    'feed_id': feed_id,
                    'provider': feed.provider,
                    'country_code': feed.country_code,
                    'subdivision': feed.subdivision,
                    'municipality': feed.municipality,
                    'download_url': feed.download_url,
                    'bbox_min_lat': feed.bbox.min_lat if feed.bbox else None,
                    'bbox_max_lat': feed.bbox.max_lat if feed.bbox else None,
                    'bbox_min_lon': feed.bbox.min_lon if feed.bbox else None,
                    'bbox_max_lon': feed.bbox.max_lon if feed.bbox else None,
                    'downloaded_at': datetime.now(),
                    'gtfs_version': self._read_feed_version(feed_path),
                    'status': feed.status,
                }))
                session.add(feed_obj)
                session.flush()

                # 2. Load agency.txt for agency names
                agency_names = self._load_agency_names(feed_path)

                # 3. Load routes.txt
                route_pk_map = self._load_routes_atomic(session, feed_path, feed_id, agency_names)
                logger.info(f"  Loaded {len(route_pk_map)} routes")

                # 4. Load stops.txt
                stop_pk_map = self._load_stops_atomic(session, feed_path, feed_id)
                logger.info(f"  Loaded {len(stop_pk_map)} stops")

                # 5. Load trips.txt
                trip_pk_map = self._load_trips_atomic(session, feed_path, feed_id, route_pk_map)
                logger.info(f"  Loaded {len(trip_pk_map)} trips")

                # 6. Load stop_times.txt
                n_stop_times = self._load_stop_times_atomic(session, feed_path, feed_id, trip_pk_map, stop_pk_map)
                logger.info(f"  Loaded {n_stop_times} stop times")

                # 7. Derive gtfs_stop_routes
                n_stop_routes = self._derive_stop_routes_atomic(session, feed_id)
                logger.info(f"  Derived {n_stop_routes} stop-route associations")

                elapsed = time.time() - t0
                logger.info(f"Loaded feed {feed_id}: {len(route_pk_map)} routes, "
                            f"{len(stop_pk_map)} stops, {len(trip_pk_map)} trips, "
                            f"{n_stop_times} stop_times (took {elapsed:.1f}s)")
                return True

        except Exception as e:
            logger.error(f"Failed to load feed {feed_id}: {e}")
            return False

    def _read_feed_version(self, feed_path: Path) -> Optional[str]:
        """Read feed version from feed_info.txt if it exists."""
        fi_path = feed_path / 'feed_info.txt'
        if not fi_path.exists():
            return None
        try:
            df = pd.read_csv(fi_path, dtype=str)
            if 'feed_version' in df.columns and len(df) > 0:
                return str(df['feed_version'].iloc[0])
        except Exception:
            pass
        return None

    def _load_agency_names(self, feed_path: Path) -> Dict[str, str]:
        """Load agency_id -> agency_name mapping from agency.txt."""
        agency_path = feed_path / 'agency.txt'
        if not agency_path.exists():
            return {}
        try:
            df = pd.read_csv(agency_path, dtype=str)
            if 'agency_id' in df.columns and 'agency_name' in df.columns:
                return dict(zip(df['agency_id'], df['agency_name']))
            elif 'agency_name' in df.columns:
                # Some feeds have no agency_id
                return {'': df['agency_name'].iloc[0] if len(df) > 0 else ''}
        except Exception as e:
            logger.warning(f"Could not load agency.txt: {e}")
        return {}

    def _load_routes(self, feed_path: Path, feed_id: str,
                     agency_names: Dict[str, str]) -> Dict[str, int]:
        """
        Load routes.txt into gtfs_routes table.

        Returns:
            Mapping of (feed route_id) -> (database pk) for foreign key lookups
        """
        routes_path = feed_path / 'routes.txt'
        if not routes_path.exists():
            logger.warning(f"No routes.txt in {feed_path}")
            return {}

        df = pd.read_csv(routes_path, dtype=str)
        records = []
        next_id = self._next_id(GTFSRoute)

        for _, row in df.iterrows():
            route_type_str = str(row.get('route_type', '3'))
            try:
                route_type = int(route_type_str)
            except ValueError:
                route_type = 3  # Default to bus

            agency_id = str(row.get('agency_id', ''))
            records.append({
                'id': next_id,
                'feed_id': feed_id,
                'route_id': str(row.get('route_id', '')),
                'route_short_name': str(row.get('route_short_name', '')),
                'route_long_name': str(row.get('route_long_name', '')),
                'route_type': route_type,
                'agency_id': agency_id,
                'agency_name': agency_names.get(agency_id, agency_names.get('', '')),
            })
            next_id += 1

        if records:
            self.db_manager.insert_records(GTFSRoute, records)

        # Build pk map by querying back
        return self._build_route_pk_map(feed_id)

    def _build_route_pk_map(self, feed_id: str) -> Dict[str, int]:
        """Build route_id -> pk mapping for a feed."""
        results = self.db_manager.query_all(GTFSRoute, filters={'feed_id': feed_id})
        return {r.route_id: r.id for r in results}

    def _load_stops(self, feed_path: Path, feed_id: str) -> Dict[str, int]:
        """
        Load stops.txt into gtfs_stops table.

        Returns:
            Mapping of (feed stop_id) -> (database pk)
        """
        stops_path = feed_path / 'stops.txt'
        if not stops_path.exists():
            logger.warning(f"No stops.txt in {feed_path}")
            return {}

        df = pd.read_csv(stops_path, dtype=str)
        records = []
        next_id = self._next_id(GTFSStop)

        for _, row in df.iterrows():
            try:
                lat = float(row.get('stop_lat', 0))
                lon = float(row.get('stop_lon', 0))
            except (ValueError, TypeError):
                continue  # Skip stops without valid coordinates

            if lat == 0 and lon == 0:
                continue

            loc_type_str = str(row.get('location_type', '0'))
            try:
                location_type = int(loc_type_str)
            except ValueError:
                location_type = 0

            records.append({
                'id': next_id,
                'feed_id': feed_id,
                'stop_id': str(row.get('stop_id', '')),
                'stop_name': str(row.get('stop_name', '')),
                'lat': lat,
                'lon': lon,
                'location_type': location_type,
                'parent_station': str(row.get('parent_station', '')) or None,
            })
            next_id += 1

        if records:
            self.db_manager.insert_records(GTFSStop, records)

        # Build pk map
        return self._build_stop_pk_map(feed_id)

    def _build_stop_pk_map(self, feed_id: str) -> Dict[str, int]:
        """Build stop_id -> pk mapping for a feed."""
        results = self.db_manager.query_all(GTFSStop, filters={'feed_id': feed_id})
        return {s.stop_id: s.id for s in results}

    def _load_trips(self, feed_path: Path, feed_id: str,
                    route_pk_map: Dict[str, int]) -> Dict[str, int]:
        """
        Load trips.txt into gtfs_trips table.

        Returns:
            Mapping of (feed trip_id) -> (database pk)
        """
        trips_path = feed_path / 'trips.txt'
        if not trips_path.exists():
            logger.warning(f"No trips.txt in {feed_path}")
            return {}

        df = pd.read_csv(trips_path, dtype=str)
        records = []
        next_id = self._next_id(GTFSTrip)

        for _, row in df.iterrows():
            route_id = str(row.get('route_id', ''))
            route_pk = route_pk_map.get(route_id)
            if route_pk is None:
                continue  # Skip trips referencing unknown routes

            dir_str = str(row.get('direction_id', ''))
            try:
                direction_id = int(dir_str)
            except ValueError:
                direction_id = None

            records.append({
                'id': next_id,
                'feed_id': feed_id,
                'trip_id': str(row.get('trip_id', '')),
                'route_pk': route_pk,
                'service_id': str(row.get('service_id', '')) or None,
                'trip_headsign': str(row.get('trip_headsign', '')) or None,
                'direction_id': direction_id,
                'shape_id': str(row.get('shape_id', '')) or None,
            })
            next_id += 1

        if records:
            self.db_manager.insert_records(GTFSTrip, records)

        # Build pk map
        return self._build_trip_pk_map(feed_id)

    def _build_trip_pk_map(self, feed_id: str) -> Dict[str, int]:
        """Build trip_id -> pk mapping for a feed."""
        results = self.db_manager.query_all(GTFSTrip, filters={'feed_id': feed_id})
        return {t.trip_id: t.id for t in results}

    def _load_stop_times(self, feed_path: Path, feed_id: str,
                         trip_pk_map: Dict[str, int],
                         stop_pk_map: Dict[str, int]) -> int:
        """
        Load stop_times.txt into gtfs_stop_times table.

        Uses chunked loading for large files.

        Returns:
            Number of stop_time records loaded
        """
        st_path = feed_path / 'stop_times.txt'
        if not st_path.exists():
            logger.warning(f"No stop_times.txt in {feed_path}")
            return 0

        total = 0
        chunk_size = 50_000
        next_id = self._next_id(GTFSStopTime)

        for chunk in pd.read_csv(st_path, dtype=str, chunksize=chunk_size):
            records = []
            for _, row in chunk.iterrows():
                trip_id = str(row.get('trip_id', ''))
                stop_id = str(row.get('stop_id', ''))

                trip_pk = trip_pk_map.get(trip_id)
                stop_pk = stop_pk_map.get(stop_id)

                if trip_pk is None or stop_pk is None:
                    continue

                seq_str = str(row.get('stop_sequence', '0'))
                try:
                    stop_sequence = int(seq_str)
                except ValueError:
                    stop_sequence = 0

                records.append({
                    'id': next_id,
                    'feed_id': feed_id,
                    'trip_pk': trip_pk,
                    'stop_pk': stop_pk,
                    'arrival_time': str(row.get('arrival_time', '')) or None,
                    'departure_time': str(row.get('departure_time', '')) or None,
                    'stop_sequence': stop_sequence,
                })
                next_id += 1

            if records:
                self.db_manager.insert_records(GTFSStopTime, records)
                total += len(records)
                logger.debug(f"  Loaded stop_times chunk: {len(records)} records (total: {total})")

        return total

    def _derive_stop_routes(self, feed_id: str) -> int:
        """
        Derive gtfs_stop_routes table: which routes serve which stops.

        Uses stop_times -> trips -> routes to find unique (stop_pk, route_pk) pairs.

        Returns:
            Number of stop-route associations created
        """
        # Query distinct (stop_pk, route_pk) via SQL
        with self.db_manager.session_scope() as session:
            from sqlalchemy import text
            result = session.execute(text("""
                SELECT DISTINCT st.stop_pk, t.route_pk
                FROM gtfs_stop_times st
                JOIN gtfs_trips t ON st.trip_pk = t.id AND st.feed_id = t.feed_id
                WHERE st.feed_id = :feed_id
            """), {'feed_id': feed_id})

            rows = result.fetchall()

        # Insert into gtfs_stop_routes
        next_id = self._next_id(GTFSStopRoute)
        records = []
        for row in rows:
            records.append({'id': next_id, 'stop_pk': row[0], 'route_pk': row[1]})
            next_id += 1

        if records:
            self.db_manager.insert_records(GTFSStopRoute, records)

        return len(records)

    # ── Atomic Load Methods (single-session, for transactional loading) ──

    def _next_id_from_session(self, session, model_class) -> int:
        """Get the next available ID using the current session (no new connection)."""
        max_id = session.query(sa_func.max(model_class.id)).scalar()
        return (max_id or 0) + 1

    def _load_routes_atomic(self, session, feed_path: Path, feed_id: str,
                            agency_names: Dict[str, str]) -> Dict[str, int]:
        """Load routes.txt within an existing session. Returns route_id -> pk map."""
        routes_path = feed_path / 'routes.txt'
        if not routes_path.exists():
            logger.warning(f"No routes.txt in {feed_path}")
            return {}

        df = pd.read_csv(routes_path, dtype=str)
        next_id = self._next_id_from_session(session, GTFSRoute)
        pk_map = {}

        objects = []
        for _, row in df.iterrows():
            route_type_str = str(row.get('route_type', '3'))
            try:
                route_type = int(route_type_str)
            except ValueError:
                route_type = 3

            agency_id = str(row.get('agency_id', ''))
            route_id = str(row.get('route_id', ''))

            objects.append(GTFSRoute(
                id=next_id,
                feed_id=feed_id,
                route_id=route_id,
                route_short_name=str(row.get('route_short_name', '')),
                route_long_name=str(row.get('route_long_name', '')),
                route_type=route_type,
                agency_id=agency_id,
                agency_name=agency_names.get(agency_id, agency_names.get('', '')),
            ))
            pk_map[route_id] = next_id
            next_id += 1

        if objects:
            session.bulk_save_objects(objects)
            session.flush()

        return pk_map

    def _load_stops_atomic(self, session, feed_path: Path, feed_id: str) -> Dict[str, int]:
        """Load stops.txt within an existing session. Returns stop_id -> pk map."""
        stops_path = feed_path / 'stops.txt'
        if not stops_path.exists():
            logger.warning(f"No stops.txt in {feed_path}")
            return {}

        df = pd.read_csv(stops_path, dtype=str)
        next_id = self._next_id_from_session(session, GTFSStop)
        pk_map = {}

        objects = []
        for _, row in df.iterrows():
            try:
                lat = float(row.get('stop_lat', 0))
                lon = float(row.get('stop_lon', 0))
            except (ValueError, TypeError):
                continue

            if lat == 0 and lon == 0:
                continue

            loc_type_str = str(row.get('location_type', '0'))
            try:
                location_type = int(loc_type_str)
            except ValueError:
                location_type = 0

            stop_id = str(row.get('stop_id', ''))
            objects.append(GTFSStop(
                id=next_id,
                feed_id=feed_id,
                stop_id=stop_id,
                stop_name=str(row.get('stop_name', '')),
                lat=lat,
                lon=lon,
                location_type=location_type,
                parent_station=str(row.get('parent_station', '')) or None,
            ))
            pk_map[stop_id] = next_id
            next_id += 1

        if objects:
            session.bulk_save_objects(objects)
            session.flush()

        return pk_map

    def _load_trips_atomic(self, session, feed_path: Path, feed_id: str,
                           route_pk_map: Dict[str, int]) -> Dict[str, int]:
        """Load trips.txt within an existing session. Returns trip_id -> pk map."""
        trips_path = feed_path / 'trips.txt'
        if not trips_path.exists():
            logger.warning(f"No trips.txt in {feed_path}")
            return {}

        df = pd.read_csv(trips_path, dtype=str)
        next_id = self._next_id_from_session(session, GTFSTrip)
        pk_map = {}

        objects = []
        for _, row in df.iterrows():
            route_id = str(row.get('route_id', ''))
            route_pk = route_pk_map.get(route_id)
            if route_pk is None:
                continue

            dir_str = str(row.get('direction_id', ''))
            try:
                direction_id = int(dir_str)
            except ValueError:
                direction_id = None

            trip_id = str(row.get('trip_id', ''))
            objects.append(GTFSTrip(
                id=next_id,
                feed_id=feed_id,
                trip_id=trip_id,
                route_pk=route_pk,
                service_id=str(row.get('service_id', '')) or None,
                trip_headsign=str(row.get('trip_headsign', '')) or None,
                direction_id=direction_id,
                shape_id=str(row.get('shape_id', '')) or None,
            ))
            pk_map[trip_id] = next_id
            next_id += 1

        if objects:
            session.bulk_save_objects(objects)
            session.flush()

        return pk_map

    def _load_stop_times_atomic(self, session, feed_path: Path, feed_id: str,
                                trip_pk_map: Dict[str, int],
                                stop_pk_map: Dict[str, int]) -> int:
        """Load stop_times.txt within an existing session. Returns count loaded."""
        st_path = feed_path / 'stop_times.txt'
        if not st_path.exists():
            logger.warning(f"No stop_times.txt in {feed_path}")
            return 0

        total = 0
        chunk_size = 50_000
        next_id = self._next_id_from_session(session, GTFSStopTime)

        for chunk in pd.read_csv(st_path, dtype=str, chunksize=chunk_size):
            objects = []
            for _, row in chunk.iterrows():
                trip_id = str(row.get('trip_id', ''))
                stop_id = str(row.get('stop_id', ''))

                trip_pk = trip_pk_map.get(trip_id)
                stop_pk = stop_pk_map.get(stop_id)

                if trip_pk is None or stop_pk is None:
                    continue

                seq_str = str(row.get('stop_sequence', '0'))
                try:
                    stop_sequence = int(seq_str)
                except ValueError:
                    stop_sequence = 0

                objects.append(GTFSStopTime(
                    id=next_id,
                    feed_id=feed_id,
                    trip_pk=trip_pk,
                    stop_pk=stop_pk,
                    arrival_time=str(row.get('arrival_time', '')) or None,
                    departure_time=str(row.get('departure_time', '')) or None,
                    stop_sequence=stop_sequence,
                ))
                next_id += 1

            if objects:
                session.bulk_save_objects(objects)
                session.flush()
                total += len(objects)

        return total

    def _derive_stop_routes_atomic(self, session, feed_id: str) -> int:
        """Derive gtfs_stop_routes within an existing session. Returns count."""
        from sqlalchemy import text
        result = session.execute(text("""
            SELECT DISTINCT st.stop_pk, t.route_pk
            FROM gtfs_stop_times st
            JOIN gtfs_trips t ON st.trip_pk = t.id AND st.feed_id = t.feed_id
            WHERE st.feed_id = :feed_id
        """), {'feed_id': feed_id})

        rows = result.fetchall()

        next_id = self._next_id_from_session(session, GTFSStopRoute)
        objects = []
        for row in rows:
            objects.append(GTFSStopRoute(id=next_id, stop_pk=row[0], route_pk=row[1]))
            next_id += 1

        if objects:
            session.bulk_save_objects(objects)
            session.flush()

        return len(objects)

    # ── Query Methods (for GTFSAvailabilityManager) ──────────────────────

    def get_stops_by_route_types(self, route_types: List[int]) -> pd.DataFrame:
        """
        Get all stops that serve routes of the specified types.

        Uses gtfs_stop_routes join to find stops serving matching routes.

        Args:
            route_types: List of GTFS route type codes (e.g., [3] for bus)

        Returns:
            DataFrame with columns: stop_pk, stop_id, stop_name, lat, lon, route_type
        """
        if not route_types:
            return pd.DataFrame()

        with self.db_manager.session_scope() as session:
            from sqlalchemy import text

            placeholders = ', '.join([f':rt{i}' for i in range(len(route_types))])
            params = {f'rt{i}': rt for i, rt in enumerate(route_types)}

            result = session.execute(text(f"""
                SELECT DISTINCT
                    s.id AS stop_pk,
                    s.stop_id,
                    s.stop_name,
                    s.lat,
                    s.lon,
                    r.route_type
                FROM gtfs_stops s
                JOIN gtfs_stop_routes sr ON s.id = sr.stop_pk
                JOIN gtfs_routes r ON sr.route_pk = r.id
                WHERE r.route_type IN ({placeholders})
            """), params)

            rows = result.fetchall()

        if not rows:
            return pd.DataFrame(columns=['stop_pk', 'stop_id', 'stop_name', 'lat', 'lon', 'route_type'])

        return pd.DataFrame(rows, columns=['stop_pk', 'stop_id', 'stop_name', 'lat', 'lon', 'route_type'])

    # ── Integrity & Cleanup ─────────────────────────────────────────────

    def _cleanup_partial_feeds(self) -> None:
        """
        Detect and remove partially loaded feeds from interrupted runs.

        A feed is considered partial if it has a GTFSFeed record but is missing
        critical data (e.g., zero stop_times or zero stops). This can happen
        when the program is interrupted mid-load.
        """
        all_feeds = self.db_manager.query_all(GTFSFeed)
        if not all_feeds:
            return

        logger.info(f"Checking integrity of {len(all_feeds)} feeds in database...")
        partial_feeds = []

        with self.db_manager.session_scope() as session:
            from sqlalchemy import text
            for feed_record in all_feeds:
                fid = feed_record.feed_id

                # Count critical tables for this feed
                counts = {}
                for table in ['gtfs_routes', 'gtfs_stops', 'gtfs_trips', 'gtfs_stop_times']:
                    result = session.execute(
                        text(f"SELECT COUNT(*) FROM {table} WHERE feed_id = :fid"),
                        {'fid': fid}
                    )
                    counts[table] = result.scalar()

                # A feed is partial if it has the metadata but is missing stops or stop_times
                has_routes = counts['gtfs_routes'] > 0
                has_stops = counts['gtfs_stops'] > 0
                has_stop_times = counts['gtfs_stop_times'] > 0

                if has_routes and (not has_stops or not has_stop_times):
                    partial_feeds.append((fid, feed_record.provider, counts))

        if not partial_feeds:
            logger.info("All feeds are complete")
            return

        logger.warning(f"Found {len(partial_feeds)} partially loaded feed(s), cleaning up:")
        for fid, provider, counts in partial_feeds:
            logger.warning(
                f"  Removing partial feed {fid} ({provider}): "
                f"routes={counts['gtfs_routes']}, stops={counts['gtfs_stops']}, "
                f"trips={counts['gtfs_trips']}, stop_times={counts['gtfs_stop_times']}"
            )
            self._delete_feed_data(fid)

        logger.info(f"Cleanup complete: removed {len(partial_feeds)} partial feed(s)")

    def _delete_feed_data(self, feed_id: str) -> None:
        """
        Delete all data for a specific feed from all GTFS tables.

        Deletes in reverse dependency order to avoid constraint issues.
        """
        # GTFSStopRoute has no feed_id — must delete via route_pk join first
        self._delete_stop_routes_for_feed(feed_id)

        # Then delete the rest in child-to-parent order
        for model in [GTFSStopTime, GTFSTrip, GTFSStop, GTFSRoute, GTFSFeed]:
            try:
                self.db_manager.delete_records(model, filters={'feed_id': feed_id})
            except Exception as e:
                logger.error(f"Failed to delete {model.__tablename__} for feed {feed_id}: {e}")

    def _delete_stop_routes_for_feed(self, feed_id: str) -> None:
        """Delete gtfs_stop_routes entries associated with a feed's routes."""
        with self.db_manager.write_session_scope() as session:
            from sqlalchemy import text
            session.execute(text("""
                DELETE FROM gtfs_stop_routes
                WHERE route_pk IN (
                    SELECT id FROM gtfs_routes WHERE feed_id = :fid
                )
            """), {'fid': feed_id})
        logger.info(f"Successfully deleted stop_routes for feed {feed_id}")

    # ── Full Pipeline ────────────────────────────────────────────────────

    def has_feeds_loaded(self) -> bool:
        """Check if any GTFS feeds are already loaded in the database.

        Used to skip redundant setup() calls when feeds were already
        loaded earlier in the pipeline (e.g., by run_experiment.setup_network).
        """
        from sqlalchemy import text
        try:
            with self.db_manager.Session() as session:
                count = session.execute(text("SELECT COUNT(*) FROM gtfs_feeds")).scalar()
                return count > 0
        except Exception:
            return False

    def setup(self) -> None:
        """
        Run the full GTFS setup pipeline:
        1. Compute region bbox
        2. Discover feeds
        3. Download feeds
        4. Load into database

        Called during PlanGenerator initialization when GTFS modes are configured.
        """
        t0 = time.time()
        logger.info("=" * 50)
        logger.info("GTFS SETUP")
        logger.info("=" * 50)

        # Check if any mode uses GTFS availability
        modes_config = self.config.get('modes', {})
        gtfs_modes = []
        for mode_name, mode_cfg in modes_config.items():
            if not isinstance(mode_cfg, dict):
                continue
            if not mode_cfg.get('enabled', True):
                continue
            avail = mode_cfg.get('availability', 'universal')
            if isinstance(avail, dict) and avail.get('type') == 'gtfs':
                gtfs_modes.append(mode_name)

        if not gtfs_modes:
            logger.info("No modes configured with GTFS availability, skipping GTFS setup")
            return

        logger.info(f"GTFS modes configured: {gtfs_modes}")

        # 1. Compute region bbox
        region_bbox = self.compute_region_bbox()

        # 2. Discover feeds
        feeds = self.discover_feeds(region_bbox)
        if not feeds:
            logger.warning("No GTFS feeds found for region. Transit availability will be disabled.")
            return

        # 3. Clean up any partially loaded feeds from previous interrupted runs
        self._cleanup_partial_feeds()

        # 4. Download and load each feed
        loaded = 0
        skipped = 0
        failed = 0
        for feed in tqdm(feeds, desc="Loading GTFS feeds", unit="feed"):
            feed_path = self.download_feed(feed)
            if feed_path is None:
                failed += 1
                continue

            success = self.load_feed_to_db(feed_path, feed)
            if success:
                if self._feed_was_already_loaded:
                    skipped += 1
                else:
                    loaded += 1
            else:
                failed += 1

        elapsed = time.time() - t0
        logger.info(f"GTFS setup complete in {elapsed:.1f}s: "
                    f"{loaded} newly loaded, {skipped} already in DB, {failed} failed "
                    f"(total {len(feeds)} feeds)")
        logger.info("=" * 50)
