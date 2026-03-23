"""
MATSim Counts Generator

Generates counts.xml from traffic count data for MATSim validation.
Supports two data sources:
  - FHA/TMAS — loaded from DB after FHACountsManager.setup() (skipped when weight=0)
  - Custom CSVs — user-provided counts_stations.csv + counts_volumes.csv (skipped when weight=0)

When both sources match to the same network link, volumes are blended using
configurable weights. Otherwise each source's volumes are used at 100%.
A source with weight=0 is skipped entirely (not loaded or matched).
"""

import gzip
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from pyproj import Transformer
from rtree import index
from shapely.geometry import Point, LineString

from models.models import FHAStation, FHAHourlyVolume
from utils.logger import setup_logger

logger = setup_logger(__name__)

HOUR_COLS = [f'h{i:02d}' for i in range(1, 25)]
# Uppercase versions for matched_devices.csv (evaluator compatibility)
HOUR_COLS_UPPER = [f'H{i:02d}' for i in range(1, 25)]


class CountsGenerator:
    """Generate MATSim counts.xml from FHA and/or custom traffic count data."""

    def __init__(self, config: Dict, db_manager=None):
        """
        Args:
            config: Main configuration dictionary.
            db_manager: DBManager instance for querying FHA data from DB.
        """
        self.config = config
        self.db_manager = db_manager

        counts_config = config.get('counts', {})
        fha_config = counts_config.get('fha', {})
        custom_config = counts_config.get('custom', {})

        self.fha_weight = fha_config.get('weight', 0.5)
        self.custom_weight = custom_config.get('weight', 0.5)
        self.custom_enabled = custom_config.get('enabled', False)
        self.year = str(fha_config.get('year', 2024))

        # Custom counts files live in evaluation folder
        eval_config = config.get('evaluation', {})
        self.eval_dir = Path(eval_config.get('ground_truth_data_dir', 'data/evaluation'))

        # Network data (loaded later)
        self.network_links = None
        self.spatial_index = None
        self.link_geometries = None
        self._reverse_node_index = None  # {(to_node, from_node): link_id}

        # Coordinate transformer (WGS84 to UTM)
        utm_epsg = config['coordinates']['utm_epsg']
        self._transformer = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)

    # ── FHA data loading (from DB) ───────────────────────────────────────────

    def load_fha_stations(self) -> pd.DataFrame:
        """
        Load FHA stations from the database.

        Returns:
            DataFrame with columns: LOCAL_ID, Latitude, Longitude, Directions, source
        """
        if self.db_manager is None:
            logger.warning("No db_manager — cannot load FHA stations")
            return pd.DataFrame()

        stations = self.db_manager.query_all(FHAStation)
        if not stations:
            return pd.DataFrame()

        records = []
        for s in stations:
            records.append({
                'LOCAL_ID': f"FHA_{s.state_code}_{s.station_id}",
                'Latitude': s.lat,
                'Longitude': s.lon,
                'Directions': 'Bidirectional',
                'source': 'fha',
            })
        return pd.DataFrame(records)

    def load_fha_volumes(self) -> pd.DataFrame:
        """
        Load FHA hourly volumes from the database.

        Returns:
            DataFrame with columns: LOCAL_ID, H01..H24
        """
        if self.db_manager is None:
            return pd.DataFrame()

        volumes = self.db_manager.query_all(FHAHourlyVolume)
        if not volumes:
            return pd.DataFrame()

        records = []
        for v in volumes:
            rec = {'LOCAL_ID': f"FHA_{v.state_code}_{v.station_id}"}
            for hcol, hcol_upper in zip(HOUR_COLS, HOUR_COLS_UPPER):
                rec[hcol_upper] = getattr(v, hcol, 0.0) or 0.0
            records.append(rec)
        return pd.DataFrame(records)

    # ── Custom CSV loading ───────────────────────────────────────────────────

    def load_custom_stations(self) -> Optional[pd.DataFrame]:
        """
        Load and validate custom counts_stations.csv.

        Returns:
            DataFrame with columns: LOCAL_ID, Latitude, Longitude, Directions, source
            or None if validation fails.
        """
        stations_path = self.eval_dir / 'counts_stations.csv'
        if not stations_path.exists():
            logger.warning(f"Custom counts: counts_stations.csv not found in "
                           f"{self.eval_dir} — skipping custom counts, using FHA only")
            return None

        df = pd.read_csv(stations_path, encoding='utf-8-sig')
        df.columns = df.columns.str.lower().str.strip()

        required_cols = {'station_id', 'latitude', 'longitude'}
        missing = required_cols - set(df.columns)
        if missing:
            logger.warning(f"Custom counts: counts_stations.csv missing required "
                           f"columns {sorted(missing)} — skipping custom counts, using FHA only")
            return None

        result = pd.DataFrame({
            'LOCAL_ID': df['station_id'].astype(str),
            'Latitude': df['latitude'].astype(float),
            'Longitude': df['longitude'].astype(float),
            'Directions': df['directions'].fillna('Bidirectional') if 'directions' in df.columns
                          else 'Bidirectional',
            'source': 'custom',
        })
        return result

    def load_custom_volumes(self) -> Optional[pd.DataFrame]:
        """
        Load and validate custom counts_volumes.csv. Filters weekdays, averages.

        Returns:
            DataFrame with columns: LOCAL_ID, H01..H24
            or None if validation fails.
        """
        volumes_path = self.eval_dir / 'counts_volumes.csv'
        if not volumes_path.exists():
            logger.warning(f"Custom counts: counts_volumes.csv not found in "
                           f"{self.eval_dir} — skipping custom counts, using FHA only")
            return None

        df = pd.read_csv(volumes_path, encoding='utf-8-sig')
        df.columns = df.columns.str.lower().str.strip()

        # Check required columns
        hour_cols_lower = [f'h{i:02d}' for i in range(1, 25)]
        required_cols = {'station_id', 'date'} | set(hour_cols_lower)
        missing = required_cols - set(df.columns)
        if missing:
            logger.warning(f"Custom counts: counts_volumes.csv missing required "
                           f"columns {sorted(missing)} — skipping custom counts, using FHA only")
            return None

        # Parse dates, filter weekdays
        raw_dates = df['date'].astype(str)
        df['date'] = pd.to_datetime(raw_dates, errors='coerce')
        if df['date'].isna().all():
            # Handle formats like "23-MAY-24 12.00.00.000000000 AM" — extract date part before space
            df['date'] = pd.to_datetime(raw_dates.str.split(' ').str[0],
                                        format='%d-%b-%y', errors='coerce')
        df = df.dropna(subset=['date'])
        df['weekday'] = df['date'].dt.weekday
        df = df[df['weekday'] < 5].copy()  # Mon-Fri

        if df.empty:
            logger.warning("Custom counts: no weekday data in counts_volumes.csv "
                           "— skipping custom counts, using FHA only")
            return None

        # Average by station
        df['station_id'] = df['station_id'].astype(str)
        avg = df.groupby('station_id')[hour_cols_lower].mean().reset_index()

        # Rename to uppercase for compatibility
        rename_map = {lc: uc for lc, uc in zip(hour_cols_lower, HOUR_COLS_UPPER)}
        avg = avg.rename(columns=rename_map)
        avg = avg.rename(columns={'station_id': 'LOCAL_ID'})

        return avg

    def _load_custom_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Try to load both custom files. If either fails validation, return (None, None).
        """
        stations = self.load_custom_stations()
        if stations is None:
            return None, None

        volumes = self.load_custom_volumes()
        if volumes is None:
            return None, None

        # Check join coverage
        matched_ids = set(stations['LOCAL_ID']) & set(volumes['LOCAL_ID'])
        if not matched_ids:
            logger.warning("Custom counts: 0 station_ids matched between stations "
                           "and volumes files — skipping custom counts, using FHA only")
            return None, None

        logger.info(f"Custom counts: loaded {len(stations)} stations, "
                    f"{len(volumes)} volume records")
        return stations, volumes

    # ── Network loading (unchanged from original) ────────────────────────────

    def load_network(self, network_path: Path) -> Tuple[pd.DataFrame, index.Index, Dict]:
        """Load MATSim network and build spatial index."""
        network_path = Path(network_path)

        if str(network_path).endswith('.gz'):
            with gzip.open(network_path, 'rt', encoding='utf-8') as f:
                tree = ET.parse(f)
        else:
            tree = ET.parse(network_path)

        root = tree.getroot()

        nodes = {}
        for node in root.findall('.//node'):
            node_id = node.get('id')
            x = float(node.get('x'))
            y = float(node.get('y'))
            nodes[node_id] = (x, y)

        links_data = []
        link_geometries = {}

        for link in root.findall('.//link'):
            link_id = link.get('id')
            from_node = link.get('from')
            to_node = link.get('to')

            if from_node in nodes and to_node in nodes:
                from_x, from_y = nodes[from_node]
                to_x, to_y = nodes[to_node]

                links_data.append({
                    'link_id': link_id,
                    'from_node': from_node,
                    'to_node': to_node,
                    'from_x': from_x,
                    'from_y': from_y,
                    'to_x': to_x,
                    'to_y': to_y
                })

                link_geometries[link_id] = LineString([(from_x, from_y), (to_x, to_y)])

        links_df = pd.DataFrame(links_data)

        spatial_idx = index.Index()
        for idx, row in links_df.iterrows():
            min_x = min(row['from_x'], row['to_x'])
            max_x = max(row['from_x'], row['to_x'])
            min_y = min(row['from_y'], row['to_y'])
            max_y = max(row['from_y'], row['to_y'])
            spatial_idx.insert(idx, (min_x, min_y, max_x, max_y))

        self.network_links = links_df
        self.spatial_index = spatial_idx
        self.link_geometries = link_geometries

        # Build reverse-node index: (to_node, from_node) → link_id
        # For a link A→B, the reverse link is B→A.
        self._reverse_node_index = {}
        for _, row in links_df.iterrows():
            key = (row['from_node'], row['to_node'])
            self._reverse_node_index[key] = row['link_id']

        return links_df, spatial_idx, link_geometries

    def convert_latlon_to_utm(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert lat/lon to UTM coordinates."""
        x, y = self._transformer.transform(lon, lat)
        return x, y

    def find_nearest_link(self, x: float, y: float,
                          buffer_m: float = 1000) -> Tuple[Optional[str], float]:
        """Find nearest link to given UTM coordinates."""
        if self.spatial_index is None or self.network_links is None:
            raise ValueError("Network not loaded. Call load_network() first.")

        point = Point(x, y)
        nearby_indices = list(self.spatial_index.intersection(
            (x - buffer_m, y - buffer_m, x + buffer_m, y + buffer_m)
        ))

        if not nearby_indices:
            return None, float('inf')

        min_distance = float('inf')
        nearest_link_id = None

        for idx in nearby_indices:
            link_id = self.network_links.iloc[idx]['link_id']
            link_geom = self.link_geometries[link_id]
            distance = point.distance(link_geom)

            if distance < min_distance:
                min_distance = distance
                nearest_link_id = link_id

        return nearest_link_id, min_distance

    def get_reverse_link_id(self, link_id: str) -> Optional[str]:
        """
        Get the reverse direction link ID.

        Tries three strategies:
        1. Suffix-based: swap 'f' ↔ 'r' suffix (road-only network convention)
        2. Node-based: find a link with swapped from/to nodes (same-node
           bidirectional roads, e.g. surface streets in pt2matsim)
        3. Spatial antiparallel: find a nearby link (~50 m) pointing in the
           opposite direction (divided highways / dual carriageways where each
           direction uses different OSM nodes)
        """
        if link_id is None:
            return None

        link_id = str(link_id)

        # Strategy 1: suffix-based (backward compat with road-only networks)
        if link_id.endswith('f'):
            reverse_id = link_id[:-1] + 'r'
        elif link_id.endswith('r'):
            reverse_id = link_id[:-1] + 'f'
        else:
            reverse_id = None

        if reverse_id and self.link_geometries is not None and reverse_id in self.link_geometries:
            return reverse_id

        # Strategy 2: node-based reverse lookup (pt2matsim / generic networks)
        if self._reverse_node_index is not None and self.network_links is not None:
            row = self.network_links[self.network_links['link_id'] == link_id]
            if not row.empty:
                from_node = row.iloc[0]['from_node']
                to_node = row.iloc[0]['to_node']
                candidate = self._reverse_node_index.get((to_node, from_node))
                if candidate and candidate != link_id:
                    return candidate

        # Strategy 3: spatial antiparallel search (divided highways)
        if self.spatial_index is not None and self.network_links is not None:
            return self._find_antiparallel_link(link_id)

        return None

    def _find_antiparallel_link(self, link_id: str,
                                buffer_m: float = 50.0,
                                cos_threshold: float = -0.7) -> Optional[str]:
        """
        Find a nearby link pointing in the opposite direction.

        Used for divided highways where the two carriageways have separate
        OSM nodes.  Searches within *buffer_m* metres of the link midpoint
        and returns the closest antiparallel link (cosine of direction
        vectors < *cos_threshold*, i.e. within ~45° of opposite).
        """
        row = self.network_links[self.network_links['link_id'] == link_id]
        if row.empty:
            return None
        row = row.iloc[0]

        # Direction vector of the primary link
        dx = row['to_x'] - row['from_x']
        dy = row['to_y'] - row['from_y']
        length = np.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            return None
        ux, uy = dx / length, dy / length

        # Search around the link midpoint
        mx = (row['from_x'] + row['to_x']) / 2.0
        my = (row['from_y'] + row['to_y']) / 2.0

        nearby_indices = list(self.spatial_index.intersection(
            (mx - buffer_m, my - buffer_m, mx + buffer_m, my + buffer_m)
        ))

        best_id = None
        best_dist = float('inf')

        midpoint = Point(mx, my)

        for idx in nearby_indices:
            cand = self.network_links.iloc[idx]
            cand_id = cand['link_id']
            if cand_id == link_id:
                continue

            # Direction vector of candidate
            cdx = cand['to_x'] - cand['from_x']
            cdy = cand['to_y'] - cand['from_y']
            clen = np.sqrt(cdx * cdx + cdy * cdy)
            if clen < 1e-6:
                continue

            # Cosine similarity (< -0.7 ≈ within ~45° of opposite)
            cos_sim = (ux * cdx + uy * cdy) / clen
            if cos_sim > cos_threshold:
                continue

            # Distance from primary midpoint to candidate geometry
            cand_geom = self.link_geometries[cand_id]
            dist = midpoint.distance(cand_geom)
            if dist < best_dist:
                best_dist = dist
                best_id = cand_id

        return best_id

    # ── Device matching ──────────────────────────────────────────────────────

    def filter_devices_in_network(self, device_locations: pd.DataFrame,
                                   buffer_m: float = 1000) -> pd.DataFrame:
        """Filter devices to only those within the network extent."""
        if self.spatial_index is None:
            raise ValueError("Network not loaded. Call load_network() first.")

        device_locations = device_locations.copy()

        utm_coords = device_locations.apply(
            lambda row: self.convert_latlon_to_utm(row['Latitude'], row['Longitude']),
            axis=1
        )
        device_locations['utm_x'] = utm_coords.apply(lambda c: c[0])
        device_locations['utm_y'] = utm_coords.apply(lambda c: c[1])

        def is_in_network(row):
            nearby_links = list(self.spatial_index.intersection(
                (row['utm_x'] - buffer_m, row['utm_y'] - buffer_m,
                 row['utm_x'] + buffer_m, row['utm_y'] + buffer_m)
            ))
            return len(nearby_links) > 0

        in_network_mask = device_locations.apply(is_in_network, axis=1)
        return device_locations[in_network_mask].copy()

    def match_devices_to_links(self, ground_truth: pd.DataFrame,
                                device_locations: pd.DataFrame) -> pd.DataFrame:
        """Match traffic counting devices to nearest MATSim network links."""
        df = ground_truth.merge(device_locations, on='LOCAL_ID', how='inner')

        if 'utm_x' not in df.columns:
            utm_coords = df.apply(
                lambda row: self.convert_latlon_to_utm(row['Latitude'], row['Longitude']),
                axis=1
            )
            df['utm_x'] = utm_coords.apply(lambda c: c[0])
            df['utm_y'] = utm_coords.apply(lambda c: c[1])

        matches = df.apply(
            lambda row: self.find_nearest_link(row['utm_x'], row['utm_y']),
            axis=1
        )
        df['matched_link_id'] = matches.apply(lambda m: m[0])
        df['distance_m'] = matches.apply(lambda m: m[1])
        df['reverse_link_id'] = df['matched_link_id'].apply(self.get_reverse_link_id)

        df = df[df['matched_link_id'].notna()]
        return df

    # ── Blending ─────────────────────────────────────────────────────────────

    def _blend_matched(self, fha_matched: pd.DataFrame,
                       custom_matched: pd.DataFrame) -> pd.DataFrame:
        """
        Blend FHA and custom matched data at the link level.

        For links matched by both sources, apply weighted average.
        For links matched by only one source, use that source at 100%.
        """
        w_fha = self.fha_weight
        w_custom = self.custom_weight
        w_total = w_fha + w_custom

        # Index by matched_link_id
        fha_by_link = {row['matched_link_id']: row for _, row in fha_matched.iterrows()}
        custom_by_link = {row['matched_link_id']: row for _, row in custom_matched.iterrows()}

        all_link_ids = set(fha_by_link.keys()) | set(custom_by_link.keys())
        overlap_count = len(set(fha_by_link.keys()) & set(custom_by_link.keys()))

        blended_rows = []
        for link_id in all_link_ids:
            fha_row = fha_by_link.get(link_id)
            custom_row = custom_by_link.get(link_id)

            if fha_row is not None and custom_row is not None:
                # Blend
                row = dict(fha_row)
                for hcol in HOUR_COLS_UPPER:
                    fha_val = fha_row.get(hcol, 0.0) or 0.0
                    custom_val = custom_row.get(hcol, 0.0) or 0.0
                    row[hcol] = (w_fha * fha_val + w_custom * custom_val) / w_total
                row['source'] = 'blended'
                blended_rows.append(row)
            elif fha_row is not None and w_fha > 0:
                row = dict(fha_row)
                row['source'] = 'fha'
                blended_rows.append(row)
            elif custom_row is not None and w_custom > 0:
                row = dict(custom_row)
                row['source'] = 'custom'
                blended_rows.append(row)

        if overlap_count > 0:
            logger.info(f"Blended {overlap_count} overlapping links "
                        f"(weights: FHA={w_fha}, custom={w_custom})")

        return pd.DataFrame(blended_rows)

    # ── County boundary filter ────────────────────────────────────────────────

    def _load_county_polygon_union(self):
        """
        Load county boundary polygons from Census TIGER shapefile and return
        their union in UTM.

        Uses region.counties (FIPS GEOIDs) from config. Downloads the shapefile
        on first use to {data_dir}/counties/.

        Returns a single Shapely geometry (union of all county polygons) or None.
        """
        county_geoids = self.config.get('region', {}).get('counties', [])
        utm_epsg = self.config.get('coordinates', {}).get('utm_epsg')
        data_dir = self.config.get('data', {}).get('data_dir', '')

        if not county_geoids or not utm_epsg or not data_dir:
            return None

        try:
            from shapely.ops import unary_union
            from utils.region_utils import load_county_polygons

            polygons = load_county_polygons(county_geoids, data_dir, utm_epsg=utm_epsg)
            if polygons:
                return unary_union(polygons)
        except Exception as e:
            logger.warning(f"Failed to load county polygons: {e}")

        return None

    def _filter_by_county_boundary(self, matched: pd.DataFrame) -> pd.DataFrame:
        """
        Filter matched devices to only those whose matched link falls within
        the configured county boundaries.

        Devices matched to buffer-zone links (outside county polygons) are
        removed to avoid false poor-GEH results from near-zero simulated volume.
        """
        county_union = self._load_county_polygon_union()
        if county_union is None:
            logger.debug("County polygon not available, skipping boundary filter")
            return matched

        from shapely.prepared import prep
        prepared_poly = prep(county_union)

        # Build link midpoint lookup from network_links
        link_midpoints = {}
        for _, row in self.network_links.iterrows():
            mid_x = (row['from_x'] + row['to_x']) / 2
            mid_y = (row['from_y'] + row['to_y']) / 2
            link_midpoints[row['link_id']] = Point(mid_x, mid_y)

        def link_in_county(link_id):
            pt = link_midpoints.get(link_id)
            if pt is None:
                return False
            return prepared_poly.contains(pt)

        mask = matched['matched_link_id'].apply(link_in_county)
        filtered = matched[mask].copy()
        n_removed = len(matched) - len(filtered)

        if n_removed > 0:
            logger.info(f"County boundary filter: removed {n_removed} devices "
                        f"matched to buffer-zone links (outside county polygons)")
        logger.info(f"County boundary filter: {len(filtered)} devices retained")

        return filtered

    @staticmethod
    def _accumulate_link_entry(link_entries: Dict, loc_id: str, cs_id: str,
                               hourly_volumes: List[int]):
        """Accumulate hourly volumes per link ID, summing when multiple stations
        map to the same link.  Each entry tracks contributing station names."""
        loc_id = str(loc_id)
        if loc_id not in link_entries:
            link_entries[loc_id] = {
                'volumes': list(hourly_volumes),
                'cs_ids': [cs_id],
                'n': 1,
            }
        else:
            entry = link_entries[loc_id]
            for i, v in enumerate(hourly_volumes):
                entry['volumes'][i] += v
            entry['cs_ids'].append(cs_id)
            entry['n'] += 1

    # ── Main pipeline ────────────────────────────────────────────────────────

    def generate_counts_xml(self, network_path: Path,
                            output_path: Path) -> Tuple[Optional[Path], Dict]:
        """
        Generate counts.xml file for MATSim.

        Returns:
            Tuple of (output_path, metadata_dict) or (None, metadata) if no data.
        """
        logger.info("Generating counts.xml for MATSim validation...")

        # Load network
        logger.info(f"Loading network from {network_path}")
        self.load_network(network_path)
        logger.info(f"Loaded {len(self.network_links):,} links")

        # ── Load FHA data (skipped when weight is 0) ─────────────────────────
        fha_matched = pd.DataFrame()
        if self.fha_weight > 0:
            fha_stations = self.load_fha_stations()
            fha_volumes = self.load_fha_volumes()

            if not fha_stations.empty and not fha_volumes.empty:
                filtered_fha = self.filter_devices_in_network(fha_stations)
                logger.info(f"FHA: {len(filtered_fha)} stations within network area "
                            f"(of {len(fha_stations)} total)")
                if not filtered_fha.empty:
                    fha_matched = self.match_devices_to_links(fha_volumes, filtered_fha)
                    logger.info(f"FHA: matched {len(fha_matched)} stations to network links")
            else:
                logger.warning("FHA: no station/volume data available")
        else:
            logger.info("FHA: skipped (weight=0)")

        # ── Load custom data (skipped when weight is 0) ──────────────────────
        custom_matched = pd.DataFrame()
        if self.custom_enabled and self.custom_weight > 0:
            custom_stations, custom_volumes = self._load_custom_data()
            if custom_stations is not None and custom_volumes is not None:
                filtered_custom = self.filter_devices_in_network(custom_stations)
                logger.info(f"Custom: {len(filtered_custom)} stations within network area "
                            f"(of {len(custom_stations)} total)")
                if not filtered_custom.empty:
                    custom_matched = self.match_devices_to_links(custom_volumes, filtered_custom)
                    logger.info(f"Custom: matched {len(custom_matched)} stations to network links")

        # ── Determine final matched data ─────────────────────────────────────
        if fha_matched.empty and custom_matched.empty:
            logger.warning("No stations could be matched to network links. "
                           "counts.xml will not be generated.")
            return None, {'num_counts': 0, 'reason': 'no_matches'}

        if not fha_matched.empty and not custom_matched.empty:
            logger.info(f"Counts sources: FHA ({len(fha_matched)} stations) + "
                        f"custom ({len(custom_matched)} stations)")
            matched = self._blend_matched(fha_matched, custom_matched)
        elif not fha_matched.empty:
            logger.info(f"Counts sources: FHA only ({len(fha_matched)} stations)")
            matched = fha_matched
        else:
            logger.info(f"Counts sources: custom only ({len(custom_matched)} stations)")
            matched = custom_matched

        # Filter out devices matched to buffer-zone links (outside county boundaries)
        matched = self._filter_by_county_boundary(matched)
        if matched.empty:
            logger.warning("All matched devices were outside county boundaries. "
                           "counts.xml will not be generated.")
            return None, {'num_counts': 0, 'reason': 'all_outside_county'}

        # ── Build counts XML ─────────────────────────────────────────────────
        root_elem = ET.Element('counts')
        root_elem.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        root_elem.set('xsi:noNamespaceSchemaLocation',
                      'http://matsim.org/files/dtd/counts_v1.xsd')
        root_elem.set('name', 'Traffic Counts')
        root_elem.set('desc', 'Traffic counts from FHA/TMAS and custom sources')
        root_elem.set('year', self.year)

        # Accumulate volumes per link, averaging when multiple stations match
        link_entries: Dict[str, dict] = {}
        bidirectional_count = 0

        for _, device in matched.iterrows():
            device_id = str(device['LOCAL_ID'])
            link_id = device['matched_link_id']
            reverse_link_id = device.get('reverse_link_id')

            is_bidirectional = (reverse_link_id is not None
                                and pd.notna(reverse_link_id))

            if is_bidirectional:
                bidirectional_count += 1
                fwd_vols = [int(round(device[h] / 2)) for h in HOUR_COLS_UPPER]
                rev_vols = list(fwd_vols)
                self._accumulate_link_entry(link_entries, str(link_id),
                                            f"{device_id}_fwd", fwd_vols)
                self._accumulate_link_entry(link_entries, str(reverse_link_id),
                                            f"{device_id}_rev", rev_vols)
            else:
                vols = [int(round(device[h])) for h in HOUR_COLS_UPPER]
                self._accumulate_link_entry(link_entries, str(link_id),
                                            device_id, vols)

        # Average duplicates and log warnings
        duplicated = 0
        for loc_id, entry in link_entries.items():
            if entry['n'] > 1:
                duplicated += 1
                entry['volumes'] = [int(round(v / entry['n']))
                                    for v in entry['volumes']]
                logger.warning(
                    f"Link {loc_id}: averaged {entry['n']} stations "
                    f"({', '.join(entry['cs_ids'])})")
        if duplicated:
            logger.info(f"Deduplicated {duplicated} links with multiple stations")

        # Write one XML <count> per unique loc_id
        counts_added = 0
        for loc_id, entry in link_entries.items():
            count_elem = ET.SubElement(root_elem, 'count')
            count_elem.set('loc_id', loc_id)
            count_elem.set('cs_id', '+'.join(entry['cs_ids']))
            for hour_idx, vol in enumerate(entry['volumes'], start=1):
                volume_elem = ET.SubElement(count_elem, 'volume')
                volume_elem.set('h', str(hour_idx))
                volume_elem.set('val', str(vol))
            counts_added += 1

        # Pretty print XML
        xml_str = ET.tostring(root_elem, encoding='unicode')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="    ")
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        pretty_xml = '\n'.join(lines)

        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)

        logger.info(f"Generated counts.xml with {counts_added} count locations")
        logger.info(f"  - Bidirectional sensors: {bidirectional_count}")
        logger.info(f"  - Saved to: {output_path}")

        # Save matched devices CSV for evaluator
        matched_devices_path = output_path.parent / 'matched_devices.csv'
        matched.to_csv(matched_devices_path, index=False)
        logger.info(f"  - Saved matched devices to: {matched_devices_path}")

        metadata = {
            'num_devices_total': len(fha_stations) + (len(custom_matched) if not custom_matched.empty else 0),
            'num_devices_matched': len(matched),
            'num_count_locations': counts_added,
            'num_bidirectional': bidirectional_count,
            'year': self.year,
            'matched_devices_path': str(matched_devices_path),
        }

        return output_path, metadata
