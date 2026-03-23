"""
Network generator for MATSim using MATSim's native SupersonicOsmNetworkReader
Downloads OSM data and converts to MATSim network.xml format using official MATSim tools.

When matsim.transit_network is true, uses pt2matsim pipeline instead:
  Step 1: Osm2MultimodalNetwork (OSM PBF → multimodal network with road + rail links)
  Step 2: Gtfs2TransitSchedule per feed (GTFS → unmapped schedule + vehicles, then merge)
  Step 3: PublicTransitMapper (map schedule to network → final network.xml + transitSchedule.xml)
"""

import subprocess
import time
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from matsim.osm_downloader import OSMDownloader

logger = logging.getLogger(__name__)


class NetworkGenerator:
    """Generate MATSim network.xml from OpenStreetMap data using MATSim-native tools"""

    def __init__(self, config: Dict):
        """
        Initialize network generator

        Args:
            config: Configuration dictionary containing MATSim settings
        """
        self.config = config
        self.matsim_config = config.get('matsim', {})
        self.downloader = OSMDownloader()

    def _get_enabled_transit_modes(self) -> list:
        """Return list of enabled mode names that map to MATSim 'pt' (transit modes)."""
        modes_config = self.config.get('modes', {})
        transit_modes = []
        for mode_name, mode_cfg in modes_config.items():
            if not isinstance(mode_cfg, dict):
                continue
            if not mode_cfg.get('enabled', True):
                continue
            if mode_cfg.get('matsim_mode') == 'pt':
                transit_modes.append(mode_name)
        return transit_modes

    def _get_enabled_route_types(self) -> set:
        """Collect GTFS route_type codes from all enabled transit modes.

        Scans the modes config for enabled modes with GTFS availability
        and returns the union of their ``route_types`` lists.  This is used
        to filter GTFS feeds so that only route types belonging to enabled
        modes are passed to pt2matsim.
        """
        modes_config = self.config.get('modes', {})
        route_types: set = set()
        for mode_name, mode_cfg in modes_config.items():
            if not isinstance(mode_cfg, dict):
                continue
            if not mode_cfg.get('enabled', True):
                continue
            if mode_cfg.get('matsim_mode') != 'pt':
                continue
            avail = mode_cfg.get('availability', 'universal')
            if isinstance(avail, dict) and avail.get('type') == 'gtfs':
                for rt in avail.get('route_types', []):
                    route_types.add(int(rt))
        return route_types

    def get_matsim_jar_path(self) -> Path:
        """
        Get path to MATSim JAR file

        Returns:
            Path to MATSim JAR file
        """
        matsim_version = self.matsim_config.get('version', 'matsim_25')
        base_path = Path(__file__).parent / matsim_version

        # Find MATSim JAR
        jar_files = list(base_path.glob('*.jar'))
        if not jar_files:
            raise FileNotFoundError(f"No MATSim JAR found in {base_path}")

        jar_path = jar_files[0]
        logger.info(f"Using MATSim JAR: {jar_path}")

        return jar_path

    def get_pt2matsim_jar_path(self) -> Path:
        """Get path to pt2matsim JAR in matsim/pt2matsim/."""
        base_path = Path(__file__).parent / 'pt2matsim'
        jar_files = list(base_path.glob('pt2matsim*.jar'))
        if not jar_files:
            raise FileNotFoundError(
                f"No pt2matsim JAR found in {base_path}. "
                "Download pt2matsim shaded JAR from "
                "https://github.com/matsim-org/pt2matsim/releases"
            )
        jar_path = jar_files[0]
        logger.info(f"Using pt2matsim JAR: {jar_path}")
        return jar_path

    def clean_network(
        self,
        input_network: Path,
        output_network: Path
    ) -> Dict:
        """
        Clean network using MATSim's NetworkCleaner
        Removes isolated nodes and simplifies the network

        Args:
            input_network: Path to input network.xml
            output_network: Path to save cleaned network.xml

        Returns:
            Dictionary with cleaning metadata
        """
        if not input_network.exists():
            raise FileNotFoundError(f"Input network not found: {input_network}")

        logger.info(f"Cleaning network using org.matsim.run.NetworkCleaner")
        logger.info(f"  Input: {input_network}")
        logger.info(f"  Output: {output_network}")

        # Get MATSim JAR
        jar_path = self.get_matsim_jar_path()

        # Run NetworkCleaner
        cmd = [
            'java',
            '-cp', str(jar_path),
            'org.matsim.run.NetworkCleaner',
            str(input_network.resolve()),
            str(output_network.resolve())
        ]

        logger.info(f"Running NetworkCleaner...")
        logger.info(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Network cleaning failed: {result.stderr}")
            raise RuntimeError(f"Network cleaning failed: {result.stderr}")

        # Log output
        logger.info(result.stdout)

        # Parse cleaned network to get statistics
        num_nodes, num_links = self._count_network_elements(output_network)
        logger.info(f"Network cleaning complete: {num_nodes} nodes, {num_links} links")

        return {
            'num_nodes': num_nodes,
            'num_links': num_links,
            'output_path': str(output_network)
        }

    @staticmethod
    def _sanitize_gtfs_shapes(shapes_file: Path) -> None:
        """Sanitize shapes.txt to fix common formatting issues.

        pt2matsim's GtfsFeedImpl.loadShapes crashes on lines with trailing
        whitespace/tabs (NumberFormatException). This reads the file, strips
        trailing whitespace from every field, and rewrites it in-place.
        Raises on I/O errors so the caller can fall back to removing the file.
        """
        import csv
        with open(shapes_file, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = []
            for row in reader:
                rows.append([field.strip() for field in row])

        with open(shapes_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def _filter_gtfs_feed_to_bbox(
        self,
        feed_dir: Path,
        output_dir: Path,
        bbox: Tuple[float, float, float, float],
        feed_label: str = '',
    ) -> Optional[Tuple[Path, Dict]]:
        """Filter a GTFS feed directory to only include stops within a bounding box.

        Creates a filtered copy of the feed with only stops inside the bbox,
        and only routes/trips/stop_times that reference those stops.
        All other GTFS files (agency.txt, calendar.txt, etc.) are copied unchanged.

        Args:
            feed_dir: Path to original GTFS feed directory (contains stops.txt, etc.)
            output_dir: Parent directory for the filtered copy
            bbox: (min_lon, min_lat, max_lon, max_lat)
            feed_label: Human-readable label for logging (e.g. "11 (Amtrak)")

        Returns:
            (filtered_feed_dir, stats_dict) or None if all stops are outside bbox
        """
        import csv
        import shutil

        min_lon, min_lat, max_lon, max_lat = bbox
        feed_id = feed_dir.name
        filtered_dir = output_dir / f'{feed_id}_filtered'
        filtered_dir.mkdir(parents=True, exist_ok=True)

        label = feed_label or feed_id
        logger.info(f"  Filtering GTFS feed {label} to region bbox "
                     f"({min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f})")

        # --- Step 1: Filter stops.txt ---
        stops_file = feed_dir / 'stops.txt'
        if not stops_file.exists():
            logger.warning(f"    No stops.txt in feed {feed_id}, skipping filter")
            return None

        retained_stop_ids = set()
        original_stop_count = 0
        with open(stops_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames

        original_stop_count = len(rows)
        filtered_stops = []
        for row in rows:
            try:
                lat = float(row['stop_lat'])
                lon = float(row['stop_lon'])
            except (ValueError, KeyError):
                continue
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                retained_stop_ids.add(row['stop_id'])
                filtered_stops.append(row)

        if not retained_stop_ids:
            logger.info(f"    Original: {original_stop_count:,} stops -> "
                         f"0 stops within bbox")
            logger.info(f"    Skipping feed {label}: no stops within simulation region")
            return None

        with open(filtered_dir / 'stops.txt', 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_stops)

        # --- Step 2: Filter stop_times.txt → collect retained trip_ids ---
        retained_trip_ids = set()
        stop_times_file = feed_dir / 'stop_times.txt'
        if stop_times_file.exists():
            with open(stop_times_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                st_rows = list(reader)
                st_fieldnames = reader.fieldnames

            filtered_stop_times = []
            for row in st_rows:
                if row.get('stop_id') in retained_stop_ids:
                    retained_trip_ids.add(row['trip_id'])
                    filtered_stop_times.append(row)

            with open(filtered_dir / 'stop_times.txt', 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=st_fieldnames)
                writer.writeheader()
                writer.writerows(filtered_stop_times)

        # --- Step 3: Filter trips.txt → collect retained route_ids ---
        retained_route_ids = set()
        trips_file = feed_dir / 'trips.txt'
        if trips_file.exists():
            with open(trips_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                trip_rows = list(reader)
                trip_fieldnames = reader.fieldnames

            original_trip_count = len(trip_rows)
            filtered_trips = []
            for row in trip_rows:
                if row.get('trip_id') in retained_trip_ids:
                    retained_route_ids.add(row['route_id'])
                    filtered_trips.append(row)

            with open(filtered_dir / 'trips.txt', 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trip_fieldnames)
                writer.writeheader()
                writer.writerows(filtered_trips)
        else:
            original_trip_count = 0

        # --- Step 4: Filter routes.txt ---
        routes_file = feed_dir / 'routes.txt'
        original_route_count = 0
        if routes_file.exists():
            with open(routes_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                route_rows = list(reader)
                route_fieldnames = reader.fieldnames

            original_route_count = len(route_rows)
            filtered_routes = [r for r in route_rows if r.get('route_id') in retained_route_ids]

            with open(filtered_dir / 'routes.txt', 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=route_fieldnames)
                writer.writeheader()
                writer.writerows(filtered_routes)

        # --- Step 5: Copy all other GTFS files unchanged ---
        filtered_files = {'stops.txt', 'stop_times.txt', 'trips.txt', 'routes.txt'}
        for src_file in feed_dir.iterdir():
            if src_file.is_file() and src_file.name not in filtered_files and not src_file.name.startswith('.'):
                shutil.copy2(src_file, filtered_dir / src_file.name)

        # --- Log results ---
        stop_reduction = (1 - len(filtered_stops) / original_stop_count) * 100 if original_stop_count else 0
        logger.info(f"    Original: {original_stop_count:,} stops, "
                     f"{original_route_count:,} routes, "
                     f"{original_trip_count:,} trips")
        logger.info(f"    Filtered: {len(filtered_stops):,} stops, "
                     f"{len(retained_route_ids):,} routes, "
                     f"{len(retained_trip_ids):,} trips "
                     f"(removed {stop_reduction:.1f}% of stops)")
        logger.info(f"    Filtered feed written to: {filtered_dir}")

        stats = {
            'stops_original': original_stop_count,
            'stops_filtered': len(filtered_stops),
            'routes_original': original_route_count,
            'routes_filtered': len(retained_route_ids),
            'trips_original': original_trip_count,
            'trips_filtered': len(retained_trip_ids),
            'stop_reduction_pct': stop_reduction,
        }
        return filtered_dir, stats

    def _filter_gtfs_feed_by_route_type(
        self,
        feed_dir: Path,
        allowed_route_types: set,
        feed_label: str = '',
    ) -> Optional[Dict]:
        """Remove routes whose route_type is not in the allowed set.

        Modifies the feed directory **in-place** (should be called on an
        already-filtered copy, not the original cache).  After removing
        disallowed routes from ``routes.txt``, cascading deletes are applied
        to ``trips.txt``, ``stop_times.txt`` and ``stops.txt``.

        Args:
            feed_dir: Path to GTFS feed directory (will be modified in-place)
            allowed_route_types: Set of integer GTFS route_type codes to keep
            feed_label: Human-readable label for logging

        Returns:
            Stats dict, or None if no routes survive the filter
        """
        import csv

        label = feed_label or feed_dir.name
        routes_file = feed_dir / 'routes.txt'
        if not routes_file.exists():
            logger.warning(f"    No routes.txt in {label}, skipping route_type filter")
            return None

        # --- Filter routes.txt ---
        with open(routes_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            route_rows = list(reader)
            route_fieldnames = reader.fieldnames

        original_route_count = len(route_rows)
        kept_routes = []
        kept_route_ids = set()
        removed_types: dict = {}  # route_type -> count removed
        for row in route_rows:
            try:
                rt = int(row.get('route_type', '-1'))
            except (ValueError, TypeError):
                rt = -1
            if rt in allowed_route_types:
                kept_routes.append(row)
                kept_route_ids.add(row['route_id'])
            else:
                removed_types[rt] = removed_types.get(rt, 0) + 1

        if not kept_route_ids:
            logger.info(f"    Feed {label}: all {original_route_count} routes removed "
                         f"by route_type filter {allowed_route_types} — skipping feed")
            return None

        if removed_types:
            logger.info(f"    Feed {label}: removed {sum(removed_types.values())} route(s) "
                         f"with disabled route_types: {removed_types}")

        with open(routes_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=route_fieldnames)
            writer.writeheader()
            writer.writerows(kept_routes)

        # --- Cascade to trips.txt ---
        trips_file = feed_dir / 'trips.txt'
        kept_trip_ids = set()
        if trips_file.exists():
            with open(trips_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                trip_rows = list(reader)
                trip_fieldnames = reader.fieldnames

            original_trip_count = len(trip_rows)
            kept_trips = []
            for row in trip_rows:
                if row.get('route_id') in kept_route_ids:
                    kept_trips.append(row)
                    kept_trip_ids.add(row['trip_id'])

            with open(trips_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trip_fieldnames)
                writer.writeheader()
                writer.writerows(kept_trips)
        else:
            original_trip_count = 0

        # --- Cascade to stop_times.txt ---
        stop_times_file = feed_dir / 'stop_times.txt'
        kept_stop_ids = set()
        if stop_times_file.exists():
            with open(stop_times_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                st_rows = list(reader)
                st_fieldnames = reader.fieldnames

            kept_stop_times = []
            for row in st_rows:
                if row.get('trip_id') in kept_trip_ids:
                    kept_stop_times.append(row)
                    kept_stop_ids.add(row.get('stop_id'))

            with open(stop_times_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=st_fieldnames)
                writer.writeheader()
                writer.writerows(kept_stop_times)

        # --- Cascade to stops.txt ---
        stops_file = feed_dir / 'stops.txt'
        if stops_file.exists() and kept_stop_ids:
            with open(stops_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                stop_rows = list(reader)
                stop_fieldnames = reader.fieldnames

            original_stop_count = len(stop_rows)
            kept_stops = [r for r in stop_rows if r.get('stop_id') in kept_stop_ids]

            with open(stops_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=stop_fieldnames)
                writer.writeheader()
                writer.writerows(kept_stops)
        else:
            original_stop_count = 0
            kept_stops = []

        logger.info(f"    Route-type filter: {len(kept_routes)}/{original_route_count} routes, "
                     f"{len(kept_trip_ids)}/{original_trip_count} trips, "
                     f"{len(kept_stops) if kept_stop_ids else 0}/{original_stop_count} stops")

        return {
            'routes_kept': len(kept_routes),
            'routes_removed': original_route_count - len(kept_routes),
            'trips_kept': len(kept_trip_ids),
            'stops_kept': len(kept_stops) if kept_stop_ids else 0,
        }

    def convert_osm_to_network(
        self,
        osm_file: Path,
        output_path: Path,
        coordinate_system: str
    ) -> Dict:
        """
        Convert OSM file to MATSim network using RunSimpleNetworkReaderJalal

        Args:
            osm_file: Path to OSM PBF or XML file
            output_path: Where to save network.xml
            coordinate_system: Target EPSG code (e.g., 'EPSG:26915')

        Returns:
            Dictionary with network metadata
        """
        if not osm_file.exists():
            raise FileNotFoundError(f"OSM file not found: {osm_file}")

        logger.info(f"Converting OSM to MATSim network using RunSimpleNetworkReaderJalal")
        logger.info(f"  Input: {osm_file}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  CRS: {coordinate_system}")

        # Get MATSim JAR
        jar_path = self.get_matsim_jar_path()

        # Run the conversion using RunSimpleNetworkReaderJalal (with coordinate fix)
        cmd = [
            'java',
            '-cp', str(jar_path),
            'org.matsim.contrib.osm.examples.RunSimpleNetworkReaderJalal',
            str(osm_file.resolve()),
            str(output_path.resolve()),
            coordinate_system
        ]

        logger.info(f"Running OSM network converter...")
        logger.info(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Conversion failed: {result.stderr}")
            raise RuntimeError(f"Network conversion failed: {result.stderr}")

        # Log output
        logger.info(result.stdout)

        # Parse the generated network to get statistics
        num_nodes, num_links = self._count_network_elements(output_path)
        logger.info(f"Network conversion complete: {num_nodes} nodes, {num_links} links")

        return {
            'num_nodes': num_nodes,
            'num_links': num_links,
            'coordinate_system': coordinate_system,
            'output_path': str(output_path)
        }

    # ── pt2matsim pipeline (Phase 4) ────────────────────────────────

    @staticmethod
    def _write_xml_with_doctype(tree: ET.ElementTree, path: Path, doctype: str) -> None:
        """Write an ElementTree with a DOCTYPE inserted after the XML declaration."""
        xml_bytes = ET.tostring(tree.getroot(), encoding='UTF-8', xml_declaration=True)
        xml_str = xml_bytes.decode('UTF-8')
        xml_str = xml_str.replace("?>\n", f"?>\n{doctype}\n", 1)
        path.write_text(xml_str, encoding='UTF-8')

    _DOCTYPE_CONFIG = '<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">'
    _DOCTYPE_NETWORK = '<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v2.dtd">'
    _DOCTYPE_SCHEDULE = '<!DOCTYPE transitSchedule SYSTEM "http://www.matsim.org/files/dtd/transitSchedule_v2.dtd">'

    @classmethod
    def _write_matsim_config_xml(cls, tree: ET.ElementTree, config_path: Path) -> None:
        """Write an ElementTree as MATSim config XML with the required DOCTYPE."""
        cls._write_xml_with_doctype(tree, config_path, cls._DOCTYPE_CONFIG)

    def _write_osm_config_xml(
        self, osm_file: Path, output_network: Path, coordinate_system: str, config_path: Path
    ) -> None:
        """Write XML config for Osm2MultimodalNetwork."""
        config = ET.Element('config')
        module = ET.SubElement(config, 'module', name='OsmConverter')

        params = {
            'osmFile': str(osm_file.resolve()),
            'outputCoordinateSystem': coordinate_system,
            'outputNetworkFile': str(output_network.resolve()),
            'keepPaths': 'false',
            'keepTagsAsAttributes': 'true',
            'keepWaysWithPublicTransit': 'true',
            'maxLinkLength': '500.0',
        }
        for name, value in params.items():
            ET.SubElement(module, 'param', name=name, value=value)

        # Determine enabled transit modes for network generation
        enabled_transit = self._get_enabled_transit_modes()
        bus_enabled = 'bus' in enabled_transit
        rail_enabled = 'rail' in enabled_transit

        # Highway wayDefaultParams: (osmValue, lanes, freespeed m/s, capacity, oneway, modes)
        # Bus-capable roads include 'bus' in allowed modes only if bus mode is enabled
        car_bus = 'car,bus' if bus_enabled else 'car'
        highway_defaults = [
            ('motorway',      '2.0', '33.33', '2000.0', 'true',  'car'),
            ('motorway_link', '1.0', '22.22', '1500.0', 'true',  'car'),
            ('trunk',         '2.0', '22.22', '2000.0', 'false', 'car'),
            ('trunk_link',    '1.0', '13.89', '1500.0', 'false', 'car'),
            ('primary',       '1.0', '22.22', '1500.0', 'false', car_bus),
            ('primary_link',  '1.0', '16.67', '1500.0', 'false', car_bus),
            ('secondary',     '1.0', '8.33',  '1000.0', 'false', car_bus),
            ('secondary_link','1.0', '8.33',  '1000.0', 'false', car_bus),
            ('tertiary',      '1.0', '6.94',  '600.0',  'false', car_bus),
            ('tertiary_link', '1.0', '6.94',  '600.0',  'false', car_bus),
            ('unclassified',  '1.0', '4.17',  '600.0',  'false', car_bus),
            ('residential',   '1.0', '4.17',  '600.0',  'false', car_bus),
            ('living_street', '1.0', '2.78',  '300.0',  'false', 'car'),
        ]
        for osm_val, lanes, speed, cap, oneway, modes in highway_defaults:
            ps = ET.SubElement(module, 'parameterset', type='wayDefaultParams')
            ET.SubElement(ps, 'param', name='osmKey', value='highway')
            ET.SubElement(ps, 'param', name='osmValue', value=osm_val)
            ET.SubElement(ps, 'param', name='lanes', value=lanes)
            ET.SubElement(ps, 'param', name='freespeed', value=speed)
            ET.SubElement(ps, 'param', name='freespeedFactor', value='1.0')
            ET.SubElement(ps, 'param', name='laneCapacity', value=cap)
            ET.SubElement(ps, 'param', name='oneway', value=oneway)
            ET.SubElement(ps, 'param', name='allowedTransportModes', value=modes)

        # Railway wayDefaultParams (only if rail mode is enabled)
        if rail_enabled:
            railway_defaults = [
                ('rail',       '1.0', '44.44', '9999.0', 'false', 'rail'),
                ('tram',       '1.0', '11.11', '9999.0', 'false', 'rail'),
                ('light_rail', '1.0', '22.22', '9999.0', 'false', 'rail'),
            ]
            for osm_val, lanes, speed, cap, oneway, modes in railway_defaults:
                ps = ET.SubElement(module, 'parameterset', type='wayDefaultParams')
                ET.SubElement(ps, 'param', name='osmKey', value='railway')
                ET.SubElement(ps, 'param', name='osmValue', value=osm_val)
                ET.SubElement(ps, 'param', name='lanes', value=lanes)
                ET.SubElement(ps, 'param', name='freespeed', value=speed)
                ET.SubElement(ps, 'param', name='freespeedFactor', value='1.0')
                ET.SubElement(ps, 'param', name='laneCapacity', value=cap)
                ET.SubElement(ps, 'param', name='oneway', value=oneway)
                ET.SubElement(ps, 'param', name='allowedTransportModes', value=modes)

        # Routable subnetworks: always car, plus enabled transit modes
        routable_subnetworks = [('car', 'car')]
        if bus_enabled:
            routable_subnetworks.append(('bus', 'bus,car'))
        if rail_enabled:
            routable_subnetworks.append(('rail', 'rail,light_rail'))
        for mode_name, allowed_modes in routable_subnetworks:
            ps = ET.SubElement(module, 'parameterset', type='routableSubnetwork')
            ET.SubElement(ps, 'param', name='subnetworkMode', value=mode_name)
            ET.SubElement(ps, 'param', name='allowedTransportModes', value=allowed_modes)

        tree = ET.ElementTree(config)
        ET.indent(tree, space='    ')
        self._write_matsim_config_xml(tree, config_path)

    def convert_osm_to_multimodal_network(
        self, osm_file: Path, output_path: Path, coordinate_system: str
    ) -> Dict:
        """Step 1: Run Osm2MultimodalNetwork (OSM → multimodal network).

        pt2matsim only reads OSM XML, not PBF. If the input is .pbf,
        it is converted to .osm XML via osmium first.

        NetworkCleaner is NOT run — pt2matsim does per-mode cluster cleaning internally.
        """
        if not osm_file.exists():
            raise FileNotFoundError(f"OSM file not found: {osm_file}")

        logger.info("Step 1: Converting OSM to multimodal network via pt2matsim")
        logger.info(f"  Input:  {osm_file}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  CRS:    {coordinate_system}")

        # pt2matsim requires OSM XML — convert PBF if needed
        osm_xml_file = osm_file
        temp_osm_xml = None
        if osm_file.suffix == '.pbf' or osm_file.name.endswith('.osm.pbf'):
            temp_osm_xml = osm_file.parent / (osm_file.stem.replace('.osm', '') + '_converted.osm')
            logger.info(f"  Converting PBF to OSM XML via osmium: {temp_osm_xml.name}")
            conv_result = subprocess.run(
                ['osmium', 'cat', str(osm_file), '-o', str(temp_osm_xml), '--overwrite'],
                capture_output=True, text=True,
            )
            if conv_result.returncode != 0:
                raise RuntimeError(f"osmium PBF→OSM conversion failed: {conv_result.stderr}")
            osm_xml_file = temp_osm_xml
            logger.info(f"  Converted to OSM XML: {osm_xml_file.stat().st_size / 1e6:.1f} MB")

        jar_path = self.get_pt2matsim_jar_path()

        # Write temp config XML
        config_xml = output_path.parent / 'pt2matsim_osm_config.xml'
        self._write_osm_config_xml(osm_xml_file, output_path, coordinate_system, config_xml)

        cmd = [
            'java', '-cp', str(jar_path),
            'org.matsim.pt2matsim.run.Osm2MultimodalNetwork',
            str(config_xml.resolve())
        ]
        logger.info(f"Command: {' '.join(cmd)}")

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0

        if result.returncode != 0:
            logger.error(f"Osm2MultimodalNetwork failed:\n{result.stderr}")
            raise RuntimeError(f"Osm2MultimodalNetwork failed: {result.stderr}")

        if result.stdout:
            logger.debug(result.stdout[-2000:])

        # Parse stats
        num_nodes, num_links = self._count_network_elements(output_path)
        logger.info(f"Step 1 complete: {num_nodes} nodes, {num_links} links (took {elapsed:.1f}s)")

        # Cleanup temp files
        config_xml.unlink(missing_ok=True)
        if temp_osm_xml is not None:
            temp_osm_xml.unlink(missing_ok=True)

        return {
            'num_nodes': num_nodes,
            'num_links': num_links,
            'coordinate_system': coordinate_system,
            'output_path': str(output_path),
        }

    def convert_gtfs_feeds_to_schedule(
        self, output_dir: Path, coordinate_system: str, db_manager,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[Path, Path]:
        """Step 2: Convert GTFS feeds to unmapped MATSim schedule.

        Queries DuckDB for loaded feeds, optionally filters each feed to
        the region bounding box, then runs Gtfs2TransitSchedule per feed
        and merges per-feed XMLs into one schedule + one vehicles file.

        Args:
            output_dir: Directory for output files
            coordinate_system: Projected CRS (e.g. 'EPSG:26915')
            db_manager: DBManager instance for querying gtfs_feeds table
            bbox: Optional (min_lon, min_lat, max_lon, max_lat) to spatially
                  filter each feed's stops before conversion. Feeds with no
                  stops inside the bbox are skipped entirely.

        Returns:
            (schedule_unmapped_path, vehicles_path)
        """
        from models.models import GTFSFeed

        sample_day = self.matsim_config.get('gtfs_sample_day', 'dayWithMostTrips')
        gtfs_cache_dir = Path(self.config.get('gtfs', {}).get('cache_dir', 'data/gtfs'))
        jar_path = self.get_pt2matsim_jar_path()

        # Query feeds from DB, pre-filtered by bbox to avoid iterating
        # over hundreds of irrelevant feeds from previous experiments
        all_feeds = db_manager.query_all(GTFSFeed)
        if not all_feeds:
            raise RuntimeError(
                "No GTFS feeds found in database. "
                "Run Phase 3 (GTFS discovery/loading) first."
            )

        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            feeds = []
            for f in all_feeds:
                # Keep feeds whose stored bbox intersects our region, or that
                # have no stored bbox (will be filtered by stops later)
                if (f.bbox_min_lat is None or f.bbox_max_lat is None or
                        f.bbox_min_lon is None or f.bbox_max_lon is None):
                    feeds.append(f)
                    continue
                # Check intersection
                if (f.bbox_max_lon >= min_lon and f.bbox_min_lon <= max_lon and
                        f.bbox_max_lat >= min_lat and f.bbox_min_lat <= max_lat):
                    feeds.append(f)
            logger.info(f"  Pre-filtered feeds by region bbox: {len(feeds)} of {len(all_feeds)} in DB")
        else:
            feeds = all_feeds

        # Collect allowed GTFS route_types from enabled modes
        allowed_route_types = self._get_enabled_route_types()
        if allowed_route_types:
            logger.info(f"  Enabled route_types: {sorted(allowed_route_types)}")
        else:
            logger.warning("  No route_types found in enabled modes — all route types will be kept")

        logger.info(f"Step 2: Converting {len(feeds)} GTFS feed(s) to MATSim schedule")
        logger.info(f"  Sample day: {sample_day}")
        logger.info(f"  CRS: {coordinate_system}")
        if bbox:
            logger.info(f"  Spatial filter: bbox ({bbox[0]:.4f}, {bbox[1]:.4f}, "
                         f"{bbox[2]:.4f}, {bbox[3]:.4f})")

        per_feed_dir = output_dir / 'per_feed_transit'
        per_feed_dir.mkdir(parents=True, exist_ok=True)

        schedule_entries = []  # list of (feed_id, schedule_path)
        vehicle_files = []
        skipped_feeds = []

        for feed in feeds:
            feed_id = feed.feed_id
            feed_dir = gtfs_cache_dir / feed_id

            if not feed_dir.exists():
                logger.warning(f"GTFS folder not found for feed {feed_id}: {feed_dir}, skipping")
                continue

            # Spatially filter the feed if bbox is provided
            feed_label = f"{feed_id} ({feed.provider})"
            effective_feed_dir = feed_dir
            if bbox:
                filter_result = self._filter_gtfs_feed_to_bbox(
                    feed_dir, per_feed_dir, bbox, feed_label
                )
                if filter_result is None:
                    skipped_feeds.append(feed_label)
                    continue
                effective_feed_dir = filter_result[0]

            # Filter by enabled route_types (removes rail/subway/ferry if disabled).
            # The route_type filter modifies files in-place, so we need a copy
            # if the bbox filter didn't already create one.
            if allowed_route_types:
                if effective_feed_dir == feed_dir:
                    import shutil
                    copy_dir = per_feed_dir / f'{feed_id}_filtered'
                    if copy_dir.exists():
                        shutil.rmtree(copy_dir)
                    shutil.copytree(feed_dir, copy_dir)
                    effective_feed_dir = copy_dir
                rt_result = self._filter_gtfs_feed_by_route_type(
                    effective_feed_dir, allowed_route_types, feed_label
                )
                if rt_result is None:
                    skipped_feeds.append(f"{feed_label} [no enabled route_types]")
                    continue

            # Sanitize shapes.txt — pt2matsim crashes on malformed shapes
            # (e.g. trailing tabs causing NumberFormatException). shapes.txt is
            # optional per GTFS spec, so removing it is safe.
            shapes_file = effective_feed_dir / 'shapes.txt'
            if shapes_file.exists():
                try:
                    self._sanitize_gtfs_shapes(shapes_file)
                except Exception as exc:
                    logger.warning(f"  Removing malformed shapes.txt for {feed_id}: {exc}")
                    shapes_file.unlink()

            schedule_out = per_feed_dir / f'schedule_{feed_id}.xml'
            vehicles_out = per_feed_dir / f'vehicles_{feed_id}.xml'

            logger.info(f"  Converting feed: {feed.provider} ({feed_id})")

            cmd = [
                'java', '-cp', str(jar_path),
                'org.matsim.pt2matsim.run.Gtfs2TransitSchedule',
                str(effective_feed_dir.resolve()),
                sample_day,
                coordinate_system,
                str(schedule_out.resolve()),
                str(vehicles_out.resolve()),
            ]

            t0 = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.time() - t0

            if result.returncode != 0:
                logger.error(f"Gtfs2TransitSchedule failed for {feed_id}:\n{result.stderr}")
                logger.warning(f"Skipping feed {feed_id} ({feed.provider}) due to conversion error")
                skipped_feeds.append(f"{feed_id} ({feed.provider})")
                continue

            if result.stdout:
                logger.debug(result.stdout[-2000:])

            logger.info(f"    Converted {feed_id} (took {elapsed:.1f}s)")
            schedule_entries.append((feed_id, schedule_out))
            vehicle_files.append(vehicles_out)

        if skipped_feeds:
            logger.info(f"  Skipped {len(skipped_feeds)} feed(s) with no stops in region: "
                         f"{', '.join(skipped_feeds)}")

        if not schedule_entries:
            raise RuntimeError("No GTFS feeds were successfully converted to schedule.")

        # Merge per-feed outputs into single files
        merged_schedule = output_dir / 'schedule_unmapped.xml'
        merged_vehicles = output_dir / 'transitVehicles.xml'

        if len(schedule_entries) == 1:
            # Single feed — just copy
            import shutil
            shutil.copy2(schedule_entries[0][1], merged_schedule)
            shutil.copy2(vehicle_files[0], merged_vehicles)
        else:
            self._merge_schedule_xmls(schedule_entries, merged_schedule)
            self._merge_vehicle_xmls(vehicle_files, merged_vehicles)

        # Validate the merged vehicles file structure
        self._validate_vehicle_xml_structure(merged_vehicles)

        logger.info(f"Step 2 complete: merged {len(schedule_entries)} feed(s)")
        logger.info(f"  Schedule: {merged_schedule}")
        logger.info(f"  Vehicles: {merged_vehicles}")

        return merged_schedule, merged_vehicles

    def _merge_schedule_xmls(self, schedule_entries: list, output: Path) -> None:
        """Merge multiple transitSchedule XMLs, prefixing IDs with feed_id to avoid collisions.

        Args:
            schedule_entries: list of (feed_id, schedule_path) tuples
            output: path for merged schedule XML
        """
        logger.info(f"Merging {len(schedule_entries)} schedule XMLs...")

        # Use first file as base, prefix its IDs
        first_feed_id, first_path = schedule_entries[0]
        base_tree = ET.parse(first_path)
        base_root = base_tree.getroot()

        # Prefix IDs in the base file
        self._prefix_schedule_ids(base_root, first_feed_id)

        existing_stop_ids = set()
        for sf in base_root.iter('stopFacility'):
            existing_stop_ids.add(sf.get('id'))

        total_lines = len(list(base_root.iter('transitLine')))

        # Merge remaining files
        for feed_id, sched_file in schedule_entries[1:]:
            tree = ET.parse(sched_file)
            root = tree.getroot()

            # Prefix IDs for this feed
            self._prefix_schedule_ids(root, feed_id)

            # Merge stop facilities (avoid duplicates)
            base_facilities = base_root.find('transitStops')
            if base_facilities is None:
                base_facilities = ET.SubElement(base_root, 'transitStops')

            for sf in root.iter('stopFacility'):
                if sf.get('id') not in existing_stop_ids:
                    base_facilities.append(sf)
                    existing_stop_ids.add(sf.get('id'))

            # Merge transit lines
            for line in root.iter('transitLine'):
                base_root.append(line)
                total_lines += 1

        logger.info(f"  Merged schedule: {total_lines} transit lines, {len(existing_stop_ids)} stop facilities")

        ET.indent(base_tree, space='    ')
        self._write_xml_with_doctype(base_tree, output, self._DOCTYPE_SCHEDULE)

    @staticmethod
    def _prefix_schedule_ids(root: ET.Element, feed_id: str) -> None:
        """Prefix all transit line and stop facility IDs with feed_id to ensure uniqueness.

        Also updates stop references inside transit routes to match renamed stop IDs.
        """
        prefix = f"f{feed_id}_"

        # Build old->new stop ID mapping
        stop_id_map = {}
        for sf in root.iter('stopFacility'):
            old_id = sf.get('id')
            new_id = prefix + old_id
            sf.set('id', new_id)
            stop_id_map[old_id] = new_id

        # Prefix transit line IDs
        for line in root.iter('transitLine'):
            line.set('id', prefix + line.get('id'))

        # Update stop references in route profiles and route stops
        for stop in root.iter('stop'):
            ref = stop.get('refId')
            if ref and ref in stop_id_map:
                stop.set('refId', stop_id_map[ref])


    _VEHICLES_NS = '{http://www.matsim.org/files/dtd}'

    def _merge_vehicle_xmls(self, vehicle_files: List[Path], output: Path) -> None:
        """Merge multiple transitVehicles XMLs by concatenating vehicle types + vehicles.

        Ensures proper MATSim schema: all vehicleType elements BEFORE all vehicle elements.
        """
        logger.info(f"Merging {len(vehicle_files)} vehicle XMLs...")
        ns = self._VEHICLES_NS

        # Collect all unique vehicle types and vehicles from all files
        all_types = {}  # id -> Element
        all_vehicles = {}  # id -> Element

        for veh_file in vehicle_files:
            tree = ET.parse(veh_file)
            root = tree.getroot()

            for vtype in root.iter(f'{ns}vehicleType'):
                vtype_id = vtype.get('id')
                if vtype_id not in all_types:
                    all_types[vtype_id] = vtype

            for vehicle in root.iter(f'{ns}vehicle'):
                veh_id = vehicle.get('id')
                if veh_id not in all_vehicles:
                    all_vehicles[veh_id] = vehicle

        # Create new root and add all types first, then all vehicles
        base_tree = ET.parse(vehicle_files[0])
        base_root = base_tree.getroot()
        base_root.clear()  # Remove all children

        # Copy namespace attributes from original root
        for key, value in ET.parse(vehicle_files[0]).getroot().attrib.items():
            base_root.set(key, value)

        # Add all vehicle types first (sorted for consistency)
        for vtype_id in sorted(all_types.keys()):
            base_root.append(all_types[vtype_id])

        # Then add all vehicle instances (sorted for consistency)
        for veh_id in sorted(all_vehicles.keys()):
            base_root.append(all_vehicles[veh_id])

        logger.info(f"  Merged vehicles: {len(all_types)} types, {len(all_vehicles)} vehicles")

        ET.indent(base_tree, space='    ')
        base_tree.write(str(output), xml_declaration=True, encoding='UTF-8')

    def _ensure_vehicle_instances(self, vehicles_path: Path, schedule_path: Path) -> int:
        """Ensure every vehicleRefId in the schedule has a <vehicle> in the vehicles file.

        pt2matsim inconsistently generates vehicle instances across feeds.
        This scans the schedule for all vehicleRefId values, checks which are
        missing from the vehicles XML, infers the vehicle type from the suffix
        (e.g. veh_5_bus → Bus), and adds the missing <vehicle> elements.

        Returns the number of vehicle instances added.
        """
        ns = self._VEHICLES_NS

        # Collect all vehicleRefIds from the schedule
        sched_tree = ET.parse(schedule_path)
        ref_ids = set()
        for dep in sched_tree.iter(f'{ns}departure'):
            vid = dep.get('vehicleRefId')
            if vid:
                ref_ids.add(vid)
        # Also check without namespace (pt2matsim output varies)
        for dep in sched_tree.iter('departure'):
            vid = dep.get('vehicleRefId')
            if vid:
                ref_ids.add(vid)

        if not ref_ids:
            return 0

        # Parse vehicles file
        veh_tree = ET.parse(vehicles_path)
        veh_root = veh_tree.getroot()

        existing_ids = set()
        for v in veh_root.iter(f'{ns}vehicle'):
            existing_ids.add(v.get('id'))

        existing_type_ids = {vt.get('id') for vt in veh_root.iter(f'{ns}vehicleType')}

        missing = ref_ids - existing_ids
        if not missing:
            return 0

        # Infer type from suffix: veh_5_bus → Bus, veh_3_rail → Rail
        # First, determine what types are needed
        missing_types = set()
        vehicles_to_add = []  # (vid, vtype)
        for vid in sorted(missing):
            suffix = vid.rsplit('_', 1)[-1]
            vtype = suffix.capitalize()  # bus → Bus, rail → Rail
            if vtype not in existing_type_ids:
                missing_types.add(vtype)
            vehicles_to_add.append((vid, vtype))

        # Create any missing vehicle types with sensible defaults
        type_defaults = {
            'Bus': {'seats': '70', 'standing': '0', 'length': '18.0', 'width': '2.5',
                    'pce': '2.8', 'mode': 'bus', 'access': '0.5', 'egress': '0.5'},
            'Rail': {'seats': '400', 'standing': '0', 'length': '200.0', 'width': '2.8',
                     'pce': '27.1', 'mode': 'rail', 'access': '0.25', 'egress': '0.25'},
            'Ferry': {'seats': '250', 'standing': '0', 'length': '50.0', 'width': '10.0',
                      'pce': '30.0', 'mode': 'ferry', 'access': '1.0', 'egress': '1.0'},
            'Tram': {'seats': '120', 'standing': '0', 'length': '36.0', 'width': '2.4',
                     'pce': '5.0', 'mode': 'tram', 'access': '0.25', 'egress': '0.25'},
        }
        fallback = {'seats': '50', 'standing': '0', 'length': '15.0', 'width': '2.5',
                     'pce': '2.0', 'mode': 'other', 'access': '0.5', 'egress': '0.5'}

        # Find insertion point: insert new types AFTER existing types but BEFORE vehicles
        last_type_idx = -1
        for idx, child in enumerate(veh_root):
            if child.tag == f'{ns}vehicleType':
                last_type_idx = idx

        # Add missing vehicle types at the correct position
        insert_idx = last_type_idx + 1
        for mtype in sorted(missing_types):
            defaults = type_defaults.get(mtype, fallback)
            vtype_elem = ET.Element(f'{ns}vehicleType', id=mtype)
            attrs = ET.SubElement(vtype_elem, f'{ns}attributes')
            for attr_name, attr_val in [
                ('accessTimeInSecondsPerPerson', defaults['access']),
                ('doorOperationMode', 'serial'),
                ('egressTimeInSecondsPerPerson', defaults['egress']),
            ]:
                a = ET.SubElement(attrs, f'{ns}attribute', name=attr_name)
                a.set('class', 'java.lang.Double' if attr_name != 'doorOperationMode'
                       else 'org.matsim.vehicles.VehicleType$DoorOperationMode')
                a.text = attr_val
            cap = ET.SubElement(vtype_elem, f'{ns}capacity',
                                seats=defaults['seats'], standingRoomInPersons=defaults['standing'])
            ET.SubElement(vtype_elem, f'{ns}length', meter=defaults['length'])
            ET.SubElement(vtype_elem, f'{ns}width', meter=defaults['width'])
            ET.SubElement(vtype_elem, f'{ns}costInformation')
            ET.SubElement(vtype_elem, f'{ns}passengerCarEquivalents', pce=defaults['pce'])
            ET.SubElement(vtype_elem, f'{ns}networkMode', networkMode=defaults['mode'])
            ET.SubElement(vtype_elem, f'{ns}flowEfficiencyFactor', factor='1.0')
            veh_root.insert(insert_idx, vtype_elem)
            insert_idx += 1
            logger.info(f"  Added missing vehicle type: {mtype}")

        # Now add missing vehicle instances (these will go at the end, which is correct)
        added = 0
        for vid, vtype in vehicles_to_add:
            ET.SubElement(veh_root, f'{ns}vehicle', id=vid, type=vtype)
            added += 1

        ET.indent(veh_tree, space='    ')
        veh_tree.write(str(vehicles_path), xml_declaration=True, encoding='UTF-8')

        logger.info(f"  Added {added} missing vehicle instances to {vehicles_path.name}"
                     f" ({len(missing_types)} new type(s))")
        return added

    def _validate_vehicle_xml_structure(self, vehicles_path: Path) -> bool:
        """Validate that transitVehicles.xml has correct structure.

        MATSim schema requires ALL vehicleType elements BEFORE ALL vehicle elements.
        Returns True if valid, raises RuntimeError if invalid.
        """
        ns = self._VEHICLES_NS
        tree = ET.parse(vehicles_path)
        root = tree.getroot()

        seen_vehicle = False
        for child in root:
            tag = child.tag
            if tag == f'{ns}vehicle':
                seen_vehicle = True
            elif tag == f'{ns}vehicleType':
                if seen_vehicle:
                    # Found a vehicleType after a vehicle - INVALID!
                    vtype_id = child.get('id')
                    raise RuntimeError(
                        f"Invalid transitVehicles.xml structure: vehicleType '{vtype_id}' "
                        f"appears after vehicle elements. MATSim requires all vehicleType "
                        f"elements before all vehicle elements."
                    )

        logger.debug(f"  Validated {vehicles_path.name}: structure is correct")
        return True

    def _write_pt_mapper_config_xml(
        self,
        input_network: Path,
        input_schedule: Path,
        output_network: Path,
        output_schedule: Path,
        config_path: Path,
    ) -> None:
        """Write XML config for PublicTransitMapper."""
        config = ET.Element('config')
        module = ET.SubElement(config, 'module', name='PublicTransitMapping')

        # Tunable PTMapper defaults — overridable via config.json matsim.pt2matsim
        pt_mapper_cfg = self.matsim_config.get('pt2matsim', {})

        params = {
            'inputNetworkFile': str(input_network.resolve()),
            'inputScheduleFile': str(input_schedule.resolve()),
            'outputNetworkFile': str(output_network.resolve()),
            'outputScheduleFile': str(output_schedule.resolve()),
            'outputStreetNetworkFile': '',
            'numOfThreads': str(pt_mapper_cfg.get('numOfThreads', 20)),
            'candidateDistanceMultiplier': str(pt_mapper_cfg.get('candidateDistanceMultiplier', 1.6)),
            'maxLinkCandidateDistance': str(pt_mapper_cfg.get('maxLinkCandidateDistance', 200.0)),
            'maxTravelCostFactor': str(pt_mapper_cfg.get('maxTravelCostFactor', 5.0)),
            'nLinkThreshold': str(pt_mapper_cfg.get('nLinkThreshold', 6)),
            'travelCostType': str(pt_mapper_cfg.get('travelCostType', 'linkLength')),
            'removeNotUsedStopFacilities': str(pt_mapper_cfg.get('removeNotUsedStopFacilities', True)).lower(),
            'routingWithCandidateDistance': str(pt_mapper_cfg.get('routingWithCandidateDistance', True)).lower(),
            'scheduleFreespeedModes': str(pt_mapper_cfg.get('scheduleFreespeedModes', 'artificial')),
        }

        # Build modesToKeepOnCleanUp dynamically from enabled transit modes
        enabled_transit = self._get_enabled_transit_modes()
        keep_modes = ['car'] + [m for m in enabled_transit]
        params['modesToKeepOnCleanUp'] = ','.join(keep_modes)

        for name, value in params.items():
            ET.SubElement(module, 'param', name=name, value=value)

        # Transport mode assignments: only for enabled transit modes
        mode_to_network = {
            'bus': 'car,bus',
            'rail': 'rail,light_rail',
        }
        for sched_mode in enabled_transit:
            net_modes = mode_to_network.get(sched_mode)
            if net_modes:
                ps = ET.SubElement(module, 'parameterset', type='transportModeAssignment')
                ET.SubElement(ps, 'param', name='scheduleMode', value=sched_mode)
                ET.SubElement(ps, 'param', name='networkModes', value=net_modes)

        tree = ET.ElementTree(config)
        ET.indent(tree, space='    ')
        self._write_matsim_config_xml(tree, config_path)

    def map_schedule_to_network(
        self, network_path: Path, schedule_path: Path, output_dir: Path
    ) -> Tuple[Path, Path]:
        """Step 3: Run PublicTransitMapper to map schedule to network.

        Returns:
            (mapped_network_path, mapped_schedule_path)
        """
        logger.info("Step 3: Mapping transit schedule to network via PublicTransitMapper")
        logger.info(f"  Network:  {network_path}")
        logger.info(f"  Schedule: {schedule_path}")

        jar_path = self.get_pt2matsim_jar_path()

        output_network = output_dir / 'network.xml'
        output_schedule = output_dir / 'transitSchedule.xml'

        config_xml = output_dir / 'pt2matsim_mapper_config.xml'
        self._write_pt_mapper_config_xml(
            network_path, schedule_path, output_network, output_schedule, config_xml
        )

        cmd = [
            'java', '-cp', str(jar_path),
            'org.matsim.pt2matsim.run.PublicTransitMapper',
            str(config_xml.resolve())
        ]
        logger.info(f"Command: {' '.join(cmd)}")

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0

        if result.returncode != 0:
            logger.error(f"PublicTransitMapper failed:\n{result.stderr}")
            raise RuntimeError(f"PublicTransitMapper failed: {result.stderr}")

        if result.stdout:
            logger.debug(result.stdout[-2000:])

        num_nodes, num_links = self._count_network_elements(output_network)
        logger.info(
            f"Step 3 complete: mapped network has {num_nodes} nodes, "
            f"{num_links} links (took {elapsed:.1f}s)"
        )

        # Cleanup temp config
        config_xml.unlink(missing_ok=True)

        return output_network, output_schedule

    def _fix_pt_stop_connectivity(self, network_path: Path) -> int:
        """Add reverse links for transit stop nodes that only have inbound access.

        pt2matsim sometimes creates one-directional links from road nodes to
        transit stop nodes but omits the return link.  MATSim's
        UmlaufInterpolator then fails because it cannot route the transit
        vehicle back from the stop to the road network.

        For every ``pt_*`` node that has at least one inbound link from a
        non-``pt_*`` node but no outbound link to a non-``pt_*`` node, this
        method creates a reverse link mirroring the first inbound connector.

        Returns the number of links added.
        """
        tree = ET.parse(network_path)
        root = tree.getroot()
        links_elem = root.find('links')
        if links_elem is None:
            return 0

        all_links = links_elem.findall('link')

        # Collect inbound / outbound connectors per pt stop (exclude self-loops)
        inbound: Dict[str, ET.Element] = {}   # pt_node → first inbound link element
        outbound_exists: set = set()           # pt_nodes that have an outbound connector

        for link in all_links:
            from_id = link.get('from', '')
            to_id = link.get('to', '')
            if from_id == to_id:
                continue  # skip self-loops

            if to_id.startswith('pt_') and not from_id.startswith('pt_'):
                if to_id not in inbound:
                    inbound[to_id] = link
            if from_id.startswith('pt_') and not to_id.startswith('pt_'):
                outbound_exists.add(from_id)

        added = 0
        for pt_node, in_link in inbound.items():
            if pt_node in outbound_exists:
                continue  # already has a way out

            # Create a reverse link: pt_node → road_node
            road_node = in_link.get('from')
            in_id = in_link.get('id')
            rev_id = f"{in_id}_rev"

            rev = ET.SubElement(links_elem, 'link')
            rev.set('id', rev_id)
            rev.set('from', pt_node)
            rev.set('to', road_node)
            rev.set('length', in_link.get('length', '1.0'))
            rev.set('freespeed', in_link.get('freespeed', '1.0'))
            rev.set('capacity', in_link.get('capacity', '9999.0'))
            rev.set('permlanes', in_link.get('permlanes', '1.0'))
            rev.set('oneway', '1')
            rev.set('modes', in_link.get('modes', 'bus,artificial'))
            added += 1
            logger.info(
                f"  Added reverse link {rev_id}: {pt_node} → {road_node}"
            )

        if added:
            ET.indent(tree, space='\t')
            self._write_xml_with_doctype(tree, network_path, self._DOCTYPE_NETWORK)
            logger.info(f"Fixed {added} dead-end transit stop(s) in {network_path.name}")
        else:
            logger.info("All transit stops have bidirectional connectivity — no fix needed")

        return added

    def _validate_transit_routes(self, schedule_path: Path,
                                 network_path: Path) -> int:
        """Remove transit routes that would crash MATSim at runtime.

        Checks two conditions for every ``<transitRoute>``:

        1. **Stop-link presence** – each stop's ``linkRefId`` must appear in
           the route's ``<link refId="..."/>`` sequence.
        2. *(Disabled)* – Stop-link order check was removed because pt2matsim
           often maps two nearby stops to the same link; the cursor-based
           scan incorrectly flagged these as mismatches.
        3. **Link connectivity** – every pair of consecutive links in the route
           must be connected in the network (the ``to`` node of link *i* must
           equal the ``from`` node of link *i+1*).

        After route validation, two additional cleanup steps run:

        4. **Unused stop removal** – ``stopFacility`` elements not referenced
           by any surviving route's ``routeProfile`` are removed.  The
           SwissRailRaptor router builds a spatial index over *all* defined
           stops but only populates transfer data for stops used in routes;
           unused stops cause a ``NullPointerException`` in
           ``SwissRailRaptorData.calculateRouteStopTransfers()``.
        5. **Stop-link network check** – any remaining ``stopFacility`` whose
           ``linkRefId`` does not exist in the network is removed (and routes
           referencing it are also removed).
        6. **Transfer cleanup** – ``<minimalTransferTimes>`` ``<relation>``
           entries that reference removed stops are pruned.  The Raptor
           router iterates these, looks up the facility, and NPEs on
           ``fromStop.getCoord()`` if the facility no longer exists.

        Additionally, *before* the per-route checks, a graph-reachability
        analysis runs:

        7. **Terminal-stop road reachability** – the first stop must be
           *reachable from* the road network and the last stop must be
           able to *reach* the road network, both via directed links.
           MATSim's ``UmlaufInterpolator`` routes transit vehicles
           between the last stop of one departure and the first stop of
           the next ("Wende"); if a terminal stop sits on a one-way
           ``pt_*``-only chain with no directed path back to the road
           network, Dijkstra fails at runtime with "No route found".

        Routes that fail any check are removed.  Empty ``<transitLine>``
        elements are also pruned.

        Returns the number of routes removed.
        """
        # --- Build network adjacency: link_id → (from_node, to_node) --------
        net_tree = ET.parse(network_path)
        net_root = net_tree.getroot()
        link_endpoints: Dict[str, Tuple[str, str]] = {}
        # Build directed adjacency graphs for transit-mode reachability.
        # MATSim's Dijkstra follows links directionally, so undirected BFS
        # gives false positives (a pt chain may be reachable inbound-only).
        fwd_adj: Dict[str, set] = defaultdict(set)   # node → outgoing neighbours
        rev_adj: Dict[str, set] = defaultdict(set)   # node → incoming neighbours
        net_links = net_root.find('links')
        if net_links is not None:
            for lnk in net_links.findall('link'):
                lid = lnk.get('id')
                if lid:
                    from_n = lnk.get('from', '')
                    to_n = lnk.get('to', '')
                    link_endpoints[lid] = (from_n, to_n)
                    modes = lnk.get('modes', '')
                    if 'bus' in modes or 'artificial' in modes:
                        if from_n != to_n:  # skip self-loops
                            fwd_adj[from_n].add(to_n)
                            rev_adj[to_n].add(from_n)

        # Nodes that CAN REACH a road node (following outgoing links).
        # Needed for last-stop validation: the vehicle must route out.
        # We do a reverse BFS from all non-pt nodes over the forward graph:
        # "which nodes have a directed path leading TO a road node?"
        can_reach_road: set = set()
        seeds = [n for n in (set(fwd_adj.keys()) | set(rev_adj.keys()))
                 if not n.startswith('pt_')]
        queue = deque(seeds)
        can_reach_road.update(seeds)
        while queue:
            node = queue.popleft()
            for pred in rev_adj.get(node, ()):
                if pred not in can_reach_road:
                    can_reach_road.add(pred)
                    queue.append(pred)

        # Nodes REACHABLE FROM a road node (following outgoing links).
        # Needed for first-stop validation: the vehicle must route in.
        reachable_from_road: set = set()
        queue = deque(seeds)
        reachable_from_road.update(seeds)
        while queue:
            node = queue.popleft()
            for succ in fwd_adj.get(node, ()):
                if succ not in reachable_from_road:
                    reachable_from_road.add(succ)
                    queue.append(succ)

        # --- Parse schedule --------------------------------------------------
        tree = ET.parse(schedule_path)
        root = tree.getroot()

        # Build stop-facility → linkRefId map
        stop_link: Dict[str, str] = {}
        for sf in root.iter('stopFacility'):
            sid = sf.get('id')
            link_ref = sf.get('linkRefId')
            if sid and link_ref:
                stop_link[sid] = link_ref

        removed = 0
        empty_lines = []

        for line_elem in root.findall('transitLine'):
            bad_routes = []

            for route_elem in line_elem.findall('transitRoute'):
                route_id = route_elem.get('id', '?')
                line_id = line_elem.get('id', '?')
                broken = False

                # --- Collect ordered link IDs in route -----------------------
                route_link_ids: List[str] = []
                route_tag = route_elem.find('route')
                if route_tag is not None:
                    for lnk in route_tag.findall('link'):
                        ref = lnk.get('refId')
                        if ref:
                            route_link_ids.append(ref)
                route_link_set = set(route_link_ids)

                # --- Check 1: every stop link is in the route ----------------
                profile = route_elem.find('routeProfile')
                if profile is not None:
                    for stop in profile.findall('stop'):
                        stop_ref = stop.get('refId')
                        if not stop_ref:
                            continue
                        link_ref = stop_link.get(stop_ref)
                        if link_ref and link_ref not in route_link_set:
                            logger.warning(
                                f"  Route {line_id}/{route_id}: stop "
                                f"{stop_ref} on link {link_ref} not in "
                                f"route link sequence — removing"
                            )
                            broken = True
                            break

                # --- Check 2: DISABLED ----------------------------------------
                #     pt2matsim often maps two nearby stops to the same
                #     network link.  The cursor-based scan below treats that
                #     as a mismatch ("already passed"), but MATSim handles
                #     it fine at runtime — the vehicle visits the link once
                #     and serves both stops.  Check 3 (link connectivity)
                #     already catches genuinely broken route sequences.
                #     In Washington DC this check removed 1,310 valid routes
                #     (95% of all removals) across 126 transit lines.
                # if not broken and profile is not None:
                #     cursor = 0  # current position in route_link_ids
                #     for stop in profile.findall('stop'):
                #         stop_ref = stop.get('refId')
                #         if not stop_ref:
                #             continue
                #         link_ref = stop_link.get(stop_ref)
                #         if not link_ref:
                #             continue
                #         # Find link_ref at position >= cursor
                #         found = False
                #         for j in range(cursor, len(route_link_ids)):
                #             if route_link_ids[j] == link_ref:
                #                 cursor = j + 1
                #                 found = True
                #                 break
                #         if not found:
                #             logger.warning(
                #                 f"  Route {line_id}/{route_id}: stop "
                #                 f"{stop_ref} on link {link_ref} — link "
                #                 f"order mismatch (already passed in route "
                #                 f"sequence) — removing"
                #             )
                #             broken = True
                #             break

                # --- Check 3: consecutive links are connected ----------------
                if not broken and len(route_link_ids) >= 2:
                    for i in range(len(route_link_ids) - 1):
                        lid_a = route_link_ids[i]
                        lid_b = route_link_ids[i + 1]
                        ep_a = link_endpoints.get(lid_a)
                        ep_b = link_endpoints.get(lid_b)
                        if ep_a is None or ep_b is None:
                            logger.warning(
                                f"  Route {line_id}/{route_id}: link "
                                f"{lid_a if ep_a is None else lid_b} not "
                                f"found in network — removing"
                            )
                            broken = True
                            break
                        if ep_a[1] != ep_b[0]:
                            logger.warning(
                                f"  Route {line_id}/{route_id}: gap between "
                                f"link {lid_a} (to={ep_a[1]}) and link "
                                f"{lid_b} (from={ep_b[0]}) — removing"
                            )
                            broken = True
                            break

                # --- Check 7: terminal stops must be road-reachable --------
                #     MATSim's UmlaufInterpolator routes transit vehicles
                #     between the last stop of one departure and the first
                #     stop of the next ("Wende").  If a terminal stop sits on
                #     a pt-only subnetwork disconnected from the road network,
                #     Dijkstra fails with "No route found".
                #     - First stop: road must be able to route TO it
                #     - Last stop: it must be able to route TO road
                if not broken and profile is not None:
                    stops = profile.findall('stop')
                    if stops:
                        checks = [('first', stops[0], reachable_from_road),
                                  ('last', stops[-1], can_reach_road)]
                        for pos_label, stop_el, reachable_set in checks:
                            sref = stop_el.get('refId')
                            if not sref:
                                continue
                            lref = stop_link.get(sref)
                            if not lref:
                                continue
                            ep = link_endpoints.get(lref)
                            if ep is None:
                                continue
                            # Check both endpoints of the stop's link
                            if (ep[0] not in reachable_set
                                    and ep[1] not in reachable_set):
                                logger.warning(
                                    f"  Route {line_id}/{route_id}: {pos_label} "
                                    f"stop {sref} on link {lref} is on an "
                                    f"isolated pt subnetwork (no road "
                                    f"connectivity) — removing"
                                )
                                broken = True
                                break

                if broken:
                    bad_routes.append(route_elem)

            for br in bad_routes:
                line_elem.remove(br)
                removed += 1

            # If no routes remain, mark the line for removal
            if len(line_elem.findall('transitRoute')) == 0:
                empty_lines.append(line_elem)

        for el in empty_lines:
            root.remove(el)
            logger.info(f"  Removed empty transit line {el.get('id')}")

        # --- Check 4: remove unused stopFacilities ----------------------------
        #     SwissRailRaptor builds a QuadTree over ALL stopFacilities but
        #     only populates routeStopsPerStopFacility for stops used in
        #     routes.  Nearby-stop lookups can return unused stops whose
        #     transfer array is null → NPE in calculateRouteStopTransfers().
        used_stop_ids: set = set()
        for line_elem in root.findall('transitLine'):
            for route_elem in line_elem.findall('transitRoute'):
                profile = route_elem.find('routeProfile')
                if profile is not None:
                    for stop in profile.findall('stop'):
                        ref = stop.get('refId')
                        if ref:
                            used_stop_ids.add(ref)

        stops_container = root.find('transitStops')
        unused_removed = 0
        if stops_container is not None:
            unused_stops = [
                sf for sf in stops_container.findall('stopFacility')
                if sf.get('id') not in used_stop_ids
            ]
            for sf in unused_stops:
                stops_container.remove(sf)
                unused_removed += 1
            if unused_removed:
                logger.info(
                    f"  Removed {unused_removed} unused stopFacilities "
                    f"(not referenced by any surviving route)"
                )

        # --- Check 5: stop linkRefId must exist in network --------------------
        #     After pruning unused stops, verify remaining stops reference
        #     links that actually exist in the network.
        network_link_ids = set(link_endpoints.keys())
        bad_stop_ids: set = set()
        if stops_container is not None:
            bad_facilities = [
                sf for sf in stops_container.findall('stopFacility')
                if sf.get('linkRefId') and sf.get('linkRefId') not in network_link_ids
            ]
            for sf in bad_facilities:
                bad_stop_ids.add(sf.get('id'))
                stops_container.remove(sf)
            if bad_stop_ids:
                logger.warning(
                    f"  Removed {len(bad_stop_ids)} stopFacilities whose "
                    f"linkRefId is missing from the network"
                )

        # If stops were removed for bad linkRefIds, re-check routes
        if bad_stop_ids:
            extra_removed = 0
            extra_empty = []
            for line_elem in root.findall('transitLine'):
                bad_routes_2 = []
                for route_elem in line_elem.findall('transitRoute'):
                    profile = route_elem.find('routeProfile')
                    if profile is not None:
                        for stop in profile.findall('stop'):
                            if stop.get('refId') in bad_stop_ids:
                                bad_routes_2.append(route_elem)
                                break
                for br in bad_routes_2:
                    line_elem.remove(br)
                    extra_removed += 1
                if len(line_elem.findall('transitRoute')) == 0:
                    extra_empty.append(line_elem)
            for el in extra_empty:
                root.remove(el)
            if extra_removed:
                logger.warning(
                    f"  Removed {extra_removed} additional route(s) "
                    f"referencing stops with missing network links"
                )
            removed += extra_removed

        # --- Check 6: clean minimalTransferTimes ---------------------------------
        #     pt2matsim generates <minimalTransferTimes> with <relation> entries
        #     referencing stops that may no longer exist after checks 4/5.
        #     SwissRailRaptor iterates these, looks up the stop facility, gets
        #     null, then NPEs on fromStop.getCoord().
        #     Build the set of surviving stop IDs and prune stale relations.
        surviving_stop_ids: set = set()
        if stops_container is not None:
            for sf in stops_container.findall('stopFacility'):
                sid = sf.get('id')
                if sid:
                    surviving_stop_ids.add(sid)

        transfer_section = root.find('minimalTransferTimes')
        transfers_removed = 0
        if transfer_section is not None:
            stale_relations = [
                rel for rel in transfer_section.findall('relation')
                if rel.get('fromStop') not in surviving_stop_ids
                or rel.get('toStop') not in surviving_stop_ids
            ]
            for rel in stale_relations:
                transfer_section.remove(rel)
                transfers_removed += 1
            # Remove the section entirely if empty
            if len(transfer_section.findall('relation')) == 0:
                root.remove(transfer_section)
            if transfers_removed:
                logger.info(
                    f"  Removed {transfers_removed} stale minimalTransferTimes "
                    f"relations referencing removed stops"
                )

        modified = (removed > 0 or unused_removed > 0
                    or len(bad_stop_ids) > 0 or transfers_removed > 0)
        if modified:
            ET.indent(tree, space='    ')
            self._write_xml_with_doctype(tree, schedule_path, self._DOCTYPE_SCHEDULE)
            logger.info(
                f"Transit schedule validated: {removed} broken route(s) removed, "
                f"{unused_removed} unused stops pruned, "
                f"{len(bad_stop_ids)} stops with missing links removed, "
                f"{transfers_removed} stale transfer relations removed"
            )
        else:
            logger.info("All transit routes pass validation")

        return removed

    def _count_network_elements(self, network_path: Path) -> Tuple[int, int]:
        """Parse a MATSim network.xml and return (num_nodes, num_links)."""
        try:
            tree = ET.parse(network_path)
            root = tree.getroot()
            nodes_elem = root.find('nodes')
            links_elem = root.find('links')
            num_nodes = len(nodes_elem.findall('node')) if nodes_elem is not None else 0
            num_links = len(links_elem.findall('link')) if links_elem is not None else 0
            return num_nodes, num_links
        except Exception as e:
            logger.warning(f"Could not parse network statistics from {network_path}: {e}")
            return 0, 0

    def generate_network(
        self,
        counties: Optional[List[Tuple[str, str]]] = None,
        output_path: Optional[Path] = None,
        coordinate_system: str = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        clean_network: bool = True,
        db_manager=None,
    ) -> Dict:
        """
        Main method to generate MATSim network from counties or bounding box.

        When ``matsim.transit_network`` is false (default):
            Downloads OSM data and converts using MATSim's SupersonicOsmNetworkReader,
            then optionally cleans with NetworkCleaner.

        When ``matsim.transit_network`` is true:
            Runs the pt2matsim 3-step pipeline (Osm2MultimodalNetwork →
            Gtfs2TransitSchedule → PublicTransitMapper) to produce a combined
            road+transit network.xml plus transitSchedule.xml and transitVehicles.xml.

        Args:
            counties: List of (county, state) tuples (optional if bbox is provided)
            output_path: Path to save network.xml
            coordinate_system: Target EPSG code (e.g., 'EPSG:26915')
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat) (optional if counties provided)
            clean_network: Whether to run NetworkCleaner after generation (default: True)
            db_manager: DBManager instance (required when transit_network is true)

        Returns:
            Dictionary with network metadata

        Raises:
            ValueError: If neither counties nor bbox is provided, or if output_path is None
        """
        if counties is None and bbox is None:
            raise ValueError("Either 'counties' or 'bbox' must be provided")

        if output_path is None:
            raise ValueError("output_path must be provided")

        # Ensure output path is a Path object
        output_path = Path(output_path)

        # Ensure the final output file is named 'network.xml'
        if output_path.name != 'network.xml':
            logger.info(f"Output filename '{output_path.name}' will be renamed to 'network.xml'")
            temp_output = output_path
            final_output = output_path.parent / 'network.xml'
        else:
            temp_output = output_path
            final_output = output_path

        # Create temp directory for OSM data
        temp_dir = output_path.parent / 'temp_osm'
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Check if OSM PBF data already exists (to avoid re-downloading)
        osm_pbf_file = temp_dir / 'osm_data.osm.pbf'

        if osm_pbf_file.exists():
            logger.info(f"Using existing OSM PBF data: {osm_pbf_file}")
            osm_file = osm_pbf_file
        else:
            if counties is not None:
                logger.info(f"Downloading OSM data for {len(counties)} counties...")
                osm_file = self.downloader.download_for_counties(
                    counties=counties,
                    output_path=osm_pbf_file,
                    method='auto'
                )
            else:
                logger.info(f"Downloading OSM data for bounding box...")
                osm_file = self.downloader.download_for_bbox(
                    bbox=bbox,
                    output_path=osm_pbf_file,
                    method='auto'
                )

        # ── Branch: pt2matsim transit path vs original road-only path ──
        use_transit = self.matsim_config.get('transit_network', False)
        enabled_transit = self._get_enabled_transit_modes()

        if use_transit and enabled_transit:
            return self._generate_transit_network(
                osm_file, final_output, coordinate_system, db_manager,
                counties=counties,
            )
        else:
            if use_transit and not enabled_transit:
                logger.warning("transit_network is true but no transit modes are enabled. "
                             "Falling back to road-only network.")

            return self._generate_road_network(
                osm_file, temp_output, final_output, coordinate_system, clean_network
            )

    def _generate_road_network(
        self, osm_file: Path, temp_output: Path, final_output: Path,
        coordinate_system: str, clean_network: bool
    ) -> Dict:
        """Original road-only network generation path."""
        if clean_network:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tmp:
                raw_network = Path(tmp.name)

            metadata = self.convert_osm_to_network(
                osm_file=osm_file,
                output_path=raw_network,
                coordinate_system=coordinate_system
            )

            logger.info(f"Generated raw network (temporary)")

            logger.info("Cleaning network with NetworkCleaner...")
            clean_metadata = self.clean_network(
                input_network=raw_network,
                output_network=final_output
            )

            raw_network.unlink()
            logger.info(f"Deleted temporary raw network")

            metadata['num_nodes_before_cleaning'] = metadata['num_nodes']
            metadata['num_links_before_cleaning'] = metadata['num_links']
            metadata['num_nodes'] = clean_metadata['num_nodes']
            metadata['num_links'] = clean_metadata['num_links']
            metadata['network_cleaned'] = True

            logger.info(f"Saved cleaned network to: {final_output}")
        else:
            metadata = self.convert_osm_to_network(
                osm_file=osm_file,
                output_path=temp_output,
                coordinate_system=coordinate_system
            )
            metadata['network_cleaned'] = False

            if temp_output != final_output:
                import shutil
                shutil.move(str(temp_output), str(final_output))
                logger.info(f"Renamed {temp_output.name} to network.xml")

        metadata['transit_network'] = False
        return metadata

    def _generate_transit_network(
        self, osm_file: Path, final_output: Path, coordinate_system: str,
        db_manager, counties: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict:
        """pt2matsim 3-step pipeline: multimodal network + transit schedule."""
        if db_manager is None:
            raise ValueError(
                "db_manager is required when matsim.transit_network is true"
            )

        output_dir = final_output.parent
        transit_dir = output_dir / 'transit_build'
        transit_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Generating transit network via pt2matsim pipeline")
        logger.info("=" * 60)

        # Compute region bbox for spatial filtering of GTFS feeds
        region_bbox = None
        if counties:
            logger.info(f"Computing region bbox from {len(counties)} counties for GTFS filtering")
            region_bbox = self.downloader.get_bbox_for_counties(counties)
            logger.info(f"Region bbox: ({region_bbox[0]:.4f}, {region_bbox[1]:.4f}, "
                         f"{region_bbox[2]:.4f}, {region_bbox[3]:.4f})")
        else:
            logger.warning("No counties provided — GTFS feeds will NOT be spatially filtered")

        t_total = time.time()

        # Step 1: OSM → multimodal network
        multimodal_network = transit_dir / 'multimodal_network.xml'
        step1_meta = self.convert_osm_to_multimodal_network(
            osm_file, multimodal_network, coordinate_system
        )

        # Step 2: GTFS feeds → unmapped schedule + vehicles
        try:
            schedule_path, vehicles_path = self.convert_gtfs_feeds_to_schedule(
                transit_dir, coordinate_system, db_manager, bbox=region_bbox
            )
        except RuntimeError as e:
            if "No GTFS feeds were successfully converted" in str(e):
                logger.warning("=" * 60)
                logger.warning("NO GTFS FEEDS WITH STOPS IN REGION")
                logger.warning("=" * 60)
                logger.warning(
                    "Transit network was requested but no GTFS feeds have stops "
                    "within the simulation region. This typically means the region's "
                    "transit agency is missing from or deprecated in the Mobility "
                    "Database catalog."
                )
                logger.warning("Falling back to road-only network.")
                logger.warning("=" * 60)
                # Fall back: use the multimodal network as road-only
                import shutil
                shutil.copy2(str(multimodal_network), str(final_output))
                metadata = step1_meta.copy()
                metadata['transit_network'] = False
                metadata['transit_fallback'] = True
                metadata['transit_fallback_reason'] = str(e)
                return metadata
            raise

        # Step 3: Map schedule to network → final network.xml + transitSchedule.xml
        mapped_network, mapped_schedule = self.map_schedule_to_network(
            multimodal_network, schedule_path, output_dir
        )

        # Fix dead-end transit stops (pt2matsim may create one-way access links)
        self._fix_pt_stop_connectivity(mapped_network)

        # pt2matsim maps stops to network links but does NOT validate that
        # each route's link sequence is topologically traversable. Without
        # this step, ~7k broken routes remain and MATSim crashes at runtime
        # with "No route found" errors (see experiment_20260215_203044).
        self._validate_transit_routes(mapped_schedule, mapped_network)

        # Copy vehicles to output dir (Step 3 doesn't modify it)
        final_vehicles = output_dir / 'transitVehicles.xml'
        if vehicles_path != final_vehicles:
            import shutil
            shutil.copy2(vehicles_path, final_vehicles)

        # Ensure all vehicle refs in schedule have matching vehicle instances
        self._ensure_vehicle_instances(final_vehicles, mapped_schedule)

        # Validate the final vehicles file structure
        self._validate_vehicle_xml_structure(final_vehicles)

        elapsed_total = time.time() - t_total
        num_nodes, num_links = self._count_network_elements(mapped_network)

        logger.info("=" * 60)
        logger.info(
            f"Transit network pipeline complete: {num_nodes} nodes, "
            f"{num_links} links (total {elapsed_total:.1f}s)"
        )
        logger.info(f"  Network:    {mapped_network}")
        logger.info(f"  Schedule:   {mapped_schedule}")
        logger.info(f"  Vehicles:   {final_vehicles}")
        logger.info("=" * 60)

        return {
            'num_nodes': num_nodes,
            'num_links': num_links,
            'num_nodes_multimodal': step1_meta['num_nodes'],
            'num_links_multimodal': step1_meta['num_links'],
            'coordinate_system': coordinate_system,
            'transit_network': True,
            'network_cleaned': False,
        }
