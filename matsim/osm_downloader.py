"""
OSM data downloader for MATSim network generation
Downloads OSM PBF files for specified regions with osmium-based extraction
"""

import requests
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import logging
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import osmnx as ox
import subprocess
import json
import time
import shutil

logger = logging.getLogger(__name__)


class OSMDownloader:
    """Download OSM data for network generation with osmium extraction support"""

    # Geofabrik download server
    GEOFABRIK_BASE = "https://download.geofabrik.de"

    # Cache directory for state-level PBF files
    CACHE_DIR = Path(__file__).parent.parent / "data" / "osm_cache"

    def __init__(self):
        # Create cache directory if it doesn't exist
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"OSM cache directory: {self.CACHE_DIR}")

    def check_osmium_installed(self) -> bool:
        """
        Check if osmium-tool is installed

        Returns:
            True if osmium is available, False otherwise
        """
        try:
            result = subprocess.run(['osmium', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip().split('\n')[0]
                logger.info(f"Found {version}")
                return True
        except FileNotFoundError:
            pass

        logger.warning("osmium-tool not found. Install with: conda install -c conda-forge osmium-tool")
        logger.warning("Or add C:\\Users\\Jalal\\miniconda3\\Library\\bin to your Windows PATH")
        return False

    def check_osmconvert_installed(self) -> bool:
        """
        Check if osmconvert is installed

        Returns:
            True if osmconvert is available, False otherwise
        """
        try:
            result = subprocess.run(['osmconvert', '--help'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Found osmconvert")
                return True
        except FileNotFoundError:
            pass

        logger.debug("osmconvert not found")
        return False

    def get_bbox_for_counties(
        self,
        counties: List[Tuple[str, str]]
    ) -> Tuple[float, float, float, float]:
        """
        Get bounding box for list of counties

        Args:
            counties: List of (county, state) tuples

        Returns:
            Tuple of (min_lon, min_lat, max_lon, max_lat)
        """
        logger.info(f"Calculating bounding box for {len(counties)} counties")

        all_bounds = []
        for county, state in counties:
            try:
                # Get county boundary from OSM
                query = f"{county}, {state}, USA"
                gdf = ox.geocode_to_gdf(query)
                bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
                all_bounds.append(bounds)
                logger.debug(f"  {county}, {state}: {bounds}")
            except Exception as e:
                logger.warning(f"Could not get bounds for {county}, {state}: {e}")

        if not all_bounds:
            raise ValueError("Could not get bounds for any counties")

        # Combine all bounds
        min_lon = min(b[0] for b in all_bounds)
        min_lat = min(b[1] for b in all_bounds)
        max_lon = max(b[2] for b in all_bounds)
        max_lat = max(b[3] for b in all_bounds)

        logger.info(f"Bounding box: ({min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f})")

        return (min_lon, min_lat, max_lon, max_lat)

    def download_osm_pbf_from_geofabrik(
        self,
        region: str,
        output_path: Optional[Path] = None,
        use_cache: bool = True
    ) -> Path:
        """
        Download OSM PBF file from Geofabrik with caching support

        Args:
            region: Region name (e.g., 'us/minnesota')
            output_path: Where to save the PBF file (if None, uses cache directory)
            use_cache: If True, check cache first and save to cache

        Returns:
            Path to downloaded file
        """
        # Determine cache file path
        region_name = region.split('/')[-1]  # e.g., 'minnesota' from 'us/minnesota'
        cache_file = self.CACHE_DIR / f"{region_name}-latest.osm.pbf"

        # If using cache and file exists, return cached version
        if use_cache and cache_file.exists():
            file_size_mb = cache_file.stat().st_size / 1024 / 1024
            logger.info(f"Using cached OSM data: {cache_file} ({file_size_mb:.1f} MB)")

            # If output_path is specified and different from cache, copy it
            if output_path and output_path != cache_file:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cache_file, output_path)
                return output_path

            return cache_file

        # Determine where to download
        download_path = cache_file if use_cache else output_path
        if download_path is None:
            download_path = cache_file

        url = f"{self.GEOFABRIK_BASE}/{region}-latest.osm.pbf"

        logger.info(f"Downloading OSM data from Geofabrik: {url}")
        download_path.parent.mkdir(parents=True, exist_ok=True)

        # Time the download
        start_time = time.time()

        # Stream download for large files
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        last_progress_log = 0

        with open(download_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192 * 128):  # 1MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    # Log every 10% to avoid spam
                    if progress - last_progress_log >= 10:
                        logger.info(f"Download progress: {progress:.1f}%")
                        last_progress_log = progress

        download_time = time.time() - start_time
        file_size_mb = download_path.stat().st_size / 1024 / 1024
        download_speed_mbps = (file_size_mb / download_time) if download_time > 0 else 0

        logger.info(f"Downloaded {file_size_mb:.1f} MB in {download_time:.1f}s ({download_speed_mbps:.2f} MB/s)")
        logger.info(f"Saved to: {download_path}")

        # If output_path is specified and different from download_path, copy it
        if output_path and output_path != download_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(download_path, output_path)
            return output_path

        return download_path

    def convert_osm_to_pbf(self, osm_file: Path, pbf_file: Path) -> Path:
        """
        Convert OSM XML file to PBF format using osmconvert

        Args:
            osm_file: Path to input OSM XML file
            pbf_file: Path to output PBF file

        Returns:
            Path to created PBF file
        """
        if not self.check_osmconvert_installed():
            logger.warning("osmconvert not found, keeping OSM XML format")
            return osm_file

        logger.info(f"Converting OSM XML to PBF format...")
        logger.info(f"  Input: {osm_file} ({osm_file.stat().st_size / 1024 / 1024:.1f} MB)")

        pbf_file.parent.mkdir(parents=True, exist_ok=True)

        # Run osmconvert
        cmd = ['osmconvert', str(osm_file), '--out-pbf', f'-o={str(pbf_file)}']

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"osmconvert failed: {result.stderr}")
            logger.warning("Keeping OSM XML format instead")
            return osm_file

        conversion_time = time.time() - start_time
        pbf_size_mb = pbf_file.stat().st_size / 1024 / 1024
        osm_size_mb = osm_file.stat().st_size / 1024 / 1024
        reduction_pct = ((osm_size_mb - pbf_size_mb) / osm_size_mb) * 100

        logger.info(f"Conversion complete in {conversion_time:.1f}s")
        logger.info(f"  Output: {pbf_file} ({pbf_size_mb:.1f} MB)")
        logger.info(f"  Size reduction: {reduction_pct:.1f}% ({osm_size_mb:.1f} MB -> {pbf_size_mb:.1f} MB)")

        # Delete original OSM file to save space
        osm_file.unlink()
        logger.info(f"Deleted original OSM file: {osm_file}")

        return pbf_file

    def download_osm_extract_for_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        output_path: Path,
        convert_to_pbf: bool = True
    ) -> Path:
        """
        Download OSM extract for a specific bounding box using Overpass API

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            output_path: Where to save the OSM file
            convert_to_pbf: If True and osmconvert is available, convert XML to PBF

        Returns:
            Path to downloaded file (may be PBF if converted)
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        # Use simpler map endpoint instead of interpreter for better compatibility
        # Format: https://overpass-api.de/api/map?bbox=west,south,east,north
        overpass_map_url = f"https://overpass-api.de/api/map?bbox={min_lon},{min_lat},{max_lon},{max_lat}"

        logger.info(f"Downloading OSM data for bbox via Overpass API")
        logger.info(f"  Bbox: ({min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f})")
        logger.info(f"  URL: {overpass_map_url}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure output has .osm extension for XML download
        if output_path.suffix != '.osm':
            temp_osm_path = output_path.with_suffix('.osm')
        else:
            temp_osm_path = output_path

        # Try the simple map API first (faster, simpler)
        try:
            logger.info(f"Downloading via map API...")
            start_time = time.time()

            response = requests.get(overpass_map_url, timeout=900)
            response.raise_for_status()

            with open(temp_osm_path, 'wb') as f:
                f.write(response.content)

            download_time = time.time() - start_time
            file_size_mb = temp_osm_path.stat().st_size / 1024 / 1024

            logger.info(f"Downloaded {file_size_mb:.1f} MB in {download_time:.1f}s")
            logger.info(f"Saved to: {temp_osm_path}")

            # Convert to PBF if requested and osmconvert is available
            if convert_to_pbf and output_path.suffix == '.pbf':
                pbf_path = output_path
                return self.convert_osm_to_pbf(temp_osm_path, pbf_path)

            return temp_osm_path

        except requests.exceptions.RequestException as e:
            logger.warning(f"Map API failed: {e}")
            logger.info("Falling back to interpreter API...")

        # Fallback to interpreter API with custom query
        overpass_urls = [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.openstreetmap.ru/api/interpreter"
        ]

        # Query for highway data in bbox
        query = f"""
        [out:xml][timeout:600][maxsize:2000000000];
        (
          way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
          node(w);
        );
        out body;
        """

        # Try each Overpass server
        last_error = None
        for overpass_url in overpass_urls:
            try:
                logger.info(f"Trying: {overpass_url}")
                response = requests.post(
                    overpass_url,
                    data={'data': query},
                    timeout=900  # 15 minute timeout
                )
                response.raise_for_status()

                with open(temp_osm_path, 'wb') as f:
                    f.write(response.content)

                logger.info(f"Downloaded OSM data to: {temp_osm_path}")

                # Convert to PBF if requested and osmconvert is available
                if convert_to_pbf and output_path.suffix == '.pbf':
                    pbf_path = output_path
                    return self.convert_osm_to_pbf(temp_osm_path, pbf_path)

                return temp_osm_path

            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed with {overpass_url}: {e}")
                last_error = e
                continue

        # If all servers failed, raise the last error
        raise RuntimeError(f"All Overpass servers failed. Last error: {last_error}")

    def extract_with_osmium_bbox(
        self,
        input_pbf: Path,
        output_pbf: Path,
        bbox: Tuple[float, float, float, float]
    ) -> Dict:
        """
        Extract area from PBF using osmium with bounding box

        Args:
            input_pbf: Path to source PBF file
            output_pbf: Path to output PBF file
            bbox: (min_lon, min_lat, max_lon, max_lat)

        Returns:
            Dictionary with extraction metadata
        """
        if not self.check_osmium_installed():
            raise RuntimeError("osmium-tool is required. Install with: conda install -c conda-forge osmium-tool")

        if not input_pbf.exists():
            raise FileNotFoundError(f"Input PBF not found: {input_pbf}")

        min_lon, min_lat, max_lon, max_lat = bbox

        logger.info(f"Extracting area with osmium (bbox)...")
        logger.info(f"  Input: {input_pbf} ({input_pbf.stat().st_size / 1024 / 1024:.1f} MB)")
        logger.info(f"  Bbox: ({min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f})")

        output_pbf.parent.mkdir(parents=True, exist_ok=True)

        # Time the extraction
        start_time = time.time()

        # Run osmium extract with bbox
        cmd = [
            'osmium',
            'extract',
            '--bbox', f"{min_lon},{min_lat},{max_lon},{max_lat}",
            str(input_pbf),
            '-o', str(output_pbf),
            '--overwrite'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Osmium extraction failed: {result.stderr}")
            raise RuntimeError(f"Osmium extraction failed: {result.stderr}")

        extraction_time = time.time() - start_time
        output_size_mb = output_pbf.stat().st_size / 1024 / 1024
        input_size_mb = input_pbf.stat().st_size / 1024 / 1024
        reduction_pct = ((input_size_mb - output_size_mb) / input_size_mb) * 100

        logger.info(f"Extraction complete in {extraction_time:.1f}s")
        logger.info(f"  Output: {output_pbf} ({output_size_mb:.1f} MB)")
        logger.info(f"  Size reduction: {reduction_pct:.1f}% ({input_size_mb:.1f} MB -> {output_size_mb:.1f} MB)")

        return {
            'input_size_mb': input_size_mb,
            'output_size_mb': output_size_mb,
            'extraction_time_sec': extraction_time,
            'reduction_percent': reduction_pct,
            'method': 'osmium_bbox'
        }

    def merge_pbf_files(
        self,
        input_pbf_files: List[Path],
        output_pbf: Path
    ) -> Dict:
        """
        Merge multiple PBF files into one using osmium

        Args:
            input_pbf_files: List of PBF files to merge
            output_pbf: Path to output merged PBF file

        Returns:
            Dictionary with merge metadata
        """
        if not self.check_osmium_installed():
            raise RuntimeError("osmium-tool is required. Install with: conda install -c conda-forge osmium-tool")

        if len(input_pbf_files) == 0:
            raise ValueError("No input PBF files provided")

        if len(input_pbf_files) == 1:
            # Only one file, just copy it
            logger.info(f"Only one PBF file, copying to output...")
            import shutil
            shutil.copy2(input_pbf_files[0], output_pbf)
            return {
                'input_files': len(input_pbf_files),
                'output_size_mb': output_pbf.stat().st_size / 1024 / 1024,
                'method': 'copy'
            }

        logger.info(f"Merging {len(input_pbf_files)} PBF files with osmium...")
        for pbf in input_pbf_files:
            logger.info(f"  Input: {pbf.name} ({pbf.stat().st_size / 1024 / 1024:.1f} MB)")

        output_pbf.parent.mkdir(parents=True, exist_ok=True)

        # Time the merge
        start_time = time.time()

        # Run osmium merge
        # Note: Input files should be non-overlapping extracts for best results
        cmd = [
            'osmium',
            'merge',
            *[str(pbf) for pbf in input_pbf_files],
            '-o', str(output_pbf),
            '--overwrite'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Osmium merge failed: {result.stderr}")
            raise RuntimeError(f"Osmium merge failed: {result.stderr}")

        merge_time = time.time() - start_time
        output_size_mb = output_pbf.stat().st_size / 1024 / 1024
        total_input_size_mb = sum(pbf.stat().st_size / 1024 / 1024 for pbf in input_pbf_files)

        logger.info(f"Merge complete in {merge_time:.1f}s")
        logger.info(f"  Output: {output_pbf} ({output_size_mb:.1f} MB)")
        logger.info(f"  Total input size: {total_input_size_mb:.1f} MB")

        return {
            'input_files': len(input_pbf_files),
            'total_input_size_mb': total_input_size_mb,
            'output_size_mb': output_size_mb,
            'merge_time_sec': merge_time,
            'method': 'osmium_merge'
        }

    def extract_with_osmium_polygon(
        self,
        input_pbf: Path,
        output_pbf: Path,
        counties: List[Tuple[str, str]]
    ) -> Dict:
        """
        Extract area from PBF using osmium with county polygons

        Args:
            input_pbf: Path to source PBF file
            output_pbf: Path to output PBF file
            counties: List of (county, state) tuples

        Returns:
            Dictionary with extraction metadata
        """
        if not self.check_osmium_installed():
            raise RuntimeError("osmium-tool is required. Install with: conda install -c conda-forge osmium-tool")

        if not input_pbf.exists():
            raise FileNotFoundError(f"Input PBF not found: {input_pbf}")

        logger.info(f"Extracting area with osmium (polygon)...")
        logger.info(f"  Input: {input_pbf} ({input_pbf.stat().st_size / 1024 / 1024:.1f} MB)")
        logger.info(f"  Counties: {', '.join([f'{c}, {s}' for c, s in counties])}")

        # Get county boundaries and combine into single polygon
        logger.info("Fetching county boundaries...")
        all_gdfs = []
        for county, state in counties:
            try:
                query = f"{county}, {state}, USA"
                gdf = ox.geocode_to_gdf(query)
                all_gdfs.append(gdf)
            except Exception as e:
                logger.warning(f"Could not get boundary for {county}, {state}: {e}")

        if not all_gdfs:
            raise ValueError("Could not get boundaries for any counties")

        # Combine all county polygons
        combined_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))

        # Create temporary GeoJSON file for osmium
        temp_geojson = output_pbf.parent / f"temp_extract_{int(time.time())}.geojson"
        combined_gdf.to_file(temp_geojson, driver='GeoJSON')

        logger.info(f"Created polygon file: {temp_geojson}")

        output_pbf.parent.mkdir(parents=True, exist_ok=True)

        # Time the extraction
        start_time = time.time()

        # Run osmium extract with polygon
        cmd = [
            'osmium',
            'extract',
            '--polygon', str(temp_geojson),
            str(input_pbf),
            '-o', str(output_pbf),
            '--overwrite'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Clean up temp file
        temp_geojson.unlink()

        if result.returncode != 0:
            logger.error(f"Osmium extraction failed: {result.stderr}")
            raise RuntimeError(f"Osmium extraction failed: {result.stderr}")

        extraction_time = time.time() - start_time
        output_size_mb = output_pbf.stat().st_size / 1024 / 1024
        input_size_mb = input_pbf.stat().st_size / 1024 / 1024
        reduction_pct = ((input_size_mb - output_size_mb) / input_size_mb) * 100

        logger.info(f"Extraction complete in {extraction_time:.1f}s")
        logger.info(f"  Output: {output_pbf} ({output_size_mb:.1f} MB)")
        logger.info(f"  Size reduction: {reduction_pct:.1f}% ({input_size_mb:.1f} MB -> {output_size_mb:.1f} MB)")

        return {
            'input_size_mb': input_size_mb,
            'output_size_mb': output_size_mb,
            'extraction_time_sec': extraction_time,
            'reduction_percent': reduction_pct,
            'method': 'osmium_polygon'
        }

    def download_for_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        state: str,
        output_path: Path,
        method: str = 'auto'
    ) -> Path:
        """
        Download OSM data for a specific bounding box

        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            state: State name (for geofabrik download)
            output_path: Where to save OSM file
            method: Download method:
                - 'osmium' (recommended): Download state PBF + extract with osmium
                - 'geofabrik': Download full state PBF only
                - 'overpass': Download via Overpass API (unreliable)
                - 'auto': Choose best method (defaults to 'osmium' if available)

        Returns:
            Path to downloaded/extracted file
        """
        # Auto-select best method
        if method == 'auto':
            if self.check_osmium_installed():
                logger.info(f"Using osmium extraction for bbox (fast, accurate)")
                method = 'osmium'
            else:
                logger.info(f"Using Overpass API for bbox (osmium not available)")
                method = 'overpass'

        # Ensure output has appropriate extension based on download method
        # osmium/geofabrik produce PBF format (more efficient, smaller file size)
        # overpass can convert to PBF if osmconvert is available
        if method in ['osmium', 'geofabrik']:
            if output_path.suffix != '.pbf':
                output_path = output_path.with_suffix('.osm.pbf')
        else:  # overpass
            # Try to use PBF if osmconvert is available, otherwise use OSM
            if self.check_osmconvert_installed():
                if output_path.suffix != '.pbf':
                    output_path = output_path.with_suffix('.osm.pbf')
                logger.info("Will convert Overpass XML to PBF using osmconvert")
            else:
                if output_path.suffix not in ['.osm', '.xml']:
                    output_path = output_path.with_suffix('.osm')
                logger.info("osmconvert not found, will keep OSM XML format")

        if method == 'osmium':
            # Step 1: Download/get cached state PBF
            region = f"north-america/us/{state.lower().replace(' ', '-')}"
            state_pbf = self.download_osm_pbf_from_geofabrik(region, use_cache=True)

            # Step 2: Extract bbox using osmium
            logger.info("Using bbox-based extraction")
            extract_metadata = self.extract_with_osmium_bbox(
                input_pbf=state_pbf,
                output_pbf=output_path,
                bbox=bbox
            )

            logger.info(f"Extraction summary: {extract_metadata}")
            return output_path

        elif method == 'geofabrik':
            # Download entire state from Geofabrik (no extraction)
            region = f"north-america/us/{state.lower().replace(' ', '-')}"
            return self.download_osm_pbf_from_geofabrik(region, output_path, use_cache=True)

        elif method == 'overpass':
            # Download via Overpass API
            logger.warning("Overpass method is unreliable and may timeout. Consider using osmium instead.")
            return self.download_osm_extract_for_bbox(bbox, output_path)

        else:
            raise ValueError(f"Unknown download method: {method}. Use 'auto', 'osmium', 'geofabrik', or 'overpass'")

    def download_for_counties(
        self,
        counties: List[Tuple[str, str]],
        output_path: Path,
        method: str = 'auto',
        extract_method: str = 'bbox'
    ) -> Path:
        """
        Download OSM data for specified counties with osmium extraction
        Supports multi-state county lists

        Args:
            counties: List of (county, state) tuples
            output_path: Where to save OSM file
            method: Download method:
                - 'osmium' (recommended): Download state PBF(s) + extract with osmium
                - 'geofabrik': Download full state PBF only (single state)
                - 'overpass': Download via Overpass API (unreliable)
                - 'auto': Choose best method (defaults to 'osmium' if available)
            extract_method: For osmium method:
                - 'bbox': Extract by bounding box (faster)
                - 'polygon': Extract by county polygons (more accurate)

        Returns:
            Path to downloaded/extracted file
        """
        # Auto-select best method
        if method == 'auto':
            if self.check_osmium_installed():
                logger.info(f"Using osmium extraction for {len(counties)} counties (fast, accurate)")
                method = 'osmium'
            else:
                logger.info(f"Using Geofabrik for {len(counties)} counties (osmium not available)")
                method = 'geofabrik'

        # Ensure output has .pbf extension
        if output_path.suffix != '.pbf':
            output_path = output_path.with_suffix('.osm.pbf')

        if method == 'osmium':
            # Group counties by state
            from collections import defaultdict
            counties_by_state = defaultdict(list)
            for county, state in counties:
                counties_by_state[state].append((county, state))

            unique_states = list(counties_by_state.keys())
            logger.info(f"Counties span {len(unique_states)} state(s): {', '.join(unique_states)}")

            # Step 1: Download/get cached state PBF files for all states
            state_pbf_files = []
            for state in unique_states:
                region = f"north-america/us/{state.lower().replace(' ', '-')}"
                state_pbf = self.download_osm_pbf_from_geofabrik(region, use_cache=True)
                state_pbf_files.append(state_pbf)
                logger.info(f"  {state}: {state_pbf.name}")

            # Step 2: Get bbox for extraction (used for both single and multi-state)
            if extract_method == 'bbox':
                bbox = self.get_bbox_for_counties(counties)
            else:
                bbox = None  # Will be handled by polygon extraction

            # Step 3: Extract from each state file separately
            extracted_files = []
            temp_files_to_cleanup = []

            try:
                for i, (state_pbf, state) in enumerate(zip(state_pbf_files, unique_states)):
                    if len(state_pbf_files) > 1:
                        # Create temporary file for this state's extract
                        temp_extract = output_path.parent / f"temp_extract_{state}_{int(time.time())}_{i}.osm.pbf"
                        temp_files_to_cleanup.append(temp_extract)
                        extract_output = temp_extract
                    else:
                        # Only one state, output directly to final path
                        extract_output = output_path

                    logger.info(f"Extracting from {state} data...")

                    if extract_method == 'polygon':
                        # Filter counties for this state
                        state_counties = counties_by_state[state]
                        logger.info(f"Using polygon-based extraction for {len(state_counties)} counties in {state}")
                        extract_metadata = self.extract_with_osmium_polygon(
                            input_pbf=state_pbf,
                            output_pbf=extract_output,
                            counties=state_counties
                        )
                    else:  # bbox
                        logger.info(f"Using bbox-based extraction for {state}")
                        extract_metadata = self.extract_with_osmium_bbox(
                            input_pbf=state_pbf,
                            output_pbf=extract_output,
                            bbox=bbox
                        )

                    extracted_files.append(extract_output)
                    logger.info(f"Extraction summary for {state}: {extract_metadata}")

                # Step 4: Merge extracted files if multiple states
                if len(extracted_files) > 1:
                    logger.info(f"Merging {len(extracted_files)} extracted regions...")
                    merge_metadata = self.merge_pbf_files(extracted_files, output_path)
                    logger.info(f"Merge summary: {merge_metadata}")

            finally:
                # Clean up temporary extracted files
                for temp_file in temp_files_to_cleanup:
                    if temp_file.exists():
                        temp_file.unlink()
                        logger.info(f"Deleted temporary extract: {temp_file}")

            return output_path

        elif method == 'geofabrik':
            # Download entire state from Geofabrik (no extraction)
            # For multi-state, only download the first state (legacy behavior)
            state = counties[0][1]
            if len(set(s for _, s in counties)) > 1:
                logger.warning(f"Multiple states detected but geofabrik method only supports single state.")
                logger.warning(f"Only downloading {state}. Consider using 'osmium' method instead.")

            region = f"north-america/us/{state.lower().replace(' ', '-')}"
            return self.download_osm_pbf_from_geofabrik(region, output_path, use_cache=True)

        elif method == 'overpass':
            # Get bbox for counties and download via Overpass
            logger.warning("Overpass method is unreliable and may timeout. Consider using osmium instead.")
            bbox = self.get_bbox_for_counties(counties)
            return self.download_osm_extract_for_bbox(bbox, output_path)

        else:
            raise ValueError(f"Unknown download method: {method}. Use 'auto', 'osmium', 'geofabrik', or 'overpass'")
