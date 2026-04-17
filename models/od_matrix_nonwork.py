"""Non-work OD matrix creation for Shopping, Recreation, and other purposes.

This module creates origin-destination matrices for non-work trips using:
1. Survey data (TBI survey filtered by purpose)
2. Gravity model with singly-constrained IPF
   - Origin constraint: non_employees at home blocks
   - Destination attractiveness: POI density per block
3. Alpha blending to combine survey and gravity model

Similar to od_matrix_v3.py but adapted for non-work trips.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from utils.logger import setup_logger
from data_sources.base_survey_trip import BaseSurveyTrip
from models.od_matrix_v3 import combine_od_matrices
from utils.poi_weighting import POIWeighting

logger = setup_logger(__name__)

# Maps geo level constant → number of GEOID characters used as the zone key
_GEO_LEVEL_PREFIX_LEN: Dict[str, int] = {
    BaseSurveyTrip.GEO_TRACT: 11,
    BaseSurveyTrip.GEO_BLOCK_GROUP: 12,
}


def _aggregate_to_geo_level(
    home_locs_dict: Dict[str, Dict[str, Any]],
    poi_density_dict: Dict[str, int],
    geo_level: str = BaseSurveyTrip.GEO_BLOCK_GROUP,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Pre-aggregate block-level data to the requested census geography level.

    Reduces the dimensionality of the cdist call — e.g. from ~164K×31K blocks
    to ~4K×3K block groups, cutting memory from ~113 GB to < 100 MB.

    Args:
        home_locs_dict: Block-level dict {geoid_15: {lat, lon, non_employees, ...}}
        poi_density_dict: Block-level dict {geoid_15: poi_count}
        geo_level: Target census geography — GEO_BLOCK_GROUP (12-digit, default)
                   or GEO_TRACT (11-digit).

    Returns:
        (zone_home_dict, zone_poi_dict) at the requested granularity.
        zone_home_dict values have keys: lat, lon, non_employees.
        Coordinates are weighted averages (weighted by non_employees).
    """
    from collections import defaultdict

    prefix_len = _GEO_LEVEL_PREFIX_LEN.get(geo_level, 12)

    # --- Aggregate home locations ---
    bg_data = defaultdict(lambda: {"sum_ne": 0.0, "wlat": 0.0, "wlon": 0.0,
                                    "ulat": 0.0, "ulon": 0.0, "n": 0})
    for geoid, data in home_locs_dict.items():
        bg = geoid[:prefix_len]
        ne = data.get("non_employees", 0) or 0
        lat = data.get("lat", 0.0) or 0.0
        lon = data.get("lon", 0.0) or 0.0
        entry = bg_data[bg]
        entry["sum_ne"] += ne
        entry["wlat"] += lat * ne
        entry["wlon"] += lon * ne
        entry["ulat"] += lat
        entry["ulon"] += lon
        entry["n"] += 1

    bg_home_dict = {}
    for bg, entry in bg_data.items():
        ne = entry["sum_ne"]
        if ne > 0:
            lat = entry["wlat"] / ne
            lon = entry["wlon"] / ne
        elif entry["n"] > 0:
            lat = entry["ulat"] / entry["n"]
            lon = entry["ulon"] / entry["n"]
        else:
            continue
        bg_home_dict[bg] = {"lat": lat, "lon": lon, "non_employees": ne}

    # --- Aggregate POI density ---
    bg_poi_dict: Dict[str, int] = defaultdict(int)
    for geoid, count in poi_density_dict.items():
        if count > 0:
            bg_poi_dict[geoid[:prefix_len]] += count

    logger.info(f"  Pre-aggregated {len(home_locs_dict):,} blocks → "
                f"{len(bg_home_dict):,} {geo_level}s (home)")
    logger.info(f"  Pre-aggregated POI blocks → "
                f"{len(bg_poi_dict):,} {geo_level}s with POIs")

    return bg_home_dict, dict(bg_poi_dict)



def calculate_poi_density_per_block(poi_data: List[Dict[str, Any]], purpose: str,
                                     blockid2homelocs: Dict[str, Dict[str, Any]],
                                     config: Dict[str, Any],
                                     poi_block_mapping: Dict[str, str] = None) -> Dict[str, int]:
    """
    Calculate POI count per block for a specific purpose.

    Args:
        poi_data: List of POI dictionaries with keys: 'osm_id', 'name', 'activity', 'lat', 'lon', 'tags'
        purpose: Trip purpose ('Shopping', 'Recreation', etc.)
        blockid2homelocs: Dict of home blocks with lat/lon for calculating which block each POI belongs to
        config: Configuration dictionary
        poi_block_mapping: Optional pre-computed mapping of POI osm_id to block_id.
                          If provided, skips expensive spatial computation.

    Returns:
        Dict mapping block_id to POI count: {block_id: poi_count}
    """
    logger.info(f"Calculating POI density for {purpose}...")
    logger.info(f"  Total POIs to process: {len(poi_data):,}")

    # Filter POIs by activity matching the canonical purpose name directly
    purpose_pois = [poi for poi in poi_data if poi.get('activity') == purpose]
    logger.info(f"  Filtered to {len(purpose_pois):,} {purpose} POIs")

    if len(purpose_pois) == 0:
        logger.warning(f"No POIs found for purpose '{purpose}'")
        return {}

    # Create block ID to POI count mapping
    block_poi_counts = {block_id: 0 for block_id in blockid2homelocs.keys()}

    # If pre-computed mapping is provided, use it (fast path)
    if poi_block_mapping is not None:
        logger.info(f"  Using pre-computed POI-to-block mapping (fast path)")
        for poi in purpose_pois:
            osm_id = poi.get('osm_id')
            if osm_id in poi_block_mapping:
                block_id = poi_block_mapping[osm_id]
                if block_id in block_poi_counts:
                    block_poi_counts[block_id] += 1
    else:
        # Fall back to computing mapping (slow path - for backwards compatibility)
        logger.info(f"  Computing POI-to-block mapping (slow path)")
        block_poi_counts = _compute_poi_density_spatial(purpose_pois, blockid2homelocs, config)

    # Log statistics
    blocks_with_pois = sum(1 for count in block_poi_counts.values() if count > 0)
    total_pois_assigned = sum(block_poi_counts.values())

    logger.info(f"  POI assignment complete:")
    logger.info(f"    Blocks with POIs: {blocks_with_pois:,}")
    logger.info(f"    Total POIs assigned: {total_pois_assigned:,}")
    if blocks_with_pois > 0:
        logger.info(f"    Avg POIs per block (non-zero): {total_pois_assigned / blocks_with_pois:.1f}")

    return block_poi_counts


def _compute_poi_density_spatial(purpose_pois: List[Dict[str, Any]],
                                  blockid2homelocs: Dict[str, Dict[str, Any]],
                                  config: Dict[str, Any]) -> Dict[str, int]:
    """
    Compute POI density using spatial operations (vectorized for performance).

    This is the slow path used when no pre-computed mapping is available.

    Args:
        purpose_pois: List of POI dictionaries already filtered by purpose
        blockid2homelocs: Dict of home blocks with lat/lon
        config: Configuration dictionary

    Returns:
        Dict mapping block_id to POI count
    """
    from shapely.geometry import Point
    import geopandas as gpd
    from scipy.spatial import cKDTree

    block_poi_counts = {block_id: 0 for block_id in blockid2homelocs.keys()}

    # Build arrays for blocks
    block_ids = []
    block_coords = []
    for block_id, block_data in blockid2homelocs.items():
        if block_data.get('lat') and block_data.get('lon'):
            block_ids.append(block_id)
            block_coords.append([block_data['lon'], block_data['lat']])

    if len(block_coords) == 0:
        logger.error("No blocks with valid coordinates found")
        return block_poi_counts

    block_coords = np.array(block_coords)

    # Get UTM CRS for accurate distance calculation
    utm_crs = config['coordinates']['utm_epsg']

    # Convert block coordinates to UTM (vectorized)
    blocks_gdf = gpd.GeoDataFrame(
        {'block_id': block_ids},
        geometry=[Point(lon, lat) for lon, lat in block_coords],
        crs='EPSG:4326'
    )
    blocks_gdf_proj = blocks_gdf.to_crs(utm_crs)

    # Extract projected coordinates for KD-tree
    block_coords_utm = np.array([[geom.x, geom.y] for geom in blocks_gdf_proj.geometry])

    # Build KD-tree for fast nearest neighbor lookup
    tree = cKDTree(block_coords_utm)

    # Build POI coordinates array
    poi_coords = []
    for poi in purpose_pois:
        poi_coords.append([poi['lon'], poi['lat']])

    if len(poi_coords) == 0:
        return block_poi_counts

    poi_coords = np.array(poi_coords)

    # Convert all POI coordinates to UTM in one batch operation
    pois_gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in poi_coords],
        crs='EPSG:4326'
    )
    pois_gdf_proj = pois_gdf.to_crs(utm_crs)

    # Extract projected coordinates
    poi_coords_utm = np.array([[geom.x, geom.y] for geom in pois_gdf_proj.geometry])

    # Query KD-tree for nearest block for all POIs at once
    _, nearest_indices = tree.query(poi_coords_utm)

    # Count POIs per block
    for idx in nearest_indices:
        block_id = block_ids[idx]
        block_poi_counts[block_id] += 1

    return block_poi_counts


def compute_poi_block_mapping(poi_data: List[Dict[str, Any]],
                               blockid2homelocs: Dict[str, Dict[str, Any]],
                               config: Dict[str, Any]) -> Dict[str, str]:
    """
    Pre-compute mapping from POI osm_id to nearest block_id for ALL POIs.

    This is done ONCE and shared across all activity types to avoid
    redundant spatial computations.

    Args:
        poi_data: List of ALL POI dictionaries (not filtered by purpose)
        blockid2homelocs: Dict of home blocks with lat/lon
        config: Configuration dictionary

    Returns:
        Dict mapping POI osm_id to block_id: {osm_id: block_id}
    """
    from shapely.geometry import Point
    import geopandas as gpd
    from scipy.spatial import cKDTree

    logger.info("Pre-computing POI-to-block mapping for ALL POIs...")
    logger.info(f"  Total POIs: {len(poi_data):,}")

    # Build arrays for blocks
    block_ids = []
    block_coords = []
    for block_id, block_data in blockid2homelocs.items():
        if block_data.get('lat') and block_data.get('lon'):
            block_ids.append(block_id)
            block_coords.append([block_data['lon'], block_data['lat']])

    if len(block_coords) == 0:
        logger.error("No blocks with valid coordinates found")
        return {}

    block_coords = np.array(block_coords)
    logger.info(f"  Blocks with coordinates: {len(block_ids):,}")

    # Get UTM CRS for accurate distance calculation
    utm_crs = config['coordinates']['utm_epsg']

    # Convert block coordinates to UTM (vectorized)
    blocks_gdf = gpd.GeoDataFrame(
        {'block_id': block_ids},
        geometry=[Point(lon, lat) for lon, lat in block_coords],
        crs='EPSG:4326'
    )
    blocks_gdf_proj = blocks_gdf.to_crs(utm_crs)

    # Extract projected coordinates for KD-tree
    block_coords_utm = np.array([[geom.x, geom.y] for geom in blocks_gdf_proj.geometry])

    # Build KD-tree for fast nearest neighbor lookup
    tree = cKDTree(block_coords_utm)
    logger.info(f"  Built KD-tree spatial index")

    # Build POI coordinates array and track osm_ids
    poi_osm_ids = []
    poi_coords = []
    for poi in poi_data:
        if poi.get('lat') is not None and poi.get('lon') is not None and poi.get('osm_id') is not None:
            poi_osm_ids.append(poi['osm_id'])
            poi_coords.append([poi['lon'], poi['lat']])

    if len(poi_coords) == 0:
        logger.warning("No POIs with valid coordinates found")
        return {}

    poi_coords = np.array(poi_coords)
    logger.info(f"  POIs with valid coordinates: {len(poi_coords):,}")

    # Convert all POI coordinates to UTM in one batch operation
    logger.info(f"  Converting POI coordinates to {utm_crs}...")
    pois_gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in poi_coords],
        crs='EPSG:4326'
    )
    pois_gdf_proj = pois_gdf.to_crs(utm_crs)

    # Extract projected coordinates
    poi_coords_utm = np.array([[geom.x, geom.y] for geom in pois_gdf_proj.geometry])

    # Query KD-tree for nearest block for all POIs at once
    logger.info(f"  Finding nearest block for each POI...")
    _, nearest_indices = tree.query(poi_coords_utm)

    # Build mapping
    poi_block_mapping = {}
    for i, osm_id in enumerate(poi_osm_ids):
        block_id = block_ids[nearest_indices[i]]
        poi_block_mapping[osm_id] = block_id

    logger.info(f"  POI-to-block mapping complete: {len(poi_block_mapping):,} POIs mapped")

    return poi_block_mapping


def create_survey_od_matrix_nonwork(survey_df: pd.DataFrame, purpose: str,
                                     config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create OD matrix from survey data for a specific non-work purpose.

    Args:
        survey_df: Survey DataFrame with canonical columns (origin_loc, destination_loc, destination_purpose)
        purpose: Canonical trip purpose to filter ('Shopping', 'Social', etc.)
        config: Configuration dictionary with nonwork_purposes settings

    Returns:
        OD matrix DataFrame (block group level)
    """
    logger.info(f"Creating survey OD matrix for {purpose}...")

    # Filter directly by canonical purpose name (no activity_mapping needed)
    d_col = BaseSurveyTrip.DESTINATION_PURPOSE
    purpose_df = survey_df[survey_df[d_col] == purpose].copy()
    logger.info(f"  Filtered to {len(purpose_df):,} trips (from {len(survey_df):,} total)")

    if len(purpose_df) == 0:
        logger.warning(f"No survey trips found for purpose '{purpose}'")
        return pd.DataFrame()

    # Ensure columns are strings
    o_col = BaseSurveyTrip.ORIGIN_LOC
    d_loc_col = BaseSurveyTrip.DESTINATION_LOC
    purpose_df[o_col] = purpose_df[o_col].astype(str).str.strip()
    purpose_df[d_loc_col] = purpose_df[d_loc_col].astype(str).str.strip()

    # Create OD matrix using crosstab
    od_matrix = pd.crosstab(purpose_df[o_col], purpose_df[d_loc_col])

    logger.info(f"  Survey OD matrix created: {od_matrix.shape}")
    logger.info(f"  Total trips in survey matrix: {od_matrix.sum().sum():,.0f}")

    return od_matrix


def calculate_trip_rate_from_survey(survey_df: pd.DataFrame, purpose: str, config: Dict[str, Any]) -> float:
    """
    Calculate trip generation rate from survey data.

    Trip rate = (number of trips for purpose) / (total number of people in survey)

    Args:
        survey_df: Survey DataFrame with canonical columns
        purpose: Canonical trip purpose ('Shopping', etc.)
        config: Configuration dictionary

    Returns:
        Trip rate as a float (e.g., 0.25 means 25% of people make this trip type)
    """
    # Count trips matching this canonical purpose directly
    d_col = BaseSurveyTrip.DESTINATION_PURPOSE
    purpose_trips = survey_df[survey_df[d_col] == purpose]
    num_purpose_trips = len(purpose_trips)

    # Get unique person count from survey
    # Assuming survey has a person_id or similar field - we'll use unique trip origins as proxy
    # This is a simplified approach; ideally we'd have person-level data
    total_survey_records = len(survey_df)

    if total_survey_records == 0:
        logger.warning(f"No survey records found for trip rate calculation")
        return 0.0

    trip_rate = num_purpose_trips / total_survey_records

    logger.info(f"Survey trip rate calculation:")
    logger.info(f"  Purpose trips: {num_purpose_trips:,}")
    logger.info(f"  Total survey records: {total_survey_records:,}")
    logger.info(f"  Calculated trip rate: {trip_rate:.2%}")

    return trip_rate


def calculate_blended_survey_trip_rate(survey_dfs: Dict[str, pd.DataFrame],
                                      weights: Dict[str, float],
                                      purpose: str,
                                      config: Dict[str, Any]) -> float:
    """Compute a weighted-average survey trip rate across multiple sources.

    Only surveys with location data (non-null ``origin_loc``) contribute
    because regional trip rates from national surveys (e.g. NHTS) are not
    representative of the study area.

    Args:
        survey_dfs: {source_key: DataFrame} — all loaded survey DataFrames.
        weights:    {source_key: float} — raw config weights.
        purpose:    Canonical trip purpose (e.g. ``'Shopping'``).
        config:     Full configuration dictionary.

    Returns:
        Weighted-average survey trip rate (float).  If no survey has
        location data, returns 0.0 (caller should fall back to
        ``config_rate``).
    """
    o_col = BaseSurveyTrip.ORIGIN_LOC

    # Filter to surveys with location data
    loc_surveys = {
        name: df for name, df in survey_dfs.items()
        if df[o_col].notna().any() if o_col in df.columns
    }

    if not loc_surveys:
        logger.info(
            f"calculate_blended_survey_trip_rate({purpose}): "
            "no survey has location data — returning 0.0"
        )
        return 0.0

    # Renormalise weights among location-capable surveys
    loc_names = list(loc_surveys.keys())
    raw = np.array([weights[n] for n in loc_names], dtype=float)
    total = raw.sum()
    if total <= 0:
        return 0.0
    norm_weights = raw / total

    # Weighted average of per-survey rates
    blended_rate = 0.0
    for name, w in zip(loc_names, norm_weights):
        rate = calculate_trip_rate_from_survey(loc_surveys[name], purpose, config)
        blended_rate += w * rate
        logger.info(
            f"  {name}: survey_rate={rate:.4f}, weight={w:.3f}"
        )

    logger.info(
        f"calculate_blended_survey_trip_rate({purpose}): "
        f"blended={blended_rate:.4f} from {len(loc_names)} sources"
    )
    return blended_rate


def create_gravity_od_matrix_nonwork(home_locs_dict: Dict[str, Dict[str, Any]],
                                      poi_density_dict: Dict[str, int],
                                      purpose: str,
                                      config: Dict[str, Any],
                                      survey_df: pd.DataFrame,
                                      beta: float = 2.0,
                                      max_iterations: int = 50,
                                      convergence_threshold: float = 1e-4,
                                      geo_level: str = BaseSurveyTrip.GEO_BLOCK_GROUP) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Create singly-constrained gravity model OD matrix for non-work trips.

    Differs from work trip gravity model:
    - Uses non_employees instead of n_employees as origin constraint
    - Uses POI density as destination attractiveness (not doubly-constrained)
    - IPF only enforces origin constraint, destinations are free
    - Applies trip generation rate (blended from survey and config)

    Args:
        home_locs_dict: Dict with home blocks and non_employees counts
        poi_density_dict: Dict mapping block_id to POI count
        purpose: Trip purpose ('Shopping', etc.)
        config: Configuration dictionary
        survey_df: Survey DataFrame for calculating trip rate
        beta: Distance decay parameter
        max_iterations: Max IPF iterations
        convergence_threshold: IPF convergence threshold

    Returns:
        Tuple of (od_matrix DataFrame, home_geoids list, dest_geoids list)
    """
    from scipy.spatial.distance import cdist

    logger.info(f"=" * 70)
    logger.info(f"SINGLY-CONSTRAINED GRAVITY MODEL - {purpose.upper()}")
    logger.info(f"=" * 70)

    # Calculate trip generation rate (blend survey and config)
    purpose_config = config.get('nonwork_purposes', {}).get(purpose, {})
    trip_gen_config = purpose_config.get('trip_generation', {})

    # Get survey rate
    survey_rate_config = trip_gen_config.get('survey_rate', 'auto')
    if survey_rate_config == 'auto':
        survey_rate = calculate_trip_rate_from_survey(survey_df, purpose, config)
    else:
        survey_rate = float(survey_rate_config)
        logger.info(f"Using configured survey rate: {survey_rate:.2%}")

    # Get config rate
    config_rate = trip_gen_config.get('config_rate', 0.30)

    # Get blend weight
    blend_weight = trip_gen_config.get('blend_weight', 0.5)

    # Calculate final trip rate (blended from survey and config)
    final_trip_rate = (1 - blend_weight) * survey_rate + blend_weight * config_rate

    logger.info(f"")
    logger.info(f"Trip Generation Rate Calculation:")
    logger.info(f"  Survey rate: {survey_rate:.2%}")
    logger.info(f"  Config rate: {config_rate:.2%}")
    logger.info(f"  Blend weight: {blend_weight:.2f} (0=survey only, 1=config only)")
    logger.info(f"  Final trip rate: {final_trip_rate:.2%}")
    logger.info(f"")

    # Pre-aggregate blocks → block groups to avoid OOM on large cdist matrices
    bg_home, bg_poi = _aggregate_to_geo_level(home_locs_dict, poi_density_dict, geo_level=geo_level)

    # Get sorted geoid lists at the aggregated geography level
    home_geoids = sorted(bg_home.keys())
    dest_geoids = sorted([gid for gid in bg_home.keys() if bg_poi.get(gid, 0) > 0])

    logger.info(f"Number of home {geo_level}s: {len(home_geoids)}")
    logger.info(f"Number of destination {geo_level}s (with POIs): {len(dest_geoids)}")
    logger.info(f"Distance decay beta: {beta}")

    if len(dest_geoids) == 0:
        logger.error("No destination block groups with POIs found!")
        return pd.DataFrame(), [], []

    # Extract origin constraint (non-employees) and apply trip rate
    Oi_base = np.array([bg_home[geoid]['non_employees'] for geoid in home_geoids], dtype=np.float64)
    Oi = Oi_base * final_trip_rate

    # Extract destination attractiveness (POI count)
    Aj = np.array([bg_poi.get(geoid, 0) for geoid in dest_geoids], dtype=np.float64)

    logger.info(f"Total non-employees (base): {Oi_base.sum():,.0f}")
    logger.info(f"Total travelers (after trip rate): {Oi.sum():,.0f}")
    logger.info(f"Total POIs (destination): {Aj.sum():,.0f}")

    # Extract coordinates
    logger.info("Extracting coordinates...")
    home_coords = np.array([(bg_home[gid]['lon'], bg_home[gid]['lat'])
                            for gid in home_geoids], dtype=np.float64)
    dest_coords = np.array([(bg_home[gid]['lon'], bg_home[gid]['lat'])
                            for gid in dest_geoids], dtype=np.float64)

    # Calculate distance matrix → friction factors in-place (single array)
    logger.info("Calculating distances and friction factors...")
    friction = cdist(home_coords, dest_coords, metric='euclidean')
    friction *= 111.32  # degrees → km
    np.maximum(friction, 0.1, out=friction)  # avoid zero distances
    np.power(friction, -beta, out=friction)   # distance^(-beta)

    # Initialize OD matrix: Tij = Oi * (Aj * f_ij) / sum_j(Aj * f_ij)
    denominator = np.sum(Aj * friction, axis=1, keepdims=True)
    np.maximum(denominator, 1e-30, out=denominator)  # avoid division by zero

    od_matrix = Oi.reshape(-1, 1) * (Aj * friction) / denominator
    del friction  # free memory

    logger.info(f"Initial gravity model total trips: {od_matrix.sum():,.0f}")

    # IPF to enforce origin constraint (singly-constrained)
    logger.info(f"\nRunning IPF iterations (max {max_iterations})...")

    for iteration in range(max_iterations):
        # Row scaling only: adjust to match origin constraints
        row_sums = od_matrix.sum(axis=1, keepdims=True)
        row_factors = np.divide(
            Oi.reshape(-1, 1),
            row_sums,
            out=np.ones_like(row_sums),
            where=row_sums != 0
        )
        od_matrix = od_matrix * row_factors

        # Check convergence
        row_sums = od_matrix.sum(axis=1)
        row_rel_diff = np.abs(row_sums - Oi) / (Oi + 1e-10)
        max_rel_diff = np.max(row_rel_diff)

        if (iteration + 1) % 10 == 0 or max_rel_diff < convergence_threshold:
            logger.info(f"  Iteration {iteration + 1}: Max relative difference = {max_rel_diff:.2e}")

        if max_rel_diff < convergence_threshold:
            logger.info(f"Converged after {iteration + 1} iterations")
            break

    # Verification
    logger.info("=" * 70)
    logger.info("VERIFICATION")
    logger.info("=" * 70)

    final_row_sums = od_matrix.sum(axis=1)
    max_row_error = np.abs(final_row_sums - Oi).max()

    logger.info(f"Origin total: {Oi.sum():,.0f} | Matrix row sum: {final_row_sums.sum():,.0f}")
    logger.info(f"Max row constraint error: {max_row_error:.2e}")
    logger.info(f"Total trips in gravity model: {od_matrix.sum():,.0f}")
    logger.info("")

    # Convert to DataFrame
    od_df = pd.DataFrame(od_matrix, index=home_geoids, columns=dest_geoids)

    return od_df, home_geoids, dest_geoids


def create_nonwork_od_matrix(config: Dict[str, Any],
                              home_locs_dict: Dict[str, Dict[str, Any]],
                              poi_data: List[Dict[str, Any]],
                              survey_df: pd.DataFrame,
                              purpose: str,
                              poi_block_mapping: Dict[str, str] = None,
                              geo_level: str = BaseSurveyTrip.GEO_BLOCK_GROUP) -> pd.DataFrame:
    """
    Create complete OD matrix for a non-work purpose by combining survey and gravity model.

    Args:
        config: Configuration dictionary
        home_locs_dict: Home locations with non_employees
        poi_data: List of all POI dictionaries
        survey_df: TBI survey data
        purpose: Trip purpose ('Shopping', 'Recreation', etc.)
        poi_block_mapping: Optional pre-computed mapping of POI osm_id to block_id.
                          If provided, significantly speeds up POI density calculation.
        geo_level: Census geography level of the survey location IDs —
                   BaseSurveyTrip.GEO_BLOCK_GROUP (default) or BaseSurveyTrip.GEO_TRACT.

    Returns:
        Combined OD matrix at the geography level matching the survey.
    """
    logger.info("=" * 70)
    logger.info(f"CREATING NON-WORK OD MATRIX - {purpose.upper()}")
    logger.info("=" * 70)

    # Get purpose-specific config
    purpose_config = config.get('nonwork_purposes', {}).get(purpose, {})
    beta = purpose_config.get('od_matrix', {}).get('beta', 2.0)
    alpha = purpose_config.get('od_matrix', {}).get('alpha', 0.1)

    logger.info(f"Configuration:")
    logger.info(f"  Beta (distance decay): {beta}")
    logger.info(f"  Alpha (survey weight): {alpha}")

    # Step 1: Calculate POI density per block (uses pre-computed mapping if available)
    poi_density_dict = calculate_poi_density_per_block(
        poi_data, purpose, home_locs_dict, config, poi_block_mapping
    )

    # Step 2: Create survey OD matrix
    survey_od_matrix = create_survey_od_matrix_nonwork(survey_df, purpose, config)

    # Step 3: Create gravity model OD matrix
    gravity_od_matrix, home_geoids, dest_geoids = create_gravity_od_matrix_nonwork(
        home_locs_dict,
        poi_density_dict,
        purpose,
        config,
        survey_df,
        beta=beta,
        geo_level=geo_level,
    )

    if gravity_od_matrix.empty:
        logger.error(f"Failed to create gravity model OD matrix for {purpose}")
        return pd.DataFrame()

    # Step 4: Gravity model is already at block-group level (pre-aggregated)
    gravity_od_bg = gravity_od_matrix
    logger.info(f"  Gravity model (block-group level): {gravity_od_bg.shape}")

    if not survey_od_matrix.empty:
        survey_od_bg = survey_od_matrix  # Already at block group level
        logger.info(f"  Survey matrix: {survey_od_bg.shape}")
    else:
        logger.warning("Survey matrix is empty, using gravity model only")
        return gravity_od_bg

    # Step 5: Combine survey and gravity models
    logger.info("\nCombining survey and gravity models...")
    combined_od_matrix = combine_od_matrices(
        survey_od_bg,
        gravity_od_bg,
        alpha=alpha,
        scale_to_total=None  # Use gravity model total
    )

    # Step 6: Apply nonwork_trip_share (fraction of non-workers who travel on a given day)
    nonwork_trip_share = config.get('nonwork_purposes', {}).get('nonwork_trip_share', 1.0)
    if nonwork_trip_share < 1.0:
        total_before = combined_od_matrix.sum().sum()
        combined_od_matrix = combined_od_matrix * nonwork_trip_share
        total_after = combined_od_matrix.sum().sum()
        logger.info(f"\nApplied nonwork_trip_share: {nonwork_trip_share:.2%}")
        logger.info(f"  Trips before scaling: {total_before:,.0f}")
        logger.info(f"  Trips after scaling: {total_after:,.0f}")

    logger.info("=" * 70)
    logger.info(f"NONWORK OD MATRIX CREATION COMPLETE - {purpose.upper()}")
    logger.info("=" * 70)

    return combined_od_matrix


def generate_samples_from_od_matrix(od_matrix: pd.DataFrame, n_samples: int | str) -> list:
    """
    Sample origin-destination pairs from an OD matrix using Hamilton's method.

    This function allocates the target number of trips proportionally across
    all OD pairs in the matrix, ensuring that the allocation matches the
    relative weights in the OD matrix.

    Args:
        od_matrix: DataFrame with origins as index, destinations as columns,
                  and cell values representing trip counts/weights
        n_samples: Number of trips to sample. Can be:
                  - int: specific number of trips to generate
                  - "all": use the total from the OD matrix

    Returns:
        List of tuples (origin_bg, dest_bg, num_trips) where num_trips
        is the allocated number of trips for that OD pair

    Example:
        >>> od_matrix = pd.DataFrame({
        ...     'dest1': [10, 5],
        ...     'dest2': [3, 2]
        ... }, index=['orig1', 'orig2'])
        >>> samples = generate_samples_from_od_matrix(od_matrix, 100)
        >>> # Returns proportional allocation, e.g.:
        >>> # [('orig1', 'dest1', 50), ('orig1', 'dest2', 15), ...]
    """
    # Handle "all" case
    if isinstance(n_samples, str) and n_samples.lower() == "all":
        n_samples = int(od_matrix.sum().sum())
    elif not isinstance(n_samples, int):
        raise ValueError(f"n_samples must be int or 'all', got {type(n_samples)}")

    # Flatten OD matrix to list of (origin, dest, weight) tuples (vectorized)
    stacked = od_matrix.stack()
    nonzero = stacked[stacked > 0]
    od_pairs = [(origin, dest, weight) for (origin, dest), weight in nonzero.items()]

    if not od_pairs:
        logger.warning("OD matrix has no non-zero entries")
        return []

    # Calculate total weight
    total_weight = sum(weight for _, _, weight in od_pairs)

    if total_weight == 0:
        logger.warning("OD matrix has zero total weight")
        return []

    # Use Hamilton's method for proportional allocation
    allocations = {}
    remainders = []
    total_allocated = 0

    for origin, dest, weight in od_pairs:
        # Calculate exact quota
        exact_quota = (weight / total_weight) * n_samples
        floor_quota = int(exact_quota)

        # Store allocation
        od_pair = (origin, dest)
        allocations[od_pair] = floor_quota
        total_allocated += floor_quota

        # Store remainder for Hamilton's method
        remainder = exact_quota - floor_quota
        if remainder > 0:
            remainders.append((remainder, od_pair))

    # Distribute remaining trips to pairs with largest remainders
    remaining_trips = n_samples - total_allocated
    remainders.sort(reverse=True)

    for i in range(remaining_trips):
        if i < len(remainders):
            _, od_pair = remainders[i]
            allocations[od_pair] += 1

    # Convert to list of (origin, dest, count) tuples
    result = [
        (origin, dest, count)
        for (origin, dest), count in allocations.items()
        if count > 0  # Only return pairs with allocated trips
    ]

    return result
