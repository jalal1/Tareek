"""Process work locations from LODES data and save to DuckDB.

This module uses the LEHD LODES (Longitudinal Employer-Household Dynamics
Origin-Destination Employment Statistics) dataset via the pygris package
to get work location data with employee counts and coordinates.

Data source: U.S. Census Bureau LODES
Access method: pygris.data.get_lodes()
"""
from typing import Dict, Any
import pandas as pd
from utils.logger import setup_logger, CHECK_MARK
from models.models import WorkLocation, initialize_tables
from utils.region_utils import RegionHelper

logger = setup_logger(__name__)


# ── Freight-relevant employment (LODES WAC CNS* sectors) ────────────────────
#
# Total employment (C000) is a poor freight attractor: a warehouse and an
# insurance office of the same headcount generate entirely different truck
# traffic. LODES publishes employment by NAICS sector in the *same* WAC file
# as C000 — verified: all 20 CNS columns arrive in the download and were
# previously discarded.
#
# These are the goods-producing and goods-handling sectors, the ones that
# generate or receive truck trips. Memphis regresses truck trip ends on
# sector employment and reports R^2 0.77-1.00 (QRFM 3rd ed., pp. 243-248),
# against total employment which cannot separate the two cases above.
#
# Weights are relative truck-trip intensity per job, not headcount shares.
# Warehousing moves freight by definition and is weighted highest;
# manufacturing and wholesale are the classic generators; retail receives
# deliveries; construction draws material trucks to sites. They are
# deliberately coarse — the point is to separate a distribution centre from an
# office park, not to model any sector precisely.
FREIGHT_SECTORS = {
    'CNS01': ('agriculture', 0.6),
    'CNS02': ('mining', 0.6),
    'CNS04': ('utilities', 0.3),
    'CNS05': ('construction', 0.8),
    'CNS06': ('manufacturing', 1.0),
    'CNS07': ('wholesale_trade', 1.0),
    'CNS08': ('retail_trade', 0.5),
    'CNS09': ('transport_warehousing', 1.2),
}

#: Every CNS column LODES publishes, so the ETL can assert the freight subset
#: is actually present rather than silently summing nothing.
ALL_CNS_COLUMNS = tuple(f'CNS{i:02d}' for i in range(1, 21))


def freight_employment(frame: "pd.DataFrame") -> "pd.Series":
    """Weighted freight-relevant employment per block.

    Returns a Series aligned to ``frame``. Missing sector columns are treated
    as zero rather than raising: LODES column sets vary slightly by year and
    state, and a partial sum is far better than losing the attractor entirely.
    """
    total = None
    present = []
    for column, (_, weight) in FREIGHT_SECTORS.items():
        if column not in frame.columns:
            continue
        # to_numeric before fillna: an all-null column arrives as object dtype,
        # where fillna triggers a pandas downcasting FutureWarning. Coercing
        # first keeps this numeric throughout.
        values = pd.to_numeric(frame[column], errors='coerce').fillna(0.0) * weight
        total = values if total is None else total + values
        present.append(column)

    if total is None:
        logger.warning(
            "No CNS sector columns in the LODES download; freight employment "
            "cannot be computed and freight will fall back to total employment."
        )
        return pd.Series(0, index=frame.index, dtype=float)

    missing = sorted(set(FREIGHT_SECTORS) - set(present))
    if missing:
        logger.warning(
            f"LODES download is missing sector columns {missing}; freight "
            f"employment is a partial sum over {len(present)} sectors."
        )
    return total


def process_work_locations(config: Dict[str, Any]) -> None:
    """
    Download and process work locations using LODES data via pygris.

    Steps performed:
    1. Initialize RegionHelper to get county FIPS codes from config
    2. Download LODES WAC (Workplace Area Characteristics) data for each state
    3. Filter by selected counties using FIPS codes
    4. Extract geoid, n_employees, lat, lon for each census block
    5. Save to DuckDB work_locations table

    Args:
        config: Configuration dictionary with 'region.counties' and 'data.lodes' sections

    Raises:
        ValueError: If county names cannot be found or LODES download fails
    """
    try:
        from pygris.data import get_lodes
    except ImportError:
        raise ImportError(
            "pygris package is required. Install with: pip install pygris"
        )

    logger.info("=" * 70)
    logger.info("PROCESSING WORK LOCATIONS FROM LODES")
    logger.info("=" * 70)

    # Initialize region helper
    region = RegionHelper(config)

    # Get county FIPS mapping and state abbreviations
    fips_mapping = region.get_county_fips_mapping()
    state_abbr_mapping = region.get_state_abbr_mapping()

    # Get LODES configuration
    lodes_config = config['data']['lodes']
    year = lodes_config['year']

    logger.info(f"LODES Configuration:")
    logger.info(f"  Year: {year}")
    logger.info(f"  Job Type: {lodes_config['job_type']}")
    logger.info(f"  Segment: {lodes_config['segment']}")

    all_data = []

    # Download LODES WAC data for each state
    for state_fips, county_fips_list in fips_mapping.items():
        state_abbr = state_abbr_mapping[state_fips]
        logger.info(f"\nProcessing {state_abbr.upper()}:")
        logger.info(f"  Downloading LODES WAC (Workplace Area Characteristics) data...")

        try:
            wac_df = get_lodes(
                state=state_abbr,
                year=year,
                lodes_type="wac",  # Workplace Area Characteristics
                return_lonlat=True,  # Get lat/lon centroids
                cache=True  # Cache for faster reuse
            )

            logger.info(f"  Downloaded {len(wac_df):,} blocks for entire state")

            # Filter by counties (GEOID positions 2-5 = county FIPS)
            mask = wac_df['w_geocode'].str[2:5].isin(county_fips_list)
            wac_df = wac_df[mask]

            logger.info(f"  Filtered to {len(wac_df):,} blocks in {len(county_fips_list)} selected counties")

            all_data.append(wac_df)

        except Exception as e:
            error_msg = f"Failed to download LODES data for {state_abbr.upper()}: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    # Combine all states
    if not all_data:
        raise ValueError("No LODES data downloaded for any state")

    logger.info(f"\nCombining data from {len(all_data)} state(s)...")
    combined = pd.concat(all_data, ignore_index=True)

    logger.info(f"Total blocks across all states: {len(combined):,}")
    logger.info(f"Available columns: {list(combined.columns)}")

    # Rename columns to match our schema
    # WAC data uses 'w_geocode' for work location geocode and 'w_lon'/'w_lat' for coordinates
    rename_map = {
        'w_geocode': 'geoid',
        'C000': 'n_employees',
        'w_lon': 'lon',
        'w_lat': 'lat'
    }

    combined = combined.rename(columns=rename_map)

    # Freight-relevant employment, computed *before* the CNS* columns are
    # dropped below. They arrive in this same download at no extra cost, and
    # discarding them was what forced the freight model to use total
    # employment as its attractor. See FREIGHT_SECTORS.
    combined['n_employees_freight'] = freight_employment(combined)

    # Keep only needed columns
    work_locs = combined[['geoid', 'n_employees', 'n_employees_freight',
                          'lat', 'lon']].copy()

    # Convert to appropriate types
    work_locs['geoid'] = work_locs['geoid'].astype(str)
    work_locs['n_employees'] = work_locs['n_employees'].fillna(0).astype(int)
    work_locs['n_employees_freight'] = (
        work_locs['n_employees_freight'].fillna(0).round().astype(int))
    work_locs['lat'] = work_locs['lat'].astype(float)
    work_locs['lon'] = work_locs['lon'].astype(float)

    # Extract state_fips and county_fips from geoid for efficient querying
    # GEOID structure (15 digits): SSCCCTTTTTTBBBB
    # - SS = state FIPS (positions 0-1)
    # - CCC = county FIPS (positions 2-4)
    work_locs['state_fips'] = work_locs['geoid'].str[0:2]
    work_locs['county_fips'] = work_locs['geoid'].str[2:5]

    # Log summary statistics
    logger.info(f"\nSummary Statistics:")
    logger.info(f"  Total blocks: {len(work_locs):,}")
    logger.info(f"  Total jobs: {work_locs['n_employees'].sum():,}")
    logger.info(f"  Avg jobs per block: {work_locs['n_employees'].mean():.1f}")
    logger.info(f"  Blocks with jobs: {(work_locs['n_employees'] > 0).sum():,}")
    logger.info(f"  Blocks with coordinates: {work_locs['lat'].notna().sum():,}")

    # Save to database
    logger.info(f"\nSaving to database...")
    data_dir = config['data']['data_dir']
    db_manager = initialize_tables(data_dir)

    try:
        records = work_locs.to_dict('records')
        db_manager.insert_records(WorkLocation, records)
        logger.info(f"{CHECK_MARK} Successfully saved {len(records):,} work locations to database")
    finally:
        db_manager.close()

    logger.info("=" * 70)
    logger.info("WORK LOCATIONS PROCESSING COMPLETE")
    logger.info("=" * 70)


def ensure_work_locations(config: Dict[str, Any]) -> None:
    """
    Ensure work locations exist in the database for all counties specified in config.

    Checks which counties from config['region']['counties'] have data in the
    work_locations table. If any counties are missing, runs process_work_locations
    for only the missing counties to download LODES WAC data and append
    them to the existing table.

    Args:
        config: Configuration dictionary with 'region.counties' and 'data' sections
    """
    from models.models import WorkLocation

    county_geoids = config['region']['counties']
    if not county_geoids:
        return

    data_dir = config['data']['data_dir']
    db_manager = initialize_tables(data_dir)

    try:
        requested_pairs = {(geoid[:2], geoid[2:5]) for geoid in county_geoids}

        with db_manager.Session() as session:
            from sqlalchemy import tuple_
            existing_rows = session.query(
                WorkLocation.state_fips,
                WorkLocation.county_fips,
            ).filter(
                tuple_(WorkLocation.state_fips, WorkLocation.county_fips).in_(list(requested_pairs))
            ).distinct().all()
            existing_pairs = {(row[0], row[1]) for row in existing_rows}

        missing_pairs = requested_pairs - existing_pairs
        if not missing_pairs:
            logger.info(f"Work locations already exist for all {len(county_geoids)} configured counties")
            return

        missing_geoids = sorted(s + c for s, c in missing_pairs)
        logger.warning(
            f"Work locations missing for {len(missing_pairs)} county(ies): {missing_geoids}. "
            f"Running ETL to process work locations..."
        )
    finally:
        db_manager.close()

    # Build a config copy with only the missing counties to avoid
    # duplicate primary key errors for counties already in the DB.
    import copy
    etl_config = copy.deepcopy(config)
    etl_config['region']['counties'] = missing_geoids
    process_work_locations(etl_config)


def load_work_locations_by_counties(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Load work locations filtered by counties specified in config.

    This function uses indexed state_fips and county_fips columns for efficient querying.
    Only loads data for counties specified in config['region']['counties'].

    Args:
        config: Configuration dictionary with 'region.counties' (list of GEOIDs) and 'data.data_dir' sections

    Returns:
        Dict mapping geoid to {'state_fips': str, 'county_fips': str, 'n_employees': int, 'lat': float, 'lon': float}
        where lat/lon are the workplace point coordinates in EPSG:4326.
    """
    from models.models import WorkLocation

    data_dir = config['data']['data_dir']
    db_manager = initialize_tables(data_dir)

    try:
        # Get county GEOIDs from config
        county_geoids = config['region']['counties']  # List of GEOIDs like ["27003", "27053", ...]

        logger.info(f"Loading work locations for {len(county_geoids)} counties from config...")

        with db_manager.Session() as session:
            if not county_geoids:
                logger.warning("No counties specified in config")
                return {}

            # Split GEOIDs into (state_fips, county_fips) pairs
            county_pairs = [(geoid[:2], geoid[2:5]) for geoid in county_geoids]

            # Use tuple_ for efficient IN clause on indexed columns
            from sqlalchemy import tuple_

            results = session.query(WorkLocation).filter(
                tuple_(WorkLocation.state_fips, WorkLocation.county_fips).in_(county_pairs)
            ).all()

            # Convert to dictionary
            out: Dict[str, Dict[str, Any]] = {}
            for row in results:
                geoid = str(row.geoid).strip()
                # n_employees_freight is None for rows written before the
                # column existed. Left as None rather than coerced to 0 so a
                # consumer can tell "no freight jobs here" from "this row
                # predates sector employment" and fall back accordingly.
                freight_jobs = getattr(row, 'n_employees_freight', None)
                out[geoid] = {
                    'state_fips': str(row.state_fips),
                    'county_fips': str(row.county_fips),
                    'n_employees': int(row.n_employees) if row.n_employees is not None else 0,
                    'n_employees_freight': (int(freight_jobs)
                                            if freight_jobs is not None else None),
                    'lat': float(row.lat) if row.lat is not None else None,
                    'lon': float(row.lon) if row.lon is not None else None
                }

            logger.info(f"Loaded {len(out):,} work location blocks for {len(county_geoids)} counties")
            return out

    finally:
        db_manager.close()

# !!! DELETE LATER !!!
# def load_work_locations(config: Dict[str, Any] = None, data_dir: str = '../data') -> Dict[str, Dict[str, Any]]:
#     """
#     Load work locations table and return mapping geoid -> dict with saved columns.

#     Args:
#         config: Configuration dictionary (optional, uses data_dir if not provided)
#         data_dir: Path to data directory (used if config not provided)

#     Returns:
#         Dict mapping geoid to {'state_fips': str, 'county_fips': str, 'n_employees': int, 'lat': float, 'lon': float}
#         where lat/lon are the workplace point coordinates in EPSG:4326.
#     """
#     if config is not None:
#         data_dir = config['data']['data_dir']

#     db_manager = initialize_tables(data_dir)
#     try:
#         # Read table while connection is open to avoid detached instances
#         df = pd.read_sql_table('work_locations', con=db_manager.engine)

#         out: Dict[str, Dict[str, Any]] = {}
#         for _, row in df.iterrows():
#             geoid = str(row['geoid']).strip()
#             out[geoid] = {
#                 'state_fips': str(row['state_fips']) if pd.notna(row.get('state_fips')) else None,
#                 'county_fips': str(row['county_fips']) if pd.notna(row.get('county_fips')) else None,
#                 'n_employees': int(row['n_employees']) if pd.notna(row.get('n_employees')) else 0,
#                 'lat': float(row['lat']) if pd.notna(row.get('lat')) else None,
#                 'lon': float(row['lon']) if pd.notna(row.get('lon')) else None
#             }
#         return out
#     finally:
#         db_manager.close()
