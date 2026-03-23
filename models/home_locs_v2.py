"""Process home locations from LODES data and save to DuckDB.

This module uses the LEHD LODES (Longitudinal Employer-Household Dynamics
Origin-Destination Employment Statistics) dataset via the pygris package
to get home location data with employee counts and coordinates.

Data source: U.S. Census Bureau LODES
Access method: pygris.data.get_lodes()
"""
from typing import Dict, Any
import pandas as pd
from utils.logger import setup_logger, CHECK_MARK
from models.models import HomeLocation, initialize_tables
from utils.region_utils import RegionHelper

logger = setup_logger(__name__)


def process_home_locations(config: Dict[str, Any]) -> None:
    """
    Download and process home locations using LODES and Census 2020 data via pygris.

    Steps performed:
    1. Initialize RegionHelper to get county FIPS codes from config
    2. Download LODES RAC (Residence Area Characteristics) data for each state
    3. Download Census 2020 block-level population (P1 table) for each state
    4. Filter by selected counties using FIPS codes
    5. Merge LODES and Census data by GEOID
    6. Calculate non_employees = max(0, total_population - n_employees)
    7. Extract geoid, n_employees, non_employees, lat, lon for each census block
    8. Save to DuckDB home_locations table

    Args:
        config: Configuration dictionary with 'region.counties' and 'data.lodes' sections

    Raises:
        ValueError: If county names cannot be found or LODES download fails
    """
    try:
        from pygris.data import get_lodes
        from pygris import blocks
        import platformdirs
    except ImportError:
        raise ImportError(
            "pygris package is required. Install with: pip install pygris"
        )

    logger.info("=" * 70)
    logger.info("PROCESSING HOME LOCATIONS FROM LODES")
    logger.info("=" * 70)

    # Set cache directory to project data folder by monkey-patching platformdirs
    import os
    data_dir = config['data']['data_dir']
    cache_dir = os.path.join(data_dir, 'pygris_cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Override platformdirs.user_cache_dir to use our custom cache directory
    original_cache_dir = platformdirs.user_cache_dir
    platformdirs.user_cache_dir = lambda app_name: cache_dir if app_name == "pygris" else original_cache_dir(app_name)
    logger.info(f"Cache directory: {cache_dir}")

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

    # Download LODES RAC data for each state
    for state_fips, county_fips_list in fips_mapping.items():
        state_abbr = state_abbr_mapping[state_fips]
        logger.info(f"\nProcessing {state_abbr.upper()}:")
        logger.info(f"  Downloading LODES RAC (Residence Area Characteristics) data...")

        try:
            rac_df = get_lodes(
                state=state_abbr,
                year=year,
                lodes_type="rac",  # Residence Area Characteristics
                return_lonlat=True,  # Get lat/lon centroids
                cache=True  # Cache for faster reuse
            )

            logger.info(f"  Downloaded {len(rac_df):,} blocks for entire state")

            # Filter by counties (GEOID positions 2-5 = county FIPS)
            # RAC data uses 'h_geocode' (home geocode), not 'w_geocode' (workplace geocode)
            mask = rac_df['h_geocode'].str[2:5].isin(county_fips_list)
            rac_df = rac_df[mask]

            logger.info(f"  Filtered to {len(rac_df):,} blocks in {len(county_fips_list)} selected counties")

            all_data.append(rac_df)

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
    logger.info(f"Available LODES columns: {list(combined.columns)}")

    # Download Census 2020 block-level population data
    logger.info("\n" + "=" * 70)
    logger.info("FETCHING CENSUS 2020 POPULATION DATA")
    logger.info("=" * 70)

    all_census_data = []

    for state_fips, county_fips_list in fips_mapping.items():
        state_abbr = state_abbr_mapping[state_fips]
        logger.info(f"\nProcessing {state_abbr.upper()}:")
        logger.info(f"  Downloading Census 2020 block-level population...")

        try:
            # Download Census 2020 blocks with population data (P1_001N = Total population)
            census_blocks = blocks(
                state=state_abbr,
                year=2020,
                cache=True
            )

            logger.info(f"  Downloaded {len(census_blocks):,} blocks for entire state")

            # Extract GEOID and population
            # Census blocks have GEOID20 field (15-digit)
            if 'GEOID20' not in census_blocks.columns:
                logger.error(f"  GEOID20 column not found. Available columns: {list(census_blocks.columns)}")
                raise ValueError(f"GEOID20 column not found in Census blocks for {state_abbr}")

            # Population is typically in POP20 or similar field
            pop_field = None
            for field in ['POP20', 'P001001', 'P1_001N', 'population']:
                if field in census_blocks.columns:
                    pop_field = field
                    break

            if pop_field is None:
                logger.error(f"  Population field not found. Available columns: {list(census_blocks.columns)}")
                raise ValueError(f"Population field not found in Census blocks for {state_abbr}")

            logger.info(f"  Using population field: {pop_field}")

            # Filter by counties
            mask = census_blocks['GEOID20'].str[2:5].isin(county_fips_list)
            census_blocks = census_blocks[mask]

            logger.info(f"  Filtered to {len(census_blocks):,} blocks in {len(county_fips_list)} selected counties")

            # Keep only GEOID and population
            census_subset = census_blocks[['GEOID20', pop_field]].copy()
            census_subset = census_subset.rename(columns={'GEOID20': 'geoid', pop_field: 'total_population'})

            all_census_data.append(census_subset)

        except Exception as e:
            error_msg = f"Failed to download Census 2020 data for {state_abbr.upper()}: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    # Combine all Census data
    logger.info(f"\nCombining Census data from {len(all_census_data)} state(s)...")
    census_combined = pd.concat(all_census_data, ignore_index=True)

    logger.info(f"Total Census blocks: {len(census_combined):,}")
    logger.info(f"Census columns: {list(census_combined.columns)}")

    # Rename columns to match our schema
    # RAC data uses 'h_geocode' for home location geocode and 'h_lon'/'h_lat' for coordinates
    rename_map = {
        'h_geocode': 'geoid',
        'C000': 'n_employees',
        'h_lon': 'lon',
        'h_lat': 'lat'
    }

    combined = combined.rename(columns=rename_map)

    # Keep only needed columns from LODES
    home_locs = combined[['geoid', 'n_employees', 'lat', 'lon']].copy()

    # Convert to appropriate types
    home_locs['geoid'] = home_locs['geoid'].astype(str)
    home_locs['n_employees'] = home_locs['n_employees'].fillna(0).astype(int)
    home_locs['lat'] = home_locs['lat'].astype(float)
    home_locs['lon'] = home_locs['lon'].astype(float)

    # Merge with Census 2020 population data
    logger.info("\n" + "=" * 70)
    logger.info("MERGING LODES AND CENSUS DATA")
    logger.info("=" * 70)

    # Ensure Census geoid is string for merging
    census_combined['geoid'] = census_combined['geoid'].astype(str)
    census_combined['total_population'] = census_combined['total_population'].fillna(0).astype(int)

    # Merge on geoid (15-digit block ID)
    home_locs = home_locs.merge(census_combined, on='geoid', how='left')

    # Log matching statistics
    matched_count = home_locs['total_population'].notna().sum()
    unmatched_count = home_locs['total_population'].isna().sum()
    logger.info(f"  Matched blocks: {matched_count:,}")
    logger.info(f"  Unmatched blocks (no Census data): {unmatched_count:,}")

    if unmatched_count > 0:
        logger.warning(f"  {unmatched_count} blocks from LODES have no matching Census 2020 data")
        # Log a few examples of unmatched blocks
        unmatched_geoids = home_locs[home_locs['total_population'].isna()]['geoid'].head(10).tolist()
        logger.warning(f"  Example unmatched GEOIDs: {unmatched_geoids}")

    # Fill missing total_population with 0
    home_locs['total_population'] = home_locs['total_population'].fillna(0).astype(int)

    # Calculate non_employees = max(0, total_population - n_employees)
    home_locs['non_employees'] = (home_locs['total_population'] - home_locs['n_employees']).clip(lower=0).astype(int)

    logger.info(f"\nNon-Employee Calculation:")
    logger.info(f"  Total population sum: {home_locs['total_population'].sum():,}")
    logger.info(f"  Total employees sum: {home_locs['n_employees'].sum():,}")
    logger.info(f"  Total non-employees sum: {home_locs['non_employees'].sum():,}")
    logger.info(f"  Blocks with non-employees > 0: {(home_locs['non_employees'] > 0).sum():,}")

    # Check for blocks where employees > population (data inconsistency)
    inconsistent_blocks = (home_locs['n_employees'] > home_locs['total_population']) & (home_locs['total_population'] > 0)
    if inconsistent_blocks.any():
        inconsistent_count = inconsistent_blocks.sum()
        logger.warning(f"  Found {inconsistent_count} blocks where n_employees > total_population")
        logger.warning(f"  These blocks will have non_employees = 0 (capped at zero)")

        # Show statistics about the inconsistent blocks
        inconsistent_df = home_locs[inconsistent_blocks]
        total_pop_lost = inconsistent_df['total_population'].sum()
        total_employees_in_blocks = inconsistent_df['n_employees'].sum()
        logger.info(f"  Inconsistent blocks stats:")
        logger.info(f"    - Total population in these blocks: {total_pop_lost:,}")
        logger.info(f"    - Total employees in these blocks: {total_employees_in_blocks:,}")
        logger.info(f"    - Ratio: {total_employees_in_blocks/max(total_pop_lost,1):.2f} employees per resident")
        logger.info(f"    - Avg population per block: {inconsistent_df['total_population'].mean():.1f}")
        logger.info(f"    - Avg employees per block: {inconsistent_df['n_employees'].mean():.1f}")
        logger.info(f"  This suggests these are primarily commercial/industrial zones with few residents")

    # Drop temporary total_population column (we don't store it in the database)
    home_locs = home_locs.drop(columns=['total_population'])

    # Extract state_fips and county_fips from geoid for efficient querying
    # GEOID structure (15 digits): SSCCCTTTTTTBBBB
    # - SS = state FIPS (positions 0-1)
    # - CCC = county FIPS (positions 2-4)
    home_locs['state_fips'] = home_locs['geoid'].str[0:2]
    home_locs['county_fips'] = home_locs['geoid'].str[2:5]

    # Log summary statistics
    logger.info(f"\n" + "=" * 70)
    logger.info(f"FINAL SUMMARY STATISTICS")
    logger.info(f"=" * 70)
    logger.info(f"  Total blocks: {len(home_locs):,}")
    logger.info(f"  Total employees: {home_locs['n_employees'].sum():,}")
    logger.info(f"  Total non-employees: {home_locs['non_employees'].sum():,}")
    logger.info(f"  Avg employees per block: {home_locs['n_employees'].mean():.1f}")
    logger.info(f"  Avg non-employees per block: {home_locs['non_employees'].mean():.1f}")
    logger.info(f"  Blocks with employees: {(home_locs['n_employees'] > 0).sum():,}")
    logger.info(f"  Blocks with non-employees: {(home_locs['non_employees'] > 0).sum():,}")
    logger.info(f"  Blocks with coordinates: {home_locs['lat'].notna().sum():,}")

    # Save to database
    logger.info(f"\nSaving to database...")
    data_dir = config['data']['data_dir']
    db_manager = initialize_tables(data_dir)

    try:
        records = home_locs.to_dict('records')
        db_manager.insert_records(HomeLocation, records)
        logger.info(f"{CHECK_MARK} Successfully saved {len(records):,} home locations to database")
    finally:
        db_manager.close()

    logger.info("=" * 70)
    logger.info("HOME LOCATIONS PROCESSING COMPLETE")
    logger.info("=" * 70)


def ensure_home_locations(config: Dict[str, Any]) -> None:
    """
    Ensure home locations exist in the database for all counties specified in config.

    Checks which counties from config['region']['counties'] have data in the
    home_locations table. If any counties are missing, runs process_home_locations
    for only the missing counties to download LODES + Census data and append
    them to the existing table.

    Args:
        config: Configuration dictionary with 'region.counties' and 'data' sections
    """
    from models.models import HomeLocation

    county_geoids = config['region']['counties']
    if not county_geoids:
        return

    data_dir = config['data']['data_dir']
    db_manager = initialize_tables(data_dir)

    try:
        requested_pairs = {(geoid[:2], geoid[2:5]) for geoid in county_geoids}

        with db_manager.Session() as session:
            # Check which (state_fips, county_fips) pairs exist in the DB
            from sqlalchemy import tuple_
            existing_rows = session.query(
                HomeLocation.state_fips,
                HomeLocation.county_fips,
            ).filter(
                tuple_(HomeLocation.state_fips, HomeLocation.county_fips).in_(list(requested_pairs))
            ).distinct().all()
            existing_pairs = {(row[0], row[1]) for row in existing_rows}

        missing_pairs = requested_pairs - existing_pairs
        if not missing_pairs:
            logger.info(f"Home locations already exist for all {len(county_geoids)} configured counties")
            return

        missing_geoids = sorted(s + c for s, c in missing_pairs)
        logger.warning(
            f"Home locations missing for {len(missing_pairs)} county(ies): {missing_geoids}. "
            f"Running ETL to process home locations..."
        )
    finally:
        db_manager.close()

    # Build a config copy with only the missing counties to avoid
    # duplicate primary key errors for counties already in the DB.
    import copy
    etl_config = copy.deepcopy(config)
    etl_config['region']['counties'] = missing_geoids
    process_home_locations(etl_config)


def load_home_locations_by_counties(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Load home locations filtered by counties specified in config.

    This function uses indexed state_fips and county_fips columns for efficient querying.
    Only loads data for counties specified in config['region']['counties'].

    Args:
        config: Configuration dictionary with 'region.counties' (list of GEOIDs) and 'data.data_dir' sections

    Returns:
        Dict mapping geoid to {'state_fips': str, 'county_fips': str, 'n_employees': int, 'non_employees': int, 'lat': float, 'lon': float}
        where lat/lon are the home point coordinates in EPSG:4326.
    """
    from models.models import HomeLocation

    data_dir = config['data']['data_dir']
    db_manager = initialize_tables(data_dir)

    try:
        # Get county GEOIDs from config
        county_geoids = config['region']['counties']  # List of GEOIDs like ["27003", "27053", ...]

        logger.info(f"Loading home locations for {len(county_geoids)} counties from config...")

        with db_manager.Session() as session:
            if not county_geoids:
                logger.warning("No counties specified in config")
                return {}

            # Split GEOIDs into (state_fips, county_fips) pairs
            county_pairs = [(geoid[:2], geoid[2:5]) for geoid in county_geoids]

            # Use tuple_ for efficient IN clause on indexed columns
            from sqlalchemy import tuple_

            results = session.query(HomeLocation).filter(
                tuple_(HomeLocation.state_fips, HomeLocation.county_fips).in_(county_pairs)
            ).all()

            # Convert to dictionary
            out: Dict[str, Dict[str, Any]] = {}
            for row in results:
                geoid = str(row.geoid).strip()
                out[geoid] = {
                    'state_fips': str(row.state_fips),
                    'county_fips': str(row.county_fips),
                    'n_employees': int(row.n_employees) if row.n_employees is not None else 0,
                    'non_employees': int(row.non_employees) if row.non_employees is not None else 0,
                    'lat': float(row.lat) if row.lat is not None else None,
                    'lon': float(row.lon) if row.lon is not None else None
                }

            logger.info(f"Loaded {len(out):,} home location blocks for {len(county_geoids)} counties")
            return out

    finally:
        db_manager.close()

# USE load_home_locations_by_counties INSTEAD
# def load_home_locations(config: Dict[str, Any] = None, data_dir: str = '../data') -> Dict[str, Dict[str, Any]]:
#     """
#     Load home locations table and return mapping geoid -> dict with saved columns.

#     Args:
#         config: Configuration dictionary (optional, uses data_dir if not provided)
#         data_dir: Path to data directory (used if config not provided)

#     Returns:
#         Dict mapping geoid to {'state_fips': str, 'county_fips': str, 'n_employees': int, 'lat': float, 'lon': float}
#         where lat/lon are the home point coordinates in EPSG:4326.
#     """
#     if config is not None:
#         data_dir = config['data']['data_dir']

#     db_manager = initialize_tables(data_dir)
#     try:
#         # Read table while connection is open to avoid detached instances
#         df = pd.read_sql_table('home_locations', con=db_manager.engine)

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
