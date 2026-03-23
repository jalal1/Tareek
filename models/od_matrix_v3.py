import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, List
from utils.logger import setup_logger
from data_sources.base_survey_trip import BaseSurveyTrip

logger = setup_logger(__name__)

def euclidean_distance_matrix(lat1, lon1, lat2, lon2):
    """Fast approximation for distances < 100km"""
    # Convert to km using simple projection (valid for small regions)
    lat1_km = lat1 * 111.32  # 1 degree latitude ≈ 111.32 km
    lon1_km = lon1 * 111.32 * np.cos(np.radians(lat1))
    
    lat2_km = lat2 * 111.32
    lon2_km = lon2 * 111.32 * np.cos(np.radians(lat2))
    
    dlat = lat2_km[:, np.newaxis] - lat1_km
    dlon = lon2_km[:, np.newaxis] - lon1_km
    
    return np.sqrt(dlat**2 + dlon**2)
   
def create_survey_od_matrix(df: pd.DataFrame):
    # Ensure all values in the columns are strings
    o_col = BaseSurveyTrip.ORIGIN_LOC
    d_col = BaseSurveyTrip.DESTINATION_LOC
    df[o_col] = df[o_col].astype(str).str.strip()
    df[d_col] = df[d_col].astype(str).str.strip()

    # Create the OD matrix using pd.crosstab
    od_matrix = pd.crosstab(df[o_col], df[d_col])

    return od_matrix

def create_survey_od_matrix_using_trip_weight(df: pd.DataFrame):
    # Ensure all values in the columns are strings
    o_col = BaseSurveyTrip.ORIGIN_LOC
    d_col = BaseSurveyTrip.DESTINATION_LOC
    w_col = BaseSurveyTrip.TRIP_WEIGHT
    df[o_col] = df[o_col].astype(str).str.strip()
    df[d_col] = df[d_col].astype(str).str.strip()

    # Create the OD matrix using pd.crosstab with trip_weight values
    od_matrix = pd.crosstab(
        df[o_col],
        df[d_col],
        values=df[w_col],
        aggfunc='sum'
    )

    # Fill NaN values with 0
    od_matrix = od_matrix.fillna(0)

    return od_matrix

def aggregate_blocks_to_blockgroups(od_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate an OD matrix from blocks (15 digits) to block groups (12 digits).
    
    Args:
        od_matrix (pd.DataFrame): OD matrix with 15-digit FIPS codes as index/columns.
        
    Returns:
        pd.DataFrame: Aggregated OD matrix with 12-digit block group FIPS codes.
    """
    # Extract first 12 digits from index and columns to get block group codes
    bg_index = od_matrix.index.astype(str).str[:12]
    bg_columns = od_matrix.columns.astype(str).str[:12]
    
    # Create a copy with block group codes
    temp_df = od_matrix.copy()
    temp_df.index = bg_index
    temp_df.columns = bg_columns
    
    # Aggregate: sum across rows (origins) then across columns (destinations)
    agg_matrix = temp_df.groupby(level=0).sum()  # Sum rows with same origin BG
    agg_matrix = agg_matrix.groupby(level=0, axis=1).sum()  # Sum columns with same destination BG
    
    return agg_matrix


def blend_survey_od_matrices(survey_ods: Dict[str, pd.DataFrame],
                             weights: Dict[str, float]) -> pd.DataFrame:
    """Combine multiple per-source survey OD matrices into one via weighted sum.

    Each matrix may cover a different set of block groups.  All matrices
    are aligned to the union of all block groups (missing cells filled
    with 0) before the weighted sum is computed.

    Args:
        survey_ods: {source_key: OD DataFrame} — one per location-capable
                    survey.  Index and columns are block-group IDs.
        weights:    {source_key: float} — raw weights from config.  Only
                    keys present in *survey_ods* are used.  Normalised
                    internally so the result is on the same scale as each
                    input matrix.

    Returns:
        Single blended survey OD DataFrame with the union of all block
        groups as index/columns.
    """
    if not survey_ods:
        raise ValueError("survey_ods must contain at least one OD matrix")

    names = list(survey_ods.keys())
    if len(names) == 1:
        logger.info("blend_survey_od_matrices: single source, returning as-is")
        return survey_ods[names[0]]

    # Normalise weights across the provided sources
    raw = np.array([weights[n] for n in names], dtype=float)
    total = raw.sum()
    if total <= 0:
        raise ValueError("Sum of OD blend weights must be positive")
    norm_weights = raw / total

    # Build union of all block group IDs
    all_rows: set = set()
    all_cols: set = set()
    for od in survey_ods.values():
        all_rows.update(od.index.astype(str))
        all_cols.update(od.columns.astype(str))

    all_rows_sorted = sorted(all_rows)
    all_cols_sorted = sorted(all_cols)

    logger.info(
        f"blend_survey_od_matrices: {len(names)} sources, "
        f"union size {len(all_rows_sorted)} origins × {len(all_cols_sorted)} destinations"
    )

    # Weighted sum on aligned matrices
    blended = pd.DataFrame(0.0, index=all_rows_sorted, columns=all_cols_sorted)
    for name, w in zip(names, norm_weights):
        od = survey_ods[name]
        aligned = od.reindex(index=all_rows_sorted, columns=all_cols_sorted,
                             fill_value=0)
        blended += w * aligned
        logger.info(f"  {name}: weight={w:.3f}, shape={od.shape}")

    return blended


def combine_od_matrices(survey_od_matrix: pd.DataFrame,
                       local_od_matrix: pd.DataFrame, 
                       alpha: float,
                       scale_to_total: int | None = None) -> pd.DataFrame:
    """
    Combine two OD matrices at block group level with weighted average,
    then scale to match a target total (e.g., census data).
    
    Uses the full extent of the local matrix. For areas with both survey and local data,
    applies weighted average. For areas with only local data, uses local values directly.
    Then scales the combined matrix to match the specified total while preserving patterns.
    
    Args:
        survey_od_matrix (pd.DataFrame): OD matrix from survey data (block group level).
        local_od_matrix (pd.DataFrame): OD matrix from local gravity model (block group level).
        alpha (float): Weight for survey matrix (0 <= alpha <= 1). 
                      1-alpha is the weight for local matrix.
        scale_to_total (int, optional): Total trips to scale combined matrix to.
                      If None (default), scales to the sum of local_od_matrix.
        
    Returns:
        pd.DataFrame: Combined OD matrix at block group level, scaled to target total.
    """
    
    logger.info("=" * 70)
    logger.info("COMBINING OD MATRICES")
    logger.info("=" * 70)
    
    # Validate alpha
    if not (0 <= alpha <= 1):
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
    
    logger.info(f"Survey matrix shape: {survey_od_matrix.shape}")
    logger.info(f"Local matrix shape: {local_od_matrix.shape}")
    logger.info(f"Alpha (survey weight): {alpha:.2f}")
    logger.info(f"Local weight: {1-alpha:.2f}")

    # Align survey to local's structure (fill missing survey values with 0)
    logger.info("Aligning survey to local matrix structure...")
    survey_aligned = survey_od_matrix.reindex(index=local_od_matrix.index, 
                                              columns=local_od_matrix.columns, 
                                              fill_value=0)
    
    # Combine with conditional weighting:
    # - Where survey is non-zero: apply weighted average with alpha
    # - Where survey is zero: use local value directly
    logger.info("Combining matrices (alpha blend where survey exists, local only where survey is empty)...")
    survey_mask = survey_aligned != 0
    combined = pd.DataFrame(0.0, index=local_od_matrix.index, columns=local_od_matrix.columns)
    
    # Apply alpha weighting where survey has data
    combined[survey_mask] = alpha * survey_aligned[survey_mask] + (1 - alpha) * local_od_matrix[survey_mask]
    
    # Use local directly where survey is empty
    combined[~survey_mask] = local_od_matrix[~survey_mask]
    
    # Determine scaling target
    if scale_to_total is None:
        scale_to_total = int(local_od_matrix.sum().sum())
        logger.info(f"No scaling target specified. Using local matrix total: {scale_to_total:,.0f}")
    else:
        logger.info(f"Scaling to specified total: {scale_to_total:,.0f}")
    
    # Scale combined matrix to target total
    combined_total = combined.sum().sum()
    if combined_total > 0:
        scale_factor = scale_to_total / combined_total
        logger.info(f"Combined matrix total before scaling: {combined_total:,.0f}")
        logger.info(f"Scale factor: {scale_factor:.6f}")
        combined = combined * scale_factor
    else:
        raise ValueError("Combined matrix sum is zero, cannot scale")
    
    # Robust rounding that guarantees target total (vectorized for speed)
    # check Hamilton's method
    logger.info("Applying robust rounding to guarantee target total...")
    
    # Step 1: Floor all values
    combined_floor = np.floor(combined).astype(int)
    
    # Step 2: Calculate fractional parts
    fractional = combined - combined_floor
    
    # Step 3: Calculate rounding remainder
    floored_total = combined_floor.sum().sum()
    remainder = scale_to_total - floored_total
    
    logger.info(f"Total after flooring: {floored_total:,.0f}")
    logger.info(f"Remainder to distribute: {remainder:,.0f}")
    
    # Step 4: Vectorized rounding using argsort on flattened values
    combined_rounded = combined_floor.copy()
    frac_flat = fractional.values.flatten()
    
    if remainder > 0:
        # Get indices of largest fractional parts
        top_indices = np.argsort(-frac_flat)[:remainder]
        combined_rounded.values.flat[top_indices] += 1
    elif remainder < 0:
        # Get indices of smallest fractional parts (only positive cells)
        mask = combined_rounded.values.flatten() > 0
        candidates = np.where(mask)[0]
        frac_candidates = frac_flat[candidates]
        smallest_idx = candidates[np.argsort(frac_candidates)[:abs(remainder)]]
        combined_rounded.values.flat[smallest_idx] -= 1
    
    # Verify
    logger.info("=" * 70)
    logger.info("VERIFICATION")
    logger.info("=" * 70)
    logger.info(f"Survey total trips: {survey_od_matrix.sum().sum():,.0f}")
    logger.info(f"Local total trips: {local_od_matrix.sum().sum():,.0f}")
    logger.info(f"Combined total trips (before scaling): {combined_total:,.0f}")
    logger.info(f"Combined total trips (after scaling & rounding): {combined_rounded.sum().sum():,.0f}")
    logger.info(f"Target total: {scale_to_total:,.0f}")
    logger.info(f"Difference: {combined_rounded.sum().sum() - scale_to_total:,.0f}")
    logger.info(f"Combined matrix shape: {combined_rounded.shape}")
    
    return combined_rounded

def create_gravity_model(work_locations_dict, home_locs_dict, beta, max_iterations=50, convergence_threshold=1e-4):
    """
    Create OD matrix using gravity model with IPF (Iterative Proportional Fitting).
    
    Parameters:
    - work_locations_dict: Dictionary with work block IDs as keys, contains 'n_employees' and 'centroid'
    - home_locs_dict: Dictionary with home block IDs as keys, contains 'n_employees' and 'centroid'
    - beta: Distance decay parameter for friction factors
    - max_iterations: Maximum IPF iterations
    - convergence_threshold: Relative difference threshold for convergence
    
    Returns:
    - od_matrix: 2D numpy array (rows=home blocks, cols=work blocks)
    - home_geoids: List of home block IDs (row order)
    - work_geoids: List of work block IDs (column order)
    """
    
    # Input validation
    if beta <= 0:
        raise ValueError("Beta parameter must be positive")
    if not work_locations_dict or not home_locs_dict:
        raise ValueError("Empty input data")
    
    # Get sorted geoid lists
    home_geoids = sorted(home_locs_dict.keys())
    work_geoids = sorted(work_locations_dict.keys())
    
    logger.info("=" * 70)
    logger.info("GRAVITY MODEL WITH IPF")
    logger.info("=" * 70)
    logger.info(f"Number of home blocks: {len(home_geoids)}")
    logger.info(f"Number of work blocks: {len(work_geoids)}")
    
    # Extract constraints (workers at home, jobs at work)
    Oi = np.array([home_locs_dict[geoid]['n_employees'] 
                   for geoid in home_geoids], dtype=np.float64)
    Dj = np.array([work_locations_dict[geoid]['n_employees'] 
                   for geoid in work_geoids], dtype=np.float64)
    
    logger.info(f"Total workers (origin): {Oi.sum():.0f}")
    logger.info(f"Total jobs (destination): {Dj.sum():.0f}")

    # Normalize if totals don't match
    if abs(Oi.sum() - Dj.sum()) > 0.01:
        logger.warning(f"Totals differ. Normalizing destination to match origin.")
        Dj = Dj * (Oi.sum() / Dj.sum())
    
    # Extract coordinates
    logger.info("Extracting coordinates...")
    home_coords = np.array([home_locs_dict[geoid]['centroid'] 
                            for geoid in home_geoids], dtype=np.float64)
    work_coords = np.array([work_locations_dict[geoid]['centroid'] 
                            for geoid in work_geoids], dtype=np.float64)
    
    # Calculate distance matrix
    logger.info("Calculating distances...")
    distances = cdist(home_coords, work_coords, metric='euclidean')
    distances = np.where(distances == 0, 0.1, distances)  # Avoid zero distances
    
    # Calculate friction factors: f_ij = distance^(-beta)
    logger.info("Calculating friction factors...")
    friction_factors = np.power(distances, -beta)
    
    # Initialize OD matrix with gravity model
    od_matrix = friction_factors.copy()
    
    # IPF iterations
    logger.info(f"Running IPF iterations (max {max_iterations})...")
    
    for iteration in range(max_iterations):
        # Row scaling: adjust to match origin constraints
        row_sums = od_matrix.sum(axis=1, keepdims=True)
        row_factors = np.divide(
            Oi.reshape(-1, 1),
            row_sums,
            out=np.ones_like(row_sums),
            where=row_sums != 0
        )
        od_matrix = od_matrix * row_factors
        
        # Column scaling: adjust to match destination constraints
        col_sums = od_matrix.sum(axis=0, keepdims=True)
        col_factors = np.divide(
            Dj.reshape(1, -1),
            col_sums,
            out=np.ones_like(col_sums),
            where=col_sums != 0
        )
        od_matrix = od_matrix * col_factors
        
        # Check convergence
        row_sums = od_matrix.sum(axis=1)
        col_sums = od_matrix.sum(axis=0)
        
        row_rel_diff = np.abs(row_sums - Oi) / (Oi + 1e-10)
        col_rel_diff = np.abs(col_sums - Dj) / (Dj + 1e-10)
        
        max_rel_diff = max(np.max(row_rel_diff), np.max(col_rel_diff))
        
        if (iteration + 1) % 10 == 0 or max_rel_diff < convergence_threshold:
            logger.info(f"  Iteration {iteration + 1}: Max relative difference = {max_rel_diff:.2e}")

        if max_rel_diff < convergence_threshold:
            logger.info(f"Converged after {iteration + 1} iterations")
            break
    
    # Verify constraints
    logger.info("=" * 70)
    logger.info("VERIFICATION")
    logger.info("=" * 70)

    final_row_sums = od_matrix.sum(axis=1)
    final_col_sums = od_matrix.sum(axis=0)

    max_row_error = np.abs(final_row_sums - Oi).max()
    max_col_error = np.abs(final_col_sums - Dj).max()

    logger.info(f"Origin total: {Oi.sum():.0f} | Matrix row sum: {final_row_sums.sum():.0f}")
    logger.info(f"Destination total: {Dj.sum():.0f} | Matrix col sum: {final_col_sums.sum():.0f}")
    logger.info(f"Max row constraint error: {max_row_error:.2e}")
    logger.info(f"Max col constraint error: {max_col_error:.2e}")
    
    return od_matrix, home_geoids, work_geoids


def create_local_od_matrix(work_locs_dict, home_locs_dict, beta=1.5, max_iterations=200, 
                          convergence_threshold=0.03):
    """
    Create origin-destination matrix using gravity model with IPF.
    
    Parameters:
    - work_locs_dict: Dictionary with workplace locations and n_employees
    - home_locs_dict: Dictionary with home locations, n_employees, and centroid
    - beta: Distance decay parameter (default 1.5 for metropolitan commuting)
    - max_iterations: Maximum iterations for IPF algorithm
    - convergence_threshold: Convergence criterion for IPF
    
    Returns:
    - result dict with:
        - 'od_matrix': 2D numpy array (rows=home blocks, cols=work blocks)
        - 'home_geoids': List of home block geoids (row order)
        - 'work_geoids': List of work block geoids (column order)
        - 'total_workers': Total number of workers
        - 'total_jobs': Total number of jobs
        - 'total_trips': Total trips in OD matrix
    """
    
    # Call the gravity model function
    od_matrix, home_geoids, work_geoids = create_gravity_model(
        work_locs_dict, 
        home_locs_dict, 
        beta=beta, 
        max_iterations=max_iterations, 
        convergence_threshold=convergence_threshold
    )
    
    # Calculate totals
    total_trips = int(od_matrix.sum())
    
    # Extract constraint totals
    total_workers = sum(home_locs_dict[g]['n_employees'] for g in home_geoids)
    total_jobs = sum(work_locs_dict[g]['n_employees'] for g in work_geoids)
    
    # Convert to DataFrame with geoid indices for easier access
    import pandas as pd
    od_df = pd.DataFrame(
        od_matrix,
        index=home_geoids,
        columns=work_geoids
    )
    
    result = {
        'od_matrix': od_df,  # DataFrame with home_geoids as rows, work_geoids as columns
        'home_geoids': home_geoids,
        'work_geoids': work_geoids,
        'total_workers': total_workers,
        'total_jobs': total_jobs,
        'total_trips': total_trips,
        'n_home_blocks': len(home_geoids),
        'n_work_blocks': len(work_geoids)
    }
    
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total workers: {result['total_workers']:,.0f}")
    logger.info(f"Total jobs: {result['total_jobs']:,.0f}")
    logger.info(f"Total trips: {result['total_trips']:,.0f}")
    logger.info(f"Home blocks: {result['n_home_blocks']}")
    logger.info(f"Work blocks: {result['n_work_blocks']}")
    logger.info(f"OD Matrix shape: {od_df.shape}")
    logger.info(f"Sample OD Matrix (first 5 rows/cols):")
    logger.info(f"\n{od_df.iloc[:5, :5]}")

    return result


def _allocate_samples(blocks_dict: Dict, total_samples: int,
                     emp_key: str, total_emp: int) -> Dict[str, int]:
    """
    Allocate samples to blocks using floor, then distribute remainder by largest quotas.
    Fast O(n) approach using Hamilton's method for apportionment.

    Args:
        blocks_dict: Dict of {block_id: block_data}
        total_samples: Total samples to allocate
        emp_key: Key in block_data for employee count
        total_emp: Total employees in all blocks

    Returns:
        Dict of {block_id: sample_count}
    """
    allocations = {}
    remainder = total_samples

    # Calculate floor allocations and track remainders
    block_remainders = []

    for bid, data in blocks_dict.items():
        emp = data[emp_key]
        if emp <= 0:
            allocations[bid] = 0
            continue

        exact_quota = (emp / total_emp) * total_samples
        floor_quota = int(exact_quota)

        allocations[bid] = floor_quota
        remainder -= floor_quota

        frac = exact_quota - floor_quota
        if frac > 0:
            block_remainders.append((frac, bid))

    # Distribute remainder to blocks with largest fractional parts
    block_remainders.sort(reverse=True)
    for i in range(remainder):
        if i < len(block_remainders):
            _, bid = block_remainders[i]
            allocations[bid] += 1

    return allocations


def generate_samples(bg_origin: str, bg_destination: str, num_trips: int,
                    blockid2homelocs: Dict, blockid2worklocs: Dict) -> Dict:
    """
    Generate home and work location samples for trips between two block groups.

    Samples blocks proportionally by n_employees within each block group,
    then returns block coordinates as (lon, lat) tuples.

    Note: Sampling is proportional and repeatable - the same block can be sampled
    across multiple OD pairs. The n_employees values represent static population
    distributions, not consumable capacity.

    Args:
        bg_origin: Origin block group ID (12 chars)
        bg_destination: Destination block group ID (12 chars)
        num_trips: Number of trips to generate
        blockid2homelocs: Dict mapping block IDs to home location info
                         Must have keys: 'n_employees', 'lat', 'lon'
        blockid2worklocs: Dict mapping block IDs to work location info
                         Must have keys: 'n_employees', 'lat', 'lon'

    Returns:
        {
            'home_locations': [(lon, lat), ...],  # List of (longitude, latitude) tuples
            'work_locations': [(lon, lat), ...]
        }

    Example:
        >>> samples = generate_samples('270030501081', '270030501082', 100,
        ...                            blockid2homelocs, blockid2worklocs)
        >>> len(samples['home_locations'])
        100
        >>> len(samples['work_locations'])
        100
    """
    # Get all blocks in origin and destination BGs
    origin_blocks = {bid: data for bid, data in blockid2homelocs.items()
                     if bid[:12] == bg_origin}
    dest_blocks = {bid: data for bid, data in blockid2worklocs.items()
                   if bid[:12] == bg_destination}

    if not origin_blocks or not dest_blocks:
        logger.warning(f"No blocks found for BG pair {bg_origin} -> {bg_destination}")
        return {'home_locations': [], 'work_locations': []}

    # Calculate total employees for each BG
    total_origin_emp = sum(b['n_employees'] for b in origin_blocks.values() if b['n_employees'] > 0)
    total_dest_emp = sum(b['n_employees'] for b in dest_blocks.values() if b['n_employees'] > 0)

    if total_origin_emp == 0 or total_dest_emp == 0:
        logger.warning(f"Zero employees in BG pair {bg_origin} -> {bg_destination}: "
                      f"origin={total_origin_emp}, dest={total_dest_emp}")
        return {'home_locations': [], 'work_locations': []}

    # Calculate samples per block using Hamilton's method
    origin_samples = _allocate_samples(origin_blocks, num_trips, 'n_employees', total_origin_emp)
    dest_samples = _allocate_samples(dest_blocks, num_trips, 'n_employees', total_dest_emp)

    # Generate location lists
    home_locs = []
    work_locs = []

    for bid, count in origin_samples.items():
        if count == 0:
            continue

        block_data = blockid2homelocs[bid]
        if 'lat' in block_data and 'lon' in block_data and block_data['lat'] is not None and block_data['lon'] is not None:
            lat = block_data['lat']
            lon = block_data['lon']

            # Add spatial jitter to avoid duplicate coordinates
            # Jitter in degrees: ~0.0005 degrees ≈ 50m at this latitude
            for _ in range(count):
                jitter_lat = np.random.normal(0, 0.0005)  # ~50m std dev
                jitter_lon = np.random.normal(0, 0.0005)
                jittered_point = (lon + jitter_lon, lat + jitter_lat)  # (lon, lat) order
                home_locs.append(jittered_point)
        else:
            logger.warning(f"Missing lat/lon for home block {bid}")

    for bid, count in dest_samples.items():
        if count == 0:
            continue

        block_data = blockid2worklocs[bid]
        if 'lat' in block_data and 'lon' in block_data and block_data['lat'] is not None and block_data['lon'] is not None:
            lat = block_data['lat']
            lon = block_data['lon']

            # Add spatial jitter to avoid duplicate coordinates
            # Jitter in degrees: ~0.0005 degrees ≈ 50m at this latitude
            for _ in range(count):
                jitter_lat = np.random.normal(0, 0.0005)  # ~50m std dev
                jitter_lon = np.random.normal(0, 0.0005)
                jittered_point = (lon + jitter_lon, lat + jitter_lat)  # (lon, lat) order
                work_locs.append(jittered_point)
        else:
            logger.warning(f"Missing lat/lon for work block {bid}")

    return {'home_locations': home_locs, 'work_locations': work_locs}