import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, List, Optional
from utils.logger import setup_logger
from utils.od_diagnostics import cosine_corrected_km
from data_sources.base_survey_trip import BaseSurveyTrip

# Maps geo level constant → number of GEOID characters used as the zone key
_GEO_LEVEL_PREFIX_LEN: Dict[str, int] = {
    BaseSurveyTrip.GEO_TRACT: 11,
    BaseSurveyTrip.GEO_BLOCK_GROUP: 12,
}

# How pair distances are measured in the gravity friction matrix.
#   cosine_km       — kilometres with a latitude cosine correction, and an
#                     area-based intrazonal distance on the diagonal.
#   legacy_degrees  — the pre-stage-1 behaviour: raw degrees with a 0.1-degree
#                     guard on exact zeros. Kept solely so the ablation's arm A
#                     ("today's behaviour") remains runnable after the fix lands.
DISTANCE_COSINE_KM = "cosine_km"
DISTANCE_LEGACY_DEGREES = "legacy_degrees"

# Which source produces the work OD matrix.
#   lodes_od — observed LODES origin-destination flows (the published data)
#   gravity  — the estimated gravity model + IPF (the fallback)
#   auto     — prefer observed; fall back to gravity, loudly, when the region
#              has no LODES coverage for the configured year (E8)
SOURCE_AUTO = "auto"
SOURCE_LODES_OD = "lodes_od"
SOURCE_GRAVITY = "gravity"
VALID_OD_SOURCES = (SOURCE_AUTO, SOURCE_LODES_OD, SOURCE_GRAVITY)

# Distance-decay functional form.
#   exponential — exp(-beta*d). Bounded at d=0, so near-diagonal cells cannot
#                 dominate the seed. beta is in 1/km.
#   gamma       — d^alpha * exp(-beta*d). Adds a rising limb, so very short
#                 trips are suppressed rather than merely bounded.
#   power       — d^(-beta). LEGACY: unbounded as d approaches zero, which is
#                 the dominant defect (proposal §5). Kept for ablation arms.
FRICTION_EXPONENTIAL = "exponential"
FRICTION_GAMMA = "gamma"
FRICTION_POWER = "power"
VALID_FRICTION_FORMS = (FRICTION_EXPONENTIAL, FRICTION_GAMMA, FRICTION_POWER)


def compute_friction(distances: np.ndarray, beta: float,
                     form: str = FRICTION_POWER,
                     gamma_alpha: float = 1.0) -> np.ndarray:
    """Turn a distance matrix into distance-decay weights.

    The functional form matters more than the parameter. An inverse power law
    has no ceiling as distance approaches zero, so the shortest pairs dominate
    the seed before IPF ever runs — and IPF cannot fix a seed's shape, only its
    margins. Measured at 15-county Twin Cities, the closest pair sits 0.4 m
    apart and carries 24.8 million times the median pair's friction.

    Bounded forms remove that failure mode: exp(-beta*d) equals 1 at d=0 and
    falls smoothly, so no pair can run away.

    Args:
        distances: pair distances. Kilometres for the bounded forms — they are
            scale-sensitive, unlike a pure power law where the unit cancels in
            IPF normalisation.
        beta: decay parameter. Units depend on the form: 1/km for exponential
            and gamma, dimensionless for power.
        form: one of VALID_FRICTION_FORMS.
        gamma_alpha: the rising-limb exponent, gamma form only.

    Returns:
        Friction weights, same shape as *distances*.
    """
    if form not in VALID_FRICTION_FORMS:
        raise ValueError(
            f"friction form must be one of {VALID_FRICTION_FORMS}, got {form!r}"
        )
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    if form == FRICTION_EXPONENTIAL:
        return np.exp(-beta * distances)

    if form == FRICTION_GAMMA:
        # d^alpha * exp(-beta*d). Bounded because the exponential term
        # dominates; the power term is a rising limb, not a singularity.
        return np.power(distances, gamma_alpha) * np.exp(-beta * distances)

    # power (legacy): unbounded as distances approach zero.
    return np.power(distances, -beta)

logger = setup_logger(__name__)


def _apply_intrazonal_distance(distances: np.ndarray,
                               home_geoids: List[str],
                               work_geoids: List[str],
                               zone_sqrt_area: Optional[Dict[str, float]],
                               intrazonal_factor: float) -> np.ndarray:
    """Replace same-zone (diagonal) distances with an area-based value.

    A zone's distance to itself is currently the gap between its home-table and
    work-table centroids — two averages over different block sets — so it is an
    artefact, not a travel distance. It is also tiny (median 159 m), which an
    unbounded power law turns into an enormous friction weight, and that is what
    drives the intrazonal over-concentration.

    The replacement is ``factor * sqrt(zone area)``: the mean trip length inside
    a zone scales with the zone's linear dimension.

    Returns the same array, modified in place.
    """
    if not zone_sqrt_area:
        logger.warning(
            "  No zone area supplied — intrazonal distances keep their "
            "(meaningless) centroid-gap values. Trips inside a zone will stay "
            "over-weighted."
        )
        return distances

    work_index = {z: j for j, z in enumerate(work_geoids)}

    rows, cols, vals = [], [], []
    missing = 0
    for i, zone in enumerate(home_geoids):
        j = work_index.get(zone)
        if j is None:
            continue  # zone is not a destination; no diagonal cell to fix
        sqrt_area = zone_sqrt_area.get(zone)
        if sqrt_area is None:
            missing += 1
            continue
        rows.append(i)
        cols.append(j)
        vals.append(intrazonal_factor * sqrt_area)

    if rows:
        old = distances[rows, cols]
        distances[rows, cols] = vals
        logger.info(
            f"  Intrazonal distance set from zone area on {len(rows):,} diagonal "
            f"cells (factor {intrazonal_factor}): median {np.median(old) * 1000:.0f} m "
            f"-> {np.median(vals) * 1000:.0f} m"
        )
    if missing:
        logger.warning(f"  {missing:,} zone(s) had no area estimate; diagonal left as-is")

    # A zero distance would make friction infinite under a power law; floor at
    # 50 m, well below any real zone's characteristic size.
    np.maximum(distances, 0.05, out=distances)
    return distances

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

def aggregate_blocks_to_geo_level(od_matrix: pd.DataFrame,
                                   geo_level: str = BaseSurveyTrip.GEO_BLOCK_GROUP) -> pd.DataFrame:
    """Aggregate an OD matrix from blocks (15-digit) to the specified census geography.

    Args:
        od_matrix: OD matrix with 15-digit block FIPS codes as index/columns.
        geo_level: Target geography level — BaseSurveyTrip.GEO_BLOCK_GROUP (default)
                   or BaseSurveyTrip.GEO_TRACT.

    Returns:
        Aggregated OD matrix whose index/columns are zone IDs at the requested level.
    """
    prefix_len = _GEO_LEVEL_PREFIX_LEN.get(geo_level)
    if prefix_len is None:
        logger.warning(
            f"aggregate_blocks_to_geo_level: unknown geo_level '{geo_level}', "
            f"falling back to block_group (prefix 12)"
        )
        prefix_len = 12

    zone_index = od_matrix.index.astype(str).str[:prefix_len]
    zone_columns = od_matrix.columns.astype(str).str[:prefix_len]

    temp_df = od_matrix.copy()
    temp_df.index = zone_index
    temp_df.columns = zone_columns

    agg_matrix = temp_df.groupby(level=0).sum()
    agg_matrix = agg_matrix.T.groupby(level=0).sum().T

    return agg_matrix


def aggregate_blocks_to_blockgroups(od_matrix: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for aggregate_blocks_to_geo_level (block_group level)."""
    return aggregate_blocks_to_geo_level(od_matrix, geo_level=BaseSurveyTrip.GEO_BLOCK_GROUP)


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


def _scale_and_round(matrix: pd.DataFrame, scale_to_total: int) -> pd.DataFrame:
    """Scale *matrix* to *scale_to_total* and round so the total is exact.

    Uses Hamilton's (largest-remainder) method to distribute the rounding
    residual, guaranteeing the rounded matrix sums to exactly scale_to_total.
    Operates on a writeable NumPy copy so it is safe even when the input
    DataFrame's backing array is read-only.
    """
    total = matrix.sum().sum()
    if total <= 0:
        raise ValueError("Matrix sum is zero, cannot scale")

    scale_factor = scale_to_total / total
    logger.info(f"Matrix total before scaling: {total:,.0f}")
    logger.info(f"Scale factor: {scale_factor:.6f}")
    scaled = matrix * scale_factor

    logger.info("Applying robust rounding to guarantee target total...")
    floored = np.floor(scaled).astype(int)
    fractional = (scaled - floored).to_numpy().flatten()

    # Writeable copy: DataFrame.values can be a read-only view, so mutating
    # `.values.flat` in place raises "assignment destination is read-only".
    arr = floored.to_numpy().copy()
    floored_total = int(arr.sum())
    remainder = scale_to_total - floored_total

    logger.info(f"Total after flooring: {floored_total:,.0f}")
    logger.info(f"Remainder to distribute: {remainder:,.0f}")

    if remainder > 0:
        # Bump the cells with the largest fractional parts.
        top_indices = np.argsort(-fractional)[:remainder]
        arr.flat[top_indices] += 1
    elif remainder < 0:
        # Decrement the smallest-fraction cells, but only positive ones.
        positive = np.where(arr.flatten() > 0)[0]
        smallest_idx = positive[np.argsort(fractional[positive])[:abs(remainder)]]
        arr.flat[smallest_idx] -= 1

    return pd.DataFrame(arr, index=matrix.index, columns=matrix.columns)


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

    # Guard: an empty or all-zero survey matrix means the survey contributed no
    # OD pairs (e.g. survey data never ingested). The blend below would silently
    # collapse to the local matrix, ignoring alpha. Make that explicit: warn and
    # return the local matrix directly instead of pretending a blend happened.
    if survey_od_matrix.empty or survey_od_matrix.to_numpy().sum() == 0:
        logger.warning(
            "Survey OD matrix is empty/all-zero — skipping survey blend and "
            "returning the local matrix unchanged. Check that survey data was "
            "ingested (alpha=%.2f will have no effect).", alpha
        )
        target = scale_to_total if scale_to_total is not None else int(local_od_matrix.sum().sum())
        return _scale_and_round(local_od_matrix, target)

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

    # Scale combined matrix to target total and round to an exact integer total
    combined_total = combined.sum().sum()
    combined_rounded = _scale_and_round(combined, scale_to_total)

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

def create_gravity_model(work_locations_dict, home_locs_dict, beta, max_iterations=50,
                         convergence_threshold=1e-4, distance_mode=DISTANCE_COSINE_KM,
                         intrazonal_factor=0.5, zone_sqrt_area=None,
                         friction_form=FRICTION_POWER, gamma_alpha=1.0):
    """
    Create OD matrix using gravity model with IPF (Iterative Proportional Fitting).

    Parameters:
    - work_locations_dict: Dictionary with work block IDs as keys, contains 'n_employees' and 'centroid'
    - home_locs_dict: Dictionary with home block IDs as keys, contains 'n_employees' and 'centroid'
    - beta: Distance decay parameter for friction factors
    - max_iterations: Maximum IPF iterations
    - convergence_threshold: Relative difference threshold for convergence
    - distance_mode: "cosine_km" (default) measures pair distances in kilometres
      with a latitude cosine correction, and sets the diagonal from zone area.
      "legacy_degrees" reproduces the pre-stage-1 behaviour exactly — raw
      degrees with a 0.1-degree zero guard — so the ablation's arm A stays
      runnable from the same working tree.
    - intrazonal_factor: intrazonal distance = factor * sqrt(zone area).
      Ignored in legacy mode.
    - zone_sqrt_area: {zone_id: sqrt(area) in km}. Required for the intrazonal
      fix; when omitted, diagonal cells keep their (meaningless) centroid
      distance and a warning is logged.
    - friction_form: "exponential" / "gamma" (bounded) or "power" (legacy,
      unbounded — the dominant defect). Bounded forms need distances in km, so
      they should be paired with distance_mode="cosine_km".
    - gamma_alpha: rising-limb exponent for the gamma form.

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
    if distance_mode not in (DISTANCE_COSINE_KM, DISTANCE_LEGACY_DEGREES):
        raise ValueError(
            f"distance_mode must be one of "
            f"{(DISTANCE_COSINE_KM, DISTANCE_LEGACY_DEGREES)}, got {distance_mode!r}"
        )
    # A pure power law is scale-free — the unit cancels in IPF normalisation —
    # but exp(-beta*d) is not: the same beta means a completely different decay
    # in degrees than in km. Pairing a bounded form with legacy degrees would
    # apply a ~111x mis-scaled beta and look plausible while being wrong.
    if (friction_form in (FRICTION_EXPONENTIAL, FRICTION_GAMMA)
            and distance_mode == DISTANCE_LEGACY_DEGREES):
        raise ValueError(
            f"friction form {friction_form!r} is scale-sensitive and requires "
            f"distances in kilometres, but distance_mode is 'legacy_degrees'. "
            f"Use distance='cosine_km', or friction='power' for the legacy arm."
        )
    
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
    logger.info(f"Calculating distances (mode: {distance_mode})...")
    if distance_mode == DISTANCE_LEGACY_DEGREES:
        # Pre-stage-1 behaviour, kept verbatim so arm A of the ablation stays
        # reproducible. Distances are raw (lon, lat) degrees — not km — and the
        # guard only replaces *exact* zeros, turning them into 0.1 degrees
        # (~11 km), which makes those zones the least attractive place to work
        # in themselves. Both defects are why this mode is legacy.
        distances = cdist(home_coords, work_coords, metric='euclidean')
        n_exact_zeros = int((distances == 0).sum())
        distances = np.where(distances == 0, 0.1, distances)
        if n_exact_zeros:
            logger.warning(
                f"  legacy_degrees: {n_exact_zeros} exact-zero pair(s) forced to "
                f"0.1 degrees (~11.1 km)"
            )
    else:
        distances = cosine_corrected_km(home_coords, work_coords)
        distances = _apply_intrazonal_distance(
            distances, home_geoids, work_geoids,
            zone_sqrt_area=zone_sqrt_area,
            intrazonal_factor=intrazonal_factor,
        )
        logger.info(f"  Distances in km: median {np.median(distances):.2f}, "
                    f"min {distances.min():.3f}, max {distances.max():.1f}")

    # Calculate friction factors from the configured functional form.
    logger.info(f"Calculating friction factors (form: {friction_form}, beta={beta})...")
    friction_factors = compute_friction(distances, beta, form=friction_form,
                                        gamma_alpha=gamma_alpha)
    logger.info(f"  Friction range: {friction_factors.min():.3e} to "
                f"{friction_factors.max():.3e} "
                f"(max/median {friction_factors.max() / np.median(friction_factors):,.0f}x)")
    
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
                          convergence_threshold=0.03, distance_mode=DISTANCE_COSINE_KM,
                          intrazonal_factor=0.5, zone_sqrt_area=None,
                          friction_form=FRICTION_POWER, gamma_alpha=1.0):
    """
    Create origin-destination matrix using gravity model with IPF.

    Parameters:
    - work_locs_dict: Dictionary with workplace locations and n_employees
    - home_locs_dict: Dictionary with home locations, n_employees, and centroid
    - beta: Distance decay parameter (default 1.5 for metropolitan commuting)
    - max_iterations: Maximum iterations for IPF algorithm
    - convergence_threshold: Convergence criterion for IPF
    - distance_mode: "cosine_km" (default) or "legacy_degrees" — see
      create_gravity_model
    - intrazonal_factor: intrazonal distance = factor * sqrt(zone area)
    - zone_sqrt_area: {zone_id: sqrt(area) in km} for the intrazonal fix

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
        convergence_threshold=convergence_threshold,
        distance_mode=distance_mode,
        intrazonal_factor=intrazonal_factor,
        zone_sqrt_area=zone_sqrt_area,
        friction_form=friction_form,
        gamma_alpha=gamma_alpha,
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
                    blockid2homelocs: Dict, blockid2worklocs: Dict,
                    geo_level: str = BaseSurveyTrip.GEO_BLOCK_GROUP) -> Dict:
    """
    Generate home and work location samples for trips between two block groups.

    Samples blocks proportionally by n_employees within each block group,
    then returns block coordinates as (lon, lat) tuples.

    Note: Sampling is proportional and repeatable - the same block can be sampled
    across multiple OD pairs. The n_employees values represent static population
    distributions, not consumable capacity.

    Args:
        bg_origin: Origin zone ID (length depends on geo_level — 11 for tract, 12 for block_group)
        bg_destination: Destination zone ID (same)
        num_trips: Number of trips to generate
        blockid2homelocs: Dict mapping block IDs (15-digit) to home location info
                         Must have keys: 'n_employees', 'lat', 'lon'
        blockid2worklocs: Dict mapping block IDs (15-digit) to work location info
                         Must have keys: 'n_employees', 'lat', 'lon'
        geo_level: Census geography level of the OD zone IDs — BaseSurveyTrip.GEO_BLOCK_GROUP
                   (default) or BaseSurveyTrip.GEO_TRACT.

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
    prefix_len = _GEO_LEVEL_PREFIX_LEN.get(geo_level, 12)

    # Get all blocks whose zone prefix matches the requested origin/destination zone
    origin_blocks = {bid: data for bid, data in blockid2homelocs.items()
                     if bid[:prefix_len] == bg_origin}
    dest_blocks = {bid: data for bid, data in blockid2worklocs.items()
                   if bid[:prefix_len] == bg_destination}

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