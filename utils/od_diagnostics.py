"""Per-run OD matrix diagnostics — the file that makes two runs comparable.

Every run writes ``od_matrix_diagnostics.json`` into its experiment folder,
whichever source produced the matrix. The gravity path and the observed-OD path
populate the *same* fields, so any two experiments diff mechanically instead of
by memory.

Background on what each field is for: ``docs/od_matrix/design.md``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger(__name__)

DIAGNOSTICS_FILENAME = "od_matrix_diagnostics.json"

# Mean Earth radius in km, used by the cosine-corrected planar approximation.
_KM_PER_DEG_LAT = 111.32


def cosine_corrected_km(home_coords: np.ndarray,
                        work_coords: np.ndarray) -> np.ndarray:
    """Pairwise great-circle-ish distances in km between (lon, lat) points.

    Converts degrees to kilometres with a latitude cosine correction on the
    longitude axis. Without it, east-west separations are overstated by
    1/cos(lat) — 1.41× at 45°N — which is direction- *and* latitude-dependent,
    so it does not cancel out of the friction matrix (proposal §5, defect 3).

    The correction uses the mean latitude of each origin-destination pair, which
    is accurate to well under a percent over a metro-sized extent.

    Args:
        home_coords: (n, 2) array of (lon, lat) in degrees.
        work_coords: (m, 2) array of (lon, lat) in degrees.

    Returns:
        (n, m) array of distances in kilometres.
    """
    home_lon = np.asarray(home_coords, dtype=float)[:, 0][:, None]
    home_lat = np.asarray(home_coords, dtype=float)[:, 1][:, None]
    work_lon = np.asarray(work_coords, dtype=float)[:, 0][None, :]
    work_lat = np.asarray(work_coords, dtype=float)[:, 1][None, :]

    dlat_km = (work_lat - home_lat) * _KM_PER_DEG_LAT
    mean_lat = np.radians((home_lat + work_lat) * 0.5)
    dlon_km = (work_lon - home_lon) * _KM_PER_DEG_LAT * np.cos(mean_lat)

    return np.sqrt(dlat_km ** 2 + dlon_km ** 2)


def zone_sqrt_area_km(block_coords: Dict[str, Sequence[Tuple[float, float]]],
                      min_extent_km: float = 0.05) -> Dict[str, float]:
    """Estimate sqrt(zone area) in km from the spread of each zone's blocks.

    Intrazonal distance — how far a trip travels when its home and work zone are
    the same — has no meaningful value in the current data. Home and work
    centroids come from different tables, so the diagonal is an artefact: median
    159 m, and exactly zero for 53 zones at 15-county Twin Cities (which the old
    guard then inflated to 0.1 degrees = 11.1 km, making a zone the *least*
    attractive place to work in itself).

    The principled replacement is a distance derived from how big the zone
    actually is. Census area is not stored per zone — only per county — so the
    extent is estimated from the member block points already in the location
    tables: 2 standard deviations on each axis, cosine-corrected, as a
    rectangle. Zones with a single block fall back to *min_extent_km*.

    Args:
        block_coords: {zone_id: [(lon, lat), ...]} member block points.
        min_extent_km: floor on each axis, so a single-block or degenerate zone
            still gets a small positive distance rather than zero.

    Returns:
        {zone_id: sqrt(area) in km}. Multiply by ``intrazonal_factor`` to get
        the intrazonal distance.
    """
    out: Dict[str, float] = {}

    for zone, coords in block_coords.items():
        pts = np.asarray(coords, dtype=float)
        if pts.size == 0:
            out[zone] = min_extent_km
            continue

        if len(pts) < 2:
            out[zone] = min_extent_km
            continue

        lat0 = np.radians(pts[:, 1].mean())
        x_km = (pts[:, 0] - pts[:, 0].mean()) * _KM_PER_DEG_LAT * np.cos(lat0)
        y_km = (pts[:, 1] - pts[:, 1].mean()) * _KM_PER_DEG_LAT

        # 2 sigma on each axis approximates the zone's extent; the product is an
        # area proxy whose square root is the characteristic zone dimension.
        ext_x = max(2.0 * float(x_km.std()), min_extent_km)
        ext_y = max(2.0 * float(y_km.std()), min_extent_km)
        out[zone] = float(np.sqrt(ext_x * ext_y))

    return out


def trip_length_stats(matrix: pd.DataFrame,
                      home_coords: Dict[str, Tuple[float, float]],
                      work_coords: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    """Trip-weighted distance distribution and intrazonal share.

    These are the numbers that expose the distance-decay defect: the gravity
    path concentrates trips inside the home zone (17.9% at 15-county scale vs
    1.2% observed) and runs ~6.5 km short at the median.

    Args:
        matrix: zone × zone trip table.
        home_coords: {zone: (lon, lat)} for the matrix rows.
        work_coords: {zone: (lon, lat)} for the matrix columns.

    Returns:
        The ``trip_length`` block of the diagnostics file. Percentiles are
        trip-weighted, so they describe the distance an average *trip* covers,
        not an average zone pair.
    """
    rows = [z for z in matrix.index if z in home_coords]
    cols = [z for z in matrix.columns if z in work_coords]

    if not rows or not cols:
        logger.warning("trip_length_stats: no zones with coordinates — skipping")
        return {"mean_km": None, "median_km": None, "p25_km": None,
                "p75_km": None, "intrazonal_share": None}

    sub = matrix.loc[rows, cols]
    weights = sub.to_numpy(dtype=float)
    total = weights.sum()
    if total <= 0:
        logger.warning("trip_length_stats: matrix has no trips — skipping")
        return {"mean_km": None, "median_km": None, "p25_km": None,
                "p75_km": None, "intrazonal_share": None}

    dist = cosine_corrected_km(
        np.array([home_coords[z] for z in rows], dtype=float),
        np.array([work_coords[z] for z in cols], dtype=float),
    )

    flat_d = dist.ravel()
    flat_w = weights.ravel()

    keep = flat_w > 0
    flat_d, flat_w = flat_d[keep], flat_w[keep]

    order = np.argsort(flat_d)
    d_sorted = flat_d[order]
    w_sorted = flat_w[order]
    cum = np.cumsum(w_sorted)
    cum_frac = cum / cum[-1]

    def _weighted_pct(p: float) -> float:
        return float(d_sorted[np.searchsorted(cum_frac, p)])

    mean_km = float((flat_d * flat_w).sum() / flat_w.sum())

    # Intrazonal = the diagonal: trips whose home and work zone are the same.
    # These barely touch the road network, which is why an inflated share
    # under-loads exactly the arterials the count stations sit on.
    shared = [z for z in rows if z in set(cols)]
    intrazonal = float(sum(matrix.at[z, z] for z in shared)) if shared else 0.0

    stats = {
        "mean_km": round(mean_km, 2),
        "median_km": round(_weighted_pct(0.5), 2),
        "p25_km": round(_weighted_pct(0.25), 2),
        "p75_km": round(_weighted_pct(0.75), 2),
        "intrazonal_share": round(intrazonal / total, 4),
    }

    logger.info(f"  Trip length: mean {stats['mean_km']} km, median {stats['median_km']} km, "
                f"p25 {stats['p25_km']} km, p75 {stats['p75_km']} km")
    logger.info(f"  Intrazonal share: {stats['intrazonal_share']:.2%}")

    return stats


class ODDiagnostics:
    """Accumulates the diagnostics for one run, then writes them out.

    Fields are filled in as the OD build progresses; blocks that do not apply to
    the path that ran stay ``None`` rather than being omitted, so the JSON has a
    stable shape across sources and arms.
    """

    def __init__(self, source_requested: str, geo_level: str):
        self.data: Dict[str, Any] = {
            "source_requested": source_requested,
            "source_used": None,
            "fallback_reason": None,
            "geo_level": geo_level,
            "lodes": None,
            "flows": None,
            "matrix": None,
            "comparison_to_gravity_base": None,
            "survey_blend": None,
            "trip_length": None,
            "boundary": None,
            "demand_coverage": None,
            "runtime_seconds": {},
        }

    def set_source(self, used: str, fallback_reason: Optional[str] = None) -> None:
        self.data["source_used"] = used
        self.data["fallback_reason"] = fallback_reason
        if fallback_reason:
            # A fallback is never silent — it names its reason at WARNING.
            logger.warning(
                f"OD source fell back to {used!r} (requested "
                f"{self.data['source_requested']!r}): {fallback_reason}"
            )
        else:
            logger.info(f"OD source: {used}")

    def update(self, **blocks: Any) -> None:
        """Merge top-level blocks into the diagnostics payload."""
        self.data.update(blocks)

    def set_runtime(self, phase: str, seconds: float) -> None:
        self.data["runtime_seconds"][phase] = round(float(seconds), 2)

    def set_comparison_to_gravity(self, gravity_total: float,
                                  actual_total: float) -> None:
        """Record the old-vs-new total ratio and implied agent delta (E1).

        Always computed — including on the gravity path, where the two totals
        are the same — so both arms report the same field and diff cleanly.
        """
        ratio = (actual_total / gravity_total) if gravity_total else None
        self.data["comparison_to_gravity_base"] = {
            "gravity_total": float(gravity_total),
            "ratio": round(ratio, 4) if ratio is not None else None,
            "agent_delta_pct": round((ratio - 1) * 100, 2) if ratio is not None else None,
        }
        if ratio is not None:
            logger.info(
                f"  vs gravity base: {actual_total:,.0f} / {gravity_total:,.0f} "
                f"= {ratio:.4f} ({(ratio - 1) * 100:+.2f}% agents)"
            )

    def write(self, experiment_dir: Path) -> Path:
        path = Path(experiment_dir) / DIAGNOSTICS_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, default=_json_default)
        logger.info(f"  Wrote OD diagnostics to: {path}")
        return path


def _json_default(obj: Any) -> Any:
    """Make numpy scalars JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_diagnostics(experiment_dir: Path) -> Dict[str, Any]:
    """Read a previously written diagnostics file."""
    path = Path(experiment_dir) / DIAGNOSTICS_FILENAME
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
