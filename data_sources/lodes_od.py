"""Observed LODES origin-destination commute flows.

LODES publishes actual home-block → work-block job counts, which the pipeline
previously only *estimated* with a gravity model. This module loads those flows
and assembles them into a zone × zone matrix on the same contract the gravity
path produces (see ``docs/od_matrix/design.md`` §8).

The two LODES OD parts and why both are needed
----------------------------------------------
``<state>_od_main`` holds flows with **both** ends in that state.
``<state>_od_aux`` holds flows whose **work** end is in that state and whose home
end is anywhere else — it is filed under the *work* state, and there is no
home-side equivalent.

A region that spans a state line therefore has internal commutes filed in
``aux``: both ends sit inside the region, but LODES calls them auxiliary because
they cross the state boundary. Building internal flows from ``main`` alone drops
them silently. The rule that is correct for every region — single- or
multi-state — is: read ``main`` and ``aux`` for every state the region touches,
concatenate, then filter both ends to the configured counties. Single-state
regions simply gain no rows from ``aux``.

Flow classes, after the county filter:
    I-I  both ends in-region   → the matrix
    I-E  home in-region, work outside  → boundary flow (see boundary_policy)
    E-I  work in-region, home outside  → boundary flow

Reference: ``docs/od_matrix/design.md`` §5.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from data_sources.base_survey_trip import BaseSurveyTrip
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Maps geo level constant → number of GEOID characters used as the zone key.
# Mirrors models/od_matrix_v3.py so both sources key zones identically.
_GEO_LEVEL_PREFIX_LEN: Dict[str, int] = {
    BaseSurveyTrip.GEO_TRACT: 11,
    BaseSurveyTrip.GEO_BLOCK_GROUP: 12,
}

# LODES8 is enumerated on 2020 Census blocks, which is what the pipeline's
# home_locations / work_locations tables store. LODES7 (2010 blocks) would not
# align with the stored 15-digit GEOIDs, so it is refused rather than silently
# producing a matrix keyed on a different block vintage (E7).
_SUPPORTED_VERSIONS = ("LODES8",)

# The Census vintage the block GEOIDs above are enumerated on. County lookups
# must use this same vintage: county FIPS codes are not stable over time, and a
# newer TIGER release can renumber or abolish them outright. Connecticut is the
# live example — it replaced its 8 counties (09001-09015) with 9 planning
# regions (09110-09190) effective 2022, so a 2020 block GEOID's county digits
# match *nothing* in a current-vintage county table.
_BLOCK_VINTAGE_YEAR = 2020

# The OD product has no segment selector in its filename — it publishes S000
# (all jobs) plus breakdown columns. RAC/WAC are pulled with a configured
# segment, so a config asking for anything else describes a universe the OD
# files cannot supply (E6).
_OD_SEGMENT = "S000"

# Column holding total jobs for the OD product.
_OD_TOTAL_COL = "S000"


class LodesODUnavailable(Exception):
    """Raised when observed OD cannot be assembled for the configured region.

    Carries the reason so callers running under ``source: "auto"`` can log why
    they fell back to gravity rather than reporting a bare failure.
    """


@dataclass
class FlowTotals:
    """Trip counts by flow class, before any zone aggregation."""

    internal_ii: int = 0
    outbound_ie: int = 0
    inbound_ei: int = 0

    @property
    def total(self) -> int:
        return self.internal_ii + self.outbound_ie + self.inbound_ei

    @property
    def internal_share(self) -> float:
        """Share of the region's employed *residents* who also work in-region.

        ``I-I / (I-I + I-E)`` — the resident-side share. This is the definition
        the design docs quote (94.18% at 15-county Twin Cities) and the one that
        explains the E1 agent drop: the gravity matrix gives work trips to every
        employed resident (I-I + I-E), while observed OD only gives one to those
        working in-region (I-I). The gap between the two totals *is* this share.

        A low value means the configured county list is narrow relative to the
        real commute shed — reported every run rather than assumed (D5).
        """
        resident_side = self.internal_ii + self.outbound_ie
        return (self.internal_ii / resident_side) if resident_side else 0.0

    @property
    def job_side_share(self) -> float:
        """Share of the region's jobs filled by residents of the region.

        ``I-I / (I-I + E-I)`` — the mirror of internal_share on the column
        margin. Reported alongside it because the two diverge (a job importer
        differs from a job exporter) and the column margin is what the work
        end of the matrix has to satisfy.
        """
        job_side = self.internal_ii + self.inbound_ei
        return (self.internal_ii / job_side) if job_side else 0.0

    @property
    def two_sided_share(self) -> float:
        """``I-I / (I-I + I-E + E-I)`` — share of all region-touching flow."""
        total = self.total
        return (self.internal_ii / total) if total else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "internal_ii": self.internal_ii,
            "outbound_ie": self.outbound_ie,
            "inbound_ei": self.inbound_ei,
            # Resident-side (I-I / (I-I + I-E)) — the headline figure; see the
            # internal_share docstring for why this is the one that matters.
            "internal_share": round(self.internal_share, 4),
            "job_side_share": round(self.job_side_share, 4),
            "two_sided_share": round(self.two_sided_share, 4),
        }


@dataclass
class LodesODResult:
    """Assembled observed OD flows plus the diagnostics the run must report."""

    # Block-level I-I flows: columns h_geocode, w_geocode, jobs.
    internal_flows: pd.DataFrame
    # Block-level boundary flows, kept for the stage-3 anchoring policy.
    outbound_flows: pd.DataFrame
    inbound_flows: pd.DataFrame

    totals: FlowTotals
    states: List[str]
    year: int
    version: str
    job_type: str
    # Rows read per source file, before the county filter: {"mn_main": 1234567}
    file_rows: Dict[str, int] = field(default_factory=dict)
    # I-I trips that came from an aux file — the check that the E4 rule fired.
    aux_sourced_ii: int = 0
    runtime_seconds: float = 0.0
    # Filled in by build_observed_od_matrix: what the boundary policy did.
    boundary_stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def aux_sourced_ii_share(self) -> float:
        return (self.aux_sourced_ii / self.totals.internal_ii) if self.totals.internal_ii else 0.0

    def as_diagnostics(self) -> Dict[str, Any]:
        """The ``lodes`` and ``flows`` blocks of od_matrix_diagnostics.json."""
        return {
            "lodes": {
                "year": self.year,
                "version": self.version,
                "job_type": self.job_type,
                "segment": _OD_SEGMENT,
                "states": self.states,
                "files": self.file_rows,
                "aux_sourced_ii": self.aux_sourced_ii,
                "aux_sourced_ii_share": round(self.aux_sourced_ii_share, 4),
            },
            "flows": self.totals.as_dict(),
        }


def _configure_pygris_cache(data_dir: str) -> None:
    """Point pygris's cache at the project data dir.

    pygris caches to a platform user-cache dir; the rest of the pipeline
    redirects it into ``{data_dir}/pygris_cache`` so downloads are shared and
    survive in the project. Same monkey-patch as models/home_locs_v2.py.
    """
    import platformdirs

    cache_dir = os.path.join(data_dir, "pygris_cache")
    os.makedirs(cache_dir, exist_ok=True)

    original = platformdirs.user_cache_dir
    if getattr(original, "_tareek_patched", False):
        return

    def _patched(app_name, *args, **kwargs):
        if app_name == "pygris":
            return cache_dir
        return original(app_name, *args, **kwargs)

    _patched._tareek_patched = True  # type: ignore[attr-defined]
    platformdirs.user_cache_dir = _patched


def _read_od_part(state_abbr: str, year: int, part: str, job_type: str,
                  version: str) -> pd.DataFrame:
    """Read one LODES OD file (main or aux) for one state.

    Raises LodesODUnavailable when the file cannot be read for this
    state-year, which is the E8 partial-coverage case: pygris writes whatever
    the HTTP request returned into its cache, so a 404 page lands on disk as a
    "cached" file and only fails when parsed. Both failure modes surface here.
    """
    from pygris.data import get_lodes

    try:
        df = get_lodes(
            state=state_abbr,
            year=year,
            lodes_type="od",
            part=part,
            job_type=job_type,
            version=version,
            agg_level="block",
            cache=True,
        )
    except Exception as e:  # noqa: BLE001 - any read failure means "no coverage"
        raise LodesODUnavailable(
            f"LODES OD {part} unavailable for {state_abbr.upper()} {year} "
            f"({version}, {job_type}): {e}"
        ) from e

    missing = {"h_geocode", "w_geocode", _OD_TOTAL_COL} - set(df.columns)
    if missing:
        raise LodesODUnavailable(
            f"LODES OD {part} for {state_abbr.upper()} {year} is missing "
            f"expected column(s) {sorted(missing)} — refusing to guess at the "
            f"schema. Columns present: {sorted(df.columns)[:12]}"
        )

    return df


def _county_key(geocode: pd.Series) -> pd.Series:
    """5-digit state+county key from a 15-digit block GEOID."""
    return geocode.str[:5]


def assemble_lodes_od(config: Dict[str, Any],
                      state_abbr_mapping: Optional[Dict[str, str]] = None) -> LodesODResult:
    """Load observed LODES OD flows for the region configured in *config*.

    Downloads (or reads from cache) ``main`` and ``aux`` for every state the
    configured counties touch, concatenates them, and splits the result into
    internal (I-I) and boundary (I-E / E-I) flows.

    Args:
        config: Full pipeline config. Uses ``region.counties``, ``data.lodes``
                (year / job_type / segment) and ``data.data_dir``.
        state_abbr_mapping: Optional {state_fips: state_abbr}. Looked up from
                the DB via RegionHelper when omitted.

    Returns:
        LodesODResult with block-level flows and the run's diagnostics.

    Raises:
        LodesODUnavailable: any state in the region lacks OD coverage for the
            configured year, or the config's vintage/segment cannot be served
            by the OD product. The whole region falls back to gravity rather
            than mixing observed and estimated flows in one matrix (E8).
    """
    start = time.perf_counter()

    lodes_config = config["data"]["lodes"]
    year = int(lodes_config["year"])
    job_type = lodes_config["job_type"]
    segment = lodes_config.get("segment", _OD_SEGMENT)
    version = lodes_config.get("version", "LODES8")

    # E7 — vintage must match the Census block vintage of the stored GEOIDs.
    if version not in _SUPPORTED_VERSIONS:
        raise LodesODUnavailable(
            f"LODES version {version!r} is not supported: the pipeline stores "
            f"2020 Census block GEOIDs, which only {_SUPPORTED_VERSIONS[0]} is "
            f"enumerated on. A different vintage would key the matrix on blocks "
            f"that do not align with home_locations / work_locations."
        )

    # E6 — the OD product publishes S000 only; a config asking for a different
    # workforce segment describes a universe OD cannot supply. Failing here
    # beats silently comparing incompatible universes against RAC/WAC.
    if segment != _OD_SEGMENT:
        raise LodesODUnavailable(
            f"data.lodes.segment is {segment!r}, but the LODES OD product has no "
            f"segment selector — it publishes {_OD_SEGMENT} (all jobs). RAC/WAC "
            f"are pulled with {segment!r}, so observed OD would cover a different "
            f"population than the stored resident/job totals."
        )

    counties: List[str] = list(config["region"]["counties"])
    if not counties:
        raise LodesODUnavailable("region.counties is empty — nothing to assemble")

    county_set: Set[str] = set(counties)
    state_fips_set = sorted({g[:2] for g in counties})

    if state_abbr_mapping is None:
        from utils.region_utils import RegionHelper

        state_abbr_mapping = RegionHelper(config).get_state_abbr_mapping()

    missing_states = [s for s in state_fips_set if s not in state_abbr_mapping]
    if missing_states:
        raise LodesODUnavailable(
            f"No state abbreviation known for FIPS {missing_states} — run the "
            f"global data setup so states are populated in the DB"
        )

    _configure_pygris_cache(config["data"]["data_dir"])

    logger.info("=" * 70)
    logger.info("LOADING OBSERVED LODES OD FLOWS")
    logger.info("=" * 70)
    logger.info(f"  Year: {year} | Version: {version} | Job type: {job_type} | Segment: {_OD_SEGMENT}")
    logger.info(f"  Region: {len(counties)} counties across {len(state_fips_set)} state(s): "
                f"{[state_abbr_mapping[s].upper() for s in state_fips_set]}")

    frames: List[pd.DataFrame] = []
    file_rows: Dict[str, int] = {}

    # E4 — main AND aux for every region state. One rule, correct everywhere:
    # single-state regions get no extra rows from aux; multi-state regions get
    # exactly the cross-border internal flows main alone would have dropped.
    for state_fips in state_fips_set:
        state_abbr = state_abbr_mapping[state_fips]
        for part in ("main", "aux"):
            key = f"{state_abbr.lower()}_{part}"
            df = _read_od_part(state_abbr, year, part, job_type, version)
            file_rows[key] = len(df)
            logger.info(f"  {key}: {len(df):,} rows downloaded")

            keep = df[["h_geocode", "w_geocode", _OD_TOTAL_COL]].copy()
            keep = keep.rename(columns={_OD_TOTAL_COL: "jobs"})
            keep["source_part"] = part
            frames.append(keep)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"  Combined: {len(combined):,} block-pair rows across all files")

    # Classify each flow by which ends fall inside the configured counties.
    h_in = _county_key(combined["h_geocode"]).isin(county_set)
    w_in = _county_key(combined["w_geocode"]).isin(county_set)

    internal = combined[h_in & w_in]
    outbound = combined[h_in & ~w_in]
    inbound = combined[~h_in & w_in]

    totals = FlowTotals(
        internal_ii=int(internal["jobs"].sum()),
        outbound_ie=int(outbound["jobs"].sum()),
        inbound_ei=int(inbound["jobs"].sum()),
    )

    aux_sourced_ii = int(internal.loc[internal["source_part"] == "aux", "jobs"].sum())

    elapsed = time.perf_counter() - start

    result = LodesODResult(
        internal_flows=internal[["h_geocode", "w_geocode", "jobs"]].reset_index(drop=True),
        outbound_flows=outbound[["h_geocode", "w_geocode", "jobs"]].reset_index(drop=True),
        inbound_flows=inbound[["h_geocode", "w_geocode", "jobs"]].reset_index(drop=True),
        totals=totals,
        states=state_fips_set,
        year=year,
        version=version,
        job_type=job_type,
        file_rows=file_rows,
        aux_sourced_ii=aux_sourced_ii,
        runtime_seconds=elapsed,
    )

    _log_flow_summary(result, multi_state=len(state_fips_set) > 1)
    return result


def _log_flow_summary(result: LodesODResult, multi_state: bool) -> None:
    """Report flow classes, internal_share and the aux contribution (E4, E5)."""
    t = result.totals
    logger.info("-" * 70)
    logger.info("  Observed flow classes (after county filter):")
    logger.info(f"    I-I internal:  {t.internal_ii:>12,}")
    logger.info(f"    I-E outbound:  {t.outbound_ie:>12,}  (dropped unless boundary_policy=anchor)")
    logger.info(f"    E-I inbound:   {t.inbound_ei:>12,}  (dropped unless boundary_policy=anchor)")
    logger.info(f"    internal_share (resident-side, I-I/(I-I+I-E)): {t.internal_share:.4f}")
    logger.info(f"    job-side share (I-I/(I-I+E-I)):                {t.job_side_share:.4f}")
    logger.info(f"    two-sided share (I-I/all region-touching):     {t.two_sided_share:.4f}")

    # E5 — never leave the boundary loss implicit.
    dropped = t.outbound_ie + t.inbound_ei
    if dropped:
        logger.warning(
            f"  {dropped:,} boundary commutes ({t.outbound_ie:,} I-E + {t.inbound_ei:,} E-I) "
            f"are outside the region's internal matrix — {1 - t.two_sided_share:.1%} of "
            f"region-touching flow. {t.outbound_ie:,} employed residents "
            f"({1 - t.internal_share:.1%}) get no work trip from observed OD."
        )

    # E4 — the aux share is the check that the main+aux rule actually fired.
    share = result.aux_sourced_ii_share
    logger.info(f"    I-I sourced from aux files: {result.aux_sourced_ii:,} ({share:.2%})")
    if multi_state and result.aux_sourced_ii == 0:
        logger.warning(
            "  Region spans multiple states but NO internal flow came from aux "
            "files. Expected cross-state internal commutes — verify the aux "
            "downloads succeeded."
        )
    logger.info("-" * 70)


def flows_to_zone_matrix(flows: pd.DataFrame,
                         home_zones: Sequence[str],
                         work_zones: Sequence[str],
                         geo_level: str = BaseSurveyTrip.GEO_BLOCK_GROUP) -> pd.DataFrame:
    """Aggregate block-level flows to zones and reindex onto the DB zone union.

    Blocks are aggregated straight from the flow table — a block-level matrix is
    never materialised (it would be ~9 GB dense at 15-county scale).

    Zones present in the location tables but absent from the flow data survive
    as explicit zeros, so the matrix keeps the same shape as the gravity path
    and downstream code sees the same zone universe (E2).

    Args:
        flows: columns h_geocode, w_geocode, jobs (15-digit block GEOIDs).
        home_zones: zone IDs from the home_locations table (matrix rows).
        work_zones: zone IDs from the work_locations table (matrix columns).
        geo_level: zone granularity — block_group (12 chars) or tract (11).

    Returns:
        DataFrame indexed by home_zones, columns work_zones, float trip counts.
    """
    prefix_len = _GEO_LEVEL_PREFIX_LEN.get(geo_level, 12)

    rows = sorted(set(home_zones))
    cols = sorted(set(work_zones))

    if flows.empty:
        logger.warning("flows_to_zone_matrix: no flows supplied — returning all-zero matrix")
        return pd.DataFrame(0.0, index=rows, columns=cols)

    zoned = pd.DataFrame({
        "h_zone": flows["h_geocode"].str[:prefix_len],
        "w_zone": flows["w_geocode"].str[:prefix_len],
        "jobs": flows["jobs"].to_numpy(),
    })

    grouped = zoned.groupby(["h_zone", "w_zone"], sort=False)["jobs"].sum()

    matrix = grouped.unstack(fill_value=0)

    # Flows on zones the location tables don't carry would be dropped by the
    # reindex below. That should not happen (verified: 0 missing blocks), so
    # report it loudly rather than losing trips quietly.
    unknown_rows = set(matrix.index) - set(rows)
    unknown_cols = set(matrix.columns) - set(cols)
    if unknown_rows or unknown_cols:
        lost = 0.0
        if unknown_rows:
            lost += float(matrix.loc[list(unknown_rows)].to_numpy().sum())
        if unknown_cols:
            lost += float(matrix[list(unknown_cols)].to_numpy().sum())
        logger.warning(
            f"flows_to_zone_matrix: {len(unknown_rows)} home and {len(unknown_cols)} work "
            f"zones in the OD flows are absent from the location tables "
            f"(~{lost:,.0f} trips dropped by the reindex)"
        )

    matrix = matrix.reindex(index=rows, columns=cols, fill_value=0).astype(float)

    logger.info(f"  Observed OD matrix: {matrix.shape[0]:,} × {matrix.shape[1]:,} zones, "
                f"{matrix.sum().sum():,.0f} trips")

    return matrix


BOUNDARY_DROP = "drop"
BOUNDARY_ANCHOR = "anchor"
VALID_BOUNDARY_POLICIES = (BOUNDARY_DROP, BOUNDARY_ANCHOR)


def external_county_coords(config: Dict[str, Any],
                           geocodes: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    """Locate out-of-region blocks by their county centroid.

    Anchoring only needs to know which *direction* a boundary trip comes from,
    so that the nearest in-region crossing zone can be found. County resolution
    is sufficient for that and costs nothing: the counties table already stores
    an interior point for every US county (3,235 rows), whereas fetching real
    block geometry would mean a TIGER shapefile download per external state.

    Args:
        config: pipeline config (for the data dir holding the DB).
        geocodes: 15-digit block GEOIDs outside the configured region.

    Counties whose FIPS code cannot be resolved fall back to their state's
    centroid, which still places the trip on the correct side of the region —
    coarse, but a boundary trip kept in roughly the right direction beats one
    silently dropped. See ``_state_centroids`` for why that case arises at all.

    Returns:
        {block_geoid: (lon, lat)} for every block whose county or state is known.
    """
    from models.models import County, initialize_tables

    wanted = {str(g)[:5] for g in geocodes}
    if not wanted:
        return {}

    db_manager = initialize_tables(config["data"]["data_dir"])
    try:
        with db_manager.session_scope() as session:
            rows = session.query(
                County.geoid, County.intptlat, County.intptlon
            ).filter(County.geoid.in_(sorted(wanted))).all()
            centroids = {
                r[0]: (float(r[2]), float(r[1]))
                for r in rows if r[1] is not None and r[2] is not None
            }
    finally:
        db_manager.close()

    missing = wanted - set(centroids)
    if missing:
        fallback = _state_centroids({m[:2] for m in missing})
        recovered = {m: fallback[m[:2]] for m in missing if m[:2] in fallback}
        centroids.update(recovered)

        still_missing = sorted(missing - set(recovered))
        logger.warning(
            f"  {len(missing)} external county FIPS not in the counties table "
            f"(e.g. {sorted(missing)[:5]}). Most likely a FIPS vintage "
            f"mismatch: LODES blocks are enumerated on {_BLOCK_VINTAGE_YEAR} "
            f"geography, and county codes change between vintages (Connecticut "
            f"replaced 09001-09015 with 09110-09190 in 2022)."
        )
        if recovered:
            logger.warning(
                f"    Falling back to the state centroid for "
                f"{len(recovered)} of them, so their trips are still kept — "
                f"direction is preserved, precision is not."
            )
        if still_missing:
            logger.warning(
                f"    No state centroid either for {len(still_missing)} "
                f"(e.g. {still_missing[:5]}) — those trips cannot be placed "
                f"and will be dropped."
            )

    return {str(g): centroids[str(g)[:5]] for g in geocodes if str(g)[:5] in centroids}


def _state_centroids(state_fips: Set[str]) -> Dict[str, Tuple[float, float]]:
    """Centroid per state FIPS, for counties whose own code cannot be resolved.

    County FIPS codes are not stable across Census vintages — they get split,
    merged, renumbered, or abolished (Connecticut 2022, Alaska's Valdez-Cordova
    2019, several Virginia independent cities). When a LODES block's county
    digits match nothing in the counties table, the state is still unambiguous
    from the first two digits, so anchoring can fall back to it rather than
    discarding the commute.

    Returns ``{}`` on any failure — the caller treats that as "cannot place".
    """
    if not state_fips:
        return {}
    try:
        from pygris import states as get_states
        gdf = get_states(cache=True, year=_BLOCK_VINTAGE_YEAR)
    except Exception as exc:
        logger.warning(f"    Could not fetch state centroids for fallback: {exc}")
        return {}

    # The state cartographic file carries no INTPTLAT/INTPTLON columns (unlike
    # the county one), so the centroid comes from the geometry. representative_
    # point() is used rather than centroid() because it is guaranteed to fall
    # inside the polygon — relevant for states whose shape is concave or split
    # across islands, where the true centroid can land in open water.
    out: Dict[str, Tuple[float, float]] = {}
    for _, row in gdf.iterrows():
        fips = str(row.get("STATEFP", ""))
        if fips not in state_fips:
            continue
        geom = row.get("geometry")
        if geom is None or geom.is_empty:
            continue
        try:
            pt = geom.representative_point()
            out[fips] = (float(pt.x), float(pt.y))
        except Exception:
            continue
    return out


def anchor_boundary_flows(outbound: pd.DataFrame,
                          inbound: pd.DataFrame,
                          home_zone_coords: Dict[str, Tuple[float, float]],
                          work_zone_coords: Dict[str, Tuple[float, float]],
                          external_block_coords: Dict[str, Tuple[float, float]],
                          geo_level: str = BaseSurveyTrip.GEO_BLOCK_GROUP,
                          ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Re-point boundary commutes at the in-region zone where they cross.

    Boundary flows are real trips that load the network — an I-E commuter drives
    from their in-region home to the edge, an E-I commuter arrives through it.
    Dropping them discards the largest *known* piece of missing demand (E5), and
    a uniform `scaling_factor` bump cannot put it back in the right place:
    boundary flow concentrates on radials, not everywhere.

    Anchoring keeps the in-region leg and replaces the out-of-region end with
    the in-region zone geographically closest to it — an approximation of the
    cordon point where the trip enters or leaves. The in-region leg then loads
    the network as it should, and no external zones are added, so the matrix
    keeps its shape and the network keeps its size.

    Args:
        outbound: I-E flows (h_geocode in region, w_geocode outside).
        inbound: E-I flows (h_geocode outside, w_geocode in region).
        home_zone_coords: {zone: (lon, lat)} for in-region home zones.
        work_zone_coords: {zone: (lon, lat)} for in-region work zones.
        external_block_coords: {block_geoid: (lon, lat)} for the out-of-region
            endpoints. Blocks without coordinates are reported, not guessed at.
        geo_level: zone granularity.

    Returns:
        (anchored_flows, stats) where anchored_flows has columns
        h_geocode/w_geocode/jobs referring only to in-region zones, ready to be
        added to the internal matrix.
    """
    from scipy.spatial import cKDTree

    prefix_len = _GEO_LEVEL_PREFIX_LEN.get(geo_level, 12)

    stats: Dict[str, Any] = {
        "policy": BOUNDARY_ANCHOR,
        "outbound_trips": int(outbound["jobs"].sum()) if not outbound.empty else 0,
        "inbound_trips": int(inbound["jobs"].sum()) if not inbound.empty else 0,
        "anchored_ie": 0,
        "anchored_ei": 0,
        "unplaceable_ie": 0,
        "unplaceable_ei": 0,
    }

    def _nearest_zone(external_geocodes: pd.Series,
                      zone_coords: Dict[str, Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Map each external block to the nearest in-region zone.

        Distances are compared in a locally-equidistant projection (degrees
        scaled by the latitude cosine), so "nearest" means nearest on the
        ground rather than nearest in raw degrees.
        """
        zones = sorted(zone_coords)
        pts = np.array([zone_coords[z] for z in zones], dtype=float)
        lat0 = np.radians(pts[:, 1].mean())
        scale = np.array([np.cos(lat0), 1.0])

        tree = cKDTree(pts * scale)

        ext_pts = np.array(
            [external_block_coords.get(g, (np.nan, np.nan)) for g in external_geocodes],
            dtype=float,
        )
        known = ~np.isnan(ext_pts).any(axis=1)

        assigned = np.full(len(ext_pts), -1, dtype=int)
        if known.any():
            _, idx = tree.query(ext_pts[known] * scale)
            assigned[known] = idx

        return assigned, known

    frames = []

    # I-E: home stays put, the work end is pulled back to the boundary zone the
    # commuter would exit through.
    if not outbound.empty and work_zone_coords:
        assigned, known = _nearest_zone(outbound["w_geocode"], work_zone_coords)
        zones = sorted(work_zone_coords)
        keep = known
        if keep.any():
            frames.append(pd.DataFrame({
                "h_geocode": outbound.loc[keep, "h_geocode"].to_numpy(),
                "w_geocode": [zones[i] for i in assigned[keep]],
                "jobs": outbound.loc[keep, "jobs"].to_numpy(),
            }))
            stats["anchored_ie"] = int(outbound.loc[keep, "jobs"].sum())
        stats["unplaceable_ie"] = int(outbound.loc[~keep, "jobs"].sum())

    # E-I: work stays put, the home end is pulled in to the entry zone.
    if not inbound.empty and home_zone_coords:
        assigned, known = _nearest_zone(inbound["h_geocode"], home_zone_coords)
        zones = sorted(home_zone_coords)
        keep = known
        if keep.any():
            frames.append(pd.DataFrame({
                "h_geocode": [zones[i] for i in assigned[keep]],
                "w_geocode": inbound.loc[keep, "w_geocode"].to_numpy(),
                "jobs": inbound.loc[keep, "jobs"].to_numpy(),
            }))
            stats["anchored_ei"] = int(inbound.loc[keep, "jobs"].sum())
        stats["unplaceable_ei"] = int(inbound.loc[~keep, "jobs"].sum())

    if frames:
        anchored = pd.concat(frames, ignore_index=True)
    else:
        anchored = pd.DataFrame(columns=["h_geocode", "w_geocode", "jobs"])

    # The anchored endpoints are already zone IDs; the in-region ends are still
    # 15-digit blocks. Normalise both so the caller can aggregate uniformly.
    if not anchored.empty:
        anchored["h_geocode"] = anchored["h_geocode"].str[:prefix_len]
        anchored["w_geocode"] = anchored["w_geocode"].str[:prefix_len]

    stats["anchored_total"] = stats["anchored_ie"] + stats["anchored_ei"]
    stats["unplaceable_total"] = stats["unplaceable_ie"] + stats["unplaceable_ei"]

    logger.info(
        f"  Boundary anchoring: {stats['anchored_ie']:,} I-E + "
        f"{stats['anchored_ei']:,} E-I = {stats['anchored_total']:,} trips "
        f"re-pointed at their crossing zone"
    )
    if stats["unplaceable_total"]:
        logger.warning(
            f"  {stats['unplaceable_total']:,} boundary trips could not be anchored "
            f"(no coordinates for the external endpoint) and remain dropped"
        )

    return anchored, stats


def _assembly_cache_key(config: Dict[str, Any], geo_level: str,
                        boundary_policy: str) -> str:
    """Stable fingerprint of everything that changes the assembled matrix.

    The raw LODES downloads are already cached by pygris; what this caches is
    the *assembly* — the county filter and groupby over several million
    block-pair rows, which is the slow part on a repeat run.

    The key covers the county list (order-independent), the LODES parameters,
    the zone level and the boundary policy. Any change to those produces a
    different matrix and so must miss the cache.
    """
    import hashlib

    lodes = config["data"]["lodes"]
    parts = [
        ",".join(sorted(config["region"]["counties"])),
        str(lodes["year"]),
        str(lodes["job_type"]),
        str(lodes.get("version", "LODES8")),
        geo_level,
        boundary_policy,
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_observed_od_matrix(config: Dict[str, Any],
                             home_zones: Sequence[str],
                             work_zones: Sequence[str],
                             geo_level: str = BaseSurveyTrip.GEO_BLOCK_GROUP,
                             state_abbr_mapping: Optional[Dict[str, str]] = None,
                             use_cache: bool = True,
                             home_zone_coords: Optional[Dict[str, Tuple[float, float]]] = None,
                             work_zone_coords: Optional[Dict[str, Tuple[float, float]]] = None,
                             ) -> Tuple[pd.DataFrame, "LodesODResult"]:
    """Assemble the observed LODES OD matrix for the configured region.

    This is the stage-2 entry point: it produces a matrix satisfying the same
    contract as the gravity path (zone IDs as index/columns, trip counts as
    values), so nothing downstream needs to know which source produced it.

    Args:
        config: Full pipeline config.
        home_zones: zone IDs from the home_locations table (matrix rows).
        work_zones: zone IDs from the work_locations table (matrix columns).
        geo_level: zone granularity.
        state_abbr_mapping: optional {state_fips: abbr}; looked up when omitted.
        use_cache: read/write the assembled matrix from the on-disk cache.
        home_zone_coords: {zone: (lon, lat)} for in-region home zones. Required
            when ``od_matrix.boundary_policy`` is "anchor".
        work_zone_coords: {zone: (lon, lat)} for in-region work zones. Required
            when ``od_matrix.boundary_policy`` is "anchor".

    Returns:
        (matrix, result) — the zone matrix and the underlying flow assembly,
        the latter carrying the diagnostics and the boundary flows stage 3
        needs.

    Raises:
        LodesODUnavailable: propagated from assemble_lodes_od so callers running
            under ``source: "auto"`` can fall back to gravity.
    """
    observed = assemble_lodes_od(config, state_abbr_mapping=state_abbr_mapping)

    od_config = config.get("od_matrix", {})
    boundary_policy = od_config.get("boundary_policy", BOUNDARY_DROP)
    if boundary_policy not in VALID_BOUNDARY_POLICIES:
        raise ValueError(
            f"od_matrix.boundary_policy must be one of {VALID_BOUNDARY_POLICIES}, "
            f"got {boundary_policy!r}"
        )

    cache_path = None
    if use_cache:
        cache_dir = Path(config["data"]["data_dir"]) / "lodes_od_cache"
        key = _assembly_cache_key(config, geo_level, boundary_policy)
        cache_path = cache_dir / f"od_matrix_{key}.parquet"

        if cache_path.exists():
            try:
                matrix = pd.read_parquet(cache_path)
                matrix.index = matrix.index.astype(str)
                matrix.columns = matrix.columns.astype(str)
                # The cached matrix was built for a specific zone universe; if
                # the location tables have changed since, it is stale.
                if (set(matrix.index) == set(home_zones)
                        and set(matrix.columns) == set(work_zones)):
                    logger.info(f"  Loaded assembled OD matrix from cache: {cache_path}")
                    return matrix, observed
                logger.info("  Cached OD matrix covers a different zone set — rebuilding")
            except Exception as e:  # noqa: BLE001 - a bad cache must never be fatal
                logger.warning(f"  Could not read OD matrix cache ({e}) — rebuilding")

    flows = observed.internal_flows

    if boundary_policy == BOUNDARY_ANCHOR:
        # Recover the boundary flows instead of discarding them (E5, D4). The
        # in-region leg still loads the network; no external zones are added.
        external = pd.concat([
            observed.outbound_flows["w_geocode"],
            observed.inbound_flows["h_geocode"],
        ], ignore_index=True).unique()

        if not home_zone_coords or not work_zone_coords:
            raise ValueError(
                "boundary_policy='anchor' needs home_zone_coords and "
                "work_zone_coords to find the crossing zone for each boundary "
                "trip. Pass them from the caller's zone dictionaries."
            )

        ext_coords = external_county_coords(config, external)
        anchored, boundary_stats = anchor_boundary_flows(
            observed.outbound_flows,
            observed.inbound_flows,
            home_zone_coords=home_zone_coords,
            work_zone_coords=work_zone_coords,
            external_block_coords=ext_coords,
            geo_level=geo_level,
        )
        if not anchored.empty:
            flows = pd.concat([flows, anchored], ignore_index=True)
        observed.boundary_stats = boundary_stats
    else:
        observed.boundary_stats = {
            "policy": BOUNDARY_DROP,
            "dropped_ie": observed.totals.outbound_ie,
            "dropped_ei": observed.totals.inbound_ei,
            "anchored_total": 0,
        }

    matrix = flows_to_zone_matrix(flows, home_zones, work_zones, geo_level=geo_level)

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            matrix.to_parquet(cache_path)
            logger.info(f"  Cached assembled OD matrix to: {cache_path}")
        except Exception as e:  # noqa: BLE001 - caching is an optimisation only
            logger.warning(f"  Could not write OD matrix cache: {e}")

    return matrix, observed


def reconcile_with_direct_aggregation(matrix: pd.DataFrame,
                                      observed: "LodesODResult",
                                      tolerance: int = 1) -> Dict[str, Any]:
    """Check the zone matrix totals against the raw flow table (gate G2).

    The matrix is built by grouping block pairs into zones, so its total must
    equal the flow total that went into it, up to reindex losses. A mismatch
    means trips were dropped or double-counted somewhere in the assembly.

    Under ``boundary_policy='anchor'`` the assembly deliberately adds the
    re-pointed I-E/E-I commutes on top of the I-I flows, so the expected total
    is internal_ii + anchored_total. Comparing against internal_ii alone would
    report every anchored run as a failure.

    Returns a dict recording the check; logs a WARNING if it fails.
    """
    matrix_total = float(matrix.to_numpy().sum())
    internal_total = float(observed.totals.internal_ii)
    anchored_total = float(
        (getattr(observed, "boundary_stats", None) or {}).get("anchored_total", 0) or 0
    )
    flow_total = internal_total + anchored_total
    delta = matrix_total - flow_total

    result = {
        "matrix_total": matrix_total,
        "direct_aggregation_total": flow_total,
        "internal_ii_total": internal_total,
        "anchored_total": anchored_total,
        "delta": delta,
        "within_tolerance": abs(delta) <= tolerance,
        "aux_sourced_ii": observed.aux_sourced_ii,
    }

    breakdown = (
        f"{internal_total:,.0f} I-I + {anchored_total:,.0f} anchored"
        if anchored_total
        else f"{internal_total:,.0f} I-I"
    )

    if result["within_tolerance"]:
        logger.info(
            f"  Reconciliation OK: matrix {matrix_total:,.0f} vs direct "
            f"aggregation {flow_total:,.0f} ({breakdown}) (delta {delta:+,.0f})"
        )
    else:
        logger.warning(
            f"  RECONCILIATION FAILED: matrix {matrix_total:,.0f} vs direct "
            f"aggregation {flow_total:,.0f} ({breakdown}) (delta {delta:+,.0f}). "
            f"Trips were lost or duplicated during zone assembly."
        )

    return result


def matrix_stats(matrix: pd.DataFrame,
                 home_residents: Optional[Dict[str, float]] = None,
                 work_jobs: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Shape, density and zero-margin statistics for an OD matrix (E2, E9).

    Args:
        matrix: zone × zone trip table.
        home_residents: optional {zone: residents} used to report how many
            residents are stranded in zero-rows. A zero-row means those
            residents generate no work trip at all.
        work_jobs: optional {zone: jobs} for the same check on zero-columns.

    Returns:
        The ``matrix`` block of od_matrix_diagnostics.json.
    """
    arr = matrix.to_numpy(dtype=float)
    n_cells = arr.size
    nonzero = int(np.count_nonzero(arr))

    row_sums = arr.sum(axis=1)
    col_sums = arr.sum(axis=0)
    zero_row_zones = [z for z, s in zip(matrix.index, row_sums) if s == 0]
    zero_col_zones = [z for z, s in zip(matrix.columns, col_sums) if s == 0]

    residents_stranded = (
        sum(float(home_residents.get(z, 0.0)) for z in zero_row_zones)
        if home_residents else None
    )
    jobs_stranded = (
        sum(float(work_jobs.get(z, 0.0)) for z in zero_col_zones)
        if work_jobs else None
    )

    stats: Dict[str, Any] = {
        "rows": int(matrix.shape[0]),
        "cols": int(matrix.shape[1]),
        "nonzero_pairs": nonzero,
        "density": round(nonzero / n_cells, 6) if n_cells else 0.0,
        "total_trips": float(arr.sum()),
        "zero_rows": len(zero_row_zones),
        "zero_cols": len(zero_col_zones),
        "residents_in_zero_rows": residents_stranded,
        "jobs_in_zero_cols": jobs_stranded,
    }

    logger.info(f"  Matrix: {stats['rows']:,} × {stats['cols']:,}, "
                f"{nonzero:,} non-zero pairs ({stats['density']:.1%} dense), "
                f"{stats['total_trips']:,.0f} trips")
    logger.info(f"  Zero-rows: {stats['zero_rows']:,} | Zero-cols: {stats['zero_cols']:,}")

    # A zero-row holding residents is the red flag: tolerable at a handful,
    # a structural problem at thousands.
    if residents_stranded:
        logger.warning(
            f"  {stats['zero_rows']:,} zero-row zone(s) hold {residents_stranded:,.0f} "
            f"employed residents who generate NO work trip"
        )
    if jobs_stranded:
        logger.warning(
            f"  {stats['zero_cols']:,} zero-column zone(s) hold {jobs_stranded:,.0f} "
            f"jobs that receive no commuter"
        )

    return stats
