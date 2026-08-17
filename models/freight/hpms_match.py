"""Match cordons to HPMS segments, so cordon weights come from observation.

**Why this exists.** ``assign_cordon_weights`` prefers an observed truck AADT per
cordon and falls back to ``capacity x CAPACITY_TO_DAILY_VOLUME x truck_share``.
Until this module existed nobody built the observed side, so *every* cordon took
the fallback: the spatial distribution of freight across gateways was driven by
road size rather than by measured truck traffic, and the regional total rested on
a design-hour estimate rather than on data. This module supplies the
``{link_id: truck_aadt}`` dict that closes that gap (design.md item 9).

**Why it is a geometry problem, not a join.** HPMS segments are LRS-referenced
polylines in EPSG:4326; MATSim links are straight segments in projected UTM
metres. There is no shared key. The match is therefore spatial, and it has to be
*bearing-aware*: a divided highway's two carriageways are metres apart and
antiparallel, so a nearest-only match assigns an inbound cordon the volume of the
outbound carriageway roughly half the time.

Three facts verified against the live service rather than assumed:

- ``returnGeometry=true`` with ``f=json`` is **rejected** by this service
  ("Invalid query parameters"). ``f=geojson`` returns the same features with
  usable ``LineString`` coordinates, so that is what we request.
- ``maxRecordCount`` is **2000**, so pagination is mandatory for any real region;
  a single unpaged query silently truncates.
- HPMS AADT is a **bidirectional** total on an undivided segment but is coded per
  carriageway on a divided one. There is no field that reliably says which, so
  see ``directional_aadt`` for how that is handled and why it is conservative.

Like every other network path in this package, failure here is never fatal: a
region with no reachable service, or no matching segments, falls back to the
capacity weighting that was the only behaviour before this module existed.

See docs/freight/design.md §9.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from models.freight.cordons import Cordon
from utils.logger import setup_logger

logger = setup_logger(__name__)


#: Fields fetched per segment. Kept minimal — geometry dominates the payload and
#: the service is slow enough that every extra field costs real time.
HPMS_FIELDS = ('AADT', 'AADT_SINGLE_UNIT', 'AADT_COMBINATION',
               'F_SYSTEM', 'URBAN_CODE')

#: The service caps a page at 2000 features regardless of what we ask for.
HPMS_PAGE_SIZE = 2000

#: How far from a cordon we look for an HPMS segment. A cordon sits on a node;
#: the HPMS centreline for the same road can sit tens of metres away where the
#: carriageways are drawn separately, and further where the LRS geometry is
#: generalised. 150 m comfortably exceeds a carriageway separation without
#: reaching a genuinely different road; ``counts_generator`` uses 60 m for point
#: sensors, but a cordon is matched to a *corridor*, not to a sensor location.
DEFAULT_MATCH_RADIUS_M = 150.0

#: How far two bearings may differ and still be "the same direction of travel".
#: 45 deg tolerates LRS generalisation and the angle between a MATSim link and
#: the HPMS centreline it represents, while still separating a carriageway from
#: its antiparallel partner (which differs by ~180 deg).
DEFAULT_BEARING_TOLERANCE_DEG = 45.0


class HPMSMatchError(RuntimeError):
    """Raised only for programming errors, never for a failed lookup."""


@dataclass
class HPMSSegment:
    """One HPMS polyline with its truck volumes, projected into network space.

    Attributes:
        points: the centreline in UTM metres, densified only as much as the
            source provides.
        truck_aadt: ``AADT_SINGLE_UNIT + AADT_COMBINATION``. The quantity tier 2
            compares against and the quantity a cordon is weighted by.
        aadt: total AADT, kept for the truck-share cross-check.
        f_system: HPMS functional class, used to reject a match between an
            interstate cordon and a collector segment.
    """
    points: np.ndarray
    truck_aadt: float
    aadt: float
    single_unit: float
    combination: float
    f_system: Optional[int] = None
    urban_code: Optional[int] = None

    @property
    def is_rural(self) -> bool:
        return self.urban_code == 99999

    def bearing_at(self, index: int) -> Optional[float]:
        """Bearing of the polyline segment starting at ``index``, grid-north."""
        if index < 0 or index + 1 >= len(self.points):
            return None
        (x1, y1), (x2, y2) = self.points[index], self.points[index + 1]
        return _bearing(x1, y1, x2, y2)


def _bearing(x1: float, y1: float, x2: float, y2: float) -> Optional[float]:
    """Bearing in degrees clockwise from grid north. None for zero length.

    Matches ``counts_generator._link_bearing`` so the two matchers agree about
    what a bearing means.
    """
    dx, dy = x2 - x1, y2 - y1
    if dx * dx + dy * dy < 1e-12:
        return None
    return float(np.degrees(np.arctan2(dx, dy)) % 360.0)


def _angular_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

class HPMSGeometryClient:
    """Fetches HPMS segments with geometry for a bounding box.

    Separate from ``truck_share.HPMSClient`` because the two ask the service
    genuinely different questions: that one wants a single statewide *statistic*
    and can use ``outStatistics`` server-side, this one wants every *feature*
    with its geometry and must paginate. Sharing a class would mean one object
    with two unrelated query paths.

    Failure-tolerant by contract: every method returns None or an empty list
    rather than raising, because a third-party service must not be able to stop
    a simulation.
    """

    DEFAULT_SERVICE_URL = (
        'https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/'
        'hpms_v2_view/FeatureServer/0'
    )

    def __init__(
        self,
        service_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        cache_days: int = 90,
        timeout_seconds: float = 120.0,
        enabled: bool = True,
        page_size: int = HPMS_PAGE_SIZE,
        max_pages: int = 40,
    ):
        """
        Args:
            timeout_seconds: geometry queries are far slower than the statistic
                query — measured at tens of seconds for a metro-sized bbox — so
                this defaults well above ``HPMSClient``'s 30 s.
            max_pages: a ceiling on pagination, so a mis-set bbox covering half
                a state cannot spin indefinitely. 40 pages x 2000 = 80k
                segments, comfortably more than any metro needs.
        """
        self.service_url = service_url or self.DEFAULT_SERVICE_URL
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_days = int(cache_days)
        self.timeout_seconds = float(timeout_seconds)
        self.enabled = bool(enabled)
        self.page_size = int(page_size)
        self.max_pages = int(max_pages)

    # -- cache ---------------------------------------------------------------

    def _cache_path(self, bbox: Tuple[float, float, float, float]) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        key = '_'.join(f"{v:.3f}" for v in bbox)
        return self.cache_dir / f"hpms_geom_{key}.json"

    def read_cache(self, bbox) -> Optional[List[Dict]]:
        """Cached raw features for this bbox, when fresh."""
        path = self._cache_path(bbox)
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            age_days = (time.time() - float(payload['fetched_at'])) / 86400.0
            if age_days > self.cache_days:
                logger.info(
                    f"HPMS geometry cache is {age_days:.0f} days old "
                    f"(limit {self.cache_days}); refetching")
                return None
            features = payload.get('features', [])
            # Entries written before this flag existed have no 'complete' key
            # and are treated as complete, which is what they were.
            if not payload.get('complete', True):
                logger.warning(
                    f"HPMS geometry cache holds an INCOMPLETE fetch "
                    f"({len(features):,} segments, truncated by a service "
                    f"error); refetching rather than serving a partial region")
                return None
            logger.info(
                f"HPMS geometry from cache: {len(features):,} segments "
                f"({age_days:.1f} days old)")
            return features
        except Exception as exc:  # noqa: BLE001 - a bad cache entry is not fatal
            logger.warning(f"Ignoring unreadable HPMS geometry cache {path}: {exc}")
            return None

    def write_cache(self, bbox, features: List[Dict],
                    complete: bool = True) -> None:
        """Store fetched features. Never raises.

        ``complete=False`` records that the fetch was truncated by a service
        error. The entry is still written — a partial region is better than
        none, and refetching 54,000 segments takes minutes — but it is marked
        so ``read_cache`` refuses to serve it as though it were the whole
        region.
        """
        path = self._cache_path(bbox)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(
                {'bbox': list(bbox), 'fetched_at': time.time(),
                 'complete': complete,
                 'features': features}), encoding='utf-8')
            state = '' if complete else ' (INCOMPLETE)'
            logger.info(
                f"Cached {len(features):,} HPMS segments to {path.name}{state}")
        except Exception as exc:  # noqa: BLE001 - caching must never fail a run
            logger.warning(f"Could not cache HPMS geometry (run unaffected): {exc}")

    # -- live query ----------------------------------------------------------

    def fetch_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        min_truck_aadt: float = 0.0,
    ) -> List[Dict]:
        """Every HPMS segment intersecting a lat/lon bbox, with geometry.

        Args:
            bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326.
            min_truck_aadt: skip segments carrying less truck traffic than this.
                A segment with no trucks cannot inform a truck weight, and
                dropping them server-side cuts the payload substantially.

        Returns:
            Raw geojson features. Empty on any failure — never raises.
        """
        cached = self.read_cache(bbox)
        if cached is not None:
            return cached

        if not self.enabled:
            logger.info("HPMS geometry disabled in config; using capacity weights")
            return []

        features: List[Dict] = []
        offset = 0
        complete = True

        for page in range(self.max_pages):
            batch = self._fetch_page(bbox, offset, min_truck_aadt)
            if batch is None:
                # A failed page mid-way still leaves usable data; keep what we
                # have rather than discarding a slow, expensive fetch. But the
                # result is now a truncated view of the region, and caching it
                # as though it were complete would hide that for `cache_days`
                # — measured on 15-county Twin Cities, a 400 at offset 54,000
                # cached 54k segments that every later run would have reused
                # believing the region fully covered.
                complete = False
                logger.warning(
                    f"HPMS geometry fetch failed at offset {offset}; continuing "
                    f"with the {len(features):,} segments already retrieved. "
                    f"This result is INCOMPLETE and will be cached as such, so "
                    f"the next run refetches rather than trusting it.")
                break
            features.extend(batch)
            if len(batch) < self.page_size:
                break
            offset += len(batch)
        else:
            complete = False
            logger.warning(
                f"HPMS geometry hit the {self.max_pages}-page ceiling; the "
                f"region may be larger than intended")

        if features:
            self.write_cache(bbox, features, complete=complete)
        else:
            logger.warning(
                "HPMS returned no segments with geometry for this region; "
                "cordon weights will fall back to capacity.")
        return features

    def _fetch_page(self, bbox, offset: int,
                    min_truck_aadt: float) -> Optional[List[Dict]]:
        """One page of features, or None on failure."""
        min_lon, min_lat, max_lon, max_lat = bbox
        envelope = {'xmin': min_lon, 'ymin': min_lat,
                    'xmax': max_lon, 'ymax': max_lat,
                    'spatialReference': {'wkid': 4326}}

        where = 'AADT > 0'
        if min_truck_aadt > 0:
            where += (f" AND (AADT_SINGLE_UNIT + AADT_COMBINATION) "
                      f">= {min_truck_aadt}")

        params = {
            'where': where,
            'geometry': json.dumps(envelope),
            'geometryType': 'esriGeometryEnvelope',
            'inSR': '4326',
            'spatialRel': 'esriSpatialRelIntersects',
            'outFields': ','.join(HPMS_FIELDS),
            'returnGeometry': 'true',
            'outSR': '4326',
            # Verified against the live service: returnGeometry=true with
            # f=json is rejected outright, while f=geojson serves the same
            # features with usable coordinates.
            'f': 'geojson',
            'resultOffset': offset,
            'resultRecordCount': self.page_size,
        }

        payload = self._request(f"{self.service_url}/query", params)
        if payload is None:
            return None
        return payload.get('features', [])

    def _request(self, url: str, params: Dict) -> Optional[Dict]:
        """One HTTP GET with a single retry. Returns parsed JSON or None."""
        # POST rather than GET: the envelope and field list push the query
        # string past what some proxies accept, and ArcGIS treats them alike.
        data = urllib.parse.urlencode(params).encode('utf-8')

        for attempt in (1, 2):
            try:
                request = urllib.request.Request(url, data=data)
                with urllib.request.urlopen(
                        request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode('utf-8'))
                # ArcGIS reports errors in a 200 body rather than an HTTP status.
                if 'error' in payload:
                    logger.warning(f"HPMS geometry service error: {payload['error']}")
                    return None
                return payload
            except Exception as exc:  # noqa: BLE001 - never fail a run
                if attempt == 1:
                    logger.debug(f"HPMS geometry request failed, retrying: {exc}")
                    continue
                logger.warning(
                    f"HPMS geometry unreachable after retry ({exc}); cordon "
                    f"weights will fall back to capacity. The run is unaffected.")
        return None


# ---------------------------------------------------------------------------
# Parsing and projection
# ---------------------------------------------------------------------------

def parse_segments(features: Sequence[Dict], converter) -> List[HPMSSegment]:
    """Turn geojson features into segments projected to network coordinates.

    A malformed feature is skipped rather than raising: HPMS is a national
    dataset assembled from 50 state submissions and a handful of bad geometries
    is expected, not exceptional.
    """
    segments: List[HPMSSegment] = []
    skipped = 0

    for feature in features:
        geometry = feature.get('geometry') or {}
        properties = feature.get('properties') or feature.get('attributes') or {}

        lines = _coordinate_lines(geometry)
        if not lines:
            skipped += 1
            continue

        try:
            single = float(properties.get('AADT_SINGLE_UNIT') or 0.0)
            combination = float(properties.get('AADT_COMBINATION') or 0.0)
            aadt = float(properties.get('AADT') or 0.0)
        except (TypeError, ValueError):
            skipped += 1
            continue

        truck_aadt = single + combination
        if truck_aadt <= 0:
            continue

        f_system = _int_or_none(properties.get('F_SYSTEM'))
        urban_code = _int_or_none(properties.get('URBAN_CODE'))

        for line in lines:
            projected = _project_line(line, converter)
            if projected is None or len(projected) < 2:
                skipped += 1
                continue
            segments.append(HPMSSegment(
                points=projected,
                truck_aadt=truck_aadt,
                aadt=aadt,
                single_unit=single,
                combination=combination,
                f_system=f_system,
                urban_code=urban_code,
            ))

    if skipped:
        logger.debug(f"Skipped {skipped} HPMS features with unusable geometry")
    logger.info(f"Parsed {len(segments):,} HPMS segments carrying truck traffic")
    return segments


def _coordinate_lines(geometry: Dict) -> List[List]:
    """Coordinate lists from either geojson or Esri geometry."""
    if 'coordinates' in geometry:
        kind = geometry.get('type')
        if kind == 'LineString':
            return [geometry['coordinates']]
        if kind == 'MultiLineString':
            return list(geometry['coordinates'])
        return []
    if 'paths' in geometry:  # Esri json, in case a caller supplies it
        return list(geometry['paths'])
    return []


def _project_line(line: Sequence, converter) -> Optional[np.ndarray]:
    """Project a lon/lat coordinate list into network UTM metres."""
    points = []
    for coordinate in line:
        try:
            lon, lat = float(coordinate[0]), float(coordinate[1])
        except (TypeError, ValueError, IndexError):
            return None
        x, y = converter.latlon_to_utm(lat, lon)
        points.append((x, y))
    return np.array(points, dtype=float) if len(points) >= 2 else None


def _int_or_none(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def link_geometry(network_path: Path,
                  link_ids: Optional[set] = None) -> Dict[str, Tuple[float, float, float, float]]:
    """Endpoint coordinates for the named links: ``{link_id: (x1,y1,x2,y2)}``.

    Reads with iterparse and keeps only the requested links, because the caller
    needs a few hundred cordon links out of a network that can hold 634,000.
    """
    network_path = Path(network_path)
    if not network_path.exists():
        raise HPMSMatchError(f"Network file not found: {network_path}")

    nodes: Dict[str, Tuple[float, float]] = {}
    pending: List[Tuple[str, str, str]] = []

    for _, elem in ET.iterparse(str(network_path), events=('end',)):
        if elem.tag == 'node':
            try:
                nodes[elem.get('id')] = (float(elem.get('x')), float(elem.get('y')))
            except (TypeError, ValueError):
                pass
            elem.clear()
        elif elem.tag == 'link':
            link_id = elem.get('id')
            if link_ids is None or link_id in link_ids:
                pending.append((link_id, elem.get('from'), elem.get('to')))
            elem.clear()

    geometry: Dict[str, Tuple[float, float, float, float]] = {}
    for link_id, from_node, to_node in pending:
        if from_node in nodes and to_node in nodes:
            (x1, y1), (x2, y2) = nodes[from_node], nodes[to_node]
            geometry[link_id] = (x1, y1, x2, y2)

    return geometry


def _nearest_point_on_segment(px, py, ax, ay, bx, by) -> Tuple[float, float]:
    """Distance from a point to a line segment, and the position along it."""
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.hypot(px - ax, py - ay), 0.0
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


def match_link_to_segment(
    link_xy: Tuple[float, float, float, float],
    segments: Sequence[HPMSSegment],
    radius_m: float = DEFAULT_MATCH_RADIUS_M,
    bearing_tolerance_deg: float = DEFAULT_BEARING_TOLERANCE_DEG,
    f_system: Optional[int] = None,
) -> Optional[Tuple[HPMSSegment, float, float]]:
    """Best HPMS segment for one MATSim link, or None.

    **Bearing is not optional here.** A divided highway's carriageways are a few
    metres apart and antiparallel; matching on distance alone gives an inbound
    cordon the outbound carriageway's volume about half the time, which would
    put the wrong truck total on the wrong side of the corridor with no symptom
    other than a poor tier-2 GEH.

    Returns ``(segment, distance_m, bearing_difference_deg)`` for the closest
    segment whose direction agrees, preferring the smallest distance among
    those that agree.
    """
    x1, y1, x2, y2 = link_xy
    link_bearing = _bearing(x1, y1, x2, y2)
    if link_bearing is None:
        return None

    midx, midy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    best: Optional[Tuple[HPMSSegment, float, float]] = None

    for segment in segments:
        if f_system is not None and segment.f_system is not None:
            # Do not let an interstate cordon take a collector's volume. One
            # class of slack is allowed because HPMS and OSM disagree at the
            # margin about where a freeway becomes an expressway.
            if abs(segment.f_system - f_system) > 1:
                continue

        points = segment.points
        for index in range(len(points) - 1):
            ax, ay = points[index]
            bx, by = points[index + 1]

            distance, _ = _nearest_point_on_segment(midx, midy, ax, ay, bx, by)
            if distance > radius_m:
                continue

            segment_bearing = segment.bearing_at(index)
            if segment_bearing is None:
                continue

            difference = _angular_diff(link_bearing, segment_bearing)
            # HPMS digitises a centreline in one direction only; on an undivided
            # road the same centreline serves both directions of travel, so a
            # segment drawn against our link is still the right segment. Accept
            # either orientation and record which it was.
            reversed_difference = 180.0 - difference
            aligned = min(difference, reversed_difference)
            if aligned > bearing_tolerance_deg:
                continue

            if best is None or distance < best[1]:
                best = (segment, distance, aligned)

    return best


def directional_aadt(segment: HPMSSegment, link_is_one_way: bool) -> float:
    """The truck AADT to credit one carriageway with.

    HPMS AADT is a **bidirectional** total on an undivided segment and a
    per-carriageway figure on a divided one, and no field reliably distinguishes
    the two. Guessing wrong in one direction double-counts a corridor; guessing
    wrong in the other halves it.

    The conservative reading is used: where our own network models the road as
    two one-way carriageways, the HPMS figure is treated as bidirectional and
    halved, so the two carriageways together reproduce the published total
    rather than twice it. Where the network models it as one two-way link, the
    figure is taken as-is.

    Over-counting is the worse error here, because it inflates the regional
    freight total that ``demand_scale`` then has to be calibrated against.
    """
    return segment.truck_aadt / 2.0 if link_is_one_way else segment.truck_aadt


def build_truck_aadt_by_link(
    cordons: Sequence[Cordon],
    segments: Sequence[HPMSSegment],
    link_xy: Dict[str, Tuple[float, float, float, float]],
    one_way_links: Optional[set] = None,
    radius_m: float = DEFAULT_MATCH_RADIUS_M,
    bearing_tolerance_deg: float = DEFAULT_BEARING_TOLERANCE_DEG,
) -> Tuple[Dict[str, float], Dict]:
    """The ``{link_id: truck_aadt}`` dict ``assign_cordon_weights`` wants.

    This is the seam item 9 exists to fill. ``assign_cordon_weights`` already
    prefers observed volume, falls back to capacity per cordon, and reports
    ``n_observed`` / ``n_fallback``; it just had nobody supplying the data.

    Returns the mapping plus statistics describing how well the match went, so
    a poor match is visible in ``freight_summary.json`` rather than showing up
    only as a bad tier-2 result.
    """
    one_way_links = one_way_links or set()
    truck_aadt: Dict[str, float] = {}

    distances: List[float] = []
    bearings: List[float] = []
    matched_cordons = 0
    matched_cordon_ids: List[str] = []

    for cordon in cordons:
        cordon_matched = False
        for link_id in cordon.link_ids:
            geometry = link_xy.get(link_id)
            if geometry is None:
                continue
            match = match_link_to_segment(
                geometry, segments,
                radius_m=radius_m,
                bearing_tolerance_deg=bearing_tolerance_deg,
            )
            if match is None:
                continue
            segment, distance, bearing_difference = match
            truck_aadt[link_id] = directional_aadt(
                segment, link_is_one_way=link_id in one_way_links)
            distances.append(distance)
            bearings.append(bearing_difference)
            cordon_matched = True
        if cordon_matched:
            matched_cordons += 1
            matched_cordon_ids.append(cordon.cordon_id)

    stats = {
        'n_segments': len(segments),
        'n_cordons': len(cordons),
        'n_cordons_matched': matched_cordons,
        'n_links_matched': len(truck_aadt),
        'match_rate': (round(matched_cordons / len(cordons), 3)
                       if cordons else 0.0),
        'radius_m': radius_m,
        'bearing_tolerance_deg': bearing_tolerance_deg,
        # Which cordons these are, not just how many. Calibration has to be
        # able to separate cordons carrying an *observed* truck volume from
        # those carrying a capacity estimate — measuring demand_scale against
        # the estimated ones would partly measure it against our own
        # assumption. Note the cordon screenline cannot judge demand_scale at
        # all (§5): that is the freight estimator's job, because it compares
        # against corridor volumes the demand was not derived from.
        'matched_cordon_ids': sorted(matched_cordon_ids),
        'observed_link_ids': sorted(truck_aadt),
    }
    if distances:
        stats['match_distance_m'] = {
            'median': round(float(np.median(distances)), 1),
            'max': round(float(np.max(distances)), 1),
        }
        stats['bearing_difference_deg'] = {
            'median': round(float(np.median(bearings)), 1),
            'max': round(float(np.max(bearings)), 1),
        }

    logger.info(
        f"HPMS match: {matched_cordons}/{len(cordons)} cordons matched an "
        f"observed truck volume ({len(truck_aadt)} links)")
    if cordons and matched_cordons == 0:
        logger.warning(
            "No cordon matched an HPMS segment. Every weight will fall back to "
            "capacity, exactly as before this stage existed. Check that the "
            "region's bbox is right and that HPMS covers these roads.")

    return truck_aadt, stats


def corridor_links(
    network_path: Path,
    min_freespeed_ms: float = 22.0,
    min_capacity_vph: float = 1500.0,
    mode: str = 'car',
) -> Dict[str, Tuple[float, float, float, float]]:
    """Every through-corridor link in the network, with its endpoints.

    Tier 2 compares truck volumes on *corridors*, not just at the cordons the
    demand was derived from — that independence is the whole point of it. The
    filter matches ``CordonDetector._is_corridor`` so both stages agree about
    what a corridor is.
    """
    network_path = Path(network_path)
    if not network_path.exists():
        raise HPMSMatchError(f"Network file not found: {network_path}")

    nodes: Dict[str, Tuple[float, float]] = {}
    pending: List[Tuple[str, str, str]] = []

    for _, elem in ET.iterparse(str(network_path), events=('end',)):
        if elem.tag == 'node':
            try:
                nodes[elem.get('id')] = (float(elem.get('x')), float(elem.get('y')))
            except (TypeError, ValueError):
                pass
            elem.clear()
        elif elem.tag == 'link':
            modes = [m.strip() for m in (elem.get('modes') or '').split(',')]
            if mode in modes:
                try:
                    freespeed = float(elem.get('freespeed') or 0.0)
                    capacity = float(elem.get('capacity') or 0.0)
                except (TypeError, ValueError):
                    freespeed = capacity = 0.0
                if freespeed >= min_freespeed_ms and capacity >= min_capacity_vph:
                    pending.append((elem.get('id'), elem.get('from'), elem.get('to')))
            elem.clear()

    geometry: Dict[str, Tuple[float, float, float, float]] = {}
    for link_id, from_node, to_node in pending:
        if from_node in nodes and to_node in nodes:
            (x1, y1), (x2, y2) = nodes[from_node], nodes[to_node]
            geometry[link_id] = (x1, y1, x2, y2)

    logger.info(f"Corridor links for tier 2: {len(geometry):,}")
    return geometry


def _segment_index(segments: Sequence[HPMSSegment], radius_m: float):
    """An rtree over every HPMS polyline sub-segment, or None if unavailable.

    Corridor-wide matching is O(links x segments x points) as a nested loop —
    roughly 10^10 operations on a 92k-link network against 7,358 segments, which
    is not viable. The index turns it into a bounded-box lookup per link.
    ``counts_generator`` already uses rtree for the same job on TMAS stations.
    """
    try:
        from rtree import index
    except ImportError:  # pragma: no cover - rtree is in requirements
        logger.warning("rtree unavailable; corridor matching will be skipped")
        return None, []

    entries: List[Tuple[HPMSSegment, int]] = []
    idx = index.Index()
    for segment in segments:
        points = segment.points
        for i in range(len(points) - 1):
            (ax, ay), (bx, by) = points[i], points[i + 1]
            idx.insert(len(entries), (min(ax, bx) - radius_m, min(ay, by) - radius_m,
                                      max(ax, bx) + radius_m, max(ay, by) + radius_m))
            entries.append((segment, i))
    return idx, entries


def match_corridor_links(
    link_xy: Dict[str, Tuple[float, float, float, float]],
    segments: Sequence[HPMSSegment],
    one_way_links: Optional[set] = None,
    radius_m: float = DEFAULT_MATCH_RADIUS_M,
    bearing_tolerance_deg: float = DEFAULT_BEARING_TOLERANCE_DEG,
) -> Tuple[Dict[str, float], Dict]:
    """``{link_id: observed truck AADT}`` for every corridor link that matches.

    This is what tier 2 validates against. Unlike the cordon matcher it is
    spatially indexed, because the link count is three orders of magnitude
    larger.

    The same bearing rule applies and for the same reason: a divided highway's
    carriageways are metres apart and antiparallel, so distance alone assigns
    half of them the opposing direction's volume.
    """
    one_way_links = one_way_links or set()
    idx, entries = _segment_index(segments, radius_m)
    if idx is None:
        return {}, {'error': 'rtree unavailable'}

    truck_aadt: Dict[str, float] = {}
    distances: List[float] = []
    # Which links landed on the same HPMS segment. A real network splits one
    # HPMS segment into many links — carriageways, ramps and consecutive
    # pieces — and each of them is credited the segment's *whole* AADT below.
    # That is correct per link (each carriageway does carry that flow) but it
    # means the caller must never sum observed volume over links: measured on
    # Anoka, 2,503 matched links map to 884 segments, so a naive sum inflates
    # the observed total 2.8x. `segment_groups` is what lets the caller
    # aggregate per segment instead. See design.md §7.
    segment_groups: Dict[int, Dict] = {}

    for link_id, (x1, y1, x2, y2) in link_xy.items():
        link_bearing = _bearing(x1, y1, x2, y2)
        if link_bearing is None:
            continue
        midx, midy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        best: Optional[Tuple[HPMSSegment, float]] = None
        for entry_id in idx.intersection((midx, midy, midx, midy)):
            segment, i = entries[entry_id]
            (ax, ay), (bx, by) = segment.points[i], segment.points[i + 1]
            distance, _ = _nearest_point_on_segment(midx, midy, ax, ay, bx, by)
            if distance > radius_m:
                continue
            segment_bearing = segment.bearing_at(i)
            if segment_bearing is None:
                continue
            difference = _angular_diff(link_bearing, segment_bearing)
            if min(difference, 180.0 - difference) > bearing_tolerance_deg:
                continue
            if best is None or distance < best[1]:
                best = (segment, distance)

        if best is not None:
            segment, distance = best
            truck_aadt[link_id] = directional_aadt(
                segment, link_is_one_way=link_id in one_way_links)
            distances.append(distance)
            group = segment_groups.setdefault(
                id(segment),
                {'truck_aadt': float(segment.truck_aadt), 'link_ids': []})
            group['link_ids'].append(link_id)

    stats = {
        'n_corridor_links': len(link_xy),
        'n_matched': len(truck_aadt),
        'match_rate': (round(len(truck_aadt) / len(link_xy), 3)
                       if link_xy else 0.0),
        'radius_m': radius_m,
        'bearing_tolerance_deg': bearing_tolerance_deg,
        'n_segments_matched': len(segment_groups),
        'links_per_segment': (round(len(truck_aadt) / len(segment_groups), 2)
                              if segment_groups else 0.0),
        # Keyed by an arbitrary index rather than id(), which is not stable
        # across processes and must not reach a JSON file.
        'segment_groups': [
            {'truck_aadt': g['truck_aadt'], 'link_ids': g['link_ids']}
            for g in segment_groups.values()
        ],
    }
    if distances:
        stats['match_distance_m'] = {
            'median': round(float(np.median(distances)), 1),
            'max': round(float(np.max(distances)), 1),
        }
    logger.info(
        f"Tier-2 corridor match: {len(truck_aadt):,}/{len(link_xy):,} links "
        f"carry an observed truck AADT, over {len(segment_groups):,} distinct "
        f"HPMS segments ({stats['links_per_segment']:.1f} links per segment)")
    return truck_aadt, stats


def cordon_bbox(
    cordons: Sequence[Cordon],
    converter,
    pad_km: float = 5.0,
) -> Tuple[float, float, float, float]:
    """Lat/lon bounding box covering every cordon, with padding.

    Padded because a cordon sits on a node while the HPMS centreline for the
    same road may lie outside the tight hull of those nodes.
    """
    if not cordons:
        raise HPMSMatchError("Cannot build a bbox with no cordons")

    lons, lats = [], []
    for cordon in cordons:
        lat, lon = converter.utm_to_latlon(cordon.x, cordon.y)
        lats.append(lat)
        lons.append(lon)

    # Degrees per km varies with latitude for longitude but not for latitude.
    mean_lat = float(np.mean(lats))
    lat_pad = pad_km / 111.0
    lon_pad = pad_km / max(1.0, 111.0 * math.cos(math.radians(mean_lat)))

    return (min(lons) - lon_pad, min(lats) - lat_pad,
            max(lons) + lon_pad, max(lats) + lat_pad)


def resolve_truck_aadt_by_link(
    config: Dict,
    cordons: Sequence[Cordon],
    network_path: Path,
    converter,
) -> Tuple[Dict[str, float], Dict]:
    """End-to-end: cordons in, ``{link_id: truck_aadt}`` out.

    Total by contract — any failure returns an empty mapping and a stats dict
    saying why, which puts every cordon on the capacity fallback that was the
    only behaviour before this stage.
    """
    freight_config = config.get('freight', {})
    hpms_config = freight_config.get('hpms', {})
    match_config = freight_config.get('hpms_match', {})

    if not hpms_config.get('enabled', True):
        return {}, {'skipped': 'freight.hpms.enabled is false'}

    try:
        bbox = cordon_bbox(cordons, converter,
                           pad_km=float(match_config.get('bbox_pad_km', 5.0)))

        client = HPMSGeometryClient(
            service_url=hpms_config.get('service_url'),
            cache_dir=_cache_dir(config),
            cache_days=hpms_config.get('cache_days', 90),
            timeout_seconds=match_config.get('timeout_seconds', 120),
            enabled=True,
        )
        features = client.fetch_bbox(
            bbox, min_truck_aadt=float(match_config.get('min_truck_aadt', 0.0)))
        if not features:
            return {}, {'bbox': [round(v, 4) for v in bbox],
                        'note': 'no HPMS segments returned'}

        segments = parse_segments(features, converter)
        if not segments:
            return {}, {'bbox': [round(v, 4) for v in bbox],
                        'note': 'no HPMS segments carried truck traffic'}

        wanted = {link_id for cordon in cordons for link_id in cordon.link_ids}
        geometry = link_geometry(Path(network_path), link_ids=wanted)
        one_way = _one_way_links(Path(network_path), wanted)

        truck_aadt, stats = build_truck_aadt_by_link(
            cordons, segments, geometry,
            one_way_links=one_way,
            radius_m=float(match_config.get('radius_m', DEFAULT_MATCH_RADIUS_M)),
            bearing_tolerance_deg=float(match_config.get(
                'bearing_tolerance_deg', DEFAULT_BEARING_TOLERANCE_DEG)),
        )
        stats['bbox'] = [round(v, 4) for v in bbox]
        return truck_aadt, stats

    except Exception as exc:  # noqa: BLE001 - must never fail a run
        logger.warning(
            f"HPMS link matching failed ({exc}); cordon weights fall back to "
            f"capacity. The run is unaffected.")
        return {}, {'error': str(exc)}


def resolve_corridor_truck_aadt(
    config: Dict,
    cordons: Sequence[Cordon],
    network_path: Path,
    converter,
) -> Tuple[Dict[str, float], Dict]:
    """Observed truck AADT per corridor link, for tier-2 validation.

    Uses the same bbox and cached HPMS fetch as the cordon matcher, so running
    both costs one download rather than two.

    Total by contract, like every other network path here: any failure returns
    an empty mapping and a reason, and tier 2 then reports that it cannot run
    rather than the run failing.
    """
    freight_config = config.get('freight', {})
    hpms_config = freight_config.get('hpms', {})
    match_config = freight_config.get('hpms_match', {})
    cordon_config = freight_config.get('cordon', {})

    if not hpms_config.get('enabled', True):
        return {}, {'skipped': 'freight.hpms.enabled is false'}

    try:
        bbox = cordon_bbox(cordons, converter,
                           pad_km=float(match_config.get('bbox_pad_km', 5.0)))
        client = HPMSGeometryClient(
            service_url=hpms_config.get('service_url'),
            cache_dir=_cache_dir(config),
            cache_days=hpms_config.get('cache_days', 90),
            timeout_seconds=match_config.get('timeout_seconds', 120),
            enabled=True,
        )
        features = client.fetch_bbox(
            bbox, min_truck_aadt=float(match_config.get('min_truck_aadt', 0.0)))
        if not features:
            return {}, {'note': 'no HPMS segments returned'}

        segments = parse_segments(features, converter)
        if not segments:
            return {}, {'note': 'no HPMS segments carried truck traffic'}

        links = corridor_links(
            Path(network_path),
            min_freespeed_ms=float(cordon_config.get('min_freespeed_ms', 22.0)),
            min_capacity_vph=float(cordon_config.get('min_capacity_vph', 1500.0)),
        )
        one_way = _one_way_links(Path(network_path), set(links))

        truck_aadt, stats = match_corridor_links(
            links, segments,
            one_way_links=one_way,
            radius_m=float(match_config.get('radius_m', DEFAULT_MATCH_RADIUS_M)),
            bearing_tolerance_deg=float(match_config.get(
                'bearing_tolerance_deg', DEFAULT_BEARING_TOLERANCE_DEG)),
        )
        stats['bbox'] = [round(v, 4) for v in bbox]
        return truck_aadt, stats

    except Exception as exc:  # noqa: BLE001 - must never fail a run
        logger.warning(
            f"Tier-2 corridor matching failed ({exc}); tier 2 will report that "
            f"it could not run. The run is unaffected.")
        return {}, {'error': str(exc)}


def _one_way_links(network_path: Path, link_ids: set) -> set:
    """Which of these links have no reverse twin — i.e. a divided carriageway.

    A link whose ``(to, from)`` node pair also exists as a link is one direction
    of a two-way road drawn as two links; a link with no such twin is a
    carriageway of a divided road. That distinction decides whether an HPMS
    bidirectional total should be halved (see ``directional_aadt``).
    """
    pairs: Dict[Tuple[str, str], str] = {}
    wanted: Dict[str, Tuple[str, str]] = {}

    for _, elem in ET.iterparse(str(network_path), events=('end',)):
        if elem.tag == 'link':
            link_id = elem.get('id')
            from_node, to_node = elem.get('from'), elem.get('to')
            if from_node and to_node:
                pairs[(from_node, to_node)] = link_id
                if link_id in link_ids:
                    wanted[link_id] = (from_node, to_node)
            elem.clear()

    return {link_id for link_id, (f, t) in wanted.items()
            if (t, f) not in pairs}


def _cache_dir(config: Dict) -> Optional[Path]:
    """Where HPMS responses are cached, from the freight config."""
    cache_dir = config.get('freight', {}).get('hpms', {}).get('cache_dir')
    return Path(cache_dir) if cache_dir else None
