"""Truck share — how much of a road's traffic is freight, per region.

The number this module resolves is the single most region-dependent input to the
whole freight model. Measured from live HPMS while designing this: urban
interstate truck share is 16.34% in Alabama, 5.84% in Minnesota and 9.23% in New
York. **A factor of three between regions**, so one global default would be
wrong nearly everywhere.

Resolution order, degrading gracefully — no layer may ever fail a run:

  ``config_pinned`` -> ``hpms_cache`` -> ``hpms_live`` -> ``national_table``

The order is deliberate. The vendored national table (layer 4) is written first
and is always present, so a region with no network access still gets a
defensible number. Every other layer is an improvement on it, and each records
which one actually supplied the value, because a figure derived from a 1997
national average must never be mistaken for a measured one.

See docs/freight/design.md §3.
"""

from __future__ import annotations

import json
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from utils.logger import setup_logger

logger = setup_logger(__name__)


# Where a resolved truck share came from. Recorded in freight_summary.json so a
# national-average result is never mistaken for a measured one.
SOURCE_CONFIG_PINNED = 'config_pinned'
SOURCE_HPMS_CACHE = 'hpms_cache'
SOURCE_HPMS_LIVE = 'hpms_live'
SOURCE_NATIONAL_TABLE = 'national_table'


# ---------------------------------------------------------------------------
# Layer 4: the vendored national table
# ---------------------------------------------------------------------------

# FHWA/LTPP, Hallenbeck et al., "Vehicle Volume Distributions by
# Classification" (1997), Table 4 — national truck percentages by functional
# class, from 99 sites in 19 states.
#
# On the 1997 vintage: this is the *fallback*, not the primary. Its role is to
# produce a sane starting number that calibration then moves away from. The
# structural pattern it encodes is stable even as levels shift — rural
# interstates carrying roughly three times the truck share of urban minor
# arterials is a consequence of road function, not of any year's freight
# economy. Our own live HPMS queries corroborate the order of magnitude
# (Minnesota rural interstate 23.11% against the table's 24.5%).
#
# The honest caveat: Alabama urban interstate measures 16.34% against the
# table's 8.5%. **National averages understate freight-heavy regions
# substantially**, which is exactly why HPMS is the primary source.
#
# Keyed by HPMS F_SYSTEM code. single_unit = FHWA classes 4-7, combination =
# classes 8-13.
NATIONAL_TRUCK_SHARES: Dict[Tuple[int, bool], Dict[str, float]] = {
    # (f_system, is_rural): shares
    (1, True):  {'single_unit': 0.035, 'combination': 0.210},  # rural interstate
    (2, True):  {'single_unit': 0.037, 'combination': 0.080},  # rural principal arterial
    (3, True):  {'single_unit': 0.035, 'combination': 0.060},  # rural other principal arterial
    (4, True):  {'single_unit': 0.033, 'combination': 0.053},  # rural minor arterial
    (5, True):  {'single_unit': 0.030, 'combination': 0.040},  # rural major collector
    (6, True):  {'single_unit': 0.028, 'combination': 0.030},  # rural minor collector
    (7, True):  {'single_unit': 0.025, 'combination': 0.020},  # rural local
    (1, False): {'single_unit': 0.026, 'combination': 0.059},  # urban interstate
    (2, False): {'single_unit': 0.037, 'combination': 0.074},  # urban freeway
    (3, False): {'single_unit': 0.032, 'combination': 0.064},  # urban principal arterial
    (4, False): {'single_unit': 0.024, 'combination': 0.032},  # urban minor arterial
    (5, False): {'single_unit': 0.022, 'combination': 0.024},  # urban major collector
    (6, False): {'single_unit': 0.020, 'combination': 0.018},  # urban minor collector
    (7, False): {'single_unit': 0.018, 'combination': 0.012},  # urban local
}

# Used when a region reports no usable functional class at all. Urban
# interstate: the class a boundary freight corridor is most likely to be.
_DEFAULT_CLASS = (1, False)


@dataclass
class TruckShare:
    """A resolved truck share, and the provenance that makes it auditable.

    Attributes:
        single_unit: share of traffic that is single-unit truck (classes 4-7).
        combination: share that is combination truck (classes 8-13).
        source: which layer supplied this. One of the SOURCE_* constants.
        f_system: the functional class it was resolved for, when applicable.
        is_rural: whether that class was the rural variant.
        detail: free-form provenance for the summary (station counts, the
            HPMS query used, cache age).
    """
    single_unit: float
    combination: float
    source: str
    f_system: Optional[int] = None
    is_rural: Optional[bool] = None
    detail: Dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        """Share of traffic that is any kind of truck."""
        return self.single_unit + self.combination

    @property
    def mix(self) -> Dict[str, float]:
        """The split between the two truck types, normalised to 1.

        This is ``vehicle_mix``: while PCE is off it only affects reporting;
        once PCE is on it determines the capacity each truck consumes.
        """
        if self.total <= 0:
            return {'single_unit': 0.5, 'combination': 0.5}
        return {'single_unit': self.single_unit / self.total,
                'combination': self.combination / self.total}

    def to_dict(self) -> Dict:
        return {
            'single_unit': round(self.single_unit, 5),
            'combination': round(self.combination, 5),
            'total': round(self.total, 5),
            'mix': {k: round(v, 4) for k, v in self.mix.items()},
            'source': self.source,
            'f_system': self.f_system,
            'is_rural': self.is_rural,
            'detail': self.detail,
        }


def national_truck_share(f_system: int, is_rural: bool) -> TruckShare:
    """Look up the vendored national table. Cannot fail, needs no network."""
    key = (int(f_system), bool(is_rural))
    shares = NATIONAL_TRUCK_SHARES.get(key)
    if shares is None:
        logger.warning(
            f"No national truck share for f_system={f_system} "
            f"rural={is_rural}; falling back to urban interstate"
        )
        key = _DEFAULT_CLASS
        shares = NATIONAL_TRUCK_SHARES[key]

    return TruckShare(
        single_unit=shares['single_unit'],
        combination=shares['combination'],
        source=SOURCE_NATIONAL_TABLE,
        f_system=key[0],
        is_rural=key[1],
        detail={'table': 'FHWA/LTPP 1997 Table 4'},
    )


# ---------------------------------------------------------------------------
# The offline path: functional class from the .STA file already on disk
# ---------------------------------------------------------------------------

def _parse_f_system(raw: str) -> Optional[Tuple[int, bool]]:
    """Turn a TMAS ``f_system`` value into (code, is_rural).

    The field is a functional-class digit with a U/R suffix — ``1U`` is urban
    interstate, ``3R`` rural principal arterial. Verified against the real file:
    Alabama's 900 station-directions use 1U/1R/3U/3R/4U/4R/5U/5R/2U/7U.
    """
    if not raw:
        return None
    text = str(raw).strip().upper()
    if not text:
        return None

    digits = ''.join(c for c in text if c.isdigit())
    if not digits:
        return None

    is_rural = text.endswith('R')
    try:
        return int(digits), is_rural
    except ValueError:
        return None


def dominant_functional_class(
    station_zip: Path,
    state_abbr: str,
    county_codes: Iterable[str],
    year: int = 2024,
) -> Optional[Tuple[int, bool, Dict]]:
    """Find a region's most common functional class from the TMAS station file.

    This is the offline path, and it needs **no download at all**: the station
    file is national and is already in ``data/FHA_counts/``. Combined with the
    vendored table it produces a defensible truck share for any US region with
    zero network access.

    Returns (f_system, is_rural, detail), or None when the region has no
    stations in the file.
    """
    station_zip = Path(station_zip)
    if not station_zip.exists():
        logger.warning(f"TMAS station file not found: {station_zip}")
        return None

    entry = f"{state_abbr}_{year} (TMAS).STA"
    counties = {str(c).strip().zfill(3) for c in county_codes}

    try:
        import pandas as pd
        with zipfile.ZipFile(station_zip) as zf:
            with zf.open(entry) as handle:
                frame = pd.read_csv(handle, sep='|', dtype=str,
                                    encoding='utf-8', on_bad_lines='skip')
    except KeyError:
        logger.warning(f"TMAS station file has no entry '{entry}'")
        return None
    except Exception as exc:  # noqa: BLE001 - the offline path must not fail a run
        logger.warning(f"Could not read TMAS station file: {exc}")
        return None

    if 'county_code' not in frame.columns or 'f_system' not in frame.columns:
        logger.warning("TMAS station file lacks county_code/f_system columns")
        return None

    regional = frame[frame['county_code'].str.strip().str.zfill(3).isin(counties)]
    if regional.empty:
        logger.warning(
            f"No TMAS stations in counties {sorted(counties)} of {state_abbr}"
        )
        return None

    counts: Counter = Counter()
    for raw in regional['f_system']:
        parsed = _parse_f_system(raw)
        if parsed is not None:
            counts[parsed] += 1

    if not counts:
        logger.warning("TMAS stations carry no parseable f_system values")
        return None

    # Weight by how many station-directions report each class: the class most
    # of the region's monitored road-miles belong to is the right default.
    (f_system, is_rural), n = counts.most_common(1)[0]
    detail = {
        'method': 'tmas_sta_f_system',
        'state': state_abbr,
        'counties': sorted(counties),
        'n_stations': int(sum(counts.values())),
        'n_dominant': int(n),
        'distribution': {f"{code}{'R' if rural else 'U'}": int(c)
                         for (code, rural), c in sorted(counts.items())},
    }
    logger.info(
        f"TMAS f_system for {state_abbr} {sorted(counties)}: "
        f"{f_system}{'R' if is_rural else 'U'} "
        f"({n}/{sum(counts.values())} station-directions)"
    )
    return f_system, is_rural, detail


# ---------------------------------------------------------------------------
# Layers 2-3: HPMS cache and live query
# ---------------------------------------------------------------------------

class HPMSClient:
    """Queries FHWA's Highway Performance Monitoring System for truck AADT.

    HPMS publishes, for every Federal-Aid highway segment in the United States,
    ``AADT``, ``AADT_SINGLE_UNIT`` and ``AADT_COMBINATION`` alongside
    ``F_SYSTEM`` and ``URBAN_CODE``. That makes it the one source that gives a
    *region-specific, current* truck share anywhere in the country — including
    where classification counts do not exist, which is most of Alabama.

    Every method here is failure-tolerant by contract: a query that cannot be
    served logs and returns None, so the caller falls through to the national
    table. The precedent is ``run_experiment.py``'s demand cache, which wraps
    itself in ``except Exception`` with the comment "caching must never fail a
    run". Same principle: a third-party service being down is not a reason for
    a simulation to stop.
    """

    #: FHWA's own REST endpoint (geo.dot.gov) is blocked from our sandbox, as is
    #: fhwa.dot.gov. The ArcGIS-hosted mirror is reachable and is what we use.
    DEFAULT_SERVICE_URL = (
        'https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/'
        'hpms_v2_view/FeatureServer/0'
    )

    def __init__(
        self,
        service_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        cache_days: int = 90,
        timeout_seconds: float = 30.0,
        enabled: bool = True,
    ):
        """
        Args:
            service_url: the feature service to query. Config, not a constant,
                because the service is *replaceable*: the same data exists as a
                downloadable national geodatabase if the hosted layer goes away.
            cache_dir: where successful responses are stored.
            cache_days: how long a cached response stays valid. Regional truck
                percentages move on an annual data cycle, so 90 days costs
                nothing in accuracy.
            timeout_seconds: per-request timeout.
            enabled: false forces the offline path.
        """
        self.service_url = service_url or self.DEFAULT_SERVICE_URL
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_days = int(cache_days)
        self.timeout_seconds = float(timeout_seconds)
        self.enabled = bool(enabled)

    # -- cache ---------------------------------------------------------------

    def _cache_path(self, state_fips: str, f_system: int, is_rural: bool) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        suffix = 'R' if is_rural else 'U'
        return self.cache_dir / f"hpms_{state_fips}_{f_system}{suffix}.json"

    def read_cache(self, state_fips: str, f_system: int,
                   is_rural: bool) -> Optional[TruckShare]:
        """Return a cached share when one is present and still fresh."""
        path = self._cache_path(state_fips, f_system, is_rural)
        if path is None or not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            age_days = (time.time() - float(payload['fetched_at'])) / 86400.0
            if age_days > self.cache_days:
                logger.info(
                    f"HPMS cache for {state_fips} f_system={f_system} is "
                    f"{age_days:.0f} days old (limit {self.cache_days}); refetching"
                )
                return None
            return TruckShare(
                single_unit=float(payload['single_unit']),
                combination=float(payload['combination']),
                source=SOURCE_HPMS_CACHE,
                f_system=f_system,
                is_rural=is_rural,
                detail={**payload.get('detail', {}),
                        'cache_age_days': round(age_days, 1)},
            )
        except Exception as exc:  # noqa: BLE001 - a bad cache entry is not fatal
            logger.warning(f"Ignoring unreadable HPMS cache entry {path}: {exc}")
            return None

    def write_cache(self, state_fips: str, share: TruckShare) -> None:
        """Store a successful query. Never raises."""
        path = self._cache_path(state_fips, share.f_system, bool(share.is_rural))
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                'state_fips': state_fips,
                'f_system': share.f_system,
                'is_rural': share.is_rural,
                'single_unit': share.single_unit,
                'combination': share.combination,
                'fetched_at': time.time(),
                'detail': share.detail,
            }, indent=2), encoding='utf-8')
            logger.debug(f"Cached HPMS truck share to {path}")
        except Exception as exc:  # noqa: BLE001 - caching must never fail a run
            logger.warning(f"Could not cache HPMS response (run unaffected): {exc}")

    # -- live query ----------------------------------------------------------

    def query_live(self, state_fips: str, f_system: int,
                   is_rural: bool) -> Optional[TruckShare]:
        """Ask HPMS for the truck share of one functional class in one state.

        Returns None on any failure — unreachable service, bad response, no
        matching segments. Never raises: the caller has a working fallback and
        a live service must not be able to stop a run.
        """
        if not self.enabled:
            return None

        # URBAN_CODE 99999 means rural; anything else is an urban area code.
        urban_clause = ('URBAN_CODE = 99999' if is_rural
                        else 'URBAN_CODE <> 99999')
        where = (f"State_Code = {int(state_fips)} AND F_SYSTEM = {int(f_system)} "
                 f"AND {urban_clause} AND AADT > 0")

        params = {
            'where': where,
            'outFields': 'AADT,AADT_SINGLE_UNIT,AADT_COMBINATION',
            'outStatistics': json.dumps([
                {'statisticType': 'sum', 'onStatisticField': 'AADT',
                 'outStatisticFieldName': 'total_aadt'},
                {'statisticType': 'sum', 'onStatisticField': 'AADT_SINGLE_UNIT',
                 'outStatisticFieldName': 'total_su'},
                {'statisticType': 'sum', 'onStatisticField': 'AADT_COMBINATION',
                 'outStatisticFieldName': 'total_cu'},
                {'statisticType': 'count', 'onStatisticField': 'AADT',
                 'outStatisticFieldName': 'n_segments'},
            ]),
            'f': 'json',
        }

        payload = self._request(f"{self.service_url}/query", params)
        if payload is None:
            return None

        try:
            attributes = payload['features'][0]['attributes']
            total = float(attributes['total_aadt'] or 0.0)
            single_unit = float(attributes['total_su'] or 0.0)
            combination = float(attributes['total_cu'] or 0.0)
            n_segments = int(attributes.get('n_segments') or 0)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning(f"HPMS response not in the expected shape: {exc}")
            return None

        if total <= 0:
            logger.warning(
                f"HPMS returned no traffic for state={state_fips} "
                f"f_system={f_system} rural={is_rural}"
            )
            return None

        share = TruckShare(
            single_unit=single_unit / total,
            combination=combination / total,
            source=SOURCE_HPMS_LIVE,
            f_system=f_system,
            is_rural=is_rural,
            detail={
                'method': 'hpms_feature_service',
                'service_url': self.service_url,
                'n_segments': n_segments,
                'total_aadt': round(total, 1),
                'where': where,
            },
        )
        logger.info(
            f"HPMS live: state={state_fips} f_system={f_system}"
            f"{'R' if is_rural else 'U'} -> {share.total * 100:.2f}% trucks "
            f"({n_segments:,} segments)"
        )
        return share

    def _request(self, url: str, params: Dict) -> Optional[Dict]:
        """One HTTP GET with a single retry. Returns parsed JSON or None."""
        import urllib.error
        import urllib.parse
        import urllib.request

        query = urllib.parse.urlencode(params)
        full_url = f"{url}?{query}"

        for attempt in (1, 2):
            try:
                with urllib.request.urlopen(full_url,
                                            timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode('utf-8'))
                # ArcGIS reports errors in a 200 body rather than an HTTP status.
                if 'error' in payload:
                    logger.warning(f"HPMS service error: {payload['error']}")
                    return None
                return payload
            except Exception as exc:  # noqa: BLE001 - never fail a run
                if attempt == 1:
                    logger.debug(f"HPMS request failed, retrying once: {exc}")
                    continue
                logger.warning(
                    f"HPMS unreachable after retry ({exc}); falling back to the "
                    f"national table. The run is unaffected."
                )
        return None


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------

def resolve_truck_share(
    config: Dict,
    state_fips: str,
    county_codes: Iterable[str],
    state_abbr: Optional[str] = None,
    station_zip: Optional[Path] = None,
) -> TruckShare:
    """Resolve a region's truck share through the four-layer fallback.

    This function is total: it always returns a usable TruckShare, whatever the
    state of the network or the data directory. Which layer supplied the value
    is recorded on the result, and belongs in ``freight_summary.json``.

    Args:
        config: the whole config dict; reads the ``freight`` section.
        state_fips: 2-digit state FIPS code.
        county_codes: the region's 3-digit county codes.
        state_abbr: 2-letter state code, for the TMAS station file.
        station_zip: path to the TMAS station zip. Defaults to the standard
            location under the configured data directory.

    Returns:
        A TruckShare, never None.
    """
    freight_config = config.get('freight', {})
    hpms_config = freight_config.get('hpms', {})

    # Layer 1: a pinned value in the config. Once a region is calibrated, this
    # makes reproducing a published experiment need no network at all.
    pinned = freight_config.get('truck_share')
    if pinned is not None:
        mix = freight_config.get('vehicle_mix', {})
        single_share = float(mix.get('single_unit', 0.45))
        logger.info(f"Truck share pinned in config: {float(pinned) * 100:.2f}%")
        return TruckShare(
            single_unit=float(pinned) * single_share,
            combination=float(pinned) * (1.0 - single_share),
            source=SOURCE_CONFIG_PINNED,
            detail={'note': 'freight.truck_share set in config'},
        )

    # Which functional class are we asking about? The offline path answers this
    # from a file we already have, so it runs first and its answer is reused by
    # the HPMS layers as the query key.
    f_system, is_rural = _DEFAULT_CLASS
    class_detail: Dict = {'method': 'default_urban_interstate'}

    if station_zip is None and state_abbr:
        data_dir = config.get('data', {}).get('data_dir')
        if data_dir:
            station_zip = Path(data_dir) / 'FHA_counts' / '2024_station_data.zip'

    if station_zip and state_abbr:
        resolved = dominant_functional_class(station_zip, state_abbr, county_codes)
        if resolved is not None:
            f_system, is_rural, class_detail = resolved

    client = HPMSClient(
        service_url=hpms_config.get('service_url'),
        cache_dir=_hpms_cache_dir(config),
        cache_days=hpms_config.get('cache_days', 90),
        timeout_seconds=hpms_config.get('timeout_seconds', 30),
        enabled=hpms_config.get('enabled', True),
    )

    # Layer 2: the on-disk cache.
    cached = client.read_cache(state_fips, f_system, is_rural)
    if cached is not None:
        cached.detail.setdefault('functional_class', class_detail)
        logger.info(f"Truck share from HPMS cache: {cached.total * 100:.2f}%")
        return cached

    # Layer 3: a live query, attempted only on a cache miss.
    live = client.query_live(state_fips, f_system, is_rural)
    if live is not None:
        live.detail.setdefault('functional_class', class_detail)
        client.write_cache(state_fips, live)
        return live

    # Layer 4: the vendored table. Cannot fail.
    fallback = national_truck_share(f_system, is_rural)
    fallback.detail['functional_class'] = class_detail
    logger.info(
        f"Truck share from the national table: {fallback.total * 100:.2f}% "
        f"(f_system={f_system}{'R' if is_rural else 'U'}). This is a 1997 "
        f"national average, not a measurement for this region."
    )
    return fallback


def _hpms_cache_dir(config: Dict) -> Optional[Path]:
    """Resolve the HPMS cache directory against the configured data dir."""
    hpms_config = config.get('freight', {}).get('hpms', {})
    cache_dir = hpms_config.get('cache_dir', 'data/hpms')
    path = Path(cache_dir)
    if path.is_absolute():
        return path
    data_dir = config.get('data', {}).get('data_dir')
    if data_dir:
        # 'data/hpms' under a configured data_dir that already ends in 'data'
        # would nest badly; take the leaf name in that case.
        return Path(data_dir) / path.name
    return path
