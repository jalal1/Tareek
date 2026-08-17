"""Smoke tests for freight truck-share resolution.

No network and no data files: the HPMS layers are exercised through a stub
client, and the offline path through a synthetic .STA zip. The contract these
guard is that **no layer may ever fail a run** — every failure mode has to fall
through to the vendored national table rather than raise.

See docs/freight/design.md §3.
"""

import json
import time
import zipfile

import pytest

from models.freight.truck_share import (
    NATIONAL_TRUCK_SHARES,
    SOURCE_CONFIG_PINNED,
    SOURCE_HPMS_CACHE,
    SOURCE_HPMS_LIVE,
    SOURCE_NATIONAL_TABLE,
    HPMSClient,
    TruckShare,
    _parse_f_system,
    dominant_functional_class,
    national_truck_share,
    resolve_truck_share,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_station_zip(path, state_abbr='AL', year=2024, rows=None):
    """Build a minimal TMAS .STA zip: pipe-delimited, one row per direction."""
    rows = rows or [
        ('073', '1U'), ('073', '1U'), ('073', '1U'),
        ('073', '4U'), ('117', '3R'),
    ]
    header = 'record_type|state_code|station_id|travel_dir|f_system|county_code'
    lines = [header]
    for index, (county, f_system) in enumerate(rows):
        lines.append(f"S|01|{index:06d}|1|{f_system}|{county}")
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr(f"{state_abbr}_{year} (TMAS).STA", '\n'.join(lines))
    return path


# ---------------------------------------------------------------------------
# the vendored national table
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_national_table_covers_every_functional_class():
    for f_system in range(1, 8):
        for is_rural in (True, False):
            assert (f_system, is_rural) in NATIONAL_TRUCK_SHARES


@pytest.mark.smoke
def test_national_table_shares_are_plausible():
    """Shares are fractions, and rural interstates carry the most freight."""
    for (f_system, is_rural), shares in NATIONAL_TRUCK_SHARES.items():
        total = shares['single_unit'] + shares['combination']
        assert 0.0 < total < 0.5, (f_system, is_rural, total)

    rural_interstate = national_truck_share(1, True).total
    urban_minor_arterial = national_truck_share(4, False).total
    # The structural pattern the design leans on: road function, not vintage.
    assert rural_interstate > 3 * urban_minor_arterial


@pytest.mark.smoke
def test_national_table_never_fails_on_unknown_class():
    share = national_truck_share(99, False)
    assert share.source == SOURCE_NATIONAL_TABLE
    assert share.total > 0


# ---------------------------------------------------------------------------
# f_system parsing
# ---------------------------------------------------------------------------

@pytest.mark.smoke
@pytest.mark.parametrize('raw,expected', [
    ('1U', (1, False)),
    ('1R', (1, True)),
    ('3R', (3, True)),
    (' 4U ', (4, False)),
    ('', None),
    (None, None),
    ('XX', None),
])
def test_parse_f_system(raw, expected):
    assert _parse_f_system(raw) == expected


@pytest.mark.smoke
def test_dominant_functional_class_from_station_file(tmp_path):
    station_zip = _write_station_zip(tmp_path / 'stations.zip')
    result = dominant_functional_class(station_zip, 'AL', ['073', '117'])

    assert result is not None
    f_system, is_rural, detail = result
    assert (f_system, is_rural) == (1, False)      # 1U is the most common
    assert detail['n_stations'] == 5
    assert detail['distribution']['1U'] == 3


@pytest.mark.smoke
def test_dominant_functional_class_filters_by_county(tmp_path):
    station_zip = _write_station_zip(tmp_path / 'stations.zip')
    result = dominant_functional_class(station_zip, 'AL', ['117'])

    assert result is not None
    f_system, is_rural, _ = result
    assert (f_system, is_rural) == (3, True)       # only the 3R row is in 117


@pytest.mark.smoke
def test_dominant_functional_class_missing_file_returns_none(tmp_path):
    assert dominant_functional_class(tmp_path / 'nope.zip', 'AL', ['073']) is None


@pytest.mark.smoke
def test_dominant_functional_class_unknown_county_returns_none(tmp_path):
    station_zip = _write_station_zip(tmp_path / 'stations.zip')
    assert dominant_functional_class(station_zip, 'AL', ['999']) is None


# ---------------------------------------------------------------------------
# TruckShare
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_truck_share_mix_normalises():
    share = TruckShare(single_unit=0.04, combination=0.12,
                       source=SOURCE_NATIONAL_TABLE)
    assert share.total == pytest.approx(0.16)
    assert share.mix['single_unit'] == pytest.approx(0.25)
    assert share.mix['combination'] == pytest.approx(0.75)
    assert sum(share.mix.values()) == pytest.approx(1.0)


@pytest.mark.smoke
def test_truck_share_mix_handles_zero_total():
    share = TruckShare(0.0, 0.0, SOURCE_NATIONAL_TABLE)
    assert sum(share.mix.values()) == pytest.approx(1.0)


@pytest.mark.smoke
def test_truck_share_to_dict_is_json_serialisable():
    payload = json.loads(json.dumps(national_truck_share(1, False).to_dict()))
    assert payload['source'] == SOURCE_NATIONAL_TABLE
    assert 'total' in payload


# ---------------------------------------------------------------------------
# the resolver's fallback chain
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_pinned_config_wins_and_needs_no_network(tmp_path):
    config = {
        'data': {'data_dir': str(tmp_path)},
        'freight': {'truck_share': 0.163,
                    'vehicle_mix': {'single_unit': 0.28, 'combination': 0.72}},
    }
    share = resolve_truck_share(config, '01', ['073'], state_abbr='AL',
                                station_zip=tmp_path / 'missing.zip')

    assert share.source == SOURCE_CONFIG_PINNED
    assert share.total == pytest.approx(0.163)
    assert share.mix['single_unit'] == pytest.approx(0.28)


@pytest.mark.smoke
def test_falls_back_to_national_table_when_hpms_disabled(tmp_path):
    station_zip = _write_station_zip(tmp_path / 'stations.zip')
    config = {'data': {'data_dir': str(tmp_path)},
              'freight': {'hpms': {'enabled': False}}}

    share = resolve_truck_share(config, '01', ['073'], state_abbr='AL',
                                station_zip=station_zip)

    assert share.source == SOURCE_NATIONAL_TABLE
    assert share.f_system == 1 and share.is_rural is False


@pytest.mark.smoke
def test_unreachable_service_never_raises(tmp_path, monkeypatch):
    """The contract: a dead service degrades, it does not stop a run."""
    monkeypatch.setattr(HPMSClient, '_request',
                        lambda self, url, params: None)
    station_zip = _write_station_zip(tmp_path / 'stations.zip')
    config = {'data': {'data_dir': str(tmp_path)},
              'freight': {'hpms': {'enabled': True,
                                   'cache_dir': str(tmp_path / 'hpms')}}}

    share = resolve_truck_share(config, '01', ['073'], state_abbr='AL',
                                station_zip=station_zip)

    assert share.source == SOURCE_NATIONAL_TABLE


@pytest.mark.smoke
def test_malformed_service_response_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(HPMSClient, '_request',
                        lambda self, url, params: {'unexpected': 'shape'})
    station_zip = _write_station_zip(tmp_path / 'stations.zip')
    config = {'data': {'data_dir': str(tmp_path)},
              'freight': {'hpms': {'enabled': True,
                                   'cache_dir': str(tmp_path / 'hpms')}}}

    share = resolve_truck_share(config, '01', ['073'], state_abbr='AL',
                                station_zip=station_zip)

    assert share.source == SOURCE_NATIONAL_TABLE


@pytest.mark.smoke
def test_live_query_result_is_cached_and_reused(tmp_path, monkeypatch):
    calls = []

    def fake_request(self, url, params):
        calls.append(url)
        return {'features': [{'attributes': {
            'total_aadt': 1000.0, 'total_su': 50.0,
            'total_cu': 110.0, 'n_segments': 42}}]}

    monkeypatch.setattr(HPMSClient, '_request', fake_request)
    station_zip = _write_station_zip(tmp_path / 'stations.zip')
    config = {'data': {'data_dir': str(tmp_path)},
              'freight': {'hpms': {'enabled': True,
                                   'cache_dir': str(tmp_path / 'hpms')}}}

    first = resolve_truck_share(config, '01', ['073'], state_abbr='AL',
                                station_zip=station_zip)
    assert first.source == SOURCE_HPMS_LIVE
    assert first.total == pytest.approx(0.16)

    second = resolve_truck_share(config, '01', ['073'], state_abbr='AL',
                                 station_zip=station_zip)
    assert second.source == SOURCE_HPMS_CACHE
    assert second.total == pytest.approx(0.16)
    assert len(calls) == 1, "the cached run must not re-query the service"


@pytest.mark.smoke
def test_stale_cache_is_ignored(tmp_path):
    cache_dir = tmp_path / 'hpms'
    cache_dir.mkdir()
    (cache_dir / 'hpms_01_1U.json').write_text(json.dumps({
        'state_fips': '01', 'f_system': 1, 'is_rural': False,
        'single_unit': 0.05, 'combination': 0.11,
        'fetched_at': time.time() - 400 * 86400, 'detail': {},
    }), encoding='utf-8')

    client = HPMSClient(cache_dir=cache_dir, cache_days=90)
    assert client.read_cache('01', 1, False) is None


@pytest.mark.smoke
def test_corrupt_cache_entry_is_ignored(tmp_path):
    cache_dir = tmp_path / 'hpms'
    cache_dir.mkdir()
    (cache_dir / 'hpms_01_1U.json').write_text('not json at all', encoding='utf-8')

    client = HPMSClient(cache_dir=cache_dir)
    assert client.read_cache('01', 1, False) is None


@pytest.mark.smoke
def test_zero_traffic_response_is_rejected(tmp_path, monkeypatch):
    """A segment set with no traffic must not produce a 0% truck share."""
    monkeypatch.setattr(HPMSClient, '_request', lambda self, url, params: {
        'features': [{'attributes': {'total_aadt': 0.0, 'total_su': 0.0,
                                     'total_cu': 0.0, 'n_segments': 0}}]})
    client = HPMSClient(cache_dir=tmp_path)
    assert client.query_live('01', 1, False) is None


@pytest.mark.smoke
def test_disabled_client_does_not_query(monkeypatch):
    monkeypatch.setattr(HPMSClient, '_request', lambda self, url, params: (_ for _ in ()).throw(
        AssertionError("must not be called")))
    client = HPMSClient(enabled=False)
    assert client.query_live('01', 1, False) is None


@pytest.mark.smoke
def test_resolver_always_returns_a_share_with_no_config(tmp_path):
    """Totality: whatever is missing, a usable number comes back."""
    share = resolve_truck_share({}, '01', ['073'])
    assert isinstance(share, TruckShare)
    assert share.total > 0
