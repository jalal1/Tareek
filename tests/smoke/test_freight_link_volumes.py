"""Smoke tests for the compact truck-stream artefact.

The artefact exists so the 20-minute events parse happens once per experiment
rather than once per estimator run. The properties that matter are therefore
round-tripping (an artefact read back must give the same volumes the events gave)
and caching (a second call must not re-parse).

See docs/freight/design.md §5.
"""

import gzip
import json

import numpy as np
import pytest

from models.freight.events import LinkVolumes
from models.freight.link_volumes import (
    ARTEFACT_NAME,
    ARTEFACT_VERSION,
    artefact_path,
    artefact_to_link_volumes,
    build_artefact,
    extract_link_volumes_artefact,
    load_artefact,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _volumes(mapping, overflow=None):
    return LinkVolumes(
        volumes={k: np.array(v, dtype=float) for k, v in mapping.items()},
        n_vehicles=3,
        n_events=int(sum(sum(v) for v in mapping.values())),
        overflow=dict(overflow or {}),
    )


@pytest.fixture
def streams():
    """Freight on two links, total carrying more, one post-midnight entry."""
    freight_hours = [0.0] * 24
    freight_hours[8] = 4.0
    freight_hours[9] = 2.0

    # L2 carries cars only: it must appear in the total stream and be absent
    # from the artefact's link table.
    other_hours = [0.0] * 24
    other_hours[8] = 1.0

    total_hours = [0.0] * 24
    total_hours[8] = 20.0
    total_hours[9] = 10.0

    freight = _volumes({'L1': freight_hours}, overflow={'L1': 1.0})
    total = _volumes({'L1': total_hours, 'L2': other_hours})
    car = _volumes({'L1': [t - f for t, f in zip(total_hours, freight_hours)],
                    'L2': [0.0] * 24})
    return {'freight': freight, 'total': total, 'car': car}


def _write_events(path, rows):
    """Minimal MATSim events file. 'entered link' is the real serialised type."""
    parts = ['<?xml version="1.0" encoding="utf-8"?>', '<events version="1.0">']
    for time, vehicle, link in rows:
        parts.append(
            f'<event time="{time}" type="entered link" '
            f'vehicle="{vehicle}" link="{link}" />')
    parts.append('</events>')
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        handle.write('\n'.join(parts))


def _write_plans(path, freight_ids, other_ids=()):
    parts = ['<?xml version="1.0" encoding="utf-8"?>', '<population>']
    for pid in freight_ids:
        parts.append(
            f'<person id="{pid}"><attributes>'
            f'<attribute name="subpopulation" class="java.lang.String">freight'
            f'</attribute></attributes><plan></plan></person>')
    for pid in other_ids:
        parts.append(f'<person id="{pid}"><plan></plan></person>')
    parts.append('</population>')
    path.write_text('\n'.join(parts), encoding='utf-8')


# ---------------------------------------------------------------------------
# build_artefact
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_artefact_records_only_links_carrying_freight(streams):
    payload = build_artefact(streams, scale_factor=1.0)
    # L2 has no freight, so it must not bloat the file.
    assert set(payload['links']) == {'L1'}
    assert payload['n_links_with_freight'] == 1


@pytest.mark.smoke
def test_artefact_daily_includes_post_midnight_overflow(streams):
    payload = build_artefact(streams, scale_factor=1.0)
    # 4 + 2 in-day, plus 1 after 24:00. Dropping the overflow would understate
    # the link against an observed daily AADT.
    assert payload['links']['L1']['freight'] == pytest.approx(7.0)


@pytest.mark.smoke
def test_artefact_hourly_excludes_overflow(streams):
    payload = build_artefact(streams, scale_factor=1.0)
    hourly = payload['links']['L1']['freight_hourly']
    assert len(hourly) == 24
    assert sum(hourly) == pytest.approx(6.0)


@pytest.mark.smoke
def test_artefact_reports_freight_share(streams):
    payload = build_artefact(streams, scale_factor=1.0)
    # freight 7 of total 31 (30 in-day on L1 + 1 on L2). Stored rounded to 3dp.
    assert payload['totals']['freight_share_pct'] == pytest.approx(
        streams['freight'].total() / streams['total'].total() * 100, abs=1e-3)


@pytest.mark.smoke
def test_artefact_carries_version(streams):
    assert build_artefact(streams, scale_factor=1.0)['version'] == ARTEFACT_VERSION


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_round_trip_preserves_daily_totals(streams):
    payload = build_artefact(streams, scale_factor=1.0)
    rebuilt = artefact_to_link_volumes(payload)
    assert rebuilt.daily('L1') == pytest.approx(streams['freight'].daily('L1'))


@pytest.mark.smoke
def test_round_trip_preserves_hourly_profile(streams):
    payload = build_artefact(streams, scale_factor=1.0)
    rebuilt = artefact_to_link_volumes(payload)
    np.testing.assert_allclose(rebuilt.hourly('L1'), streams['freight'].hourly('L1'))


@pytest.mark.smoke
def test_round_trip_recovers_overflow_separately(streams):
    """The overflow must survive as overflow, not be folded into hour 23 — a
    fabricated late-night peak would read as a timing error in tier 3."""
    payload = build_artefact(streams, scale_factor=1.0)
    rebuilt = artefact_to_link_volumes(payload)
    assert rebuilt.overflow.get('L1') == pytest.approx(1.0)
    assert rebuilt.hourly('L1')[23] == 0.0


@pytest.mark.smoke
def test_round_trip_does_not_rescale(streams):
    """Artefact volumes are already scaled; applying scale_factor again would
    silently square it."""
    scaled = {k: v.scaled(4.0) for k, v in streams.items()}
    payload = build_artefact(scaled, scale_factor=4.0)
    rebuilt = artefact_to_link_volumes(payload)
    assert rebuilt.daily('L1') == pytest.approx(28.0)


# ---------------------------------------------------------------------------
# extraction and caching
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_extraction_writes_artefact(tmp_path):
    events = tmp_path / 'output' / 'output_events.xml.gz'
    events.parent.mkdir(parents=True)
    _write_events(events, [
        (28800, 'F1', 'L1'), (28900, 'F1', 'L2'), (32400, 'C1', 'L1'),
    ])
    plans = tmp_path / 'plans.xml'
    _write_plans(plans, ['F1'], ['C1'])

    payload = extract_link_volumes_artefact(tmp_path, scale_factor=1.0)

    assert artefact_path(tmp_path).exists()
    assert payload['links']['L1']['freight'] == pytest.approx(1.0)
    assert payload['links']['L1']['total'] == pytest.approx(2.0)


@pytest.mark.smoke
def test_second_call_reuses_artefact_without_events(tmp_path):
    """The whole point of the artefact: re-estimating must not re-parse. Proven
    by deleting the events file and reading again."""
    events = tmp_path / 'output' / 'output_events.xml.gz'
    events.parent.mkdir(parents=True)
    _write_events(events, [(28800, 'F1', 'L1')])
    plans = tmp_path / 'plans.xml'
    _write_plans(plans, ['F1'])

    first = extract_link_volumes_artefact(tmp_path, scale_factor=1.0)
    events.unlink()
    second = extract_link_volumes_artefact(tmp_path, scale_factor=1.0)

    assert second['links'] == first['links']


@pytest.mark.smoke
def test_force_reparse_ignores_cached_artefact(tmp_path):
    events = tmp_path / 'output' / 'output_events.xml.gz'
    events.parent.mkdir(parents=True)
    _write_events(events, [(28800, 'F1', 'L1')])
    plans = tmp_path / 'plans.xml'
    _write_plans(plans, ['F1'])

    extract_link_volumes_artefact(tmp_path, scale_factor=1.0)
    _write_events(events, [(28800, 'F1', 'L1'), (28900, 'F1', 'L1')])
    forced = extract_link_volumes_artefact(tmp_path, scale_factor=1.0, force=True)

    assert forced['links']['L1']['freight'] == pytest.approx(2.0)


@pytest.mark.smoke
def test_stale_version_is_reextracted(tmp_path):
    events = tmp_path / 'output' / 'output_events.xml.gz'
    events.parent.mkdir(parents=True)
    _write_events(events, [(28800, 'F1', 'L1')])
    plans = tmp_path / 'plans.xml'
    _write_plans(plans, ['F1'])

    stale = {'version': ARTEFACT_VERSION - 1, 'links': {'BOGUS': {}}}
    artefact_path(tmp_path).write_text(json.dumps(stale), encoding='utf-8')

    payload = extract_link_volumes_artefact(tmp_path, scale_factor=1.0)
    assert 'BOGUS' not in payload['links']
    assert payload['version'] == ARTEFACT_VERSION


@pytest.mark.smoke
def test_load_artefact_rejects_stale_version(tmp_path):
    artefact_path(tmp_path).write_text(
        json.dumps({'version': ARTEFACT_VERSION - 1}), encoding='utf-8')
    assert load_artefact(tmp_path) is None


@pytest.mark.smoke
def test_load_artefact_missing_returns_none(tmp_path):
    assert load_artefact(tmp_path) is None


@pytest.mark.smoke
def test_load_artefact_unreadable_returns_none(tmp_path):
    artefact_path(tmp_path).write_text('{not json', encoding='utf-8')
    assert load_artefact(tmp_path) is None


@pytest.mark.smoke
def test_scale_factor_applied_to_extracted_volumes(tmp_path):
    """Comparing an unscaled sample against observed AADT understates by exactly
    1/flowCapacityFactor, so the scaling has to reach the artefact."""
    events = tmp_path / 'output' / 'output_events.xml.gz'
    events.parent.mkdir(parents=True)
    _write_events(events, [(28800, 'F1', 'L1')])
    plans = tmp_path / 'plans.xml'
    _write_plans(plans, ['F1'])

    payload = extract_link_volumes_artefact(tmp_path, scale_factor=10.0)
    assert payload['scale_factor'] == 10.0
    assert payload['links']['L1']['freight'] == pytest.approx(10.0)
