"""Smoke tests for the freight estimator's decision logic.

These test *what it decides to write*, not the HPMS network path — the resolver
has its own tests and is total by contract. The properties that matter here are
the refusals: an estimator that writes a national-average truck share, or writes
a through share it cannot measure, would look calibrated while being wrong.

See docs/freight/design.md §3 and §5.
"""

import json

import pytest

from estimators.freight_estimator import (
    MIN_SEGMENTS,
    RATIO_BAND,
    estimate_demand_scale,
    report_through_share,
    update_estimated_config,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _payload(ratio, n_segments=884, demand_scale=0.2):
    return {
        'by_segment': {
            'ratio': ratio,
            'n_segments': n_segments,
            'geh_median': 38.88,
        },
        'comparison': {'ratio': ratio, 'n_links': 2503},
        '_summary': {'demand': {'demand_scale': demand_scale}},
    }


# ---------------------------------------------------------------------------
# demand_scale
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_implied_scale_is_computed_not_hardcoded():
    """Anoka, measured: ratio 0.334 at demand_scale 0.2 implies 0.599."""
    recs = estimate_demand_scale(_payload(0.334, demand_scale=0.2))
    assert len(recs) == 1
    assert recs[0]['recommended'] == pytest.approx(0.599, abs=0.001)


@pytest.mark.smoke
def test_implied_scale_at_the_fifteen_county_measurement():
    """15-county, measured: ratio 0.393 at 0.2 implies 0.509."""
    recs = estimate_demand_scale(_payload(0.393, demand_scale=0.2))
    assert recs[0]['recommended'] == pytest.approx(0.509, abs=0.001)


@pytest.mark.smoke
def test_scale_read_from_per_segment_not_per_link():
    """The per-link denominator counts one segment's AADT once per matched
    link, which on Anoka inflated the observed total 2.8x."""
    payload = _payload(0.334)
    payload['comparison']['ratio'] = 0.52  # the superseded per-link figure
    recs = estimate_demand_scale(payload)
    assert recs[0]['recommended'] == pytest.approx(0.599, abs=0.001)
    assert 'per-segment' in recs[0]['reason']


@pytest.mark.smoke
def test_falls_back_to_per_link_when_no_segments():
    payload = _payload(0.52)
    payload['by_segment'] = {'n_segments': 0}
    recs = estimate_demand_scale(payload)
    assert recs and 'per-link' in recs[0]['reason']


@pytest.mark.smoke
def test_ratio_inside_band_writes_nothing():
    assert estimate_demand_scale(_payload(1.0)) == []
    assert estimate_demand_scale(_payload(1.0 + RATIO_BAND * 0.5)) == []


@pytest.mark.smoke
def test_ratio_outside_band_writes_a_recommendation():
    assert estimate_demand_scale(_payload(1.0 + RATIO_BAND * 2)) != []


@pytest.mark.smoke
def test_thin_sample_is_not_calibrated_against():
    """A ratio from a handful of segments is noise, not a measurement."""
    assert estimate_demand_scale(_payload(0.334, n_segments=MIN_SEGMENTS - 1)) == []


@pytest.mark.smoke
def test_missing_ratio_writes_nothing():
    payload = _payload(None)
    payload['comparison']['ratio'] = None
    assert estimate_demand_scale(payload) == []


@pytest.mark.smoke
def test_recommendation_targets_the_right_config_key():
    recs = estimate_demand_scale(_payload(0.334))
    assert recs[0]['parameter'] == 'freight.demand_scale'


@pytest.mark.smoke
def test_shape_caveat_travels_with_the_number():
    """One multiplier fixes the aggregate and makes the per-link fit worse
    (GEH 38.9 -> 42.6). Writing the value without that caveat would be a
    headline ratio bought with a worse model."""
    reason = estimate_demand_scale(_payload(0.334))[0]['reason']
    assert 'LEVEL' in reason and 'SHAPE' in reason
    assert '42.6' in reason


# ---------------------------------------------------------------------------
# through share — reported, never written
# ---------------------------------------------------------------------------

def _through_config(through=0.30):
    return {'freight': {'class_shares': {
        'external_to_internal': 0.35,
        'internal_to_external': 0.35,
        'through': through,
    }}}


def _through_summary(through=0.30):
    return {'realised': {'class_shares': {
        'external_to_internal': 0.35,
        'internal_to_external': 0.35,
        'through': through,
    }}}


@pytest.mark.smoke
def test_through_share_returns_no_recommendations():
    """It must never reach the config: the generator produced the realised
    split FROM the configured one, so writing it back learns nothing."""
    assert report_through_share(_through_config(), _through_summary()) is None


@pytest.mark.smoke
def test_through_share_reports_agreement(capsys):
    report_through_share(_through_config(), _through_summary())
    out = capsys.readouterr().out
    assert 'NOT written' in out
    assert 'health check' in out


@pytest.mark.smoke
def test_through_share_flags_divergence(capsys):
    """A realised 0.00 against a configured 0.30 is the angular-pairing failure
    that silently drops every through trip."""
    report_through_share(_through_config(0.30), _through_summary(0.0))
    out = capsys.readouterr().out
    assert 'DRIFT' in out
    assert 'GENERATOR BUG' in out


@pytest.mark.smoke
def test_through_share_handles_missing_realised(capsys):
    report_through_share(_through_config(), {})
    assert capsys.readouterr().out == ''


# ---------------------------------------------------------------------------
# config merge — the partial-block regression
# ---------------------------------------------------------------------------

def _full_freight_config():
    """A freight block with the nesting the real config has."""
    return {
        'region': {'counties': ['27003']},
        'freight': {
            'enabled': True,
            'demand_scale': 0.2,
            'class_shares': {'external_to_internal': 0.35,
                             'internal_to_external': 0.35,
                             'through': 0.30},
            'cordon': {'min_peripherality': 0.6},
            'od_matrix': {'beta': 0.02},
            'pce': {'enabled': False},
        },
    }


@pytest.mark.smoke
def test_merge_onto_config_without_freight_keeps_whole_block(tmp_path):
    """Measured on the server: merging onto a config_estimated.json written by
    the other estimators produced a THREE-key freight block where the source had
    twelve, because a dotted path only creates the parents it needs. Every
    dropped key would then silently fall back to its default."""
    estimated = tmp_path / 'config_estimated.json'
    # What demand/mode-share leave behind: no freight section at all.
    estimated.write_text(json.dumps({'plan_generation': {'scaling_factor': 0.15}}),
                         encoding='utf-8')

    config = _full_freight_config()
    update_estimated_config(config, estimated, [{
        'parameter': 'freight.demand_scale',
        'current': 0.2, 'recommended': 0.5988, 'reason': 'tier 2',
    }])

    written = json.loads(estimated.read_text(encoding='utf-8'))['freight']
    assert written['demand_scale'] == 0.5988          # the estimate applied
    assert written['class_shares']['through'] == 0.30  # and nothing was lost
    assert written['cordon']['min_peripherality'] == 0.6
    assert written['od_matrix']['beta'] == 0.02
    assert written['pce']['enabled'] is False
    # The other estimators' work must survive too.
    assert written is not None
    assert json.loads(estimated.read_text(
        encoding='utf-8'))['plan_generation']['scaling_factor'] == 0.15


@pytest.mark.smoke
def test_merge_prefers_existing_estimated_values(tmp_path):
    """A prior estimator's freight value is more current than the source
    config's, so seeding must not clobber it."""
    estimated = tmp_path / 'config_estimated.json'
    estimated.write_text(
        json.dumps({'freight': {'truck_share': 0.1627, 'demand_scale': 0.4}}),
        encoding='utf-8')

    update_estimated_config(_full_freight_config(), estimated, [])

    written = json.loads(estimated.read_text(encoding='utf-8'))['freight']
    assert written['truck_share'] == 0.1627   # kept from the estimated file
    assert written['demand_scale'] == 0.4     # kept, not reset to the source 0.2
    assert written['class_shares']['through'] == 0.30  # filled in from source


@pytest.mark.smoke
def test_merge_creates_file_when_absent(tmp_path):
    estimated = tmp_path / 'config_estimated.json'
    update_estimated_config(_full_freight_config(), estimated, [{
        'parameter': 'freight.demand_scale',
        'current': 0.2, 'recommended': 0.5988, 'reason': 'tier 2',
    }])
    written = json.loads(estimated.read_text(encoding='utf-8'))['freight']
    assert written['demand_scale'] == 0.5988
    assert written['class_shares']['through'] == 0.30
