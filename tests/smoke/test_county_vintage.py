"""County FIPS vintage pinning, and the state-centroid fallback behind it.

County FIPS codes are not stable across Census vintages. Connecticut is the
live case: it abolished its 8 counties (09001-09015) in favour of 9 planning
regions (09110-09190) effective 2022, with *no* overlap between the two sets.
LODES blocks are enumerated on 2020 geography, so an unpinned county fetch
returns codes that match nothing in a LODES block GEOID — which is how
boundary commutes through Connecticut became unplaceable and were dropped.

These tests pin both halves of the fix: the vintage constant used for the
fetch, and the fallback that keeps a trip even when its county cannot be
resolved at all.
"""

import tempfile

import pytest

from data_sources.lodes_od import (
    _BLOCK_VINTAGE_YEAR,
    _state_centroids,
    external_county_coords,
)
from utils.region_utils import _COUNTY_VINTAGE_YEAR


# --------------------------------------------------------------------------
# The vintage pin itself
# --------------------------------------------------------------------------

@pytest.mark.smoke
def test_county_vintage_matches_block_vintage():
    """County lookups must use the same vintage as the stored block GEOIDs.

    If these ever diverge, county digits sliced out of a block GEOID can
    resolve to a different county — or to none at all.
    """
    assert _COUNTY_VINTAGE_YEAR == _BLOCK_VINTAGE_YEAR == 2020


# --------------------------------------------------------------------------
# The fallback, exercised without touching the network
# --------------------------------------------------------------------------

@pytest.mark.smoke
def test_unresolvable_county_falls_back_to_state(monkeypatch):
    """A county FIPS absent from the table still yields a placeable point.

    This is the Connecticut case. Before the fallback existed these blocks
    returned nothing and their commutes were silently dropped.
    """
    from models.models import County, initialize_tables

    data_dir = tempfile.mkdtemp(prefix="county_vintage_")
    db = initialize_tables(data_dir)
    try:
        with db.session_scope() as session:
            session.add(County(
                geoid="27053", state_fips="27", county_fips="053",
                county_name="Hennepin", intptlat=45.0, intptlon=-93.4,
            ))
    finally:
        db.close()

    # Stub the network fetch so the test is deterministic and offline.
    monkeypatch.setattr(
        "data_sources.lodes_od._state_centroids",
        lambda fips: {"09": (-72.66, 41.52)} if "09" in fips else {},
    )

    coords = external_county_coords(
        {"data": {"data_dir": data_dir}},
        ["270530001001001", "090010001001001"],
    )

    # Both placed: the known county from the table, the unknown one via state.
    # Coordinates are compared approximately — the DB stores them as float32.
    assert len(coords) == 2
    assert coords["270530001001001"] == pytest.approx((-93.4, 45.0), abs=1e-4)
    assert coords["090010001001001"] == pytest.approx((-72.66, 41.52), abs=1e-4)


@pytest.mark.smoke
def test_trips_dropped_only_when_state_also_unresolvable(monkeypatch):
    """With no state centroid either, the block is omitted rather than faked."""
    from models.models import County, initialize_tables

    data_dir = tempfile.mkdtemp(prefix="county_vintage_none_")
    initialize_tables(data_dir).close()

    monkeypatch.setattr("data_sources.lodes_od._state_centroids", lambda fips: {})

    coords = external_county_coords(
        {"data": {"data_dir": data_dir}}, ["090010001001001"],
    )
    assert coords == {}


@pytest.mark.smoke
def test_empty_input_makes_no_lookup():
    """No external blocks means no DB hit and no network call."""
    assert _state_centroids(set()) == {}


# --------------------------------------------------------------------------
# Live TIGER check — skipped when offline
# --------------------------------------------------------------------------

@pytest.mark.smoke
def test_pinned_fetch_returns_lodes_era_connecticut_counties():
    """The pinned vintage returns the county codes LODES blocks actually use.

    Guards the regression directly: at the unpinned default this returns the
    09110-09190 planning regions instead, which no LODES block references.
    """
    pytest.importorskip("pygris")
    try:
        from pygris import counties as get_counties
        gdf = get_counties(state="09", cache=True, year=_COUNTY_VINTAGE_YEAR)
    except Exception as exc:  # offline, or Census server down
        pytest.skip(f"TIGER unavailable: {exc}")

    geoids = set(gdf["GEOID"])
    assert "09001" in geoids, "expected LODES-era Connecticut county FIPS"
    assert not any(g.startswith("091") for g in geoids), \
        "post-2022 planning regions must not appear at the pinned vintage"
