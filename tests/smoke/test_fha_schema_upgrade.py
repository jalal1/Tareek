"""Smoke tests for the automatic FHA per-direction schema upgrade.

An FHA table created before the per-direction change has no ``travel_dir``
column. The pipeline used to stop and tell the user to run
``scripts/migrate_fha_perdirection.py`` by hand, which every fresh clone with
an existing database hit on its first run. ``_check_schema`` now performs that
same drop + recreate itself, because both FHA tables hold only data re-ingested
from the TMAS zips.

These tests use a real DuckDB file (no Java, no network) so they run in the
smoke tier.
"""

import pytest
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from data_sources.fha_counts_manager import FHACountsManager, FHASchemaError
from utils.duckdb_manager import DBManager


def _make_old_schema_db(tmp_path):
    """Create a DB whose FHA tables predate the travel_dir column.

    DBManager's constructor creates the current-schema tables, so the old shape
    is produced by dropping them and recreating a travel_dir-less version.
    """
    db = DBManager(str(tmp_path))

    OldBase = declarative_base()

    class OldFHAStation(OldBase):
        __tablename__ = 'fha_stations'
        station_id = Column(String, primary_key=True)
        state_code = Column(String)

    class OldFHAHourlyVolume(OldBase):
        __tablename__ = 'fha_hourly_volumes'
        station_id = Column(String, primary_key=True)
        hour = Column(Integer, primary_key=True)

    engine = create_engine(f'duckdb:///{db.db_path}',
                           connect_args={'read_only': False},
                           poolclass=NullPool)
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql('DROP TABLE IF EXISTS fha_hourly_volumes')
            conn.exec_driver_sql('DROP TABLE IF EXISTS fha_stations')
        OldBase.metadata.create_all(engine)
    finally:
        engine.dispose()
    return db


def _manager(db):
    return FHACountsManager({'counts': {'fha': {}}, 'region': {'counties': []}}, db)


@pytest.mark.smoke
def test_old_schema_is_upgraded_in_place(tmp_path):
    """An old-shape DB gains travel_dir instead of raising."""
    db = _make_old_schema_db(tmp_path)
    assert 'travel_dir' not in db.get_table_columns('fha_stations')

    _manager(db)._check_schema()

    for table in ('fha_stations', 'fha_hourly_volumes'):
        assert 'travel_dir' in db.get_table_columns(table), (
            f"{table} was not upgraded to the per-direction schema"
        )


@pytest.mark.smoke
def test_current_schema_is_left_alone(tmp_path):
    """A DB already on the new schema keeps its rows — no needless rebuild."""
    db = DBManager(str(tmp_path))
    assert 'travel_dir' in db.get_table_columns('fha_stations')

    from models.models import FHAStation
    db.insert_records(FHAStation, [{
        'id': '55_000001_1', 'state_code': '55', 'station_id': '000001',
        'travel_dir': 1, 'lat': 43.07, 'lon': -89.40,
        'county_code': '025', 'year': 2024,
    }])

    _manager(db)._check_schema()

    engine = create_engine(f'duckdb:///{db.db_path}',
                           connect_args={'read_only': False},
                           poolclass=NullPool)
    try:
        with engine.begin() as conn:
            n = conn.exec_driver_sql(
                'SELECT COUNT(*) FROM fha_stations').scalar()
    finally:
        engine.dispose()
    assert n == 1, "an up-to-date schema must not be dropped and rebuilt"


@pytest.mark.smoke
def test_failed_upgrade_raises_fha_schema_error(tmp_path, monkeypatch):
    """If the rebuild cannot complete, the run still stops loudly."""
    db = _make_old_schema_db(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("table is locked")

    monkeypatch.setattr(db, 'drop_table', _boom)

    with pytest.raises(FHASchemaError, match="per-direction"):
        _manager(db)._check_schema()
