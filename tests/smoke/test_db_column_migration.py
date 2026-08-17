"""Smoke tests for additive column migration in DBManager.

``Base.metadata.create_all`` creates missing *tables* and leaves existing ones
untouched. So adding a column to a model breaks every database created before
it — SQLAlchemy selects the new column by name and DuckDB raises "Table does not
have a column named ...". This was found while adding
``WorkLocation.n_employees_freight`` (design.md item 12): the query failed
outright against a pre-existing database, which would have included the server's.

These databases are rebuildable caches of public data, so a migration framework
would be disproportionate — but silently breaking an existing one is worse. The
compromise is additive-only: new nullable columns are added, nothing is ever
dropped, renamed or retyped.
"""

from pathlib import Path

import duckdb
import pytest

from models.models import WorkLocation, initialize_tables


def _seed_old_schema(tmp_path: Path, rows: str = "") -> Path:
    """A database whose work_locations table predates n_employees_freight."""
    db_dir = tmp_path / 'db'
    db_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(db_dir / 'trafficsim1.2.duckdb'))
    connection.execute(
        'CREATE TABLE work_locations ('
        ' geoid VARCHAR PRIMARY KEY, state_fips VARCHAR, county_fips VARCHAR,'
        ' n_employees INTEGER, lat DOUBLE, lon DOUBLE)')
    if rows:
        connection.execute(f'INSERT INTO work_locations VALUES {rows}')
    connection.close()
    return tmp_path


@pytest.mark.smoke
def test_new_column_is_added_to_an_existing_table(tmp_path):
    """Without this the whole freight run dies on a pre-existing database."""
    data_dir = _seed_old_schema(
        tmp_path, "('010730001','01','073',100,33.5,-86.8)")

    manager = initialize_tables(str(data_dir))
    try:
        with manager.Session() as session:
            rows = session.query(WorkLocation).all()
            assert len(rows) == 1
            assert rows[0].n_employees_freight is None
    finally:
        manager.close()


@pytest.mark.smoke
def test_existing_data_survives_the_migration(tmp_path):
    """Additive only: the migration must never lose a row or a value."""
    data_dir = _seed_old_schema(
        tmp_path,
        "('010730001','01','073',100,33.5,-86.8),"
        "('010730002','01','073',250,33.6,-86.9)")

    manager = initialize_tables(str(data_dir))
    try:
        with manager.Session() as session:
            employees = sorted(r.n_employees
                               for r in session.query(WorkLocation).all())
            assert employees == [100, 250]
    finally:
        manager.close()


@pytest.mark.smoke
def test_migration_is_idempotent(tmp_path):
    """initialize_tables runs on every entry point, so it must be re-runnable."""
    data_dir = _seed_old_schema(
        tmp_path, "('010730001','01','073',100,33.5,-86.8)")

    for _ in range(3):
        manager = initialize_tables(str(data_dir))
        try:
            with manager.Session() as session:
                assert session.query(WorkLocation).count() == 1
        finally:
            manager.close()


@pytest.mark.smoke
def test_a_fresh_database_needs_no_migration(tmp_path):
    """create_all builds the current schema; the migration must be a no-op."""
    manager = initialize_tables(str(tmp_path))
    try:
        with manager.Session() as session:
            session.query(WorkLocation).all()  # must not raise
    finally:
        manager.close()
