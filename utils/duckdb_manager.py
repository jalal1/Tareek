from sqlalchemy import create_engine, MetaData, Table, Column, inspect, Integer, String, DateTime, Float
from sqlalchemy.types import *
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import NullPool
from typing import Dict, List, Any, Optional, Type, TypeVar
from contextlib import contextmanager
from pathlib import Path
from filelock import FileLock
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Define a TypeVar for Base
ModelType = TypeVar("ModelType", bound=DeclarativeBase)

class DBManager:
    """SQLAlchemy-based DuckDB database manager with cross-process concurrency.

    DuckDB does not support concurrent access from multiple processes to the
    same database file on Windows. To support multiple concurrent experiments:

    - ALL database access (reads and writes) is serialized via an exclusive
      file lock. This ensures only one process has the .duckdb file open at
      any given time.
    - Engines are short-lived (NullPool) and disposed immediately after each
      operation so the file is released for other processes.
    """

    def __init__(self, data_dir: str, db_name: str = 'trafficsim1.2.duckdb'):
        """
        Initialize database manager

        Args:
            data_dir: Base directory for data
            db_name: Name of the database file
        """
        self.models_dir = os.path.join(data_dir, 'db')
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        self.db_path = os.path.join(self.models_dir, db_name)
        self._lock_path = self.db_path + '.lock'
        self._file_lock = FileLock(self._lock_path, timeout=300)

        # Ensure tables exist (needs write access, done once)
        self._ensure_tables()

    def _ensure_tables(self):
        """Create tables if they don't exist (requires write lock)."""
        from models.models import Base
        with self._file_lock:
            engine = create_engine(
                f'duckdb:///{self.db_path}',
                connect_args={'read_only': False},
                poolclass=NullPool,
            )
            try:
                Base.metadata.create_all(engine)
            finally:
                engine.dispose()

    def _make_engine(self, read_only=False):
        """Create a short-lived engine."""
        return create_engine(
            f'duckdb:///{self.db_path}',
            connect_args={'read_only': read_only},
            poolclass=NullPool,
        )

    @property
    def engine(self):
        """Return a new engine under file lock.

        NOTE: Caller is responsible for disposing this engine when done.
        Prefer session_scope() or write_engine_scope() instead.
        """
        return self._make_engine(read_only=False)

    def Session(self):
        """Create a session under file lock (short-lived engine, auto-disposed on close).

        Compatible with both ``with db_manager.Session() as session:`` and
        ``session = db_manager.Session()`` patterns.

        Acquires the file lock on creation and releases it on close to prevent
        DuckDB IOException when multiple processes access the same file.
        """
        self._file_lock.acquire()
        try:
            engine = self._make_engine(read_only=False)
        except Exception:
            self._file_lock.release()
            raise
        factory = sessionmaker(bind=engine)
        session = factory()
        original_close = session.close
        disposed = [False]  # mutable flag for closure
        lock = self._file_lock

        def close_and_dispose():
            original_close()
            if not disposed[0]:
                disposed[0] = True
                engine.dispose()
                lock.release()

        session.close = close_and_dispose
        return session

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations.

        Acquires the file lock for the duration to prevent DuckDB IOException
        when multiple processes access the same database file.
        """
        with self._file_lock:
            engine = self._make_engine(read_only=False)
            factory = sessionmaker(bind=engine)
            session = factory()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
                engine.dispose()

    @contextmanager
    def write_session_scope(self):
        """Provide a write transactional scope protected by a file lock.

        Acquires an exclusive file lock, opens a read-write engine,
        yields a session, commits, then disposes the engine and releases the lock.
        """
        with self._file_lock:
            write_engine = self._make_engine(read_only=False)
            WriteSession = sessionmaker(bind=write_engine)
            session = WriteSession()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
                write_engine.dispose()

    @contextmanager
    def write_engine_scope(self):
        """Provide a write engine protected by a file lock.

        Use this for operations that need direct engine access (e.g. drop_table, create_all).
        """
        with self._file_lock:
            write_engine = self._make_engine(read_only=False)
            try:
                yield write_engine
            finally:
                write_engine.dispose()

    def insert_records(self, model_class: Type[ModelType], records: List[Dict[str, Any]]):
        """Optimized batch insert of records"""
        try:
            with self.write_session_scope() as session:
                # Convert all records to model instances at once
                objects = [model_class(**self.handle_binary_data(model_class, record))
                        for record in records]

                # Bulk insert
                session.bulk_save_objects(objects)

            logger.info(f"Successfully inserted {len(records)} records into {model_class.__tablename__}")
        except SQLAlchemyError as e:
            logger.error(f"Error inserting records: {str(e)}")
            raise

    def query_all(self, model_class: Type[ModelType], filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Query all records with optional filters

        Args:
            model_class: SQLAlchemy model class
            filters: Dictionary of filters to apply

        Returns:
            List of model instances (detached from session, safe to use outside context)
        """
        try:
            with self.session_scope() as session:
                query = session.query(model_class)
                if filters:
                    query = query.filter_by(**filters)
                results = query.all()

                # Detach objects from session so they can be used after session closes
                # This loads all attributes into the object's __dict__ before expunging
                for obj in results:
                    # Access __dict__ to ensure all immediate attributes are loaded
                    _ = obj.__dict__
                    session.expunge(obj)

                return results
        except SQLAlchemyError as e:
            logger.error(f"Error querying records: {str(e)}")
            raise

    def update_records(self, model_class: Type[ModelType], filters: Dict[str, Any],
                      updates: Dict[str, Any]):
        """
        Update records matching filters

        Args:
            model_class: SQLAlchemy model class
            filters: Dictionary of filters to identify records
            updates: Dictionary of updates to apply
        """
        try:
            with self.write_session_scope() as session:
                session.query(model_class).filter_by(**filters).update(updates)
            logger.info(f"Successfully updated records in {model_class.__tablename__}")
        except SQLAlchemyError as e:
            logger.error(f"Error updating records: {str(e)}")
            raise


    def delete_records(self, model_class: Type[ModelType], filters: Dict[str, Any]):
        """
        Delete records matching filters

        Args:
            model_class: SQLAlchemy model class
            filters: Dictionary of filters to identify records
        """
        try:
            with self.write_session_scope() as session:
                session.query(model_class).filter_by(**filters).delete()
            logger.info(f"Successfully deleted records from {model_class.__tablename__}")
        except SQLAlchemyError as e:
            logger.error(f"Error deleting records: {str(e)}")
            raise

    def handle_binary_data(self, model_class: Type[ModelType], record: Dict[str, Any]) -> Dict[str, Any]:
        """Handle binary data conversion before insert and validate required fields"""
        processed_record = record.copy()

        # Get all columns from the model
        mapper = inspect(model_class)

        for column in mapper.columns:
            col_name = column.name
            col_type = column.type

            # Handle LargeBinary columns
            if isinstance(col_type, LargeBinary):
                if col_name in processed_record:
                    value = processed_record[col_name]
                    if value is not None and not isinstance(value, bytes):
                        logger.warning(f"Converting non-bytes data for column {col_name}")
                        processed_record[col_name] = bytes(value)

            # Ensure nullable fields are explicitly set to None if missing
            if col_name not in processed_record:
                if column.nullable:
                    processed_record[col_name] = None
                elif column.default is None and column.server_default is None:
                    logger.warning(f"Required column {col_name} missing from record")

        return processed_record

    def drop_table(self, model_class: Type[ModelType]):
        """
        Drop a specific table from the database

        Args:
            model_class: SQLAlchemy model class representing the table to drop
        """
        try:
            with self.write_engine_scope() as write_engine:
                model_class.__table__.drop(write_engine, checkfirst=True)
            logger.info(f"Successfully dropped table {model_class.__tablename__}")
        except SQLAlchemyError as e:
            logger.error(f"Error dropping table: {str(e)}")
            raise

    def close(self):
        """
        Close all database connections and dispose of the engine.
        No-op since engines are now short-lived and disposed after each use.
        """
        logger.debug("DBManager.close() called (no-op, engines are short-lived)")

    def __enter__(self):
        """Support context manager protocol"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ARG002
        """Ensure connection is closed when exiting context"""
        self.close()
        return False

    def __del__(self):
        """Ensure connection is closed when object is deleted"""
        self.close()
