from abc import ABC, abstractmethod
import uuid
import pandas as pd
from typing import Dict, List, Optional
from utils.logger import setup_logger
from models.models import SurveyTrip, initialize_tables
from models.mode_types import (
    MODE_CAR, MODE_BUS, MODE_RAIL, MODE_WALK, MODE_BIKE,
    MODE_SCHOOL_BUS, MODE_RIDESHARE, MODE_OTHER, CANONICAL_MODES
)

logger = setup_logger(__name__)


class BaseSurveyTrip(ABC):
    """Abstract base class for all survey trip data sources.

    Defines the canonical schema that all survey trip sources must produce.
    Subclasses implement extract_data() and clean_data() with survey-specific
    logic, then call validate_schema() to ensure conformance.
    """

    # ── Column name constants (single source of truth) ──────────────────
    PERSON_ID = 'person_id'
    MODE_TYPE = 'mode_type'
    ORIGIN_LOC = 'origin_loc'
    DESTINATION_LOC = 'destination_loc'
    ORIGIN_PURPOSE = 'origin_purpose'
    DESTINATION_PURPOSE = 'destination_purpose'
    DEPART_TIME = 'depart_time'
    ARRIVE_TIME = 'arrive_time'

    # Optional columns — downstream code must check existence before using
    DURATION = 'duration_seconds'
    DISTANCE = 'distance_miles'
    TRIP_WEIGHT = 'trip_weight'

    # Metadata columns — added by clean_data(), used for DB filtering
    SOURCE_TYPE = 'source_type'
    SOURCE_YEAR = 'source_year'

    # Required for all surveys — validate_schema() checks these
    REQUIRED_COLUMNS = {
        PERSON_ID, MODE_TYPE,
        ORIGIN_PURPOSE, DESTINATION_PURPOSE,
        DEPART_TIME, ARRIVE_TIME,
    }

    # Location columns — required for OD matrices but not all surveys
    # (e.g. NHTS public-use has no block group identifiers).
    # Downstream OD code must check for non-null values before using.
    LOCATION_COLUMNS = {ORIGIN_LOC, DESTINATION_LOC}

    # ── Census geography level constants ────────────────────────────────
    # Determined automatically from the GEOID length in origin_loc values.
    GEO_TRACT = 'tract'            # 11-digit GEOID: state(2) + county(3) + tract(6)
    GEO_BLOCK_GROUP = 'block_group'  # 12-digit GEOID: state(2) + county(3) + tract(6) + bg(1)

    # Maps GEOID string length → geo level constant
    GEOID_LENGTH_TO_GEO_LEVEL = {
        11: GEO_TRACT,
        12: GEO_BLOCK_GROUP,
    }

    # ── Canonical activity types (single source of truth) ───────────────
    # All surveys must map their raw purpose labels to these values.
    # All downstream code must reference these constants.
    ACT_HOME = 'Home'
    ACT_WORK = 'Work'
    ACT_SCHOOL = 'School'
    ACT_SHOPPING = 'Shopping'
    ACT_SOCIAL = 'Social'
    ACT_DINING = 'Dining'
    ACT_ESCORT = 'Escort'
    ACT_OTHER = 'Other'

    CANONICAL_ACTIVITIES = {
        ACT_HOME, ACT_WORK, ACT_SCHOOL, ACT_SHOPPING,
        ACT_SOCIAL, ACT_DINING, ACT_ESCORT, ACT_OTHER,
    }

    # ── Canonical transport modes (imported from mode_types.py) ─────────
    # All surveys must map their raw mode labels to these values.
    # Note: These are imported from models.mode_types and re-exported here
    # for backwards compatibility with existing survey implementations.
    MODE_CAR = MODE_CAR
    MODE_BUS = MODE_BUS
    MODE_RAIL = MODE_RAIL
    MODE_WALK = MODE_WALK
    MODE_BIKE = MODE_BIKE
    MODE_SCHOOL_BUS = MODE_SCHOOL_BUS
    MODE_RIDESHARE = MODE_RIDESHARE
    MODE_OTHER = MODE_OTHER

    CANONICAL_MODES = CANONICAL_MODES

    def __init__(self, config: Dict):
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.persons: Optional[Dict] = None
        self.metadata: Dict[str, str] = {}  # populated by subclass (source_type, source_year)

    # ── Abstract methods (subclasses must implement) ────────────────────

    @abstractmethod
    def extract_data(self, year: str) -> pd.DataFrame:
        """Read raw survey data for *year*.

        Each subclass reads its own file/API, selects its RAW_COLUMNS,
        and stores the result in self.data.
        """

    @abstractmethod
    def clean_data(self, **kwargs) -> None:
        """Clean data using survey-specific rules.

        At minimum a subclass must:
        1. Apply survey-specific filters and transformations.
        2. Rename raw columns to canonical names using its COLUMN_MAP.
        3. Add SOURCE_TYPE and SOURCE_YEAR columns.
        4. Call self.validate_schema() as the final step.
        """

    # ── Concrete methods (shared by all surveys) ────────────────────────

    def validate_schema(self) -> bool:
        """Verify self.data contains all REQUIRED_COLUMNS.

        Raises ValueError with a list of missing columns on failure.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call extract_data and clean_data first.")

        present = set(self.data.columns)
        missing = self.REQUIRED_COLUMNS - present
        if missing:
            raise ValueError(
                f"Schema validation failed. Missing columns: {sorted(missing)}"
            )

        logger.info(
            f"Schema validation passed — {len(self.REQUIRED_COLUMNS)} required "
            f"columns present, {len(present) - len(self.REQUIRED_COLUMNS)} extra columns"
        )
        return True

    def detect_geo_level(self) -> None:
        """Detect the census geography level from non-null origin_loc values.

        Samples the ORIGIN_LOC column, measures the GEOID string length, and
        stores the result in self.metadata['geo_level'].  Must be called after
        clean_data() has renamed raw columns to canonical names.

        Supports:
            11 digits → tract
            12 digits → block_group

        Sets self.metadata['geo_level'] to None when all origin_loc values are
        null (e.g. NHTS public-use file which has no geographic identifiers).

        Raises:
            ValueError: If GEOID lengths are non-uniform (dirty data) or if the
                        length does not match any known census geography.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call extract_data and clean_data first.")

        non_null = self.data[self.ORIGIN_LOC].dropna()
        non_null = non_null[non_null.astype(str).str.strip() != '']

        if non_null.empty:
            logger.warning(
                "detect_geo_level: all origin_loc values are null — "
                "geo_level set to None (survey has no geographic identifiers)"
            )
            self.metadata['geo_level'] = None
            return

        lengths = non_null.astype(str).str.strip().str.len().unique()

        if len(lengths) > 1:
            raise ValueError(
                f"detect_geo_level: non-uniform GEOID lengths found in origin_loc: "
                f"{sorted(lengths)}. Check for dirty data or mixed geography types."
            )

        length = int(lengths[0])
        geo_level = self.GEOID_LENGTH_TO_GEO_LEVEL.get(length)

        if geo_level is None:
            raise ValueError(
                f"detect_geo_level: GEOID length {length} does not match any known "
                f"census geography. Supported lengths: {list(self.GEOID_LENGTH_TO_GEO_LEVEL.keys())}"
            )

        self.metadata['geo_level'] = geo_level
        logger.info(f"Detected geo level: '{geo_level}' (GEOID length {length})")

    def save_data(self, batch_size: int = 50000) -> None:
        """Save cleaned data to DuckDB (survey_trips table).

        Iterates over SurveyTrip model columns dynamically so any survey
        that produces a conforming DataFrame is handled automatically.
        """
        if self.data is None:
            raise ValueError("No data to save. Process data first.")

        db_manager = initialize_tables(self.config['data']['data_dir'])
        try:
            df = self.data.copy()

            # Ensure an 'id' column exists for the primary key
            # Use UUIDs so multiple surveys can coexist in the same table
            df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]

            # Get the set of columns the DB model expects
            model_columns = {
                col.name for col in SurveyTrip.__table__.columns
            }

            logger.info("Converting DataFrame to records...")
            records_raw = df.to_dict('records')
            total = len(records_raw)

            logger.info(f"Saving {total} records in batches of {batch_size}...")

            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                batch = []
                for row in records_raw[start:end]:
                    record = {}
                    for col_name in model_columns:
                        val = row.get(col_name)
                        if val is None or (isinstance(val, float) and pd.isna(val)):
                            record[col_name] = None
                        elif col_name in (self.DEPART_TIME, self.ARRIVE_TIME):
                            record[col_name] = pd.Timestamp(val).to_pydatetime() if pd.notna(val) else None
                        elif col_name in ('id', self.PERSON_ID, self.MODE_TYPE,
                                          self.ORIGIN_LOC, self.DESTINATION_LOC,
                                          self.ORIGIN_PURPOSE, self.DESTINATION_PURPOSE,
                                          self.SOURCE_TYPE, self.SOURCE_YEAR):
                            record[col_name] = str(val) if pd.notna(val) else None
                        elif col_name in (self.DURATION, self.DISTANCE, self.TRIP_WEIGHT):
                            record[col_name] = float(val) if pd.notna(val) else None
                        else:
                            record[col_name] = val
                    batch.append(record)

                db_manager.insert_records(SurveyTrip, batch)
                logger.info(f"Saved records {start} to {end}")

            logger.info("Successfully saved all data to DuckDB")

        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise
        finally:
            db_manager.close()

    def load_data(self) -> pd.DataFrame:
        """Load survey data from DuckDB, filtered by this source's metadata.

        Uses source_type and source_year from self.metadata so each survey
        instance only retrieves its own rows.
        """
        logger.info("Loading survey data from DuckDB...")

        db_manager = initialize_tables(self.config['data']['data_dir'])
        try:
            filters = {}
            if self.metadata.get('source_type'):
                filters['source_type'] = self.metadata['source_type']
            if self.metadata.get('source_year'):
                filters['source_year'] = self.metadata['source_year']

            with db_manager.session_scope() as session:
                query = session.query(SurveyTrip)
                if filters:
                    query = query.filter_by(**filters)
                records = query.all()

                if not records:
                    logger.warning("No records found in database")
                    self.data = pd.DataFrame()
                    return self.data

                data = [
                    {
                        col.name: getattr(rec, col.name)
                        for col in SurveyTrip.__table__.columns
                        if col.name != 'id'
                    }
                    for rec in records
                ]
                self.data = pd.DataFrame(data)

            logger.info(f"Loaded {len(self.data)} records from DuckDB")
            return self.data

        except Exception as e:
            logger.error(f"Error loading survey data: {e}")
            raise
        finally:
            db_manager.close()

    def process_persons(self, max_persons: Optional[int] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Group trips by person_id then by date.

        Returns:
            {person_id: {date_str: trips_df}}
        """
        if self.data is None:
            raise ValueError("No data loaded. Call extract_data and clean_data first.")

        logger.info("Processing persons and trips...")

        persons_dict: Dict[str, Dict[str, pd.DataFrame]] = {}
        grouped = self.data.groupby(self.PERSON_ID)

        if max_persons:
            grouped = list(grouped)[:max_persons]

        for person_id, person_trips in grouped:
            person_trips = person_trips.copy()
            person_trips['date'] = person_trips[self.DEPART_TIME].dt.date
            days_dict = {
                str(date): group.copy()
                for date, group in person_trips.groupby('date')
            }
            persons_dict[person_id] = days_dict

        self.persons = persons_dict
        num_days = sum(len(days) for days in persons_dict.values())
        logger.info(f"Processed {len(persons_dict)} persons across {num_days} person-days")
        return self.persons
