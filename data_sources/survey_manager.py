from typing import Dict, List, Optional
import pandas as pd
from utils.logger import setup_logger
from data_sources.base_survey_trip import BaseSurveyTrip
from models.models import SurveyTrip, initialize_tables

logger = setup_logger(__name__)


def ensure_surveys(config: Dict) -> None:
    """Ensure survey data exists in the database for all active survey sources.

    For each survey entry in config['data']['surveys'] with weight > 0,
    checks whether the survey_trips table already has rows for that
    (source_type, source_year).  If not, runs the full ETL pipeline:
    extract_data → clean_data → save_data.
    """
    survey_configs = config['data'].get('surveys', [])
    active_entries = [e for e in survey_configs if e.get('weight', 1.0) > 0]
    if not active_entries:
        return

    data_dir = config['data']['data_dir']
    db_manager = initialize_tables(data_dir)

    try:
        # Check which (source_type, source_year) pairs already exist
        with db_manager.session_scope() as session:
            existing_rows = session.query(
                SurveyTrip.source_type,
                SurveyTrip.source_year,
            ).distinct().all()
            existing_pairs = {(row[0], row[1]) for row in existing_rows}

        missing_entries = [
            e for e in active_entries
            if (e['type'], e.get('year', '')) not in existing_pairs
        ]

        if not missing_entries:
            logger.info(
                f"Survey data already exists for all {len(active_entries)} "
                f"active source(s)"
            )
            return

        logger.info(
            f"Survey data missing for {len(missing_entries)} source(s): "
            f"{[e['type'] + '/' + e.get('year', '?') for e in missing_entries]}. "
            f"Running ETL..."
        )
    finally:
        db_manager.close()

    # Run ETL for each missing source
    SurveyManager._ensure_registry()
    for entry in missing_entries:
        survey_type = entry['type']
        year = entry.get('year', '')
        survey_class = SurveyManager.SURVEY_REGISTRY.get(survey_type)
        if survey_class is None:
            logger.warning(f"Unknown survey type '{survey_type}', skipping ETL")
            continue

        logger.info(f"ETL for {survey_type}/{year}...")
        source = survey_class(config)
        source.metadata['source_type'] = survey_type
        source.metadata['source_year'] = year

        file_path = entry.get('file', '')
        source.extract_data(year=year, file_path=file_path if file_path else None)
        source.clean_data()
        source.save_data()
        logger.info(f"ETL complete for {survey_type}/{year}")


class SurveyManager:
    """Facade that selects and manages survey sources based on config.

    Reads the ``data.surveys`` list from config, instantiates the
    appropriate ``BaseSurveyTrip`` subclass for each entry, and provides
    a single interface for downstream code to load data and process
    persons.

    Typical single-survey usage::

        manager = SurveyManager(config)
        survey_df = manager.get_survey_df()       # single DataFrame
        persons   = manager.get_persons()          # single persons dict

    Multi-survey usage (consumed by blending in Step 6)::

        all_data    = manager.load_data()          # {'tbi': df, 'nhts': df2}
        all_persons = manager.process_persons()    # {'tbi': {...}, 'nhts': {...}}
        weights     = manager.get_blend_weights()  # {'tbi': 0.7, 'nhts': 0.3}
    """

    # Registry mapping survey type strings to their classes.
    # New survey types are registered here.
    SURVEY_REGISTRY: Dict[str, type] = {}

    @classmethod
    def _ensure_registry(cls) -> None:
        """Lazily populate the registry to avoid circular imports."""
        if cls.SURVEY_REGISTRY:
            return
        from data_sources.tbi_survey import TBISurveyTrip
        from data_sources.nhts_survey_trip import NHTSSurveyTrip
        cls.SURVEY_REGISTRY = {
            'tbi': TBISurveyTrip,
            'nhts': NHTSSurveyTrip,
        }

    def __init__(self, config: Dict):
        self.config = config
        self.survey_configs: List[Dict] = config['data']['surveys']
        self.sources: Dict[str, BaseSurveyTrip] = {}
        self._init_sources()

    def _init_sources(self) -> None:
        """Instantiate one BaseSurveyTrip subclass per config entry with weight > 0."""
        self._ensure_registry()

        for entry in self.survey_configs:
            survey_type = entry['type']
            weight = entry.get('weight', 1.0)
            if weight <= 0:
                logger.info(f"Skipping survey source '{survey_type}' (weight={weight})")
                continue
            survey_class = self.SURVEY_REGISTRY.get(survey_type)
            if survey_class is None:
                raise ValueError(
                    f"Unknown survey type '{survey_type}'. "
                    f"Registered types: {list(self.SURVEY_REGISTRY.keys())}"
                )
            source = survey_class(self.config)
            # Set metadata from config entry
            source.metadata['source_type'] = survey_type
            source.metadata['source_year'] = entry.get('year', '')
            self.sources[survey_type] = source
            logger.info(f"Initialized survey source: {survey_type} (year={entry.get('year', '?')})")

    # ── Multi-source interface ──────────────────────────────────────────

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from DB for each active source.

        Returns:
            ``{'tbi': df1, 'nhts': df2, ...}``
        """
        result = {}
        for key, source in self.sources.items():
            result[key] = source.load_data()
            logger.info(f"Loaded {len(result[key])} records for '{key}'")
        return result

    def process_persons(self) -> Dict[str, Dict]:
        """Process persons from each active source.

        Returns:
            ``{'tbi': persons_dict, 'nhts': persons_dict, ...}``
        """
        result = {}
        for key, source in self.sources.items():
            result[key] = source.process_persons()
            logger.info(f"Processed {len(result[key])} persons for '{key}'")
        return result

    def get_blend_weights(self) -> Dict[str, float]:
        """Build weights dict from active config entries (weight > 0).

        Returns:
            ``{'tbi': 0.7, 'nhts': 0.3}``
        """
        return {
            entry['type']: entry.get('weight', 1.0)
            for entry in self.survey_configs
            if entry.get('weight', 1.0) > 0
        }

    def has_multiple_sources(self) -> bool:
        """Return True when more than one survey is configured with weight > 0."""
        active = [
            entry for entry in self.survey_configs
            if entry.get('weight', 1.0) > 0
        ]
        return len(active) > 1

    def get_surveys_with_locations(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Filter loaded survey DataFrames to those with non-null location data.

        A survey "has locations" if its ``origin_loc`` column contains at
        least one non-null value.  Surveys like NHTS (public-use) set
        ``origin_loc`` to ``None`` for every row and are therefore
        excluded from OD matrix and trip-rate blending.

        Args:
            all_data: Result of ``load_data()`` — ``{source_key: DataFrame}``.

        Returns:
            Subset dict containing only sources with location data.
        """
        o_col = BaseSurveyTrip.ORIGIN_LOC
        result = {}
        for name, df in all_data.items():
            if o_col in df.columns and df[o_col].notna().any():
                result[name] = df
            else:
                logger.info(
                    f"Survey '{name}' excluded from location-based models "
                    f"(no non-null {o_col} values)"
                )
        return result

    # ── Single-source convenience methods ───────────────────────────────

    def get_single_source(self) -> Optional[BaseSurveyTrip]:
        """Return the single source when only one survey is configured.

        Returns ``None`` if multiple sources are active.
        """
        if len(self.sources) == 1:
            return next(iter(self.sources.values()))
        return None

    def get_survey_df(self) -> pd.DataFrame:
        """Convenience: load and return a single combined DataFrame.

        When one survey is configured this returns its DataFrame directly.
        When multiple surveys are configured this concatenates them (each
        row retains its ``source_type`` column for downstream filtering).
        """
        all_data = self.load_data()
        if len(all_data) == 1:
            return next(iter(all_data.values()))
        return pd.concat(all_data.values(), ignore_index=True)

    def get_persons(self) -> Dict:
        """Convenience: process and return a single persons dict.

        When one survey is configured this returns its persons dict
        directly.  When multiple surveys are configured this merges them
        (person IDs are already unique per survey since they include
        source-specific identifiers).
        """
        all_persons = self.process_persons()
        if len(all_persons) == 1:
            return next(iter(all_persons.values()))
        merged = {}
        for persons_dict in all_persons.values():
            merged.update(persons_dict)
        return merged
