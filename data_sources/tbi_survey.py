import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from utils.logger import setup_logger
from data_sources.base_survey_trip import BaseSurveyTrip
from models.mode_types import (
    MODE_CAR, MODE_BUS, MODE_RAIL, MODE_WALK, MODE_BIKE,
    MODE_SCHOOL_BUS, MODE_RIDESHARE, MODE_OTHER
)

logger = setup_logger(__name__)


class TBISurveyTrip(BaseSurveyTrip):
    """TBI (Travel Behavior Inventory) survey trip data source.

    Reads TBI CSV files, cleans with TBI-specific rules, and maps
    raw column names to the canonical schema defined in BaseSurveyTrip.
    """

    # ── Raw CSV columns needed from the TBI file ───────────────────────
    RAW_COLUMNS = [
        'person_id', 'mode_type',
        'o_bg_2020', 'd_bg_2020',
        'o_purpose_category', 'd_purpose_category',
        'o_purpose_category_broad', 'd_purpose_category_broad',
        'depart_time', 'arrive_time',
        'duration_seconds', 'distance_miles',
        'trip_o_county', 'trip_d_county',
        'trip_survey_complete', 'trip_weight',
    ]

    # ── Mapping from raw TBI column names to canonical names ───────────
    COLUMN_MAP = {
        'o_bg_2020': BaseSurveyTrip.ORIGIN_LOC,
        'd_bg_2020': BaseSurveyTrip.DESTINATION_LOC,
        'o_purpose_category_broad': BaseSurveyTrip.ORIGIN_PURPOSE,
        'd_purpose_category_broad': BaseSurveyTrip.DESTINATION_PURPOSE,
    }

    # ── TBI-only columns used during cleaning then dropped ─────────────
    _CLEANING_ONLY_COLUMNS = [
        'o_purpose_category', 'd_purpose_category',
        'trip_o_county', 'trip_d_county',
        'trip_survey_complete',
    ]

    # ── Mapping from raw TBI purpose labels to canonical activity types ──
    # Multiple raw labels can map to the same canonical type.
    PURPOSE_MAP = {
        'Went home': BaseSurveyTrip.ACT_HOME,
        'Work': BaseSurveyTrip.ACT_WORK,
        'School': BaseSurveyTrip.ACT_SCHOOL,
        'Shopping/Errands': BaseSurveyTrip.ACT_SHOPPING,
        'Escort': BaseSurveyTrip.ACT_SHOPPING,
        'Social/Recreation': BaseSurveyTrip.ACT_SOCIAL,
        'Dining': BaseSurveyTrip.ACT_DINING,
        'Other': BaseSurveyTrip.ACT_OTHER,
    }

    # ── Mapping from raw TBI mode labels to canonical transport modes ────
    MODE_MAP = {
        'Household Vehicle': MODE_CAR,
        'Other Vehicle': MODE_CAR,
        'Walk': MODE_WALK,
        'Public Bus': MODE_BUS,
        'Other Bus': MODE_BUS,
        'Rail': MODE_RAIL,
        'Micromobility': MODE_BIKE,
        'School Bus': MODE_SCHOOL_BUS,
        'Smartphone ridehailing service': MODE_RIDESHARE,
        'Other': MODE_OTHER,
        'For-Hire Vehicle': MODE_OTHER,
    }

    def __init__(self, config: Dict):
        super().__init__(config)
        self.metadata = {
            'source_type': 'tbi',
        }

    def _get_allowed_county_fips(self) -> set:
        """Extract 3-digit county FIPS codes from config region.counties.

        Config stores 5-character GEOIDs like '27053'. We take characters
        [2:5] to get the county FIPS portion (e.g. '053'), matching how
        block group strings encode county identity.
        """
        county_geoids = self.config['region']['counties']
        return {geoid[2:5] for geoid in county_geoids}

    def extract_data(self, year: str, file_path: Optional[str] = None) -> pd.DataFrame:
        """Read TBI CSV for the given year and apply geographic filtering.

        Args:
            year: Survey year (e.g. '2023').  Stored in metadata.
            file_path: Absolute or relative path to the CSV file.
                       When called via SurveyManager, this is resolved from
                       the ``data.surveys[].file`` config entry.  When called
                       directly (e.g. from a notebook), falls back to
                       ``config['data']['surveys']`` lookup.
        """
        try:
            data_dir = self.config['data']['data_dir']
            if file_path is None:
                # Fallback: look up from config surveys list
                for entry in self.config['data'].get('surveys', []):
                    if entry.get('type') == 'tbi' and entry.get('year') == year:
                        file_path = str(Path(data_dir) / entry['file'])
                        break
                if file_path is None:
                    raise KeyError(f"No TBI survey entry for year {year} in config['data']['surveys']")
            else:
                # Resolve relative paths against data_dir
                if not Path(file_path).is_absolute():
                    file_path = str(Path(data_dir) / file_path)
            logger.info(f"Reading {year} TBI survey data from {file_path}")

            df = pd.read_csv(
                file_path,
                usecols=self.RAW_COLUMNS,
                encoding='iso-8859-1',
                low_memory=False,
            )

            # Filter trips by block groups using county FIPS from config
            try:
                allowed_fips = self._get_allowed_county_fips()
                logger.info(f"Allowed county FIPS codes: {allowed_fips}")

                # Block group format: state_fips(2) + county_fips(3) + tract + block_group
                df['o_fips'] = df['o_bg_2020'].astype(str).str[2:5]
                df['d_fips'] = df['d_bg_2020'].astype(str).str[2:5]

                before_filter = len(df)
                df = df[
                    df['o_fips'].isin(allowed_fips) &
                    df['d_fips'].isin(allowed_fips)
                ].copy()
                after_filter = len(df)
                logger.info(f"Filtered by region FIPS codes: {before_filter} -> {after_filter} records")

                df.drop(columns=['o_fips', 'd_fips'], inplace=True)

            except KeyError as e:
                logger.warning(
                    f"Block group columns 'o_bg_2020' or 'd_bg_2020' not found; "
                    f"skipping FIPS filter: {e}"
                )

            # Convert time columns to datetime
            for col in ['depart_time', 'arrive_time']:
                logger.info(f"Converting {col} to datetime...")
                df[col] = pd.to_datetime(df[col], errors='coerce')

            self.data = df
            self.metadata['source_year'] = year
            logger.info(f"Successfully extracted {len(df)} records")
            return df

        except KeyError:
            logger.error(f"Year {year} not found in config")
            raise
        except Exception as e:
            logger.error(f"Error extracting data: {e}")
            raise

    def clean_data(self, timezone: str = 'America/Chicago',
                   duration_std_multiplier: float = 3.0,
                   distance_std_multiplier: float = 3.0) -> None:
        """Clean TBI data with TBI-specific filters, rename to canonical columns."""
        if self.data is None:
            raise ValueError("No data loaded. Call extract_data first.")

        try:
            df = self.data.copy()
            initial_count = len(df)
            logger.info(f"Starting TBI data cleaning. Initial records: {initial_count}")

            # ── TBI-specific basic filters ──────────────────────────────
            df = df[
                (df['trip_survey_complete'] == 'Yes') &
                (df['mode_type'].notna()) &
                (df['mode_type'] != 'Long distance passenger mode') &
                (df['mode_type'] != 'Missing') &
                (df['duration_seconds'].notna()) &
                (df['duration_seconds'] > 0) &
                (df['distance_miles'].notna()) &
                (df['distance_miles'] >= 0) &
                (df['depart_time'].notna()) &
                (df['arrive_time'].notna()) &
                (df['arrive_time'] >= df['depart_time']) &
                (df['o_bg_2020'].notna()) &
                (df['d_bg_2020'].notna()) &
                (df['o_bg_2020'] != 'Missing') &
                (df['d_bg_2020'] != 'Missing') &
                (df['o_purpose_category'] != 'Missing') &
                (df['d_purpose_category'] != 'Missing') &
                (df['trip_o_county'] != 'Missing') &
                (df['trip_d_county'] != 'Missing') &
                (df['trip_o_county'].notna()) &
                (df['trip_d_county'].notna()) &
                (df['o_purpose_category_broad'] != 'Missing') &
                (df['d_purpose_category_broad'] != 'Missing') &
                (df['d_purpose_category_broad'].notna()) &
                (df['o_purpose_category_broad'].notna()) &
                (~df['d_purpose_category_broad'].isin(['Change mode', 'Not imputable'])) &
                (~df['o_purpose_category_broad'].isin(['Change mode', 'Not imputable']))
            ]

            after_basic = len(df)
            logger.info(f"After basic filters: {after_basic} records ({initial_count - after_basic} removed)")

            # ── Statistical outlier filtering: duration ──────────────────
            dur_mean = df['duration_seconds'].mean()
            dur_std = df['duration_seconds'].std()
            dur_threshold = dur_mean + duration_std_multiplier * dur_std

            logger.info(f"Duration stats: mean={dur_mean/60:.2f} min, std={dur_std/60:.2f} min")
            logger.info(f"Duration threshold (mean + {duration_std_multiplier}*std): {dur_threshold/60:.2f} min")

            before = len(df)
            df = df[df['duration_seconds'] <= dur_threshold]
            logger.info(f"After duration filter: {len(df)} records ({before - len(df)} removed)")

            # ── Statistical outlier filtering: distance ──────────────────
            dist_mean = df['distance_miles'].mean()
            dist_std = df['distance_miles'].std()
            dist_threshold = dist_mean + distance_std_multiplier * dist_std

            logger.info(f"Distance stats: mean={dist_mean:.2f} mi, std={dist_std:.2f} mi")
            logger.info(f"Distance threshold (mean + {distance_std_multiplier}*std): {dist_threshold:.2f} mi")

            before = len(df)
            df = df[df['distance_miles'] <= dist_threshold]
            logger.info(f"After distance filter: {len(df)} records ({before - len(df)} removed)")

            # ── Timezone conversion ──────────────────────────────────────
            for col in ['arrive_time', 'depart_time']:
                df[col] = df[col].dt.tz_convert(timezone)

            # ── Clean block group IDs ────────────────────────────────────
            for col in ['o_bg_2020', 'd_bg_2020']:
                df[col] = df[col].astype(str).str.replace('.0', '', regex=False)

            # ── Drop TBI-only columns that aren't part of the canonical schema
            cols_to_drop = [c for c in self._CLEANING_ONLY_COLUMNS if c in df.columns]
            df.drop(columns=cols_to_drop, inplace=True)

            # ── Rename raw TBI columns to canonical names ────────────────
            df.rename(columns=self.COLUMN_MAP, inplace=True)

            # ── Map raw purpose labels to canonical activity types ────────
            for col in [self.ORIGIN_PURPOSE, self.DESTINATION_PURPOSE]:
                unmapped = set(df[col].unique()) - set(self.PURPOSE_MAP.keys())
                if unmapped:
                    logger.warning(f"Unmapped purpose values in {col}: {unmapped} — mapping to '{self.ACT_OTHER}'")
                df[col] = df[col].map(self.PURPOSE_MAP).fillna(self.ACT_OTHER)

            # ── Map raw mode labels to canonical transport modes ──────────
            unmapped_modes = set(df[self.MODE_TYPE].unique()) - set(self.MODE_MAP.keys())
            if unmapped_modes:
                logger.warning(f"Unmapped mode values: {unmapped_modes} — mapping to '{self.MODE_OTHER}'")
            df[self.MODE_TYPE] = df[self.MODE_TYPE].map(self.MODE_MAP).fillna(self.MODE_OTHER)

            # ── Add source metadata columns ──────────────────────────────
            df[self.SOURCE_TYPE] = self.metadata['source_type']
            df[self.SOURCE_YEAR] = self.metadata.get('source_year', '')

            final_count = len(df)
            logger.info(f"Cleaning complete. {initial_count - final_count} records removed. Final: {final_count}")

            self.data = df

            # ── Validate canonical schema ────────────────────────────────
            self.validate_schema()

        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            raise
