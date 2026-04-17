import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
from utils.logger import setup_logger
from data_sources.base_survey_trip import BaseSurveyTrip
from models.mode_types import (
    MODE_CAR, MODE_BUS, MODE_RAIL, MODE_WALK, MODE_BIKE,
    MODE_SCHOOL_BUS, MODE_RIDESHARE, MODE_OTHER
)

logger = setup_logger(__name__)


class NHTSSurveyTrip(BaseSurveyTrip):
    """NHTS (National Household Travel Survey) trip data source.

    Reads NHTS public-use CSV files, cleans with NHTS-specific rules, and
    maps raw column names/codes to the canonical schema defined in
    BaseSurveyTrip.

    Important limitations of the public-use NHTS data:
    - No block group / tract / county identifiers for trip origins or
      destinations.  ``origin_loc`` and ``destination_loc`` are set to
      None.  This means NHTS data **cannot** contribute to OD matrices;
      it can only feed trip-chain, trip-duration, and activity-duration
      models.
    - Times are in HHMM integer format with no real calendar date.
      Datetimes are synthesised with a fixed date (2022-01-01) so that
      ``process_persons()`` grouping and time-arithmetic work correctly.
    """

    # ── Raw CSV columns needed from the NHTS trip file ───────────────────
    RAW_COLUMNS = [
        'HOUSEID', 'PERSONID', 'TRIPID',
        'TRPTRANS',
        'WHYTO', 'WHYFROM', 'WHYTRP1S',
        'STRTTIME', 'ENDTIME',
        'TRVLCMIN', 'TRPMILES',
        'WTTRDFIN',
        'TDAYDATE',
    ]

    # ── NHTS WHYTO → WHYTRP1S deterministic mapping ─────────────────────
    # Verified from 2022 NHTS data: every WHYTO code maps to exactly one
    # WHYTRP1S code with zero exceptions.  We apply this same mapping to
    # WHYFROM (which uses the same code set) to derive a synthetic
    # origin-purpose in WHYTRP1S space.
    #
    # WHYTO/WHYFROM codes (2022 NHTS):
    #   1  = Regular activities at home
    #   2  = Work from home (paid)
    #   3  = Work at a non-home location
    #   4  = Work activity to drop-off/pickup someone/something
    #   5  = Other work-related activities
    #   6  = Attend school as a student
    #   7  = Attend childcare or adult care
    #   8  = Volunteer activities (not paid)
    #   9  = Change type of transportation
    #   10 = Drop off/pick up someone (personal)
    #   11 = Health care visit
    #   12 = Buy meals
    #   13 = Shop/buy/pick-up or return goods
    #   14 = Other family/personal errands
    #   15 = Recreational activities
    #   16 = Exercise
    #   17 = Visit friends or relatives
    #   18 = Rest or relaxation/vacation
    #   19 = Religious or other community activities
    #   97 = Something else (specify)
    #  -9  = Not ascertained (WHYFROM only; dropped during cleaning)
    WHYTO_TO_WHYTRP1S = {
        1: 1,       # Home
        2: 1,       # Work from home → Home
        3: 10,      # Work
        4: 10,      # Work errand → Work
        5: 10,      # Other work → Work
        6: 20,      # School
        7: 20,      # Childcare/adult care → School/Daycare
        8: 97,      # Volunteer → Something Else
        9: 97,      # Change transportation → Something Else
        10: 70,     # Drop off/pick up → Transport Someone
        11: 30,     # Health care → Medical/Dental
        12: 80,     # Buy meals → Meals
        13: 40,     # Shopping → Shopping/Errands
        14: 40,     # Other errands → Shopping/Errands
        15: 50,     # Recreation → Social/Recreational
        16: 50,     # Exercise → Social/Recreational
        17: 50,     # Visit friends → Social/Recreational
        18: 50,     # Relaxation → Social/Recreational
        19: 20,     # Religious/community → School/Daycare/Religious
        97: 97,     # Something else
    }

    # ── WHYTRP1S code → canonical activity type ──────────────────────────
    # WHYTRP1S codes (2022 NHTS):
    #   1  = Home
    #   10 = Work
    #   20 = School/Daycare/Religious
    #   30 = Medical/Dental services
    #   40 = Shopping/Errands
    #   50 = Social/Recreational
    #   70 = Transport someone
    #   80 = Meals
    #   97 = Something else
    PURPOSE_MAP = {
        1: BaseSurveyTrip.ACT_HOME,
        10: BaseSurveyTrip.ACT_WORK,
        20: BaseSurveyTrip.ACT_SCHOOL,
        30: BaseSurveyTrip.ACT_OTHER,       # Medical/Dental → Other (no medical category)
        40: BaseSurveyTrip.ACT_SHOPPING,
        50: BaseSurveyTrip.ACT_SOCIAL,
        70: BaseSurveyTrip.ACT_SHOPPING,     # Transport someone → Shopping (matches TBI: Escort → Shopping)
        80: BaseSurveyTrip.ACT_DINING,
        97: BaseSurveyTrip.ACT_OTHER,
    }

    # ── TRPTRANS code → canonical transport mode ─────────────────────────
    # TRPTRANS codes (2022 NHTS):
    #   1  = Car
    #   2  = Van
    #   3  = SUV/Crossover
    #   4  = Pickup truck
    #   6  = Recreational Vehicle
    #   7  = Motorcycle
    #   8  = Public or commuter bus
    #   9  = School bus
    #   10 = Street car or trolley car
    #   11 = Subway or elevated rail
    #   12 = Commuter rail
    #   13 = Amtrak
    #   14 = Airplane
    #   15 = Taxicab or limo service
    #   16 = Other ride-sharing service
    #   17 = Paratransit / Dial a ride
    #   18 = Bicycle (including bikeshare, ebike, etc.)
    #   19 = E-scooter
    #   20 = Walked
    #   21 = Other (specify)
    MODE_MAP = {
        1: MODE_CAR,
        2: MODE_CAR,
        3: MODE_CAR,
        4: MODE_CAR,
        6: MODE_CAR,          # RV → Car
        7: MODE_OTHER,         # Motorcycle
        8: MODE_BUS,
        9: MODE_SCHOOL_BUS,
        10: MODE_RAIL,         # Streetcar/trolley → Rail
        11: MODE_RAIL,         # Subway
        12: MODE_RAIL,         # Commuter rail
        13: MODE_RAIL,         # Amtrak → Rail
        14: MODE_OTHER,        # Airplane
        15: MODE_RIDESHARE,    # Taxi/limo → Rideshare
        16: MODE_RIDESHARE,
        17: MODE_BUS,          # Paratransit → Bus
        18: MODE_BIKE,
        19: MODE_BIKE,         # E-scooter → Bike
        20: MODE_WALK,
        21: MODE_OTHER,
    }

    def __init__(self, config: Dict):
        super().__init__(config)
        self.metadata = {
            'source_type': 'nhts',
        }

    def extract_data(self, year: str, file_path: Optional[str] = None) -> pd.DataFrame:
        """Read NHTS trip CSV for the given year.

        Args:
            year: Survey year (e.g. '2022').  Stored in metadata.
            file_path: Absolute or relative path to the CSV file.
                       Falls back to ``config['data']['surveys']`` lookup.
        """
        try:
            data_dir = self.config['data']['data_dir']
            if file_path is None:
                for entry in self.config['data'].get('surveys', []):
                    if entry.get('type') == 'nhts' and entry.get('year') == year:
                        file_path = str(Path(data_dir) / entry['file'])
                        break
                if file_path is None:
                    raise KeyError(
                        f"No NHTS survey entry for year {year} in config['data']['surveys']"
                    )
            else:
                if not Path(file_path).is_absolute():
                    file_path = str(Path(data_dir) / file_path)

            logger.info(f"Reading {year} NHTS survey data from {file_path}")

            df = pd.read_csv(
                file_path,
                usecols=self.RAW_COLUMNS,
                low_memory=False,
            )

            logger.info(f"Read {len(df)} raw NHTS trip records")

            # ── Drop records with -9 (not ascertained) in critical fields ─
            sentinel_fields = ['WHYFROM', 'STRTTIME', 'ENDTIME', 'TRVLCMIN', 'TRPMILES']
            before = len(df)
            for field in sentinel_fields:
                df = df[df[field] != -9]
            after = len(df)
            logger.info(f"Dropped {before - after} records with -9 sentinel values")

            # ── Build composite person_id: HOUSEID_PERSONID ───────────────
            df[self.PERSON_ID] = (
                df['HOUSEID'].astype(str) + '_' + df['PERSONID'].astype(str)
            )

            # ── Derive purpose codes using WHYTO → WHYTRP1S mapping ──────
            # Apply the deterministic mapping to both WHYTO and WHYFROM
            # so both origin and destination purposes are in WHYTRP1S space.
            df['dest_purpose_code'] = df['WHYTO'].map(self.WHYTO_TO_WHYTRP1S)
            df['orig_purpose_code'] = df['WHYFROM'].map(self.WHYTO_TO_WHYTRP1S)

            # Drop any rows where mapping failed (unexpected codes)
            unmapped_dest = df['dest_purpose_code'].isna().sum()
            unmapped_orig = df['orig_purpose_code'].isna().sum()
            if unmapped_dest > 0 or unmapped_orig > 0:
                logger.warning(
                    f"Unmapped purpose codes: {unmapped_dest} dest, {unmapped_orig} orig — dropping"
                )
                df = df.dropna(subset=['dest_purpose_code', 'orig_purpose_code'])

            # Map WHYTRP1S codes to canonical activity strings
            df[self.DESTINATION_PURPOSE] = df['dest_purpose_code'].astype(int).map(self.PURPOSE_MAP)
            df[self.ORIGIN_PURPOSE] = df['orig_purpose_code'].astype(int).map(self.PURPOSE_MAP)

            # ── Map transport mode ────────────────────────────────────────
            df[self.MODE_TYPE] = df['TRPTRANS'].map(self.MODE_MAP).fillna(self.MODE_OTHER)

            # ── Construct datetime from HHMM integers ─────────────────────
            # Use a fixed synthetic date; only time-of-day matters.
            def hhmm_to_datetime(hhmm_series):
                hh = hhmm_series // 100
                mm = hhmm_series % 100
                # Clamp to valid range
                hh = hh.clip(0, 23)
                mm = mm.clip(0, 59)
                return pd.to_datetime(
                    '2022-01-01'
                ) + pd.to_timedelta(hh * 60 + mm, unit='m')

            df[self.DEPART_TIME] = hhmm_to_datetime(df['STRTTIME'])
            df[self.ARRIVE_TIME] = hhmm_to_datetime(df['ENDTIME'])

            # Handle trips that cross midnight: if arrive < depart, add 1 day
            cross_midnight = df[self.ARRIVE_TIME] < df[self.DEPART_TIME]
            if cross_midnight.any():
                logger.info(f"{cross_midnight.sum()} trips cross midnight — adjusting arrive_time")
                df.loc[cross_midnight, self.ARRIVE_TIME] += pd.Timedelta(days=1)

            # ── Rename numeric fields to canonical names ──────────────────
            df[self.DURATION] = df['TRVLCMIN'] * 60  # minutes → seconds
            df[self.DISTANCE] = df['TRPMILES']
            df[self.TRIP_WEIGHT] = df['WTTRDFIN']

            # ── Location columns: None (public NHTS has no geography) ─────
            df[self.ORIGIN_LOC] = None
            df[self.DESTINATION_LOC] = None

            self.data = df
            self.metadata['source_year'] = year
            logger.info(f"Successfully extracted {len(df)} NHTS records")
            return df

        except KeyError:
            logger.error(f"Year {year} not found in config")
            raise
        except Exception as e:
            logger.error(f"Error extracting NHTS data: {e}")
            raise

    def clean_data(self, duration_std_multiplier: float = 3.0,
                   distance_std_multiplier: float = 3.0) -> None:
        """Clean NHTS data with NHTS-specific filters."""
        if self.data is None:
            raise ValueError("No data loaded. Call extract_data first.")

        try:
            df = self.data.copy()
            initial_count = len(df)
            logger.info(f"Starting NHTS data cleaning. Initial records: {initial_count}")

            # ── Basic validity filters ────────────────────────────────────
            df = df[
                (df[self.DURATION].notna()) &
                (df[self.DURATION] > 0) &
                (df[self.DISTANCE].notna()) &
                (df[self.DISTANCE] >= 0) &
                (df[self.DEPART_TIME].notna()) &
                (df[self.ARRIVE_TIME].notna()) &
                (df[self.ARRIVE_TIME] >= df[self.DEPART_TIME]) &
                (df[self.ORIGIN_PURPOSE].notna()) &
                (df[self.DESTINATION_PURPOSE].notna())
            ]

            after_basic = len(df)
            logger.info(f"After basic filters: {after_basic} records ({initial_count - after_basic} removed)")

            # ── Statistical outlier filtering: duration ────────────────────
            dur_mean = df[self.DURATION].mean()
            dur_std = df[self.DURATION].std()
            dur_threshold = dur_mean + duration_std_multiplier * dur_std

            logger.info(f"Duration stats: mean={dur_mean/60:.2f} min, std={dur_std/60:.2f} min")
            logger.info(f"Duration threshold (mean + {duration_std_multiplier}*std): {dur_threshold/60:.2f} min")

            before = len(df)
            df = df[df[self.DURATION] <= dur_threshold]
            logger.info(f"After duration filter: {len(df)} records ({before - len(df)} removed)")

            # ── Statistical outlier filtering: distance ────────────────────
            dist_mean = df[self.DISTANCE].mean()
            dist_std = df[self.DISTANCE].std()
            dist_threshold = dist_mean + distance_std_multiplier * dist_std

            logger.info(f"Distance stats: mean={dist_mean:.2f} mi, std={dist_std:.2f} mi")
            logger.info(f"Distance threshold (mean + {distance_std_multiplier}*std): {dist_threshold:.2f} mi")

            before = len(df)
            df = df[df[self.DISTANCE] <= dist_threshold]
            logger.info(f"After distance filter: {len(df)} records ({before - len(df)} removed)")

            # ── Drop raw NHTS columns, keep only canonical ────────────────
            canonical_cols = (
                {self.PERSON_ID, self.MODE_TYPE,
                 self.ORIGIN_LOC, self.DESTINATION_LOC,
                 self.ORIGIN_PURPOSE, self.DESTINATION_PURPOSE,
                 self.DEPART_TIME, self.ARRIVE_TIME,
                 self.DURATION, self.DISTANCE, self.TRIP_WEIGHT,
                 self.SOURCE_TYPE, self.SOURCE_YEAR}
            )
            # Keep only columns that exist in our canonical set
            cols_to_keep = [c for c in df.columns if c in canonical_cols]
            df = df[cols_to_keep]

            # ── Add source metadata columns ───────────────────────────────
            df[self.SOURCE_TYPE] = self.metadata['source_type']
            df[self.SOURCE_YEAR] = self.metadata.get('source_year', '')

            final_count = len(df)
            logger.info(
                f"NHTS cleaning complete. {initial_count - final_count} records removed. "
                f"Final: {final_count}"
            )

            self.data = df

            # ── Validate canonical schema ─────────────────────────────────
            self.validate_schema()

            # ── Detect census geography level from origin_loc values ──────
            self.detect_geo_level()

        except Exception as e:
            logger.error(f"Error cleaning NHTS data: {e}")
            raise
