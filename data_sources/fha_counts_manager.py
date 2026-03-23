"""
FHA/TMAS Traffic Counts Manager

Handles extraction, transformation, and loading of Federal Highway Administration
Traffic Monitoring Analysis System (TMAS) data. Reads pipe-delimited .STA (station)
and .VOL (volume) files from zip archives, filters by configured counties, aggregates
to bidirectional hourly averages, and stores results in the database.

Follows the GTFSManager pattern: a manager with a setup() method called from
run_experiment.py, with data cached in the DB so ETL only runs once per region.
"""

import zipfile
from pathlib import Path
from typing import Any, Dict, Set

import pandas as pd

from models.models import FHAStation, FHAHourlyVolume
from utils.logger import setup_logger

logger = setup_logger(__name__)

# FIPS state code -> 2-letter postal abbreviation (all 50 states + DC)
STATE_FIPS_TO_ABBR = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA',
    '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL',
    '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN',
    '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME',
    '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS',
    '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH',
    '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND',
    '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
    '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT',
    '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI',
    '56': 'WY',
}

# Opposite direction pairs per FHWA coding: 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW
OPPOSITE_DIRS = {1: 5, 5: 1, 3: 7, 7: 3, 2: 6, 6: 2, 4: 8, 8: 4}


class FHACountsManager:
    """
    Manages FHA/TMAS traffic count data: discovery, extraction, aggregation, and DB loading.

    Usage:
        manager = FHACountsManager(config, db_manager)
        success = manager.setup()
    """

    def __init__(self, config: Dict[str, Any], db_manager):
        self.config = config
        self.db_manager = db_manager

        counts_config = config.get('counts', {})
        fha_config = counts_config.get('fha', {})

        self.data_dir = Path(fha_config.get('data_dir', 'data/FHA_counts'))
        self.year = fha_config.get('year', 2024)
        self.month = fha_config.get('month', 7)

        # Month number -> 3-letter abbreviation for volume file names
        self._month_abbrs = {
            1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN',
            7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC',
        }

    def setup(self) -> bool:
        """
        Full ETL pipeline: extract from zips, filter, aggregate, load to DB.

        Returns:
            True if data was loaded successfully, False otherwise.
        """
        logger.info("Setting up FHA counts data...")

        if self.has_data_for_region():
            logger.info("FHA counts data already loaded in DB — skipping ETL")
            return True

        # Determine which states/counties we need
        needed = self._get_needed_states()
        if not needed:
            logger.warning("FHA: no counties configured — cannot load FHA data")
            return False

        # Find zip files
        station_zip = self.data_dir / f"{self.year}_station_data.zip"
        month_abbr = self._month_abbrs.get(self.month, 'JUL').lower()
        volume_zip = self.data_dir / f"{month_abbr}_{self.year}_ccs_data.zip"

        if not station_zip.exists():
            logger.error(f"FHA station zip not found: {station_zip}")
            return False
        if not volume_zip.exists():
            logger.error(f"FHA volume zip not found: {volume_zip}")
            return False

        all_stations = []
        all_volumes = []

        for state_fips, county_codes in needed.items():
            state_abbr = self._state_fips_to_abbr(state_fips)
            if not state_abbr:
                logger.warning(f"FHA: unknown state FIPS '{state_fips}' — skipping")
                continue

            logger.info(f"FHA: parsing stations for state {state_abbr}, "
                        f"filtering to counties {sorted(county_codes)}")
            stations_df = self._parse_stations(station_zip, state_abbr, state_fips, county_codes)

            if stations_df.empty:
                logger.warning(f"FHA: no stations found for state {state_abbr} "
                               f"counties {sorted(county_codes)}")
                continue

            station_ids = set(stations_df['station_id'].unique())
            logger.info(f"FHA: parsing volumes for state {state_abbr}, "
                        f"{len(station_ids)} station IDs")
            volumes_df = self._parse_volumes(volume_zip, state_abbr, state_fips, station_ids)

            all_stations.append(stations_df)
            if not volumes_df.empty:
                all_volumes.append(volumes_df)

        if not all_stations:
            logger.warning("FHA: no stations found for configured counties — "
                           "counts.xml will not be generated")
            return False

        stations_combined = pd.concat(all_stations, ignore_index=True)
        if not all_volumes:
            logger.warning("FHA: no volume data found — counts.xml will not be generated")
            return False

        volumes_combined = pd.concat(all_volumes, ignore_index=True)

        # Aggregate to bidirectional hourly averages
        agg_volumes = self._aggregate_to_bidirectional(volumes_combined)

        # Filter stations to only those with volume data
        stations_with_vol = stations_combined[
            stations_combined['station_id'].isin(agg_volumes['station_id'].unique())
        ].copy()

        # Load to DB
        self._load_to_db(stations_with_vol, agg_volumes)

        logger.info(f"FHA: loaded {len(stations_with_vol)} stations, "
                    f"{len(agg_volumes)} volume records for "
                    f"{len(self._get_needed_states())} state(s)")
        return True

    def has_data_for_region(self) -> bool:
        """Check if FHA data is already loaded in the DB for the configured region.

        Returns True if at least one station exists for each needed state.
        Some counties may legitimately have no FHA stations — that's not a reason
        to re-run the ETL.
        """
        needed = self._get_needed_states()
        if not needed:
            return False

        for state_fips in needed:
            results = self.db_manager.query_all(
                FHAStation, filters={'state_code': state_fips}
            )
            if not results:
                return False
        return True

    def _get_needed_states(self) -> Dict[str, Set[str]]:
        """
        From configured county GEOIDs, build {state_fips: {county_code, ...}}.

        Returns:
            Dict mapping 2-digit state FIPS to set of 3-digit county codes.
        """
        region_config = self.config.get('region', {})
        counties = region_config.get('counties') or self.config.get('network', {}).get('counties', [])

        result = {}
        for geoid in counties:
            state_fips = geoid[:2]
            county_code = geoid[2:]
            result.setdefault(state_fips, set()).add(county_code)
        return result

    def _state_fips_to_abbr(self, state_fips: str) -> str:
        """Convert 2-digit FIPS state code to 2-letter postal abbreviation."""
        return STATE_FIPS_TO_ABBR.get(state_fips, '')

    def _parse_stations(self, zip_path: Path, state_abbr: str,
                        state_fips: str, county_codes: Set[str]) -> pd.DataFrame:
        """
        Parse station data from a .STA file inside the zip, filtered by county.

        Returns:
            DataFrame with columns: station_id, lat, lon, county_code, f_system,
            station_location, state_code, year. One row per unique station
            (deduplicated from per-lane rows).
        """
        entry_name = f"{state_abbr}_{self.year} (TMAS).STA"

        try:
            with zipfile.ZipFile(zip_path) as zf:
                with zf.open(entry_name) as f:
                    df = pd.read_csv(f, sep='|', dtype=str, encoding='utf-8',
                                     on_bad_lines='skip')
        except KeyError:
            logger.warning(f"FHA: entry '{entry_name}' not found in {zip_path.name}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"FHA: error reading {entry_name}: {e}")
            return pd.DataFrame()

        # Strip whitespace from column values
        for col in ['station_id', 'county_code', 'latitude', 'longitude']:
            if col in df.columns:
                df[col] = df[col].str.strip()

        # Filter to our counties
        df = df[df['county_code'].isin(county_codes)].copy()
        if df.empty:
            return pd.DataFrame()

        # Convert lat/lon: scaled integers → decimal, negate longitude for US
        df['lat'] = df['latitude'].astype(float) / 1_000_000
        df['lon'] = -df['longitude'].astype(float) / 1_000_000

        # Deduplicate: keep one row per station_id (multiple rows per lane/dir)
        stations = df.drop_duplicates(subset='station_id', keep='first')

        result = pd.DataFrame({
            'station_id': stations['station_id'],
            'lat': stations['lat'],
            'lon': stations['lon'],
            'county_code': stations['county_code'],
            'f_system': stations['f_system'].str.strip() if 'f_system' in stations.columns else '',
            'station_location': (stations['station_location'].str.strip()
                                 if 'station_location' in stations.columns else ''),
            'state_code': state_fips,
            'year': self.year,
        })

        return result.reset_index(drop=True)

    def _parse_volumes(self, zip_path: Path, state_abbr: str,
                       state_fips: str, station_ids: Set[str]) -> pd.DataFrame:
        """
        Parse volume data from a .VOL file inside the zip, filtered to given stations.

        Returns:
            DataFrame with columns: station_id, state_code, travel_dir, travel_lane,
            day_of_week, hour_00..hour_23
        """
        month_abbr = self._month_abbrs.get(self.month, 'JUL')
        entry_name = f"{state_abbr}_{month_abbr}_{self.year} (TMAS).VOL"

        try:
            with zipfile.ZipFile(zip_path) as zf:
                with zf.open(entry_name) as f:
                    df = pd.read_csv(f, sep='|', dtype=str, encoding='utf-8',
                                     on_bad_lines='skip')
        except KeyError:
            logger.warning(f"FHA: entry '{entry_name}' not found in {zip_path.name}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"FHA: error reading {entry_name}: {e}")
            return pd.DataFrame()

        # Strip station_id whitespace
        if 'station_id' in df.columns:
            df['station_id'] = df['station_id'].str.strip()

        # Filter to our stations
        df = df[df['station_id'].isin(station_ids)].copy()
        if df.empty:
            return pd.DataFrame()

        # Convert numeric columns
        hour_cols = [f'hour_{i:02d}' for i in range(24)]
        for col in hour_cols + ['travel_dir', 'travel_lane', 'day_of_week']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['state_code'] = state_fips

        return df

    def _aggregate_to_bidirectional(self, volumes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate raw volume data to one row per station with 24 bidirectional
        hourly averages:
          1. Filter to weekdays only (day_of_week 2-6 = Mon-Fri)
          2. Sum across lanes per (station, direction, day, hour)
          3. Average across weekdays per (station, direction, hour)
          4. Sum opposite direction pairs per station

        Returns:
            DataFrame with columns: station_id, state_code, h01..h24, num_weekdays_averaged
        """
        hour_cols = [f'hour_{i:02d}' for i in range(24)]

        # Step 1: Filter weekdays (2=Mon .. 6=Fri)
        df = volumes_df[volumes_df['day_of_week'].between(2, 6)].copy()
        if df.empty:
            logger.warning("FHA: no weekday volume data found")
            return pd.DataFrame()

        # Step 2: Sum across lanes per (station, direction, day_of_week, date-like grouping)
        # Since we have per-day rows, group by station+dir+day_record to sum lanes
        day_col = 'day_record' if 'day_record' in df.columns else 'day_of_week'
        group_cols = ['station_id', 'state_code', 'travel_dir', day_col]
        lane_summed = df.groupby(group_cols, as_index=False)[hour_cols].sum()

        # Step 3: Average across weekdays per (station, direction, hour)
        dir_avg = lane_summed.groupby(
            ['station_id', 'state_code', 'travel_dir'], as_index=False
        )[hour_cols].mean()

        # Count weekdays per station for metadata
        weekday_counts = lane_summed.groupby('station_id')[day_col].nunique().reset_index()
        weekday_counts.columns = ['station_id', 'num_weekdays_averaged']

        # Step 4: Sum opposite direction pairs per station
        # Group by station and sum all directions (bidirectional total)
        bidir = dir_avg.groupby(
            ['station_id', 'state_code'], as_index=False
        )[hour_cols].sum()

        # Rename hour_00..hour_23 -> h01..h24
        rename_map = {f'hour_{i:02d}': f'h{i+1:02d}' for i in range(24)}
        bidir = bidir.rename(columns=rename_map)

        # Merge weekday counts
        bidir = bidir.merge(weekday_counts, on='station_id', how='left')

        return bidir

    def _load_to_db(self, stations_df: pd.DataFrame, volumes_df: pd.DataFrame):
        """Load stations and volumes to the database."""
        # Build station records
        station_records = []
        for _, row in stations_df.iterrows():
            pk = f"{row['state_code']}_{row['station_id']}"
            station_records.append({
                'id': pk,
                'state_code': str(row['state_code']),
                'station_id': str(row['station_id']),
                'lat': float(row['lat']),
                'lon': float(row['lon']),
                'county_code': str(row['county_code']),
                'f_system': str(row.get('f_system', '')),
                'station_location': str(row.get('station_location', '')),
                'year': int(row['year']),
            })

        # Build volume records
        hour_cols = [f'h{i:02d}' for i in range(1, 25)]
        volume_records = []
        for _, row in volumes_df.iterrows():
            pk = f"{row['state_code']}_{row['station_id']}"
            rec = {
                'id': pk,
                'station_pk': pk,
                'state_code': str(row['state_code']),
                'station_id': str(row['station_id']),
                'num_weekdays_averaged': int(row.get('num_weekdays_averaged', 0)),
            }
            for hcol in hour_cols:
                rec[hcol] = float(row[hcol]) if pd.notna(row.get(hcol)) else 0.0
            volume_records.append(rec)

        # Insert to DB
        if station_records:
            self.db_manager.insert_records(FHAStation, station_records)
        if volume_records:
            self.db_manager.insert_records(FHAHourlyVolume, volume_records)
