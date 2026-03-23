# models.py
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Float, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, LargeBinary
from utils.duckdb_manager import DBManager

# Create a single Base instance to be shared
Base = declarative_base()

def initialize_tables(data_dir: str):
    """Initialize database tables"""
    db_manager = DBManager(data_dir)
    with db_manager.write_engine_scope() as write_engine:
        Base.metadata.create_all(write_engine)
    return db_manager

class SurveyTrip(Base):
    __tablename__ = 'survey_trips'

    id = Column(String, primary_key=True)
    person_id = Column(String)
    mode_type = Column(String)
    origin_loc = Column(String)
    destination_loc = Column(String)
    origin_purpose = Column(String)
    destination_purpose = Column(String)
    depart_time = Column(DateTime)
    arrive_time = Column(DateTime)
    duration_seconds = Column(Float, nullable=True)
    distance_miles = Column(Float, nullable=True)
    trip_weight = Column(Float, nullable=True)
    source_type = Column(String, nullable=True)
    source_year = Column(String, nullable=True)

# Define the HomeLocation model
class HomeLocation(Base):
    __tablename__ = 'home_locations'

    # Census block identifier (15-digit string)
    geoid = Column(String, primary_key=True)  # Required

    # Geographic identifiers (extracted from geoid for efficient querying)
    state_fips = Column(String, nullable=False, index=True)  # First 2 digits of geoid (e.g., "27" for MN)
    county_fips = Column(String, nullable=False, index=True)  # Digits 3-5 of geoid (e.g., "053" for Hennepin)

    # Number of employees living in the block (from LODES RAC C000 field)
    n_employees = Column(Integer, nullable=False, default=0)  # Required

    # Number of non-employees living in the block (from Census 2020 P1 - n_employees)
    non_employees = Column(Integer, nullable=True, default=0)  # Total population minus employees

    # Home location point coordinates (geographic coordinates in EPSG:4326)
    lat = Column(Float, nullable=True)  # Latitude of home location point
    lon = Column(Float, nullable=True)  # Longitude of home location point


class WorkLocation(Base):
    __tablename__ = 'work_locations'

    # Census block identifier (15-digit string)
    geoid = Column(String, primary_key=True)

    # Geographic identifiers (extracted from geoid for efficient querying)
    state_fips = Column(String, nullable=False, index=True)  # First 2 digits of geoid (e.g., "27" for MN)
    county_fips = Column(String, nullable=False, index=True)  # Digits 3-5 of geoid (e.g., "053" for Hennepin)

    # Number of employees working in the block
    n_employees = Column(Integer, nullable=False, default=0)

    # Work location point coordinates (geographic coordinates in EPSG:4326)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)

class POI(Base):
    __tablename__ = 'pois'

    osm_id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    activity = Column(String, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    tags = Column(String, nullable=True)  # JSON stored as string
    source = Column(String, nullable=True)  # 'county' or 'city'
    source_name = Column(String, nullable=True)
    state_fips = Column(String, nullable=False, index=True)   # First 2 digits of county GEOID (e.g., "27" for MN)
    county_fips = Column(String, nullable=False, index=True)   # Digits 3-5 of county GEOID (e.g., "053" for Hennepin)
    fetched_at = Column(DateTime, default=datetime.now)

class State(Base):
    __tablename__ = 'states'

    state_fips = Column(String, primary_key=True)  # "27" for Minnesota
    state_name = Column(String, unique=True, nullable=False)  # "Minnesota"
    state_abbr = Column(String, unique=True, nullable=False)  # "MN"
    geoid = Column(String, nullable=True)  # "27"
    aland = Column(Float, nullable=True)  # Land area in square meters
    awater = Column(Float, nullable=True)  # Water area in square meters
    created_at = Column(DateTime, default=datetime.now)

class County(Base):
    __tablename__ = 'counties'

    geoid = Column(String, primary_key=True)  # "27053" for Hennepin (state+county FIPS)
    state_fips = Column(String, nullable=False, index=True)  # "27" for Minnesota
    county_fips = Column(String, nullable=False)  # "053" for Hennepin
    county_name = Column(String, nullable=False)  # "Hennepin"
    county_name_full = Column(String, nullable=True)  # "Hennepin County"
    aland = Column(Float, nullable=True)  # Land area in square meters
    awater = Column(Float, nullable=True)  # Water area in square meters
    intptlat = Column(Float, nullable=True)  # Internal point latitude
    intptlon = Column(Float, nullable=True)  # Internal point longitude
    created_at = Column(DateTime, default=datetime.now)


# ── GTFS Tables (Phase 3: Transit Availability) ─────────────────────────────

class GTFSFeed(Base):
    __tablename__ = 'gtfs_feeds'

    feed_id = Column(String, primary_key=True)       # From Mobility Database 'mdb_source_id'
    provider = Column(String, nullable=True)          # Agency/provider name
    country_code = Column(String, nullable=True)      # 'US'
    subdivision = Column(String, nullable=True)        # State name, e.g. 'Minnesota'
    municipality = Column(String, nullable=True)       # City if available
    download_url = Column(Text, nullable=False)        # Direct download URL for GTFS zip
    bbox_min_lat = Column(Float, nullable=True)
    bbox_max_lat = Column(Float, nullable=True)
    bbox_min_lon = Column(Float, nullable=True)
    bbox_max_lon = Column(Float, nullable=True)
    downloaded_at = Column(DateTime, nullable=True)    # When feed was last downloaded
    gtfs_version = Column(String, nullable=True)       # From feed_info.txt if exists
    status = Column(String, nullable=True)             # active/deprecated/etc


class GTFSRoute(Base):
    __tablename__ = 'gtfs_routes'

    id = Column(Integer, autoincrement=False, primary_key=True)
    feed_id = Column(String, nullable=False, index=True)
    route_id = Column(String, nullable=False)          # From routes.txt
    route_short_name = Column(String, nullable=True)   # e.g., "5", "Blue"
    route_long_name = Column(String, nullable=True)    # e.g., "Chicago Ave Line"
    route_type = Column(Integer, nullable=False)       # 0=tram, 1=subway, 3=bus...
    agency_id = Column(String, nullable=True)          # From routes.txt
    agency_name = Column(String, nullable=True)        # From agency.txt


class GTFSTrip(Base):
    __tablename__ = 'gtfs_trips'

    id = Column(Integer, autoincrement=False, primary_key=True)
    feed_id = Column(String, nullable=False, index=True)
    trip_id = Column(String, nullable=False)           # From trips.txt
    route_pk = Column(Integer, nullable=False)         # References gtfs_routes.id
    service_id = Column(String, nullable=True)         # For calendar filtering
    trip_headsign = Column(String, nullable=True)
    direction_id = Column(Integer, nullable=True)      # 0 or 1
    shape_id = Column(String, nullable=True)           # Optional, for Phase 4


class GTFSStop(Base):
    __tablename__ = 'gtfs_stops'

    id = Column(Integer, autoincrement=False, primary_key=True)
    feed_id = Column(String, nullable=False, index=True)
    stop_id = Column(String, nullable=False)           # From stops.txt
    stop_name = Column(String, nullable=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    location_type = Column(Integer, nullable=True)     # 0=stop, 1=station
    parent_station = Column(String, nullable=True)


class GTFSStopTime(Base):
    __tablename__ = 'gtfs_stop_times'

    id = Column(Integer, autoincrement=False, primary_key=True)
    feed_id = Column(String, nullable=False, index=True)
    trip_pk = Column(Integer, nullable=False)          # References gtfs_trips.id
    stop_pk = Column(Integer, nullable=False)          # References gtfs_stops.id
    arrival_time = Column(String, nullable=True)       # HH:MM:SS (can be > 24:00)
    departure_time = Column(String, nullable=True)     # HH:MM:SS
    stop_sequence = Column(Integer, nullable=False)


class GTFSStopRoute(Base):
    """Derived table: which routes serve which stops (for fast availability queries)."""
    __tablename__ = 'gtfs_stop_routes'

    id = Column(Integer, autoincrement=False, primary_key=True)
    stop_pk = Column(Integer, nullable=False, index=True)   # References gtfs_stops.id
    route_pk = Column(Integer, nullable=False, index=True)  # References gtfs_routes.id


# ── FHA Traffic Counts Tables ─────────────────────────────────────────────────

class FHAStation(Base):
    """FHA/TMAS traffic monitoring station."""
    __tablename__ = 'fha_stations'

    id = Column(String, primary_key=True)            # "{state_code}_{station_id}" e.g. "27_000026"
    state_code = Column(String, nullable=False, index=True)   # 2-digit FIPS state code
    station_id = Column(String, nullable=False)       # Station ID within state
    lat = Column(Float, nullable=False)               # Decimal latitude
    lon = Column(Float, nullable=False)               # Decimal longitude (negative for US)
    county_code = Column(String, nullable=False, index=True)  # 3-digit FIPS county code
    f_system = Column(String, nullable=True)          # Functional system classification
    station_location = Column(String, nullable=True)  # Human-readable description
    year = Column(Integer, nullable=False)            # Data year


class FHAHourlyVolume(Base):
    """FHA/TMAS aggregated bidirectional hourly volumes (one row per station)."""
    __tablename__ = 'fha_hourly_volumes'

    id = Column(String, primary_key=True)             # "{state_code}_{station_id}"
    station_pk = Column(String, nullable=False, index=True)   # FK to fha_stations.id
    state_code = Column(String, nullable=False)       # 2-digit FIPS state code
    station_id = Column(String, nullable=False)       # Station ID within state
    h01 = Column(Float, nullable=True)
    h02 = Column(Float, nullable=True)
    h03 = Column(Float, nullable=True)
    h04 = Column(Float, nullable=True)
    h05 = Column(Float, nullable=True)
    h06 = Column(Float, nullable=True)
    h07 = Column(Float, nullable=True)
    h08 = Column(Float, nullable=True)
    h09 = Column(Float, nullable=True)
    h10 = Column(Float, nullable=True)
    h11 = Column(Float, nullable=True)
    h12 = Column(Float, nullable=True)
    h13 = Column(Float, nullable=True)
    h14 = Column(Float, nullable=True)
    h15 = Column(Float, nullable=True)
    h16 = Column(Float, nullable=True)
    h17 = Column(Float, nullable=True)
    h18 = Column(Float, nullable=True)
    h19 = Column(Float, nullable=True)
    h20 = Column(Float, nullable=True)
    h21 = Column(Float, nullable=True)
    h22 = Column(Float, nullable=True)
    h23 = Column(Float, nullable=True)
    h24 = Column(Float, nullable=True)
    num_weekdays_averaged = Column(Integer, nullable=True)  # Number of weekdays in the average

