# DuckDB Database

The file `trafficsim1.2.duckdb` is a pre-populated [DuckDB](https://duckdb.org/) database that contains reference data (states, counties, census blocks, POIs, GTFS transit data, etc.) used by the simulation.

## Tables

| Table | Description |
|---|---|
| `states` | US states |
| `counties` | US counties with geometry and population |
| `home_locations` | Census block home locations with population |
| `work_locations` | Census block work locations with employment |
| `pois` | Points of interest (shopping, schools, etc.) |
| `survey_trips` | Processed survey trip records |
| `gtfs_feeds` | GTFS transit feed metadata |
| `gtfs_routes` | Transit routes |
| `gtfs_trips` | Transit trips |
| `gtfs_stops` | Transit stops with coordinates |
| `gtfs_stop_times` | Stop arrival/departure times |
| `gtfs_stop_routes` | Stop-to-route mapping |
| `fha_stations` | FHA traffic count stations |
| `fha_hourly_volumes` | Hourly traffic volume observations |

## Connecting from Python

The project uses SQLAlchemy via `utils/duckdb_manager.py`. You don't need to connect manually — the `DBManager` class handles connections and locking automatically.

```python
from utils.duckdb_manager import DBManager

db = DBManager(data_dir="../data")

# Query example
with db.get_session() as session:
    counties = session.query(County).all()
```

## Connecting via the DuckDB CLI

1. Download the CLI from https://duckdb.org/docs/installation/
2. Open a terminal in this directory:

```bash
duckdb trafficsim1.2.duckdb
```

3. Common commands:

```sql
-- List all tables
.tables

-- Describe a table
DESCRIBE counties;

-- Query data
SELECT * FROM counties LIMIT 10;

-- Exit
.quit
```

## Important notes

- DuckDB does not support concurrent access from multiple processes on Windows. The project uses file locking to handle this, but if you get a lock error, make sure no other Python process or CLI session has the database open.
- If you need to reset the database, delete `trafficsim1.2.duckdb` and re-run the experiment — tables will be recreated automatically.
