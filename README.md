# Tareek

An agent-based travel demand model that generates synthetic populations and activity plans for [MATSim](https://matsim.org/) traffic simulations. Given a set of US counties, it builds a complete simulation from census data, survey trips, transit feeds, and road networks.

## Quick Start

### 1. Prerequisites

- Python 3.13+
- Java 17+ (for MATSim)

### 2. Setup

```bash
git clone https://github.com/YOUR_USERNAME/Tareek.git
cd Tareek

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure

Copy one of the example configs from `config/USA/` and edit it:

```bash
cp config/USA/TwinCities/config_twin.json config/config_local.json
```

At minimum, set:

```json
{
  "region": {
    "counties": ["27053", "27123"]
  },
  "data": {
    "data_dir": "data",
    "surveys": [
      {
        "type": "nhts",
        "year": "2022",
        "file": "nhts/csv/tripv2pub.csv",
        "weight": 1
      }
    ]
  }
}
```

- `counties` — FIPS GEOIDs (2-digit state + 3-digit county). Find codes at [census.gov](https://www.census.gov/library/reference/code-lists/ansi.html).
- `surveys` — at least one survey with `weight > 0`.

### 4. Run

```bash
python run_experiment.py --config config/config_local.json
```

Options:

| Flag | Description |
|---|---|
| `--config` | Path to config JSON (required) |
| `--experiment-id` | Custom experiment name (optional, auto-generated) |
| `--skip-simulation` | Generate plans only, don't run MATSim |

Output goes to `experiments/<experiment-id>/`.

---

## Survey Data

The system needs household travel survey data to generate realistic trip patterns. At least one survey must be configured with `weight > 0`.

### NHTS (National Household Travel Survey)

The NHTS 2022 trip file is included in the repo at `data/nhts/csv/tripv2pub.csv`. It is a national survey that works as a default for any US region. No download needed.

If you want to use a newer NHTS release, download it from [NHTS](https://nhts.ornl.gov/) and replace the file.

```json
{
  "type": "nhts",
  "year": "2022",
  "file": "nhts/csv/tripv2pub.csv",
  "weight": 1
}
```

### Adding a Regional Survey

The NHTS provides a good baseline, but a region-specific survey will produce better results. To add your own survey, create a new file in `data_sources/` that subclasses `BaseSurveyTrip` (see `data_sources/tbi_survey.py` as an example):

```python
from data_sources.base_survey_trip import BaseSurveyTrip

class MySurveyTrip(BaseSurveyTrip):
    def extract_data(self, year: str):
        # Read your CSV/file into self.data (a pandas DataFrame)
        ...

    def clean_data(self, **kwargs):
        # Map your raw columns to the canonical schema:
        #   person_id, mode_type, origin_purpose, destination_purpose,
        #   depart_time, arrive_time
        #
        # Map activities to: Home, Work, School, Shopping, Social, Dining, Escort, Other
        # Map modes to: Car, Bus, Rail, Walk, Bike, SchoolBus, Rideshare, Other
        #
        # Add metadata columns:
        #   self.data['source_type'] = 'my_survey'
        #   self.data['source_year'] = year
        #
        # Then validate:
        self.validate_schema()
```

Register it in `data_sources/survey_manager.py`:

```python
# In SurveyManager._ensure_registry():
from data_sources.my_survey import MySurveyTrip
cls.SURVEY_REGISTRY['my_survey'] = MySurveyTrip
```

Use it in your config:

```json
{ "type": "my_survey", "year": "2024", "file": "path/to/data.csv", "weight": 1 }
```

### Multi-Survey Blending

You can enable multiple surveys with different weights. Weights are normalized automatically:

```json
"surveys": [
  { "type": "my_survey", "year": "2024", "file": "...", "weight": 0.7 },
  { "type": "nhts",      "year": "2022", "file": "...", "weight": 0.3 }
]
```

Set `weight: 0` to disable a survey without removing it from the config.

---

## Traffic Counts

Traffic counts are used for MATSim calibration and post-simulation evaluation. The repo includes FHA/TMAS data that covers all US states. Configure in the `counts` section.

### FHA/TMAS Counts (Included)

The FHA station and volume zip files are included in `data/FHA_counts/`. These are national data — the system automatically filters to your configured counties.

```json
"counts": {
  "enabled": true,
  "fha": {
    "data_dir": "data/FHA_counts",
    "year": 2024,
    "month": 7,
    "weight": 1
  }
}
```

To use a different month or year, download updated files from [FHWA TMAS](https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/) and replace the zips.

### Custom Counts (CSV)

You can provide your own traffic count data as two CSV files in `data/evaluation/`:

**counts_stations.csv** — one row per station:

```csv
station_id,latitude,longitude
10069,44.81059,-93.21900
10205,44.83642,-92.96883
```

**counts_volumes.csv** — hourly volumes per station per day:

```csv
station_id,date,h01,h02,h03,...,h24
10069,2024-05-23,587,391,232,...,1150
10069,2024-10-25,619,343,279,...,1718
```

- `h01`–`h24` are hourly volumes (h01 = midnight to 1am)
- Multiple days per station are averaged (weekdays only)

```json
"counts": {
  "enabled": true,
  "fha": { "weight": 0 },
  "custom": { "enabled": true, "weight": 1 }
}
```

### Blending FHA + Custom

Set both weights to blend the two sources:

```json
"counts": {
  "enabled": true,
  "fha": { "data_dir": "data/FHA_counts", "year": 2024, "month": 7, "weight": 0.5 },
  "custom": { "enabled": true, "weight": 0.5 }
}
```

---

## Evaluation

The `data/evaluation/` folder contains ground-truth traffic count data for post-simulation comparison. The included example files are for the Twin Cities area — replace them with data for your region.

| File | Description |
|---|---|
| `counts_stations.csv` | Station locations (station_id, lat, lon) |
| `counts_volumes.csv` | Observed hourly volumes per station |

```json
"evaluation": {
  "run_evaluation": true,
  "ground_truth_data_dir": "data/evaluation"
}
```

---

## Config Wizard (Web UI)

A web-based wizard for building config files interactively — select counties on a map, configure modes, set scaling factors, and export a ready-to-use `config.json`.

```bash
cd webapp
python run.py
```

Opens at **http://localhost:8000**. See [webapp/README.md](webapp/README.md) for details.

---

## Database

The included DuckDB database (`data/db/trafficsim1.2.duckdb`) contains pre-populated reference data (US states, counties, census blocks). See [data/db/README.md](data/db/README.md) for details on connecting and querying.

---

## Project Structure

```
Tareek/
  run_experiment.py          # Main entry point
  config/
    USA/                     # Example configs for different cities
  data/
    db/                      # DuckDB database with reference data
    counties/                # US county shapefiles
    nhts/                    # NHTS survey data (included)
    FHA_counts/              # FHA traffic counts (included)
    evaluation/              # Ground-truth counts for validation (region-specific)
  data_sources/              # Survey and counts data loaders
  models/                    # Plan generation, OD matrices, mode choice
  matsim/                    # Network generation, MATSim runner, evaluation
  utils/                     # DB manager, logging, spatial utilities
  webapp/                    # Optional web visualization
```
