# Simulation Config Wizard

A web-based step-by-step wizard for building MATSim simulation configuration files.

## Prerequisites

- Python 3.9+ with the project's `.venv` activated
- Required packages: `fastapi`, `uvicorn`, `geopandas`, `requests`

## Setup

1. **Install dependencies** (from project root):

   ```bash
   .venv/Scripts/pip install fastapi "uvicorn[standard]"
   ```

   (`geopandas` and `requests` should already be installed in the project venv.)

2. **Ensure county shapefile exists** at `data/counties/cb_2022_us_county_500k.shp`.
   If missing, the server will auto-download it on first startup from the Census Bureau.

## Running

From the `webapp/` directory:

```bash
cd webapp
../.venv/Scripts/python run.py
```

The server starts at **http://localhost:8000**.

## First Startup

On first launch, the backend will:
1. Convert the county shapefile to simplified GeoJSON (~30s)
2. Fetch county population data from the Census API (~10s)

These are cached to `data/counties/counties.geojson` and `data/counties/county_population.json` so subsequent starts are instant.

## Structure

```
webapp/
  backend/
    main.py              # FastAPI server, API endpoints, config assembly
    requirements.txt     # Backend dependencies
  frontend/
    index.html           # Main wizard page
    css/style.css        # Design system (single stylesheet)
    js/app.js            # Wizard logic, map, form state, export
  run.py                 # Entry point (uvicorn launcher)
```

## Wizard Panels

1. **Region** — Select counties on an interactive map (population capped at 5M)
2. **Surveys** — Choose NHTS (default) or upload custom survey data
3. **Transportation Modes** — Enable/disable car, bus, rail, walk, bike with per-mode settings
4. **Transit** — GTFS feed settings and pt2matsim mapping parameters
5. **Travel Demand** — Scaling factor, activity chains, non-work trip purposes
6. **Time & Duration** — Activity duration constraints, time model settings
7. **Simulation** — MATSim iterations, capacity factors, traffic counts, evaluation

The final step exports a complete `config.json` ready for `run_experiment.py`.
