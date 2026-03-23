"""
Config Wizard Backend - FastAPI server for the simulation config builder.

Serves:
  - Static frontend files
  - /api/counties  -> GeoJSON county boundaries + population
  - /api/export    -> Assemble and download config.json
"""

import json
import zipfile
import io
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import geopandas as gpd
import requests

app = FastAPI(title="Simulation Config Wizard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # e:/projects/code
DATA_DIR = PROJECT_ROOT / "data"
COUNTIES_DIR = DATA_DIR / "counties"
SHAPEFILE_PATH = COUNTIES_DIR / "cb_2022_us_county_500k.shp"
COUNTY_GEOJSON_CACHE = COUNTIES_DIR / "counties.geojson"
COUNTY_POP_CACHE = COUNTIES_DIR / "county_population.json"
SHAPEFILE_URL = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_500k.zip"

# Census API for population (ACS 5-year, most recent)
CENSUS_POP_URL = "https://api.census.gov/data/2022/acs/acs5"

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Default config template (reference)
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "USA" / "LA" / "config_la.json"


def ensure_shapefile():
    """Download county shapefile if not already cached."""
    if SHAPEFILE_PATH.exists():
        return
    COUNTIES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading county shapefile from {SHAPEFILE_URL}...")
    resp = requests.get(SHAPEFILE_URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(COUNTIES_DIR)
    print("County shapefile downloaded and extracted.")


def build_county_geojson():
    """
    Convert shapefile to simplified GeoJSON for frontend map.
    Cached to disk so we only do this once.
    """
    if COUNTY_GEOJSON_CACHE.exists():
        return

    ensure_shapefile()
    print("Building county GeoJSON (simplified for web)...")
    gdf = gpd.read_file(SHAPEFILE_PATH)

    # Build 5-digit GEOID from STATEFP + COUNTYFP
    gdf["GEOID"] = gdf["STATEFP"] + gdf["COUNTYFP"]

    # Simplify geometry for faster rendering (tolerance in degrees ~500m)
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.005, preserve_topology=True)

    # Keep only needed columns
    gdf = gdf[["GEOID", "STATEFP", "COUNTYFP", "NAMELSAD", "ALAND", "geometry"]]
    gdf = gdf.rename(columns={"NAMELSAD": "name"})

    # Convert ALAND to sq miles for display
    gdf["area_sq_mi"] = (gdf["ALAND"] / 2_589_988).round(1)
    gdf = gdf.drop(columns=["ALAND"])

    gdf.to_file(COUNTY_GEOJSON_CACHE, driver="GeoJSON")
    print(f"County GeoJSON cached: {COUNTY_GEOJSON_CACHE}")


def build_population_lookup():
    """
    Fetch county-level population from Census ACS API.
    Builds a JSON lookup: { "GEOID": population, ... }
    Falls back to a simplified estimate from land area if API fails.
    """
    if COUNTY_POP_CACHE.exists():
        return

    print("Fetching county population data from Census API...")
    pop_data = {}

    try:
        # ACS 5-year: B01003_001E = Total Population
        params = {
            "get": "B01003_001E,NAME",
            "for": "county:*",
        }
        resp = requests.get(CENSUS_POP_URL, params=params, timeout=60)
        resp.raise_for_status()
        rows = resp.json()

        # First row is header: ['B01003_001E', 'NAME', 'state', 'county']
        for row in rows[1:]:
            pop_str, name, state_fips, county_fips = row
            geoid = f"{state_fips}{county_fips}"
            try:
                pop_data[geoid] = int(pop_str)
            except (ValueError, TypeError):
                pop_data[geoid] = 0

        print(f"Loaded population for {len(pop_data)} counties")

    except Exception as e:
        print(f"Census API failed: {e}")
        print("Population data will not be available for county selection cap.")

    COUNTIES_DIR.mkdir(parents=True, exist_ok=True)
    with open(COUNTY_POP_CACHE, "w") as f:
        json.dump(pop_data, f)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    build_county_geojson()
    build_population_lookup()


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/counties")
async def get_counties():
    """Return county GeoJSON with population data merged in."""
    if not COUNTY_GEOJSON_CACHE.exists():
        raise HTTPException(500, "County GeoJSON not built yet")

    with open(COUNTY_GEOJSON_CACHE, "r") as f:
        geojson = json.load(f)

    # Merge population
    pop_data = {}
    if COUNTY_POP_CACHE.exists():
        with open(COUNTY_POP_CACHE, "r") as f:
            pop_data = json.load(f)

    for feature in geojson.get("features", []):
        geoid = feature["properties"].get("GEOID", "")
        feature["properties"]["population"] = pop_data.get(geoid, 0)

    return JSONResponse(geojson)


@app.post("/api/export")
async def export_config(request: Request):
    """
    Receive wizard form data and assemble a complete config.json.
    Returns the config as a JSON response for download.
    """
    try:
        form_data = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    config = assemble_config(form_data)
    return JSONResponse(config)


@app.get("/api/defaults")
async def get_defaults():
    """Return default config values for pre-populating the wizard."""
    defaults = get_default_config()
    return JSONResponse(defaults)


def get_default_config():
    """Load defaults from the reference config, stripping comments."""
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            config = json.load(f)
        # Strip _comment and _*_help keys for cleaner defaults
        return strip_comments(config)
    return {}


def strip_comments(obj):
    """Recursively remove keys starting with _ from dicts."""
    if isinstance(obj, dict):
        return {k: strip_comments(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [strip_comments(item) for item in obj]
    return obj


def assemble_config(form_data: dict) -> dict:
    """
    Build a complete config.json from wizard form data.
    Merges user selections with sensible defaults.
    """
    defaults = get_default_config()

    config = {
        "region": {
            "counties": form_data.get("counties", [])
        },
        "data": {
            "data_dir": defaults.get("data", {}).get("data_dir", "data"),
            "lodes": defaults.get("data", {}).get("lodes", {
                "year": 2023,
                "job_type": "JT00",
                "segment": "S000"
            }),
            "surveys": form_data.get("surveys", defaults.get("data", {}).get("surveys", []))
        },
        "gtfs": _build_gtfs_config(form_data, defaults),
        "modes": _build_modes_config(form_data, defaults),
        "mode_choice": _build_mode_choice_config(form_data, defaults),
        "plan_generation": _build_plan_generation_config(form_data, defaults),
        "od_matrix": form_data.get("od_matrix", defaults.get("od_matrix", {})),
        "time_models": form_data.get("time_models", defaults.get("time_models", {})),
        "duration_constraints": _build_duration_config(form_data, defaults),
        "poi_assignment": form_data.get("poi_assignment", defaults.get("poi_assignment", {})),
        "coordinates": {
            "precision_decimal_places": 2
        },
        "chains": _build_chains_config(form_data, defaults),
        "logging": {
            "level": "DEBUG",
            "show_progress_bar": True,
            "log_to_file": True,
            "log_to_console": True,
            "verbose": False
        },
        "network": {
            "clean_network": True,
            "rebuild_network": False
        },
        "matsim": _build_matsim_config(form_data, defaults),
        "counts": _build_counts_config(form_data, defaults),
        "evaluation": form_data.get("evaluation", defaults.get("evaluation", {})),
        "nonwork_purposes": _build_nonwork_config(form_data, defaults),
    }

    return config


def _build_gtfs_config(form_data, defaults):
    gtfs_defaults = defaults.get("gtfs", {})
    gtfs = form_data.get("gtfs", {})
    return {
        "catalog_url": gtfs.get("catalog_url", gtfs_defaults.get("catalog_url", "https://files.mobilitydatabase.org/feeds_v2.csv")),
        "catalog_max_age_days": gtfs.get("catalog_max_age_days", gtfs_defaults.get("catalog_max_age_days", 7)),
        "feed_max_age_days": gtfs.get("feed_max_age_days", gtfs_defaults.get("feed_max_age_days", 30)),
        "cache_dir": "data/gtfs",
        "country_filter": "US",
        "api_keys": gtfs.get("api_keys", gtfs_defaults.get("api_keys", {}))
    }


def _build_modes_config(form_data, defaults):
    return form_data.get("modes", defaults.get("modes", {}))


def _build_mode_choice_config(form_data, defaults):
    mc = form_data.get("mode_choice", {})
    mc_defaults = defaults.get("mode_choice", {})
    return {
        "method": mc.get("method", mc_defaults.get("method", "survey_rates")),
        "fallback_mode": mc.get("fallback_mode", mc_defaults.get("fallback_mode", "car")),
        "chain_consistency": mc.get("chain_consistency", mc_defaults.get("chain_consistency", True)),
        "min_samples_per_purpose": mc.get("min_samples_per_purpose", mc_defaults.get("min_samples_per_purpose", 30)),
        "max_chain_mode_retries": mc.get("max_chain_mode_retries", mc_defaults.get("max_chain_mode_retries", 10)),
    }


def _build_plan_generation_config(form_data, defaults):
    pg = form_data.get("plan_generation", {})
    pg_defaults = defaults.get("plan_generation", {})
    return {
        "target_plans": pg.get("target_plans", pg_defaults.get("target_plans", "all")),
        "scaling_factor": pg.get("scaling_factor", pg_defaults.get("scaling_factor", 0.1)),
        "work_scaling_multiplier": pg.get("work_scaling_multiplier", pg_defaults.get("work_scaling_multiplier", 1.0)),
        "random_seed": pg.get("random_seed", pg_defaults.get("random_seed", 42)),
        "skip_if_exists": pg.get("skip_if_exists", pg_defaults.get("skip_if_exists", True)),
        "supported_chain_types": pg.get("supported_chain_types", pg_defaults.get("supported_chain_types", ["home_work_home"])),
        "chain_sampling_method": pg.get("chain_sampling_method", pg_defaults.get("chain_sampling_method", "generated")),
        "max_chain_retries": pg.get("max_chain_retries", pg_defaults.get("max_chain_retries", 10)),
        "num_processes": pg.get("num_processes", pg_defaults.get("num_processes", 30)),
        "default_mode": pg.get("default_mode", pg_defaults.get("default_mode", "car")),
    }


def _build_chains_config(form_data, defaults):
    ch = form_data.get("chains", {})
    ch_defaults = defaults.get("chains", {})
    return {
        "home_boost_factor": ch.get("home_boost_factor", ch_defaults.get("home_boost_factor", 1.0)),
        "use_weighted_chains": ch.get("use_weighted_chains", ch_defaults.get("use_weighted_chains", True)),
        "max_length": ch.get("max_length", ch_defaults.get("max_length", None)),
        "min_length": ch.get("min_length", ch_defaults.get("min_length", 3)),
        "max_work_activities": ch.get("max_work_activities", ch_defaults.get("max_work_activities", 2)),
        "early_stop_exponent": ch.get("early_stop_exponent", ch_defaults.get("early_stop_exponent", 2.0)),
    }


def _build_matsim_config(form_data, defaults):
    m = form_data.get("matsim", {})
    m_defaults = defaults.get("matsim", {})
    cp = m.get("configurable_params", {})
    cp_defaults = m_defaults.get("configurable_params", {})
    return {
        "version": "matsim_25",
        "mode": m.get("mode", m_defaults.get("mode", "basic")),
        "run_simulation": m.get("run_simulation", m_defaults.get("run_simulation", True)),
        "heap_size_gb": m_defaults.get("heap_size_gb", 70),
        "transit_network": m.get("transit_network", m_defaults.get("transit_network", True)),
        "gtfs_sample_day": m.get("gtfs_sample_day", m_defaults.get("gtfs_sample_day", "dayWithMostTrips")),
        "pt2matsim": m.get("pt2matsim", m_defaults.get("pt2matsim", {})),
        "configurable_params": {
            "coordinateSystem": "auto",
            "lastIteration": cp.get("lastIteration", cp_defaults.get("lastIteration", 100)),
            "outputDirectory": "output",
            "global.numberOfThreads": m_defaults.get("configurable_params", {}).get("global.numberOfThreads", 30),
            "qsim.numberOfThreads": m_defaults.get("configurable_params", {}).get("qsim.numberOfThreads", 30),
            "qsim.flowCapacityFactor": cp.get("qsim.flowCapacityFactor", cp_defaults.get("qsim.flowCapacityFactor", 0.1)),
            "qsim.storageCapacityFactor": cp.get("qsim.storageCapacityFactor", cp_defaults.get("qsim.storageCapacityFactor", 0.12)),
            "linkStats.averageLinkStatsOverIterations": cp.get("linkStats.averageLinkStatsOverIterations", cp_defaults.get("linkStats.averageLinkStatsOverIterations", 5)),
        }
    }


def _build_counts_config(form_data, defaults):
    c = form_data.get("counts", {})
    c_defaults = defaults.get("counts", {})
    fha = c.get("fha", {})
    fha_defaults = c_defaults.get("fha", {})
    return {
        "enabled": c.get("enabled", c_defaults.get("enabled", True)),
        "rebuild": c.get("rebuild", c_defaults.get("rebuild", False)),
        "fha": {
            "data_dir": "data/FHA_counts",
            "year": fha.get("year", fha_defaults.get("year", 2024)),
            "month": fha.get("month", fha_defaults.get("month", 7)),
            "weight": fha.get("weight", fha_defaults.get("weight", 1)),
        },
        "custom": {
            "enabled": c.get("custom", {}).get("enabled", False),
            "weight": c.get("custom", {}).get("weight", 0),
        }
    }


def _build_duration_config(form_data, defaults):
    return form_data.get("duration_constraints", defaults.get("duration_constraints", {}))


def _build_nonwork_config(form_data, defaults):
    return form_data.get("nonwork_purposes", defaults.get("nonwork_purposes", {}))


# ---------------------------------------------------------------------------
# Static file serving (frontend)
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
