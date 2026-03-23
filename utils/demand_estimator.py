"""Demand Estimator — pre-run calibration tool for MATSim experiments.

Reads a config file, queries the DuckDB database for population and survey data,
fetches Census ACS commute data, and estimates whether the configured demand
parameters will produce realistic trip volumes.
Generates a calibrated config file with adjusted parameters.

All benchmark values (trips/capita, avg legs per chain, travel-day participation
rate) are computed from real data in the database — no hardcoded constants.
This ensures the estimator works correctly for any region defined by the counties
in config.json.

Usage:
    python utils/demand_estimator.py config/USA/TwinCities/config_twin.json
    python utils/demand_estimator.py config/USA/TwinCities/config_twin.json --experiment-dir experiments/exp_20260301

Output:
    config/USA/TwinCities/config_twin_estimated.json
"""
import argparse
import copy
import json
import sys
import requests
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Add project root to path so we can import project modules
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TeeWriter:
    """Write to both stdout and a log file simultaneously."""

    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message: str) -> int:
        self.terminal.write(message)
        self.log_file.write(message)
        return len(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()


# ---------------------------------------------------------------------------
# Census ACS API
# ---------------------------------------------------------------------------
ACS_BASE_URL = "https://api.census.gov/data/{year}/acs/acs5"
ACS_YEAR = 2022  # latest 5-year ACS

# B08301: Means of Transportation to Work
#   _001E  Total workers 16+
#   _002E  Car, truck, or van (total)
#   _003E    Drove alone
#   _004E    Carpooled
#   _010E  Public transportation (total)
#   _011E    Bus or trolleybus
#   _012E    Subway or elevated rail
#   _013E    Long-distance train or commuter rail
#   _014E    Light rail, streetcar, or trolley
#   _015E    Ferryboat
#   _018E  Bicycle
#   _019E  Walked
#   _021E  Worked from home
B08301_VARIABLES = [
    "B08301_001E",  # total workers
    "B08301_002E",  # car/truck/van
    "B08301_003E",  # drove alone
    "B08301_004E",  # carpooled
    "B08301_010E",  # public transit (total)
    "B08301_011E",  # bus or trolleybus
    "B08301_012E",  # subway or elevated rail
    "B08301_013E",  # commuter rail
    "B08301_014E",  # light rail / streetcar
    "B08301_015E",  # ferryboat
    "B08301_018E",  # bicycle
    "B08301_019E",  # walked
    "B08301_021E",  # work from home
]

B08301_LABELS = {
    "B08301_001E": "total_workers",
    "B08301_002E": "car_truck_van",
    "B08301_003E": "drove_alone",
    "B08301_004E": "carpooled",
    "B08301_010E": "public_transit",
    "B08301_011E": "bus",
    "B08301_012E": "subway",
    "B08301_013E": "commuter_rail",
    "B08301_014E": "light_rail",
    "B08301_015E": "ferryboat",
    "B08301_018E": "bicycle",
    "B08301_019E": "walked",
    "B08301_021E": "work_from_home",
}


def fetch_acs_commute_data(
    county_fips_list: List[str],
    api_key: str,
    acs_year: int = ACS_YEAR,
) -> Dict[str, Dict[str, int]]:
    """Fetch ACS B08301 (commute mode) data per county via Census API.

    Args:
        county_fips_list: List of 5-digit FIPS codes (e.g. ["27003", "55093"]).
        api_key: Census API key.
        acs_year: ACS 5-year dataset year (default 2022).

    Returns:
        {county_fips: {label: value, ...}, ...}
    """
    # Group counties by state
    by_state: Dict[str, List[str]] = {}
    for fips in county_fips_list:
        state = fips[:2]
        county = fips[2:]
        by_state.setdefault(state, []).append(county)

    results: Dict[str, Dict[str, int]] = {}
    var_str = ",".join(B08301_VARIABLES)

    for state_fips, county_codes in by_state.items():
        county_str = ",".join(county_codes)
        url = ACS_BASE_URL.format(year=acs_year)
        params = {
            "get": var_str,
            "for": f"county:{county_str}",
            "in": f"state:{state_fips}",
        }
        if api_key:
            params["key"] = api_key

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            # Census returns HTML (not JSON) for invalid/unactivated keys
            content_type = resp.headers.get("content-type", "")
            if "json" not in content_type:
                if api_key:
                    print(f"  WARNING: Census API key may be invalid or not yet activated.")
                    print(f"  Check your email for the activation link from api.census.gov")
                    print(f"  Retrying without key...")
                    params.pop("key", None)
                    resp = requests.get(url, params=params, timeout=30)
                    resp.raise_for_status()
                    if "json" not in resp.headers.get("content-type", ""):
                        print(f"  WARNING: Census API still returning non-JSON for state {state_fips}")
                        continue
                else:
                    print(f"  WARNING: Census API returned non-JSON for state {state_fips}")
                    continue
            data = resp.json()
        except requests.RequestException as e:
            print(f"  WARNING: Census API request failed for state {state_fips}: {e}")
            continue
        except json.JSONDecodeError:
            print(f"  WARNING: Census API returned invalid JSON for state {state_fips}")
            continue

        if len(data) < 2:
            continue

        headers = data[0]
        for row in data[1:]:
            row_dict = dict(zip(headers, row))
            county_fips = state_fips + row_dict.get("county", "")
            entry = {}
            for var in B08301_VARIABLES:
                label = B08301_LABELS[var]
                val = row_dict.get(var)
                entry[label] = int(val) if val and val not in ("-", "null", "N") else 0
            results[county_fips] = entry

    return results


# ---------------------------------------------------------------------------
# Compute real values from database
# ---------------------------------------------------------------------------

def compute_population_from_db(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get population from the home_locations table in DuckDB.

    Calls ensure_home_locations() to auto-download LODES+Census if not in DB,
    then queries home_locations filtered by config counties.

    Returns:
        {total_population, total_employees, total_non_employees, source}
    """
    from models.home_locs_v2 import ensure_home_locations, load_home_locations_by_counties

    # Resolve data_dir to absolute path
    data_dir = config.get("data", {}).get("data_dir", "")
    if data_dir and not Path(data_dir).is_absolute():
        # Try relative to config file location or current dir
        config_dir = config.get("_config_dir", ".")
        resolved = (Path(config_dir) / data_dir).resolve()
        if resolved.exists():
            config = copy.deepcopy(config)
            config["data"]["data_dir"] = str(resolved)

    print("  [Source: LODES + Census via DuckDB home_locations table]")
    print("  Loading population data from database...")
    try:
        ensure_home_locations(config)
    except Exception as e:
        print(f"  ERROR: Failed to ensure home locations in database: {e}")
        print(f"  Check that data_dir in config points to a valid location with LODES data,")
        print(f"  or run a full experiment first to populate the database.")
        sys.exit(1)

    home_locs_dict = load_home_locations_by_counties(config)

    if not home_locs_dict:
        print("  ERROR: No home locations found in database for configured counties.")
        sys.exit(1)

    total_employees = sum(d.get('n_employees', 0) for d in home_locs_dict.values())
    total_non_employees = sum(d.get('non_employees', 0) for d in home_locs_dict.values())
    total_pop = total_employees + total_non_employees

    counties = config.get("region", {}).get("counties", [])
    return {
        "total_population": total_pop,
        "total_employees": total_employees,
        "total_non_employees": total_non_employees,
        "source": f"LODES/Census DB ({len(counties)} counties)",
    }


def compute_survey_benchmarks(
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, float], float]:
    """Compute trips-per-capita, avg-legs-per-chain, and travel-day participation
    rate from survey data in the DB.

    Loads survey data once from the survey_trips table, then computes all
    metrics from that single load.

    Returns:
        Tuple of (tpc_data, avg_legs, travel_day_rate) where:
        - tpc_data: {source_name: float, ..., 'blended': float}
        - avg_legs: {'work': float, 'nonwork': float}
        - travel_day_rate: blended share of person-days with at least one trip (0.0-1.0)
    """
    from data_sources.survey_manager import SurveyManager
    from data_sources.base_survey_trip import BaseSurveyTrip
    from models.chains import process_trip_chains, is_home_work_home_chain, is_home_other_home_chain
    import pandas as pd

    survey_manager = SurveyManager(config)
    all_data = survey_manager.load_data()  # loads from DB, sets self.data on each source
    blend_weights = survey_manager.get_blend_weights()

    # ----- Trips per capita -----
    print("  [Source: survey_trips table in DuckDB — TBI + NHTS surveys]")
    print("  Computing trips-per-capita from survey data...")

    tpc_result = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for source_name, df in all_data.items():
        if df.empty:
            print(f"    {source_name}: no data in database — skipping")
            continue

        person_col = BaseSurveyTrip.PERSON_ID

        unique_persons = df[person_col].nunique()
        if unique_persons == 0:
            continue

        # Trips per person for ONE simulated day.
        # We use unique_persons as the denominator (1 day per person) because
        # the simulation generates plans for a single day. This gives us
        # trips among trip-makers only; we'll adjust for non-travelers below
        # using the travel-day participation rate.
        tpc = len(df) / unique_persons

        tpc_result[source_name] = round(tpc, 2)
        source_weight = blend_weights.get(source_name, 1.0)
        weighted_sum += tpc * source_weight
        total_weight += source_weight

        print(f"    {source_name}: {tpc:.2f} trips/person (trip-makers only) "
              f"({len(df):,} trips, {unique_persons:,} persons, "
              f"weight={source_weight})")

    if total_weight > 0:
        tpc_result["blended"] = round(weighted_sum / total_weight, 2)
    else:
        print("  ERROR: No survey data found in database.")
        sys.exit(1)

    print(f"  => Blended trips/person (trip-makers): {tpc_result['blended']:.2f}")

    # ----- Travel-day participation rate -----
    # Try to compute from survey sources that track non-travelers.
    # get_persons() returns {person_id: {date: trips_df, ...}}.
    # For each person-day, if trips_df is empty, that's a non-travel day.
    print("  Computing travel-day participation rate from survey data...")

    persons = survey_manager.get_persons()

    if not persons:
        print("  ERROR: No person data found in database for chain analysis.")
        sys.exit(1)

    # Count person-days with and without trips across all sources
    total_person_days_all = 0
    travel_person_days_all = 0
    for person_id, days_dict in persons.items():
        for date_key, trips_df in days_dict.items():
            total_person_days_all += 1
            if len(trips_df) > 0:
                travel_person_days_all += 1

    if total_person_days_all > 0:
        computed_travel_rate = travel_person_days_all / total_person_days_all
    else:
        computed_travel_rate = 0.85  # fallback only if no data at all
        print("  WARNING: Could not compute travel-day rate from survey data, using 0.85 fallback")

    print(f"    {travel_person_days_all:,} travel person-days / "
          f"{total_person_days_all:,} total person-days")
    print(f"  => Travel-day participation rate: {computed_travel_rate:.2%}  "
          f"[config param: nonwork_purposes.nonwork_trip_share]")

    # Adjust trips/person to account for non-travelers.
    # tpc so far is among trip-makers only. Multiply by travel-day rate to get
    # the population-wide trips/capita/day (including people who made 0 trips).
    unadjusted_blended = tpc_result["blended"]
    tpc_result["blended"] = round(unadjusted_blended * computed_travel_rate, 2)
    print(f"  => Adjusted trips/capita/day: {unadjusted_blended:.2f} × "
          f"{computed_travel_rate:.2%} = {tpc_result['blended']:.2f}  "
          f"[accounts for non-travelers]")

    # ----- Avg legs per chain -----
    print("  Computing avg legs per chain from survey trip chains...")

    use_weight = config.get('chains', {}).get('use_weighted_chains', True)
    chains = process_trip_chains(persons, use_weight=use_weight)

    if not chains:
        print("  ERROR: No trip chains could be extracted from survey data.")
        sys.exit(1)

    chains_df = pd.DataFrame(chains)

    # Compute weighted average legs for each chain type
    def _weighted_avg_legs(chains_df, filter_func):
        """Compute weighted average legs for chains matching filter_func."""
        mask = chains_df['pattern'].apply(filter_func)
        filtered = chains_df[mask]
        if filtered.empty:
            return None

        # Each pattern's legs = number of activities - 1
        legs = filtered['pattern'].apply(lambda p: len(p.split('-')) - 1)

        if 'probability' in filtered.columns:
            weights = filtered['probability']
            return (legs * weights).sum() / weights.sum()
        elif 'count' in filtered.columns:
            weights = filtered['count']
            return (legs * weights).sum() / weights.sum()
        else:
            return legs.mean()

    avg_work = _weighted_avg_legs(chains_df, is_home_work_home_chain)
    avg_nonwork = _weighted_avg_legs(chains_df, is_home_other_home_chain)

    if avg_work is None:
        print("  WARNING: No Home->...->Work->...->Home chains found in survey data.")
        print("  Cannot compute avg_legs_work_chain.")
        sys.exit(1)

    if avg_nonwork is None:
        print("  WARNING: No Home->...->Home (nonwork) chains found in survey data.")
        print("  Cannot compute avg_legs_nonwork_chain.")
        sys.exit(1)

    avg_legs_result = {
        'work': round(avg_work, 2),
        'nonwork': round(avg_nonwork, 2),
    }

    print(f"    Work chains (Home->...->Work->...->Home):   {avg_legs_result['work']:.2f} avg legs")
    print(f"    Nonwork chains (Home->...->Home, no Work): {avg_legs_result['nonwork']:.2f} avg legs")

    return tpc_result, avg_legs_result, round(computed_travel_rate, 3)


# ---------------------------------------------------------------------------
# Demand estimation logic (mirrors plan_generator formulas)
# ---------------------------------------------------------------------------

def estimate_demand(
    config: Dict[str, Any],
    population_stats: Dict[str, Any],
    avg_legs: Dict[str, float],
    survey_trips_per_capita: float,
) -> Dict[str, Any]:
    """Estimate total demand from config parameters without running the sim.

    Mirrors the formulas in plan_generator.py to predict how many plans and
    trips the current config will produce.

    Args:
        config: Configuration dictionary.
        population_stats: From compute_population_from_db().
        avg_legs: From compute_survey_benchmarks(), e.g. {'work': 2.7, 'nonwork': 2.0}.
        survey_trips_per_capita: Blended trips/capita from surveys.

    Returns a dict with all estimation metrics.
    """
    plan_gen = config.get("plan_generation", {})
    scaling_factor = plan_gen.get("scaling_factor", 0.1)
    work_scaling_multiplier = plan_gen.get("work_scaling_multiplier", 1.0)

    avg_legs_work = avg_legs['work']
    avg_legs_nonwork = avg_legs['nonwork']

    total_pop = population_stats["total_population"]
    employees = population_stats["total_employees"]
    non_employees = population_stats["total_non_employees"]
    pop_source = population_stats["source"]

    # --- MATSim capacity factors ---
    matsim_params = config.get("matsim", {}).get("configurable_params", {})
    flow_cap = matsim_params.get("qsim.flowCapacityFactor", scaling_factor)
    storage_cap = matsim_params.get("qsim.storageCapacityFactor", scaling_factor)

    # --- Work trips ---
    # Formula: work_plans = employees * scaling_factor * work_scaling_multiplier
    effective_work_scaling = scaling_factor * work_scaling_multiplier
    work_plans_scaled = employees * effective_work_scaling
    work_plans_unscaled = employees  # 1 plan per worker

    # --- Nonwork trips ---
    # Formula: nonwork_plans = non_employees * sum(purpose_rates) * nonwork_trip_share
    nonwork_trip_share = config.get("nonwork_purposes", {}).get("nonwork_trip_share", 1.0)
    nonwork_purposes = _get_nonwork_purposes(config)

    nonwork_plans_unscaled = 0
    purpose_details = {}
    for purpose, info in nonwork_purposes.items():
        survey_rate = info["survey_rate"]
        config_rate = info["config_rate"]
        blend_weight = info["blend_weight"]

        # Blend formula from od_matrix_nonwork.py
        if survey_rate == "auto":
            # Can't compute survey rate without survey data; use config_rate
            final_rate = config_rate
            survey_rate_used = None
        else:
            final_rate = (1 - blend_weight) * float(survey_rate) + blend_weight * config_rate
            survey_rate_used = float(survey_rate)

        purpose_trips = non_employees * final_rate * nonwork_trip_share
        nonwork_plans_unscaled += purpose_trips

        purpose_details[purpose] = {
            "survey_rate": survey_rate_used,
            "config_rate": config_rate,
            "blend_weight": blend_weight,
            "final_rate": final_rate,
            "unscaled_trips": purpose_trips,
            "scaled_trips": purpose_trips * scaling_factor,
        }

    nonwork_plans_scaled = nonwork_plans_unscaled * scaling_factor

    # --- Total plans and trips ---
    total_plans_scaled = work_plans_scaled + nonwork_plans_scaled
    total_plans_unscaled = work_plans_unscaled + nonwork_plans_unscaled

    # Plans -> trips (each plan has multiple legs)
    work_trips_scaled = work_plans_scaled * avg_legs_work
    nonwork_trips_scaled = nonwork_plans_scaled * avg_legs_nonwork
    total_trips_scaled = work_trips_scaled + nonwork_trips_scaled

    total_trips_unscaled = (work_plans_unscaled * avg_legs_work +
                            nonwork_plans_unscaled * avg_legs_nonwork)

    trips_per_capita = total_trips_unscaled / total_pop if total_pop > 0 else 0

    # --- Effective network demand ---
    # MATSim scales link volumes back up by 1/scaling_factor for comparison
    # with ground truth. The ratio flow_cap / scaling_factor determines how
    # "generous" the network is relative to the agent sample:
    #   ratio > 1  =>  network can handle MORE than the agent proportion
    #                   (less congestion, but agents under-fill links)
    #   ratio == 1 =>  balanced
    #   ratio < 1  =>  network is tighter than agent proportion (more gridlock)
    cap_ratio = flow_cap / scaling_factor if scaling_factor > 0 else 1.0

    return {
        "population": {
            "total": total_pop,
            "employees": employees,
            "non_employees": non_employees,
            "source": pop_source,
        },
        "scaling": {
            "scaling_factor": scaling_factor,
            "work_scaling_multiplier": work_scaling_multiplier,
        },
        "capacity_factors": {
            "flow_capacity_factor": flow_cap,
            "storage_capacity_factor": storage_cap,
            "cap_to_scale_ratio": round(cap_ratio, 3),
        },
        "work": {
            "plans_unscaled": work_plans_unscaled,
            "plans_scaled": work_plans_scaled,
            "trips_scaled": work_trips_scaled,
            "avg_legs": avg_legs_work,
        },
        "nonwork": {
            "trip_share": nonwork_trip_share,
            "plans_unscaled": nonwork_plans_unscaled,
            "plans_scaled": nonwork_plans_scaled,
            "trips_scaled": nonwork_trips_scaled,
            "avg_legs": avg_legs_nonwork,
            "purposes": purpose_details,
        },
        "totals": {
            "plans_scaled": total_plans_scaled,
            "plans_unscaled": total_plans_unscaled,
            "trips_scaled": total_trips_scaled,
            "trips_unscaled": total_trips_unscaled,
            "trips_per_capita": trips_per_capita,
        },
        "benchmarks": {
            "survey_trips_per_capita": survey_trips_per_capita,
            "target_low": survey_trips_per_capita - 0.5,
            "target_high": survey_trips_per_capita + 0.5,
            "avg_legs_work": avg_legs_work,
            "avg_legs_nonwork": avg_legs_nonwork,
        },
    }


def _get_nonwork_purposes(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract nonwork purpose configs."""
    nonwork = config.get("nonwork_purposes", {})
    purposes = {}
    skip_keys = {"nonwork_trip_share", "_nonwork_trip_share_help"}

    for key, val in nonwork.items():
        if key.startswith("_") or key in skip_keys or not isinstance(val, dict):
            continue
        if not val.get("enabled", True):
            continue
        trip_gen = val.get("trip_generation", {})
        purposes[key] = {
            "survey_rate": trip_gen.get("survey_rate", "auto"),
            "config_rate": trip_gen.get("config_rate", 0.15),
            "blend_weight": trip_gen.get("blend_weight", 0.5),
        }

    return purposes


# ---------------------------------------------------------------------------
# Transit calibration from ACS bus/rail data
# ---------------------------------------------------------------------------

def compute_transit_calibration(
    acs_data: Dict[str, Dict[str, int]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute transit mode parameter recommendations from ACS county-level data.

    Uses ACS B08301 bus/rail breakdown to calculate region-specific config_rate,
    blend_weight, access_buffer_meters, and scaling_factor adjustments for the
    bus and rail modes in the config.

    The calibration logic follows the approach described in the evaluation:
    - Compute regional weighted-average bus and rail commute shares from ACS
    - Set config_rate above the target output share because the chain consistency
      constraint (all legs must have the mode available) removes 40-70% of
      transit-eligible chains
    - Set blend_weight based on how far the regional transit share is from the
      NHTS national average (~3-5% total transit)
    - Set access_buffer_meters based on transit density (higher share = denser
      network = larger reasonable walking catchment)
    - Adjust scaling_factor to compensate for agents shifting from car to transit

    Args:
        acs_data: Per-county ACS data from fetch_acs_commute_data().
        config: Current config dictionary.

    Returns:
        Dict with keys:
          - acs_regional: regional aggregated ACS shares
          - bus: recommended {config_rate, blend_weight, access_buffer_meters}
          - rail: recommended {config_rate, blend_weight, access_buffer_meters}
          - scaling_factor: recommended scaling_factor adjustment
          - recommendations: list of {parameter, current, recommended, reason}
    """
    if not acs_data:
        return {"recommendations": []}

    # --- Aggregate ACS data across all counties ---
    total_workers = sum(d.get("total_workers", 0) for d in acs_data.values())
    if total_workers == 0:
        return {"recommendations": []}

    total_transit = sum(d.get("public_transit", 0) for d in acs_data.values())
    total_bus = sum(d.get("bus", 0) for d in acs_data.values())
    total_subway = sum(d.get("subway", 0) for d in acs_data.values())
    total_commuter_rail = sum(d.get("commuter_rail", 0) for d in acs_data.values())
    total_light_rail = sum(d.get("light_rail", 0) for d in acs_data.values())
    total_ferry = sum(d.get("ferryboat", 0) for d in acs_data.values())

    # Combine rail modes: subway + commuter rail + light rail
    total_rail = total_subway + total_commuter_rail + total_light_rail

    transit_share = total_transit / total_workers
    bus_share = total_bus / total_workers
    rail_share = total_rail / total_workers

    acs_regional = {
        "total_workers": total_workers,
        "transit_share": round(transit_share, 4),
        "bus_share": round(bus_share, 4),
        "rail_share": round(rail_share, 4),
        "bus_count": total_bus,
        "rail_count": total_rail,
        "ferry_count": total_ferry,
    }

    # --- Per-county breakdown for diagnostics ---
    county_details = {}
    for fips, d in acs_data.items():
        cw = d.get("total_workers", 0)
        if cw > 0:
            county_details[fips] = {
                "workers": cw,
                "transit_share": round(d.get("public_transit", 0) / cw, 4),
                "bus_share": round(d.get("bus", 0) / cw, 4),
                "rail_share": round(
                    (d.get("subway", 0) + d.get("commuter_rail", 0) + d.get("light_rail", 0)) / cw, 4
                ),
            }

    # --- Current config values ---
    modes = config.get("modes", {})
    bus_cfg = modes.get("bus", {})
    rail_cfg = modes.get("rail", {})
    plan_gen = config.get("plan_generation", {})

    cur_bus_rate = bus_cfg.get("config_rate") or 0.0
    cur_bus_bw = bus_cfg.get("blend_weight", 0.5)
    cur_bus_buffer = bus_cfg.get("availability", {}).get("access_buffer_meters", 800) if isinstance(bus_cfg.get("availability"), dict) else 800
    cur_rail_rate = rail_cfg.get("config_rate") or 0.0
    cur_rail_bw = rail_cfg.get("blend_weight", 0.5)
    cur_rail_buffer = rail_cfg.get("availability", {}).get("access_buffer_meters", 1200) if isinstance(rail_cfg.get("availability"), dict) else 1200
    cur_scaling = plan_gen.get("scaling_factor", 0.1)

    # --- NHTS national baselines (approximate) ---
    nhts_bus_share = 0.02
    nhts_rail_share = 0.01

    # --- Compute recommended parameters ---
    # The chain consistency intersection filter removes a fraction of
    # transit-eligible chains. The reduction depends on transit coverage
    # density: denser networks lose fewer chains.
    #   - Very high transit (>20% share): ~40-50% reduction
    #   - High transit (10-20%): ~50-60% reduction
    #   - Medium transit (5-10%): ~55-65% reduction
    #   - Low transit (<5%): ~60-70% reduction
    if transit_share > 0.20:
        chain_reduction = 0.45
    elif transit_share > 0.10:
        chain_reduction = 0.55
    elif transit_share > 0.05:
        chain_reduction = 0.60
    else:
        chain_reduction = 0.65

    # ACS is commute-only; all-trip-types transit share is typically lower.
    # For regions with high commute transit share, all-trip share is roughly
    # 50-70% of the commute share (non-work trips use transit less).
    if transit_share > 0.20:
        all_trip_factor = 0.55
    elif transit_share > 0.10:
        all_trip_factor = 0.65
    else:
        all_trip_factor = 0.80  # low-transit regions, gap is smaller

    target_output_bus = bus_share * all_trip_factor
    target_output_rail = rail_share * all_trip_factor

    # Pre-filter rate = target output / (1 - chain_reduction)
    # This is the rate we need BEFORE the chain consistency filter removes chains
    prefilt_bus = target_output_bus / (1 - chain_reduction) if chain_reduction < 1 else target_output_bus
    prefilt_rail = target_output_rail / (1 - chain_reduction) if chain_reduction < 1 else target_output_rail

    # blend_weight: how much to lean on config_rate vs NHTS survey rate
    # If regional transit is close to NHTS national (~3-5%), use lower blend_weight
    # If regional transit is far from NHTS, lean heavily on config_rate
    transit_divergence = abs(transit_share - 0.04)  # distance from NHTS national avg
    if transit_divergence < 0.02:
        rec_blend_weight = 0.5  # close to national, moderate blending
    elif transit_divergence < 0.05:
        rec_blend_weight = 0.6
    else:
        rec_blend_weight = 0.7  # far from national, lean on config_rate

    # Back-calculate config_rate from the blend formula:
    # prefilt = (1 - bw) * nhts_rate + bw * config_rate
    # config_rate = (prefilt - (1 - bw) * nhts_rate) / bw
    if rec_blend_weight > 0:
        rec_bus_config_rate = (prefilt_bus - (1 - rec_blend_weight) * nhts_bus_share) / rec_blend_weight
        rec_rail_config_rate = (prefilt_rail - (1 - rec_blend_weight) * nhts_rail_share) / rec_blend_weight
    else:
        rec_bus_config_rate = prefilt_bus
        rec_rail_config_rate = prefilt_rail

    # Clamp to reasonable range [0.01, 0.40]
    rec_bus_config_rate = round(max(0.01, min(rec_bus_config_rate, 0.40)), 2)
    rec_rail_config_rate = round(max(0.01, min(rec_rail_config_rate, 0.40)), 2)

    # access_buffer_meters: based on transit network density
    # Denser networks (higher share) justify larger walking catchments
    if bus_share > 0.10:
        rec_bus_buffer = 1200
    elif bus_share > 0.05:
        rec_bus_buffer = 1000
    elif bus_share > 0.02:
        rec_bus_buffer = 900
    else:
        rec_bus_buffer = 800  # default

    if rail_share > 0.10:
        rec_rail_buffer = 1500
    elif rail_share > 0.03:
        rec_rail_buffer = 1500
    else:
        rec_rail_buffer = 1200  # default

    # scaling_factor adjustment: compensate for agents shifting car -> transit
    # More transit = fewer car agents, so increase scaling to maintain road volumes
    expected_transit_output = target_output_bus + target_output_rail
    # car fraction after transit shift: (1 - expected_transit_output)
    # to maintain same car volume: new_sf * (1 - transit_out) ≈ old_sf * ~0.99
    if expected_transit_output > 0.01:
        rec_scaling = round(cur_scaling * 0.99 / (1 - expected_transit_output), 3)
    else:
        rec_scaling = cur_scaling

    # --- Build recommendations list ---
    recommendations = []

    # Bus config_rate
    if abs(rec_bus_config_rate - cur_bus_rate) > 0.005:
        recommendations.append({
            "parameter": "modes.bus.config_rate",
            "current": cur_bus_rate,
            "recommended": rec_bus_config_rate,
            "reason": f"ACS regional bus commute share is {bus_share:.1%}. "
                      f"Target all-trip output ~{target_output_bus:.1%} after "
                      f"~{chain_reduction:.0%} chain consistency reduction. "
                      f"Pre-filter rate ~{prefilt_bus:.1%} with blend_weight={rec_blend_weight}.",
        })

    # Bus blend_weight
    if abs(rec_blend_weight - cur_bus_bw) > 0.05:
        recommendations.append({
            "parameter": "modes.bus.blend_weight",
            "current": cur_bus_bw,
            "recommended": rec_blend_weight,
            "reason": f"Regional transit share ({transit_share:.1%}) "
                      f"{'differs significantly from' if transit_divergence > 0.03 else 'is close to'} "
                      f"NHTS national average (~4%).",
        })

    # Bus access_buffer_meters
    if rec_bus_buffer != cur_bus_buffer:
        recommendations.append({
            "parameter": "modes.bus.availability.access_buffer_meters",
            "current": cur_bus_buffer,
            "recommended": rec_bus_buffer,
            "reason": f"Bus commute share {bus_share:.1%} suggests "
                      f"{'dense' if bus_share > 0.05 else 'moderate' if bus_share > 0.02 else 'sparse'} "
                      f"bus network coverage.",
        })

    # Rail config_rate
    if abs(rec_rail_config_rate - cur_rail_rate) > 0.005:
        recommendations.append({
            "parameter": "modes.rail.config_rate",
            "current": cur_rail_rate,
            "recommended": rec_rail_config_rate,
            "reason": f"ACS regional rail commute share is {rail_share:.1%} "
                      f"(subway={total_subway/total_workers:.1%}, "
                      f"commuter={total_commuter_rail/total_workers:.1%}, "
                      f"light rail={total_light_rail/total_workers:.1%}). "
                      f"Target all-trip output ~{target_output_rail:.1%} after "
                      f"~{chain_reduction:.0%} chain consistency reduction.",
        })

    # Rail blend_weight
    if abs(rec_blend_weight - cur_rail_bw) > 0.05:
        recommendations.append({
            "parameter": "modes.rail.blend_weight",
            "current": cur_rail_bw,
            "recommended": rec_blend_weight,
            "reason": f"Regional transit share ({transit_share:.1%}) "
                      f"{'differs significantly from' if transit_divergence > 0.03 else 'is close to'} "
                      f"NHTS national average (~4%).",
        })

    # Rail access_buffer_meters
    if rec_rail_buffer != cur_rail_buffer:
        recommendations.append({
            "parameter": "modes.rail.availability.access_buffer_meters",
            "current": cur_rail_buffer,
            "recommended": rec_rail_buffer,
            "reason": f"Rail commute share {rail_share:.1%} suggests "
                      f"{'dense' if rail_share > 0.05 else 'moderate'} rail station spacing.",
        })

    # Scaling factor
    if rec_scaling > cur_scaling + 0.002:
        recommendations.append({
            "parameter": "plan_generation.scaling_factor",
            "current": cur_scaling,
            "recommended": rec_scaling,
            "reason": f"Compensate for ~{expected_transit_output:.1%} of agents shifting from car to transit. "
                      f"Maintains comparable road network volumes: "
                      f"{rec_scaling} x {1 - expected_transit_output:.2f} ≈ {cur_scaling} x 0.99.",
        })

    return {
        "acs_regional": acs_regional,
        "county_details": county_details,
        "bus": {
            "config_rate": rec_bus_config_rate,
            "blend_weight": rec_blend_weight,
            "access_buffer_meters": rec_bus_buffer,
            "prefilt_rate": round(prefilt_bus, 4),
            "target_output": round(target_output_bus, 4),
        },
        "rail": {
            "config_rate": rec_rail_config_rate,
            "blend_weight": rec_blend_weight,
            "access_buffer_meters": rec_rail_buffer,
            "prefilt_rate": round(prefilt_rail, 4),
            "target_output": round(target_output_rail, 4),
        },
        "chain_reduction": chain_reduction,
        "all_trip_factor": all_trip_factor,
        "scaling_factor": rec_scaling,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# Scorecard & recommendations
# ---------------------------------------------------------------------------

def compute_scorecard(
    estimate: Dict[str, Any],
    acs_data: Dict[str, Dict[str, int]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute calibration scorecard comparing estimated vs benchmark values."""
    scorecard = {}

    benchmarks = estimate["benchmarks"]
    target_low = benchmarks["target_low"]
    target_high = benchmarks["target_high"]

    # 1. Trips per capita
    tpc = estimate["totals"]["trips_per_capita"]
    scorecard["trips_per_capita"] = {
        "current": round(tpc, 2),
        "target_low": round(target_low, 2),
        "target_high": round(target_high, 2),
        "survey_benchmark": benchmarks["survey_trips_per_capita"],
        "status": "OK" if target_low <= tpc <= target_high else
                  "LOW" if tpc < target_low else "HIGH",
    }

    # 2. Work trips vs ACS commuters
    if acs_data:
        acs_total_workers = sum(d.get("total_workers", 0) for d in acs_data.values())
        lodes_employees = estimate["population"]["employees"]
        ratio = lodes_employees / acs_total_workers if acs_total_workers > 0 else 0
        scorecard["work_trips_vs_acs"] = {
            "lodes_employees": lodes_employees,
            "acs_total_workers": acs_total_workers,
            "ratio": round(ratio, 3),
            "status": "OK" if 0.9 <= ratio <= 1.1 else "MISMATCH",
        }

        # 3. Mode share comparison
        acs_mode_share = {}
        if acs_total_workers > 0:
            acs_mode_share = {
                "car": round((sum(d.get("drove_alone", 0) + d.get("carpooled", 0) for d in acs_data.values())) / acs_total_workers, 3),
                "transit": round(sum(d.get("public_transit", 0) for d in acs_data.values()) / acs_total_workers, 3),
                "bus": round(sum(d.get("bus", 0) for d in acs_data.values()) / acs_total_workers, 3),
                "subway": round(sum(d.get("subway", 0) for d in acs_data.values()) / acs_total_workers, 3),
                "commuter_rail": round(sum(d.get("commuter_rail", 0) for d in acs_data.values()) / acs_total_workers, 3),
                "light_rail": round(sum(d.get("light_rail", 0) for d in acs_data.values()) / acs_total_workers, 3),
                "walk": round(sum(d.get("walked", 0) for d in acs_data.values()) / acs_total_workers, 3),
                "bike": round(sum(d.get("bicycle", 0) for d in acs_data.values()) / acs_total_workers, 3),
                "work_from_home": round(sum(d.get("work_from_home", 0) for d in acs_data.values()) / acs_total_workers, 3),
            }
        scorecard["acs_mode_share"] = acs_mode_share

    return scorecard


def load_experiment_feedback(config: Dict[str, Any]) -> Dict[str, Any] | None:
    """Look for a previous experiment's summary and evaluation results.

    Searches the experiments/ directory for the most recent experiment folder
    that has an experiment_summary.json file.

    Returns:
        Dict with experiment metrics, or None if no experiment found.
    """
    config_dir = config.get("_config_dir", ".")
    # Typical layout: config is in config/USA/TwinCities/, experiments at project root
    project_root = Path(config_dir)
    # Walk up until we find experiments/
    for _ in range(5):
        experiments_dir = project_root / "experiments"
        if experiments_dir.is_dir():
            break
        project_root = project_root.parent
    else:
        return None

    # Find all experiment folders, pick the most recent
    best_summary = None
    best_eval = None
    best_mtime = 0

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        summary_file = exp_dir / "experiment_summary.json"
        if summary_file.is_file():
            mtime = summary_file.stat().st_mtime
            if mtime > best_mtime:
                best_mtime = mtime
                best_summary = summary_file
                eval_file = exp_dir / "evaluation" / "summary_metrics.json"
                best_eval = eval_file if eval_file.is_file() else None

    if best_summary is None:
        return None

    try:
        with open(best_summary, "r") as f:
            summary = json.load(f)
        eval_data = None
        if best_eval:
            with open(best_eval, "r") as f:
                eval_data = json.load(f)
        return {
            "summary": summary,
            "evaluation": eval_data,
            "experiment_dir": str(best_summary.parent),
        }
    except (json.JSONDecodeError, OSError):
        return None


def recommend_adjustments(
    estimate: Dict[str, Any],
    scorecard: Dict[str, Any],
    survey_travel_day_rate: float,
    experiment_feedback: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Generate config adjustment recommendations to reach target trips/capita.

    Strategy (in order):
    0. Check capacity factor mismatch and experiment feedback FIRST
    1. Set nonwork_trip_share to survey-computed travel-day participation rate
    2. Back-calculate needed purpose rates to close the gap
    3. Set blend_weight to 0.7 so config_rate dominates
    4. If rates alone can't close the gap, suggest scaling_factor increase
    """
    recommendations = []

    benchmarks = estimate["benchmarks"]
    target = benchmarks["survey_trips_per_capita"]
    target_low = benchmarks["target_low"]
    avg_legs_work = benchmarks["avg_legs_work"]
    avg_legs_nonwork = benchmarks["avg_legs_nonwork"]

    tpc = estimate["totals"]["trips_per_capita"]

    total_pop = estimate["population"]["total"]
    employees = estimate["population"]["employees"]
    non_employees = estimate["population"]["non_employees"]

    cap_info = estimate.get("capacity_factors", {})
    flow_cap = cap_info.get("flow_capacity_factor", estimate["scaling"]["scaling_factor"])
    storage_cap = cap_info.get("storage_capacity_factor", estimate["scaling"]["scaling_factor"])
    scaling_factor = estimate["scaling"]["scaling_factor"]
    cap_ratio = cap_info.get("cap_to_scale_ratio", 1.0)

    # --- Check 0a: Capacity factor / scaling factor relationship ---
    # Capacity factors control road network throughput and are independent
    # calibration parameters. They do NOT have to match scaling_factor.
    # - flowCapacityFactor < scaling_factor → tighter network, more congestion,
    #   higher link volumes per agent (can help if sim under-estimates traffic)
    # - flowCapacityFactor = scaling_factor → textbook balanced (standard approach)
    # - flowCapacityFactor > scaling_factor → looser network, less congestion
    # Report as diagnostic info, not a hard recommendation.
    if abs(flow_cap - scaling_factor) > 0.01:
        if cap_ratio > 1:
            note = "over-provisioned — less congestion than real-world"
        else:
            note = "tighter than sample fraction — more congestion, higher volumes per agent"
        recommendations.append({
            "parameter": "_info.capacity_factor_mismatch",
            "current": f"flow={flow_cap}, storage={storage_cap}, scaling={scaling_factor}, ratio={cap_ratio:.2f}x",
            "recommended": f"Calibration parameter — adjust based on volume validation",
            "reason": f"Capacity factors != scaling_factor ({cap_ratio:.2f}x). "
                      f"Network is {note}. "
                      f"This is a calibration lever, not necessarily a problem. "
                      f"countsScaleFactor (auto-set to 1/scaling_factor) handles volume comparison.",
        })

    # --- Check 0b: Experiment feedback — actual MATSim output vs planned ---
    needs_more_demand = False
    demand_multiplier = 1.0

    if experiment_feedback:
        summary = experiment_feedback.get("summary", {})
        eval_data = experiment_feedback.get("evaluation")

        # Compare actual output vs what was generated
        actual_legs = summary.get("matsim_output", {}).get("output_legs_count", 0)
        actual_plans = summary.get("matsim_output", {}).get("output_persons_count", 0)
        generated_plans = summary.get("plans", {}).get("total", 0)
        estimated_plans_scaled = estimate["totals"]["plans_scaled"]
        estimated_legs_scaled = estimate["totals"]["trips_scaled"]  # estimator calls these "trips" but they are legs

        if actual_plans > 0 and estimated_plans_scaled > 0:
            # Plan generation gap: estimated vs actually generated
            plan_gen_ratio = generated_plans / estimated_plans_scaled if estimated_plans_scaled > 0 else 1.0

            # Legs comparison: use output_legs (not output_trips) against estimated legs
            actual_legs_per_agent = actual_legs / actual_plans if actual_plans > 0 else 0
            estimated_legs_per_agent = estimated_legs_scaled / estimated_plans_scaled if estimated_plans_scaled > 0 else 0

            if plan_gen_ratio < 0.9:
                recommendations.append({
                    "parameter": "_info.plan_generation_gap",
                    "current": f"{generated_plans:,} plans ({plan_gen_ratio:.1%} of estimated)",
                    "recommended": f"~{estimated_plans_scaled:,.0f} plans",
                    "reason": f"Plan generator produced {generated_plans:,} plans vs "
                              f"estimator prediction of {estimated_plans_scaled:,.0f} "
                              f"({1 - plan_gen_ratio:.1%} lost to rounding, OD sparsity, or failures).",
                })

            if actual_legs_per_agent < estimated_legs_per_agent * 0.8:
                recommendations.append({
                    "parameter": "_info.legs_per_agent_gap",
                    "current": f"{actual_legs_per_agent:.1f} legs/agent",
                    "recommended": f"{estimated_legs_per_agent:.1f} legs/agent (survey avg)",
                    "reason": f"Actual plans average {actual_legs_per_agent:.1f} legs/agent vs "
                              f"survey-based estimate of {estimated_legs_per_agent:.1f}. "
                              f"Plan generator produces shorter chains than survey averages "
                              f"(most plans are simple Home-Work-Home with 2 legs).",
                })

        # Check evaluation metrics — the real ground truth
        if eval_data:
            mean_pct_error = eval_data.get("mean_pct_error", 0)
            geh_lt_5 = eval_data.get("geh_lt_5_pct", 0)

            if mean_pct_error < -30:
                # Simulation significantly under-estimates real traffic
                # Back-calculate how much more demand we need:
                # If mean_pct_error = -92%, sim produces ~8% of real volume
                # We need roughly 1/(1 + mean_pct_error/100) times more demand
                actual_fraction = 1 + mean_pct_error / 100  # e.g. 0.08 for -92%
                if actual_fraction > 0:
                    demand_multiplier = 1.0 / actual_fraction
                else:
                    demand_multiplier = 10.0  # cap
                # Cap at a reasonable multiplier to avoid overshooting
                demand_multiplier = min(demand_multiplier, 10.0)
                needs_more_demand = True

                recommendations.append({
                    "parameter": "_info.experiment_under_demand",
                    "current": f"mean_pct_error={mean_pct_error:.1f}%, GEH<5={geh_lt_5:.1f}%",
                    "recommended": "mean_pct_error ~ 0%, GEH<5 > 85%",
                    "reason": f"Previous experiment shows simulated volumes are "
                              f"{abs(mean_pct_error):.0f}% below observed traffic. "
                              f"Demand needs ~{demand_multiplier:.1f}x increase to match counts. "
                              f"(GEH<5 target: >85%, current: {geh_lt_5:.1f}%)",
                })

    # If trips/capita looks OK but experiment shows massive under-demand,
    # the theoretical estimate is misleading — proceed with recommendations.
    # When experiment shows under-demand, the fix is primarily to increase
    # scaling_factor (more agents) and fix capacity factor mismatch, NOT to
    # inflate trips/capita beyond the survey benchmark.
    if tpc >= target_low and not needs_more_demand:
        return recommendations  # demand genuinely looks OK (and no experiment contradicts it)

    # --- Lever 1: nonwork_trip_share (from survey travel-day rate) ---
    current_share = estimate["nonwork"]["trip_share"]
    new_share = survey_travel_day_rate
    if current_share < new_share:
        recommendations.append({
            "parameter": "nonwork_purposes.nonwork_trip_share",
            "current": current_share,
            "recommended": round(new_share, 2),
            "reason": f"Survey data shows {new_share:.0%} of person-days have at least one trip "
                      f"(current: {current_share:.0%})",
        })

    effective_share = new_share if current_share < new_share else current_share

    # --- Lever 2: back-calculate needed purpose rates ---
    # Use the survey target as-is — do NOT inflate trips/capita beyond the
    # survey benchmark. If the experiment shows under-demand, the fix is
    # to increase scaling_factor (Lever 4), not to over-generate trips.
    target_trips = target * total_pop
    work_trips = employees * avg_legs_work
    needed_nonwork_trips = target_trips - work_trips
    needed_nonwork_plans = needed_nonwork_trips / avg_legs_nonwork

    needed_total_rate = (needed_nonwork_plans / (non_employees * effective_share)
                         if (non_employees * effective_share) > 0 else 1.0)

    purposes = estimate["nonwork"]["purposes"]
    current_rates = {p: info["config_rate"] for p, info in purposes.items()}
    current_total_rate = sum(current_rates.values())
    rate_multiplier = needed_total_rate / current_total_rate if current_total_rate > 0 else 1.0

    for purpose, info in purposes.items():
        current_rate = info["config_rate"]
        new_rate = round(min(current_rate * rate_multiplier, 0.50), 2)
        new_rate = max(new_rate, current_rate)
        if new_rate > current_rate:
            recommendations.append({
                "parameter": f"nonwork_purposes.{purpose}.trip_generation.config_rate",
                "current": current_rate,
                "recommended": new_rate,
                "reason": f"Back-calculated to reach ~{target:.1f} trips/capita "
                         f"(multiplier: {rate_multiplier:.2f}x, capped at 0.50)",
            })

    # --- Lever 3: blend_weight ---
    for purpose, info in purposes.items():
        current_bw = info["blend_weight"]
        if current_bw < 0.7:
            recommendations.append({
                "parameter": f"nonwork_purposes.{purpose}.trip_generation.blend_weight",
                "current": current_bw,
                "recommended": 0.7,
                "reason": "Lean more on calibrated config_rate vs raw survey rate",
            })

    # --- Lever 4: scaling_factor ---
    # If experiment shows under-demand, the primary fix is more agents.
    # The demand_multiplier from experiment tells us how much more volume
    # we need, but we apply it conservatively to scaling_factor.
    current_sf = estimate["scaling"]["scaling_factor"]

    if needs_more_demand:
        # Experiment-informed: use a conservative power of the demand multiplier
        # e.g. if we need ~13x more volume, try ~2.5x more agents (13^0.35 ≈ 2.5)
        # This is conservative because more agents + matched capacity factors
        # should have a compounding effect on link volumes.
        conservative_sf_mult = min(demand_multiplier ** 0.35, 2.5)
        new_sf = round(min(current_sf * conservative_sf_mult, 1.0), 2)
    else:
        # Purely trips/capita driven
        new_total_rate = sum(min(r * rate_multiplier, 0.50) for r in current_rates.values())
        projected_nonwork_plans = non_employees * new_total_rate * effective_share
        projected_nonwork_trips = projected_nonwork_plans * avg_legs_nonwork
        projected_total_trips = work_trips + projected_nonwork_trips
        projected_tpc = projected_total_trips / total_pop if total_pop > 0 else 0

        if projected_tpc < target_low:
            remaining_gap = target / projected_tpc if projected_tpc > 0 else 2.0
            new_sf = round(min(current_sf * remaining_gap, 1.0), 2)
        else:
            new_sf = current_sf  # no scaling change needed

    if new_sf > current_sf:
        recommendations.append({
            "parameter": "plan_generation.scaling_factor",
            "current": current_sf,
            "recommended": new_sf,
            "reason": f"{'Experiment shows significant under-demand. ' if needs_more_demand else ''}"
                     f"Increase scaling to generate more agents "
                     f"(from {current_sf:.0%} to {new_sf:.0%} of population). "
                     f"Capacity factors are NOT changed — they are independent "
                     f"calibration parameters. countsScaleFactor will auto-adjust "
                     f"to 1/{new_sf} = {1/new_sf:.1f}.",
        })

    return recommendations


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def apply_recommendations(
    config: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply recommendations to a copy of the config and return it."""
    new_config = copy.deepcopy(config)

    for rec in recommendations:
        path = rec["parameter"]
        value = rec["recommended"]
        reason = rec["reason"]

        # Skip diagnostic-only items (not real config params)
        if path.startswith("_info."):
            continue

        _set_nested(new_config, path, value)
        # Add an estimator note explaining the change
        note_path = path.rsplit(".", 1)
        if len(note_path) == 2:
            parent_path, key = note_path
            note_key = f"_estimator_{key}"
            _set_nested(new_config, f"{parent_path}.{note_key}", reason)
        else:
            _set_nested(new_config, f"_estimator_{path}", reason)

    return new_config


def _set_nested(d: Dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot-separated key path."""
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_scorecard(
    estimate: Dict[str, Any],
    scorecard: Dict[str, Any],
    acs_data: Dict[str, Dict[str, int]],
    recommendations: List[Dict[str, Any]],
    survey_travel_day_rate: float,
    experiment_feedback: Dict[str, Any] | None = None,
    transit_calibration: Dict[str, Any] | None = None,
) -> None:
    """Print a human-readable calibration scorecard."""
    benchmarks = estimate["benchmarks"]
    avg_legs_work = benchmarks["avg_legs_work"]
    avg_legs_nonwork = benchmarks["avg_legs_nonwork"]
    target_low = benchmarks["target_low"]
    target_high = benchmarks["target_high"]

    print()
    print("=" * 70)
    print("  DEMAND CALIBRATION SCORECARD")
    print("=" * 70)

    # --- Section 1: Benchmarks from real data ---
    print(f"\n  BENCHMARKS (all computed from survey data, not hardcoded)")
    print(f"  {'─' * 60}")
    print(f"    Trips/capita/day (survey):       {benchmarks['survey_trips_per_capita']:.2f}")
    print(f"      Source: blended TBI + NHTS survey trips / person-days")
    print(f"    Target range:                    {target_low:.2f} - {target_high:.2f}")
    print(f"      Source: survey trips/capita ± 0.5 tolerance")
    print(f"    Avg legs per work chain:         {avg_legs_work:.2f}")
    print(f"      Source: survey Home->Work->Home chains")
    print(f"    Avg legs per nonwork chain:      {avg_legs_nonwork:.2f}")
    print(f"      Source: survey Home->...->Home (no Work) chains")
    print(f"    Travel-day participation rate:   {survey_travel_day_rate:.2%}")
    print(f"      Source: share of survey person-days with >= 1 trip")

    # --- Section 2: Population ---
    pop = estimate["population"]
    print(f"\n  POPULATION ({pop['source']})")
    print(f"  {'─' * 60}")
    print(f"    Total population:   {pop['total']:>12,}")
    print(f"    Employees (LODES):  {pop['employees']:>12,}")
    print(f"    Non-employees:      {pop['non_employees']:>12,}")

    # --- Section 3: Config parameters used ---
    sf = estimate["scaling"]
    cap = estimate.get("capacity_factors", {})
    print(f"\n  CURRENT CONFIG PARAMETERS")
    print(f"  {'─' * 60}")
    print(f"    plan_generation.scaling_factor:          {sf['scaling_factor']}")
    print(f"    plan_generation.work_scaling_multiplier: {sf['work_scaling_multiplier']}")
    print(f"    nonwork_purposes.nonwork_trip_share:     {estimate['nonwork']['trip_share']:.2f}")

    # Show capacity factors (independent calibration parameters)
    flow_cap = cap.get("flow_capacity_factor", sf["scaling_factor"])
    storage_cap = cap.get("storage_capacity_factor", sf["scaling_factor"])
    cap_ratio = cap.get("cap_to_scale_ratio", 1.0)

    print(f"\n    MATSim Capacity Factors (calibration parameters):")
    print(f"    qsim.flowCapacityFactor:        {flow_cap}")
    print(f"    qsim.storageCapacityFactor:      {storage_cap}")
    print(f"    capacity/scaling ratio:           {cap_ratio:.2f}x")
    if cap_ratio > 1.05:
        print(f"      Network over-provisioned — less congestion than real-world.")
    elif cap_ratio < 0.95:
        print(f"      Network tighter than sample fraction — more congestion, higher volumes/agent.")
    else:
        print(f"      Balanced — capacity matches sample fraction.")

    # Nonwork purpose breakdown
    print(f"\n    Nonwork Purpose Rates:")
    print(f"    {'Purpose':<12} {'config_rate':>11} {'blend_wt':>9} {'final_rate':>11} {'unscaled plans':>15}")
    print(f"    {'-'*12} {'-'*11} {'-'*9} {'-'*11} {'-'*15}")
    for purpose, info in estimate["nonwork"]["purposes"].items():
        print(f"    {purpose:<12} {info['config_rate']:>11.2%} {info['blend_weight']:>9.2f} {info['final_rate']:>11.2%} {info['unscaled_trips']:>15,.0f}")

    # --- Section 4: Estimated demand ---
    print(f"\n  ESTIMATED DEMAND (what the config will produce)")
    print(f"  {'─' * 60}")
    print(f"    Work plans (1 per employee):        {estimate['work']['plans_unscaled']:>12,.0f}")
    print(f"      x {avg_legs_work:.1f} avg legs/chain =            {estimate['work']['plans_unscaled'] * avg_legs_work:>12,.0f} work trips")
    print(f"    Nonwork plans:                      {estimate['nonwork']['plans_unscaled']:>12,.0f}")
    print(f"      x {avg_legs_nonwork:.1f} avg legs/chain =            {estimate['nonwork']['plans_unscaled'] * avg_legs_nonwork:>12,.0f} nonwork trips")
    print(f"    {'─' * 48}")
    print(f"    Total trips (unscaled):             {estimate['totals']['trips_unscaled']:>12,.0f}")

    # Trips per capita verdict
    tpc_info = scorecard.get("trips_per_capita", {})
    tpc = tpc_info.get("current", 0)
    status = tpc_info.get("status", "?")
    status_marker = "OK" if status == "OK" else "!! " + status
    print(f"\n    => Trips/capita/day: {tpc:.2f}  [{status_marker}]")
    print(f"       Target:          {target_low:.2f} - {target_high:.2f}  (from survey)")
    if status != "OK":
        gap = benchmarks["survey_trips_per_capita"] - tpc
        print(f"       Gap:             {gap:+.2f} trips/capita/day")

    # --- Section 5: Scaled output (what MATSim will actually run) ---
    print(f"\n  SCALED OUTPUT (agents MATSim will simulate)")
    print(f"  {'─' * 60}")
    print(f"    scaling_factor = {sf['scaling_factor']}  "
          f"(simulates {sf['scaling_factor']:.0%} of full population)")
    print(f"    Scaled plans:  {estimate['totals']['plans_scaled']:>12,.0f}")
    print(f"    Scaled trips:  {estimate['totals']['trips_scaled']:>12,.0f}")

    # --- Section 6: ACS cross-check ---
    if "work_trips_vs_acs" in scorecard:
        wt = scorecard["work_trips_vs_acs"]
        print(f"\n  CROSS-CHECK: LODES vs Census ACS (Source: ACS B08301 API)")
        print(f"  {'─' * 60}")
        print(f"    LODES employees:    {wt['lodes_employees']:>10,}")
        print(f"    ACS total workers:  {wt['acs_total_workers']:>10,}")
        print(f"    Ratio:              {wt['ratio']:>10.3f}  [{wt['status']}]")
        print(f"      (1.0 = perfect match, 0.9-1.1 = acceptable)")

    if "acs_mode_share" in scorecard and scorecard["acs_mode_share"]:
        ms = scorecard["acs_mode_share"]
        print(f"\n  ACS Commute Mode Share (work trips only, Source: ACS B08301):")
        print(f"    Car (drove+carpool): {ms.get('car', 0):>6.1%}")
        print(f"    Public transit:      {ms.get('transit', 0):>6.1%}")
        print(f"      Bus:               {ms.get('bus', 0):>6.1%}")
        print(f"      Subway/metro:      {ms.get('subway', 0):>6.1%}")
        print(f"      Commuter rail:     {ms.get('commuter_rail', 0):>6.1%}")
        print(f"      Light rail:        {ms.get('light_rail', 0):>6.1%}")
        print(f"    Walk:                {ms.get('walk', 0):>6.1%}")
        print(f"    Bike:                {ms.get('bike', 0):>6.1%}")
        print(f"    Work from home:      {ms.get('work_from_home', 0):>6.1%}")

    # --- Section 6b: Transit calibration from ACS ---
    if transit_calibration and transit_calibration.get("acs_regional"):
        tc = transit_calibration
        acs_r = tc["acs_regional"]
        print(f"\n  TRANSIT CALIBRATION (from ACS bus/rail breakdown)")
        print(f"  {'─' * 60}")
        print(f"    ACS regional transit share:  {acs_r['transit_share']:>6.1%}")
        print(f"      Bus commute share:         {acs_r['bus_share']:>6.1%}")
        print(f"      Rail commute share:        {acs_r['rail_share']:>6.1%}")
        print(f"    All-trip adjustment factor:  {tc['all_trip_factor']:.2f}")
        print(f"    Chain consistency reduction: ~{tc['chain_reduction']:.0%}")

        # Bus calibration
        bus = tc["bus"]
        print(f"\n    Bus mode calibration:")
        print(f"      Target output share:       {bus['target_output']:>6.1%}")
        print(f"      Pre-filter rate needed:    {bus['prefilt_rate']:>6.1%}")
        print(f"      Recommended config_rate:   {bus['config_rate']:>6.1%}")
        print(f"      Recommended blend_weight:  {bus['blend_weight']:>6.2f}")
        print(f"      Recommended access buffer: {bus['access_buffer_meters']:>5d} m")

        # Rail calibration
        rail = tc["rail"]
        print(f"\n    Rail mode calibration:")
        print(f"      Target output share:       {rail['target_output']:>6.1%}")
        print(f"      Pre-filter rate needed:    {rail['prefilt_rate']:>6.1%}")
        print(f"      Recommended config_rate:   {rail['config_rate']:>6.1%}")
        print(f"      Recommended blend_weight:  {rail['blend_weight']:>6.2f}")
        print(f"      Recommended access buffer: {rail['access_buffer_meters']:>5d} m")

        # Per-county breakdown
        county_details = tc.get("county_details", {})
        if county_details:
            print(f"\n    Per-county transit shares (ACS B08301):")
            print(f"    {'FIPS':<8} {'Workers':>10} {'Transit':>8} {'Bus':>8} {'Rail':>8}")
            print(f"    {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
            for fips, cd in sorted(county_details.items(),
                                   key=lambda x: x[1]["workers"], reverse=True):
                print(f"    {fips:<8} {cd['workers']:>10,} {cd['transit_share']:>7.1%} "
                      f"{cd['bus_share']:>7.1%} {cd['rail_share']:>7.1%}")

        if tc.get("scaling_factor") and tc["scaling_factor"] > 0:
            print(f"\n    Recommended scaling_factor:   {tc['scaling_factor']}")

    # --- Section 7: Previous experiment feedback ---
    if experiment_feedback:
        summary = experiment_feedback.get("summary", {})
        eval_data = experiment_feedback.get("evaluation")
        exp_dir = experiment_feedback.get("experiment_dir", "")

        print(f"\n  PREVIOUS EXPERIMENT FEEDBACK")
        print(f"  {'─' * 60}")
        print(f"    Source: {Path(exp_dir).name}")

        matsim_out = summary.get("matsim_output", {})
        actual_plans = matsim_out.get("output_persons_count", 0)
        actual_trips = matsim_out.get("output_trips_count", 0)
        actual_legs = matsim_out.get("output_legs_count", 0)
        generated_plans = summary.get("plans", {}).get("total", 0)
        stuck = matsim_out.get("total_stuck_agents", 0)

        if actual_plans > 0:
            estimated_plans = estimate["totals"]["plans_scaled"]
            estimated_legs_per_agent = (estimate["totals"]["trips_scaled"] / estimated_plans
                                        if estimated_plans > 0 else 0)
            actual_legs_per_agent = actual_legs / actual_plans if actual_plans > 0 else 0

            print(f"    Estimated plans:      {estimated_plans:>12,.0f}")
            print(f"    Generated plans:      {generated_plans:>12,}")
            if estimated_plans > 0:
                plan_gen_pct = generated_plans / estimated_plans * 100
                print(f"      Plan generation:    {plan_gen_pct:>11.1f}% of estimate")
            print(f"    Agents simulated:     {actual_plans:>12,}")
            print(f"    Trips completed:      {actual_trips:>12,}")
            print(f"    Legs completed:       {actual_legs:>12,}")
            print(f"    Legs/agent (actual):    {actual_legs_per_agent:>10.1f}")
            print(f"    Legs/agent (survey est):{estimated_legs_per_agent:>10.1f}")
            if actual_legs_per_agent < estimated_legs_per_agent * 0.8:
                print(f"      NOTE: Actual chains are shorter than survey averages.")
                print(f"      Plan generator favors simple Home-Work-Home (2 legs).")
                print(f"      This is NOT MATSim dropping trips.")
            if stuck > 0:
                print(f"    Stuck agents:         {stuck:>12,}")

        if eval_data:
            mean_pct = eval_data.get("mean_pct_error", 0)
            geh_lt5 = eval_data.get("geh_lt_5_pct", 0)
            corr = eval_data.get("correlation", 0)
            mae = eval_data.get("mae", 0)

            print(f"\n    Ground Truth Comparison:")
            pct_status = "OK" if -30 < mean_pct < 30 else ("!! UNDER" if mean_pct < -30 else "!! OVER")
            geh_status = "OK" if geh_lt5 > 85 else "!! LOW"
            print(f"    Mean % error:          {mean_pct:>8.1f}%  [{pct_status}]")
            print(f"    GEH < 5:               {geh_lt5:>8.1f}%  [{geh_status}]  (target: >85%)")
            print(f"    Correlation:           {corr:>8.3f}")
            print(f"    MAE (veh/hr):          {mae:>8.0f}")

            if mean_pct < -30:
                actual_frac = 1 + mean_pct / 100
                if actual_frac > 0:
                    needed_mult = 1.0 / actual_frac
                    print(f"\n    Sim produces ~{actual_frac:.0%} of real traffic volumes.")
                    print(f"    Raw demand multiplier needed: ~{needed_mult:.1f}x")

    # --- Section 8: Recommendations ---
    if recommendations:
        # Separate info items from actionable recommendations
        info_recs = [r for r in recommendations if r["parameter"].startswith("_info.")]
        action_recs = [r for r in recommendations if not r["parameter"].startswith("_info.")]

        if info_recs:
            print(f"\n  {'=' * 66}")
            print(f"  DIAGNOSTIC FINDINGS")
            print(f"  {'=' * 66}")
            for i, rec in enumerate(info_recs, 1):
                label = rec["parameter"].replace("_info.", "").replace("_", " ").title()
                print(f"\n  {i}. {label}")
                print(f"     Current:  {rec['current']}")
                print(f"     Target:   {rec['recommended']}")
                print(f"     Details:  {rec['reason']}")

        if action_recs:
            print(f"\n  {'=' * 66}")
            print(f"  RECOMMENDATIONS (config changes to improve demand)")
            print(f"  {'=' * 66}")
            for i, rec in enumerate(action_recs, 1):
                print(f"\n  {i}. Config param: {rec['parameter']}")
                print(f"     Current value:     {rec['current']}")
                print(f"     Recommended value: {rec['recommended']}")
                print(f"     Reason:            {rec['reason']}")

        print(f"\n  {'=' * 66}")
        print(f"  PROJECTED DEMAND AFTER ADJUSTMENTS")
        print(f"  {'=' * 66}")
    else:
        print(f"\n  Demand looks reasonable - no adjustments needed.")

    print()


def print_projected_demand(
    config: Dict[str, Any],
    new_config: Dict[str, Any],
    estimate: Dict[str, Any],
    population_stats: Dict[str, Any],
    avg_legs: Dict[str, float],
    survey_trips_per_capita: float,
) -> None:
    """Estimate and print projected demand with the new config."""
    new_estimate = estimate_demand(new_config, population_stats, avg_legs, survey_trips_per_capita)
    new_tpc = new_estimate["totals"]["trips_per_capita"]

    old_tpc = estimate["totals"]["trips_per_capita"]
    target = estimate["benchmarks"]["survey_trips_per_capita"]

    print(f"    Trips/capita/day:       {old_tpc:.2f}  -->  {new_tpc:.2f}  (target: {target:.2f})")
    print(f"    Total plans (unscaled): {estimate['totals']['plans_unscaled']:>10,.0f}  -->  {new_estimate['totals']['plans_unscaled']:>10,.0f}")
    print(f"    Total trips (unscaled): {estimate['totals']['trips_unscaled']:>10,.0f}  -->  {new_estimate['totals']['trips_unscaled']:>10,.0f}")
    print(f"    Total plans (scaled):   {estimate['totals']['plans_scaled']:>10,.0f}  -->  {new_estimate['totals']['plans_scaled']:>10,.0f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Demand estimator - pre-run calibration tool for MATSim experiments"
    )
    parser.add_argument("config_file", help="Path to config JSON file (e.g. config/USA/TwinCities/config_twin.json)")
    parser.add_argument("--no-acs", action="store_true", help="Skip Census ACS API calls")
    parser.add_argument("--experiment-dir", type=str, default=None,
                        help="Path to a previous experiment folder to compare against")
    args = parser.parse_args()

    # Set up log file in logs/ directory
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"demand_estimator_{timestamp}.log"
    tee = TeeWriter(log_path)
    sys.stdout = tee

    # Load config
    config_path = Path(args.config_file)
    if not config_path.is_file():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    # Store config dir for relative path resolution
    config["_config_dir"] = str(config_path.parent)

    # Resolve relative data_dir to absolute (same logic as run_experiment.py)
    data_dir = config.get("data", {}).get("data_dir", "")
    if data_dir and not Path(data_dir).is_absolute():
        resolved = (config_path.parent / data_dir).resolve()
        config["data"]["data_dir"] = str(resolved)

    print(f"Loaded config: {config_path}")
    counties = config.get("region", {}).get("counties", [])
    print(f"Region: {len(counties)} counties")

    # --- Step 1: Compute real values from database ---
    print()
    print("-" * 70)
    print("  STEP 1: COMPUTING BENCHMARKS FROM DATABASE & SURVEYS")
    print("-" * 70)

    population_stats = compute_population_from_db(config)
    tpc_data, avg_legs, survey_travel_day_rate = compute_survey_benchmarks(config)

    survey_trips_per_capita = tpc_data["blended"]

    # --- Step 2: Fetch Census ACS data (mode share, cross-check) ---
    acs_data: Dict[str, Dict[str, int]] = {}
    api_key = config.get("data", {}).get("census_api_key", "")

    if not args.no_acs and counties and api_key:
        print()
        print("-" * 70)
        print("  STEP 2: FETCHING CENSUS ACS DATA (cross-check)")
        print("-" * 70)
        print(f"  [Source: Census ACS 5-year B08301 API, year={ACS_YEAR}]")
        print(f"  Fetching commute mode data for {len(counties)} counties...")
        acs_data = fetch_acs_commute_data(counties, api_key)
        print(f"  Retrieved data for {len(acs_data)} counties")
    elif not args.no_acs and not api_key:
        print("\n  No census_api_key in config - skipping ACS data fetch")
        print("  Add 'census_api_key' under 'data' section to enable")

    # --- Step 2b: Transit calibration from ACS bus/rail breakdown ---
    transit_calibration = None
    if acs_data:
        print()
        print("-" * 70)
        print("  STEP 2b: TRANSIT CALIBRATION FROM ACS BUS/RAIL DATA")
        print("-" * 70)
        transit_calibration = compute_transit_calibration(acs_data, config)
        if transit_calibration.get("acs_regional"):
            acs_r = transit_calibration["acs_regional"]
            print(f"  Regional transit share:  {acs_r['transit_share']:.1%}")
            print(f"    Bus:  {acs_r['bus_share']:.1%}  |  Rail: {acs_r['rail_share']:.1%}")
            print(f"  Transit calibration recommendations: "
                  f"{len(transit_calibration['recommendations'])}")
        else:
            print("  No transit data available for calibration")

    # --- Step 3: Load previous experiment feedback (only if --experiment-dir provided) ---
    experiment_feedback = None
    if args.experiment_dir:
        print()
        print("-" * 70)
        print("  STEP 3: LOADING PREVIOUS EXPERIMENT RESULTS")
        print("-" * 70)

        exp_dir = Path(args.experiment_dir)
        summary_file = exp_dir / "experiment_summary.json"
        if summary_file.is_file():
            try:
                with open(summary_file, "r") as f:
                    summary = json.load(f)
                eval_file = exp_dir / "evaluation" / "summary_metrics.json"
                eval_data = None
                if eval_file.is_file():
                    with open(eval_file, "r") as f:
                        eval_data = json.load(f)
                experiment_feedback = {
                    "summary": summary,
                    "evaluation": eval_data,
                    "experiment_dir": str(exp_dir),
                }
            except (json.JSONDecodeError, OSError) as e:
                print(f"  WARNING: Failed to load experiment from {exp_dir}: {e}")
        else:
            print(f"  WARNING: No experiment_summary.json found in {exp_dir}")

        if experiment_feedback:
            exp_name = Path(experiment_feedback["experiment_dir"]).name
            print(f"  Found experiment: {exp_name}")
            eval_data = experiment_feedback.get("evaluation")
            if eval_data:
                print(f"  Mean % error: {eval_data.get('mean_pct_error', 0):.1f}%")
                print(f"  GEH < 5:      {eval_data.get('geh_lt_5_pct', 0):.1f}%")
            else:
                print("  No evaluation data found")

    # --- Step 4: Estimate demand ---
    print()
    print("-" * 70)
    print("  STEP 4: ESTIMATING DEMAND FROM CONFIG PARAMETERS")
    print("-" * 70)
    estimate = estimate_demand(config, population_stats, avg_legs, survey_trips_per_capita)

    # --- Step 5: Compute scorecard ---
    scorecard = compute_scorecard(estimate, acs_data, config)

    # --- Step 6: Generate recommendations ---
    recommendations = recommend_adjustments(
        estimate, scorecard, survey_travel_day_rate,
        experiment_feedback=experiment_feedback,
    )

    # Merge transit calibration recommendations
    # Transit recs handle modes.bus.* and modes.rail.* parameters;
    # demand recs handle nonwork_purposes.* and plan_generation.scaling_factor.
    # If both transit calibration and demand adjustment recommend scaling_factor,
    # keep the larger value (more conservative — produces more agents).
    if transit_calibration and transit_calibration.get("recommendations"):
        transit_recs = transit_calibration["recommendations"]
        # Check for scaling_factor conflict
        demand_sf_rec = next(
            (r for r in recommendations if r["parameter"] == "plan_generation.scaling_factor"),
            None,
        )
        transit_sf_rec = next(
            (r for r in transit_recs if r["parameter"] == "plan_generation.scaling_factor"),
            None,
        )
        if demand_sf_rec and transit_sf_rec:
            # Keep the larger scaling_factor, remove the transit one
            if transit_sf_rec["recommended"] > demand_sf_rec["recommended"]:
                demand_sf_rec["recommended"] = transit_sf_rec["recommended"]
                demand_sf_rec["reason"] += (
                    f" (also adjusted for transit shift: {transit_sf_rec['reason']})"
                )
            transit_recs = [r for r in transit_recs
                           if r["parameter"] != "plan_generation.scaling_factor"]

        recommendations.extend(transit_recs)

    # --- Step 7: Print scorecard ---
    print_scorecard(
        estimate, scorecard, acs_data, recommendations, survey_travel_day_rate,
        experiment_feedback=experiment_feedback,
        transit_calibration=transit_calibration,
    )

    # --- Step 8: Generate adjusted config ---
    # Filter to actionable recommendations only (not _info. diagnostics)
    actionable_recs = [r for r in recommendations if not r["parameter"].startswith("_info.")]
    if actionable_recs:
        new_config = apply_recommendations(config, actionable_recs)

        # Remove internal keys before saving
        new_config.pop("_config_dir", None)

        # Print projected demand
        print_projected_demand(config, new_config, estimate, population_stats, avg_legs, survey_trips_per_capita)

        # Write output config
        stem = config_path.stem
        output_path = config_path.with_name(f"{stem}_estimated{config_path.suffix}")
        with open(output_path, "w") as f:
            json.dump(new_config, f, indent=2)
        print(f"Estimated config written to: {output_path}")
    else:
        print("No config changes needed.")

    # Close log file
    print(f"\nLog saved to: {log_path}")
    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()
