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
    # Cold start - positional is the base config JSON
    python estimators/demand_estimator.py config/USA/TwinCities/config_twin.json

    # Feedback - positional is the region FOLDER, --experiment-dir provides
    # the simulation result; the estimator reads <exp>/config_used.json and
    # writes <region>/config_estimated.json
    python estimators/demand_estimator.py config/USA/TwinCities \
        --experiment-dir experiments/exp_20260301

Output:
    Cold start:  <config_dir>/<stem>_estimated.json
    Feedback:    <region_folder>/config_estimated.json
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


def _resolve_experiment_dir(value: str) -> Path:
    """Accept either a full path to an experiment folder or just its name.

    A bare name (no separators or no leading drive/slash) is resolved
    against <project_root>/experiments/<name>. A full path is used as-is.
    """
    p = Path(value)
    if p.is_dir():
        return p
    # Bare name (or relative path that doesn't exist as-is): try project_root/experiments/
    candidate = project_root / "experiments" / value
    if candidate.is_dir():
        return candidate
    raise SystemExit(
        f"ERROR: --experiment-dir does not exist: tried {p} and {candidate}"
    )


def _print_acs_coverage_error(
    requested_fips: List[str],
    returned_fips: List[str],
) -> None:
    """Write a structured ACS-coverage failure report to stderr.

    Called from both estimators when the ACS fetch returned data for fewer
    than half of the configured counties. The message distinguishes the two
    failure modes (no states returned anything vs partial coverage) and
    lists the failed states so the user can correlate with the per-state
    retry warnings printed earlier in stdout.
    """
    requested_states = sorted({f[:2] for f in requested_fips})
    returned_states = sorted({f[:2] for f in returned_fips})
    failed_states = sorted(set(requested_states) - set(returned_states))
    missing_counties = sorted(set(requested_fips) - set(returned_fips))
    coverage = len(returned_fips) / len(requested_fips) if requested_fips else 0.0

    print("=" * 70, file=sys.stderr)
    print("  ERROR: ACS coverage too low to calibrate the region.", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(
        f"  Coverage: {coverage:.0%} ({len(returned_fips)}/{len(requested_fips)} counties)\n"
        f"  Requested states: {', '.join(requested_states)}\n"
        f"  Failed states:    {', '.join(failed_states) if failed_states else '(none — all states returned data)'}\n"
        f"  Missing counties: {', '.join(missing_counties)}",
        file=sys.stderr,
    )
    if not returned_fips:
        # Zero states succeeded → almost always key, network, or API outage.
        print(
            "\n  Likely cause: ALL state requests failed. The most common reasons are:\n"
            "    1. Census API key is invalid or not yet activated\n"
            "       (check email from api.census.gov for the activation link)\n"
            "    2. Network / DNS / VPN issue from this machine\n"
            "       (see 'DNS RESOLUTION FAILED' lines in the run log above)\n"
            "    3. Census API is down or rate-limiting (rare; retry in ~15min)\n"
            "  See the per-state retry warnings in stdout above for the exact\n"
            "  HTTP errors encountered. Refusing to write config_estimated.json.",
            file=sys.stderr,
        )
    else:
        # Some states returned data, others didn't. Either transient outage
        # for specific states, or the user configured an invalid/deprecated
        # county FIPS (e.g. Connecticut's 2022 reorganization away from
        # county FIPS to planning-region FIPS).
        print(
            "\n  Likely cause: PARTIAL failure — some states returned data, others didn't.\n"
            "    1. Transient outage for the failed states above\n"
            "       (see retry warnings in stdout above; rerun in ~15min)\n"
            "    2. Deprecated / invalid county FIPS in config.region.counties\n"
            "       (e.g. Connecticut switched from counties to planning regions\n"
            "       in 2022; old CT county FIPS no longer return data in ACS 2023+)\n"
            "    3. Typo in a county FIPS — verify each missing county against\n"
            "       https://www.census.gov/library/reference/code-lists/ansi.html\n"
            "  Refusing to write config_estimated.json.",
            file=sys.stderr,
        )


def require_acs_key(config: Dict[str, Any], config_path: Path | str) -> str:
    """Return the Census ACS key from config or exit loudly.

    Both estimators rely on ACS B08301. Running without a key produced silent
    failures (empty ACS dict -> 0% targets -> nonsense recommendations), so
    this is a hard gate. Shared between estimator.py, demand_estimator.py and
    mode_share_estimator.py so the message is identical no matter which entry
    point is used.
    """
    api_key = (config.get("data", {}).get("census_api_key", "") or "").strip()
    if api_key:
        return api_key
    print("=" * 70, file=sys.stderr)
    print("  ERROR: census_api_key missing from config.", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(
        f"  Config read: {config_path}\n"
        f"  Both estimators rely on Census ACS B08301 (mode share) data.\n"
        f"  Without a key the demand estimator cannot calibrate transit\n"
        f"  parameters and the mode-share estimator cannot compute targets;\n"
        f"  the resulting config_estimated.json would be unsafe to apply.\n"
        f"  Add 'census_api_key' under 'data' in config.json.\n"
        f"  Free key signup: https://api.census.gov/data/key_signup.html",
        file=sys.stderr,
    )
    sys.exit(2)


def resolve_estimator_inputs(
    positional: str,
    experiment_dir: str | None,
) -> Tuple[Path, Path]:
    """Resolve (config_to_read, estimated_output_path) for an estimator run.

    Two operating modes:

      Cold start (experiment_dir is None):
          positional must be a config JSON file.
          Read positional, write <stem>_estimated.json next to it.
          If positional already ends with _estimated, write back in place
          (chained re-run on the same estimate).

      Feedback (experiment_dir is provided):
          positional must be a region folder.
          Read <experiment_dir>/config_used.json as the state to update.
          Write to <region_folder>/config_estimated.json.

    Returns (read_from, write_to) as absolute paths.
    """
    pos = Path(positional)
    if experiment_dir is None:
        if not pos.is_file():
            raise SystemExit(
                f"ERROR: Config file not found: {pos}\n"
                f"(Cold start expects a JSON file. For feedback runs, pass a "
                f"region folder and --experiment-dir.)"
            )
        stem = pos.stem
        if stem.endswith("_estimated"):
            write_to = pos
        else:
            write_to = pos.with_name(f"{stem}_estimated{pos.suffix}")
        return pos, write_to

    if not pos.is_dir():
        raise SystemExit(
            f"ERROR: With --experiment-dir, the positional argument must be a "
            f"region folder, not a file: {pos}"
        )
    exp = _resolve_experiment_dir(experiment_dir)
    config_used = exp / "config_used.json"
    if not config_used.is_file():
        raise SystemExit(
            f"ERROR: config_used.json not found in experiment dir: {config_used}"
        )
    write_to = pos / "config_estimated.json"
    return config_used, write_to


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
ACS_YEAR = 2023  # latest 5-year ACS

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


def _request_acs_with_retry(
    url: str,
    params: Dict[str, str],
    state_fips: str,
    timeout: int = 60,
    max_attempts: int = 5,
) -> requests.Response | None:
    """GET the ACS endpoint with retries on timeout / connection / 5xx.

    Logs each attempt explicitly so that a transient timeout is visible in
    the run log instead of being silently swallowed. Returns the successful
    Response, or None if all attempts failed (caller should treat the state
    as missing, not silently zero).

    Backoff schedule: 5s, 15s, 30s, 60s between attempts (capped). Longer
    than typical API rate-limit waits because the common failure mode here
    is local DNS / VPN flakiness (NameResolutionError on api.census.gov),
    which can take 10-30s to recover.
    """
    import time
    backoff = 5.0
    max_backoff = 60.0
    last_err: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            # Retry on transient server errors; let 4xx fall through immediately.
            if resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}"
                print(f"  Census API state {state_fips}: attempt {attempt}/{max_attempts} "
                      f"got {resp.status_code}, retrying in {backoff:.0f}s...")
            else:
                resp.raise_for_status()
                if attempt > 1:
                    print(f"  Census API state {state_fips}: succeeded on attempt {attempt}")
                return resp
        except requests.Timeout:
            last_err = f"timeout after {timeout}s"
            print(f"  Census API state {state_fips}: attempt {attempt}/{max_attempts} "
                  f"TIMED OUT (>{timeout}s)"
                  f"{f'  retrying in {backoff:.0f}s...' if attempt < max_attempts else ''}")
        except requests.ConnectionError as e:
            # Surface DNS-resolution failures distinctly — those usually indicate
            # local network / VPN issues rather than a Census API outage.
            err_str = str(e)
            is_dns = "NameResolutionError" in err_str or "Failed to resolve" in err_str
            last_err = f"DNS resolution failure: {e}" if is_dns else f"connection error: {e}"
            kind = "DNS RESOLUTION FAILED" if is_dns else "connection error"
            print(f"  Census API state {state_fips}: attempt {attempt}/{max_attempts} "
                  f"{kind}"
                  f"{f'  retrying in {backoff:.0f}s...' if attempt < max_attempts else ''}")
        except requests.HTTPError as e:
            # Non-5xx HTTP errors: don't retry, surface and bail.
            print(f"  Census API state {state_fips}: HTTP error {e} — not retrying")
            return None
        except requests.RequestException as e:
            last_err = str(e)
            print(f"  Census API state {state_fips}: attempt {attempt}/{max_attempts} "
                  f"failed ({e})"
                  f"{f'  retrying in {backoff:.0f}s...' if attempt < max_attempts else ''}")

        if attempt < max_attempts:
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    print(f"  !! Census API state {state_fips}: gave up after {max_attempts} attempts "
          f"(last error: {last_err}). This state's counties will be MISSING from ACS data.")
    return None


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

    Notes:
        Per-state requests are retried on timeout / connection error / 5xx
        with exponential backoff. A state that exhausts its retries is
        logged loudly and its counties are omitted from the result so the
        caller can detect partial coverage (see the coverage check in
        ``main`` below). Silent zero-coverage was the cause of the
        Twin Cities mis-calibration where MN timed out and the regional
        transit share collapsed to the WI-only value.
    """
    # Group counties by state
    by_state: Dict[str, List[str]] = {}
    for fips in county_fips_list:
        state = fips[:2]
        county = fips[2:]
        by_state.setdefault(state, []).append(county)

    results: Dict[str, Dict[str, int]] = {}
    failed_states: List[str] = []
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

        resp = _request_acs_with_retry(url, params, state_fips)
        if resp is None:
            failed_states.append(state_fips)
            continue

        # Census returns HTML (not JSON) for invalid/unactivated keys.
        content_type = resp.headers.get("content-type", "")
        if "json" not in content_type:
            if api_key:
                print(f"  WARNING: Census API key may be invalid or not yet activated.")
                print(f"  Check your email for the activation link from api.census.gov")
                print(f"  Retrying state {state_fips} without API key...")
                params.pop("key", None)
                resp = _request_acs_with_retry(url, params, state_fips)
                if resp is None or "json" not in resp.headers.get("content-type", ""):
                    print(f"  WARNING: Census API still returning non-JSON for state {state_fips}")
                    failed_states.append(state_fips)
                    continue
            else:
                print(f"  WARNING: Census API returned non-JSON for state {state_fips}")
                failed_states.append(state_fips)
                continue

        try:
            data = resp.json()
        except json.JSONDecodeError:
            print(f"  WARNING: Census API returned invalid JSON for state {state_fips}")
            failed_states.append(state_fips)
            continue

        if len(data) < 2:
            print(f"  WARNING: Census API returned no rows for state {state_fips}")
            failed_states.append(state_fips)
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

    if failed_states:
        print(f"  !! ACS fetch summary: {len(failed_states)}/{len(by_state)} states "
              f"failed: {', '.join(sorted(failed_states))}")
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

    # We need person-day counts per source to compute a correct daily rate
    # for multi-day surveys (TBI). process_persons() groups trips by
    # person_id then by date, so len(days_dict) is the number of distinct
    # diary days that person reported — works for both single-day (NHTS)
    # and multi-day (TBI) surveys without hardcoding survey-specific logic.
    print("  [Source: survey_trips table in DuckDB — TBI + NHTS surveys]")
    print("  Computing trips-per-capita from survey data...")

    all_persons_by_source = survey_manager.process_persons()

    tpc_result = {}
    weighted_sum = 0.0
    total_weight = 0.0
    person_days_by_source: Dict[str, int] = {}

    for source_name, df in all_data.items():
        if df.empty:
            print(f"    {source_name}: no data in database — skipping")
            continue

        persons_dict = all_persons_by_source.get(source_name, {})
        # Total reported diary days across all persons in this survey.
        n_person_days = sum(len(days) for days in persons_dict.values())
        n_persons = len(persons_dict)
        if n_person_days == 0:
            continue

        # Daily trips/person among reporting person-days.
        # Caveat: surveys' trip tables only contain rows for days with >=1
        # trip, so a person who reported 0 trips on a given diary day is
        # invisible here. This rate is therefore "trips per *travel*
        # person-day" — we multiply by the participation rate below to
        # convert to a population-wide trips/capita/day.
        tpc_per_travel_day = len(df) / n_person_days

        tpc_result[source_name] = round(tpc_per_travel_day, 2)
        person_days_by_source[source_name] = n_person_days
        source_weight = blend_weights.get(source_name, 1.0)
        weighted_sum += tpc_per_travel_day * source_weight
        total_weight += source_weight

        print(f"    {source_name}: {tpc_per_travel_day:.2f} trips/person-day "
              f"({len(df):,} trips, {n_persons:,} persons, "
              f"{n_person_days:,} person-days, weight={source_weight})")

    if total_weight > 0:
        tpc_result["blended"] = round(weighted_sum / total_weight, 2)
    else:
        print("  ERROR: No survey data found in database.")
        sys.exit(1)

    print(f"  => Blended trips/travel-person-day: {tpc_result['blended']:.2f}")

    # ----- Travel-day participation rate -----
    # The trip table contains rows only for days with >=1 trip, so simply
    # counting persons-with-trips / total-persons-in-table will always be
    # ~100%. To get a defensible participation rate we'd need a person
    # roster including zero-trip days, which neither TBI nor NHTS exposes
    # through survey_trips. Until a roster table exists, fall back to the
    # config-provided nonwork_trip_share (typically ~0.80) and skip the
    # auto-recommendation that pushed it to 1.0 based on the bogus 100%.
    nonwork_cfg = config.get("nonwork_purposes", {})
    config_share = float(nonwork_cfg.get("nonwork_trip_share", 0.80))
    # Clamp to [0, 1] to be safe.
    config_share = max(0.0, min(1.0, config_share))
    computed_travel_rate = config_share

    print("  Travel-day participation rate (non-travel days are not in the trip "
          "table, so we cannot compute this from surveys directly):")
    print(f"    using config nonwork_purposes.nonwork_trip_share = "
          f"{config_share:.2%} as participation proxy")

    # Adjust trips/travel-day to a population-wide trips/capita/day.
    # Multiplying by the participation rate spreads trip-day totals over
    # the full population (including non-travelers).
    unadjusted_blended = tpc_result["blended"]
    tpc_result["blended"] = round(unadjusted_blended * computed_travel_rate, 2)
    print(f"  => Adjusted trips/capita/day: {unadjusted_blended:.2f} × "
          f"{computed_travel_rate:.2%} = {tpc_result['blended']:.2f}  "
          f"[accounts for non-travelers via nonwork_trip_share]")

    # ----- Avg legs per chain -----
    print("  Computing avg legs per chain from survey trip chains...")

    # Merge persons across sources for chain processing (same logic as
    # SurveyManager.get_persons(), but we already have processed data).
    persons: Dict = {}
    for persons_dict in all_persons_by_source.values():
        persons.update(persons_dict)

    if not persons:
        print("  ERROR: No person data found in database for chain analysis.")
        sys.exit(1)

    use_weight = config.get('chains', {}).get('use_weighted_chains', True)
    chains = process_trip_chains(persons, use_weight=use_weight)

    if not chains:
        print("  ERROR: No trip chains could be extracted from survey data.")
        sys.exit(1)

    chains_df = pd.DataFrame(chains)

    # Compute weighted average legs for each chain type
    def _weighted_avg_legs(chains_df, filter_func, label: str):
        """Compute weighted average legs for chains matching filter_func."""
        mask = chains_df['pattern'].apply(filter_func)
        filtered = chains_df[mask]
        n_total_patterns = len(chains_df)
        n_matched_patterns = len(filtered)
        if filtered.empty:
            print(f"    [{label}] no chains matched filter "
                  f"(checked {n_total_patterns} unique patterns)")
            return None

        # Each pattern's legs = number of activities - 1
        legs = filtered['pattern'].apply(lambda p: len(p.split('-')) - 1)

        if 'probability' in filtered.columns:
            weights = filtered['probability']
        elif 'count' in filtered.columns:
            weights = filtered['count']
        else:
            weights = None

        if weights is not None:
            avg = (legs * weights).sum() / weights.sum()
            weight_share = weights.sum()  # share of all chains in this bucket
        else:
            avg = legs.mean()
            weight_share = n_matched_patterns / n_total_patterns

        # Top-3 patterns by weight for visibility
        top3 = (
            filtered.assign(_legs=legs, _w=weights if weights is not None
                            else pd.Series(1, index=filtered.index))
            .nlargest(3, '_w')[['pattern', '_legs', '_w']]
        )
        print(f"    [{label}] {n_matched_patterns}/{n_total_patterns} unique "
              f"patterns, weight-share={weight_share:.3f}, "
              f"raw_avg_legs={avg:.4f}")
        for _, row in top3.iterrows():
            print(f"      top: {row['pattern']!r}  legs={row['_legs']}  "
                  f"weight={row['_w']:.4f}")
        return avg

    avg_work = _weighted_avg_legs(chains_df, is_home_work_home_chain, "work")
    avg_nonwork = _weighted_avg_legs(chains_df, is_home_other_home_chain, "nonwork")

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
    acs_data: Dict[str, Dict[str, int]] | None = None,
) -> Dict[str, Any]:
    """Estimate total demand from config parameters without running the sim.

    Mirrors the formulas in plan_generator.py to predict how many plans and
    trips the current config will produce.

    Args:
        config: Configuration dictionary.
        population_stats: From compute_population_from_db().
        avg_legs: From compute_survey_benchmarks(), e.g. {'work': 2.7, 'nonwork': 2.0}.
        survey_trips_per_capita: Blended trips/capita from surveys.
        acs_data: Optional per-county ACS commute mode data. When present, the
            ACS work-from-home share is used to discount work-plan generation
            (WFH workers don't commute) and those WFH workers are reassigned
            to the nonwork pool.

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

    # --- Population reallocation (WFH + worker-nonwork share) ---
    # Two adjustments applied to the raw LODES population split:
    #
    # 1. Work-from-home discount (Fix #2). ACS B08301_021E reports the share
    #    of workers who primarily worked from home. These workers exist as
    #    agents but do not generate a Home->Work->Home commute. Move them
    #    into the nonwork pool so they still produce nonwork trips
    #    (errands, school drop-offs, etc.).
    #
    # 2. Worker-nonwork-share approximation (Fix #4). The nonwork generator
    #    currently runs only on `non_employees`, so the "before/after work"
    #    nonwork legs that commuting workers make (lunch, gym, daycare) are
    #    effectively missing — the work generator handles them only via
    #    avg_legs_work, and the dominant Home-Work-Home pattern under-counts
    #    them. As a quick approximation, treat a fraction `alpha` of
    #    commuting workers as additional nonwork-pool members. NHTS suggests
    #    workers contribute ~40-50% of all nonwork person-trips, so 0.45 is
    #    a reasonable default. This is a *quick fix* — it inflates totals
    #    correctly but does not produce worker-attached chains.
    nonwork_cfg = config.get("nonwork_purposes", {})
    wfh_rate = 0.0
    wfh_source = None
    if acs_data:
        acs_total_workers = sum(d.get("total_workers", 0) for d in acs_data.values())
        acs_total_wfh = sum(d.get("work_from_home", 0) for d in acs_data.values())
        if acs_total_workers > 0:
            wfh_rate = acs_total_wfh / acs_total_workers
            wfh_source = "ACS B08301_021E"
    # Allow config override (e.g. for sensitivity analysis or when ACS is off).
    cfg_wfh = nonwork_cfg.get("wfh_rate_override")
    if cfg_wfh is not None:
        wfh_rate = max(0.0, min(1.0, float(cfg_wfh)))
        wfh_source = "config nonwork_purposes.wfh_rate_override"

    worker_nonwork_alpha = float(
        nonwork_cfg.get("worker_nonwork_share", 0.45)
    )
    worker_nonwork_alpha = max(0.0, min(1.0, worker_nonwork_alpha))

    wfh_employees = employees * wfh_rate
    commuting_employees = employees - wfh_employees
    # Workers added to nonwork pool: WFH workers (full count, they're home
    # all day) plus alpha * commuting workers (partial — these workers also
    # appear on the work side).
    workers_in_nonwork_pool = wfh_employees + worker_nonwork_alpha * commuting_employees
    effective_nonwork_population = non_employees + workers_in_nonwork_pool

    # --- Work trips ---
    # Only commuting workers generate work plans.
    effective_work_scaling = scaling_factor * work_scaling_multiplier
    work_plans_unscaled = commuting_employees
    work_plans_scaled = work_plans_unscaled * effective_work_scaling

    # --- Nonwork trips ---
    # Pool = non_employees + WFH workers + alpha * commuting workers.
    # nonwork_trip_share is then applied as the per-day participation rate
    # within that pool (default ~0.80 — config-driven, not auto-pushed).
    nonwork_trip_share = nonwork_cfg.get("nonwork_trip_share", 1.0)
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

        purpose_trips = effective_nonwork_population * final_rate * nonwork_trip_share
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
    cap_ratio = flow_cap / scaling_factor if scaling_factor > 0 else 1.0

    return {
        "population": {
            "total": total_pop,
            "employees": employees,
            "non_employees": non_employees,
            "wfh_rate": round(wfh_rate, 4),
            "wfh_source": wfh_source,
            "wfh_employees": wfh_employees,
            "commuting_employees": commuting_employees,
            "worker_nonwork_alpha": worker_nonwork_alpha,
            "workers_in_nonwork_pool": workers_in_nonwork_pool,
            "effective_nonwork_population": effective_nonwork_population,
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
    skip_keys = {
        "nonwork_trip_share", "_nonwork_trip_share_help",
        "worker_nonwork_share", "wfh_rate_override",
    }

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

# NHTS national baselines (approximate). Kept at module scope so the empirical
# back-calculation can reuse the exact same constants as the forward formula.
NHTS_BUS_SHARE = 0.02
NHTS_RAIL_SHARE = 0.01


def compute_empirical_chain_reduction(
    experiment_feedback: Dict[str, Any] | None,
) -> Dict[str, float] | None:
    """Back out chain_reduction per mode from a prior run's recorded stats.

    Uses the end-to-end identity
        observed_output_share = (1 - chain_reduction) * prefilt_rate
    with
        prefilt_rate = (1 - blend_weight) * nhts_share + blend_weight * config_rate
    where mode_choice's mode_distribution gives `observed_output_share` and the
    prior run's config (snapshotted in experiment_summary.json under
    `parameters` -> `modes`) gives `blend_weight` and `config_rate`.

    Returns {"bus": float, "rail": float} (each key omitted if it cannot be
    derived) or None if no usable data is present. Values are clamped to a
    plausible [0.05, 0.90] band; anything outside that range almost certainly
    means the formula's preconditions don't hold (e.g. config_rate=0).
    """
    if not experiment_feedback:
        return None
    summary = experiment_feedback.get("summary") or {}
    mode_choice = summary.get("mode_choice") or {}
    mode_dist = mode_choice.get("mode_distribution") or {}
    prior_modes = (summary.get("parameters") or {}).get("modes") or {}
    if not mode_dist or not prior_modes:
        return None

    out: Dict[str, float] = {}
    for mode_name, nhts_share in (("bus", NHTS_BUS_SHARE), ("rail", NHTS_RAIL_SHARE)):
        cfg = prior_modes.get(mode_name) or {}
        config_rate = cfg.get("config_rate")
        bw = cfg.get("blend_weight")
        observed = mode_dist.get(mode_name)
        if config_rate is None or bw is None or observed is None:
            continue
        prefilt = (1 - bw) * nhts_share + bw * config_rate
        if prefilt <= 0:
            continue
        cr = 1.0 - (observed / prefilt)
        if cr < 0.05 or cr > 0.90:
            # Outside plausible band; treat as unmeasured and fall back to seed.
            continue
        out[mode_name] = cr
    return out or None


def compute_transit_calibration(
    acs_data: Dict[str, Dict[str, int]],
    config: Dict[str, Any],
    empirical_chain_reduction: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Compute transit mode parameter recommendations from ACS county-level data.

    Uses ACS B08301 bus/rail breakdown to calculate region-specific config_rate,
    blend_weight, access_buffer_meters, and scaling_factor adjustments for the
    bus and rail modes in the config.

    The calibration logic follows the approach described in the evaluation:
    - Compute regional weighted-average bus and rail commute shares from ACS
    - Set config_rate above the target output share because the chain
      consistency constraint (all legs must have the mode available)
      removes ~30-50% of transit-eligible chains depending on regional
      transit share (seed values are 30/40/45/50% for transit shares
      >20/10/5/0%). After the walk-pass-through change in mode_choice.py
      these seeds were lowered ~15pp from the previous 45-65% range —
      short interior legs are now excused from the chain intersection
      and emitted as walk rather than dropping the whole chain. Empirical
      values from a prior experiment override the seed when available.
    - Set blend_weight based on how far the regional transit share is from the
      NHTS national average (~3-5% total transit)
    - Set access_buffer_meters based on transit density (higher share = denser
      network = larger reasonable walking catchment)

    Note: this function intentionally does NOT touch scaling_factor or the
    capacity factors. scaling_factor is the population sample fraction —
    it controls how many agents are simulated regardless of mode, so a
    mode shift from car to transit does not change it. If the resulting
    car-only volumes look low after a transit-share calibration, that is
    a separate decision driven by counts, not by ACS mode shares.

    Args:
        acs_data: Per-county ACS data from fetch_acs_commute_data().
        config: Current config dictionary.
        empirical_chain_reduction: Optional {"bus": float, "rail": float} measured
            from a prior run's experiment_summary.json. When provided, overrides
            the cold-start seed table for the mode(s) present.

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
    matsim_params = config.get("matsim", {}).get("configurable_params", {})
    cur_flow_cap = matsim_params.get("qsim.flowCapacityFactor", cur_scaling)
    cur_storage_cap = matsim_params.get("qsim.storageCapacityFactor", cur_scaling)

    # --- NHTS national baselines (approximate) ---
    nhts_bus_share = NHTS_BUS_SHARE
    nhts_rail_share = NHTS_RAIL_SHARE

    # --- Compute recommended parameters ---
    # chain_reduction = 1 - (observed_pt_share / prefilt_pt_rate). Empirical
    # value (from a prior run's experiment_summary.json) is preferred; the
    # table below is the cold-start seed.
    #
    # Seed values were lowered by ~15pp vs. the original heuristic to reflect
    # walk-pass-through (mode_choice.py:541-554): short interior legs are now
    # excused from the chain intersection and emitted as walk, so mixed-
    # feasibility chains no longer get dropped wholesale.
    bus_empirical = (empirical_chain_reduction or {}).get("bus")
    rail_empirical = (empirical_chain_reduction or {}).get("rail")

    if transit_share > 0.20:
        chain_reduction_seed = 0.30
    elif transit_share > 0.10:
        chain_reduction_seed = 0.40
    elif transit_share > 0.05:
        chain_reduction_seed = 0.45
    else:
        chain_reduction_seed = 0.50

    chain_reduction_bus = bus_empirical if bus_empirical is not None else chain_reduction_seed
    chain_reduction_rail = rail_empirical if rail_empirical is not None else chain_reduction_seed
    # Clamp to a safe band so a degenerate prior run can't blow up prefilt.
    chain_reduction_bus = max(0.05, min(chain_reduction_bus, 0.90))
    chain_reduction_rail = max(0.05, min(chain_reduction_rail, 0.90))
    # Legacy alias kept for log/reporting paths that still expect a single value.
    chain_reduction = (chain_reduction_bus + chain_reduction_rail) / 2.0

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
    # This is the rate we need BEFORE the chain consistency filter removes chains.
    # bus and rail use independently estimated reductions when an empirical
    # value is available (their feasibility geometry is different).
    prefilt_bus = target_output_bus / (1 - chain_reduction_bus) if chain_reduction_bus < 1 else target_output_bus
    prefilt_rail = target_output_rail / (1 - chain_reduction_rail) if chain_reduction_rail < 1 else target_output_rail

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

    # Measure the expected car-volume reduction from the transit shift, for
    # the diagnostic warning only. scaling_factor / flow / storage are left
    # untouched: scaling_factor is the population sample fraction and does
    # not depend on mode mix. If counts later show car volumes are low, the
    # counts-driven path in recommend_adjustments handles that separately.
    expected_transit_output = target_output_bus + target_output_rail

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
                      f"~{chain_reduction_bus:.0%} chain consistency reduction "
                      f"({'empirical' if bus_empirical is not None else 'seed'}). "
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
                      f"~{chain_reduction_rail:.0%} chain consistency reduction "
                      f"({'empirical' if rail_empirical is not None else 'seed'}).",
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

    # Informational only: warn that calibrating PT/rail upward will reduce
    # the car share, which mechanically lowers simulated car volumes on the
    # network. scaling_factor and capacity factors are intentionally NOT
    # changed here — see the function docstring. The counts-driven path in
    # recommend_adjustments is the only place that may adjust scaling_factor.
    if expected_transit_output > 0.01:
        car_volume_drop = expected_transit_output  # car share goes down by ~ this much
        recommendations.append({
            "parameter": "_info.transit_shift_volume_warning",
            "current": (
                f"target transit output ~{expected_transit_output:.1%} "
                f"(bus ~{target_output_bus:.1%}, rail ~{target_output_rail:.1%})"
            ),
            "recommended": "no automatic scaling_factor change",
            "reason": (
                f"After applying the transit calibration, the simulated car "
                f"mode share will drop by ~{car_volume_drop:.1%}, mechanically "
                f"reducing car volumes at count stations by a similar amount. "
                f"scaling_factor (currently {cur_scaling}) and capacity factors "
                f"(flow={cur_flow_cap}, storage={cur_storage_cap}) are NOT "
                f"changed here: scaling_factor is the population sample "
                f"fraction and does not depend on mode mix. If observed-vs-sim "
                f"car volumes look low after this calibration runs, address "
                f"that through the counts-driven path (which uses iqr_mean of "
                f"per-station sim/obs ratios), not by inflating scaling_factor "
                f"to compensate for a mode shift."
            ),
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
        "chain_reduction_bus": round(chain_reduction_bus, 4),
        "chain_reduction_rail": round(chain_reduction_rail, 4),
        "chain_reduction_source": {
            "bus": "empirical" if bus_empirical is not None else "seed",
            "rail": "empirical" if rail_empirical is not None else "seed",
            "seed": round(chain_reduction_seed, 4),
        },
        "all_trip_factor": all_trip_factor,
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
        eval_data = read_evaluation_metrics(best_summary.parent, summary)
        return {
            "summary": summary,
            "evaluation": eval_data,
            "experiment_dir": str(best_summary.parent),
        }
    except (json.JSONDecodeError, OSError):
        return None


def read_evaluation_metrics(
    exp_dir: Path,
    summary: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """Evaluation metrics for a run, from experiment_summary.json.

    The summary's ``evaluation`` section is the single source of truth. Runs
    produced before that consolidation also wrote
    ``evaluation/summary_metrics.json``; that file is read as a fallback so
    older experiments stay usable, but nothing writes it any more.
    """
    exp_dir = Path(exp_dir)
    if summary is None:
        summary_file = exp_dir / "experiment_summary.json"
        if summary_file.is_file():
            try:
                with open(summary_file, "r") as f:
                    summary = json.load(f)
            except (json.JSONDecodeError, OSError):
                summary = None

    section = (summary or {}).get("evaluation")
    if isinstance(section, dict) and section:
        return section

    legacy = exp_dir / "evaluation" / "summary_metrics.json"
    if legacy.is_file():
        try:
            with open(legacy, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
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
    # The sf recommendation only compensates for *measured* simulation losses
    # (plan-generation shortfall + stuck agents). It does NOT try to close
    # the count gap by inflating scaling_factor — scaling_factor represents
    # the sample fraction of the *resident* population, so the right ceiling
    # is "10% sampled, plus a small allowance for losses around that 10%."
    # Anything larger over-represents residents to compensate for things
    # that aren't residents (freight, through-traffic, visitors, survey
    # under-estimation) and degrades calibration rather than helping it.
    needs_more_demand = False
    needs_less_demand = False   # counts say the sim is over-producing (uniform error only)
    over_ratio = 1.0            # measured iqr_mean when over-producing
    plan_gen_ratio = 1.0    # measured plan-generator yield (1.0 = no measurement)
    stuck_fraction = 0.0    # measured stuck-agent fraction (0.0 = no measurement)

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
            # Plan generation gap: estimated vs actually generated.
            # Captured here so Lever 4 (scaling_factor) can compensate for
            # plan-generator losses (Fix #7).
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

        # Evaluation metrics decide whether to recommend a sf bump.
        #
        # Gate is driven by iqr_mean (interquartile mean of per-station
        # sim/obs ratios — robust to boundary/through-traffic outliers and
        # to a few catastrophic stations that drag mean_pct_error and GEH<5
        # down without telling us anything about the interior of the
        # modeled region). NHTS-implied yield is reported as advisory only
        # because the survey benchmark is a national rate and is known to
        # disagree with regional count totals (e.g. for LA, NHTS gives
        # ~2.35 trips/capita/day vs counts implying ~4x higher car volumes
        # driven by external/freight/visitor trips the survey can't see).
        #
        # The sf recommendation that comes out of this block ONLY
        # compensates for measured simulation losses (plan-gen + stuck);
        # see the sf math after the rate-levers section.
        stuck_agents = summary.get("matsim_output", {}).get("total_stuck_agents", 0)
        if actual_plans > 0:
            stuck_fraction = stuck_agents / actual_plans
        if eval_data:
            mean_pct_error = eval_data.get("mean_pct_error", 0)
            geh_lt_5 = eval_data.get("geh_lt_5_pct", 0)
            iqr_mean = eval_data.get("interquartile_mean_ratio")
            pct_below_10 = eval_data.get("pct_stations_below_10pct", 0.0)
            num_below_10 = eval_data.get("num_stations_below_10pct", 0)

            # Advisory: NHTS-implied trip-volume yield. Printed for context
            # so the user can see whether the simulation produced trips at
            # the survey-implied rate. NOT used to gate sf decisions.
            actual_trips = summary.get("matsim_output", {}).get("output_trips_count", 0)
            expected_unscaled = total_pop * benchmarks["survey_trips_per_capita"]
            expected_scaled = expected_unscaled * scaling_factor
            yield_ratio = actual_trips / expected_scaled if expected_scaled > 0 else 0.0

            # iqr_mean fallback for older experiments that didn't write it.
            if iqr_mean is None or iqr_mean <= 0:
                iqr_mean_used = max(1.0 + mean_pct_error / 100.0, 0.0)
                iqr_source = "fallback from mean_pct_error"
            else:
                iqr_mean_used = float(iqr_mean)
                iqr_source = "interquartile_mean_ratio"

            # Thresholds — counts-driven, single gate on iqr_mean:
            #   >= IQR_OK     -> counts look fine, no sf change.
            #   >= IQR_LOW    -> counts moderately low, small compensation bump.
            #   <  IQR_LOW    -> counts catastrophically low, same compensation
            #                    bump PLUS a loud user-review diagnostic.
            IQR_OK = 0.70
            IQR_LOW = 0.50
            # Over-production gate. Unlike the under-production side (which only
            # ever compensates for measured losses), a decrease is a genuine
            # correction — so it needs a second condition proving the error is
            # actually global.
            IQR_HIGH = 1.15     # above this, the sim is materially over-producing
            CV_UNIFORM = 0.35   # per-station spread below this = uniform error

            iqr_ok = IQR_OK <= iqr_mean_used <= IQR_HIGH
            iqr_catastrophic = iqr_mean_used < IQR_LOW

            # Concentrated-vs-uniform test (G6). Written by the evaluator as
            # station_ratio_cv. Without it we cannot tell "every station is
            # 35% high" (one global lever fixes it) from "a few corridors are
            # wildly high and the rest are fine" (a global lever makes things
            # worse), so a missing CV blocks the decrease rather than guessing.
            station_cv = eval_data.get("station_ratio_cv")
            cv_known = station_cv is not None and station_cv > 0
            cv_uniform = cv_known and float(station_cv) < CV_UNIFORM

            if iqr_mean_used > IQR_HIGH:
                over_ratio = iqr_mean_used
                if cv_uniform:
                    needs_less_demand = True
                    recommendations.append({
                        "parameter": "_info.counts_over_production",
                        "current": (
                            f"iqr_mean={iqr_mean_used:.1%}, "
                            f"station_ratio_cv={float(station_cv):.3f} (uniform)"
                        ),
                        "recommended": f"reduce sf toward 1/{iqr_mean_used:.3f}",
                        "reason": (
                            f"Simulation is producing {iqr_mean_used:.0%} of observed "
                            f"volumes and the per-station spread is uniform "
                            f"(CV {float(station_cv):.3f} < {CV_UNIFORM}), so every "
                            f"station is off by roughly the same factor. That is the "
                            f"one case where a single global lever is the right tool. "
                            f"Recommending a capped scaling_factor reduction below."
                        ),
                    })
                else:
                    cv_txt = (f"station_ratio_cv={float(station_cv):.3f}"
                              if cv_known else "station_ratio_cv not recorded")
                    recommendations.append({
                        "parameter": "_info.over_production_not_uniform",
                        "current": f"iqr_mean={iqr_mean_used:.1%}, {cv_txt}",
                        "recommended": "no automatic scaling_factor reduction",
                        "reason": (
                            f"Simulation is over-producing ({iqr_mean_used:.0%}) but the "
                            f"error is NOT uniform across stations"
                            + (f" (CV {float(station_cv):.3f} >= {CV_UNIFORM})"
                               if cv_known else
                               " (CV not recorded by this run's evaluator)")
                            + f". Lowering scaling_factor globally would pull down the "
                            f"stations that are already correct while leaving the "
                            f"concentrated excess in place. Fix the spatial "
                            f"distribution first (OD source, boundary policy, "
                            f"network/counts matching), then re-check."
                        ),
                    })

            # Always report what we measured, regardless of decision.
            recommendations.append({
                "parameter": "_info.station_ratio_summary",
                "current": (
                    f"iqr_mean={iqr_mean_used:.1%} ({iqr_source}), "
                    f"station_ratio_cv="
                    + (f"{float(station_cv):.3f}" if cv_known else "n/a")
                    + f", mean_pct_error={mean_pct_error:.1f}%, "
                    f"GEH<5={geh_lt_5:.1f}%, "
                    f"stations<10%={num_below_10} ({pct_below_10:.0f}%)"
                ),
                "recommended": f"{IQR_OK:.0%} <= iqr_mean <= {IQR_HIGH:.0%}",
                "reason": (
                    f"Interquartile mean of per-station sim/obs ratios drops "
                    f"the worst & best 25% of stations and averages the middle 50%. "
                    f"Robust to boundary stations with through-traffic from outside "
                    f"the modeled area. station_ratio_cv is the concentrated-vs-"
                    f"uniform test: below {CV_UNIFORM} the error is global and a "
                    f"scaling_factor change is the right lever; above it the error "
                    f"sits on particular corridors and a global multiplier would "
                    f"make the already-correct stations worse. mean_pct_error and "
                    f"GEH<5 are reported for context but are volume-weighted and "
                    f"dominated by a few boundary outliers, so they do not gate "
                    f"sf decisions."
                ),
            })
            recommendations.append({
                "parameter": "_info.yield_advisory",
                "current": (
                    f"output_trips={actual_trips:,}, "
                    f"expected_scaled={expected_scaled:,.0f}, "
                    f"yield={yield_ratio:.1%}"
                ),
                "recommended": "advisory only — not used to gate sf",
                "reason": (
                    f"NHTS-implied yield = output_trips / (survey_tpc * total_pop * sf) "
                    f"= {yield_ratio:.1%}. The NHTS benchmark ({benchmarks['survey_trips_per_capita']:.2f} "
                    f"trips/capita/day) is a national survey rate and can disagree "
                    f"with regional count totals because of external trips, freight, "
                    f"visitors, and survey under-estimation of local daily activity. "
                    f"Counts (iqr_mean above) drive the decision; this number is "
                    f"shown for transparency only."
                ),
            })

            if iqr_mean_used < IQR_OK:
                # Counts say there is room for more agents — but only as far
                # as measured simulation losses justify. The sf math after
                # the rate-levers section caps the change at +0.03 absolute
                # and 0.13 ceiling (i.e. stays in the ~10% sample band).
                # Keyed on being genuinely BELOW target, not merely outside
                # the band: iqr_ok is now two-sided, so `not iqr_ok` is also
                # true when the sim is over-producing, which must not request
                # *more* demand.
                needs_more_demand = True

            if iqr_catastrophic:
                # Loud diagnostic: the auto-bump only compensates for plan-gen
                # and stuck-agent losses. The implied raw multiplier from
                # counts is much larger and is almost certainly NOT something
                # to chase by inflating sf beyond ~13%.
                raw_multiplier = 1.0 / iqr_mean_used if iqr_mean_used > 0 else float("inf")
                recommendations.append({
                    "parameter": "_info.user_review_required",
                    "current": (
                        f"iqr_mean={iqr_mean_used:.0%} (catastrophic), "
                        f"raw multiplier implied by counts ~{raw_multiplier:.1f}x"
                    ),
                    "recommended": "manual decision on scaling_factor",
                    "reason": (
                        f"USER REVIEW REQUIRED. Counts indicate sim is at "
                        f"{iqr_mean_used:.0%} of observed volumes region-wide "
                        f"(raw multiplier ~{raw_multiplier:.1f}x). The auto-recommended "
                        f"sf change ONLY compensates for measured plan-generation "
                        f"and stuck-agent losses (typical 1.05-1.30x range) — it "
                        f"does NOT and SHOULD NOT try to close the {raw_multiplier:.1f}x "
                        f"gap. scaling_factor is the sample fraction of the resident "
                        f"population; inflating it past ~13% over-represents residents "
                        f"to compensate for things that aren't residents: "
                        f"(1) external / through-traffic, (2) freight and commercial "
                        f"vehicles, (3) non-resident trips, (4) the survey under-"
                        f"estimating local daily activity. If you decide a larger sf "
                        f"is justified, you MUST raise flowCapacityFactor and "
                        f"storageCapacityFactor in lockstep (flow=sf, storage~=sf*1.2) "
                        f"or the network will gridlock — review the run manually."
                    ),
                })
            elif iqr_mean_used < IQR_OK:
                # Moderate: just say what's happening, no loud header needed.
                recommendations.append({
                    "parameter": "_info.iqr_moderate_low",
                    "current": f"iqr_mean={iqr_mean_used:.0%} (moderate)",
                    "recommended": "small sf compensation bump",
                    "reason": (
                        f"Counts are moderately below sim ({iqr_mean_used:.0%} "
                        f"interquartile mean). Applying compensation-only sf bump "
                        f"for measured plan-gen / stuck-agent losses; see sf "
                        f"recommendation below."
                    ),
                })

    # Two distinct kinds of "needs adjustment":
    #   A) tpc < target_low                — sim under-generates trips/agent;
    #                                        rate levers (Levers 1-3) are the fix.
    #   B) needs_more_demand (from exp)    — sim under-generates total volume;
    #                                        scaling_factor (Lever 4) is the fix.
    # When (B) is true but (A) is false, the demand-side answer is *more
    # agents at the same per-capita rate*, NOT *more trips per agent*. The
    # comment above already says this; the previous logic ran rate levers
    # anyway, producing tiny (+0.01) noise recommendations and risking
    # over-target trips/capita. We now split the paths cleanly.
    if tpc >= target_low and not needs_more_demand and not needs_less_demand:
        return recommendations  # demand genuinely looks OK (and no experiment contradicts it)

    # Population pieces used by both paths.
    pop_block = estimate["population"]
    commuting_employees = pop_block.get("commuting_employees", employees)
    effective_nonwork_pop = pop_block.get(
        "effective_nonwork_population", non_employees
    )
    current_share = estimate["nonwork"]["trip_share"]
    effective_share = current_share
    purposes = estimate["nonwork"]["purposes"]
    current_rates = {p: info["config_rate"] for p, info in purposes.items()}
    current_total_rate = sum(current_rates.values()) or 1.0
    work_trips = commuting_employees * avg_legs_work

    # Computed only on the rate-levers path; default no-op if skipped.
    rate_multiplier = 1.0

    # ─── Path A: trips/capita is too low — adjust rates ─────────────────
    if tpc < target_low:
        # --- Lever 1: nonwork_trip_share ---
        # No auto-push to a survey-derived "100% travel-day" value: the
        # trip table can't tell us about non-travel days. Keep config as-is.

        # --- Lever 2: back-calculate needed purpose rates ---
        # Use the survey target as-is — do NOT inflate trips/capita beyond
        # the survey benchmark. Use the post-WFH commuting-employee count
        # for work trips, and the effective nonwork pool for the nonwork
        # side, matching estimate_demand().
        target_trips = target * total_pop
        needed_nonwork_trips = max(target_trips - work_trips, 0.0)
        needed_nonwork_plans = needed_nonwork_trips / avg_legs_nonwork

        denom = effective_nonwork_pop * effective_share
        needed_total_rate = needed_nonwork_plans / denom if denom > 0 else 1.0
        rate_multiplier = needed_total_rate / current_total_rate

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
    # Two ways the sf can need to move:
    #   (a) needs_more_demand from the counts gate above: compensate for
    #       *measured* simulation losses (plan-gen + stuck) only. Capped
    #       hard at +0.03 absolute step and 0.13 absolute ceiling so we
    #       stay in the "~10% sample" band. This is intentionally too
    #       small to close a large count gap; large gaps are reported via
    #       _info.user_review_required and left to the user.
    #   (b) Rate-levers path (tpc too low): the rate bumps above raise
    #       trips/capita; if a residual gap remains, nudge sf up by the
    #       same small amount and capped the same way.
    #   (c) needs_less_demand from the over-production gate: counts say the sim
    #       is materially above observed AND the per-station spread is uniform,
    #       so a single global lever is valid. Unlike (a) this is a real
    #       correction rather than loss compensation, but it is held to the same
    #       step cap so no single noisy evaluation can move sf far.
    SF_STEP_MAX = 0.03   # max absolute step per iteration (either direction)
    SF_CEILING = 0.13    # max absolute sf value the estimator will write
    SF_FLOOR = 0.02      # min absolute sf value the estimator will write

    current_sf = estimate["scaling"]["scaling_factor"]
    # The ceiling must never drag a healthy larger sf downward: it caps
    # increases, so anchor it at or above the value already in use.
    sf_ceiling = max(SF_CEILING, current_sf)

    if needs_less_demand:
        # Direct correction: sf_new = sf * (1 / over_ratio) brings simulated
        # volumes to ~1.0x observed if the error really is global.
        proposed_sf = current_sf / over_ratio if over_ratio > 0 else current_sf
    elif needs_more_demand:
        # Compensation = inverse of the measured deployment yield.
        # plan_gen_ratio  = generated_plans / estimated_plans   (typically 0.7-1.0)
        # stuck_fraction  = stuck_agents   / simulated_persons   (typically 0-0.05)
        deployment_yield = max(plan_gen_ratio * (1.0 - stuck_fraction), 0.5)
        sf_compensation = 1.0 / deployment_yield
        proposed_sf = current_sf * sf_compensation
    else:
        # Rate-levers path: compute residual tpc gap after rate bumps.
        new_total_rate = sum(min(r * rate_multiplier, 0.50) for r in current_rates.values())
        projected_nonwork_plans = effective_nonwork_pop * new_total_rate * effective_share
        projected_nonwork_trips = projected_nonwork_plans * avg_legs_nonwork
        projected_total_trips = work_trips + projected_nonwork_trips
        projected_tpc = projected_total_trips / total_pop if total_pop > 0 else 0
        if projected_tpc < target_low and projected_tpc > 0:
            proposed_sf = current_sf * (target / projected_tpc)
        else:
            proposed_sf = current_sf

    # Apply the band symmetrically: never more than SF_STEP_MAX away from the
    # current value in either direction, and always within [SF_FLOOR, ceiling].
    capped_sf = min(proposed_sf, current_sf + SF_STEP_MAX, sf_ceiling)
    capped_sf = max(capped_sf, current_sf - SF_STEP_MAX, SF_FLOOR)
    new_sf = round(capped_sf, 3)

    if needs_less_demand and new_sf < current_sf - 0.001:
        recommendations.append({
            "parameter": "plan_generation.scaling_factor",
            "current": current_sf,
            "recommended": new_sf,
            "reason": (
                f"Counts-driven reduction: simulation is at {over_ratio:.0%} of "
                f"observed volumes with a uniform per-station spread, so the "
                f"excess is global. Full correction would be "
                f"{current_sf:.3f}/{over_ratio:.3f} = {proposed_sf:.3f}; capped to "
                f"{new_sf:.3f} (max step {SF_STEP_MAX}, floor {SF_FLOOR}). "
                f"flowCapacityFactor ({flow_cap}) and storageCapacityFactor "
                f"({storage_cap}) are UNCHANGED — lowering sf reduces demand, "
                f"which cannot gridlock a network sized for the larger sample. "
                f"countsScaleFactor will auto-adjust to 1/{new_sf} = {1/new_sf:.1f}. "
                f"Re-run and re-check: if the step cap kept this short of the full "
                f"correction, the next iteration continues closing the gap."
            ),
        })
    elif new_sf > current_sf + 0.001:
        if needs_more_demand:
            reason = (
                f"Compensation-only sf bump: deployment_yield = "
                f"plan_gen_ratio({plan_gen_ratio:.2f}) * (1 - stuck_fraction"
                f"({stuck_fraction:.3f})) = {plan_gen_ratio * (1 - stuck_fraction):.2f}; "
                f"sf_compensation = 1/{plan_gen_ratio * (1 - stuck_fraction):.2f} = "
                f"{1.0 / max(plan_gen_ratio * (1 - stuck_fraction), 0.5):.2f}x. "
                f"Proposed sf {current_sf:.3f} -> {proposed_sf:.3f}, capped to "
                f"{new_sf:.3f} (max step +{SF_STEP_MAX}, ceiling {SF_CEILING}). "
                f"flowCapacityFactor ({flow_cap}) and storageCapacityFactor "
                f"({storage_cap}) are UNCHANGED — that's by design: sf only "
                f"recovers measured losses, so capacity should still match the "
                f"original sample fraction. countsScaleFactor will auto-adjust "
                f"to 1/{new_sf} = {1/new_sf:.1f}. This bump does NOT try to "
                f"close any count gap beyond loss compensation; see "
                f"_info.user_review_required if iqr_mean is catastrophic."
            )
        else:
            reason = (
                f"Increase scaling to close residual trips/capita gap "
                f"(from {current_sf:.3f} to {new_sf:.3f}). Capped to "
                f"+{SF_STEP_MAX} per step, ceiling {SF_CEILING}. Capacity "
                f"factors unchanged. countsScaleFactor will auto-adjust to "
                f"1/{new_sf} = {1/new_sf:.1f}."
            )
        recommendations.append({
            "parameter": "plan_generation.scaling_factor",
            "current": current_sf,
            "recommended": new_sf,
            "reason": reason,
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

        _set_with_flat_leaf(new_config, path, value)
        # Add an estimator note next to the recommended value. The leaf key
        # is computed by the same flat-leaf rule so notes for flat-leaf
        # parents (e.g. matsim.configurable_params) sit alongside the dotted
        # key rather than collapsing into a nested sub-dict.
        parent, leaf = _split_for_flat_leaf(path)
        note_leaf = f"_estimator_{leaf}"
        if parent is None:
            new_config[note_leaf] = reason
        else:
            _set_with_flat_leaf(new_config, f"{parent}.{note_leaf}", reason)

    return new_config


# Parents below preserve dotted leaf keys instead of further nesting them.
# This matches how ``matsim.configurable_params`` is consumed by
# ``ConfigManager``: a flat dict whose keys are ``module.parameter`` strings
# (e.g. ``qsim.flowCapacityFactor`` or ``scoring.modeParams.pt.constant``).
# Nesting those into sub-dicts would silently break the loader.
_FLAT_LEAF_PARENTS = (
    "matsim.configurable_params",
)


def _split_for_flat_leaf(dotted_key: str) -> Tuple[str | None, str]:
    """Split a dotted path into (parent_path, leaf_key).

    For paths under a flat-leaf parent (see ``_FLAT_LEAF_PARENTS``), the
    leaf is everything after the parent prefix joined back with dots, so
    the caller can store it as a single dictionary key. For all other
    paths this is just ``rsplit('.', 1)``.
    """
    for parent in _FLAT_LEAF_PARENTS:
        prefix = parent + "."
        if dotted_key.startswith(prefix):
            return parent, dotted_key[len(prefix):]
    if "." in dotted_key:
        parent, leaf = dotted_key.rsplit(".", 1)
        return parent, leaf
    return None, dotted_key


def _set_with_flat_leaf(d: Dict, dotted_key: str, value: Any) -> None:
    """Like ``_set_nested`` but preserves dotted leaves under registered parents.

    For ``matsim.configurable_params.scoring.modeParams.pt.constant`` this
    walks to ``d['matsim']['configurable_params']`` and then sets the
    key ``'scoring.modeParams.pt.constant'`` to ``value`` — matching
    the flat-dict shape that ``ConfigManager`` expects.
    """
    parent, leaf = _split_for_flat_leaf(dotted_key)
    if parent is None:
        d[leaf] = value
        return
    cursor = d
    for segment in parent.split("."):
        if segment not in cursor or not isinstance(cursor[segment], dict):
            cursor[segment] = {}
        cursor = cursor[segment]
    cursor[leaf] = value


def _set_nested(d: Dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot-separated key path.

    Retained for any callers that explicitly want full nesting; the
    recommendation pipeline now uses :func:`_set_with_flat_leaf`.
    """
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
    print(f"      Source: blended TBI + NHTS trips / *person-days* "
          f"× nonwork_trip_share")
    print(f"    Target range:                    {target_low:.2f} - {target_high:.2f}")
    print(f"      Source: survey trips/capita ± 0.5 tolerance")
    print(f"    Avg legs per work chain:         {avg_legs_work:.2f}")
    print(f"      Source: survey Home->Work->Home chains")
    print(f"    Avg legs per nonwork chain:      {avg_legs_nonwork:.2f}")
    print(f"      Source: survey Home->...->Home (no Work) chains")
    print(f"    Travel-day participation rate:   {survey_travel_day_rate:.2%}")
    print(f"      Source: config nonwork_purposes.nonwork_trip_share "
          f"(non-travel days are not in the trip table — cannot derive directly)")

    # --- Section 2: Population ---
    pop = estimate["population"]
    print(f"\n  POPULATION ({pop['source']})")
    print(f"  {'─' * 60}")
    print(f"    Total population:   {pop['total']:>12,}")
    print(f"    Employees (LODES):  {pop['employees']:>12,}")
    print(f"    Non-employees:      {pop['non_employees']:>12,}")

    # WFH + worker-nonwork-share reallocation (Fix #2 + Fix #4)
    wfh_rate = pop.get("wfh_rate", 0.0)
    wfh_source = pop.get("wfh_source")
    alpha = pop.get("worker_nonwork_alpha", 0.0)
    wfh_emp = pop.get("wfh_employees", 0)
    commuting_emp = pop.get("commuting_employees", pop["employees"])
    workers_in_pool = pop.get("workers_in_nonwork_pool", 0)
    eff_nonwork_pop = pop.get("effective_nonwork_population", pop["non_employees"])
    if wfh_rate > 0 or alpha > 0:
        print(f"\n    Reallocation (Fix #2 WFH + Fix #4 worker-nonwork share):")
        if wfh_rate > 0:
            print(f"      WFH rate:                  {wfh_rate:.1%}  "
                  f"[{wfh_source or 'unknown source'}]")
            print(f"      WFH employees (no commute):    {wfh_emp:>12,.0f}")
        print(f"      Commuting employees:           {commuting_emp:>12,.0f}")
        print(f"      Worker-nonwork share (alpha):  {alpha:.2f}  "
              f"[config nonwork_purposes.worker_nonwork_share]")
        print(f"      Workers added to nonwork pool: {workers_in_pool:>12,.0f}")
        print(f"      Effective nonwork pool:        {eff_nonwork_pop:>12,.0f}  "
              f"(non_employees + WFH + alpha*commuting)")

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
    print(f"    Work plans (commuting employees):   {estimate['work']['plans_unscaled']:>12,.0f}")
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
        cr_src = tc.get("chain_reduction_source", {})
        print(f"    Chain consistency reduction:")
        print(f"      Bus:  ~{tc.get('chain_reduction_bus', tc['chain_reduction']):>5.0%} ({cr_src.get('bus', 'seed')})")
        print(f"      Rail: ~{tc.get('chain_reduction_rail', tc['chain_reduction']):>5.0%} ({cr_src.get('rail', 'seed')})")

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

        # Transit calibration intentionally does not recommend a scaling_factor;
        # see compute_transit_calibration docstring. Any sf change comes from
        # the counts-driven path in recommend_adjustments.

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
    acs_data: Dict[str, Dict[str, int]] | None = None,
) -> None:
    """Estimate and print projected demand with the new config."""
    new_estimate = estimate_demand(
        new_config, population_stats, avg_legs, survey_trips_per_capita,
        acs_data=acs_data,
    )
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
    parser.add_argument(
        "config_or_region",
        help="Cold start: path to config JSON (e.g. config/USA/TwinCities/config_twin.json). "
             "Feedback (with --experiment-dir): path to the region folder "
             "(e.g. config/USA/TwinCities) - the estimator reads "
             "<experiment-dir>/config_used.json as the state to update.",
    )
    parser.add_argument("--experiment-dir", type=str, default=None,
                        help="Path to a previous experiment folder. When given, the "
                             "positional arg must be a region folder; the estimator "
                             "reads <experiment-dir>/config_used.json and writes "
                             "<region_folder>/config_estimated.json.")
    args = parser.parse_args()

    read_from, output_path = resolve_estimator_inputs(
        args.config_or_region, args.experiment_dir
    )

    # Set up log file in logs/ directory
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"demand_estimator_{timestamp}.log"
    tee = TeeWriter(log_path)
    sys.stdout = tee

    # Load config from the resolved source (base config or experiment's
    # config_used.json). The write-back target lives in the region folder.
    with open(read_from, "r") as f:
        config = json.load(f)

    # _config_dir must point at the region folder (where the estimated config
    # will live), not at the experiment folder, so relative paths in the JSON
    # resolve against the region. config_used.json already has absolute paths.
    region_dir = output_path.parent
    config["_config_dir"] = str(region_dir)

    # Resolve relative data_dir to absolute (same logic as run_experiment.py)
    data_dir = config.get("data", {}).get("data_dir", "")
    if data_dir and not Path(data_dir).is_absolute():
        resolved = (region_dir / data_dir).resolve()
        config["data"]["data_dir"] = str(resolved)

    print(f"Loaded config: {read_from}")
    print(f"Output target: {output_path}")
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
    # ACS is required for both estimators; refuse to run without a key, and
    # treat an empty / low-coverage fetch as a hard error so the estimator
    # never writes config_estimated.json from bogus zero data.
    api_key = require_acs_key(config, read_from)
    acs_data: Dict[str, Dict[str, int]] = {}
    if counties:
        print()
        print("-" * 70)
        print("  STEP 2: FETCHING CENSUS ACS DATA (cross-check)")
        print("-" * 70)
        print(f"  [Source: Census ACS 5-year B08301 API, year={ACS_YEAR}]")
        print(f"  Fetching commute mode data for {len(counties)} counties...")
        acs_data = fetch_acs_commute_data(counties, api_key)
        coverage = len(acs_data) / len(counties) if counties else 0.0
        print(f"  Retrieved data for {len(acs_data)}/{len(counties)} counties "
              f"({coverage:.0%} coverage)")
        if coverage < 0.5:
            _print_acs_coverage_error(counties, list(acs_data.keys()))
            sys.exit(3)

    # --- Load previous experiment feedback up-front so Step 2b can use the
    # empirical chain_reduction. Display of the feedback summary still happens
    # under "STEP 3" below to keep the user-visible step order stable. ---
    experiment_feedback = None
    if args.experiment_dir:
        exp_dir = _resolve_experiment_dir(args.experiment_dir)
        summary_file = exp_dir / "experiment_summary.json"
        if summary_file.is_file():
            try:
                with open(summary_file, "r") as f:
                    summary = json.load(f)
                experiment_feedback = {
                    "summary": summary,
                    "evaluation": read_evaluation_metrics(exp_dir, summary),
                    "experiment_dir": str(exp_dir),
                }
            except (json.JSONDecodeError, OSError) as e:
                experiment_feedback = {"_load_error": str(e), "experiment_dir": str(exp_dir)}

    empirical_chain_reduction = compute_empirical_chain_reduction(experiment_feedback)

    # --- Step 2b: Transit calibration from ACS bus/rail breakdown ---
    # acs_data is guaranteed non-empty here: Step 2 exits on low coverage.
    transit_calibration = None
    if acs_data:
        print()
        print("-" * 70)
        print("  STEP 2b: TRANSIT CALIBRATION FROM ACS BUS/RAIL DATA")
        print("-" * 70)
        transit_calibration = compute_transit_calibration(
            acs_data, config,
            empirical_chain_reduction=empirical_chain_reduction,
        )
        if transit_calibration.get("acs_regional"):
            acs_r = transit_calibration["acs_regional"]
            print(f"  Regional transit share:  {acs_r['transit_share']:.1%}")
            print(f"    Bus:  {acs_r['bus_share']:.1%}  |  Rail: {acs_r['rail_share']:.1%}")
            src = transit_calibration.get("chain_reduction_source", {})
            cr_bus = transit_calibration.get("chain_reduction_bus")
            cr_rail = transit_calibration.get("chain_reduction_rail")
            if cr_bus is not None and cr_rail is not None:
                print(f"  Chain consistency reduction:")
                print(f"    Bus:  {cr_bus:.0%} ({src.get('bus', 'seed')})")
                print(f"    Rail: {cr_rail:.0%} ({src.get('rail', 'seed')})")
            print(f"  Transit calibration recommendations: "
                  f"{len(transit_calibration['recommendations'])}")
        else:
            print("  No transit data available for calibration")

    # --- Step 3: Display previous experiment feedback (loaded above) ---
    if args.experiment_dir:
        print()
        print("-" * 70)
        print("  STEP 3: LOADING PREVIOUS EXPERIMENT RESULTS")
        print("-" * 70)

        if experiment_feedback is None:
            print(f"  WARNING: No experiment_summary.json found in {args.experiment_dir}")
        elif "_load_error" in experiment_feedback:
            print(f"  WARNING: Failed to load experiment from "
                  f"{experiment_feedback['experiment_dir']}: {experiment_feedback['_load_error']}")
            experiment_feedback = None
        else:
            exp_name = Path(experiment_feedback["experiment_dir"]).name
            print(f"  Found experiment: {exp_name}")
            eval_data = experiment_feedback.get("evaluation")
            if eval_data:
                print(f"  Mean % error: {eval_data.get('mean_pct_error', 0):.1f}%")
                print(f"  GEH < 5:      {eval_data.get('geh_lt_5_pct', 0):.1f}%")
            else:
                print("  No evaluation data found")
            if empirical_chain_reduction:
                print(f"  Empirical chain_reduction recovered from prior run:")
                for mode_name, val in empirical_chain_reduction.items():
                    print(f"    {mode_name}: {val:.0%}")

    # --- Step 4: Estimate demand ---
    print()
    print("-" * 70)
    print("  STEP 4: ESTIMATING DEMAND FROM CONFIG PARAMETERS")
    print("-" * 70)
    estimate = estimate_demand(
        config, population_stats, avg_legs, survey_trips_per_capita,
        acs_data=acs_data,
    )

    # --- Step 5: Compute scorecard ---
    scorecard = compute_scorecard(estimate, acs_data, config)

    # --- Step 6: Generate recommendations ---
    recommendations = recommend_adjustments(
        estimate, scorecard, survey_travel_day_rate,
        experiment_feedback=experiment_feedback,
    )

    # Merge transit calibration recommendations.
    # Transit recs handle modes.bus.* and modes.rail.* parameters only;
    # demand recs handle nonwork_purposes.* and plan_generation.scaling_factor.
    # Since transit calibration no longer touches scaling_factor (see
    # compute_transit_calibration docstring), there is no sf conflict to
    # resolve — just append the transit recs.
    if transit_calibration and transit_calibration.get("recommendations"):
        recommendations.extend(transit_calibration["recommendations"])

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
        print_projected_demand(
            config, new_config, estimate, population_stats, avg_legs,
            survey_trips_per_capita, acs_data=acs_data,
        )

        # Write output config (path resolved up-front by resolve_estimator_inputs)
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
