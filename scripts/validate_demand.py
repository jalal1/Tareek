"""Validate generated demand against the household travel survey.

This is the level of validation that traffic counts cannot provide. Count
stations measure vehicle crossings on a handful of major links — demand,
routing and capacity effects confounded — so they validate *assignment*, not
demand. The survey is the reference for demand itself: how many trips people
make, by what mode, how far, and when.

Standard practice compares four quantities, and this script reports exactly
those rather than inventing new ones:

  1. Trips per person per day   — is the right amount of travel generated?
  2. Mode share                 — is it on the right modes?
  3. Trip length distribution   — are the trips the right length?
  4. Departure-time profile     — do they happen at the right times?

Everything is weighted by the survey's own ``trip_weight`` so the survey side
is population-representative; the simulated side is a uniform sample, so its
shares need no weighting.

Two comparability rules, both easy to get wrong:

  * **Modes must be collapsed the same way.** MATSim reports Bus, Rail and
    School Bus all as ``pt``, so the survey's canonical modes are collapsed to
    the same vocabulary before comparing. Comparing raw labels would show a
    fake deficit in every transit sub-mode.
  * **Distances must share a unit and a definition.** The survey stores miles;
    MATSim reports metres of network distance travelled. Both are converted to
    km, and the *routed* distance is used on both sides where available.

Results go into the run's ``experiment_summary.json`` under
``demand_validation`` and into its report, so a run keeps one record rather
than scattering side files. The pipeline calls ``compute_demand_validation()``
automatically; this script is for inspecting or backfilling an existing run.

Usage:
    python scripts/validate_demand.py <experiment_dir>                  # print
    python scripts/validate_demand.py <experiment_dir> --update-summary # backfill
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

MILES_TO_KM = 1.609344

# Survey canonical modes -> the vocabulary MATSim actually writes. MATSim
# collapses every transit sub-mode to "pt", so the survey must be collapsed
# identically or transit looks under-served when it is merely aggregated.
SURVEY_TO_MATSIM_MODE = {
    'Car': 'car',
    'Rideshare': 'car',      # a car trip on the network
    'Bus': 'pt',
    'Rail': 'pt',
    'School Bus': 'pt',
    'Walk': 'walk',
    'Bike': 'bike',
    'Other': 'other',
}

# Time-of-day blocks, matching the evaluator's so count-side and demand-side
# temporal findings can be read against each other.
TIME_BLOCKS = (('night', 0, 3), ('morning', 4, 9),
               ('midday', 10, 17), ('evening', 18, 23))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

# The national survey. It describes the whole country, so it is a valid
# reference for any region. Every other survey in survey_trips is a
# metro-specific household travel survey belonging to ONE region.
NATIONAL_SOURCE = 'nhts'


def configured_surveys(exp_dir: Path) -> "Optional[List[str]]":
    """Survey types this run actually declared, lower-cased.

    The ``survey_trips`` table records only ``source_type``/``source_year`` —
    it carries no region — so nothing in the database says which metro a local
    survey belongs to. The run's own ``data.surveys`` config is the only
    statement of that, which is why it is the filter: a survey loaded for
    another region is not evidence about this one.

    ``weight > 0`` is the project-wide "this survey is active" convention
    (see data_sources/survey_manager.py), so it is honoured here too.

    Returns None when the config cannot be read, meaning "no filter" — the
    caller then falls back to the old behaviour rather than validating against
    nothing.
    """
    cfg_path = exp_dir / "config_used.json"
    if not cfg_path.is_file():
        return None
    try:
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return None
    entries = (cfg.get("data") or {}).get("surveys")
    if not isinstance(entries, list):
        return None
    return [str(e.get("type", "")).lower() for e in entries
            if isinstance(e, dict) and (e.get("weight", 1.0) or 0) > 0]


def load_surveys(data_dir: str,
                 allowed: "Optional[List[str]]" = None) -> "Dict[str, pd.DataFrame]":
    """Load the surveys this run declared, keyed by source type.

    *allowed* is the run's configured survey list. Surveys outside it are
    dropped: the database is shared across regions, so without this filter a
    local survey loaded for one metro (Twin Cities' TBI) is silently used as
    the authoritative reference for another (Birmingham), comparing a model
    against a population it never represented.

    When both a national and this region's local survey are present, both are
    returned and the report compares against each. They measure the same
    quantities on different populations, so the gap between them is itself
    informative: it says how much this metro departs from national travel
    behaviour, and whether a model/survey mismatch is a modelling problem or
    just the wrong reference population.
    """
    from models.models import initialize_tables

    db = initialize_tables(data_dir)
    try:
        with db.session_scope() as session:
            df = pd.read_sql("SELECT * FROM survey_trips", session.bind)
    finally:
        db.close()

    if df.empty:
        raise SystemExit("No survey trips in the database — nothing to validate against.")

    logger.info("Survey sources in DB: %s",
                dict(df.groupby(['source_type', 'source_year']).size()))

    out: Dict[str, pd.DataFrame] = {}
    for source_type, group in df.groupby('source_type'):
        name = str(source_type)
        if allowed is not None and name.lower() not in allowed:
            logger.info("Ignoring survey '%s': not declared by this run's config. "
                        "It belongs to a different region, so it is not a valid "
                        "reference here.", name)
            continue
        out[name] = group

    if not out:
        raise SystemExit(
            "None of this run's configured surveys are loaded in the database "
            f"(configured: {allowed}). Load one, or add an entry to "
            "config data.surveys with weight > 0.")

    if not any(is_local(k) for k in out):
        logger.info("No local survey for this region — comparing against national %s "
                    "only. A metro-specific household travel survey is the better "
                    "reference; add one to config data.surveys with weight > 0.",
                    NATIONAL_SOURCE.upper())
    return out


def is_local(source_type: str) -> bool:
    """Is this a metro-specific survey rather than the national one?

    Only meaningful once the caller has already restricted the surveys to the
    ones this run configured — see load_surveys. A local survey that belongs to
    a *different* region must be filtered out there, because this test cannot
    tell regions apart: the survey table has no region column.
    """
    return str(source_type).lower() != NATIONAL_SOURCE


def load_simulated_trips(exp_dir: Path) -> pd.DataFrame:
    """Load MATSim's per-trip output from the final iteration.

    ``*.trips.csv.gz`` is one row per completed trip with the routed distance,
    main mode and departure time — the simulated counterpart of a survey trip
    record.
    """
    iters = exp_dir / "output" / "ITERS"
    candidates = sorted(iters.glob("it.*/*.trips.csv.gz")) if iters.exists() else []
    if not candidates:
        candidates = sorted(exp_dir.glob("**/*.trips.csv.gz"))
    if not candidates:
        raise SystemExit(f"No *.trips.csv.gz under {exp_dir}")

    def iter_num(p: Path) -> int:
        try:
            return int(p.parent.name.split(".")[-1])
        except (ValueError, IndexError):
            return -1

    path = max(candidates, key=iter_num)
    logger.info("Simulated trips: %s", path)
    with gzip.open(path, 'rt') as fh:
        df = pd.read_csv(fh, sep=';', low_memory=False)
    return df


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _hour_from_time(series: pd.Series) -> pd.Series:
    """Hour of day from either a datetime column or an HH:MM:SS string."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.hour
    parsed = pd.to_datetime(series, errors='coerce')
    if parsed.notna().any():
        return parsed.dt.hour
    return pd.to_numeric(series.astype(str).str.split(':').str[0], errors='coerce')


def normalise_survey(df: pd.DataFrame) -> pd.DataFrame:
    """Survey trips -> (mode, distance_km, hour, weight, purpose)."""
    out = pd.DataFrame(index=df.index)
    out['mode'] = df['mode_type'].map(SURVEY_TO_MATSIM_MODE).fillna('other')
    out['distance_km'] = pd.to_numeric(df['distance_miles'], errors='coerce') * MILES_TO_KM
    out['hour'] = _hour_from_time(df['depart_time'])
    # Unweighted rows would let an over-sampled group dominate; the survey's own
    # expansion weight is what makes the reference population-representative.
    out['weight'] = pd.to_numeric(df['trip_weight'], errors='coerce').fillna(0.0)
    out['person'] = df['person_id']
    out['purpose'] = df['destination_purpose']
    # The travel DATE, needed because surveys differ in how many days they
    # observe each person. NHTS records a single travel day; TBI follows each
    # person for several. Dividing trips by persons would then make a
    # multi-day survey look several times more travel-intensive than a
    # single-day one, so the denominator must be person-DAYS.
    depart = df['depart_time']
    if not pd.api.types.is_datetime64_any_dtype(depart):
        depart = pd.to_datetime(depart, errors='coerce')
    out['date'] = depart.dt.normalize()
    return out


def normalise_simulated(df: pd.DataFrame) -> pd.DataFrame:
    """MATSim trips -> the same columns, weight 1 (uniform sample)."""
    out = pd.DataFrame(index=df.index)
    out['mode'] = df['main_mode'].astype(str)
    out['distance_km'] = pd.to_numeric(df['traveled_distance'], errors='coerce') / 1000.0
    out['hour'] = _hour_from_time(df['dep_time'])
    out['weight'] = 1.0
    out['person'] = df['person']
    out['purpose'] = df.get('end_activity_type')
    return out


# ---------------------------------------------------------------------------
# The four standard comparisons
# ---------------------------------------------------------------------------

def _wmean(values: pd.Series, weights: pd.Series) -> float:
    m = values.notna() & weights.notna() & (weights > 0)
    if not m.any():
        return float('nan')
    return float(np.average(values[m], weights=weights[m]))


def _wquantile(values: pd.Series, weights: pd.Series, q: float) -> float:
    """Weighted quantile — needed because survey rows carry expansion weights."""
    m = values.notna() & weights.notna() & (weights > 0)
    if not m.any():
        return float('nan')
    v = values[m].to_numpy(dtype=float)
    w = weights[m].to_numpy(dtype=float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w) - 0.5 * w
    cum /= w.sum()
    return float(np.interp(q, cum, v))


def trips_per_person(survey: pd.DataFrame, sim: pd.DataFrame) -> Dict[str, Any]:
    """Trips per person per day — the headline demand-generation number."""
    # Denominator is person-DAYS, not persons: surveys differ in how many days
    # they observe each respondent (NHTS one, TBI several), and per-person
    # would make a multi-day survey look proportionally more travel-intensive.
    # Each person-day's weight is taken once — its trips share it — otherwise a
    # person with many trips would be counted many times in the denominator.
    if 'date' in survey.columns and survey['date'].notna().any():
        key = survey['person'].astype(str) + '|' + survey['date'].astype(str)
        n_days = survey.groupby('person')['date'].nunique()
        days_per_person = float(n_days.mean()) if len(n_days) else 1.0
    else:
        key = survey['person'].astype(str)
        days_per_person = 1.0
    unit_w = survey.groupby(key)['weight'].first()
    survey_trips = survey['weight'].sum()
    survey_units = unit_w.sum()

    # The simulated population is one modelled day, so its person count is
    # already a person-day count.
    sim_trips = len(sim)
    sim_persons = sim['person'].nunique()
    return {
        'survey_trips_per_person': round(survey_trips / survey_units, 3) if survey_units else None,
        'simulated_trips_per_person': round(sim_trips / sim_persons, 3) if sim_persons else None,
        'survey_person_days': int(survey_units),
        'survey_days_per_person': round(days_per_person, 2),
        'simulated_persons': int(sim_persons),
        'simulated_trips': int(sim_trips),
    }


def mode_share(survey: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    """Weighted mode share, survey vs simulated, on MATSim's vocabulary."""
    s = survey.groupby('mode')['weight'].sum()
    s = (s / s.sum() * 100).rename('survey_pct')
    m = sim.groupby('mode').size()
    m = (m / m.sum() * 100).rename('simulated_pct')
    out = pd.concat([s, m], axis=1).fillna(0.0)
    out['diff_pp'] = out['simulated_pct'] - out['survey_pct']
    return out.sort_values('survey_pct', ascending=False).round(2)


def trip_length(survey: pd.DataFrame, sim: pd.DataFrame) -> Dict[str, Any]:
    """Trip length distribution — mean, median and quartiles, in km."""
    out: Dict[str, Any] = {}
    for name, df in (('survey', survey), ('simulated', sim)):
        d, w = df['distance_km'], df['weight']
        out[f'{name}_mean_km'] = round(_wmean(d, w), 2)
        out[f'{name}_median_km'] = round(_wquantile(d, w, 0.50), 2)
        out[f'{name}_p25_km'] = round(_wquantile(d, w, 0.25), 2)
        out[f'{name}_p75_km'] = round(_wquantile(d, w, 0.75), 2)
    for k in ('mean_km', 'median_km'):
        s, m = out.get(f'survey_{k}'), out.get(f'simulated_{k}')
        if s:
            out[f'{k}_ratio'] = round(m / s, 3)
    return out


def departure_profile(survey: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    """Share of departures by hour, and by time block."""
    s = survey.groupby('hour')['weight'].sum()
    s = (s / s.sum() * 100)
    m = sim.groupby('hour').size()
    m = (m / m.sum() * 100)
    out = pd.DataFrame({'survey_pct': s, 'simulated_pct': m}).reindex(range(24)).fillna(0.0)
    out['diff_pp'] = out['simulated_pct'] - out['survey_pct']
    return out.round(2)


def block_shares(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, lo, hi in TIME_BLOCKS:
        blk = profile.loc[lo:hi]
        rows.append({'block': name, 'hours': f'{lo}-{hi}',
                     'survey_pct': round(blk['survey_pct'].sum(), 2),
                     'simulated_pct': round(blk['simulated_pct'].sum(), 2)})
    out = pd.DataFrame(rows)
    out['diff_pp'] = (out['simulated_pct'] - out['survey_pct']).round(2)
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _md_table(df: pd.DataFrame, index_name: str = "") -> str:
    """Render a DataFrame as a markdown table.

    Written by hand rather than via ``DataFrame.to_markdown`` so the script has
    no dependency on ``tabulate``, which is not installed in the pipeline venv.
    """
    cols = list(df.columns)
    head = "| " + " | ".join([index_name] + [str(c) for c in cols]) + " |"
    divide = "|---" + "|---:" * len(cols) + "|"
    rows = [head, divide]
    for idx, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:,.2f}" if isinstance(v, float) else f"{v}")
        rows.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(rows)


def build_report(results: Dict[str, Any], exp_name: str) -> str:
    """Console/markdown report: one block per loaded survey."""
    L = [f"# Demand validation — {exp_name}", "",
         "Generated demand compared against the household travel survey. This",
         "validates **demand**; traffic counts validate assignment given demand.",
         ""]
    surveys = results['surveys']
    if len(surveys) > 1:
        L += ["Compared against **all loaded surveys**. A local household travel",
              "survey is the authoritative reference for its own metro; the",
              "national survey is shown alongside because the gap between them",
              "is a property of the metro, not a model error.", ""]

    for name, e in surveys.items():
        kind = "LOCAL" if e['is_local'] else "national"
        L += [f"## {e['label']}  ({kind}, {e['n_trips']:,} trips)", ""]
        tpp, tl = e['trips_per_person'], e['trip_length']
        s, m = tpp['survey_trips_per_person'], tpp['simulated_trips_per_person']
        ratio = round(m / s, 3) if s else None
        L += ["### Trips per person per day", "",
              "| | survey | simulated | ratio |", "|---|---:|---:|---:|",
              f"| trips/person/day | {s} | {m} | **{ratio}** |", "",
              "### Mode share", ""]
        L.append(_md_table(e['mode_share'], 'mode'))
        L += ["", "### Trip length distribution", "",
              "| statistic | survey | simulated | ratio |", "|---|---:|---:|---:|",
              f"| median km | {tl['survey_median_km']} | {tl['simulated_median_km']} | "
              f"**{tl.get('median_km_ratio')}** |",
              f"| mean km | {tl['survey_mean_km']} | {tl['simulated_mean_km']} | "
              f"**{tl.get('mean_km_ratio')}** |",
              f"| p25 km | {tl['survey_p25_km']} | {tl['simulated_p25_km']} | |",
              f"| p75 km | {tl['survey_p75_km']} | {tl['simulated_p75_km']} | |", "",
              "### Departure-time profile (by block)", ""]
        L.append(_md_table(e['blocks'].set_index('block'), 'block'))
        L.append("")

    L += ["> The **ratio** is the number that matters: it compares the model",
          "> against a survey's own population, so it needs no external",
          "> benchmark. Absolute levels vary with how each survey defines and",
          "> counts a trip, so figures from different surveys are not",
          "> interchangeable — which is exactly why both are shown separately",
          "> rather than merged."]
    return "\n".join(L)


def compute_demand_validation(exp_dir: Path,
                              data_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Run the four comparisons and return a summary-ready dict.

    This is the entry point the pipeline calls, so demand validation lands in
    ``experiment_summary.json`` alongside the count validation rather than in a
    separate file — one run, one record.

    Returns ``None`` (with a warning) when the inputs are missing, so a run
    without survey data or without MATSim trip output still completes.
    """
    if data_dir is None:
        cfg_path = exp_dir / "config_used.json"
        if cfg_path.is_file():
            with open(cfg_path, encoding="utf-8") as fh:
                data_dir = json.load(fh).get("data", {}).get("data_dir", "data")
        else:
            data_dir = "data"

    try:
        # Restricted to the surveys this run declared. The survey table is
        # shared across regions and carries no region column, so without this
        # another metro's local survey would be picked up as the reference.
        surveys_raw = load_surveys(data_dir, configured_surveys(exp_dir))
        sim_raw = load_simulated_trips(exp_dir)
    except SystemExit as exc:
        logger.warning("Demand validation skipped: %s", exc)
        return None
    except Exception as exc:
        logger.warning("Demand validation failed: %s", exc)
        return None

    sim = normalise_simulated(sim_raw)

    # One comparison per survey. Local surveys are listed first: for its own
    # metro a local household travel survey is the authoritative reference,
    # and the national one is context.
    per_survey: Dict[str, Any] = {}
    for source_type in sorted(surveys_raw, key=lambda s: (not is_local(s), s)):
        raw = surveys_raw[source_type]
        survey = normalise_survey(raw)
        profile = departure_profile(survey, sim)
        per_survey[source_type] = {
            'label': f"{source_type} {raw['source_year'].iloc[0]}",
            'is_local': is_local(source_type),
            'n_trips': int(len(raw)),
            'trips_per_person': trips_per_person(survey, sim),
            'mode_share': mode_share(survey, sim),
            'trip_length': trip_length(survey, sim),
            'profile': profile,
            'blocks': block_shares(profile),
        }

    primary = next(iter(per_survey))
    return {
        'surveys': per_survey,
        # The reference the verdict is drawn from: local when available.
        'primary': primary,
        'has_local': any(v['is_local'] for v in per_survey.values()),
    }


def _one_survey_section(entry: Dict[str, Any]) -> Dict[str, Any]:
    """The per-survey block of the summary section."""
    tpp = entry['trips_per_person']
    tl = entry['trip_length']
    s, m = tpp['survey_trips_per_person'], tpp['simulated_trips_per_person']
    return {
        'label': entry['label'],
        'is_local': entry['is_local'],
        'survey_trips': entry['n_trips'],
        'trips_per_person_survey': s,
        'trips_per_person_simulated': m,
        'trips_per_person_ratio': round(m / s, 3) if s else None,
        'survey_days_per_person': tpp.get('survey_days_per_person'),
        'mode_share': entry['mode_share'].to_dict(orient='index'),
        **{k: v for k, v in tl.items()},
        'departure_blocks': entry['blocks'].to_dict(orient='records'),
    }


def to_summary_section(results: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the results into the JSON-serialisable summary section.

    Every loaded survey gets its own block under ``surveys``. When a metro has
    both a local household travel survey and the national one, both are kept:
    the local survey is the authoritative reference for that metro, and the
    national one shows how far the metro departs from national behaviour.
    """
    surveys = {name: _one_survey_section(entry)
               for name, entry in results['surveys'].items()}
    return {
        '_comment': 'Generated demand vs the household travel survey(s). This validates '
                    'DEMAND (how many trips, by what mode, how far, when); the '
                    'evaluation section validates ASSIGNMENT of that demand onto the '
                    'network. Count stations cannot measure demand: they sit on a few '
                    'major links and mix demand with routing and capacity effects.',
        'primary_survey': results['primary'],
        'primary_survey_comment': 'The survey the verdict is drawn from. A metro-specific '
                                  '(local) household travel survey wins when one is '
                                  'loaded; otherwise the national NHTS is used. Where both '
                                  'exist, differences between them are a property of the '
                                  'metro, not a model error.',
        'has_local_survey': results['has_local'],
        'surveys': surveys,
        'surveys_comment': 'One block per loaded survey, keyed by source type. '
                           'trips_per_person_ratio is the headline trip-generation check '
                           '(simulated / survey); a ratio well below 1.0 means demand is '
                           'short of trips, which no assignment change can fix. mode_share '
                           'is on MATSim\'s vocabulary — every transit sub-mode is "pt" on '
                           'BOTH sides because MATSim writes only "pt". Trip lengths are km; '
                           'compare median before mean, since a high mean with a matching '
                           'median means the long tail is too heavy rather than every trip '
                           'being too long. departure_blocks should be read against the '
                           'evaluation section\'s ratio_* fields: departures matching the '
                           'survey while hourly counts do not points at routing or capacity.',
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment_dir", type=Path)
    ap.add_argument("--data-dir", default=None,
                    help="override the data dir holding the DB")
    ap.add_argument("--update-summary", action="store_true",
                    help="write the results into the run's experiment_summary.json "
                         "(the pipeline does this automatically for new runs; use this "
                         "to backfill an older run).")
    args = ap.parse_args()

    exp_dir = args.experiment_dir
    if not exp_dir.is_dir():
        raise SystemExit(f"Not a directory: {exp_dir}")

    results = compute_demand_validation(exp_dir, args.data_dir)
    if results is None:
        raise SystemExit("Could not compute demand validation.")

    print(build_report(results, exp_dir.name))

    if args.update_summary:
        summary_path = exp_dir / "experiment_summary.json"
        if not summary_path.is_file():
            raise SystemExit(f"No experiment_summary.json in {exp_dir}")
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)
        summary['demand_validation'] = to_summary_section(results)
        with open(summary_path, 'w', encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nUpdated: {summary_path}")


if __name__ == "__main__":
    main()
