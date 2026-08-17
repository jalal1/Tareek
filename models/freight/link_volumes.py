"""The compact truck-stream artefact: ``freight_link_volumes.json``.

**Why this file exists at all.** Tier-2 validation needs simulated truck volumes
per link, and no MATSim output carries them. Verified against the pipeline and
the MATSim outputs a run actually produces:

===========================  ===========================================
output                       what it carries
===========================  ===========================================
``linkstats.txt.gz``         ``link_id`` + 24 hourly totals. **No vehicle
                             identity** — see ``evaluator.load_linkstats``.
``countscompare.txt``        one row per count station, total volume only.
``output_trips.csv``         person id, so a subpopulation *is* joinable —
                             but origin/destination only, **no link path**.
``output_vehicles.xml.gz``   vehicle to type. A roster, no volumes.
``scorestats_freight.csv``   subpopulation scores. No volumes.
===========================  ===========================================

So the precise statement is not "MATSim hides vehicle identity" — ``output_trips``
exposes it — but that **no MATSim output gives per-link volume broken down by
vehicle type or subpopulation**. ``linkStats`` is written by ``VolumesAnalyzer``,
which consumes the same link-enter events this module reads and discards vehicle
identity before writing. The split is thrown away upstream of the file, not
absent from the simulation. Recovering it in MATSim would mean a custom Java
``EventHandler``; recovering it here means re-reading the events.

**What this module is for.** The events file is 2.1 GB at 15-county scale and
takes ~20 minutes to parse. That parse belongs to the estimator, not to every
simulation run — but it should happen at most *once* per experiment. So the
first estimator run extracts the truck stream into this small JSON, and every
subsequent run reads the JSON in milliseconds.

The artefact is deliberately small: per-link daily totals for the freight and
total streams, the 24-hour freight profile, and the provenance needed to know
whether the numbers can be compared against observed AADT at all. Hourly detail
for the *total* stream is dropped — ``linkstats`` already carries it, and
keeping it here would multiply the file size for no reader.

See docs/freight/design.md §5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from models.freight.events import LinkVolumes, extract_freight_and_total
from utils.logger import setup_logger

logger = setup_logger(__name__)

#: Written next to the other per-experiment freight outputs.
ARTEFACT_NAME = 'freight_link_volumes.json'

#: Bumped when the payload shape changes in a way a reader must notice. A stale
#: artefact is then re-extracted rather than silently misread.
ARTEFACT_VERSION = 1


def artefact_path(experiment_dir: Path) -> Path:
    return Path(experiment_dir) / ARTEFACT_NAME


def build_artefact(
    streams: Dict[str, LinkVolumes],
    *,
    scale_factor: float,
    events_path: Optional[Path] = None,
    plans_path: Optional[Path] = None,
) -> Dict:
    """Turn the extracted streams into the serialisable artefact.

    Split out from :func:`extract_link_volumes_artefact` so it can be tested
    without an events file.
    """
    freight = streams['freight']
    total = streams['total']

    links: Dict[str, Dict] = {}
    for link_id in set(freight.volumes) | set(freight.overflow):
        daily = freight.daily(link_id)
        if daily <= 0:
            continue
        links[link_id] = {
            'freight': round(daily, 3),
            'total': round(total.daily(link_id), 3),
            # The hourly freight profile is what tier 3 compares; rounding to
            # 3dp keeps the file small without losing a scaled count.
            'freight_hourly': [round(float(v), 3) for v in freight.hourly(link_id)],
        }

    # Links with no freight still matter to tier 2: an HPMS segment that carries
    # observed trucks but no simulated ones is the coverage failure the design
    # calls out, and it is invisible if only non-zero links are stored. They are
    # recorded as a bare total so the reader can tell "no trucks here" from
    # "this link was never simulated".
    freight_total_daily = freight.total()
    n_links_with_freight = len(links)

    return {
        'version': ARTEFACT_VERSION,
        'scale_factor': scale_factor,
        'source': {
            'events': str(events_path) if events_path else None,
            'plans': str(plans_path) if plans_path else None,
        },
        'totals': {
            'freight': round(freight_total_daily, 1),
            'car': round(streams['car'].total(), 1),
            'all': round(total.total(), 1),
            'freight_share_pct': (round(freight_total_daily / total.total() * 100, 3)
                                  if total.total() > 0 else None),
            'freight_vehicles': freight.n_vehicles,
            'freight_events': freight.n_events,
        },
        'n_links_with_freight': n_links_with_freight,
        'n_links_simulated': len(total.volumes),
        'links': links,
    }


def extract_link_volumes_artefact(
    experiment_dir: Path,
    *,
    events_path: Optional[Path] = None,
    plans_path: Optional[Path] = None,
    scale_factor: float = 1.0,
    subpopulation: str = 'freight',
    force: bool = False,
) -> Dict:
    """Read the events file once and write the compact artefact.

    Returns the artefact payload. If one already exists and ``force`` is false,
    it is read from disk instead — the whole point of the file is that the
    20-minute parse happens once.

    Args:
        experiment_dir: where the artefact is written.
        events_path: defaults to ``<dir>/output/output_events.xml.gz``.
        plans_path: defaults to ``<dir>/plans.xml``.
        scale_factor: normally ``1 / flowCapacityFactor``, matching MATSim's
            ``countsScaleFactor``. Comparing an unscaled sample against observed
            AADT would understate by exactly this factor.
        force: re-parse even when an artefact is present.
    """
    experiment_dir = Path(experiment_dir)
    out = artefact_path(experiment_dir)

    if out.exists() and not force:
        try:
            cached = json.loads(out.read_text(encoding='utf-8'))
            if cached.get('version') == ARTEFACT_VERSION:
                logger.info(
                    f"Reusing {ARTEFACT_NAME} ({cached.get('n_links_with_freight', 0):,} "
                    f"links with freight) — delete it or pass force to re-parse."
                )
                return cached
            logger.info(
                f"{ARTEFACT_NAME} is version {cached.get('version')}, expected "
                f"{ARTEFACT_VERSION}; re-extracting from the events file."
            )
        except (ValueError, OSError) as exc:
            logger.warning(f"{ARTEFACT_NAME} unreadable ({exc}); re-extracting.")

    events = Path(events_path) if events_path else (
        experiment_dir / 'output' / 'output_events.xml.gz')
    plans = Path(plans_path) if plans_path else (experiment_dir / 'plans.xml')

    logger.info(
        f"Extracting the truck stream from {events.name}. This is the slow step "
        f"— budget ~20 min for a 2 GB events file — and it runs once per "
        f"experiment."
    )
    streams = extract_freight_and_total(
        events, plans, subpopulation=subpopulation, scale_factor=scale_factor)

    payload = build_artefact(streams, scale_factor=scale_factor,
                             events_path=events, plans_path=plans)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    logger.info(
        f"Wrote {out} — {payload['n_links_with_freight']:,} links carry "
        f"freight, {payload['totals']['freight_share_pct']}% of link entries."
    )
    return payload


def load_artefact(experiment_dir: Path) -> Optional[Dict]:
    """Read the artefact if it is present and current, else None."""
    out = artefact_path(experiment_dir)
    if not out.exists():
        return None
    try:
        payload = json.loads(out.read_text(encoding='utf-8'))
    except (ValueError, OSError):
        return None
    return payload if payload.get('version') == ARTEFACT_VERSION else None


def artefact_to_link_volumes(payload: Dict) -> LinkVolumes:
    """Rebuild a :class:`LinkVolumes` from the artefact.

    Lets everything downstream — ``compare_by_segment``, ``compare_against_observed``
    — take the artefact and the freshly-parsed events interchangeably, so the
    estimator has one code path rather than two.

    Volumes in the artefact are **already scaled**, so ``scale_factor`` is
    carried for provenance and must not be applied a second time.
    """
    import numpy as np

    volumes: Dict[str, 'np.ndarray'] = {}
    overflow: Dict[str, float] = {}

    for link_id, record in (payload.get('links') or {}).items():
        hourly = record.get('freight_hourly')
        if hourly:
            series = np.array(hourly, dtype=float)
        else:
            series = np.zeros(24)
        volumes[link_id] = series
        # daily() adds overflow to the hourly sum, so the residual between the
        # stored daily total and the profile is exactly the post-midnight
        # traffic. Recovering it here keeps daily totals identical across a
        # re-read, which is what makes the artefact a faithful substitute.
        residual = float(record.get('freight', 0.0)) - float(series.sum())
        if residual > 1e-9:
            overflow[link_id] = residual

    totals = payload.get('totals') or {}
    return LinkVolumes(
        volumes=volumes,
        n_vehicles=int(totals.get('freight_vehicles') or 0),
        n_events=int(totals.get('freight_events') or 0),
        scale_factor=float(payload.get('scale_factor') or 1.0),
        overflow=overflow,
    )
