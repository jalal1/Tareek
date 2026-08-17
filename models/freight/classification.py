"""Tier 3 — validating the time-of-day model against classification counts.

Tier 2 compares daily truck volumes and cannot say anything about *when* trucks
travel. Tier 3 can, but only where continuous-count sensors report FHWA vehicle
classes, and that coverage is genuinely partial: nationally about 49% of
station-directions report all 13 classes and a further 39% report usable bins,
but it runs from 100% in some states to zero in others (FL, LA, OK). So coverage
is **checked per region, never assumed** — that check is most of this module.

What we have on disk (`data/FHA_counts/`) is `.STA` station metadata and `.VOL`
hourly volumes. Classification lives in separate `.CLA` files which we do not
download. The `.STA` file does carry ``num_classes`` per station-direction,
which is exactly what decides whether tier 3 can run at all — so this module
answers that question offline, and reports what would be needed when the answer
is no.

**One caution that must not be misread.** Truck *percentage* peaks at night —
30-50% of rural traffic in the small hours — but that is because car volume
collapses, not because truck volume rises. Tier 3 therefore compares truck
*counts* per hour, never truck share per hour.

See docs/freight/design.md §5.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from models.freight.departure import HOURS_PER_DAY
from utils.logger import setup_logger

logger = setup_logger(__name__)

#: A station-direction reporting this many classes has full FHWA 1-13 detail.
FULL_CLASS_COUNT = 13

#: Fewer bins than this and the length-based groupings cannot be mapped onto
#: truck classes with any confidence.
MIN_USABLE_CLASSES = 4


@dataclass
class ClassificationCoverage:
    """Whether tier 3 can run for a region, and on how much of it."""
    n_stations: int = 0
    n_full: int = 0
    n_usable: int = 0
    n_volume_only: int = 0
    distribution: Dict[str, int] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.n_usable > 0

    @property
    def pct_full(self) -> float:
        return (self.n_full / self.n_stations * 100) if self.n_stations else 0.0

    def to_dict(self) -> Dict:
        return {
            'available': self.available,
            'n_stations': self.n_stations,
            'n_full_13_class': self.n_full,
            'n_usable': self.n_usable,
            'n_volume_only': self.n_volume_only,
            'pct_full': round(self.pct_full, 1),
            'num_classes_distribution': self.distribution,
        }


def check_coverage(
    station_zip: Path,
    state_abbr: str,
    county_codes: Iterable[str],
    year: int = 2024,
) -> ClassificationCoverage:
    """Ask whether a region's count stations report vehicle classes.

    Answered from the `.STA` file already on disk — no download. A region where
    this returns ``available=False`` cannot run tier 3 at all, and tiers 1 and 2
    have to stand on their own there.
    """
    station_zip = Path(station_zip)
    coverage = ClassificationCoverage()

    if not station_zip.exists():
        logger.warning(f"TMAS station file not found: {station_zip}")
        return coverage

    entry = f"{state_abbr}_{year} (TMAS).STA"
    counties = {str(c).strip().zfill(3) for c in county_codes}

    try:
        import pandas as pd
        with zipfile.ZipFile(station_zip) as zf:
            with zf.open(entry) as handle:
                frame = pd.read_csv(handle, sep='|', dtype=str,
                                    encoding='utf-8', on_bad_lines='skip')
    except Exception as exc:  # noqa: BLE001 - a coverage check must not fail a run
        logger.warning(f"Could not read TMAS station file: {exc}")
        return coverage

    if 'county_code' not in frame.columns or 'num_classes' not in frame.columns:
        logger.warning("TMAS station file lacks county_code/num_classes")
        return coverage

    regional = frame[frame['county_code'].str.strip().str.zfill(3).isin(counties)]
    if regional.empty:
        return coverage

    counts: Dict[str, int] = {}
    for raw in regional['num_classes']:
        key = str(raw).strip() or '00'
        counts[key] = counts.get(key, 0) + 1

    coverage.n_stations = len(regional)
    coverage.distribution = dict(sorted(counts.items()))
    for key, n in counts.items():
        try:
            n_classes = int(key)
        except ValueError:
            continue
        if n_classes >= FULL_CLASS_COUNT:
            coverage.n_full += n
            coverage.n_usable += n
        elif n_classes >= MIN_USABLE_CLASSES:
            coverage.n_usable += n
        else:
            coverage.n_volume_only += n

    logger.info(
        f"Classification coverage for {state_abbr} {sorted(counties)}: "
        f"{coverage.n_full}/{coverage.n_stations} station-directions report "
        f"{FULL_CLASS_COUNT} classes, {coverage.n_usable} usable"
    )
    if not coverage.available:
        logger.info(
            "Tier 3 is unavailable for this region: no station reports enough "
            "vehicle classes. Tiers 1 and 2 still stand."
        )
    return coverage


def compare_hourly_profile(
    simulated_hourly: Sequence[float],
    observed_hourly: Sequence[float],
) -> Dict:
    """Compare simulated and observed truck counts by hour.

    Both are **counts**, not shares, and both are normalised to a shape before
    comparison so a level difference (which tier 2 already measures) does not
    masquerade as a timing error. What tier 3 tests is *when*, not *how many*.
    """
    simulated = np.asarray(list(simulated_hourly), dtype=float)
    observed = np.asarray(list(observed_hourly), dtype=float)

    if simulated.shape != (HOURS_PER_DAY,) or observed.shape != (HOURS_PER_DAY,):
        raise ValueError(
            f"both profiles must have {HOURS_PER_DAY} hourly values, got "
            f"{simulated.size} and {observed.size}"
        )

    if simulated.sum() <= 0 or observed.sum() <= 0:
        return {'comparable': False,
                'note': 'one of the profiles carries no volume'}

    simulated_shape = simulated / simulated.sum()
    observed_shape = observed / observed.sum()
    deviation = np.abs(simulated_shape - observed_shape)

    # Correlation over the daily shape: catches a profile that peaks at the
    # wrong time even when the hourly deviations are individually small.
    correlation = float(np.corrcoef(simulated_shape, observed_shape)[0, 1])

    return {
        'comparable': True,
        'max_hourly_deviation': round(float(deviation.max()), 4),
        'mean_hourly_deviation': round(float(deviation.mean()), 4),
        'correlation': round(correlation, 4),
        'simulated_peak_hour': int(simulated_shape.argmax()),
        'observed_peak_hour': int(observed_shape.argmax()),
        'peak_hour_offset': int(abs(int(simulated_shape.argmax())
                                    - int(observed_shape.argmax()))),
        'simulated_shape': [round(float(v), 5) for v in simulated_shape],
        'observed_shape': [round(float(v), 5) for v in observed_shape],
    }


def validate_tier3(
    comparison: Dict,
    max_peak_offset: int = 2,
    min_correlation: float = 0.7,
) -> 'ValidationReport':
    """Does the simulated truck day match the observed one in shape?"""
    from models.freight.validation import CheckResult, ValidationReport

    report = ValidationReport(tier=3)

    if not comparison.get('comparable', False):
        report.add(CheckResult(
            name='classification_data_available',
            passed=False,
            detail=comparison.get('note', 'no classification counts for this region'),
        ))
        return report

    offset = comparison['peak_hour_offset']
    report.add(CheckResult(
        name='peak_hour_alignment',
        passed=offset <= max_peak_offset,
        detail=(f"simulated peak at hour {comparison['simulated_peak_hour']}, "
                f"observed at {comparison['observed_peak_hour']}"),
        value=float(offset),
        tolerance=float(max_peak_offset),
    ))

    correlation = comparison['correlation']
    report.add(CheckResult(
        name='daily_shape_correlation',
        passed=correlation >= min_correlation,
        detail=f"correlation between hourly shapes = {correlation:.3f}",
        value=correlation,
        tolerance=min_correlation,
    ))

    return report
