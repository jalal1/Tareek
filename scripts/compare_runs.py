#!/usr/bin/env python3
"""Compare two or more experiments on their count-validation metrics.

Reads each run's ``experiment_summary.json`` (the single source of truth) and
prints a side-by-side table, so before/after comparisons don't have to be
assembled by hand.

The metrics shown are the four that decide whether a model is working, in the
order they should be read:

  station_ratio_cv   is the error uniform or concentrated?  Read this FIRST:
                     it decides whether a global lever (scaling_factor) can
                     fix anything at all. Below ~0.35 the error is global.
  geh_lt_5_pct       % of hourly link counts scoring GEH < 5, the standard
                     counts criterion. Higher is better; >=85% is the target.
  correlation        spatial pattern, on station-hours.
  interquartile_mean_ratio
                     volume level, robust to a few odd stations. 1.0 = right.

``aggregate`` (total simulated / total observed) is shown too, but it is the
easiest number to be fooled by: a model can score 1.0 by over-simulating one
corridor and under-simulating another. Judge it together with the CV.

Usage:
    python scripts/compare_runs.py experiments/exp_A experiments/exp_B
    python scripts/compare_runs.py --labels "drop,anchor" exp_A exp_B
    python scripts/compare_runs.py --csv out.csv exp_A exp_B exp_C
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


# (key, label, decimals, direction) — direction says which way is better:
# "down" lower is better, "up" higher is better, "one" closer to 1.0 is better,
# None means no verdict (context only).
METRICS: List[Tuple[str, str, int, Optional[str]]] = [
    ("station_ratio_cv", "per-station ratio CV", 3, "down"),
    # No mean GEH: a GEH is defined for one count over one period and grows
    # with volume, so averaging across stations of different sizes measures
    # station size as much as model error. The pass rates below test each
    # count against the threshold on its own.
    ("geh_lt_5_pct", "% hourly counts GEH < 5", 1, "up"),
    ("station_daily_geh_lt5_pct", "% stations daily GEH < 5", 1, "up"),
    ("correlation", "correlation (station-hour)", 3, "up"),
    ("interquartile_mean_ratio", "iqr_mean (volume level)", 3, "one"),
    ("aggregate_ratio", "aggregate sim/obs", 3, "one"),
    ("station_ratio_p10", "ratio p10", 3, None),
    ("station_ratio_p90", "ratio p90", 3, None),
    ("num_stations_below_10pct", "stations under 10%", 0, "down"),
    ("num_devices", "physical stations", 0, None),
    ("num_directional_counts", "station-directions", 0, None),
]

# Config / run facts worth seeing beside the metrics, since they usually
# explain the differences. Third element is the decimal places to show.
CONTEXT: List[Tuple[List[str], str, int]] = [
    (["parameters", "scaling_factor"], "scaling_factor", 3),
    (["parameters", "flow_capacity_factor"], "flow_capacity_factor", 3),
    (["parameters", "iterations"], "iterations", 0),
    (["matsim_output", "output_persons_count"], "agents simulated", 0),
    (["matsim_output", "total_stuck_agents"], "stuck agents", 0),
]


def _dig(d: Any, path: List[str]) -> Any:
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def load_run(exp_dir: Path) -> Dict[str, Any]:
    """Metrics + context for one experiment.

    Evaluation metrics come from experiment_summary.json's ``evaluation``
    section. Runs made before that consolidation are still readable via the
    legacy evaluation/summary_metrics.json.
    """
    summary_file = exp_dir / "experiment_summary.json"
    if not summary_file.is_file():
        raise SystemExit(f"ERROR: no experiment_summary.json in {exp_dir}")
    with open(summary_file, encoding="utf-8") as f:
        summary = json.load(f)

    evaluation = summary.get("evaluation") or {}
    if not evaluation:
        legacy = exp_dir / "evaluation" / "summary_metrics.json"
        if legacy.is_file():
            with open(legacy, encoding="utf-8") as f:
                evaluation = json.load(f)

    row: Dict[str, Any] = dict(evaluation)
    row["_name"] = exp_dir.name
    for path, label, _ in CONTEXT:
        row[label] = _dig(summary, path)

    # The aggregate ratio is not stored; recompute it from the per-station
    # comparison when that file is present.
    row.setdefault("aggregate_ratio", _aggregate_ratio(exp_dir))
    return row


def _aggregate_ratio(exp_dir: Path) -> Optional[float]:
    """Total simulated / total observed, from volume_comparison.csv."""
    path = exp_dir / "evaluation" / "volume_comparison.csv"
    if not path.is_file():
        return None
    sim = obs = 0.0
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for rec in csv.DictReader(f):
                try:
                    o = float(rec.get("observed") or 0)
                    s = float(rec.get("simulated") or 0)
                except (TypeError, ValueError):
                    continue
                if o > 0:
                    obs += o
                    sim += s
    except OSError:
        return None
    return sim / obs if obs > 0 else None


def _verdict(first: Any, last: Any, direction: Optional[str]) -> str:
    if direction is None or first is None or last is None:
        return ""
    try:
        a, b = float(first), float(last)
    except (TypeError, ValueError):
        return ""
    if direction == "one":
        da, db = abs(a - 1.0), abs(b - 1.0)
        if abs(da - db) < 1e-9:
            return "same"
        return "better" if db < da else "worse"
    if abs(a - b) < 1e-9:
        return "same"
    if direction == "down":
        return "better" if b < a else "worse"
    return "better" if b > a else "worse"


def _fmt(value: Any, decimals: int) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:,.{decimals}f}" if decimals else f"{value:,.0f}"
    return str(value)


def print_table(rows: List[Dict[str, Any]], labels: List[str]) -> None:
    width = max(28, max(len(l) for l in labels) + 2)
    header = f"{'metric':<28}" + "".join(f"{l:>{width}}" for l in labels)
    if len(rows) > 1:
        header += "   verdict"
    print(header)
    print("-" * len(header))

    print("  CONFIGURATION")
    for _, label, decimals in CONTEXT:
        vals = [r.get(label) for r in rows]
        if all(v is None for v in vals):
            continue
        line = f"{label:<28}" + "".join(f"{_fmt(v, decimals):>{width}}" for v in vals)
        print(line)

    print("\n  COUNT VALIDATION")
    for key, label, decimals, direction in METRICS:
        vals = [r.get(key) for r in rows]
        if all(v is None for v in vals):
            continue
        line = f"{label:<28}" + "".join(f"{_fmt(v, decimals):>{width}}" for v in vals)
        if len(rows) > 1:
            line += f"   {_verdict(vals[0], vals[-1], direction)}"
        print(line)

    _print_reading(rows[-1])


def _print_reading(row: Dict[str, Any]) -> None:
    """A short plain-language verdict on the most recent run."""
    cv = row.get("station_ratio_cv")
    iqr = row.get("interquartile_mean_ratio")
    geh_lt5 = row.get("geh_lt_5_pct")
    print("\n  READING THE LAST COLUMN")
    if cv is None:
        print("    per-station CV not recorded — cannot tell whether the error is")
        print("    uniform or concentrated, so a scaling_factor change is not justified.")
    elif cv < 0.35:
        print(f"    CV {cv:.3f} — error is roughly UNIFORM across stations, so a global")
        print("    lever (scaling_factor) is a valid correction.")
    else:
        print(f"    CV {cv:.3f} — error is CONCENTRATED on particular corridors. Fix the")
        print("    spatial distribution (OD source, boundary policy, counts matching)")
        print("    before touching scaling_factor; a global multiplier would drag the")
        print("    already-correct stations off target.")
    if iqr is not None:
        if 0.9 <= iqr <= 1.1:
            print(f"    iqr_mean {iqr:.3f} — volume level is about right.")
        elif iqr > 1.1:
            print(f"    iqr_mean {iqr:.3f} — over-producing by ~{(iqr - 1) * 100:.0f}%.")
        else:
            print(f"    iqr_mean {iqr:.3f} — under-producing by ~{(1 - iqr) * 100:.0f}%.")
    if geh_lt5 is not None:
        print(f"    GEH < 5 on {geh_lt5:.1f}% of hourly counts — the standard criterion")
        print("    targets >=85%. Each count is tested on its own, so this is a pass")
        print("    rate rather than an average.")


def write_csv(rows: List[Dict[str, Any]], labels: List[str], out: Path) -> None:
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + labels)
        for _, label, _decimals in CONTEXT:
            w.writerow([label] + [r.get(label) for r in rows])
        for key, label, _, _ in METRICS:
            w.writerow([label] + [r.get(key) for r in rows])
    print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare experiments on their count-validation metrics.")
    ap.add_argument("experiments", nargs="+", help="experiment directories, oldest first")
    ap.add_argument("--labels", help="comma-separated column labels (default: dir names)")
    ap.add_argument("--csv", help="also write the table to this CSV path")
    args = ap.parse_args()

    dirs = [Path(e) for e in args.experiments]
    for d in dirs:
        if not d.is_dir():
            raise SystemExit(f"ERROR: not a directory: {d}")

    rows = [load_run(d) for d in dirs]
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(rows):
            raise SystemExit(
                f"ERROR: {len(labels)} labels for {len(rows)} experiments")
    else:
        labels = [d.name.replace("experiment_", "") for d in dirs]

    print()
    print_table(rows, labels)
    if args.csv:
        write_csv(rows, labels, Path(args.csv))


if __name__ == "__main__":
    main()
