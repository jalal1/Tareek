"""Estimator Orchestrator — runs all estimators for a region config.

Calls sub-estimators in sequence:
  1. demand_estimator  — trip demand, transit config_rate, scaling factors
  2. mode_share_estimator — MATSim scoring + transitRouter params from ACS
     transit share and (optionally) a prior experiment's realised mode shares
  3. freight_estimator — truck_share and vehicle_mix from HPMS, and (with
     --experiment-dir) demand_scale from tier-2 corridor validation. Skips
     itself when freight.enabled is false, which is the common case.

Each sub-estimator writes its own log file under logs/ and updates
config_estimated.json in the region folder. That JSON is the single source
of truth: at experiment time, ConfigManager overlays its
matsim.configurable_params onto the base MATSim template
(matsim/configs/<mode>/config.xml). The estimators do not write any XML.

Running the orchestrator is equivalent to running both sub-estimators
individually but with a single command and a combined summary at the end.

Usage:
    # Cold start - positional is the base config JSON
    python estimators/estimator.py config/USA/TwinCities/config_twin.json

    # Feedback - positional is the region FOLDER; the estimator reads
    # <experiment-dir>/config_used.json and writes <region>/config_estimated.json
    python estimators/estimator.py config/USA/TwinCities \
        --experiment-dir E:/jetstream2_experiments/april2026/experiment_20260430_121156

The --experiment-dir flag is forwarded to ALL sub-estimators:
  - demand_estimator uses it to load experiment_summary.json and tune demand.
  - mode_share_estimator uses it to load modestats.csv + config.xml and
    apply the clamped log-ratio update toward ACS targets.
  - freight_estimator uses it to run tier 2 against the events file and derive
    demand_scale. Without it, freight still estimates truck_share and
    vehicle_mix from HPMS, which need no experiment.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from estimators.demand_estimator import resolve_estimator_inputs


def _run(script: Path, config_file: str, extra_args: list) -> int:
    """Run a sub-estimator script and stream its output. Returns exit code."""
    cmd = [sys.executable, str(script), config_file] + extra_args
    result = subprocess.run(cmd)
    return result.returncode


def _require_acs_key(positional: str, experiment_dir: str | None) -> None:
    """Single point of truth: refuse to run either estimator without a Census
    ACS key. The key is read from the same config the sub-estimators will read,
    so what we check here is exactly what they will use.
    """
    read_from, _ = resolve_estimator_inputs(positional, experiment_dir)
    with open(read_from, "r") as f:
        config = json.load(f)
    api_key = (config.get("data", {}).get("census_api_key", "") or "").strip()
    if not api_key:
        print("=" * 70, file=sys.stderr)
        print("  ERROR: census_api_key missing from config.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(
            f"  Config read: {read_from}\n"
            f"  Both estimators rely on Census ACS B08301 (mode share) data.\n"
            f"  Without a key the demand estimator cannot calibrate transit\n"
            f"  parameters and the mode-share estimator cannot compute targets;\n"
            f"  the resulting config_estimated.json would be unsafe to apply.\n"
            f"  Add 'census_api_key' under 'data' in config.json.\n"
            f"  Free key signup: https://api.census.gov/data/key_signup.html",
            file=sys.stderr,
        )
        sys.exit(2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimator orchestrator — runs demand and mode share estimators"
    )
    parser.add_argument(
        "config_or_region",
        help="Cold start: path to config JSON. Feedback (with --experiment-dir): "
             "path to the region folder. See module docstring for examples.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="Path to a previous experiment folder (passed to demand_estimator)",
    )
    parser.add_argument(
        "--skip-demand",
        action="store_true",
        help="Skip demand_estimator and run only mode_share_estimator",
    )
    parser.add_argument(
        "--skip-mode-share",
        action="store_true",
        help="Skip mode_share_estimator and run only demand_estimator",
    )
    parser.add_argument(
        "--skip-freight",
        action="store_true",
        help="Skip freight_estimator. It already no-ops when freight.enabled "
             "is false; use this to skip it for a freight-enabled region, e.g. "
             "to avoid the events-file parse on a first feedback run.",
    )
    args = parser.parse_args()

    estimators_dir = Path(__file__).parent
    demand_script     = estimators_dir / "demand_estimator.py"
    mode_share_script = estimators_dir / "mode_share_estimator.py"
    freight_script    = estimators_dir / "freight_estimator.py"

    print("=" * 70)
    print("  ESTIMATOR ORCHESTRATOR")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Input : {args.config_or_region}")
    if args.experiment_dir:
        print(f"  Mode  : FEEDBACK (--experiment-dir {args.experiment_dir})")
    else:
        print(f"  Mode  : COLD START")
    print("=" * 70)

    _require_acs_key(args.config_or_region, args.experiment_dir)

    results = {}

    # ------------------------------------------------------------------
    # 1. Demand estimator
    # ------------------------------------------------------------------
    if not args.skip_demand:
        print()
        print("=" * 70)
        print("  RUNNING: demand_estimator")
        print("=" * 70)
        extra = []
        if args.experiment_dir:
            extra += ["--experiment-dir", args.experiment_dir]
        rc = _run(demand_script, args.config_or_region, extra)
        results["demand_estimator"] = "OK" if rc == 0 else f"FAILED (exit {rc})"
        if rc != 0:
            print(f"\n!! demand_estimator exited with code {rc}")
    else:
        results["demand_estimator"] = "skipped"

    # ------------------------------------------------------------------
    # 2. Mode share estimator
    # ------------------------------------------------------------------
    if not args.skip_mode_share:
        print()
        print("=" * 70)
        print("  RUNNING: mode_share_estimator")
        print("=" * 70)
        extra = []
        if args.experiment_dir:
            extra += ["--experiment-dir", args.experiment_dir]
        rc = _run(mode_share_script, args.config_or_region, extra)
        results["mode_share_estimator"] = "OK" if rc == 0 else f"FAILED (exit {rc})"
        if rc != 0:
            print(f"\n!! mode_share_estimator exited with code {rc}")
    else:
        results["mode_share_estimator"] = "skipped"

    # ------------------------------------------------------------------
    # 3. Freight estimator
    #
    # Runs last so it merges onto an estimated config the other two have
    # already written. It exits 0 without touching anything when freight is
    # disabled, so it is safe to run for every region.
    # ------------------------------------------------------------------
    if not args.skip_freight:
        print()
        print("=" * 70)
        print("  RUNNING: freight_estimator")
        print("=" * 70)
        extra = []
        if args.experiment_dir:
            extra += ["--experiment-dir", args.experiment_dir]
        rc = _run(freight_script, args.config_or_region, extra)
        results["freight_estimator"] = "OK" if rc == 0 else f"FAILED (exit {rc})"
        if rc != 0:
            print(f"\n!! freight_estimator exited with code {rc}")
    else:
        results["freight_estimator"] = "skipped"

    # ------------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  ORCHESTRATOR SUMMARY")
    print("=" * 70)
    for name, status in results.items():
        print(f"  {name:<30}  {status}")
    pos = Path(args.config_or_region)
    if args.experiment_dir:
        estimated = pos / "config_estimated.json"
    elif pos.stem.endswith("_estimated"):
        estimated = pos
    else:
        estimated = pos.with_name(f"{pos.stem}_estimated{pos.suffix}")
    print()
    print("  Outputs (if estimators succeeded):")
    print(f"    {estimated}")
    print(f"    logs/demand_estimator_*.log")
    print(f"    logs/mode_share_estimator_*.log")
    print(f"    logs/freight_estimator_*.log")
    print()

    any_failed = any("FAILED" in s for s in results.values())
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
