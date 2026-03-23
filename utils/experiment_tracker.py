"""
Experiment Tracker - CSV comparison file for tracking and comparing experiments

This module reads experiment_summary.json and maintains a CSV file that compares
experiments across multiple dimensions to help identify parameter tuning opportunities.

Usage:
    from utils.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker()
    tracker.record_experiment(
        experiment_dir=Path("experiments/experiment_20260118_031147")
    )
"""

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


class ExperimentTracker:
    """Tracks and compares experiments in a CSV file"""

    # Base CSV columns (fixed) - nonwork plan columns are added dynamically
    # Ordered to tell the story: Population → Trip Generation → Scaling → Simulation → Results
    BASE_COLUMNS_BEFORE_PLANS = [
        # Identification
        'experiment_id',
        'date',
        'time',

        # Population Base - Who are we simulating?
        'total_population',
        'total_employees',
        'total_non_employees',

        # Unscaled Trips - Raw trip generation before scaling
        'unscaled_work_trips',
        'unscaled_nonwork_trips',
        'unscaled_total_trips',  # derived: sum of above
        'work_trip_rate',  # derived: unscaled_work_trips / unscaled_total_trips
        'nonwork_trip_rate',  # derived: unscaled_nonwork_trips / unscaled_total_trips

        # Scaling
        'scaling_factor',

        # Scaled Plans - work_plans is fixed, nonwork plans are dynamic
        'work_plans',
        'nonwork_plans',  # derived: total of all nonwork plan columns
    ]

    # These columns come after the dynamic nonwork plan columns
    BASE_COLUMNS_AFTER_PLANS = [
        'total_plans',

        # Simulation Scaling - What the plans represent in the real world
        'flow_capacity_factor',
        'storage_capacity_factor',
        'represented_real_trips',  # derived: total_plans / scaling_factor (actual trips these plans represent)
        'capacity_demand_ratio',  # derived: flow_capacity_factor / scaling_factor (>1 = less congestion, <1 = more congestion)

        # Plan percentages
        'work_pct',
        'nonwork_pct',

        # MATSim parameters
        'iterations',

        # Evaluation - primary
        'geh_lt_5_pct',
        'correlation',
        'peak_hour_correlation',
        'mean_geh',

        # Evaluation - secondary
        'mae',
        'rmse',
        'mean_pct_error',

        # Validation coverage
        'num_devices',

        # Diagnostics
        'chain_retries',
        'chain_retries_too_short',
        'chain_retries_bad_structure',
        'chain_retries_missing_purpose',
        'chain_retries_missing_work',
        'chain_retries_has_work',
        'chain_retries_too_many_work',
        'chain_attempts',
        'poi_retries',

        # Runtime
        'runtime_total_min',
        'runtime_plans_min',
        'runtime_matsim_min',
        'runtime_eval_min',

        # Suggestions & notes
        'suggestion',
        'notes',
    ]

    # Keys to exclude when extracting nonwork purpose names from plans dict
    PLANS_NON_PURPOSE_KEYS = {
        'work', 'total', 'success_rate',
        'chain_retries', 'chain_retries_too_short', 'chain_retries_bad_structure',
        'chain_retries_missing_purpose', 'chain_retries_missing_work',
        'chain_retries_has_work', 'chain_retries_too_many_work', 'chain_attempts',
        'poi_retries', 'time_retries'
    }

    def __init__(self, csv_path: Optional[Path] = None):
        """
        Initialize experiment tracker

        Args:
            csv_path: Path to CSV file. Defaults to experiments/experiment_comparison.csv
        """
        if csv_path is None:
            # Default location
            project_root = Path(__file__).parent.parent
            csv_path = project_root / 'experiments' / 'experiment_comparison.csv'

        self.csv_path = Path(csv_path)
        # Dynamic columns will be determined from experiment data
        self._nonwork_purpose_columns = []

    def _get_columns(self, nonwork_purposes: List[str] = None) -> List[str]:
        """
        Get full column list with dynamic nonwork plan columns.

        Args:
            nonwork_purposes: List of nonwork purpose names (lowercase)

        Returns:
            Complete list of column names
        """
        if nonwork_purposes is None:
            nonwork_purposes = self._nonwork_purpose_columns

        # Build dynamic plan columns (e.g., 'shopping_plans', 'school_plans')
        nonwork_plan_columns = [f'{p}_plans' for p in nonwork_purposes]

        return (
            self.BASE_COLUMNS_BEFORE_PLANS +
            nonwork_plan_columns +
            self.BASE_COLUMNS_AFTER_PLANS
        )

    def _extract_nonwork_purposes(self, plans_dict: Dict) -> List[str]:
        """
        Extract nonwork purpose names from plans dictionary.

        Args:
            plans_dict: The 'plans' dict from experiment_summary.json

        Returns:
            List of nonwork purpose names (lowercase)
        """
        purposes = []
        for key in plans_dict.keys():
            if key not in self.PLANS_NON_PURPOSE_KEYS:
                purposes.append(key)
        return sorted(purposes)  # Sort for consistent column order

    def _ensure_csv_exists(self, columns: List[str]):
        """Create CSV file with headers if it doesn't exist"""
        if not self.csv_path.exists():
            # Ensure parent directory exists
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)

            # Write header row
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()

    def _parse_summary_json(self, summary_path: Path) -> Tuple[Dict[str, Any], List[str]]:
        """
        Parse experiment_summary.json to extract all metrics

        Args:
            summary_path: Path to experiment_summary.json

        Returns:
            Tuple of (row dict with extracted metrics, list of nonwork purpose names)
        """
        if not summary_path.exists():
            return {}, []

        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
        except Exception:
            return {}, []

        # Extract nonwork purposes from plans dict
        plans = summary.get('plans', {})
        nonwork_purposes = self._extract_nonwork_purposes(plans)

        # Initialize row with all columns
        columns = self._get_columns(nonwork_purposes)
        row = {col: None for col in columns}

        # Identification
        row['experiment_id'] = summary.get('experiment_id')

        # Extract date and time from experiment_id or created_at
        exp_id = summary.get('experiment_id', '')
        date_time_match = re.search(r'experiment_(\d{8})_(\d{6})', exp_id)
        if date_time_match:
            date_str = date_time_match.group(1)
            time_str = date_time_match.group(2)
            row['date'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            row['time'] = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
        else:
            created_at = summary.get('created_at', '')
            if created_at:
                row['date'] = created_at[:10]
                if len(created_at) >= 19:
                    row['time'] = created_at[11:19]

        # Population base
        population = summary.get('population', {})
        row['total_population'] = population.get('total_population')
        row['total_employees'] = population.get('total_employees')
        row['total_non_employees'] = population.get('total_non_employees')

        # Unscaled trips
        unscaled = summary.get('unscaled_trips', {})
        row['unscaled_work_trips'] = unscaled.get('work')
        row['unscaled_nonwork_trips'] = unscaled.get('nonwork')

        # Plan counts - work is always present
        row['work_plans'] = plans.get('work', 0)
        row['total_plans'] = plans.get('total', 0)

        # Dynamic nonwork plan counts from config
        for purpose in nonwork_purposes:
            col_name = f'{purpose}_plans'
            row[col_name] = plans.get(purpose, 0)

        # Diagnostics from plans
        row['chain_retries'] = plans.get('chain_retries', 0)
        row['chain_retries_too_short'] = plans.get('chain_retries_too_short', 0)
        row['chain_retries_bad_structure'] = plans.get('chain_retries_bad_structure', 0)
        row['chain_retries_missing_purpose'] = plans.get('chain_retries_missing_purpose', 0)
        row['chain_retries_missing_work'] = plans.get('chain_retries_missing_work', 0)
        row['chain_retries_has_work'] = plans.get('chain_retries_has_work', 0)
        row['chain_retries_too_many_work'] = plans.get('chain_retries_too_many_work', 0)
        row['chain_attempts'] = plans.get('chain_attempts', 0)
        row['poi_retries'] = plans.get('poi_retries', 0)

        # Parameters
        params = summary.get('parameters', {})
        row['scaling_factor'] = params.get('scaling_factor')
        row['iterations'] = params.get('iterations')
        row['flow_capacity_factor'] = params.get('flow_capacity_factor')
        row['storage_capacity_factor'] = params.get('storage_capacity_factor')

        # Runtime
        runtime = summary.get('runtime', {})
        row['runtime_total_min'] = runtime.get('total_min')
        row['runtime_plans_min'] = runtime.get('plans_min')
        row['runtime_matsim_min'] = runtime.get('matsim_min')
        row['runtime_eval_min'] = runtime.get('eval_min')

        # Evaluation metrics
        evaluation = summary.get('evaluation', {})
        if evaluation:
            row['geh_lt_5_pct'] = evaluation.get('geh_lt_5_pct')
            row['correlation'] = evaluation.get('correlation')
            row['peak_hour_correlation'] = evaluation.get('peak_hour_correlation')
            row['mean_geh'] = evaluation.get('mean_geh')
            row['mae'] = evaluation.get('mae')
            row['rmse'] = evaluation.get('rmse')
            row['mean_pct_error'] = evaluation.get('mean_pct_error')
            row['num_devices'] = evaluation.get('num_devices')

        return row, nonwork_purposes

    def _calculate_derived_metrics(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate derived metrics from raw data

        Args:
            row: Dictionary with raw metrics

        Returns:
            Updated dictionary with derived metrics
        """
        total = row.get('total_plans', 0) or 0
        work = row.get('work_plans', 0) or 0

        # Calculate nonwork_plans as total - work
        row['nonwork_plans'] = total - work if total > 0 else 0

        # Calculate unscaled total and trip rates
        unscaled_work = row.get('unscaled_work_trips') or 0
        unscaled_nonwork = row.get('unscaled_nonwork_trips') or 0
        row['unscaled_total_trips'] = unscaled_work + unscaled_nonwork if (unscaled_work or unscaled_nonwork) else None

        if row['unscaled_total_trips'] and row['unscaled_total_trips'] > 0:
            row['work_trip_rate'] = round(unscaled_work / row['unscaled_total_trips'], 3)
            row['nonwork_trip_rate'] = round(unscaled_nonwork / row['unscaled_total_trips'], 3)
        else:
            row['work_trip_rate'] = None
            row['nonwork_trip_rate'] = None

        # Calculate plan percentages
        if total > 0:
            row['work_pct'] = round((row.get('work_plans', 0) or 0) / total * 100, 1)

            nonwork = total - (row.get('work_plans', 0) or 0)
            row['nonwork_pct'] = round(nonwork / total * 100, 1)
        else:
            row['work_pct'] = None
            row['nonwork_pct'] = None

        # Calculate represented real trips and capacity/demand ratio
        flow_cap = row.get('flow_capacity_factor')
        scaling = row.get('scaling_factor')

        if scaling and scaling > 0 and total > 0:
            # Represented real trips = actual real-world trips these plans represent
            # Example: 397,052 plans with scaling_factor=0.11 represents 3,609,564 real trips
            row['represented_real_trips'] = int(total / scaling)
        else:
            row['represented_real_trips'] = None

        if scaling and scaling > 0 and flow_cap and flow_cap > 0:
            # Capacity/demand ratio = road capacity scaling vs demand scaling
            # > 1.0 means roads have MORE capacity than demand (LESS congestion than real world)
            # < 1.0 means roads have LESS capacity than demand (MORE congestion than real world)
            # = 1.0 means balanced (realistic congestion)
            # Example: flow_cap=0.13, scaling=0.11 -> ratio=1.18 (roads 18% less congested)
            row['capacity_demand_ratio'] = round(flow_cap / scaling, 2)
        else:
            row['capacity_demand_ratio'] = None

        return row

    def _generate_suggestion(self, row: Dict[str, Any]) -> str:
        """
        Generate actionable suggestions based on metrics

        Args:
            row: Dictionary with all experiment metrics

        Returns:
            Suggestion string
        """
        suggestions = []

        geh_good = row.get('geh_lt_5_pct')
        bias = row.get('mean_pct_error')
        iters = row.get('iterations')
        corr = row.get('correlation')

        # Skip if no evaluation data
        if geh_good is None:
            return "No evaluation data available"

        # Volume/bias issues (most important)
        if bias is not None:
            if bias < -50:
                suggestions.append(
                    f"Under-simulating by {abs(bias):.0f}%. "
                    "Enable more trip purposes or increase trip generation rates."
                )
            elif bias > 50:
                suggestions.append(
                    f"Over-simulating by {bias:.0f}%. "
                    "Reduce scaling_factor or trip generation rates."
                )

        # Calibration quality
        if geh_good < 10:
            if iters is not None and iters < 10:
                suggestions.append(
                    f"Very poor calibration ({geh_good:.1f}% GEH<5). "
                    f"Try increasing iterations from {iters} to 15-20."
                )
            else:
                suggestions.append(
                    f"Very poor calibration ({geh_good:.1f}% GEH<5). "
                    "Fundamental volume or OD distribution issue."
                )
        elif geh_good < 40:
            if iters is not None and iters < 15:
                suggestions.append(f"Increase iterations from {iters} to 15-20 for better convergence.")
            else:
                suggestions.append("Tune trip generation rates or OD matrix beta parameter.")
        elif geh_good < 70:
            suggestions.append("Approaching acceptable. Fine-tune trip generation rates or time distributions.")
        else:
            suggestions.append("Good calibration (>70% GEH<5). Consider higher scaling for production.")

        # Correlation issues
        if corr is not None and corr < 0.5:
            suggestions.append("Low temporal correlation. Check departure time distributions.")

        return " | ".join(suggestions) if suggestions else "No issues detected."

    def record_experiment(
        self,
        experiment_dir: Path,
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Record an experiment to the comparison CSV

        Reads all metrics from experiment_summary.json (single source of truth).
        Columns are dynamically determined from the nonwork_purposes in the experiment.

        Args:
            experiment_dir: Path to experiment directory
            notes: Optional notes to add

        Returns:
            Dictionary with recorded row data
        """
        experiment_dir = Path(experiment_dir)
        summary_path = experiment_dir / "experiment_summary.json"

        # Parse summary JSON (single source of truth)
        row, nonwork_purposes = self._parse_summary_json(summary_path)

        if not row:
            # Return empty dict if parsing failed
            return {}

        # Store nonwork purposes for column generation
        self._nonwork_purpose_columns = nonwork_purposes

        # Calculate derived metrics
        row = self._calculate_derived_metrics(row)

        # Generate suggestion
        row['suggestion'] = self._generate_suggestion(row)

        # Add notes
        row['notes'] = notes

        # Get columns for this experiment
        columns = self._get_columns(nonwork_purposes)

        # Ensure CSV exists with correct headers
        self._ensure_csv_exists(columns)

        # Check if experiment already exists in CSV
        existing_rows = self._read_existing_rows()

        experiment_id = row.get('experiment_id')
        if experiment_id:
            # Find the latest row for this experiment (could be base or _runN)
            latest_match = None
            max_run = 0
            for existing_row in existing_rows:
                existing_id = existing_row.get('experiment_id', '')
                if existing_id == experiment_id:
                    latest_match = existing_row
                    max_run = max(max_run, 1)
                else:
                    run_match = re.match(
                        re.escape(experiment_id) + r'_run(\d+)$', existing_id
                    )
                    if run_match:
                        latest_match = existing_row
                        run_num = int(run_match.group(1))
                        max_run = max(max_run, run_num)

            if latest_match is not None:
                # Carry forward non-zero values from previous run for fields
                # that would be 0/None when only evaluation is rerun
                self._carry_forward_values(row, latest_match)

                # Recalculate derived metrics after carry-forward
                row = self._calculate_derived_metrics(row)
                row['suggestion'] = self._generate_suggestion(row)

                # Assign new run number
                new_run = max_run + 1 if max_run >= 1 else 2
                row['experiment_id'] = f"{experiment_id}_run{new_run}"

        existing_rows.append(row)

        # Write all rows back to CSV with current columns
        self._write_rows(existing_rows, columns)

        return row

    # Columns that should be carried forward from a previous run when the
    # current value is 0 or None (i.e., when only evaluation was rerun)
    _CARRY_FORWARD_COLUMNS = [
        'total_population', 'total_employees', 'total_non_employees',
        'unscaled_work_trips', 'unscaled_nonwork_trips',
        'scaling_factor', 'work_plans', 'total_plans',
        'flow_capacity_factor', 'storage_capacity_factor',
        'iterations', 'chain_retries', 'poi_retries',
        'runtime_plans_min', 'runtime_matsim_min',
    ]

    def _carry_forward_values(self, new_row: Dict[str, Any], old_row: Dict[str, Any]):
        """
        For eval-only reruns, carry forward values from the previous run
        when the new values are 0 or None (meaning those steps weren't run).
        Also carries forward dynamic nonwork plan columns.
        """
        all_columns = list(self._CARRY_FORWARD_COLUMNS)
        # Also carry forward any dynamic nonwork plan columns
        for key in old_row:
            if key.endswith('_plans') and key not in ('work_plans', 'total_plans', 'nonwork_plans'):
                if key not in all_columns:
                    all_columns.append(key)

        for col in all_columns:
            new_val = new_row.get(col)
            old_val = old_row.get(col)
            # Treat 0, None, empty string, and '0' as "not set"
            if self._is_empty(new_val) and not self._is_empty(old_val):
                # Convert string values from CSV back to numbers
                try:
                    old_val = float(old_val)
                    if old_val == int(old_val):
                        old_val = int(old_val)
                except (ValueError, TypeError):
                    pass
                new_row[col] = old_val

    @staticmethod
    def _is_empty(val) -> bool:
        """Check if a value is empty/zero (meaning the step wasn't run)"""
        if val is None or val == '' or val == 'None':
            return True
        try:
            return float(val) == 0
        except (ValueError, TypeError):
            return False

    def _read_existing_rows(self) -> List[Dict[str, Any]]:
        """Read existing rows from CSV file"""
        rows = []
        if self.csv_path.exists():
            try:
                with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
            except Exception:
                pass
        return rows

    def _write_rows(self, rows: List[Dict[str, Any]], columns: List[str]):
        """Write all rows to CSV file with specified columns"""
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

    def get_comparison_summary(self) -> str:
        """
        Get a formatted summary of all experiments

        Returns:
            Formatted string summary
        """
        rows = self._read_existing_rows()

        if not rows:
            return "No experiments recorded yet."

        summary_lines = [
            f"Experiment Comparison Summary ({len(rows)} experiments)",
            "=" * 60,
            ""
        ]

        for row in rows:
            exp_id = row.get('experiment_id', 'Unknown')
            total = row.get('total_plans', 'N/A')
            geh = row.get('geh_lt_5_pct', 'N/A')
            corr = row.get('correlation', 'N/A')
            peak_corr = row.get('peak_hour_correlation', 'N/A')

            summary_lines.append(
                f"{exp_id}: {total} plans, GEH<5: {geh}%, Corr: {corr}, Peak Corr: {peak_corr}"
            )

        return "\n".join(summary_lines)


def record_experiment_from_runner(experiment_dir: Path, notes: str = "") -> Dict[str, Any]:
    """
    Convenience function to record an experiment after run_experiment.py completes

    Args:
        experiment_dir: Path to experiment directory
        notes: Optional notes

    Returns:
        Dictionary with recorded row data
    """
    tracker = ExperimentTracker()
    return tracker.record_experiment(experiment_dir, notes=notes)
