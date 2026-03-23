import multiprocessing
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Optional, Tuple
from utils.logger import setup_logger
from data_sources.base_survey_trip import BaseSurveyTrip

warnings.filterwarnings('ignore')
logger = setup_logger(__name__)


def _is_main_process() -> bool:
    """Return True if running in the main process (not a multiprocessing worker)."""
    return multiprocessing.current_process().name == 'MainProcess'

class TripDurationModel:
    """
    Model departure and arrival time distributions by activity pair.
    Stores KDE models for sampling realistic synthetic trip times.
    """

    def __init__(self, df,
                 depart_col=BaseSurveyTrip.DEPART_TIME,
                 arrive_col=BaseSurveyTrip.ARRIVE_TIME,
                 origin_col=BaseSurveyTrip.ORIGIN_PURPOSE,
                 dest_col=BaseSurveyTrip.DESTINATION_PURPOSE,
                 config: Optional[Dict] = None):
        """
        Args:
            df: DataFrame with datetime columns and purpose columns
            depart_col: name of departure time column (dtype datetime64[ns])
            arrive_col: name of arrival time column (dtype datetime64[ns])
            origin_col: name of origin purpose category column
            dest_col: name of destination purpose category column
            config: Configuration dictionary (for trip duration constraints)
        """
        self.df = df.copy()
        self.depart_col = depart_col
        self.arrive_col = arrive_col
        self.origin_col = origin_col
        self.dest_col = dest_col
        self.config = config or {}

        self.depart_models = {}  # KDE models for departure times
        self.arrive_models = {}  # KDE models for arrival times
        self.activity_pairs = []  # List of (origin, destination) pairs
        self._mean_trip_durations = {}  # Mean trip duration (minutes) per activity pair
        self._global_mean_duration = None  # Fallback global mean

        self._fit_distributions()
    
    def _time_to_minutes(self, dt_series):
        """Convert datetime to minutes since midnight."""
        return dt_series.dt.hour * 60 + dt_series.dt.minute + dt_series.dt.second / 60
    
    def _fit_distributions(self):
        """Fit KDE models for each activity pair."""
        # Get trip duration limits from config, with hardcoded fallbacks
        trip_constraints = self.config.get('duration_constraints', {}).get('trip_durations', {}).get('default', {})
        max_trip_dur = trip_constraints.get('max_minutes', 180)
        min_trip_dur = trip_constraints.get('min_minutes', 1)
        if 'duration_constraints' not in self.config or 'trip_durations' not in self.config.get('duration_constraints', {}):
            if _is_main_process():
                logger.warning(
                    f"FALLBACK: No 'duration_constraints.trip_durations' in config. "
                    f"Using hardcoded trip duration filter range [{min_trip_dur}, {max_trip_dur}] minutes. "
                    f"Add 'duration_constraints.trip_durations.default' to config.json to control this."
                )

        min_kde_samples = 5

        # Group by origin-destination purpose pair
        grouped = self.df.groupby([self.origin_col, self.dest_col])

        for (origin, dest), group in grouped:
            activity_key = (origin, dest)
            self.activity_pairs.append(activity_key)

            # Convert times to minutes since midnight
            depart_min = self._time_to_minutes(group[self.depart_col].dropna())
            arrive_min = self._time_to_minutes(group[self.arrive_col].dropna())

            # Fit KDE if we have enough samples
            if len(depart_min) > min_kde_samples:
                try:
                    self.depart_models[activity_key] = gaussian_kde(
                        depart_min,
                        bw_method='scott'
                    )
                except Exception:
                    if _is_main_process():
                        logger.warning(f"Could not fit KDE for depart {activity_key}")
            elif _is_main_process():
                logger.warning(
                    f"INSUFFICIENT DATA: Only {len(depart_min)} departure samples for "
                    f"{activity_key} (need >{min_kde_samples}). No KDE model fitted for this pair."
                )

            if len(arrive_min) > min_kde_samples:
                try:
                    self.arrive_models[activity_key] = gaussian_kde(
                        arrive_min,
                        bw_method='scott'
                    )
                except Exception:
                    if _is_main_process():
                        logger.warning(f"Could not fit KDE for arrive {activity_key}")
            elif _is_main_process():
                logger.warning(
                    f"INSUFFICIENT DATA: Only {len(arrive_min)} arrival samples for "
                    f"{activity_key} (need >{min_kde_samples}). No KDE model fitted for this pair."
                )

            # Compute mean trip duration (arrive - depart) for this pair
            # Filter using config-driven limits
            trip_dur = arrive_min - depart_min
            valid_dur = trip_dur[(trip_dur > min_trip_dur) & (trip_dur <= max_trip_dur)]
            n_rejected = len(trip_dur) - len(valid_dur)
            if n_rejected > 0:
                logger.debug(
                    f"Trip duration filter for {activity_key}: kept {len(valid_dur)}/{len(trip_dur)} "
                    f"(rejected {n_rejected} outside [{min_trip_dur}, {max_trip_dur}] min range)"
                )
            if len(valid_dur) > 0:
                self._mean_trip_durations[activity_key] = float(valid_dur.mean())

        # Compute global mean as fallback for unknown pairs
        all_durations = [mean_dur for mean_dur in self._mean_trip_durations.values()]
        if all_durations:
            self._global_mean_duration = float(np.mean(all_durations))
        else:
            self._global_mean_duration = float((min_trip_dur + max_trip_dur) / 2.0)
            if _is_main_process():
                logger.warning(
                    f"FALLBACK: No valid trip durations found in survey data. "
                    f"Using midpoint of config range as global mean: {self._global_mean_duration:.1f} min. "
                    f"This means the survey has NO usable trip duration data — check survey loading."
                )

        logger.info(f"Fitted models for {len(self.activity_pairs)} activity pairs")
        logger.info(f"Mean trip durations: global={self._global_mean_duration:.1f}min, "
                    f"per-pair={len(self._mean_trip_durations)} pairs")
        
    def mean_trip_duration(self, origin_purpose: str, dest_purpose: str) -> float:
        """
        Get mean trip duration (minutes) for a given activity pair.

        Falls back to the global mean if the specific pair is not available.

        Args:
            origin_purpose: Origin activity type
            dest_purpose: Destination activity type

        Returns:
            Mean trip duration in minutes
        """
        activity_key = (origin_purpose, dest_purpose)
        if activity_key not in self._mean_trip_durations:
            logger.warning(
                f"FALLBACK: No survey mean trip duration for {activity_key}. "
                f"Using global mean={self._global_mean_duration:.1f} min instead."
            )
            return self._global_mean_duration
        return self._mean_trip_durations[activity_key]

    def get_sample_counts(self):
        """Get number of samples per activity pair."""
        # sort the function below by count
        # counts = self.df.groupby([self.origin_col, self.dest_col]).size().reset_index(name='count')
        counts = self.df.groupby([self.origin_col, self.dest_col]).size().sort_values(ascending=False).reset_index(name='count')
        return counts
    
    def sample_dep_arr_time(self, origin_purpose, dest_purpose, n_samples=1, random_state=None):
        """
        Sample departure and arrival times for a given activity pair.
        
        Args:
            origin_purpose: origin purpose category value
            dest_purpose: destination purpose category value
            n_samples: number of trips to sample
            random_state: random seed for reproducibility
            
        Returns:
            DataFrame with columns: depart_time, arrive_time, duration_min, 
            origin_purpose, dest_purpose
        """
        activity_key = (origin_purpose, dest_purpose)
        
        if activity_key not in self.depart_models:
            raise ValueError(f"No model for activity pair {activity_key}")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        depart_kde = self.depart_models[activity_key]
        arrive_kde = self.arrive_models[activity_key]
        
        # Sample departure and arrival times (in minutes since midnight)
        depart_min = depart_kde.resample(n_samples)[0]  # KDE.resample returns (n_features, n_samples)
        arrive_min = arrive_kde.resample(n_samples)[0]
        
        # Clip to valid range [0, 1440] minutes
        depart_min = np.clip(depart_min, 0, 1440)
        arrive_min = np.clip(arrive_min, 0, 1440)

        return depart_min, arrive_min
    
    def visualize_distributions(self, origin_purpose=None, dest_purpose=None):
        """
        Visualize empirical and fitted distributions for activity pair(s).
        Side-by-side layout: departure and arrival in separate columns.
        
        If origin_purpose and dest_purpose are provided, visualize that pair.
        Otherwise, visualize all activity pairs.
        
        Args:
            origin_purpose: origin purpose category (optional)
            dest_purpose: destination purpose category (optional)
        """
        if origin_purpose is not None and dest_purpose is not None:
            pairs_to_plot = [(origin_purpose, dest_purpose)]
        else:
            pairs_to_plot = self.activity_pairs
        
        n_pairs = len(pairs_to_plot)
        fig, axes = plt.subplots(n_pairs, 2, figsize=(14, 4 * n_pairs))
        
        # Handle single pair case (axes won't be 2D)
        if n_pairs == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (origin, dest) in enumerate(pairs_to_plot):
            activity_key = (origin, dest)
            
            # Get empirical data
            mask = (self.df[self.origin_col] == origin) & \
                   (self.df[self.dest_col] == dest)
            subset = self.df[mask]
            
            depart_min = self._time_to_minutes(subset[self.depart_col].dropna())
            arrive_min = self._time_to_minutes(subset[self.arrive_col].dropna())
            
            # Departure time
            ax = axes[idx, 0]
            ax.hist(depart_min / 60, bins=30, density=True, alpha=0.6, label='Empirical')
            if activity_key in self.depart_models:
                x = np.linspace(0, 1440, 500)
                kde_vals = self.depart_models[activity_key](x)
                ax.plot(x / 60, kde_vals * 60, 'r-', lw=2, label='KDE')
            ax.set_xlabel('Time of Day (hours)')
            ax.set_ylabel('Density')
            ax.set_title(f'Departure Time\n{origin} → {dest}\n(n={len(depart_min)})')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Arrival time
            ax = axes[idx, 1]
            ax.hist(arrive_min / 60, bins=30, density=True, alpha=0.6, label='Empirical')
            if activity_key in self.arrive_models:
                x = np.linspace(0, 1440, 500)
                kde_vals = self.arrive_models[activity_key](x)
                ax.plot(x / 60, kde_vals * 60, 'r-', lw=2, label='KDE')
            ax.set_xlabel('Time of Day (hours)')
            ax.set_ylabel('Density')
            ax.set_title(f'Arrival Time\n{origin} → {dest}\n(n={len(arrive_min)})')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_distributions_overlay(self, origin_purpose=None, dest_purpose=None):
        """
        Visualize departure and arrival KDE lines overlaid on the same figure for activity pair(s).
        
        If origin_purpose and dest_purpose are provided, visualize that pair.
        Otherwise, visualize all activity pairs (one figure per pair).
        
        Args:
            origin_purpose: origin purpose category (optional)
            dest_purpose: destination purpose category (optional)
        """
        if origin_purpose is not None and dest_purpose is not None:
            pairs_to_plot = [(origin_purpose, dest_purpose)]
        else:
            pairs_to_plot = self.activity_pairs
        
        figs = []
        for origin, dest in pairs_to_plot:
            activity_key = (origin, dest)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot KDE lines only
            x = np.linspace(0, 24, 500)
            x_min = x * 60  # Convert hours back to minutes for KDE evaluation
            
            if activity_key in self.depart_models:
                kde_vals = self.depart_models[activity_key](x_min)
                ax.plot(x, kde_vals / 60, color='#1f77b4', linewidth=3, label='Departure', linestyle='-')
            
            if activity_key in self.arrive_models:
                kde_vals = self.arrive_models[activity_key](x_min)
                ax.plot(x, kde_vals / 60, color='#ff7f0e', linewidth=3, label='Arrival', linestyle='-')
            
            ax.set_xlabel('Time of Day (hours)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax.set_title(f'{origin} → {dest}', 
                        fontsize=13, fontweight='bold')
            ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_xlim(0, 24)
            
            plt.tight_layout()
            figs.append(fig)
        
        return figs

class ActivityDurationModel:
    """
    Model activity duration distributions using KDE.

    Extracts durations from consecutive trips in person-day data:
    - Person goes Home→Work at 7:00-7:30, then Work→Shop at 13:00-13:15
    - Work duration = 13:00 - 7:30 = 5.5 hours

    Only fits "middle" activities (not first/last of the day).
    """

    def __init__(self, persons_dict: Dict, bw_method: str = 'scott', config: Optional[Dict] = None):
        """
        Initialize activity duration model.

        Args:
            persons_dict: From survey.process_persons()
                         {person_id: {date: trips_df}}
            bw_method: KDE bandwidth method ('scott' or 'silverman')
            config: Configuration dictionary with duration constraints
        """
        self.persons_dict = persons_dict
        self.bw_method = bw_method
        self.config = config or {}

        # Get activity duration constraints from config
        self.activity_constraints = self.config.get('duration_constraints', {}).get('activity_durations', {})

        self.survey_duration_models = {}  # Survey KDE models by activity type
        self.survey_duration_models_binned = {}  # Binned KDE models by (activity, bin_label)
        self.target_duration_models = {}  # Target KDE models (from mean+std) by activity type
        self.activity_types = []

        # Arrival-time bins for conditioning activity duration on time-of-day.
        # Separate duration KDEs are fitted per (activity, bin) pair so that,
        # e.g., workers arriving in the afternoon get shorter shift durations
        # instead of the global ~6-7h peak that causes a 21-22h departure spike.
        self._arrival_time_bins = [(0, 10), (10, 14), (14, 18), (18, 24)]
        self._arrival_time_bin_labels = ['morning', 'midday', 'afternoon', 'evening']

        # Extract and fit durations
        self.durations_df = self._extract_activity_durations()
        self._fit_distributions()

    def _get_arrival_bin(self, hour: float) -> Optional[str]:
        """Map an arrival hour (0-24) to a time-of-day bin label.

        Used to select a bin-specific duration KDE so that activity durations
        are conditioned on when the person actually arrives at the activity
        (e.g., afternoon arrivals at Work get shorter shift durations).

        Args:
            hour: Arrival hour in fractional hours (e.g., 14.5 = 2:30 PM)

        Returns:
            Bin label ('morning', 'midday', 'afternoon', 'evening'), or None
            if the hour falls outside all defined bins.
        """
        for (lo, hi), label in zip(self._arrival_time_bins, self._arrival_time_bin_labels):
            if lo <= hour < hi:
                return label
        return None

    def _extract_activity_durations(self) -> pd.DataFrame:
        """
        Extract activity durations from consecutive trips.

        For each person-day:
            For each consecutive trip pair (trip_i, trip_j):
                activity = trip_i.destination
                duration = trip_j.depart_time - trip_i.arrive_time

        Returns:
            DataFrame with columns: [activity, duration_minutes]
        """
        # Use per-activity constraints from config when available, otherwise global bounds
        global_min = 1    # minutes
        global_max = 720  # minutes (12 hours)

        durations = []
        n_rejected = 0

        for person_id, days in self.persons_dict.items():
            for date, trips_df in days.items():
                if trips_df.empty or len(trips_df) < 2:
                    continue

                # Sort trips by departure time
                trips_sorted = trips_df.sort_values(BaseSurveyTrip.DEPART_TIME).reset_index(drop=True)

                # Process consecutive trip pairs
                for i in range(len(trips_sorted) - 1):
                    trip_i = trips_sorted.iloc[i]
                    trip_j = trips_sorted.iloc[i + 1]

                    # Activity is the destination of trip_i
                    activity = trip_i[BaseSurveyTrip.DESTINATION_PURPOSE]

                    # Skip Home activities (first/last of the day)
                    if activity == BaseSurveyTrip.ACT_HOME:
                        continue

                    # Duration = departure time of next trip - arrival time of current trip
                    arrive_time = pd.to_datetime(trip_i[BaseSurveyTrip.ARRIVE_TIME])
                    depart_time = pd.to_datetime(trip_j[BaseSurveyTrip.DEPART_TIME])

                    duration_seconds = (depart_time - arrive_time).total_seconds()
                    duration_minutes = duration_seconds / 60.0

                    # Use per-activity config bounds if available, else global
                    act_cfg = self.activity_constraints.get(activity, {})
                    min_dur = act_cfg.get('min_minutes', global_min)
                    max_dur = act_cfg.get('max_minutes', global_max)

                    if min_dur <= duration_minutes <= max_dur:
                        durations.append({
                            'activity': activity,
                            'duration_minutes': duration_minutes,
                            'arrival_hour': arrive_time.hour + arrive_time.minute / 60.0,
                        })
                    else:
                        n_rejected += 1

        is_main = multiprocessing.current_process().name == 'MainProcess'
        if n_rejected > 0 and is_main:
            logger.warning(
                f"FILTER: Rejected {n_rejected} activity durations from survey "
                f"(outside per-activity [min_minutes, max_minutes] bounds from config). "
                f"Kept {len(durations)} valid durations."
            )
        if is_main:
            logger.info(f"Extracted {len(durations)} activity durations from survey")
        return pd.DataFrame(durations)

    def _fit_distributions(self):
        """Fit KDE models for each activity type (both survey and target if configured)."""
        if self.durations_df.empty:
            if _is_main_process():
                logger.warning(
                    "CRITICAL: No activity durations extracted from survey — cannot fit any duration models. "
                    "Check that the survey has consecutive trips per person-day."
                )
            return

        min_kde_samples = 5
        grouped = self.durations_df.groupby('activity')

        survey_fitted_count = 0
        target_fitted_count = 0

        for activity, group in grouped:
            self.activity_types.append(activity)

            duration_min = group['duration_minutes'].values

            # Fit survey KDE if we have enough samples
            if len(duration_min) > min_kde_samples:
                try:
                    self.survey_duration_models[activity] = gaussian_kde(
                        duration_min,
                        bw_method=self.bw_method
                    )
                    survey_fitted_count += 1
                    if _is_main_process():
                        logger.info(
                            f"  Activity '{activity}': fitted survey KDE from {len(duration_min)} samples "
                            f"(mean={duration_min.mean():.1f} min, std={duration_min.std():.1f} min)"
                        )
                except Exception as e:
                    if _is_main_process():
                        logger.warning(f"Could not fit survey KDE for activity {activity}: {e}")
            elif _is_main_process():
                logger.warning(
                    f"INSUFFICIENT DATA: Only {len(duration_min)} duration samples for activity "
                    f"'{activity}' (need >{min_kde_samples}). No survey KDE model fitted — "
                    f"sampling this activity will fail."
                )

            # Fit target KDE if target_mean_minutes and target_std_minutes are provided in config
            if activity in self.activity_constraints:
                constraints = self.activity_constraints[activity]
                target_mean = constraints.get('target_mean_minutes')
                target_std = constraints.get('target_std_minutes')
                blend_weight = constraints.get('blend_weight', 0.0)

                if target_mean is not None and target_std is not None:
                    try:
                        # Generate synthetic samples from Normal(mean, std)
                        np.random.seed(42)  # Reproducible synthetic data
                        synthetic_samples = np.random.normal(target_mean, target_std, size=1000)

                        # Fit KDE on synthetic samples
                        self.target_duration_models[activity] = gaussian_kde(
                            synthetic_samples,
                            bw_method=self.bw_method
                        )
                        target_fitted_count += 1

                        if _is_main_process():
                            if blend_weight > 0:
                                logger.info(
                                    f"  Activity '{activity}': target KDE (mean={target_mean:.1f}, "
                                    f"std={target_std:.1f}) with blend_weight={blend_weight:.2f} — "
                                    f"OVERRIDING survey distribution with weighted blend"
                                )
                            else:
                                logger.debug(
                                    f"  Activity '{activity}': target KDE fitted but blend_weight=0 "
                                    f"(pure survey, target inactive)"
                                )
                    except Exception as e:
                        if _is_main_process():
                            logger.warning(f"Could not fit target KDE for activity {activity}: {e}")

            # Warn if activity exists in survey but has no config constraints
            if activity not in self.activity_constraints and _is_main_process():
                logger.warning(
                    f"MISSING CONFIG: Activity '{activity}' found in survey data but has no entry "
                    f"in 'duration_constraints.activity_durations' in config.json. "
                    f"Sampling will fail for this activity."
                )

        if _is_main_process():
            logger.info(f"Fitted {survey_fitted_count} survey KDE models for activity types")
            if target_fitted_count > 0:
                logger.info(f"Fitted {target_fitted_count} target KDE models for activity types")

        # --- Fit arrival-time-binned KDEs ---
        # To fix the evening departure spike: departure time and activity duration
        # are sampled independently, so late-starting workers (14-16h) all draw from
        # the global ~6-7h KDE and converge on 21-22h end times. By fitting separate
        # duration KDEs per arrival-time bin, afternoon workers get shorter (shift-
        # appropriate) durations. Falls back to the global KDE when a bin has fewer
        # than min_kde_bin_samples observations.
        min_kde_bin_samples = 20
        if 'arrival_hour' in self.durations_df.columns:
            self.durations_df['arrival_bin'] = self.durations_df['arrival_hour'].apply(self._get_arrival_bin)
            binned_grouped = self.durations_df.dropna(subset=['arrival_bin']).groupby(['activity', 'arrival_bin'])
            binned_count = 0
            for (activity, bin_label), grp in binned_grouped:
                dur_vals = grp['duration_minutes'].values
                if len(dur_vals) >= min_kde_bin_samples:
                    try:
                        self.survey_duration_models_binned[(activity, bin_label)] = gaussian_kde(
                            dur_vals, bw_method=self.bw_method
                        )
                        binned_count += 1
                        if _is_main_process():
                            logger.info(
                                f"  Activity '{activity}' bin '{bin_label}': fitted binned KDE from "
                                f"{len(dur_vals)} samples (mean={dur_vals.mean():.1f} min, "
                                f"std={dur_vals.std():.1f} min)"
                            )
                    except Exception as e:
                        if _is_main_process():
                            logger.warning(f"Could not fit binned KDE for ({activity}, {bin_label}): {e}")
                else:
                    if _is_main_process():
                        logger.debug(
                            f"  Activity '{activity}' bin '{bin_label}': only {len(dur_vals)} samples "
                            f"(< {min_kde_bin_samples}), using global KDE as fallback"
                        )
            if _is_main_process():
                logger.info(f"Fitted {binned_count} arrival-time-binned KDE models")

    def sample_duration(self, activity: str, n_samples: int = 1,
                        arrival_hour: Optional[float] = None) -> np.ndarray:
        """
        Sample activity duration with optional blending between survey and target KDEs.

        If arrival_hour is provided and a binned KDE exists for the corresponding
        arrival-time bin, uses that bin-specific KDE instead of the global one.
        Falls back to the global KDE if no binned model is available.

        Args:
            activity: Activity type (e.g., 'Work', 'Shopping')
            n_samples: Number of samples to generate
            arrival_hour: Estimated arrival hour at this activity (0-24), used to
                          select a time-bin-specific duration KDE. None uses global KDE.

        Returns:
            Array of activity durations in minutes

        Raises:
            ValueError: If activity not found, constraints missing, or cannot generate valid samples
        """
        if activity not in self.survey_duration_models:
            raise ValueError(
                f"No duration model found for activity '{activity}'. "
                f"Available activities: {list(self.survey_duration_models.keys())}"
            )

        # Get constraints from config (must exist, no defaults)
        if activity not in self.activity_constraints:
            raise ValueError(
                f"No activity constraints found for activity '{activity}' in config.json. "
                f"Please add constraints for this activity. "
                f"Available activities: {list(self.activity_constraints.keys())}"
            )
        constraints = self.activity_constraints[activity]

        # Extract min/max - fail explicitly if not in config
        try:
            min_dur = constraints['min_minutes']
            max_dur = constraints['max_minutes']
        except KeyError as e:
            raise ValueError(
                f"Missing required constraint {e} for activity '{activity}' in config.json. "
                f"Both 'min_minutes' and 'max_minutes' must be specified."
            )

        # Get blend_weight (default 0.0 = pure survey)
        blend_weight = constraints.get('blend_weight', 0.0)

        # Get max attempts from config (with default)
        max_attempts = self.config.get('time_models', {}).get('max_duration_sample_attempts', 100)

        valid_samples = []
        total_generated = 0

        for attempt in range(max_attempts):
            # Generate samples (oversample to increase efficiency)
            batch_size = max(n_samples - len(valid_samples), n_samples)

            # Select survey KDE: use binned model if arrival_hour provided and available
            survey_kde = self.survey_duration_models[activity]  # global fallback
            if arrival_hour is not None:
                bin_label = self._get_arrival_bin(arrival_hour)
                binned_key = (activity, bin_label)
                if bin_label is not None and binned_key in self.survey_duration_models_binned:
                    survey_kde = self.survey_duration_models_binned[binned_key]
            survey_samples = survey_kde.resample(batch_size)[0]

            # If target KDE exists and blend_weight > 0, blend with target samples
            if activity in self.target_duration_models and blend_weight > 0:
                target_kde = self.target_duration_models[activity]
                target_samples = target_kde.resample(batch_size)[0]

                # Weighted blend: (1-w)*survey + w*target
                blended_samples = (1 - blend_weight) * survey_samples + blend_weight * target_samples

                logger.debug(
                    f"Blending {activity} durations: "
                    f"survey_mean={survey_samples.mean():.1f}, "
                    f"target_mean={target_samples.mean():.1f}, "
                    f"blend_weight={blend_weight:.2f}, "
                    f"result_mean={blended_samples.mean():.1f}"
                )
            else:
                blended_samples = survey_samples

            total_generated += batch_size

            # Keep only samples within valid range
            valid_mask = (blended_samples >= min_dur) & (blended_samples <= max_dur)
            valid_samples.extend(blended_samples[valid_mask])

            # Break if we have enough valid samples
            if len(valid_samples) >= n_samples:
                break

        # Check if we got enough valid samples
        if len(valid_samples) < n_samples:
            acceptance_rate = len(valid_samples) / total_generated * 100 if total_generated > 0 else 0
            raise ValueError(
                f"Could not generate {n_samples} valid samples for '{activity}' "
                f"after {max_attempts} attempts. Only got {len(valid_samples)} samples "
                f"(acceptance rate: {acceptance_rate:.1f}%). "
                f"This suggests the KDE distribution is poorly matched to the constraints "
                f"[{min_dur}, {max_dur}] minutes. Consider adjusting the constraints or "
                f"reviewing the empirical data for this activity."
            )

        duration_min = np.array(valid_samples[:n_samples])

        logger.debug(
            f"Sampled duration for '{activity}': {duration_min[0]:.1f} min "
            f"(range: {min_dur}-{max_dur}, mean: {duration_min.mean():.1f})"
        )

        return duration_min

    def get_statistics(self) -> pd.DataFrame:
        """Get summary statistics for activity durations."""
        if self.durations_df.empty:
            return pd.DataFrame()

        stats = self.durations_df.groupby('activity')['duration_minutes'].agg([
            'count', 'mean', 'std', 'min', 'median', 'max'
        ]).round(2)

        return stats.sort_values('count', ascending=False)


class BlendedTripDurationModel:
    """Blends multiple per-source TripDurationModels at sampling time.

    When ``sample_dep_arr_time`` is called, a source is picked with
    probability proportional to its configured weight.  The sample is
    then drawn from that source's KDE model.

    If the chosen source has no model for the requested activity pair,
    the other sources are tried in weight-descending order as fallback.
    """

    def __init__(self, models: Dict[str, TripDurationModel],
                 weights: Dict[str, float]):
        """
        Args:
            models:  {source_key: TripDurationModel}
            weights: {source_key: float} from ``data.surveys[].weight``.
        """
        self.models = models
        self.source_names = list(models.keys())

        raw = np.array([weights[name] for name in self.source_names],
                       dtype=float)
        total = raw.sum()
        if total <= 0:
            raise ValueError("Sum of blend weights must be positive")
        self.probabilities = raw / total

        logger.info(
            f"BlendedTripDurationModel: {len(self.source_names)} sources — "
            + ", ".join(
                f"{name}: {prob:.2%}"
                for name, prob in zip(self.source_names, self.probabilities)
            )
        )

    # ── Public interface (matches TripDurationModel) ──────────────────

    def sample_dep_arr_time(self, origin_purpose, dest_purpose,
                            n_samples=1, random_state=None):
        """Sample departure/arrival times from a weighted-random source.

        Falls back to other sources if the chosen one lacks the pair.
        """
        activity_key = (origin_purpose, dest_purpose)

        # Build a priority order: random primary, then descending weight
        source = np.random.choice(self.source_names, p=self.probabilities)
        fallback_order = [source] + [
            s for s in self.source_names if s != source
        ]

        for src in fallback_order:
            model = self.models[src]
            if activity_key in model.depart_models:
                return model.sample_dep_arr_time(
                    origin_purpose, dest_purpose,
                    n_samples=n_samples, random_state=random_state,
                )

        raise ValueError(
            f"No source has a model for activity pair {activity_key}"
        )

    def mean_trip_duration(self, origin_purpose: str, dest_purpose: str) -> float:
        """
        Get weighted-average mean trip duration across sources.

        Each source's mean for the pair is weighted by its blend probability.
        Sources missing the pair are skipped and weights renormalized.

        Args:
            origin_purpose: Origin activity type
            dest_purpose: Destination activity type

        Returns:
            Weighted mean trip duration in minutes
        """
        activity_key = (origin_purpose, dest_purpose)
        weighted_sum = 0.0
        weight_sum = 0.0

        for name, prob in zip(self.source_names, self.probabilities):
            model = self.models[name]
            if activity_key in model._mean_trip_durations:
                weighted_sum += prob * model._mean_trip_durations[activity_key]
                weight_sum += prob

        if weight_sum > 0:
            return weighted_sum / weight_sum

        # Fallback: weighted average of global means
        logger.warning(
            f"FALLBACK: No source has survey mean for {activity_key}. "
            f"Falling back to weighted average of global means across sources."
        )
        for name, prob in zip(self.source_names, self.probabilities):
            model = self.models[name]
            weighted_sum += prob * model._global_mean_duration
            weight_sum += prob

        if weight_sum > 0:
            return weighted_sum / weight_sum

        logger.warning(
            f"FALLBACK: All sources have zero weight — cannot compute mean trip duration "
            f"for {activity_key}. This should not happen; check blend weights."
        )
        return 30.0  # last-resort fallback, should never be reached

    @property
    def activity_pairs(self) -> List[Tuple]:
        """Union of activity pairs across all sources."""
        pairs = set()
        for model in self.models.values():
            pairs.update(model.activity_pairs)
        return list(pairs)


class BlendedActivityDurationModel:
    """Blends multiple per-source ActivityDurationModels at sampling time.

    Source selection works the same way as BlendedTripDurationModel:
    pick a source at random (weighted), fall back if that source has no
    model for the requested activity.
    """

    def __init__(self, models: Dict[str, ActivityDurationModel],
                 weights: Dict[str, float]):
        """
        Args:
            models:  {source_key: ActivityDurationModel}
            weights: {source_key: float} from ``data.surveys[].weight``.
        """
        self.models = models
        self.source_names = list(models.keys())

        raw = np.array([weights[name] for name in self.source_names],
                       dtype=float)
        total = raw.sum()
        if total <= 0:
            raise ValueError("Sum of blend weights must be positive")
        self.probabilities = raw / total

        logger.info(
            f"BlendedActivityDurationModel: {len(self.source_names)} sources — "
            + ", ".join(
                f"{name}: {prob:.2%}"
                for name, prob in zip(self.source_names, self.probabilities)
            )
        )

    # ── Public interface (matches ActivityDurationModel) ──────────────

    def sample_duration(self, activity: str, n_samples: int = 1,
                        arrival_hour: Optional[float] = None) -> np.ndarray:
        """Sample activity duration from a weighted-random source.

        Falls back to other sources if the chosen one lacks the activity.
        """
        source = np.random.choice(self.source_names, p=self.probabilities)
        fallback_order = [source] + [
            s for s in self.source_names if s != source
        ]

        for src in fallback_order:
            model = self.models[src]
            if activity in model.survey_duration_models:
                return model.sample_duration(activity, n_samples=n_samples,
                                             arrival_hour=arrival_hour)

        raise ValueError(
            f"No source has a duration model for activity '{activity}'"
        )

    @property
    def activity_types(self) -> List[str]:
        """Union of activity types across all sources."""
        types = set()
        for model in self.models.values():
            types.update(model.activity_types)
        return list(types)