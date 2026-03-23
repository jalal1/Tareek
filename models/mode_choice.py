"""
Mode choice model for multi-modal transportation.

This module provides the ModeChoiceModel class which selects transportation
modes for trips based on survey rates and availability constraints.

Phase 2 Implementation:
- Computes mode shares from survey data (blended across multiple surveys)
- Applies config rate blending and share adjustments
- Filters to available modes and renormalizes
- Samples mode from final distribution
- Supports chain consistency (entire chain uses same mode)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set

import numpy as np
import pandas as pd

from models.mode_types import ModeType, ModeConfig, get_default_car_config, MODE_CAR
from models.mode_availability import ModeAvailabilityManager, Location

logger = logging.getLogger(__name__)


@dataclass
class Leg:
    """Represents travel between two activities."""
    mode: str  # Output format mode string (e.g., "car", "pt")


class ModeChoiceModel:
    """
    Selects transportation mode for each trip based on survey rates.

    The mode choice process:
    1. Get available modes for the OD pair (from availability_manager)
    2. Get survey rates for the trip purpose (blended across surveys)
    3. Apply config rate blending and share adjustments per mode
    4. Filter to available modes only
    5. Renormalize so probabilities sum to 1.0
    6. Sample mode from the distribution

    For chain consistency, the entire activity chain uses the same mode,
    with retry logic if the chosen mode isn't available for all legs.
    """

    def __init__(self, config: Dict[str, Any],
                 availability_manager: Optional[ModeAvailabilityManager] = None,
                 survey_data: Optional[Dict[str, pd.DataFrame]] = None,
                 survey_weights: Optional[Dict[str, float]] = None,
                 gtfs_avail_manager=None):
        """
        Initialize mode choice model.

        Args:
            config: Full configuration dict (expects 'modes' and 'mode_choice' sections)
            availability_manager: Optional availability manager. If None, one will be created.
            survey_data: Optional dict of {survey_name: DataFrame} with survey trip data.
                        Each DataFrame should have 'mode_type' and 'destination_purpose' columns.
            survey_weights: Optional dict of {survey_name: weight} for blending surveys.
                           If None, equal weights are used.
            gtfs_avail_manager: Optional GTFSAvailabilityManager for transit availability.
                               Passed through to ModeAvailabilityManager if creating one.
        """
        self.config = config

        # Get modes config, default to car if not present
        modes_config = config.get('modes')
        if not modes_config:
            logger.warning("No 'modes' section in config, defaulting to car-only mode")
            modes_config = get_default_car_config()

        # Parse mode configurations
        self.mode_configs: Dict[ModeType, ModeConfig] = {}
        for mode_name, mode_cfg in modes_config.items():
            if not isinstance(mode_cfg, dict):
                continue
            if not mode_cfg.get('enabled', True):
                continue
            mc = ModeConfig.from_config(mode_name, mode_cfg)
            self.mode_configs[mc.mode_type] = mc

        # Ensure car mode is always available
        if ModeType.CAR not in self.mode_configs:
            logger.warning("Car mode not in config, adding default car mode")
            car_cfg = get_default_car_config()['car']
            self.mode_configs[ModeType.CAR] = ModeConfig.from_config('car', car_cfg)

        # Initialize or use provided availability manager
        if availability_manager is not None:
            self.availability_manager = availability_manager
        else:
            self.availability_manager = ModeAvailabilityManager(
                modes_config, gtfs_avail_manager=gtfs_avail_manager
            )

        # Mode choice parameters
        mode_choice_config = config.get('mode_choice', {})
        self.fallback_mode = ModeType(mode_choice_config.get('fallback_mode', 'car'))
        self.chain_consistency = mode_choice_config.get('chain_consistency', True)
        self.min_samples_per_purpose = mode_choice_config.get('min_samples_per_purpose', 30)
        self.max_chain_mode_retries = mode_choice_config.get('max_chain_mode_retries', 5)

        # Store survey data for rate computation
        self.survey_data = survey_data
        self.survey_weights = survey_weights

        # Compute survey rates (mode shares per purpose)
        self.survey_rates: Dict[str, Dict[ModeType, float]] = {}
        if survey_data:
            self._compute_survey_rates()
        else:
            logger.warning("No survey data provided - mode choice will use fallback mode")

        # Statistics for logging
        self.stats = {
            'mode_samples': {mt: 0 for mt in ModeType},
            'fallback_used': 0,
            'chain_retries': 0,
            'purposes_with_fallback': set(),
            'chain_mode_selections': {},  # {(mode, purpose, num_legs): count}
            'chain_availability': {},     # {frozenset(modes): count}
        }

        logger.info(f"ModeChoiceModel initialized with {len(self.mode_configs)} modes: "
                    f"{[m.value for m in self.mode_configs.keys()]}")

    def _compute_survey_rates(self) -> None:
        """
        Compute mode shares from survey data, blended across multiple surveys.

        Computes rates per destination_purpose and overall ('all').
        Uses survey weights for blending when multiple surveys are provided.

        Results stored in self.survey_rates: {purpose: {ModeType: share}}
        """
        if not self.survey_data:
            logger.warning("No survey data available for rate computation")
            return

        logger.info("Computing mode shares from survey data...")

        # Normalize survey weights
        if self.survey_weights:
            total_weight = sum(self.survey_weights.values())
            normalized_weights = {k: v / total_weight for k, v in self.survey_weights.items()}
        else:
            # Equal weights if not specified
            n_surveys = len(self.survey_data)
            normalized_weights = {name: 1.0 / n_surveys for name in self.survey_data}

        logger.debug(f"  Survey weights (normalized): {normalized_weights}")

        # Get all unique purposes across all surveys
        all_purposes = set()
        for df in self.survey_data.values():
            if 'destination_purpose' in df.columns:
                all_purposes.update(df['destination_purpose'].dropna().unique())

        purposes_to_compute = list(all_purposes) + ['all']
        logger.debug(f"  Computing rates for purposes: {purposes_to_compute}")

        # Compute rates per purpose
        for purpose in purposes_to_compute:
            blended_rates: Dict[ModeType, float] = {}

            for survey_name, df in self.survey_data.items():
                weight = normalized_weights.get(survey_name, 0.0)
                if weight == 0:
                    continue

                # Filter by purpose (or use all trips for 'all')
                if purpose == 'all':
                    purpose_df = df
                else:
                    purpose_df = df[df['destination_purpose'] == purpose]

                if len(purpose_df) < self.min_samples_per_purpose:
                    logger.debug(f"  Survey '{survey_name}' has only {len(purpose_df)} trips for "
                                f"purpose '{purpose}' (min: {self.min_samples_per_purpose})")
                    continue

                # Compute mode shares for this survey
                if 'mode_type' not in purpose_df.columns:
                    logger.warning(f"  Survey '{survey_name}' missing 'mode_type' column")
                    continue

                mode_counts = purpose_df['mode_type'].value_counts(normalize=True)

                # Blend into overall rates
                for mode_str, share in mode_counts.items():
                    try:
                        mode_type = ModeType.from_survey_mode(mode_str)
                        if mode_type not in blended_rates:
                            blended_rates[mode_type] = 0.0
                        blended_rates[mode_type] += weight * share
                    except (ValueError, KeyError) as e:
                        logger.debug(f"  Unknown mode '{mode_str}' in survey: {e}")

            if blended_rates:
                # Renormalize (weights may not sum to 1 if some surveys skipped)
                total = sum(blended_rates.values())
                if total > 0:
                    blended_rates = {k: v / total for k, v in blended_rates.items()}

                self.survey_rates[purpose] = blended_rates
                logger.debug(f"  Mode rates for '{purpose}': {self._format_rates(blended_rates)}")
            else:
                logger.debug(f"  No valid data for purpose '{purpose}'")

        # Log summary
        logger.info(f"Computed mode rates for {len(self.survey_rates)} purposes")
        if 'all' in self.survey_rates:
            logger.info(f"  Overall mode distribution: {self._format_rates(self.survey_rates['all'])}")

    def _format_rates(self, rates: Dict[ModeType, float]) -> str:
        """Format rates dict for logging."""
        return ', '.join(f"{k.value}={v:.1%}" for k, v in sorted(rates.items(), key=lambda x: -x[1]))

    def _get_base_rates(self, purpose: Optional[str] = None) -> Dict[ModeType, float]:
        """
        Get base mode rates for a purpose, with fallback to 'all'.

        Args:
            purpose: Trip purpose (e.g., 'Work', 'Shopping')

        Returns:
            Dict mapping ModeType to share (0.0-1.0)
        """
        if not self.survey_rates:
            # No survey data - return 100% for fallback mode
            logger.debug("No survey rates available, using fallback mode")
            return {self.fallback_mode: 1.0}

        # Try purpose-specific rates first
        if purpose and purpose in self.survey_rates:
            return self.survey_rates[purpose].copy()

        # Fall back to overall rates
        if 'all' in self.survey_rates:
            if purpose:
                self.stats['purposes_with_fallback'].add(purpose)
                logger.debug(f"No rates for purpose '{purpose}', using 'all' rates")
            return self.survey_rates['all'].copy()

        # Last resort - fallback mode
        logger.warning("No survey rates found, using fallback mode")
        self.stats['fallback_used'] += 1
        return {self.fallback_mode: 1.0}

    def _apply_rate_adjustments(self, base_rates: Dict[ModeType, float]) -> Dict[ModeType, float]:
        """
        Apply config rate blending and share adjustments to base rates.

        For each mode in config:
        1. If config_rate is set, blend: (1-blend_weight)*survey + blend_weight*config
        2. Apply share_adjustment (additive)

        Args:
            base_rates: Base survey rates {ModeType: share}

        Returns:
            Adjusted rates (not yet renormalized)
        """
        adjusted_rates = base_rates.copy()

        for mode_type, mode_config in self.mode_configs.items():
            # Get current rate (0 if mode not in survey)
            current_rate = adjusted_rates.get(mode_type, 0.0)

            # Step 1: Blend with config_rate if specified
            if mode_config.config_rate is not None:
                survey_rate = current_rate
                if mode_config.survey_rate != 'auto':
                    # Use fixed survey_rate from config instead of computed
                    survey_rate = float(mode_config.survey_rate)

                blended = ((1 - mode_config.blend_weight) * survey_rate +
                          mode_config.blend_weight * mode_config.config_rate)
                current_rate = blended
                # !!! Printed many times !!!
                # logger.debug(f"  {mode_type.value}: blended rate = {blended:.3f} "
                #            f"(survey={survey_rate:.3f}, config={mode_config.config_rate:.3f}, "
                #            f"weight={mode_config.blend_weight:.2f})")

            # Step 2: Apply share_adjustment (additive)
            if mode_config.share_adjustment != 0.0:
                old_rate = current_rate
                current_rate = max(0.0, current_rate + mode_config.share_adjustment)
                logger.debug(f"  {mode_type.value}: adjusted {old_rate:.3f} -> {current_rate:.3f} "
                           f"(adjustment={mode_config.share_adjustment:+.3f})")

            adjusted_rates[mode_type] = current_rate

        return adjusted_rates

    def _filter_and_renormalize(self, rates: Dict[ModeType, float],
                                 available_modes: Set[ModeType]) -> Dict[ModeType, float]:
        """
        Filter rates to available modes and renormalize to sum to 1.0.

        Args:
            rates: Mode rates (may not sum to 1.0)
            available_modes: Set of available ModeTypes for this OD pair

        Returns:
            Filtered and normalized rates
        """
        # Filter to available modes
        filtered = {k: v for k, v in rates.items() if k in available_modes and v > 0}

        if not filtered:
            # No available modes with positive rates - use fallback
            if self.fallback_mode in available_modes:
                logger.debug("No modes with positive rates available, using fallback")
                return {self.fallback_mode: 1.0}
            else:
                # Fallback not available either - use first available
                first_available = next(iter(available_modes), self.fallback_mode)
                logger.warning(f"Fallback mode not available, using {first_available.value}")
                return {first_available: 1.0}

        # Renormalize
        total = sum(filtered.values())
        if total > 0:
            filtered = {k: v / total for k, v in filtered.items()}

        return filtered

    def _sample_mode(self, rates: Dict[ModeType, float],
                     rng: Optional[np.random.Generator] = None) -> ModeType:
        """
        Sample a mode from the rate distribution.

        Args:
            rates: Normalized mode rates (should sum to 1.0)
            rng: Random number generator

        Returns:
            Sampled ModeType
        """
        if not rates:
            return self.fallback_mode

        if rng is None:
            rng = np.random.default_rng()

        modes = list(rates.keys())
        probs = list(rates.values())

        # Sample
        idx = rng.choice(len(modes), p=probs)
        selected = modes[idx]

        # Update stats
        self.stats['mode_samples'][selected] += 1

        return selected

    def choose_mode(self, origin: Location, destination: Location,
                    purpose: Optional[str] = None,
                    rng: Optional[np.random.Generator] = None) -> ModeType:
        """
        Select mode for a single trip.

        Process:
        1. Get available modes for this OD pair
        2. Get base survey rates for the purpose
        3. Apply config rate blending and share adjustments
        4. Filter to available modes and renormalize
        5. Sample mode from distribution

        Args:
            origin: Trip origin location
            destination: Trip destination location
            purpose: Optional trip purpose (e.g., 'Work', 'Shopping')
            rng: Random number generator for reproducibility

        Returns:
            Selected ModeType
        """
        # Step 1: Get available modes
        available_modes = self.availability_manager.get_available_modes(origin, destination)
        if not available_modes:
            logger.warning("No modes available for OD pair, using fallback")
            self.stats['fallback_used'] += 1
            return self.fallback_mode

        logger.debug(f"Available modes for trip: {[m.value for m in available_modes]}")

        # Step 2: Get base survey rates
        base_rates = self._get_base_rates(purpose)

        # Step 3: Apply adjustments
        adjusted_rates = self._apply_rate_adjustments(base_rates)

        # Step 4: Filter and renormalize
        final_rates = self._filter_and_renormalize(adjusted_rates, available_modes)

        logger.debug(f"Final rates for purpose '{purpose}': {self._format_rates(final_rates)}")

        # Step 5: Sample
        return self._sample_mode(final_rates, rng)

    def choose_modes_for_chain(self, activities: List[Any],
                                locations: List[Location],
                                rng: Optional[np.random.Generator] = None) -> List[ModeType]:
        """
        Choose modes for an activity chain.

        For n activities, there are n-1 legs (trips between activities).

        With chain_consistency=True, the entire chain uses the same mode.
        If the chosen mode isn't available for all legs, we retry with a
        different mode (up to max_chain_mode_retries times).

        Args:
            activities: List of Activity objects (must have 'type' attribute)
            locations: List of Location objects (one per activity)
            rng: Random number generator for reproducibility

        Returns:
            List of ModeType for each leg (length = len(activities) - 1)
        """
        if len(activities) < 2:
            return []

        num_legs = len(activities) - 1

        if rng is None:
            rng = np.random.default_rng()

        if not self.chain_consistency:
            # No consistency - choose mode independently for each leg
            modes = []
            for i in range(num_legs):
                origin = locations[i]
                destination = locations[i + 1]
                purpose = getattr(activities[i + 1], 'type', None)
                mode = self.choose_mode(origin, destination, purpose, rng)
                modes.append(mode)
            return modes

        # Chain consistency: entire chain uses same mode
        # Get available modes for ALL legs
        available_for_all: Optional[Set[ModeType]] = None

        for i in range(num_legs):
            origin = locations[i]
            destination = locations[i + 1]
            leg_available = self.availability_manager.get_available_modes(origin, destination)

            if available_for_all is None:
                available_for_all = leg_available.copy()
            else:
                available_for_all &= leg_available  # Intersection

        if not available_for_all:
            logger.warning("No mode available for all legs in chain, using fallback for each leg")
            self.stats['fallback_used'] += 1
            return [self.fallback_mode] * num_legs

        # Track availability combinations for summary
        avail_key = frozenset(m.value for m in available_for_all)
        self.stats['chain_availability'][avail_key] = self.stats['chain_availability'].get(avail_key, 0) + 1

        # Get the dominant purpose for mode choice (use first non-Home activity)
        dominant_purpose = None
        for act in activities[1:]:  # Skip first (usually Home)
            act_type = getattr(act, 'type', None)
            if act_type and act_type != 'Home':
                dominant_purpose = act_type
                break

        # Get rates and filter to modes available for all legs
        base_rates = self._get_base_rates(dominant_purpose)
        adjusted_rates = self._apply_rate_adjustments(base_rates)
        final_rates = self._filter_and_renormalize(adjusted_rates, available_for_all)

        # Sample mode for the entire chain
        chain_mode = self._sample_mode(final_rates, rng)

        # Track chain mode selections for summary
        sel_key = (chain_mode.value, dominant_purpose, num_legs)
        self.stats['chain_mode_selections'][sel_key] = self.stats['chain_mode_selections'].get(sel_key, 0) + 1

        return [chain_mode] * num_legs

    def get_output_mode(self, mode_type: ModeType, output_format: str = 'matsim') -> str:
        """
        Get the output format mode string for a ModeType.

        Args:
            mode_type: The mode type
            output_format: Target format ('matsim', etc.)

        Returns:
            Mode string for output (e.g., 'car', 'pt')
        """
        mode_config = self.mode_configs.get(mode_type)
        if mode_config and output_format == 'matsim':
            return mode_config.matsim_mode
        return mode_type.to_output_mode(output_format)

    def create_legs(self, activities: List[Any],
                    locations: List[Location],
                    rng: Optional[np.random.Generator] = None,
                    output_format: str = 'matsim') -> List[Leg]:
        """
        Create Leg objects for an activity chain.

        Convenience method that combines mode choice with Leg creation.

        Args:
            activities: List of Activity objects
            locations: List of Location objects
            rng: Random number generator
            output_format: Target output format

        Returns:
            List of Leg objects with mode strings
        """
        mode_types = self.choose_modes_for_chain(activities, locations, rng)
        return [Leg(mode=self.get_output_mode(mt, output_format)) for mt in mode_types]

    def get_stats_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for logging.

        Returns:
            Dict with mode choice statistics
        """
        total_samples = sum(self.stats['mode_samples'].values())
        mode_distribution = {}
        if total_samples > 0:
            mode_distribution = {
                k.value: v / total_samples
                for k, v in self.stats['mode_samples'].items()
                if v > 0
            }

        # Aggregate chain selections by mode and by purpose
        chain_by_mode = {}
        chain_by_purpose = {}
        for (mode, purpose, num_legs), count in self.stats['chain_mode_selections'].items():
            chain_by_mode[mode] = chain_by_mode.get(mode, 0) + count
            purpose_key = purpose or 'Unknown'
            chain_by_purpose[purpose_key] = chain_by_purpose.get(purpose_key, 0) + count

        # Aggregate availability combinations
        avail_summary = {}
        for modes_set, count in self.stats['chain_availability'].items():
            key = ', '.join(sorted(modes_set))
            avail_summary[key] = avail_summary.get(key, 0) + count

        return {
            'total_mode_choices': total_samples,
            'mode_distribution': mode_distribution,
            'fallback_used': self.stats['fallback_used'],
            'chain_retries': self.stats['chain_retries'],
            'purposes_using_fallback_rates': list(self.stats['purposes_with_fallback']),
            'chain_selections_by_mode': chain_by_mode,
            'chain_selections_by_purpose': chain_by_purpose,
            'chain_availability_combos': avail_summary,
        }

    def log_stats_summary(self) -> None:
        """Log summary statistics."""
        stats = self.get_stats_summary()
        logger.info("=" * 50)
        logger.info("MODE CHOICE STATISTICS")
        logger.info("=" * 50)
        logger.info(f"  Total mode choices: {stats['total_mode_choices']}")
        logger.info(f"  Fallback mode used: {stats['fallback_used']} times")
        logger.info(f"  Chain retries: {stats['chain_retries']}")

        if stats['mode_distribution']:
            logger.info("  Mode distribution in generated plans:")
            for mode, share in sorted(stats['mode_distribution'].items(), key=lambda x: -x[1]):
                logger.info(f"    {mode}: {share:.1%}")

        if stats['purposes_using_fallback_rates']:
            logger.info(f"  Purposes using 'all' rates: {stats['purposes_using_fallback_rates']}")

        if stats.get('chain_selections_by_mode'):
            logger.info("  Chain mode selections:")
            for mode, count in sorted(stats['chain_selections_by_mode'].items(), key=lambda x: -x[1]):
                logger.info(f"    {mode}: {count}")

        if stats.get('chain_selections_by_purpose'):
            logger.info("  Chain selections by purpose:")
            for purpose, count in sorted(stats['chain_selections_by_purpose'].items(), key=lambda x: -x[1]):
                logger.info(f"    {purpose}: {count}")

        if stats.get('chain_availability_combos'):
            logger.info("  Mode availability combinations across chains:")
            for combo, count in sorted(stats['chain_availability_combos'].items(), key=lambda x: -x[1]):
                logger.info(f"    [{combo}]: {count} chains")

        logger.info("=" * 50)
